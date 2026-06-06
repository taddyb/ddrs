//! Content-addressed cache for the managed adjacency build.
//!
//! Layout under `<workspace_root>/adjacency/<key>/`:
//!
//! ```text
//! merit_conus_adjacency.zarr/      — written by zarr_write::write_conus_store
//! merit_gages_conus_adjacency.zarr/ — written by zarr_write::write_gauges_store
//! manifest.json                    — input paths + fingerprints, graph dims,
//!                                    dropped COMIDs, build duration, git SHA
//! ```
//!
//! The key is blake3 of the .dbf file bytes ∥ gages CSV file bytes ∥
//! BUILDER_VERSION, truncated to 16 hex chars. Content fingerprints (not
//! stat/path) are used so that moving or renaming the files does NOT
//! invalidate, and so two files with identical bytes share a cache entry.
//!
//! Build is crash-safe: everything is written into a temp dir
//! `<root>/adjacency/.tmp-<key>` then atomically renamed into place.

use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::adjacency::build::{build_conus_adjacency, BuildError};
use crate::adjacency::dbf::{read_flowpath_records, resolve_dbf_path};
use crate::adjacency::gauges::build_gauge_subgraphs;
use crate::adjacency::zarr_write::{write_conus_store, write_gauges_store};
use crate::adjacency::BUILDER_VERSION;
use crate::data::error::DataError;

/// Paths to the two zarr stores produced by (or read from) the cache.
#[derive(Debug, Clone)]
pub struct AdjacencyCachePaths {
    pub conus: PathBuf,
    pub gages: PathBuf,
}

/// Result returned by [`resolve_or_build`].
#[derive(Debug)]
pub struct AdjacencyCacheOutcome {
    pub paths: AdjacencyCachePaths,
    /// 16-hex-char content-addressed key.
    pub key: String,
    /// `true` when the cache directory already existed and was reused.
    pub cache_hit: bool,
}

/// Errors that can arise when resolving or building the adjacency cache.
#[derive(Debug, thiserror::Error)]
pub enum AdjacencyCacheError {
    #[error("adjacency data error: {0}")]
    Data(#[from] DataError),
    #[error("adjacency build error: {0}")]
    Build(#[from] BuildError),
    #[error("adjacency cache I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("adjacency cache manifest serialization: {0}")]
    Json(#[from] serde_json::Error),
}

// ── manifest schema ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    key: String,
    builder_version: u32,
    /// Absolute path of the .dbf (or .shp) input supplied by the caller.
    fabric_path: PathBuf,
    /// Absolute path of the resolved .dbf file that was hashed.
    /// May differ from `fabric_path` when the caller supplies a .shp sibling.
    dbf_path: PathBuf,
    /// blake3 hex of the resolved .dbf file bytes.
    fabric_fingerprint: String,
    /// Absolute path of the gages CSV.
    gages_path: PathBuf,
    /// blake3 hex of the gages CSV file bytes.
    gages_fingerprint: String,
    /// Number of CONUS reaches in `order`.
    n: usize,
    /// Number of COO edges (nnz).
    nnz: usize,
    /// Number of gauge subgraphs built.
    n_gauges: usize,
    /// COMIDs dropped because they were on a simple cycle.
    dropped_comids: Vec<i64>,
    /// Wall-clock build duration in seconds.
    build_duration_secs: f64,
    /// `git rev-parse HEAD` at build time, empty string if not available.
    ddrs_git_sha: String,
}

// ── public API ───────────────────────────────────────────────────────────────

/// Resolve the adjacency cache for the given inputs, building if necessary.
///
/// `fabric` may be a `.shp` or `.dbf` path — the `.dbf` sibling is opened for
/// hashing and reading (same resolution as [`read_flowpath_records`]).
/// `workspace_root` is the ddrs workspace root (`.ddrs/..` parent); the cache
/// lands at `<workspace_root>/adjacency/<key>/`.
///
/// On a cache miss the build takes ~1–2 min on real MERIT data (108 MB dbf +
/// ~3 000 BFS traversals + zarr writes). A progress line is printed before
/// the build begins so the user knows it is not hung.
pub fn resolve_or_build(
    workspace_root: &Path,
    fabric: &Path,
    gages_csv: &Path,
) -> Result<AdjacencyCacheOutcome, AdjacencyCacheError> {
    // Resolve the actual .dbf path (mirrors the dbf reader's resolution).
    let dbf_path = resolve_dbf_path(fabric);

    // --- 1. Compute the content key -----------------------------------------
    let dbf_fp = file_fingerprint(&dbf_path)?;
    let gages_fp = file_fingerprint(gages_csv)?;
    let key = content_key(&dbf_fp, &gages_fp);

    // --- 2. Cache hit? -------------------------------------------------------
    let cache_dir = adjacency_cache_dir(workspace_root, &key);
    if is_cache_hit(&cache_dir) {
        return Ok(AdjacencyCacheOutcome {
            paths: store_paths(&cache_dir),
            key,
            cache_hit: true,
        });
    }

    // --- 3. Cache miss: build into a temp dir then rename -------------------
    println!(
        "  building MERIT adjacency from {} — first run takes ~1-2 min",
        dbf_path.display()
    );

    let tmp_dir = workspace_root
        .join("adjacency")
        .join(format!(".tmp-{}", key));
    // Clean up any leftover temp dir from a previous crashed build.
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir)?;

    let t0 = Instant::now();

    let records = read_flowpath_records(&dbf_path)
        .map_err(AdjacencyCacheError::Data)?;

    let conus = build_conus_adjacency(&records)
        .map_err(AdjacencyCacheError::Build)?;

    let n = conus.order.len();
    let nnz = conus.rows.len();
    let dropped_comids = conus.dropped_comids.clone();

    let conus_dest = tmp_dir.join("merit_conus_adjacency.zarr");
    write_conus_store(&conus, &conus_dest)
        .map_err(AdjacencyCacheError::Data)?;

    let subgraphs = build_gauge_subgraphs(&conus, gages_csv)
        .map_err(AdjacencyCacheError::Data)?;
    let n_gauges = subgraphs.len();

    let gages_dest = tmp_dir.join("merit_gages_conus_adjacency.zarr");
    write_gauges_store(&subgraphs, n, &gages_dest)
        .map_err(AdjacencyCacheError::Data)?;

    let build_duration_secs = t0.elapsed().as_secs_f64();

    // Write manifest.
    let manifest = CacheManifest {
        key: key.clone(),
        builder_version: BUILDER_VERSION,
        fabric_path: fabric.to_path_buf(),
        dbf_path: dbf_path.clone(),
        fabric_fingerprint: dbf_fp,
        gages_path: gages_csv.to_path_buf(),
        gages_fingerprint: gages_fp,
        n,
        nnz,
        n_gauges,
        dropped_comids,
        build_duration_secs,
        ddrs_git_sha: capture_git_sha(),
    };
    let json = serde_json::to_string_pretty(&manifest)?;
    fs::write(tmp_dir.join("manifest.json"), json)?;

    // Atomic rename into place.  If the target already exists (lost race
    // between two concurrent builds) treat it as a hit and discard the tmp.
    match fs::rename(&tmp_dir, &cache_dir) {
        Ok(()) => {}
        Err(_) if is_cache_hit(&cache_dir) => {
            // Another process won the race; discard our tmp.
            let _ = fs::remove_dir_all(&tmp_dir);
        }
        Err(e) => return Err(AdjacencyCacheError::Io(e)),
    }

    println!(
        "  adjacency cache written to {} ({:.1}s)",
        cache_dir.display(),
        build_duration_secs
    );

    Ok(AdjacencyCacheOutcome {
        paths: store_paths(&cache_dir),
        key,
        cache_hit: false,
    })
}

/// Resolve `<workspace_root>/adjacency/<key>/` for a given key.
pub fn adjacency_cache_dir(workspace_root: &Path, key: &str) -> PathBuf {
    workspace_root.join("adjacency").join(key)
}

// ── internal helpers ─────────────────────────────────────────────────────────

/// The two zarr store paths inside a cache directory.
fn store_paths(cache_dir: &Path) -> AdjacencyCachePaths {
    AdjacencyCachePaths {
        conus: cache_dir.join("merit_conus_adjacency.zarr"),
        gages: cache_dir.join("merit_gages_conus_adjacency.zarr"),
    }
}

/// A cache hit requires the directory to exist with a manifest.json present.
/// Mirrors baseline/cache.rs's hit criterion (manifest presence = valid cache).
fn is_cache_hit(dir: &Path) -> bool {
    dir.join("manifest.json").exists()
}

/// 16-hex-char (64-bit prefix) content key: blake3(dbf_fp ∥ gages_fp ∥ version).
///
/// Inputs are the full hex fingerprints of the file bytes, not the paths.
/// Collision-free at our scale; safe for filesystem use.
fn content_key(dbf_fp: &str, gages_fp: &str) -> String {
    let mut h = blake3::Hasher::new();
    h.update(dbf_fp.as_bytes());
    h.update(b"\n");
    h.update(gages_fp.as_bytes());
    h.update(b"\n");
    h.update(BUILDER_VERSION.to_le_bytes().as_ref());
    let hex = h.finalize().to_hex();
    hex.as_str()[..16].to_string()
}

/// blake3 hex over the full contents of a file, using a buffered reader.
fn file_fingerprint(path: &Path) -> Result<String, AdjacencyCacheError> {
    let file = fs::File::open(path).map_err(AdjacencyCacheError::Io)?;
    let mut reader = BufReader::new(file);
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = reader.read(&mut buf).map_err(AdjacencyCacheError::Io)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Capture `git rev-parse HEAD` at runtime, matching `capture_git()` in
/// `src/cli/run.rs`. Returns an empty string if git is unavailable.
fn capture_git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dbase::{FieldName, TableWriterBuilder};
    use std::convert::TryFrom;

    // ── synthetic helpers ────────────────────────────────────────────────────

    /// Row data for `write_dbf`: (COMID, lengthkm, slope, NextDownID, up1).
    ///
    /// up2/up3/up4 are always 0.0; this covers the two-reach topology used by
    /// all test fixtures.
    type DbfRow = (f64, f64, f64, f64, f64);

    /// Write a synthetic .dbf at `path` with the given rows.
    ///
    /// Schema: COMID, lengthkm, slope, NextDownID, up1, up2, up3, up4.
    /// up2/up3/up4 are always 0.0.
    fn write_dbf(path: &Path, rows: &[DbfRow]) {
        let builder = TableWriterBuilder::new()
            .add_numeric_field(FieldName::try_from("COMID").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("lengthkm").unwrap(), 12, 6)
            .add_numeric_field(FieldName::try_from("slope").unwrap(), 12, 6)
            .add_numeric_field(FieldName::try_from("NextDownID").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up1").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up2").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up3").unwrap(), 10, 0)
            .add_numeric_field(FieldName::try_from("up4").unwrap(), 10, 0);

        let mut writer = builder.build_with_file_dest(path).expect("create dbf writer");
        for &(comid, lengthkm, slope, next_down, up1) in rows {
            let mut rec = dbase::Record::default();
            rec.insert("COMID".to_owned(), dbase::FieldValue::Numeric(Some(comid)));
            rec.insert("lengthkm".to_owned(), dbase::FieldValue::Numeric(Some(lengthkm)));
            rec.insert("slope".to_owned(), dbase::FieldValue::Numeric(Some(slope)));
            rec.insert("NextDownID".to_owned(), dbase::FieldValue::Numeric(Some(next_down)));
            rec.insert("up1".to_owned(), dbase::FieldValue::Numeric(Some(up1)));
            rec.insert("up2".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
            rec.insert("up3".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
            rec.insert("up4".to_owned(), dbase::FieldValue::Numeric(Some(0.0)));
            writer.write_record(&rec).expect("write record");
        }
        writer.finalize().expect("close dbf writer");
    }

    /// Default two-reach topology: reach 1 (headwater) → reach 2 (outlet).
    fn write_tiny_dbf(path: &Path) {
        write_dbf(
            path,
            &[
                (1.0, 1.0, 0.001, 2.0, 0.0), // reach 1: feeds reach 2
                (2.0, 2.0, 0.002, 0.0, 1.0), // reach 2: outlet
            ],
        );
    }

    /// Write a tiny gages CSV with one gauge at COMID 2 (the outlet).
    fn write_tiny_gages(path: &Path) {
        let csv = "STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID\n\
                   00000002,OUTLET,10.0,1.0,2.0,2\n";
        fs::write(path, csv).expect("write gages csv");
    }

    fn tmp_workspace(tag: &str) -> PathBuf {
        let p = std::env::temp_dir()
            .join(format!("ddrs_adj_cache_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).expect("mkdir workspace");
        p
    }

    // ── key tests ────────────────────────────────────────────────────────────

    #[test]
    fn content_key_is_16_hex_chars() {
        let k = content_key("aabbcc", "ddeeff");
        assert_eq!(k.len(), 16, "key must be 16 hex chars, got: {k}");
        assert!(k.chars().all(|c| c.is_ascii_hexdigit()), "key must be hex: {k}");
    }

    #[test]
    fn content_key_is_stable() {
        let k1 = content_key("fp1", "fp2");
        let k2 = content_key("fp1", "fp2");
        assert_eq!(k1, k2);
    }

    #[test]
    fn content_key_differs_on_different_inputs() {
        let k1 = content_key("aaaa", "bbbb");
        let k2 = content_key("aaaa", "cccc");
        assert_ne!(k1, k2);
    }

    // ── end-to-end cache tests ───────────────────────────────────────────────

    #[test]
    fn cache_miss_builds_and_second_call_is_hit() {
        let ws = tmp_workspace("e2e");
        let dbf_path = ws.join("fabric.dbf");
        let gages_path = ws.join("gages.csv");
        write_tiny_dbf(&dbf_path);
        write_tiny_gages(&gages_path);

        // First call: cache miss → build.
        let out1 = resolve_or_build(&ws, &dbf_path, &gages_path)
            .expect("first build");
        assert!(!out1.cache_hit, "first call should be a cache miss");
        assert_eq!(out1.key.len(), 16);

        // zarr directories must exist and be readable.
        assert!(out1.paths.conus.exists(), "conus zarr must exist");
        assert!(out1.paths.gages.exists(), "gages zarr must exist");

        // manifest.json must be present with sensible fields.
        let manifest_path = adjacency_cache_dir(&ws, &out1.key).join("manifest.json");
        assert!(manifest_path.exists(), "manifest.json must exist");
        let raw = fs::read_to_string(&manifest_path).expect("read manifest");
        let m: CacheManifest = serde_json::from_str(&raw).expect("parse manifest");
        assert_eq!(m.key, out1.key);
        assert_eq!(m.n, 2, "two reaches in the tiny network");
        assert_eq!(m.nnz, 1, "one edge in the tiny network");
        assert_eq!(m.n_gauges, 1, "one gauge");
        assert!(m.dropped_comids.is_empty(), "no cycles in tiny network");
        assert!(m.build_duration_secs >= 0.0);
        assert_eq!(m.builder_version, BUILDER_VERSION);
        // dbf_path and fabric_path must agree (both are .dbf here).
        assert_eq!(m.fabric_path, dbf_path);
        assert_eq!(m.dbf_path, dbf_path);

        // Second call: cache hit.
        let out2 = resolve_or_build(&ws, &dbf_path, &gages_path)
            .expect("second call");
        assert!(out2.cache_hit, "second call should be a cache hit");
        assert_eq!(out2.key, out1.key, "same key on hit");

        let _ = fs::remove_dir_all(&ws);
    }

    /// Two dbfs with different content must produce different cache keys AND
    /// each build must land in its own cache directory with a manifest.json.
    #[test]
    fn key_changes_when_dbf_content_changes() {
        let ws = tmp_workspace("key_change");
        let dbf_v1 = ws.join("v1.dbf");
        let dbf_v2 = ws.join("v2.dbf");
        let gages_path = ws.join("gages.csv");

        // v1: standard two-reach topology.
        write_tiny_dbf(&dbf_v1);
        // v2: same topology but lengthkm of reach 1 changed → different bytes.
        write_dbf(
            &dbf_v2,
            &[
                (1.0, 99.0, 0.001, 2.0, 0.0), // lengthkm changed to 99
                (2.0, 2.0, 0.002, 0.0, 1.0),
            ],
        );
        write_tiny_gages(&gages_path);

        let out1 = resolve_or_build(&ws, &dbf_v1, &gages_path).expect("build v1");
        let out2 = resolve_or_build(&ws, &dbf_v2, &gages_path).expect("build v2");

        assert_ne!(out1.key, out2.key, "keys must differ for different dbf content");
        assert!(!out1.cache_hit);
        assert!(!out2.cache_hit);

        // Both cache directories must exist and contain a manifest.
        let dir1 = adjacency_cache_dir(&ws, &out1.key);
        let dir2 = adjacency_cache_dir(&ws, &out2.key);
        assert!(dir1.join("manifest.json").exists(), "v1 manifest must exist");
        assert!(dir2.join("manifest.json").exists(), "v2 manifest must exist");
        assert_ne!(dir1, dir2, "cache dirs must be distinct");

        let _ = fs::remove_dir_all(&ws);
    }

    #[test]
    fn content_key_fn_mixes_in_builder_version() {
        // Verify that the BUILDER_VERSION is mixed into the key by comparing
        // content_key output against a manual hash that includes the version.
        let dbf_fp = "deadbeef";
        let gages_fp = "cafebabe";
        let k = content_key(dbf_fp, gages_fp);

        let mut h = blake3::Hasher::new();
        h.update(dbf_fp.as_bytes());
        h.update(b"\n");
        h.update(gages_fp.as_bytes());
        h.update(b"\n");
        h.update(BUILDER_VERSION.to_le_bytes().as_ref());
        let expected = &h.finalize().to_hex().as_str()[..16].to_string();
        assert_eq!(&k, expected, "content_key must include BUILDER_VERSION");
    }

    #[test]
    fn zarr_stores_readable_after_build() {
        use crate::data::ids::{Comid, Staid};
        use crate::data::store::zarr::{ConusAdjacencyStore, GagesAdjacencyStore};

        let ws = tmp_workspace("readable");
        let dbf_path = ws.join("fabric.dbf");
        let gages_path = ws.join("gages.csv");
        write_tiny_dbf(&dbf_path);
        write_tiny_gages(&gages_path);

        let out = resolve_or_build(&ws, &dbf_path, &gages_path).expect("build");
        assert!(!out.cache_hit);

        // Verify the conus store round-trips.
        let conus = ConusAdjacencyStore::open(&out.paths.conus).expect("open conus");
        assert_eq!(conus.n, 2);
        assert_eq!(conus.nnz, 1);
        // Topological order has reach 1 first (headwater), reach 2 last (outlet).
        assert!(conus.order.contains(&Comid(1)));
        assert!(conus.order.contains(&Comid(2)));

        // Verify the gages store round-trips.
        let staids = vec![Staid::new("2")];
        let gages = GagesAdjacencyStore::open(&out.paths.gages, &staids).expect("open gages");
        assert_eq!(gages.len(), 1);
        let g = gages.get(&Staid::new("2")).expect("gauge 2");
        assert_eq!(g.gage_catchment, "2");
        assert_eq!(g.gage_idx, 1); // reach 2 is at position 1 (outlet, topo-last)

        let _ = fs::remove_dir_all(&ws);
    }
}
