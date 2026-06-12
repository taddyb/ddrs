//! Content-addressed cache for the managed adjacency build.
//!
//! Layout under `<workspace_root>/adjacency/<key>/`:
//!
//! ```text
//! <fabric>_adjacency.zarr/       — written by zarr_write::write_conus_store
//! <fabric>_gages_adjacency.zarr/ — written by zarr_write::write_gauges_store
//! manifest.json                  — input paths + fingerprints, graph dims,
//!                                  dropped COMIDs, build duration, git SHA
//! ```
//!
//! `<fabric>` is the geospatial-fabric file stem (e.g. `global_merit_riv`
//! for the global gpkg, `riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1`
//! for the CONUS shapefile), so the store names say what network they
//! hold. Caches written before this scheme used fixed
//! `merit_conus_adjacency.zarr` / `merit_gages_conus_adjacency.zarr`
//! names regardless of scope; `store_paths` falls back to whatever
//! `*_adjacency.zarr` pair is present so old caches keep hitting.
//!
//! The key is blake3 of the resolved fabric file bytes (.dbf or .gpkg) ∥
//! gages CSV file bytes ∥ optional gpkg layer name ∥ BUILDER_VERSION,
//! truncated to 16 hex chars. Content fingerprints (not stat/path) are used
//! so that moving or renaming the files does NOT invalidate, and so two
//! files with identical bytes share a cache entry.
//!
//! Build is crash-safe: everything is written into a temp dir
//! `<root>/adjacency/.tmp-<key>` then atomically renamed into place.

use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::adjacency::build::{build_conus_adjacency, BuildError};
use crate::adjacency::fabric::{read_fabric_records, resolve_fabric};
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
    /// Absolute path of the fabric input supplied by the caller
    /// (.shp, .dbf, or .gpkg).
    fabric_path: PathBuf,
    /// Absolute path of the resolved file that was hashed: the sibling .dbf
    /// for a .shp input, the file itself for .dbf/.gpkg.
    /// (Serde alias keeps manifests written before gpkg support readable.)
    #[serde(alias = "dbf_path")]
    fabric_resolved_path: PathBuf,
    /// gpkg feature layer used, when the fabric is a .gpkg with an explicit
    /// `geospatial_fabric_layer`. `None` for dBASE fabrics / single-layer gpkg.
    #[serde(default)]
    fabric_layer: Option<String>,
    /// blake3 hex of the resolved fabric file bytes.
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
/// `fabric` may be a `.shp` (sibling `.dbf` opened), `.dbf`, or `.gpkg` path —
/// see `fabric::resolve_fabric`. `fabric_layer` selects the gpkg feature layer
/// (ignored for dBASE inputs; config validation rejects that combination).
/// `workspace_root` is the ddrs workspace root (`.ddrs/..` parent); the cache
/// lands at `<workspace_root>/adjacency/<key>/`.
///
/// On a cache miss the build takes ~10 s on real CONUS MERIT data (108 MB dbf
/// + ~3 000 BFS traversals + zarr writes; measured ~2 s on an NVMe host). A
/// progress line is printed before the build begins so the user knows it is
/// not hung. For a multi-GB global gpkg the content fingerprint itself costs
/// a few seconds of hashing on first run.
pub fn resolve_or_build(
    workspace_root: &Path,
    fabric: &Path,
    fabric_layer: Option<&str>,
    gages_csv: &Path,
) -> Result<AdjacencyCacheOutcome, AdjacencyCacheError> {
    // Resolve the hashable artifact (.shp → sibling .dbf; .dbf/.gpkg as-is).
    let resolved = resolve_fabric(fabric).map_err(AdjacencyCacheError::Data)?;
    let resolved_path = resolved.resolved_path().to_path_buf();

    // Store names carry the fabric stem (`global_merit_riv_adjacency.zarr`)
    // so a cache directory says what network it holds.
    let fabric_stem = fabric
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "merit".into());

    // --- 1. Compute the content key -----------------------------------------
    let fabric_fp = file_fingerprint(&resolved_path)?;
    let gages_fp = gages_fingerprint(gages_csv)?;
    let key = content_key(&fabric_fp, &gages_fp, fabric_layer);

    // --- 2. Cache hit? -------------------------------------------------------
    let cache_dir = adjacency_cache_dir(workspace_root, &key);
    if is_cache_hit(&cache_dir) {
        let paths = store_paths(&cache_dir, &fabric_stem);
        // A manifest with missing stores (e.g. hand-deleted zarr dirs) is a
        // stale cache, not a hit — clear it and rebuild.
        if paths.conus.is_dir() && paths.gages.is_dir() {
            return Ok(AdjacencyCacheOutcome {
                paths,
                key,
                cache_hit: true,
            });
        }
        eprintln!(
            "  adjacency cache {} is incomplete (manifest without stores) — rebuilding",
            cache_dir.display()
        );
        fs::remove_dir_all(&cache_dir)?;
    }

    // --- 3. Cache miss: build into a temp dir then rename -------------------
    println!(
        "  building MERIT adjacency from {} — first run takes ~10 s (CONUS dbf) \
         to a few minutes (global gpkg)",
        resolved_path.display()
    );

    let tmp_dir = workspace_root
        .join("adjacency")
        .join(format!(".tmp-{}", key));
    // Clean up any leftover temp dir from a previous crashed build.
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir)?;

    let t0 = Instant::now();

    let records = read_fabric_records(fabric, fabric_layer)
        .map_err(AdjacencyCacheError::Data)?;

    let conus = build_conus_adjacency(&records)
        .map_err(AdjacencyCacheError::Build)?;

    let n = conus.order.len();
    let nnz = conus.rows.len();
    let dropped_comids = conus.dropped_comids.clone();

    let conus_dest = tmp_dir.join(format!("{fabric_stem}_adjacency.zarr"));
    write_conus_store(&conus, &conus_dest)
        .map_err(AdjacencyCacheError::Data)?;

    let subgraphs = build_gauge_subgraphs(&conus, gages_csv)
        .map_err(AdjacencyCacheError::Data)?;
    let n_gauges = subgraphs.len();

    let gages_dest = tmp_dir.join(format!("{fabric_stem}_gages_adjacency.zarr"));
    write_gauges_store(&subgraphs, n, &gages_dest)
        .map_err(AdjacencyCacheError::Data)?;

    let build_duration_secs = t0.elapsed().as_secs_f64();

    // Write manifest.
    let manifest = CacheManifest {
        key: key.clone(),
        builder_version: BUILDER_VERSION,
        fabric_path: fabric.to_path_buf(),
        fabric_resolved_path: resolved_path.clone(),
        fabric_layer: fabric_layer.map(str::to_string),
        fabric_fingerprint: fabric_fp,
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
        Err(_) if is_cache_hit(&cache_dir)
            && store_paths(&cache_dir, &fabric_stem).conus.is_dir() => {
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
        paths: store_paths(&cache_dir, &fabric_stem),
        key,
        cache_hit: false,
    })
}

/// Resolve `<workspace_root>/adjacency/<key>/` for a given key.
pub fn adjacency_cache_dir(workspace_root: &Path, key: &str) -> PathBuf {
    workspace_root.join("adjacency").join(key)
}

// ── internal helpers ─────────────────────────────────────────────────────────

/// The two zarr store paths inside a cache directory, named after the
/// fabric stem. Falls back to whatever `*_adjacency.zarr` pair already
/// exists (legacy fixed `merit_*conus*` names, or a different stem when the
/// fabric file was renamed — the content key is byte-addressed, so a rename
/// still hits the same cache entry).
fn store_paths(cache_dir: &Path, fabric_stem: &str) -> AdjacencyCachePaths {
    let network = cache_dir.join(format!("{fabric_stem}_adjacency.zarr"));
    let gages = cache_dir.join(format!("{fabric_stem}_gages_adjacency.zarr"));
    if network.is_dir() && gages.is_dir() {
        return AdjacencyCachePaths { conus: network, gages };
    }
    // Fallback scan: any existing pair, classified by the _gages_ marker.
    let mut found_network = None;
    let mut found_gages = None;
    if let Ok(entries) = fs::read_dir(cache_dir) {
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for p in paths {
            let Some(name) = p.file_name().map(|n| n.to_string_lossy().into_owned()) else {
                continue;
            };
            if !name.ends_with("adjacency.zarr") || !p.is_dir() {
                continue;
            }
            let is_gages = name.ends_with("_gages_adjacency.zarr")
                || name == "merit_gages_conus_adjacency.zarr";
            if is_gages {
                found_gages.get_or_insert(p);
            } else {
                found_network.get_or_insert(p);
            }
        }
    }
    match (found_network, found_gages) {
        (Some(c), Some(g)) => AdjacencyCachePaths { conus: c, gages: g },
        // Fresh build (or corrupt cache → miss): the stem names.
        _ => AdjacencyCachePaths { conus: network, gages },
    }
}

/// A cache hit requires the directory to exist with a manifest.json present.
/// Mirrors baseline/cache.rs's hit criterion (manifest presence = valid cache).
fn is_cache_hit(dir: &Path) -> bool {
    dir.join("manifest.json").exists()
}

/// 16-hex-char (64-bit prefix) content key:
/// blake3(fabric_fp ∥ gages_fp ∥ [layer ∥] version).
///
/// Inputs are the full hex fingerprints of the file bytes, not the paths.
/// Collision-free at our scale; safe for filesystem use.
///
/// `layer` participates in the key only when `Some`: a multi-layer gpkg has
/// identical bytes regardless of which layer is selected, so the layer name
/// must distinguish cache entries. Folding it in only when set keeps every
/// pre-gpkg cache key (dbf fabrics, layer always `None`) unchanged.
fn content_key(fabric_fp: &str, gages_fp: &str, layer: Option<&str>) -> String {
    let mut h = blake3::Hasher::new();
    h.update(fabric_fp.as_bytes());
    h.update(b"\n");
    h.update(gages_fp.as_bytes());
    h.update(b"\n");
    if let Some(layer) = layer {
        h.update(layer.as_bytes());
        h.update(b"\n");
    }
    h.update(BUILDER_VERSION.to_le_bytes().as_ref());
    let hex = h.finalize().to_hex();
    hex.as_str()[..16].to_string()
}

/// blake3 hex over the full contents of a file, using a buffered reader.
/// Content fingerprint of the gages input: a single CSV file, or a directory
/// of per-zone CSVs (e.g. `v3.1/8km/<zone>_all.csv` — see
/// `GageMetadata::open`). For a directory, hash every `*.csv` in sorted
/// filename order (name + NUL + bytes), so renaming, adding, or editing any
/// zone file changes the adjacency cache key.
fn gages_fingerprint(path: &Path) -> Result<String, AdjacencyCacheError> {
    let md = fs::metadata(path).map_err(AdjacencyCacheError::Io)?;
    if !md.is_dir() {
        return file_fingerprint(path);
    }
    let mut csvs: Vec<PathBuf> = fs::read_dir(path)
        .map_err(AdjacencyCacheError::Io)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "csv"))
        .collect();
    csvs.sort();
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 65536];
    for csv in &csvs {
        if let Some(name) = csv.file_name() {
            hasher.update(name.to_string_lossy().as_bytes());
            hasher.update(&[0]);
        }
        let file = fs::File::open(csv).map_err(AdjacencyCacheError::Io)?;
        let mut reader = BufReader::new(file);
        loop {
            let n = reader.read(&mut buf).map_err(AdjacencyCacheError::Io)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }
    Ok(hasher.finalize().to_hex().to_string())
}

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
        let k = content_key("aabbcc", "ddeeff", None);
        assert_eq!(k.len(), 16, "key must be 16 hex chars, got: {k}");
        assert!(k.chars().all(|c| c.is_ascii_hexdigit()), "key must be hex: {k}");
    }

    #[test]
    fn content_key_is_stable() {
        let k1 = content_key("fp1", "fp2", None);
        let k2 = content_key("fp1", "fp2", None);
        assert_eq!(k1, k2);
    }

    #[test]
    fn content_key_differs_on_different_inputs() {
        let k1 = content_key("aaaa", "bbbb", None);
        let k2 = content_key("aaaa", "cccc", None);
        assert_ne!(k1, k2);
    }

    #[test]
    fn content_key_differs_on_layer() {
        // Same fabric bytes, different gpkg layer → distinct cache entries.
        let base = content_key("aaaa", "bbbb", None);
        let l1 = content_key("aaaa", "bbbb", Some("flowlines"));
        let l2 = content_key("aaaa", "bbbb", Some("catchments"));
        assert_ne!(l1, l2);
        assert_ne!(base, l1);
    }

    // ── gages fingerprint over a directory ───────────────────────────────────

    #[test]
    fn gages_fingerprint_handles_directories() {
        let ws = tmp_workspace("gfp");
        // Single file: same as file_fingerprint.
        let single = ws.join("gages.csv");
        write_tiny_gages(&single);
        assert_eq!(
            gages_fingerprint(&single).unwrap(),
            file_fingerprint(&single).unwrap()
        );

        // Directory of zone CSVs: stable, sensitive to content and to
        // filename, blind to non-CSV files.
        let dir = ws.join("zones");
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("11_all.csv"), "STAID,COMID\na,1\n").unwrap();
        fs::write(dir.join("74_all.csv"), "STAID,COMID\nb,2\n").unwrap();
        let fp1 = gages_fingerprint(&dir).unwrap();
        assert_eq!(fp1, gages_fingerprint(&dir).unwrap());

        fs::write(dir.join("readme.txt"), "ignored").unwrap();
        assert_eq!(fp1, gages_fingerprint(&dir).unwrap(), "non-CSV ignored");

        fs::write(dir.join("74_all.csv"), "STAID,COMID\nb,3\n").unwrap();
        let fp2 = gages_fingerprint(&dir).unwrap();
        assert_ne!(fp1, fp2, "editing a zone CSV must change the key");

        fs::rename(dir.join("74_all.csv"), dir.join("75_all.csv")).unwrap();
        assert_ne!(fp2, gages_fingerprint(&dir).unwrap(), "rename changes key");
    }

    /// Caches written before the fabric-stem naming scheme used fixed
    /// `merit_*conus*` names — they must still hit and resolve to the
    /// legacy store paths (rebuild would also invalidate the path-keyed
    /// baseline cache).
    #[test]
    fn legacy_conus_named_cache_still_hits() {
        let ws = tmp_workspace("legacy");
        let dbf_path = ws.join("fabric.dbf");
        let gages_path = ws.join("gages.csv");
        write_tiny_dbf(&dbf_path);
        write_tiny_gages(&gages_path);

        let out1 = resolve_or_build(&ws, &dbf_path, None, &gages_path).expect("build");
        let dir = adjacency_cache_dir(&ws, &out1.key);
        fs::rename(&out1.paths.conus, dir.join("merit_conus_adjacency.zarr")).unwrap();
        fs::rename(&out1.paths.gages, dir.join("merit_gages_conus_adjacency.zarr")).unwrap();

        let out2 = resolve_or_build(&ws, &dbf_path, None, &gages_path).expect("hit");
        assert!(out2.cache_hit, "legacy-named cache must still hit");
        assert!(out2.paths.conus.ends_with("merit_conus_adjacency.zarr"));
        assert!(out2.paths.gages.ends_with("merit_gages_conus_adjacency.zarr"));
        assert!(out2.paths.conus.is_dir() && out2.paths.gages.is_dir());
    }

    /// A manifest whose zarr stores were deleted by hand must rebuild, not
    /// report a hit pointing at nothing.
    #[test]
    fn manifest_without_stores_rebuilds() {
        let ws = tmp_workspace("stale");
        let dbf_path = ws.join("fabric.dbf");
        let gages_path = ws.join("gages.csv");
        write_tiny_dbf(&dbf_path);
        write_tiny_gages(&gages_path);

        let out1 = resolve_or_build(&ws, &dbf_path, None, &gages_path).expect("build");
        fs::remove_dir_all(&out1.paths.conus).unwrap();
        fs::remove_dir_all(&out1.paths.gages).unwrap();

        let out2 = resolve_or_build(&ws, &dbf_path, None, &gages_path).expect("rebuild");
        assert!(!out2.cache_hit, "stale cache must rebuild, not hit");
        assert!(out2.paths.conus.is_dir() && out2.paths.gages.is_dir());
    }

    /// End-to-end: a directory of gage CSVs flows through resolve_or_build
    /// (the `ddrs plan` path that EISDIR'd before this fix).
    #[test]
    fn cache_builds_with_gages_directory() {
        let ws = tmp_workspace("gdir");
        let dbf_path = ws.join("fabric.dbf");
        write_tiny_dbf(&dbf_path);
        let gages_dir = ws.join("gage_csvs");
        fs::create_dir(&gages_dir).unwrap();
        write_tiny_gages(&gages_dir.join("74_all.csv"));

        let out = resolve_or_build(&ws, &dbf_path, None, &gages_dir)
            .expect("build with gages directory");
        assert!(!out.cache_hit);
        let out2 = resolve_or_build(&ws, &dbf_path, None, &gages_dir)
            .expect("second call");
        assert!(out2.cache_hit, "same dir contents → cache hit");
        assert_eq!(out.key, out2.key);
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
        let out1 = resolve_or_build(&ws, &dbf_path, None, &gages_path)
            .expect("first build");
        assert!(!out1.cache_hit, "first call should be a cache miss");
        assert_eq!(out1.key.len(), 16);

        // zarr directories must exist, be readable, and carry the fabric stem.
        assert!(out1.paths.conus.exists(), "network zarr must exist");
        assert!(out1.paths.gages.exists(), "gages zarr must exist");
        assert!(out1.paths.conus.ends_with("fabric_adjacency.zarr"));
        assert!(out1.paths.gages.ends_with("fabric_gages_adjacency.zarr"));

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
        // resolved path and fabric_path must agree (both are .dbf here).
        assert_eq!(m.fabric_path, dbf_path);
        assert_eq!(m.fabric_resolved_path, dbf_path);
        assert!(m.fabric_layer.is_none(), "dbf fabric has no layer");

        // Second call: cache hit.
        let out2 = resolve_or_build(&ws, &dbf_path, None, &gages_path)
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

        let out1 = resolve_or_build(&ws, &dbf_v1, None, &gages_path).expect("build v1");
        let out2 = resolve_or_build(&ws, &dbf_v2, None, &gages_path).expect("build v2");

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
        let k = content_key(dbf_fp, gages_fp, None);

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

        let out = resolve_or_build(&ws, &dbf_path, None, &gages_path).expect("build");
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
