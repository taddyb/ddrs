use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::cli::CliError;
use crate::data::store::icechunk::main_branch_snapshot;

/// Stat-and-content fingerprint stored in `sources.lock` and the per-run
/// manifest. `fp` is opaque — see spec § Schemas.
///
/// `snapshot` is set only for icechunk repositories, where it carries the
/// `main` branch tip that `fp` is derived from. It is optional and skipped
/// when absent so locks and manifests written before content fingerprints
/// still parse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub path: PathBuf,
    pub mtime: String,
    pub size: u64,
    pub fp: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
}

#[derive(Debug)]
pub struct ReuseResult {
    pub fp: String,
    pub mtime: String,
    pub size: u64,
    pub snapshot: Option<String>,
    pub reused: bool,
}

pub fn fingerprint_path(path: &Path) -> Result<Fingerprint, CliError> {
    fingerprint_path_at(path, None)
}

/// As [`fingerprint_path`], but for a source pinned to an icechunk snapshot
/// (`data_sources.pins`): the fingerprint then records the PIN, because that is
/// the data version the run will actually read. Unpinned icechunk sources still
/// record the resolved `main` tip.
pub fn fingerprint_path_at(path: &Path, pin: Option<&str>) -> Result<Fingerprint, CliError> {
    let md = fs::metadata(path).map_err(|_| CliError::DataSourceMissing { path: path.into() })?;
    let size = md.len();
    let mtime = systime_to_iso(md.modified()?);
    let (fp, snapshot) = compute_fp(path, &md, pin)?;
    Ok(Fingerprint { path: path.into(), mtime, size, fp, snapshot })
}

/// Reuse the locked `fp` when `(path, mtime, size)` is unchanged; otherwise
/// re-hash.
///
/// The stat short-circuit applies to **regular files only**. A directory's own
/// mtime and size do not change when a chunk deep inside it is rewritten, so
/// reusing on stat would make a mutated store look pristine. Directories are
/// therefore always recomputed, and the cost is real: one full recursive walk
/// per `ddrs plan` of every directory source, which for the CONUS adjacency
/// and Q' stores is on the order of the store's file count (icechunk repos
/// escape this — their fingerprint is one ref read, no walk).
pub fn reuse_if_unchanged(path: &Path, locked: &Fingerprint) -> Result<ReuseResult, CliError> {
    reuse_if_unchanged_at(path, locked, None)
}

/// As [`reuse_if_unchanged`], with the source's `data_sources.pins` value.
pub fn reuse_if_unchanged_at(
    path: &Path,
    locked: &Fingerprint,
    pin: Option<&str>,
) -> Result<ReuseResult, CliError> {
    let md = fs::metadata(path).map_err(|_| CliError::DataSourceMissing { path: path.into() })?;
    let size = md.len();
    let mtime = systime_to_iso(md.modified()?);
    if !md.is_dir() && size == locked.size && mtime == locked.mtime {
        return Ok(ReuseResult {
            fp: locked.fp.clone(),
            mtime,
            size,
            snapshot: locked.snapshot.clone(),
            reused: true,
        });
    }
    let (fp, snapshot) = compute_fp(path, &md, pin)?;
    Ok(ReuseResult { fp, mtime, size, snapshot, reused: false })
}

/// Returns `(fp, snapshot)`.
///
/// * regular file (CSV, NetCDF, gpkg) — blake3 over the full content.
/// * icechunk repository — `icechunk:<snapshot id>`: the configured `pin` when
///   there is one, else the `main` branch tip. Either way it is the version the
///   run will read, and a snapshot id is a content hash by construction, so no
///   walk is needed.
/// * any other directory (zarr v2/v3 stores, unpacked fabrics) — blake3 over
///   the recursive `<relative path>\n<size>\n` listing plus the bytes of the
///   root metadata file, if any.
fn compute_fp(
    path: &Path,
    md: &fs::Metadata,
    pin: Option<&str>,
) -> Result<(String, Option<String>), CliError> {
    if !md.is_dir() {
        return Ok((format!("blake3:{}", blake3::hash(&fs::read(path)?).to_hex()), None));
    }
    if is_icechunk_repo(path) {
        // A pin is already an immutable snapshot id — no ref read needed, and
        // recording the tip here would name a version the run never touches.
        // An id that does not resolve is caught when the store is opened.
        let snapshot = match pin {
            Some(id) => id.to_string(),
            None => main_branch_snapshot(path).map_err(|e| {
                CliError::Runtime(format!(
                    "icechunk fingerprint failed for {}: {e}",
                    path.display()
                ))
            })?,
        };
        return Ok((format!("icechunk:{snapshot}"), Some(snapshot)));
    }
    let (hex, files) = hash_dir(path)?;
    if std::env::var_os("DDRS_DEBUG").is_some() {
        eprintln!("fingerprint: {} files walked under {}", files, path.display());
    }
    Ok((format!("blake3:{hex}"), None))
}

/// An icechunk repository keeps its snapshots and chunk manifests in fixed
/// top-level directories (`refs/` in the v1 layout).
fn is_icechunk_repo(path: &Path) -> bool {
    (path.join("snapshots").is_dir() && path.join("manifests").is_dir())
        || path.join("refs").is_dir()
}

/// Hash the recursive contents of `root`. Returns `(hex digest, file count)`.
///
/// Paths are relative to `root` and sorted, so the digest is independent of
/// readdir order. Sizes stand in for chunk bytes: a zarr chunk rewrite that
/// preserves length is caught by the root metadata bytes only if the metadata
/// changed, so this is a cheap-but-not-cryptographic content check — the same
/// tradeoff the mtime/size stat check makes, applied one level deeper.
/// Symlinks are recorded by their own size and never followed, so a cycle
/// cannot hang the walk.
fn hash_dir(root: &Path) -> Result<(String, usize), CliError> {
    let mut entries: Vec<(String, u64)> = Vec::new();
    collect(root, root, &mut entries)?;
    entries.sort();

    let mut hasher = blake3::Hasher::new();
    for (rel, size) in &entries {
        hasher.update(rel.as_bytes());
        hasher.update(b"\n");
        hasher.update(size.to_string().as_bytes());
        hasher.update(b"\n");
    }
    for candidate in ["zarr.json", ".zarray", ".zgroup"] {
        let p = root.join(candidate);
        if p.is_file() {
            hasher.update(&fs::read(p)?);
            break;
        }
    }
    Ok((hasher.finalize().to_hex().to_string(), entries.len()))
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<(String, u64)>) -> Result<(), CliError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        let p = entry.path();
        if ft.is_dir() {
            collect(root, &p, out)?;
        } else {
            let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().into_owned();
            out.push((rel, entry.metadata()?.len()));
        }
    }
    Ok(())
}

fn systime_to_iso(t: SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Utc> = t.into();
    dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
