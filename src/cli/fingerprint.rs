use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::cli::CliError;

/// Stat-and-content fingerprint stored in `sources.lock` and the per-run
/// manifest. `fp` is opaque — see spec § Schemas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub path: PathBuf,
    pub mtime: String,
    pub size: u64,
    pub fp: String,
}

#[derive(Debug)]
pub struct ReuseResult {
    pub fp: String,
    pub mtime: String,
    pub size: u64,
    pub reused: bool,
}

pub fn fingerprint_path(path: &Path) -> Result<Fingerprint, CliError> {
    let md = fs::metadata(path).map_err(|_| CliError::DataSourceMissing { path: path.into() })?;
    let size = md.len();
    let mtime = systime_to_iso(md.modified()?);
    let fp = compute_fp(path, &md)?;
    Ok(Fingerprint { path: path.into(), mtime, size, fp })
}

/// Reuse the locked `fp` when `(path, mtime, size)` is unchanged; otherwise
/// re-hash.
pub fn reuse_if_unchanged(path: &Path, locked: &Fingerprint) -> Result<ReuseResult, CliError> {
    let md = fs::metadata(path).map_err(|_| CliError::DataSourceMissing { path: path.into() })?;
    let size = md.len();
    let mtime = systime_to_iso(md.modified()?);
    if size == locked.size && mtime == locked.mtime {
        return Ok(ReuseResult { fp: locked.fp.clone(), mtime, size, reused: true });
    }
    let fp = compute_fp(path, &md)?;
    Ok(ReuseResult { fp, mtime, size, reused: false })
}

fn compute_fp(path: &Path, md: &fs::Metadata) -> Result<String, CliError> {
    // For directories (zarr / icechunk stores), hash the root metadata file.
    // For regular files (CSV, NetCDF), hash full content.
    let bytes = if md.is_dir() {
        for candidate in ["zarr.json", ".zarray", ".zgroup"] {
            let p = path.join(candidate);
            if p.is_file() {
                return Ok(format!("blake3:{}", blake3::hash(&fs::read(p)?).to_hex()));
            }
        }
        let mut names: Vec<_> = fs::read_dir(path)?
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names.join("\n").into_bytes()
    } else {
        fs::read(path)?
    };
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

fn systime_to_iso(t: SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Utc> = t.into();
    dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
