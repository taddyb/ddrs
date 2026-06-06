use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use crate::cli::{manifest::Manifest, workspace::Workspace};
use crate::error::CliError;

pub struct GcInput {
    pub keep: Option<usize>,
    pub keep_successful: bool,
    pub older_than: Option<Duration>,
    pub dry_run: bool,
}

/// Prune `.ddrs/runs/` per the filters in `input`.
///
/// v1 deliberately leaves `.ddrs/adjacency/` caches alone: they are
/// content-addressed and shared across runs, and a managed build is the
/// expensive (~1–2 min) artifact gc exists to protect. Key-based GC of stale
/// adjacency entries (no run references a given key) is a follow-up.
pub fn run_gc(ws: &Workspace, input: GcInput) -> Result<Vec<PathBuf>, CliError> {
    let runs_dir = ws.runs_dir();
    if !runs_dir.is_dir() { return Ok(vec![]); }

    // Without any filter, gc is a no-op — both for actual deletion and for
    // the returned "what would be deleted" list. Avoids the CLI printing
    // "would delete" for every existing run when the user passes no flags.
    let has_filter =
        input.keep.is_some() || input.keep_successful || input.older_than.is_some();
    if !has_filter {
        return Ok(vec![]);
    }

    let mut entries: Vec<PathBuf> = fs::read_dir(&runs_dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();  // oldest first
    let n = entries.len();
    let mut to_delete: Vec<PathBuf> = Vec::new();

    for (idx, dir) in entries.iter().enumerate() {
        let from_newest = n - idx - 1;
        if let Some(k) = input.keep {
            if from_newest < k { continue; }
        }
        if input.keep_successful {
            let mpath = dir.join("manifest.json");
            if mpath.is_file() {
                if let Ok(m) = Manifest::read(&mpath) {
                    if matches!(m.status, crate::cli::types::RunStatus::Ok) { continue; }
                }
            }
        }
        if let Some(threshold) = input.older_than {
            let md = fs::metadata(dir)?;
            let age = md.modified().ok()
                .and_then(|t| t.elapsed().ok())
                .unwrap_or_default();
            if age < threshold { continue; }
        }
        to_delete.push(dir.clone());
    }

    // Filters compose with AND — only delete if ANY filter was passed.
    if !input.dry_run
        && (input.keep.is_some() || input.keep_successful || input.older_than.is_some())
    {
        for d in &to_delete {
            let _ = fs::remove_dir_all(d);
        }
    }
    Ok(to_delete)
}
