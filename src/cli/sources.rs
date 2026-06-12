//! `ddrs sources` — named data-source groups ("save files").
//!
//! A group is a YAML file holding exactly one top-level `data_sources:`
//! block, stored under `config/sources/<name>.yaml` (sibling of the
//! workspace config, tracked in git so groups are shareable). The point is
//! switching the whole input set — e.g. CONUS icechunk ↔ global
//! dMC_global_v3.1 — without hand-editing `ddrs.yaml` each time:
//!
//! ```text
//! ddrs sources save conus     # snapshot current data_sources as "conus"
//! ddrs sources use global     # splice group "global" into ddrs.yaml + re-lock
//! ddrs sources list           # show groups; '*' marks the active one
//! ```
//!
//! `save` and `use` are **textual**: the `data_sources:` block is extracted
//! / replaced as lines, so comments inside the block travel with it and the
//! rest of `ddrs.yaml` (experiment, kan, comments) is byte-identical after
//! a switch. `use` validates the spliced config parses before committing
//! (write-to-temp + rename) and refreshes `sources.lock` when a workspace
//! exists, so `ddrs plan` sees no drift.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::cli::fingerprint::fingerprint_path;
use crate::cli::lockfile::Lockfile;
use crate::cli::workspace::Workspace;
use crate::config::{Config, ConfigMode, DataSources};
use crate::error::CliError;

/// Where group files live: `<config dir>/config/sources/`. For the standard
/// repo-root `ddrs.yaml` this is `config/sources/`.
pub fn groups_dir(cfg_path: &Path) -> PathBuf {
    cfg_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("config")
        .join("sources")
}

/// Used only to validate that a group file / extracted block is a
/// well-formed `data_sources:` mapping.
#[derive(Deserialize)]
struct GroupFile {
    #[allow(dead_code)]
    data_sources: DataSources,
}

fn validate_name(name: &str) -> Result<(), CliError> {
    let ok = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && !name.starts_with('.');
    if ok {
        Ok(())
    } else {
        Err(CliError::Runtime(format!(
            "invalid group name {name:?}: use alphanumerics, '-', '_'"
        )))
    }
}

fn group_path(cfg_path: &Path, name: &str) -> PathBuf {
    groups_dir(cfg_path).join(format!("{name}.yaml"))
}

/// `[start, end)` line range of the top-level `data_sources:` block.
/// The block runs from the `data_sources:` line until the first following
/// line that starts a new top-level construct (any non-blank line without
/// leading whitespace — keys and unindented comments both end the block).
fn block_range(lines: &[&str]) -> Option<(usize, usize)> {
    let start = lines.iter().position(|l| {
        l.trim_end() == "data_sources:" || l.starts_with("data_sources:")
    })?;
    let mut end = lines.len();
    for (i, l) in lines.iter().enumerate().skip(start + 1) {
        let blank = l.trim().is_empty();
        let indented = l.starts_with(' ') || l.starts_with('\t');
        if !blank && !indented {
            end = i;
            break;
        }
    }
    // Don't swallow trailing blank separator lines into the block.
    while end > start + 1 && lines[end - 1].trim().is_empty() {
        end -= 1;
    }
    Some((start, end))
}

/// Extract the `data_sources:` block from the config, verbatim.
fn extract_block(cfg_text: &str, cfg_path: &Path) -> Result<String, CliError> {
    let lines: Vec<&str> = cfg_text.lines().collect();
    let (start, end) = block_range(&lines).ok_or_else(|| CliError::ConfigInvalid {
        path: cfg_path.to_path_buf(),
        source: "no top-level `data_sources:` block found".into(),
    })?;
    let mut block = lines[start..end].join("\n");
    block.push('\n');
    Ok(block)
}

/// Save the current config's `data_sources:` block as group `name`.
pub fn run_save(cfg_path: &Path, name: &str, force: bool) -> Result<PathBuf, CliError> {
    validate_name(name)?;
    let cfg_text = fs::read_to_string(cfg_path)?;
    let block = extract_block(&cfg_text, cfg_path)?;
    // Validate before persisting.
    serde_yaml::from_str::<GroupFile>(&block).map_err(|e| CliError::ConfigInvalid {
        path: cfg_path.to_path_buf(),
        source: Box::new(e),
    })?;

    let dest = group_path(cfg_path, name);
    if dest.exists() && !force {
        return Err(CliError::Runtime(format!(
            "group {name:?} already exists at {} — pass --force to overwrite",
            dest.display()
        )));
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&dest, block)?;
    Ok(dest)
}

/// Splice group `name` into the config's `data_sources:` block and refresh
/// `sources.lock` (when a workspace exists). Returns true if the lock was
/// refreshed.
pub fn run_use(cfg_path: &Path, name: &str, ws: &Workspace) -> Result<bool, CliError> {
    validate_name(name)?;
    let src = group_path(cfg_path, name);
    if !src.is_file() {
        let available = list_group_names(cfg_path)?;
        return Err(CliError::Runtime(format!(
            "no group {name:?} at {} (available: {})",
            src.display(),
            if available.is_empty() { "none".into() } else { available.join(", ") },
        )));
    }
    let group_text = fs::read_to_string(&src)?;
    serde_yaml::from_str::<GroupFile>(&group_text).map_err(|e| CliError::ConfigInvalid {
        path: src.clone(),
        source: Box::new(e),
    })?;

    let cfg_text = fs::read_to_string(cfg_path)?;
    let lines: Vec<&str> = cfg_text.lines().collect();
    let (start, end) = block_range(&lines).ok_or_else(|| CliError::ConfigInvalid {
        path: cfg_path.to_path_buf(),
        source: "no top-level `data_sources:` block found".into(),
    })?;

    let mut out = String::new();
    for l in &lines[..start] {
        out.push_str(l);
        out.push('\n');
    }
    out.push_str(group_text.trim_end());
    out.push('\n');
    for l in &lines[end..] {
        out.push_str(l);
        out.push('\n');
    }

    // Validate the spliced config parses before touching ddrs.yaml.
    let tmp = cfg_path.with_extension("yaml.tmp");
    fs::write(&tmp, &out)?;
    let parsed = Config::from_yaml_file(&tmp);
    if let Err(e) = parsed {
        let _ = fs::remove_file(&tmp);
        return Err(CliError::ConfigInvalid {
            path: src,
            source: Box::new(e),
        });
    }
    fs::rename(&tmp, cfg_path)?;

    // Re-lock so `ddrs plan` sees no drift. Only when the workspace exists —
    // before the first `ddrs plan` there is nothing to refresh.
    if ws.lockfile().is_file() {
        lock_sources_from_config(cfg_path, ws)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Fingerprint every configured data source and write `sources.lock`.
/// Used by `ddrs sources use` to re-lock after switching groups (formerly
/// `ddrs init` Phase E; `ddrs plan` carries its own drift-aware variant in
/// `plan.rs` that reuses prior fingerprints via the stat fast-path).
fn lock_sources_from_config(cfg_path: &Path, ws: &Workspace) -> Result<(), CliError> {
    let cfg = Config::from_yaml_file_with_mode(cfg_path, ConfigMode::Training)
        .map_err(|e| CliError::ConfigInvalid { path: cfg_path.to_path_buf(), source: Box::new(e) })?;
    let ds = cfg.data_sources.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: cfg_path.to_path_buf(),
        source: Box::<dyn std::error::Error + Send + Sync>::from("data_sources: missing"),
    })?;

    // Build the fingerprint pairs. Optional keys (adjacency zarr stores,
    // geospatial_fabric) are only locked when explicitly configured.
    let mut pairs: Vec<(&str, PathBuf)> = vec![
        ("attributes", ds.attributes.clone()),
        ("streamflow", ds.streamflow.clone()),
        ("observations", ds.observations.clone()),
        ("gages", ds.gages.clone()),
    ];
    if let Some(p) = &ds.conus_adjacency {
        pairs.push(("conus_adjacency", p.clone()));
    }
    if let Some(p) = &ds.gages_adjacency {
        pairs.push(("gages_adjacency", p.clone()));
    }
    if let Some(p) = &ds.geospatial_fabric {
        pairs.push(("geospatial_fabric", p.clone()));
    }

    // Parallel reachability + fingerprint. std::thread::scope is fine — these
    // are I/O-bound and the count is small.
    let results: Result<Vec<(String, Result<_, CliError>)>, CliError> = std::thread::scope(|s| {
        let handles: Vec<_> = pairs
            .iter()
            .map(|(k, p)| {
                let p = p.clone();
                let k = k.to_string();
                s.spawn(move || (k, fingerprint_path(&p)))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| CliError::Runtime("fingerprint thread panicked".into()))
            })
            .collect()
    });
    let results = results?;
    let mut sources = BTreeMap::new();
    for (k, r) in results {
        sources.insert(k, r?);
    }
    let lock = Lockfile {
        ddrs_version: env!("CARGO_PKG_VERSION").into(),
        created_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        sources,
    };
    lock.write_atomic(&ws.lockfile())?;
    Ok(())
}

fn list_group_names(cfg_path: &Path) -> Result<Vec<String>, CliError> {
    let dir = groups_dir(cfg_path);
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut names: Vec<String> = fs::read_dir(&dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect();
    names.sort();
    Ok(names)
}

pub struct GroupEntry {
    pub name: String,
    pub active: bool,
}

/// List saved groups; `active` marks the one whose block matches the
/// config's current `data_sources:` (compared structurally, not textually,
/// so comment/ordering differences don't break the match).
pub fn run_list(cfg_path: &Path) -> Result<Vec<GroupEntry>, CliError> {
    let current: Option<serde_yaml::Value> = fs::read_to_string(cfg_path)
        .ok()
        .and_then(|t| extract_block(&t, cfg_path).ok())
        .and_then(|b| serde_yaml::from_str(&b).ok());

    let mut out = Vec::new();
    for name in list_group_names(cfg_path)? {
        let group: Option<serde_yaml::Value> = fs::read_to_string(group_path(cfg_path, &name))
            .ok()
            .and_then(|t| serde_yaml::from_str(&t).ok());
        let active = match (&current, &group) {
            (Some(c), Some(g)) => c == g,
            _ => false,
        };
        out.push(GroupEntry { name, active });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: &str = "\
mode: training
geodataset: merit
seed: 1
np_seed: 1
# top-level comment that must survive switching
data_sources:
  attributes: /dev/null/attrs.nc
  # an indented comment that belongs to the conus group
  conus_adjacency: /dev/null/conus.zarr
  gages_adjacency: /dev/null/gages.zarr
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
# trailing top-level comment
";

    const GLOBAL_GROUP: &str = "\
data_sources:
  attributes: /dev/null/attrs.nc
  # global fabric comment travels with the group
  geospatial_fabric: /dev/null/global.gpkg
  streamflow: /dev/null/global_sf
  observations: /dev/null/global_obs
  gages: /dev/null/gage_dir
";

    fn setup() -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join("ddrs.yaml");
        fs::write(&cfg, CFG).unwrap();
        (tmp, cfg)
    }

    #[test]
    fn save_extracts_block_verbatim() {
        let (_tmp, cfg) = setup();
        let dest = run_save(&cfg, "conus", false).unwrap();
        let saved = fs::read_to_string(&dest).unwrap();
        assert!(saved.starts_with("data_sources:\n"));
        assert!(saved.contains("# an indented comment"));
        assert!(saved.contains("conus_adjacency: /dev/null/conus.zarr"));
        assert!(!saved.contains("# trailing top-level comment"));
        assert!(!saved.contains("seed:"));

        // Second save without --force refuses; with force succeeds.
        assert!(run_save(&cfg, "conus", false).is_err());
        run_save(&cfg, "conus", true).unwrap();
    }

    #[test]
    fn use_splices_group_and_preserves_rest() {
        let (tmp, cfg) = setup();
        run_save(&cfg, "conus", false).unwrap();
        fs::write(groups_dir(&cfg).join("global.yaml"), GLOBAL_GROUP).unwrap();

        let ws = Workspace::with_root(tmp.path().join(".ddrs")); // no lockfile
        let relocked = run_use(&cfg, "global", &ws).unwrap();
        assert!(!relocked, "no workspace yet — nothing to re-lock");

        let after = fs::read_to_string(&cfg).unwrap();
        assert!(after.contains("geospatial_fabric: /dev/null/global.gpkg"));
        assert!(after.contains("# global fabric comment"));
        assert!(!after.contains("conus_adjacency"));
        // Everything outside the block is untouched.
        assert!(after.contains("# top-level comment that must survive switching"));
        assert!(after.contains("# trailing top-level comment"));
        assert!(after.contains("seed: 1"));
        // And the result is a valid config.
        Config::from_yaml_file(&cfg).expect("spliced config parses");

        // Round-trip back to conus restores the original sources.
        run_use(&cfg, "conus", &ws).unwrap();
        let back = fs::read_to_string(&cfg).unwrap();
        assert!(back.contains("conus_adjacency: /dev/null/conus.zarr"));
        assert!(!back.contains("geospatial_fabric"));
    }

    #[test]
    fn use_rejects_invalid_group_without_touching_config() {
        let (tmp, cfg) = setup();
        let dir = groups_dir(&cfg);
        fs::create_dir_all(&dir).unwrap();
        // `streamflow` missing → DataSources won't deserialize.
        fs::write(dir.join("broken.yaml"), "data_sources:\n  attributes: /a.nc\n").unwrap();

        let ws = Workspace::with_root(tmp.path().join(".ddrs"));
        let before = fs::read_to_string(&cfg).unwrap();
        assert!(run_use(&cfg, "broken", &ws).is_err());
        assert_eq!(before, fs::read_to_string(&cfg).unwrap());

        // Unknown group name errors and names what exists.
        let err = run_use(&cfg, "nope", &ws).unwrap_err().to_string();
        assert!(err.contains("broken"), "got: {err}");
    }

    #[test]
    fn list_marks_active_group() {
        let (tmp, cfg) = setup();
        run_save(&cfg, "conus", false).unwrap();
        fs::write(groups_dir(&cfg).join("global.yaml"), GLOBAL_GROUP).unwrap();

        let entries = run_list(&cfg).unwrap();
        let active: Vec<&str> = entries
            .iter()
            .filter(|e| e.active)
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(active, vec!["conus"]);

        let ws = Workspace::with_root(tmp.path().join(".ddrs"));
        run_use(&cfg, "global", &ws).unwrap();
        let entries = run_list(&cfg).unwrap();
        let active: Vec<&str> = entries
            .iter()
            .filter(|e| e.active)
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(active, vec!["global"]);
    }
}
