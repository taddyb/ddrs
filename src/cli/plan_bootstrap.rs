use serde::Deserialize;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cli::error::CliError;

pub struct BootstrapInput {
    pub target: PathBuf,
    pub runs_dir: PathBuf,
    pub bundled_template: PathBuf,
    pub editor_cmd: Option<String>,
    pub interactive: bool,
}

#[derive(Debug)]
pub enum BootstrapSource {
    LastSuccessful(PathBuf),
    Template,
}

pub fn bootstrap(input: BootstrapInput) -> Result<BootstrapSource, CliError> {
    // TTY guard for interactive mode (skipped during tests with interactive: false).
    if input.interactive && !std::io::stdin().is_terminal() {
        return Err(CliError::ConfigInvalid {
            path: input.target.clone(),
            source: Box::<dyn std::error::Error + Send + Sync>::from(
                "no ddrs.yaml found; pass --config or run interactively",
            ),
        });
    }

    let chosen = pick_source(&input)?;
    let src_path = match &chosen {
        BootstrapSource::LastSuccessful(p) => p.clone(),
        BootstrapSource::Template => input.bundled_template.clone(),
    };
    // Ensure target dir exists, then copy.
    if let Some(parent) = input.target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&src_path, &input.target)?;

    let editor = input
        .editor_cmd
        .or_else(|| std::env::var("EDITOR").ok())
        .unwrap_or_else(|| "vi".to_string());
    Command::new(&editor).arg(&input.target).status()?;
    Ok(chosen)
}

#[derive(Deserialize)]
struct ManifestStatusOnly {
    status: String,
}

fn pick_source(input: &BootstrapInput) -> Result<BootstrapSource, CliError> {
    match latest_successful_run(&input.runs_dir)? {
        Some(p) => Ok(BootstrapSource::LastSuccessful(p)),
        None => Ok(BootstrapSource::Template),
    }
}

fn latest_successful_run(runs_dir: &Path) -> Result<Option<PathBuf>, CliError> {
    if !runs_dir.is_dir() {
        return Ok(None);
    }
    let mut entries: Vec<_> = fs::read_dir(runs_dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.reverse(); // latest first
    for d in entries {
        let mpath = d.join("manifest.json");
        if !mpath.is_file() {
            continue;
        }
        let s = fs::read_to_string(&mpath)?;
        if let Ok(m) = serde_json::from_str::<ManifestStatusOnly>(&s) {
            if m.status == "ok" {
                return Ok(Some(d.join("config.yaml")));
            }
        }
    }
    Ok(None)
}
