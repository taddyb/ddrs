use std::fs;
use std::path::Path;

use crate::cli::workspace::Workspace;
use crate::error::CliError;

pub fn run_status(ws: &Workspace, as_json: bool) -> Result<(), CliError> {
    let runs = ws.runs_dir();
    let total = walk_size(&runs).unwrap_or(0);
    let total_gb = total as f64 / 1e9;
    let last_run = latest_run_id(&runs)?;
    let lock_present = ws.lockfile().is_file();
    if as_json {
        let v = serde_json::json!({
            "workspace": ws.root(),
            "lockfile_present": lock_present,
            "last_run": last_run,
            "runs_dir_bytes": total,
            "runs_dir_gb": total_gb,
        });
        println!("{}", serde_json::to_string_pretty(&v)
            .map_err(|e| CliError::Other(Box::new(e)))?);
    } else {
        println!("workspace     {}", ws.root().display());
        println!("lockfile      {}", if lock_present { "present" } else { "missing" });
        println!("last run      {}", last_run.unwrap_or_else(|| "(none)".into()));
        println!(".ddrs/runs/   {:.2} GB", total_gb);
        if total_gb > 10.0 {
            println!("hint: total runs/ exceeds 10 GB — consider `ddrs gc`");
        }
    }
    Ok(())
}

fn walk_size(p: &Path) -> std::io::Result<u64> {
    if !p.is_dir() { return Ok(0); }
    let mut total = 0;
    for e in fs::read_dir(p)? {
        let e = e?;
        let md = e.metadata()?;
        total += if md.is_dir() { walk_size(&e.path())? } else { md.len() };
    }
    Ok(total)
}

fn latest_run_id(runs_dir: &Path) -> Result<Option<String>, CliError> {
    if !runs_dir.is_dir() { return Ok(None); }
    let mut e: Vec<_> = fs::read_dir(runs_dir)?
        .filter_map(Result::ok)
        .map(|x| x.file_name().to_string_lossy().into_owned())
        .collect();
    e.sort();
    Ok(e.pop())
}
