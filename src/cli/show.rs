use crate::cli::{manifest::Manifest, workspace::Workspace};
use crate::error::CliError;

pub fn run_show(ws: &Workspace, run_id: &str, as_json: bool) -> Result<(), CliError> {
    let path = ws.runs_dir().join(run_id).join("manifest.json");
    let m = Manifest::read(&path)?;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&m)
            .map_err(|e| CliError::Other(Box::new(e)))?);
    } else {
        println!("run      {}", m.run_id);
        println!("status   {:?}", m.status);
        println!("workflow {:?}", m.workflow);
        println!("started  {}", m.started_at);
        if let Some(f) = &m.finished_at { println!("finished {}", f); }
        println!("git      {} ({})", m.git.sha, if m.git.dirty { "dirty" } else { "clean" });
        println!("drift    {:?}", m.source_lock.drift);
        if let Some(ra) = &m.resolved_adjacency {
            println!("adjacency");
            println!("  conus  {}", ra.conus.display());
            println!("  gages  {}", ra.gages.display());
            if let Some(key) = &ra.cache_key {
                println!(
                    "  cache  {} ({})",
                    key,
                    if ra.cache_hit == Some(true) { "hit" } else { "built" },
                );
            }
        }
        if let Some(p) = &m.outputs.plot { println!("plot     {}", p.display()); }
    }
    Ok(())
}
