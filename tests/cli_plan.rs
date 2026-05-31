use ddrs::cli::plan::plan;
use ddrs::cli::types::Workflow;
use ddrs::cli::workspace::Workspace;
use std::path::Path;

#[test]
#[ignore = "requires the merit data sources to be reachable; runs locally"]
fn plan_succeeds_on_repo_config() {
    let cfg = Path::new("config/merit_training.yaml");
    let ws = Workspace::with_root(std::env::temp_dir().join("ddrs_plan_test/.ddrs"));
    let _ = plan(cfg, Some(Workflow::Train), &ws);
}
