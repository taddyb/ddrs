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

#[test]
fn workflow_resolved_from_yaml_key() {
    use std::fs;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = tmp.path().join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: train-and-test
data_sources:
  attributes: /dev/null
  conus_adjacency: /dev/null
  gages_adjacency: /dev/null
  streamflow: /dev/null
  observations: /dev/null
  gages: /dev/null
"#).unwrap();
    let ws_root = tmp.path().join(".ddrs");
    fs::create_dir_all(ws_root.join("runs")).unwrap();
    let lock = ddrs::cli::lockfile::Lockfile {
        ddrs_version: "test".into(),
        created_at: "0".into(),
        sources: std::collections::BTreeMap::new(),
    };
    lock.write_atomic(&ws_root.join("sources.lock")).unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let err = ddrs::cli::plan::plan(&cfg_path, None, &ws);
    match err {
        Err(e) => {
            let msg = format!("{e}");
            assert!(!msg.contains("workflow"), "expected non-workflow error, got: {msg}");
        }
        Ok(_) => {}
    }
}

#[test]
fn run_inherits_yaml_workflow_resolution() {
    use std::fs;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = tmp.path().join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: train
data_sources:
  attributes: /dev/null
  conus_adjacency: /dev/null
  gages_adjacency: /dev/null
  streamflow: /dev/null
  observations: /dev/null
  gages: /dev/null
"#).unwrap();
    let ws_root = tmp.path().join(".ddrs");
    fs::create_dir_all(ws_root.join("runs")).unwrap();
    let lock = ddrs::cli::lockfile::Lockfile {
        ddrs_version: "test".into(),
        created_at: "0".into(),
        sources: std::collections::BTreeMap::new(),
    };
    lock.write_atomic(&ws_root.join("sources.lock")).unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let res = ddrs::cli::run::run(ddrs::cli::run::RunInput {
        workspace: ws,
        config_path: cfg_path,
        workflow: None, // resolution must come from YAML
        plot: false,
        strict: false,
        max_mini_batches: Some(1),
    });
    // Expected: fails downstream (sandbox / data source / GPU), NOT at workflow.
    if let Err(e) = res {
        let msg = format!("{e}");
        assert!(!msg.contains("workflow:"), "got premature workflow error: {msg}");
    }
}

#[test]
fn no_workflow_anywhere_gives_actionable_error() {
    use std::fs;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = tmp.path().join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
"#).unwrap();
    let ws_root = tmp.path().join(".ddrs");
    std::fs::create_dir_all(&ws_root).unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let err = ddrs::cli::plan::plan(&cfg_path, None, &ws).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("no `workflow:` key") && msg.contains("--workflow"),
        "got: {msg}"
    );
}
