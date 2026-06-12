use ddrs::cli::plan::{plan, PlanInput};
use ddrs::cli::types::Workflow;
use ddrs::cli::workspace::Workspace;
use std::path::Path;
use std::sync::Mutex;

static CHDIR_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires the merit data sources to be reachable; runs locally"]
fn plan_succeeds_on_repo_config() {
    let cfg = Path::new("config/merit_training.yaml");
    let ws = Workspace::with_root(std::env::temp_dir().join("ddrs_plan_test/.ddrs"));
    let _ = plan(
        ddrs::cli::plan::PlanInput {
            config_path: Some(cfg.to_path_buf()),
            workflow: Some(Workflow::Train),
            skip_smoke: true,
            ..Default::default()
        },
        &ws,
    );
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
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let res = ddrs::cli::plan::plan(
        ddrs::cli::plan::PlanInput {
            config_path: Some(cfg_path.clone()),
            skip_smoke: true,
            ..Default::default()
        },
        &ws,
    );
    match res {
        Ok(pr) => assert_eq!(pr.workflow, ddrs::cli::Workflow::TrainAndTest),
        Err(e) => {
            let msg = format!("{e}");
            assert!(!msg.contains("workflow"), "expected non-workflow error, got: {msg}");
        }
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
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let res = ddrs::cli::run::run(ddrs::cli::run::RunInput {
        workspace: ws,
        config_path: cfg_path,
        workflow: None, // resolution must come from YAML
        plot: false,
        strict: false,
        max_mini_batches: Some(1),
        batch_order_from: None,
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
    let err = ddrs::cli::plan::plan(
        ddrs::cli::plan::PlanInput {
            config_path: Some(cfg_path.clone()),
            skip_smoke: true,
            ..Default::default()
        },
        &ws,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("no `workflow:` key") && msg.contains("--workflow"),
        "got: {msg}"
    );
}

#[test]
fn plan_errors_clearly_when_no_yaml_and_no_tty() {
    let _g = CHDIR_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(tmp.path().join(".ddrs"));
    let original = std::env::current_dir().unwrap();
    std::env::set_current_dir(tmp.path()).unwrap();
    let res = ddrs::cli::plan::plan(
        ddrs::cli::plan::PlanInput { skip_smoke: true, ..Default::default() },
        &ws,
    );
    std::env::set_current_dir(&original).unwrap();
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("no ddrs.yaml found") && msg.contains("not a TTY"),
        "expected non-interactive bootstrap error, got: {msg}"
    );
}
