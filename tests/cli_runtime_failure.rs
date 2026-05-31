use ddrs::cli::init::{run_init, InitInput};
use ddrs::cli::manifest::Manifest;
use ddrs::cli::run::{run, RunInput};
use ddrs::cli::types::{RunStatus, Workflow};
use ddrs::cli::workspace::Workspace;
use std::fs;

/// The v1 `dispatch()` is a stub that always returns `RunStatus::Failed`.
/// `run` should still write the manifest with `status="failed"` and a
/// populated `exit_reason`, and return Ok(run_dir) so the caller can
/// inspect the manifest. This exercises the failure-manifest path
/// end-to-end without needing GPU + merit data.
#[test]
fn run_writes_failure_manifest_when_dispatch_stub_fails() {
    let d = tempfile::tempdir().unwrap();
    let cfg = d.path().join("ddrs.yaml");
    fs::copy(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/merit_training.yaml"),
        &cfg,
    )
    .unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));

    // Phase A only (skip_smoke + no config-required Phase B work for run).
    // But `run` calls `plan` which requires a lockfile; we need init Phase B.
    // Skip this test if the merit data sources aren't reachable.
    let init_result = run_init(InitInput {
        workspace: ws.root().to_path_buf(),
        config_path: Some(cfg.clone()),
        min_free_gpu_gb: 0.0,
        force: false,
        skip_smoke: true,
    });
    if let Err(e) = init_result {
        eprintln!("skipping: data sources not reachable ({e})");
        return;
    }

    let run_dir = run(RunInput {
        workspace: Workspace::with_root(ws.root()),
        config_path: cfg,
        workflow: Some(Workflow::Train),
        plot: false,
        strict: false,
        max_mini_batches: Some(1),
    })
    .expect("run should return Ok(run_dir) even when dispatch fails");

    let manifest = Manifest::read(&run_dir.join("manifest.json")).unwrap();
    assert_eq!(manifest.status, RunStatus::Failed);
    assert!(manifest.exit_reason.is_some(), "exit_reason should be populated");
    assert!(
        manifest.exit_reason.as_deref().unwrap().contains("dispatch stub"),
        "exit_reason should mention the dispatch stub: {:?}",
        manifest.exit_reason
    );
}
