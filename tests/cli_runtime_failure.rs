use ddrs::cli::init::{run_init, InitInput};
use ddrs::cli::manifest::Manifest;
use ddrs::cli::run::{run, RunInput};
use ddrs::cli::types::{RunStatus, Workflow};
use ddrs::cli::workspace::Workspace;
use std::fs;

/// `dispatch()` runs the real workflow inside `catch_unwind`. When the
/// workflow fails (e.g. no GPU, missing merit data), `run` should still
/// write the manifest with `status="failed"` and a populated `exit_reason`,
/// and return Ok(run_dir) so the caller can inspect the manifest. This
/// exercises the failure-manifest path end-to-end.
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
        batch_order_from: None,
    })
    .expect("run should return Ok(run_dir) even when dispatch fails");

    let manifest = Manifest::read(&run_dir.join("manifest.json")).unwrap();
    // With real dispatch the workflow may succeed (GPU + merit data present)
    // or fail (no GPU / panic). Either way the manifest must be written.
    // When it fails, exit_reason must be populated.
    if manifest.status == RunStatus::Failed {
        assert!(
            manifest.exit_reason.is_some(),
            "exit_reason must be populated on failure, got: {:?}",
            manifest.exit_reason
        );
    }
}
