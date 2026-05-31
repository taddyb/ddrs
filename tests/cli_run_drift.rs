//! End-to-end drift detection test.
//!
//! Requires the merit data sources at the paths defined in
//! `config/merit_training.yaml`. The test is `#[ignore]`-gated; run with
//! `cargo test --test cli_run_drift -- --ignored` on a host where those
//! paths exist.

use ddrs::cli::init::{run_init, InitInput};
use ddrs::cli::run::{run, RunInput};
use ddrs::cli::types::Workflow;
use ddrs::cli::workspace::Workspace;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
#[ignore = "requires merit data sources reachable + filetime dev-dep"]
fn run_warns_on_drift_strict_fails_on_drift() {
    let d = tempfile::tempdir().unwrap();
    let cfg = d.path().join("ddrs.yaml");
    fs::copy(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/merit_training.yaml"),
        &cfg,
    )
    .unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));

    run_init(InitInput {
        workspace: ws.root().to_path_buf(),
        config_path: Some(cfg.clone()),
        min_free_gpu_gb: 0.0,
        force: false,
        skip_smoke: true,
    })
    .expect("init must succeed when merit data is reachable");

    // Touch one of the data source files to force a drift fp diff.
    let gages = std::path::PathBuf::from(
        "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv",
    );
    if gages.exists() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        // Use `filetime` if available; otherwise use std::fs::File + set_modified
        // (Rust 1.75+).
        if let Ok(f) = std::fs::OpenOptions::new().write(true).open(&gages) {
            let _ = f.set_modified(SystemTime::now());
            let _ = now;
        }
    }

    // Strict mode → LockDrift error.
    let err = run(RunInput {
        workspace: Workspace::with_root(ws.root()),
        config_path: cfg.clone(),
        workflow: Some(Workflow::Train),
        plot: false,
        strict: true,
        max_mini_batches: Some(1),
    })
    .unwrap_err();
    assert!(
        matches!(err, ddrs::cli::CliError::LockDrift { .. }),
        "expected LockDrift, got: {err:?}"
    );
}
