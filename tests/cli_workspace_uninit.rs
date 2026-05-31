use ddrs::cli::plan::plan;
use ddrs::cli::types::Workflow;
use ddrs::cli::workspace::Workspace;
use std::fs;

/// `plan` against a workspace with no `.ddrs/sources.lock` must exit
/// `WorkspaceNotInitialized` (6) so the user knows to run `ddrs init`.
#[test]
fn plan_without_init_exits_workspace_not_initialized() {
    let d = tempfile::tempdir().unwrap();
    let cfg = d.path().join("ddrs.yaml");
    // Minimal config — `plan` will fail on data_sources before we get there,
    // but the order of checks matters: lockfile-missing check comes first.
    // Use a copy of merit_training.yaml so the schema parses cleanly.
    fs::copy(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/merit_training.yaml"),
        &cfg,
    )
    .unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    fs::create_dir_all(ws.root()).unwrap();
    let err = plan(&cfg, Some(Workflow::Train), &ws).unwrap_err();
    assert!(
        matches!(err, ddrs::cli::CliError::WorkspaceNotInitialized { .. }),
        "expected WorkspaceNotInitialized, got: {err:?}"
    );
}
