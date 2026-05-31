use ddrs::cli::show::run_show;
use ddrs::cli::workspace::Workspace;
use std::fs;

fn write_stub_manifest(dir: &std::path::Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("manifest.json"),
        r#"{
            "run_id": "stub-train",
            "ddrs_version": "0.1.0",
            "git": {"sha": "deadbeef", "dirty": false, "branch": "main"},
            "workflow": "train",
            "config_path": "config.yaml",
            "started_at": "2026-05-30T00:00:00Z",
            "finished_at": "2026-05-30T01:00:00Z",
            "status": "ok",
            "exit_reason": null,
            "system": {},
            "sources": {},
            "source_lock": {"lockfile": ".ddrs/sources.lock", "matched": true, "drift": []},
            "outputs": {"checkpoints": [], "plot": null},
            "metrics": {"final_loss": 0.385},
            "max_mini_batches": null
        }"#,
    )
    .unwrap();
}

#[test]
fn show_renders_human_output_for_stub_manifest() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    write_stub_manifest(&ws.runs_dir().join("stub-train"));
    run_show(&ws, "stub-train", false).expect("show should succeed");
}

#[test]
fn show_renders_json_output_for_stub_manifest() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    write_stub_manifest(&ws.runs_dir().join("stub-train"));
    run_show(&ws, "stub-train", true).expect("show --json should succeed");
}

#[test]
fn show_errors_on_missing_run_id() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let err = run_show(&ws, "no-such-run", false).unwrap_err();
    assert!(matches!(err, ddrs::cli::CliError::Io(_)));
}
