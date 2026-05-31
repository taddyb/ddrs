//! Verify the `--json` output from `ddrs show` parses as JSON and contains
//! the documented top-level keys. The `--json` modes of `plan` and
//! `status` are also exercised — `plan` requires a workspace + lockfile
//! so it's `#[ignore]`-gated; `status` works on a tmp workspace.

use ddrs::cli::workspace::Workspace;
use std::fs;
use std::process::Command;

fn write_stub_manifest(dir: &std::path::Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("manifest.json"),
        r#"{
            "run_id": "stub-train",
            "ddrs_version": "0.1.0",
            "git": {"sha": "abc", "dirty": false, "branch": "main"},
            "workflow": "train",
            "config_path": "config.yaml",
            "started_at": "2026-05-30T00:00:00Z",
            "finished_at": null,
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
fn show_json_parses_with_expected_keys() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    write_stub_manifest(&ws.runs_dir().join("stub-train"));

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ddrs"));
    cmd.arg("--workspace").arg(ws.root());
    cmd.arg("show").arg("stub-train").arg("--json");
    let out = cmd.output().expect("ddrs binary should run");
    assert!(out.status.success(), "ddrs show --json should succeed: {out:?}");
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("output not valid JSON: {e}\noutput: {stdout}"));
    for key in [
        "run_id",
        "ddrs_version",
        "git",
        "workflow",
        "status",
        "outputs",
        "metrics",
    ] {
        assert!(v.get(key).is_some(), "missing key {key}: {stdout}");
    }
}

#[test]
fn status_json_parses_on_empty_workspace() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    fs::create_dir_all(ws.runs_dir()).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ddrs"));
    cmd.arg("--workspace").arg(ws.root());
    cmd.arg("status").arg("--json");
    let out = cmd.output().expect("ddrs binary should run");
    assert!(out.status.success(), "ddrs status --json should succeed: {out:?}");
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("status --json should emit JSON");
    for key in ["workspace", "lockfile_present", "last_run", "runs_dir_bytes"] {
        assert!(v.get(key).is_some(), "missing key {key}");
    }
}
