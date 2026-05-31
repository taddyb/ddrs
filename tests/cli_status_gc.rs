use ddrs::cli::gc::{run_gc, GcInput};
use ddrs::cli::status::run_status;
use ddrs::cli::workspace::Workspace;
use std::fs;

fn make_run(runs_dir: &std::path::Path, id: &str, status: &str) {
    let d = runs_dir.join(id);
    fs::create_dir_all(&d).unwrap();
    fs::write(
        d.join("manifest.json"),
        format!(
            r#"{{
                "run_id": "{id}",
                "ddrs_version": "x",
                "git": {{"sha": "x", "dirty": false, "branch": "x"}},
                "workflow": "train",
                "config_path": "x",
                "started_at": "x",
                "finished_at": null,
                "status": "{status}",
                "exit_reason": null,
                "system": {{}},
                "sources": {{}},
                "source_lock": {{"lockfile": "x", "matched": true, "drift": []}},
                "outputs": {{"checkpoints": [], "plot": null}},
                "metrics": {{}},
                "max_mini_batches": null
            }}"#
        ),
    )
    .unwrap();
}

#[test]
fn status_runs_against_empty_workspace() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    fs::create_dir_all(ws.runs_dir()).unwrap();
    run_status(&ws, false).expect("status should succeed on empty workspace");
    run_status(&ws, true).expect("status --json should succeed");
}

#[test]
fn gc_dry_run_lists_but_does_not_delete() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    fs::create_dir_all(ws.runs_dir()).unwrap();
    for i in 0..3 {
        make_run(&ws.runs_dir(), &format!("2026-05-3{i}T00-00-00-train"), "failed");
    }
    let listed = run_gc(
        &ws,
        GcInput { keep: Some(1), keep_successful: false, older_than: None, dry_run: true },
    )
    .unwrap();
    assert_eq!(listed.len(), 2, "dry-run should list 2 deletions (keep 1)");
    assert_eq!(
        fs::read_dir(ws.runs_dir()).unwrap().count(),
        3,
        "dry-run should not delete anything",
    );
}

#[test]
fn gc_keep_n_deletes_oldest() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    fs::create_dir_all(ws.runs_dir()).unwrap();
    for i in 0..3 {
        make_run(&ws.runs_dir(), &format!("2026-05-3{i}T00-00-00-train"), "failed");
    }
    let deleted = run_gc(
        &ws,
        GcInput { keep: Some(1), keep_successful: false, older_than: None, dry_run: false },
    )
    .unwrap();
    assert_eq!(deleted.len(), 2);
    assert_eq!(fs::read_dir(ws.runs_dir()).unwrap().count(), 1);
}

#[test]
fn gc_keep_successful_preserves_ok_runs() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    fs::create_dir_all(ws.runs_dir()).unwrap();
    make_run(&ws.runs_dir(), "2026-05-30T00-00-00-train", "ok");
    make_run(&ws.runs_dir(), "2026-05-31T00-00-00-train", "failed");
    let deleted = run_gc(
        &ws,
        GcInput {
            keep: Some(0),
            keep_successful: true,
            older_than: None,
            dry_run: false,
        },
    )
    .unwrap();
    // The "ok" run is kept, the "failed" one is deleted.
    assert_eq!(deleted.len(), 1);
    assert!(deleted[0].file_name().unwrap() == "2026-05-31T00-00-00-train");
}

#[test]
fn gc_with_no_filters_deletes_nothing() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    fs::create_dir_all(ws.runs_dir()).unwrap();
    for i in 0..2 {
        make_run(&ws.runs_dir(), &format!("2026-05-3{i}T00-00-00-train"), "failed");
    }
    let listed = run_gc(
        &ws,
        GcInput {
            keep: None,
            keep_successful: false,
            older_than: None,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(listed.len(), 0, "no filters → nothing matched");
    assert_eq!(fs::read_dir(ws.runs_dir()).unwrap().count(), 2);
}
