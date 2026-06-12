//! `system::ensure_system_ready` — the former `init` Phase A as a library
//! call: workspace skeleton + GPU probe + cached smoke test.

use ddrs::cli::manifest::SystemProbe;
use ddrs::cli::system::ensure_system_ready;
use ddrs::cli::workspace::Workspace;

#[test]
fn creates_skeleton_and_system_json() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let r = ensure_system_ready(&ws, false, 0.0, true).unwrap();
    assert!(r.smoke_passed, "skip_smoke=true reports passed");
    assert!(ws.root().join("version").is_file());
    assert!(ws.system_json().is_file());
    assert!(ws.runs_dir().is_dir());
}

#[test]
fn second_call_reuses_smoke_verdict() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    // First call runs the real (CPU on CI) smoke and records a verdict.
    let first = ensure_system_ready(&ws, false, 0.0, false).unwrap();
    assert!(first.smoke_passed);
    assert!(!first.smoke_reused, "first run must execute the smoke");
    let passed_at_1 = SystemProbe::read(&ws.system_json())
        .unwrap().smoke_test.expect("system.json should contain a smoke_test record after first run").passed_at;
    // Second call reuses the cached verdict — passed_at unchanged.
    let second = ensure_system_ready(&ws, false, 0.0, false).unwrap();
    assert!(second.smoke_reused, "second run must reuse the cache");
    let passed_at_2 = SystemProbe::read(&ws.system_json())
        .unwrap().smoke_test.expect("system.json should contain a smoke_test record after first run").passed_at;
    assert_eq!(passed_at_1, passed_at_2);
}

#[test]
fn force_reruns_smoke_without_touching_runs_dir() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    ensure_system_ready(&ws, false, 0.0, false).unwrap();
    // Plant a fake run dir; --force must NOT delete it (old init --force
    // nuked the whole workspace — that behavior is dropped).
    let fake_run = ws.runs_dir().join("2026-01-01T00-00-00Z-train");
    std::fs::create_dir_all(&fake_run).unwrap();
    let r = ensure_system_ready(&ws, true, 0.0, false).unwrap();
    assert!(!r.smoke_reused, "force must re-run the smoke");
    assert!(fake_run.is_dir(), "force must never touch .ddrs/runs/");
}
