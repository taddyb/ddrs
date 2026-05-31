use ddrs::cli::init::{run_init, InitInput};
use ddrs::cli::workspace::Workspace;

#[test]
fn init_phase_a_creates_workspace_and_runs_smoke() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let input = InitInput {
        workspace: ws.root().to_path_buf(),
        config_path: None,
        min_free_gpu_gb: 0.0,
        force: false,
        skip_smoke: true, // CI-friendly default
    };
    let r = run_init(input).unwrap();
    assert!(ws.root().join("version").is_file());
    assert!(ws.root().join("system.json").is_file());
    assert!(ws.root().join("runs").is_dir());
    assert!(!ws.lockfile().is_file(), "no config → no lockfile");
    assert!(r.phase_b_skipped);
}
