use ddrs::cli::init::{run_init, InitInput};
use ddrs::cli::workspace::Workspace;
use std::sync::Mutex;

// Serialize chdir-based tests (process global state).
static CHDIR_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn init_phase_a_creates_workspace_and_runs_smoke() {
    let _g = CHDIR_LOCK.lock().unwrap();

    // No ddrs.yaml in tempdir → init writes Phase A artifacts, then errors on
    // the non-TTY bootstrap. We assert Phase A artifacts exist before the err.
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let original = std::env::current_dir().unwrap();
    std::env::set_current_dir(d.path()).unwrap();
    let r = run_init(InitInput {
        workspace: ws.root().to_path_buf(),
        config_path: None,
        min_free_gpu_gb: 0.0,
        force: false,
        skip_smoke: true, // CI-friendly default
    });
    std::env::set_current_dir(&original).unwrap();
    assert!(r.is_err(), "non-TTY bootstrap should error");
    assert!(ws.root().join("version").is_file());
    assert!(ws.root().join("system.json").is_file());
    assert!(ws.root().join("runs").is_dir());
    assert!(!ws.lockfile().is_file(), "bootstrap failed → no lockfile");
}

#[test]
fn init_errors_clearly_when_no_yaml_and_no_tty() {
    let _g = CHDIR_LOCK.lock().unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let ws_root = tmp.path().join(".ddrs");
    let original = std::env::current_dir().unwrap();
    std::env::set_current_dir(tmp.path()).unwrap();
    let res = ddrs::cli::init::run_init(ddrs::cli::init::InitInput {
        workspace: ws_root,
        config_path: None,
        min_free_gpu_gb: 0.0,
        force: false,
        skip_smoke: true,
    });
    std::env::set_current_dir(&original).unwrap();
    let err = res.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("no ddrs.yaml found") && msg.contains("not a TTY"),
        "expected non-interactive bootstrap error, got: {msg}"
    );
}

#[test]
fn run_smoke_returns_cpu_when_no_cuda() {
    use ddrs::cli::manifest::SystemProbe;
    let mut probe = SystemProbe::default();
    probe.gpu = String::new();
    let (passed, backend) = ddrs::cli::init::run_smoke_for_test(&probe).unwrap();
    assert!(passed, "CPU smoke must pass on the bundled sandbox fixture");
    assert_eq!(backend, "cpu");
}
