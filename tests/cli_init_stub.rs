//! `ddrs init` is a hidden stub (removed in 0.4): prints a redirect and
//! exits 2 so muscle-memory scripts fail loudly.

use std::process::Command;

#[test]
fn init_stub_redirects_to_plan_with_exit_2() {
    let out = Command::new(env!("CARGO_BIN_EXE_ddrs"))
        .arg("init")
        .output()
        .expect("ddrs binary should run");
    assert_eq!(out.status.code(), Some(2), "stub must exit 2");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("merged into ddrs plan"),
        "stub must redirect to plan, got: {stderr}"
    );
}

#[test]
fn init_does_not_appear_in_help() {
    let out = Command::new(env!("CARGO_BIN_EXE_ddrs"))
        .arg("--help")
        .output()
        .expect("ddrs binary should run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.contains("init"), "init must be hidden from --help");
}
