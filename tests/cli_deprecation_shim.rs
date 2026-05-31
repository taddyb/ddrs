//! Verify the deprecated binaries print a deprecation warning to stderr.
//! The shim is at the top of `main()` so any invocation triggers it,
//! including `--help`.

use std::process::Command;

#[test]
fn train_binary_prints_deprecation_warning_to_stderr() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_train"));
    cmd.args(["--help"]);
    let out = cmd.output().expect("train binary should be built");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("deprecated"),
        "expected `deprecated` in stderr, got: {stderr}",
    );
}

#[test]
fn eval_binary_prints_deprecation_warning_to_stderr() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_eval"));
    cmd.args(["--help"]);
    let out = cmd.output().expect("eval binary should be built");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("deprecated"),
        "expected `deprecated` in stderr, got: {stderr}",
    );
}

#[test]
fn train_and_test_binary_prints_deprecation_warning_to_stderr() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_train_and_test"));
    cmd.args(["--help"]);
    let out = cmd.output().expect("train_and_test binary should be built");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("deprecated"),
        "expected `deprecated` in stderr, got: {stderr}",
    );
}
