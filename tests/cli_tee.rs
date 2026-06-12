//! `cli::tee::tee_to` — fd-level capture with per-line UTC timestamps.
//!
//! Writes go through `io::stdout()`/a child process rather than `println!`
//! because the libtest harness captures the print macros at the std level;
//! the tee operates below that, on fds 1/2.

use std::io::Write;
use std::sync::Mutex;

use ddrs::cli::tee::tee_to;

/// Both tests rewire the process-global fds 1/2; libtest runs tests on
/// parallel threads, so serialize them.
static FD_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn tee_stamps_and_captures_both_streams_and_children() {
    let _g = FD_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("run.log");

    let r = tee_to(&log, || {
        writeln!(std::io::stdout(), "hello stdout").unwrap();
        writeln!(std::io::stderr(), "hello stderr").unwrap();
        // Child processes inherit fds 1/2 — their output must land too.
        std::process::Command::new("sh")
            .args(["-c", "echo child out; echo child err >&2"])
            .status()
            .unwrap();
        Ok(42)
    })
    .unwrap();
    assert_eq!(r, 42);

    let text = std::fs::read_to_string(&log).unwrap();
    for needle in ["hello stdout", "hello stderr", "child out", "child err"] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }
    // Every line carries a `[<RFC3339 UTC>] ` prefix.
    for line in text.lines() {
        assert!(
            line.starts_with('[') && line.contains("Z] "),
            "unstamped line: {line:?}"
        );
    }
}

#[test]
fn tee_restores_fds_and_propagates_errors() {
    let _g = FD_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("run.log");

    let err = tee_to::<_, ()>(&log, || {
        writeln!(std::io::stdout(), "before failure").unwrap();
        Err(ddrs::cli::CliError::Runtime("boom".into()))
    })
    .unwrap_err();
    assert!(format!("{err}").contains("boom"));

    // Output captured up to the failure, and fds work again afterwards.
    let text = std::fs::read_to_string(&log).unwrap();
    assert!(text.contains("before failure"));
    writeln!(std::io::stdout(), "fd 1 alive after tee").unwrap();
}
