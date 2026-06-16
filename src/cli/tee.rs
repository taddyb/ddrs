//! Tee the process's stdout/stderr into a per-run log file with per-line
//! UTC timestamps, while still forwarding to the terminal.
//!
//! Works at the fd level (`dup2`) so everything the run prints — Rust
//! `println!`/`eprintln!`, CUDA driver messages on stderr, any child
//! process inheriting fds 1/2 — lands in the log. Each complete line is
//! stamped `[2026-06-12T08:15:42Z] ...` on both the terminal and the file,
//! and the file is opened `O_APPEND` + flushed per line so output is not
//! lost on crash.

use std::fs::File;
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use os_pipe::pipe;

use crate::error::CliError;

fn stamp() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Drain one pipe: split into lines, stamp each, write to the shared log
/// file and the saved terminal fd. Partial trailing output (no newline at
/// EOF) is flushed with a stamp so crashes don't swallow the last line.
fn pump(mut src: os_pipe::PipeReader, log: Arc<Mutex<File>>, mut term: File) {
    use std::io::Read;
    let mut buf = [0u8; 8192];
    let mut pending: Vec<u8> = Vec::new();
    let emit = |line: &[u8], term: &mut File| {
        let prefix = format!("[{}] ", stamp());
        if let Ok(mut f) = log.lock() {
            let _ = f.write_all(prefix.as_bytes());
            let _ = f.write_all(line);
            let _ = f.flush();
        }
        let _ = term.write_all(prefix.as_bytes());
        let _ = term.write_all(line);
        let _ = term.flush();
    };
    loop {
        let n = match src.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        pending.extend_from_slice(&buf[..n]);
        while let Some(nl) = pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = pending.drain(..=nl).collect();
            emit(&line, &mut term);
        }
    }
    if !pending.is_empty() {
        pending.push(b'\n');
        emit(&pending, &mut term);
    }
}

/// Run `body` with fds 1/2 redirected through timestamping tee threads into
/// `log_path`. The original fds are restored before returning, error or not.
pub fn tee_to<F, R>(log_path: &Path, body: F) -> Result<R, CliError>
where
    F: FnOnce() -> Result<R, CliError>,
{
    let log = Arc::new(Mutex::new(
        std::fs::OpenOptions::new().create(true).append(true).open(log_path)?,
    ));

    let (out_r, out_w) = pipe()?;
    let (err_r, err_w) = pipe()?;

    // Flush Rust-side buffers, then save the real terminal fds and point
    // 1/2 at the pipe writers.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    let saved = unsafe {
        let o = libc::dup(1);
        let e = libc::dup(2);
        if o < 0 || e < 0 {
            return Err(CliError::Runtime("tee: dup(1/2) failed".into()));
        }
        if libc::dup2(out_w.as_raw_fd(), 1) < 0 || libc::dup2(err_w.as_raw_fd(), 2) < 0 {
            libc::close(o);
            libc::close(e);
            return Err(CliError::Runtime("tee: dup2 onto stdout/stderr failed".into()));
        }
        (o, e)
    };
    // The reader threads forward to the SAVED fds, not 1/2 (now pipes).
    let term_out = unsafe { File::from_raw_fd(libc::dup(saved.0)) };
    let term_err = unsafe { File::from_raw_fd(libc::dup(saved.1)) };

    let h_out = thread::spawn({
        let log = Arc::clone(&log);
        move || pump(out_r, log, term_out)
    });
    let h_err = thread::spawn({
        let log = Arc::clone(&log);
        move || pump(err_r, log, term_err)
    });

    let result = body();

    // Restore fds 1/2 (this also closes their pipe-writer duplicates), then
    // drop the original writer ends so the pumps see EOF and drain fully.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    unsafe {
        libc::dup2(saved.0, 1);
        libc::dup2(saved.1, 2);
        libc::close(saved.0);
        libc::close(saved.1);
    }
    drop(out_w);
    drop(err_w);
    let _ = h_out.join();
    let _ = h_err.join();

    result
}
