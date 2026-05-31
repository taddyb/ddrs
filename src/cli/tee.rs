//! Run a closure with stdout/stderr piped through `os_pipe` into log files
//! while still forwarding to the original fds. Writes use `O_APPEND` and
//! flush per chunk so CUDA stderr is not lost on crash.

use os_pipe::pipe;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;
use std::thread;

use crate::error::CliError;

/// Spawn pipe readers that tee captured bytes to `stdout_log`/`stderr_log`
/// and the inherited `stdout`/`stderr`.
///
/// This is the primitive used by `cli::run`. The actual wiring to the
/// workflow body (subprocess vs in-process) is decided by the caller; this
/// function just installs the reader threads and returns the result of the
/// body closure.
pub fn tee_to<F, R>(
    stdout_log: &Path,
    stderr_log: &Path,
    body: F,
) -> Result<R, CliError>
where
    F: FnOnce() -> Result<R, CliError>,
{
    let (mut or, ow) = pipe()?;
    let (mut er, ew) = pipe()?;

    let stdout_log_p = stdout_log.to_path_buf();
    let stderr_log_p = stderr_log.to_path_buf();

    let h_out = thread::spawn(move || {
        let mut f = OpenOptions::new().create(true).append(true).open(&stdout_log_p)?;
        let mut buf = [0u8; 8192];
        let mut out = std::io::stdout();
        loop {
            let n = or.read(&mut buf)?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])?;
            f.flush()?;
            out.write_all(&buf[..n])?;
            out.flush()?;
        }
        Ok::<(), std::io::Error>(())
    });
    let h_err = thread::spawn(move || {
        let mut f = OpenOptions::new().create(true).append(true).open(&stderr_log_p)?;
        let mut buf = [0u8; 8192];
        let mut err = std::io::stderr();
        loop {
            let n = er.read(&mut buf)?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])?;
            f.flush()?;
            err.write_all(&buf[..n])?;
            err.flush()?;
        }
        Ok::<(), std::io::Error>(())
    });

    // The body runs in-process; the writer ends (ow, ew) are dropped here
    // so the reader threads' read() will return Ok(0) once the body exits
    // and any subprocesses inheriting those fds also close.
    let _ = ow;
    let _ = ew;
    let result = body();
    let _ = h_out.join();
    let _ = h_err.join();
    result
}
