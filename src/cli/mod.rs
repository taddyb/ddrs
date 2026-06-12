//! `ddrs` CLI implementation. Entrypoint lives at `src/bin/ddrs.rs`.

pub mod error;
pub mod fingerprint;
pub mod gc;
pub mod lockfile;
pub mod manifest;
pub mod plan;
pub mod plan_bootstrap;
pub mod run;
pub mod show;
pub mod sources;
pub mod status;
pub mod system;
pub mod tee;
pub mod types;
pub mod workspace;

pub use error::CliError;
pub use types::{ExitCode, RunStatus, Workflow};
