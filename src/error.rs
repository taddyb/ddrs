//! CLI error type that maps cleanly onto `ExitCode`.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("config invalid at {path}: {source}")]
    ConfigInvalid { path: PathBuf, #[source] source: Box<dyn std::error::Error + Send + Sync> },

    #[error("data source unreachable: {path}")]
    DataSourceMissing { path: PathBuf },

    #[error("lock drift in --strict mode: {fields:?}")]
    LockDrift { fields: Vec<String> },

    #[error("runtime failure during workflow: {0}")]
    Runtime(String),

    #[error("workspace not initialized at {path}; run `ddrs init`")]
    WorkspaceNotInitialized { path: PathBuf },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
