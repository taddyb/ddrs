use serde::{Deserialize, Serialize};

pub use crate::config::Workflow;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunStatus {
    Ok,
    Failed,
    Interrupted,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Generic = 1,
    ConfigInvalid = 2,
    DataSourceMissing = 3,
    LockDrift = 4,
    RuntimeFailure = 5,
    WorkspaceNotInitialized = 6,
}

impl ExitCode {
    pub fn exit(self) -> ! {
        std::process::exit(self as i32);
    }
}

impl From<&crate::cli::CliError> for ExitCode {
    fn from(e: &crate::cli::CliError) -> Self {
        use crate::cli::CliError::*;
        match e {
            ConfigInvalid { .. } => ExitCode::ConfigInvalid,
            DataSourceMissing { .. } => ExitCode::DataSourceMissing,
            LockDrift { .. } => ExitCode::LockDrift,
            Runtime(_) => ExitCode::RuntimeFailure,
            WorkspaceNotInitialized { .. } => ExitCode::WorkspaceNotInitialized,
            Io(_) | Other(_) => ExitCode::Generic,
        }
    }
}
