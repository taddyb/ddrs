use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::cli::CliError;
use crate::cli::fingerprint::Fingerprint;
use crate::cli::types::{RunStatus, Workflow};

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SystemProbe {
    #[serde(default)]
    pub ddrs_version: String,
    #[serde(default)]
    pub probed_at: String,
    #[serde(default)]
    pub gpu: String,
    #[serde(default)]
    pub cuda_runtime: String,
    #[serde(default)]
    pub driver: String,
    #[serde(default)]
    pub sm: String,
    #[serde(default)]
    pub free_gpu_gb_at_probe: f32,
    #[serde(default)]
    pub smoke_test: Option<SmokeTestRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmokeTestRecord {
    pub key: String,
    pub passed_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitInfo {
    pub sha: String,
    pub dirty: bool,
    pub branch: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLockRef {
    pub lockfile: PathBuf,
    pub matched: bool,
    pub drift: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunOutputs {
    pub checkpoints: Vec<PathBuf>,
    pub plot: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub run_id: String,
    pub ddrs_version: String,
    pub git: GitInfo,
    pub workflow: Workflow,
    pub config_path: PathBuf,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub status: RunStatus,
    pub exit_reason: Option<String>,
    pub system: SystemProbe,
    pub sources: BTreeMap<String, Fingerprint>,
    pub source_lock: SourceLockRef,
    pub outputs: RunOutputs,
    pub metrics: serde_json::Value,
    pub max_mini_batches: Option<usize>,
}

// serde_json::Value only implements PartialEq; we assert Eq is safe here
// because the manifest uses JSON values only for metrics (no NaN/Inf floats).
impl Eq for Manifest {}

// f32 field free_gpu_gb_at_probe won't be NaN in practice; assert Eq is safe.
impl Eq for SystemProbe {}

impl Manifest {
    pub fn read(path: &Path) -> Result<Self, CliError> {
        serde_json::from_str(&fs::read_to_string(path)?)
            .map_err(|e| CliError::Other(Box::new(e)))
    }

    pub fn write_atomic(&self, path: &Path) -> Result<(), CliError> {
        let tmp = path.with_extension("json.tmp");
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| CliError::Other(Box::new(e)))?;
        let mut f = fs::File::create(&tmp)?;
        f.write_all(s.as_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)?;
        Ok(())
    }
}

impl SystemProbe {
    pub fn read(path: &Path) -> Result<Self, CliError> {
        serde_json::from_str(&fs::read_to_string(path)?)
            .map_err(|e| CliError::Other(Box::new(e)))
    }

    pub fn write_atomic(&self, path: &Path) -> Result<(), CliError> {
        let tmp = path.with_extension("json.tmp");
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| CliError::Other(Box::new(e)))?;
        let mut f = fs::File::create(&tmp)?;
        f.write_all(s.as_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)?;
        Ok(())
    }
}
