//! `ddrs experiment <name>`: paper studies over already-trained runs.
//!
//! A study is a checked-in **bundle** (`experiments/<name>/experiment.yaml`,
//! data only) plus a Rust module under this one (code). Arms are resolved
//! from run ids under `<workspace>/runs/`, never from loose configs, so an
//! experiment always analyzes exactly what a run's `config.yaml` snapshot and
//! checkpoint recorded. Outputs land in
//! `<workspace>/experiments/<name>/<UTC ts>/` with a manifest.
//!
//! Spec: `docs/superpowers/specs/2026-09-03-ddrs-experiment-adjoint-design.md`.

pub mod adjoint;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cli::manifest::GitInfo;
use crate::cli::workspace::Workspace;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// `experiment.yaml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExperimentSpec {
    /// Which study module runs (`adjoint`).
    pub study: String,
    pub arms: Vec<ArmSpec>,
    /// Study-specific block for `study: adjoint`.
    #[serde(default)]
    pub adjoint: Option<adjoint::AdjointSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArmSpec {
    pub name: String,
    /// Run id under `<workspace>/runs/`.
    pub run: String,
}

impl ExperimentSpec {
    pub fn load(bundle_dir: &Path) -> Result<Self, BoxError> {
        let path = bundle_dir.join("experiment.yaml");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let spec: Self =
            serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if spec.arms.is_empty() {
            return Err(format!("{}: `arms` is empty", path.display()).into());
        }
        let mut seen = std::collections::HashSet::new();
        for a in &spec.arms {
            if !seen.insert(a.name.as_str()) {
                return Err(format!("{}: duplicate arm name `{}`", path.display(), a.name).into());
            }
        }
        Ok(spec)
    }
}

/// An arm bound to a concrete run directory, config snapshot, and checkpoint.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedArm {
    pub name: String,
    pub run_id: String,
    pub run_dir: PathBuf,
    pub config_path: PathBuf,
    pub checkpoint_dir: PathBuf,
    /// `epoch_E_mb_M`.
    pub checkpoint_label: String,
}

pub fn resolve_arm(ws: &Workspace, arm: &ArmSpec) -> Result<ResolvedArm, BoxError> {
    let run_dir = ws.runs_dir().join(&arm.run);
    if !run_dir.is_dir() {
        return Err(format!(
            "arm `{}`: run `{}` not found under {}",
            arm.name,
            arm.run,
            ws.runs_dir().display()
        )
        .into());
    }
    let config_path = run_dir.join("config.yaml");
    if !config_path.is_file() {
        return Err(format!(
            "arm `{}`: run `{}` has no config.yaml snapshot",
            arm.name, arm.run
        )
        .into());
    }
    let (checkpoint_label, checkpoint_dir) = latest_checkpoint(&run_dir.join("checkpoints"))
        .map_err(|e| format!("arm `{}` (run `{}`): {e}", arm.name, arm.run))?;
    Ok(ResolvedArm {
        name: arm.name.clone(),
        run_id: arm.run.clone(),
        run_dir,
        config_path,
        checkpoint_dir,
        checkpoint_label,
    })
}

/// Latest `epoch_E_mb_M/` checkpoint directory by `(E, M)`.
///
/// Flat `epoch_E_mb_M.mpk` files are refused: they are written by a stale
/// pre-checkpoint-resume binary (CLAUDE.md, stale-binary trap).
pub fn latest_checkpoint(dir: &Path) -> Result<(String, PathBuf), BoxError> {
    if !dir.is_dir() {
        return Err(format!("no checkpoints directory at {}", dir.display()).into());
    }
    let mut best: Option<(usize, usize, String, PathBuf)> = None;
    let mut flat_seen = false;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let Some((ep, mb)) = parse_checkpoint_label(&name) else { continue };
        if entry.path().is_dir() {
            let better = match &best {
                Some((be, bm, _, _)) => (ep, mb) > (*be, *bm),
                None => true,
            };
            if better {
                best = Some((ep, mb, name.clone(), entry.path()));
            }
        } else if name.ends_with(".mpk") {
            flat_seen = true;
        }
    }
    match best {
        Some((_, _, label, path)) => Ok((label, path)),
        None if flat_seen => Err(format!(
            "{}: only flat epoch_E_mb_M.mpk files found — written by a stale \
             pre-checkpoint-resume binary; retrain with a current `ddrs`",
            dir.display()
        )
        .into()),
        None => Err(format!("{}: no epoch_E_mb_M checkpoints", dir.display()).into()),
    }
}

fn parse_checkpoint_label(name: &str) -> Option<(usize, usize)> {
    let stem = name.strip_suffix(".mpk").unwrap_or(name);
    let rest = stem.strip_prefix("epoch_")?;
    let (e, m) = rest.split_once("_mb_")?;
    Some((e.parse().ok()?, m.parse().ok()?))
}

/// Output directory for one experiment invocation.
pub struct ExperimentRun {
    pub dir: PathBuf,
    pub started_utc: String,
}

impl ExperimentRun {
    pub fn create(ws: &Workspace, name: &str) -> Result<Self, BoxError> {
        let started_utc = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();
        let dir = ws.root().join("experiments").join(name).join(&started_utc);
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, started_utc })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentManifest {
    pub study: String,
    pub started_utc: String,
    pub finished_utc: Option<String>,
    pub status: String,
    pub bundle_dir: PathBuf,
    pub spec: ExperimentSpec,
    pub arms: Vec<ResolvedArm>,
    pub git: GitInfo,
    pub backend: String,
    pub ddrs_version: String,
    /// Study-specific result block (e.g. the adjoint finite-difference gate).
    pub validation: Option<serde_json::Value>,
    pub notes: Vec<String>,
}

impl ExperimentManifest {
    pub fn write(&self, path: &Path) -> Result<(), BoxError> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_checkpoint_labels() {
        assert_eq!(parse_checkpoint_label("epoch_9_mb_1"), Some((9, 1)));
        assert_eq!(parse_checkpoint_label("epoch_25_mb_8.mpk"), Some((25, 8)));
        assert_eq!(parse_checkpoint_label("head.mpk"), None);
        assert_eq!(parse_checkpoint_label("epoch_x_mb_1"), None);
    }

    #[test]
    fn latest_checkpoint_picks_max_epoch_then_minibatch() {
        let tmp = std::env::temp_dir().join(format!("ddrs-exp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        for d in ["epoch_1_mb_9", "epoch_10_mb_0", "epoch_9_mb_30"] {
            std::fs::create_dir_all(tmp.join(d)).unwrap();
        }
        let (label, path) = latest_checkpoint(&tmp).unwrap();
        assert_eq!(label, "epoch_10_mb_0");
        assert_eq!(path, tmp.join("epoch_10_mb_0"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn latest_checkpoint_refuses_flat_mpk() {
        let tmp = std::env::temp_dir().join(format!("ddrs-exp-flat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("epoch_5_mb_9.mpk"), b"x").unwrap();
        let err = latest_checkpoint(&tmp).unwrap_err().to_string();
        assert!(err.contains("stale"), "{err}");
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
