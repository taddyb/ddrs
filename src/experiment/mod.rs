//! `ddrs experiment <name>`: paper studies over already-trained runs.
//!
//! A study is a checked-in **bundle** (`experiments/<name>/experiment.yaml`,
//! data only) plus a Rust module under this one (code). Arms are resolved
//! from run ids under `<workspace>/runs/`, never from loose configs, so an
//! experiment always analyzes exactly what a run's `config.yaml` snapshot and
//! checkpoint recorded. Outputs land in
//! `<workspace>/experiments/<name>/<UTC ts>/` with a manifest.
//!
//! Spec: `research/specs/2026-09-03-ddrs-experiment-adjoint-design.md`.

pub mod adjoint;
pub mod landscape;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cli::manifest::GitInfo;
use crate::cli::workspace::Workspace;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A shard spec `I/K`: this process handles every K-th gauge starting at
/// index `I`, once the population is sorted by staid. Parsed from
/// `--shard I/K`; `K == 0` or `I >= K` is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Shard {
    pub index: usize,
    pub count: usize,
}

impl std::fmt::Display for Shard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.index, self.count)
    }
}

impl std::str::FromStr for Shard {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (i, k) = s
            .split_once('/')
            .ok_or_else(|| format!("--shard must be I/K (e.g. 0/24), got `{s}`"))?;
        let index: usize = i
            .parse()
            .map_err(|_| format!("--shard index `{i}` is not a non-negative integer"))?;
        let count: usize = k
            .parse()
            .map_err(|_| format!("--shard count `{k}` is not a non-negative integer"))?;
        if count == 0 {
            return Err("--shard count K must be >= 1".into());
        }
        if index >= count {
            return Err(format!("--shard index {index} must be < count {count}"));
        }
        Ok(Shard { index, count })
    }
}

/// Sort `gauges` by staid, then keep only the entries at sorted-index `i`
/// where `i % shard.count == shard.index`. A no-op (order untouched) when
/// `shard` is `None`, so bundles that never pass `--shard` see no behavior
/// change.
pub fn shard_gauges(gauges: Vec<adjoint::gauges::GaugeEntry>, shard: Option<Shard>) -> Vec<adjoint::gauges::GaugeEntry> {
    match shard {
        None => gauges,
        Some(s) => {
            let mut sorted = gauges;
            sorted.sort_by(|a, b| a.staid.cmp(&b.staid));
            sorted.into_iter().enumerate().filter(|(i, _)| i % s.count == s.index).map(|(_, g)| g).collect()
        }
    }
}

/// `experiment.yaml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExperimentSpec {
    /// Which study module runs (`adjoint`).
    pub study: String,
    pub arms: Vec<ArmSpec>,
    /// Study-specific block for `study: adjoint`.
    #[serde(default)]
    pub adjoint: Option<adjoint::AdjointSpec>,
    /// Study-specific block for `study: landscape`.
    #[serde(default)]
    pub landscape: Option<landscape::LandscapeSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArmSpec {
    pub name: String,
    /// Run id under `<workspace>/runs/`.
    pub run: String,
    /// Checkpoint label override (`epoch_E_mb_M`, or `init` for the
    /// pre-training head). Defaults to the run's latest checkpoint.
    #[serde(default)]
    pub checkpoint: Option<String>,
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
    let checkpoints_dir = run_dir.join("checkpoints");
    let (checkpoint_label, checkpoint_dir) = match &arm.checkpoint {
        Some(label) if label == "init" => ("init".to_string(), checkpoints_dir),
        Some(label) => {
            let dir = checkpoints_dir.join(label);
            if !dir.is_dir() {
                return Err(format!(
                    "arm `{}` (run `{}`): checkpoint `{}` not found under {}",
                    arm.name,
                    arm.run,
                    label,
                    checkpoints_dir.display()
                )
                .into());
            }
            (label.clone(), dir)
        }
        None => latest_checkpoint(&checkpoints_dir)
            .map_err(|e| format!("arm `{}` (run `{}`): {e}", arm.name, arm.run))?,
    };
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
    /// `shard` is folded into the run directory's name only
    /// (`<ts>-shard-I-of-K`), not `started_utc`, so K shards launched in the
    /// same second land in distinct directories without colliding.
    pub fn create(ws: &Workspace, name: &str, shard: Option<Shard>) -> Result<Self, BoxError> {
        let started_utc = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();
        let dir_name = match shard {
            Some(s) => format!("{started_utc}-shard-{}-of-{}", s.index, s.count),
            None => started_utc.clone(),
        };
        let dir = ws.root().join("experiments").join(name).join(&dir_name);
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
    /// `"I/K"` when `--shard I/K` was passed, else `None`.
    pub shard: Option<String>,
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
    fn resolve_arm_explicit_checkpoint_label_wins() {
        let tmp = std::env::temp_dir().join(format!("ddrs-exp-resolve-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let run_dir = tmp.join("runs").join("run-a");
        std::fs::create_dir_all(run_dir.join("checkpoints").join("epoch_1_mb_0")).unwrap();
        std::fs::create_dir_all(run_dir.join("checkpoints").join("epoch_30_mb_1")).unwrap();
        std::fs::write(run_dir.join("config.yaml"), "x: 1\n").unwrap();
        let ws = Workspace::with_root(tmp.clone());

        // Explicit label wins over the latest checkpoint.
        let arm = ArmSpec { name: "a".into(), run: "run-a".into(), checkpoint: Some("epoch_1_mb_0".into()) };
        let resolved = resolve_arm(&ws, &arm).unwrap();
        assert_eq!(resolved.checkpoint_label, "epoch_1_mb_0");
        assert_eq!(resolved.checkpoint_dir, run_dir.join("checkpoints").join("epoch_1_mb_0"));

        // A missing label errors.
        let missing = ArmSpec { name: "a".into(), run: "run-a".into(), checkpoint: Some("epoch_99_mb_0".into()) };
        let err = resolve_arm(&ws, &missing).unwrap_err().to_string();
        assert!(err.contains("epoch_99_mb_0"), "{err}");

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

    #[test]
    fn shard_parses_valid_and_rejects_bad_input() {
        assert_eq!("0/24".parse::<Shard>().unwrap(), Shard { index: 0, count: 24 });
        assert_eq!("23/24".parse::<Shard>().unwrap(), Shard { index: 23, count: 24 });
        assert!("24/24".parse::<Shard>().is_err(), "index must be < count");
        assert!("0/0".parse::<Shard>().is_err(), "count must be >= 1");
        assert!("x/3".parse::<Shard>().is_err());
        assert!("1".parse::<Shard>().is_err());
    }

    fn entry(staid: &str) -> adjoint::gauges::GaugeEntry {
        adjoint::gauges::GaugeEntry { staid: staid.into(), role: "gauge", pair: 0, upstream: vec![], comid: None, class: None }
    }

    #[test]
    fn shard_gauges_keeps_index_mod_k_after_sorting_by_staid() {
        // Unsorted input; sorted-by-staid order is 01,02,03,04,05.
        let gauges = vec![entry("03"), entry("01"), entry("05"), entry("02"), entry("04")];
        let shard0 = shard_gauges(gauges.clone(), Some(Shard { index: 0, count: 2 }));
        assert_eq!(shard0.iter().map(|g| g.staid.as_str()).collect::<Vec<_>>(), vec!["01", "03", "05"]);
        let shard1 = shard_gauges(gauges.clone(), Some(Shard { index: 1, count: 2 }));
        assert_eq!(shard1.iter().map(|g| g.staid.as_str()).collect::<Vec<_>>(), vec!["02", "04"]);
        // Every gauge lands in exactly one shard.
        let mut all: Vec<String> = shard0.into_iter().chain(shard1).map(|g| g.staid).collect();
        all.sort();
        assert_eq!(all, vec!["01", "02", "03", "04", "05"]);
    }

    #[test]
    fn shard_gauges_is_noop_when_shard_is_none() {
        let gauges = vec![entry("03"), entry("01"), entry("02")];
        let out = shard_gauges(gauges.clone(), None);
        assert_eq!(out.iter().map(|g| g.staid.as_str()).collect::<Vec<_>>(), vec!["03", "01", "02"]);
    }
}
