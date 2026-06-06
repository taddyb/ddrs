//! Dry-run validation. Returns a PlanResult that `run` consumes directly
//! to avoid duplicated I/O.

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::cli::fingerprint::{Fingerprint, fingerprint_path, reuse_if_unchanged};
use crate::cli::lockfile::{Lockfile, diff_against_live};
use crate::cli::types::Workflow;
use crate::cli::workspace::Workspace;
use crate::config::{Config, ConfigMode};
use crate::error::CliError;
use crate::training::metrics::Metrics;

#[derive(Debug, Clone, Serialize)]
pub struct PlanResult {
    #[serde(skip)]
    pub config: Config,
    pub config_path: PathBuf,
    pub workflow: Workflow,
    pub sources: BTreeMap<String, Fingerprint>,
    pub drift: Vec<String>,
    pub summary: PlanSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline: Option<BaselineInfo>,
}

/// Summed Q' baseline result attached to `PlanResult`. The full `Metrics`
/// vector is held in-memory but skipped from JSON output (NaN handling +
/// size); the JSON view exposes only the small identifying triple.
#[derive(Debug, Clone, Serialize)]
pub struct BaselineInfo {
    pub key: String,
    pub cache_hit: bool,
    pub n_gauges: usize,
    pub cache_dir: PathBuf,
    #[serde(skip)]
    pub metrics: Metrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanSummary {
    pub workflow: Workflow,
    pub n_gauges: Option<usize>,
    pub batches_per_epoch: Option<usize>,
    pub epochs: Option<usize>,
    pub est_timesteps: Option<usize>,
    pub n_days: Option<usize>,
    pub chunks: Option<usize>,
    pub gpu_mem_gb_upper_bound: Option<f32>,
}

pub fn plan(
    config_path: &Path,
    workflow_override: Option<Workflow>,
    workspace: &Workspace,
) -> Result<PlanResult, CliError> {
    // Step 1: load config once (preview in Training mode — we'll re-parse
    // if the resolved workflow says testing).
    let preview = Config::from_yaml_file_with_mode(config_path, ConfigMode::Training)
        .map_err(|e| CliError::ConfigInvalid {
            path: config_path.into(),
            source: Box::new(e),
        })?;

    // Step 2: resolve workflow — CLI flag wins, then YAML, then error.
    let workflow = workflow_override.or(preview.workflow).ok_or_else(|| {
        CliError::ConfigInvalid {
            path: config_path.into(),
            source: format!(
                "no `workflow:` key in {}. Add `workflow: train-and-test` \
                 (or `train` / `eval`), or pass `--workflow <name>`.",
                config_path.display()
            ).into(),
        }
    })?;

    // Step 3: re-parse if the resolved workflow needs Testing overlay.
    let mode = match workflow {
        Workflow::Train | Workflow::TrainAndTest => ConfigMode::Training,
        Workflow::Eval => ConfigMode::Testing,
    };
    let config = if mode == ConfigMode::Training {
        preview
    } else {
        Config::from_yaml_file_with_mode(config_path, mode)
            .map_err(|e| CliError::ConfigInvalid {
                path: config_path.into(),
                source: Box::new(e),
            })?
    };

    // Step 4: read lockfile (required; init must have produced it).
    let lock_path = workspace.lockfile();
    if !lock_path.is_file() {
        return Err(CliError::WorkspaceNotInitialized { path: workspace.root().into() });
    }
    let lock = Lockfile::read(&lock_path)?;

    // Step 5: compute live fingerprints (reusing locked fp when stat matches).
    let data_sources = config.data_sources.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: config_path.into(),
        source: "data_sources: missing".into(),
    })?;
    // Optional adjacency keys are only fingerprinted when explicitly configured.
    let mut pairs: Vec<(String, PathBuf)> = vec![
        ("attributes".into(),   data_sources.attributes.clone()),
        ("streamflow".into(),   data_sources.streamflow.clone()),
        ("observations".into(), data_sources.observations.clone()),
        ("gages".into(),        data_sources.gages.clone()),
    ];
    if let Some(p) = &data_sources.conus_adjacency {
        pairs.push(("conus_adjacency".into(), p.clone()));
    }
    if let Some(p) = &data_sources.gages_adjacency {
        pairs.push(("gages_adjacency".into(), p.clone()));
    }
    if let Some(p) = &data_sources.geospatial_fabric {
        pairs.push(("geospatial_fabric".into(), p.clone()));
    }
    let mut sources = BTreeMap::new();
    for (key, path) in pairs {
        let live = match lock.sources.get(&key) {
            Some(locked) => {
                let r = reuse_if_unchanged(&path, locked)?;
                Fingerprint {
                    path: path.clone(),
                    mtime: r.mtime,
                    size: r.size,
                    fp: r.fp,
                }
            }
            None => fingerprint_path(&path)?,
        };
        sources.insert(key, live);
    }
    let drift = diff_against_live(&lock, &sources);

    // Step 6: zarr/icechunk metadata-only validation. v1 stub — the
    // existing ConusAdjacencyStore::open is the only "open" that's both
    // cheap and present today. The time-window check against the
    // streamflow store is deferred to a follow-up; not a blocker for
    // shipping plan validation today (the integration test in Task 21
    // exercises the lockfile + fingerprint paths, which are the bulk of
    // plan's value).

    // Step 7: compute summary.
    let summary = compute_summary(&config, workflow)?;

    // Step 8: summed Q' baseline. Always uses the testing-mode overlay
    // (the eval window the trained model is judged against), even when
    // workflow=Train, so the user sees the same reference number across
    // workflows. Failures here are non-fatal — they shouldn't block the
    // plan validation, which is the user's primary signal.
    let baseline = compute_baseline(config_path, workspace);

    Ok(PlanResult {
        config,
        config_path: config_path.into(),
        workflow,
        sources,
        drift,
        summary,
        baseline,
    })
}

fn compute_baseline(config_path: &Path, workspace: &Workspace) -> Option<BaselineInfo> {
    let test_cfg = match Config::from_yaml_file_with_mode(config_path, ConfigMode::Testing) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("warning: summed Q' baseline skipped (testing config: {e})");
            return None;
        }
    };
    match crate::baseline::compute_or_load_cached(&test_cfg, workspace.root()) {
        Ok((q, key, cache_hit)) => Some(BaselineInfo {
            cache_dir: crate::baseline::cache_dir(workspace.root(), &key),
            key,
            cache_hit,
            n_gauges: q.gage_ids.len(),
            metrics: q.metrics,
        }),
        Err(e) => {
            eprintln!("warning: summed Q' baseline failed: {e}");
            None
        }
    }
}

fn compute_summary(cfg: &Config, workflow: Workflow) -> Result<PlanSummary, CliError> {
    let exp = cfg.experiment.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: "experiment".into(),
        source: "experiment: missing".into(),
    })?;
    // n_gauges + max_subgraph_size require opening adjacency stores; left
    // as Option<None> for v1. Tasks 16/22 can tighten by counting gauges
    // from the gages CSV.
    let n_gauges: Option<usize> = None;
    let batches_per_epoch = n_gauges.map(|n| n.div_ceil(exp.batch_size));
    let rho = exp.rho.unwrap_or(0);
    let est_timesteps = batches_per_epoch.map(|b| rho * b * exp.epochs);
    let gpu_mem_gb_upper_bound: Option<f32> = None;

    Ok(PlanSummary {
        workflow,
        n_gauges,
        batches_per_epoch,
        epochs: Some(exp.epochs),
        est_timesteps,
        n_days: None,
        chunks: None,
        gpu_mem_gb_upper_bound,
    })
}
