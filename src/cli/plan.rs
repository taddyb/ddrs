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
    pub resolved_adjacency: ResolvedAdjacency,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline: Option<BaselineInfo>,
}

/// The CONUS + gages adjacency stores `run` will actually open, after
/// resolution. For explicit-path configs these mirror the configured paths and
/// the cache fields are `None`. For fabric-only (managed-build) configs these
/// point at `<workspace_root>/adjacency/<key>/` and the cache fields are set.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedAdjacency {
    pub conus: PathBuf,
    pub gages: PathBuf,
    /// Content-addressed cache key — `Some` only for managed builds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
    /// Whether the managed-build cache was a hit — `Some` only for managed builds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
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
    let mut config = if mode == ConfigMode::Training {
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

    // Step 6: resolve adjacency. Explicit paths → validate every required
    // array exists up front (naming the missing array + store on failure).
    // Fabric-only → cache lookup by content key; hit reuses, miss builds
    // (with a progress line). Same side-effectful-plan precedent as the Q'
    // baseline. The resolved paths are materialized back into the in-memory
    // `config` so every downstream consumer (dataset, baseline, dump) reads
    // them — but `run` snapshots the ORIGINAL config file (`fs::copy` of
    // `config_path`), so the mutation never leaks into the bootstrap source.
    let resolved_adjacency = resolve_adjacency(&config, config_path, workspace)?;
    if let Some(ds) = config.data_sources.as_mut() {
        ds.conus_adjacency = Some(resolved_adjacency.conus.clone());
        ds.gages_adjacency = Some(resolved_adjacency.gages.clone());
    }

    // Step 7: compute summary.
    let summary = compute_summary(&config, workflow)?;

    // Step 8: summed Q' baseline. Always uses the testing-mode overlay
    // (the eval window the trained model is judged against), even when
    // workflow=Train, so the user sees the same reference number across
    // workflows. Failures here are non-fatal — they shouldn't block the
    // plan validation, which is the user's primary signal.
    let baseline = compute_baseline(config_path, workspace, &resolved_adjacency);

    Ok(PlanResult {
        config,
        config_path: config_path.into(),
        workflow,
        sources,
        drift,
        summary,
        resolved_adjacency,
        baseline,
    })
}

/// Resolve the CONUS + gages adjacency stores `run` will open.
///
/// - Both adjacency keys present → validate each required array exists up front
///   (cheap fs checks; the actual open happens downstream as today).
/// - Keys absent (fabric-only) → content-addressed cache lookup; hit reuses,
///   miss builds. `adjacency::cache::resolve_or_build` prints its own progress.
fn resolve_adjacency(
    config: &Config,
    config_path: &Path,
    workspace: &Workspace,
) -> Result<ResolvedAdjacency, CliError> {
    let ds = config.data_sources.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: config_path.into(),
        source: "data_sources: missing".into(),
    })?;

    match (&ds.conus_adjacency, &ds.gages_adjacency) {
        (Some(conus), Some(gages)) => {
            // Explicit paths: validate layout, naming the missing array + store.
            crate::adjacency::validate::validate_conus_store_layout(conus)
                .map_err(|e| CliError::ConfigInvalid {
                    path: config_path.into(),
                    source: Box::new(e),
                })?;
            crate::adjacency::validate::validate_gages_store_layout(gages)
                .map_err(|e| CliError::ConfigInvalid {
                    path: config_path.into(),
                    source: Box::new(e),
                })?;
            Ok(ResolvedAdjacency {
                conus: conus.clone(),
                gages: gages.clone(),
                cache_key: None,
                cache_hit: None,
            })
        }
        // Fabric-only (managed build). `validate_data_sources` (config.rs) has
        // already rejected the partial-adjacency and neither-source cases at
        // load time, so a missing fabric here is an internal invariant break.
        _ => {
            let fabric = ds.geospatial_fabric.as_ref().ok_or_else(|| CliError::ConfigInvalid {
                path: config_path.into(),
                source: "data_sources: no adjacency paths and no geospatial_fabric".into(),
            })?;
            let outcome =
                crate::adjacency::cache::resolve_or_build(workspace.root(), fabric, &ds.gages)
                    .map_err(|e| CliError::ConfigInvalid {
                        path: config_path.into(),
                        source: Box::new(e),
                    })?;
            Ok(ResolvedAdjacency {
                conus: outcome.paths.conus,
                gages: outcome.paths.gages,
                cache_key: Some(outcome.key),
                cache_hit: Some(outcome.cache_hit),
            })
        }
    }
}

fn compute_baseline(
    config_path: &Path,
    workspace: &Workspace,
    resolved: &ResolvedAdjacency,
) -> Option<BaselineInfo> {
    let mut test_cfg = match Config::from_yaml_file_with_mode(config_path, ConfigMode::Testing) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("warning: summed Q' baseline skipped (testing config: {e})");
            return None;
        }
    };
    // The testing config is re-parsed from disk, so it carries the original
    // (possibly absent) adjacency keys. Materialize the resolved paths so the
    // baseline cache key hashes the SAME paths the dataset will open — a
    // managed rebuild under a new key correctly invalidates the baseline.
    if let Some(ds) = test_cfg.data_sources.as_mut() {
        ds.conus_adjacency = Some(resolved.conus.clone());
        ds.gages_adjacency = Some(resolved.gages.clone());
    }
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
