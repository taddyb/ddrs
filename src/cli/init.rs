use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::cli::{
    fingerprint::fingerprint_path,
    lockfile::Lockfile,
    manifest::SystemProbe,
    system,
    workspace::Workspace,
};
use crate::config::{Config, ConfigMode};
use crate::error::CliError;

pub struct InitInput {
    pub workspace: PathBuf,
    pub config_path: Option<PathBuf>,
    pub min_free_gpu_gb: f32,
    pub force: bool,
    pub skip_smoke: bool,
}

#[derive(Debug)]
pub struct InitOutput {
    pub smoke_passed: bool,
    pub smoke_reused: bool,
}

pub fn run_init(input: InitInput) -> Result<InitOutput, CliError> {
    if input.force && input.workspace.exists() {
        if input.workspace.file_name().and_then(|n| n.to_str()) != Some(".ddrs") {
            return Err(CliError::Other(
                format!(
                    "refusing to --force-remove workspace {:?}: directory name must be `.ddrs` for safety",
                    input.workspace,
                )
                .into(),
            ));
        }
        fs::remove_dir_all(&input.workspace)?;
    }
    let ws = Workspace::with_root(&input.workspace);

    // ── Phase A: install-level probes (no config required) ─────────────
    let mut probe = system::probe()?.unwrap_or_default();
    if probe.free_gpu_gb_at_probe < input.min_free_gpu_gb && probe.free_gpu_gb_at_probe > 0.0 {
        eprintln!(
            "warning: free GPU memory {:.1} GB is below floor {} GB",
            probe.free_gpu_gb_at_probe, input.min_free_gpu_gb
        );
    }
    fs::create_dir_all(ws.runs_dir())?;
    fs::write(ws.version_file(), env!("CARGO_PKG_VERSION"))?;

    // Pick backend up-front so the cache key matches the work we'd do.
    let backend = if probe.gpu.is_empty() { "cpu" } else { "cuda" };
    let key = system::smoke_key(&probe, backend);
    let cached_passing = SystemProbe::read(&ws.system_json())
        .ok()
        .and_then(|p| p.smoke_test)
        .map(|s| s.key == key)
        .unwrap_or(false);
    let (smoke_passed, smoke_reused) = if input.skip_smoke {
        // Don't claim "reused" if there's no prior record — just "passed".
        (true, cached_passing)
    } else if cached_passing && !input.force {
        (true, true)
    } else {
        let (ok, _b) = run_smoke(&probe)?;
        (ok, false)
    };
    if smoke_passed && !smoke_reused {
        system::record_smoke(&mut probe, key, backend);
    } else if smoke_reused {
        // Preserve the prior smoke_test record.
        if let Ok(prior) = SystemProbe::read(&ws.system_json()) {
            probe.smoke_test = prior.smoke_test;
        }
    }
    probe.write_atomic(&ws.system_json())?;

    // ── Phase D: bootstrap ddrs.yaml if missing (interactive) ─────────
    let config_path = input.config_path.or_else(|| {
        crate::cli::workspace::discover_config(Path::new("."))
    });
    let cfg_path = match config_path {
        Some(p) => p,
        None => {
            let target = std::env::current_dir()
                .map_err(CliError::from)?
                .join("ddrs.yaml");
            let bundled = PathBuf::from("config/merit_training.yaml");
            crate::cli::plan_bootstrap::bootstrap(
                crate::cli::plan_bootstrap::BootstrapInput {
                    target: target.clone(),
                    runs_dir: ws.runs_dir(),
                    bundled_template: bundled,
                    editor_cmd: None,
                    interactive: true,
                },
            ).map_err(|e| {
                let msg = format!("{e}");
                if msg.contains("not a TTY") || msg.contains("run interactively") {
                    CliError::ConfigInvalid {
                        path: target.clone(),
                        source: "no ddrs.yaml found and stdin is not a TTY. \
                                 Write ddrs.yaml manually, then re-run `ddrs init`."
                                .into(),
                    }
                } else {
                    e
                }
            })?;
            target
        }
    };

    // ── Phase E: lock data sources from the (now-present) yaml ─────────
    let cfg = Config::from_yaml_file_with_mode(&cfg_path, ConfigMode::Training)
        .map_err(|e| CliError::ConfigInvalid { path: cfg_path.clone(), source: Box::new(e) })?;
    let ds = cfg.data_sources.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: cfg_path.clone(),
        source: Box::<dyn std::error::Error + Send + Sync>::from("data_sources: missing"),
    })?;

    // Build the fingerprint pairs. Optional keys (adjacency zarr stores,
    // geospatial_fabric) are only locked when explicitly configured.
    let mut pairs: Vec<(&str, PathBuf)> = vec![
        ("attributes", ds.attributes.clone()),
        ("streamflow", ds.streamflow.clone()),
        ("observations", ds.observations.clone()),
        ("gages", ds.gages.clone()),
    ];
    if let Some(p) = &ds.conus_adjacency {
        pairs.push(("conus_adjacency", p.clone()));
    }
    if let Some(p) = &ds.gages_adjacency {
        pairs.push(("gages_adjacency", p.clone()));
    }
    if let Some(p) = &ds.geospatial_fabric {
        pairs.push(("geospatial_fabric", p.clone()));
    }

    // Parallel reachability + fingerprint. std::thread::scope is fine — these
    // are I/O-bound and the count is small.
    let results: Result<Vec<(String, Result<_, CliError>)>, CliError> = std::thread::scope(|s| {
        let handles: Vec<_> = pairs
            .iter()
            .map(|(k, p)| {
                let p = p.clone();
                let k = k.to_string();
                s.spawn(move || (k, fingerprint_path(&p)))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| CliError::Runtime("fingerprint thread panicked".into()))
            })
            .collect()
    });
    let results = results?;
    let mut sources = BTreeMap::new();
    for (k, r) in results {
        sources.insert(k, r?);
    }
    let lock = Lockfile {
        ddrs_version: env!("CARGO_PKG_VERSION").into(),
        created_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        sources,
    };
    lock.write_atomic(&ws.lockfile())?;
    Ok(InitOutput { smoke_passed, smoke_reused })
}

fn run_smoke(probe: &crate::cli::manifest::SystemProbe)
    -> Result<(bool, &'static str), CliError>
{
    let inputs = crate::sandbox::load_embedded()
        .or_else(|_| crate::sandbox::load_from_dir(Path::new("fixtures/sandbox")))?;
    if probe.gpu.is_empty() {
        eprintln!("no CUDA detected — running CPU smoke (slower but functionally equivalent)");
        type I = burn::backend::NdArray<f32>;
        let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
        let r = crate::sandbox::smoke::<I>(&inputs, &device)?;
        Ok((r.passed, "cpu"))
    } else {
        type I = burn_cuda::Cuda<f32, i32>;
        let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
        let r = crate::sandbox::smoke::<I>(&inputs, &device)?;
        Ok((r.passed, "cuda"))
    }
}

/// Test-only re-export so integration tests can drive the backend selection.
#[doc(hidden)]
pub fn run_smoke_for_test(probe: &crate::cli::manifest::SystemProbe)
    -> Result<(bool, &'static str), CliError>
{
    run_smoke(probe)
}
