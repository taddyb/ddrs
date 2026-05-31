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

pub struct InitOutput {
    pub smoke_passed: bool,
    pub smoke_reused: bool,
    pub phase_b_skipped: bool,
}

pub fn run_init(input: InitInput) -> Result<InitOutput, CliError> {
    if input.force && input.workspace.exists() {
        fs::remove_dir_all(&input.workspace)?;
    }
    let ws = Workspace::with_root(&input.workspace);

    // ── Phase A: install-level checks (no config required) ─────────────
    let mut probe = system::probe()?.unwrap_or_default();
    if probe.gpu.is_empty() {
        eprintln!(
            "warning: no CUDA device found; install nvidia driver \
             ≥ 530 or build with `--features cpu`"
        );
    }
    if probe.free_gpu_gb_at_probe < input.min_free_gpu_gb && probe.free_gpu_gb_at_probe > 0.0 {
        eprintln!(
            "warning: free GPU memory {:.1} GB is below floor {} GB",
            probe.free_gpu_gb_at_probe, input.min_free_gpu_gb
        );
    }
    fs::create_dir_all(ws.runs_dir())?;
    fs::write(ws.version_file(), env!("CARGO_PKG_VERSION"))?;

    let key = system::smoke_key(&probe);
    let cached_passing = SystemProbe::read(&ws.system_json())
        .ok()
        .and_then(|p| p.smoke_test)
        .map(|s| s.key == key)
        .unwrap_or(false);
    let (smoke_passed, smoke_reused) = if input.skip_smoke {
        (true, true)
    } else if cached_passing && !input.force {
        (true, true)
    } else {
        (run_smoke()?, false)
    };
    if smoke_passed && !smoke_reused {
        system::record_smoke(&mut probe, key);
    } else if smoke_reused {
        // Preserve the prior smoke_test record.
        if let Ok(prior) = SystemProbe::read(&ws.system_json()) {
            probe.smoke_test = prior.smoke_test;
        }
    }
    probe.write_atomic(&ws.system_json())?;

    // ── Phase B: data-source lock (requires ddrs.yaml) ─────────────────
    let config_path = input.config_path.or_else(|| {
        crate::cli::workspace::discover_config(Path::new("."))
    });
    let Some(cfg_path) = config_path else {
        eprintln!(
            "no ddrs.yaml found — run `ddrs plan` to bootstrap one, \
             then re-run `ddrs init` to lock data sources."
        );
        return Ok(InitOutput { smoke_passed, smoke_reused, phase_b_skipped: true });
    };
    let cfg = Config::from_yaml_file_with_mode(&cfg_path, ConfigMode::Training)
        .map_err(|e| CliError::ConfigInvalid { path: cfg_path.clone(), source: Box::new(e) })?;
    let ds = cfg.data_sources.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: cfg_path.clone(),
        source: Box::<dyn std::error::Error + Send + Sync>::from("data_sources: missing"),
    })?;

    let pairs = [
        ("attributes", ds.attributes.clone()),
        ("conus_adjacency", ds.conus_adjacency.clone()),
        ("gages_adjacency", ds.gages_adjacency.clone()),
        ("streamflow", ds.streamflow.clone()),
        ("observations", ds.observations.clone()),
        ("gages", ds.gages.clone()),
    ];
    // Parallel reachability + fingerprint. std::thread::scope is fine — these
    // are I/O-bound and the count is small.
    let results: Vec<(String, Result<_, CliError>)> = std::thread::scope(|s| {
        let handles: Vec<_> = pairs
            .iter()
            .map(|(k, p)| {
                let p = p.clone();
                let k = k.to_string();
                s.spawn(move || (k, fingerprint_path(&p)))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
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
    Ok(InitOutput { smoke_passed, smoke_reused, phase_b_skipped: false })
}

fn run_smoke() -> Result<bool, CliError> {
    let inputs = crate::sandbox::load_embedded()
        .or_else(|_| crate::sandbox::load_from_dir(Path::new("fixtures/sandbox")))?;
    type I = burn_cuda::Cuda<f32, i32>;
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let r = crate::sandbox::smoke::<I>(&inputs, &device)?;
    Ok(r.passed)
}
