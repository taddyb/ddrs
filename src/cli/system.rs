use std::fs;
use std::path::Path;

use chrono::SecondsFormat;
use cudarc::driver::{result, CudaContext};

use crate::cli::CliError;
use crate::cli::manifest::{SmokeTestRecord, SystemProbe};
use crate::cli::workspace::Workspace;

/// In-process GPU probe via cudarc. Returns `Ok(None)` when no CUDA device
/// is present (so the caller can present a remediation hint).
pub fn probe() -> Result<Option<SystemProbe>, CliError> {
    // 1. Try cudarc init. On failure → Ok(None).
    if result::init().is_err() {
        return Ok(None);
    }

    // 2. Open CudaContext for device 0. On failure → Ok(None).
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(_) => return Ok(None),
    };

    // 3. Device name.
    let gpu = ctx.name().unwrap_or_default();

    // 4. Compute capability → "sm = M.m".
    let sm = ctx
        .compute_capability()
        .map(|(major, minor)| format!("{}.{}", major, minor))
        .unwrap_or_default();

    // 5. CUDA driver version integer (e.g. 12040) → "12.40".
    let cuda_runtime = {
        let mut ver: std::ffi::c_int = 0;
        // SAFETY: cuDriverGetVersion writes exactly one i32; no aliasing.
        let ok =
            unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut ver as *mut _).result().is_ok() };
        if ok && ver > 0 {
            let major = ver / 1000;
            let minor = (ver % 1000) / 10;
            format!("{}.{}", major, minor)
        } else {
            String::new()
        }
    };

    // 6. Free GPU memory → GB f32.
    let free_gpu_gb_at_probe = result::mem_get_info()
        .map(|(free, _total)| free as f32 / (1u64 << 30) as f32)
        .unwrap_or(0.0);

    // 7. NVIDIA driver string from nvidia-smi (best-effort).
    let driver = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    Ok(Some(SystemProbe {
        ddrs_version: env!("CARGO_PKG_VERSION").to_string(),
        probed_at: chrono::Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        gpu,
        cuda_runtime,
        driver,
        sm,
        free_gpu_gb_at_probe,
        smoke_test: None,
    }))
}

/// Stable key used to decide whether a cached smoke-test verdict is still
/// valid. Re-run when this string changes. Backend-aware: switching between
/// CUDA and CPU invalidates the cache.
pub fn smoke_key(probe: &SystemProbe, backend: &str) -> String {
    format!(
        "driver={};cuda={};ddrs={};sm={};backend={}",
        probe.driver, probe.cuda_runtime, probe.ddrs_version, probe.sm, backend
    )
}

pub fn record_smoke(probe: &mut SystemProbe, key: String, backend: &str) {
    probe.smoke_test = Some(SmokeTestRecord {
        key,
        passed_at: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        backend: Some(backend.to_string()),
    });
}

/// Result of [`ensure_system_ready`].
pub struct SystemReadiness {
    /// Post-write probe snapshot (matches what was persisted to
    /// `.ddrs/system.json`); available to callers needing GPU metadata.
    pub probe: SystemProbe,
    pub smoke_passed: bool,
    pub smoke_reused: bool,
}

/// Ensure the workspace skeleton exists and the GPU probe + smoke test are
/// recorded in `.ddrs/system.json`. Idempotent: a cached smoke verdict
/// (keyed by [`smoke_key`]) is reused unless `force` is set. This is the
/// former `ddrs init` Phase A, now the first step of `ddrs plan`.
pub fn ensure_system_ready(
    ws: &Workspace,
    force: bool,
    min_free_gpu_gb: f32,
    skip_smoke: bool,
) -> Result<SystemReadiness, CliError> {
    let mut probe = probe()?.unwrap_or_default();
    // Skip the warning on CPU-only hosts and when mem_get_info failed — both
    // report free_gpu_gb_at_probe == 0.0.
    if probe.free_gpu_gb_at_probe < min_free_gpu_gb && probe.free_gpu_gb_at_probe > 0.0 {
        eprintln!(
            "warning: free GPU memory {:.1} GB is below floor {} GB",
            probe.free_gpu_gb_at_probe, min_free_gpu_gb
        );
    }
    fs::create_dir_all(ws.runs_dir())?;
    fs::write(ws.version_file(), env!("CARGO_PKG_VERSION"))?;

    // Pick backend up-front so the cache key matches the work we'd do.
    let backend = if probe.gpu.is_empty() { "cpu" } else { "cuda" };
    let key = smoke_key(&probe, backend);
    let cached_passing = SystemProbe::read(&ws.system_json())
        .ok()
        .and_then(|p| p.smoke_test)
        .map(|s| s.key == key)
        .unwrap_or(false);
    let (smoke_passed, smoke_reused) = if skip_smoke {
        // Don't claim "reused" if there's no prior record — just "passed".
        (true, cached_passing)
    } else if cached_passing && !force {
        (true, true)
    } else {
        let (ok, _b) = run_smoke(&probe)?;
        (ok, false)
    };
    if smoke_passed && !smoke_reused {
        record_smoke(&mut probe, key, backend);
    } else if smoke_reused {
        // Preserve the prior smoke_test record.
        if let Ok(prior) = SystemProbe::read(&ws.system_json()) {
            probe.smoke_test = prior.smoke_test;
        }
    }
    probe.write_atomic(&ws.system_json())?;
    Ok(SystemReadiness { probe, smoke_passed, smoke_reused })
}

fn run_smoke(probe: &SystemProbe) -> Result<(bool, &'static str), CliError> {
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
pub fn run_smoke_for_test(probe: &SystemProbe) -> Result<(bool, &'static str), CliError> {
    run_smoke(probe)
}
