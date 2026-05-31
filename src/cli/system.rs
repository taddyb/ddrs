use chrono::SecondsFormat;
use cudarc::driver::{result, CudaContext};

use crate::cli::CliError;
use crate::cli::manifest::{SmokeTestRecord, SystemProbe};

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
/// valid. Re-run when this string changes.
pub fn smoke_key(probe: &SystemProbe) -> String {
    format!(
        "driver={};cuda={};ddrs={};sm={}",
        probe.driver, probe.cuda_runtime, probe.ddrs_version, probe.sm
    )
}

pub fn record_smoke(probe: &mut SystemProbe, key: String) {
    probe.smoke_test = Some(SmokeTestRecord {
        key,
        passed_at: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    });
}
