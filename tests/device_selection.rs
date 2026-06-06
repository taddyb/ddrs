//! Validates that the routing core works on a config-selected, non-zero
//! CUDA device (top-level `device:` YAML key, `Config::device`).
//!
//! Before this existed, three internal sites hardcoded ordinal 0:
//!   - `build_cuda_pattern_cache` (src/sparse/cusparse.rs)
//!   - forward graph capture `cuDeviceGet(0)` (src/sparse/cusparse.rs)
//!   - graph replay `cuDeviceGet(0)` (src/routing/mmc_op.rs)
//! so a tensor on device 1 would have mixed contexts with a cache/graph on
//! device 0. This test runs the full sandbox smoke (MC forward) on device 1
//! across all three solver configurations and cross-checks the result
//! against device 0.
//!
//! Skips cleanly on hosts with fewer than 2 CUDA devices.

use burn::tensor::backend::BackendTypes;

use ddrs::config::SparseSolver;
use ddrs::sandbox::{load_embedded, smoke, SandboxInputs, SmokeResult};

type B = burn_cuda::Cuda<f32, i32>;
type Dev = <B as BackendTypes>::Device;

fn cuda_device_count() -> usize {
    if cudarc::driver::result::init().is_err() {
        return 0;
    }
    cudarc::driver::result::device::get_count()
        .map(|c| c as usize)
        .unwrap_or(0)
}

fn run_smoke(inputs: &SandboxInputs, device_index: usize) -> SmokeResult {
    let device = Dev::new(device_index);
    smoke::<B>(inputs, &device)
        .unwrap_or_else(|e| panic!("sandbox smoke failed on cuda:{device_index}: {e}"))
}

/// One test fn (not three) — the cusparse pattern cache is documented as
/// single-threaded ("SP-7 single-threaded training contract"), and cargo
/// runs tests within a file concurrently.
#[test]
fn sandbox_smoke_passes_on_nonzero_cuda_device() {
    let count = cuda_device_count();
    if count < 2 {
        eprintln!("skipping: need >= 2 CUDA devices, found {count}");
        return;
    }

    let mut inputs = load_embedded().expect("embedded sandbox fixture");

    // Stage 1: BURN kernels only (CPU sparse solve) on device 1.
    inputs.config.params.sparse_solver = SparseSolver::Cpu;
    inputs.config.params.use_cuda_graphs = false;
    eprintln!("[stage 1] cpu solver on cuda:1");
    let r = run_smoke(&inputs, 1);
    assert!(r.passed, "cpu-solver smoke failed on cuda:1: {r:?}");

    // Stage 2: cuSPARSE solve — exercises build_cuda_pattern_cache on a
    // non-zero ordinal.
    inputs.config.params.sparse_solver = SparseSolver::Cuda;
    inputs.config.params.use_cuda_graphs = false;
    eprintln!("[stage 2] cusparse on cuda:0");
    let r0 = run_smoke(&inputs, 0);
    eprintln!("[stage 2] cusparse on cuda:1");
    let r1 = run_smoke(&inputs, 1);
    assert!(r0.passed, "cusparse smoke failed on cuda:0: {r0:?}");
    assert!(r1.passed, "cusparse smoke failed on cuda:1: {r1:?}");
    // Same computation on two devices must agree (f32 floor; scatter-add
    // atomics allow tiny reorder diffs).
    assert!(
        (r0.max_q - r1.max_q).abs() < 1e-3,
        "device 0 vs 1 max_q diverged: {} vs {}",
        r0.max_q,
        r1.max_q
    );

    // Stage 3: cuSPARSE + CUDA-graph capture/replay — exercises the
    // cuDeviceGet(ordinal) context-binding paths on device 1. If capture
    // falls back, smoke still runs the direct path; `passed` must hold
    // either way.
    inputs.config.params.use_cuda_graphs = true;
    eprintln!("[stage 3] cusparse + cuda graphs on cuda:1");
    let rg = run_smoke(&inputs, 1);
    assert!(rg.passed, "cuda-graph smoke failed on cuda:1: {rg:?}");
    assert!(
        (r0.max_q - rg.max_q).abs() < 1e-3,
        "graph path on cuda:1 diverged from cuda:0: {} vs {}",
        r0.max_q,
        rg.max_q
    );
}
