//! SP-10 Task 0: capture/replay spike (de-risking gate).
//!
//! Verifies that cubecl + CUDA stream capture interoperate well enough to
//! capture a single BURN tensor add and replay it 3x without panic. If this
//! test fails, the SP-10 design must shift to primitive-only `_inner`
//! functions — see 2026-05-26-sp10-cuda-graphs-plan.md.
//!
//! Marked `#[ignore]` so it does not run in the default `cargo test` flow;
//! invoke explicitly with `--ignored`. The expected success line is
//!     spike OK: capture+replay survived 3 launches without panic
//!
//! ## Hybrid-capture poison spikes (SP-10 mid-investigation)
//!
//! Three additional `#[ignore]`d tests below isolate the three cuSPARSE call
//! sites of the per-timestep chain to determine which (if any) is the
//! actual graph-capture poison:
//!   - `spike_capture_spmv_alone`    — cusparseSpMV via spmv_primitive
//!   - `spike_capture_solve_alone`   — cusparseSpSV_solve via triangular_csr_solve
//!   - `spike_capture_assemble_alone`— assemble_primitive (no cuSPARSE; control)
//!
//! Each test runs the call once outside capture (warm-up + JIT), then captures
//! a single invocation, instantiates, replays 5 times with a context sync after
//! each. A replay that succeeds proves the call is graph-capture-safe; an
//! ILLEGAL_ADDRESS surfaces capture poisoning. See report at end of session.

use std::sync::Arc;

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use burn::tensor::TensorPrimitive;
use burn::tensor::backend::BackendTypes;

use ddrs::sparse::{CsrPattern, SparseAdjacency, spmv_primitive, triangular_csr_solve, assemble_primitive};

use cudarc::driver::result::{graph as graph_api, stream as stream_api};
use cudarc::driver::sys::{
    CUgraphInstantiate_flags, CUstreamCaptureMode_enum,
};

#[test]
#[ignore]
fn cuda_graph_capture_replay_spike() {
    // Use burn_cuda::Cuda directly — mirrors tests/cusparse_ptr_spike.rs
    // (burn::backend::Cuda needs the umbrella crate's "cuda" feature, which
    // is not enabled in ddrs's Cargo.toml).
    type B = burn_cuda::Cuda<f32, i32>;
    type Dev = <B as BackendTypes>::Device;

    // Skip cleanly on CPU-only hosts.
    let cuda_available = std::panic::catch_unwind(|| {
        let _d: Dev = Default::default();
    })
    .is_ok();
    if !cuda_available {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let device: Dev = Default::default();

    // -- Step 1: warm-up so cubecl JITs the add kernel before capture.
    // A captured region cannot tolerate the JIT compile path (it does
    // host syncs and module loads which break stream capture).
    let warm_a = Tensor::<B, 1>::from_floats([1.0_f32, 2.0, 3.0, 4.0], &device);
    let warm_b = Tensor::<B, 1>::from_floats([10.0_f32, 20.0, 30.0, 40.0], &device);
    let warm_sum = warm_a + warm_b;
    let _ = warm_sum.into_data(); // host-sync to ensure kernel finished compiling/running
    eprintln!("warm-up add complete; kernel JIT'd");

    // -- Step 2: fetch cubecl's active CUDA stream (the one BURN kernels run on).
    let stream = ddrs::sparse::cusparse::__spike_active_stream::<B>(&device);
    assert!(!stream.is_null(), "cubecl active stream is null");
    eprintln!("cubecl active stream: {:#x}", stream as usize);

    // cubecl binds the CUDA primary context only on its server-bound thread.
    // The stream-capture and graph APIs below require a current context on
    // *this* (test) thread, otherwise `graph::instantiate` errors with
    // CUDA_ERROR_INVALID_CONTEXT. Retain+set_current the same primary context
    // cubecl uses (device 0 by default — cubecl::cuda::CudaDevice::default()
    // resolves to the same ordinal).
    //
    // For real SP-10 code, the capture/instantiate should be done from
    // within `ComputeClient::exclusive_with_server`, which already runs on
    // the server thread (where the context is current). The spike binds it
    // explicitly so we can prove the fundamental capability from here.
    unsafe {
        let cu_device = cudarc::driver::result::device::get(0).expect("cuDeviceGet(0) failed");
        let ctx = cudarc::driver::result::primary_ctx::retain(cu_device)
            .expect("primary_ctx::retain failed");
        cudarc::driver::result::ctx::set_current(ctx).expect("ctx::set_current failed");
    }

    // Pre-allocate captured-region operand tensors *before* begin_capture.
    // `from_floats` does an H2D copy; doing it inside the captured region
    // would likely add memcpy nodes to the graph (probably fine) but the
    // simpler/cleaner spike is to capture only the kernel launch.
    let a = Tensor::<B, 1>::from_floats([1.0_f32, 2.0, 3.0, 4.0], &device);
    let b = Tensor::<B, 1>::from_floats([5.0_f32, 6.0, 7.0, 8.0], &device);
    // Touch the tensors via a sync so their handles are realized on the GPU.
    let _ = a.clone().into_data();
    let _ = b.clone().into_data();

    // -- Step 3: begin capture in THREAD_LOCAL mode (per plan).
    // THREAD_LOCAL is the strictest mode (any cross-thread CUDA call errors
    // out), which is what we want for de-risking: any leak from cubecl into
    // another thread's CUDA context will surface here, not silently corrupt.
    unsafe {
        stream_api::begin_capture(
            stream,
            CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .expect("begin_capture failed");
    }
    eprintln!("begin_capture OK");

    // -- Step 4: run a single BURN tensor add inside the captured region.
    // No host syncs! `.into_data()` would error with
    // "operation not permitted under stream capture".
    let _c = a + b;
    // Intentionally do not read `_c`; the captured kernel is what we want
    // to record. Dropping the tensor just queues the free on the stream
    // (cubecl uses async frees), which the graph can include.

    // -- Step 5: end capture and get the CUgraph handle.
    let graph = unsafe {
        stream_api::end_capture(stream).expect(
            "end_capture failed — host-sync inside captured region is the usual culprit",
        )
    };
    eprintln!("end_capture OK; CUgraph = {:?}", graph);

    // -- Step 6: instantiate with no flags. The cudarc API takes a typed
    // enum (no zero variant), so transmute 0u32 to express "no flags".
    let no_flags: CUgraphInstantiate_flags =
        unsafe { std::mem::transmute::<u32, CUgraphInstantiate_flags>(0u32) };
    let graph_exec = unsafe {
        graph_api::instantiate(graph, no_flags).expect("graph::instantiate failed")
    };
    eprintln!("graph::instantiate OK; CUgraphExec = {:?}", graph_exec);

    // -- Step 7: launch 3 times on the same stream cubecl uses.
    for i in 0..3 {
        unsafe {
            graph_api::launch(graph_exec, stream)
                .unwrap_or_else(|e| panic!("graph::launch #{i} failed: {e:?}"));
        }
    }
    // Sync so any deferred errors surface before we destroy.
    cudarc::driver::result::ctx::synchronize().expect("ctx::synchronize failed after launches");
    eprintln!("3 launches + sync OK");

    // -- Step 8: clean up — destroy exec then graph.
    unsafe {
        graph_api::exec_destroy(graph_exec).expect("exec_destroy failed");
        graph_api::destroy(graph).expect("graph::destroy failed");
    }

    println!("spike OK: capture+replay survived 3 launches without panic");
}

// ============================================================================
// SP-10 hybrid-capture poison spikes (3 tests)
// ============================================================================

const SPIKE_N: usize = 5;
const SPIKE_REPLAYS: usize = 5;

/// Build a tiny 5-node linear-chain CsrPattern (same shape as sparse_cusparse_v8).
fn spike_pattern() -> Arc<CsrPattern> {
    let mut dense = vec![0.0_f32; SPIKE_N * SPIKE_N];
    for i in 0..SPIKE_N - 1 {
        dense[(i + 1) * SPIKE_N + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(
        SPIKE_N,
        &dense,
        vec![1000.0; SPIKE_N],
        vec![0.001; SPIKE_N],
    );
    Arc::new(CsrPattern::from_sparse(&adj))
}

/// Bind cubecl's primary CUDA context to this thread (calling-thread pattern
/// from the original spike, lines 54-70). Required for cudarc capture/graph
/// APIs to see a "current" context.
fn bind_primary_ctx() {
    unsafe {
        let cu_device = cudarc::driver::result::device::get(0).expect("cuDeviceGet(0) failed");
        let ctx = cudarc::driver::result::primary_ctx::retain(cu_device)
            .expect("primary_ctx::retain failed");
        cudarc::driver::result::ctx::set_current(ctx).expect("ctx::set_current failed");
    }
}

fn cuda_available_spike() -> bool {
    std::panic::catch_unwind(|| {
        type B = burn_cuda::Cuda<f32, i32>;
        type Dev = <B as BackendTypes>::Device;
        let _d: Dev = Default::default();
    })
    .is_ok()
}

/// Replay the captured exec graph N times with `ctx::synchronize` after each
/// launch. Returns Ok(()) if all N replays succeed, or Err((replay_idx, error))
/// on the first failing replay.
unsafe fn replay_and_sync(
    graph_exec: cudarc::driver::sys::CUgraphExec,
    stream: cudarc::driver::sys::CUstream,
    n: usize,
) -> Result<(), (usize, cudarc::driver::DriverError)> {
    for i in 0..n {
        if let Err(e) = unsafe { graph_api::launch(graph_exec, stream) } {
            return Err((i, e));
        }
        if let Err(e) = cudarc::driver::result::ctx::synchronize() {
            return Err((i, e));
        }
        eprintln!("  replay #{i} OK");
    }
    Ok(())
}

/// Begin THREAD_LOCAL stream capture; panic on failure (consistent with
/// the original spike).
unsafe fn begin_capture(stream: cudarc::driver::sys::CUstream) {
    unsafe {
        stream_api::begin_capture(
            stream,
            CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .expect("begin_capture failed");
    }
}

/// End stream capture and instantiate. Panics on either step's failure.
unsafe fn end_capture_and_instantiate(
    stream: cudarc::driver::sys::CUstream,
) -> (
    cudarc::driver::sys::CUgraph,
    cudarc::driver::sys::CUgraphExec,
) {
    let graph = unsafe { stream_api::end_capture(stream).expect("end_capture failed") };
    let no_flags: CUgraphInstantiate_flags =
        unsafe { std::mem::transmute::<u32, CUgraphInstantiate_flags>(0u32) };
    let graph_exec =
        unsafe { graph_api::instantiate(graph, no_flags).expect("graph::instantiate failed") };
    (graph, graph_exec)
}

unsafe fn destroy_graph(
    graph: cudarc::driver::sys::CUgraph,
    graph_exec: cudarc::driver::sys::CUgraphExec,
) {
    unsafe {
        let _ = graph_api::exec_destroy(graph_exec);
        let _ = graph_api::destroy(graph);
    }
}

/// Spike 1: Does `cusparseSpMV` poison CUDA Graph capture?
///
/// Captures one `spmv_primitive` (with use_cuda=true) inside a stream-capture
/// region, instantiates, and replays 5 times. If all 5 replays succeed, SpMV
/// is graph-capture safe (no internal `cuMemAllocAsync`/`cuMemFreeAsync` from
/// cuSPARSE 12.x). ILLEGAL_ADDRESS on any replay = SpMV poisons capture.
#[test]
#[ignore]
fn spike_capture_spmv_alone() {
    type B = burn_cuda::Cuda<f32, i32>;
    if !cuda_available_spike() {
        eprintln!("spike_capture_spmv_alone: skip — no CUDA");
        return;
    }
    let device = <B as BackendTypes>::Device::default();
    let pattern = spike_pattern();

    // Pre-build inputs and warm up the SpMV path so cuSPARSE handles + cubecl
    // JIT kernels are all initialised BEFORE the capture region.
    let q_data: Vec<f32> = (0..SPIKE_N).map(|i| 1.0 + i as f32 * 0.1).collect();
    let q: Tensor<B, 1> = Tensor::from_floats(q_data.as_slice(), &device);

    // Warm-up: run spmv once, sync via into_data so cubecl JIT + cuSPARSE
    // descriptor analysis are complete.
    {
        let q_prim = match q.clone().into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!(),
        };
        let y_prim = spmv_primitive::<B>(&pattern, q_prim, &device, true, None);
        let _ = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(y_prim)).into_data();
    }
    eprintln!("spike_capture_spmv_alone: warm-up OK");

    // Bind primary context on calling thread.
    bind_primary_ctx();
    let stream = ddrs::sparse::cusparse::__spike_active_stream::<B>(&device);
    assert!(!stream.is_null(), "active stream is null");

    // Pre-allocate the capture-region input. .into_data() forces sync so the
    // tensor lives on a known devptr before begin_capture.
    let q_cap: Tensor<B, 1> = Tensor::from_floats(q_data.as_slice(), &device);
    let _ = q_cap.clone().into_data();

    unsafe { begin_capture(stream) };
    eprintln!("spike_capture_spmv_alone: begin_capture OK");

    // The captured op: one spmv_primitive call. No host syncs inside.
    let q_prim_cap = match q_cap.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    let _y_cap_prim = spmv_primitive::<B>(&pattern, q_prim_cap, &device, true, None);
    // Drop _y_cap_prim — its slice may be freed async on the stream (cubecl).

    let (graph, graph_exec) = unsafe { end_capture_and_instantiate(stream) };
    eprintln!("spike_capture_spmv_alone: end_capture + instantiate OK");

    let res = unsafe { replay_and_sync(graph_exec, stream, SPIKE_REPLAYS) };
    unsafe { destroy_graph(graph, graph_exec) };

    match res {
        Ok(()) => println!(
            "SPMV CAPTURE-SAFE: {} replays succeeded — SpMV does NOT poison capture",
            SPIKE_REPLAYS
        ),
        Err((i, e)) => panic!("SPMV CAPTURE-POISON: replay #{i} failed: {e:?}"),
    }
}

/// Spike 2: Does `cusparseSpSV_solve` poison CUDA Graph capture?
///
/// Captures one triangular-CSR solve `A · x = b` (with use_cuda=true) where
/// `A = I` (a_values == diag_mask). Replays 5 times. If all succeed, the
/// solve is graph-capture-safe; ILLEGAL_ADDRESS means cusparseSpSV_solve is
/// the poison.
///
/// Note: `triangular_csr_solve` returns an Autodiff tensor, but the autograd
/// tape bookkeeping is host-side — only the inner cuSPARSE solve call lands
/// on the stream.
#[test]
#[ignore]
fn spike_capture_solve_alone() {
    type B = burn_cuda::Cuda<f32, i32>;
    type AD = Autodiff<B>;
    if !cuda_available_spike() {
        eprintln!("spike_capture_solve_alone: skip — no CUDA");
        return;
    }
    let device = <B as BackendTypes>::Device::default();
    let pattern = spike_pattern();

    // a_values shape == [nnz]; with a_values == diag_mask, A == I.
    let a_data: Vec<f32> = pattern.diag_mask.clone();
    let b_data: Vec<f32> = (0..SPIKE_N).map(|i| 1.0 + i as f32 * 0.1).collect();

    // Warm-up: run solve once, sync, so analysis + JIT are done.
    {
        let a: Tensor<AD, 1> = Tensor::from_floats(a_data.as_slice(), &device);
        let b: Tensor<AD, 1> = Tensor::from_floats(b_data.as_slice(), &device);
        let x = triangular_csr_solve::<B>(&pattern, a, b, true);
        let _ = x.into_data();
    }
    eprintln!("spike_capture_solve_alone: warm-up OK");

    bind_primary_ctx();
    let stream = ddrs::sparse::cusparse::__spike_active_stream::<B>(&device);
    assert!(!stream.is_null(), "active stream is null");

    // Pre-allocate capture-region inputs.
    let a_cap: Tensor<AD, 1> = Tensor::from_floats(a_data.as_slice(), &device);
    let b_cap: Tensor<AD, 1> = Tensor::from_floats(b_data.as_slice(), &device);
    let _ = a_cap.clone().into_data();
    let _ = b_cap.clone().into_data();

    unsafe { begin_capture(stream) };
    eprintln!("spike_capture_solve_alone: begin_capture OK");

    // The captured op: one solve.
    let _x_cap = triangular_csr_solve::<B>(&pattern, a_cap, b_cap, true);

    let (graph, graph_exec) = unsafe { end_capture_and_instantiate(stream) };
    eprintln!("spike_capture_solve_alone: end_capture + instantiate OK");

    let res = unsafe { replay_and_sync(graph_exec, stream, SPIKE_REPLAYS) };
    unsafe { destroy_graph(graph, graph_exec) };

    match res {
        Ok(()) => println!(
            "SOLVE CAPTURE-SAFE: {} replays succeeded — cusparseSpSV_solve does NOT poison capture",
            SPIKE_REPLAYS
        ),
        Err((i, e)) => panic!("SOLVE CAPTURE-POISON: replay #{i} failed: {e:?}"),
    }
}

/// Spike 3: Does `assemble_primitive` poison capture? (Control test.)
///
/// `assemble_primitive` uses only BURN tensor ops (gather + neg + mul + add),
/// no cuSPARSE — so this is the negative control. Expected result: PASS.
/// If this FAILS too, the problem is broader than cuSPARSE (e.g. cubecl
/// itself or our pin-handling).
#[test]
#[ignore]
fn spike_capture_assemble_alone() {
    type B = burn_cuda::Cuda<f32, i32>;
    if !cuda_available_spike() {
        eprintln!("spike_capture_assemble_alone: skip — no CUDA");
        return;
    }
    let device = <B as BackendTypes>::Device::default();
    let pattern = spike_pattern();

    let c_data: Vec<f32> = (0..SPIKE_N).map(|i| 0.5 + i as f32 * 0.1).collect();

    // Warm-up.
    {
        let c: Tensor<B, 1> = Tensor::from_floats(c_data.as_slice(), &device);
        let c_prim = match c.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!(),
        };
        let a_prim = assemble_primitive::<B>(&pattern, c_prim, &device, None);
        let _ = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(a_prim)).into_data();
    }
    eprintln!("spike_capture_assemble_alone: warm-up OK");

    bind_primary_ctx();
    let stream = ddrs::sparse::cusparse::__spike_active_stream::<B>(&device);
    assert!(!stream.is_null(), "active stream is null");

    let c_cap: Tensor<B, 1> = Tensor::from_floats(c_data.as_slice(), &device);
    let _ = c_cap.clone().into_data();

    unsafe { begin_capture(stream) };
    eprintln!("spike_capture_assemble_alone: begin_capture OK");

    let c_prim_cap = match c_cap.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    let _a_cap_prim = assemble_primitive::<B>(&pattern, c_prim_cap, &device, None);

    let (graph, graph_exec) = unsafe { end_capture_and_instantiate(stream) };
    eprintln!("spike_capture_assemble_alone: end_capture + instantiate OK");

    let res = unsafe { replay_and_sync(graph_exec, stream, SPIKE_REPLAYS) };
    unsafe { destroy_graph(graph, graph_exec) };

    match res {
        Ok(()) => println!(
            "ASSEMBLE CAPTURE-SAFE: {} replays succeeded — control passes (cubecl tensor ops are fine)",
            SPIKE_REPLAYS
        ),
        Err((i, e)) => panic!("ASSEMBLE CAPTURE-POISON: replay #{i} failed: {e:?} (control FAILED — cubecl itself is the problem)"),
    }
}

// ============================================================================
// SP-10 SPIKE #5: fused #[cube] kernel for S1..S14 captures+replays cleanly?
// ============================================================================

/// Spike 5: Does a fused `#[cube]` geometry kernel (S1..S14 of
/// `forward_chain_inner`) capture and replay 5x without ILLEGAL_ADDRESS, and
/// does its output match a CPU-side reference?
///
/// The architectural hypothesis under test: when all intermediates of the
/// dense forward chain live in GPU registers (rather than cubecl-pool
/// handles), the captured region's allocations are zero and there is nothing
/// for cubecl's pool to recycle between capture and the first replay.
///
/// Setup: 5-node fixture, 5 input rank-1 tensors of f32, 5 scratch output
/// handles preallocated via `client.empty`.
///
/// Procedure:
///   1. Warm-up launch (JIT compile, descriptor analysis).
///   2. Probe pass — launch kernel into the pre-allocated outputs, sync,
///      read outputs back to host. Record device pointers.
///   3. Bind primary CUDA context on the test thread.
///   4. Begin THREAD_LOCAL stream capture.
///   5. Launch the kernel ONCE with the same scratch handles.
///   6. End capture, instantiate.
///   7. Replay 5 times with `cuStreamSynchronize` after each launch.
///   8. After all replays, read scratch outputs and verify against CPU
///      reference.
///
/// PASS criteria:
///   - All 5 replays succeed (no ILLEGAL_ADDRESS).
///   - Output `depth`/`top_width`/`side_slope`/`bottom_width`/
///     `hydraulic_radius` match `cpu_reference_s1_s14` within f32 precision
///     (max rel diff < 1e-5).
///
/// If PASS: the fused-kernel architectural path is viable. SP-10 can proceed
/// by fusing S1..S14, S25, b_rhs, and S28 around the cuSPARSE calls.
///
/// If FAIL (ILLEGAL_ADDRESS): even register-only kernels are poisoned. The
/// architecture mismatch is deeper than cubecl pool recycling, and SP-10
/// should be closed.
#[test]
#[ignore]
fn spike_fused_kernel_capture_replay() {
    use ddrs::cuda_graph::geometry_kernel::{cpu_reference_s1_s14, fused_geometry_s1_s14};

    type B = burn_cuda::Cuda<f32, i32>;
    type Dev = <B as BackendTypes>::Device;

    if !cuda_available_spike() {
        eprintln!("spike_fused_kernel_capture_replay: skip — no CUDA");
        return;
    }
    let device: Dev = Default::default();

    let n = SPIKE_N;
    // Reasonable per-reach values; not zero (zero hits the lower-bound
    // clamps and would mask kernel errors as "matches reference").
    let qsp_vec: Vec<f32> = (0..n).map(|i| 0.4_f32 + i as f32 * 0.01).collect();
    let qt_vec: Vec<f32> = (0..n).map(|i| 1.0_f32 + i as f32 * 0.1).collect();
    let psp_vec = vec![10.0_f32; n];
    let slope_vec = vec![0.001_f32; n];
    let n_vec = vec![0.03_f32; n]; // Manning's n

    // CPU reference for verification.
    let expected = cpu_reference_s1_s14(
        &qsp_vec, &qt_vec, &psp_vec, &slope_vec, &n_vec, 0.01, 0.01,
    );

    let client = ddrs::sparse::cusparse::__spike_compute_client::<B>(&device);

    let bytes_per = (n * std::mem::size_of::<f32>()) as usize;

    // Helper to upload an f32 slice as a Handle.
    let upload = |data: &[f32]| -> burn_cubecl::cubecl::server::Handle {
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes_per)
        };
        client.create_from_slice(raw)
    };

    // Input handles.
    let h_qsp = upload(&qsp_vec);
    let h_qt = upload(&qt_vec);
    let h_psp = upload(&psp_vec);
    let h_slope = upload(&slope_vec);
    let h_n = upload(&n_vec);

    // Pre-allocate 5 scratch output handles.
    let h_depth = client.empty(bytes_per);
    let h_topw = client.empty(bytes_per);
    let h_side = client.empty(bytes_per);
    let h_botw = client.empty(bytes_per);
    let h_hr = client.empty(bytes_per);

    eprintln!("spike #5: handles allocated");

    use burn_cubecl::cubecl::{
        CubeCount, CubeDim,
        prelude::{TensorArg, Runtime},
    };
    type CudaRT = burn_cubecl::cubecl::cuda::CudaRuntime;

    // Launch helper closure — same args every time (handles are reused).
    let launch = || {
        let stride = vec![1_usize];
        let shape = vec![n];
        unsafe {
            fused_geometry_s1_s14::launch::<f32, CudaRT>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(n as u32),
                TensorArg::from_raw_parts(h_qsp.clone(), stride.clone().into(), shape.clone().into()),
                TensorArg::from_raw_parts(h_qt.clone(), stride.clone().into(), shape.clone().into()),
                TensorArg::from_raw_parts(h_psp.clone(), stride.clone().into(), shape.clone().into()),
                TensorArg::from_raw_parts(h_slope.clone(), stride.clone().into(), shape.clone().into()),
                TensorArg::from_raw_parts(h_n.clone(), stride.clone().into(), shape.clone().into()),
                TensorArg::from_raw_parts(h_depth.clone(), stride.clone().into(), shape.clone().into()),
                TensorArg::from_raw_parts(h_topw.clone(), stride.clone().into(), shape.clone().into()),
                TensorArg::from_raw_parts(h_side.clone(), stride.clone().into(), shape.clone().into()),
                TensorArg::from_raw_parts(h_botw.clone(), stride.clone().into(), shape.clone().into()),
                TensorArg::from_raw_parts(h_hr.clone(), stride.into(), shape.into()),
            );
        }
    };

    // Warm-up: launch + read to JIT and verify outputs match CPU reference
    // BEFORE capture (validates the kernel arithmetic alone).
    launch();
    let warm_depth = client.read_one(h_depth.clone()).expect("warm-up depth read");
    let warm_depth_f32 = unsafe {
        std::slice::from_raw_parts(warm_depth.as_ptr() as *const f32, n)
    };
    for i in 0..n {
        let diff = (warm_depth_f32[i] - expected[0][i]).abs();
        let rel = diff / expected[0][i].abs().max(1e-9);
        assert!(
            rel < 1e-5,
            "warm-up kernel arithmetic FAIL at i={i}: gpu={} cpu={} rel={rel}",
            warm_depth_f32[i],
            expected[0][i]
        );
    }
    eprintln!("spike #5: warm-up arithmetic verified vs CPU reference");

    // Now do the capture/replay sequence.
    bind_primary_ctx();
    let stream = ddrs::sparse::cusparse::__spike_active_stream::<B>(&device);
    assert!(!stream.is_null(), "active stream is null");

    // PROBE PASS: launch into the same handles, sync, record output pointers.
    // (Useful diagnostic if replay misbehaves — confirms handle ptrs are
    // stable across launches.)
    launch();
    cudarc::driver::result::ctx::synchronize().expect("probe sync failed");
    eprintln!("spike #5: probe pass OK");

    unsafe { begin_capture(stream) };
    eprintln!("spike #5: begin_capture OK");

    // Captured op: ONE launch of the fused kernel.
    launch();

    let (graph, graph_exec) = unsafe { end_capture_and_instantiate(stream) };
    eprintln!("spike #5: end_capture + instantiate OK");

    let replay_result = unsafe { replay_and_sync(graph_exec, stream, SPIKE_REPLAYS) };

    // Read final outputs (after all replays) — we only verify after the last
    // replay since all 5 launches write to the same scratch.
    let read_out = |h: &burn_cubecl::cubecl::server::Handle| -> Vec<f32> {
        let bytes = client.read_one(h.clone()).expect("read after replay");
        let slice = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n)
        };
        slice.to_vec()
    };

    let mut final_depth = Vec::new();
    let mut final_topw = Vec::new();
    let mut final_side = Vec::new();
    let mut final_botw = Vec::new();
    let mut final_hr = Vec::new();
    if replay_result.is_ok() {
        final_depth = read_out(&h_depth);
        final_topw = read_out(&h_topw);
        final_side = read_out(&h_side);
        final_botw = read_out(&h_botw);
        final_hr = read_out(&h_hr);
    }

    unsafe { destroy_graph(graph, graph_exec) };

    match replay_result {
        Ok(()) => {
            // Verify all 5 outputs match CPU reference within f32 precision.
            let names = ["depth", "top_width", "side_slope", "bottom_width", "hydraulic_radius"];
            let actuals = [&final_depth, &final_topw, &final_side, &final_botw, &final_hr];
            for (k, (name, actual)) in names.iter().zip(actuals.iter()).enumerate() {
                for i in 0..n {
                    let exp = expected[k][i];
                    let act = actual[i];
                    let diff = (act - exp).abs();
                    let rel = diff / exp.abs().max(1e-9);
                    assert!(
                        rel < 1e-5,
                        "OUTPUT MISMATCH after replay: {name}[{i}] gpu={act} cpu={exp} rel={rel}"
                    );
                }
            }
            println!(
                "FUSED-KERNEL CAPTURE-SAFE: {} replays succeeded AND outputs match CPU reference. \
                 Verdict: fused-kernel architecture UNLOCKS SP-10.",
                SPIKE_REPLAYS
            );
        }
        Err((i, e)) => panic!(
            "FUSED-KERNEL CAPTURE-POISON: replay #{i} failed: {e:?}. \
             Verdict: even register-only kernels are poisoned — SP-10 architectural \
             path does NOT unlock cleanly. CLOSE SP-10."
        ),
    }
}

// ============================================================================
// SP-10 diagnostic SPIKE #4: full chain, no pinning, no persistent mode.
// (Disabled — depends on `__spike_capture_full_chain_no_pinning` helper that
// was never added to src/sparse/cusparse.rs. Left intact for documentation;
// gated to keep the file compiling.)
// ============================================================================

/// Spike 4: Does the full S1..S28 chain survive capture + replay WITHOUT
/// `PersistentModeGuard` and WITHOUT pinning the intermediate Handles?
///
/// Spikes 1-3 showed each individual cuSPARSE call is graph-capture safe in
/// isolation, but the full chain in `try_capture_forward` (which uses pinning
/// + persistent mode) still fails at replay #2-3 with ILLEGAL_ADDRESS. This
/// spike isolates:
///   - (a) the FULL chain's kernel interaction is the bug — independent of
///         pinning/persistent. The hybrid (split-capture) plan is the right
///         architectural fix.
///   - (b) pinning + persistent mode itself perturbs cubecl state in a way
///         that breaks otherwise-fine sequences. The fix is to strip those
///         and find an alternative.
///
/// Reuses `__spike_capture_full_chain_no_pinning` which mirrors
/// `try_capture_forward` minus the two pinning machinery pieces. On
/// `Ok(())` -> verdict (b); on `Err(...)` -> verdict (a).
#[cfg(any())]
#[test]
#[ignore]
fn spike_capture_full_chain_no_pinning() {
    type B = burn_cuda::Cuda<f32, i32>;
    if !cuda_available_spike() {
        eprintln!("spike_capture_full_chain_no_pinning: skip — no CUDA");
        return;
    }
    let device = <B as BackendTypes>::Device::default();
    let pattern = spike_pattern();

    // Cfg with sparse_solver = Cuda so forward_chain_inner takes the SpMV /
    // cuSPARSE solve path (matching production).
    let mut cfg = ddrs::config::Config::default();
    cfg.params.sparse_solver = ddrs::config::SparseSolver::Cuda;

    // Eight inputs of shape [n] = [5]. Reasonable per-reach values (real
    // capture only needs the structure to be valid; numeric correctness is
    // not asserted here).
    let n_vec = vec![0.03_f32; SPIKE_N];                  // Manning n
    let qsp_vec: Vec<f32> = (0..SPIKE_N).map(|i| 0.4_f32 + i as f32 * 0.01).collect(); // q_spatial
    let psp_vec = vec![10.0_f32; SPIKE_N];                // p_spatial
    let qt_vec: Vec<f32> = (0..SPIKE_N).map(|i| 1.0_f32 + i as f32 * 0.1).collect();   // q_t
    let qpt_vec: Vec<f32> = (0..SPIKE_N).map(|i| 0.9_f32 + i as f32 * 0.1).collect();  // q_prime_t
    let len_vec = vec![1000.0_f32; SPIKE_N];              // length
    let slope_vec = vec![0.001_f32; SPIKE_N];             // slope
    let xst_vec = vec![0.3_f32; SPIKE_N];                 // x_storage

    let mk_prim = |v: &[f32]| -> <B as BackendTypes>::FloatTensorPrimitive {
        let t = Tensor::<B, 1>::from_floats(v, &device);
        match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!(),
        }
    };

    let n_prim = mk_prim(&n_vec);
    let qsp_prim = mk_prim(&qsp_vec);
    let psp_prim = mk_prim(&psp_vec);
    let len_prim = mk_prim(&len_vec);
    let slope_prim = mk_prim(&slope_vec);
    let xst_prim = mk_prim(&xst_vec);
    // q_t / q_prime_t are routed through scratch inside the helper, so they
    // don't need primitives here — the helper allocates `in_q`/`in_qp` itself.
    // But the chain still needs SOME 5-element [n]-shaped input; the helper's
    // scratch is empty (un-initialized device memory). For an SpMV-only
    // structural test that's fine; if the chain triggers NaNs/Infs they're
    // still valid CUDA reads. Suppress any unused warnings:
    let _ = (qt_vec, qpt_vec);

    eprintln!("spike_capture_full_chain_no_pinning: starting");
    let res = ddrs::sparse::cusparse::__spike_capture_full_chain_no_pinning::<B>(
        &cfg,
        &pattern,
        n_prim,
        qsp_prim,
        psp_prim,
        len_prim,
        slope_prim,
        xst_prim,
        &device,
        SPIKE_REPLAYS,
    );

    match res {
        Ok(()) => println!(
            "FULL-CHAIN NO-PIN CAPTURE-SAFE: {} replays succeeded — verdict (b): \
             pinning + persistent mode is the culprit in try_capture_forward.",
            SPIKE_REPLAYS
        ),
        Err(msg) => panic!(
            "FULL-CHAIN NO-PIN CAPTURE-POISON: {msg} — verdict (a): the chain \
             interaction itself is the bug; hybrid (split-capture) is the right \
             next step."
        ),
    }
}
