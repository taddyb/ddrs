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

use burn::tensor::Tensor;
use burn::tensor::backend::BackendTypes;

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
