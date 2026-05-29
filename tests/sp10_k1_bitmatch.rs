//! SP-10 Phase 1: bit-match test for the production fused K1 kernel.
//!
//! Compares all 19 outputs of [`forward_k1_kernel`] against the BURN-chain
//! reference (the S1..S23 outputs of [`forward_chain_inner`]) on a 5-node
//! linear-chain fixture. Threshold is abs < 1e-5 — the f32 precision floor
//! at the values exercised here. Differences may arise from:
//!
//!   - `(e * ln(x)).exp()` vs cubecl's `powf` lowering vs CPU `powf`
//!   - BURN ops splitting expressions across kernels (intermediate stores
//!     round to f32) vs the fused kernel keeping intermediates in registers
//!   - FMA fusion differences between cube codegen and BURN's per-op
//!     PTX/CUDA backends
//!
//! Marked `#[ignore]` so it does not run in the default `cargo test` flow.
//! Invoke explicitly with:
//!
//!     cargo test --release --test sp10_k1_bitmatch -- --ignored --nocapture

use std::sync::Arc;

use burn::tensor::Tensor;
use burn::tensor::backend::BackendTypes;

use ddrs::cuda_graph::geometry_kernel::forward_k1_kernel;
use ddrs::routing::mmc_op::__spike_forward_chain_k1_outputs;
use ddrs::sparse::{CsrPattern, SparseAdjacency};

const K1_N: usize = 5;

/// Build a tiny 5-node linear-chain CsrPattern (same shape as
/// `spike_pattern` in tests/sp10_spike_capture.rs).
fn k1_pattern() -> Arc<CsrPattern> {
    let mut dense = vec![0.0_f32; K1_N * K1_N];
    for i in 0..K1_N - 1 {
        dense[(i + 1) * K1_N + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(
        K1_N,
        &dense,
        vec![1000.0; K1_N],
        vec![0.001; K1_N],
    );
    Arc::new(CsrPattern::from_sparse(&adj))
}

fn cuda_available() -> bool {
    std::panic::catch_unwind(|| {
        type B = burn_cuda::Cuda<f32, i32>;
        type Dev = <B as BackendTypes>::Device;
        let _d: Dev = Default::default();
    })
    .is_ok()
}

/// Names of the 19 K1 outputs in launch-arg order. Used only for
/// human-readable diagnostics on mismatch.
const OUTPUT_NAMES: [&str; 19] = [
    "depth",
    "top_width",
    "side_slope",
    "bottom_width",
    "hydraulic_radius",
    "velocity_unclamped",
    "velocity_clamped",
    "celerity",
    "k_muskingum",
    "denom",
    "c1",
    "c2",
    "c3",
    "c4",
    "ratio",
    "denominator",
    "q_eps",
    "side_slope_raw",
    "bw_raw",
];

#[test]
#[ignore]
fn k1_bitmatch_burn_reference() {
    type B = burn_cuda::Cuda<f32, i32>;
    type Dev = <B as BackendTypes>::Device;

    if !cuda_available() {
        eprintln!("k1_bitmatch_burn_reference: skip — no CUDA");
        return;
    }
    let device: Dev = Default::default();

    // 5-node fixture. Values chosen to be away from clamp boundaries so the
    // test exercises real arithmetic (not constant clamps masking errors).
    let n_vec: Vec<f32> = vec![0.03_f32; K1_N];                              // Manning n
    let qsp_vec: Vec<f32> = (0..K1_N).map(|i| 0.4 + i as f32 * 0.01).collect();  // q_spatial
    let psp_vec: Vec<f32> = vec![10.0_f32; K1_N];                            // p_spatial
    let qt_vec: Vec<f32> = (0..K1_N).map(|i| 1.0 + i as f32 * 0.1).collect();    // q_t
    let qpt_vec: Vec<f32> = (0..K1_N).map(|i| 0.9 + i as f32 * 0.1).collect();   // q_prime_t
    let length_vec: Vec<f32> = vec![1000.0_f32; K1_N];                       // length
    let slope_vec: Vec<f32> = vec![0.001_f32; K1_N];                         // slope
    let xst_vec: Vec<f32> = vec![0.3_f32; K1_N];                             // x_storage

    let pattern = k1_pattern();

    // BURN reference. cfg.params.sparse_solver = Cuda is the production
    // setting; doesn't affect the 19 K1 outputs (those are purely dense ops),
    // but keeps the path identical to what K1 will replace.
    let mut cfg = ddrs::config::Config::default();
    cfg.params.sparse_solver = ddrs::config::SparseSolver::Cuda;

    let mk = |v: &[f32]| -> Tensor<B, 1> { Tensor::from_floats(v, &device) };
    let expected = __spike_forward_chain_k1_outputs::<B>(
        &cfg,
        &pattern,
        mk(&n_vec),
        mk(&qsp_vec),
        mk(&psp_vec),
        mk(&qt_vec),
        mk(&qpt_vec),
        mk(&length_vec),
        mk(&slope_vec),
        mk(&xst_vec),
    );
    assert_eq!(expected.len(), 19, "BURN reference should produce 19 outputs");

    // K1 launch. Upload all 8 inputs as cubecl handles; pre-allocate 19
    // output handles.
    let client = ddrs::sparse::cusparse::__spike_compute_client::<B>(&device);
    let bytes_per = K1_N * std::mem::size_of::<f32>();
    let upload = |data: &[f32]| -> burn_cubecl::cubecl::server::Handle {
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes_per)
        };
        client.create_from_slice(raw)
    };

    let h_n = upload(&n_vec);
    let h_qsp = upload(&qsp_vec);
    let h_psp = upload(&psp_vec);
    let h_qt = upload(&qt_vec);
    let h_qpt = upload(&qpt_vec);
    let h_length = upload(&length_vec);
    let h_slope = upload(&slope_vec);
    let h_xst = upload(&xst_vec);

    let h_out: Vec<burn_cubecl::cubecl::server::Handle> =
        (0..19).map(|_| client.empty(bytes_per)).collect();

    use burn_cubecl::cubecl::{
        CubeCount, CubeDim,
        prelude::TensorArg,
    };
    type CudaRT = burn_cubecl::cubecl::cuda::CudaRuntime;

    let stride = vec![1_usize];
    let shape = vec![K1_N];
    let mk_in = |h: &burn_cubecl::cubecl::server::Handle| -> TensorArg<CudaRT> {
        unsafe { TensorArg::from_raw_parts(h.clone(), stride.clone().into(), shape.clone().into()) }
    };
    let mk_out = |h: &burn_cubecl::cubecl::server::Handle| -> TensorArg<CudaRT> {
        unsafe { TensorArg::from_raw_parts(h.clone(), stride.clone().into(), shape.clone().into()) }
    };

    // Scalar bounds — match cfg.params.attribute_minimums defaults (mirrors
    // forward_chain_inner). dt is the hardcoded routing timestep.
    let bw_lb = cfg.params.attribute_minimums.bottom_width;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let dt = ddrs::routing::mmc::DT_SECONDS;

    forward_k1_kernel::launch::<f32, CudaRT>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(K1_N as u32),
        mk_in(&h_n),
        mk_in(&h_qsp),
        mk_in(&h_psp),
        mk_in(&h_qt),
        mk_in(&h_qpt),
        mk_in(&h_length),
        mk_in(&h_slope),
        mk_in(&h_xst),
        mk_out(&h_out[0]),  mk_out(&h_out[1]),  mk_out(&h_out[2]),
        mk_out(&h_out[3]),  mk_out(&h_out[4]),  mk_out(&h_out[5]),
        mk_out(&h_out[6]),  mk_out(&h_out[7]),  mk_out(&h_out[8]),
        mk_out(&h_out[9]),  mk_out(&h_out[10]), mk_out(&h_out[11]),
        mk_out(&h_out[12]), mk_out(&h_out[13]), mk_out(&h_out[14]),
        mk_out(&h_out[15]), mk_out(&h_out[16]), mk_out(&h_out[17]),
        mk_out(&h_out[18]),
        bw_lb,
        depth_lb,
        velocity_lb,
        dt,
    );

    // Read K1 outputs back to host.
    let read = |h: &burn_cubecl::cubecl::server::Handle| -> Vec<f32> {
        let bytes = client.read_one(h.clone()).expect("read K1 output");
        let slice = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, K1_N)
        };
        slice.to_vec()
    };
    let actual: Vec<Vec<f32>> = h_out.iter().map(read).collect();

    // Compare elementwise. Track worst diff for diagnostics.
    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    let mut worst_where = (String::new(), 0_usize);

    // f32 has ~7 decimal digits of precision, so abs diff scales with
    // magnitude. Use a hybrid threshold: abs < max(1e-5, 1e-5 * |value|).
    // This is the f32 precision floor — anything smaller would require f64
    // or fused-multiply-add ordering to match exactly. The spike pattern
    // checks `rel < 1e-5`, which is equivalent at this scale.
    for (k, name) in OUTPUT_NAMES.iter().enumerate() {
        for i in 0..K1_N {
            let a = actual[k][i];
            let e = expected[k][i];
            let abs = (a - e).abs();
            let rel = abs / e.abs().max(1e-9);
            if abs > worst_abs {
                worst_abs = abs;
                worst_rel = rel;
                worst_where = (name.to_string(), i);
            }
            let tol = (1e-5_f32 * e.abs()).max(1e-5_f32);
            assert!(
                abs < tol,
                "BIT-MATCH FAIL: {name}[{i}] kernel={a} burn={e} abs={abs:.3e} \
                 rel={rel:.3e} tol={tol:.3e}"
            );
        }
    }

    println!(
        "K1 BIT-MATCH OK: 19 outputs × {} segments — worst abs {:.3e} (rel {:.3e}) at {}[{}]",
        K1_N, worst_abs, worst_rel, worst_where.0, worst_where.1
    );
}
