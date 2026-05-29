//! SP-10 Phase 2: bit-match tests for the K2 (b_rhs) and K3 (q_clamp) fused
//! kernels.
//!
//! K2 computes S25 of `forward_chain_inner`:
//!
//!     b_rhs = c2 * i_t + c3 * q_t + c4 * q_prime_t
//!
//! K3 computes S28:
//!
//!     q_next = max(x_sol, discharge_lb)
//!
//! Both tests build a 5-node linear-chain fixture (same shape as the K1
//! bit-match test), run the BURN chain as the reference, then launch the
//! fused kernel on the same inputs and compare elementwise.
//!
//! Tolerance: abs < max(1e-5, 1e-5 * |value|). The arithmetic in K2 / K3 is
//! trivial (no transcendentals), so we expect actual diffs at the floor.
//!
//! Both marked `#[ignore]` so they don't run in the default `cargo test`
//! flow. Invoke explicitly with:
//!
//!     cargo test --release --test sp10_k23_bitmatch -- --ignored --nocapture

use std::sync::Arc;

use burn::tensor::Tensor;
use burn::tensor::backend::BackendTypes;

use ddrs::cuda_graph::geometry_kernel::{b_rhs_kernel, q_clamp_kernel};
use ddrs::routing::mmc_op::__spike_forward_chain_k23_outputs;
use ddrs::sparse::{CsrPattern, SparseAdjacency};

const K23_N: usize = 5;

/// Build a tiny 5-node linear-chain CsrPattern (same shape as `k1_pattern`).
fn k23_pattern() -> Arc<CsrPattern> {
    let mut dense = vec![0.0_f32; K23_N * K23_N];
    for i in 0..K23_N - 1 {
        dense[(i + 1) * K23_N + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(
        K23_N,
        &dense,
        vec![1000.0; K23_N],
        vec![0.001; K23_N],
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

/// Shared 5-node fixture matching `k1_bitmatch_burn_reference`.
fn k23_inputs() -> (
    Vec<f32>, // n
    Vec<f32>, // qsp
    Vec<f32>, // psp
    Vec<f32>, // qt
    Vec<f32>, // qpt
    Vec<f32>, // length
    Vec<f32>, // slope
    Vec<f32>, // xst
) {
    let n: Vec<f32> = vec![0.03_f32; K23_N];
    let qsp: Vec<f32> = (0..K23_N).map(|i| 0.4 + i as f32 * 0.01).collect();
    let psp: Vec<f32> = vec![10.0_f32; K23_N];
    let qt: Vec<f32> = (0..K23_N).map(|i| 1.0 + i as f32 * 0.1).collect();
    let qpt: Vec<f32> = (0..K23_N).map(|i| 0.9 + i as f32 * 0.1).collect();
    let length: Vec<f32> = vec![1000.0_f32; K23_N];
    let slope: Vec<f32> = vec![0.001_f32; K23_N];
    let xst: Vec<f32> = vec![0.3_f32; K23_N];
    (n, qsp, psp, qt, qpt, length, slope, xst)
}

#[test]
#[ignore]
fn k2_bitmatch_burn_reference() {
    type B = burn_cuda::Cuda<f32, i32>;
    type Dev = <B as BackendTypes>::Device;

    if !cuda_available() {
        eprintln!("k2_bitmatch_burn_reference: skip — no CUDA");
        return;
    }
    let device: Dev = Default::default();

    let (n_v, qsp_v, psp_v, qt_v, qpt_v, length_v, slope_v, xst_v) = k23_inputs();

    let pattern = k23_pattern();

    let mut cfg = ddrs::config::Config::default();
    cfg.params.sparse_solver = ddrs::config::SparseSolver::Cuda;

    let mk = |v: &[f32]| -> Tensor<B, 1> { Tensor::from_floats(v, &device) };

    // Reference: full BURN chain. Returns (b_rhs, i_t, x_sol, q_next).
    let (b_rhs_ref, i_t_ref, _x_sol_ref, _q_next_ref) =
        __spike_forward_chain_k23_outputs::<B>(
            &cfg,
            &pattern,
            mk(&n_v),
            mk(&qsp_v),
            mk(&psp_v),
            mk(&qt_v),
            mk(&qpt_v),
            mk(&length_v),
            mk(&slope_v),
            mk(&xst_v),
        );

    // Also need c2, c3, c4 (K1 outputs at indices 11, 12, 13 in OUTPUT_NAMES
    // order from K1 bit-match). Just re-extract via the K1 helper.
    let k1_outputs = ddrs::routing::mmc_op::__spike_forward_chain_k1_outputs::<B>(
        &cfg,
        &pattern,
        mk(&n_v),
        mk(&qsp_v),
        mk(&psp_v),
        mk(&qt_v),
        mk(&qpt_v),
        mk(&length_v),
        mk(&slope_v),
        mk(&xst_v),
    );
    // K1 outputs order matches OUTPUT_NAMES in sp10_k1_bitmatch.rs:
    //   10 c1, 11 c2, 12 c3, 13 c4
    let c2_v: Vec<f32> = k1_outputs[11].clone();
    let c3_v: Vec<f32> = k1_outputs[12].clone();
    let c4_v: Vec<f32> = k1_outputs[13].clone();

    // K2 launch.
    let client = ddrs::sparse::cusparse::__spike_compute_client::<B>(&device);
    let bytes_per = K23_N * std::mem::size_of::<f32>();
    let upload = |data: &[f32]| -> burn_cubecl::cubecl::server::Handle {
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes_per)
        };
        client.create_from_slice(raw)
    };

    let h_c2 = upload(&c2_v);
    let h_c3 = upload(&c3_v);
    let h_c4 = upload(&c4_v);
    let h_it = upload(&i_t_ref);
    let h_qt = upload(&qt_v);
    let h_qpt = upload(&qpt_v);
    let h_out = client.empty(bytes_per);

    use burn_cubecl::cubecl::{
        CubeCount, CubeDim,
        prelude::TensorArg,
    };
    type CudaRT = burn_cubecl::cubecl::cuda::CudaRuntime;

    let stride = vec![1_usize];
    let shape = vec![K23_N];
    let mk_t = |h: &burn_cubecl::cubecl::server::Handle| -> TensorArg<CudaRT> {
        unsafe { TensorArg::from_raw_parts(h.clone(), stride.clone().into(), shape.clone().into()) }
    };

    b_rhs_kernel::launch::<f32, CudaRT>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(K23_N as u32),
        mk_t(&h_c2),
        mk_t(&h_c3),
        mk_t(&h_c4),
        mk_t(&h_it),
        mk_t(&h_qt),
        mk_t(&h_qpt),
        mk_t(&h_out),
    );

    let bytes = client.read_one(h_out.clone()).expect("read K2 b_rhs");
    let actual: Vec<f32> = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, K23_N)
    }
    .to_vec();

    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    let mut worst_i = 0_usize;
    for i in 0..K23_N {
        let a = actual[i];
        let e = b_rhs_ref[i];
        let abs = (a - e).abs();
        let rel = abs / e.abs().max(1e-9);
        if abs > worst_abs {
            worst_abs = abs;
            worst_rel = rel;
            worst_i = i;
        }
        let tol = (1e-5_f32 * e.abs()).max(1e-5_f32);
        assert!(
            abs < tol,
            "K2 BIT-MATCH FAIL: b_rhs[{i}] kernel={a} burn={e} abs={abs:.3e} \
             rel={rel:.3e} tol={tol:.3e}"
        );
    }

    println!(
        "K2 BIT-MATCH OK: b_rhs × {} segments — worst abs {:.3e} (rel {:.3e}) at [{}]",
        K23_N, worst_abs, worst_rel, worst_i
    );
}

#[test]
#[ignore]
fn k3_bitmatch_burn_reference() {
    type B = burn_cuda::Cuda<f32, i32>;
    type Dev = <B as BackendTypes>::Device;

    if !cuda_available() {
        eprintln!("k3_bitmatch_burn_reference: skip — no CUDA");
        return;
    }
    let device: Dev = Default::default();

    let (n_v, qsp_v, psp_v, qt_v, qpt_v, length_v, slope_v, xst_v) = k23_inputs();
    let pattern = k23_pattern();

    let mut cfg = ddrs::config::Config::default();
    cfg.params.sparse_solver = ddrs::config::SparseSolver::Cuda;

    let mk = |v: &[f32]| -> Tensor<B, 1> { Tensor::from_floats(v, &device) };

    // Reference. We need x_sol (input to K3) and q_next (K3's expected output).
    let (_b_rhs, _i_t, x_sol_ref, q_next_ref) =
        __spike_forward_chain_k23_outputs::<B>(
            &cfg,
            &pattern,
            mk(&n_v),
            mk(&qsp_v),
            mk(&psp_v),
            mk(&qt_v),
            mk(&qpt_v),
            mk(&length_v),
            mk(&slope_v),
            mk(&xst_v),
        );

    let discharge_lb = cfg.params.attribute_minimums.discharge;

    // K3 launch.
    let client = ddrs::sparse::cusparse::__spike_compute_client::<B>(&device);
    let bytes_per = K23_N * std::mem::size_of::<f32>();
    let upload = |data: &[f32]| -> burn_cubecl::cubecl::server::Handle {
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes_per)
        };
        client.create_from_slice(raw)
    };

    let h_x_sol = upload(&x_sol_ref);
    let h_out = client.empty(bytes_per);

    use burn_cubecl::cubecl::{
        CubeCount, CubeDim,
        prelude::TensorArg,
    };
    type CudaRT = burn_cubecl::cubecl::cuda::CudaRuntime;

    let stride = vec![1_usize];
    let shape = vec![K23_N];
    let mk_t = |h: &burn_cubecl::cubecl::server::Handle| -> TensorArg<CudaRT> {
        unsafe { TensorArg::from_raw_parts(h.clone(), stride.clone().into(), shape.clone().into()) }
    };

    q_clamp_kernel::launch::<f32, CudaRT>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(K23_N as u32),
        mk_t(&h_x_sol),
        mk_t(&h_out),
        discharge_lb,
    );

    let bytes = client.read_one(h_out.clone()).expect("read K3 q_next");
    let actual: Vec<f32> = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, K23_N)
    }
    .to_vec();

    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    let mut worst_i = 0_usize;
    for i in 0..K23_N {
        let a = actual[i];
        let e = q_next_ref[i];
        let abs = (a - e).abs();
        let rel = abs / e.abs().max(1e-9);
        if abs > worst_abs {
            worst_abs = abs;
            worst_rel = rel;
            worst_i = i;
        }
        let tol = (1e-5_f32 * e.abs()).max(1e-5_f32);
        assert!(
            abs < tol,
            "K3 BIT-MATCH FAIL: q_next[{i}] kernel={a} burn={e} abs={abs:.3e} \
             rel={rel:.3e} tol={tol:.3e}"
        );
    }

    println!(
        "K3 BIT-MATCH OK: q_next × {} segments — worst abs {:.3e} (rel {:.3e}) at [{}]",
        K23_N, worst_abs, worst_rel, worst_i
    );
}
