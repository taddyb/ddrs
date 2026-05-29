//! SP-9 V8: per-site CPU/CUDA bit-match for cuSPARSE SpMV.
//!
//! Compares the three new cusparse_spmv_* functions against their CPU
//! .scatter-based counterparts on a small synthetic CsrPattern. Tolerance
//! 1e-5 relative (matches the existing V5 CSR-solve bit-match).
//!
//! Run manually:
//!   cargo test --release --test sparse_cusparse_v8 -- --ignored --nocapture

use std::sync::Arc;

use burn::backend::NdArray;
use burn::tensor::{backend::BackendTypes, Tensor, TensorPrimitive};

use ddrs::sparse::{
    assemble_backward_primitive, spmv_backward_primitive, spmv_primitive, CsrPattern,
    SparseAdjacency,
};

const N: usize = 8;
const REL_TOL: f32 = 1e-5;

fn linear_chain_sparse() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![1000.0; N], vec![0.001; N])
}

fn cuda_available() -> bool {
    std::panic::catch_unwind(|| {
        type CudaInner = burn_cuda::Cuda<f32, i32>;
        type Dev = <CudaInner as BackendTypes>::Device;
        let _d: Dev = Default::default();
    })
    .is_ok()
}

fn build_pattern() -> Arc<CsrPattern> {
    let adj = linear_chain_sparse();
    Arc::new(CsrPattern::from_sparse(&adj))
}

fn vec_max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(av, bv)| {
            let denom = av.abs().max(bv.abs()).max(1e-12);
            (av - bv).abs() / denom
        })
        .fold(0.0_f32, f32::max)
}

#[test]
#[ignore]
fn v8_spmv_forward_cpu_vs_cuda_bit_match() {
    type CudaB = burn_cuda::Cuda<f32, i32>;
    type NdB = NdArray<f32>;
    if !cuda_available() {
        eprintln!("V8 spmv_forward: skip — no CUDA");
        return;
    }
    let pattern = build_pattern();

    let nd_dev = <NdB as BackendTypes>::Device::default();
    let q_data: Vec<f32> = (0..N).map(|i| 1.0 + i as f32 * 0.1).collect();
    let q_cpu: Tensor<NdB, 1> = Tensor::from_floats(q_data.as_slice(), &nd_dev);
    let q_cpu_prim = match q_cpu.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    // use_cuda=false → CPU .scatter path.
    let y_cpu_prim = spmv_primitive::<NdB>(&pattern, q_cpu_prim, &nd_dev, false, None);
    let y_cpu_vec: Vec<f32> =
        Tensor::<NdB, 1>::from_primitive(TensorPrimitive::Float(y_cpu_prim))
            .to_data()
            .to_vec()
            .unwrap();

    let cuda_dev = <CudaB as BackendTypes>::Device::default();
    let q_cuda: Tensor<CudaB, 1> = Tensor::from_floats(q_data.as_slice(), &cuda_dev);
    let q_cuda_prim = match q_cuda.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    // use_cuda=true → cusparseSpMV path.
    let y_cuda_prim = spmv_primitive::<CudaB>(&pattern, q_cuda_prim, &cuda_dev, true, None);
    let y_cuda_vec: Vec<f32> =
        Tensor::<CudaB, 1>::from_primitive(TensorPrimitive::Float(y_cuda_prim))
            .to_data()
            .to_vec()
            .unwrap();

    let rel = vec_max_rel_diff(&y_cpu_vec, &y_cuda_vec);
    eprintln!(
        "V8 spmv_forward: y_cpu={:?}, y_cuda={:?}, max_rel={}",
        y_cpu_vec, y_cuda_vec, rel
    );
    assert!(rel < REL_TOL, "V8 spmv_forward: max_rel={rel} >= {REL_TOL}");
}

#[test]
#[ignore]
fn v8_spmv_backward_cpu_vs_cuda_bit_match() {
    type CudaB = burn_cuda::Cuda<f32, i32>;
    type NdB = NdArray<f32>;
    if !cuda_available() {
        eprintln!("V8 spmv_backward: skip — no CUDA");
        return;
    }
    let pattern = build_pattern();
    let nd_dev = <NdB as BackendTypes>::Device::default();
    let cuda_dev = <CudaB as BackendTypes>::Device::default();

    let gi_data: Vec<f32> = (0..N).map(|i| 0.5 + i as f32 * 0.2).collect();

    let gi_cpu: Tensor<NdB, 1> = Tensor::from_floats(gi_data.as_slice(), &nd_dev);
    let gi_cpu_prim = match gi_cpu.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    let y_cpu_prim = spmv_backward_primitive::<NdB>(&pattern, gi_cpu_prim, &nd_dev, false);
    let y_cpu_vec: Vec<f32> =
        Tensor::<NdB, 1>::from_primitive(TensorPrimitive::Float(y_cpu_prim))
            .to_data()
            .to_vec()
            .unwrap();

    let gi_cuda: Tensor<CudaB, 1> = Tensor::from_floats(gi_data.as_slice(), &cuda_dev);
    let gi_cuda_prim = match gi_cuda.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    let y_cuda_prim = spmv_backward_primitive::<CudaB>(&pattern, gi_cuda_prim, &cuda_dev, true);
    let y_cuda_vec: Vec<f32> =
        Tensor::<CudaB, 1>::from_primitive(TensorPrimitive::Float(y_cuda_prim))
            .to_data()
            .to_vec()
            .unwrap();

    let rel = vec_max_rel_diff(&y_cpu_vec, &y_cuda_vec);
    eprintln!("V8 spmv_backward: max_rel={rel}");
    assert!(rel < REL_TOL, "V8 spmv_backward: max_rel={rel} >= {REL_TOL}");
}

#[test]
#[ignore]
fn v8_assemble_backward_cpu_vs_cuda_bit_match() {
    type CudaB = burn_cuda::Cuda<f32, i32>;
    type NdB = NdArray<f32>;
    if !cuda_available() {
        eprintln!("V8 assemble_backward: skip — no CUDA");
        return;
    }
    let pattern = build_pattern();
    let nd_dev = <NdB as BackendTypes>::Device::default();
    let cuda_dev = <CudaB as BackendTypes>::Device::default();

    let nnz = pattern.nnz();
    let ga_data: Vec<f32> = (0..nnz).map(|i| 0.3 + i as f32 * 0.05).collect();

    let ga_cpu: Tensor<NdB, 1> = Tensor::from_floats(ga_data.as_slice(), &nd_dev);
    let ga_cpu_prim = match ga_cpu.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    let y_cpu_prim = assemble_backward_primitive::<NdB>(&pattern, ga_cpu_prim, &nd_dev, false);
    let y_cpu_vec: Vec<f32> =
        Tensor::<NdB, 1>::from_primitive(TensorPrimitive::Float(y_cpu_prim))
            .to_data()
            .to_vec()
            .unwrap();

    let ga_cuda: Tensor<CudaB, 1> = Tensor::from_floats(ga_data.as_slice(), &cuda_dev);
    let ga_cuda_prim = match ga_cuda.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    let y_cuda_prim =
        assemble_backward_primitive::<CudaB>(&pattern, ga_cuda_prim, &cuda_dev, true);
    let y_cuda_vec: Vec<f32> =
        Tensor::<CudaB, 1>::from_primitive(TensorPrimitive::Float(y_cuda_prim))
            .to_data()
            .to_vec()
            .unwrap();

    let rel = vec_max_rel_diff(&y_cpu_vec, &y_cuda_vec);
    eprintln!("V8 assemble_backward: max_rel={rel}");
    assert!(rel < REL_TOL, "V8 assemble_backward: max_rel={rel} >= {REL_TOL}");
}
