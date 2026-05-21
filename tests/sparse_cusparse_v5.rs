//! SP-6 V5 (placeholder for Task 11): forward-only smoke that verifies the
//! CPU and CUDA paths run end-to-end. Bit-match assertions land in Task 11.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{backend::BackendTypes, Tensor};

use ddrs::sparse::{triangular_csr_solve, CsrPattern, SparseAdjacency};

/// Build a small 5×5 lower-triangular test pattern.
/// A[i, i] = 2.0, A[i, i-1] = 0.5 for i > 0.
fn small_lower_pattern() -> Arc<CsrPattern> {
    let n = 5;
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n {
        dense[i * n + i] = 2.0;
        if i > 0 {
            dense[i * n + (i - 1)] = 0.5;
        }
    }
    let adj = SparseAdjacency::from_dense(
        n,
        &dense,
        vec![1000.0; n],
        vec![0.001; n],
    );
    Arc::new(CsrPattern::from_sparse(&adj))
}

#[test]
fn forward_cpu_smoke() {
    type B = Autodiff<NdArray<f32>>;
    let device = <NdArray<f32> as BackendTypes>::Device::default();
    let pattern = small_lower_pattern();
    let nnz = pattern.col.len();

    // a_values: all 1.0 — matrix has diagonal=1 and sub-diagonal=1.
    let a: Tensor<B, 1> =
        Tensor::from_floats(vec![1.0_f32; nnz].as_slice(), &device);
    let b: Tensor<B, 1> =
        Tensor::from_floats(vec![1.0_f32; pattern.n].as_slice(), &device);

    let x = triangular_csr_solve(&pattern, a, b, /* use_cuda = */ false);
    let v: Vec<f32> = x.into_data().to_vec().unwrap();

    assert_eq!(v.len(), pattern.n, "output length must match n");
    assert!(
        v.iter().all(|x| x.is_finite()),
        "CPU forward produced non-finite values: {v:?}"
    );
}

#[test]
fn forward_cuda_smoke() {
    type CudaB = burn_cuda::Cuda<f32, i32>;
    type B = Autodiff<CudaB>;
    type Dev = <CudaB as BackendTypes>::Device;

    // Guard: skip if no CUDA device is available.
    let cuda_available = std::panic::catch_unwind(|| {
        let _d: Dev = Default::default();
    })
    .is_ok();
    if !cuda_available {
        eprintln!("forward_cuda_smoke: skipping — no CUDA device available");
        return;
    }

    let device: Dev = Default::default();
    let pattern = small_lower_pattern();
    let nnz = pattern.col.len();

    let a: Tensor<B, 1> =
        Tensor::from_floats(vec![1.0_f32; nnz].as_slice(), &device);
    let b: Tensor<B, 1> =
        Tensor::from_floats(vec![1.0_f32; pattern.n].as_slice(), &device);

    let x = triangular_csr_solve(&pattern, a, b, /* use_cuda = */ true);
    let v: Vec<f32> = x.into_data().to_vec().unwrap();

    assert_eq!(v.len(), pattern.n, "output length must match n");
    assert!(
        v.iter().all(|x| x.is_finite()),
        "CUDA forward produced non-finite values: {v:?}"
    );

    eprintln!("forward_cuda_smoke: x = {v:?}");
}

#[test]
fn backward_cuda_smoke() {
    type CudaB = burn_cuda::Cuda<f32, i32>;
    type B = Autodiff<CudaB>;
    type Dev = <CudaB as BackendTypes>::Device;

    // Guard: skip if no CUDA device is available.
    let cuda_available = std::panic::catch_unwind(|| {
        let _d: Dev = Default::default();
    })
    .is_ok();
    if !cuda_available {
        eprintln!("backward_cuda_smoke: skipping — no CUDA device available");
        return;
    }

    let device: Dev = Default::default();
    let pattern = small_lower_pattern();
    let nnz = pattern.col.len();

    // Build autograd tensors.
    let a: Tensor<B, 1> =
        Tensor::from_floats(vec![1.0_f32; nnz].as_slice(), &device).require_grad();
    let b: Tensor<B, 1> =
        Tensor::from_floats(vec![1.0_f32; pattern.n].as_slice(), &device).require_grad();

    // Forward + backward through the CUDA path.
    let x = triangular_csr_solve(&pattern, a.clone(), b.clone(), /* use_cuda = */ true);
    let loss = x.sum();
    let grads = loss.backward();

    let grad_b: Vec<f32> = b
        .grad(&grads)
        .expect("grad_b missing")
        .into_data()
        .to_vec()
        .unwrap();
    let grad_a: Vec<f32> = a
        .grad(&grads)
        .expect("grad_a missing")
        .into_data()
        .to_vec()
        .unwrap();

    assert!(
        grad_b.iter().all(|v| v.is_finite()),
        "non-finite grad_b from cuSPARSE backward: {grad_b:?}"
    );
    assert!(
        grad_a.iter().all(|v| v.is_finite()),
        "non-finite grad_a from CPU scatter: {grad_a:?}"
    );

    eprintln!("backward_cuda_smoke: grad_b = {grad_b:?}");
    eprintln!("backward_cuda_smoke: grad_a = {grad_a:?}");
}
