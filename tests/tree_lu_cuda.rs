//! CUDA arm of `tree_lu_solve`. Stub until the device solve exists; the real
//! device parity tests replace the stub test.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use burn::backend::Autodiff;
use burn::tensor::{Tensor, TensorData};
use ddrs::sparse::tree_lu::tree_lu_solve;
use ddrs::sparse::{CsrPattern, SparseAdjacency};

type Gpu = burn_cuda::Cuda<f32, i32>;

#[test]
#[should_panic(expected = "stage_routing CUDA path: Phase G")]
fn cuda_arm_is_the_phase_g_stub() {
    // A 3-reach chain 0 -> 1 -> 2 (copied from tests/stage_op_parity.rs::chain).
    let n = 3;
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(n, &dense, vec![1000.0; n], vec![1e-3; n]);
    let p = Arc::new(CsrPattern::from_sparse(&adj));
    let device = Default::default();
    let a = Tensor::<Autodiff<Gpu>, 1>::from_data(TensorData::new(vec![1.0f32; 2 * p.nnz()], [2 * p.nnz()]), &device);
    let b = Tensor::<Autodiff<Gpu>, 1>::from_data(TensorData::new(vec![1.0f32; n], [n]), &device);
    let _ = tree_lu_solve(&p, a, b, true);
}
