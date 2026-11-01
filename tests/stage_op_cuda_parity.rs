//! CUDA arm of the fused stage op. Stub until the device stage step exists; the
//! real CPU/CUDA parity tests replace it.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use burn::tensor::Tensor;
use ddrs::config::DownstreamBoundary;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

type Gpu = burn_cuda::Cuda<f32, i32>;

const DEPTH_LB: f32 = 0.01;

// copied from tests/stage_op_parity.rs::chain
fn chain(len: &[f32], slope: &[f32]) -> (SparseAdjacency, Arc<CsrPattern>) {
    let n = len.len();
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(n, &dense, len.to_vec(), slope.to_vec());
    let pat = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, pat)
}

#[test]
#[should_panic(expected = "stage_routing CUDA path: Phase G")]
fn cuda_arm_is_the_phase_g_stub() {
    let device = Default::default();
    let tv = |v: &[f32]| Tensor::<burn::backend::Autodiff<Gpu>, 1>::from_floats(v, &device);
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z = net.z_mid.clone();
    let sp = StageParams {
        theta: 1.0,
        theta_control: 0.5,
        substeps: 1,
        boundary: DownstreamBoundary::NormalDepth,
        depth_lb: DEPTH_LB,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: true,
    };
    let mut block = StageBlock::<Gpu>::new(net, sp, &device);
    let h: Vec<f32> = z.iter().map(|z| z + 0.5).collect();
    block.set_stage(tv(&h));
    let _ = block.step_fused(
        &pattern,
        tv(&[0.035; 3]),
        tv(&[21.0; 3]),
        tv(&[0.5; 3]),
        tv(&[1.0; 3]),
        tv(&[0.0; 3]),
        tv(&[1.0; 3]),
    );
}
