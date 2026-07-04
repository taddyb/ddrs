//! Stage-1 probe core: lifting normalized leakance params to autograd leaves
//! must (a) leave the routed forward byte-identical, (b) yield finite nonzero
//! leaf grads on a losing chain, and (c) accumulate by COMID exactly.

mod common;

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use common::{
    mock_config, mock_routing_inputs, mock_streamflow, InnerBackend, TestDevice,
};
use ddrs::routing::{MuskingumCunge, SpatialParameters};
use ddrs::training::probe::{lift_leaf, GradAccum};

type AB = Autodiff<InnerBackend>;

/// Losing-regime leakance params with the three leakance vectors lifted as
/// require_grad leaves. Mirrors tests/zeta_accum.rs::leakance_params.
fn probed_params(
    n: usize,
    device: &TestDevice,
) -> (SpatialParameters<InnerBackend>, [Tensor<AB, 1>; 3]) {
    let k_d = lift_leaf::<InnerBackend>(Tensor::<AB, 1>::ones([n], device));
    let d_gw = lift_leaf::<InnerBackend>(Tensor::<AB, 1>::zeros([n], device));
    let factor = lift_leaf::<InnerBackend>(Tensor::<AB, 1>::ones([n], device) * 0.5);
    (
        SpatialParameters {
            n: Tensor::<AB, 1>::ones([n], device) * 0.5,
            q_spatial: Tensor::<AB, 1>::ones([n], device) * 0.5,
            p_spatial: None,
            k_d: Some(k_d.clone()),
            d_gw: Some(d_gw.clone()),
            leakance_factor: Some(factor.clone()),
        },
        [k_d, d_gw, factor],
    )
}

#[test]
fn lifted_leaves_do_not_perturb_forward() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 24usize);
    let cfg = mock_config();

    // Plain (non-leaf) leakance run — same values as probed_params.
    let mut mc_plain = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
    mc_plain.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        SpatialParameters {
            n: Tensor::<AB, 1>::ones([n], &device) * 0.5,
            q_spatial: Tensor::<AB, 1>::ones([n], &device) * 0.5,
            p_spatial: None,
            k_d: Some(Tensor::<AB, 1>::ones([n], &device)),
            d_gw: Some(Tensor::<AB, 1>::zeros([n], &device)),
            leakance_factor: Some(Tensor::<AB, 1>::ones([n], &device) * 0.5),
        },
        false,
        None,
    );
    let out_plain: Vec<f32> = mc_plain.forward().into_data().to_vec().unwrap();

    let (params, _leaves) = probed_params(n, &device);
    let mut mc_leaf = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc_leaf.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        params,
        false, None
    );
    let out_leaf: Vec<f32> = mc_leaf.forward().into_data().to_vec().unwrap();

    assert_eq!(out_plain, out_leaf, "lifting leaves must not change routing");
}

#[test]
fn leaf_grads_are_finite_and_nonzero_on_losing_chain() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 24usize);
    let cfg = mock_config();

    let (params, [k_d, d_gw, factor]) = probed_params(n, &device);
    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        params,
        false, None
    );
    let loss = mc.forward().sum(); // any scalar downstream of every q_next
    let grads = loss.backward();

    for (name, leaf) in [("k_d", &k_d), ("d_gw", &d_gw), ("factor", &factor)] {
        let g: Vec<f32> = leaf
            .grad(&grads)
            .unwrap_or_else(|| panic!("{name}: no grad on leaf"))
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(g.len(), n);
        assert!(g.iter().all(|v| v.is_finite()), "{name}: non-finite grad {g:?}");
        assert!(
            g.iter().any(|v| v.abs() > 0.0),
            "{name}: all-zero grad on a losing chain {g:?}"
        );
    }
}

#[test]
fn grad_accum_by_comid_sums_across_batches() {
    let mut acc = GradAccum::new();
    // Batch 1: comids 10, 20.
    acc.add(&[10, 20], &[1.0, 2.0], &[1.0, -2.0]);
    // Batch 2: comids 20, 30 (overlap on 20).
    acc.add(&[20, 30], &[3.0, 4.0], &[3.0, 4.0]);

    let rows = acc.into_sorted_rows();
    // (comid, abs_sum, net_sum, count)
    assert_eq!(rows[0], (10, 1.0, 1.0, 1));
    assert_eq!(rows[1], (20, 5.0, 1.0, 2));
    assert_eq!(rows[2], (30, 4.0, 4.0, 1));
}
