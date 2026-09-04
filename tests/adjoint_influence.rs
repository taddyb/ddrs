//! The adjoint influence map lifts the hourly lateral inflow as a gradient
//! leaf and reads `dQ_outlet(t0)/dq'(reach, hour)` back from the existing
//! routing backward. This test proves the leaf is on the tape and the
//! gradient matches a central finite difference on a 4-reach chain.
//!
//! Spec: docs/superpowers/specs/2026-09-03-ddrs-experiment-adjoint-design.md §4.

mod common;

use burn::tensor::Tensor;
use common::{mock_config, mock_routing_inputs, mock_streamflow, InnerBackend, TestBackend, TestDevice};
use ddrs::routing::{MuskingumCunge, SpatialParameters};

const N: usize = 4;
const T: usize = 48;
const T0: usize = 40;

fn route(q: Tensor<TestBackend, 2>, device: &TestDevice) -> Tensor<TestBackend, 2> {
    let mut mc = MuskingumCunge::<InnerBackend>::new(mock_config(), device.clone());
    let params = SpatialParameters {
        n: Tensor::<TestBackend, 1>::ones([N], device) * 0.5,
        q_spatial: Tensor::<TestBackend, 1>::ones([N], device) * 0.5,
        p_spatial: Some(Tensor::<TestBackend, 1>::ones([N], device) * 0.5),
        k_d: None,
        d_gw: None,
        leakance_factor: None,
        impervious_mask: None,
    };
    mc.setup_inputs(mock_routing_inputs(N, device), q, params, false, None);
    mc.forward() // (N, T)
}

fn outlet_at(q: Tensor<TestBackend, 2>, device: &TestDevice) -> f32 {
    let out = route(q, device);
    let v: Vec<f32> = out.slice([N - 1..N, T0..T0 + 1]).inner().into_data().to_vec().unwrap();
    v[0]
}

#[test]
fn lifted_inflow_leaf_gradient_matches_finite_difference() {
    let device = TestDevice::default();
    let base = mock_streamflow(T, N, &device); // (T, N), no grad
    let q_leaf = Tensor::<TestBackend, 2>::from_inner(base.clone().inner()).require_grad();

    let out = route(q_leaf.clone(), &device);
    let scalar = out.slice([N - 1..N, T0..T0 + 1]).reshape([1]);
    let grads = scalar.backward();
    let g = q_leaf.grad(&grads).expect("inflow leaf must receive a gradient");
    assert_eq!(g.dims(), [T, N]);
    let g: Vec<f32> = g.into_data().to_vec().unwrap();

    // Causality: inflow after the anchor cannot influence Q(t0).
    for h in T0..T {
        for r in 0..N {
            assert_eq!(g[h * N + r], 0.0, "grad at future hour {h} reach {r} must be 0");
        }
    }
    // Somewhere before the anchor the gradient is non-zero.
    assert!(g[..T0 * N].iter().any(|v| v.abs() > 1e-6), "gradient is identically zero before t0");

    // Central finite difference at a headwater, a mid reach, and the outlet.
    let base_vals: Vec<f32> = base.clone().inner().into_data().to_vec().unwrap();
    let eps = 1e-2_f32;
    let mut n_checked = 0;
    for (hour, reach) in [(30usize, 0usize), (36, 1), (39, 3), (20, 2), (38, 3)] {
        let mut plus = base_vals.clone();
        let mut minus = base_vals.clone();
        plus[hour * N + reach] += eps;
        minus[hour * N + reach] -= eps;
        let qp = Tensor::<TestBackend, 1>::from_floats(plus.as_slice(), &device).reshape([T, N]);
        let qm = Tensor::<TestBackend, 1>::from_floats(minus.as_slice(), &device).reshape([T, N]);
        let fd = (outlet_at(qp, &device) - outlet_at(qm, &device)) / (2.0 * eps);
        let ad = g[hour * N + reach];
        let tol = 2e-2 * ad.abs().max(fd.abs()) + 1e-4;
        assert!(
            (fd - ad).abs() <= tol,
            "hour {hour} reach {reach}: adjoint {ad:.6} vs finite difference {fd:.6} (tol {tol:.2e})"
        );
        n_checked += 1;
    }
    assert_eq!(n_checked, 5);
}
