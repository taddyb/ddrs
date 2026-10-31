//! Sanity tests for `compute_trapezoidal_geometry`.
//!
//! DDR has no dedicated geometry unit tests (the function is exercised
//! transitively via `test_mmc.py`). These tests verify the basic invariants
//! the routing code relies on: finite values, depth/area positivity, and that
//! velocity scales as `R^{2/3} / n` per Manning's equation.

mod common;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::geometry::compute_trapezoidal_geometry;

use common::{TestBackend, TestDevice};

#[test]
fn geometry_finite_and_positive() {
    let device = TestDevice::default();
    let n: Tensor<TestBackend, 1> = Tensor::from_floats([0.035_f32, 0.04, 0.05], &device);
    let p: Tensor<TestBackend, 1> = Tensor::from_floats([10.0_f32, 12.0, 15.0], &device);
    let q: Tensor<TestBackend, 1> = Tensor::from_floats([0.4_f32, 0.5, 0.6], &device);
    let qd: Tensor<TestBackend, 1> = Tensor::from_floats([5.0_f32, 10.0, 20.0], &device);
    let s: Tensor<TestBackend, 1> = Tensor::from_floats([1e-3_f32, 2e-3, 5e-4], &device);

    let geom = compute_trapezoidal_geometry(n, p, q, qd, s, 0.01, 0.01);
    for t in [
        geom.depth,
        geom.top_width,
        geom.bottom_width,
        geom.side_slope,
        geom.cross_sectional_area,
        geom.wetted_perimeter,
        geom.hydraulic_radius,
        geom.velocity,
    ] {
        for v in t.into_data().to_vec::<f32>().unwrap() {
            assert!(v.is_finite() && v > 0.0, "geometry value not positive-finite: {}", v);
        }
    }
}

/// Manning's: doubling `n` halves `velocity` (other inputs fixed at a regime
/// where lower-bound clamps don't bite).
#[test]
fn velocity_scales_inversely_with_manning_n() {
    let device = TestDevice::default();
    let p: Tensor<TestBackend, 1> = Tensor::from_floats([20.0_f32], &device);
    let q: Tensor<TestBackend, 1> = Tensor::from_floats([0.4_f32], &device);
    let qd: Tensor<TestBackend, 1> = Tensor::from_floats([50.0_f32], &device);
    let s: Tensor<TestBackend, 1> = Tensor::from_floats([1e-3_f32], &device);

    let n1: Tensor<TestBackend, 1> = Tensor::from_floats([0.03_f32], &device);
    let n2: Tensor<TestBackend, 1> = Tensor::from_floats([0.06_f32], &device);
    let v1 = compute_trapezoidal_geometry(n1, p.clone(), q.clone(), qd.clone(), s.clone(), 0.01, 0.01)
        .velocity
        .into_scalar();
    let v2 = compute_trapezoidal_geometry(n2, p, q, qd, s, 0.01, 0.01)
        .velocity
        .into_scalar();
    // Geometry shifts with n (depth comes out of an n-dependent inversion), so
    // the ratio isn't exactly 0.5 — but velocity must decrease meaningfully.
    assert!(v2 < v1, "doubling n should reduce velocity: v1={} v2={}", v1, v2);
    assert!(v2 / v1 < 0.95, "velocity reduction too small: ratio={}", v2 / v1);
}

#[test]
fn geometry_gradients_flow() {
    type AD = Autodiff<NdArray<f32>>;
    type Dev = <AD as burn::tensor::backend::BackendTypes>::Device;
    let device = Dev::default();
    let n: Tensor<AD, 1> = Tensor::from_floats([0.035_f32], &device).require_grad();
    let p: Tensor<AD, 1> = Tensor::from_floats([20.0_f32], &device);
    let q: Tensor<AD, 1> = Tensor::from_floats([0.4_f32], &device);
    let qd: Tensor<AD, 1> = Tensor::from_floats([20.0_f32], &device);
    let s: Tensor<AD, 1> = Tensor::from_floats([1e-3_f32], &device);

    let geom = compute_trapezoidal_geometry(n.clone(), p, q, qd, s, 0.01, 0.01);
    let grads = geom.velocity.sum().backward();
    let g = n.grad(&grads).expect("n must receive gradient");
    let gv = g.into_scalar();
    assert!(gv.is_finite() && gv != 0.0, "n grad should be finite and nonzero, got {}", gv);
    // Manning's: dv/dn < 0 (velocity decreases with n)
    assert!(gv < 0.0, "dv/dn should be negative, got {}", gv);
}

#[test]
fn trapezoid_from_depth_matches_inversion_round_trip() {
    use ddrs::geometry::trapezoid_from_depth;
    let device = TestDevice::default();
    for &qd in &[1.0_f32, 50.0, 800.0] {
        let t = |v: f32| Tensor::<TestBackend, 1>::from_floats([v], &device);
        let inv = compute_trapezoidal_geometry(t(0.035), t(21.0), t(0.5), t(qd), t(1e-3), 0.01, 0.01);
        let g = trapezoid_from_depth(t(21.0), t(0.5), inv.depth.clone(), 0.01);
        let pairs = [
            (g.top_width, inv.top_width),
            (g.area, inv.cross_sectional_area),
            (g.hydraulic_radius, inv.hydraulic_radius),
        ];
        for (a, b) in pairs {
            let (a, b) = (a.into_scalar(), b.into_scalar());
            assert!(((a - b) / b).abs() < 1e-6, "Q={qd}: {a} vs {b}");
        }
    }
}

#[test]
fn trapezoid_from_depth_is_differentiable_in_depth() {
    use ddrs::geometry::trapezoid_from_depth;
    type AD = Autodiff<NdArray<f32>>;
    type Dev = <AD as burn::tensor::backend::BackendTypes>::Device;
    let device = Dev::default();
    let t = |v: f32| Tensor::<AD, 1>::from_floats([v], &device);
    let d = t(1.0).require_grad();
    let g = trapezoid_from_depth(t(21.0), t(0.5), d.clone(), 0.01);
    let tw = g.top_width.clone().into_scalar();
    let grads = g.area.sum().backward();
    let ad = d.grad(&grads).expect("depth grad").into_scalar();

    let eps = 1e-3_f32;
    let area_at = |x: f32| {
        trapezoid_from_depth(t(21.0), t(0.5), t(x), 0.01).area.into_scalar()
    };
    let fd = (area_at(1.0 + eps) - area_at(1.0 - eps)) / (2.0 * eps);
    assert!(((ad - fd) / fd).abs() < 1e-3, "autodiff {ad} vs central diff {fd}");
    let closed = 1.125 * tw;
    assert!(((ad - closed) / closed).abs() < 1e-3, "autodiff {ad} vs closed form {closed}");
}
