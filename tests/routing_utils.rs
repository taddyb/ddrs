//! Rust mirror of `tests/routing/test_routing_utils.py`.
//!
//! Covers `denormalize` (linear, log-space, gradient flow) and the
//! lower-triangular solve including the canonical 3×3 fixture
//! `L = [[2,0,0],[1,3,0],[0,1,4]] b = [2,7,13] → x = [1,2,2.75]`.

mod common;

use approx::assert_relative_eq;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::routing::{denormalize, triangular_solve_lower};

use common::{TestBackend, TestDevice};

#[test]
fn denormalize_linear_midpoint() {
    let device = TestDevice::default();
    let v: Tensor<TestBackend, 1> = Tensor::from_floats([0.5_f32], &device);
    let r = denormalize(v, [0.0, 10.0], false);
    let val = r.into_scalar();
    assert_relative_eq!(val, 5.0, epsilon = 1e-5);
}

#[test]
fn denormalize_linear_bounds() {
    let device = TestDevice::default();
    let lo = denormalize::<TestBackend>(Tensor::from_floats([0.0_f32], &device), [0.0, 10.0], false)
        .into_scalar();
    let hi = denormalize::<TestBackend>(Tensor::from_floats([1.0_f32], &device), [0.0, 10.0], false)
        .into_scalar();
    assert_relative_eq!(lo, 0.0, epsilon = 1e-5);
    assert_relative_eq!(hi, 10.0, epsilon = 1e-5);
}

#[test]
fn denormalize_log_space_geometric_mean() {
    let device = TestDevice::default();
    let r = denormalize::<TestBackend>(
        Tensor::from_floats([0.5_f32], &device),
        [1.0, 100.0],
        true,
    )
    .into_scalar();
    // geometric mean of 1 and 100 = 10  (Python tolerates atol=0.5)
    assert!((r - 10.0).abs() < 0.5, "log midpoint = {}", r);
}

#[test]
fn denormalize_log_space_bounds() {
    let device = TestDevice::default();
    let lo = denormalize::<TestBackend>(Tensor::from_floats([0.0_f32], &device), [1.0, 100.0], true)
        .into_scalar();
    let hi = denormalize::<TestBackend>(Tensor::from_floats([1.0_f32], &device), [1.0, 100.0], true)
        .into_scalar();
    assert!((lo - 1.0).abs() < 0.01);
    assert!((hi - 100.0).abs() < 0.1);
}

#[test]
fn denormalize_vector_input() {
    let device = TestDevice::default();
    let v: Tensor<TestBackend, 1> = Tensor::from_floats([0.0, 0.25, 0.5, 0.75, 1.0], &device);
    let r = denormalize(v, [0.0, 10.0], false);
    let data: Vec<f32> = r.into_data().to_vec().unwrap();
    let expected = [0.0, 2.5, 5.0, 7.5, 10.0];
    for (a, b) in data.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-5);
    }
}

#[test]
fn denormalize_preserves_gradient() {
    type AD = Autodiff<NdArray<f32>>;
    let device = <AD as burn::tensor::backend::BackendTypes>::Device::default();
    let v: Tensor<AD, 1> = Tensor::from_floats([0.5_f32], &device).require_grad();
    let y = denormalize(v.clone(), [0.0, 10.0], false);
    let grads = y.sum().backward();
    let g = v.grad(&grads).expect("gradient should propagate").into_scalar();
    assert!(g.is_finite());
    // d/dx (x · 10) = 10
    assert_relative_eq!(g, 10.0, epsilon = 1e-5);
}

#[test]
fn triangular_solve_identity() {
    let device = TestDevice::default();
    let n = 5usize;
    let eye: Tensor<TestBackend, 2> = Tensor::eye(n, &device);
    let b: Tensor<TestBackend, 1> = Tensor::from_floats([1.0_f32, 2.0, 3.0, 4.0, 5.0], &device);
    let x = triangular_solve_lower(eye, b.clone());
    let xv: Vec<f32> = x.into_data().to_vec().unwrap();
    let bv: Vec<f32> = b.into_data().to_vec().unwrap();
    for (a, b) in xv.iter().zip(bv.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-5);
    }
}

#[test]
fn triangular_solve_known_system() {
    // L = [[2,0,0],[1,3,0],[0,1,4]],  b = [2,7,13]  →  x = [1, 2, 2.75]
    let device = TestDevice::default();
    let l: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(
        [2.0_f32, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 4.0],
        &device,
    )
    .reshape([3, 3]);
    let b: Tensor<TestBackend, 1> = Tensor::from_floats([2.0_f32, 7.0, 13.0], &device);
    let x = triangular_solve_lower(l, b);
    let v: Vec<f32> = x.into_data().to_vec().unwrap();
    assert_relative_eq!(v[0], 1.0, epsilon = 1e-5);
    assert_relative_eq!(v[1], 2.0, epsilon = 1e-5);
    assert_relative_eq!(v[2], 2.75, epsilon = 1e-5);
}

#[test]
fn triangular_solve_backward_is_finite() {
    type AD = Autodiff<NdArray<f32>>;
    let device = <AD as burn::tensor::backend::BackendTypes>::Device::default();
    let l: Tensor<AD, 2> = Tensor::<AD, 1>::from_floats(
        [2.0_f32, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 4.0],
        &device,
    )
    .reshape([3, 3])
    .require_grad();
    let b: Tensor<AD, 1> = Tensor::from_floats([2.0_f32, 7.0, 13.0], &device).require_grad();

    let x = triangular_solve_lower(l.clone(), b.clone());
    let grads = x.sum().backward();

    let gl = l.grad(&grads).expect("L gradient");
    let gb = b.grad(&grads).expect("b gradient");
    for v in gl.into_data().to_vec::<f32>().unwrap() {
        assert!(v.is_finite());
    }
    for v in gb.into_data().to_vec::<f32>().unwrap() {
        assert!(v.is_finite());
    }
}

// ---------------------------------------------------------------------------
// Log-space lower-bound regression (2026-09-15)
//
// `denormalize`'s log branch used to compute its lower bound as `(lo + 1e-6).ln()`
// unconditionally. The guard exists for `lo == 0`, but it swamps any lower bound
// that is small next to 1e-6. For `k_d = [1e-8, 1e-6]` the guarded lower bound is
// 1.01e-6, which sits ABOVE the upper bound: the map inverted and spanned 1 %
// instead of the intended two orders of magnitude, so `K_D` was a frozen constant
// rather than a learned parameter. See
// `.claude/skills/ddrs-dev/references/traps.md` T17.
// ---------------------------------------------------------------------------

/// The `K_D` box maps endpoint-to-endpoint across its full span.
///
/// Under the old guard this returned 1.01e-6 at `u = 0` and 1.00e-6 at `u = 1`.
#[test]
fn denormalize_log_space_tiny_lower_bound_spans_full_box() {
    let device = TestDevice::default();
    let range = [1e-8_f32, 1e-6];

    let lo = denormalize::<TestBackend>(Tensor::from_floats([0.0_f32], &device), range, true)
        .into_scalar();
    let hi = denormalize::<TestBackend>(Tensor::from_floats([1.0_f32], &device), range, true)
        .into_scalar();

    assert_relative_eq!(lo, 1e-8, max_relative = 1e-5);
    assert_relative_eq!(hi, 1e-6, max_relative = 1e-5);

    // Two orders of magnitude of span, not the 1 % band the old guard produced.
    assert!(
        (hi / lo - 100.0).abs() < 1e-2,
        "log-space box [1e-8, 1e-6] should span 100x, got {} / {} = {}",
        hi,
        lo,
        hi / lo
    );
}

/// The map over the `K_D` box is strictly increasing across `[0, 1]`.
///
/// The old guard produced a NEGATIVE log span here, so a larger neural-net
/// output yielded a smaller physical `K_D`.
#[test]
fn denormalize_log_space_tiny_lower_bound_is_monotonic() {
    let device = TestDevice::default();
    let range = [1e-8_f32, 1e-6];
    let u: Tensor<TestBackend, 1> =
        Tensor::from_floats([0.0_f32, 0.2, 0.4, 0.6, 0.8, 1.0], &device);
    let out: Vec<f32> = denormalize(u, range, true).into_data().to_vec().unwrap();

    for w in out.windows(2) {
        assert!(
            w[1] > w[0],
            "denormalize must be increasing on [1e-8, 1e-6], got {:?}",
            out
        );
    }
    // Everything stays inside the declared box.
    for v in &out {
        assert!(
            *v >= 1e-8 * 0.999 && *v <= 1e-6 * 1.001,
            "value {} escaped the box [1e-8, 1e-6]: {:?}",
            v,
            out
        );
    }
}

/// A `lo == 0` box still needs the epsilon guard, and keeps its old mapping.
///
/// `ln(0)` is `-inf`, so without the guard every output would be 0 or NaN.
#[test]
fn denormalize_log_space_zero_lower_bound_is_still_guarded() {
    let device = TestDevice::default();
    let range = [0.0_f32, 1.0];
    let us = [0.0_f32, 0.5, 1.0];
    let u: Tensor<TestBackend, 1> = Tensor::from_floats(us, &device);
    let out: Vec<f32> = denormalize(u, range, true).into_data().to_vec().unwrap();

    for v in &out {
        assert!(v.is_finite(), "lo == 0 must stay finite, got {:?}", out);
        assert!(*v > 0.0, "lo == 0 must stay positive, got {:?}", out);
    }
    // Unchanged from the pre-fix formula: exp(u · (ln(1) − ln(1e-6)) + ln(1e-6)).
    let log_min = 1e-6_f32.ln();
    for (i, u_i) in us.iter().enumerate() {
        let expected = (u_i * (0.0 - log_min) + log_min).exp();
        assert_relative_eq!(out[i], expected, max_relative = 1e-5);
    }
}

/// `p_spatial`'s `[1, 200]` box is unchanged to f32 tolerance.
///
/// This is the only log-space parameter enabled by default, and it is what the
/// `compare_ddr_sandbox` invariant rides on. Dropping the 1e-6 shifts values by
/// under one part in a million.
#[test]
fn denormalize_log_space_p_spatial_unchanged() {
    let device = TestDevice::default();
    let range = [1.0_f32, 200.0];
    let us = [0.0_f32, 0.25, 0.5, 0.65, 0.75, 1.0];
    let u: Tensor<TestBackend, 1> = Tensor::from_floats(us, &device);
    let out: Vec<f32> = denormalize(u, range, true).into_data().to_vec().unwrap();

    // Pre-fix formula, evaluated in f64 to isolate the guard from f32 rounding.
    let log_min_old = (1.0_f64 + 1e-6).ln();
    let log_max = 200.0_f64.ln();
    for (i, u_i) in us.iter().enumerate() {
        let old = ((*u_i as f64) * (log_max - log_min_old) + log_min_old).exp();
        let rel = ((out[i] as f64) - old).abs() / old;
        assert!(
            rel < 1e-5,
            "p_spatial at u={} moved by {} (new {}, old {})",
            u_i,
            rel,
            out[i],
            old
        );
    }
    // Endpoints still land on the box.
    assert_relative_eq!(out[0], 1.0, max_relative = 1e-5);
    assert_relative_eq!(out[5], 200.0, max_relative = 1e-4);
}

/// Cross the `sigmoid -> denormalize` seam with the real KAN head.
///
/// The existing leakance gradient checks feed physical values straight into the
/// routing forward, so they are structurally blind to a collapsed parameter box.
/// This drives a real `KanHead` (which ends in a sigmoid) and denormalizes its
/// `K_D` output with the shipped `[1e-8, 1e-6]` range.
///
/// Both assertions fail under the old guard regardless of how widely the head's
/// sigmoid outputs happen to spread: values exceeded the box's upper bound
/// (up to 1.01e-6 > 1e-6), and the ordering was reversed.
#[test]
fn kan_head_sigmoid_denormalizes_into_the_k_d_box() {
    use burn::tensor::Distribution;
    use ddrs::nn::{KanHead, KanHeadConfig};

    let device = TestDevice::default();
    let cfg = KanHeadConfig::new(
        (0..5).map(|i| format!("attr_{i}")).collect(),
        vec!["K_D".to_string()],
        42,
    )
    .with_hidden_size(11)
    .with_num_hidden_layers(1);
    let head: KanHead<TestBackend> = cfg.init::<TestBackend>(&device);

    let attrs: Tensor<TestBackend, 2> = Tensor::random([32, 5], Distribution::Default, &device);
    let raw = head.forward(attrs)["K_D"].clone();
    let u: Vec<f32> = raw.clone().into_data().to_vec().unwrap();

    let k_d_range = [1e-8_f32, 1e-6];
    let phys: Vec<f32> = denormalize(raw, k_d_range, true)
        .into_data()
        .to_vec()
        .unwrap();

    // 1. Every denormalized K_D lands inside the declared box.
    for (i, v) in phys.iter().enumerate() {
        assert!(
            *v >= 1e-8 * 0.999 && *v <= 1e-6 * 1.001,
            "K_D[{}] = {} escaped the box [1e-8, 1e-6] (sigmoid output {})",
            i,
            v,
            u[i]
        );
    }

    // 2. Ordering survives the map: the largest sigmoid output must give the
    //    largest physical K_D. The old guard inverted this.
    let arg_max_u = (0..u.len()).max_by(|a, b| u[*a].total_cmp(&u[*b])).unwrap();
    let arg_min_u = (0..u.len()).min_by(|a, b| u[*a].total_cmp(&u[*b])).unwrap();
    assert!(
        phys[arg_max_u] > phys[arg_min_u],
        "denormalize inverted the head's ordering: u {} -> {} but u {} -> {}",
        u[arg_max_u],
        phys[arg_max_u],
        u[arg_min_u],
        phys[arg_min_u]
    );
}
