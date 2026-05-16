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
