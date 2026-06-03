//! Project-controlled seeded initialization for `Linear` weights.
//!
//! These helpers replace `burn::nn::Initializer::{KaimingNormal, XavierNormal}`
//! in the KAN head so that:
//!   (a) DDRS head init is reproducible across runs at a fixed seed,
//!   (b) it uses the same `rand::rngs::StdRng` family as `rskan::KanLayer`,
//!       removing one cross-module RNG source as a parity-test variable.
//!
//! Formulas mirror PyTorch (`torch/nn/init.py:578` for Kaiming,
//! `torch/nn/init.py:469` for Xavier) so the distributions match DDR's
//! `nn/kan.py:45-46` calls element-for-element (modulo RNG bytes, per
//! spec C4).

use burn::module::Param;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;

/// Sample a Kaiming-normal weight matrix for a `Linear(in_dim, out_dim)` whose
/// downstream nonlinearity is ReLU.
///
/// Returns shape `[in_dim, out_dim]` to match burn's `Linear` weight layout
/// (burn stores weight as `[d_input, d_output]`, not PyTorch's `[out, in]`).
///
/// `std = sqrt(2) / sqrt(in_dim)` — equivalent to PyTorch's
/// `kaiming_normal_(mode="fan_in", nonlinearity="relu")`.
pub fn sample_kaiming_normal_relu(
    rng: &mut StdRng,
    in_dim: usize,
    out_dim: usize,
) -> Array2<f32> {
    let std = (2.0_f32 / in_dim as f32).sqrt();
    Array2::from_shape_fn((in_dim, out_dim), |_| {
        let v: f64 = rng.sample(rand_distr::StandardNormal);
        v as f32 * std
    })
}

/// Sample a Xavier-normal weight matrix for a Linear whose downstream
/// nonlinearity is sigmoid/tanh-like.
///
/// Returns shape `[in_dim, out_dim]` to match burn's `Linear` weight layout.
///
/// `std = gain * sqrt(2 / (in_dim + out_dim))` — equivalent to PyTorch's
/// `xavier_normal_(gain=gain)`.
pub fn sample_xavier_normal(
    rng: &mut StdRng,
    in_dim: usize,
    out_dim: usize,
    gain: f32,
) -> Array2<f32> {
    let std = gain * (2.0_f32 / (in_dim + out_dim) as f32).sqrt();
    Array2::from_shape_fn((in_dim, out_dim), |_| {
        let v: f64 = rng.sample(rand_distr::StandardNormal);
        v as f32 * std
    })
}

/// Promote an `ndarray::Array2<f32>` into a Burn `Param<Tensor<B, 2>>`.
pub fn to_param_weight<B: Backend>(
    arr: Array2<f32>,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let data = TensorData::new(arr.as_slice().unwrap().to_vec(), [rows, cols]);
    Param::from_tensor(Tensor::from_data(data, device))
}

/// Construct a zero-initialised bias tensor.
pub fn zero_bias_tensor<B: Backend>(
    dim: usize,
    device: &B::Device,
) -> Param<Tensor<B, 1>> {
    Param::from_tensor(Tensor::<B, 1>::zeros([dim], device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn kaiming_normal_relu_is_reproducible() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let a = sample_kaiming_normal_relu(&mut rng_a, 10, 21);
        let b = sample_kaiming_normal_relu(&mut rng_b, 10, 21);
        assert_eq!(a, b);
    }

    #[test]
    fn xavier_normal_is_reproducible() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let a = sample_xavier_normal(&mut rng_a, 21, 3, 0.1);
        let b = sample_xavier_normal(&mut rng_b, 21, 3, 0.1);
        assert_eq!(a, b);
    }

    #[test]
    fn kaiming_std_matches_formula_at_large_n() {
        let mut rng = StdRng::seed_from_u64(0);
        let arr = sample_kaiming_normal_relu(&mut rng, 100, 5_000);
        let mean = arr.mean().unwrap();
        let var: f32 = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap();
        let std = var.sqrt();
        let expected = (2.0_f32 / 100.0).sqrt();
        assert!(
            (std - expected).abs() < 1e-3,
            "std={std}, expected≈{expected}"
        );
    }

    #[test]
    fn xavier_std_matches_formula_at_large_n() {
        let mut rng = StdRng::seed_from_u64(0);
        let arr = sample_xavier_normal(&mut rng, 100, 5_000, 0.1);
        let mean = arr.mean().unwrap();
        let var: f32 = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap();
        let std = var.sqrt();
        let expected = 0.1 * (2.0_f32 / 5_100.0).sqrt();
        assert!(
            (std - expected).abs() < 1e-4,
            "std={std}, expected≈{expected}"
        );
    }
}
