//! A plain Burn-autodiff copy of one MC timestep, as a reference for the
//! hand-written backward of the fused op.
//!
//! `reference_timestep` computes S1..S28 with ordinary autodiff tensor ops,
//! using the same op order as `mmc_op::forward_chain_inner`. The zero-pattern
//! test (`tests/clamp_zero_parity.rs`) and the straddle tests compare its
//! gradient with the fused one.
//!
//! The reference does not call the fused forward, any `TimestepOp`, or
//! `sparse::spmv_primitive`. It may call pure forward helpers that are plain
//! Burn ops: `mmc_op::depth_exponent`, the power-law geometry functions and
//! `mmc_op::POSITIVITY_DELTA`. Burn's `clamp` and `clamp_min` pass the
//! gradient at an exact tie, and `min_pair` sends a tie to the first operand.
//! That is the tie rule the fused backward follows.
//!
//! Left out on purpose: leakance, reservoirs, stage routing and `ddr_match`.

use std::sync::Arc;

use burn::backend::Autodiff;
use burn::tensor::{backend::Backend, Tensor};

use crate::config::{Config, GeometryForm, SparseSolver};
use crate::geometry::{compute_power_law_geometry, compute_power_law_geometry_field};
use crate::routing::mmc::DT_SECONDS;
use crate::routing::mmc_op::{depth_exponent, POSITIVITY_DELTA};
use crate::sparse::{self, AValuesAssembler, CsrPattern};

/// One MC timestep with Burn autodiff ops. Returns `q_next` for the step.
///
/// Panics on `ddr_match`. `x_storage_at` is unused: with `ddr_match` off the
/// Cunge X replaces it. It stays in the signature to mirror
/// `mmc_op::timestep_forward`. `gamma_at: Some` overrides the scalar stage
/// roughness exponent, as in the fused forward.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn reference_timestep<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    assembler: &AValuesAssembler<I>,
    n_at: Tensor<Autodiff<I>, 1>,
    q_spatial_at: Tensor<Autodiff<I>, 1>,
    p_spatial_at: Tensor<Autodiff<I>, 1>,
    q_t_at: Tensor<Autodiff<I>, 1>,
    q_prime_t_at: Tensor<Autodiff<I>, 1>,
    length_at: Tensor<Autodiff<I>, 1>,
    slope_at: Tensor<Autodiff<I>, 1>,
    x_storage_at: Tensor<Autodiff<I>, 1>,
    gamma_at: Option<Tensor<Autodiff<I>, 1>>,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    type T<I> = Tensor<Autodiff<I>, 1>;
    let _ = x_storage_at;
    assert!(!cfg.params.ddr_match, "reference_timestep does not support ddr_match");

    let dt = DT_SECONDS;
    let bottom_width_lb = cfg.params.attribute_minimums.bottom_width;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let discharge_lb = cfg.params.attribute_minimums.discharge;
    let use_cuda = cfg.params.sparse_solver == SparseSolver::Cuda;
    let (gamma, d_ref) = cfg.params.stage_roughness_params();
    let enforce_pos = cfg.params.enforce_positivity;

    // S1
    let q_eps = q_spatial_at.clone() + 1e-6_f32;

    // S2..S17: the channel section. Gives top width, celerity and the rest.
    let (top_width, celerity): (T<I>, T<I>) = match cfg.params.geometry_form {
        GeometryForm::Trapezoid => {
            // S2
            let numerator = q_t_at.clone() * n_at.clone() * (q_eps.clone() + 1.0);
            let numerator = if gamma != 0.0 && d_ref != 1.0 {
                numerator * d_ref.powf(gamma)
            } else {
                numerator
            };
            // S3
            let denominator = p_spatial_at.clone() * slope_at.clone().sqrt() + 1e-8_f32;
            // S4
            let ratio = numerator / denominator;
            // S5
            let exponent = depth_exponent(q_eps.clone(), gamma, gamma_at.as_ref());
            // S6
            let depth = ratio.powf(exponent).clamp_min(depth_lb);
            // S7
            let top_width = p_spatial_at.clone() * depth.clone().powf(q_eps.clone());
            // S8
            let side_slope_raw = top_width.clone() * q_eps.clone() / (depth.clone() * 2.0);
            // S9
            let side_slope = side_slope_raw.clamp(0.5, 50.0);
            // S10
            let bw_raw = top_width.clone() - side_slope.clone() * depth.clone() * 2.0;
            // S11
            let bottom_width = bw_raw.clamp_min(bottom_width_lb);
            // S12
            let area = (top_width.clone() + bottom_width.clone()) * depth.clone() / 2.0;
            // S13
            let wp = bottom_width
                + depth.clone() * (side_slope.clone().powf_scalar(2.0) + 1.0).sqrt() * 2.0;
            // S14
            let hyd_radius = area.clone() / wp.clone();
            // S15
            let n_recip = match &gamma_at {
                Some(g) => n_at.clone().recip() * (depth.clone() / d_ref).powf(g.clone()),
                None if gamma != 0.0 => {
                    n_at.clone().recip() * (depth.clone() / d_ref).powf_scalar(gamma)
                }
                None => n_at.clone().recip(),
            };
            let velocity_un =
                n_recip * hyd_radius.powf_scalar(2.0 / 3.0) * slope_at.clone().sqrt();
            // S16
            let velocity_cl = velocity_un.clamp(velocity_lb, 15.0);
            // S17 (ddr_match false): exact trapezoidal celerity.
            let root = (side_slope.powf_scalar(2.0) + 1.0).sqrt();
            let beta = -(area.clone() * root) / (top_width.clone() * wp) * (4.0 / 3.0) + (5.0 / 3.0);
            let beta = match &gamma_at {
                Some(g) => beta + area * g.clone() / (top_width.clone() * depth),
                None if gamma != 0.0 => beta + area * gamma / (top_width.clone() * depth),
                None => beta,
            };
            (top_width, velocity_cl * beta)
        }
        GeometryForm::PowerLaw => {
            let g = match &gamma_at {
                Some(gt) => compute_power_law_geometry_field::<Autodiff<I>>(
                    n_at.clone(),
                    p_spatial_at.clone(),
                    q_spatial_at.clone(),
                    q_t_at.clone(),
                    slope_at.clone(),
                    depth_lb,
                    gt.clone(),
                    d_ref,
                ),
                None => compute_power_law_geometry::<Autodiff<I>>(
                    n_at.clone(),
                    p_spatial_at.clone(),
                    q_spatial_at.clone(),
                    q_t_at.clone(),
                    slope_at.clone(),
                    depth_lb,
                    gamma,
                    d_ref,
                ),
            };
            let velocity_cl = g.velocity.clamp(velocity_lb, 15.0);
            (g.top_width, velocity_cl * g.celerity_ratio)
        }
    };

    // S18: K = L / c. S18': the K floor under `enforce_positivity`.
    let k_raw = length_at.clone() / celerity.clone();
    let k_muskingum = if enforce_pos {
        k_raw.clamp_min(dt * (1.0 + POSITIVITY_DELTA) / 2.0)
    } else {
        k_raw
    };

    // S19: Cunge X, clamped to [0, 0.5]. S19': the stability caps.
    let w = q_t_at.clone()
        / (top_width * slope_at * celerity * length_at + 1e-12);
    let x_cunge = ((-w + 1.0) * 0.5).clamp(0.0, 0.5);
    let x_eff = if enforce_pos {
        let cr = k_muskingum.clone().recip() * dt;
        let hi_a = cr.clone() * (0.5 * (1.0 - POSITIVITY_DELTA));
        let hi_b = (-cr * 0.5 + 1.0) * (1.0 - POSITIVITY_DELTA);
        x_cunge.min_pair(hi_a).min_pair(hi_b)
    } else {
        x_cunge
    };

    // S20..S23: Muskingum coefficients.
    let one_minus_x = -x_eff.clone() + 1.0;
    let two_k = k_muskingum * 2.0;
    let two_kx = two_k.clone() * x_eff;
    let two_k_1mx = two_k * one_minus_x;
    let denom = two_k_1mx.clone() + dt;
    let c1 = (-two_kx.clone() + dt) / denom.clone();
    let c2 = (two_kx + dt) / denom.clone();
    let c3 = (two_k_1mx - dt) / denom.clone();
    let c4 = denom.recip() * (2.0 * dt);

    // S24: i_t = N q_t, with the autodiff SpMV.
    let i_t = assembler.spmv(q_t_at.clone());
    // S25
    let b_rhs = c2 * i_t + c3 * q_t_at + c4 * q_prime_t_at;
    // S26
    let a_values = assembler.assemble(c1);
    // S27
    let x_sol = sparse::triangular_csr_solve::<I>(pattern, a_values, b_rhs, use_cuda);
    // S28
    x_sol.clamp_min(discharge_lb)
}
