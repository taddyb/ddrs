//! Pure trapezoidal channel geometry.
//!
//! Direct port of `~/projects/ddr/src/ddr/geometry/trapezoidal.py`.
//! Computes depth, width, area, hydraulic radius, and Manning's velocity from
//! `n`, Leopold & Maddock `(p, q)`, discharge `Q`, and channel slope.

use burn::tensor::{backend::Backend, Tensor};

/// Trapezoidal geometry returned by [`compute_trapezoidal_geometry`].
///
/// All fields are rank-1 tensors of length `N` (reaches).
pub struct TrapezoidalGeometry<B: Backend> {
    pub depth: Tensor<B, 1>,
    pub top_width: Tensor<B, 1>,
    pub bottom_width: Tensor<B, 1>,
    pub side_slope: Tensor<B, 1>,
    pub cross_sectional_area: Tensor<B, 1>,
    pub wetted_perimeter: Tensor<B, 1>,
    pub hydraulic_radius: Tensor<B, 1>,
    pub velocity: Tensor<B, 1>,
}

/// Invert Manning's equation for a trapezoidal section, then derive the rest
/// of the geometry from the Leopold & Maddock power law.
///
/// Matches the reference Python implementation including the `q + 1e-6` epsilon,
/// the `(3 / (5 + 3q))` exponent on depth, and the `[0.5, 50]` side-slope clamp.
pub fn compute_trapezoidal_geometry<B: Backend>(
    n: Tensor<B, 1>,
    p_spatial: Tensor<B, 1>,
    q_spatial: Tensor<B, 1>,
    discharge: Tensor<B, 1>,
    slope: Tensor<B, 1>,
    depth_lb: f32,
    bottom_width_lb: f32,
) -> TrapezoidalGeometry<B> {
    compute_trapezoidal_geometry_gamma(
        n,
        p_spatial,
        q_spatial,
        discharge,
        slope,
        depth_lb,
        bottom_width_lb,
        0.0,
        1.0,
    )
}

/// [`compute_trapezoidal_geometry`] with stage-dependent Manning roughness.
///
/// ```text
///   n(d) = n_0 · (d / d_ref)^(−gamma)
/// ```
///
/// Substituting that into the Manning inversion moves one power of depth across,
/// so the depth exponent becomes `3/(5 + 3q + 3·gamma)` and the numerator picks
/// up a constant `d_ref^gamma`. `gamma = 0` reproduces the historical geometry
/// exactly: `5.0 + 3.0*0.0` is `5.0` and `d_ref^0` is `1.0`, both without
/// rounding.
///
/// **This does not compute celerity**, which also gains a `gamma · A/(T·d)` term
/// — see `src/routing/mmc_op.rs` S17 and
/// `docs/superpowers/specs/2026-09-12-stage-dependent-roughness-design.md`.
#[allow(clippy::too_many_arguments)]
pub fn compute_trapezoidal_geometry_gamma<B: Backend>(
    n: Tensor<B, 1>,
    p_spatial: Tensor<B, 1>,
    q_spatial: Tensor<B, 1>,
    discharge: Tensor<B, 1>,
    slope: Tensor<B, 1>,
    depth_lb: f32,
    bottom_width_lb: f32,
    gamma: f32,
    d_ref: f32,
) -> TrapezoidalGeometry<B> {
    let q_eps = q_spatial.clone() + 1e-6;

    // depth = ((Q · n · (q+1) · d_ref^gamma) / (p · √s))^(3 / (5 + 3q + 3·gamma))
    let numerator = discharge * n.clone() * (q_eps.clone() + 1.0);
    let numerator = if gamma != 0.0 && d_ref != 1.0 {
        numerator * d_ref.powf(gamma)
    } else {
        numerator
    };
    let denominator = p_spatial.clone() * slope.clone().sqrt();
    let ratio = numerator / (denominator + 1e-8);
    let exponent = (q_eps.clone() * 3.0 + (5.0 + 3.0 * gamma)).recip() * 3.0;
    let depth = ratio.powf(exponent).clamp_min(depth_lb);

    // top_width = p · depth^q
    let top_width = p_spatial * depth.clone().powf(q_eps.clone());

    // side_slope (z:1 H:V): clamped to [0.5, 50]
    let side_slope = (top_width.clone() * q_eps.clone() / (depth.clone() * 2.0)).clamp(0.5, 50.0);

    // bottom_width = clamp(top_width − 2·side_slope·depth, btm_lb)
    let bottom_width = (top_width.clone() - side_slope.clone() * depth.clone() * 2.0)
        .clamp_min(bottom_width_lb);

    // area = (TW + BW) · d / 2
    let area = (top_width.clone() + bottom_width.clone()) * depth.clone() / 2.0;

    // wetted_perimeter = BW + 2·d·√(1 + side_slope²)
    let wetted_perimeter =
        bottom_width.clone() + depth.clone() * (side_slope.clone().powf_scalar(2.0) + 1.0).sqrt() * 2.0;

    // R = area / wetted_perimeter
    let hydraulic_radius = area.clone() / wetted_perimeter.clone();

    // v = (1/n) · R^(2/3) · √s
    // v = (1/n(d)) · R^(2/3) · √s, and with stage-dependent roughness
    // 1/n(d) = (1/n_0)·(d/d_ref)^gamma. Forgetting this factor leaves the
    // velocity exponent at Manning's 2f/3 while the depth exponent moves,
    // which is silently wrong rather than loudly wrong.
    let inv_n = if gamma != 0.0 {
        n.recip() * (depth.clone() / d_ref).powf_scalar(gamma)
    } else {
        n.recip()
    };
    let velocity = inv_n * hydraulic_radius.clone().powf_scalar(2.0 / 3.0) * slope.sqrt();

    TrapezoidalGeometry {
        depth,
        top_width,
        bottom_width,
        side_slope,
        cross_sectional_area: area,
        wetted_perimeter,
        hydraulic_radius,
        velocity,
    }
}
