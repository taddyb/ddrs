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
    let q_eps = q_spatial.clone() + 1e-6;

    // depth = ((Q · n · (q+1)) / (p · √s))^(3 / (5 + 3q))
    let numerator = discharge * n.clone() * (q_eps.clone() + 1.0);
    let denominator = p_spatial.clone() * slope.clone().sqrt();
    let ratio = numerator / (denominator + 1e-8);
    let exponent = (q_eps.clone() * 3.0 + 5.0).recip() * 3.0;
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
    let velocity = n.recip() * hydraulic_radius.clone().powf_scalar(2.0 / 3.0) * slope.sqrt();

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
