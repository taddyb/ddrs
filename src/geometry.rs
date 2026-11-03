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

/// Invert Manning's equation on the power-law section with `R = d`, then
/// build a tangent trapezoid on that depth (sections 4.3 and 12.6 of
/// `research/specs/2026-09-17-model-equations-reference.md`).
/// The two sections differ, so `Q != vA` on the result. The trapezoid
/// default keeps this; `params.geometry_form: power_law` uses one consistent
/// section instead.
///
/// The rest of the geometry comes from the Leopold & Maddock power law.
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
/// `research/specs/2026-09-12-stage-dependent-roughness-design.md`.
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

/// Cross-section of the Leopold & Maddock trapezoid at a given depth.
pub struct DepthGeometry<B: Backend> {
    pub top_width: Tensor<B, 1>,
    pub bottom_width: Tensor<B, 1>,
    pub side_slope: Tensor<B, 1>,
    pub area: Tensor<B, 1>,
    pub wetted_perimeter: Tensor<B, 1>,
    pub hydraulic_radius: Tensor<B, 1>,
}

/// Build the trapezoid for a known depth. Same formulas as S7 to S14 of
/// `src/routing/mmc_op.rs::forward_chain_inner` and as the tail of
/// [`compute_trapezoidal_geometry_gamma`], with `q_eps = q + 1e-6`.
/// The stage block and the MC block share this cross-section.
pub fn trapezoid_from_depth<B: Backend>(
    p_spatial: Tensor<B, 1>,
    q_spatial: Tensor<B, 1>,
    depth: Tensor<B, 1>,
    bottom_width_lb: f32,
) -> DepthGeometry<B> {
    let q_eps = q_spatial + 1e-6;
    let top_width = p_spatial * depth.clone().powf(q_eps.clone());
    let side_slope = (top_width.clone() * q_eps / (depth.clone() * 2.0)).clamp(0.5, 50.0);
    let bottom_width =
        (top_width.clone() - side_slope.clone() * depth.clone() * 2.0).clamp_min(bottom_width_lb);
    let area = (top_width.clone() + bottom_width.clone()) * depth.clone() / 2.0;
    let wetted_perimeter =
        bottom_width.clone() + depth * (side_slope.clone().powf_scalar(2.0) + 1.0).sqrt() * 2.0;
    let hydraulic_radius = area.clone() / wetted_perimeter.clone();
    DepthGeometry {
        top_width,
        bottom_width,
        side_slope,
        area,
        wetted_perimeter,
        hydraulic_radius,
    }
}

/// Closed-form `(dA/dd, dP/dd)` of the trapezoid built by [`trapezoid_from_depth`],
/// with the same `q_eps = q + 1e-6` and the same clamp branches: the side slope
/// is constant where its clamp is active, and the bottom width is constant where
/// its lower bound is active. `geom` must come from `trapezoid_from_depth` at
/// the same `(q, depth, bottom_width_lb)`. The result stays on the tape: the stage step
/// differentiates through it.
pub fn trapezoid_depth_derivatives<B: Backend>(
    q_spatial: Tensor<B, 1>,
    depth: Tensor<B, 1>,
    geom: &DepthGeometry<B>,
    bottom_width_lb: f32,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let q_eps = q_spatial + 1e-6;
    let tw = geom.top_width.clone();
    let ss = geom.side_slope.clone();
    let bw = geom.bottom_width.clone();
    let tw_d = q_eps.clone() * tw.clone() / depth.clone();
    // Branch masks from the unclamped values.
    let ss_raw = tw.clone() * q_eps / (depth.clone() * 2.0);
    let ss_active = ss_raw.clone().greater_elem(0.5).int() * ss_raw.clone().lower_elem(50.0).int();
    let ss_active = ss_active.float();
    // `tw_d / tw * depth - 1` equals `q_eps - 1` in exact arithmetic. It stays in this form
    // because `q_eps - 1` rounds differently, and the fused stage op matches this bit for bit.
    let ss_d = (ss_raw / depth.clone()) * (tw_d.clone() / tw.clone() * depth.clone() - 1.0) * ss_active;
    let bw_raw = tw.clone() - ss.clone() * depth.clone() * 2.0;
    let bw_active = bw_raw.greater_elem(bottom_width_lb).float();
    let bw_d = (tw_d.clone() - ss.clone() * 2.0 - depth.clone() * ss_d.clone() * 2.0) * bw_active;
    let area_d = (tw + bw) / 2.0 + depth.clone() * (tw_d + bw_d.clone()) / 2.0;
    let root = (ss.clone().powf_scalar(2.0) + 1.0).sqrt();
    let per_d = bw_d + root.clone() * 2.0 + depth * ss.clone() * ss_d * 2.0 / root;
    (area_d, per_d)
}

// ---------------------------------------------------------------------------
// Power-law channel section
// ---------------------------------------------------------------------------

/// Power-law section at a known depth: `T = p d^q`, `A = T d/(q+1)`,
/// `R = A/T = d/(q+1)`.
///
/// This is the reference for the stage block. Here `dA/dd = T` exactly, so the
/// section, the discharge `Q = v A`, and the celerity `c = dQ/dA` agree.
pub struct PowerLawSection<B: Backend> {
    pub top_width: Tensor<B, 1>,
    pub area: Tensor<B, 1>,
    pub hydraulic_radius: Tensor<B, 1>,
}

/// Power-law geometry from discharge, returned by [`compute_power_law_geometry`]
/// and [`compute_power_law_geometry_field`]. All fields are rank-1, one entry per reach.
pub struct PowerLawGeometry<B: Backend> {
    /// Natural log of the closed-form bracket, before the exponent is applied.
    pub ln_ratio: Tensor<B, 1>,
    /// `e = 3 / (5 + 3 q_eps + 3 gamma)`.
    pub exponent: Tensor<B, 1>,
    /// `clamp_min(exp(e * ln_ratio), depth_lb)`.
    pub depth: Tensor<B, 1>,
    pub top_width: Tensor<B, 1>,
    pub area: Tensor<B, 1>,
    pub hydraulic_radius: Tensor<B, 1>,
    /// `Q / A`. No clamp, also at the depth floor.
    pub velocity: Tensor<B, 1>,
    /// `beta = (5 + 3 q + 3 gamma) / (3 (q + 1))`. Celerity is `velocity * beta`.
    pub celerity_ratio: Tensor<B, 1>,
}

/// Build the power-law section for a known depth, with `q_eps = q + 1e-6`.
///
/// `top_width` uses the same op as the trapezoid, so `T` is bitwise equal on
/// both forms. Fused forward code calls this function, so its op order is fixed.
pub fn power_law_from_depth<B: Backend>(
    p_spatial: Tensor<B, 1>,
    q_spatial: Tensor<B, 1>,
    depth: Tensor<B, 1>,
) -> PowerLawSection<B> {
    let q_eps = q_spatial + 1e-6;
    let top_width = p_spatial * depth.clone().powf(q_eps.clone());
    let qp1 = q_eps + 1.0;
    let area = top_width.clone() * depth.clone() / qp1.clone();
    let hydraulic_radius = depth / qp1;
    PowerLawSection {
        top_width,
        area,
        hydraulic_radius,
    }
}

/// Closed-form `(dA/dd, dR/dd)` of the section from [`power_law_from_depth`].
/// `dA/dd` is `section.top_width` itself. `dR/dd` is `1 / (q_eps + 1)`.
/// The results stay on the tape.
pub fn power_law_depth_derivatives<B: Backend>(
    q_spatial: Tensor<B, 1>,
    section: &PowerLawSection<B>,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let q_eps = q_spatial + 1e-6;
    (section.top_width.clone(), (q_eps + 1.0).recip())
}

/// Depth, section, velocity and celerity ratio of the power-law channel, for a
/// scalar stage-roughness exponent `gamma` (`n(d) = n (d/d_ref)^-gamma`).
///
/// Depth solves Manning with `v = Q/A`:
/// `d = [Q n (q+1)^(5/3) d_ref^gamma / (p sqrt(s))]^(3 / (5 + 3q + 3 gamma))`.
/// It is computed in log space and floored at `depth_lb`. The `gamma ln(d_ref)`
/// term enters only when `gamma != 0` and `d_ref != 1`.
///
/// Because `A = T d/(q+1)`, `dA/dd = T`, and the celerity is
/// `c = v * celerity_ratio`, which equals `dQ/dA` exactly.
///
/// `clamp_min` is Burn's: at an exact tie with `depth_lb` the gradient passes.
/// Below the floor the depth is constant, so no gradient reaches `n`, `p`, `q`,
/// `Q` or `s` through it. The op order is a contract with the fused forward
/// and its hand-written backward, so do not reorder or fuse these ops.
///
/// Autodiff gradients need `discharge > 0`: `ln_ratio` uses `discharge.log()`,
/// and the gradient at `Q = 0` can be NaN (0/0) even though the forward value
/// is fine, because the depth floors.
#[allow(clippy::too_many_arguments)]
pub fn compute_power_law_geometry<B: Backend>(
    n: Tensor<B, 1>,
    p_spatial: Tensor<B, 1>,
    q_spatial: Tensor<B, 1>,
    discharge: Tensor<B, 1>,
    slope: Tensor<B, 1>,
    depth_lb: f32,
    gamma: f32,
    d_ref: f32,
) -> PowerLawGeometry<B> {
    let q_eps = q_spatial.clone() + 1e-6;
    let qp1 = q_eps.clone() + 1.0;
    let ln_ratio = discharge.clone().log() + n.log() + qp1.clone().log() * (5.0_f32 / 3.0_f32)
        - p_spatial.clone().log()
        - slope.log() * 0.5;
    let ln_ratio = if gamma != 0.0 && d_ref != 1.0 {
        ln_ratio + gamma * d_ref.ln()
    } else {
        ln_ratio
    };
    let five_plus_three_gamma = 5.0_f32 + 3.0_f32 * gamma;
    let exponent = (q_eps * 3.0 + five_plus_three_gamma).recip() * 3.0;
    let depth = (ln_ratio.clone() * exponent.clone())
        .exp()
        .clamp_min(depth_lb);
    let s = power_law_from_depth(p_spatial, q_spatial, depth.clone());
    let velocity = discharge / s.area.clone();
    let celerity_ratio = (exponent.clone() * qp1).recip();
    PowerLawGeometry {
        ln_ratio,
        exponent,
        depth,
        top_width: s.top_width,
        area: s.area,
        hydraulic_radius: s.hydraulic_radius,
        velocity,
        celerity_ratio,
    }
}

/// Same as [`compute_power_law_geometry`], with `gamma` as a per-reach tensor.
/// The same notes on the clamp tie rule, op order and `discharge > 0` apply.
#[allow(clippy::too_many_arguments)]
pub fn compute_power_law_geometry_field<B: Backend>(
    n: Tensor<B, 1>,
    p_spatial: Tensor<B, 1>,
    q_spatial: Tensor<B, 1>,
    discharge: Tensor<B, 1>,
    slope: Tensor<B, 1>,
    depth_lb: f32,
    gamma: Tensor<B, 1>,
    d_ref: f32,
) -> PowerLawGeometry<B> {
    let q_eps = q_spatial.clone() + 1e-6;
    let qp1 = q_eps.clone() + 1.0;
    let ln_ratio = discharge.clone().log() + n.log() + qp1.clone().log() * (5.0_f32 / 3.0_f32)
        - p_spatial.clone().log()
        - slope.log() * 0.5;
    let ln_ratio = if d_ref != 1.0 {
        ln_ratio + gamma.clone() * d_ref.ln()
    } else {
        ln_ratio
    };
    let exponent = (q_eps * 3.0 + gamma * 3.0 + 5.0).recip() * 3.0;
    let depth = (ln_ratio.clone() * exponent.clone())
        .exp()
        .clamp_min(depth_lb);
    let s = power_law_from_depth(p_spatial, q_spatial, depth.clone());
    let velocity = discharge / s.area.clone();
    let celerity_ratio = (exponent.clone() * qp1).recip();
    PowerLawGeometry {
        ln_ratio,
        exponent,
        depth,
        top_width: s.top_width,
        area: s.area,
        hydraulic_radius: s.hydraulic_radius,
        velocity,
        celerity_ratio,
    }
}
