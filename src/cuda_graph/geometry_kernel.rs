//! SP-10 SPIKE: fused `#[cube]` kernel for trapezoidal geometry chain S1..S14.
//!
//! Implements the pre-SpMV dense geometry block of `forward_chain_inner`
//! (`src/routing/mmc_op.rs:580-608`) as a single launched cube kernel.
//! Intermediates (q_eps, numerator, denominator, ratio, exponent, area, wp)
//! live in GPU registers; only the five named outputs are written to global
//! memory. This eliminates the cubecl heap allocations from the captured
//! region for this slice of the chain, which is the de-risking question
//! SP-10 needs answered before committing to a full fused-chain rewrite.
//!
//! Mirrors mmc_op.rs:580-608 exactly (same arithmetic, same constants).
//!
//! Outputs: 5 buffers of length n (`depth`, `top_width`, `side_slope`,
//! `bottom_width`, `hydraulic_radius`). Inputs: 5 buffers of length n
//! (`qsp`, `qt`, `psp`, `slope`, `n`) + 2 scalar lower bounds (`depth_lb`,
//! `bottom_width_lb`).
//!
//! `powf` with a runtime exponent is expressed as `exp(exponent * ln(base))`
//! since cubecl's `F::powf` is also available but the explicit form documents
//! the trapezoidal-geometry math better.

use cubecl::prelude::*;

/// Fused S1..S14 of the per-timestep dense chain.
///
/// One thread per segment (rank-1, n threads). Each thread reads its row of
/// the 5 inputs, computes the 8 intermediates in registers, writes the 5
/// named outputs.
#[cube(launch)]
pub fn fused_geometry_s1_s14<F: Float>(
    qsp: &Tensor<F>,
    qt: &Tensor<F>,
    psp: &Tensor<F>,
    slope: &Tensor<F>,
    n: &Tensor<F>,
    depth_out: &mut Tensor<F>,
    top_width_out: &mut Tensor<F>,
    side_slope_out: &mut Tensor<F>,
    bottom_width_out: &mut Tensor<F>,
    hydraulic_radius_out: &mut Tensor<F>,
) {
    // Lower bounds match `AttributeMinimums::default()` in config.rs.
    // Hardcoded inline via `F::new(...)` at each use site below.
    if ABSOLUTE_POS >= qsp.len() {
        terminate!();
    }
    let i = ABSOLUTE_POS;

    let qsp_i = qsp[i];
    let qt_i = qt[i];
    let psp_i = psp[i];
    let slope_i = slope[i];
    let n_i = n[i];

    // S1: q_eps = qsp + 1e-6
    let q_eps = qsp_i + F::new(1e-6);

    // S2: numerator = qt * n * (q_eps + 1.0)
    let numerator = qt_i * n_i * (q_eps + F::new(1.0));

    // S3: denominator = psp * sqrt(slope) + 1e-8
    let denominator = psp_i * slope_i.sqrt() + F::new(1e-8);

    // S4: ratio = numerator / denominator
    let ratio = numerator / denominator;

    // S5: exponent = (q_eps * 3 + 5).recip() * 3
    let exponent = (q_eps * F::new(3.0) + F::new(5.0)).recip() * F::new(3.0);

    // S6: depth = clamp_min(ratio^exponent, depth_lb)
    //     ratio^exponent = exp(exponent * ln(ratio))
    let depth_raw = (exponent * ratio.ln()).exp();
    let depth = depth_raw.max(F::new(0.01));

    // S7: top_width = psp * depth^q_eps
    let top_width = psp_i * (q_eps * depth.ln()).exp();

    // S8: side_slope_raw = top_width * q_eps / (depth * 2)
    let side_slope_raw = top_width * q_eps / (depth * F::new(2.0));

    // S9: side_slope = clamp(side_slope_raw, 0.5, 50.0)
    let side_slope = side_slope_raw.clamp(F::new(0.5), F::new(50.0));

    // S10: bw_raw = top_width - side_slope * depth * 2
    let bw_raw = top_width - side_slope * depth * F::new(2.0);

    // S11: bottom_width = clamp_min(bw_raw, bottom_width_lb)
    let bottom_width = bw_raw.max(F::new(0.01));

    // S12: area = (top_width + bottom_width) * depth / 2
    let area = (top_width + bottom_width) * depth / F::new(2.0);

    // S13: wp = bottom_width + depth * sqrt(side_slope^2 + 1) * 2
    let wp = bottom_width
        + depth * (side_slope * side_slope + F::new(1.0)).sqrt() * F::new(2.0);

    // S14: hyd_radius = area / wp
    let hyd_radius = area / wp;

    depth_out[i] = depth;
    top_width_out[i] = top_width;
    side_slope_out[i] = side_slope;
    bottom_width_out[i] = bottom_width;
    hydraulic_radius_out[i] = hyd_radius;
}

/// CPU-side reference for the same chain (numerical verification of the
/// fused kernel output). Mirrors S1..S14 line-by-line.
pub fn cpu_reference_s1_s14(
    qsp: &[f32],
    qt: &[f32],
    psp: &[f32],
    slope: &[f32],
    n: &[f32],
    depth_lb: f32,
    bottom_width_lb: f32,
) -> [Vec<f32>; 5] {
    let len = qsp.len();
    let mut depth = vec![0.0_f32; len];
    let mut top_width = vec![0.0_f32; len];
    let mut side_slope = vec![0.0_f32; len];
    let mut bottom_width = vec![0.0_f32; len];
    let mut hydraulic_radius = vec![0.0_f32; len];

    for i in 0..len {
        let q_eps = qsp[i] + 1e-6_f32;
        let numerator = qt[i] * n[i] * (q_eps + 1.0);
        let denominator = psp[i] * slope[i].sqrt() + 1e-8_f32;
        let ratio = numerator / denominator;
        let exponent = (q_eps * 3.0 + 5.0).recip() * 3.0;
        let depth_raw = ratio.powf(exponent);
        let d = depth_raw.max(depth_lb);
        let tw = psp[i] * d.powf(q_eps);
        let ss_raw = tw * q_eps / (d * 2.0);
        let ss = ss_raw.clamp(0.5, 50.0);
        let bw_raw = tw - ss * d * 2.0;
        let bw = bw_raw.max(bottom_width_lb);
        let area = (tw + bw) * d / 2.0;
        let wp = bw + d * (ss * ss + 1.0).sqrt() * 2.0;
        let hr = area / wp;

        depth[i] = d;
        top_width[i] = tw;
        side_slope[i] = ss;
        bottom_width[i] = bw;
        hydraulic_radius[i] = hr;
    }

    [depth, top_width, side_slope, bottom_width, hydraulic_radius]
}
