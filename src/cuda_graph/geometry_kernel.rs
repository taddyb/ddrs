//! SP-10: fused `#[cube]` kernels for the dense forward chain.
//!
//! Two kernels live here:
//!
//! 1. [`fused_geometry_s1_s14`] — the original SP-10 SPIKE kernel, kept
//!    intact because `tests/sp10_spike_capture.rs::spike_fused_kernel_capture_replay`
//!    still references it. Covers only S1..S14 with 5 outputs.
//!
//! 2. [`forward_k1_kernel`] — Phase 1 production kernel covering S1..S23
//!    (trapezoidal geometry + Muskingum coefficients) with 19 outputs. The
//!    captured forward region for SP-10 will be:
//!
//!        K1 → cuSPARSE SpMV → K2 (b_rhs) → assemble → cuSPARSE SpSV → K3 (clamp)
//!
//!    K1 produces every saved-state intermediate that `TimestepOp::backward`
//!    consumes from the pre-SpMV part of the chain — i.e., everything in
//!    `TimestepState` except `{a_values, b_rhs, i_t, x_sol}` (those come from
//!    the cuSPARSE / K2 / K3 stages).
//!
//! Both kernels mirror `forward_chain_inner` in `src/routing/mmc_op.rs:580+`
//! line-by-line. Intermediates live in GPU registers; only named outputs are
//! written to global memory.

use cubecl::prelude::*;

// ===========================================================================
// SP-10 SPIKE kernel (S1..S14, 5 outputs). Kept for spike test compatibility.
// ===========================================================================

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

// ===========================================================================
// SP-10 Phase 1 production kernel: K1 = S1..S23 (19 outputs).
// ===========================================================================

/// SP-10 Phase 1 production fused kernel: S1..S23 of `forward_chain_inner`.
///
/// Fuses the entire pre-SpMV dense portion of the per-timestep chain
/// (trapezoidal geometry S1..S17 + Muskingum coefficients S18..S23) into a
/// single launched cube kernel. Intermediates live in GPU registers; only
/// the 19 named outputs are written to global memory.
///
/// Output order MUST stay aligned with [`mmc_op::forward_saved_idx`] and
/// [`PersistentScratch`] field order. The 4 saved-state intermediates not
/// produced here (a_values, b_rhs, i_t, x_sol) come from later stages
/// (assemble / K2 / SpMV / SpSV).
///
/// Arithmetic mirrors `forward_chain_inner` in `src/routing/mmc_op.rs:580-627`
/// line-by-line. `powf(x, e)` is expressed as `(e * ln(x)).exp()` per the
/// spike kernel's pattern (cube's `F::powf` exists but is less explicit about
/// the trapezoidal-geometry math).
///
/// Scalar bounds and `dt` are passed as `F` arguments — cubecl supports `F`
/// scalar kernel parameters via its `ScalarArgSettings` impl (see
/// `cubecl-core/src/frontend/element/numeric.rs:99-115`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn forward_k1_kernel<F: Float + cubecl::CubeElement>(
    // ── 8 inputs ───────────────────────────────────────────────────────
    n_input: &Tensor<F>,
    qsp_input: &Tensor<F>,
    psp_input: &Tensor<F>,
    qt_input: &Tensor<F>,
    _qpt_input: &Tensor<F>, // unused in S1..S23 (Muskingum c4 multiplies q_prime_t in S25, not here)
    length_input: &Tensor<F>,
    slope_input: &Tensor<F>,
    xst_input: &Tensor<F>,

    // ── 19 outputs (each shape [n_segments]) ───────────────────────────
    // Order matches `forward_saved_idx` minus {A_VALUES, B_RHS, I_T, X_SOL}.
    o_depth: &mut Tensor<F>,             //  0  DEPTH
    o_top_width: &mut Tensor<F>,         //  1  TOP_WIDTH
    o_side_slope: &mut Tensor<F>,        //  2  SIDE_SLOPE
    o_bottom_width: &mut Tensor<F>,      //  3  BOTTOM_WIDTH
    o_hydraulic_radius: &mut Tensor<F>,  //  4  HYDRAULIC_RADIUS
    o_velocity_unclamped: &mut Tensor<F>,//  5  VELOCITY_UNCLAMPED
    o_velocity_clamped: &mut Tensor<F>,  //  6  VELOCITY_CLAMPED
    o_celerity: &mut Tensor<F>,          //  7  CELERITY
    o_k_muskingum: &mut Tensor<F>,       //  8  K_MUSKINGUM
    o_denom: &mut Tensor<F>,             //  9  DENOM
    o_c1: &mut Tensor<F>,                // 10  C1
    o_c2: &mut Tensor<F>,                // 11  C2
    o_c3: &mut Tensor<F>,                // 12  C3
    o_c4: &mut Tensor<F>,                // 13  C4
    // {14 A_VALUES, 15 B_RHS, 16 I_T, 17 X_SOL} skipped — produced later.
    o_ratio: &mut Tensor<F>,             // 18  RATIO
    o_denominator: &mut Tensor<F>,       // 19  DENOMINATOR
    o_q_eps: &mut Tensor<F>,             // 20  Q_EPS
    o_side_slope_raw: &mut Tensor<F>,    // 21  SIDE_SLOPE_RAW
    o_bw_raw: &mut Tensor<F>,            // 22  BW_RAW

    // ── scalar parameters ──────────────────────────────────────────────
    bottom_width_lb: F,
    depth_lb: F,
    velocity_lb: F,
    dt: F,
) {
    if ABSOLUTE_POS >= n_input.len() {
        terminate!();
    }
    let i = ABSOLUTE_POS;

    let n_i = n_input[i];
    let qsp_i = qsp_input[i];
    let psp_i = psp_input[i];
    let qt_i = qt_input[i];
    let length_i = length_input[i];
    let slope_i = slope_input[i];
    let xst_i = xst_input[i];

    // ── S1..S14: trapezoidal geometry ──────────────────────────────────
    // S1: q_eps = q_spatial + 1e-6
    let q_eps = qsp_i + F::new(1e-6);

    // S2: numerator = q_t * n * (q_eps + 1)
    let numerator = qt_i * n_i * (q_eps + F::new(1.0));

    // S3: denominator = p_spatial * sqrt(slope) + 1e-8
    let denominator = psp_i * slope_i.sqrt() + F::new(1e-8);

    // S4: ratio = numerator / denominator
    let ratio = numerator / denominator;

    // S5: exponent = (q_eps * 3 + 5).recip() * 3
    let exponent = (q_eps * F::new(3.0) + F::new(5.0)).recip() * F::new(3.0);

    // S6: depth = clamp_min(ratio^exponent, depth_lb)
    //     ratio^exponent = exp(exponent * ln(ratio))
    let depth_raw = (exponent * ratio.ln()).exp();
    let depth = depth_raw.max(depth_lb);

    // S7: top_width = p_spatial * depth^q_eps
    let top_width = psp_i * (q_eps * depth.ln()).exp();

    // S8: side_slope_raw = top_width * q_eps / (depth * 2)
    let side_slope_raw = top_width * q_eps / (depth * F::new(2.0));

    // S9: side_slope = clamp(side_slope_raw, 0.5, 50.0)
    let side_slope = side_slope_raw.clamp(F::new(0.5), F::new(50.0));

    // S10: bw_raw = top_width - side_slope * depth * 2
    let bw_raw = top_width - side_slope * depth * F::new(2.0);

    // S11: bottom_width = clamp_min(bw_raw, bottom_width_lb)
    let bottom_width = bw_raw.max(bottom_width_lb);

    // S12: area = (top_width + bottom_width) * depth / 2
    let area = (top_width + bottom_width) * depth / F::new(2.0);

    // S13: wp = bottom_width + depth * sqrt(side_slope^2 + 1) * 2
    let wp = bottom_width
        + depth * (side_slope * side_slope + F::new(1.0)).sqrt() * F::new(2.0);

    // S14: hyd_radius = area / wp
    let hyd_radius = area / wp;

    // ── S15..S17: velocity & celerity ──────────────────────────────────
    // S15: velocity_unclamped = (1/n) * hyd_radius^(2/3) * sqrt(slope)
    //      hyd_radius^(2/3) = exp((2/3) * ln(hyd_radius))
    let two_thirds = F::new(2.0) / F::new(3.0);
    let hr_pow_23 = (two_thirds * hyd_radius.ln()).exp();
    let velocity_un = n_i.recip() * hr_pow_23 * slope_i.sqrt();

    // S16: velocity_clamped = clamp(velocity_un, velocity_lb, 15.0)
    let velocity_cl = velocity_un.clamp(velocity_lb, F::new(15.0));

    // S17: celerity = velocity_clamped * 5/3
    let celerity = velocity_cl * (F::new(5.0) / F::new(3.0));

    // ── S18..S23: Muskingum coefficients ───────────────────────────────
    // S18: k_muskingum = length / celerity
    let k_muskingum = length_i / celerity;

    // S19..S22 helpers
    let one_minus_x = F::new(1.0) - xst_i;
    let two_k = k_muskingum * F::new(2.0);
    let two_kx = two_k * xst_i;
    let two_k_1mx = two_k * one_minus_x;

    // denom = two_k_1mx + dt
    let denom = two_k_1mx + dt;

    // S19: c1 = (-two_kx + dt) / denom
    let c1 = (-two_kx + dt) / denom;

    // S20: c2 = (two_kx + dt) / denom
    let c2 = (two_kx + dt) / denom;

    // S21: c3 = (two_k_1mx - dt) / denom
    let c3 = (two_k_1mx - dt) / denom;

    // S22: c4 = (2 * dt) / denom   [`denom.recip() * (2*dt)` in BURN]
    let c4 = denom.recip() * (F::new(2.0) * dt);

    // ── write 19 outputs ───────────────────────────────────────────────
    o_depth[i] = depth;
    o_top_width[i] = top_width;
    o_side_slope[i] = side_slope;
    o_bottom_width[i] = bottom_width;
    o_hydraulic_radius[i] = hyd_radius;
    o_velocity_unclamped[i] = velocity_un;
    o_velocity_clamped[i] = velocity_cl;
    o_celerity[i] = celerity;
    o_k_muskingum[i] = k_muskingum;
    o_denom[i] = denom;
    o_c1[i] = c1;
    o_c2[i] = c2;
    o_c3[i] = c3;
    o_c4[i] = c4;
    o_ratio[i] = ratio;
    o_denominator[i] = denominator;
    o_q_eps[i] = q_eps;
    o_side_slope_raw[i] = side_slope_raw;
    o_bw_raw[i] = bw_raw;

    // Silence dead-param warning — qpt is plumbed through for symmetry with
    // the 8-input signature but not consumed in S1..S23 (c4 * q_prime_t
    // happens in S25, K2's domain).
    let _ = area;
}

// ===========================================================================
// SP-10 Phase 2 production kernels: K2 (S25 b_rhs) + K3 (S28 q_clamp).
// ===========================================================================

/// SP-10 Phase 2 K2: fused S25 of `forward_chain_inner`.
///
///     b_rhs = c2 * i_t + c3 * q_t + c4 * q_prime_t
///
/// Single elementwise linear combination over `[n_segments]`. `c2`, `c3`, `c4`
/// come from K1; `i_t` is the cuSPARSE SpMV output; `qt`, `qpt` are per-step
/// inputs. Mirrors `mmc_op.rs:633-635`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn b_rhs_kernel<F: Float + cubecl::CubeElement>(
    c2_input: &Tensor<F>,
    c3_input: &Tensor<F>,
    c4_input: &Tensor<F>,
    i_t_input: &Tensor<F>,   // cuSPARSE SpMV output
    qt_input: &Tensor<F>,
    qpt_input: &Tensor<F>,
    o_b_rhs: &mut Tensor<F>,
) {
    if ABSOLUTE_POS >= o_b_rhs.len() {
        terminate!();
    }
    let i = ABSOLUTE_POS;
    o_b_rhs[i] =
        c2_input[i] * i_t_input[i] + c3_input[i] * qt_input[i] + c4_input[i] * qpt_input[i];
}

/// SP-10 Phase 2 K3: fused S28 of `forward_chain_inner`.
///
///     q_next = max(x_sol, discharge_lb)
///
/// Single elementwise lower-clamp over `[n_segments]`. `x_sol` is the
/// cuSPARSE SpSV solve output. Mirrors `mmc_op.rs:653-654`
/// (`q_next = x_sol.clamp_min(discharge_lb)`).
#[cube(launch)]
pub fn q_clamp_kernel<F: Float + cubecl::CubeElement>(
    x_sol_input: &Tensor<F>,
    o_q_next: &mut Tensor<F>,
    discharge_lb: F,
) {
    if ABSOLUTE_POS >= o_q_next.len() {
        terminate!();
    }
    let i = ABSOLUTE_POS;
    o_q_next[i] = x_sol_input[i].max(discharge_lb);
}
