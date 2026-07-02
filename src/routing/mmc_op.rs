//! Fused MC timestep custom autodiff op. SP-8.
//!
//! Replaces the ~33 BURN-tensor-op chain in `MuskingumCunge::route_timestep`
//! with a single autograd node. Pattern mirrors `CsrSolveOp` in
//! `src/sparse/mod.rs:415-462`: a `Backward<B, N>` impl with a saved-state
//! struct holding backend primitives (no autograd participation).
//!
//! Parents in fixed order: [n, q_spatial, p_spatial, q_t, q_prime_t].

use std::sync::Arc;

use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::{backend::Backend, Tensor, TensorPrimitive};

use crate::config::Config;
use crate::sparse::{self, dispatch, primitive_to_vec, AValuesAssembler, CsrPattern};

/// Inner-backend leakance inputs threaded into `forward_chain_inner`.
#[derive(Clone)]
pub(crate) struct LeakanceTensors<I: Backend> {
    pub k_d: Tensor<I, 1>,
    pub d_gw: Tensor<I, 1>,
    pub leakance_factor: Tensor<I, 1>,
}

/// Extra saved-state (inner-backend primitives) the leakance backward needs,
/// beyond what `TimestepState` already saves (`depth`, `p_spatial`, `q_eps` are
/// reused from there). This is the ONE leakance saved-state type — reused as the
/// `leak` field of `TimestepLeakanceState` (do not introduce a second).
#[derive(Clone, Debug)]
pub(crate) struct LeakanceSaved<I: Backend> {
    pub area_z: I::FloatTensorPrimitive,
    pub k_d: I::FloatTensorPrimitive,
    pub d_gw: I::FloatTensorPrimitive,
    pub leakance_factor: I::FloatTensorPrimitive,
}

/// Saved primitives used by `TimestepOp::backward`.
///
/// Forward inputs (the 5 autograd-tracked parents + the 3 constants) plus the
/// intermediates needed to evaluate the analytical chain rule.
#[derive(Clone, Debug)]
pub(crate) struct TimestepState<B: Backend> {
    pub pattern: Arc<CsrPattern>,
    // Inputs (autograd-tracked parents — read by backward to compute scalars
    // that flow into gradient register calls).
    pub n: B::FloatTensorPrimitive,
    pub q_spatial: B::FloatTensorPrimitive,
    pub p_spatial: B::FloatTensorPrimitive,
    pub q_t: B::FloatTensorPrimitive,
    pub q_prime_t: B::FloatTensorPrimitive,
    // Constants (not parents — used by backward but not differentiated through).
    pub length: B::FloatTensorPrimitive,
    pub slope: B::FloatTensorPrimitive,
    pub x_storage: B::FloatTensorPrimitive,
    // Forward intermediates (saved for backward).
    pub depth: B::FloatTensorPrimitive,
    pub top_width: B::FloatTensorPrimitive,
    pub side_slope: B::FloatTensorPrimitive,
    pub bottom_width: B::FloatTensorPrimitive,
    pub hydraulic_radius: B::FloatTensorPrimitive,
    pub velocity_unclamped: B::FloatTensorPrimitive,
    pub velocity_clamped: B::FloatTensorPrimitive,
    pub celerity: B::FloatTensorPrimitive,
    pub k_muskingum: B::FloatTensorPrimitive,
    pub denom: B::FloatTensorPrimitive,
    pub c1: B::FloatTensorPrimitive,
    pub c2: B::FloatTensorPrimitive,
    pub c3: B::FloatTensorPrimitive,
    pub c4: B::FloatTensorPrimitive,
    pub a_values: B::FloatTensorPrimitive,
    pub b_rhs: B::FloatTensorPrimitive,
    pub i_t: B::FloatTensorPrimitive, // N · Q_t  (SpMV result)
    pub x_sol: B::FloatTensorPrimitive, // pre-clamp solve output
    // Saved geometry intermediates needed by the chain rule (S2..S6).
    pub ratio: B::FloatTensorPrimitive,
    pub denominator: B::FloatTensorPrimitive, // p·√s + 1e-8 (S3)
    pub q_eps: B::FloatTensorPrimitive,       // q_spatial + 1e-6 (S1)
    pub side_slope_raw: B::FloatTensorPrimitive, // pre-clamp (S8)
    pub bw_raw: B::FloatTensorPrimitive,      // pre-clamp (S10)
    pub bottom_width_lb: f32,
    pub depth_lb: f32,
    pub velocity_lb: f32,
    pub discharge_lb: f32,
    pub dt: f32,
    pub use_cuda: bool,
}

#[derive(Debug)]
pub(crate) struct TimestepOp;

/// Zeta's geometry-side gradient contributions, folded into the shared backward
/// core at the three accumulation points (`gd_total`, `gq_spatial`, `gp_total`).
/// All tensors live on the inner backend `I` (no autograd tape).
pub(crate) struct ZetaGeomGrads<I: Backend> {
    pub g_depth: Tensor<I, 1>,
    pub g_q_eps: Tensor<I, 1>,
    pub g_p_spatial: Tensor<I, 1>,
}

/// The five accumulated parent gradients produced by [`timestep_backward_core`],
/// in parent order `[n, q_spatial, p_spatial, q_t, q_prime_t]`.
pub(crate) struct FiveGrads<I: Backend> {
    pub gn_total: Tensor<I, 1>,
    pub gq_spatial: Tensor<I, 1>,
    pub gp_total: Tensor<I, 1>,
    pub gq_t_total: Tensor<I, 1>,
    pub gq_prime_t: Tensor<I, 1>,
}

/// Shared analytical backward body for both [`TimestepOp`] (5 parents) and
/// [`TimestepLeakanceOp`] (8 parents). Computes the five parent gradients from
/// the saved `TimestepState`.
///
/// The leakance op subtracts `zeta` from `b_rhs` in the forward, so its backward
/// must (a) read `gb_rhs` to compute zeta's parent grads and (b) fold zeta's
/// geometry-side grads (`g_depth`, `g_q_eps`, `g_p_spatial`) into the existing
/// accumulators. Both halves of that circular dependency are resolved by the
/// `zeta_hook` closure: it is called with `gb_rhs` the moment it is available
/// (right after B27) and returns the geometry grads to inject. The 5-parent op
/// passes a hook that returns `None`, recovering the pre-leakance math exactly.
pub(crate) fn timestep_backward_core<I: Backend + 'static>(
    state: &TimestepState<I>,
    grad_out: I::FloatTensorPrimitive,
    zeta_hook: impl FnOnce(&Tensor<I, 1>) -> Option<ZetaGeomGrads<I>>,
) -> FiveGrads<I>
where
    I::FloatTensorPrimitive: 'static,
{
    {
        let device = I::float_device(&grad_out);

        // Wrap saved primitives as inner-backend Tensors. These are non-autodiff
        // tensors — every op below is a plain `I` tensor op, no tape pushes.
        let wrap = |p: I::FloatTensorPrimitive| -> Tensor<I, 1> {
            Tensor::from_primitive(TensorPrimitive::Float(p))
        };
        let unwrap = |t: Tensor<I, 1>| -> I::FloatTensorPrimitive {
            match t.into_primitive() {
                TensorPrimitive::Float(p) => p,
                _ => unreachable!(),
            }
        };

        let gy = wrap(grad_out);

        let n_t = wrap(state.n.clone());
        let _q_spatial = wrap(state.q_spatial.clone());
        let p_spatial = wrap(state.p_spatial.clone());
        let q_t = wrap(state.q_t.clone());
        let q_prime_t = wrap(state.q_prime_t.clone());
        let length = wrap(state.length.clone());
        let slope = wrap(state.slope.clone());
        let x_storage = wrap(state.x_storage.clone());

        let depth = wrap(state.depth.clone());
        let top_width = wrap(state.top_width.clone());
        let side_slope = wrap(state.side_slope.clone());
        let _bottom_width = wrap(state.bottom_width.clone());
        let hyd_radius = wrap(state.hydraulic_radius.clone());
        let velocity_un = wrap(state.velocity_unclamped.clone());
        let _velocity_cl = wrap(state.velocity_clamped.clone());
        let celerity = wrap(state.celerity.clone());
        let k_muskingum = wrap(state.k_muskingum.clone());
        let denom = wrap(state.denom.clone());
        let _c1 = wrap(state.c1.clone());
        let _c2 = wrap(state.c2.clone());
        let _c3 = wrap(state.c3.clone());
        let c4 = wrap(state.c4.clone());
        let i_t = wrap(state.i_t.clone());
        let x_sol = wrap(state.x_sol.clone());

        let ratio = wrap(state.ratio.clone());
        let denominator = wrap(state.denominator.clone());
        let q_eps = wrap(state.q_eps.clone());
        let side_slope_raw = wrap(state.side_slope_raw.clone());
        let bw_raw = wrap(state.bw_raw.clone());

        let dt = state.dt;
        let bottom_width_lb = state.bottom_width_lb;
        let depth_lb = state.depth_lb;
        let velocity_lb = state.velocity_lb;
        let discharge_lb = state.discharge_lb;

        // ===========================================================
        // B28. gx_sol = gy · mask(x_sol > discharge_lb)
        // ===========================================================
        // clamp_min(x, lb): gradient passes through where x > lb, zero where x == lb (saturated).
        let mask_x = x_sol.clone().greater_elem(discharge_lb);
        let gx_sol = gy.mask_fill(mask_x.bool_not(), 0.0);

        // ===========================================================
        // B27. (gA_values, gb_rhs) = csr_solve_backward(...)
        //   gb_rhs = (A^T)^{-1} · gx_sol
        //   gA_values[k] = -gb_rhs[row[k]] * x_sol[col[k]]
        // ===========================================================
        let gx_sol_prim = unwrap(gx_sol);
        let gb_rhs_prim = dispatch::backward_solve_primitive::<I>(
            &state.pattern,
            &state.a_values,
            &gx_sol_prim,
            &device,
            state.use_cuda,
        );

        // gA_values via direct gather+multiply on primitives (mirrors dispatch::grada_primitive).
        let gb_rhs = wrap(gb_rhs_prim.clone());

        // Leakance fold-in: zeta = ... was subtracted from b_rhs, so its
        // parent grads derive from `gb_rhs`. The hook computes zeta_backward
        // (with the 3 leakance parents registered by the caller) and returns
        // the geometry-side grads to inject below. `None` ⇒ pre-leakance math.
        let zeta_geom = zeta_hook(&gb_rhs);

        let g_a_values_prim = {
            // -gb[row] * x[col]
            let gradb_host: Vec<f32> = primitive_to_vec::<I>(gb_rhs_prim.clone());
            let x_host: Vec<f32> = primitive_to_vec::<I>(state.x_sol.clone());
            let pattern = &state.pattern;
            let grada: Vec<f32> = pattern
                .row_for_nnz
                .iter()
                .zip(pattern.col.iter())
                .map(|(&r, &c)| -gradb_host[r as usize] * x_host[c as usize])
                .collect();
            I::float_from_data(burn::tensor::TensorData::from(grada.as_slice()), &device)
        };

        // ===========================================================
        // B26. gc1 = assemble_backward(gA_values)
        // ===========================================================
        let gc1_prim =
            sparse::assemble_backward_primitive::<I>(&state.pattern, g_a_values_prim, &device, state.use_cuda);
        let gc1 = wrap(gc1_prim);

        // ===========================================================
        // B25. b_rhs = c2·i_t + c3·q_t + c4·q_prime_t
        //   gc2 = gb_rhs · i_t
        //   gc3 = gb_rhs · q_t
        //   gc4 = gb_rhs · q_prime_t
        //   gi_t = c2 · gb_rhs                         (partial)
        //   gq_t_from_S25 = c3 · gb_rhs                (partial)
        //   gq_prime_t = c4 · gb_rhs                   (final)
        // ===========================================================
        let c2_t = wrap(state.c2.clone());
        let c3_t = wrap(state.c3.clone());
        let gc2 = gb_rhs.clone() * i_t.clone();
        let gc3 = gb_rhs.clone() * q_t.clone();
        let gc4 = gb_rhs.clone() * q_prime_t.clone();
        let gi_t = c2_t.clone() * gb_rhs.clone();
        let gq_t_from_s25 = c3_t.clone() * gb_rhs.clone();
        let gq_prime_t = c4.clone() * gb_rhs.clone();

        // ===========================================================
        // B24. i_t = N · q_t  →  gq_t_from_S24 = N^T · gi_t
        // ===========================================================
        let gi_t_prim = unwrap(gi_t);
        let gq_t_from_s24_prim =
            sparse::spmv_backward_primitive::<I>(&state.pattern, gi_t_prim, &device, state.use_cuda);
        let gq_t_from_s24 = wrap(gq_t_from_s24_prim);

        // ===========================================================
        // B23-B20. Chain rule through c1..c4 → k_muskingum.
        //
        //   denom = 2k(1-x) + dt
        //   c1 = (-2kx + dt) / denom
        //   c2 = ( 2kx + dt) / denom
        //   c3 = ( 2k(1-x) - dt) / denom
        //   c4 = 2dt / denom
        //
        // For each ci, ∂L/∂denom_from_ci = gci · ∂ci/∂denom
        //                                = -gci · num_i / denom²
        //                                (since ci = num_i / denom, ∂ci/∂denom = -num_i/denom²)
        //
        // and ∂L/∂num_i_from_ci = gci / denom.
        // ===========================================================
        let denom_sq = denom.clone() * denom.clone();
        let one_minus_x = -x_storage.clone() + 1.0;
        let two_k = k_muskingum.clone() * 2.0;
        let two_kx = two_k.clone() * x_storage.clone();
        let two_k_1mx = two_k.clone() * one_minus_x.clone();

        // num_c1 = -2kx + dt
        let num_c1 = -two_kx.clone() + dt;
        // num_c2 = 2kx + dt
        let num_c2 = two_kx.clone() + dt;
        // num_c3 = 2k(1-x) - dt
        let num_c3 = two_k_1mx.clone() - dt;
        // num_c4 = 2dt (constant, no dependence on denom in numerator)

        // ∂denom_from_ci = -gci · num_i / denom²
        let gdenom_from_c1 = -gc1.clone() * num_c1.clone() / denom_sq.clone();
        let gdenom_from_c2 = -gc2.clone() * num_c2.clone() / denom_sq.clone();
        let gdenom_from_c3 = -gc3.clone() * num_c3.clone() / denom_sq.clone();
        // c4 = 2dt / denom  →  ∂c4/∂denom = -2dt/denom²
        let gdenom_from_c4 = -gc4.clone() * (2.0 * dt) / denom_sq.clone();
        let gdenom_total = gdenom_from_c1 + gdenom_from_c2 + gdenom_from_c3 + gdenom_from_c4;

        // ∂num_c1/∂(2kx) = -1   →   g_2kx_from_c1 = -gc1 / denom
        let g_2kx_from_c1 = -gc1.clone() / denom.clone();
        // ∂num_c2/∂(2kx) = +1   →   g_2kx_from_c2 = +gc2 / denom
        let g_2kx_from_c2 = gc2.clone() / denom.clone();
        // ∂num_c3/∂(2k(1-x)) = +1 → g_2k1mx_from_c3 = +gc3 / denom
        let g_2k1mx_from_c3 = gc3.clone() / denom.clone();
        // denom = 2k(1-x) + dt → ∂denom/∂(2k(1-x)) = 1 → g_2k1mx_from_denom = gdenom_total
        let g_2k1mx_from_denom = gdenom_total.clone();

        let g_2kx_total = g_2kx_from_c1 + g_2kx_from_c2;
        let g_2k1mx_total = g_2k1mx_from_c3 + g_2k1mx_from_denom;

        // 2kx = 2k · x_storage → ∂(2kx)/∂(2k) = x_storage
        let g_2k_from_2kx = g_2kx_total.clone() * x_storage.clone();
        // 2k(1-x) = 2k · (1 - x_storage) → ∂(2k(1-x))/∂(2k) = (1 - x_storage)
        let g_2k_from_2k1mx = g_2k1mx_total.clone() * one_minus_x.clone();
        let g_2k_total = g_2k_from_2kx + g_2k_from_2k1mx;
        // 2k = 2 · k_muskingum
        let gk_muskingum = g_2k_total * 2.0;

        // ===========================================================
        // B18. k_muskingum = length / celerity
        //   ∂k/∂celerity = -length / celerity²
        // ===========================================================
        let celerity_sq = celerity.clone() * celerity.clone();
        let gcelerity = -gk_muskingum * length.clone() / celerity_sq;

        // ===========================================================
        // B17. celerity = velocity_cl · 5/3
        // ===========================================================
        let gvelocity_cl = gcelerity * (5.0 / 3.0);

        // ===========================================================
        // B16. velocity_cl = clamp(velocity_un, velocity_lb, 15)
        //   gradient passes where velocity_lb < velocity_un < 15
        // ===========================================================
        let mask_v_lo = velocity_un.clone().greater_elem(velocity_lb);
        let mask_v_hi = velocity_un.clone().lower_elem(15.0);
        let mask_v = mask_v_lo.bool_and(mask_v_hi);
        let gvelocity_un = gvelocity_cl.mask_fill(mask_v.bool_not(), 0.0);

        // ===========================================================
        // B15. velocity_un = (1/n) · R^(2/3) · √slope
        //   Treating v = velocity_un saved primitive:
        //   ∂v/∂n = -v / n
        //   ∂v/∂R = (2/3) · v / R
        //   (slope is constant — dropped)
        // ===========================================================
        let gn_from_s15 = gvelocity_un.clone() * (-velocity_un.clone() / n_t.clone());
        let gr_from_s15 =
            gvelocity_un.clone() * (velocity_un.clone() * (2.0f32 / 3.0f32)) / hyd_radius.clone();

        // ===========================================================
        // B14. R = area / wp
        //   ∂R/∂area = 1 / wp
        //   ∂R/∂wp = -area / wp² = -R/wp
        // Need wp and area. We saved wp implicitly via area = R·wp. Re-derive wp:
        // ===========================================================
        // Recompute wp from saved bottom_width + 2·depth·sqrt(1+ss²) — equivalently
        // wp = area / hyd_radius (cheap and exact).
        // Use area from saved? We didn't save area, but area = R · wp. We need wp directly.
        // Recompute: wp = bottom_width + 2·depth·sqrt(1 + side_slope²).
        let one_plus_ss_sq = side_slope.clone() * side_slope.clone() + 1.0;
        let sqrt_1_plus_ss_sq = one_plus_ss_sq.clone().sqrt();
        let wp = _bottom_width.clone() + depth.clone() * sqrt_1_plus_ss_sq.clone() * 2.0;
        let area = hyd_radius.clone() * wp.clone();

        let gr = gr_from_s15;
        let garea_from_r = gr.clone() / wp.clone();
        let gwp_from_r = -gr * area.clone() / (wp.clone() * wp.clone());

        // ===========================================================
        // B13. wp = bw + 2·d·sqrt(1+ss²)
        //   ∂wp/∂bw = 1
        //   ∂wp/∂d = 2·sqrt(1+ss²)
        //   ∂wp/∂ss = 2·d · ss / sqrt(1+ss²)
        // ===========================================================
        let gwp = gwp_from_r;
        let gbw_from_s13 = gwp.clone();
        let gd_from_s13 = gwp.clone() * sqrt_1_plus_ss_sq.clone() * 2.0;
        let gss_from_s13 = gwp * depth.clone() * 2.0 * side_slope.clone() / sqrt_1_plus_ss_sq;

        // ===========================================================
        // B12. area = (tw + bw) · d / 2
        //   ∂area/∂tw = d/2
        //   ∂area/∂bw = d/2
        //   ∂area/∂d  = (tw + bw)/2
        // ===========================================================
        let garea = garea_from_r;
        let half_d = depth.clone() * 0.5;
        let gtw_from_s12 = garea.clone() * half_d.clone();
        let gbw_from_s12 = garea.clone() * half_d.clone();
        let gd_from_s12 = garea * (top_width.clone() + _bottom_width.clone()) * 0.5;

        // ===========================================================
        // B11. bottom_width = max(bw_raw, bottom_width_lb)
        //   gradient passes where bw_raw > lb.
        // ===========================================================
        let mask_bw = bw_raw.clone().greater_elem(bottom_width_lb);
        let gbw_total = gbw_from_s13 + gbw_from_s12;
        let gbw_raw = gbw_total.mask_fill(mask_bw.bool_not(), 0.0);

        // ===========================================================
        // B10. bw_raw = tw - 2·ss·d
        //   ∂bw_raw/∂tw = 1
        //   ∂bw_raw/∂ss = -2·d
        //   ∂bw_raw/∂d  = -2·ss
        // ===========================================================
        let gtw_from_s10 = gbw_raw.clone();
        let gss_from_s10 = -gbw_raw.clone() * 2.0 * depth.clone();
        let gd_from_s10 = -gbw_raw * 2.0 * side_slope.clone();

        // ===========================================================
        // B9. side_slope = clamp(side_slope_raw, 0.5, 50)
        //   gradient passes where 0.5 < ss_raw < 50.
        // ===========================================================
        let gss_combined = gss_from_s13 + gss_from_s10;
        let mask_ss_lo = side_slope_raw.clone().greater_elem(0.5);
        let mask_ss_hi = side_slope_raw.clone().lower_elem(50.0);
        let mask_ss = mask_ss_lo.bool_and(mask_ss_hi);
        let gss_from_clamp = gss_combined.mask_fill(mask_ss.bool_not(), 0.0);

        // ===========================================================
        // B8. side_slope_raw = tw · q_eps / (2·d)
        //   ∂ss_raw/∂tw    = q_eps / (2·d)
        //   ∂ss_raw/∂q_eps = tw / (2·d)
        //   ∂ss_raw/∂d     = -tw · q_eps / (2·d²)
        // ===========================================================
        let two_d = depth.clone() * 2.0;
        let gtw_from_s8 = gss_from_clamp.clone() * q_eps.clone() / two_d.clone();
        let gqeps_from_s8 = gss_from_clamp.clone() * top_width.clone() / two_d.clone();
        let gd_from_s8 = -gss_from_clamp
            * top_width.clone()
            * q_eps.clone()
            / (two_d.clone() * depth.clone());

        // Accumulate gtw before S7 (since S7 produces depth → uses tw in its derivative wrt q_eps).
        let gtw_total = gtw_from_s12 + gtw_from_s10 + gtw_from_s8;

        // ===========================================================
        // B7. top_width = p · depth^q_eps
        //   ∂tw/∂p     = depth^q_eps  = top_width / p
        //   ∂tw/∂depth = p · q_eps · depth^(q_eps - 1) = top_width · q_eps / depth
        //   ∂tw/∂q_eps = tw · ln(depth)
        // ===========================================================
        let gp_from_s7 = gtw_total.clone() * top_width.clone() / p_spatial.clone();
        let gdepth_from_s7 = gtw_total.clone() * top_width.clone() * q_eps.clone() / depth.clone();
        let gqeps_from_s7 = gtw_total * top_width.clone() * depth.clone().log();

        // ===========================================================
        // B6. depth = max(ratio^exponent, depth_lb)
        //   gradient passes where ratio^exponent > depth_lb.
        //   d = ratio^exponent
        //   ∂d/∂ratio = exponent · ratio^(exponent-1) = d · exponent / ratio
        //   ∂d/∂exponent = d · ln(ratio)
        // ===========================================================
        let mut gd_total = gd_from_s13 + gd_from_s12 + gd_from_s10 + gd_from_s8 + gdepth_from_s7;
        if let Some(zg) = zeta_geom.as_ref() {
            gd_total = gd_total + zg.g_depth.clone();
        }
        // depth saturates when ratio^exp == depth_lb (clamped). Gradient passes through
        // when depth > depth_lb. depth itself is the saved post-clamp value, so use
        // depth > depth_lb as the mask (when the clamp is inactive, depth > depth_lb).
        let mask_d = depth.clone().greater_elem(depth_lb);
        let gd_pre_clamp = gd_total.mask_fill(mask_d.bool_not(), 0.0);

        // d = ratio^exponent  (NOTE: depth saved is post-clamp; the un-clamped
        // value at differentiation point equals depth when mask is true, so we can
        // use depth in derivative expressions where mask is true.)
        let exponent = (q_eps.clone() * 3.0 + 5.0).recip() * 3.0;
        let gratio_from_s6 =
            gd_pre_clamp.clone() * exponent.clone() * depth.clone() / ratio.clone();
        let gexp_from_s6 = gd_pre_clamp * depth.clone() * ratio.clone().log();

        // ===========================================================
        // B5. exponent = 3 / (5 + 3·q_eps)
        //   ∂exp/∂q_eps = -9 / (5 + 3·q_eps)² = -3 · exp² / 3 = ... use direct:
        //   d/dq[3/(5+3q)] = -9/(5+3q)²
        // ===========================================================
        let five_plus_three_qeps = q_eps.clone() * 3.0 + 5.0;
        let gqeps_from_s5 = -gexp_from_s6 * 9.0 / (five_plus_three_qeps.clone() * five_plus_three_qeps);

        // ===========================================================
        // B4. ratio = numerator / denominator
        //   ∂ratio/∂num = 1/den
        //   ∂ratio/∂den = -num/den² = -ratio/den
        // ===========================================================
        let gnum = gratio_from_s6.clone() / denominator.clone();
        let gden = -gratio_from_s6 * ratio.clone() / denominator.clone();

        // ===========================================================
        // B3. denominator = p · √slope + 1e-8
        //   ∂den/∂p = √slope  (slope is constant)
        // ===========================================================
        let gp_from_s3 = gden * slope.clone().sqrt();

        // ===========================================================
        // B2. numerator = q_t · n · (q_eps + 1)
        //   ∂num/∂q_t   = n · (q_eps + 1)
        //   ∂num/∂n     = q_t · (q_eps + 1)
        //   ∂num/∂q_eps = q_t · n
        // ===========================================================
        let q_eps_plus_one = q_eps.clone() + 1.0;
        let gq_t_from_s2 = gnum.clone() * n_t.clone() * q_eps_plus_one.clone();
        let gn_from_s2 = gnum.clone() * q_t.clone() * q_eps_plus_one.clone();
        let gqeps_from_s2 = gnum * q_t.clone() * n_t.clone();

        // ===========================================================
        // B1. q_eps = q_spatial + 1e-6  →  ∂q_eps/∂q_spatial = 1
        // ===========================================================
        let mut gq_spatial = gqeps_from_s8 + gqeps_from_s7 + gqeps_from_s5 + gqeps_from_s2;
        if let Some(zg) = zeta_geom.as_ref() {
            gq_spatial = gq_spatial + zg.g_q_eps.clone();
        }

        // ===========================================================
        // Final accumulations on the 5 parents:
        // ===========================================================
        let gn_total = gn_from_s15 + gn_from_s2;
        let mut gp_total = gp_from_s7 + gp_from_s3;
        if let Some(zg) = zeta_geom.as_ref() {
            gp_total = gp_total + zg.g_p_spatial.clone();
        }
        let gq_t_total = gq_t_from_s25 + gq_t_from_s24 + gq_t_from_s2;

        // Touch unused intermediate bindings to silence dead-code warnings.
        let _ = (_q_spatial, _velocity_cl);

        FiveGrads {
            gn_total,
            gq_spatial,
            gp_total,
            gq_t_total,
            gq_prime_t,
        }
    }
}

impl<I: Backend + 'static> Backward<I, 5> for TimestepOp
where
    I::FloatTensorPrimitive: 'static,
{
    type State = TimestepState<I>;

    fn backward(
        self,
        ops: Ops<Self::State, 5>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let [p_n, p_qsp, p_psp, p_qt, p_qpt] = ops.parents;

        let grad_out = grads.consume::<I>(&ops.node);

        let unwrap = |t: Tensor<I, 1>| -> I::FloatTensorPrimitive {
            match t.into_primitive() {
                TensorPrimitive::Float(p) => p,
                _ => unreachable!(),
            }
        };

        // No leakance ⇒ hook returns None ⇒ pre-leakance math, byte-identical.
        let g = timestep_backward_core::<I>(&state, grad_out, |_gb_rhs| None);

        if let Some(node) = p_n {
            grads.register::<I>(node.id, unwrap(g.gn_total));
        }
        if let Some(node) = p_qsp {
            grads.register::<I>(node.id, unwrap(g.gq_spatial));
        }
        if let Some(node) = p_psp {
            grads.register::<I>(node.id, unwrap(g.gp_total));
        }
        if let Some(node) = p_qt {
            grads.register::<I>(node.id, unwrap(g.gq_t_total));
        }
        if let Some(node) = p_qpt {
            grads.register::<I>(node.id, unwrap(g.gq_prime_t));
        }
    }
}

/// Saved primitives for the leakance op: the base `TimestepState` plus the
/// extra leakance intermediates ([`LeakanceSaved`]). Reuses the SAME
/// `LeakanceSaved` type produced by `forward_chain_inner` (no second type).
#[derive(Clone, Debug)]
pub(crate) struct TimestepLeakanceState<I: Backend> {
    pub base: TimestepState<I>,
    pub leak: LeakanceSaved<I>,
}

#[derive(Debug)]
pub(crate) struct TimestepLeakanceOp;

impl<I: Backend + 'static> Backward<I, 8> for TimestepLeakanceOp
where
    I::FloatTensorPrimitive: 'static,
{
    type State = TimestepLeakanceState<I>;

    fn backward(
        self,
        ops: Ops<Self::State, 8>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let [p_n, p_qsp, p_psp, p_qt, p_qpt, p_kd, p_dgw, p_fac] = ops.parents;

        let grad_out = grads.consume::<I>(&ops.node);

        let wrap = |p: I::FloatTensorPrimitive| -> Tensor<I, 1> {
            Tensor::from_primitive(TensorPrimitive::Float(p))
        };
        let unwrap = |t: Tensor<I, 1>| -> I::FloatTensorPrimitive {
            match t.into_primitive() {
                TensorPrimitive::Float(p) => p,
                _ => unreachable!(),
            }
        };

        // Geometry inputs zeta depends on, read from the SHARED base state.
        let depth = wrap(state.base.depth.clone());
        let p_spatial = wrap(state.base.p_spatial.clone());
        let q_eps = wrap(state.base.q_eps.clone());
        // Leakance-only saved intermediates.
        let area_z = wrap(state.leak.area_z.clone());
        let k_d = wrap(state.leak.k_d.clone());
        let d_gw = wrap(state.leak.d_gw.clone());
        let leakance_factor = wrap(state.leak.leakance_factor.clone());

        // Capture zeta's 3 leakance-parent grads out of the hook so we can
        // register them after `core` returns. The hook runs zeta_backward with
        // `gb_rhs` (no pre-negation — zeta_backward negates internally) and
        // returns the geometry grads for `core` to fold into the 5 base grads.
        let mut zeta_param_grads: Option<(
            I::FloatTensorPrimitive,
            I::FloatTensorPrimitive,
            I::FloatTensorPrimitive,
        )> = None;
        let g = timestep_backward_core::<I>(&state.base, grad_out, |gb_rhs| {
            let zg = crate::routing::leakance::zeta_backward::<I>(
                gb_rhs.clone(),
                depth.clone(),
                p_spatial.clone(),
                q_eps.clone(),
                area_z.clone(),
                k_d.clone(),
                d_gw.clone(),
                leakance_factor.clone(),
            );
            zeta_param_grads = Some((
                unwrap(zg.g_k_d),
                unwrap(zg.g_d_gw),
                unwrap(zg.g_leakance_factor),
            ));
            Some(ZetaGeomGrads {
                g_depth: zg.g_depth,
                g_q_eps: zg.g_q_eps,
                g_p_spatial: zg.g_p_spatial,
            })
        });

        // Register the 5 base parents (zeta geom already folded in by `core`).
        if let Some(node) = p_n {
            grads.register::<I>(node.id, unwrap(g.gn_total));
        }
        if let Some(node) = p_qsp {
            grads.register::<I>(node.id, unwrap(g.gq_spatial));
        }
        if let Some(node) = p_psp {
            grads.register::<I>(node.id, unwrap(g.gp_total));
        }
        if let Some(node) = p_qt {
            grads.register::<I>(node.id, unwrap(g.gq_t_total));
        }
        if let Some(node) = p_qpt {
            grads.register::<I>(node.id, unwrap(g.gq_prime_t));
        }

        // Register the 3 leakance parents.
        let (g_k_d, g_d_gw, g_fac) =
            zeta_param_grads.expect("zeta_hook always runs in the leakance backward");
        if let Some(node) = p_kd {
            grads.register::<I>(node.id, g_k_d);
        }
        if let Some(node) = p_dgw {
            grads.register::<I>(node.id, g_d_gw);
        }
        if let Some(node) = p_fac {
            grads.register::<I>(node.id, g_fac);
        }
    }
}

/// Number of saved-state primitives `forward_chain_inner` returns, in
/// addition to `q_next`. Matches the count of `state_*` fields on
/// `crate::cuda_graph::PersistentScratch` (and the indices in
/// [`ForwardSavedIdx`]).
pub(crate) const NUM_SAVED_STATE: usize = 23;

/// Position of each saved-state primitive in the array returned by
/// [`forward_chain_inner`]. Mirrors the declaration order of the `state_*`
/// fields on `PersistentScratch` so the SP-10 capture path can index by
/// position without re-typing the field name list.
#[allow(dead_code)]
pub(crate) mod forward_saved_idx {
    pub const DEPTH: usize = 0;
    pub const TOP_WIDTH: usize = 1;
    pub const SIDE_SLOPE: usize = 2;
    pub const BOTTOM_WIDTH: usize = 3;
    pub const HYDRAULIC_RADIUS: usize = 4;
    pub const VELOCITY_UNCLAMPED: usize = 5;
    pub const VELOCITY_CLAMPED: usize = 6;
    pub const CELERITY: usize = 7;
    pub const K_MUSKINGUM: usize = 8;
    pub const DENOM: usize = 9;
    pub const C1: usize = 10;
    pub const C2: usize = 11;
    pub const C3: usize = 12;
    pub const C4: usize = 13;
    pub const A_VALUES: usize = 14;
    pub const B_RHS: usize = 15;
    pub const I_T: usize = 16;
    pub const X_SOL: usize = 17;
    pub const RATIO: usize = 18;
    pub const DENOMINATOR: usize = 19;
    pub const Q_EPS: usize = 20;
    pub const SIDE_SLOPE_RAW: usize = 21;
    pub const BW_RAW: usize = 22;
}

/// Inner-backend S1..S28 chain. Operates on `I::FloatTensorPrimitive`s — no
/// autograd tape pushes. Returns `(q_next, [saved_state; NUM_SAVED_STATE])`.
///
/// Shared between [`timestep_forward`] (regular per-step path) and SP-10's
/// `try_capture_forward` (CUDA-graph capture path) so that exactly the same
/// kernel sequence runs in both cases — keeping V1 ABSOLUTE MATCH while also
/// giving capture a deterministic chain to record.
///
/// The saved-state array is indexed by [`forward_saved_idx`], whose order
/// mirrors `crate::cuda_graph::PersistentScratch::state_*` declaration order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_chain_inner<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    n_in: Tensor<I, 1>,
    qsp_in: Tensor<I, 1>,
    psp_in: Tensor<I, 1>,
    qt_in: Tensor<I, 1>,
    qpt_in: Tensor<I, 1>,
    length_in: Tensor<I, 1>,
    slope_in: Tensor<I, 1>,
    xst_in: Tensor<I, 1>,
    leakance: Option<LeakanceTensors<I>>,
    leak_out: &mut Option<LeakanceSaved<I>>,
) -> (
    I::FloatTensorPrimitive,
    [I::FloatTensorPrimitive; NUM_SAVED_STATE],
)
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    use crate::config::SparseSolver;

    let dt = crate::routing::mmc::DT_SECONDS;
    let bottom_width_lb = cfg.params.attribute_minimums.bottom_width;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let discharge_lb = cfg.params.attribute_minimums.discharge;
    let use_cuda = cfg.params.sparse_solver == SparseSolver::Cuda;

    let unwrap = |t: Tensor<I, 1>| -> I::FloatTensorPrimitive {
        match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!(),
        }
    };
    let wrap = |p: I::FloatTensorPrimitive| -> Tensor<I, 1> {
        Tensor::from_primitive(TensorPrimitive::Float(p))
    };

    let qt_prim_for_spmv = unwrap(qt_in.clone());
    let device = I::float_device(&qt_prim_for_spmv);

    // S1
    let q_eps = qsp_in.clone() + 1e-6_f32;
    // S2
    let numerator = qt_in.clone() * n_in.clone() * (q_eps.clone() + 1.0);
    // S3
    let denominator = psp_in.clone() * slope_in.clone().sqrt() + 1e-8_f32;
    // S4
    let ratio = numerator.clone() / denominator.clone();
    // S5
    let exponent = (q_eps.clone() * 3.0 + 5.0).recip() * 3.0;
    // S6
    let depth = ratio.clone().powf(exponent.clone()).clamp_min(depth_lb);
    // S7
    let top_width = psp_in.clone() * depth.clone().powf(q_eps.clone());
    // S8 (pre-clamp side slope)
    let side_slope_raw = top_width.clone() * q_eps.clone() / (depth.clone() * 2.0);
    // S9
    let side_slope = side_slope_raw.clone().clamp(0.5, 50.0);
    // S10 (pre-clamp bw)
    let bw_raw = top_width.clone() - side_slope.clone() * depth.clone() * 2.0;
    // S11
    let bottom_width = bw_raw.clone().clamp_min(bottom_width_lb);
    // S12
    let _area = (top_width.clone() + bottom_width.clone()) * depth.clone() / 2.0;
    // S13
    let wp = bottom_width.clone()
        + depth.clone() * (side_slope.clone().powf_scalar(2.0) + 1.0).sqrt() * 2.0;
    // S14
    let hyd_radius = _area.clone() / wp.clone();
    // S15
    let velocity_un = n_in.clone().recip() * hyd_radius.clone().powf_scalar(2.0 / 3.0)
        * slope_in.clone().sqrt();
    // S16
    let velocity_cl = velocity_un.clone().clamp(velocity_lb, 15.0);
    // S17
    let celerity = velocity_cl.clone() * (5.0_f32 / 3.0_f32);

    // S18..S23: Muskingum coefficients
    let k_muskingum = length_in.clone() / celerity.clone();
    let one_minus_x = -xst_in.clone() + 1.0;
    let two_k = k_muskingum.clone() * 2.0;
    let two_kx = two_k.clone() * xst_in.clone();
    let two_k_1mx = two_k.clone() * one_minus_x.clone();
    let denom = two_k_1mx.clone() + dt;
    let c1 = (-two_kx.clone() + dt) / denom.clone();
    let c2 = (two_kx.clone() + dt) / denom.clone();
    let c3 = (two_k_1mx.clone() - dt) / denom.clone();
    let c4 = denom.clone().recip() * (2.0 * dt);

    // S24: i_t = N · q_t (inner-backend SpMV)
    let i_t_prim = sparse::spmv_primitive::<I>(pattern, qt_prim_for_spmv.clone(), &device, use_cuda, None);
    let i_t = wrap(i_t_prim.clone());

    // Leakance: compute zeta from the SHARED depth/q_eps (S6/S1) so it can be
    // subtracted from b_rhs below. `None` ⇒ this block is skipped entirely and
    // the kernel order is byte-identical to the pre-leakance path.
    let zeta_opt = leakance.as_ref().map(|lk| {
        let (_w, area_z, zeta) = crate::routing::leakance::zeta_forward::<I>(
            depth.clone(),
            psp_in.clone(),
            q_eps.clone(),
            length_in.clone(),
            lk.k_d.clone(),
            lk.d_gw.clone(),
            lk.leakance_factor.clone(),
        );
        *leak_out = Some(LeakanceSaved {
            area_z: unwrap(area_z.clone()),
            k_d: unwrap(lk.k_d.clone()),
            d_gw: unwrap(lk.d_gw.clone()),
            leakance_factor: unwrap(lk.leakance_factor.clone()),
        });
        zeta
    });

    // S25: b_rhs = c2·i_t + c3·q_t + c4·q_prime_t  (− zeta when leakance active)
    let b_rhs_base =
        c2.clone() * i_t.clone() + c3.clone() * qt_in.clone() + c4.clone() * qpt_in.clone();
    let b_rhs = match zeta_opt {
        Some(zeta) => b_rhs_base - zeta,
        None => b_rhs_base,
    };

    // S26: A_values = assemble_primitive(c1)
    let c1_prim = unwrap(c1.clone());
    let a_values_prim = sparse::assemble_primitive::<I>(pattern, c1_prim.clone(), &device, None);

    // S27: x_sol = triangular_csr_solve(a_values, b_rhs) — at primitive level via dispatch.
    let b_rhs_prim = unwrap(b_rhs.clone());
    let (x_sol_prim, _saved_x) = dispatch::forward_primitive::<I>(
        pattern,
        &a_values_prim,
        &b_rhs_prim,
        &device,
        use_cuda,
        None,
    );
    let x_sol = wrap(x_sol_prim.clone());

    // S28: q_next = max(x_sol, discharge_lb)
    let q_next = x_sol.clone().clamp_min(discharge_lb);
    let q_next_prim = unwrap(q_next);

    // Saved-state array — order MUST match `forward_saved_idx`.
    let saved: [I::FloatTensorPrimitive; NUM_SAVED_STATE] = [
        unwrap(depth),         // 0  DEPTH
        unwrap(top_width),     // 1  TOP_WIDTH
        unwrap(side_slope),    // 2  SIDE_SLOPE
        unwrap(bottom_width),  // 3  BOTTOM_WIDTH
        unwrap(hyd_radius),    // 4  HYDRAULIC_RADIUS
        unwrap(velocity_un),   // 5  VELOCITY_UNCLAMPED
        unwrap(velocity_cl),   // 6  VELOCITY_CLAMPED
        unwrap(celerity),      // 7  CELERITY
        unwrap(k_muskingum),   // 8  K_MUSKINGUM
        unwrap(denom),         // 9  DENOM
        c1_prim,               // 10 C1
        unwrap(c2),            // 11 C2
        unwrap(c3),            // 12 C3
        unwrap(c4),            // 13 C4
        a_values_prim,         // 14 A_VALUES
        b_rhs_prim,            // 15 B_RHS
        i_t_prim,              // 16 I_T
        x_sol_prim,            // 17 X_SOL
        unwrap(ratio),         // 18 RATIO
        unwrap(denominator),   // 19 DENOMINATOR
        unwrap(q_eps),         // 20 Q_EPS
        unwrap(side_slope_raw),// 21 SIDE_SLOPE_RAW
        unwrap(bw_raw),        // 22 BW_RAW
    ];

    (q_next_prim, saved)
}

/// Pinning variant of [`forward_chain_inner`]. Mirrors the existing chain
/// line-for-line and kernel-for-kernel, but pushes a clone of every
/// intermediate `FloatTensorPrimitive` into `pin` so the caller can keep the
/// underlying cubecl `Handle`s alive past the closure scope.
///
/// Why this exists: cubecl's persistent memory pool may recycle the device
/// address of a `Handle` the moment the last reference drops. When a CUDA
/// graph captures kernels that read/write those addresses, replaying the
/// graph after the addresses have been recycled triggers
/// `CUDA_ERROR_ILLEGAL_ADDRESS`. Holding every intermediate handle in a
/// caller-owned sink prevents that recycling for the lifetime of the graph.
///
/// Compound BURN expressions in [`forward_chain_inner`] are split here into
/// single-op steps so every sub-expression temp is also pinned — not just
/// the named outputs. Keep the kernel order identical to
/// [`forward_chain_inner`] so V9 bit-match still holds. If you change one,
/// change both.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn forward_chain_inner_pinned<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    n_in: Tensor<I, 1>,
    qsp_in: Tensor<I, 1>,
    psp_in: Tensor<I, 1>,
    qt_in: Tensor<I, 1>,
    qpt_in: Tensor<I, 1>,
    length_in: Tensor<I, 1>,
    slope_in: Tensor<I, 1>,
    xst_in: Tensor<I, 1>,
    pin: &mut Vec<I::FloatTensorPrimitive>,
    // SP-10 Task C2: type-erased sink for Handles materialized inside
    // assemble_primitive / spmv_primitive / forward_primitive that are not
    // exposed as named outputs (e.g. pattern uploads `row_idx` / `adj` /
    // `diag`, and the arithmetic-chain intermediates in `assemble_primitive`).
    // These would otherwise drop on helper return and their persistent-pool
    // slices would be recycled into post-capture allocations whose addresses
    // overlap with what the captured graph still reads — corrupting replay.
    //
    // Type-erased because the bucket can hold Float OR Int primitives.
    extra_pin: &mut Vec<Box<dyn std::any::Any + Send>>,
) -> (
    I::FloatTensorPrimitive,
    [I::FloatTensorPrimitive; NUM_SAVED_STATE],
)
where
    I::FloatTensorPrimitive: 'static,
    I::IntTensorPrimitive: 'static,
    I::Device: 'static,
{
    use crate::config::SparseSolver;

    let dt = crate::routing::mmc::DT_SECONDS;
    let bottom_width_lb = cfg.params.attribute_minimums.bottom_width;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let discharge_lb = cfg.params.attribute_minimums.discharge;
    let use_cuda = cfg.params.sparse_solver == SparseSolver::Cuda;

    let unwrap = |t: Tensor<I, 1>| -> I::FloatTensorPrimitive {
        match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!(),
        }
    };
    let wrap = |p: I::FloatTensorPrimitive| -> Tensor<I, 1> {
        Tensor::from_primitive(TensorPrimitive::Float(p))
    };
    // Clone the underlying primitive of `t` (cheap Arc-bump on cubecl Handle)
    // and push it into `pin`. Does not consume the tensor.
    let pin_clone = |t: &Tensor<I, 1>, pin: &mut Vec<I::FloatTensorPrimitive>| {
        let p = match t.clone().into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!(),
        };
        pin.push(p);
    };

    let qt_prim_for_spmv = unwrap(qt_in.clone());
    let device = I::float_device(&qt_prim_for_spmv);

    // S1
    let q_eps = qsp_in.clone() + 1e-6_f32;
    pin_clone(&q_eps, pin);

    // S2: numerator = qt_in * n_in * (q_eps + 1.0)
    //   Rust eval order: ((qt_in*n_in) * (q_eps+1.0)). Outer mul evaluates
    //   lhs first (the qt*n product), then rhs (q_eps + 1.0).
    let qt_n = qt_in.clone() * n_in.clone();
    pin_clone(&qt_n, pin);
    let q_eps_plus_one = q_eps.clone() + 1.0;
    pin_clone(&q_eps_plus_one, pin);
    let numerator = qt_n * q_eps_plus_one;
    pin_clone(&numerator, pin);

    // S3: denominator = psp_in * slope_in.sqrt() + 1e-8
    let slope_sqrt = slope_in.clone().sqrt();
    pin_clone(&slope_sqrt, pin);
    let psp_slope_sqrt = psp_in.clone() * slope_sqrt;
    pin_clone(&psp_slope_sqrt, pin);
    let denominator = psp_slope_sqrt + 1e-8_f32;
    pin_clone(&denominator, pin);

    // S4
    let ratio = numerator.clone() / denominator.clone();
    pin_clone(&ratio, pin);

    // S5: exponent = (q_eps * 3.0 + 5.0).recip() * 3.0
    let q_eps_times_3 = q_eps.clone() * 3.0;
    pin_clone(&q_eps_times_3, pin);
    let q_eps_times_3_plus_5 = q_eps_times_3 + 5.0;
    pin_clone(&q_eps_times_3_plus_5, pin);
    let exponent_recip = q_eps_times_3_plus_5.recip();
    pin_clone(&exponent_recip, pin);
    let exponent = exponent_recip * 3.0;
    pin_clone(&exponent, pin);

    // S6: depth = ratio.powf(exponent).clamp_min(depth_lb)
    let depth_pre_clamp = ratio.clone().powf(exponent.clone());
    pin_clone(&depth_pre_clamp, pin);
    let depth = depth_pre_clamp.clamp_min(depth_lb);
    pin_clone(&depth, pin);

    // S7: top_width = psp_in * depth.powf(q_eps)
    //   Mul evaluates lhs first (psp_in, trivial), then rhs (powf).
    let depth_pow_q_eps = depth.clone().powf(q_eps.clone());
    pin_clone(&depth_pow_q_eps, pin);
    let top_width = psp_in.clone() * depth_pow_q_eps;
    pin_clone(&top_width, pin);

    // S8: side_slope_raw = top_width * q_eps / (depth * 2.0)
    //   Parses as ((top_width*q_eps) / (depth*2.0)). lhs first.
    let top_width_q_eps = top_width.clone() * q_eps.clone();
    pin_clone(&top_width_q_eps, pin);
    let depth_times_2 = depth.clone() * 2.0;
    pin_clone(&depth_times_2, pin);
    let side_slope_raw = top_width_q_eps / depth_times_2;
    pin_clone(&side_slope_raw, pin);

    // S9
    let side_slope = side_slope_raw.clone().clamp(0.5, 50.0);
    pin_clone(&side_slope, pin);

    // S10: bw_raw = top_width - side_slope * depth * 2.0
    //   Parses as top_width - ((side_slope*depth)*2.0). lhs (top_width)
    //   trivial; eval rhs: side_slope*depth, then *2.0, then sub.
    let ss_depth = side_slope.clone() * depth.clone();
    pin_clone(&ss_depth, pin);
    let ss_depth_2 = ss_depth * 2.0;
    pin_clone(&ss_depth_2, pin);
    let bw_raw = top_width.clone() - ss_depth_2;
    pin_clone(&bw_raw, pin);

    // S11
    let bottom_width = bw_raw.clone().clamp_min(bottom_width_lb);
    pin_clone(&bottom_width, pin);

    // S12: _area = (top_width + bottom_width) * depth / 2.0
    let tw_plus_bw = top_width.clone() + bottom_width.clone();
    pin_clone(&tw_plus_bw, pin);
    let area_times_2 = tw_plus_bw * depth.clone();
    pin_clone(&area_times_2, pin);
    let _area = area_times_2 / 2.0;
    pin_clone(&_area, pin);

    // S13: wp = bottom_width + depth * (side_slope^2 + 1).sqrt() * 2.0
    //   Precedence: bottom_width + (depth * sqrt(ss^2+1) * 2.0).
    //   Inner mul left-assoc: (depth * sqrt(...)) * 2.0.
    //   Outer + evaluates lhs (bottom_width, trivial), then rhs.
    //   Inside rhs: outer mul lhs first → inner mul lhs first (depth,
    //   trivial) → inner mul rhs (sqrt arg → ss^2 → +1 → sqrt) → inner
    //   mul → *2.0 → outer +.
    let ss_sq = side_slope.clone().powf_scalar(2.0);
    pin_clone(&ss_sq, pin);
    let ss_sq_plus_one = ss_sq + 1.0;
    pin_clone(&ss_sq_plus_one, pin);
    let ss_sq_plus_one_sqrt = ss_sq_plus_one.sqrt();
    pin_clone(&ss_sq_plus_one_sqrt, pin);
    let depth_ss = depth.clone() * ss_sq_plus_one_sqrt;
    pin_clone(&depth_ss, pin);
    let depth_ss_2 = depth_ss * 2.0;
    pin_clone(&depth_ss_2, pin);
    let wp = bottom_width.clone() + depth_ss_2;
    pin_clone(&wp, pin);

    // S14
    let hyd_radius = _area.clone() / wp.clone();
    pin_clone(&hyd_radius, pin);

    // S15: velocity_un = n_in.recip() * hyd_radius.powf_scalar(2/3) * slope_in.sqrt()
    //   Parses as ((n_in.recip() * hr_pow) * slope_in.sqrt()).
    //   Eval: outer mul lhs first → inner mul lhs (n_in.recip) → inner
    //   mul rhs (hr_pow) → inner mul → outer mul rhs (slope_sqrt2) →
    //   outer mul.
    let n_recip = n_in.clone().recip();
    pin_clone(&n_recip, pin);
    let hr_pow = hyd_radius.clone().powf_scalar(2.0 / 3.0);
    pin_clone(&hr_pow, pin);
    let n_recip_hr = n_recip * hr_pow;
    pin_clone(&n_recip_hr, pin);
    let slope_sqrt2 = slope_in.clone().sqrt();
    pin_clone(&slope_sqrt2, pin);
    let velocity_un = n_recip_hr * slope_sqrt2;
    pin_clone(&velocity_un, pin);

    // S16
    let velocity_cl = velocity_un.clone().clamp(velocity_lb, 15.0);
    pin_clone(&velocity_cl, pin);

    // S17
    let celerity = velocity_cl.clone() * (5.0_f32 / 3.0_f32);
    pin_clone(&celerity, pin);

    // S18..S23: Muskingum coefficients
    let k_muskingum = length_in.clone() / celerity.clone();
    pin_clone(&k_muskingum, pin);

    // S19: one_minus_x = -xst_in + 1.0
    let neg_xst = -xst_in.clone();
    pin_clone(&neg_xst, pin);
    let one_minus_x = neg_xst + 1.0;
    pin_clone(&one_minus_x, pin);

    // S20
    let two_k = k_muskingum.clone() * 2.0;
    pin_clone(&two_k, pin);
    // S21
    let two_kx = two_k.clone() * xst_in.clone();
    pin_clone(&two_kx, pin);
    // S22
    let two_k_1mx = two_k.clone() * one_minus_x.clone();
    pin_clone(&two_k_1mx, pin);
    // S23
    let denom = two_k_1mx.clone() + dt;
    pin_clone(&denom, pin);

    // c1 = (-two_kx + dt) / denom
    let neg_two_kx = -two_kx.clone();
    pin_clone(&neg_two_kx, pin);
    let c1_num = neg_two_kx + dt;
    pin_clone(&c1_num, pin);
    let c1 = c1_num / denom.clone();
    pin_clone(&c1, pin);

    // c2 = (two_kx + dt) / denom
    let c2_num = two_kx.clone() + dt;
    pin_clone(&c2_num, pin);
    let c2 = c2_num / denom.clone();
    pin_clone(&c2, pin);

    // c3 = (two_k_1mx - dt) / denom
    let c3_num = two_k_1mx.clone() - dt;
    pin_clone(&c3_num, pin);
    let c3 = c3_num / denom.clone();
    pin_clone(&c3, pin);

    // c4 = denom.recip() * (2.0 * dt)
    let denom_recip = denom.clone().recip();
    pin_clone(&denom_recip, pin);
    let c4 = denom_recip * (2.0 * dt);
    pin_clone(&c4, pin);

    // S24: i_t = N · q_t (inner-backend SpMV) — opaque primitive op.
    let i_t_prim = sparse::spmv_primitive::<I>(
        pattern,
        qt_prim_for_spmv.clone(),
        &device,
        use_cuda,
        Some(extra_pin),
    );
    pin.push(i_t_prim.clone());
    let i_t = wrap(i_t_prim.clone());

    // S25: b_rhs = c2*i_t + c3*qt_in + c4*qpt_in
    //   Parses as ((c2*i_t) + (c3*qt_in)) + (c4*qpt_in). Left-assoc +.
    //   Eval: outer + lhs first → inner + lhs (c2*i_t) → inner + rhs
    //   (c3*qt_in) → inner + → outer + rhs (c4*qpt_in) → outer +.
    let c2_it = c2.clone() * i_t.clone();
    pin_clone(&c2_it, pin);
    let c3_qt = c3.clone() * qt_in.clone();
    pin_clone(&c3_qt, pin);
    let c2_it_c3_qt = c2_it + c3_qt;
    pin_clone(&c2_it_c3_qt, pin);
    let c4_qpt = c4.clone() * qpt_in.clone();
    pin_clone(&c4_qpt, pin);
    let b_rhs = c2_it_c3_qt + c4_qpt;
    pin_clone(&b_rhs, pin);

    // S26: A_values = assemble_primitive(c1) — opaque primitive op.
    let c1_prim = unwrap(c1.clone());
    let a_values_prim = sparse::assemble_primitive::<I>(
        pattern,
        c1_prim.clone(),
        &device,
        Some(extra_pin),
    );
    pin.push(a_values_prim.clone());

    // S27: x_sol = triangular_csr_solve(a_values, b_rhs) — opaque primitive op.
    let b_rhs_prim = unwrap(b_rhs.clone());
    let (x_sol_prim, _saved_x) = dispatch::forward_primitive::<I>(
        pattern,
        &a_values_prim,
        &b_rhs_prim,
        &device,
        use_cuda,
        Some(extra_pin),
    );
    pin.push(x_sol_prim.clone());
    let x_sol = wrap(x_sol_prim.clone());

    // S28: q_next = max(x_sol, discharge_lb)
    let q_next = x_sol.clone().clamp_min(discharge_lb);
    pin_clone(&q_next, pin);
    let q_next_prim = unwrap(q_next);

    // Saved-state array — order MUST match `forward_saved_idx` and mirror
    // the original `forward_chain_inner` exactly.
    let saved: [I::FloatTensorPrimitive; NUM_SAVED_STATE] = [
        unwrap(depth),         // 0  DEPTH
        unwrap(top_width),     // 1  TOP_WIDTH
        unwrap(side_slope),    // 2  SIDE_SLOPE
        unwrap(bottom_width),  // 3  BOTTOM_WIDTH
        unwrap(hyd_radius),    // 4  HYDRAULIC_RADIUS
        unwrap(velocity_un),   // 5  VELOCITY_UNCLAMPED
        unwrap(velocity_cl),   // 6  VELOCITY_CLAMPED
        unwrap(celerity),      // 7  CELERITY
        unwrap(k_muskingum),   // 8  K_MUSKINGUM
        unwrap(denom),         // 9  DENOM
        c1_prim,               // 10 C1
        unwrap(c2),            // 11 C2
        unwrap(c3),            // 12 C3
        unwrap(c4),            // 13 C4
        a_values_prim,         // 14 A_VALUES
        b_rhs_prim,            // 15 B_RHS
        i_t_prim,              // 16 I_T
        x_sol_prim,            // 17 X_SOL
        unwrap(ratio),         // 18 RATIO
        unwrap(denominator),   // 19 DENOMINATOR
        unwrap(q_eps),         // 20 Q_EPS
        unwrap(side_slope_raw),// 21 SIDE_SLOPE_RAW
        unwrap(bw_raw),        // 22 BW_RAW
    ];

    (q_next_prim, saved)
}

/// Forward + register-on-tape entry point. Called from
/// `MuskingumCunge::route_timestep` (Task 4). Returns Q_{t+1} as an
/// autograd-tracked rank-1 tensor.
///
/// Parent order: [n, q_spatial, p_spatial, q_t, q_prime_t]. The three
/// constants (length, slope, x_storage) are not differentiated through.
///
/// Computes the entire S1..S28 chain at the **inner-backend** level (no
/// autograd tape pushes inside the op body), saves the intermediates needed
/// by the analytical backward to a `TimestepState`, and registers a single
/// `TimestepOp` node on the autograd tape.
#[allow(clippy::too_many_arguments)]
pub fn timestep_forward<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    _assembler: &AValuesAssembler<I>,
    n_at: Tensor<Autodiff<I>, 1>,
    q_spatial_at: Tensor<Autodiff<I>, 1>,
    p_spatial_at: Tensor<Autodiff<I>, 1>,
    q_t_at: Tensor<Autodiff<I>, 1>,
    q_prime_t_at: Tensor<Autodiff<I>, 1>,
    length_at: Tensor<Autodiff<I>, 1>,
    slope_at: Tensor<Autodiff<I>, 1>,
    x_storage_at: Tensor<Autodiff<I>, 1>,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    use crate::config::SparseSolver;

    let dt = crate::routing::mmc::DT_SECONDS;
    let bottom_width_lb = cfg.params.attribute_minimums.bottom_width;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let discharge_lb = cfg.params.attribute_minimums.discharge;
    let use_cuda = cfg.params.sparse_solver == SparseSolver::Cuda;

    // Extract AutodiffTensor (carries `primitive` + `node`).
    let unwrap_at = |t: Tensor<Autodiff<I>, 1>| match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => panic!("expected float tensor"),
    };
    let n_aut = unwrap_at(n_at);
    let qsp_aut = unwrap_at(q_spatial_at);
    let psp_aut = unwrap_at(p_spatial_at);
    let qt_aut = unwrap_at(q_t_at);
    let qpt_aut = unwrap_at(q_prime_t_at);
    let length_aut = unwrap_at(length_at);
    let slope_aut = unwrap_at(slope_at);
    let xst_aut = unwrap_at(x_storage_at);

    let n_p = n_aut.primitive.clone();
    let qsp_p = qsp_aut.primitive.clone();
    let psp_p = psp_aut.primitive.clone();
    let qt_p = qt_aut.primitive.clone();
    let qpt_p = qpt_aut.primitive.clone();
    let length_p = length_aut.primitive.clone();
    let slope_p = slope_aut.primitive.clone();
    let xst_p = xst_aut.primitive.clone();

    // --- S1..S28: trapezoidal geometry + clamp + celerity + solve. Compute on inner backend. ---
    let wrap = |p: I::FloatTensorPrimitive| -> Tensor<I, 1> {
        Tensor::from_primitive(TensorPrimitive::Float(p))
    };

    let (q_next_prim, saved) = forward_chain_inner::<I>(
        cfg,
        pattern,
        wrap(n_p.clone()),
        wrap(qsp_p.clone()),
        wrap(psp_p.clone()),
        wrap(qt_p.clone()),
        wrap(qpt_p.clone()),
        wrap(length_p.clone()),
        wrap(slope_p.clone()),
        wrap(xst_p.clone()),
        None,
        &mut None,
    );

    // Unpack saved-state array into named TimestepState fields. Indices MUST
    // match `forward_saved_idx`.
    use forward_saved_idx as fsi;
    let [
        depth_p, top_width_p, side_slope_p, bottom_width_p,
        hyd_radius_p, velocity_un_p, velocity_cl_p, celerity_p,
        k_muskingum_p, denom_p, c1_prim, c2_p, c3_p, c4_p,
        a_values_prim, b_rhs_prim, i_t_prim, x_sol_prim,
        ratio_p, denominator_p, q_eps_p, side_slope_raw_p, bw_raw_p,
    ] = saved;
    // Compile-time sanity: confirm the index constants are aligned with the
    // destructure above (touch each so a future re-order is caught).
    let _ = (fsi::DEPTH, fsi::BW_RAW);

    // Build TimestepState saving every intermediate the backward needs.
    let state = TimestepState::<I> {
        pattern: pattern.clone(),
        n: n_p,
        q_spatial: qsp_p,
        p_spatial: psp_p,
        q_t: qt_p,
        q_prime_t: qpt_p,
        length: length_p,
        slope: slope_p,
        x_storage: xst_p,
        depth: depth_p,
        top_width: top_width_p,
        side_slope: side_slope_p,
        bottom_width: bottom_width_p,
        hydraulic_radius: hyd_radius_p,
        velocity_unclamped: velocity_un_p,
        velocity_clamped: velocity_cl_p,
        celerity: celerity_p,
        k_muskingum: k_muskingum_p,
        denom: denom_p,
        c1: c1_prim,
        c2: c2_p,
        c3: c3_p,
        c4: c4_p,
        a_values: a_values_prim,
        b_rhs: b_rhs_prim,
        i_t: i_t_prim,
        x_sol: x_sol_prim,
        ratio: ratio_p,
        denominator: denominator_p,
        q_eps: q_eps_p,
        side_slope_raw: side_slope_raw_p,
        bw_raw: bw_raw_p,
        bottom_width_lb,
        depth_lb,
        velocity_lb,
        discharge_lb,
        dt,
        use_cuda,
    };

    // Register the op on the autograd tape.
    let result_prim = match TimestepOp
        .prepare::<NoCheckpointing>([
            n_aut.node.clone(),
            qsp_aut.node.clone(),
            psp_aut.node.clone(),
            qt_aut.node.clone(),
            qpt_aut.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(state, q_next_prim),
        OpsKind::UnTracked(prep) => prep.finish(q_next_prim),
    };

    Tensor::from_primitive(TensorPrimitive::Float(result_prim))
}

/// Leakance variant of [`timestep_forward`]. Identical to it, plus three extra
/// autograd-tracked parents (`K_D`, `d_gw`, `leakance_factor`) threaded into
/// `forward_chain_inner`'s leakance gate so `zeta` is subtracted from `b_rhs`.
/// Registers a [`TimestepLeakanceOp`] node (8 parents). Never uses CUDA graphs
/// (leakance forces `use_cuda_graphs: false`).
///
/// `zeta_out`: eval-time diagnostic sink. When `Some`, receives this step's
/// zeta (inner backend, no tape), recomputed from the SAME saved primitives
/// the backward reads — so the reported value is exactly what was subtracted
/// from `b_rhs`. `None` (the training path) adds zero kernels.
#[allow(clippy::too_many_arguments)]
pub fn timestep_forward_leakance<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    _assembler: &AValuesAssembler<I>,
    n_at: Tensor<Autodiff<I>, 1>,
    q_spatial_at: Tensor<Autodiff<I>, 1>,
    p_spatial_at: Tensor<Autodiff<I>, 1>,
    q_t_at: Tensor<Autodiff<I>, 1>,
    q_prime_t_at: Tensor<Autodiff<I>, 1>,
    length_at: Tensor<Autodiff<I>, 1>,
    slope_at: Tensor<Autodiff<I>, 1>,
    x_storage_at: Tensor<Autodiff<I>, 1>,
    k_d_at: Tensor<Autodiff<I>, 1>,
    d_gw_at: Tensor<Autodiff<I>, 1>,
    leakance_factor_at: Tensor<Autodiff<I>, 1>,
    zeta_out: Option<&mut Option<Tensor<I, 1>>>,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    use crate::config::SparseSolver;

    let dt = crate::routing::mmc::DT_SECONDS;
    let bottom_width_lb = cfg.params.attribute_minimums.bottom_width;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let discharge_lb = cfg.params.attribute_minimums.discharge;
    let use_cuda = cfg.params.sparse_solver == SparseSolver::Cuda;

    let unwrap_at = |t: Tensor<Autodiff<I>, 1>| match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => panic!("expected float tensor"),
    };
    let n_aut = unwrap_at(n_at);
    let qsp_aut = unwrap_at(q_spatial_at);
    let psp_aut = unwrap_at(p_spatial_at);
    let qt_aut = unwrap_at(q_t_at);
    let qpt_aut = unwrap_at(q_prime_t_at);
    let length_aut = unwrap_at(length_at);
    let slope_aut = unwrap_at(slope_at);
    let xst_aut = unwrap_at(x_storage_at);
    let kd_aut = unwrap_at(k_d_at);
    let dgw_aut = unwrap_at(d_gw_at);
    let fac_aut = unwrap_at(leakance_factor_at);

    let n_p = n_aut.primitive.clone();
    let qsp_p = qsp_aut.primitive.clone();
    let psp_p = psp_aut.primitive.clone();
    let qt_p = qt_aut.primitive.clone();
    let qpt_p = qpt_aut.primitive.clone();
    let length_p = length_aut.primitive.clone();
    let slope_p = slope_aut.primitive.clone();
    let xst_p = xst_aut.primitive.clone();
    let kd_p = kd_aut.primitive.clone();
    let dgw_p = dgw_aut.primitive.clone();
    let fac_p = fac_aut.primitive.clone();

    let wrap = |p: I::FloatTensorPrimitive| -> Tensor<I, 1> {
        Tensor::from_primitive(TensorPrimitive::Float(p))
    };

    let leakance = LeakanceTensors {
        k_d: wrap(kd_p.clone()),
        d_gw: wrap(dgw_p.clone()),
        leakance_factor: wrap(fac_p.clone()),
    };
    let mut leak_out: Option<LeakanceSaved<I>> = None;

    let (q_next_prim, saved) = forward_chain_inner::<I>(
        cfg,
        pattern,
        wrap(n_p.clone()),
        wrap(qsp_p.clone()),
        wrap(psp_p.clone()),
        wrap(qt_p.clone()),
        wrap(qpt_p.clone()),
        wrap(length_p.clone()),
        wrap(slope_p.clone()),
        wrap(xst_p.clone()),
        Some(leakance),
        &mut leak_out,
    );
    let leak = leak_out.expect("forward_chain_inner must populate LeakanceSaved when leakance is Some");

    use forward_saved_idx as fsi;
    let [
        depth_p, top_width_p, side_slope_p, bottom_width_p,
        hyd_radius_p, velocity_un_p, velocity_cl_p, celerity_p,
        k_muskingum_p, denom_p, c1_prim, c2_p, c3_p, c4_p,
        a_values_prim, b_rhs_prim, i_t_prim, x_sol_prim,
        ratio_p, denominator_p, q_eps_p, side_slope_raw_p, bw_raw_p,
    ] = saved;
    let _ = (fsi::DEPTH, fsi::BW_RAW);

    // Eval-time zeta diagnostic: zeta = factor · area_z · K_D · (depth − d_gw),
    // recomputed from the saved primitives (cheap: 3 elementwise kernels,
    // only when a sink is supplied).
    if let Some(out) = zeta_out {
        let m = wrap(depth_p.clone()) - wrap(leak.d_gw.clone());
        *out = Some(
            wrap(leak.leakance_factor.clone()) * wrap(leak.area_z.clone()) * wrap(leak.k_d.clone()) * m,
        );
    }

    let base = TimestepState::<I> {
        pattern: pattern.clone(),
        n: n_p,
        q_spatial: qsp_p,
        p_spatial: psp_p,
        q_t: qt_p,
        q_prime_t: qpt_p,
        length: length_p,
        slope: slope_p,
        x_storage: xst_p,
        depth: depth_p,
        top_width: top_width_p,
        side_slope: side_slope_p,
        bottom_width: bottom_width_p,
        hydraulic_radius: hyd_radius_p,
        velocity_unclamped: velocity_un_p,
        velocity_clamped: velocity_cl_p,
        celerity: celerity_p,
        k_muskingum: k_muskingum_p,
        denom: denom_p,
        c1: c1_prim,
        c2: c2_p,
        c3: c3_p,
        c4: c4_p,
        a_values: a_values_prim,
        b_rhs: b_rhs_prim,
        i_t: i_t_prim,
        x_sol: x_sol_prim,
        ratio: ratio_p,
        denominator: denominator_p,
        q_eps: q_eps_p,
        side_slope_raw: side_slope_raw_p,
        bw_raw: bw_raw_p,
        bottom_width_lb,
        depth_lb,
        velocity_lb,
        discharge_lb,
        dt,
        use_cuda,
    };

    let state = TimestepLeakanceState::<I> { base, leak };

    let result_prim = match TimestepLeakanceOp
        .prepare::<NoCheckpointing>([
            n_aut.node.clone(),
            qsp_aut.node.clone(),
            psp_aut.node.clone(),
            qt_aut.node.clone(),
            qpt_aut.node.clone(),
            kd_aut.node.clone(),
            dgw_aut.node.clone(),
            fac_aut.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(state, q_next_prim),
        OpsKind::UnTracked(prep) => prep.finish(q_next_prim),
    };

    Tensor::from_primitive(TensorPrimitive::Float(result_prim))
}

/// SP-10 Task 6: forward-graph replay path.
///
/// Same external contract as [`timestep_forward`], but instead of building
/// `Q_next` by re-running the S1..S28 kernel chain, it:
///
///   1. D2D-copies the per-step `q_t` → `scratch.in_q` and
///      `q_prime_t` → `scratch.in_qp`.
///   2. `cuGraphLaunch`'s the once-captured forward CUDA graph (lives on
///      `cache.graph_fwd`). Replay writes all 24 outputs (Q_next + 23 saved
///      intermediates) into the persistent scratch destinations.
///   3. For each output, allocates a fresh cubecl handle and D2D-copies from
///      its scratch destination into it (via
///      [`crate::sparse::cusparse::fresh_primitive_from_scratch`]). The fresh
///      handles are autograd-tape-eligible primitives owned by the per-step
///      result; they keep their content alive past the next replay (which
///      would otherwise overwrite the scratch contents).
///   4. Builds the `TimestepState` from the 23 fresh state primitives and
///      registers a `TimestepOp` node on the tape (identical to the
///      direct-launch path).
///
/// Falls back to [`timestep_forward`] if no graph is installed on the cache
/// (e.g. when capture failed at setup time and recorded a fallback reason).
#[allow(clippy::too_many_arguments)]
pub fn timestep_forward_via_graph<I: Backend + 'static>(
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
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    use crate::config::SparseSolver;

    // SAFETY: route_timestep is the training thread's entry; no other thread
    // accesses this pattern's cuda cache. The borrow lives only for the
    // duration of this call. Autodiff<I>::Device == I::Device, so the
    // tensor's device is the inner-backend device the cache must live on.
    let cache_device = q_t_at.device();
    let cache =
        unsafe { crate::sparse::cusparse::ensure_cuda_cache::<I>(pattern, &cache_device) };

    // If no graph was installed (capture failed), fall through to direct launch.
    if cache.graph_fwd.is_none() || cache.scratch.is_none() {
        return timestep_forward::<I>(
            cfg, pattern, assembler,
            n_at, q_spatial_at, p_spatial_at,
            q_t_at, q_prime_t_at, length_at, slope_at, x_storage_at,
        );
    }

    let scratch = cache.scratch.as_ref().expect("scratch must exist when graph_fwd does");
    let graph = cache.graph_fwd.as_ref().expect("graph_fwd checked above");
    // Hop the raw CUgraphExec pointer across the closure as `usize` —
    // `&CudaGraph` is !Send because the raw CUgraphExec is `*mut _`.
    let graph_exec_addr: usize = graph.exec_raw() as usize;
    let n_seg = pattern.n;
    let nnz = pattern.col.len();
    let bytes_n = n_seg * std::mem::size_of::<f32>();

    let dt = crate::routing::mmc::DT_SECONDS;
    let bottom_width_lb = cfg.params.attribute_minimums.bottom_width;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let discharge_lb = cfg.params.attribute_minimums.discharge;
    let use_cuda = cfg.params.sparse_solver == SparseSolver::Cuda;

    // Unwrap autograd primitives.
    let unwrap_at = |t: Tensor<Autodiff<I>, 1>| match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => panic!("expected float tensor"),
    };
    let n_aut = unwrap_at(n_at);
    let qsp_aut = unwrap_at(q_spatial_at);
    let psp_aut = unwrap_at(p_spatial_at);
    let qt_aut = unwrap_at(q_t_at);
    let qpt_aut = unwrap_at(q_prime_t_at);
    let length_aut = unwrap_at(length_at);
    let slope_aut = unwrap_at(slope_at);
    let xst_aut = unwrap_at(x_storage_at);

    let n_p = n_aut.primitive.clone();
    let qsp_p = qsp_aut.primitive.clone();
    let psp_p = psp_aut.primitive.clone();
    let qt_p = qt_aut.primitive.clone();
    let qpt_p = qpt_aut.primitive.clone();
    let length_p = length_aut.primitive.clone();
    let slope_p = slope_aut.primitive.clone();
    let xst_p = xst_aut.primitive.clone();

    let device = I::float_device(&qt_p);

    // Get per-step input devptrs (we'll D2D-copy these into scratch.in_q/in_qp).
    let qt_devptr = crate::sparse::cusparse::primitive_devptr::<I>(&qt_p)
        .expect("q_t primitive must be CUDA tensor in graph-replay path");
    let qpt_devptr = crate::sparse::cusparse::primitive_devptr::<I>(&qpt_p)
        .expect("q_prime_t primitive must be CUDA tensor in graph-replay path");

    let client = crate::sparse::cusparse::compute_client::<I>(&device);
    // Flush any pending cubecl work before the graph replay so prior kernels
    // serialize correctly with the in_q/in_qp seeding copies below.
    client.flush().expect("flush before graph replay");

    // Persistent scratch destination pointers.
    let dst_in_q = unsafe { crate::sparse::cusparse::handle_devptr(&client, &scratch.in_q) };
    let dst_in_qp = unsafe { crate::sparse::cusparse::handle_devptr(&client, &scratch.in_qp) };

    // -------- Drive CUDA work directly from this thread --------
    //
    // We CANNOT submit BURN/cubecl work from inside `exclusive_with_server`
    // (the channel deadlocks since the server is busy in our closure). The
    // capture path (`try_capture_forward`) takes the same approach: bind
    // cubecl's primary CUDA context to this thread, then drive all CUDA APIs
    // directly. Mirror that here so replay sees the same context.
    //
    // SAFETY: retain+set_current is harmless if already bound. The primary
    // context is retained for the SAME ordinal cubecl is using (read from
    // the tensors' device — config-selected, no longer hardcoded 0).
    // cubecl's ComputeClient::empty (used by fresh_primitive_from_scratch)
    // tolerates being called from this thread.
    let cuda_ordinal = crate::sparse::cusparse::cuda_device_index::<I>(&device) as i32;
    unsafe {
        let cu_dev = cudarc::driver::result::device::get(cuda_ordinal)
            .expect("graph-replay: cuDeviceGet failed");
        let ctx = cudarc::driver::result::primary_ctx::retain(cu_dev)
            .expect("graph-replay: primary_ctx::retain failed");
        cudarc::driver::result::ctx::set_current(ctx)
            .expect("graph-replay: ctx::set_current failed");
    }

    let stream_addr: usize = crate::sparse::cusparse::cubecl_stream_active::<I>(&device) as usize;
    let stream = stream_addr as cudarc::driver::sys::CUstream;

    // 1. Seed: D2D q_t -> in_q, q_prime_t -> in_qp.
    // SAFETY: src/dst pointers are valid CUDA device pointers on cubecl's
    // memory pool; bytes <= alloc size; stream is cubecl's primary stream;
    // context is bound on this thread.
    unsafe {
        cudarc::driver::result::memcpy_dtod_async(
            dst_in_q, qt_devptr, bytes_n, stream,
        )
        .expect("graph-replay: D2D q_t -> in_q failed");
        cudarc::driver::result::memcpy_dtod_async(
            dst_in_qp, qpt_devptr, bytes_n, stream,
        )
        .expect("graph-replay: D2D q_prime_t -> in_qp failed");
    }

    // 2. Replay the captured graph.
    //
    // SAFETY: graph_exec is the valid CUgraphExec owned by `cache.graph_fwd`
    // (alive for the cache lifetime). Context is bound on this thread.
    let graph_exec = graph_exec_addr as cudarc::driver::sys::CUgraphExec;
    unsafe {
        cudarc::driver::result::graph::launch(graph_exec, stream)
            .expect("graph-replay: cuGraphLaunch failed");
    }

    // 3. Allocate fresh handles + D2D-copy outputs.
    //
    // SAFETY: src scratch handles alive for the cache lifetime; stream is
    // cubecl's primary stream; copies are stream-ordered after the launch.
    use crate::sparse::cusparse::fresh_primitive_from_scratch;
    let q_next_prim = unsafe {
        fresh_primitive_from_scratch::<I>(&scratch.out_q, n_seg, stream, &device)
    };
    let state_arr: [I::FloatTensorPrimitive; NUM_SAVED_STATE] = unsafe {
        [
            fresh_primitive_from_scratch::<I>(&scratch.state_depth, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_top_width, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_side_slope, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_bottom_width, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_hydraulic_radius, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_velocity_unclamped, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_velocity_clamped, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_celerity, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_k_muskingum, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_denom, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_c1, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_c2, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_c3, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_c4, n_seg, stream, &device),
            // state_a_values has size [nnz], not [n].
            fresh_primitive_from_scratch::<I>(&scratch.state_a_values, nnz, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_b_rhs, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_i_t, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_x_sol, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_ratio, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_denominator, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_q_eps, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_side_slope_raw, n_seg, stream, &device),
            fresh_primitive_from_scratch::<I>(&scratch.state_bw_raw, n_seg, stream, &device),
        ]
    };

    let _ = assembler; // unused — same as direct-launch path; kept for API parity.

    // Unpack state primitives. Indices MUST match `forward_saved_idx`.
    use forward_saved_idx as fsi;
    let [
        depth_p, top_width_p, side_slope_p, bottom_width_p,
        hyd_radius_p, velocity_un_p, velocity_cl_p, celerity_p,
        k_muskingum_p, denom_p, c1_prim, c2_p, c3_p, c4_p,
        a_values_prim, b_rhs_prim, i_t_prim, x_sol_prim,
        ratio_p, denominator_p, q_eps_p, side_slope_raw_p, bw_raw_p,
    ] = state_arr;
    let _ = (fsi::DEPTH, fsi::BW_RAW); // index sanity touch

    // Build TimestepState. Backward needs the per-step `q_t`/`q_prime_t`
    // primitives (NOT scratch handles — scratch gets overwritten on next
    // replay, but the backward closure runs on `loss.backward()` after the
    // forward window completes, so it has to see the per-step values).
    let state = TimestepState::<I> {
        pattern: pattern.clone(),
        n: n_p,
        q_spatial: qsp_p,
        p_spatial: psp_p,
        q_t: qt_p,
        q_prime_t: qpt_p,
        length: length_p,
        slope: slope_p,
        x_storage: xst_p,
        depth: depth_p,
        top_width: top_width_p,
        side_slope: side_slope_p,
        bottom_width: bottom_width_p,
        hydraulic_radius: hyd_radius_p,
        velocity_unclamped: velocity_un_p,
        velocity_clamped: velocity_cl_p,
        celerity: celerity_p,
        k_muskingum: k_muskingum_p,
        denom: denom_p,
        c1: c1_prim,
        c2: c2_p,
        c3: c3_p,
        c4: c4_p,
        a_values: a_values_prim,
        b_rhs: b_rhs_prim,
        i_t: i_t_prim,
        x_sol: x_sol_prim,
        ratio: ratio_p,
        denominator: denominator_p,
        q_eps: q_eps_p,
        side_slope_raw: side_slope_raw_p,
        bw_raw: bw_raw_p,
        bottom_width_lb,
        depth_lb,
        velocity_lb,
        discharge_lb,
        dt,
        use_cuda,
    };

    let result_prim = match TimestepOp
        .prepare::<NoCheckpointing>([
            n_aut.node.clone(),
            qsp_aut.node.clone(),
            psp_aut.node.clone(),
            qt_aut.node.clone(),
            qpt_aut.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(state, q_next_prim),
        OpsKind::UnTracked(prep) => prep.finish(q_next_prim),
    };

    Tensor::from_primitive(TensorPrimitive::Float(result_prim))
}

/// SP-10 Phase 1: BURN-chain reference for the K1 fused kernel bit-match
/// test. Runs `forward_chain_inner` and returns the 19 saved-state
/// intermediates that K1 produces (everything in the saved-state array except
/// indices 14..=17: A_VALUES, B_RHS, I_T, X_SOL).
///
/// Returns each output as `Vec<f32>` via `into_data`, which forces a host
/// sync so all kernels have completed before comparison.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn __spike_forward_chain_k1_outputs<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    n_in: Tensor<I, 1>,
    qsp_in: Tensor<I, 1>,
    psp_in: Tensor<I, 1>,
    qt_in: Tensor<I, 1>,
    qpt_in: Tensor<I, 1>,
    length_in: Tensor<I, 1>,
    slope_in: Tensor<I, 1>,
    xst_in: Tensor<I, 1>,
) -> Vec<Vec<f32>>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let (_q_next, saved) = forward_chain_inner::<I>(
        cfg, pattern, n_in, qsp_in, psp_in, qt_in, qpt_in, length_in, slope_in, xst_in, None,
        &mut None,
    );

    // Indices K1 produces (skip 14..=17: A_VALUES, B_RHS, I_T, X_SOL).
    let k1_indices: [usize; 19] = [
        forward_saved_idx::DEPTH,
        forward_saved_idx::TOP_WIDTH,
        forward_saved_idx::SIDE_SLOPE,
        forward_saved_idx::BOTTOM_WIDTH,
        forward_saved_idx::HYDRAULIC_RADIUS,
        forward_saved_idx::VELOCITY_UNCLAMPED,
        forward_saved_idx::VELOCITY_CLAMPED,
        forward_saved_idx::CELERITY,
        forward_saved_idx::K_MUSKINGUM,
        forward_saved_idx::DENOM,
        forward_saved_idx::C1,
        forward_saved_idx::C2,
        forward_saved_idx::C3,
        forward_saved_idx::C4,
        forward_saved_idx::RATIO,
        forward_saved_idx::DENOMINATOR,
        forward_saved_idx::Q_EPS,
        forward_saved_idx::SIDE_SLOPE_RAW,
        forward_saved_idx::BW_RAW,
    ];

    k1_indices
        .iter()
        .map(|&idx| {
            let prim = saved[idx].clone();
            let t = Tensor::<I, 1>::from_primitive(TensorPrimitive::Float(prim));
            t.into_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("convert saved-state to Vec<f32>")
        })
        .collect()
}

/// SP-10 Phase 2: BURN-chain reference for the K2 + K3 fused kernel bit-match
/// tests. Runs `forward_chain_inner` and returns `(b_rhs, i_t, x_sol, q_next)`
/// as host `Vec<f32>`. K2 consumes `i_t` and produces `b_rhs`; K3 consumes
/// `x_sol` and produces `q_next`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn __spike_forward_chain_k23_outputs<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    n_in: Tensor<I, 1>,
    qsp_in: Tensor<I, 1>,
    psp_in: Tensor<I, 1>,
    qt_in: Tensor<I, 1>,
    qpt_in: Tensor<I, 1>,
    length_in: Tensor<I, 1>,
    slope_in: Tensor<I, 1>,
    xst_in: Tensor<I, 1>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let (q_next_prim, saved) = forward_chain_inner::<I>(
        cfg, pattern, n_in, qsp_in, psp_in, qt_in, qpt_in, length_in, slope_in, xst_in, None,
        &mut None,
    );

    let to_vec = |prim: I::FloatTensorPrimitive| -> Vec<f32> {
        let t = Tensor::<I, 1>::from_primitive(TensorPrimitive::Float(prim));
        t.into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("convert primitive to Vec<f32>")
    };

    let b_rhs = to_vec(saved[forward_saved_idx::B_RHS].clone());
    let i_t = to_vec(saved[forward_saved_idx::I_T].clone());
    let x_sol = to_vec(saved[forward_saved_idx::X_SOL].clone());
    let q_next = to_vec(q_next_prim);

    (b_rhs, i_t, x_sol, q_next)
}
