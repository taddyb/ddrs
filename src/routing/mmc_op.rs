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

        // ∂L/∂q_next  (shape [N])  — inner-backend primitive.
        let grad_out = grads.consume::<I>(&ops.node);
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
            sparse::assemble_backward_primitive::<I>(&state.pattern, g_a_values_prim, &device);
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
            sparse::spmv_backward_primitive::<I>(&state.pattern, gi_t_prim, &device);
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
        let gd_total = gd_from_s13 + gd_from_s12 + gd_from_s10 + gd_from_s8 + gdepth_from_s7;
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
        let gq_spatial = gqeps_from_s8 + gqeps_from_s7 + gqeps_from_s5 + gqeps_from_s2;

        // ===========================================================
        // Final accumulations on the 5 parents:
        // ===========================================================
        let gn_total = gn_from_s15 + gn_from_s2;
        let gp_total = gp_from_s7 + gp_from_s3;
        let gq_t_total = gq_t_from_s25 + gq_t_from_s24 + gq_t_from_s2;

        if let Some(node) = p_n {
            grads.register::<I>(node.id, unwrap(gn_total));
        }
        if let Some(node) = p_qsp {
            grads.register::<I>(node.id, unwrap(gq_spatial));
        }
        if let Some(node) = p_psp {
            grads.register::<I>(node.id, unwrap(gp_total));
        }
        if let Some(node) = p_qt {
            grads.register::<I>(node.id, unwrap(gq_t_total));
        }
        if let Some(node) = p_qpt {
            grads.register::<I>(node.id, unwrap(gq_prime_t));
        }

        // Touch unused intermediate bindings to silence dead-code warnings.
        let _ = (_q_spatial, _velocity_cl);
    }
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

    let device = I::float_device(&n_p);

    // --- S1..S17: trapezoidal geometry + clamp + celerity. Compute on inner backend. ---
    let wrap = |p: I::FloatTensorPrimitive| -> Tensor<I, 1> {
        Tensor::from_primitive(TensorPrimitive::Float(p))
    };
    let unwrap = |t: Tensor<I, 1>| -> I::FloatTensorPrimitive {
        match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!(),
        }
    };

    let n_in = wrap(n_p.clone());
    let qsp_in = wrap(qsp_p.clone());
    let psp_in = wrap(psp_p.clone());
    let qt_in = wrap(qt_p.clone());
    let qpt_in = wrap(qpt_p.clone());
    let length_in = wrap(length_p.clone());
    let slope_in = wrap(slope_p.clone());
    let xst_in = wrap(xst_p.clone());

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
    let area = (top_width.clone() + bottom_width.clone()) * depth.clone() / 2.0;
    // S13
    let wp = bottom_width.clone()
        + depth.clone() * (side_slope.clone().powf_scalar(2.0) + 1.0).sqrt() * 2.0;
    // S14
    let hyd_radius = area.clone() / wp.clone();
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
    let i_t_prim = sparse::spmv_primitive::<I>(pattern, qt_p.clone(), &device);
    let i_t = wrap(i_t_prim.clone());

    // S25: b_rhs = c2·i_t + c3·q_t + c4·q_prime_t
    let b_rhs =
        c2.clone() * i_t.clone() + c3.clone() * qt_in.clone() + c4.clone() * qpt_in.clone();

    // S26: A_values = assemble_primitive(c1)
    let c1_prim = unwrap(c1.clone());
    let a_values_prim = sparse::assemble_primitive::<I>(pattern, c1_prim.clone(), &device);

    // S27: x_sol = triangular_csr_solve(a_values, b_rhs) — at primitive level via dispatch.
    let b_rhs_prim = unwrap(b_rhs.clone());
    let (x_sol_prim, _saved_x) = dispatch::forward_primitive::<I>(
        pattern,
        &a_values_prim,
        &b_rhs_prim,
        &device,
        use_cuda,
    );
    let x_sol = wrap(x_sol_prim.clone());

    // S28: q_next = max(x_sol, discharge_lb)
    let q_next = x_sol.clone().clamp_min(discharge_lb);
    let q_next_prim = unwrap(q_next);

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
        depth: unwrap(depth),
        top_width: unwrap(top_width),
        side_slope: unwrap(side_slope),
        bottom_width: unwrap(bottom_width),
        hydraulic_radius: unwrap(hyd_radius),
        velocity_unclamped: unwrap(velocity_un),
        velocity_clamped: unwrap(velocity_cl),
        celerity: unwrap(celerity),
        k_muskingum: unwrap(k_muskingum),
        denom: unwrap(denom),
        c1: c1_prim,
        c2: unwrap(c2),
        c3: unwrap(c3),
        c4: unwrap(c4),
        a_values: a_values_prim,
        b_rhs: b_rhs_prim,
        i_t: i_t_prim,
        x_sol: x_sol_prim,
        ratio: unwrap(ratio),
        denominator: unwrap(denominator),
        q_eps: unwrap(q_eps),
        side_slope_raw: unwrap(side_slope_raw),
        bw_raw: unwrap(bw_raw),
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
