//! Hand-written backward of the power-law channel section.
//!
//! The section is `T = p d^q`, `A = T d / (q + 1)`, `R = d / (q + 1)`, with
//! `d = max(exp(e ln_r), d_lb)` and `v = Q / A`, `c = v_cl / (e (q + 1))`.
//! Here `q` is `q_eps`, `e = 3 / (3q + 3g + 5)` and `ln_r` is the saved RATIO.
//!
//! [`section_backward`] runs the labels in this order: B17p (celerity), B16
//! (velocity band), B15p (`v = Q / A`), B12p (`A = T d / (q + 1)`), B7
//! (`T = p d^q`), B6p (depth and its floor) and B5 (the exponent `e`).
//! It reuses the forward's exact ops for `e`, `area` and `beta`, so the
//! recomputed values match the forward bit for bit.
//!
//! Tie rule: a clamp bound is inactive at an exact tie, so the gradient passes
//! (see `clamp_rule`). The depth floor reads the PRE-clamp depth.
//!
//! The Cunge X term (B19) is folded in by the caller: its `dX/dc` term joins
//! `gcelerity` before this function, and its `dX/dT` term joins `gT` here.
//!
//! Leakance: zeta reads the post-clamp depth, `q_eps` and `p`. Its depth
//! gradient joins the depth total after every other depth term and BEFORE the
//! depth-floor mask, so the mask cuts it as it cuts the others. Its `q_eps`
//! and `p` gradients join `gq_eps` and `gp` after their other terms.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorPrimitive};

use crate::routing::clamp_rule::{ceiling_active, floor_active};
use crate::routing::mmc_op::{TimestepState, ZetaGeomGrads};

/// Gradients of the section inputs. `gq_t` is the part of `dL/dQ_t` that flows
/// through the section (velocity and depth). `ggamma` is `Some` only when
/// gamma is a learned per-reach field.
pub(crate) struct SectionGrads<I: Backend> {
    pub gn: Tensor<I, 1>,
    pub gp: Tensor<I, 1>,
    pub gq_eps: Tensor<I, 1>,
    pub gq_t: Tensor<I, 1>,
    pub ggamma: Option<Tensor<I, 1>>,
}

/// Backward of the power-law section. `gcelerity` is `dL/dc` with B18 and the
/// B19 `dX/dc` term already folded in. `g_top_width_from_x` is B19's `dX/dT`.
/// `zeta_geom` is the leakance term's depth, `q_eps` and `p` gradients.
pub(crate) fn section_backward<I: Backend>(
    state: &TimestepState<I>,
    gcelerity: Tensor<I, 1>,
    g_top_width_from_x: Option<Tensor<I, 1>>,
    zeta_geom: Option<&ZetaGeomGrads<I>>,
) -> SectionGrads<I> {
    let wrap = |p: I::FloatTensorPrimitive| -> Tensor<I, 1> {
        Tensor::from_primitive(TensorPrimitive::Float(p))
    };
    let n = wrap(state.n.clone());
    let p = wrap(state.p_spatial.clone());
    let q_t = wrap(state.q_t.clone());
    let d = wrap(state.depth.clone());
    let t = wrap(state.top_width.clone());
    let v_un = wrap(state.velocity_unclamped.clone());
    let v_cl = wrap(state.velocity_clamped.clone());
    let q_eps = wrap(state.q_eps.clone());
    let ln_r = wrap(state.ratio.clone());
    let gamma_t = state.gamma_t.clone().map(wrap);
    let d_ref = state.d_ref;
    let depth_lb = state.depth_lb;
    let velocity_lb = state.velocity_lb;

    let qp1 = q_eps.clone() + 1.0;
    // The forward's exact ops for the exponent.
    let e = match gamma_t.as_ref() {
        Some(g) => (q_eps.clone() * 3.0 + g.clone() * 3.0 + 5.0).recip() * 3.0,
        None => (q_eps.clone() * 3.0 + (5.0_f32 + 3.0_f32 * state.gamma)).recip() * 3.0,
    };
    let two_plus_three_gamma = match gamma_t.as_ref() {
        Some(g) => g.clone() * 3.0 + 2.0,
        None => q_eps.zeros_like() + (2.0 + 3.0 * state.gamma),
    };
    // The forward's op for A, so this is bitwise its A.
    let area = t.clone() * d.clone() / qp1.clone();
    let beta = (e.clone() * qp1.clone()).recip();

    // B17p: c = v_cl * beta;  dbeta/dq = -(2 + 3g) / (3 qp1^2);  dbeta/dg = 1/qp1
    let gv_cl = gcelerity.clone() * beta;
    let gbeta = gcelerity * v_cl;
    let gq_from_beta = -gbeta.clone() * two_plus_three_gamma / (qp1.clone() * qp1.clone() * 3.0);
    let gg_from_beta = gamma_t.as_ref().map(|_| gbeta.clone() / qp1.clone());

    // B16: velocity band, read from the pre-clamp velocity.
    let gv_un = gv_cl.mask_fill(
        floor_active(v_un.clone(), velocity_lb).bool_or(ceiling_active(v_un.clone(), 15.0)),
        0.0,
    );

    // B15p: v = Q / A
    let gq_t_from_v = gv_un.clone() / area.clone();
    let garea = -gv_un * v_un / area.clone();

    // B12p: A = T d / qp1. B19's dX/dT joins gT here, before B7.
    let gt = garea.clone() * d.clone() / qp1.clone();
    let gt = match g_top_width_from_x {
        Some(x) => gt + x,
        None => gt,
    };
    let gd_from_area = garea.clone() * t.clone() / qp1.clone();
    let gq_from_area = -garea * area / qp1.clone();

    // B7: T = p d^q
    let gp_from_t = gt.clone() * t.clone() / p.clone();
    let gd_from_t = gt.clone() * t.clone() * q_eps.clone() / d.clone();
    let gq_from_t = gt * t * d.clone().log();

    // B6p: d = max(exp(e ln_r), d_lb). The mask reads the PRE-clamp depth.
    let floor = floor_active((ln_r.clone() * e.clone()).exp(), depth_lb);
    let gd_total = gd_from_area + gd_from_t;
    // Zeta's depth gradient joins here, before the floor mask below.
    let gd_total = match zeta_geom {
        Some(zg) => gd_total + zg.g_depth.clone(),
        None => gd_total,
    };
    let g_lnd = gd_total * d;
    let g_lnr = g_lnd.clone() * e.clone();
    let g_e = g_lnd * ln_r;
    // Cut AFTER each product: Q_t = 0 gives ln_r = -inf and 0/0.
    let cut = |x: Tensor<I, 1>| x.mask_fill(floor.clone(), 0.0);
    let gq_t_from_d = cut(g_lnr.clone() / q_t);
    let gn_from_d = cut(g_lnr.clone() / n);
    let gp_from_d = cut(-g_lnr.clone() / p);
    let gq_from_lnr = cut(g_lnr.clone() * (5.0_f32 / 3.0_f32) / qp1);

    // B5: de/dq = de/dg = -e^2
    let ge = -g_e * e.clone() * e;
    let gq_from_e = cut(ge.clone());
    let gg_from_e = gamma_t.as_ref().map(|_| cut(ge.clone()));
    let gg_from_lnr = (gamma_t.is_some() && d_ref != 1.0).then(|| cut(g_lnr * d_ref.ln()));

    let ggamma = match (gg_from_beta, gg_from_e) {
        (Some(a), Some(b)) => {
            let s = a + b;
            Some(match gg_from_lnr {
                Some(c) => s + c,
                None => s,
            })
        }
        (None, None) => None,
        _ => unreachable!("gamma_t is Some for both terms or for neither"),
    };

    let gp = gp_from_t + gp_from_d;
    let gp = match zeta_geom {
        Some(zg) => gp + zg.g_p_spatial.clone(),
        None => gp,
    };
    let gq_eps = gq_from_beta + gq_from_area + gq_from_t + gq_from_lnr + gq_from_e;
    let gq_eps = match zeta_geom {
        Some(zg) => gq_eps + zg.g_q_eps.clone(),
        None => gq_eps,
    };

    SectionGrads {
        gn: gn_from_d,
        gp,
        gq_eps,
        gq_t: gq_t_from_v + gq_t_from_d,
        ggamma,
    }
}
