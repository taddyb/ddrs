//! The fused stage timestep: one autodiff node per routing hour.
//!
//! The definition is `StageBlock::step_reference` (`src/routing/stage.rs`).
//! Forward: every line keeps that step's op order and association, so
//! the forward is bitwise. Block names (H0 to H5, 1a to 9) follow the order
//! of the blocks in `step_reference`. Adjoints: hand-written host adjoints of every block
//! (B1 to B8, BH0 to BH3), composed by [`stage_hour_backward`], and the op
//! [`StageTimestepOp`] (`Backward<B, 7>`, parents `n, q, p, h_t, q_t,
//! q_next_mc, q'`) built by [`stage_timestep`].
//!
//! # CPU or CUDA: one choice per op
//!
//! [`stage_timestep`] (and `tree_lu_solve` in `src/sparse/tree_lu.rs`) pick the
//! backend once, on `effective_use_cuda`. The host block fns in this file are
//! the CPU path and the reference for the device mirrors. The device arms live
//! in the CUDA kernels for the stage op, once they exist. The orchestration
//! fns [`stage_hour_forward`] and [`stage_hour_backward`] loop over substeps
//! only and call the host fns directly. Until the CUDA path exists, the CUDA arm
//! is `unimplemented!`. `use_cuda: true` on a backend that is not CUDA falls
//! back to the host path.
//!
//! # Host tie rule per clamp
//!
//! This table is the binding tie rule for both arms. Ties pass for every clamp
//! gradient, as Burn's default does. Strict masks apply only to the constant
//! indicators. A kernel must use the operator of the host line (strict or not).
//! `clamp_min(x, lb)` is `x < lb ? lb : x`, so a NaN passes through.
//!
//! | Line | Forward | Backward mask | At the tie |
//! |---|---|---|---|
//! | 1a depth floor | `clamp_min(x, lb)` = `x < lb ? lb : x` | `!(x < lb)` | passes |
//! | 1c Bates max | `h < hd ? hd : h` | to `h` if `!(h < hd)`, to `hd` if `h < hd` | goes to `h` |
//! | 1d edge depth | `clamp_min(der, lb)` | `!(der < lb)` | passes |
//! | 1e `above`, `up` | `[der > lb]`, `[(h - hd) >= 0]` | constants | `above = 0` at `der == lb` (the kappa term is off while `d_e` passes) |
//! | 2a side slope | `clamp(ssr, 0.5, 50)` | `ssr >= 0.5 && ssr <= 50` | passes at both ends (0 for NaN) |
//! | 2b bottom width | `clamp_min(bwr, bwlb)` | `!(bwr < bwlb)` | passes |
//! | 2d, 2e `sa`, `ba` | `[ssr > 0.5] * [ssr < 50]`, `[bwr > bwlb]` | constants | 0 at the tie (the derivative terms are off) |
//! | 3a `abs(Q)` | `q.abs()` | `sign0(q)` | 0 at `Q = 0` |
//! | 3f `a`, `b` | `clamp_min(ra, 0) * f`, `clamp_min(rb, 0) * f` | `!(ra < 0)`, `!(rb < 0)` | passes |
//! | 4b table release | `clamp_min(ex, 0)`; `act = [ex > 0]` | `!(ex < 0)`; `act` constant | the release passes to `h`; the `R'` part is 0 |
//! | 4d `R'` | `clamp_min(rpr, 0)` | `!(rpr < 0)` | passes |
//! | 7b stage floor | `hs < hf ? hf : hs` | `!(hs < hf)` | passes (to `h_solved`) |
//! | table rows | `keep = (-m) + 1` | `* keep` on the head `n`, `p`, `q` | exactly 0 (finite cotangent) |
//! | MC rows | `* in_stage` (BH0, BH3 `q'`) | | exactly 0 |
// `!(x < y)` is kept on purpose: with a NaN it is true, as in the Burn ops this file mirrors.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::sync::Arc;

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

use crate::sparse::dispatch::effective_use_cuda;
use crate::sparse::tree_lu::{tree_lu_factor, tree_lu_grad_a_host, TreeFactors, CUDA_TODO};
use crate::sparse::{back_sub_upper_transposed, forward_sub_lower, primitive_to_vec, CsrPattern};

// Scalars the reference passes as f64 literals; Burn casts them to f32.
pub(crate) const F43: f32 = (4.0f64 / 3.0) as f32;
pub(crate) const F23: f32 = (2.0f64 / 3.0) as f32;
pub(crate) const F53: f32 = (5.0f64 / 3.0) as f32;

/// Host copies of the block's constants. Built once per (block, pattern,
/// table) and shared by `Arc`. Float fields are read back from the device
/// constants, so both paths see the same bits.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct StageOpConsts {
    pub n: usize,
    pub nnz: usize,
    pub substeps: u32,
    pub theta: f32,
    pub theta_c: f32,
    pub g: f32,
    pub depth_lb: f32,
    pub bw_lb: f32,
    pub dt: f32,
    pub prescribed: bool,
    pub has_table: bool,
    /// The downstream row; the row itself at a network outlet.
    pub down: Vec<usize>,
    pub in_stage: Vec<f32>,
    pub friction: Vec<f32>,
    pub dam: Vec<f32>,
    pub ctrl: Vec<f32>,
    pub nd: Vec<f32>,
    pub w_ctrl: Vec<f32>,
    pub w_edge: Vec<f32>,
    pub has_down: Vec<f32>,
    pub outlet: Vec<f32>,
    pub z_mid: Vec<f32>,
    pub z_down: Vec<f32>,
    pub length: Vec<f32>,
    pub edge_length: Vec<f32>,
    pub dam_t: Vec<f32>,
    /// `sqrt(s0)`: exact, so one sqrt per block equals the reference's per-step sqrt.
    pub s0_sqrt: Vec<f32>,
    /// Table constants; empty without a table.
    pub t_mask: Vec<f32>,
    pub t_tw: Vec<f32>,
    pub t_n: Vec<f32>,
    pub t_pool: Vec<f32>,
    pub row: Vec<usize>,
    pub col: Vec<usize>,
    pub diag_slot: Vec<f32>,
    pub adj_stage: Vec<f32>,
    pub adj_mc: Vec<f32>,
    pub adj_friction: Vec<f32>,
    pub dam_slot: Vec<f32>,
}

/// The hour prologue (H0 to H3), length `n` each.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct HourHost {
    pub p: Vec<f32>,
    pub q_eps: Vec<f32>,
    pub n_eff: Vec<f32>,
    pub n2: Vec<f32>,
    pub qp_stage: Vec<f32>,
    pub i_t: Vec<f32>,
    pub i_delta: Vec<f32>,
    pub q_next_mc: Vec<f32>,
    pub b0: f32,
    pub b1: f32,
}

/// Blocks 1a to 1e.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct DepthOut {
    pub x: Vec<f32>,
    pub depth: Vec<f32>,
    pub h_down: Vec<f32>,
    pub der: Vec<f32>,
    pub d_edge: Vec<f32>,
    pub s_up: Vec<f32>,
    pub s_dn: Vec<f32>,
}

/// Blocks 2a to 2g at one depth. `out_*` is the section after the table blend
/// (the head itself without a table).
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct SectionHost {
    pub pw: Vec<f32>,
    pub tw: Vec<f32>,
    pub ss_raw: Vec<f32>,
    pub ss: Vec<f32>,
    pub bw_raw: Vec<f32>,
    pub bw: Vec<f32>,
    pub area: Vec<f32>,
    pub root: Vec<f32>,
    pub perim: Vec<f32>,
    pub radius: Vec<f32>,
    pub tw_d: Vec<f32>,
    pub ss_d: Vec<f32>,
    pub bw_d: Vec<f32>,
    pub a_d: Vec<f32>,
    pub p_d: Vec<f32>,
    pub out_tw: Vec<f32>,
    pub out_area: Vec<f32>,
    pub out_perim: Vec<f32>,
    pub out_radius: Vec<f32>,
    pub out_a_d: Vec<f32>,
    pub out_p_d: Vec<f32>,
}

/// Blocks 3a to 3f.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct NewtonOut {
    pub phi: Vec<f32>,
    pub dt_phi: Vec<f32>,
    pub den: Vec<f32>,
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
    pub phi_d: Vec<f32>,
    pub dh_edge: Vec<f32>,
    pub k1: Vec<f32>,
    pub kappa: Vec<f32>,
    pub q_star: Vec<f32>,
    pub ra: Vec<f32>,
    pub rb: Vec<f32>,
    pub a_up: Vec<f32>,
    pub b_down: Vec<f32>,
}

/// Blocks 4a to 4d. `ex` and `act` are empty without a table.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ControlOut {
    pub rd_head: Vec<f32>,
    pub rpd_head: Vec<f32>,
    pub ex: Vec<f32>,
    pub act: Vec<f32>,
    pub rd: Vec<f32>,
    pub rpd: Vec<f32>,
    pub kk: Vec<f32>,
    pub w: Vec<f32>,
    pub dk_dd: Vec<f32>,
    pub r_t: Vec<f32>,
    pub rpr: Vec<f32>,
    pub r_prime: Vec<f32>,
}

/// H4 and blocks 1 to 6 of one substep.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct Assembled {
    pub i_mid: Vec<f32>,
    pub hb0: f32,
    pub hb1: f32,
    pub dt_s: f32,
    pub depth: DepthOut,
    pub edge: SectionHost,
    pub store: SectionHost,
    pub d_s: Vec<f32>,
    pub newton: NewtonOut,
    pub control: ControlOut,
    /// Length `2 * nnz`.
    pub a_values: Vec<f32>,
    pub r: Vec<f32>,
}

/// Blocks 7b to 9. `mass` is `[storage, net, floor, gross, mc_in]`.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct Readout {
    pub h_solved: Vec<f32>,
    pub h_new: Vec<f32>,
    pub dh_down: Vec<f32>,
    pub q_new: Vec<f32>,
    pub mass: [f32; 5],
}

/// One substep's start state, its solve output and its factors.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SubstepSaved {
    pub h: Vec<f32>,
    pub qc: Vec<f32>,
    pub dh: Vec<f32>,
    pub factors: TreeFactors,
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct HourForward {
    pub h_end: Vec<f32>,
    pub q_out: Vec<f32>,
    pub last: Assembled,
    pub mass: [f32; 5],
    /// Empty unless `save`.
    pub subs: Vec<SubstepSaved>,
}

// ------------------------------------------------------------------ helpers

/// `Σ_{k→i} x[col[k]]·w[k]`, in slot order (the order `scatter(.., Add)` uses).
fn spmv(c: &StageOpConsts, x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; c.n];
    for k in 0..c.nnz {
        out[c.row[k]] += x[c.col[k]] * w[k];
    }
    out
}

fn sum(v: impl Iterator<Item = f32>) -> f32 {
    v.fold(0.0f32, |a, b| a + b)
}

/// Burn's `clamp_min`: `x < lb ? lb : x`.
#[inline]
fn clamp_min(x: f32, lb: f32) -> f32 {
    if x < lb {
        lb
    } else {
        x
    }
}

/// Burn's `clamp(lo, hi)`: `clamp_min(clamp_max(x, hi), lo)`.
#[inline]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    clamp_min(if x > hi { hi } else { x }, lo)
}

#[inline]
fn ind(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// `(-m) + 1`.
#[inline]
fn keep_of(m: f32) -> f32 {
    (-m) + 1.0
}

// ----------------------------------------------------------------- prologue

/// H0 to H3.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn stage_prologue_host(
    c: &StageOpConsts,
    n: &[f32],
    q: &[f32],
    p: &[f32],
    q_t: &[f32],
    q_next_mc: &[f32],
    q_prime: &[f32],
    b0: f32,
    b1: f32,
) -> HourHost {
    let nn = c.n;
    let i_t = spmv(c, q_t, &c.adj_mc);
    let i_n = spmv(c, q_next_mc, &c.adj_mc);
    let i_delta: Vec<f32> = (0..nn).map(|i| i_n[i] - i_t[i]).collect();
    let q_eps: Vec<f32> = (0..nn).map(|i| q[i] + 1e-6f32).collect();
    let n_eff: Vec<f32> = if c.has_table {
        (0..nn).map(|i| n[i] * keep_of(c.t_mask[i]) + c.t_n[i] * c.t_mask[i]).collect()
    } else {
        n.to_vec()
    };
    let n2: Vec<f32> = (0..nn).map(|i| n_eff[i] * n_eff[i]).collect();
    let qp_stage: Vec<f32> = (0..nn).map(|i| q_prime[i] * c.in_stage[i]).collect();
    HourHost {
        p: p.to_vec(),
        q_eps,
        n_eff,
        n2,
        qp_stage,
        i_t,
        i_delta,
        q_next_mc: q_next_mc.to_vec(),
        b0,
        b1,
    }
}

// ------------------------------------------------------------------- blocks

/// 1a to 1e.
#[doc(hidden)]
pub fn depth_host(c: &StageOpConsts, h: &[f32], hb0: f32) -> DepthOut {
    let n = c.n;
    let lb = c.depth_lb;
    let mut o = DepthOut {
        x: vec![0.0; n],
        depth: vec![0.0; n],
        h_down: vec![0.0; n],
        der: vec![0.0; n],
        d_edge: vec![0.0; n],
        s_up: vec![0.0; n],
        s_dn: vec![0.0; n],
    };
    for i in 0..n {
        let x = h[i] - c.z_mid[i];
        o.x[i] = x;
        o.depth[i] = clamp_min(x, lb);
        let mut hd = h[c.down[i]];
        if c.prescribed {
            // hd·((−outlet) + 1) + outlet·hb0, outlet exactly 0 or 1
            hd = if c.outlet[i] == 1.0 { hb0 } else { hd };
        }
        o.h_down[i] = hd;
        let mh = if h[i] < hd { hd } else { h[i] };
        let mz = if c.z_mid[i] < c.z_down[i] { c.z_down[i] } else { c.z_mid[i] };
        let der = mh - mz;
        o.der[i] = der;
        o.d_edge[i] = clamp_min(der, lb);
        let above = ind(der > lb);
        let up = ind((h[i] - hd) >= 0.0);
        o.s_up[i] = up * above;
        o.s_dn[i] = keep_of(up) * above;
    }
    o
}

/// 2a to 2g at the depths `d`.
#[doc(hidden)]
pub fn section_host(c: &StageOpConsts, p: &[f32], q_eps: &[f32], d: &[f32]) -> SectionHost {
    let n = c.n;
    let bwlb = c.bw_lb;
    let v = || vec![0.0f32; n];
    let mut s = SectionHost {
        pw: v(),
        tw: v(),
        ss_raw: v(),
        ss: v(),
        bw_raw: v(),
        bw: v(),
        area: v(),
        root: v(),
        perim: v(),
        radius: v(),
        tw_d: v(),
        ss_d: v(),
        bw_d: v(),
        a_d: v(),
        p_d: v(),
        out_tw: v(),
        out_area: v(),
        out_perim: v(),
        out_radius: v(),
        out_a_d: v(),
        out_p_d: v(),
    };
    for i in 0..n {
        let (e, dd) = (q_eps[i], d[i]);
        // 2a
        let pw = dd.powf(e);
        let tw = p[i] * pw;
        let ssr = (tw * e) / (dd * 2.0);
        let ss = clamp(ssr, 0.5, 50.0);
        // 2b
        let bwr = tw - (ss * dd) * 2.0;
        let bw = clamp_min(bwr, bwlb);
        // 2c
        let area = ((tw + bw) * dd) / 2.0;
        let rt = (ss * ss + 1.0).sqrt();
        let perim = bw + (dd * rt) * 2.0;
        let radius = area / perim;
        // 2d
        let twd = (e * tw) / dd;
        let sa = ind(ssr > 0.5) * ind(ssr < 50.0);
        let ssd = ((ssr / dd) * ((twd / tw) * dd - 1.0)) * sa;
        // 2e
        let ba = ind(bwr > bwlb);
        let bwd = ((twd - ss * 2.0) - (dd * ssd) * 2.0) * ba;
        // 2f
        let ad = (tw + bw) / 2.0 + (dd * (twd + bwd)) / 2.0;
        let pd = (bwd + rt * 2.0) + (((dd * ss) * ssd) * 2.0) / rt;
        s.pw[i] = pw;
        s.tw[i] = tw;
        s.ss_raw[i] = ssr;
        s.ss[i] = ss;
        s.bw_raw[i] = bwr;
        s.bw[i] = bw;
        s.area[i] = area;
        s.root[i] = rt;
        s.perim[i] = perim;
        s.radius[i] = radius;
        s.tw_d[i] = twd;
        s.ss_d[i] = ssd;
        s.bw_d[i] = bwd;
        s.a_d[i] = ad;
        s.p_d[i] = pd;
        // 2g
        if c.has_table {
            let (m, ttw) = (c.t_mask[i], c.t_tw[i]);
            let keep = keep_of(m);
            let area_r = ttw * dd;
            let perim_r = ttw + dd * 2.0;
            let radius_r = area_r / perim_r;
            s.out_tw[i] = tw * keep + ttw * m;
            s.out_area[i] = area * keep + area_r * m;
            s.out_perim[i] = perim * keep + perim_r * m;
            s.out_radius[i] = radius * keep + radius_r * m;
            s.out_a_d[i] = ad * keep + ttw * m;
            s.out_p_d[i] = pd * keep + (1.0f32 * 2.0) * m;
        } else {
            s.out_tw[i] = tw;
            s.out_area[i] = area;
            s.out_perim[i] = perim;
            s.out_radius[i] = radius;
            s.out_a_d[i] = ad;
            s.out_p_d[i] = pd;
        }
    }
    s
}

/// 3a to 3f.
#[doc(hidden)]
pub fn newton_host(
    c: &StageOpConsts,
    dt_s: f32,
    h: &[f32],
    qc: &[f32],
    n2: &[f32],
    dep: &DepthOut,
    edge: &SectionHost,
) -> NewtonOut {
    let n = c.n;
    let th = c.theta;
    let v = || vec![0.0f32; n];
    let mut o = NewtonOut {
        phi: v(),
        dt_phi: v(),
        den: v(),
        alpha: v(),
        beta: v(),
        phi_d: v(),
        dh_edge: v(),
        k1: v(),
        kappa: v(),
        q_star: v(),
        ra: v(),
        rb: v(),
        a_up: v(),
        b_down: v(),
    };
    let gdt = c.g * dt_s;
    for i in 0..n {
        let (a_e, r_e, ad_e, pd_e, p_e) =
            (edge.out_area[i], edge.out_radius[i], edge.out_a_d[i], edge.out_p_d[i], edge.out_perim[i]);
        let f = c.friction[i];
        let q = qc[i];
        // 3a
        let phi = ((n2[i] * q.abs()) * c.g) / (a_e * r_e.powf(F43));
        let tau = phi * dt_s;
        let den = tau * 2.0 + 1.0;
        // 3b
        let alpha = ((tau + 1.0) / den) * f;
        let beta = ((a_e * gdt) / (c.edge_length[i] * den)) * f;
        // 3c
        let phi_d = ad_e / a_e + (ad_e / a_e - pd_e / p_e) * F43;
        // 3d
        let dhe = h[i] - dep.h_down[i];
        let k1 = (tau * q) * phi_d + ((ad_e * dhe) * gdt) / c.edge_length[i];
        let kappa = (k1 / den) * f;
        // 3e
        let q_star = alpha * q + beta * dhe;
        // 3f
        let ra = beta * th + kappa * dep.s_up[i];
        let a = clamp_min(ra, 0.0) * f;
        let rb = beta * th - kappa * dep.s_dn[i];
        let b = clamp_min(rb, 0.0) * f;
        o.phi[i] = phi;
        o.dt_phi[i] = tau;
        o.den[i] = den;
        o.alpha[i] = alpha;
        o.beta[i] = beta;
        o.phi_d[i] = phi_d;
        o.dh_edge[i] = dhe;
        o.k1[i] = k1;
        o.kappa[i] = kappa;
        o.q_star[i] = q_star;
        o.ra[i] = ra;
        o.rb[i] = rb;
        o.a_up[i] = a;
        o.b_down[i] = b;
    }
    o
}

/// 4a to 4d.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn control_host(
    c: &StageOpConsts,
    dt_s: f32,
    h: &[f32],
    p: &[f32],
    q_eps: &[f32],
    n_eff: &[f32],
    depth: &[f32],
    store: &SectionHost,
    d_s: &[f32],
) -> ControlOut {
    let n = c.n;
    let v = || vec![0.0f32; n];
    let mut o = ControlOut {
        rd_head: v(),
        rpd_head: v(),
        ex: if c.has_table { v() } else { vec![] },
        act: if c.has_table { v() } else { vec![] },
        rd: v(),
        rpd: v(),
        kk: v(),
        w: v(),
        dk_dd: v(),
        r_t: v(),
        rpr: v(),
        r_prime: v(),
    };
    for i in 0..n {
        // 4a
        let e1 = q_eps[i] + 1.0;
        let rd_head = (((c.length[i] * p[i]) * depth[i].powf(e1)) / e1) / c.dam_t[i];
        let rpd_head = (d_s[i] * dt_s) / c.dam_t[i];
        let (mut rd, mut rpd) = (rd_head, rpd_head);
        // 4b
        if c.has_table {
            let m = c.t_mask[i];
            let keep = keep_of(m);
            let ex = h[i] - c.t_pool[i];
            let act = ind(ex > 0.0);
            let rtb = ((c.t_tw[i] * c.length[i]) * clamp_min(ex, 0.0)) / c.dam_t[i];
            let rptb = rpd_head * act;
            rd = rd_head * keep + rtb * m;
            rpd = rpd_head * keep + rptb * m;
            o.ex[i] = ex;
            o.act[i] = act;
        }
        // 4c
        let (a_s, r_s, ad_s, pd_s, p_s) =
            (store.out_area[i], store.out_radius[i], store.out_a_d[i], store.out_p_d[i], store.out_perim[i]);
        let kk = (a_s * r_s.powf(F23)) / n_eff[i];
        let rnd = kk * c.s0_sqrt[i];
        let w = (ad_s / a_s) * F53 - (pd_s / p_s) * F23;
        let dk = kk * w;
        let rpnd = dk * c.s0_sqrt[i];
        // 4d
        let r_t = c.dam[i] * rd + c.nd[i] * rnd;
        let rpr = c.dam[i] * rpd + c.nd[i] * rpnd;
        o.rd_head[i] = rd_head;
        o.rpd_head[i] = rpd_head;
        o.rd[i] = rd;
        o.rpd[i] = rpd;
        o.kk[i] = kk;
        o.w[i] = w;
        o.dk_dd[i] = dk;
        o.r_t[i] = r_t;
        o.rpr[i] = rpr;
        o.r_prime[i] = clamp_min(rpr, 0.0);
    }
    o
}

/// 5a to 5d. Returns `2 * nnz` values.
#[doc(hidden)]
pub fn matrix_host(c: &StageOpConsts, d_s: &[f32], a_up: &[f32], b_down: &[f32], r_prime: &[f32]) -> Vec<f32> {
    let (n, nnz) = (c.n, c.nnz);
    let (th, thc) = (c.theta, c.theta_c);
    let be: Vec<f32> = (0..nnz).map(|k| b_down[c.col[k]] * c.adj_friction[k]).collect();
    let mut sib = vec![0.0f32; n];
    for k in 0..nnz {
        sib[c.row[k]] += be[k];
    }
    let diag: Vec<f32> = (0..n)
        .map(|i| {
            let dst = d_s[i] + (a_up[i] + sib[i]) * th;
            (dst * c.in_stage[i] + c.w_ctrl[i] * r_prime[i]) + keep_of(c.in_stage[i])
        })
        .collect();
    let mut out = vec![0.0f32; 2 * nnz];
    for k in 0..nnz {
        let ae = a_up[c.col[k]] * c.adj_friction[k];
        let lowf = (-ae) * th;
        let upf = (-be[k]) * th;
        let offd = (r_prime[c.col[k]] * c.dam_slot[k]) * (-thc);
        out[k] = (lowf + offd) + diag[c.row[k]] * c.diag_slot[k];
        out[nnz + k] = upf;
    }
    out
}

/// 6a to 6c.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn rhs_host(
    c: &StageOpConsts,
    qc: &[f32],
    q_star: &[f32],
    r_t: &[f32],
    b_down: &[f32],
    qp: &[f32],
    i_mid: &[f32],
    hb0: f32,
    hb1: f32,
) -> Vec<f32> {
    let n = c.n;
    let th = c.theta;
    let eo: Vec<f32> = (0..n)
        .map(|i| {
            let fo = c.friction[i] * ((q_star[i] * th) + qc[i] * (1.0 - th));
            let co = c.w_ctrl[i] * r_t[i] + (c.ctrl[i] - c.w_ctrl[i]) * qc[i];
            fo + co
        })
        .collect();
    let mut ine = vec![0.0f32; n];
    for k in 0..c.nnz {
        ine[c.row[k]] += eo[c.col[k]] * c.adj_stage[k];
    }
    (0..n)
        .map(|i| {
            let mut r = ((qp[i] + i_mid[i]) + ine[i]) - eo[i];
            if c.prescribed {
                r += (c.outlet[i] * b_down[i]) * (th * (hb1 - hb0));
            }
            r
        })
        .collect()
}

// ------------------------------------------------------------ the substep

/// H4 and blocks 1 to 6 of substep `s`, from the start state `(h, qc)`.
#[doc(hidden)]
pub fn stage_assemble_host(c: &StageOpConsts, hour: &HourHost, s: usize, h: &[f32], qc: &[f32]) -> Assembled {
    let n = c.n;
    let subs = c.substeps.max(1) as usize;
    let dt_s = c.dt / subs as f32;
    let (f0, f1) = (s as f32 / subs as f32, (s + 1) as f32 / subs as f32);
    // H4
    let i_mid: Vec<f32> =
        (0..n).map(|i| (hour.i_t[i] * 2.0 + hour.i_delta[i] * (f0 + f1)) * 0.5).collect();
    let (hb0, hb1) = (hour.b0 + (hour.b1 - hour.b0) * f0, hour.b0 + (hour.b1 - hour.b0) * f1);
    let depth = depth_host(c, h, hb0);
    let edge = section_host(c, &hour.p, &hour.q_eps, &depth.d_edge);
    let store = section_host(c, &hour.p, &hour.q_eps, &depth.depth);
    // 2h
    let d_s: Vec<f32> = (0..n).map(|i| (store.out_tw[i] * c.length[i]) / dt_s).collect();
    let newton = newton_host(c, dt_s, h, qc, &hour.n2, &depth, &edge);
    let control = control_host(c, dt_s, h, &hour.p, &hour.q_eps, &hour.n_eff, &depth.depth, &store, &d_s);
    let a_values = matrix_host(c, &d_s, &newton.a_up, &newton.b_down, &control.r_prime);
    let r = rhs_host(c, qc, &newton.q_star, &control.r_t, &newton.b_down, &hour.qp_stage, &i_mid, hb0, hb1);
    Assembled { i_mid, hb0, hb1, dt_s, depth, edge, store, d_s, newton, control, a_values, r }
}

/// 7b to 9, from the solve output `dh`.
#[doc(hidden)]
pub fn stage_readout_host(
    c: &StageOpConsts,
    hour: &HourHost,
    h: &[f32],
    qc: &[f32],
    asm: &Assembled,
    dh: &[f32],
) -> Readout {
    let n = c.n;
    let lb = c.depth_lb;
    let nw = &asm.newton;
    let mut h_solved = vec![0.0f32; n];
    let mut h_new = vec![0.0f32; n];
    let mut dh_down = vec![0.0f32; n];
    let mut q_new = vec![0.0f32; n];
    for i in 0..n {
        // 7b
        let hs = h[i] + dh[i];
        let hf = c.z_mid[i] + lb;
        h_solved[i] = hs;
        h_new[i] = if hs < hf { hf } else { hs };
        // 8a
        let mut dhd = dh[c.down[i]];
        if c.prescribed {
            dhd = if c.outlet[i] == 1.0 { asm.hb1 - asm.hb0 } else { dhd };
        }
        dh_down[i] = dhd;
        // 8b
        let q_fr = (nw.q_star[i] + nw.a_up[i] * dh[i]) - nw.b_down[i] * dhd;
        let q_ctrl = (asm.control.r_t[i] + asm.control.r_prime[i] * dh[i]) * c.ctrl[i];
        q_new[i] = q_fr + q_ctrl;
    }
    // 9
    let dt_s = asm.dt_s;
    let fe: Vec<f32> = (0..n)
        .map(|i| (c.w_edge[i] * q_new[i] + keep_of(c.w_edge[i]) * qc[i]) * c.in_stage[i])
        .collect();
    let mut into = vec![0.0f32; n];
    for k in 0..c.nnz {
        into[c.row[k]] += fe[c.col[k]] * c.adj_stage[k];
    }
    let net = sum((0..n).map(|i| ((into[i] + asm.i_mid[i]) + hour.qp_stage[i]) - fe[i])) * dt_s;
    let storage = sum((0..n).map(|i| (asm.d_s[i] * (h_new[i] - h[i])) * c.in_stage[i])) * dt_s;
    let floor = sum((0..n).map(|i| (asm.d_s[i] * (h_new[i] - h_solved[i])) * c.in_stage[i])) * dt_s;
    let gross = ((sum((0..n).map(|i| fe[i].abs() * (c.in_stage[i] + c.has_down[i] * c.in_stage[i])))
        + sum(asm.i_mid.iter().map(|v| v.abs())))
        + sum(hour.qp_stage.iter().map(|v| v.abs())))
        * dt_s;
    let mc_in = sum(asm.i_mid.iter().copied()) * dt_s;
    Readout { h_solved, h_new, dh_down, q_new, mass: [storage, net, floor, gross, mc_in] }
}

// ---------------------------------------------------------------- adjoints
//
// The adjoints of the elementwise blocks (B1 to B4), one loop over reaches
// each. These are the CPU path; the device mirrors are chosen at the op. Masks use the
// forward line's comparison (see the tie table in the module doc). An empty cotangent Vec is zeros.

/// `v[i]`, or 0.0 for an empty cotangent Vec.
#[inline]
fn at(v: &[f32], i: usize) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v[i]
    }
}

/// Burn's `abs` gradient factor: `sign(0) = 0`.
#[inline]
fn sign0(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Cotangents of the six outputs of block 2g. Empty means zeros.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct SectionBar {
    pub top_width: Vec<f32>,
    pub area: Vec<f32>,
    pub perim: Vec<f32>,
    pub radius: Vec<f32>,
    pub a_d: Vec<f32>,
    pub p_d: Vec<f32>,
}

#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct SectionGrads {
    pub p: Vec<f32>,
    pub q_eps: Vec<f32>,
    pub d: Vec<f32>,
}

/// B2: the reverse of 2g to 2a at the depths `d`. `q_eps` is `q + 1e-6`, so
/// `SectionGrads::q_eps` is also the gradient to `q`.
#[doc(hidden)]
pub fn section_backward_host(
    c: &StageOpConsts,
    p: &[f32],
    q_eps: &[f32],
    d: &[f32],
    s: &SectionHost,
    bar: &SectionBar,
) -> SectionGrads {
    let n = c.n;
    let bwlb = c.bw_lb;
    let mut out = SectionGrads { p: vec![0.0; n], q_eps: vec![0.0; n], d: vec![0.0; n] };
    for i in 0..n {
        let (e, dd) = (q_eps[i], d[i]);
        let (mut g_tw, mut g_a, mut g_p) = (at(&bar.top_width, i), at(&bar.area, i), at(&bar.perim, i));
        let (mut g_r, mut g_ad, mut g_pd) = (at(&bar.radius, i), at(&bar.a_d, i), at(&bar.p_d, i));
        let mut gd = 0.0f32;
        let mut ge = 0.0f32;
        if c.has_table {
            // The rectangle depends on d; the head gets the cotangents times keep.
            let m = c.t_mask[i];
            let keep = keep_of(m);
            let ttw = c.t_tw[i];
            let perim_r = ttw + dd * 2.0;
            gd += m
                * (g_a * ttw
                    + g_p * 2.0
                    + g_r * ((ttw * perim_r - (ttw * dd) * 2.0) / (perim_r * perim_r)));
            g_tw *= keep;
            g_a *= keep;
            g_p *= keep;
            g_r *= keep;
            g_ad *= keep;
            g_pd *= keep;
        }
        let (pw, tw, ssr, ss, bwr, bw) = (s.pw[i], s.tw[i], s.ss_raw[i], s.ss[i], s.bw_raw[i], s.bw[i]);
        let (area, rt, perim) = (s.area[i], s.root[i], s.perim[i]);
        let (twd, ssd, bwd) = (s.tw_d[i], s.ss_d[i], s.bw_d[i]);
        let sa = ind(ssr > 0.5) * ind(ssr < 50.0);
        let ba = ind(bwr > bwlb);
        let (mut g_ss, mut g_bw, mut g_rt) = (0.0f32, 0.0f32, 0.0f32);
        let (mut g_twd, mut g_bwd, mut g_ssd, mut g_ssr) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        // 2f: Pd = (bwd + rt*2) + (((d*ss)*ssd)*2)/rt
        let u = ((dd * ss) * ssd) * 2.0;
        g_bwd += g_pd;
        g_rt += g_pd * 2.0;
        g_rt += g_pd * (-(u / (rt * rt)));
        let g_u = g_pd / rt;
        let g_dss_ssd = g_u * 2.0;
        let g_dss = g_dss_ssd * ssd;
        g_ssd += g_dss_ssd * (dd * ss);
        gd += g_dss * ss;
        g_ss += g_dss * dd;
        // 2f: Ad = (tw + bw)/2 + (d*(twd + bwd))/2
        g_tw += g_ad / 2.0;
        g_bw += g_ad / 2.0;
        let g_x = g_ad / 2.0;
        gd += g_x * (twd + bwd);
        g_twd += g_x * dd;
        g_bwd += g_x * dd;

        // 2e: bwd = ((twd - ss*2) - (d*ssd)*2) * ba
        let g_b = g_bwd * ba;
        g_twd += g_b;
        g_ss += g_b * (-2.0);
        let g_dssd = g_b * (-2.0);
        gd += g_dssd * ssd;
        g_ssd += g_dssd * dd;

        // 2d: ssd = ((ssr/d) * ((twd/tw)*d - 1)) * sa
        let g_s = g_ssd * sa;
        let ratio = ssr / dd;
        let u2 = (twd / tw) * dd - 1.0;
        let g_ratio = g_s * u2;
        let g_u2 = g_s * ratio;
        g_ssr += g_ratio / dd;
        gd += -g_ratio * ssr / (dd * dd);
        let g_tt = g_u2 * dd;
        gd += g_u2 * (twd / tw);
        g_twd += g_tt / tw;
        g_tw += -g_tt * twd / (tw * tw);
        // 2d: twd = (e*tw)/d
        ge += g_twd * tw / dd;
        g_tw += g_twd * e / dd;
        gd += -g_twd * (e * tw) / (dd * dd);

        // 2c: R = A/P; P = bw + (d*rt)*2; rt = sqrt(ss*ss + 1); A = ((tw + bw)*d)/2
        let g_area = g_a + g_r / perim;
        let g_perim = g_p - g_r * area / (perim * perim);
        g_bw += g_perim;
        gd += g_perim * 2.0 * rt;
        g_rt += g_perim * 2.0 * dd;
        g_ss += g_rt * ss / rt;
        g_tw += g_area * dd / 2.0;
        g_bw += g_area * dd / 2.0;
        gd += g_area * (tw + bw) / 2.0;

        // 2b: bw = clamp_min(bwr, bwlb); bwr = tw - (ss*d)*2
        let g_bwr = g_bw * ind(!(bwr < bwlb));
        g_tw += g_bwr;
        g_ss += g_bwr * (-2.0 * dd);
        gd += g_bwr * (-2.0 * ss);

        // 2a: ss = clamp(ssr, 0.5, 50); ssr = (tw*e)/(d*2); tw = p*pw; pw = d^e
        g_ssr += g_ss * ind(ssr >= 0.5 && ssr <= 50.0);
        g_tw += g_ssr * e / (dd * 2.0);
        ge += g_ssr * tw / (dd * 2.0);
        gd += -g_ssr * (tw * e) / (2.0 * dd * dd);
        out.p[i] = g_tw * pw;
        let g_pw = g_tw * p[i];
        gd += g_pw * e * dd.powf(e - 1.0);
        ge += g_pw * pw * dd.ln();

        out.q_eps[i] = ge;
        out.d[i] = gd;
    }
    out
}

/// Cotangents of the Newton block outputs `Q*`, `a`, `b`.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct NewtonBar {
    pub q_star: Vec<f32>,
    pub a_up: Vec<f32>,
    pub b_down: Vec<f32>,
}

/// `edge` holds the cotangents of the edge section's `A`, `P`, `R`, `A_d`,
/// `P_d`; `edge.top_width` stays empty.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct NewtonGrads {
    pub qc: Vec<f32>,
    pub n2: Vec<f32>,
    pub h: Vec<f32>,
    pub h_down: Vec<f32>,
    pub edge: SectionBar,
}

/// B3: the reverse of 3f to 3a.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn newton_backward_host(
    c: &StageOpConsts,
    dt_s: f32,
    h: &[f32],
    qc: &[f32],
    n2: &[f32],
    dep: &DepthOut,
    edge: &SectionHost,
    o: &NewtonOut,
    bar: &NewtonBar,
) -> NewtonGrads {
    let n = c.n;
    let _ = h; // dh_edge is saved in `o`
    let th = c.theta;
    let gdt = c.g * dt_s;
    let v = || vec![0.0f32; n];
    let mut out = NewtonGrads {
        qc: v(),
        n2: v(),
        h: v(),
        h_down: v(),
        edge: SectionBar { top_width: vec![], area: v(), perim: v(), radius: v(), a_d: v(), p_d: v() },
    };
    for i in 0..n {
        let f = c.friction[i];
        let q = qc[i];
        let (a_e, r_e, ad_e, pd_e, p_e) =
            (edge.out_area[i], edge.out_radius[i], edge.out_a_d[i], edge.out_p_d[i], edge.out_perim[i]);
        let (phi, tau, den) = (o.phi[i], o.dt_phi[i], o.den[i]);
        let (alpha, beta, phi_d, dhe, k1) = (o.alpha[i], o.beta[i], o.phi_d[i], o.dh_edge[i], o.k1[i]);
        let le = c.edge_length[i];
        let (s_up, s_dn) = (dep.s_up[i], dep.s_dn[i]);
        let (g_qs, g_a, g_b) = (at(&bar.q_star, i), at(&bar.a_up, i), at(&bar.b_down, i));

        // 3f
        let g_ra = g_a * f * ind(!(o.ra[i] < 0.0));
        let g_rb = g_b * f * ind(!(o.rb[i] < 0.0));
        let g_beta = (g_ra + g_rb) * th + g_qs * dhe;
        let g_kappa = g_ra * s_up - g_rb * s_dn;
        // 3e
        let g_alpha = g_qs * q;
        let mut g_q = g_qs * alpha;
        let mut g_dhe = g_qs * beta;
        // 3d
        let g_k1 = g_kappa * f / den;
        let mut g_den = -g_kappa * f * k1 / (den * den);
        // 3b
        let mut g_a_e = g_beta * f * gdt / (le * den);
        g_den += -g_beta * f * (a_e * gdt) / (le * den * den);
        g_den += -g_alpha * f * (tau + 1.0) / (den * den);
        let mut g_tau = g_alpha * f / den;
        // 3d: K1 = (tau*Q)*phi_d + ((Ad*dhe)*g*dt)/Le
        g_tau += g_k1 * q * phi_d;
        g_q += g_k1 * tau * phi_d;
        let g_phi_d = g_k1 * tau * q;
        let mut g_ad_e = g_k1 * dhe * gdt / le;
        g_dhe += g_k1 * ad_e * gdt / le;
        // 3a: den = tau*2 + 1
        g_tau += 2.0 * g_den;
        // 3c: phi_d = Ad/A + (Ad/A - Pd/P)*4/3
        g_ad_e += g_phi_d * (1.0 + F43) / a_e;
        g_a_e -= g_phi_d * (1.0 + F43) * ad_e / (a_e * a_e);
        out.edge.p_d[i] -= g_phi_d * F43 / p_e;
        out.edge.perim[i] += g_phi_d * F43 * pd_e / (p_e * p_e);
        // 3a: tau = phi*dt; phi = ((n2*|Q|)*g)/(A*R^(4/3))
        let g_phi = g_tau * dt_s;
        let cc = c.g / (a_e * r_e.powf(F43));
        out.n2[i] += g_phi * q.abs() * cc;
        g_q += g_phi * n2[i] * cc * sign0(q);
        g_a_e -= g_phi * phi / a_e;
        out.edge.radius[i] -= g_phi * phi * F43 / r_e;

        out.qc[i] += g_q;
        out.edge.area[i] += g_a_e;
        out.edge.a_d[i] += g_ad_e;
        out.h[i] += g_dhe;
        out.h_down[i] -= g_dhe;
    }
    out
}

/// Cotangents of `R` (`r_t`) and the clamped `R'` (`r_prime`).
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ControlBar {
    pub r_t: Vec<f32>,
    pub r_prime: Vec<f32>,
}

/// `store.top_width` stays empty: the caller adds `ḡD_s·L/Δ` (B2).
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ControlGrads {
    pub h: Vec<f32>,
    pub depth: Vec<f32>,
    pub p: Vec<f32>,
    pub q_eps: Vec<f32>,
    pub n_eff: Vec<f32>,
    pub d_s: Vec<f32>,
    pub store: SectionBar,
}

/// B4: the reverse of 4d to 4a.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn control_backward_host(
    c: &StageOpConsts,
    dt_s: f32,
    h: &[f32],
    p: &[f32],
    q_eps: &[f32],
    n_eff: &[f32],
    depth: &[f32],
    store: &SectionHost,
    d_s: &[f32],
    o: &ControlOut,
    bar: &ControlBar,
) -> ControlGrads {
    let _ = (h, d_s);
    let n = c.n;
    let v = || vec![0.0f32; n];
    let mut out = ControlGrads {
        h: v(),
        depth: v(),
        p: v(),
        q_eps: v(),
        n_eff: v(),
        d_s: v(),
        store: SectionBar { top_width: vec![], area: v(), perim: v(), radius: v(), a_d: v(), p_d: v() },
    };
    for i in 0..n {
        let (g_rt, g_rp) = (at(&bar.r_t, i), at(&bar.r_prime, i));
        let s0s = c.s0_sqrt[i];
        // 4d
        let g_rpr = g_rp * ind(!(o.rpr[i] < 0.0));
        let g_rpd = g_rpr * c.dam[i];
        let g_rpnd = g_rpr * c.nd[i];
        let mut g_rd = g_rt * c.dam[i];
        let g_rnd = g_rt * c.nd[i];
        // 4c
        let (a_s, r_s, ad_s, pd_s, p_s) =
            (store.out_area[i], store.out_radius[i], store.out_a_d[i], store.out_p_d[i], store.out_perim[i]);
        let (kk, w) = (o.kk[i], o.w[i]);
        let g_dk = g_rpnd * s0s;
        let g_kk = g_rnd * s0s + g_dk * w;
        let g_w = g_dk * kk;
        out.store.a_d[i] += g_w * F53 / a_s;
        out.store.area[i] -= g_w * F53 * ad_s / (a_s * a_s);
        out.store.p_d[i] -= g_w * F23 / p_s;
        out.store.perim[i] += g_w * F23 * pd_s / (p_s * p_s);
        out.store.area[i] += g_kk * r_s.powf(F23) / n_eff[i];
        out.store.radius[i] += g_kk * (a_s / n_eff[i]) * F23 * r_s.powf(F23 - 1.0);
        out.n_eff[i] -= g_kk * kk / n_eff[i];
        // 4b
        let mut g_rpd_head = g_rpd;
        if c.has_table {
            let m = c.t_mask[i];
            let keep = keep_of(m);
            let g_rtb = g_rd * m;
            let g_rptb = g_rpd * m;
            g_rd *= keep;
            g_rpd_head = g_rpd * keep + g_rptb * o.act[i];
            out.h[i] += g_rtb * ((c.t_tw[i] * c.length[i]) / c.dam_t[i]) * ind(!(o.ex[i] < 0.0));
        }
        // 4a
        out.d_s[i] += g_rpd_head * dt_s / c.dam_t[i];
        let e1 = q_eps[i] + 1.0;
        let u = depth[i].powf(e1);
        let lp = (c.length[i] * p[i]) / (e1 * c.dam_t[i]);
        out.p[i] += g_rd * (c.length[i] * u) / (e1 * c.dam_t[i]);
        out.depth[i] += g_rd * lp * e1 * depth[i].powf(e1 - 1.0);
        out.q_eps[i] += g_rd * (lp * u * depth[i].ln() - o.rd_head[i] / e1);
    }
    out
}

/// B1: returns `ḡh` of blocks 1a to 1e. `g_h_down` is the cotangent of the
/// `h_down` output (after the PrescribedStage mix), which B3 fills with `−ḡdhe`.
#[doc(hidden)]
pub fn depth_backward_host(
    c: &StageOpConsts,
    h: &[f32],
    dep: &DepthOut,
    g_depth: &[f32],
    g_d_edge: &[f32],
    g_h_down: &[f32],
) -> Vec<f32> {
    let n = c.n;
    let lb = c.depth_lb;
    let mut gh = vec![0.0f32; n];
    let mut ghd = vec![0.0f32; n];
    for i in 0..n {
        let hd = dep.h_down[i];
        let g_der = at(g_d_edge, i) * ind(!(dep.der[i] < lb));
        let lt = h[i] < hd;
        gh[i] += g_der * ind(!lt);
        ghd[i] += g_der * ind(lt) + at(g_h_down, i);
        gh[i] += at(g_depth, i) * ind(!(dep.x[i] < lb));
    }
    for i in 0..n {
        let g = if c.prescribed { ghd[i] * keep_of(c.outlet[i]) } else { ghd[i] };
        gh[c.down[i]] += g;
    }
    gh
}

// ---------------------------------------------------------------------------
// Adjoints of blocks B5 to B8 and the hour adjoints BH0 to BH3. One loop over reaches or
// slots each. The CUDA arms are chosen at the op, not here.
// ---------------------------------------------------------------------------

/// Cotangents out of the readout. `dh` is the cotangent the solve adjoint takes
/// (`ḡdh` of B8 and B7); `h` is `ḡh` of the start state from the floor.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ReadoutGrads {
    pub dh: Vec<f32>,
    pub h: Vec<f32>,
    pub q_star: Vec<f32>,
    pub a_up: Vec<f32>,
    pub b_down: Vec<f32>,
    pub r_t: Vec<f32>,
    pub r_prime: Vec<f32>,
}

/// B8 and B7 (before the solve): `hs` and `dhd` are recomputed as 7b and 8a.
#[doc(hidden)]
pub fn readout_backward_host(
    c: &StageOpConsts,
    h: &[f32],
    asm: &Assembled,
    dh: &[f32],
    g_h_new: &[f32],
    g_q_new: &[f32],
) -> ReadoutGrads {
    let n = c.n;
    let lb = c.depth_lb;
    let v = || vec![0.0f32; n];
    let mut o = ReadoutGrads { dh: v(), h: v(), q_star: v(), a_up: v(), b_down: v(), r_t: v(), r_prime: v() };
    let mut g_dhd = v();
    for i in 0..n {
        let gq = at(g_q_new, i);
        let mut dhd = dh[c.down[i]];
        if c.prescribed {
            dhd = if c.outlet[i] == 1.0 { asm.hb1 - asm.hb0 } else { dhd };
        }
        // B8
        o.q_star[i] = gq;
        o.a_up[i] = gq * dh[i];
        o.b_down[i] = -gq * dhd;
        o.dh[i] += gq * asm.newton.a_up[i] + (gq * c.ctrl[i]) * asm.control.r_prime[i];
        o.r_t[i] = gq * c.ctrl[i];
        o.r_prime[i] = (gq * c.ctrl[i]) * dh[i];
        let g = -gq * asm.newton.b_down[i];
        g_dhd[i] = if c.prescribed { g * keep_of(c.outlet[i]) } else { g };
        // B7: the floor
        let hs = h[i] + dh[i];
        let hf = c.z_mid[i] + lb;
        let ghs = at(g_h_new, i) * ind(!(hs < hf));
        o.h[i] = ghs;
        o.dh[i] += ghs;
    }
    // the gather's transpose
    for i in 0..n {
        o.dh[c.down[i]] += g_dhd[i];
    }
    o
}

#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct RhsGrads {
    pub qp: Vec<f32>,
    pub i_mid: Vec<f32>,
    pub q_star: Vec<f32>,
    pub qc: Vec<f32>,
    pub r_t: Vec<f32>,
    pub b_down: Vec<f32>,
}

/// B6: the reverse of 6a to 6c, given `ḡr`.
#[doc(hidden)]
pub fn rhs_backward_host(c: &StageOpConsts, asm: &Assembled, g_r: &[f32]) -> RhsGrads {
    let n = c.n;
    let th = c.theta;
    let mut g_eo: Vec<f32> = g_r.iter().map(|g| -g).collect();
    for k in 0..c.nnz {
        g_eo[c.col[k]] += g_r[c.row[k]] * c.adj_stage[k];
    }
    let mut o = RhsGrads {
        qp: g_r.to_vec(),
        i_mid: g_r.to_vec(),
        q_star: vec![0.0; n],
        qc: vec![0.0; n],
        r_t: vec![0.0; n],
        b_down: vec![0.0; n],
    };
    for i in 0..n {
        if c.prescribed {
            o.b_down[i] = (g_r[i] * c.outlet[i]) * (th * (asm.hb1 - asm.hb0));
        }
        o.q_star[i] = (g_eo[i] * c.friction[i]) * th;
        o.qc[i] = g_eo[i] * (c.friction[i] * (1.0 - th) + (c.ctrl[i] - c.w_ctrl[i]));
        o.r_t[i] = g_eo[i] * c.w_ctrl[i];
    }
    o
}

#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct MatrixGrads {
    pub d_s: Vec<f32>,
    pub a_up: Vec<f32>,
    pub b_down: Vec<f32>,
    pub r_prime: Vec<f32>,
}

/// B5: the reverse of 5d to 5a. `g_a_values` is `[ḡlow, ḡupf]`, length `2 * nnz`.
#[doc(hidden)]
pub fn matrix_backward_host(c: &StageOpConsts, g_a_values: &[f32]) -> MatrixGrads {
    let (n, nnz) = (c.n, c.nnz);
    let (th, thc) = (c.theta, c.theta_c);
    assert_eq!(g_a_values.len(), 2 * nnz, "g_a_values must have length 2 * nnz");
    let (g_low, g_upf) = g_a_values.split_at(nnz);
    // 5d: low_k = (lowf + offd) + diag[row] * diag_slot
    let mut g_diag = vec![0.0f32; n];
    for k in 0..nnz {
        g_diag[c.row[k]] += g_low[k] * c.diag_slot[k];
    }
    let mut o = MatrixGrads {
        d_s: vec![0.0; n],
        a_up: vec![0.0; n],
        b_down: vec![0.0; n],
        r_prime: vec![0.0; n],
    };
    // 5c
    let mut g_sib = vec![0.0f32; n];
    for i in 0..n {
        let g_dst = g_diag[i] * c.in_stage[i];
        o.d_s[i] += g_dst;
        o.a_up[i] += g_dst * th;
        g_sib[i] = g_dst * th;
        o.r_prime[i] += g_diag[i] * c.w_ctrl[i];
    }
    // 5b, 5a, 5c's sib scatter
    for k in 0..nnz {
        let col = c.col[k];
        o.a_up[col] += (-th * g_low[k]) * c.adj_friction[k];
        o.b_down[col] += (g_sib[c.row[k]] - th * g_upf[k]) * c.adj_friction[k];
        o.r_prime[col] += (-thc * g_low[k]) * c.dam_slot[k];
    }
    o
}

/// Hour accumulators, summed over substeps. Start from `HourAcc::zeros(n)`.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct HourAcc {
    pub i_t: Vec<f32>,
    pub i_delta: Vec<f32>,
    pub n2: Vec<f32>,
    pub n_eff: Vec<f32>,
    pub p: Vec<f32>,
    pub q_eps: Vec<f32>,
    pub qp: Vec<f32>,
}

impl HourAcc {
    pub fn zeros(n: usize) -> Self {
        let v = || vec![0.0f32; n];
        HourAcc { i_t: v(), i_delta: v(), n2: v(), n_eff: v(), p: v(), q_eps: v(), qp: v() }
    }
}

fn add2(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn add_into(acc: &mut [f32], x: &[f32]) {
    for (a, v) in acc.iter_mut().zip(x) {
        *a += v;
    }
}

/// B6 to B1 for substep `s`, composed from the block adjoints `rhs_backward_host`, `matrix_backward_host`, `control_backward_host`, `newton_backward_host`, `section_backward_host` and `depth_backward_host`. Adds into `acc`.
/// Returns `(ḡh, ḡQ)` of the substep's start state.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn assemble_backward_host(
    c: &StageOpConsts,
    hour: &HourHost,
    s: usize,
    h: &[f32],
    qc: &[f32],
    asm: &Assembled,
    rg: &ReadoutGrads,
    g_r: &[f32],
    g_a_values: &[f32],
    acc: &mut HourAcc,
) -> (Vec<f32>, Vec<f32>) {
    let n = c.n;
    let subs = c.substeps.max(1) as usize;
    let (f0, f1) = (s as f32 / subs as f32, (s + 1) as f32 / subs as f32);
    // B6, B5
    let rhs = rhs_backward_host(c, asm, g_r);
    let mat = matrix_backward_host(c, g_a_values);
    // B4: cotangents of R and R' from the readout, the RHS and the matrix
    let ctl_bar = ControlBar { r_t: add2(&rg.r_t, &rhs.r_t), r_prime: add2(&rg.r_prime, &mat.r_prime) };
    let ctl = control_backward_host(
        c, asm.dt_s, h, &hour.p, &hour.q_eps, &hour.n_eff, &asm.depth.depth, &asm.store, &asm.d_s,
        &asm.control, &ctl_bar
    );
    // B3
    let nw_bar = NewtonBar {
        q_star: add2(&rg.q_star, &rhs.q_star),
        a_up: add2(&rg.a_up, &mat.a_up),
        b_down: add2(&add2(&rg.b_down, &rhs.b_down), &mat.b_down),
    };
    let nw = newton_backward_host(
        c, asm.dt_s, h, qc, &hour.n2, &asm.depth, &asm.edge, &asm.newton, &nw_bar
    );
    // B2: the storage top width carries ḡD_s·L/Δ (the storage path)
    let g_ds = add2(&mat.d_s, &ctl.d_s);
    let mut store_bar = ctl.store.clone();
    store_bar.top_width = (0..n).map(|i| g_ds[i] * c.length[i] / asm.dt_s).collect();
    let sec_e = section_backward_host(c, &hour.p, &hour.q_eps, &asm.depth.d_edge, &asm.edge, &nw.edge);
    let sec_s = section_backward_host(c, &hour.p, &hour.q_eps, &asm.depth.depth, &asm.store, &store_bar);
    // B1
    let g_depth = add2(&ctl.depth, &sec_s.d);
    let gh_depth = depth_backward_host(c, h, &asm.depth, &g_depth, &sec_e.d, &nw.h_down);
    let g_h: Vec<f32> = (0..n).map(|i| rg.h[i] + ctl.h[i] + nw.h[i] + gh_depth[i]).collect();
    let g_q = add2(&rhs.qc, &nw.qc);
    // hour accumulators (BH1 weights i_delta)
    add_into(&mut acc.n2, &nw.n2);
    add_into(&mut acc.n_eff, &ctl.n_eff);
    for part in [&sec_e.p, &sec_s.p, &ctl.p] {
        add_into(&mut acc.p, part);
    }
    for part in [&sec_e.q_eps, &sec_s.q_eps, &ctl.q_eps] {
        add_into(&mut acc.q_eps, part);
    }
    add_into(&mut acc.qp, &rhs.qp);
    add_into(&mut acc.i_t, &rhs.i_mid);
    let w = 0.5 * (f0 + f1);
    for i in 0..n {
        acc.i_delta[i] += rhs.i_mid[i] * w;
    }
    (g_h, g_q)
}

/// BH0: `(ḡh_S, ḡQ_S, ḡq_next_mc direct)` from the `[2n]` output cotangent.
#[doc(hidden)]
pub fn output_backward_host(c: &StageOpConsts, g_out: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = c.n;
    assert_eq!(g_out.len(), 2 * n, "g_out must have length 2n");
    let g_h = g_out[..n].to_vec();
    let g_q: Vec<f32> = (0..n).map(|i| g_out[n + i] * c.in_stage[i]).collect();
    let g_mc: Vec<f32> = (0..n).map(|i| g_out[n + i] * keep_of(c.in_stage[i])).collect();
    (g_h, g_q, g_mc)
}

/// The gradients of the seven parents.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct HourGrads {
    pub n: Vec<f32>,
    pub q: Vec<f32>,
    pub p: Vec<f32>,
    pub h_t: Vec<f32>,
    pub q_t: Vec<f32>,
    pub q_next_mc: Vec<f32>,
    pub q_prime: Vec<f32>,
}

/// BH2 and BH3.
#[doc(hidden)]
pub fn prologue_backward_host(
    c: &StageOpConsts,
    hour: &HourHost,
    acc: &HourAcc,
    g_h0: &[f32],
    g_q0: &[f32],
    g_qmc_direct: &[f32],
) -> HourGrads {
    let n = c.n;
    // BH2
    let g_i_t: Vec<f32> = (0..n).map(|i| acc.i_t[i] - acc.i_delta[i]).collect();
    let mut g_q_t = g_q0.to_vec();
    let mut g_mc = g_qmc_direct.to_vec();
    for k in 0..c.nnz {
        g_q_t[c.col[k]] += g_i_t[c.row[k]] * c.adj_mc[k];
        g_mc[c.col[k]] += acc.i_delta[c.row[k]] * c.adj_mc[k];
    }
    // BH3
    let g_qp: Vec<f32> = (0..n).map(|i| acc.qp[i] * c.in_stage[i]).collect();
    let g_n: Vec<f32> = (0..n)
        .map(|i| {
            let g_n_eff = acc.n_eff[i] + 2.0 * hour.n_eff[i] * acc.n2[i];
            if c.has_table {
                g_n_eff * keep_of(c.t_mask[i])
            } else {
                g_n_eff
            }
        })
        .collect();
    HourGrads {
        n: g_n,
        q: acc.q_eps.clone(),
        p: acc.p.clone(),
        h_t: g_h0.to_vec(),
        q_t: g_q_t,
        q_next_mc: g_mc,
        q_prime: g_qp,
    }
}

/// The hour forward: `substeps` substeps of assemble, factor, L solve, U solve
/// and readout. Sums the mass over the substeps and keeps the last
/// `Assembled`. Pushes a [`SubstepSaved`] per substep when `save`. No loop
/// over reaches.
#[doc(hidden)]
pub fn stage_hour_forward(
    c: &StageOpConsts,
    pattern: &CsrPattern,
    hour: &HourHost,
    h_t: &[f32],
    q_t: &[f32],
    save: bool,
) -> HourForward {
    let subs = c.substeps.max(1) as usize;
    let mut h = h_t.to_vec();
    let mut qc = q_t.to_vec();
    let mut mass = [0.0f32; 5];
    let mut saved = Vec::new();
    let mut last = Assembled::default();
    for s in 0..subs {
        let asm = stage_assemble_host(c, hour, s, &h, &qc);
        let factors = tree_lu_factor(pattern, &asm.a_values);
        let y = forward_sub_lower(pattern, &factors.l_values, &asm.r);
        let dh = back_sub_upper_transposed(pattern, &factors.u_values, &y);
        let ro = stage_readout_host(c, hour, &h, &qc, &asm, &dh);
        for (m, v) in mass.iter_mut().zip(ro.mass) {
            *m += v;
        }
        if save {
            saved.push(SubstepSaved { h: h.clone(), qc: qc.clone(), dh, factors });
        }
        h = ro.h_new;
        qc = ro.q_new;
        last = asm;
    }
    let q_out = stage_epilogue_host(c, hour, &qc);
    HourForward { h_end: h, q_out, last, mass, subs: saved }
}

/// H5: `q_next_i = in_i == 1 ? Q_i : q_next_mc_i`.
#[doc(hidden)]
pub fn stage_epilogue_host(c: &StageOpConsts, hour: &HourHost, qc: &[f32]) -> Vec<f32> {
    (0..c.n).map(|i| if c.in_stage[i] == 1.0 { qc[i] } else { hour.q_next_mc[i] }).collect()
}

/// The hour backward: BH0, then for `s = S-1..0` the rematerialized assembly,
/// the readout adjoint, the solve adjoint (`CsrSolveOp` rule on both halves, as
/// in `TreeSolveOp::backward`) and the assembly adjoint; then BH2 and BH3.
/// Loops over substeps only.
#[doc(hidden)]
pub fn stage_hour_backward(
    c: &StageOpConsts,
    pattern: &CsrPattern,
    hour: &HourHost,
    subs: &[SubstepSaved],
    g_out: &[f32],
) -> HourGrads {
    let (mut g_h, mut g_q, g_mc) = output_backward_host(c, g_out);
    let mut acc = HourAcc::zeros(c.n);
    for s in (0..subs.len()).rev() {
        let sv = &subs[s];
        let asm = stage_assemble_host(c, hour, s, &sv.h, &sv.qc);
        let rg = readout_backward_host(c, &sv.h, &asm, &sv.dh, &g_h, &g_q);
        // g = A^{-T} ḡdh: Uᵀ z = ḡdh, then Lᵀ g = z, on the saved factors.
        let z = forward_sub_lower(pattern, &sv.factors.u_values, &rg.dh);
        let g_r = back_sub_upper_transposed(pattern, &sv.factors.l_values, &z);
        let g_a = tree_lu_grad_a_host(pattern, &g_r, &sv.dh);
        let (nh, nq) = assemble_backward_host(
            c, hour, s, &sv.h, &sv.qc, &asm, &rg, &g_r, &g_a, &mut acc,
        );
        g_h = nh;
        g_q = nq;
    }
    prologue_backward_host(c, hour, &acc, &g_h, &g_q, &g_mc)
}

// ------------------------------------------------------------------ the op

/// What the backward needs from the forward. The CPU/CUDA choice is made once
/// per op, in [`stage_timestep`]; the variant records which arm ran.
#[derive(Clone, Debug)]
pub(crate) enum HourSaved {
    Host { hour: Arc<HourHost>, subs: Arc<Vec<SubstepSaved>> },
}

/// Saved state of one routing hour. Everything is behind an `Arc`, so a clone
/// onto the tape is a refcount bump.
#[derive(Clone, Debug)]
pub(crate) struct StageTimestepState {
    consts: Arc<StageOpConsts>,
    pattern: Arc<CsrPattern>,
    saved: HourSaved,
}

#[derive(Debug)]
pub(crate) struct StageTimestepOp;

impl<B: Backend + 'static> Backward<B, 7> for StageTimestepOp
where
    B::FloatTensorPrimitive: 'static,
{
    type State = StageTimestepState;

    fn backward(self, ops: Ops<Self::State, 7>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let StageTimestepState { consts, pattern, saved } = ops.state;
        let HourSaved::Host { hour, subs } = saved;
        let [p_n, p_q, p_p, p_ht, p_qt, p_mc, p_qp] = ops.parents;
        let grad_out = grads.consume::<B>(&ops.node);
        let device = B::float_device(&grad_out);
        let g_out = primitive_to_vec::<B>(grad_out);
        let g = stage_hour_backward(&consts, &pattern, &hour, &subs, &g_out);
        let n = consts.n;
        macro_rules! reg {
            ($p:expr, $v:expr) => {
                if let Some(p) = $p {
                    grads.register::<B>(p.id, B::float_from_data(TensorData::new($v, [n]), &device));
                }
            };
        }
        reg!(p_n, g.n);
        reg!(p_q, g.q);
        reg!(p_p, g.p);
        reg!(p_ht, g.h_t);
        reg!(p_qt, g.q_t);
        reg!(p_mc, g.q_next_mc);
        reg!(p_qp, g.q_prime);
    }
}

/// One routing hour as one autodiff node. Slots: `[n, q_spatial, p_spatial,
/// h_t, q_t, q_next_mc, q_prime_t]`. The `[2n]` output is `[h_end, q_next]`;
/// the host side outputs come back in `HourForward` (its `subs` is empty).
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_timestep<I: Backend + 'static>(
    consts: &Arc<StageOpConsts>,
    pattern: &Arc<CsrPattern>,
    n: Tensor<Autodiff<I>, 1>,
    q_spatial: Tensor<Autodiff<I>, 1>,
    p_spatial: Tensor<Autodiff<I>, 1>,
    h_t: Tensor<Autodiff<I>, 1>,
    q_t: Tensor<Autodiff<I>, 1>,
    q_next_mc: Tensor<Autodiff<I>, 1>,
    q_prime_t: Tensor<Autodiff<I>, 1>,
    b0: f32,
    b1: f32,
    use_cuda: bool,
) -> (Tensor<Autodiff<I>, 1>, HourForward)
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    if effective_use_cuda::<I>(use_cuda) {
        unimplemented!("{}", CUDA_TODO)
    }
    let float = |t: Tensor<Autodiff<I>, 1>| match t.into_primitive() {
        TensorPrimitive::Float(a) => a,
        TensorPrimitive::QFloat(_) => panic!("expected float tensor"),
    };
    let (a_n, a_q, a_p, a_h, a_qt, a_mc, a_qp) = (
        float(n),
        float(q_spatial),
        float(p_spatial),
        float(h_t),
        float(q_t),
        float(q_next_mc),
        float(q_prime_t),
    );
    let device = I::float_device(&a_n.primitive);
    // `prepare` before the forward: the forward saves state only when tracked.
    let kind = StageTimestepOp
        .prepare::<NoCheckpointing>([
            a_n.node.clone(),
            a_q.node.clone(),
            a_p.node.clone(),
            a_h.node.clone(),
            a_qt.node.clone(),
            a_mc.node.clone(),
            a_qp.node.clone(),
        ])
        .compute_bound()
        .stateful();
    let save = matches!(kind, OpsKind::Tracked(_));
    let (n_h, q_h, p_h) = (
        primitive_to_vec::<I>(a_n.primitive),
        primitive_to_vec::<I>(a_q.primitive),
        primitive_to_vec::<I>(a_p.primitive),
    );
    let (h_h, qt_h, mc_h, qp_h) = (
        primitive_to_vec::<I>(a_h.primitive),
        primitive_to_vec::<I>(a_qt.primitive),
        primitive_to_vec::<I>(a_mc.primitive),
        primitive_to_vec::<I>(a_qp.primitive),
    );
    let hour = stage_prologue_host(consts, &n_h, &q_h, &p_h, &qt_h, &mc_h, &qp_h, b0, b1);
    let mut fwd = stage_hour_forward(consts, pattern, &hour, &h_h, &qt_h, save);
    let nn = consts.n;
    let out = I::float_from_data(
        TensorData::new([fwd.h_end.as_slice(), fwd.q_out.as_slice()].concat(), [2 * nn]),
        &device,
    );
    let result = match kind {
        OpsKind::Tracked(prep) => {
            let state = StageTimestepState {
                consts: consts.clone(),
                pattern: pattern.clone(),
                saved: HourSaved::Host {
                    hour: Arc::new(hour),
                    subs: Arc::new(std::mem::take(&mut fwd.subs)),
                },
            };
            prep.finish(state, out)
        }
        OpsKind::UnTracked(prep) => prep.finish(out),
    };
    (Tensor::from_primitive(TensorPrimitive::Float(result)), fwd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_backward_splits_the_discharge_cotangent() {
        let c = StageOpConsts { n: 2, in_stage: vec![1.0, 0.0], ..Default::default() };
        let (gh, gq, gmc) = output_backward_host(&c, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!((gh, gq, gmc), (vec![1.0, 2.0], vec![3.0, 0.0], vec![0.0, 4.0]));
    }

    #[test]
    fn epilogue_cpu_branch_mixes_rows() {
        let c = StageOpConsts { n: 2, in_stage: vec![1.0, 0.0], ..Default::default() };
        let hour = HourHost { q_next_mc: vec![7.0, 8.0], ..Default::default() };
        assert_eq!(stage_epilogue_host(&c, &hour, &[1.0, 2.0]), vec![1.0, 8.0]);
    }
}
