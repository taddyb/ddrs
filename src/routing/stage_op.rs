//! Host forward of the fused stage timestep (Task 16a).
//!
//! The definition is `StageBlock::step_reference` (`src/routing/stage.rs`).
//! Every line here keeps that step's op order and association, so the forward
//! can be bitwise. Block names (H0 to H5, 1a to 9) follow the Task 16 overview.
//!
//! The CPU branch of each dispatch fn is a plain host loop over reaches or
//! slots. The CUDA branch is `unimplemented!` (Phase G). The orchestration fn
//! [`stage_hour_forward`] loops over substeps only.

use crate::sparse::tree_lu::{
    dispatch_factor, dispatch_l_solve, dispatch_u_solve, TreeFactors, CUDA_TODO,
};
use crate::sparse::CsrPattern;

// Scalars the reference passes as f64 literals; Burn casts them to f32.
const F43: f32 = (4.0f64 / 3.0) as f32;
const F23: f32 = (2.0f64 / 3.0) as f32;
const F53: f32 = (5.0f64 / 3.0) as f32;

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
// Task 16b: the adjoints of the elementwise blocks (B1 to B4 of the Task 16
// overview), one loop over reaches each. These are the CPU branches; the
// dispatch fns that wrap them with a `use_cuda` arm come in 16c. Masks use the
// forward line's comparison (overview 16.5). An empty cotangent Vec is zeros.

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

// -------------------------------------------------------------- dispatch

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_stage_prologue(
    c: &StageOpConsts,
    n: &[f32],
    q: &[f32],
    p: &[f32],
    q_t: &[f32],
    q_next_mc: &[f32],
    q_prime: &[f32],
    b0: f32,
    b1: f32,
    use_cuda: bool,
) -> HourHost {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    stage_prologue_host(c, n, q, p, q_t, q_next_mc, q_prime, b0, b1)
}

pub(crate) fn dispatch_stage_assemble(
    c: &StageOpConsts,
    hour: &HourHost,
    s: usize,
    h: &[f32],
    qc: &[f32],
    use_cuda: bool,
) -> Assembled {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    stage_assemble_host(c, hour, s, h, qc)
}

pub(crate) fn dispatch_stage_readout(
    c: &StageOpConsts,
    hour: &HourHost,
    h: &[f32],
    qc: &[f32],
    asm: &Assembled,
    dh: &[f32],
    use_cuda: bool,
) -> Readout {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    stage_readout_host(c, hour, h, qc, asm, dh)
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
    use_cuda: bool,
) -> HourForward {
    let subs = c.substeps.max(1) as usize;
    let mut h = h_t.to_vec();
    let mut qc = q_t.to_vec();
    let mut mass = [0.0f32; 5];
    let mut saved = Vec::new();
    let mut last = Assembled::default();
    for s in 0..subs {
        let asm = dispatch_stage_assemble(c, hour, s, &h, &qc, use_cuda);
        let factors = dispatch_factor(pattern, &asm.a_values, use_cuda);
        let y = dispatch_l_solve(pattern, &factors.l_values, &asm.r, use_cuda);
        let dh = dispatch_u_solve(pattern, &factors.u_values, &y, use_cuda);
        let ro = dispatch_stage_readout(c, hour, &h, &qc, &asm, &dh, use_cuda);
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
    let q_out = dispatch_stage_epilogue(c, hour, &qc, use_cuda);
    HourForward { h_end: h, q_out, last, mass, subs: saved }
}

/// H5: `q_next_i = in_i == 1 ? Q_i : q_next_mc_i`.
#[doc(hidden)]
pub fn stage_epilogue_host(c: &StageOpConsts, hour: &HourHost, qc: &[f32]) -> Vec<f32> {
    (0..c.n).map(|i| if c.in_stage[i] == 1.0 { qc[i] } else { hour.q_next_mc[i] }).collect()
}

pub(crate) fn dispatch_stage_epilogue(
    c: &StageOpConsts,
    hour: &HourHost,
    qc: &[f32],
    use_cuda: bool,
) -> Vec<f32> {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    stage_epilogue_host(c, hour, qc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "stage_routing CUDA path: Phase G")]
    fn epilogue_cuda_branch_is_unimplemented() {
        let c = StageOpConsts { n: 1, in_stage: vec![1.0], ..Default::default() };
        let hour = HourHost { q_next_mc: vec![0.0], ..Default::default() };
        let _ = dispatch_stage_epilogue(&c, &hour, &[1.0], true);
    }

    #[test]
    fn epilogue_cpu_branch_mixes_rows() {
        let c = StageOpConsts { n: 2, in_stage: vec![1.0, 0.0], ..Default::default() };
        let hour = HourHost { q_next_mc: vec![7.0, 8.0], ..Default::default() };
        assert_eq!(dispatch_stage_epilogue(&c, &hour, &[1.0, 2.0], false), vec![1.0, 8.0]);
    }
}
