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
    // H5
    let q_out: Vec<f32> =
        (0..c.n).map(|i| if c.in_stage[i] == 1.0 { qc[i] } else { hour.q_next_mc[i] }).collect();
    HourForward { h_end: h, q_out, last, mass, subs: saved }
}
