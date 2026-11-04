//! Forward kernels of the stage block: one cubecl kernel per host block fn of
//! `src/routing/stage_op.rs`, each body the host loop body line for line.
//!
//! Rules (see the tie table in the `stage_op.rs` module doc):
//! - the association of every product is the host's;
//! - `clamp_min(x, lb)` is `select(x < lb, lb, x)`, never `max`;
//! - the scalars `theta`, `g`, `dt_s`, `depth_lb`, `bw_lb`, `F43`, `F23`, `F53`
//!   and `1e-6` are f32 arguments read from the host, not retyped literals;
//! - one thread per reach; the row sums are gathers over the CSR rows, in
//!   slot order, the host's loop order.

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim};

use burn::tensor::ops::FloatTensorOps;
use burn::tensor::{DType, Shape};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;

use super::{up, ControlDev, DepthDev, HourDev, NewtonDev, SectionDev, StageOpConstsDev};
use crate::routing::stage_op::{
    Assembled, ControlOut, DepthOut, HourHost, NewtonOut, Readout, SectionHost, SubstepSaved, F23, F43, F53,
};
use crate::sparse::cube_buf::{cube_to_host, empty_f32, from_cube, grid, into_cube, upload_f32, zeros_f32, Cube, Gpu};
use crate::sparse::tree_lu_cuda::{tree_factor_dev, tree_solve_dev, TreeFactorsDev};
use crate::sparse::CsrPattern;

/// The `q + 1e-6` of the width exponent (`stage_prologue_host`).
const Q_EPS: f32 = 1e-6f32;

macro_rules! a {
    ($x:expr) => {
        $x.clone().into_tensor_arg()
    };
}

// ------------------------------------------------------------ cube helpers

/// Burn's `clamp_min`: `x < lb ? lb : x`, so a NaN passes through.
#[cube]
fn clamp_min_f(x: f32, lb: f32) -> f32 {
    select(x < lb, lb, x)
}

/// 1.0 when `b`, else 0.0.
#[cube]
fn ind(b: bool) -> f32 {
    select(b, 1.0f32, 0.0f32)
}

/// `(-m) + 1`.
#[cube]
fn keep_of(m: f32) -> f32 {
    (-m) + 1.0f32
}

// ---------------------------------------------------------------- prologue

/// H0 to H3. Mirrors `stage_prologue_host`. The row sums `i_t`, `i_n` gather
/// over the CSR row in slot order, as the host `spmv` does.
#[cube(launch)]
fn prologue_kernel(
    n_man: &Tensor<f32>,
    q: &Tensor<f32>,
    p: &Tensor<f32>,
    q_t: &Tensor<f32>,
    q_next_mc: &Tensor<f32>,
    q_prime: &Tensor<f32>,
    crow: &Tensor<i32>,
    col: &Tensor<i32>,
    adj_mc: &Tensor<f32>,
    in_stage: &Tensor<f32>,
    t_mask: &Tensor<f32>,
    t_n: &Tensor<f32>,
    o_p: &mut Tensor<f32>,
    o_q_eps: &mut Tensor<f32>,
    o_n_eff: &mut Tensor<f32>,
    o_n2: &mut Tensor<f32>,
    o_qp_stage: &mut Tensor<f32>,
    o_i_t: &mut Tensor<f32>,
    o_i_delta: &mut Tensor<f32>,
    o_q_next_mc: &mut Tensor<f32>,
    n: u32,
    q_eps_add: f32,
    #[comptime] has_table: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let s0 = usize::cast_from(crow[i]);
        let s1 = usize::cast_from(crow[i + 1]);
        let mut i_t = 0.0f32;
        let mut i_n = 0.0f32;
        for k in s0..s1 {
            let c = usize::cast_from(col[k]);
            i_t = fma(q_t[c], adj_mc[k], i_t);
            i_n = fma(q_next_mc[c], adj_mc[k], i_n);
        }
        let q_eps = q[i] + q_eps_add;
        let mut n_eff = n_man[i];
        if has_table {
            n_eff = fma(n_man[i], keep_of(t_mask[i]), t_n[i] * t_mask[i]);
        }
        o_p[i] = p[i];
        o_q_eps[i] = q_eps;
        o_n_eff[i] = n_eff;
        o_n2[i] = n_eff * n_eff;
        o_qp_stage[i] = q_prime[i] * in_stage[i];
        o_i_t[i] = i_t;
        o_i_delta[i] = i_n - i_t;
        o_q_next_mc[i] = q_next_mc[i];
    }
}

/// H0 to H3 on the device. Mirrors `stage_prologue_host`, same argument order.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn stage_prologue_dev(
    c: &StageOpConstsDev,
    n: &Cube,
    q: &Cube,
    p: &Cube,
    q_t: &Cube,
    q_next_mc: &Cube,
    q_prime: &Cube,
    b0: f32,
    b1: f32,
) -> HourDev {
    let nn = c.host.n;
    let o = || empty_f32(q, nn);
    let out = HourDev {
        p: o(),
        q_eps: o(),
        n_eff: o(),
        n2: o(),
        qp_stage: o(),
        i_t: o(),
        i_delta: o(),
        q_next_mc: o(),
        b0,
        b1,
    };
    let (count, dim) = grid(nn);
    prologue_kernel::launch::<CudaRuntime>(
        &q.client,
        count,
        dim,
        a!(n),
        a!(q),
        a!(p),
        a!(q_t),
        a!(q_next_mc),
        a!(q_prime),
        a!(c.tree.crow),
        a!(c.tree.col),
        a!(c.adj_mc),
        a!(c.in_stage),
        a!(c.t_mask),
        a!(c.t_n),
        a!(out.p),
        a!(out.q_eps),
        a!(out.n_eff),
        a!(out.n2),
        a!(out.qp_stage),
        a!(out.i_t),
        a!(out.i_delta),
        a!(out.q_next_mc),
        nn as u32,
        Q_EPS,
        c.host.has_table,
    );
    out
}

/// H4: `(i_t * 2 + i_delta * (f0 + f1)) * 0.5`. Mirrors the H4 line of
/// `stage_assemble_host`.
#[cube(launch)]
fn i_mid_kernel(
    i_t: &Tensor<f32>,
    i_delta: &Tensor<f32>,
    o_i_mid: &mut Tensor<f32>,
    n: u32,
    f0: f32,
    f1: f32,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        o_i_mid[i] = fma(i_delta[i], f0 + f1, i_t[i] * 2.0f32) * 0.5f32;
    }
}

/// H4 of substep `s` on the device.
#[doc(hidden)]
pub fn i_mid_dev(c: &StageOpConstsDev, hour: &HourDev, s: usize) -> Cube {
    let nn = c.host.n;
    let subs = c.host.substeps.max(1) as usize;
    let (f0, f1) = (s as f32 / subs as f32, (s + 1) as f32 / subs as f32);
    let out = empty_f32(&hour.i_t, nn);
    let (count, dim) = grid(nn);
    i_mid_kernel::launch::<CudaRuntime>(
        &hour.i_t.client,
        count,
        dim,
        a!(hour.i_t),
        a!(hour.i_delta),
        a!(out),
        nn as u32,
        f0,
        f1,
    );
    out
}

// ------------------------------------------------------------------- depth

/// 1a to 1e. Mirrors `depth_host`. `down` is the reach itself at an outlet.
#[cube(launch)]
fn depth_kernel(
    h: &Tensor<f32>,
    down: &Tensor<i32>,
    outlet: &Tensor<f32>,
    z_mid: &Tensor<f32>,
    z_down: &Tensor<f32>,
    o_x: &mut Tensor<f32>,
    o_depth: &mut Tensor<f32>,
    o_h_down: &mut Tensor<f32>,
    o_der: &mut Tensor<f32>,
    o_d_edge: &mut Tensor<f32>,
    o_s_up: &mut Tensor<f32>,
    o_s_dn: &mut Tensor<f32>,
    n: u32,
    lb: f32,
    hb0: f32,
    #[comptime] prescribed: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let x = h[i] - z_mid[i];
        o_x[i] = x;
        o_depth[i] = clamp_min_f(x, lb);
        let mut hd = h[usize::cast_from(down[i])];
        if prescribed {
            hd = select(outlet[i] == 1.0f32, hb0, hd);
        }
        o_h_down[i] = hd;
        let mh = select(h[i] < hd, hd, h[i]);
        let mz = select(z_mid[i] < z_down[i], z_down[i], z_mid[i]);
        let der = mh - mz;
        o_der[i] = der;
        o_d_edge[i] = clamp_min_f(der, lb);
        let above = ind(der > lb);
        let up = ind((h[i] - hd) >= 0.0f32);
        o_s_up[i] = up * above;
        o_s_dn[i] = keep_of(up) * above;
    }
}

/// 1a to 1e on the device. Mirrors `depth_host`.
#[doc(hidden)]
pub fn depth_dev(c: &StageOpConstsDev, h: &Cube, hb0: f32) -> DepthDev {
    let nn = c.host.n;
    let o = || empty_f32(h, nn);
    let out = DepthDev { x: o(), depth: o(), h_down: o(), der: o(), d_edge: o(), s_up: o(), s_dn: o() };
    let (count, dim) = grid(nn);
    depth_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(h),
        a!(c.down),
        a!(c.outlet),
        a!(c.z_mid),
        a!(c.z_down),
        a!(out.x),
        a!(out.depth),
        a!(out.h_down),
        a!(out.der),
        a!(out.d_edge),
        a!(out.s_up),
        a!(out.s_dn),
        nn as u32,
        c.host.depth_lb,
        hb0,
        c.host.prescribed,
    );
    out
}

// ----------------------------------------------------------------- section

/// 2a to 2g and nothing else. Mirrors `section_host`. This is the one forward
/// section seam of the CUDA path.
#[cube(launch)]
fn section_kernel(
    p: &Tensor<f32>,
    q_eps: &Tensor<f32>,
    d: &Tensor<f32>,
    t_mask: &Tensor<f32>,
    t_tw: &Tensor<f32>,
    o_pw: &mut Tensor<f32>,
    o_tw: &mut Tensor<f32>,
    o_ss_raw: &mut Tensor<f32>,
    o_ss: &mut Tensor<f32>,
    o_bw_raw: &mut Tensor<f32>,
    o_bw: &mut Tensor<f32>,
    o_area: &mut Tensor<f32>,
    o_root: &mut Tensor<f32>,
    o_perim: &mut Tensor<f32>,
    o_radius: &mut Tensor<f32>,
    o_tw_d: &mut Tensor<f32>,
    o_ss_d: &mut Tensor<f32>,
    o_bw_d: &mut Tensor<f32>,
    o_a_d: &mut Tensor<f32>,
    o_p_d: &mut Tensor<f32>,
    o_out_tw: &mut Tensor<f32>,
    o_out_area: &mut Tensor<f32>,
    o_out_perim: &mut Tensor<f32>,
    o_out_radius: &mut Tensor<f32>,
    o_out_a_d: &mut Tensor<f32>,
    o_out_p_d: &mut Tensor<f32>,
    n: u32,
    bwlb: f32,
    #[comptime] has_table: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let e = q_eps[i];
        let dd = d[i];
        // 2a
        let pw = f32::powf(dd, e);
        let tw = p[i] * pw;
        let ssr = (tw * e) / (dd * 2.0f32);
        // clamp(ssr, 0.5, 50): clamp_min(clamp_max(x, hi), lo)
        let ss = clamp_min_f(select(ssr > 50.0f32, 50.0f32, ssr), 0.5f32);
        // 2b
        let bwr = tw - (ss * dd) * 2.0f32;
        let bw = clamp_min_f(bwr, bwlb);
        // 2c
        let area = ((tw + bw) * dd) / 2.0f32;
        let rt = f32::sqrt(fma(ss, ss, 1.0f32));
        let perim = bw + (dd * rt) * 2.0f32;
        let radius = area / perim;
        // 2d
        let twd = (e * tw) / dd;
        let sa = ind(ssr > 0.5f32) * ind(ssr < 50.0f32);
        let ssd = ((ssr / dd) * fma(twd / tw, dd, -1.0f32)) * sa;
        // 2e
        let ba = ind(bwr > bwlb);
        let bwd = ((twd - ss * 2.0f32) - (dd * ssd) * 2.0f32) * ba;
        // 2f
        let ad = (tw + bw) / 2.0f32 + (dd * (twd + bwd)) / 2.0f32;
        let pd = (bwd + rt * 2.0f32) + (((dd * ss) * ssd) * 2.0f32) / rt;
        o_pw[i] = pw;
        o_tw[i] = tw;
        o_ss_raw[i] = ssr;
        o_ss[i] = ss;
        o_bw_raw[i] = bwr;
        o_bw[i] = bw;
        o_area[i] = area;
        o_root[i] = rt;
        o_perim[i] = perim;
        o_radius[i] = radius;
        o_tw_d[i] = twd;
        o_ss_d[i] = ssd;
        o_bw_d[i] = bwd;
        o_a_d[i] = ad;
        o_p_d[i] = pd;
        // 2g
        if has_table {
            let m = t_mask[i];
            let ttw = t_tw[i];
            let keep = keep_of(m);
            let area_r = ttw * dd;
            let perim_r = ttw + dd * 2.0f32;
            let radius_r = area_r / perim_r;
            o_out_tw[i] = fma(tw, keep, ttw * m);
            o_out_area[i] = fma(area, keep, area_r * m);
            o_out_perim[i] = fma(perim, keep, perim_r * m);
            o_out_radius[i] = fma(radius, keep, radius_r * m);
            o_out_a_d[i] = fma(ad, keep, ttw * m);
            o_out_p_d[i] = fma(pd, keep, (1.0f32 * 2.0f32) * m);
        } else {
            o_out_tw[i] = tw;
            o_out_area[i] = area;
            o_out_perim[i] = perim;
            o_out_radius[i] = radius;
            o_out_a_d[i] = ad;
            o_out_p_d[i] = pd;
        }
    }
}

/// 2a to 2g at the depths `d` on the device. THE forward section seam of the
/// CUDA path. Mirrors `section_host`.
#[doc(hidden)]
pub fn section_dev(c: &StageOpConstsDev, p: &Cube, q_eps: &Cube, d: &Cube) -> SectionDev {
    let nn = c.host.n;
    let o = || empty_f32(d, nn);
    let out = SectionDev {
        pw: o(),
        tw: o(),
        ss_raw: o(),
        ss: o(),
        bw_raw: o(),
        bw: o(),
        area: o(),
        root: o(),
        perim: o(),
        radius: o(),
        tw_d: o(),
        ss_d: o(),
        bw_d: o(),
        a_d: o(),
        p_d: o(),
        out_tw: o(),
        out_area: o(),
        out_perim: o(),
        out_radius: o(),
        out_a_d: o(),
        out_p_d: o(),
    };
    let (count, dim) = grid(nn);
    section_kernel::launch::<CudaRuntime>(
        &d.client,
        count,
        dim,
        a!(p),
        a!(q_eps),
        a!(d),
        a!(c.t_mask),
        a!(c.t_tw),
        a!(out.pw),
        a!(out.tw),
        a!(out.ss_raw),
        a!(out.ss),
        a!(out.bw_raw),
        a!(out.bw),
        a!(out.area),
        a!(out.root),
        a!(out.perim),
        a!(out.radius),
        a!(out.tw_d),
        a!(out.ss_d),
        a!(out.bw_d),
        a!(out.a_d),
        a!(out.p_d),
        a!(out.out_tw),
        a!(out.out_area),
        a!(out.out_perim),
        a!(out.out_radius),
        a!(out.out_a_d),
        a!(out.out_p_d),
        nn as u32,
        c.host.bw_lb,
        c.host.has_table,
    );
    out
}

// ------------------------------------------------------------ storage rate

/// 2h: `(out_tw * length) / dt_s`. Mirrors the `d_s` line of
/// `stage_assemble_host`.
#[cube(launch)]
fn storage_rate_kernel(out_tw: &Tensor<f32>, length: &Tensor<f32>, o_d_s: &mut Tensor<f32>, n: u32, dt_s: f32) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        o_d_s[i] = (out_tw[i] * length[i]) / dt_s;
    }
}

/// 2h on the device: the storage rate `d_s`.
#[doc(hidden)]
pub fn storage_rate_dev(c: &StageOpConstsDev, store: &SectionDev, dt_s: f32) -> Cube {
    let nn = c.host.n;
    let out = empty_f32(&store.out_tw, nn);
    let (count, dim) = grid(nn);
    storage_rate_kernel::launch::<CudaRuntime>(
        &store.out_tw.client,
        count,
        dim,
        a!(store.out_tw),
        a!(c.length),
        a!(out),
        nn as u32,
        dt_s,
    );
    out
}

// ------------------------------------------------------------------ newton

/// 3a to 3f. Mirrors `newton_host`. The section outputs are read from the
/// edge section; no section math is repeated here.
#[cube(launch)]
fn newton_kernel(
    h: &Tensor<f32>,
    qc: &Tensor<f32>,
    n2: &Tensor<f32>,
    h_down: &Tensor<f32>,
    s_up: &Tensor<f32>,
    s_dn: &Tensor<f32>,
    e_area: &Tensor<f32>,
    e_radius: &Tensor<f32>,
    e_a_d: &Tensor<f32>,
    e_p_d: &Tensor<f32>,
    e_perim: &Tensor<f32>,
    friction: &Tensor<f32>,
    edge_length: &Tensor<f32>,
    o_phi: &mut Tensor<f32>,
    o_dt_phi: &mut Tensor<f32>,
    o_den: &mut Tensor<f32>,
    o_alpha: &mut Tensor<f32>,
    o_beta: &mut Tensor<f32>,
    o_phi_d: &mut Tensor<f32>,
    o_dh_edge: &mut Tensor<f32>,
    o_k1: &mut Tensor<f32>,
    o_kappa: &mut Tensor<f32>,
    o_q_star: &mut Tensor<f32>,
    o_ra: &mut Tensor<f32>,
    o_rb: &mut Tensor<f32>,
    o_a_up: &mut Tensor<f32>,
    o_b_down: &mut Tensor<f32>,
    n: u32,
    dt_s: f32,
    th: f32,
    g: f32,
    f43: f32,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let gdt = g * dt_s;
        let a_e = e_area[i];
        let r_e = e_radius[i];
        let ad_e = e_a_d[i];
        let pd_e = e_p_d[i];
        let p_e = e_perim[i];
        let f = friction[i];
        let q = qc[i];
        // 3a: |Q| has no sign at 0
        let phi = ((n2[i] * f32::abs(q)) * g) / (a_e * f32::powf(r_e, f43));
        let tau = phi * dt_s;
        let den = tau * 2.0f32 + 1.0f32;
        // 3b
        let alpha = ((tau + 1.0f32) / den) * f;
        let beta = ((a_e * gdt) / (edge_length[i] * den)) * f;
        // 3c
        let phi_d = fma(ad_e / a_e - pd_e / p_e, f43, ad_e / a_e);
        // 3d
        let dhe = h[i] - h_down[i];
        let k1 = fma(tau * q, phi_d, ((ad_e * dhe) * gdt) / edge_length[i]);
        let kappa = (k1 / den) * f;
        // 3e
        let q_star = fma(alpha, q, beta * dhe);
        // 3f
        let ra = fma(beta, th, kappa * s_up[i]);
        let a = clamp_min_f(ra, 0.0f32) * f;
        let rb = fma(beta, th, -(kappa * s_dn[i]));
        let b = clamp_min_f(rb, 0.0f32) * f;
        o_phi[i] = phi;
        o_dt_phi[i] = tau;
        o_den[i] = den;
        o_alpha[i] = alpha;
        o_beta[i] = beta;
        o_phi_d[i] = phi_d;
        o_dh_edge[i] = dhe;
        o_k1[i] = k1;
        o_kappa[i] = kappa;
        o_q_star[i] = q_star;
        o_ra[i] = ra;
        o_rb[i] = rb;
        o_a_up[i] = a;
        o_b_down[i] = b;
    }
}

/// 3a to 3f on the device. Mirrors `newton_host`, same argument order.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn newton_dev(
    c: &StageOpConstsDev,
    dt_s: f32,
    h: &Cube,
    qc: &Cube,
    n2: &Cube,
    dep: &DepthDev,
    edge: &SectionDev,
) -> NewtonDev {
    let nn = c.host.n;
    let o = || empty_f32(h, nn);
    let out = NewtonDev {
        phi: o(),
        dt_phi: o(),
        den: o(),
        alpha: o(),
        beta: o(),
        phi_d: o(),
        dh_edge: o(),
        k1: o(),
        kappa: o(),
        q_star: o(),
        ra: o(),
        rb: o(),
        a_up: o(),
        b_down: o(),
    };
    let (count, dim) = grid(nn);
    newton_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(h),
        a!(qc),
        a!(n2),
        a!(dep.h_down),
        a!(dep.s_up),
        a!(dep.s_dn),
        a!(edge.out_area),
        a!(edge.out_radius),
        a!(edge.out_a_d),
        a!(edge.out_p_d),
        a!(edge.out_perim),
        a!(c.friction),
        a!(c.edge_length),
        a!(out.phi),
        a!(out.dt_phi),
        a!(out.den),
        a!(out.alpha),
        a!(out.beta),
        a!(out.phi_d),
        a!(out.dh_edge),
        a!(out.k1),
        a!(out.kappa),
        a!(out.q_star),
        a!(out.ra),
        a!(out.rb),
        a!(out.a_up),
        a!(out.b_down),
        nn as u32,
        dt_s,
        c.host.theta,
        c.host.g,
        F43,
    );
    out
}

// ----------------------------------------------------------------- control

/// 4a to 4d. Mirrors `control_host`. `ex` and `act` are written only with a
/// table; without one they are length-1 dummies.
#[cube(launch)]
fn control_kernel(
    h: &Tensor<f32>,
    p: &Tensor<f32>,
    q_eps: &Tensor<f32>,
    n_eff: &Tensor<f32>,
    depth: &Tensor<f32>,
    d_s: &Tensor<f32>,
    s_area: &Tensor<f32>,
    s_radius: &Tensor<f32>,
    s_a_d: &Tensor<f32>,
    s_p_d: &Tensor<f32>,
    s_perim: &Tensor<f32>,
    length: &Tensor<f32>,
    dam_t: &Tensor<f32>,
    t_mask: &Tensor<f32>,
    t_tw: &Tensor<f32>,
    t_pool: &Tensor<f32>,
    s0_sqrt: &Tensor<f32>,
    dam: &Tensor<f32>,
    nd: &Tensor<f32>,
    o_rd_head: &mut Tensor<f32>,
    o_rpd_head: &mut Tensor<f32>,
    o_ex: &mut Tensor<f32>,
    o_act: &mut Tensor<f32>,
    o_rd: &mut Tensor<f32>,
    o_rpd: &mut Tensor<f32>,
    o_kk: &mut Tensor<f32>,
    o_w: &mut Tensor<f32>,
    o_dk_dd: &mut Tensor<f32>,
    o_r_t: &mut Tensor<f32>,
    o_rpr: &mut Tensor<f32>,
    o_r_prime: &mut Tensor<f32>,
    n: u32,
    dt_s: f32,
    f23: f32,
    f53: f32,
    #[comptime] has_table: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        // 4a
        let e1 = q_eps[i] + 1.0f32;
        let rd_head = (((length[i] * p[i]) * f32::powf(depth[i], e1)) / e1) / dam_t[i];
        let rpd_head = (d_s[i] * dt_s) / dam_t[i];
        let mut rd = rd_head;
        let mut rpd = rpd_head;
        // 4b
        if has_table {
            let m = t_mask[i];
            let keep = keep_of(m);
            let ex = h[i] - t_pool[i];
            let act = ind(ex > 0.0f32);
            let rtb = ((t_tw[i] * length[i]) * clamp_min_f(ex, 0.0f32)) / dam_t[i];
            let rptb = rpd_head * act;
            rd = fma(rd_head, keep, rtb * m);
            rpd = fma(rpd_head, keep, rptb * m);
            o_ex[i] = ex;
            o_act[i] = act;
        }
        // 4c
        let a_s = s_area[i];
        let r_s = s_radius[i];
        let ad_s = s_a_d[i];
        let pd_s = s_p_d[i];
        let p_s = s_perim[i];
        let kk = (a_s * f32::powf(r_s, f23)) / n_eff[i];
        let rnd = kk * s0_sqrt[i];
        let w = fma(ad_s / a_s, f53, -((pd_s / p_s) * f23));
        let dk = kk * w;
        let rpnd = dk * s0_sqrt[i];
        // 4d
        let r_t = fma(dam[i], rd, nd[i] * rnd);
        let rpr = fma(dam[i], rpd, nd[i] * rpnd);
        o_rd_head[i] = rd_head;
        o_rpd_head[i] = rpd_head;
        o_rd[i] = rd;
        o_rpd[i] = rpd;
        o_kk[i] = kk;
        o_w[i] = w;
        o_dk_dd[i] = dk;
        o_r_t[i] = r_t;
        o_rpr[i] = rpr;
        o_r_prime[i] = clamp_min_f(rpr, 0.0f32);
    }
}

/// 4a to 4d on the device. Mirrors `control_host`, same argument order.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn control_dev(
    c: &StageOpConstsDev,
    dt_s: f32,
    h: &Cube,
    p: &Cube,
    q_eps: &Cube,
    n_eff: &Cube,
    depth: &Cube,
    store: &SectionDev,
    d_s: &Cube,
) -> ControlDev {
    let nn = c.host.n;
    let o = || empty_f32(h, nn);
    // Without a table `ex` and `act` are length-1 dummies the kernel never writes.
    let dummy = || zeros_f32(h, if c.host.has_table { nn } else { 1 });
    let out = ControlDev {
        rd_head: o(),
        rpd_head: o(),
        rd: o(),
        rpd: o(),
        kk: o(),
        w: o(),
        dk_dd: o(),
        r_t: o(),
        rpr: o(),
        r_prime: o(),
        ex: dummy(),
        act: dummy(),
    };
    let (count, dim) = grid(nn);
    control_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(h),
        a!(p),
        a!(q_eps),
        a!(n_eff),
        a!(depth),
        a!(d_s),
        a!(store.out_area),
        a!(store.out_radius),
        a!(store.out_a_d),
        a!(store.out_p_d),
        a!(store.out_perim),
        a!(c.length),
        a!(c.dam_t),
        a!(c.t_mask),
        a!(c.t_tw),
        a!(c.t_pool),
        a!(c.s0_sqrt),
        a!(c.dam),
        a!(c.nd),
        a!(out.rd_head),
        a!(out.rpd_head),
        a!(out.ex),
        a!(out.act),
        a!(out.rd),
        a!(out.rpd),
        a!(out.kk),
        a!(out.w),
        a!(out.dk_dd),
        a!(out.r_t),
        a!(out.rpr),
        a!(out.r_prime),
        nn as u32,
        dt_s,
        F23,
        F53,
        c.host.has_table,
    );
    out
}

// ------------------------------------------------------------------ matrix

/// 5a to 5d. Mirrors `matrix_host`: one thread per slot. The row sum `sib` is
/// a gather over the slot's row in slot order, the host's loop order, recomputed per slot.
#[cube(launch)]
fn matrix_kernel(
    d_s: &Tensor<f32>,
    a_up: &Tensor<f32>,
    b_down: &Tensor<f32>,
    r_prime: &Tensor<f32>,
    row: &Tensor<i32>,
    col: &Tensor<i32>,
    crow: &Tensor<i32>,
    adj_friction: &Tensor<f32>,
    dam_slot: &Tensor<f32>,
    diag_slot: &Tensor<f32>,
    in_stage: &Tensor<f32>,
    w_ctrl: &Tensor<f32>,
    out: &mut Tensor<f32>,
    nnz: u32,
    th: f32,
    thc: f32,
) {
    let k = ABSOLUTE_POS;
    if k < usize::cast_from(nnz) {
        let r = usize::cast_from(row[k]);
        let c = usize::cast_from(col[k]);
        let s0 = usize::cast_from(crow[r]);
        let s1 = usize::cast_from(crow[r + 1]);
        let mut sib = 0.0f32;
        for j in s0..s1 {
            let cj = usize::cast_from(col[j]);
            sib = fma(b_down[cj], adj_friction[j], sib);
        }
        let dst = fma(a_up[r] + sib, th, d_s[r]);
        let diag = fma(dst, in_stage[r], w_ctrl[r] * r_prime[r]) + keep_of(in_stage[r]);
        let ae = a_up[c] * adj_friction[k];
        let lowf_a = -ae;
        let be = b_down[c] * adj_friction[k];
        let upf = (-be) * th;
        let offd_a = r_prime[c] * dam_slot[k];
        let offd = offd_a * (-thc);
        out[k] = fma(diag, diag_slot[k], fma(lowf_a, th, offd));
        out[usize::cast_from(nnz) + k] = upf;
    }
}

/// 5a to 5d on the device. Mirrors `matrix_host`. Returns `2 * nnz` values.
#[doc(hidden)]
pub fn matrix_dev(c: &StageOpConstsDev, d_s: &Cube, a_up: &Cube, b_down: &Cube, r_prime: &Cube) -> Cube {
    let nnz = c.host.nnz;
    let out = empty_f32(d_s, 2 * nnz);
    let (count, dim) = grid(nnz);
    matrix_kernel::launch::<CudaRuntime>(
        &d_s.client,
        count,
        dim,
        a!(d_s),
        a!(a_up),
        a!(b_down),
        a!(r_prime),
        a!(c.tree.row),
        a!(c.tree.col),
        a!(c.tree.crow),
        a!(c.adj_friction),
        a!(c.dam_slot),
        a!(c.diag_slot),
        a!(c.in_stage),
        a!(c.w_ctrl),
        a!(out),
        nnz as u32,
        c.host.theta,
        c.host.theta_c,
    );
    out
}

// --------------------------------------------------------------------- rhs

/// 6a: the edge outflow `eo`. Mirrors the first closure of `rhs_host`.
#[cube(launch)]
fn eo_kernel(
    friction: &Tensor<f32>,
    q_star: &Tensor<f32>,
    qc: &Tensor<f32>,
    w_ctrl: &Tensor<f32>,
    r_t: &Tensor<f32>,
    ctrl: &Tensor<f32>,
    o_eo: &mut Tensor<f32>,
    n: u32,
    th: f32,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let inner = fma(q_star[i], th, qc[i] * (1.0f32 - th));
        let co = fma(w_ctrl[i], r_t[i], (ctrl[i] - w_ctrl[i]) * qc[i]);
        o_eo[i] = fma(friction[i], inner, co);
    }
}

/// 6b, 6c. Mirrors the second closure of `rhs_host`: `ine` is a row gather.
#[cube(launch)]
fn rhs_kernel(
    eo: &Tensor<f32>,
    crow: &Tensor<i32>,
    col: &Tensor<i32>,
    adj_stage: &Tensor<f32>,
    qp: &Tensor<f32>,
    i_mid: &Tensor<f32>,
    outlet: &Tensor<f32>,
    b_down: &Tensor<f32>,
    o_r: &mut Tensor<f32>,
    n: u32,
    th: f32,
    hb0: f32,
    hb1: f32,
    #[comptime] prescribed: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let s0 = usize::cast_from(crow[i]);
        let s1 = usize::cast_from(crow[i + 1]);
        let mut ine = 0.0f32;
        for k in s0..s1 {
            let cc = usize::cast_from(col[k]);
            ine = fma(eo[cc], adj_stage[k], ine);
        }
        let mut r = ((qp[i] + i_mid[i]) + ine) - eo[i];
        if prescribed {
            r = fma(outlet[i] * b_down[i], th * (hb1 - hb0), r);
        }
        o_r[i] = r;
    }
}

/// 6a to 6c on the device. Mirrors `rhs_host`, same argument order.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn rhs_dev(
    c: &StageOpConstsDev,
    qc: &Cube,
    q_star: &Cube,
    r_t: &Cube,
    b_down: &Cube,
    qp: &Cube,
    i_mid: &Cube,
    hb0: f32,
    hb1: f32,
) -> Cube {
    let nn = c.host.n;
    let eo = empty_f32(qc, nn);
    let out = empty_f32(qc, nn);
    let (count, dim) = grid(nn);
    eo_kernel::launch::<CudaRuntime>(
        &qc.client,
        count.clone(),
        dim.clone(),
        a!(c.friction),
        a!(q_star),
        a!(qc),
        a!(c.w_ctrl),
        a!(r_t),
        a!(c.ctrl),
        a!(eo),
        nn as u32,
        c.host.theta,
    );
    rhs_kernel::launch::<CudaRuntime>(
        &qc.client,
        count,
        dim,
        a!(eo),
        a!(c.tree.crow),
        a!(c.tree.col),
        a!(c.adj_stage),
        a!(qp),
        a!(i_mid),
        a!(c.outlet),
        a!(b_down),
        a!(out),
        nn as u32,
        c.host.theta,
        hb0,
        hb1,
        c.host.prescribed,
    );
    out
}

// ---------------------------------------------------------------- assemble

/// H4 and blocks 1 to 6 of one substep on the device (the twin of `Assembled`).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct AssembledDev {
    pub i_mid: Cube,
    pub hb0: f32,
    pub hb1: f32,
    pub dt_s: f32,
    pub depth: DepthDev,
    pub edge: SectionDev,
    pub store: SectionDev,
    pub d_s: Cube,
    pub newton: NewtonDev,
    pub control: ControlDev,
    /// Length `2 * nnz`.
    pub a_values: Cube,
    pub r: Cube,
}

impl AssembledDev {
    /// Copy `a` to the device of `like`. An empty host Vec becomes zeros; an
    /// empty `a_values` becomes `2 * nnz` zeros.
    #[doc(hidden)]
    pub fn upload(a: &Assembled, like: &Cube, nnz: usize) -> Self {
        let n = [a.i_mid.len(), a.d_s.len(), a.r.len()].into_iter().max().unwrap_or(1).max(1);
        Self {
            i_mid: up(like, &a.i_mid, n),
            hb0: a.hb0,
            hb1: a.hb1,
            dt_s: a.dt_s,
            depth: DepthDev::upload(&a.depth, like),
            edge: SectionDev::upload(&a.edge, like),
            store: SectionDev::upload(&a.store, like),
            d_s: up(like, &a.d_s, n),
            newton: NewtonDev::upload(&a.newton, like),
            control: ControlDev::upload(&a.control, like),
            a_values: up(like, &a.a_values, 2 * nnz),
            r: up(like, &a.r, n),
        }
    }
}

/// H4 and blocks 1 to 6 of substep `s` from the start state `(h, qc)`.
/// Mirrors `stage_assemble_host`, block by block.
#[doc(hidden)]
pub fn stage_assemble_dev(c: &StageOpConstsDev, hour: &HourDev, s: usize, h: &Cube, qc: &Cube) -> AssembledDev {
    let subs = c.host.substeps.max(1) as usize;
    let dt_s = c.host.dt / subs as f32;
    let (f0, f1) = (s as f32 / subs as f32, (s + 1) as f32 / subs as f32);
    let i_mid = i_mid_dev(c, hour, s);
    let (hb0, hb1) = (hour.b0 + (hour.b1 - hour.b0) * f0, hour.b0 + (hour.b1 - hour.b0) * f1);
    let depth = depth_dev(c, h, hb0);
    let edge = section_dev(c, &hour.p, &hour.q_eps, &depth.d_edge);
    let store = section_dev(c, &hour.p, &hour.q_eps, &depth.depth);
    let d_s = storage_rate_dev(c, &store, dt_s);
    let newton = newton_dev(c, dt_s, h, qc, &hour.n2, &depth, &edge);
    let control = control_dev(c, dt_s, h, &hour.p, &hour.q_eps, &hour.n_eff, &depth.depth, &store, &d_s);
    let a_values = matrix_dev(c, &d_s, &newton.a_up, &newton.b_down, &control.r_prime);
    let r = rhs_dev(c, qc, &newton.q_star, &control.r_t, &newton.b_down, &hour.qp_stage, &i_mid, hb0, hb1);
    AssembledDev { i_mid, hb0, hb1, dt_s, depth, edge, store, d_s, newton, control, a_values, r }
}

/// Test hook: copy an `AssembledDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn assembled_to_host(a: &AssembledDev) -> Assembled {
    Assembled {
        i_mid: cube_to_host(&a.i_mid),
        hb0: a.hb0,
        hb1: a.hb1,
        dt_s: a.dt_s,
        depth: a.depth.download(),
        edge: a.edge.download(),
        store: a.store.download(),
        d_s: cube_to_host(&a.d_s),
        newton: a.newton.download(),
        control: a.control.download(),
        a_values: cube_to_host(&a.a_values),
        r: cube_to_host(&a.r),
    }
}

// ----------------------------------------------------------------- readout

/// `(w_edge * q_new + keep_of(w_edge) * qc) * in_stage`: the edge flux of the
/// mass sums. Mirrors the `fe` closure of `stage_readout_host`.
#[cube]
fn fe_inner_of(w_edge: &Tensor<f32>, q_new: &Tensor<f32>, qc: &Tensor<f32>, i: usize) -> f32 {
    fma(w_edge[i], q_new[i], keep_of(w_edge[i]) * qc[i])
}

/// 7b to 8b. Mirrors the first loop of `stage_readout_host`.
#[cube(launch)]
fn readout_kernel(
    h: &Tensor<f32>,
    dh: &Tensor<f32>,
    down: &Tensor<i32>,
    outlet: &Tensor<f32>,
    z_mid: &Tensor<f32>,
    ctrl: &Tensor<f32>,
    q_star: &Tensor<f32>,
    a_up: &Tensor<f32>,
    b_down: &Tensor<f32>,
    r_t: &Tensor<f32>,
    r_prime: &Tensor<f32>,
    o_h_solved: &mut Tensor<f32>,
    o_h_new: &mut Tensor<f32>,
    o_dh_down: &mut Tensor<f32>,
    o_q_new: &mut Tensor<f32>,
    n: u32,
    lb: f32,
    dhb: f32,
    #[comptime] prescribed: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        // 7b
        let hs = h[i] + dh[i];
        let hf = z_mid[i] + lb;
        o_h_solved[i] = hs;
        o_h_new[i] = select(hs < hf, hf, hs);
        // 8a
        let mut dhd = dh[usize::cast_from(down[i])];
        if prescribed {
            dhd = select(outlet[i] == 1.0f32, dhb, dhd);
        }
        o_dh_down[i] = dhd;
        // 8b
        let q_fr = fma(-b_down[i], dhd, fma(a_up[i], dh[i], q_star[i]));
        let q_ctrl_sum = fma(r_prime[i], dh[i], r_t[i]);
        o_q_new[i] = fma(q_ctrl_sum, ctrl[i], q_fr);
    }
}

/// Block 9, per reach: row `t` of `o` (length `7 * n`) holds the terms of the
/// sums, each before the `* dt_s`: storage, net, floor, the three gross parts,
/// `mc_in`. Mirrors the `fe`, `into`, `net`, `storage`, `floor`, `gross` and
/// `mc_in` lines of `stage_readout_host`.
#[cube(launch)]
fn mass_terms_kernel(
    h: &Tensor<f32>,
    qc: &Tensor<f32>,
    h_new: &Tensor<f32>,
    h_solved: &Tensor<f32>,
    q_new: &Tensor<f32>,
    d_s: &Tensor<f32>,
    i_mid: &Tensor<f32>,
    qp_stage: &Tensor<f32>,
    in_stage: &Tensor<f32>,
    has_down: &Tensor<f32>,
    w_edge: &Tensor<f32>,
    crow: &Tensor<i32>,
    col: &Tensor<i32>,
    adj_stage: &Tensor<f32>,
    o: &mut Tensor<f32>,
    n: u32,
) {
    let i = ABSOLUTE_POS;
    let nn = usize::cast_from(n);
    if i < nn {
        let fe_inner = fe_inner_of(w_edge, q_new, qc, i);
        let fe = fe_inner * in_stage[i];
        let s0 = usize::cast_from(crow[i]);
        let s1 = usize::cast_from(crow[i + 1]);
        let mut into = 0.0f32;
        for k in s0..s1 {
            let cc = usize::cast_from(col[k]);
            into = fma(fe_inner_of(w_edge, q_new, qc, cc) * in_stage[cc], adj_stage[k], into);
        }
        o[i] = (d_s[i] * (h_new[i] - h[i])) * in_stage[i];
        o[nn + i] = ((into + i_mid[i]) + qp_stage[i]) - fe;
        o[2 * nn + i] = (d_s[i] * (h_new[i] - h_solved[i])) * in_stage[i];
        o[3 * nn + i] = f32::abs(fe) * fma(has_down[i], in_stage[i], in_stage[i]);
        o[4 * nn + i] = f32::abs(i_mid[i]);
        o[5 * nn + i] = f32::abs(qp_stage[i]);
        o[6 * nn + i] = i_mid[i];
    }
}

/// `[storage, net, floor, gross, mc_in]` from the seven sums:
/// each `sum * dt_s`, and `gross = ((g_fe + g_imid) + g_qp) * dt_s`.
#[cube(launch)]
fn mass_combine_kernel(sums: &Tensor<f32>, o: &mut Tensor<f32>, dt_s: f32) {
    if ABSOLUTE_POS == 0 {
        o[0] = sums[0] * dt_s;
        o[1] = sums[1] * dt_s;
        o[2] = sums[2] * dt_s;
        o[3] = ((sums[3] + sums[4]) + sums[5]) * dt_s;
        o[4] = sums[6] * dt_s;
    }
}

/// Output of blocks 7b to 9.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ReadoutDev {
    pub h_solved: Cube,
    pub h_new: Cube,
    pub dh_down: Cube,
    pub q_new: Cube,
    /// `[storage, net, floor, gross, mc_in]`.
    pub mass: Cube,
}

/// 7b to 9 on the device, from the solve output `dh`. Mirrors
/// `stage_readout_host`. The seven mass terms go to a `[7, n]` buffer and one
/// `float_sum_dim` reduces them; the combine is done on the device.
#[doc(hidden)]
pub fn stage_readout_dev(
    c: &StageOpConstsDev,
    hour: &HourDev,
    h: &Cube,
    qc: &Cube,
    asm: &AssembledDev,
    dh: &Cube,
) -> ReadoutDev {
    let nn = c.host.n;
    let o = || empty_f32(h, nn);
    let (h_solved, h_new, dh_down, q_new) = (o(), o(), o(), o());
    let (count, dim) = grid(nn);
    readout_kernel::launch::<CudaRuntime>(
        &h.client,
        count.clone(),
        dim.clone(),
        a!(h),
        a!(dh),
        a!(c.down),
        a!(c.outlet),
        a!(c.z_mid),
        a!(c.ctrl),
        a!(asm.newton.q_star),
        a!(asm.newton.a_up),
        a!(asm.newton.b_down),
        a!(asm.control.r_t),
        a!(asm.control.r_prime),
        a!(h_solved),
        a!(h_new),
        a!(dh_down),
        a!(q_new),
        nn as u32,
        c.host.depth_lb,
        asm.hb1 - asm.hb0,
        c.host.prescribed,
    );
    let terms = empty_device_contiguous_dtype(
        h.client.clone(),
        h.device.clone(),
        Shape::from(vec![7, nn]),
        DType::F32,
    );
    mass_terms_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(h),
        a!(qc),
        a!(h_new),
        a!(h_solved),
        a!(q_new),
        a!(asm.d_s),
        a!(asm.i_mid),
        a!(hour.qp_stage),
        a!(c.in_stage),
        a!(c.has_down),
        a!(c.w_edge),
        a!(c.tree.crow),
        a!(c.tree.col),
        a!(c.adj_stage),
        a!(terms),
        nn as u32,
    );
    let sums = into_cube::<Gpu>(<Gpu as FloatTensorOps<Gpu>>::float_sum_dim(from_cube::<Gpu>(terms), 1));
    let mass = empty_f32(h, 5);
    mass_combine_kernel::launch::<CudaRuntime>(
        &h.client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(32),
        a!(sums),
        a!(mass),
        asm.dt_s,
    );
    ReadoutDev { h_solved, h_new, dh_down, q_new, mass }
}

// --------------------------------------------------------------- the hour

/// `acc[i] += x[i]`, the host's `*m += v` over the five mass sums.
#[cube(launch)]
fn mass_add_kernel(acc: &mut Tensor<f32>, x: &Tensor<f32>) {
    let i = ABSOLUTE_POS;
    if i < 5 {
        acc[i] = acc[i] + x[i];
    }
}

/// H5 and the output: `out = [h_end, q_next]`, `q_next_i = in_i == 1 ? Q_i :
/// q_next_mc_i`. Mirrors `stage_epilogue_host`.
#[cube(launch)]
fn epilogue_kernel(
    h_end: &Tensor<f32>,
    qc: &Tensor<f32>,
    q_next_mc: &Tensor<f32>,
    in_stage: &Tensor<f32>,
    out: &mut Tensor<f32>,
    n: u32,
) {
    let i = ABSOLUTE_POS;
    let nn = usize::cast_from(n);
    if i < nn {
        out[i] = h_end[i];
        out[nn + i] = select(in_stage[i] == 1.0f32, qc[i], q_next_mc[i]);
    }
}

/// H5 on the device: `[h_end, q_next]` from the end state `(h, qc)`.
/// Mirrors `stage_epilogue_host` plus the stage copy.
#[doc(hidden)]
pub fn stage_epilogue_dev(c: &StageOpConstsDev, hour: &HourDev, h: &Cube, qc: &Cube) -> Cube {
    let nn = c.host.n;
    let out = empty_f32(h, 2 * nn);
    let (count, dim) = grid(nn);
    epilogue_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(h),
        a!(qc),
        a!(hour.q_next_mc),
        a!(c.in_stage),
        a!(out),
        nn as u32,
    );
    out
}

/// One substep's start state, its solve output and its factors, on the device.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SubstepDev {
    pub h: Cube,
    pub qc: Cube,
    pub dh: Cube,
    pub factors: TreeFactorsDev,
}

impl SubstepDev {
    /// Copy a host `SubstepSaved` (start state, solve output, factors) to the device of `like`.
    /// Tests use it to feed the device backward the host's own saved state.
    #[doc(hidden)]
    pub fn upload(sv: &SubstepSaved, like: &Cube) -> Self {
        Self {
            h: upload_f32(like, &sv.h),
            qc: upload_f32(like, &sv.qc),
            dh: upload_f32(like, &sv.dh),
            factors: TreeFactorsDev {
                l_values: upload_f32(like, &sv.factors.l_values),
                u_values: upload_f32(like, &sv.factors.u_values),
            },
        }
    }
}

/// The hour forward on the device (the twin of `HourForward`).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct HourForwardDev {
    /// `[2n] = [h_end, q_next]`.
    pub out: Cube,
    pub h_end: Cube,
    pub last: AssembledDev,
    /// `[storage, net, floor, gross, mc_in]`.
    pub mass: Cube,
    /// Empty unless `save`.
    pub subs: Vec<SubstepDev>,
}

/// The hour forward: `substeps` substeps of assemble, factor, solve and
/// readout. Mirrors `stage_hour_forward`. No host read.
#[doc(hidden)]
pub fn stage_hour_forward_dev(
    c: &StageOpConstsDev,
    pattern: &CsrPattern,
    hour: &HourDev,
    h_t: &Cube,
    q_t: &Cube,
    save: bool,
) -> HourForwardDev {
    let subs = c.host.substeps.max(1) as usize;
    let mut h = h_t.clone();
    let mut qc = q_t.clone();
    let mass = zeros_f32(h_t, 5);
    let mut saved = Vec::new();
    let mut last = None;
    for s in 0..subs {
        let asm = stage_assemble_dev(c, hour, s, &h, &qc);
        let factors = tree_factor_dev(&c.tree, &asm.a_values);
        let dh = tree_solve_dev(pattern, &factors, &asm.r);
        let ro = stage_readout_dev(c, hour, &h, &qc, &asm, &dh);
        mass_add_kernel::launch::<CudaRuntime>(
            &h.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            a!(mass),
            a!(ro.mass),
        );
        if save {
            saved.push(SubstepDev { h: h.clone(), qc: qc.clone(), dh, factors });
        }
        h = ro.h_new;
        qc = ro.q_new;
        last = Some(asm);
    }
    let out = stage_epilogue_dev(c, hour, &h, &qc);
    HourForwardDev { out, h_end: h, last: last.expect("at least one substep"), mass, subs: saved }
}

/// Test hook: copy a `[5]` mass buffer to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn mass_to_host(m: &Cube) -> [f32; 5] {
    let v = cube_to_host(m);
    [v[0], v[1], v[2], v[3], v[4]]
}

/// Test hook: copy a `ReadoutDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn readout_to_host(r: &ReadoutDev) -> Readout {
    Readout {
        h_solved: cube_to_host(&r.h_solved),
        h_new: cube_to_host(&r.h_new),
        dh_down: cube_to_host(&r.dh_down),
        q_new: cube_to_host(&r.q_new),
        mass: mass_to_host(&r.mass),
    }
}

// ------------------------------------------------------------- test hooks

/// Test hook: copy a `DepthDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn depth_to_host(d: &DepthDev) -> DepthOut {
    d.download()
}

/// Test hook: copy a `SectionDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn section_to_host(d: &SectionDev) -> SectionHost {
    d.download()
}

/// Test hook: copy a `NewtonDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn newton_to_host(d: &NewtonDev) -> NewtonOut {
    d.download()
}

/// Test hook: copy a `ControlDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn control_to_host(d: &ControlDev) -> ControlOut {
    d.download()
}

/// Test hook: copy an `HourDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn hour_to_host(d: &HourDev) -> HourHost {
    d.download()
}
