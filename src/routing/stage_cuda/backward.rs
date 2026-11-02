//! Backward kernels of the stage block: the output, readout, RHS, matrix,
//! section, Newton, control, depth and prologue adjoints, and the glue that
//! composes one substep adjoint. Each kernel body is the loop body of the host
//! adjoint fn in `src/routing/stage_op.rs` (`output_backward_host`,
//! `readout_backward_host`, `rhs_backward_host`, `matrix_backward_host`,
//! `section_backward_host`, `newton_backward_host`, `control_backward_host`,
//! `depth_backward_host`, `prologue_backward_host`), line for line.
//!
//! Rules (see the tie table in the `stage_op.rs` module doc):
//! - every add that takes a product is one explicit `fma`, at the same site
//!   and in the same operand order as the `f32::mul_add` of the host fn.
//!   NVRTC fuses a product that feeds an add even when the product has other
//!   uses, so the per-kernel SASS gap (FFMA with --fmad=true against
//!   --fmad=false) decides which sites are fused. Every such site is written
//!   as an explicit `fma` here and `mul_add` on the host;
//! - the mask of the floor uses the host comparison, `!(hs < hf)`;
//! - a scatter of the host (`x[down[i]] += ...`, `x[col[k]] += ...`) is a
//!   gather here, in the host's slot order. In a tree every reach has one
//!   downstream reach, so a column has at most two slots: its diagonal slot
//!   and its `down_slot`, in that order.
//!
//! - a sum that starts at 0 (`d = 0; d += x`) copies the host's `+=` onto a
//!   zeroed Vec, which turns a -0 into +0. The kernels start at 0.0 and add
//!   too. Do not "simplify" these to a direct store.
//! - `section_backward_kernel` is the one backward section kernel. Newton,
//!   control and the glue read its outputs; no other kernel does section
//!   adjoint math.
//!
//! Nothing here reads a device buffer to the host except the `*_to_host` test hooks.

use burn::tensor::ops::FloatTensorOps;
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

use super::forward::AssembledDev;
use super::{up, ControlDev, DepthDev, HourDev, NewtonDev, SectionDev, StageOpConstsDev};
use crate::routing::stage_op::{
    ControlBar, ControlGrads, HourAcc, HourGrads, MatrixGrads, NewtonBar, NewtonGrads, ReadoutGrads, RhsGrads,
    SectionBar, SectionGrads, F23, F43, F53,
};
use crate::sparse::cube_buf::{cube_to_host, empty_f32, from_cube, grid, into_cube, zeros_f32, Cube, Gpu};

macro_rules! a {
    ($x:expr) => {
        $x.clone().into_tensor_arg()
    };
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

// ------------------------------------------------------------------ output

/// BH0. Mirrors `output_backward_host`. No sum takes a product, so nothing fuses.
#[cube(launch)]
fn output_backward_kernel(
    g_out: &Tensor<f32>,
    in_stage: &Tensor<f32>,
    o_g_h: &mut Tensor<f32>,
    o_g_q: &mut Tensor<f32>,
    o_g_mc: &mut Tensor<f32>,
    n: u32,
) {
    let i = ABSOLUTE_POS;
    let nn = usize::cast_from(n);
    if i < nn {
        o_g_h[i] = g_out[i];
        o_g_q[i] = g_out[nn + i] * in_stage[i];
        o_g_mc[i] = g_out[nn + i] * keep_of(in_stage[i]);
    }
}

/// BH0 on the device: `(g_h, g_q, g_mc)` from the `[2n]` output cotangent.
#[doc(hidden)]
pub fn output_backward_dev(c: &StageOpConstsDev, g_out: &Cube) -> (Cube, Cube, Cube) {
    let nn = c.host.n;
    let (g_h, g_q, g_mc) = (empty_f32(g_out, nn), empty_f32(g_out, nn), empty_f32(g_out, nn));
    let (count, dim) = grid(nn);
    output_backward_kernel::launch::<CudaRuntime>(
        &g_out.client,
        count,
        dim,
        a!(g_out),
        a!(c.in_stage),
        a!(g_h),
        a!(g_q),
        a!(g_mc),
        nn as u32,
    );
    (g_h, g_q, g_mc)
}

// ----------------------------------------------------------------- readout

/// B8 and B7 per reach. Mirrors the first loop of `readout_backward_host`.
/// `o_g_dhd` is the cotangent of the gathered `dh[down]`; the next kernel
/// adds it into `o_dh`.
#[cube(launch)]
fn readout_backward_kernel(
    h: &Tensor<f32>,
    dh: &Tensor<f32>,
    down: &Tensor<i32>,
    outlet: &Tensor<f32>,
    z_mid: &Tensor<f32>,
    ctrl: &Tensor<f32>,
    a_up: &Tensor<f32>,
    b_down: &Tensor<f32>,
    r_prime: &Tensor<f32>,
    g_h_new: &Tensor<f32>,
    g_q_new: &Tensor<f32>,
    o_dh: &mut Tensor<f32>,
    o_h: &mut Tensor<f32>,
    o_q_star: &mut Tensor<f32>,
    o_a_up: &mut Tensor<f32>,
    o_b_down: &mut Tensor<f32>,
    o_r_t: &mut Tensor<f32>,
    o_r_prime: &mut Tensor<f32>,
    o_g_dhd: &mut Tensor<f32>,
    n: u32,
    lb: f32,
    dhb: f32,
    #[comptime] prescribed: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let gq = g_q_new[i];
        let mut dhd = dh[usize::cast_from(down[i])];
        if prescribed {
            dhd = select(outlet[i] == 1.0f32, dhb, dhd);
        }
        // B8
        o_q_star[i] = gq;
        o_a_up[i] = gq * dh[i];
        o_b_down[i] = -gq * dhd;
        let gc = gq * ctrl[i];
        // The left product fuses; `gc` has three uses and stays rounded.
        // `d` starts at 0 and adds, as the host's `+=` on a zeroed Vec does (-0 becomes +0). Do not make it a direct store.
        let mut d = 0.0f32;
        d += fma(gq, a_up[i], gc * r_prime[i]);
        o_r_t[i] = gc;
        o_r_prime[i] = gc * dh[i];
        let g = -gq * b_down[i];
        if prescribed {
            o_g_dhd[i] = g * keep_of(outlet[i]);
        } else {
            o_g_dhd[i] = g;
        }
        // B7: the floor
        let hs = h[i] + dh[i];
        let hf = z_mid[i] + lb;
        let ghs = g_h_new[i] * ind(!(hs < hf));
        o_h[i] = ghs;
        d += ghs;
        o_dh[i] = d;
    }
}

/// The gather's transpose: `dh[down[i]] += g_dhd[i]`, as a gather. The
/// reaches whose `down` is `t` are the other columns of row `t` (in column
/// order, which is the host's loop order), then `t` itself at an outlet,
/// where the host loop reaches `i == t` last.
#[cube(launch)]
fn down_transpose_kernel(
    g_dhd: &Tensor<f32>,
    crow: &Tensor<i32>,
    col: &Tensor<i32>,
    down_slot: &Tensor<i32>,
    o_dh: &mut Tensor<f32>,
    n: u32,
) {
    let t = ABSOLUTE_POS;
    if t < usize::cast_from(n) {
        let s0 = usize::cast_from(crow[t]);
        let s1 = usize::cast_from(crow[t + 1]);
        let mut acc = o_dh[t];
        for j in s0..s1 {
            let cj = usize::cast_from(col[j]);
            if cj != t {
                acc += g_dhd[cj];
            }
        }
        if down_slot[t] < 0 {
            acc += g_dhd[t];
        }
        o_dh[t] = acc;
    }
}

/// Cotangents out of the readout, on the device (the twin of `ReadoutGrads`).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ReadoutGradsDev {
    pub dh: Cube,
    pub h: Cube,
    pub q_star: Cube,
    pub a_up: Cube,
    pub b_down: Cube,
    pub r_t: Cube,
    pub r_prime: Cube,
}

/// B8 and B7 on the device. Mirrors `readout_backward_host`.
#[doc(hidden)]
pub fn readout_backward_dev(
    c: &StageOpConstsDev,
    h: &Cube,
    asm: &AssembledDev,
    dh: &Cube,
    g_h_new: &Cube,
    g_q_new: &Cube,
) -> ReadoutGradsDev {
    let nn = c.host.n;
    let o = || empty_f32(h, nn);
    let (o_dh, o_h, o_q_star, o_a_up, o_b_down, o_r_t, o_r_prime, g_dhd) = (o(), o(), o(), o(), o(), o(), o(), o());
    let (count, dim) = grid(nn);
    readout_backward_kernel::launch::<CudaRuntime>(
        &h.client,
        count.clone(),
        dim.clone(),
        a!(h),
        a!(dh),
        a!(c.down),
        a!(c.outlet),
        a!(c.z_mid),
        a!(c.ctrl),
        a!(asm.newton.a_up),
        a!(asm.newton.b_down),
        a!(asm.control.r_prime),
        a!(g_h_new),
        a!(g_q_new),
        a!(o_dh),
        a!(o_h),
        a!(o_q_star),
        a!(o_a_up),
        a!(o_b_down),
        a!(o_r_t),
        a!(o_r_prime),
        a!(g_dhd),
        nn as u32,
        c.host.depth_lb,
        asm.hb1 - asm.hb0,
        c.host.prescribed,
    );
    down_transpose_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(g_dhd),
        a!(c.tree.crow),
        a!(c.tree.col),
        a!(c.tree.down_slot),
        a!(o_dh),
        nn as u32,
    );
    ReadoutGradsDev { dh: o_dh, h: o_h, q_star: o_q_star, a_up: o_a_up, b_down: o_b_down, r_t: o_r_t, r_prime: o_r_prime }
}

/// Test hook: copy a `ReadoutGradsDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn readout_grads_to_host(g: &ReadoutGradsDev) -> ReadoutGrads {
    ReadoutGrads {
        dh: cube_to_host(&g.dh),
        h: cube_to_host(&g.h),
        q_star: cube_to_host(&g.q_star),
        a_up: cube_to_host(&g.a_up),
        b_down: cube_to_host(&g.b_down),
        r_t: cube_to_host(&g.r_t),
        r_prime: cube_to_host(&g.r_prime),
    }
}

// --------------------------------------------------------------------- rhs

/// B6 per reach. Mirrors `rhs_backward_host`. `g_eo` is `-g_r`, plus the
/// scatter `g_eo[col] += g_r[row] * adj_stage` as a gather over the two slots
/// of column `i` (diagonal slot, then `down_slot`).
#[cube(launch)]
fn rhs_backward_kernel(
    g_r: &Tensor<f32>,
    crow: &Tensor<i32>,
    down: &Tensor<i32>,
    down_slot: &Tensor<i32>,
    adj_stage: &Tensor<f32>,
    friction: &Tensor<f32>,
    ctrl: &Tensor<f32>,
    w_ctrl: &Tensor<f32>,
    outlet: &Tensor<f32>,
    o_qp: &mut Tensor<f32>,
    o_i_mid: &mut Tensor<f32>,
    o_q_star: &mut Tensor<f32>,
    o_qc: &mut Tensor<f32>,
    o_r_t: &mut Tensor<f32>,
    o_b_down: &mut Tensor<f32>,
    n: u32,
    th: f32,
    hb0: f32,
    hb1: f32,
    #[comptime] prescribed: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let mut g_eo = -g_r[i];
        let diag = usize::cast_from(crow[i + 1]) - 1;
        g_eo = fma(g_r[i], adj_stage[diag], g_eo);
        let ds = down_slot[i];
        if ds >= 0 {
            g_eo = fma(g_r[usize::cast_from(down[i])], adj_stage[usize::cast_from(ds)], g_eo);
        }
        o_qp[i] = g_r[i];
        o_i_mid[i] = g_r[i];
        if prescribed {
            o_b_down[i] = (g_r[i] * outlet[i]) * (th * (hb1 - hb0));
        } else {
            o_b_down[i] = 0.0f32;
        }
        o_q_star[i] = (g_eo * friction[i]) * th;
        o_qc[i] = g_eo * fma(friction[i], 1.0f32 - th, ctrl[i] - w_ctrl[i]);
        o_r_t[i] = g_eo * w_ctrl[i];
    }
}

/// Cotangents out of the RHS, on the device (the twin of `RhsGrads`).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RhsGradsDev {
    pub qp: Cube,
    pub i_mid: Cube,
    pub q_star: Cube,
    pub qc: Cube,
    pub r_t: Cube,
    pub b_down: Cube,
}

/// B6 on the device. Mirrors `rhs_backward_host`; reads only `hb0` and `hb1` of `asm`.
#[doc(hidden)]
pub fn rhs_backward_dev(c: &StageOpConstsDev, asm: &AssembledDev, g_r: &Cube) -> RhsGradsDev {
    let nn = c.host.n;
    let o = || empty_f32(g_r, nn);
    let (qp, i_mid, q_star, qc, r_t, b_down) = (o(), o(), o(), o(), o(), o());
    let (count, dim) = grid(nn);
    rhs_backward_kernel::launch::<CudaRuntime>(
        &g_r.client,
        count,
        dim,
        a!(g_r),
        a!(c.tree.crow),
        a!(c.tree.down),
        a!(c.tree.down_slot),
        a!(c.adj_stage),
        a!(c.friction),
        a!(c.ctrl),
        a!(c.w_ctrl),
        a!(c.outlet),
        a!(qp),
        a!(i_mid),
        a!(q_star),
        a!(qc),
        a!(r_t),
        a!(b_down),
        nn as u32,
        c.host.theta,
        asm.hb0,
        asm.hb1,
        c.host.prescribed,
    );
    RhsGradsDev { qp, i_mid, q_star, qc, r_t, b_down }
}

/// Test hook: copy a `RhsGradsDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn rhs_grads_to_host(g: &RhsGradsDev) -> RhsGrads {
    RhsGrads {
        qp: cube_to_host(&g.qp),
        i_mid: cube_to_host(&g.i_mid),
        q_star: cube_to_host(&g.q_star),
        qc: cube_to_host(&g.qc),
        r_t: cube_to_host(&g.r_t),
        b_down: cube_to_host(&g.b_down),
    }
}

// ------------------------------------------------------------------ matrix

/// B5, 5d and 5c, per row. Mirrors the first two loops of `matrix_backward_host`.
/// `g_a` is `[g_low, g_upf]`. `o_g_sib` is a temporary for the next kernel.
#[cube(launch)]
fn matrix_backward_row_kernel(
    g_a: &Tensor<f32>,
    crow: &Tensor<i32>,
    diag_slot: &Tensor<f32>,
    in_stage: &Tensor<f32>,
    w_ctrl: &Tensor<f32>,
    o_d_s: &mut Tensor<f32>,
    o_a_up: &mut Tensor<f32>,
    o_r_prime: &mut Tensor<f32>,
    o_g_sib: &mut Tensor<f32>,
    n: u32,
    th: f32,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let s0 = usize::cast_from(crow[i]);
        let s1 = usize::cast_from(crow[i + 1]);
        // `g_diag` starts at 0 and adds, as the host's `+=` on a zeroed Vec does (-0 becomes +0). Do not make it a direct store.
        let mut g_diag = 0.0f32;
        for k in s0..s1 {
            g_diag = fma(g_a[k], diag_slot[k], g_diag);
        }
        // `g_dst` and `g_dst * th` each have two uses and stay plain multiplies.
        let g_dst = g_diag * in_stage[i];
        let gs = g_dst * th;
        // `d`, `au` and the `fma(.., 0.0)` below start at 0 and add, like the host's `+=` on zeroed Vecs
        // (-0 becomes +0). Do not make them direct stores.
        let mut d = 0.0f32;
        d += g_dst;
        o_d_s[i] = d;
        let mut au = 0.0f32;
        au += gs;
        o_a_up[i] = au;
        o_g_sib[i] = gs;
        o_r_prime[i] = fma(g_diag, w_ctrl[i], 0.0f32);
    }
}

/// B5, 5b and 5a, per reach (the scatters over `col`, as a gather over the
/// two slots of column `c`: diagonal slot, then `down_slot`).
#[cube(launch)]
fn matrix_backward_col_kernel(
    g_a: &Tensor<f32>,
    g_sib: &Tensor<f32>,
    crow: &Tensor<i32>,
    down: &Tensor<i32>,
    down_slot: &Tensor<i32>,
    adj_friction: &Tensor<f32>,
    dam_slot: &Tensor<f32>,
    o_a_up: &mut Tensor<f32>,
    o_b_down: &mut Tensor<f32>,
    o_r_prime: &mut Tensor<f32>,
    n: u32,
    nnz: u32,
    th: f32,
    thc: f32,
) {
    let c = ABSOLUTE_POS;
    if c < usize::cast_from(n) {
        let nz = usize::cast_from(nnz);
        // `bd` starts at 0 and adds, as the host's `+=` on a zeroed Vec does (-0 becomes +0). Do not make it a direct store.
        let mut au = o_a_up[c];
        let mut bd = 0.0f32;
        let mut rp = o_r_prime[c];
        let nth = -th;
        let nthc = -thc;
        let ds = down_slot[c];
        for t in 0..2usize {
            let first = t == 0;
            let k = select(first, usize::cast_from(crow[c + 1]) - 1, usize::cast_from(ds));
            let row = select(first, c, usize::cast_from(down[c]));
            if first || ds >= 0 {
                let g_low = g_a[k];
                let g_upf = g_a[nz + k];
                au = fma(nth * g_low, adj_friction[k], au);
                bd = fma(fma(nth, g_upf, g_sib[row]), adj_friction[k], bd);
                rp = fma(nthc * g_low, dam_slot[k], rp);
            }
        }
        o_a_up[c] = au;
        o_b_down[c] = bd;
        o_r_prime[c] = rp;
    }
}

/// Cotangents out of the matrix, on the device (the twin of `MatrixGrads`).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct MatrixGradsDev {
    pub d_s: Cube,
    pub a_up: Cube,
    pub b_down: Cube,
    pub r_prime: Cube,
}

/// B5 on the device. Mirrors `matrix_backward_host`; `g_a_values` has length `2 * nnz`.
#[doc(hidden)]
pub fn matrix_backward_dev(c: &StageOpConstsDev, g_a_values: &Cube) -> MatrixGradsDev {
    let (nn, nnz) = (c.host.n, c.host.nnz);
    let o = || empty_f32(g_a_values, nn);
    let (d_s, a_up, b_down, r_prime, g_sib) = (o(), o(), o(), o(), o());
    let (count, dim) = grid(nn);
    matrix_backward_row_kernel::launch::<CudaRuntime>(
        &g_a_values.client,
        count.clone(),
        dim.clone(),
        a!(g_a_values),
        a!(c.tree.crow),
        a!(c.diag_slot),
        a!(c.in_stage),
        a!(c.w_ctrl),
        a!(d_s),
        a!(a_up),
        a!(r_prime),
        a!(g_sib),
        nn as u32,
        c.host.theta,
    );
    matrix_backward_col_kernel::launch::<CudaRuntime>(
        &g_a_values.client,
        count,
        dim,
        a!(g_a_values),
        a!(g_sib),
        a!(c.tree.crow),
        a!(c.tree.down),
        a!(c.tree.down_slot),
        a!(c.adj_friction),
        a!(c.dam_slot),
        a!(a_up),
        a!(b_down),
        a!(r_prime),
        nn as u32,
        nnz as u32,
        c.host.theta,
        c.host.theta_c,
    );
    MatrixGradsDev { d_s, a_up, b_down, r_prime }
}

/// Test hook: copy a `MatrixGradsDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn matrix_grads_to_host(g: &MatrixGradsDev) -> MatrixGrads {
    MatrixGrads {
        d_s: cube_to_host(&g.d_s),
        a_up: cube_to_host(&g.a_up),
        b_down: cube_to_host(&g.b_down),
        r_prime: cube_to_host(&g.r_prime),
    }
}

// ----------------------------------------------------------------- section

/// `sign(x)` with 0 at 0 (and at NaN). Mirrors `sign0` of `stage_op.rs`.
#[cube]
fn sign0(x: f32) -> f32 {
    select(x > 0.0f32, 1.0f32, select(x < 0.0f32, -1.0f32, 0.0f32))
}

/// B2 per reach: the reverse of 2g to 2a. Mirrors `section_backward_host`. This
/// is the one backward section kernel of the CUDA path. Fused sites are the
/// `mul_add` calls of the host fn, in the same operand order. NVRTC fuses a
/// product that feeds an add even when the product has other uses, and some
/// products by 2.0 too. The per-kernel SASS gap (FFMA at --fmad=true against
/// --fmad=false) decides which sites are fused; each one is an explicit `fma`
/// here and `mul_add` on the host.
#[cube(launch)]
fn section_backward_kernel(
    p: &Tensor<f32>,
    q_eps: &Tensor<f32>,
    d: &Tensor<f32>,
    s_pw: &Tensor<f32>,
    s_tw: &Tensor<f32>,
    s_ss_raw: &Tensor<f32>,
    s_ss: &Tensor<f32>,
    s_bw_raw: &Tensor<f32>,
    s_bw: &Tensor<f32>,
    s_area: &Tensor<f32>,
    s_root: &Tensor<f32>,
    s_perim: &Tensor<f32>,
    s_tw_d: &Tensor<f32>,
    s_ss_d: &Tensor<f32>,
    s_bw_d: &Tensor<f32>,
    b_tw: &Tensor<f32>,
    b_a: &Tensor<f32>,
    b_p: &Tensor<f32>,
    b_r: &Tensor<f32>,
    b_ad: &Tensor<f32>,
    b_pd: &Tensor<f32>,
    t_mask: &Tensor<f32>,
    t_tw: &Tensor<f32>,
    o_p: &mut Tensor<f32>,
    o_q_eps: &mut Tensor<f32>,
    o_d: &mut Tensor<f32>,
    n: u32,
    bwlb: f32,
    #[comptime] has_table: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let e = q_eps[i];
        let dd = d[i];
        let r_tw = b_tw[i];
        let r_a = b_a[i];
        let r_p = b_p[i];
        let r_r = b_r[i];
        let r_ad = b_ad[i];
        let r_pd = b_pd[i];
        // gd and ge start at 0 and add, as the host's `+=` on a zeroed value does (-0 becomes +0).
        let mut gd = 0.0f32;
        let mut ge = 0.0f32;
        // keep is 1.0 without a table, so `x * keep` is exact and the code below is one path.
        let mut keep = 1.0f32;
        if has_table {
            let m = t_mask[i];
            keep = keep_of(m);
            let ttw = t_tw[i];
            let perim_r = ttw + dd * 2.0f32;
            let num = fma(ttw, perim_r, -((ttw * dd) * 2.0f32));
            let inner = fma(r_a, ttw, r_p * 2.0f32);
            let sum = fma(r_r, num / (perim_r * perim_r), inner);
            gd = fma(m, sum, gd);
        }
        let g_r = r_r * keep;
        let g_ad = r_ad * keep;
        let g_pd = r_pd * keep;
        let pw = s_pw[i];
        let tw = s_tw[i];
        let ssr = s_ss_raw[i];
        let ss = s_ss[i];
        let bwr = s_bw_raw[i];
        let bw = s_bw[i];
        let area = s_area[i];
        let rt = s_root[i];
        let perim = s_perim[i];
        let twd = s_tw_d[i];
        let ssd = s_ss_d[i];
        let bwd = s_bw_d[i];
        let sa = ind(ssr > 0.5f32) * ind(ssr < 50.0f32);
        let ba = ind(bwr > bwlb);

        // 2f: Pd = (bwd + rt*2) + (((d*ss)*ssd)*2)/rt
        let u = ((dd * ss) * ssd) * 2.0f32;
        // The start at 0 copies the host's add onto a zeroed Vec (a -0 becomes +0). Do not simplify it to a plain product.
        let mut g_bwd = fma(r_pd, keep, 0.0f32);
        // The start at 0 copies the host's add onto a zeroed Vec (a -0 becomes +0). Do not simplify it to a plain product.
        let mut g_rt = fma(g_pd, 2.0f32, 0.0f32);
        g_rt = fma(g_pd, -(u / (rt * rt)), g_rt);
        let g_u = g_pd / rt;
        let g_dss_ssd = g_u * 2.0f32;
        let g_dss = g_dss_ssd * ssd;
        // The start at 0 copies the host's add onto a zeroed Vec (a -0 becomes +0). Do not simplify it to a plain product.
        let mut g_ssd = fma(g_dss_ssd, dd * ss, 0.0f32);
        gd = fma(g_dss, ss, gd);
        // The start at 0 copies the host's add onto a zeroed Vec (a -0 becomes +0). Do not simplify it to a plain product.
        let mut g_ss = fma(g_dss, dd, 0.0f32);
        // 2f: Ad = (tw + bw)/2 + (d*(twd + bwd))/2
        let g_ad_half = g_ad * 0.5f32;
        let mut g_tw = fma(r_tw, keep, g_ad_half);
        // The start at 0 copies the host's add onto a zeroed Vec (a -0 becomes +0). Do not simplify it to a plain product.
        let mut g_bw = fma(g_ad, 0.5f32, 0.0f32);
        let g_x = g_ad_half;
        gd = fma(g_x, twd + bwd, gd);
        // The start at 0 copies the host's add onto a zeroed Vec (a -0 becomes +0). Do not simplify it to a plain product.
        let mut g_twd = fma(g_x, dd, 0.0f32);
        g_bwd = fma(g_x, dd, g_bwd);

        // 2e: bwd = ((twd - ss*2) - (d*ssd)*2) * ba
        let g_b = g_bwd * ba;
        g_twd = fma(g_bwd, ba, g_twd);
        g_ss = fma(g_b, -2.0f32, g_ss);
        let g_dssd = g_b * -2.0f32;
        gd = fma(g_dssd, ssd, gd);
        g_ssd = fma(g_dssd, dd, g_ssd);

        // 2d: ssd = ((ssr/d) * ((twd/tw)*d - 1)) * sa
        let g_s = g_ssd * sa;
        let ratio = ssr / dd;
        let u2 = fma(twd / tw, dd, -1.0f32);
        let g_ratio = g_s * u2;
        let g_u2 = g_s * ratio;
        let mut g_ssr = 0.0f32 + g_ratio / dd;
        gd = gd + (-g_ratio * ssr) / (dd * dd);
        let g_tt = g_u2 * dd;
        gd = fma(g_u2, twd / tw, gd);
        g_twd = g_twd + g_tt / tw;
        g_tw = g_tw + (-g_tt * twd) / (tw * tw);
        // 2d: twd = (e*tw)/d
        ge = ge + (g_twd * tw) / dd;
        g_tw = g_tw + (g_twd * e) / dd;
        gd = gd + (-g_twd * (e * tw)) / (dd * dd);

        // 2c: R = A/P; P = bw + (d*rt)*2; rt = sqrt(ss*ss + 1); A = ((tw + bw)*d)/2
        let g_area = fma(r_a, keep, g_r / perim);
        let g_perim = fma(r_p, keep, -((g_r * area) / (perim * perim)));
        g_bw = g_bw + g_perim;
        let g_perim2 = g_perim * 2.0f32;
        gd = fma(g_perim2, rt, gd);
        g_rt = fma(g_perim2, dd, g_rt);
        g_ss = g_ss + (g_rt * ss) / rt;
        g_tw = fma(g_area * dd, 0.5f32, g_tw);
        g_bw = fma(g_area * dd, 0.5f32, g_bw);
        gd = fma(g_area * (tw + bw), 0.5f32, gd);

        // 2b: bw = clamp_min(bwr, bwlb); bwr = tw - (ss*d)*2
        let ib = ind(!(bwr < bwlb));
        let g_bwr = g_bw * ib;
        g_tw = fma(g_bw, ib, g_tw);
        g_ss = fma(g_bwr, -2.0f32 * dd, g_ss);
        gd = fma(g_bwr, -2.0f32 * ss, gd);

        // 2a: ss = clamp(ssr, 0.5, 50); ssr = (tw*e)/(d*2); tw = p*pw; pw = d^e
        g_ssr = fma(g_ss, ind(ssr >= 0.5f32 && ssr <= 50.0f32), g_ssr);
        g_tw = g_tw + (g_ssr * e) / (dd * 2.0f32);
        ge = ge + (g_ssr * tw) / (dd * 2.0f32);
        gd = gd + (-g_ssr * (tw * e)) / ((2.0f32 * dd) * dd);
        o_p[i] = g_tw * pw;
        let g_pw = g_tw * p[i];
        gd = fma(g_pw * e, f32::powf(dd, e - 1.0f32), gd);
        ge = fma(g_pw * pw, f32::ln(dd), ge);

        o_q_eps[i] = ge;
        o_d[i] = gd;
    }
}

/// Cotangents of the six outputs of block 2g, on the device. Every field has
/// length `n`; a host field that is empty uploads as zeros.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SectionBarDev {
    pub top_width: Cube,
    pub area: Cube,
    pub perim: Cube,
    pub radius: Cube,
    pub a_d: Cube,
    pub p_d: Cube,
}

impl SectionBarDev {
    /// Copy `h` to the device of `like`. An empty host Vec becomes zeros; the
    /// length is the longest field (at least 1).
    #[doc(hidden)]
    pub fn upload(h: &SectionBar, like: &Cube) -> Self {
        let n = [h.top_width.len(), h.area.len(), h.perim.len(), h.radius.len(), h.a_d.len(), h.p_d.len()]
            .into_iter()
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            top_width: up(like, &h.top_width, n),
            area: up(like, &h.area, n),
            perim: up(like, &h.perim, n),
            radius: up(like, &h.radius, n),
            a_d: up(like, &h.a_d, n),
            p_d: up(like, &h.p_d, n),
        }
    }
}

/// The device twin of `SectionGrads`.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SectionGradsDev {
    pub p: Cube,
    pub q_eps: Cube,
    pub d: Cube,
}

/// B2 on the device: the reverse of 2g to 2a at the depths `d`. THE backward
/// section seam of the CUDA path, one kernel. Mirrors `section_backward_host`.
#[doc(hidden)]
pub fn section_backward_dev(
    c: &StageOpConstsDev,
    p: &Cube,
    q_eps: &Cube,
    d: &Cube,
    s: &SectionDev,
    bar: &SectionBarDev,
) -> SectionGradsDev {
    let nn = c.host.n;
    let (o_p, o_q_eps, o_d) = (empty_f32(d, nn), empty_f32(d, nn), empty_f32(d, nn));
    let (count, dim) = grid(nn);
    section_backward_kernel::launch::<CudaRuntime>(
        &d.client,
        count,
        dim,
        a!(p),
        a!(q_eps),
        a!(d),
        a!(s.pw),
        a!(s.tw),
        a!(s.ss_raw),
        a!(s.ss),
        a!(s.bw_raw),
        a!(s.bw),
        a!(s.area),
        a!(s.root),
        a!(s.perim),
        a!(s.tw_d),
        a!(s.ss_d),
        a!(s.bw_d),
        a!(bar.top_width),
        a!(bar.area),
        a!(bar.perim),
        a!(bar.radius),
        a!(bar.a_d),
        a!(bar.p_d),
        a!(c.t_mask),
        a!(c.t_tw),
        a!(o_p),
        a!(o_q_eps),
        a!(o_d),
        nn as u32,
        c.host.bw_lb,
        c.host.has_table,
    );
    SectionGradsDev { p: o_p, q_eps: o_q_eps, d: o_d }
}

/// Test hook: copy a `SectionGradsDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn section_grads_to_host(g: &SectionGradsDev) -> SectionGrads {
    SectionGrads { p: cube_to_host(&g.p), q_eps: cube_to_host(&g.q_eps), d: cube_to_host(&g.d) }
}

// ------------------------------------------------------------------ newton

/// B3 per reach: the reverse of 3f to 3a. Mirrors `newton_backward_host`. The
/// section values it reads are the saved edge section; no section math here.
#[cube(launch)]
fn newton_backward_kernel(
    qc: &Tensor<f32>,
    n2: &Tensor<f32>,
    s_up: &Tensor<f32>,
    s_dn: &Tensor<f32>,
    e_area: &Tensor<f32>,
    e_radius: &Tensor<f32>,
    e_a_d: &Tensor<f32>,
    e_p_d: &Tensor<f32>,
    e_perim: &Tensor<f32>,
    friction: &Tensor<f32>,
    edge_length: &Tensor<f32>,
    o_phi: &Tensor<f32>,
    o_dt_phi: &Tensor<f32>,
    o_den: &Tensor<f32>,
    o_alpha: &Tensor<f32>,
    o_beta: &Tensor<f32>,
    o_phi_d: &Tensor<f32>,
    o_dh_edge: &Tensor<f32>,
    o_k1: &Tensor<f32>,
    o_ra: &Tensor<f32>,
    o_rb: &Tensor<f32>,
    b_q_star: &Tensor<f32>,
    b_a_up: &Tensor<f32>,
    b_b_down: &Tensor<f32>,
    r_qc: &mut Tensor<f32>,
    r_n2: &mut Tensor<f32>,
    r_h: &mut Tensor<f32>,
    r_h_down: &mut Tensor<f32>,
    r_area: &mut Tensor<f32>,
    r_perim: &mut Tensor<f32>,
    r_radius: &mut Tensor<f32>,
    r_a_d: &mut Tensor<f32>,
    r_p_d: &mut Tensor<f32>,
    n: u32,
    dt_s: f32,
    th: f32,
    g: f32,
    f43: f32,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let gdt = g * dt_s;
        let f = friction[i];
        let q = qc[i];
        let a_e = e_area[i];
        let r_e = e_radius[i];
        let ad_e = e_a_d[i];
        let pd_e = e_p_d[i];
        let p_e = e_perim[i];
        let phi = o_phi[i];
        let tau = o_dt_phi[i];
        let den = o_den[i];
        let alpha = o_alpha[i];
        let beta = o_beta[i];
        let phi_d = o_phi_d[i];
        let dhe = o_dh_edge[i];
        let k1 = o_k1[i];
        let le = edge_length[i];
        let su = s_up[i];
        let sd = s_dn[i];
        let g_qs = b_q_star[i];
        let g_a = b_a_up[i];
        let g_b = b_b_down[i];

        // 3f
        let g_ra = g_a * f * ind(!(o_ra[i] < 0.0f32));
        let g_rb = g_b * f * ind(!(o_rb[i] < 0.0f32));
        let g_beta = fma(g_ra + g_rb, th, g_qs * dhe);
        let g_kappa = fma(g_ra, su, -(g_rb * sd));
        // 3e
        let g_alpha = g_qs * q;
        // 3d
        let g_k1 = g_kappa * f / den;
        let mut g_den = -g_kappa * f * k1 / (den * den);
        // 3b
        let mut g_a_e = g_beta * f * gdt / (le * den);
        g_den = g_den + (-g_beta * f * (a_e * gdt)) / ((le * den) * den);
        g_den = g_den + (-g_alpha * f * (tau + 1.0f32)) / (den * den);
        let mut g_tau = g_alpha * f / den;
        // 3d: K1 = (tau*Q)*phi_d + ((Ad*dhe)*g*dt)/Le
        g_tau = fma(g_k1 * q, phi_d, g_tau);
        // g_q starts as `g_qs * alpha` and g_dhe as `g_qs * beta`; the first `+=` of each fuses it.
        let mut g_q = fma(g_qs, alpha, (g_k1 * tau) * phi_d);
        let g_phi_d = g_k1 * tau * q;
        let mut g_ad_e = g_k1 * dhe * gdt / le;
        let g_dhe = fma(g_qs, beta, g_k1 * ad_e * gdt / le);
        // 3a: den = tau*2 + 1
        g_tau = fma(2.0f32, g_den, g_tau);
        // 3c: phi_d = Ad/A + (Ad/A - Pd/P)*4/3
        g_ad_e = g_ad_e + g_phi_d * (1.0f32 + f43) / a_e;
        g_a_e = g_a_e - g_phi_d * (1.0f32 + f43) * ad_e / (a_e * a_e);
        // The outputs start at 0 and add, as the host's `+=` on a zeroed Vec does (-0 becomes +0).
        let mut x_p_d = 0.0f32;
        x_p_d = x_p_d - g_phi_d * f43 / p_e;
        let mut x_perim = 0.0f32;
        x_perim = x_perim + g_phi_d * f43 * pd_e / (p_e * p_e);
        // 3a: tau = phi*dt; phi = ((n2*|Q|)*g)/(A*R^(4/3))
        let g_phi = g_tau * dt_s;
        let cc = g / (a_e * f32::powf(r_e, f43));
        let mut x_n2 = 0.0f32;
        x_n2 = fma(g_phi * f32::abs(q), cc, x_n2);
        g_q = fma((g_phi * n2[i]) * cc, sign0(q), g_q);
        g_a_e = g_a_e - g_phi * phi / a_e;
        let mut x_radius = 0.0f32;
        x_radius = x_radius - g_phi * phi * f43 / r_e;

        let mut x_qc = 0.0f32;
        x_qc = x_qc + g_q;
        let mut x_area = 0.0f32;
        x_area = x_area + g_a_e;
        let mut x_a_d = 0.0f32;
        x_a_d = x_a_d + g_ad_e;
        let mut x_h = 0.0f32;
        x_h = x_h + g_dhe;
        let mut x_h_down = 0.0f32;
        x_h_down = x_h_down - g_dhe;
        r_qc[i] = x_qc;
        r_n2[i] = x_n2;
        r_h[i] = x_h;
        r_h_down[i] = x_h_down;
        r_area[i] = x_area;
        r_perim[i] = x_perim;
        r_radius[i] = x_radius;
        r_a_d[i] = x_a_d;
        r_p_d[i] = x_p_d;
    }
}

/// Cotangents of the Newton block outputs, on the device (the twin of `NewtonBar`).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct NewtonBarDev {
    pub q_star: Cube,
    pub a_up: Cube,
    pub b_down: Cube,
}

impl NewtonBarDev {
    /// Copy `h` to the device of `like`. An empty host Vec becomes zeros.
    #[doc(hidden)]
    pub fn upload(h: &NewtonBar, like: &Cube) -> Self {
        let n = [h.q_star.len(), h.a_up.len(), h.b_down.len()].into_iter().max().unwrap_or(1).max(1);
        Self { q_star: up(like, &h.q_star, n), a_up: up(like, &h.a_up, n), b_down: up(like, &h.b_down, n) }
    }
}

/// The device twin of `NewtonGrads`. `edge.top_width` is a buffer of zeros.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct NewtonGradsDev {
    pub qc: Cube,
    pub n2: Cube,
    pub h: Cube,
    pub h_down: Cube,
    pub edge: SectionBarDev,
}

/// B3 on the device. Mirrors `newton_backward_host`, same argument order.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn newton_backward_dev(
    c: &StageOpConstsDev,
    dt_s: f32,
    h: &Cube,
    qc: &Cube,
    n2: &Cube,
    dep: &DepthDev,
    edge: &SectionDev,
    o: &NewtonDev,
    bar: &NewtonBarDev,
) -> NewtonGradsDev {
    let nn = c.host.n;
    let e = || empty_f32(h, nn);
    let out = NewtonGradsDev {
        qc: e(),
        n2: e(),
        h: e(),
        h_down: e(),
        edge: SectionBarDev { top_width: zeros_f32(h, nn), area: e(), perim: e(), radius: e(), a_d: e(), p_d: e() },
    };
    let (count, dim) = grid(nn);
    newton_backward_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(qc),
        a!(n2),
        a!(dep.s_up),
        a!(dep.s_dn),
        a!(edge.out_area),
        a!(edge.out_radius),
        a!(edge.out_a_d),
        a!(edge.out_p_d),
        a!(edge.out_perim),
        a!(c.friction),
        a!(c.edge_length),
        a!(o.phi),
        a!(o.dt_phi),
        a!(o.den),
        a!(o.alpha),
        a!(o.beta),
        a!(o.phi_d),
        a!(o.dh_edge),
        a!(o.k1),
        a!(o.ra),
        a!(o.rb),
        a!(bar.q_star),
        a!(bar.a_up),
        a!(bar.b_down),
        a!(out.qc),
        a!(out.n2),
        a!(out.h),
        a!(out.h_down),
        a!(out.edge.area),
        a!(out.edge.perim),
        a!(out.edge.radius),
        a!(out.edge.a_d),
        a!(out.edge.p_d),
        nn as u32,
        dt_s,
        c.host.theta,
        c.host.g,
        F43,
    );
    out
}

/// Test hook: copy a `NewtonGradsDev` to the host (bumps the host-read counter).
/// The host leaves `edge.top_width` empty.
#[doc(hidden)]
pub fn newton_grads_to_host(g: &NewtonGradsDev) -> NewtonGrads {
    NewtonGrads {
        qc: cube_to_host(&g.qc),
        n2: cube_to_host(&g.n2),
        h: cube_to_host(&g.h),
        h_down: cube_to_host(&g.h_down),
        edge: SectionBar {
            top_width: vec![],
            area: cube_to_host(&g.edge.area),
            perim: cube_to_host(&g.edge.perim),
            radius: cube_to_host(&g.edge.radius),
            a_d: cube_to_host(&g.edge.a_d),
            p_d: cube_to_host(&g.edge.p_d),
        },
    }
}

// ----------------------------------------------------------------- control

/// B4 per reach: the reverse of 4d to 4a. Mirrors `control_backward_host`.
/// `ex` and `act` are length-1 dummies without a table.
#[cube(launch)]
fn control_backward_kernel(
    p: &Tensor<f32>,
    q_eps: &Tensor<f32>,
    n_eff: &Tensor<f32>,
    depth: &Tensor<f32>,
    s_area: &Tensor<f32>,
    s_radius: &Tensor<f32>,
    s_a_d: &Tensor<f32>,
    s_p_d: &Tensor<f32>,
    s_perim: &Tensor<f32>,
    length: &Tensor<f32>,
    dam_t: &Tensor<f32>,
    t_mask: &Tensor<f32>,
    t_tw: &Tensor<f32>,
    s0_sqrt: &Tensor<f32>,
    dam: &Tensor<f32>,
    nd: &Tensor<f32>,
    c_rd_head: &Tensor<f32>,
    c_ex: &Tensor<f32>,
    c_act: &Tensor<f32>,
    c_kk: &Tensor<f32>,
    c_w: &Tensor<f32>,
    c_rpr: &Tensor<f32>,
    b_r_t: &Tensor<f32>,
    b_r_prime: &Tensor<f32>,
    r_h: &mut Tensor<f32>,
    r_depth: &mut Tensor<f32>,
    r_p: &mut Tensor<f32>,
    r_q_eps: &mut Tensor<f32>,
    r_n_eff: &mut Tensor<f32>,
    r_d_s: &mut Tensor<f32>,
    r_area: &mut Tensor<f32>,
    r_perim: &mut Tensor<f32>,
    r_radius: &mut Tensor<f32>,
    r_a_d: &mut Tensor<f32>,
    r_p_d: &mut Tensor<f32>,
    n: u32,
    dt_s: f32,
    f23: f32,
    f53: f32,
    #[comptime] has_table: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let g_rt = b_r_t[i];
        let g_rp = b_r_prime[i];
        let s0s = s0_sqrt[i];
        // 4d
        let g_rpr = g_rp * ind(!(c_rpr[i] < 0.0f32));
        let g_rpd = g_rpr * dam[i];
        let g_rpnd = g_rpr * nd[i];
        let mut g_rd = g_rt * dam[i];
        let g_rnd = g_rt * nd[i];
        // 4c
        let a_s = s_area[i];
        let r_s = s_radius[i];
        let ad_s = s_a_d[i];
        let pd_s = s_p_d[i];
        let p_s = s_perim[i];
        let kk = c_kk[i];
        let w = c_w[i];
        let g_dk = g_rpnd * s0s;
        let g_kk = fma(g_rnd, s0s, g_dk * w);
        let g_w = g_dk * kk;
        // The outputs start at 0 and add, as the host's `+=` on a zeroed Vec does (-0 becomes +0).
        let mut x_a_d = 0.0f32;
        x_a_d = x_a_d + g_w * f53 / a_s;
        let mut x_area = 0.0f32;
        x_area = x_area - g_w * f53 * ad_s / (a_s * a_s);
        let mut x_p_d = 0.0f32;
        x_p_d = x_p_d - g_w * f23 / p_s;
        let mut x_perim = 0.0f32;
        x_perim = x_perim + g_w * f23 * pd_s / (p_s * p_s);
        x_area = x_area + g_kk * f32::powf(r_s, f23) / n_eff[i];
        let mut x_radius = 0.0f32;
        x_radius = fma(g_kk * (a_s / n_eff[i]) * f23, f32::powf(r_s, f23 - 1.0f32), x_radius);
        let mut x_n_eff = 0.0f32;
        x_n_eff = x_n_eff - g_kk * kk / n_eff[i];
        // 4b
        let mut g_rpd_head = g_rpd;
        let mut x_h = 0.0f32;
        if has_table {
            let m = t_mask[i];
            let keep = keep_of(m);
            let g_rtb = g_rd * m;
            let g_rptb = g_rpd * m;
            g_rd = g_rd * keep;
            g_rpd_head = fma(g_rpd, keep, g_rptb * c_act[i]);
            x_h = fma(g_rtb * ((t_tw[i] * length[i]) / dam_t[i]), ind(!(c_ex[i] < 0.0f32)), x_h);
        }
        // 4a
        let mut x_d_s = 0.0f32;
        x_d_s = x_d_s + g_rpd_head * dt_s / dam_t[i];
        let e1 = q_eps[i] + 1.0f32;
        let u = f32::powf(depth[i], e1);
        let lp = (length[i] * p[i]) / (e1 * dam_t[i]);
        let mut x_p = 0.0f32;
        x_p = x_p + g_rd * (length[i] * u) / (e1 * dam_t[i]);
        let mut x_depth = 0.0f32;
        x_depth = fma((g_rd * lp) * e1, f32::powf(depth[i], e1 - 1.0f32), x_depth);
        let x = fma(lp * u, f32::ln(depth[i]), -(c_rd_head[i] / e1));
        let mut x_q_eps = 0.0f32;
        x_q_eps = fma(g_rd, x, x_q_eps);
        r_h[i] = x_h;
        r_depth[i] = x_depth;
        r_p[i] = x_p;
        r_q_eps[i] = x_q_eps;
        r_n_eff[i] = x_n_eff;
        r_d_s[i] = x_d_s;
        r_area[i] = x_area;
        r_perim[i] = x_perim;
        r_radius[i] = x_radius;
        r_a_d[i] = x_a_d;
        r_p_d[i] = x_p_d;
    }
}

/// Cotangents of `R` and the clamped `R'`, on the device (the twin of `ControlBar`).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ControlBarDev {
    pub r_t: Cube,
    pub r_prime: Cube,
}

impl ControlBarDev {
    /// Copy `h` to the device of `like`. An empty host Vec becomes zeros.
    #[doc(hidden)]
    pub fn upload(h: &ControlBar, like: &Cube) -> Self {
        let n = [h.r_t.len(), h.r_prime.len()].into_iter().max().unwrap_or(1).max(1);
        Self { r_t: up(like, &h.r_t, n), r_prime: up(like, &h.r_prime, n) }
    }
}

/// The device twin of `ControlGrads`. `store.top_width` is a buffer of zeros;
/// the glue replaces it with `g_ds * L / dt_s`.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ControlGradsDev {
    pub h: Cube,
    pub depth: Cube,
    pub p: Cube,
    pub q_eps: Cube,
    pub n_eff: Cube,
    pub d_s: Cube,
    pub store: SectionBarDev,
}

/// B4 on the device. Mirrors `control_backward_host`, same argument order.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn control_backward_dev(
    c: &StageOpConstsDev,
    dt_s: f32,
    h: &Cube,
    p: &Cube,
    q_eps: &Cube,
    n_eff: &Cube,
    depth: &Cube,
    store: &SectionDev,
    d_s: &Cube,
    o: &ControlDev,
    bar: &ControlBarDev,
) -> ControlGradsDev {
    // `h` and `d_s` are unused: the host fn ignores them too (the table pool stage is in `ex`).
    let _ = d_s;
    let nn = c.host.n;
    let e = || empty_f32(h, nn);
    let out = ControlGradsDev {
        h: e(),
        depth: e(),
        p: e(),
        q_eps: e(),
        n_eff: e(),
        d_s: e(),
        store: SectionBarDev { top_width: zeros_f32(h, nn), area: e(), perim: e(), radius: e(), a_d: e(), p_d: e() },
    };
    let (count, dim) = grid(nn);
    control_backward_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(p),
        a!(q_eps),
        a!(n_eff),
        a!(depth),
        a!(store.out_area),
        a!(store.out_radius),
        a!(store.out_a_d),
        a!(store.out_p_d),
        a!(store.out_perim),
        a!(c.length),
        a!(c.dam_t),
        a!(c.t_mask),
        a!(c.t_tw),
        a!(c.s0_sqrt),
        a!(c.dam),
        a!(c.nd),
        a!(o.rd_head),
        a!(o.ex),
        a!(o.act),
        a!(o.kk),
        a!(o.w),
        a!(o.rpr),
        a!(bar.r_t),
        a!(bar.r_prime),
        a!(out.h),
        a!(out.depth),
        a!(out.p),
        a!(out.q_eps),
        a!(out.n_eff),
        a!(out.d_s),
        a!(out.store.area),
        a!(out.store.perim),
        a!(out.store.radius),
        a!(out.store.a_d),
        a!(out.store.p_d),
        nn as u32,
        dt_s,
        F23,
        F53,
        c.host.has_table,
    );
    out
}

/// Test hook: copy a `ControlGradsDev` to the host (bumps the host-read counter).
/// The host leaves `store.top_width` empty.
#[doc(hidden)]
pub fn control_grads_to_host(g: &ControlGradsDev) -> ControlGrads {
    ControlGrads {
        h: cube_to_host(&g.h),
        depth: cube_to_host(&g.depth),
        p: cube_to_host(&g.p),
        q_eps: cube_to_host(&g.q_eps),
        n_eff: cube_to_host(&g.n_eff),
        d_s: cube_to_host(&g.d_s),
        store: SectionBar {
            top_width: vec![],
            area: cube_to_host(&g.store.area),
            perim: cube_to_host(&g.store.perim),
            radius: cube_to_host(&g.store.radius),
            a_d: cube_to_host(&g.store.a_d),
            p_d: cube_to_host(&g.store.p_d),
        },
    }
}

// ------------------------------------------------------------------- depth

/// B1 per reach, first loop of `depth_backward_host`. `o_g_down` is the
/// cotangent of the gathered `h[down]` (after the PrescribedStage mask); the
/// next kernel adds it into `o_gh`.
#[cube(launch)]
fn depth_backward_kernel(
    h: &Tensor<f32>,
    h_down: &Tensor<f32>,
    der: &Tensor<f32>,
    x: &Tensor<f32>,
    outlet: &Tensor<f32>,
    g_depth: &Tensor<f32>,
    g_d_edge: &Tensor<f32>,
    g_h_down: &Tensor<f32>,
    o_gh: &mut Tensor<f32>,
    o_g_down: &mut Tensor<f32>,
    n: u32,
    lb: f32,
    #[comptime] prescribed: bool,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        let hd = h_down[i];
        let g_der = g_d_edge[i] * ind(!(der[i] < lb));
        let lt = h[i] < hd;
        // gh and ghd start at 0 and add, as the host's `+=` on a zeroed Vec does (-0 becomes +0).
        let mut gh = 0.0f32;
        gh = fma(g_der, ind(!lt), gh);
        let mut ghd = 0.0f32;
        ghd = ghd + fma(g_der, ind(lt), g_h_down[i]);
        gh = fma(g_depth[i], ind(!(x[i] < lb)), gh);
        o_gh[i] = gh;
        if prescribed {
            o_g_down[i] = ghd * keep_of(outlet[i]);
        } else {
            o_g_down[i] = ghd;
        }
    }
}

/// B1 on the device: `g_h` of blocks 1a to 1e. Mirrors `depth_backward_host`.
/// `g_h_down` is the cotangent of the `h_down` output (after the
/// PrescribedStage mix), which Newton fills with `-g_dhe`.
#[doc(hidden)]
pub fn depth_backward_dev(
    c: &StageOpConstsDev,
    h: &Cube,
    dep: &DepthDev,
    g_depth: &Cube,
    g_d_edge: &Cube,
    g_h_down: &Cube,
) -> Cube {
    let nn = c.host.n;
    let (gh, g_down) = (empty_f32(h, nn), empty_f32(h, nn));
    let (count, dim) = grid(nn);
    depth_backward_kernel::launch::<CudaRuntime>(
        &h.client,
        count.clone(),
        dim.clone(),
        a!(h),
        a!(dep.h_down),
        a!(dep.der),
        a!(dep.x),
        a!(c.outlet),
        a!(g_depth),
        a!(g_d_edge),
        a!(g_h_down),
        a!(gh),
        a!(g_down),
        nn as u32,
        c.host.depth_lb,
        c.host.prescribed,
    );
    // The gather's transpose, as in the readout: `gh[down[i]] += g_down[i]`.
    down_transpose_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(g_down),
        a!(c.tree.crow),
        a!(c.tree.col),
        a!(c.tree.down_slot),
        a!(gh),
        nn as u32,
    );
    gh
}

// ---------------------------------------------------------------- prologue

/// BH2 and BH3, per reach. Mirrors `prologue_backward_host`. The scatter over
/// the slots (`g_q_t[col] += g_i_t[row] * adj_mc`) is a gather over the two
/// slots of column `c`: its diagonal slot, then its `down_slot`.
#[cube(launch)]
fn prologue_backward_kernel(
    a_i_t: &Tensor<f32>,
    a_i_delta: &Tensor<f32>,
    a_n2: &Tensor<f32>,
    a_n_eff: &Tensor<f32>,
    a_qp: &Tensor<f32>,
    hour_n_eff: &Tensor<f32>,
    g_q0: &Tensor<f32>,
    g_qmc: &Tensor<f32>,
    crow: &Tensor<i32>,
    down: &Tensor<i32>,
    down_slot: &Tensor<i32>,
    adj_mc: &Tensor<f32>,
    in_stage: &Tensor<f32>,
    t_mask: &Tensor<f32>,
    o_n: &mut Tensor<f32>,
    o_q_t: &mut Tensor<f32>,
    o_q_next_mc: &mut Tensor<f32>,
    o_q_prime: &mut Tensor<f32>,
    n: u32,
    #[comptime] has_table: bool,
) {
    let c = ABSOLUTE_POS;
    if c < usize::cast_from(n) {
        // BH2: g_i_t = i_t - i_delta, per row.
        let mut gq = g_q0[c];
        let mut gm = g_qmc[c];
        let diag = usize::cast_from(crow[c + 1]) - 1;
        gq = fma(a_i_t[c] - a_i_delta[c], adj_mc[diag], gq);
        gm = fma(a_i_delta[c], adj_mc[diag], gm);
        let ds = down_slot[c];
        if ds >= 0 {
            let r = usize::cast_from(down[c]);
            gq = fma(a_i_t[r] - a_i_delta[r], adj_mc[usize::cast_from(ds)], gq);
            gm = fma(a_i_delta[r], adj_mc[usize::cast_from(ds)], gm);
        }
        o_q_t[c] = gq;
        o_q_next_mc[c] = gm;
        // BH3
        o_q_prime[c] = a_qp[c] * in_stage[c];
        // The last multiply fuses; `2.0 * n_eff` is exact.
        let mut g_n_eff = fma(2.0f32 * hour_n_eff[c], a_n2[c], a_n_eff[c]);
        if has_table {
            g_n_eff = g_n_eff * keep_of(t_mask[c]);
        }
        o_n[c] = g_n_eff;
    }
}

/// The device twin of `HourGrads`.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct HourGradsDev {
    pub n: Cube,
    pub q: Cube,
    pub p: Cube,
    pub h_t: Cube,
    pub q_t: Cube,
    pub q_next_mc: Cube,
    pub q_prime: Cube,
}

/// Hour accumulators on the device, summed over substeps (the twin of `HourAcc`).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct HourAccDev {
    pub i_t: Cube,
    pub i_delta: Cube,
    pub n2: Cube,
    pub n_eff: Cube,
    pub p: Cube,
    pub q_eps: Cube,
    pub qp: Cube,
}

impl HourAccDev {
    /// All seven accumulators zero, length `n`, on the device of `like`.
    #[doc(hidden)]
    pub fn zeros(like: &Cube, n: usize) -> Self {
        let z = || zeros_f32(like, n);
        Self { i_t: z(), i_delta: z(), n2: z(), n_eff: z(), p: z(), q_eps: z(), qp: z() }
    }

    /// Copy `h` to the device of `like`. An empty host Vec becomes zeros.
    #[doc(hidden)]
    pub fn upload(h: &HourAcc, like: &Cube) -> Self {
        let n = [h.i_t.len(), h.i_delta.len(), h.n2.len(), h.n_eff.len(), h.p.len(), h.q_eps.len(), h.qp.len()]
            .into_iter()
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            i_t: up(like, &h.i_t, n),
            i_delta: up(like, &h.i_delta, n),
            n2: up(like, &h.n2, n),
            n_eff: up(like, &h.n_eff, n),
            p: up(like, &h.p, n),
            q_eps: up(like, &h.q_eps, n),
            qp: up(like, &h.qp, n),
        }
    }
}

/// BH2 and BH3 on the device. Mirrors `prologue_backward_host`. `q`, `p` and
/// `h_t` are the accumulator and cotangent buffers themselves (shared, not copied).
#[doc(hidden)]
pub fn prologue_backward_dev(
    c: &StageOpConstsDev,
    hour: &HourDev,
    acc: &HourAccDev,
    g_h0: &Cube,
    g_q0: &Cube,
    g_qmc_direct: &Cube,
) -> HourGradsDev {
    let nn = c.host.n;
    let e = || empty_f32(g_q0, nn);
    let (o_n, o_q_t, o_q_next_mc, o_q_prime) = (e(), e(), e(), e());
    let (count, dim) = grid(nn);
    prologue_backward_kernel::launch::<CudaRuntime>(
        &g_q0.client,
        count,
        dim,
        a!(acc.i_t),
        a!(acc.i_delta),
        a!(acc.n2),
        a!(acc.n_eff),
        a!(acc.qp),
        a!(hour.n_eff),
        a!(g_q0),
        a!(g_qmc_direct),
        a!(c.tree.crow),
        a!(c.tree.down),
        a!(c.tree.down_slot),
        a!(c.adj_mc),
        a!(c.in_stage),
        a!(c.t_mask),
        a!(o_n),
        a!(o_q_t),
        a!(o_q_next_mc),
        a!(o_q_prime),
        nn as u32,
        c.host.has_table,
    );
    HourGradsDev {
        n: o_n,
        q: acc.q_eps.clone(),
        p: acc.p.clone(),
        h_t: g_h0.clone(),
        q_t: o_q_t,
        q_next_mc: o_q_next_mc,
        q_prime: o_q_prime,
    }
}

/// Test hook: copy a `HourAccDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn hour_acc_to_host(a: &HourAccDev) -> HourAcc {
    HourAcc {
        i_t: cube_to_host(&a.i_t),
        i_delta: cube_to_host(&a.i_delta),
        n2: cube_to_host(&a.n2),
        n_eff: cube_to_host(&a.n_eff),
        p: cube_to_host(&a.p),
        q_eps: cube_to_host(&a.q_eps),
        qp: cube_to_host(&a.qp),
    }
}

/// Test hook: copy a `HourGradsDev` to the host (bumps the host-read counter).
#[doc(hidden)]
pub fn hour_grads_to_host(g: &HourGradsDev) -> HourGrads {
    HourGrads {
        n: cube_to_host(&g.n),
        q: cube_to_host(&g.q),
        p: cube_to_host(&g.p),
        h_t: cube_to_host(&g.h_t),
        q_t: cube_to_host(&g.q_t),
        q_next_mc: cube_to_host(&g.q_next_mc),
        q_prime: cube_to_host(&g.q_prime),
    }
}

// ------------------------------------------------------------------- glue

/// Elementwise `a + b` on the device (a backend op on primitives).
fn add(a: &Cube, b: &Cube) -> Cube {
    into_cube::<Gpu>(<Gpu as FloatTensorOps<Gpu>>::float_add(from_cube::<Gpu>(a.clone()), from_cube::<Gpu>(b.clone())))
}

/// The storage top-width cotangent `(g_mat_d_s + g_ctl_d_s) * L / dt_s`. Mirrors
/// `store_bar.top_width` of `assemble_backward_host`.
#[cube(launch)]
fn store_top_width_kernel(
    g_mat_d_s: &Tensor<f32>,
    g_ctl_d_s: &Tensor<f32>,
    length: &Tensor<f32>,
    o_top_width: &mut Tensor<f32>,
    n: u32,
    dt_s: f32,
) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        o_top_width[i] = ((g_mat_d_s[i] + g_ctl_d_s[i]) * length[i]) / dt_s;
    }
}

/// `acc_i_delta += i_mid * w`, as one fused multiply-add. Mirrors the
/// `acc.i_delta` loop of `assemble_backward_host`. Writes a new buffer.
#[cube(launch)]
fn i_delta_acc_kernel(i_mid: &Tensor<f32>, acc: &Tensor<f32>, o_acc: &mut Tensor<f32>, n: u32, w: f32) {
    let i = ABSOLUTE_POS;
    if i < usize::cast_from(n) {
        o_acc[i] = fma(i_mid[i], w, acc[i]);
    }
}

/// B6 to B1 for substep `s`, on the device. Mirrors `assemble_backward_host`
/// step for step, including every `add2` and `add_into`. Adds into `acc` and
/// returns `(g_h, g_q)` of the substep's start state. The sums of two vectors
/// are backend adds on primitives; fusing them waits for a profile.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn assemble_backward_dev(
    c: &StageOpConstsDev,
    hour: &HourDev,
    s: usize,
    h: &Cube,
    qc: &Cube,
    asm: &AssembledDev,
    rg: &ReadoutGradsDev,
    g_r: &Cube,
    g_a_values: &Cube,
    acc: &mut HourAccDev,
) -> (Cube, Cube) {
    let nn = c.host.n;
    let subs = c.host.substeps.max(1) as usize;
    let (f0, f1) = (s as f32 / subs as f32, (s + 1) as f32 / subs as f32);
    // B6, B5
    let rhs = rhs_backward_dev(c, asm, g_r);
    let mat = matrix_backward_dev(c, g_a_values);
    // B4: cotangents of R and R' from the readout, the RHS and the matrix
    let ctl_bar = ControlBarDev { r_t: add(&rg.r_t, &rhs.r_t), r_prime: add(&rg.r_prime, &mat.r_prime) };
    let ctl = control_backward_dev(
        c,
        asm.dt_s,
        h,
        &hour.p,
        &hour.q_eps,
        &hour.n_eff,
        &asm.depth.depth,
        &asm.store,
        &asm.d_s,
        &asm.control,
        &ctl_bar,
    );
    // B3
    let nw_bar = NewtonBarDev {
        q_star: add(&rg.q_star, &rhs.q_star),
        a_up: add(&rg.a_up, &mat.a_up),
        b_down: add(&add(&rg.b_down, &rhs.b_down), &mat.b_down),
    };
    let nw = newton_backward_dev(c, asm.dt_s, h, qc, &hour.n2, &asm.depth, &asm.edge, &asm.newton, &nw_bar);
    // B2: the storage top width carries g_ds * L / dt_s (the storage path)
    let top_width = empty_f32(h, nn);
    let (count, dim) = grid(nn);
    store_top_width_kernel::launch::<CudaRuntime>(
        &h.client,
        count.clone(),
        dim.clone(),
        a!(mat.d_s),
        a!(ctl.d_s),
        a!(c.length),
        a!(top_width),
        nn as u32,
        asm.dt_s,
    );
    let store_bar = SectionBarDev { top_width, ..ctl.store.clone() };
    let sec_e = section_backward_dev(c, &hour.p, &hour.q_eps, &asm.depth.d_edge, &asm.edge, &nw.edge);
    let sec_s = section_backward_dev(c, &hour.p, &hour.q_eps, &asm.depth.depth, &asm.store, &store_bar);
    // B1
    let g_depth = add(&ctl.depth, &sec_s.d);
    let gh_depth = depth_backward_dev(c, h, &asm.depth, &g_depth, &sec_e.d, &nw.h_down);
    let g_h = add(&add(&add(&rg.h, &ctl.h), &nw.h), &gh_depth);
    let g_q = add(&rhs.qc, &nw.qc);
    // hour accumulators (BH1 weights i_delta)
    acc.n2 = add(&acc.n2, &nw.n2);
    acc.n_eff = add(&acc.n_eff, &ctl.n_eff);
    acc.p = add(&add(&add(&acc.p, &sec_e.p), &sec_s.p), &ctl.p);
    acc.q_eps = add(&add(&add(&acc.q_eps, &sec_e.q_eps), &sec_s.q_eps), &ctl.q_eps);
    acc.qp = add(&acc.qp, &rhs.qp);
    acc.i_t = add(&acc.i_t, &rhs.i_mid);
    let w = 0.5 * (f0 + f1);
    let i_delta = empty_f32(h, nn);
    i_delta_acc_kernel::launch::<CudaRuntime>(
        &h.client,
        count,
        dim,
        a!(rhs.i_mid),
        a!(acc.i_delta),
        a!(i_delta),
        nn as u32,
        w,
    );
    acc.i_delta = i_delta;
    (g_h, g_q)
}

impl ReadoutGradsDev {
    /// Copy `h` to the device of `like`. An empty host Vec becomes zeros.
    #[doc(hidden)]
    pub fn upload(h: &ReadoutGrads, like: &Cube) -> Self {
        let n = [h.dh.len(), h.h.len(), h.q_star.len(), h.a_up.len(), h.b_down.len(), h.r_t.len(), h.r_prime.len()]
            .into_iter()
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            dh: up(like, &h.dh, n),
            h: up(like, &h.h, n),
            q_star: up(like, &h.q_star, n),
            a_up: up(like, &h.a_up, n),
            b_down: up(like, &h.b_down, n),
            r_t: up(like, &h.r_t, n),
            r_prime: up(like, &h.r_prime, n),
        }
    }
}
