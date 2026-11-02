//! Backward kernels of the stage block: the output, readout, RHS and matrix
//! adjoints. Each kernel body is the loop body of the host adjoint fn in
//! `src/routing/stage_op.rs` (`output_backward_host`, `readout_backward_host`,
//! `rhs_backward_host`, `matrix_backward_host`), line for line.
//!
//! Rules (see the tie table in the `stage_op.rs` module doc):
//! - every add that takes a product is one explicit `fma`, at the same site
//!   and in the same operand order as the `f32::mul_add` of the host fn. A
//!   product with several uses stays a plain multiply on both sides;
//! - the mask of the floor uses the host comparison, `!(hs < hf)`;
//! - a scatter of the host (`x[down[i]] += ...`, `x[col[k]] += ...`) is a
//!   gather here, in the host's slot order. In a tree every reach has one
//!   downstream reach, so a column has at most two slots: its diagonal slot
//!   and its `down_slot`, in that order.
//!
//! Nothing here reads a device buffer to the host except the `*_to_host` test hooks.

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

use super::forward::AssembledDev;
use super::StageOpConstsDev;
use crate::routing::stage_op::{MatrixGrads, ReadoutGrads, RhsGrads};
use crate::sparse::cube_buf::{cube_to_host, empty_f32, grid, Cube};

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
        let mut g_diag = 0.0f32;
        for k in s0..s1 {
            g_diag = fma(g_a[k], diag_slot[k], g_diag);
        }
        // `g_dst` and `g_dst * th` each have two uses and stay plain multiplies.
        let g_dst = g_diag * in_stage[i];
        let gs = g_dst * th;
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
