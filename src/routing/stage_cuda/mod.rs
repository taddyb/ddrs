//! The stage block on CUDA: device copies of the constants and of the hour
//! structs, and (in `forward`) one kernel per host forward block.
//!
//! Every device struct has the field names of its host twin in
//! `src/routing/stage_op.rs`. Each field is a full-length f32 buffer. An
//! `upload` turns an empty host Vec into zeros (the table-only fields into a
//! buffer of length 1), so the kernels never meet an empty buffer.

use std::sync::Arc;

use burn::tensor::backend::Backend;

use crate::routing::stage_op::{ControlOut, DepthOut, HourHost, NewtonOut, SectionHost, StageOpConsts};
use crate::sparse::cube_buf::{cube_to_host, from_cube, into_cube, upload_f32, upload_i32, zeros_f32, Cube};
use crate::sparse::tree_lu::{tree_levels, TreeLevels};
use crate::sparse::tree_lu_cuda::TreeDev;
use crate::sparse::CsrPattern;

pub mod backward;
pub mod forward;

use forward::{stage_hour_forward_dev, stage_prologue_dev, HourForwardDev, SubstepDev};

/// `v` on the device, or zeros of length `len` when `v` is empty.
fn up(like: &Cube, v: &[f32], len: usize) -> Cube {
    if v.is_empty() {
        zeros_f32(like, len.max(1))
    } else {
        upload_f32(like, v)
    }
}

/// Device copy of `StageOpConsts`, uploaded from the host copy (same bits).
/// The scalars are read from `host`.
#[doc(hidden)]
#[derive(Debug)]
pub struct StageOpConstsDev {
    pub host: Arc<StageOpConsts>,
    pub tree: TreeDev,
    /// `StageOpConsts::down` (i32): the reach itself at an outlet, where
    /// `tree.down` is -1.
    pub down: Cube,
    pub in_stage: Cube,
    pub friction: Cube,
    pub dam: Cube,
    pub ctrl: Cube,
    pub nd: Cube,
    pub w_ctrl: Cube,
    pub w_edge: Cube,
    pub has_down: Cube,
    pub outlet: Cube,
    pub z_mid: Cube,
    pub z_down: Cube,
    pub length: Cube,
    pub edge_length: Cube,
    pub dam_t: Cube,
    pub s0_sqrt: Cube,
    /// Length 1 dummies without a table.
    pub t_mask: Cube,
    pub t_tw: Cube,
    pub t_n: Cube,
    pub t_pool: Cube,
    pub diag_slot: Cube,
    pub adj_stage: Cube,
    pub adj_mc: Cube,
    pub adj_friction: Cube,
    pub dam_slot: Cube,
}

/// One-time host check that the device gathers and the host loops read the same
/// network. The backward kernels gather over the pattern structure (`tree.crow`,
/// `tree.col`, `tree.down`, `tree.down_slot`), while the host adjoint loops and
/// `readout_backward_kernel` use `StageOpConsts::down`, `row` and `col`. Nothing else
/// ties the two together, so a mismatch would give silently wrong gradients.
/// Reads host vectors only. Panics with the first mismatch.
fn check_tree_matches_consts(h: &StageOpConsts, tree: &TreeLevels) {
    let n = h.n;
    assert_eq!(tree.n, n, "StageOpConstsDev::upload: tree has {} reaches but the consts have {n}", tree.n);
    assert_eq!(h.down.len(), n, "StageOpConstsDev::upload: down has length {} but n is {n}", h.down.len());
    for i in 0..n {
        let (s0, s1) = (tree.crow[i] as usize, tree.crow[i + 1] as usize);
        if tree.down[i] >= 0 {
            assert!(
                h.down[i] == tree.down[i] as usize && tree.down_slot[i] >= 0,
                "StageOpConstsDev::upload: reach {i} has down {} in the consts but {} in the tree",
                h.down[i],
                tree.down[i]
            );
        } else {
            assert!(
                h.down[i] == i && tree.down_slot[i] < 0,
                "StageOpConstsDev::upload: reach {i} is an outlet in the tree but has down {} in the consts",
                h.down[i]
            );
        }
        // The diagonal is the last slot of the row; the other columns ascend and lie below it.
        assert!(
            s1 > s0 && tree.col[s1 - 1] as usize == i,
            "StageOpConstsDev::upload: row {i} does not end with its diagonal slot"
        );
        for s in s0..s1 - 1 {
            let cs = tree.col[s] as usize;
            assert!(
                cs < i && (s + 1 == s1 - 1 || cs < tree.col[s + 1] as usize),
                "StageOpConstsDev::upload: row {i}: off-diagonal columns are not ascending below the diagonal"
            );
        }
    }
    assert!(
        h.row.len() == tree.nnz && h.col.len() == tree.nnz,
        "StageOpConstsDev::upload: row/col have length {}/{} but nnz is {}",
        h.row.len(),
        h.col.len(),
        tree.nnz
    );
    for k in 0..tree.nnz {
        assert!(
            h.row[k] == tree.row[k] as usize && h.col[k] == tree.col[k] as usize,
            "StageOpConstsDev::upload: slot {k} is ({}, {}) in the consts but ({}, {}) in the tree",
            h.row[k],
            h.col[k],
            tree.row[k],
            tree.col[k]
        );
    }
}

impl StageOpConstsDev {
    /// Copy every array of `host` to the device of `like`.
    pub fn upload(host: &Arc<StageOpConsts>, pattern: &CsrPattern, like: &Cube) -> Self {
        let h = host.as_ref();
        let (n, nnz) = (h.n, h.nnz);
        if h.has_table {
            for (name, v) in [("t_mask", &h.t_mask), ("t_tw", &h.t_tw), ("t_n", &h.t_n), ("t_pool", &h.t_pool)] {
                assert!(v.len() == n, "StageOpConstsDev::upload: {name} has length {} but n is {n}", v.len());
            }
        }
        let tree = tree_levels(pattern).expect("stage pattern is a tree");
        check_tree_matches_consts(h, &tree);
        let down: Vec<i32> = h.down.iter().map(|&d| d as i32).collect();
        Self {
            host: host.clone(),
            tree: TreeDev::upload(&tree, like),
            down: upload_i32(like, &down),
            in_stage: up(like, &h.in_stage, n),
            friction: up(like, &h.friction, n),
            dam: up(like, &h.dam, n),
            ctrl: up(like, &h.ctrl, n),
            nd: up(like, &h.nd, n),
            w_ctrl: up(like, &h.w_ctrl, n),
            w_edge: up(like, &h.w_edge, n),
            has_down: up(like, &h.has_down, n),
            outlet: up(like, &h.outlet, n),
            z_mid: up(like, &h.z_mid, n),
            z_down: up(like, &h.z_down, n),
            length: up(like, &h.length, n),
            edge_length: up(like, &h.edge_length, n),
            dam_t: up(like, &h.dam_t, n),
            s0_sqrt: up(like, &h.s0_sqrt, n),
            t_mask: up(like, &h.t_mask, 1),
            t_tw: up(like, &h.t_tw, 1),
            t_n: up(like, &h.t_n, 1),
            t_pool: up(like, &h.t_pool, 1),
            diag_slot: up(like, &h.diag_slot, nnz),
            adj_stage: up(like, &h.adj_stage, nnz),
            adj_mc: up(like, &h.adj_mc, nnz),
            adj_friction: up(like, &h.adj_friction, nnz),
            dam_slot: up(like, &h.dam_slot, nnz),
        }
    }
}

/// A device twin of a host struct whose fields are all `Vec<f32>`.
/// `full` fields have length `n`; `dummy` fields have length 1 when the host
/// Vec is empty (a table-only field without a table).
macro_rules! dev_struct {
    ($(#[$m:meta])* $name:ident, $host:ident, full [$($f:ident),* $(,)?], dummy [$($d:ident),* $(,)?]) => {
        $(#[$m])*
        #[doc(hidden)]
        #[derive(Clone, Debug)]
        pub struct $name {
            $(pub $f: Cube,)*
            $(pub $d: Cube,)*
        }

        impl $name {
            /// Copy `h` to the device of `like`. An empty host Vec becomes zeros.
            #[doc(hidden)]
            pub fn upload(h: &$host, like: &Cube) -> Self {
                let n = [$(h.$f.len()),*].into_iter().max().unwrap_or(1).max(1);
                Self {
                    $($f: up(like, &h.$f, n),)*
                    $($d: up(like, &h.$d, 1),)*
                }
            }

            /// Test hook: copy every field back to the host (one host read each).
            #[doc(hidden)]
            pub fn download(&self) -> $host {
                $host {
                    $($f: cube_to_host(&self.$f),)*
                    $($d: cube_to_host(&self.$d),)*
                }
            }
        }
    };
}

dev_struct!(
    /// Blocks 1a to 1e.
    DepthDev, DepthOut,
    full [x, depth, h_down, der, d_edge, s_up, s_dn],
    dummy []
);

dev_struct!(
    /// Blocks 2a to 2g.
    SectionDev, SectionHost,
    full [
        pw, tw, ss_raw, ss, bw_raw, bw, area, root, perim, radius, tw_d, ss_d, bw_d, a_d, p_d,
        out_tw, out_area, out_perim, out_radius, out_a_d, out_p_d
    ],
    dummy []
);

dev_struct!(
    /// Blocks 3a to 3f.
    NewtonDev, NewtonOut,
    full [phi, dt_phi, den, alpha, beta, phi_d, dh_edge, k1, kappa, q_star, ra, rb, a_up, b_down],
    dummy []
);

dev_struct!(
    /// Blocks 4a to 4d. `ex` and `act` are length 1 dummies without a table.
    ControlDev, ControlOut,
    full [rd_head, rpd_head, rd, rpd, kk, w, dk_dd, r_t, rpr, r_prime],
    dummy [ex, act]
);

/// The hour prologue (H0 to H3) on the device.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct HourDev {
    pub p: Cube,
    pub q_eps: Cube,
    pub n_eff: Cube,
    pub n2: Cube,
    pub qp_stage: Cube,
    pub i_t: Cube,
    pub i_delta: Cube,
    pub q_next_mc: Cube,
    pub b0: f32,
    pub b1: f32,
}

impl HourDev {
    /// Copy `h` to the device of `like`. An empty host Vec becomes zeros.
    #[doc(hidden)]
    pub fn upload(h: &HourHost, like: &Cube) -> Self {
        let n = [
            h.p.len(),
            h.q_eps.len(),
            h.n_eff.len(),
            h.n2.len(),
            h.qp_stage.len(),
            h.i_t.len(),
            h.i_delta.len(),
            h.q_next_mc.len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(1)
        .max(1);
        Self {
            p: up(like, &h.p, n),
            q_eps: up(like, &h.q_eps, n),
            n_eff: up(like, &h.n_eff, n),
            n2: up(like, &h.n2, n),
            qp_stage: up(like, &h.qp_stage, n),
            i_t: up(like, &h.i_t, n),
            i_delta: up(like, &h.i_delta, n),
            q_next_mc: up(like, &h.q_next_mc, n),
            b0: h.b0,
            b1: h.b1,
        }
    }

    /// Test hook: copy every field back to the host.
    #[doc(hidden)]
    pub fn download(&self) -> HourHost {
        HourHost {
            p: cube_to_host(&self.p),
            q_eps: cube_to_host(&self.q_eps),
            n_eff: cube_to_host(&self.n_eff),
            n2: cube_to_host(&self.n2),
            qp_stage: cube_to_host(&self.qp_stage),
            i_t: cube_to_host(&self.i_t),
            i_delta: cube_to_host(&self.i_delta),
            q_next_mc: cube_to_host(&self.q_next_mc),
            b0: self.b0,
            b1: self.b1,
        }
    }
}

/// What the device backward needs from the forward (Task 8 reads it).
#[derive(Clone, Debug)]
pub(crate) struct HourSavedDev {
    pub consts: Arc<StageOpConstsDev>,
    pub hour: HourDev,
    pub subs: Vec<SubstepDev>,
}

/// The op's device arm: converts the seven parents with `into_cube`, runs the
/// hour on the device and returns the `[2n]` output primitive. Nothing here
/// reads a device buffer to the host. When `save` is set it also returns the
/// state the backward needs, and `HourForwardDev::subs` is left empty.
/// Parents: `[n, q_spatial, p_spatial, h_t, q_t, q_next_mc, q_prime_t]`.
pub(crate) fn stage_timestep_cuda<I: Backend + 'static>(
    consts_dev: &Arc<StageOpConstsDev>,
    pattern: &CsrPattern,
    parents: [I::FloatTensorPrimitive; 7],
    b0: f32,
    b1: f32,
    save: bool,
) -> (I::FloatTensorPrimitive, HourForwardDev, Option<HourSavedDev>) {
    let [n, q, p, h_t, q_t, q_next_mc, q_prime] = parents.map(into_cube::<I>);
    let hour = stage_prologue_dev(consts_dev, &n, &q, &p, &q_t, &q_next_mc, &q_prime, b0, b1);
    let mut fwd = stage_hour_forward_dev(consts_dev, pattern, &hour, &h_t, &q_t, save);
    let saved = save.then(|| HourSavedDev {
        consts: consts_dev.clone(),
        hour,
        subs: std::mem::take(&mut fwd.subs),
    });
    (from_cube::<I>(fwd.out.clone()), fwd, saved)
}
