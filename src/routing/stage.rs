//! The stage routing block (f32): per-network constants and the step.
//!
//! The stage set is a downstream-closed set of reaches that drains to exactly
//! one network outlet. `StageNetwork` (host side) builds the set, the bed
//! elevations, the edge lengths, the edge masks on the `CsrPattern` layout and
//! the dam rows. `StageBlock` (device side) takes one Newton-linearised
//! local-inertial step per routing hour: assembly, tree solve, readout.

use std::sync::Arc;

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor, TensorData, TensorPrimitive};

use crate::config::DownstreamBoundary;
use crate::data::store::ChannelGeometryRows;
use crate::geometry::compute_trapezoidal_geometry;
use crate::routing::stage_cuda::forward::AssembledDev;
use crate::routing::stage_cuda::StageOpConstsDev;
use crate::routing::stage_fma::{fma3, fma_s, spmv_fma};
use crate::routing::stage_op::{stage_timestep, Assembled, HourOut, StageOpConsts};
use crate::sparse::cube_buf::{cube_to_host, from_cube, into_cube, Cube};
use crate::sparse::dispatch::effective_use_cuda;
use crate::sparse::tree_lu::{assert_tree, tree_lu_solve};
use crate::sparse::{CsrPattern, SparseAdjacency};

const SECONDS_PER_DAY: f32 = 86_400.0;

/// Constants of one routed network for the stage step.
#[derive(Clone, Debug)]
pub struct StageNetwork {
    pub n: usize,
    /// Downstream reach of each reach, -1 at a network outlet.
    pub down: Vec<i32>,
    /// The first outlet of the stage set (the only one under `prescribed_stage`).
    pub outlet: usize,
    /// Every outlet of the stage set. Each is a network outlet, because the set
    /// is downstream-closed. More than one is a forest, allowed only with a
    /// normal-depth boundary (each outlet gets its own control edge).
    pub outlets: Vec<usize>,
    /// Downstream-closed stage membership.
    pub in_stage: Vec<bool>,
    /// Size of the stage set.
    pub n_stage: usize,
    /// Reaches added by the downstream closure.
    pub n_added: usize,
    /// Bed at the reach outlet; `anchor_m` at every network outlet.
    pub z_out: Vec<f32>,
    /// `z_out + slope * length / 2` (raw slope, no floor).
    pub z_mid: Vec<f32>,
    /// `(L_i + L_down)/2`; at a network outlet `L_i / 2`.
    pub edge_length: Vec<f32>,
    /// Length nnz, 1.0 where the column is MC and the row is stage.
    pub adj_mc_to_stage: Vec<f32>,
    /// Length nnz, 1.0 where both ends are stage.
    pub adj_stage_to_stage: Vec<f32>,
    /// Dams inside the stage set.
    pub dam_rows: Vec<usize>,
    /// Dam residence time in seconds, converted from days.
    pub dam_t_seconds: Vec<f32>,
    /// Reach length in metres (the storage term `D_s` needs it).
    pub length_m: Vec<f32>,
    /// Raw bed slope (the normal-depth outlet reads its own).
    pub slope: Vec<f32>,
}

fn down_of(adj: &SparseAdjacency) -> Vec<i32> {
    let mut down = vec![-1i32; adj.n];
    for (&r, &c) in adj.rows.iter().zip(&adj.cols) {
        if r != c {
            down[c as usize] = r;
        }
    }
    down
}

/// Bed at the downstream end of each reach's edge. The outlet's downstream bed
/// is `z_out[outlet]`, where `h_bc` sits.
fn z_down_of(net: &StageNetwork) -> Vec<f32> {
    (0..net.n)
        .map(|i| {
            if net.down[i] < 0 {
                net.z_out[i]
            } else if net.down[i] >= 0 {
                net.z_mid[net.down[i] as usize]
            } else {
                net.z_mid[i]
            }
        })
        .collect()
}

/// Mark `seeds` and every reach downstream of them. Returns the mask and the
/// number of reaches the closure added beyond the seeds.
pub fn downstream_closure(adj: &SparseAdjacency, seeds: &[usize]) -> (Vec<bool>, usize) {
    let down = down_of(adj);
    let mut in_stage = vec![false; adj.n];
    for &s in seeds {
        in_stage[s] = true;
    }
    let seed_count = in_stage.iter().filter(|&&b| b).count();
    // rows >= cols: a downstream reach has a higher index than its upstream
    // neighbour, so an ascending sweep marks the whole path in one pass.
    for i in 0..adj.n {
        if in_stage[i] && down[i] >= 0 {
            in_stage[down[i] as usize] = true;
        }
    }
    let total = in_stage.iter().filter(|&&b| b).count();
    (in_stage, total - seed_count)
}

/// Bed elevation at each reach outlet and midpoint. Integrates `slope * length`
/// upstream from `anchor_m` at every network outlet.
pub fn bed_elevation(adj: &SparseAdjacency, anchor_m: f32) -> (Vec<f32>, Vec<f32>) {
    let down = down_of(adj);
    let mut z_out = vec![anchor_m; adj.n];
    for i in (0..adj.n).rev() {
        if down[i] >= 0 {
            let d = down[i] as usize;
            z_out[i] = z_out[d] + adj.slope[d] * adj.length_m[d];
        }
    }
    let z_mid = (0..adj.n)
        .map(|i| z_out[i] + adj.slope[i] * adj.length_m[i] / 2.0)
        .collect();
    (z_out, z_mid)
}

impl StageNetwork {
    /// Build the constants. `seeds = None` means every reach. `dams` holds
    /// `(row, T in days)`. A dam row outside the stage set is skipped. Returns
    /// `Err` when the pattern is not a forest or the stage set is empty. A stage set with
    /// several outlets is built; `prescribed_stage` allows only one (`set_stage_block`).
    pub fn build(
        adj: &SparseAdjacency,
        pattern: &CsrPattern,
        seeds: Option<&[usize]>,
        anchor_m: f32,
        dams: &[(usize, f32)],
    ) -> Result<Self, String> {
        assert_tree(pattern)?;
        let n = adj.n;
        let down = down_of(adj);
        let all: Vec<usize>;
        let seed_slice = match seeds {
            Some(s) => s,
            None => {
                all = (0..n).collect();
                &all
            }
        };
        if let Some(&bad) = seed_slice.iter().find(|&&s| s >= n) {
            return Err(format!("stage seed row {bad} is out of range (n = {n})"));
        }
        let (in_stage, n_added) = downstream_closure(adj, seed_slice);
        let n_stage = in_stage.iter().filter(|&&b| b).count();
        let outlets: Vec<usize> = (0..n).filter(|&i| in_stage[i] && down[i] < 0).collect();
        if outlets.is_empty() {
            return Err("the stage set is empty: it has no outlet".to_string());
        }
        let outlet = outlets[0];

        let (z_out, z_mid) = bed_elevation(adj, anchor_m);
        let edge_length = (0..n)
            .map(|i| {
                if down[i] < 0 {
                    adj.length_m[i] / 2.0
                } else {
                    (adj.length_m[i] + adj.length_m[down[i] as usize]) / 2.0
                }
            })
            .collect();

        let nnz = pattern.nnz();
        let mut adj_mc_to_stage = vec![0.0f32; nnz];
        let mut adj_stage_to_stage = vec![0.0f32; nnz];
        for s in 0..nnz {
            let r = pattern.row_for_nnz[s] as usize;
            let c = pattern.col[s] as usize;
            if r == c {
                continue;
            }
            if in_stage[r] && in_stage[c] {
                adj_stage_to_stage[s] = 1.0;
            } else if in_stage[r] {
                adj_mc_to_stage[s] = 1.0;
            }
        }

        let mut dam_rows = Vec::new();
        let mut dam_t_seconds = Vec::new();
        for &(row, t_days) in dams {
            if row < n && in_stage[row] {
                dam_rows.push(row);
                dam_t_seconds.push(t_days * SECONDS_PER_DAY);
            }
        }

        Ok(Self {
            n,
            down,
            outlet,
            outlets,
            in_stage,
            n_stage,
            n_added,
            z_out,
            z_mid,
            edge_length,
            adj_mc_to_stage,
            adj_stage_to_stage,
            dam_rows,
            dam_t_seconds,
            length_m: adj.length_m.clone(),
            slope: adj.slope.clone(),
        })
    }
}

// =============================================================================
// The stage step (design: Physics, Time weights per edge, Coupling and state).
// =============================================================================

type Ad<I> = Autodiff<I>;

/// Settings of one stage block.
#[derive(Debug, Clone, Copy)]
pub struct StageParams {
    /// Weight of friction edges and the normal-depth outlet, in `[0.5, 1]`.
    pub theta: f32,
    /// Weight of dam edges, in `(0, 1]`.
    pub theta_control: f32,
    /// Internal steps per routing hour.
    pub substeps: u32,
    pub boundary: DownstreamBoundary,
    pub depth_lb: f32,
    pub bottom_width_lb: f32,
    /// Routing step in seconds (3600).
    pub dt: f32,
    /// Gravity, 9.81.
    pub g: f32,
    /// Passed to `tree_lu_solve` and `stage_timestep`. True runs the CUDA arms of the stage op and the tree solve.
    pub use_cuda: bool,
}

/// What the engine hook needs to arm a stage block (`MuskingumCunge::set_stage_block`).
#[derive(Debug, Clone)]
pub struct StageInputs {
    /// Seed rows of the stage set (`StageBatch::seed_rows`); `None` is every reach.
    pub seeds: Option<Vec<usize>>,
    /// `stage_routing.anchor_elevation_m`. When the stage outlet has a table
    /// geometry row, the table's `bed_z_m` replaces it (see `set_stage_block`).
    pub anchor_m: f32,
    /// Hourly boundary stage, one value per q' row (`StageBatch::boundary`).
    pub boundary_series: Option<Vec<f32>>,
    /// Network row of `boundary_comid` (`StageBatch::boundary_row`).
    pub boundary_row: Option<usize>,
    /// Table rows and `strict` (= `geometry == Table`).
    pub geometry: Option<(ChannelGeometryRows, bool)>,
    /// `(row, T in days)` of the window's active fixed-table dams.
    pub dams: Vec<(usize, f32)>,
}

/// The block's state at the end of a chunk, per reach in network order. Only
/// the stage rows are read back. `q` is the unclamped discharge.
#[derive(Debug, Clone)]
pub struct StageCarry {
    pub h: Vec<f32>,
    pub q: Vec<f32>,
}

/// Volume bookkeeping of the stage rows, summed over the steps taken (m3).
#[derive(Debug, Clone, Copy, Default)]
pub struct StageMassBalance {
    /// `sum D_s (h_new - h) dt_s`, after the depth floor.
    pub storage_change_m3: f32,
    /// `sum (inflows + q' - outflow) dt_s` with the per-edge time weights.
    pub net_flux_m3: f32,
    /// Volume the depth floor added.
    pub floor_created_m3: f32,
    /// `sum (|in| + |out| + |q'|) dt_s`.
    pub gross_flux_m3: f32,
    /// `sum dt_s (I_start + I_end)/2` over the MC inflow edges.
    pub mc_inflow_m3: f32,
    /// Routing hours stepped (one per `StageBlock::step` call).
    pub steps: usize,
}

impl StageMassBalance {
    /// `|storage_change - net_flux - floor_created| / gross_flux`.
    pub fn rel(&self) -> f32 {
        (self.storage_change_m3 - self.net_flux_m3 - self.floor_created_m3).abs()
            / self.gross_flux_m3.max(f32::MIN_POSITIVE)
    }

    /// Sum every field. Eval chunks add up this way.
    pub fn merge(&mut self, other: &StageMassBalance) {
        self.storage_change_m3 += other.storage_change_m3;
        self.net_flux_m3 += other.net_flux_m3;
        self.floor_created_m3 += other.floor_created_m3;
        self.gross_flux_m3 += other.gross_flux_m3;
        self.mc_inflow_m3 += other.mc_inflow_m3;
        self.steps += other.steps;
    }
}

/// Coefficients of the last substep, on the inner backend (test hook).
#[doc(hidden)]
pub struct StageCoefficients<I: Backend> {
    /// `(1 + dt_s*phi)/(1 + 2*dt_s*phi)` on friction rows, 0 elsewhere.
    pub alpha: Tensor<I, 1>,
    /// `g*A_e*dt_s/(L_edge*(1 + 2*dt_s*phi))` on friction rows, 0 elsewhere.
    pub beta: Tensor<I, 1>,
    /// Signed section sensitivity of the edge flux, 0 off friction rows.
    pub kappa: Tensor<I, 1>,
    /// `max(theta*beta + kappa*s_up, 0)`.
    pub a_up: Tensor<I, 1>,
    /// `max(theta*beta - kappa*s_dn, 0)`.
    pub b_down: Tensor<I, 1>,
    pub area_edge: Tensor<I, 1>,
    pub radius_edge: Tensor<I, 1>,
    pub d_s: Tensor<I, 1>,
    pub dt_phi: Tensor<I, 1>,
    pub r_prime: Tensor<I, 1>,
}

/// The ten `StageCoefficients` fields of the last substep as device buffers
/// (the CUDA fused path; no host copy).
struct LastCoeffsDev {
    alpha: Cube,
    beta: Cube,
    kappa: Cube,
    a_up: Cube,
    b_down: Cube,
    area_edge: Cube,
    radius_edge: Cube,
    d_s: Cube,
    dt_phi: Cube,
    r_prime: Cube,
}

impl LastCoeffsDev {
    fn of(a: &AssembledDev) -> Self {
        Self {
            alpha: a.newton.alpha.clone(),
            beta: a.newton.beta.clone(),
            kappa: a.newton.kappa.clone(),
            a_up: a.newton.a_up.clone(),
            b_down: a.newton.b_down.clone(),
            area_edge: a.edge.out_area.clone(),
            radius_edge: a.edge.out_radius.clone(),
            d_s: a.d_s.clone(),
            dt_phi: a.newton.dt_phi.clone(),
            r_prime: a.control.r_prime.clone(),
        }
    }
}

/// Row-level constants (length `n`) on the device.
struct RowConsts<I: Backend> {
    down_idx: Tensor<Ad<I>, 1, Int>,
    in_stage_b: Tensor<Ad<I>, 1, Bool>,
    in_stage: Tensor<Ad<I>, 1>,
    /// Friction rows: stage, not a dam, not the normal-depth outlet.
    friction: Tensor<Ad<I>, 1>,
    dam: Tensor<Ad<I>, 1>,
    /// Control rows: dams and the normal-depth outlet.
    ctrl: Tensor<Ad<I>, 1>,
    /// The normal-depth outlet when it is not a dam.
    nd: Tensor<Ad<I>, 1>,
    /// Time weight of the control row's own edge: `theta_c` on dams, `theta` on the outlet.
    w_ctrl: Tensor<Ad<I>, 1>,
    /// Time weight of every stage row's outgoing edge.
    w_edge: Tensor<Ad<I>, 1>,
    /// 1 where the reach has a downstream reach.
    has_down: Tensor<Ad<I>, 1>,
    outlet: Tensor<Ad<I>, 1>,
    z_mid: Tensor<Ad<I>, 1>,
    z_down: Tensor<Ad<I>, 1>,
    length: Tensor<Ad<I>, 1>,
    edge_length: Tensor<Ad<I>, 1>,
    /// Dam residence time in seconds; 1 on other rows so divisions stay finite.
    dam_t: Tensor<Ad<I>, 1>,
    /// Bed slope for the normal-depth outlet, floored at 1e-5.
    s0: Tensor<Ad<I>, 1>,
}

/// Table geometry on the device (`set_table_geometry`). Rows without a table
/// row have `mask` 0 and neutral values, so the blends return the head's value
/// exactly.
struct TableConsts<I: Backend> {
    /// 1 on table rows in the stage set.
    mask: Tensor<Ad<I>, 1>,
    /// Top width; 1 off the table.
    tw: Tensor<Ad<I>, 1>,
    /// Fixed Manning `n`; 1 off the table.
    n: Tensor<Ad<I>, 1>,
    /// Dam release reference stage (`pool_z`, or the bed); 0 off the table.
    pool: Tensor<Ad<I>, 1>,
    /// 1 where the table row gave an explicit `pool_z_m`.
    has_pool: Tensor<Ad<I>, 1>,
    /// Host copy of the bed for the hotstart.
    bed_h: Vec<f32>,
}

/// Edge or storage section: head trapezoid, or the rectangle on table rows.
struct Section<I: Backend> {
    top_width: Tensor<Ad<I>, 1>,
    area: Tensor<Ad<I>, 1>,
    wetted_perimeter: Tensor<Ad<I>, 1>,
    hydraulic_radius: Tensor<Ad<I>, 1>,
    /// `dA/dd`.
    a_d: Tensor<Ad<I>, 1>,
    /// `dP/dd`.
    p_d: Tensor<Ad<I>, 1>,
}

/// Slot-level constants (length `nnz`), built on the first step.
struct SlotConsts<I: Backend> {
    pattern: Arc<CsrPattern>,
    row_idx: Tensor<Ad<I>, 1, Int>,
    col_idx: Tensor<Ad<I>, 1, Int>,
    diag_slot: Tensor<Ad<I>, 1>,
    adj_stage: Tensor<Ad<I>, 1>,
    adj_mc: Tensor<Ad<I>, 1>,
    /// `adj_stage_to_stage` with control-row columns removed.
    adj_friction: Tensor<Ad<I>, 1>,
    /// Stage-to-stage slots whose column is a dam.
    dam_slot: Tensor<Ad<I>, 1>,
}

/// Implicit local-inertial stage solve on the stage rows of a routed network.
pub struct StageBlock<I: Backend> {
    net: StageNetwork,
    params: StageParams,
    c: RowConsts<I>,
    slots: Option<SlotConsts<I>>,
    table: Option<TableConsts<I>>,
    h_t: Option<Tensor<Ad<I>, 1>>,
    boundary_series: Option<Vec<f32>>,
    step_index: usize,
    /// `[storage_change, net_flux, floor_created, gross_flux, mc_inflow]`, inner backend.
    acc: Tensor<I, 1>,
    steps: usize,
    last_coeffs: Option<StageCoefficients<I>>,
    /// Last substep of the fused path (host); `last_coeffs` is `None` then.
    last_host: Option<Assembled>,
    /// Last substep of the fused path on CUDA (device); `last_host` is `None` then.
    last_dev: Option<LastCoeffsDev>,
    /// Host copy of the constants for the fused op; cleared when the table or the slots change.
    op_consts: Option<Arc<StageOpConsts>>,
    /// Device copy of `op_consts`; cleared with it.
    op_consts_dev: Option<Arc<StageOpConstsDev>>,
    reference_step: bool,
    record: bool,
    coeff_record: Vec<(Vec<f32>, Vec<f32>)>,
    /// Per hour, the end-of-hour stage of every row (kept when `record` is on).
    stage_record: Vec<Vec<f32>>,
}

fn f32_tensor<I: Backend>(v: &[f32], device: &I::Device) -> Tensor<Ad<I>, 1> {
    Tensor::<Ad<I>, 1>::from_floats(v, device)
}

fn one_minus<I: Backend>(t: Tensor<Ad<I>, 1>) -> Tensor<Ad<I>, 1> {
    t.neg().add_scalar(1.0)
}

impl<I: Backend + 'static> StageBlock<I>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    /// Build the constants. Row masks (friction rows, control rows, weights) are
    /// built here; the slot masks wait for the pattern of the first `step`.
    pub fn new(net: StageNetwork, params: StageParams, device: &I::Device) -> Self {
        let n = net.n;
        let prescribed = params.boundary == DownstreamBoundary::PrescribedStage;
        let mut dam = vec![0.0f32; n];
        let mut dam_t = vec![1.0f32; n];
        for (&r, &t) in net.dam_rows.iter().zip(&net.dam_t_seconds) {
            dam[r] = 1.0;
            dam_t[r] = t;
        }
        let f = |b: bool| if b { 1.0f32 } else { 0.0 };
        let in_stage: Vec<f32> = net.in_stage.iter().map(|&b| f(b)).collect();
        let nd: Vec<f32> = (0..n)
            .map(|i| f(!prescribed && net.in_stage[i] && net.down[i] < 0 && dam[i] == 0.0))
            .collect();
        let friction: Vec<f32> = (0..n)
            .map(|i| f(net.in_stage[i] && dam[i] == 0.0 && nd[i] == 0.0))
            .collect();
        let ctrl: Vec<f32> = (0..n).map(|i| dam[i] + nd[i]).collect();
        let w_ctrl: Vec<f32> = (0..n)
            .map(|i| params.theta_control * dam[i] + params.theta * nd[i])
            .collect();
        let w_edge: Vec<f32> = (0..n)
            .map(|i| params.theta * (friction[i] + nd[i]) + params.theta_control * dam[i])
            .collect();
        let has_down: Vec<f32> = net.down.iter().map(|&d| f(d >= 0)).collect();
        let down_idx: Vec<i32> = (0..n)
            .map(|i| if net.down[i] >= 0 { net.down[i] } else { i as i32 })
            .collect();
        let mut outlet = vec![0.0f32; n];
        for &o in &net.outlets {
            outlet[o] = 1.0;
        }
        let z_down = z_down_of(&net);
        let s0: Vec<f32> = net.slope.iter().map(|s| s.max(1e-5)).collect();
        let c = RowConsts {
            down_idx: Tensor::<Ad<I>, 1, Int>::from_data(TensorData::from(down_idx.as_slice()), device),
            in_stage_b: Tensor::<Ad<I>, 1, Bool>::from_data(TensorData::from(net.in_stage.as_slice()), device),
            in_stage: f32_tensor::<I>(&in_stage, device),
            friction: f32_tensor::<I>(&friction, device),
            dam: f32_tensor::<I>(&dam, device),
            ctrl: f32_tensor::<I>(&ctrl, device),
            nd: f32_tensor::<I>(&nd, device),
            w_ctrl: f32_tensor::<I>(&w_ctrl, device),
            w_edge: f32_tensor::<I>(&w_edge, device),
            has_down: f32_tensor::<I>(&has_down, device),
            outlet: f32_tensor::<I>(&outlet, device),
            z_mid: f32_tensor::<I>(&net.z_mid, device),
            z_down: f32_tensor::<I>(&z_down, device),
            length: f32_tensor::<I>(&net.length_m, device),
            edge_length: f32_tensor::<I>(&net.edge_length, device),
            dam_t: f32_tensor::<I>(&dam_t, device),
            s0: f32_tensor::<I>(&s0, device),
        };
        Self {
            net,
            params,
            c,
            slots: None,
            table: None,
            h_t: None,
            boundary_series: None,
            step_index: 0,
            acc: Tensor::<I, 1>::zeros([5], device),
            steps: 0,
            last_coeffs: None,
            last_host: None,
            last_dev: None,
            op_consts: None,
            op_consts_dev: None,
            reference_step: false,
            record: false,
            coeff_record: Vec::new(),
            stage_record: Vec::new(),
        }
    }

    /// Section at `depth`: the head's trapezoid, with the rectangle (`A_d = TW`,
    /// `P_d = 2`) on table rows. The head's `p`, `q` get exactly zero gradient
    /// on table rows (the mask multiplies the trapezoid branch).
    fn section(
        &self,
        p: &Tensor<Ad<I>, 1>,
        q: &Tensor<Ad<I>, 1>,
        depth: Tensor<Ad<I>, 1>,
    ) -> Section<I> {
        // The reference section is written out here, with the fused multiply-adds (NVRTC
        // contracts a*b+c into one rounding, so the CPU fuses the same sites), line for line with `stage_op.rs::section_host`. `src/geometry.rs`
        // (shared with the hotstart) keeps its unfused form.
        let bw_lb = self.params.bottom_width_lb;
        let one = Tensor::<Ad<I>, 1>::ones_like(&depth);
        let q_eps = q.clone() + 1e-6;
        // 2a
        let tw = p.clone() * depth.clone().powf(q_eps.clone());
        let ssr = tw.clone() * q_eps.clone() / (depth.clone() * 2.0);
        let ss = ssr.clone().clamp(0.5, 50.0);
        // 2b
        let bwr = tw.clone() - ss.clone() * depth.clone() * 2.0;
        let bw = bwr.clone().clamp_min(bw_lb);
        // 2c
        let area = (tw.clone() + bw.clone()) * depth.clone() / 2.0;
        let root = fma3::<I>(ss.clone(), ss.clone(), one.clone()).sqrt();
        let perim = bw.clone() + depth.clone() * root.clone() * 2.0;
        let radius = area.clone() / perim.clone();
        // 2d
        let tw_d = q_eps * tw.clone() / depth.clone();
        let ss_active = ssr.clone().greater_elem(0.5).int() * ssr.clone().lower_elem(50.0).int();
        let ss_active = ss_active.float();
        let ss_d = (ssr / depth.clone())
            * fma3::<I>(tw_d.clone() / tw.clone(), depth.clone(), one.clone().neg())
            * ss_active;
        // 2e
        let bw_active = bwr.greater_elem(bw_lb).float();
        let bw_d = (tw_d.clone() - ss.clone() * 2.0 - depth.clone() * ss_d.clone() * 2.0) * bw_active;
        // 2f
        let a_d = (tw.clone() + bw.clone()) / 2.0 + depth.clone() * (tw_d + bw_d.clone()) / 2.0;
        let p_d = bw_d + root.clone() * 2.0 + depth.clone() * ss * ss_d * 2.0 / root;
        let Some(t) = &self.table else {
            return Section { top_width: tw, area, wetted_perimeter: perim, hydraulic_radius: radius, a_d, p_d };
        };
        let keep = one_minus::<I>(t.mask.clone());
        let area_r = t.tw.clone() * depth.clone();
        let perim_r = t.tw.clone() + depth * 2.0;
        let radius_r = area_r.clone() / perim_r.clone();
        let mix = |head: Tensor<Ad<I>, 1>, tab: Tensor<Ad<I>, 1>| fma3::<I>(head, keep.clone(), tab * t.mask.clone());
        Section {
            top_width: mix(tw, t.tw.clone()),
            area: mix(area, area_r),
            wetted_perimeter: mix(perim, perim_r),
            hydraulic_radius: mix(radius, radius_r),
            a_d: mix(a_d, t.tw.clone()),
            p_d: mix(p_d, Tensor::<Ad<I>, 1>::ones_like(&t.tw) * 2.0),
        }
    }

    /// Use the table for the reaches it lists (`geometry: table | mixed`).
    /// Table rows are rectangles of width `top_width`, fixed `n`, bed `bed_z`
    /// (it replaces the head's `z_mid` there) and, on dam rows, release
    /// reference `pool_z` (the bed when `None`). Table rows outside the stage
    /// set are ignored. With `strict` (`geometry: table`) a stage row without
    /// a table row is an error; otherwise it keeps the head's geometry.
    /// Call before `hotstart`, after `new`.
    pub fn set_table_geometry(&mut self, rows: ChannelGeometryRows, strict: bool) -> Result<(), String> {
        let nn = self.net.n;
        let k = rows.rows.len();
        if [rows.top_width.len(), rows.bed_z.len(), rows.n.len(), rows.pool_z.len()]
            .iter()
            .any(|&l| l != k)
        {
            return Err("table geometry columns differ in length".into());
        }
        let mut mask = vec![0.0f32; nn];
        let mut listed = vec![false; nn];
        let mut tw = vec![1.0f32; nn];
        let mut nt = vec![1.0f32; nn];
        let mut bed = self.net.z_mid.clone();
        let mut pool = vec![0.0f32; nn];
        let mut has_pool = vec![0.0f32; nn];
        for j in 0..k {
            let r = rows.rows[j];
            if r >= nn {
                return Err(format!("table geometry row {r} is out of range (n = {nn})"));
            }
            if listed[r] {
                return Err(format!("table geometry lists row {r} twice"));
            }
            listed[r] = true;
            if !self.net.in_stage[r] {
                continue;
            }
            mask[r] = 1.0;
            tw[r] = rows.top_width[j];
            nt[r] = rows.n[j];
            bed[r] = rows.bed_z[j];
            pool[r] = rows.pool_z[j].unwrap_or(rows.bed_z[j]);
            has_pool[r] = if rows.pool_z[j].is_some() { 1.0 } else { 0.0 };
        }
        if strict {
            if let Some(i) = (0..nn).find(|&i| self.net.in_stage[i] && mask[i] == 0.0) {
                return Err(format!(
                    "stage row {i} has no row in the channel geometry table (geometry: table)"
                ));
            }
        }
        self.op_consts = None;
        self.op_consts_dev = None;
        let device = self.c.z_mid.device();
        self.net.z_mid = bed;
        self.c.z_mid = f32_tensor::<I>(&self.net.z_mid, &device);
        self.c.z_down = f32_tensor::<I>(&z_down_of(&self.net), &device);
        self.table = Some(TableConsts {
            mask: f32_tensor::<I>(&mask, &device),
            tw: f32_tensor::<I>(&tw, &device),
            n: f32_tensor::<I>(&nt, &device),
            pool: f32_tensor::<I>(&pool, &device),
            has_pool: f32_tensor::<I>(&has_pool, &device),
            bed_h: self.net.z_mid.clone(),
        });
        Ok(())
    }

    /// Hourly boundary stage, one value per output column. Hour `k` (0-based)
    /// reads `series[k]` at its start and `series[k + 1]` at its end, so
    /// `H` steps need `H + 1` values. Used with `PrescribedStage`.
    pub fn set_boundary_series(&mut self, series: Vec<f32>) -> Result<(), String> {
        if series.len() < 2 {
            return Err(format!(
                "the boundary series needs at least 2 values (start and end of one hour), got {}",
                series.len()
            ));
        }
        if let Some(i) = series.iter().position(|v| !v.is_finite()) {
            return Err(format!("the boundary series has a non-finite value at index {i}"));
        }
        self.boundary_series = Some(series);
        Ok(())
    }

    /// Start state: normal depth from `q0` at the floored slope on friction
    /// reaches, `S(d0) = T·Q0` on dams (`S = L·p·d^(q+1)/(q+1)`).
    pub fn hotstart(
        &mut self,
        q0: Tensor<Ad<I>, 1>,
        n: Tensor<Ad<I>, 1>,
        p: Tensor<Ad<I>, 1>,
        q: Tensor<Ad<I>, 1>,
        slope_floored: Tensor<Ad<I>, 1>,
    ) {
        let pr = &self.params;
        let q0 = q0.clamp_min(1e-6);
        let (q0_tab, slope_tab) = (q0.clone(), slope_floored.clone());
        let geom = compute_trapezoidal_geometry(
            n,
            p.clone(),
            q.clone(),
            q0.clone(),
            slope_floored,
            pr.depth_lb,
            pr.bottom_width_lb,
        );
        let q_eps = q + 1e-6;
        let e = q_eps.clone() + 1.0;
        let d_dam = (e.clone() * self.c.dam_t.clone() * q0 / (self.c.length.clone() * p))
            .powf(e.recip())
            .clamp_min(pr.depth_lb);
        let depth = geom.depth * one_minus::<I>(self.c.dam.clone()) + d_dam * self.c.dam.clone();
        let h_head = self.c.z_mid.clone() + depth;
        self.h_t = Some(match &self.table {
            None => h_head,
            Some(t) => {
                let h_tab = self.table_hotstart(t, q0_tab, slope_tab);
                h_head * one_minus::<I>(t.mask.clone()) + h_tab * t.mask.clone()
            }
        });
        self.step_index = 0;
    }

    /// Table-row start stage. Non-dam rows: the boundary at t0 under
    /// `PrescribedStage` (at least `bed + depth_lb`), else bed plus the
    /// wide-channel normal depth of the rectangle. Dam rows: option C on the
    /// active storage, `pool + T*Q0/(TW*L)`.
    fn table_hotstart(
        &self,
        t: &TableConsts<I>,
        q0: Tensor<Ad<I>, 1>,
        slope_floored: Tensor<Ad<I>, 1>,
    ) -> Tensor<Ad<I>, 1> {
        let pr = &self.params;
        let device = q0.device();
        let nn = self.net.n;
        let h_dam = t.pool.clone()
            + self.c.dam_t.clone() * q0.clone() / (t.tw.clone() * self.c.length.clone());
        let prescribed = pr.boundary == DownstreamBoundary::PrescribedStage;
        let h_flow = match (prescribed, self.boundary_series.as_ref()) {
            (true, Some(s)) => {
                let v: Vec<f32> = (0..nn).map(|i| (t.bed_h[i] + pr.depth_lb).max(s[0])).collect();
                f32_tensor::<I>(&v, &device)
            }
            _ => {
                let dn = (t.n.clone() * q0 / (t.tw.clone() * slope_floored.sqrt()))
                    .powf_scalar(0.6)
                    .clamp_min(pr.depth_lb);
                f32_tensor::<I>(&t.bed_h, &device) + dn
            }
        };
        // A non-dam row with `pool_z_m` starts full to normal pool.
        let h_flow = h_flow * one_minus::<I>(t.has_pool.clone()) + t.pool.clone() * t.has_pool.clone();
        h_dam * self.c.dam.clone() + h_flow * one_minus::<I>(self.c.dam.clone())
    }

    /// Stage-set membership per reach.
    pub fn stage_rows(&self) -> &[bool] {
        &self.net.in_stage
    }

    /// Current stage (`None` before `hotstart`).
    pub fn stage(&self) -> Option<Tensor<Ad<I>, 1>> {
        self.h_t.clone()
    }

    /// Volume bookkeeping so far. Reads five scalars from the device.
    pub fn mass_balance(&self) -> StageMassBalance {
        let v: Vec<f32> = self.acc.clone().into_data().to_vec().expect("f32 accumulators");
        StageMassBalance {
            storage_change_m3: v[0],
            net_flux_m3: v[1],
            floor_created_m3: v[2],
            gross_flux_m3: v[3],
            mc_inflow_m3: v[4],
            steps: self.steps,
        }
    }

    /// Test hook: replace the carried stage (the gradient check perturbs it).
    #[doc(hidden)]
    pub fn set_stage(&mut self, h: Tensor<Ad<I>, 1>) {
        self.h_t = Some(h);
    }

    /// Test hook: coefficients of the last substep.
    #[doc(hidden)]
    pub fn last_coefficients(&self) -> Option<StageCoefficients<I>> {
        if let Some(a) = &self.last_host {
            let device = self.c.z_mid.device();
            let t = |v: &[f32]| Tensor::<I, 1>::from_floats(v, &device);
            return Some(StageCoefficients {
                alpha: t(&a.newton.alpha),
                beta: t(&a.newton.beta),
                kappa: t(&a.newton.kappa),
                a_up: t(&a.newton.a_up),
                b_down: t(&a.newton.b_down),
                area_edge: t(&a.edge.out_area),
                radius_edge: t(&a.edge.out_radius),
                d_s: t(&a.d_s),
                dt_phi: t(&a.newton.dt_phi),
                r_prime: t(&a.control.r_prime),
            });
        }
        if let Some(d) = &self.last_dev {
            let t = |c: &Cube| {
                Tensor::<I, 1>::from_primitive(TensorPrimitive::Float(from_cube::<I>(c.clone())))
            };
            return Some(StageCoefficients {
                alpha: t(&d.alpha),
                beta: t(&d.beta),
                kappa: t(&d.kappa),
                a_up: t(&d.a_up),
                b_down: t(&d.b_down),
                area_edge: t(&d.area_edge),
                radius_edge: t(&d.radius_edge),
                d_s: t(&d.d_s),
                dt_phi: t(&d.dt_phi),
                r_prime: t(&d.r_prime),
            });
        }
        self.last_coeffs.as_ref().map(|c| StageCoefficients {
            alpha: c.alpha.clone(),
            beta: c.beta.clone(),
            kappa: c.kappa.clone(),
            a_up: c.a_up.clone(),
            b_down: c.b_down.clone(),
            area_edge: c.area_edge.clone(),
            radius_edge: c.radius_edge.clone(),
            d_s: c.d_s.clone(),
            dt_phi: c.dt_phi.clone(),
            r_prime: c.r_prime.clone(),
        })
    }

    /// Keep each hour's `dt_phi` and `alpha` (last substep) on the host.
    #[doc(hidden)]
    pub fn record_coefficients(&mut self, on: bool) {
        self.record = on;
    }

    /// Per hour `(dt_phi, alpha)`, each of length `n`.
    #[doc(hidden)]
    pub fn coefficient_record(&self) -> &[(Vec<f32>, Vec<f32>)] {
        &self.coeff_record
    }

    /// Per hour, the stage `h` of every row at the hour's end (recorded with
    /// [`Self::record_coefficients`]).
    #[doc(hidden)]
    pub fn stage_record(&self) -> &[Vec<f32>] {
        &self.stage_record
    }

    fn ensure_slots(&mut self, pattern: &Arc<CsrPattern>, device: &I::Device) {
        if let Some(s) = &self.slots {
            if Arc::ptr_eq(&s.pattern, pattern) {
                return;
            }
        }
        self.op_consts = None;
        self.op_consts_dev = None;
        let nnz = pattern.nnz();
        assert_eq!(pattern.n, self.net.n, "pattern and stage network differ in size");
        assert_eq!(self.net.adj_stage_to_stage.len(), nnz, "pattern and stage masks differ in nnz");
        let mut dam = vec![false; self.net.n];
        for &r in &self.net.dam_rows {
            dam[r] = true;
        }
        let mut adj_friction = vec![0.0f32; nnz];
        let mut dam_slot = vec![0.0f32; nnz];
        let mut diag = vec![0.0f32; nnz];
        for s in 0..nnz {
            let col = pattern.col[s] as usize;
            if pattern.row_for_nnz[s] as usize == col {
                diag[s] = 1.0;
            } else if self.net.adj_stage_to_stage[s] == 1.0 {
                if dam[col] {
                    dam_slot[s] = 1.0;
                } else {
                    adj_friction[s] = 1.0;
                }
            }
        }
        self.slots = Some(SlotConsts {
            pattern: pattern.clone(),
            row_idx: Tensor::<Ad<I>, 1, Int>::from_data(TensorData::from(pattern.row_for_nnz.as_slice()), device),
            col_idx: Tensor::<Ad<I>, 1, Int>::from_data(TensorData::from(pattern.col.as_slice()), device),
            diag_slot: f32_tensor::<I>(&diag, device),
            adj_stage: f32_tensor::<I>(&self.net.adj_stage_to_stage, device),
            adj_mc: f32_tensor::<I>(&self.net.adj_mc_to_stage, device),
            adj_friction: f32_tensor::<I>(&adj_friction, device),
            dam_slot: f32_tensor::<I>(&dam_slot, device),
        });
    }

    /// `q_t` as the MC op must read it: stage rows become
    /// `q_t.clamp_min(discharge_lb).detach()`, MC rows are unchanged. The
    /// stage rows of the MC output are discarded, so the clamp only keeps the
    /// MC op finite when a stage row carries backflow. The block's own `step`
    /// keeps reading the unclamped `q_t`.
    pub fn mc_view(&self, q_t: Tensor<Ad<I>, 1>, discharge_lb: f32) -> Tensor<Ad<I>, 1> {
        let clamped = q_t.clone().clamp_min(discharge_lb).detach();
        q_t.mask_where(self.c.in_stage_b.clone(), clamped)
    }

    /// One routing hour. Returns `q_next_mc` with the stage rows replaced by the
    /// stage discharge; that value is the next hour's `q_t` on stage rows.
    /// Runs the fused op `step_fused` by default, or the autodiff reference
    /// `step_reference` after `set_reference_step(true)`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        pattern: &Arc<CsrPattern>,
        n_man: Tensor<Ad<I>, 1>,
        p: Tensor<Ad<I>, 1>,
        q: Tensor<Ad<I>, 1>,
        q_t: Tensor<Ad<I>, 1>,
        q_next_mc: Tensor<Ad<I>, 1>,
        q_prime_t: Tensor<Ad<I>, 1>,
    ) -> Tensor<Ad<I>, 1> {
        if self.reference_step {
            self.step_reference(pattern, n_man, p, q, q_t, q_next_mc, q_prime_t)
        } else {
            self.step_fused(pattern, n_man, p, q, q_t, q_next_mc, q_prime_t)
        }
    }

    /// Choose the step path. On, `step` runs the Burn-autodiff reference
    /// (`step_reference`); off (the default), it runs the fused op
    /// (`step_fused`). The reference exists for the parity tests and the
    /// tape count.
    #[doc(hidden)]
    pub fn set_reference_step(&mut self, on: bool) {
        self.reference_step = on;
    }

    /// The Burn-autodiff step: the definition of correct. The fused op
    /// (`step_fused`) is pinned to it.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn step_reference(
        &mut self,
        pattern: &Arc<CsrPattern>,
        n_man: Tensor<Ad<I>, 1>,
        p: Tensor<Ad<I>, 1>,
        q: Tensor<Ad<I>, 1>,
        q_t: Tensor<Ad<I>, 1>,
        q_next_mc: Tensor<Ad<I>, 1>,
        q_prime_t: Tensor<Ad<I>, 1>,
    ) -> Tensor<Ad<I>, 1> {
        let device = q_t.device();
        self.ensure_slots(pattern, &device);
        let sl = self.slots.as_ref().expect("slots built");
        let c = &self.c;
        let pr = self.params;
        let (th, thc) = (pr.theta, pr.theta_control);
        let subs = pr.substeps.max(1) as usize;
        let dt_s = pr.dt / subs as f32;
        let prescribed = pr.boundary == DownstreamBoundary::PrescribedStage;
        let mut h = self.h_t.clone().expect("StageBlock::hotstart (or set_stage) before step");
        let (b0, b1) = if prescribed {
            let s = self
                .boundary_series
                .as_ref()
                .expect("PrescribedStage needs set_boundary_series");
            assert!(
                self.step_index + 1 < s.len(),
                "boundary series has {} values, hour {} needs index {}",
                s.len(),
                self.step_index,
                self.step_index + 1
            );
            (s[self.step_index], s[self.step_index + 1])
        } else {
            (0.0, 0.0)
        };

        // MC inflow at the hour's two ends; substeps interpolate linearly.
        // Row sums fuse the product into the sum, as the fused op does (one rounding per a*b+c site).
        let spmv_mc = |x: Tensor<Ad<I>, 1>| spmv_fma::<I>(&sl.pattern, x, sl.adj_mc.clone());
        let i_t = spmv_mc(q_t.clone());
        let i_n = spmv_mc(q_next_mc.clone());
        let i_delta = i_n - i_t.clone();

        let q_eps = q.clone() + 1e-6;
        // Fixed n on table rows (zero gradient to the head's n there).
        let n_man = match &self.table {
            None => n_man,
            Some(t) => fma3::<I>(n_man, one_minus::<I>(t.mask.clone()), t.n.clone() * t.mask.clone()),
        };
        let n2 = n_man.clone() * n_man.clone();
        let qp_stage = q_prime_t * c.in_stage.clone();

        let mut qc = q_t;
        let mut acc = Tensor::<I, 1>::zeros([5], &device);
        let mut coeffs = None;

        for s in 0..subs {
            let (f0, f1) = (s as f32 / subs as f32, (s + 1) as f32 / subs as f32);
            let i_mid = fma_s::<I>(i_delta.clone(), f0 + f1, i_t.clone() * 2.0) * 0.5;
            let (hbc0, hbc1) = (b0 + (b1 - b0) * f0, b0 + (b1 - b0) * f1);

            // 1. depths and the edge's down-stage
            let depth = (h.clone() - c.z_mid.clone()).clamp_min(pr.depth_lb);
            let mut h_down = h.clone().gather(0, c.down_idx.clone());
            if prescribed {
                h_down = h_down * one_minus::<I>(c.outlet.clone()) + c.outlet.clone() * hbc0;
            }
            let d_edge_raw = h.clone().max_pair(h_down.clone()) - c.z_mid.clone().max_pair(c.z_down.clone());
            let d_edge = d_edge_raw.clone().clamp_min(pr.depth_lb);
            // Which end the Bates max follows, where the floor is inactive (no gradient).
            let above = d_edge_raw.detach().greater_elem(pr.depth_lb).float();
            let up_ge = (h.clone().detach() - h_down.clone().detach()).greater_equal_elem(0.0).float();
            let s_up = up_ge.clone() * above.clone();
            let s_dn = one_minus::<I>(up_ge) * above;

            // 2. edge section at d_edge, storage section at depth
            let ge = self.section(&p, &q, d_edge.clone());
            let (a_d, p_d) = (ge.a_d.clone(), ge.p_d.clone());
            let gs = self.section(&p, &q, depth.clone());
            let (a_ds, p_ds) = (gs.a_d.clone(), gs.p_d.clone());
            let d_s = gs.top_width.clone() * c.length.clone() / dt_s;

            // 3. friction, alpha, beta, Q*, kappa, a, b (zero on control and non-stage rows)
            let phi = n2.clone() * qc.clone().abs() * pr.g
                / (ge.area.clone() * ge.hydraulic_radius.clone().powf_scalar(4.0 / 3.0));
            let dt_phi = phi * dt_s;
            let den = dt_phi.clone() * 2.0 + 1.0;
            let alpha = (dt_phi.clone() + 1.0) / den.clone() * c.friction.clone();
            let beta = ge.area.clone() * (pr.g * dt_s) / (c.edge_length.clone() * den.clone()) * c.friction.clone();
            let phi_d = fma_s::<I>(
                a_d.clone() / ge.area.clone() - p_d / ge.wetted_perimeter.clone(),
                (4.0f64 / 3.0) as f32,
                a_d.clone() / ge.area.clone(),
            );
            let dh_edge = h.clone() - h_down;
            let kappa = fma3::<I>(
                dt_phi.clone() * qc.clone(),
                phi_d,
                a_d * dh_edge.clone() * (pr.g * dt_s) / c.edge_length.clone(),
            ) / den
                * c.friction.clone();
            let q_star = fma3::<I>(alpha.clone(), qc.clone(), beta.clone() * dh_edge);
            let a_up = fma_s::<I>(beta.clone(), th, kappa.clone() * s_up).clamp_min(0.0) * c.friction.clone();
            let b_down = fma_s::<I>(beta.clone(), th, (kappa.clone() * s_dn).neg()).clamp_min(0.0) * c.friction.clone();

            // 4. control edges: dams (power-law storage over T) and the normal-depth outlet
            let e1 = q_eps.clone() + 1.0;
            let mut r_dam = c.length.clone() * p.clone() * depth.clone().powf(e1.clone()) / e1 / c.dam_t.clone();
            let mut rp_dam = d_s.clone() * dt_s / c.dam_t.clone();
            if let Some(t) = &self.table {
                // Table dam rows: release over the active storage above the
                // pool level, R = TW*L*max(h - pool, 0)/T (option C).
                let excess = h.clone() - t.pool.clone();
                let active = excess.clone().detach().greater_elem(0.0).float();
                let r_tab = t.tw.clone() * c.length.clone() * excess.clamp_min(0.0) / c.dam_t.clone();
                let rp_tab = rp_dam.clone() * active;
                r_dam = fma3::<I>(r_dam, one_minus::<I>(t.mask.clone()), r_tab * t.mask.clone());
                rp_dam = fma3::<I>(rp_dam, one_minus::<I>(t.mask.clone()), rp_tab * t.mask.clone());
            }
            let kk = gs.area.clone() * gs.hydraulic_radius.clone().powf_scalar(2.0 / 3.0) / n_man.clone();
            let s0_sqrt = c.s0.clone().sqrt();
            let r_nd = kk.clone() * s0_sqrt.clone();
            let w_kk = fma_s::<I>(
                a_ds / gs.area.clone(),
                (5.0f64 / 3.0) as f32,
                (p_ds / gs.wetted_perimeter.clone() * (2.0 / 3.0)).neg(),
            );
            let dk_dd = kk * w_kk;
            let rp_nd = dk_dd * s0_sqrt;
            let r_t = fma3::<I>(c.dam.clone(), r_dam, c.nd.clone() * r_nd);
            let r_prime = fma3::<I>(c.dam.clone(), rp_dam, c.nd.clone() * rp_nd).clamp_min(0.0);

            // 5. matrix values [2·nnz]; the unknown is the increment dh
            let a_edge = a_up.clone().gather(0, sl.col_idx.clone()) * sl.adj_friction.clone();
            let b_edge = b_down.clone().gather(0, sl.col_idx.clone()) * sl.adj_friction.clone();
            let upper_f = b_edge.clone().neg() * th;
            let off_a = r_prime.clone().gather(0, sl.col_idx.clone()) * sl.dam_slot.clone();
            // sum_in_b fuses b_down[col] * adj_friction into the row sum; `b_edge` stays rounded for `upper_f`.
            let sum_in_b = spmv_fma::<I>(&sl.pattern, b_down.clone(), sl.adj_friction.clone());
            let diag_stage = fma_s::<I>(a_up.clone() + sum_in_b, th, d_s.clone());
            let diag = fma3::<I>(diag_stage, c.in_stage.clone(), c.w_ctrl.clone() * r_prime.clone())
                + one_minus::<I>(c.in_stage.clone());
            let off_dam = off_a * (-thc);
            let lower = fma3::<I>(
                diag.gather(0, sl.row_idx.clone()),
                sl.diag_slot.clone(),
                fma_s::<I>(a_edge.neg(), th, off_dam),
            );
            let a_values = Tensor::cat(vec![lower, upper_f], 0);

            // 6. right-hand side r: fluxes only, each edge's old flux once with its own weight
            let inner_old = fma_s::<I>(q_star.clone(), th, qc.clone() * (1.0 - th));
            let ctrl_old = fma3::<I>(c.w_ctrl.clone(), r_t.clone(), (c.ctrl.clone() - c.w_ctrl.clone()) * qc.clone());
            let e_old = fma3::<I>(c.friction.clone(), inner_old, ctrl_old);
            let in_e = spmv_fma::<I>(&sl.pattern, e_old.clone(), sl.adj_stage.clone());
            let mut r = qp_stage.clone() + i_mid.clone() + in_e - e_old;
            if prescribed {
                r = fma_s::<I>(c.outlet.clone() * b_down.clone(), th * (hbc1 - hbc0), r);
            }

            // 7. solve for dh, then floor the stage; the created volume is counted
            let dh = tree_lu_solve(pattern, a_values, r, pr.use_cuda);
            let h_solved = h.clone() + dh.clone();
            let h_floor = c.z_mid.clone() + pr.depth_lb;
            let h_new = h_solved.clone().max_pair(h_floor);

            // 8. readout from dh before the floor (never Q* - a*h + b*h_down)
            let mut dh_down = dh.clone().gather(0, c.down_idx.clone());
            if prescribed {
                dh_down = dh_down * one_minus::<I>(c.outlet.clone()) + c.outlet.clone() * (hbc1 - hbc0);
            }
            let q_fr = fma3::<I>(
                b_down.clone().neg(),
                dh_down,
                fma3::<I>(a_up.clone(), dh.clone(), q_star),
            );
            let q_ctrl_sum = fma3::<I>(r_prime.clone(), dh, r_t);
            let q_new = fma3::<I>(q_ctrl_sum, c.ctrl.clone(), q_fr);

            // 9. mass balance on the inner backend
            {
                let w = c.w_edge.clone();
                let f_inner = fma3::<I>(w.clone(), q_new.clone().detach(), one_minus::<I>(w) * qc.clone().detach());
                let f_edge = f_inner.clone() * c.in_stage.clone();
                let into = spmv_fma::<I>(&sl.pattern, f_edge.clone(), sl.adj_stage.clone());
                let i_mid_d = i_mid.clone().detach();
                let qp_d = qp_stage.clone().detach();
                let net_flux = ((into.clone() + i_mid_d.clone() + qp_d.clone() - f_edge.clone()).sum() * dt_s).inner();
                let storage = ((d_s.clone().detach() * (h_new.clone().detach() - h.clone().detach())
                    * c.in_stage.clone())
                .sum()
                    * dt_s)
                    .inner();
                let floor = ((d_s.clone().detach() * (h_new.clone().detach() - h_solved.clone().detach())
                    * c.in_stage.clone())
                .sum()
                    * dt_s)
                    .inner();
                let gross = (((f_edge.abs() * fma3::<I>(c.has_down.clone(), c.in_stage.clone(), c.in_stage.clone()))
                    .sum()
                    + i_mid_d.clone().abs().sum()
                    + qp_d.abs().sum())
                    * dt_s)
                    .inner();
                let mc_in = (i_mid_d.sum() * dt_s).inner();
                acc = acc + Tensor::cat(vec![storage, net_flux, floor, gross, mc_in], 0);
            }

            if s + 1 == subs {
                coeffs = Some(StageCoefficients {
                    alpha: alpha.inner(),
                    beta: beta.inner(),
                    kappa: kappa.inner(),
                    a_up: a_up.inner(),
                    b_down: b_down.inner(),
                    area_edge: ge.area.inner(),
                    radius_edge: ge.hydraulic_radius.inner(),
                    d_s: d_s.inner(),
                    dt_phi: dt_phi.inner(),
                    r_prime: r_prime.inner(),
                });
            }
            h = h_new;
            qc = q_new;
        }

        self.acc = self.acc.clone() + acc;
        self.steps += 1;
        if self.record {
            if let Some(cf) = &coeffs {
                self.coeff_record.push((
                    cf.dt_phi.clone().into_data().to_vec().expect("f32"),
                    cf.alpha.clone().into_data().to_vec().expect("f32"),
                ));
            }
        }
        self.last_host = None;
        self.last_dev = None;
        self.last_coeffs = coeffs;
        if self.record {
            self.stage_record.push(h.clone().inner().into_data().to_vec().expect("f32"));
        }
        self.h_t = Some(h);
        self.step_index += 1;
        q_next_mc.mask_where(self.c.in_stage_b.clone(), qc)
    }

    /// Host copy of the block's constants for the fused op. Reads the float
    /// constants back from the device tensors once, so both paths see the same
    /// bits. Cached until the table or the slots change.
    #[doc(hidden)]
    pub fn op_consts(&mut self, pattern: &Arc<CsrPattern>) -> Arc<StageOpConsts> {
        let device = self.c.z_mid.device();
        self.ensure_slots(pattern, &device);
        if let Some(c) = &self.op_consts {
            return c.clone();
        }
        let host = |t: &Tensor<Ad<I>, 1>| -> Vec<f32> { t.clone().inner().into_data().to_vec().expect("f32") };
        let sl = self.slots.as_ref().expect("slots built");
        let c = &self.c;
        let nn = self.net.n;
        let (t_mask, t_tw, t_n, t_pool) = match &self.table {
            Some(t) => (host(&t.mask), host(&t.tw), host(&t.n), host(&t.pool)),
            None => (vec![], vec![], vec![], vec![]),
        };
        let out = StageOpConsts {
            n: nn,
            nnz: pattern.nnz(),
            substeps: self.params.substeps.max(1),
            theta: self.params.theta,
            theta_c: self.params.theta_control,
            g: self.params.g,
            depth_lb: self.params.depth_lb,
            bw_lb: self.params.bottom_width_lb,
            dt: self.params.dt,
            prescribed: self.params.boundary == DownstreamBoundary::PrescribedStage,
            has_table: self.table.is_some(),
            down: (0..nn)
                .map(|i| if self.net.down[i] >= 0 { self.net.down[i] as usize } else { i })
                .collect(),
            in_stage: host(&c.in_stage),
            friction: host(&c.friction),
            dam: host(&c.dam),
            ctrl: host(&c.ctrl),
            nd: host(&c.nd),
            w_ctrl: host(&c.w_ctrl),
            w_edge: host(&c.w_edge),
            has_down: host(&c.has_down),
            outlet: host(&c.outlet),
            z_mid: host(&c.z_mid),
            z_down: host(&c.z_down),
            length: host(&c.length),
            edge_length: host(&c.edge_length),
            dam_t: host(&c.dam_t),
            s0_sqrt: c.s0.clone().inner().sqrt().into_data().to_vec().expect("f32"),
            t_mask,
            t_tw,
            t_n,
            t_pool,
            row: pattern.row_for_nnz.iter().map(|&r| r as usize).collect(),
            col: pattern.col.iter().map(|&c| c as usize).collect(),
            diag_slot: host(&sl.diag_slot),
            adj_stage: host(&sl.adj_stage),
            adj_mc: host(&sl.adj_mc),
            adj_friction: host(&sl.adj_friction),
            dam_slot: host(&sl.dam_slot),
        };
        let out = Arc::new(out);
        self.op_consts = Some(out.clone());
        out
    }

    /// Device copy of `op_consts`, cached and cleared with it. Panics unless
    /// the block's backend is `Cuda<f32, i32>`.
    #[doc(hidden)]
    pub fn op_consts_dev(&mut self, pattern: &Arc<CsrPattern>) -> Arc<StageOpConstsDev> {
        let host = self.op_consts(pattern);
        if let Some(d) = &self.op_consts_dev {
            return d.clone();
        }
        let prim = match self.c.z_mid.clone().inner().into_primitive() {
            TensorPrimitive::Float(p) => p,
            TensorPrimitive::QFloat(_) => panic!("expected float tensor"),
        };
        let like = into_cube::<I>(prim);
        let dev = Arc::new(StageOpConstsDev::upload(&host, pattern, &like));
        self.op_consts_dev = Some(dev.clone());
        dev
    }

    /// One routing hour through the fused op: one autodiff node
    /// with a hand-written backward. Same arguments and result as
    /// `step_reference`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn step_fused(
        &mut self,
        pattern: &Arc<CsrPattern>,
        n_man: Tensor<Ad<I>, 1>,
        p: Tensor<Ad<I>, 1>,
        q: Tensor<Ad<I>, 1>,
        q_t: Tensor<Ad<I>, 1>,
        q_next_mc: Tensor<Ad<I>, 1>,
        q_prime_t: Tensor<Ad<I>, 1>,
    ) -> Tensor<Ad<I>, 1> {
        let device = q_t.device();
        self.ensure_slots(pattern, &device);
        let pr = self.params;
        let prescribed = pr.boundary == DownstreamBoundary::PrescribedStage;
        let h_t = self.h_t.clone().expect("StageBlock::hotstart (or set_stage) before step");
        // Mirrors the prologue of `step_reference`.
        let (b0, b1) = if prescribed {
            let s = self
                .boundary_series
                .as_ref()
                .expect("PrescribedStage needs set_boundary_series");
            assert!(
                self.step_index + 1 < s.len(),
                "boundary series has {} values, hour {} needs index {}",
                s.len(),
                self.step_index,
                self.step_index + 1
            );
            (s[self.step_index], s[self.step_index + 1])
        } else {
            (0.0, 0.0)
        };
        let consts = self.op_consts(pattern);
        // The device constants are built only on the CUDA arm.
        let consts_dev = if effective_use_cuda::<I>(pr.use_cuda) { Some(self.op_consts_dev(pattern)) } else { None };
        let (out, fwd) = stage_timestep::<I>(
            &consts,
            consts_dev.as_ref(),
            pattern,
            n_man,
            q,
            p,
            h_t,
            q_t,
            q_next_mc,
            q_prime_t,
            b0,
            b1,
            pr.use_cuda,
        );
        self.finish_fused_hour(out, fwd, &device)
    }

    /// The hour epilogue of `step_fused`, shared by both arms: the mass
    /// accumulator, `steps`, the record path, the last coefficients, `h_t` and
    /// `step_index`. The device arm reads nothing back unless `record` is on.
    fn finish_fused_hour(&mut self, out: Tensor<Ad<I>, 1>, fwd: HourOut, device: &I::Device) -> Tensor<Ad<I>, 1> {
        let nn = self.net.n;
        let h_end = out.clone().slice([0..nn]);
        let q_out = out.slice([nn..2 * nn]);
        match fwd {
            HourOut::Host(fwd) => {
                self.acc = self.acc.clone() + Tensor::<I, 1>::from_floats(fwd.mass, device);
                if self.record {
                    self.coeff_record.push((fwd.last.newton.dt_phi.clone(), fwd.last.newton.alpha.clone()));
                    self.stage_record.push(fwd.h_end.clone());
                }
                self.last_host = Some(fwd.last);
                self.last_dev = None;
            }
            HourOut::Dev(fwd) => {
                let mass = Tensor::<I, 1>::from_primitive(TensorPrimitive::Float(from_cube::<I>(fwd.mass.clone())));
                self.acc = self.acc.clone() + mass;
                if self.record {
                    self.coeff_record
                        .push((cube_to_host(&fwd.last.newton.dt_phi), cube_to_host(&fwd.last.newton.alpha)));
                    self.stage_record.push(cube_to_host(&fwd.h_end));
                }
                self.last_dev = Some(LastCoeffsDev::of(&fwd.last));
                self.last_host = None;
            }
        }
        self.steps += 1;
        self.last_coeffs = None;
        self.h_t = Some(h_end);
        self.step_index += 1;
        q_out
    }
}
