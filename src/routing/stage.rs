//! Per-network constants for the stage routing block (host side, f32).
//!
//! The stage set is a downstream-closed set of reaches that drains to exactly
//! one network outlet. This module builds the set, the bed elevations, the
//! edge lengths, the edge masks on the `CsrPattern` layout and the dam rows.

use std::sync::Arc;

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, IndexingUpdateOp, Int, Tensor, TensorData};

use crate::config::DownstreamBoundary;
use crate::geometry::{compute_trapezoidal_geometry, trapezoid_from_depth};
use crate::sparse::tree_lu::{assert_tree, tree_lu_solve};
use crate::sparse::{CsrPattern, SparseAdjacency};

const SECONDS_PER_DAY: f32 = 86_400.0;

/// Constants of one routed network for the stage step.
#[derive(Clone, Debug)]
pub struct StageNetwork {
    pub n: usize,
    /// Downstream reach of each reach, -1 at a network outlet.
    pub down: Vec<i32>,
    /// The stage set's one outlet (C1). It is a network outlet, because the set is downstream-closed.
    pub outlet: usize,
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
    /// `(L_i + L_down)/2`; at a network outlet `L_i / 2` (C21).
    pub edge_length: Vec<f32>,
    /// Length nnz, 1.0 where the column is MC and the row is stage.
    pub adj_mc_to_stage: Vec<f32>,
    /// Length nnz, 1.0 where both ends are stage.
    pub adj_stage_to_stage: Vec<f32>,
    /// Dams inside the stage set.
    pub dam_rows: Vec<usize>,
    /// Dam residence time in seconds, converted from days (C30).
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
    /// `Err` when the pattern is not a tree or the stage set has 0 or 2+ outlets.
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
        if outlets.len() != 1 {
            return Err(format!(
                "the stage set must drain to exactly 1 outlet, found {} outlet(s) {:?}",
                outlets.len(),
                outlets
            ));
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
    /// Passed to `tree_lu_solve`; false in v1 (the validator rejects CUDA).
    pub use_cuda: bool,
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
    /// `sum (|in| + |out| + |q'|) dt_s` (C27).
    pub gross_flux_m3: f32,
    /// `sum dt_s (I_start + I_end)/2` over the MC inflow edges (C22).
    pub mc_inflow_m3: f32,
    /// Routing hours stepped (one per `StageBlock::step` call).
    pub steps: usize,
}

impl StageMassBalance {
    /// `|storage_change - net_flux - floor_created| / gross_flux` (C27).
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

/// Coefficients of the last substep, on the inner backend (test hook, C22).
#[doc(hidden)]
pub struct StageCoefficients<I: Backend> {
    pub alpha: Tensor<I, 1>,
    pub beta: Tensor<I, 1>,
    pub area_edge: Tensor<I, 1>,
    pub radius_edge: Tensor<I, 1>,
    pub d_s: Tensor<I, 1>,
    pub dt_phi: Tensor<I, 1>,
    pub r_prime: Tensor<I, 1>,
}

/// Row-level constants (length `n`) on the device.
struct RowConsts<I: Backend> {
    down_idx: Tensor<Ad<I>, 1, Int>,
    in_stage_b: Tensor<Ad<I>, 1, Bool>,
    in_stage: Tensor<Ad<I>, 1>,
    /// Friction rows: stage, not a dam, not the normal-depth outlet (C8).
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

/// Slot-level constants (length `nnz`), built on the first step.
struct SlotConsts<I: Backend> {
    pattern: Arc<CsrPattern>,
    row_idx: Tensor<Ad<I>, 1, Int>,
    col_idx: Tensor<Ad<I>, 1, Int>,
    diag_slot: Tensor<Ad<I>, 1>,
    adj_stage: Tensor<Ad<I>, 1>,
    adj_mc: Tensor<Ad<I>, 1>,
    /// `adj_stage_to_stage` with control-row columns removed (C8).
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
    h_t: Option<Tensor<Ad<I>, 1>>,
    boundary_series: Option<Vec<f32>>,
    step_index: usize,
    /// `[storage_change, net_flux, floor_created, gross_flux, mc_inflow]`, inner backend.
    acc: Tensor<I, 1>,
    steps: usize,
    last_coeffs: Option<StageCoefficients<I>>,
    record: bool,
    coeff_record: Vec<(Vec<f32>, Vec<f32>)>,
}

fn f32_tensor<I: Backend>(v: &[f32], device: &I::Device) -> Tensor<Ad<I>, 1> {
    Tensor::<Ad<I>, 1>::from_floats(v, device)
}

fn one_minus<I: Backend>(t: Tensor<Ad<I>, 1>) -> Tensor<Ad<I>, 1> {
    t.neg().add_scalar(1.0)
}

fn scatter_add<I: Backend>(
    n: usize,
    idx: &Tensor<Ad<I>, 1, Int>,
    vals: Tensor<Ad<I>, 1>,
) -> Tensor<Ad<I>, 1> {
    let zeros = Tensor::<Ad<I>, 1>::zeros([n], &vals.device());
    zeros.scatter(0, idx.clone(), vals, IndexingUpdateOp::Add)
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
            .map(|i| f(!prescribed && i == net.outlet && dam[i] == 0.0))
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
        outlet[net.outlet] = 1.0;
        // The outlet's downstream bed is `z_out[outlet]`, where `h_bc` sits (C21).
        let z_down: Vec<f32> = (0..n)
            .map(|i| {
                if i == net.outlet {
                    net.z_out[i]
                } else if net.down[i] >= 0 {
                    net.z_mid[net.down[i] as usize]
                } else {
                    net.z_mid[i]
                }
            })
            .collect();
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
            h_t: None,
            boundary_series: None,
            step_index: 0,
            acc: Tensor::<I, 1>::zeros([5], device),
            steps: 0,
            last_coeffs: None,
            record: false,
            coeff_record: Vec::new(),
        }
    }

    /// Hourly boundary stage, one value per output column. Hour `k` (0-based)
    /// reads `series[k]` at its start and `series[k + 1]` at its end, so a run
    /// of `H` hours needs `H + 1` values. Used with `PrescribedStage`.
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
        self.h_t = Some(self.c.z_mid.clone() + depth);
        self.step_index = 0;
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
        self.last_coeffs.as_ref().map(|c| StageCoefficients {
            alpha: c.alpha.clone(),
            beta: c.beta.clone(),
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

    fn ensure_slots(&mut self, pattern: &Arc<CsrPattern>, device: &I::Device) {
        if let Some(s) = &self.slots {
            if Arc::ptr_eq(&s.pattern, pattern) {
                return;
            }
        }
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

    /// One routing hour. Returns `q_next_mc` with the stage rows replaced by the
    /// stage discharge; that value is the next hour's `q_t` on stage rows.
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
        let device = q_t.device();
        self.ensure_slots(pattern, &device);
        let sl = self.slots.as_ref().expect("slots built");
        let c = &self.c;
        let pr = self.params;
        let nn = self.net.n;
        let (th, thc) = (pr.theta, pr.theta_control);
        let th2 = th * th;
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
        let spmv_mc = |x: Tensor<Ad<I>, 1>| {
            scatter_add::<I>(nn, &sl.row_idx, x.gather(0, sl.col_idx.clone()) * sl.adj_mc.clone())
        };
        let i_t = spmv_mc(q_t.clone());
        let i_n = spmv_mc(q_next_mc.clone());
        let i_delta = i_n - i_t.clone();

        let q_eps = q.clone() + 1e-6;
        let n2 = n_man.clone() * n_man.clone();
        let qp_stage = q_prime_t * c.in_stage.clone();

        let mut qc = q_t;
        let mut acc = Tensor::<I, 1>::zeros([5], &device);
        let mut coeffs = None;

        for s in 0..subs {
            let (f0, f1) = (s as f32 / subs as f32, (s + 1) as f32 / subs as f32);
            let i_mid = (i_t.clone() * 2.0 + i_delta.clone() * (f0 + f1)) * 0.5;
            let (hbc0, hbc1) = (b0 + (b1 - b0) * f0, b0 + (b1 - b0) * f1);

            // 1. depths and the edge's down-stage
            let depth = (h.clone() - c.z_mid.clone()).clamp_min(pr.depth_lb);
            let mut h_down = h.clone().gather(0, c.down_idx.clone());
            if prescribed {
                h_down = h_down * one_minus::<I>(c.outlet.clone()) + c.outlet.clone() * hbc0;
            }
            let d_edge = (h.clone().max_pair(h_down.clone()) - c.z_mid.clone().max_pair(c.z_down.clone()))
                .clamp_min(pr.depth_lb);

            // 2. edge section at d_edge, storage section at depth
            let ge = trapezoid_from_depth(p.clone(), q.clone(), d_edge, pr.bottom_width_lb);
            let gs = trapezoid_from_depth(p.clone(), q.clone(), depth.clone(), pr.bottom_width_lb);
            let d_s = gs.top_width.clone() * c.length.clone() / dt_s;

            // 3. friction, alpha, beta, Q_hat (zero on control and non-stage rows, C8)
            let phi = n2.clone() * qc.clone().abs() * pr.g
                / (ge.area.clone() * ge.hydraulic_radius.clone().powf_scalar(4.0 / 3.0));
            let dt_phi = phi * dt_s;
            let inv = (dt_phi.clone() + 1.0).recip();
            let alpha = inv.clone() * c.friction.clone();
            let beta = ge.area.clone() * inv * c.edge_length.clone().recip() * (pr.g * dt_s) * c.friction.clone();
            let q_hat = alpha.clone() * qc.clone() + beta.clone() * (h.clone() - h_down) * (1.0 - th);

            // 4. control edges: dams (power-law storage over T) and the normal-depth outlet
            let e1 = q_eps.clone() + 1.0;
            let r_dam = c.length.clone() * p.clone() * depth.clone().powf(e1.clone()) / e1 / c.dam_t.clone();
            let rp_dam = d_s.clone() * dt_s / c.dam_t.clone();
            let kk = gs.area.clone() * gs.hydraulic_radius.clone().powf_scalar(2.0 / 3.0) / n_man.clone();
            let s0_sqrt = c.s0.clone().sqrt();
            let r_nd = kk.clone() * s0_sqrt.clone();
            let dp_dd = (gs.side_slope.clone().powf_scalar(2.0) + 1.0).sqrt() * 2.0;
            let dk_dd = kk.clone() * gs.top_width.clone() / gs.area.clone() * (5.0 / 3.0)
                - kk * dp_dd / gs.wetted_perimeter.clone() * (2.0 / 3.0);
            let rp_nd = dk_dd * s0_sqrt;
            let r_t = c.dam.clone() * r_dam + c.nd.clone() * r_nd;
            let r_prime = (c.dam.clone() * rp_dam + c.nd.clone() * rp_nd).clamp_min(0.0);

            // 5. matrix values [2·nnz]
            let beta_edge = beta.clone().gather(0, sl.col_idx.clone()) * sl.adj_friction.clone();
            let off_f = beta_edge.clone() * (-th2);
            let off_dam = r_prime.clone().gather(0, sl.col_idx.clone()) * sl.dam_slot.clone() * (-thc);
            let sum_in_beta = scatter_add::<I>(nn, &sl.row_idx, beta_edge);
            let diag_stage = d_s.clone() + (beta.clone() + sum_in_beta) * th2;
            let diag = diag_stage * c.in_stage.clone()
                + c.w_ctrl.clone() * r_prime.clone()
                + one_minus::<I>(c.in_stage.clone());
            let lower = off_f.clone() + off_dam + diag.gather(0, sl.row_idx.clone()) * sl.diag_slot.clone();
            let a_values = Tensor::cat(vec![lower, off_f], 0);

            // 6. right-hand side: each edge's old flux once, with its own weight
            let flux_old = c.friction.clone() * ((q_hat.clone() * th) + qc.clone() * (1.0 - th));
            let ctrl_old = c.w_ctrl.clone() * (r_t.clone() - r_prime.clone() * h.clone())
                + (c.ctrl.clone() - c.w_ctrl.clone()) * qc.clone();
            let e_old = flux_old + ctrl_old;
            let in_e = scatter_add::<I>(
                nn,
                &sl.row_idx,
                e_old.clone().gather(0, sl.col_idx.clone()) * sl.adj_stage.clone(),
            );
            let mut b = d_s.clone() * h.clone() * c.in_stage.clone() + qp_stage.clone() + i_mid.clone() + in_e - e_old;
            if prescribed {
                b = b + c.outlet.clone() * beta.clone() * (th2 * hbc1);
            }

            // 7. solve, then floor the stage; the created volume is counted
            let h_solved = tree_lu_solve(pattern, a_values, b, pr.use_cuda);
            let h_floor = c.z_mid.clone() + pr.depth_lb;
            let h_new = h_solved.clone().max_pair(h_floor);

            // 8. readout from the solved stage (so the balance closes exactly)
            let mut hs_down = h_solved.clone().gather(0, c.down_idx.clone());
            if prescribed {
                hs_down = hs_down * one_minus::<I>(c.outlet.clone()) + c.outlet.clone() * hbc1;
            }
            let q_fr = q_hat + beta.clone() * (h_solved.clone() - hs_down) * th;
            let q_ctrl = (r_t + r_prime.clone() * (h_solved.clone() - h.clone())) * c.ctrl.clone();
            let q_new = q_fr + q_ctrl;

            // 9. mass balance on the inner backend
            {
                let w = c.w_edge.clone();
                let f_edge = (w.clone() * q_new.clone().detach() + one_minus::<I>(w) * qc.clone().detach())
                    * c.in_stage.clone();
                let into = scatter_add::<I>(
                    nn,
                    &sl.row_idx,
                    f_edge.clone().gather(0, sl.col_idx.clone()) * sl.adj_stage.clone(),
                );
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
                let gross = (((f_edge.abs() * (c.in_stage.clone() + c.has_down.clone() * c.in_stage.clone()))
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
        self.last_coeffs = coeffs;
        self.h_t = Some(h);
        self.step_index += 1;
        q_next_mc.mask_where(self.c.in_stage_b.clone(), qc)
    }
}
