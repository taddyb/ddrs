//! One gauge, one arm, one functional: the gradient of a scalar of routed
//! gauge discharge with respect to the hourly lateral inflow at every
//! upstream reach and hour.
//!
//! The inflow gradient has always been computed by the routing backward —
//! `TimestepOp` registers it on its fifth parent (`src/routing/mmc_op.rs`)
//! and `CsrSolveOp` on its RHS parent (`src/sparse/mod.rs`) — but nothing
//! ever attached that parent to a leaf. This module lifts the hourly inflow
//! tensor as a `require_grad` leaf (the 2-D analogue of
//! `training::probe::lift_leaf`) and reads the gradient back after
//! `backward()`. No `Backward` impl is touched.

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use chrono::Duration;

use crate::config::{kan_config, Config, ConfigMode, SparseSolver};
use crate::data::dataset::{MeritGagesDataset, RoutingBatch, RoutingTensors};
use crate::data::dates::{RhoWindow, TimeAxis};
use crate::data::ids::Staid;
use crate::nn::kan_head::KanHead;
use crate::routing::utils::denormalize;
use crate::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use crate::training::checkpoint::{head_base, load_kan_head};
use crate::training::forward::{gather_params_to_subreaches, scatter_add_by_group};
use crate::training::loss::tau_trim_and_downsample;

use super::super::{BoxError, ResolvedArm};

type AD<I> = Autodiff<I>;

/// Per-arm state: the arm's config (Testing mode), dataset, and frozen head.
pub struct InfluenceContext<I: Backend> {
    pub cfg: Config,
    pub dataset: MeritGagesDataset,
    pub head: KanHead<AD<I>>,
    pub device: I::Device,
    pub axis: TimeAxis,
    pub tau: u32,
    pub warmup: usize,
}

impl<I: Backend + 'static> InfluenceContext<I>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    pub fn open(arm: &ResolvedArm, device: &I::Device, force_cpu: bool) -> Result<Self, BoxError> {
        let mut cfg = Config::from_yaml_file_with_mode(&arm.config_path, ConfigMode::Testing)
            .map_err(|e| format!("arm `{}`: {e}", arm.name))?;
        if force_cpu {
            cfg.params.sparse_solver = SparseSolver::Cpu;
            cfg.params.use_cuda_graphs = false;
        }
        if cfg.params.use_leakance {
            return Err(format!("arm `{}`: leakance arms are out of scope for the adjoint study", arm.name).into());
        }
        if cfg.params.ddr_match {
            return Err(format!(
                "arm `{}`: ddr_match arms are unsupported (gauge prediction omits the gauge reach)",
                arm.name
            )
            .into());
        }
        let dataset = MeritGagesDataset::open(&cfg).map_err(|e| format!("arm `{}`: {e}", arm.name))?;
        <I as Backend>::seed(device, cfg.seed);
        let section = cfg
            .kan_head
            .as_ref()
            .ok_or_else(|| format!("arm `{}`: config has no kan_head section", arm.name))?;
        let template: KanHead<AD<I>> = kan_config(section, cfg.seed).init::<AD<I>>(device);
        let head = load_kan_head::<AD<I>>(&head_base(&arm.checkpoint_dir), template, device)
            .map_err(|e| format!("arm `{}`: {e}", arm.name))?;
        let axis = dataset.time_axis().clone();
        let tau = cfg.params.tau;
        let warmup = cfg.experiment.as_ref().map(|e| e.warmup).unwrap_or(5);
        Ok(Self { cfg, dataset, head, device: device.clone(), axis, tau, warmup })
    }

    /// The gauge's observed daily series over the whole eval axis
    /// (`n_days_full`), NaN = missing.
    pub fn gauge_observations(&self, staid: &Staid) -> Result<Vec<f32>, BoxError> {
        let col = self
            .dataset
            .staids()
            .iter()
            .position(|s| s == staid)
            .ok_or_else(|| format!("gauge {staid} is not in the arm's dataset (dropped or not in gages CSV)"))?;
        let obs = self.dataset.full_observations()?;
        Ok(obs.column(col).to_vec())
    }

    /// Collate the gauge's own subgraph over `[start_day_idx, start_day_idx + window_days)`.
    pub fn collate_gauge(
        &self,
        staid: &Staid,
        start_day_idx: usize,
        window_days: usize,
    ) -> Result<RoutingBatch, BoxError> {
        if start_day_idx + window_days > self.axis.num_days {
            return Err(format!(
                "window [{start_day_idx}, +{window_days}) exceeds the eval axis ({} days)",
                self.axis.num_days
            )
            .into());
        }
        let window = RhoWindow {
            start_day_idx,
            rho_days: window_days,
            window_start: self.axis.start + Duration::days(start_day_idx as i64),
        };
        let batch = self.dataset.collate(&[staid.clone()], &window)?;
        if batch.gauge_staids.len() != 1 {
            return Err(format!(
                "collate returned {} gauges for {staid}; expected exactly 1",
                batch.gauge_staids.len()
            )
            .into());
        }
        if batch.outflow_idx[0].len() != 1 {
            return Err(format!(
                "gauge {staid}: outflow_idx has {} rows; expected 1 (ddr_match=false)",
                batch.outflow_idx[0].len()
            )
            .into());
        }
        Ok(batch)
    }

    /// Forward the batch with the hourly inflow lifted as a gradient leaf.
    ///
    /// `inflow_override` replaces the hourly inflow (validation perturbations);
    /// `None` uses the disaggregation head when the checkpoint carries one,
    /// else the flat repeat-24 `q_prime` — mirroring `forward_eval_core`.
    pub fn forward_with_inflow_leaf(
        &self,
        tensors: &RoutingTensors<AD<I>>,
        inflow_override: Option<Tensor<I, 2>>,
    ) -> LeafForward<I> {
        let device = &self.device;
        let n_active = tensors.adjacency.n;
        let params_map = gather_params_to_subreaches(
            self.head.forward(tensors.spatial_attributes.clone()),
            tensors.adjacency.parent_offset.as_ref(),
            n_active,
            device,
        );
        // Detach every head output: the study differentiates w.r.t. inflow only.
        let detach = |t: Tensor<AD<I>, 1>| Tensor::<AD<I>, 1>::from_inner(t.inner());
        let n_param = detach(params_map.get("n").expect("KAN head missing n").clone());
        let q_param = detach(params_map.get("q_spatial").expect("KAN head missing q_spatial").clone());
        let p_param = params_map.get("p_spatial").cloned().map(detach);
        let (n_param_keep, q_param_keep, p_param_keep) = (n_param.clone(), q_param.clone(), p_param.clone());
        let x_storage: Tensor<AD<I>, 1> = match params_map.get("x_storage") {
            Some(x) => denormalize(
                detach(x.clone()),
                self.cfg.params.parameter_ranges.x_storage,
                self.cfg.params.log_space_parameters.iter().any(|s| s == "x_storage"),
            ),
            None => Tensor::full([n_active], 0.3_f32, device),
        };

        let n_hourly = tensors.q_prime.dims()[0];
        let q_hourly_inner: Tensor<I, 2> = match inflow_override {
            Some(q) => q,
            None => match &self.head.disagg {
                Some(d) => d
                    .forward(tensors.q_prime_daily.clone(), tensors.precip_hourly.clone(), n_hourly)
                    .inner(),
                None => tensors.q_prime.clone().inner(),
            },
        };
        let q_leaf = Tensor::<AD<I>, 2>::from_inner(q_hourly_inner.clone()).require_grad();

        let mut engine = MuskingumCunge::<I>::new(self.cfg.clone(), device.clone());
        engine.setup_inputs(
            RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage },
            q_leaf.clone(),
            SpatialParameters {
                n: n_param,
                q_spatial: q_param,
                p_spatial: p_param,
                k_d: None,
                d_gw: None,
                leakance_factor: None,
                impervious_mask: None,
            },
            false,
            None,
        );
        let runoff = engine.forward(); // (N, T)
        let runoff_inner = runoff.clone().inner();
        let gauge_series = scatter_add_by_group(
            runoff,
            tensors.flat_indices.clone(),
            tensors.group_ids.clone(),
            tensors.num_gauges,
        ); // (1, T)
        // Physical (denormalized) parameters and clamped slope, as the engine
        // uses them, for hydraulic travel-time checks.
        let ranges = &self.cfg.params.parameter_ranges;
        let log = &self.cfg.params.log_space_parameters;
        let n_phys = denormalize(n_param_keep, ranges.n, log.iter().any(|s| s == "n")).inner();
        let q_phys = denormalize(q_param_keep, ranges.q_spatial, log.iter().any(|s| s == "q_spatial")).inner();
        let p_phys = match p_param_keep {
            Some(p) => denormalize(p, ranges.p_spatial, log.iter().any(|s| s == "p_spatial")).inner(),
            None => {
                let d = *self.cfg.params.defaults.get("p_spatial").unwrap_or(&21.0);
                Tensor::<I, 1>::full([n_active], d, device)
            }
        };
        let slope = Tensor::<I, 1>::from_floats(tensors.adjacency.slope.as_slice(), device)
            .clamp_min(self.cfg.params.attribute_minimums.slope);
        let length = Tensor::<I, 1>::from_floats(tensors.adjacency.length_m.as_slice(), device);
        LeafForward { q_leaf, gauge_series, q_hourly_inner, runoff_inner, n_phys, p_phys, q_phys, slope, length }
    }

    /// Daily gauge series under the training convention (`tau` trim + pool).
    pub fn daily(&self, gauge_series: Tensor<AD<I>, 2>) -> Tensor<AD<I>, 2> {
        tau_trim_and_downsample(gauge_series, self.tau)
    }
}

pub struct LeafForward<I: Backend> {
    /// `(T, N)` hourly inflow leaf.
    pub q_leaf: Tensor<AD<I>, 2>,
    /// `(1, T)` routed discharge at the gauge.
    pub gauge_series: Tensor<AD<I>, 2>,
    /// `(T, N)` the inflow actually routed (post-disaggregation), no grad.
    pub q_hourly_inner: Tensor<I, 2>,
    /// `(N, T)` routed discharge at every reach, no grad.
    pub runoff_inner: Tensor<I, 2>,
    /// Denormalized channel parameters and clamped slope / length (`N`).
    pub n_phys: Tensor<I, 1>,
    pub p_phys: Tensor<I, 1>,
    pub q_phys: Tensor<I, 1>,
    pub slope: Tensor<I, 1>,
    pub length: Tensor<I, 1>,
}

/// Gradient of `scalar` (shape `[1]`) w.r.t. the inflow leaf, as a row-major
/// `(T, N)` buffer. Consumes the autodiff graph.
pub fn inflow_gradient<I: Backend>(scalar: Tensor<AD<I>, 1>, q_leaf: &Tensor<AD<I>, 2>) -> Grad {
    let [t, n] = q_leaf.dims();
    let grads = scalar.backward();
    let g = q_leaf
        .grad(&grads)
        .expect("inflow leaf received no gradient — the leaf is not on the tape");
    let values: Vec<f32> = g.into_data().to_vec::<f32>().expect("f32 gradient");
    Grad { t, n, values }
}

/// Row-major `(T, N)` gradient buffer.
#[derive(Debug, Clone)]
pub struct Grad {
    pub t: usize,
    pub n: usize,
    pub values: Vec<f32>,
}

impl Grad {
    #[inline]
    pub fn at(&self, hour: usize, reach: usize) -> f32 {
        self.values[hour * self.n + reach]
    }

    /// Mean over source hours `[from, to)` for every reach.
    pub fn time_mean(&self, from: usize, to: usize) -> Vec<f32> {
        assert!(from < to && to <= self.t, "time_mean range [{from},{to}) outside T={}", self.t);
        let mut out = vec![0.0f32; self.n];
        for h in from..to {
            let row = &self.values[h * self.n..(h + 1) * self.n];
            for (o, v) in out.iter_mut().zip(row) {
                *o += *v;
            }
        }
        let k = (to - from) as f32;
        out.iter_mut().for_each(|v| *v /= k);
        out
    }

    /// Mean over reaches for every source hour.
    pub fn reach_mean(&self) -> Vec<f32> {
        (0..self.t)
            .map(|h| {
                let row = &self.values[h * self.n..(h + 1) * self.n];
                row.iter().sum::<f32>() / self.n as f32
            })
            .collect()
    }

    /// Kernel by daily lag for an anchor hour `t0`: entry `[reach][lag_day]` is
    /// the sum over the 24 source hours `t0 − (24·lag_day + h)`, h in 0..24.
    /// Hours before the window start contribute zero.
    pub fn kernel_daily(&self, t0: usize, lag_days: usize) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0.0f32; lag_days + 1]; self.n];
        for lag in 0..=lag_days {
            for h in 0..24 {
                let back = 24 * lag + h;
                if back > t0 {
                    break;
                }
                let src = t0 - back;
                for reach in 0..self.n {
                    out[reach][lag] += self.at(src, reach);
                }
            }
        }
        out
    }

    /// Kernel by hourly lag summed over reaches, lag in `0..24·lag_days`.
    pub fn kernel_hourly(&self, t0: usize, lag_days: usize) -> Vec<f32> {
        (0..24 * lag_days)
            .map(|back| {
                if back > t0 {
                    return 0.0;
                }
                let src = t0 - back;
                (0..self.n).map(|r| self.at(src, r)).sum()
            })
            .collect()
    }

    /// Per-reach kernel mass and mean lag (days) from the hourly kernel.
    pub fn kernel_moments(&self, t0: usize, lag_days: usize) -> (Vec<f32>, Vec<f32>) {
        let mut mass = vec![0.0f32; self.n];
        let mut first = vec![0.0f32; self.n];
        for back in 0..24 * lag_days {
            if back > t0 {
                break;
            }
            let src = t0 - back;
            for r in 0..self.n {
                let g = self.at(src, r);
                mass[r] += g;
                first[r] += g * back as f32;
            }
        }
        let mean_lag_days: Vec<f32> = mass
            .iter()
            .zip(&first)
            .map(|(m, f)| if m.abs() > 1e-12 { f / m / 24.0 } else { f32::NAN })
            .collect();
        (mass, mean_lag_days)
    }
}

/// Along-channel distance (m) from each reach's outlet to the gauge outlet,
/// on the batch adjacency (`rows` = downstream index, `cols` = upstream index):
/// `dist[gauge] = 0`, `dist[upstream] = dist[downstream] + length_m[downstream]`.
/// Unreachable reaches get NaN.
pub fn dist_to_gauge(adj: &crate::sparse::SparseAdjacency, gauge_row: usize) -> Vec<f32> {
    let n = adj.n;
    let mut upstream_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (r, c) in adj.rows.iter().zip(&adj.cols) {
        let (r, c) = (*r as usize, *c as usize);
        if r != c {
            upstream_of[r].push(c);
        }
    }
    let mut dist = vec![f32::NAN; n];
    dist[gauge_row] = 0.0;
    let mut stack = vec![gauge_row];
    while let Some(d) = stack.pop() {
        for &u in &upstream_of[d] {
            if dist[u].is_nan() {
                dist[u] = dist[d] + adj.length_m[d];
                stack.push(u);
            }
        }
    }
    dist
}

/// Column means of a row-major `(T, N)` buffer.
pub fn column_means(values: &[f32], t: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for h in 0..t {
        for r in 0..n {
            out[r] += values[h * n + r];
        }
    }
    out.iter_mut().for_each(|v| *v /= t.max(1) as f32);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseAdjacency;

    #[test]
    fn kernel_daily_bins_source_hours_by_lag() {
        // T=100, N=2; gradient = 1 everywhere for reach 0, 0 for reach 1.
        let t = 100;
        let n = 2;
        let mut values = vec![0.0f32; t * n];
        for h in 0..t {
            values[h * n] = 1.0;
        }
        let g = Grad { t, n, values };
        let k = g.kernel_daily(80, 2); // lags 0,1,2 → source hours 80..9, all inside the window
        assert_eq!(k[0], vec![24.0, 24.0, 24.0]);
        assert_eq!(k[1], vec![0.0, 0.0, 0.0]);
        // Truncation at the window start: t0=30, lag 1 covers hours 6..=30-24=6 → 24 hours; lag 2 partial.
        let k = g.kernel_daily(30, 2);
        assert_eq!(k[0][0], 24.0);
        assert_eq!(k[0][1], 7.0); // hours 6..=0 → 7 hours
        assert_eq!(k[0][2], 0.0);
    }

    #[test]
    fn dist_to_gauge_walks_upstream() {
        // chain 0 → 1 → 2 (gauge at 2); lengths 100, 200, 300.
        let dense = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let adj = SparseAdjacency::from_dense(3, &dense, vec![100.0, 200.0, 300.0], vec![0.001; 3]);
        let d = dist_to_gauge(&adj, 2);
        assert_eq!(d[2], 0.0);
        assert_eq!(d[1], 300.0);
        assert_eq!(d[0], 500.0);
    }

    #[test]
    fn time_mean_and_moments() {
        let g = Grad { t: 4, n: 1, values: vec![0.0, 2.0, 4.0, 0.0] };
        assert_eq!(g.time_mean(1, 3), vec![3.0]);
        let (mass, lag) = g.kernel_moments(3, 1); // backs 0..24 but t0=3 → hours 3,2,1,0
        assert_eq!(mass, vec![6.0]);
        // first moment: 2*back(1)? hour2 is back 1 → 4*1, hour1 back 2 → 2*2 = 8; /6/24
        assert!((lag[0] - 8.0 / 6.0 / 24.0).abs() < 1e-6);
    }
}
