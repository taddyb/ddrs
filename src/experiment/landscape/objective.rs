//! Per-gauge objective in basin-uniform log-multiplier space.
//!
//! `α = (α_0, α_1, α_2)` scales the trained physical fields of the three AXIS
//! parameters (`LandscapeSpec::axes`, default `(n, p_spatial, q_spatial)`;
//! `gamma` may take a slot on an arm that learns it), `x₀·e^{α}`, clamped to
//! the arm's parameter ranges and re-normalized for the engine. Parameters
//! that are not axes are carried at their trained (or default) field.
//! `L_g(α)` is the NSE-batch loss (training objective) over the configured
//! windows; its gradient comes from autograd with `α` lifted as leaves.
//! Spec: docs/superpowers/specs/2026-09-07-adjoint-landscape-design.md §1–§2.

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use chrono::{Duration, NaiveDate};

use crate::data::dataset::RoutingTensors;
use crate::data::ids::Staid;
use crate::experiment::adjoint::influence::InfluenceContext;
use crate::experiment::BoxError;
use crate::routing::utils::denormalize;
use crate::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use crate::training::forward::{gather_params_to_subreaches, scatter_add_by_group};

type AD<I> = Autodiff<I>;

/// One evaluation window of the objective.
pub struct WindowData<I: Backend> {
    pub start_day: usize,
    pub tensors: RoutingTensors<AD<I>>,
    /// Observed daily series over the window (rho_days), NaN = missing.
    pub obs: Vec<f32>,
    /// Trained physical field of each axis SLOT, in `Objective::axes` order
    /// (inner backend, constant).
    pub x0: [Tensor<I, 1>; 3],
    /// Trained physical fields of the parameters that are NOT axes, by name
    /// (`n`, `p_spatial`, `q_spatial`, and `gamma` when the arm learns it).
    /// `forward_loss` holds these constant.
    pub carried: Vec<(String, Tensor<I, 1>)>,
    pub x_storage: Tensor<I, 1>,
    pub q_prime: Tensor<I, 2>,
    pub slope: Tensor<I, 1>,
    pub length: Tensor<I, 1>,
    pub gauge_row: usize,
    pub comids: Vec<i64>,
}

pub struct Objective<'a, I: Backend> {
    pub ctx: &'a InfluenceContext<I>,
    pub windows: Vec<WindowData<I>>,
    pub sigma: f32,
    pub eps: f32,
    /// Parameter name in each alpha slot (`LandscapeSpec::axes`).
    pub axes: [String; 3],
    pub ranges: [[f32; 2]; 3],
    pub log_space: [bool; 3],
    /// `active[k]` is true when the slot-`k` axis parameter is in the head's
    /// `learnable_parameters` (a real model parameter); false when it is
    /// fixed at `params.defaults` for this arm. See `Objective::active`.
    active: [bool; 3],
    /// The arm learns `gamma` (it is a head output), so the engine must be
    /// handed the field whether or not `gamma` is an axis.
    learns_gamma: bool,
    /// Which per-window loss `forward_loss` builds: "nse-batch", "kge" or
    /// "nse-deriv". Validated against `landscape::VALID_OBJECTIVES` before
    /// this is built.
    objective: String,
    /// `lambda` on the time-derivative term of the `"nse-deriv"` objective.
    /// Ignored (never read) for every other objective, so an `nse-batch` or
    /// `kge` study is byte-identical to one built before this field existed.
    deriv_weight: f32,
}

/// Result of one objective evaluation.
#[derive(Debug, Clone)]
pub struct Eval {
    pub loss: f32,
    /// Per-window NSE of the daily series.
    pub nse: Vec<f32>,
    pub nse_mean: f32,
    /// Per-window KGE of the daily series (always computed, regardless of
    /// `objective`; see `objective::kge`).
    pub kge: Vec<f32>,
    pub kge_mean: f32,
    /// Fraction of (reach, parameter) entries clamped at a range edge, max over windows.
    pub clamped_frac: f32,
    pub grad: Option<[f32; 3]>,
}

/// Per-reach `dL/d ln x` for each axis slot (`Objective::axes` order), length `n_reach` each.
#[derive(Debug, Clone)]
pub struct ReachGrad {
    pub slots: [Vec<f32>; 3],
}

/// Physical fields at some `alpha`, by name, for hydraulic checks.
pub struct PhysFields<I: Backend> {
    pub n: Tensor<I, 1>,
    pub p: Tensor<I, 1>,
    pub q: Tensor<I, 1>,
    /// `Some` only when the arm learns `gamma`.
    pub gamma: Option<Tensor<I, 1>>,
}

/// Every parameter the landscape can hold or perturb.
pub const AXIS_PARAMS: [&str; 4] = ["n", "p_spatial", "q_spatial", "gamma"];

/// `parameter_ranges` entry for an axis parameter.
pub fn param_range(cfg: &crate::config::Config, name: &str) -> [f32; 2] {
    let r = &cfg.params.parameter_ranges;
    match name {
        "n" => r.n,
        "p_spatial" => r.p_spatial,
        "q_spatial" => r.q_spatial,
        "gamma" => r.gamma,
        other => panic!("`{other}` is not a landscape axis parameter; valid: {AXIS_PARAMS:?}"),
    }
}

fn param_log(cfg: &crate::config::Config, name: &str) -> bool {
    cfg.params.log_space_parameters.iter().any(|s| s == name)
}

/// Returned by `Objective::build` when a gauge has zero valid (finite,
/// non-negative) observations in every configured window.
///
/// The run-level gauge filter (`gages_adjacency filter`) only guarantees
/// observation coverage over the full configured training/testing window;
/// a landscape study's `window_days`/`water_year` slice can be much shorter
/// and land entirely in a gap. That is an expected per-gauge data
/// condition, not a bug, so `run_arm` downcasts this error to log a
/// `skipped` line and continue with the rest of the gauge population,
/// instead of the whole arm panicking (`forward_loss`'s old
/// `.expect("at least one window with valid observations")`).
#[derive(Debug)]
pub struct NoValidObservations {
    pub start: NaiveDate,
    pub end: NaiveDate,
}

impl std::fmt::Display for NoValidObservations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no valid observations in the window {} .. {}", self.start, self.end)
    }
}

impl std::error::Error for NoValidObservations {}

impl<'a, I: Backend + 'static> Objective<'a, I>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    pub fn build(
        ctx: &'a InfluenceContext<I>,
        staid: &Staid,
        window_starts: &[usize],
        window_days: usize,
        objective: &str,
        deriv_weight: f32,
        axes: &[String; 3],
    ) -> Result<Self, BoxError> {
        let device = &ctx.device;
        for a in axes {
            if !AXIS_PARAMS.contains(&a.as_str()) {
                return Err(format!("unknown landscape axis `{a}`; valid axes are {AXIS_PARAMS:?}").into());
            }
        }
        if axes[0] == axes[1] || axes[0] == axes[2] || axes[1] == axes[2] {
            return Err(format!("landscape axes must be distinct, got {axes:?}").into());
        }
        let ranges: [[f32; 2]; 3] = std::array::from_fn(|k| param_range(&ctx.cfg, &axes[k]));
        let log_space: [bool; 3] = std::array::from_fn(|k| param_log(&ctx.cfg, &axes[k]));
        let log = &ctx.cfg.params.log_space_parameters;
        let sigma = ctx
            .dataset
            .gauge_obs_std(&[staid.clone()])
            .map_err(|e| format!("gauge std: {e}"))?
            .first()
            .copied()
            .ok_or("gauge std unavailable (loss.kind must be nse-batch in the arm's config)")?;
        let eps = ctx.cfg.experiment.as_ref().map(|e| e.loss.eps).unwrap_or(0.1);
        // Which of (n, p_spatial, q_spatial) is a real model parameter vs
        // fixed at `params.defaults` for this arm. Same source of truth as
        // `InfluenceContext::open`'s learnable/fixed log line, so the two
        // never disagree.
        let section = ctx.cfg.kan_head.as_ref().ok_or("arm config has no kan_head section")?;
        let learns = |name: &str| section.learnable_parameters.iter().any(|s| s == name);
        let learns_gamma = learns("gamma");
        if axes.iter().any(|a| a == "gamma") && !learns_gamma {
            return Err("landscape axis `gamma` needs an arm whose head learns gamma \
                        (`gamma` in kan_head.learnable_parameters); a global \
                        `params.stage_roughness.gamma` is a constant, not a field to perturb."
                .into());
        }
        let active: [bool; 3] = std::array::from_fn(|k| learns(&axes[k]));
        // The routed daily series `forward_loss` compares `obs` against has
        // length `window_days - 2` regardless of `tau`: `n_hourly =
        // (window_days - 1) * 24` (`RhoWindow::n_hourly`,
        // src/data/dates.rs) and `tau_trim_and_downsample` trims exactly 24
        // hours off that for any `tau in [0, 24)` (src/training/loss.rs:46-67).
        // Mirrored here so a gauge with no observations in the window forward
        // pass would actually score is caught before running the (expensive)
        // routing forward, instead of surfacing as a panic deep inside `eval`.
        let d = window_days.saturating_sub(2);
        let mut any_valid_window = false;
        let mut windows = Vec::new();
        for &start in window_starts {
            let batch = ctx.collate_gauge(staid, start, window_days)?;
            let obs: Vec<f32> = batch.observations.column(0).to_vec();
            any_valid_window |= has_valid_observation(&obs, ctx.warmup, d);
            let gauge_row = batch.outflow_idx[0][0];
            let comids: Vec<i64> = batch.divide_comids.iter().map(|c| c.0).collect();
            let tensors = batch.to_tensors::<AD<I>>(device);
            let n_active = tensors.adjacency.n;
            let params_map = gather_params_to_subreaches(
                ctx.head.forward(tensors.spatial_attributes.clone()),
                tensors.adjacency.parent_offset.as_ref(),
                n_active,
                device,
            );
            // Trained field for each of n/p_spatial/q_spatial: denormalized
            // head output when the parameter is in
            // `kan_head.learnable_parameters`, else the constant physical
            // default broadcast over reaches (see `trained_field` and
            // `resolve_default` below). `forward_loss` applies the alpha
            // multiplier and range clamp to this field identically either
            // way, so the landscape over a fixed parameter is still defined.
            let field = |name: &str, range: [f32; 2], log: bool| -> Result<Tensor<I, 1>, BoxError> {
                let head_out = params_map.get(name).cloned().map(|t| t.inner());
                let default = if head_out.is_some() { 0.0 } else { resolve_default(&ctx.cfg.params.defaults, name)? };
                Ok(trained_field(head_out, range, log, default, n_active, device))
            };
            let x0: [Tensor<I, 1>; 3] = [
                field(&axes[0], ranges[0], log_space[0])?,
                field(&axes[1], ranges[1], log_space[1])?,
                field(&axes[2], ranges[2], log_space[2])?,
            ];
            // Everything that is not an axis rides along at its trained (or
            // default) field: the three channel parameters always, gamma
            // only when the arm learns it (otherwise the engine's scalar
            // fallback is the right model, and `gamma` is not a head output).
            let mut carried = Vec::new();
            for name in AXIS_PARAMS {
                if axes.iter().any(|a| a == name) {
                    continue;
                }
                if name == "gamma" && !learns_gamma {
                    continue;
                }
                carried.push((name.to_string(), field(name, param_range(&ctx.cfg, name), param_log(&ctx.cfg, name))?));
            }
            let x_storage = match params_map.get("x_storage") {
                Some(x) => denormalize(
                    x.clone().inner(),
                    ctx.cfg.params.parameter_ranges.x_storage,
                    log.iter().any(|s| s == "x_storage"),
                ),
                None => Tensor::<I, 1>::full([n_active], 0.3_f32, device),
            };
            let n_hourly = tensors.q_prime.dims()[0];
            let q_prime: Tensor<I, 2> = match &ctx.head.disagg {
                Some(d) => d.forward(tensors.q_prime_daily.clone(), tensors.precip_hourly.clone(), n_hourly).inner(),
                None => tensors.q_prime.clone().inner(),
            };
            let slope = Tensor::<I, 1>::from_floats(tensors.adjacency.slope.as_slice(), device)
                .clamp_min(ctx.cfg.params.attribute_minimums.slope);
            let length = Tensor::<I, 1>::from_floats(tensors.adjacency.length_m.as_slice(), device);
            windows.push(WindowData { start_day: start, tensors, obs, x0, carried, x_storage, q_prime, slope, length, gauge_row, comids });
        }
        if !any_valid_window {
            let start = ctx.axis.start + Duration::days(window_starts[0] as i64);
            let end = start + Duration::days(window_days.saturating_sub(1) as i64);
            return Err(Box::new(NoValidObservations { start, end }));
        }
        Ok(Self { ctx, windows, sigma, eps, axes: axes.clone(), ranges, log_space, active, learns_gamma, objective: objective.to_string(), deriv_weight })
    }

    /// Trained physical field of parameter `name` (axis slot or carried);
    /// `None` when the arm has no such field (`gamma` on an arm that does not
    /// learn it).
    pub fn trained_field(&self, w: &WindowData<I>, name: &str) -> Option<Tensor<I, 1>> {
        if let Some(k) = self.axes.iter().position(|a| a == name) {
            return Some(w.x0[k].clone());
        }
        w.carried.iter().find(|(n, _)| n == name).map(|(_, t)| t.clone())
    }

    pub fn n_reach(&self) -> usize {
        self.windows[0].tensors.adjacency.n
    }

    /// `active[k]` for slot `k` of `axes`: true when the parameter is in the
    /// head's `learnable_parameters` and so is a real model parameter; false
    /// when it's fixed at `params.defaults` and must not be treated as an
    /// optimization axis.
    pub fn active(&self) -> [bool; 3] {
        self.active
    }

    /// Physical fields at `α` (inner backend), by name, for hydraulic checks.
    /// Axis parameters are scaled and clamped; carried ones are constant.
    pub fn fields_at(&self, w: &WindowData<I>, alpha: [f32; 3]) -> PhysFields<I> {
        // NB: clamp the FIELD, not the scalar multiplier (precedence).
        let at = |name: &str| -> Option<Tensor<I, 1>> {
            if let Some(k) = self.axes.iter().position(|a| a == name) {
                let r = self.ranges[k];
                return Some((w.x0[k].clone() * alpha[k].exp()).clamp(r[0], r[1]));
            }
            w.carried.iter().find(|(n, _)| n == name).map(|(_, t)| t.clone())
        };
        PhysFields {
            n: at("n").expect("n is always an axis or carried"),
            p: at("p_spatial").expect("p_spatial is always an axis or carried"),
            q: at("q_spatial").expect("q_spatial is always an axis or carried"),
            gamma: at("gamma"),
        }
    }

    /// Normalized (engine-space) field from a physical field with autograd.
    fn normalize_ad(x_phys: Tensor<AD<I>, 1>, range: [f32; 2], log_space: bool) -> Tensor<AD<I>, 1> {
        let [lo, hi] = range;
        if log_space {
            let log_lo = (lo + 1e-6).ln();
            let log_hi = hi.ln();
            (x_phys.log() - log_lo) / (log_hi - log_lo)
        } else {
            (x_phys - lo) / (hi - lo)
        }
    }

    /// Shared forward + loss over all windows, given the three `(n, p, q)`
    /// log-scale leaves. A leaf may be shape `[1]` (basin-uniform, broadcast
    /// via `ones`) or shape `[n_reach]` (per-reach); the multiply-by-`ones`
    /// below broadcasts the former and is a no-op for the latter, so both
    /// callers (`eval`, `reach_grad`) share this one forward path. Returns
    /// the batch-mean loss tensor (still on the autodiff tape, built from
    /// `self.objective`), per-window NSE, per-window KGE (always computed
    /// regardless of `self.objective` -- see `kge`), the max clamped
    /// fraction over windows, and each window's daily gauge series (m3/s;
    /// consumed by `daily_series`, dropped by `eval`).
    fn forward_loss(&self, leaves: &[Tensor<AD<I>, 1>; 3]) -> (Tensor<AD<I>, 1>, Vec<f32>, Vec<f32>, f32, Vec<Vec<f32>>) {
        let device = &self.ctx.device;
        let n = self.n_reach();
        let ones = Tensor::<AD<I>, 1>::ones([n], device);
        let mut total: Option<Tensor<AD<I>, 1>> = None;
        let mut nses = Vec::new();
        let mut kges = Vec::new();
        let mut dailies: Vec<Vec<f32>> = Vec::new();
        let mut clamped_max = 0.0f32;
        for w in &self.windows {
            // Normalized (engine-space) field for every parameter: the alpha
            // slots are scaled, clamped and re-normalized; carried ones are
            // constant.
            let mut norm: Vec<(String, Tensor<AD<I>, 1>)> = Vec::new();
            for (k, x0) in w.x0.iter().enumerate() {
                let scale = leaves[k].clone().exp() * ones.clone(); // [n]
                let phys = Tensor::<AD<I>, 1>::from_inner(x0.clone()) * scale;
                let [lo, hi] = self.ranges[k];
                // clamped fraction (diagnostic)
                let v: Vec<f32> = phys.clone().inner().into_data().to_vec::<f32>().unwrap();
                let cf = v.iter().filter(|x| **x <= lo || **x >= hi).count() as f32 / n as f32;
                clamped_max = clamped_max.max(cf);
                let phys = phys.clamp(lo, hi);
                norm.push((self.axes[k].clone(), Self::normalize_ad(phys, self.ranges[k], self.log_space[k])));
            }
            for (name, x0) in &w.carried {
                let phys = Tensor::<AD<I>, 1>::from_inner(x0.clone());
                norm.push((name.clone(), Self::normalize_ad(phys, param_range(&self.ctx.cfg, name), param_log(&self.ctx.cfg, name))));
            }
            let get = |name: &str| norm.iter().find(|(nm, _)| nm == name).map(|(_, t)| t.clone());
            let mut engine = MuskingumCunge::<I>::new(self.ctx.cfg.clone(), device.clone());
            engine.setup_inputs(
                RoutingInputs { adjacency: w.tensors.adjacency.clone(), x_storage: Tensor::from_inner(w.x_storage.clone()) },
                Tensor::<AD<I>, 2>::from_inner(w.q_prime.clone()),
                SpatialParameters {
                    n: get("n").expect("n field"),
                    q_spatial: get("q_spatial").expect("q_spatial field"),
                    p_spatial: Some(get("p_spatial").expect("p_spatial field")),
                    k_d: None,
                    d_gw: None,
                    leakance_factor: None,
                    impervious_mask: None,
                    gamma: if self.learns_gamma { Some(get("gamma").expect("gamma field")) } else { None },
                },
                false,
                None,
            );
            let runoff = engine.forward();
            let gauge = scatter_add_by_group(runoff, w.tensors.flat_indices.clone(), w.tensors.group_ids.clone(), w.tensors.num_gauges);
            let daily = self.ctx.daily(gauge); // (1, D)
            let d = daily.dims()[1];
            let dv: Vec<f32> = daily.clone().inner().into_data().to_vec::<f32>().unwrap();
            dailies.push(dv.clone());
            let valid: Vec<usize> = (self.ctx.warmup..d).filter(|&i| i < w.obs.len() && w.obs[i].is_finite() && w.obs[i] >= 0.0).collect();
            if valid.is_empty() {
                nses.push(f32::NAN);
                kges.push(f32::NAN);
                continue;
            }
            // NSE + KGE (diagnostics, both always computed)
            let om: f32 = valid.iter().map(|&i| w.obs[i]).sum::<f32>() / valid.len() as f32;
            let sse: f32 = valid.iter().map(|&i| (dv[i] - w.obs[i]).powi(2)).sum();
            let sst: f32 = valid.iter().map(|&i| (w.obs[i] - om).powi(2)).sum();
            nses.push(if sst > 0.0 { 1.0 - sse / sst } else { f32::NAN });
            let sim_valid: Vec<f32> = valid.iter().map(|&i| dv[i]).collect();
            let obs_valid: Vec<f32> = valid.iter().map(|&i| w.obs[i]).collect();
            kges.push(kge(&sim_valid, &obs_valid));

            let loss_w = match self.objective.as_str() {
                "kge" => {
                    // Differentiable `1 - KGE` restricted to the valid days:
                    // gather them with `select` into a (1, n_valid) tensor
                    // and reuse `training::loss::nnse_kge_loss`'s KGE term
                    // (nnse_weight 0, kge_weight 1) -- it already implements
                    // this exact formula on (G, T) tensors, and (1, n_valid)
                    // is the same tensor shape family.
                    let idx_i32: Vec<i32> = valid.iter().map(|&i| i as i32).collect();
                    let idx = Tensor::<AD<I>, 1, Int>::from_data(TensorData::from(idx_i32.as_slice()), device);
                    let p_valid = daily.clone().select(1, idx);
                    let o_valid = Tensor::<AD<I>, 1>::from_floats(obs_valid.as_slice(), device).reshape([1, valid.len()]);
                    crate::training::loss::nnse_kge_loss(p_valid, o_valid, 0.0, 1.0, self.eps)
                }
                _ => {
                    // "nse-batch" (the `_` arm proper) and "nse-deriv",
                    // which is the same level term plus `lambda` times the
                    // time-derivative term. At `lambda = 0` — or on a window
                    // with too few adjacent valid pairs for `sigma_d` —
                    // `nse-deriv` reduces to exactly `nse-batch`.
                    let level = nse_batch_window_loss::<AD<I>>(daily.clone(), &w.obs, &valid, self.sigma, self.eps, device);
                    if self.objective == crate::experiment::landscape::OBJECTIVE_NSE_DERIV {
                        match deriv_window_loss::<AD<I>>(daily.clone(), &w.obs, &valid, self.eps, device) {
                            Some(deriv) => level + deriv.mul_scalar(self.deriv_weight),
                            None => level,
                        }
                    } else {
                        level
                    }
                }
            };
            total = Some(match total {
                Some(t) => t + loss_w,
                None => loss_w,
            });
        }
        let n_w = nses.iter().filter(|v| v.is_finite()).count().max(1) as f32;
        let total = total.expect("at least one window with valid observations");
        (total / n_w, nses, kges, clamped_max, dailies)
    }

    /// Evaluate `L_g(α)` (mean over windows) and optionally its gradient.
    ///
    /// The alpha leaves always get `require_grad()` and `total.backward()` is
    /// always called below, even when `with_grad` is false -- the gradient
    /// is simply not copied into `Eval.grad` in that case. This is not for
    /// the gradient value: it's because a routing forward on the Autodiff
    /// backend keeps its whole per-timestep tape alive until a `.backward()`
    /// consumes it, even when no tensor is tracked. `MuskingumCunge<I>` only
    /// runs on `Autodiff<I>` (there is no tape-free forward path), so a
    /// forward-only `eval` leaked tape memory every call; the landscape
    /// slice-grid loop (`src/experiment/landscape/mod.rs::run_gauge`) calls
    /// `eval(_, with_grad=false)` thousands of times per gauge, which grew a
    /// run from 31 GB to 77 GB RSS before this fix. See
    /// `examples/leak_probe.rs` for the isolated repro (`ad-track-nobackward`
    /// leaks, `ad-track-backward` does not).
    pub fn eval(&self, alpha: [f32; 3], with_grad: bool) -> Eval {
        let device = &self.ctx.device;
        // α leaves (shape [1]) — one set shared across windows. Always
        // require_grad so the tape-releasing backward below has leaves to
        // walk back to.
        let leaves: [Tensor<AD<I>, 1>; 3] =
            std::array::from_fn(|k| Tensor::<AD<I>, 1>::from_floats([alpha[k]].as_slice(), device).require_grad());
        let (total, nses, kges, clamped_max, _dailies) = self.forward_loss(&leaves);
        let loss: f32 = total.clone().inner().into_data().to_vec::<f32>().unwrap()[0];
        let grads = total.backward();
        let grad = if with_grad {
            let mut g: Vec<f32> = leaves
                .iter()
                .map(|l| l.grad(&grads).map(|t| t.into_data().to_vec::<f32>().unwrap()[0]).unwrap_or(0.0))
                .collect();
            // A fixed parameter's leaf still multiplies into the forward
            // pass (the alpha-scaled constant default field), so autograd
            // would otherwise report a nonzero derivative along an axis that
            // is not a model parameter. Force it to exactly 0.0 so Newton,
            // the Hessian, and every downstream consumer treat that axis as
            // never moving.
            for k in 0..3 {
                if !self.active[k] {
                    g[k] = 0.0;
                }
            }
            Some([g[0], g[1], g[2]])
        } else {
            None
        };
        let nse_mean = {
            let f: Vec<f32> = nses.iter().copied().filter(|v| v.is_finite()).collect();
            if f.is_empty() { f32::NAN } else { f.iter().sum::<f32>() / f.len() as f32 }
        };
        let kge_mean = {
            let f: Vec<f32> = kges.iter().copied().filter(|v| v.is_finite()).collect();
            if f.is_empty() { f32::NAN } else { f.iter().sum::<f32>() / f.len() as f32 }
        };
        Eval { loss, nse: nses, nse_mean, kge: kges, kge_mean, clamped_frac: clamped_max, grad }
    }

    /// Routed daily discharge (m3/s) at the gauge, window 0, at `alpha`. Runs
    /// its own forward + backward pass (`backward()` is required even though
    /// the gradient is discarded, to release the routing tape -- see the doc
    /// comment on `eval`). For the series output (`landscape.series: true`)
    /// only; not cached against `eval`'s calls at the same `alpha`.
    pub fn daily_series(&self, alpha: [f32; 3]) -> Vec<f32> {
        let device = &self.ctx.device;
        let leaves: [Tensor<AD<I>, 1>; 3] =
            std::array::from_fn(|k| Tensor::<AD<I>, 1>::from_floats([alpha[k]].as_slice(), device).require_grad());
        let (total, _nses, _kges, _clamped_max, dailies) = self.forward_loss(&leaves);
        let _ = total.backward();
        dailies.into_iter().next().expect("at least one window")
    }

    /// Per-reach `g_i = dL/d ln x_i` for each axis slot at
    /// `α`: same forward as `eval`, but each reach gets its own log-scale
    /// leaf (initialized to `α`, broadcast) instead of one shared scalar.
    /// Where the field is clamped at a range edge the local gradient is
    /// zero (the clamp backward is exact); that's a correct answer, not a
    /// bug: the gauge cannot see past the boundary at that reach.
    pub fn reach_grad(&self, alpha: [f32; 3]) -> ReachGrad {
        let device = &self.ctx.device;
        let n = self.n_reach();
        let leaves: [Tensor<AD<I>, 1>; 3] =
            std::array::from_fn(|k| Tensor::<AD<I>, 1>::from_floats(vec![alpha[k]; n].as_slice(), device).require_grad());
        let (total, _nses, _kges, _clamped_max, _dailies) = self.forward_loss(&leaves);
        let grads = total.backward();
        let extract = |l: &Tensor<AD<I>, 1>| -> Vec<f32> {
            l.grad(&grads).map(|t| t.into_data().to_vec::<f32>().unwrap()).unwrap_or_else(|| vec![0.0; n])
        };
        ReachGrad { slots: [extract(&leaves[0]), extract(&leaves[1]), extract(&leaves[2])] }
    }

    /// Central-difference Hessian of `L_g` at `α` (symmetrized), step `h`.
    /// Only active components are perturbed: a fixed parameter's column
    /// stays 0 (never finite-differenced), and its row stays 0 too since
    /// `eval` forces that gradient component to 0.0 for every trial point.
    pub fn hessian(&self, alpha: [f32; 3], h: f32) -> [[f32; 3]; 3] {
        let mut hm = [[0.0f32; 3]; 3];
        for k in 0..3 {
            if !self.active[k] {
                continue;
            }
            let mut ap = alpha;
            ap[k] += h;
            let mut am = alpha;
            am[k] -= h;
            let gp = self.eval(ap, true).grad.unwrap();
            let gm = self.eval(am, true).grad.unwrap();
            for j in 0..3 {
                hm[j][k] = (gp[j] - gm[j]) / (2.0 * h);
            }
        }
        // symmetrize
        let mut s = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] = 0.5 * (hm[i][j] + hm[j][i]);
            }
        }
        s
    }
}

/// NSE-batch level loss for one window: `sum_i w_i (sim_i - obs_i)^2 /
/// (sigma + eps)^2` with `w_i = 1/n_valid` on the valid days and `0`
/// elsewhere. `daily` is `(1, d)`; masking is a dense elementwise multiply
/// rather than a gather so the whole thing stays differentiable in `daily`.
/// Pure tensor algebra — no I/O — so it's unit-tested below without a
/// trained checkpoint.
fn nse_batch_window_loss<B: Backend>(
    daily: Tensor<B, 2>,
    obs: &[f32],
    valid: &[usize],
    sigma: f32,
    eps: f32,
    device: &B::Device,
) -> Tensor<B, 1> {
    let d = daily.dims()[1];
    let mut wts = vec![0.0f32; d];
    let mut obs_f = vec![0.0f32; d];
    for &i in valid {
        wts[i] = 1.0 / valid.len() as f32;
        obs_f[i] = obs[i];
    }
    let w_t = Tensor::<B, 1>::from_floats(wts.as_slice(), device).reshape([1, d]);
    let o_t = Tensor::<B, 1>::from_floats(obs_f.as_slice(), device).reshape([1, d]);
    let denom = (sigma + eps) * (sigma + eps);
    let sq = (daily - o_t).powf_scalar(2.0) * w_t;
    sq.sum() / denom
}

/// Indices `j` in `0..d-1` where BOTH day `j` and day `j+1` are valid — the
/// pair set of the `"nse-deriv"` derivative term. Derived from the same
/// `valid` day list the level term uses, so the two terms can never
/// disagree about which days count. With non-contiguous valid days (a NaN
/// in the middle of the window) this is strictly fewer than
/// `valid.len() - 1` pairs. Pure — unit-tested below.
fn adjacent_valid_pairs(valid: &[usize], d: usize) -> Vec<usize> {
    let mut is_valid = vec![false; d];
    for &i in valid {
        is_valid[i] = true;
    }
    (0..d.saturating_sub(1)).filter(|&j| is_valid[j] && is_valid[j + 1]).collect()
}

/// Time-derivative term for one window: `sum_j wd_j (dsim_j - dobs_j)^2 /
/// (sigma_d + eps)^2` over the ADJACENT valid pairs `j` (both `j` and `j+1`
/// in `valid`), with `wd_j = 1/n_pairs` and `dx_j = x_{j+1} - x_j` on the
/// daily series. `sigma_d` is the population standard deviation of the
/// OBSERVED differences over those pairs — observations only, so it is
/// constant in `alpha` and the landscape stays a fixed objective (same
/// convention as `dataset::gauge_obs_std`, which divides by `n`).
///
/// Returns `None` — meaning "contribute only the level term for this
/// window", never a NaN — when the window has fewer than two valid pairs or
/// `sigma_d` is not finite and positive.
///
/// Like `nse_batch_window_loss` the masking is dense (`(1, d-1)` weight and
/// observed-difference vectors, zero at invalid pairs) so no gather is
/// needed and the term is differentiable in `daily`.
fn deriv_window_loss<B: Backend>(
    daily: Tensor<B, 2>,
    obs: &[f32],
    valid: &[usize],
    eps: f32,
    device: &B::Device,
) -> Option<Tensor<B, 1>> {
    let d = daily.dims()[1];
    if d < 2 {
        return None;
    }
    let pairs = adjacent_valid_pairs(valid, d);
    if pairs.len() < 2 {
        return None;
    }
    let dobs: Vec<f32> = pairs.iter().map(|&j| obs[j + 1] - obs[j]).collect();
    let mean_d = dobs.iter().sum::<f32>() / pairs.len() as f32;
    let sigma_d = (dobs.iter().map(|v| (v - mean_d).powi(2)).sum::<f32>() / pairs.len() as f32).sqrt();
    if !sigma_d.is_finite() || sigma_d <= 0.0 {
        return None;
    }
    let mut wts = vec![0.0f32; d - 1];
    let mut obs_d = vec![0.0f32; d - 1];
    for (&j, &o) in pairs.iter().zip(dobs.iter()) {
        wts[j] = 1.0 / pairs.len() as f32;
        obs_d[j] = o;
    }
    let w_t = Tensor::<B, 1>::from_floats(wts.as_slice(), device).reshape([1, d - 1]);
    let o_t = Tensor::<B, 1>::from_floats(obs_d.as_slice(), device).reshape([1, d - 1]);
    let dsim = daily.clone().slice([0..1, 1..d]) - daily.slice([0..1, 0..d - 1]);
    let denom = (sigma_d + eps) * (sigma_d + eps);
    Some(((dsim - o_t).powf_scalar(2.0) * w_t).sum() / denom)
}

/// True when `obs[i]` is finite and non-negative for at least one `i` in
/// `warmup..d` (also bounded by `obs.len()`) -- the same predicate
/// `forward_loss` uses to build its per-window `valid` day list. Used by
/// `Objective::build` to detect, before running any routing forward, a
/// gauge/window pair with zero valid observations (see `NoValidObservations`).
/// Pure -- no I/O -- so it's unit-tested below without a trained checkpoint.
fn has_valid_observation(obs: &[f32], warmup: usize, d: usize) -> bool {
    (warmup..d).any(|i| i < obs.len() && obs[i].is_finite() && obs[i] >= 0.0)
}

/// Trained physical field for one channel parameter, length `n_active`.
/// `Some(head_out)` denormalizes the KAN head's `[0,1]` output to the
/// physical range (the parameter is in `kan_head.learnable_parameters`);
/// `None` broadcasts the constant `default` — the field is the same value at
/// every reach, but the alpha multiplier and range clamp in `forward_loss`
/// still apply to it exactly as for a learned field, so the landscape over a
/// fixed parameter stays defined. Pure — no I/O — so it's unit-tested below
/// without a trained checkpoint.
fn trained_field<I: Backend>(
    head_out: Option<Tensor<I, 1>>,
    range: [f32; 2],
    log_space: bool,
    default: f32,
    n_active: usize,
    device: &I::Device,
) -> Tensor<I, 1> {
    match head_out {
        Some(v) => denormalize(v, range, log_space),
        None => Tensor::<I, 1>::full([n_active], default, device),
    }
}

/// Constant physical default for a parameter absent from
/// `kan_head.learnable_parameters`. `p_spatial` mirrors the routing engine's
/// own fallback (`Params::default()` seeds `defaults["p_spatial"] = 21.0`,
/// and `mmc.rs::setup_inputs` falls back to that same key when its own
/// `p_spatial` input is `None`); `n` and `q_spatial` have no established
/// "fixed" convention elsewhere in this codebase (both are hard-required in
/// `training/forward.rs`), so a config that leaves either non-learnable
/// without a matching `params.defaults.<name>` entry is a config error, not
/// a silent guess.
fn resolve_default(defaults: &std::collections::HashMap<String, f32>, name: &str) -> Result<f32, BoxError> {
    match defaults.get(name) {
        Some(&v) => Ok(v),
        None if name == "p_spatial" => Ok(21.0),
        None => Err(format!(
            "parameter `{name}` is not in kan_head.learnable_parameters and has no params.defaults.{name} entry"
        )
        .into()),
    }
}

/// Kling-Gupta Efficiency of `sim` vs `obs` (population moments; equal
/// length; caller filters NaNs/invalid days beforehand -- see the `valid`
/// day list in `forward_loss`).
///
/// `KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)`, with `r` the
/// Pearson correlation, `alpha = std(sim)/std(obs)`, `beta =
/// mean(sim)/mean(obs)`. This is the diagnostic metric (reported as
/// `kge0`/`kge_star`), always computed regardless of `LandscapeSpec::objective`;
/// the differentiable training-objective path (`forward_loss`'s `"kge"` arm)
/// instead reuses `training::loss::nnse_kge_loss`'s eps-regularized tensor
/// formula. Returns `NaN` when `obs` has zero variance or zero mean (the
/// ratios are degenerate), mirroring the NSE diagnostic's `sst > 0.0` guard
/// in `forward_loss`.
pub fn kge(sim: &[f32], obs: &[f32]) -> f32 {
    assert_eq!(sim.len(), obs.len(), "kge: sim and obs must be the same length");
    let n = sim.len() as f32;
    if n == 0.0 {
        return f32::NAN;
    }
    let mean_s = sim.iter().sum::<f32>() / n;
    let mean_o = obs.iter().sum::<f32>() / n;
    let var_o = obs.iter().map(|v| (v - mean_o).powi(2)).sum::<f32>() / n;
    if var_o <= 0.0 || mean_o == 0.0 {
        return f32::NAN;
    }
    let var_s = sim.iter().map(|v| (v - mean_s).powi(2)).sum::<f32>() / n;
    let cov = sim.iter().zip(obs).map(|(&s, &o)| (s - mean_s) * (o - mean_o)).sum::<f32>() / n;
    let std_s = var_s.sqrt();
    let std_o = var_o.sqrt();
    let r = cov / (std_s * std_o);
    let alpha = std_s / std_o;
    let beta = mean_s / mean_o;
    1.0 - ((r - 1.0).powi(2) + (alpha - 1.0).powi(2) + (beta - 1.0).powi(2)).sqrt()
}

/// Jacobi eigen-decomposition of a symmetric 3×3: returns (eigenvalues desc, eigenvectors as columns).
pub fn eig3(a: [[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut a = [[a[0][0] as f64, a[0][1] as f64, a[0][2] as f64], [a[1][0] as f64, a[1][1] as f64, a[1][2] as f64], [a[2][0] as f64, a[2][1] as f64, a[2][2] as f64]];
    let mut v = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..60 {
        let mut p = 0;
        let mut q = 1;
        let mut mx = a[0][1].abs();
        for (i, j) in [(0, 2), (1, 2)] {
            if a[i][j].abs() > mx {
                mx = a[i][j].abs();
                p = i;
                q = j;
            }
        }
        if mx < 1e-12 {
            break;
        }
        let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        for k in 0..3 {
            let akp = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let apk = a[p][k];
            let aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for k in 0..3 {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| a[j][j].partial_cmp(&a[i][i]).unwrap());
    let vals = [a[idx[0]][idx[0]] as f32, a[idx[1]][idx[1]] as f32, a[idx[2]][idx[2]] as f32];
    let mut vecs = [[0.0f32; 3]; 3];
    for (c, &i) in idx.iter().enumerate() {
        for r in 0..3 {
            vecs[r][c] = v[r][i] as f32;
        }
    }
    (vals, vecs)
}

/// `solve3` restricted to the `active` sub-block: each inactive row/column
/// of `a` is zeroed with 1 on its diagonal and 0 in `b`, so the corresponding
/// component of the returned step is exactly 0 regardless of `a` and `b`
/// there. Reduces to a plain `solve3` call when all three components are
/// active.
pub fn solve_active(mut a: [[f32; 3]; 3], mut b: [f32; 3], active: [bool; 3]) -> Option<[f32; 3]> {
    for k in 0..3 {
        if !active[k] {
            for j in 0..3 {
                a[k][j] = 0.0;
                a[j][k] = 0.0;
            }
            a[k][k] = 1.0;
            b[k] = 0.0;
        }
    }
    solve3(a, b)
}

/// `eig3` restricted to the `active` sub-block. With one component fixed,
/// its row/column is zeroed and its diagonal set to a sentinel far below any
/// realistic curvature value, which decouples it from the real 2x2 active
/// block and (because `eig3` sorts descending) always sorts it last. That
/// slot is then overwritten with `NaN` / the fixed axis's own unit vector,
/// per the module's placed-last convention. All-active input is unchanged
/// (delegates straight to `eig3`).
pub fn eig_active(mut a: [[f32; 3]; 3], active: [bool; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let Some(i) = active.iter().position(|&x| !x) else {
        return eig3(a);
    };
    const SENTINEL: f32 = -1e30;
    for j in 0..3 {
        a[i][j] = 0.0;
        a[j][i] = 0.0;
    }
    a[i][i] = SENTINEL;
    let (mut vals, mut vecs) = eig3(a);
    vals[2] = f32::NAN;
    for r in 0..3 {
        vecs[r][2] = if r == i { 1.0 } else { 0.0 };
    }
    (vals, vecs)
}

/// Levenberg-Marquardt-damped Newton direction `d = -(H + mu*I)^-1 g`,
/// restricted to `active` components, capped to length `step_cap` (active
/// components only), with a steepest-descent fallback.
///
/// For small basins the Hessian `H` is nearly flat or indefinite, so the raw
/// Newton direction can be hundreds of log units long; every backtracking
/// trial in `run_gauge`'s line search then lands on the same clamped box
/// corner, is rejected on `clamped_frac`, and the search reports
/// `hit_range_bound` with zero iterations even though the gradient is not
/// small. Capping the step length (rather than clamping to the box before
/// scaling) fixes this: the direction is preserved, only its length is
/// bounded, so backtracking can still find an acceptable interior point.
///
/// If the damped Newton step is not itself a descent direction (`dot(d,
/// grad) >= 0` over active components) -- possible when `H + mu*I` is still
/// indefinite on the active sub-block -- or the damped system is singular,
/// `d` is replaced by the steepest-descent direction `-grad` (active
/// components only), rescaled to `step_cap`. Fixed (non-active) components
/// are always exactly 0.
///
/// Also reports whether the fallback was used, so callers can log it and
/// record it in `LandscapeResult::used_gradient_fallback`.
pub fn newton_step_capped(hess: [[f32; 3]; 3], grad: [f32; 3], mu: f32, active: [bool; 3], step_cap: f32) -> ([f32; 3], bool) {
    let mut hd = hess;
    for k in 0..3 {
        hd[k][k] += mu;
    }
    let neg_g = [-grad[0], -grad[1], -grad[2]];
    let dot_active = |d: &[f32; 3]| -> f32 { (0..3).filter(|&k| active[k]).map(|k| d[k] * grad[k]).sum() };
    let norm_active = |d: &[f32; 3]| -> f32 { (0..3).filter(|&k| active[k]).map(|k| d[k] * d[k]).sum::<f32>().sqrt() };
    let rescale_to = |mut d: [f32; 3], target: f32| -> [f32; 3] {
        let n = norm_active(&d);
        if n > 1e-12 {
            let s = target / n;
            for v in d.iter_mut() {
                *v *= s;
            }
        }
        d
    };
    if let Some(d) = solve_active(hd, neg_g, active) {
        if dot_active(&d) < 0.0 {
            let n = norm_active(&d);
            let d = if n > step_cap { rescale_to(d, step_cap) } else { d };
            return (d, false);
        }
    }
    let mut steepest = [0.0f32; 3];
    for k in 0..3 {
        if active[k] {
            steepest[k] = -grad[k];
        }
    }
    (rescale_to(steepest, step_cap), true)
}

/// `newton_step_capped` without the fallback flag; the pure function used
/// for unit tests below.
pub fn newton_direction(hess: [[f32; 3]; 3], grad: [f32; 3], mu: f32, active: [bool; 3], step_cap: f32) -> [f32; 3] {
    newton_step_capped(hess, grad, mu, active, step_cap).0
}

/// Solve the 3×3 system `a x = b` by Gaussian elimination with partial pivoting.
pub fn solve3(a: [[f32; 3]; 3], b: [f32; 3]) -> Option<[f32; 3]> {
    let mut m = [[a[0][0] as f64, a[0][1] as f64, a[0][2] as f64, b[0] as f64], [a[1][0] as f64, a[1][1] as f64, a[1][2] as f64, b[1] as f64], [a[2][0] as f64, a[2][1] as f64, a[2][2] as f64, b[2] as f64]];
    for c in 0..3 {
        let piv = (c..3).max_by(|&i, &j| m[i][c].abs().partial_cmp(&m[j][c].abs()).unwrap())?;
        if m[piv][c].abs() < 1e-14 {
            return None;
        }
        m.swap(c, piv);
        for r in 0..3 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..4 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    Some([(m[0][3] / m[0][0]) as f32, (m[1][3] / m[1][1]) as f32, (m[2][3] / m[2][2]) as f32])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use std::collections::HashMap;

    type TestBackend = NdArray<f32>;

    #[test]
    fn trained_field_denormalizes_head_output_when_present() {
        let device = Default::default();
        // head output 0.5 in [0,1] over range [0.0, 10.0], linear space -> 5.0.
        let head_out = Tensor::<TestBackend, 1>::from_floats([0.0f32, 0.5, 1.0].as_slice(), &device);
        let f = trained_field::<TestBackend>(Some(head_out), [0.0, 10.0], false, 999.0, 3, &device);
        let v: Vec<f32> = f.into_data().to_vec::<f32>().unwrap();
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[1] - 5.0).abs() < 1e-6);
        assert!((v[2] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn trained_field_broadcasts_default_when_absent() {
        let device = Default::default();
        let f = trained_field::<TestBackend>(None, [0.0, 10.0], false, 21.0, 4, &device);
        let v: Vec<f32> = f.into_data().to_vec::<f32>().unwrap();
        assert_eq!(v, vec![21.0, 21.0, 21.0, 21.0]);
    }

    /// Scalar value of a `(1,)` loss tensor.
    fn scalar(t: Tensor<TestBackend, 1>) -> f32 {
        t.into_data().to_vec::<f32>().unwrap()[0]
    }

    /// Critical regression guard: at `lambda = 0` the `nse-deriv` objective
    /// must be numerically identical to `nse-batch` on the same input, so a
    /// bundle can be re-run with the new objective and a zero weight and
    /// reproduce the existing landscape exactly.
    #[test]
    fn deriv_term_at_zero_weight_is_identical_to_nse_batch() {
        let device = Default::default();
        let obs = [1.0f32, 2.0, 4.0, 4.0, 3.0, 5.0];
        let sim = [1.5f32, 2.0, 3.0, 5.0, 3.5, 4.0];
        let valid: Vec<usize> = (0..obs.len()).collect();
        let daily = Tensor::<TestBackend, 1>::from_floats(sim.as_slice(), &device).reshape([1, obs.len()]);
        let level = scalar(nse_batch_window_loss::<TestBackend>(daily.clone(), &obs, &valid, 2.0, 0.1, &device));
        let deriv = deriv_window_loss::<TestBackend>(daily, &obs, &valid, 0.1, &device).expect("valid pairs exist");
        let total = level + scalar(deriv) * 0.0;
        assert_eq!(total, level, "lambda = 0 must reproduce nse-batch exactly");
    }

    /// Hand-computed window (sigma = 2, eps = 0, lambda = 0.5):
    ///   sim  = [1.5, 2.0, 3.0, 5.0], obs = [1.0, 2.0, 4.0, 4.0]
    ///   L_nse   = mean(0.25, 0, 1, 1) / 2^2 = 0.5625 / 4 = 0.140625
    ///   dsim = [0.5, 1.0, 2.0], dobs = [1.0, 2.0, 0.0]
    ///   mean(dobs) = 1, sigma_d^2 = (0 + 1 + 1)/3 = 2/3
    ///   L_deriv = mean(0.25, 1, 4) / (2/3) = 1.75 * 1.5 = 2.625
    ///   L       = 0.140625 + 0.5 * 2.625 = 1.453125
    #[test]
    fn deriv_objective_matches_hand_computed_window() {
        let device = Default::default();
        let obs = [1.0f32, 2.0, 4.0, 4.0];
        let sim = [1.5f32, 2.0, 3.0, 5.0];
        let valid: Vec<usize> = (0..obs.len()).collect();
        let daily = Tensor::<TestBackend, 1>::from_floats(sim.as_slice(), &device).reshape([1, obs.len()]);
        let level = scalar(nse_batch_window_loss::<TestBackend>(daily.clone(), &obs, &valid, 2.0, 0.0, &device));
        let deriv = scalar(deriv_window_loss::<TestBackend>(daily, &obs, &valid, 0.0, &device).expect("3 valid pairs"));
        assert!((level - 0.140625).abs() < 1e-5, "L_nse = {level}");
        assert!((deriv - 2.625).abs() < 1e-5, "L_deriv = {deriv}");
        let total = level + 0.5 * deriv;
        assert!((total - 1.453125).abs() < 1e-5, "L = {total}");
    }

    /// A NaN in the middle breaks the pair chain: 5 valid days with a gap
    /// give 3 ADJACENT pairs, not `n_valid - 1 = 4`.
    #[test]
    fn pair_set_counts_adjacent_valid_days_only() {
        let obs = [1.0f32, 2.0, f32::NAN, 4.0, 5.0, 6.0];
        let valid: Vec<usize> = (0..obs.len()).filter(|&i| obs[i].is_finite()).collect();
        assert_eq!(valid, vec![0, 1, 3, 4, 5]);
        let pairs = adjacent_valid_pairs(&valid, obs.len());
        assert_eq!(pairs, vec![0, 3, 4]);
        assert_eq!(pairs.len(), 3, "must not be n_valid - 1 = 4");
    }

    /// Fewer than two adjacent valid pairs (here: one) leaves `sigma_d`
    /// undefined, so the window contributes only its level term instead of
    /// a NaN.
    #[test]
    fn deriv_term_is_skipped_when_too_few_pairs() {
        let device = Default::default();
        let obs = [1.0f32, 2.0, f32::NAN, 4.0];
        let sim = [1.5f32, 2.0, 3.0, 5.0];
        let valid = vec![0usize, 1, 3];
        let daily = Tensor::<TestBackend, 1>::from_floats(sim.as_slice(), &device).reshape([1, obs.len()]);
        assert!(deriv_window_loss::<TestBackend>(daily, &obs, &valid, 0.1, &device).is_none());
    }

    /// Constant observed differences give `sigma_d = 0`, which would divide
    /// by `eps^2` (or by zero at `eps = 0`); the term is skipped instead.
    #[test]
    fn deriv_term_is_skipped_when_sigma_d_is_zero() {
        let device = Default::default();
        let obs = [1.0f32, 2.0, 3.0, 4.0];
        let sim = [1.5f32, 2.0, 3.0, 5.0];
        let valid: Vec<usize> = (0..obs.len()).collect();
        let daily = Tensor::<TestBackend, 1>::from_floats(sim.as_slice(), &device).reshape([1, obs.len()]);
        assert!(deriv_window_loss::<TestBackend>(daily, &obs, &valid, 0.1, &device).is_none());
    }

    #[test]
    fn has_valid_observation_true_when_one_day_is_finite_and_nonnegative() {
        let obs = vec![f32::NAN, f32::NAN, 3.5, f32::NAN];
        assert!(has_valid_observation(&obs, 0, obs.len()));
    }

    #[test]
    fn has_valid_observation_false_when_every_day_is_nan() {
        let obs = vec![f32::NAN; 5];
        assert!(!has_valid_observation(&obs, 0, obs.len()));
    }

    #[test]
    fn has_valid_observation_false_when_valid_day_is_outside_the_warmup_bound() {
        // The only finite day (index 0) is before `warmup`, mirroring a
        // landscape window whose sub-slice of the gauge's record happens to
        // land entirely in an observation gap after warmup -- this is the
        // condition `Objective::build` turns into `NoValidObservations`
        // instead of letting `forward_loss` panic on an empty valid set.
        let obs = vec![3.5, f32::NAN, f32::NAN, f32::NAN];
        assert!(!has_valid_observation(&obs, 1, obs.len()));
    }

    #[test]
    fn has_valid_observation_false_for_negative_fill_values() {
        // Negative observations (e.g. a -0.001 fill/sentinel) are not valid.
        let obs = vec![-0.001, -1.0];
        assert!(!has_valid_observation(&obs, 0, obs.len()));
    }

    #[test]
    fn no_valid_observations_display_matches_the_skip_log_format() {
        let e = NoValidObservations {
            start: NaiveDate::from_ymd_opt(1996, 1, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(1996, 12, 31).unwrap(),
        };
        assert_eq!(e.to_string(), "no valid observations in the window 1996-01-01 .. 1996-12-31");
    }

    #[test]
    fn resolve_default_uses_configured_entry() {
        let mut defaults = HashMap::new();
        defaults.insert("n".to_string(), 0.05);
        assert_eq!(resolve_default(&defaults, "n").unwrap(), 0.05);
    }

    #[test]
    fn resolve_default_falls_back_to_21_for_p_spatial_only() {
        let defaults = HashMap::new();
        assert_eq!(resolve_default(&defaults, "p_spatial").unwrap(), 21.0);
    }

    #[test]
    fn resolve_default_errors_for_n_and_q_spatial_without_a_configured_default() {
        let defaults = HashMap::new();
        assert!(resolve_default(&defaults, "n").is_err());
        assert!(resolve_default(&defaults, "q_spatial").is_err());
    }

    #[test]
    fn kge_perfect_prediction_is_one() {
        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = kge(&obs, &obs);
        assert!((v - 1.0).abs() < 1e-5, "expected KGE 1.0 (loss 0), got {v}");
    }

    #[test]
    fn kge_doubled_matches_hand_computation() {
        // sim = 2*obs: r = 1 (perfectly correlated), alpha = std(sim)/std(obs) = 2,
        // beta = mean(sim)/mean(obs) = 2. KGE = 1 - sqrt(0^2 + 1^2 + 1^2) = 1 - sqrt(2).
        let obs = vec![1.0, 2.0, 3.0, 4.0];
        let sim: Vec<f32> = obs.iter().map(|v| 2.0 * v).collect();
        let v = kge(&sim, &obs);
        let expected = 1.0 - std::f32::consts::SQRT_2;
        assert!((v - expected).abs() < 1e-4, "expected {expected}, got {v}");
    }

    #[test]
    fn kge_degenerate_obs_is_nan() {
        let obs = vec![0.0, 0.0, 0.0];
        let sim = vec![1.0, 2.0, 3.0];
        assert!(kge(&sim, &obs).is_nan());
    }

    #[test]
    fn eig3_diagonalizes_known_matrix() {
        let (vals, vecs) = eig3([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(vals, [5.0, 2.0, 1.0]);
        assert!((vecs[1][0].abs() - 1.0).abs() < 1e-6);
        let (vals, vecs) = eig3([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
        assert!((vals[0] - 3.0).abs() < 1e-5 && (vals[1] - 3.0).abs() < 1e-5 && (vals[2] - 1.0).abs() < 1e-5);
        // eigenvector for eigenvalue 1 is (1,-1,0)/√2
        let c = 2;
        assert!((vecs[0][c].abs() - 0.7071).abs() < 1e-3 && (vecs[1][c].abs() - 0.7071).abs() < 1e-3);
    }

    #[test]
    fn solve_active_zeros_the_inactive_component() {
        let a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let b = [1.0, 2.0, 3.0];
        // p (index 1) fixed: rows/cols 0 and 2 are diagonal-only in this `a`,
        // so the masked solve reduces to two independent scalar equations.
        let d = solve_active(a, b, [true, false, true]).unwrap();
        assert_eq!(d[1], 0.0);
        assert!((d[0] - 0.25).abs() < 1e-5);
        assert!((d[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn solve_active_matches_solve3_when_all_active() {
        let a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let b = [1.0, 2.0, 3.0];
        assert_eq!(solve_active(a, b, [true, true, true]), solve3(a, b));
    }

    #[test]
    fn eig_active_drops_the_inactive_component() {
        // p (index 1) fixed. Diagonal matrix so the active eigenpairs (2.0
        // at index 0, 1.0 at index 2) are known exactly.
        let (vals, vecs) = eig_active([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]], [true, false, true]);
        assert!((vals[0] - 2.0).abs() < 1e-5);
        assert!((vals[1] - 1.0).abs() < 1e-5);
        assert!(vals[2].is_nan());
        // eigenvector for the dropped slot is the unit vector on the fixed axis (index 1).
        assert_eq!(vecs[0][2], 0.0);
        assert_eq!(vecs[1][2], 1.0);
        assert_eq!(vecs[2][2], 0.0);
    }

    #[test]
    fn eig_active_matches_eig3_when_all_active() {
        let a = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        assert_eq!(eig_active(a, [true, true, true]), eig3(a));
    }

    #[test]
    fn solve3_inverts() {
        let x = solve3([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], [1.0, 2.0, 3.0]).unwrap();
        let r = [4.0 * x[0] + x[1], x[0] + 3.0 * x[1] + x[2], x[1] + 2.0 * x[2]];
        assert!((r[0] - 1.0).abs() < 1e-4 && (r[1] - 2.0).abs() < 1e-4 && (r[2] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn newton_direction_caps_flat_hessian_step() {
        // Flat 2-D Hessian (n, p active; q fixed): mu alone provides the
        // curvature, so the raw Newton step is -grad/mu, arbitrarily long
        // for small mu. The capped direction must have length exactly
        // step_cap over the active components.
        let hess = [[0.0; 3]; 3];
        let grad = [3.0, -4.0, 0.0];
        let mu = 1e-4;
        let active = [true, true, false];
        let step_cap = 1.0;
        let d = newton_direction(hess, grad, mu, active, step_cap);
        assert_eq!(d[2], 0.0);
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        assert!((len - step_cap).abs() < 1e-4, "expected length {step_cap}, got {len}");
        // direction is descent: dot(d, grad) < 0.
        assert!(d[0] * grad[0] + d[1] * grad[1] < 0.0);
    }

    #[test]
    fn newton_direction_falls_back_to_gradient_when_uphill() {
        // Indefinite Hessian; with this grad the damped Newton step points
        // uphill (dot(d, grad) >= 0), so the fallback must kick in and
        // return the (rescaled) steepest-descent direction.
        let hess = [[-5.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let grad = [1.0, 0.0, 0.0];
        let mu = 0.0; // damping alone would flip the sign; keep it at 0 to force the uphill case
        let active = [true, false, false];
        let step_cap = 2.0;
        let d = newton_direction(hess, grad, mu, active, step_cap);
        // Raw Newton step here is -grad/hess[0][0] = -1/-5 = 0.2 (uphill:
        // dot(d, grad) = 0.2 > 0), so the fallback direction is -grad
        // rescaled to step_cap: (-step_cap, 0, 0).
        assert!((d[0] - (-step_cap)).abs() < 1e-5);
        assert_eq!(d[1], 0.0);
        assert_eq!(d[2], 0.0);
    }

    #[test]
    fn newton_direction_unmodified_when_well_conditioned_and_short() {
        // Well-conditioned diagonal Hessian; raw Newton step is well within
        // step_cap, so it must come back unmodified.
        let hess = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
        let grad = [1.0, -1.0, 2.0];
        let mu = 0.0;
        let active = [true, true, true];
        let step_cap = 10.0;
        let d = newton_direction(hess, grad, mu, active, step_cap);
        let expected = solve_active(hess, [-grad[0], -grad[1], -grad[2]], active).unwrap();
        assert_eq!(d, expected);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!(len < step_cap);
    }

    /// Consistency check (chain rule): the basin-uniform gradient is the sum
    /// over reaches of the per-reach log-gradient, because the uniform
    /// scalar leaf and the per-reach leaves both feed the same
    /// `x_i = x0_i * exp(leaf_i)` multiply, so `dL/d(uniform leaf) = Σ_i dL/d
    /// ln x_i`. `sum(obj.reach_grad(alpha).n/p/q)` should equal
    /// `obj.eval(alpha, true).grad` component-wise to 1e-3 relative.
    ///
    /// Ignored: building an `Objective` needs a real trained run.
    /// `Objective::build` reads an `InfluenceContext` opened from a
    /// `ResolvedArm` (a `.ddrs/runs/<id>` checkpoint) plus real gauge
    /// windows, which this crate's fixture-free unit tests don't have. To
    /// run it against the Juniata example:
    ///   1. `target/release/ddrs --config examples/juniata/ddrs.yaml run \
    ///        --workflow train-and-test --backend cpu` and note the run id.
    ///   2. Build a `ResolvedArm` for that run (see `resolve_arm` in
    ///      `src/experiment/mod.rs`), open an `InfluenceContext`, and call
    ///      `Objective::build(&ctx, &Staid::new("01567000"), &starts, 90, "nse-batch")`.
    ///   3. Compare `obj.reach_grad(alpha)` summed per component against
    ///      `obj.eval(alpha, true).grad`.
    /// The same check runs unconditionally at runtime in `run_gauge` (logged,
    /// not asserted) whenever `landscape.reach_grad: true`.
    #[test]
    #[ignore = "needs a real trained run; see doc comment for the manual recipe against examples/juniata"]
    fn reach_grad_sums_to_uniform_gradient() {}
}
