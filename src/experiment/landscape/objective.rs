//! Per-gauge objective in basin-uniform log-multiplier space.
//!
//! `α = (α_n, α_p, α_q)` scales the trained physical fields `n₀·e^{α_n}` etc.,
//! clamped to the arm's parameter ranges and re-normalized for the engine.
//! `L_g(α)` is the NSE-batch loss (training objective) over the configured
//! windows; its gradient comes from autograd with `α` lifted as leaves.
//! Spec: docs/superpowers/specs/2026-09-07-adjoint-landscape-design.md §1–§2.

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

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
    /// Trained physical fields (inner backend, constant).
    pub n0: Tensor<I, 1>,
    pub p0: Tensor<I, 1>,
    pub q0: Tensor<I, 1>,
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
    pub ranges: [[f32; 2]; 3],
    pub log_space: [bool; 3],
}

/// Result of one objective evaluation.
#[derive(Debug, Clone)]
pub struct Eval {
    pub loss: f32,
    /// Per-window NSE of the daily series.
    pub nse: Vec<f32>,
    pub nse_mean: f32,
    /// Fraction of (reach, parameter) entries clamped at a range edge, max over windows.
    pub clamped_frac: f32,
    pub grad: Option<[f32; 3]>,
}

/// Per-reach `dL/d ln x` for `x` in `(n, p_spatial, q_spatial)`, length `n_reach` each.
#[derive(Debug, Clone)]
pub struct ReachGrad {
    pub n: Vec<f32>,
    pub p: Vec<f32>,
    pub q: Vec<f32>,
}

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
    ) -> Result<Self, BoxError> {
        let device = &ctx.device;
        let ranges = [
            ctx.cfg.params.parameter_ranges.n,
            ctx.cfg.params.parameter_ranges.p_spatial,
            ctx.cfg.params.parameter_ranges.q_spatial,
        ];
        let log = &ctx.cfg.params.log_space_parameters;
        let log_space = [
            log.iter().any(|s| s == "n"),
            log.iter().any(|s| s == "p_spatial"),
            log.iter().any(|s| s == "q_spatial"),
        ];
        let sigma = ctx
            .dataset
            .gauge_obs_std(&[staid.clone()])
            .map_err(|e| format!("gauge std: {e}"))?
            .first()
            .copied()
            .ok_or("gauge std unavailable (loss.kind must be nse-batch in the arm's config)")?;
        let eps = ctx.cfg.experiment.as_ref().map(|e| e.loss.eps).unwrap_or(0.1);
        let mut windows = Vec::new();
        for &start in window_starts {
            let batch = ctx.collate_gauge(staid, start, window_days)?;
            let obs: Vec<f32> = batch.observations.column(0).to_vec();
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
            let get = |k: &str| params_map.get(k).unwrap_or_else(|| panic!("KAN head missing {k}")).clone().inner();
            let n0 = denormalize(get("n"), ranges[0], log_space[0]);
            let q0 = denormalize(get("q_spatial"), ranges[2], log_space[2]);
            let p0 = match params_map.get("p_spatial") {
                Some(p) => denormalize(p.clone().inner(), ranges[1], log_space[1]),
                None => Tensor::<I, 1>::full([n_active], *ctx.cfg.params.defaults.get("p_spatial").unwrap_or(&21.0), device),
            };
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
            windows.push(WindowData { start_day: start, tensors, obs, n0, p0, q0, x_storage, q_prime, slope, length, gauge_row, comids });
        }
        Ok(Self { ctx, windows, sigma, eps, ranges, log_space })
    }

    pub fn n_reach(&self) -> usize {
        self.windows[0].tensors.adjacency.n
    }

    /// Physical fields at `α` (inner backend), for hydraulic checks.
    pub fn fields_at(&self, w: &WindowData<I>, alpha: [f32; 3]) -> (Tensor<I, 1>, Tensor<I, 1>, Tensor<I, 1>) {
        // NB: clamp the FIELD, not the scalar multiplier (precedence).
        let f = |x0: &Tensor<I, 1>, a: f32, r: [f32; 2]| (x0.clone() * a.exp()).clamp(r[0], r[1]);
        (f(&w.n0, alpha[0], self.ranges[0]), f(&w.p0, alpha[1], self.ranges[1]), f(&w.q0, alpha[2], self.ranges[2]))
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
    /// the batch-mean loss tensor (still on the autodiff tape), per-window
    /// NSE, and the max clamped fraction over windows.
    fn forward_loss(&self, leaves: &[Tensor<AD<I>, 1>; 3]) -> (Tensor<AD<I>, 1>, Vec<f32>, f32) {
        let device = &self.ctx.device;
        let n = self.n_reach();
        let ones = Tensor::<AD<I>, 1>::ones([n], device);
        let mut total: Option<Tensor<AD<I>, 1>> = None;
        let mut nses = Vec::new();
        let mut clamped_max = 0.0f32;
        for w in &self.windows {
            let mut norm = Vec::new();
            for (k, x0) in [&w.n0, &w.p0, &w.q0].into_iter().enumerate() {
                let scale = leaves[k].clone().exp() * ones.clone(); // [n]
                let phys = Tensor::<AD<I>, 1>::from_inner(x0.clone()) * scale;
                let [lo, hi] = self.ranges[k];
                // clamped fraction (diagnostic)
                let v: Vec<f32> = phys.clone().inner().into_data().to_vec::<f32>().unwrap();
                let cf = v.iter().filter(|x| **x <= lo || **x >= hi).count() as f32 / n as f32;
                clamped_max = clamped_max.max(cf);
                let phys = phys.clamp(lo, hi);
                norm.push(Self::normalize_ad(phys, self.ranges[k], self.log_space[k]));
            }
            let mut engine = MuskingumCunge::<I>::new(self.ctx.cfg.clone(), device.clone());
            engine.setup_inputs(
                RoutingInputs { adjacency: w.tensors.adjacency.clone(), x_storage: Tensor::from_inner(w.x_storage.clone()) },
                Tensor::<AD<I>, 2>::from_inner(w.q_prime.clone()),
                SpatialParameters {
                    n: norm[0].clone(),
                    q_spatial: norm[2].clone(),
                    p_spatial: Some(norm[1].clone()),
                    k_d: None,
                    d_gw: None,
                    leakance_factor: None,
                    impervious_mask: None,
                },
                false,
                None,
            );
            let runoff = engine.forward();
            let gauge = scatter_add_by_group(runoff, w.tensors.flat_indices.clone(), w.tensors.group_ids.clone(), w.tensors.num_gauges);
            let daily = self.ctx.daily(gauge); // (1, D)
            let d = daily.dims()[1];
            let valid: Vec<usize> = (self.ctx.warmup..d).filter(|&i| i < w.obs.len() && w.obs[i].is_finite() && w.obs[i] >= 0.0).collect();
            if valid.is_empty() {
                nses.push(f32::NAN);
                continue;
            }
            let mut wts = vec![0.0f32; d];
            let mut obs_f = vec![0.0f32; d];
            for &i in &valid {
                wts[i] = 1.0 / valid.len() as f32;
                obs_f[i] = w.obs[i];
            }
            let w_t = Tensor::<AD<I>, 1>::from_floats(wts.as_slice(), device).reshape([1, d]);
            let o_t = Tensor::<AD<I>, 1>::from_floats(obs_f.as_slice(), device).reshape([1, d]);
            let denom = (self.sigma + self.eps) * (self.sigma + self.eps);
            let sq = (daily.clone() - o_t).powf_scalar(2.0) * w_t;
            let loss_w = sq.sum() / denom;
            // NSE (diagnostic)
            let dv: Vec<f32> = daily.inner().into_data().to_vec::<f32>().unwrap();
            let om: f32 = valid.iter().map(|&i| w.obs[i]).sum::<f32>() / valid.len() as f32;
            let sse: f32 = valid.iter().map(|&i| (dv[i] - w.obs[i]).powi(2)).sum();
            let sst: f32 = valid.iter().map(|&i| (w.obs[i] - om).powi(2)).sum();
            nses.push(if sst > 0.0 { 1.0 - sse / sst } else { f32::NAN });
            total = Some(match total {
                Some(t) => t + loss_w,
                None => loss_w,
            });
        }
        let n_w = nses.iter().filter(|v| v.is_finite()).count().max(1) as f32;
        let total = total.expect("at least one window with valid observations");
        (total / n_w, nses, clamped_max)
    }

    /// Evaluate `L_g(α)` (mean over windows) and optionally its gradient.
    pub fn eval(&self, alpha: [f32; 3], with_grad: bool) -> Eval {
        let device = &self.ctx.device;
        // α leaves (shape [1]) — one set shared across windows.
        let leaves: [Tensor<AD<I>, 1>; 3] = std::array::from_fn(|k| {
            let t = Tensor::<AD<I>, 1>::from_floats([alpha[k]].as_slice(), device);
            if with_grad { t.require_grad() } else { t }
        });
        let (total, nses, clamped_max) = self.forward_loss(&leaves);
        let loss: f32 = total.clone().inner().into_data().to_vec::<f32>().unwrap()[0];
        let grad = if with_grad {
            let grads = total.backward();
            let g: Vec<f32> = leaves
                .iter()
                .map(|l| l.grad(&grads).map(|t| t.into_data().to_vec::<f32>().unwrap()[0]).unwrap_or(0.0))
                .collect();
            Some([g[0], g[1], g[2]])
        } else {
            None
        };
        let nse_mean = {
            let f: Vec<f32> = nses.iter().copied().filter(|v| v.is_finite()).collect();
            if f.is_empty() { f32::NAN } else { f.iter().sum::<f32>() / f.len() as f32 }
        };
        Eval { loss, nse: nses, nse_mean, clamped_frac: clamped_max, grad }
    }

    /// Per-reach `g_i = dL/d ln x_i` for `x` in `(n, p_spatial, q_spatial)` at
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
        let (total, _nses, _clamped_max) = self.forward_loss(&leaves);
        let grads = total.backward();
        let extract = |l: &Tensor<AD<I>, 1>| -> Vec<f32> {
            l.grad(&grads).map(|t| t.into_data().to_vec::<f32>().unwrap()).unwrap_or_else(|| vec![0.0; n])
        };
        ReachGrad { n: extract(&leaves[0]), p: extract(&leaves[1]), q: extract(&leaves[2]) }
    }

    /// Central-difference Hessian of `L_g` at `α` (symmetrized), step `h`.
    pub fn hessian(&self, alpha: [f32; 3], h: f32) -> [[f32; 3]; 3] {
        let mut hm = [[0.0f32; 3]; 3];
        for k in 0..3 {
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
    fn solve3_inverts() {
        let x = solve3([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], [1.0, 2.0, 3.0]).unwrap();
        let r = [4.0 * x[0] + x[1], x[0] + 3.0 * x[1] + x[2], x[1] + 2.0 * x[2]];
        assert!((r[0] - 1.0).abs() < 1e-4 && (r[1] - 2.0).abs() < 1e-4 && (r[2] - 3.0).abs() < 1e-4);
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
    ///      `Objective::build(&ctx, &Staid::new("01567000"), &starts, 90)`.
    ///   3. Compare `obj.reach_grad(alpha)` summed per component against
    ///      `obj.eval(alpha, true).grad`.
    /// The same check runs unconditionally at runtime in `run_gauge` (logged,
    /// not asserted) whenever `landscape.reach_grad: true`.
    #[test]
    #[ignore = "needs a real trained run; see doc comment for the manual recipe against examples/juniata"]
    fn reach_grad_sums_to_uniform_gradient() {}
}
