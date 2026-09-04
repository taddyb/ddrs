//! Finite-difference gate (spec §2.5, "approach C"): the adjoint kernel must
//! predict the effect of a real inflow perturbation to within `tol`.

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::data::dataset::RoutingTensors;

use super::influence::{Grad, InfluenceContext};
use super::super::BoxError;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReachCheck {
    pub role: String,
    pub reach_row: usize,
    pub comid: i64,
    pub dist_to_gauge_m: f32,
    pub delta_m3s: f32,
    pub dq_predicted: f32,
    pub dq_actual: f32,
    pub rel_err: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ValidationResult {
    pub arm: String,
    pub staid: String,
    pub anchor_day_idx: usize,
    pub t0: usize,
    pub perturb_fraction: f32,
    pub lag_days: usize,
    pub tol: f32,
    pub passed: bool,
    pub checks: Vec<ReachCheck>,
}

pub struct GateInputs<'a> {
    pub arm: &'a str,
    pub staid: &'a str,
    pub comids: &'a [i64],
    pub anchor_day_idx: usize,
    pub t0: usize,
    pub lag_days: usize,
    pub grad: &'a Grad,
    /// Unperturbed `Q_g(t0)`.
    pub base_q_t0: f32,
    /// Row-major `(T, N)` inflow that was routed.
    pub q_hourly: &'a [f32],
    pub dist_to_gauge_m: &'a [f32],
    pub gauge_row: usize,
    pub frac: f32,
    pub tol: f32,
}

/// Perturb inflow by `+frac · mean(q'_i)` over source hours `[t0 − 24·lag_days, t0)`
/// at three reaches (gauge, median distance, farthest), rerun the forward, and
/// compare `ΔQ_g(t0)` with `Σ grad·δ`.
pub fn finite_difference_gate<I: Backend + 'static>(
    ctx: &InfluenceContext<I>,
    tensors: &RoutingTensors<Autodiff<I>>,
    g: GateInputs<'_>,
) -> Result<ValidationResult, BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let (t, n) = (g.grad.t, g.grad.n);
    let from = g.t0.saturating_sub(24 * g.lag_days);
    let mut order: Vec<usize> = (0..n).filter(|&i| g.dist_to_gauge_m[i].is_finite()).collect();
    order.sort_by(|a, b| g.dist_to_gauge_m[*a].partial_cmp(&g.dist_to_gauge_m[*b]).unwrap());
    if order.is_empty() {
        return Err("finite-difference gate: no reach has a finite distance to the gauge".into());
    }
    let picks: Vec<(&str, usize)> = vec![
        ("gauge", g.gauge_row),
        ("median", order[order.len() / 2]),
        ("farthest", *order.last().unwrap()),
    ];
    let global_mean: f32 = g.q_hourly.iter().sum::<f32>() / g.q_hourly.len().max(1) as f32;
    let mut checks = Vec::new();
    for (role, reach) in picks {
        let mean_i: f32 = (0..t).map(|h| g.q_hourly[h * n + reach]).sum::<f32>() / t as f32;
        let delta = g.frac * if mean_i > 0.0 { mean_i } else { global_mean };
        let mut perturbed = g.q_hourly.to_vec();
        for h in from..g.t0 {
            perturbed[h * n + reach] += delta;
        }
        let dq_predicted: f32 = (from..g.t0).map(|h| g.grad.at(h, reach)).sum::<f32>() * delta;
        let q_pert: Tensor<I, 2> =
            Tensor::<I, 1>::from_floats(perturbed.as_slice(), &ctx.device).reshape([t, n]);
        let lf = ctx.forward_with_inflow_leaf(tensors, Some(q_pert));
        let series: Vec<f32> = lf.gauge_series.inner().into_data().to_vec::<f32>().unwrap();
        let dq_actual = series[g.t0] - g.base_q_t0;
        let rel_err = (dq_actual - dq_predicted).abs() / dq_actual.abs().max(1e-6);
        println!(
            "  FD gate {role:>8} row {reach:>5} comid {} dist {:>9.0} m  delta {delta:.4}  dQ pred {dq_predicted:.5}  actual {dq_actual:.5}  rel_err {rel_err:.4}",
            g.comids[reach], g.dist_to_gauge_m[reach]
        );
        checks.push(ReachCheck {
            role: role.to_string(),
            reach_row: reach,
            comid: g.comids[reach],
            dist_to_gauge_m: g.dist_to_gauge_m[reach],
            delta_m3s: delta,
            dq_predicted,
            dq_actual,
            rel_err,
        });
    }
    let passed = checks.iter().all(|c| c.rel_err.is_finite() && c.rel_err < g.tol);
    Ok(ValidationResult {
        arm: g.arm.to_string(),
        staid: g.staid.to_string(),
        anchor_day_idx: g.anchor_day_idx,
        t0: g.t0,
        perturb_fraction: g.frac,
        lag_days: g.lag_days,
        tol: g.tol,
        passed,
        checks,
    })
}
