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


// ---------------------------------------------------------------------------
// Full-map finite-difference validation (check 1): central differences at
// every `reach_stride`-th reach and several lags, against the kernel entry.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct SweepPoint {
    pub delta_m3s: f32,
    /// Central difference per unit inflow.
    pub dq_actual: f32,
    /// One-sided forward `(Q(q+δ) − Q(q)) / δ` and backward `(Q(q) − Q(q−δ)) / δ`
    /// slopes; a kink at the base state shows as forward ≠ backward with the
    /// gradient matching one of them.
    pub dq_forward: f32,
    pub dq_backward: f32,
    pub rel_err: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FullMapRow {
    pub reach_row: usize,
    pub comid: i64,
    pub dist_to_gauge_m: f32,
    pub lag_hours: usize,
    pub delta_m3s: f32,
    /// Kernel entry (per unit inflow).
    pub dq_predicted: f32,
    /// Central difference (per unit inflow).
    pub dq_actual: f32,
    pub dq_total_pred_m3s: f32,
    pub dq_total_actual_m3s: f32,
    pub abs_err_m3s: f32,
    pub rel_err: f32,
    pub unresolvable: bool,
    pub passed: bool,
    pub sweep: Vec<SweepPoint>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FullMapResult {
    pub kind: String,
    pub arm: String,
    pub staid: String,
    pub anchor_day_idx: usize,
    pub anchor_kind: String,
    pub t0: usize,
    pub q_gauge_t0: f32,
    pub noise_floor_m3s: f32,
    pub reach_stride: usize,
    pub lags_hours: Vec<usize>,
    pub perturb_fraction: f32,
    pub rel_tol: f32,
    pub abs_tol: f32,
    pub n_checks: usize,
    pub n_passed: usize,
    pub n_unresolvable: usize,
    pub frac_passed: f32,
    pub passed: bool,
    pub max_rel_err_among_failed: f32,
    #[serde(skip)]
    pub rows: Vec<FullMapRow>,
}

pub struct FullMapInputs<'a> {
    pub arm: &'a str,
    pub staid: &'a str,
    pub comids: &'a [i64],
    pub anchor_day_idx: usize,
    pub anchor_kind: &'a str,
    pub t0: usize,
    /// Unperturbed `Q_g(t0)`, sets the f32 noise floor.
    pub base_q_t0: f32,
    pub grad: &'a Grad,
    pub q_hourly: &'a [f32],
    pub dist_to_gauge_m: &'a [f32],
    pub reach_stride: usize,
    pub lags_hours: &'a [usize],
    pub perturb_fraction: f32,
    pub rel_tol: f32,
    pub abs_tol: f32,
    pub pass_fraction: f32,
    /// Fractions of the mean inflow to sweep on failing entries.
    pub delta_sweep: &'a [f32],
}

/// For each sampled reach `i` and lag `h`, perturb `q'(i, t0-h)` by `±δ` for
/// that single hour, rerun the forward twice, and compare the central
/// difference `(Q⁺(t0) − Q⁻(t0)) / 2δ` with the kernel entry `grad[t0-h, i]`.
///
/// Tolerances are in **discharge units** so f32 quantisation of `Q_g` is
/// handled explicitly: the noise floor is `noise = max(abs_tol, 4·ulp(Q_g(t0)))`
/// with `ulp = 1.19e-7·|Q_g|`. δ is scaled so the predicted `|ΔQ| = |grad|·2δ`
/// is about `50·noise` (bounded to `[0.02, 0.5·mean inflow]` m³/s, never below
/// `perturb_fraction`·mean); entries whose predicted and realised `|ΔQ|` are
/// both below `2·noise` pass as unresolvable; otherwise the check passes when
/// `|ΔQ_actual − ΔQ_pred| ≤ noise` or the relative error is `≤ rel_tol`.
/// Every failing entry is re-checked over a δ sweep (`delta_sweep` fractions of
/// the mean inflow) so a nonlinearity/clamp kink (FD → gradient as δ → 0) can be
/// told from a gradient error (FD does not converge).
pub fn full_map_gate<I: Backend + 'static>(
    ctx: &InfluenceContext<I>,
    tensors: &RoutingTensors<Autodiff<I>>,
    g: FullMapInputs<'_>,
) -> Result<FullMapResult, BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let (t, n) = (g.grad.t, g.grad.n);
    let stride = g.reach_stride.max(1);
    let mut rows = Vec::new();
    let mut reaches: Vec<usize> = (0..n).step_by(stride).collect();
    let far = (0..n)
        .filter(|&i| g.dist_to_gauge_m[i].is_finite())
        .max_by(|a, b| g.dist_to_gauge_m[*a].partial_cmp(&g.dist_to_gauge_m[*b]).unwrap());
    if let Some(f) = far {
        if !reaches.contains(&f) {
            reaches.push(f);
        }
    }
    let noise = g.abs_tol.max(4.0 * 1.19e-7 * g.base_q_t0.abs());

    // One central-difference evaluation: returns (dq_actual per unit inflow, ΔQ_actual, ΔQ_pred, effective 2δ, forward slope, backward slope).
    let eval = |reach: usize, src: usize, delta: f32| -> (f32, f32, f32, f32, f32, f32) {
        let mut plus = g.q_hourly.to_vec();
        let mut minus = g.q_hourly.to_vec();
        plus[src * n + reach] += delta;
        minus[src * n + reach] = (minus[src * n + reach] - delta).max(0.0);
        let two_delta = plus[src * n + reach] - minus[src * n + reach];
        let q_plus: Tensor<I, 2> = Tensor::<I, 1>::from_floats(plus.as_slice(), &ctx.device).reshape([t, n]);
        let q_minus: Tensor<I, 2> = Tensor::<I, 1>::from_floats(minus.as_slice(), &ctx.device).reshape([t, n]);
        let s_plus: Vec<f32> = ctx.forward_with_inflow_leaf(tensors, Some(q_plus)).gauge_series.inner().into_data().to_vec::<f32>().unwrap();
        let s_minus: Vec<f32> = ctx.forward_with_inflow_leaf(tensors, Some(q_minus)).gauge_series.inner().into_data().to_vec::<f32>().unwrap();
        let dq_total = s_plus[g.t0] - s_minus[g.t0];
        let dq_actual = dq_total / two_delta;
        let dq_pred_total = g.grad.at(src, reach) * two_delta;
        let d_plus = plus[src * n + reach] - g.q_hourly[src * n + reach];
        let d_minus = g.q_hourly[src * n + reach] - minus[src * n + reach];
        let fwd = if d_plus > 0.0 { (s_plus[g.t0] - g.base_q_t0) / d_plus } else { f32::NAN };
        let bwd = if d_minus > 0.0 { (g.base_q_t0 - s_minus[g.t0]) / d_minus } else { f32::NAN };
        (dq_actual, dq_total, dq_pred_total, two_delta, fwd, bwd)
    };

    for &reach in &reaches {
        let mean_i: f32 = (0..t).map(|h| g.q_hourly[h * n + reach]).sum::<f32>() / t as f32;
        for &lag in g.lags_hours {
            if lag == 0 || lag > g.t0 {
                continue;
            }
            let src = g.t0 - lag;
            let grad_entry = g.grad.at(src, reach);
            // δ sized for a resolvable response, bounded for linearity.
            let delta_min = (g.perturb_fraction * mean_i).max(0.02);
            let delta_max = (0.5 * mean_i).max(delta_min);
            let delta_target = if grad_entry.abs() > 1e-12 { 50.0 * noise / (2.0 * grad_entry.abs()) } else { delta_max };
            let delta = delta_target.clamp(delta_min, delta_max);
            let (dq_actual, dq_total, dq_pred_total, two_delta, _fwd, _bwd) = eval(reach, src, delta);
            let dq_predicted = grad_entry;
            let abs_err_q = (dq_total - dq_pred_total).abs();
            let rel_err = abs_err_q / dq_total.abs().max(1e-12);
            let unresolvable = dq_total.abs() < 2.0 * noise && dq_pred_total.abs() < 2.0 * noise;
            let passed = unresolvable || abs_err_q <= noise || rel_err <= g.rel_tol;
            let mut sweep = Vec::new();
            if !passed {
                for &f in g.delta_sweep {
                    let d = (f * mean_i).max(0.005);
                    let (a, tot, pred_tot, _, fwd, bwd) = eval(reach, src, d);
                    sweep.push(SweepPoint { delta_m3s: d, dq_actual: a, dq_forward: fwd, dq_backward: bwd, rel_err: (tot - pred_tot).abs() / tot.abs().max(1e-12) });
                }
            }
            rows.push(FullMapRow {
                reach_row: reach,
                comid: g.comids[reach],
                dist_to_gauge_m: g.dist_to_gauge_m[reach],
                lag_hours: lag,
                delta_m3s: two_delta / 2.0,
                dq_predicted,
                dq_actual,
                dq_total_pred_m3s: dq_pred_total,
                dq_total_actual_m3s: dq_total,
                abs_err_m3s: abs_err_q,
                rel_err,
                unresolvable,
                passed,
                sweep,
            });
        }
    }
    let n_checks = rows.len();
    let n_passed = rows.iter().filter(|r| r.passed).count();
    let n_unresolvable = rows.iter().filter(|r| r.unresolvable).count();
    let frac = if n_checks > 0 { n_passed as f32 / n_checks as f32 } else { 0.0 };
    let max_rel_failed = rows.iter().filter(|r| !r.passed).map(|r| r.rel_err).fold(0.0f32, f32::max);
    println!(
        "  [{}] full-map FD {} {} anchor day {} (Q_g={:.1}, noise {:.2e} m3/s): {}/{} pass ({:.1} %), {} unresolvable, worst failed rel_err {:.3}",
        g.arm, g.staid, g.anchor_kind, g.anchor_day_idx, g.base_q_t0, noise, n_passed, n_checks, 100.0 * frac, n_unresolvable, max_rel_failed
    );
    Ok(FullMapResult {
        kind: "full_map".into(),
        arm: g.arm.to_string(),
        staid: g.staid.to_string(),
        anchor_day_idx: g.anchor_day_idx,
        anchor_kind: g.anchor_kind.to_string(),
        t0: g.t0,
        q_gauge_t0: g.base_q_t0,
        noise_floor_m3s: noise,
        reach_stride: stride,
        lags_hours: g.lags_hours.to_vec(),
        perturb_fraction: g.perturb_fraction,
        rel_tol: g.rel_tol,
        abs_tol: g.abs_tol,
        n_checks,
        n_passed,
        n_unresolvable,
        frac_passed: frac,
        passed: frac >= g.pass_fraction,
        max_rel_err_among_failed: max_rel_failed,
        rows,
    })
}

pub fn write_full_map_csv(path: &std::path::Path, r: &FullMapResult) -> Result<(), BoxError> {
    use std::io::Write;
    let mut w = std::fs::File::create(path)?;
    writeln!(w, "arm,staid,anchor_day_idx,anchor_kind,t0,q_gauge_t0,noise_floor_m3s,reach_row,comid,dist_to_gauge_m,lag_hours,delta_m3s,dq_predicted,dq_actual,dq_total_pred_m3s,dq_total_actual_m3s,abs_err_m3s,rel_err,unresolvable,passed,sweep_deltas,sweep_dq_actual,sweep_dq_forward,sweep_dq_backward,sweep_rel_err")?;
    for x in &r.rows {
        let sd: Vec<String> = x.sweep.iter().map(|p| format!("{:.5}", p.delta_m3s)).collect();
        let sa: Vec<String> = x.sweep.iter().map(|p| format!("{:.6e}", p.dq_actual)).collect();
        let sr: Vec<String> = x.sweep.iter().map(|p| format!("{:.4}", p.rel_err)).collect();
        let sf: Vec<String> = x.sweep.iter().map(|p| format!("{:.6e}", p.dq_forward)).collect();
        let sb: Vec<String> = x.sweep.iter().map(|p| format!("{:.6e}", p.dq_backward)).collect();
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.arm, r.staid, r.anchor_day_idx, r.anchor_kind, r.t0, r.q_gauge_t0, r.noise_floor_m3s, x.reach_row, x.comid, x.dist_to_gauge_m, x.lag_hours,
            x.delta_m3s, x.dq_predicted, x.dq_actual, x.dq_total_pred_m3s, x.dq_total_actual_m3s, x.abs_err_m3s, x.rel_err,
            x.unresolvable as i32, x.passed as i32, sd.join(";"), sa.join(";"), sf.join(";"), sb.join(";"), sr.join(";")
        )?;
    }
    Ok(())
}
