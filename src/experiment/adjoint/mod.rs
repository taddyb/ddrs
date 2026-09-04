//! The adjoint influence-map study: gradient of routed gauge discharge with
//! respect to lateral inflow at every upstream reach and hour, per trained arm.
//!
//! Spec: `docs/superpowers/specs/2026-09-03-ddrs-experiment-adjoint-design.md` §2.

pub mod influence;
pub mod output;
pub mod validate;

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use chrono::{Duration, NaiveDate};
use serde::{Deserialize, Serialize};

use crate::data::dates::TimeAxis;
use crate::data::ids::Staid;

use self::influence::{column_means, dist_to_gauge, inflow_gradient, InfluenceContext};
use self::output::{append_summary, write_gauge_netcdf, AnchorRecord, GaugeResult, WindowRecord};
use self::validate::{finite_difference_gate, GateInputs};
use super::{BoxError, ExperimentManifest, ResolvedArm};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AdjointSpec {
    pub gauges: GaugeSpec,
    #[serde(default = "d_window_days")]
    pub window_days: usize,
    #[serde(default)]
    pub anchors: AnchorSpec,
    #[serde(default = "d_lag_days")]
    pub lag_days: usize,
    #[serde(default = "d_water_year")]
    pub water_year: i32,
    #[serde(default = "d_functionals")]
    pub functionals: Vec<Functional>,
    #[serde(default = "d_anchor_min_gap_days")]
    pub anchor_min_gap_days: usize,
    #[serde(default = "d_tail_days")]
    pub tail_days: usize,
    #[serde(default = "d_fd_fraction")]
    pub fd_perturb_fraction: f32,
    #[serde(default = "d_fd_tol")]
    pub fd_rel_tol: f32,
}

fn d_window_days() -> usize { 90 }
fn d_lag_days() -> usize { 30 }
fn d_water_year() -> i32 { 2000 }
fn d_functionals() -> Vec<Functional> { vec![Functional::Kernel, Functional::Volume, Functional::Residual] }
fn d_anchor_min_gap_days() -> usize { 60 }
fn d_tail_days() -> usize { 7 }
fn d_fd_fraction() -> f32 { 0.05 }
fn d_fd_tol() -> f32 { 0.05 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GaugeSpec {
    /// `[upstream, downstream]` staid pairs. PoC scope; the GAGES-II
    /// nested-reference selection is a follow-up.
    pub pairs: Vec<[String; 2]>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnchorSpec {
    #[serde(default = "two")]
    pub high: usize,
    #[serde(default = "two")]
    pub low: usize,
}
fn two() -> usize { 2 }
impl Default for AnchorSpec {
    fn default() -> Self { Self { high: 2, low: 2 } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Functional {
    Kernel,
    Volume,
    Residual,
}

pub struct AdjointOptions {
    pub max_gauges: Option<usize>,
    pub skip_validate: bool,
    pub force_cpu: bool,
}

#[derive(Debug, Clone)]
pub struct GaugeEntry {
    pub staid: String,
    pub role: &'static str,
    pub pair: usize,
    pub upstream: Vec<String>,
}

/// Unique gauges from the pair list, upstream before downstream within a pair.
pub fn gauge_list(spec: &AdjointSpec) -> Vec<GaugeEntry> {
    let mut out: Vec<GaugeEntry> = Vec::new();
    for (pi, [up, down]) in spec.gauges.pairs.iter().enumerate() {
        if !out.iter().any(|g| &g.staid == up) {
            out.push(GaugeEntry { staid: up.clone(), role: "upstream", pair: pi, upstream: vec![] });
        }
        match out.iter_mut().find(|g| &g.staid == down) {
            Some(g) => {
                g.role = "downstream";
                g.upstream.push(up.clone());
            }
            None => out.push(GaugeEntry {
                staid: down.clone(),
                role: "downstream",
                pair: pi,
                upstream: vec![up.clone()],
            }),
        }
    }
    out
}

fn write_gauges_csv(path: &Path, gauges: &[GaugeEntry]) -> Result<(), BoxError> {
    let mut w = std::fs::File::create(path)?;
    writeln!(w, "staid,role,pair,upstream_staids")?;
    for g in gauges {
        writeln!(w, "{},{},{},{}", g.staid, g.role, g.pair, g.upstream.join(";"))?;
    }
    Ok(())
}

/// Anchor days: the `n_high` highest and `n_low` lowest strictly-positive
/// valid days in `[min_day, max_day]`, each at least `min_gap` days from every
/// other chosen anchor. Returns `(day_idx, obs, kind)`.
pub fn select_anchors(
    obs: &[f32],
    n_high: usize,
    n_low: usize,
    min_gap: usize,
    min_day: usize,
    max_day: usize,
) -> Vec<(usize, f32, &'static str)> {
    let valid: Vec<(usize, f32)> = obs
        .iter()
        .enumerate()
        .filter(|(d, v)| *d >= min_day && *d <= max_day && v.is_finite() && **v > 0.0)
        .map(|(d, v)| (d, *v))
        .collect();
    let mut chosen: Vec<(usize, f32, &'static str)> = Vec::new();
    let far_enough = |chosen: &[(usize, f32, &str)], d: usize| {
        chosen.iter().all(|(c, _, _)| (*c as i64 - d as i64).unsigned_abs() as usize >= min_gap)
    };
    let mut by_high = valid.clone();
    by_high.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (d, v) in &by_high {
        if chosen.len() >= n_high {
            break;
        }
        if far_enough(&chosen, *d) {
            chosen.push((*d, *v, "high"));
        }
    }
    let mut by_low = valid;
    by_low.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut n_low_chosen = 0;
    for (d, v) in &by_low {
        if n_low_chosen >= n_low {
            break;
        }
        if far_enough(&chosen, *d) {
            chosen.push((*d, *v, "low"));
            n_low_chosen += 1;
        }
    }
    chosen
}

/// Four seasonal windows of `window_days` starting Oct 1 (+0, +92, +182, +273
/// days) of the given water year.
pub fn seasonal_window_starts(
    axis: &TimeAxis,
    water_year: i32,
    window_days: usize,
) -> Result<Vec<(usize, NaiveDate)>, BoxError> {
    let oct1 = NaiveDate::from_ymd_opt(water_year - 1, 10, 1).ok_or("bad water_year")?;
    let mut out = Vec::new();
    for off in [0i64, 92, 182, 273] {
        let date = oct1 + Duration::days(off);
        let idx = axis
            .day_index(date)
            .ok_or_else(|| format!("window start {date} outside the eval axis [{}, {}]", axis.start, axis.end))?;
        if idx + window_days > axis.num_days {
            return Err(format!("window starting {date} (+{window_days} d) runs past the eval axis end {}", axis.end).into());
        }
        out.push((idx, date));
    }
    Ok(out)
}

pub fn run_adjoint<I: Backend + 'static>(
    adjoint: &AdjointSpec,
    arms: &[ResolvedArm],
    out_dir: &Path,
    opts: &AdjointOptions,
    device: &I::Device,
    manifest: &mut ExperimentManifest,
) -> Result<(), BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    if adjoint.window_days < adjoint.tail_days + 2 {
        return Err("adjoint.window_days must exceed tail_days + 1".into());
    }
    if 24 * adjoint.lag_days > (adjoint.window_days - 1 - adjoint.tail_days) * 24 {
        return Err("adjoint.lag_days must fit before the anchor: lag_days <= window_days - 1 - tail_days".into());
    }
    let mut gauges = gauge_list(adjoint);
    if let Some(k) = opts.max_gauges {
        gauges.truncate(k);
    }
    write_gauges_csv(&out_dir.join("gauges.csv"), &gauges)?;
    println!(
        "adjoint: {} arm(s), {} gauge(s), functionals {:?}, window {} d, lag {} d, anchors {}h/{}l",
        arms.len(),
        gauges.len(),
        adjoint.functionals,
        adjoint.window_days,
        adjoint.lag_days,
        adjoint.anchors.high,
        adjoint.anchors.low
    );

    let n_hourly = (adjoint.window_days - 1) * 24;
    let anchor_offset = adjoint.window_days - 1 - adjoint.tail_days;
    let t0 = anchor_offset * 24 + 12;
    let mut validated = opts.skip_validate;

    for arm in arms {
        let t_arm = Instant::now();
        println!("=== arm {} (run {}, checkpoint {}) ===", arm.name, arm.run_id, arm.checkpoint_label);
        let ctx = InfluenceContext::<I>::open(arm, device, opts.force_cpu)?;
        let warm_h = ctx.warmup * 24;
        let tail_h = adjoint.tail_days * 24;
        let reduce_to = n_hourly - tail_h;
        if warm_h >= reduce_to {
            return Err("warmup + tail leave no source hours to reduce over".into());
        }
        let arm_dir = out_dir.join(&arm.name);
        std::fs::create_dir_all(arm_dir.join("gauges"))?;
        let seasonal = seasonal_window_starts(&ctx.axis, adjoint.water_year, adjoint.window_days)?;

        for g in &gauges {
            let t_g = Instant::now();
            let staid = Staid::new(&g.staid);
            let obs_full = ctx.gauge_observations(&staid)?;
            let mut result: Option<GaugeResult> = None;

            // ---------------- kernel ----------------
            if adjoint.functionals.contains(&Functional::Kernel) {
                let max_day = ctx.axis.num_days - 1 - adjoint.tail_days;
                let picks = select_anchors(
                    &obs_full,
                    adjoint.anchors.high,
                    adjoint.anchors.low,
                    adjoint.anchor_min_gap_days,
                    anchor_offset,
                    max_day,
                );
                if picks.is_empty() {
                    manifest.notes.push(format!("{}/{}: no valid anchor days", arm.name, g.staid));
                }
                for (day, obs, kind) in picks {
                    let start = day - anchor_offset;
                    let batch = ctx.collate_gauge(&staid, start, adjoint.window_days)?;
                    let comids: Vec<i64> = batch.divide_comids.iter().map(|c| c.0).collect();
                    let gauge_row = batch.outflow_idx[0][0];
                    let adjacency = batch.adjacency.clone();
                    let tensors = batch.to_tensors::<Autodiff<I>>(device);
                    let lf = ctx.forward_with_inflow_leaf(&tensors, None);
                    let series: Vec<f32> = lf.gauge_series.clone().inner().into_data().to_vec::<f32>().unwrap();
                    let scalar = lf.gauge_series.clone().slice([0..1, t0..t0 + 1]).reshape([1]);
                    let grad = inflow_gradient::<I>(scalar, &lf.q_leaf);
                    let q_hourly: Vec<f32> = lf.q_hourly_inner.clone().into_data().to_vec::<f32>().unwrap();
                    let dist = dist_to_gauge(&adjacency, gauge_row);
                    let r = result.get_or_insert_with(|| {
                        new_result(arm, adjoint, &ctx, &g.staid, comids.clone(), dist.clone(), gauge_row, column_means(&q_hourly, grad.t, grad.n))
                    });
                    let (mass, mean_lag) = grad.kernel_moments(t0, adjoint.lag_days);
                    r.kernel.push(grad.kernel_daily(t0, adjoint.lag_days));
                    r.kernel_hourly.push(grad.kernel_hourly(t0, adjoint.lag_days));
                    r.kernel_mass.push(mass);
                    r.kernel_mean_lag_days.push(mean_lag);
                    r.q_prime_mean_by_anchor.push(column_means(&q_hourly, grad.t, grad.n));
                    r.anchors.push(AnchorRecord {
                        day_idx: day,
                        date: ctx.axis.start + Duration::days(day as i64),
                        obs,
                        kind: kind.to_string(),
                        window_start_day: start,
                        t0,
                    });
                    println!(
                        "  {} {} kernel {kind:>4} anchor {} (obs {obs:.1}) Q(t0)={:.2} reaches {} mass@gauge {:.3}",
                        arm.name,
                        g.staid,
                        ctx.axis.start + Duration::days(day as i64),
                        series[t0],
                        grad.n,
                        r.kernel_mass.last().unwrap()[gauge_row]
                    );
                    if !validated {
                        println!("  finite-difference gate on {}/{} anchor day {day} …", arm.name, g.staid);
                        let vr = finite_difference_gate::<I>(
                            &ctx,
                            &tensors,
                            GateInputs {
                                arm: &arm.name,
                                staid: &g.staid,
                                comids: &comids,
                                anchor_day_idx: day,
                                t0,
                                lag_days: adjoint.lag_days,
                                grad: &grad,
                                base_q_t0: series[t0],
                                q_hourly: &q_hourly,
                                dist_to_gauge_m: &dist,
                                gauge_row,
                                frac: adjoint.fd_perturb_fraction,
                                tol: adjoint.fd_rel_tol,
                            },
                        )?;
                        manifest.validation = Some(serde_json::to_value(&vr)?);
                        validated = true;
                        if !vr.passed {
                            return Err(format!(
                                "finite-difference gate FAILED on {}/{}: max rel_err {:.4} > tol {:.4}",
                                arm.name,
                                g.staid,
                                vr.checks.iter().map(|c| c.rel_err).fold(0.0f32, f32::max),
                                vr.tol
                            )
                            .into());
                        }
                        println!("  finite-difference gate PASSED");
                    }
                }
            }

            // ---------------- volume ----------------
            if adjoint.functionals.contains(&Functional::Volume) {
                let (start, _) = seasonal[0];
                let batch = ctx.collate_gauge(&staid, start, adjoint.window_days)?;
                let comids: Vec<i64> = batch.divide_comids.iter().map(|c| c.0).collect();
                let gauge_row = batch.outflow_idx[0][0];
                let adjacency = batch.adjacency.clone();
                let tensors = batch.to_tensors::<Autodiff<I>>(device);
                let lf = ctx.forward_with_inflow_leaf(&tensors, None);
                let scalar = lf.gauge_series.clone().slice([0..1, warm_h..n_hourly]).sum();
                let grad = inflow_gradient::<I>(scalar, &lf.q_leaf);
                let q_hourly: Vec<f32> = lf.q_hourly_inner.clone().into_data().to_vec::<f32>().unwrap();
                let r = result.get_or_insert_with(|| {
                    new_result(arm, adjoint, &ctx, &g.staid, comids.clone(), dist_to_gauge(&adjacency, gauge_row), gauge_row, column_means(&q_hourly, grad.t, grad.n))
                });
                r.volume_window_start_day = Some(start);
                r.volume_sens = grad.time_mean(warm_h, reduce_to);
                r.volume_profile = grad.reach_mean();
                let med = median(&r.volume_sens);
                println!("  {} {} volume: median sens {med:.4}, at gauge {:.4}", arm.name, g.staid, r.volume_sens[gauge_row]);
            }

            // ---------------- residual ----------------
            if adjoint.functionals.contains(&Functional::Residual) {
                for (start, date) in &seasonal {
                    let batch = ctx.collate_gauge(&staid, *start, adjoint.window_days)?;
                    let comids: Vec<i64> = batch.divide_comids.iter().map(|c| c.0).collect();
                    let gauge_row = batch.outflow_idx[0][0];
                    let adjacency = batch.adjacency.clone();
                    let obs_win: Vec<f32> = batch.observations.column(0).to_vec();
                    let tensors = batch.to_tensors::<Autodiff<I>>(device);
                    let lf = ctx.forward_with_inflow_leaf(&tensors, None);
                    let daily = ctx.daily(lf.gauge_series.clone());
                    let n_days = daily.dims()[1];
                    let daily_vals: Vec<f32> = daily.clone().inner().into_data().to_vec::<f32>().unwrap();
                    let valid: Vec<usize> = (ctx.warmup..n_days)
                        .filter(|&d| d < obs_win.len() && obs_win[d].is_finite() && obs_win[d] >= 0.0)
                        .collect();
                    let q_hourly: Vec<f32> = lf.q_hourly_inner.clone().into_data().to_vec::<f32>().unwrap();
                    let (t_h, n_r) = (lf.q_leaf.dims()[0], lf.q_leaf.dims()[1]);
                    let r = result.get_or_insert_with(|| {
                        new_result(arm, adjoint, &ctx, &g.staid, comids.clone(), dist_to_gauge(&adjacency, gauge_row), gauge_row, column_means(&q_hourly, t_h, n_r))
                    });
                    if valid.is_empty() {
                        manifest.notes.push(format!("{}/{}: residual window {date} has no valid observations", arm.name, g.staid));
                        r.residual_windows.push(WindowRecord { start_day: *start, start_date: *date, gauge_mean_residual: f32::NAN, gauge_mse: f32::NAN, n_valid: 0 });
                        r.residual_attr_by_window.push(vec![f32::NAN; n_r]);
                        continue;
                    }
                    // Squared-error functional: d[mean_valid (Qbar - obs)^2]/dq' =
                    // (2/n) Σ_d (Qbar_d - obs_d) · dQbar_d/dq'. The observations enter
                    // through the residual weights, so the gradient points at the
                    // reaches whose inflow the gauge would "correct" (positive ⇒
                    // this reach's water arrives when the gauge over-predicts).
                    let mut w = vec![0.0f32; n_days];
                    let mut obs_filled = vec![0.0f32; n_days];
                    for &d in &valid {
                        w[d] = 1.0 / valid.len() as f32;
                        obs_filled[d] = obs_win[d];
                    }
                    let gauge_mean_residual: f32 =
                        valid.iter().map(|&d| daily_vals[d] - obs_win[d]).sum::<f32>() / valid.len() as f32;
                    let gauge_mse: f32 =
                        valid.iter().map(|&d| (daily_vals[d] - obs_win[d]).powi(2)).sum::<f32>() / valid.len() as f32;
                    let w_t = Tensor::<Autodiff<I>, 1>::from_floats(w.as_slice(), device).reshape([1, n_days]);
                    let obs_t = Tensor::<Autodiff<I>, 1>::from_floats(obs_filled.as_slice(), device).reshape([1, n_days]);
                    let scalar = ((daily - obs_t).powf_scalar(2.0) * w_t).sum();
                    let grad = inflow_gradient::<I>(scalar, &lf.q_leaf);
                    r.residual_attr_by_window.push(grad.time_mean(warm_h, reduce_to));
                    r.residual_windows.push(WindowRecord {
                        start_day: *start,
                        start_date: *date,
                        gauge_mean_residual,
                        gauge_mse,
                        n_valid: valid.len(),
                    });
                    println!(
                        "  {} {} residual window {date}: mean(pred-obs) {gauge_mean_residual:+.3} m3/s, RMSE {:.3} over {} days",
                        arm.name,
                        g.staid,
                        gauge_mse.sqrt(),
                        valid.len()
                    );
                }
                if let Some(r) = result.as_mut() {
                    r.residual_attr = mean_over_windows(&r.residual_attr_by_window);
                }
            }

            match result {
                Some(r) => {
                    let nc = arm_dir.join("gauges").join(format!("{}.nc", g.staid));
                    write_gauge_netcdf(&nc, &r)?;
                    append_summary(&arm_dir.join("summary.csv"), &r)?;
                    println!("  {} {} done in {:.1} s → {}", arm.name, g.staid, t_g.elapsed().as_secs_f32(), nc.display());
                }
                None => manifest.notes.push(format!("{}/{}: nothing computed", arm.name, g.staid)),
            }
        }
        println!("=== arm {} done in {:.1} s ===", arm.name, t_arm.elapsed().as_secs_f32());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn new_result<I: Backend>(
    arm: &ResolvedArm,
    adjoint: &AdjointSpec,
    ctx: &InfluenceContext<I>,
    staid: &str,
    comids: Vec<i64>,
    dist_to_gauge_m: Vec<f32>,
    gauge_row: usize,
    q_prime_mean: Vec<f32>,
) -> GaugeResult {
    GaugeResult {
        staid: staid.to_string(),
        arm: arm.name.clone(),
        run_id: arm.run_id.clone(),
        checkpoint: arm.checkpoint_dir.display().to_string(),
        comids,
        dist_to_gauge_m,
        q_prime_mean,
        gauge_row,
        window_days: adjoint.window_days,
        lag_days: adjoint.lag_days,
        tail_days: adjoint.tail_days,
        tau: ctx.tau,
        warmup: ctx.warmup,
        anchors: vec![],
        kernel: vec![],
        kernel_hourly: vec![],
        kernel_mass: vec![],
        kernel_mean_lag_days: vec![],
        q_prime_mean_by_anchor: vec![],
        volume_window_start_day: None,
        volume_sens: vec![],
        volume_profile: vec![],
        residual_windows: vec![],
        residual_attr_by_window: vec![],
        residual_attr: vec![],
    }
}

fn mean_over_windows(rows: &[Vec<f32>]) -> Vec<f32> {
    if rows.is_empty() {
        return vec![];
    }
    let n = rows[0].len();
    (0..n)
        .map(|i| {
            let vals: Vec<f32> = rows.iter().map(|r| r[i]).filter(|v| v.is_finite()).collect();
            if vals.is_empty() { f32::NAN } else { vals.iter().sum::<f32>() / vals.len() as f32 }
        })
        .collect()
}

fn median(v: &[f32]) -> f32 {
    let mut s: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return f32::NAN;
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_respect_gap_and_bounds() {
        let mut obs = vec![1.0f32; 400];
        obs[100] = 100.0;
        obs[110] = 90.0; // too close to 100
        obs[300] = 80.0;
        obs[50] = 0.1;
        obs[52] = 0.2; // too close to 50
        obs[250] = f32::NAN;
        let a = select_anchors(&obs, 2, 1, 60, 82, 392);
        assert_eq!(a.len(), 3);
        assert_eq!(a[0], (100, 100.0, "high"));
        assert_eq!(a[1], (300, 80.0, "high"));
        // low: day 50 is below min_day 82 → the next lowest far from 100/300 is any 1.0 day ≥ 82 with gap ≥ 60
        assert_eq!(a[2].2, "low");
        assert!(a[2].0 >= 82 && (a[2].0 as i64 - 100).abs() >= 60 && (a[2].0 as i64 - 300).abs() >= 60);
    }

    #[test]
    fn gauge_list_marks_roles() {
        let spec = AdjointSpec {
            gauges: GaugeSpec { pairs: vec![["A".into(), "B".into()], ["B".into(), "C".into()]] },
            window_days: 90,
            anchors: AnchorSpec::default(),
            lag_days: 30,
            water_year: 2000,
            functionals: d_functionals(),
            anchor_min_gap_days: 60,
            tail_days: 7,
            fd_perturb_fraction: 0.05,
            fd_rel_tol: 0.05,
        };
        let g = gauge_list(&spec);
        assert_eq!(g.iter().map(|x| x.staid.as_str()).collect::<Vec<_>>(), vec!["A", "B", "C"]);
        assert_eq!(g[1].role, "downstream");
        assert_eq!(g[1].upstream, vec!["A".to_string()]);
        assert_eq!(g[2].upstream, vec!["B".to_string()]);
    }
}
