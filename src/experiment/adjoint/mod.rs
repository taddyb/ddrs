//! The adjoint influence-map study: gradient of routed gauge discharge with
//! respect to lateral inflow at every upstream reach and hour, per trained arm.
//!
//! Spec: `docs/superpowers/specs/2026-09-03-ddrs-experiment-adjoint-design.md` §2.

pub mod gauges;
pub mod hydraulics;
pub mod influence;
pub mod output;
pub mod validate;

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use chrono::{Duration, NaiveDate};
use serde::{Deserialize, Serialize};

use crate::data::dates::TimeAxis;
use crate::data::ids::Staid;

use self::gauges::{gauge_list_from_pairs, nested_reference_selection, read_gages_ii_class, write_gauges_csv, GaugeEntry};
use self::influence::{column_means, dist_to_gauge, downstream_rows, inflow_gradient, InfluenceContext};
use self::output::{append_summary, write_gauge_netcdf, AnchorRecord, GaugeResult, WindowRecord};
use self::hydraulics::{mean_reach_k_hours, path_travel_time_hours, reach_k_hours};
use self::validate::{finite_difference_gate, full_map_gate, write_full_map_csv, FullMapInputs, FullMapResult, GateInputs, ValidationResult};
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
    /// Optional full-map finite-difference validation (check 1).
    #[serde(default)]
    pub validation: Option<FullMapSpec>,
    /// Optional forward pulse traces (check 4): inject a sustained inflow
    /// perturbation at one reach and record ΔQ at every reach and hour.
    #[serde(default)]
    pub traces: Vec<TraceSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TraceSpec {
    pub staid: String,
    pub comid: i64,
    /// Added inflow, m³/s, sustained over `[from_day, to_day)` of the volume window.
    #[serde(default = "d_trace_delta")]
    pub delta_m3s: f32,
    #[serde(default = "d_trace_from")]
    pub from_day: usize,
    #[serde(default = "d_trace_to")]
    pub to_day: usize,
}
fn d_trace_delta() -> f32 { 1.0 }
fn d_trace_from() -> usize { 10 }
fn d_trace_to() -> usize { 40 }

/// Central-difference check of the kernel at every `reach_stride`-th reach and
/// each of `lags_hours`, on the first high-flow and first low-flow anchor of
/// every gauge. Pass: `pass_fraction` of checks within `rel_tol` or `abs_tol`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FullMapSpec {
    #[serde(default = "d_stride")]
    pub reach_stride: usize,
    #[serde(default = "d_lags")]
    pub lags_hours: Vec<usize>,
    #[serde(default = "d_fm_frac")]
    pub perturb_fraction: f32,
    #[serde(default = "d_fm_rel")]
    pub rel_tol: f32,
    #[serde(default = "d_fm_abs")]
    pub abs_tol: f32,
    #[serde(default = "d_fm_pass")]
    pub pass_fraction: f32,
    /// δ sweep (fractions of mean inflow) applied to failing entries.
    #[serde(default = "d_fm_sweep")]
    pub delta_sweep: Vec<f32>,
}
fn d_fm_sweep() -> Vec<f32> { vec![0.02, 0.05, 0.1, 0.2, 0.5, 1.0] }
fn d_stride() -> usize { 10 }
fn d_lags() -> Vec<usize> { vec![6, 48, 240] }
fn d_fm_frac() -> f32 { 0.1 }
fn d_fm_rel() -> f32 { 0.02 }
fn d_fm_abs() -> f32 { 1e-4 }
fn d_fm_pass() -> f32 { 0.95 }

fn d_window_days() -> usize { 365 }
fn d_lag_days() -> usize { 30 }
fn d_water_year() -> i32 { 2000 }
fn d_functionals() -> Vec<Functional> { vec![Functional::Kernel, Functional::Volume, Functional::Residual] }
fn d_anchor_min_gap_days() -> usize { 60 }
fn d_tail_days() -> usize { 7 }
fn d_fd_fraction() -> f32 { 0.05 }
fn d_fd_tol() -> f32 { 0.05 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum GaugeSource {
    /// `pairs:` lists `[upstream, downstream]` staids explicitly.
    #[default]
    Explicit,
    /// GAGES-II `Ref` downstream gauges with ≥1 nested training gauge
    /// (spec §2.1); requires `gages_ii_dbf`.
    NestedReference,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GaugeSpec {
    #[serde(default)]
    pub source: GaugeSource,
    #[serde(default)]
    pub pairs: Vec<[String; 2]>,
    /// GAGES-II point shapefile dbf (STAID, CLASS columns).
    #[serde(default)]
    pub gages_ii_dbf: Option<PathBuf>,
    /// Cap on downstream gauges (sorted by staid) for smoke runs.
    #[serde(default)]
    pub max_downstream: Option<usize>,
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
    /// Arms run concurrently, one thread each, up to this many at a time.
    pub jobs: usize,
    /// Select gauges, write `gauges.csv`, and stop.
    pub dry_run: bool,
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

/// Per-arm report merged into the manifest after the threads join.
struct ArmReport {
    name: String,
    notes: Vec<String>,
    validation: Option<ValidationResult>,
    full_maps: Vec<FullMapResult>,
    n_done: usize,
    secs: f32,
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
    I::Device: 'static + Send + Sync,
{
    if adjoint.window_days < adjoint.tail_days + 2 {
        return Err("adjoint.window_days must exceed tail_days + 1".into());
    }
    if adjoint.lag_days > adjoint.window_days - 1 - adjoint.tail_days {
        return Err("adjoint.lag_days must fit before the anchor: lag_days <= window_days - 1 - tail_days".into());
    }
    if arms.is_empty() {
        return Err("no arms selected".into());
    }

    // ---- all arms must share the gauge CSV (population-matched by construction) ----
    let mut gages_csv: Vec<(String, PathBuf)> = Vec::new();
    for arm in arms {
        let cfg = crate::config::Config::from_yaml_file_with_mode(&arm.config_path, crate::config::ConfigMode::Testing)
            .map_err(|e| format!("arm `{}`: {e}", arm.name))?;
        let g = cfg.data_sources.as_ref().map(|d| d.gages.clone()).unwrap_or_default();
        gages_csv.push((arm.name.clone(), g));
    }
    if gages_csv.iter().any(|(_, g)| g != &gages_csv[0].1) {
        let list: Vec<String> = gages_csv.iter().map(|(a, g)| format!("{a}: {}", g.display())).collect();
        return Err(format!(
            "arms were trained on different gauge CSVs — cross-arm comparison would not be population-matched:\n  {}",
            list.join("\n  ")
        )
        .into());
    }

    // ---- population (from the first arm's dataset; all arms share the gauge CSV) ----
    let mut gauges: Vec<GaugeEntry> = match adjoint.gauges.source {
        GaugeSource::Explicit => {
            if adjoint.gauges.pairs.is_empty() {
                return Err("adjoint.gauges.pairs is empty".into());
            }
            gauge_list_from_pairs(&adjoint.gauges.pairs)
        }
        GaugeSource::NestedReference => {
            let dbf = adjoint
                .gauges
                .gages_ii_dbf
                .as_ref()
                .ok_or("gauges.source: nested-reference requires gauges.gages_ii_dbf")?;
            let class = read_gages_ii_class(dbf)?;
            println!("GAGES-II classes: {} gauges read from {}", class.len(), dbf.display());
            let ctx = InfluenceContext::<I>::open(&arms[0], device, opts.force_cpu)?;
            let list = nested_reference_selection(&ctx.dataset, &class, adjoint.gauges.max_downstream)?;
            let n_down = list.iter().filter(|g| g.role == "downstream").count();
            println!(
                "nested-reference selection: {} downstream Ref gauges with nested training gauges, {} gauges total",
                n_down,
                list.len()
            );
            list
        }
    };
    if let Some(k) = opts.max_gauges {
        gauges.truncate(k);
    }
    write_gauges_csv(&out_dir.join("gauges.csv"), &gauges)?;
    println!(
        "adjoint: {} arm(s), {} gauge(s), functionals {:?}, window {} d, lag {} d, anchors {}h/{}l, jobs {}",
        arms.len(),
        gauges.len(),
        adjoint.functionals,
        adjoint.window_days,
        adjoint.lag_days,
        adjoint.anchors.high,
        adjoint.anchors.low,
        opts.jobs.max(1)
    );
    if opts.dry_run {
        println!("dry run: gauges.csv written, stopping");
        return Ok(());
    }

    // ---- arms: one thread each, `jobs` at a time ----
    let jobs = opts.jobs.max(1);
    let mut reports: Vec<ArmReport> = Vec::new();
    for chunk in arms.chunks(jobs) {
        let chunk_reports: Vec<Result<ArmReport, BoxError>> = std::thread::scope(|s| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|arm| {
                    let gauges = &gauges;
                    let device = device.clone();
                    s.spawn(move || run_arm::<I>(adjoint, arm, gauges, out_dir, opts, &device))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("arm thread panicked".into())))
                .collect()
        });
        for r in chunk_reports {
            reports.push(r?);
        }
    }

    let mut validations = Vec::new();
    for r in reports {
        println!("=== arm {} done: {} gauge(s) in {:.1} s ===", r.name, r.n_done, r.secs);
        manifest.notes.extend(r.notes);
        if let Some(v) = r.validation {
            validations.push(serde_json::to_value(v)?);
        }
        for fm in r.full_maps {
            validations.push(serde_json::to_value(fm)?);
        }
    }
    if !validations.is_empty() {
        manifest.validation = Some(serde_json::Value::Array(validations));
    }
    Ok(())
}

fn run_arm<I: Backend + 'static>(
    adjoint: &AdjointSpec,
    arm: &ResolvedArm,
    gauges: &[GaugeEntry],
    out_dir: &Path,
    opts: &AdjointOptions,
    device: &I::Device,
) -> Result<ArmReport, BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let t_arm = Instant::now();
    println!("=== arm {} (run {}, checkpoint {}) start ===", arm.name, arm.run_id, arm.checkpoint_label);
    let ctx = InfluenceContext::<I>::open(arm, device, opts.force_cpu)?;
    let arm_dir = out_dir.join(&arm.name);
    std::fs::create_dir_all(arm_dir.join("gauges"))?;
    let seasonal = seasonal_window_starts(&ctx.axis, adjoint.water_year, adjoint.window_days)?;
    let mut notes = Vec::new();
    let mut validation: Option<ValidationResult> = None;
    let mut full_maps: Vec<FullMapResult> = Vec::new();
    let mut n_done = 0;
    let mut validate_pending = !opts.skip_validate;
    if adjoint.validation.is_some() {
        std::fs::create_dir_all(arm_dir.join("validation"))?;
    }
    for (gi, g) in gauges.iter().enumerate() {
        let t_g = Instant::now();
        match run_gauge::<I>(&ctx, adjoint, arm, g, &arm_dir, &seasonal, validate_pending) {
            Ok((vr, gauge_notes, fms)) => {
                notes.extend(gauge_notes);
                full_maps.extend(fms);
                if let Some(v) = vr {
                    if !v.passed {
                        return Err(format!(
                            "finite-difference gate FAILED on {}/{}: max rel_err {:.4} > tol {:.4}",
                            arm.name,
                            g.staid,
                            v.checks.iter().map(|c| c.rel_err).fold(0.0f32, f32::max),
                            v.tol
                        )
                        .into());
                    }
                    validation = Some(v);
                    validate_pending = false;
                }
                n_done += 1;
                println!(
                    "  [{}] {} {} done in {:.1} s ({} of {})",
                    arm.name,
                    g.staid,
                    g.role,
                    t_g.elapsed().as_secs_f32(),
                    gi + 1,
                    gauges.len()
                );
            }
            Err(e) => {
                let msg = format!("{}/{}: FAILED — {e}", arm.name, g.staid);
                eprintln!("  {msg}");
                notes.push(msg);
            }
        }
    }
    Ok(ArmReport { name: arm.name.clone(), notes, validation, full_maps, n_done, secs: t_arm.elapsed().as_secs_f32() })
}

/// One gauge under one arm: all functionals, netCDF + summary row.
fn run_gauge<I: Backend + 'static>(
    ctx: &InfluenceContext<I>,
    adjoint: &AdjointSpec,
    arm: &ResolvedArm,
    g: &GaugeEntry,
    arm_dir: &Path,
    seasonal: &[(usize, NaiveDate)],
    validate: bool,
) -> Result<(Option<ValidationResult>, Vec<String>, Vec<FullMapResult>), BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let device = &ctx.device;
    let mut full_maps: Vec<FullMapResult> = Vec::new();
    let mut full_map_done: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    let n_hourly = (adjoint.window_days - 1) * 24;
    let anchor_offset = adjoint.window_days - 1 - adjoint.tail_days;
    let t0 = anchor_offset * 24 + 12;
    let warm_h = ctx.warmup * 24;
    let tail_h = adjoint.tail_days * 24;
    let reduce_to = n_hourly - tail_h;
    if warm_h >= reduce_to {
        return Err("warmup + tail leave no source hours to reduce over".into());
    }
    let staid = Staid::new(&g.staid);
    let obs_full = ctx.gauge_observations(&staid)?;
    let mut notes = Vec::new();
    let mut validation = None;
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
            notes.push(format!("{}/{}: no valid anchor days", arm.name, g.staid));
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
                new_result(arm, adjoint, ctx, &g.staid, comids.clone(), dist.clone(), gauge_row, column_means(&q_hourly, grad.t, grad.n), downstream_rows(&adjacency))
            });
            let (mass, mean_lag) = grad.kernel_moments(t0, adjoint.lag_days);
            r.kernel.push(grad.kernel_daily(t0, adjoint.lag_days));
            r.kernel_hourly.push(grad.kernel_hourly(t0, adjoint.lag_days));
            r.kernel_mass.push(mass);
            r.kernel_mean_lag_days.push(mean_lag);
            r.q_prime_mean_by_anchor.push(column_means(&q_hourly, grad.t, grad.n));
            // Check 2: hydraulic travel time Σ K = Σ L/c along the path, from
            // the trained geometry at (a) the anchor hour and (b) the mean over
            // the lag horizon, in days.
            {
                let lag_from = t0.saturating_sub(24 * adjoint.lag_days);
                let k_mean = mean_reach_k_hours::<I>(&ctx.cfg, &lf.n_phys, &lf.p_phys, &lf.q_phys, &lf.runoff_inner, &lf.slope, &lf.length, lag_from, t0 + 1);
                let n_reach = grad.n;
                let q_t0 = lf.runoff_inner.clone().slice([0..n_reach, t0..t0 + 1]).reshape([n_reach]);
                let k_t0 = reach_k_hours::<I>(&ctx.cfg, &lf.n_phys, &lf.p_phys, &lf.q_phys, q_t0, &lf.slope, &lf.length);
                let path_mean = path_travel_time_hours(&adjacency, gauge_row, &k_mean);
                let path_t0 = path_travel_time_hours(&adjacency, gauge_row, &k_t0);
                r.hydraulic_lag_days.push(path_mean.iter().map(|h| h / 24.0).collect());
                r.hydraulic_lag_t0_days.push(path_t0.iter().map(|h| h / 24.0).collect());
            }
            // Check 1: full-map central finite differences, once per anchor kind.
            if let Some(fm) = adjoint.validation.as_ref() {
                if !full_map_done.contains(kind) {
                    let res = full_map_gate::<I>(
                        ctx,
                        &tensors,
                        FullMapInputs {
                            arm: &arm.name,
                            staid: &g.staid,
                            comids: &comids,
                            anchor_day_idx: day,
                            anchor_kind: kind,
                            t0,
                            base_q_t0: series[t0],
                            grad: &grad,
                            q_hourly: &q_hourly,
                            dist_to_gauge_m: &dist,
                            reach_stride: fm.reach_stride,
                            lags_hours: &fm.lags_hours,
                            perturb_fraction: fm.perturb_fraction,
                            rel_tol: fm.rel_tol,
                            abs_tol: fm.abs_tol,
                            pass_fraction: fm.pass_fraction,
                            delta_sweep: &fm.delta_sweep,
                        },
                    )?;
                    write_full_map_csv(&arm_dir.join("validation").join(format!("{}_{}_day{}.csv", g.staid, kind, day)), &res)?;
                    full_maps.push(res);
                    full_map_done.insert(kind);
                }
            }
            r.anchors.push(AnchorRecord {
                day_idx: day,
                date: ctx.axis.start + Duration::days(day as i64),
                obs,
                kind: kind.to_string(),
                window_start_day: start,
                t0,
            });
            if validate && validation.is_none() {
                println!("  [{}] finite-difference gate on {} anchor day {day} …", arm.name, g.staid);
                let vr = finite_difference_gate::<I>(
                    ctx,
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
                println!("  [{}] finite-difference gate {}", arm.name, if vr.passed { "PASSED" } else { "FAILED" });
                validation = Some(vr);
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
            new_result(arm, adjoint, ctx, &g.staid, comids.clone(), dist_to_gauge(&adjacency, gauge_row), gauge_row, column_means(&q_hourly, grad.t, grad.n), downstream_rows(&adjacency))
        });
        r.volume_window_start_day = Some(start);
        r.volume_sens = grad.time_mean(warm_h, reduce_to);
        r.volume_profile = grad.reach_mean();
        // Check 3: lag-aware volume sensitivity (per-reach kernel mean lag,
        // max over anchors, from the kernel functional above).
        let n_reach = grad.n;
        r.volume_sens_lag_aware = (0..n_reach)
            .map(|i| {
                let lag_h = r
                    .kernel_mean_lag_days
                    .iter()
                    .map(|a| a[i])
                    .filter(|v| v.is_finite())
                    .fold(f32::NAN, |m, v| if m.is_nan() { v } else { m.max(v) })
                    * 24.0;
                if !lag_h.is_finite() {
                    return f32::NAN;
                }
                // Negative mean lags occur where the kernel oscillates; treat as 0.
                // Never average past `reduce_to` (the raw functional's bound).
                let lag_h = lag_h.max(0.0);
                let to = ((n_hourly as f32 - tail_h as f32 - lag_h).floor() as isize).min(reduce_to as isize);
                if to <= warm_h as isize + 24 {
                    return f32::NAN;
                }
                let to = to as usize;
                let mut acc = 0.0f32;
                for h in warm_h..to {
                    acc += grad.at(h, i);
                }
                acc / (to - warm_h) as f32
            })
            .collect();
        // Check 4: fraction of timesteps at the discharge clamp floor.
        let floor = ctx.cfg.params.attribute_minimums.discharge * 1.001;
        let runoff: Vec<f32> = lf.runoff_inner.clone().into_data().to_vec::<f32>().unwrap(); // (N, T) row-major
        let t_all = lf.runoff_inner.dims()[1];
        r.floor_frac = (0..n_reach)
            .map(|i| {
                let row = &runoff[i * t_all..(i + 1) * t_all];
                row[warm_h.min(t_all)..].iter().filter(|q| **q <= floor).count() as f32 / (t_all - warm_h.min(t_all)).max(1) as f32
            })
            .collect();
        // Intermittency: fraction of source hours with inflow above the floor.
        {
            let q: Vec<f32> = lf.q_hourly_inner.clone().into_data().to_vec::<f32>().unwrap(); // (T, N)
            let (t_h, n_r) = (lf.q_leaf.dims()[0], lf.q_leaf.dims()[1]);
            let lo = warm_h.min(t_h);
            let hi = reduce_to.min(t_h);
            r.q_prime_wet_frac = (0..n_r)
                .map(|i| (lo..hi).filter(|&h| q[h * n_r + i] > floor).count() as f32 / (hi - lo).max(1) as f32)
                .collect();
            r.volume_sens_wet = (0..n_r)
                .map(|i| {
                    let wet: Vec<usize> = (lo..hi).filter(|&h| q[h * n_r + i] > floor).collect();
                    if wet.is_empty() { f32::NAN } else { wet.iter().map(|&h| grad.at(h, i)).sum::<f32>() / wet.len() as f32 }
                })
                .collect();
        }
        // Check 4 traces: sustained pulse at one reach, ΔQ everywhere.
        for tr in adjoint.traces.iter().filter(|t| t.staid == g.staid) {
            let Some(reach) = comids.iter().position(|c| *c == tr.comid) else {
                notes.push(format!("{}/{}: trace COMID {} not in subgraph", arm.name, g.staid, tr.comid));
                continue;
            };
            let base_q: Vec<f32> = lf.q_hourly_inner.clone().into_data().to_vec::<f32>().unwrap(); // (T, N)
            let (t_h, n_r) = (lf.q_leaf.dims()[0], lf.q_leaf.dims()[1]);
            let mut pert = base_q.clone();
            let (h0, h1) = (tr.from_day * 24, (tr.to_day * 24).min(t_h));
            for h in h0..h1 {
                pert[h * n_r + reach] += tr.delta_m3s;
            }
            let q_pert: Tensor<I, 2> = Tensor::<I, 1>::from_floats(pert.as_slice(), device).reshape([t_h, n_r]);
            let lf2 = ctx.forward_with_inflow_leaf(&tensors, Some(q_pert));
            let run2: Vec<f32> = lf2.runoff_inner.into_data().to_vec::<f32>().unwrap(); // (N, T)
            let dq: Vec<f32> = run2.iter().zip(&runoff).map(|(a, b)| a - b).collect();
            let tdir = arm_dir.join("trace");
            std::fs::create_dir_all(&tdir)?;
            let path = tdir.join(format!("{}_{}.nc", g.staid, tr.comid));
            output::write_trace_netcdf(&path, &g.staid, tr.comid, reach, gauge_row, tr.delta_m3s, h0, h1, &comids, &r.dist_to_gauge_m, &r.downstream_row, &runoff, &dq, n_reach, t_all)?;
            let injected = tr.delta_m3s * (h1 - h0) as f32;
            let at_gauge: f32 = (0..t_all).map(|t| dq[gauge_row * t_all + t]).sum();
            let at_reach: f32 = (0..t_all).map(|t| dq[reach * t_all + t]).sum();
            println!(
                "  [{}] trace {} COMID {} row {}: injected {:.1} m3/s·h; ΣΔQ at reach {:.1} ({:.0} %), at gauge {:.1} ({:.0} %)",
                arm.name, g.staid, tr.comid, reach, injected, at_reach, 100.0 * at_reach / injected, at_gauge, 100.0 * at_gauge / injected
            );
        }
    }

    // ---------------- residual (squared error) ----------------
    if adjoint.functionals.contains(&Functional::Residual) {
        for (start, date) in seasonal {
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
                new_result(arm, adjoint, ctx, &g.staid, comids.clone(), dist_to_gauge(&adjacency, gauge_row), gauge_row, column_means(&q_hourly, t_h, n_r), downstream_rows(&adjacency))
            });
            if valid.is_empty() {
                notes.push(format!("{}/{}: residual window {date} has no valid observations", arm.name, g.staid));
                r.residual_windows.push(WindowRecord { start_day: *start, start_date: *date, gauge_mean_residual: f32::NAN, gauge_mse: f32::NAN, n_valid: 0 });
                r.residual_attr_by_window.push(vec![f32::NAN; n_r]);
                continue;
            }
            // d[mean_valid (Qbar - obs)^2]/dq' = (2/n) Σ_d (Qbar_d - obs_d) · dQbar_d/dq'.
            // Positive ⇒ this reach's water arrives when the gauge over-predicts.
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
        }
        None => notes.push(format!("{}/{}: nothing computed", arm.name, g.staid)),
    }
    Ok((validation, notes, full_maps))
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
    downstream_row: Vec<i32>,
) -> GaugeResult {
    GaugeResult {
        staid: staid.to_string(),
        arm: arm.name.clone(),
        run_id: arm.run_id.clone(),
        checkpoint: arm.checkpoint_dir.display().to_string(),
        comids,
        downstream_row,
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
        hydraulic_lag_days: vec![],
        hydraulic_lag_t0_days: vec![],
        volume_window_start_day: None,
        volume_sens: vec![],
        volume_sens_lag_aware: vec![],
        q_prime_wet_frac: vec![],
        volume_sens_wet: vec![],
        floor_frac: vec![],
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
        assert_eq!(a[2].2, "low");
        assert!(a[2].0 >= 82 && (a[2].0 as i64 - 100).abs() >= 60 && (a[2].0 as i64 - 300).abs() >= 60);
    }

    #[test]
    fn gauge_spec_defaults_to_explicit() {
        let y = "gauges:\n  pairs: [[\"A\", \"B\"]]\n";
        let s: AdjointSpec = serde_yaml::from_str(y).unwrap();
        assert_eq!(s.gauges.source, GaugeSource::Explicit);
        assert_eq!(s.window_days, 365);
        let y2 = "gauges:\n  source: nested-reference\n  gages_ii_dbf: /tmp/x.dbf\n";
        let s2: AdjointSpec = serde_yaml::from_str(y2).unwrap();
        assert_eq!(s2.gauges.source, GaugeSource::NestedReference);
    }
}
