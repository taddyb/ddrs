//! Phase 3 follow-up experiment: does turning on `DisaggHead`'s
//! `boundary_blend` (already-built, unused in the baseline pretrain run)
//! reduce the day-boundary step artifacts visible in the storm-comparison
//! plots? `boundary_blend` only has an effect when a SINGLE `forward()` call
//! processes `d_use > 1` consecutive days (it blends adjacent days'
//! boundary-hour probabilities) -- the baseline run trained on independent
//! single-day rows (`d_use` always 1), so blend was structurally a no-op
//! there regardless of its config value. This binary trains on genuine
//! CONSECUTIVE-day windows (`extract_complete_day_windows`) so the blend
//! mechanism is actually exercised, and compares against the un-blended
//! baseline on the same real storm days.
//!
//! Per-day logits are still computed fully independently per day (the
//! `Linear`/`KanLayer` stages never mix across days) -- `boundary_blend` is
//! the ONLY cross-day channel this experiment tests, not a revival of the
//! pre-KAN `[d-1,d,d+1]` window feature (that would need new INPUT
//! features, a separate, bigger architecture change).
//!
//!   cargo run --release --bin pretrain_disagg_blend -- --boundary-blend 0.5

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::optim::{GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};
use chrono::NaiveDate;
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ddrs::data::ids::Staid;
use ddrs::data::store::gage_csv::{GageMetadata, GageRow};
use ddrs::data::store::icechunk::UsgsObservationsStore;
use ddrs::data::store::CamelsHourlyStore;
use ddrs::nn::{DisaggHead, DisaggHeadConfig};
use ddrs::pretrain::{
    build_split_manifest, extract_complete_day_windows, normalize_gauge_precip, qobs_mm_hr_to_m3s,
    reconcile_gauge, slice_precip_features,
};
use ddrs::training::build_adam;

type I = NdArray<f32>;
type AutoI = Autodiff<I>;

const GAGES_CSV: &str = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";
const CAMELS_NC: &str = "/mnt/ssd1/data/camels_hourly/usgs-streamflow-nldas_hourly.nc";
const USGS_DAILY: &str = "/mnt/ssd1/data/icechunk/usgs_daily_observations";
const SEED: u64 = 42;

const TRAIN_START: (i32, u32, u32) = (1998, 1, 1);
const TRAIN_END: (i32, u32, u32) = (2013, 12, 31);
const HELDOUT_START: (i32, u32, u32) = (2014, 1, 1);
const HELDOUT_END: (i32, u32, u32) = (2018, 12, 31);

const N_STEPS: usize = 2000;
const BATCH_SIZE: usize = 64;
const VAL_EVERY: usize = 100;
const MIN_DAILY_Q_M3S: f32 = 0.01;
const WINDOW_DAYS: usize = 2;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value_t = 0.5)]
    boundary_blend: f32,
    #[arg(long, default_value = "output/disagg_pretrain/best_disagg_blend")]
    output_checkpoint: String,
    /// Skip training and re-evaluate an already-saved checkpoint (training
    /// is deterministic given the seed, so this is only useful after an
    /// evaluation-only bug fix).
    #[arg(long)]
    skip_train: bool,
}

/// A `WINDOW_DAYS`-consecutive-day training window: `daily_q_m3s[d]` and
/// `target_hourly_m3s[d*24+h]` for `d in 0..WINDOW_DAYS`.
#[derive(Clone)]
struct Row {
    daily_q_m3s: [f32; WINDOW_DAYS],
    precip_feat: Vec<f32>,        // len WINDOW_DAYS*24
    target_hourly_m3s: Vec<f32>,  // len WINDOW_DAYS*24
    daily_precip_total_mm: [f32; WINDOW_DAYS],
}

fn ymd(t: (i32, u32, u32)) -> NaiveDate {
    NaiveDate::from_ymd_opt(t.0, t.1, t.2).unwrap()
}

fn gauge_rows(camels: &CamelsHourlyStore, staid: &Staid, drain_sqkm: f64, start: NaiveDate, end: NaiveDate) -> Vec<Row> {
    let start_dt = start.and_hms_opt(0, 0, 0).unwrap();
    let n_days = (end - start).num_days() as usize + 1;
    let n_hours = n_days * 24;
    let Ok((qobs, precip)) = camels.read_window(start_dt, n_hours, std::slice::from_ref(staid)) else {
        return Vec::new();
    };

    let qobs_m3s: ndarray::Array1<f32> = qobs
        .column(0)
        .mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, drain_sqkm) } else { f32::NAN });
    let windows = extract_complete_day_windows(&qobs_m3s, WINDOW_DAYS);
    let windows: Vec<_> = windows
        .into_iter()
        .filter(|w| w.iter().all(|r| r.daily_q_m3s >= MIN_DAILY_Q_M3S))
        .collect();

    let precip_col: ndarray::Array1<f32> = precip.column(0).to_owned();
    let precip_norm = normalize_gauge_precip(precip_col.clone());

    windows
        .into_iter()
        .map(|w| {
            let mut daily_q_m3s = [0f32; WINDOW_DAYS];
            let mut daily_precip_total_mm = [0f32; WINDOW_DAYS];
            let mut target_hourly_m3s = Vec::with_capacity(WINDOW_DAYS * 24);
            let mut precip_feat = Vec::with_capacity(WINDOW_DAYS * 24);
            for (i, day_row) in w.iter().enumerate() {
                daily_q_m3s[i] = day_row.daily_q_m3s;
                target_hourly_m3s.extend_from_slice(&day_row.target_hourly_m3s);
                let feats = slice_precip_features(&precip_norm, &[day_row.day_index]);
                precip_feat.extend_from_slice(&feats[0]);
                daily_precip_total_mm[i] = (0..24)
                    .map(|h| precip_col[day_row.day_index * 24 + h].max(0.0))
                    .sum();
            }
            // Mass-balance invariant, explicit for EACH day in the window.
            for (i, day_row) in w.iter().enumerate() {
                let block = &target_hourly_m3s[i * 24..i * 24 + 24];
                let mean: f32 = block.iter().sum::<f32>() / 24.0;
                assert!(
                    (mean - day_row.daily_q_m3s).abs() < 1e-3,
                    "mass-balance invariant violated: day {} mean={mean} != daily_q={}",
                    day_row.day_index,
                    day_row.daily_q_m3s
                );
            }
            Row {
                daily_q_m3s,
                precip_feat,
                target_hourly_m3s,
                daily_precip_total_mm,
            }
        })
        .collect()
}

fn shape_of(row_daily_q: &[f32; WINDOW_DAYS], target_hourly: &[f32], h_global: usize) -> f32 {
    let d = h_global / 24;
    target_hourly[h_global] / (24.0 * row_daily_q[d])
}

fn train_step(
    head: DisaggHead<AutoI>,
    optimizer: &mut impl Optimizer<DisaggHead<AutoI>, AutoI>,
    rows: &[&Row],
    device: &<AutoI as BackendTypes>::Device,
    lr: f64,
) -> (DisaggHead<AutoI>, f32) {
    let n = rows.len();
    let n_hourly = WINDOW_DAYS * 24;
    let mut q_buf = vec![0f32; WINDOW_DAYS * n];
    let mut precip_buf = Vec::with_capacity(n * n_hourly);
    let mut target_shape_buf = Vec::with_capacity(n * n_hourly);
    for (col, r) in rows.iter().enumerate() {
        for d in 0..WINDOW_DAYS {
            q_buf[d * n + col] = r.daily_q_m3s[d];
        }
        precip_buf.extend_from_slice(&r.precip_feat);
        for h in 0..n_hourly {
            target_shape_buf.push(shape_of(&r.daily_q_m3s, &r.target_hourly_m3s, h));
        }
    }
    let daily_q = Tensor::<AutoI, 2>::from_data(TensorData::new(q_buf, [WINDOW_DAYS, n]), device);
    let precip = Tensor::<AutoI, 2>::from_data(TensorData::new(precip_buf, [n, n_hourly]), device).transpose();
    let target_shape = Tensor::<AutoI, 2>::from_data(TensorData::new(target_shape_buf, [n, n_hourly]), device).transpose();

    let pred_hourly = head.forward(daily_q.clone(), precip, n_hourly); // (n_hourly, n)
    // Broadcast daily_q (WINDOW_DAYS, n) -> (n_hourly, n) by repeating each day's row 24x.
    let daily_q_rep = daily_q.reshape([WINDOW_DAYS, 1, n]).repeat_dim(1, 24).reshape([n_hourly, n]);
    let pred_shape = pred_hourly / (daily_q_rep * 24.0);
    let diff = pred_shape - target_shape;
    let loss = diff.powf_scalar(2.0).mean();
    let loss_val: f32 = loss.clone().into_scalar();

    let grads = GradientsParams::from_grads(loss.backward(), &head);
    let head = optimizer.step(lr, head, grads);
    (head, loss_val)
}

/// Forward one window through `head` and return (per-day shape MSE, per-day predicted shape).
fn eval_row(head: &DisaggHead<I>, r: &Row, device: &<I as BackendTypes>::Device) -> Vec<(f32, [f32; 24])> {
    let n_hourly = WINDOW_DAYS * 24;
    let daily_q = Tensor::<I, 2>::from_data(TensorData::new(r.daily_q_m3s.to_vec(), [WINDOW_DAYS, 1]), device);
    let precip = Tensor::<I, 2>::from_data(TensorData::new(r.precip_feat.clone(), [n_hourly, 1]), device);
    let pred = head.forward(daily_q, precip, n_hourly);
    let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();

    (0..WINDOW_DAYS)
        .map(|d| {
            let mut shape = [0f32; 24];
            let mut mse = 0f32;
            for h in 0..24 {
                let g = d * 24 + h;
                shape[h] = pred_vec[g] / (24.0 * r.daily_q_m3s[d]);
                let target_shape_h = shape_of(&r.daily_q_m3s, &r.target_hourly_m3s, g);
                mse += (shape[h] - target_shape_h).powi(2);
            }
            (mse / 24.0, shape)
        })
        .collect()
}

/// Flat-shape (1/24 every hour) MSE against a GAUGE's entire heldout period
/// (all windows/days), NOT a single window -- must match the baseline
/// binary's `shape_mse_vs_flat` methodology (a stable, many-day denominator)
/// or NSE values from the two experiments are not comparable. A per-window
/// (48-hour) denominator is a noisy single-instance estimate and was the
/// bug that made this run's aggregate numbers look wildly different from
/// the baseline's.
fn flat_mse_for_gauge(rows: &[Row]) -> f32 {
    let mut total = 0f32;
    let mut n = 0usize;
    let n_hourly = WINDOW_DAYS * 24;
    for r in rows {
        for h in 0..n_hourly {
            let target_shape_h = shape_of(&r.daily_q_m3s, &r.target_hourly_m3s, h);
            total += (1.0 / 24.0 - target_shape_h).powi(2);
            n += 1;
        }
    }
    total / n as f32
}

fn circ_dist(a: usize, b: usize) -> usize {
    let d = (a as i64 - b as i64).unsigned_abs() as usize;
    d.min(24 - d)
}
fn argmax(v: &[f32; 24]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
}
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { return f32::NAN; }
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    println!("=== Phase 3 follow-up: boundary_blend={} (window_days={WINDOW_DAYS}) ===\n", cli.boundary_blend);

    let gages = GageMetadata::open(GAGES_CSV)?;
    let camels = CamelsHourlyStore::open(CAMELS_NC)?;
    let obs = UsgsObservationsStore::open(USGS_DAILY)?;

    let overlap: Vec<&GageRow> = gages.rows.iter().filter(|r| camels.index.position(&r.staid).is_some()).collect();
    let mut kept: Vec<GageRow> = Vec::new();
    for row in &overlap {
        let start_dt = ymd(TRAIN_START).and_hms_opt(0, 0, 0).unwrap();
        let n_days = (ymd(TRAIN_END) - ymd(TRAIN_START)).num_days() as usize + 1;
        let Ok((qobs, _)) = camels.read_window(start_dt, n_days * 24, std::slice::from_ref(&row.staid)) else { continue };
        let qobs_m3s: ndarray::Array1<f32> = qobs.column(0).mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, row.drain_sqkm) } else { f32::NAN });
        let windows = extract_complete_day_windows(&qobs_m3s, 1);
        if windows.len() < 30 { continue }
        let daily_vals: Vec<(NaiveDate, f32)> = windows.iter().map(|w| (ymd(TRAIN_START) + chrono::Duration::days(w[0].day_index as i64), w[0].daily_q_m3s)).collect();
        if let Ok(result) = reconcile_gauge(&row.staid, &daily_vals, &obs) {
            if result.keep { kept.push((*row).clone()); }
        }
    }
    println!("reconciled + kept: {}/{}\n", kept.len(), overlap.len());

    let split = build_split_manifest(&kept, SEED);
    println!("split: train={} val={} test={}", split.train.len(), split.val.len(), split.test.len());
    let gauge_by_staid: HashMap<&Staid, &GageRow> = kept.iter().map(|g| (&g.staid, g)).collect();

    let mut train_rows: Vec<Row> = Vec::new();
    for staid in &split.train {
        let g = gauge_by_staid[staid];
        train_rows.extend(gauge_rows(&camels, staid, g.drain_sqkm, ymd(TRAIN_START), ymd(TRAIN_END)));
    }
    let mut val_rows: Vec<Row> = Vec::new();
    for staid in &split.val {
        let g = gauge_by_staid[staid];
        val_rows.extend(gauge_rows(&camels, staid, g.drain_sqkm, ymd(TRAIN_START), ymd(TRAIN_END)));
    }
    let mut test_train_period_by_gauge: HashMap<Staid, Vec<Row>> = HashMap::new();
    let mut test_heldout_by_gauge: HashMap<Staid, Vec<Row>> = HashMap::new();
    for staid in &split.test {
        let g = gauge_by_staid[staid];
        test_train_period_by_gauge.insert(staid.clone(), gauge_rows(&camels, staid, g.drain_sqkm, ymd(TRAIN_START), ymd(TRAIN_END)));
        test_heldout_by_gauge.insert(staid.clone(), gauge_rows(&camels, staid, g.drain_sqkm, ymd(HELDOUT_START), ymd(HELDOUT_END)));
    }
    println!(
        "windows: train={} val={} test_train_period={} test_heldout={}\n",
        train_rows.len(), val_rows.len(),
        test_train_period_by_gauge.values().map(|v| v.len()).sum::<usize>(),
        test_heldout_by_gauge.values().map(|v| v.len()).sum::<usize>(),
    );

    let device: <AutoI as BackendTypes>::Device = Default::default();
    <AutoI as Backend>::seed(&device, SEED);
    let mut head: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED).with_boundary_blend(cli.boundary_blend).init(&device);
    let mut optimizer = build_adam::<DisaggHead<AutoI>, AutoI>();
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xC0FFEE);

    let mut best_val_loss = f32::INFINITY;
    let mut best_head_record = head.clone().into_record();

    if cli.skip_train {
        println!("--skip-train: loading checkpoint {} for re-evaluation only\n", cli.output_checkpoint);
        let record = CompactRecorder::new().load(cli.output_checkpoint.clone().into(), &device)?;
        let loaded: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED).with_boundary_blend(cli.boundary_blend).init(&device).load_record(record);
        best_head_record = loaded.into_record();
    } else {
    for step in 0..N_STEPS {
        let batch: Vec<&Row> = (0..BATCH_SIZE).map(|_| &train_rows[rng.gen_range(0..train_rows.len())]).collect();
        let (new_head, loss) = train_step(head, &mut optimizer, &batch, &device, 1e-3);
        head = new_head;

        if step % VAL_EVERY == 0 || step == N_STEPS - 1 {
            let inner_head = head.clone().valid();
            let inner_device: <I as BackendTypes>::Device = Default::default();
            let mut total_mse = 0f32;
            let mut count = 0usize;
            for r in &val_rows {
                for (mse, _) in eval_row(&inner_head, r, &inner_device) {
                    total_mse += mse;
                    count += 1;
                }
            }
            let val_loss = total_mse / count as f32;
            println!("step {step:5}  train_loss={loss:.6}  val_loss={val_loss:.6}");
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_head_record = head.clone().into_record();
            }
        }
    }
    println!("\nbest val_loss = {best_val_loss:.6}\n");
    }

    let best_head: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED).with_boundary_blend(cli.boundary_blend).init(&device).load_record(best_head_record);
    let best_head_inference = best_head.valid();
    let inner_device: <I as BackendTypes>::Device = Default::default();

    // ---------- Held-out gate, storm-day-restricted (same methodology as the baseline) ----------
    let mut per_gauge_nse: HashMap<Staid, Vec<f32>> = HashMap::new();
    let mut per_gauge_shape_nse: Vec<f32> = Vec::new();
    let mut climatology_shape_nse: Vec<f32> = Vec::new();
    let mut head_peak_err: Vec<usize> = Vec::new();
    let mut clim_peak_err: Vec<usize> = Vec::new();
    let mut head_storm_nse: Vec<f32> = Vec::new();
    let mut clim_storm_nse: Vec<f32> = Vec::new();

    for staid in &split.test {
        let heldout = &test_heldout_by_gauge[staid];
        let train_period = &test_train_period_by_gauge[staid];
        if heldout.is_empty() || train_period.is_empty() { continue }

        let mut clim_shape = [0f32; 24];
        let mut n_clim_days = 0usize;
        for r in train_period {
            for d in 0..WINDOW_DAYS {
                for h in 0..24 {
                    clim_shape[h] += shape_of(&r.daily_q_m3s, &r.target_hourly_m3s, d * 24 + h);
                }
                n_clim_days += 1;
            }
        }
        for v in &mut clim_shape { *v /= n_clim_days as f32; }

        // One stable, per-gauge flat-baseline denominator shared by every
        // row for this gauge (matches the baseline binary's methodology).
        let flat_mse = flat_mse_for_gauge(heldout);

        let mut gauge_nse = Vec::new();
        for r in heldout {
            let day_scores = eval_row(&best_head_inference, r, &inner_device);
            for (d, (head_mse, head_shape)) in day_scores.iter().enumerate() {
                let nse = 1.0 - head_mse / flat_mse.max(1e-12);
                per_gauge_shape_nse.push(nse);
                gauge_nse.push(nse);

                let mut clim_mse = 0f32;
                for h in 0..24 {
                    let target_shape_h = shape_of(&r.daily_q_m3s, &r.target_hourly_m3s, d * 24 + h);
                    clim_mse += (clim_shape[h] - target_shape_h).powi(2);
                }
                clim_mse /= 24.0;
                let clim_nse = 1.0 - clim_mse / flat_mse.max(1e-12);
                climatology_shape_nse.push(clim_nse);

                if r.daily_precip_total_mm[d] > 5.0 {
                    let mut true_target = [0f32; 24];
                    for h in 0..24 { true_target[h] = r.target_hourly_m3s[d * 24 + h]; }
                    let true_peak = argmax(&true_target);
                    head_peak_err.push(circ_dist(argmax(head_shape), true_peak));
                    clim_peak_err.push(circ_dist(argmax(&clim_shape), true_peak));
                    head_storm_nse.push(nse);
                    clim_storm_nse.push(clim_nse);
                }
            }
        }
        per_gauge_nse.insert(staid.clone(), gauge_nse);
    }

    let mut boot_medians: Vec<f32> = Vec::new();
    let mut boot_rng = StdRng::seed_from_u64(SEED ^ 0xB007);
    let test_gauges_with_data: Vec<&Staid> = per_gauge_nse.keys().collect();
    for _ in 0..2000 {
        let mut resample: Vec<f32> = Vec::new();
        for _ in 0..test_gauges_with_data.len() {
            let g = test_gauges_with_data[boot_rng.gen_range(0..test_gauges_with_data.len())];
            resample.extend(per_gauge_nse[g].iter().copied());
        }
        if !resample.is_empty() { boot_medians.push(median(resample)); }
    }
    boot_medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lo = boot_medians[(boot_medians.len() as f32 * 0.025) as usize];
    let ci_hi = boot_medians[(boot_medians.len() as f32 * 0.975) as usize];

    println!("=== Held-out gate (boundary_blend={}) ===", cli.boundary_blend);
    println!("median shape-NSE (head vs flat):   {:.4}  (95% CI [{:.4}, {:.4}])", median(per_gauge_shape_nse.clone()), ci_lo, ci_hi);
    println!("median shape-NSE (climatology):    {:.4}", median(climatology_shape_nse));
    println!("\n--- Genuine storm days only (n={}) ---", head_storm_nse.len());
    println!("median shape-NSE (head):        {:.4}", median(head_storm_nse.clone()));
    println!("median shape-NSE (climatology): {:.4}", median(clim_storm_nse.clone()));
    println!(
        "median peak-hour error: head={:.2}h  climatology={:.2}h",
        median(head_peak_err.iter().map(|&v| v as f32).collect()),
        median(clim_peak_err.iter().map(|&v| v as f32).collect())
    );

    std::fs::create_dir_all("output/disagg_pretrain")?;
    CompactRecorder::new().record(best_head.into_record(), cli.output_checkpoint.clone().into())?;
    println!("\nsaved checkpoint -> {}.mpk", cli.output_checkpoint);

    Ok(())
}
