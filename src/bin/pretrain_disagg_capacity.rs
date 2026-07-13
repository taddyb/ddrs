//! Ablation: does the 72-hour (chunk_days=3) lookback window actually help,
//! independent of the KAN capacity increase (2 KanLayers, grid=20) diagnosed
//! from the original KAN paper review? Uses the REAL PRODUCTION
//! `ddrs::nn::DisaggHead`/`DisaggHeadConfig` directly (chunk_days is now a
//! native config field) -- no separate experimental struct needed.
//!
//! Both experiments draw from the IDENTICAL 9-consecutive-real-day data
//! pool (`extract_complete_day_windows(_, WINDOW_DAYS=9)`); only
//! `--chunk-days` differs (1 = independent per-day softmax, matching the
//! original single-day contract; 3 = joint 72h softmax). Evaluation is
//! always PER-DAY (24h shape-NSE + peak-hour circular error), regardless of
//! how the head internally chunked its softmax, so the two runs are
//! apples-to-apples comparable.
//!
//!   cargo run --release --bin pretrain_disagg_capacity -- --chunk-days 1
//!   cargo run --release --bin pretrain_disagg_capacity -- --chunk-days 3

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
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
    reconcile_gauge,
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

const WINDOW_DAYS: usize = 9; // fixed data pool for BOTH experiments
const WINDOW_HOURS: usize = WINDOW_DAYS * 24;
const N_STEPS: usize = 2000;
const BATCH_SIZE: usize = 64;
const VAL_EVERY: usize = 100;
const MIN_DAILY_Q_M3S: f32 = 0.01;

#[derive(Parser, Debug)]
struct Cli {
    /// 1 = independent per-day softmax (original contract). 3 = joint
    /// 72-hour softmax (the relaxed mass-balance design).
    #[arg(long)]
    chunk_days: usize,
    #[arg(long, default_value_t = 2)]
    num_hidden_layers: usize,
    #[arg(long, default_value_t = 20)]
    grid: usize,
    #[arg(long, default_value_t = 16)]
    hidden_size: usize,
    #[arg(long)]
    output_checkpoint: Option<String>,
    #[arg(long)]
    skip_train: bool,
    /// Dump one row per (gauge, storm-day) with its precip magnitude + head
    /// vs. climatology shape-NSE, so storm-day performance can be
    /// stratified by magnitude instead of collapsed into one median.
    #[arg(long)]
    dump_storm_csv: Option<String>,
}

#[derive(Clone)]
struct Row {
    daily_q: [f32; WINDOW_DAYS],
    precip_feat: [f32; WINDOW_HOURS],
    target_hourly_m3s: [f32; WINDOW_HOURS],
    daily_precip_total_mm: [f32; WINDOW_DAYS],
    /// Calendar date of day 0 of this window -- lets storm-day CSV dumps
    /// record which real date each row is, for later re-plotting.
    window_start_date: NaiveDate,
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
    let qobs_m3s: ndarray::Array1<f32> = qobs.column(0).mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, drain_sqkm) } else { f32::NAN });
    let windows = extract_complete_day_windows(&qobs_m3s, WINDOW_DAYS);
    let windows: Vec<_> = windows.into_iter().filter(|w| w.iter().all(|r| r.daily_q_m3s >= MIN_DAILY_Q_M3S)).collect();

    let precip_col: ndarray::Array1<f32> = precip.column(0).to_owned();
    let precip_norm = normalize_gauge_precip(precip_col.clone());

    windows
        .into_iter()
        .map(|w| {
            let mut daily_q = [0f32; WINDOW_DAYS];
            let mut daily_precip_total_mm = [0f32; WINDOW_DAYS];
            let mut target_hourly_m3s = [0f32; WINDOW_HOURS];
            let start_hour = w[0].day_index * 24;
            let mut precip_feat = [0f32; WINDOW_HOURS];
            for h in 0..WINDOW_HOURS {
                precip_feat[h] = precip_norm[(start_hour + h, 0)];
            }
            for (i, day_row) in w.iter().enumerate() {
                daily_q[i] = day_row.daily_q_m3s;
                target_hourly_m3s[i * 24..i * 24 + 24].copy_from_slice(&day_row.target_hourly_m3s);
                daily_precip_total_mm[i] = (0..24).map(|h| precip_col[day_row.day_index * 24 + h].max(0.0)).sum();
            }
            let window_start_date = start + chrono::Duration::days(w[0].day_index as i64);
            Row { daily_q, precip_feat, target_hourly_m3s, daily_precip_total_mm, window_start_date }
        })
        .collect()
}

fn to_tensor(data: Vec<f32>, shape: [usize; 2], device: &<AutoI as BackendTypes>::Device) -> Tensor<AutoI, 2> {
    Tensor::<AutoI, 2>::from_data(TensorData::new(data, shape), device)
}

fn train_step(
    head: DisaggHead<AutoI>,
    optimizer: &mut impl burn::optim::Optimizer<DisaggHead<AutoI>, AutoI>,
    rows: &[&Row],
    device: &<AutoI as BackendTypes>::Device,
    lr: f64,
) -> (DisaggHead<AutoI>, f32) {
    let n = rows.len();
    let mut q_buf = vec![0f32; WINDOW_DAYS * n];
    let mut precip_buf = Vec::with_capacity(n * WINDOW_HOURS);
    let mut target_shape_buf = Vec::with_capacity(n * WINDOW_HOURS);
    for (col, r) in rows.iter().enumerate() {
        for d in 0..WINDOW_DAYS {
            q_buf[d * n + col] = r.daily_q[d];
        }
        precip_buf.extend_from_slice(&r.precip_feat);
        let window_mean: f32 = r.daily_q.iter().sum::<f32>() / WINDOW_DAYS as f32;
        for h in 0..WINDOW_HOURS {
            target_shape_buf.push(r.target_hourly_m3s[h] / (WINDOW_HOURS as f32 * window_mean));
        }
    }
    let daily_q = to_tensor(q_buf, [WINDOW_DAYS, n], device);
    let precip = to_tensor(precip_buf, [n, WINDOW_HOURS], device).transpose();
    let target_shape = to_tensor(target_shape_buf, [n, WINDOW_HOURS], device).transpose();

    let pred_hourly = head.forward(daily_q.clone(), precip, WINDOW_HOURS); // (WINDOW_HOURS, n)
    let window_mean_t = daily_q.mean_dim(0).reshape([1, n]);
    let pred_shape = pred_hourly / (window_mean_t * (WINDOW_HOURS as f32));
    let loss = (pred_shape - target_shape).powf_scalar(2.0).mean();
    let loss_val: f32 = loss.clone().into_scalar();

    let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &head);
    let head = optimizer.step(lr, head, grads);
    (head, loss_val)
}

/// Forward one window and return the raw (WINDOW_HOURS,) predicted hourly
/// values (physical units, mass-conserving at whatever granularity the
/// head's own `chunk_days` implies).
fn eval_row(head: &DisaggHead<I>, r: &Row, device: &<I as BackendTypes>::Device) -> [f32; WINDOW_HOURS] {
    let daily_q = Tensor::<I, 2>::from_data(TensorData::new(r.daily_q.to_vec(), [WINDOW_DAYS, 1]), device);
    let precip = Tensor::<I, 2>::from_data(TensorData::new(r.precip_feat.to_vec(), [WINDOW_HOURS, 1]), device);
    let pred = head.forward(daily_q, precip, WINDOW_HOURS);
    let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();
    let mut out = [0f32; WINDOW_HOURS];
    out.copy_from_slice(&pred_vec);
    out
}

fn circ_dist(a: usize, b: usize) -> usize {
    let d = (a as i64 - b as i64).unsigned_abs() as usize;
    d.min(24 - d)
}
fn argmax24(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
}
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { return f32::NAN; }
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    println!(
        "=== Capacity/lookback ablation: chunk_days={} num_hidden_layers={} grid={} hidden_size={} ===\n",
        cli.chunk_days, cli.num_hidden_layers, cli.grid, cli.hidden_size
    );

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
        "windows (9-day): train={} val={} test_train_period={} test_heldout={}\n",
        train_rows.len(), val_rows.len(),
        test_train_period_by_gauge.values().map(|v| v.len()).sum::<usize>(),
        test_heldout_by_gauge.values().map(|v| v.len()).sum::<usize>(),
    );

    let device: <AutoI as BackendTypes>::Device = Default::default();
    <AutoI as Backend>::seed(&device, SEED);
    let mut head: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED)
        .with_hidden_size(cli.hidden_size)
        .with_num_hidden_layers(cli.num_hidden_layers)
        .with_grid(cli.grid)
        .with_chunk_days(cli.chunk_days)
        .init(&device);
    let mut optimizer = build_adam::<DisaggHead<AutoI>, AutoI>();
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xC0FFEE);

    let ckpt_path = cli.output_checkpoint.clone().unwrap_or_else(|| {
        format!("output/disagg_pretrain/capacity_chunk{}", cli.chunk_days)
    });

    let mut best_val_loss = f32::INFINITY;
    let mut best_head_record = head.clone().into_record();

    if cli.skip_train {
        println!("--skip-train: loading checkpoint {ckpt_path}\n");
        let record = CompactRecorder::new().load(ckpt_path.clone().into(), &device)?;
        let loaded: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED)
            .with_hidden_size(cli.hidden_size)
            .with_num_hidden_layers(cli.num_hidden_layers)
            .with_grid(cli.grid)
            .with_chunk_days(cli.chunk_days)
            .init(&device)
            .load_record(record);
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
                for r in &val_rows {
                    let pred = eval_row(&inner_head, r, &inner_device);
                    let window_mean: f32 = r.daily_q.iter().sum::<f32>() / WINDOW_DAYS as f32;
                    let mut mse = 0f32;
                    for h in 0..WINDOW_HOURS {
                        let pred_shape = pred[h] / (WINDOW_HOURS as f32 * window_mean);
                        let target_shape = r.target_hourly_m3s[h] / (WINDOW_HOURS as f32 * window_mean);
                        mse += (pred_shape - target_shape).powi(2);
                    }
                    total_mse += mse / WINDOW_HOURS as f32;
                }
                let val_loss = total_mse / val_rows.len() as f32;
                println!("step {step:5}  train_loss={loss:.6}  val_loss={val_loss:.6}");
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    best_head_record = head.clone().into_record();
                }
            }
        }
        println!("\nbest val_loss = {best_val_loss:.6}\n");
    }

    let best_head: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED)
        .with_hidden_size(cli.hidden_size)
        .with_num_hidden_layers(cli.num_hidden_layers)
        .with_grid(cli.grid)
        .with_chunk_days(cli.chunk_days)
        .init(&device)
        .load_record(best_head_record);
    let best_head_inference = best_head.valid();
    let inner_device: <I as BackendTypes>::Device = Default::default();

    // ---------- Held-out gate: ALWAYS evaluated PER-DAY (24h slices) so
    // chunk_days=1 and chunk_days=3 runs are directly comparable. ----------
    let mut per_gauge_nse: HashMap<Staid, Vec<f32>> = HashMap::new();
    let mut per_day_shape_nse: Vec<f32> = Vec::new();
    let mut climatology_shape_nse: Vec<f32> = Vec::new();
    let mut head_peak_err: Vec<usize> = Vec::new();
    let mut clim_peak_err: Vec<usize> = Vec::new();
    let mut head_storm_nse: Vec<f32> = Vec::new();
    let mut clim_storm_nse: Vec<f32> = Vec::new();
    // (staid, date, precip_mm, daily_q, head_nse, clim_nse, head_peak_hour_err,
    // clim_peak_hour_err, head_peak_mag_relerr, clim_peak_mag_relerr) -- for
    // stratifying by storm magnitude AND re-plotting representative days.
    #[allow(clippy::type_complexity)]
    let mut storm_detail_rows: Vec<(String, NaiveDate, f32, f32, f32, f32, usize, usize, f32, f32)> = Vec::new();

    for staid in &split.test {
        let heldout = &test_heldout_by_gauge[staid];
        let train_period = &test_train_period_by_gauge[staid];
        if heldout.is_empty() || train_period.is_empty() { continue }

        // Per-gauge 24h climatology template from its OWN train-period days.
        let mut clim_shape_24 = [0f32; 24];
        let mut n_clim_days = 0usize;
        for r in train_period {
            for d in 0..WINDOW_DAYS {
                for h in 0..24 {
                    clim_shape_24[h] += r.target_hourly_m3s[d * 24 + h] / (24.0 * r.daily_q[d]);
                }
                n_clim_days += 1;
            }
        }
        for v in &mut clim_shape_24 { *v /= n_clim_days as f32; }

        // Stable per-gauge flat-shape MSE denominator (over ALL days in
        // the heldout period, not per-window).
        let mut flat_mse_total = 0f32;
        let mut flat_mse_n = 0usize;
        for r in heldout {
            for d in 0..WINDOW_DAYS {
                for h in 0..24 {
                    let target_shape_h = r.target_hourly_m3s[d * 24 + h] / (24.0 * r.daily_q[d]);
                    flat_mse_total += (1.0 / 24.0 - target_shape_h).powi(2);
                    flat_mse_n += 1;
                }
            }
        }
        let flat_mse = flat_mse_total / flat_mse_n as f32;

        let mut gauge_nse = Vec::new();
        for r in heldout {
            let pred = eval_row(&best_head_inference, r, &inner_device);
            for d in 0..WINDOW_DAYS {
                let day_q = r.daily_q[d];
                let pred_day: Vec<f32> = (0..24).map(|h| pred[d * 24 + h] / (24.0 * day_q)).collect();
                let target_day: Vec<f32> = (0..24).map(|h| r.target_hourly_m3s[d * 24 + h] / (24.0 * day_q)).collect();

                let mut head_mse = 0f32;
                for h in 0..24 {
                    head_mse += (pred_day[h] - target_day[h]).powi(2);
                }
                head_mse /= 24.0;
                let nse = 1.0 - head_mse / flat_mse.max(1e-12);
                per_day_shape_nse.push(nse);
                gauge_nse.push(nse);

                let mut clim_mse = 0f32;
                for h in 0..24 {
                    clim_mse += (clim_shape_24[h] - target_day[h]).powi(2);
                }
                clim_mse /= 24.0;
                let clim_nse = 1.0 - clim_mse / flat_mse.max(1e-12);
                climatology_shape_nse.push(clim_nse);

                if r.daily_precip_total_mm[d] > 5.0 {
                    let true_peak = argmax24(&target_day);
                    let head_peak_hour_err = circ_dist(argmax24(&pred_day), true_peak);
                    let clim_peak_hour_err = circ_dist(argmax24(&clim_shape_24), true_peak);
                    head_peak_err.push(head_peak_hour_err);
                    clim_peak_err.push(clim_peak_hour_err);
                    head_storm_nse.push(nse);
                    clim_storm_nse.push(clim_nse);

                    // Peak MAGNITUDE error: physical hourly values at the
                    // observed peak hour, relative error vs the real peak.
                    let real_peak_val = target_day[true_peak].max(1e-9);
                    let head_peak_mag_relerr = (pred_day[true_peak] - real_peak_val) / real_peak_val;
                    let clim_peak_mag_relerr = (clim_shape_24[true_peak] - real_peak_val) / real_peak_val;

                    let date = r.window_start_date + chrono::Duration::days(d as i64);
                    storm_detail_rows.push((
                        staid.to_string(), date, r.daily_precip_total_mm[d], day_q, nse, clim_nse,
                        head_peak_hour_err, clim_peak_hour_err, head_peak_mag_relerr, clim_peak_mag_relerr,
                    ));
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

    println!("=== Held-out gate (chunk_days={}, layers={}, grid={}) — ALL metrics per-DAY (24h) ===", cli.chunk_days, cli.num_hidden_layers, cli.grid);
    println!("median shape-NSE (head vs flat):   {:.4}  (95% CI [{:.4}, {:.4}])", median(per_day_shape_nse.clone()), ci_lo, ci_hi);
    println!("median shape-NSE (climatology):    {:.4}", median(climatology_shape_nse));
    println!("\n--- Genuine storm days only (n={}) ---", head_storm_nse.len());
    println!("median shape-NSE (head):        {:.4}", median(head_storm_nse));
    println!("median shape-NSE (climatology): {:.4}", median(clim_storm_nse));
    println!(
        "median peak-hour error (0-23h): head={:.2}h  climatology={:.2}h",
        median(head_peak_err.iter().map(|&v| v as f32).collect()),
        median(clim_peak_err.iter().map(|&v| v as f32).collect())
    );
    println!(
        "median |peak-magnitude relerr|: head={:.3}  climatology={:.3}",
        median(storm_detail_rows.iter().map(|r| r.8.abs()).collect()),
        median(storm_detail_rows.iter().map(|r| r.9.abs()).collect())
    );

    std::fs::create_dir_all("output/disagg_pretrain")?;
    CompactRecorder::new().record(best_head.into_record(), ckpt_path.clone().into())?;
    println!("\nsaved checkpoint -> {ckpt_path}.mpk");

    if let Some(csv_path) = &cli.dump_storm_csv {
        let mut f = std::fs::File::create(csv_path)?;
        use std::io::Write;
        writeln!(f, "staid,date,precip_mm,daily_q,head_nse,clim_nse,head_peak_hour_err,clim_peak_hour_err,head_peak_mag_relerr,clim_peak_mag_relerr")?;
        for (staid, date, precip_mm, daily_q, head_nse, clim_nse, head_peak_hour_err, clim_peak_hour_err, head_peak_mag_relerr, clim_peak_mag_relerr) in &storm_detail_rows {
            writeln!(f, "{staid},{date},{precip_mm},{daily_q},{head_nse},{clim_nse},{head_peak_hour_err},{clim_peak_hour_err},{head_peak_mag_relerr},{clim_peak_mag_relerr}")?;
        }
        println!("wrote {csv_path} ({} storm-day rows)", storm_detail_rows.len());
    }

    Ok(())
}
