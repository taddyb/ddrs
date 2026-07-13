//! Phase 3 follow-up experiment #2: relax mass conservation from "each
//! calendar day exactly" to "the trailing 72-hour (3-day) window exactly."
//! Unlike `pretrain_disagg_blend` (which kept 3 INDEPENDENT per-day
//! softmaxes and only smoothed the seam post-hoc via `boundary_blend`),
//! this experiment uses ONE joint softmax over all 72 hours, so the network
//! can freely place mass anywhere across day boundaries within the window
//! -- the only guarantee is that the 72-hour MEAN equals the 3-day MEAN
//! `daily_q`, not that each individual day's own sub-mean is exact.
//!
//! This is a genuinely different architecture from production's
//! `DisaggHead` (which this binary does NOT touch -- fully standalone,
//! exploratory, per instructions: validate here first, only port into
//! `src/nn/disagg_head.rs` + full ddrs retrain if the storm-comparison
//! plots actually show improved precip-responsiveness).
//!
//! Input: 3 days' log(daily_q) (3 features) + the window's 72h precip (72
//! features) = 75 features -> Linear(75,H) -> KanLayer(H,H) -> Linear(H,72)
//! -> ONE softmax over 72 -> hourly = mean(daily_q_3)*72*shape.
//!
//! Mass-balance invariant (explicitly tested, not just structural): for
//! every window, mean(predicted 72h) == mean(daily_q_3) -- holds by
//! construction of the softmax formula; NOT claimed per-individual-day.
//!
//!   cargo run --release --bin pretrain_disagg_window72

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::config::Config;
use burn::module::{AutodiffModule, Module};
use burn::nn::Linear;
use burn::optim::{GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::softmax;
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};
use chrono::NaiveDate;
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rskan::{KanLayer, KanLayerConfig};

use ddrs::data::ids::Staid;
use ddrs::data::store::gage_csv::{GageMetadata, GageRow};
use ddrs::data::store::icechunk::UsgsObservationsStore;
use ddrs::data::store::CamelsHourlyStore;
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

const N_STEPS: usize = 2000;
const BATCH_SIZE: usize = 64;
const VAL_EVERY: usize = 100;
const MIN_DAILY_Q_M3S: f32 = 0.01;
const WINDOW_DAYS: usize = 3;
const WINDOW_HOURS: usize = WINDOW_DAYS * 24; // 72
const NUM_FEATURES: usize = WINDOW_DAYS + WINDOW_HOURS; // 3 + 72 = 75
const LOG_EPS: f32 = 1.0e-3;
const HIDDEN: usize = 16;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "output/disagg_pretrain/best_disagg_window72")]
    output_checkpoint: String,
    #[arg(long)]
    skip_train: bool,
}

// ---------------------------------------------------------------------------
// Standalone experimental head (NOT src/nn/disagg_head.rs -- exploratory)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Window72Head<B: Backend> {
    input: Linear<B>,
    hidden: Vec<KanLayer<B>>,
    output: Linear<B>,
}

#[derive(Config, Debug)]
struct Window72HeadConfig {
    seed: u64,
    #[config(default = 16)]
    hidden_size: usize,
    #[config(default = 1)]
    num_hidden_layers: usize,
    #[config(default = 3)]
    grid: usize,
    #[config(default = 3)]
    k: usize,
}

impl Window72HeadConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Window72Head<B> {
        let h = self.hidden_size;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let input_weight = ddrs::nn::init::sample_kaiming_normal_relu(&mut rng, NUM_FEATURES, h);
        let output_weight = ddrs::nn::init::sample_xavier_normal(&mut rng, h, WINDOW_HOURS, 1.0);
        let input = Linear {
            weight: ddrs::nn::init::to_param_weight::<B>(input_weight, device),
            bias: Some(ddrs::nn::init::zero_bias_tensor::<B>(h, device)),
        };
        let output = Linear {
            weight: ddrs::nn::init::to_param_weight::<B>(output_weight, device),
            bias: Some(ddrs::nn::init::zero_bias_tensor::<B>(WINDOW_HOURS, device)),
        };
        let hidden: Vec<KanLayer<B>> = (0..self.num_hidden_layers)
            .map(|_| KanLayerConfig::new(h, h, self.seed).with_num(self.grid).with_k(self.k).init(device))
            .collect();
        Window72Head { input, hidden, output }
    }
}

impl<B: Backend> Window72Head<B> {
    /// `daily_q_3`: (3, N) -- 3 days' daily Q' (m³/s). `precip_72`: (72, N)
    /// -- the SAME window's 72h normalized precip. Returns (72, N) hourly.
    /// Mass balance: mean over the 72 output hours == mean over the 3
    /// `daily_q_3` values (the aggregate 3-day mean), NOT each individual
    /// day's own mean -- this is the deliberate relaxation being tested.
    fn forward(&self, daily_q_3: Tensor<B, 2>, precip_72: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_, n] = daily_q_3.dims();
        let logq = daily_q_3.clone().add_scalar(LOG_EPS).log().transpose(); // (N, 3)
        let precip_t = precip_72.transpose(); // (N, 72)
        let feats = Tensor::cat(vec![logq, precip_t], 1); // (N, 75)

        let mut x = self.input.forward(feats);
        for layer in &self.hidden {
            x = layer.forward(x);
        }
        let logits = self.output.forward(x); // (N, 72)
        let shape = softmax(logits, 1); // (N, 72), sums to 1 per row

        let three_day_mean = daily_q_3.mean_dim(0).reshape([n, 1]); // (N,1) mean over the 3 days
        let hourly = shape * three_day_mean * (WINDOW_HOURS as f32); // (N, 72)
        hourly.transpose() // (72, N)
    }
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Row {
    daily_q_3: [f32; WINDOW_DAYS],
    precip_feat: [f32; WINDOW_HOURS],
    target_hourly_m3s: [f32; WINDOW_HOURS],
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
    let qobs_m3s: ndarray::Array1<f32> = qobs.column(0).mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, drain_sqkm) } else { f32::NAN });
    let windows = extract_complete_day_windows(&qobs_m3s, WINDOW_DAYS);
    let windows: Vec<_> = windows.into_iter().filter(|w| w.iter().all(|r| r.daily_q_m3s >= MIN_DAILY_Q_M3S)).collect();

    let precip_col: ndarray::Array1<f32> = precip.column(0).to_owned();
    let precip_norm = normalize_gauge_precip(precip_col.clone());

    windows
        .into_iter()
        .map(|w| {
            let mut daily_q_3 = [0f32; WINDOW_DAYS];
            let mut daily_precip_total_mm = [0f32; WINDOW_DAYS];
            let mut target_hourly_m3s = [0f32; WINDOW_HOURS];
            let start_hour = w[0].day_index * 24;
            let mut precip_feat = [0f32; WINDOW_HOURS];
            for h in 0..WINDOW_HOURS {
                precip_feat[h] = precip_norm[(start_hour + h, 0)];
            }
            for (i, day_row) in w.iter().enumerate() {
                daily_q_3[i] = day_row.daily_q_m3s;
                target_hourly_m3s[i * 24..i * 24 + 24].copy_from_slice(&day_row.target_hourly_m3s);
                daily_precip_total_mm[i] = (0..24).map(|h| precip_col[day_row.day_index * 24 + h].max(0.0)).sum();
            }
            // Explicit mass-balance invariant: the 72h mean must equal the
            // 3-day mean daily_q, by construction of extract_complete_days.
            let mean72: f32 = target_hourly_m3s.iter().sum::<f32>() / WINDOW_HOURS as f32;
            let mean3: f32 = daily_q_3.iter().sum::<f32>() / WINDOW_DAYS as f32;
            assert!((mean72 - mean3).abs() < 1e-3, "72h mass-balance invariant violated: {mean72} != {mean3}");

            Row { daily_q_3, precip_feat, target_hourly_m3s, daily_precip_total_mm }
        })
        .collect()
}

fn to_tensor(data: Vec<f32>, shape: [usize; 2], device: &<AutoI as BackendTypes>::Device) -> Tensor<AutoI, 2> {
    Tensor::<AutoI, 2>::from_data(TensorData::new(data, shape), device)
}

fn train_step(
    head: Window72Head<AutoI>,
    optimizer: &mut impl Optimizer<Window72Head<AutoI>, AutoI>,
    rows: &[&Row],
    device: &<AutoI as BackendTypes>::Device,
    lr: f64,
) -> (Window72Head<AutoI>, f32) {
    let n = rows.len();
    let mut q_buf = vec![0f32; WINDOW_DAYS * n];
    let mut precip_buf = Vec::with_capacity(n * WINDOW_HOURS);
    let mut target_shape_buf = Vec::with_capacity(n * WINDOW_HOURS);
    for (col, r) in rows.iter().enumerate() {
        for d in 0..WINDOW_DAYS {
            q_buf[d * n + col] = r.daily_q_3[d];
        }
        precip_buf.extend_from_slice(&r.precip_feat);
        let mean3: f32 = r.daily_q_3.iter().sum::<f32>() / WINDOW_DAYS as f32;
        for h in 0..WINDOW_HOURS {
            target_shape_buf.push(r.target_hourly_m3s[h] / (WINDOW_HOURS as f32 * mean3));
        }
    }
    let daily_q_3 = to_tensor(q_buf, [WINDOW_DAYS, n], device);
    let precip = to_tensor(precip_buf, [n, WINDOW_HOURS], device).transpose();
    let target_shape = to_tensor(target_shape_buf, [n, WINDOW_HOURS], device).transpose();

    let pred_hourly = head.forward(daily_q_3.clone(), precip); // (72, n)
    let mean3 = daily_q_3.mean_dim(0).reshape([1, n]); // (1,n)
    let pred_shape = pred_hourly / (mean3 * (WINDOW_HOURS as f32));
    let loss = (pred_shape - target_shape).powf_scalar(2.0).mean();
    let loss_val: f32 = loss.clone().into_scalar();

    let grads = GradientsParams::from_grads(loss.backward(), &head);
    let head = optimizer.step(lr, head, grads);
    (head, loss_val)
}

fn eval_row(head: &Window72Head<I>, r: &Row, device: &<I as BackendTypes>::Device) -> (f32, [f32; WINDOW_HOURS]) {
    let daily_q_3 = Tensor::<I, 2>::from_data(TensorData::new(r.daily_q_3.to_vec(), [WINDOW_DAYS, 1]), device);
    let precip = Tensor::<I, 2>::from_data(TensorData::new(r.precip_feat.to_vec(), [WINDOW_HOURS, 1]), device);
    let pred = head.forward(daily_q_3, precip);
    let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();
    let mean3: f32 = r.daily_q_3.iter().sum::<f32>() / WINDOW_DAYS as f32;
    let mut shape = [0f32; WINDOW_HOURS];
    let mut mse = 0f32;
    for h in 0..WINDOW_HOURS {
        shape[h] = pred_vec[h] / (WINDOW_HOURS as f32 * mean3);
        let target_shape_h = r.target_hourly_m3s[h] / (WINDOW_HOURS as f32 * mean3);
        mse += (shape[h] - target_shape_h).powi(2);
    }
    (mse / WINDOW_HOURS as f32, shape)
}

fn flat_mse_for_gauge(rows: &[Row]) -> f32 {
    let mut total = 0f32;
    let mut n = 0usize;
    for r in rows {
        let mean3: f32 = r.daily_q_3.iter().sum::<f32>() / WINDOW_DAYS as f32;
        for h in 0..WINDOW_HOURS {
            let target_shape_h = r.target_hourly_m3s[h] / (WINDOW_HOURS as f32 * mean3);
            total += (1.0 / WINDOW_HOURS as f32 - target_shape_h).powi(2);
            n += 1;
        }
    }
    total / n as f32
}

fn circ_dist72(a: usize, b: usize) -> usize {
    let d = (a as i64 - b as i64).unsigned_abs() as usize;
    d.min(WINDOW_HOURS - d)
}
fn argmax72(v: &[f32; WINDOW_HOURS]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
}
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { return f32::NAN; }
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    println!("=== Phase 3 follow-up #2: 72-hour joint mass-balance window ===\n");

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
    let mut head: Window72Head<AutoI> = Window72HeadConfig::new(SEED).with_hidden_size(HIDDEN).init(&device);
    let mut optimizer = build_adam::<Window72Head<AutoI>, AutoI>();
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xC0FFEE);

    let mut best_val_loss = f32::INFINITY;
    let mut best_head_record = head.clone().into_record();

    if cli.skip_train {
        println!("--skip-train: loading checkpoint {}\n", cli.output_checkpoint);
        let record = CompactRecorder::new().load(cli.output_checkpoint.clone().into(), &device)?;
        let loaded: Window72Head<AutoI> = Window72HeadConfig::new(SEED).with_hidden_size(HIDDEN).init(&device).load_record(record);
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
                    let (mse, _) = eval_row(&inner_head, r, &inner_device);
                    total_mse += mse;
                    count += 1;
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

    let best_head: Window72Head<AutoI> = Window72HeadConfig::new(SEED).with_hidden_size(HIDDEN).init(&device).load_record(best_head_record);
    let best_head_inference = best_head.valid();
    let inner_device: <I as BackendTypes>::Device = Default::default();

    // ---------- Held-out gate ----------
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

        // Per-gauge 24h climatology template (same mechanism as before),
        // applied independently to EACH of the 3 days in a window so the
        // comparison basis matches what the head is being asked to beat.
        let mut clim_shape_24 = [0f32; 24];
        let mut n_clim_days = 0usize;
        for r in train_period {
            for d in 0..WINDOW_DAYS {
                for h in 0..24 {
                    clim_shape_24[h] += r.target_hourly_m3s[d * 24 + h] / (24.0 * r.daily_q_3[d]);
                }
                n_clim_days += 1;
            }
        }
        for v in &mut clim_shape_24 { *v /= n_clim_days as f32; }

        let flat_mse = flat_mse_for_gauge(heldout);
        let mut gauge_nse = Vec::new();

        for r in heldout {
            let mean3: f32 = r.daily_q_3.iter().sum::<f32>() / WINDOW_DAYS as f32;
            let (head_mse, head_shape) = eval_row(&best_head_inference, r, &inner_device);
            let nse = 1.0 - head_mse / flat_mse.max(1e-12);
            per_gauge_shape_nse.push(nse);
            gauge_nse.push(nse);

            // Climatology's 72h shape: apply the 24h template per-day using
            // EACH day's own daily_q (independent per-day, as before), then
            // convert to the shared 72h-mean-denominator shape space.
            let mut clim_shape_72 = [0f32; WINDOW_HOURS];
            for d in 0..WINDOW_DAYS {
                for h in 0..24 {
                    let hourly_val = clim_shape_24[h] * 24.0 * r.daily_q_3[d];
                    clim_shape_72[d * 24 + h] = hourly_val / (WINDOW_HOURS as f32 * mean3);
                }
            }
            let mut clim_mse = 0f32;
            for h in 0..WINDOW_HOURS {
                let target_shape_h = r.target_hourly_m3s[h] / (WINDOW_HOURS as f32 * mean3);
                clim_mse += (clim_shape_72[h] - target_shape_h).powi(2);
            }
            clim_mse /= WINDOW_HOURS as f32;
            let clim_nse = 1.0 - clim_mse / flat_mse.max(1e-12);
            climatology_shape_nse.push(clim_nse);

            if r.daily_precip_total_mm.iter().any(|&p| p > 5.0) {
                let true_peak = argmax72(&r.target_hourly_m3s);
                head_peak_err.push(circ_dist72(argmax72(&head_shape), true_peak));
                clim_peak_err.push(circ_dist72(argmax72(&clim_shape_72), true_peak));
                head_storm_nse.push(nse);
                clim_storm_nse.push(clim_nse);
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

    println!("=== Held-out gate (72h joint mass balance) ===");
    println!("median shape-NSE (head vs flat):   {:.4}  (95% CI [{:.4}, {:.4}])", median(per_gauge_shape_nse.clone()), ci_lo, ci_hi);
    println!("median shape-NSE (climatology):    {:.4}", median(climatology_shape_nse));
    println!("\n--- Genuine storm days only (n={}) ---", head_storm_nse.len());
    println!("median shape-NSE (head):        {:.4}", median(head_storm_nse.clone()));
    println!("median shape-NSE (climatology): {:.4}", median(clim_storm_nse.clone()));
    println!(
        "median peak-hour error (0-71h scale): head={:.2}h  climatology={:.2}h",
        median(head_peak_err.iter().map(|&v| v as f32).collect()),
        median(clim_peak_err.iter().map(|&v| v as f32).collect())
    );

    std::fs::create_dir_all("output/disagg_pretrain")?;
    CompactRecorder::new().record(best_head.into_record(), cli.output_checkpoint.clone().into())?;
    println!("\nsaved checkpoint -> {}.mpk", cli.output_checkpoint);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mass-balance invariant for `Window72Head` itself (fresh init, no
    /// training, several random daily_q/precip combinations): the PREDICTED
    /// 72-hour mean must equal the 3-day input mean exactly (to f32
    /// tolerance), by construction of the shared softmax formula. This is
    /// the "72-hour window" version of the same invariant tested for the
    /// production `DisaggHead` -- explicitly checked here BEFORE any real
    /// training, per the requirement that mass balance is always verified,
    /// not just assumed from the architecture.
    #[test]
    fn window72_head_conserves_mass_at_fresh_init() {
        let device: <I as BackendTypes>::Device = Default::default();
        let head: Window72Head<I> = Window72HeadConfig::new(1).with_hidden_size(HIDDEN).init(&device);

        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..20 {
            let daily_q: Vec<f32> = (0..WINDOW_DAYS).map(|_| rng.gen_range(0.1f32..500.0)).collect();
            let precip: Vec<f32> = (0..WINDOW_HOURS).map(|_| rng.gen_range(-2.0f32..3.0)).collect();
            let daily_q_t = Tensor::<I, 2>::from_data(TensorData::new(daily_q.clone(), [WINDOW_DAYS, 1]), &device);
            let precip_t = Tensor::<I, 2>::from_data(TensorData::new(precip, [WINDOW_HOURS, 1]), &device);
            let pred = head.forward(daily_q_t, precip_t);
            let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();

            let pred_mean: f32 = pred_vec.iter().sum::<f32>() / WINDOW_HOURS as f32;
            let input_mean: f32 = daily_q.iter().sum::<f32>() / WINDOW_DAYS as f32;
            assert!(
                (pred_mean - input_mean).abs() < 1e-3 * input_mean.max(1.0),
                "mass-balance invariant violated: pred_mean={pred_mean} != input_mean={input_mean}"
            );
        }
    }

    /// Confirms the invariant is genuinely a 72-hour AGGREGATE guarantee,
    /// NOT a per-individual-day one: construct daily_q values that differ
    /// a lot day-to-day and verify the model is free to allocate mass
    /// unevenly across the 3 days (i.e. this is NOT accidentally still
    /// enforcing the old per-day-exact constraint).
    #[test]
    fn window72_head_does_not_enforce_per_day_exactness() {
        let device: <I as BackendTypes>::Device = Default::default();
        // Deliberately non-flat output layer so the shape isn't uniform.
        let mut head: Window72Head<I> = Window72HeadConfig::new(1).with_hidden_size(HIDDEN).init(&device);
        let biased_output = ndarray::Array2::<f32>::from_shape_fn((HIDDEN, WINDOW_HOURS), |(i, j)| {
            if i % WINDOW_HOURS == j { 3.0 } else { 0.05 * ((i + j) as f32).sin() }
        });
        head.output.weight = ddrs::nn::init::to_param_weight::<I>(biased_output, &device);

        let daily_q_t = Tensor::<I, 2>::from_data(TensorData::new(vec![1.0f32, 50.0, 1.0], [WINDOW_DAYS, 1]), &device);
        let precip_t = Tensor::<I, 2>::from_data(TensorData::new(vec![0.1f32; WINDOW_HOURS], [WINDOW_HOURS, 1]), &device);
        let pred = head.forward(daily_q_t, precip_t);
        let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();

        // Per-day sub-means should NOT all equal their own day's daily_q
        // (that would mean the "relaxation" silently isn't happening).
        let day_means: Vec<f32> = (0..WINDOW_DAYS)
            .map(|d| pred_vec[d * 24..d * 24 + 24].iter().sum::<f32>() / 24.0)
            .collect();
        let per_day_exact = day_means.iter().zip([1.0f32, 50.0, 1.0]).all(|(m, q)| (m - q).abs() < 1e-2);
        assert!(!per_day_exact, "per-day means {day_means:?} suspiciously match [1,50,1] exactly -- relaxation may not be active");

        // But the 72h aggregate mean must still be exact.
        let mean72: f32 = pred_vec.iter().sum::<f32>() / WINDOW_HOURS as f32;
        let mean3 = (1.0 + 50.0 + 1.0) / 3.0;
        assert!((mean72 - mean3).abs() < 1e-3 * mean3, "72h aggregate mean {mean72} != {mean3}");
    }
}
