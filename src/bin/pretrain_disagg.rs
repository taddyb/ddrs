//! Phase 3: standalone pretraining of `DisaggHead` on REAL USGS hourly
//! observations (no LSTM product, no routing pipeline, no daily-aggregated
//! loss) + a pre-registered GO/NO-GO gate on held-out gauges x held-out
//! years. See `src/pretrain/mod.rs` module docs and the campaign plan.
//!
//! Data: real USGS daily obs -> real drainage-area unit conversion -> real
//! hourly USGS obs (`camels_hourly`) as the disaggregation TARGET, real
//! NLDAS hourly precip (bundled in the same file) as the conditioning
//! input. Zero second-model bias anywhere in this pipeline.
//!
//!   cargo run --release --bin pretrain_disagg

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

#[derive(Parser, Debug)]
struct Cli {
    /// Skip training entirely and re-evaluate an already-saved checkpoint
    /// (e.g. after fixing an evaluation-only bug in the gate criteria --
    /// no need to pay for another 2000-step training run).
    #[arg(long)]
    skip_train: bool,
    #[arg(long, default_value = "output/disagg_pretrain/best_disagg")]
    checkpoint: String,
}

use ddrs::data::ids::Staid;
use ddrs::data::store::gage_csv::{GageMetadata, GageRow};
use ddrs::data::store::icechunk::UsgsObservationsStore;
use ddrs::data::store::CamelsHourlyStore;
use ddrs::nn::{DisaggHead, DisaggHeadConfig};
use ddrs::pretrain::{
    assert_mass_balance, build_split_manifest, extract_complete_days, normalize_gauge_precip,
    qobs_mm_hr_to_m3s, reconcile_gauge, slice_precip_features, PretrainRow,
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
const MASS_BALANCE_CHECK_EVERY: usize = 200;
const MIN_DAILY_Q_M3S: f32 = 0.01; // guard near-zero-flow days (degenerate shape target)

/// One (gauge, day) sample ready for the head: daily_q feature + 24h
/// normalized precip feature + real 24h target (m³/s).
#[derive(Clone)]
struct Row {
    daily_q_m3s: f32,
    precip_feat: [f32; 24],
    target_hourly_m3s: [f32; 24],
    /// Raw (un-normalized) total precip for the day, mm -- used ONLY to
    /// define "genuine storm day" for the precip-active gate criterion
    /// (the z-scored `precip_feat` is nonzero on almost every day, which
    /// makes it useless for that purpose -- see the gate report).
    daily_precip_total_mm: f32,
}

fn ymd(t: (i32, u32, u32)) -> NaiveDate {
    NaiveDate::from_ymd_opt(t.0, t.1, t.2).unwrap()
}

/// Read + convert + normalize + mask one gauge's rows for `[start, end]`.
fn gauge_rows(
    camels: &CamelsHourlyStore,
    staid: &Staid,
    drain_sqkm: f64,
    start: NaiveDate,
    end: NaiveDate,
) -> Vec<Row> {
    let start_dt = start.and_hms_opt(0, 0, 0).unwrap();
    let n_days = (end - start).num_days() as usize + 1;
    let n_hours = n_days * 24;
    let Ok((qobs, precip)) = camels.read_window(start_dt, n_hours, std::slice::from_ref(staid)) else {
        return Vec::new();
    };

    let qobs_m3s: ndarray::Array1<f32> = qobs
        .column(0)
        .mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, drain_sqkm) } else { f32::NAN });
    let complete: Vec<PretrainRow> = extract_complete_days(&qobs_m3s)
        .into_iter()
        .filter(|r| r.daily_q_m3s >= MIN_DAILY_Q_M3S)
        .collect();
    assert_mass_balance(&complete, 1e-3);

    let precip_col: ndarray::Array1<f32> = precip.column(0).to_owned();
    let precip_norm = normalize_gauge_precip(precip_col.clone());
    let day_indices: Vec<usize> = complete.iter().map(|r| r.day_index).collect();
    let precip_feats = slice_precip_features(&precip_norm, &day_indices);
    let daily_precip_totals: Vec<f32> = day_indices
        .iter()
        .map(|&d| (0..24).map(|h| precip_col[d * 24 + h].max(0.0)).sum())
        .collect();

    complete
        .into_iter()
        .zip(precip_feats)
        .zip(daily_precip_totals)
        .map(|((r, feat), total_mm)| Row {
            daily_q_m3s: r.daily_q_m3s,
            precip_feat: feat,
            target_hourly_m3s: r.target_hourly_m3s,
            daily_precip_total_mm: total_mm,
        })
        .collect()
}

fn to_tensor_2d(data: Vec<f32>, shape: [usize; 2], device: &<AutoI as BackendTypes>::Device) -> Tensor<AutoI, 2> {
    Tensor::<AutoI, 2>::from_data(TensorData::new(data, shape), device)
}

/// One training step: build (daily_q, precip, target) tensors for a
/// minibatch of `rows`, forward, shape-space MSE, backward, Adam step.
/// Returns the scalar loss.
#[allow(clippy::too_many_arguments)]
fn train_step(
    head: DisaggHead<AutoI>,
    optimizer: &mut impl Optimizer<DisaggHead<AutoI>, AutoI>,
    rows: &[&Row],
    device: &<AutoI as BackendTypes>::Device,
    lr: f64,
) -> (DisaggHead<AutoI>, f32) {
    let n = rows.len();
    let mut q_buf = Vec::with_capacity(n);
    let mut precip_buf = Vec::with_capacity(n * 24);
    let mut target_shape_buf = Vec::with_capacity(n * 24);
    for r in rows {
        q_buf.push(r.daily_q_m3s);
        for h in 0..24 {
            precip_buf.push(r.precip_feat[h]);
        }
        for h in 0..24 {
            target_shape_buf.push(r.target_hourly_m3s[h] / (24.0 * r.daily_q_m3s));
        }
    }
    let daily_q = to_tensor_2d(q_buf, [1, n], device);
    let precip = to_tensor_2d(precip_buf, [n, 24], device).transpose(); // (24, n)
    let target_shape = to_tensor_2d(target_shape_buf, [n, 24], device).transpose(); // (24, n)

    let pred_hourly = head.forward(daily_q.clone(), precip, 24); // (24, n)
    let daily_q_bcast = daily_q.reshape([1, n]);
    let pred_shape = pred_hourly / (daily_q_bcast * 24.0);
    let diff = pred_shape - target_shape;
    let loss = diff.powf_scalar(2.0).mean();
    let loss_val: f32 = loss.clone().into_scalar();

    let grads = GradientsParams::from_grads(loss.backward(), &head);
    let head = optimizer.step(lr, head, grads);
    (head, loss_val)
}

/// Forward `rows` through `head` (inference backend, no grad) and return
/// per-row shape MSE + predicted 24h shape.
fn eval_rows(head: &DisaggHead<I>, rows: &[Row], device: &<I as BackendTypes>::Device) -> Vec<(f32, [f32; 24])> {
    rows.iter()
        .map(|r| {
            let daily_q = Tensor::<I, 2>::from_data(TensorData::new(vec![r.daily_q_m3s], [1, 1]), device);
            let precip = Tensor::<I, 2>::from_data(TensorData::new(r.precip_feat.to_vec(), [24, 1]), device);
            let pred_hourly = head.forward(daily_q, precip, 24);
            let pred_vec: Vec<f32> = pred_hourly.into_data().to_vec().unwrap();
            let mut pred_shape = [0f32; 24];
            let mut mse = 0f32;
            for h in 0..24 {
                pred_shape[h] = pred_vec[h] / (24.0 * r.daily_q_m3s);
                let target_shape_h = r.target_hourly_m3s[h] / (24.0 * r.daily_q_m3s);
                mse += (pred_shape[h] - target_shape_h).powi(2);
            }
            (mse / 24.0, pred_shape)
        })
        .collect()
}

fn shape_mse_vs_flat(rows: &[Row]) -> f32 {
    // flat shape = 1/24 every hour; MSE vs target shape.
    let mut total = 0f32;
    for r in rows {
        for h in 0..24 {
            let target_shape_h = r.target_hourly_m3s[h] / (24.0 * r.daily_q_m3s);
            total += (1.0 / 24.0 - target_shape_h).powi(2);
        }
    }
    total / (rows.len() as f32 * 24.0)
}

fn circ_dist(a: usize, b: usize) -> usize {
    let d = (a as i64 - b as i64).unsigned_abs() as usize;
    d.min(24 - d)
}

fn argmax(v: &[f32; 24]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return f32::NAN;
    }
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    println!("=== Phase 3: DisaggHead pretraining on real USGS hourly data ===\n");

    // ---------- 1. Load gauges + reconcile against production daily obs ----------
    let gages = GageMetadata::open(GAGES_CSV)?;
    let camels = CamelsHourlyStore::open(CAMELS_NC)?;
    let obs = UsgsObservationsStore::open(USGS_DAILY)?;

    let overlap: Vec<&GageRow> = gages
        .rows
        .iter()
        .filter(|r| camels.index.position(&r.staid).is_some())
        .collect();
    println!("overlap gauges: {}/{}", overlap.len(), gages.rows.len());

    let mut kept: Vec<GageRow> = Vec::new();
    for row in &overlap {
        let start_dt = ymd(TRAIN_START).and_hms_opt(0, 0, 0).unwrap();
        let n_days = (ymd(TRAIN_END) - ymd(TRAIN_START)).num_days() as usize + 1;
        let Ok((qobs, _)) = camels.read_window(start_dt, n_days * 24, std::slice::from_ref(&row.staid)) else {
            continue;
        };
        let qobs_m3s: ndarray::Array1<f32> = qobs
            .column(0)
            .mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, row.drain_sqkm) } else { f32::NAN });
        let complete = extract_complete_days(&qobs_m3s);
        if complete.len() < 30 {
            continue;
        }
        let daily_vals: Vec<(NaiveDate, f32)> = complete
            .iter()
            .map(|r| (ymd(TRAIN_START) + chrono::Duration::days(r.day_index as i64), r.daily_q_m3s))
            .collect();
        if let Ok(result) = reconcile_gauge(&row.staid, &daily_vals, &obs) {
            if result.keep {
                kept.push((*row).clone());
            }
        }
    }
    println!("reconciled + kept: {}/{}\n", kept.len(), overlap.len());

    // ---------- 2. Split ----------
    let split = build_split_manifest(&kept, SEED);
    println!(
        "split: train={} val={} test={}",
        split.train.len(),
        split.val.len(),
        split.test.len()
    );
    let gauge_by_staid: HashMap<&Staid, &GageRow> = kept.iter().map(|g| (&g.staid, g)).collect();

    // ---------- 3. Build row pools ----------
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
    // Test gauges: BOTH periods -- train-period rows feed the per-gauge
    // climatology baseline; held-out rows are the actual GO/NO-GO set.
    let mut test_train_period_by_gauge: HashMap<Staid, Vec<Row>> = HashMap::new();
    let mut test_heldout_by_gauge: HashMap<Staid, Vec<Row>> = HashMap::new();
    for staid in &split.test {
        let g = gauge_by_staid[staid];
        test_train_period_by_gauge.insert(
            staid.clone(),
            gauge_rows(&camels, staid, g.drain_sqkm, ymd(TRAIN_START), ymd(TRAIN_END)),
        );
        test_heldout_by_gauge.insert(
            staid.clone(),
            gauge_rows(&camels, staid, g.drain_sqkm, ymd(HELDOUT_START), ymd(HELDOUT_END)),
        );
    }
    println!(
        "rows: train={} val={} test_train_period={} test_heldout={}\n",
        train_rows.len(),
        val_rows.len(),
        test_train_period_by_gauge.values().map(|v| v.len()).sum::<usize>(),
        test_heldout_by_gauge.values().map(|v| v.len()).sum::<usize>(),
    );

    // ---------- 4. Train (or load an already-trained checkpoint) ----------
    let device: <AutoI as BackendTypes>::Device = Default::default();
    <AutoI as Backend>::seed(&device, SEED);
    let mut head: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED).init(&device);
    let mut optimizer = build_adam::<DisaggHead<AutoI>, AutoI>();
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xC0FFEE);

    let mut best_val_loss = f32::INFINITY;
    let mut best_head_record = head.clone().into_record();

    if cli.skip_train {
        println!("--skip-train: loading checkpoint {} for re-evaluation only\n", cli.checkpoint);
        let record = CompactRecorder::new().load(cli.checkpoint.clone().into(), &device)?;
        let loaded: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED).init(&device).load_record(record);
        best_head_record = loaded.into_record();
    } else {
    for step in 0..N_STEPS {
        let batch: Vec<&Row> = (0..BATCH_SIZE)
            .map(|_| &train_rows[rng.gen_range(0..train_rows.len())])
            .collect();
        let (new_head, loss) = train_step(head, &mut optimizer, &batch, &device, 1e-3);
        head = new_head;

        if step % MASS_BALANCE_CHECK_EVERY == 0 {
            // Mass-balance re-check DURING training (not just at data-load
            // time): forward a fresh batch and assert output day-mean ==
            // daily_q to f32 tolerance, catching any training-harness
            // regression immediately.
            let inner_head = head.clone().valid(); // -> DisaggHead<I> (inference)
            let inner_device: <I as BackendTypes>::Device = Default::default();
            let check_rows: Vec<Row> = batch.iter().map(|r| (*r).clone()).collect();
            for r in &check_rows {
                let daily_q = Tensor::<I, 2>::from_data(TensorData::new(vec![r.daily_q_m3s], [1, 1]), &inner_device);
                let precip = Tensor::<I, 2>::from_data(TensorData::new(r.precip_feat.to_vec(), [24, 1]), &inner_device);
                let pred = inner_head.forward(daily_q, precip, 24);
                let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();
                let mean: f32 = pred_vec.iter().sum::<f32>() / 24.0;
                assert!(
                    (mean - r.daily_q_m3s).abs() < 1e-3 * r.daily_q_m3s.max(1.0),
                    "MASS BALANCE VIOLATED at step {step}: pred mean={mean}, daily_q={}",
                    r.daily_q_m3s
                );
            }
        }

        if step % VAL_EVERY == 0 || step == N_STEPS - 1 {
            let inner_head = head.clone().valid();
            let inner_device: <I as BackendTypes>::Device = Default::default();
            let val_scores = eval_rows(&inner_head, &val_rows, &inner_device);
            let val_loss = val_scores.iter().map(|(mse, _)| mse).sum::<f32>() / val_scores.len() as f32;
            println!("step {step:5}  train_loss={loss:.6}  val_loss={val_loss:.6}");
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_head_record = head.clone().into_record();
            }
        }
    }
    println!("\nbest val_loss = {best_val_loss:.6}\n");
    }

    // Restore best checkpoint for evaluation.
    let best_head: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED).init(&device).load_record(best_head_record);
    let best_head_inference = best_head.valid();
    let inner_device: <I as BackendTypes>::Device = Default::default();

    // ---------- 5. Held-out GO/NO-GO evaluation ----------
    // Forward passes happen EXACTLY ONCE per (gauge, held-out row) here;
    // the per-gauge NSE vectors are cached and reused by the bootstrap
    // below (resampling gauges, not recomputing forward passes).
    let mut per_gauge_nse: HashMap<Staid, Vec<f32>> = HashMap::new();
    let mut per_gauge_shape_nse: Vec<f32> = Vec::new();
    let mut climatology_shape_nse: Vec<f32> = Vec::new();
    let mut head_peak_err_precip_active: Vec<usize> = Vec::new();
    let mut clim_peak_err_precip_active: Vec<usize> = Vec::new();
    let mut head_storm_day_nse: Vec<f32> = Vec::new();
    let mut clim_storm_day_nse: Vec<f32> = Vec::new();

    for staid in &split.test {
        let heldout = &test_heldout_by_gauge[staid];
        let train_period = &test_train_period_by_gauge[staid];
        if heldout.is_empty() || train_period.is_empty() {
            continue;
        }

        // Per-gauge climatology template: mean shape across its OWN
        // train-period days (test gauges are never seen by the neural
        // model, but climatology is a NON-learned per-gauge baseline).
        let mut clim_shape = [0f32; 24];
        for r in train_period {
            for h in 0..24 {
                clim_shape[h] += r.target_hourly_m3s[h] / (24.0 * r.daily_q_m3s);
            }
        }
        for h in 0..24 {
            clim_shape[h] /= train_period.len() as f32;
        }

        let head_scores = eval_rows(&best_head_inference, heldout, &inner_device);
        let flat_mse = shape_mse_vs_flat(heldout);
        let mut gauge_nse = Vec::with_capacity(heldout.len());

        for (row, (head_mse, head_shape)) in heldout.iter().zip(&head_scores) {
            let nse = 1.0 - head_mse / flat_mse.max(1e-12);
            per_gauge_shape_nse.push(nse);
            gauge_nse.push(nse);

            let mut clim_mse = 0f32;
            for h in 0..24 {
                let target_shape_h = row.target_hourly_m3s[h] / (24.0 * row.daily_q_m3s);
                clim_mse += (clim_shape[h] - target_shape_h).powi(2);
            }
            clim_mse /= 24.0;
            let clim_nse = 1.0 - clim_mse / flat_mse.max(1e-12);
            climatology_shape_nse.push(clim_nse);

            // Genuine storm day = a real, meaningful rain total (>5mm),
            // NOT "any nonzero z-scored feature value" (that matched
            // nearly every row, including near-dry days, diluting this
            // comparison against near-flat days a fixed climatology
            // handles just as well).
            let storm_day = row.daily_precip_total_mm > 5.0;
            if storm_day {
                let true_peak = argmax(&row.target_hourly_m3s);
                head_peak_err_precip_active.push(circ_dist(argmax(head_shape), true_peak));
                clim_peak_err_precip_active.push(circ_dist(argmax(&clim_shape), true_peak));
                head_storm_day_nse.push(nse);
                clim_storm_day_nse.push(clim_nse);
            }
        }
        per_gauge_nse.insert(staid.clone(), gauge_nse);
    }

    // ---------- 6. Bootstrap CI over TEST GAUGES (resamples CACHED scores) ----------
    let mut boot_medians: Vec<f32> = Vec::new();
    let mut boot_rng = StdRng::seed_from_u64(SEED ^ 0xB007);
    let test_gauges_with_data: Vec<&Staid> = per_gauge_nse.keys().collect();
    for _ in 0..2000 {
        let mut resample_nse: Vec<f32> = Vec::new();
        for _ in 0..test_gauges_with_data.len() {
            let g = test_gauges_with_data[boot_rng.gen_range(0..test_gauges_with_data.len())];
            resample_nse.extend(per_gauge_nse[g].iter().copied());
        }
        if !resample_nse.is_empty() {
            boot_medians.push(median(resample_nse));
        }
    }
    boot_medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lo = boot_medians[(boot_medians.len() as f32 * 0.025) as usize];
    let ci_hi = boot_medians[(boot_medians.len() as f32 * 0.975) as usize];

    // ---------- 7. Report ----------
    let median_shape_nse = median(per_gauge_shape_nse.clone());
    let median_clim_nse = median(climatology_shape_nse.clone());
    let median_head_peak_err = median(head_peak_err_precip_active.iter().map(|&v| v as f32).collect());
    let median_clim_peak_err = median(clim_peak_err_precip_active.iter().map(|&v| v as f32).collect());
    let median_head_storm_nse = median(head_storm_day_nse.clone());
    let median_clim_storm_nse = median(clim_storm_day_nse.clone());

    println!("=== Held-out GO/NO-GO gate (test gauges x 2014-2018) ===");
    println!("n test gauges (with heldout data): {}", test_heldout_by_gauge.values().filter(|v| !v.is_empty()).count());
    println!("n heldout (gauge,day) rows:         {}", per_gauge_shape_nse.len());
    println!("median shape-NSE (head vs flat):    {median_shape_nse:.4}  (95% CI [{ci_lo:.4}, {ci_hi:.4}])");
    println!("median shape-NSE (climatology):     {median_clim_nse:.4}");
    println!(
        "\n--- Genuine storm days only (daily precip > 5mm, n={}) ---",
        head_storm_day_nse.len()
    );
    println!("median shape-NSE (head):         {median_head_storm_nse:.4}");
    println!("median shape-NSE (climatology):  {median_clim_storm_nse:.4}");
    println!(
        "median peak-hour error: head={median_head_peak_err:.2}h  climatology={median_clim_peak_err:.2}h"
    );

    let go_primary = ci_lo > 0.0;
    let go_beats_climatology_storm = median_head_storm_nse > median_clim_storm_nse;
    let go_timing = median_head_peak_err < median_clim_peak_err;
    println!("\n--- Pre-registered GO/NO-GO criteria (storm-day-restricted) ---");
    println!("(1) median shape-NSE CI excludes 0:              {go_primary}");
    println!("(2) head beats climatology on storm-day shape-NSE: {go_beats_climatology_storm}");
    println!("(3) head beats climatology on storm-day timing:    {go_timing}");

    let verdict = if go_primary && go_beats_climatology_storm && go_timing {
        "GO"
    } else if !go_primary {
        "INCONCLUSIVE (primary CI straddles zero)"
    } else {
        "PARTIAL / NO-GO on secondary criteria"
    };
    println!("\nVERDICT: {verdict}");

    // ---------- 8. Save best checkpoint (inline; formal save/load fn is Phase 4) ----------
    std::fs::create_dir_all("output/disagg_pretrain")?;
    CompactRecorder::new()
        .record(best_head.into_record(), "output/disagg_pretrain/best_disagg".into())
        .expect("save pretrained disagg head");
    println!("\nsaved checkpoint -> output/disagg_pretrain/best_disagg.mpk");

    Ok(())
}
