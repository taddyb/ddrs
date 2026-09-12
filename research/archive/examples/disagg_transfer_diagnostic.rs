//! Offline distribution-shift diagnostic for the pretrained capacity-boosted
//! disagg head (`output/disagg_pretrain/capacity_chunk1.mpk`), gating the
//! frozen / fine-tuned CONUS training arms (see
//! `config/experiments/kan_disagg_conus_{frozen,finetune}_chunk1.yaml`).
//!
//! The head was pretrained on CAMELS gauge-outlet, whole-watershed integrated
//! hydrographs; production feeds it per-reach incremental lateral inflow (Q'
//! at unit-catchment scale). This example forwards the SAME checkpoint against
//! both domains — a modest sample of real CONUS reaches (dHBV2-UH daily Q' +
//! AORC hourly precip, the exact production stores) and a CAMELS gauge sample
//! — and compares two output-shape diagnostics per forwarded day:
//!
//!   - peak-hour probability mass: max_k shape[k], where shape[k] =
//!     pred[k] / (24 · daily_q) is the head's implied within-day softmax
//!     (uniform = 1/24 ≈ 0.0417, delta collapse → 1).
//!   - peakiness ratio: peak-hour value / mean hourly value (= 24 · peak
//!     mass under per-day mass conservation; uniform = 1).
//!
//! This is a sanity gate, not a statistical test: if production-domain
//! outputs collapse near-uniform / near-delta or land on a wildly different
//! peakiness scale than the pretraining domain, the transfer is visibly
//! broken and the GPU runs are not worth starting.
//!
//!   cargo run --release --example disagg_transfer_diagnostic

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::BackendTypes;
use burn::tensor::{Tensor, TensorData};
use chrono::NaiveDate;

use ddrs::data::ids::Comid;
use ddrs::data::store::gage_csv::GageMetadata;
use ddrs::data::store::{AorcPrecipStore, CamelsHourlyStore, StreamflowStore};
use ddrs::nn::{DisaggHead, DisaggHeadConfig};
use ddrs::pretrain::{normalize_gauge_precip, qobs_mm_hr_to_m3s};

type I = NdArray<f32>;

/// Pretrained checkpoint base (`CompactRecorder` appends/keeps `.mpk`) and
/// its exact training architecture (`pretrain_disagg_capacity --chunk-days 1`).
const CHECKPOINT: &str = "output/disagg_pretrain/capacity_chunk1";
const SEED: u64 = 42;
const HIDDEN_SIZE: usize = 16;
const NUM_HIDDEN_LAYERS: usize = 2;
const GRID: usize = 20;
const CHUNK_DAYS: usize = 1;

// Production stores (same paths as the experiment configs / conus-hourly).
const STREAMFLOW: &str = "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic";
const AORC: &str = "/mnt/ssd1/data/aorc/merit_unit_catchments.zarr";

// Pretraining-domain stores (same as pretrain_disagg_capacity).
const GAGES_CSV: &str = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";
const CAMELS_NC: &str = "/mnt/ssd1/data/camels_hourly/usgs-streamflow-nldas_hourly.nc";

/// Representative multi-day window (spring storm season, inside every store's
/// axis). 60 days ≈ the same order as a production rho-window (90 days).
const WINDOW_START: (i32, u32, u32) = (1998, 5, 1);
const WINDOW_DAYS: usize = 60;
const WINDOW_HOURS: usize = WINDOW_DAYS * 24;

const N_PROD_REACHES: usize = 20;
const N_CAMELS_GAUGES: usize = 15;
/// Pretraining's storm-pool floor (`pretrain_disagg_capacity`,
/// MIN_DAILY_Q_M3S) — applied to the CAMELS reference sample only, so it
/// represents the distribution the head was actually trained on.
const CAMELS_MIN_DAILY_Q: f32 = 0.01;

#[derive(Clone)]
struct DayDiag {
    peak_mass: f32,
    peakiness: f32,
    daily_q: f32,
    /// That day's raw precip total (mm) — for storm-day stratification with
    /// the same >5 mm threshold `pretrain_disagg_capacity` used.
    precip_mm: f32,
}

/// Forward one reach/gauge window and collect per-day shape diagnostics.
/// `precip_raw_mm_hr` is the UN-normalized hourly precip (mm/hr), used only
/// to tag each day's storm magnitude.
fn forward_diags(
    head: &DisaggHead<I>,
    daily_q: &[f32],
    precip_norm: &[f32],
    precip_raw_mm_hr: &[f32],
    device: &<I as BackendTypes>::Device,
) -> Vec<DayDiag> {
    let d = daily_q.len();
    let n_hourly = d * 24;
    assert_eq!(precip_norm.len(), n_hourly);
    assert_eq!(precip_raw_mm_hr.len(), n_hourly);
    let q_t = Tensor::<I, 2>::from_data(TensorData::new(daily_q.to_vec(), [d, 1]), device);
    let p_t = Tensor::<I, 2>::from_data(TensorData::new(precip_norm.to_vec(), [n_hourly, 1]), device);
    let pred: Vec<f32> = head.forward(q_t, p_t, n_hourly).into_data().to_vec().unwrap();

    let mut out = Vec::with_capacity(d);
    for day in 0..d {
        let q = daily_q[day];
        if !(q > 0.0) {
            continue;
        }
        let day_pred = &pred[day * 24..day * 24 + 24];
        let peak = day_pred.iter().cloned().fold(f32::MIN, f32::max);
        let mean = day_pred.iter().sum::<f32>() / 24.0;
        if !(mean > 0.0) {
            continue;
        }
        let precip_mm: f32 = precip_raw_mm_hr[day * 24..day * 24 + 24]
            .iter()
            .map(|&v| v.max(0.0))
            .sum();
        out.push(DayDiag {
            peak_mass: peak / (24.0 * q),
            peakiness: peak / mean,
            daily_q: q,
            precip_mm,
        });
    }
    out
}

fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((sorted.len() - 1) as f32 * q).round() as usize;
    sorted[idx]
}

struct Summary {
    n_days: usize,
    peak_mass: [f32; 3],  // q25, median, q75
    peakiness: [f32; 3],
    median_daily_q: f32,
}

fn summarize(diags: &[DayDiag]) -> Summary {
    let mut pm: Vec<f32> = diags.iter().map(|d| d.peak_mass).collect();
    let mut pk: Vec<f32> = diags.iter().map(|d| d.peakiness).collect();
    let mut dq: Vec<f32> = diags.iter().map(|d| d.daily_q).collect();
    pm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pk.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dq.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Summary {
        n_days: diags.len(),
        peak_mass: [quantile(&pm, 0.25), quantile(&pm, 0.5), quantile(&pm, 0.75)],
        peakiness: [quantile(&pk, 0.25), quantile(&pk, 0.5), quantile(&pk, 0.75)],
        median_daily_q: quantile(&dq, 0.5),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device: <I as BackendTypes>::Device = Default::default();
    let (y, m, d) = WINDOW_START;
    let window_start = NaiveDate::from_ymd_opt(y, m, d).unwrap();

    println!("=== disagg transfer diagnostic: {CHECKPOINT}.mpk ===");
    println!(
        "architecture: hidden={HIDDEN_SIZE} layers={NUM_HIDDEN_LAYERS} grid={GRID} chunk_days={CHUNK_DAYS}"
    );
    println!("window: {window_start} + {WINDOW_DAYS} days\n");

    let template: DisaggHead<I> = DisaggHeadConfig::new(SEED)
        .with_hidden_size(HIDDEN_SIZE)
        .with_num_hidden_layers(NUM_HIDDEN_LAYERS)
        .with_grid(GRID)
        .with_chunk_days(CHUNK_DAYS)
        .init(&device);
    let record = CompactRecorder::new().load(CHECKPOINT.into(), &device)?;
    let head: DisaggHead<I> = template.load_record(record);

    // ---- Production domain: real CONUS reaches, dHBV2-UH Q' + AORC precip ----
    let streamflow = StreamflowStore::open(STREAMFLOW)?;
    let aorc = AorcPrecipStore::open(AORC)?;

    // Deterministic stride sample over the store's divide axis, keeping only
    // reaches with AORC coverage (0.0-filled reaches see a flat precip window
    // and would understate the shift, not reveal it).
    let all_ids = streamflow.index.ids();
    let stride = (all_ids.len() / 4096).max(1);
    let mut prod_comids: Vec<Comid> = Vec::with_capacity(N_PROD_REACHES);
    for c in all_ids.iter().step_by(stride) {
        if aorc.coverage(std::slice::from_ref(c)) == 1 {
            prod_comids.push(*c);
            if prod_comids.len() == N_PROD_REACHES {
                break;
            }
        }
    }
    println!(
        "production sample: {} reaches (stride {stride} over {} divide ids)",
        prod_comids.len(),
        all_ids.len()
    );

    let q_daily = streamflow.read_window_daily(window_start, WINDOW_DAYS, &prod_comids)?;
    let precip_raw = aorc.read_window_hourly(window_start, WINDOW_HOURS, &prod_comids)?;

    let mut prod_diags: Vec<DayDiag> = Vec::new();
    for (col, comid) in prod_comids.iter().enumerate() {
        let daily: Vec<f32> = (0..WINDOW_DAYS).map(|t| q_daily[(t, col)]).collect();
        // Per-reach window z-score of log1p(precip) — the exact production
        // transform (`normalize_gauge_precip` wraps `normalize_precip`).
        let precip_raw_vec: Vec<f32> = (0..WINDOW_HOURS).map(|h| precip_raw[(h, col)]).collect();
        let precip_norm = normalize_gauge_precip(ndarray::Array1::from(precip_raw_vec.clone()));
        let precip_vec: Vec<f32> = (0..WINDOW_HOURS).map(|h| precip_norm[(h, 0)]).collect();
        let diags = forward_diags(&head, &daily, &precip_vec, &precip_raw_vec, &device);
        if col < 3 {
            println!("  e.g. {comid:?}: median daily Q' = {:.4} m³/s, {} days", {
                let mut v = daily.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            }, diags.len());
        }
        prod_diags.extend(diags);
    }

    // ---- Pretraining domain: CAMELS gauge-outlet hydrograph inputs ----
    let gages = GageMetadata::open(GAGES_CSV)?;
    let camels = CamelsHourlyStore::open(CAMELS_NC)?;
    let start_dt = window_start.and_hms_opt(0, 0, 0).unwrap();

    let mut camels_diags: Vec<DayDiag> = Vec::new();
    let mut n_gauges = 0usize;
    for row in &gages.rows {
        if n_gauges == N_CAMELS_GAUGES {
            break;
        }
        if camels.index.position(&row.staid).is_none() {
            continue;
        }
        let Ok((qobs, precip)) =
            camels.read_window(start_dt, WINDOW_HOURS, std::slice::from_ref(&row.staid))
        else {
            continue;
        };
        let qobs_m3s: Vec<f32> = qobs
            .column(0)
            .iter()
            .map(|&v| if v.is_finite() { qobs_mm_hr_to_m3s(v, row.drain_sqkm) } else { f32::NAN })
            .collect();
        let precip_col = precip.column(0).to_owned();
        let precip_norm = normalize_gauge_precip(precip_col);

        // Complete days only (real gauge data has gaps), pretraining's
        // low-flow floor applied so this sample matches the trained domain.
        let mut gauge_days = 0usize;
        for day in 0..WINDOW_DAYS {
            let hours = &qobs_m3s[day * 24..day * 24 + 24];
            if hours.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let daily_q = hours.iter().sum::<f32>() / 24.0;
            if daily_q < CAMELS_MIN_DAILY_Q {
                continue;
            }
            let precip_day: Vec<f32> =
                (day * 24..day * 24 + 24).map(|h| precip_norm[(h, 0)]).collect();
            let precip_day_raw: Vec<f32> = (day * 24..day * 24 + 24)
                .map(|h| {
                    let v = precip[(h, 0)];
                    if v.is_finite() {
                        v
                    } else {
                        0.0
                    }
                })
                .collect();
            camels_diags.extend(forward_diags(
                &head,
                &[daily_q],
                &precip_day,
                &precip_day_raw,
                &device,
            ));
            gauge_days += 1;
        }
        if gauge_days > 0 {
            n_gauges += 1;
        }
    }
    println!("pretraining-domain sample: {n_gauges} CAMELS gauges\n");

    // ---- Summary + verdict ----
    /// Same storm-day threshold as `pretrain_disagg_capacity`'s storm CSV.
    const STORM_MM: f32 = 5.0;
    let storms = |diags: &[DayDiag]| -> Vec<DayDiag> {
        diags.iter().filter(|d| d.precip_mm > STORM_MM).cloned().collect()
    };
    let prod_storm_diags = storms(&prod_diags);
    let cam_storm_diags = storms(&camels_diags);
    let prod = summarize(&prod_diags);
    let cam = summarize(&camels_diags);
    let prod_storm = summarize(&prod_storm_diags);
    let cam_storm = summarize(&cam_storm_diags);
    let uniform = 1.0 / 24.0;

    let print_block = |label: &str, p: &Summary, c: &Summary| {
        println!("--- {label} ---");
        println!("{:<34} {:>22} {:>22}", "", "production (CONUS)", "pretrain (CAMELS)");
        println!("{:<34} {:>22} {:>22}", "forwarded days", p.n_days, c.n_days);
        println!(
            "{:<34} {:>22.4} {:>22.4}",
            "median daily Q [m³/s]", p.median_daily_q, c.median_daily_q
        );
        println!(
            "{:<34} {:>22} {:>22}",
            "peak-hour mass median [IQR]",
            format!("{:.4} [{:.4},{:.4}]", p.peak_mass[1], p.peak_mass[0], p.peak_mass[2]),
            format!("{:.4} [{:.4},{:.4}]", c.peak_mass[1], c.peak_mass[0], c.peak_mass[2]),
        );
        println!(
            "{:<34} {:>22} {:>22}",
            "peakiness (peak/mean) median [IQR]",
            format!("{:.3} [{:.3},{:.3}]", p.peakiness[1], p.peakiness[0], p.peakiness[2]),
            format!("{:.3} [{:.3},{:.3}]", c.peakiness[1], c.peakiness[0], c.peakiness[2]),
        );
        println!();
    };
    print_block("all forwarded days", &prod, &cam);
    print_block(
        &format!("storm days only (raw precip > {STORM_MM} mm/day)"),
        &prod_storm,
        &cam_storm,
    );
    println!("(uniform softmax: peak mass = {uniform:.4}, peakiness = 1.000; delta collapse: peak mass -> 1.0)\n");

    // Sanity-gate rules (deliberately coarse):
    //  - near-uniform collapse: production STORM-day shapes are essentially
    //    flat (quiet baseflow days are legitimately near-flat, so the
    //    collapse test must be on storm days).
    //  - near-delta collapse: production dumps a day into one hour.
    //  - scale mismatch: production peakiness EXCESS (over uniform's 1.0) is
    //    >5x or <0.2x the pretraining domain's, on the storm-day subset when
    //    both domains have enough storm days (else all days).
    let enough_storms = prod_storm.n_days >= 20 && cam_storm.n_days >= 20;
    let (p_gate, c_gate, gate_label) = if enough_storms {
        (&prod_storm, &cam_storm, "storm days")
    } else {
        (&prod, &cam, "all days (too few storm days)")
    };
    let prod_excess = p_gate.peakiness[1] - 1.0;
    let cam_excess = c_gate.peakiness[1] - 1.0;
    let ratio = prod_excess / cam_excess.max(1e-6);
    let collapsed_uniform = prod_excess < 0.05;
    let collapsed_delta = p_gate.peak_mass[1] > 0.9;
    let scale_mismatch = !(0.2..=5.0).contains(&ratio);
    println!(
        "gate ({gate_label}): peakiness excess production {prod_excess:.3}, pretrain {cam_excess:.3} (ratio {ratio:.2})"
    );

    if collapsed_uniform || collapsed_delta || scale_mismatch {
        println!(
            "verdict: TRANSFER LOOKS DEGENERATE ({}{}{})",
            if collapsed_uniform { "near-uniform collapse; " } else { "" },
            if collapsed_delta { "near-delta collapse; " } else { "" },
            if scale_mismatch { "peakiness scale mismatch; " } else { "" },
        );
    } else {
        println!("verdict: TRANSFER LOOKS REASONABLE");
    }
    Ok(())
}
