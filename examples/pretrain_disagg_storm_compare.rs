//! Direct visual comparison on a GENUINE storm day: real hourly USGS
//! discharge vs. the pretrained-on-real-data DisaggHead vs. a per-gauge
//! climatology template (mean shape from that gauge's OWN 1998-2013 days,
//! no precip information at all). This is the concrete comparison behind
//! the storm-day gate numbers in the Phase 3 findings -- answers "does the
//! pretrained head actually add value over a naive non-precip-aware
//! average template, on the days that matter."
//!
//!   cargo run --release --example pretrain_disagg_storm_compare -- \
//!       --checkpoint output/disagg_pretrain/best_disagg \
//!       --staid 02198100 --storm-date 2016-09-02 --n-days-context 3 \
//!       --output output/disagg_pretrain/storm_compare_02198100.csv

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};
use chrono::NaiveDate;
use clap::Parser;

use ddrs::data::ids::Staid;
use ddrs::data::store::gage_csv::GageMetadata;
use ddrs::data::store::CamelsHourlyStore;
use ddrs::nn::{DisaggHead, DisaggHeadConfig};
use ddrs::pretrain::{extract_complete_days, normalize_gauge_precip, qobs_mm_hr_to_m3s, slice_precip_features};

type I = NdArray<f32>;

const GAGES_CSV: &str = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";
const CAMELS_NC: &str = "/mnt/ssd1/data/camels_hourly/usgs-streamflow-nldas_hourly.nc";
const SEED: u64 = 42;
const CLIMATOLOGY_START: (i32, u32, u32) = (1998, 1, 1);
const CLIMATOLOGY_END: (i32, u32, u32) = (2013, 12, 31);

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "output/disagg_pretrain/best_disagg")]
    checkpoint: String,
    #[arg(long)]
    staid: String,
    #[arg(long)]
    storm_date: String,
    #[arg(long, default_value_t = 3)]
    n_days_context: i64,
    #[arg(long)]
    output: String,
}

fn ymd(t: (i32, u32, u32)) -> NaiveDate {
    NaiveDate::from_ymd_opt(t.0, t.1, t.2).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if let Some(parent) = std::path::Path::new(&cli.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let gages = GageMetadata::open(GAGES_CSV)?;
    let camels = CamelsHourlyStore::open(CAMELS_NC)?;
    let staid = Staid::new(&cli.staid);
    let gauge = gages.rows.iter().find(|r| r.staid == staid).expect("staid not in gages_3000.csv");

    // ---------- 1. Climatology template from the gauge's OWN 1998-2013 days ----------
    let clim_start_dt = ymd(CLIMATOLOGY_START).and_hms_opt(0, 0, 0).unwrap();
    let clim_n_days = (ymd(CLIMATOLOGY_END) - ymd(CLIMATOLOGY_START)).num_days() as usize + 1;
    let (clim_qobs, _) = camels.read_window(clim_start_dt, clim_n_days * 24, std::slice::from_ref(&staid))?;
    let clim_qobs_m3s: ndarray::Array1<f32> = clim_qobs
        .column(0)
        .mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, gauge.drain_sqkm) } else { f32::NAN });
    let clim_days = extract_complete_days(&clim_qobs_m3s);
    let mut clim_shape = [0f32; 24];
    for r in &clim_days {
        for h in 0..24 {
            clim_shape[h] += r.target_hourly_m3s[h] / (24.0 * r.daily_q_m3s);
        }
    }
    for v in &mut clim_shape {
        *v /= clim_days.len() as f32;
    }
    println!("climatology template built from {} training-period days", clim_days.len());

    // ---------- 2. Load the pretrained head ----------
    let device: <I as BackendTypes>::Device = Default::default();
    let template: DisaggHead<I> = DisaggHeadConfig::new(SEED).init(&device);
    let record = CompactRecorder::new().load(cli.checkpoint.clone().into(), &device)?;
    let head: DisaggHead<I> = template.load_record(record);

    // ---------- 3. Read the storm window (+/- context days) ----------
    let storm_date = NaiveDate::parse_from_str(&cli.storm_date, "%Y-%m-%d")?;
    let start = storm_date - chrono::Duration::days(cli.n_days_context);
    let n_days = (2 * cli.n_days_context + 1) as usize;
    let start_dt = start.and_hms_opt(0, 0, 0).unwrap();
    let (qobs, precip_raw) = camels.read_window(start_dt, n_days * 24, std::slice::from_ref(&staid))?;
    let qobs_m3s: ndarray::Array1<f32> = qobs
        .column(0)
        .mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, gauge.drain_sqkm) } else { f32::NAN });
    let complete = extract_complete_days(&qobs_m3s);
    assert!(!complete.is_empty(), "no complete days around the storm date -- pick a different one");

    let precip_col: ndarray::Array1<f32> = precip_raw.column(0).to_owned();
    let precip_norm = normalize_gauge_precip(precip_col.clone());
    let day_indices: Vec<usize> = complete.iter().map(|r| r.day_index).collect();
    let precip_feats = slice_precip_features(&precip_norm, &day_indices);

    let mut f = std::fs::File::create(&cli.output)?;
    use std::io::Write;
    writeln!(f, "staid,hour,daily_input,real_hourly,head_pred,climatology_pred,precip_raw_mm_hr")?;

    for (row, feat) in complete.iter().zip(&precip_feats) {
        let daily_q = Tensor::<I, 2>::from_data(TensorData::new(vec![row.daily_q_m3s], [1, 1]), &device);
        let precip_t = Tensor::<I, 2>::from_data(TensorData::new(feat.to_vec(), [24, 1]), &device);
        let pred = head.forward(daily_q, precip_t, 24);
        let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();

        for h in 0..24 {
            let abs_hour = row.day_index * 24 + h;
            let clim_pred = clim_shape[h] * 24.0 * row.daily_q_m3s;
            writeln!(
                f,
                "{staid},{abs_hour},{},{},{},{},{}",
                row.daily_q_m3s, row.target_hourly_m3s[h], pred_vec[h], clim_pred, precip_col[abs_hour]
            )?;
        }
    }
    println!("wrote {}", cli.output);
    Ok(())
}
