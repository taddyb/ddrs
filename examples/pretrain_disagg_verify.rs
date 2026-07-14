//! Verification for the Phase-3 REAL-USGS-hourly-pretrained DisaggHead:
//! does it actually respect rain? Unlike the production checkpoints (which
//! only ever see daily-aggregated loss), this pretrained head was trained
//! DIRECTLY against real hourly USGS discharge, so we can compare its
//! prediction against the REAL hourly ground truth (not just mass-balance
//! self-consistency) -- the strongest possible check.
//!
//! Dumps a CSV per requested gauge: hour, daily_input, disagg_hourly (m3/s),
//! real_hourly_target (m3/s, actual measured USGS discharge), precip_raw_mm_hr.
//!
//!   cargo run --release --example pretrain_disagg_verify -- \
//!       --checkpoint output/disagg_pretrain/best_disagg \
//!       --output output/disagg_pretrain/verify.csv \
//!       --staids 01031500,03592718,02196000 --start-date 2015-01-01 --n-days 10

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

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "output/disagg_pretrain/best_disagg")]
    checkpoint: String,
    #[arg(long, default_value = "output/disagg_pretrain/verify.csv")]
    output: String,
    /// Comma-separated STAIDs (held-out test gauges recommended).
    #[arg(long)]
    staids: String,
    #[arg(long)]
    start_date: String,
    #[arg(long, default_value_t = 10)]
    n_days: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if let Some(parent) = std::path::Path::new(&cli.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let gages = GageMetadata::open(GAGES_CSV)?;
    let camels = CamelsHourlyStore::open(CAMELS_NC)?;

    let device: <I as BackendTypes>::Device = Default::default();
    let template: DisaggHead<I> = DisaggHeadConfig::new(SEED).init(&device);
    let record = CompactRecorder::new().load(cli.checkpoint.clone().into(), &device)?;
    let head: DisaggHead<I> = template.load_record(record);

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")?;
    let end = start + chrono::Duration::days(cli.n_days as i64 - 1);
    let start_dt = start.and_hms_opt(0, 0, 0).unwrap();
    let n_hours = cli.n_days * 24;

    let mut f = std::fs::File::create(&cli.output)?;
    use std::io::Write;
    writeln!(f, "staid,comid,hour,day,daily_input,disagg_hourly,real_hourly_target,precip_raw_mm_hr")?;

    let mut total_rows = 0usize;
    for staid_str in cli.staids.split(',') {
        let staid = Staid::new(staid_str.trim());
        let Some(gauge) = gages.rows.iter().find(|r| r.staid == staid) else {
            eprintln!("skipping {staid}: not in gages_3000.csv");
            continue;
        };
        let comid = gauge.comid.unwrap_or(-1);

        let Ok((qobs, precip_raw)) = camels.read_window(start_dt, n_hours, std::slice::from_ref(&staid)) else {
            eprintln!("skipping {staid}: read_window failed (out of range or missing)");
            continue;
        };
        let qobs_m3s: ndarray::Array1<f32> = qobs
            .column(0)
            .mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, gauge.drain_sqkm) } else { f32::NAN });
        let complete = extract_complete_days(&qobs_m3s);
        if complete.is_empty() {
            eprintln!("skipping {staid}: no complete days in [{start}, {end}]");
            continue;
        }

        let precip_col: ndarray::Array1<f32> = precip_raw.column(0).to_owned();
        let precip_norm = normalize_gauge_precip(precip_col.clone());
        let day_indices: Vec<usize> = complete.iter().map(|r| r.day_index).collect();
        let precip_feats = slice_precip_features(&precip_norm, &day_indices);

        for (row, feat) in complete.iter().zip(&precip_feats) {
            let daily_q = Tensor::<I, 2>::from_data(TensorData::new(vec![row.daily_q_m3s], [1, 1]), &device);
            let precip_t = Tensor::<I, 2>::from_data(TensorData::new(feat.to_vec(), [24, 1]), &device);
            let pred = head.forward(daily_q, precip_t, 24);
            let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();

            for h in 0..24 {
                let abs_hour = row.day_index * 24 + h;
                let precip_mm_hr = precip_col[abs_hour];
                writeln!(
                    f,
                    "{staid},{comid},{abs_hour},{},{},{},{},{}",
                    row.day_index, row.daily_q_m3s, pred_vec[h], row.target_hourly_m3s[h], precip_mm_hr
                )?;
                total_rows += 1;
            }
        }
        println!("{staid}: wrote {} complete days", complete.len());
    }

    println!("wrote {} ({total_rows} rows)", cli.output);
    Ok(())
}
