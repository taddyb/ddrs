//! Storm-day visual comparison for the capacity/lookback ablation
//! checkpoints (`pretrain_disagg_capacity`, boosted to 2 KanLayers/grid=20).
//! Uses the REAL PRODUCTION `DisaggHead`/`DisaggHeadConfig` directly. Mirrors
//! the same real storm days used in earlier rounds for direct comparability.
//!
//!   cargo run --release --example pretrain_disagg_capacity_storm_compare -- \
//!       --checkpoint output/disagg_pretrain/capacity_chunk1 --chunk-days 1 \
//!       --staid 02198100 --storm-date 2016-09-02 \
//!       --output output/disagg_pretrain/capacity_chunk1_storm_02198100.csv

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
use ddrs::pretrain::{extract_complete_day_windows, normalize_gauge_precip, qobs_mm_hr_to_m3s};

type I = NdArray<f32>;

const GAGES_CSV: &str = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";
const CAMELS_NC: &str = "/mnt/ssd1/data/camels_hourly/usgs-streamflow-nldas_hourly.nc";
const SEED: u64 = 42;
const CLIMATOLOGY_START: (i32, u32, u32) = (1998, 1, 1);
const CLIMATOLOGY_END: (i32, u32, u32) = (2013, 12, 31);
const WINDOW_DAYS: usize = 9;
const WINDOW_HOURS: usize = WINDOW_DAYS * 24;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    checkpoint: String,
    #[arg(long)]
    chunk_days: usize,
    #[arg(long, default_value_t = 2)]
    num_hidden_layers: usize,
    #[arg(long, default_value_t = 20)]
    grid: usize,
    #[arg(long, default_value_t = 16)]
    hidden_size: usize,
    #[arg(long)]
    staid: String,
    #[arg(long)]
    storm_date: String,
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

    // Climatology: 24h template from the gauge's own 1998-2013 days.
    let clim_start_dt = ymd(CLIMATOLOGY_START).and_hms_opt(0, 0, 0).unwrap();
    let clim_n_days = (ymd(CLIMATOLOGY_END) - ymd(CLIMATOLOGY_START)).num_days() as usize + 1;
    let (clim_qobs, _) = camels.read_window(clim_start_dt, clim_n_days * 24, std::slice::from_ref(&staid))?;
    let clim_qobs_m3s: ndarray::Array1<f32> = clim_qobs.column(0).mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, gauge.drain_sqkm) } else { f32::NAN });
    let clim_windows = extract_complete_day_windows(&clim_qobs_m3s, 1);
    let mut clim_shape_24 = [0f32; 24];
    for w in &clim_windows {
        let r = &w[0];
        for h in 0..24 {
            clim_shape_24[h] += r.target_hourly_m3s[h] / (24.0 * r.daily_q_m3s);
        }
    }
    for v in &mut clim_shape_24 { *v /= clim_windows.len() as f32; }

    let device: <I as BackendTypes>::Device = Default::default();
    let template: DisaggHead<I> = DisaggHeadConfig::new(SEED)
        .with_hidden_size(cli.hidden_size)
        .with_num_hidden_layers(cli.num_hidden_layers)
        .with_grid(cli.grid)
        .with_chunk_days(cli.chunk_days)
        .init(&device);
    let record = CompactRecorder::new().load(cli.checkpoint.clone().into(), &device)?;
    let head: DisaggHead<I> = template.load_record(record);

    let storm_date = NaiveDate::parse_from_str(&cli.storm_date, "%Y-%m-%d")?;
    let win_start = storm_date - chrono::Duration::days((WINDOW_DAYS as i64 - 1) / 2);
    let start_dt = win_start.and_hms_opt(0, 0, 0).unwrap();
    let (qobs, precip_raw) = camels.read_window(start_dt, WINDOW_HOURS, std::slice::from_ref(&staid))?;
    let qobs_m3s: ndarray::Array1<f32> = qobs.column(0).mapv(|v| if v.is_finite() { qobs_mm_hr_to_m3s(v, gauge.drain_sqkm) } else { f32::NAN });
    let windows = extract_complete_day_windows(&qobs_m3s, WINDOW_DAYS);
    assert!(!windows.is_empty(), "no complete {WINDOW_DAYS}-day window around the storm date -- pick a different one");
    let w = &windows[0];

    let precip_col: ndarray::Array1<f32> = precip_raw.column(0).to_owned();
    let precip_norm = normalize_gauge_precip(precip_col.clone());
    let precip_feat: Vec<f32> = (0..WINDOW_HOURS).map(|h| precip_norm[(h, 0)]).collect();

    let mut daily_q = [0f32; WINDOW_DAYS];
    let mut target = [0f32; WINDOW_HOURS];
    for (i, r) in w.iter().enumerate() {
        daily_q[i] = r.daily_q_m3s;
        target[i * 24..i * 24 + 24].copy_from_slice(&r.target_hourly_m3s);
    }

    let daily_q_t = Tensor::<I, 2>::from_data(TensorData::new(daily_q.to_vec(), [WINDOW_DAYS, 1]), &device);
    let precip_t = Tensor::<I, 2>::from_data(TensorData::new(precip_feat, [WINDOW_HOURS, 1]), &device);
    let pred = head.forward(daily_q_t, precip_t, WINDOW_HOURS);
    let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();

    let mut f = std::fs::File::create(&cli.output)?;
    use std::io::Write;
    writeln!(f, "staid,hour,daily_input,real_hourly,head_pred,climatology_pred,precip_raw_mm_hr")?;
    for h in 0..WINDOW_HOURS {
        let d = h / 24;
        let hh = h % 24;
        let clim_pred = clim_shape_24[hh] * 24.0 * daily_q[d];
        writeln!(f, "{staid},{h},{},{},{},{},{}", daily_q[d], target[h], pred_vec[h], clim_pred, precip_col[h])?;
    }
    println!("wrote {}", cli.output);
    Ok(())
}
