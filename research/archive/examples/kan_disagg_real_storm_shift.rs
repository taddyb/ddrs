//! Real-storm circular-shift test: does the trained disagg head's output
//! peak actually TRACK precip timing when given a realistic (bursty, real
//! AORC) precip shape, rather than the synthetic flat-background+single-hour
//! spike used in `kan_disagg_trained_sensitivity.rs`? Takes ONE real day's
//! actual 24h precip vector for a real gauge (kept in-distribution — same
//! amplitude, same burstiness, just rotated), circularly shifts it by k
//! hours, holds that day's REAL daily_q fixed, and re-runs the trained
//! disagg head for each shift. If the output peak follows the shift, the
//! head IS precip-timing-sensitive under realistic inputs (the earlier
//! synthetic sweep was simply OOD). If the peak stays pinned regardless of
//! shift, the earlier finding (fixed per-reach template, precip timing
//! doesn't matter) is real, not a probe artifact.
//!
//!   cargo run --release --example kan_disagg_real_storm_shift -- \
//!       --config <run_dir>/config.yaml \
//!       --checkpoint <run_dir>/checkpoints/epoch_5_mb_35/head \
//!       --output <run_dir>/plots/real_storm_shift.csv \
//!       --staid 14301000 --day 9

use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Tensor, TensorData};
use clap::Parser;

use ddrs::config::{Config, ConfigMode};
use ddrs::data::{MeritGagesDataset, TestWindow};
use ddrs::nn::kan_head::KanHead;
use ddrs::training::load_kan_head;

type I = NdArray<f32>;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 10)]
    n_days: usize,
    #[arg(long, default_value_t = 3)]
    n_gauges: usize,
    /// Which gauge STAID (must be one of the first n_gauges) to use as the
    /// real-storm source.
    #[arg(long)]
    staid: String,
    /// Which day index (0-based within the window) supplies the real storm.
    #[arg(long)]
    day: usize,
}

fn to_tensor(data: Vec<f32>, shape: [usize; 2], device: &<I as BackendTypes>::Device) -> Tensor<I, 2> {
    Tensor::<I, 2>::from_data(TensorData::new(data, shape), device)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)?;
    let head_cfg = cfg
        .kan_head
        .as_ref()
        .expect("kan_head section required for trained-KAN inference");

    let dataset = MeritGagesDataset::open(&cfg)?;
    let axis = dataset.time_axis().clone();
    let window = TestWindow::new(&axis, 0, cli.n_days);
    let batch = dataset.collate_window(&window)?;

    let n_gauges = cli.n_gauges.min(batch.gauge_staids.len());
    let mut col = None;
    for g in 0..n_gauges {
        if batch.gauge_staids[g].to_string() == cli.staid {
            col = batch.outflow_idx[g].first().copied();
            break;
        }
    }
    let col = col.unwrap_or_else(|| panic!("staid {} not found in first {n_gauges} gauges", cli.staid));

    // Real 24h precip vector + real daily_q for the chosen (gauge, day).
    let d = cli.day;
    let real_precip: Vec<f32> = (0..24).map(|h| batch.precip_hourly[(d * 24 + h, col)]).collect();
    let real_daily_q = batch.q_prime_daily[(d, col)];
    println!(
        "real precip (normalized) for {} day {}: min={:.6} max={:.6} sum={:.6}, daily_q={:.4}",
        cli.staid,
        d,
        real_precip.iter().cloned().fold(f32::INFINITY, f32::min),
        real_precip.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        real_precip.iter().sum::<f32>(),
        real_daily_q
    );

    let device = Default::default();
    <I as Backend>::seed(&device, cfg.seed);
    let head_template: KanHead<I> = ddrs::config::kan_config(head_cfg, cfg.seed).init::<I>(&device);
    let head = load_kan_head::<I>(&cli.checkpoint, head_template, &device)?;
    let disagg = head.disagg.expect("checkpoint's kan_head has no disaggregation block");

    let shifts = [0usize, 3, 6, 9, 12, 15, 18, 21];
    let mut rows: Vec<(usize, usize, f32, f32)> = Vec::new(); // (shift, hour, hourly_value, precip_used)
    for &shift in &shifts {
        // Circular shift: shifted[h] = real_precip[(h - shift) mod 24], so the
        // storm shape moves FORWARD by `shift` hours.
        let shifted: Vec<f32> = (0..24).map(|h| real_precip[(h + 24 - shift) % 24]).collect();
        let q = to_tensor(vec![real_daily_q], [1, 1], &device);
        let precip = to_tensor(shifted.clone(), [24, 1], &device);
        let hourly = disagg.forward(q, precip, 24);
        let hourly_vec: Vec<f32> = hourly.into_data().to_vec().unwrap();
        for (h, &v) in hourly_vec.iter().enumerate() {
            rows.push((shift, h, v, shifted[h]));
        }
    }

    let mut f = std::fs::File::create(&cli.output)?;
    use std::io::Write;
    writeln!(f, "shift,hour,hourly_value,precip_used")?;
    for (shift, hour, val, p) in &rows {
        writeln!(f, "{shift},{hour},{val},{p}")?;
    }
    println!("wrote {} ({} rows)", cli.output.display(), rows.len());
    Ok(())
}
