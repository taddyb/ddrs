//! KAN-interpretability sensitivity sweep against a TRAINED checkpoint's
//! disaggregation head (not a synthetic one — see
//! `examples/kan_sensitivity_sweep.rs` for the synthetic-mechanism version).
//! Loads the real `.mpk` disagg weights, sweeps precip intensity at a fixed
//! hour with daily Q' held fixed, and records the resulting 24-hour
//! disaggregated shape. See
//! `.claude/skills/ddrs-eval-plots/references/kan_interpretability.md`.
//!
//!   cargo run --release --example kan_disagg_trained_sensitivity -- \
//!       --config <run_dir>/config.yaml \
//!       --checkpoint <run_dir>/checkpoints/epoch_5_mb_9/head \
//!       --output <run_dir>/plots/precip_sensitivity_trained.csv

use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use clap::Parser;

use ddrs::config::{Config, ConfigMode};
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
}

fn run_one_day(
    disagg: &ddrs::nn::DisaggHead<I>,
    daily_q: f32,
    precip_at_hour: usize,
    precip_intensity: f32,
) -> Vec<f32> {
    let device = Default::default();
    let q = Tensor::<I, 1>::from_data(TensorData::new(vec![daily_q], [1]), &device).reshape([1, 1]);
    let mut p = vec![0.1f32; 24];
    p[precip_at_hour] = precip_intensity;
    let precip = Tensor::<I, 1>::from_data(TensorData::new(p, [24]), &device).reshape([24, 1]); // (n_hourly=24, N=1)
    let hourly = disagg.forward(q, precip, 24); // (24, 1)
    hourly.into_data().to_vec().unwrap()
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

    let device = Default::default();
    <I as Backend>::seed(&device, cfg.seed);
    let head_template: KanHead<I> = ddrs::config::kan_config(head_cfg, cfg.seed).init::<I>(&device);
    let head = load_kan_head::<I>(&cli.checkpoint, head_template, &device)?;
    let disagg = head.disagg.expect("checkpoint's kan_head has no disaggregation block");

    // Representative daily Q' — arbitrary but fixed; only the SHAPE (not
    // scale) of the response is being inspected here.
    let daily_q = 10.0f32;
    let intensities = [0.1f32, 1.0, 2.0, 4.0, 8.0, 16.0];
    let mut rows: Vec<(f32, usize, f32)> = Vec::new();
    for &intensity in &intensities {
        let out = run_one_day(&disagg, daily_q, 12, intensity);
        for (h, &v) in out.iter().enumerate() {
            rows.push((intensity, h, v));
        }
    }

    let mut f = std::fs::File::create(&cli.output)?;
    use std::io::Write;
    writeln!(f, "precip_intensity,hour,hourly_value")?;
    for (intensity, hour, val) in &rows {
        writeln!(f, "{intensity},{hour},{val}")?;
    }
    println!("wrote {} ({} rows)", cli.output.display(), rows.len());

    // Diagnostic 2: does the response's PEAK LOCATION move with the swept
    // hour, or is it pinned regardless of where the storm is injected? Fixed
    // high intensity, vary WHICH hour gets it.
    let hour_output = cli.output.with_file_name(format!(
        "{}_hour_position.csv",
        cli.output.file_stem().unwrap().to_string_lossy()
    ));
    let swept_hours = [0usize, 3, 6, 9, 12, 15, 18, 21];
    let mut hour_rows: Vec<(usize, usize, f32)> = Vec::new(); // (swept_hour, output_hour, value)
    for &sh in &swept_hours {
        let out = run_one_day(&disagg, daily_q, sh, 8.0);
        for (h, &v) in out.iter().enumerate() {
            hour_rows.push((sh, h, v));
        }
    }
    let mut fh = std::fs::File::create(&hour_output)?;
    writeln!(fh, "swept_hour,hour,hourly_value")?;
    for (sh, h, v) in &hour_rows {
        writeln!(fh, "{sh},{h},{v}")?;
    }
    println!("wrote {} ({} rows)", hour_output.display(), hour_rows.len());
    Ok(())
}
