//! KAN-interpretability sensitivity sweep: how does the disagg KAN's learned
//! within-day shape respond to precip intensity at a single hour, holding
//! everything else fixed? A black-box partial-dependence-style sweep — needs
//! only forward-pass evaluation, no access to rskan's internal spline
//! coefficients. See `.claude/skills/ddrs-eval-plots/references/kan_interpretability.md`.
//!
//!   cargo run --release --example kan_sensitivity_sweep -- \
//!       --output output/disagg_verification/precip_sensitivity.csv

use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use clap::Parser;
use ndarray::Array2;

use ddrs::nn::{DisaggHead, DisaggHeadConfig};

type I = NdArray<f32>;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "output/disagg_verification/precip_sensitivity.csv")]
    output: PathBuf,
}

fn build_head() -> DisaggHead<I> {
    let device = Default::default();
    let cfg = DisaggHeadConfig::new(11);
    let mut head = cfg.init::<I>(&device);
    // Non-degenerate output map so the sweep produces a visible response
    // (default init is intentionally MILD, per the module docs — a stronger
    // pattern here just makes the sweep's shape response easier to see).
    head.output.weight = ddrs::nn::init::to_param_weight::<I>(
        Array2::<f32>::from_shape_fn((cfg.hidden_size, 24), |(i, j)| {
            if i % 24 == j { 2.0 } else { 0.05 * ((i + j) as f32).cos() }
        }),
        &device,
    );
    head
}

fn run_one_day(head: &DisaggHead<I>, daily_q: f32, precip_at_hour: usize, precip_intensity: f32) -> Vec<f32> {
    let device = Default::default();
    let q = Tensor::<I, 1>::from_data(TensorData::new(vec![daily_q], [1]), &device).reshape([1, 1]);
    let mut p = vec![0.1f32; 24];
    p[precip_at_hour] = precip_intensity;
    let precip = Tensor::<I, 1>::from_data(TensorData::new(p, [24]), &device).reshape([24, 1]);
    let hourly = head.forward(q, precip, 24); // (24, 1)
    hourly.into_data().to_vec().unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let head = build_head();

    // Sweep 1: precip intensity at a fixed hour (hour 12), holding daily_q fixed.
    // Records the resulting 24-hour shape for each intensity -> shows how
    // the KAN's within-day redistribution responds to a stronger/weaker
    // storm at the same time of day.
    let intensities = [0.1f32, 1.0, 2.0, 4.0, 8.0, 16.0];
    let mut rows: Vec<(f32, usize, f32)> = Vec::new(); // (intensity, hour, hourly_value)
    for &intensity in &intensities {
        let out = run_one_day(&head, 10.0, 12, intensity);
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
    Ok(())
}
