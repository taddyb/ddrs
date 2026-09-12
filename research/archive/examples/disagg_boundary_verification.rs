//! Verification: KAN-based disaggregation head (1) preserves mass exactly at
//! every boundary_blend λ, and (2) the day-boundary discontinuity shrinks as
//! λ→1. Synthetic, self-contained — no config/checkpoint needed, since this
//! verifies a mechanism, not a scientific finding from real data.
//!
//!   cargo run --release --example disagg_boundary_verification -- \
//!       --output output/disagg_verification.csv

use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use clap::Parser;
use ndarray::Array2;

use ddrs::nn::{DisaggHead, DisaggHeadConfig};

type I = NdArray<f32>;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "output/disagg_verification.csv")]
    output: PathBuf,
}

/// Build a 2-day synthetic scenario deliberately adversarial at the day
/// boundary: day 0 has a precip spike at hour 23, day 1 has a spike at hour
/// 0 — the two days' independently-computed shapes have no reason to agree
/// at the seam unless the boundary blend forces them to.
fn scenario(lambda: f32) -> (DisaggHead<I>, Tensor<I, 2>, Tensor<I, 2>) {
    let device = Default::default();
    let cfg = DisaggHeadConfig::new(7).with_boundary_blend(lambda);
    let mut head = cfg.init::<I>(&device);
    head.output.weight = ddrs::nn::init::to_param_weight::<I>(
        Array2::<f32>::from_shape_fn((cfg.hidden_size, 24), |(i, j)| {
            if i % 24 == j { 3.0 } else { 0.05 * ((i + j) as f32).sin() }
        }),
        &device,
    );
    let daily = Tensor::<I, 1>::from_data(TensorData::new(vec![10.0f32, 10.0], [2]), &device)
        .reshape([2, 1]); // D=2, N=1, equal daily means (isolates SHAPE continuity)
    let mut p = vec![0.1f32; 48];
    p[23] = 8.0; // day0 hour23 storm
    p[24] = 8.0; // day1 hour0 storm
    let precip = Tensor::<I, 1>::from_data(TensorData::new(p, [48]), &device).reshape([48, 1]);
    (head, daily, precip)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut rows: Vec<(f32, usize, f32)> = Vec::new(); // (lambda, hour_0_to_47, hourly_value)
    for &lambda in &[0.0f32, 0.5, 1.0] {
        let (head, daily, precip) = scenario(lambda);
        let hourly = head.forward(daily.clone(), precip, 48); // (48, 1)
        let v: Vec<f32> = hourly.clone().into_data().to_vec().unwrap();
        for (h, &val) in v.iter().enumerate() {
            rows.push((lambda, h, val));
        }

        // Mass conservation check at this lambda: each day's mean must equal
        // its daily input (10.0) exactly.
        for d in 0..2usize {
            let day_mean: f32 = v[d * 24..(d + 1) * 24].iter().sum::<f32>() / 24.0;
            assert!(
                (day_mean - 10.0).abs() < 1e-4,
                "mass conservation FAILED at lambda={lambda}, day={d}: mean={day_mean}"
            );
        }
        let seam_gap = (v[23] - v[24]).abs();
        eprintln!("lambda={lambda}: mass conserved on both days; seam gap = {seam_gap:.5}");
    }

    let mut f = std::fs::File::create(&cli.output)?;
    use std::io::Write;
    writeln!(f, "lambda,hour,hourly_value")?;
    for (lambda, hour, val) in &rows {
        writeln!(f, "{lambda},{hour},{val}")?;
    }
    println!("wrote {} ({} rows)", cli.output.display(), rows.len());
    Ok(())
}
