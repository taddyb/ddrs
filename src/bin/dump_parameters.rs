//! Dump per-reach KAN parameter predictions for a trained ddrs checkpoint.
//!
//! Now a thin wrapper around `ddrs::dump_parameters::dump`. Library logic
//! lives at `src/dump_parameters.rs` for reuse by `ddrs run --plot`.

use std::path::PathBuf;

use burn::tensor::backend::BackendTypes;
use burn_cuda::Cuda;
use clap::Parser;

use ddrs::config::{Config, ConfigMode};

#[derive(Parser, Debug)]
#[command(name = "dump_parameters", about = "Dump trained KAN parameter predictions per CONUS reach")]
struct Cli {
    /// Training YAML (same one used at train time — supplies kan_head section,
    /// parameter_ranges, log_space_parameters, and data_sources.attributes).
    #[arg(long)]
    config: PathBuf,

    /// Trained KAN checkpoint base path (no `.mpk` suffix).
    #[arg(long)]
    checkpoint: PathBuf,

    /// Output NetCDF4 path.
    #[arg(long)]
    output: PathBuf,

    /// Reaches per KAN forward call. 50_000 matches DDR's
    /// `geometry_predictor.py` and fits comfortably on a 24 GB GPU.
    #[arg(long, default_value_t = 50_000)]
    batch_size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)?;
    type I = Cuda<f32, i32>;
    let device = <I as BackendTypes>::Device::default();
    let n = ddrs::dump_parameters::dump::<I>(&cfg, &cli.checkpoint, &cli.output, cli.batch_size, &device)?;
    println!("wrote {n} reaches → {}", cli.output.display());
    Ok(())
}
