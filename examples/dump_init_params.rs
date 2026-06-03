//! Sweep all CONUS MERIT reaches through a freshly-initialised DDRS
//! `KanHead` (no checkpoint, no training) and write per-COMID denormalised
//! parameters to a NetCDF. Used by the Layer 4 init-distribution
//! comparison against DDR.
//!
//! Run:
//!     cargo run --release --example dump_init_params -- \
//!         --config config/merit_training.yaml \
//!         --out    /tmp/kan_init_params_ddrs.nc
//!
//! Mirrors `scripts/dump_ddr_init_params.py` on the DDR side.

use std::path::PathBuf;

use burn::tensor::backend::BackendTypes;
use burn_cuda::Cuda;
use clap::Parser;

use ddrs::config::{Config, ConfigMode};

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "config/merit_training.yaml")]
    config: PathBuf,
    #[arg(long, default_value = "/tmp/kan_init_params_ddrs.nc")]
    out: PathBuf,
    #[arg(long, default_value_t = 50_000)]
    batch_size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Training)?;
    type I = Cuda<f32, i32>;
    let device = <I as BackendTypes>::Device::default();
    let n = ddrs::dump_parameters::dump_init::<I>(&cfg, &cli.out, cli.batch_size, &device)?;
    println!("wrote {n} reaches → {}", cli.out.display());
    Ok(())
}
