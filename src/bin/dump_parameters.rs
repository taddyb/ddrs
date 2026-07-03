//! Dump per-reach KAN parameter predictions for a trained ddrs checkpoint.
//!
//! Now a thin wrapper around `ddrs::dump_parameters::dump`. Library logic
//! lives at `src/dump_parameters.rs` for reuse by `ddrs run --plot`.

use std::path::PathBuf;

use clap::Parser;

use burn::tensor::backend::Backend;

use ddrs::config::{Config, ConfigMode, SparseSolver};

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

    /// Backend: "cpu" (NdArray, deterministic; forces sparse_solver=cpu) or "cuda".
    #[arg(long, default_value = "cuda")]
    backend: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let mut cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)?;
    match cli.backend.as_str() {
        "cpu" => {
            type I = burn::backend::NdArray<f32>;
            let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
            cfg.params.sparse_solver = SparseSolver::Cpu;
            cfg.params.use_cuda_graphs = false;
            eprintln!("backend: cpu (NdArray, deterministic; sparse_solver forced to cpu)");
            run::<I>(cfg, cli, device)
        }
        "cuda" => {
            type I = burn_cuda::Cuda<f32, i32>;
            // Config-selected CUDA ordinal (top-level `device:` key).
            let device = cubecl::cuda::CudaDevice::new(cfg.device);
            run::<I>(cfg, cli, device)
        }
        other => Err(format!("unknown --backend {other} (expected \"cpu\" or \"cuda\")").into()),
    }
}

fn run<I: Backend>(cfg: Config, cli: Cli, device: I::Device) -> Result<(), Box<dyn std::error::Error>> {
    let n = ddrs::dump_parameters::dump::<I>(&cfg, &cli.checkpoint, &cli.output, cli.batch_size, &device)?;
    println!("wrote {n} reaches → {}", cli.output.display());
    Ok(())
}
