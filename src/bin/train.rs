//! Training-only entrypoint. Phase 1 of `train_and_test` with no Phase 2.
//! Use for benchmarking / profiling the training loop in isolation.
//!
//! Usage:
//!   cargo run --release --bin train -- \
//!       --config config/merit_training.yaml \
//!       --checkpoint-dir output/saved_models
//!
//! Optional `--max-mini-batches N` caps the per-epoch inner loop for
//! quick profiling (still runs all configured epochs, just fewer batches
//! per epoch).
//!
//! For nsys profiling:
//!   nsys profile --trace=cuda --sample=none --cpuctxsw=none \
//!       --output=PROFILE --force-overwrite=true \
//!       target/release/train --config ... --checkpoint-dir ... --max-mini-batches 3

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use burn::backend::Autodiff;
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};

use ddrs::config::{Config, ConfigMode, SparseSolver};
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::training::bootstrap::bootstrap_head_and_state;
use ddrs::training::driver::train;

#[derive(Parser, Debug)]
#[command(name = "train", about = "ddrs training-only entrypoint (no test phase)")]
struct Cli {
    #[arg(long)]
    config: PathBuf,

    /// Directory for per-mini-batch .mpk checkpoints.
    #[arg(long)]
    checkpoint_dir: PathBuf,

    /// Cap on mini-batches per epoch (for profiling / smoke tests).
    /// Default: full per-epoch sweep per cfg.experiment.batch_size.
    #[arg(long)]
    max_mini_batches: Option<usize>,

    /// Backend: "cpu" (NdArray, deterministic; forces sparse_solver=cpu) or "cuda".
    #[arg(long, default_value = "cuda")]
    backend: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "warning: `train` is deprecated and will be removed in 0.4. \
         use `ddrs run --workflow train` instead."
    );
    let cli = Cli::parse();
    std::fs::create_dir_all(&cli.checkpoint_dir)?;

    let mut cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Training)?;
    match cli.backend.as_str() {
        "cpu" => {
            type I = burn::backend::NdArray<f32>;
            let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
            cfg.params.sparse_solver = SparseSolver::Cpu;
            cfg.params.use_cuda_graphs = false;
            eprintln!("backend: cpu (NdArray; sparse_solver forced to cpu)");
            run::<I>(cfg, cli, device)
        }
        "cuda" => {
            type I = burn_cuda::Cuda<f32, i32>;
            let device = cubecl::cuda::CudaDevice::new(cfg.device);
            run::<I>(cfg, cli, device)
        }
        other => Err(format!("unknown --backend {other} (expected \"cpu\" or \"cuda\")").into()),
    }
}

fn run<I>(cfg: Config, cli: Cli, device: I::Device) -> Result<(), Box<dyn std::error::Error>>
where
    I: Backend,
    Autodiff<I>: AutodiffBackend<InnerBackend = I>,
    I: BackendTypes<Device = <Autodiff<I> as BackendTypes>::Device>,
{
    let start = Instant::now();
    println!("=== Training ===");
    let dataset = MeritGagesDataset::open(&cfg)?;
    println!(
        "training: {} gauges, dates {} .. {}, epochs={}",
        dataset.len(),
        cfg.experiment.as_ref().unwrap().start_time,
        cfg.experiment.as_ref().unwrap().end_time,
        cfg.experiment.as_ref().unwrap().epochs,
    );
    println!(
        "sparse_solver={:?} use_cuda_graphs={}",
        cfg.params.sparse_solver, cfg.params.use_cuda_graphs,
    );

    let (_, mut state, mut optimizer) = bootstrap_head_and_state::<I>(&cfg, &device)?;

    train::<I>(
        &cfg,
        &dataset,
        &mut state,
        &mut optimizer,
        &device,
        &cli.checkpoint_dir,
        cli.max_mini_batches,
        None,
    )?;
    let elapsed = start.elapsed();
    println!(
        "Training complete in {:.2} min ({} epochs × ending at mini-batch {})",
        elapsed.as_secs_f32() / 60.0,
        state.epoch.saturating_sub(1),
        state.mini_batch
    );

    Ok(())
}
