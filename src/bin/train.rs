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

use burn::backend::Autodiff;
use burn::tensor::backend::BackendTypes;
use burn_cuda::Cuda;
use clap::Parser;

use ddrs::config::{Config, ConfigMode};
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::nn::kan_head::KanHead;
use ddrs::training::bootstrap::bootstrap_head_and_state;
use ddrs::training::driver::train;
use ddrs::training::optimizer::build_adam;

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "warning: `train` is deprecated and will be removed in 0.4. \
         use `ddrs run --workflow train` instead."
    );
    let cli = Cli::parse();
    std::fs::create_dir_all(&cli.checkpoint_dir)?;

    type I = Cuda<f32, i32>;
    type AB = Autodiff<I>;
    let device = <I as BackendTypes>::Device::default();

    let start = Instant::now();
    println!("=== Training ===");
    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Training)?;
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

    let (_, mut state) = bootstrap_head_and_state::<I>(&cfg, &device);
    let mut optimizer = build_adam::<KanHead<AB>, AB>();

    train::<I>(
        &cfg,
        &dataset,
        &mut state,
        &mut optimizer,
        &device,
        &cli.checkpoint_dir,
        cli.max_mini_batches,
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
