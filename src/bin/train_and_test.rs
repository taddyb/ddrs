//! Full train + test entrypoint. Sequences SP-4's `train()` (Phase 1) and
//! SP-5's `evaluate()` (Phase 2) in one process. Mirrors DDR's
//! `~/projects/ddr/scripts/train_and_test.py`.
//!
//! Usage:
//!   cargo run --release --bin train_and_test -- \
//!       --config config/merit_training.yaml \
//!       --checkpoint-dir output/saved_models \
//!       --output output/model_test.zarr
//!
//! Phase 1 (training):
//!   - Load config in Training mode (batch_size=64 gauges, rho=90 days,
//!     epochs=5 per `config/merit_training.yaml`).
//!   - Initialize MLP from `cfg.mlp` (Kaiming weights, zero biases).
//!   - Build Adam (eps=1e-8 PyTorch default; see optimizer.rs).
//!   - Run train() — writes one .mpk per mini-batch to --checkpoint-dir.
//!
//! Phase 2 (testing):
//!   - Reload config in Testing mode (overlays `testing:` block: 1995-2010,
//!     batch_size=15 days, rho=null).
//!   - Auto-discover the latest .mpk in --checkpoint-dir.
//!   - Load MLP from that checkpoint (now contains learned weights).
//!   - Run evaluate() with EvalParams::KanHead, write zarr, log NSE summary.
//!
//! Optional --max-mini-batches caps Phase 1 for smoke testing (default: full
//! training).

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::backend::Autodiff;
use burn_cuda::Cuda;
use clap::Parser;

use ddrs::config::{Config, ConfigMode};
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::nn::kan_head::{KanHead, KanHeadConfig};
use ddrs::training::bootstrap::bootstrap_head_and_state;
use ddrs::training::checkpoint::load_kan_head;
use ddrs::training::driver::train;
use ddrs::training::optimizer::build_adam;
use ddrs::training::{
    evaluate, write_predictions_zarr, EvalParams, ZarrAttrs,
};

#[derive(Parser, Debug)]
#[command(name = "train_and_test", about = "ddrs full train + test pipeline")]
struct Cli {
    #[arg(long)]
    config: PathBuf,

    /// Directory where Phase 1 writes per-mini-batch .mpk checkpoints
    /// and where Phase 2 looks for the latest one.
    #[arg(long)]
    checkpoint_dir: PathBuf,

    /// Output zarr path for Phase 2 predictions + observations.
    #[arg(long)]
    output: PathBuf,

    /// Days per chunk in Phase 2. Default 15 matches DDR's test config.
    #[arg(long, default_value_t = 15)]
    batch_size_days: usize,

    /// Cap on mini-batches in Phase 1 (for smoke testing). Default: full
    /// training per cfg.experiment.epochs * mini-batches-per-epoch.
    #[arg(long)]
    max_mini_batches: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "warning: `train_and_test` is deprecated and will be removed in 0.4. \
         use `ddrs run --workflow train-and-test` instead."
    );
    let cli = Cli::parse();
    std::fs::create_dir_all(&cli.checkpoint_dir)?;
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    type I = Cuda<f32, i32>;
    type AB = Autodiff<I>;

    // -----------------------------------------------------------------------
    // Phase 1: training
    // -----------------------------------------------------------------------
    let phase1_start = Instant::now();
    println!("=== Phase 1: training ===");
    let train_cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Training)?;
    // Config-selected CUDA ordinal (top-level `device:` key).
    let device = cubecl::cuda::CudaDevice::new(train_cfg.device);
    let train_dataset = MeritGagesDataset::open(&train_cfg)?;
    println!(
        "training: {} gauges, dates {} .. {}",
        train_dataset.len(),
        train_cfg.experiment.as_ref().unwrap().start_time,
        train_cfg.experiment.as_ref().unwrap().end_time
    );
    println!(
        "sparse_solver={:?} use_cuda_graphs={}",
        train_cfg.params.sparse_solver, train_cfg.params.use_cuda_graphs,
    );

    let head_section = train_cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = KanHeadConfig::new(
        head_section.input_var_names.clone(),
        head_section.learnable_parameters.clone(),
        train_cfg.seed,
    )
    .with_hidden_size(head_section.hidden_size)
    .with_num_hidden_layers(head_section.num_hidden_layers)
    .with_grid(head_section.grid)
    .with_k(head_section.k);

    let (_, mut state) = bootstrap_head_and_state::<I>(&train_cfg, &device);
    let mut optimizer = build_adam::<KanHead<AB>, AB>();

    train::<I>(
        &train_cfg,
        &train_dataset,
        &mut state,
        &mut optimizer,
        &device,
        &cli.checkpoint_dir,
        cli.max_mini_batches,
        None,
    )?;
    let phase1_elapsed = phase1_start.elapsed();
    println!(
        "Phase 1 complete in {:.2} min ({} epochs × ending at mini-batch {})",
        phase1_elapsed.as_secs_f32() / 60.0,
        state.epoch.saturating_sub(1),
        state.mini_batch
    );

    // Free Phase 1 GPU/CPU state before Phase 2 (the optimizer holds momentum
    // buffers ~3x MLP size; the MLP itself stays alive via the checkpoint).
    drop(optimizer);
    drop(state);
    drop(train_dataset);

    // -----------------------------------------------------------------------
    // Phase 2: testing
    // -----------------------------------------------------------------------
    let phase2_start = Instant::now();
    println!("=== Phase 2: testing ===");
    let test_cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)?;
    let test_dataset = MeritGagesDataset::open(&test_cfg)?;
    println!(
        "testing: {} gauges, dates {} .. {}",
        test_dataset.len(),
        test_cfg.experiment.as_ref().unwrap().start_time,
        test_cfg.experiment.as_ref().unwrap().end_time
    );
    println!(
        "sparse_solver={:?} use_cuda_graphs={}",
        test_cfg.params.sparse_solver, test_cfg.params.use_cuda_graphs,
    );

    let latest_ckpt = find_latest_mpk(&cli.checkpoint_dir)?;
    println!("loading checkpoint: {}", latest_ckpt.display());

    let head_template: KanHead<I> = head_cfg.init::<I>(&device);
    let head = load_kan_head::<I>(&latest_ckpt, head_template, &device)?;

    let output = evaluate::<I>(
        &test_cfg,
        &test_dataset,
        EvalParams::KanHead(&head),
        &device,
        cli.batch_size_days,
    )?;
    let phase2_elapsed = phase2_start.elapsed();
    println!("Phase 2 complete in {:.2} min", phase2_elapsed.as_secs_f32() / 60.0);

    // Write the zarr.
    let exp = test_cfg.experiment.as_ref().unwrap();
    let gages_csv_path = test_cfg.data_sources.as_ref().unwrap().gages.clone();
    write_predictions_zarr(
        &cli.output,
        &output,
        ZarrAttrs {
            start_time: &exp.start_time,
            end_time: &exp.end_time,
            version: env!("CARGO_PKG_VERSION"),
            evaluation_basins_file: &gages_csv_path,
            model_label: &latest_ckpt.display().to_string(),
        },
    )?;

    // Metrics summary. Per-gauge mean is misleading on right-skewed NSE
    // distributions (a few bad gauges drag the mean); only median is reported.
    let median = |xs: &[f32]| -> f32 {
        let mut v: Vec<f32> = xs.iter().copied().filter(|x| x.is_finite()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() { f32::NAN } else { v[v.len() / 2] }
    };
    let median_nse = median(&output.metrics.nse);
    let median_kge = median(&output.metrics.kge);
    let n_finite_nse = output.metrics.nse.iter().filter(|v| v.is_finite()).count();
    println!("wrote {}", cli.output.display());
    println!(
        "gauges with finite NSE: {} / {}",
        n_finite_nse,
        output.metrics.nse.len()
    );
    println!("median NSE (finite only): {median_nse:.4}");
    println!("median KGE (finite only): {median_kge:.4}");
    println!(
        "Total time: {:.2} min",
        (phase1_elapsed + phase2_elapsed).as_secs_f32() / 60.0
    );

    Ok(())
}

/// Find the most-recently-modified `.mpk` file under `dir`. Returns the path
/// WITHOUT the `.mpk` suffix so it can be passed straight to `load_kan_head` (which
/// re-appends `.mpk` via `CompactRecorder::set_extension`).
fn find_latest_mpk(dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut latest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("mpk") {
            continue;
        }
        let mtime = entry.metadata()?.modified()?;
        if latest.as_ref().map_or(true, |(t, _)| mtime > *t) {
            latest = Some((mtime, path));
        }
    }
    let (_, p) = latest.ok_or_else(|| {
        format!(
            "no .mpk checkpoints found in {} — did Phase 1 produce any?",
            dir.display()
        )
    })?;
    // Strip .mpk so CompactRecorder's set_extension produces the right path.
    Ok(p.with_extension(""))
}
