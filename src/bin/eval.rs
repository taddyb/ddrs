//! Test-phase entrypoint. Loads a config + MLP checkpoint (or runs with
//! frozen scalar params for dev), runs `evaluate()`, writes the
//! DDR-compatible predictions zarr, and logs a metrics summary.
//!
//! Usage:
//!   cargo run --release --bin eval -- \
//!       --config config/merit_training.yaml \
//!       --checkpoint output/saved_models/epoch_5 \
//!       --output output/model_test.zarr \
//!       --batch-size-days 15
//!
//! With --frozen, --checkpoint is optional (V4 dev path):
//!   cargo run --release --bin eval -- \
//!       --config config/merit_training.yaml \
//!       --frozen \
//!       --output output/v4_test.zarr
//!
//! NOT for distribution — the MLP architecture mirrors DDR's KAN at the I/O
//! contract level but the internal weights are not transferable from DDR
//! .pt files. Use a ddrs-trained .mpk checkpoint only.

use std::path::PathBuf;

use burn::tensor::backend::BackendTypes;
use burn_cuda::Cuda;
use clap::Parser;

use ddrs::config::{Config, ConfigMode};
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::data::TestWindow;
use ddrs::nn::mlp::{Mlp, MlpConfig};
use ddrs::training::checkpoint::load_mlp;
use ddrs::training::{evaluate, write_predictions_zarr, EvalParams, FrozenParams, ZarrAttrs};

#[derive(Parser, Debug)]
#[command(name = "eval", about = "ddrs test-phase evaluation")]
struct Cli {
    #[arg(long)]
    config: PathBuf,

    /// MLP checkpoint base path (no .mpk suffix). Required unless --frozen.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Output zarr path.
    #[arg(long)]
    output: PathBuf,

    /// Days per chunk. Default 15 matches DDR's test config.
    #[arg(long, default_value_t = 15)]
    batch_size_days: usize,

    /// Use FROZEN_N/Q_SPATIAL/P_SPATIAL constants instead of an MLP.
    #[arg(long)]
    frozen: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if !cli.frozen && cli.checkpoint.is_none() {
        eprintln!("--checkpoint is required unless --frozen is set");
        std::process::exit(2);
    }

    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)?;
    let dataset = MeritGagesDataset::open(&cfg)?;
    println!(
        "sparse_solver={:?} use_cuda_graphs={}",
        cfg.params.sparse_solver, cfg.params.use_cuda_graphs,
    );

    type I = Cuda<f32, i32>;
    let device = <I as BackendTypes>::Device::default();

    // Seed the backend RNG so MLP template init is deterministic across runs.
    // (Per BURN 0.21 docs at burn-backend-0.21.0/src/backend/base.rs:141 —
    // ensures single-threaded determinism; CUDA atomic-add in scatter_add
    // is still non-deterministic, but at least the load_record template
    // doesn't drift between runs.)
    <I as burn::tensor::backend::Backend>::seed(&device, cfg.seed);

    let output = if cli.frozen {
        // Probe with a 1-day window to size FrozenParams (cheap — forces the
        // static-network cache to build but only reads 1 day of streamflow).
        let axis = dataset.time_axis().clone();
        let probe_window = TestWindow::new(&axis, 0, 1);
        let probe = dataset.collate_window(&probe_window)?;
        let frozen = FrozenParams::constant(probe.adjacency.n);
        evaluate::<I>(&cfg, &dataset, EvalParams::Frozen(&frozen), &device, cli.batch_size_days)?
    } else {
        let mlp_section = cfg.mlp.as_ref().expect("mlp config required for MLP eval");
        let mlp_cfg = MlpConfig::new(
            mlp_section.input_var_names.clone(),
            mlp_section.learnable_parameters.clone(),
        )
        .with_hidden_size(mlp_section.hidden_size)
        .with_num_hidden_layers(mlp_section.num_hidden_layers);
        let mlp_template: Mlp<I> = mlp_cfg.init::<I>(&device);
        let mlp = load_mlp::<I>(cli.checkpoint.as_ref().unwrap(), mlp_template, &device)?;
        evaluate::<I>(&cfg, &dataset, EvalParams::Mlp(&mlp), &device, cli.batch_size_days)?
    };

    // Write the zarr.
    let exp = cfg.experiment.as_ref().unwrap();
    let model_label = match &cli.checkpoint {
        Some(p) => p.display().to_string(),
        None => "frozen".to_string(),
    };
    let gages_csv_path = cfg.data_sources.as_ref().unwrap().gages.clone();
    write_predictions_zarr(
        &cli.output,
        &output,
        ZarrAttrs {
            start_time: &exp.start_time,
            end_time: &exp.end_time,
            version: env!("CARGO_PKG_VERSION"),
            evaluation_basins_file: &gages_csv_path,
            model_label: &model_label,
        },
    )?;

    // Metrics summary.
    let nse_clean: Vec<f32> = output
        .metrics
        .nse
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let mean_nse = nse_clean.iter().sum::<f32>() / (nse_clean.len() as f32).max(1.0);
    println!("wrote {}", cli.output.display());
    println!(
        "gauges with finite NSE: {} / {}",
        nse_clean.len(),
        output.metrics.nse.len()
    );
    println!("mean NSE (finite only): {mean_nse:.4}");

    Ok(())
}
