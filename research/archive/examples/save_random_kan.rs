//! Throwaway smoke helper: build a `KanHead` from the YAML's `kan_head`
//! section (random init) and save it as a `.mpk` checkpoint. Used to verify
//! `dump_parameters` end-to-end when no real trained checkpoint is on disk.
//!
//! Not for distribution — the resulting checkpoint has random weights and
//! produces meaningless parameter predictions.
//!
//!   cargo run --release --example save_random_kan -- \
//!       --config config/merit_training.yaml \
//!       --output /tmp/random_kan

use std::path::PathBuf;

use burn::tensor::backend::{Backend, BackendTypes};
use burn_cuda::Cuda;
use clap::Parser;

use ddrs::config::{Config, ConfigMode};
use ddrs::nn::{KanHead, KanHeadConfig};
use ddrs::training::checkpoint::save_kan_head;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Training)?;
    let head_cfg = cfg.kan_head.as_ref().expect("kan_head section required");

    type I = Cuda<f32, i32>;
    let device = <I as BackendTypes>::Device::default();
    <I as Backend>::seed(&device, cfg.seed);

    let head: KanHead<I> = KanHeadConfig::new(
        head_cfg.input_var_names.clone(),
        head_cfg.learnable_parameters.clone(),
        cfg.seed,
    )
    .with_hidden_size(head_cfg.hidden_size)
    .with_num_hidden_layers(head_cfg.num_hidden_layers)
    .with_grid(head_cfg.grid)
    .with_k(head_cfg.k)
    .init::<I>(&device);

    save_kan_head(&cli.output, &head)?;
    println!("wrote random KAN to {}.mpk", cli.output.display());
    Ok(())
}
