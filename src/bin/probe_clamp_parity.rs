//! Probe: does the fused MC gradient have the same exact-zero pattern as
//! plain Burn autodiff, on a real routed trajectory?
//!
//! It builds the routing engine as `probe_courant` does (KAN head, then
//! `setup_inputs`, then hot start) and calls
//! `ddrs::routing::clamp_parity::run_clamp_parity`. CPU only. It prints the
//! report and exits 1 when any zero-set mismatch is found.
//!
//! ```bash
//! cargo run --release --bin probe_clamp_parity -- \
//!   --config .ddrs/runs/<id>/config.yaml \
//!   --checkpoint .ddrs/runs/<id>/checkpoints/epoch_30_mb_0 \
//!   --gauges 1 --rho 90 --steps 2000 --sample-every 20 --workspace .ddrs
//! ```

use std::path::PathBuf;

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use chrono::Duration;
use clap::Parser;

use ddrs::config::{Config, ConfigMode, SparseSolver};
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::data::dates::RhoWindow;
use ddrs::routing::clamp_ledger::ClampSite;
use ddrs::routing::clamp_parity::{run_clamp_parity, PARENTS};
use ddrs::routing::denormalize;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::training::bootstrap_head_and_state;

#[derive(Parser, Debug)]
#[command(
    name = "probe_clamp_parity",
    about = "Compare the fused MC gradient zero pattern with Burn autodiff on a routed trajectory"
)]
struct Cli {
    /// Training YAML (a run's `config.yaml` or `ddrs.yaml`). Loaded in TRAINING mode.
    #[arg(long)]
    config: PathBuf,

    /// Checkpoint DIRECTORY (`.../epoch_E_mb_M`) to take the KAN head from.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Gauges in the mini-batch (the first N gauges of the dataset).
    #[arg(long, default_value_t = 64)]
    gauges: usize,

    /// First day index of the rho window into the experiment time axis.
    #[arg(long, default_value_t = 0)]
    start_day: usize,

    /// Rho window length in days. Hourly steps available = (rho - 1) * 24.
    #[arg(long, default_value_t = 30)]
    rho: usize,

    /// Hourly timesteps to route. Capped by the window.
    #[arg(long, default_value_t = 500)]
    steps: usize,

    /// Compare gradients every Nth timestep.
    #[arg(long, default_value_t = 20)]
    sample_every: usize,

    /// Workspace root. Kept for the same command line as `probe_courant`;
    /// unused when the config names explicit adjacency stores.
    #[arg(long, default_value = ".ddrs")]
    workspace: PathBuf,
}

type R<T> = Result<T, Box<dyn std::error::Error>>;
type I = burn::backend::NdArray<f32>;

fn main() -> R<()> {
    let cli = Cli::parse();
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();

    let mut cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Training)
        .map_err(|e| format!("load config: {e}"))?;
    cfg.params.sparse_solver = SparseSolver::Cpu;
    cfg.params.use_cuda_graphs = false;
    if cfg.params.ddr_match {
        return Err("this probe requires params.ddr_match: false".into());
    }
    if cfg.params.use_reservoirs {
        return Err("this probe does not support params.use_reservoirs: true (it routes every reach as a channel)".into());
    }
    if cfg.stage_routing_on() {
        return Err("this probe does not support params.stage_routing (it probes the Muskingum-Cunge gradient)".into());
    }
    if let (Some(exp), Some(ckpt)) = (cfg.experiment.as_mut(), cli.checkpoint.clone()) {
        exp.checkpoint = Some(ckpt);
    }

    let dataset = MeritGagesDataset::open(&cfg).map_err(|e| format!("open dataset: {e}"))?;
    let axis = dataset.time_axis().clone();
    let staids: Vec<_> = dataset.staids().iter().take(cli.gauges).cloned().collect();
    let window = RhoWindow {
        start_day_idx: cli.start_day,
        rho_days: cli.rho,
        window_start: axis.start + Duration::days(cli.start_day as i64),
    };
    eprintln!(
        "batch: {} gauges {:?}, window {} .. +{} d (day idx {})",
        staids.len(),
        staids,
        window.window_start,
        cli.rho,
        cli.start_day
    );

    let batch = dataset
        .collate(&staids, &window)
        .map_err(|e| format!("collate: {e}"))?;
    let tensors = batch.to_tensors::<Autodiff<I>>(&device);
    let n_active = tensors.adjacency.n;
    eprintln!("network: {n_active} reaches");

    let (head, _state, _optim) = bootstrap_head_and_state::<I>(&cfg, &device)
        .map_err(|e| format!("bootstrap head: {e}"))?;

    let params_map = ddrs::training::forward::gather_params_to_subreaches(
        head.forward(tensors.spatial_attributes.clone()),
        tensors.adjacency.parent_offset.as_ref(),
        n_active,
        &device,
    );
    let n_param = params_map.get("n").expect("head missing n").clone();
    let q_param = params_map.get("q_spatial").expect("head missing q_spatial").clone();
    let p_param = params_map.get("p_spatial").cloned();
    let x_storage: Tensor<Autodiff<I>, 1> = match params_map.get("x_storage") {
        Some(x) => denormalize(
            x.clone(),
            cfg.params.parameter_ranges.x_storage,
            cfg.params.log_space_parameters.iter().any(|s| s == "x_storage"),
        ),
        None => Tensor::full([n_active], 0.3_f32, &device),
    };

    let n_hourly = tensors.q_prime.dims()[0];
    let q_prime_hourly = match &head.disagg {
        Some(d) => d.forward(
            tensors.q_prime_daily.clone(),
            tensors.precip_hourly.clone(),
            n_hourly,
        ),
        None => tensors.q_prime.clone(),
    };

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    engine.setup_inputs(
        RoutingInputs {
            adjacency: tensors.adjacency.clone(),
            x_storage: x_storage.clone(),
        },
        q_prime_hourly.clone(),
        SpatialParameters {
            n: n_param.clone(),
            q_spatial: q_param.clone(),
            p_spatial: p_param.clone(),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        },
        false,
        tensors.initial_state.clone(),
    );
    let inp = engine.probe_inputs();

    let t0 = std::time::Instant::now();
    let rep = run_clamp_parity::<I>(&cfg, &inp, cli.steps, cli.sample_every);
    let secs = t0.elapsed().as_secs_f64();

    println!("geometry_form: {:?}", cfg.params.geometry_form);
    println!("network {} reaches, {} sampled steps, {:.1}s", rep.n_reaches, rep.n_sampled_steps, secs);
    println!("max_q_rel_diff: {:e}", rep.max_q_rel_diff);
    println!("{:<12} {:>10} {:>10} {:>14}", "parent", "compared", "mismatch", "value_outliers");
    for p in PARENTS {
        println!("{:<12} {:>10} {:>10} {:>14}", p, rep.compared[p], rep.mismatches[p], rep.value_outliers[p]);
    }
    println!("active (step, reach) entries per clamp site:");
    for s in ClampSite::ALL {
        println!("  {:<20} {}", s.name(), rep.active.get(&s).copied().unwrap_or(0));
    }
    if !rep.first_mismatches.is_empty() {
        println!("first mismatches:");
        for m in &rep.first_mismatches {
            println!("  {m}");
        }
    }
    let total = rep.total_mismatches();
    println!("total zero-set mismatches: {total}");
    if total > 0 {
        std::process::exit(1);
    }
    Ok(())
}
