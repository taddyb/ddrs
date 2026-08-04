//! Probe: measure what `params.enforce_positivity` (the S18'/S19' clamp)
//! actually does on REAL CONUS data, with no training.
//!
//! Reports, for `enforce_positivity` false and true over the SAME network,
//! parameters, forcing and hot-start:
//!
//!   1. `negative solves before clamp` — count / percentage, exact over every
//!      routed timestep (the same atomic counters the training log prints).
//!   2. `Cr = dt / K` percentiles, before and after the K floor, plus the
//!      fraction of reach-timesteps with `Cr_raw > 2` (where the floor bites).
//!   3. The effective Muskingum `X` percentiles.
//!   4. `k_musk / k_raw` — travel-time inflation, over the floored reaches.
//!
//! It builds the routing engine exactly as `training::forward` does (KAN head
//! -> setup_inputs -> hot-start) and then drives `forward_chain_inner` in the
//! same q_next-fed-back loop as `MuskingumCunge::forward`, so the numbers come
//! from the production kernel sequence rather than a re-derivation.
//!
//! ```bash
//! cargo run --release --bin probe_courant -- \
//!   --config ddrs.yaml --backend cuda \
//!   --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_18 \
//!   --gauges 64 --steps 500
//! ```

use std::path::PathBuf;

use burn::backend::Autodiff;
use burn::tensor::{backend::Backend, Tensor};
use clap::Parser;
use chrono::Duration;

use ddrs::config::{Config, ConfigMode, SparseSolver};
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::data::dates::RhoWindow;
use ddrs::routing::courant_probe::{run_courant_probe, CourantReport};
use ddrs::routing::denormalize;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::training::bootstrap_head_and_state;

#[derive(Parser, Debug)]
#[command(name = "probe_courant", about = "Measure Cr / X / K-floor / negative solves, clamp off vs on")]
struct Cli {
    /// Training YAML (e.g. `ddrs.yaml`). Loaded in TRAINING mode.
    #[arg(long)]
    config: PathBuf,

    /// Checkpoint DIRECTORY (`.../epoch_E_mb_M`) to take the KAN head from.
    /// Omit to probe the seed-initialised head.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Gauges in the mini-batch (matches `experiment.batch_size` semantics).
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

    /// Harvest the Cr / X / K distributions every Nth timestep. The solve
    /// counters are exact over ALL timesteps regardless.
    #[arg(long, default_value_t = 10)]
    sample_every: usize,

    /// "cpu" (NdArray) or "cuda".
    #[arg(long, default_value = "cuda")]
    backend: String,

    /// Optional CSV of the percentile table.
    #[arg(long)]
    output: Option<PathBuf>,
}

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> R<()> {
    let cli = Cli::parse();
    match cli.backend.as_str() {
        "cpu" => {
            type I = burn::backend::NdArray<f32>;
            let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
            run::<I>(cli, device)
        }
        "cuda" => {
            type I = burn_cuda::Cuda<f32, i32>;
            let device = cubecl::cuda::CudaDevice::new(0);
            run::<I>(cli, device)
        }
        other => Err(format!("unknown --backend {other}").into()),
    }
}

fn quantile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

struct Summary {
    label: &'static str,
    rep: CourantReport,
}

fn pct(v: &mut Vec<f32>) -> [f32; 5] {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    [
        quantile(v, 0.05),
        quantile(v, 0.25),
        quantile(v, 0.50),
        quantile(v, 0.75),
        quantile(v, 0.95),
    ]
}

fn run<I: Backend>(cli: Cli, device: I::Device) -> R<()>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let mut cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Training)
        .map_err(|e| format!("load config: {e}"))?;
    if cli.backend == "cpu" {
        cfg.params.sparse_solver = SparseSolver::Cpu;
    }
    // The graph path never enters `forward_chain_inner`, so it can neither be
    // counted nor probed.
    cfg.params.use_cuda_graphs = false;
    if cfg.params.ddr_match {
        return Err("this probe requires params.ddr_match: false (enforce_positivity is gated on it)".into());
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
        "batch: {} gauges, window {} .. +{} d (day idx {})",
        staids.len(),
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

    // Head exactly as the trainer bootstraps it (disagg warm start + resume).
    let (head, _state, _optim) = bootstrap_head_and_state::<I>(&cfg, &device)
        .map_err(|e| format!("bootstrap head: {e}"))?;

    let params_map = head.forward(tensors.spatial_attributes.clone());
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

    // Same forcing the trainer would route (disagg head when configured).
    let n_hourly = tensors.q_prime.dims()[0];
    let q_prime_hourly = match &head.disagg {
        Some(d) => d.forward(
            tensors.q_prime_daily.clone(),
            tensors.precip_hourly.clone(),
            n_hourly,
        ),
        None => tensors.q_prime.clone(),
    };

    let mut summaries: Vec<Summary> = Vec::new();
    for (label, enforce) in [("off", false), ("on", true)] {
        let mut cfg_v = cfg.clone();
        cfg_v.params.enforce_positivity = enforce;

        let mut engine = MuskingumCunge::<I>::new(cfg_v.clone(), device.clone());
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
            },
            false,
            tensors.initial_state.clone(),
        );
        let inp = engine.probe_inputs();
        eprintln!("--- routing with enforce_positivity: {enforce} ---");
        let rep = run_courant_probe::<I>(&cfg_v, &inp, cli.steps, cli.sample_every);
        eprintln!(
            "    {} steps, {} sampled, negative solves {}/{}",
            rep.n_steps, rep.n_sampled_steps, rep.neg_solves, rep.total_solves
        );
        summaries.push(Summary { label, rep });
    }

    // ---- Report -------------------------------------------------------------
    let mut csv = String::from("flag,metric,p5,p25,p50,p75,p95,extra\n");
    for s in &mut summaries {
        let r = &mut s.rep;
        let pc = 100.0 * r.neg_solves as f64 / r.total_solves.max(1) as f64;
        println!("\n================ enforce_positivity: {} ================", s.label);
        println!(
            "network {} reaches, {} routed timesteps, {} sampled steps ({} reach-timesteps sampled)",
            r.n_reaches,
            r.n_steps,
            r.n_sampled_steps,
            r.cr.len()
        );
        println!(
            "negative solves before clamp: {}/{} ({:.4}%)",
            r.neg_solves, r.total_solves, pc
        );

        let n_s = r.cr.len() as f64;
        let frac_cr_raw_gt2 = r.cr_raw.iter().filter(|&&v| v > 2.0).count() as f64 / n_s;
        let frac_floored = r.k_ratio.iter().filter(|&&v| v > 1.0 + 1e-6).count() as f64 / n_s;
        let mut floored_ratio: Vec<f32> =
            r.k_ratio.iter().copied().filter(|&v| v > 1.0 + 1e-6).collect();
        let cap_binds = r
            .x_eff
            .iter()
            .zip(r.x_cunge.iter())
            .filter(|(e, c)| **e < **c - 1e-6)
            .count() as f64
            / n_s;
        let neg_c1 = r.c1.iter().filter(|&&v| v < 0.0).count();
        let neg_c3 = r.c3.iter().filter(|&&v| v < 0.0).count();
        let min_c1 = r.c1.iter().copied().fold(f32::INFINITY, f32::min);
        let min_c3 = r.c3.iter().copied().fold(f32::INFINITY, f32::min);

        let q_cr_raw = pct(&mut r.cr_raw);
        let q_cr = pct(&mut r.cr);
        let q_x = pct(&mut r.x_eff);
        let q_xc = pct(&mut r.x_cunge);

        println!("\n{:<22} {:>9} {:>9} {:>9} {:>9} {:>9}", "", "p5", "p25", "p50", "p75", "p95");
        let row = |name: &str, q: [f32; 5]| {
            println!(
                "{name:<22} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
                q[0], q[1], q[2], q[3], q[4]
            );
        };
        row("Cr_raw = dt/k_raw", q_cr_raw);
        row("Cr    = dt/k_musk", q_cr);
        row("X_cunge (pre-cap)", q_xc);
        row("X_eff   (used)", q_x);
        if !floored_ratio.is_empty() {
            row("k_musk/k_raw | floored", pct(&mut floored_ratio));
        } else {
            println!("{:<22} {:>9}", "k_musk/k_raw | floored", "n/a (none floored)");
        }
        println!(
            "\nfrac Cr_raw > 2      : {:.4}  ({} of {})",
            frac_cr_raw_gt2,
            r.cr_raw.iter().filter(|&&v| v > 2.0).count(),
            r.cr_raw.len()
        );
        println!("frac K floored       : {frac_floored:.4}");
        println!("frac X cap binds     : {cap_binds:.4}  (x_eff < x_cunge)");
        println!("min c1 = {min_c1:.3e}  (c1 < 0: {neg_c1})");
        println!("min c3 = {min_c3:.3e}  (c3 < 0: {neg_c3})");

        let mut push = |metric: &str, q: [f32; 5], extra: String| {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                s.label, metric, q[0], q[1], q[2], q[3], q[4], extra
            ));
        };
        push("cr_raw", q_cr_raw, format!("frac_gt2={frac_cr_raw_gt2}"));
        push("cr", q_cr, String::new());
        push("x_cunge", q_xc, String::new());
        push("x_eff", q_x, format!("frac_cap_binds={cap_binds}"));
        csv.push_str(&format!(
            "{},negative_solves,,,,,,{}/{} ({:.4}%)\n",
            s.label, r.neg_solves, r.total_solves, pc
        ));
        csv.push_str(&format!(
            "{},coeff_min,,,,,,min_c1={min_c1:.3e} min_c3={min_c3:.3e} neg_c1={neg_c1} neg_c3={neg_c3} frac_k_floored={frac_floored:.4}\n",
            s.label
        ));
    }

    if let Some(p) = cli.output {
        std::fs::write(&p, csv).map_err(|e| format!("write {}: {e}", p.display()))?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}
