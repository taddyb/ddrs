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

    // ── reach subdivision ────────────────────────────────────────────────────
    /// MERIT flowlines fabric (`.shp`/`.dbf`/`.gpkg`). Switches the run to the
    /// MANAGED adjacency build, overriding `data_sources.conus_adjacency` /
    /// `gages_adjacency`. Required for `--max-pieces` and `--clamp-report`,
    /// because subdivision only happens inside that builder.
    #[arg(long)]
    fabric: Option<PathBuf>,

    /// Layer name for a multi-layer `.gpkg` fabric.
    #[arg(long)]
    fabric_layer: Option<String>,

    /// Workspace root holding `adjacency/<key>/` build caches. Note: subdivided
    /// stores are cached separately per cap, so re-running a cap is instant.
    #[arg(long, default_value = ".ddrs")]
    workspace: PathBuf,

    /// Enable subdivision with this `max_pieces` cap. Omit for the un-split
    /// control. Implies `--fabric`.
    #[arg(long)]
    max_pieces: Option<usize>,

    /// Override `params.subdivision.reference_n`. The default 0.05 is a
    /// *guess* at the trained CONUS median; the reference celerity scales as
    /// `1/n`, so this directly sets `dx_target` and therefore both the piece
    /// count and the clamped fraction. Sweep it against the checkpoint's actual
    /// median `n` before concluding anything about subdivision.
    #[arg(long)]
    reference_n: Option<f32>,

    /// Override `params.subdivision.min_length_fraction` (short-reach clamp
    /// target, as a fraction of `dx_target`; 0 disables the clamp).
    #[arg(long)]
    min_length_fraction: Option<f32>,

    /// Report the reach-plan cost (piece histogram, clamped fraction,
    /// clamp-factor percentiles, total length inflation) and exit without
    /// routing. Requires `--fabric`; `--max-pieces` selects the cap.
    #[arg(long, default_value_t = false)]
    clamp_report: bool,

    /// Divide the cold-start `q'_0` by each row's piece count, so the hot-start
    /// `(I − N)·Q_0 = q'_0` solve sees the same forcing `forward` routes.
    /// Without it the subdivided initial condition is inflated ~m× per reach.
    #[arg(long, default_value_t = false)]
    divide_hotstart: bool,

    /// Trace the first N timesteps of basin-outlet discharge to
    /// `<--output>.trace.csv`, for measuring hot-start wash-out. 0 = off.
    #[arg(long, default_value_t = 0)]
    trace_steps: usize,
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

/// Report the reach-plan cost on the real fabric without writing a store.
///
/// This is the price of the SHORT-reach branch of the two-sided rule: reaches
/// stretched up to `dx_target` (bounded by `max_clamp_factor`) get a
/// proportionally longer travel time, which is a physical distortion traded for
/// numerical stability. Reaches pinned AT the ceiling stay over-Courant.
fn clamp_report(
    cli: &Cli,
    fabric: &std::path::Path,
    subdivision: &ddrs::config::Subdivision,
    attributes: &[PathBuf],
) -> R<()> {
    use ddrs::adjacency::cache::{parent_adjacency_from_fabric, reach_plan};
    use ddrs::adjacency::subdivide::plan_stats;

    let t0 = std::time::Instant::now();
    let conus = parent_adjacency_from_fabric(fabric, cli.fabric_layer.as_deref())
        .map_err(|e| format!("read fabric: {e}"))?;
    eprintln!(
        "fabric: {} parents, {} edges ({:.1}s)",
        conus.order.len(),
        conus.rows.len(),
        t0.elapsed().as_secs_f64()
    );

    let plan = reach_plan(&conus, subdivision, attributes)
        .map_err(|e| format!("reach plan: {e}"))?;
    let st = plan_stats(&conus.length_m, &plan, subdivision);

    println!("\n================ reach plan (max_pieces = {}, min_length_fraction = {}, max_clamp_factor = {}) ================",
        subdivision.max_pieces, subdivision.min_length_fraction, subdivision.max_clamp_factor);
    println!("parents                 : {}", st.n);
    println!(
        "sub-reaches (Σm)        : {} ({:.3}x)",
        st.sum_pieces,
        st.sum_pieces as f64 / st.n as f64
    );
    println!(
        "reaches split (m > 1)   : {} ({:.2}%)",
        st.n_split,
        100.0 * st.n_split as f64 / st.n as f64
    );
    print!("piece histogram         :");
    for (m, &c) in st.pieces_hist.iter().enumerate().skip(1) {
        print!(" m={m}:{c}");
    }
    println!();
    println!(
        "reaches length-clamped  : {} ({:.2}%)",
        st.n_clamped,
        100.0 * st.n_clamped as f64 / st.n as f64
    );
    println!(
        "  pinned at max_clamp_factor={:.1}: {} ({:.2}% of all, {:.2}% of clamped) \
         — these stay over-Courant BY DESIGN",
        subdivision.max_clamp_factor,
        st.n_at_clamp_ceiling,
        100.0 * st.n_at_clamp_ceiling as f64 / st.n as f64,
        100.0 * st.n_at_clamp_ceiling as f64 / st.n_clamped.max(1) as f64
    );
    println!(
        "  clamp factor p50/p95/p99/max: {:.3} / {:.3} / {:.3} / {:.3}",
        st.clamp_factor_p50, st.clamp_factor_p95, st.clamp_factor_p99, st.clamp_factor_max
    );
    println!(
        "total channel length    : {:.1} km → {:.1} km ({:+.2}%)",
        st.total_length_before_m / 1000.0,
        st.total_length_after_m / 1000.0,
        100.0 * st.length_inflation()
    );
    Ok(())
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

    // --- reach subdivision -------------------------------------------------
    // Subdivision happens inside the MANAGED adjacency builder, which is only
    // reached when `data_sources` carries no explicit adjacency paths. So
    // `--fabric` swaps the config over to the managed path and splices the
    // resolved (possibly subdivided) store paths back in.
    if let Some(fabric) = cli.fabric.clone() {
        cfg.params.subdivision.enabled = cli.max_pieces.is_some();
        if let Some(m) = cli.max_pieces {
            cfg.params.subdivision.max_pieces = m;
        }
        if let Some(v) = cli.reference_n {
            cfg.params.subdivision.reference_n = v;
        }
        if let Some(v) = cli.min_length_fraction {
            cfg.params.subdivision.min_length_fraction = v;
        }
        let ds = cfg.data_sources.as_mut().ok_or("config has no data_sources")?;
        ds.conus_adjacency = None;
        ds.gages_adjacency = None;
        ds.geospatial_fabric = Some(fabric.clone());
        ds.geospatial_fabric_layer = cli.fabric_layer.clone();
        let (gages_csv, attributes) = (ds.gages.clone(), ds.attributes.clone());

        if cli.clamp_report {
            return clamp_report(&cli, &fabric, &cfg.params.subdivision, &attributes);
        }

        let outcome = ddrs::adjacency::cache::resolve_or_build(
            &cli.workspace,
            &fabric,
            cli.fabric_layer.as_deref(),
            &gages_csv,
            &cfg.params.subdivision,
            &attributes,
        )
        .map_err(|e| format!("managed adjacency build: {e}"))?;
        eprintln!(
            "adjacency: {} (cache {})",
            outcome.paths.conus.display(),
            if outcome.cache_hit { "hit" } else { "miss" }
        );
        let ds = cfg.data_sources.as_mut().unwrap();
        ds.conus_adjacency = Some(outcome.paths.conus);
        ds.gages_adjacency = Some(outcome.paths.gages);
    } else if cli.max_pieces.is_some() || cli.clamp_report {
        return Err("--max-pieces / --clamp-report require --fabric: subdivision \
                    only runs inside the managed adjacency builder"
            .into());
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

    // The head runs at PARENT resolution; expand onto the routing's sub-reach
    // rows exactly as `training::forward` does. No-op when not subdivided.
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
        engine.divide_hotstart_by_pieces = cli.divide_hotstart;
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
        let t0 = std::time::Instant::now();
        let rep = run_courant_probe::<I>(&cfg_v, &inp, cli.steps, cli.sample_every, cli.trace_steps);
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "    {} steps, {} sampled, negative solves {}/{} — {:.2}s ({:.2} ms/step)",
            rep.n_steps,
            rep.n_sampled_steps,
            rep.neg_solves,
            rep.total_solves,
            secs,
            1000.0 * secs / rep.n_steps.max(1) as f64
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
        let frac_cr_raw_lt05 = r.cr_raw.iter().filter(|&&v| v < 0.5).count() as f64 / n_s;
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
        // The whole claim under test: "Cr ~ 1 makes c1 and c3 non-negative by
        // construction". Both are >= 0 exactly when Cr lands inside the
        // Muskingum window `2X <= Cr <= 2(1-X)`, whose width is `2(1-2X)` — it
        // COLLAPSES as X -> 0.5. So report the window width alongside the hit
        // rate; a 1%-wide window cannot be hit by a static piece count.
        let both_ok = r
            .c1
            .iter()
            .zip(r.c3.iter())
            .filter(|(a, b)| **a >= 0.0 && **b >= 0.0)
            .count() as f64
            / n_s;
        let mut window: Vec<f32> = r.x_cunge.iter().map(|&x| 2.0 * (1.0 - 2.0 * x)).collect();
        let q_win = pct(&mut window);
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
        println!(
            "frac Cr_raw < 0.5    : {:.4}  ({} of {})",
            frac_cr_raw_lt05,
            r.cr_raw.iter().filter(|&&v| v < 0.5).count(),
            r.cr_raw.len()
        );
        println!("frac K floored       : {frac_floored:.4}");
        println!("frac X cap binds     : {cap_binds:.4}  (x_eff < x_cunge)");
        println!("min c1 = {min_c1:.3e}  (c1 < 0: {neg_c1}, frac {:.4})", neg_c1 as f64 / n_s);
        println!("min c3 = {min_c3:.3e}  (c3 < 0: {neg_c3}, frac {:.4})", neg_c3 as f64 / n_s);
        println!("frac c1>=0 AND c3>=0 : {both_ok:.4}");
        println!(
            "non-neg window 2(1-2X) p5/p50/p95: {:.5} / {:.5} / {:.5}  \
             (Cr must land in [2X, 2-2X] for BOTH coefficients)",
            q_win[0], q_win[2], q_win[4]
        );

        let mut push = |metric: &str, q: [f32; 5], extra: String| {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                s.label, metric, q[0], q[1], q[2], q[3], q[4], extra
            ));
        };
        push(
            "cr_raw",
            q_cr_raw,
            format!("frac_gt2={frac_cr_raw_gt2} frac_lt0.5={frac_cr_raw_lt05}"),
        );
        push("cr", q_cr, String::new());
        push("x_cunge", q_xc, String::new());
        push("x_eff", q_x, format!("frac_cap_binds={cap_binds}"));
        csv.push_str(&format!(
            "{},negative_solves,,,,,,{}/{} ({:.4}%)\n",
            s.label, r.neg_solves, r.total_solves, pc
        ));
        csv.push_str(&format!(
            "{},coeff_min,,,,,,min_c1={min_c1:.3e} min_c3={min_c3:.3e} neg_c1={neg_c1} neg_c3={neg_c3} frac_both_nonneg={both_ok:.4} frac_k_floored={frac_floored:.4}\n",
            s.label
        ));
    }

    if let Some(p) = cli.output {
        std::fs::write(&p, csv).map_err(|e| format!("write {}: {e}", p.display()))?;
        eprintln!("wrote {}", p.display());

        // Hot-start wash-out trace: total network discharge per timestep, one
        // column per enforce_positivity arm. Index 0 is the cold-start Q_0.
        if cli.trace_steps > 0 {
            let mut t = String::from("step,flag,total_q\n");
            for s in &summaries {
                for (i, v) in s.rep.trace_total_q.iter().enumerate() {
                    t.push_str(&format!("{i},{},{v}\n", s.label));
                }
            }
            let tp = p.with_extension("trace.csv");
            std::fs::write(&tp, t).map_err(|e| format!("write {}: {e}", tp.display()))?;
            eprintln!("wrote {}", tp.display());
        }
    }
    Ok(())
}
