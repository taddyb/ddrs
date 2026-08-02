//! Probe: does the routing objective actually want a stronger Manning's-`n`
//! vs basin-scale relationship, or is the near-flat field the trained head
//! produces already optimal?
//!
//! # Why this exists
//!
//! Across the 2026-07-30..08-02 CONUS run series every trained head converged
//! to a nearly scale-independent `n` field with a faint positive tilt against
//! `log10_uparea` (Spearman +0.205 for the best run, +0.076 for the gentlest,
//! versus -0.230 at initialization). Two incompatible explanations fit that
//! observation equally well:
//!
//!   1. UNDER-TRAINED — the objective wants a stronger slope but 30-50
//!      optimizer steps under a count-dominated loss never got there.
//!   2. ALREADY OPTIMAL — a near-flat field IS this objective's optimum, and
//!      every optimizer / learning-rate / epoch experiment was doomed.
//!
//! Distinguishing them by training costs ~13 h per attempt. This binary does
//! it with forward passes only: it reads the trained parameter field, rewrites
//! ONLY the component of `n` that is linear in `log10_uparea`, and re-scores.
//! `p_spatial` and `q_spatial` are held at their trained values throughout, so
//! the slope is the single manipulated variable.
//!
//! # Reading the result
//!
//! * amplified fields (k > 1) score BETTER than k=1 -> hypothesis 1; the run
//!   is under-trained and more steps / rebalanced sampling should help.
//! * amplified fields score WORSE -> hypothesis 2; stop tuning the optimizer
//!   and change the objective, the metric, or the timestep.
//! * `floor` (n pinned at the range minimum = identity routing) scores about
//!   the same as k=1 -> the pooled metric carries no routing signal at all,
//!   which invalidates ranking models by it.
//!
//! # Caveats
//!
//! * `EvalParams::Frozen` does NOT support cross-chunk state injection
//!   (`eval.rs`): it cold-restarts from hotstart every chunk, whereas the
//!   KanHead path carries state. So k=1 here will NOT exactly reproduce the
//!   training run's headline NSE. That is expected and harmless — every field
//!   below goes through the identical path, so the COMPARISON is clean. Use
//!   k=1 as the internal reference, never the run manifest's number.
//! * Full 15-year eval costs ~4.5 h PER FIELD. Default here is a 2-water-year
//!   window (~0.6 h/field). Widen with --start/--end once a direction shows.
//!
//! ```bash
//! cargo run --release --bin probe_n_slope -- \
//!   --config .ddrs/runs/<id>/config.yaml \
//!   --checkpoint .ddrs/runs/<id>/checkpoints/epoch_30_mb_0/head \
//!   --output /tmp/n_slope_probe.csv
//! ```

use std::path::PathBuf;

use burn::tensor::{backend::Backend, Tensor};
use clap::Parser;

use ddrs::config::{kan_config, Config, ConfigMode, SparseSolver};
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::data::test_window::TestWindow;
use ddrs::nn::kan_head::KanHead;
use ddrs::routing::denormalize;
use ddrs::training::checkpoint::load_kan_head;
use ddrs::training::forward::FrozenParams;
use ddrs::training::{evaluate, EvalParams};

#[derive(Parser, Debug)]
#[command(name = "probe_n_slope", about = "Score hand-built Manning's-n fields without training")]
struct Cli {
    /// Training YAML from the run whose head you are probing.
    #[arg(long)]
    config: PathBuf,

    /// Trained KAN checkpoint base path (no `.mpk`), e.g. `.../epoch_30_mb_0/head`.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Eval window start (YYYY/MM/DD). Defaults to the config's `testing.start_time`.
    #[arg(long)]
    start: Option<String>,

    /// Eval window end (YYYY/MM/DD). Default trims to two water years from `start`
    /// so the probe costs ~0.6 h per field instead of ~4.5 h.
    #[arg(long)]
    end: Option<String>,

    /// Slope multipliers to score. `1` reproduces the trained field; `0` removes
    /// the uparea-linear component while keeping residual spatial structure.
    #[arg(long, default_value = "0,1,2,5,10", value_delimiter = ',')]
    amplify: Vec<f32>,

    /// Impose `n` as a monotone ramp in `log10_uparea` between two physical
    /// endpoints, e.g. `0.02:0.08` (smallest reach -> largest). Repeatable.
    ///
    /// Why this exists alongside --amplify: when the trained field has
    /// collapsed onto a range bound, amplifying its slope has no headroom
    /// (clamping eats the downward half and the surviving linear term is
    /// negligible), so --amplify cannot test whether a scale relationship
    /// would help. An imposed ramp asks that question directly, independent
    /// of where training landed. Reaches are ranked by `log10_uparea`, so the
    /// ramp is robust to that attribute's distribution.
    #[arg(long = "impose", value_name = "LO:HI")]
    impose: Vec<String>,

    /// Skip the spatially flat control (flat `n` at the trained median).
    #[arg(long, action = clap::ArgAction::SetTrue)]
    no_flat: bool,

    /// Skip the identity-routing control (`n` pinned at the range floor).
    #[arg(long, action = clap::ArgAction::SetTrue)]
    no_floor: bool,

    /// Days per chunk. Default 15 matches DDR's test config.
    #[arg(long, default_value_t = 15)]
    batch_size_days: usize,

    /// CSV output path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// "cpu" (NdArray, deterministic) or "cuda".
    #[arg(long, default_value = "cuda")]
    backend: String,
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
        other => Err(format!("unknown --backend {other} (expected \"cpu\" or \"cuda\")").into()),
    }
}

/// One field to score: a label plus the per-reach physical `n` vector.
struct Field {
    label: String,
    n: Vec<f32>,
}

fn median(v: &[f32]) -> f32 {
    let mut s: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return f32::NAN;
    }
    s.sort_unstable_by(|a, b| a.partial_cmp(b).expect("finite"));
    let m = s.len() / 2;
    if s.len() % 2 == 0 {
        0.5 * (s[m - 1] + s[m])
    } else {
        s[m]
    }
}

fn run<I: Backend>(cli: Cli, device: I::Device) -> R<()> {
    let mut cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)
        .map_err(|e| format!("load config in testing mode: {e}"))?;
    if cli.backend == "cpu" {
        cfg.params.sparse_solver = SparseSolver::Cpu;
        cfg.params.use_cuda_graphs = false;
        eprintln!("backend: cpu (NdArray, deterministic; sparse_solver forced to cpu, cuda graphs off)");
    } else {
        eprintln!(
            "backend: cuda (sparse_solver={:?}, use_cuda_graphs={})",
            cfg.params.sparse_solver, cfg.params.use_cuda_graphs
        );
    }

    // Narrow the eval window. Full CONUS eval is ~4.5 h per field; the probe
    // only needs enough record to rank fields, not to publish a metric.
    {
        let exp = cfg.experiment.as_mut().expect("experiment section");
        if let Some(s) = cli.start.clone() {
            exp.start_time = s;
        }
        exp.end_time = match cli.end.clone() {
            Some(e) => e,
            None => {
                let y: i32 = exp.start_time[0..4].parse().map_err(|e| format!("parse start year: {e}"))?;
                format!("{}/09/30", y + 2)
            }
        };
        eprintln!("eval window: {} -> {}", exp.start_time, exp.end_time);
    }

    let dataset = MeritGagesDataset::open(&cfg).map_err(|e| format!("open dataset: {e}"))?;

    // Probe the eval network once to get its reach count and attribute matrix.
    // `evaluate` builds the same network internally, in the same order.
    let axis = dataset.time_axis().clone();
    let probe = dataset
        .collate_window(&TestWindow::new(&axis, 0, 1))
        .map_err(|e| format!("probe collate: {e}"))?;
    let n_reaches = probe.divide_comids.len();
    let tensors = probe.to_tensors::<I>(&device);
    eprintln!("eval network: {n_reaches} reaches");

    // ---- Trained parameter field -------------------------------------------
    <I as Backend>::seed(&device, cfg.seed);
    let head_cfg = cfg.kan_head.as_ref().expect("kan_head section");
    let template: KanHead<I> = kan_config(head_cfg, cfg.seed).init::<I>(&device);
    let head = load_kan_head::<I>(&cli.checkpoint, template, &device)
        .map_err(|e| format!("load checkpoint: {e}"))?;

    let raw = head.forward(tensors.spatial_attributes.clone());
    let is_log = |k: &str| cfg.params.log_space_parameters.iter().any(|s| s == k);
    let to_vec = |t: Tensor<I, 1>| -> Vec<f32> { t.into_data().to_vec::<f32>().unwrap() };

    let ranges = &cfg.params.parameter_ranges;
    let n_trained = to_vec(denormalize(raw["n"].clone(), ranges.n, is_log("n")));
    let q_trained = to_vec(denormalize(
        raw["q_spatial"].clone(),
        ranges.q_spatial,
        is_log("q_spatial"),
    ));
    let p_trained = if head_cfg.learnable_parameters.iter().any(|s| s == "p_spatial") {
        to_vec(denormalize(
            raw["p_spatial"].clone(),
            ranges.p_spatial,
            is_log("p_spatial"),
        ))
    } else {
        let d = *cfg.params.defaults.get("p_spatial").unwrap_or(&21.0);
        vec![d; n_reaches]
    };

    // ---- Decompose n into (uparea-linear trend) + residual ------------------
    // `spatial_attributes` is already z-scored, and normalization is monotone,
    // so the standardized column is a faithful stand-in for log10(uparea).
    let u_idx = head_cfg
        .input_var_names
        .iter()
        .position(|v| v == "log10_uparea")
        .ok_or("kan_head.input_var_names must contain log10_uparea for this probe")?;
    let attrs: Vec<f32> = tensors
        .spatial_attributes
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let n_feat = head_cfg.input_var_names.len();
    let u: Vec<f32> = (0..n_reaches).map(|i| attrs[i * n_feat + u_idx]).collect();

    let u_bar = u.iter().sum::<f32>() / n_reaches as f32;
    let n_bar = n_trained.iter().sum::<f32>() / n_reaches as f32;
    let (mut sxy, mut sxx) = (0f64, 0f64);
    for i in 0..n_reaches {
        let du = (u[i] - u_bar) as f64;
        sxy += du * (n_trained[i] - n_bar) as f64;
        sxx += du * du;
    }
    let slope = if sxx > 0.0 { (sxy / sxx) as f32 } else { 0.0 };
    eprintln!(
        "trained field: median n = {:.5}, d(n)/d(z_uparea) = {:+.6}",
        median(&n_trained),
        slope
    );

    let [lo, hi] = ranges.n;
    let clamp = |x: f32| x.clamp(lo, hi);

    let mut fields: Vec<Field> = Vec::new();
    for &k in &cli.amplify {
        // Amplify ONLY the uparea-linear component; residual structure is kept
        // so the k=1 case is bit-for-bit the trained field.
        let n: Vec<f32> = (0..n_reaches)
            .map(|i| clamp(n_trained[i] + (k - 1.0) * slope * (u[i] - u_bar)))
            .collect();
        fields.push(Field { label: format!("slope x{k}"), n });
    }
    // Imposed ramps: rank reaches by log10_uparea and map to [lo, hi]. Rank
    // rather than value keeps the ramp well-conditioned however uparea is
    // distributed, and guarantees the full endpoint span is realized.
    let mut order: Vec<usize> = (0..n_reaches).collect();
    order.sort_unstable_by(|&a, &b| u[a].partial_cmp(&u[b]).expect("finite attr"));
    for spec in &cli.impose {
        let (a, b) = spec
            .split_once(':')
            .ok_or_else(|| format!("--impose expects LO:HI, got `{spec}`"))?;
        let lo_i: f32 = a.parse().map_err(|e| format!("--impose LO `{a}`: {e}"))?;
        let hi_i: f32 = b.parse().map_err(|e| format!("--impose HI `{b}`: {e}"))?;
        let mut n = vec![0f32; n_reaches];
        for (rank, &i) in order.iter().enumerate() {
            let f = if n_reaches > 1 { rank as f32 / (n_reaches - 1) as f32 } else { 0.0 };
            n[i] = clamp(lo_i + (hi_i - lo_i) * f);
        }
        fields.push(Field { label: format!("ramp {lo_i}->{hi_i}"), n });
    }

    if !cli.no_flat {
        let m = median(&n_trained);
        fields.push(Field { label: "flat (median n)".into(), n: vec![m; n_reaches] });
    }
    if !cli.no_floor {
        fields.push(Field { label: "floor (identity)".into(), n: vec![lo; n_reaches] });
    }

    // ---- Score every field through the identical eval path -----------------
    let mut rows: Vec<(String, f32, f32, f32, f32, usize)> = Vec::new();
    for f in &fields {
        let frozen = FrozenParams {
            n: f.n.clone(),
            q_spatial: q_trained.clone(),
            p_spatial: p_trained.clone(),
        };
        eprintln!("--- scoring {:<18} median n = {:.5}", f.label, median(&f.n));
        let out = evaluate::<I>(
            &cfg,
            &dataset,
            EvalParams::Frozen(&frozen),
            &device,
            cli.batch_size_days,
            &cli.checkpoint,
        )
        .map_err(|e| format!("evaluate field {}: {e}", f.label))?;
        let finite = out.metrics.nse.iter().filter(|x| x.is_finite()).count();
        let (nse, kge) = (median(&out.metrics.nse), median(&out.metrics.kge));
        eprintln!("    median NSE {nse:.4}   median KGE {kge:.4}   ({finite} finite gauges)");
        rows.push((f.label.clone(), median(&f.n), nse, kge, median(&out.metrics.fhv), finite));
    }

    // ---- Report -------------------------------------------------------------
    let ref_nse = rows
        .iter()
        .find(|r| r.0 == "slope x1")
        .map(|r| r.2)
        .unwrap_or(f32::NAN);
    println!("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>8}", "field", "median_n", "NSE", "dNSE", "KGE", "FHV");
    for (label, mn, nse, kge, fhv, _) in &rows {
        println!("{label:<20} {mn:>10.5} {nse:>10.4} {:>+10.4} {kge:>10.4} {fhv:>8.2}", nse - ref_nse);
    }
    println!("\ndNSE is relative to `slope x1` (the trained field scored through this same path).");

    if let Some(p) = cli.output {
        let mut s = String::from("field,median_n,nse,dnse,kge,fhv,n_finite\n");
        for (label, mn, nse, kge, fhv, fin) in &rows {
            s.push_str(&format!("{label},{mn},{nse},{},{kge},{fhv},{fin}\n", nse - ref_nse));
        }
        std::fs::write(&p, s).map_err(|e| format!("write {}: {e}", p.display()))?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}
