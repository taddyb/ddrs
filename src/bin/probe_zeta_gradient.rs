//! Stage-1 adjoint reachability probe driver (spec:
//! docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md).
//!
//! Samples training-style batches (same sampler + rho-window machinery as the
//! training driver, but a LOCAL rng seeded from --seed), runs
//! `probe_forward` + the config loss + backward at a FIXED head (no optimizer
//! step ever), reads the gradients on the lifted normalized KAN-head output
//! leaves (default: the leakance trio K_D, d_gw, leakance_factor; override via
//! --params), accumulates per-COMID, and writes the mean |g| / signed g map
//! via `write_grad_netcdf` (default) or `write_param_grad_netcdf` (--params).
//!
//! Usage (CPU, deterministic — the GPU may be busy training):
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --config config/experiments/leakance_hourly_on.yaml \
//!       --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
//!       --windows 32 --seed 42 \
//!       --output output/grad_probe_trained.nc
//!
//! Omit --checkpoint for the cold (fresh-init) probe point.
//!
//! Stage 2 (`--mode perturb`): forward-only q'-perturbation rounds. Each round
//! of the `--probe-plan` CSV adds a constant +delta m³/s to the lateral-inflow
//! forcing at the round's probe reaches, runs the chunked eval loop over the
//! first `--eval-days` days, and writes daily gauge predictions to
//! `<output_dir>/round_<k>.nc`. Two unperturbed baselines run first
//! (`baseline_1.nc`, `baseline_2.nc`) and must be byte-identical on the CPU
//! backend (asserted in-binary):
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode perturb \
//!       --config config/experiments/leakance_hourly_on.yaml \
//!       --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
//!       --probe-plan output/probe_plan.csv --eval-days 1095 \
//!       --output output/perturb_runs/
//!
//! Stage 3 (`--mode teacher`): planted-leakance world — overrides the KAN
//! head's normalized leakance outputs at specified reaches with values from a
//! CSV, then runs the chunked eval loop and writes (a) synthetic daily gauge
//! observations as a zarr-v2 store and (b) a per-reach zeta answer key netCDF.
//! `--output` is not used in teacher mode.
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode teacher \
//!       --config config/experiments/leakance_hourly_on.yaml \
//!       --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
//!       --eval-days 1095 \
//!       --plant-file output/plant_sites.csv \
//!       --obs-output output/teacher_obs/ \
//!       --zeta-output output/teacher_zeta.nc
//!
//! Stage 4 (`--mode floor`): per-day windowed |pred-obs| residuals for
//! transient-floor curves. Teacher weights + self-generated synthetic obs →
//! measures the IC-transient noise floor at every warmup post-hoc (one run per
//! rho). `--checkpoint` is REQUIRED; `--output` is the output netCDF file.
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode floor --backend cpu \
//!       --config config/experiments/floor_rho90.yaml \
//!       --checkpoint output/recoverability/init_head \
//!       --windows 96 --seed 42 \
//!       --output output/floor_fix/floor_rho90.nc
//!
//! Post-hoc floor at warmup W: `nanmean(abs_residual[:, :, W:])` over the
//! output netCDF. Spec: docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md §4.
//!
//! Stage 5 (`--mode state-cache`): continuous-run day-boundary discharge cache
//! for warm hotstart injection. Runs one continuous forward over the eval
//! window (TESTING-mode config) and saves per-reach discharge at every day
//! boundary as a netCDF `(day, COMID)` f32 array. Row 0 = approximated
//! hotstart (discharge after hour 1 of day 0); row d (d ≥ 1) = discharge
//! after the last hourly step of day d−1. Consumed by `experiment.state_cache`
//! (Task 2) to seed each training window from a realistic in-sequence state
//! instead of the synthetic hotstart heuristic. Refuses to overwrite a
//! non-empty `--output` file. `--checkpoint` is REQUIRED; `--eval-days`
//! controls the window length.
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode state-cache --backend cpu \
//!       --config config/experiments/recoverability_measure.yaml \
//!       --checkpoint output/recoverability/init_head \
//!       --eval-days 999999 \
//!       --output output/floor_fix/state_cache_teacher.nc

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use burn::backend::Autodiff;
use burn::prelude::ElementConversion;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use ndarray::Array2;

use ddrs::config::{kan_config, Config, ConfigMode, SparseSolver};
use ddrs::data::dataset::{MeritGagesDataset, RoutingTensors};
use ddrs::data::GageMetadata;
use ddrs::data::sampler::{BatchSource, RandomSampler};
use ddrs::data::TestWindow;
use ddrs::dump_parameters::{write_grad_netcdf, write_param_grad_netcdf, write_zeta_netcdf};
use ddrs::nn::kan_head::KanHead;
use ddrs::training::checkpoint::{head_base, load_kan_head};
use ddrs::training::probe::{probe_forward, GradAccum};
use ddrs::training::{batch_loss, forward_eval, forward_eval_reaches, scatter_add_by_group, tau_trim_and_downsample, LeakanceOverride, ZetaSums};

#[derive(Parser, Debug)]
#[command(name = "probe_zeta_gradient", about = "leakance zeta gradient probe (stage 1)")]
struct Cli {
    #[arg(long)]
    config: PathBuf,

    /// Checkpoint DIRECTORY (epoch_E_mb_M/). Omit for the cold (fresh-init) point.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    #[arg(long, default_value = "grad")]
    mode: String,

    /// Stage 1: number of training-style batches to sample.
    #[arg(long, default_value_t = 32)]
    windows: usize,

    /// Sampler seed (identical seed+windows ⇒ identical batch/window sample).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// grad mode: output netCDF FILE. perturb mode: output DIRECTORY
    /// (created; receives baseline_{1,2}.nc + round_<k>.nc).
    /// Not used in teacher mode (teacher writes via --obs-output / --zeta-output).
    #[arg(long)]
    output: Option<PathBuf>,

    /// Stage 2 only: probe plan CSV (round,comid,delta,...). REQUIRED in perturb mode.
    #[arg(long)]
    probe_plan: Option<PathBuf>,

    /// teacher mode: plant CSV (comid,k_d_norm,d_gw_norm,factor_norm,...).
    #[arg(long)]
    plant_file: Option<PathBuf>,

    /// teacher mode: directory for the synthetic-obs zarr-v2 store.
    #[arg(long)]
    obs_output: Option<PathBuf>,

    /// teacher mode: answer-key netCDF (zeta accumulation over the window).
    #[arg(long)]
    zeta_output: Option<PathBuf>,

    /// Backend: "cpu" (NdArray, deterministic; forces sparse_solver=cpu) or "cuda".
    #[arg(long, default_value = "cpu")]
    backend: String,

    /// Stage 2 only: route only the first D days of the eval period.
    #[arg(long, default_value_t = 1095)]
    eval_days: usize,

    /// grad mode only: comma-separated KAN-head outputs to probe.
    /// Defaults to the leakance trio (K_D,d_gw,leakance_factor).
    /// Example: --params n,q_spatial,p_spatial
    #[arg(long)]
    params: Option<String>,
}

#[derive(Copy, Clone, PartialEq)]
enum Mode {
    Grad,
    Perturb,
    Teacher,
    Floor,
    StateCache,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let mode = match cli.mode.as_str() {
        "grad" => Mode::Grad,
        "perturb" => Mode::Perturb,
        "teacher" => Mode::Teacher,
        "floor" => Mode::Floor,
        "state-cache" => Mode::StateCache,
        other => {
            return Err(
                format!(
                    "unknown --mode {other} \
                     (expected \"grad\", \"perturb\", \"teacher\", \"floor\", or \"state-cache\")"
                )
                .into(),
            )
        }
    };

    // --params is only valid in grad mode.
    if cli.params.is_some() && mode != Mode::Grad {
        return Err(format!(
            "--params is only valid in --mode grad (got --mode {})",
            cli.mode
        )
        .into());
    }

    // grad/floor: training-mode config (probe batches replicate the training
    // sampler). perturb/teacher/state-cache: testing-mode config (replicates
    // the eval loop over the test window).
    let cfg_mode = match mode {
        Mode::Grad | Mode::Floor => ConfigMode::Training,
        Mode::Perturb | Mode::Teacher | Mode::StateCache => ConfigMode::Testing,
    };
    let mut cfg = Config::from_yaml_file_with_mode(&cli.config, cfg_mode)?;

    match cli.backend.as_str() {
        "cpu" => {
            type I = burn::backend::NdArray<f32>;
            let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
            cfg.params.sparse_solver = SparseSolver::Cpu;
            eprintln!("backend: cpu (NdArray, deterministic; sparse_solver forced to cpu)");
            <I as burn::tensor::backend::Backend>::seed(&device, cfg.seed);
            match mode {
                Mode::Grad => run::<I>(cfg, cli, device),
                Mode::Perturb => run_perturb::<I>(cfg, cli, device),
                Mode::Teacher => run_teacher::<I>(cfg, cli, device),
                Mode::Floor => run_floor::<I>(cfg, cli, device),
                Mode::StateCache => run_state_cache::<I>(cfg, cli, device),
            }
        }
        "cuda" => {
            type I = burn_cuda::Cuda<f32, i32>;
            // Config-selected CUDA ordinal (top-level `device:` key).
            let device = cubecl::cuda::CudaDevice::new(cfg.device);
            <I as burn::tensor::backend::Backend>::seed(&device, cfg.seed);
            match mode {
                Mode::Grad => run::<I>(cfg, cli, device),
                Mode::Perturb => run_perturb::<I>(cfg, cli, device),
                Mode::Teacher => run_teacher::<I>(cfg, cli, device),
                Mode::Floor => run_floor::<I>(cfg, cli, device),
                Mode::StateCache => run_state_cache::<I>(cfg, cli, device),
            }
        }
        other => Err(format!("unknown --backend {other} (expected \"cpu\" or \"cuda\")").into()),
    }
}

fn run<I: Backend>(cfg: Config, cli: Cli, device: I::Device) -> Result<(), Box<dyn std::error::Error>> {
    let lift: Vec<String> = match &cli.params {
        Some(s) => s.split(',').map(|p| p.trim().to_string()).collect(),
        None => vec!["K_D".into(), "d_gw".into(), "leakance_factor".into()],
    };
    let wants_leakance = lift
        .iter()
        .any(|p| matches!(p.as_str(), "K_D" | "d_gw" | "leakance_factor"));
    if wants_leakance {
        assert!(
            cfg.params.use_leakance,
            "probing leakance params requires params.use_leakance: true — \
             use a leakance experiment config"
        );
    }
    let output = cli.output.ok_or("--output is required in grad mode")?;

    let dataset = MeritGagesDataset::open(&cfg)?;
    let exp = cfg.experiment.as_ref().expect("experiment section required");
    let rho = exp.rho.expect("probe requires experiment.rho (training-style windows)");
    let warmup = exp.warmup;

    // Head on Autodiff<I>. Seed before init so the fresh-init (cold) point is
    // deterministic; mirrors bootstrap_head_and_state (src/training/bootstrap.rs:58).
    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = kan_config(head_section, cfg.seed);
    <Autodiff<I> as Backend>::seed(&device, cfg.seed);
    let head_template: KanHead<Autodiff<I>> = head_cfg.init::<Autodiff<I>>(&device);
    let (head, checkpoint_label) = match &cli.checkpoint {
        Some(dir) => {
            let head = load_kan_head::<Autodiff<I>>(&head_base(dir), head_template, &device)?;
            eprintln!("loaded checkpoint: {}", head_base(dir).display());
            (head, dir.display().to_string())
        }
        None => {
            eprintln!("cold point: fresh-init head (seed {})", cfg.seed);
            (head_template, format!("cold-init-seed{}", cfg.seed))
        }
    };

    // Sampler replica of the training driver (src/training/driver.rs:73-93)
    // with a LOCAL rng: identical --seed ⇒ identical batches/windows across
    // trained and cold runs.
    let mut rng = ChaCha12Rng::seed_from_u64(cli.seed);
    let mut sampler =
        BatchSource::Shuffle(RandomSampler::new(dataset.len(), exp.batch_size, true));
    sampler.reshuffle(&mut rng);

    let mut accums: HashMap<String, GradAccum> =
        lift.iter().map(|name| (name.clone(), GradAccum::new())).collect();

    let mut processed = 0usize;
    let mut total_batches = 0usize;
    while processed < cli.windows {
        if total_batches > 10 * cli.windows {
            return Err(format!(
                "retry cap exceeded: sampled {total_batches} batches but only processed \
                 {processed}/{} — {} skipped (all-NaN batches); check dataset NaN coverage",
                cli.windows,
                total_batches - processed
            )
            .into());
        }

        let idx = match sampler.next_batch() {
            Some(idx) => idx,
            None => {
                sampler.reshuffle(&mut rng);
                continue;
            }
        };
        total_batches += 1;

        // Mirrors src/training/driver.rs:92-181 minus optimizer/checkpointing.
        let staids: Vec<_> = idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
        let window = dataset.time_axis().sample_rho_window(&mut rng, rho);
        let batch = dataset.collate(&staids, &window)?;
        let num_gauges = batch.gauge_staids.len();

        // Capture COMIDs before to_tensors consumes the batch (eval.rs:66 pattern).
        let comids: Vec<i64> = batch.divide_comids.iter().map(|c| c.0).collect();

        // Save observations before consuming `batch` in to_tensors.
        // SP-3 layout: observations shape is (rho_days, G).
        let obs_arr = batch.observations.clone();
        let t_days_full = obs_arr.nrows();

        let tensors = batch.to_tensors::<Autodiff<I>>(&device);
        let (pred_hourly, leaves) = probe_forward::<I>(&cfg, &tensors, &head, &device, &lift);
        let daily = tau_trim_and_downsample(pred_hourly, cfg.params.tau);
        let dims = daily.dims();
        let (g, t_days) = (dims[0], dims[1]);
        debug_assert_eq!(g, num_gauges);

        // Build obs tensor preserving NaN so the filter can detect them.
        assert!(
            t_days_full >= 2 + t_days,
            "obs/pred shape mismatch: obs rows={t_days_full} pred t_days={t_days}"
        );
        let mut obs_buf: Vec<f32> = Vec::with_capacity(g * t_days);
        for gi in 0..g {
            for ti in 0..t_days {
                obs_buf.push(obs_arr[(ti + 1, gi)]);
            }
        }
        let obs_t: Tensor<Autodiff<I>, 2> =
            Tensor::<Autodiff<I>, 1>::from_data(TensorData::new(obs_buf, [g * t_days]), &device)
                .reshape([g, t_days]);

        debug_assert!(warmup < t_days, "warmup={warmup} >= t_days={t_days}; increase rho");
        let p_post = daily.slice([0..g, warmup..t_days]);
        let o_post = obs_t.slice([0..g, warmup..t_days]);

        // Filter gauges whose post-warmup obs window contains any NaN
        // (driver.rs:139-177). Autograd stays alive via Tensor::select.
        let o_post_vec: Vec<f32> = o_post.clone().into_data().into_vec().unwrap();
        let t_post = t_days - warmup;
        let keep_indices: Vec<i32> = (0..g)
            .filter(|&gi| (0..t_post).all(|ti| !o_post_vec[gi * t_post + ti].is_nan()))
            .map(|gi| gi as i32)
            .collect();

        let surviving_g = keep_indices.len();
        if surviving_g == 0 {
            // Skipped batches don't count toward --windows, but the sampler
            // and rng have already advanced — the skip pattern is
            // data-dependent and deterministic, so identical --seed still
            // yields identical windows across trained/cold runs.
            eprintln!("  batch skipped: all {g} gauges have NaN in post-warmup window");
            continue;
        }

        let keep_t: Tensor<Autodiff<I>, 1, burn::tensor::Int> =
            Tensor::from_data(TensorData::new(keep_indices, [surviving_g]), &device);
        let p_filt = p_post.select(0, keep_t.clone());
        let o_filt = o_post.select(0, keep_t);

        let loss = batch_loss(p_filt, o_filt, &exp.loss);
        let loss_f32: f32 = loss.clone().into_scalar().elem::<f32>();

        let grads = loss.backward();
        let grad_vecs: Vec<(String, Vec<f32>)> = lift
            .iter()
            .map(|name| {
                let g: Vec<f32> = leaves
                    .get(name)
                    .grad(&grads)
                    .unwrap_or_else(|| panic!("no grad for '{name}'"))
                    .into_data()
                    .into_vec()
                    .unwrap();
                (name.clone(), g)
            })
            .collect();

        // Fail fast on non-finite gradients: a poisoned mean invalidates the
        // entire accumulation map — do NOT skip-and-continue, which would
        // desync accumulator coverage.
        let mut any_nf = false;
        for (name, g) in &grad_vecs {
            let nf = g.iter().filter(|v| !v.is_finite()).count();
            if nf > 0 {
                eprintln!("batch {}: non-finite grads for '{name}': {nf}", processed + 1);
                any_nf = true;
            }
        }
        if any_nf {
            std::process::exit(1);
        }

        for (name, g) in &grad_vecs {
            accums.get_mut(name.as_str()).unwrap().add(&comids, g, g);
        }

        eprintln!(
            "batch {}/{}: loss={loss_f32:.5} gauges={surviving_g}",
            processed + 1,
            cli.windows
        );
        processed += 1;
    }

    // Drain accumulators in lift order → sorted rows per param.
    let sorted_rows: Vec<(String, Vec<(i64, f64, f64, u32)>)> = lift
        .iter()
        .map(|name| {
            let rows =
                accums.remove(name.as_str()).expect("accumulator missing").into_sorted_rows();
            (name.clone(), rows)
        })
        .collect();

    // All accumulators saw identical (comids, count) streams — any divergence
    // indicates a per-param skip bug was introduced somewhere.
    if sorted_rows.len() > 1 {
        let ref_comids: Vec<i64> = sorted_rows[0].1.iter().map(|r| r.0).collect();
        for (name, rows) in &sorted_rows[1..] {
            assert!(
                rows.iter().map(|r| r.0).eq(ref_comids.iter().copied()),
                "accumulator COMID sets diverged for '{name}' vs '{}' \
                 — per-param skipping was introduced somewhere",
                sorted_rows[0].0
            );
        }
    }

    let comids: Vec<i64> = sorted_rows[0].1.iter().map(|r| r.0).collect();
    let n_windows: Vec<i32> = sorted_rows[0].1.iter().map(|r| r.3 as i32).collect();
    let mean_abs = |rows: &[(i64, f64, f64, u32)]| -> Vec<f32> {
        rows.iter().map(|r| (r.1 / r.3 as f64) as f32).collect()
    };
    let mean_net = |rows: &[(i64, f64, f64, u32)]| -> Vec<f32> {
        rows.iter().map(|r| (r.2 / r.3 as f64) as f32).collect()
    };

    if cli.params.is_some() {
        // Generic writer: one variable pair per probed param.
        let param_names: Vec<&str> = sorted_rows.iter().map(|(n, _)| n.as_str()).collect();
        let abs_vecs: Vec<Vec<f32>> = sorted_rows.iter().map(|(_, r)| mean_abs(r)).collect();
        let net_vecs: Vec<Vec<f32>> = sorted_rows.iter().map(|(_, r)| mean_net(r)).collect();
        write_param_grad_netcdf(
            &output,
            &param_names,
            &comids,
            &abs_vecs,
            &net_vecs,
            &n_windows,
            &checkpoint_label,
            cli.windows,
            cli.seed,
        )
        .map_err(|e| -> Box<dyn std::error::Error> { e })?;
    } else {
        // Legacy writer: leakance trio in fixed variable-name order —
        // byte-identical to the pre-generalization output.
        let get_rows = |name: &str| -> &Vec<(i64, f64, f64, u32)> {
            sorted_rows
                .iter()
                .find(|(n, _)| n.as_str() == name)
                .map(|(_, r)| r)
                .unwrap_or_else(|| panic!("legacy leakance param '{name}' missing"))
        };
        write_grad_netcdf(
            &output,
            &comids,
            &mean_abs(get_rows("leakance_factor")),
            &mean_net(get_rows("leakance_factor")),
            &mean_abs(get_rows("d_gw")),
            &mean_net(get_rows("d_gw")),
            &mean_abs(get_rows("K_D")),
            &mean_net(get_rows("K_D")),
            &n_windows,
            &checkpoint_label,
            cli.windows,
            cli.seed,
        )
        .map_err(|e| -> Box<dyn std::error::Error> { e })?;
    }

    println!(
        "wrote {} ({} reaches, {} batches, seed {})",
        output.display(),
        comids.len(),
        cli.windows,
        cli.seed
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 2: --mode perturb
// ---------------------------------------------------------------------------

/// Parse the probe-plan CSV. Only `round`, `comid`, `delta` are consumed; the
/// stratification columns are for the analysis script.
fn parse_probe_plan(
    path: &Path,
) -> Result<BTreeMap<usize, Vec<(i64, f32)>>, Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let col = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("{}: missing column '{name}'", path.display()).into())
    };
    let (ci_round, ci_comid, ci_delta) = (col("round")?, col("comid")?, col("delta")?);

    let mut rounds: BTreeMap<usize, Vec<(i64, f32)>> = BTreeMap::new();
    let mut seen: BTreeMap<usize, HashSet<i64>> = BTreeMap::new();
    for (row_idx, rec) in rdr.records().enumerate() {
        let rec = rec?;
        let ctx = |e| format!("{} row {}: {e}", path.display(), row_idx + 2);
        let round: usize = rec[ci_round].parse().map_err(|e| ctx(format!("{e}")))?;
        let comid: i64 = rec[ci_comid].parse().map_err(|e| ctx(format!("{e}")))?;
        let delta: f32 = rec[ci_delta].parse().map_err(|e| ctx(format!("{e}")))?;
        if !seen.entry(round).or_default().insert(comid) {
            return Err(
                format!("{}: duplicate comid {comid} in round {round}", path.display()).into(),
            );
        }
        rounds.entry(round).or_default().push((comid, delta));
    }
    if rounds.is_empty() {
        return Err(format!("{}: probe plan has no data rows", path.display()).into());
    }
    Ok(rounds)
}

/// Write one run's daily gauge predictions: dims `(gauge, day)`, f32
/// `predictions`, plus a `gauge_staid` NC_STRING coordinate (file is
/// netCDF-4, which supports string variables natively).
fn write_round_netcdf(
    path: &Path,
    gauge_staids: &[String],
    preds: &Array2<f32>,
    day0: &str,
    eval_days: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    let (g, d) = preds.dim();
    debug_assert_eq!(g, gauge_staids.len());
    let mut file = netcdf::create(path)?;
    file.add_dimension("gauge", g)?;
    file.add_dimension("day", d)?;
    file.add_attribute("day0", day0)?;
    file.add_attribute("eval_days", eval_days)?;
    {
        let mut v = file.add_variable::<f32>("predictions", &["gauge", "day"])?;
        let flat: Vec<f32> = preds.iter().copied().collect();
        v.put_values(&flat, ..)?;
        v.put_attribute("units", "m^3/s")?;
        v.put_attribute(
            "long_name",
            "daily routed gauge predictions (tau-trimmed, last day dropped)",
        )?;
    }
    {
        let mut v = file.add_string_variable("gauge_staid", &["gauge"])?;
        for (i, s) in gauge_staids.iter().enumerate() {
            v.put_string(s, i)?;
        }
    }
    Ok(())
}

/// Stage-2 driver: forward-only q'-perturbation rounds against two
/// deterministic baselines. Replicates `evaluate`'s chunk loop
/// (src/training/eval.rs:56-151) — it can't call `evaluate` directly because
/// the forcing tensors must be perturbed between `to_tensors` and the forward.
fn run_perturb<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    const BATCH_SIZE_DAYS: usize = 15; // eval default (bin/eval.rs --batch-size-days)

    let output = cli.output.ok_or("--output is required in perturb mode")?;
    let plan_path = cli
        .probe_plan
        .as_ref()
        .ok_or("--probe-plan is required in perturb mode")?;
    let rounds = parse_probe_plan(plan_path)?;
    let checkpoint = cli
        .checkpoint
        .as_ref()
        .ok_or("--checkpoint is required in perturb mode")?;

    // Head on the INNER backend I — forward-only, no autograd (bin/eval.rs:106-107).
    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = kan_config(head_section, cfg.seed);
    let head_template: KanHead<I> = head_cfg.init::<I>(&device);
    let head = load_kan_head::<I>(&head_base(checkpoint), head_template, &device)?;
    eprintln!("loaded checkpoint: {}", head_base(checkpoint).display());

    let dataset = MeritGagesDataset::open(&cfg)?;
    let axis = dataset.time_axis().clone();
    let n_days = axis.num_days.min(cli.eval_days);
    assert!(
        n_days >= 3,
        "eval window too short: {n_days} days (need >= 3 for tau-trim + last-day drop)"
    );
    let n_hours = n_days * 24;

    // 1-day probe window: sizes gauges + forces the static-network cache.
    let probe = TestWindow::new(&axis, 0, 1);
    let probe_batch = dataset.collate_window(&probe)?;
    let n_all_gauges = probe_batch.gauge_staids.len();
    let gauge_staids: Vec<String> = probe_batch
        .gauge_staids
        .iter()
        .map(|s| s.as_str().to_string())
        .collect();
    eprintln!(
        "perturb eval: {n_all_gauges} gauges, {n_days} days ({} chunks of {BATCH_SIZE_DAYS})",
        n_days.div_ceil(BATCH_SIZE_DAYS)
    );

    // Validate all plan COMIDs against the eval network before any compute
    // (fail in seconds, not after a 30h sweep).
    {
        let network_comids: HashSet<i64> =
            probe_batch.divide_comids.iter().map(|c| c.0).collect();
        let mut missing: Vec<(usize, i64)> = Vec::new();
        for (&round, probes) in &rounds {
            for &(comid, _) in probes {
                if !network_comids.contains(&comid) {
                    missing.push((round, comid));
                }
            }
        }
        if !missing.is_empty() {
            let msg = missing
                .iter()
                .map(|(r, c)| format!("(round={r}, comid={c})"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(
                format!("probe plan COMIDs not in eval network: {msg}").into(),
            );
        }
    }

    // day0 = ISO date of prediction column 0 (axis.start + 1 day, post-trim).
    let day0 = (axis.start + chrono::Duration::days(1))
        .format("%Y-%m-%d")
        .to_string();

    std::fs::create_dir_all(&output)?;

    // One full chunked eval (mirrors evaluate's loop, eval.rs:97-151) with a
    // constant +delta added to the q' forcing at the run's probe reaches.
    let run_one = |probes: &[(i64, f32)]| -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let mut predictions_full = Array2::<f32>::zeros((n_all_gauges, n_hours));
        let n_chunks_total = n_days.div_ceil(BATCH_SIZE_DAYS);
        let mut day_offset = 0usize;
        let mut chunk_idx = 0usize;
        while day_offset < n_days {
            let chunk_n = (n_days - day_offset).min(BATCH_SIZE_DAYS);
            let win = TestWindow::new(&axis, day_offset, chunk_n);
            let batch = dataset.collate_window(&win)?;
            // Capture COMIDs before to_tensors consumes the batch (eval.rs:66).
            let batch_divide_comids = batch.divide_comids.clone();
            let tensors = batch.to_tensors::<I>(&device);

            // Perturbation: +delta on both forcing fields (hourly repeat-24
            // AND daily disagg input — forward_eval consumes whichever the
            // head config selects). Baselines skip the tensor ops entirely.
            let tensors = if probes.is_empty() {
                tensors
            } else {
                let comid_col: HashMap<i64, usize> = batch_divide_comids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c.0, i))
                    .collect();
                let n_reaches = tensors.q_prime.dims()[1];
                let mut delta_row = vec![0.0f32; n_reaches];
                for &(comid, delta) in probes {
                    if let Some(&col) = comid_col.get(&comid) {
                        delta_row[col] = delta;
                    }
                }
                let delta_t: Tensor<I, 1> =
                    Tensor::from_data(TensorData::new(delta_row, [n_reaches]), &device);
                // Clone the two forcing fields first, then functional-record-
                // update — moving a field inside the same FRU expression is
                // E0382 (partially moved value). Materialize the broadcast
                // per-tensor with each tensor's own dim-0.
                let t_rows = tensors.q_prime.dims()[0];
                let d_rows = tensors.q_prime_daily.dims()[0];
                let q_prime = tensors.q_prime.clone()
                    + delta_t.clone().unsqueeze_dim::<2>(0).expand([t_rows, n_reaches]);
                let q_prime_daily = tensors.q_prime_daily.clone()
                    + delta_t.unsqueeze_dim::<2>(0).expand([d_rows, n_reaches]);
                RoutingTensors::<I> { q_prime, q_prime_daily, ..tensors }
            };

            let pred = forward_eval::<I>(&cfg, &tensors, &head, &device, chunk_idx > 0, None, None);
            let dims = pred.dims();
            debug_assert_eq!(dims[0], n_all_gauges);
            debug_assert_eq!(dims[1], win.n_hourly());
            let v: Vec<f32> = pred.into_data().into_vec().unwrap();
            let pred_arr = Array2::from_shape_vec((dims[0], dims[1]), v).unwrap();

            let h_start = day_offset * 24;
            let h_end = h_start + win.n_hourly();
            predictions_full
                .slice_mut(ndarray::s![.., h_start..h_end])
                .assign(&pred_arr);
            eprintln!(
                "  chunk {}/{}: days {}..{} ({} days)",
                chunk_idx + 1,
                n_chunks_total,
                day_offset,
                day_offset + chunk_n,
                chunk_n,
            );
            day_offset += chunk_n;
            chunk_idx += 1;
        }

        // End-of-pipeline tau-trim + daily downsample (eval.rs:119-129), then
        // drop the LAST day (eval.rs:143-151) → (G, n_days - 2).
        let pred_full_vec: Vec<f32> = predictions_full.iter().copied().collect();
        let pred_full_t: Tensor<I, 2> =
            Tensor::<I, 1>::from_floats(pred_full_vec.as_slice(), &device)
                .reshape([n_all_gauges, n_hours]);
        let daily_t = tau_trim_and_downsample(pred_full_t, cfg.params.tau);
        let daily_dims = daily_t.dims();
        let daily_vec: Vec<f32> = daily_t.into_data().into_vec().unwrap();
        let predictions_daily =
            Array2::from_shape_vec((daily_dims[0], daily_dims[1]), daily_vec).unwrap();
        let pd_dims = predictions_daily.dim();
        Ok(predictions_daily
            .slice(ndarray::s![.., 0..pd_dims.1 - 1])
            .to_owned())
    };

    let n_rounds = rounds.len();

    // Two unperturbed baselines: byte-identity on the CPU backend proves the
    // whole forward is deterministic, so round deltas are attributable to the
    // perturbation alone.
    eprintln!("baseline 1/2 (0 probes)");
    let b1 = run_one(&[])?;
    write_round_netcdf(&output.join("baseline_1.nc"), &gauge_staids, &b1, &day0, n_days as i64)?;
    eprintln!("baseline 2/2 (0 probes)");
    let b2 = run_one(&[])?;
    write_round_netcdf(&output.join("baseline_2.nc"), &gauge_staids, &b2, &day0, n_days as i64)?;

    let max_diff = b1
        .iter()
        .zip(b2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if max_diff == 0.0 {
        eprintln!("determinism check: max |b1-b2| = 0");
    } else if cli.backend == "cpu" {
        eprintln!(
            "determinism check FAILED: max |b1-b2| = {max_diff:e} (expected 0 on cpu backend)"
        );
        std::process::exit(1);
    } else {
        eprintln!(
            "determinism check: max |b1-b2| = {max_diff:e} — non-zero is expected on the \
             cuda backend (atomic scatter_add); use --backend cpu for attributable deltas"
        );
    }

    for (k_idx, (round, round_probes)) in rounds.iter().enumerate() {
        eprintln!(
            "round {}/{n_rounds} (round id {round}, {} probes)",
            k_idx + 1,
            round_probes.len()
        );
        let preds = run_one(round_probes)?;
        write_round_netcdf(
            &output.join(format!("round_{round}.nc")),
            &gauge_staids,
            &preds,
            &day0,
            n_days as i64,
        )?;
    }

    println!(
        "wrote {} runs (2 baselines + {n_rounds} rounds) to {} ({} gauges x {} days)",
        n_rounds + 2,
        output.display(),
        n_all_gauges,
        n_days - 2,
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 3: --mode teacher
// ---------------------------------------------------------------------------

/// Parse the plant-file CSV. Required columns: comid, k_d_norm, d_gw_norm,
/// factor_norm. All three norm columns must be in [0, 1]. Duplicate COMIDs
/// are rejected.
fn parse_plant_file(
    path: &Path,
) -> Result<Vec<(i64, f32, f32, f32)>, Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let col = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("{}: missing column '{name}'", path.display()).into())
    };
    let (ci_c, ci_k, ci_d, ci_f) =
        (col("comid")?, col("k_d_norm")?, col("d_gw_norm")?, col("factor_norm")?);
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for (row_idx, rec) in rdr.records().enumerate() {
        let rec = rec?;
        let ctx = |e: String| format!("{} row {}: {e}", path.display(), row_idx + 2);
        let comid: i64 = rec[ci_c].parse().map_err(|e| ctx(format!("{e}")))?;
        if !seen.insert(comid) {
            return Err(
                format!("{} row {}: duplicate plant comid {comid}", path.display(), row_idx + 2)
                    .into(),
            );
        }
        let norm = |s: &str, name: &str| -> Result<f32, Box<dyn std::error::Error>> {
            let v: f32 = s.parse().map_err(|e| -> Box<dyn std::error::Error> {
                format!("{} row {}: {e}", path.display(), row_idx + 2).into()
            })?;
            if !(0.0..=1.0).contains(&v) {
                return Err(
                    format!(
                        "{} row {}: {name} {v} outside [0,1] for comid {comid}",
                        path.display(),
                        row_idx + 2
                    )
                    .into(),
                );
            }
            Ok(v)
        };
        rows.push((
            comid,
            norm(&rec[ci_k], "k_d_norm")?,
            norm(&rec[ci_d], "d_gw_norm")?,
            norm(&rec[ci_f], "factor_norm")?,
        ));
    }
    if rows.is_empty() {
        return Err(format!("{}: no plant rows", path.display()).into());
    }
    Ok(rows)
}

/// Stage-3 driver: planted-leakance world. Overrides the KAN head's
/// normalized leakance outputs at specified reaches, runs the chunked eval
/// loop, and writes synthetic daily gauge observations (zarr-v2) plus a
/// per-reach zeta answer key (netCDF).
fn run_teacher<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    // Large chunk size reduces disagg boundary-artifact density (left-clamp at chunk
    // day 0 and precip right-clamp at chunk day C-1). With C=365 each artifact appears
    // only ~14 times over 5115 teacher days (0.82%), vs 341 times with C=15 (20%).
    // 70 GB RAM easily holds a 365-day AORC precip chunk (~2.3 GB).
    const BATCH_SIZE_DAYS: usize = 365;
    assert!(cfg.params.use_leakance, "teacher requires params.use_leakance: true");

    let plants = parse_plant_file(
        cli.plant_file.as_ref().ok_or("--plant-file is required in teacher mode")?,
    )?;
    let obs_dir = cli.obs_output.as_ref().ok_or("--obs-output is required in teacher mode")?;
    if obs_dir.exists() && obs_dir.read_dir()?.next().is_some() {
        return Err(format!(
            "--obs-output {} already exists and is non-empty; remove it before \
             re-running teacher mode to prevent stale gauge data",
            obs_dir.display()
        )
        .into());
    }
    let zeta_path = cli.zeta_output.as_ref().ok_or("--zeta-output is required in teacher mode")?;
    if let Some(p) = zeta_path.parent() {
        if !p.as_os_str().is_empty() && !p.exists() {
            return Err(
                format!("--zeta-output parent dir does not exist: {}", p.display()).into(),
            );
        }
    }
    let checkpoint = cli.checkpoint.as_ref().ok_or("--checkpoint is required in teacher mode")?;

    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = kan_config(head_section, cfg.seed);
    let head_template: KanHead<I> = head_cfg.init::<I>(&device);
    let head = load_kan_head::<I>(&head_base(checkpoint), head_template, &device)?;
    eprintln!("loaded checkpoint: {}", head_base(checkpoint).display());

    let dataset = MeritGagesDataset::open(&cfg)?;
    let axis = dataset.time_axis().clone();
    let n_days = axis.num_days.min(cli.eval_days);
    assert!(
        n_days >= 3,
        "eval window too short: {n_days} days (need >= 3 for tau-trim + last-day drop)"
    );
    if n_days < axis.num_days {
        eprintln!(
            "WARNING: truncated teacher window ({n_days}/{} days via --eval-days) — \
             synthetic obs will NOT cover the student training axis; timing-gate use only",
            axis.num_days
        );
    }
    let n_hours = n_days * 24;

    // 1-day probe window: sizes gauges + forces the static-network cache.
    let probe = TestWindow::new(&axis, 0, 1);
    let probe_batch = dataset.collate_window(&probe)?;
    let n_all_gauges = probe_batch.gauge_staids.len();
    let gauge_staids: Vec<String> =
        probe_batch.gauge_staids.iter().map(|s| s.as_str().to_string()).collect();
    let network_comids: Vec<i64> = probe_batch.divide_comids.iter().map(|c| c.0).collect();

    // Fail fast: every plant COMID must be in the eval network.
    {
        let set: HashSet<i64> = network_comids.iter().copied().collect();
        let missing: Vec<i64> =
            plants.iter().map(|p| p.0).filter(|c| !set.contains(c)).collect();
        if !missing.is_empty() {
            return Err(format!("plant COMIDs not in eval network: {missing:?}").into());
        }
    }
    eprintln!(
        "teacher: {} plants, {n_all_gauges} gauges, {} reaches, {n_days} days",
        plants.len(),
        network_comids.len()
    );

    // Dense override vectors over the network's reach columns.
    let comid_col: HashMap<i64, usize> =
        network_comids.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let n_reaches = network_comids.len();
    let mut ov = LeakanceOverride {
        mask: vec![0.0; n_reaches],
        k_d: vec![0.0; n_reaches],
        d_gw: vec![0.0; n_reaches],
        factor: vec![0.0; n_reaches],
    };
    for &(comid, k, d, f) in &plants {
        let col = comid_col[&comid];
        ov.mask[col] = 1.0;
        ov.k_d[col] = k;
        ov.d_gw[col] = d;
        ov.factor[col] = f;
    }

    // Chunked CONTINUOUS forward with overrides + zeta accumulation. Continuity
    // across chunks is maintained by injecting the previous chunk's final
    // per-reach discharge as `tensors.initial_state` (the `carry_state` flag is
    // a no-op through forward_eval — a fresh engine is constructed per call, so
    // its `discharge_t` is always None and the hotstart always ran; this bug
    // made the pre-2026-07-04 synthetic obs restart from hotstart every 15 days,
    // leaving 15-day-periodic discontinuities 10-15x the day-to-day baseline on
    // large basins). The per-reach forward is used so the final column can be
    // extracted; gauge aggregation happens here via scatter_add_by_group.
    let mut zeta_sink = ZetaSums::<I>::new();
    let mut predictions_full = Array2::<f32>::zeros((n_all_gauges, n_hours));
    let n_chunks_total = n_days.div_ceil(BATCH_SIZE_DAYS);
    let mut day_offset = 0usize;
    let mut chunk_idx = 0usize;
    let mut prev_final_state: Option<Vec<f32>> = None;
    while day_offset < n_days {
        let chunk_n = (n_days - day_offset).min(BATCH_SIZE_DAYS);
        let win = TestWindow::new(&axis, day_offset, chunk_n);
        // Disagg lookahead: pass one extra daily Q' row so the disagg "next"
        // for the last day of this chunk uses the NEXT chunk's first day instead
        // of repeating the last day (right-clamp). Skip for the last chunk (no
        // next day to peek at). Without this, the floor's continuous routing
        // diverges from the teacher's chunked routing at each chunk boundary,
        // producing false large-river residuals at the floor bar.
        let has_next_chunk = day_offset + chunk_n < n_days;
        let batch = if has_next_chunk {
            dataset.collate_window_with_daily_lookahead(&win)?
        } else {
            dataset.collate_window(&win)?
        };
        assert!(
            batch.divide_comids.iter().map(|c| c.0).eq(network_comids.iter().copied()),
            "reach column order changed between chunks — override vectors invalid"
        );
        let mut tensors = batch.to_tensors::<I>(&device);
        if let Some(ref state_vec) = prev_final_state {
            let n = state_vec.len();
            tensors.initial_state = Some(Tensor::<I, 1>::from_data(
                TensorData::new(state_vec.clone(), [n]),
                &device,
            ));
        }
        let runoff = forward_eval_reaches::<I>(
            &cfg,
            &tensors,
            &head,
            &device,
            false,
            Some(&mut zeta_sink),
            Some(&ov),
        );
        let chunk_hours = win.n_hourly();
        let runoff_vec: Vec<f32> = runoff.clone().into_data().into_vec().unwrap();
        prev_final_state = Some(
            (0..n_reaches)
                .map(|r| runoff_vec[r * chunk_hours + (chunk_hours - 1)])
                .collect(),
        );
        let pred = scatter_add_by_group(
            runoff,
            tensors.flat_indices.clone(),
            tensors.group_ids.clone(),
            tensors.num_gauges,
        );
        let dims = pred.dims();
        let v: Vec<f32> = pred.into_data().into_vec().unwrap();
        let pred_arr = Array2::from_shape_vec((dims[0], dims[1]), v).unwrap();
        let h_start = day_offset * 24;
        predictions_full
            .slice_mut(ndarray::s![.., h_start..h_start + win.n_hourly()])
            .assign(&pred_arr);
        eprintln!("  chunk {}/{n_chunks_total}", chunk_idx + 1);
        day_offset += chunk_n;
        chunk_idx += 1;
    }

    // tau-trim + daily downsample + drop last day (same as run_perturb).
    let pred_full_vec: Vec<f32> = predictions_full.iter().copied().collect();
    let pred_full_t: Tensor<I, 2> =
        Tensor::<I, 1>::from_floats(pred_full_vec.as_slice(), &device)
            .reshape([n_all_gauges, n_hours]);
    let daily_t = tau_trim_and_downsample(pred_full_t, cfg.params.tau);
    let daily_dims = daily_t.dims();
    let daily_vec: Vec<f32> = daily_t.into_data().into_vec().unwrap();
    let daily_all = Array2::from_shape_vec((daily_dims[0], daily_dims[1]), daily_vec).unwrap();
    let daily = daily_all.slice(ndarray::s![.., 0..daily_dims[1] - 1]).to_owned();

    // Synthetic obs: day0 = axis.start + 1 (tau-trim drops day 0).
    let epoch = chrono::NaiveDate::from_ymd_opt(1980, 1, 1).unwrap();
    let day0 = axis.start + chrono::Duration::days(1);
    ddrs::data::store::obs_writer::write_obs_zarr_v2(obs_dir, &gauge_staids, epoch, day0, &daily)?;
    eprintln!(
        "synthetic obs → {} ({n_all_gauges} gauges, {} days from {day0})",
        obs_dir.display(),
        daily.dim().1
    );

    // Answer key: zeta means over the routed window.
    let scale = 1.0_f32 / zeta_sink.steps as f32;
    assert!(scale.is_finite() && scale > 0.0, "zeta accumulation empty — leakance inactive?");
    let mean_vec = |t: Option<Tensor<I, 1>>| -> Vec<f32> {
        (t.expect("zeta sums present") * scale).into_data().into_vec().unwrap()
    };
    write_zeta_netcdf(
        zeta_path,
        &network_comids,
        &mean_vec(zeta_sink.abs_sum),
        &mean_vec(zeta_sink.net_sum),
        &mean_vec(zeta_sink.depth_sum),
        &mean_vec(zeta_sink.area_z_sum),
        &mean_vec(zeta_sink.q_sum),
        &format!("teacher:{}", checkpoint.display()),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { e })?;
    println!("answer key → {} ({} reaches)", zeta_path.display(), network_comids.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 4: --mode floor
// ---------------------------------------------------------------------------

/// Floor-mode output: per-window, per-gauge-slot, per-post-trim-day |pred-obs|.
/// Dimensions: `(window, gauge_slot, day)`; `gauge_staid` (window, gauge_slot)
/// NC_STRING identifies which gauge occupied each slot (batches differ per
/// window); `uparea` (window, gauge_slot) f32 carries the gauge's drainage area
/// (km²) for decay-by-basin-size stratification. NaN = obs was NaN that
/// window-day; floor(warmup) = nanmean over days >= warmup.
///
/// String storage choice: 2-D NC_STRING variable indexed by (window, gauge_slot)
/// tuple — netcdf v0.12's `put_string` accepts any `E: TryInto<Extents>`, and
/// `(usize, usize)` satisfies this via the crate's tuple blanket impl.
#[allow(clippy::too_many_arguments)]
fn write_floor_netcdf(
    path: &Path,
    n_windows: usize,
    n_slots: usize,
    n_days: usize,
    abs_residual: &[f32],
    gauge_staids: &[String],
    uparea: &[f32],
    seed: u64,
    checkpoint_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(abs_residual.len(), n_windows * n_slots * n_days);
    debug_assert_eq!(gauge_staids.len(), n_windows * n_slots);
    debug_assert_eq!(uparea.len(), n_windows * n_slots);
    let mut file = netcdf::create(path)?;
    file.add_dimension("window", n_windows)?;
    file.add_dimension("gauge_slot", n_slots)?;
    file.add_dimension("day", n_days)?;
    file.add_attribute("seed", seed as i64)?;
    file.add_attribute("checkpoint", checkpoint_label)?;
    file.add_attribute(
        "note",
        "windowed |pred-obs| residuals vs configured obs; day 0 = first post-trim day; \
         floor(warmup) = nanmean over days >= warmup",
    )?;
    {
        let mut v = file.add_variable::<f32>("abs_residual", &["window", "gauge_slot", "day"])?;
        v.put_values(abs_residual, ..)?;
        v.put_attribute("units", "m^3/s")?;
    }
    {
        let mut v = file.add_string_variable("gauge_staid", &["window", "gauge_slot"])?;
        for (i, s) in gauge_staids.iter().enumerate() {
            v.put_string(s, (i / n_slots, i % n_slots))?;
        }
    }
    {
        let mut v = file.add_variable::<f32>("uparea", &["window", "gauge_slot"])?;
        v.put_values(uparea, ..)?;
        v.put_attribute("units", "km^2")?;
    }
    Ok(())
}

/// Build a staid→drainage-area lookup from the gages CSV at `gages_path`.
/// Returns a closure that maps a staid string to drain_sqkm (km²) as f32.
/// Source column: DRAIN_SQKM in the gages CSV (same `GageMetadata` the dataset
/// opens internally). If a staid is not in the CSV (should not happen for
/// training gauges), returns f32::NAN; the analysis stratification then uses obs
/// mean flow instead.
fn build_gauge_uparea_lookup(
    gages_path: &Path,
) -> Result<impl Fn(&str) -> f32, Box<dyn std::error::Error>> {
    let meta = GageMetadata::open(gages_path)?;
    let lookup: HashMap<String, f32> = meta
        .rows
        .iter()
        .map(|r| (r.staid.as_str().to_string(), r.drain_sqkm as f32))
        .collect();
    Ok(move |staid: &str| match lookup.get(staid) {
        Some(&a) => a,
        None => f32::NAN,
    })
}

/// Floor mode: teacher-weights windowed residuals vs self-generated obs.
/// Mirrors `run` (grad mode) sampling exactly — LOCAL ChaCha12 rng from
/// --seed, same batch/window draws — but forward-only on the inner backend,
/// no loss, no backward. Saves |pred - obs| for every post-trim day so the
/// floor at ANY warmup is computable post-hoc: floor(w) = nanmean over
/// days >= w. The head is loaded from --checkpoint (REQUIRED).
///
/// NaN obs propagates to NaN residual (the analysis nanmeans over valid days).
/// No gauge NaN filter: NaN carries the information about which days are
/// outside the obs record.
fn run_floor<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = cli.output.as_ref().ok_or("--output is required in floor mode")?;
    let checkpoint = cli.checkpoint.as_ref().ok_or("--checkpoint is required in floor mode")?;

    // Head on the inner backend I — forward-only, no autograd (same as perturb/teacher).
    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = kan_config(head_section, cfg.seed);
    let head_template: KanHead<I> = head_cfg.init::<I>(&device);
    let head = load_kan_head::<I>(&head_base(checkpoint), head_template, &device)?;
    eprintln!("loaded checkpoint: {}", head_base(checkpoint).display());

    // Extract the gages CSV path before MeritGagesDataset::open borrows cfg.
    let gages_path = cfg
        .data_sources
        .as_ref()
        .ok_or("floor mode requires data_sources (gages path)")?
        .gages
        .clone();

    let dataset = MeritGagesDataset::open(&cfg)?;
    let exp = cfg.experiment.as_ref().expect("experiment section required");
    let rho = exp.rho.expect("floor mode requires experiment.rho");

    // Sampler replica of the training driver (src/training/driver.rs:73-93) with a LOCAL rng
    // seeded from --seed — same sequence as grad mode for the same --seed.
    let mut rng = ChaCha12Rng::seed_from_u64(cli.seed);
    let mut sampler =
        BatchSource::Shuffle(RandomSampler::new(dataset.len(), exp.batch_size, true));
    sampler.reshuffle(&mut rng);

    // Staid → drainage area (km²) from the gages CSV (DRAIN_SQKM column).
    // GageMetadata is opened again here to avoid depending on pub(crate) dataset
    // internals; the file is small and cached in the OS page cache after dataset.open().
    let uparea_of = build_gauge_uparea_lookup(&gages_path)?;

    let (mut all_resid, mut all_staids, mut all_uparea) =
        (Vec::<f32>::new(), Vec::<String>::new(), Vec::<f32>::new());
    let mut n_days_expected: Option<usize> = None;
    let mut processed = 0usize;
    let mut total = 0usize;

    while processed < cli.windows {
        if total > 10 * cli.windows {
            return Err(
                format!(
                    "retry cap exceeded: sampled {total} batches but only {processed}/{} \
                     windows processed — check dataset NaN coverage",
                    cli.windows
                )
                .into(),
            );
        }

        let idx = match sampler.next_batch() {
            Some(idx) => idx,
            None => {
                sampler.reshuffle(&mut rng);
                continue;
            }
        };
        total += 1;

        let staids: Vec<_> = idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
        let window = dataset.time_axis().sample_rho_window(&mut rng, rho);
        let batch = dataset.collate(&staids, &window)?;

        // Save obs + staids before to_tensors consumes the batch (same pattern as
        // grad mode saving obs_arr before to_tensors<Autodiff<I>>).
        let obs_arr = batch.observations.clone();
        let t_days_full = obs_arr.nrows();
        let batch_staids: Vec<String> =
            batch.gauge_staids.iter().map(|s| s.as_str().to_string()).collect();

        let tensors = batch.to_tensors::<I>(&device);
        // forward_eval returns gauge-aggregated hourly predictions (G, T_hours);
        // tau_trim_and_downsample converts to daily (G, T_days).
        // carry_state=false ON PURPOSE: each window is an independent fresh
        // routing from its own hotstart IC — that transient IS the quantity
        // being measured (contrast: perturb/teacher pass chunk_idx > 0 to
        // continue one long eval).
        let pred_hourly =
            forward_eval::<I>(&cfg, &tensors, &head, &device, false, None, None);
        let daily = tau_trim_and_downsample(pred_hourly, cfg.params.tau);
        let dims = daily.dims();
        let (g, t_days) = (dims[0], dims[1]);

        // Assert t_days is constant across windows (rho is fixed → same window
        // length → same tau-trim output length every time).
        if *n_days_expected.get_or_insert(t_days) != t_days {
            return Err(format!(
                "t_days varies across windows: expected {} got {} (window {})",
                n_days_expected.unwrap(), t_days, processed
            ).into());
        }

        let pred: Vec<f32> = daily.into_data().into_vec().unwrap();

        // Mirrors grad mode's shape guard (run ~line 290): obs must be at least
        // t_days + 2 rows (alignment offset 1 + tau-trim drop at end).
        assert!(
            t_days_full >= 2 + t_days,
            "obs/pred shape mismatch: obs rows={t_days_full} pred t_days={t_days}"
        );

        // |pred - obs| with the SAME obs alignment as training (grad mode's
        // obs_arr[(ti + 1, gi)]); NaN obs propagates to NaN residual.
        for gi in 0..g {
            for ti in 0..t_days {
                let o = obs_arr[(ti + 1, gi)];
                all_resid.push((pred[gi * t_days + ti] - o).abs());
            }
        }
        all_staids.extend(batch_staids.iter().cloned());
        all_uparea.extend(batch_staids.iter().map(|s| uparea_of(s)));

        eprintln!(
            "window {}/{} (rho {rho}, {g} gauges, {} days)",
            processed + 1,
            cli.windows,
            t_days
        );
        processed += 1;
    }

    // Summary uparea-miss warning (one line, not per-miss noise).
    {
        let n_miss = all_uparea.iter().filter(|v| v.is_nan()).count();
        let total = all_uparea.len();
        if n_miss > 0 {
            eprintln!(
                "warning: {n_miss}/{total} gauge-window slots had no DRAIN_SQKM \
                 — uparea stratification will be partial"
            );
        }
    }

    // Assert constant batch size: RandomSampler(drop_last=true) guarantees this,
    // but verify before the integer division that computes n_slots.
    assert!(
        all_staids.len() % processed == 0,
        "batch sizes are not constant across {processed} windows \
         (total gauge-slot count {} is not divisible by window count)",
        all_staids.len()
    );
    let n_slots = all_staids.len() / processed;

    write_floor_netcdf(
        output,
        processed,
        n_slots,
        n_days_expected.unwrap(),
        &all_resid,
        &all_staids,
        &all_uparea,
        cli.seed,
        &checkpoint.display().to_string(),
    )?;
    println!(
        "wrote {} ({} windows x {} slots x {} days)",
        output.display(),
        processed,
        n_slots,
        n_days_expected.unwrap()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 5: --mode state-cache
// ---------------------------------------------------------------------------

/// Write a day-boundary discharge state cache as netCDF.
///
/// Layout: `q_state[(d, r)]` = discharge (m³/s) at reach column `r` entering
/// day `d` (hour 0). `day0` = ISO date of the first row. COMID order matches
/// the continuous run's network column order; consumers must look up values
/// by COMID, not by position.
fn write_state_cache_netcdf(
    path: &Path,
    n_days: usize,
    comids: &[i64],
    q_state: &[f32],
    day0: &str,
    checkpoint_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(q_state.len(), n_days * comids.len());
    let mut file = netcdf::create(path)?;
    file.add_dimension("day", n_days)?;
    file.add_dimension("COMID", comids.len())?;
    file.add_attribute("day0", day0)?;
    file.add_attribute("checkpoint", checkpoint_label)?;
    {
        let mut v = file.add_variable::<i64>("COMID", &["COMID"])?;
        v.put_values(comids, ..)?;
    }
    {
        let mut v = file.add_variable::<f32>("q_state", &["day", "COMID"])?;
        v.put_values(q_state, ..)?;
        v.put_attribute("units", "m^3/s")?;
    }
    Ok(())
}

/// Stage-5 driver: one continuous forward over the eval window, capturing
/// per-reach discharge at every day boundary into a state-cache netCDF.
///
/// The cache allows training windows to start from realistic in-sequence
/// routing states instead of the synthetic hotstart heuristic. Each cache row
/// `d` holds the discharge vector entering day `d` (hour 0):
///
/// - Row 0: discharge after the first routing step (hour 1 of day 0). This
///   approximates the hotstart vector; the error is one hourly routing step
///   from the true initial condition, which affects only training windows
///   starting exactly at day 0.
/// - Row d (d ≥ 1): discharge after the last step of day d−1 (the exact
///   state entering day d).
///
/// Mirrors `run_teacher`'s chunked loop (carry_state = chunk_idx > 0) but
/// without plants/obs/zeta overhead.
fn run_state_cache<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    // Must match run_teacher's BATCH_SIZE_DAYS so state boundaries align.
    const BATCH_SIZE_DAYS: usize = 365;

    let output = cli.output.as_ref().ok_or("--output is required in state-cache mode")?;

    // Pre-flight: refuse to overwrite an existing non-empty file.
    if output.exists() {
        let meta = std::fs::metadata(output)?;
        if meta.len() > 0 {
            return Err(format!(
                "--output {} already exists and is non-empty; remove it before \
                 re-running state-cache mode to prevent stale cache data",
                output.display()
            )
            .into());
        }
    }

    let checkpoint = cli.checkpoint.as_ref().ok_or("--checkpoint is required in state-cache mode")?;

    // Head on the inner backend — forward-only, no autograd.
    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = kan_config(head_section, cfg.seed);
    let head_template: KanHead<I> = head_cfg.init::<I>(&device);
    let head = load_kan_head::<I>(&head_base(checkpoint), head_template, &device)?;
    eprintln!("loaded checkpoint: {}", head_base(checkpoint).display());

    let dataset = MeritGagesDataset::open(&cfg)?;
    let axis = dataset.time_axis().clone();
    let n_days = axis.num_days.min(cli.eval_days);
    assert!(
        n_days >= 1,
        "eval window too short: {n_days} days (need >= 1)"
    );

    // 1-day probe: sizes the network (static across chunks) and gets COMID order.
    let probe = TestWindow::new(&axis, 0, 1);
    let probe_batch = dataset.collate_window(&probe)?;
    let n_all_gauges = probe_batch.gauge_staids.len();
    let network_comids: Vec<i64> = probe_batch.divide_comids.iter().map(|c| c.0).collect();
    let n_reaches = network_comids.len();

    eprintln!(
        "state-cache: {n_all_gauges} gauges, {n_reaches} reaches, {n_days} days \
         ({} chunks of {BATCH_SIZE_DAYS})",
        n_days.div_ceil(BATCH_SIZE_DAYS)
    );

    // Pre-allocate the full cache: (n_days, n_reaches) row-major f32.
    let mut q_state = vec![0.0f32; n_days * n_reaches];

    let day0_str = axis.start.format("%Y-%m-%d").to_string();
    let checkpoint_label = checkpoint.display().to_string();

    // Chunked continuous forward. State continuity is maintained by injecting
    // the final discharge vector from each chunk as `tensors.initial_state` for
    // the next chunk (via `setup_inputs`'s `Some(q0_ext)` branch, which takes
    // priority over the carry_state / hotstart path). A fresh MuskingumCunge
    // engine is created per chunk, so `carry_state = true` alone has no effect
    // (discharge_t is always None on a fresh engine). The explicit injection
    // here is the correct mechanism.
    let n_chunks_total = n_days.div_ceil(BATCH_SIZE_DAYS);
    let mut day_offset = 0usize;
    let mut chunk_idx = 0usize;
    // Final discharge from the previous chunk (per-reach, network column order).
    // None for the first chunk → hotstart runs; Some for subsequent chunks →
    // injected as initial_state so routing continues exactly from where the
    // previous chunk left off.
    let mut prev_final_state: Option<Vec<f32>> = None;
    while day_offset < n_days {
        let chunk_n = (n_days - day_offset).min(BATCH_SIZE_DAYS);
        let win = TestWindow::new(&axis, day_offset, chunk_n);
        // Disagg lookahead: same fix as run_teacher — peek one extra daily Q' row
        // so the state entering the next chunk comes from a consistent disagg that
        // agrees with the floor's continuous-window routing. Without this, the state
        // cache at every chunk boundary reflects the wrong right-clamped disagg,
        // which the floor's continuous routing then diverges from.
        let has_next_chunk = day_offset + chunk_n < n_days;
        let batch = if has_next_chunk {
            dataset.collate_window_with_daily_lookahead(&win)?
        } else {
            dataset.collate_window(&win)?
        };
        assert!(
            batch.divide_comids.iter().map(|c| c.0).eq(network_comids.iter().copied()),
            "reach column order changed between chunks — COMID index invalid"
        );
        let mut tensors = batch.to_tensors::<I>(&device);

        // Inject the previous chunk's final state for cross-chunk continuity.
        // Chunk 0: prev_final_state = None → hotstart runs (correct cold start).
        // Chunk k>0: inject so routing continues from the exact state where
        // chunk k-1 ended, producing truly continuous day-boundary states.
        if let Some(ref state_vec) = prev_final_state {
            let n = state_vec.len();
            tensors.initial_state = Some(Tensor::<I, 1>::from_data(
                TensorData::new(state_vec.clone(), [n]),
                &device,
            ));
        }

        // Per-reach hourly discharge: (n_reaches, chunk_n * 24).
        // carry_state=false: continuity is provided by tensors.initial_state above.
        let runoff = forward_eval_reaches::<I>(&cfg, &tensors, &head, &device, false, None, None);
        let chunk_hours = chunk_n * 24;
        let runoff_vec: Vec<f32> = runoff.into_data().into_vec().unwrap();

        // Save the last column of this chunk as the initial state for the next chunk.
        // Column (chunk_hours - 1) is the discharge after the final routing step,
        // consistent with what q_state[day_offset + chunk_n] captures below.
        prev_final_state = Some(
            (0..n_reaches)
                .map(|r| runoff_vec[r * chunk_hours + (chunk_hours - 1)])
                .collect(),
        );

        // Row 0 of the cache ≈ hotstart: captured as column 0 of the first
        // chunk (discharge after the first routing step). This is one hourly
        // step later than the true initial condition — simpler than extracting
        // the pre-routing hotstart vector (internal to forward_eval_core) and
        // affects only training windows starting exactly on day 0. The plan
        // flagged day-0 indexing as a known unknown; this is the documented
        // resolution (see module doc).
        if chunk_idx == 0 {
            for r in 0..n_reaches {
                q_state[r] = runoff_vec[r * chunk_hours];
            }
        }

        // Capture states entering days day_offset+1 through day_offset+chunk_n.
        // State entering global day d = last step of day d−1 = local hourly
        // column (d − day_offset)*24 − 1.
        // Only store rows within [1, n_days): row n_days would require one
        // more routing step beyond the eval window.
        for k in 1..=chunk_n {
            let global_day = day_offset + k;
            if global_day >= n_days {
                break;
            }
            let local_col = k * 24 - 1;
            for r in 0..n_reaches {
                q_state[global_day * n_reaches + r] = runoff_vec[r * chunk_hours + local_col];
            }
        }

        eprintln!(
            "  chunk {}/{n_chunks_total}: days {day_offset}..{}",
            chunk_idx + 1,
            day_offset + chunk_n
        );
        day_offset += chunk_n;
        chunk_idx += 1;
    }

    write_state_cache_netcdf(output, n_days, &network_comids, &q_state, &day0_str, &checkpoint_label)?;

    println!(
        "wrote {} ({n_days} days x {n_reaches} reaches)",
        output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify write_floor_netcdf round-trips: 3-D abs_residual, 2-D string
    /// gauge_staid, 2-D uparea, NaN preservation, and dim layout.
    #[test]
    fn floor_netcdf_roundtrip() {
        let dir = std::env::temp_dir().join("probe_floor_nc_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("floor_test.nc");
        let _ = std::fs::remove_file(&path);

        // 2 windows x 2 gauges x 3 days of |pred - obs|; NaN = filtered gauge-day.
        let resid = vec![
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, // window 0: gauge rows
            0.7, f32::NAN, 0.9, 1.0, 1.1, 1.2, // window 1
        ];
        let staids = vec![
            "01010000".into(), "02020000".into(),
            "01010000".into(), "03030000".into(),
        ];
        let uparea = vec![100.0f32, 2000.0, 100.0, 55000.0];
        write_floor_netcdf(&path, 2, 2, 3, &resid, &staids, &uparea, 42, "ckpt").unwrap();

        let f = netcdf::open(&path).unwrap();
        let v = f.variable("abs_residual").unwrap();
        assert_eq!(
            v.dimensions().iter().map(|d| d.len()).collect::<Vec<_>>(),
            vec![2, 2, 3]
        );
        let vals: Vec<f32> = v.get_values(..).unwrap();
        assert_eq!(vals[0], 0.1);
        assert!(vals[7].is_nan());
        let ua = f.variable("uparea").unwrap();
        let uv: Vec<f32> = ua.get_values(..).unwrap();
        assert_eq!(uv[3], 55000.0);
    }

    /// Verify write_state_cache_netcdf round-trips: (day, COMID) q_state grid,
    /// COMID coordinate, day0 global attribute, and dim layout.
    #[test]
    fn state_cache_netcdf_roundtrip() {
        let dir = std::env::temp_dir().join("probe_state_cache_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache_test.nc");
        let _ = std::fs::remove_file(&path);

        // 3 days x 2 reaches of day-boundary discharge states.
        let q = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let comids = vec![10i64, 20];
        write_state_cache_netcdf(&path, 3, &comids, &q, "1981-10-01", "ckpt").unwrap();

        let f = netcdf::open(&path).unwrap();
        let v = f.variable("q_state").unwrap();
        assert_eq!(
            v.dimensions().iter().map(|d| d.len()).collect::<Vec<_>>(),
            vec![3, 2]
        );
        let vals: Vec<f32> = v.get_values(..).unwrap();
        assert_eq!(vals, q);
        let c = f.variable("COMID").unwrap();
        let cv: Vec<i64> = c.get_values(..).unwrap();
        assert_eq!(cv, comids);
        let d0: String = f.attribute("day0").unwrap().value().unwrap().try_into().unwrap();
        assert_eq!(d0, "1981-10-01");
    }

    /// The task-6 design note flagged NC_STRING support as the one uncertain
    /// API — prove the round file round-trips (f32 grid + string coordinate).
    #[test]
    fn round_netcdf_roundtrip_with_string_gauges() {
        let dir = std::env::temp_dir().join("probe_perturb_nc_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("round_test.nc");
        let _ = std::fs::remove_file(&path);

        let preds =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let staids = vec!["01010000".to_string(), "USGS__02020000".to_string()];
        write_round_netcdf(&path, &staids, &preds, "1981-10-02", 1095).unwrap();

        let f = netcdf::open(&path).unwrap();
        let v = f.variable("predictions").unwrap();
        assert_eq!(v.dimensions()[0].len(), 2);
        assert_eq!(v.dimensions()[1].len(), 3);
        let vals: Vec<f32> = v.get_values(..).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let g = f.variable("gauge_staid").unwrap();
        assert_eq!(g.get_string(0).unwrap(), "01010000");
        assert_eq!(g.get_string(1).unwrap(), "USGS__02020000");
    }
}
