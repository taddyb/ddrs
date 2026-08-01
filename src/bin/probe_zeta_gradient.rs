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
//! Stage 3 (`--mode teacher`): synthetic-twin ground-truth generator — runs
//! the chunked eval loop and writes synthetic daily gauge observations as a
//! zarr-v2 store. Two INDEPENDENT, orthogonal overrides may be combined or
//! used alone:
//!   (a) planted-leakance world (`--plant-file` + `--zeta-output`) — overrides
//!       the KAN head's normalized leakance outputs at specified reaches with
//!       values from a CSV; also writes a per-reach zeta answer-key netCDF.
//!       Requires `params.use_leakance: true`.
//!   (b) routing-parameter donor world (`--donor-params-nc`, docs:
//!       docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md)
//!       — overrides ALL THREE of n/q_spatial/p_spatial from a
//!       `dump_parameters::write_netcdf`-schema donor NetCDF (the same
//!       mechanism `--mode eval-loss`'s "full-swap" composition uses).
//! `--output` is not used in teacher mode.
//!
//! Leakance-only (original usage, unchanged):
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode teacher \
//!       --config config/experiments/leakance_hourly_on.yaml \
//!       --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
//!       --eval-days 1095 \
//!       --plant-file output/plant_sites.csv \
//!       --obs-output output/teacher_obs/ \
//!       --zeta-output output/teacher_zeta.nc
//!
//! Routing-parameter donor only (no leakance, no --plant-file/--zeta-output):
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode teacher --backend cpu \
//!       --config config/experiments/synthetic_n_teacher.yaml \
//!       --checkpoint .ddrs/runs/2026-07-16T02-22-14Z-train-and-test/checkpoints/epoch_5_mb_35 \
//!       --eval-days 999999 \
//!       --donor-params-nc output/synthetic_n/truth_leopold_maddock.nc \
//!       --obs-output output/synthetic_n/synthetic_obs/
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
//!
//! Stage 6 (`--mode eval-loss`): H5/H6 selective-equifinality parameter-swap
//! test (docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md).
//! Samples the SAME deterministic rho-window/gauge plan as floor/grad mode,
//! then evaluates the training L1 loss over that fixed plan under each of
//! `own`/`n-swap`/`geo-swap`/`full-swap` — injecting `n`/`q_spatial`/
//! `p_spatial` from `--donor-params-nc` (a `dump_parameters::write_netcdf`
//! dump) in place of the checkpoint's own KAN-head output. `--checkpoint` is
//! REQUIRED; `--loss-output` writes a tidy CSV (`composition,window,
//! mean_loss`). Optional `--per-gauge-output` additionally writes a
//! per-gauge tidy CSV (`composition,window,staid,gauge_loss`) — one row per
//! surviving gauge per window per composition, for per-gauge distributions /
//! DA-stratified medians (the aggregate CSV above cannot be un-averaged back
//! into per-gauge values).
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode eval-loss --backend cpu \
//!       --config .ddrs/runs/<r1-id>/config.yaml \
//!       --checkpoint .ddrs/runs/<r1-id>/checkpoints/epoch_5_mb_9 \
//!       --donor-params-nc output/equif/R3_kan_parameters.nc \
//!       --windows 96 --seed 42 \
//!       --loss-output output/equif/h5_r1_under_r1.csv
//!
//! Stage 7 (`--mode landscape`): H6 loss-landscape grid scan + linear barrier
//! (same spec, "H6 — Forcing-indexed valley"). BOTH arms' `dump_parameters`
//! NetCDFs are ALWAYS required (`--params-nc-a`, `--params-nc-b`) regardless
//! of which arm's forcing `--config`/`--checkpoint` select — the anchor field
//! is derived from both dumps and is identical across the two per-forcing
//! invocations. Two measurements, each optional via its output flag (at least
//! one required):
//!
//! - Surface (`--surface-output`, CSV `log2_alpha,log2_beta,window,mean_loss`):
//!   evaluates the (α, β)-scaled arm-mean ANCHOR field on a `--grid-n` ×
//!   `--grid-n` grid over (log2 α, log2 β) ∈ [-1.5, 1.5]². Anchor = per-COMID
//!   arm mean of the two dumps — geometric mean for config-log-space fields,
//!   arithmetic otherwise (the same per-field convention `denormalize`/
//!   `physical_to_normalized` use). n is scaled by α = 2^log2_alpha,
//!   p_spatial by β = 2^log2_beta; q_spatial is HELD at the anchor value at
//!   every grid point (only 2 axes are scanned). `--single-point
//!   "<log2_alpha>,<log2_beta>"` restricts the surface to one point (the
//!   full-96-window re-evaluation of the grid argmin).
//! - Barrier (`--barrier-output`, CSV `t,window,mean_loss`): evaluates
//!   θ(t) = exp((1−t)·ln θ_A + t·ln θ_B) per-COMID for ALL THREE of
//!   n/q_spatial/p_spatial (log-space interpolation for all three regardless
//!   of each field's own log/linear config flag — the registered formula,
//!   deliberate) at t = i/(`--barrier-points`−1). t=0 reproduces arm A's own
//!   dump exactly, t=1 arm B's. The barrier statistic printed at the end is
//!   informational; the authoritative statistics/verdicts come from the
//!   Python analysis step.
//!
//! The window/gauge plan is sampled ONCE per (config, seed) — the same
//! Phase-1 machinery as eval-loss — and replayed unchanged across every
//! grid/barrier point. Registered H6 runs pass `--windows 16` explicitly
//! (the CLI default is 32) and leave `--grid-n`/`--barrier-points` at their
//! registered defaults (11, 21); smaller values exist for smoke tests only.
//! The split-half noise floor is a re-invocation with a different `--seed`.
//!   cargo run --release --bin probe_zeta_gradient -- \
//!       --mode landscape --backend cpu \
//!       --config .ddrs/runs/<r1-id>/config.yaml \
//!       --checkpoint .ddrs/runs/<r1-id>/checkpoints/epoch_5_mb_35 \
//!       --params-nc-a output/equif/R1_kan_parameters.nc \
//!       --params-nc-b output/equif/R3_kan_parameters.nc \
//!       --windows 16 --seed 42 \
//!       --surface-output output/equif/h6/r1_surface.csv \
//!       --barrier-output output/equif/h6/r1_barrier.csv

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
use ddrs::data::{load_comid_field, GageMetadata, RhoWindow, Staid};
use ddrs::data::sampler::{BatchSource, RandomSampler};
use ddrs::data::TestWindow;
use ddrs::dump_parameters::{write_grad_netcdf, write_param_grad_netcdf, write_zeta_netcdf};
use ddrs::nn::kan_head::KanHead;
use ddrs::training::checkpoint::{head_base, load_kan_head};
use ddrs::training::probe::{probe_forward, GradAccum};
use ddrs::training::{
    batch_loss, filter_nan_gauges, forward_eval, forward_eval_reaches, l1_loss_post_warmup,
    physical_to_normalized, scatter_add_by_group, tau_trim_and_downsample, FilteredPair,
    LeakanceOverride, RoutingParamOverride, ZetaSums,
};

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
    /// Optional — omit together with --zeta-output when only overriding
    /// n/q_spatial/p_spatial via --donor-params-nc (no leakance planting).
    #[arg(long)]
    plant_file: Option<PathBuf>,

    /// teacher mode: directory for the synthetic-obs zarr-v2 store. Always
    /// required in teacher mode regardless of which override(s) are active.
    #[arg(long)]
    obs_output: Option<PathBuf>,

    /// teacher mode: answer-key netCDF (zeta accumulation over the window).
    /// Required IFF --plant-file is given (leakance-planting world only);
    /// omit both together for a routing-parameter-donor-only teacher run.
    #[arg(long)]
    zeta_output: Option<PathBuf>,

    /// teacher mode: continuous-forward chunk length in days. The default
    /// (365) peaks at ~65 GB RSS over the 64,892-reach eval network — on a
    /// 93 GB desktop with ambient apps this gets OOM-killed; 180 halves the
    /// peak at the cost of doubling disagg boundary-artifact density
    /// (0.55% → 1.1% of days over the 29-year window; artifacts stay rare).
    /// State continuity across chunks is exact either way.
    #[arg(long, default_value_t = 365)]
    chunk_days: usize,

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

    /// eval-loss mode: donor NetCDF (dump_parameters::write_netcdf schema,
    /// COMID-keyed, physical units) supplying n/q_spatial/p_spatial for any
    /// composition other than "own". Required unless --compositions is "own"
    /// only.
    ///
    /// teacher mode: same donor-NetCDF schema, but ALL THREE of
    /// n/q_spatial/p_spatial are always overridden together (no partial
    /// swap) — the synthetic-twin ground-truth generator (see
    /// docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md).
    /// Optional; independent of --plant-file/--zeta-output.
    #[arg(long)]
    donor_params_nc: Option<PathBuf>,

    /// eval-loss mode: comma-separated compositions to evaluate, from
    /// {own, n-swap, geo-swap, full-swap}. Default: all four.
    #[arg(long)]
    compositions: Option<String>,

    /// eval-loss mode: output CSV path (tidy long format:
    /// composition,window,mean_loss).
    #[arg(long)]
    loss_output: Option<PathBuf>,

    /// eval-loss mode: optional per-gauge output CSV (tidy long format:
    /// composition,window,staid,gauge_loss). Absent by default — the
    /// existing --loss-output CSV schema is unchanged either way.
    #[arg(long)]
    per_gauge_output: Option<PathBuf>,

    /// landscape mode: arm A's parameter dump (dump_parameters::write_netcdf
    /// schema, COMID-keyed, physical units). ALWAYS required in landscape
    /// mode — together with --params-nc-b it defines the anchor field
    /// (surface) and the t=0 barrier endpoint, independent of which arm's
    /// forcing --config/--checkpoint select.
    #[arg(long)]
    params_nc_a: Option<PathBuf>,

    /// landscape mode: arm B's parameter dump (see --params-nc-a); the t=1
    /// barrier endpoint.
    #[arg(long)]
    params_nc_b: Option<PathBuf>,

    /// landscape mode: surface CSV output (tidy long format:
    /// log2_alpha,log2_beta,window,mean_loss). Enables the (α, β) grid scan.
    #[arg(long)]
    surface_output: Option<PathBuf>,

    /// landscape mode: barrier CSV output (tidy long format:
    /// t,window,mean_loss). Enables the linear (log-space) barrier scan
    /// between the two arms' own fields.
    #[arg(long)]
    barrier_output: Option<PathBuf>,

    /// landscape mode: evaluate a SINGLE grid point "log2_alpha,log2_beta"
    /// instead of the full grid (e.g. the 16-window grid's argmin re-checked
    /// at --windows 96). Requires --surface-output.
    #[arg(long)]
    single_point: Option<String>,

    /// landscape mode: grid resolution per axis over [-1.5, 1.5]. The
    /// registered H6 protocol uses the default 11 (11×11); smaller values
    /// (e.g. 3) exist for smoke tests only.
    #[arg(long, default_value_t = 11)]
    grid_n: usize,

    /// landscape mode: number of barrier t-points over [0, 1]. The
    /// registered H6 protocol uses the default 21 (t = 0, 0.05, …, 1);
    /// smaller values exist for smoke tests only.
    #[arg(long, default_value_t = 21)]
    barrier_points: usize,
}

#[derive(Copy, Clone, PartialEq)]
enum Mode {
    Grad,
    Perturb,
    Teacher,
    Floor,
    StateCache,
    EvalLoss,
    Landscape,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let mode = match cli.mode.as_str() {
        "grad" => Mode::Grad,
        "perturb" => Mode::Perturb,
        "teacher" => Mode::Teacher,
        "floor" => Mode::Floor,
        "state-cache" => Mode::StateCache,
        "eval-loss" => Mode::EvalLoss,
        "landscape" => Mode::Landscape,
        other => {
            return Err(
                format!(
                    "unknown --mode {other} \
                     (expected \"grad\", \"perturb\", \"teacher\", \"floor\", \"state-cache\", \
                     \"eval-loss\", or \"landscape\")"
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

    // --compositions/--loss-output/--per-gauge-output are only valid in
    // eval-loss mode. --donor-params-nc is ALSO valid in teacher mode (the
    // synthetic-n routing-parameter donor override).
    if (cli.compositions.is_some() || cli.loss_output.is_some() || cli.per_gauge_output.is_some())
        && mode != Mode::EvalLoss
    {
        return Err(format!(
            "--compositions/--loss-output/--per-gauge-output are only \
             valid in --mode eval-loss (got --mode {})",
            cli.mode
        )
        .into());
    }
    if cli.donor_params_nc.is_some() && mode != Mode::EvalLoss && mode != Mode::Teacher {
        return Err(format!(
            "--donor-params-nc is only valid in --mode eval-loss or --mode teacher (got --mode {})",
            cli.mode
        )
        .into());
    }

    // --params-nc-a/--params-nc-b/--surface-output/--barrier-output/
    // --single-point are only valid in landscape mode.
    if (cli.params_nc_a.is_some()
        || cli.params_nc_b.is_some()
        || cli.surface_output.is_some()
        || cli.barrier_output.is_some()
        || cli.single_point.is_some())
        && mode != Mode::Landscape
    {
        return Err(format!(
            "--params-nc-a/--params-nc-b/--surface-output/--barrier-output/--single-point \
             are only valid in --mode landscape (got --mode {})",
            cli.mode
        )
        .into());
    }

    // grad/floor/eval-loss/landscape: training-mode config (probe batches
    // replicate the training sampler). perturb/teacher/state-cache:
    // testing-mode config (replicates the eval loop over the test window).
    let cfg_mode = match mode {
        Mode::Grad | Mode::Floor | Mode::EvalLoss | Mode::Landscape => ConfigMode::Training,
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
                Mode::EvalLoss => run_eval_loss::<I>(cfg, cli, device),
                Mode::Landscape => run_landscape::<I>(cfg, cli, device),
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
                Mode::EvalLoss => run_eval_loss::<I>(cfg, cli, device),
                Mode::Landscape => run_landscape::<I>(cfg, cli, device),
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

        let loss = batch_loss(p_filt, o_filt, &exp.loss, None);
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

            let pred =
                forward_eval::<I>(&cfg, &tensors, &head, &device, chunk_idx > 0, None, None, None);
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
    // Peak RSS scales with chunk length (~65 GB at C=365 on the 64,892-reach eval
    // network) — --chunk-days trades artifact density for memory headroom.
    let batch_size_days = cli.chunk_days;
    assert!(batch_size_days >= 3, "--chunk-days too small: {batch_size_days} (need >= 3)");

    let plants = match &cli.plant_file {
        Some(p) => parse_plant_file(p)?,
        None => Vec::new(),
    };
    let leakance_active = !plants.is_empty();
    if leakance_active {
        assert!(
            cfg.params.use_leakance,
            "teacher requires params.use_leakance: true when --plant-file is given"
        );
    }

    let obs_dir = cli.obs_output.as_ref().ok_or("--obs-output is required in teacher mode")?;
    if obs_dir.exists() && obs_dir.read_dir()?.next().is_some() {
        return Err(format!(
            "--obs-output {} already exists and is non-empty; remove it before \
             re-running teacher mode to prevent stale gauge data",
            obs_dir.display()
        )
        .into());
    }
    let zeta_path: Option<PathBuf> = if leakance_active {
        let p = cli
            .zeta_output
            .as_ref()
            .ok_or("--zeta-output is required in teacher mode when --plant-file is given")?;
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(
                    format!("--zeta-output parent dir does not exist: {}", parent.display())
                        .into(),
                );
            }
        }
        Some(p.clone())
    } else {
        None
    };
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

    let leakance_ov: Option<LeakanceOverride> = if leakance_active {
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
        Some(ov)
    } else {
        None
    };

    // Optional n/q_spatial/p_spatial donor override (the synthetic-n
    // routing-parameter twin — docs/superpowers/specs/2026-07-22-synthetic-n-
    // recoverability-design.md). Reuses the same --donor-params-nc /
    // load_comid_field / gather_by_comid / physical_to_normalized machinery
    // as --mode eval-loss's full-swap composition.
    let param_ov: Option<RoutingParamOverride> = match &cli.donor_params_nc {
        Some(donor_path) => {
            let log_space =
                |name: &str| cfg.params.log_space_parameters.iter().any(|s| s == name);
            let n_map = load_comid_field(donor_path, "n")?;
            let q_map = load_comid_field(donor_path, "q_spatial")?;
            let p_map = load_comid_field(donor_path, "p_spatial")?;
            let n_vals = gather_by_comid(&n_map, &network_comids)?;
            let q_vals = gather_by_comid(&q_map, &network_comids)?;
            let p_vals = gather_by_comid(&p_map, &network_comids)?;
            Some(RoutingParamOverride {
                n: Some(physical_to_normalized(
                    &n_vals,
                    cfg.params.parameter_ranges.n,
                    log_space("n"),
                )),
                q_spatial: Some(physical_to_normalized(
                    &q_vals,
                    cfg.params.parameter_ranges.q_spatial,
                    log_space("q_spatial"),
                )),
                p_spatial: Some(physical_to_normalized(
                    &p_vals,
                    cfg.params.parameter_ranges.p_spatial,
                    log_space("p_spatial"),
                )),
            })
        }
        None => None,
    };

    // Chunked CONTINUOUS forward with overrides + zeta accumulation. Continuity
    // across chunks is maintained by injecting the previous chunk's final
    // per-reach discharge as `tensors.initial_state` (the `carry_state` flag is
    // a no-op through forward_eval — a fresh engine is constructed per call, so
    // its `discharge_t` is always None and the hotstart always ran; this bug
    // made the pre-2026-07-04 synthetic obs restart from hotstart every 15 days,
    // leaving 15-day-periodic discontinuities 10-15x the day-to-day baseline on
    // large basins). The per-reach forward is used so the final column can be
    // extracted; gauge aggregation happens here via scatter_add_by_group.
    let mut zeta_sink: Option<ZetaSums<I>> = leakance_active.then(ZetaSums::<I>::new);
    let mut predictions_full = Array2::<f32>::zeros((n_all_gauges, n_hours));
    let n_chunks_total = n_days.div_ceil(batch_size_days);
    let mut day_offset = 0usize;
    let mut chunk_idx = 0usize;
    let mut prev_final_state: Option<Vec<f32>> = None;
    while day_offset < n_days {
        let chunk_n = (n_days - day_offset).min(batch_size_days);
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
            zeta_sink.as_mut(),
            leakance_ov.as_ref(),
            param_ov.as_ref(),
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

    // Answer key: zeta means over the routed window (only when leakance was
    // planted — a routing-parameter-donor-only teacher run has no zeta term
    // to report).
    if let Some(sink) = zeta_sink {
        let zeta_path = zeta_path.expect("zeta_path is set whenever leakance_active");
        let scale = 1.0_f32 / sink.steps as f32;
        assert!(scale.is_finite() && scale > 0.0, "zeta accumulation empty — leakance inactive?");
        let mean_vec = |t: Option<Tensor<I, 1>>| -> Vec<f32> {
            (t.expect("zeta sums present") * scale).into_data().into_vec().unwrap()
        };
        write_zeta_netcdf(
            &zeta_path,
            &network_comids,
            &mean_vec(sink.abs_sum),
            &mean_vec(sink.net_sum),
            &mean_vec(sink.depth_sum),
            &mean_vec(sink.area_z_sum),
            &mean_vec(sink.q_sum),
            &format!("teacher:{}", checkpoint.display()),
        )
        .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        println!("answer key → {} ({} reaches)", zeta_path.display(), network_comids.len());
    } else {
        println!("no --plant-file given — skipping zeta answer-key write");
    }
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
            forward_eval::<I>(&cfg, &tensors, &head, &device, false, None, None, None);
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
// Stage 6: --mode eval-loss (H5/H6 selective-equifinality parameter swap)
// ---------------------------------------------------------------------------

/// The four H5/H6 parameter compositions
/// (`docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md`).
/// `own` needs no donor file; the other three swap a subset of
/// `n`/`q_spatial`/`p_spatial` in from `--donor-params-nc`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Composition {
    Own,
    NSwap,
    GeoSwap,
    FullSwap,
}

impl Composition {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "own" => Ok(Self::Own),
            "n-swap" => Ok(Self::NSwap),
            "geo-swap" => Ok(Self::GeoSwap),
            "full-swap" => Ok(Self::FullSwap),
            other => Err(format!(
                "unknown composition '{other}' (expected own, n-swap, geo-swap, full-swap)"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Own => "own",
            Self::NSwap => "n-swap",
            Self::GeoSwap => "geo-swap",
            Self::FullSwap => "full-swap",
        }
    }

    fn needs_donor(self) -> bool {
        !matches!(self, Self::Own)
    }

    /// `None` = no override (the "own" composition, the true
    /// checkpoint-native baseline); `Some` selects which of the
    /// already-gathered-and-normalized donor fields replace this window's
    /// head output.
    fn build_override(
        self,
        donor_n: &Option<Vec<f32>>,
        donor_q: &Option<Vec<f32>>,
        donor_p: &Option<Vec<f32>>,
    ) -> Option<RoutingParamOverride> {
        match self {
            Self::Own => None,
            Self::NSwap => Some(RoutingParamOverride {
                n: donor_n.clone(),
                q_spatial: None,
                p_spatial: None,
            }),
            Self::GeoSwap => Some(RoutingParamOverride {
                n: None,
                q_spatial: donor_q.clone(),
                p_spatial: donor_p.clone(),
            }),
            Self::FullSwap => Some(RoutingParamOverride {
                n: donor_n.clone(),
                q_spatial: donor_q.clone(),
                p_spatial: donor_p.clone(),
            }),
        }
    }
}

/// Gather a COMID-keyed donor map into a batch's reach column order.
/// Hard-errors on a missing COMID: a missing routing parameter here means
/// the donor dump doesn't actually cover this arm's network, which would
/// silently corrupt the routing solve if allowed through (unlike attribute
/// NaN-fill, which handles genuinely absent coverage).
fn gather_by_comid(donor: &HashMap<i64, f32>, comids: &[i64]) -> Result<Vec<f32>, String> {
    comids
        .iter()
        .map(|c| {
            donor
                .get(c)
                .copied()
                .ok_or_else(|| format!("COMID {c} missing from donor NetCDF"))
        })
        .collect()
}

/// Post-tau-trim daily length for a fixed `rho`-day window. Mirrors
/// `tau_trim_and_downsample`'s arithmetic without running the tensor op:
/// `n_hourly = (rho - 1) * 24` depends only on `rho` (`RhoWindow::n_hourly`),
/// so this is constant across every window at a fixed `rho` and lets the
/// plan-building pass (Phase 1 below) check obs validity without a routing
/// forward pass.
fn trimmed_days(rho: usize, tau: u32) -> usize {
    let n_hourly = (rho - 1) * 24;
    let start = 13 + tau as usize;
    let end = n_hourly - 11 + tau as usize;
    (end - start) / 24
}

fn write_loss_csv(
    path: &Path,
    rows: &[(String, usize, f32)],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "composition,window,mean_loss")?;
    for (composition, window, loss) in rows {
        writeln!(f, "{composition},{window},{loss}")?;
    }
    Ok(())
}

/// Column-wise mean absolute error: for each surviving gauge column, its own
/// mean `|pred - obs|` over time — the per-gauge counterpart of
/// `l1_loss_post_warmup`, which instead averages over ALL columns into one
/// scalar. Both arrays are `(T_days, G_kept)` (the `FilteredPair` layout
/// `filter_nan_gauges` produces).
fn per_gauge_l1(predictions: &Array2<f32>, observations: &Array2<f32>) -> Vec<f32> {
    let (t_days, g) = predictions.dim();
    assert_eq!(observations.dim(), (t_days, g));
    (0..g)
        .map(|j| {
            let col_p = predictions.column(j);
            let col_o = observations.column(j);
            col_p
                .iter()
                .zip(col_o.iter())
                .map(|(p, o)| (p - o).abs())
                .sum::<f32>()
                / (t_days as f32)
        })
        .collect()
}

/// Recover, in kept-column order, the `Staid` of each gauge that survived
/// `filter_nan_gauges`. `mask` has length = original G (pre-filter) and is
/// `true` at the columns that were kept, in the same relative order they
/// appear in the filtered arrays' columns.
fn kept_gauge_staids(all_staids: &[Staid], mask: &[bool]) -> Vec<Staid> {
    all_staids
        .iter()
        .zip(mask.iter())
        .filter(|(_, &kept)| kept)
        .map(|(s, _)| s.clone())
        .collect()
}

fn write_per_gauge_loss_csv(
    path: &Path,
    rows: &[(String, usize, Staid, f32)],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "composition,window,staid,gauge_loss")?;
    for (composition, window, staid, loss) in rows {
        writeln!(f, "{composition},{window},{staid},{loss}")?;
    }
    Ok(())
}

/// Sample the deterministic rho-window/gauge plan EXACTLY ONCE (Phase 1 of
/// eval-loss and landscape modes). Replicates `run_floor`/grad mode's
/// sampler (LOCAL ChaCha12 rng from `seed`, same draw sequence), keeping a
/// window iff at least one gauge has a NaN-free post-warmup obs record —
/// a rule that depends only on `batch.observations`, never on any parameter,
/// so the plan cannot diverge between the conditions later replayed over it
/// (mirrors grad mode's all-NaN skip rule, `run` above).
fn sample_window_plan(
    dataset: &MeritGagesDataset,
    batch_size: usize,
    rho: usize,
    warmup: usize,
    t_days: usize,
    windows: usize,
    seed: u64,
) -> Result<Vec<(Vec<Staid>, RhoWindow)>, Box<dyn std::error::Error>> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let mut sampler =
        BatchSource::Shuffle(RandomSampler::new(dataset.len(), batch_size, true));
    sampler.reshuffle(&mut rng);

    let mut plan: Vec<(Vec<Staid>, RhoWindow)> = Vec::with_capacity(windows);
    let mut processed = 0usize;
    let mut total = 0usize;
    while processed < windows {
        if total > 10 * windows {
            return Err(format!(
                "retry cap exceeded: sampled {total} batches but only {processed}/{windows} \
                 windows planned — check dataset NaN coverage"
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
        total += 1;

        let staids: Vec<Staid> = idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
        let window = dataset.time_axis().sample_rho_window(&mut rng, rho);
        let batch = dataset.collate(&staids, &window)?;

        assert!(
            batch.observations.nrows() >= 2 + t_days,
            "obs/pred shape mismatch: obs rows={} t_days={t_days}",
            batch.observations.nrows()
        );
        let surviving = (0..batch.gauge_staids.len()).any(|gi| {
            (warmup..t_days).all(|ti| !batch.observations[(ti + 1, gi)].is_nan())
        });
        if !surviving {
            eprintln!("  window skipped: all gauges have NaN in post-warmup window");
            continue;
        }

        plan.push((staids, window));
        processed += 1;
        eprintln!("plan {processed}/{windows}");
    }
    Ok(plan)
}

/// One forward-eval of an already-collated window under an optional
/// parameter override, reduced to the NaN-filtered (pred, obs) pair the L1
/// loss is computed from. Shared by eval-loss (H5) and landscape (H6):
/// `forward_eval` → tau-trim/daily downsample → post-warmup slice with the
/// training +1-day obs alignment (matches driver.rs and grad mode `run`,
/// above, exactly) → `filter_nan_gauges`. `predictions` are `(G, T_post)`
/// and `observations` `(T_post, G)` going into the filter, per its
/// convention; the returned pair is `(T_post, G_kept)`.
#[allow(clippy::too_many_arguments)]
fn eval_window_filtered<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    head: &KanHead<I>,
    device: &I::Device,
    obs_arr: &Array2<f32>,
    warmup: usize,
    t_days: usize,
    ov: Option<&RoutingParamOverride>,
) -> FilteredPair {
    let pred_hourly = forward_eval::<I>(cfg, tensors, head, device, false, None, None, ov);
    let daily = tau_trim_and_downsample(pred_hourly, cfg.params.tau);
    let dims = daily.dims();
    let daily_arr: Array2<f32> = {
        let flat: Vec<f32> = daily.into_data().into_vec().unwrap();
        Array2::from_shape_vec((dims[0], dims[1]), flat).unwrap()
    };

    let g = daily_arr.nrows();
    let pred_post = daily_arr.slice(ndarray::s![.., warmup..t_days]).to_owned();
    let mut obs_post = Array2::<f32>::zeros((t_days - warmup, g));
    for gi in 0..g {
        for ti in warmup..t_days {
            obs_post[(ti - warmup, gi)] = obs_arr[(ti + 1, gi)];
        }
    }

    filter_nan_gauges(&pred_post, &obs_post)
}

/// H5/H6 parameter-swap evaluation. Samples the SAME deterministic
/// rho-window/gauge plan as `run_floor`/grad mode (LOCAL ChaCha12 rng from
/// `--seed`), then replays that fixed plan unchanged across every requested
/// composition (`own`/`n-swap`/`geo-swap`/`full-swap`) — so any loss
/// difference between compositions is attributable only to the parameter
/// swap, never to different data. `RoutingTensors` is borrowed (not
/// consumed) by `forward_eval`, so one `dataset.collate` per window is
/// reused across all compositions rather than re-collated per composition.
///
/// The window/gauge plan is sampled ONCE, before any composition is even
/// considered — `sample_window_plan` decides which windows to keep using
/// only `batch.observations` (never routes anything), so the plan cannot
/// diverge between compositions: no parameter ever influences which windows
/// are drawn or skipped.
fn run_eval_loss<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = cli
        .checkpoint
        .as_ref()
        .ok_or("--checkpoint is required in eval-loss mode")?;
    let loss_output = cli
        .loss_output
        .as_ref()
        .ok_or("--loss-output is required in eval-loss mode")?;

    let compositions: Vec<Composition> = cli
        .compositions
        .as_deref()
        .unwrap_or("own,n-swap,geo-swap,full-swap")
        .split(',')
        .map(|s| Composition::parse(s.trim()))
        .collect::<Result<_, _>>()?;
    if compositions.is_empty() {
        return Err("--compositions must name at least one composition".into());
    }

    let needs_donor = compositions.iter().any(|c| c.needs_donor());
    let donor_maps = if needs_donor {
        let donor_path = cli
            .donor_params_nc
            .as_ref()
            .ok_or("--donor-params-nc is required unless --compositions is \"own\" only")?;
        Some((
            load_comid_field(donor_path, "n")?,
            load_comid_field(donor_path, "q_spatial")?,
            load_comid_field(donor_path, "p_spatial")?,
        ))
    } else {
        None
    };

    // Head on the inner backend I — forward-only, no autograd (same as floor/perturb/teacher).
    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = kan_config(head_section, cfg.seed);
    let head_template: KanHead<I> = head_cfg.init::<I>(&device);
    let head = load_kan_head::<I>(&head_base(checkpoint), head_template, &device)?;
    eprintln!("loaded checkpoint: {}", head_base(checkpoint).display());

    let dataset = MeritGagesDataset::open(&cfg)?;
    let exp = cfg.experiment.as_ref().expect("experiment section required");
    let rho = exp
        .rho
        .expect("eval-loss mode requires experiment.rho (training-style windows)");
    let warmup = exp.warmup;
    let t_days = trimmed_days(rho, cfg.params.tau);

    // H5/H6 report the "training L1 loss" specifically (spec:
    // docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md).
    // l1_loss_post_warmup below always computes L1 regardless of
    // experiment.loss.kind — assert the config agrees so a non-L1 config
    // can't silently produce numbers under the wrong objective.
    assert_eq!(
        exp.loss.kind,
        ddrs::config::LossKind::L1,
        "eval-loss mode always computes L1 (docs/superpowers/specs/2026-07-08-\
         landscape-hypotheses-h5-h6-draft.md); config has experiment.loss.kind={:?}",
        exp.loss.kind
    );

    // ----- Phase 1: sample the window/gauge plan EXACTLY ONCE -----
    let plan =
        sample_window_plan(&dataset, exp.batch_size, rho, warmup, t_days, cli.windows, cli.seed)?;

    // ----- Phase 2: replay the fixed plan across every composition -----
    let mut rows: Vec<(String, usize, f32)> = Vec::with_capacity(plan.len() * compositions.len());
    let mut per_gauge_rows: Vec<(String, usize, Staid, f32)> = Vec::new();
    for (window_idx, (staids, window)) in plan.iter().enumerate() {
        let batch = dataset.collate(staids, window)?;
        let comids: Vec<i64> = batch.divide_comids.iter().map(|c| c.0).collect();
        let obs_arr = batch.observations.clone();
        let gauge_staids = batch.gauge_staids.clone();
        let tensors = batch.to_tensors::<I>(&device);

        // Gather + normalize donor fields ONCE per window, reused by every
        // composition that needs them (not recomputed per composition).
        let log_space = |name: &str| cfg.params.log_space_parameters.iter().any(|s| s == name);
        let (donor_n, donor_q, donor_p) = match &donor_maps {
            Some((n_map, q_map, p_map)) => {
                let n_vals = gather_by_comid(n_map, &comids)?;
                let q_vals = gather_by_comid(q_map, &comids)?;
                let p_vals = gather_by_comid(p_map, &comids)?;
                (
                    Some(physical_to_normalized(
                        &n_vals,
                        cfg.params.parameter_ranges.n,
                        log_space("n"),
                    )),
                    Some(physical_to_normalized(
                        &q_vals,
                        cfg.params.parameter_ranges.q_spatial,
                        log_space("q_spatial"),
                    )),
                    Some(physical_to_normalized(
                        &p_vals,
                        cfg.params.parameter_ranges.p_spatial,
                        log_space("p_spatial"),
                    )),
                )
            }
            None => (None, None, None),
        };

        for &composition in &compositions {
            let ov = composition.build_override(&donor_n, &donor_q, &donor_p);
            let filtered = eval_window_filtered::<I>(
                &cfg,
                &tensors,
                &head,
                &device,
                &obs_arr,
                warmup,
                t_days,
                ov.as_ref(),
            );
            let n_kept = filtered.mask.iter().filter(|&&v| v).count();
            if n_kept == 0 {
                return Err(format!(
                    "window {window_idx} composition {}: all gauges NaN post-filter — \
                     Phase 1's validity check should have excluded this window",
                    composition.label()
                )
                .into());
            }
            let loss = l1_loss_post_warmup(&filtered.predictions, &filtered.observations, 0);
            rows.push((composition.label().to_string(), window_idx, loss));

            if cli.per_gauge_output.is_some() {
                let kept_staids = kept_gauge_staids(&gauge_staids, &filtered.mask);
                let gauge_losses = per_gauge_l1(&filtered.predictions, &filtered.observations);
                debug_assert_eq!(kept_staids.len(), gauge_losses.len());
                for (staid, gauge_loss) in kept_staids.into_iter().zip(gauge_losses) {
                    per_gauge_rows.push((
                        composition.label().to_string(),
                        window_idx,
                        staid,
                        gauge_loss,
                    ));
                }
            }
        }

        eprintln!(
            "window {}/{} evaluated ({} compositions)",
            window_idx + 1,
            plan.len(),
            compositions.len()
        );
    }

    write_loss_csv(loss_output, &rows)?;
    println!(
        "wrote {} ({} windows x {} compositions)",
        loss_output.display(),
        plan.len(),
        compositions.len()
    );

    if let Some(per_gauge_output) = &cli.per_gauge_output {
        write_per_gauge_loss_csv(per_gauge_output, &per_gauge_rows)?;
        println!(
            "wrote {} ({} rows: windows x compositions x kept-gauges, varies per window)",
            per_gauge_output.display(),
            per_gauge_rows.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 7: --mode landscape (H6 loss-landscape grid scan + linear barrier)
// ---------------------------------------------------------------------------

/// Evenly-spaced grid coordinates over [-1.5, 1.5] (per axis). `n = 11` is
/// the registered H6 resolution (spacing 0.3); smaller `n` is for smoke
/// tests. Computed in f64 then narrowed so interior points land on the
/// nearest-f32 of the exact decimal (clean CSV strings) and the endpoints
/// are exactly ±1.5.
fn grid_coords(n: usize) -> Vec<f32> {
    assert!(n >= 2, "--grid-n must be >= 2 (got {n})");
    (0..n)
        .map(|i| (-1.5 + i as f64 * 3.0 / (n - 1) as f64) as f32)
        .collect()
}

/// Barrier interpolation coordinates t = i/(n-1) over [0, 1]. `n = 21` is
/// the registered H6 resolution (t = 0, 0.05, …, 1). Endpoints are exactly
/// 0 and 1 (the special-cased own-field reproductions in `log_interp`).
fn barrier_ts(n: usize) -> Vec<f32> {
    assert!(n >= 2, "--barrier-points must be >= 2 (got {n})");
    (0..n).map(|i| (i as f64 / (n - 1) as f64) as f32).collect()
}

/// Per-COMID arm-mean of two dumps' fields — the H6 anchor θ̄. Geometric
/// mean `exp((ln a + ln b)/2)` when the field is config-flagged log-space,
/// arithmetic mean `(a+b)/2` otherwise — the SAME per-field convention
/// `denormalize`/`physical_to_normalized` use, read from
/// `params.log_space_parameters` at the call site (never hardcoded).
///
/// Only COMIDs present in BOTH maps are averaged: a one-sided COMID has no
/// defined arm mean, so it is SKIPPED here rather than erroring — the
/// downstream `gather_by_comid` hard-errors if a batch ever needs a skipped
/// COMID, so a coverage gap cannot silently pass through (the caller
/// reports the intersection/skip counts).
fn arm_mean_field(
    a: &HashMap<i64, f32>,
    b: &HashMap<i64, f32>,
    log_space: bool,
) -> HashMap<i64, f32> {
    a.iter()
        .filter_map(|(comid, &va)| {
            b.get(comid).map(|&vb| {
                let mean = if log_space {
                    assert!(
                        va > 0.0 && vb > 0.0,
                        "geometric arm-mean requires positive values \
                         (COMID {comid}: {va}, {vb})"
                    );
                    ((va.ln() + vb.ln()) / 2.0).exp()
                } else {
                    (va + vb) / 2.0
                };
                (*comid, mean)
            })
        })
        .collect()
}

/// Global multiplicative grid-point scaling: `v · 2^log2_factor` per COMID.
/// `log2_factor = 0` is the exact identity (`exp2(0) == 1`).
fn scale_by_log2(vals: &[f32], log2_factor: f32) -> Vec<f32> {
    let factor = log2_factor.exp2();
    vals.iter().map(|v| v * factor).collect()
}

/// The registered H6 barrier path: `θ(t) = exp((1−t)·ln a + t·ln b)`
/// per-COMID — log-space interpolation for EVERY field regardless of its own
/// log/linear config flag (the pre-registered formula, deliberate; do not
/// swap in per-field linear interpolation). `t = 0`/`t = 1` are
/// special-cased to return the endpoint fields bit-exactly (avoiding the
/// 1-ulp `exp(ln x)` wobble), so the barrier endpoints reproduce each arm's
/// own dump exactly.
fn log_interp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "log_interp length mismatch");
    if t == 0.0 {
        return a.to_vec();
    }
    if t == 1.0 {
        return b.to_vec();
    }
    a.iter()
        .zip(b.iter())
        .map(|(&va, &vb)| {
            assert!(
                va > 0.0 && vb > 0.0,
                "log_interp requires positive values (got {va}, {vb})"
            );
            ((1.0 - t) * va.ln() + t * vb.ln()).exp()
        })
        .collect()
}

/// Parse `--single-point "<log2_alpha>,<log2_beta>"`.
fn parse_single_point(s: &str) -> Result<(f32, f32), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err(format!(
            "--single-point expects \"<log2_alpha>,<log2_beta>\" (got '{s}')"
        ));
    }
    let la: f32 = parts[0]
        .trim()
        .parse()
        .map_err(|e| format!("--single-point log2_alpha '{}': {e}", parts[0].trim()))?;
    let lb: f32 = parts[1]
        .trim()
        .parse()
        .map_err(|e| format!("--single-point log2_beta '{}': {e}", parts[1].trim()))?;
    Ok((la, lb))
}

fn write_surface_csv(
    path: &Path,
    rows: &[(f32, f32, usize, f32)],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "log2_alpha,log2_beta,window,mean_loss")?;
    for (la, lb, window, loss) in rows {
        writeln!(f, "{la},{lb},{window},{loss}")?;
    }
    Ok(())
}

fn write_barrier_csv(
    path: &Path,
    rows: &[(f32, usize, f32)],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "t,window,mean_loss")?;
    for (t, window, loss) in rows {
        writeln!(f, "{t},{window},{loss}")?;
    }
    Ok(())
}

/// H6 loss-landscape driver (`--mode landscape`). Loads BOTH arms' dumps,
/// builds the anchor field (surface) and the endpoint fields (barrier),
/// samples the deterministic window plan ONCE via `sample_window_plan`, and
/// replays it unchanged across every grid/barrier point — per window, the
/// batch is collated and `to_tensors`'d once and the COMID gathers happen
/// once; each grid point is only a cheap scale-then-renormalize on top of
/// the gathered anchor vectors. All three of n/q_spatial/p_spatial are
/// overridden at every point (the head's own output never leaks into the
/// scanned surface), through the SAME `RoutingParamOverride` seam H5 uses —
/// so the loud p_spatial-missing panic in `forward_eval_core` is inherited.
fn run_landscape<I: Backend>(
    cfg: Config,
    cli: Cli,
    device: I::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = cli
        .checkpoint
        .as_ref()
        .ok_or("--checkpoint is required in landscape mode")?;
    let params_a = cli
        .params_nc_a
        .as_ref()
        .ok_or("--params-nc-a is required in landscape mode (both arms' dumps, always)")?;
    let params_b = cli
        .params_nc_b
        .as_ref()
        .ok_or("--params-nc-b is required in landscape mode (both arms' dumps, always)")?;
    if cli.surface_output.is_none() && cli.barrier_output.is_none() {
        return Err(
            "landscape mode requires --surface-output and/or --barrier-output".into(),
        );
    }
    let single_point = match &cli.single_point {
        Some(s) => {
            if cli.surface_output.is_none() {
                return Err("--single-point requires --surface-output".into());
            }
            Some(parse_single_point(s)?)
        }
        None => None,
    };

    // Head on the inner backend I — forward-only, no autograd (same as eval-loss).
    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = kan_config(head_section, cfg.seed);
    let head_template: KanHead<I> = head_cfg.init::<I>(&device);
    let head = load_kan_head::<I>(&head_base(checkpoint), head_template, &device)?;
    eprintln!("loaded checkpoint: {}", head_base(checkpoint).display());

    let dataset = MeritGagesDataset::open(&cfg)?;
    let exp = cfg.experiment.as_ref().expect("experiment section required");
    let rho = exp
        .rho
        .expect("landscape mode requires experiment.rho (training-style windows)");
    let warmup = exp.warmup;
    let t_days = trimmed_days(rho, cfg.params.tau);

    // Same L1 guard as eval-loss: H6 reports the training L1 loss.
    assert_eq!(
        exp.loss.kind,
        ddrs::config::LossKind::L1,
        "landscape mode always computes L1 (docs/superpowers/specs/2026-07-08-\
         landscape-hypotheses-h5-h6-draft.md); config has experiment.loss.kind={:?}",
        exp.loss.kind
    );

    let ranges = &cfg.params.parameter_ranges;
    let log_space = |name: &str| cfg.params.log_space_parameters.iter().any(|s| s == name);

    // Both arms' dumps, always (the anchor does not depend on which arm's
    // forcing this invocation evaluates under).
    let a_n = load_comid_field(params_a, "n")?;
    let a_q = load_comid_field(params_a, "q_spatial")?;
    let a_p = load_comid_field(params_a, "p_spatial")?;
    let b_n = load_comid_field(params_b, "n")?;
    let b_q = load_comid_field(params_b, "q_spatial")?;
    let b_p = load_comid_field(params_b, "p_spatial")?;

    // Anchor θ̄ (intersection of the two dumps' COMID sets, per-field
    // log/linear arm mean). Only needed for the surface, but cheap.
    let anchor_n = arm_mean_field(&a_n, &b_n, log_space("n"));
    let anchor_q = arm_mean_field(&a_q, &b_q, log_space("q_spatial"));
    let anchor_p = arm_mean_field(&a_p, &b_p, log_space("p_spatial"));
    eprintln!(
        "anchor: {} common COMIDs (A: {}, B: {}); log-space: n={} q_spatial={} p_spatial={}",
        anchor_n.len(),
        a_n.len(),
        b_n.len(),
        log_space("n"),
        log_space("q_spatial"),
        log_space("p_spatial")
    );
    if anchor_n.len() < a_n.len().max(b_n.len()) {
        eprintln!(
            "warning: {} COMIDs present in only one dump were skipped from the anchor — \
             a batch that needs one of them will hard-error at gather time",
            a_n.len() + b_n.len() - 2 * anchor_n.len()
        );
    }

    // Grid points (row-major over (log2_alpha, log2_beta)), or the single
    // re-check point.
    let grid: Vec<(f32, f32)> = match single_point {
        Some(pt) => vec![pt],
        None => {
            let coords = grid_coords(cli.grid_n);
            coords
                .iter()
                .flat_map(|&la| coords.iter().map(move |&lb| (la, lb)))
                .collect()
        }
    };
    let ts = barrier_ts(cli.barrier_points);

    // ----- Phase 1: sample the window/gauge plan EXACTLY ONCE -----
    let plan =
        sample_window_plan(&dataset, exp.batch_size, rho, warmup, t_days, cli.windows, cli.seed)?;

    // ----- Phase 2: replay the fixed plan across every grid/barrier point -----
    let mut surface_rows: Vec<(f32, f32, usize, f32)> = Vec::new();
    let mut barrier_rows: Vec<(f32, usize, f32)> = Vec::new();
    for (window_idx, (staids, window)) in plan.iter().enumerate() {
        let batch = dataset.collate(staids, window)?;
        let comids: Vec<i64> = batch.divide_comids.iter().map(|c| c.0).collect();
        let obs_arr = batch.observations.clone();
        let tensors = batch.to_tensors::<I>(&device);

        // Loss for one override at this window; hard-errors on a
        // parameter-induced all-NaN filter result (Phase 1 guarantees at
        // least one clean gauge exists, so n_kept == 0 here is a bug).
        let eval_point = |ov: &RoutingParamOverride,
                          label: &str|
         -> Result<f32, Box<dyn std::error::Error>> {
            let filtered = eval_window_filtered::<I>(
                &cfg, &tensors, &head, &device, &obs_arr, warmup, t_days, Some(ov),
            );
            let n_kept = filtered.mask.iter().filter(|&&v| v).count();
            if n_kept == 0 {
                return Err(format!(
                    "window {window_idx} point {label}: all gauges NaN post-filter — \
                     Phase 1's validity check should have excluded this window"
                )
                .into());
            }
            Ok(l1_loss_post_warmup(&filtered.predictions, &filtered.observations, 0))
        };

        if cli.surface_output.is_some() {
            // Gather the anchor ONCE per window; q_spatial is never scaled,
            // so its normalization is also once per window.
            let anc_n = gather_by_comid(&anchor_n, &comids)?;
            let anc_q = gather_by_comid(&anchor_q, &comids)?;
            let anc_p = gather_by_comid(&anchor_p, &comids)?;
            let q_norm =
                physical_to_normalized(&anc_q, ranges.q_spatial, log_space("q_spatial"));

            for (pi, &(la, lb)) in grid.iter().enumerate() {
                let n_phys = scale_by_log2(&anc_n, la);
                let p_phys = scale_by_log2(&anc_p, lb);
                let ov = RoutingParamOverride {
                    n: Some(physical_to_normalized(&n_phys, ranges.n, log_space("n"))),
                    q_spatial: Some(q_norm.clone()),
                    p_spatial: Some(physical_to_normalized(
                        &p_phys,
                        ranges.p_spatial,
                        log_space("p_spatial"),
                    )),
                };
                let loss = eval_point(&ov, &format!("(log2_alpha={la}, log2_beta={lb})"))?;
                surface_rows.push((la, lb, window_idx, loss));
                eprintln!(
                    "window {}/{} surface {}/{} (log2_alpha={la}, log2_beta={lb}) \
                     loss={loss:.5}",
                    window_idx + 1,
                    plan.len(),
                    pi + 1,
                    grid.len()
                );
            }
        }

        if cli.barrier_output.is_some() {
            // Endpoint fields gathered ONCE per window.
            let a_nv = gather_by_comid(&a_n, &comids)?;
            let a_qv = gather_by_comid(&a_q, &comids)?;
            let a_pv = gather_by_comid(&a_p, &comids)?;
            let b_nv = gather_by_comid(&b_n, &comids)?;
            let b_qv = gather_by_comid(&b_q, &comids)?;
            let b_pv = gather_by_comid(&b_p, &comids)?;

            for (ti, &t) in ts.iter().enumerate() {
                let n_phys = log_interp(&a_nv, &b_nv, t);
                let q_phys = log_interp(&a_qv, &b_qv, t);
                let p_phys = log_interp(&a_pv, &b_pv, t);
                let ov = RoutingParamOverride {
                    n: Some(physical_to_normalized(&n_phys, ranges.n, log_space("n"))),
                    q_spatial: Some(physical_to_normalized(
                        &q_phys,
                        ranges.q_spatial,
                        log_space("q_spatial"),
                    )),
                    p_spatial: Some(physical_to_normalized(
                        &p_phys,
                        ranges.p_spatial,
                        log_space("p_spatial"),
                    )),
                };
                let loss = eval_point(&ov, &format!("(t={t})"))?;
                barrier_rows.push((t, window_idx, loss));
                eprintln!(
                    "window {}/{} barrier {}/{} (t={t}) loss={loss:.5}",
                    window_idx + 1,
                    plan.len(),
                    ti + 1,
                    ts.len()
                );
            }
        }
    }

    if let Some(surface_output) = &cli.surface_output {
        write_surface_csv(surface_output, &surface_rows)?;
        println!(
            "wrote {} ({} grid points x {} windows)",
            surface_output.display(),
            grid.len(),
            plan.len()
        );
    }

    if let Some(barrier_output) = &cli.barrier_output {
        write_barrier_csv(barrier_output, &barrier_rows)?;
        println!(
            "wrote {} ({} t-points x {} windows)",
            barrier_output.display(),
            ts.len(),
            plan.len()
        );

        // Informational barrier statistic (window-mean per t). The
        // AUTHORITATIVE H6 statistics/verdicts are computed by the Python
        // analysis script over the CSV — this print is a smoke-check only.
        let mean_at = |t: f32| -> f32 {
            let vals: Vec<f32> = barrier_rows
                .iter()
                .filter(|(rt, _, _)| *rt == t)
                .map(|(_, _, l)| *l)
                .collect();
            vals.iter().sum::<f32>() / vals.len() as f32
        };
        let l0 = mean_at(ts[0]);
        let l1 = mean_at(*ts.last().unwrap());
        let (t_max, l_max) = ts
            .iter()
            .map(|&t| (t, mean_at(t)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        println!(
            "barrier (informational): L(t=0)={l0:.5} L(t=1)={l1:.5} \
             max_t L={l_max:.5} at t={t_max} => B = {:.5}",
            l_max - l0.max(l1)
        );
    }

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
    // Must match the teacher run's chunk length (--chunk-days, default 365)
    // so state boundaries align.
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
        let runoff =
            forward_eval_reaches::<I>(&cfg, &tensors, &head, &device, false, None, None, None);
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

    // -----------------------------------------------------------------------
    // eval-loss mode: Composition parsing + gather_by_comid + trimmed_days
    // -----------------------------------------------------------------------

    #[test]
    fn composition_parse_accepts_all_four_names() {
        assert!(Composition::parse("own") == Ok(Composition::Own));
        assert!(Composition::parse("n-swap") == Ok(Composition::NSwap));
        assert!(Composition::parse("geo-swap") == Ok(Composition::GeoSwap));
        assert!(Composition::parse("full-swap") == Ok(Composition::FullSwap));
    }

    #[test]
    fn composition_parse_rejects_unknown_name() {
        let err = Composition::parse("bogus").unwrap_err();
        assert!(err.contains("bogus"), "error should name the bad input: {err}");
    }

    #[test]
    fn composition_needs_donor_only_for_non_own() {
        assert!(!Composition::Own.needs_donor());
        assert!(Composition::NSwap.needs_donor());
        assert!(Composition::GeoSwap.needs_donor());
        assert!(Composition::FullSwap.needs_donor());
    }

    #[test]
    fn composition_build_override_selects_the_right_fields() {
        let n = Some(vec![0.1_f32]);
        let q = Some(vec![0.2_f32]);
        let p = Some(vec![0.3_f32]);

        assert!(Composition::Own.build_override(&n, &q, &p).is_none());

        let ov = Composition::NSwap.build_override(&n, &q, &p).unwrap();
        assert_eq!(ov.n, n);
        assert_eq!(ov.q_spatial, None);
        assert_eq!(ov.p_spatial, None);

        let ov = Composition::GeoSwap.build_override(&n, &q, &p).unwrap();
        assert_eq!(ov.n, None);
        assert_eq!(ov.q_spatial, q);
        assert_eq!(ov.p_spatial, p);

        let ov = Composition::FullSwap.build_override(&n, &q, &p).unwrap();
        assert_eq!(ov.n, n);
        assert_eq!(ov.q_spatial, q);
        assert_eq!(ov.p_spatial, p);
    }

    #[test]
    fn gather_by_comid_preserves_batch_order() {
        let donor: HashMap<i64, f32> = [(10i64, 1.0f32), (20, 2.0), (30, 3.0)].into_iter().collect();
        // Request in a different order than the map's insertion/iteration order.
        let got = gather_by_comid(&donor, &[30, 10, 20]).unwrap();
        assert_eq!(got, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn gather_by_comid_errors_on_missing_comid() {
        let donor: HashMap<i64, f32> = [(10i64, 1.0f32)].into_iter().collect();
        let err = gather_by_comid(&donor, &[10, 99]).unwrap_err();
        assert!(err.contains("99"), "error should name the missing COMID: {err}");
    }

    #[test]
    fn trimmed_days_matches_tau_trim_and_downsample_arithmetic() {
        // rho=90 -> n_hourly=(90-1)*24=2136; tau=0 -> start=13,end=2125,
        // trimmed=2112, days=2112/24=88.
        assert_eq!(trimmed_days(90, 0), 88);
    }

    // -----------------------------------------------------------------------
    // eval-loss mode: per-gauge output (H5/H6 gap 2)
    // -----------------------------------------------------------------------

    #[test]
    fn per_gauge_l1_matches_hand_computed_column_means() {
        // (T_days=3, G_kept=2).
        let pred = Array2::from_shape_vec(
            (3, 2),
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let obs = Array2::from_shape_vec(
            (3, 2),
            vec![1.0_f32, 2.0, 4.0, 4.0, 5.0, 9.0],
        )
        .unwrap();
        // Column 0 (gauge A): |1-1|+|3-4|+|5-5| = 0+1+0 = 1; mean = 1/3.
        // Column 1 (gauge B): |2-2|+|4-4|+|6-9| = 0+0+3 = 3; mean = 1.0.
        let losses = per_gauge_l1(&pred, &obs);
        assert_eq!(losses.len(), 2);
        assert!((losses[0] - 1.0 / 3.0).abs() < 1e-6, "got {}", losses[0]);
        assert!((losses[1] - 1.0).abs() < 1e-6, "got {}", losses[1]);
    }

    #[test]
    #[should_panic]
    fn per_gauge_l1_asserts_matching_shapes() {
        let pred = Array2::<f32>::zeros((3, 2));
        let obs = Array2::<f32>::zeros((3, 3));
        let _ = per_gauge_l1(&pred, &obs);
    }

    #[test]
    fn kept_gauge_staids_selects_and_orders_by_mask() {
        let staids: Vec<Staid> = vec!["A".into(), "B".into(), "C".into()];
        let mask = vec![true, false, true];
        let kept = kept_gauge_staids(&staids, &mask);
        assert_eq!(kept, vec![Staid::from("A"), Staid::from("C")]);
    }

    #[test]
    fn write_per_gauge_loss_csv_roundtrips_and_has_expected_row_count() {
        let dir = std::env::temp_dir().join("probe_per_gauge_csv_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("per_gauge_test.csv");
        let _ = std::fs::remove_file(&path);

        // Synthetic 2-window x 2-composition plan where kept-gauge count
        // VARIES per window (window 0: 2 gauges kept; window 1: 1 gauge
        // kept, mirroring how a real NaN-filtered window can drop gauges)
        // — the row count must be n_windows * n_compositions * n_kept_gauges
        // summed per window, NOT a fixed n_windows * n_compositions * G.
        let rows: Vec<(String, usize, Staid, f32)> = vec![
            ("own".to_string(), 0, Staid::from("A"), 0.5),
            ("own".to_string(), 0, Staid::from("B"), 0.7),
            ("own".to_string(), 1, Staid::from("A"), 0.9),
            ("n-swap".to_string(), 0, Staid::from("A"), 0.6),
            ("n-swap".to_string(), 0, Staid::from("B"), 0.8),
            ("n-swap".to_string(), 1, Staid::from("A"), 1.0),
        ];
        write_per_gauge_loss_csv(&path, &rows).unwrap();

        let csv = std::fs::read_to_string(&path).unwrap();
        let mut lines = csv.lines();
        assert_eq!(lines.next().unwrap(), "composition,window,staid,gauge_loss");
        let data_rows: Vec<&str> = lines.collect();
        assert_eq!(data_rows.len(), rows.len());
        // Staid zero-pads to 8 chars (Staid::new) — assert against that
        // canonical form, not the raw literal passed to `Staid::from`.
        assert_eq!(data_rows[0], "own,0,0000000A,0.5");
        assert_eq!(data_rows[2], "own,1,0000000A,0.9");
    }

    // -----------------------------------------------------------------------
    // landscape mode (H6): grid coords, anchor, scaling, barrier, CSVs
    // -----------------------------------------------------------------------

    #[test]
    fn grid_coords_registered_11_points_span_and_spacing() {
        let c = grid_coords(11);
        assert_eq!(c.len(), 11);
        assert_eq!(c[0], -1.5, "first coord must be exactly -1.5");
        assert_eq!(c[10], 1.5, "last coord must be exactly 1.5");
        for w in c.windows(2) {
            assert!(
                (w[1] - w[0] - 0.3).abs() < 1e-6,
                "spacing must be 0.3, got {}",
                w[1] - w[0]
            );
        }
        // Interior points land on clean nearest-f32 decimals (f64 compute).
        assert_eq!(c[2], -0.9_f32);
        assert_eq!(c[5], 0.0_f32);
    }

    #[test]
    fn grid_coords_smoke_3_points() {
        assert_eq!(grid_coords(3), vec![-1.5, 0.0, 1.5]);
    }

    #[test]
    fn barrier_ts_registered_21_points() {
        let t = barrier_ts(21);
        assert_eq!(t.len(), 21);
        assert_eq!(t[0], 0.0);
        assert_eq!(t[20], 1.0);
        assert!((t[1] - 0.05).abs() < 1e-7);
        assert!((t[10] - 0.5).abs() < 1e-7);
    }

    #[test]
    fn arm_mean_field_geometric_for_log_space() {
        let a: HashMap<i64, f32> = [(1i64, 2.0f32), (2, 8.0)].into_iter().collect();
        let b: HashMap<i64, f32> = [(1i64, 8.0f32), (2, 2.0)].into_iter().collect();
        let m = arm_mean_field(&a, &b, true);
        // Geometric mean of {2, 8} is 4 for both COMIDs.
        assert!((m[&1] - 4.0).abs() < 1e-6, "got {}", m[&1]);
        assert!((m[&2] - 4.0).abs() < 1e-6, "got {}", m[&2]);
    }

    #[test]
    fn arm_mean_field_arithmetic_for_linear() {
        let a: HashMap<i64, f32> = [(1i64, 2.0f32), (2, 8.0)].into_iter().collect();
        let b: HashMap<i64, f32> = [(1i64, 8.0f32), (2, 2.0)].into_iter().collect();
        let m = arm_mean_field(&a, &b, false);
        // Arithmetic mean of {2, 8} is 5.
        assert!((m[&1] - 5.0).abs() < 1e-6, "got {}", m[&1]);
        assert!((m[&2] - 5.0).abs() < 1e-6, "got {}", m[&2]);
    }

    #[test]
    fn arm_mean_field_skips_one_sided_comids() {
        let a: HashMap<i64, f32> = [(1i64, 2.0f32), (3, 6.0)].into_iter().collect();
        let b: HashMap<i64, f32> = [(1i64, 4.0f32), (4, 9.0)].into_iter().collect();
        let m = arm_mean_field(&a, &b, false);
        // Only COMID 1 is in both; 3 and 4 are skipped (documented choice —
        // gather_by_comid hard-errors later if a batch needs them).
        assert_eq!(m.len(), 1);
        assert!((m[&1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn scale_by_log2_matches_pow2_by_hand() {
        // anchor 0.04 at log2_alpha = 1.5: 0.04 * 2^1.5 = 0.04 * 2.8284271...
        let got = scale_by_log2(&[0.04], 1.5);
        assert!((got[0] - 0.04 * 2.828_427_1).abs() < 1e-7, "got {}", got[0]);
        // log2 = -1.5 halves-and-halves-again-and-sqrt: 0.04 / 2.8284271.
        let got = scale_by_log2(&[0.04], -1.5);
        assert!((got[0] - 0.04 / 2.828_427_1).abs() < 1e-7, "got {}", got[0]);
        // log2 = 0 is the exact identity.
        assert_eq!(scale_by_log2(&[0.123_456], 0.0), vec![0.123_456]);
    }

    #[test]
    fn log_interp_endpoints_reproduce_own_fields_exactly() {
        let a = vec![0.031_f32, 21.7, 0.999];
        let b = vec![0.187_f32, 3.2, 0.001];
        // Bit-exact at both endpoints (special-cased; no exp(ln x) wobble).
        assert_eq!(log_interp(&a, &b, 0.0), a);
        assert_eq!(log_interp(&a, &b, 1.0), b);
    }

    #[test]
    fn log_interp_midpoint_is_geometric_mean() {
        // t = 0.5: exp((ln 2 + ln 8)/2) = 4.
        let got = log_interp(&[2.0], &[8.0], 0.5);
        assert!((got[0] - 4.0).abs() < 1e-6, "got {}", got[0]);
    }

    #[test]
    #[should_panic(expected = "log_interp requires positive values")]
    fn log_interp_rejects_nonpositive_values() {
        let _ = log_interp(&[0.0], &[1.0], 0.5);
    }

    #[test]
    fn parse_single_point_accepts_signed_pair() {
        assert_eq!(parse_single_point("0.3,-0.6"), Ok((0.3, -0.6)));
        assert_eq!(parse_single_point(" -1.5 , 1.5 "), Ok((-1.5, 1.5)));
    }

    #[test]
    fn parse_single_point_rejects_malformed_input() {
        assert!(parse_single_point("0.3").is_err());
        assert!(parse_single_point("0.3,0.4,0.5").is_err());
        assert!(parse_single_point("abc,0.4").is_err());
    }

    #[test]
    fn write_surface_csv_roundtrips() {
        let dir = std::env::temp_dir().join("probe_surface_csv_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("surface_test.csv");
        let _ = std::fs::remove_file(&path);

        let rows: Vec<(f32, f32, usize, f32)> = vec![
            (-1.5, -1.5, 0, 12.34),
            (-1.5, 0.0, 0, 11.98),
            (-1.5, -1.5, 1, 13.01),
        ];
        write_surface_csv(&path, &rows).unwrap();

        let csv = std::fs::read_to_string(&path).unwrap();
        let mut lines = csv.lines();
        assert_eq!(lines.next().unwrap(), "log2_alpha,log2_beta,window,mean_loss");
        let data: Vec<&str> = lines.collect();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0], "-1.5,-1.5,0,12.34");
        assert_eq!(data[1], "-1.5,0,0,11.98");
    }

    #[test]
    fn write_barrier_csv_roundtrips() {
        let dir = std::env::temp_dir().join("probe_barrier_csv_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("barrier_test.csv");
        let _ = std::fs::remove_file(&path);

        let rows: Vec<(f32, usize, f32)> = vec![(0.0, 0, 5.5), (0.05, 0, 5.7), (1.0, 1, 6.1)];
        write_barrier_csv(&path, &rows).unwrap();

        let csv = std::fs::read_to_string(&path).unwrap();
        let mut lines = csv.lines();
        assert_eq!(lines.next().unwrap(), "t,window,mean_loss");
        let data: Vec<&str> = lines.collect();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0], "0,0,5.5");
        assert_eq!(data[1], "0.05,0,5.7");
    }
}
