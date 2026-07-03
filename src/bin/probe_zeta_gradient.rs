//! Stage-1 adjoint reachability probe driver (spec:
//! docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md).
//!
//! Samples training-style batches (same sampler + rho-window machinery as the
//! training driver, but a LOCAL rng seeded from --seed), runs
//! `probe_forward` + the config loss + backward at a FIXED head (no optimizer
//! step ever), reads the gradients on the lifted normalized leakance leaves,
//! accumulates per-COMID, and writes the mean |g| / signed g map via
//! `write_grad_netcdf`.
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
use ddrs::data::sampler::{BatchSource, RandomSampler};
use ddrs::data::TestWindow;
use ddrs::dump_parameters::write_grad_netcdf;
use ddrs::nn::kan_head::KanHead;
use ddrs::training::checkpoint::{head_base, load_kan_head};
use ddrs::training::probe::{probe_forward, GradAccum};
use ddrs::training::{batch_loss, forward_eval, tau_trim_and_downsample};

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
    #[arg(long)]
    output: PathBuf,

    /// Stage 2 only: probe plan CSV (round,comid,delta,...). REQUIRED in perturb mode.
    #[arg(long)]
    probe_plan: Option<PathBuf>,

    /// Backend: "cpu" (NdArray, deterministic; forces sparse_solver=cpu) or "cuda".
    #[arg(long, default_value = "cpu")]
    backend: String,

    /// Stage 2 only: route only the first D days of the eval period.
    #[arg(long, default_value_t = 1095)]
    eval_days: usize,
}

#[derive(Copy, Clone, PartialEq)]
enum Mode {
    Grad,
    Perturb,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let mode = match cli.mode.as_str() {
        "grad" => Mode::Grad,
        "perturb" => Mode::Perturb,
        other => {
            return Err(
                format!("unknown --mode {other} (expected \"grad\" or \"perturb\")").into(),
            )
        }
    };

    // grad: training-mode config (probe batches replicate the training
    // sampler). perturb: testing-mode config (replicates the eval loop over
    // the test window).
    let cfg_mode = match mode {
        Mode::Grad => ConfigMode::Training,
        Mode::Perturb => ConfigMode::Testing,
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
            }
        }
        other => Err(format!("unknown --backend {other} (expected \"cpu\" or \"cuda\")").into()),
    }
}

fn run<I: Backend>(cfg: Config, cli: Cli, device: I::Device) -> Result<(), Box<dyn std::error::Error>> {
    assert!(
        cfg.params.use_leakance,
        "probe requires params.use_leakance: true — use a leakance experiment config"
    );

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

    let mut accum_factor = GradAccum::new();
    let mut accum_dgw = GradAccum::new();
    let mut accum_kd = GradAccum::new();

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
        let (pred_hourly, leaves) = probe_forward::<I>(&cfg, &tensors, &head, &device);
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
        let g_factor: Vec<f32> =
            leaves.factor.grad(&grads).expect("factor grad").into_data().into_vec().unwrap();
        let g_dgw: Vec<f32> =
            leaves.d_gw.grad(&grads).expect("d_gw grad").into_data().into_vec().unwrap();
        let g_kd: Vec<f32> =
            leaves.k_d.grad(&grads).expect("k_d grad").into_data().into_vec().unwrap();

        // Fail fast on non-finite gradients: a poisoned mean invalidates the
        // entire accumulation map — do NOT skip-and-continue, which would
        // desync accumulator coverage.
        let nf_factor = g_factor.iter().filter(|v| !v.is_finite()).count();
        let nf_dgw = g_dgw.iter().filter(|v| !v.is_finite()).count();
        let nf_kd = g_kd.iter().filter(|v| !v.is_finite()).count();
        if nf_factor > 0 || nf_dgw > 0 || nf_kd > 0 {
            eprintln!(
                "batch {}: non-finite grads — factor:{nf_factor} d_gw:{nf_dgw} k_d:{nf_kd}",
                processed + 1
            );
            std::process::exit(1);
        }

        accum_factor.add(&comids, &g_factor, &g_factor);
        accum_dgw.add(&comids, &g_dgw, &g_dgw);
        accum_kd.add(&comids, &g_kd, &g_kd);

        eprintln!(
            "batch {}/{}: loss={loss_f32:.5} gauges={surviving_g}",
            processed + 1,
            cli.windows
        );
        processed += 1;
    }

    // All three accumulators saw identical (comids, count) streams — the
    // factor rows define the master COMID order.
    let rows_factor = accum_factor.into_sorted_rows();
    let rows_dgw = accum_dgw.into_sorted_rows();
    let rows_kd = accum_kd.into_sorted_rows();
    assert_eq!(rows_factor.len(), rows_dgw.len());
    assert_eq!(rows_factor.len(), rows_kd.len());
    assert!(
        rows_factor.iter().map(|r| r.0).eq(rows_dgw.iter().map(|r| r.0))
            && rows_factor.iter().map(|r| r.0).eq(rows_kd.iter().map(|r| r.0)),
        "accumulator COMID sets diverged — per-param skipping was introduced somewhere"
    );

    let comids: Vec<i64> = rows_factor.iter().map(|r| r.0).collect();
    let n_windows: Vec<i32> = rows_factor.iter().map(|r| r.3 as i32).collect();
    let mean_abs = |rows: &[(i64, f64, f64, u32)]| -> Vec<f32> {
        rows.iter().map(|r| (r.1 / r.3 as f64) as f32).collect()
    };
    let mean_net = |rows: &[(i64, f64, f64, u32)]| -> Vec<f32> {
        rows.iter().map(|r| (r.2 / r.3 as f64) as f32).collect()
    };

    write_grad_netcdf(
        &cli.output,
        &comids,
        &mean_abs(&rows_factor),
        &mean_net(&rows_factor),
        &mean_abs(&rows_dgw),
        &mean_net(&rows_dgw),
        &mean_abs(&rows_kd),
        &mean_net(&rows_kd),
        &n_windows,
        &checkpoint_label,
        cli.windows,
        cli.seed,
    )
    .map_err(|e| -> Box<dyn std::error::Error> { e })?;

    println!(
        "wrote {} ({} reaches, {} batches, seed {})",
        cli.output.display(),
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

    std::fs::create_dir_all(&cli.output)?;

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
    write_round_netcdf(&cli.output.join("baseline_1.nc"), &gauge_staids, &b1, &day0, n_days as i64)?;
    eprintln!("baseline 2/2 (0 probes)");
    let b2 = run_one(&[])?;
    write_round_netcdf(&cli.output.join("baseline_2.nc"), &gauge_staids, &b2, &day0, n_days as i64)?;

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
            &cli.output.join(format!("round_{round}.nc")),
            &gauge_staids,
            &preds,
            &day0,
            n_days as i64,
        )?;
    }

    println!(
        "wrote {} runs (2 baselines + {n_rounds} rounds) to {} ({} gauges x {} days)",
        n_rounds + 2,
        cli.output.display(),
        n_all_gauges,
        n_days - 2,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
