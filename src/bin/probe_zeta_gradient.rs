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

use std::path::PathBuf;

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use burn::backend::Autodiff;
use burn::prelude::ElementConversion;
use burn::tensor::{backend::Backend, Tensor, TensorData};

use ddrs::config::{kan_config, Config, ConfigMode, SparseSolver};
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::data::sampler::{BatchSource, RandomSampler};
use ddrs::dump_parameters::write_grad_netcdf;
use ddrs::nn::kan_head::KanHead;
use ddrs::training::checkpoint::{head_base, load_kan_head};
use ddrs::training::probe::{probe_forward, GradAccum};
use ddrs::training::{batch_loss, tau_trim_and_downsample};

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

    #[arg(long)]
    output: PathBuf,

    /// Stage 2 only: probe plan CSV (round,comid,delta) — implemented in a LATER task.
    #[arg(long)]
    #[allow(dead_code)]
    probe_plan: Option<PathBuf>,

    /// Backend: "cpu" (NdArray, deterministic; forces sparse_solver=cpu) or "cuda".
    #[arg(long, default_value = "cpu")]
    backend: String,

    /// Stage 2 only: route only the first D days of the eval period.
    #[arg(long, default_value_t = 1095)]
    #[allow(dead_code)]
    eval_days: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.mode.as_str() {
        "grad" => {}
        "perturb" => return Err("perturb mode lands in a later task".into()),
        other => return Err(format!("unknown --mode {other} (expected \"grad\")").into()),
    }

    // Training-mode config: probe batches replicate the training sampler.
    let mut cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Training)?;

    match cli.backend.as_str() {
        "cpu" => {
            type I = burn::backend::NdArray<f32>;
            let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
            cfg.params.sparse_solver = SparseSolver::Cpu;
            eprintln!("backend: cpu (NdArray, deterministic; sparse_solver forced to cpu)");
            <I as burn::tensor::backend::Backend>::seed(&device, cfg.seed);
            run::<I>(cfg, cli, device)
        }
        "cuda" => {
            type I = burn_cuda::Cuda<f32, i32>;
            // Config-selected CUDA ordinal (top-level `device:` key).
            let device = cubecl::cuda::CudaDevice::new(cfg.device);
            <I as burn::tensor::backend::Backend>::seed(&device, cfg.seed);
            run::<I>(cfg, cli, device)
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
