//! Top-level training driver. Mirrors `~/projects/ddr/scripts/train.py:23-128`.
//!
//! Per-mini-batch flow:
//!   1. Sample a batch (gauges + rho-window).
//!   2. Collate → RoutingBatch → RoutingTensors<Autodiff<I>>.
//!   3. forward() — KAN head  MC engine + scatter-add to per-gauge.
//!   4. tau-trim + daily downsample → (G, T_days) BURN tensor.
//!   5. NaN-gauge filter (DDR train.py:75-89): gauges with any NaN in the
//!      post-warmup obs window are dropped from both predictions and
//!      observations before the L1 loss. Autograd stays alive on predictions
//!      via Tensor::select on the kept-gauge index.
//!   6. L1 loss in BURN-tensor space (autograd alive).
//!   7. loss.backward() → clip_grad_norm → optimizer.step.
//!   8. save_kan_head to checkpoint_dir.
//!
//! ## Optimizer micro-batching (`experiment.grad_accum_steps` > 1)
//!
//! Steps 1–6 run once per MICRO-batch; step 7 runs once per GROUP of up to
//! `grad_accum_steps` micro-batches. Each micro-batch loss (a mean over its
//! own valid denominator `n_i` — see `loss_denominator`) is scaled by `n_i`
//! before `backward()`, the gradients are summed in a `GradientsAccumulator`,
//! and the total is rescaled by `1 / Σ n_i` before clip + step. By linearity
//! this equals the gradient of the pooled large-batch loss exactly
//! (`tests/grad_accum_equivalence.rs`), while peak memory stays that of one
//! micro-batch — activations are freed between micro-batches.
//!
//! Degenerate-batch semantics match the single-batch loop: a micro-batch
//! whose gauges are all NaN-filtered contributes nothing (`n_i = 0`); if
//! EVERY micro-batch in the group is degenerate the step is skipped, exactly
//! as the single-batch path skips such a mini-batch. When accumulating, the
//! sampler is built with `drop_last = false` so the `n_gauges % batch_size`
//! tail joins the final group instead of being silently dropped — the
//! valid-count weighting makes the unequal group size exact rather than
//! approximate. `max_mini_batches` counts optimizer STEPS (groups) in
//! accumulation mode.

use std::path::Path;

use rand_chacha::ChaCha12Rng;

use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::ElementConversion;
use burn::tensor::{backend::Backend, Tensor, TensorData};

use crate::config::Config;
use crate::data::dataset::MeritGagesDataset;
use crate::data::dates::RhoWindow;
use crate::data::error::{DataError, Result};
use crate::data::ids::Staid;
use crate::data::sampler::{BatchSource, RandomSampler};
use crate::nn::kan_head::KanHead;
use crate::training::checkpoint::{
    head_base, optim_base, save_optimizer, save_train_state, state_path, TrainCkptState,
};
use crate::training::forward::forward;
use crate::training::loss::loss_denominator;
use crate::training::optimizer::scale_grads;
use crate::training::{clip_grad_norm, resolve_lr, save_kan_head, tau_trim_and_downsample};

/// Mutable state threaded through the training loop.
///
/// Parameterized on the inner backend `I` so the autodiff type
/// (`Autodiff<I>`) is always explicit and matches the `forward` signature.
///
/// `rng` is `ChaCha12Rng` (= rand 0.8's `StdRng`, identical stream from
/// `seed_from_u64`) so it can be serde-checkpointed for exact resume.
pub struct TrainState<I: Backend> {
    pub head: KanHead<Autodiff<I>>,
    pub epoch: usize,
    pub mini_batch: usize,
    pub rng: ChaCha12Rng,
    /// Mid-epoch resume: the in-flight epoch's sampler permutation + cursor
    /// from a checkpoint sidecar. Consumed by `train` on its first epoch
    /// (skipping the reshuffle); `None` for fresh runs.
    pub resume_sampler: Option<(Vec<usize>, usize)>,
}

/// One micro-batch's forward + loss, before any backward.
struct MicroBatchOutcome<I: Backend> {
    /// Scalar batch loss (mean over `n_valid`), autograd alive.
    loss: Tensor<Autodiff<I>, 1>,
    /// The loss as f32, for logging.
    loss_f32: f32,
    /// Pooled-mean denominator of `loss` (see `loss_denominator`).
    n_valid: usize,
    /// Median physical Manning's `n` over this batch's network, and the
    /// fraction of its reaches within 1% of the range floor. Logging only —
    /// see `forward::manning_n_stats` for why.
    median_n: f32,
    n_at_floor: f32,
}

/// Steps 2–6 of the per-mini-batch flow (collate → forward → NaN filter →
/// loss). Shared verbatim between the single-batch and accumulation paths so
/// the default path stays byte-identical to the pre-accumulation driver.
///
/// Returns `Ok(None)` when every gauge in the batch has NaN in its
/// post-warmup obs window (the caller decides skip semantics).
fn run_micro_batch<I: Backend>(
    cfg: &Config,
    dataset: &MeritGagesDataset,
    head: &KanHead<Autodiff<I>>,
    device: &I::Device,
    staids: &[Staid],
    window: &RhoWindow,
    mb_label: usize,
) -> Result<Option<MicroBatchOutcome<I>>> {
    let exp = cfg.experiment.as_ref().expect("experiment");
    let batch = dataset.collate(staids, window)?;
    let num_gauges = batch.gauge_staids.len();
    let batch_gauge_std = batch.gauge_obs_std.clone();

    // Save observations before consuming `batch` in to_tensors.
    // SP-3 layout: observations shape is (rho_days, G) — rows are daily
    // timesteps, columns are gauges.
    let obs_arr = batch.observations.clone(); // (rho_days, G)
    let t_days_full = obs_arr.nrows();

    // to_tensors::<Autodiff<I>> lifts plain ndarray buffers to the device.
    let tensors = batch.to_tensors::<Autodiff<I>>(device);
    let (median_n, n_at_floor) =
        crate::training::forward::manning_n_stats::<I>(cfg, &tensors, head);
    let pred_hourly = forward::<I>(cfg, &tensors, head, device, false);
    let daily = tau_trim_and_downsample(pred_hourly, cfg.params.tau);
    let dims = daily.dims();
    let (g, t_days) = (dims[0], dims[1]);
    debug_assert_eq!(g, num_gauges);

    // Build obs tensor preserving NaN so the filter can detect them.
    // Shape: obs_arr is (rho_days, G); trim first/last day → (t_days, G).
    assert!(
        t_days_full >= 2 + t_days,
        "obs/pred shape mismatch: obs rows={} pred t_days={}",
        t_days_full,
        t_days
    );
    let mut obs_buf: Vec<f32> = Vec::with_capacity(g * t_days);
    for gi in 0..g {
        for ti in 0..t_days {
            // obs row index after trim = ti + 1; column = gi.
            obs_buf.push(obs_arr[(ti + 1, gi)]);
        }
    }
    let obs_t: Tensor<Autodiff<I>, 2> =
        Tensor::<Autodiff<I>, 1>::from_data(TensorData::new(obs_buf, [g * t_days]), device)
            .reshape([g, t_days]);

    // Apply post-warmup slice along axis 1.
    let warmup = exp.warmup;
    debug_assert!(
        warmup < t_days,
        "warmup={warmup} >= t_days={t_days}; increase rho"
    );
    let p_post = daily.slice([0..g, warmup..t_days]); // (G, post)
    let o_post = obs_t.slice([0..g, warmup..t_days]);

    // Filter gauges whose post-warmup obs window contains any NaN.
    // Mirrors DDR's per-gauge NaN-mask at scripts/train.py:75-89.
    // Without this, the NaN→0.0 substitution biases the head toward
    // predicting near-zero flow and saturates Manning's n at the lower
    // bound (~0.030 in log space).
    //
    // Strategy: extract the obs values from `o_post` (no autograd on
    // obs), build the keep-indices in ndarray space, then apply
    // Tensor::select to both p_post and o_post so autograd stays alive
    // on predictions.
    let o_post_vec: Vec<f32> = o_post.clone().into_data().into_vec().unwrap();
    let t_post = t_days - warmup;
    let keep_indices: Vec<i32> = (0..g)
        .filter(|&gi| (0..t_post).all(|ti| !o_post_vec[gi * t_post + ti].is_nan()))
        .map(|gi| gi as i32)
        .collect();

    let surviving_g = keep_indices.len();
    if surviving_g == 0 {
        eprintln!("  mb={mb_label} skipped: all {g} gauges have NaN in post-warmup window");
        return Ok(None);
    }

    // Per-gauge training-period std for the SURVIVING gauges, in the same
    // row order as `p_filt` (empty unless `loss.kind: nse-batch`).
    let sigma = if batch_gauge_std.is_empty() {
        None
    } else {
        let vals: Vec<f32> = keep_indices.iter().map(|&gi| batch_gauge_std[gi as usize]).collect();
        Some(Tensor::<Autodiff<I>, 1>::from_data(
            TensorData::new(vals, [surviving_g]),
            device,
        ))
    };

    let keep_t: Tensor<Autodiff<I>, 1, burn::tensor::Int> =
        Tensor::from_data(TensorData::new(keep_indices, [surviving_g]), device);
    let p_filt = p_post.select(0, keep_t.clone());
    let o_filt = o_post.select(0, keep_t);

    // Config-selected objective (default L1); autograd alive on `p_filt`.
    let loss = crate::training::batch_loss(p_filt, o_filt, &exp.loss, sigma);
    let loss_f32: f32 = loss.clone().into_scalar().elem::<f32>();

    Ok(Some(MicroBatchOutcome {
        loss,
        loss_f32,
        n_valid: loss_denominator(&exp.loss, surviving_g, t_post),
        median_n,
        n_at_floor,
    }))
}

/// Step 8: write the `epoch_E_mb_M/` checkpoint directory (head.mpk +
/// optim.mpk + state.json). See checkpoint.rs module docs.
fn save_step_checkpoint<I: Backend>(
    checkpoint_dir: &Path,
    epoch: usize,
    state: &TrainState<I>,
    optimizer: &impl Optimizer<KanHead<Autodiff<I>>, Autodiff<I>>,
    sampler: &BatchSource,
) -> Result<()> {
    let ckpt_dir = checkpoint_dir.join(format!("epoch_{epoch}_mb_{}", state.mini_batch));
    std::fs::create_dir_all(&ckpt_dir).map_err(|e| DataError::Io {
        path: ckpt_dir.clone(),
        source: e,
    })?;

    // .valid() strips autodiff; save_kan_head<I> writes to disk.
    save_kan_head(&head_base(&ckpt_dir), &state.head.clone().valid())?;

    // Adam moments + train-loop position (rng, sampler permutation/cursor)
    // for exact resume.
    save_optimizer(&optim_base(&ckpt_dir), optimizer)?;
    if let Some((sampler_indices, sampler_cursor)) = sampler.snapshot() {
        save_train_state(
            &state_path(&ckpt_dir),
            &TrainCkptState {
                epoch,
                next_mini_batch: state.mini_batch + 1,
                rng: state.rng.clone(),
                sampler_indices,
                sampler_cursor,
            },
        )?;
    }
    Ok(())
}

/// Run the full training loop.
///
/// `device` is the inner-backend device (e.g. `NdArrayDevice::default()`).
/// `Autodiff<I>::Device == I::Device` in BURN 0.21.
pub fn train<I: Backend>(
    cfg: &Config,
    dataset: &MeritGagesDataset,
    state: &mut TrainState<I>,
    optimizer: &mut impl Optimizer<KanHead<Autodiff<I>>, Autodiff<I>>,
    device: &I::Device,
    checkpoint_dir: &Path,
    max_mini_batches: Option<usize>,
    batch_source: Option<BatchSource>,
) -> Result<()> {
    let exp = cfg.experiment.as_ref().expect("experiment");
    let rho = exp.rho.expect("training requires rho");
    let grad_clip = exp.grad_clip_max_norm.unwrap_or(1.0);
    let accum_steps = exp.effective_grad_accum_steps();

    let mut sampler = batch_source.unwrap_or_else(|| {
        // Accumulation keeps the partial tail batch (drop_last = false):
        // the valid-count weighting makes unequal group sizes exact, so the
        // `n % batch_size` tail no longer has to be silently dropped.
        BatchSource::Shuffle(RandomSampler::new(
            dataset.len(),
            exp.batch_size,
            accum_steps == 1,
        ))
    });

    // Mid-epoch resume: the first epoch reuses the checkpointed permutation +
    // cursor instead of reshuffling (the shuffle that produced it already
    // consumed the rng in the original run).
    let mut pending_restore = state.resume_sampler.take();

    for epoch in state.epoch..=exp.epochs {
        match pending_restore.take() {
            Some((indices, cursor)) => sampler.restore(indices, cursor),
            None => sampler.reshuffle(&mut state.rng),
        }
        let lr = resolve_lr(&exp.learning_rate, epoch);
        eprintln!("epoch {epoch} lr={lr}");

        let mut mb_done = 0usize;

        if accum_steps == 1 {
            // ── single-batch path: one optimizer step per mini-batch ──────
            // Byte-identical to the pre-accumulation driver.
            while let Some(idx) = sampler.next_batch() {
                let staids: Vec<_> = idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
                let window = dataset.time_axis().sample_rho_window(&mut state.rng, rho);
                let outcome = run_micro_batch::<I>(
                    cfg,
                    dataset,
                    &state.head,
                    device,
                    &staids,
                    &window,
                    state.mini_batch,
                )?;
                let Some(MicroBatchOutcome {
                    loss,
                    loss_f32,
                    median_n,
                    n_at_floor,
                    ..
                }) = outcome
                else {
                    state.mini_batch += 1;
                    mb_done += 1;
                    if let Some(limit) = max_mini_batches {
                        if mb_done >= limit {
                            break;
                        }
                    }
                    continue;
                };

                let grads = GradientsParams::from_grads(loss.backward(), &state.head);
                let grads = clip_grad_norm(grads, &state.head, grad_clip);
                state.head = optimizer.step(lr as f64, state.head.clone(), grads);

                save_step_checkpoint::<I>(checkpoint_dir, epoch, state, &*optimizer, &sampler)?;

                // SP-10 multi-batch OOM fix: every per-timestep call to
                // `fresh_primitive_from_scratch` (~24 per t on the CUDA-graph
                // replay path) allocates a fresh persistent-pool slice. The
                // persistent pool never recycles slices of differently-sized
                // gauge subgraphs across batches; without an explicit cleanup the
                // pool grows ~1 GB per batch and CUDA OOMs after ~4 mini-batches.
                // `cuda_memory_cleanup` calls `client.memory_cleanup()`, which
                // dealloc's all currently-free persistent slices. No-op on
                // non-CUDA backends. See `cusparse::cuda_memory_cleanup` for
                // the full diagnosis.
                crate::sparse::cusparse::cuda_memory_cleanup::<I>(device);

                eprintln!(
                    "  mb={} loss={:.6} median_n={median_n:.5} n_at_floor={:.1}%",
                    state.mini_batch,
                    loss_f32,
                    n_at_floor * 100.0
                );
                state.mini_batch += 1;
                mb_done += 1;
                if let Some(limit) = max_mini_batches {
                    if mb_done >= limit {
                        break;
                    }
                }
            }
        } else {
            // ── accumulation path: one optimizer step per group of up to
            //    `accum_steps` micro-batches ─────────────────────────────────
            loop {
                let mut accumulator = GradientsAccumulator::<KanHead<Autodiff<I>>>::new();
                let mut total_n = 0usize;
                let mut loss_weighted_sum = 0.0f64;
                let mut micros_drawn = 0usize;
                let mut micros_valid = 0usize;

                while micros_drawn < accum_steps {
                    let Some(idx) = sampler.next_batch() else { break };
                    micros_drawn += 1;
                    let staids: Vec<_> =
                        idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
                    let window = dataset.time_axis().sample_rho_window(&mut state.rng, rho);
                    let outcome = run_micro_batch::<I>(
                        cfg,
                        dataset,
                        &state.head,
                        device,
                        &staids,
                        &window,
                        state.mini_batch,
                    )?;
                    if let Some(MicroBatchOutcome {
                        loss,
                        loss_f32,
                        n_valid,
                        median_n,
                        n_at_floor,
                    }) = outcome
                    {
                        // Scale the mean loss back to a SUM before backward;
                        // the group total is renormalized by 1/Σn below, so
                        // the accumulated gradient is exactly the pooled-mean
                        // gradient (see module docs + equivalence test).
                        let scaled = loss.mul_scalar(n_valid as f32);
                        let grads =
                            GradientsParams::from_grads(scaled.backward(), &state.head);
                        accumulator.accumulate(&state.head, grads);
                        total_n += n_valid;
                        loss_weighted_sum += loss_f32 as f64 * n_valid as f64;
                        micros_valid += 1;
                        eprintln!(
                            "    micro {micros_drawn}/{accum_steps} gauges={} loss={loss_f32:.6} \
                             n={n_valid} median_n={median_n:.5} n_at_floor={:.1}%",
                            idx.len(),
                            n_at_floor * 100.0,
                        );
                    }
                    // Free the micro-batch's activations/pool slices before the
                    // next forward — this is what keeps peak memory at one
                    // micro-batch (see the SP-10 note on the single path).
                    crate::sparse::cusparse::cuda_memory_cleanup::<I>(device);
                }

                if micros_drawn == 0 {
                    break; // epoch exhausted
                }
                if micros_valid == 0 {
                    // Whole group degenerate — skip the step, exactly as the
                    // single-batch path skips an all-NaN mini-batch.
                    eprintln!(
                        "  mb={} skipped: all {micros_drawn} micro-batches degenerate",
                        state.mini_batch
                    );
                } else {
                    let grads =
                        scale_grads(accumulator.grads(), &state.head, 1.0 / total_n as f32);
                    let grads = clip_grad_norm(grads, &state.head, grad_clip);
                    state.head = optimizer.step(lr as f64, state.head.clone(), grads);

                    save_step_checkpoint::<I>(
                        checkpoint_dir,
                        epoch,
                        state,
                        &*optimizer,
                        &sampler,
                    )?;

                    eprintln!(
                        "  mb={} loss={:.6} (accumulated {micros_valid}/{micros_drawn} micro-batches, n={total_n})",
                        state.mini_batch,
                        loss_weighted_sum / total_n as f64,
                    );
                }

                state.mini_batch += 1;
                mb_done += 1;
                if let Some(limit) = max_mini_batches {
                    if mb_done >= limit {
                        break;
                    }
                }
            }
        }

        state.mini_batch = 0;
        state.epoch = epoch + 1;
    }
    Ok(())
}
