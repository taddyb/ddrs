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

use std::path::Path;

use rand::rngs::StdRng;

use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::ElementConversion;
use burn::tensor::{backend::Backend, Tensor, TensorData};

use crate::config::Config;
use crate::data::dataset::MeritGagesDataset;
use crate::data::error::Result;
use crate::data::sampler::{BatchSource, RandomSampler};
use crate::nn::kan_head::KanHead;
use crate::training::forward::forward;
use crate::training::{clip_grad_norm, resolve_lr, save_kan_head, tau_trim_and_downsample};

/// Mutable state threaded through the training loop.
///
/// Parameterized on the inner backend `I` so the autodiff type
/// (`Autodiff<I>`) is always explicit and matches the `forward` signature.
pub struct TrainState<I: Backend> {
    pub head: KanHead<Autodiff<I>>,
    pub epoch: usize,
    pub mini_batch: usize,
    pub rng: StdRng,
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

    let mut sampler = batch_source.unwrap_or_else(|| {
        BatchSource::Shuffle(RandomSampler::new(dataset.len(), exp.batch_size, true))
    });

    for epoch in state.epoch..=exp.epochs {
        sampler.reshuffle(&mut state.rng);
        let lr = resolve_lr(&exp.learning_rate, epoch);
        eprintln!("epoch {epoch} lr={lr}");

        let mut mb_done = 0usize;
        while let Some(idx) = sampler.next_batch() {
            let staids: Vec<_> = idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
            let window = dataset.time_axis().sample_rho_window(&mut state.rng, rho);
            let batch = dataset.collate(&staids, &window)?;
            let num_gauges = batch.gauge_staids.len();

            // Save observations before consuming `batch` in to_tensors.
            // SP-3 layout: observations shape is (rho_days, G) — rows are daily
            // timesteps, columns are gauges.
            let obs_arr = batch.observations.clone(); // (rho_days, G)
            let t_days_full = obs_arr.nrows();

            // to_tensors::<Autodiff<I>> lifts plain ndarray buffers to the device.
            let tensors = batch.to_tensors::<Autodiff<I>>(device);
            let pred_hourly = forward::<I>(cfg, &tensors, &state.head, device, false);
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
                .filter(|&gi| {
                    (0..t_post).all(|ti| !o_post_vec[gi * t_post + ti].is_nan())
                })
                .map(|gi| gi as i32)
                .collect();

            let surviving_g = keep_indices.len();
            if surviving_g == 0 {
                eprintln!(
                    "  mb={} skipped: all {g} gauges have NaN in post-warmup window",
                    state.mini_batch,
                );
                state.mini_batch += 1;
                mb_done += 1;
                if let Some(limit) = max_mini_batches {
                    if mb_done >= limit {
                        break;
                    }
                }
                continue;
            }

            let keep_t: Tensor<Autodiff<I>, 1, burn::tensor::Int> =
                Tensor::from_data(TensorData::new(keep_indices, [surviving_g]), device);
            let p_filt = p_post.select(0, keep_t.clone());
            let o_filt = o_post.select(0, keep_t);

            // L1 loss = mean(|p - o|); autograd alive on `p_filt`.
            let loss = (p_filt - o_filt).abs().mean();
            let loss_f32: f32 = loss.clone().into_scalar().elem::<f32>();

            let grads = GradientsParams::from_grads(loss.backward(), &state.head);
            let grads = clip_grad_norm(grads, &state.head, grad_clip);
            state.head = optimizer.step(lr as f64, state.head.clone(), grads);

            // Checkpoint: .valid() strips autodiff; save_kan_head<I> writes to disk.
            let ckpt_path =
                checkpoint_dir.join(format!("epoch_{epoch}_mb_{}", state.mini_batch));
            save_kan_head(&ckpt_path, &state.head.clone().valid())?;

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

            eprintln!("  mb={} loss={:.6}", state.mini_batch, loss_f32);
            state.mini_batch += 1;
            mb_done += 1;
            if let Some(limit) = max_mini_batches {
                if mb_done >= limit {
                    break;
                }
            }
        }
        state.mini_batch = 0;
        state.epoch = epoch + 1;
    }
    Ok(())
}
