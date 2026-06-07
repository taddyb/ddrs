//! Shared head + state + optimizer constructor used by the training binaries
//! and the CLI.
//!
//! Extracts the KAN-head + TrainState + Adam setup that was duplicated across
//! `train` and `train_and_test`, and centralises checkpoint resume: when
//! `experiment.checkpoint` is set, the head weights, Adam moments, and the
//! train-loop position (epoch, mini-batch, rng, sampler permutation) are all
//! restored from the checkpoint + its sidecars (see `checkpoint.rs`).

use burn::backend::Autodiff;
use burn::optim::Optimizer;
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::config::Config;
use crate::data::error::Result;
use crate::nn::kan_head::{KanHead, KanHeadConfig};
use crate::training::checkpoint::{load_kan_head, load_optimizer, load_train_state, optim_base, state_path};
use crate::training::driver::TrainState;
use crate::training::optimizer::build_adam;

/// Initialise the KAN head, the mutable training state, and the Adam
/// optimizer from `cfg`.
///
/// Type parameter `I` is the **inner** (non-autodiff) backend, matching the
/// convention used by `TrainState<I>` and the training binaries
/// (`type I = Cuda<f32, i32>`).
///
/// Seed ordering: `<I as Backend>::seed` is called BEFORE `head_cfg.init`
/// so that Linear Kaiming/Xavier draws are deterministic (BURN 0.21 docs,
/// `burn-backend-0.21.0/src/backend/base.rs:141`). KanLayer uses its own
/// seeded StdRng on CPU and is independent of the backend RNG.
///
/// Resume: when `experiment.checkpoint` is set, the head weights are loaded
/// from the `.mpk`, and — if the sidecars written by `driver::train` exist —
/// the Adam moments (`*_optim.mpk`) and train-loop position (`*_state.json`:
/// epoch, next mini-batch, rng, sampler permutation + cursor) are restored
/// too, making the resumed run continue exactly where the original left off
/// (same gauge batches, same rho-windows, lr schedule at the true epoch).
/// Checkpoints from before the sidecar scheme resume weights-only with the
/// epoch counter back at 1.
pub fn bootstrap_head_and_state<I>(
    cfg: &Config,
    device: &<Autodiff<I> as burn::tensor::backend::BackendTypes>::Device,
) -> Result<(
    KanHead<Autodiff<I>>,
    TrainState<I>,
    impl Optimizer<KanHead<Autodiff<I>>, Autodiff<I>>,
)>
where
    I: Backend,
    Autodiff<I>: AutodiffBackend<InnerBackend = I>,
{
    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = KanHeadConfig::new(
        head_section.input_var_names.clone(),
        head_section.learnable_parameters.clone(),
        cfg.seed,
    )
    .with_hidden_size(head_section.hidden_size)
    .with_num_hidden_layers(head_section.num_hidden_layers)
    .with_grid(head_section.grid)
    .with_k(head_section.k);

    <Autodiff<I> as Backend>::seed(device, cfg.seed);
    let head: KanHead<Autodiff<I>> = head_cfg.init::<Autodiff<I>>(device);
    let mut optimizer = build_adam::<KanHead<Autodiff<I>>, Autodiff<I>>();

    let mut state = TrainState::<I> {
        head,
        epoch: 1,
        mini_batch: 0,
        rng: ChaCha12Rng::seed_from_u64(cfg.seed),
        resume_sampler: None,
    };

    // Resume from `experiment.checkpoint` if set. The seed-initialised head
    // above doubles as the architecture template; its values are discarded
    // by load_record.
    if let Some(ckpt) = cfg.experiment.as_ref().and_then(|e| e.checkpoint.as_ref()) {
        // CompactRecorder appends `.mpk` itself; accept the path with or
        // without the extension.
        let base = if ckpt.extension().is_some_and(|e| e == "mpk") {
            ckpt.with_extension("")
        } else {
            ckpt.clone()
        };
        state.head = load_kan_head::<Autodiff<I>>(&base, state.head, device)?;
        println!("warm start: loaded KAN head from {}.mpk", base.display());

        // Sidecar 1: Adam moments.
        let optim = optim_base(&base);
        if optim.with_extension("mpk").is_file() {
            optimizer = load_optimizer(&optim, optimizer, device)?;
            println!("warm start: restored Adam state from {}.mpk", optim.display());
        } else {
            println!("warm start: no {}.mpk — Adam starts cold", optim.display());
        }

        // Sidecar 2: train-loop position (epoch, mini-batch, rng, sampler).
        let st_path = state_path(&base);
        if st_path.is_file() {
            let st = load_train_state(&st_path)?;
            state.epoch = st.epoch;
            state.mini_batch = st.next_mini_batch;
            state.rng = st.rng;
            state.resume_sampler = Some((st.sampler_indices, st.sampler_cursor));
            println!(
                "warm start: resuming at epoch {} mb {} (rng + sampler restored)",
                st.epoch, st.next_mini_batch
            );
        } else {
            println!(
                "warm start: no {} — restarting at epoch 1 with a fresh shuffle",
                st_path.display()
            );
        }
    }

    Ok((state.head.clone(), state, optimizer))
}
