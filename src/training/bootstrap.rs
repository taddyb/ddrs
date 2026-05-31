//! Shared head + state constructor used by the training binaries and the CLI.
//!
//! Extracts the ~12-line KAN-head + TrainState setup that was duplicated
//! across `train` and `train_and_test`. The optimizer is NOT included here
//! because `build_adam` returns `impl Optimizer<M, B>` (an opaque type that
//! cannot be named in a struct field without exposing BURN internals).
//! Callers construct it with one additional line:
//!   `let mut optimizer = build_adam::<KanHead<AB>, AB>();`

use burn::backend::Autodiff;
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::config::Config;
use crate::nn::kan_head::{KanHead, KanHeadConfig};
use crate::training::driver::TrainState;

/// Initialise the KAN head and the mutable training state from `cfg`.
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
/// The optimizer is intentionally excluded — call
/// `build_adam::<KanHead<Autodiff<I>>, Autodiff<I>>()` at the call site.
pub fn bootstrap_head_and_state<I>(
    cfg: &Config,
    device: &<Autodiff<I> as burn::tensor::backend::BackendTypes>::Device,
) -> (KanHead<Autodiff<I>>, TrainState<I>)
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

    let state = TrainState::<I> {
        head: head.clone(),
        epoch: 1,
        mini_batch: 0,
        rng: StdRng::seed_from_u64(cfg.seed),
    };

    (head, state)
}
