//! Shared head + state + optimizer constructor used by the training binaries
//! and the CLI.
//!
//! Extracts the KAN-head + TrainState + Adam setup that was duplicated across
//! `train` and `train_and_test`, and centralises checkpoint resume: when
//! `experiment.checkpoint` is set, the head weights, Adam moments, and the
//! train-loop position (epoch, mini-batch, rng, sampler permutation) are all
//! restored from the checkpoint + its sidecars (see `checkpoint.rs`).

use burn::backend::Autodiff;
use burn::module::Module;
use burn::optim::Optimizer;
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::config::Config;
use crate::data::error::Result;
use crate::nn::kan_head::KanHead;
use crate::training::checkpoint::{
    head_base, load_disagg_head, load_kan_head, load_optimizer, load_train_state, optim_base,
    release_head_base, release_optim_base, state_path,
};
use crate::training::driver::{ReleaseTrainer, TrainState};
use crate::training::optimizer::{build_head_optimizer, HeadOptimizer};

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
/// Resume: when `experiment.checkpoint` is set (a checkpoint DIRECTORY
/// `epoch_E_mb_M/`), the head weights are loaded from `head.mpk`, and — if
/// `optim.mpk` / `state.json` exist — the Adam moments and train-loop position
/// (epoch, next mini-batch, rng, sampler permutation + cursor) are restored
/// too, making the resumed run continue exactly where the original left off
/// (same gauge batches, same rho-windows, lr schedule at the true epoch).
pub fn bootstrap_head_and_state<I>(
    cfg: &Config,
    device: &<Autodiff<I> as burn::tensor::backend::BackendTypes>::Device,
) -> Result<(
    KanHead<Autodiff<I>>,
    TrainState<I>,
    HeadOptimizer<KanHead<Autodiff<I>>, Autodiff<I>>,
)>
where
    I: Backend,
    Autodiff<I>: AutodiffBackend<InnerBackend = I>,
{
    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = crate::config::kan_config(head_section, cfg.seed);

    <Autodiff<I> as Backend>::seed(device, cfg.seed);
    let mut head: KanHead<Autodiff<I>> = head_cfg.init::<Autodiff<I>>(device);
    let optim_kind = cfg
        .experiment
        .as_ref()
        .map(|e| e.optimizer)
        .unwrap_or_default();
    eprintln!("optimizer: {optim_kind:?}");
    let mut optimizer =
        build_head_optimizer::<KanHead<Autodiff<I>>, Autodiff<I>>(optim_kind);

    // Warm-start the disaggregation submodule from a standalone-pretrained
    // checkpoint (`kan_head.disaggregation.pretrained_checkpoint`), optionally
    // freezing it. Runs BEFORE the `experiment.checkpoint` resume below: a
    // full-head resume overwrites the *values*, but `Param::transform_for_load`
    // preserves the template's `require_grad`, so a frozen disagg head stays
    // frozen across resume.
    if let Some(d) = cfg.kan_head.as_ref().and_then(|k| k.disaggregation.as_ref()) {
        if let Some(path) = d.pretrained_checkpoint.as_ref() {
            let template = head
                .disagg
                .take()
                .expect("disaggregation section present but head has no disagg submodule");
            let mut disagg = load_disagg_head::<Autodiff<I>>(path, template, device)?;
            println!(
                "disagg warm start: loaded pretrained DisaggHead from {}",
                path.display()
            );
            if d.freeze {
                disagg = disagg.no_grad();
                println!("disagg warm start: froze DisaggHead params (no further training)");
            }
            head.disagg = Some(disagg);
        }
    }

    // The learned dam release head (`params.reservoir_release: learned`): a
    // separate module + optimizer of the same kind, initialised after the
    // routing head from its own seeded RNG, so the routing head's init is
    // untouched. See `nn::release_head` for the pass-through init.
    let learned_release = cfg.params.use_reservoirs
        && cfg.params.reservoir_release == crate::config::ReservoirRelease::Learned;
    let release = if learned_release {
        let section = cfg
            .release_head
            .as_ref()
            .expect("reservoir_release: learned requires a release_head block (validated at load)");
        let head = crate::nn::release_head::init_release_head::<Autodiff<I>>(
            section,
            &cfg.params.parameter_ranges,
            cfg.seed,
            device,
        );
        eprintln!(
            "release head: {} inputs, outputs {:?}, T0 at init {} h",
            section.input_var_names.len(),
            head.learnable_parameters(),
            crate::nn::release_head::INIT_T0_HOURS
        );
        Some(ReleaseTrainer::<I> {
            head,
            optimizer: build_head_optimizer::<KanHead<Autodiff<I>>, Autodiff<I>>(optim_kind),
        })
    } else {
        None
    };

    let mut state = TrainState::<I> {
        head,
        release,
        epoch: 1,
        mini_batch: 0,
        rng: ChaCha12Rng::seed_from_u64(cfg.seed),
        resume_sampler: None,
    };

    // Resume from `experiment.checkpoint` if set. It points at a checkpoint
    // DIRECTORY `epoch_E_mb_M/` holding head.mpk + optim.mpk + state.json. The
    // seed-initialised head above doubles as the architecture template; its
    // values are discarded by load_record.
    if let Some(ckpt_dir) = cfg.experiment.as_ref().and_then(|e| e.checkpoint.as_ref()) {
        state.head = load_kan_head::<Autodiff<I>>(&head_base(ckpt_dir), state.head, device)?;
        println!("warm start: loaded KAN head from {}", head_base(ckpt_dir).display());

        // Adam moments.
        let optim = optim_base(ckpt_dir);
        if optim.with_extension("mpk").is_file() {
            optimizer = load_optimizer(&optim, optimizer, device)?;
            println!("warm start: restored Adam state from {}.mpk", optim.display());
        } else {
            println!("warm start: no {}.mpk — Adam starts cold", optim.display());
        }

        // The learned dam release head and its optimizer. A checkpoint from a
        // run without the release (e.g. warm-starting the routing head from a
        // no-dam run) has neither file, and the release head starts cold.
        if let Some(r) = state.release.as_mut() {
            let rhead = release_head_base(ckpt_dir);
            if rhead.with_extension("mpk").is_file() {
                r.head = load_kan_head::<Autodiff<I>>(&rhead, r.head.clone(), device)?;
                println!("warm start: loaded release head from {}.mpk", rhead.display());
                let roptim = release_optim_base(ckpt_dir);
                if roptim.with_extension("mpk").is_file() {
                    r.optimizer = load_optimizer(&roptim, r.optimizer.clone(), device)?;
                    println!("warm start: restored release optimizer from {}.mpk", roptim.display());
                } else {
                    println!("warm start: no {}.mpk; release optimizer starts cold", roptim.display());
                }
            } else {
                println!(
                    "warm start: no {}.mpk; release head starts cold (T0 = {} h)",
                    rhead.display(),
                    crate::nn::release_head::INIT_T0_HOURS
                );
            }
        }

        // Train-loop position (epoch, mini-batch, rng, sampler).
        let st_path = state_path(ckpt_dir);
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
