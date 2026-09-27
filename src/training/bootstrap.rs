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
    head_base, load_dam_params, load_disagg_head, load_kan_head, load_optimizer, load_train_state,
    optim_base, release_dams_base, release_dams_optim_base, release_dams_optim_path,
    release_head_base, release_optim_base, state_path,
};
use crate::training::driver::{DamTrainer, ReleaseTrainer, TrainState};
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

    // Release-only training (`release_head.routing_checkpoint` /
    // `freeze_routing`): the routing head's WEIGHTS come from another run's
    // checkpoint directory; its optimizer and train-loop state do not (the
    // routing optimizer starts cold and the run at epoch 1). Frozen, the
    // head is detached with `no_grad`, like the disaggregation freeze above:
    // its parameters are not autodiff leaves, so the backward spends nothing
    // on them and the driver takes no routing step (`Config::routing_frozen`).
    // `load_record` keeps the template's `require_grad`, so a frozen head
    // stays frozen across the `experiment.checkpoint` resume below.
    if let Some(rh) = cfg.release_head.as_ref().filter(|_| learned_release) {
        if let Some(dir) = rh.routing_checkpoint.as_ref() {
            head = load_kan_head::<Autodiff<I>>(&head_base(dir), head, device)?;
            println!(
                "routing warm start: loaded routing head weights from {}.mpk \
                 (weights only: optimizer and state.json not read)",
                head_base(dir).display()
            );
            if rh.freeze_routing {
                head = head.no_grad();
                println!(
                    "routing warm start: routing head FROZEN (release_head.freeze_routing); \
                     only the release head trains"
                );
            }
        }
    }
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
        // Per-dam free parameters (rule curve, per-dam T0): one row per dam
        // of the feature table, all zero, with their own Adam (constant
        // `per_dam_lr`, stepped by the driver).
        let dams = if section.has_per_dam() {
            let path = cfg
                .data_sources
                .as_ref()
                .and_then(|d| d.reservoirs.as_ref())
                .expect("use_reservoirs requires data_sources.reservoirs (validated at load)");
            let n = crate::data::store::read_dam_features(path, &section.input_var_names)?.comids.len();
            let params = crate::nn::dam_params::DamParams::<Autodiff<I>>::zeros(
                n,
                section.rule_curve,
                section.per_dam_t0,
                device,
            );
            eprintln!(
                "release head: per-dam parameters for {n} table dams (rule_curve: {}, per_dam_t0: {}, \
                 lr {} constant, l2 {})",
                section.rule_curve, section.per_dam_t0, section.per_dam_lr, section.per_dam_l2
            );
            let optimizer = crate::training::lazy_adam::LazyAdam::new(&params);
            Some(DamTrainer::<I> { params, optimizer })
        } else {
            None
        };
        Some(ReleaseTrainer::<I> {
            head,
            optimizer: build_head_optimizer::<KanHead<Autodiff<I>>, Autodiff<I>>(optim_kind),
            dams,
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
            // The per-dam parameters and their optimizer, when both the run
            // and the checkpoint carry them.
            if let Some(d) = r.dams.as_mut() {
                let dbase = release_dams_base(ckpt_dir);
                if dbase.with_extension("mpk").is_file() {
                    d.params = load_dam_params::<Autodiff<I>>(&dbase, d.params.clone(), device)?;
                    println!("warm start: loaded per-dam parameters from {}.mpk", dbase.display());
                    let doptim = release_dams_optim_path(ckpt_dir);
                    if doptim.is_file() {
                        d.optimizer = d.optimizer.clone().load(&doptim)?;
                        println!("warm start: restored per-dam row-sparse Adam from {}", doptim.display());
                    } else if release_dams_optim_base(ckpt_dir).with_extension("mpk").is_file() {
                        println!(
                            "warm start: {}.mpk is a dense-Adam record from an older build; the \
                             per-dam row-sparse Adam starts cold",
                            release_dams_optim_base(ckpt_dir).display()
                        );
                    }
                } else {
                    println!("warm start: no {}.mpk; per-dam parameters start at zero", dbase.display());
                }
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
