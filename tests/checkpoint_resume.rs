//! Unit tests for the folder-based checkpoint layout.
//!
//! A checkpoint is a directory `epoch_E_mb_M/` holding three fixed-name files:
//!
//! ```text
//! epoch_E_mb_M/
//! ├── head.mpk      KAN weights      (CompactRecorder)
//! ├── optim.mpk     Adam moments     (CompactRecorder)
//! └── state.json    epoch/mb/rng/sampler position
//! ```
//!
//! These tests lock the behaviors that make resume exact:
//!   1. the path helpers map a checkpoint dir → the fixed filenames,
//!   2. a full save writes exactly those three files,
//!   3. head weights, Adam moments, and the train-loop state each round-trip.

use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;

use ddrs::nn::{KanHead, KanHeadConfig};
use ddrs::training::{
    build_adam, head_base, load_kan_head, load_optimizer, load_train_state, optim_base,
    save_kan_head, save_optimizer, save_train_state, state_path, TrainCkptState,
};

const SEED: u64 = 42;

fn make_head<B: Backend>(seed: u64, device: &B::Device) -> KanHead<B> {
    KanHeadConfig::new(
        (0..4).map(|i| format!("attr_{i}")).collect(),
        vec!["n".to_string()],
        seed,
    )
    .with_hidden_size(8)
    .with_num_hidden_layers(1)
    .init::<B>(device)
}

/// A fixed (non-random) input so every forward/step is deterministic without
/// touching the backend RNG. `variant` selects distinct inputs so successive
/// steps see different gradients (otherwise Adam's moment history is moot).
fn fixed_input<B: Backend>(variant: usize, device: &B::Device) -> Tensor<B, 2> {
    let off = variant as f32 * 0.37;
    let data: Vec<f32> = (0..24).map(|i| ((i as f32) * 0.1 - 1.0 + off).sin()).collect();
    Tensor::from_data(TensorData::new(data, [6, 4]), device)
}

/// Deterministic scalar fingerprint of a head's response to input variant 0.
fn fingerprint<B: Backend>(head: &KanHead<B>, device: &B::Device) -> f32 {
    head.forward(fixed_input(0, device))["n"].clone().sum().into_scalar().elem()
}

/// Richer fingerprint: the head's `n` output across four probe inputs (24
/// values). Far more discriminating than a single scalar sum.
fn probe<B: Backend>(head: &KanHead<B>, device: &B::Device) -> Vec<f32> {
    let mut out = Vec::new();
    for v in 0..4 {
        let o: Vec<f32> =
            head.forward(fixed_input(v, device))["n"].clone().into_data().into_vec().unwrap();
        out.extend(o);
    }
    out
}

/// L1 distance between two probe fingerprints.
fn l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

type AB = Autodiff<NdArray<f32>>;

/// One Adam step on input `variant`. Generic over the concrete optimizer type
/// that `build_adam` returns.
fn opt_step<O>(
    head: KanHead<AB>,
    opt: &mut O,
    variant: usize,
    device: &<AB as BackendTypes>::Device,
) -> KanHead<AB>
where
    O: Optimizer<KanHead<AB>, AB>,
{
    let loss = head.forward(fixed_input::<AB>(variant, device))["n"].clone().sum();
    let grads = GradientsParams::from_grads(loss.backward(), &head);
    opt.step(0.01, head, grads)
}

// ---------------------------------------------------------------------------
// 1. Path layout
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_dir_maps_to_fixed_filenames() {
    let dir = Path::new("/runs/abc/checkpoints/epoch_3_mb_7");
    // head/optim are recorder *bases* — CompactRecorder appends `.mpk`.
    assert_eq!(head_base(dir), dir.join("head"));
    assert_eq!(optim_base(dir), dir.join("optim"));
    assert_eq!(state_path(dir), dir.join("state.json"));
}

// ---------------------------------------------------------------------------
// 2. Full save writes exactly the three fixed files
// ---------------------------------------------------------------------------

#[test]
fn saving_a_checkpoint_writes_three_fixed_files() {
    let device = Default::default();
    let tmp = tempfile::tempdir().unwrap();
    let ckpt = tmp.path().join("epoch_2_mb_5");
    std::fs::create_dir_all(&ckpt).unwrap();

    let head = make_head::<AB>(SEED, &device);
    let optimizer = build_adam::<KanHead<AB>, AB>();
    save_kan_head(&head_base(&ckpt), &head.valid()).unwrap();
    save_optimizer::<AB, KanHead<AB>, _>(&optim_base(&ckpt), &optimizer).unwrap();
    save_train_state(
        &state_path(&ckpt),
        &TrainCkptState {
            epoch: 2,
            next_mini_batch: 6,
            rng: ChaCha12Rng::seed_from_u64(SEED),
            sampler_indices: vec![0, 1, 2],
            sampler_cursor: 1,
        },
    )
    .unwrap();

    assert!(ckpt.join("head.mpk").is_file(), "head.mpk missing");
    assert!(ckpt.join("optim.mpk").is_file(), "optim.mpk missing");
    assert!(ckpt.join("state.json").is_file(), "state.json missing");
}

// ---------------------------------------------------------------------------
// 3a. Head weights round-trip
// ---------------------------------------------------------------------------

#[test]
fn head_weights_round_trip_through_dir() {
    type B = NdArray<f32>;
    let device = Default::default();
    let tmp = tempfile::tempdir().unwrap();
    let ckpt = tmp.path().join("epoch_1_mb_0");
    std::fs::create_dir_all(&ckpt).unwrap();

    let saved = make_head::<B>(SEED, &device);
    let want = fingerprint(&saved, &device);
    save_kan_head(&head_base(&ckpt), &saved).unwrap();

    // Template with a DIFFERENT seed — if load is a no-op the fingerprints
    // diverge; only a real restore makes them match.
    let template = make_head::<B>(SEED + 1, &device);
    assert_ne!(want, fingerprint(&template, &device), "template must differ");

    let loaded = load_kan_head::<B>(&head_base(&ckpt), template, &device).unwrap();
    let got = fingerprint(&loaded, &device);
    assert!(
        (got - want).abs() < 1e-5,
        "restored head fingerprint {got} != saved {want}"
    );
}

// ---------------------------------------------------------------------------
// 3b. Adam optimizer state is restored on load (next step ≠ cold)
// ---------------------------------------------------------------------------
//
// A loaded optimizer must carry non-trivial Adam state: stepping a head with it
// must produce a DIFFERENT result than stepping the same head with a fresh
// (cold) optimizer, and the SAME result as re-loading the file again
// (deterministic). This is robust to CompactRecorder's f16 moment storage and
// to record-serialization ordering; exact warm==uninterrupted equality is NOT
// claimed (f16 storage of small second moments degrades it — a known recorder
// limitation, intentionally left as-is).

#[test]
fn optimizer_state_restored_on_load() {
    let device = Default::default();
    let tmp = tempfile::tempdir().unwrap();
    let ckpt = tmp.path().join("epoch_1_mb_3");
    std::fs::create_dir_all(&ckpt).unwrap();

    // Build real Adam moments with a few steps on input 0, then save.
    let mut h = make_head::<AB>(SEED, &device);
    let mut opt = build_adam::<KanHead<AB>, AB>();
    for _ in 0..4 {
        h = opt_step(h, &mut opt, 0, &device);
    }
    let h_saved = h.clone();
    save_optimizer::<AB, KanHead<AB>, _>(&optim_base(&ckpt), &opt).unwrap();

    // Warm: load the saved optimizer, take a step on a DIFFERENT input.
    let mut warm_opt =
        load_optimizer::<AB, KanHead<AB>, _>(&optim_base(&ckpt), build_adam(), &device).unwrap();
    let warm = probe(&opt_step(h_saved.clone(), &mut warm_opt, 1, &device).valid(), &device);

    // Warm again (re-load the same file): must be deterministic.
    let mut warm_opt2 =
        load_optimizer::<AB, KanHead<AB>, _>(&optim_base(&ckpt), build_adam(), &device).unwrap();
    let warm2 = probe(&opt_step(h_saved.clone(), &mut warm_opt2, 1, &device).valid(), &device);

    // Cold: fresh optimizer (no restored state), same head + input.
    let mut cold_opt = build_adam::<KanHead<AB>, AB>();
    let cold = probe(&opt_step(h_saved, &mut cold_opt, 1, &device).valid(), &device);

    let warm_vs_cold = l1(&warm, &cold);
    let warm_vs_warm = l1(&warm, &warm2);
    eprintln!("L1 warm-vs-cold={warm_vs_cold}  warm-vs-warm(reload)={warm_vs_warm}");

    assert!(
        warm_vs_warm < 1e-6,
        "two loads of the same optimizer file diverged ({warm_vs_warm}) — load is nondeterministic"
    );
    assert!(
        warm_vs_cold > 1e-3,
        "loaded optimizer stepped identically to a cold one ({warm_vs_cold}) — state not restored"
    );
}

// ---------------------------------------------------------------------------
// 3c. Train-loop state round-trips (epoch / mb / rng / sampler)
// ---------------------------------------------------------------------------

#[test]
fn train_state_round_trips_through_json() {
    let tmp = tempfile::tempdir().unwrap();
    let ckpt = tmp.path().join("epoch_4_mb_9");
    std::fs::create_dir_all(&ckpt).unwrap();

    let mut rng = ChaCha12Rng::seed_from_u64(SEED);
    rng.next_u64(); // advance so the saved stream is mid-flight
    let saved = TrainCkptState {
        epoch: 4,
        next_mini_batch: 10,
        rng: rng.clone(),
        sampler_indices: vec![5, 3, 8, 1],
        sampler_cursor: 2,
    };
    save_train_state(&state_path(&ckpt), &saved).unwrap();

    let loaded = load_train_state(&state_path(&ckpt)).unwrap();
    assert_eq!(loaded.epoch, 4);
    assert_eq!(loaded.next_mini_batch, 10);
    assert_eq!(loaded.sampler_indices, vec![5, 3, 8, 1]);
    assert_eq!(loaded.sampler_cursor, 2);

    // The restored rng must continue the SAME stream as the saved one.
    let mut want = rng;
    let mut got = loaded.rng;
    assert_eq!(want.next_u64(), got.next_u64(), "rng stream diverged after restore");
}
