# Handoff: full checkpoint resume (weights + Adam + batch position)

**Date:** 2026-06-07 · **Branch:** `wukong-tests` · **Validated on:** wukong (A100, CUDA)

## What this is

`experiment.checkpoint:` in `ddrs.yaml` now resumes training **exactly** —
not just warm-starting weights.

> **Layout update (2026-06-07, post-handoff):** checkpoints are now a
> **directory** `epoch_E_mb_M/` with fixed inner filenames, not flat
> `epoch_E_mb_M*` siblings. `experiment.checkpoint:` points at the directory.
> This retires the `set_extension("mpk")` underscore-suffix hack below and
> makes a checkpoint copy/delete/gc as one unit. Path helpers:
> `head_base(dir)`→`dir/head`, `optim_base(dir)`→`dir/optim` (recorder appends
> `.mpk`), `state_path(dir)`→`dir/state.json`. Unit tests:
> `tests/checkpoint_resume.rs`.

Every per-mini-batch checkpoint written by `ddrs run` (and the legacy
`train`/`train_and_test` binaries) is a directory of three files:

| File (in `epoch_E_mb_M/`) | Contents | Format |
|---|---|---|
| `head.mpk` | KAN head weights | burn CompactRecorder |
| `optim.mpk` | Adam record (both moment tensors) | burn CompactRecorder |
| `state.json` | epoch, next mini-batch, serialized rng, sampler permutation + cursor | JSON |

On resume, `bootstrap_head_and_state` (`src/training/bootstrap.rs`) restores
all three, so the resumed run:

- continues at the **true epoch / mini-batch** (lr schedule keyed correctly),
- draws the **same gauge batches** — including the rest of an in-flight
  epoch's shuffle — and the **same rho-windows** the original run would have,
- steps Adam with **warm moments** instead of restarting cold.

## How to use

```yaml
# ddrs.yaml
experiment:
  epochs: 50        # must exceed the checkpoint's epoch or nothing trains
  checkpoint: .ddrs/runs/<run-id>/checkpoints/epoch_25_mb_8.mpk   # .mpk optional
```

then `ddrs run --workflow train` (or `train-and-test`). Expected log:

```
warm start: loaded KAN head from .../epoch_25_mb_8.mpk
warm start: restored Adam state from .../epoch_25_mb_8_optim.mpk
warm start: resuming at epoch 25 mb 9 (rng + sampler restored)
```

**Old checkpoints** (pre-sidecar, e.g. anything from before 2026-06-07)
resume weights-only: Adam cold, epoch counter back at 1, fresh shuffle. The
log says which sidecars were found.

## Validation performed (wukong, 2026-06-07)

Reference: run A = cold 2-epoch run (9 mb/epoch, batch 256, seed 42).
Test: run B = resume from A's **mid-epoch** `epoch_1_mb_4` checkpoint.

- B logged `resuming at epoch 1 mb 5 (rng + sampler restored)`.
- B's first batch (`mb=5 loss=7.453495`) matched A's (`7.453494`) to the last
  f32 digit → identical gauges, window, weights, and Adam state.
- B's remaining losses track A's within 0.02 %→0.8 % (growing): this drift is
  the **f16 checkpoint roundtrip** (burn's `CompactRecorder` stores half
  precision — pre-existing behavior, also applies to the train→eval handover),
  compounded through training. It is NOT batch-order divergence: A and B agree
  batch-for-batch on the loss-trajectory shape.
- Run B′ (identical resume rerun) reproduced B bitwise → the drift is
  deterministic rounding, not nondeterminism.
- Negative paths: missing checkpoint → clean `FileNotFound` with path in the
  run manifest; missing sidecars → weights-only with explicit log lines.

Cold-start determinism on this machine was also confirmed (two cold runs gave
bitwise-identical first-batch loss), which is what makes the comparison above
meaningful.

### To re-validate on the desktop

```bash
cargo build --release --bin ddrs
# A: cold reference (use a scratch config: workflow: train, epochs: 2)
./target/release/ddrs --config <cfg-A> run --workflow train   # note the losses
# B: add to cfg:  checkpoint: .ddrs/runs/<run-A>/checkpoints/epoch_1_mb_4.mpk
./target/release/ddrs --config <cfg-B> run --workflow train
# expect: the three "warm start:" lines; mb=5 loss equal to run A's mb=5
# to ~1e-7; later losses within ~1% (f16 checkpoint rounding).
cargo test --lib                                              # 131 pass
```

## Code changes

| File | Change |
|---|---|
| `src/training/checkpoint.rs` | `TrainCkptState`, `save/load_optimizer`, `save/load_train_state`, `optim_base`/`state_path` helpers + module docs |
| `src/training/driver.rs` | `TrainState.rng` is now `ChaCha12Rng` (identical stream to rand 0.8's `StdRng`, but serde-serializable); new `resume_sampler` field; epoch loop restores the sampler instead of reshuffling on resume; writes both sidecars next to each head checkpoint |
| `src/training/bootstrap.rs` | builds the Adam optimizer (now returned as the 3rd tuple element) and performs the full restore; callers no longer call `build_adam` themselves |
| `src/data/sampler.rs` | `RandomSampler::snapshot/restore` (+ `BatchSource` passthrough; `Replay` is exempt — it has its own batch record) |
| `src/cli/run.rs` | `latest_checkpoint_base` now parses `epoch_E_mb_M` numerically — fixes a **pre-existing bug** where lexicographic sort ranked `epoch_9_*` above `epoch_25_*`, so train-and-test phase 2 evaluated the wrong checkpoint; also skips `_optim` sidecars |
| `src/bin/train_and_test.rs` | `find_latest_mpk` skips `_optim` sidecars (it picks by mtime, and the optim file is written immediately after the head file) |
| `Cargo.toml` | + `rand_chacha = { version = "0.3", features = ["serde1"] }` |
| `tests/training_verification.rs` | `TrainState` literal updated for the new fields |

## Known limitations / follow-ups

1. **f16 checkpoint storage** (`CompactRecorder`): resume is exact in state
   but the stored weights/moments are half-precision, so a resumed trajectory
   slowly drifts from the uninterrupted one (~1 % after a dozen batches).
   Switching to `DefaultFileRecorder<FullPrecisionSettings>` would make
   resume bit-exact *and* make phase-2 eval see exactly the trained weights —
   but it's a checkpoint-format break (old `.mpk`s unreadable) and double the
   size, so it was left as is. Decide deliberately.
2. **`BatchSource::Replay`** (matched-batch experiment) doesn't write the
   state sidecar; resuming a replay run is weights+Adam only, by design.
3. A skipped mini-batch (all-NaN gauges) writes no checkpoint — resume from
   the previous one; the rng/sampler replay reproduces the skip.
4. The old run `2026-06-06T21-29-45Z-train-and-test` predates the sidecars —
   resuming from its `epoch_25_mb_8` warm-starts weights only. With the lr
   schedule restarting at epoch 1 (lr=0.001 instead of 0.0005), consider
   flattening `learning_rate:` to `{1: 0.0005}` for that resume.
