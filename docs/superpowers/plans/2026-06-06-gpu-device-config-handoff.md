# Handoff: config-selectable GPU device (`device:` key)

**Date:** 2026-06-06
**Status:** implementation complete; validation IN PROGRESS — blocked on a
SIGFPE in the new integration test (undiagnosed, see "Where it stands").

## Goal (user request, verbatim intent)

> There must be a config setting for setting the GPU device. Currently this
> hardcodes and defaults to device 0. That is fine but I need to be able to
> switch between them. Validate the devices works with our testing.

Design decision: mirror DDR-Python's top-level `device:` key
(`~/projects/ddr/config/merit_training_config.yaml:8` has `device: 2`) as a
top-level `device: <ordinal>` key in ddrs YAML → `cubecl::cuda::CudaDevice::new(ordinal)`.
We chose proper device threading over a `CUDA_VISIBLE_DEVICES` remap.

## Environment facts

- This box has **8× A100-SXM4-80GB** (`nvidia-smi` indices 0–7), so non-zero
  ordinals are testable here.
- `gdb` on this host is broken (`libssl.so.1.1` missing) — no backtraces that
  way. Try `coredumpctl`, `rust-lldb`, or printf-debugging instead.

## What was changed (all compiles clean; `cargo build` exit 0)

### 1. Config field — `src/config.rs`
- `Config` gained `pub device: usize` (doc: CUDA ordinal, mirrors DDR).
- `ConfigRaw` gained `device: Option<usize>`; `From<ConfigRaw>` maps
  `r.device.unwrap_or(0)`.
- Unit tests added in `src/config.rs::tests`:
  `device_parses_and_defaults_to_zero`, plus `assert_eq!(cfg.device, 0)` in
  `loads_merit_training_yaml`.
- `config/merit_training.yaml` and workspace `ddrs.yaml` (gitignored) both
  gained `device: 0` under the top-level scalars.

### 2. CLI + legacy bins (replace `Device::default()`)
- `src/cli/run.rs`: 3 sites now `cubecl::cuda::CudaDevice::new(pr.config.device)`
  (Train, TrainAndTest, and the `--plot` post-step). Unused
  `use burn::tensor::backend::BackendTypes` import removed.
- `src/bin/train.rs`, `src/bin/train_and_test.rs`: device construction moved
  AFTER config load, uses `cfg.device` / `train_cfg.device`.
- `src/bin/eval.rs`, `src/bin/dump_parameters.rs`: same, `cfg.device`.
- All four bins' now-unused `BackendTypes` imports removed.
- NOT changed: `src/cli/init.rs::run_smoke` still smokes on
  `Device::default()` (device 0). Deliberate — `ddrs init` runs before a
  workspace config exists. Flag to user if they want init to honor `--config`.

### 3. Internal hardcoded-device-0 fixes (the real correctness work)
These were the three sites that would have silently broken a non-zero device
(pattern cache / graph context on dev 0 while tensors live on dev N):

- `src/sparse/cusparse.rs`:
  - New helper `cuda_device_index<B>(device: &B::Device) -> usize` next to
    `compute_client` (same TypeId-assert + pointer-cast pattern).
  - `ensure_cuda_cache` → now generic: `ensure_cuda_cache<'a, B: Backend +
    'static>(pattern: &'a CsrPattern, device: &B::Device) -> &'a CudaPatternCache`.
    Same for `ensure_cuda_cache_mut`. Explicit `'a` ties the return to
    `pattern` (compiler demanded it).
  - `build_cuda_pattern_cache(pattern, device_index)` — was
    `CudaDevice::default()`, now `CudaDevice::new(device_index)`.
  - Forward graph capture (`try_capture_forward`, ~line 2480): `cuDeviceGet(0)`
    → `cuDeviceGet(cuda_device_index::<I>(device))`. The `unsafe` block around
    `device::get` was dropped (it's a safe fn; compiler warned).
- `src/sparse/dispatch.rs`: 3 call sites pass `::<I>(pattern, device)`.
- `src/routing/mmc.rs:227`: `ensure_cuda_cache_mut::<I>(pattern, &self.device)`.
- `src/routing/mmc_op.rs`:
  - `timestep_forward_via_graph` (~1232): gets device via `q_t_at.device()`
    (Autodiff<I>::Device == I::Device) and passes it to `ensure_cuda_cache`.
  - Graph replay (~1312): `cuDeviceGet(0)` → ordinal from
    `cuda_device_index::<I>(&device)` (the `I::float_device(&qt_p)` local).

### 4. Drive-by fix (pre-existing breakage, unrelated to this work)
- `tests/training_verification.rs:548`: `train::<I>(...)` was missing the new
  8th `Option<BatchSource>` arg added by the matched-batch-replay PR (merge
  49c1402). Master's `cargo test` did not compile because of this. Fixed by
  passing `None` (default shuffle), same as other call sites.

### 5. New integration test — `tests/device_selection.rs` (THE BLOCKER)
Single `#[test]` fn (deliberately one fn — cusparse cache contract is
single-threaded) that:
1. gates on ≥2 CUDA devices via `cudarc::driver::result::init()` +
   `device::get_count()`,
2. runs `ddrs::sandbox::smoke::<Cuda<f32,i32>>` on `CudaDevice::new(1)` with
   CPU sparse solver,
3. then with `SparseSolver::Cuda` on dev 0 AND dev 1, asserting
   `|max_q(dev0) − max_q(dev1)| < 1e-3`,
4. then with `use_cuda_graphs: true` on dev 1 (exercises capture+replay
   context binding), same cross-device tolerance.

## Where it stands — the SIGFPE

`cargo test --test device_selection` **dies with SIGFPE (signal 8)** before
printing ANY output, even with `--nocapture` (not even the skip/eprintln
lines). Compilation is fine; the process crashes at/near startup of the test
run.

Undiagnosed. Hypotheses, in the order I'd check:

1. **`cudarc::driver::result::init()` / `device::get_count()` in the gate.**
   No other test in this repo calls raw cudarc init for gating — they ALL use
   `std::panic::catch_unwind(|| { let _d: Dev = Default::default(); })`
   (see `tests/cusparse_ptr_spike.rs:14-22`, `tests/sp10_spike_capture.rs:49-57`).
   The crash being before any eprintln output points at the first statements
   of the test = the cudarc gate. **First move: swap the gate to the
   repo-standard catch_unwind pattern + probe `CudaDevice::new(1)` the same
   way, and rerun.** A scratch probe was started at `/tmp/sigfpe_probe.rs`
   (just calls init + get_count; never compiled/run — finish or delete).
2. If still crashing: bisect by commenting stages out (stage 1 CPU-solver
   only, etc.) to find whether SIGFPE is in smoke on dev 1 (then it's a real
   multi-device bug worth fixing in src/, likely an integer div-by-zero
   somewhere in pattern/scratch sizing) vs in the gating.
3. SIGFPE = integer arithmetic, not float. grep candidates: `%` or `/` on
   `usize` in `cuda_graph/scratch.rs`, `sparse/cusparse.rs` workspace sizing.

## Remaining validation checklist (task #4)

- [ ] Fix/diagnose SIGFPE; get `cargo test --test device_selection` green
      (must actually run on dev 1, not skip — this box has 8 GPUs).
- [ ] Full `cargo test` green (last full run failed ONLY on the
      pre-existing `training_verification.rs` compile error, now fixed;
      rerun to confirm. Note: the prior background run reported exit 0
      misleadingly — read its output, don't trust the code).
- [ ] **Parity gate (CLAUDE.md invariant 1)** — `src/routing/` and
      `src/sparse/` were touched, so MUST run:
      `cargo run --release --example compare_ddr_sandbox`
      → must print ABSOLUTE MATCH (max abs diff < 1e-3 m³/s).
- [ ] KAN parity tests NOT required (no `src/nn/`, no rskan pin change), but
      cheap if paranoid.
- [ ] Optional end-to-end: set `device: 1` in `ddrs.yaml`, run
      `ddrs run --workflow train --max-mini-batches 1` (or the train bin) and
      watch `nvidia-smi` to confirm memory lands on GPU 1. This is the
      validation the user actually asked for in spirit.
- [ ] Clean up `/tmp/sigfpe_probe.rs`.

## Gotchas for the next agent

- `Cargo.lock` + `Cargo.toml` were ALREADY dirty before this session (not
  mine; don't revert blindly).
- `.cargo/`, `.ddrs/`, `ddrs.yaml` are untracked workspace artifacts —
  expected, leave them.
- f32 invariant: nothing in this change may introduce f64 casts in the
  routing core.
- The pattern cache is built ONCE per `CsrPattern` on whichever device the
  first `ensure_cuda_cache` call passes; the doc comment now states a
  pattern lives on exactly one GPU. Multi-GPU-simultaneous is out of scope.
- `cargo test` and other cargo invocations serialize on the build lock —
  don't run two in parallel and misread the interleaved output (this bit me:
  a background `cargo test` finished "exit 0" while actually having failed
  to compile a test crate).
- User-facing docs not yet updated: README "Getting started" + CLAUDE.md
  don't mention `device:`. Consider a one-liner in README's config section
  once validation passes.
