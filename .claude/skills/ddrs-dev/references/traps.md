# Traps

Deduplicated from six retired skills. Each entry leads with the **discriminating
test** — the cheap check that tells you whether this is your problem — because that
is what a future session needs, not the narrative.

## Symptom → trap

| Symptom | Start at |
|---|---|
| Config change appears to do nothing; two arms byte-identical | T1, then "no `deny_unknown_fields`" in `config.md` |
| `compare_ddr_sandbox` reports a mismatch | T2 |
| Loss is finite but suspiciously constant; eval all-NaN | T3 |
| Baseline NSE is far worse than the trained model | T4 |
| Trained loss is flat across epochs | T5 |
| Eval phase hangs at ~900% CPU producing nothing | T6 |
| Long CPU forward dies with no error in its own log | T7 |
| Mid-eval `object not found` from icechunk | T8 |
| `.ddrs/` appeared somewhere unexpected | T9 |
| Resumed run trains zero batches | T10 |
| Baseline predicts ~0 (FHV −100 %) on a new Q′ store, no error | T11 |
| Fresh checkout/worktree fails in `cudarc`'s build script: `Unsupported cuda toolkit version` | T12 |
| Training crawls; GPU is resident but idle at single-digit utilisation | T13 |
| Eval metrics belong to a different model than the one trained (a new head output changes training loss but not eval) | T14 |
| Training runs 6x slower than the 3.5 s/micro-batch pace on a new Q′ store, CPU pegged in decompression | T15 |
| A forward-only sweep (grid search, sensitivity scan) grows RSS without bound | T16 |
| A log-space parameter never moves off one end of its range, yet gradient checks pass | T17 |
| `training_step_layer_b` step3 (MC forward max abs 1.01e-1) and `training_step_layer_c` step2 fail locally, green in CI | T18 |
| Same checkpoint, same gauge, different prediction when evaluated with a different gauge CSV | T19 |
| Stage op CPU and CUDA differ by 1 ulp, then diverge on a stiff cell; or an edit to a stage kernel changes CPU/GPU bits | T20 |
| The stage reference step (`step_reference`) stops matching the fused op after a cross-section change | T21 |
| CPU stage training got about 1.3x slower after the fused multiply-add change | T22 |

---

## T1 — Stale binary

`~/.cargo/bin/ddrs` is not updated by `cargo build` or `cargo run`. The run manifest
stamps `git.sha` from `.git` at runtime, so the manifest looks correct while a
weeks-old binary executed.

**Discriminating tests**
- Checkpoints must be **directories** (`epoch_E_mb_M/{head,optim,state}`). Flat
  `epoch_E_mb_M.mpk` ⇒ pre-checkpoint-resume binary.
- `stat ~/.cargo/bin/ddrs` — compare mtime to your last `src/` edit.
- Head-file size, measured: ~103,459 B no-disagg · 107,178 B disagg ·
  107,320 B disagg + leakance.

**Fix**: `cargo install --path .` (canonical), or
`cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs`, or
bypass with `cargo run --release --bin ddrs -- run …`.

**The incident that makes this stick (2026-07-01).** The installed `ddrs` was dated
2026-06-03 — before disaggregation (2026-06-19) and leakance (2026-06-29). The
hourly-forcing cell silently ran flat repeat-24 with no leakance. hourly-ON and
daily-ON produced **byte-identical** eval predictions. The manifest showed
`git.sha = 2cdd341` (the correct HEAD), masking it completely. A second occurrence
came from running `cargo install --path .` in the *main tree* while working in a
worktree, silently replacing the worktree build.

## T2 — DDR sandbox mismatch

**Triage order:**
1. `mkdir -p output && cargo run --release --example compare_ddr_sandbox`, then also
   with `DDRS_FORCE_GRAPHS=1`. (Note: that env var only selects the CUDA *backend* —
   it does **not** enable graph capture, which additionally requires
   `use_cuda_graphs && sparse_solver == Cuda`, and the sandbox config sets neither.)
2. Inspect `output/ddrs_vs_ddr.csv`. **One bad reach ⇒ geometry/parameters.
   Globally bad ⇒ solver/kernel.**
3. `grep -rn "f64\|bf16\|cast\|to_dtype" src/routing/ src/geometry.rs src/sparse/`
   — invariant 2 says the core is f32 throughout.
4. `cargo test --test sparse_gradcheck`. **Also failing ⇒ the algorithm changed.
   V1-only ⇒ kernel ordering or arithmetic fusion.**

**Wrong-reference failure mode:** a fixture regenerated from a PRE-#192 DDR
clone diverges ~1% (max abs ≈ 0.55 m³/s) at *every* ddrs commit. Since
2026-08-19 any DDR checkout at or past DeepGroundwater/ddr#192 is a valid
reference (fixture + comparison both on corrected physics) — see
`build-and-env.md`.

## T3 — CUDA graphs mask NaN

`use_cuda_graphs: true` captures a kernel graph on the first forward and replays it.
If the first forward produced a NaN, the replay returns stale finite values instead
of propagating it — a constant, plausible-looking loss over a corrupt computation.

**Discriminating test:** rerun with `use_cuda_graphs: false`. If the loss now goes
NaN, this was it.

**Origin:** AORC precip carries ~14% genuine NaN (ocean / no coverage) → `log1p` →
NaN softmax → all 2,365 gauges NaN at eval while training loss looked healthy. Fixed
in `a5972d9` (zero-fill sanitization).

**Rule:** validate every new data path with graphs off before enabling them.
`use_leakance: true` + `use_cuda_graphs: true` is rejected at config load.

## T4 — Phantom-zero baseline (fixed 2026-07-28/29)

Single-divide (zero-edge) gauges have no edge endpoints, so `upstream_comids()`
returned an empty set and the NaN-filtered sum produced **exactly 0.0** predictions
scored against real observations. 513 of 3,211 gauges (16%) at median NSE −0.305,
dragging the baseline median from 0.315 to 0.142.

**Fix**: `valid_gauges()` skips them via `GageSubgraph::is_headwater()`, matching
training's own filter. The cache key now carries `BASELINE_ALGO_VERSION =
"baseline-algo-v2"`, hashed first.

**Diagnostic**: count all-zero rows in `<workspace>/baselines/<key>/predictions.f32`.
Any nonzero count on a post-fix binary is a new bug.

**Generalized rule — the reusable part:** *the baseline must score the same gauge
population the model trains and evaluates on.* Skip filtered gauges; never impute.
This also means the "own baseline" columns in the 2026-07-07 and 2026-07-16 findings
docs are population-mismatched (3,211-gauge baseline vs 2,365-gauge trained medians)
— see `research-status.md`.

## T5 — Flat training loss

**When training loss is flat, fix the gradient path — do not try new loss
functions.** Two separate loss experiments (L1 vs NNSE-KGE; α-weighted KGE with a
learnable Muskingum X) both produced flat loss. Learnable X sat at its init (median
0.246, p10–p90 0.214–0.253, 0% at either bound despite α-weight 2), i.e.
∂loss/∂(routing params) ≈ 0.

**Root cause:** daily `repeat-24` upsampling plus daily-mean aggregation puts
routing's within-day effect in the gradient **null space**. The disaggregation head
was the only change that ever made loss descend (1.224 → 1.02) and X move
(0.246 → 0.217).

**Corollary:** fixing an optimization barrier can expose a structural ceiling.
Bare (precip-free) disagg unsticks the gradient but overfits (held-out KGE 0.624);
real precip timing is what rescues KGE. Once the gradient flows, `nnse-kge` does
*not* fix the KGE gap (0.7100 vs L1's 0.7106) and a temperature channel does not earn
its keep (0.7155/0.7088 vs 0.7152/0.7106).

## T6 — GPU eval-phase OOM that never propagates

`--backend cuda` `train-and-test` trains fine, then Phase 2 panics inside a
cubecl-cuda worker thread: `thread 'DSD-0-0' panicked … can't allocate buffer of
size: 4178264064`. **The panic does not propagate.** The main thread loops all
chunks at ~900% CPU writing a half-written GPU buffer.

Happens **solo on the GPU** — Phase 1 alone leaves too little headroom for Phase 2's
~4.2 GB per-chunk buffer at `testing.batch_size: 15` on a 16 GB card. Confirmed
twice independently (2026-07-16).

**Detection landed 2026-07-16** in `src/training/eval.rs`: two layered detectors
returning `DataError::CorruptedEvalChunk` — `corrupted_chunk_reason` (all-zero /
non-finite) and a process-global panic hook (`WORKER_PANICKED` /
`ensure_panic_hook_installed`) checked after every chunk. The first detector **alone
missed incident #2** — the stale buffer looked plausible. Gate:
`cargo test --lib training::eval::tests`.

**Recovery without retraining:**
```bash
~/.cargo/bin/eval --config <run_dir>/config.yaml \
  --checkpoint <run_dir>/checkpoints/epoch_E_mb_M \
  --output <run_dir>/eval/predictions.zarr --backend cpu
```

**Operational rule:** do not co-schedule two `--backend cuda` trainings. Phase 1
alone is ~9.9 GB on a 16 GB card; the second OOMs within seconds. Give each arm a
different backend from the start.

## T7 — Silent kernel OOM on long CPU forwards

`probe_zeta_gradient --mode teacher` at the default `--chunk-days 365` peaks ~65 GB
RSS on the 64,892-reach network and gets kernel-OOM-killed with **nothing in its own
log** — output just ends after `teacher: N plants, …`.

**Discriminating test:** `journalctl -k | grep -i 'oom\|killed process'`.
RSS climbing *within* a chunk and collapsing to ~4 GB at boundaries is the signature.

**Fix:** `--chunk-days 180` (~45 GB), at the cost of doubling disagg boundary
artifacts (0.55% → 1.1% of days). State continuity is exact either way.
**Caveat:** `run_state_cache` still hardcodes 365 — chunk lengths must match if you
pair a state cache with a teacher run.

## T8 — Transient icechunk read, not a data hole

A mid-eval `object not found` after hundreds of clean chunks is transient. These Q′
stores are **divide-major** (`Qr` chunk = 200 divides × ALL 14,976 days), so any
time-slice read touches every chunk object — chunk 364 cannot be missing an object
that chunks 1–363 already read.

**Confirm:** open the store directly in Python (`icechunk.Repository.open`) and run
`ddrs import <store> --dry-run`. A clean read confirms transient.

**Recovery:** training and eval are separate phases, so the final checkpoint stays
valid. Treat "nonzero exit but final checkpoint exists" as continue-with-warning.

## T9 — `.ddrs/` beside the config

`Workspace::beside(config)` is the default, so `--config config/experiments/x.yaml`
creates `config/experiments/.ddrs/`. `--workspace` takes the `.ddrs` **directory
itself**, not its parent. Always pass
`--workspace /home/tbindas/projects/ddrs/.ddrs`.

## T10 — `--checkpoint` means different things per binary

| Binary | Argument |
|---|---|
| `eval`, `probe_zeta_gradient` | the **directory** `epoch_E_mb_M` |
| `dump_parameters` | the **head base** `epoch_E_mb_M/head` (no `.mpk`) |

Resume additionally requires `experiment.epochs > E` or the resumed run trains zero
batches. Stored weights are f16, so a resumed trajectory drifts slowly — expected,
not a bug. Since 2026-09-27 a `train-and-test` resume that takes no step (the resumed
epoch is the last and its sampler is exhausted) no longer fails the test phase with
`no .mpk checkpoints found after Phase 1`: it logs `Phase 1 took no optimizer step and
wrote no checkpoint; testing the resumed checkpoint <dir>` and evaluates that
checkpoint. That is the in-engine replay recipe for a fixed dam table
(`config/experiments/dam_release_smoke_replay_L2.yaml`); `juniata_acceptance` pins it
bitwise against the run it resumes.

## T11 — Time-major `Qr(time, divide_id)` read as fill values (fixed 2026-09-09)

DDR's gridded (DDM30) `build_gridded_qprime.py` writes `Qr(time, divide_id)`;
the MERIT stores and the contract are `Qr(divide_id, time)`. zarrs answers an
out-of-range `retrieve_array_subset` with **fill values, not an error**, so the
old fixed-axis `read_slab` returned NaN for every cell and the summed-Q′
baseline scored NSE −0.655 with FHV −100 % — no message anywhere. `open` now
decides the axis order from the array's `dimension_names`, falling back to its
shape against the coordinate lengths, and refuses a shape matching neither
(`detect_time_major`, `src/data/store/icechunk.rs`). Fixture:
`tests/fixtures/qr_daily_time_major.ic`. A baseline that is *deterministic and
matches DDR* is the check that catches this class of bug; that is why both
acceptance tests band the baseline tightly.

## T12 — CUDA toolkit newer than `cudarc` knows (2026-09-09)

`cudarc 0.19.7` (via `cuda-version-from-build-system`) parses `nvcc --version`
and panics on a version outside its table (max 13.2). This host moved to CUDA
13.3.1 in 2026-09; the main tree's `target/` still carried the 13.2 build-script
output, so only a fresh worktree/checkout hit it. Fix: `CUDARC_CUDA_VERSION=13020`
(gitignored `.cargo/config.toml` `[env]` block, or export it). 13.3 is
driver/ABI-compatible with 13.2 for the dynamic-loading path ddrs uses. The
main tree will hit the same panic the next time its cudarc build script reruns
(any `CUDA_*` env change or a `cargo clean`).

## Exit codes

`src/cli/types.rs`: 0 Success · 1 Generic · 2 ConfigInvalid · 3 DataSourceMissing ·
4 LockDrift · 5 RuntimeFailure · 6 WorkspaceNotInitialized.

`ddrs run --strict` aborts with 4 *before* relocking, preserving the drift evidence;
plain `plan` reports drift then refreshes the lock.

## Pre-flight before a long run

1. `cargo install --path .` — refresh the binary.
2. `mkdir -p output && cargo run --release --example compare_ddr_sandbox` — V1 gate.
3. Confirm `--workspace` points at the repo-root `.ddrs`.
4. Confirm `--backend` matches `params.sparse_solver` intent.
5. Smoke it: `ddrs run --workflow train --max-mini-batches 2`.
6. Grep the smoke log for `streamflow resolution:` and the gauge-filter line.
7. Check no other CUDA job is resident if you are using `--backend cuda`.

## T13 — CUDA is the SLOW path for training this model (2026-09-11)

**Symptom.** `ddrs run --workflow train-and-test` with the default backend takes
~20 s per micro-batch. `nvidia-smi` shows the process holding ~1.4 GB of device
memory at **7 % utilisation**, and the host process using only ~2 of 32 cores.

**Measured, same config, same 2,365 gauges, 64-gauge micro-batches:**

| backend | per micro-batch | 50 epochs |
|---|---|---|
| cuda | ~20 s | ~11 h |
| **cpu** | **~3.5 s** | **~97 min** |

**Why.** The sparse triangular solve is sequential in the topologically ordered
reach index, so there is nothing to parallelise within a timestep. CUDA pays
kernel-launch overhead per timestep and recovers none of it. The 7 %
utilisation is the tell.

**Do not misdiagnose this as resource contention.** Six concurrent CPU-bound
landscape shards were the obvious suspect; pausing all six with `SIGSTOP` moved
the CUDA rate only from 22 s to 20 s, which is what identified the backend
rather than the neighbours. Measure before you blame.

**Consequence for comparisons.** The reference CONUS results were produced on
CPU: run `2026-09-10T21-21-48Z-conus-train-and-test` (NSE 0.7376 / KGE 0.7600,
the current best) logs `backend: cpu (NdArray, deterministic; sparse_solver
forced to cpu)` on its first line. Any new arm meant to be compared against it
must also run `--backend cpu`, both for speed and because NdArray is
deterministic while CUDA scatter-add is not.

So `--backend cpu` is not only the diagnostics convention (see the standing
preference for CPU in experiments); for training on this network it is simply
faster. Check `head -1 <run>/run.log` to see which backend any historical run
actually used before quoting its wall clock.

### The stage op on CUDA (measured 2026-11-02)

The stage block has CUDA arms now, and the same rule holds: CUDA is slower for the
examples. `tests/stage_cuda_scale.rs` times the fused op (24 timed hours, forward
plus one backward, warm-up history cut) on uniform input. Release build, idle machine,
one sample per row. Theta is 1.0 except Juniata and Hudson at 4 substeps (0.6).

| network | reaches | substeps | CPU ms/h | CUDA ms/h | CUDA/CPU |
|---|---|---|---|---|---|
| juniata | 213 | 1 | 0.168 | 2.05 | 12.2 |
| juniata | 213 | 4 | 0.454 | 8.32 | 18.3 |
| hudson | 756 | 1 | 0.498 | 2.08 | 4.17 |
| hudson | 756 | 4 | 1.536 | 8.49 | 5.53 |
| CONUS sub-basin | 1,000 | 1 | 0.653 | 2.09 | 3.20 |
| CONUS sub-basin | 1,000 | 4 | 2.024 | 6.55 | 3.23 |
| CONUS sub-basin | 10,002 | 1 | 6.51 | 1.62 | 0.25 |
| CONUS sub-basin | 10,002 | 4 | 20.46 | 6.34 | 0.31 |
| CONUS sub-basin | 51,441 | 1 | 36.2 | 5.09 | 0.14 |
| CONUS sub-basin | 51,441 | 4 | 110.9 | 19.7 | 0.18 |
| CONUS full | 346,321 | 1 | 245.4 | 18.4 | 0.07 |
| CONUS full | 346,321 | 4 | 765.3 | 71.6 | 0.09 |

**Crossover: about 10,000 reaches.** CUDA is 3 to 18 times slower at 1,000 reaches and
below, and 11 to 13 times faster on the full CONUS network. The small-network CUDA
rows are launch bound and move by up to about 1.5x between runs of one binary (the
CPU rows stay within about 1 %). Use them as a range, not a point.

**End to end it is far slower.** Seconds per mini-batch (`run.log`, 1 s resolution,
epoch 1 skipped):

| example | CPU | CUDA | CUDA/CPU |
|---|---|---|---|
| Juniata stage, rho 90 | 0.615 | 135.3 | 220 |
| Juniata stage, 4 substeps | 1.316 | 465.0 | 353 |
| Juniata MC only (no stage block) | 0.263 | 32.3 | 123 |
| Hudson | 4.333 | 335.3 | 77 |

The MC-only example is 123 times slower on CUDA, so most of the gap is in the engine
path and not in the stage op. For one rho 90 Juniata batch the scale table predicts 3.0
to 4.4 s for the op alone, against 135 s measured. The rest was not profiled. Keep
`--backend cpu` for both examples.

**Why.** Small networks are launch bound: Juniata at 1 substep makes about 110 kernel
launches per hour (2,852 for 26 hours), the GPU kernels total 7.5 ms inside about 46 ms
of wall time, and the GPU sits idle between launches. At full CONUS one mass-balance
`float_sum_dim` per hour (the reduction in `src/routing/stage_cuda/forward.rs::stage_readout_dev`)
runs as a single cube of 32 threads. It takes 11 ms per call and 62 % of the GPU time
(cuSPARSE `spsm_v2` 19 %, `tree_factor_kernel` 11 %). Without it CONUS would run near
7 ms per hour. This is not optimized.

**Theta 0.6 goes NaN on large CONUS networks, on the CPU path too.** With the scale test's
uniform input (n 0.035, p 21, q 0.5, q' 1.0 on every reach) and the normal-depth hotstart,
the stage scheme blows up at theta 0.6: at 1 substep on the 51,441-reach basin and the
full network (about hour 20), and at 4 substeps on the full network (about hour 23).
Theta 1.0 is stable on every row, so the scale test runs the CONUS rows at theta 1.0. The
op does the same work at any theta.

Cause (measured on the CPU, 2026-11-02): a nonlinear instability of the theta < 1 scheme,
set off by the start state. The hotstart puts each reach at its own normal depth, so
tributaries and short steep reaches start up to 40 m (51 m on the full network) below the
water surface downstream. Each such edge gives a backflow predictor with almost no
friction, because the friction term is linearised about `|Q^t|`, which is small. At
theta < 1 the `(1 - theta)*Q^t` term flips the flux sign on storage-poor reaches every
step (factor `-(1 - theta)/theta`, -0.67 at theta 0.6). Each flip passes `|Q|` through
zero, so the friction collapses again, and the amplitude grows 5 to 30 times per hour
until f32 overflows. At theta 1.0 the same overshoots decay in about 13 hours.

It predates the CUDA work: the commit before the fused multiply-adds and the reciprocal
solve fails the same way at the same stability boundary, and only the NaN hour moves. The
CPU/CUDA divergence of the scale test from hour 3 fits the same chaos, not a kernel error.

Stability map (uniform input, hotstart, 100 hours): the critical theta falls as substeps
rise, from between 0.65 and 0.7 at 1 substep to 0.55 to 0.65 at 4 and 0.5 to 0.6 at 8.
Juniata stays finite from 0.5 to 1.0. The Hudson network fails only at theta 0.5 with
1 substep.

A consistent start (fill each reach up to the stage downstream, or run 24 hours at theta
1.0 first) keeps theta 0.6 finite on the 51,441-reach basin, but a disturbance of 100 % of
the flow brings the blow-up back. **No fix is chosen.** Candidates: a consistent hotstart,
theta 1.0 for non-tidal networks (the current setting), theta 1.0 only on storage-poor or
floored edges, and a friction linearisation that cannot collapse. Not checked: the Hudson
tidal input with its table geometry, the engine's real start state, and the backward pass.

---
## T16 — Forward-only Autodiff evaluations retain the tape

**Symptom.** A process that runs many routing forwards without training (grid sweeps, sensitivity scans) grows
without bound: about 18 MB per forward on the 5-reach sandbox at 2,142 steps, 20 MB per eval at Newport (213
reaches, 4 windows of 90 days); the 41x41 landscape grid reached 77 GB RSS.

**Cause.** `MuskingumCunge<I>` is written only for `Autodiff<I>`; every forward builds a tape, and the tape is
released only when a `.backward()` consumes it. Untracked tensors (no `require_grad`) do not help: the tape is
still retained (`examples/leak_probe.rs`, modes `ad-notrack` 17.7 MB/iter, `ad-track-nobackward` 25.3 MB/iter,
`ad-track-backward` 0.0). Training and the adjoint studies never see it because they always call backward.

**Discriminating test.** `cargo run --release --example leak_probe -- ad-notrack 40` prints RSS per iteration and
the slope; a leak-free path has a slope near 0 MB/iter.

**Fix / rule.** Any forward-only evaluation on the Autodiff backend must end with a backward on the scalar it
computed (mark one leaf `require_grad`, call `backward()`, discard the grads). `Objective::eval` in
`src/experiment/landscape/objective.rs` does this since `658cbfc`; numerics are unchanged and the cost is one
backward per eval (about 2x the forward-only time). Reading `.inner()` does not release anything. A tape-free
forward would need the engine to be generic over the inner backend; not done.

## T14 — A head output the eval path never reads (2026-09-12)

`src/training/forward.rs` has THREE hand-written readers of the KAN output
map, kept deliberately separate (WET): `forward` (training),
`forward_eval_core` (eval, behind `forward_eval` / `forward_eval_reaches`), and
`probe_forward` (`src/training/probe.rs`). Each builds `SpatialParameters` by
naming every key. A key added to `forward` but not to the other two is not an
error: the solver takes its fallback (`None` ⇒ the config scalar, ⇒ 0 for
`gamma`) and the eval scores a model that was never trained. No warning, no
NaN, metrics look plausible.

That is what the first learned-`gamma` CONUS arm did
(`2026-09-12T13-38-27Z`, stopped in eval). The landscape objective
(`src/experiment/landscape/objective.rs`) has the same shape and now REFUSES a
learned-gamma arm rather than routing it at `gamma = 0`.

**Rule.** A new `SpatialParameters` field is added to all three readers in the
same commit, and to the table in `tests/gamma_eval_parity.rs`, which routes one
head through `forward` and `forward_eval` and asserts the hydrographs agree.
That test is the discriminator: it fails at ~2e-2 relative when a reader is
missing the key and passes at f32 round-off when it is not.

## T15 — A Q′ store chunked along the full time axis (2026-09-13)

The trainer reads a 90-day window for a few hundred divides per batch. A store chunked `(divides, full
time)` — the dHBV2 and hydroDL AORC2f products were `(200, 14976)` and `(100, 14976)` — forces every batch
to decompress whole 41-year rows: 72–96 s per accumulated mini-batch against 13 s on the retrospective's
`(3080, 468)` layout, with the load average at 50–60 from zstd threads and three parallel arms pegged.
Symptom looks like CPU oversubscription; it is I/O amplification.

**Fix.** `scripts/rechunk_qprime_store.py <src.ic> <dst.ic>` copies Qr, divide_id and time into a new
icechunk repo with `(3080, 468)` chunks (30 s per store, verified bit-identical), and the configs point at
the `_rechunked.ic` copy. Check `chunks` on any new product before training on it:
`zarr.open_group(repo.readonly_session("main").store)["Qr"].chunks`. The hourly-native store is chunked
`(3080, 11232)` and is inherently ~10x slower per batch (2.2 min per mini-batch); budget for it.

## T17 — A log-space parameter box silently collapsed by the epsilon guard (fixed 2026-09-15)

**Symptom.** A log-space parameter never moves. Its trained values sit in a razor-thin
band at one end of the declared range on essentially every reach, the band looks like a
saturated bound, and every gradient check on the parameter passes.

**Cause.** `src/routing/utils.rs::denormalize` computed its log-space lower bound as
`(lo + 1e-6).ln()` unconditionally. The `1e-6` exists to keep `ln` finite when `lo == 0`,
but it swamps any lower bound that is small next to `1e-6`. `k_d = [1e-8, 1e-6]` became
`[1.01e-6, 1e-6]`: a band 1 % wide, at the top of the box, and **inverted**, since the
guarded lower bound sits above the upper bound. `K_D` was therefore a frozen constant in
every leakance run ever executed, not a learned parameter.

| box | `u = 0` | `u = 1` | log span | intended span |
|---|---|---|---|---|
| `[1e-8, 1e-6]` (`k_d`) | 1.01e-6 | 1.00e-6 | **-0.0100** (inverted) | +4.6052 |
| `[1e-8, 1e-5]` | 1.01e-6 | 1.00e-5 | +2.2926 | +6.9078 |
| `[1, 200]` (`p_spatial`) | 1.0000010 | 200.0 | +5.2983 | +5.2983 |

**Why nothing caught it.** The leakance gradient checks
(`tests/leakance_gradcheck.rs`) feed *physical* parameter values straight into the
routing forward, so they never cross the `sigmoid -> denormalize` seam where the box
lives. A collapsed box is invisible to every test that starts below it: the gradients
really are exact, the parameter really is differentiable, and the reachable set is
still a valid set of numbers. Only `p_spatial`'s `[1, 200]` box was exercised
end-to-end, and there the lower bound is large enough that the guard is a rounding
error.

**Discriminating test.** `cargo test --test routing_utils
denormalize_log_space_tiny_lower_bound`, which asserts the `[1e-8, 1e-6]` box maps
endpoint-to-endpoint and increases monotonically.

**Fix / rule.** `src/routing/utils.rs::log_space_lower` applies the epsilon only when
`lo <= 0`; a genuinely positive `lo` uses `lo.ln()` directly. It is shared with
`src/training/forward.rs::physical_to_normalized` so the forward and inverse maps
cannot drift apart. **Whenever you add a log-space parameter, assert its box maps
endpoint-to-endpoint before trusting any result that depends on it**. A denormalized
value that escapes its own declared bounds is the cheapest tell.

**DDR parity.** DDR's Python
`~/projects/ddr/src/ddr/routing/utils.py::denormalize` still carries the unconditional
`bounds[0] + 1e-6` and needs the same patch. Until it lands, ddrs and DDR disagree on
log-space boxes with a small positive lower bound, and agree everywhere else to about
one part in a million. That one part in a million is visible in
`compare_ddr_sandbox`: the max abs diff moved from 1.53e-5 to 1.53e-4 m³/s when the fix
landed, still `ABSOLUTE MATCH` with 6.5x margin under the 1e-3 bar, because the fixture
was generated by a DDR that still shifts `p_spatial` by `1e-6`.

## T18: Pre-#192 training-step fixtures routed on the corrected physics (known failure, 2026-09-25)

**Symptom.** Where the real CONUS adjacency exists
(`/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr` and
`merit_gages_conus_adjacency.zarr`),
`cargo test --features fixtures --test training_step_layer_b --test training_step_layer_c`
fails two sub-tests, with or without any recent branch:

| sub-test | measured | bar |
|---|---|---|
| `layer_b_step3_mc_forward_matches_ddr` | MC forward max abs 1.01e-1, mean 5.18e-3 m³/s vs `tests/fixtures/training_step/mc_forward_77006074.npz` | 1e-5 |
| `layer_c_step2_gradients_match_ddr` | `grad_output_weight` max abs 2.03e-4 (loss diff 4.02e-4) | 1e-4 |

Every other sub-test in both files passes, and `training_step_layer_d` passes. CI never
sees the failure: the stores are absent on runners, so `open_stores` returns `None` and
each sub-test prints `SKIP` and returns as a pass.

**Cause (verified 2026-09-25).** A physics mismatch, not a routing bug.
`tests/training_step_layer_b.rs::fixture_config` (and its twin in
`tests/training_step_layer_c.rs::fixture_config`) loads `config/merit_training.yaml`,
which sets no `ddr_match`, so the engine runs the default `false`: the corrected physics
(β celerity, Cunge X) since 2026-08-19. The fixtures are legacy: their `manifest.json`
records DDR commit `c9a0242` (#187), an ancestor of `3125dac` (#192, the physics port).
The `ddr_match: true` comment in `layer_b_step1_subgraph_adjacency_matches_ddr` refers only
to the `outflow_idx` argument that sub-test passes to `collate::compress`; step 3 builds
its own upstream-neighbour outflow columns and routes with `fixture_config`.

**Discriminating test.** Add `ddr_match: true` under `params:` in
`config/merit_training.yaml`, rerun both files, then revert the YAML. On 2026-09-25 this
made all 7 sub-tests pass: step 3 max abs 3.34e-6, layer C `grad_output_weight` 1.76e-5
(worst gradient 7.16e-5 on `grad_output_bias`, the value the test's own tolerance comment
records).

**Fix options (not applied).** (a) Set `cfg.params.ddr_match = true` in the two
`fixture_config` helpers. That restores the parity the tests were written for, on the
deprecated path, and dies when `ddr_match` is removed. (b) Regenerate the fixtures from
DDR at or past #192 with `scripts/dump_ddr_training_step.py` under `~/projects/ddr/.venv`.
That tests the physics ddrs runs, but step 3's hand-built upstream-neighbour outflow and
its `ln(1 + 1e-6)` log-space lower bound (pre-T17) would have to follow. Until one lands,
these two failures are known and carry no regression signal; a failure in any OTHER
training-step sub-test is still real.

## T19: A gauge's prediction depends on the other gauges in the eval set (2026-09-26)

**Symptom.** The same checkpoint scores a gauge differently when the gauge CSV
(`data_sources.gages`) changes, on the same backend and window. Found evaluating the
121-gauge dam benchmark against the 2,365-gauge eval of the same run: 73 of 121 gauges
differed, NSE by up to 1.58, per-gauge median relative flow difference up to 29 %.

**Cause.** `src/data/collate.rs::build_flow_scale` (mirrors DDR `readers.py:270-330`) scales
the lateral inflow `q'` of every batch gauge's outlet reach by that gauge's `FLOW_SCALE`.
The scaled inflow is routed, so it reaches every gauge downstream. Adding or removing a
gauge with `FLOW_SCALE` ≠ 1 changes the flow at all gauges below it. Not a CPU/CUDA effect.

**Discriminating test.** For each differing gauge, list the eval gauges whose outlet COMID
is in its subgraph (`merit_gages_conus_adjacency.zarr/<staid>/order`) and are present in
one gauge set but not the other. On 2026-09-26 the gauges with no such gauge of
`FLOW_SCALE` ≠ 1 matched the full eval to 1e-4 m³/s at 47 of 47; 73 of the 74 with one
differed.

**Consequence.** Paired comparisons are valid only within one gauge CSV. Never compare a
subset eval's absolute metrics with a full-population table. Fix options (not applied):
apply `FLOW_SCALE` to the gauge's output rather than the reach inflow (breaks DDR parity),
or always evaluate on the full population and subset afterwards.

---

## T20: Stage op CPU and CUDA must fuse the same multiply-adds (2026-11-01)

**Discriminating test.** Run the `scripts/fma_sass_check.py` check (procedure in
`testing.md` "Stage on CUDA"). For each stage kernel, `FFMA` at `--fmad=true` minus
`FFMA` at `--fmad=false` must be 0, or only named bit-neutral doublings
(`(a*b)*2.0`, which the compiler writes as `x + x`).

**Why.** NVRTC contracts `a*b + c` into one rounding by default, and the cubecl fork
passes no `--fmad` option. The CPU rounds twice. On a line that cancels (`q_new` on a
dam cell, the depth-floor drain) a 1 ulp difference grows into a branch flip. The user
ruled on 2026-11-01 that the CPU does the same fusing as the GPU. So every site is now
written as an explicit fused multiply-add on all sides, line for line:

- the kernels in `src/routing/stage_cuda/` and `src/sparse/tree_lu_cuda.rs`, with the
  cubecl `fma`;
- the host fns of `src/routing/stage_op.rs` and `src/sparse/tree_lu.rs`, with
  `f32::mul_add`;
- the reference `StageBlock::step_reference`, through the ops of
  `src/routing/stage_fma.rs` (used only by the reference).

**What is bitwise.** CPU equals GPU bitwise on every forward and backward block field
whose chain has no `powf` or `ln`. The tree solves are bitwise because the stage host
solves copy the form cuSPARSE uses, `x = (b - sum) * (1 / diag)`
(`src/sparse/tree_lu.rs::tree_lu_solve_host` and `tree_lu_solve_transposed_host`). The
shared Muskingum-Cunge solves in `src/sparse/mod.rs` keep `/ diag`, so the DDR sandbox
match does not move. CUDA's `powf` uses a hardware reciprocal approximation that the CPU
cannot copy, so fields with `powf` or `ln` in the chain differ by a few ulp. That
residue is why the trajectory gates are two-level (see `testing.md`).

**Compiler facts.** A product by `2.0` is not a fusing candidate (NVRTC writes `x*2`
as `x + x`, and fusing it is bit-neutral). `hb0 = b0 + (b1 - b0) * f0` stays unfused. A
product with several uses stays unfused only when none of its uses feeds an add or a
subtract. When one use does, NVRTC fuses it (this happened in the section, Newton and
depth backward kernels). So the rule is: the SASS gap decides, per kernel.

**After a CUDA or cubecl update, rerun the check.** The compiler's choices can change.
A site that was unfused can become fused, and the GPU bits then move away from the CPU.

---

## T21: The stage reference has its own fused cross-section (2026-11-01)

`StageBlock::section` in `src/routing/stage.rs` does not call the shared trapezoid fns
of `src/geometry.rs` for its arithmetic. It writes the section out with its own fused
multiply-adds (`fma3` from `src/routing/stage_fma.rs` at the fused sites), line for line
with `src/routing/stage_op.rs::section_host`, so that it stays bitwise equal to the fused
op. `src/geometry.rs` (shared with the hotstart) keeps its unfused form.

A change to the stage cross-section, for example a power-law section selected by
`params.geometry_form`, must change all of these together: `section_host` and
`section_backward_host` in `src/routing/stage_op.rs`, `section_kernel` and
`section_backward_kernel` in `src/routing/stage_cuda/`, and the reference copy in
`StageBlock::section`. It must also change `StageBlock::hotstart`, which takes the start
depth from `compute_trapezoidal_geometry`. A hotstart left on the trapezoid gives an
inconsistent start state, and the theta 0.6 blow-up (T13) shows the start state matters.
Changing `src/geometry.rs` alone moves the hotstart and leaves the stage step on the old
section. `tests/stage_op_parity.rs` (fused equals reference,
bitwise) and `stage_cuda_blocks` fail if one side is missed.

---

## T22: The fused CPU stage step is about 1.3x slower without the `fma` target feature (2026-11-02)

`f32::mul_add` is a libm `fmaf` call unless the build has the `fma` CPU feature.
The default x86-64 build has none. Measured with the stage training examples
(`run.log` has 1 s resolution, so 3 to 5 % noise on a 3 s figure):

| build | Juniata stage, rho 90 (s/mb) | Juniata stage, 4 substeps (s/mb) | Hudson (s/mb) |
|---|---|---|---|
| before the fused multiply-add change | 0.538 | 1.000 | 3.000 |
| default | 0.615 to 0.641 | 1.263 to 1.316 | 4.333 to 4.444 |
| `-C target-cpu=native` | 0.538 | 0.947 | not measured |
| `-C target-cpu=x86-64-v3` | 0.538 | 0.947 | 2.889 |

A native or x86-64-v3 build removes the slowdown on this host and keeps the golden
bits (`stage_cpu_golden` passes on both). The Hudson figures come from a separate
check on an idle machine on 2026-11-02, interleaved: default 4.444 and 4.333,
x86-64-v3 2.889 and 2.889. A pure-Muskingum-Cunge run does not change (0.263
against 0.273). **The build choice is open.** The user has not decided whether to set
a target feature in `.cargo/config.toml` or accept the default.

