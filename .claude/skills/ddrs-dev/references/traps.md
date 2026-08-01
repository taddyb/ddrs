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

**Wrong-reference failure mode:** a fixture regenerated from a clean DDR clone
diverges ~1% (max abs ≈ 0.55 m³/s) at *every* ddrs commit. Only the desktop's
`~/projects/ddr` working tree is valid — see `build-and-env.md`.

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
not a bug.

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
