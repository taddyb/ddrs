---
name: ddrs-run-and-operate
description: "Use when you need to install, configure, launch, or monitor a ddrs training/eval run; diagnose a stale binary; resume from a checkpoint; select a CUDA device; manage data-source groups; or understand workspace artifact layout. Also use when troubleshooting silent correctness failures caused by stale PATH binaries, CUDA graphs masking NaN, or a GPU eval Phase 2 that spins at high CPU without exiting (cubecl worker-thread OOM panic — see §14)."
---

# ddrs: Run and Operate

## Glossary (read once, referenced throughout)

| Term | Meaning |
|---|---|
| **ddrs** | BURN-0.21 Rust binary that trains and evaluates a Muskingum-Cunge routing model. Analogous to a PyTorch training script, but compiled. |
| **DDR** | The Python/PyTorch reference implementation at `~/projects/ddr/`. ddrs must be gradient-exact against it. |
| **KAN head** | Kolmogorov–Arnold Network (rskan v0.1.3). Maps catchment attributes → routed hydrograph parameters. Replaces an MLP. |
| **Workflow** | `train`, `eval`, or `train-and-test`. Declared in `ddrs.yaml`; can be overridden per invocation with `--workflow`. |
| **Run directory** | `.ddrs/runs/<id>/` — all outputs of one `ddrs run` invocation live here. |
| **Checkpoint** | A DIRECTORY `epoch_E_mb_M/` holding `head.mpk` (weights), `optim.mpk` (Adam moments), `state.json` (rng + sampler state). NOT a flat file. |
| **Q' (Qprime)** | Input streamflow forcing from an upstream rainfall-runoff model. Summed upstream Q' without routing is the performance baseline. |
| **Leakance** | Experimental groundwater–surface-water loss term (`zeta`). Off by default. |
| **CUDA graphs** | NVIDIA kernel graph capture that improves throughput but CANNOT capture the leakance kernel. Enabled by `use_cuda_graphs: true`; rejected at config load when `use_leakance: true`. |

---

## When NOT to use this skill

- Changing routing math, geometry, or the sparse backward → use **ddrs-architecture-contract**
- Debugging gradients, autograd tape, or KAN head parity → use **ddrs-validation-and-qa**
- Interpreting experiment results or writing findings → use **ddrs-research-frontier** or **ddrs-identifiability-campaign**
- Building or editing config YAML keys beyond what is covered here → use **ddrs-config-and-flags**

---

## 1. Install and Path Setup

```bash
# One-time install — puts `ddrs` in ~/.cargo/bin/
cargo install --path .

# Ensure ~/.cargo/bin is on PATH (add to ~/.bashrc if missing)
export PATH="$HOME/.cargo/bin:$PATH"
```

### STALE-BINARY TRAP (most common silent failure)

`cargo build` and `cargo run` compile to `target/release/ddrs`. They do NOT update `~/.cargo/bin/ddrs`. If you edit `src/` and then type `ddrs run`, you silently execute the OLD binary.

The manifest's `git.sha` is stamped from `.git` at runtime (not from the binary itself), so a run can appear to show the current commit SHA while actually executing weeks-old code. This caused the 2026-07-01 leakance×hourly 2×2 to silently run a pre-disaggregation binary, producing byte-identical cells that looked like "disagg no-op."

**After every `src/` change, do ONE of:**

```bash
# Canonical (always correct):
cargo install --path .

# Faster if target/release is already current:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs

# Bypass the installed copy entirely for one-off runs:
cargo run --release --bin ddrs -- run --workflow train-and-test
```

**Quick self-check:** Current checkpoints are DIRECTORIES (`epoch_E_mb_M/head.mpk`). If you see flat `.mpk` files (e.g. `epoch_5_mb_9.mpk`), you ran a stale pre-checkpoint-resume binary.

---

## 2. First-Time Workspace Setup

```bash
# Step 1: probe GPU, validate config, build adjacency/baseline caches
ddrs plan

# Step 2: run the selected workflow
ddrs run
```

`ddrs plan` on first invocation:
1. Probes the GPU and runs the 5-reach RAPID smoke test (cached to `.ddrs/system.json`; subsequent plans are instant).
2. Opens `$EDITOR` on `ddrs.yaml` if the file does not exist.
3. Builds CONUS + per-gauge adjacency zarr stores into `.ddrs/adjacency/<key>/` from the raw geospatial fabric (~10 s for CONUS `.dbf`; ~25 s for global `.gpkg` with 2.94M reaches). Content-addressed; cache hits are instant.
4. Computes the summed-Q' baseline into `.ddrs/baselines/<key>/` (~370 MB read on first run; cached).
5. Reports drift vs `.ddrs/sources.lock` and relocks.

`ddrs plan` is NOT side-effect-free on first run.

### Forcing a clean template

```bash
ddrs --config config/merit_training.yaml plan --workflow train-and-test
ddrs --config config/merit_training.yaml run  --workflow train-and-test
```

### mode/workflow contract

`mode:` and `workflow:` must agree. `ddrs plan` rejects contradictions at load time.

| `mode:` | Allowed `workflow:` |
|---|---|
| `training` | `train`, `train-and-test` |
| `testing` | `eval` |

---

## 3. Core Commands

```bash
ddrs plan                          # validate, cache, print plan
ddrs run                           # execute workflow from ddrs.yaml
ddrs run --workflow train          # override workflow for one invocation
ddrs run --workflow eval
ddrs run --workflow train-and-test
ddrs run --strict                  # abort with exit 4 if sources.lock drifted (does NOT relock)
ddrs show <run-id>                 # inspect a past run's manifest.json
ddrs status                        # workspace summary + disk usage
ddrs gc --keep 5 --keep-successful # prune .ddrs/runs/
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Config invalid |
| 3 | Data source missing |
| 4 | Lock drift (`--strict`) |
| 5 | Runtime failure |
| 6 | Workspace not initialized |

---

## 4. Workspace Artifact Layout

```
<project-root>/
├── ddrs.yaml                             # Workflow + experiment config (gitignored)
└── .ddrs/
    ├── system.json                       # GPU/driver/smoke-test record
    ├── sources.lock                      # Fingerprints of data_sources paths
    ├── adjacency/<key>/                  # Managed CONUS + gauges zarr stores
    ├── baselines/<key>/                  # Summed-Q' baseline cache
    └── runs/
        └── <id>/
            ├── manifest.json             # Config + sources + git SHA + outputs
            ├── config.yaml               # Snapshot of config that produced this run
            ├── run.log                   # Timestamped tee of stdout+stderr (incl. CUDA)
            ├── checkpoints/
            │   └── epoch_E_mb_M/         # Checkpoint DIRECTORY (NOT a flat file)
            │       ├── head.mpk          # KAN weights (HalfPrecisionSettings / f16)
            │       ├── optim.mpk         # Adam moments
            │       └── state.json        # Epoch, mini-batch, rng, sampler state
            ├── eval/
            │   └── predictions.zarr      # Eval predictions for plotting
            ├── baseline/                 # Copy of .ddrs/baselines/<key>/
            ├── plot/
            │   └── kan_parameters.nc     # Per-COMID KAN outputs (--plot only)
            └── kan_parameters.nc         # Leakance zeta diagnostic (leakance runs only)
```

**Run ID format:** `<UTC timestamp>-[<group>-]<workflow>`  
Example: `2026-06-12T14-02-10Z-conus-train-and-test`  
The `<group>` segment is present when `data_sources` matches a saved source group.

---

## 5. Device Selection

```yaml
# ddrs.yaml
device: 0    # CUDA device ordinal (default 0)
```

On multi-GPU hosts, set `device: 1` (or higher) to route training off the display GPU. Mirrors DDR's `device:` key. Training (`train`, `train-and-test`) requires a CUDA GPU; `ddrs run` returns exit 5 if none is found.

---

## 6. Data-Source Groups

Named snapshots of the `data_sources:` block, stored in `config/sources/<name>.yaml` (tracked in git).

Shipped groups (as of 2026-07-05):

| Group | Notes |
|---|---|
| `conus` | MERIT CONUS fabric + USGS daily observations |
| `conus-hourly` | `conus` + AORC precip store at `/mnt/ssd1/data/aorc/merit_unit_catchments.zarr` |
| `global` | Global MERIT gpkg + global zarr-v2 obs/streamflow |
| `daily-lstm` | NH CudaLSTM daily Q' forwards |
| `hourly-lstm` | NH MTS-LSTM hourly Q' forwards |

```bash
ddrs sources list                # '*' marks the group currently matching ddrs.yaml
ddrs sources save <name>         # snapshot current data_sources block
ddrs sources use  <name>         # splice group into ddrs.yaml + refresh sources.lock
```

`save`/`use` are textual operations (comments inside the block are preserved). `use` validates the spliced config parses before committing.

**Switching from CONUS to global:**
```bash
ddrs sources use global && ddrs plan --workflow train && ddrs run --workflow train
```

### Hourly forcing requirements

Hourly forcing requires BOTH conditions or it fails at `MeritGagesDataset::open`:

1. `aorc_precip` source present (use `conus-hourly` group, or inline in config)
2. `kan_head.disaggregation.use_precip: true` in the experiment config

A missing `aorc_precip` source with `use_precip: true` is a hard error — not a silent fallback to flat-daily. The hourly Q' stores start 1981-01-01; experiment windows must not reach into 1980. Check the dataset-open log line `streamflow resolution: Daily|Hourly` to confirm.

### Importing a Q' store

```bash
ddrs import <store> --dry-run          # validate + coverage report only
ddrs import <store> --name <group>     # validate + register config/sources/<group>.yaml
```

Any store meeting the DDR Q' contract (`Qr(divide_id, time)` f32 m³/s, CF `days since`/`hours since` axis) can be registered. The reader auto-detects daily vs hourly-native from CF time units.

---

## 7. Resuming from a Checkpoint

A checkpoint is a DIRECTORY. Set `experiment.checkpoint:` in `ddrs.yaml` to the directory path:

```yaml
experiment:
  checkpoint: .ddrs/runs/2026-06-12T14-02-10Z-conus-train-and-test/checkpoints/epoch_25_mb_8
  epochs: 50    # MUST be > checkpoint epoch, or zero batches will train
```

`bootstrap_head_and_state` (`src/training/bootstrap.rs`) restores weights, Adam moments, rng state, and the sampler permutation + cursor. The resumed run draws the same gauge batches the original would have, and the learning-rate schedule continues at the true epoch.

**Caveats:**
- Checkpoint weights and moments are stored at f16 (`CompactRecorder` = `HalfPrecisionSettings`). A resumed trajectory drifts slowly from an uninterrupted run.
- Checkpoint compatibility with DDR `.pt` files is NOT supported.

---

## 8. CUDA Graphs — NaN Masking Bug

When `use_cuda_graphs: true` (default in `config/merit_training.yaml`), CUDA graph capture replays stale finite-loss values if the forward pass encounters NaN. The loss looks valid but gradients are wrong or zero.

**Checklist before trusting a run with CUDA graphs:**
- [ ] Verify no NaN in Q' forcing for your time window
- [ ] If a run shows suspiciously flat or zero loss, disable graphs and re-run:

```yaml
params:
  use_cuda_graphs: false
```

**Hard rule:** `use_leakance: true` + `use_cuda_graphs: true` is rejected at config load (exit 2). Never combine them.

---

## 9. Leakance — Enabling and Operating

Leakance is OFF by default. Three config changes are required together:

```yaml
params:
  use_leakance: true
  use_cuda_graphs: false    # mandatory — config load rejects true+true

kan_head:
  learnable_parameters: [n, k, x, K_D, d_gw, leakance_factor]
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-6]      # log-space; hydraulic exchange rate (1/s)
    d_gw: [-2.0, 2.0]           # groundwater depth offset (m)
    leakance_factor: [0.0, 1.0] # dimensionless scale
```

See `config/experiments/leakance_hourly_on.yaml` for a complete working config.

### Zeta diagnostic (eval-only)

`zeta` (mean |zeta|, m³/s) and `zeta_net` (signed mean; positive = losing reach) are written to `.ddrs/runs/<id>/kan_parameters.nc` during eval. `train-and-test` produces it automatically in Phase 2. For an existing checkpoint without retraining:

```bash
cargo build --release --bin eval
target/release/eval \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
  --output /tmp/eval.zarr \
  --zeta-output .ddrs/runs/<id>/kan_parameters.nc
```

### Leakance status (as of 2026-07-01)

Verdict: **GO, marginal** — all three gate criteria met on the 2×2 (leakance × forcing) experiment:
- Leakance helps on the losing-stream subset under hourly forcing (DNSE +0.0005, DKGE +0.0018, 55.5% of gauges improve)
- Leakance hurts under daily forcing (DNSE -0.0017, DKGE -0.0009, 35.6% improve)
- |zeta| > 0.01 m³/s on 10.4% of 64,892 eval reaches (median 6.4e-4; 53.7% net-losing)
- K_D pinned at the 1e-6 ceiling (binding constraint; range widening NOT recommended as of 2026-07-05 — see identifiability campaign findings)

---

## 10. Performance Baseline Reference

The summed-Q' baseline (no routing, no learned parameters) is computed by `ddrs plan` and copied to each run directory. If trained KAN median NSE does not beat this, check training loss curves and KAN head gradient stats first.

Baseline (CONUS, as of 2026-07-05): median NSE 0.689 / KGE 0.723  
Best trained result (precip-driven disagg + L1 loss, 2365 gauges, 2026-06-23): NSE 0.715 / KGE 0.711

Note: KGE does NOT beat the summed-Q' baseline in any config as of 2026-07-05. NSE beats it (+0.037 with precip disagg). This is a known open problem.

These numbers are for the default `merit_dhbv2_UH_retrospective.ic` streamflow store. Swapping in an alternative Q' store (dHBV AORC2F variants, NH LSTM daily/hourly) changes both the baseline and the trained result substantially — see `docs/2026-07-16-aorc2f-wave1-findings.md` / `docs/2026-07-16-wave2-cross-wave-findings.md` for a 4-store comparison (all four came in below this benchmark).

---

## 11. Legacy Binaries (deprecated, removed in 0.4)

```bash
# These still work but print a deprecation warning:
cargo run --release --bin train -- ...
cargo run --release --bin eval -- ...
cargo run --release --bin train_and_test -- ...
```

Use `ddrs run --workflow <name>` instead.

---

## 12. Regression Gate — Must Never Break

After any change to `src/routing/`, `src/geometry.rs`, or `src/sparse.rs`:

```bash
cargo run --release --example compare_ddr_sandbox
# Must print: ABSOLUTE MATCH (max abs diff < 1e-3 m³/s on 5-reach RAPID sandbox)
```

Caveat (2026-06-06): the reference DDR state lives only on the local desktop's `~/projects/ddr` working tree (unpushed `geometry/trapezoidal.py`). A fixture regenerated from a clean DDR clone diverges ~1% — that is a wrong-reference problem, not a port bug. See `.claude/references/ddrs-comparing-to-ddr.md` for regeneration instructions.

---

## 13. Common Failure Checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| Flat or byte-identical hourly/daily cells | Stale binary (pre-disagg) | `cargo install --path .` |
| `exit 4` from `ddrs run --strict` | `sources.lock` drifted | Run `ddrs plan` to relock, or investigate which source changed |
| Zero or NaN loss with `use_cuda_graphs: true` | CUDA graphs masking NaN forward | Set `use_cuda_graphs: false`, investigate NaN in forcing |
| `exit 2` on leakance config | `use_leakance: true` + `use_cuda_graphs: true` | Set `use_cuda_graphs: false` |
| `exit 2` mode/workflow mismatch | `mode: testing` with `workflow: train` | Align both keys in `ddrs.yaml` |
| Resumed run trains 0 batches | `experiment.epochs` <= checkpoint epoch | Raise `epochs` above the checkpoint's epoch number |
| Flat files `epoch_E_mb_M.mpk` in checkpoints/ | Stale pre-checkpoint-resume binary | `cargo install --path .` |
| Hourly run uses flat repeat-24 | Missing `aorc_precip` source OR `use_precip: false` | Add `conus-hourly` source group AND set `use_precip: true` |
| Dataset open error with `use_precip: true` | `aorc_precip` not in `data_sources` | `ddrs sources use conus-hourly` |
| `train-and-test`'s Phase 2 (testing) spins at 900%+ CPU, repeatedly logging `thread 'DSD-0-0' panicked ... can't allocate buffer` without ever exiting | cubecl-cuda background-worker OOM panic that doesn't propagate to the caller (Phase 2's per-chunk buffer, ~4.2 GB at `batch_size: 15` days, is far larger than a training minibatch's) | Kill the process; recover via the standalone `eval` binary against the last checkpoint with `--backend cpu` (see §14). Since 2026-07-16 a fresh binary hard-fails instead of spinning — reinstall if you see this. |

---

## 14. GPU Eval OOM — Silent Corruption Bug (fixed 2026-07-16)

**Symptom:** `--backend cuda` `train-and-test`/`eval` completes training fine,
then Phase 2 (testing) hits a CUDA OOM inside a cubecl-cuda background
worker thread (`thread 'DSD-0-0' panicked ... can't allocate buffer of size:
4178264064`). The panic does **not** propagate to the caller — cubecl's
server loop catches it, logs, and drops the task — so the main thread just
keeps looping through all 366 eval chunks, writing whatever half-written GPU
buffer is left. **This happens even with no other process on the GPU** —
Phase 1 alone can leave too little headroom for Phase 2's much larger
per-chunk buffer at the default `testing.batch_size: 15` (days/chunk) on a
16 GB card. Confirmed twice independently (2026-07-16, distributed-source
and daily-lstm-source arms), including once running solo.

**Fixed as of 2026-07-16** (`src/training/eval.rs`): `evaluate()` now
detects this via two layered checks and returns
`DataError::CorruptedEvalChunk` instead of silently continuing —
1. `corrupted_chunk_reason` — all-zero or non-finite chunk values (kept as
   defense-in-depth; alone it MISSED the second incident because the
   corrupted chunk happened to contain plausible-looking stale finite
   values).
2. `WORKER_PANICKED` — a process-global flag set by a custom panic hook
   (`ensure_panic_hook_installed`), checked after every chunk and once more
   after the post-loop tensor readback. This is the reliable detector — it
   fires on ANY background-thread panic regardless of what ends up in the
   output buffer.

A binary built before 2026-07-16 will still exhibit the silent-spin
behavior — `cargo install --path .` to pick up the fix (see the
stale-binary trap in §1). 6 unit tests cover both detectors:
`cargo test --lib training::eval::tests`.

**Practical recommendation:** on hardware where this triggers, either (a)
default to `--backend cpu` for `train-and-test`/`eval` workflows (training
alone is fine on GPU), or (b) try lowering `testing.batch_size` (days/chunk)
to shrink the per-chunk buffer — not yet validated as a GPU fix. **Recovery
without retraining:** the standalone `eval` binary reads any completed
checkpoint directly and supports `--backend cpu` independent of what backend
Phase 1 used:

```bash
cargo install --path .   # only if your eval binary predates the fix
~/.cargo/bin/eval --config <run_dir>/config.yaml \
  --checkpoint <run_dir>/checkpoints/epoch_E_mb_M/head \
  --output <run_dir>/eval/predictions.zarr \
  --backend cpu
```

**Running two arms in parallel on one GPU:** don't co-schedule two
`--backend cuda` trainings — Phase 1 alone uses ~9.9 GB on a 16 GB card, so
a second concurrent process reliably OOMs within seconds. Give each arm a
different backend from the start (one `--backend cuda`, one `--backend
cpu`) rather than launching both on GPU and recovering after a crash. See
`config/experiments/aorc2f_distributed_frozen_chunk1.yaml` /
`aorc2f_lumped_frozen_chunk1.yaml` / `lstm_daily_frozen_chunk1.yaml` /
`lstm_hourly_native.yaml` and
`docs/2026-07-16-aorc2f-wave1-findings.md` /
`docs/2026-07-16-wave2-cross-wave-findings.md` for a full worked campaign
(4 CONUS train-and-test arms, 2 OOM incidents, full recovery).

---

## Provenance and Maintenance

Verified from source on 2026-07-05. Key files:

```bash
# Re-verify CLI command surface:
grep -n "pub fn" /home/tbindas/projects/ddrs/src/cli/run.rs
grep -n "pub fn" /home/tbindas/projects/ddrs/src/cli/plan.rs

# Re-verify exit codes:
cat /home/tbindas/projects/ddrs/src/cli/types.rs

# Re-verify checkpoint directory layout:
head -35 /home/tbindas/projects/ddrs/src/training/checkpoint.rs

# Re-verify CUDA graphs + leakance rejection:
grep -n "use_leakance.*use_cuda_graphs\|use_cuda_graphs.*use_leakance" /home/tbindas/projects/ddrs/src/config.rs

# Re-verify shipped source groups:
ls /home/tbindas/projects/ddrs/config/sources/

# Re-verify stale-binary trap documentation:
grep -A 20 "STALE-BINARY TRAP" /home/tbindas/projects/ddrs/CLAUDE.md

# Re-verify the GPU eval OOM fix (§14):
grep -n "CorruptedEvalChunk\|WORKER_PANICKED" /home/tbindas/projects/ddrs/src/training/eval.rs /home/tbindas/projects/ddrs/src/data/error.rs
```

Volatile facts dated 2026-07-05: baseline metrics, best-run metrics, leakance GO verdict, K_D ceiling observation, KGE vs baseline status. Re-check these after any new experiment campaign.

Volatile facts dated 2026-07-16: the GPU eval OOM bug (§14) and its fix; the
4-arm AORC2F/LSTM campaign benchmark numbers in
`docs/2026-07-16-aorc2f-wave1-findings.md` and
`docs/2026-07-16-wave2-cross-wave-findings.md`. Re-check the OOM fix is
still present (grep above) before trusting a GPU `train-and-test` run's
Phase 2 output on new hardware.
