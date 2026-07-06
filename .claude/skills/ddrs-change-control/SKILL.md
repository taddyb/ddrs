---
name: ddrs-change-control
description: "Use when reviewing, gating, or merging a change to ddrs source code, config files, or Cargo dependencies; when assessing whether a modification to src/routing/, src/sparse.rs, src/geometry.rs, src/nn/kan_head.rs, or Cargo.toml is safe; when a run produced unexpected results and binary staleness or an invariant violation may be the cause; or when designing an experiment that touches leakance, CUDA graphs, or the routing core."
---

# ddrs change-control runbook

## Overview

`ddrs` is a BURN-0.21 Rust port of the DDR differentiable Muskingum-Cunge routing model (Python/PyTorch reference at `~/projects/ddr/`). The port must remain **gradient-exact** against DDR. Breaking any of the seven invariants below makes the port meaningless; every PR that touches the affected files must clear its gate before merge.

**Glossary for PyTorch engineers:**
- `BURN` — Rust deep-learning framework, analogous to PyTorch. BURN 0.21 is pinned.
- `Backward<I, N>` — BURN's trait for a custom autograd function (analogous to `torch.autograd.Function`). `I` = backend (CPU/CUDA), `N` = number of saved tensors.
- `CsrPattern` — the sparsity structure of the river network adjacency matrix, stored as a Rust struct (row/col index arrays). Analogous to `torch.sparse_csr_tensor`.
- `KanLayer` — Kolmogorov-Arnold Network layer from the `rskan` crate (Rust equivalent of DDR's `kan.py`).
- `CompactRecorder` / `HalfPrecisionSettings` — BURN serializer for checkpoints. Saves weights as f16.
- `COMID` — unique 64-bit integer ID for each river reach in the MERIT-Hydro fabric.
- `Q'` (Qr) — lateral inflow forcing (divide-level runoff, m³/s) from the pre-trained DHBV2 model. Not observed discharge.
- `zeta` — leakance flux (m³/s), the GW–SW water-loss term. Subtracted from the routing RHS at every timestep.
- `rho-window` — a training mini-batch: a contiguous time slice of length `rho` (default 90 days) sampled from the training period.

---

## When NOT to use this skill

Do not use this skill for:
- **Plotting or analysis scripts only** (no `src/` change) — use `ddrs-eval-plots` instead.
- **Config tuning within documented safe ranges** (changing `experiment.epochs`, `learning_rate`, `batch_size`, loss weights) — no gate applies; these do not affect the routing core or port invariants.
- **Data source path changes only** — consult `CLAUDE.md §Data sources` directly.
- **CLI / workspace questions** — consult `CLAUDE.md §ddrs CLI`.

---

## Change classification matrix

Every change falls into one of four tiers. Look up the modified file(s) in the left column; the tier determines which gate checklist you must run.

| Modified file(s) | Tier | Rationale |
|---|---|---|
| `src/routing/mmc.rs`, `src/routing/mmc_op.rs`, `src/routing/utils.rs` | **A — routing core** | Directly implements the Muskingum-Cunge timestep; must remain gradient-exact vs DDR |
| `src/routing/leakance.rs` | **A — routing core + leakance** | Custom `Backward<I,8>` for the GW–SW term; both the forward kernel and analytical gradients must stay exact |
| `src/geometry.rs` | **A — routing core** | Trapezoidal geometry; changes cascade into every geometry-dependent variable |
| `src/sparse.rs` | **A — routing core** | Hand-written CSR triangular solve + custom `CsrSolveOp: Backward`; O(nnz) autograd tape invariant |
| `src/nn/kan_head.rs` | **B — KAN head** | Must match DDR `kan.py` exactly; rskan version pin governs this |
| `Cargo.toml` (rskan tag) | **B — KAN head** | rskan pin is the single authoritative version for KAN parity |
| `src/config.rs` | **C — config/ranges** | Parameter ranges and log-space flags affect denormalization; wrong range silently mis-scales gradients |
| `src/training/loss.rs` | **C — objective** | Autograd is unchanged (invariant 4 intact) but loss changes affect all metrics comparisons |
| `src/training/forward.rs` | **C — training path** | Disaggregation, leakance threading; changes can silently no-op features (see STALE-BINARY TRAP) |
| `config/experiments/*.yaml`, `config/sources/*.yaml` | **D — config only** | No Rust changes; validate with `ddrs plan` before running |
| Any other `src/` file | **C — default** | Run full test suite and DDR regression |

---

## Tier A gate — routing core

Run ALL of the following. A single failure is a merge blocker.

```bash
# 1. DDR parity — THE non-negotiable regression
cargo run --release --example compare_ddr_sandbox
# Must print: "ABSOLUTE MATCH" with max abs diff < 1e-3 m³/s

# 2. Core unit + integration tests
cargo test --lib
cargo test --test mmc
cargo test --test sparse_gradcheck

# 3. Leakance gates (required even if you did not touch leakance.rs,
#    because any routing change can disturb the leakance OFF parity)
cargo test --test leakance_gradcheck       # 8/8 analytical ≈ finite-difference
cargo test --test leakance_off_parity      # 3/3 byte-identical to no-leakance when off
cargo test --test zeta_accum              # 6/6 accumulated zeta == headwater q difference
```

If you touched `src/routing/leakance.rs` specifically:
```bash
# Confirm gradient-exactness for all 8 leakance backward inputs
cargo test --test leakance_gradcheck -- --nocapture
```

**After passing all Tier A gates, run the binary self-check:**
```bash
# Stale-binary check: directory checkpoints = current binary
ls .ddrs/runs/<last-run-id>/checkpoints/
# Must show directories like epoch_5_mb_9/, NOT flat files like epoch_5_mb_9.mpk
# Flat files = stale binary ran. Refresh: cargo install --path .
```

---

## Tier B gate — KAN head

```bash
# Full KAN parity sweep (required on every PR touching src/nn/ or Cargo.toml rskan pin)
cargo test --features fixtures \
  --test kan_head_init_repro \
  --test kan_head_init_parity \
  --test kan_head_fixture_forward \
  --test kan_head_fixture_backward

# Then run Tier A DDR parity to confirm the head change did not break routing
cargo run --release --example compare_ddr_sandbox
```

If a DDR-side change to `kan.py` broke the fixture:
```bash
# Regenerate under DDR's venv, then re-validate
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_kan_head.py
# Re-run the fixture tests above
```

---

## Tier C gate — config, training path, other src/

```bash
cargo test --lib
cargo test                              # full suite
cargo run --release --example compare_ddr_sandbox   # DDR parity
```

For `src/training/forward.rs` changes that affect disaggregation or leakance threading, also run:
```bash
cargo test --test leakance_off_parity  # leakance OFF must stay byte-identical
```

---

## Tier D gate — config files only

```bash
# Validate the config parses and data sources resolve
ddrs plan --config config/experiments/<your_config>.yaml \
          --workspace /home/tbindas/projects/ddrs/.ddrs
# Must exit 0 with no "drift" warnings
```

---

## The 7 non-negotiables with rationale and incidents

### Invariant 1 — DDR sandbox ABSOLUTE MATCH

**Rule.** `cargo run --release --example compare_ddr_sandbox` must print "ABSOLUTE MATCH" (max abs diff < 1e-3 m³/s on the 5-reach RAPID sandbox). Re-run after every change to `src/routing/`, `src/geometry.rs`, or `src/sparse.rs`.

**Rationale.** The port exists to be gradient-exact against DDR. Any drift makes subsequent metric comparisons meaningless — you cannot tell whether a difference is a port bug or a genuine model improvement.

**Caveat (as of 2026-06-06).** The reference DDR state lives only in the desktop's `~/projects/ddr` working tree (contains unpushed `geometry/trapezoidal.py` changes). A fixture regenerated from a clean DDR clone diverges ~1% per commit — that is a wrong reference, not a port bug. See `.claude/references/ddrs-comparing-to-ddr.md §Regenerating fixtures` before regenerating.

---

### Invariant 2 — f32 throughout routing core

**Rule.** No casts to f64 or bf16 inside `src/routing/`, `src/geometry.rs`, or `src/sparse.rs`. The DDR comparison sits at the f32 precision floor (~1e-7 relative difference per reach); any precision change breaks reproducibility.

**Rationale.** Mixed precision introduces per-reach rounding that accumulates across the 346,321-reach CONUS network; the 1e-3 m³/s sandbox tolerance is calibrated for f32-only arithmetic.

---

### Invariant 3 — lower-triangular adjacency

**Rule.** The adjacency matrix must be topologically sorted and lower-triangular: `rows[k] >= cols[k]` for every non-zero entry. The forward-substitution solver (`triangular_solve_lower`) assumes no upstream values are uncomputed when it processes a reach.

**Rationale.** Forward substitution over a topological order is the entire basis for the O(n) per-timestep solve. A non-lower-triangular entry means a downstream reach tries to read an upstream value before that upstream reach is solved — silent wrong output, no error.

**Test.** `cargo test data_zarr_store::conus_adjacency_loads_real_merit_zarr` verifies the invariant on the real CONUS zarr store.

---

### Invariant 4 — hand-written sparse backward

**Rule.** Do NOT replace the hand-written `CsrSolveOp impl Backward` in `src/sparse.rs` with autograd-tape unrolling.

**Rationale.** The entire point of the custom backward is O(nnz) tape entries per timestep. Tape unrolling would be O(n²) for a triangular solve — quadratic memory and time, infeasible at CONUS scale (346,321 reaches). The analytical backward is `∇A = -gradb[rows]·x[cols]`, exactly as in DDR's `torch.autograd.Function`.

**Reference.** `.claude/references/ddrs-burn-autograd.md` has the full BURN-0.21 recipe.

---

### Invariant 5 — KAN head architecture matches DDR

**Rule.** The routing head is `rskan::KanLayer` via `src/nn/kan_head.rs`. The architecture is `Linear(F, H) → KanLayer(H, H) × num_hidden_layers → Linear(H, P) → Sigmoid`. No inter-block ReLU. All `num_hidden_layers` inner KanLayers receive the SAME seed (DDR's `kan.py` lines 24–34 quirk — preserved for parity).

**Rationale.** DDR parity requires identical weight initialization. A ReLU between KAN blocks or different per-layer seeds changes the initialization and breaks the fixture tests.

**What NOT to do.** Do not reintroduce the prior MLP placeholder.

---

### Invariant 6 — rskan pinned to a tag

**Rule.** `rskan` in `Cargo.toml` must be a git dependency pinned to a tag, currently `v0.1.3`. When updating, bump the tag, then re-run all Tier B tests and the Tier A DDR regression before merging.

**Rationale.** An unpinned git dependency (`branch = "main"`) can change silently on `cargo update`, breaking KAN parity without any local code change.

**Current pin (as of 2026-07-05):**
```toml
rskan = { git = "https://github.com/taddyb/rskan.git", tag = "v0.1.3" }
```

---

### Invariant 7 — KAN head parity on every relevant PR

**Rule.** Any PR touching `src/nn/`, `Cargo.toml`'s rskan pin, or DDR's `nn/kan.py` must pass the full KAN parity suite (see Tier B gate above).

**Rationale.** Invariant 5 is not self-enforcing at the compiler level. The fixture tests are the only automated check that the architecture, seed, and weight initialization actually match DDR.

---

## The STALE-BINARY TRAP (historical incident: 2026-07-01)

**Rule.** After touching ANY file under `src/`, refresh the installed binary before running experiments.

```bash
# Canonical (always correct):
cargo install --path .

# Faster if target/release is already built:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs

# Bypass entirely (safest for one-off experiments):
cargo run --release --bin ddrs -- run --workflow train-and-test ...
```

**Why this matters.** `ddrs` on your PATH is `~/.cargo/bin/ddrs`. `cargo build` and `cargo run` do NOT update it. The manifest's `git.sha` is stamped from `.git` at runtime, so a run appears to have the correct SHA while silently executing a weeks-old binary.

**What happened (2026-07-01).** The installed `ddrs` was dated 2026-06-03, before disaggregation (landed 2026-06-19) and leakance (landed 2026-06-29). The hourly-forcing cell silently ran flat repeat-24 with no leakance. Both hourly-ON and daily-ON cells produced byte-identical eval predictions (`52ec721`). The manifest showed `git.sha = 2cdd341` (correct HEAD), masking the stale binary completely.

**Self-check.** Current checkpoints are DIRECTORIES: `.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/head.mpk`. A stale pre-checkpoint-resume binary writes FLAT files: `epoch_E_mb_M.mpk`. Flat files = stale binary.

---

## CUDA graphs + NaN masking (known gotcha)

**Rule.** Validate model forwards with `use_cuda_graphs: false` when debugging NaN loss or unexpected constant loss.

**Why.** `use_cuda_graphs: true` captures a kernel graph on the first forward pass and replays it on subsequent passes. If the first forward produces a NaN (e.g., during early training on bad data), the captured graph replays stale finite values rather than recomputing and propagating the NaN. The result is a constant finite loss that does not go to NaN even when the actual computation is invalid. See memory file `cuda-graphs-mask-nan.md` for the full diagnosis.

**Hard constraint.** `params.use_leakance: true` combined with `use_cuda_graphs: true` is REJECTED at config load time. The leakance kernel is not captured in the current CUDA graph implementation; the rejection prevents silent wrong results.

---

## Leakance-specific gates

Leakance (`params.use_leakance: true`) is experimental and off by default. Any change that enables, modifies, or interacts with leakance must satisfy these gates in addition to the appropriate tier gates.

### Enabling leakance requires three config changes together

Missing any one causes either a config-load error or silent wrong behavior:

| Config key | Required value | Why |
|---|---|---|
| `params.use_leakance` | `true` | Activates the leakance kernel in `route_timestep` |
| `kan_head.learnable_parameters` | Include `K_D`, `d_gw`, `leakance_factor` | Without these, the KAN head does not emit leakance params → all-zero zeta |
| `params.parameter_ranges.K_D` | `[1e-8, 1e-6]` (log-space) | Range gate; current recommendation is `[1e-8, 1e-5]` for recoverability experiments (see §Research status) |
| `params.parameter_ranges.d_gw` | `[-2, 2]` | Groundwater depth offset (m) |
| `params.parameter_ranges.leakance_factor` | `[0, 1]` | Dimensionless scale |
| `use_cuda_graphs` | `false` | Enforced by config load; leakance + graphs = rejected |

### Leakance gradient-exactness gate

```bash
cargo test --test leakance_gradcheck       # 8/8 — all analytical grads match finite-diff
cargo test --test leakance_off_parity      # 3/3 — OFF is byte-identical to no-leakance
cargo test --test zeta_accum              # 6/6 — accumulated zeta == headwater q difference
cargo run --release --example compare_ddr_sandbox  # must still say ABSOLUTE MATCH
```

### Leakance eval-time zeta diagnostic

`dump_parameters` exports learned `K_D`/`d_gw`/`leakance_factor` per COMID but NOT the actual zeta flux (which depends on routed depth, only available during eval). The zeta diagnostic runs during `ddrs run --workflow train-and-test` Phase 2 automatically. For an existing checkpoint:

```bash
cargo build --release --bin eval
target/release/eval \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
  --output /tmp/eval.zarr \
  --zeta-output .ddrs/runs/<id>/kan_parameters.nc
```

Output variables in `kan_parameters.nc` (dimension `COMID_eval`, 64,892 reaches on the CONUS eval network):
- `zeta` — mean |zeta| (m³/s) over eval window
- `zeta_net` — signed mean zeta (positive = losing reach)

**GO/NO-GO bar.** |zeta| > 0.01 m³/s on at least 10% of eval reaches = zeta is physically active.

---

## Research-status facts (as of 2026-07-05)

These facts govern what claims can be made about leakance. Cite dates; do not generalize beyond what is measured.

### Leakance 2×2 (as of 2026-07-01) — DONE

Four valid arms, seed 42, eval window 1995/10/01–2010/09/30, 2,365 gauges:

| arm | run id | NSE med | KGE med |
|---|---|---|---|
| hourly-OFF | `2026-06-23T02-49-12Z-conus-hourly-train-and-test` | 0.7153 | 0.7104 |
| hourly-ON  | `2026-07-01T13-43-32Z-train-and-test` | 0.7145 | 0.7150 |
| daily-OFF  | `2026-06-05T01-41-16Z-train-and-test` | 0.7004 | 0.7244 |
| daily-ON   | `2026-07-01T21-20-27Z-train-and-test` | 0.6963 | 0.7250 |

Losing-stream subset (1,883/2,365 gauges): hourly leakance ΔNSE +0.0005, ΔKGE +0.0018, 55.5% of gauges improve. Daily leakance ΔNSE −0.0017, ΔKGE −0.0009. Verdict: **GO — marginal** (3/3 gates met; zeta |>0.01| on 10.4% of 64,892 eval reaches).

**Summed-Q' baseline (CONUS):** median NSE 0.689, KGE 0.723. Best trained result (precip-driven disagg + L1): median NSE 0.715, KGE 0.711 (2,365 gauges). NSE beats the baseline by +0.026; KGE does NOT beat the summed-Q' baseline in any config as of 2026-07-05.

### Low-zeta diagnosis (as of 2026-07-02)

| Hypothesis | Verdict |
|---|---|
| H1 — K_D ceiling clips zeta | REFUTED (71.5% of reaches CAN exceed 0.01 m³/s in-box; utilization median 3.4%) |
| H2 — driving-head starvation | SUPPORTED (median head 0.021 m; 47% of reaches gaining at eval-window mean) |
| H3 — KAN variance collapse | REFUTED (K_D–aridity ρ +0.61, d_gw–meanP ρ +0.71 — strong learned structure) |
| H4 — gauge bias / gradient starvation | SUPPORTED (gauged median |zeta| 11× ungauged; dry/wet zeta ratio 0.40, inverse of physics) |
| H5 — equifinality (daily only) | SUPPORTED (daily Δn +0.012, 0.59 IQR; hourly Δn nil) |
| H6 — wrong yardstick | REFUTED (fractional loss agrees: 8.4% of reaches lose >1% of local flow) |
| H7 — model-form error | REFUTED (0.0% of d_gw at bounds) |

**Implication.** The K_D-widening follow-up (`[1e-8, 1e-5]`) recommended in the 2×2 findings is NOT recommended by the diagnosis. The diagnosis shows the K_D box is not the binding constraint. Widening K_D alone is expected to re-pin at the new ceiling with negligible zeta or skill change.

### Gradient probe (as of 2026-07-03, worktree: zeta-sensitivity)

| Probe | Verdict | Key number |
|---|---|---|
| P1 — gradient starvation | REFUTED | gauged/ungauged \|g\| ratio 1.5× trained, 2.9× cold (bar: ≥10×) |
| P2 — rejection at trained point | REFUTED | 52.5% of dry-tercile grads push zeta down (bar: >67%) |
| P3 — detectability | NO-GO | 4.2% of Ref probes detectable at δ=0.01 m³/s (bar: ≥10%); median 5%-band 0.531 m³/s vs planted signal 0.01 m³/s = **53× dilution** |

P3 NO-GO means: gauge-only discharge supervision cannot distinguish real-world leakance magnitudes from measurement uncertainty. Transmission is fine (~95% fidelity); the problem is signal-to-noise at the sensor.

### Synthetic recoverability positive control (as of 2026-07-04, worktree: zeta-sensitivity)

| Metric | Measured | Verdict |
|---|---|---|
| R1 — recovery ratio median (n=58) | 0.009 | FAILED (bar: ≥0.5) |
| R2 — non-planted spatial precision | 1.11× baseline | PRECISE (trivial: model didn't move) |
| R3 — loss gap A vs B | A=1.339 vs B=2.317, +42% | A<B but CONFOUNDED (B's step-0 handicap accounts for gap) |
| R5 — cold emergence ratio | 1.20 | SUPPRESSED (bar: >3) |

**Root cause.** Windowed training objective (rho=90, warmup=5) has a ~130× hotstart-transient noise floor vs the planted signal. The continuous residual with teacher weights + teacher obs is 0.0076 mean L1; step-0 windowed training loss is 1.017. The optimizer chases irreducible initial-condition noise; after 5 epochs the continuous residual degrades from 0.0076 to 0.4431.

**Implication.** Leakance identifiability is NOT proven. Phase B (state-cache hotstart, target: windowed loss ≤ 0.25 mean L1, ≤10% of a converged run) is required before any identifiability claim. Phase B is NOT yet complete as of 2026-07-05.

---

## Binary management quick-reference

| Goal | Command |
|---|---|
| Refresh installed binary after src/ change | `cargo install --path .` |
| Fast refresh (target/release already built) | `cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs` |
| Bypass installed binary for one run | `cargo run --release --bin ddrs -- run --workflow train-and-test ...` |
| Check if stale binary ran | Look for FLAT checkpoint files `epoch_E_mb_M.mpk`; current binary writes DIRECTORIES `epoch_E_mb_M/head.mpk` |

---

## Workspace flag gotcha

`--workspace` takes the path to the `.ddrs` DIRECTORY ITSELF, not its parent. Experiment configs in `config/experiments/` default to `config/experiments/.ddrs` (wrong). Always pass the root workspace explicitly:

```bash
ddrs run --config config/experiments/leakance_hourly_on.yaml \
         --workspace /home/tbindas/projects/ddrs/.ddrs \
         --workflow train-and-test
```

---

## Provenance and maintenance

Files read to write this skill (re-read to verify any fact):

```bash
# Core invariants and CLI behavior
cat /home/tbindas/projects/ddrs/CLAUDE.md

# Architecture and per-timestep dataflow
cat /home/tbindas/projects/ddrs/.claude/ARCHITECTURE.md

# Stale-binary incident + 2×2 experiment
cat /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-experiment-handoff.md

# 2×2 final findings and GO-marginal verdict
cat /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-findings.md

# Low-zeta diagnosis (H1-H7 verdicts)
cat /home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md

# Gradient probe (P1-P3 verdicts) — worktree
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-03-zeta-gradient-probe-findings.md

# Recoverability positive control failure — worktree
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-04-synthetic-recoverability-findings.md

# rskan version pin
grep rskan /home/tbindas/projects/ddrs/Cargo.toml
```

Re-verification commands:
```bash
# Confirm invariant 1 still holds on current HEAD
cargo run --release --example compare_ddr_sandbox

# Confirm rskan pin
grep rskan /home/tbindas/projects/ddrs/Cargo.toml

# Confirm leakance tests pass
cargo test --test leakance_gradcheck --test leakance_off_parity --test zeta_accum
```
