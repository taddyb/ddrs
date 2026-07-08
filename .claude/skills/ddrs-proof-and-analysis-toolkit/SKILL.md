---
name: ddrs-proof-and-analysis-toolkit
description: "Use when you need to verify gradient correctness, prove a new op does not break existing behavior, design a controlled ablation experiment, pre-register falsifiable hypotheses before a training run, or measure whether a learned parameter is identifiable from gauge observations. Also use when a new routing term (analogous to leakance) is proposed and needs an evidence chain before GPU time is spent."
---

# ddrs proof and analysis toolkit

Five first-principles methods, each as a self-contained recipe. Verified against
actual ddrs experiments as of 2026-07-05.

---

## Jargon glossary (read once, skip later)

| Term | Definition |
|---|---|
| **BURN** | Rust deep-learning framework (v0.21). ddrs uses `burn::backend::Autodiff<B>` for autograd. Analogous to `torch.autograd`. |
| **NdArray backend** | CPU-only BURN backend (`NdArray<f32>`). Deterministic: two identical runs produce bit-identical output. Use it for all gradcheck and parity work. |
| **CsrSolveOp** | Custom BURN `Backward<B,2>` impl in `src/sparse.rs`. Solves the lower-triangular reach system per timestep. O(nnz) tape entries — must never be replaced by unrolled autograd. |
| **TimestepLeakanceOp** | Custom `Backward<B,8>` in `src/routing/mmc_op.rs`. Eight tracked parents: n, q_spatial, p_spatial, q_t, q_prime_t, K_D, d_gw, leakance_factor. |
| **zeta** | Leakance flux (m³/s), subtracted from routing RHS: `zeta = leakance_factor · area_z · K_D · (depth − d_gw)`. Positive = losing reach. |
| **Q'** (q_prime) | Upstream-summed lateral inflow forcing (m³/s) at each MERIT reach, read from the icechunk store. Not routed discharge. |
| **rho-window** | A contiguous training time slice of length `rho` days, hot-started with heuristic initial conditions. The training objective is L1 averaged over gauges in that window. |
| **COMID** | Unique integer ID for a MERIT reach (one row in the geospatial fabric). |
| **STAID** | USGS or global gage identifier string (e.g. `"USGS__01234567"`). |
| **KAN head** | `rskan::KanLayer`-based neural network (`src/nn/kan_head.rs`). Maps per-reach catchment attributes → normalized routing parameters. |
| **ABSOLUTE MATCH** | Max abs diff < 1e-3 m³/s on the 5-reach DDR sandbox. The V1 gate — any routing change must clear it. |
| **IQR** | Interquartile range (p75 − p25). Used throughout as a spread estimator for learned parameters. |
| **Spearman ρ** | Rank correlation. Preferred over Pearson here because parameter distributions have long tails. |

---

## When NOT to use this skill

| You want to... | Use instead |
|---|---|
| Add a new routing feature (design phase) | `/superpowers brainstorming` → `/superpowers writing-plans` |
| Debug a training divergence or NaN | `CUDA graphs mask NaN` memory note + `use_cuda_graphs: false` + `--test training_verification` |
| Port a new formula from DDR Python | `.claude/ARCHITECTURE.md` + `ddrs-burn-autograd.md` reference |
| Check the stale-binary trap | CLAUDE.md §STALE-BINARY TRAP |
| Run a full train+eval experiment | `ddrs plan && ddrs run --workflow train-and-test` |

---

## Method 1: Gradcheck (analytical vs finite-difference gradient)

**Purpose.** Prove a custom BURN `Backward` impl computes the correct adjoint.
Required after every change to `src/routing/leakance.rs`, `src/routing/mmc_op.rs`,
or `src/sparse.rs`.

### How it works

For each tracked parent tensor `θ_i`:

1. Compute analytical gradient via `loss.backward()` + `.grad(&grads)`.
2. Estimate the same gradient with central finite differences:
   `fd_grad[i] = (loss(θ_i + ε) − loss(θ_i − ε)) / (2ε)`
3. Accept if `|analytical − fd| < REL_TOL · max(|analytical|, |fd|)` OR `|analytical − fd| < ABS_TOL`.

### Choosing ε

| Parent type | Recommended ε | Reason |
|---|---|---|
| Geometry params (n, q_spatial, p_spatial) | `max(1e-3 · |base|, 1e-3)` | Nonlinear; large ε introduces truncation error |
| Linear params (K_D, d_gw, leakance_factor) | Large relative step (0.4–1.5× base) | zeta is exactly linear in these; tiny ε sinks into f32 round-off of `q_next.sum()` (~O(500)) |

The leakance gradcheck uses `EPS=4e-7` for K_D, `1.5` for d_gw, `0.4` for
leakance_factor — explicitly documented in `tests/leakance_gradcheck.rs:316-320`.

### Commands

```bash
# Base routing op (5 parents): n, q_spatial, p_spatial, q_t, q_prime_t
cargo test --test sp8_gradcheck

# Leakance op (8 parents: base 5 + K_D, d_gw, leakance_factor)
cargo test --test leakance_gradcheck

# Sparse CSR backward vs DDR Python TriangularSparseSolver adjoint
# (requires fixtures from scripts/dump_solver_gradcheck.py run once)
cargo test --test sparse_gradcheck
```

All three run on `NdArray<f32>` (CPU, deterministic). Tolerances: `REL_TOL=5e-3`,
`ABS_TOL=1e-4` (from `leakance_gradcheck.rs:30-31`).

### Worked example

From `tests/leakance_gradcheck.rs`: a 4-reach linear chain, interior leakance
values (`K_D=5e-7`, `d_gw=0.0`, `leakance_factor=0.5`), losing config
(`depth > d_gw`). Each of the 8 parents is swept independently: the analytical
grad for parent `i` is extracted, then FD is computed element-wise. The test
confirms `TimestepLeakanceOp`'s `Backward<B,8>` adjoint is correct before any
leakance training run.

### Checklist

- [ ] Base point is interior to all parameter ranges (no clamp saturation)
- [ ] Losing config for leakance tests: `depth > d_gw` at base point
- [ ] x_storage is NOT a tracked parent of `TimestepLeakanceOp` — do not try to sweep it
- [ ] After any leakance change: run full guard suite (see §Guard suite)

---

## Method 2: Parity test (byte-identical OFF path)

**Purpose.** Prove that adding a new feature does not alter results when the
feature is disabled. Catches silent regressions where the OFF-path code is
accidentally modified.

### Pattern

```
committed expected output (captured before the change)
  ↕  must be byte-identical (abs diff < 1e-6 f32 units)
new code with feature=None/false
```

### Commands

```bash
# Leakance-off path byte-identical to pre-leakance routing output
cargo test --test leakance_off_parity

# Full DDR parity regression (max abs diff must be < 1e-3 m³/s)
cargo run --release --example compare_ddr_sandbox
```

### Worked example

`tests/leakance_off_parity.rs::leakance_none_matches_baseline_chain`: a
5-reach × 24-step linear chain. The constant `EXPECTED: [f32; 120]` is the
committed pre-leakance hydrograph. The test runs `MuskingumCunge::forward()`
with `leakance = None` and asserts every element matches within 1e-6.

A second test (`leakance_removes_water_on_losing_config`) guards the opposite:
with a losing config (`K_D` at ceiling, `d_gw` at floor, `factor=1`), the
sum of routed discharge must be strictly less than without leakance. This
prevents silent no-ops in the ON path.

### Checklist

- [ ] Capture the expected output BEFORE introducing the new feature branch
- [ ] Use `NdArray<f32>` (deterministic backend) for the comparison
- [ ] Test both "feature off → same as before" AND "feature on → output actually changes"
- [ ] Run `compare_ddr_sandbox` after any `src/routing/` or `src/sparse.rs` change

---

## Method 3: Paired ON/OFF experiment (equifinality / compensation test)

**Purpose.** Measure whether routing parameters (Manning's n, x_storage)
shift when a new term is added — the signal of equifinality (Beven's
bias-compensation: other parameters absorb what the new term would explain).

### Setup

Train two runs with IDENTICAL seed, window, data, and checkpoint schedule.
Differ ONLY in the feature flag. Export `dump_parameters` for both.

```bash
# Build a leakance-ON config (clone from the OFF config, add use_leakance + K_D/d_gw/factor ranges)
cp config/experiments/daily_baseline.yaml config/experiments/leakance_daily_on.yaml
# edit: params.use_leakance: true, add K_D/d_gw/leakance_factor to learnable_parameters + ranges

# Run both (same seed)
ddrs run --workflow train-and-test --config config/experiments/daily_baseline.yaml
ddrs run --workflow train-and-test --config config/experiments/leakance_daily_on.yaml

# Export full-CONUS learned params for each
target/release/dump_parameters --config <ON config> --checkpoint <ON ckpt/head> --output params_on.nc
target/release/dump_parameters --config <OFF config> --checkpoint <OFF ckpt/head> --output params_off.nc
```

### How to read the result

Compute `Δn = n_ON − n_OFF` per COMID. Normalize by the parameter's own IQR:

```
equifinality signal = median(|Δn|) / IQR(n)
```

| Value | Interpretation |
|---|---|
| < 0.10 | No equifinality — parameters independent |
| 0.10 – 0.49 | Weak compensation — ambiguous |
| ≥ 0.50 | Equifinality confirmed — routing co-adjusts to absorb the new term |

### Worked example (as of 2026-07-02, H5 diagnosis)

From `docs/2026-07-02-leakance-diagnosis-findings.md`:

| Forcing | Δn median | IQR(n) | IQR-normalized shift | Verdict |
|---|---|---|---|---|
| Daily | +0.012 | ~0.020 | 0.59 | SUPPORTED — n absorbs leakance |
| Hourly | +0.001 | ~0.019 | 0.05 | REFUTED — mechanisms decouple |

Interpretation: under flat-daily forcing, Manning's n rises ~20% when leakance is
added (the Kirchner bias-compensation signature). Under hourly forcing the
sub-daily depth signal provides additional discriminating information and the
shift is nil. **This is the mechanistic basis for the "leakance requires hourly
forcing" design gate.**

### Caveats

- Single seed per arm. CUDA scatter-add nondeterminism adds ~2–5% noise to
  parameter distributions. Treat as suggestive, not decisive.
- x_storage equifinality requires BOTH arms to learn x_storage. If the OFF run
  used a fixed x_storage constant, the comparison is undefined — note it.
- The ON/OFF pair does not isolate CUDA nondeterminism from true equifinality.
  If the verdict is ambiguous, run each arm twice (different seeds) and compare
  the within-arm spread to the cross-arm shift.

---

## Method 4: Pre-registered hypothesis battery

**Purpose.** Distinguish competing explanations for an anomalous result
(e.g., learned parameters are smaller than expected) without spending GPU on
a speculative retrain. Register hypotheses BEFORE analyzing data; assign
verdicts from pre-specified falsification criteria.

### Recipe

1. State the anomaly (one sentence, quantified).
2. Write 5–10 mutually exclusive hypotheses. Group them: "the box is wrong" /
   "the gradient is wrong" / "the question is wrong".
3. For each hypothesis, specify:
   - A test (what to compute, from which artifact)
   - A falsification criterion (the measurable condition that would REFUTE it)
   - A pre-registered verdict threshold
4. Run a single analysis script. Print verdicts as `[SUPPORTED]` / `[REFUTED]` /
   `[INCONCLUSIVE]` with the key number.
5. Write a findings doc. The ranked table of verdicts is the deliverable.
6. **Phase 3 gate**: define a boolean gate from the verdict table BEFORE the
   battery. Spend GPU only if the gate opens.

### Template

```markdown
## Hypotheses

| # | Hypothesis | Test | Falsified if |
|---|---|---|---|
| H1 | Structural (box/architecture) | closed-form max-feasible value from learned geometry | most units could exceed the bar |
| H2 | Parameter throttling (driving head) | distribution of the active factor | large fraction of active factor > threshold |
| H3 | Network variance collapse | parameter dispersion + attribute Spearman ρ | max |ρ| > 0.4 against any attribute |
| H4 | Gradient bias toward training signal | stratify metric by coverage tercile | no stratification pattern |
| H5 | Equifinality with existing params | paired ON/OFF IQR-normalized shift | shift < 0.5 IQR in both arms |
| H6 | Wrong yardstick (absolute vs relative) | fractional metric agrees with absolute | fractional metric also negligible |
| H7 | Model-form error | boundary-pinning rate | < 5% of units pin at a bound |

## Phase-3 gate
Gate opens iff: H_structural SUPPORTED AND H_gradient NOT showing total collapse.
```

### Worked example (2026-07-01 to 2026-07-02 leakance diagnosis)

**Anomaly:** The leakance 2×2 returned GO-marginal. Median |zeta| = 6.4e-4 m³/s
(hourly-ON run); K_D pinned at the 1e-6 s⁻¹ ceiling on 100% of reaches.
Question: why is zeta so small?

Seven pre-registered hypotheses (spec:
`docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md`):

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H1 | Structural K_D ceiling | REFUTED | 71.5% of reaches CAN exceed 0.01 m³/s in-box; median utilization 3.4% |
| H2 | Driving-head starvation | SUPPORTED | median head 0.021 m; 47.0% of reaches gaining at mean |
| H3 | KAN variance collapse | REFUTED | K_D–aridity Spearman +0.61, d_gw–meanP +0.71 |
| H4 | Gauge bias / gradient starvation | SUPPORTED | zeta–uparea ρ +0.76; gauged median |zeta| 11× ungauged |
| H5 | Equifinality (daily forcing only) | SUPPORTED | daily Δn +0.012, IQR ratio 0.59; hourly 0.05 |
| H6 | Wrong yardstick (absolute bar) | REFUTED | fractional loss agrees: 8.4% > 1%, 3.2% > 5% |
| H7 | Model-form error (d_gw boundary-pinning) | REFUTED | 0.0% of d_gw at bounds |

**Phase-3 gate outcome:** H1 REFUTED (the box is not the cap) → gate FAILED →
widened-K_D retrain NOT run. This supersedes the "widen K_D" recommendation
in `docs/2026-07-01-leakance-hourly-findings.md`.

**Script:**
```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/leakance_diagnosis.py
```

**Full raw output** in `docs/2026-07-02-leakance-diagnosis-findings.md` §6.

### Checklist

- [ ] Hypotheses written and spec committed BEFORE opening the data
- [ ] Each hypothesis has a single falsification criterion, not a range
- [ ] Phase-3 gate defined before the battery runs
- [ ] Verdicts reported with key numbers even when REFUTED
- [ ] Confounds acknowledged (CUDA nondeterminism, mean vs. peak depth, single seed)

---

## Method 5: Input-perturbation identifiability test

**Purpose.** Measure whether a gradient signal exists and whether training can
recover a planted parameter value. Two instruments, run in sequence:

| Instrument | What it measures | Training required |
|---|---|---|
| Adjoint reachability map | Does `∂Loss/∂param` reach the target reaches at all? | No (one mini-batch loop) |
| Detectability bound | Does a planted delta produce a gauge signal above observational noise? | No (forward-only eval) |
| Synthetic recoverability | Does training actually attribute the planted signal to the correct parameter? | Yes (full training run) |

Run the adjoint map and detectability bound BEFORE training. If detectability
is NO-GO, stop — no training objective can learn the term.

### Instrument A: adjoint reachability map

```bash
# Build the probe binary (worktree-zeta-sensitivity has it; rebuild after any src/ change)
cargo build --release --bin probe_zeta_gradient

# Run in --mode grad: replicates training mini-batch loop, accumulates per-reach |∂L/∂param|
target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
  --mode grad --windows 96 --seed 42 \
  --output output/probe/grad_trained.nc

# Also run without checkpoint (cold init) to separate "converged-flat" from "never-saw-signal"
target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --mode grad --windows 96 --seed 42 \
  --output output/probe/grad_cold.nc
```

**Reading the result.** Stratify `mean|∂L/∂factor|` by gauged/ungauged and
aridity tercile. Pre-register the starvation bar before looking at the numbers.

| gauged/ungauged ratio | Verdict |
|---|---|
| ≥ 10 at both trained AND cold | SUPPORTED — gradient is dead off-gauge |
| < 10 at either point | REFUTED — signal reaches everywhere |

**Worked example (2026-07-03 gradient probe, as of 2026-07-05):**

| Point | Gauged | Ungauged | Ratio | Starvation bar (≥10) |
|---|---|---|---|---|
| Trained (`epoch_5_mb_9`) | 6.04e-5 | 4.00e-5 | 1.5× | REFUTED |
| Cold (seed-42 init) | 3.92e-4 | 1.36e-4 | 2.9× | REFUTED |

P1 starvation REFUTED. The gradient reaches everywhere.

Cold-point sign map: 80.5% of ungauged gradients push zeta DOWN at the cold
point (initial-training suppression). At the trained point: 52.5% dry-tercile
push-down (approximately neutral). The physics term differentiated
correctly at convergence; the obstacle is the signal level at sensors.

### Instrument B: detectability bound

```bash
# Select probe sites (GAGES-II Ref gauges, stratified by area × aridity × stage-1 reachability)
uv run python scripts/zeta_probe_sites.py   # outputs probe_plan.csv

# Run --mode perturb: forward-only eval with +delta m3/s added to q_prime at planted reaches
target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
  --mode perturb --probe-plan output/probe/probe_plan.csv \
  --eval-days 1095 --output output/probe/perturb

# Verdicts
cd ~/projects/ddr && uv run python scripts/zeta_gradient_analysis.py
```

**Detection criterion (pre-register before running):**
`|mean ΔQ|` > 99th-pct rerun noise AND > 5% of gauge's mean flow
(McCallum 2012 differential-gauging detectability band).

**Detectability NO-GO bar: < 10% of Ref probes at `δ=0.01 m³/s` detectable.**

**Worked example (2026-07-03, as of 2026-07-05):**

| Probe class | δ | Detectable | Bar |
|---|---|---|---|
| Ref gauges (GAGES-II) | 0.01 m³/s | 4.2% (4/96) | NO-GO (< 10%) |
| Ref gauges | 0.1 m³/s | 16.7% | — |
| Non-ref gauges | 0.01 m³/s | 0.0% | — |
| Non-ref gauges | 0.1 m³/s | 2.1% | — |

Failure decomposition: planted flux transmits to gauges at ~95% fidelity
(median `|mean ΔQ|/δ = 0.946`). Failure is dilution, not transmission.
Median Ref 5%-band = 0.53 m³/s — 53× the 0.01 m³/s literature-magnitude loss.

**P3 detectability: NO-GO.** No gauge-only objective can learn
literature-magnitude leakance signals. This is empirically measured, not argued.

### Instrument C: synthetic recoverability (positive control)

Run this ONLY if detectability passes or if you construct a synthetic world
where the signal is detectable by construction. If the positive control fails,
stop — no supervised rescue can work until the underlying noise floor is reduced.

```bash
# 1. Teacher: generate synthetic obs with known planted zeta
target/release/probe_zeta_gradient \
  --mode teacher --backend cpu \
  --config config/experiments/recoverability_teacher.yaml \
  --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
  --plant-file output/recoverability/plants.csv \
  --obs-output output/recoverability/synthetic_obs \
  --zeta-output output/recoverability/answer_key.nc

# 2. Student A: warm-start from teacher weights, leakance ON
#    (bootstrap_head_and_state: place head.mpk in a dir; missing optim.mpk → Adam cold)
target/release/train --backend cpu \
  --config config/experiments/recoverability_student_a.yaml \
  --checkpoint-dir output/recoverability/students/a

# 3. Student B: same warm-start, leakance OFF (equifinality control)
target/release/train --backend cpu \
  --config config/experiments/recoverability_student_b.yaml \
  --checkpoint-dir output/recoverability/students/b

# 4. Measure A's final zeta vs the answer key
target/release/eval --backend cpu \
  --config config/experiments/recoverability_measure.yaml \
  --checkpoint output/recoverability/students/a/epoch_5_mb_<N> \
  --zeta-output output/recoverability/zeta_a.nc

# 5. Verdicts
uv run python scripts/recoverability_analysis.py
```

**Pre-registered verdict bars:**

| Metric | Pass bar | Fail bar |
|---|---|---|
| R1: median recovery ratio | ≥ 0.50 RECOVERED | ≤ 0.10 FAILED |
| R2: non-planted |zeta_net| A/baseline | < 2× PRECISE | ≥ 2× SMEARED |
| R3: final-epoch loss A vs B | A < B by > 5% rel | A ≈ B |
| R5: cold emergence ratio | > 3× EMERGES | — |

**Headline pass criterion: R1 ≥ 0.50 AND R3 shows A < B.**

**Worked example (2026-07-04, as of 2026-07-05):**

| Metric | Measured | Verdict |
|---|---|---|
| R1 recovery ratio (n=58 planted reaches) | 0.009 (p10=−0.073, p90=0.199) | FAILED |
| R2 non-planted precision | 1.11× baseline | PRECISE (but trivially — nothing moved) |
| R3 loss gap A vs B | 42.2% (A 1.339, B 2.317) | CONFOUNDED (see below) |
| R5 cold emergence | 1.20× | SUPPRESSED |

**HEADLINE: POSITIVE CONTROL FAILED.**

**Root cause (decomposition):** The training objective (rho-90 windows, warmup-5)
has a ~130× hotstart-transient noise floor relative to the planted signal:

| Quantity | Value |
|---|---|
| Continuous residual at teacher weights + teacher obs (full-window eval) | 0.00759 mean L1 |
| Step-0 windowed training loss (run A, warm-started) | 1.017 |
| Ratio (noise floor / signal) | ~130× |
| Run A continuous residual AFTER 5 epochs | 0.4431 (58× worse than not training) |

The planted signal is 0.8% of the training loss. Adam chases irreducible
initial-condition noise and makes the model worse.

**Implication:** Leakance identifiability is NOT proven. Phase B (state-cache
hotstart, noise floor target ≤ 0.25 mean L1) is required before any
identifiability claim. This is the current blocker as of 2026-07-05.

R3 confound: B's step-0 handicap (loss 2.323 vs A's 1.017) accounts for
approximately its final gap. The 42.2% loss gap measures that the leakance BASE
FIELD matters in aggregate — not that individual planted fluxes were learnable.

### Full identifiability checklist

- [ ] Adjoint reachability map run at BOTH trained AND cold init points
- [ ] Detectability bound uses pre-registered δ (literature-magnitude loss)
- [ ] Detectability sites filter to GAGES-II Ref class (exclude regulated rivers)
- [ ] Positive control has a CONTINUOUS baseline eval before training starts
- [ ] Recovery target is the FLUX FIELD (per-reach zeta_net), not the degenerate parameter triple
- [ ] Phase-3 gate defined: do NOT train if detectability is NO-GO
- [ ] Phase B (state-cache hotstart) is mandatory if windowed noise floor > 0.25 mean L1

---

## Guard suite (run after every change to routing core)

```bash
cargo test --test leakance_gradcheck        # 8/8 parents, analytical ≈ FD
cargo test --test leakance_off_parity       # 3/3: byte-equal off, water removed on, head-driven differs
cargo test --test zeta_accum               # 6/6: accumulator identity, no routing perturbation
cargo run --release --example compare_ddr_sandbox  # ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

All four must be green before any PR merge that touches `src/routing/`,
`src/sparse.rs`, `src/geometry.rs`, or `src/routing/leakance.rs`.

For KAN head changes:

```bash
cargo test --features fixtures \
  --test kan_head_init_repro \
  --test kan_head_init_parity \
  --test kan_head_fixture_forward \
  --test kan_head_fixture_backward
```

---

## Key numbers reference (as of 2026-07-05)

| Metric | Value | Source |
|---|---|---|
| Summed-Q' baseline NSE (CONUS, 2365 gauges) | 0.689 | CLAUDE.md |
| Summed-Q' baseline KGE (CONUS, 2365 gauges) | 0.723 | CLAUDE.md |
| Best result: precip-disagg + L1 NSE | 0.715 | CLAUDE.md (2026-06-23) |
| Best result: precip-disagg + L1 KGE | 0.711 | CLAUDE.md (2026-06-23) |
| KGE vs summed-Q' baseline | KGE does NOT beat baseline in any config | CLAUDE.md |
| Leakance 2×2 GO gate: |zeta| > 0.01 on eval reaches | 10.4% (bar: ≥10%) | docs/2026-07-01-leakance-hourly-findings.md |
| Gradient probe P3 detectability (Ref, δ=0.01) | 4.2% (4/96) — NO-GO | docs/2026-07-03-zeta-gradient-probe-findings.md |
| Recoverability positive control R1 recovery ratio | 0.009 — FAILED | docs/2026-07-04-synthetic-recoverability-findings.md |
| Windowed training noise floor / planted signal | ~130× | docs/2026-07-04-synthetic-recoverability-findings.md |
| Phase B target noise floor | ≤ 0.25 mean L1 | CLAUDE.md |
| CONUS MERIT reaches | 346,321 | CLAUDE.md |
| Eval network size | 64,892 reaches | docs/2026-07-02-leakance-diagnosis-findings.md |
| Gauged reaches on eval network | 2,698 | docs/2026-07-02-leakance-diagnosis-findings.md |

---

## Critical invariants that analysis methods must not break

1. **f32 throughout routing core.** No f64 casts in gradcheck test setups.
2. **Lower-triangular adjacency.** `SparseAdjacency::from_dense` in tests
   must have `rows[k] >= cols[k]` — row `i+1` depends on col `i`.
3. **Hand-written sparse backward.** Never replace `CsrSolveOp impl Backward`
   with autograd-tape unrolling in test code (it would pass, but proves nothing
   about the O(nnz) production path).
4. **leakance + use_cuda_graphs:true is REJECTED at config load time.** Test
   configs must always set `use_cuda_graphs: false` when `use_leakance: true`.
5. **compare_ddr_sandbox is the V1 gate.** Run it after every test that touches
   routing. Max abs diff < 1e-3 m³/s.

---

## Provenance and maintenance

All facts verified from source files on 2026-07-05.

```bash
# Re-verify key numbers are still consistent
cargo test --test leakance_gradcheck && echo "gradcheck OK"
cargo test --test leakance_off_parity && echo "parity OK"
cargo test --test zeta_accum && echo "accum OK"
cargo run --release --example compare_ddr_sandbox | grep -E "ABSOLUTE MATCH|max abs"

# Verify the diagnosis script still runs (requires ddr venv + ON run artifacts)
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/leakance_diagnosis.py 2>&1 | grep -E "SUPPORTED|REFUTED|INCONCLUSIVE"

# Verify gradient probe binary exists in the worktree
ls /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/target/release/probe_zeta_gradient
```

Source documents:
- `tests/leakance_gradcheck.rs` — gradcheck recipe and tolerances
- `tests/leakance_off_parity.rs` — parity test fixtures
- `tests/zeta_accum.rs` — accumulator identity tests
- `docs/2026-07-02-leakance-diagnosis-findings.md` — hypothesis battery worked example (H1–H7 verdicts + raw script output)
- `docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md` — battery design template
- `.claude/worktrees/zeta-sensitivity/docs/2026-07-03-zeta-gradient-probe-findings.md` — gradient probe (P1/P2/P3)
- `.claude/worktrees/zeta-sensitivity/docs/2026-07-04-synthetic-recoverability-findings.md` — positive control failure + noise-floor decomposition
- `.claude/references/ddrs-burn-autograd.md` — BURN 0.21 custom Backward recipe
