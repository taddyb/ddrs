---
name: ddrs-hydrology-reference
description: "Use when you need hydrology domain knowledge to work on ddrs: interpreting NSE/KGE metrics, understanding Muskingum-Cunge routing physics, reasoning about the summed-Q baseline, diagnosing leakance behavior, or evaluating equifinality and identifiability claims. Also use when a collaborator asks why routing does not beat the baseline, what zeta represents, or how gauge bias affects training. NOT for Rust/BURN implementation details (use ddrs-burn-autograd or ddrs-algorithm instead) or CLI lifecycle (use ddrs-running-the-code)."
---

# ddrs Hydrology Reference

Domain knowledge pack for a PyTorch-fluent engineer who has NOT worked in hydrology before. Every term is defined once on first use. Commands are copy-pasteable from a ddrs workspace root.

---

## When NOT to Use This Skill

| Question | Use instead |
|---|---|
| How does the BURN autograd tape work? | `ddrs-burn-autograd` |
| How is the sparse solve differentiated? | `ddrs-algorithm` |
| How do I run `ddrs plan` / `ddrs run`? | `ddrs-running-the-code` |
| How are zarr stores read? | `ddrs-reading-inputs` |
| How are eval outputs interpreted? | `ddrs-reading-outputs` |

---

## 1. Glossary — Hydrology Terms Used in ddrs

| Term | One-line definition |
|---|---|
| **Reach** | A single river segment (one row in the adjacency matrix). CONUS has 346,321 reaches. |
| **COMID** | Integer ID for a reach in the MERIT Hydro dataset. Newtype `Comid(i64)` in code. |
| **Gauge / gage** | A physical streamflow measurement station. Training loss is computed only at gauged locations. |
| **Q** | Discharge, m³/s. The quantity routed and observed. |
| **Q'** (Q-prime) | **Lateral inflow** — runoff that enters a reach from its catchment, not from upstream. Produced by an upstream land-surface model (dHBV2, LSTM). This is the forcing signal ddrs receives. |
| **Summed-Q' baseline** | Per-gauge sum of all upstream divide Q', with no routing. The no-routing reference: if the trained model doesn't beat this, the routing isn't helping. |
| **Manning's n** | Channel roughness coefficient. High n = more friction, slower/deeper flow. Learned per-reach by the KAN head. |
| **Leopold-Maddock exponents** (p, q) | Power-law scaling: top width T = p·d^q where d = depth. Describe how channel cross-section changes with discharge. Learned per-reach. |
| **Muskingum-Cunge (MC)** | A linearized flood-routing method. Propagates a flood wave down a network by solving a sparse linear system each timestep. |
| **Hotstart / warm start** | Initializing the routing state (channel storage) from the previous timestep rather than from zero. Cold-start transient = gap between initial-condition and dynamic equilibrium. rho-window = length-of-training-window. warmup = number of timesteps trimmed off the front to let the routing spin up. |
| **NSE** | Nash-Sutcliffe Efficiency. `1 - Σ(obs-sim)² / Σ(obs-mean)²`. Range (-∞, 1], 1=perfect. Penalizes variance error and bias, not amplitude. Maximized at α < 1 (over-attenuated model). |
| **KGE** | Kling-Gupta Efficiency. `1 - sqrt[(r-1)² + (α-1)² + (β-1)²]`. Three components: correlation r, amplitude ratio α=σ_sim/σ_obs, volume bias β=μ_sim/μ_obs. Range (-∞, 1], 1=perfect. Penalizes amplitude loss explicitly. |
| **α (alpha)** | σ_sim / σ_obs. The amplitude ratio in KGE. α < 1 = model over-attenuates peaks. ddrs systematically produces α ≈ 0.85 under L1 loss (baseline α = 0.96). |
| **Equifinality** | Multiple parameter sets produce indistinguishable model outputs. The Beven (2006) critique: you cannot identify unique parameters from output observations alone. |
| **Identifiability** | The degree to which a parameter can be uniquely recovered from observations. A parameter is identifiable if perturbing it produces a detectable change in the loss. |
| **Leakance (zeta)** | GW-SW water exchange term. `zeta = leakance_factor · area_z · K_D · (depth − d_gw)`. Positive = losing stream (water leaves the channel into the aquifer). |
| **K_D** | Streambed hydraulic leakance, 1/s. Analogous to MODFLOW's riverbed conductance K_v/b'. Log-space learnable parameter. |
| **d_gw** | Groundwater depth offset, m. Controls whether a reach is losing (depth > d_gw) or gaining (depth < d_gw). |
| **Disaggregation head** | Neural network that converts daily Q' into a non-flat 24-hour shape, mass-conserving. Fixes the gradient-bottleneck caused by flat `repeat-24` upsampling. |
| **Differential gauging** | Measuring streamflow loss/gain by comparing two gauges bracketing a reach. Detection threshold ≈ 5% of discharge (McCallum 2012). |
| **Non-perennial / ephemeral stream** | A stream that does not flow year-round. Where losing-stream physics is most important. |

---

## 2. The Muskingum-Cunge Routing Equation

MC routes a flood wave down a reach using four coefficients c1..c4 computed from channel geometry and discharge:

```
Q[t+1] = c1·I[t+1] + c2·I[t] + c3·Q[t] + c4·Q'[t]
```

where I = inflow from upstream reaches. Written for every reach simultaneously:

```
(I - C1·N)·Q[t+1] = C2·(N·Q[t]) + C3·Q[t] + C4·Q'[t]
```

N is the adjacency matrix (lower-triangular, topologically ordered = upstream reaches first). The left-hand matrix `A = I - C1·N` is also lower-triangular, so the system is solved by **forward substitution** in O(edges) time — no matrix inversion.

**What is learned:** Manning's n and the Leopold-Maddock exponents (p, q) per reach, emitted by the KAN head from static catchment attributes. These shape the geometry chain: Q → depth → top width → celerity → k (storage time) → c1..c4.

**Why differentiable:** `TimestepOp` (`src/routing/mmc_op.rs`) registers a single `Backward<I,5>` node per timestep with closed-form partials. Tape size is O(saved_state) per step, not O(n²).

**Precision invariant:** f32 throughout. Do not cast to f64 inside the routing chain.

---

## 3. Metrics — NSE, KGE, and the Amplitude Problem

### Definitions

```
NSE = 1 - Σ(obs - sim)² / Σ(obs - mean_obs)²
KGE = 1 - sqrt[(r - 1)² + (α - 1)² + (β - 1)²]
  where r = Pearson corr, α = σ_sim/σ_obs, β = μ_sim/μ_obs
```

### Why L1 (and NSE) reward over-attenuation

L1's optimum and NSE's optimum both sit at `α = r < 1`. The model gets a free pass for shrinking peaks. KGE's `(α-1)²` term actively penalizes amplitude loss.

### Current benchmarks (as of 2026-07-05, CONUS eval, matched gauge set 5,224 gauges)

| Configuration | median NSE | median KGE | α | Notes |
|---|---|---|---|---|
| Summed-Q' baseline | **0.689** | **0.723** | 0.96 | No routing, no learned params |
| L1, fixed X=0.3 | 0.684 | 0.701 | 0.85 | Global run |
| NNSE-KGE loss | 0.684 | 0.699 | — | No improvement over L1 |
| Learnable X | 0.676 | 0.698 | 0.847 | X stuck at init |
| Disagg head (epoch 2 early stop) | 0.681 | 0.671 | — | KGE peak, still below baseline |
| Precip disagg + L1 (2026-06-23) | **0.715** | **0.711** | — | 2365 gauges (CONUS-hourly run) |

**KGE does NOT beat the summed-Q' baseline in any config as of 2026-07-05.** NSE beats it by +0.037 with precip-driven disaggregation. This is the current external-positioning state.

### Interpretation checklist

- [ ] Is α < 0.9? The model is over-attenuating — check training loss curves and gradient flow to n.
- [ ] Is NSE improving while KGE falls? Classic L1/NSE bias: peaks are being flattened.
- [ ] Is training loss flat across all epochs? The gradient is not reaching the routing parameters — check the forcing pipeline (flat repeat-24 vs. disaggregation head).
- [ ] Does the model beat the summed-Q' baseline on NSE? If not, routing is not earning its keep.

---

## 4. The Summed-Q' Baseline

**What it is.** For each gauge, sum all upstream divide Q' (lateral inflows) over the eval window, compare to observed discharge. No routing, no learned parameters — pure forcing.

**Why it matters.** If the trained KAN doesn't beat this, the MC routing is adding noise, not signal. The baseline exercises the data pipeline and the Q' quality, not the routing physics.

**How it is computed.** `ddrs plan` triggers baseline computation on first run (~370 MB of daily Qr, cached under `.ddrs/baselines/<key>/`). `ddrs run --workflow train-and-test` copies it into `<run_dir>/baseline/`.

**The structural ceiling.** As of the 6_19_26 journal: at daily resolution over pre-UH-routed Q', MC routing has no generalizable held-out skill to add beyond the summed-Q' number. To beat it reliably, change the problem: sub-daily observations, less-pre-routed forcing (hillslope runoff instead of UH-routed Q'), or a disaggregation head that is regularized by sub-daily obs.

---

## 5. Training Gradient Bottlenecks (Diagnosed)

Three layers were diagnosed in experiments 0-4 (6_19_26_journal.md):

### Layer 1 — Loss selection (NOT the bottleneck)

L1, NNSE-KGE, and component-KGE all produced flat training loss. The loss function is not the limiter. Switching losses is not productive until the gradient path is unblocked.

### Layer 2 — Flat repeat-24 interpolation (THE gradient bottleneck)

ddrs trains on hourly routing but Q' is produced daily. The pipeline upsampled daily → hourly via flat `repeat-24`, then averaged hourly routed discharge back to daily for the loss. Routing's within-day effect falls entirely in the daily-mean's null space — `∂loss/∂(routing params) ≈ 0`. Evidence: Muskingum X (the main routing parameter) stayed at its sigmoid init (median 0.246) in every run before the disaggregation fix.

**Fix:** Learnable disaggregation head (`src/nn/disagg_head.rs`) converts daily Q' to a non-flat 24-hour shape (mass-conserving). This is the only change that ever made the training loss descend.

### Layer 3 — Structural ceiling (AFTER gradient is unblocked)

Even with a working gradient (disagg head), held-out KGE regressed. The model learned training-period within-day shapes that over-attenuate out-of-sample. This is overfitting on an unsupervised disaggregation signal. To get generalizable skill, the disaggregation needs sub-daily supervision.

---

## 6. Leakance — GW-SW Exchange Term

### What it computes

```
area_z = (p · depth)^q_eps · reach_length        # plan-view wetted area, m²
zeta   = leakance_factor · area_z · K_D · (depth − d_gw)   # m³/s
b_rhs ← b_rhs − zeta                              # positive zeta = losing reach
```

Implementation: `src/routing/leakance.rs`. Gradient via `TimestepLeakanceOp: Backward<I,8>`, gradient-exact against finite differences. Eval-time diagnostic accumulates per-reach mean |zeta| and zeta_net into `<run_dir>/kan_parameters.nc`.

### How to enable (three changes required together)

```yaml
# ddrs.yaml
params:
  use_leakance: true          # also forces use_cuda_graphs: false (config rejects the combo)
  parameter_ranges:
    K_D: [1e-8, 1e-6]         # hydraulic exchange, 1/s (log-space)
    d_gw: [-2, 2]             # GW depth offset, m
    leakance_factor: [0, 1]   # dimensionless scale

kan_head:
  learnable_parameters: [n, q_spatial, p_spatial, K_D, d_gw, leakance_factor]
```

**Hard constraint: `use_leakance: true` + `use_cuda_graphs: true` is rejected at config load.**

### Validation commands

```bash
cargo test --test leakance_gradcheck        # analytical == finite-difference
cargo test --test leakance_off_parity       # byte-identical to no-leakance when OFF
cargo test --test zeta_accum               # zeta diagnostic matches b_rhs subtraction
cargo run --release --example compare_ddr_sandbox  # still ABSOLUTE MATCH
```

### Leakance status (as of 2026-07-05)

**2×2 result (2026-07-01):** GO-marginal. Leakance helps under hourly forcing on losing-stream subset (ΔNSE +0.0005, ΔKGE +0.0018, 55.5% of gauges improve), hurts under daily (ΔNSE −0.0017, ΔKGE −0.0009). |zeta| > 0.01 m³/s on 10.4% of 64,892 eval reaches — barely clears the ≥10% proxy bar.

**Diagnosis (2026-07-02, leakance-diagnosis-findings.md):**

| Hypothesis | Verdict | Key number |
|---|---|---|
| H1: K_D box clips zeta | REFUTED | median utilization 3.4%; 71.5% of reaches CAN exceed 0.01 m³/s inside the current box |
| H2: driving-head starvation | SUPPORTED | median head 0.021 m; 47% of reaches gaining at mean |
| H3: KAN variance collapse | REFUTED | K_D-aridity ρ +0.61, d_gw-meanP ρ +0.71 (strong spatial structure) |
| H4: gauge bias / gradient starvation | SUPPORTED | gauged median |zeta| 11× ungauged; zeta-uparea ρ +0.76; dry/wet ratio 0.40 (inverted from physics) |
| H5: equifinality with routing params | SUPPORTED (daily only) | daily Δn = +0.012 (0.59 IQR); hourly Δn 0.05 IQR (nil) |
| H6: wrong yardstick | REFUTED | fractional loss agrees: 8.4% of reaches lose >1% of local flow |
| H7: model-form error (d_gw pinning) | REFUTED | 0.0% of d_gw at bounds, incl. dry tercile |

**One-line diagnosis:** zeta is small because the optimizer throttles the flux through the driving head (d_gw learned ≈ depth, median head 2 cm), and the gradient only reaches gauged/large-river reaches — not because K_D is boxed.

**Gradient probe (2026-07-03, zeta-gradient-probe-findings.md):**

| Hypothesis | Verdict | Key number |
|---|---|---|
| P1: gradient starvation | REFUTED | gauged/ungauged |∂L/∂factor| ratio = 1.5 (trained), 2.9 (cold) — bar was ≥10 |
| P2: rejection | REFUTED | 52.5% of dry-tercile gradients push zeta down (bar was >67%) |
| P3: detectability | NO-GO | 4.2% of Ref probes detectable at δ=0.01 m³/s — bar was ≥10% |

Root mechanism for P3: a 0.01 m³/s reach loss transmits to its measurement gauge at ~95% fidelity (not lost in routing) but is 53× smaller than the median reference gauge's 5% discharge-uncertainty band. Detection fails on dilution, not transmission.

**Synthetic recoverability (2026-07-04, synthetic-recoverability-findings.md):**

FAILED. R1 median recovery ratio 0.009 vs ≥0.5 bar. Root cause: the windowed training objective (rho-90, warmup-5) has a ~130× hotstart-transient noise floor relative to the planted signal (step-0 loss 1.017 vs continuous residual 0.00759). Even in a noise-free world with warm-started weights and detectable signals, the windowed objective's initial-condition transient buries the planted gradient before it can accumulate.

**Phase B objective (NOT YET MET as of 2026-07-05):** Reduce the noise floor to ≤0.25 mean L1 (≤10% of converged-run loss) via state-cache hotstart. Until this is met, leakance identifiability is NOT proven.

### Physical magnitudes for context (from leakance-litreview.md)

| Streambed regime | K_v/b' leakance (1/s) | ddrs range covers? |
|---|---|---|
| Clogged silt/clay | 1e-7 | Yes (at the floor) |
| Silty sand | 1e-5 | No — 10× above ddrs ceiling |
| Clean sand | 1e-4 | No — 100× above |
| Clean sand, thin bed | 1e-3 to 1e-2 | No — 1000-10000× above |

The current K_D range [1e-8, 1e-6] covers only clogged beds. Sandy/gravel losing reaches sit 2-4 orders above the ceiling. Note: diagnosis (H1 REFUTED) shows the ceiling is not what limits the flux in the current model — the driving head throttle (H2) and gauge-supervision bias (H4) dominate.

---

## 7. Equifinality in ddrs

### The concept

Equifinality (Beven 2006): many parameter sets produce model outputs that are indistinguishable within observational uncertainty. For channel routing, this means multiple (n, p, q) combinations produce similar downstream hydrographs.

### Selective equifinality (the paper's thesis)

Bindas et al. (in prep, `/home/tbindas/projects/ddr_equifinality/paper.tex`): equifinality is **selective**. Channel geometry parameters (p, q → depth, top width at reference discharge) are **identifiable** — physically constrained by geomorphic processes and convergent across different forcing models. Manning's n is a **bias absorber** — it drifts to compensate biases in the upstream Q' model rather than reflecting true channel roughness.

**Evidence in ddrs experiments:**
- Daily leakance ON vs OFF: Manning's n shifts +0.012 (0.59 IQR, ~20% of median) when leakance is added under daily forcing — n absorbs the loss the leakance term would explain (H5 SUPPORTED).
- Under hourly forcing, n does NOT shift (0.05 IQR) when leakance is added — the sub-daily depth signal decouples the two mechanisms.

### Identifiability test protocol (from the paper)

Train 4 models with structurally different lateral inflow sources (two LSTM variants, two dHBV2.0 variants) on the same MERIT network. Compare:
1. Raw learned parameters (p, q, n)
2. Realized channel geometry at reference discharge (depth, top width, hydraulic radius)
3. Routing performance

Parameters that converge across forcing sources are identifiable. Parameters that diverge are compensatory.

---

## 8. Gauged vs. Ungauged Reaches

### Why this matters for training

The training loss is computed only at gauged locations. Gradients flow backward through the routing graph from gauges. A reach with no downstream gauge receives no direct training signal.

### Distribution

In the CONUS eval network (64,892 reaches):
- Gauged: 2,698 reaches (4.2%)
- Ungauged: 62,194 reaches (95.8%)

Gauged reaches are on large, perennial rivers. Losing/ephemeral streams are systematically ungauged (Krabbenhoft et al. 2022 — gauges disproportionately on large rivers; Messager et al. 2021 — >50% of global river length is non-perennial).

### Consequence for leakance

Zeta tracks upstream area (ρ = +0.76) because depth does, and depth is supervised only where gauges exist. Dry-tercile reaches (where physics says losing streams live) have 2.5× LESS zeta than wet-tercile reaches — inverted from physical expectation. The gradient is alive everywhere (gauged/ungauged ratio 1.5–2.9, not 10×), but the training signal only supervises the wrong population.

### Implication

Any gauge-only objective is blind to the 95% ungauged network. Auxiliary spatial constraints (e.g., groundwater-level observations, losing-potential maps from Jasechko 2021) are required to push leakance parameters toward physically plausible values on arid/ephemeral reaches.

---

## 9. Operational Traps

### Trap 1 — Stale binary

```
cargo build           # does NOT update ~/.cargo/bin/ddrs
ddrs run ...          # silently runs the OLD binary
```

After any `src/` change, do ONE of:
```bash
cargo install --path .                              # canonical
# OR
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
# OR bypass entirely:
cargo run --release --bin ddrs -- run --workflow train
```

Self-check: current checkpoints are **directories** (`epoch_E_mb_M/head.mpk`). Flat files (`epoch_E_mb_M.mpk`) mean a stale binary ran.

### Trap 2 — CUDA graphs mask NaN

`use_cuda_graphs: true` captures a CUDA graph on the first forward pass and replays it. If a NaN appears in a later forward, the graph returns the stale finite result from the capture — the NaN is silently dropped and training loss looks normal.

**Detection:**
```bash
# Temporarily set use_cuda_graphs: false in ddrs.yaml and rerun one epoch
# NaN loss will now appear explicitly
```

**Rule:** `use_leakance: true` + `use_cuda_graphs: true` is rejected at config load time. For all other NaN debugging, disable CUDA graphs first.

### Trap 3 — Leakance CUDA graphs incompatibility

Config load rejects `use_leakance: true` + `use_cuda_graphs: true`. The message is explicit. Fix: add `use_cuda_graphs: false` to the config.

### Trap 4 — Warmup too short for recoverability experiments

The windowed training objective (rho-90, warmup-5) has a ~130× hotstart-transient noise floor: big rivers carry routing memory of tens to hundreds of days; 5-day warmup trims far too little. Any experiment that needs to attribute reach-scale flux changes to specific parameters requires either longer warmup, persistent-state training windows, or transient-weighted loss. Phase B (state-cache hotstart) targets ≤0.25 mean L1 noise floor.

---

## 10. Critical Invariants Checklist

Run this before declaring any change correct:

```bash
# V1: port correctness (must always pass)
cargo run --release --example compare_ddr_sandbox
# → "ABSOLUTE MATCH" means max abs diff < 1e-3 m³/s against DDR Python

# V2: f32 precision (no mixed precision in routing core)
# Verify by inspection — any f64 cast in src/routing/ is a bug

# V3: adjacency lower-triangular (topological order)
cargo test data_zarr_store::conus_adjacency_loads_real_merit_zarr

# V4: sparse backward not replaced with tape unrolling
# Verify by inspection — src/sparse.rs CsrSolveOp must have its own impl Backward

# Leakance guard (when touching src/routing/leakance.rs)
cargo test --test leakance_gradcheck
cargo test --test leakance_off_parity
cargo test --test zeta_accum
```

---

## 11. Metric Baseline Reference (verified as of 2026-07-05)

| Reference | Source | NSE | KGE | α |
|---|---|---|---|---|
| Summed-Q' (no routing) | 6_19_26_journal.md | 0.689 | 0.723 | 0.96 |
| Best trained (precip disagg + L1, 2365 gauges) | CLAUDE.md | 0.715 | 0.711 | — |
| Leakance hourly-ON, losing-stream subset ΔKGE | leakance-hourly-findings.md | +0.0005 | +0.0018 | — |

NSE beats the no-routing baseline by +0.037 with precip disaggregation. KGE does not beat the baseline in any configuration. The leakance gain is marginal and on a subset.

---

## 12. Phase B State (as of 2026-07-05)

**Objective:** Reduce windowed training objective noise floor to ≤0.25 mean L1 (≤10% of converged run) via state-cache hotstart.

**Current state:** NOT MET. Measured noise floor ≈130× the planted signal (step-0 loss 1.017 vs continuous residual 0.0076 in recoverability experiment).

**Implication for any leakance identifiability claim:** Until Phase B is met, leakance is NOT proven identifiable via gauge-only training. The recoverability positive control FAILED with recovery ratio 0.009 vs the ≥0.5 bar. Any paper or status update must state this plainly.

**Phase B design:** `docs/superpowers/plans/2026-07-04-phase-b-floor-fix.md` (in worktree `zeta-sensitivity`). State-cache: run teacher-weights forward on the full historical window, cache per-reach routing states, use cached states as hotstart for training windows instead of heuristic initial conditions.

---

## Provenance and Maintenance

All facts in this skill were verified from these files on 2026-07-05:

```bash
# Verify metric numbers
cat /home/tbindas/projects/ddrs/docs/6_19_26_journal.md          # experiment 0-4 table
cat /home/tbindas/projects/ddrs/CLAUDE.md                         # best result, leakance status

# Verify leakance diagnosis verdicts
cat /home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md   # §3 results table

# Verify gradient probe verdicts
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-03-zeta-gradient-probe-findings.md  # §4 table

# Verify recoverability FAIL
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-04-synthetic-recoverability-findings.md  # §4 table

# Verify physical magnitudes
cat /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-litreview.md    # §A1 table

# Verify equifinality paper title/thesis
head -40 /home/tbindas/projects/ddr_equifinality/paper.tex

# Verify algorithm math
cat /home/tbindas/projects/ddrs/.claude/references/ddrs-algorithm.md
```

Update this skill whenever: a new baseline beat is established, leakance Phase B is completed, or a new experiment changes a SUPPORTED/REFUTED/NO-GO verdict.
