---
name: ddrs-external-positioning
description: "Use when writing about ddrs for an external audience — paper drafts, grant proposals, README summaries, reviewer responses, or conference abstracts. Trigger on: claims about novelty vs prior work, metric comparisons to baselines or operational routers, statements about what is proven vs hypothesized, reproducibility assertions, or any sentence that positions ddrs against DDR (Python), NWM, or other routing frameworks. Do NOT use for internal debugging, training-run operations, or config changes."
---

# ddrs external positioning

This skill is a reference for any writing that makes claims about ddrs to an
external reader. Every fact below was verified from source documents as of
2026-07-05. Volatile metrics are stamped with that date. Use
SUPPORTED/REFUTED/INCONCLUSIVE for hypothesis verdicts, never "likely" or
"suggests" when a pre-registered test has run.

---

## When NOT to use this skill

| Situation | Use instead |
|---|---|
| You are debugging a training run or a broken binary | `ddrs-debugging-playbook` |
| You are writing or editing config YAML | `ddrs-config-and-flags` |
| You need to run an evaluation or inspect a checkpoint | `ddrs-run-and-operate` |
| You want to generate plots for a run | `ddrs-eval-plots` |
| You are planning the next experiment | `ddrs-research-frontier` |
| You want to understand the identifiability campaign arc | `ddrs-identifiability-campaign` |

---

## 1. What ddrs is

**ddrs** is a Rust port of **DDR** (Differentiable Discharge Routing), a
Python/PyTorch differentiable Muskingum-Cunge river routing model. The port
uses the **BURN 0.21** deep-learning framework and targets CONUS-scale
(346,321 reaches, 338,814 edges) on a single consumer GPU.

### Relationship to DDR

| Property | DDR (Python) | ddrs (Rust) |
|---|---|---|
| Language / framework | Python + PyTorch | Rust + BURN 0.21 |
| Routing algorithm | Muskingum-Cunge | Muskingum-Cunge (exact port) |
| Precision | f32 | f32 throughout routing core (invariant) |
| Parameter head | KAN (PyTorch `kan.py`) | `rskan::KanLayer` v0.1.3 |
| Sparse backward | PyTorch autograd | Hand-written O(nnz) custom Backward |
| Parity status | reference | port; must match DDR to < 1e-3 m³/s max abs diff |

The parity gate is:

```bash
cargo run --release --example compare_ddr_sandbox
# must print "ABSOLUTE MATCH" (max abs diff < 1e-3 m³/s, 5-reach RAPID sandbox)
```

This gate must pass after every change to `src/routing/`, `src/geometry.rs`,
or `src/sparse.rs`.

### Relationship to operational routers

| System | Parameterization | Gradient-based calibration | Scale |
|---|---|---|---|
| NWM (CONUS) | Lookup tables / power-law scaling | No | 3.6M reaches |
| DDR / ddrs | KAN head learns p, q, n from obs | Yes | 346,321 reaches (CONUS) |
| Global ddrs | Same head, 2.94M-reach MERIT fabric | Yes (planned/partial) | Global |

The primary novelty claim: **observation-constrained channel-parameter learning
at continental scale via differentiable routing**. This is shared with DDR; ddrs
adds the Rust/BURN implementation and infrastructure experiments described below.

---

## 2. Critical invariants — never violate these in writing or code

1. **Gradient exactness vs DDR.** The `compare_ddr_sandbox` example is the
   reproducibility anchor. Do not claim gradient-exactness without running it.
2. **f32 throughout the routing core.** No mixed precision. Any cast to f64/bf16
   breaks DDR parity.
3. **Adjacency is lower-triangular.** Topological order is a prerequisite of the
   forward-substitution solver. This is tested in `data_zarr_store::conus_adjacency_loads_real_merit_zarr`.
4. **Hand-written sparse backward in `src/sparse.rs`.** Do NOT replace it with
   autograd tape unrolling. The point is O(nnz) tape entries per timestep, not
   O(n²). This is the performance-critical invariant for large-scale training.
5. **KAN head is `rskan::KanLayer` v0.1.3.** No inter-block ReLU. Architecture
   matches DDR's `kan.py` exactly.

---

## 3. Skill metrics table — as of 2026-07-05

Every number here comes from a named run with a named eval window. Do not
extrapolate to other configs unless the finding document confirms it.

### 3.1 Core skill results (CONUS, 2365 gauges, eval 1995-10-01 to 2010-09-30)

| Config | Median NSE | Median KGE | Source |
|---|---|---|---|
| **Summed-Q' baseline (no routing)** | **0.678** | **0.717** | `2026-06-23T02-49-12Z-conus-hourly-train-and-test`, same-run baseline |
| Precip-disagg OFF, L1 (`daily-OFF`) | 0.700 | 0.724 | `2026-06-05T01-41-16Z-train-and-test` |
| Precip-disagg only (no precip signal), L1 | 0.696 | 0.693 | `2026-06-23T13-03-35Z-train-and-test` |
| **Precip-driven disagg, L1** | **0.715** | **0.711** | `2026-06-23T02-49-12Z-conus-hourly-train-and-test` |
| Precip-driven disagg, nnse-kge loss | 0.710 | 0.710 | `2026-06-24T00-03-01Z-conus-hourly-train-and-test` |
| Precip + temp, L1 | 0.716 | 0.709 | `2026-06-24T02-10-49Z-conus-hourly-train-and-test` |

**Critical external claim:** KGE does NOT beat the summed-Q' baseline in any
config as of 2026-07-05. NSE beats baseline by +0.037 (best config: precip-driven
disagg + L1). State this plainly; do not soften it to "approaches" or "narrows
the gap".

### 3.2 What the precip contribution is (as of 2026-07-05)

Decomposition from the precip-ON vs precip-OFF paired comparison (same seed,
same gauge batches):

| Effect | Delta NSE | Delta KGE |
|---|---|---|
| Real precip timing (ON minus OFF) | +0.020 | +0.018 |
| Bare disaggregation vs baseline (OFF minus base) | +0.018 | -0.025 |
| Net vs baseline (ON minus base) | +0.037 | -0.007 |

The bare disaggregation (invented within-day shape) trades KGE for NSE. Real
precip rescues the KGE the bare disaggregation destroys. The residual -0.007
KGE gap relative to baseline is structural: the summed-Q' reference has the
highest KGE of any config because routing already-smoothed Q' cannot beat the
no-routing baseline's variance ratio. This has been confirmed by trying the
nnse-kge α-restoring loss (finding: KGE moved essentially nowhere; the L1
over-attenuation hypothesis was REFUTED by the nnse-kge experiment).

### 3.3 Leakance (GW-SW exchange term) — status as of 2026-07-05

**What it is.** An optional losing-stream correction:
```
zeta = leakance_factor * area_z * K_D * (depth - d_gw)
```
subtracted from the routing RHS at each timestep. Off by default. Enabled via
`params.use_leakance: true` + three learnable params (K_D, d_gw, leakance_factor).
Incompatible with `use_cuda_graphs: true` (rejected at config load time).

**2x2 verdict (as of 2026-07-01).** GO-marginal. All three gate criteria met:
- Skill gain on losing-stream subset under hourly forcing: DELTA NSE +0.0005, DELTA KGE +0.0018, 55.5% of gauges improve.
- Effect absent/weaker under daily: daily DELTA NSE -0.0017, DELTA KGE -0.0009 (actively negative).
- |zeta| > 0.01 m³/s on 10.4% of eval reaches (64,892 reaches, eval window 1995-2010). Passes the >= 10% proxy bar with no headroom.

**Leakance diagnosis (as of 2026-07-02).** Seven pre-registered hypotheses on why learned zeta is small:

| Hypothesis | Verdict |
|---|---|
| H1: K_D box clips flux | REFUTED — 71.5% of reaches CAN exceed 0.01 m³/s inside current box; median utilization 3.4% |
| H2: driving-head starvation (d_gw near depth) | SUPPORTED — median head 0.021 m; 47.0% of reaches gaining at mean |
| H3: KAN variance collapse | REFUTED — K_D-aridity Spearman +0.61, d_gw-meanP +0.71 |
| H4: gauge bias / gradient starvation | SUPPORTED — gauged median |zeta| 11x ungauged; dry/wet ratio 0.40 (physics says dry >> wet) |
| H5: equifinality (n absorbs attenuation) | SUPPORTED (daily only) — daily Delta-n +0.012 (~20% shift); hourly Delta-n nil |
| H6: wrong yardstick (absolute bar) | REFUTED — fractional loss agrees: 8.4% lose >1% local flow |
| H7: model-form error (d_gw boundary-pinning) | REFUTED — 0.0% of d_gw at bounds |

K_D-ceiling widening is NOT recommended. The diagnosis showed K_D is not the
binding constraint; widening it would re-pin at the new ceiling with little
change in flux or skill.

**Gradient probe (as of 2026-07-03, worktree `origin/worktree-zeta-sensitivity`).**
Three pre-registered hypotheses on mechanism:

| Hypothesis | Verdict | Key number |
|---|---|---|
| P1: gradient starvation (dead off-gauge) | REFUTED | gauged/ungauged |g| ratio only 1.5-2.9x (bar >= 10x) |
| P2: rejection (gradient pushes zeta down) | REFUTED (trained point) | 52.5% dry-tercile push-down (neutral; bar > 67%) |
| P3: detectability (real-magnitude loss visible at gauge) | NO-GO | 4.2% of Ref probes detectable at delta=0.01 m³/s; median Ref gauge 5%-band = 0.53 m³/s (53x the planted flux) |

P3 NO-GO is a hard constraint: a 0.01 m³/s reach loss transmits to its
measurement gauge at ~95% fidelity but is 53x smaller than the median reference
gauge's 5% discharge-uncertainty band. No objective computed from gauged
discharge alone can reward what it cannot distinguish from measurement
uncertainty.

**Synthetic recoverability positive control (as of 2026-07-04, worktree
`origin/worktree-zeta-sensitivity`).** FAILED. Planted signal was never visible
to the windowed training objective.

Pre-registered bars and outcomes:

| Metric | Bar | Measured | Verdict |
|---|---|---|---|
| R1: recovery ratio median | >= 0.5 | 0.009 (p10=-0.073, p90=0.199) | FAILED |
| R2: non-planted |zeta_net| ratio | < 2x | 1.11 | PRECISE (but trivially — nothing moved) |
| R3: final-epoch loss A vs B | A < B by > 5% | A=1.34 vs B=2.32 (+42%) | A < B but CONFOUNDED (see below) |
| R4: absorption map | descriptive | median Delta-n planted = -0.019 | — |
| R5: cold emergence ratio | > 3x | 1.20 | SUPPRESSED |

Root cause: the windowed training objective (rho-90, warmup-5) has a
hotstart-transient noise floor ~130x larger than the planted signal:
- Continuous residual (teacher weights on teacher obs, full window eval): 0.0076 mean L1
- Step-0 windowed training loss (student A): 1.017
- Ratio: ~130x

The planted signal (0.0076 mean L1) is 0.8% of the training loss. Adam cannot
see it. R3's 42% gap is confounded: it reflects the loss of the leakance BASE
FIELD (aggregate effect across thousands of upstream reaches), not recovery of
individual planted fluxes.

**Implication for external claims:** leakance identifiability is NOT proven as
of 2026-07-05. Do not claim it. The required precondition for any identifiability
claim is Phase B: state-cache hotstart that brings the training noise floor to
<= 0.25 mean L1 (<= ~10% of converged run). Phase B is not yet complete.

---

## 4. What is novel vs known — checklist

Before making a novelty claim, verify it against this table.

| Claim | Status | Condition on claim |
|---|---|---|
| Differentiable MC routing with KAN head at CONUS scale | SUPPORTED (in DDR; ddrs is the Rust port) | Attribute to DDR + Bindas et al. |
| Hand-written O(nnz) sparse backward in BURN | NOVEL in ddrs | Citable as ddrs implementation contribution |
| NSE beats summed-Q' baseline (+0.037) | SUPPORTED (precip-disagg + L1, 2365 gauges) | State config and date; do not round up |
| KGE beats summed-Q' baseline | NOT SUPPORTED in any config as of 2026-07-05 | Do not claim |
| Precip-driven disaggregation improves both NSE and KGE vs disagg-only | SUPPORTED | Delta NSE +0.020, Delta KGE +0.018 vs disagg-OFF |
| Leakance term identifiable (non-collapsed) | SUPPORTED (K_D != floor; parameters carry attribute structure) | Qualified: term is identifiable in the sense that K_D is not collapsed to floor; flux magnitudes are small and unlearnable by gauge-only supervision |
| Leakance flux recoverable from gauged discharge | NOT PROVEN (positive control FAILED) | Requires Phase B before this claim can be made |
| Channel geometry (p, q) identifiable across inflow sources | HYPOTHESIS (paper `ddr_equifinality/paper.tex`; not yet a run result) | Do not cite as result; cite as planned experiment |
| Manning's n is a bias absorber, not a physical truth | HYPOTHESIS (same paper) | Do not cite as result |
| Gradient reaches ungauged reaches | SUPPORTED (P1 REFUTED: ratio only 1.5-2.9x, not >= 10x) | Reframe: gradient is alive everywhere but signal at the gauge is too small to differentiate real-magnitude leakance |

---

## 5. Reproducibility standards

Any result cited externally must meet ALL of the following:

- [ ] Named run ID (format: `YYYY-MM-DDTHH-MM-SSZ-[group-]workflow`)
- [ ] Eval window stated (default: 1995-10-01 to 2010-09-30)
- [ ] Number of finite-NSE gauges stated (default CONUS set: 2365)
- [ ] Metric code is the same for all compared configs (same summed-Q' baseline, same eval window)
- [ ] Binary provenance confirmed: NOT a stale installed binary (see stale-binary trap below)
- [ ] `compare_ddr_sandbox` ABSOLUTE MATCH confirmed for the commit used

### Stale-binary trap

`cargo build` does NOT update `~/.cargo/bin/ddrs`. The installed binary is what
runs when you type `ddrs` at the shell. After any `src/` change:

```bash
cargo install --path .
# or, faster if target/release is current:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
```

Quick self-check: current checkpoints are DIRECTORIES
(`.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/head.mpk`). A stale pre-checkpoint-resume
binary writes flat files (`epoch_E_mb_M.mpk`). If you see flat files, you ran an
old binary and your results are invalid.

The 2026-07-01 leakance 2x2 was initially invalidated by exactly this: the
installed binary was from before the disaggregation feature, so both the hourly
and daily cells ran flat-repeat-24 and came out byte-identical (false "disagg no-op").

### CUDA graphs mask NaN

`use_cuda_graphs: true` returns a stale finite loss when the forward produces
NaN. Symptoms: printed training loss looks finite; checkpoint weights are NaN
from an early mini-batch; eval produces 0/N finite NSE. Always validate with
`use_cuda_graphs: false` before treating a loss curve as trustworthy. This is
how the AORC NaN bug (real NaN in AORC `total_precipitation` arrays, ~14% of
values) went undetected for a full run.

---

## 6. ddrs vs DDR — what to say when asked

DDR is the authoritative Python/PyTorch reference implementation. ddrs is a
gradient-exact port. The relationship is:

- DDR is cited for the algorithm (Muskingum-Cunge as differentiable graph
  message-passing, KAN parameterization, CONUS training setup).
- ddrs is cited for the Rust/BURN implementation, the infrastructure choices
  (BURN 0.21, O(nnz) backward, the CLI lifecycle, global-scale data readers),
  and any experiments run exclusively in ddrs (leakance 2x2, gradient probe,
  recoverability control).
- If a result was produced by ddrs code but the algorithm is identical to DDR,
  say "using the ddrs implementation of DDR" or similar.
- The `ddr_equifinality/paper.tex` paper ("Beyond Equifinality in Differentiable
  River Routing", Bindas & Shen) is in draft. Its central hypothesis ("selective
  equifinality: geometry identifiable, Manning's n is a bias absorber") is
  NOT yet a result — the four-inflow-source experiment has not been run.

---

## 7. Key external claims and how to hedge them

### 7.1 Skill gain from routing

Correct framing: "differentiable MC routing with precip-driven disaggregation
improves median NSE by +0.037 over the no-routing summed-Q' baseline (0.715 vs
0.678) on 2365 CONUS gauges; median KGE does not beat the no-routing baseline
(-0.007, 0.711 vs 0.717)."

Do not write: "ddrs achieves state-of-the-art performance on CONUS routing."
The summed-Q' baseline is a strong prior and KGE does not beat it.

### 7.2 Leakance

Correct framing: "A GW-SW exchange term (leakance) benefits losing-stream gauge
skill under hourly forcing (Delta KGE +0.0018, 55.5% of gauges improve) but
degrades skill under daily forcing. Identifiability analysis (seven
pre-registered hypotheses) shows the small learned flux magnitude is a
training-signal problem, not a parameter-box or architecture problem: real-world
leakance magnitudes are below the observational detectability band at 95.8% of
reference gauges, and the windowed training objective carries a ~130x noise floor
relative to the planted signal in synthetic recoverability tests. Leakance
remains experimental and hourly-gated; no claim of identifiability from
gauge-supervised training alone is made."

### 7.3 Equifinality

Correct framing: "We hypothesize that equifinality is selective in differentiable
routing: channel geometry parameters are identifiable from discharge observations
while Manning's roughness absorbs systematic biases from upstream inflow models.
Testing this hypothesis requires training on four structurally different lateral
inflow sources and comparing parameter convergence; this experiment is in
preparation."

Do not write: "We show that channel geometry is identifiable." The four-inflow
experiment has not been run.

---

## 8. Architecture facts for methods sections

- **Network:** 346,321 MERIT-Hydro reaches, 338,814 edges (lower-triangular CSR).
- **Parameterization head:** KAN (Kolmogorov-Arnold Network via `rskan::KanLayer`): `Linear(F, H) -> KanLayer(H, H) x num_hidden_layers -> Linear(H, P) -> Sigmoid`.
- **Learned parameters (base):** Manning's roughness `n`, Leopold-Maddock exponents `p` (top width) and `q_spatial` (depth), Muskingum storage coefficient `x_storage`.
- **Optional learnable (leakance):** K_D (hydraulic exchange rate, 1/s), d_gw (groundwater depth offset, m), leakance_factor (dimensionless).
- **Precision:** f32 throughout.
- **Solver:** forward substitution, O(|E|) per timestep, custom BURN Backward in `src/sparse.rs`.
- **Forcing:** lateral inflow Q' from dHBV2.0-UH (CONUS MERIT), optionally disaggregated from daily to hourly via precip-driven head.
- **Objective:** L1 (default) or nnse-kge (selectable via config).

---

## 9. Paper-in-progress: Beyond Equifinality

File: `/home/tbindas/projects/ddr_equifinality/paper.tex`

Title: "Beyond Equifinality in Differentiable River Routing"
Authors: Tadd Bindas, Chaopeng Shen

Central claim: selective equifinality — geometry identifiable, Manning's n is a
bias absorber that shifts peak timing to compensate for lateral inflow errors.

Status (2026-07-05): abstract + introduction + methods section drafted. Results
section is a placeholder — the four-inflow experiment (two LSTM variants, two
dHBV2.0 variants, same MERIT network) is not yet run.

What is safe to cite from this paper draft:
- The conceptual framing (selective equifinality, input-perturbation
  identifiability test protocol).
- The MC-as-graph-message-passing formulation.
- The leakance term definition (equation in §Methods).

What is NOT safe to cite as a result from this paper:
- Any number from the results section (none exist yet).
- The claim that geometry converges across inflow sources.
- The claim that n diverges as a function of inflow disagreement.

---

## Provenance and maintenance

Re-verify core skill numbers after any new CONUS train-and-test run:

```bash
# Check summed-Q' baseline for the run you care about
cat .ddrs/runs/<id>/baseline/manifest.json | python3 -m json.tool | grep -E "nse|kge"

# Confirm DDR parity is intact on current commit
cargo run --release --example compare_ddr_sandbox 2>&1 | grep -E "ABSOLUTE|max abs"

# Confirm KAN head tests pass
cargo test --features fixtures --test kan_head_init_repro --test kan_head_init_parity 2>&1 | tail -5
```

Finding documents referenced in this skill (all in `/home/tbindas/projects/ddrs/docs/` unless noted):
- `2026-06-23-precip-disaggregation-findings.md` — precip disagg 4-way table + precip contribution decomposition
- `2026-07-01-leakance-hourly-findings.md` — leakance 2x2 GO-marginal verdict + zeta gate numbers
- `2026-07-02-leakance-diagnosis-findings.md` — H1-H7 verdict table + raw script output
- `docs/2026-07-03-zeta-gradient-probe-findings.md` (worktree `origin/worktree-zeta-sensitivity`) — P1/P2/P3 verdicts + detectability decomposition
- `docs/2026-07-04-synthetic-recoverability-findings.md` (worktree `origin/worktree-zeta-sensitivity`) — R1-R5 verdicts + 130x noise floor finding
- `/home/tbindas/projects/ddr_equifinality/paper.tex` — selective equifinality paper draft
