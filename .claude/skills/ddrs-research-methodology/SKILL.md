---
name: ddrs-research-methodology
description: >
  Use when designing a new ddrs experiment, deciding whether to run a retrain,
  writing a hypothesis-verdict report, evaluating a gate criterion, or debugging
  why a previous experiment produced a null or unexpected result. Also use when
  onboarding to the ddrs research arc and needing to understand the evidence
  standard, the experiment lifecycle, and what has already been proven or
  refuted. Do NOT use for implementation questions about the Rust routing core,
  BURN autograd, or the KAN head — those are covered by the architecture and
  burn-autograd reference files.
---

# ddrs Research Methodology

A runbook for turning a physical hunch into a defensible experiment result in
the `ddrs` differentiable routing system. Covers evidence standards, hypothesis
lifecycle, the idea-to-experiment pipeline, gate discipline, CPU-first
validation, and a datestamped arc of every hypothesis tried as of 2026-07-05.

---

## Table of contents

1. Glossary
2. When NOT to use this skill
3. Evidence standard
4. Hypothesis lifecycle
5. Idea-to-experiment pipeline
6. Gate discipline
7. CPU-first principle
8. Critical operational traps
9. The research arc (chronological, with verdicts)
10. Current state and open questions (as of 2026-07-05)
11. Provenance and maintenance

---

## 1. Glossary

| Term | Definition |
|---|---|
| **ddrs** | BURN-0.21 Rust port of DDR (Python/PyTorch). Implements differentiable Muskingum-Cunge (MC) channel routing with a KAN parameter head. |
| **DDR** | Python/PyTorch reference implementation at `~/projects/ddr/`. ddrs must remain gradient-exact against it. |
| **MC routing** | Muskingum-Cunge: a linearised shallow-water channel model that routes a hydrograph downstream using per-reach parameters (Manning's n, geometry). |
| **KAN head** | Kolmogorov-Arnold Network (`rskan::KanLayer` v0.1.3) that maps per-reach catchment attributes to routing parameters. |
| **Q' (q-prime)** | Upstream-routed unit-hydrograph streamflow forcing, in m³/s. The model inputs come pre-routed; ddrs adds the channel-routing correction. |
| **Summed-Q' baseline** | Per-gauge sum of upstream Q' predictions with zero routing. The no-routing reference. Median NSE 0.689 / KGE 0.723 on 5,224 CONUS gauges. |
| **KGE** | Kling-Gupta Efficiency: 1 - sqrt((r-1)² + (α-1)² + (β-1)²) where r=correlation, α=σ_sim/σ_obs, β=μ_sim/μ_obs. |
| **NSE** | Nash-Sutcliffe Efficiency: 1 - SS_res/SS_tot. |
| **α** (alpha) | KGE variability ratio σ_sim/σ_obs. α < 1 means over-attenuation. |
| **Leakance (zeta)** | Experimental GW-SW water-loss term: `zeta = leakance_factor * area_z * K_D * (depth - d_gw)`. Subtracted from the routing RHS b at each timestep. |
| **zeta** | Mean |leakance flux| per eval reach over the eval window (m³/s). |
| **zeta_net** | Signed mean leakance per reach (positive = net-losing). |
| **rho window** | Training subsequence length (default 90 days). |
| **warmup** | Days at the start of each rho window discarded from loss to reduce hotstart-transient noise (default 5 days). |
| **COMID** | NHD reach identifier. ddrs newtype: `Comid(i64)`. |
| **STAID** | USGS/global gauge identifier. ddrs newtype: `Staid(String)`. |
| **eval network** | The gauge-subgraph union used at eval time (~64,892 CONUS reaches). Smaller than full CONUS (346,321 reaches). |
| **Phase B** | Planned next phase of leakance identifiability work: state-cache hotstart to reduce windowed training noise floor to <=0.25 mean L1. NOT yet run as of 2026-07-05. |

---

## 2. When NOT to use this skill

| Task | Use instead |
|---|---|
| Editing `src/routing/`, `src/sparse.rs` | `.claude/ARCHITECTURE.md` + `.claude/references/ddrs-burn-autograd.md` |
| Debugging BURN autograd / sparse backward | `.claude/references/ddrs-burn-autograd.md` |
| Porting a DDR algorithm | `~/projects/ddr/CLAUDE.md` + cite DDR line numbers |
| CLI lifecycle / `ddrs plan` / checkpoints | `CLAUDE.md` §`ddrs` CLI |
| Data source formats / zarr layout | `CLAUDE.md` §Data sources |
| Adding a new learnable parameter | `CLAUDE.md` §Leakance (for the config pattern) |

---

## 3. Evidence standard

Every hypothesis in ddrs must be:

- **Pre-registered** before running GPU-expensive code. State the test, the
  falsification criterion, and the pre-registered bar in the spec doc before
  touching a training run.
- **Falsifiable** by a cheap instrument: parameter statistics, gradient probes,
  CPU forward-only passes, or closed-form bounds. Avoid "let's retrain and see."
- **Verdicted** as SUPPORTED / REFUTED / INCONCLUSIVE with an effect size.
  "The numbers look okay" is not a verdict.

### Metric hierarchy

```
ABSOLUTE MATCH (max abs diff < 1e-3 m³/s)   ← DDR parity; must never break
  |
  v
Beats summed-Q' baseline (NSE > 0.689)       ← routing must earn its keep
  |
  v
Beats summed-Q' baseline on KGE (> 0.723)    ← NOT yet achieved in any config
  |
  v
Experimental feature GO gate                 ← hypothesis-specific; see §6
```

**As of 2026-07-05:**
- Best NSE: 0.715 (precip-driven disagg + L1, 2365 gauges, run `2026-06-23T02-49-12Z`). Beats baseline by +0.026.
- Best KGE: 0.711 (same run). Does NOT beat the 0.723 baseline. No config has beaten the KGE baseline.
- Leakance under hourly forcing lifts KGE on the losing-stream subset (+0.0018) but does not lift all-gauge KGE above the baseline.

---

## 4. Hypothesis lifecycle

```
 idea
   |
   v
[1] SPEC DOC  ─── pre-register hypotheses with falsification bars
   |               (docs/superpowers/specs/YYYY-MM-DD-<name>-design.md)
   v
[2] CHEAP FIRST ── gradient probe / parameter stats / CPU closed-form bounds
   |               (test with no-train instruments before spending GPU)
   v
[3] GATE CHECK ─── does the evidence warrant a retrain? (see §6)
   |               If NO → write NO-GO findings doc and stop
   v
[4] RETRAIN ─────── one GPU run per hypothesis; minimal config changes
   |               (record binary version; see STALE-BINARY TRAP §8)
   v
[5] FINDINGS DOC ── verdict per hypothesis with raw script output
   |               (docs/YYYY-MM-DD-<name>-findings.md)
   v
[6] NEXT STEPS ─── ranked by evidence priority, not researcher preference
```

### Verdict vocabulary

| Verdict | Meaning |
|---|---|
| SUPPORTED | Evidence exceeds the pre-registered bar in the predicted direction |
| REFUTED | Evidence exceeds the pre-registered bar in the opposite direction |
| INCONCLUSIVE | Evidence does not clear the bar in either direction |
| NO-GO | Evidence closes the path: the proposed remedy cannot work as stated |
| GO-MARGINAL | All gate criteria met with little headroom |

---

## 5. Idea-to-experiment pipeline

### Step 0 — sanity check (before writing a spec)

Ask three questions. If any is "no," rethink the idea before writing a spec.

- [ ] Is the effect physically plausible at the spatial/temporal scale of ddrs?
- [ ] Can the gradient path reach the proposed mechanism? (Run a gradient probe if uncertain.)
- [ ] Is the proposed intervention expressible given the current parameter ranges?

### Step 1 — write the spec

Create `docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md` with:

```
## Problem
[One paragraph: what observation is unexplained?]

## Hypotheses and tests
| # | Hypothesis | Test | Falsified if |
|---|---|---|---|

## Phase 1 — cheap instruments (no retrain)
## Phase 2 — if gate opens: retrain plan
## Gate: retrain only if [condition on cheap instrument results]
## Deliverables
## Concerns / assumptions
```

### Step 2 — run cheap instruments first

| Instrument | Cost | When to use |
|---|---|---|
| `scripts/leakance_diagnosis.py` | ~2 min (Python) | Parameter statistics, spatial structure, equifinality checks |
| `probe_zeta_gradient --mode grad` | ~35 min (CPU) | Gradient magnitude / sign by stratum |
| `probe_zeta_gradient --mode perturb` | ~overnight (CPU) | Detectability of a planted signal at gauges |
| Closed-form bounds | 0 | Check whether a parameter box can produce the required magnitude |
| `dump_parameters` | ~5 min | Parameter distributions over full CONUS |
| `eval --zeta-output` | ~10 min | Per-reach zeta means on existing checkpoint |

### Step 3 — write the findings doc

Create `docs/YYYY-MM-DD-<slug>-findings.md`. Mandatory sections:

1. **One-line answer** at the top.
2. **Methods** — what was changed, what was not, what guards stayed green.
3. **Results table** with verdict and key number per hypothesis.
4. **Conclusions** numbered, with explicit supersession of outdated recommendations.
5. **Next steps** ranked.
6. **Raw script output** (copy-paste, not paraphrase).

---

## 6. Gate discipline

Gates exist to avoid spending GPU when the gradient path is demonstrably broken.
A gate condition must be pre-registered in the spec. If the gate fails, document
the failure, state what it implies, and do NOT run the retrain.

### The three-criteria GO gate (leakance-class experiments)

All three must be met:

| Criterion | Bar |
|---|---|
| Skill gain on target subset (ΔNSE or ΔKGE > 0) | Positive delta on the target condition |
| Effect absent or weaker under the control condition | Negative or null delta under control |
| Physical magnitude above detectability proxy | |zeta| > 0.01 m³/s on ≥10% of eval reaches |

### Pre-registering bars

Bad: "we'll see if the metrics improve."
Good: "SUPPORTED if gauged/ungauged |gradient| ratio < 10 at both trained and cold points."

The bar must be expressible as a single boolean on a number you can compute
before running the test.

### When to skip the gate and write a NO-GO directly

Write a NO-GO findings doc (no retrain) when cheap instruments show:
- The gradient is dead or inverted everywhere the effect needs to operate.
- The proposed parameter range cannot produce the required magnitude (H1-style closed-form refutation).
- A previously accepted recommendation is superseded by new evidence (explicitly state the supersession).

**Example (2026-07-02):** The diagnosis spec had a Phase-3 gate requiring H1 SUPPORTED
(K_D box caps zeta) AND gradient alive (H3/H4 not showing total collapse). H1 was
REFUTED (median utilization 3.4% — the box is not the cap). Gate failed. No K_D
widening retrain was run. The "widen K_D" recommendation from the prior findings
doc was explicitly superseded.

---

## 7. CPU-first principle

**Always prototype and validate on CPU before any GPU retrain.**

Rationale:
- GPU runs are expensive (hours to days) and the GPU is a shared resource.
- CPU (`NdArray<f32>`) is deterministic: identical seeds produce byte-identical
  results. This turns noise-floor measurement into a hard equality check.
- A subtle gradient or data-alignment bug burns the same GPU time whether you
  find it on CPU or not.

### CPU-first checklist

- [ ] Run `--backend cpu` first for any new binary or config path.
- [ ] Confirm DDR parity: `cargo run --release --example compare_ddr_sandbox` → ABSOLUTE MATCH.
- [ ] Run the test suite: `cargo test --test leakance_gradcheck --test leakance_off_parity --test zeta_accum`.
- [ ] Validate alignment (lag test): compute lag-0 vs lag-±1 mean L1; a ±1-day lag should be ~10× worse.
- [ ] Confirm baselines are deterministic: run two identical forward passes and check `max|b1−b2| = 0`.

### When CPU is not enough

CPU runs are mandatory for validation; they are not required to reach final
metrics (GPU scatter-add is non-deterministic but the magnitude is known and
accounted for in final metric comparisons).

---

## 8. Critical operational traps

### Trap 1 — STALE-BINARY TRAP (most common failure mode)

`cargo build` and `cargo run` do NOT update `~/.cargo/bin/ddrs`. If you edit
`src/` and then run `ddrs run …`, you silently run the old binary.

**Detection:** current binaries write checkpoints as DIRECTORIES
(`epoch_E_mb_M/head.mpk`). Flat files (`epoch_E_mb_M.mpk`) mean you ran a
pre-checkpoint-resume binary.

**Fix (always one of these after a `src/` change):**

```bash
cargo install --path .                          # canonical; updates ~/.cargo/bin/ddrs
# faster if target/release is current:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
# bypass installed copy entirely:
cargo run --release --bin ddrs -- run --workflow train-and-test
```

This trap invalidated the 2026-07-01 2×2 morning runs (stale June-3 binary;
both cells byte-identical; required full re-runs with valid binary).

### Trap 2 — CUDA graphs mask NaN

`use_cuda_graphs: true` returns stale finite loss on a NaN forward pass. The
bug is silent — loss looks normal, metrics are wrong.

**Rule:** always validate a new config with `use_cuda_graphs: false` first.
Also: leakance + `use_cuda_graphs: true` is rejected at config-load time
(the combination is blocked by design).

### Trap 3 — Windowed training has a ~130× hotstart-transient noise floor

The rho-window (default 90 days) + warmup (default 5 days) objective has a
step-0 loss dominated by initial-condition transients, not the signal you want
to learn. As of 2026-07-04, this floor is measured at ~130× the planted signal
in the synthetic recoverability experiment.

**Implication:** fine-tuning a converged model on synthetic observations with
warmup=5 made continuous residual 58× WORSE after 5 epochs (0.0076 → 0.4431
mean L1). The optimizer chased the transient, not the signal.

**Rule:** any identifiability claim requires the noise floor to be below 0.25
mean L1 (Phase B target). This has NOT been met as of 2026-07-05.

### Trap 4 — Worktree stale binary paths

In a git worktree (`ddrs/.claude/worktrees/<name>/`), `target/release` resolves
relative to the worktree, not the main tree. Fresh worktrees lack gitignored
fixtures (`output/`, `.ddrs/`). Always build from the worktree root and reference
data from the main tree's `.ddrs/` by absolute path.

### Trap 5 — Dam/lake regulation confound

Any differential-gauging validation (gauge-pair flow difference as a proxy for
GW-SW exchange) is confounded by regulation. A dam or lake between the gauge
pair produces storage/release cycles that are indistinguishable from reach loss
at the hydrograph level.

**Rule for synthetic experiments:** plant synthetic zeta and evaluate ONLY at
unregulated gauge pairs (GAGES-II reference class, NID dam proximity filter).
This filter was applied in the gradient probe (GAGES-II `CLASS = Ref`) and
must be applied in any future real-data differential-gauging validation.

---

## 9. The research arc (chronological, with verdicts)

### Phase 0 — Baseline characterization (2026-06-19 journal)

**Observation:** trained KAN + MC routing does not beat the summed-Q' baseline.
**Root cause (three layered findings):**

| Finding | Status |
|---|---|
| Loss objective (L1 vs NNSE-KGE vs component-KGE) is NOT the limiter | PROVEN — switching loss did nothing; training loss flat across all runs |
| Daily→hourly flat `repeat-24` + daily-mean loss kills the routing gradient | PROVEN — X stuck at init (median 0.246, IQR <0.012) in all flat-daily runs |
| Structural ceiling: daily-resolution routing over pre-UH-routed forcing has no generalizable held-out skill beyond summed-Q' | PROVEN — disagg enabled loss descent but revealed the ceiling (early stopping: best KGE 0.671, still below baseline 0.723) |

**Implication:** to beat the baseline on KGE requires changing the problem:
sub-daily observations, less-pre-routed forcing, or a regularized disagg head.
More model capacity overfits faster without addressing the structural ceiling.

### Phase 1 — Precip-driven disaggregation (2026-06-23)

**Change:** replace attribute-static disagg with a precip-conditioned head using
AORC hourly catchment-scale precipitation as an additional input.

**Result (as of 2026-07-05):** NSE 0.715 / KGE 0.711 on 2365 CONUS gauges.
NSE beats the 0.689 baseline by +0.026. KGE does NOT beat the 0.723 baseline.

**Run ID:** `2026-06-23T02-49-12Z-conus-hourly-train-and-test` (hourly-OFF arm).

### Phase 2 — Leakance x hourly 2x2 (2026-07-01)

**Experiment:** 2x2 factorial (leakance ON/OFF x forcing daily/hourly).

**Results:**

| arm | NSE | KGE |
|---|---|---|
| hourly-OFF | 0.7153 | 0.7104 |
| hourly-ON  | 0.7145 | 0.7150 |
| daily-OFF  | 0.7004 | 0.7244 |
| daily-ON   | 0.6963 | 0.7250 |

Losing-stream subset (1883/2365 gauges where baseline over-predicts):
- hourly ON-OFF: ΔNSE +0.0005, ΔKGE +0.0018, 55.5% gauges improve
- daily ON-OFF: ΔNSE -0.0017, ΔKGE -0.0009, 35.6% improve

Zeta diagnostic (hourly-ON, 64,892 eval reaches):
- median |zeta| = 6.4e-4 m³/s
- |zeta| > 0.01 m³/s on 10.4% of reaches (GO bar: >=10%)
- 53.7% net-losing

**Verdict: GO-marginal.** All 3 gate criteria met. K_D pins at ceiling (1e-6 s-1)
on 100% of reaches. "Widen K_D" listed as top follow-up at this point.

### Phase 3 — Low-zeta diagnosis battery (2026-07-02)

**Experiment:** 7 pre-registered hypotheses explaining why zeta is small.
Script: `scripts/leakance_diagnosis.py`.

**Results:**

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H1 | Structural ceiling (K_D box) | REFUTED | 71.5% of reaches CAN exceed 0.01 m³/s in-box; median utilization 3.4% |
| H2 | Driving-head starvation | SUPPORTED | median head 0.021 m; 47.0% of reaches gaining at eval-window mean |
| H3 | KAN variance collapse | REFUTED | d_gw-meanP Spearman +0.71; K_D-aridity +0.61 — strong spatial structure |
| H4 | Gauge bias / gradient starvation | SUPPORTED | zeta-uparea Spearman +0.76; gauged median |zeta| 11x ungauged; dry/wet ratio 0.40 (inverted) |
| H5 | Equifinality (routing params absorb leakance) | SUPPORTED (daily only) | daily Δn = +0.012 (0.59 IQR); hourly Δn nil |
| H6 | Wrong yardstick | REFUTED | fractional loss agrees: 8.4% lose >1% of local flow |
| H7 | Model-form error (disconnected regime) | REFUTED | 0.0% of d_gw at bounds in any tercile |

**Phase-3 gate FAILED** (H1 REFUTED; box is not the cap). K_D widening retrain NOT run.
"Widen K_D" recommendation from Phase 2 is **explicitly superseded**.

One-line answer: zeta is small because the optimizer throttles the flux
through the driving head (H2), the gradient only lives near large gauged rivers
(H4), and under daily forcing routing parameters compensate (H5) — NOT because
the K_D box clips it.

### Phase 4 — Gradient probe (2026-07-03)

**Experiment:** Two instruments with zero training:
- Stage 1: adjoint reachability map (96 windows, trained + cold head).
- Stage 2: detectability bound (292 planted-delta probes at 104 gauges, 8 rounds, CPU deterministic).

**Results:**

| Hypothesis | Bar | Measured | Verdict |
|---|---|---|---|
| P1 Starvation | gauged/ungauged |g| >= 10x at both points | 1.5x (trained), 2.9x (cold) | REFUTED |
| P2 Rejection | >67% dry-tercile gradients push zeta down (trained) | 52.5% (~neutral) | REFUTED |
| P3 Detectability | >=10% of Ref probes at delta=0.01 detectable | 4.2% (4/96) | NO-GO |

Key numbers:
- A 0.01 m³/s planted loss arrives at its nearest gauge at ~95% fidelity.
- Median Ref gauge 5% observational band: 0.531 m³/s — 53x the planted signal.
- Detection fails on dilution, not transmission.

Post-hoc finding: at the cold initialization, 80.5% of ungauged gradients push
zeta DOWN (initial-training suppression before convergence).

Trained sign map is physically coherent: "wants more leakance" clusters in the
interior West / High Plains (correct losing-stream country).

**Conclusion:** the physics term is healthy; the supervision is the obstacle.
Gauge-only discharge training cannot learn real-world leakance — measured, not argued.

### Phase 5 — Synthetic recoverability positive control (2026-07-04)

**Experiment:** Plant a known zeta in a teacher world (58 reaches, factor_norm p50=0.5),
generate synthetic obs, train students from the teacher's weights, measure recovery.

**Results:**

| Metric | Measured | Bar | Verdict |
|---|---|---|---|
| R1 Recovery ratio median (n=58) | 0.009 | >=0.5 RECOVERED | FAILED |
| R2 Non-planted |zeta_net| A/baseline | 1.11 | <2 PRECISE | PRECISE (trivially) |
| R3 Final-epoch loss A vs B | A=1.339 vs B=2.317 (+42.2%) | A<B by >5% | A<B but CONFOUNDED |
| R4 Absorption map | median Δn planted = -0.019; global | descriptive | no localization |
| R5 Cold emergence ratio | 1.20 | >3 EMERGES | SUPPRESSED |

**HEADLINE: positive control FAILED.**

Decomposition of why R1 failed even in the best possible world:

| Quantity | Value |
|---|---|
| Continuous residual (teacher weights + teacher obs, full-window eval) | 0.00759 mean L1 |
| Step-0 windowed training loss (student A) | 1.017 |
| Ratio (noise floor / signal) | ~130x |
| Student A continuous residual after 5 epochs | 0.4431 (58x worse than not training) |

**Root cause:** warmup=5 under-trims hotstart transients by ~2 orders of magnitude.
Big rivers carry memory of tens to hundreds of days. The optimizer chases the
transient noise, not the planted signal.

**Critical implication:** leakance identifiability is NOT proven. Phase B
(state-cache hotstart, <=0.25 mean L1 noise-floor target) is required before any
identifiability claim. Phase B has NOT been run as of 2026-07-05.

K_D ceiling binding a second time: widening from [1e-8, 1e-6] to [1e-8, 1e-5]
was required to achieve adequate expressibility at plant sites (58/96 kept vs
23/96 with the original range). The widened range [1e-8, 1e-5] is now the
recommended default for any future leakance work.

---

## 10. Current state and open questions (as of 2026-07-05)

### What is proven

- NSE beats summed-Q' baseline (+0.026) with precip-driven disaggregation.
- KGE does NOT beat the summed-Q' baseline (0.711 vs 0.723) in any config.
- Leakance under hourly forcing improves KGE on the losing-stream subset (+0.0018).
- The leakance gradient reaches all 64,892 eval reaches (P1 refuted).
- A literature-magnitude loss (0.01 m³/s) transmits to its nearest gauge at ~95% fidelity.
- Gauge-supervised training cannot detect that signal (53x dilution, P3 NO-GO).
- The windowed training objective's noise floor is ~130x the planted signal (warmup=5).
- The KAN head correctly encodes physical attribute structure in leakance parameters (H3 refuted).
- K_D pinning at the ceiling is a consequence of the driving-head throttle (H1 refuted), not a root cause.

### What is NOT proven

- Leakance identifiability from gauge-supervised training (recoverability control FAILED).
- Any config beating the KGE baseline.
- Phase B (state-cache hotstart) can reduce the noise floor to <=0.25 mean L1.

### Open questions ranked by evidence priority

1. **Phase B — state-cache hotstart noise floor.** Can persistent-state training
   windows (or warmup >= 30-60 days) reduce the floor below 0.25 mean L1? This
   is forward-only (cheap) and must be answered before any identifiability claim.

2. **Auxiliary spatial constraint.** Regularize `zeta_net` or `d_gw` against an
   independent losing-potential signal (water-table depth, aridity). Must act
   directly on head outputs, NOT through routed discharge (which is subject to
   the 130x noise floor). Requires staged training: leakance-OFF convergence
   first, then leakance-ON + auxiliary term.

3. **Beating KGE baseline.** Sub-daily USGS observations to supervise the
   disaggregation head directly. Less-pre-routed forcing (hillslope runoff).

4. *(Deferred)* K_D widening past [1e-8, 1e-5] — only revisit after (1) or (2)
   restores a gradient path.

---

## 11. Provenance and maintenance

Source files verified before writing (2026-07-05). Re-run to verify still current:

```bash
# Leakance 2x2 metrics
grep -n "VERDICT\|GO\|median" /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-findings.md | head -20

# Low-zeta diagnosis verdicts
grep -n "SUPPORTED\|REFUTED" /home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md

# Gradient probe verdicts
grep -n "REFUTED\|NO-GO" /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-03-zeta-gradient-probe-findings.md

# Recoverability HEADLINE
grep -n "HEADLINE\|R1\|noise floor" /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-04-synthetic-recoverability-findings.md | head -20

# Baseline metrics
grep -n "summed-Q\|0.689\|0.723" /home/tbindas/projects/ddrs/docs/6_19_26_journal.md | head -10

# DDR parity guard (must say ABSOLUTE MATCH)
cargo run --release --example compare_ddr_sandbox 2>&1 | grep -i "absolute\|match\|diff"
```

Files read to produce this skill:
- `/home/tbindas/projects/ddrs/docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md`
- `/home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md`
- `/home/tbindas/projects/ddrs/docs/6_19_26_journal.md`
- `/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-03-zeta-gradient-probe-findings.md`
- `/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-04-synthetic-recoverability-findings.md`
- `/home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-findings.md`
- `/home/tbindas/projects/ddrs/CLAUDE.md`
