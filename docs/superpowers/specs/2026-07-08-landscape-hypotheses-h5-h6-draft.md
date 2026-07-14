# H5/H6 — parameter transfer and loss-landscape hypotheses (H5 REGISTERED / H6 INFRA-READY)

Date: 2026-07-08. **Status: H5 REGISTERED (2026-07-08) — the enabling
`--mode eval-loss` binary exists, is unit- and integration-tested
(`tests/eval_loss_own_parity.rs` passes against a real checkpoint), and no
H5 evaluation has yet been run under the registered protocol above (96
windows). H6 INFRA-READY (2026-07-08) — the enabling `--mode landscape`
mode exists on the same binary (Stage 7 in `src/bin/probe_zeta_gradient.rs`:
11×11 surface scan over the log/linear-aware arm-mean anchor, 21-point
log-space linear barrier, `--single-point` argmin re-check; the split-half
noise floor is a `--seed` re-invocation), is unit-tested and smoke-tested
against the real R1/R3 checkpoints — but NO H6 evaluation has been run under
the registered protocol (16 windows, 256-gauge subsample, 11×11 grid); do
not interpret any H6 result until a registered run exists and this line is
updated again.** Continues the numbering of the LSTM equifinality campaign (H1–H4,
`docs/2026-07-07-lstm-equifinality-v2-findings.md`). Literature basis:
`docs/2026-07-08-equifinality-litreview-experiments.md` (E1/E2/E3/E6 there).

Interpretive frame (Chis et al. 2016; Renard et al. 2010): the LSTM arms left two
readings of Manning's n open — *sloppy* (poorly constrained but structurally
identifiable: same optimum, wide basin) vs *compensator* (structural
non-identifiability with input error: forcing-indexed family of optima). H5 tests
this at the endpoint level, H6 at the landscape level. Verdicts:
SUPPORTED / REFUTED / INCONCLUSIVE only.

## Enabling infrastructure (one item, before registration)

**Built as `--mode eval-loss` on the existing `probe_zeta_gradient` binary**
(not a new bin — Stage 6 in `src/bin/probe_zeta_gradient.rs`, doc comment
search "Stage 6"), reusing the SAME deterministic training-style
rho-window/gauge sampler as the floor/grad-probe modes (LOCAL `ChaCha12Rng`
seeded from `--seed`, default 42). It evaluates the training L1 loss over a
fixed window plan under each of `own`/`n-swap`/`geo-swap`/`full-swap`,
injecting `n`/`q_spatial`/`p_spatial` from `--donor-params-nc` (a
`dump_parameters::write_netcdf` dump) in place of the checkpoint's own
KAN-head output, bypassing the head for the swapped fields only.

Real flags (see the binary's module doc for the full invocation):
`--mode eval-loss`, `--config` (selects Q′ source + gauge set),
`--checkpoint` (required — the arm's trained KAN head), `--donor-params-nc`
(required unless `--compositions own`), `--compositions` (comma-separated
subset of `own,n-swap,geo-swap,full-swap`; default all four), `--windows`
(sampler window count — **CLI default is 32**, inherited from the older
grad-mode probe; **not** H5's registered sample size, see the protocol
section below), `--seed` (sampler seed; default 42), `--loss-output`
(required — tidy CSV `composition,window,mean_loss`), and `--backend`
(`cpu`/`cuda`). There is no `--params-nc`, `--gauges-subset`, or
`--forward-only` flag — those names were provisional at draft time and never
implemented; the eval-loss mode has no separate forward-only toggle because
it is inherently forward-only (no backward pass, no optimizer).

**Analysis-set scope (deliberate, not an oversight).** The 132,336-reach
"analysis set" referenced below is a *different* population than what H5
actually swaps over. That set was defined for the H1–H4 descriptive
convergence analysis (`scripts/equif_convergence_analysis.py`) — a common
cross-arm reach subset with full coverage across masks, used for
parameter-distribution statistics. H5 does not intersect with it: the
swap/eval operates on whatever reaches the training-style rho-window sampler
draws into each batch's subgraph (`batch.divide_comids`), which varies window
to window and may include reaches outside the 132,336-reach set entirely.
This is because H5 measures the training LOSS via the live training sampler's
own windows/gauges (2,365 training gauges' subgraphs), a fundamentally
different mechanism from the H1–H4 Level 1/2 per-reach statistics computed
directly over the fixed analysis set. Restricting H5's swap to that set would
require threading an extra reach mask through `dataset.collate` /
`RandomSampler` — out of scope for this registration; the donor NetCDF itself
(`dump_parameters::write_netcdf`) already covers every reach any arm's
sampler could draw, so no swap ever silently falls back to an untouched
field for an in-scope reach.

Primary arm pair: R1 (daily LSTM flat) vs R3 (hourly MTS-LSTM) — the only fully
independent pair. R1↔R2 (shared store) is the low-disagreement control pair.
All coordinates in log/normalized space (Dinh et al. 2017 reparameterization
caveat).

---

## H5 — Forcing-bound roughness (parameter swap/transfer test)

**Hypothesis.** Learned Manning's n is bound to its training forcing: exchanging
n fields between independently forced arms degrades the training objective
substantially more than exchanging geometry fields (p, q), under either arm's Q′.

**Rationale.** If n compensates source-specific inflow error (Kavetski et al.
2006b), arm B's n encodes corrections for arm B's forcing and is *wrong* under
arm A's forcing; realized-geometry convergence (v2 findings §4.2) predicts
geometry swaps are comparatively harmless. Renard et al. 2010 implies the
mixture is non-identifiable from discharge alone — the swap test measures how
much of each arm's fit is forcing-specific, per parameter class.
Non-transferability of compensating optima: Bárdossy & Singh 2008.

**Design.** For (A, B) = (R1, R3) and each forcing X ∈ {A, B}, evaluate on the
full 96-window set, 2,365 training gauges:

**Protocol (registered).** Every registered H5 invocation MUST pass
`--windows 96` explicitly — the `probe_zeta_gradient --mode eval-loss` CLI
default is 32, inherited from the older grad-mode probe (Stage 1) and not
appropriate for H5's own registered sample size; a run that omits `--windows`
silently under-samples to 32 windows and does not satisfy this design. There
is no `--gauges-subset` flag: H5 evaluates whatever gauges the sampler draws
for each window (2,365 training gauges is the full population, not a
subset selection this binary needs to apply) — H6 will need a gauge subset
(its 256-gauge fixed stratified subsample), and even there the plan is to
point `data_sources.gages` at a pre-generated CSV subset rather than add a
new CLI flag.

| θ evaluated | composition |
|---|---|
| θ_X (own) | baseline |
| n-swap | n from the other arm; geometry own |
| geo-swap | p, q from the other arm; n own |
| θ_other (full swap) | both from the other arm |

Transfer penalties: P_n = L_X(n-swap) − L_X(own); P_geo = L_X(geo-swap) − L_X(own);
attribution fraction f_n = P_n / (P_n + P_geo). Report absolute and normalized by
the full-swap penalty; per-gauge distributions and DA-stratified medians —
produced via the optional `--per-gauge-output` CSV (`composition,window,
staid,gauge_loss`, one row per surviving gauge per window per composition;
schema and DA-join documented in
`.claude/skills/ddrs-eval-plots/references/parameter_swap.md`), which sits
alongside the required `--loss-output` CSV (`composition,window,mean_loss`)
without changing its schema. Control: same table for R1↔R2 (expect small
penalties; their difference from R1↔R3 is the source-disagreement effect).
Noise floor: split-half over windows.

**Falsification bars (registered).**
- H5 SUPPORTED iff f_n ≥ 2/3 under BOTH forcings and P_n exceeds the split-half
  noise floor by ≥ 3×.
- H5 REFUTED iff f_n ≤ 1/2 under EITHER forcing (geometry carries at least as
  much transfer damage as n).
- Otherwise INCONCLUSIVE.

**Expected outcomes.** Compensator: f_n ≫ 2/3, and P_n larger under the
*receiving* forcing whose Q′ disagrees most with the donor's. Sloppy-but-
identifiable n: P_n ≈ 0 (n transfers freely; wide basin, shared optimum).
Identifiable-but-different-optima (structural input error): P_n large AND
symmetric — distinguished from sloppiness, not from compensation (H6 separates).

**Cost.** 8 evaluations + controls; hours on CPU, no retraining.

---

## H6 — Forcing-indexed valley (loss-landscape overlay)

**Hypothesis.** In physical parameter space, the training objective under each
Q′ has a degenerate n–geometry valley (sloppy direction), and the valley-floor
location shifts with the Q′ source by more than the noise floor — cross-arm n
divergence is movement along a forcing-indexed family of minima, not noise
around a shared minimum.

**Rationale.** Response-surface degeneracy is the classical signature of
hydrologic equifinality (Sorooshian & Gupta 1983; Duan et al. 1992); normalized
2D slices and barriers are its modern ML instruments (Li et al. 2018; Frankle et
al. 2020); the profile L*(α) = min_β L(α, β) is the Raue 2009 diagnostic and
falls out of the surface for free.

**Design.** Anchor θ̄ = arm-mean field in log space. Axes: α = global
multiplicative n scaling, β = global multiplicative p scaling (log2 α, log2 β ∈
[−1.5, 1.5]); 11×11 grid. Cost engineering (registered as part of design):
forward-only evaluation, 16-window fixed subset, 256-gauge fixed stratified
subsample — stability verified by split-half agreement of the surface minimum
before interpretation; one full-96-window row through the minimum as anchor.
Measurements, each under BOTH R1's and R3's Q′:
1. Surface L_X(α, β); minima (α*_X, β*_X); 5%-sublevel set geometry.
2. Linear barrier: L_X(θ(t)), θ(t) = exp((1−t) log θ_R1 + t log θ_R3),
   t ∈ {0, 0.05, …, 1}; barrier B_X = max_t L_X − max(endpoint losses).
3. Profiles L*_X(α) and L*_X(β) from the surface.
Noise floor for the minimum location: recompute one surface with the alternate
window subset (split-half); floor = resulting minimum displacement.

**Falsification bars (registered).**
- Degeneracy: valley exists iff the 5%-sublevel set aspect ratio ≥ 3:1.
- Forcing dependence: SUPPORTED iff ‖(α*, β*)_R1 − (α*, β*)_R3‖ (log coords)
  ≥ 3× the split-half noise displacement, with displacement predominantly along
  the valley axis.
- H6 SUPPORTED iff both hold. REFUTED iff minima coincide within noise
  (floor pinned) — regardless of valley flatness. Otherwise INCONCLUSIVE.
- Barrier report (secondary, no bar): B_X under both forcings; same-basin vs
  separate-basin classification per forcing.

**Joint interpretation (registered up front).**

| | floor pinned across Q′ | floor moves with Q′ |
|---|---|---|
| **sharp basin** | n identifiable | forcing-specific identification (structural input error) |
| **flat valley** | n sloppy, NOT compensatory | n is a compensator (selective equifinality, revised thesis) |

H5 and H6 land in the same cell under each reading; disagreement between them is
itself diagnostic (e.g., H5 supported + H6 floor-pinned ⇒ transfer damage comes
from reach-scale pattern, not global level — analyze per-reach residual fields).

**Cost.** ~121 × 2 forward-only subset evals (detached overnight) + 21 × 2
barrier points + 2 split-half surfaces. No retraining.

---

## Ordering and provenance rules

Run AFTER the eval binary lands, BEFORE the dHBV2 arms (these interrogate
existing checkpoints; dHBV2 then extends both tests cross-family for free).
Standard rules: `cargo install --path .` after src changes; detached `nohup`
for the grid; no gated files; results doc cites this spec verbatim.
