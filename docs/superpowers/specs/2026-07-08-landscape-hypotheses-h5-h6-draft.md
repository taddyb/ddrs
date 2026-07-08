# H5/H6 — parameter transfer and loss-landscape hypotheses (DRAFT pre-registration)

Date: 2026-07-08. Status: DRAFT — becomes registered when the enabling binary
exists and before any H5/H6 evaluation is run. Continues the numbering of the
LSTM equifinality campaign (H1–H4, `docs/2026-07-07-lstm-equifinality-v2-findings.md`).
Literature basis: `docs/2026-07-08-equifinality-litreview-experiments.md` (E1/E2/E3/E6 there).

Interpretive frame (Chis et al. 2016; Renard et al. 2010): the LSTM arms left two
readings of Manning's n open — *sloppy* (poorly constrained but structurally
identifiable: same optimum, wide basin) vs *compensator* (structural
non-identifiability with input error: forcing-indexed family of optima). H5 tests
this at the endpoint level, H6 at the landscape level. Verdicts:
SUPPORTED / REFUTED / INCONCLUSIVE only.

## Enabling infrastructure (one item, before registration)

`eval_loss` mode (extend probe binary or new bin): evaluate the training L1 loss
over deterministic windows (seed 42) with per-reach `n, q_spatial, p_spatial`
INJECTED from a NetCDF file, bypassing the KAN head. Flags: `--params-nc`,
`--config` (selects Q′ source), `--windows N`, `--gauges-subset` (optional CSV),
`--forward-only`. All H5/H6 measurements are calls of this binary over parameter
files generated in Python.

Primary arm pair: R1 (daily LSTM flat) vs R3 (hourly MTS-LSTM) — the only fully
independent pair. R1↔R2 (shared store) is the low-disagreement control pair.
All fields restricted to the 132,336-reach analysis set; all coordinates in
log/normalized space (Dinh et al. 2017 reparameterization caveat).

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

| θ evaluated | composition |
|---|---|
| θ_X (own) | baseline |
| n-swap | n from the other arm; geometry own |
| geo-swap | p, q from the other arm; n own |
| θ_other (full swap) | both from the other arm |

Transfer penalties: P_n = L_X(n-swap) − L_X(own); P_geo = L_X(geo-swap) − L_X(own);
attribution fraction f_n = P_n / (P_n + P_geo). Report absolute and normalized by
the full-swap penalty; per-gauge distributions and DA-stratified medians.
Control: same table for R1↔R2 (expect small penalties; their difference from
R1↔R3 is the source-disagreement effect). Noise floor: split-half over windows.

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
