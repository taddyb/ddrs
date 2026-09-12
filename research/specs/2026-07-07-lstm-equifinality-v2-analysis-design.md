# LSTM-source selective-equifinality — v2 analysis design (audit-corrected rules)

Date: 2026-07-07. Branch: `unit_catchments`. Supersedes the ANALYSIS RULES (not
the runs, not the raw artifacts) of
`docs/superpowers/specs/2026-07-06-lstm-equifinality-cpu-design.md`.

Inputs already on disk, NOT re-run by this spec: R1/R2/R3 training+eval
(`2026-07-07T03-55-53Z` / `04-49-19Z` / `06-50-28Z-train-and-test`, sha
`b261e1d`), `output/equif/R{1,2,3}_kan_parameters.nc`,
`output/equif_probe/grad_R{1,2,3}.nc` (seed 42, 96 windows) +
`grad_R{1,2,3}_seed123.nc` (noise-ceiling replicate). This spec governs the
analysis script only (`scripts/equif_convergence_analysis.py`) and the
findings doc it feeds.

## Why this spec exists

The v1 registered rules (H1/H2 spec above) produced verdicts H1–H3 REFUTED /
H4 INCONCLUSIVE. An independent audit
(`/tmp/experiment-handoff-lstm-equifinality-audit.md`, verified twice) found
the v1 H1 rule compared geometry spread normalized by per-reach cross-arm mean
against n spread normalized by its parameter-box width (0.235, ≈3× typical
learned n) — a metric asymmetry, not a computational error, that reverses the
H1 direction when corrected. The audit's items 1–3 (dual-metric H1/H2,
timing-axis H2, CM-removed H3 + noise ceiling) were implemented in this
session (commits `7dc58c2`, `9810c32`) and are ALREADY the registered `audit`
key of `verdicts.json` / findings doc §8. This spec formalizes those as the
v2 PRIMARY rules (not a second post-hoc pass), pre-registers the two items
the audit flagged as still-untested (network-scale H2, distributional/DA
Level 1.5) BEFORE they are computed, and states N-arm-ready aggregation rules
so the deferred dHBV2 arms slot in without a rules rewrite.

Per repo convention (SUPPORTED/REFUTED/INCONCLUSIVE only; no HARKing): the
v1-registered verdicts are reported as registered, unchanged. This spec adds
v2 verdicts computed under the rules below and does not retroactively edit
the v1 numbers.

## V2 rules, fixed before computing

### H1 / H2 spread metric — dual normalization, relative-to-mean is primary

Report BOTH for every parameter and every realized-geometry quantity:

- **Box-normalized** (v1-registered): `(max−min across arms) / range_width`,
  where `range_width` comes from `params.parameter_ranges`. Reported as the
  v1-registered sensitivity check, unchanged formula.
- **Relative-to-mean** (v2 PRIMARY for any n-vs-geometry comparison):
  `(max−min across arms) / (|cross-arm mean| + eps)`, same formula already
  used for geometry in v1. This is the only like-for-like comparison between
  n (a coefficient with an arbitrary box) and geometry (a physical quantity
  whose natural scale is its own mean).

Reported statistic: median over analysis reaches, both normalizations, side
by side, for every quantity — never one alone.

### H2 disagreement axis — timing/shape primary, network-scale variant added

Muskingum–Cunge conserves mass, so n cannot compensate a Q′ source's mean-
volume disagreement under a daily-aggregated L1 loss; the mechanism the paper
actually hypothesizes is timing/attenuation compensation. Primary axis
(already computed, `stage_b2`, per-reach):

- Inter-store Pearson r of summed daily hydrographs (daily-lstm vs
  hourly-lstm — the only 2 distinct stores in this run set).
- Richards–Baker-style flashiness difference.

ρ(n rel-spread, 1−r) and ρ(n rel-spread, flashiness-diff) are v2 PRIMARY;
volume-based ρ (v1) reported alongside as the registered sensitivity check.

**New: network-scale variant.** Per-reach timing disagreement mixes gauge-
local and far-upstream contributions that a single calibrated n at a gauge
cannot separately compensate — the physically relevant unit for a
bias-absorber hypothesis is the gauge's INTEGRATED upstream response, not an
isolated reach. For each gauge g with upstream network U(g) (from the gauge
adjacency store, already loaded in Stage A):

- `n_disagreement(g)` = median of per-reach n rel-spread over `U(g)`.
- `timing_disagreement(g)` = median of per-reach `(1 − pearson_r)` over
  `U(g)` (equivalently, network-mean flashiness-diff, reported alongside).
- Spearman ρ across gauges between `n_disagreement` and `timing_disagreement`.

Falsification bar unchanged from v1: ρ ≤ 0.2 (either axis, either scale) or
n-rel-spread ≈ geometry-rel-spread ⇒ H2 refuted at that scale. Because this
is a genuinely new test (not previously run), its result is reported as its
own v2 verdict, not folded into the v1 REFUTED verdict.

### H3 — CM-removed cosines are PRIMARY, raw reported alongside

Already implemented (`stage_e_ext`), formalized as v2 primary here:

- Null for CM-removed pairwise cosines at k=3 arms is **−1/(k−1) = −0.5**
  (residuals sum to zero per reach), not 0. All CM-removed numbers are
  reported against this null, never against 0.
- Within-arm split-half noise ceiling (seed-42 vs seed-123 window sets, 96
  windows each) reported alongside every cosine as the SNR floor.
- Raw (non-CM-removed) cosines reported alongside for continuity with v1.
- Explicit caveat carried forward: CM removal cannot distinguish
  shared-initialization descent from a genuinely source-independent signal —
  both are common mode. A CM-removed cosine near the null is NOT proof of "no
  real signal"; it is consistent with either "no signal" or "all signal is
  common-mode-confounded with init." This spec does not claim to resolve that
  ambiguity (the seed replicate in findings §5 item 3 is the resolving
  experiment, not yet run).

### New Level 1.5 — per-arm distributions and drainage-area conditioning

Spread MEDIANS (Levels 1–2) collapse the full distribution and any
covariate structure. This level, run once per arm and once cross-arm, adds:

1. **Per-arm percentiles** (p5/p25/p50/p75/p95) of n, q_spatial, p_spatial
   over the analysis set, per arm — exposes distributional shape (e.g. one
   arm's parameter collapsing to near-constant) that a median cannot.
2. **Drainage-area conditioning.** Join analysis-set COMIDs to
   `log10_uparea` from `~/projects/ddr/data/merit_global_attributes_v2.nc`
   (2,939,404 COMIDs; all 132,336 analysis COMIDs present — the eval network
   is a MERIT CONUS subset). Per arm:
   - Binned median of the raw parameter vs log10_uparea decile bins.
   - OLS slope of `ln(parameter)` vs `log10_uparea` (log-log space — matches
     classical downstream hydraulic-geometry scaling `n ~ DA^b` /
     Leopold–Maddock `width, depth ~ DA^b`; simple `np.polyfit`, not a
     weighted/robust fit — this is exploratory, not a hypothesis test).
   Applied to n, q_spatial, p_spatial AND realized depth/top_width/hydraulic_
   radius (arm-own reference discharge, consistent with Level 2 primary).
3. **Spread-vs-DA profile.** Median cross-arm rel-spread (the v2-primary
   relative-to-mean metric) within each log10_uparea decile bin, for every
   quantity — reveals whether cross-arm disagreement concentrates in
   headwaters or is DA-uniform, which a single collapsed median cannot show.

No falsification bar — Level 1.5 is descriptive context for interpreting
H1/H2, not itself a hypothesis test. Reported as "Level 1.5" in the findings
doc, between raw parameters (Level 1) and realized geometry (Level 2).

**Reproduction targets** (audit's manual recomputation, to be matched by the
pipeline implementation, 132,336-reach analysis set):
- Box-normalized n spread median: **0.1555** (v1-registered, unchanged).
- Relative-to-mean n spread median: **0.4512**.
- Per-arm median n: R1 **0.0835**, R2 **0.1001**, R3 **0.0651**.
- ln(n)-vs-log10(DA) OLS slope: R1 **≈+0.184**, R2 **≈−0.027**, R3 **≈+0.145**.

A mismatch beyond float/percentile-method noise (~1e-3) on any of these four
numbers means the new stage has a bug — halt and debug before trusting any
new number the same stage produces.

### N-arm-ready aggregation rules

The dHBV2 arms are NOT run by this spec (deferred, findings §5 item 1). Rules
are stated so a 4th/5th arm is a data addition, not a rules rewrite:

- **Spread** (both normalizations): `(max−min across k arms) / denom` is
  already k-agnostic — `np.stack([...], axis=0)` over a list, not a fixed
  triple. Code must accept a `list[dict]` of per-arm parameter dicts, not
  three positional dict arguments.
- **CM-removed cosine null**: `−1/(k−1)`, parametrized by `k = len(arms)`,
  already derived that way — carries over unchanged.
- **Spearman/ρ statistics**: pairwise by construction (H2, H3 raw cosines);
  with k>3 arms, report the full pairwise matrix (not just adjacent pairs)
  plus the mean off-diagonal as a summary, so adding arms doesn't silently
  drop pairs.
- **Percentiles / DA slopes (Level 1.5)**: already per-arm (a `for arm in
  arms` loop), trivially k-agnostic.
- Practical scope for THIS spec: refactor the shared helper functions
  (`param_stats`, `geometry_spread_at_Q`, CM-removal, percentile/DA helpers)
  to take `list[dict]` rather than three positional args, so the formulas are
  k-agnostic. Do NOT build out a variadic `--params-r4`/`--grads-r4` CLI or
  wire in dHBV2 data paths now — that is real work belonging to the dHBV2
  arm's own task, not this rules-correction pass. Verify k-agnosticism by
  running the refactored k=3 path and reproducing the exact numbers above,
  not by running k=4 (there is no 4th arm's data yet).

## What does NOT change

- The three arms, their configs, checkpoints, and dumps (`docs/superpowers/
  specs/2026-07-06-lstm-equifinality-cpu-design.md` §Arms) — no retraining.
- The v1-registered H1–H4 verdicts in `docs/2026-07-07-lstm-equifinality-
  findings.md` §3 — reported as registered, alongside the v2 numbers, never
  overwritten.
- Stage A/B/D (network, coverage, routing skill) — unaffected by this spec.
- `use_leakance: false` on every arm (leakance is out of scope for this
  paper, per project CLAUDE.md).

## Deliverables

1. `stage_b2` extended with the network-scale H2 variant (per-gauge
   aggregation over `U(g)`).
2. New stage (`stage_g` or equivalent) implementing Level 1.5: percentiles,
   DA-conditioned binned medians + OLS slopes, spread-vs-DA profiles.
3. Shared spread/CM/percentile helpers refactored to `list[dict]` inputs
   (N-arm-ready), verified k=3 reproduces the four target numbers above.
4. `stage_f` (verdicts) extended with v2 verdicts for the network-scale H2
   test (new, own verdict) and Level 1.5 summary (descriptive, no verdict).
5. New dated findings doc reporting v1-registered verdicts (unchanged) +
   v2-primary numbers/verdicts, per project convention (SUPPORTED / REFUTED /
   INCONCLUSIVE only).

## Concerns / assumptions

- **Concern — network-scale H2 could still show ρ>0.2 while per-reach ρ<0**
  (or vice versa): aggregation over `U(g)` changes the sample from ~132k
  reaches to ~8,945 gauges and could shift statistical power either
  direction. *Mitigation:* report both scales, no verdict privileges one over
  the other a priori — this was the audit's explicit ask ("untested
  network-scale variant"), not a predicted-direction change.
- **Concern — Level 1.5's OLS slopes are exploratory, not causal**: log-log
  regression against a single covariate (DA) on parameters that also depend
  on Q′ source, gauge density, and physiographic region. *Mitigation:*
  reported as descriptive context (§ heading says so explicitly), not used to
  adjudicate H1–H4 verdicts.
- **Assumption — `log10_uparea` join is exact-COMID, no interpolation**:
  MERIT attributes and MERIT CONUS adjacency share the same COMID space by
  construction (both derive from the same fabric); a missing-COMID count >0
  after the join is a bug, not expected data loss — assert on it.
- **Why this change:** the audit is correct that shipping v1's verdicts as
  the paper's Results without the dual-metric/timing/CM-removal correction
  would let a referee reverse the headline with a five-line recomputation;
  formalizing the corrected rules as a pre-registered v2 spec (rather than
  ad hoc post-hoc analysis) preserves the falsification discipline the project
  requires, and Level 1.5 / network-scale H2 close the two gaps the audit
  flagged as still-open before the paper can respond to the audit's own
  demands.
