# Leakance low-zeta diagnosis — experiment report (2026-07-02)

Spec: `docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md`
Lit review: `docs/2026-07-01-leakance-litreview.md`
Script: `scripts/leakance_diagnosis.py` (raw output archived in §6)
Prior experiment: `docs/2026-07-01-leakance-hourly-findings.md` (the 2×2 that
motivated this)

**One-line answer: zeta is small because the optimizer *chooses* small — it
throttles the flux through the driving head (`d_gw` learned ≈ depth, median
head 0.02 m, 47% of reaches gaining at the mean), and the gradient that could
open the head only reaches gauged, large-river reaches (zeta–uparea ρ = +0.76;
gauged median |zeta| is 11× ungauged) — NOT because the `K_D ≤ 1e-6` box clips
it (median utilization of the in-box zeta ceiling is 3.4%).**

---

## 1. Motivating observation and hypotheses

The leakance × hourly 2×2 experiment returned GO-but-marginal: the learned
GW–SW exchange `zeta = leakance_factor · area_z · K_D · (depth − d_gw)` came
out tiny — median |zeta| 6.4e-4 m³/s (hourly-ON), |zeta| > 0.01 m³/s on only
10.4% of the 64,892 eval reaches, `K_D` pinned at its `1e-6 s⁻¹` ceiling on
100% of reaches. Values that small imply leakance has minimal-to-negligible
effect on Q′ and therefore routing. The question: **why?**

Seven falsifiable hypotheses were pre-registered (spec §Hypotheses), informed
by a literature review (two threads: physical leakance magnitudes; and
identifiability of exchange terms trained on gauged discharge only):

| # | Hypothesis | Mechanism proposed |
|---|---|---|
| H1 | Structural ceiling | The `K_D ∈ [1e-8, 1e-6]` box dimensionally caps zeta below detectability — literature streambed leakance reaches 1e-4–1e-3 s⁻¹ (sand beds), so our ceiling covers only clogged silt/clay beds |
| H2 | Driving-head starvation | The learned `d_gw` sits near typical depths, so `(depth − d_gw)` ≈ 0 and the flux is throttled at the head, not the rate constant |
| H3 | KAN variance collapse *(the original user hypothesis)* | The head maps attributes to near-constant leakance params — the network can't express the spatial variance needed to single out losing reaches |
| H4 | Gauge bias / gradient starvation | The loss only samples gauged, large, perennial rivers (Krabbenhoft 2022); losing/ephemeral reaches are systematically ungauged, so no gradient pressure ever grows zeta where it physically belongs |
| H5 | Equifinality | Manning's n / storage absorb the attenuation leakance would provide (Beven; Kirchner bias-compensation), leaving `∂Loss/∂zeta ≈ 0` |
| H6 | Wrong yardstick | The absolute 0.01 m³/s bar is meaningless on headwaters; fractional loss `|zeta|/Q` might already be non-trivial |
| H7 | Model-form error | The linear `(depth − d_gw)` law is a connected-regime Darcy law; the strongest-losing reaches are *disconnected* (Brunner 2009) where flux saturates — the learned `d_gw` would strain against its `[−2, 2] m` bounds |

Each hypothesis carried a pre-registered test and falsification criterion,
plus a **Phase-3 gate**: spend GPU on a widened-K_D retrain *only if* H1 was
supported and the gradient provably alive (H3/H4 not showing total collapse).

## 2. Methods — how the experiment was done

**Instrumentation (Rust, this branch).** zeta needs the routed per-timestep
depth, so H1/H2/H6 could not be tested from the parameter dump alone. The
eval-time zeta accumulator (`MuskingumCunge::enable_zeta_accumulation`) was
extended to also accumulate per-reach routed **depth**, plan-view wetted area
(**area_z**), and **discharge** on the inner backend (no autograd tape), with
eval-window means exported as `depth_mean` (m), `area_z_mean` (m²), `q_mean`
(m³/s) on the `COMID_eval` dimension of `<run_dir>/kan_parameters.nc` —
alongside the existing `zeta`/`zeta_net`. The per-step values are recomputed
from the SAME saved primitives the backward reads, so they are exactly the
quantities that entered `b_rhs`. Training path untouched (zero extra kernels
when the sink is `None`); guards: `tests/zeta_accum.rs` (6 tests, incl.
q-sum ≡ routed-output-columns identity and depth/area_z leakance-invariance),
`leakance_gradcheck` 8/8, `leakance_off_parity` 3/3, `compare_ddr_sandbox`
ABSOLUTE MATCH (max rel diff 2.1e-7).

**Data.** Four valid arms from the 2×2 (seed 42, eval 1995/10–2010/09,
eval network 64,892 reaches):

| arm | run id | inputs used |
|---|---|---|
| hourly-ON | `2026-07-01T13-43-32Z-train-and-test` | re-evaled with the new exports (zeta stats reproduced: 6.4e-4 / 10.4%) |
| daily-ON | `2026-07-01T21-20-27Z-train-and-test` | re-evaled likewise (8.1e-4 / 14.2%) |
| hourly-OFF | `2026-06-23T02-49-12Z-conus-hourly-train-and-test` | `dump_parameters` pass (checkpoint `epoch_5_mb_35/head`) for H5 |
| daily-OFF | `2026-06-05T01-41-16Z-train-and-test` | existing dump (`n` only — that run didn't learn `x_storage`) |

Plus `merit_global_attributes_v2.nc` (aridity, permeability, Porosity,
log10_uparea, meanP, meanslope; 100% COMID match on the eval network) and
`gages_3000.csv` (2,698 gauged reaches on the eval network).

**Battery.** `scripts/leakance_diagnosis.py` runs one section per hypothesis
with fixed thresholds and prints a suggested verdict per section (all
thresholds pre-registered in the plan; a reviewer independently re-derived the
COMID_eval→COMID index mapping and the H1/H2 medians by hand — zero
discrepancies). Key tests: H1 computes the per-reach *in-box ceiling*
`zeta_max = 1·area_z·1e-6·(depth + 2)` and the utilization `zeta/zeta_max`;
H2 the driving-head distribution `depth_mean − d_gw`; H3 param dispersion +
Spearman structure against attributes; H4 stratification by gauged-ness,
upstream area, and aridity terciles; H5 paired ON−OFF shifts of `n`/
`x_storage`; H6 `|zeta|/q_mean`; H7 `d_gw` boundary-pinning by aridity.

## 3. Results — how it was resolved

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H2 | driving-head starvation | **SUPPORTED** | median head 0.021 m; 57.6% < 0.1 m; 47.0% ≤ 0 |
| H4 | gauge bias / gradient starvation | **SUPPORTED** | zeta–uparea ρ +0.76; gauged 6.7e-3 vs ungauged 5.9e-4; dry/wet ratio 0.40 (inverse of physics) |
| H5 | equifinality with routing params | **SUPPORTED** (daily only) | daily Δn = +0.012 (0.59 IQR, n up ~20%); hourly Δn 0.05 IQR (nil) |
| H1 | structural ceiling (K_D box) | **REFUTED** | 71.5% of reaches CAN exceed 0.01 m³/s in-box; median utilization 0.034 |
| H3 | KAN variance collapse | **REFUTED** | d_gw–meanP ρ +0.71, K_D–aridity +0.61 — strong learned structure |
| H6 | wrong yardstick (absolute bar) | **REFUTED** | fractional loss agrees: only 8.4% lose >1% of local flow |
| H7 | model-form error (d_gw pinning) | **REFUTED** | 0.0% of d_gw at bounds, incl. dry tercile |

**The K_D-ceiling story is dead.** Going in, the leading suspect was the
`K_D` box: K_D pins at the ceiling on ~100% of reaches (log10 median −5.999,
IQR 3.6e-4 — essentially a delta function), and the literature says the
ceiling only covers clogged silt/clay beds. But H1 shows the box is not what
caps the flux: with K_D at ceiling, factor at 1, and d_gw at its floor, 71.5%
of reaches could exceed the 0.01 m³/s bar — the learned zeta uses a median
3.4% of that in-box capacity. The optimizer maxes the rate constant and then
throttles the product elsewhere.

**Where the throttle actually is: the driving head.** The KAN learned `d_gw`
as a nearly constant ~0.3 m activation threshold (p5–p95: 0.09–0.44 m) against
depths spanning 0.03–3.3 m. Result: `(depth − d_gw)` is ≤ 0 on 47% of reaches
(gaining at the eval-window mean) and < 0.1 m on 58%. Leakance activates only
where mean depth exceeds ~0.3 m — i.e., on larger rivers. This single learned
threshold explains the H4 pattern mechanically: zeta tracks upstream area
(+0.76) because depth does. *Caveat:* these are eval-window means; sub-daily
storm depths cross the threshold more often than the mean suggests, so 47%
"gaining" overstates true inactivity somewhat — but the median head of 2 cm
leaves no doubt about the throttling.

**Why the head stays closed: the gradient only lives near gauges (H4).** The
2,698 gauged reaches carry a median |zeta| of 6.7e-3 (40.2% above the bar);
the 62,194 ungauged reaches sit at 5.9e-4 (9.1%). And the aridity
stratification is *inverted* from physics: dry-tercile reaches (where Jasechko
2021 puts the losing streams) have 2.5× LESS zeta than wet ones. The flux
lives where the training signal lives — near gauges, on big perennial rivers —
not where losing streams live.

**The KAN is not the problem (H3 refuted — the original hypothesis).** The
leakance parameters carry strong, *physically sensible* attribute structure:
K_D is higher in arid regions (+0.61), d_gw is lower where precipitation is
low (−0.67 vs aridity, +0.71 vs meanP — deeper water table where dry). The
network pushes the *parameters* the right way in dry regions; the *flux*
still ends up largest in wet/large rivers because depth and wetted area
dominate the product and the loss never rewards opening the head on arid
headwaters it cannot see. Parameter-level physics: right. Flux-level outcome:
gauge-shaped.

**Equifinality shows up under daily forcing only (H5).** Under flat-daily
forcing, adding leakance shifts Manning's n up by +0.012 (0.59 IQR, ~20% of
the median) — the routing co-adjusts with the new loss term, the Kirchner
bias-compensation signature, consistent with daily-ON's skill degradation in
the 2×2. Under hourly forcing the shift is nil (0.05 IQR): the sub-daily
depth signal decouples the two mechanisms. (Caveat: single seed per arm;
CUDA nondeterminism makes this suggestive, not decisive — as pre-registered.)

**The magnitude criteria were sound (H6, H7 refuted).** Fractional loss
agrees with the absolute bar (8.4% of reaches lose >1% of local flow, 3.2%
>5% — below the McCallum differential-gauging detectability band), and d_gw
sits interior everywhere, so the linear connected-regime law is not being
strained against its bounds. The Brunner disconnected-regime critique remains
theoretically valid for the arid West, but it is not what limits *this* model.

## 4. Conclusions

1. **Leakance's small magnitude is a training-signal problem, not a
   parameter-box or architecture problem.** The three supported hypotheses
   (H2 head throttling, H4 gauge bias, H5 daily equifinality) form one
   coherent mechanism; the three structural suspects (box, KAN capacity,
   model form) are all refuted.
2. **The pre-registered Phase-3 gate FAILED ⇒ the widened-K_D retrain was
   NOT run** (no `leakance_hourly_on_kd4.yaml`, no GPU spent). The diagnosis
   predicts widening would re-pin K_D at the new ceiling (or shrink the head
   further) with little zeta or skill change, because the binding constraint
   is the signal, not the box. **This supersedes the "widen K_D past 1e-6 —
   top follow-up" recommendation in
   `docs/2026-07-01-leakance-hourly-findings.md` §5 item 2.**
3. **The 2×2's GO-marginal verdict stands** — this diagnosis explains the
   marginality (the term is real but only expressible near the training
   signal); it does not overturn the losing-subset skill gain under hourly
   forcing.
4. **Hourly forcing is a precondition, not a nice-to-have**: under daily
   forcing leakance degenerates into an n-compensated fudge factor (H5);
   under hourly the mechanisms decouple.

## 5. Next steps

In priority order (from the battery + the identifiability literature):

1. **Synthetic losing-reach recoverability experiment** — impose a known zeta
   on synthetic data and test whether the gradient path recovers it through
   gauged-only observations (the Bindas 2024 WRR identifiability methodology
   applied to leakance). Cheap, and decisive on whether the term is
   recoverable *at all* under gauge bias. **Confound to design around: many
   gauge pairs sit near dams/lakes**, where regulation storage/release
   masquerades as reach loss/gain. In the synthetic setting the truth is
   controlled, so the confound enters through site selection: plant synthetic
   zeta (and place the evaluating gauge pairs) only on unregulated reaches —
   filter by GAGES-II reference class / NID dam proximity or an equivalent
   regulation flag — and treat regulated subnetworks as a held-out contrast
   group. The same filter is mandatory for any later real-data
   differential-gauging validation, where regulation is indistinguishable
   from GW–SW exchange at the hydrograph level.
2. **Auxiliary spatial constraint** — regularize `d_gw` (or `zeta_net`)
   against an independent losing-potential signal (Jasechko 2021
   well-vs-stream levels, or water-table-depth attributes) so arid ungauged
   reaches get supervision the discharge loss cannot provide (Nijzink 2018
   multi-source calibration).
3. **Gate leakance on sub-daily forcing** — any promotion of
   `use_leakance` from experimental should require the disaggregation head
   (or native hourly forcing), per H5.
4. *(Dropped)* widening `K_D` alone — superseded per Conclusion 2. Revisit
   only after (1) or (2) restores a gradient path to the losing-stream
   population.

## 6. Raw script output

```
attributes matched for 100.0% of 64892 reaches
aridity vs meanP spearman = -0.84 → aridity is a DRYNESS index

========================================================================
H1 — structural ceiling: can zeta exceed the bar inside the current box?
========================================================================
zeta_max within CURRENT box:  p5=0.001762 p25=0.008541 p50=0.02191 p75=0.05059 p95=0.1553 | frac > 0.01: 71.5%
zeta_max with K_D=0.0001: p5=0.1762 p25=0.8541 p50=2.191 p75=5.059 p95=15.53 | frac > 0.01: 100.0%
utilization zeta/zeta_max:    p5=0.0123 p25=0.02482 p50=0.03359 p75=0.06189 p95=0.1768

  → [REFUTED] H1 structural ceiling: only 71.5% of reaches CAN exceed 0.01 m³/s inside the current box (vs 100.0% at K_D=0.0001); median utilization 0.03

========================================================================
H2 — driving head (depth_mean − d_gw)
========================================================================
depth_mean: p5=0.02683 p25=0.1584 p50=0.3327 p75=0.8283 p95=3.307  |  d_gw: p5=0.08703 p25=0.2274 p50=0.3353 p75=0.3788 p95=0.4414  |  head: p5=-0.2399 p25=-0.109 p50=0.02134 p75=0.5034 p95=2.971
head ≤ 0 (gaining/neutral at the mean): 47.0%   head < 0.1 m: 57.6%

  → [SUPPORTED] H2 driving-head starvation: 57.6% of reaches have <0.1 m mean driving head (47.0% ≤ 0)

========================================================================
H3 — KAN variance collapse (leakance vs routing params, full CONUS)
========================================================================
K_D              median=   -5.999  IQR=0.0003591  IQR/p5-p95=0.366
d_gw             median=   0.2945  IQR=   0.1524  IQR/p5-p95=0.382
leakance_factor  median=   0.3273  IQR=  0.08144  IQR/p5-p95=0.363
n                median=  0.06096  IQR=   0.0283  IQR/p5-p95=0.351
q_spatial        median=   0.3654  IQR=  0.06486  IQR/p5-p95=0.359
x_storage        median=      0.3  IQR=        0  IQR/p5-p95=0.000

spearman corr of leakance params vs attributes (eval network):
  K_D      aridity=+0.61  permeability=+0.27  Porosity=+0.00  log10_uparea=-0.10  meanP=-0.64  meanslope=+0.09
  d_gw     aridity=-0.67  permeability=-0.24  Porosity=+0.00  log10_uparea=-0.02  meanP=+0.71  meanslope=-0.04
  factor   aridity=+0.59  permeability=+0.26  Porosity=+0.02  log10_uparea=-0.09  meanP=-0.62  meanslope=+0.15
  zeta     aridity=-0.22  permeability=-0.08  Porosity=+0.02  log10_uparea=+0.76  meanP=+0.26  meanslope=-0.09

  → [REFUTED] H3 KAN variance collapse: max |spearman| of any leakance param vs any attribute = 0.71 (<0.2 ⇒ no learned spatial structure; ≥0.4 ⇒ clearly attribute-driven)

========================================================================
H4 — stratification by gauged-ness, upstream area, aridity
========================================================================
gauged reaches on eval network: 2698 / 64892
  gauged    median|zeta|=6.706e-03  frac>0.01: 40.2%  median q=7.49
  ungauged  median|zeta|=5.938e-04  frac>0.01: 9.1%  median q=0.759
  dry tercile  median|zeta|=3.439e-04  frac>0.01: 6.3%
  wet tercile  median|zeta|=8.577e-04  frac>0.01: 12.4%
  spearman log|zeta| vs log10_uparea = +0.76

  → [SUPPORTED] H4 gauge bias / gradient starvation: dry/wet median-zeta ratio = 0.40 (physics says dry ≫ wet), zeta–uparea corr +0.76 (zeta tracks river size, not aridity)

========================================================================
H5 — did n / x_storage shift between paired ON/OFF runs?
========================================================================
  hourly Δn         median=+0.001429  IQR=0.01914  median-shift/param-IQR=0.05
  hourly Δx_storage median=+0  IQR=0  median-shift/param-IQR=0.00
  daily  Δn         median=+0.01231  IQR=0.01016  median-shift/param-IQR=0.59
  daily  Δx_storage SKIPPED (not learned in the OFF run — constant)

  → [SUPPORTED] H5 equifinality: routing params shifted materially between ON/OFF (shift > 0.5 IQR) — n/storage absorb what leakance would explain

========================================================================
H6 — |zeta| / q_mean (is the loss non-trivial RELATIVE to local flow?)
========================================================================
|zeta|/q: p5=9.963e-05 p25=0.0003946 p50=0.0008428 p75=0.001894 p95=0.02972
frac loss > 1% of local flow: 8.4%   > 5% (gauge-detectability band): 3.2%

  → [REFUTED] H6 wrong yardstick: 8.4% of reaches lose >1% of local flow (3.2% >5%) — the absolute 0.01 m³/s bar under/over-states the term's activity

========================================================================
H7 — d_gw boundary-pinning where disconnection is plausible (dry reaches)
========================================================================
d_gw within 5% of bounds: floor 0.0%  ceiling 0.0% (overall)
  dry tercile: floor 0.0%  ceiling 0.0%
  wet tercile: floor 0.0%  ceiling 0.0%

  → [REFUTED] H7 model-form error: 0.0% of dry-tercile reaches pin d_gw at a bound — the linear connected-regime law is straining toward the saturating (disconnected) regime

========================================================================
SUMMARY (suggested verdicts — final judgment in the findings doc)
========================================================================
  [REFUTED] H1 structural ceiling: only 71.5% of reaches CAN exceed 0.01 m³/s inside the current box (vs 100.0% at K_D=0.0001); median utilization 0.03
  [SUPPORTED] H2 driving-head starvation: 57.6% of reaches have <0.1 m mean driving head (47.0% ≤ 0)
  [REFUTED] H3 KAN variance collapse: max |spearman| of any leakance param vs any attribute = 0.71 (<0.2 ⇒ no learned spatial structure; ≥0.4 ⇒ clearly attribute-driven)
  [SUPPORTED] H4 gauge bias / gradient starvation: dry/wet median-zeta ratio = 0.40 (physics says dry ≫ wet), zeta–uparea corr +0.76 (zeta tracks river size, not aridity)
  [SUPPORTED] H5 equifinality: routing params shifted materially between ON/OFF (shift > 0.5 IQR) — n/storage absorb what leakance would explain
  [REFUTED] H6 wrong yardstick: 8.4% of reaches lose >1% of local flow (3.2% >5%) — the absolute 0.01 m³/s bar under/over-states the term's activity
  [REFUTED] H7 model-form error: 0.0% of dry-tercile reaches pin d_gw at a bound — the linear connected-regime law is straining toward the saturating (disconnected) regime
```

Note on verdict strings: each hypothesis's detail clause is a fixed template
written for the supported case, so on REFUTED lines the framing can read
backwards — H1's "only 71.5% CAN exceed" (71.5% is the *refuting* fact),
H6's "under/over-states the term's activity" (the finding is that both
yardsticks *agree*), and H7's "…is straining toward the saturating regime"
(with 0.0% pinning the model is NOT straining against the d_gw bounds). The
numbers are authoritative; §3 gives the correct readings.

## 7. Reproduce

```bash
cd ~/projects/ddr && uv run python \
  ~/projects/ddrs/scripts/leakance_diagnosis.py   # defaults point at the four runs
```

Prereqs: the two ON runs' `kan_parameters.nc` must contain the
`depth_mean`/`area_z_mean`/`q_mean` exports (re-eval with the current binary's
`--zeta-output`), and the hourly-OFF run needs a `dump_parameters` pass
(checkpoint base `checkpoints/epoch_5_mb_35/head`).
