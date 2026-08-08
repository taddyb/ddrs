# Per-gauge tau sweep — WY1996 pilot findings

- **Spec:** `docs/superpowers/specs/2026-08-05-per-gauge-tau-sweep-design.md`
- **Script:** `scripts/tau_sweep.py` (offline, on the `DDRS_HOURLY_DUMP` from one eval run)
- **Checkpoint:** `2026-08-05T04-58-58Z-conus-experimental-train-and-test/checkpoints/epoch_30_mb_1` (epoch 30, area-balanced 1,841 gauges)
- **Binary provenance:** `eval` built from commit `76a4020` (dump diagnostic committed with the spec); run log `output/tau_sweep/eval.log`.
- **Window:** eval full 1995/10/01–2010/09/30; pilot analysis restricted to WY1996 (1995/10/01–1996/09/30, 365 days). **Pilot numbers are single-year and (for per-gauge best) in-sample; they are method-validation and headroom estimates, not conclusions.**

**One-line verdict: H-TAU SUPPORTED at pilot strength — the shipped `tau = 3` is
grossly mis-set; the NSE(tau) optimum sits at tau ≈ 14–19 (daily pooling bin
starting 03:00–08:00 UTC next day, i.e. CONUS local midnight), and a single
global tau ≈ 16–19 already ties or beats the summed-Q' baseline in the
small-basin bins that motivated the experiment.**

## 1. Pre-registered hypotheses and gates

- **H-TAU:** the small-basin NSE deficit is substantially a pooling-phase
  (timing) error; per-gauge tau recovers most of the gap.
- **Method gate (pilot):** (1) offline tau=3 reconstruction matches the eval's
  own daily zarr < 1e-3 rel; (2) recomputed median NSE matches < 0.001;
  (3) ≥95% of gauges get full NSE(tau) curves.
- **Decision gate (Phase 2, user-selected, not yet evaluated):** per-gauge best
  tau lifts <1,000 km² median NSE to ≥ the baseline, full window, with an
  out-of-sample selection protocol (protocol deliberately open).

## 2. Method

One eval of the epoch-30 checkpoint over the full eval window with
`DDRS_HOURLY_DUMP` writing the pre-trim hourly series (1841 × 131,496 f32,
0.97 GB). Offline, for each tau ∈ {0..23}: daily prediction bin *i* = block
mean of hours `[13+tau+24i, 13+tau+24(i+1))`, scored against obs day *i+1*
(reproduces `tau_trim_and_downsample` exactly at tau = 3; the trimmed window is
divisible by 24 so block mean = area pool). Per-gauge NSE on WY1996 days,
≥100 valid days required. Baseline = the run's own `baseline/` arrays (same
1,841-gauge population), same days.

## 3. Results

Method gate: **all three PASS** (max rel 7.06e-4; medians 0.6206 vs 0.6205;
98.7% coverage).

Median NSE, WY1996, by drainage area:

| bin (km²) | n | tau=3 (shipped) | best single global tau | per-gauge best (in-sample ceiling) | Σ Q' baseline |
|---|---|---|---|---|---|
| 0–1,000 | 841 | 0.553 | **0.677** (tau=16) | 0.699 | 0.674 |
| 1,000–5,000 | 418 | 0.592 | **0.702** (tau=23) | 0.722 | 0.660 |
| 5,000–10,000 | 244 | 0.544 | **0.668** (tau=22) | 0.671 | 0.512 |
| 10,000–30,000 | 266 | 0.479 | **0.548** (tau=14) | 0.566 | 0.379 |
| 30,000–50,000 | 72 | −0.021 | −0.021 (flat) | −0.019 | −0.334 |
| ALL | 1,841 | 0.546 | **0.660** (tau=19) | — | 0.605 |

Key observations:

1. **The NSE(tau) curve rises monotonically from tau=0 and plateaus at
   tau ≈ 14–19 in every bin below 30,000 km²** (`nse_vs_tau_by_area.png`).
   Window start hour is `13+tau`, so the plateau is 03:00–08:00 UTC (next
   day) — precisely the CONUS local-midnight band (ET midnight = 05 UTC,
   PT midnight = 08 UTC). The spec's arithmetic concern is confirmed: shipped
   tau=3 pools over a bin starting 16:00 UTC, ~11–16 h out of phase with the
   local observation day.
2. **A single global tau fixes most of it.** +0.114 median NSE overall with
   zero per-gauge freedom (no selection bias applies to this number, though it
   is still single-year).
3. **Per-gauge freedom adds only ~+0.02 over the global optimum** — and the
   longitude fingerprint is weak (Spearman −0.129 on the 1,353 gauges improved
   > 0.01). The broad plateau swallows the 3-hour timezone spread; per-gauge
   tau is second-order compared to just fixing the global phase.
4. **Best-tau histogram is edge-heavy** (661 gauges at tau=23, 232 at tau=0),
   so the true optimum for a subpopulation lies outside {0..23} under the fixed
   day mapping — the sweep range needs a ±1-day mapping extension in Phase 2.
5. Above 30,000 km² the curve is flat: multi-day response integrates out the
   phase, and both model and baseline are poor there (regulation, n=72).

## 4. Conclusions

- The small-basin deficit that motivated this experiment is at least
  substantially a **pooling-phase error in `tau`**, not a routing-physics
  failure: at global tau=16 the <1,000 km² bin ties the baseline (0.677 vs
  0.674) in the pilot, and per-gauge best exceeds it (0.699).
- **Training implication (potentially the bigger one):** training also scores
  through `tau_trim_and_downsample` at tau=3, so every gradient the KAN head
  has ever received was computed against misaligned observations. The head may
  be learning spurious lag/attenuation to compensate — a candidate mechanism
  for the over-attenuation signature. A retrain at corrected tau is the test.
- Caveats: single water year; per-gauge numbers in-sample; global-tau numbers
  free of selection but still one year.

## 5. Next steps (tomorrow's tuning session)

1. Extend the sweep with a ±1-day mapping shift to un-pin the tau=0/23 edges.
2. Full-window sweep + split-sample protocol (select on 1995–2003, score on
   2003–2010) to evaluate the Phase 2 gate honestly.
3. Decide the fix tier: global tau ≈ 16–19 (needs `tau_trim_and_downsample` to
   accept tau > 11 — the current slice arithmetic caps tau at 11), per-timezone,
   or per-gauge local-midnight window.
4. Retrain at corrected tau and re-run the area-balanced eval — tests the
   training-misalignment mechanism.

## 5a. Adversarial review corrections (2026-08-06, Fable subagent — verdict: sound-with-corrections)

The arithmetic and the +0.114 selection-free gain reproduce exactly. Three
interpretive claims are corrected in place:

1. **§3 obs 1 "precisely the CONUS local-midnight band" — RETRACTED as
   over-claimed.** This run had `disaggregation.enabled: false` (verified in
   the config snapshot), so the hourly signal was flat repeat-24: UTC-day means
   capture a median **97.6%** of hourly variance, and the pooled series at any
   tau is reproduced by a two-point blend of adjacent UTC-day means with median
   R² 0.994–0.997. The sweep therefore measures a **day pairing plus blend
   weight (~half-day resolution), not sub-daily phase**. Hour-scale readings
   (tau=16 "Eastern midnight" vs tau=19 "Pacific midnight") are below the
   method's effective resolution. Corrected headline: *the shipped
   (tau=3, obs day i+1) mapping pools a window ~half a local day early; any
   tau in 14–19 fixes the day-boundary blend.*
2. **Phase semantics / sign convention.** Window-start offset relative to the
   scored day's UTC midnight is **(tau − 11) h**: tau=3 → −8 h (13/24 of the
   pooled mass from the wrong local day); tau=16 → +5 h (= ET midnight);
   tau=19 → +8 h (= PT midnight); tau=23 → +12 h. **Larger optimal tau ⇒ the
   model hydrograph is LATE relative to observations.**
3. **§3 obs 3 (longitude) — the timezone fingerprint is absent-to-contradicted,
   not merely "weak".** The −0.129 Spearman was computed on a 48%-edge-pinned
   set; with censored gauges excluded the sign FLIPS to +0.174. The **robust
   covariate is drainage area**: Spearman(best_tau, log10 area) = +0.164
   (uncensored, p=1e-5) to +0.341 (censored incl.), surviving partialling on
   longitude (+0.208). Larger basins prefer later windows — an accumulated-lag
   / travel-time signature. So "not a routing-physics failure" (§4) is too
   strong: phase error is the dominant term, not the only term.
4. **Censoring is asymmetric and real:** of 661 tau=23 pins, 596 are genuinely
   improved (optima beyond +12 h, needing the day-(i+2) mapping); of 232 tau=0
   pins only 56 are improved (flat-curve noise). The ±1-day extension is
   necessary and predominantly on the late side. Curve sharpness tracks
   sub-daily structure (Spearman +0.701), as the blend mechanism predicts.
5. Minor script defect: no-curve gauges are back-filled with `best_tau =
   tau_shipped` in the CSV, slightly contaminating the histogram/correlations.

Phase-2 design implication: select a **day-mapping × blend-weight** per gauge
(split-sample), do NOT commit to an hour-precision "local-midnight" fix tier on
this evidence, freeze the selection protocol before any retrain (the head has
trained 30 epochs against a ~half-day-early target and has plausibly learned
compensating lag), and re-run the sweep on a disagg-ON or hourly-native run to
test whether any hour-scale tau signal exists at all.

## 5b. External mechanistic prior (2026-08-06, `/tmp/handoff-aorc-usgs-recording-times.md`)

Independent of the sweep, the recording conventions predict the offset a priori:

- **USGS daily values are local-STANDARD-time midnight-to-midnight, year-round
  (no DST)** — authoritative per
  https://waterdata.usgs.gov/statistics-documentation/.
- **AORC forcing is UTC-hourly**, and the AORC-driven Q' stores (incl. this
  run's `daily_dhbv2_distributed_aorc2f`) define "day t" as UTC 24-hour blocks.
- Predicted misalignment: +5 h (EST) to +8 h (PST). In tau units
  (window offset = tau − 11 h) that predicts **optimal tau ∈ [16, 19]** —
  exactly the measured plateau. What the review demoted to
  "consistent-with" now has a documented mechanism.
- Pipeline verification (this session): `src/data/` contains **no timezone
  logic anywhere**; all stores are indexed positionally on their native axes.
  `params.tau` is the only alignment knob in the system.
- Because USGS uses LST year-round, the correct per-gauge correction is a
  **fixed deterministic offset from the gauge's standard-time zone** — no DST
  seasonality to model.
- Open tension with §5a: the mechanism predicts western gauges (more negative
  longitude) prefer LARGER tau, i.e. a negative lng correlation; the
  uncensored pilot showed +0.174. Candidate explanations: half-day sweep
  resolution blurring a 3-h span, the area confound, or geographic structure
  in the censored tau=23 tail. Unresolved; the interpolation arms (sharper
  curves) are the discriminating instrument.
- Open question inherited from the handoff: whether every Q' store shares the
  UTC-day convention (dHBV2-UH, daily-LSTM, hourly-LSTM vs the AORC2F pair).
  If they differ, tau is per-STORE as well as per-gauge, and cross-store
  parameter comparisons (the AGU H069 framing) inherit the bias. Check each
  store's CF axis + forcing provenance before the next cross-store run.

## 5c. Interpolation arms (2026-08-06): nearest vs linear vs quadratic, gages_3000

Three evals of the same epoch-30 checkpoint on the standard **2,365-gauge**
population (gages_3000 after filters), full window, differing ONLY in
`DDRS_QPRIME_INTERP` (commit `e4fb66d`); WY1996 sweep per arm, all method
gates PASS (99.9% curve coverage). Driver: `scripts/run_tau_interp_arms.sh`;
overlay plot `output/tau_sweep/interp_arms_nse_vs_tau.png`.

**Verdict: the tau mis-set is confirmed on the benchmark population and is
NOT an artifact of step-function upsampling — smoother q' input neither
sharpens nor shifts the optimum. Interpolation is not the fix; the day
mapping is.**

| arm | full-window median NSE @ tau=3 | WY1996 median @ tau=3 | WY1996 argmax | WY1996 max | curve range (median) |
|---|---|---|---|---|---|
| nearest | 0.6426 | 0.578 | tau=20 | 0.6997 | 0.104 |
| linear | 0.6496 | 0.589 | tau=19 | 0.6983 | 0.094 |
| quadratic | 0.6347 | 0.573 | tau=18 | 0.6943 | 0.104 |

1. **Interpolation buys almost nothing, and nothing at the optimum.** Linear
   gains +0.011 at the mis-set tau=3 (smearing partially absorbs the
   misalignment) but at each arm's own optimum the three arms converge within
   0.005, ordered nearest ≥ linear ≥ quadratic — consistent with the predicted
   peak attenuation of the smoothing kernels. Quadratic is strictly worse than
   nearest at tau=3.
2. **Curves do not sharpen** (linear is slightly FLATTER), so the half-day
   resolution limit of §5a is a property of the daily-information content, not
   of the step discontinuities. Sub-daily structure cannot be conjured by
   interpolation; only a disagg-ON or hourly-native store can supply it.
3. **The optimum sits at tau=18–20 on this population, at/beyond the PT edge
   of the LST band [16,19], with 707–728 gauges (~30%) still pinned at
   tau=23** (real optima beyond +12 h; only ~60 at tau=0). The timezone
   convention alone under-predicts the shift.
4. **Correlations replicate across all arms:** best_tau vs log10(area)
   +0.16 to +0.19 (uncensored), vs longitude +0.13 to +0.19 (uncensored —
   still the WRONG sign for the timezone mechanism, in every arm).
5. **Small basins (<1,000 km², n=1,267): single global tau=19 scores 0.674 vs
   baseline 0.645** (WY1996) — the "beat the baseline" bar is cleared on this
   population too, again with zero per-gauge freedom.
6. **Emerging synthesis:** optimal shift ≈ (LST-vs-UTC convention offset,
   +5..8 h) + (an area-growing lag term). The area correlation, the beyond-band
   optimum, and the late-side censored tail all point at extra model lag on top
   of the convention offset — the leading candidate being **double routing**
   (MC travel time stacked on whatever routing/UH the Q' store already embeds
   to place flow at its outlet; DDR's own tau docstring says "handle double
   routing and timezone differences"). Discriminating test: repeat the sweep on
   a UH-free vs UH-embedded store pair.

## 5d. Sample calculation: tau_g = 11 + tz(gauge) + c·A^b (2026-08-06)

Fitted on the nearest-arm WY1996 curves (2,365 gauges), tz from longitude
(midpoints of standard meridians → 5/6/7/8 h), objective = median NSE over the
per-gauge integer tau_g, grid over (c, b). In-sample (2 free params).

| scheme | median NSE | note |
|---|---|---|
| tau=3 (shipped) | 0.5780 | |
| tau=18 / 19 / 20 constant | 0.6990 / 0.6992 / 0.6997 | |
| tz only (tau = 11+tz) | 0.6966 | WORSE than constant 19 |
| **formula b=0.30, c=0.28** | **0.7018** | joint grid (b=0.40, c=0.12) ties at 0.7019 — b unidentified |
| per-gauge best (ceiling) | 0.7228 | in-sample selection |

Per-bin: formula ties constants below 5,000 km² (0.6728 vs 0.6736 at tau=19),
gains +0.005 in 10,000–30,000 km² (0.7257 vs 0.7203). Fitted lag term:
1.1 h @ 100 km², 2.2 h @ 1,000, 4.4 h @ 10,000, 6.2 h @ 30,000 — magnitudes
consistent with Allen et al. (2018) celerity-based travel times. Median
predicted tau_g: 19 (p10 18, p90 22), 1% clip at 23.

Diagnostics: the area term DOES absorb the area signal (residual-vs-log-area
Spearman drops +0.16 → −0.10), but residual-vs-tz is **−0.452** — per-gauge
optima do not track the timezone term at this resolution (echoes §5c's
wrong-sign longitude), and tz-only underperforms a flat constant. Vs constant
tau=19 the formula moves 76% of gauges and improves 922 vs worsens 872
(median Δ +0.0001) — a coin flip per gauge, small net win from the mid-size
bins.

**Reading:** the half-day blend resolution (§5a/§5c) leaves hour-scale
refinements below the instrument's discrimination; a constant tau ≈ 19–20
captures essentially all recoverable skill on this population (0.6997 vs
0.7018 formula vs 0.7228 unreachable ceiling). The formula is physically
defensible and never hurts materially — a fine choice for the retrain — but
the decision between it and a constant should be made split-sample in
Phase 2, not on these in-sample numbers.

## 5e. Cross-source arms (2026-08-07): is the lag a property of the store?

Same epoch-30 checkpoint, same 2,365-gauge network, only
`data_sources.streamflow` swapped (`config/experiments/tau_src_*.yaml`,
`scripts/run_tau_source_arms.sh`; this set ran on cuda, before the CPU
policy). All three method gates PASS on every arm (recon match, NSE match,
99.9% curve coverage). WY1996 sweep, median over 2,365 gauges:

| store | resolution | full-window med NSE @ tau=3 | best const tau | med @ best | per-gauge best-tau median | % optima at tau=0 / 23 |
|---|---|---|---|---|---|---|
| aorc2f distributed (ref) | Daily | 0.6426 | 20 | 0.6997 | 18 | 11 / 33 |
| UH retrospective | Daily | 0.6375 | 21 | 0.7011 | 18 | 12 / 34 |
| daily LSTM | Daily | 0.5562 | 17 | 0.6135 | 18 | 15 / 32 |
| hourly LSTM (native) | Hourly | 0.5316 | 19 | 0.5515 | 19 | 15 / 34 |
| **aorc2f lumped** | Daily | 0.5103 | **3** | 0.4588* | **1** | **47 / 6** |

\* pilot-window median at its own optimum; levels are not comparable across
stores (the head was trained on aorc2f distributed only), but optimum
LOCATIONS and curve shapes are.

**Four of five stores replicate the tau 17–21 optimum.** UH retrospective,
daily LSTM, and hourly LSTM all peak within 3 h of the reference despite
being entirely different models of runoff generation. The per-gauge best-tau
median is 18–19 on all four, and the censored-tail fractions (pile-ups at
tau=0 and tau=23) are near-identical. Whatever produces the lag, it is not a
quirk of the aorc2f-distributed store: it is shared by every store that uses
the standard daily convention, consistent with the UTC-vs-local-standard-time
mismatch as the dominant term. The replication cannot, however, apportion the
shared lag between the day convention and the common MC routing (the
double-routing candidate): the routing head, parameters, and network are
common-mode across all five arms, and the optima sit at or beyond the
predicted LST band [16, 19]. The residual beyond-band lag remains
unattributed (see §5f for the discriminating measurement).

**The aorc2f lumped store is the outlier and the exception that probes the
rule.** Its median curve is flat over tau 0–3 and then falls monotonically;
47% of per-gauge optima sit at the tau=0 edge (left-censored). Extended
sweep (§5f): the median curve peaks at tau=+3 with the per-gauge median at
−1 (50.1% below 0). Its shipped-tau full-window median (0.5103) is already
near its own optimum. Obs-free cross-correlation of routed hydrographs
(§5f) shows the lumped arm LEADS the reference by ~23 h (median), so the
store's data are aligned about one day differently from every other store.
The CF-day-convention candidate is REFUTED: both aorc2f stores' icechunk
time metadata are byte-identical (`days since 1980-01-01`,
proleptic_gregorian, 14,976 steps). The shift is in the data the lumped
pipeline wrote (day-indexing off-by-one or different event-day assignment),
and the ~6 h residual between the waveform shift (~23 h) and the NSE-optimum
shift (~17 h) implies the lumped timing content also differs beyond a pure
relabeling. Do not use this store in timing-sensitive comparisons until the
pipeline-side indexing is resolved.

**Hourly-native arm: no sharpening, and no timezone fingerprint either.**
The hourly LSTM curve is the flattest of the four lagged stores (gain from
tau=3 to optimum +0.073 vs +0.122 for the reference), not sharper, so real
sub-daily structure did not turn the sweep into an hour-resolution
instrument. The flatness is not a skill-floor artifact: at matched skill
levels the hourly arm is still flatter (§5f). The improved-subset longitude
correlation (−0.220) initially looked like the timezone-predicted sign, but
§5f shows it is a censoring artifact: on uncensored interior optima the
sign flips to +0.075 (wrong sign, weakest of the four lagged arms), and the
lumped arm, where the timezone mechanism has no standing, shows −0.197 on
its own improved subset. The timezone fingerprint remains
absent-to-contradicted even with native sub-daily structure, an informative
negative result.

Plot: `output/tau_sweep/cross_source_nse_vs_tau.png`. Raw per-arm outputs in
`output/tau_sweep/src_{uh_retro,daily_lstm,hourly_lstm,aorc2f_lumped}/`.

## 5f. Adversarial review of §5e (2026-08-07, Fable subagent — verdict: sound-with-corrections)

Independent read-only review; all §5e corrections above were folded in from
it. Verdicts: claim "4/5 replicate ⇒ shared convention" SOUND-WITH-
CORRECTIONS (replication real, attribution overreached); lumped-outlier
claim SOUND-WITH-CORRECTIONS (strengthened, CF candidate refuted); "no
sharpening" SOUND; the longitude half UNSUPPORTED (cut); levels-not-
comparable SOUND. New measurements it contributed:

- **Obs-free cross-correlation of routed hydrographs** (500-gauge sample,
  WY1996, lags ±48 h; observations never enter): median lag vs reference is
  UH retro +0 h (79% within 6 h), daily LSTM −1 h, hourly LSTM −2 h,
  **lumped −23 h** (IQR −31 to −19, 84% at or below −12 h). This refutes the
  checkpoint-mismatch explanation for the lumped outlier twice over: the two
  LSTM arms are at least as mismatched to the trained head yet show zero
  shift, and the −23 h appears with no observations involved.
- **Extended sweep tau −13..47** (recomputed from raw dumps): the reference
  constant-tau curve peaks interior at 20 (0.6997, declining to 0.6853 at 24
  and 0.6373 at 30), so the constant-tau conclusion does not depend on the
  missing ±1-day extension. Per gauge, though, 32.6% of reference optima are
  genuinely beyond tau=23 and 11.2% below 0 (median 18, p10 −5, p90 40) —
  the per-gauge tail structure is real, not an artifact of the 0..23 window.
- **Censoring asymmetry supports the lumped reading:** the reference's
  tau=0 pile is 78% flat-curve noise (only 22% improved) while the lumped
  tau=0 pile is 64% genuine improvement — the left pile is signal for the
  lumped arm, unlike its mirror image in the reference.
- **Flatness is not a floor effect:** within-arm Spearman(curve range, curve
  max) ≈ 0 to +0.13 in every arm; level-matched gauges (|Δmax NSE| < 0.05,
  n=409) still leave the hourly arm flatter (median range 0.074 vs 0.091).
- **Mechanical comparability PASS:** identical gauge ID vectors, identical
  `nse_baseline`, identical obs arrays, has_curve 2362/2365 with the same 3
  NaN gauges in every arm. The known no-curve backfill defect
  (`scripts/tau_sweep.py:143`) touches only those 3 gauges here.
- Interior-optima area correlation is unstable across source arms (+0.089
  reference, −0.118 hourly-native) — weaker than the §5c interp-arm numbers.

**Single most informative next measurement (proposed):** sweep tau on the
routing-free summed upstream q' (the baseline construction, repeat-24, same
gauges and obs, fully offline from existing stores). All five arms share the
MC routing, so its lag contribution is invisible to the cross-source design.
If the no-routing optimum also sits at 19–21, the entire lag is the
store/obs day convention and the double-routing candidate dies; if it sits
near 14–16, the gap directly quantifies the MC network's added travel time.

## 5g. The discriminator: tau sweep on routing-free summed q' (2026-08-07)

The §5f-proposed measurement, run entirely offline from existing data: the
baseline's summed upstream daily q' (cache
`.ddrs/runs/2026-07-30T00-24-24Z-train-and-test/baseline`, full coverage of
all 2,365 eval gauges) repeat-24'd into a synthetic hourly dump in the SAME
phase as the routed arms' disaggregation input, then swept with the
identical `scripts/tau_sweep.py` (all gates PASS) plus an extended sweep
tau −13..47. Construction validated: sweep tau=11 is algebraically the
standard day-aligned baseline scoring, and its per-gauge NSE reproduces the
cached baseline NSE (median |diff| 0.0003). WY1996, medians:

| area bin (km²) | summed-q' opt tau | routed opt tau (uncensored) | summed early by | routed late by | routing-added delay |
|---|---|---|---|---|---|
| 0–1,000 | 9 | 19 | 2 h | 8 h | 10 h |
| 1,000–5,000 | 3 | 19 | 8 h | 8 h | 16 h |
| 5,000–10,000 | −3 | 25 | 14 h | 14 h | 28 h |
| 10,000–30,000 | −8 | 30 | 19 h | 19 h | 38 h |
| global median | 6 (0.6731) | 20 (0.6997) | 5 h | 9 h | 14 h |

("early/late by" = |tau_opt − 11|; the 30,000–50,000 bin, n=12, is too
noisy to read. Per-gauge uncensored summed-q' best tau: median 6, p10 −10,
p90 20, only 6.5% at the −13 edge.)

**Neither §5f-anticipated outcome occurred, and the measurement is the more
decisive for it.** The no-routing optimum is not 19–21 (all convention) and
not 14–16 (convention plus routing residual): it is **6**, below the
day-aligned point. Three conclusions follow:

1. **The UTC-vs-LST convention story (§5b) is REFUTED as the dominant
   term.** A timing-correct hydrograph scored against LST-labeled daily obs
   should show optimum tau ≈ 16–19 even without routing. The smallest
   basins, where travel time is minimal, sit at tau=9, and the area trend
   extrapolates to ≈ 10–11 at zero area: the convention offset is ≈ 0–2 h,
   not 5–8. This finally explains why the longitude fingerprint failed in
   every arm and subset (§5a, §5c, §5f): there was no timezone signal to
   find. Whatever conventions the stores and obs use, they net out to
   near-UTC-day alignment.
2. **The summed q' leads the gauges by an area-growing travel time** (2 h
   at <1,000 km² to ~19 h at 10,000–30,000 km²), exactly the unmodeled
   network travel time the routing exists to supply. Corollary: day-aligned
   scoring understates the baseline's skill in large basins (5,000–10,000:
   0.636 day-aligned → 0.755 at its optimum). Baseline comparisons at
   large basins should keep this in mind.
3. **The MC routing over-delays by almost exactly 2× the required travel
   time.** Bin by bin (≥1,000 km²), routed lateness equals summed-q'
   earliness: the routing added twice the delay the gap required. This is
   the "double routing" of DDR's own tau docstring, now measured: the q'
   stores already route runoff to the unit-catchment outlet (dHBV2's UH),
   and the MC network then adds what amounts to the full travel time again.
   The tau 18–20 optimum of every routed arm is COMPENSATION for this
   over-delay, not a data-convention fix.

**Routing still earns its keep once both sides are timing-corrected:** at
per-bin optima the routed model beats the summed q' everywhere that
matters: +0.022 (<1,000 km²), +0.013, +0.019, +0.051 (10,000–30,000 km²).
The value added is real; it is currently masked at shipped tau=3 and
partially masked at any constant tau by the over-delay.

**Reframing for the fix tier:** a constant tau ≈ 19–20 remains the correct
empirical patch for the current checkpoint, but the root cause is now in
the routing timing (double-carried travel time, and possibly the slow
trained celerity: median Manning's n 0.130 vs reference 0.05), not in the
data pipeline. A retrain at corrected tau tests the patch; the deeper fix
candidates (injection geometry, celerity prior) are a separate experiment.

Artifacts: `output/tau_sweep/summed_qprime/` (synthetic dump, sweep
outputs, `extended_curves.npy`),
`output/tau_sweep/g3000_nearest/extended_curves.npy`,
`output/tau_sweep/summed_qprime_vs_routed_tau.png`.

## 6. Raw output

`output/tau_sweep/`: `summary_wy1996.md`, `nse_by_tau_wy1996.csv` (1841×24),
`best_tau_wy1996.csv`, `nse_vs_tau_by_area.png`, `best_tau_hist.png`,
`best_tau_vs_longitude.png`, `eval.log`, `hourly_full.f32(.json)`,
`eval_full.zarr`.

## 7. Reproduce

```bash
# eval + dump (~1.3 h GPU) — binary from commit 76a4020
RUN=.ddrs/runs/2026-08-05T04-58-58Z-conus-experimental-train-and-test
DDRS_HOURLY_DUMP=$PWD/output/tau_sweep/hourly_full.f32 target/release/eval \
  --config $RUN/config.yaml --checkpoint $RUN/checkpoints/epoch_30_mb_1 \
  --output $PWD/output/tau_sweep/eval_full.zarr

# offline sweep
uv run --with "zarr>=3" --with numpy --with pandas --with matplotlib --with tabulate \
  python scripts/tau_sweep.py \
  --dump output/tau_sweep/hourly_full.f32 --zarr output/tau_sweep/eval_full.zarr \
  --baseline-dir $RUN/baseline \
  --gages-csv ~/projects/ddr/references/gage_info/gages_2000_area_balanced.csv \
  --out-dir output/tau_sweep --pilot-start 1995-10-01 --pilot-end 1996-09-30
```
