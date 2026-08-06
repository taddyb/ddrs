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
