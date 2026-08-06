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
