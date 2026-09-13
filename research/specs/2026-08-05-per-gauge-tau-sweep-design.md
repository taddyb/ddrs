# Per-gauge tau sweep — design

- **Date:** 2026-08-05
- **Motivation:** area-balanced eval (run `2026-08-05T04-58-58Z-conus-experimental-train-and-test`,
  epoch 30, 1,841 gauges) shows ddrs LOSES to the summed-Q' baseline in small
  basins: median NSE 0.630 vs 0.720 in the <1,000 km² bin (841 gauges), while
  winning above 5,000 km². Hypothesis: a sub-daily timing/alignment error that
  aliases across the daily pooling boundary, worst for flashy small basins.
- **The suspect:** `params.tau` (default 3) — a single CONUS-wide phase offset in
  `tau_trim_and_downsample` (`src/training/loss.rs:25-46`, slice
  `[13+tau : -11+tau]` then daily area-pool). Inherited unvetted from DDR
  (`configs.py:116`: "handle double routing and timezone differences"). USGS
  daily values are local-day means; CONUS spans four time zones; one scalar
  cannot be right everywhere. With `tau = 3` the daily bin boundary sits at
  16:00 UTC, while CONUS local midnight is 05:00–08:00 UTC.

## Hypothesis (pre-registered)

**H-TAU:** the small-basin NSE deficit is substantially a pooling-phase
(timing) error: per-gauge choice of tau recovers most of the gap.

Verdict states: SUPPORTED / REFUTED / INCONCLUSIVE only.

## Method

Two phases. Phase 1 (pilot, tonight) validates the method on WY1996; Phase 2
(full sweep, after tuning) runs the same analysis on the full eval window with
the pre-registered gate.

### Instrument

1. **One eval run, full window** (1995/10/01–2010/09/30), legacy `eval` binary,
   checkpoint `epoch_30_mb_1` of the run above, its own `config.yaml` snapshot
   (tau = 3), with `DDRS_HOURLY_DUMP=<path>` — the opt-in diagnostic in
   `src/training/eval.rs` that writes the PRE-TRIM hourly series
   (n_gauges × n_hours raw f32 + `.json` dims sidecar, ~1 GB). No retraining.
2. **Offline sweep** (`scripts/tau_sweep.py`, NumPy only): for each
   `tau ∈ {0..23}` reconstruct daily predictions as block means of hours
   `[13+tau+24i, 13+tau+24(i+1))`, scored against obs day `i+1` (the day-0 drop
   convention). `tau = 3` reproduces shipped behavior exactly; 0..23 covers the
   full 24-hour phase cycle. The trimmed window is divisible by 24, so block
   mean equals the area pool.
3. Per gauge and per tau: NSE against the eval zarr's `observations` (NaN
   masked, ≥100 valid days required). Pilot restricts scored days to WY1996
   (1995/10/01–1996/09/30).
4. Baseline comparison from the run's own `baseline/` arrays (identical
   1,841-gauge population), sliced to the same days.
5. Covariates joined from `gages_2000_area_balanced.csv`: `DRAIN_SQKM`,
   `LNG_GAGE`.

### Method-validation gate (pilot, Phase 1)

Boolean, all three must hold:

1. Offline `tau = 3` daily reconstruction matches the eval run's own
   `predictions.zarr` to < 1e-3 relative (f32 floor).
2. Recomputed full-window median NSE at `tau = 3` matches the eval's reported
   median to < 0.001.
3. Sweep produces per-gauge NSE(tau) curves for ≥ 95% of the 1,841 gauges.

If this gate fails, fix the method before interpreting any number.

### Decision gate (full sweep, Phase 2 — user-selected)

**GO for building per-gauge local-time pooling iff per-gauge-best tau lifts the
<1,000 km² median NSE from 0.630 to ≥ 0.720 (the summed-Q' baseline).**
Beating the baseline is the strictest of the considered bars; a large-but-
insufficient gain reads as "timing real but not the whole story" (INCONCLUSIVE
for H-TAU as the *dominant* cause).

Secondary (diagnostic, not gating): Spearman correlation of per-gauge best tau
with gauge longitude — the timezone-mechanism fingerprint; NSE(tau) medians
stratified by the five drainage-area bins.

### Known bias, accepted for the pilot

Choosing tau per gauge on the same days it is scored is in-sample (1 free
integer parameter per gauge); the pilot number is a headroom CEILING. The
Phase 2 protocol (split-sample selection vs scoring) is deliberately left open
for tomorrow's tuning session.

## Artifacts

| Path | Content |
|---|---|
| `output/tau_sweep/eval_full.zarr` | fresh eval daily predictions + obs |
| `output/tau_sweep/hourly_full.f32` (+`.json`) | pre-trim hourly dump |
| `output/tau_sweep/nse_by_tau_wy1996.csv` | gauge × tau NSE matrix (pilot) |
| `output/tau_sweep/best_tau_wy1996.csv` | per-gauge best tau + covariates |
| `output/tau_sweep/summary_wy1996.md` | gate numbers, bin medians, correlation |
| `output/tau_sweep/*.png` | NSE-vs-tau by area bin, best-tau histogram, best-tau vs longitude |

## Concerns / assumptions

- **Assumption:** the eval zarr's `observations` rows are aligned with
  `predictions` rows (probe convention: pred bin i ↔ obs day i+1). Validation
  gate item 2 tests this indirectly.
- **Could go wrong:** rerunning eval from the same checkpoint may not bit-match
  the 04-58-58Z run (GPU nondeterminism); the method gate therefore compares
  against the NEW run's own zarr, not the old one.
- **Could go wrong:** WY1996 may be a hydrologically unusual year; pilot numbers
  are for method-shakeout and tuning, not conclusions.
- **Binary provenance:** the dump code is uncommitted at design time; it is
  committed with this spec, and the binary is built from that commit
  (2026-07-01 stale-binary lesson).
- **Why not inverse routing:** recovering hourly timing from daily gauges is
  ill-posed both spatially (a gauge observes a network sum — the leakance
  NO-GO mechanism) and temporally (24 unknowns/day vs 1 obs/day). The
  deterministic per-gauge pooling window is the cheap falsifiable step first.
