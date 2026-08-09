# Building and changing the training gauge population

Written 2026-08-02, the session that produced `gages_2000_area_balanced.csv`.
Covers: how a gauge CSV is constructed from GAGES-II, the filters a gauge must
survive to actually train, the exact regeneration command, and what breaks when
the population changes. Motivating diagnosis:
`/tmp/experiment-handoff-small-basin-domination.md` (small basins dominated
gradient share and pooled median NSE selected for identity routing).

## The one command

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/build_gages_2000_area_balanced.py
```

Deterministic (seed 42, no network access — everything is local). Rewrites
`~/projects/ddr/references/gage_info/gages_2000_area_balanced.csv` and prints
the filter funnel, a coverage-sensitivity table, and the final area histogram.
To vary the recipe, edit the constants at the top of the script
(`REL_TOL`, `COVERAGE_MIN`, `TRAIN_WINDOW`, `EVAL_WINDOW`, `SEED`,
`N_SMALL`, `N_LARGE`) — do not fork the logic.

## Inputs (all local; never fetch from USGS)

| Input | Path | Role |
|---|---|---|
| GAGES-II gauge table | `~/projects/ddr/references/gage_info/GAGES-II.csv` | 8,931 rows, same schema as `gages_3000.csv` incl. `COMID`, `ABS_DIFF`, `DA_VALID` |
| Observations | `/mnt/ssd1/data/icechunk/usgs_daily_observations` | 9,067 gauges × 14,610 days (1980-01-01..2019-12-31), `gage_id` zero-padded strings, m³/s, NaN = missing |
| Gauge subgraphs | `~/projects/ddr/data/merit_gages_conus_adjacency.zarr` | 8,945 groups keyed by zero-padded STAID; `order` length 1 ⇔ headwater |

## The filter funnel (why each stage exists)

A gauge in the CSV is useless unless it survives *every* downstream filter in
ddrs, so the builder applies them all up front — the CSV population equals the
training population by construction (the lesson of the phantom-zero baseline
incident, traps.md T4).

1. **`DA_VALID` must be relative, not absolute.** The GAGES-II/gages_3000
   precomputed column is `ABS_DIFF <= COMID_UNITAREA_SQKM` — effectively an
   absolute area tolerance, which deletes large basins for ~2% relative
   disagreement while admitting small basins at >6,000%. Recompute as
   `ABS_DIFF / DRAIN_SQKM <= 0.10`. Effect on GAGES-II: 7,919 → 5,528
   (recovers 433 large, drops 2,824 mostly-small). ddrs only *reads* the
   column (`src/data/store/gage_csv.rs:62`); the fix must happen at CSV
   construction.
2. **Observation coverage in BOTH configured windows.** Check the windows
   training actually slices (`config/merit_training.yaml`): train
   1981-10-01..1995-09-30, eval 1995-10-01..2010-09-30 — not round calendar
   years. Bar used: ≥ 80% non-NaN days in each window. Sensitivity on the
   5,528 pool: any-data 3,636 · ≥50% 2,948 · ≥80% 2,512 · ≥90% 2,392 ·
   100% 2,168.
3. **Non-headwater subgraph exists.** Mirror `GageSubgraph::is_headwater`
   (`src/data/store/zarr.rs:128`): zarr group present AND `order` length > 1.
   Single-divide gauges have empty upstream sets → all-zero summed-Q'
   predictions. Dropped 97 of 2,512 → eligible pool 2,415.
4. **Area-balanced subsample.** Keep ALL basins ≥ 5,000 km² (they carry the
   routing signal); top up [1,000, 5,000) to reach the ≥1,000 km² target;
   random-draw the < 1,000 km² stratum. Keep small basins deliberately — they
   teach the identity-routing regime; the goal is reducing their gradient
   share, not removing them. 2026-08-02 result: 582 + 418 + 841 = **1,841**
   (45.7% / 54.3% either side of 1,000 km²). The small stratum was the
   binding constraint (only 841 eligible), so the set is 1,841 not 2,000.

## Output contract

Same schema `gage_csv.rs` reads (`STAID` zero-padded, `DA_VALID` literal
`True`, STANAME quoted) plus audit columns `REL_DIFF`, `COV_TRAIN`,
`COV_EVAL` — the serde reader ignores unknown headers. Full derivation is
also documented in `~/projects/ddr/references/gage_info/README.md`; the CSV
is committed in the **ddr** repo (05796ba), the builder in **ddrs**.

## Consequences of changing the population — read before repointing

- **The cached summed-Q' baseline is invalidated.** The gages path is in the
  baseline cache key; `ddrs plan` recomputes it. This is correct — but it
  means **no metric from runs on the old population is comparable**, in
  either direction. Establish the new baseline before any improvement claim
  (research-status.md §Gauge-set definitions has the standing warning).
- **Point `data_sources.gages` at the new CSV** (via `ddrs sources save/use`
  or editing `ddrs.yaml`); adjacency stores need no rebuild — subgraphs are
  keyed by STAID and already exist for every selected gauge.
- **Verify with the run log**, not assumptions: expect
  `gages_adjacency filter: kept 1841 (dropped 0 missing, 0 headwater)` —
  nonzero drops mean the CSV and the filters disagree and the builder's
  assumptions no longer hold.
- **Spot-check recovered gauges before trusting a training run.** The 202
  final gauges recovered by the relative criterion (median 14,189 km²) have
  relative DA disagreement up to 10% by construction; compare a sample's
  observed hydrographs against summed upstream Q' (this remains open as of
  2026-08-02).
