# Precip-driven mass-preserving daily→hourly disaggregation — Design

**Date:** 2026-06-22
**Branch:** `hourly-forcings` (merged from `origin/routing`, which carries the Exp-3 `DisaggHead`)
**Status:** approved, implementing

## Motivation

The 2026-06-19 routing journal pinned three layered findings:

1. The **loss** was never the limiter (L1, NNSE-KGE, component-KGE → same flat result).
2. The flat `repeat-24` daily→hourly interpolation (`icechunk.rs::daily_to_hourly_trim`)
   was the **gradient limiter** — routing's within-day effect lands in the daily-mean
   loss's null-space, so `∂loss/∂(routing params) ≈ 0` (Muskingum X stuck at its 0.246
   init across three runs).
3. Exp-3's learnable mass-preserving `DisaggHead` fixed (2) — loss descended and X moved
   for the first time — but **overfit**, because nothing supervises or grounds the
   sub-daily shape. Its own doc comment names the gap: *"There is no hourly precip
   globally ... so this is a learned prior, not recovery of the true sub-daily signal."*

Real hourly AORC precipitation is exactly that missing physical signal. This design feeds
it into the `DisaggHead` as the **dominant within-day input**, so the disaggregation shape
is grounded in real meteorology instead of invented from daily Q′ alone.

## Goal

A mass-preserving daily→hourly disaggregation head whose within-day shape is **driven by
real hourly AORC precip**, that (1) generalizes well enough on **daily** held-out metrics to
beat the summed-Q′ baseline (NSE 0.689 / KGE 0.723), and (3) produces physically-plausible
sub-daily hydrographs. Mass preserved exactly: disaggregated hourly Q′ daily-averages back
to the trusted daily Q′ (per catchment per day).

## Decisions (from brainstorming)

- **Architecture:** precip-driven softmax head. NN(full `[d-1,d,d+1]` hourly precip = 72
  values + daily-Q 3-tap window + static attrs) → 24 softmax weights → `daily_Q · 24 · shape`.
- **Mass preservation:** softmax-over-24h × 24, unchanged from Exp-3 (already unit-tested).
- **Precip normalization: pre-head, in the data-batching layer** (not inside the head).
- **No smoothness regularizer** (revisit only if overfitting reappears).
- **Loss: L1** (do not reuse kge/nnse) — clean comparison against the journal's runs.
- **Integration: extend `DisaggHead` in place** — reuse all `KanHead.disagg` /
  checkpoint / eval / forward / mass-preservation plumbing; opt-in; precip-off path is
  byte-identical to today.
- **Validation: daily USGS** (inputs are daily; daily-mean comparison). Hourly USGS IV is
  an optional future phase.
- **New data-source group: `conus-hourly`.**

## Data: AORC store (verified)

`/mnt/ssd1/data/aorc/merit_unit_catchments.zarr` — zarr **v3** group:

- `total_precipitation`: `(290878 catchments, 359424 hours)` f32, **catchment-major**,
  **mm/hr**, chunks `(290878, 48)`, fill `0.0`.
- `gauge_id`: `<U8` strings = MERIT COMIDs (`'71022453'`), parse to `Comid(i64)`.
- `date`: hourly `datetime64`, `1980-01-01T00:00 → 2020-12-31T23:00` = **14,976 days**,
  byte-aligned with the streamflow Q′ axis. Hour rows `[t·24 : (t+1)·24]` are day `t`
  (days since 1980-01-01).
- 290,878 catchments ⊂ 346,321 CONUS MERIT reaches (~55k gap → 0.0 fill).

## Components

1. **`src/data/store/zarr_aorc.rs` — `AorcPrecipStore`** (new). Mirrors
   `GlobalStreamflowStore` but: zarr-v3 single store, **catchment-major** `(COMID, time)`,
   string `gauge_id`→`Comid`. `read_window_hourly(window_start, n_hourly, &comids) ->
   (n_hourly, N)`: slices hour rows from `window_start·24`, gathers COMID rows, transposes
   to time-major. Missing COMIDs → `0.0`. Mirrors the daily/test-window read entry points.

2. **`src/data/store/mod.rs`** — export `AorcPrecipStore`. Single format → no enum wrapper.

3. **`src/config.rs`** — `DataSources.aorc_precip: Option<PathBuf>` (default None).
   `KanHeadConfig.disagg_use_precip: bool` (default false), threaded through
   `kan_config`. Validation: `disagg_use_precip` ⇒ `aorc_precip` must be set.

4. **`src/data/dataset.rs`** — `MeritGagesDataset` holds `Option<Arc<AorcPrecipStore>>`.
   `RoutingBatch`/`RoutingTensors` gain `precip_hourly` (`(n_hourly, N)`; empty `(0,N)`
   when precip off). Both collate sites read precip for the same window as `q_prime`,
   then **normalize pre-head**: `z = standardize_per_reach(log1p(precip))` over the
   window; all-dry/constant columns → zeros. `to_tensors` lifts it.

5. **`src/nn/disagg_head.rs`** — `DisaggHead` gains `use_precip: bool`; input layer width
   `f = 3 + (72 if use_precip) + (F if use_attributes)`. `forward(daily_q, attrs,
   precip_hourly, n_hourly)`: when `use_precip`, reshape `precip_hourly (n_hourly,N)` →
   `(d_use,24,N)`, gather `[d-1,d,d+1]` edge-clamped → `(d_use·N, 72)` in the **same
   row-major `day·N+reach`** order as the log-Q feats, concat. Mass-preservation algebra
   unchanged. `use_precip=false` → byte-identical to today.

6. **`src/training/forward.rs`** — pass `tensors.precip_hourly` into `disagg.forward(...)`.

7. **`config/sources/conus-hourly.yaml`** — CONUS group + `aorc_precip` path.

## Mass-preservation invariant

`mean_{k∈0..24} hourly[d·24+k, r] = daily_q[d, r]` exactly, at any weights (softmax sums
to 1, scaled ×24). The precip extension changes only the logits, never this algebra — the
existing `mass_is_conserved_*` tests remain the guard.

## Concerns

- **Precip ≠ runoff; Q′ is already dHBV-UH-routed.** Precip timing may not match the
  already-smoothed daily Q′'s true sub-daily shape; the MLP learns a transfer, but if the
  UH already encodes the within-day response, precip adds noise. *Mitigation:* compare vs
  precip-off disagg + early-stop sweep.
- **Structural ceiling (journal finding 3) is not removed by better forcing timing.**
  Precip may unstick the gradient yet still not beat baseline on held-out daily metrics. A
  null result is itself the answer the journal asked for.
- **Coverage gap (55k reaches)** → 0.0 fill → flat precip → daily-Q fallback. Log covered fraction.
- **Read cost:** catchment-major `(290878,48)` chunks; a window read decompresses full-width
  chunks. Matches `zarr_qprime` behavior; benchmark, cache if hot.

## Assumptions (verified)

- AORC time axis = streamflow axis (both 1980-01-01 daily-aligned) → `window_start` indexes
  both, no date crosswalk.
- `gauge_id` parses to MERIT COMID.
- CONUS-only; global runs leave `aorc_precip` unset → daily-Q disagg / repeat-24 fallback.

## Blast radius

Additive, opt-in. New: `zarr_aorc.rs`, `config/sources/conus-hourly.yaml`. Edited:
`store/mod.rs`, `config.rs`, `dataset.rs`, `disagg_head.rs`, `kan_head.rs`, `forward.rs`.
Invariants 1–7 untouched; `precip` off ⇒ byte-identical path (`compare_ddr_sandbox` +
KAN-head parity tests stay green).

## Phasing

1. `AorcPrecipStore` + reader tests (alignment, COMID gather, gap fill).
2. Batch plumbing (`precip_hourly` carry + pre-head normalization).
3. `DisaggHead` precip extension + mass-preservation/parity tests.
4. Config + `conus-hourly` group + `forward.rs` wire-up.
5. Smoke test (tiny CPU run) → full train-and-test on `conus-hourly` (L1) → report.
