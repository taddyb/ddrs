# NH → icechunk → route: Q' store contract, hourly-native reading, and `ddrs import`

**Date:** 2026-07-01
**Status:** Approved (brainstorming session)
**Scope decision trail:** full pipeline (NH outputs → icechunk distributed Q' → route);
both existing LSTM stores in scope including hourly-native routing; NH-side forward
step stays in `~/projects/neuralhydrology` (it runs NH); ddrs gains an `import`
module; success = plumbing verified + short smoke train on each store;
hourly reading via resolution sniffing (approach A).

## Problem

ddr trained routing on top of neural hydrology (NH) LSTM outputs by an offline
pattern: forward the trained LSTM once over ~288K MERIT unit catchments
(`~/projects/neuralhydrology/examples/merit_hydro/forward_merit.py`), write
`Qr(divide_id, time)` in m³/s to an icechunk store, and point
`data_sources.streamflow` at it. ddrs must support the same workflow so each
new NH dataset becomes routable.

What exists today — `/mnt/ssd1/data/icechunk/` holds **all** unit-catchment
forward outputs, and every one of them is an import target:

| Store | Size | Resolution | Producer |
|---|---|---|---|
| `daily_lstm_merit_unit_catchments.ic` | 26 GB | daily | CudaLSTM run `merit_hydro_daily_cudalstm_1004_213620`, `forward_merit.py --mode daily` |
| `hourly_lstm_merit_unit_catchments.ic` | 257 GB | hourly | MTS-LSTM run `merit_hydro_mtslstm_1104_163854`, `forward_merit.py --mode hourly` |
| `daily_dhbv2_merit_unit_catchments.ic` | 17 GB | daily | dHBV2 unit-catchment forward |
| `merit_dhbv2_UH_retrospective.ic` | 9.5 GB | daily | dHBV2 UH retrospective (current ddrs training source) |

Gaps in ddrs:

1. `StreamflowStore` (icechunk reader) assumes a **daily** CF time axis and
   upsamples via repeat-24 or the disaggregation head. The hourly store has an
   hourly time axis and no read path.
2. No first-class way to validate + register a new Q' store; today it means
   hand-editing `ddrs.yaml` or a source-group file.
3. The producer/consumer interface is implicit (whatever `forward_merit.py`
   happens to write).

## Design

### 1. Store contract (`docs/nh-qprime-store-contract.md`)

A new doc codifies what any NH forward script must emit for ddrs to route it:

- icechunk repo, `main` branch, root group, one variable **`Qr(divide_id, time)`**,
  f32, attr `units: m^3/s`.
- `divide_id`: int64 MERIT COMIDs. Values are the **local** lateral inflow per
  unit catchment (no upstream accumulation — routing does that).
- `time`: CF-encoded int, **`days since …`** (daily) or **`hours since …`**
  (hourly), contiguous, no gaps.
- Values strictly positive; producer floors NaN/negatives to 1e-6
  (as `forward_merit.py` already does).
- COMIDs absent from the store are handled by ddrs (0.001 fill at read),
  never an error.

The contract is written from `forward_merit.py`'s output, so both in-scope
stores conform and **the NH repo needs no changes** for them. A future NH
dataset = adapt/write a forward script in the NH repo to emit a conforming
store, then `ddrs import` it here.

### 2. Resolution-aware icechunk reader (`src/data/store/icechunk.rs`)

`StreamflowStore` gains `resolution: data::dates::Frequency`, sniffed at open
from the CF units string:

- `days since …` → `Daily` (today's behavior, unchanged)
- `hours since …` → `Hourly` (new)
- anything else → hard `DataError` carrying the store path and units string.
  No silent fallback.

The three-method contract is unchanged for callers (`StreamflowSource` enum
and dispatch in `src/data/store/mod.rs` untouched apart from plumbing):

| method | daily store | hourly store (new) |
|---|---|---|
| `read_window_daily(start, n_days, comids)` | unchanged | read `n_days*24` rows, mean over each 24-block → `(n_days, N)` |
| `read_window(&RhoWindow, comids)` | unchanged (`daily_to_hourly_trim` repeat-24) | direct slice of `window.hourly_range()` → `(n_hourly, N)` |
| `read_test_window(&TestWindow, comids)` | unchanged | direct `n_days*24` contiguous rows |

Rationale for the 24-block mean: Q' is a rate (m³/s), so the daily value is
the day's average flow; this keeps the **summed-Q' baseline** and any other
`read_window_daily` caller working on hourly stores unmodified.

Time offsets are computed in hours for hourly stores. A requested window
falling outside the store's time range is a **hard error** — the hourly store
starts 1981-01-01, and an experiment configured from 1980 must fail loudly,
not clamp. If the current daily path silently tolerates out-of-range windows,
that latent bug is fixed as part of this work.

**Guardrail:** `MeritGagesDataset::open` (`src/data/dataset.rs`) rejects
`kan_head.disaggregation` when the streamflow source is hourly-native —
disaggregating an already-hourly signal is a config contradiction. Same
enforcement style as the existing missing-`aorc_precip` error
(`dataset.rs:355`). `flow_scale` applies unchanged (per-column constant,
resolution-independent).

**Regression guard:** the daily path must stay byte-identical; a
`leakance_off_parity`-style test enforces it.

### 3. `ddrs import` module (`src/cli/import.rs`)

```
ddrs import <store-path> --name <group-name> [--dry-run]
```

1. **Open & detect** — `StreamflowSource::open` (existing icechunk-vs-zarr
   sniff), then report format, detected resolution, time range, basin count.
2. **Validate the contract** — `Qr` present with expected dims/dtype; time
   axis parses and is contiguous; sample read of a few COMIDs is finite and
   positive.
3. **Coverage report** — intersect store `divide_id`s against the workspace
   adjacency when `.ddrs/` exists: "N of M fabric COMIDs covered; X% get the
   0.001 fill." No workspace → skip with a warning, don't fail.
4. **Register** — copy the current `ddrs.yaml` `data_sources:` block, swap
   `streamflow:` to the store path, save as `config/sources/<name>.yaml` via
   the existing `sources::save` machinery, re-lock. `--dry-run` stops after
   step 3.

Precondition: an existing `ddrs.yaml` supplies the non-streamflow source keys
(same precondition as `ddrs sources save`). Post-import flow:
`ddrs sources use <name> && ddrs plan && ddrs run --workflow train`.

### 4. Verification (success criterion: plumbing + short smoke train)

- **Verify-first step:** before building the sniff, check the real stores'
  actual CF time encodings (a one-line xarray inspection under the ddr venv —
  was classifier-blocked during brainstorming). If xarray encoded the hourly
  axis as something other than `hours since` (e.g. minutes/seconds since),
  the sniff grammar adapts before anything else is built.
- **Tests:** CF `hours since` parsing; hourly read shapes + alignment (hour
  *h* of day *d* returns the stored value) against a small fixture store;
  24-block-mean correctness; daily-path byte-parity regression; the
  disagg-rejection error.
- **Import validation:** `ddrs import --dry-run` must pass on all four
  unit-catchment stores in `/mnt/ssd1/data/icechunk/` (the two LSTM stores,
  `daily_dhbv2_merit_unit_catchments.ic`, and
  `merit_dhbv2_UH_retrospective.ic` — the latter doubles as a known-good
  control since ddrs already trains on it).
- **Smoke trains:** `ddrs import` both LSTM stores, then a few-epoch small-batch
  train on `daily-lstm` (unchanged path, new store) and on `hourly-lstm`
  (new path): finite loss, directory-style checkpoints, and a log line
  confirming `resolution: hourly` actually executed (stale-binary lesson —
  reinstall or `cargo run` the working-tree binary).

### 5. Concerns and assumptions

**Concerns (what could go wrong, why):**

- *Time-encoding surprise* — xarray auto-picks CF units; the hourly axis may
  not literally be `hours since`. Low risk; the verify-first step catches it
  before any reader code is written.
- *Hourly read cost* — 24× the rows per window from a 257 GB store (chunks
  ~3080 basins × 11232 hours); random rho-windows could make epochs
  I/O-bound. The smoke train measures wall-clock; chunk-aligned reads are a
  possible follow-up, explicitly out of scope now.
- *Store starts 1981* — out-of-range windows must hard-error (see §2); this
  may surface (and fix) a latent tolerance in the daily path.
- *Coverage gaps* — the hourly store was filtered to the UH store's
  divide_ids; 0.001 fill for missing reaches is existing intended behavior,
  but the import report makes the magnitude visible instead of silent.

**Assumptions (and why):**

- The two existing stores conform to the contract — it was written from their
  producer's code.
- Source groups are the right registration target — they are already how
  datasets are switched in ddrs.
- The NH-side forward script needs no changes for the in-scope datasets;
  generalizing it for future forcing families is NH-repo work, out of scope.

**Benefit:** every NH-trained model becomes a routable ddrs dataset via one
command, and hourly-native routing is unlocked — real MTS-LSTM hourly forcing
instead of the disagg approximation — slotting a third forcing option into
the leakance × forcing experiment line.

## Out of scope

- Generalizing `forward_merit.py` for new forcing families (NH-repo work).
- Chunk-aligned / performance-tuned hourly reads (follow-up if smoke train
  shows I/O-bound epochs).
- Full science-quality training runs and baseline comparisons (experiments,
  not plumbing).
- Any change to the routing core, sparse solver, or KAN head (invariants 1–7
  in CLAUDE.md untouched).
