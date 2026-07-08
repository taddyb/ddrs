# DDR Q' store contract

The interface between runoff producers (neural-hydrology LSTMs, dHBV2, …)
and ddrs routing. Any store meeting this contract can be validated and
registered with `ddrs import <store> --name <group>` and then routed.

The reference producer is
`~/projects/neuralhydrology/examples/merit_hydro/forward_merit.py`
(`--mode daily|hourly`), which runs a trained NH model over the MERIT unit
catchments and writes a conforming store. Producers that RUN neural
hydrology live in the NH repo; everything downstream of the written store
lives here.

## Contract

- An **icechunk repository** (`main` branch, local filesystem), root group.
- One data variable **`Qr(divide_id, time)`**, dtype **float32**, attr
  `units: m^3/s`.
- `Qr` values are the **local lateral inflow per MERIT unit catchment** —
  no upstream accumulation (routing does that).
- `divide_id`: int64 MERIT COMIDs.
- `time`: int64, CF-encoded as either
  - `days since YYYY-MM-DD[ HH:MM:SS]` — a **daily** store, or
  - `hours since YYYY-MM-DD[ HH:MM:SS]` — an **hourly** store.
  The axis must be contiguous (no gaps); an hourly axis must start at
  hour 0 of a calendar day. Any other units string is rejected at open.
- Values strictly positive: producers floor NaN/negative predictions to
  `1e-6` (as `forward_merit.py::mm_day_to_m3s` does).
- COMIDs **absent** from the store are ddrs's concern, not the producer's:
  reads fill them with `0.001` m³/s, never error.

## How ddrs reads each resolution

| ddrs read | daily store | hourly store |
|---|---|---|
| `read_window` (training) | repeat-24 + trailing-day trim (or disagg head) | native hourly slice |
| `read_test_window` (eval) | repeat-24, `n_days*24` | native hourly slice |
| `read_window_daily` (baseline, disagg input) | direct | mean of each 24-h block |

`kan_head.disaggregation` is **rejected** when the streamflow source is
hourly-native — disaggregating an already-hourly signal is a config
contradiction (`src/data/dataset.rs::validate_disagg_vs_resolution`).

## Conforming stores (2026-07-01)

| Store (`/mnt/ssd1/data/icechunk/`) | resolution | range | divides |
|---|---|---|---|
| `daily_lstm_merit_unit_catchments.ic` | daily | 1981-01-01 → 2020-12-30 | 288,421 |
| `hourly_lstm_merit_unit_catchments.ic` | hourly | 1981-01-01 → 2020-12-31T23 | 197,088 |
| `daily_dhbv2_merit_unit_catchments.ic` | daily | 1980-01-01 → 2020-12-30 | 288,421 |
| `merit_dhbv2_UH_retrospective.ic` | daily | 1980-01-01 → 2020-12-31 | 197,088 |

Note the hourly store starts **1981-01-01** (1980 was LSTM warmup): an
experiment window reaching into 1980 hard-errors rather than clamping.

## Onboarding a new NH dataset

1. In `~/projects/neuralhydrology`, write/adapt a forward script that emits
   a conforming store (start from `forward_merit.py`).
2. `ddrs import <store> --dry-run` — validates the contract + prints a
   COMID-coverage report.
3. `ddrs import <store> --name <group>` — registers it under
   `config/sources/<group>.yaml`.
4. `ddrs sources use <group> && ddrs plan && ddrs run --workflow train`.

Design history: `docs/superpowers/specs/2026-07-01-nh-qprime-import-design.md`.
