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
  `units: m^3/s`. The dtype is not checked at `open` — `open` only
  resolves the array handle (`src/data/store/icechunk.rs:239-241`). It is
  enforced at the first *read*, where `retrieve_array_subset::<Vec<f32>>`
  errors on a non-f32 array. `ddrs import` triggers that read itself via
  `sample_read`, so a dtype violation surfaces at import rather than
  mid-training.
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

## Conforming stores (2026-07-01, `daily_dhbv2`/`daily_dhbv_aorc2f` re-verified 2026-07-15 after a 2026-07-10 rebuild)

| Store (`/mnt/ssd1/data/icechunk/`) | resolution | range | divides |
|---|---|---|---|
| `daily_lstm_merit_unit_catchments.ic` | daily | 1981-01-01 → 2020-12-30 | 288,421 |
| `hourly_lstm_merit_unit_catchments.ic` | hourly | 1981-01-01 → 2020-12-31T23 | 197,088 |
| `daily_dhbv2_merit_unit_catchments.ic` | daily | 1980-01-01 → 2020-12-31 | 197,088 |
| `daily_dhbv_aorc2f_merit_unit_catchments.ic` | daily | 1980-01-01 → 2020-12-31 | 197,088 |
| `merit_dhbv2_UH_retrospective.ic` | daily | 1980-01-01 → 2020-12-31 | 197,088 |

`daily_dhbv2_merit_unit_catchments.ic` was rebuilt 2026-07-10 (previously
288,421 divides as of 2026-07-01); `ddrs import --dry-run` is the source of
truth for the current 197,088/197,084-fabric-coverage numbers, not this
table.

Note the hourly store starts **1981-01-01** (1980 was LSTM warmup): an
experiment window reaching into 1980 hard-errors rather than clamping.

## Onboarding a new NH dataset

1. In `~/projects/neuralhydrology`, write/adapt a forward script that emits
   a conforming store (start from `forward_merit.py`).
2. `ddrs import <store> --dry-run` — validates the contract + prints a
   COMID-coverage report.
3. `ddrs import <store> --name <group>` — registers it under
   `config/sources/<group>.yaml`. Add `--force` to overwrite an existing
   group of the same name; without it, a name collision is an error
   (`src/cli/import.rs:36-37`).
4. `ddrs sources use <group> && ddrs plan && ddrs run --workflow train`.

### Only icechunk stores are contract-checked

`ddrs import` validates the full contract **only for icechunk stores**.
When `StreamflowSource::open` sniffs a global zarr-v2 store instead, the
command prints

```text
note        detailed contract validation and coverage are icechunk-only;
            open succeeded, which exercises the same reader the training
            loop uses
```

and skips the units check, the `sample_read`, and the COMID-coverage
report entirely (`src/cli/import.rs:99-106`). A successful import of a
zarr-v2 store therefore means "the reader opened it", not "it satisfies
this contract" — the checks in the sections above are unenforced for that
format.

Design history: `docs/superpowers/specs/2026-07-01-nh-qprime-import-design.md`.
