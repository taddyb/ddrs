# Juniata single-catchment sample

ddrs on one basin: the Juniata River at Newport, PA (USGS 01567000,
8,657 km², 213 MERIT reaches). Everything needed is in `data/` — no HPC,
S3, CUDA, or external stores. This is the ddrs mirror of DDR's
`examples/juniata` (DeepGroundwater/ddr PR #193); the two run on the
byte-identical bundle with the same hyperparameters, so results are
directly comparable across implementations.

## Quickstart

From the **repo root** (the config's data paths are repo-root-relative):

    cargo build --release --bin ddrs
    target/release/ddrs --config examples/juniata/ddrs.yaml plan
    target/release/ddrs --config examples/juniata/ddrs.yaml run --workflow train-and-test --backend cpu

`plan` validates the bundle and prints the summed-Q' baseline (no learned
parameters — the bar routing has to beat). `run` trains the KAN head for
30 optimizer steps (one random 90-day window per epoch on a single gauge),
then evaluates over water years 1996–2010. The workspace lands beside the
config at `examples/juniata/.ddrs/` (gitignored); inspect a finished run
with:

    target/release/ddrs --config examples/juniata/ddrs.yaml show <run-id>

30 steps demonstrates learning and physically plausible parameters, not a
converged CONUS-grade model. The full train-and-test workflow takes well
under a minute on CPU.

Reference results (CPU, 30 epochs, test 1995-10-01 – 2010-09-30):

| | NSE | KGE |
|---|---|---|
| ddrs routed | 0.790 | 0.881 |
| DDR (Python) routed | 0.784 | 0.877 |
| summed-Q' baseline | 0.695 | 0.819 |

The ddrs baseline reproduces DDR's (0.695 / 0.820) to rounding — the two
implementations read the same bundle through independent readers. The
routed numbers agree to well within sampling noise: the residual comes
only from the 30 random training windows being drawn from different RNG
streams (torch's global RNG vs ChaCha12), so exact-match is impossible by
construction. With 30 noisy single-gauge steps, expect seed-to-seed
spread in the routed metrics: ddrs seeds {42, 7, 123, 2026} scored NSE
0.790–0.800 / KGE 0.881–0.886.

Both examples run the corrected physics (trapezoid-exact celerity β,
Cunge-matched X, own-reach gauge readout): DDR removed its legacy path in
PR #192, and ddrs deprecated `params.ddr_match` on 2026-08-19 — the
default is now `false` (corrected), so this config needs no flag at all.

## What's in the bundle

| File | Contents |
|---|---|
| `juniata_qprime.ic` | icechunk, `Qr(divide_id, time)` daily m³/s, 213 divides, 1980–2010 (dHBV2 UH retrospective) |
| `juniata_obs.ic` | icechunk, `streamflow(gage_id, time)` daily m³/s, USGS 01567000, 1980–2010 |
| `juniata_attributes.nc` | 10 KAN input attributes per COMID |
| `juniata_conus_adjacency.zarr` | binsparse COO subgraph + `length_m`, `slope`, `order` (compact 0..212 indexing, topologically ordered, lower-triangular) |
| `juniata_gages_adjacency.zarr` | single-gage COO group, same schema as the CONUS store |
| `juniata_gage.csv` | one-row gage metadata (gages_3000 schema) |
| `statistics/…json` | attribute normalization statistics over the 213 catchments |

One deviation from DDR's bundle: `statistics/` is **committed** here.
DDR computes it on first run (`set_statistics`); ddrs never recomputes
statistics (`src/data/statistics.rs`), so the JSON ships with the bundle.

Both icechunk stores start 1980-01-01 (required by the readers'
positional time origin). The bundle itself is regenerated from the CONUS
stores by DDR's `examples/juniata/extract_bundle.py` (maintainer-run);
this copy is taken verbatim from the ddr repo — regenerate there, then
re-copy `data/` and re-copy the generated statistics JSON.

## Gate

`tests/juniata_bundle.rs` asserts the bundle contract through the real
ddrs readers (213 reaches / 212 edges, lower-triangular adjacency,
1980-01-01 time origins, all 10 KAN inputs present). It runs in the
default `cargo test` sweep; if you touch the bundle or `src/data/`, make
sure it still passes.
