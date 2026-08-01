# The summed Q' baseline

Before a trained KAN routing model can claim it is "earning its keep,"
ddrs computes a deliberately dumb reference to beat: the **summed Q'
baseline**. For each evaluation gauge it sums the daily lateral inflow
(`Qr`, "Q prime") of every upstream divide — no routing, no learned
parameters, no Muskingum-Cunge solver — and scores that sum against USGS
daily observations. If the trained model's median NSE does not clear this
number, the routing is not adding value, and the place to look is the
training loss curve and KAN-head gradient stats, not the sparse solver.

The implementation lives in `src/baseline/` and ports
`~/projects/ddr/scripts/summed_q_prime.py`. It is computed by `ddrs plan`
and reused by `ddrs run --workflow train-and-test`, with a
content-addressed cache so that re-planning after a config tweak is
instant.

## What it is

The baseline answers one question per gauge: *if we did nothing but add
up the runoff generated upstream, how well would that match the gauge's
observed streamflow?* The pipeline in
`src/baseline/summed_q_prime.rs::compute` is six steps:

1. Open the CONUS adjacency store, the per-gauge adjacency store, the
   observations store, and the streamflow store — `ConusAdjacencyStore`,
   `GagesAdjacencyStore`, `ObservationsStore::open`, and
   `StreamflowSource::open` (`src/baseline/summed_q_prime.rs:24-26, 90-95,
   124`). The last two are the **format-sniffing enums** from
   `src/data/store/mod.rs`, not concrete store types: `ObservationsStore`
   is `Usgs(UsgsObservationsStore) | Global(GlobalObservationsStore)` and
   `StreamflowSource` is `Icechunk(StreamflowStore) |
   GlobalZarr(GlobalStreamflowStore)`. The baseline therefore works on
   CONUS icechunk and global zarr-v2 sources alike, with no branch of its
   own.
2. Filter the gauges CSV down to the scorable population (see
   [The gauge filter](#the-gauge-filter), below), then for each surviving
   gauge derive its upstream COMID set from the gauge's COO indices via
   `subgraph.upstream_comids(&conus)`.
3. Bulk-read daily `Qr` for the **union** of all upstream divides across
   all gauges, in one `read_window_daily` call. `compute` logs the actual
   shape first — `summed Q': {gauges} gauges × {divides} unique upstream
   divides × {days} days` — which is the number to trust; the design-time
   estimate was ~17k divides over a 15-year window
   (`docs/superpowers/plans/2026-06-02-summed-q-prime.md` §Concerns 5,
   never re-measured after the 2026-07-29 population change).
4. For each gauge, NaN-skipping `nansum` of `Qr` across that gauge's
   upstream slice gives a per-day predicted streamflow.
5. Bulk-read the matching daily observations via the same
   `ObservationsStore`'s `read_window_daily`.
6. `Metrics::compute` produces NSE, RMSE, KGE, bias, FHV, and FLV per
   gauge.

The work is split cleanly so the math is unit-testable without icechunk:
`compute` opens the stores and drives the reads, while the pure reducer
`assemble_from_arrays` operates on already-loaded `Array2<f32>` arrays.
The prediction for gauge $g$ on day $t$ is simply

$$
\hat{Q}_{g,t} \;=\; \sum_{c \,\in\, \mathrm{upstream}(g)} \;\mathbb{1}\!\left[Q_{c,t} \text{ finite}\right]\, Q_{c,t}
$$

where `Comid`s missing from the loaded `Qr` columns or carrying a NaN are
silently skipped. The result is bundled into a `SummedQPrime`:

```rust
pub struct SummedQPrime {
    pub predictions: Array2<f32>,   // (n_gauges, n_days), m³/s
    pub observations: Array2<f32>,  // (n_gauges, n_days), m³/s, NaN where missing
    pub gage_ids: Vec<Staid>,       // gauges surviving the three filters below
    pub time_range_daily: Vec<NaiveDate>,
    pub metrics: Metrics,
}
```

The eval window comes from the **testing-mode** config's
`experiment.start_time` / `experiment.end_time` (`%Y/%m/%d`), parsed
inclusive of both endpoints to match pandas
`date_range(inclusive="both")`. An inverted window
(`end < start`) is rejected with `BaselineError::InvertedWindow`.

### The gauge filter

`valid_gauges` (`src/baseline/summed_q_prime.rs:145-176`) applies **three**
filters, in order, preserving gages-CSV order:

1. **No subgraph in `gages_adjacency`** — `gages_adj.get(s).is_some()`.
   Gauges listed in the CSV that the adjacency build never produced a
   subgraph for are dropped silently (mirrors DDR's
   `valid_gauges_mask = np.isin(...)`).
2. **Single-divide / headwater gauges** — `GageSubgraph::is_headwater()`,
   i.e. the subgraph has **zero edges** (`indices_0.is_empty()`,
   `src/data/store/zarr.rs:128-130`). Dropped, with a
   `summed Q' gauge filter: dropped N headwater` line on stderr.
3. **No observation series** — `observations.contains(staid)`. Reads
   hard-error on missing STAIDs, and global gage CSVs list a few dozen
   gauges the observation product lacks. Dropped with a per-gauge
   `warning: gauge <id> has no observation series; skipping`.

Filter 2 landed 2026-07-28/29 and is the load-bearing one. A zero-edge
subgraph has an **empty** `upstream_comids()` set, so the NaN-skipping sum
in step 4 produced exactly `0.0` for every day — a phantom prediction
scored against real observations. On the CONUS gage list that was **513 of
3,211 gauges (16%)** at median NSE **−0.305**, dragging the baseline median
NSE from **0.315** down to **0.142**. Training's own dataset filter had
always dropped these gauges, so the trained median was computed on a
different (2,365-gauge) population — the comparison manufactured a win out
of nothing. Post-fix the baseline population is **2,698** gauges
(3,211 − 513).

> **The generalized rule: the baseline must score the same gauge
> population training uses.** Skip filtered gauges; never impute a value
> for them. Any new filter added to `src/data/dataset.rs` needs a matching
> skip here, or the two medians stop being comparable. Diagnostic: count
> all-zero rows in `<workspace>/baselines/<key>/predictions.f32` — a
> nonzero count on a post-fix binary is a new bug.

Because this changed the *computation* and not just the inputs, the cache
key carries an algorithm-version salt so stale caches recompute rather
than silently persisting the phantom zeros — see
[The cache key](#the-cache-key).

### The six metrics

`Metrics::compute` (in `src/training/metrics.rs`) returns one value per
gauge for each of:

| Metric | Definition |
|---|---|
| **NSE** | Nash–Sutcliffe efficiency, $1 - \frac{\sum (\hat{Q}-Q)^2}{\sum (Q-\bar{Q})^2}$ |
| **RMSE** | Root-mean-square error of the paired daily series |
| **KGE** | Kling–Gupta efficiency (correlation / bias / variability) |
| **Bias** | $\mathrm{mean}(\hat{Q} - Q)$ |
| **FHV (%)** | Flow-duration high-volume bias: sort each series independently, take the top 2% slice (`round(0.98·N)..N`), $100 \cdot \frac{\sum(\hat{Q}-Q)}{\sum Q}$ |
| **FLV (%)** | Flow-duration low-volume bias: same but the bottom 30% slice (`0..round(0.30·N)`) |

FHV and FLV are *not* timestep-paired — each series is sorted on its own
before slicing. A gauge with fewer than roughly 50 finite paired days
yields `NaN` for FHV/FLV, because the 2% / 30% slice indices round to 0
or `N` and the slice is empty. Those NaNs are expected and handled
gracefully throughout (see [NaN handling](#nan-handling), below).

## How to use it

You do not invoke the baseline directly — it rides along with the
standard lifecycle commands. See [Running the code](../usage/running.md)
for the full `init → plan → run` flow.

```bash
# `ddrs plan` computes the baseline as a side effect and prints the table.
ddrs plan --workflow train-and-test

# `ddrs run --workflow train-and-test` reuses the cached baseline and
# copies it into the run directory.
ddrs run --workflow train-and-test
```

After `ddrs plan` resolves and validates your config, it computes the
summed Q' baseline and prints a DDR-parity summary table to stdout via
`src/baseline/print.rs::print_metrics_summary`:

```text
================================================================================
                            SUMMED Q' METRICS SUMMARY
================================================================================
Total Gauges Evaluated: 482
--------------------------------------------------------------------------------
METRIC          MEDIAN      MEAN       Q25       Q75   VALID
--------------------------------------------------------------------------------
Bias             0.000     0.000     0.000     0.000     482
FLV (%)          0.00      0.00      0.00      0.00      460
FHV (%)          0.00      0.00      0.00      0.00      460
KGE              0.000     0.000     0.000     0.000     482
NSE              0.000     0.000     0.000     0.000     482
================================================================================
```

(The numbers above are illustrative; the layout, column set, and per-row
`VALID` count — the number of finite, non-NaN gauges for that metric —
are exactly what the renderer emits.) The summary stats are computed over
the finite subset only; a metric that is NaN for every gauge renders
`VALID = 0`.

The median NSE printed here is the number to beat. After
`ddrs run --workflow train-and-test` finishes its Phase-2 testing, it
prints the trained model's own `median NSE (finite only)` and
`median KGE (finite only)` — compare the two.

### Where the baseline lands in a run

`ddrs run --workflow train-and-test` re-loads the cached baseline and
copies its three files into the run directory
(`copy_baseline_into_run_dir` in `src/cli/run.rs`), so the artifacts
travel with the manifest:

```text
.ddrs/runs/<id>/baseline/
├── predictions.f32
├── observations.f32
└── manifest.json
```

These relative paths are recorded in the run manifest's `RunOutputs`
fields `baseline_predictions`, `baseline_observations`, and
`baseline_manifest`. The copy is best-effort: if it fails, the run still
succeeds (training and eval already completed) and the fields are left
`None`.

## The cache

The baseline is content-addressed under
`<workspace_root>/baselines/<key>/` (i.e. `.ddrs/baselines/<key>/`), with
three files:

```text
<key>/
├── predictions.f32   # raw f32 little-endian, row-major (n_gauges × n_days)
├── observations.f32  # same shape
└── manifest.json     # key, dims, gage_ids, time_range, metrics, source provenance
```

`predictions.f32` and `observations.f32` are flat little-endian f32
buffers in row-major `(n_gauges, n_days)` order — written and read by the
`write_f32_matrix` / `read_f32_matrix` helpers in
`src/baseline/cache.rs`, which validate the byte count against the
manifest dims on load. `manifest.json` is a `CacheManifest` carrying the
key, `n_gauges`, `n_days`, the `gage_ids`, the ISO-8601 daily timestamps,
the metrics, and a `SourceProvenance` record of the exact source paths
and window that produced the entry.

### The cache key

`cache_key` (in `src/baseline/cache.rs:40-86`) is the **first 16 hex
characters (64 bits)** of a `blake3` hash. The hasher consumes, in order,
the **algorithm-version salt**, then the five data-source paths each
followed by a newline, then the start and end times:

```text
BASELINE_ALGO_VERSION         ∥ "\n"    ← currently "baseline-algo-v2"
canonicalize(streamflow)      ∥ "\n"
canonicalize(observations)    ∥ "\n"
canonicalize(gages)           ∥ "\n"
canonicalize(gages_adjacency) ∥ "\n"
canonicalize(conus_adjacency) ∥ "\n"
start_time                    ∥ "\n"
end_time
```

`BASELINE_ALGO_VERSION` (`src/baseline/cache.rs:35`) is hashed **first**,
and is bumped whenever the baseline *computation* changes behavior rather
than its inputs. `v2` marks the headwater skip described above: pre-v2 keys
hashed only paths + window, so a machine with a warm cache would have kept
serving the phantom-zero baseline after the fix landed. The test
`cache_key_incorporates_algorithm_version` pins this by reconstructing the
legacy (salt-free) hash and asserting the current key differs from it.
Bump the constant in the same commit as any behavior change to `compute`
or `valid_gauges`.

Each path is canonicalized when the filesystem entry exists, falling back
to the raw path string otherwise (the fallback is what lets the cache
tests hash non-existent `/dev/null/...` paths). Crucially,
**training-only fields do not participate in the key** — `seed`,
`kan_head.*`, and `learning_rate` are all absent from the hash. That is
deliberate: tweaking a training knob and re-running `ddrs plan` is an
instant cache hit rather than a 30-second-to-2-minute recompute. Changing
the eval window or any data-source path *does* invalidate it.

If both adjacency paths are absent (a fabric-only config that has not been
through plan's managed-build resolution), `cache_key` returns
`BaselineError::ConfigMissing("conus_adjacency — …")` rather than
panicking. In normal use this never fires: `ddrs plan` resolves and
materializes the adjacency paths into the in-memory config *before*
computing the baseline (`apply_resolved` in `src/cli/plan.rs`), so the key
hashes the same stores the dataset will open.

### The shared entry point

Both `plan` and `run` go through one function:

```rust
pub fn compute_or_load_cached(
    test_cfg: &Config,
    workspace_root: &Path,
) -> Result<(SummedQPrime, String, bool), BaselineError>
```

It computes the key, returns the cached `SummedQPrime` on a hit
(`bool = true`), or computes-then-persists on a miss (`bool = false`).

- **`src/cli/plan.rs::compute_baseline`** always re-parses the config in
  `ConfigMode::Testing`, so the reference reflects the eval window the
  trained model is judged against — even when the workflow is
  `train`. The result is attached to `PlanResult.baseline` as a
  `BaselineInfo { key, cache_hit, n_gauges, cache_dir, metrics }`. The
  full `Metrics` vector is `#[serde(skip)]`, so `ddrs plan --json` emits
  only the small identifying fields, not the per-gauge arrays.
- **`src/cli/run.rs::copy_baseline_into_run_dir`** calls the same
  function after Phase-2 testing, then copies the three cache files into
  `<run_dir>/baseline/`.

Both paths **fail soft**: a baseline error prints a `warning:` line to
stderr and yields `None`. The baseline is informational and must never
block a plan or a run.

### NaN handling

`serde_json` will *write* an f32 `NaN` as JSON `null`, but it refuses to
*deserialize* `null` back into an f32. To survive the round trip, the
cache layer wraps `Metrics` in a private `MetricsJson` whose fields are
`Vec<Option<f32>>` — `None` at the JSON boundary, `f32::NAN` in memory
(`finite_to_option` / `option_to_nan`). The public `Metrics` type stays
`Vec<f32>`; the conversion happens only at the file boundary. This is why
a gauge with sparse observations (NaN FHV/FLV) survives a save → load
round trip with its NaNs intact, as verified by
`round_trip_save_load_preserves_values`.

## Side-effects in `plan`

`ddrs plan` is **not** side-effect-free. On the first invocation for a
given input set it opens icechunk, reads the daily `Qr` block, runs the
reduction, and writes the cache. Subsequent plans on the same sources +
window are instant cache hits. (This is on top of the managed adjacency
build that `plan` may also perform for fabric-only configs.)

The commonly quoted read size is **~370 MB** for a CONUS 15-year window.
That is a design-time *estimate*, not a measurement: it comes from
`docs/superpowers/plans/2026-06-02-summed-q-prime.md` §Concerns 5
(`(17k divides × 5,479 days) f32 ≈ 370 MB`), dated 2026-06-02 and not
re-derived since the 2026-07-29 gauge-population change. Treat it as an
order of magnitude for RAM budgeting; the `summed Q': … gauges × …
divides × … days` line `compute` prints is the ground truth for any
given run.

This was a deliberate trade-off
(`docs/superpowers/plans/2026-06-02-summed-q-prime.md` §"Concerns"):
computing the baseline lazily in `run` instead would waste that 30
s–2 min on every `train-and-test` with a warm cache, so it is front-loaded
into `plan` where the result can be reused.

## Reference

### Modules

| Path | Role |
|---|---|
| `src/baseline/summed_q_prime.rs` | `compute` (opens stores), `valid_gauges` (the three-filter gauge population), `assemble_from_arrays` (pure reduction), `parse_window`, `SummedQPrime`, `BaselineError` |
| `src/baseline/cache.rs` | `cache_key`, `cache_dir`, `load_cached`, `save_cached`, `compute_or_load_cached`, `CacheManifest`, `MetricsJson`, raw f32 I/O |
| `src/baseline/print.rs` | `write_metrics_summary` (testable, writes to a `Write`), `print_metrics_summary` (stdout wrapper) |
| `src/cli/plan.rs` | `compute_baseline`, `BaselineInfo`, `apply_resolved` |
| `src/cli/run.rs` | `copy_baseline_into_run_dir` |

### `BaselineError` variants

- `ConfigMissing(&'static str)` — a required config field (or unresolved
  adjacency) is absent.
- `BadDate { value, source }` — `start_time` / `end_time` failed to parse
  as `%Y/%m/%d`.
- `InvertedWindow { start, end }` — `end_time` precedes `start_time`.
- `NoGauges { gages, adj }` — no CSV gauge had a subgraph in
  `gages_adjacency`.
- `Data(#[from] DataError)` — a store read or cache I/O error, carrying
  its source path.

### Tests

- `src/baseline/summed_q_prime.rs::tests` — `assemble_from_arrays` over a
  synthetic 2-gauge × 3-day fixture (per-gauge upstream sums, NaN-skipping
  Qr), `valid_gauges_skips_single_divide_headwaters` (a zero-edge
  subgraph is dropped while an edged one survives — the regression pin for
  the phantom-zero bug), plus `parse_window` inclusive-endpoint and
  inverted-window cases.
- `src/baseline/cache.rs::tests` — key stability and length (16 chars),
  `cache_key_incorporates_algorithm_version` (current key ≠ the legacy
  salt-free hash), window-change invalidation, seed (training-field)
  non-invalidation, a full save → load round trip preserving NaN metrics,
  missing-cache → `None`, and the absent-adjacency → `Err` guard.
- `src/baseline/print.rs::tests` — table header/row presence, and a
  NaN-only metric rendering `VALID = 0`.

There is no live-icechunk integration test: the `compute` /
`assemble_from_arrays` split means the arithmetic is fully unit-tested
without spinning up icechunk repos.

## See also

- [Running the code](../usage/running.md) — the `init → plan → run`
  lifecycle that drives the baseline.
- [Reading outputs](../usage/outputs.md) — the run-directory layout the
  `baseline/` files live in.
- [Reading inputs](../usage/inputs-reading.md) — the streamflow,
  observations, and adjacency stores the baseline opens.
- [Graph objects](../usage/graph-objects.md) — the CONUS / per-gauge
  adjacency and `upstream_comids` traversal.
- [Comparing to DDR](ddr-comparison.md) — the other reference ddrs is
  held against, and the Python script (`summed_q_prime.py`) this baseline
  ports.
