---
name: ddrs-baseline
description: The summed Q' reference baseline — per-gauge sum of upstream divide Qr for sanity-checking trained KAN routing performance, with cache layout under .ddrs/baselines/<key>/.
output: reference/baseline.md
sources:
  - src/baseline/summed_q_prime.rs
  - src/baseline/cache.rs
  - src/baseline/print.rs
  - src/cli/plan.rs
  - src/cli/run.rs
---

# ddrs-baseline

> Canonical agent-readable skill. Published chapter at `docs/reference/baseline.md`
> is regenerated from this file by `/regenerate-docs`.

## What to know

The **summed Q' baseline** is a non-routing reference computed by `ddrs plan`
and `ddrs run --workflow train-and-test`. For each evaluation gauge it:

1. Looks up the upstream subgraph (the union of CONUS-position COMIDs in
   `gages_adjacency[gauge].indices_0 ∪ indices_1`).
2. Bulk-reads daily Qr for the union of all upstream divides over the
   testing window via `StreamflowStore::read_window_daily`.
3. Sums Qr across each gauge's upstream divides (NaN-skipping) → predicted
   daily streamflow for that gauge.
4. Compares against `UsgsObservationsStore::read_window_daily` → NSE,
   KGE, RMSE, bias, FHV, FLV per gauge via `Metrics::compute`.

**Why it exists:** if the trained KAN's median NSE doesn't beat this
number, the routing isn't earning its keep. Check training loss curves
and KAN-head gradient stats first, not the sparse solver. The Python
reference is `~/projects/ddr/scripts/summed_q_prime.py`.

## Cache contract

Content-addressed under `<workspace_root>/baselines/<key>/`:

```
<key>/
├── predictions.f32   # raw f32 LE, row-major (n_gauges × n_days)
├── observations.f32  # same shape
└── manifest.json     # gage_ids, time_range, metrics, provenance
```

`key` is the 16-hex-char (64-bit) prefix of `blake3` over:

```
streamflow ∥ observations ∥ gages ∥ gages_adjacency ∥ conus_adjacency
∥ start_time ∥ end_time
```

Paths are canonicalized when possible (fall back to raw on missing
filesystem entries — used by tests). Training-only fields (`seed`,
`kan_head.*`, `learning_rate`) do **not** participate in the key, so
re-running `ddrs plan` after tweaking training knobs is an instant
cache hit.

## NaN handling at the JSON boundary

`serde_json` writes f32 `NaN` as JSON `null` but refuses to deserialize
`null` back into f32. The cache layer wraps `Metrics` in a private
`MetricsJson` with `Vec<Option<f32>>` fields at the file boundary; the
public `Metrics` type stays as `Vec<f32>`. FHV/FLV NaNs happen for any
gauge with fewer than ~50 finite paired days (the 2% / 30% slice
indices round to 0 or N).

## How `plan` and `run` share the result

`compute_or_load_cached(test_cfg, workspace_root) -> (SummedQPrime, key, hit)`
is called from both:

- `cli/plan.rs::compute_baseline` — always uses `ConfigMode::Testing` so
  the user sees the same reference regardless of which workflow they
  asked to plan. Result attached to `PlanResult.baseline` as
  `BaselineInfo` (in-memory; full `Metrics` is `#[serde(skip)]` so
  `--json` mode emits only key/cache_hit/n_gauges/cache_dir).
- `cli/run.rs::copy_baseline_into_run_dir` — after Phase-2 testing,
  copies the three cache files into `<run_dir>/baseline/`. Populates
  `RunOutputs.baseline_{predictions,observations,manifest}`.

Both paths fail soft. The baseline is informational, never blocking.

## Side-effects in `plan`

`ddrs plan` is no longer pure read-only — first invocation opens
icechunk, reads ~370 MB of daily Qr (for ~17k unique upstream divides
over a 15-yr window), and writes the cache. Subsequent invocations on
the same input set are instant cache hits. The decision was deliberate
(see `docs/superpowers/plans/2026-06-02-summed-q-prime.md` "Concerns")
because recomputing in `run` would waste 30s–2min per train-and-test
on a warm cache.

## Tests

- `src/baseline/summed_q_prime.rs::tests` — `assemble_from_arrays` over
  synthetic 2-gauge × 3-day data; NaN-skipping; date-window parsing
  (inclusive endpoints, inverted-window rejection).
- `src/baseline/cache.rs::tests` — key stability, training-field
  non-invalidation, full save→load round trip preserving NaN metrics.
- `src/baseline/print.rs::tests` — table header/row presence, NaN-only
  metric shows `VALID = 0`.

No live-icechunk integration test today; the pure-compute split (`compute`
opens stores, `assemble_from_arrays` does the math) means the math is
fully unit-tested without spinning up icechunk repos.
