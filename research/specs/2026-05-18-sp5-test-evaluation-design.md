# SP-5: test-phase `evaluate()` loop + `bin/eval` entrypoint

**Status:** Draft, pending user review
**Date:** 2026-05-18
**Parent spec:** `.claude/specs/2026-05-17-train_and_test-replication-design.md`
**Prior sub-projects:** SP-1 (static data), SP-2 (icechunk), SP-3 (collate),
SP-4 (training loop). All landed and integration-validated against DDR.

## Goal

Replicate DDR's `_test` evaluation loop in ddrs, write a DDR-layout-compatible
zarr of predictions + observations, and verify per-gauge daily predictions
match DDR to f32 floor across the full 1995-2010 test period.

A follow-up (not part of this spec) ships the `bin/train_and_test` wrapper
that sequences SP-4's `train` then SP-5's `evaluate`.

## Test-mode batching semantics

Reframed per user clarification — this is the most important shape difference
from training mode:

| Mode | Iterated axis | Network | carry_state |
|---|---|---|---|
| Training (SP-4) | Gauges (`batch_size = N_gauges` per mini-batch) | Per-batch subgraph union, rebuilt | unused |
| Testing (SP-5) | **Time** (`batch_size = N_days` per chunk) | Static all-gauges union, built once | `true` for `i > 0` |

In test mode all filtered gauges are routed in every batch; what changes
between batches is the time window. The engine carries `_discharge_t` from
the end of chunk `i` into the start of chunk `i+1`. Predictions accumulate
into a `(n_all_gauges, n_hours_full)` buffer; each batch writes
`predictions[:, hourly_offset..hourly_offset + chunk_hours]`.

## Verification ladder

**V4 — single-batch full-test-period equivalence (load-bearing):**

- Inputs: full 1995-10-01..2010-09-30 test period, all filtered gauges
  (~2365), frozen scalar params `n=0.05, q_spatial=0.5, p_spatial=21.0`
  (same constants as V1/V2, mirrored in Rust + Python).
- Both DDR and ddrs run with `batch_size = n_days_total` (one batch covering
  the whole window — no chunking, unambiguous reference).
- Compare per-gauge daily predictions:
  `max_g max_t |pred_ddrs[g,t] - pred_ddr[g,t]| / max(|pred_ddr[g,t]|, ε) < 1e-4`.
- Tolerance is `1e-4` (same as V2). May relax to `1e-3` if 5475-day f32
  accumulation drift exceeds it empirically — comment in test inline.

**V4b — multi-batch self-validation:**

- ddrs runs the same window with `batch_size = 15 days` (multiple batches,
  carry_state=true).
- Compare ddrs-multi-batch to ddrs-single-batch: should agree to f32 floor.
- Validates the chunking + carry_state plumbing. No second DDR dump needed.
- If V4b fails (but V4 passes), the issue is in carry_state or per-batch
  scattering — not in the engine math.

V3 (SP-4's smoke test) and the existing `compare_ddr_sandbox` regression
must continue to pass after SP-5 changes.

## File layout

**Created:**

- `src/training/eval.rs` — `evaluate()` driver + `EvalOutput` + `EvalParams`
- `src/training/zarr_io.rs` — `write_predictions_zarr()` matching DDR
- `src/bin/eval.rs` — `cargo run --bin eval -- --mode testing --config ...
                       --checkpoint ... --output ...`
- `scripts/dump_ddr_test_predictions.py` — DDR-side V4 reference dump
- `fixtures/sp5/v4_ddr_test.zarr/` — committed (force-add past gitignore)
- Test additions inside `tests/training_verification.rs` (no new file)

**Modified:**

- `src/data/dataset.rs` — add `MeritGagesDataset::collate_window` +
  `StaticNetworkCache` (private internal)
- `src/training/forward.rs` — add `carry_state: bool` arg to
  `forward` and `forward_with_frozen_params`
- `src/config.rs` — add `testing:` overlay section + `--mode` overlay loader
- `config/merit_training.yaml` — add `testing:` overrides section
- `Cargo.toml` — add `clap = { version = "4", features = ["derive"] }`
- `src/training/mod.rs` — re-export `evaluate`, `EvalOutput`, `EvalParams`,
  `write_predictions_zarr`
- `src/lib.rs` — no changes (training module already exposed)
- `tests/training_verification.rs` — add V4 + V4b tests

## Architecture in one screen

```
                MeritGagesDataset
                       │
                       │  collate_window(day_offset, n_days)
                       │      ┌────────────────────────────────┐
                       │      │ static_network: OnceCell<...>  │
                       │      │  - adjacency (all-gauges union)│
                       │      │  - flow_scale                  │
                       │      │  - spatial_attributes_norm     │
                       │      │  - full_observations           │
                       │      └────────────────────────────────┘
                       ▼
                RoutingBatch (static network, sliced q_prime/obs)
                       │
                       │  to_tensors::<I>
                       ▼
                RoutingTensors<I>
                       │
                       │  forward_with_frozen_params(..., carry_state)
                       │      OR forward(..., carry_state)
                       ▼
                pred_hourly (n_all_gauges, chunk_hours)  ← autograd OFF (eval)
                       │
                       │  scatter into predictions_full[:, offset..offset+ch]
                       ▼
                predictions_full (n_all_gauges, n_hours_full)
                       │
                       │  tau_trim_and_downsample (end-of-pipeline, single call)
                       ▼
                predictions_daily (n_all_gauges, n_days_trimmed)
                       │
                       │  Metrics::compute (post-warmup)
                       ▼
                EvalOutput { predictions, observations, metrics, ... }
                       │
                       │  write_predictions_zarr (DDR layout)
                       ▼
                output/model_test.zarr/
```

## Component contracts

### `MeritGagesDataset::collate_window`

```rust
impl MeritGagesDataset {
    /// Test-mode collation. Builds and caches the all-gauges static network
    /// on first call (lazy); subsequent calls just slice q_prime + obs.
    ///
    /// Mirrors DDR's `_test` per-batch RoutingDataclass construction with
    /// the simplification that the network is the full filtered-gauge union.
    ///
    /// Returns a `RoutingBatch` whose adjacency / spatial_attributes /
    /// outflow_idx / gauge_staids are identical across calls; only `q_prime`
    /// and `observations` change with the window.
    pub fn collate_window(
        &self,
        day_offset: usize,
        n_days: usize,
    ) -> Result<RoutingBatch>;
}
```

Internal: `static_network: OnceCell<StaticNetworkCache>`. First call:
- union all gauge subgraphs (reuse SP-3's `subgraph_union` logic)
- compress to lower-triangular CSR (reuse SP-3's `compress`)
- compute flow_scale from gage CSV metadata (reuse SP-3's `build_flow_scale`)
- read attributes for the all-gauges active reach set + normalize
- read full-period observations from icechunk once

Per call: slice the cached attributes/observations to the requested window,
read the q_prime slice for `[day_offset..day_offset + n_days]` from
`StreamflowStore::read_window`.

### `forward(..., carry_state)` / `forward_with_frozen_params(..., carry_state)`

```rust
pub fn forward_with_frozen_params<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    frozen: &FrozenParams,
    device: &I::Device,
    carry_state: bool,    // ← new
) -> Tensor<I, 2>;

pub fn forward<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<Autodiff<I>>,
    mlp: &Mlp<Autodiff<I>>,
    device: &I::Device,
    carry_state: bool,    // ← new
) -> Tensor<Autodiff<I>, 2>;
```

Internally passes through to `MuskingumCunge::setup_inputs(..., carry_state)`.
Existing call sites in training pass `false`; `evaluate()` passes `i > 0`.

**Required engine verification (Task 2 of the plan):** confirm that
`MuskingumCunge::setup_inputs(..., carry_state=true)` actually preserves
`_discharge_t` from the previous call. If it nulls or resets, extend the
engine to honor the flag.

### `evaluate()`

```rust
pub enum EvalParams<'a, I: Backend> {
    /// V4 verification path — uniform scalar n/q/p across every reach.
    Frozen(&'a FrozenParams),
    /// Production path — pass through an already-trained MLP.
    Mlp(&'a Mlp<I>),
}

pub struct EvalOutput {
    pub predictions_daily: Array2<f32>,    // (n_all_gauges, n_days_trimmed)
    pub observations_daily: Array2<f32>,   // same shape, obs[1..-1] trimmed
    pub gage_ids: Vec<String>,             // n_all_gauges STAIDs (zero-padded)
    pub time_range_daily: Vec<NaiveDate>,  // daily_time_range[1..-1]
    pub metrics: Metrics,                  // post-warmup per-gauge
}

pub fn evaluate<I: Backend>(
    cfg: &Config,
    dataset: &MeritGagesDataset,
    params: EvalParams<I>,
    device: &I::Device,
    batch_size_days: usize,
) -> Result<EvalOutput>;
```

Loop:

1. Compute `n_days_total` from `cfg.experiment.start_time/end_time`.
2. Iterate `day_offset` from `0` in `batch_size_days` increments. Final chunk
   may be partial.
3. Per chunk:
   - `batch = dataset.collate_window(day_offset, chunk_n_days)?`
   - `tensors = batch.to_tensors::<I>(device)`
   - `pred = forward_with_frozen_params(cfg, &tensors, frozen, device, day_offset > 0)`
     OR equivalent MLP path
   - Compute the hourly offset: `hourly_offset = day_offset * 24`
   - `predictions_full[.., hourly_offset..hourly_offset + chunk_hours] = pred`
   - (No tau-trim per batch — single end-of-pipeline call.)
4. After all chunks: lift `predictions_full` into a 1-shot
   `tau_trim_and_downsample` → `predictions_daily` shape
   `(n_all_gauges, n_days_trimmed)`.
5. Slice cached full observations to `[1..-1]` along time axis →
   `observations_daily`.
6. `Metrics::compute(predictions_daily.slice(s![.., warmup..]),
   observations_daily.slice(s![.., warmup..]))`.
7. Return `EvalOutput`.

### `write_predictions_zarr`

```rust
pub struct ZarrAttrs<'a> {
    pub start_time: &'a str,
    pub end_time: &'a str,
    pub version: &'a str,
    pub evaluation_basins_file: &'a Path,
    pub model_label: &'a str,  // checkpoint path or "frozen"
}

pub fn write_predictions_zarr(
    path: &Path,
    output: &EvalOutput,
    attrs: ZarrAttrs<'_>,
) -> Result<()>;
```

Layout — match `xarray.Dataset.to_zarr` output for `_test`:

- Group attrs (top-level): `description`, `start time`, `end time`, `version`,
  `evaluation basins file`, `model`.
- `predictions`: shape `(n_gauges, n_days)`, dtype `f64` (xarray default for
  `np.zeros` → `numpy.float64`; ddrs writes f64 to match), dimension names
  `["gage_ids", "time"]`, attrs `{units: "m3/s", long_name: "Streamflow"}`.
- `observations`: same shape/dtype, attrs `{units: "m3/s",
  long_name: "Observed Streamflow"}`.
- `gage_ids`: shape `(n_gauges,)`, variable-length UTF-8 string (zarr v3
  codec). Fallback if codec unsupported: fixed-width ASCII length 8.
- `time`: shape `(n_days,)`, dtype `i64` (nanoseconds since epoch), attr
  `{units: "nanoseconds since 1970-01-01", calendar: "proleptic_gregorian"}`.

Implementation reference: inspect one of DDR's existing zarrs (e.g., the
streamflow icechunk store) with `python -c "import zarr; print(zarr.open('...'))"`
to confirm codec choices before writing the SP-5 plan.

### Single YAML + mode overlay

`config/merit_training.yaml` extended:

```yaml
experiment:
  # training defaults (unchanged)
  start_time: 1981/10/01
  end_time: 1995/09/30
  batch_size: 64       # gauges
  rho: 90
  warmup: 5
  ...

testing:               # overlay applied when --mode testing
  start_time: 1995/10/01
  end_time: 2010/09/30
  batch_size: 15       # DAYS (semantic shift from training)
  rho: null
  # warmup, grad_clip_max_norm, learning_rate, checkpoint inherited
```

Loader logic in `Config::from_yaml_file`:

```rust
pub fn from_yaml_file_with_mode(
    path: impl AsRef<Path>,
    mode: ConfigMode,
) -> Result<Config>;

pub enum ConfigMode { Training, Testing }
```

For `Training`, the `testing:` section is ignored. For `Testing`, the
`testing:` keys overlay onto `experiment:` (key-by-key replace; absent keys
inherit from `experiment`).

The existing `Config::from_yaml_file(path)` keeps its meaning (= Training)
for SP-1..4 test back-compat.

### `bin/eval.rs`

```text
cargo run --release --bin eval -- \
    --config config/merit_training.yaml \
    --checkpoint output/saved_models/epoch_5 \
    --output output/model_test.zarr \
    --batch-size-days 15 \
    [--frozen]   # for V4-style dump from the binary
```

clap derive struct, four required args plus an optional `--frozen` flag for
dev convenience. The default mode is testing. The `--checkpoint` arg is
required unless `--frozen` is set.

Logs metrics summary on completion: NSE quartiles, RMSE quartiles, KGE
quartiles across gauges.

## V4 reference dump (`scripts/dump_ddr_test_predictions.py`)

Mirrors `scripts/dump_ddr_loss.py` (SP-4 Task 5) but for the test phase:

- Same FROZEN_N / FROZEN_Q_SPATIAL / FROZEN_P_SPATIAL constants.
- Same `physical_to_normalized` helper (linear + log-space with `+1e-6`
  epsilon) — load_log_space-respecting.
- Runs DDR's `_test` flow with `batch_size = len(daily_time_range)` (single
  batch covering whole window).
- Writes `~/projects/ddrs/fixtures/sp5/v4_ddr_test.zarr` with predictions +
  observations + gage_ids + time.

Force-add past `.gitignore` (`git add -f`). Larger than V1/V2 fixtures
(predictions = 2365 × 5475 × f64 ≈ 100 MB pre-compression; zstd should
shrink it). If the compressed size exceeds 50 MB, fall back to:
- 1-year reference window (1995-10-01..1996-09-30) instead of 15 years
- OR sub-sample to ~100 representative gauges

## V4b multi-batch test

```rust
#[test]
fn v4b_multi_batch_matches_single_batch_within_tolerance() {
    // Same dataset + frozen params as V4.
    // Run evaluate() twice: once with batch_size_days = n_days_total,
    // once with batch_size_days = 15.
    // Compare per-gauge predictions to f32 floor.
}
```

This catches:
- Off-by-one errors in `hourly_offset` arithmetic
- carry_state not actually carrying (would manifest as boundary-discharge
  discontinuities at chunk boundaries)
- Static-network caching bug (different network across calls)

## Concerns

1. **`MuskingumCunge::setup_inputs(..., carry_state=true)` correctness.**
   The engine accepts the flag (SP-4 callers pass `false`). Whether it
   actually preserves `_discharge_t` across calls is unverified. Task 2 of
   the plan starts with a small unit test: run `setup_inputs` twice with
   `carry_state=true`, confirm `discharge_state()` is unchanged before the
   second call's solve. If the engine clears it, extend.
2. **Memory at full CONUS test scale.** `predictions_full` is `2365 × 131400
   × f32 ≈ 1.2 GB`. Per-batch BURN tensors are small (≤15 days × 65k reaches
   × f32 ≈ 200 MB peak). Total RSS likely 2-3 GB. Should fit on a 16 GB
   laptop; tight on 8 GB.
3. **Streamflow per-batch reads.** `StreamflowStore::read_window` is the
   existing per-window read; SP-5 calls it once per chunk. icechunk session
   reuse across calls — confirm no per-call session cost. If it's expensive,
   refactor to open the session once at evaluate() entry and pass it to
   `read_window`.
4. **Zarr v3 variable-length string codec.** `zarrs` 0.23 may or may not
   support xarray-style VL strings cleanly. Fallback: fixed-width ASCII 8
   chars (matches STAID format). If even that fails, write a fixed-width
   byte array + a `dtype: U8` attr.
5. **DDR's `_test` clears `_discharge_t` between batches.** ddrs's evaluate
   will NOT. This means ddrs's MULTI-batch behavior may diverge from DDR's
   actual multi-batch behavior. V4 (single-batch reference) sidesteps this.
   V4b's premise is that ddrs-multi-batch reproduces ddrs-single-batch when
   carry_state works. If V4b fails, diagnostic info points to carry_state
   or chunking math, not engine math.
6. **`batch_size` semantic shift between training and testing.** A reader of
   `merit_training.yaml` could easily mistake the test `batch_size: 15` for
   gauges. Inline YAML comment + `Config::from_yaml_file_with_mode` doc
   comment must call this out loudly.
7. **clap as a new dep.** Adds compile time but well-established and tiny.
   No alternative considered.
8. **Chunk-time semantics vs SP-3's `RhoWindow::n_hourly`.** SP-3 defines
   `n_hourly = (rho_days - 1) * 24` to mirror DDR's pandas `inclusive='left'`
   on the hourly range — every training batch's q_prime is one day shorter
   than its daily range. For chunked test mode, this creates a 24-hour gap
   between chunks: chunk 0 covering days [0, 15) yields hours [0, 336)
   (loses the last day); chunk 1 starting at day 15 yields hours [360, 696);
   hours [336, 360) belong to no chunk. SP-5 needs *contiguous* hourly
   coverage: each chunk's q_prime must be exactly `chunk_n_days * 24` hours.
   Either (a) introduce a `TestWindow` type that does NOT apply the
   `inclusive='left'` trim, or (b) extend `RhoWindow` with a `Mode`
   parameter. Plan Task 1 resolves this; (a) is the simpler call.
   Diagnostic test: confirm `evaluate()` over two consecutive 1-day chunks
   produces the same predictions as one 2-day chunk (with carry_state).

## Open assumptions

1. `MuskingumCunge::setup_inputs` will honor `carry_state=true` correctly
   after engine inspection (verify in plan Task 2).
2. The single-batch DDR reference dump is computationally feasible on the
   user's hardware. If memory-bound or runtime > 30 min, fall back to a
   shorter reference window and adjust V4 accordingly.
3. `testing:` section as an additive YAML overlay does not require a
   breaking schema change to existing `ParamsRaw` / `ExperimentRaw`
   deserialization — verify by reading SP-3 Task 1's config schema.
4. DDR's predictions zarr is zarr v3. Confirm by running `zarr.info` on one
   of DDR's existing outputs (or on the streamflow icechunk store) before
   writing the plan.
5. The `flow_scale` correction at gauge outlets, already implemented in
   SP-3, applies identically in test mode. Verify by inspecting one batch's
   q_prime against DDR's.

## Out of scope (deferred follow-up)

- `bin/train_and_test` wrapper. Trivial sequencing of `train()` + `evaluate()`;
  add after SP-5 lands.
- Per-gauge prediction plots (DDR has these but they're notebook concerns).
- GPU backend swap (`Wgpu`/`CudaJit`). Independent perf project.
- Cross-runtime checkpoint loading (DDR `.pt` → ddrs MLP). Architectures
  differ; not a simple weight transfer.

## Next steps after this spec is approved

Invoke the writing-plans skill to produce a task-by-task implementation plan
(`.claude/specs/2026-05-18-sp5-test-evaluation-plan.md`) covering all 9
phases listed above. Then execute via subagent-driven-development (same
pattern as SP-1..4).
