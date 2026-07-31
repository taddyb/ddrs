# Reading outputs

**Everything a `ddrs run` produces lands in one directory.** `src/cli/run.rs`
computes `run_dir = workspace.runs_dir().join(&run_id)` and writes the
config snapshot, the log, every checkpoint, the eval zarr, the baseline copy,
the diagnostic NetCDFs, and the manifest underneath it. There is nothing to
collect from scattered paths and no `--output` flag to remember: the run
directory *is* the result.

The only outputs that live outside `.ddrs/runs/` are the two `examples/`
artefacts (`output/ddrs_vs_ddr.{csv,png}`, `output/hydrograph.{csv,png}`) and
whatever explicit `--output` you pass to a legacy binary or a diagnostic
tool.

## What it is

### The run directory

```text
.ddrs/runs/<UTC-timestamp>-[<group>-]<workflow>/
├── manifest.json               provenance + metrics + output index   (JSON)
├── config.yaml                 verbatim snapshot of the input config (YAML)
├── Cargo.lock                  dependency snapshot                   (TOML)
├── run.log                     timestamped tee of stdout + stderr    (text)
├── checkpoints/
│   └── epoch_E_mb_M/           one directory per checkpoint
│       ├── head.mpk            KAN head weights  (CompactRecorder, f16)
│       ├── optim.mpk           Adam moments      (CompactRecorder, f16)
│       └── state.json          epoch, next mini-batch, rng, sampler state
├── eval/
│   └── predictions.zarr/       DDR-compatible predictions store      (zarr v3)
├── baseline/                   copy of the cached summed-Q′ baseline
│   ├── predictions.f32         raw f32, row-major (n_gauges, n_days)
│   ├── observations.f32        same shape
│   └── manifest.json           gage_ids, time_range, metrics, provenance
├── kan_parameters.nc           eval-window zeta diagnostic, dim COMID_eval
│                               (leakance runs only)                  (NetCDF4)
└── plot/
    └── kan_parameters.nc       per-COMID KAN params, dim COMID       (NetCDF4)
                                (--plot only, full CONUS)
```

Which of these appear depends on the workflow:

| Artefact | `--workflow train` | `--workflow train-and-test` |
|---|---|---|
| `manifest.json`, `config.yaml`, `Cargo.lock`, `run.log` | ✓ | ✓ |
| `checkpoints/epoch_E_mb_M/` | ✓ | ✓ (Phase 1) |
| `eval/predictions.zarr/` | — | ✓ (Phase 2) |
| `baseline/` | — | ✓ (Phase 2) |
| `kan_parameters.nc` | — | ✓ *if* `params.use_leakance: true` |
| `plot/kan_parameters.nc` | ✓ with `--plot` | ✓ with `--plot` |

`--workflow eval` produces nothing — it errors before doing any work (see
[Running the code](running.md#evaluating-a-checkpoint)).

### The run id

`<UTC timestamp>-[<group>-]<workflow>`, e.g.
`2026-07-30T01-58-07Z-train-and-test` or
`2026-06-12T14-02-10Z-global-train-and-test`. The timestamp is RFC-3339 to
second precision with `:` replaced by `-`. The optional `<group>` segment is
the active data-source group (`sources::active_group`), present when the
config's `data_sources` block matches a saved `config/sources/<name>.yaml`,
so a run directory says which dataset it was trained on. Group names are
sanitized to `[A-Za-z0-9._-]`.

### The four on-disk shapes

Four formats appear across the tree: **BURN NamedMessagePack** (`.mpk`,
readable only via `CompactRecorder` against a matching module shape),
**JSON** (manifest, state sidecar, baseline manifest), **zarr v3** (the eval
predictions), **NetCDF4** (the parameter and zeta diagnostics), plus raw
little-endian `f32` blobs for the baseline arrays and CSV+PNG from the
examples.

## How to use it

### `manifest.json` — the index of everything

Written last, atomically (`tmp` + rename), and written **even when the run
fails**, so a failed run still leaves a readable record with `status:
"failed"` and an `exit_reason`. A panic inside the workflow is caught and
recorded the same way.

Top-level schema (`src/cli/manifest.rs:82-101`):

| Key | Contents |
|---|---|
| `run_id` | The directory name |
| `ddrs_version` | `CARGO_PKG_VERSION` |
| `git` | `{sha, dirty, branch}` — captured by shelling out to `git` at run time |
| `workflow` | `train` \| `eval` \| `train-and-test` |
| `config_path` | Absolute path to this run's `config.yaml` snapshot |
| `started_at`, `finished_at` | RFC-3339 UTC |
| `status` | `ok` \| `failed` |
| `exit_reason` | `null`, or the error string |
| `system` | `SystemProbe`: gpu, cuda_runtime, driver, sm, `free_gpu_gb_at_probe`, cached `smoke_test` record |
| `sources` | Map of data-source name → `Fingerprint` |
| `resolved_adjacency` | `{conus, gages, cache_key?, cache_hit?}` — the stores actually opened |
| `source_lock` | `{lockfile, matched, drift}` **as of plan entry** |
| `outputs` | See `RunOutputs` below |
| `metrics` | Free-form JSON, shape depends on workflow |
| `max_mini_batches` | The `--max-mini-batches` value, or `null` |

> `source_lock.matched` / `.drift` record the state **at plan entry**. On a
> non-strict run with drift, `plan()` has already refreshed `sources.lock`,
> so the file on disk can match `sources` even when `matched: false`.

**`outputs` (`RunOutputs`, `src/cli/manifest.rs:65-80`)** — every path is
*relative to the run directory*:

```json
"outputs": {
  "checkpoints": ["checkpoints/epoch_1_mb_0", "checkpoints/epoch_2_mb_0", "..."],
  "plot": null,
  "eval_zarr": "eval/predictions.zarr",
  "baseline_predictions": "baseline/predictions.f32",
  "baseline_observations": "baseline/observations.f32",
  "baseline_manifest": "baseline/manifest.json",
  "run_log": "run.log"
}
```

`checkpoints` is always present (empty array if none); the rest are omitted
from the JSON when unset, except `plot`, which serializes as `null`. Note
`checkpoints` is sorted **lexicographically**, so `epoch_10_mb_0` precedes
`epoch_2_mb_0` — do not treat the last element as the latest checkpoint.
The code that needs the latest one parses `epoch_E_mb_M` numerically
(`latest_checkpoint_base`).

**`metrics`** for `train-and-test`:

```json
"metrics": {
  "epochs_completed": 30,
  "final_mini_batch": 0,
  "phase1_seconds": 43034.773,
  "phase2_seconds": 24653.926,
  "n_gauges_finite_nse": 2365,
  "n_gauges_total": 2365,
  "mean_nse_finite": 0.2824,
  "median_nse_finite": 0.6209,
  "median_kge_finite": 0.6903
}
```

For `train`, only the first three keys are present.

Read it with `ddrs show <run-id>` (human) or `ddrs show <run-id> --json`.
`ddrs run --json` does **not** work — the flag is parsed and discarded.

### `config.yaml` and `Cargo.lock` — reproducibility snapshots

`config.yaml` is a byte copy of the input config, made *before* the workflow
starts, so it captures exactly what you asked for — including the original
(possibly absent) adjacency keys. The paths `plan` resolved are recorded
separately in `manifest.resolved_adjacency`; the snapshot is never mutated.

This snapshot is also what `ddrs plan`'s bootstrap prompt offers as a
starting point when materializing a fresh `ddrs.yaml`.

`Cargo.lock` is copied on a best-effort basis from `./Cargo.lock` or
`../Cargo.lock`; if neither is reachable the copy is silently skipped.

### `run.log` — the timestamped tee

`cli::tee::tee_to` redirects file descriptors 1 and 2 for the duration of the
workflow, so it captures CUDA's stderr chatter and any child-process output,
not just Rust `println!`. Each line is prefixed with a UTC timestamp:

```
[2026-07-30T01:58:07Z] backend: cpu (NdArray, deterministic; sparse_solver forced to cpu)
[2026-07-30T01:58:07Z] DA_VALID filter: kept 2859/3211 gauges
[2026-07-30T01:58:08Z] gages_adjacency filter: kept 2365 gauges (dropped 0 missing, 494 headwater)
[2026-07-30T01:58:08Z] streamflow resolution: Daily
```

Grep this first when a run's numbers look wrong — the gauge-filter counts and
the `streamflow resolution: Daily|Hourly` line are the fastest way to confirm
the run read what you think it read.

### Training checkpoints (`checkpoints/epoch_E_mb_M/`)

The training driver writes **one checkpoint per mini-batch**. A checkpoint is
a **directory**, not a single file:

```text
checkpoints/epoch_30_mb_0/
├── head.mpk      KAN head weights   (CompactRecorder, f16 on disk)   ~131 KB
├── optim.mpk     Adam moments       (CompactRecorder, f16 on disk)   ~189 KB
└── state.json    epoch, next mini-batch, serialized rng, sampler
                  permutation + cursor                                ~11 KB
```

(Sizes are for the CONUS KAN head; they scale with `hidden_size` /
`num_hidden_layers`, not with the network.)

The directory name is `epoch_{epoch}_mb_{mini_batch}` (see
`src/training/driver.rs`, which calls `create_dir_all` on it then writes the
three files via `head_base`/`optim_base`/`state_path` from
`src/training/checkpoint.rs`). The `.mpk` extension on `head.mpk` and
`optim.mpk` is appended **by the recorder** — the in-code *base* paths are
`dir/head` and `dir/optim` (`src/training/checkpoint.rs:103, 109`).

Format: `CompactRecorder = NamedMpkFileRecorder<HalfPrecisionSettings>`.
Two consequences:

1. **Weights and Adam moments are stored in half precision (`f16`)** on
   disk. They expand to `f32` on load to match the routing-core dtype.
   Saving never widens — re-saving a loaded checkpoint loses the LSBs of
   the in-memory `f32`, so a resumed trajectory drifts slowly from the
   uninterrupted one.
2. **No portable C struct.** The `.mpk` files are BURN's named-MessagePack
   serialization; you cannot reliably parse them with a generic msgpack
   reader because field names depend on the `#[derive(Module)]` shape of
   the KAN head at compile time. Read the head from Rust via
   `load_kan_head` with a template built at the same architecture:

```rust
use ddrs::nn::kan_head::{KanHead, KanHeadConfig};
use ddrs::training::checkpoint::{head_base, load_kan_head};

// Template with the SAME architecture (hidden_size, num_hidden_layers,
// grid, k) as when it was saved.
let head_cfg = KanHeadConfig::new(input_var_names, learnable_parameters, seed)
    .with_hidden_size(hidden_size)
    .with_num_hidden_layers(num_hidden_layers)
    .with_grid(grid)
    .with_k(k);
let head_template: KanHead<B> = head_cfg.init::<B>(&device);

// Pass the checkpoint DIRECTORY; head_base appends `head`, and the
// recorder re-appends `.mpk`.
let head = load_kan_head::<B>(&head_base(&ckpt_dir), head_template, &device)?;
```

The legacy `eval` binary does exactly this: `--checkpoint` takes the
`epoch_E_mb_M/` directory, and `load_kan_head(&head_base(...))` reads
`head.mpk` from inside it. To resume training instead, point
`experiment.checkpoint:` in `ddrs.yaml` at the same directory;
`bootstrap_head_and_state` restores head + optimizer + `state.json` so the
resumed run draws the same gauge batches and rho-windows the original would
have.

### Eval predictions (`eval/predictions.zarr/`)

Phase 2 of `train-and-test` writes a zarr v3 store. Layout (see
`src/training/zarr_io.rs`):

```
/predictions    (n_gauges, n_days)  f64   units "m3/s"
/observations   (n_gauges, n_days)  f64   units "m3/s"
/gage_ids       (n_gauges, W)       u8    fixed-width ASCII STAID (_dtype_hint "|SW")
/time           (n_days,)           i64   nanoseconds since 1970-01-01
```

Each 2-D array is stored as a **single chunk**, so a read touches the whole
array. `/predictions` and `/observations` carry `_ARRAY_DIMENSIONS:
["gage_ids", "time"]` for xarray; `/time` also carries `calendar:
"proleptic_gregorian"`; `/gage_ids` carries `["gage_ids", "char"]`.

`W` is `max(longest gage id, 8)` — zarr v3 has no fixed-length string dtype,
so IDs are zero-padded `u8` with a `_dtype_hint` attr for readers. On CONUS
USGS STAIDs that is `W = 8`, `_dtype_hint: "|S8"`; on the global stores,
whose ids look like `GRDC__1286661`, it widens. (A hardcoded 8 used to
truncate `GRDC__1286661` → `GRDC__12`, collapsing 5,224 gauges onto 93
distinct prefixes — hence the `.max(8)`.)

Group attributes record run metadata: `description`, `start time`, `end
time`, `version` (the ddrs `CARGO_PKG_VERSION`), `evaluation basins file`
(the gages CSV path), and `model`.

> **`model` is the head *base* path, not the checkpoint directory**, when the
> store came from `ddrs run` — e.g.
> `…/checkpoints/epoch_30_mb_0/head`. The legacy `eval` binary instead
> records the literal `--checkpoint` value (the directory), or the string
> `"frozen"` when `--frozen` was passed.

Read it from xarray:

```python
import xarray as xr
ds = xr.open_zarr(".ddrs/runs/<run-id>/eval/predictions.zarr")
print(ds.predictions.shape, ds.attrs["model"])
```

The format is DDR-compatible — DDR's analysis notebooks open it without
modification.

Phase 2 also logs a metrics summary to stdout (and therefore to `run.log`).
Per-gauge mean is misleading on right-skewed NSE distributions, so only the
**median** is reported:

```
gauges with finite NSE: 412 / 430
median NSE (finite only): 0.6843
median KGE (finite only): 0.7012
```

Per-gauge NSE/KGE are **not** written to the zarr — recompute from
`predictions` vs `observations` if you need them persisted, or read the
aggregates from `manifest.json`'s `metrics` block (which additionally records
`mean_nse_finite`).

### The baseline copy (`baseline/`)

`train-and-test` loads (or computes) the cached summed-Q′ baseline and copies
three files out of `.ddrs/baselines/<key>/` into the run directory, so the
comparison travels with the manifest:

| File | Contents |
|---|---|
| `predictions.f32` | Raw little-endian `f32`, row-major `(n_gauges, n_days)`. No header |
| `observations.f32` | Same shape and layout |
| `manifest.json` | `{key, n_gauges, n_days, gage_ids, time_range_daily, metrics, sources}` |

`metrics` holds per-gauge `nse` / `kge` arrays (parallel to `gage_ids`), and
`time_range_daily` is a list of `YYYY-MM-DD` strings — so the blob shape is
fully recoverable from the sidecar:

```python
import json, numpy as np
m = json.load(open(".ddrs/runs/<run-id>/baseline/manifest.json"))
shape = (m["n_gauges"], m["n_days"])
pred = np.fromfile(".ddrs/runs/<run-id>/baseline/predictions.f32",
                   dtype="<f4").reshape(shape)
```

The copy is best-effort: if the baseline can't be computed or copied the run
still succeeds, a warning goes to `run.log`, and the three `RunOutputs`
fields stay unset.

> The baseline's gauge population is keyed on its own cache key and may not
> equal the run's `n_gauges_total`. Join on `gage_ids` before comparing —
> don't assume the rows line up positionally with the eval zarr.

### Diagnostic NetCDFs

Two different NetCDF files can appear, on **two different dimensions**. They
are distinct products and can even coexist in one file, which is why they use
different dimension names.

**`plot/kan_parameters.nc` — full-CONUS learned parameters.** Written by
`ddrs run --plot` (and by the `dump_parameters` binary). Dimension `COMID`
over every reach in the network (346,321 for CONUS):

```
dimensions:  COMID = 346321
variables:
  int64 COMID(COMID)             MERIT reach identifier
  float n(COMID)                 Manning's roughness         s/m^(1/3)
  float q_spatial(COMID)         discharge scaling exponent  dimensionless
  float p_spatial(COMID)         width-to-depth ratio        dimensionless
  float x_storage(COMID)         Muskingum X storage weight  dimensionless
  float slope(COMID)             channel slope (clamped)     m/m
  # plus K_D, d_gw, leakance_factor when leakance is enabled
global attributes:
  :checkpoint, :ddrs_version, :n_reaches, :note
```

Values are **denormalised** — physical units, not the head's `[0,1]` output.

**`kan_parameters.nc` at the run root — the eval-window zeta diagnostic.**
Written by `train-and-test` Phase 2 when `params.use_leakance: true`, or by
`eval --zeta-output`. Dimension `COMID_eval` — the **eval network** (the
gauge-subgraph union), *not* full CONUS:

```
dimensions:  COMID_eval
variables:
  int64 COMID_eval(COMID_eval)
  float zeta(COMID_eval)         mean |zeta| over the eval window, m³/s
  float zeta_net(COMID_eval)     signed mean; positive = losing reach
  float depth_mean(COMID_eval)
  float area_z_mean(COMID_eval)
  float q_mean(COMID_eval)
global attributes:
  :zeta_checkpoint, :zeta_ddrs_version, :zeta_note
```

`zeta` is the quantity the `|zeta| > 0.01 m³/s` GO/NO-GO bar is measured
against, and it is exactly what was subtracted from the routing RHS `b` —
`tests/zeta_accum.rs` proves this via the headwater identity. The writer
appends to an existing file rather than clobbering it, and **errors** if an
existing `COMID_eval` dimension has a different length ("delete the file (or
the stale zeta vars) and re-run").

### V1 sandbox diff (`output/ddrs_vs_ddr.{csv,png}`)

`compare_ddr_sandbox` writes the per-reach diff CSV with header:

```
reach_id,max_abs_diff,mean_abs_diff,max_rel_diff,ddr_mean,ddrs_mean,corr
```

One row per RAPID2-ordered reach (5 rows for the canonical fixture).
`corr` is the Pearson correlation between DDR's discharge and ddrs's,
computed over the whole window per reach. All diffs are in m³/s;
`max_rel_diff` is unitless (`|a-b| / |a|`, skipping `|a| < 1e-6`).

The verdict line on stdout summarises the cross-reach maxima:

```
verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

or `close match (max rel < 1%) — see plot for visual confirmation`, or
`DIVERGENCE — investigate`. **Only the first counts as passing the V1
invariant** (see [Comparing to DDR](../reference/ddr-comparison.md)).

`output/ddrs_vs_ddr.png` overlays DDR (solid coloured line) and ddrs
(dashed black) one panel per reach. Use it for a visual sanity-check;
the CSV is the actual gate.

### Hydrograph (`output/hydrograph.{csv,png}`)

`benchmark_hydrograph` writes a **wide** CSV — one column per reach,
one row per hourly timestep:

```
t_hours,reach_0,reach_1,...,reach_9
0,5.000000,5.000000,...
1,5.349...,5.121...,...
...
```

72 rows (3 days of hourly steps), with a `t_hours` column plus
`reach_0`…`reach_9` (10 reaches → 11 columns total). Read with pandas:

```python
import pandas as pd
df = pd.read_csv("output/hydrograph.csv", index_col="t_hours")
df.plot()  # 10 reach hydrographs
```

`output/hydrograph.png` is styled to mirror DDR's
`plot_routing_hydrograph` (1500×675 px at 150 dpi, white background,
tab10 palette, "DDR Routed Discharge" title, m³/s y-axis). Useful as a
visual smoke test that the routing core hasn't drifted between dev
sessions — the diurnal sweep should peak at roughly the same hours every
run.

## Reference

### Gotchas

- **Don't go looking for a results directory — you're already in it.**
  `.ddrs/runs/<id>/` holds everything. `ddrs status` lists the runs;
  `ddrs show <id>` prints the manifest; `ddrs gc` prunes them (but never
  touches `.ddrs/adjacency/`).
- **`output/` must exist before running the examples.** Both
  `compare_ddr_sandbox` and `benchmark_hydrograph` call
  `BufWriter::new(File::create("output/..."))` with no `create_dir_all`
  guard and panic on a missing directory. One-time fix: `mkdir -p output`.
- **Only `train` and `train_and_test` create their output directories.**
  `src/bin/train.rs:57` and `src/bin/train_and_test.rs:75-78` call
  `create_dir_all`. **`src/bin/eval.rs` calls it nowhere** — `eval --output
  some/missing/dir/x.zarr` fails at write time. Create the parent yourself.
- **A checkpoint is a directory, not a file.** Pass the `epoch_E_mb_M/`
  directory to `eval --checkpoint` and to `experiment.checkpoint:`. The inner
  filenames (`head.mpk`, `optim.mpk`, `state.json`) are hardcoded; do not
  point at one of the inner files. The whole directory copies/deletes as one
  unit, so nothing can clobber the head.
- **Pass the base path, not `.mpk`, to the loaders.** `head_base` /
  `optim_base` return `dir/head` / `dir/optim`; the recorder re-appends
  `.mpk`. Passing `head.mpk` produces `head.mpk.mpk` and a load failure.
  (`dump_parameters --checkpoint` wants that base path directly, unlike
  `eval --checkpoint`, which wants the directory.)
- **`outputs.checkpoints` is sorted as text.** `epoch_10_mb_0` sorts before
  `epoch_2_mb_0`. Parse `epoch_E_mb_M` numerically to find the latest.
- **A `failed` run still has a manifest.** Check `status` and `exit_reason`
  before trusting anything else in the directory — checkpoints from a run
  that died mid-epoch are perfectly loadable and perfectly useless.
- **`.mpk` files are not portable across BURN minor versions.** BURN bumps
  may rename module fields and the `NamedMpkFileRecorder` will reject the old
  file. Re-record after a BURN upgrade; treat checkpoints as throwaway across
  version bumps, not as artefacts to archive long-term.
- **`.mpk` files are not portable from DDR either.** DDR's `.pt` files match
  the KAN head's I/O contract but not its on-disk record format;
  `load_kan_head`'s `load_record` rejects them.
- **Half-precision saves lose `f32` LSBs.** Don't round-trip a checkpoint
  through save→load→save expecting bit-identity — the first save quantises to
  `f16` and subsequent saves preserve only that. This is why a resumed
  trajectory drifts slowly from an uninterrupted one.
- **`plotters` axis style depends on the pinned version in `Cargo.toml`.**
  Upgrading will silently change tick labels / line caps in PNGs even though
  the CSVs are unchanged. Pin or pixel-diff if you care.

### Verification

| Path | Covered by |
|---|---|
| Manifest schema + atomic write | `cargo test --test cli_manifest` |
| `--json` output shape | `cargo test --test cli_json_contract` |
| Run-log tee (fd-level, timestamps) | `cargo test --test cli_tee` |
| Checkpoint save + exact resume | `cargo test --test checkpoint_resume` |
| Eval zarr layout | `cargo test --lib training::zarr_io` |
| `--plot` post-step | `cargo test --test cli_plot` |
| Baseline cache round-trip | `cargo test --lib baseline::` |
| Zeta diagnostic == what's subtracted from `b` | `cargo test --test zeta_accum` |
| V1 CSV row count + verdict | `cargo run --release --example compare_ddr_sandbox` then `wc -l output/ddrs_vs_ddr.csv` (expect 6 = 1 header + 5 reaches) |
| Hydrograph wide format | `cargo run --release --example benchmark_hydrograph` then `head -1 output/hydrograph.csv` (expect `t_hours,reach_0,...,reach_9`) |

> `src/training/checkpoint.rs` contains **no** `#[test]` items — a filter
> like `cargo test --lib training::checkpoint` matches zero tests and exits
> green. The real coverage is `tests/checkpoint_resume.rs`.

The V1 CSV + verdict path is the only output that gates correctness; the
others are debugging / interpretability aids.

## See also

- [Running the code](running.md) — the producer side of every artefact here,
  including the `plan → run` lifecycle and the flags that toggle the optional
  outputs.
- [Comparing to DDR](../reference/ddr-comparison.md) — what the V1 diff CSV
  measures and how to interpret it.
- [The baseline](../reference/baseline.md) — how the summed-Q′ reference in
  `baseline/` is computed and cached.
- [Formatting inputs](inputs-formatting.md) — the config keys behind these
  outputs, including `experiment.checkpoint:` and `params.use_leakance:`.
