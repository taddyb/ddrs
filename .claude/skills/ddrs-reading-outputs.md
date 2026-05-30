---
name: ddrs-reading-outputs
description: How to read ddrs's outputs — .mpk checkpoints from training, hydrograph CSV/PNG from benchmark_hydrograph, and the per-reach diff CSV from compare_ddr_sandbox.
output: usage/outputs.md
sources:
  - examples/compare_ddr_sandbox.rs
  - examples/benchmark_hydrograph.rs
  - src/bin/train.rs
  - src/bin/eval.rs
---

# ddrs-reading-outputs

> Canonical agent-readable skill. Published chapter at `docs/usage/outputs.md`
> is regenerated from this file by `/regenerate-docs`.

## What to know

Every ddrs binary writes its artefacts to one of two places: the
repo-local `output/` directory (the two examples, plus `eval` by default
when the user passes `--output output/...`) or a user-supplied
`--checkpoint-dir` (training). Three formats appear: binary `.mpk` (BURN's
NamedMessagePack record format — needs `CompactRecorder` to read), plain
CSV (readable with any tool — pandas/polars/awk), and PNG (rendered with
`plotters`). Eval also writes a zarr v3 store. There is no global "results
directory" — each binary owns its own paths.

| Artefact | Producer | Path | Format |
|---|---|---|---|
| MLP checkpoint | `train` / Phase 1 of `train_and_test` | `<checkpoint-dir>/epoch_{e}_mb_{mb}.mpk` | BURN MPK (f16 weights on disk) |
| V1 diff | `examples/compare_ddr_sandbox` | `output/ddrs_vs_ddr.{csv,png}` | CSV + PNG |
| Hydrograph | `examples/benchmark_hydrograph` | `output/hydrograph.{csv,png}` | CSV + PNG |
| Eval predictions | `bin/eval` | user `--output` (typically `*.zarr`) | zarr v3 |

## Training checkpoints (`.mpk`)

`train.rs` and the Phase-1 half of `train_and_test.rs` write **one
checkpoint per mini-batch** under `--checkpoint-dir`, named
`epoch_{epoch}_mb_{mini_batch}.mpk`. The file extension is appended by
`CompactRecorder::set_extension` — the in-code path does **not** include
`.mpk`, but the on-disk file always does. See
`src/training/driver.rs:130`.

Format: `CompactRecorder = NamedMpkFileRecorder<HalfPrecisionSettings>`
(see `src/training/checkpoint.rs:10-12`). Two consequences:

1. **Weights are stored in half precision** (`f16`) on disk. They expand
   to `f32` on load to match the routing-core dtype. Saving never widens —
   re-saving a loaded checkpoint loses the LSBs of the in-memory `f32`.
2. **No portable C struct.** The file is BURN's named-MessagePack
   serialization; you cannot reliably parse it with a generic msgpack
   reader because field names depend on the `#[derive(Module)]` shape of
   `Mlp<B>` at compile time. Read it from Rust:

```rust
use ddrs::training::checkpoint::load_mlp;
use ddrs::nn::mlp::{Mlp, MlpConfig};

// Construct a template with the SAME architecture as when it was saved.
let mlp_cfg = MlpConfig::new(input_names, learnable_names)
    .with_hidden_size(64)
    .with_num_hidden_layers(2);
let mlp_template: Mlp<B> = mlp_cfg.init::<B>(&device);

// `path` is the base — pass `epoch_5_mb_120`, NOT `epoch_5_mb_120.mpk`.
let mlp = load_mlp::<B>(&path, mlp_template, &device)?;
```

`train_and_test.rs` does this automatically via `find_latest_mpk`
(`src/bin/train_and_test.rs:222-244`), which scans the directory by mtime
and strips the `.mpk` suffix before calling `load_mlp`.

## V1 sandbox diff (`output/ddrs_vs_ddr.{csv,png}`)

`compare_ddr_sandbox` writes the per-reach diff CSV with header:

```
reach_id,max_abs_diff,mean_abs_diff,max_rel_diff,ddr_mean,ddrs_mean,corr
```

One row per RAPID2-ordered reach (5 rows for the canonical fixture).
`corr` is Pearson correlation between DDR's discharge and ddrs's, computed
on the whole window per reach. All diffs are in m³/s; `max_rel_diff` is
unitless (`|a-b| / |a|`, skipping `|a| < 1e-6`).

The verdict line on stdout summarises the cross-reach maxima:

```
verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

or `close match (max rel < 1%) — see plot for visual confirmation`, or
`DIVERGENCE — investigate`. **Only the first counts as passing the V1
invariant** (see `CLAUDE.md`).

`output/ddrs_vs_ddr.png` overlays DDR (solid coloured line) and ddrs
(dashed black) one panel per reach. Use it for visual sanity-check; the
CSV is the actual gate.

## Hydrograph (`output/hydrograph.{csv,png}`)

`benchmark_hydrograph` writes a **wide** CSV (one column per reach):

```
t_hours,reach_0,reach_1,...,reach_9
0,5.000,5.000,...
1,5.349,5.121,...
...
```

72 rows (3 days of hourly steps), 11 columns. Read with pandas:

```python
import pandas as pd
df = pd.read_csv("output/hydrograph.csv", index_col="t_hours")
df.plot()  # 10 reach hydrographs
```

The PNG is styled to mirror DDR's `plot_routing_hydrograph` (1500×675
px at 150 dpi, white background, tab10 palette, m³/s y-axis). Useful as a
visual smoke test that the routing core hasn't drifted between dev
sessions — the diurnal sweep should peak roughly at the same hours every
run.

## Eval outputs (zarr predictions)

`bin/eval` writes a zarr v3 store at the path given by `--output`. Layout
(see `src/training/zarr_io.rs:14-20`):

```
/predictions    (n_gauges, n_days)  f64   "m^3/s"
/observations   (n_gauges, n_days)  f64   "m^3/s"
/gage_ids       (n_gauges, 8)       u8    fixed-length STAID strings
/time           (n_days,)           i64   nanoseconds since epoch
```

Group attributes record run metadata: `description`, `start time`,
`end time`, `version` (the ddrs `CARGO_PKG_VERSION`),
`evaluation basins file` (the gages CSV path), `model` (checkpoint base
path, or the literal `"frozen"` when `--frozen` was passed).

Read it from xarray:

```python
import xarray as xr
ds = xr.open_zarr("output/model_test.zarr")
print(ds.predictions.shape, ds.attrs["model"])
```

The format is DDR-compatible — DDR's analysis notebooks open it without
modification.

`eval` also logs a one-line summary to stdout:

```
wrote output/model_test.zarr
gauges with finite NSE: 412 / 430
mean NSE (finite only): 0.6843
```

NSE per gauge is **not** written to the zarr — recompute from
`predictions` vs `observations` if you need it persisted.

## Gotchas

- **`output/` must exist before running the examples.** Both
  `compare_ddr_sandbox` and `benchmark_hydrograph` call
  `BufWriter::new(File::create("output/..."))` with no `create_dir_all`
  guard and panic on a missing directory. One-time fix: `mkdir -p output`.
  `train` and `eval` do call `create_dir_all` on `--checkpoint-dir` /
  `--output` so they are forgiving.
- **`.mpk` files are not portable across BURN minor versions.** BURN
  bumps may rename module fields and the `NamedMpkFileRecorder` will
  reject the old file. Re-record after a BURN upgrade; treat checkpoints
  as throwaway across version bumps, not as artefacts to archive
  long-term.
- **`.mpk` files are not portable from DDR either.** DDR `.pt`
  files match the I/O contract of `Mlp<B>` but not the internal
  architecture (DDR's KAN ≠ ddrs's MLP). `eval` rejects them implicitly
  via `load_record`'s shape check.
- **Pass the base path, not `.mpk`, to `load_mlp`.** The recorder
  re-appends the extension. Passing `epoch_5.mpk` produces
  `epoch_5.mpk.mpk` and a load failure.
- **Half-precision saves lose `f32` LSBs.** Don't round-trip a checkpoint
  through save→load→save expecting bit-identity — the first save quantises
  to `f16` and subsequent saves preserve only that.
- **`plotters` axis style depends on the pinned version in `Cargo.toml`.**
  Upgrading will silently change tick labels / line caps in PNGs even
  though the CSVs are unchanged. Pin or pixel-diff if you care.

## Verification

| Path | Covered by |
|---|---|
| V1 CSV row count + verdict | `cargo run --release --example compare_ddr_sandbox` then `wc -l output/ddrs_vs_ddr.csv` (expect 6 = 1 header + 5 reaches) |
| Hydrograph wide format | `cargo run --release --example benchmark_hydrograph` then `head -1 output/hydrograph.csv` (expect `t_hours,reach_0,...,reach_9`) |
| Checkpoint round-trip | `cargo test --lib training::checkpoint` and inspect `<checkpoint-dir>/*.mpk` after `train` |
| Eval zarr layout | `cargo run --release --bin eval -- --frozen --output /tmp/probe.zarr ...` and open with `xarray.open_zarr` |

The V1 CSV+verdict path is the only output that gates correctness; the
others are debugging / interpretability aids.
