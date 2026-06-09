# Reading outputs

Every ddrs binary writes its artefacts to one of two places: the
repo-local `output/` directory (the two examples, plus `eval` by
default when you pass `--output output/...`) or a user-supplied
`--checkpoint-dir` (training). Four on-disk shapes appear: a checkpoint
*directory* `epoch_E_mb_M/` holding two binary `.mpk` records plus a
JSON sidecar (BURN's NamedMessagePack format needs `CompactRecorder` to
read; the sidecar is plain JSON), plain CSV (readable with any tool —
pandas/polars/awk), PNG (rendered with `plotters`), and a zarr v3 store
from `eval`. There is no global "results directory" — each binary owns
its own paths.

| Artefact | Producer | Path | Format |
|---|---|---|---|
| Training checkpoint | `train` (Phase 1 of `train-and-test`) | `<checkpoint-dir>/epoch_{e}_mb_{mb}/` (directory) | BURN MPK + JSON |
| V1 diff | `examples/compare_ddr_sandbox` | `output/ddrs_vs_ddr.{csv,png}` | CSV + PNG |
| Hydrograph | `examples/benchmark_hydrograph` | `output/hydrograph.{csv,png}` | CSV + PNG |
| Eval predictions | `bin/eval` | user `--output` (typically `*.zarr`) | zarr v3 |

## What it is

The three artefacts answer three different questions. The **checkpoint
directory** is the trained model plus everything needed to resume the
exact training trajectory. The **V1 diff CSV** is the correctness gate
against the DDR reference — the only output that decides whether the
port is still faithful. The **hydrograph** and **eval zarr** are
inspection aids: a routing-core smoke test and the DDR-compatible
predictions table, respectively.

## How to use it

### Training checkpoints (`epoch_E_mb_M/` directories)

`train.rs` (and the Phase-1 half of `train-and-test`) writes **one
checkpoint per mini-batch** under `--checkpoint-dir`. As of the
exact-resume work, a checkpoint is a **directory**, not a single file:

```text
<checkpoint-dir>/epoch_3_mb_8/
├── head.mpk      KAN head weights   (CompactRecorder, f16 on disk)
├── optim.mpk     Adam moments       (CompactRecorder, f16 on disk)
└── state.json    epoch, next mini-batch, serialized rng, sampler
                  permutation + cursor
```

The directory name is `epoch_{epoch}_mb_{mini_batch}` (see
`src/training/driver.rs`, which calls `create_dir_all` on it then writes
the three files via `head_base`/`optim_base`/`state_path` from
`src/training/checkpoint.rs`). The `.mpk` extension on `head.mpk` and
`optim.mpk` is appended by the recorder — the in-code *base* paths are
`dir/head` and `dir/optim`.

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

`eval` does exactly this: `--checkpoint` takes the `epoch_E_mb_M/`
directory, and `load_kan_head(&head_base(...))` reads `head.mpk` from
inside it. To resume training instead, point `experiment.checkpoint:`
in `ddrs.yaml` at the same directory; `bootstrap_head_and_state`
restores head + optimizer + `state.json` so the resumed run draws the
same gauge batches and rho-windows the original would have.

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

### Eval outputs (zarr predictions)

`bin/eval` writes a zarr v3 store at the path given by `--output`.
Layout (see `src/training/zarr_io.rs`):

```
/predictions    (n_gauges, n_days)  f64   units "m3/s"
/observations   (n_gauges, n_days)  f64   units "m3/s"
/gage_ids       (n_gauges, 8)       u8    fixed-width ASCII STAID (_dtype_hint "|S8")
/time           (n_days,)           i64   nanoseconds since 1970-01-01
```

Group attributes record run metadata: `description`, `start time`,
`end time`, `version` (the ddrs `CARGO_PKG_VERSION`), `evaluation basins
file` (the gages CSV path), and `model` (the `--checkpoint` directory
path, or the literal `"frozen"` when `--frozen` was passed).

Read it from xarray:

```python
import xarray as xr
ds = xr.open_zarr("output/model_test.zarr")
print(ds.predictions.shape, ds.attrs["model"])
```

The format is DDR-compatible — DDR's analysis notebooks open it without
modification.

`eval` also logs a metrics summary to stdout. Per-gauge mean is
misleading on right-skewed NSE distributions, so only the **median** is
reported:

```
wrote output/model_test.zarr
gauges with finite NSE: 412 / 430
median NSE (finite only): 0.6843
median KGE (finite only): 0.7012
```

Per-gauge NSE/KGE are **not** written to the zarr — recompute from
`predictions` vs `observations` if you need them persisted.

## Reference

### Gotchas

- **`output/` must exist before running the examples.** Both
  `compare_ddr_sandbox` and `benchmark_hydrograph` call
  `BufWriter::new(File::create("output/..."))` with no `create_dir_all`
  guard and panic on a missing directory. One-time fix: `mkdir -p
  output`. `train` and `eval` do call `create_dir_all` on
  `--checkpoint-dir` / `--output`, so they are forgiving.
- **A checkpoint is a directory, not a file.** Pass the
  `epoch_E_mb_M/` directory to `eval --checkpoint` and to
  `experiment.checkpoint:`. The inner filenames (`head.mpk`,
  `optim.mpk`, `state.json`) are hardcoded; do not point at one of the
  inner files. The whole directory copies/deletes as one unit, so
  nothing can clobber the head.
- **Pass the base path, not `.mpk`, to the loaders.** `head_base` /
  `optim_base` return `dir/head` / `dir/optim`; the recorder re-appends
  `.mpk`. Passing `head.mpk` produces `head.mpk.mpk` and a load failure.
- **`.mpk` files are not portable across BURN minor versions.** BURN
  bumps may rename module fields and the `NamedMpkFileRecorder` will
  reject the old file. Re-record after a BURN upgrade; treat checkpoints
  as throwaway across version bumps, not as artefacts to archive
  long-term.
- **`.mpk` files are not portable from DDR either.** DDR's `.pt` files
  match the KAN head's I/O contract but not its on-disk record format;
  `load_kan_head`'s `load_record` rejects them.
- **Half-precision saves lose `f32` LSBs.** Don't round-trip a checkpoint
  through save→load→save expecting bit-identity — the first save
  quantises to `f16` and subsequent saves preserve only that. This is why
  a resumed trajectory drifts slowly from an uninterrupted one.
- **`plotters` axis style depends on the pinned version in
  `Cargo.toml`.** Upgrading will silently change tick labels / line caps
  in PNGs even though the CSVs are unchanged. Pin or pixel-diff if you
  care.

### Verification

| Path | Covered by |
|---|---|
| V1 CSV row count + verdict | `cargo run --release --example compare_ddr_sandbox` then `wc -l output/ddrs_vs_ddr.csv` (expect 6 = 1 header + 5 reaches) |
| Hydrograph wide format | `cargo run --release --example benchmark_hydrograph` then `head -1 output/hydrograph.csv` (expect `t_hours,reach_0,...,reach_9`) |
| Checkpoint round-trip | `cargo test --lib training::checkpoint` and inspect `<checkpoint-dir>/epoch_*_mb_*/` after `train` |
| Eval zarr layout | `cargo run --release --bin eval -- --frozen --output /tmp/probe.zarr ...` and open with `xarray.open_zarr` |

The V1 CSV + verdict path is the only output that gates correctness; the
others are debugging / interpretability aids.

## See also

- [Running the code](running.md) — the producer side of each artefact,
  including the `ddrs run` lifecycle that supersedes the deprecated
  `train`/`eval` binaries.
- [Comparing to DDR](../reference/ddr-comparison.md) — what the V1 diff
  CSV measures and how to interpret it.
- [The baseline](../reference/baseline.md) — the summed-Q′ reference the
  eval predictions are judged against.
- [Formatting inputs](inputs-formatting.md) — how to point the binaries
  at different output paths and set `experiment.checkpoint:`.
