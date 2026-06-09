# Running the code

This chapter covers how to build ddrs, train and evaluate a KAN routing
head, run the two regression examples, and flip between the CPU and CUDA
paths. The preferred entrypoint is the `ddrs` CLI
(`init → plan → run --workflow ...`); the three legacy `src/bin/`
binaries (`train`, `eval`, `train_and_test`) still exist but print a
deprecation warning on entry and are slated for removal in 0.4. Both are
documented here, with the `ddrs run` equivalents called out alongside
each legacy invocation. Configuration defaults come from
`config/merit_training.yaml`, the verbatim mirror of DDR's
`merit_training_config.yaml`, which ships with `sparse_solver: cuda` and
`use_cuda_graphs: true`.

## What it is

ddrs exposes two layers of entrypoints:

- **The `ddrs` CLI** — a terraform-style lifecycle
  (`ddrs init`, `ddrs plan`, `ddrs run --workflow {train,eval,train-and-test}`).
  This is the documented, supported path; see `CLAUDE.md` and the lifecycle
  spec under `docs/superpowers/specs/` for the full surface.
- **Three legacy `src/bin/` binaries** — `train` (Phase 1 only), `eval`
  (Phase 2 only), and `train_and_test` (both phases in one process). Each
  is `clap`-parsed and each prints, for example:

  ```
  warning: `train` is deprecated and will be removed in 0.4. use `ddrs run --workflow train` instead.
  ```

  on `stderr` before doing any work. They share the `--config <yaml>`
  shape and were the original interface before the `ddrs` CLI landed.

Two `examples/` round out the set: `compare_ddr_sandbox` (the V1
cross-language regression against DDR) and `benchmark_hydrograph` (a
fixture-free routing sanity check). Outputs land under `output/` at the
repo root — `.mpk` checkpoints under `output/saved_models/`, prediction
zarrs at `output/model_test.zarr`, and the example artifacts
(`hydrograph.{csv,png}`, `ddrs_vs_ddr.{csv,png}`) directly in `output/`.

## How to use it

### Build

```bash
cargo build --release    # binaries + examples, LTO=thin
cargo test               # ~54 tests across 7 integration files + lib units
```

The `--release` profile is mandatory for anything that touches the
routing core. Debug builds are roughly 20× slower and not useful for the
V1 regression. The release profile also enables `lto = "thin"`
(`Cargo.toml` `[profile.release]`), which gives the fused routing chain a
measurable extra inlining win across the routing/sparse boundary.

`cargo test` runs CPU-only (`burn-ndarray`) and exercises the full
unit-test plus integration-test set without touching CUDA, so it is a
useful smoke check even on a GPU-less machine.

### Train

Preferred:

```bash
ddrs run --workflow train
```

Legacy equivalent:

```bash
target/release/train \
    --config config/merit_training.yaml \
    --checkpoint-dir output/saved_models
```

Runs **Phase 1 only** — training without the test phase. Writes one
checkpoint per mini-batch under `--checkpoint-dir`, each a directory
`epoch_E_mb_M/` (see [Checkpoints and resume](#checkpoints-and-resume)
below). For smoke tests or `nsys` profiling, cap the inner loop with
`--max-mini-batches N` (still runs all configured epochs, just `N`
batches each):

```bash
target/release/train \
    --config config/merit_training.yaml \
    --checkpoint-dir output/saved_models \
    --max-mini-batches 3
```

`train.rs` calls `std::fs::create_dir_all(&cli.checkpoint_dir)` before
training begins, so `--checkpoint-dir` does not need to pre-exist. On
startup it prints the gauge count, date range, epoch count, and the
active `sparse_solver` / `use_cuda_graphs` settings.

### Evaluate

Preferred:

```bash
ddrs run --workflow eval
```

Legacy equivalent:

```bash
target/release/eval \
    --config config/merit_training.yaml \
    --checkpoint output/saved_models/epoch_5_mb_8 \
    --output output/model_test.zarr \
    --batch-size-days 15
```

`--checkpoint` points at the checkpoint **directory** `epoch_E_mb_M/`
(which holds `head.mpk`); the binary derives the recorder base from it via
`head_base`, which appends the `.mpk` suffix internally. `--batch-size-days`
defaults to `15`, matching DDR's test config.

Use `--frozen` to skip KAN-head loading and run with scalar default
parameters (the V4 dev path). With `--frozen`, `--checkpoint` is optional;
without it, the binary exits with status `2` if no `--checkpoint` is given:

```bash
target/release/eval \
    --config config/merit_training.yaml \
    --frozen \
    --output output/model_test.zarr \
    --batch-size-days 15
```

`eval` seeds the backend RNG from `cfg.seed` for deterministic head-template
init, writes a DDR-compatible zarr at `--output`, and logs a metrics summary
on stdout: the count of gauges with finite NSE plus the **median** NSE and
KGE (means are omitted because the NSE distribution is right-skewed — a few
bad gauges drag the mean). See [Reading outputs](outputs.md) for the zarr
layout.

### Train + test (full pipeline)

Preferred:

```bash
ddrs run --workflow train-and-test
```

Legacy equivalent:

```bash
target/release/train_and_test \
    --config config/merit_training.yaml \
    --checkpoint-dir output/saved_models \
    --output output/model_test.zarr
```

Sequences Phase 1 (`train()`) then Phase 2 (`evaluate()`) in one process,
auto-discovering the latest checkpoint directory in `--checkpoint-dir`
between phases via the `find_latest_mpk` helper (which parses
`epoch_E_mb_M` names and picks the max by `(epoch, mb)`). It reloads the
config in `Testing` mode for Phase 2, drops the Phase 1 optimizer and
dataset to free GPU memory first, and accepts the same `--max-mini-batches
N` cap as `train`. Mirrors DDR's `scripts/train_and_test.py`.

This is the canonical "I have a fresh machine and I want to verify the
end-to-end stack" command. If it completes and the post-Phase-2 NSE
summary looks sensible, the full pipeline is sound.

### Checkpoints and resume

A checkpoint is a **directory** `epoch_E_mb_M/` holding three fixed-name
files:

| File | Contents | Format |
|---|---|---|
| `head.mpk` | KAN head weights | burn `CompactRecorder` (f16) |
| `optim.mpk` | Adam record (both moment tensors) | burn `CompactRecorder` (f16) |
| `state.json` | epoch, next mini-batch, serialized rng, sampler permutation + cursor | JSON |

Resume is a `ddrs` CLI feature, driven by `experiment.checkpoint:` in
`ddrs.yaml`:

```yaml
# ddrs.yaml
experiment:
  epochs: 50                                                   # must exceed the checkpoint's epoch
  checkpoint: .ddrs/runs/<run-id>/checkpoints/epoch_25_mb_8    # the directory
```

then `ddrs run --workflow train` (or `train-and-test`). On resume,
`bootstrap_head_and_state` (`src/training/bootstrap.rs`) restores all three
files, so the resumed run continues at the **true epoch / mini-batch** (the
learning-rate schedule keys correctly), draws the **same gauge batches** —
including the remainder of an in-flight epoch's shuffle — and the same
rho-windows, and steps Adam with **warm moments** rather than restarting
cold. Remember to raise `experiment.epochs` past the checkpoint's epoch, or
the resumed run trains zero batches.

Resume position is exact, but the stored weights and moments are f16
(`CompactRecorder` uses half-precision settings), so a resumed trajectory
drifts slowly from an uninterrupted one — a known follow-up tracked in
`docs/2026-06-07-checkpoint-resume-handoff.md`. Old checkpoints written
before the directory layout resume weights-only: Adam cold, epoch counter
back at 1, fresh shuffle.

### V1 regression (compare_ddr_sandbox)

```bash
cargo run --release --example compare_ddr_sandbox
```

Reads the DDR-exported fixture under `fixtures/sandbox/` (gitignored —
regenerate via `cd ~/projects/ddr && uv run python
~/projects/ddrs/scripts/export_ddr_sandbox.py`), replays it through ddrs's
`MuskingumCunge` solver, reorders the output to RAPID2 order, and prints a
per-reach diff table followed by a verdict:

```
verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

This is the V1 invariant from `CLAUDE.md` — it must hold after every
change to `src/routing/`, `src/geometry.rs`, or `src/sparse.rs`. The
example also writes `output/ddrs_vs_ddr.csv` (per-reach max/mean abs diff,
max rel diff, means, Pearson correlation) and `output/ddrs_vs_ddr.png`
(both hydrographs overlaid) for visual inspection. If the overall max abs
diff exceeds `1e-3` but max rel diff is under 1%, the verdict softens to
`close match`; beyond that it reports `DIVERGENCE — investigate`.

By default the example runs on the CPU inner backend (`NdArray<f32>`) for
deterministic comparison. To verify the CUDA-graph capture path also
produces ABSOLUTE MATCH:

```bash
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox
```

Setting `DDRS_FORCE_GRAPHS=1` dispatches the run through the
`Cuda<f32, i32>` inner backend with `use_cuda_graphs=true`, regardless of
any YAML — the example builds its own `RoutingInputs`, so it does not read
`config/merit_training.yaml` for the backend choice.

### Hydrograph plot (benchmark_hydrograph)

```bash
cargo run --release --example benchmark_hydrograph
```

Routes a synthetic diurnal lateral-inflow signal (5 m³/s baseline plus a
±2 m³/s sine sweep) through a 10-reach linear chain for 72 hourly steps
and writes:

- `output/hydrograph.csv` — wide CSV, columns `t_hours, reach_0..reach_9`,
  72 data rows.
- `output/hydrograph.png` — one line per reach, 1500×675 px at 150 dpi,
  styled to match DDR's `plot_routing_hydrograph`.

No fixtures required — useful as a sanity check on the routing core when
the V1 fixture is unavailable, or as a visual smoke test that the routing
core hasn't drifted between dev sessions. It also prints setup/forward
timings and per-reach min/mean/max discharge to the terminal.

### CPU vs CUDA toggles

Two YAML keys under `params:` in `config/merit_training.yaml` switch the
sparse path:

```yaml
params:
  sparse_solver: cuda    # cpu | cuda — selects ndarray vs cuSPARSE SpMV
  use_cuda_graphs: true  # CUDA backend only; forward-only graph capture+replay
```

The shipped defaults (the literal above) are CUDA-on. On CPU-only
machines, override by editing the YAML or by passing a temp YAML:

```yaml
params:
  sparse_solver: cpu
  use_cuda_graphs: false
```

`use_cuda_graphs: true` paired with `sparse_solver: cpu` is a silent
no-op: the captured kernel sequence assumes the cuSPARSE path, and the CPU
sparse solver has nothing to capture. The unit tests under `cargo test`
exercise the CPU path (`burn-ndarray`) and run without CUDA.

A third, top-level key selects the GPU on multi-GPU hosts:

```yaml
device: 0              # CUDA device ordinal (mirrors DDR's `device:` key)
```

`device: 1` runs the entire training (tensors, cuSPARSE cache, graph
capture/replay) on the second GPU. The legacy binaries read this key as
`cfg.device` and pass it to `CudaDevice::new(...)`; it is validated by
`tests/device_selection.rs` (which skips on hosts with fewer than 2 GPUs).

See [Formatting inputs](inputs-formatting.md) for the complete list of
toggles, and [Performance & CUDA Graphs](../reference/perf.md) for the
capture architecture under the hood.

## Reference

### Legacy binary flags

| Binary | Flag | Type / default | Meaning |
|---|---|---|---|
| `train` | `--config` | path (required) | YAML config |
| `train` | `--checkpoint-dir` | path (required) | Per-mini-batch checkpoint directory |
| `train` | `--max-mini-batches` | usize (optional) | Cap mini-batches per epoch |
| `eval` | `--config` | path (required) | YAML config |
| `eval` | `--checkpoint` | path (optional) | Checkpoint dir `epoch_E_mb_M/`; required unless `--frozen` |
| `eval` | `--output` | path (required) | Output zarr path |
| `eval` | `--batch-size-days` | usize, default `15` | Days per chunk |
| `eval` | `--frozen` | flag | Use frozen scalar params instead of a KAN head |
| `train_and_test` | `--config` | path (required) | YAML config |
| `train_and_test` | `--checkpoint-dir` | path (required) | Phase 1 writes / Phase 2 discovers here |
| `train_and_test` | `--output` | path (required) | Phase 2 predictions zarr |
| `train_and_test` | `--batch-size-days` | usize, default `15` | Days per chunk in Phase 2 |
| `train_and_test` | `--max-mini-batches` | usize (optional) | Cap Phase 1 mini-batches |

The `ddrs run --workflow {train,eval,train-and-test}` subcommands replace
these; they read the same `config/merit_training.yaml` keys but manage
checkpoint directories and outputs through the `.ddrs/runs/<id>/`
workspace rather than explicit `--checkpoint-dir` / `--output` flags. See
`CLAUDE.md` for the CLI surface.

### Examples

| Example | Inputs | Writes | Purpose |
|---|---|---|---|
| `compare_ddr_sandbox` | `fixtures/sandbox/` CSVs | `output/ddrs_vs_ddr.{csv,png}` | V1 cross-language regression (`< 1e-3 m³/s`) |
| `benchmark_hydrograph` | none | `output/hydrograph.{csv,png}` | Fixture-free routing sanity check |

### Verification matrix

| Path | Covered by |
|---|---|
| Config parse + ranges | `cargo test --lib config::` |
| Routing core (dense + sparse, CPU) | `cargo test --test mmc` |
| Sparse autograd (gradcheck) | `cargo test --test sparse_gradcheck` |
| Data readers (zarr/netcdf) | `cargo test --test data_zarr_store` |
| End-to-end bit-match vs DDR | `cargo run --release --example compare_ddr_sandbox` |
| Graph-capture bit-match | `DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox` |

The V1 example is the only test that locks the cross-language invariant;
the `cargo test` suite covers everything else CPU-side and runs without
CUDA.

### Gotchas

- **Checkpoint directory is auto-created.** `train.rs` and
  `train_and_test.rs` call `std::fs::create_dir_all(&cli.checkpoint_dir)`
  before training begins, so the path need not pre-exist;
  `train_and_test` also creates the parent of `--output`.
- **`--checkpoint` is a directory, not a `.mpk` file.** It points at
  `epoch_E_mb_M/`; the recorder appends `.mpk` to the derived base. Passing
  a path that already ends in `.mpk` will produce a double-suffix load
  failure.
- **Data files must exist.** The binaries panic on missing files at the
  paths listed in [Setup](../setup.md) (MERIT zarr, attributes netcdf,
  icechunk forcing/observations). If a path differs on a new machine, edit
  `config/merit_training.yaml` rather than symlinking.
- **`output/` must exist for the examples.** Both `compare_ddr_sandbox`
  and `benchmark_hydrograph` write CSV+PNG directly to `output/` and will
  panic on `BufWriter::new(File::create(...))` if the directory is
  missing. Create it once: `mkdir -p output`.
- **`fixtures/sandbox/` is gitignored.** Missing fixtures →
  `compare_ddr_sandbox` panics at the first CSV read. Regenerate via the
  DDR `uv` venv (see [Setup](../setup.md)).
- **KAN-head checkpoints are not transferable from DDR.** `eval` accepts
  only ddrs-trained `.mpk` files; DDR's `.pt` weights match the I/O
  contract but not the internal record layout.

## See also

- [Setup](../setup.md) — the prerequisites these commands assume.
- [Formatting inputs](inputs-formatting.md) — what's in
  `config/merit_training.yaml` and how to edit it.
- [Reading inputs](inputs-reading.md) — what the data-source paths point
  at.
- [Reading outputs](outputs.md) — what `train`, `eval`, and the examples
  write.
- [Comparing to DDR](../reference/ddr-comparison.md) — the V1 regression
  in detail.
- [Performance & CUDA Graphs](../reference/perf.md) — what the CUDA
  toggles actually do.
