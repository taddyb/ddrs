---
name: ddrs-running-the-code
description: How to build, train, evaluate, and run the regression examples. Covers the train/eval/train_and_test binaries, the compare_ddr_sandbox and benchmark_hydrograph examples, and the CPU vs CUDA + use_cuda_graphs toggles.
output: usage/running.md
sources:
  - src/bin/train.rs
  - src/bin/eval.rs
  - src/bin/train_and_test.rs
  - examples/compare_ddr_sandbox.rs
  - examples/benchmark_hydrograph.rs
---

# ddrs-running-the-code

> Canonical agent-readable skill. Published chapter at `docs/usage/running.md`
> is regenerated from this file by `/regenerate-docs`.

## What to know

ddrs ships three `src/bin/` entrypoints and two `examples/`. All three binaries
are `clap`-parsed and share the same `--config <yaml>` + `--checkpoint-dir
<path>` shape. Defaults come from `config/merit_training.yaml` (verbatim mirror
of DDR's `merit_training_config.yaml`), which currently selects
`sparse_solver: cuda` and `use_cuda_graphs: true` per the SP-9/SP-10 close
commits. Outputs land under `output/` at the repo root: `.mpk` checkpoints
per mini-batch under `output/saved_models/`, prediction zarrs at
`output/model_test.zarr`, and the example artifacts (`hydrograph.{csv,png}`,
`ddrs_vs_ddr.{csv,png}`) directly in `output/`.

## Build

```bash
cargo build --release    # binaries + examples, LTO=thin
cargo test               # ~54 tests across 7 integration files + lib units
```

The `--release` profile is mandatory for anything that touches the routing
core; debug builds are ~20× slower and not useful for the V1 regression.

## Train

```bash
target/release/train \
    --config config/merit_training.yaml \
    --checkpoint-dir output/saved_models
```

Runs Phase 1 only (no test phase). Writes one checkpoint per mini-batch to
`--checkpoint-dir`, each a directory `epoch_E_mb_M/` (holding
`head.mpk`/`optim.mpk`/`state.json`). For smoke tests / nsys profiling, cap
the inner loop with
`--max-mini-batches N` (still runs all configured epochs, just `N` batches
each).

## Evaluate

```bash
target/release/eval \
    --config config/merit_training.yaml \
    --checkpoint output/saved_models/epoch_5_mb_8 \
    --output output/model_test.zarr \
    --batch-size-days 15
```

`--checkpoint` points at the checkpoint **directory** `epoch_E_mb_M/`
(which holds `head.mpk`); the binary derives the recorder base from it via
`head_base`, which appends the `.mpk` suffix internally. Use `--frozen` to
skip KAN-head loading and run with scalar default parameters (V4 dev path).
Writes a DDR-compatible zarr at `--output` and logs an NSE summary.

## Train + test (full pipeline)

```bash
target/release/train_and_test \
    --config config/merit_training.yaml \
    --checkpoint-dir output/saved_models \
    --output output/model_test.zarr
```

Sequences Phase 1 (`train()`) then Phase 2 (`evaluate()`) in one process,
auto-discovering the latest checkpoint directory `epoch_E_mb_M/` in
`--checkpoint-dir` between phases. Mirrors DDR's `scripts/train_and_test.py`.
Accepts the same `--max-mini-batches N` cap.

## V1 regression (compare_ddr_sandbox)

```bash
cargo run --release --example compare_ddr_sandbox
```

Reads the DDR-exported fixture under `fixtures/sandbox/` (gitignored —
regenerate via `cd ~/projects/ddr && uv run python
~/projects/ddrs/scripts/export_ddr_sandbox.py`), replays it through ddrs, and
prints either:

```
ABSOLUTE MATCH: max abs <REAL> m³/s
```

or a diff dump if outside `< 1e-3 m³/s`. This is the V1 invariant from
`CLAUDE.md` — must hold after every change to `src/routing/`,
`src/geometry.rs`, or `src/sparse.rs`. Also writes `output/ddrs_vs_ddr.csv`
+ `output/ddrs_vs_ddr.png` for visual inspection.

## Hydrograph plot (benchmark_hydrograph)

```bash
cargo run --release --example benchmark_hydrograph
```

Routes a synthetic diurnal lateral-inflow signal through a 10-reach linear
chain for 72 hourly steps and writes `output/hydrograph.csv` (columns:
`t_hours, reach_0..reach_9`) plus `output/hydrograph.png`. No fixtures
required — useful as a sanity check on the routing core when the V1
fixture is unavailable.

## CPU vs CUDA toggles

Two YAML keys under `params:` in `config/merit_training.yaml` switch the
sparse path:

```yaml
params:
  sparse_solver: cuda    # cpu | cuda — selects ndarray vs cuSPARSE SpMV
  use_cuda_graphs: true  # CUDA backend only; forward-only graph capture+replay
```

A third, top-level key selects the GPU on multi-GPU hosts:

```yaml
device: 0              # CUDA device ordinal (mirrors DDR's `device:` key)
```

`device: 1` runs the entire training (tensors, cuSPARSE cache, graph
capture/replay) on the second GPU. Validated by
`tests/device_selection.rs` (skips on hosts with fewer than 2 GPUs).

Current shipped defaults are `cuda` + `true` (per SP-9 and SP-10 close
commits). On CPU-only machines, override by editing the YAML or by passing a
temp YAML. The unit tests under `cargo test` exercise the CPU path
(`burn-ndarray`) and run without CUDA.

`compare_ddr_sandbox` honours `DDRS_FORCE_GRAPHS=1` as an env override —
when set, the example dispatches to the `Cuda<f32, i32>` inner backend with
`use_cuda_graphs=true` regardless of the YAML. Without it, the example
runs CPU-only (`NdArray<f32>`) for deterministic V1 comparison.

## Gotchas

- **Checkpoint directory is auto-created.** `train.rs` calls
  `std::fs::create_dir_all(&cli.checkpoint_dir)` before training begins, so
  the path need not pre-exist. `--output` for zarrs does need a writable
  parent.
- **Data files must exist.** The binaries panic on missing files at the
  paths listed in `CLAUDE.md` (MERIT zarr, attributes netcdf, icechunk
  forcing/observations). If a path differs on a new machine, edit
  `config/merit_training.yaml` rather than symlinking.
- **`output/` must exist for the examples.** Both `compare_ddr_sandbox`
  and `benchmark_hydrograph` write CSV+PNG directly to `output/` and will
  panic on `BufWriter::new(File::create(...))` if the directory is missing.
  Create it once: `mkdir -p output`.
- **fixtures/sandbox/ is gitignored.** Missing fixtures → `compare_ddr_sandbox`
  panics at the first CSV read. Regenerate via the DDR `uv` venv (see
  Setup skill).
- **KAN-head checkpoints are not transferable from DDR.** `eval.rs` accepts
  only ddrs-trained `.mpk` files; DDR's `.pt` weights match the I/O contract
  but not the internal record layout.

## Verification

| Path | Covered by |
|---|---|
| Config parse + ranges | `cargo test --lib config::` |
| Routing core (dense + sparse, CPU) | `cargo test --test mmc` |
| Sparse autograd (gradcheck) | `cargo test --test sparse_gradcheck` |
| Data readers (zarr/netcdf) | `cargo test --test data_zarr_store` |
| End-to-end bit-match vs DDR | `cargo run --release --example compare_ddr_sandbox` |

The V1 example is the only test that locks the cross-language invariant;
the `cargo test` suite covers everything else CPU-side and runs without
CUDA.
