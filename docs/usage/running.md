# Running the code

This chapter covers how to build, train, evaluate, and run the
regression examples. ddrs ships three `src/bin/` entry points
(`train`, `eval`, `train_and_test`) and two `examples/`
(`compare_ddr_sandbox`, `benchmark_hydrograph`). All three binaries
are `clap`-parsed and share the same `--config <yaml>` +
`--checkpoint-dir <path>` shape; defaults come from
`config/merit_training.yaml`, which is the verbatim mirror of DDR's
`merit_training_config.yaml`.

Currently shipped defaults are `sparse_solver: cuda` and
`use_cuda_graphs: true` per the SP-9 and SP-10 close commits. The CPU
path stays available as a one-line YAML override.

## Build

```bash
cargo build --release    # binaries + examples, LTO=thin
cargo test               # ~54 tests across 7 integration files + lib units
```

The `--release` profile is mandatory for anything that touches the
routing core. Debug builds are roughly 20× slower and not useful for
the V1 regression. The release profile also enables `lto = "thin"`
(`Cargo.toml` `[profile.release]`), which gives the fused
`forward_chain_inner` chain a measurable extra ~5% from inlining
across the routing/sparse boundary.

`cargo test` itself runs CPU-only (`burn-ndarray`) and exercises the
full unit-test + integration-test set without touching CUDA, so it is
a useful smoke check even on a GPU-less machine.

## Train

```bash
target/release/train \
    --config config/merit_training.yaml \
    --checkpoint-dir output/saved_models
```

Runs **Phase 1 only** — training without the test phase. Writes one
`.mpk` checkpoint per mini-batch to `--checkpoint-dir`. For smoke
tests or `nsys` profiling, cap the inner loop with `--max-mini-batches
N` (still runs all configured epochs, just `N` batches each):

```bash
target/release/train \
    --config config/merit_training.yaml \
    --checkpoint-dir output/saved_models \
    --max-mini-batches 3
```

`train.rs` calls `std::fs::create_dir_all(&cli.checkpoint_dir)` before
training begins, so `--checkpoint-dir` does not need to pre-exist.

## Evaluate

```bash
target/release/eval \
    --config config/merit_training.yaml \
    --checkpoint output/saved_models/epoch_5 \
    --output output/model_test.zarr \
    --batch-size-days 15
```

`--checkpoint` points at the `.mpk` base path **without the suffix** —
the recorder re-appends `.mpk` internally. Passing the full filename
(`epoch_5.mpk`) produces `epoch_5.mpk.mpk` and a load failure.

Use `--frozen` to skip MLP loading and run with scalar default
parameters (the V4 dev path):

```bash
target/release/eval \
    --config config/merit_training.yaml \
    --frozen \
    --output output/model_test.zarr \
    --batch-size-days 15
```

`eval` writes a DDR-compatible zarr v3 store at `--output` and logs an
NSE summary on stdout. See [Reading outputs](outputs.md) for the zarr
layout.

## Train + test (full pipeline)

```bash
target/release/train_and_test \
    --config config/merit_training.yaml \
    --checkpoint-dir output/saved_models \
    --output output/model_test.zarr
```

Sequences Phase 1 (`train()`) then Phase 2 (`evaluate()`) in one
process, auto-discovering the latest `.mpk` in `--checkpoint-dir`
between phases via the `find_latest_mpk` helper. Mirrors DDR's
`scripts/train_and_test.py`. Accepts the same `--max-mini-batches N`
cap as `train`.

This is the canonical "I have a fresh machine and I want to verify the
end-to-end stack" command. If it completes and the post-Phase-2 NSE
summary looks sensible, the full pipeline is sound.

## V1 regression (compare_ddr_sandbox)

```bash
cargo run --release --example compare_ddr_sandbox
```

Reads the DDR-exported fixture under `fixtures/sandbox/` (gitignored —
regenerate via `cd ~/projects/ddr && uv run python
~/projects/ddrs/scripts/export_ddr_sandbox.py`), replays it through
ddrs's `MuskingumCunge` solver, and prints either:

```
verdict: ABSOLUTE MATCH (max abs <REAL> m³/s)
```

or a diff dump if outside `< 1e-3 m³/s`. This is the V1 invariant from
`CLAUDE.md` — must hold after every change to `src/routing/`,
`src/geometry.rs`, `src/sparse/`, or `src/cuda_graph/`. Also writes
`output/ddrs_vs_ddr.csv` + `output/ddrs_vs_ddr.png` for visual
inspection.

Typical passing result on CPU is `max abs ≈ 1.5e-5 m³/s` — roughly
two orders of magnitude under the threshold. That margin is the f32
precision floor; closing it further is not a goal.

To verify the CUDA-graph capture path also produces ABSOLUTE MATCH:

```bash
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox
```

`DDRS_FORCE_GRAPHS=1` toggles the example to the `Cuda<f32, i32>`
inner backend with `use_cuda_graphs=true` regardless of the YAML.
Without it, the example runs CPU-only (`NdArray<f32>`) for
deterministic V1 comparison.

## Hydrograph plot (benchmark_hydrograph)

```bash
cargo run --release --example benchmark_hydrograph
```

Routes a synthetic diurnal lateral-inflow signal through a 10-reach
linear chain for 72 hourly steps and writes:

- `output/hydrograph.csv` — wide CSV, columns `t_hours, reach_0..reach_9`,
  72 data rows.
- `output/hydrograph.png` — overlaid hydrograph plot, 1500×675 px at
  150 dpi.

No fixtures required — useful as a sanity check on the routing core
when the V1 fixture is unavailable, or as a visual smoke test that
the routing core hasn't drifted between dev sessions. The diurnal
sweep should peak roughly at the same hours every run.

## CPU vs CUDA toggles

Two YAML keys under `params:` in `config/merit_training.yaml` switch
the sparse path:

```yaml
params:
  sparse_solver: cuda    # cpu | cuda — selects ndarray vs cuSPARSE SpMV
  use_cuda_graphs: true  # CUDA backend only; forward-only graph capture+replay
```

Current shipped defaults (the YAML literal above) are CUDA-on. On
CPU-only machines, override by editing the YAML or by passing a temp
YAML:

```yaml
params:
  sparse_solver: cpu
  use_cuda_graphs: false
```

The unit tests under `cargo test` exercise the CPU path
(`burn-ndarray`) and run without CUDA. `use_cuda_graphs: true` with
`sparse_solver: cpu` is a silent no-op — the captured kernel sequence
assumes the cuSPARSE path; the CPU sparse solver has nothing to
capture.

See [Formatting inputs](inputs-formatting.md) for the complete list
of toggles and how to add new ones, and
[Performance & CUDA Graphs](../reference/perf.md) for the capture
architecture under the hood.

## Gotchas

- **Checkpoint directory is auto-created.** `train.rs` calls
  `std::fs::create_dir_all(&cli.checkpoint_dir)` before training
  begins, so the path need not pre-exist. `--output` for zarrs does
  need a writable parent — the `eval` binary calls `create_dir_all`
  on the parent of `--output`, but the zarr-creation step itself
  panics if the parent is read-only.
- **Data files must exist.** The binaries panic on missing files at
  the paths listed in [Setup](../setup.md) (MERIT zarr, attributes
  netcdf, icechunk forcing/observations). If a path differs on a new
  machine, edit `config/merit_training.yaml` rather than symlinking —
  symlinks have historically masked stale fixtures.
- **`output/` must exist for the examples.** Both
  `compare_ddr_sandbox` and `benchmark_hydrograph` write CSV+PNG
  directly to `output/` and will panic on
  `BufWriter::new(File::create(...))` if the directory is missing.
  Create it once: `mkdir -p output`.
- **`fixtures/sandbox/` is gitignored.** Missing fixtures →
  `compare_ddr_sandbox` panics at the first CSV read. Regenerate via
  the DDR `uv` venv (see [Setup](../setup.md)).
- **MLP checkpoints are not transferable from DDR.** `eval.rs` accepts
  only ddrs-trained `.mpk` files; DDR's `.pt` weights match the I/O
  contract but not the internal architecture (DDR's KAN ≠ ddrs's
  MLP).
- **`--checkpoint` takes the BASE path, not `.mpk`.** Passing
  `epoch_5.mpk` produces `epoch_5.mpk.mpk` and a load failure.

## Verification

| Path | Covered by |
|---|---|
| Config parse + ranges | `cargo test --lib config::` |
| Routing core (dense + sparse, CPU) | `cargo test --test mmc` |
| Sparse autograd (gradcheck) | `cargo test --test sparse_gradcheck` |
| Data readers (zarr/netcdf) | `cargo test --test data_zarr_store` |
| End-to-end bit-match vs DDR | `cargo run --release --example compare_ddr_sandbox` |
| Graph-capture bit-match | `DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox` |

The V1 example is the only test that locks the cross-language
invariant; the `cargo test` suite covers everything else CPU-side
and runs without CUDA.

## See also

- [Setup](../setup.md) — the prerequisites these commands assume.
- [Formatting inputs](inputs-formatting.md) — what's in
  `config/merit_training.yaml` and how to edit it.
- [Reading inputs](inputs-reading.md) — what the data-source paths
  point at.
- [Reading outputs](outputs.md) — what `train`, `eval`, and the
  examples write.
- [Comparing to DDR](../reference/ddr-comparison.md) — the V1
  regression in detail.
- [Performance & CUDA Graphs](../reference/perf.md) — what the CUDA
  toggles actually do.
