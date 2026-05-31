# Setup

This chapter walks through the four external dependencies a fresh ddrs
checkout resolves against — the Rust toolchain, the `taddyb/cubecl` fork
on branch `ddrs-release`, the CUDA toolkit + driver for the GPU path,
and the DDR reference repository used to regenerate the V1 sandbox
fixture — plus the canonical data-source paths referenced from
`config/merit_training.yaml`. The expected end-state is a working
release build whose V1 regression reports `ABSOLUTE MATCH`.

## What you need

ddrs's `Cargo.toml` resolves against four things that are not on
crates.io and not pure-Rust pip-installable. Get them right once at
setup; everything downstream just works.

1. **Rust stable** — version 1.80 or later. ddrs has been built and
   tested on `1.94.0`; the language features in use (let-else, GATs in
   `Backend`, `impl Trait` in trait bounds) all stabilized well before
   1.80, so any recent stable will compile. Install via `rustup`:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup default stable
   ```

2. **The `taddyb/cubecl` fork on branch `ddrs-release`.** The
   `[patch.crates-io]` block in `Cargo.toml` redirects every
   `cubecl-*` and `burn-*` crate to local clones under
   `/home/tbindas/projects/cubecl/crates/` and
   `/home/tbindas/projects/burn/crates/`:

   ```toml
   [patch.crates-io]
   cubecl         = { path = "/home/tbindas/projects/cubecl/crates/cubecl" }
   cubecl-cuda    = { path = "/home/tbindas/projects/cubecl/crates/cubecl-cuda" }
   cubecl-common  = { path = "/home/tbindas/projects/cubecl/crates/cubecl-common" }
   # ...one entry per cubecl-* and burn-* crate
   ```

   The fork carries three ddrs-specific patches over upstream cubecl
   0.10:

   - `stream_accessor` (SP-7) — exposes the thread-bound cubecl stream
     so ddrs can submit GPU work on the same stream the JIT compiles
     into.
   - `exclusive_with_server` (SP-9) — re-entrant safe acquisition of
     the cubecl server context, needed before binding a thread to the
     CUDA stream once at capture time.
   - `flush_no_sync` (SP-10) — flush variant that submits queued work
     without the `cuEventSynchronize` call that would invalidate a
     stream mid-capture.

3. **CUDA Toolkit 12+ with driver 595+** for the GPU path. ddrs's
   current `config/merit_training.yaml` defaults to `sparse_solver:
   cuda` and `use_cuda_graphs: true`, so the GPU path is exercised on
   every default training run. The CPU path needs none of this — it
   uses `burn-ndarray` and runs end-to-end without a single CUDA call.
   The unit tests under `cargo test` exercise the CPU path and pass on
   a CUDA-less machine.

4. **DDR reference repository** at `/home/tbindas/projects/ddr` for V1
   fixture regeneration. `scripts/export_ddr_sandbox.py` runs under
   DDR's `uv` venv to produce the six CSVs that
   `examples/compare_ddr_sandbox.rs` reads back to verify the port.

## Data file paths

The training config `config/merit_training.yaml` references five
on-disk data sources by absolute path. From the
`data_sources:` block:

```yaml
data_sources:
  attributes: /home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc
  conus_adjacency: /home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr
  gages_adjacency: /home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr
  streamflow: /mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic
  observations: /mnt/ssd1/data/icechunk/usgs_daily_observations
  gages: /home/tbindas/projects/ddr/references/gage_info/gages_3000.csv
```

Mapping to the data layer:

| Source | Path | Reader |
|---|---|---|
| MERIT adjacency | `~/projects/ddr/data/merit_conus_adjacency.zarr` | `zarrs` (`src/data/store/zarr.rs::ConusAdjacencyStore`) |
| Per-gauge subgraphs | `~/projects/ddr/data/merit_gages_conus_adjacency.zarr` | `zarrs` (`GagesAdjacencyStore`) |
| Catchment attributes | `~/projects/ddr/data/merit_global_attributes_v2.nc` | `netcdf` (TODO per `CLAUDE.md`) |
| Streamflow forcing | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` | `icechunk` (TODO) |
| USGS observations | `/mnt/ssd1/data/icechunk/usgs_daily_observations` | `icechunk` (TODO) |
| Gauges list | `~/projects/ddr/references/gage_info/gages_3000.csv` | `csv` |

If any of these live elsewhere on your machine, edit the YAML rather
than symlinking — symlinks under `~/projects/ddr/data/` have historically
masked stale fixtures.

## Setup steps

The full bring-up is five commands. Run them in order; the last is the
sanity gate that proves the rest worked.

```bash
# 1. Clone ddrs + cubecl fork.
git clone git@github.com:taddyb/ddrs ~/projects/ddrs
git clone -b ddrs-release git@github.com:taddyb/cubecl ~/projects/cubecl

# 2. Verify the cubecl path in Cargo.toml matches your clone location.
grep "/home/.*cubecl/crates/" ~/projects/ddrs/Cargo.toml

# 3. Clone the DDR reference (for V1 fixtures).
git clone git@github.com:mhpi/ddr ~/projects/ddr
cd ~/projects/ddr && uv sync --all-packages

# 4. Build.
cd ~/projects/ddrs
cargo build --release

# 5. Sanity check: V1 regression must report ABSOLUTE MATCH.
cargo run --release --example compare_ddr_sandbox
```

A passing V1 run prints

```
verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

on stdout and writes `output/ddrs_vs_ddr.{csv,png}` for visual
inspection. If you see anything other than `ABSOLUTE MATCH`, your
setup is wrong — see Gotchas below before reaching for the routing
code.

## The `[patch.crates-io]` block in detail

The full patch list from `Cargo.toml` redirects every cubecl crate and
every BURN crate to local paths:

```toml
[patch.crates-io]
cubecl         = { path = "/home/tbindas/projects/cubecl/crates/cubecl" }
cubecl-cuda    = { path = "/home/tbindas/projects/cubecl/crates/cubecl-cuda" }
cubecl-common  = { path = "/home/tbindas/projects/cubecl/crates/cubecl-common" }
cubecl-core    = { path = "/home/tbindas/projects/cubecl/crates/cubecl-core" }
cubecl-cpp     = { path = "/home/tbindas/projects/cubecl/crates/cubecl-cpp" }
cubecl-ir      = { path = "/home/tbindas/projects/cubecl/crates/cubecl-ir" }
cubecl-macros  = { path = "/home/tbindas/projects/cubecl/crates/cubecl-macros" }
cubecl-opt     = { path = "/home/tbindas/projects/cubecl/crates/cubecl-opt" }
cubecl-runtime = { path = "/home/tbindas/projects/cubecl/crates/cubecl-runtime" }
cubecl-std     = { path = "/home/tbindas/projects/cubecl/crates/cubecl-std" }
cubecl-zspace  = { path = "/home/tbindas/projects/cubecl/crates/cubecl-zspace" }
burn-cubecl    = { path = "/home/tbindas/projects/burn/crates/burn-cubecl" }
burn-autodiff  = { path = "/home/tbindas/projects/burn/crates/burn-autodiff" }
burn-backend   = { path = "/home/tbindas/projects/burn/crates/burn-backend" }
burn-core      = { path = "/home/tbindas/projects/burn/crates/burn-core" }
burn-cuda      = { path = "/home/tbindas/projects/burn/crates/burn-cuda" }
burn-derive    = { path = "/home/tbindas/projects/burn/crates/burn-derive" }
burn-ir        = { path = "/home/tbindas/projects/burn/crates/burn-ir" }
burn-ndarray   = { path = "/home/tbindas/projects/burn/crates/burn-ndarray" }
burn-nn        = { path = "/home/tbindas/projects/burn/crates/burn-nn" }
burn-optim     = { path = "/home/tbindas/projects/burn/crates/burn-optim" }
burn-std       = { path = "/home/tbindas/projects/burn/crates/burn-std" }
burn-tensor    = { path = "/home/tbindas/projects/burn/crates/burn-tensor" }
burn-fusion    = { path = "/home/tbindas/projects/burn/crates/burn-fusion" }
```

All `burn-*` crates must come from the same source to avoid duplicate
trait-object conflicts (the inner-backend `Device` type has to unify
across `burn-tensor`, `burn-cubecl`, and `burn-ndarray`). The cubecl
monorepo cascades the same way — patching only `cubecl-cuda` is not
enough, because `cubecl-cuda` depends on `cubecl-runtime` which depends
on `cubecl-core` and so on. The whole forest gets patched together.

If your clones live at paths other than the literal
`/home/tbindas/...` strings above, edit `Cargo.toml` in your local
checkout. There is no environment variable that overrides these — the
path is baked into the patch table.

## Gotchas

- **The fork's branch was renamed.** It used to be called
  `ddrs-sp7-stream-accessor`. If you cloned earlier, run `git fetch
  origin && git checkout ddrs-release` on the fork to pick up the
  SP-9 and SP-10 patches.
- **`fixtures/sandbox/` is gitignored.** A fresh `ddrs` clone has no
  fixtures, and `compare_ddr_sandbox` will panic at the first CSV read.
  Regenerate them with:

  ```bash
  cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py
  ```

  This must run under DDR's `uv` venv — the script imports
  `tests.benchmarks.conftest`, `ddr_engine.merit`, and `ddr.dmc`, none
  of which are on ddrs's side.
- **CUDA defaults are on.** `config/merit_training.yaml` ships with
  `sparse_solver: cuda` and `use_cuda_graphs: true`. On a CPU-only
  machine, either edit the YAML or pass a temp YAML that overrides:

  ```yaml
  params:
    sparse_solver: cpu
    use_cuda_graphs: false
  ```

  The `cargo test` suite already runs CPU-only, so `cargo test` is a
  fine first check that the build is sound even without CUDA.
- **`output/` must exist** for the examples. Both `compare_ddr_sandbox`
  and `benchmark_hydrograph` write CSV+PNG directly to `output/` and
  panic on `BufWriter::new(File::create(...))` if the directory is
  missing. Create it once at the repo root:

  ```bash
  mkdir -p output
  ```

  `train.rs` and `eval.rs` do call `create_dir_all` on
  `--checkpoint-dir` / `--output`, so they are forgiving — only the
  examples need this one-time `mkdir`.

## Verification

The V1 invariant from `CLAUDE.md` is the canonical "is this set up
correctly?" check:

```bash
cargo run --release --example compare_ddr_sandbox
```

Expected output:

```
verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

If that fails, check the troubleshooting order:

1. Does `cargo build --release` complete? If not, the `[patch.crates-io]`
   paths probably don't match your clone locations.
2. Does `fixtures/sandbox/` exist with six CSVs? If not, regenerate
   via the DDR `uv` venv.
3. Does `output/` exist? If not, `mkdir -p output`.
4. Is the build actually `--release`? Debug builds are ~20× slower and
   not a useful sanity check.

For broader confidence:

```bash
cargo test                                 # all tests (~54 across 7 files + lib units)
cargo test --test sparse_gradcheck         # one integration test file
cargo test --test mmc mc_routes_linear_chain   # one specific test
```

The CPU-side `cargo test` suite passes without CUDA. If you have a
CUDA box and want to verify the graph-capture path:

```bash
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox
```

This routes the example through the `Cuda<f32, i32>` inner backend
with `use_cuda_graphs=true` regardless of the YAML. Both runs must
report `ABSOLUTE MATCH` for a clean setup.

## See also

- [Running the code](usage/running.md) — train, evaluate, and run the
  examples once setup is green.
- [Comparing to DDR](reference/ddr-comparison.md) — what the V1
  regression measures and how to regenerate its fixtures.
- [Formatting inputs](usage/inputs-formatting.md) — the YAML keys this
  page references, and how to flip the CUDA defaults to CPU.
