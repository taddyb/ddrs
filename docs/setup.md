# Setup

This chapter walks through the external dependencies a fresh ddrs
checkout resolves against — the Rust toolchain, the `taddyb/cubecl` and
`taddyb/burn` forks pinned as git dependencies, the CUDA toolkit +
driver for the GPU path, and the DDR reference repository used to
regenerate the V1 sandbox fixture — plus the canonical data-source
paths referenced from `config/merit_training.yaml`. The expected
end-state is a working release build whose V1 regression reports
`ABSOLUTE MATCH`.

## What you need

ddrs's `Cargo.toml` resolves against a handful of things that are not
plain crates.io entries. Get them right once at setup; everything
downstream just works. The good news since the early bring-up: the
cubecl and burn forks are now pinned as **git dependencies**, so a
plain `git clone && cargo build` works — no sibling clones to wire up.

1. **Rust stable** — version 1.80 or later. ddrs has been built and
   tested on `1.94.0`; the language features in use (let-else, GATs in
   `Backend`, `impl Trait` in trait bounds) all stabilized well before
   1.80, so any recent stable will compile. Install via `rustup`:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup default stable
   ```

2. **The `taddyb/cubecl` fork on branch `ddrs-release` and the
   `taddyb/burn` fork on branch `ddrs-sp7-primitive-ctor`.** The
   `[patch.crates-io]` block in `Cargo.toml` redirects every
   `cubecl-*` crate to the cubecl fork and every `burn-*` crate to the
   burn fork, both as git dependencies — no local clones required,
   `git clone && cargo build` works as-is:

   ```toml
   [patch.crates-io]
   cubecl         = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
   cubecl-cuda    = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
   # ...one entry per cubecl-* crate, then every burn-* crate on the burn fork
   burn-cubecl    = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
   # ...one entry per burn-* crate
   ```

   The cubecl fork carries three ddrs-specific patches over upstream
   cubecl 0.10:

   - `stream_accessor` (SP-7) — exposes the thread-bound cubecl stream
     so ddrs can submit GPU work on the same stream the JIT compiles
     into.
   - `exclusive_with_server` (SP-9) — re-entrant safe acquisition of
     the cubecl server context, needed before binding a thread to the
     CUDA stream once at capture time.
   - `flush_no_sync` (SP-10) — flush variant that submits queued work
     without the `cuEventSynchronize` call that would invalidate a
     stream mid-capture.

   The burn fork (`ddrs-sp7-primitive-ctor`) carries the primitive
   constructor accessors ddrs needs to hand cuSPARSE a raw device
   pointer for the GPU triangular solve. See the full block in
   [The `[patch.crates-io]` block in detail](#the-patchcrates-io-block-in-detail)
   below.

3. **CUDA Toolkit 12+ with a CUDA-12-capable driver.** Validated on
   driver 575.57.08 (8× A100, sm_80) and a desktop RTX 4080. ddrs's
   current `config/merit_training.yaml` defaults to `sparse_solver:
   cuda` and `use_cuda_graphs: true`, so the GPU path is exercised on
   every default training run. The CUDA device ordinal is selectable
   via the top-level `device:` key in the config (default `0`); on a
   multi-GPU host, set e.g. `device: 1` to keep training off the
   display/shared GPU.

   > **A CUDA toolkit is required to *compile*, even for CPU-only use.**
   > `burn-cuda` and `cudarc` (the latter with the
   > `cuda-version-from-build-system` feature) are **non-optional**
   > dependencies in `Cargo.toml`, and `burn_cuda::Cuda` is imported
   > unconditionally by `src/sparse/dispatch.rs`, `src/sparse/cusparse.rs`,
   > `src/cli/run.rs`, and `src/cli/system.rs` (which also calls
   > `cudarc::driver` directly). There is no `cfg(feature = "cuda")`
   > gate around them. A machine with no CUDA toolkit therefore fails at
   > `cargo build`, *not* at runtime.
   >
   > CPU **execution** is fully supported once built: pass
   > `ddrs run --backend cpu` (NdArray, deterministic; `sparse_solver`
   > is forced to `cpu`), and the `cargo test` suite runs CPU-only.

4. **DDR reference repository** at `~/projects/ddr` for V1 fixture
   regeneration. `scripts/export_ddr_sandbox.py` runs under DDR's `uv`
   venv to produce the CSVs that `examples/compare_ddr_sandbox.rs`
   reads back to verify the port. NB: a valid V1 fixture currently
   requires the desktop's DDR working tree — see
   [Comparing to DDR](reference/ddr-comparison.md).

There is one more pinned dependency worth naming, though it is an
ordinary git dependency (not a `[patch.crates-io]` override) and needs
no extra setup: the KAN routing head crate.

- **`rskan`**, pinned to **tag `v0.1.3`** in `Cargo.toml`
  (`rskan = { git = "https://github.com/taddyb/rskan.git", tag = "v0.1.3" }`).
  This is the pure-Rust port of `pykan.KANLayer` that backs
  `src/nn/kan_head.rs`. When updating it, bump the tag and re-run the
  KAN parity sweep (CLAUDE.md invariant 7).

## Data file paths

The training config `config/merit_training.yaml` references the on-disk
data sources by absolute path. The adjacency stores are now **managed**:
the config points at the raw `geospatial_fabric` (MERIT flowlines) and
`ddrs plan` builds the CONUS + per-gauge zarr adjacency stores into
`.ddrs/adjacency/<key>/` on first run. From the `data_sources:` block:

```yaml
data_sources:
  attributes:        /home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc
  geospatial_fabric: /projects/mhpi/data/MERIT/raw/continent/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp
  streamflow:        /mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic
  observations:      /mnt/ssd1/data/icechunk/usgs_daily_observations
  gages:             /home/tbindas/projects/ddr/references/gage_info/gages_3000.csv
```

Mapping to the data layer:

| Source | Path | Reader |
|---|---|---|
| Geospatial fabric | `riv_pfaf_7_..._bugfix1.shp` (sibling `.dbf` only) or a `.gpkg` | `dbase` / `rusqlite` |
| MERIT adjacency | managed → `.ddrs/adjacency/<key>/` | `zarrs` (`src/data/store/zarr.rs::ConusAdjacencyStore`) |
| Per-gauge subgraphs | managed → `.ddrs/adjacency/<key>/` | `zarrs` (`GagesAdjacencyStore`) |
| Catchment attributes | `~/projects/ddr/data/merit_global_attributes_v2.nc` | `netcdf` (TODO per `CLAUDE.md`) |
| Streamflow forcing | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` | `icechunk` (TODO) |
| USGS observations | `/mnt/ssd1/data/icechunk/usgs_daily_observations` | `icechunk` (TODO) |
| Gauges list | `~/projects/ddr/references/gage_info/gages_3000.csv` | `csv` |

Only the fabric's **attribute table** is read — `.shp` geometry / gpkg
geometry blobs are never opened. For a merged global MERIT `.gpkg` with
more than one feature layer, set `geospatial_fabric_layer: <name>` (it
participates in the adjacency cache key). To skip the managed build
entirely, drop `geospatial_fabric` and set both `conus_adjacency` and
`gages_adjacency` to pre-built zarr stores instead.

If any of these live elsewhere on your machine, change the paths rather
than symlinking — symlinks under `~/projects/ddr/data/` have historically
masked stale fixtures. For a one-off path, edit the `data_sources:` block
in your `ddrs.yaml`; for a whole dataset you switch between, prefer the
saved groups: `ddrs sources list` / `ddrs sources save <name>` /
`ddrs sources use <name>` splice the block in for you and refresh
`.ddrs/sources.lock`, so no hand-editing is needed. See
[Reading inputs](usage/inputs-reading.md) and
[Graph objects](usage/graph-objects.md) for what each source contributes.

## Setup steps

The bring-up is a handful of commands. Run them in order; the last is
the sanity gate that proves the rest worked.

```bash
# 1. Clone ddrs. The cubecl/burn forks resolve as git dependencies,
#    so no sibling clones are needed.
git clone git@github.com:taddyb/ddrs ~/projects/ddrs

# 2. Clone the DDR reference (for V1 fixtures).
git clone git@github.com:mhpi/ddr ~/projects/ddr
cd ~/projects/ddr && uv sync --all-packages

# 3. Build.
cd ~/projects/ddrs
cargo build --release

# 4. Sanity check: V1 regression must report ABSOLUTE MATCH.
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

For day-to-day use, install the `ddrs` CLI and drive the
`plan → run` lifecycle instead of the raw examples:

```bash
cargo install --path .   # puts `ddrs` in ~/.cargo/bin/
ddrs plan                # GPU probe + cached smoke test, bootstraps ./ddrs.yaml,
                         # locks data sources, builds managed adjacency + baseline
ddrs run                 # execute the workflow, write manifest + outputs
```

`ddrs plan` is the first command — there is **no `ddrs init`**. The old
`init` subcommand was merged into `plan`; the remaining stub is hidden
from `--help`, prints `ddrs init has been merged into ddrs plan` and
exits **2** (`src/bin/ddrs.rs`, asserted by `tests/cli_init_stub.rs`),
so old muscle-memory scripts fail loudly rather than silently skipping
setup.

> **⚠️ STALE-BINARY TRAP — re-install after every `src/` change.**
> `ddrs` on your PATH is `~/.cargo/bin/ddrs`, **not** your working tree.
> `cargo build` and `cargo run` do **not** update it, so editing `src/`
> and then typing `ddrs run …` silently runs the *old* binary. The run
> manifest stamps `git.sha` from `.git` at runtime, not from the binary,
> so a stale run still *looks* current. After touching `src/`, do one of:
>
> ```bash
> cargo install --path .                              # canonical refresh
> cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
> cargo run --release --bin ddrs -- run --workflow …  # bypass the installed copy
> ```
>
> **Self-check:** current checkpoints are **directories**
> (`.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/` containing `head.mpk`,
> `optim.mpk`, `state.json`). Flat `epoch_E_mb_M.mpk` files mean a
> pre-checkpoint-resume binary ran — you are on a stale `ddrs`.

Two flags are global (`src/bin/ddrs.rs`), so they work on every
subcommand:

- `--config <path>` — the experiment config to use. Default: discover
  `ddrs.yaml` upward from the current directory, stopping at the first
  `.git` ancestor.
- `--workspace <dir>` — the workspace directory. **Default: `.ddrs/`
  *beside the config*.** This is a trap: `--config
  config/experiments/x.yaml` silently creates and uses
  `config/experiments/.ddrs/`, not the repo-root workspace, so cached
  adjacency/baselines are rebuilt and runs land somewhere unexpected.
  Always pass `--workspace` explicitly alongside `--config`:

  ```bash
  ddrs --config config/experiments/x.yaml \
       --workspace ~/projects/ddrs/.ddrs run --workflow train
  ```

Beyond `plan` and `run`, the CLI also has `ddrs sources`
(named data-source groups under `config/sources/`), `ddrs import`
(register an external Q' store — see
[the Q' store contract](nh-qprime-store-contract.md)), `ddrs show`
(inspect a past run's manifest), `ddrs status`, and `ddrs gc`. See
`README.md` and [Running the code](usage/running.md) for the full
lifecycle.

## The `[patch.crates-io]` block in detail

The full patch list from `Cargo.toml` redirects every cubecl crate to
the cubecl fork (branch `ddrs-release`) and every BURN crate to the
burn fork (branch `ddrs-sp7-primitive-ctor`) — all as git dependencies:

```toml
[patch.crates-io]
cubecl         = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-cuda    = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-common  = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-core    = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-cpp     = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-ir      = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-macros  = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-opt     = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-runtime = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-std     = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-zspace  = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
burn-cubecl    = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-autodiff  = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-backend   = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-core      = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-cuda      = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-derive    = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-ir        = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-ndarray   = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-nn        = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-optim     = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-std       = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-tensor    = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
burn-fusion    = { git = "https://github.com/taddyb/burn.git",   branch = "ddrs-sp7-primitive-ctor" }
```

All `burn-*` crates must come from the same source to avoid duplicate
trait-object conflicts (the inner-backend `Device` type has to unify
across `burn-tensor`, `burn-cubecl`, and `burn-ndarray`). The cubecl
monorepo cascades the same way — patching only `cubecl-cuda` is not
enough, because `cubecl-cuda` depends on `cubecl-runtime` which depends
on `cubecl-core` and so on. The whole forest gets patched together.

Because these are git pins rather than local `path =` entries, there is
nothing to edit per-machine — `cargo build` fetches the forks for you.
For local fork development, push your changes to the fork branch and run
`cargo update -p cubecl -p burn-core` (or whichever crate(s) you
changed) to pull the new commits.

## Gotchas

- **The cubecl fork branch was renamed.** It used to be called
  `ddrs-sp7-stream-accessor`. If you have an old checkout or a stale
  `Cargo.lock` pinning the old branch, `cargo update` against the
  current `Cargo.toml` picks up `ddrs-release` (and the SP-9/SP-10
  patches).
- **`fixtures/sandbox/` is gitignored.** A fresh `ddrs` clone has no
  fixtures, and `compare_ddr_sandbox` will panic at the first CSV read.
  Regenerate them with:

  ```bash
  cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py
  ```

  This must run under DDR's `uv` venv — the script imports DDR-side
  modules that are not on ddrs's side. See
  [Comparing to DDR](reference/ddr-comparison.md) for the fixture
  caveat about the desktop DDR working tree.
- **CUDA defaults are on.** `config/merit_training.yaml` ships with
  `sparse_solver: cuda` and `use_cuda_graphs: true`. To run on the CPU,
  do **not** hand-edit the YAML — `ddrs run` takes a `--backend` flag
  (`src/bin/ddrs.rs`, `default_value = "cuda"`):

  ```bash
  ddrs run --workflow train --backend cpu
  ```

  This binds the `NdArray<f32>` backend (deterministic) and forces
  `sparse_solver` to `cpu`; the run logs
  `backend: cpu (NdArray, deterministic; sparse_solver forced to cpu)`.
  Only `cpu` and `cuda` are accepted — anything else is rejected before
  the workspace is touched. If you do need it pinned in config instead,
  the equivalent override is:

  ```yaml
  params:
    sparse_solver: cpu
    use_cuda_graphs: false
  ```

  The `cargo test` suite already runs CPU-only, so `cargo test` is a
  fine first check that the build is sound. (It still needs the CUDA
  toolkit to *compile* — see item 3 above.)
- **`output/` must exist** for the examples. Both `compare_ddr_sandbox`
  and `benchmark_hydrograph` write CSV+PNG directly to `output/` and
  panic on `BufWriter::new(File::create(...))` if the directory is
  missing. Create it once at the repo root:

  ```bash
  mkdir -p output
  ```

  The CLI `ddrs run` path creates its own run directories under
  `.ddrs/runs/`, so this `mkdir` only matters for the raw examples.

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

1. Does `cargo build --release` complete? If not, the cubecl/burn git
   forks probably failed to fetch — check network access to
   `github.com/taddyb`.
2. Does `fixtures/sandbox/` exist with its CSVs? If not, regenerate
   via the DDR `uv` venv.
3. Does `output/` exist? If not, `mkdir -p output`.
4. Is the build actually `--release`? Debug builds are ~20× slower and
   not a useful sanity check.

For broader confidence:

```bash
cargo test                                # all tests (each tests/ file is its own crate)
cargo test --test sparse_gradcheck        # one integration test file
cargo test --test mmc forward_reproducible  # one specific test
```

The CPU-side `cargo test` suite passes without CUDA. If you have a
CUDA box and want to verify the graph-capture path:

```bash
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox
```

This routes the example through the `Cuda<f32, i32>` inner backend
with `use_cuda_graphs=true` regardless of the YAML. Both runs must
report `ABSOLUTE MATCH` for a clean setup.

### Cargo features

`Cargo.toml` defines exactly two features, both off by default:

- **`fixtures`** — enables the optional `ndarray-npy` dependency and
  un-gates the fixture-backed tests, which are compiled out via
  `#![cfg(feature = "fixtures")]` otherwise. These read the tracked
  `tests/fixtures/` directory (e.g. `kan_head_init_seed42.npz`,
  `kan_init_stats_ddr.csv`), so nothing needs regenerating — unlike the
  root-level `fixtures/` used by `compare_ddr_sandbox`, `tests/fixtures/`
  is checked into git.
- **`cuda`** — enables the GPU branch of those fixture tests. (It does
  *not* control whether CUDA is linked; that is unconditional — see
  item 3 under [What you need](#what-you-need).)

If you touched `src/nn/`, the rskan pin in `Cargo.toml`, or DDR's
`nn/kan.py`, run the KAN parity sweep (CLAUDE.md invariant 7):

```bash
cargo test --features fixtures \
  --test kan_head_init_repro --test kan_head_init_parity \
  --test kan_head_fixture_forward --test kan_head_fixture_backward
```

## See also

- [Running the code](usage/running.md) — the `plan → run`
  lifecycle, training, evaluation, and the examples once setup is green.
- [Comparing to DDR](reference/ddr-comparison.md) — what the V1
  regression measures and how to regenerate its fixtures.
- [Reading inputs](usage/inputs-reading.md) and
  [Formatting inputs](usage/inputs-formatting.md) — the data sources and
  YAML keys this page references, and how to flip the CUDA defaults to
  CPU.
- [Graph objects](usage/graph-objects.md) — how the managed adjacency
  stores are built from the geospatial fabric.
- [BURN autograd](reference/burn-autograd.md) — why the cubecl/burn
  forks are needed for the cuSPARSE GPU solve.
