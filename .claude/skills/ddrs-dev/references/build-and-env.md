# Build and environment

Verified against `Cargo.toml` and source on 2026-07-30.

## Hard prerequisites

- **cmake.** `netcdf = { version = "0.12", features = ["static"] }` compiles bundled
  HDF5 + netcdf-c. Symptom without it: `error: failed to execute process 'cmake'`.
  The static build is deliberate — HPC module-provided netcdf/hdf5 are
  Intel-MPI-flavored shared libs that break `ddrs` outside `module load`.
- **A CUDA toolkit, even for CPU-only work.** `burn-cuda` and `cudarc`
  (`cuda-version-from-build-system`) are **non-optional** dependencies, and
  `use burn_cuda::Cuda;` is unconditional in `src/sparse/dispatch.rs`,
  `src/routing/mmc.rs`, `src/cli/run.rs`, `src/training/driver.rs`,
  `src/cli/system.rs`. CPU *execution* is supported via `--backend cpu`; CPU-only
  *compilation* is not. (`docs/setup.md` claims otherwise — it is wrong.)

If a system HDF5 leaks into the static build: `unset HDF5_DIR; unset NETCDF_DIR`.

## Fork pins

All **13** `burn-*` crates must resolve from `github.com/taddyb/burn` branch
`ddrs-sp7-primitive-ctor`; all **11** `cubecl*` crates from
`github.com/taddyb/cubecl` branch `ddrs-release`. `rskan` is pinned to tag
`v0.1.3`.

The patch exists for exactly two accessors upstream keeps `pub(crate)`:
`cubecl-cuda`'s `CudaServer::stream() -> CUstream` and `burn-cubecl`'s
`CubeTensor::from_handle(...)`. The plan is to upstream them as SP-8 and drop
`[patch.crates-io]` entirely.

| Symptom | Cause | Fix |
|---|---|---|
| `error[E0277]: the trait Device is not implemented` | Split resolution — some burn crates from crates.io, some from the fork | `cargo tree -p burn-std` and reconcile |
| `error: failed to resolve patches` | Fork branch renamed or deleted | Check the branch exists on the fork |

Iterating on a fork: push to the fork branch, then `cargo update -p <crate>`.
**Never commit `path = "..."` overrides** — they break the public build.

## Fixtures

Gitignored (`/fixtures/`) and not auto-created. Two families:

**V1 sandbox** — `fixtures/sandbox/`, read by `examples/compare_ddr_sandbox.rs`.
Regenerate with:
```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py
```
> **Any DDR checkout at or past DeepGroundwater/ddr#192 is a valid reference**
> (2026-08-19: fixture regenerated from post-#192 master; comparison matches at
> the f32 floor on the corrected physics, which is now ddrs's default —
> `ddr_match` is deprecated). The old rule — only the desktop's working tree
> with unpushed `geometry/trapezoidal.py` work — applied before #192 landed;
> a PRE-#192 clean clone still diverges ~1% (max abs ≈ 0.55 m³/s). Details:
> `docs/reference/ddr-comparison.md` §Regenerating fixtures.

**KAN parity** — `tests/fixtures/` (tracked, unlike `/fixtures/`), loaded behind
`#[cfg(feature = "fixtures")]`. Regenerate under DDR's venv with
`scripts/dump_kan_fixture.py` (and `scripts/dump_kan_init_stats.py`).
There is no `dump_kan_head.py` / `dump_kan_weights.py` / `dump_kan_forward.py` —
three retired skills named files that do not exist.

## Gitignored artifacts

`output/`, `/fixtures/`, `examples/fixtures/`, `.ddrs/`, `.ddrs-synthetic-n-*/`,
`ddrs.yaml`. Anything under `output/` is machine-local and not reproducible from a
clone — state that when citing data living there.

## Cargo features

`fixtures` (KAN parity tests) and `cuda`. `--features fixtures` is required for the
Tier B sweep.

## Worktrees

Worktrees live in `.claude/worktrees/<name>/` (gitignored). Two gotchas:
- A relative `target/release/...` path resolves against the **main tree**, so you
  can silently run a stale main-tree binary from inside a worktree. Use absolute
  paths or `cargo run`.
- Fresh worktrees lack gitignored `fixtures/` and `output/`. Copy `fixtures/` from
  the main tree (watch for accidental nesting into `fixtures/fixtures/`) and
  `mkdir -p output`.
