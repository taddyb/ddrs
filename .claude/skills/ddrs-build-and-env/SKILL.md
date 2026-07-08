---
name: ddrs-build-and-env
description: Use when setting up a fresh ddrs checkout, diagnosing build failures, hitting the forked-dependency trap, missing fixture errors, static netcdf/HDF5 cmake issues, stale-binary symptoms, or CUDA graphs masking NaN. Also use when cargo build succeeds but runtime behaves as an old version, or when fixture regeneration is needed after DDR solver changes.
---

# ddrs Build and Environment Runbook

## Glossary (read once; terms used throughout)

| Term | Meaning |
|---|---|
| **BURN** | Rust deep-learning framework (like PyTorch but for Rust). Version 0.21 in ddrs. |
| **DDR** | The Python/PyTorch reference implementation at `~/projects/ddr`. ddrs is its Rust port. |
| **V1 gate** | The regression test that must always pass: `compare_ddr_sandbox` reports ABSOLUTE MATCH (max abs diff < 1e-3 m³/s). |
| **`[patch.crates-io]`** | Cargo mechanism to globally replace a dependency's source. ddrs uses this to swap crates.io `cubecl`/`burn` for forked GitHub branches. |
| **KAN head** | Kolmogorov-Arnold Network routing head (`rskan::KanLayer`). Replaces MLP. Must stay at tag v0.1.3. |
| **f32 invariant** | All routing-core tensors stay float32. No f64, bf16, or mixed precision. |
| **uv** | Python package manager (like pip + venv). DDR's venv is managed by uv; needed for fixture regeneration only. |
| **icechunk** | Transactional Zarr-over-filesystem store for streamflow + observations data. |
| **cuSPARSE** | NVIDIA sparse linear algebra library. Used for the GPU triangular solve in `src/sparse.rs`. |

---

## When NOT to use this skill

| If you need... | Use instead |
|---|---|
| MC routing algorithm math | `.claude/references/ddrs-algorithm.md` |
| Autograd / sparse backward internals | `.claude/references/ddrs-burn-autograd.md` |
| DDR parity / V1 failure debugging | `.claude/references/ddrs-comparing-to-ddr.md` |
| Training a run from scratch | `CLAUDE.md` §"ddrs CLI" or README §"Getting started" |
| Leakance experiment status | `docs/2026-07-01-leakance-hourly-findings.md` |
| Architecture diagram | `.claude/ARCHITECTURE.md` |

---

## 1. Prerequisites Checklist

Before `cargo build` can succeed, verify every item below.

### 1a. Rust toolchain

```bash
rustc --version   # must be >= 1.80; tested on 1.94.0 as of 2026-07-05
cargo --version
```

Install or update via rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### 1b. cmake (required for static netcdf/HDF5)

`Cargo.toml` declares `netcdf = { version = "0.12", features = ["static"] }`. The `static` feature compiles bundled HDF5 and netcdf-c from source during `cargo build`. This avoids depending on system-installed dev packages (HPC hosts often have only Intel-MPI-flavored shared libs that would break `ddrs` outside `module load`).

**Consequence:** cmake must be available on your PATH before building.

```bash
cmake --version   # must exist; any reasonably recent version works
```

If missing:
```bash
# Debian/Ubuntu
sudo apt-get install cmake
# Arch
sudo pacman -S cmake
# HPC (check module system)
module load cmake
```

### 1c. CUDA Toolkit

Required only for the GPU path (`sparse_solver: cuda`, `use_cuda_graphs: true`). The CPU/NdArray backend path builds and runs without CUDA.

```bash
nvcc --version   # want CUDA 12+ (13.2 validated as of 2026-07-05)
nvidia-smi       # driver must support the CUDA version
```

Validated configurations:
- RTX 4080 SUPER, driver 610.43.02, CUDA 13.2 (desktop, as of 2026-07-05)
- 8× A100, driver 575.57.08, CUDA 12, sm_80 (HPC)

For CPU-only machines: skip CUDA setup. Override the default GPU config at runtime (see §5 "CPU-only override").

### 1d. git (for fork resolution)

The forked cubecl and burn are fetched via HTTPS git by Cargo automatically. No local clones of cubecl or burn are needed — `git clone ddrs && cargo build` is sufficient.

```bash
git --version    # any recent version
```

---

## 2. Build Steps

```bash
# Clone ddrs
git clone git@github.com:taddyb/ddrs ~/projects/ddrs
cd ~/projects/ddrs

# Release build (LTO=thin; matches what ddrs CLI uses)
cargo build --release

# Sanity check: V1 gate
mkdir -p output
cargo run --release --example compare_ddr_sandbox
# Expected last line: "verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)"
```

The first build fetches:
- `github.com/taddyb/cubecl` branch `ddrs-release` (all cubecl-* crates)
- `github.com/taddyb/burn` branch `ddrs-sp7-primitive-ctor` (all burn-* crates)
- `github.com/taddyb/rskan` tag `v0.1.3`

This takes several minutes on first run; subsequent builds use the Cargo registry cache.

---

## 3. The Forked-Dependency Trap

### What is patched and why

`Cargo.toml` contains a `[patch.crates-io]` block that replaces the published crates.io versions of cubecl and burn with fork branches on github.com/taddyb:

```
[patch.crates-io]
cubecl         = { git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }
cubecl-cuda    = { git = "..." }   # ... and 8 more cubecl-* crates
burn-cubecl    = { git = "https://github.com/taddyb/burn.git", branch = "ddrs-sp7-primitive-ctor" }
burn-autodiff  = { git = "..." }   # ... and 12 more burn-* crates
```

The patches add exactly two `pub` accessors needed by ddrs's cuSPARSE GPU solve (SP-7):

| Crate | Added accessor |
|---|---|
| `cubecl-cuda` 0.10 | `pub fn CudaServer::stream() -> CUstream` |
| `burn-cubecl` 0.21 | `pub fn CubeTensor::from_handle(...) -> Self` |

These were `pub(crate)` in the upstream releases. The plan is to upstream them as SP-8 and remove `[patch.crates-io]` once merged.

### Rules when working with forks

1. **All burn-* crates must come from the same fork branch.** If Cargo resolves any burn-* crate from crates.io while another resolves from the fork, you get duplicate `Device` trait objects and cryptic link errors. The `[patch.crates-io]` block covers all 13 burn sub-crates; do not add a direct dependency on a crates.io burn sub-crate that would escape the patch.

2. **Same rule for cubecl-***: all 10 crates in the monorepo must come from `ddrs-release`.

3. **To iterate on the fork locally**: push your changes to the fork branch, then pull into ddrs with `cargo update -p cubecl` (or the changed crate). Do NOT commit `path = "..."` overrides — they break the public build.

4. **rskan is pinned to a tag**, not a branch: `rskan = { git = "https://github.com/taddyb/rskan.git", tag = "v0.1.3" }`. Bumping the tag requires re-running the KAN parity sweep (CLAUDE.md invariant 6-7).

### Diagnosing fork resolution failures

```
error[E0277]: the trait `Device` is not implemented for ...
```
→ burn crate split across crates.io and fork. Run `cargo tree -p burn-std` to find which sub-crate is from crates.io. Add it to `[patch.crates-io]`.

```
error: failed to resolve patches for ...
```
→ The fork branch was renamed or deleted. Check `github.com/taddyb/cubecl` or `github.com/taddyb/burn` for current branch name.

---

## 4. Static netcdf/HDF5 Build Details

`netcdf = { version = "0.12", features = ["static"] }` causes Cargo's build script to:
1. Download HDF5 and netcdf-c sources.
2. Compile them via cmake during `cargo build`.
3. Link them statically into the final binary.

This is intentional: the HPC hosts lack usable `libnetcdf-dev`/`libhdf5-dev` packages (the module-provided ones are Intel-MPI-flavored and break `ddrs` outside `module load`).

**Build time impact**: first build takes several extra minutes for the cmake compile. Subsequent builds are cached by Cargo.

**If cmake is not found**:
```
error: failed to execute process `cmake`: No such file or directory
```
Install cmake (see §1b) and retry `cargo build`.

**If cmake finds system HDF5 but produces link errors**: the `static` feature should bypass system HDF5. Ensure you have not set `HDF5_DIR` in your environment pointing at an incompatible installation:
```bash
unset HDF5_DIR
unset NETCDF_DIR
cargo build --release
```

---

## 5. CPU-only Override

On a machine without CUDA, the default config (`config/merit_training.yaml`) fails because it requests `sparse_solver: cuda`. Override it via a minimal config file:

```yaml
# cpu_override.yaml  (do not commit)
sparse_solver: cpu
use_cuda_graphs: false
```

```bash
ddrs --config cpu_override.yaml plan
ddrs --config cpu_override.yaml run --workflow train
# or for just the V1 gate (V1 always defaults to NdArray/CPU):
cargo run --release --example compare_ddr_sandbox   # no override needed
```

The CPU NdArray backend is the default for `compare_ddr_sandbox`; V1 always passes on CPU. Only training-scale runs need the GPU config override.

---

## 6. Gitignored Fixtures and Outputs

Three categories of files are gitignored and must be recreated after a fresh clone:

### 6a. Sandbox fixtures (`/fixtures/`, `examples/fixtures/`)

These are the V1 gate inputs generated by DDR's Python solver. They are gitignored because they are derived artifacts. `tests/fixtures/` IS tracked; only root-level and examples-level fixtures are excluded.

```bash
# Regenerate after DDR's solver changes, or on a fresh clone
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py
```

CRITICAL CAVEAT (as of 2026-07-05): The valid V1 fixture lives ONLY in the desktop's `~/projects/ddr` working tree. That tree contains unpushed work — `src/ddr/geometry/trapezoidal.py` — that does not exist in any DDR public commit. Regenerating from a clean DDR clone produces a ~1%-divergent reference (max abs ≈ 0.55 m³/s) that would make V1 fail at every ddrs commit. This is a wrong-reference artifact, not a port bug. Until DDR's geometry work is pushed, only the desktop DDR tree produces a valid V1 fixture.

### 6b. Output directory (`output/`)

The `compare_ddr_sandbox` example writes to `output/ddrs_vs_ddr.{csv,png}` using `File::create`, which does NOT `mkdir -p`. A fresh clone will panic on the file create.

```bash
mkdir -p output
cargo run --release --example compare_ddr_sandbox
```

### 6c. Workspace artifacts (`.ddrs/`)

The entire `.ddrs/` directory is gitignored. This includes adjacency caches, baseline caches, run manifests, checkpoints, and run logs. They are rebuilt by `ddrs plan` on first run.

---

## 7. Stale-Binary Trap

This is one of the most common sources of silent wrong behavior.

**The problem**: `ddrs` on your PATH is `~/.cargo/bin/ddrs`. `cargo build` and `cargo run` compile into `target/release/ddrs` but do NOT copy it to `~/.cargo/bin/`. If you edit `src/` and then type `ddrs run`, you silently execute the old binary.

The manifest's `git.sha` field is stamped from `.git` at runtime, NOT from the binary. A run can look like current code in the manifest while a weeks-old binary actually executed. This caused the 2026-07-01 leakance×hourly 2×2 to produce byte-identical hourly cells — the installed binary predated the disaggregation feature.

**Self-check**: current checkpoints are DIRECTORIES (`epoch_E_mb_M/head.mpk` etc.). If you see flat `.mpk` files at `epoch_E_mb_M.mpk`, you ran a pre-checkpoint-resume binary.

**Fix after every `src/` change**:

```bash
# Canonical (always correct):
cargo install --path .

# Faster when target/release is already built:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs

# Bypass the installed binary entirely (safest during development):
cargo run --release --bin ddrs -- run --workflow train
```

---

## 8. CUDA Graphs Masking NaN

**The trap**: `use_cuda_graphs: true` records a graph on the first forward pass and replays it on subsequent steps. If the first forward passes with a finite loss but a later one would produce NaN (e.g., from a bad parameter initialization or data batch), the graph replay returns the stale finite loss from the captured pass. You see a training run that appears to converge smoothly but is actually computing nothing.

**Affected config key**: `use_cuda_graphs: true` in `ddrs.yaml` or `config/merit_training.yaml`.

**Config rejection rule**: `use_leakance: true` + `use_cuda_graphs: true` is REJECTED at config load time (no exception path). These cannot be used together; the leakance kernel has no separate CUDA graph capture path.

**Diagnosis**:

```bash
# 1. Disable graphs and rerun the suspicious config
#    Add to ddrs.yaml:
#      use_cuda_graphs: false

# 2. Watch for NaN in the loss log — if it now appears, graphs were masking it

# 3. Find the NaN source: typically a parameter initialized to zero or
#    a data batch with all-NaN streamflow
```

**Rule**: always validate new configs and new parameter initializations with `use_cuda_graphs: false` first. Only re-enable graphs after confirming the forward is NaN-free.

---

## 9. Critical Invariants (Do Not Break)

Breaking any of these makes the ddrs port meaningless or incorrect.

| # | Invariant | Test |
|---|---|---|
| 1 | `compare_ddr_sandbox` must report ABSOLUTE MATCH (max abs < 1e-3 m³/s) | `cargo run --release --example compare_ddr_sandbox` |
| 2 | f32 throughout routing core; no f64/bf16 casts in `src/routing/`, `src/geometry.rs`, `src/sparse.rs` | `grep -rn 'f64\|bf16\|cast\|to_dtype' src/routing/ src/geometry.rs src/sparse.rs` |
| 3 | Adjacency is topologically ordered, lower-triangular (`rows[k] >= cols[k]`) | `cargo test data_zarr_store::conus_adjacency_loads_real_merit_zarr` |
| 4 | Hand-written sparse backward in `src/sparse.rs` must NOT be replaced with tape unrolling | `cargo test --test sparse_gradcheck` |
| 5 | KAN head = `rskan::KanLayer` via `src/nn/kan_head.rs`; no MLP placeholder; no inter-block ReLU | `cargo test --test kan_head` |
| 6 | rskan pinned to tag `v0.1.3` in Cargo.toml | `grep 'rskan.*tag' Cargo.toml` |
| 7 | KAN parity vs DDR must pass on every PR touching `src/nn/`, rskan pin, or DDR's `nn/kan.py` | See §11 KAN parity command |

---

## 10. Full Verification Command Set

Run these in order after a fresh build or after touching `src/`:

```bash
# V1 gate — must report ABSOLUTE MATCH
mkdir -p output
cargo run --release --example compare_ddr_sandbox

# V1 gate on CUDA + graph-capture path (if GPU available)
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox

# Sparse gradient correctness
cargo test --test sparse_gradcheck

# Routing correctness (linear chain)
cargo test --test mmc mc_routes_linear_chain

# Leakance gradient-exactness (if leakance was touched)
cargo test --test leakance_gradcheck
cargo test --test leakance_off_parity
cargo test --test zeta_accum
```

---

## 11. KAN Head Parity (after touching `src/nn/`, rskan pin, or DDR's `nn/kan.py`)

```bash
cargo test --features fixtures \
  --test kan_head_init_repro \
  --test kan_head_init_parity \
  --test kan_head_fixture_forward \
  --test kan_head_fixture_backward
```

If DDR's `nn/kan.py` changed: regenerate fixtures first, then re-validate:
```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_kan_fixture.py
cd ~/projects/ddrs && cargo test --features fixtures --test kan_head_fixture_forward --test kan_head_fixture_backward
```

---

## 12. DDR Reference Clone (for fixture regeneration)

```bash
git clone git@github.com:mhpi/ddr ~/projects/ddr
cd ~/projects/ddr && uv sync --all-packages
```

uv creates a `.venv` automatically. Fixtures are generated by running scripts under this venv:

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py
```

The uv venv must remain at `~/projects/ddr/.venv` — ddrs scripts import from DDR's Python packages.

---

## 13. Data File Paths

`config/merit_training.yaml`'s `data_sources:` block references these paths. Edit the YAML to match your machine if they live elsewhere.

| Source | Default path |
|---|---|
| Geospatial fabric | `riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp` (+ sibling `.dbf`), or a `.gpkg` |
| MERIT adjacency | Managed — built by `ddrs plan` into `.ddrs/adjacency/<key>/` |
| Per-gauge subgraphs | Managed — same directory |
| Catchment attributes | `~/projects/ddr/data/merit_global_attributes_v2.nc` |
| Streamflow forcing | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` |
| USGS observations | `/mnt/ssd1/data/icechunk/usgs_daily_observations` |
| Gauges list | `~/projects/ddr/references/gage_info/gages_3000.csv` |

To skip the managed adjacency build (e.g., you have pre-built zarr stores):
```yaml
# in ddrs.yaml, replace geospatial_fabric with:
conus_adjacency: /path/to/merit_conus_adjacency.zarr
gages_adjacency: /path/to/merit_gages_conus_adjacency.zarr
```

---

## 14. Common Failure Modes at a Glance

| Symptom | Root cause | Fix |
|---|---|---|
| `cmake: No such file or directory` during `cargo build` | cmake not on PATH; needed for static netcdf | Install cmake (§1b) |
| `the trait Device is not implemented` link error | burn crate split across crates.io and fork | Check `cargo tree -p burn-std`; add missing crate to `[patch.crates-io]` |
| `failed to resolve patches` | Fork branch renamed or deleted | Check `github.com/taddyb/{cubecl,burn}` for current branch |
| `thread 'main' panicked at 'No such file or directory' (output/...)` | `output/` missing on fresh clone | `mkdir -p output` |
| V1 fails with max abs ≈ 0.55 m³/s | Fixtures regenerated from wrong DDR clone (no trapezoidal.py) | Use desktop's `~/projects/ddr` working tree |
| Training appears to converge but loss never moves | CUDA graphs masking NaN | Set `use_cuda_graphs: false` and check for NaN (§8) |
| `ddrs run` uses wrong/old feature after `cargo build` | Stale installed binary at `~/.cargo/bin/ddrs` | `cargo install --path .` (§7) |
| Checkpoint is a flat `.mpk` file, not a directory | Stale pre-checkpoint-resume binary executed | `cargo install --path .` and re-run |
| `use_leakance + use_cuda_graphs rejected at config load` | These two are mutually exclusive by design | Remove `use_cuda_graphs: true` from leakance configs |
| `fixtures/sandbox/` missing, V1 panics on CSV read | gitignored artifact not regenerated | `cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py` |

---

## Provenance and maintenance

Skill written 2026-07-05 from `Cargo.toml`, `CLAUDE.md`, `README.md`, `.claude/references/ddrs-setup.md`, `.claude/references/ddrs-comparing-to-ddr.md`, and `vendor/README.md`.

Re-verification commands:
```bash
# Confirm fork branches still exist
git ls-remote https://github.com/taddyb/cubecl.git ddrs-release
git ls-remote https://github.com/taddyb/burn.git ddrs-sp7-primitive-ctor
git ls-remote https://github.com/taddyb/rskan.git refs/tags/v0.1.3

# Confirm static netcdf feature still declared
grep 'netcdf.*static' /home/tbindas/projects/ddrs/Cargo.toml

# Confirm gitignore entries
grep -E 'fixtures|output|\.ddrs' /home/tbindas/projects/ddrs/.gitignore

# Confirm Rust version meets minimum
rustc --version   # must be >= 1.80

# V1 gate
mkdir -p /home/tbindas/projects/ddrs/output
cargo run --release --example compare_ddr_sandbox 2>&1 | grep -E 'verdict|ABSOLUTE|FAIL'
```
