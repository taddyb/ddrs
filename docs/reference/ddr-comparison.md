# Comparing to DDR

ddrs is a gradient-exact Rust/BURN port of the differentiable
Muskingum-Cunge routing solver from `~/projects/ddr` (Python/PyTorch).
The whole point of the port is to reproduce DDR's forward and backward
numerics bit-for-bit at the f32 precision floor; everything else (CUDA
backends, sparse solvers, training loops) is downstream of that
contract.

The `compare_ddr_sandbox` example — "V1" — is the regression that pins
this. Any drift means the port has broken its contract, and the
offending change must be reverted or reconciled before shipping. There
is no acceptable narrative where V1 is "close enough but not matching":
either it reports `ABSOLUTE MATCH` or the port is broken. This is the
first invariant in `CLAUDE.md`, and it is the canonical "is this set up
correctly?" gate for a fresh checkout.

## What it is

V1 replays DDR's `tests/benchmarks/test_ddr.py` sandbox routing through
ddrs's `MuskingumCunge` solver on byte-identical inputs, then compares
the routed discharge reach-by-reach. The sandbox is deliberately tiny so
that any numerical divergence is unambiguous:

- **5 reaches**, reported in RAPID2 order `[10, 20, 30, 40, 50]`.
- Lateral inflow read from `qprime_topo.csv`, shape `(T, N)`, in
  topological order. The fixture currently carries **238 timesteps**
  (DDR interpolates the 3-hourly `Qext` sandbox forcing to hourly).
- Identical dense adjacency (lower-triangular), channel geometry
  (`length = 5000 m`, `slope = 0.001`, `x_storage = 0.25`), and spatial
  parameters (`n = 0.5`, `q_spatial = 0.5`, `p_spatial` left at the
  `21.0` default — `SpatialParameters.p_spatial: None` in
  `examples/compare_ddr_sandbox.rs`).

The example loads the fixtures, runs the forward solve under an
`Autodiff` backend, reorders the topological output into RAPID2 order to
line up with DDR's dump, and computes per-reach metrics: `max_abs_diff`,
`mean_abs_diff`, `max_rel_diff`, `ddr_mean`, `ddrs_mean`, and the
Pearson correlation `corr`.

### The threshold and the verdict

The pass/fail decision is a single comparison in
`examples/compare_ddr_sandbox.rs`:

```rust
let absolute_match = overall_max_abs < 1e-3;
```

`overall_max_abs` is the maximum absolute discharge difference across
every reach and timestep, in m³/s. A passing run prints

```
verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

The example has two fallback verdicts below the bar — `close match
(max rel < 1%)` when `overall_max_rel < 1e-2`, and `DIVERGENCE —
investigate` otherwise — but for the port these are failure states, not
acceptable outcomes. Only `ABSOLUTE MATCH` clears V1.

Typical passing result on CPU (the default NdArray inner backend) is
`max abs ≈ 1.5e-5 m³/s` — roughly two orders of magnitude under the
threshold. That margin is the f32 precision floor; closing it further is
not a goal.

### Outputs

The example writes two artifacts (and `output/` must already exist —
see Gotchas):

- `output/ddrs_vs_ddr.csv` — one row per reach with the header
  `reach_id,max_abs_diff,mean_abs_diff,max_rel_diff,ddr_mean,ddrs_mean,corr`.
  This is the first diagnostic when V1 fails.
- `output/ddrs_vs_ddr.png` — DDR (solid) and ddrs (dashed) hydrographs
  overlaid, one panel per reach.

## How to use it

Run the regression from the repo root:

```bash
mkdir -p output
cargo run --release --example compare_ddr_sandbox
```

`--release` is mandatory. Debug builds are far slower and exercise
different fused kernels in some BURN ops; the regression target assumes
release.

By default the example runs on the CPU NdArray inner backend. To verify
the CUDA graph-capture path — the path production training actually uses
— set `DDRS_FORCE_GRAPHS=1`. The example then dispatches the same
forward solve through the `Cuda<f32, i32>` inner backend instead of
NdArray:

```bash
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox
# expect: same ABSOLUTE MATCH verdict on the CUDA path
```

Both runs must report `ABSOLUTE MATCH` for a clean V1.

### Regenerating fixtures

The fixture directory `fixtures/sandbox/` is gitignored — the fixtures
are treated as derived artifacts, not source — so a fresh clone has none
and the example will panic at the first CSV read. Regenerate them from
the DDR reference whenever DDR's solver changes upstream, or after a
fresh checkout:

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py
```

This **must** run under DDR's `uv` venv: `scripts/export_ddr_sandbox.py`
imports `tests.benchmarks.conftest`, `ddr_engine.merit`, and `ddr.dmc`,
none of which live on ddrs's side. The script rebuilds the sandbox
adjacency zarr in a temp dir, interpolates the `Qext` forcing to hourly,
runs DDR's `dmc` model on CPU, and writes six files into
`fixtures/sandbox/`:

| File | Shape | Meaning |
|---|---|---|
| `topo_order.csv` | `(5,)` | Reach IDs in topological order |
| `rapid2_order.csv` | `(5,)` | Reach IDs in RAPID2 order `[10,20,30,40,50]` |
| `qprime_topo.csv` | `(T, N)` | Lateral inflow, topological order |
| `adjacency_topo.csv` | `(N, N)` | Dense adjacency, lower-triangular |
| `ddr_discharge_rapid2.csv` | `(N, T)` | DDR's routed output, RAPID2 order |
| `config.csv` | key=value | Length, slope, x_storage, params snapshot |

If you regenerate fixtures, V1 must still pass. If it suddenly fails
after regeneration, DDR's solver moved — investigate the DDR diff before
touching ddrs.

### The desktop-only DDR reference (2026-06-06)

> **The reference DDR state is NOT a pushed commit (as of 2026-06-06).**
> The port mirrors the desktop checkout of `~/projects/ddr`, which
> contains unpushed work — notably `src/ddr/geometry/trapezoidal.py`
> (the source of ddrs's `src/geometry.rs`), which exists at *no* commit
> in DDR's public history. DDR-at-HEAD instead *learns*
> `top_width`/`side_slope` (sandbox: denorm(0.5) log-space), while ddrs
> derives them Leopold-Maddock-style (`top_width = p·depth^q`).
> Regenerating the fixture from a clean DDR clone therefore produces a
> ~1%-divergent reference (max abs ≈ 0.55 m³/s, NSE ≈ 0.9996) at *every*
> ddrs commit — that is a wrong-reference artifact, not a port bug.
> Until the DDR-side geometry work is pushed, only the desktop's DDR tree
> generates a valid V1 fixture. See
> `docs/superpowers/plans/2026-06-06-sigfpe-wukong-debug-handoff.md`
> §Outcome for the full investigation.

In short: if a freshly cloned DDR makes V1 drift by ~1%, suspect the
reference before the port. A genuine port regression shows up as a
larger or structurally different divergence, and the desktop fixture
still passes.

### When V1 fails

The example's own `output/ddrs_vs_ddr.csv` is the first diagnostic. Walk
it in this order:

1. **Inspect `output/ddrs_vs_ddr.csv`** for the worst-offending
   reach(es). Per-reach `max_abs_diff`, `max_rel_diff`, and `corr`
   isolate whether the divergence is a single bad reach (often a
   geometry / parameter bug) or global (often a solver / kernel issue).
2. **Confirm fixtures aren't stale.** Re-run
   `scripts/export_ddr_sandbox.py` under DDR's venv and
   `git diff fixtures/sandbox/`. If DDR's outputs changed, the
   regression target moved — not a ddrs bug. (See the desktop-reference
   caveat above before concluding anything.)
3. **Audit recent commits to `src/routing/`, `src/geometry.rs`,
   `src/sparse/`, and `src/cuda_graph/`.** These are the only paths that
   affect V1. Anything else (data loaders, training loop, CLI) cannot
   have broken it. Use `git log -p` on those directories since the last
   known-good SHA.
4. **Check for accidental precision shifts.** Grep the routing path for
   `f64`, `bf16`, `cast`, or `to_dtype` — any silent widening or
   narrowing of the tensor dtype breaks reproducibility against the
   reference.
5. **Cross-check with `tests/sparse_gradcheck.rs`.** If the gradcheck
   also fails, the algorithm itself changed (numerator / denominator
   math, sparse solve, etc.). If only V1 fails, it's almost always a
   kernel-ordering or arithmetic-fusion difference at the f32 precision
   floor — look for new `fuse`/`einsum`/reduction-order changes in BURN
   ops or in your own tensor pipeline.

## Reference

**Run V1 (CPU, default):**

```bash
mkdir -p output
cargo run --release --example compare_ddr_sandbox
# expect: verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

**Run V1 (CUDA graph-capture path):**

```bash
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox
# expect: same verdict on the Cuda<f32, i32> inner backend
```

**Regenerate fixtures (DDR `uv` venv only):**

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py
```

| Item | Value | Source |
|---|---|---|
| Pass threshold | `overall_max_abs < 1e-3` m³/s | `examples/compare_ddr_sandbox.rs` |
| Typical CPU result | `max abs ≈ 1.5e-5` m³/s | f32 precision floor |
| Reaches | 5 — RAPID2 `[10,20,30,40,50]` | `examples/compare_ddr_sandbox.rs` |
| Geometry | length 5000 m, slope 0.001, x_storage 0.25 | `examples/compare_ddr_sandbox.rs` |
| Spatial params | n 0.5, q_spatial 0.5, p_spatial default 21.0 | `examples/compare_ddr_sandbox.rs` |
| Default backend | NdArray (CPU) under `Autodiff` | `examples/compare_ddr_sandbox.rs` |
| GPU override | `DDRS_FORCE_GRAPHS=1` → `Cuda<f32, i32>` | `examples/compare_ddr_sandbox.rs` |
| CSV header | `reach_id,max_abs_diff,mean_abs_diff,max_rel_diff,ddr_mean,ddrs_mean,corr` | `examples/compare_ddr_sandbox.rs` |
| Fixture dir | `fixtures/sandbox/` (gitignored, 6 files) | `scripts/export_ddr_sandbox.py` |

**Gotchas:**

- **`fixtures/sandbox/` is gitignored.** Fresh clones must regenerate
  via the DDR `uv` venv. The fixtures are not large but they're derived
  artifacts, not source.
- **`output/` must exist before running.** The example writes
  `output/ddrs_vs_ddr.{csv,png}` via `File::create`, which does not
  `mkdir -p`. A fresh worktree panics on the file create if `output/` is
  missing — run `mkdir -p output` first.
- **`DDRS_FORCE_GRAPHS=1`** switches the inner backend to
  `Cuda<f32, i32>` so V1 exercises the CUDA graph-capture path. The CPU
  NdArray run is the default, but the graph path is what production
  training uses, so both must pass for a clean V1.
- **`--release` is mandatory.** Debug builds are slower and exercise
  different fused kernels in some BURN ops; the regression target
  assumes release.
- **V1 is the load-bearing port invariant.** Never bypass it, never
  relax the `1e-3 m³/s` threshold, never declare a "good enough" pass.
  If a change cannot keep V1 green, it cannot land.

There is no separate test that wraps V1 — the example *is* the
verification gate. Reproducibility floor: `overall_max_abs` stays in the
`~1e-5 m³/s` range across machines and BURN backends. Drifts within that
floor are noise; drifts above the `1e-3 m³/s` threshold are bugs.

## See also

- [Setup](../setup.md) — bringing up the toolchain, the cubecl fork, and
  the DDR reference repo that feeds V1's fixtures.
- [Architecture](../architecture.md) — module map and which paths affect
  V1.
- [Algorithm](../algorithm.md) — the Muskingum-Cunge math V1 is
  verifying.
- [Performance & CUDA Graphs](perf.md) — the graph-capture path that
  `DDRS_FORCE_GRAPHS=1` exercises.
- [Reading outputs](../usage/outputs.md) — the format of
  `output/ddrs_vs_ddr.{csv,png}`.
- [Running the code](../usage/running.md) — how the example is invoked
  alongside training and evaluation.
