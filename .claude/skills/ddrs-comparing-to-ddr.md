---
name: ddrs-comparing-to-ddr
description: How to verify ddrs against the DDR reference — V1 ABSOLUTE MATCH invariant on the 5-reach RAPID sandbox; regenerating fixtures via scripts/export_ddr_sandbox.py under DDR's uv venv.
output: reference/ddr-comparison.md
sources:
  - examples/compare_ddr_sandbox.rs
  - scripts/export_ddr_sandbox.py
  - CLAUDE.md
---

# ddrs-comparing-to-ddr

> Canonical agent-readable skill. Published chapter at `docs/reference/ddr-comparison.md`
> is regenerated from this file by `/regenerate-docs`.

## What to know

ddrs is a gradient-exact Rust/BURN port of the differentiable Muskingum-Cunge
routing solver from `~/projects/ddr` (Python/PyTorch). The whole point of the
port is to reproduce DDR's forward and backward numerics bit-for-bit at the
f32 precision floor; everything else (CUDA backends, sparse solvers, training
loops) is downstream of that contract. The `compare_ddr_sandbox` example —
"V1" — is the regression that pins this. Any drift means the port has broken
its contract, and the offending change must be reverted or reconciled before
shipping. There is no acceptable narrative where V1 is "close enough but not
matching" — either it reports ABSOLUTE MATCH or the port is broken.

## The V1 regression

```bash
cargo run --release --example compare_ddr_sandbox
```

This replays DDR's `tests/benchmarks/test_ddr.py` sandbox routing through
ddrs's `MuskingumCunge` solver on identical inputs:

- **5 reaches** in RAPID2 order `[10, 20, 30, 40, 50]`.
- **238 timesteps** of lateral inflow (`qprime_topo.csv`).
- Identical dense adjacency, channel geometry (length=5000 m, slope=0.001,
  `x_storage=0.25`), and spatial parameters (`n=0.5`, `q_spatial=0.5`,
  `p_spatial=21.0` default).

Threshold: `overall_max_abs < 1e-3 m³/s`. A passing run prints

```
verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

Typical passing result on CPU (NdArray backend) is `max abs ≈ 1.5e-5 m³/s` —
roughly two orders of magnitude under the threshold. That margin is the
f32 precision floor; closing it further is not a goal.

Outputs land at:

- `output/ddrs_vs_ddr.csv` — per-reach `max_abs_diff`, `mean_abs_diff`,
  `max_rel_diff`, `ddr_mean`, `ddrs_mean`, Pearson `corr`.
- `output/ddrs_vs_ddr.png` — DDR (solid) and ddrs (dashed) hydrographs
  overlaid, one panel per reach.

## Regenerating fixtures

The fixture directory `fixtures/sandbox/` is gitignored. When DDR's solver
changes upstream, regenerate the fixtures from the DDR reference:

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py
```

This must run under DDR's `uv` venv — the script imports
`tests.benchmarks.conftest`, `ddr_engine.merit`, and `ddr.dmc`, none of which
are on ddrs's side. It writes six files into `fixtures/sandbox/`:

| File | Shape | Meaning |
|---|---|---|
| `topo_order.csv` | `(5,)` | Reach IDs in topological order |
| `rapid2_order.csv` | `(5,)` | Reach IDs in RAPID2 order `[10,20,30,40,50]` |
| `qprime_topo.csv` | `(T, N)` | Lateral inflow, topological order |
| `adjacency_topo.csv` | `(N, N)` | Dense adjacency, lower-triangular |
| `ddr_discharge_rapid2.csv` | `(N, T)` | DDR's routed output, RAPID2 order |
| `config.csv` | key=value | Length, slope, x_storage, params snapshot |

If you regenerate fixtures, V1 must still pass. If it suddenly fails after
regeneration, DDR's solver moved — investigate the DDR diff before touching
ddrs.

> **The reference DDR state is NOT a pushed commit (as of 2026-06-06).**
> The port mirrors the desktop checkout of `~/projects/ddr`, which contains
> unpushed work — notably `src/ddr/geometry/trapezoidal.py` (the source of
> ddrs's `src/geometry.rs`), which exists at *no* commit in DDR's public
> history. DDR-at-HEAD instead *learns* `top_width`/`side_slope`
> (sandbox: denorm(0.5) log-space), while ddrs derives them
> Leopold-Maddock-style (`top_width = p·depth^q`). Regenerating the
> fixture from a clean DDR clone therefore produces a ~1%-divergent
> reference (max abs ≈ 0.55 m³/s, NSE ≈ 0.9996) at *every* ddrs commit —
> that is a wrong-reference artifact, not a port bug. Until the DDR-side
> geometry work is pushed, only the desktop's DDR tree generates a valid
> V1 fixture. See
> `docs/superpowers/plans/2026-06-06-sigfpe-wukong-debug-handoff.md`
> §Outcome for the full investigation.

## When V1 fails

The example's own `output/ddrs_vs_ddr.csv` is the first diagnostic. Walk it
in this order:

1. **Inspect `output/ddrs_vs_ddr.csv`** for the worst-offending reach(es).
   Per-reach `max_abs_diff`, `max_rel_diff`, and `corr` will isolate whether
   the divergence is a single bad reach (often a geometry / parameter bug) or
   global (often a solver / kernel issue).
2. **Confirm fixtures aren't stale.** Re-run `scripts/export_ddr_sandbox.py`
   under DDR's venv and `git diff fixtures/sandbox/`. If DDR's outputs
   changed, the regression target moved — not a ddrs bug.
3. **Audit recent commits to `src/routing/`, `src/geometry.rs`, `src/sparse/`.**
   These are the only paths that affect V1. Anything else (data loaders,
   training loop, CLI) cannot have broken it. Use `git log -p` on those
   directories since the last known-good SHA.
4. **Check for accidental precision shifts.** Grep the routing path for
   `f64`, `bf16`, `cast`, or `to_dtype` — any silent widening or narrowing of
   the tensor dtype breaks reproducibility against the reference.
5. **Cross-check with `tests/sparse_gradcheck.rs`.** If the gradcheck also
   fails, the algorithm itself changed (numerator / denominator math, sparse
   solve, etc.). If only V1 fails, it's almost always a kernel-ordering or
   arithmetic-fusion difference at the f32 precision floor — look for new
   `fuse`/`einsum`/reduction-order changes in BURN ops or in your own
   tensor pipeline.

## Gotchas

- **`fixtures/sandbox/` is gitignored.** Fresh clones must regenerate via the
  DDR uv venv (see above). The fixtures are not large but they're treated as
  derived artifacts, not source.
- **V1 is the load-bearing port invariant.** Never bypass it, never relax the
  `1e-3 m³/s` threshold, never declare a "good enough" pass. If a change
  cannot keep V1 green, it cannot land.
- **`output/` must exist before running the example.** The example writes
  `output/ddrs_vs_ddr.{csv,png}` with `File::create`, which does not
  `mkdir -p`. `cargo run --release --example compare_ddr_sandbox` from a
  fresh worktree will panic on the file create if `output/` is missing —
  `mkdir -p output` first.
- **`DDRS_FORCE_GRAPHS=1`** env override toggles the CUDA backend + flips
  `use_cuda_graphs=true` and forces `sparse_solver=Cuda`. Use this to verify
  the graph-capture path also produces ABSOLUTE MATCH — the CPU NdArray run
  is the default, but the graph path is what production training uses, so
  both must pass for a clean V1.
- **`--release` is mandatory.** Debug builds are ~20× slower and exercise
  different fused kernels in some BURN ops; the regression target assumes
  release.

## Verification

V1 itself is the verification gate — there is no separate test that wraps
it. The runbook before claiming a routing-core change is safe to merge:

```bash
mkdir -p output
cargo run --release --example compare_ddr_sandbox
# expect: verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox
# expect: same verdict on the CUDA + graph-capture path
```

Reproducibility floor: `overall_max_abs` stays in the `~1e-5 m³/s` range
across machines and BURN backends. Drifts within that floor are noise; drifts
above the `1e-3 m³/s` threshold are bugs.
