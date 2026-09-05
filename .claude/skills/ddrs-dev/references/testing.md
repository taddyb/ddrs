# Testing: gates, inventory, authoring

70 test files, 233 `#[test]` fns. Full suite runs in 2–5 minutes on the dev machine.
Each `tests/*.rs` compiles as its own crate; `mod common;` pulls `tests/common.rs`.

**Do not hard-code test counts in docs.** Prior skills claimed "leakance_gradcheck
8/8" and "zeta_accum 6/6"; the real counts are 16 and 8, and they grow. Assert "all
pass" instead.

## Tier gates

### Tier A — routing core
`src/routing/`, `src/geometry.rs`, `src/sparse/`

```bash
cargo test --test ddr_sandbox_match     # machine-enforced invariant 1 (added 2026-09-02)
mkdir -p output && cargo run --release --example compare_ddr_sandbox  # must print ABSOLUTE MATCH
cargo test --lib
cargo test --test mmc
cargo test --test sparse_gradcheck
cargo test --test leakance_gradcheck    # run even if you did not touch leakance —
cargo test --test leakance_off_parity   # any routing change can disturb OFF-parity
cargo test --test zeta_accum
```

Since 2026-09-02 the sandbox gate is machine-enforced twice over:
`tests/ddr_sandbox_match.rs` asserts max abs < 1e-3 in-process (CPU NdArray,
part of plain `cargo test`), and the example itself **exits 1** on anything
short of ABSOLUTE MATCH (after writing its CSV/PNG diagnostics), so scripted
callers no longer need to read stdout. The example remains the diagnostic
form (per-reach table, PNG, `DDRS_FORCE_GRAPHS` GPU path).

### Acceptance — end-to-end metric floors (Juniata)

```bash
cargo test --release --test juniata_acceptance -- --nocapture   # ~20 s after build
```

The only test covering the full data → train → route → eval → metric chain.
Runs `train-and-test` on the committed Juniata bundle (CPU, tempdir
workspace), then asserts against the run manifest: routed median NSE ≥ 0.75
and KGE ≥ 0.80 (seed-42 reference 0.790 / 0.881; floors are loose on purpose
so legitimate op-reordering noise doesn't trip them), routed NSE beats the
summed-Q' baseline, and baseline NSE ∈ [0.67, 0.72] (deterministic, 0.6947
measured — doubles as a data-reader regression check). **Silently skips under
debug_assertions** (a debug train takes minutes), so it only runs with
`--release`; run it for any change to routing, training, eval, or the data
readers when you want end-to-end confirmation. Verified 2026-09-02:
NSE 0.7903 / KGE 0.8810 / baseline 0.6947 in 18.3 s.

### Tier B — KAN head
`src/nn/`, `Cargo.toml` rskan tag

```bash
cargo test --features fixtures \
  --test kan_head_init_repro --test kan_head_init_parity \
  --test kan_head_fixture_forward --test kan_head_fixture_backward
```
Then run Tier A to confirm the head change did not disturb routing.

### Tier C — everything else in `src/`
```bash
cargo test --lib && cargo test && \
  mkdir -p output && cargo run --release --example compare_ddr_sandbox
```
If you touched `src/training/forward.rs` (disagg / leakance threading), also run
`cargo test --test leakance_off_parity`.

### Tier D — config YAML only
```bash
ddrs plan --config config/experiments/<x>.yaml \
          --workspace /home/tbindas/projects/ddrs/.ddrs
```
Exit 0, no drift warnings.

## CI (added 2026-09-03)

`.github/workflows/ci.yml`, on every PR and master push (no path filters:
required checks must never be skipped): job `test` = debug
`cargo test --features fixtures --no-fail-fast` (13 min warm, 12–17 min
cold); job `acceptance` = release `compare_ddr_sandbox` +
`juniata_acceptance` (17 min warm, ~25 min cold). Warm is only modestly
faster because `rust-cache` keeps dependency artifacts only: the ddrs
crate and every test binary rebuild on each run (measured 2026-09-04,
runs 33817851139 cold and 33830503587 warm). The acceptance job
builds with `CARGO_PROFILE_RELEASE_LTO=false` (env override in the workflow
only): the thin-LTO link of the test binary was 30 of 40 minutes on the
2-core runner while the 30 training epochs took ~13 s, so the training is
not the cost and the full metric floors are kept. Branch protection requires
both; `enforce_admins` is off so admin direct pushes remain possible (vetted
post-hoc by the push run).

**What green CI does NOT prove:** `--features cuda` tests (compiled out),
data-dependent tests (self-skip: no `/mnt/ssd1`/cluster data on runners),
and anything in the do-not-use list. A data-touching change still needs the
local tier gates.

Both jobs install a CUDA toolkit only because compilation requires it
(build-and-env.md). Local hook: `git config core.hooksPath .githooks`
enables `.githooks/pre-push` (runs `ddr_sandbox_match`; bypass with
`git push --no-verify`).

## What covers what

| Area | Tests |
|---|---|
| Routing | `mmc`, `routing_utils`, `geometry` |
| DDR parity (invariant 1) | `ddr_sandbox_match` (in-process, plain `cargo test`) + `compare_ddr_sandbox` example (diagnostics, exits 1 on mismatch) |
| End-to-end acceptance | `juniata_acceptance` (release-only; metric floors + beats-baseline on the committed bundle) |
| Sparse / autograd | `sparse_gradcheck`, `sp8_gradcheck` |
| KAN head | the 4 `kan_head_*` fixture tests (need `--features fixtures`) |
| Leakance | `leakance_gradcheck`, `leakance_off_parity`, `zeta_accum` |
| Adjacency | `adjacency_parity` (managed builder byte-identical to the petgraph engine on `order`/`indices_0`/`indices_1`), `adjacency_build`, `data_zarr_store::conus_adjacency_loads_real_merit_zarr` (invariant 3 on real CONUS data) |
| CLI / data | `data_dataset`, `data_static`, `cli_manifest`, `cli_lockfile`, `cli_json_contract` |
| Checkpointing | `checkpoint_resume` (**not** `cargo test --lib training::checkpoint` — that module has zero tests) |
| Eval robustness | `cargo test --lib training::eval::tests` |
| Disagg freeze | `disagg_freeze` |
| Grad accumulation | `grad_accum_equivalence`, `adadelta_nse_smoke` (on `exp_train`) |

Two commands that appear in CLAUDE.md and `docs/` and **run zero tests**:
`cargo test --test mmc mc_routes_linear_chain` (no such test) and
`cargo test --test sp8_gradcheck -- --ignored` (nothing in that file is `#[ignore]`,
so it passes vacuously). Do not use either as a gate.

## Acceptance thresholds

| Check | Bar |
|---|---|
| DDR sandbox max abs diff | < 1e-3 m³/s |
| Typical per-reach rel diff | ~1e-7 (the f32 floor). Regression past 1e-6 warrants investigation |
| `zeta_accum` headwater identity | abs error < 1e-6 · max(\|zeta\|, 1.0) |
| KAN fixture forward/backward | bit-for-bit |
| Gradcheck | rel < 5e-3 **or** abs < 1e-4 |

## Authoring patterns

### Gradcheck
Follow `tests/leakance_gradcheck.rs`. Central difference
`(f(x+eps) − f(x−eps)) / (2·eps)` on the deterministic `NdArray<f32>` backend.

**ε depends on the parent's nonlinearity — this is the non-obvious part:**
- Parents the output is **nonlinear** in (`n`, `q_spatial`, `p_spatial`):
  `eps = max(1e-3·|base|, 1e-3)`. Smaller hits the f32 noise floor.
- Parents the output is **exactly linear** in (`K_D`, `d_gw`, `leakance_factor`):
  use a **large** step — `K_D 4e-7` (base 5e-7), `d_gw 1.5` (base 0.0),
  `factor 0.4` (base 0.5). Central differences have zero truncation error for
  exactly-linear parents, while a tiny step sinks Δloss into the f32 round-off of
  `q_next.sum()` (~O(500)).

Tolerances `REL_TOL = 5e-3`, `ABS_TOL = 1e-4`; accept on rel **or** abs. The base
point must be **interior** to all ranges (no clamp saturation), and for leakance a
**losing** config (`depth > d_gw`). `x_storage` is a non-differentiated constant of
`TimestepLeakanceOp` — `.grad()` returns `None`; do not try to sweep it.

### Parity
Follow `tests/leakance_off_parity.rs`. Capture the `EXPECTED` array **before** the
feature branch exists, on the deterministic CPU backend, and assert **bit-exact**
equality — not approximate. Test **both** directions:
1. feature off ⇒ byte-identical to the committed expectation, and
2. feature on ⇒ output demonstrably changes.

Only the second catches a silent no-op, which is exactly what the 2026-07-01 stale
binary produced.

### Fixtures
Generate under `~/projects/ddr/.venv`, write raw bytes to `tests/fixtures/`
(**tracked**, unlike the gitignored top-level `/fixtures/`), load behind
`#[cfg(feature = "fixtures")]`.

## Why the zeta_accum headwater identity works

Reach 0 is a headwater (no upstream), so the triangular solve gives
`x_sol[0] = b_rhs[0]`. Therefore `q_no_leak[0] − q_leak[0] == zeta[0]` exactly.
That identity is what proves the eval-time zeta accumulator reports precisely what
was subtracted from `b_rhs`, rather than a recomputation that might drift.

## Checkpoint f16 drift is expected

`CompactRecorder` = `HalfPrecisionSettings`, so weights and Adam moments are stored
as f16. A resumed trajectory drifts slowly from an uninterrupted one. This is not a
bug unless drift exceeds a few thousandths of NSE over many epochs. Resume *state*
(epoch, mini-batch cursor, rng, sampler permutation) is exact.
