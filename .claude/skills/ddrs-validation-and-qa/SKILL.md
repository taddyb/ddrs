---
name: ddrs-validation-and-qa
description: "Use when deciding whether a ddrs code change is safe to merge, verifying a run produced correct output, understanding what tests exist and what they guarantee, adding a new test, or interpreting a test failure. Also use when you need the acceptance thresholds for gradient-exactness, DDR parity, KAN head parity, or eval-time metrics."
---

# ddrs: Validation and QA Runbook

**Audience:** A mid-level ML engineer who knows PyTorch but not Rust/BURN.
Every jargon term is defined on first use.

---

## When NOT to use this skill

- You are writing a new feature from scratch — use `superpowers:writing-plans` first.
- You want to understand the routing algorithm — use `.claude/ARCHITECTURE.md`.
- You want to understand the autograd / sparse-backward design — use `.claude/references/ddrs-burn-autograd.md`.
- You are diagnosing a hydrology result (NSE/KGE interpretation) — use `ddrs-research-frontier`.

---

## Glossary (one definition each)

| Term | Meaning |
|---|---|
| **BURN** | Rust deep-learning framework (like PyTorch but for Rust); version 0.21 in ddrs |
| **DDR** | The Python/PyTorch reference implementation at `~/projects/ddr/` |
| **CSR** | Compressed Sparse Row — the matrix format for the reach-adjacency graph |
| **MC / MMC** | Muskingum-Cunge routing — the per-timestep hydraulic solver |
| **KAN head** | Kolmogorov-Arnold Network head (`src/nn/kan_head.rs`); maps catchment attributes to routing parameters |
| **rskan** | Rust KAN library, pinned at `v0.1.3` in `Cargo.toml` |
| **Q' (Qr)** | Predicted upstream streamflow forcing fed into the routing |
| **zeta** | The eval-time GW–SW water-loss flux: `leakance_factor · area_z · K_D · (depth − d_gw)` |
| **f32** | 32-bit float; the precision invariant throughout the routing core |
| **NSE** | Nash-Sutcliffe Efficiency; 1 = perfect, 0 = climatology mean |
| **KGE** | Kling-Gupta Efficiency; includes bias and variance-ratio terms L1 misses |
| **COMID** | MERIT reach identifier (integer) |
| **STAID** | Gauge identifier (string, e.g. `USGS__01234500`) |
| **rho-window** | A randomly sampled contiguous sub-sequence of the time axis used in mini-batch training |
| **hotstart transient** | The initial-condition mismatch at the start of a rho-window; unresolvable without persistent state |

---

## The Five Critical Invariants

These are the properties that make the port meaningful. Breaking any one invalidates the whole codebase.

| # | Invariant | What breaks if violated |
|---|---|---|
| 1 | `examples/compare_ddr_sandbox` reports **ABSOLUTE MATCH** (max abs diff < 1e-3 m³/s) | Port diverges from DDR reference |
| 2 | **f32 throughout the routing core** — no f64, no bf16 casts | DDR comparison leaves the f32 precision floor (~1e-7 rel diff/reach) |
| 3 | **Adjacency is lower-triangular** (`rows[k] >= cols[k]`) | Forward-sub solver produces wrong routing order |
| 4 | **Hand-written sparse backward** in `src/sparse.rs` (`CsrSolveOp impl Backward`) is never replaced with autograd unrolling | Tape size grows from O(nnz) to O(n²) per timestep |
| 5 | **KAN head is `rskan::KanLayer` v0.1.3** — `Linear(F,H) → KanLayer(H,H)×N → Linear(H,P) → Sigmoid`, no inter-block ReLU, same seed for all inner KanLayers | Head deviates from DDR's `kan.py`; gradient parity breaks |

---

## Certified Test Inventory

Every test here is a file under `tests/` (each file compiles as a separate Rust crate). Run with `cargo test --test <name>`.

### Gate 1 — DDR parity (must pass before any merge to `src/routing/`, `src/geometry.rs`, `src/sparse.rs`)

```bash
cargo run --release --example compare_ddr_sandbox
# Expected output: "ABSOLUTE MATCH" + max abs diff printed (must be < 1e-3 m³/s)
```

**What it checks:** 5-reach RAPID sandbox routed for multiple timesteps; ddrs output vs DDR fixture.
**Caveat (as of 2026-06-06):** The fixture was generated from `~/projects/ddr` (unpushed `geometry/trapezoidal.py`). A fixture regenerated from a clean DDR clone will differ ~1% — that is a wrong-reference problem, not a port bug. See `.claude/references/ddrs-comparing-to-ddr.md §Regenerating fixtures`.

### Gate 2 — Sparse backward gradient-exactness

```bash
cargo test --test sparse_gradcheck
```

**What it checks:** The custom `CsrSolveOp` backward (invariant 4) matches finite differences. If this fails, gradient flow through the sparse solver is broken.

### Gate 3 — Routing correctness (unit and integration)

```bash
cargo test --test mmc                           # all MMC tests
cargo test --test mmc mc_routes_linear_chain    # one specific test by name
cargo test --test routing_utils
cargo test --test geometry
```

**What they check:** `MuskingumCunge` forward produces expected output on simple chain networks; geometry (trapezoidal channel) computes depth/area correctly.

### Gate 4 — KAN head parity vs DDR (required on any PR touching `src/nn/`, `Cargo.toml` rskan pin, or DDR `nn/kan.py`)

```bash
cargo test --features fixtures --test kan_head_init_repro
cargo test --features fixtures --test kan_head_init_parity
cargo test --features fixtures --test kan_head_fixture_forward
cargo test --features fixtures --test kan_head_fixture_backward
```

**What they check:** KAN head weights initialize identically to DDR; forward and backward passes match DDR fixtures bit-for-bit. Fixtures live in `tests/fixtures/` and are generated by `scripts/dump_kan_*.py` under `~/projects/ddr/.venv`. If DDR changes `kan.py`, regenerate fixtures before re-running.

Additional head shape/gradient tests (no fixture required):

```bash
cargo test --test kan_head
```

### Gate 5 — Leakance gradient-exactness and isolation (required on any change to `src/routing/leakance.rs`)

```bash
cargo test --test leakance_gradcheck    # analytical backward ≈ finite-difference (8 cases)
cargo test --test leakance_off_parity   # byte-identical to no-leakance when disabled (3 cases)
cargo test --test zeta_accum            # eval zeta accumulation identity (6 cases)
```

**What they check:**

- `leakance_gradcheck`: The `TimestepLeakanceOp: Backward` impl is gradient-exact.
- `leakance_off_parity`: With `use_leakance: false`, routing output is bit-identical to the base (no zeta subtraction sneaking in).
- `zeta_accum` (verified from `tests/zeta_accum.rs`): Six tests covering:
  1. `zeta_sums_none_when_leakance_off_or_not_enabled`
  2. `accumulation_does_not_perturb_discharge`
  3. `accumulated_zeta_equals_headwater_qnext_difference` — the **headwater identity**: reach 0 is a headwater (no upstream), so `x_sol[0] = b_rhs[0]`, therefore `q_no_leak[0] - q_leak[0] == zeta[0]`. This proves the accumulator reports exactly what was subtracted from `b_rhs`.
  4. `zeta_is_linear_in_leakance_factor_on_single_step`
  5. `q_mean_matches_routed_discharge`
  6. `depth_and_area_z_are_leakance_independent_primitives`

### Gate 6 — Adjacency correctness

```bash
cargo test --test adjacency_parity      # managed builder matches engine-built store element-for-element
cargo test --test adjacency_build
cargo test --test data_zarr_store       # includes conus_adjacency_loads_real_merit_zarr (invariant 3)
```

**What they check:** The lower-triangular invariant (invariant 3) is satisfied on real MERIT data; the pure-Rust adjacency builder produces byte-identical `order`, `indices_0`, `indices_1` as the petgraph engine.

### Gate 7 — Data readers and CLI contracts

```bash
cargo test --test data_dataset
cargo test --test data_zarr_store
cargo test --test data_static
cargo test --test cli_manifest
cargo test --test cli_lockfile
cargo test --test cli_json_contract
```

**When to run:** When touching data readers (`src/data/`), CLI lifecycle (`src/cli/`), or manifest schema.

### Run all tests at once

```bash
cargo test
```

Expected: all pass on a clean build. The full suite takes 2–5 minutes on the dev machine.

---

## Acceptance Thresholds

| Metric / Property | Pass Threshold | Fail Action |
|---|---|---|
| DDR sandbox max abs diff | < 1e-3 m³/s | Block merge; bisect `src/routing/`, `src/geometry.rs`, `src/sparse.rs` |
| DDR sandbox rel diff (typical) | ~1e-7 per reach (f32 floor) | Any regression past 1e-6 warrants investigation |
| Sparse backward gradcheck | All cases within finite-diff tolerance | Block merge; the backward formula has a bug |
| Leakance gradcheck | 8/8 cases pass | Block merge to `src/routing/leakance.rs` |
| Leakance off-parity | 3/3 byte-identical | Block merge |
| zeta_accum headwater identity | abs error < 1e-6 × max(|zeta|, 1.0) | Block merge |
| KAN head fixture forward/backward | Bit-for-bit DDR match | Regenerate fixture or fix head; block PR |
| Eval: median NSE on 2365 CONUS gauges | > 0.689 (summed-Q' baseline, as of 2026-07-05) | Training is not earning its keep; debug gradient stats |
| Eval: median KGE on 2365 CONUS gauges | KGE does NOT beat 0.723 baseline in any config (as of 2026-07-05) — this is a known gap, not a test failure | Investigate KGE `α` term; use `nnse-kge` loss |
| Leakance eval: `|zeta| > 0.01 m³/s` fraction | ≥ 10% of eval reaches (GO bar) | Report as NO-GO on leakance feasibility |

---

## Performance Baselines (as of 2026-07-05)

These are empirical results, not test pass/fail thresholds, but you need them to interpret eval output.

| Config | Median NSE | Median KGE | Gauges | Run ID |
|---|---|---|---|---|
| Summed-Q' (no routing) | 0.689 | 0.723 | CONUS | — |
| Precip-disagg + L1 (hourly, best run) | 0.7152 | 0.7106 | 2365 | `2026-06-23T02-49-12Z-conus-hourly-train-and-test` |
| Daily L1 baseline | ~0.684 | ~0.701 | 2365 | see `docs/6_19_26_journal.md` |

**Critical fact:** KGE does NOT beat the summed-Q' baseline in any config as of 2026-07-05. NSE beats it (+0.037 with precip disagg). The deficit is diagnosed as L1 loss rewarding over-attenuation (NSE optimum at α < 1, which depresses the KGE α term). The `nnse-kge` loss (`src/training/loss.rs`) exists to fix this but has not yet produced a KGE win.

---

## Known Gotchas That Silently Invalidate Results

These pitfalls produce plausible-looking output that is actually wrong.

### Gotcha 1: Stale-Binary Trap

`cargo build` and `cargo run` do NOT update `~/.cargo/bin/ddrs`. If you type `ddrs run …` after editing `src/`, you run the old binary silently. The manifest's `git.sha` is stamped from `.git` at runtime — it will say the current SHA even if the binary is weeks old.

**How to detect:** Current checkpoints are **directories** (`.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/head.mpk`). If you see flat `.mpk` files at `.ddrs/runs/<id>/checkpoints/epoch_E_mb_M.mpk`, you ran a stale pre-checkpoint-resume binary.

**Fix (choose one):**
```bash
cargo install --path .                             # canonical refresh (slow)
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs  # fast if already built
cargo run --release --bin ddrs -- run --workflow … # bypass installed copy entirely
```

**Historical impact:** The 2026-07-01 leakance×hourly 2×2 ran the stale binary before disaggregation was added, producing byte-identical hourly and daily cells (a false "disagg no-op"). See `docs/2026-07-01-leakance-hourly-experiment-handoff.md`.

### Gotcha 2: CUDA Graphs Mask NaN

`use_cuda_graphs: true` caches the computation graph. When a forward pass produces NaN (e.g., a parameter out of range), the CUDA graph returns the stale finite value from the previous cached execution. Loss appears to descend normally while the model is broken.

**Diagnosis:** Run with `use_cuda_graphs: false` and check for non-finite loss. Config `leakance + use_cuda_graphs: true` is rejected at config-load time (this combination cannot be captured without a separate graph path).

### Gotcha 3: Leakance Requires Three Config Changes Together

Enabling leakance with only `params.use_leakance: true` and forgetting to add `K_D`/`d_gw`/`leakance_factor` to `kan_head.learnable_parameters` and `params.parameter_ranges` produces a silently degraded run. The KAN head will not emit leakance parameters and the term will be zero.

Required config block:
```yaml
params:
  use_leakance: true
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-6]    # log-space; 1/s
    d_gw: [-2.0, 2.0]         # m
    leakance_factor: [0.0, 1.0]
kan_head:
  learnable_parameters: [n, q_spatial, K_D, d_gw, leakance_factor]
```

Note: K_D was widened to `[1e-8, 1e-5]` in recoverability experiments (2026-07-03) after the original ceiling proved binding for expressibility. Consider this wider range for future leakance experiments.

### Gotcha 4: Checkpoint Resume Slow Drift

Checkpoints store weights/moments in f16 (`CompactRecorder = HalfPrecisionSettings`). A resumed trajectory drifts slowly from the uninterrupted one. This is expected and acceptable; do not treat it as a bug unless the drift exceeds a few thousandths of NSE after many epochs.

---

## Leakance Experimental Status (as of 2026-07-05)

The leakance GW–SW term is **experimental**, off by default. This section documents what has been measured so future agents do not re-run closed experiments.

### 2×2 (leakance × forcing): GO-marginal verdict (2026-07-01)

All three gate criteria met. Key numbers from the eval network (64,892 CONUS reaches):

| Criterion | Measured | Bar | Status |
|---|---|---|---|
| Losing-subset skill gain under hourly | ΔNSE +0.0005, ΔKGE +0.0018, 55.5% gauges improve | positive | GO |
| Effect weaker/absent under daily | ΔNSE −0.0017, ΔKGE −0.0009 | daily should hurt | GO |
| |zeta| > 0.01 m³/s fraction | 10.4% of 64,892 eval reaches | ≥ 10% | GO (no headroom) |

### Low-zeta diagnosis (2026-07-02)

Seven pre-registered hypotheses on why learned zeta is small (median |zeta| 6.4e-4 m³/s):

| Hypothesis | Verdict | Key number |
|---|---|---|
| H1: K_D structural ceiling | REFUTED | 71.5% of reaches CAN exceed bar in-box; median utilization 3.4% |
| H2: driving-head starvation | SUPPORTED | median head 0.021 m; 47.0% of reaches gaining at mean |
| H3: KAN variance collapse | REFUTED | d_gw–meanP Spearman +0.71; K_D–aridity +0.61 |
| H4: gauge bias / gradient starvation | SUPPORTED | zeta–uparea ρ +0.76; gauged 6.7e-3 vs ungauged 5.9e-4; dry/wet ratio 0.40 (inverted from physics) |
| H5: equifinality with routing params | SUPPORTED (daily only) | daily Δn = +0.012 (0.59 IQR); hourly Δn nil |
| H6: wrong yardstick | REFUTED | fractional loss agrees: 8.4% lose >1% of local flow |
| H7: model-form error (d_gw pinning) | REFUTED | 0.0% of d_gw at bounds |

**Implication:** K_D widening is NOT recommended (was the top follow-up from 2026-07-01; superseded by the diagnosis). The binding constraint is training signal, not the parameter box.

### Gradient probe (2026-07-03, worktree `zeta-sensitivity`)

| Hypothesis | Verdict | Key number |
|---|---|---|
| P1: gradient starvation | REFUTED | gauged/ungauged |g| ratio 1.5× trained, 2.9× cold (bar: ≥10×) |
| P2: objective rejection | REFUTED | 52.5% dry-tercile grads push zeta down (bar: >67%) |
| P3: detectability | NO-GO | 4.2% of Ref probes detectable at δ=0.01 m³/s (bar: ≥10%) |

Mechanism: a 0.01 m³/s reach loss transmits to its gauge at ~95% fidelity, but the median Ref gauge's 5% uncertainty band is 0.531 m³/s — 53× larger. No gauge-discharge objective can reward what it cannot distinguish from measurement noise.

### Synthetic recoverability positive control (2026-07-04, worktree `zeta-sensitivity`)

**FAILED.** Positive control verdict R1 median recovery ratio = 0.009 (bar: ≥0.5).

Root cause: the windowed training objective (rho=90, warmup=5) has a hotstart-transient noise floor ~130× larger than the planted signal (step-0 windowed loss 1.017 vs continuous residual 0.0076 mean L1). The optimizer chases irreducible initial-condition noise and never sees the planted signal. After 5 epochs, continuous residual grows from 0.0076 to 0.4431 (58× worse than not training).

**Implication: leakance identifiability is NOT proven as of 2026-07-05.** Phase B (state-cache hotstart, objective: ≤ 0.25 mean L1 noise floor, ≤ 10% of converged-run loss) is required before any identifiability claim. Do NOT state that leakance is identifiable.

---

## How to Add a New Test

Rust integration tests in `tests/` each compile as a separate crate. To add one:

1. Create `tests/my_test.rs`.
2. Add `mod common;` at the top if you need the shared test helpers (mock configs, mock inputs, etc.) from `tests/common.rs`.
3. Write `#[test]` functions. Each function that panics = test failure.
4. Run it: `cargo test --test my_test`.

**For a gradcheck test** (verifying a new backward op is gradient-exact):
- Follow the pattern in `tests/leakance_gradcheck.rs`.
- Perturb each input tensor by `+eps` and `-eps`, compute `(f(x+eps) - f(x-eps)) / (2*eps)`, compare to the analytic backward. Use `eps = 1e-3` for f32 (smaller values hit f32 noise floor).
- Tolerance: `abs_error < 1e-4 * max(|grad|, 1.0)` is typical for f32.

**For a parity test** (verifying a new feature does not perturb the default path):
- Follow the pattern in `tests/leakance_off_parity.rs`.
- Run identical inputs through the feature-off path and the new code with the feature disabled. Assert `==` on the output tensors (bit-exact, not approximate).

**For a DDR-fixture test:**
- Generate the fixture in Python: `cd ~/projects/ddr && uv run python <script>`.
- Write the fixture bytes into `tests/fixtures/` (tracked in git under the `fixtures` feature flag).
- Load and compare in the test with `#[cfg(feature = "fixtures")]`.

---

## Quick Reference: What to Run Before Merging

### Any change to `src/routing/`, `src/geometry.rs`, `src/sparse.rs`

```bash
cargo test --test sparse_gradcheck
cargo test --test mmc
cargo test --test routing_utils
cargo test --test geometry
cargo run --release --example compare_ddr_sandbox
# Must print: "ABSOLUTE MATCH"
```

### Any change to `src/routing/leakance.rs`

```bash
cargo test --test leakance_gradcheck
cargo test --test leakance_off_parity
cargo test --test zeta_accum
cargo run --release --example compare_ddr_sandbox
```

### Any change to `src/nn/kan_head.rs` or `Cargo.toml` rskan pin

```bash
cargo test --test kan_head
cargo test --features fixtures --test kan_head_init_repro
cargo test --features fixtures --test kan_head_init_parity
cargo test --features fixtures --test kan_head_fixture_forward
cargo test --features fixtures --test kan_head_fixture_backward
```

### Any change to `src/data/` or `src/cli/`

```bash
cargo test --test data_dataset
cargo test --test data_zarr_store
cargo test --test adjacency_parity
cargo test --test cli_manifest
cargo test --test cli_lockfile
```

### Full suite (before tagging a release or merging a major feature)

```bash
cargo test
cargo run --release --example compare_ddr_sandbox
```

---

## Provenance and Maintenance

Facts and thresholds in this skill are sourced from (as of 2026-07-05):

- `/home/tbindas/projects/ddrs/CLAUDE.md` — invariants, commands, leakance config, stale-binary trap
- `/home/tbindas/projects/ddrs/tests/zeta_accum.rs` — zeta accumulator test names and identities
- `/home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md` — H1–H7 verdicts and all key numbers
- `/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-03-zeta-gradient-probe-findings.md` — P1/P2/P3 verdicts
- `/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-04-synthetic-recoverability-findings.md` — R1–R5 verdicts, 130× noise floor
- `/home/tbindas/projects/ddrs/docs/2026-06-23-precip-disaggregation-findings.md` — best NSE/KGE result
- `/home/tbindas/projects/ddrs/docs/6_19_26_journal.md` — summed-Q' baseline 0.689/0.723
- `/home/tbindas/projects/ddrs/Cargo.toml` — rskan v0.1.3

Re-verification commands (run these if the skill is stale):

```bash
# Confirm rskan version
grep "rskan" /home/tbindas/projects/ddrs/Cargo.toml

# Confirm test files exist
ls /home/tbindas/projects/ddrs/tests/leakance_gradcheck.rs \
      /home/tbindas/projects/ddrs/tests/leakance_off_parity.rs \
      /home/tbindas/projects/ddrs/tests/zeta_accum.rs \
      /home/tbindas/projects/ddrs/tests/sparse_gradcheck.rs \
      /home/tbindas/projects/ddrs/tests/kan_head.rs

# Confirm invariant 1 still holds on current code
cargo run --release --example compare_ddr_sandbox \
  --manifest-path /home/tbindas/projects/ddrs/Cargo.toml
```
