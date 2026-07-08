---
name: ddrs-architecture-contract
description: "Use when you are about to touch src/routing/, src/sparse.rs, src/geometry.rs, src/nn/kan_head.rs, src/cuda_graph/, Cargo.toml's rskan pin, or any training/eval path; when you need to know which invariants are load-bearing and why; when a test fails and you need to triage against known weak points; when you are running an experiment and need to know the current performance baseline and which claims are proven vs open; or when setting up the binary install / CLI workflow."
---

# ddrs Architecture Contract

## When NOT to use this skill

- For CLI lifecycle / workflow orchestration details → read `docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md`
- For the BURN 0.21 autograd API recipe → `.claude/references/ddrs-burn-autograd.md`
- For data-source path details (zarr layout, icechunk sniffing) → `.claude/references/ddrs-reading-inputs.md`
- For eval output format / zeta netcdf schema → `.claude/references/ddrs-reading-outputs.md`

---

## 1. What ddrs is (two sentences)

`ddrs` is a **BURN-0.21 Rust port** of DDR, a differentiable Muskingum-Cunge routing solver originally in Python/PyTorch. The port must produce **gradient-exact** outputs against DDR's reference on the 5-reach RAPID sandbox; that guarantee is the V1 invariant and must hold after every commit.

**BURN** = Rust deep-learning framework (analogous to PyTorch). **Muskingum-Cunge** (MC) = a linear reservoir routing scheme where each reach has coefficients c1–c4 derived from channel geometry and Manning's equation. **Gradient-exact** = `max(|ddrs_output - ddr_output|) < 1e-3 m³/s` on that sandbox.

---

## 2. The seven invariants — break any of these and the port is meaningless

| # | Invariant | File(s) | Test / guard |
|---|---|---|---|
| 1 | `examples/compare_ddr_sandbox` reports **ABSOLUTE MATCH** (max abs diff < 1e-3 m³/s) | `src/routing/`, `src/geometry.rs`, `src/sparse.rs` | `cargo run --release --example compare_ddr_sandbox` |
| 2 | **f32 throughout the routing core** — no f64 or bf16 casts inside the timestep chain | `src/routing/mmc_op.rs`, `src/sparse/mod.rs`, `src/geometry.rs` | V1 test; any precision drift breaks DDR parity at the f32 floor (~1e-7 rel diff per reach) |
| 3 | **Adjacency is topologically ordered, lower-triangular** (`rows[k] >= cols[k]`) | `src/sparse/`, `src/adjacency/build.rs` | `cargo test data_zarr_store::conus_adjacency_loads_real_merit_zarr` |
| 4 | **Do NOT replace the hand-written sparse backward** in `src/sparse/mod.rs` (`CsrSolveOp impl Backward`) | `src/sparse/mod.rs` | `cargo test --test sparse_gradcheck` |
| 5 | **Routing head is `rskan::KanLayer`** via `src/nn/kan_head.rs` — `Linear(F,H) → KanLayer(H,H)×N → Linear(H,P) → Sigmoid`, no inter-block ReLU | `src/nn/kan_head.rs` | `cargo test --test kan_head` |
| 6 | **rskan pinned to a tag** in `Cargo.toml` — bump tag, re-run KAN parity tests, validate before merging | `Cargo.toml` | `cargo test --features fixtures --test kan_head_init_repro --test kan_head_init_parity --test kan_head_fixture_forward --test kan_head_fixture_backward` |
| 7 | **leakance + `use_cuda_graphs: true` is a config error** — config load rejects this combination | `src/routing/leakance.rs`, `src/config.rs` | Config validation at `ddrs plan` / `ddrs run` |

### Why invariant 4 matters (O(nnz) vs O(n²))

BURN's default autograd records one node per tensor operation. If you replace `CsrSolveOp`'s hand-written backward with plain tensor unrolling, the tape grows O(n²) per timestep (n = 346,321 CONUS reaches). The custom backward keeps it O(nnz) = O(338,814 edges). Same logic applies to `TimestepOp` in `src/routing/mmc_op.rs` — one node per timestep, not ~33.

### Why invariant 3 matters

Forward substitution on `A = I − c1·N` requires N to be strictly lower-triangular (every reach appears after all its upstream neighbors in the sorted order). If any `rows[k] < cols[k]` entry exists, the solver silently produces wrong answers with no error.

---

## 3. Source tree in one screen

```
src/
├── routing/
│   ├── mmc.rs          MuskingumCunge<I>: setup_inputs, forward, route_timestep
│   ├── mmc_op.rs       TimestepOp — single Backward<I,5> per timestep; saves 23 intermediates
│   ├── leakance.rs     GW–SW loss term; TimestepLeakanceOp: Backward<I,8> (experimental)
│   └── utils.rs        denormalize, hotstart, dense helpers
├── sparse/
│   ├── mod.rs          CsrPattern (Arc-shared), CsrSolveOp + hand-written Backward
│   ├── cusparse.rs     cuSPARSE SpMV + SpSV FFI wrappers (SP-9)
│   └── dispatch.rs     CPU forward-sub vs cuSPARSE SpSV selector
├── cuda_graph/         SP-10 CUDA Graph capture/replay (forward-only; backward not yet captured)
│   ├── capture.rs
│   ├── geometry_kernel.rs  fused #[cube] kernels K1/K2/K3
│   └── scratch.rs
├── geometry.rs         Trapezoidal channel geometry (Leopold & Maddock)
├── config.rs           YAML config, parameter ranges, log-space flags, SparseSolver enum
├── nn/kan_head.rs      KAN head via rskan — matches DDR's kan.py exactly
├── adjacency/build.rs  Managed adjacency builder (topological_sort matches petgraph DFS)
├── data/               Live zarr/netcdf/icechunk readers — no export step
│   ├── ids.rs          Comid(i64), Staid(String) newtypes; IdIndex<T>
│   ├── dates.rs        TimeAxis + rho-window sampler
│   └── store/          zarr.rs, zarr_obs.rs, zarr_qprime.rs, obs_writer.rs
├── training/
│   ├── loss.rs         L1 (default) or nnse-kge; config-selectable
│   ├── forward.rs      LeakanceOverride seam (eval path only)
│   ├── bootstrap.rs    Checkpoint resume: weights + optim + RNG state
│   └── probe.rs        lift_leaf, probe_forward, GradAccum (gradient probe instruments)
└── bin/
    ├── ddrs.rs         Primary CLI: plan / run / show / status / gc / sources / import
    ├── probe_zeta_gradient.rs  gradient probe + synthetic teacher (--mode grad|perturb|teacher|floor|state-cache)
    ├── train.rs        Legacy (deprecated, removed in 0.4)
    └── eval.rs         Legacy; still used for --zeta-output on existing checkpoints
```

---

## 4. Per-timestep dataflow (the MC routing step)

Everything below runs inside `forward_chain_inner` in `src/routing/mmc_op.rs` at the **inner-backend primitive level** — no autograd nodes are created inside this function. One `TimestepOp` node wraps the entire chain.

```
inputs: (n, q_spatial, p_spatial, q_t, q_prime_t)
fixed:  (length, slope, x_storage, dt=3600 s)

K1 — geometry + Muskingum coefficients (one fused #[cube] kernel on CUDA):
  depth        = ((Q·n·(q+1)) / (p·√slope))^(3/(3q+5))
  top_width    = p · depth^q
  side_slope   = clamp(top_width·q / (2·depth), 0.5, 50)
  bottom_width = clamp(top_width − 2·side_slope·depth, bw_lb)
  hyd_radius   = ((top_width+bottom_width)·depth/2) / (bottom_width + 2·depth·√(ss²+1))
  velocity     = clamp((1/n)·R^(2/3)·√slope, v_lb, 15)
  celerity     = velocity · 5/3
  k_musk       = length / celerity
  denom        = 2·k·(1−x) + dt
  c1..c4       = Muskingum coefficients

SpMV:     i_t = N · q_t               (cuSPARSE SpMV on GPU; scatter on CPU)

K2 — RHS assembly:
  b_rhs = c2·i_t + c3·q_t + c4·q_prime_t

[optional leakance, when params.use_leakance: true]
  area_z = (p · depth)^q_eps · length
  zeta   = leakance_factor · area_z · K_D · (depth − d_gw)
  b_rhs  = b_rhs − zeta

A-values:  a_values = assemble_primitive(c1)   [CSR values of A = I − c1·N]

SpSV:      x_sol = triangular_csr_solve(a_values, b_rhs)   [lower-triangular]

K3 — clamp:
  q_next = clamp_min(x_sol, discharge_lb)
```

On the CUDA path (SP-10), K1+K2+K3 are fused `#[cube]` kernels, and the captured per-step sequence is **K1 → SpMV → K2 → assemble → SpSV → K3** — six kernel launches replayed as one `cuGraphLaunch`.

**Cold start (t = 0):** solves `(I − N)·Q_0 = q'_0`. On a linear chain this reduces to `Q_0[i] = Σ_{j ≤ i} q'_0[j]` (cumulative sum).

---

## 5. KAN head architecture

```
Linear(F, H)
  → KanLayer(H, H)   ×  num_hidden_layers   [ALL layers receive the SAME init seed — DDR kan.py :24-34 quirk]
  → Linear(H, P)
  → Sigmoid
→ output in [0, 1]   (denormalized to physical units in setup_inputs via config.rs bounds)
```

- **F** = number of catchment attributes, **H** = hidden size, **P** = number of learnable routing parameters
- **No inter-block ReLU** — DDR's `kan.py` has none; adding one breaks parity
- `rskan` version as of 2026-07-05: **v0.1.3** (verify with `grep rskan Cargo.toml`)
- All `num_hidden_layers` KanLayers use the **same seed** (a DDR quirk preserved intentionally for parity — see `src/nn/kan_head.rs`)

---

## 6. Operational traps and known weak points

### STALE-BINARY TRAP (high severity — has caused silent wrong results)

`cargo build` and `cargo run` do NOT update `~/.cargo/bin/ddrs`. If you type `ddrs run` after editing `src/`, you silently execute the old binary. The manifest's `git.sha` is stamped from `.git` at runtime, not from the binary, so the run log looks current.

**After any `src/` change, do ONE of:**
```bash
cargo install --path .                              # canonical refresh
# or, faster if target/release is warm:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
# or bypass the installed copy entirely:
cargo run --release --bin ddrs -- run --workflow …
```

**Self-check:** current checkpoints are **directories** (`.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/head.mpk`). A stale pre-checkpoint-resume binary writes flat files (`epoch_E_mb_M.mpk`). Flat files = stale binary.

This trap caused the 2026-07-01 leakance×hourly 2×2 first run: the hourly cell silently ran flat-repeat-24 because the installed binary predated the disaggregation feature.

### CUDA Graphs mask NaN (high severity)

`use_cuda_graphs: true` can return stale finite loss when the forward produces NaN. Always validate new forward-path changes with `use_cuda_graphs: false` before benchmarking with graphs on.

`leakance + use_cuda_graphs: true` is rejected at config load time — this combination is a hard error, not a silent wrong result.

### Checkpoint resume drifts slightly

Checkpoints store weights/moments in **f16** (`CompactRecorder = HalfPrecisionSettings`). A resumed trajectory diverges slowly from an uninterrupted one. Exact state (epoch, mini-batch cursor, RNG permutation) is preserved, but weight precision is not. See `docs/2026-06-07-checkpoint-resume-handoff.md` follow-up #1.

### Fixture regeneration caveat (as of 2026-07-05)

The DDR reference state used to validate V1 lives only in the desktop's `~/projects/ddr` working tree (unpushed `geometry/trapezoidal.py` work). A fixture regenerated from a clean DDR clone diverges ~1% at every ddrs commit — that is a wrong reference, not a port bug. See `.claude/references/ddrs-comparing-to-ddr.md` §Regenerating fixtures.

### Worktree binary path

Fresh worktrees lack the gitignored `output/` and fixture directories. Relative `target/release` resolves to the main tree's stale binary in some shells. Always use absolute paths or `cargo run` in worktrees. See `.claude/memory/ddrs-worktree-gotchas.md`.

---

## 7. Performance numbers (as of 2026-07-05)

### CONUS network

| Metric | Value |
|---|---|
| Reaches (CONUS MERIT) | 346,321 |
| Edges | 338,814 |
| Gauge training set | ~2,365 (CONUS) |

### Summed-Q' baseline (no routing, no learned params)

This is the sanity floor: per-gauge sum of upstream divide Q' over the eval window.

| Metric | Value |
|---|---|
| Median NSE | 0.689 |
| Median KGE | 0.723 |

**If a trained run does not beat NSE 0.689, routing is not earning its keep. Check training loss curves and KAN gradient stats first.**

### Best trained result (as of 2026-07-05)

Precip-driven disaggregation + L1 loss, 2,365 CONUS gauges, eval 2026-06-23:

| Metric | Value |
|---|---|
| Median NSE | 0.715 (+0.037 vs baseline) |
| Median KGE | 0.711 (−0.012 vs baseline) |

**KGE does NOT beat the summed-Q' baseline in any config as of 2026-07-05.** NSE does (+0.037 with precip disagg). The NSE gain is real; the KGE regression traces to over-attenuation of flood peaks (the L1 / NSE gradient rewards the MC solver for attenuating, reducing `α = σ_sim/σ_obs` below 1).

The `nnse-kge` loss mode (`experiment.loss.kind: nnse-kge`) exists to restore the KGE gradient, but no validated CONUS result with this mode is available as of 2026-07-05.

---

## 8. Leakance — experimental GW–SW water-loss term

### What it is

A losing-stream correction subtracted from the routing RHS `b` at each timestep:

```
zeta = leakance_factor · area_z · K_D · (depth − d_gw)
area_z = (p · depth)^q_eps · length      (plan-view wetted area, m²)
b ← b − zeta                             positive zeta = losing reach
```

Implementation: `src/routing/leakance.rs`. Gradient is analytical via `TimestepLeakanceOp: Backward<I,8>`.

### How to enable (three required config changes)

```yaml
params:
  use_leakance: true               # activates term; forces use_cuda_graphs: false
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-5]         # log-space; hydraulic exchange rate, 1/s
    d_gw: [-2.0, 2.0]             # groundwater depth offset, m
    leakance_factor: [0.0, 1.0]   # dimensionless scale

kan_head:
  learnable_parameters: [n, q_spatial, x_storage, K_D, d_gw, leakance_factor]
```

Note: original K_D range was `[1e-8, 1e-6]`. The recoverability experiment (2026-07-04) widened to `[1e-8, 1e-5]` to achieve 58/96 expressible sites (vs 23/96 at the original ceiling). Use `[1e-8, 1e-5]` for any future leakance work.

### Gradient-exactness guard (run after any change to leakance.rs)

```bash
cargo test --test leakance_gradcheck       # analytical ≈ finite-difference (8/8)
cargo test --test leakance_off_parity      # byte-identical to no-leakance when off (3/3)
cargo test --test zeta_accum               # eval zeta == what was subtracted from b (6/6)
cargo run --release --example compare_ddr_sandbox   # must still report ABSOLUTE MATCH
```

### Leakance status summary (as of 2026-07-05)

**2×2 verdict (leakance × forcing, 2026-07-01): GO-marginal.**
- Leakance + hourly: ΔNSE +0.0005, ΔKGE +0.0018 on the losing-stream subset (55.5% of gauges improve)
- Leakance + daily: ΔNSE −0.0017, ΔKGE −0.0009 (35.6% improve — hurts)
- Zeta gate: |zeta| > 0.01 m³/s on 10.4% of 64,892 eval reaches (bar: ≥10%)

**Low-zeta diagnosis (2026-07-02):**

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H1 — K_D box clips flux | REFUTED | 71.5% of reaches CAN exceed 0.01 m³/s inside the current box; median utilization 3.4% |
| H2 — driving-head starvation | SUPPORTED | median head `(depth − d_gw)` = 0.02 m; 47% of reaches gaining at eval-window mean |
| H3 — KAN variance collapse | REFUTED | d_gw–meanP Spearman +0.71; K_D–aridity +0.61 — strong learned structure |
| H4 — gauge bias / gradient starvation | SUPPORTED | zeta–uparea ρ +0.76; gauged median |zeta| 6.7e-3 vs ungauged 5.9e-4; dry/wet ratio 0.40 (inverted from physics) |
| H5 — equifinality (n absorbs loss) | SUPPORTED (daily only) | daily Δn = +0.012 (0.59 IQR, ~20%); hourly Δn nil (0.05 IQR) |
| H6 — wrong yardstick | REFUTED | fractional loss agrees: 8.4% lose >1% of local flow |
| H7 — d_gw model-form error | REFUTED | 0.0% of d_gw at bounds, incl. dry tercile |

**Implication of diagnosis: K_D widening alone is NOT recommended.** The binding constraint is the training signal (H2 + H4), not the parameter box. The pre-registered Phase-3 gate FAILED; the widened-K_D retrain was not run. This supersedes the "widen K_D — top follow-up" recommendation from the 2026-07-01 findings.

**Gradient probe (2026-07-03, worktree `origin/worktree-zeta-sensitivity`):**

| Hypothesis | Bar | Measured | Verdict |
|---|---|---|---|
| P1 — starvation (gradient dead off-gauge) | gauged/ungauged |g| ≥ 10× | 1.5× (trained), 2.9× (cold) | REFUTED |
| P2 — rejection (gradient pushes zeta down) | >67% dry-tercile push-down | 52.5% (≈ neutral) | REFUTED |
| P3 — detectability (real-magnitude loss visible at gauge) | ≥10% of Ref δ=0.01 probes detectable | 4.2% (4/96); delta is 53× smaller than median 5% discharge-uncertainty band | NO-GO |

**Synthetic recoverability positive control (2026-07-04, worktree):**

The positive control FAILED: median recovery ratio = 0.009 (bar: ≥0.5). Root cause: the windowed training objective has a ~130× hotstart-transient noise floor. Continuous residual with teacher weights on teacher obs = 0.0076 mean L1; step-0 windowed training loss = 1.017. The planted signal (0.8% of training loss) is invisible. After 5 epochs, Adam actively degrades the model (continuous residual grew from 0.0076 to 0.4431 — 58× worse than not training).

**Leakance identifiability is NOT proven. The positive control must pass (Phase B objective: windowed training loss ≤ 0.25 mean L1, i.e. ≤10% of a converged run's loss) before any identifiability claim can be made.**

---

## 9. Phase B objective and current state (as of 2026-07-05)

**Phase B goal:** state-cache hotstart — inject continuous-run discharge state at each training window boundary to eliminate the hotstart-transient noise floor.

**Target:** windowed training loss ≤ 0.25 mean L1 (≤10% of a converged run's 1.017 step-0 loss).

**Status:** NOT YET MET as of 2026-07-05. The state-cache infrastructure (`experiment.state_cache`, `src/data/store/obs_writer.rs`, `src/training/forward.rs` injection seam, `--mode state-cache` in probe binary) is implemented in `origin/worktree-zeta-sensitivity` but the floor validation target has not been hit.

**Until Phase B passes, do not claim leakance is learnable from gauge-only supervision.**

---

## 10. CLI quick-reference

### First-time setup

```bash
cargo install --path .      # installs ddrs to ~/.cargo/bin/
ddrs plan                   # GPU probe + smoke test + writes ddrs.yaml (opens $EDITOR)
ddrs run --workflow train-and-test   # train + eval + write manifest
```

### After any src/ change

```bash
cargo install --path .      # or: cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
```

### Key commands

```bash
ddrs sources use conus-hourly        # switch to hourly AORC precip source group
ddrs sources list                    # show active group (* = match)
ddrs show <run-id>                   # inspect run manifest
ddrs status                          # workspace summary + disk usage
ddrs gc --keep 5 --keep-successful   # prune old runs

# Hourly forcing requires BOTH:
#   1. ddrs sources use conus-hourly  (adds aorc_precip path)
#   2. kan_head.disaggregation.use_precip: true in ddrs.yaml
# Without aorc_precip, config with use_precip: true is a hard error.

# Resume from checkpoint:
# Set experiment.checkpoint: .ddrs/runs/<id>/checkpoints/epoch_E_mb_M in ddrs.yaml
# Then raise experiment.epochs above E.
```

### Regression tests (run after touching core routing)

```bash
cargo run --release --example compare_ddr_sandbox           # V1: must print ABSOLUTE MATCH
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox  # V9: CUDA graphs bit-match
cargo test --test mmc                                        # hotstart, coefficients, forward, autodiff
cargo test --test sparse_gradcheck                          # CsrSolveOp backward
cargo test --test sp8_gradcheck                             # fused TimestepOp backward
cargo test --test leakance_gradcheck                        # leakance backward (8/8)
cargo test --test leakance_off_parity                       # byte-identical to no-leakance (3/3)
cargo test --test zeta_accum                                # zeta diagnostic identity (6/6)
```

---

## 11. Workspace layout

| Path | Purpose |
|---|---|
| `ddrs.yaml` | Workflow + experiment config (gitignored) |
| `.ddrs/system.json` | GPU/driver/smoke-test record |
| `.ddrs/sources.lock` | Fingerprints of data_sources paths |
| `.ddrs/adjacency/<key>/` | Cached CONUS + gauges adjacency zarr stores (content-addressed) |
| `.ddrs/baselines/<key>/` | Cached summed-Q' baseline (blake3 of data sources + time window) |
| `.ddrs/runs/<id>/manifest.json` | Per-run manifest (config + sources + git SHA + outputs) |
| `.ddrs/runs/<id>/config.yaml` | Snapshot of the config that produced this run |
| `.ddrs/runs/<id>/run.log` | Timestamped stdout+stderr (fd-level tee) |
| `.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/` | Checkpoint directory: `head.mpk`, `optim.mpk`, `state.json` |
| `.ddrs/runs/<id>/kan_parameters.nc` | Eval-window per-reach zeta/zeta_net/depth_mean/area_z_mean/q_mean |

---

## 12. Data sources summary

| Source | Type | Path (as of 2026-07-05) |
|---|---|---|
| MERIT adjacency | managed zarr (built from fabric) | `.ddrs/adjacency/<key>/` |
| Streamflow Q' (CONUS) | icechunk | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` |
| USGS observations (CONUS) | icechunk | `/mnt/ssd1/data/icechunk/usgs_daily_observations` |
| AORC precip (hourly) | zarr-v3, catchment-major, mm/hr | `/mnt/ssd1/data/aorc/merit_unit_catchments.zarr` |
| Global streamflow Q' | zarr-v2 multi-zone (60 zones) | `/gpfs/hjj5218/data/dmc_forcing/streamflow/zarr/8km/merit_global_v2.7` |
| Global observations | zarr-v2, one array per `Provider__GageId` | `/gpfs/hjj5218/data/dmc_forcing/observation/dMC_global_v3.1` |

Global Q' units: m³/s (confirmed empirically — no units attribute on the zarr). Time axis: CF `days since 1980-01-01`. ~42k fabric reaches lack predictions → 0.001 fill at read.

Hourly AORC precip: zarr-v3, catchment-major (COMID-first), mm/hr, starts 1980-01-01 UTC. Experiment windows using hourly forcing must not reach into 1980 (hourly-lstm store starts 1981-01-01).

---

## 13. Branches

| Branch | Description |
|---|---|
| `master` | Main integration branch |
| `unit_catchments` | Current working branch (as of 2026-07-05) |
| `origin/worktree-zeta-sensitivity` | Most advanced — Phase B state-cache hotstart, gradient probe, recoverability control, unit-catchment attribute wiring |

---

## Provenance and maintenance

Re-verify commands (copy-pasteable, all from project root):

```bash
# V1 invariant
cargo run --release --example compare_ddr_sandbox

# Leakance gradient-exactness suite
cargo test --test leakance_gradcheck && cargo test --test leakance_off_parity && cargo test --test zeta_accum

# KAN head parity
cargo test --features fixtures --test kan_head_init_repro --test kan_head_init_parity --test kan_head_fixture_forward --test kan_head_fixture_backward

# Sparse backward
cargo test --test sparse_gradcheck && cargo test --test sp8_gradcheck

# Check rskan version
grep rskan Cargo.toml

# Check current binary is fresh (flat files = stale)
ls ~/.cargo/bin/ddrs -la && ls .ddrs/runs/ 2>/dev/null | tail -3
```

Ground-truth sources read to produce this skill: `CLAUDE.md`, `.claude/ARCHITECTURE.md`, `.claude/references/ddrs-burn-autograd.md`, `.claude/references/ddrs-architecture.md`, `docs/2026-07-02-leakance-diagnosis-findings.md`, `origin/worktree-zeta-sensitivity:docs/2026-07-03-zeta-gradient-probe-findings.md`, `origin/worktree-zeta-sensitivity:docs/2026-07-04-synthetic-recoverability-findings.md`. Volatile facts dated 2026-07-05. Re-read those files when key numbers or experiment verdicts change.
