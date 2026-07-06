---
name: ddrs-diagnostics-and-tooling
description: >
  Use when you need to MEASURE a ddrs result rather than eyeball it — diagnosing
  a failed run, verifying gradient health, interpreting leakance GO/NO-GO gates,
  checking the summed-Q' baseline, probing identifiability, or deciding whether
  a hypothesis is SUPPORTED/REFUTED/INCONCLUSIVE. Triggers: "why is zeta small",
  "how do I tell if training is working", "what does K_D ceiling mean",
  "compare two runs", "is my gradient alive", "how do I reproduce the 2x2
  verdict". Do NOT use for architecture changes, config authoring, or writing
  new training code — use ddrs-change-control or ddrs-architecture-contract
  instead.
---

# ddrs Diagnostics and Tooling

## Glossary (jargon defined once)

| Term | Definition |
|---|---|
| **ddrs** | BURN-0.21 Rust port of the DDR differentiable Muskingum-Cunge solver |
| **BURN** | Rust deep-learning framework (analogous to PyTorch); autograd tapes differ |
| **KAN head** | `rskan::KanLayer` network (`Linear→KanLayer×N→Linear→Sigmoid`); maps catchment attributes to per-reach routing parameters |
| **Q'** (Q-prime) | Upstream-summed divide streamflow forcing from a pre-computed DHBv2 retrospective |
| **zeta (ζ)** | Per-reach GW–SW exchange flux (m³/s): `leakance_factor · area_z · K_D · (depth − d_gw)`. Positive = losing reach |
| **K_D** | Hydraulic exchange rate (1/s); log-space parameter in `[1e-8, 1e-6]` by default |
| **Muskingum-Cunge (MC)** | Linear flood-routing method; ddrs solves a CSR lower-triangular system per timestep |
| **eval network** | Gauge-subgraph union used during evaluation (64,892 reaches for CONUS) |
| **NSE / KGE** | Nash-Sutcliffe Efficiency / Kling-Gupta Efficiency; standard hydrology skill scores |
| **NNSE** | Normalized NSE: `NSE/(2-NSE)`, range [0,1], avoids -∞ floor |
| **summed-Q' baseline** | Upper bound with no routing: sum of upstream Q' at each gauge; median NSE 0.689 / KGE 0.723 (CONUS, as of 2026-07-05) |
| **2×2** | Leakance ON/OFF × hourly/daily forcing factorial experiment |
| **rho-window** | Training sub-sequence length (default 90 days); sampled from the full training period |
| **hotstart transient** | Initial-condition mismatch at window start; big rivers carry memory >> warmup days |
| **CsrSolveOp** | Hand-written BURN autograd backward for the sparse triangular solve (invariant 4) |
| **dump_parameters** | Binary/CLI command that exports full-CONUS KAN outputs to `kan_parameters.nc` |
| **run-id** | `<UTC_ts>-[<source_group>-]<workflow>` directory name under `.ddrs/runs/` |

---

## When NOT to use this skill

- **Changing `src/routing/`, `src/sparse.rs`, or `src/geometry.rs`** — use ddrs-change-control (blast-radius analysis required)
- **KAN head architecture changes** — use ddrs-architecture-contract (invariants 5-6)
- **Writing new Python analysis scripts** — use ddrs-proof-and-analysis-toolkit
- **Interpreting the research roadmap / phase gating** — use ddrs-identifiability-campaign or ddrs-research-frontier

---

## Part 1: Non-negotiable regression gates

Run these before and after ANY change to `src/routing/`, `src/geometry.rs`, or `src/sparse.rs`.

### 1.1 DDR parity gate (invariant 1)

```bash
cargo run --release --example compare_ddr_sandbox
```

**Pass:** prints `ABSOLUTE MATCH` — max abs diff < 1e-3 m³/s on the 5-reach RAPID sandbox.
**Fail:** any diff >= 1e-3 m³/s means the port broke. Do NOT merge.

**Caveat (as of 2026-06-06):** the reference fixture requires the desktop's `~/projects/ddr` with the unpushed `geometry/trapezoidal.py` changes. A clean DDR clone will diverge ~1%. That is a wrong-reference failure, not a port failure. See `.claude/references/ddrs-comparing-to-ddr.md` §Regenerating fixtures before concluding a real regression.

### 1.2 Leakance gradient-exactness gates

Run whenever `src/routing/leakance.rs` or its backward op changes:

```bash
cargo test --test leakance_gradcheck      # analytical grad ≈ finite-difference (8 cases)
cargo test --test leakance_off_parity     # byte-identical to no-leakance when off (3 cases)
cargo test --test zeta_accum              # accumulated zeta == b_rhs delta (6 cases)
cargo run --release --example compare_ddr_sandbox  # still ABSOLUTE MATCH
```

**Interpretation:** `leakance_gradcheck` failing means the analytical backward diverges from finite-diff; this breaks training correctness. `leakance_off_parity` failing means leakance bleeds into non-leakance paths.

### 1.3 KAN head parity gates

Run when `src/nn/kan_head.rs`, `Cargo.toml` rskan pin, or DDR's `nn/kan.py` changes:

```bash
cargo test --features fixtures \
  --test kan_head_init_repro \
  --test kan_head_init_parity \
  --test kan_head_fixture_forward \
  --test kan_head_fixture_backward
```

### 1.4 Sparse gradcheck

```bash
cargo test --test sparse_gradcheck
```

Verifies the CSR backward (invariant 4) is gradient-exact.

---

## Part 2: Stale binary trap (the most common failure mode)

`~/.cargo/bin/ddrs` is installed once by `cargo install`. `cargo build` does NOT update it.

**Symptom:** run looks fine, metrics make no sense, or a new feature is silently missing.

**Check whether you have the right binary:**

```bash
# Current checkpoints are DIRECTORIES:
ls .ddrs/runs/<run-id>/checkpoints/
# Should show: epoch_5_mb_9/  (a directory)
# If you see: epoch_5_mb_9.mpk  (a flat file) → stale binary
```

**Fix:**

```bash
cargo install --path .                  # canonical, ~2 min
# or faster:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
# or bypass installed binary entirely:
cargo run --release --bin ddrs -- run --workflow train-and-test
```

The stale-binary trap caused the 2026-07-01 2×2 to produce byte-identical hourly and daily cells (the installed binary predated the disaggregation feature).

---

## Part 3: CUDA graphs masking NaN

**Symptom:** training loss is finite and slowly decreasing, but intermediate checks show NaN activations.

**Cause:** `use_cuda_graphs: true` replays a captured graph; a NaN in a subsequent forward returns stale (pre-NaN) finite values. The loss looks healthy while the model is broken.

**Diagnosis:**

```bash
# In ddrs.yaml, temporarily set:
use_cuda_graphs: false
# Then re-run one mini-batch and inspect:
# If loss is NaN → confirmed NaN forward; debug with use_cuda_graphs: false
# If loss is fine → not a NaN issue
```

**Note:** `use_leakance: true` combined with `use_cuda_graphs: true` is rejected at config load time with a hard error.

---

## Part 4: Run inspection and workspace navigation

### 4.1 Check run status and disk

```bash
ddrs status                          # workspace summary + disk usage by run
ddrs show <run-id>                   # full manifest: config, sources, git SHA, metrics
```

### 4.2 Read a run's log

```bash
cat .ddrs/runs/<run-id>/run.log      # timestamped stdout+stderr (fd-level capture)
```

Useful patterns in the log to check:

| What to grep | Meaning |
|---|---|
| `streamflow resolution: Daily\|Hourly` | Confirms whether icechunk store was read as daily or hourly |
| `warm start: loaded KAN head` | Checkpoint resume loaded correctly |
| `no …/optim.mpk` | Adam starts cold (expected for head-only warm-start) |
| `ABSOLUTE MATCH` | Sandbox regression passed during this run |
| `precip loading` | AORC precip store opened (needed for disaggregation) |

### 4.3 Inspect a run's config

```bash
cat .ddrs/runs/<run-id>/config.yaml  # exact config that produced this run
```

### 4.4 Compare two runs' metrics

```bash
ddrs show <run-id-A> | grep -E "nse|kge|loss"
ddrs show <run-id-B> | grep -E "nse|kge|loss"
```

---

## Part 5: Summed-Q' baseline

**What it is:** per-gauge sum of upstream Q' with NO routing or learning. It is the ceiling that trained routing must beat.

**Reference numbers (CONUS, as of 2026-07-05):**
- Median NSE: 0.689
- Median KGE: 0.723

**Best trained result (precip-driven disaggregation + L1, 2365 gauges, as of 2026-06-23):**
- Median NSE: 0.715 (+0.037 vs baseline — beats it)
- Median KGE: 0.711 (-0.012 vs baseline — does NOT beat it)

**Critical:** KGE does NOT beat the summed-Q' baseline in any config as of 2026-07-05. NSE beats it with precip disaggregation. This is a known open problem (over-attenuation; L1 and NSE reward low variance).

**Reproduce baseline:**

```bash
ddrs plan    # computes and caches baseline automatically on first run
# or read the cached version:
cat .ddrs/baselines/<key>/manifest.json   # shows metrics, provenance
```

**Interpretation:** if your trained run's median NSE does NOT beat 0.689, the routing term earns nothing. Debug training loss curves and KAN head gradient stats before touching the sparse solver.

---

## Part 6: dump_parameters — export learned KAN outputs to NetCDF

```bash
# Via the legacy eval binary (required for leakance zeta export):
cargo build --release --bin eval
target/release/eval \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/<run-id>/checkpoints/epoch_5_mb_9 \
  --output /tmp/eval.zarr \
  --zeta-output .ddrs/runs/<run-id>/kan_parameters.nc

# Or (no zeta, just KAN params):
cargo build --release --bin dump_parameters
target/release/dump_parameters \
  --config ddrs.yaml \
  --checkpoint .ddrs/runs/<run-id>/checkpoints/epoch_5_mb_9/head \
  --output .ddrs/runs/<run-id>/plot/kan_parameters.nc
```

**Output file layout (`kan_parameters.nc`):**

| Variable | Dimension | Unit | Notes |
|---|---|---|---|
| `COMID` | `(COMID,)` | — | Reach IDs for full-CONUS params |
| `n` | `(COMID,)` | — | Manning's roughness, denormalized |
| `q_spatial` | `(COMID,)` | — | Channel geometry exponent |
| `x_storage` | `(COMID,)` | — | Muskingum X (storage weighting) |
| `K_D` | `(COMID,)` | 1/s | Hydraulic exchange rate (leakance only) |
| `d_gw` | `(COMID,)` | m | GW depth threshold (leakance only) |
| `leakance_factor` | `(COMID,)` | — | Scale factor (leakance only) |
| `COMID_eval` | `(COMID_eval,)` | — | Eval-network reach IDs (leakance only) |
| `zeta` | `(COMID_eval,)` | m³/s | Mean \|zeta\| over eval window |
| `zeta_net` | `(COMID_eval,)` | m³/s | Signed mean; positive = losing reach |
| `depth_mean` | `(COMID_eval,)` | m | Eval-window mean routed depth |
| `area_z_mean` | `(COMID_eval,)` | m² | Eval-window mean plan-view wetted area |
| `q_mean` | `(COMID_eval,)` | m³/s | Eval-window mean routed discharge |

**Load in Python:**

```python
import xarray as xr
ds = xr.open_dataset(".ddrs/runs/<run-id>/kan_parameters.nc")
# for a quick K_D ceiling check:
kd = ds["K_D"].values
import numpy as np
print(f"K_D: min={kd.min():.2e}  median={np.median(kd):.2e}  max={kd.max():.2e}")
print(f"fraction at ceiling (1e-6): {(kd > 9.9e-7).mean():.1%}")
```

---

## Part 7: Leakance GO/NO-GO evaluation

### 7.1 The three gate criteria (per spec)

| Gate | Threshold | Interpretation |
|---|---|---|
| 1 | ΔNSE or ΔKGE > 0 (median) on losing-stream subset, hourly arm | Leakance improves skill where physics predicts it should |
| 2 | Effect absent or weaker in daily arm | Rules out fudge-factor behavior |
| 3 | \|zeta\| > 0.01 m³/s on ≥ 10% of eval reaches | Learned exchange is non-trivially active |

**Current status (as of 2026-07-01):** GO — but marginal (10.4% vs 10% threshold for gate 3, no headroom).

### 7.2 Running the full verdict script

```bash
cd ~/projects/ddr
uv run python ~/projects/ddrs/scripts/leakance_subset_analysis.py \
  --hourly-on  2026-07-01T13-43-32Z-train-and-test \
  --daily-on   2026-07-01T21-20-27Z-train-and-test \
  --hourly-off 2026-06-23T02-49-12Z-conus-hourly-train-and-test \
  --daily-off  2026-06-05T01-41-16Z-train-and-test \
  --ddrs-runs-dir /home/tbindas/projects/ddrs/.ddrs/runs
```

**Prerequisites:** both ON arms must have `kan_parameters.nc` with the `zeta`/`COMID_eval` variables (produced by `--zeta-output` or `train-and-test` Phase 2).

**Output block to look for:**

```
VERDICT: GO
  Leakance improves skill on the losing-stream subset under hourly forcing ...
```

or `VERDICT: NO-GO` with reasons, or `VERDICT: NEEDS_ZETA_EXPORT`.

**Losing-stream subset definition:** gauges where the summed-Q' baseline mean(pred)/mean(obs) > 1 on the hourly-OFF run. CONUS result: 1883/2365 gauges (79.6%).

### 7.3 Interpreting leakance parameter outputs

| Observation | Interpretation |
|---|---|
| `K_D` 100% at ceiling (1e-6) | Optimizer wants MORE exchange; box is binding. NOT a model failure (H1 REFUTED, as of 2026-07-02) |
| `leakance_factor` interior (≈0.33) | Gate is open; reaches are actively exchanging |
| `d_gw` near mean depth | Driving head throttled; ~47% of reaches gaining at eval-window mean |
| zeta–uparea Spearman +0.76 | Exchange tracks river size, not aridity — gauge bias, not starvation |
| dry-tercile zeta < wet-tercile | Inverse of physics; training signal concentrates near large gauged rivers |

---

## Part 8: Seven-hypothesis diagnosis battery (leakance low-zeta)

**When to run:** after any leakance experiment returns small zeta (median |zeta| < 0.01 m³/s).

**Prereqs:** the ON run's `kan_parameters.nc` must contain `depth_mean`, `area_z_mean`, `q_mean` on `COMID_eval`. Requires the re-eval pass with the current binary.

```bash
cd ~/projects/ddr
uv run python ~/projects/ddrs/scripts/leakance_diagnosis.py
# uses hardcoded run IDs in ARM_IDS dict; edit if using different runs
```

**Hypothesis reference table (results as of 2026-07-02):**

| # | Hypothesis | Verdict (2026-07-02) | Key number |
|---|---|---|---|
| H1 | K_D box clips zeta below detectability | REFUTED | 71.5% of reaches CAN exceed 0.01 m³/s in-box; median utilization 3.4% |
| H2 | d_gw near depth → driving head ≈ 0 | SUPPORTED | 57.6% of reaches < 0.1 m mean driving head; 47.0% ≤ 0 |
| H3 | KAN variance collapse (original hypothesis) | REFUTED | K_D–aridity ρ = +0.61; d_gw–meanP ρ = +0.71 — strong spatial structure |
| H4 | Gauge bias / gradient starvation | SUPPORTED | gauged median |zeta| 6.7e-3 vs ungauged 5.9e-4 (11×); dry/wet ratio 0.40 (inverse of physics) |
| H5 | Equifinality with n/x_storage | SUPPORTED (daily only) | daily Δn = +0.012 (0.59 IQR); hourly Δn nil (0.05 IQR) |
| H6 | Wrong yardstick (absolute 0.01 bar) | REFUTED | 8.4% >1% fractional loss agrees with absolute bar |
| H7 | d_gw boundary pinning (disconnected regime) | REFUTED | 0.0% of d_gw at bounds |

**Diagnosis conclusion:** zeta is small because the optimizer throttles through the driving head (H2) and the gradient only reaches gauged large rivers (H4), not because the K_D box or KAN architecture fails. Widening K_D past 1e-6 is NOT recommended (supersedes the "top follow-up" in `docs/2026-07-01-leakance-hourly-findings.md`).

---

## Part 9: Gradient probe (adjoint reachability + detectability)

**Location:** `origin/worktree-zeta-sensitivity` branch.
**When to run:** when you want to know whether the leakance gradient is alive at a reach, or whether a real-magnitude loss would be detectable at a downstream gauge.

### 9.1 Stage 1 — adjoint reachability map

```bash
# Trained checkpoint (use worktree binary):
WT=/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --windows 96 --seed 42 \
  --output output/zeta_probe/grad_trained.nc

# Cold head (omit --checkpoint):
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --windows 96 --seed 42 \
  --output output/zeta_probe/grad_cold.nc
```

**Output:** per-reach `|∂L/∂factor|`, `∂L/∂factor`, coverage count in NetCDF.

**Interpretation thresholds:**

| Ratio (gauged/ungauged |g|) | Interpretation |
|---|---|
| ≥ 10× at both trained and cold points | SUPPORTED starvation — auxiliary supervision fills genuine gap |
| < 10× | REFUTED starvation — gradient reaches everywhere |

**Measured result (2026-07-03, as of 2026-07-05):** gauged/ungauged ratio = 1.5× (trained), 2.9× (cold). P1 starvation REFUTED. The gradient is alive everywhere.

### 9.2 Stage 2 — planted-delta detectability

```bash
# Plan sites first (ddrs-py venv):
cd ddrs-py && uv run python ../scripts/zeta_probe_sites.py

# Perturb pass:
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --mode perturb \
  --probe-plan output/zeta_probe/probe_plan.csv \
  --eval-days 1095 \
  --output output/zeta_probe/perturb
```

**Detectability criterion:** `|mean ΔQ| > 99th-pct noise floor AND > 5% of gauge's mean flow`.

**Measured result (2026-07-03, as of 2026-07-05):**

| Delta | Reference (Ref) gauges | Non-reference |
|---|---|---|
| 0.01 m³/s (literature-magnitude) | 4.2% detectable | 0.0% |
| 0.1 m³/s (upper-literature) | 16.7% detectable | 2.1% |

P3 detectability: NO-GO. The planted loss arrives at gauges at ~95% fidelity (transmission fine) but is 53× smaller than the median Ref gauge's 5% discharge-uncertainty band. Detection fails on dilution, not transmission. No gauge-only objective can learn real-world leakance.

---

## Part 10: Synthetic recoverability control (Phase B)

**Location:** `origin/worktree-zeta-sensitivity` branch.
**Status (as of 2026-07-05):** FAILED — positive control not passed.

**One-line result:** recovery ratio median 0.009 (bar: ≥ 0.5). Root cause: the windowed training objective (rho-90, warmup-5) has a ~130× hotstart-transient noise floor relative to the planted signal. The signal is invisible even with zero observation noise and warm-started weights.

**Phase B objective:** reduce the noise floor to ≤ 0.25 mean L1 (≤ 10% of a converged run). This is NOT YET MET as of 2026-07-05. Required before any identifiability claim for leakance.

**Key decomposition:**

| Quantity | Value |
|---|---|
| Planted signal (continuous residual, teacher weights + teacher obs) | 0.0076 mean L1 |
| Step-0 windowed training loss (warm-started run A) | 1.017 mean L1 |
| Noise floor / signal ratio | ~130× |
| Run A continuous residual after 5 epochs of training | 0.4431 (58× worse than start) |

**Implication for leakance identifiability:** gauge-loss training cannot reward reach-scale leakance even with: detectable gauge signal (constructed), zero obs noise, expressible head, and warm-start from the answer. Leakance identifiability is NOT proven. Phase B (state-cache hotstart, ≤ 0.25 mean L1 target) is required before any identifiability claim.

**Verdicts from the control run:**

| # | Metric | Measured | Verdict |
|---|---|---|---|
| R1 | Recovery ratio median | 0.009 | FAILED (bar: ≥ 0.5) |
| R2 | Non-planted \|zeta_net\| A/baseline | 1.11 | PRECISE — trivially, nothing moved |
| R3 | Final-epoch loss A vs B (42.2% gap) | A < B | CONFOUNDED — B's handicap accounts for gap |
| R4 | Manning's n shift (run B) | Δn = −0.019 (global, not localized) | H5 equifinality confirmed at global scale |
| R5 | Cold emergence ratio | 1.20 | SUPPRESSED (bar: > 3) |

---

## Part 11: Leakance configuration checklist

Three config changes are ALL required together to enable leakance. Missing any one causes silent failure or a config-load error.

```yaml
# 1. Activate the term (also disables CUDA graphs):
params:
  use_leakance: true

# 2. Tell the KAN head to emit leakance parameters:
kan_head:
  learnable_parameters:
    - K_D
    - d_gw
    - leakance_factor
    - n            # keep existing routing params
    - q_spatial
    # x_storage    # optional

# 3. Set parameter ranges:
params:
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-6]        # log-space; 1/s
    d_gw: [-2.0, 2.0]            # m
    leakance_factor: [0.0, 1.0]  # dimensionless
```

**Important:** `use_leakance: true` AND `use_cuda_graphs: true` is rejected at config load. The combination is not supported without a separate capture path.

---

## Part 12: Training monitoring checklist

Use this list when a run produces unexpected metrics.

- [ ] **Check the binary is current** — flat checkpoint files mean stale binary (Part 2)
- [ ] **Check `streamflow resolution` in `run.log`** — `Daily` vs `Hourly` must match your intent
- [ ] **Check precip loaded** — if `use_precip: true` in disagg block, grep `precip loading` in run.log
- [ ] **Disable CUDA graphs if loss is suspiciously smooth** — guards against NaN masking (Part 3)
- [ ] **Check `kan_parameters.nc` K_D ceiling fraction** — 100% at ceiling means the box is binding
- [ ] **Run `leakance_subset_analysis.py`** for GO/NO-GO after any leakance experiment (Part 7.2)
- [ ] **Compare against baseline** — trained NSE should exceed 0.689; KGE may not (Part 5)
- [ ] **Check `ddrs show <run-id>`** for final metrics in the manifest
- [ ] **Verify checkpoint format** — directory `epoch_E_mb_M/` with `head.mpk`, `optim.mpk`, `state.json`

---

## Part 13: Run ID and workspace layout quick reference

```
.ddrs/
  system.json                    # GPU/driver probe result
  sources.lock                   # fingerprints of data_sources paths
  adjacency/<key>/               # managed CONUS + gauge adjacency (content-addressed)
  baselines/<key>/               # summed-Q' baseline cache
    manifest.json                # metrics + gage_ids
    predictions.f32              # row-major [n_gauges, n_days]
    observations.f32
  runs/<run-id>/
    manifest.json                # config + sources + git SHA + output metrics
    config.yaml                  # exact config snapshot
    run.log                      # timestamped stdout+stderr
    checkpoints/
      epoch_E_mb_M/              # DIRECTORY (flat .mpk = stale binary)
        head.mpk
        optim.mpk
        state.json
    eval/
      predictions.zarr/          # zarr-v3 group
        predictions/             # float64 [n_gauges, n_days]
        observations/
        gage_ids/                # uint8 [n_gauges, 8] fixed-width ASCII
    baseline/                    # copy of .ddrs/baselines/<key>/
    kan_parameters.nc            # KAN outputs + zeta (leakance) or plot params
    plot/
      kan_parameters.nc          # full-CONUS dump_parameters output
```

---

## Part 14: Zeta accumulator — what it measures and how to verify

The zeta accumulator is enabled automatically during eval when `use_leakance: true`. It recomputes per-step zeta from the SAME saved primitives the backward reads, then accumulates per-reach means over the eval window.

**Correctness identity (from `tests/zeta_accum.rs`):**

```
q_no_leak[0] − q_leak[0] == zeta[0]     (headwater reach; exact equality)
```

**Verify the identity is preserved:**

```bash
cargo test --test zeta_accum
```

**Check zeta is non-trivial after a leakance run:**

```python
import xarray as xr, numpy as np
ds = xr.open_dataset(".ddrs/runs/<run-id>/kan_parameters.nc")
z = np.abs(ds["zeta"].values)
print(f"median |zeta|: {np.median(z):.4e} m3/s")
print(f"|zeta| > 0.01 on {np.mean(z > 0.01):.1%} of eval reaches")
# CONUS target (as of 2026-07-01): 10.4% of 64,892 reaches
```

**Zeta dimensionality:** `COMID_eval` dimension = gauge-subgraph union = eval network (64,892 reaches for CONUS). NOT the full 346,321-reach CONUS.

---

## Provenance and maintenance

```bash
# Re-verify Part 1 regression gates:
cargo run --release --example compare_ddr_sandbox       # ABSOLUTE MATCH
cargo test --test leakance_gradcheck                     # 8/8
cargo test --test zeta_accum                             # 6/6
cargo test --test sparse_gradcheck                       # pass

# Re-verify leakance 2x2 results:
cd ~/projects/ddr && uv run python \
  ~/projects/ddrs/scripts/leakance_subset_analysis.py \
  --hourly-on  2026-07-01T13-43-32Z-train-and-test \
  --daily-on   2026-07-01T21-20-27Z-train-and-test \
  --hourly-off 2026-06-23T02-49-12Z-conus-hourly-train-and-test \
  --daily-off  2026-06-05T01-41-16Z-train-and-test \
  --ddrs-runs-dir /home/tbindas/projects/ddrs/.ddrs/runs

# Re-verify 7-hypothesis diagnosis:
cd ~/projects/ddr && uv run python \
  ~/projects/ddrs/scripts/leakance_diagnosis.py

# Source files for this skill:
# /home/tbindas/projects/ddrs/CLAUDE.md
# /home/tbindas/projects/ddrs/scripts/leakance_subset_analysis.py
# /home/tbindas/projects/ddrs/scripts/leakance_diagnosis.py
# /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-findings.md
# /home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md
# origin/worktree-zeta-sensitivity:docs/2026-07-03-zeta-gradient-probe-findings.md
# origin/worktree-zeta-sensitivity:docs/2026-07-04-synthetic-recoverability-findings.md
# Skill last verified: 2026-07-05
# Volatile facts: summed-Q baseline metrics, leakance GO/NO-GO verdict,
#   recoverability Phase B status — re-verify after any new experiment run
```
