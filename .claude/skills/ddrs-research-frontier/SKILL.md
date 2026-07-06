---
name: ddrs-research-frontier
description: "Use when planning or executing novel research extensions to ddrs — specifically: writing or advancing the selective equifinality paper, diagnosing why leakance identifiability is blocked and designing the next experiment, capturing the CUDA Graphs backward pass, or scaling ddrs to global MERIT routing. Also use when a collaborator asks where ddrs can advance SOTA relative to existing differentiable routing work. Do NOT use for day-to-day training runs, bug fixes, config changes, or reproducing existing experiments — use the sibling skills ddrs-identifiability-campaign or ddrs-eval-plots for those."
---

# ddrs Research Frontier

**Audience:** Mid-level ML engineer who knows PyTorch but not Rust/BURN. Every domain term is defined on first use.

## Glossary (read once, then skip)

| Term | Meaning |
|---|---|
| ddrs | BURN-0.21 Rust port of DDR (Python/PyTorch); routes streamflow through river networks using Muskingum-Cunge (MC) physics |
| BURN | A Rust deep-learning framework (like PyTorch but compiled Rust). BURN 0.21 is pinned. |
| Muskingum-Cunge (MC) | A linear wave-routing equation: Q_{t+1} = C1*I_{t+1} + C2*I_{t} + C3*Q_t + C4*q'. It is the physics kernel. |
| KAN head | Kolmogorov-Arnold Network; maps catchment attributes to physical routing parameters (Manning's n, channel geometry). Implemented via `rskan::KanLayer`. |
| MERIT | A global 90m DEM-derived river network. CONUS subset: 346,321 reaches. Global: 2,939,408 reaches. |
| Q' (Q prime) | Lateral inflow: the runoff prediction from an upstream land-surface model that enters each reach. ddrs learns to route it, not predict it. |
| zeta | GW-SW (groundwater-surface water) exchange flux, m³/s. Positive = losing reach. `zeta = leakance_factor * area_z * K_D * (depth - d_gw)` |
| NSE | Nash-Sutcliffe efficiency. Range (-inf, 1]. Higher is better. |
| KGE | Kling-Gupta efficiency. Range (-inf, 1]. Penalizes variance ratio separately from correlation. |
| sparse backward | The hand-written O(nnz) autograd backward for the triangular-solve step, in `src/sparse/`. Do NOT replace with tape unrolling. |
| CUDA Graphs | NVIDIA feature: capture a sequence of GPU kernels once, replay as a single host call. Eliminates per-kernel launch overhead (~2.3 us/call). |
| rho window | Training window length in days (typically 90). Each mini-batch samples rho-length windows from the full time series. |
| hotstart transient | When a rho-window begins, river storage is initialized heuristically (not from a converged state), causing a warmup artifact. |

---

## When NOT to use this skill

- **Reproducing an existing run:** use `ddrs run --workflow train-and-test`.
- **Debugging a training crash or NaN loss:** see `.claude/references/ddrs-burn-autograd.md` + the CUDA-graphs-mask-NaN memory note.
- **Plotting eval results:** use the `ddrs-eval-plots` skill.
- **Leakance identifiability campaign (Phase B):** use the `ddrs-identifiability-campaign` skill, which tracks the ongoing experimental state.

---

## Current SOTA Baseline (as of 2026-07-05)

These are the numbers any new work must beat or explain.

| Config | Median NSE | Median KGE | Notes |
|---|---|---|---|
| Summed-Q' (no routing, no learning) | 0.689 | 0.723 | The baseline ddrs must beat |
| Best ddrs result: precip disagg + L1 | 0.715 | 0.711 | 2365 gauges, CONUS, run 2026-06-23 |

**Critical:** KGE does NOT beat the summed-Q' baseline in any ddrs config as of 2026-07-05. NSE beats it by +0.037 with precip disaggregation. Any paper claim about routing skill must acknowledge this.

---

## Problem 1: Selective Equifinality Paper

### Why current SOTA fails

Differentiable routing papers (DDR, MC-LSTM, HydroNets) report physically plausible learned parameters but cannot answer the equifinality critique: "many parameter sets produce equivalent outputs — are your learned geometries real or compensatory?" No standard protocol exists to distinguish identifiable parameters from bias-absorbers. Every paper is vulnerable.

### The ddrs asset

ddrs can run the same MC solver with four structurally different lateral-inflow sources (two LSTMs, two dHBV2 variants) on the identical MERIT network, attributes, and observations. Only Q' changes. Parameters that converge across sources are identifiable; parameters that diverge are compensatory. The paper claim (from `/home/tbindas/projects/ddr_equifinality/paper.tex`):

> **Channel geometry is identifiable. Manning's n is a bias-absorber** — it shifts peak timing to compensate for upstream model error, not to represent physical roughness.

This is the "selective equifinality" thesis. Three levels of comparison: raw parameters (p, q spatial), realized geometry at reference discharge (depth, top width, hydraulic radius), and routing performance.

### First 3 steps

1. **Set up the four-arm training matrix.** Four configs, each differing only in the `streamflow:` data-source path:

   ```bash
   # Verify the four Q' stores are reachable:
   ls /mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic   # dHBV2-UH (CONUS daily)
   ls /gpfs/hjj5218/data/dmc_forcing/streamflow/zarr/8km/merit_global_v2.7  # global daily
   # LSTM stores: check docs or ask Tadd — paths are not yet confirmed in CLAUDE.md
   ```

   Each arm uses `config/merit_training.yaml` as the template, changing only `data_sources.streamflow`. Run all four:

   ```bash
   ddrs sources use conus && ddrs run --workflow train-and-test  # arm 1
   # repeat for each inflow source
   ```

2. **Export realized geometry for all four arms.** After training, run `dump_parameters` on each checkpoint to get the full CONUS parameter field, then compute depth and top width at a reference discharge (e.g., mean annual flow from the Q' store):

   ```bash
   cargo build --release --bin dump_parameters
   target/release/dump_parameters \
     --config config/merit_training.yaml \
     --checkpoint .ddrs/runs/<arm_id>/checkpoints/epoch_5_mb_35/head \
     --output output/equifinality/params_arm1.nc
   ```

   Then compute realized geometry in Python (`ddrs-py` venv):

   ```bash
   cd /home/tbindas/projects/ddrs/ddrs-py && uv run python scripts/compute_realized_geometry.py \
     --params output/equifinality/params_arm1.nc \
     --qref <mean_annual_q_nc>
   ```

3. **Convergence test.** For each parameter and realized-geometry quantity, compute the across-arm coefficient of variation (CV = std/mean, per COMID) and the Spearman rank correlation between arm pairs. Parameters with low CV (< 0.2) and high pairwise correlation (Spearman > 0.8) across all four arms are identifiable. Manning's n is expected to show high CV and low correlation (the hypothesis).

### Falsifiable milestone

**Milestone:** Compute across-arm Spearman correlation for Manning's n vs realized depth at 1000 randomly sampled CONUS reaches. If Manning's n correlation < 0.4 while depth correlation > 0.7, the selective equifinality claim is supported and the paper has its central result.

---

## Problem 2: Leakance Under a Fixed Training Objective

### Why current SOTA fails

Existing differentiable routing models (including DDR master) have no GW-SW exchange term. Where losing streams exist (arid West, fractured aquifer systems), the routing error is systematic and physically attributable. Leakance has been implemented in ddrs (`src/routing/leakance.rs`), the 2x2 experiment (2026-07-01) cleared the GO gate, and the diagnosis (2026-07-02) identified the root causes of the marginal result. However, the identifiability positive control (2026-07-04) FAILED.

### Current experimental state (as of 2026-07-05)

**The 2x2 verdict (SUPPORTED):** leakance under hourly forcing improves KGE +0.0046 globally and +0.0018 on the losing-stream subset (55.5% of gauges improve). Under daily forcing it degrades skill. Hourly is a precondition.

**Diagnosis hypotheses (2026-07-02):**

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H1: K_D box clips flux | REFUTED | 71.5% of reaches can exceed 0.01 m³/s inside the current box; median utilization 3.4% |
| H2: driving-head starvation | SUPPORTED | 47.0% of reaches have d_gw >= depth (gaining/neutral); median driving head 0.021 m |
| H3: KAN variance collapse | REFUTED | K_D-aridity Spearman +0.61, d_gw-meanP +0.71 — strong learned spatial structure |
| H4: gauge bias / gradient starvation | SUPPORTED | gauged median zeta 6.7e-3 vs ungauged 5.9e-4; dry-tercile zeta 2.5x LESS than wet (inverted from physics) |
| H5: equifinality with routing params | SUPPORTED (daily only) | daily Δn = +0.012 (0.59 IQR) when leakance ON; hourly Δn negligible |
| H6: wrong yardstick | REFUTED | fractional loss agrees: 8.4% lose >1% of local flow |
| H7: model-form error (d_gw pinning) | REFUTED | 0.0% of d_gw at bounds in dry tercile |

**K_D widening is NOT recommended.** The diagnosis showed the binding constraint is the training signal, not the box. Widening re-pins K_D at the new ceiling with no skill change. This supersedes the "widen K_D" recommendation in `docs/2026-07-01-leakance-hourly-findings.md` §5 item 2.

**Gradient probe (2026-07-03):**
- P1 starvation: REFUTED — gauged/ungauged gradient ratio 1.5-2.9 (not 10x). Gradient is alive everywhere.
- P3 detectability: NO-GO — a real-magnitude 0.01 m³/s reach loss arrives at the measurement gauge at 95% fidelity but is 53x smaller than the median reference gauge's 5% discharge-uncertainty band. Only 4.2% of reference-quality probe sites are detectable. **Gauge-only discharge supervision cannot see real-world leakance.**

**Synthetic recoverability (2026-07-04): FAILED.**
- Recovery ratio R1 = 0.009 (bar: >= 0.5). Nothing recovered.
- Root cause: the rho-90/warmup-5 windowed training objective has a ~130x hotstart-transient noise floor relative to the planted signal (step-0 loss 1.017 vs continuous residual 0.0076). The planted signal is 0.8% of the training loss — invisible. Adam actively degrades the model (continuous residual after 5 training epochs: 0.4431, vs 0.0076 before training).
- This is a GENERAL ddrs finding: warmup=5 under-trims hotstart transients by ~2 orders of magnitude.

**Implication: leakance identifiability is NOT proven.** Phase B (state-cache hotstart, target <= 0.25 mean L1 noise floor) is mandatory before any identifiability claim.

### The ddrs asset

- `src/routing/leakance.rs` with gradient-exact `TimestepLeakanceOp: Backward<I,8>`.
- Eval-time zeta accumulator (`MuskingumCunge::enable_zeta_accumulation`) exports `zeta`/`zeta_net`/`depth_mean`/`area_z_mean`/`q_mean` to `<run_dir>/kan_parameters.nc`.
- `LeakanceOverride` eval-path seam in `src/training/forward.rs` for synthetic experiments.
- Zarr-v2 synthetic-obs writer in `src/data/store/obs_writer.rs`.
- `--backend cpu` flag for deterministic CPU runs.

### First 3 steps for Phase B

1. **Warmup/transient floor curve (cheap, forward-only).** Quantify the windowed loss floor vs warmup length. No training required — just evaluate teacher weights at different warmup lengths:

   ```bash
   # In the worktree:
   WT=/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity
   cd /home/tbindas/projects/ddrs
   for warmup in 5 15 30 60; do
     cargo run --release --bin probe_zeta_gradient -- \
       --mode grad --backend cpu \
       --config $WT/config/experiments/leakance_hourly_on.yaml \
       --checkpoint .ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
       --warmup $warmup \
       --output output/phase_b/floor_warmup${warmup}.nc
   done
   ```

   Target: find the warmup length at which step-0 windowed loss <= 0.25 mean L1. That is the Phase B noise-floor gate.

2. **Implement state-cache hotstart.** Persistent routing state (Q_t at the end of each window) must be carried forward across training windows so the hotstart-transient is eliminated. This is a `src/training/` change. Design constraint: must not break invariant 4 (sparse backward) or cause memory OOM on CONUS (346,321 reaches x f32 = 1.3 MB per state — negligible).

3. **Re-run the positive control with state-cache.** After step-2, repeat the synthetic recoverability experiment from `docs/2026-07-04-synthetic-recoverability-findings.md` §8. The target for Phase B passage is R1 >= 0.5 (recovery ratio) AND the step-0 windowed loss <= 0.25 mean L1.

### Falsifiable milestone

**Phase B milestone:** windowed training loss at warmup=W drops to <= 0.25 mean L1 (the threshold from `docs/2026-07-04-synthetic-recoverability-findings.md`). Only then re-run the synthetic recoverability control and check R1 >= 0.5.

**Auxiliary-constraint milestone (Phase C, conditional on Phase B):** train with a spatial regularizer on `d_gw` or `zeta_net` against an independent losing-potential signal (Jasechko 2021 well-vs-stream levels, or water-table-depth attributes). The aux loss must act directly on head outputs, not through routed discharge — see conclusions #3 in `docs/2026-07-04-synthetic-recoverability-findings.md`.

### Config checklist for any leakance run

```yaml
params:
  use_leakance: true
  use_cuda_graphs: false  # REQUIRED — leakance + CUDA graphs is rejected at config load
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-5]   # widened from original 1e-6 ceiling (needed for site expressibility)
    d_gw: [-2.0, 2.0]
    leakance_factor: [0.0, 1.0]
kan_head:
  learnable_parameters: [..., K_D, d_gw, leakance_factor]
  disaggregation:
    use_precip: true  # REQUIRED — daily forcing degrades leakance (H5 equifinality)
data_sources:
  aorc_precip: /mnt/ssd1/data/aorc/merit_unit_catchments.zarr
```

---

## Problem 3: Backward CUDA Graphs (SP-11)

### Why current SOTA fails

Differentiable routing at CONUS scale (346,321 reaches) must run the full MC solver + autograd for thousands of timesteps per training step. Each BURN tensor op issues a `cuLaunchKernel` call (~2.3 us each). At 8M+ launches per training step, launch overhead dominates wall time. The forward pass was fixed by SP-10; the backward pass was not.

### Current state (as of 2026-07-05)

SP-10 (CUDA Graphs, forward-only) achieved:
- V9 (graph vs no-graph bit-match): GREEN — ABSOLUTE MATCH at f32 precision floor
- V7a (cuda/cpu wall-time ratio <= 0.7): GREEN — 0.385 (was 0.919 before SP-10)
- V10 (cuLaunchKernel drop >= 40%): PARTIAL — 29.2% (target: 40%)

The backward path still uses SP-9 direct-launch kernels. Roughly half the per-step launches are in the backward, which is why V10 missed its target. Closing the backward gap would push V10 above 50% and likely V7a below 0.3.

The critical blocker: leakance + CUDA graphs is **rejected at config load** (`src/config.rs`). Any backward-graph work must either keep leakance as a separate capture path or build a unified capture that handles the leakance kernel.

### The ddrs asset

- Working forward CUDA Graph capture at `src/cuda_graph/` using fused cubecl kernels (K1: geometry+MC coefficients, K2: RHS assembly, K3: post-solve clamp).
- cubecl fork with `flush_no_sync` patch (`taddyb/cubecl`, branch `ddrs-sp7-stream-accessor`, commit `d562ab99`).
- SP-10's seven-layer fix history is documented in `.claude/ARCHITECTURE.md` §SP-10. Read this before attempting the backward capture — all seven failure modes will recur.

### First 3 steps

1. **Profile the backward to identify the top-5 kernel launches.** Run `nsys profile` on a smoke train with CUDA graphs disabled, filter to backward-only kernels:

   ```bash
   nsys profile --trace=cuda,nvtx -o output/sp11_backward_profile \
     cargo run --release --example benchmark_hydrograph
   nsys stats output/sp11_backward_profile.nsys-rep \
     --report gputrace --format csv | grep -v forward | sort -t, -k5 -nr | head -20
   ```

2. **Write fused backward cubecl kernels for the geometry gradient.** The SP-10 forward fuses S1-S28 into K1/K2/K3. The backward needs equivalent fused kernels for `∂L/∂n`, `∂L/∂p`, `∂L/∂q` (all derivable from the saved 24-intermediate struct in `mmc_op.rs`). The gradients are defined in `src/routing/mmc_op.rs` (the analytical backward). Implement as `src/cuda_graph/backward_kernels.rs`, mirroring the forward kernel structure.

3. **Capture the backward graph.** Follow the same seven-step capture discipline from SP-10 (see `.claude/ARCHITECTURE.md` §SP-10). Key constraint: the backward graph must share the same cubecl server context binding as the forward graph, or the exclusive-lock deadlock (layer 2) recurs. Verify with:

   ```bash
   cargo test --test sparse_gradcheck  # must pass — backward correctness
   cargo run --release --example compare_ddr_sandbox  # must report ABSOLUTE MATCH
   ```

### Falsifiable milestone

**V10 milestone:** `cuLaunchKernel` count drops >= 40% relative to SP-9 baseline (target: from 7,684,365 to <= 4,610,619). Verify with `nsys profile` on the smoke train.

**V7a stretch:** cuda/cpu wall-time ratio <= 0.3 (currently 0.385 with forward-only graphs).

---

## Problem 4: Global Routing

### Why current SOTA fails

All published differentiable routing results are CONUS-only (346,321 reaches, ~2,365 gauges). MERIT covers 2,939,408 global reaches with 6,051 gauges from 25+ providers. No differentiable routing model has been trained or evaluated at global scale with gradient-based parameter learning.

### The ddrs asset

Infrastructure is already in place (as of 2026-07-05):

| Component | Status | Path |
|---|---|---|
| Global fabric | Available | `/projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg` (2,939,408 flowlines) |
| Global Q' forcing | Available | `/gpfs/hjj5218/data/dmc_forcing/streamflow/zarr/8km/merit_global_v2.7` (60 pfaf-2 zones) |
| Global observations | Available | `/gpfs/hjj5218/data/dmc_forcing/observation/dMC_global_v3.1` (6,051 gages, 25+ providers) |
| Global gage metadata | Available | `/gpfs/hjj5218/data/dmc_forcing/gage_information/formatted_gage_csvs/v3.1/8km/` (57 per-zone CSVs) |
| Global attributes | Available | `~/projects/ddr/data/merit_global_attributes_v2.nc` (2,939,404 COMIDs) |
| Managed adjacency build | Implemented | `ddrs plan` builds from gpkg in ~25 s; content-addressed cache |
| Global source group | Implemented | `config/sources/global.yaml` ships in-repo |

Switch to global in one command:

```bash
ddrs sources use global && ddrs plan --workflow train-and-test
```

### First 3 steps

1. **Smoke test the global adjacency build.** Run `ddrs plan` with the global source group on a machine with access to PSC storage (`/gpfs/`). Verify the managed adjacency build completes and the zarr stores pass the topological-order invariant:

   ```bash
   ddrs sources use global
   ddrs plan --workflow train-and-test
   # Expected: "adjacency build: 2,939,408 reaches, 338,814+ edges" and
   # "ABSOLUTE MATCH" on the sandbox (parity unaffected by network scale)
   cargo test --test adjacency_parity  # byte-identical to engine-built store
   ```

2. **First global train: 1-epoch smoke.** Set `experiment.epochs: 1` and `experiment.mini_batches: 5` to verify data loading, loss computation, and checkpoint writing work at global scale without OOM:

   ```bash
   ddrs --config config/merit_training.yaml run --workflow train
   # Watch for: CUDA OOM (batch_size may need reduction for 2.9M reaches),
   # NaN loss (use_cuda_graphs: false for first global run — see CUDA-graphs-mask-NaN memory note),
   # missing STAID resolution (global obs provider__gageid format)
   ```

3. **Eval against global observations.** After a converged global train (3-5 epochs), run eval against the global observation store and compare per-provider median NSE/KGE:

   ```bash
   ddrs run --workflow eval
   # Compare against the summed-Q' global baseline (ddrs plan computes this automatically)
   ```

### Falsifiable milestone

**Smoke milestone:** global `ddrs plan` completes without error, adjacency parity test passes, and 1-epoch smoke train finishes without OOM or NaN loss.

**Performance milestone:** global trained ddrs beats summed-Q' baseline at median NSE across all 6,051 global gauges. The CONUS result (+0.037 NSE) is the reference benchmark.

### Known risks

- PSC `/gpfs/` access required for global data. These paths are not available on the local machine.
- Global adjacency build is ~25 s but produces a large zarr store. Verify disk space in `.ddrs/adjacency/<key>/` before building.
- Global batch size may need reduction. CONUS uses `batch_size: 64` (64 gauge subgraphs per mini-batch). At global scale, subgraph sizes may be larger; monitor peak GPU memory.
- Global observations use `Provider__GageId` format (not USGS STAID). The `GageMetadata::open` reader handles this, but verify STAID resolution logs show 0 missing.

---

## Cross-Cutting Concerns

### Stale-binary trap

After any `src/` change, the installed `~/.cargo/bin/ddrs` is stale. Always reinstall:

```bash
cargo install --path .
```

Quick self-check: current checkpoints are directories (`epoch_E_mb_M/head.mpk`). Flat files (`epoch_E_mb_M.mpk`) mean you ran an old binary.

### Regression gate (mandatory after any routing change)

```bash
cargo run --release --example compare_ddr_sandbox
# Must print "ABSOLUTE MATCH" (max abs diff < 1e-3 m³/s)
```

### CUDA Graphs + leakance incompatibility

`use_leakance: true` and `use_cuda_graphs: true` is rejected at config load. Do not attempt to combine them until SP-11 adds a unified capture path.

### KGE vs NSE

L1 and NSE are both maximized at simulated variance below observed — they reward over-attenuation. KGE penalizes variance ratio (alpha term). The `nnse-kge` loss option exists in `src/training/loss.rs` for this reason. Any new experiment targeting KGE improvement should test `experiment.loss.kind: nnse-kge`.

---

## Priority Order (as of 2026-07-05)

| Priority | Problem | Bottleneck | Unblocks |
|---|---|---|---|
| 1 | Selective equifinality paper | Need 4-arm training matrix + analysis script | Publication; identifiability protocol for the field |
| 2 | Leakance Phase B (state-cache hotstart) | ~130x noise floor blocks positive control | Leakance identifiability claim; auxiliary-constraint design |
| 3 | Backward CUDA Graphs (SP-11) | Fused backward cubecl kernels not written | Full CUDA graph speedup; V10 gate |
| 4 | Global routing | PSC `/gpfs/` access required | First global differentiable routing result |

---

## Provenance and Maintenance

All metrics and verdicts in this skill are sourced from the following files (re-verify before any publication claim):

```bash
# Equifinality paper draft:
cat /home/tbindas/projects/ddr_equifinality/paper.tex | head -100

# Leakance 2x2 findings:
cat /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-findings.md

# Leakance diagnosis (H1-H7 verdicts):
cat /home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md

# Gradient probe (P1-P3 verdicts):
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-03-zeta-gradient-probe-findings.md

# Synthetic recoverability (R1-R5 verdicts, noise floor):
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-04-synthetic-recoverability-findings.md

# Architecture and SP-10/SP-11 CUDA graph history:
cat /home/tbindas/projects/ddrs/.claude/ARCHITECTURE.md

# Regression gate:
cargo run --release --example compare_ddr_sandbox
```
