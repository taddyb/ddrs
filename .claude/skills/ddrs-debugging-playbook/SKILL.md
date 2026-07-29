---
name: ddrs-debugging-playbook
description: "Use when a ddrs run produces wrong results, silent failures, metric regressions, NaN loss, stale checkpoints, V1 mismatch, leakance anomalies, KAN head divergence, adjacency errors, data-source alignment issues, or any symptom that costs debug time. Also use before attributing a result to a code bug — many apparent bugs are operator error (stale binary, wrong fixture, config contradiction)."
---

# ddrs debugging playbook

**Audience:** Sonnet-class AI or mid-level ML engineer who knows PyTorch but not Rust/BURN.
**Voice:** imperative runbook. Copy-paste every command. Verify before claiming.

---

## Glossary (terms used throughout)

| Term | Meaning |
|---|---|
| **ddrs** | BURN-based Rust port of DDR (Python/PyTorch Muskingum-Cunge routing solver) |
| **DDR** | Python reference: `~/projects/ddr/`. The gold standard for numerical parity |
| **BURN** | Rust deep-learning framework (version 0.21 in this project) |
| **MC solver** | Muskingum-Cunge routing: converts upstream + lateral inflow to routed discharge per reach per timestep |
| **V1 / ABSOLUTE MATCH** | Regression gate: `compare_ddr_sandbox` max abs diff < 1e-3 m³/s vs DDR |
| **KAN head** | The neural network head (`rskan::KanLayer` v0.1.3): maps catchment attributes → routing parameters |
| **f32 invariant** | All tensors in the routing core must stay float32; f64/bf16 casts break DDR parity |
| **lower-triangular adjacency** | The CSR sparse pattern has `rows[k] >= cols[k]`; the forward-sub solver requires this |
| **sparse backward** | Hand-written O(nnz) `CsrSolveOp: Backward` in `src/sparse.rs`; must not be replaced by autograd unrolling |
| **leakance** | Experimental GW–SW water-loss term (`src/routing/leakance.rs`); off by default |
| **zeta** | The per-reach per-timestep leakance flux (m³/s): `zeta = leakance_factor · area_z · K_D · (depth − d_gw)` |
| **summed-Q baseline** | No-routing reference: per-gauge sum of upstream divide Qr. CONUS: median NSE 0.689 / KGE 0.723 (as of 2026-07-05) |
| **CUDA Graphs** | CUDA kernel-replay optimization (`use_cuda_graphs: true`); incompatible with leakance; masks NaN loss |
| **Q'** | Lateral inflow (m³/s) from an upstream forcing model (DHBV, LSTM, etc.) |
| **worktree** | Git worktree at a branch tip — used for experimental campaigns without touching main tree |

---

## When NOT to use this skill

- You want to understand the routing math or architecture → read `.claude/ARCHITECTURE.md` and `.claude/references/ddrs-algorithm.md`
- You want to port or verify a new feature against DDR → use skill `ddrs-comparing-to-ddr` (`.claude/references/ddrs-comparing-to-ddr.md`)
- You want to set up a new experiment from scratch → read `CLAUDE.md` §"ddrs CLI"
- You are doing leakance identifiability research → see `docs/2026-07-02-leakance-diagnosis-findings.md` for the completed hypothesis battery

---

## Part 1 — Symptom → triage table

Scan this table first. Each row points to a Part 2 entry with the full story and fix.

| Symptom | Most likely trap | Go to |
|---|---|---|
| Two runs that differ in config produce byte-identical predictions | Stale installed binary | [T1](#t1-stale-binary-trap) |
| Checkpoint files are flat `.mpk` (not a directory) | Stale binary | [T1](#t1-stale-binary-trap) |
| `manifest.json` shows current git SHA but behavior looks old | Stale binary (SHA stamps from `.git` at runtime, not the binary) | [T1](#t1-stale-binary-trap) |
| `ddrs run` silently ignores `disaggregation:` block | Stale binary (pre-disagg binary ignores unknown serde fields) | [T1](#t1-stale-binary-trap) |
| `compare_ddr_sandbox` reports diff > 1e-3 m³/s | V1 regression | [T2](#t2-v1-regression) |
| `compare_ddr_sandbox` fails after regenerating fixtures | Wrong DDR reference tree | [T2](#t2-v1-regression) |
| Loss goes NaN, but only with `use_cuda_graphs: true` | CUDA Graphs mask NaN | [T3](#t3-cuda-graphs-mask-nan) |
| Loss is finite but suspiciously constant across steps | CUDA Graphs returning stale capture | [T3](#t3-cuda-graphs-mask-nan) |
| Config parse error: "`use_leakance: true` requires `use_cuda_graphs: false`" | Config contradiction (intentional rejection) | [T4](#t4-leakance-config-contradictions) |
| Leakance run still uses CUDA graphs silently | Missing `use_cuda_graphs: false` in config | [T4](#t4-leakance-config-contradictions) |
| K_D pinned at ceiling (100% of reaches) | K_D box is binding — or head throttling (H2); see diagnosis | [T5](#t5-leakance-parameter-collapse-or-ceiling) |
| K_D collapsed to floor (sub-1e-8) | Replicates DDR's original revert failure; check forcing resolution | [T5](#t5-leakance-parameter-collapse-or-ceiling) |
| Gradient check fails on leakance op | Regression in `src/routing/leakance.rs` backward | [T6](#t6-leakance-gradient-correctness) |
| `zeta_accum` test fails | Accumulated zeta not matching headwater identity | [T6](#t6-leakance-gradient-correctness) |
| KAN head shape or init diverges from DDR | rskan version bump or inter-block ReLU accidentally re-added | [T7](#t7-kan-head-divergence) |
| KAN parity tests fail after `Cargo.toml` rskan bump | Fixture needs regeneration | [T7](#t7-kan-head-divergence) |
| Adjacency test fails or topological ordering wrong | `rows[k] < cols[k]` somewhere; lower-triangular invariant violated | [T8](#t8-adjacency-invariant) |
| `ddrs plan` hangs or errors on adjacency build | Bad fabric path or multi-layer gpkg needs `geospatial_fabric_layer` | [T8](#t8-adjacency-invariant) |
| Training NSE well below summed-Q baseline | Routing not earning its keep; loss or gradient issue | [T9](#t9-metric-regression-below-baseline) |
| KGE lower than baseline in every config | Expected — L1 loss penalizes variance; this is known behavior | [T9](#t9-metric-regression-below-baseline) |
| Baseline median NSE absurdly LOW; many gauges' baseline predictions exactly 0.0 | Pre-2026-07-28 binary: single-divide gauges summed an empty upstream set | [T9](#t9-metric-regression-below-baseline) |
| Hourly run produces same predictions as daily run | Stale binary (pre-disagg) or `aorc_precip` source missing | [T1](#t1-stale-binary-trap), [T10](#t10-disaggregation-no-op) |
| `MeritGagesDataset::open` errors with `use_precip: true` | `aorc_precip` source not configured | [T10](#t10-disaggregation-no-op) |
| Checkpoint resume trains zero batches | `experiment.epochs` not raised above checkpoint epoch | [T11](#t11-checkpoint-resume-issues) |
| Resumed run drifts from uninterrupted run | Expected: weights stored as f16 (CompactRecorder) | [T11](#t11-checkpoint-resume-issues) |
| `ddrs run --strict` exits with code 4 | Source fingerprint drift vs `.ddrs/sources.lock` | [T12](#t12-source-lock-drift) |
| Recoverability / identifiability experiment fails | Hotstart transient noise floor issue (Phase B not yet met) | [T13](#t13-leakance-identifiability-status) |

---

## Part 2 — Trap stories and fixes

### T1: Stale binary trap

**Story (2026-07-01).** The leakance × hourly 2×2 experiment produced two runs that were byte-identical despite different configs (one with hourly disaggregation, one without). The `manifest.json` showed the current git SHA `2cdd341` — which made it look like a code bug. Root cause: `~/.cargo/bin/ddrs` had mtime 2026-06-03, predating both the disaggregation feature (June 19) and leakance (June 29). The installed binary silently ignored the `disaggregation:` config block (serde ignores unknown fields) and wrote flat `.mpk` checkpoints instead of the current directory format.

**Discriminating test.** Check checkpoint format:
```bash
# Current binaries write DIRECTORIES:
ls .ddrs/runs/<id>/checkpoints/
# → epoch_5_mb_9/   (directory = current binary)
# → epoch_5_mb_35.mpk  (flat file = stale binary)
```

Check binary age:
```bash
stat ~/.cargo/bin/ddrs | grep Modify
# Should be >= your last src/ change date
```

**Fix (canonical):**
```bash
cargo install --path .
# OR faster if target/release/ is already built:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
# OR bypass the installed binary entirely:
cargo run --release --bin ddrs -- run --workflow train-and-test
```

**Rule:** `cargo build` does NOT update `~/.cargo/bin/ddrs`. Re-install after every `src/` change before invoking `ddrs` by name.

**Head size cross-check (CONUS, as of 2026-07-05):**
- No disagg, no leakance: ~103,459 B
- Disagg only: ~107,178 B
- Disagg + leakance (3 extra output cols): ~107,320 B

---

### T2: V1 regression

**What V1 is.** `examples/compare_ddr_sandbox` replays DDR's 5-reach RAPID2 sandbox through ddrs's MC solver. The threshold is `max abs diff < 1e-3 m³/s`. A passing run prints:
```
verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```
Typical passing value is ~1.5e-5 m³/s — two orders of magnitude under the threshold.

**Run it:**
```bash
mkdir -p output   # required — the example does not mkdir -p
cargo run --release --example compare_ddr_sandbox
# Also test the CUDA + graph-capture path:
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox
```

**Triage checklist when V1 fails:**

1. Inspect `output/ddrs_vs_ddr.csv` — which reaches are worst? Single bad reach = geometry/parameter bug. Global failure = solver or kernel issue.

2. Check if fixtures are stale:
   ```bash
   cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py
   cd ~/projects/ddrs && git diff fixtures/sandbox/
   ```
   **WARNING:** Only the local `~/projects/ddr` checkout is valid. The unpushed `geometry/trapezoidal.py` work is not in any public DDR commit. Regenerating from a clean DDR clone produces ~0.55 m³/s divergence at every ddrs commit — that is a wrong-reference artifact, NOT a port bug (as of 2026-06-06).

3. Audit recent changes to the only paths that affect V1:
   ```bash
   git log -p -- src/routing/ src/geometry.rs src/sparse/
   ```

4. Look for precision leaks:
   ```bash
   grep -rn "f64\|bf16\|cast\|to_dtype" src/routing/ src/geometry.rs src/sparse/
   ```
   Any cast away from f32 in these paths breaks DDR parity.

5. Cross-check with gradcheck:
   ```bash
   cargo test --test sparse_gradcheck
   ```
   If gradcheck fails too, the algorithm changed. If only V1 fails, it's a kernel-ordering or arithmetic-fusion difference.

**The threshold is non-negotiable.** Never relax `1e-3 m³/s`. Never declare "good enough."

---

### T3: CUDA Graphs mask NaN

**What happens.** When a forward pass produces NaN and `use_cuda_graphs: true`, the CUDA graph replays the stale pre-NaN capture instead of the live computation. The loss appears finite. You get silently wrong results with no error.

**Confirmed behavior (as of 2026-07-05):** `use_cuda_graphs: true` returns stale finite loss on a NaN forward.

**Discriminating test:**
```bash
# Reproduce the stale-loss symptom:
# Run once with cuda_graphs on vs off, inject a NaN input, compare loss values.
# If cuda_graphs=true gives finite loss and cuda_graphs=false gives NaN → confirmed.
```

**Fix:** Disable CUDA graphs when debugging any NaN or suspiciously-constant loss:
```yaml
# in ddrs.yaml or experiment config:
params:
  use_cuda_graphs: false
```

**Rule:** Always debug loss anomalies with `use_cuda_graphs: false`. Re-enable only after confirming the forward is NaN-free.

---

### T4: Leakance config contradictions

**What happens.** Two config errors involving leakance are caught at load time.

**Error 1 — leakance + CUDA graphs:**
```
params: `use_leakance: true` requires `use_cuda_graphs: false`
```
CUDA Graphs cannot capture the extra leakance kernel without a separate capture path. This is an intentional hard rejection in `src/config.rs:626`.

**Fix:**
```yaml
params:
  use_leakance: true
  use_cuda_graphs: false   # REQUIRED when use_leakance is true
```

**Error 2 — leakance parameters missing from KAN head.**
If `params.use_leakance: true` but `K_D`, `d_gw`, `leakance_factor` are not in `kan_head.learnable_parameters`, the head emits no leakance parameters and the routing silently has no exchange. No error is thrown — the leakance term gets a zero or garbage input.

**Complete leakance config checklist:**
```yaml
params:
  use_leakance: true
  use_cuda_graphs: false
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-6]          # log-space; hydraulic exchange rate 1/s
    d_gw: [-2.0, 2.0]              # groundwater depth offset, m
    leakance_factor: [0.0, 1.0]    # dimensionless scale

kan_head:
  learnable_parameters:
    - K_D
    - d_gw
    - leakance_factor
    # ... plus your routing params (n, q_spatial, etc.)
```

---

### T5: Leakance parameter collapse or ceiling

**Two failure modes:**

**Mode A — K_D at ceiling (100% of reaches).**
Observed in both leakance-ON arms of the 2026-07-01 2×2 (hourly and daily).
- `K_D` median log10 = −5.999, IQR = 3.6e-4 (essentially a delta function at the `1e-6` upper bound).
- This is NOT the K_D box clipping the flux. Diagnosis (2026-07-02) showed median in-box utilization is only 3.4% — the optimizer maxes the rate constant and then throttles the product via the driving head (`d_gw` learned near typical depths, so `depth − d_gw ≈ 0`).
- **K_D widening is NOT recommended** — the Phase-3 gate failed because H1 (structural ceiling) was REFUTED. Root cause: H2 (head throttling) + H4 (gauge bias) + H5 (equifinality under daily forcing).

**Mode B — K_D at floor (collapse to sub-1e-8).**
This replicates DDR's original revert failure (sub-0.01 m³/s exchange, physically negligible). If you see this, check:
- Is daily forcing being used? Under flat-daily forcing the depth dynamic range is too small for `zeta ∝ (depth − d_gw)` to be identifiable.
- Is hourly disaggregation actually running? Verify the binary is current (T1) and `aorc_precip` source is configured (T10).

**Discriminating check after any leakance run:**
```bash
cargo build --release --bin dump_parameters
target/release/dump_parameters \
  --config <your_config.yaml> \
  --checkpoint .ddrs/runs/<id>/checkpoints/epoch_E_mb_M/head \
  --output /tmp/kp.nc 2>&1 | grep -E "K_D|leakance_factor|d_gw|frac@"
```
Expected for a non-collapsed run: `K_D` interior or at ceiling (not floor), `leakance_factor` interior (0.1–0.5), `d_gw` spatially varying.

---

### T6: Leakance gradient correctness

**Guard tests — run all four after any change to `src/routing/leakance.rs` or `mmc_op.rs`:**

```bash
cargo test --test leakance_gradcheck      # analytical ≈ finite-difference (8 params)
cargo test --test leakance_off_parity     # byte-identical to no-leakance when off (3 tests)
cargo test --test zeta_accum              # accumulated zeta == headwater identity
cargo run --release --example compare_ddr_sandbox  # V1 must still pass
```

**If `leakance_gradcheck` fails:** The analytical backward in `TimestepLeakanceOp: Backward<I,8>` is wrong. Compare against `src/routing/leakance.rs` math: `zeta = leakance_factor · area_z · K_D · (depth − d_gw)` where `area_z = (p · depth)^q_eps · length`. All partial derivatives are straightforward products/chains; check each of the 8 inputs.

**If `zeta_accum` fails:** The accumulator in `evaluate` is not recomputing from the same primitives the backward used. The test verifies the headwater identity `q_no_leak[0] − q_leak[0] == zeta[0]`.

**If `leakance_off_parity` fails:** The leakance gating is broken — the `None` path is not byte-identical to a run without leakance compiled in.

---

### T7: KAN head divergence

**Architecture (must not change without explicit intent):**
```
Linear(F, H) → KanLayer(H, H) × num_hidden_layers → Linear(H, P) → Sigmoid
```
- No inter-block ReLU. DDR's `kan.py` has none; adding one breaks parity.
- All `num_hidden_layers` inner KanLayers get the SAME initialization seed (DDR `kan.py:24-34` quirk — preserved for parity).
- rskan version: `v0.1.3` (as of 2026-07-05). Pinned in `Cargo.toml:27`.

**Parity test suite:**
```bash
cargo test --features fixtures \
  --test kan_head_init_repro \
  --test kan_head_init_parity \
  --test kan_head_fixture_forward \
  --test kan_head_fixture_backward
```

**If tests fail after an rskan bump:** The fixtures need regeneration:
```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_kan_weights.py
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_kan_forward.py
```
Then re-run the test suite. If it still fails, the rskan API changed in a parity-breaking way — audit the diff and decide whether to update the DDR reference or roll back the bump.

**If tests fail without an rskan bump:** Check for accidental re-introduction of inter-block ReLU in `src/nn/kan_head.rs`.

---

### T8: Adjacency invariant

**The invariant:** The CSR adjacency pattern must be lower-triangular: every nonzero at `(row, col)` must have `row >= col`. The forward-substitution solver reads rows in order and assumes all upstream contributions are already resolved.

**Test:**
```bash
cargo test data_zarr_store::conus_adjacency_loads_real_merit_zarr
# Also:
cargo test --test adjacency_parity
```

**If adjacency build fails during `ddrs plan`:**
- Check the fabric path exists and is readable.
- For `.gpkg` files with multiple layers, set `geospatial_fabric_layer:` in config.
- The builder reads only the attribute table (`.dbf` or gpkg attributes) — never the geometry. Check `geospatial_fabric` points to the right file.
- On a fresh worktree, `.ddrs/adjacency/` does not exist — `ddrs plan` builds it on first run (~10 s for CONUS `.dbf`).

**If topological ordering is wrong:** The managed builder replicates petgraph's deterministic DFS finish-time order. Check `src/adjacency/build.rs::topological_sort` against the engine's version.

---

### T9: Metric regression below baseline

**Baseline numbers (CONUS, as of 2026-07-05):**
| config | median NSE | median KGE | gauges |
|---|---|---|---|
| Summed-Q′ baseline, same-run (2,365-gauge eval set) | 0.678 | 0.717 | 2365 |
| Best result: precip-disagg + L1 | 0.715 | 0.711 | 2365 |

**Critical known behavior:** KGE does NOT beat the summed-Q baseline in any trained config as of 2026-07-05. NSE beats it (+0.037 with precip disagg). The KGE regression is structural: the L1 loss maximizes at simulated variance below observed (α < 1), rewarding over-attenuation. The whole KGE drop is in the `α = σ_sim/σ_obs` term.

**Triage if NSE is far below baseline:**
1. Is the binary current? (T1)
2. Is the loss descending? Check `run.log` for epoch-mean L1.
3. Is CUDA graphs masking NaN? (T3 — disable and re-check)
4. Are gauge batch sizes reasonable? Too few gauges per batch → noisy gradient.
5. Is the data source correct? `streamflow resolution: Daily|Hourly` is logged at dataset open — verify it.

**The baseline itself can be the bug — phantom-zero single-divide gauges (fixed 2026-07-28).**
Gauges whose catchment is a single MERIT divide have ZERO edges in the gages
adjacency store (empty `indices_0`/`indices_1`, length-1 `order`). Before the fix,
`GageSubgraph::upstream_comids` derived the upstream set from edge endpoints only,
so these gauges summed an empty COMID set → the baseline predicted exactly 0.0 for
the whole window and scored it against real observations. On the
`daily_dhbv2_distributed_aorc2f` 3,211-gauge workspace this was 513 gauges (16%)
at median NSE −0.305, dragging the baseline median from 0.290 to 0.142. The fix
falls back to the outlet's own position (`gage_idx`) when a subgraph has no edges
(matches DDR's `summed_q_prime.py`, which reads the subgroup `order` array), and the
baseline cache key now includes `baseline-algo-v2` so stale pre-fix caches recompute.
Diagnostic: load `<workspace>/baselines/<key>/predictions.f32` and count rows that
are all-zero — any nonzero count on a post-fix binary is a new bug. Training was
never affected: `dataset.rs` drops zero-edge gauges as headwaters before batching.

**To improve KGE above baseline:** Switch to `experiment.loss.kind: nnse-kge`. The `(α-1)²` term in KGE provides the restoring gradient. This requires explicit config:
```yaml
experiment:
  loss:
    kind: nnse-kge
    nnse_weight: 1.0
    kge_weight: 1.0
```

---

### T10: Disaggregation no-op

**Symptoms:**
- Hourly and daily runs produce byte-identical predictions.
- Head file size is ~103,459 B (no-disagg size) even when config has `disaggregation: ...`.

**Root causes (in priority order):**

1. **Stale binary** — most likely. See T1. The pre-disagg binary silently ignores the `disaggregation:` block.

2. **Missing `aorc_precip` source.** The AORC precip zarr at `/mnt/ssd1/data/aorc/merit_unit_catchments.zarr` must be in `data_sources:`. Without it, `MeritGagesDataset::open` errors when `use_precip: true`. Check:
   ```bash
   ddrs sources list   # '*' marks active group
   # conus-hourly group includes aorc_precip
   ddrs sources use conus-hourly
   ```

3. **`use_precip: false` in config.** The `aorc_precip` source must be present AND `kan_head.disaggregation.use_precip: true` must be set. The source group (`conus-hourly`) splices in the source; the disagg block must be in the experiment config separately.

**Verification after a fix:**
```bash
# Binary check: head file should be ~107,320 B (disagg + leakance) or ~107,178 B (disagg only)
ls -la .ddrs/runs/<id>/checkpoints/epoch_5_mb_9/head.mpk

# Dataset log at run start:
grep "AORC precip store" .ddrs/runs/<id>/run.log
# Should show: "AORC precip store: 290878 catchments"

# Forcing verification: eval predictions should differ between hourly and daily runs:
md5sum .ddrs/runs/<hourly-id>/eval/predictions.zarr/predictions/0.0
md5sum .ddrs/runs/<daily-id>/eval/predictions.zarr/predictions/0.0
# These must NOT be identical
```

---

### T11: Checkpoint resume issues

**Resume requires three files in a directory:**
```
.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/
  head.mpk      # KAN weights (f16, CompactRecorder)
  optim.mpk     # Adam moments (f16)
  state.json    # epoch, next mini-batch, rng state, sampler permutation + cursor
```

**Resume trains zero batches:** `experiment.epochs` is at or below the checkpoint epoch. Fix: raise `experiment.epochs` past `E` in `ddrs.yaml`.

**Resumed trajectory drifts from uninterrupted run:** Expected. Weights and moments are stored as f16 (`CompactRecorder = HalfPrecisionSettings`). The resumed trajectory is numerically valid but will not be bit-identical to an uninterrupted run.

**`dump_parameters --checkpoint` path gotcha:** Pass the HEAD BASE, not the directory:
```bash
# CORRECT (head base — CompactRecorder appends .mpk):
target/release/dump_parameters --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9/head ...
# WRONG (directory — will fail to find head.mpk):
target/release/dump_parameters --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 ...
```

---

### T12: Source lock drift

**What happens.** `ddrs run --strict` exits with code 4 when the data-source fingerprints in `.ddrs/sources.lock` differ from the current `ddrs.yaml`. This preserves evidence; re-locking would overwrite it.

**Fix (normal):**
```bash
ddrs plan   # re-locks sources.lock to match current ddrs.yaml
ddrs run --workflow <wf>
```

**Fix (investigate first):**
```bash
cat .ddrs/sources.lock   # shows last-locked fingerprints
# Compare against current data_sources: paths in ddrs.yaml
# If a path moved or a store was updated, decide whether to re-plan or roll back
```

---

### T13: Leakance identifiability status (as of 2026-07-05)

**This section describes an active research limitation — not a bug to fix.**

**Positive control experiment (2026-07-04, worktree):** A synthetic recoverability test was run to check whether the gradient path can recover a known planted zeta through gauged-only observations. The experiment FAILED: recovery ratio 0.009 vs the >=0.5 bar. Root cause: the windowed training objective has a hotstart-transient noise floor approximately 130× larger than the leakance signal.

**Implication:** Leakance identifiability is NOT proven. The 2×2 GO-marginal verdict (leakance helps skill on the losing-stream subset under hourly forcing) stands, but the mechanism cannot be confirmed as genuine GW–SW exchange recovery until Phase B is complete.

**Phase B objective (NOT YET MET as of 2026-07-05):** noise floor <= 0.25 mean L1 (i.e., <= 10% of a converged run's loss). Requires a state-cache hotstart to eliminate the transient. Do not make identifiability claims until Phase B passes.

**Gradient probe results (2026-07-03, worktree):**
- P1 (gradient starvation to leakance params): REFUTED
- P3 (detectability — signal vs 5% obs band): NO-GO, signal 53× smaller than detectability threshold

**Summary of leakance diagnosis verdicts (as of 2026-07-02):**

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H1: K_D box clips zeta | REFUTED | Median utilization 3.4%; 71.5% of reaches CAN exceed 0.01 m³/s in-box |
| H2: Driving-head starvation | SUPPORTED | Median head 0.021 m; 47% of reaches gaining at eval-window mean |
| H3: KAN variance collapse | REFUTED | Max Spearman(param, attribute) = 0.71 (strong spatial structure) |
| H4: Gauge bias / gradient starvation | SUPPORTED | zeta–uparea ρ +0.76; gauged 11× ungauged median zeta; dry/wet ratio inverted |
| H5: Equifinality with routing params | SUPPORTED (daily only) | Daily Δn = +0.012 (0.59 IQR); hourly Δn nil |
| H6: Wrong yardstick (absolute bar) | REFUTED | Fractional loss agrees: 8.4% lose >1% of local flow |
| H7: Model-form error (d_gw bounds) | REFUTED | 0.0% of d_gw at bounds in any aridity tercile |

**Do not run K_D widening.** The Phase-3 gate for K_D widening FAILED because H1 was REFUTED. The constraint is the signal, not the box.

---

## Part 3 — Pre-flight checklist before any training run

Use this before starting a new experiment to prevent the most common traps:

- [ ] `stat ~/.cargo/bin/ddrs` — mtime is newer than your last `src/` change
- [ ] `ddrs sources list` — active group (`*`) matches intended dataset
- [ ] `ddrs plan` — no source drift warnings; `mode:` and `workflow:` agree
- [ ] Config leakance consistency: if `use_leakance: true`, confirm `use_cuda_graphs: false` and all three params in `kan_head.learnable_parameters`
- [ ] If hourly disagg: config has `aorc_precip:` source AND `kan_head.disaggregation.use_precip: true`
- [ ] If resuming: `experiment.epochs` is greater than the checkpoint epoch; checkpoint path ends at `head` base (not the directory)
- [ ] If touching `src/routing/`, `src/geometry.rs`, or `src/sparse/`: run `cargo run --release --example compare_ddr_sandbox` and confirm ABSOLUTE MATCH

---

## Part 4 — Quick-reference test commands

```bash
# V1 regression gate (routing core, geometry, sparse):
mkdir -p output && cargo run --release --example compare_ddr_sandbox

# V1 on CUDA + graph-capture path:
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox

# Leakance gradient correctness:
cargo test --test leakance_gradcheck
cargo test --test leakance_off_parity
cargo test --test zeta_accum

# Sparse backward correctness:
cargo test --test sparse_gradcheck

# KAN head parity vs DDR:
cargo test --features fixtures \
  --test kan_head_init_repro \
  --test kan_head_init_parity \
  --test kan_head_fixture_forward \
  --test kan_head_fixture_backward

# Adjacency ordering and builder parity:
cargo test data_zarr_store::conus_adjacency_loads_real_merit_zarr
cargo test --test adjacency_parity

# All lib unit tests:
cargo test --lib

# Full test suite:
cargo test
```

---

## Provenance and maintenance

Ground truth for this skill (re-read these files to verify facts remain current):

```bash
# V1 / comparing-to-DDR reference:
cat /home/tbindas/projects/ddrs/.claude/references/ddrs-comparing-to-ddr.md

# Stale-binary trap story + leakance 2x2 re-run:
cat /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-experiment-handoff.md

# 2x2 findings (all four arms, GO verdict):
cat /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-findings.md

# Low-zeta diagnosis (H1–H7 hypothesis verdicts):
cat /home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md

# Config rules (invariants, leakance enable, CLI lifecycle):
cat /home/tbindas/projects/ddrs/CLAUDE.md

# CUDA graphs mask NaN (memory note):
cat /home/tbindas/projects/ddrs/.claude/memories/cuda-graphs-mask-nan.md

# Re-verify rskan version:
grep rskan /home/tbindas/projects/ddrs/Cargo.toml

# Re-verify config leakance rejection:
grep -n "use_leakance.*use_cuda_graphs\|use_cuda_graphs.*use_leakance" \
  /home/tbindas/projects/ddrs/src/config.rs
```
