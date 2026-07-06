---
name: ddrs-failure-archaeology
description: "Use when about to re-investigate a question that has already been settled in ddrs — for example before retrying a rejected fix, re-opening a dead-end hypothesis, or proposing a change that a prior experiment already tested. Also use when onboarding to understand what NOT to attempt and why. Covers every major investigation, dead end, rejected fix, and revert from project inception through 2026-07-05."
---

# ddrs Failure Archaeology

**Audience:** Sonnet-class AI model or mid-level ML engineer who knows PyTorch but not Rust/BURN.

**When NOT to use this skill:** If you need to understand the current correct architecture or algorithm, use `ddrs-architecture` or `.claude/ARCHITECTURE.md`. If you need to reproduce an existing result, use the run IDs and reproduce commands in the findings docs directly.

---

## Terminology (defined once)

| Term | Meaning |
|---|---|
| **BURN** | Rust deep learning framework (like PyTorch for Rust). autograd = automatic differentiation, how gradients flow backward. |
| **MC / MMC / Muskingum-Cunge** | A physics-based river routing algorithm. Takes upstream inflow + channel properties → downstream discharge. ddrs implements this as a differentiable Rust port of DDR (Python/PyTorch). |
| **KAN head** | Kolmogorov-Arnold Network head. The neural net that maps static reach attributes → per-reach physical parameters (Manning's n, geometry, etc.). Implemented via `rskan::KanLayer`. |
| **Q' (Q-prime)** | Lateral runoff inflow to each reach (m³/s), produced by dHBV2-UH upstream. The raw forcing before routing. |
| **summed-Q' baseline** | Per-gauge sum of upstream Q' with NO routing. The "do nothing" benchmark. If trained routing can't beat this, routing adds no value. |
| **NSE / KGE** | Nash-Sutcliffe Efficiency / Kling-Gupta Efficiency. Both on [−∞, 1]; higher = better. KGE decomposes into correlation r, variance ratio α, and bias β. |
| **α (alpha)** | σ_sim / σ_obs. If < 1 the model over-attenuates (flattens peaks). |
| **zeta (ζ)** | The leakance flux: `zeta = leakance_factor · area_z · K_D · (depth − d_gw)` in m³/s. Subtracted from routing RHS → stream loses water to groundwater. |
| **disagg head** | Daily→hourly disaggregation head. Converts daily Q' into a 24-hour shape, mass-preserving. |
| **rho-window** | Training uses short windows of length `rho` days (default 90) rather than the full time series. |
| **hotstart transient** | The error introduced when starting a routing window from an approximated initial condition rather than the true continuous state. |
| **CONUS** | Contiguous United States. 346,321 MERIT reaches in the CONUS training network. |
| **CsrSolveOp / TimestepLeakanceOp** | Hand-written BURN autograd backward ops. These are custom Jacobian implementations, not autograd tape unrolling. |
| **CUDA Graphs** | A GPU optimization that captures a sequence of GPU kernel launches and replays them as one. In ddrs: `use_cuda_graphs: true` in config. |
| **`ddrs` binary** | The main CLI binary installed to `~/.cargo/bin/ddrs`. NOT automatically updated by `cargo build`. |

---

## Overview: the arc of investigations

```
2026-05 to 06-04  Performance optimization (SP-6 through SP-10, CUDA Graphs)
                      → mixed results; several dead ends documented below
2026-06-16 to 19  Loss & optimization failures (why training loss is flat)
                      → 4 experiments; root cause: daily→hourly gradient null-space
2026-06-22 to 24  Disaggregation head: precip-driven (BEST RESULT to date)
                      → NSE beats baseline +0.037; KGE does not (structural ceiling)
2026-07-01        Leakance × hourly 2x2: GO-marginal verdict
2026-07-02        Leakance low-zeta diagnosis (H1–H7 battery)
2026-07-03        Gradient probe (P1/P2 REFUTED; P3 NO-GO)
2026-07-04        Synthetic recoverability positive control (FAILED; ~130x noise floor)
                      → Phase B objective: reduce floor to ≤0.25 mean L1
```

---

## Part 1: Operational traps (do not re-open)

### Trap 1: The stale-binary trap

**What happened (2026-07-01):** The `ddrs` binary on PATH was a June-3 build with no disaggregation and no leakance. The 2x2 experiment's hourly and daily cells were byte-identical — a false "disagg no-op". All results were invalid and had to be rerun.

**Rule:** `cargo build` and `cargo run` do NOT update `~/.cargo/bin/ddrs`. After any `src/` change, do one of:
```bash
cargo install --path .                                         # canonical
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs  # faster
cargo run --release --bin ddrs -- run --workflow ...           # bypass installed copy
```
**Quick self-check:** Current checkpoints are DIRECTORIES (`epoch_E_mb_M/head.mpk`). Flat files (`epoch_E_mb_M.mpk`) mean you ran a stale pre-checkpoint-resume binary.

**Do NOT:** Trust the manifest's `git.sha` field as proof of which binary ran. The SHA is stamped from `.git` at runtime, not baked into the binary.

---

### Trap 2: CUDA Graphs mask NaN (use_cuda_graphs: true)

**What happened (2026-06-23):** AORC precip array contained ~14% real NaN values (ocean/no-coverage catchments). NaN flowed through `log1p` → NaN softmax → NaN forcing → NaN routing. With `use_cuda_graphs: true`, the printed training losses looked finite — CUDA Graph replay returns a stale loss scalar from the capture pass, not the current forward. All 2365 eval gauges had NaN NSE.

**Rule:** `use_cuda_graphs: true` + NaN in the forward pass = silently corrupt weights and finite-looking losses. **Always validate a new data path with `use_cuda_graphs: false` first.**

**Fix applied:** Zero-fill non-finite precip at `AorcPrecipStore` read time + `max(0.0)` before `log1p` (commit a5972d9).

**Do NOT:** Trust a finite printed training loss as proof of a finite forward when CUDA Graphs are on.

---

### Trap 3: leakance + use_cuda_graphs: true is a config error

**Status:** This combination is rejected at config load time. The CUDA Graphs capture path does not include the leakance kernel. Do not attempt to combine them.

---

### Trap 4: Widening K_D past 1e-6 was superseded

**Background:** After the 2x2 GO-marginal result, the top follow-up was "widen K_D from [1e-8, 1e-6] to find where it really wants to sit." This was superseded by the 2026-07-02 low-zeta diagnosis.

**Diagnosis result (H1 REFUTED):** The K_D box does NOT cap the flux. With K_D at ceiling, factor=1, d_gw at floor, 71.5% of reaches could already exceed the 0.01 m³/s bar — but median utilization is only 3.4%. The optimizer maxes the rate constant and then throttles the product via the driving head. Widening K_D would just re-pin at the new ceiling.

**Do NOT:** Run a widened-K_D retrain. The Phase-3 gate pre-registered for this was NOT passed.

**Note:** The recoverability control (2026-07-04) DID use K_D [1e-8, 1e-5] for expressibility reasons in that specific synthetic experiment. That does not reopen the real-data K_D widening question.

---

### Trap 5: The DDR revert story is backwards for ddrs

**DDR history:** DDR (Python/PyTorch) saw K_D collapse to the **floor** (sub-0.01 exchange) and reverted leakance.

**ddrs result:** ddrs saw K_D pin at the **ceiling** (100% of reaches at 1e-6). These are OPPOSITE behaviors. Do not apply the DDR revert rationale to ddrs.

---

## Part 2: Optimization experiments that failed

### Experiment 0 (2026-06-16): L1 vs NNSE-KGE loss

**Hypothesis:** L1 rewards over-attenuation (α < 1 is its optimum). KGE-aware loss with `(α−1)²` term should restore amplitude and beat baseline.

**Result:**
| config | median NSE | median KGE | α |
|---|---|---|---|
| L1 | 0.684 | 0.701 | 0.85 |
| NNSE-KGE | 0.684 | 0.699 | — |
| summed-Q' baseline | 0.689 | 0.723 | 0.96 |

**Verdict: FAILED.** Both below baseline. Training loss was **flat across all 10 epochs in both runs**. The objective is not the binding constraint.

**Lesson:** When training loss is flat, fix the gradient path. Do not try new loss functions yet.

---

### Experiment 1 (2026-06-17): Component-weighted KGE + learnable Muskingum X

**Hypothesis:** Add learnable per-reach X (attenuation-vs-translation knob, range [0, 0.5]) + α-weighted KGE loss (α-weight=2). Physical knob + matching loss should de-attenuate.

**Result:** NSE 0.676 / KGE 0.698 — slightly worse. Loss stayed flat (~1.2). X stuck at init: median 0.246, range [0.135, 0.257]. 0% of reaches reached either bound despite α-weight=2.

**Verdict: FAILED.** ∂loss/∂(routing params) ≈ 0. X cannot learn if the gradient is zero.

**Root cause identified (Experiment 2 below):** daily→hourly `repeat-24` upsampling + daily-mean aggregation puts routing's within-day effect in the gradient's null-space.

---

### Experiment 2 (2026-06-17): Dump learned X field

**Purpose:** Confirm gradient is dead by inspecting X values.

**Result:** X median 0.246, p10–p90 = 0.214–0.253. Decisive.

**Implementation added:** `src/dump_parameters.rs` extended to emit `x_storage`.

---

### Experiment 3 (2026-06-18): Learnable daily→hourly disaggregation head (no precip)

**Hypothesis:** Flat `repeat-24` upsampling is the gradient bottleneck. Mass-preserving learnable disagg head (3-tap windowed log-Q' → MLP → softmax over 24 hours) will unstick it.

**Result:**
- Training loss **descended for the first time ever** (1.224 → 1.02)
- X **moved for the first time** (0.246 → 0.217)
- Held-out: NSE 0.680, KGE 0.624 (WORSE than baseline), α 0.761

**Verdict:** Hypothesis CONFIRMED (disagg unsticks gradient). But held-out regressed = overfitting. The sub-daily unsupervised disagg learns within-day shapes for more attenuation that do not generalize. Disagg head without real precip supervision is NOT a solution.

---

### Experiment 4 (2026-06-18): Early-stopping sweep over Experiment 3 checkpoints

**Result:**
| epoch | NSE | KGE |
|---|---|---|
| 1 | 0.648 | 0.665 |
| 2 (best) | 0.681 | 0.671 |
| 10 | 0.680 | 0.624 |

**Verdict:** NSE plateaus after epoch 2; KGE degrades monotonically. Even epoch-2 LOSES to baseline on both metrics (baseline: NSE 0.689, KGE 0.723). Early stopping confirms overfitting but does not fix the structural ceiling.

**Lesson:** Fixing the optimization barrier (disaggregation) exposed a structural ceiling: at daily resolution over already-UH-routed forcing, learnable routing cannot add generalizable held-out skill beyond summed-Q'.

---

## Part 3: The best result and its limits

### Precip-driven disaggregation (2026-06-23)

Run `2026-06-23T02-49-12Z-conus-hourly-train-and-test`. Real hourly AORC precip drives the within-day shape.

**BEST RESULT as of 2026-07-05 (2365 CONUS gauges, eval 1995-10 to 2010-09):**

| config | median NSE | median KGE |
|---|---|---|
| Trained (precip-disagg + L1) | **0.7152** | **0.7106** |
| summed-Q' baseline | 0.6781 | 0.7172 |
| Delta | **+0.037** | **−0.007** |

**Critical fact:** NSE beats baseline by +0.037. KGE does NOT beat baseline (−0.007). This is structural: summed-Q' already has the highest KGE of any config. Routing already-dHBV2-UH-smoothed Q' cannot beat the no-routing KGE. As of 2026-07-05, no config beats the KGE baseline.

**Decomposition (precip-ON vs precip-OFF vs baseline):**

| effect | Δ NSE | Δ KGE |
|---|---|---|
| Precip contribution (ON − OFF) | +0.0196 | +0.0180 |
| Bare disagg alone vs baseline | +0.0176 | −0.0245 |
| Net vs baseline | +0.0372 | −0.0065 |

Bare disagg trades KGE for NSE (over-attenuation from invented within-day shapes). Real precip timing rescues the KGE the bare disagg destroys.

### NNSE-KGE loss with precip (2026-06-24)

Prediction: α-restoring loss would lift KGE over baseline. Result: KGE 0.7100 vs 0.7106 (L1) — essentially unchanged. **The L1-over-attenuation hypothesis was REFUTED when the gradient actually flows.**

**Do NOT:** Try balanced nnse-kge as the KGE fix for precip-driven runs. It does not earn its keep.

### Temperature as second disagg channel (2026-06-24)

NSE unchanged (0.7155 vs 0.7152). KGE marginally worse (0.7088 vs 0.7106). Temperature does not earn its keep on CONUS-median skill.

---

## Part 4: Performance optimization dead ends

All of these are in `.claude/ARCHITECTURE.md` §SP-8 through §SP-10.

### SP-8: MC timestep fusion

- Goal: collapse ~33 autograd nodes per timestep into 1.
- Result: 27% wall-time improvement (5.58 → 4.06 min).
- **Did NOT meet V7 GPU gates:** CPU/CUDA ratio = 1.000 (fusion sped both symmetrically because the win was autograd-graph collapse on the Rust side, not GPU-specific). `scatter_kernel` still at 77.5% of GPU time.
- Lesson: autograd-graph collapse is cheap; the scatter_kernel hotspot was not touched.

### SP-9: cuSPARSE SpMV

- Replaced `Tensor::scatter(..., Add)` in `src/sparse/mod.rs` with `cusparseSpMV`.
- `scatter_kernel` dropped from 77.5% to **0.0%** of GPU time. V7b GREEN.
- CPU/CUDA ratio: 0.919 (partial; missed 0.7 target). Remaining floor: launch overhead (~8M `cuLaunchKernel` calls at ~2.3 μs each).

### SP-10: CUDA Graphs (forward-only)

Seven architectural layers had to be solved before capture worked. Key failures:
1. `cuEventSynchronize` inside cubecl `flush()` invalidated the capture stream. Fixed via cubecl-fork patch `flush_no_sync` (branch `taddyb/cubecl:ddrs-sp7-stream-accessor`).
2. Re-entrant `exclusive_with_server` deadlocked on capture.
3. Transient cubecl-pool allocations baked into the graph caused address collisions on replay.
4. Persistent-mode + handle-pinning caused `CUDA_ERROR_ILLEGAL_ADDRESS`. **Pinning was abandoned.**
5. cuSPARSE workspace allocation: suspected un-capturable; proved FALSE (cuSPARSE 12.x manages workspace externally).
6. Solution: fused `#[cube]` kernels (K1 geometry, K2 RHS, K3 clamp) so zero cubecl-pool allocations in the captured region.
7. Per-batch pool fragmentation: fixed by `client.memory_cleanup()` after optimizer step.

**Result:** V7a GREEN (CPU/CUDA ratio 0.385, 2.4× over SP-9). V10 PARTIAL (29.2% launch-kernel reduction; backward path not yet captured).

**Known issue:** `use_cuda_graphs: true` is incompatible with leakance and is rejected at config load time.

---

## Part 5: Leakance investigations

### 2x2 (leakance × forcing) — 2026-07-01

**Hypothesis:** Leakance helps on losing-stream subset under hourly forcing; effect absent or weaker under daily.

**Valid arms (re-run after stale-binary fix):**

| arm | run id | median NSE | median KGE |
|---|---|---|---|
| hourly-OFF (control) | `2026-06-23T02-49-12Z-conus-hourly-train-and-test` | 0.7153 | 0.7104 |
| hourly-ON | `2026-07-01T13-43-32Z-train-and-test` | 0.7145 | 0.7150 |
| daily-OFF (control) | `2026-06-05T01-41-16Z-train-and-test` | 0.7004 | 0.7244 |
| daily-ON | `2026-07-01T21-20-27Z-train-and-test` | 0.6963 | 0.7250 |

**Losing-stream subset (1883/2365 gauges where baseline over-predicts):**

| arm | ΔNSE | ΔKGE | frac(ΔNSE>0) |
|---|---|---|---|
| hourly ON−OFF | +0.0005 | +0.0018 | 55.5% |
| daily ON−OFF | −0.0017 | −0.0009 | 35.6% |

**K_D pinning (hourly-ON dump):** 100% of 346,321 reaches at the 1e-6 ceiling. `leakance_factor` interior ≈0.33. `d_gw` interior ≈0.29 m.

**Eval-time zeta (hourly-ON, 64,892 eval reaches):** median |zeta| = 6.4e-4 m³/s; |zeta| > 0.01 on **10.4%** of reaches (gate: ≥10%). Net-losing on 53.7%.

**Verdict: GO — marginal.** All three gate criteria met. 10.4% vs 10% threshold = no headroom.

---

### Low-zeta diagnosis (2026-07-02) — H1–H7 battery

**Why zeta is small: the full hypothesis test.**

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H1 | Structural ceiling (K_D box caps flux) | **REFUTED** | 71.5% of reaches CAN exceed 0.01 m³/s in-box; median utilization 3.4% |
| H2 | Driving-head starvation (d_gw ≈ depth) | **SUPPORTED** | median head 0.021 m; 57.6% < 0.1 m; 47.0% ≤ 0 |
| H3 | KAN variance collapse | **REFUTED** | d_gw–meanP Spearman +0.71; K_D–aridity +0.61 — strong learned structure |
| H4 | Gauge bias / gradient starvation | **SUPPORTED** | zeta–uparea ρ +0.76; gauged median 6.7e-3 vs ungauged 5.9e-4; dry/wet ratio 0.40 (inverted from physics) |
| H5 | Equifinality (n absorbs leakance) | **SUPPORTED** (daily only) | daily Δn = +0.012 (0.59 IQR); hourly Δn ~nil |
| H6 | Wrong yardstick (absolute bar irrelevant) | **REFUTED** | fractional loss agrees: 8.4% lose >1%; 3.2% >5% |
| H7 | Model-form error (d_gw boundary-pinning) | **REFUTED** | 0.0% of d_gw at bounds |

**The coherent mechanism:** H2 + H4 + H5 form one story. The optimizer maxes K_D and then throttles flux via d_gw (which it learned at ≈ typical depth). The gradient that could open the head only reaches gauged, large-river reaches — not arid, ungauged, losing headwaters (the physics says dry reaches should have more loss; empirically they have less).

**The K_D-ceiling story is dead.** Do not re-open it without H1 being supported.

**Phase-3 gate (widened-K_D retrain): NOT passed. No GPU was spent.** The diagnosis predicts widening re-pins at the new ceiling with little change.

---

### Gradient probe (2026-07-03, worktree: zeta-sensitivity)

**Hypotheses tested:**

| # | Hypothesis | Verdict | Key numbers |
|---|---|---|---|
| P1 | Gradient starvation (∂L/∂zeta dead off-gauge) | **REFUTED** | gauged/ungauged \|g\| ratio: 1.5× (trained), 2.9× (cold); bar was ≥10× |
| P2 | Gradient rejection (optimizer actively pushes zeta down away from gauges) | **REFUTED** (trained); POST-HOC discovery: 80.5% of ungauged cold-init grads push DOWN | 52.5% dry-tercile push-down at trained point; bar was >67% |
| P3 | Detectability (real-magnitude loss signal exceeds gauge uncertainty band) | **NO-GO** | 4.2% of Ref probes at δ=0.01 detectable; bar was ≥10% |

**The P3 decomposition:** Planted 0.01 m³/s loss transmits to gauge at **94.6% fidelity** (routing is fine). But median Ref gauge 5%-uncertainty band is **0.53 m³/s — 53× the planted loss**. Detection fails on dilution, not transmission.

**Post-hoc discovery (P2 cold):** At the cold init point, 80.5% of ungauged gradients push zeta DOWN with 15.9× wet/dry gradient asymmetry. Early training actively suppressed leakance nearly everywhere before any spatial signal could develop. This is "initial-training suppression" — a second barrier beyond P3.

**Conclusion:** Gauge-only discharge supervision cannot learn real-world leakance. Auxiliary spatial supervision is empirically forced, not just recommended.

---

### Synthetic recoverability positive control (2026-07-04, worktree: zeta-sensitivity)

**What it tested:** In the BEST POSSIBLE WORLD (teacher-generated observations, warm-started from teacher weights, detectable-by-construction planted flux, zero obs noise), does training recover the planted leakance?

**Setup:**
- 58 planted reaches (from Ref GAGES-II sites, K_D widened to [1e-8, 1e-5] for expressibility)
- 3 students: A (warm-start, leakance ON), B (warm-start, leakance OFF), C (cold, leakance ON)
- 5 epochs × 36 mini-batches each on CPU (NdArray, deterministic)

**Results:**

| # | Metric | Measured | Verdict |
|---|---|---|---|
| R1 | Recovery ratio median (n=58 planted reaches) | **0.009** | **FAILED** (bar: ≥0.5) |
| R2 | Non-planted \|zeta_net\| A/baseline | 1.11 | PRECISE (bar: <2) |
| R3 | Final-epoch loss A vs B | A=1.339, B=2.317, +42.2% gap | A<B — but CONFOUNDED |
| R4 | Δn absorption at planted basins | median Δn = −0.019, global | descriptive |
| R5 | Cold emergence ratio | 1.20 | SUPPRESSED (bar: >3) |

**HEADLINE: FAIL.**

**The root cause — the ~130x noise floor:**

| Quantity | Value |
|---|---|
| Continuous residual (teacher weights + teacher obs, full-window eval) | 0.0076 mean L1 |
| Step-0 windowed training loss (run A) | 1.017 |
| Ratio (noise floor / signal) | ~130× |
| Run A continuous residual after 5 epochs of training | 0.4431 (58× worse than not training) |

The windowed objective (rho=90, warmup=5) starts every window from heuristic initial conditions. Big rivers carry memory of tens to hundreds of days; warmup=5 trims far too little. The planted signal (0.0076 mean L1) is 0.8% of the training loss (1.017) — invisible. Adam then ACTIVELY degrades the model chasing initial-condition noise.

**R3 is CONFOUNDED:** The 42.2% loss gap between A and B measures that the leakance BASE FIELD matters in aggregate (removing it forces global n re-equilibration). It does NOT measure whether individual planted fluxes were recoverable.

**Implication:** The windowed training objective is a third, independent masking layer on top of P3 (observational uncertainty) and P2-cold (initial-training suppression). Even in a noise-free world with perfectly warm-started weights, the objective's own noise floor swamps the signal.

**General ddrs training finding:** warmup=5 under-trims hotstart transients by ~2 orders of magnitude. Fine-tuning a converged model on its own synthetic outputs makes it WORSE.

---

## Part 6: Phase B objective and current state

**Phase B target (as of 2026-07-05): NOT YET MET.**

- Goal: reduce the windowed training objective's noise floor to ≤0.25 mean L1 (≤10% of a converged run's loss, vs the current ~40%).
- Approach options: (B1) longer warmup/rho curve (forward-only, cheap diagnostic), (B2) state-cache hotstart (persistent reach states carried across windows).
- The Phase B plan is at `docs/superpowers/plans/2026-07-04-phase-b-floor-fix.md` and `docs/superpowers/plans/2026-07-04-phase-b2-state-cache.md`.

**Leakance identifiability is NOT proven.** The positive control failed. Until Phase B meets its ≤0.25 mean L1 target, no identifiability claim can be made about leakance.

---

## Part 7: What IS settled and cannot be re-argued

Use SUPPORTED / REFUTED / INCONCLUSIVE as evidence labels.

| Claim | Status | Evidence |
|---|---|---|
| Daily repeat-24 upsampling kills routing gradient | SUPPORTED | Exp 3: only change that ever made loss descend and X move |
| Real precip timing improves both NSE and KGE over daily-Q-only disagg | SUPPORTED | precip-ON vs precip-OFF delta: +0.020 NSE, +0.018 KGE |
| KGE beats summed-Q' baseline under any current config | REFUTED | best result: KGE 0.7106 vs baseline 0.7172; as of 2026-07-05 |
| NSE beats summed-Q' baseline with precip disagg + L1 | SUPPORTED | +0.037 (2365 gauges) |
| Balanced NNSE-KGE loss fixes the KGE gap | REFUTED | KGE 0.7100 vs L1's 0.7106 |
| Temperature channel helps CONUS-median KGE | REFUTED | 0.7088 vs precip-only 0.7106 |
| K_D box is what caps learned zeta | REFUTED | H1: 71.5% of reaches can exceed bar in-box; median utilization 3.4% |
| KAN head has collapsed spatial variance | REFUTED | H3: K_D-aridity Spearman +0.61, d_gw-meanP +0.71 |
| Gradient starvation (∂L/∂zeta dead off-gauge) | REFUTED | P1: gauged/ungauged ratio 1.5-2.9×, not ≥10× |
| Early training suppresses leakance at cold init | SUPPORTED | P2: 80.5% of ungauged cold grads push DOWN, 15.9× wet/dry asymmetry |
| Real-magnitude leakance (0.01 m³/s) is detectable at gauges | REFUTED | P3: signal 53× below 5% gauge uncertainty band at median Ref gauge |
| Leakance recoverable under windowed training (even with detectable signal, warm start) | REFUTED | R1: recovery ratio 0.009; ~130× noise floor |
| Leakance term load-bearing for aggregate fit | SUPPORTED | R3: removing leakance costs 42% training loss, forces global n re-equilibration |
| Hourly forcing is a precondition for leakance (not nice-to-have) | SUPPORTED | H5: under daily forcing n compensates for leakance (daily Δn 0.59 IQR); under hourly mechanisms decouple |
| CUDA graphs + use_cuda_graphs:true can return stale finite loss on NaN forward | SUPPORTED | AorcPrecip bug: all 2365 gauges NaN at eval; losses looked finite |
| Widening K_D past 1e-6 is useful for leakance (real-data training) | REFUTED | H1 REFUTED; K_D re-pinned at new ceiling in recoverability control |

---

## Part 8: Critical invariants that must not be broken

These are structural properties of the codebase. Breaking them invalidates the DDR parity guarantee.

1. **`compare_ddr_sandbox` must report ABSOLUTE MATCH** (max abs diff < 1e-3 m³/s). Run after any change to `src/routing/`, `src/geometry.rs`, or `src/sparse.rs`.
   ```bash
   cargo run --release --example compare_ddr_sandbox
   ```
2. **f32 throughout the routing core.** No casts to f64/bf16. DDR comparison sits at the f32 precision floor (~1e-7 rel diff per reach).
3. **Adjacency is topologically ordered and lower-triangular** (`rows[k] >= cols[k]`). The forward-substitution solver assumes it.
4. **Do NOT replace the hand-written sparse backward** in `src/sparse.rs`. The point is O(nnz) tape entries per timestep. Autograd-tape unrolling would be O(n²).
5. **KAN head is `rskan::KanLayer`** (v0.1.3 as of 2026-07-05). No inter-block ReLU. All `num_hidden_layers` inner KanLayers share the SAME seed (DDR quirk, preserved for parity).
6. **rskan is pinned to a git tag.** Bumping the tag requires re-running `tests/kan_head.rs` and the full parity sweep.

---

## Part 9: Fast symptom → root-cause lookup

| Symptom | Likely root cause | Do NOT do |
|---|---|---|
| Training loss flat at ~1.2 across all epochs | Gradient null-space (daily repeat-24 with daily aggregation) | Switch loss functions; it will not help without fixing the forcing path |
| K_D at ceiling on 100% of reaches | Optimizer compensates via driving head; box is not the constraint | Widen K_D range (H1 REFUTED) |
| zeta small everywhere | H2 + H4: head throttled by d_gw, gradient only reaches large rivers | KAN capacity tuning (H3 REFUTED) |
| Leakance recovery ratio ≈ 0 even with planted signal | ~130× hotstart-transient noise floor in windowed training | More training epochs, different seed |
| `use_cuda_graphs:true` shows finite loss but NaN eval | CUDA Graph replay returns stale cached scalar, not current forward | Trust finite printed loss as validity proof |
| Hourly and daily experiment cells byte-identical | Stale binary without disaggregation feature | Re-run without first refreshing `~/.cargo/bin/ddrs` |
| NSE beats baseline but KGE does not | Structural ceiling: routing over pre-UH-smoothed Q' cannot beat no-routing KGE | Balanced NNSE-KGE loss (tried 2026-06-24; did not help) |
| Leakance + use_cuda_graphs rejected at load | Config validation; correct behavior | Try to override |

---

## Part 10: Key run IDs (as of 2026-07-05)

| Run | Purpose | Result |
|---|---|---|
| `2026-06-23T02-49-12Z-conus-hourly-train-and-test` | BEST RESULT: precip disagg + L1 | NSE 0.7152 / KGE 0.7106 |
| `2026-07-01T13-43-32Z-train-and-test` | Leakance hourly-ON (2x2) | NSE 0.7145 / KGE 0.7150 |
| `2026-07-01T21-20-27Z-train-and-test` | Leakance daily-ON (2x2) | NSE 0.6963 / KGE 0.7250 |
| `2026-06-05T01-41-16Z-train-and-test` | Leakance daily-OFF control (2x2) | NSE 0.7004 / KGE 0.7244 |

All runs live at `/home/tbindas/projects/ddrs/.ddrs/runs/<run-id>/`.

---

## Provenance and maintenance

All findings in this document are sourced from the following files. Re-read them before updating this skill.

```bash
# Verify experiment results still match documented numbers
cat /home/tbindas/projects/ddrs/docs/6_19_26_journal.md
cat /home/tbindas/projects/ddrs/docs/2026-06-23-precip-disaggregation-findings.md
cat /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-findings.md
cat /home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-03-zeta-gradient-probe-findings.md
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-04-synthetic-recoverability-findings.md
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-04-leakance-literature-review.md
cat /home/tbindas/projects/ddrs/.claude/ARCHITECTURE.md

# Verify parity invariant is intact
cargo run --release --example compare_ddr_sandbox 2>&1 | grep -E "ABSOLUTE|max abs"

# Verify Phase B is still open (NOT YET MET as of 2026-07-05)
# Phase B target: windowed training floor ≤0.25 mean L1
# Current floor: ~1.017 step-0 loss on teacher-obs warm-start

# Check the gate program spec for what Phase B/C look like
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md | head -100
```

**Volatile facts (date-stamped 2026-07-05):**
- Best result: NSE 0.7152 / KGE 0.7106 (run `2026-06-23T02-49-12Z-conus-hourly-train-and-test`)
- Summed-Q' baseline: NSE 0.689 / KGE 0.723 (CONUS, matched gauge set)
- KGE does NOT beat baseline in any current config
- Phase B floor target (≤0.25 mean L1): NOT YET MET
- Leakance identifiability: NOT PROVEN (positive control FAILED)
- Most advanced branch: `origin/worktree-zeta-sensitivity`
- Current working branch: `unit_catchments`
