---
name: ddrs-identifiability-campaign
description: "Use when planning or executing the leakance identifiability gate, the selective-equifinality paper experiments, the Phase B objective-floor fix, or the Phase C leakance promotion experiment. Triggers: any task touching leakance training, the synthetic recoverability control, the gradient probe, the zeta-sensitivity worktree, the three-phase gate program, or the paper at /home/tbindas/projects/ddr_equifinality/paper.tex. Also use as the source-of-truth for what each phase's gate criteria are, which experiments have already run, and which commands reproduce the results."
---

# ddrs identifiability campaign — executable runbook

**As of 2026-07-05.** All facts in this document are verified from source files.
Volatile sections are date-stamped. Metric numbers are NOT invented.

---

## Glossary (for PyTorch engineers who don't know Rust/BURN)

| Term | Definition |
|---|---|
| **ddrs** | BURN-based Rust port of DDR (Python/PyTorch). BURN is Rust's deep-learning framework. The port is gradient-exact against DDR. |
| **DDR** | Python/PyTorch differentiable Muskingum-Cunge routing model. Reference lives at `~/projects/ddr/`. |
| **Muskingum-Cunge (MC)** | Physics-based 1D routing method. For each reach: `Q_{t+1} = C1·I_{t+1} + C2·I_t + C3·Q_t + C4·q'_t`. Coefficients depend on learned geometry and roughness. |
| **Q' (q-prime)** | Lateral inflow from a land-surface/LSTM model — the upstream forcing that MC routes through the network. |
| **zeta (ζ)** | Leakance flux: `zeta = leakance_factor · area_z · K_D · (depth − d_gw)`. Subtracted from the RHS at each timestep. Positive = losing reach. |
| **KAN head** | Kolmogorov-Arnold Network (via `rskan::KanLayer` v0.1.3). Maps per-reach attributes → routing parameters. Replaces MLP. |
| **CSR / sparse backward** | The routing solve is a triangular sparse linear system. ddrs uses a hand-written O(nnz) backward — do NOT replace with autograd tape unrolling. |
| **rho window** | Training samples random sub-windows of length `rho` days. Warmup = first N days discarded from the loss. |
| **CONUS** | Contiguous US MERIT network: 346,321 reaches × 338,814 edges. |
| **eval network** | Union of gauge subgraphs used in eval: 64,892 reaches (subset of CONUS). |
| **worktree** | Git worktree at `.claude/worktrees/zeta-sensitivity/` (branch `worktree-zeta-sensitivity`). Contains all leakance probe/recoverability binaries and configs. |
| **floor** | Windowed training loss at the teacher-weights point. The noise due to hotstart-transient initial conditions. |

---

## When NOT to use this skill

- **Routine train/eval runs** with no leakance and no paper work: use `ddrs plan`/`ddrs run` directly.
- **Sparse solver or autograd questions**: see `.claude/references/ddrs-burn-autograd.md`.
- **Architecture/porting questions**: see `.claude/ARCHITECTURE.md` and `~/projects/ddr/CLAUDE.md`.
- **CLI lifecycle questions**: see `docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md`.

---

## Campaign overview (state as of 2026-07-05)

Two parallel tracks share a common "objective floor fix" dependency:

```
TRACK 1: LEAKANCE IDENTIFIABILITY GATE
  2x2 (leakance × forcing)          DONE — GO marginal (2026-07-01)
  Diagnosis (H1–H7)                 DONE — H2/H4/H5 SUPPORTED (2026-07-02)
  Gradient probe (P1/P2/P3)         DONE — P3 detectability NO-GO (2026-07-03)
  Positive control (recoverability) DONE — FAILED (2026-07-04) ← BLOCKER
  Phase B: objective floor fix      NOT STARTED (target: ≤0.25 mean L1)
  Phase C: promotion gate           BLOCKED on Phase A + Phase B

TRACK 2: SELECTIVE EQUIFINALITY PAPER
  Paper draft                       IN PROGRESS — /home/tbindas/projects/ddr_equifinality/paper.tex
  Thesis: geometry identifiable, n is bias-absorber
  Four inflow sources needed: daily-LSTM, hourly-LSTM, dHBV2 lumped, dHBV2-UH
  Results section: NOT YET WRITTEN (no four-model comparison run exists yet)
```

---

## Invariants — break these and every result is invalid

Before any code change, confirm these still hold:

```bash
# Invariant 1: DDR parity gate (run after ANY change to src/routing/, src/geometry.rs, src/sparse.rs)
cargo run --release --example compare_ddr_sandbox
# Expected output: "ABSOLUTE MATCH" (max abs diff < 1e-3 m³/s on 5-reach sandbox)

# Invariant 4: leakance gradient exactness (run after ANY change to src/routing/leakance.rs)
cargo test --test leakance_gradcheck
cargo test --test leakance_off_parity
cargo test --test zeta_accum
```

| # | Invariant | Why it matters |
|---|---|---|
| 1 | `compare_ddr_sandbox` ABSOLUTE MATCH (max abs < 1e-3 m³/s) | Port correctness vs Python reference |
| 2 | f32 throughout routing core | Mixed precision breaks DDR comparison at f32 floor |
| 3 | Adjacency lower-triangular, topologically ordered | Forward-sub solver assumption |
| 4 | Hand-written sparse backward in `src/sparse.rs` | O(nnz) tape entries per timestep, not O(n²) |
| 5–6 | KAN head = rskan v0.1.3, no MLP placeholder | DDR parity on head forward/backward |

---

## STALE-BINARY TRAP (read before any experiment)

`cargo build` does NOT update `~/.cargo/bin/ddrs`. If you edit `src/` and type
`ddrs run`, you silently run the old binary. This invalidated the first 2x2 runs
(2026-07-01 morning) — the installed binary had no disaggregation and the hourly
and daily cells were byte-identical.

After any `src/` change, run ONE of:

```bash
cargo install --path .                                        # canonical
# or faster if target/release is current:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
# or bypass installed copy entirely:
cargo run --release --bin ddrs -- run --workflow train-and-test
```

**Self-check**: current checkpoints are DIRECTORIES
(`.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/head.mpk`).
Flat files (`epoch_E_mb_M.mpk`) mean you ran a stale binary.

---

## Track 1: Leakance identifiability gate

### 1.1 The leakance term

```
zeta_i = leakance_factor_i · area_z_i · K_D_i · (depth_i − d_gw_i)
area_z  = (p · depth)^q_eps · length      (plan-view wetted area, m²)
b_rhs  ← b_rhs − zeta                     (positive = losing reach)
```

Implementation: `src/routing/leakance.rs`, analytical backward via
`TimestepLeakanceOp: Backward<I,8>`.

**Config requirements (all three required together):**

```yaml
params:
  use_leakance: true
  # use_cuda_graphs must be false — config load REJECTS the combination
  use_cuda_graphs: false
  parameter_ranges:
    K_D:              [1.0e-8, 1.0e-6]   # log-space, 1/s
    d_gw:             [-2.0, 2.0]        # m
    leakance_factor:  [0.0, 1.0]         # dimensionless
kan_head:
  learnable_parameters: [n, q_spatial, p_spatial, K_D, d_gw, leakance_factor]
```

### 1.2 The 2x2 experiment (DONE — results verified, 2026-07-01)

Four arms: daily-OFF, daily-ON, hourly-OFF, hourly-ON. Seed 42, eval
1995/10/01–2010/09/30, 2,365 finite-NSE CONUS gauges.

**All-gauge medians:**

| arm | NSE | KGE | delta-NSE vs OFF | delta-KGE vs OFF |
|---|---|---|---|---|
| hourly-OFF | 0.7153 | 0.7104 | — | — |
| hourly-ON  | 0.7145 | 0.7150 | −0.0008 | **+0.0046** |
| daily-OFF  | 0.7004 | 0.7244 | — | — |
| daily-ON   | 0.6963 | 0.7250 | −0.0041 | +0.0006 |

**Losing-stream subset (79.6% of gauges, mean pred/obs > 1 under hourly-OFF):**

| arm | ΔNSE med | ΔKGE med | frac(ΔNSE>0) |
|---|---|---|---|
| hourly ON−OFF | **+0.0005** | **+0.0018** | **55.5%** |
| daily  ON−OFF | −0.0017 | −0.0009 | 35.6% |

**Identifiability check:**

| param | median | pinning |
|---|---|---|
| `K_D` (1/s)      | 1.003e-6 | **100% at ceiling** — wants more exchange |
| `leakance_factor` | 0.327 | interior (0.12–0.53) — not collapsed |
| `d_gw` (m)       | 0.294 | interior (−0.02–0.78) |

**GO/NO-GO gate (3 of 3 passed, verdict: GO — marginal):**

- [x] Gate 1: skill gain on losing subset (ΔNSE +0.0005, ΔKGE +0.0018 under hourly)
- [x] Gate 2: effect absent/weaker under daily (ΔNSE −0.0017)
- [x] Gate 3: |zeta| > 0.01 m³/s on **10.4%** of 64,892 eval reaches (bar: ≥10%, no headroom)

Reproduce:

```bash
cd ~/projects/ddr
uv run python ~/projects/ddrs/scripts/leakance_subset_analysis.py \
  --hourly-on  2026-07-01T13-43-32Z-train-and-test \
  --daily-on   2026-07-01T21-20-27Z-train-and-test \
  --hourly-off 2026-06-23T02-49-12Z-conus-hourly-train-and-test \
  --daily-off  2026-06-05T01-41-16Z-train-and-test \
  --ddrs-runs-dir /home/tbindas/projects/ddrs/.ddrs/runs
```

### 1.3 Low-zeta diagnosis H1–H7 (DONE, 2026-07-02)

The question: why is median |zeta| only 6.4e-4 m³/s when K_D is pinned at ceiling?

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | Structural K_D ceiling clips zeta | **REFUTED** — median in-box ceiling utilization is 3.4%; K_D is not the limiting factor |
| H2 | Driving-head starvation (`d_gw` ≈ depth) | **SUPPORTED** — median driving head 0.02 m; 47% of reaches effectively gaining at mean depth |
| H3 | KAN variance collapse | **REFUTED** — `d_gw` and `leakance_factor` show spatially non-trivial fields |
| H4 | Gauge bias / gradient starvation | **SUPPORTED (re-mechanized)** — zeta–uparea ρ = +0.76; gauged median |zeta| 11× ungauged; but the mechanism is signal starvation at the sensor, not gradient death (see P1/P2 below) |
| H5 | Equifinality absorption | **SUPPORTED** — Manning's n absorbs the attenuation leakance would provide |
| H6 | Wrong yardstick | **REFUTED** — fractional loss |zeta|/Q also tiny |
| H7 | Disconnection / model-form error | **REFUTED** — `d_gw` does not strain bounds; disconnection not the dominant regime |

**Phase-3 gate (K_D widening retrain): NO-GO.** H1 is refuted — the box is not
the bottleneck. Widening K_D will not fix identifiability. See §1.4.

### 1.4 Gradient probe P1/P2/P3 (DONE, 2026-07-03)

Instruments: `probe_zeta_gradient` binary in
`.claude/worktrees/zeta-sensitivity/`. Stage 1 = adjoint reachability map
(96 training-style windows on CPU, deterministic). Stage 2 = planted
delta-flux detectability at 104 measurement gauges.

| # | Hypothesis | Pre-registered bar | Measured | Verdict |
|---|---|---|---|---|
| P1 | Starvation | gauged/ungauged |g| ≥ 10× at both checkpoints | **1.5× trained, 2.9× cold** | **REFUTED** |
| P2 | Rejection | >67% dry-tercile gradients push zeta down | **52.5%** (≈ neutral at trained point) | **REFUTED** |
| P3 | Detectability | ≥10% of Ref δ=0.01 probes detectable | **4.2% of 96 Ref probes** (δ=0.1: 16.7%) | **NO-GO** |

**P3 decomposition**: a 0.01 m³/s loss transmits to its nearest gauge at 94.6%
fidelity, but the median Ref gauge's 5% observational band is 0.531 m³/s — the
signal is 53× smaller than what any discharge objective can distinguish from
measurement noise. Detection fails on dilution, not transmission.

**Discovered mechanism (post-hoc, labeled as such)**: at the cold point 80.5%
of ungauged gradients push zeta DOWN, with 15.9× wet/dry magnitude asymmetry.
Early training suppresses leakance before equilibration. The trained sign map is
physically coherent (interior West / High Plains want more leakance) but cannot
be rewarded further.

**Conclusion**: gauge-only discharge supervision cannot learn real-world
leakance. Auxiliary spatial supervision is the only viable path.

Reproduce:

```bash
WT=/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity
# Stage 1 (CPU, ~35 min per run)
nice -n 10 $WT/target/release/probe_zeta_gradient \
  --config /home/tbindas/projects/ddrs/config/experiments/leakance_hourly_on.yaml \
  --checkpoint /home/tbindas/projects/ddrs/.ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9 \
  --windows 96 --seed 42 --output /home/tbindas/projects/ddrs/output/zeta_probe/grad_trained.nc
# Stage 2 verdicts
cd /home/tbindas/projects/ddrs/ddrs-py
uv run python $WT/scripts/zeta_gradient_analysis.py
```

### 1.5 Positive control — synthetic recoverability (DONE, FAILED, 2026-07-04)

**This is the critical blocker as of 2026-07-05.**

The experiment planted known-magnitude losses (58 sites, median 7.21e-2 m³/s
— 41× background) in a teacher world with zero observation noise and
warm-started students from the teacher's own weights, then measured whether
training recovered the planted flux.

**Pre-registered gate: PASS iff R1 ≥ 0.5 AND run A beats run B on final loss.**

| # | Metric | Bar | Measured | Verdict |
|---|---|---|---|---|
| R1 | Recovery ratio median (n=58) | ≥ 0.5 | **0.009** | **FAILED** |
| R2 | Non-planted |zeta_net| A/baseline | < 2 | 1.11 | PRECISE (trivially, since R1 fails) |
| R3 | Final-epoch mean loss A vs B | A < B by >5% | A=1.339, B=2.317, +42.2% | A<B — CONFOUNDED |
| R4 | Δn absorption map | descriptive | Δn IQR planted ≈ Δn all = −0.019 | absorbed globally |
| R5 | Cold emergence ratio | > 3 | 1.20 | SUPPRESSED |

**HEADLINE: FAIL.**

**Root cause — the windowed training objective's hotstart-transient noise floor:**

| Quantity | Value |
|---|---|
| Continuous residual (teacher weights + teacher obs, full-window eval) | 0.00759 mean L1 |
| Step-0 windowed training loss (run A) | 1.017 mean L1 |
| Ratio (noise floor / planted signal) | **~130×** |
| Run A continuous residual AFTER 5 epochs of training | 0.4431 (58× WORSE than initial) |

The synthetic obs were generated by continuous routing with fully developed
storage. A warmup of 5 days trims far too little — big rivers carry memory of
tens to hundreds of days. The planted signal (0.8% of the training loss) is
invisible. Adam then actively degrades the model.

**Implication: leakance identifiability is NOT proven.** The positive control
must pass before any identifiability claim. Phase B (state-cache hotstart,
target ≤0.25 mean L1) is required.

### 1.6 Phase B: objective floor fix (NOT STARTED, 2026-07-05)

**Goal**: reduce the windowed training objective floor from ~40% of a converged
run's loss to ≤25% (≤0.25 mean L1). This unblocks Phase C.

**Two candidate approaches:**

| Option | Description | When to choose |
|---|---|---|
| A (config-only) | Longer warmup and/or rho | If floor(warmup=60, rho=180) ≤ 0.25 mean L1 |
| B (state-cache) | Cache a continuous run's daily reach states over the training window (~64,892 × 5,113 f32 ≈ 1.3 GB); initialize each training window from the cached state at its start date | If option A fails |

**B1 step (floor-vs-warmup curve, forward-only, no training required):**

```bash
# Uses recoverability experiment assets
WT=/home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity
# Teacher weights + teacher obs exist at:
#   /home/tbindas/projects/ddrs/output/recoverability/synthetic_obs
#   /home/tbindas/projects/ddrs/.ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9
# Run teacher-weights windowed loss at warmup in {5, 15, 30, 60} days
# (forward-only, CPU, ~15 min total for all four)
# Decision: if warmup ≤ 60 @ rho ≤ 180 reaches ≤ 0.25 mean L1 → option A
```

**B3 bar (pre-registered)**: fixed-objective floor ≤ 0.25 mean L1 (≤10% of
converged loss, vs ~40% today). CPU rerun-noise = 0 (check persists).

This fix is general — it applies to ALL ddrs training, not just leakance.

### 1.7 Phase C: promotion gate (BLOCKED on Phase A + Phase B)

Phase A = attribute extraction in `extractrs` (channel-corridor + GW attributes).
Phase B = objective floor fix (above).
Phase C = the actual gate experiment.

**C2 training design (user decision 2026-07-04) — STAGED, two-head:**

```
Stage 1 (shared): routing head only, leakance OFF, hourly, fixed objective,
                  enriched inputs, seed 42.
                  → This IS the OFF cell's result AND the ON cell's frozen base.

Stage 2 (ON only): freeze routing head; train leakance head alone
                   (losing-only clamp: max(0, depth − d_gw);
                    impervious hard-zero mask from corridor_impervious > 0.7)

Stage 3 (ON only): brief joint fine-tune (both heads, few epochs, lower lr)
                   Equal-budget control: OFF cell continues same epoch count
                   without leakance.
```

K_D range for Phase C: `[1e-8, 1e-5]` (widened from original `[1e-8, 1e-6]`;
the 2x2 pinned K_D at 1e-6 and the recoverability sites needed [1e-8, 1e-5]
to achieve adequate expressibility).

**C3 gate (pre-registered — do not revise after C2 runs):**

| Leg | Test | Bar |
|---|---|---|
| 1 — metrics | losing-subset median ΔNSE, ΔKGE (ON−OFF) | both ≥ **+0.01** (5× the 2×2 effect) |
| 1 — metrics | overall median NSE, KGE | degrade ≤ 0.002 |
| 1 — metrics | median |FHV|, |FLV| | not worse on either gauge set |
| 2 — equifinality | Δn(ON−OFF) per-reach IQR | < 0.1 (daily anti-pattern was 0.59) |
| 2 — equifinality | spearman ρ(Δn, zeta_net) on nonzero-zeta reaches | |ρ| < 0.2 |
| 3 — external | ρ(learned zeta magnitude, continuous bed-relative WTD) | > 0.3 |
| 3 — external | zeta ≈ 0 on lined-urban deep-WTD reaches (LA-River set) | median |zeta| ≥ 5× below losing-reach median |
| 3 — external | magnitudes vs Shanafield & Cook 2014 transmission-loss ranges | qualitative, reported |

**Decision rule:**

- **PROMOTE**: all 3 legs pass → leakance default-on in hourly configs
- **KILL**: leg 1 or leg 2 fails → documented NO-GO, stays experimental
- **REVISE**: only leg 3 fails → reparameterize (Rushton 2007 direction), no promotion

---

## Track 2: Selective equifinality paper

**Paper**: `/home/tbindas/projects/ddr_equifinality/paper.tex`
**Title**: "Beyond Equifinality in Differentiable River Routing" (Bindas, Shen)
**Thesis**: equifinality is selective — channel geometry is identifiable, Manning's n is a bias-absorber.

**Test design**: train ddrs with four structurally different lateral inflow sources on the same MERIT network, attributes, and observations. Parameters that converge across sources are identifiable; those that diverge are compensatory.

**Required four inflow sources:**

| Source | Config group | Streamflow store |
|---|---|---|
| daily-LSTM | `config/sources/daily-lstm.yaml` | `/mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic` |
| hourly-LSTM | `config/sources/hourly-lstm.yaml` | `/mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic` |
| dHBV2 lumped | (not yet configured) | TBD |
| dHBV2-UH | `conus` source group (MERIT dHBV2 UH) | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` |

**Current baseline metrics (CONUS, 2,365 gauges, 1995/10–2010/09):**

| Configuration | NSE | KGE | Notes |
|---|---|---|---|
| Summed-Q baseline, same-run (2,365-gauge eval set) | 0.678 | 0.717 | source: docs/2026-06-23-precip-disaggregation-findings.md |
| Precip-driven disagg + L1 (best result as of 2026-06-23) | 0.715 | 0.711 | hourly-OFF arm |
| daily-OFF (flat repeat-24) | 0.700 | 0.724 | |

> **Note:** `ddrs plan` prints a global plan-cache baseline of NSE 0.689 / KGE 0.723 (5,224 gauges). The same-run baseline above (0.678 / 0.717) is the correct comparison for the 2,365-gauge eval set. Do not mix the two when computing deltas.

**KGE does NOT beat the summed-Q baseline in any configuration as of 2026-07-05.**
NSE beats it by +0.037 (0.678 → 0.715) with precip-driven disaggregation.

**Research questions (verified from paper.tex):**

1. Do physically realized channel geometries (depth, top width, hydraulic radius) converge across four independent lateral inflow sources, even when raw parameters (p, q) do not?
2. Does Manning's roughness absorb systematic biases from lateral inflow models, and can roughness divergence be predicted from inter-model inflow disagreement?
3. Can convergence of physically derived quantities under input perturbation serve as a general identifiability test?

**To run a four-source comparison:**

```bash
# 1. Confirm sources are mounted:
ls /mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic
ls /mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic

# 2. Run each arm (same seed, same eval window, same attributes):
ddrs sources use daily-lstm
ddrs plan --workflow train-and-test
ddrs run --workflow train-and-test
# Repeat for hourly-lstm, dHBV2-lumped, dHBV2-UH

# 3. Compare: dump_parameters on all four checkpoints, then compare
# per-reach n, p_spatial, q_spatial distributions and realized geometry
# at reference discharge.
```

---

## Baseline and reference numbers

| Metric | Value | Source |
|---|---|---|
| Summed-Q median NSE (CONUS) | 0.689 | `ddrs plan` cached baseline |
| Summed-Q median KGE (CONUS) | 0.723 | same |
| Best trained NSE (precip-disagg + L1, 2026-06-23) | 0.715 | run `2026-06-23T02-49-12Z-conus-hourly-train-and-test` |
| Best trained KGE (precip-disagg + L1, 2026-06-23) | 0.711 | same |
| CONUS MERIT reaches | 346,321 | fabric |
| CONUS eval network reaches | 64,892 | gauge-subgraph union |
| CONUS training gauges (finite-NSE) | 2,365 | same run |

---

## Wrong paths — do not go here

| Temptation | Why it fails |
|---|---|
| Widen K_D to fix the low-zeta problem | H1 is REFUTED — the K_D box is not the bottleneck. Widening changes expressibility but not identifiability. The recoverability experiment already used [1e-8, 1e-5] and still failed (R1 = 0.009). |
| Use `use_cuda_graphs: true` with leakance | Config load REJECTS this combination. Hard error. |
| Run a new leakance experiment before Phase B is done | The objective floor is ~130× the reach-scale signal. Any metric improvement under the current objective is uninterpretable for identifiability. |
| Claim identifiability from 2x2 GO-marginal alone | The positive control (recoverability) failed. GO-marginal from skill metrics is necessary but not sufficient for identifiability. |
| Set `use_leakance: false` in the config but keep K_D/d_gw/leakance_factor in `learnable_parameters` | Config validation only rejects `use_leakance: true` + `use_cuda_graphs: true`. The B-student in the recoverability experiment used this pattern deliberately (leakance-OFF head with ON architecture) — verify intent before doing this. |
| Run `ddrs` binary without reinstalling after `src/` changes | STALE-BINARY TRAP. The 2026-07-01 first-pass runs were all invalid for this reason. |
| Replace the sparse backward with autograd tape unrolling | Breaks O(nnz) tape invariant. Invariant 4. |

---

## Active experiments and checkpoints

| Experiment | Checkpoint / output | Status |
|---|---|---|
| hourly-ON (2x2) | `.ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9` | **Reference checkpoint for all leakance work** |
| hourly-OFF (best trained run) | `.ddrs/runs/2026-06-23T02-49-12Z-conus-hourly-train-and-test/checkpoints/epoch_5_mb_35` | Best overall model as of 2026-07-05 |
| Recoverability teacher obs | `output/recoverability/synthetic_obs` | Used by Phase B B1 step |
| Recoverability answer key | `output/recoverability/answer_key.nc` | 58 planted sites, answer-key zeta_net |
| Recoverability verdicts | `output/recoverability/logs/verdicts.log` | R1=0.009, FAIL |
| Gradient probe | `output/zeta_probe/` | P3 NO-GO; detectability_rows.csv |

---

## Checklist: starting a new leakance experiment

- [ ] Check for stale binary: `ddrs --version` matches expected SHA, or reinstall
- [ ] Config has `use_cuda_graphs: false` if `use_leakance: true`
- [ ] K_D range is `[1e-8, 1e-5]` for any new leakance work (not the old [1e-8, 1e-6])
- [ ] Leakance gradcheck still passes: `cargo test --test leakance_gradcheck`
- [ ] DDR parity gate: `cargo run --release --example compare_ddr_sandbox` → ABSOLUTE MATCH
- [ ] Phase B floor target (≤0.25 mean L1) is met — if not, results are uninterpretable for identifiability

## Checklist: before merging anything to master

- [ ] `cargo test` passes
- [ ] `cargo run --release --example compare_ddr_sandbox` → ABSOLUTE MATCH
- [ ] No `use_cuda_graphs: true` + `use_leakance: true` in any tracked config
- [ ] Any new leakance backward changes pass: `leakance_gradcheck`, `leakance_off_parity`, `zeta_accum`
- [ ] Invariants 1–6 intact (see table above)

---

## Provenance and maintenance

Source files read to produce this skill (re-read to verify facts before updating):

```bash
# Paper
cat /home/tbindas/projects/ddr_equifinality/paper.tex | head -300

# 2x2 findings
cat /home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-findings.md

# Recoverability (the critical FAIL)
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-04-synthetic-recoverability-findings.md

# Gradient probe
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/2026-07-03-zeta-gradient-probe-findings.md

# Phase A/B/C gate program design
cat /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md

# Source groups
cat /home/tbindas/projects/ddrs/config/sources/daily-lstm.yaml
cat /home/tbindas/projects/ddrs/config/sources/hourly-lstm.yaml
```

Re-verification commands (run before updating any metric in this file):

```bash
# Confirm 2x2 metrics still reproduce
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/leakance_subset_analysis.py \
  --hourly-on 2026-07-01T13-43-32Z-train-and-test \
  --daily-on  2026-07-01T21-20-27Z-train-and-test \
  --hourly-off 2026-06-23T02-49-12Z-conus-hourly-train-and-test \
  --daily-off  2026-06-05T01-41-16Z-train-and-test \
  --ddrs-runs-dir /home/tbindas/projects/ddrs/.ddrs/runs

# Confirm recoverability verdicts still reproduce
cd /home/tbindas/projects/ddrs/ddrs-py
uv run python /home/tbindas/projects/ddrs/.claude/worktrees/zeta-sensitivity/scripts/recoverability_analysis.py

# Confirm DDR parity gate
cd /home/tbindas/projects/ddrs
cargo run --release --example compare_ddr_sandbox
```

Date-stamped facts in this file: all metrics are as of 2026-07-05. The
Phase B objective floor fix had NOT been started as of that date. The paper
results section had NOT been written as of that date.
