# Phase C Implementation Plan — Leakance Promotion Gate

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Sonnet subagents only. Steps use `- [ ]`.

**Goal:** Build and run the leakance promotion gate: losing-only clamp + impervious hard-zero + staged two-head training + three-leg gate, on the clean objective with enriched inputs.

**Experiment doc:** `docs/2026-07-05-phase-c-leakance-gate-experiment.md`
**Spec:** `docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md` §5
**Worktree:** `zeta-sensitivity`; CPU-only; seed 42; guard every code task with `leakance_gradcheck`, `leakance_off_parity`, `zeta_accum`, `compare_ddr_sandbox` ABSOLUTE MATCH.

**Prereqs already done:** C0 (multi-store attributes), Phase A (`merit_channel_attributes_v1.nc`), Phase B (state cache), continuity fixes, disagg lookahead.

---

### Task 1: Losing-only clamp in the leakance flux

**Files:** `src/routing/leakance.rs`, `tests/leakance_gradcheck.rs`

- [ ] Change `zeta = factor · area_z · K_D · (depth − d_gw)` → `... · max(0, depth − d_gw)` (differentiable relu on the head term only). Read the current `_compute_zeta` / `TimestepLeakanceOp` forward+backward FIRST; the relu changes both the forward and the analytical `∂zeta/∂d_gw` (zero where `depth ≤ d_gw`). Subgradient at the kink = 0 (measure-zero, FD-safe).
- [ ] Add a config flag `params.leakance_losing_only: bool` (default **true** for Phase C; keep the un-clamped path behind `false` for exact back-compat with prior runs / the recovery control's answer key).
- [ ] Extend `leakance_gradcheck` to cover the clamped branch on a gaining chain (`depth < d_gw` → zeta ≡ 0, gradient ≡ 0) AND a losing chain (unchanged from today). Finite-difference must match analytical on both sides of the kink.
- [ ] Guards green. Commit `feat(routing): losing-only leakance clamp max(0, depth-d_gw)`.

### Task 2: Impervious hard-zero mask

**Files:** `src/routing/leakance.rs` (or the setup path), config, a test

- [ ] Static per-reach multiplier `zeta ← zeta · mask` where `mask = (corridor_impervious ≤ threshold)`, threshold from `params.leakance_impervious_threshold` (default 0.7). The mask is a constant tensor (no grad), applied like the losing-only clamp — off the autograd-sensitive path. `corridor_impervious` arrives as a per-reach input via C0's attribute concat; thread it to the routing setup as a constant vector (NOT a KAN output).
- [ ] Test: a reach with impervious > 0.7 gets zeta ≡ 0 and zero gradient to its leakance params; a reach below threshold unchanged. Verify on the LA-River falsification COMID if a fixture allows, else a synthetic mask.
- [ ] Guards green. Commit `feat(routing): impervious hard-zero mask for leakance`.

### Task 3: Two-head architecture (routing head + leakance head)

**Files:** `src/nn/kan_head.rs` (or a new `leakance_head.rs`), `src/training/forward.rs`, config

- [ ] Split the single KAN head into a ROUTING head (n, q_spatial, p_spatial + disagg) and a separate LEAKANCE head (K_D, d_gw, leakance_factor). Config `kan_head.leakance_head:` block (own hidden_size/layers/inputs — leakance head takes the GW/channel inputs; routing head keeps its current inputs). Both consume the C0-concatenated attribute matrix, selecting their own `input_var_names` subset.
- [ ] `forward`/`forward_eval` build both heads; leakance params come from the leakance head. When `use_leakance: false`, the leakance head is not constructed (byte-identical to a routing-only run).
- [ ] KAN-parity guard: the routing head alone (leakance off) must match the current single-head init/forward exactly (the split must not perturb roughness). Add a parity test.
- [ ] Guards + kan_head tests green. Commit `feat(nn): split routing and leakance KAN heads`.

### Task 4: Per-head freezing in the optimizer

**Files:** `src/training/optimizer.rs` / `driver.rs`, config

- [ ] `experiment.freeze: [routing_head]` (or `[leakance_head]`) — a config list naming heads whose parameters get zero gradient this run (detach or zero-grad before the optimizer step). Read how the Adam optimizer iterates params; freezing = exclude those param groups from the update (their grads may compute but aren't applied). Default empty (all trainable).
- [ ] Test: with `freeze: [routing_head]`, a training step changes leakance-head params but leaves routing-head params bit-identical.
- [ ] Guards green. Commit `feat(training): per-head parameter freezing`.

### Task 5: Phase C configs (staged recipe)

**Files:** `config/experiments/phase_c_{stage1,on,off}.yaml`

- [ ] Derive from `leakance_hourly_on.yaml` + these deltas: `data_sources.attributes` = LIST [global v2, channel v1] (C0); `experiment.state_cache` = the teacher/real state cache (Phase B); enriched `input_var_names` for the leakance head (channel_wtd_bed_rel, losing_fraction, corridor_impervious, alluvium_fraction, bfi, permeability, bankfull_depth); `leakance_losing_only: true`; `leakance_impervious_threshold: 0.7`.
  - `phase_c_stage1.yaml`: `use_leakance: false`, routing head only, N epochs.
  - `phase_c_on.yaml`: `use_leakance: true`, `experiment.checkpoint` = stage-1 output, `freeze: [routing_head]` for stage 2, then a stage-3 sub-config unfreezing both at lower lr (or a `stage3` epochs block).
  - `phase_c_off.yaml`: `use_leakance: false`, `checkpoint` = stage-1, equal stage-2+3 epoch budget (the control).
- [ ] Validate all parse; each `input_var_names` entry resolves across the two attribute stores (C0 hard-errors otherwise — good).
- [ ] NOTE: a real state cache for the REAL-obs training window must be generated (the recovery run's cache is for synthetic obs). Add a step: generate `state_cache_real.nc` via `--mode state-cache` on the real-obs config before stage 1. (~2.5 h; can run while code tasks proceed.)
- [ ] Commit `config: Phase C staged gate configs`.

### Task 6: Run the staged experiment

- [ ] Real-obs state cache (Task 5) done → Stage 1 (routing head, ~85 min) → Stage 2 (freeze routing, leakance head) → Stage 3 (joint fine-tune) → OFF cell (equal budget). All CPU, seed 42, tee'd, guarded.
- [ ] Measurement: seam-free eval-with-zeta on the REAL test window (post-continuity-fix `evaluate`) for ON and OFF → per-reach zeta/zeta_net; `dump_parameters` ON vs OFF for Δn.

### Task 7: Three-leg gate analysis

**Files:** `scripts/phase_c_gate.py`

- [ ] Leg 1: per-gauge NSE/KGE/FHV/FLV ON vs OFF, losing-subset (reuse the 2×2 subset) + overall; bars per experiment doc §2.3.
- [ ] Leg 2: Δn(ON−OFF) IQR + spearman(Δn, zeta_net) on nonzero-zeta reaches (with the directional note).
- [ ] Leg 3: spearman(|zeta|, channel_wtd_bed_rel); LA-River falsification-set zeta vs losing-reach median; magnitudes vs Shanafield & Cook ranges.
- [ ] Print PROMOTE / KILL / REVISE per the pre-registered logic. Commit `feat(scripts): Phase C three-leg gate`.

### Task 8: Findings + paper hook

- [ ] `docs/2026-07-XX-phase-c-findings.md` in the experiment-report structure; connect the verdict to the selective-equifinality narrative (`ddr_equifinality/paper.tex`). Guards green. Commit.

---

## Self-review
- Spec §5 C1 (clamp) → Task 1; hard-zero → Task 2; C2 staged two-head → Tasks 3/4/5/6; C3 gate → Task 7; C0 dependency → Task 5 config uses the list. Real-obs state cache flagged as a NEW prereq (Phase B's cache was synthetic-obs).
- Back-compat: `leakance_losing_only` and the two-head split both have `false`/off paths preserving prior behavior + the recovery answer key.
- Right-sizing: if the recovery R1 ≈ 0 (KILL expected), Tasks 3/4 (staged two-head) are still built but the experiment's PROMOTE path is unlikely; the code is reusable and the real-obs metric leg still needs running to confirm on observations. Do not skip — the gate verdict must be measured, not assumed.
