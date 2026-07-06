# Phase C — Leakance Promotion Gate Experiment

Date: 2026-07-05
Worktree: `zeta-sensitivity` (branch `worktree-zeta-sensitivity`)
Program spec: `docs/superpowers/specs/2026-07-04-leakance-gate-program-design.md` §5
Depends on: Phase A (channel/GW attributes, `merit_channel_attributes_v1.nc`),
Phase B (objective floor fix — state-cache hotstart), C0 (multi-store attributes).
Certification gate: the recovery-control-on-clean-objective result (R1); this
experiment proceeds on the machinery regardless, but its interpretation depends
on whether leakance is recoverable in principle.

---

## 1. Hypothesis

**H_C: When (a) the training objective is clean of initial-condition noise
(Phase B state cache), (b) the leakance head is given genuinely informative
groundwater/channel inputs (Phase A), and (c) leakance is trained in a staged
regime that prevents roughness cannibalization, then a sign-constrained
leakance term improves real-observation metrics (NSE/KGE) on losing-stream
gauges WITHOUT degrading them elsewhere and WITHOUT repainting the roughness
solution — and its learned field is consistent with independent groundwater
data.**

This is the culminating test of the whole program. Every prior experiment
removed one confound:
- 2×2 (2026-07-01): leakance's raw effect is marginal (+0.0018 KGE) and
  daily-forcing-harmful — but measured on a noisy objective with uninformed
  inputs.
- Diagnosis (2026-07-02): the marginality traces to driving-head starvation
  (`d_gw ≈ depth`) and lumped-conductance non-identifiability — the KAN had
  no groundwater signal to learn `d_gw` from.
- Gradient probe (2026-07-03): the gradient reaches every reach; the failure
  is signal detectability at the sensor, not gradient starvation.
- Recovery control (2026-07-04): planted leakance is unrecoverable from gauge
  discharge — but the objective carried a ~1.5 m³/s IC noise floor, ~200× the
  planted signal.
- Phase B (2026-07-05): the floor is fixed on the gauges where leakance lives
  (small/mid strata 1.5 → 0.11 m³/s); the residual mean floor is a
  continental-river magnitude artifact irrelevant to leakance.

Phase C is the first test with ALL confounds addressed simultaneously. If
leakance cannot clear the gate here, it cannot be promoted — and the
selective-equifinality result (a physically-motivated term that gauge
supervision genuinely cannot identify) is the paper's finding.

## 2. Experiment

### 2.1 What changes vs every prior leakance run

| Axis | Prior runs | Phase C |
|---|---|---|
| Objective | windowed L1, IC-noise floor ~1.5 | state-cache hotstart, floor ~0.11 on losing gauges |
| Inputs | 10 attributes, NO groundwater signal, `permeability` unused | + channel-WTD bed-relative, losing_fraction, corridor_impervious, alluvium, BFI, bankfull, `permeability` unlocked (via C0 multi-store) |
| Sign | free (gaining branch double-counts Q′ baseflow) | losing-only clamp `max(0, depth − d_gw)` |
| Lined channels | learned (error absorption) | structural zero where `corridor_impervious` > 0.7 |
| Training | cold, both params at once (equifinality) | staged two-head: roughness → freeze → leakance → joint fine-tune |
| Metric | mean L1 (mega-river dominated) | per-gauge NSE/KGE/FHV/FLV (magnitude-normalized) |

### 2.2 Code to build (spec §5 C1)

1. **Losing-only clamp** in `src/routing/leakance.rs`: `zeta = factor · area_z ·
   K_D · max(0, depth − d_gw)` — a differentiable relu on the head term.
   Gradient-exactness re-verified (`leakance_gradcheck` extended); the relu
   kink is subgradient-safe (measure-zero).
2. **Impervious hard-zero mask**: a static per-reach multiplier `zeta ← zeta ·
   (corridor_impervious ≤ 0.7)`, applied off the autograd-sensitive path (like
   the losing-only clamp), so no gradient flows to concrete-lined reaches.
3. **Staged two-head training** (spec §5 C2): a separate leakance KAN head so
   freezing roughness is a per-head optimizer choice, not gradient surgery.
   - Stage 1: routing head only, leakance OFF, fixed objective, enriched
     inputs, seed 42 → shared OFF cell + ON-cell frozen base.
   - Stage 2 (ON): freeze routing head; train leakance head only.
   - Stage 3 (ON): brief joint fine-tune (both heads, lower lr) so n relaxes
     out of stage-1 compensations while leakance holds the losing signal.
   - Equal-budget control: OFF cell trains the stage-2+3 epoch budget without
     leakance, so ON-vs-OFF is attributable to the term, not extra optimization.
4. **Gate analysis script**: computes the three legs (below) and prints
   pre-registered verdicts.

### 2.3 The three-leg gate (pre-registered, spec §5 C3)

Judged ON-cell vs OFF-cell, both on the fixed objective + enriched inputs:

| Leg | Test | Bar |
|---|---|---|
| 1 metrics | losing-subset median ΔNSE and ΔKGE (ON−OFF) | both ≥ +0.01 |
| 1 metrics | overall median NSE, KGE degrade | ≤ 0.002 |
| 1 metrics | median \|FHV\|, \|FLV\| worse on either gauge set | not worse |
| 2 equifinality | Δn(ON−OFF) per-reach IQR | < 0.1 (daily anti-pattern was 0.59) |
| 2 equifinality | spearman ρ(Δn, zeta_net) on nonzero-zeta reaches | \|ρ\| < 0.2 (or ≥ 0 if n relaxes rougher where leakance takes over — the physical direction) |
| 3 external | ρ(learned zeta magnitude, continuous bed-relative WTD) on nonzero-zeta reaches | > 0.3 (WTD threshold not a training target — non-circular) |
| 3 external | zeta ≈ 0 on lined-urban deep-WTD reaches (LA-River falsification set) | median \|zeta\| ≥ 5× below the losing-reach median |
| 3 external | magnitudes within Shanafield & Cook (2014) transmission-loss ranges | qualitative, reported |

### 2.4 Runs

- Forcing: hourly (the 2×2 showed daily hurts). CPU (`NdArray`), seed 42.
- Stage 1 shared → OFF cell = stage-1 model continued for the equal budget;
  ON cell = stage 2 + stage 3 from the frozen stage-1 base.
- Measurement: eval-with-zeta over the test window (seam-free eval, post the
  continuity fix) → per-reach `zeta`/`zeta_net`; `dump_parameters` for Δn.
- Real-obs eval (NOT synthetic) for the metric leg — this is the promotion
  question, judged against USGS gauges.

## 3. Expected outcomes

Three pre-registered scenarios and what each means:

### Scenario PROMOTE (all three legs pass)
Leakance improves losing-subset NSE/KGE by ≥ 0.01 without global harm, doesn't
cannibalize roughness, and its learned field aligns with independent
groundwater data. **Meaning:** the term earns default-on status under hourly
forcing; the negative history was objective noise + uninformed inputs, now
removed. The paper becomes "a physically-motivated routing term, correctly
supervised, improves prediction" — a positive differentiable-modeling result.
*Prior probability (my estimate): moderate-low.* The recovery-control R1 (due
imminently) is the leading indicator: R1 ≥ 0.5 makes PROMOTE plausible; R1 ≈ 0
makes it very unlikely.

### Scenario KILL (leg 1 or leg 2 fails)
Either leakance doesn't move the metrics even on the clean objective with good
inputs (leg 1), or it only "helps" by stealing roughness's job (leg 2).
**Meaning:** leakance is not promotable via gauge supervision. Combined with
the recovery control, this is the **selective-equifinality paper's central
result**: a term can be physically real, have live gradients, be given the
right inputs and a clean objective, and STILL be unidentifiable from gauge
discharge — because the signal is below the observational detectability band
(P3: 53× dilution). This is a stronger, more publishable finding than a
marginal metric win, because it's a measured impossibility result with every
confound controlled.

### Scenario REVISE (only leg 3 fails)
Leakance helps and doesn't cannibalize, but its learned field doesn't match
groundwater data — it's fitting residuals via a leakance-shaped knob rather
than representing the physics. **Meaning:** reparameterize toward the
Rushton (2007) bed/aquifer conductance split before any promotion; no promote
on a REVISE.

### The falsification lens (spec's cleanest test)
Regardless of scenario, the LA-River set is the sharpest single check: lined
urban channels sit over deep water tables, so WTD says "losing-possible" while
the bed says "sealed." If the trained head — given both `channel_wtd_bed_rel`
and `corridor_impervious` — learns zeta ≈ 0 there while staying positive on
alluvial losing reaches, that's direct evidence it represents physics, not
residual-fitting. If it can't make that distinction, that informs KILL.

## 4. What the imminent recovery-control result tells us first

The recovery control (running now) is the pre-flight for this experiment:
- **R1 ≥ 0.5** (planted leakance recovered on the clean objective): the
  objective fix worked; Phase C's PROMOTE scenario is live; proceed with
  confidence.
- **R1 ≈ 0** (still unrecoverable despite the clean objective): leakance is
  fundamentally unidentifiable from gauge discharge (detectability floor, not
  objective noise). Phase C will almost certainly KILL — and that IS the
  paper. We still run Phase C on real gauges to confirm the metric leg on
  observations (the recovery control uses synthetic obs), but the expected
  outcome shifts to KILL/selective-equifinality.

Either way the code (clamp, hard-zero, staged training, gate) is built and the
experiment runs — the recovery result sets the expected verdict, not whether
to proceed.

## 5. Reproduce (once built)

```bash
# Stage 1 (shared): routing head, leakance OFF, clean objective, enriched inputs
ddrs run --config config/experiments/phase_c_stage1.yaml
# Stage 2+3 (ON cell): freeze routing, train leakance head, joint fine-tune
ddrs run --config config/experiments/phase_c_on.yaml
# OFF cell: equal-budget continuation
ddrs run --config config/experiments/phase_c_off.yaml
# Gate
python scripts/phase_c_gate.py   # prints leg 1/2/3 verdicts + PROMOTE/KILL/REVISE
```

(Config names provisional; the implementation plan fixes them.)
