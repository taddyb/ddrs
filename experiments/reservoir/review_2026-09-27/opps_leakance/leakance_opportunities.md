# Leakance opportunities in light of the learned dam release

**Reviewer angle:** groundwater-surface water exchange (`params.use_leakance`) against the new dam-release
infrastructure on branch `dam-release-head` (commit `61a4a50`, read-only worktree `agent-a92e512a7c47c97b4`).
**Date:** 2026-09-27. **Scripts and outputs:** `/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_leakance/`
(`leak_vs_dams.py` -> `leak_vs_dams.json`, `leak_vs_base_per_gauge.csv`; `null_scale.py` -> `null_scale.json`;
`artifact_checks.py` -> `artifact_checks.json`). All analyses are reads of existing run outputs, no training.

## 0. Past conclusions this review depends on, and which it does not re-open

| Conclusion | Source | Status here |
|---|---|---|
| A gauge observes the network sum of zeta; per-reach flux is not identifiable from discharge alone | `research/findings/2026-07-06-leakance-nogo-scientific-summary.md` §3, §6 (structural argument, unaffected by the `K_D` bug) | **Depended on, not re-opened.** Every proposal below either tests something other than per-reach recovery, or adds supervision outside the gauge sum |
| The 2026-07-06 recovery ratio, noise floor and Phase C legs were measured with `K_D` frozen | research-status.md §Leakance, do-not-use list | Depended on: numbers not cited |
| Two-way leakance + learned n(d) arm (`2026-09-17T16-38-16Z`) failed both pre-registered water-table tests | `research/findings/2026-09-18-two-way-leakance-handoff.md` | Depended on; extended by the new measurements in §1 |
| Plains gauges see the conductance magnitude, not the water-table sign; the leverage tracks inflow-volume bias (Spearman +0.54); the arid West is an inflow problem | `research/findings/2026-09-22-regional-leakance-curvature-findings.md` §7 | Depended on: this is the pivot of opportunities A and D |
| Regulated-gauge deficit is mostly timing, partly volume; DOR > 0.5 gauges have median abs(1 - beta) 0.148 vs 0.095 | `research/findings/2026-09-25-reservoir-representation-options.md` §3 | Depended on |
| Learned release: dammed gauges +0.0014 / +0.0039 NSE (seeds 42, 43); undammed -0.0007 via the co-trained head; storage-conserving dam row; smoke set 458 + 458 | `<agent worktree>/research/findings/2026-09-27-learned-dam-release-findings.md` §6-7 | Depended on |

One caveat governs every leakance number in §1: the leakance arm `2026-09-17T16-38-16Z` differs from its nearest
no-leakance arm `2026-09-12T23-39-03Z` (sr_n0_gamma) in the gamma box as well ([-0.2, 0.85] vs [0, 0.5]; config
header of `config/experiments/sr_n0_gamma_leakance.yaml`), and the matched control was never run (handoff §Blockers
item 6). Paired differences below are therefore "leakance arm minus sr_n0_gamma", not "leakance minus no leakance".
The reach-level parameter reads (§1.3) do not have this problem.

## 1. New measurements on existing outputs

### 1.1 The two-way leakance arm is a near-uniform 2 % volume sink, not a spatial field

Per gauge, test window WY1996-2010, 2,365 gauges (`leak_vs_dams.json`):

| | leakance arm minus sr_n0_gamma | up / down |
|---|---:|---:|
| volume ratio (sum pred / sum obs), all | **-0.019 [-0.020, -0.018]** | 130 / 2,235 |
| volume ratio, gauges where sr_n0_gamma has a surplus (ratio > 1.05, n 1,151) | -0.024 | 39 / 1,112 |
| volume ratio, gauges where sr_n0_gamma has a **deficit** (ratio < 0.95, n 531) | **-0.010** | 81 / 450 |
| NSE, all | -0.0013 [-0.0016, -0.0008] | 1,030 / 1,335 |
| NSE, surplus gauges (dammed / undammed) | +0.0022 / +0.0010 | 254/174, 404/319 |
| NSE, deficit gauges (dammed / undammed) | **-0.0052 / -0.0045** | 36/167, 71/257 |

The two-way term (`leakance_losing_only: false`, `d_gw` box [-2, 3]) removed water at 94.5 % of gauges, including
85 % of the gauges that were already short of water. The "losing only fixes a surplus" asymmetry is therefore
measured directly: the arm helps where the inflow product over-delivers and hurts where it under-delivers, and the
two-way freedom did not turn into gaining reaches where they were needed. The learned field is 76.8 % losing by
reach (`share_losing_all`), net 194 m3/s over the 64,892 reaches of the scored network against 200 m3/s of
absolute flux, i.e. the gaining part is 3 % of the total.

Field spread (`plot/kan_parameters.nc`, scored-network reaches): `K_D` p10-p50-p90 = 1.04e-7, 1.19e-7, 1.37e-7 on
a [1e-9, 1e-5] box (geometric centre 1e-7: it never left initialization, as the handoff's |u_med - 0.5| = 0.020
already said); `leakance_factor` 0.61 / 0.86 / 0.95; `d_gw` 0.14 / 0.29 / 0.38 m. The zeta field is therefore
`constant x area_z x (depth - 0.3 m)`: its spatial pattern is the channel geometry, not anything learned.

### 1.2 A single global scale factor is a better volume corrector than the learned leakance field

Scaling sr_n0_gamma's routed predictions by one scalar (`null_scale.json`):

| scale | median NSE | paired dNSE median | up / down | Spearman with the leakance arm's per-gauge dNSE |
|---|---:|---:|---:|---:|
| 1.000 (sr_n0_gamma) | 0.7391 | | | |
| 0.981 (the leakance arm's median volume change) | 0.7406 | +0.0017 | 1,434 / 931 | 0.45 |
| 0.950 | 0.7438 | +0.0021 | 1,312 / 1,053 | 0.45 |
| leakance arm | 0.7300 | -0.0013 | 1,030 / 1,335 | |

One parameter, applied after the fact, beats the 3-output leakance head on the same gauges and the same window, and
its per-gauge signature correlates with the leakance arm's at 0.45. This is the cheapest decisive version of the
09-22 follow-up 3 ("separate leakance leverage from inflow-volume bias"), and it lands on the bias side.

### 1.3 Leakance did not localize on dam reaches: the extra dam-reach loss is geometry

Scored-network reaches, 1,024 NID dam COMIDs (>= 10 MCM) against all others, by decile of mean discharge
(`leak_vs_dams.json` `by_q_decile`, `null_scale.json` `dam_vs_other_by_q_decile`):

- Dam reaches lose 1.5 to 1.8x more of their flow than size-matched non-dam reaches (e.g. decile 7, q 3.9 m3/s:
  0.00093 vs 0.00052 of q_mean, Mann-Whitney p 2e-14); 94 % of dam reaches are losing vs 77 % overall.
- But the learned parameters are identical: `K_D` 1.21e-7 vs 1.21e-7, `leakance_factor` within 0.03, `d_gw` within
  0.02 m in every decile. What differs is `area_z / q_mean`, 1.5 to 2x larger on dam reaches (dam reaches are longer
  and wider per unit flow). The zeta excess at dams is `area_z`, not learning.

So the reservoir infrastructure does **not** remove a confound that corrupted the July measurements: the head never
distinguished dam reaches, and 1,024 reaches carrying 5.9 % of the network sink cannot have flipped a correlation
computed over 65k reaches (Phase C Leg 3, rho -0.355). The NO-GO does not depend on dams.

### 1.4 No gauge-level evidence that leakance absorbed dam effects (an artifact, caught)

The leakance arm's per-gauge dNSE correlates with the dam-release gain at dammed gauges (Spearman 0.30, p 5e-20)
and this rises to 0.59 after controlling for volume, area and base NSE. That is a shared-base artifact: the same
correlation is **0.41 at undammed gauges**, where the release cannot act, and becomes **-0.02 (dammed) and -0.38
(undammed)** against the seed-43 dam-release run, which has a different base (`artifact_checks.json`). Do not read
any correlation between two per-gauge deltas that share a base run.

What survives: the leakance arm's dNSE at dammed vs undammed gauges differs by 0.0005 (Mann-Whitney p 0.067);
dam minus matched control on the 458 smoke pairs +0.0023 (262 / 196), +0.0033 at DOR > 0.5 (129 / 85); by DOR,
-0.0035 at DOR <= 0.1 rising to +0.0012 at DOR 1-2. A small concentration at heavily regulated gauges, with the
gamma-box confound unresolved.

### 1.5 Heavily regulated gauges do carry a volume surplus, and it exceeds their matched controls

sr_n0_gamma volume ratio by NID degree of regulation (`null_scale.json` `vol_base_by_dor`) and paired against the
smoke controls (`artifact_checks.json`):

| DOR | n | median ratio | share > 1.05 | share < 0.95 | paired log-ratio, dam minus control (n) |
|---|---:|---:|---:|---:|---:|
| undammed | 1,448 | 1.050 | 0.50 | 0.23 | |
| <= 0.1 | 275 | 1.035 | 0.42 | 0.14 | -0.024 (74) |
| 0.1-0.5 | 337 | 1.021 | 0.40 | 0.24 | -0.033 (170) |
| 0.5-1 | 140 | 1.053 | 0.51 | 0.25 | -0.002 (98) |
| 1-2 | 87 | **1.119** | 0.67 | 0.28 | **+0.060** (63, dam more surplus in 54 %) |
| > 2 | 78 | **1.136** | 0.60 | 0.32 | +0.026 (53) |

Above one year of storage the model delivers 12 to 14 % more water than the gauge sees, 2.6 to 6 % more than the
matched undammed control in the same HUC2 (which carries the same inflow-product bias). That is the size of a
reservoir loss (Santa Rosa releases 0.72 of inflow, reservoir-options §2.3), and it is the population a dam-row
loss term can address. Below DOR 0.5 dammed gauges have **less** surplus than controls; nothing to remove there.
Note the deficits: 28 to 32 % of high-DOR gauges are short of water (imports, e.g. Abiquiu), which no loss term
can fix and which a two-way term would "fix" by manufacturing water.

## 2. Answers to the five questions

**(1) Leakance and reservoirs together.** Engineering is small for fixed `T` and moderate for the learned release.
`forward_chain_inner` (`src/routing/mmc_op.rs`) already takes the `reservoir: Option<&ReservoirTensors>` argument
and applies S19'' before the coefficients; `timestep_forward_leakance` passes `None` and sets
`reservoir_mask: None` in its saved state, and `MuskingumCunge::route_timestep` asserts `self.reservoir.is_none()`
on the leakance branch. `timestep_backward_core` already masks `gk_muskingum` and `gx` on dam rows from
`state.reservoir_mask` (B19''), so carrying the mask through `TimestepLeakanceOp` / `TimestepLeakanceGammaOp` is
plumbing plus a gradcheck. For the learned release the leakance-gamma op gains the two `T` parents (`T_t`,
`T_{t+1}`) of `TimestepReleaseGammaOp`, so 11 parents, and `timestep_backward_core`'s B19''' read-out is reused.
Two things are not plumbing (see §4): zeta on a dam row is computed from a Manning depth the row no longer uses,
and the mass bound `alpha * relu(b_base)` on a dam row with `c3` near 1 permits removing 25 % of the stored water
per hour. Both need a dam mask on zeta (the impervious-mask slot exists). **Does the reservoir remove a confound?**
No, on the evidence of §1.3 and §1.4: the leakance head never distinguished dam reaches, and the earlier NO-GO does
not depend on dams. Leakance and the release compete only for volume at high-DOR gauges, and there the release is
storage-conserving and cannot help, so the competition is between leakance (a smeared sink upstream) and a dam-row
loss (a point sink). That is question (2).

**(2) Reservoir evaporation / withdrawal as a loss at dam rows.** This is the one leakance-like term whose
identifiability is structurally better: the sink has one known location, a directly gauged one in 377 of 917 dammed
gauges, its magnitude has an external expectation (PET x NID surface area, both on disk), and the matched-control
design measures the inflow-product bias the sink would otherwise absorb. §1.5 gives the target population (165
gauges above DOR 1, 12 to 14 % surplus) and the control-corrected size (3 to 6 %). It belongs in the release head,
not in leakance: a fourth output `e_d` (loss rate on storage, so `E = e T Q`), losing-only, bounded. Implementation
is the storage-conserving row with `(1 + e_d T)` multiplying the outflow term: only the `Q` coefficients change,
`e = 0` is bitwise the current row.

**(3) Matched controls and paired seeds as the clean leakance identifiability test.** Yes, and it is the first
time the design exists: 916 gauges, 458 pairs, about 1 h CPU per arm, two seeds already run for the release. The
test that matters is not skill but the sign of the volume change against the sign of the base bias (§1.1), and
the read-out must be on arms that change one factor. The 09-17 arm changed the gamma box, the `K_D` box, the
`d_gw` box and turned on two-way at once.

**(4) Where leakance vs inflow-bias correction should live.** Inflow-bias correction lives upstream of routing, on
`q'`, as a one-parameter (global) or per-HUC2 scalar, and it is the null model every loss term must beat; §1.2
shows it beats the current leakance field outright. Leakance should be restricted to what only it can express:
stage-dependent, sign-changing exchange, tested on the plains gauges where the loss sees `K_D` (09-22 §3.2) and
scored against the scalar null. Dam losses live in the release head.

**(5) What the new work shows is wrong in the leakance code.** §4.

## 3. Ranked opportunities

Ranking is (expected gain x confidence) / cost. Gains are paired median dNSE on the population named.

| # | Opportunity | Expected gain | Confidence | Cost | Cheapest decisive test |
|---|---|---|---|---|---|
| A | Global / per-HUC2 `q'` volume scalar as the mandatory null model for any loss term | +0.002 population; settles the leakance volume question | high (§1.2 already shows it) | trivial (offline done; a 1-parameter learnable scalar is an afternoon) | done offline; training confirmation 2 smoke arms |
| B | Dam-row loss `e_d` (evaporation/withdrawal) as a 4th release-head output | +0.01 to +0.02 at DOR > 1 gauges (165), about +0.001 population | medium (surplus is there, §1.5; ceiling bounded by the deficit half) | moderate: coefficient change on dam rows + 1 parent + gradcheck + head output, about the size of the storage-conserving fix | 2 smoke arms x 2 seeds, dam-minus-control at DOR > 1, plus the PET x surface-area check |
| C | Fix the leakance initialization: gate off at init, or L1 on sum of zeta (sparse solution) | unknown; could turn the smeared field into a localized one or turn leakance off honestly | medium-low | low (config for the gate; a loss term for L1) | synthetic planted-reach recovery with `K_D` free, R1 >= 0.5 bar |
| D | Single-factor leakance test on the smoke set: {leak off, on} x {dams off, learned}, 2 seeds, same boxes | resolves the confound in every 09-17 number; sign-of-volume-change read-out | high that it is decisive; low that leakance wins | 8 arms x 1 h CPU | the 8 arms; go/no-go on dVOL sign agreement and beating A |
| E | Leakance + reservoirs coexistence (mask plumbing, dam-masked zeta, dam-masked bound) | enabling only | high | low for fixed T, moderate for learned | gradcheck + `leakance_off_parity` + smoke check 1a |
| F | Dam completion as a natural experiment: before/after volume ratio at the 153 gauges whose dam was built inside 1981-2010 | calibrates B's `e_d` prior with no dam data as input | medium | trivial (existing zarr outputs) | the before/after ratio at 153 gauges vs the same gauges' controls |

### A. The scalar null model (rank 1)

- **Evidence.** §1.2: scaling sr_n0_gamma by 0.95 gives median NSE 0.7438 vs 0.7391 (1,312 / 1,053 up / down);
  the leakance arm gives 0.7300. The 09-22 census: leakance-only gain correlates with the baseline volume error at
  +0.54, and 36 of 38 "more losing" gauges have inflow above observed.
- **Gain.** Small in skill, large in what it settles: no leakance or evaporation result can be credited until it
  beats one parameter.
- **Cost.** Offline, done. In the engine: a learnable scalar (or per-HUC2 vector) multiplying `q'` before
  `setup_inputs`, log-space, init 1; it is not a routing-core change (`q'` is already an autodiff parent).
- **Risk.** It absorbs exactly the bias the loss wants absorbed; that is its job as a null. It must not be shipped
  as physics.
- **Test.** Smoke set, two arms, seed 42: sr_n0_gamma recipe with and without the scalar. **Go** for anything
  downstream: leakance (D) or `e_d` (B) must beat this arm's paired dNSE with the 95 % interval clear. **No-go**
  for leakance's volume role if the scalar arm is within the interval of the leakance arm.

### B. Dam-row loss in the release head (rank 2)

- **Evidence.** §1.5: DOR 1-2 gauges run 12 % surplus, +6 % against matched controls (n 63); Santa Rosa 0.72
  release ratio; reservoir-options §3 "perfect volume" ceiling at DOR > 0.5 is +0.058 NSE, of which the surplus
  half (§1.5, 60 to 67 %) is reachable by a sink. The withdrawal term added nothing at Raystown and Abiquiu
  (§2.5), which is the expected humid-region null and a built-in negative control.
- **Form.** `dS/dt = I - Q - e S` with `S = T Q`: the storage-conserving trapezoid becomes
  `T_{t+1} Q_{t+1} - T_t Q_t = dt [ (I_t + I_{t+1})/2 + q' - (1 + e T_{t+1}) (Q_t + Q_{t+1})/2 ]` (or `e` at both
  ends), i.e. `c1 = c2 = dt / (2 T_{t+1} + dt (1 + e T_{t+1}))`, `c3 = (2 T_t - dt (1 + e T_t)) / (...)`, `c4`
  likewise. `e = 0` is the current row bit for bit; `e` in `[0, e_max]` with `e_max T <= 1` keeps `c3 >= 0` at
  `T >= dt`. Losing-only by construction (no manufactured water). One more parent per dam row (`e`, or `e T`
  folded into a modified `T` parent pair). Head: `Linear(H, 4)`, `reservoir_e` range in log space; add
  `surface_km2` and the dam reach's `aridity` / `meanP` (the brief's optional v1 inputs) since evaporation scales
  with them.
- **Gain.** +0.01 to +0.02 at the 165 DOR > 1 gauges; population +0.001. Modest, but with an external check the
  leakance term never had.
- **Cost.** Moderate: the same shape as the storage-conserving change (one day, per the findings §6), plus a
  gradcheck case and `release_params.csv` column.
- **Risk.** A point sink at the dam still absorbs the upstream network's inflow surplus; the matched control is
  what detects that (controls carry the same bias and cannot lose water). Also a slow leak on a large `T` looks
  like a small `T` on the gauge (both cut the release), so `e` and `T` are partially degenerate; the seasonal
  phase and the PET check separate them.
- **Test.** Smoke set, arms {learned release, learned release + `e_d`} x seeds {42, 43}, 4 h CPU. Read-outs:
  (i) dam minus matched control dNSE at DOR > 1 pairs (63) and at all 458; (ii) the volume ratio at dam gauges
  moves toward 1 while controls' does not; (iii) Spearman(learned `e_d` x surface area, PET x surface area)
  across active dams. **Go:** (i) > +0.01 with CI clear at DOR > 1, (iii) > +0.3. **No-go:** `e_d` uniform across
  dams (p10-p90 within a factor of 2, the §1.1 signature) or (iii) <= 0: it is another bias absorber.

### C. Sparse leakance: gate off at init, or L1 on zeta (rank 3)

- **Evidence.** §1.1: `K_D` p10-p90 within 30 % of the box centre, factor 0.86, the field is the initialization
  plus geometry. Handoff item 3: the gate anneal froze decisions by epoch 30. The July mechanism (§4 of the NO-GO)
  is that the optimizer selects a smeared minimum-norm field; the standard alternative for a sum-observation
  inverse problem is a sparsity prior, which was never tried.
- **Gain.** Either localization (which would be the first positive leakance result) or an honest zero field.
- **Cost.** Low: `leakance_gate` init bias so gate is 0 at epoch 1 (needs the Concrete noise back or a
  straight-through estimator, handoff item 5); an `l1_zeta` weight on `sum |zeta|` in `src/training/loss.rs`.
- **Risk.** L1 turns the term off everywhere (then A wins, and that is the answer).
- **Test.** The 2026-07-04 synthetic recovery harness (58 planted reaches, clean objective) with `K_D` free and the
  sparse variant. **Go:** R1 >= 0.5 (the campaign's bar) and the planted reaches hold > 50 % of the flux.
  **No-go:** R1 < 0.1 again: close the leakance parameterization question for good.

### D. Single-factor leakance test on the smoke set (rank 4)

- **Evidence.** §0 caveat: the 09-17 arm is confounded; §1.1 gives the read-out that would have been decisive had
  the control existed (volume removed at deficit gauges).
- **Design.** Same boxes throughout (sr_n0_gamma_leakance's), arms {leak off, leak on} x {dams off, learned
  release}, seeds 42 and 43: 8 x 1 h CPU. Requires E for the two combined cells (or drop them: 4 arms).
- **Read-outs.** (i) dVOL sign vs base-bias sign per gauge: share of gauges where leakance moves volume toward the
  observed; (ii) paired dNSE dam minus control; (iii) `|u_med - 0.5|` for `K_D` and the p10-p90 of zeta/q.
- **Go:** (i) > 65 % and the arm beats A. **No-go:** (i) near 50 % or below (the 09-17 arm: about 30 %, since it
  removed water at 85 % of deficit gauges): leakance is a bias term, record it, and stop training it on skill.

### E. Coexistence plumbing (rank 5, enabling)

Carry `reservoir` into `timestep_forward_leakance` and `reservoir_mask` into `TimestepLeakanceState`; zero zeta on
dam rows by folding the dam mask into the impervious mask (`LeakanceTensors::mask`), which also zeroes the mass
bound there; for the learned release, an 11-parent `TimestepLeakanceReleaseGammaOp`. Replace the
`validate_reservoirs` rejection ("the leakance timestep op has no linear-reservoir K/X override") with the
CUDA-graphs-style rejection only. Gates: `leakance_gradcheck`, `leakance_gamma_gradcheck` extended with dam rows,
`leakance_off_parity`, `reservoir_release_gradcheck`, smoke check 1a. Note the completion-year activation makes
dam rows time-varying; a static dam mask on zeta over-masks before completion, which is acceptable for v1 and
should be stated.

### F. Dam completion as a natural experiment (rank 6, cheap)

`experiments/reservoir/full_run/dam_age_check.json`: 112 gauges' dams were completed during the train window and
41 during test. At those gauges the obs/pred volume ratio before and after completion, minus the same difference
at their matched controls, is the dam's net loss (evaporation + withdrawal - imports) measured with observations
only. It gives B a prior and an out-of-sample check, and it tests whether the leakance arm's field responds to a
sink that switches on. No training; an afternoon on the existing prediction zarrs.

## 4. What the current leakance code the new work shows is wrong or should change

1. **Zeta on a dam row uses a fictitious channel depth.** `zeta_forward` takes `depth` from the shared Manning
   power law (S6) of the dam reach; S19'' replaces that reach's routing with `K := T, X := 0`, so the depth has no
   meaning on the row. With coexistence, zeta must be masked on dam rows (or replaced by B's `e_d`).
2. **The mass bound is wrong on a dam row.** `|zeta| <= alpha * relu(b_base)` with `b_base = c2 I + c3 Q + c4 q'`
   and `c3 -> 1` at large `T` bounds the loss by the *stored* water, so `alpha = 0.25` permits removing a quarter
   of the reservoir per hour. The bound was designed for channel rows where `b_base` is an hour's throughflow.
3. **The field is the initialization.** `K_D` sits at the box centre (p10-p90 1.04e-7 to 1.37e-7 on a four-decade
   box) after 50 epochs; the gate anneal (`leakance_gate.temperature` 1.0 -> 0.05) freezes the factor by epoch 30
   (handoff item 3). The box-centre-is-the-initialization finding (`2026-09-17-box-centre-is-the-initialization`)
   applies: the term should start off, not at a uniform 1.2e-7.
4. **Two-way exchange did not produce gaining reaches where water was missing** (§1.1: volume removed at 85 % of
   deficit gauges). The `d_gw` box [-2, 3] with `bed_thickness` 1.0 leaves `d_gw` at 0.14 to 0.38 m, all
   connected and all below stage at any depth above 0.4 m; the sign never flips in practice. Either the two-way
   option is dropped (losing-only is honest about what the term does) or the head is given a reason to raise the
   table (the plains gauges of 09-22 §7.1 prefer raising it 10 : 4).
5. **Experimental hygiene.** The 09-17 arm changed four things at once and its three matched controls were never
   run; every paired number from it, including those in §1, is confounded by the gamma box. The smoke set fixes
   the cost of doing this properly (1 h per arm).
6. **The `collect_zeta` diagnostic** writes `zeta_net_mean` per scored reach but not the bound's binding share; the
   pre-registration listed "the bound binding on a large share of reach-timesteps" as a reason to distrust a pass,
   and it cannot currently be read. Add the binding fraction to the per-reach diagnostic.

## 5. Caveats

- All gauge-level leakance numbers carry the gamma-box confound (§0). The reach-level parameter comparison (§1.3)
  and the scalar null (§1.2) do not.
- The volume ratio here is sum(pred)/sum(obs) over WY1996-2010 on days with both finite; "surplus" is > 1.05 and
  "deficit" < 0.95.
- DOR is NID normal storage over mean annual flow (`nid_dor` in `nid_dams_by_gauge.csv`); the 165 high-DOR
  gauges include the arid West, where the inflow product over-delivers at undammed reference gauges too (09-22
  §7.2). The matched-control column of §1.5 is the population-corrected number; use it, not the raw ratio.
- The smoke pairs are loosely area-matched (README known issue 1); dam gauges are larger, and volume ratio may
  depend on size.
- Nothing here re-measures per-reach recovery; the structural non-identifiability stands and every proposal is
  either a null model, a point sink with an external check, or a sparsity prior tested on the synthetic harness.
