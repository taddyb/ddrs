# Handoff: two-way leakance + learned n(d) — built, run, and refuted

**Date:** 2026-09-18
**Next focus:** read the 400-gauge landscape census when it lands, write the findings
doc for the registered failure, and decide whether `d_gw`'s orthogonality to roughness
is worth one more arm or whether leakance closes for good.

## Context

`ddrs` is a BURN/Rust differentiable Muskingum-Cunge router over CONUS MERIT, feeding
an AGU talk and a paper responding to Beven's 2001 Dalton lecture. The user's framing
question: can a leakance / n(d) model (a) learn physically realistic parameters,
(b) quantify inflow bias, (c) improve predictions, (d) speak to equifinality, scale and
uncertainty. This session made leakance and stage-dependent roughness able to coexist
for the first time, fixed three real physics defects found along the way, ran a full
50-epoch CONUS arm, and the pre-registered read-out **failed**.

Everything is on branch **`leakance-gamma-coexist`**, 18 commits ahead of `origin/master`,
nothing pushed, no PR.

## What was done this session

- **Nine-parent operator** `TimestepLeakanceGammaOp` so leakance and a LEARNED gamma can
  coexist; previously `validate_learned_gamma` rejected the pair because the eight-parent
  op had no gamma slot. Commit `daad5f3`. Gradcheck `tests/leakance_gamma_gradcheck.rs`.
- **Disconnection cap** `head = min(depth - d_gw, depth + M)` (`leakance_bed_thickness`),
  the MODFLOW RBOT behaviour. Commit `d16eb7d`.
- **Mass bound** `|zeta| <= alpha * relu(b_rhs_base)` (`leakance_max_rhs_fraction`),
  symmetric. Commits `5fef057`, `47d0eaf`.
- **Two-way exchange** (`leakance_losing_only: false`) plus `d_gw` box `[-2, 3]`. Commit `47d0eaf`.
- **`K_D` box fixed** to `[1e-9, 1e-5]` after a NaN. Commit `e2f7b96`; findings
  `research/findings/2026-09-17-box-centre-is-the-initialization-findings.md`.
- **Pre-registration** written BEFORE the result: `research/findings/2026-09-17-two-way-leakance-preregistration.md`,
  commit `8e96886`. Read-out script `experiments/leakance/validate_water_table.py`.
- **Landscape can now measure leakance arms**: carries the leakance fields (commit `6a81588`),
  and `K_D`/`d_gw` are sweepable axes (commit `83d6805`).
- Corrected two of my own published numbers: the no-leakance negative-discharge baseline
  (commit `166a539`), and a false claim that the July campaign never tested external
  agreement (CLAUDE.md).

## Current state

**The CONUS arm finished**: `.ddrs/runs/2026-09-17T16-38-16Z-train-and-test`, 50 epochs,
2,365 gauges, **zero NaN**, status `ok`. Median NSE **0.7300**, KGE **0.7567** against a
summed-Q' baseline of 0.6781 / 0.7172, i.e. **+0.052 NSE**. CUDA OOM panics appear in the
eval-phase log (trap T6) but did not corrupt the metrics or the parameter dump.

**Both pre-registered tests FAILED.**

| test | prior arm | this arm | bar |
|---|---|---|---|
| magnitude partial corr, all 10 head inputs | +0.027 | **+0.003** | > +0.10 FAIL |
| sign agreement | 60.8% | **42.1%** | > 60.8% FAIL |
| Matthews correlation | +0.139 | **-0.081** | > 0 FAIL |
| regime partial corr | +0.011 | **-0.087** | > +0.10 FAIL |

The model calls 30.3% of reaches gaining; the reference says 60.8%. Three of the four
patterns registered as reasons to distrust a result are present: `d_gw` spans only
0.14-0.38 m (p90-p10 = 6.9% of its box) against a reference spanning -4.46 to +2.18;
`K_D` sits at its box centre (|u_med - 0.5| = **0.020**, median 1.21e-7 vs geometric
centre 1e-7); and skill improved while the partial correlation did not.

**Landscape results, `epoch_30_mb_9`, Juniata pair, Hessian at the TRAINED point:**

- gamma has **no interior optimum**: pins at the wall at `alpha_max` 1.5 AND 4.605.
  `corr(n, gamma)` = **-0.947 / -0.732** — near-singular Hessian, i.e. §31's "one latent
  direction relabelled as two", measured.
- `corr(n, d_gw)` = **+0.028 / +0.015** — `d_gw` is essentially ORTHOGONAL to roughness.
  This is the one result that does not repeat the established pattern.
- `K_D` has **negative** diagonal curvature and pins at the search ceiling: same failure
  mode `q_spatial` showed in §29. The 18-30% loss gains ride a boundary.

**Still running:** `landscape-twoway-strat400`, the 400-gauge census, started 04:45 UTC,
~89/400 done when this was written. Log:
`/tmp/claude-1000/-home-tbindas-projects-ddrs/621f68a3-62a4-44cd-a0ad-e64c2089f38e/scratchpad/landscape-twoway-strat400.log`

## Blockers / open questions

1. **Leakance is refuted again, on a sharper test than before.** The honest reading is that
   `d_gw` is not recovering a water table. Do NOT run more leakance arms scored on skill.
2. **The one live thread is `d_gw`'s orthogonality to n** (+0.015 to +0.028), on two gauges.
   If the census does not support it, leakance closes.
3. **The gate anneal freezes decisions.** `d(gate)/d(u)` = 3.2e-8 at u=0.1, tau=0.1, so
   gate decisions were fixed by epoch ~30 and the last 20 epochs could not revise them.
   Our gate is deterministic Binary Concrete with the noise REMOVED, which is what supplies
   exploration in the literature (Maddison et al. 2017; Jang et al. 2017).
4. **`params.tau` (u32, hour offset, =9) and the gate temperature are different things**
   sharing a name under `params:`. Renaming the gate field to `sharpness` is on the list;
   not done because it would change config hashes mid-run.
5. **Landscape alpha is a log multiplier** `phys = x0*exp(alpha)`, so it can never reach
   NEGATIVE gamma even though the box now admits `[-0.2, 0.85]`. The negative-gamma
   direction is untested, and the audit found gamma pinned at the old floor of 0 on one product.
6. **The three matched controls were never run** (`*_ctl_neither`, `*_ctl_gamma`,
   `*_ctl_leakance`). Judged not blocking: the summed-Q' baseline is the control for skill,
   and the validation tests are absolute, not comparative.
7. **The reference field is itself a model** (`channel_wtd_bed_rel`, a national groundwater
   model interpolated to channels). "Wrong" means disagreeing with another model.

## Next steps

1. **Read the census** when `ALL_LANDSCAPES_DONE` appears in `scratchpad/landscapes_all.log`.
   The question: does `corr(n, d_gw)` stay near zero across 400 gauges, and does `K_D` keep
   negative curvature? Compare against §37.4's 14-gauge gamma census (median |H_gg|/|H_nn| = 0.024).
2. **Write `research/findings/2026-09-18-two-way-leakance-findings.md`**, citing the
   pre-registration commit `8e96886`, and state the failure plainly. Update the
   `ddrs-dev` do-not-use list.
3. **Fill the journal entries** for the landscape runs (`python3 scripts/journal.py --mode status`).
   NOTE: `--mode backfill` corrupted the journal on 2026-09-16; do not run it.
4. **Decide on leakance.** The evidence now says the term does not learn a water table.
   The interesting residue is that `d_gw` occupies a different information channel from
   roughness, which is a *structural* result worth reporting even though the *parameter*
   is not recovered.
5. If more work is wanted: restore the Concrete noise or switch to a straight-through
   estimator so gate decisions are not frozen by the schedule.

## Suggested skills

- `/ddrs-eval-plots` — reading the finished run's parameter maps and eval output
- `/ddrs-journal` — the landscape runs have open entries; rules for writing them
- `/ddrs-dev` — `references/research-status.md` before citing any number externally
- `/explain-plot` — the user wants hypothesis / plain-English / graph / result / conclusion, in that order, for every figure

## Key file references

| Path | Why it matters |
|---|---|
| `research/findings/2026-09-17-two-way-leakance-preregistration.md` | The bars this run was judged against, written before the result |
| `experiments/leakance/validate_water_table.py` | The read-out: magnitude partial correlation + the sign test |
| `research/findings/2026-09-17-box-centre-is-the-initialization-findings.md` | Why the first run NaN'd; the box centre IS the initialization |
| `research/findings/2026-09-08-landscape-hypothesis-tests-findings.md` | §29 (q), §31 (one latent direction), §37.4 (gamma census). Read before re-measuring anything |
| `src/routing/mmc_op.rs` | `TimestepLeakanceGammaOp`, `leakance_backward_body`, the mass-bound split of `gb_rhs`. Invariant-4 code |
| `src/routing/leakance.rs` | `zeta_forward`/`zeta_backward` with the disconnection cap; `cap_tests` |
| `src/experiment/landscape/objective.rs` | Carries leakance fields; `AXIS_PARAMS` now 6. Had a third copy of the PR #46 denormalize bug, fixed |
| `config/experiments/sr_n0_gamma_leakance.yaml` | The arm that ran. Every box choice is justified inline |
| `.ddrs/runs/2026-09-17T16-38-16Z-train-and-test/` | The finished run: metrics, checkpoints, `plot/kan_parameters.nc` |
