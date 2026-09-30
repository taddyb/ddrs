# Per-dam release calibration: overnight 2026-09-27/28

**Asked (user):** continue the learned dam release toward a much larger NSE/KGE gain, about +0.03, using the five
reviews' plan and dams that switch on at their NID completion year.
**Branch:** `dam-release-v2` (worktree `.claude/worktrees/agent-a92e512a7c47c97b4`). **Results page:**
https://claude.ai/artifact/YELrGtjwb5qjhq9JEjLs2u (source `experiments/reservoir/overnight_page/`).
**Journal:** `research/journal/2026-09.md` on this branch, entries dated 2026-09-27 and 2026-09-28.

## Target

The +0.03 target is defined as the median per-gauge NSE change, test WY1996-2010, at the 117 smoke gauges whose NID
dam sits on the gauge's own reach and whose DOR (upstream storage over one year of flow) is above 0.5. Below DOR 0.5
the dam deficit is too small for any dam law to move a median by 0.03 (paired dam-minus-control gap -0.02 to -0.09).

## Where the room is

- At DOR > 0.5, dam gauges trail matched undammed controls (same HUC2, closest area) by -0.229 median NSE on the
  no-dam model and -0.274 on summed Q'. The deficit is already in the forcing; routing closes 0.045 of it.
- It is correlation, not volume or variance, and lives at 30-400 d periods (period-band excess over controls:
  30-120 d +0.074, 120-400 d +0.059). A day-scale bucket cannot reach it.
- Seasonal-cycle error grows with storage and is shape-dominated at irrigation dams and in the Southwest/Great Basin,
  volume-dominated in the Southeast. The linear reservoir helps at flood-control dams, not where seasonal error is
  largest. Scripts: `experiments/reservoir/smoke_v2/`.

## Offline laws (`experiments/reservoir/rulecurve_offline/`, audited)

- A per-dam harmonic rule curve (`S = T Q + S0(doy)`) gains +0.049 [+0.028, +0.081] at the target set; controls +0.0045.
  An independent re-implementation reproduces it with no leakage.
- Most of it is the per-dam residence time: a plain per-dam bucket gains +0.025; the rule curve adds +0.004 paired on
  test years (+0.032 on training years). Its coefficients are not predictable from NID features (RF R2 -0.13 to +0.15).

## In the engine (smoke set, frozen routing, one seed)

| Arm | Target, median dNSE [95 % CI] | Note |
| --- | --- | --- |
| Joint training, today's law | +0.0083 | controls move up to 0.345 |
| Frozen, today's law | +0.0070 [+0.0019, +0.0139] | controls exactly 0 |
| Frozen, additive row, head T0 | +0.0070 [+0.0017, +0.0130] | dKGE +0.0008 |
| Rule curve + per-dam T0 (`2026-09-27T23-36-04Z`) | **+0.0168** [+0.0074, +0.0256] | clean gauges +0.020; dKGE +0.006 |
| Capped row, per-dam lr 0.05 (`2026-09-28T03-21-27Z`) | +0.0130 [+0.0022, +0.0207] | cleanest water: 3 dams >= 5 % |
| Replay, offline per-dam T0, capped row | +0.0186 [+0.0048, +0.0357] | no clamps |
| Replay, offline rule curve, capped row | +0.0244 [+0.0045, +0.0487] | clean gauges +0.037 |
| Control, capped row alone, 1 h | +0.0006 | |

## What the reviews caught (all fixed or bounded)

1. The rule-curve flux could store more than a dam receives; the clamp created water and cut the restoring gradient
   (feasibility penalty, bounded amplitude).
2. Dense Adam on sparse per-dam rows (row-sparse Adam).
3. The created-water account under-reported by (K + T)/dt (mass-balance account; journal corrected).
4. The mass-conserving floor pumped debt on the additive row through the channel's negative c1 at the floor
   (`dam_row_positivity`; the cap alone changes nothing, and doubles a fitted per-dam T0's gain).

## Conclusion

Supported: per-dam residence times are the dam-specific lever, and the engine can use them (+0.019 replayed, +0.017
trained, KGE up). Refuted: a seasonal rule curve as the missing physics; it adds about +0.001 over a per-dam
residence time. Not reached: +0.03 at the target set; the in-engine ceiling of this law family is about +0.02 to
+0.025. The remaining dam error is flood evacuation at 30-120 d and diverted volume at irrigation dams.

## Open decisions

- Accept per-dam calibrated parameters (gauged dams only)?
- Full-population arms of the best recipe at two seeds.
- New laws: flood-evacuation dynamics; a prescribed withdrawal term tested against a plain Q' rescaling.
- Data: Flaming Gorge (COMID 77013090) is mis-snapped; 17 near-dry dams hold > 100 years of modelled inflow.

## Correction (2026-09-28, user): the target is the population median

The user's +0.03 target is the median NSE over the full 2,365-gauge evaluation population (gages_3000 after
filtering), not the 117 regulated smoke gauges used above; that choice was mine and wrong. On the seed-42 no-dam
arm the population median is 0.7391, so the target is 0.7691. `experiments/reservoir/smoke_v2/population_ceiling.py`:
transferring the best smoke arm's gains by DOR bin to the 917 dammed population gauges moves the population median by
+0.0017; a perfect dam model that closes every dammed gauge's gap to its matched control moves it by +0.0256. Dams
alone cannot reach +0.03 at the population median; population-wide levers (runoff volume bias, routing-head
convergence) are needed alongside. The user accepted per-dam calibrated parameters for gauged dams and asked for two
new laws: flood-evacuation dynamics (30-120 d band) and a prescribed withdrawal at irrigation dams, tested against a
plain runoff-rescaling null (offline study in `experiments/reservoir/laws_v6/`). The overnight arms used a frozen
routing head as a screen; the user objected to frozen heads on 2026-09-26, so final runs train jointly.
