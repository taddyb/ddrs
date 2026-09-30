<!-- agent aa03fb48e3b547d5c | Fable: leakance with the new code | handback | 2026-09-27T13:48:33.871Z -->
> The agent's final handback message, verbatim; the full report is `report.md` beside it. Saved 2026-09-27.

Full report: /home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_leakance/report.md (analysis scripts and JSON outputs in the same directory: leak_vs_dams.py/.json, null_scale.py/.json, artifact_checks.py/.json, leak_vs_base_per_gauge.csv). All measurements are reads of existing run outputs, no training.

## New evidence (the ranking rests on it)

1. The two-way leakance arm (2026-09-17T16-38-16Z) is a near-uniform 2 % volume sink, not a spatial field. Against sr_n0_gamma (2026-09-12T23-39-03Z) it lowered the volume ratio at 2,235 of 2,365 gauges, including 450 of 531 gauges that were already short of water (ratio < 0.95). NSE: +0.0022 at surplus gauges, -0.0052 at deficit gauges. Learned parameters barely vary: K_D p10-p90 1.04e-7 to 1.37e-7 on a [1e-9, 1e-5] box (the initialization), factor 0.86, d_gw 0.29 m. The field is constant x area_z x (depth - 0.3 m).
2. One global scalar beats it. Scaling sr_n0_gamma's predictions by 0.95 gives median NSE 0.7438 (1,312 up / 1,053 down) vs 0.7391 base; the leakance arm gives 0.7300. The scalar's per-gauge signature correlates with the leakance arm's at Spearman 0.45.
3. Leakance did not localize on dams. Dam reaches lose 1.5 to 1.8x more than size-matched reaches (p < 1e-4), but K_D, factor and d_gw are identical on dam and non-dam reaches in every discharge decile; the excess is area_z geometry. Reservoirs therefore do NOT remove a confound in the July NO-GO; that verdict does not depend on dams.
4. An artifact caught: the leakance gain correlates with the dam-release gain at dammed gauges (0.30, partial 0.59), but equally at undammed gauges (0.41) and negatively against the seed-43 dam-release run (-0.02 / -0.38). Shared-base artifact; do not use.
5. Heavily regulated gauges carry a real volume surplus: DOR 1-2 median ratio 1.119, DOR > 2 1.136, vs 1.050 undammed; paired against matched smoke controls +6.0 % (n 63) and +2.6 % (n 53). Below DOR 0.5 dammed gauges have less surplus than controls. 28 to 32 % of high-DOR gauges are deficits no sink can fix.

Caveat on every gauge-level leakance number: the 09-17 arm also widened the gamma box ([0,0.5] -> [-0.2,0.85]) and its matched control was never run. Reach-level reads and the scalar null are free of this.

## Top 5, ranked by (gain x confidence) / cost

1. Global (or per-HUC2) q' volume scalar as the mandatory null model for any loss term. Evidence: item 2. Test: two smoke arms, sr_n0_gamma recipe with and without a learnable scalar on q' (not a routing-core change). Go for any downstream loss term only if it beats this arm's paired dNSE with CI clear; no-go for leakance's volume role otherwise. Cost: an afternoon.

2. Dam-row loss e_d (evaporation/withdrawal) as a 4th release-head output, losing-only, bounded. Evidence: item 5, Santa Rosa 0.72, reservoir-options §3 volume ceiling +0.058 at DOR > 0.5. Form: dS/dt = I - Q - eS on the storage-conserving row, only the Q coefficients change, e = 0 bitwise today's row; add NID surface_km2 and dam-reach aridity as head inputs. Expected +0.01 to +0.02 at the 165 DOR > 1 gauges, +0.001 population. Test: {learned, learned + e_d} x seeds {42, 43} on the smoke set, 4 h CPU. Go: dam-minus-control dNSE > +0.01 at DOR > 1 pairs with CI clear AND Spearman(learned e_d x surface, PET x surface) > 0.3. No-go: e_d uniform across dams or the external correlation <= 0 (another bias absorber). Risk: a point sink still absorbs upstream inflow surplus; the matched controls detect that.

3. Sparse leakance: gate off at init or L1 on sum|zeta|. Evidence: item 1 (field = initialization; gate anneal froze decisions by epoch 30). Test: the 2026-07-04 synthetic planted-reach harness with K_D free. Go: R1 >= 0.5 and planted reaches hold > 50 % of flux. No-go: R1 < 0.1, close the parameterization question for good. Depends on the NO-GO's smearing mechanism, does not re-open the structural argument.

4. Single-factor leakance test on the smoke set: {leak off, on} x {dams off, learned}, seeds 42 and 43, same boxes, 8 x 1 h CPU. Read-out is not skill: share of gauges where leakance moves volume toward observed (the 09-17 arm: about 30 %). Go: > 65 % and beats opportunity 1. No-go: near 50 % or below; leakance is a bias term, record it, stop training it on skill.

5. Coexistence plumbing (enabling only): forward_chain_inner already takes the reservoir argument and timestep_backward_core already masks dam rows from state.reservoir_mask; timestep_forward_leakance passes None and route_timestep asserts. Carry the mask through TimestepLeakanceOp/GammaOp (11 parents with the learned release's T_t, T_{t+1}), fold the dam mask into the impervious mask so zeta and the mass bound are zero on dam rows. Gates: leakance gradchecks with dam rows, leakance_off_parity, smoke check 1a.

Also cheap (rank 6): dam completion as a natural experiment, 153 gauges whose dam was built inside 1981-2010 (dam_age_check.json): before/after volume ratio minus the controls' gives the dam's net loss from observations only, a prior for e_d.

## Code findings the new work exposes (report §4)
Zeta on a dam row uses a Manning depth the row no longer computes; the mass bound alpha x relu(b_base) on a dam row with c3 near 1 permits removing 25 % of stored water per hour; K_D never left the box centre and the annealed gate freezes by epoch 30; two-way exchange produced no gaining reaches where water was missing (d_gw 0.14 to 0.38 m never crosses stage); the collect_zeta diagnostic cannot report the bound's binding fraction, which the pre-registration listed as a distrust criterion.

## Past conclusions depended on
NO-GO structural argument (network sum not invertible, 2026-07-06 §3, §6); the 09-18 pre-registered failure; the 09-22 finding that leakance leverage tracks volume bias (+0.54); reservoir-options §3 (timing vs volume); the 09-27 dam-release results. None re-opened; every proposal is a null model, a point sink with an external check, or a sparsity prior on the synthetic harness.
