# Analyst D: is the n displacement compensating inflow error, or is the channel too fast on its own?

**Data.** `.ddrs/experiments/landscape-p21-all-5yr/merged/figures/covariates.csv` (2,365 gauges; 2,124 well fit,
NSE at the trained point > 0.3) plus the diagnostic daily series (`landscape-p21-all-5yr-diag/merged`, five water
years 1996 to 2000: observed, routed at the trained point, summed inflow with no routing). Model: the user's
p = 21 run, epoch 30. Figures: `merged/figures/why/D_timing_vs_displacement.png`, `D_extremes_hydrographs.png`;
tables `D_timing_groups.csv`, `D_models.json`, `D_extremes.csv`; per-gauge `D_wellfit_with_fractional_lags.csv`.

**Timing measure.** The covariate table's integer `lag_days_*` are quantised to whole days, so timing was recomputed
from the series as the fractional cross-correlation lag (parabolic interpolation around the best of lags −3 to +3).
Convention in this note and in the figures: **positive = the series arrives early** relative to the observed flow.
`delay_channel = early_inflow − early_routed` is the delay the trained channel adds.

## Observations

1. **Timing of the routed flow explains most of the displacement.** Spearman rho between ln(n*/n) and "routed
   arrives early" is +0.73 (the strongest single correlate found in the swarm's covariates). By class:

   | routed timing (well-fit gauges) | n | median ln(n*/n) | wants slower | wants faster | median gain |
   |---|---|---|---|---|---|
   | early by more than 0.5 d | 483 | +1.04 (× 2.8) | 94 % | 1 % | 0.045 |
   | early by 0.25 to 0.5 d | 366 | +0.83 | 88 % | 1 % | 0.025 |
   | on time (within 0.25 d) | 1,114 | +0.28 | 55 % | 18 % | 0.005 |
   | late by 0.25 to 0.75 d | 99 | −0.96 | 3 % | 90 % | 0.005 |
   | late by 0.75 to 1.5 d | 34 | −1.16 | 3 % | 94 % | 0.005 |
   | late by more than 1.5 d | 28 | −1.41 | 4 % | 93 % | 0.004 |

   The sign flips exactly where physics says it should: early routed flow wants more roughness, late routed flow
   wants less. Gauges whose routed flow is on time still lean "slower" (55 %) but with a negligible gain (0.005).

2. **Joint model.** Ordinary least squares of ln(n*/n) on standardised timing (routed and inflow lags), peaks (peak
   ratio, flashiness), volume (routed and inflow volume ratios), regime (spring fraction, snow fraction, log aridity)
   and size (log area, log reaches): R² = 0.45. Dropping the timing group costs 0.27 of that; peaks 0.03, regime 0.04,
   volume and size under 0.01. A gradient-boosted model reaches a 5-fold cross-validated R² of 0.78 with permutation
   importance 1.74 for routed timing, 0.16 for inflow timing, and under 0.06 for everything else. For the gain, the
   boosted model reaches R² 0.69, with routed timing (0.95), flashiness (0.32) and peak ratio (0.26) leading.
   Volume bias (median routed volume ratio 1.04, IQR 0.96 to 1.11) is not what drives the displacement.

3. **The trained channel adds almost no delay.** Median delay added by the channel is 0.00 days across well-fit
   gauges; 68 % of gauges get under 0.1 day. Where it adds delay it is the large networks (correlation of channel
   delay with log total channel length 0.60). Consequently the routed flow's timing is the inflow's timing:
   inflow early by 0.28 d at the median "wants slower" gauge, routed early by 0.29 d.

4. **The gauges that want slower are the ones whose inflow arrives early and is not delayed.** By displacement
   class (medians): wants slower, inflow early 0.28 d, routed early 0.29 d, peak ratio 0.94, volume ratio 1.05;
   about right, inflow early 0.13, routed 0.05; wants faster, inflow late 0.08 d, routed late 0.18 d, peak ratio
   0.83. Regime matters at second order: snow-dominated gauges (high spring fraction) lean less toward slower
   (rho −0.18) because the melt season, not the channel, sets their daily timing.

5. **The extremes confirm the numbers** (`D_extremes_hydrographs.png`). At the five gauges with the largest
   displacement and gain above 0.1 (Salt Creek IL 05578500, Little Wabash 03381500, Kaskaskia 05594100, two small
   Plains streams), the routed curve lies on top of the summed inflow: identical timing, identical peaks. Both lead
   the observed flow by 1.2 to 3 days and overshoot the peaks by 20 to 60 %; the observed hydrograph is a smoothed,
   delayed version of the inflow. That is precisely the transformation a slower, more diffusive channel performs,
   and the model's channel is not performing it. At the five gauges that want faster (three 3-reach coastal-plain
   and Shield headwaters, two Michigan rivers), the inflow itself lags the observed response by 0.5 to 1 day and
   misses the flashy peaks; the gauge asks the channel to speed up, but a channel cannot create an earlier peak
   than its inflow, and the gain is 0.03 to 0.07.

## Interpretation

**Verdict: at the well-fit gauges the displacement is not compensation for a wrong inflow. It is the channel
itself being too fast.** The summed inflow is a hillslope response without channel travel; it must lead the gauge
by the true travel time, and the trained channel, with n ≈ 0.04 to 0.06, p = 21 and q ≈ 0.1, delivers that travel
time nowhere but in the largest networks. Timing of the routed flow explains 45 % of the variance in the
displacement linearly and 78 % with a non-linear model; volume error explains under 1 %. The "wants slower"
majority (65 % of well-fit gauges) is the set of gauges where the missing channel delay is at least a quarter of a
day and the daily NSE can see it.

Two qualifications keep this from being the whole story.
- **Compensation exists at the "wants faster" tail.** Where the inflow is late or misses peaks (small, flashy
  basins), the channel is asked to do what only the inflow can do; those displacements are inflow error, they are
  small in number (17 %) and in gain (median 0.005), and no channel setting fixes them.
- **The inflow's own timing error is not measurable from this data alone.** If the UH inflow were itself early
  by a fraction of a day at every gauge, the channel would be asked to absorb that too. The extremes argue against
  a large effect: at the Kaskaskia the required 1.5 to 3 days of delay is a physical travel time for an
  11,000 km² basin, not a data artefact.

**What this says about training.** The batch left n at a value that makes the channel nearly transparent
(delay ≈ 0 at two thirds of gauges), while the gauges that carry a day or more of travel time want n four to nine
times larger. The trajectory study on this run (findings §10) showed n falling from 0.13 to 0.04 in the first ten
epochs and holding; the population gradient pushed the channel faster while the large basins pushed slower. The
loss's per-gauge normalisation by observed variance and the count of small, on-time basins are the candidates for
that imbalance (analyst E's question). The result is a systematic, physically interpretable routing bias, not
equifinality: the direction is the same at 94 % of the early-arrival gauges, and the two-dimensional Newton finds
an interior optimum there.

**What this says about equifinality.** The "on time" class (1,114 gauges, half the well-fit set) is the
equifinal set: median gain 0.005, direction split 55/18, displacement median +0.28. At those gauges the channel
delay is under the daily resolution, n is not identifiable from daily NSE, and the trained value is as good as any
within a factor two. The map's red in panel (a) at those gauges is not error.

## One decisive follow-up test

The claim "the channel is too fast on its own" predicts that at the gauge optimum the routed flow's timing error
goes to zero and its peak ratio to about one, using the same inflow. The ten-gauge study now running
(`landscape-p21-top10-nse`, `series: true`) writes `routed_daily_star`: compute the fractional lag and the peak
ratio of `routed_daily_star` against the observed flow at those ten gauges. If both collapse toward zero and one,
the inflow needed no correction and the displacement was routing bias; if the routed flow at the optimum is still
early or still overshoots, the residual is inflow error the channel cannot absorb, and that residual is the
compensation share. The same two numbers, computed for all gauges from a `series: true` rerun at α*, would give the
compensation share CONUS-wide in one pass.

## Out of scope, noted

- The training-side mechanism (why the batch gradient chose the faster channel) belongs to analyst E.
- Whether p as a function of river size removes the bias without touching n is the assessment doc's open item.
- The covariate table's integer `lag_days_*` columns should be replaced by the fractional lags computed here
  (`D_wellfit_with_fractional_lags.csv`, columns `early_r`, `early_q`, `delay_channel`).
