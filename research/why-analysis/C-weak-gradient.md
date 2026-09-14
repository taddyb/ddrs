# C. Weak gradient: is roughness unidentifiable where channel travel time is short relative to the daily step?

**Analyst C, why-not-at-optimum swarm, 2026-09-09.**
**Data:** `.ddrs/experiments/landscape-p21-all-5yr/merged/figures/covariates.csv` (five-year census of the p = 21
model, 2,365 gauges; 2,124 well fit with NSE > 0.3 at the trained point), per-gauge netCDFs for the gauge-reach
geometry. Figures: `merged/figures/why/C_scaling.png`, `C_quadrants.png`.

## Definitions and assumptions

- Curvature of the loss in roughness: `H_nn`, the second derivative of the five-year NSE-batch loss with respect to
  ln n at the trained point (basin-uniform multiplier). `lambda1` is the stiff eigenvalue of the Hessian at the
  gauge optimum. "Flat" means |H_nn| < 0.02, at which a factor-2 change in n costs less than 0.005 in loss.
- Far from optimum: |ln(n*/n)| > ln 1.25.
- Travel-time proxy: a Manning velocity at the gauge reach under the observed mean flow, using the model's own
  depth and width there (hydraulic radius w d / (w + 2d)), the trained n and the reach slope (floored at 1e-3 as
  the routing does), and a main-stem length from a Hack-type relation L = 1.4 A^0.6 km. This is a proxy for the
  order of magnitude, not a path integral: the netCDF does not store the main path. Median proxy 1.4 days, IQR
  1.0 to 2.1, range 0.26 to 20 days.
- A caveat on slope: 1,335 of the 2,124 gauges have a gauge-reach slope between 0.001 and 0.002, i.e. at or just
  above the routing's slope floor. "Slope" below therefore separates floor-slope (flat) rivers from steeper ones
  more than it grades among them.

## Observations

1. **Curvature scales with size and length, and inversely with slope.** Spearman rank correlations of |H_nn| with
   total channel length +0.31, drainage area +0.27, reach count +0.26, the travel-time proxy +0.26, mean flow +0.18,
   and slope −0.43 (all p < 1e-7, n = 2,124). The best single predictor is slope; the joint log-log fit
   log|H_nn| = −5.1 + 0.39 log A − 0.84 log S − 0.14 log Q explains 22 % of the variance (slope alone 18 %, the
   travel-time proxy alone 8 %).
2. **Identifiability sets in around one day of travel.** Binned by the proxy, the median |H_nn| is 0.006 to 0.008
   below one day, 0.023 to 0.025 between one and four days, 0.017 at four to eight, and 0.11 beyond eight (24 gauges).
   By slope: 0.030 at floor slope, falling to 0.009, 0.005, 0.004, 0.002 as slope rises to 0.05.
3. **The displacement runs the other way.** Median ln(n*/n) is +1.47 (n* about 4.4 n) for proxies under half a day,
   +0.77 for half a day to a day, +0.53 for one to two days, +0.43 for two to four, +0.14 for four to eight, and
   −0.01 beyond eight days. The fraction "far" falls from 97 % to 50 % across the same bins. Small, fast basins are the
   farthest from their optima; the largest, slowest basins are at them.
4. **Gain does not track travel time.** Median gain is 0.017 under half a day, 0.009 to 0.012 for one to four days,
   0.003 to 0.007 beyond. Rank correlation of gain with the proxy is +0.03 (not significant); with slope −0.27; with
   size +0.12 to +0.14. The joint fit explains 7 %.
5. **Half the trained points sit on concave ground.** H_nn at the trained point is negative at 46 % of well-fit
   gauges (the loss is locally concave in ln n there), while lambda1 at the optimum is positive at 96 %. The
   concave fraction rises with slope: 41 % at floor slope, 60 to 69 % for slopes above 0.005.
6. **Quadrants** (2,124 well-fit gauges):

| | flat (|H_nn| < 0.02) | curved |
|---|---|---|
| near (|ln n*/n| ≤ ln 1.25) | 8 % (172), gain 0.001 | 10 % (207), gain 0.001 |
| far | **45 % (963)**, gain 0.009, median area 615 km², 82 % want slower | **37 % (782)**, gain 0.028, median area 1,226 km², 76 % want slower |

## Interpretation

The prediction holds for the curvature and fails for the displacement, and the two halves together answer the
question. Where travel time is under a day, or the reach slope is well above the floor, the daily-pooled loss
barely responds to roughness: the flood wave crosses the basin within one time step and roughness changes only how
the day's volume is shared with its neighbours. That is the weak-gradient regime, and it holds 45 % of the well-fit
gauges: far from their optima, but on flat ground, with a median 0.009 NSE at stake. The batch never received a
gradient from these gauges, so they stayed where the population's other gauges put n, and their "optimum" at four
times the trained roughness is mostly noise on a flat floor (the largest displacements are exactly where the
curvature is smallest).

The other 37 % are the costly cases: larger, floor-slope, low-gradient rivers with travel times of one to four days,
where the loss is curved in n and the trained point still sits a factor 1.9 from the optimum, for 0.028 NSE. These
are the gauges the batch pulled away from their optima, and the direction is the same as everywhere else, slower
(76 %). Analysts D and E should own why: inflow timing (the routed lag correlates with the displacement at 0.49
in the first look) and population weighting.

The concave trained points are a separate warning: at nearly half the gauges the trained n is on a ridge or shoulder
of the per-gauge loss, so the local Hessian there says nothing about the optimum, and any diagnostic that uses the
trained-point curvature alone (including the flat/curved split above) understates identifiability at those gauges.
The Newton search reaches a convex optimum at 96 % of them, so the optimum exists; the trained point is simply not
in its basin of quadratic approximation.

Beyond eight days of travel the 24 gauges have ten times the curvature and sit at their optima. At that scale
roughness is well identified and training got it right. That is the population-scale version of the Juniata
finding: the gauges that can see n do not disagree with the batch.

## Verdict

Partially supported. Identifiability of n scales with travel time as predicted, with a threshold near one day and
slope as the strongest single control, but "far from optimum" is not the same as "weakly identified": the
weak-gradient regime explains the far-but-cheap 45 %, not the far-and-costly 37 %. Roughness is a daily-step
problem for most of CONUS and a real training miss for a minority of large, flat rivers.

## Decisive follow-up

Evaluate the landscape at the hourly step at a sample of short-travel-time gauges using USGS instantaneous
observations (15-minute data pooled to hours) instead of daily means, with the same model and window. Prediction:
|H_nn| rises by roughly the ratio of the daily step to the travel time (a factor 2 to 5 at these basins) and the
displacement shrinks toward the one-to-four-day values. If it does, the daily objective is the cause and a sub-daily
or timing-aware loss is the training change; if it does not, the flatness is physical and these gauges cannot
constrain n at any step, which is equifinality.
