# B. Does clamping kill the gradient?

**Question.** The engine clamps negative discharges (enforce positivity), floors discharge at 1e-4 m³/s, depth at
0.01 m and bottom width at 0.01 m, and the landscape clamps n and q to their ranges; each clamp zeroes the local
gradient. Are the gauges whose trained roughness sits far from their own optimum, or whose landscape is flat, the
gauges with clamping evidence?

**Data.** `covariates.csv` on the five-year census (`landscape-p21-all-5yr/merged`, 2,365 gauges, 2,124 well fit at
NSE > 0.3) with the diagnostic series (`landscape-p21-all-5yr-diag/merged`); the trained per-reach fields `n0`, `q0`
from the census netCDFs. Figures: `figures/why/B_clamp_shares.png`, `B_groups.png`, `B_poor_by_region.png`.

## Observations

**1. Parameter-range clamping at the trained point does not occur.** Across all 2,124 well-fit gauges, no reach has
its trained n within 1 % of the range edges (0.015, 0.25) or its trained q at an edge (0, 1). Trained n runs 0.027 to
0.107 (5th to 95th percentile of basin medians), q 0.05 to 0.40. The range clamp cannot be zeroing the gradient
where the model was left. The clamp does act at the *optimum*: 44 % of well-fit gauges have some reaches clamped
at the point the search stops, mostly q pushed to its ceiling or floor by the multiplier.

**2. Discharge floors and zero flow are rare where the model fits.**

| indicator, share of gauges with any occurrence | well fit (2,124) | poorly fit (241) |
|---|---|---|
| routed discharge at the 1e-3 m³/s floor on any day | 2.2 % | 5.8 % |
| observed zero-flow days | 8.2 % | 21 % |
| observed low-flow days (below 0.1 × mean) | 67 % | 59 % |
| search hit the range bound | 0.5 % | 2.1 % |
| optimum on the alpha box (q at ± ln 10) | 11 % | 3.7 % |
| Newton direction not a descent direction (gradient fallback) | 8.0 % | 13 % |

**3. Gauges with clamping evidence are steeper, not flatter.** Grouping the well-fit gauges (medians):

| group | n | |H_nn| | |H_qq| natural | NSE gain | ln(n*/n) |
|---|---|---|---|---|---|
| no floor, no zero flow, low flow ≤ 20 % of days, no fallback | 1,436 | 0.014 | 0.049 | 0.008 | 0.53 |
| routed floor days > 0 | 46 | 0.078 | 0.121 | 0.011 | −0.05 |
| observed zero-flow days > 0 | 175 | 0.026 | 0.068 | 0.011 | 0.48 |
| low flow > 20 % of days | 495 | 0.023 | 0.077 | 0.013 | 0.67 |
| gradient fallback (indefinite Hessian) | 169 | 0.051 | 0.138 | 0.045 | 1.04 |

The floor-day gauges have five times the curvature of the clean group in n (p = 2e-5) and want no change in n
(median ln ratio −0.05 against +0.53; p = 8e-6). The zero-flow gauges are marginally steeper (p = 0.002) and
otherwise indistinguishable. Rank correlations of floor and zero-flow shares with |H_nn|, |H_qq| and gain are all
below 0.11 in magnitude.

**4. Flat landscapes are not clamped landscapes.** Only 1.9 % of well-fit gauges have both |H_nn| and |H_qq| below
1e-3. Those 40 gauges have no zero-flow days, one has a floor day, median low-flow share is 0, and they are tiny
basins: median 7 reaches and 3.2 m³/s mean flow, against 19 reaches and 10.5 m³/s for the rest.

**5. The negative-discharge clamp is small and not attributable per gauge.** The training log reports it only per
mini-batch: 0.04 to 0.05 % of reach-timesteps pre-clamp (1,110 batch lines, e.g. 3,378 of 8,164,240). Per-gauge
attribution would need the eval-time accumulator that the leakance work built for zeta, applied to negative
discharges. Not available in this pass.

**6. The poorly fit gauges are intermittent, and regionally so.** Among the 241 gauges with NSE ≤ 0.3, 21 % have
observed zero-flow days, 6 % have routed floor days, and their curvature is comparable to the well-fit gauges
(|H_nn| 0.016, |H_qq| 0.029), with median gain 0.015. By ecoregion: Western Xeric (46 gauges) 43 % zero-flow, 13 %
floor days, inflow volume ratio 0.82; Western Plains (11) 64 % zero-flow, 45 % floor days, ratio 0.90; Western
Mountains (104, the largest group) 14 % zero-flow, 1 % floor, ratio 1.04. Only the Xeric and Plains subgroups are
plausibly clamp-dominated; the Western Mountains poor fits are not.

## Interpretation

Clamping does not explain the displacement of the trained roughness from the gauge optima, and it does not explain
flat q landscapes. Where the routed flow or the observation touches the floor, the landscape is steeper and the
trained n is already at the gauge's preferred value; where the landscape is flat, there is no clamping at all and
the basin is simply too small for channel timing to matter at the daily step (analyst C's territory). The one
clamping effect that is real is at the optimum, not at the trained point: 11 % of well-fit optima sit on the q box
and 44 % have some reaches clamped there, so the "optimum" for q is a bound in about one gauge in nine.

The 8 % of well-fit gauges where the Newton direction was not a descent direction deserve separate attention: they
are the gauges with the largest gains (0.045 median) and displacements (n × 2.8), steep in both parameters, and
their trained point sits on a ridge or saddle of the per-gauge loss. That is not clamping; it is the batch solution
landing on a non-convex part of the gauge's landscape (analysts C and F).

For the hollow gauges on the map, intermittency is the marker of a small, arid subset (Xeric and Plains); the largest
poorly fit group, the Western Mountains, is neither intermittent nor floor-clamped, and its inflow volume is right on
average, so its failure is timing in the inflow (snowmelt), not the channel and not clamping.

## Verdict

REFUTED for the well-fit population: clamping (discharge floor, zero flow, parameter ranges, positivity) is rare at
the trained point, does not co-occur with flat landscapes, and where it occurs the landscape is steeper, not flatter.
SUPPORTED only as a marker of the arid intermittent subset of the poorly fit gauges (about 60 gauges), where the
problem is upstream of the channel anyway.

## One decisive follow-up

Add a per-gauge accumulator of clamp events to the diagnostic pass (hours at the discharge floor, hours with negative
pre-clamp discharge, reaches at the depth or width floor) over the gauge's subgraph, written next to the series; then
regress |H_qq| on the clamp-hour share within the 40 flat gauges and the 46 floor-day gauges. If the flat gauges show
zero clamp hours, as the day-level series say, the clamping hypothesis is closed at the hourly level too.
