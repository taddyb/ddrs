# Analyst A: why the loss landscapes are flat in the width exponent q

**Question.** At most gauges the (n, q) landscape barely changes along q. Is that low flow (depth near 1 m, where
d ln w / d q = ln d vanishes), negative-discharge or floor clamping (zero gradient), or an intrinsically weak gradient?

**Data.** Five-year census (`landscape-p21-all-5yr/merged`, 2,365 gauges, p = 21 model, epoch 30), its diagnostic
pass with daily series, and the covariate table `merged/figures/covariates.csv`. Well fit = NSE at the trained point
above 0.3 (2,124 gauges). Figures: `merged/figures/why/A_flatq_depth.png`, `A_flatq_sign_lowflow.png`.

## 1. Operational definition of "flat"

For each gauge the loss change for a factor-2 move in a parameter, from the gradient and Hessian at the trained point:
`dL = |g| ln 2 + 0.5 |H| (ln 2)^2`, relative to the loss there. Flat = that change is under 1 % of the loss.
(Thresholds of 0.5 % and 2 % give the same picture: 72 % / 85 % / 93 % flat in q.)

## 2. Observations

1. **q is flat at 85 % of well-fit gauges; n at 11 %.** The two never swap: no gauge is flat in n but curved in q.
   The gradient in q is 2 % of the gradient in n at the median gauge.
2. **Flatness in q falls monotonically with depth at the gauge (and with basin size).**

   | depth at gauge under mean flow | gauges | flat in q | flat in n | median mean flow (m3/s) | median gain |
   |---|---|---|---|---|---|
   | < 0.3 m | 306 | 94 % | 22 % | 1.5 | 0.009 |
   | 0.3 to 0.5 | 402 | 95 % | 11 % | 4 | 0.008 |
   | 0.5 to 0.8 | 486 | 90 % | 12 % | 8 | 0.008 |
   | 0.8 to 1.25 | 383 | 86 % | 9 % | 18 | 0.009 |
   | 1.25 to 2 | 274 | 78 % | 6 % | 37 | 0.012 |
   | 2 to 4 | 216 | 61 % | 1 % | 94 | 0.015 |
   | > 4 m | 57 | 28 % | 4 % | 298 | 0.037 |

3. **The symmetric-leverage prediction fails.** If q were flat because ln d is near zero, flatness would peak at
   d = 1 m and fall on both sides. It does not: the shallowest gauges (|ln d| largest) are the flattest. Spearman of
   log(q curvature / L0) with log depth is +0.36; with |ln d| it is −0.18.
4. **Low flow and clamping are not the cause.** Spearman with the low-flow fraction −0.04 (not significant), with the
   zero-flow fraction −0.01. Routed flow at the gauge is at the discharge floor on more than 1 % of days at 0.3 % of
   gauges; observed zero flow on more than 1 % of days at 6 %. Neither predicts flatness.
5. **Partial analysis** (OLS of log q-curvature on standardized covariates, R2 0.17): trained q (+0.26), gauge slope
   (−0.23), drainage area (+0.19), depth itself (−0.05 once area and slope are in), mean flow (+0.04), low-flow
   fraction (+0.03). For n the same regression gives R2 0.28 with area (+0.29) and slope (−0.32) dominant.
6. **Sign.** The q curvature at the trained point is negative at 69 % of gauges, evenly across depth bins (64 to
   73 %). Gauges with negative q curvature have larger available gain (0.012 vs 0.006) and their q optimum moves
   farther (median |alpha_q*| 0.99 vs 0.79). 11.6 % of well-fit q optima sit on the box edge (8.7 % at the floor,
   2.7 % at the ceiling); the share is the same for flat and non-flat gauges (11 % vs 13 %) and rises with depth
   (21 % for the deep bins).
7. **The 15 % that are not flat** (327 gauges): median depth 1.5 m, mean flow 29 m3/s, area 2,600 km2, 57 reaches,
   against 0.7 m, 10 m3/s, 860 km2, 19 reaches for all well-fit gauges; 39 % of them are deeper than 2 m (8 % among
   the flat ones). Ecoregions: Northeast 102, Southeast Plains 50, Western Mountains 41, Eastern Highlands 39,
   Central Plains 39.

## 3. Interpretation

- **Weak gradient by construction, not low flow and not clamping.** q enters the routing only through width
  (w = 21 d^q) and thence through attenuation and the Cunge weighting. At a small or steep basin the channel travel
  time is well under a day, so the daily NSE cannot see attenuation at all, and no width change registers. Depth at
  the gauge is a proxy for basin size and travel time, which is why flatness tracks depth monotonically and why depth
  drops out of the regression once area and slope are in. The low-flow and floor hypotheses are refuted by the
  covariates: the flat gauges are not the dry or clamped ones.
- **The ridge sign.** Negative q curvature at two thirds of gauges means the trained q sits on a gentle crest along
  q: moving q either way lowers the gauge loss slightly. That is consistent with q being set by the batch (a CONUS
  compromise) at a value that is locally the worst for the individual gauge, but the crest is so low (1 % of the loss
  for a factor 2) that it costs nothing. It is why the q optimum scatters to both box edges across gauges and flips
  sign between windows (findings §12): the optimum along a flat direction is noise.
- **Where q is identifiable.** Deep, large, low-slope rivers: the Northeast and Southeast Plains main stems, where
  travel time is days and attenuation shapes the daily hydrograph. There a factor 2 in q moves the loss by several
  percent and the q optimum is meaningful. This is the same population where n is identifiable, and the same
  population where the ten most egregious gauges live.
- **Equifinality statement.** For 85 % of gauges the model's width exponent is unidentifiable from daily discharge:
  any q in the range gives the same loss within 1 %. Learning q per reach from daily gauges is therefore learning
  noise there; the attribute-conditioned head can only propagate a value from the deep-river minority.

## 4. Verdict

| candidate cause | share of flat cases it explains | evidence |
|---|---|---|
| low flow (depth near 1 m) | none | flatness is monotonic in depth, not peaked at 1 m; low-flow fraction uncorrelated |
| negative-discharge or floor clamping | none at the gauge outlet | routed floor days above 1 % at 0.3 % of gauges; no correlation |
| weak gradient (daily objective blind to attenuation in small, steep basins) | essentially all | depth, area, slope, and trained q explain the curvature; the exceptions are the deep rivers |

Residual: the regression explains 17 % of the variance in q curvature, so most of the gauge-to-gauge scatter is
unexplained by these covariates; the sign structure (ridge) is part of it.

## 5. Decisive follow-up test

Score the same landscape at hourly rather than daily resolution at 60 gauges drawn evenly across the seven depth bins
(the objective pools to daily under the training `tau`; add a `pool: hourly` option and use the hourly USGS store).
If q curvature rises by an order of magnitude in the shallow bins, the flatness is the daily objective's blindness
and hourly training would make q learnable; if it stays flat, q is structurally unidentifiable in this geometry and
should be prescribed (or tied to depth via a downstream hydraulic-geometry relation) rather than learned.
