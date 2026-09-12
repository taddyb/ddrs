# E. Training-side causes of gauges sitting away from their optima

**Data.** `covariates.csv` (five-year census on the p = 21 model, epoch 30, plus the diagnostic series pass), 2,124 well-fit
gauges (NSE at the trained point > 0.3). Displacement `ln(n*/n)` = signed log ratio of the gauge-optimal to the trained
basin-median n; `gain` = NSE at the gauge optimum minus NSE at the trained point.
**Training setup** (config snapshot of `2026-09-08T15-55-52Z-conus-train-and-test`): NSE-batch loss (per-day squared error
divided by `(sigma_g + 0.1)^2`, sigma_g the training-period observed std), Adam, micro-batch 64 gauges with gradient
accumulation over 20 micro-batches, so one optimizer update per 1,280 gauges and **2 updates per epoch, 60 in total**;
lr 0.005 (epochs 1 to 10), 0.001 (11 to 20), 0.0005 (21 to 30); rho 90 days. Figures: `figures/why/E_*.png`.

## 1. Observations

**1.1 Loss weighting (candidate 1).** The per-day weight of a gauge in the batch loss is proportional to
`(1 - NSE0) * (sigma / (sigma + 0.1))^2`, which the `sigma` normalisation makes nearly uniform except for gauges with
sigma below about 1 m3/s. Its Spearman correlation with |ln(n*/n)| is +0.24 and with gain +0.23: **the gauges that weigh
most in the loss sit farther from their optima, not nearer.** By sigma quintile, the largest-sigma fifth (big rivers)
has the highest share wanting slower routing (77 %) and the largest median gain (0.022); the smallest fifth has gain
0.006.

**1.2 Basin size (candidate 2).** The training list (gages_3000) has median area 700 km2 and 30 % of gauges below
300 km2. By area quintile the smallest basins are farthest from their optima in log terms (median |ln(n*/n)| 0.87)
but gain nothing (0.009); the largest basins are closer (0.66) and gain most (0.020) with the only appreciable
curvature in n (median H_nn 0.008 against about 0 elsewhere). In a joint linear model with standardised covariates
and ecoregion dummies (n = 2,124):

| response | R2 | per-SD coefficients (sign and size) | largest single contributors (drop-one dR2) |
|---|---|---|---|
| ln(n*/n) | 0.31 | lag of routed flow vs obs +0.31, area -0.33, log sigma +0.29, peak ratio +0.17, spring fraction -0.11 | routed lag 0.14, peak ratio 0.04, area 0.03, sigma 0.03, region dummies 0.03 |
| gain | 0.33 | routed lag +0.016, peak ratio +0.015, log sigma +0.008 | routed lag 0.07, region 0.06, peak ratio 0.06, area 0.00 |

The timing of the routed flow against the observation (`lag_days_routed`, positive when the routed flow leads) is the
dominant covariate of both the displacement and the gain; area and sigma matter for the displacement only after
that, and ecoregion adds 0.03. Flashiness adds nothing.

**1.3 Attribute capacity (candidate 3).** For each gauge, the nearest neighbour in standardised MERIT attribute space
(area, slope, aridity, snow, precipitation, temperature, NDVI, permeability, porosity, catchment size):

| | attribute nearest neighbour | random pair |
|---|---|---|
| median |difference in ln n_trained| | 0.10 | |
| median |difference in ln n*| (gauge-optimal) | 0.56 | 0.79 |
| median |difference in ln(n*/n)| | 0.60 | 0.75 |
| same sign of displacement | 67 % | 61 % |

A 10-neighbour attribute regression predicts the gauge-optimal ln n* out of sample with R2 = 0.12. The trained ln n
varies across gauges with SD 0.43; the gauge-optimal ln n* with SD 0.79. **The attributes the head sees carry about a
tenth of the information needed to place n at the gauge optimum**; neighbours in attribute space want n values
that differ by a factor 1.75 at the median, only slightly less than random pairs.

**1.4 Optimizer steps (candidate 4).** 60 updates in total. On this population n reached its final value by epoch
10, i.e. after 20 updates at lr 0.005 (findings §10); the remaining 40 updates at 0.001 and 0.0005 moved it by
under 1 %. The sloppy coordinate q was still drifting at epoch 30 in the area-balanced run (§2).

## 2. Interpretation

- The gap between trained and gauge-optimal n is **not a weighting problem**: the normalised loss treats gauges
  nearly equally per day, and the gauges with the most weight are the ones left farthest off. Reweighting by sigma
  or area would move the median displacement little (their partial contributions are 0.03 each).
- It is **not primarily a step-count problem** for n: n converged early on this population and the population's
  aggregate gradient held it there. More steps would mostly move q.
- It is mostly a **capacity and information problem**: what a gauge wants from n is set by the timing error of the
  routed flow, which the attributes do not encode (R2 0.12), so the shared head cannot deliver it even in principle.
  Two gauges with the same attributes want n values a factor 1.75 apart. This is the batch compromise mechanism at
  the level of individual gauges: the head assigns one n per attribute vector, and the gauges sharing that vector
  disagree.
- Where the routed flow leads the observation, the gauge asks for more roughness; that is the channel absorbing an
  inflow timing error. Whether the timing error is in the runoff product (dHBV2 UH) or in the routing's missing
  storages (floodplain, tile drainage, reservoirs) cannot be settled from these covariates; the ten-gauge hydrographs
  (analysts A to D, top-10 reports) are the place to look. Large rivers carry the actionable gain because they are the
  only basins where n has curvature.

## 3. Testable training changes, ranked by expected effect

1. **Give the head timing information.** Add attributes that predict routing needs and are absent today: channel
   width or width-to-depth (GRWL), sinuosity, floodplain or wetland fraction, tile-drainage or cropland fraction,
   reservoir and lake storage index. Evidence: attribute-neighbour disagreement (1.3) and the routed-lag dominance
   (1.2). Confirming experiment: retrain with the augmented attributes on gages_3000; success = the kNN R2 of ln n*
   on attributes rises well above 0.12 and the census median |ln(n*/n)| falls below 0.5 at well-fit gauges.
2. **Learn a per-basin timing term instead of asking n to carry it.** A learnable inflow delay or unit-hydrograph
   scale per catchment (or a learnable tau) trained jointly. Evidence: lag of routed flow is the strongest covariate
   of both displacement and gain. Confirming experiment: retrain with the delay; success = displacement in n loses
   its correlation with the lag and the systematic "slower" majority (65 %) drops toward 50 %.
3. **Width as a function of river size (p from drainage area).** Evidence: the largest basins want slower routing at
   77 % and gain most; a 21 m channel on a main stem forces n and q to compensate (findings §7). Confirming
   experiment: the Moody and Troutman p(Q_ref) arm; success = big-river displacement collapses while small-basin
   displacement is unchanged.
4. **Area- or sigma-stratified sampling or reweighting.** Expected effect small (partial dR2 0.03 each); worth one
   run only after 1 to 3 because it is cheap: reweight the batch loss by drainage-area class and re-census.
5. **More optimizer steps at the high learning rate.** Expected effect on n near zero on this population (converged
   by update 20); may matter for q. Cheapest test: resume the epoch-30 checkpoint for 30 more epochs at lr 0.005 with
   grad accumulation off (29 updates per epoch on the area-balanced list) and re-census; success = q displacement
   shrinks, n unchanged.

## 4. Caveats

Correlational, one model, one population (gages_3000, in sample). The attribute test uses MERIT attributes only;
the head also sees the normalisation statistics. `lag_days_routed` is a whole-window cross-correlation lag in
integer days and is coarse for basins with sub-daily travel times. The loss-weight proxy uses the five-year NSE,
not the training-period one.
