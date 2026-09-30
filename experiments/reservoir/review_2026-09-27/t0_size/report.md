# Does the learned dam response time T0 correlate with dam size?

Runs: seed 42 `2026-09-27T07-29-55Z-train-and-test`, seed 43 `2026-09-27T10-31-50Z-train-and-test`
(1,024 dam reaches each; NID >= 10 MCM, snap class A-D, several dams on one reach aggregated as in
`build_dam_features.py`: storages, surface, max discharge summed; height = max; drainage, year, purpose from
the largest dam). Scripts and tables live in this directory: `assemble.py` (joins), `analyze.py` (stats and
figures), `t0_size_table.csv` (the joined per-COMID table), `results.json`.

## Data and proxies

- Learned T0 (days) from `release_params.csv`. Seed 42: median 0.35 d, IQR 0.19 to 0.84 d, max 17.4 d
  (Dworshak). Seed 43: median 0.53 d, IQR 0.32 to 1.05 d, max 75.6 d (Flaming Gorge; 9.4 d in seed 42).
- Size measures from the NID intersection table (`experiments/reservoir/nid/`), aggregated per COMID exactly
  as the head's inputs are.
- Mean annual flow at the dam, for residence time = storage / mean flow. Two proxies:
  (a) gauge specific discharge times dam drainage area: for the 247 dams on a gauge reach, that gauge's mean
  observed discharge over WY1996-2010 (baseline cache `observations.f32`) divided by its `DRAIN_SQKM`; for the
  other 777, the nearest gauge (great-circle, median 13 km, 90th percentile 43 km) whose drainage area is within
  a factor 30 of the dam's. Covers all 1,024.
  (b) GRanD `GRanD_MEANFLOW_CUMECS` from ISTARF-CONUS joined through `grand_to_merit_comid.csv`; covers 757.
  The two agree: Spearman 0.93, median log10 ratio (a)/(b) = -0.015, IQR -0.11 to +0.08. Residence times from
  the two agree at Spearman 0.90 (n = 757), and the gauge proxy agrees with HydroLAKES `Res_time` at 0.74
  (n = 109). Proxy (a) has isolated bad picks (Flaming Gorge: 144,646 d against about 845 d from GRanD), so
  a combined column (GRanD where available, else gauge) is also reported; conclusions do not change.
- Independent T estimates: offline per-gauge bucket fits on routed inflow (`expected_release_fit.csv`,
  seasonal `seas_T0` and linear `lin_T0`; 202 dams on their gauge reach, 339 total; the fit's search box is
  0.05 to 1000 d and 44 of the 202 on-reach seasonal fits sit on the 0.05 d floor, 2 on the 1000 d cap), and
  the 41 ResOpsUS dams with T fitted from their own inflow and release (`reservoir_T_fits.csv`, status fitted;
  21 of 41 have fit NSE < 0.5, 2 are on the 3000 d upper wall).

## 1. T0 against size (Spearman rho, 95 % bootstrap CI, 1000 resamples)

| variable | n | seed 42 | seed 43 |
|---|---|---|---|
| normal storage [MCM] | 1024 | 0.665 [0.628, 0.698] | 0.597 [0.550, 0.637] |
| max storage [MCM] | 988 | 0.778 [0.752, 0.804] | 0.718 [0.681, 0.751] |
| surface area [km2] | 949 | 0.405 [0.349, 0.459] | 0.439 [0.383, 0.488] |
| dam height [m] | 1024 | 0.552 [0.509, 0.594] | 0.403 [0.353, 0.452] |
| drainage area [km2] | 1024 | 0.101 [0.036, 0.169] | 0.007 [-0.064, 0.076] |
| max discharge [m3/s] | 844 | 0.345 [0.285, 0.407] | 0.086 [0.022, 0.154] |
| residence time, gauge proxy [d] | 1024 | 0.364 [0.304, 0.419] | 0.422 [0.372, 0.480] |
| residence time, GRanD mean flow [d] | 757 | 0.248 [0.178, 0.315] | 0.330 [0.256, 0.396] |
| residence time, combined [d] | 1024 | 0.349 [0.285, 0.409] | 0.411 [0.356, 0.467] |
| storage per upstream area [m] | 1024 | 0.412 [0.355, 0.466] | 0.478 [0.424, 0.529] |
| year completed | 997 | 0.387 [0.329, 0.437] | 0.340 [0.288, 0.394] |

Partial rank correlations (rank residuals on the conditioning variable), seed 42 / seed 43:
residence time given storage 0.22 / 0.30; storage given residence time 0.62 / 0.54; storage given drainage
0.68 / 0.65; drainage given storage -0.21 / -0.28; height given storage 0.39 / 0.21. Collinearity among
predictors: storage vs drainage 0.39, storage vs residence time 0.32, storage vs height 0.44, storage vs
max storage 0.90.

So: absolute storage is the dominant correlate, max (crest) storage more than normal storage. Relative size
(residence time, storage per unit area) carries a weaker, positive signal that mostly survives conditioning on
storage (0.22 to 0.30). Drainage area alone is uncorrelated, and negative once storage is held fixed: for a
given storage, a bigger river gets a shorter T0, which is the direction residence time predicts.

By primary purpose (table2_by_purpose.csv), seed 42 medians: flood risk reduction 1.34 d (n = 233),
irrigation 0.50, water supply 0.42, hydroelectric 0.26, recreation 0.19, navigation 0.11 (n = 25, IQR
0.10 to 0.11: run-of-river, collapsed to the floor). Seed 43 medians are 1.3 to 2 times longer for every
purpose except flood control (1.33 d) and navigation (0.11 d). Within purpose, rho with storage is 0.70 to 0.89
(seed 42) and 0.57 to 0.79 (seed 43) for the four big classes. Within flood-control dams the residence-time
correlation vanishes (0.10 / 0.14) while the storage correlation stays (0.70 / 0.57): the head raises T0 for
flood dams through the `purpose_flood` flag and max storage (their max/normal storage ratio has median 1.93,
against 1.24 overall), not through normal-pool residence time.

## 2. Which inputs explain T0 (surrogates from the 19 head inputs to log10 T0, 5-fold CV)

| surrogate | seed 42 CV R2 | seed 43 CV R2 |
|---|---|---|
| ridge, all 19 normalised inputs | 0.907 | 0.889 |
| gradient boosting (300 trees, depth 3), all 19 | 0.954 | 0.931 |
| linear, log storage only | 0.525 | 0.456 |
| linear, log residence time only | 0.124 | 0.213 |
| linear, log storage + log residence time | 0.539 | 0.516 |
| linear, log storage + height + log drainage + log residence time | 0.582 | 0.538 |

Permutation importance of the boosted surrogate (drop in R2), seed 42 / 43: `log10_storage_max` 0.53 / 0.58,
`purpose_flood` 0.19 / 0.24, `log10_storage_per_area` 0.06 / 0.12, then `year`, `purpose_irrigation`,
`log10_surface(_missing)`, `log10_max_discharge` all below 0.07. `height` is last (0.02 / 0.01): its raw rank
correlation with T0 (0.55 / 0.40) is inherited from storage. Ridge coefficients put `purpose_flood` (+0.34 /
+0.32 in log10 d), the missing-value flags `year_missing` (+0.32 / +0.20) and `log10_surface_missing`
(-0.28 / -0.35), and `log10_storage_max` (+0.20 / +0.19) at the top; the flags are proxies for small,
poorly documented dams. Partial dependence on `log10_storage_max` is monotone and spans about 0.8 log10 units
(a factor 6 to 7 in T0) across the data range in both seeds; on `log10_storage_per_area` it spans about 0.3
(factor 2).

This section must be read for what it is: T0 is a deterministic function of exactly these 19 inputs (the
release head has no other information about a dam), so an R2 near 1 is guaranteed in principle and the
surrogate only shows which inputs the head chose to use and how. A correlation between T0 and storage
therefore says that training pushed the head to a storage-dependent release, not that T0 measures a physical
storage timescale. The fact that the physically motivated relative-size input (storage per area, a proxy for
residence time) ranks a distant third behind absolute storage and the flood flag is the informative part.

## 3. Agreement with independent T estimates

| comparison | n | seed 42 rho [CI], median learned/indep | seed 43 rho [CI], median learned/indep | within x2 (42 / 43) |
|---|---|---|---|---|
| offline seasonal fit, dam on gauge reach | 202 | 0.46 [0.35, 0.57], 0.56 | 0.50 [0.39, 0.59], 0.54 | 18 % / 18 % |
| offline linear fit, dam on gauge reach | 202 | 0.44 [0.32, 0.56], 1.03 | 0.46 [0.34, 0.56], 1.05 | 21 % / 25 % |
| offline seasonal, on reach, offline NSE > 0.5 | 122 | 0.38 [0.21, 0.52], 1.02 | 0.37 [0.19, 0.52], 1.16 | 17 % / 16 % |
| offline seasonal, all 339 (nearest large dam) | 339 | 0.39 [0.30, 0.48], 1.03 | 0.42 [0.33, 0.51], 1.15 | 17 % / 17 % |
| ResOpsUS fit, own inflow and release (no wall) | 39 | 0.12 [-0.25, 0.45], 0.13 | 0.03 [-0.31, 0.36], 0.14 | 13 % / 8 % |
| HydroLAKES residence time | 109 | 0.56 [0.40, 0.69], 0.006 | 0.57 [0.41, 0.70], 0.006 | 0 % |
| residence time, gauge proxy | 1024 | 0.36 [0.30, 0.42], 0.003 | 0.42 [0.36, 0.48], 0.004 | 0 % |

Offline bucket fits (same routed inflow, same 1981-1995 window, but one bucket fitted per gauge instead of a
head shared by 1,024 dams) rank the dams like the head does at rho about 0.45 to 0.5, with an unbiased median
ratio; but only about a fifth of dams agree within a factor 2. The offline seasonal fit has a wide search box
and 44 of 202 on-reach fits sit on its 0.05 d floor where the head's T0 is 0.1 to 1 d; those drive the
median ratio 0.56 for the seasonal variant (table4_worst_offline.csv: Camanche 2.9 d vs 0.05 d, Coyote Valley
1.7 d vs 0.05 d, Raystown 5.1 d vs 0.3 d). The other tail is dams whose offline fit went to hundreds of days
with an NSE near or below zero (Sumner, Courtright, Platoro: 1000 d fits at NSE 0.06, -3.5, 0.55) where the
head kept T0 under 1 d. The disagreement does not scale with residence time (rho -0.06 / -0.09) and only
weakly with storage (-0.14 / -0.22).

ResOpsUS fits from the dam's own inflow and release are longer by a median factor 7 to 8 (learned 1.3 to 1.5 d
against 15 d) and the learned T0 does not rank them (rho about 0). The ratio learned/ResOps falls steeply
with the ResOps T (rho -0.79 / -0.88): the head's range is compressed. At the short end it is close (Bagnell
1.6 d vs 1.0 d, Folsom 3.6 vs 3.5, Blue Marsh 1.3 vs 1.8, Chatfield 1.2 vs 1.9, W. Kerr Scott 2.1 vs 3.3); at
the long end it is not (Oroville 16 vs 66 d, Navajo 13 vs 161 d, Monticello/Berryessa 4.6 vs 588 d, Flaming
Gorge 9 vs 397 d, Green Mountain 0.5 vs 52 d). Three reasons, all consistent with the data: the routed inflow
reaching a dam is already attenuated by Muskingum-Cunge and by upstream dams, so the head sees less variance
to remove; 90-day training windows cannot express a multi-month memory (a T of 100 d would look like a
constant offset inside one window, and the L1 loss on the window is then better served by a short T plus
whatever the upstream roughness does); and the ResOps fits themselves are poorly constrained at the long end
(half have NSE < 0.5, Berryessa and Englebright have negative NSE, so their "T" is closer to a residence time
than to a release lag).

Residence time is a different quantity. Learned T0 is 0.3 to 0.6 % of storage / mean flow (median ratio about
1/300) and no dam is within a factor 2; the rank correlation (0.36 to 0.57 depending on the proxy and subset)
means larger relative storage gets a somewhat longer lag, not that T0 approximates the hydraulic residence
time. This is expected: a linear reservoir with T = 150 d would flatten the hydrograph to a near constant,
and gauges show that most dams here pass daily to weekly variation.

## 4. Seed stability

T0 ranks reproduce between seeds at rho 0.824 [0.799, 0.847] (n = 1024); seed 43 is longer by a median
factor 1.33 (log10 ratio 0.125, IQR 0.00 to 0.28) and 75 % of dams agree within a factor 2. Every
size correlation above has overlapping or near-overlapping CIs between seeds, with the same ordering
(max storage > normal storage > height, storage per area, residence time > drainage area). The largest seed
differences are height (0.55 vs 0.40) and max discharge (0.35 vs 0.09), both variables the surrogate says the
head barely uses; the differences are therefore in what the head inherits through collinearity, not in what it
targets. Surrogate importances agree between seeds (top three identical, same order); the partial dependence
curves overlay. The independent-estimate comparisons give the same rho and ratio in both seeds.

## Figures

- `/home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size/fig1_t0_vs_size.png`: log-log T0 vs normal storage,
  residence time (gauge proxy), height, drainage area, storage per upstream area, both seeds with rho and CI;
  sixth panel T0 by primary purpose.
- `/home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size/fig2_surrogate.png`: permutation importance of the
  boosted surrogate for both seeds; partial dependence on `log10_storage_max` and `log10_storage_per_area`.
- `/home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size/fig3_independent.png`: learned T0 against the offline
  seasonal fit (202 on-reach dams), the ResOpsUS fits (41), and residence time (1,024), with 1:1 lines.

## Tables

`table1_spearman_size.csv`, `table2_by_purpose.csv`, `table3_independent.csv`,
`table4_worst_offline.csv` (largest offline disagreements), `table5_resops.csv` (all 41 ResOps dams),
`table6_top_T0.csv` (twelve longest learned T0 with every independent estimate available).

## Caveats

- The head only sees the 19 features, so any T0-feature relation is a statement about the trained mapping.
  Physics enters only through the independent estimates in section 3, and there the agreement is rank-level
  at best (offline fits) or absent (ResOps).
- Residence-time proxies rely on mean flow assigned from a nearby gauge (median 13 km) or GRanD; the two agree
  to within about 25 % for most dams, but isolated bad assignments exist (Flaming Gorge).
- The offline fits share the routed inflow and the training window with the head, so they are independent of
  the head, not of the routing.
- Two seeds; CIs are bootstrap over dams, not over training randomness.
