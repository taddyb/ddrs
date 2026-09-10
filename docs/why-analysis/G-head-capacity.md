# G. Does head capacity (KAN depth/width) let n reach where each gauge wants it?

**Data.** `covariates.csv` p=21 census (config `config.yaml`, run `2026-09-08T15-55-52Z-conus-train-and-test`), filtered to well-fit gauges (nse0 > 0.3), excluding box_edge == 1 and hit_range_bound == 1: n = 1874 gauges. Target ln n* = ln(n0_med) + alpha_n_star. Inputs: the 10 attributes in `kan_head.input_var_names` (SoilGrids1km_clay, aridity, meanelevation, meanP, NDVI, meanslope, log10_uparea, SoilGrids1km_sand, ETPOT_Hargr, Porosity), read per gauge-reach COMID (via `gages_3000.csv` STAID->COMID, cross-checked against a gauge subgraph's `comid[gauge_reach_row]`) from `merit_global_attributes_v2.nc`, z-scored with the training statistics JSON (mean/std per attribute), NaNs filled to the training mean (z=0) as `finalize_attrs` does in `src/data/dataset.rs`.

**Weights.** Two independent schemes, each clipped and renormalised to mean 1: curvature (`lambda1`, clipped to [1e-4, 1]) and gain (`gain` = NSE(alpha*) - NSE(alpha=0), clipped to [0, 0.1]). All models are trained once per fold with the curvature weights as sample weight (kNN has no native sample-weighted fit, so it trains unweighted, as in analyst E's kNN-10 reference); both weight schemes are then applied at evaluation time to the same held-out predictions, plus an unweighted R2 and a train-set (curvature-weighted) R2 for overfitting.

**Protocol.** 5-fold CV, same folds for every model. Ridge: `RidgeCV` over alpha in {0.01,0.1,1,10,100}. kNN: k=10, unweighted. GBM: LightGBM, 500 trees, early-stopped on a 20% inner validation split (30-round patience). KAN: DDR's own `ddr.nn.kan.kan` (`Linear(F,H) -> KanLayer(H,H) x L -> Linear(H,1) -> sigmoid -> denormalize(n)`), Adam (lr=0.02), early-stopped on a 20% inner validation split (patience=15, max 120 epochs), 3 seeds averaged per fold. n is NOT in `params.log_space_parameters` for this config, so denormalize is the LINEAR branch (n = lo + sigmoid*(hi-lo)); the loss is weighted MSE on ln(n).


> **CORRECTION (2026-09-10, parent session). The four KAN rows below are invalid and must not be quoted.**
> All four variants report weighted R2 between -0.595 and -0.596, and a train R2 of -0.581. A constant predictor at
> the sigmoid midpoint (n = 0.1325, ln n = -2.021) scores exactly -0.581 on this target and weight vector. The four
> KAN arms therefore never left their initialisation: identical scores across depth and width, and a train score equal
> to the untrained constant, are the signature of a model that did not train (early stopping on a validation loss that
> never improved would restore the initial weights). Suspects: the Adam learning rate against grid-50 spline
> coefficients, or the patience of 15 firing before the splines move. The reference rows (ridge, kNN, GBM) are
> unaffected and carry the finding.
>
> **What survives.** On the ten attributes the head consumes, held-out weighted R2 for predicting ln n* is 0.101
> (ridge), 0.080 (kNN-10) and 0.160 (gradient boosting, the nonparametric ceiling). Even an unconstrained
> nonparametric model of these inputs explains about a sixth of the variance in the roughness the gauges want. That
> supports the INFORMATION verdict on its own, without any KAN arm.
>
> **What is NOT tested.** Whether more KAN layers would help. That claim needs either this offline test with a working
> optimiser, or the architecture sweep with real retraining. Do not cite the depth comparison until then.

## Results

| model | depth/width | R2 (curvature w) | R2 (gain w) | R2 (unweighted) | R2 (train) |
|---|---|---|---|---|---|
| current head (n0_med) | alpha=0 | -0.876 | -2.324 | -0.345 | nan |
| ridge | linear | 0.101 +/- 0.084 | -0.245 +/- 0.184 | 0.028 +/- 0.062 | 0.157 +/- 0.023 |
| knn10 | k=10 | 0.080 +/- 0.086 | -0.378 +/- 0.179 | 0.098 +/- 0.035 | 0.256 +/- 0.015 |
| gbm | lightgbm | 0.160 +/- 0.089 | -0.155 +/- 0.137 | 0.078 +/- 0.031 | 0.489 +/- 0.105 |
| kan_h21_l2 | H=21,L=2 (current) | -0.596 +/- 0.135 | -0.037 +/- 0.018 | -0.443 +/- 0.084 | -0.581 +/- 0.035 |
| kan_h21_l3 | H=21,L=3 (+1 layer) | -0.596 +/- 0.135 | -0.038 +/- 0.019 | -0.444 +/- 0.086 | -0.582 +/- 0.034 |
| kan_h21_l4 | H=21,L=4 (+2 layers) | -0.596 +/- 0.134 | -0.037 +/- 0.018 | -0.447 +/- 0.089 | -0.581 +/- 0.035 |
| kan_h42_l2 | H=42,L=2 (2x width) | -0.595 +/- 0.135 | -0.038 +/- 0.019 | -0.443 +/- 0.082 | -0.581 +/- 0.034 |

**Weight-implied noise floor (assumption).** We have no independent replicate of alpha_n_star per gauge, so we cannot compute an absolute noise variance from the weights alone (they are normalised to mean 1, a relative precision, not a calibrated one). As a stand-in, we treat the nonparametric ceiling's (GBM) held-out curvature-weighted R2 as the empirical estimate of the signal fraction any model of these inputs can reach; the complement (1 - R2_gbm = 0.84) is the fraction attributed to unmodelled noise plus information genuinely absent from the 10 attributes. This is a working definition, not a calibrated bound -- state it as such if reused.

## Observations

- The trained head's own n (alpha=0) reaches weighted R2 = -0.876 (curvature) / -2.324 (gain) against ln n* directly -- this is what 30 epochs of training already achieved on this population, with zero degrees of freedom spent fitting alpha_n_star itself.
- Ridge (linear in the 10 attributes) reaches 0.101 +/- 0.084; kNN-10 reaches 0.080 +/- 0.086; GBM (nonparametric ceiling) reaches 0.160 +/- 0.089.
- The current-depth KAN (H=21,L=2 (current)) reaches -0.596 +/- 0.135. Adding capacity moves it to: H=21,L=3 (+1 layer) -> -0.596 +/- 0.135; H=21,L=4 (+2 layers) -> -0.596 +/- 0.134; H=42,L=2 (2x width) -> -0.595 +/- 0.135.
- Best KAN variant: H=42,L=2 (2x width) at -0.595 +/- 0.135, at or below the GBM ceiling (0.160 +/- 0.089).
- Train-vs-held-out gap for the best KAN variant: -0.581 (train) vs -0.595 (held out), gap = 0.014 (small, not overfitting).

## Interpretation

- If every KAN variant (current depth through +2 layers and 2x width) lands within noise of each other and of the linear ridge reference, and all sit near the GBM ceiling, added KAN capacity is not the bottleneck: the 10 attributes the head consumes do not encode where a gauge wants n, regardless of how flexible the function mapping attributes to n is allowed to be. This is an INFORMATION problem, consistent with the E-series finding that a 10-neighbour attribute regression alone reaches R2 around 0.12 on a closely related population, and that the dominant covariate of the gap (routed-flow timing lag) is not among the 10 inputs at all.
- If deeper/wider KAN variants show a clear, monotonic, out-of-fold gain over the current depth (train R2 also rising without a corresponding held-out gap), the shared head was genuinely under-fitting its own inputs, and depth is worth the wall-clock cost of a real retrain.

## Verdict

**INFORMATION, on the reference models.** The nonparametric ceiling on these inputs is a weighted R2 of 0.160, so the ten attributes carry little of what sets the gauge-optimal roughness. The KAN depth comparison that would have tested capacity directly did not run (see the correction above). Superseded claim, kept for the record: more KAN layers/width do not move the held-out fit to ln n* materially beyond the current architecture or the GBM ceiling on the same inputs. The head cannot place n where a gauge wants it because the 10 attributes it sees do not carry that signal, not because the function class is too small.

**Decisive follow-up.** Retrain the real KAN head (ddrs, on-graph, with routing) at the best capacity variant found here on the same gages_3000 population and re-run the census (`covariates.py`) to see whether alpha_n_star and gain actually shrink -- this offline test only checks whether the function class *could* separate the training signal from noise on frozen inputs; it cannot rule out optimisation or batch-compromise effects that only appear when many gauges share one gradient (see finding E).

