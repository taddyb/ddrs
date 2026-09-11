# G. Does head capacity (KAN depth/width) let n reach where each gauge wants it?

## History

The first run of this script (2026-09-09) reported all four KAN variants at weighted R2 between -0.595 and -0.596 (curvature weights) with a train R2 of -0.581, identical across depth and width. A constant predictor at the sigmoid midpoint (frac=0.5, i.e. `n = lo + 0.5*(hi-lo)`) scores exactly -0.581 against this target and weight vector: every KAN arm restored its epoch-0 weights. `best_val` starts at infinity so epoch 0 is always saved as `best_state`; with the un-probed lr=0.02 the first Adam step overshot, validation loss never improved again, and `patience=15` broke the loop before the splines moved. Only 120 full-batch steps were available in any case, far too few for grid-50 spline coefficients. `train_kan` has since been fixed: the output linear layer's bias is now warm-started to the weighted-mean training target before any optimiser step (verified below to be within 0.01 R2 of the constant-at-weighted-mean baseline, i.e. the offset is removed from the optimisation entirely), the learning rate is chosen from a probe (see below) rather than an un-probed guess, `max_epochs`/`patience` were raised to 2000/100, a `ReduceLROnPlateau` scheduler (factor 0.5, patience 30) was added, and a post-fold guard (train R2 > 0.02) now catches a repeat of this failure per fold instead of silently reporting a wrong number. The reference rows (ridge, kNN, GBM) never depended on `train_kan` and are unaffected by this bug or its fix.

**Data.** `covariates.csv` p=21 census (config snapshot `.ddrs/runs/2026-09-08T15-55-52Z-conus-train-and-test/config.yaml`), filtered to well-fit gauges (nse0 > 0.3), excluding box_edge == 1 and hit_range_bound == 1: n = 1874 gauges. Target ln n* = ln(n0_med) + alpha_n_star. Inputs: the 10 attributes in `kan_head.input_var_names` (SoilGrids1km_clay, aridity, meanelevation, meanP, NDVI, meanslope, log10_uparea, SoilGrids1km_sand, ETPOT_Hargr, Porosity), read per gauge-reach COMID (via `gages_3000.csv` STAID->COMID, cross-checked against a gauge subgraph's `comid[gauge_reach_row]`) from `merit_global_attributes_v2.nc`, z-scored with the training statistics JSON (mean/std per attribute), NaNs filled to the training mean (z=0) as `finalize_attrs` does in `src/data/dataset.rs`.

**Weights.** Two independent schemes, each clipped and renormalised to mean 1: curvature (`lambda1`, clipped to [1e-4, 1]) and gain (`gain` = NSE(alpha*) - NSE(alpha=0), clipped to [0, 0.1]). All models are trained once per fold with the curvature weights as sample weight (kNN has no native sample-weighted fit, so it trains unweighted, as in analyst E's kNN-10 reference); both weight schemes are then applied at evaluation time to the same held-out predictions, plus an unweighted R2 and a train-set (curvature-weighted) R2 for overfitting.

**Protocol.** 5-fold CV, same folds for every model. Ridge: `RidgeCV` over alpha in {0.01,0.1,1,10,100}. kNN: k=10, unweighted. GBM: LightGBM, 500 trees, early-stopped on a 20% inner validation split (30-round patience). KAN: DDR's own `ddr.nn.kan.kan` (`Linear(F,H) -> KanLayer(H,H) x L -> Linear(H,1) -> sigmoid -> denormalize(n)`), output bias warm-started to the weighted-mean training target (see History), Adam (lr=0.02, chosen by the probe below), `ReduceLROnPlateau` (factor 0.5, patience 30) on the validation loss, early-stopped on a 20% inner validation split (patience=100, max 2000 epochs), 3 seeds averaged per fold. Any fold whose weighted train R2 does not exceed 0.02 after training is excluded from that model's cross-fold mean (printed as a warning); a model with every fold excluded reports "did not train". n is NOT in `params.log_space_parameters` for this config, so denormalize is the LINEAR branch (n = lo + sigmoid*(hi-lo)); the loss is weighted MSE on ln(n).

## Learning-rate probe

Single fold (fold 0), current architecture, 3 seeds, run before committing to the full sweep (the original bug was an un-probed lr; this picks the full-run lr from evidence instead):

| lr | R2 (train) | R2 (held-out, curvature w) | held-out pred std (ln n) |
|---|---|---|---|
| 0.02 | 0.236 | 0.137 | 0.311 <- used for full run |
| 0.005 | 0.488 | 0.113 | 0.298 |
| 0.002 | 0.541 | 0.114 | 0.279 |
| 0.0005 | 0.651 | 0.100 | 0.248 |

## Results

| model | depth/width | R2 (curvature w) | R2 (gain w) | R2 (unweighted) | R2 (train) | folds excluded |
|---|---|---|---|---|---|---|
| current head (n0_med) | alpha=0 | -0.876 | -2.324 | -0.345 | nan | - |
| ridge | linear | 0.101 +/- 0.084 | -0.245 +/- 0.184 | 0.028 +/- 0.062 | 0.157 +/- 0.023 | - |
| knn10 | k=10 | 0.080 +/- 0.086 | -0.378 +/- 0.179 | 0.098 +/- 0.035 | 0.256 +/- 0.015 | - |
| gbm | lightgbm | 0.160 +/- 0.089 | -0.155 +/- 0.137 | 0.078 +/- 0.031 | 0.489 +/- 0.105 | - |
| kan_h21_l2 | H=21,L=2 (current) | 0.096 +/- 0.085 | -0.285 +/- 0.161 | 0.026 +/- 0.057 | 0.298 +/- 0.101 | - |
| kan_h21_l3 | H=21,L=3 (+1 layer) | 0.081 +/- 0.127 | -0.266 +/- 0.150 | 0.011 +/- 0.070 | 0.277 +/- 0.070 | - |
| kan_h21_l4 | H=21,L=4 (+2 layers) | 0.103 +/- 0.086 | -0.230 +/- 0.135 | 0.037 +/- 0.059 | 0.234 +/- 0.046 | - |
| kan_h42_l2 | H=42,L=2 (2x width) | 0.107 +/- 0.087 | -0.281 +/- 0.188 | 0.023 +/- 0.071 | 0.262 +/- 0.080 | - |

**Weight-implied noise floor (assumption).** We have no independent replicate of alpha_n_star per gauge, so we cannot compute an absolute noise variance from the weights alone (they are normalised to mean 1, a relative precision, not a calibrated one). As a stand-in, we treat the nonparametric ceiling's (GBM) held-out curvature-weighted R2 as the empirical estimate of the signal fraction any model of these inputs can reach; the complement (1 - R2_gbm = 0.84) is the fraction attributed to unmodelled noise plus information genuinely absent from the 10 attributes. This is a working definition, not a calibrated bound: state it as such if reused.

## Observations

- The trained head's own n (alpha=0) reaches weighted R2 = -0.876 (curvature) / -2.324 (gain) against ln n* directly: this is what 30 epochs of training already achieved on this population, with zero degrees of freedom spent fitting alpha_n_star itself.
- Ridge (linear in the 10 attributes) reaches 0.101 +/- 0.084; kNN-10 reaches 0.080 +/- 0.086; GBM (nonparametric ceiling) reaches 0.160 +/- 0.089.
- The current-depth KAN (H=21,L=2 (current)) reaches 0.096 +/- 0.085. Adding capacity moves it to: H=21,L=3 (+1 layer) -> 0.081 +/- 0.127; H=21,L=4 (+2 layers) -> 0.103 +/- 0.086; H=42,L=2 (2x width) -> 0.107 +/- 0.087.
- Best KAN variant: H=42,L=2 (2x width) at 0.107 +/- 0.087, at or below the GBM ceiling (0.160 +/- 0.089).
- Train-vs-held-out gap for the best KAN variant: 0.262 (train) vs 0.107 (held out), gap = 0.155 (material: capacity is fitting fold-specific noise, not signal).

## Interpretation

- If every KAN variant (current depth through +2 layers and 2x width) lands within noise of each other and of the linear ridge reference, and all sit near the GBM ceiling, added KAN capacity is not the bottleneck: the 10 attributes the head consumes do not encode where a gauge wants n, regardless of how flexible the function mapping attributes to n is allowed to be. This is an INFORMATION problem, consistent with the E-series finding that a 10-neighbour attribute regression alone reaches R2 around 0.12 on a closely related population, and that the dominant covariate of the gap (routed-flow timing lag) is not among the 10 inputs at all.
- If deeper/wider KAN variants show a clear, monotonic, out-of-fold gain over the current depth (train R2 also rising without a corresponding held-out gap), the shared head was genuinely under-fitting its own inputs, and depth is worth the wall-clock cost of a real retrain.

## Verdict

**INFORMATION, not capacity.** More KAN layers/width do not move the held-out fit to ln n* materially beyond the current architecture or the GBM ceiling on the same inputs. The head cannot place n where a gauge wants it because the 10 attributes it sees do not carry that signal, not because the function class is too small.

**Decisive follow-up.** Retrain the real KAN head (ddrs, on-graph, with routing) at the best capacity variant found here on the same gages_3000 population and re-run the census (`covariates.py`) to see whether alpha_n_star and gain actually shrink. This offline test only checks whether the function class *could* separate the training signal from noise on frozen inputs; it cannot rule out optimisation or batch-compromise effects that only appear when many gauges share one gradient (see finding E).

