# p = 21, NSE-batch, gages_3000: first dual win over summed-Q′ on CONUS: findings, 2026-09-08

**Run:** `.ddrs/runs/2026-09-08T15-55-52Z-conus-train-and-test` (branch `experiment-adjoint`, sha `d983739`).
**Question:** with the width coefficient p pinned at 21 (the 2024 setting) and only n and q learned, trained with the
batch NSE loss on the gages_3000 population, does routing earn its keep over summed inflow, and does pinning p cost
anything at the population median? Context: the landscape study (`docs/2026-09-08-landscape-hypothesis-tests-findings.md`
§5 to §10) had predicted from per-gauge loss surfaces that the Juniata wants n near 0.03 to 0.05, and that the n/p
degeneracy is what kept the learned-p models away from it.
**Plots and notebooks:** `<run>/plots/` (`metrics.ipynb`, `hydrograph.ipynb`, `parameter_maps.ipynb`,
`metrics_summary.json`, `parameter_convergence_stats.json`). Per-epoch parameter dumps: `<run>/plot/kan_parameters_epoch{1,10,20,30}.nc`.

## 1. Setup

| | |
|---|---|
| Population | `gages_3000.csv`, 2,859 gauges after the drainage-area filter in training; 2,365 scored at eval |
| Inflow | `merit_dhbv2_UH_retrospective.ic`, daily, no disaggregation (flat daily) |
| Windows | train 1981-10-01 to 1995-09-30, rho 90, warmup 5; test 1995-10-01 to 2010-09-30 |
| Head | KAN, hidden 21, 2 layers, grid 50, k 2, 10 attribute inputs; learnable `n`, `q_spatial`; `p_spatial` fixed 21 |
| Ranges | n [0.015, 0.25], q [0, 1] |
| Loss, optimizer | `nse-batch`, Adam, lr 0.005 / 0.001 / 0.0005 at epochs 1 / 11 / 21, grad clip 1.0, grad accum 20 micro-batches of 64 gauges |
| Epochs, seed | 30, seed 42 |
| Solver | CUDA sparse, CUDA graphs off, `enforce_positivity` off |
| Wall time | train 65 min, eval 34 min |

## 2. Result against the summed-Q′ baseline

Scored on the intersection of the run's predictions and its own baseline: 2,365 gauges × 5,477 days, identical
`ddr.validation.Metrics` code. The recomputed baseline agrees with the documented CONUS bar (0.6781 / 0.7172) to the
third decimal, so the population is the right one.

| 2,365 gauges | Baseline | Trained | Δ |
|---|---|---|---|
| Median NSE | 0.6785 | **0.7200** | **+0.042** |
| Median KGE | 0.7171 | **0.7537** | **+0.037** |
| Mean NSE | 0.452 | 0.559 | +0.107 |
| Median bias, m³/s | 0.96 | 0.35 | |
| Median FHV, % | +3.8 | −5.2 | |
| Median FLV, % | 48 | 37 | |
| Gauges where trained > baseline | | 69 % (NSE), 59 % (KGE) | |

Against the documented benchmarks (`ddrs-dev/references/research-status.md`):

| Run | NSE | KGE | Δ vs baseline |
|---|---|---|---|
| Precip-driven disagg + L1, `2026-06-23T02-49-12Z` (previous best) | 0.7152 | 0.7106 | +0.037 / −0.007 |
| Daily flat, L1, `2026-06-05T01-41-16Z` | 0.700 | 0.724 | +0.022 / +0.007 |
| **This run: daily flat, p = 21, nse-batch** | **0.7200** | **0.7537** | **+0.042 / +0.037** |

This is the first run on the dHBV2-UH store that beats the baseline on both metrics, and it does so with no
disaggregation head. The June journal's "structural ceiling" conclusion (daily routing over UH-routed inflow has no
generalizable skill beyond summed-Q′, `6_19_26_journal.md`) does not survive this run; that conclusion was drawn from
learned-p, L1 and KGE-loss models on the global store.

## 3. Where the gain is: basin size

| Drainage area, km² | Sites | Baseline median NSE | Trained median NSE | Δ |
|---|---|---|---|---|
| 0 to 1,000 | 1,267 | 0.694 | 0.716 | +0.02 |
| 1,000 to 5,000 | 803 | 0.688 | 0.727 | +0.04 |
| 5,000 to 10,000 | 157 | 0.613 | 0.724 | +0.11 |
| 10,000 to 30,000 | 126 | 0.562 | 0.722 | +0.16 |
| 30,000 to 50,000 | 12 | 0.347 | 0.571 | +0.22 |

The baseline degrades with basin size because unrouted inflow arrives too early and too peaky; the trained model holds
about 0.72 across every bin below 30,000 km². That is the signature of a routing correction rather than an inflow
correction, and it is why the population median (dominated by the 1,267 small basins) understates what routing does.

Spatially the trained NSE map (`plots/metrics_gauge_map.png`) shows the familiar dHBV pattern: low skill in the arid
Southwest, Great Basin, High Plains and Texas coast, high skill in the humid East and the Pacific Northwest.

## 4. Hydrographs

| Gauge | Area, km² | WY2000 NSE routed / summed | Full-period NSE routed / summed |
|---|---|---|---|
| 01567000 Juniata at Newport, PA | 8,657 | 0.816 / 0.584 | 0.858 / 0.695 |
| 02212600 Falling Creek near Juliette, GA (median gauge) | 188 | 0.659 / 0.689 | 0.720 / |
| 07301500 N. Fork Red River near Carter, OK (largest gain) | 6,885 | −0.005 / −1.93 | 0.567 / −0.775 |

At Newport every summed peak overshoots by 100 to 150 m³/s and the routed peaks land on the observed ones. At the
188 km² median gauge the routed and summed series overlap; routing neither fixes the inflow's missed February peaks nor
removes its spurious September event. At the Red River the whole gain is the removal of two phantom inflow spikes, and
the routed series still misses the observed recession. Routing corrects timing and peak magnitude; it cannot add or
remove inflow volume.

## 5. Learned parameters (all 346,321 CONUS reaches, epoch 30)

| | Median | Mean | p1 | p99 | Within 1 % of a bound |
|---|---|---|---|---|---|
| n | 0.053 | 0.061 | 0.022 | 0.132 | 0.02 % |
| q | 0.163 | 0.196 | 0.028 | 0.507 | 0.02 % |

Median by drainage area (20 bins of log10 area, `plots/parameter_scatter_*`):

| log10 area, km² | 1.5 | 2.0 | 3.0 | 4.0 | 5.0 | 5.8 | 6.3 |
|---|---|---|---|---|---|---|---|
| n | 0.062 | 0.056 | 0.048 | 0.043 | 0.040 | 0.035 | 0.026 |
| q | 0.20 | 0.17 | 0.14 | 0.12 | 0.11 | 0.08 | 0.05 |

The n map (`plots/parameter_map_n_conus.png`) is not uniform: 0.03 to 0.05 across the humid East, Gulf coast and central
plains; 0.10 to 0.15 in the mountain West, northern Rockies, Upper Midwest and Northeast. The q map is close to its
mirror image, low in the East and high across the Great Basin and Rockies. Both decline monotonically with basin size,
which is the direction hydraulic geometry expects (rougher, deepening headwaters; smoother, widening main stems).

This corrects the reading in the landscape findings §10 that the head's output is "close to CONUS-uniform" at 0.040 to
0.058. That was measured at four gauges that all sit in the low-n East. CONUS-wide the head spans a factor of six.

Open question: the high-n, high-q regions coincide with the poor-skill regions on the gauge map. That could be channel
physics (steep, coarse-bedded western channels) or compensation for bad inflow (slowing and attenuating spiky runoff).
The per-reach gradient instrument from the landscape study can separate these; see follow-ups.

## 6. Convergence

Per-epoch dumps at 1, 10, 20, 30 (`plots/parameter_convergence_*.png`, `parameter_convergence_stats.json`), compared
against the last run examined this way (`2026-07-30T01-58-07Z`, learned p, L1):

| Diagnostic | n | q | 2026-07-30, n |
|---|---|---|---|
| Median at epoch 1 / 10 / 20 / 30 | 0.130 / 0.057 / 0.054 / 0.053 | 0.49 / 0.19 / 0.17 / 0.16 | |
| IQR at epoch 1 / 10 / 30 | 0.0015 / 0.039 / 0.039 | 0.008 / 0.170 / 0.170 | 0.0011 / / 0.0027 |
| Late-half movement fraction | 0.009 | 0.010 | 0.59 |
| Median move epoch 20 to 30, % of range | 0.30 | 0.34 | |
| Realized p1 to p99 span, % of declared range | 47 | 48 | 4.5 |

The whole trajectory happened by epoch 10 (about 450 optimizer steps), the spread then held through both learning-rate
decays, and nothing is pinned at a bound. This is the first ddrs run that converged by these diagnostics; every earlier
run was still near initialization at epoch 30. The training-log batch median of n tells the same story (0.133 at epoch 1,
0.046 by epoch 10, 0.04 to 0.06 after), while the epoch-mean loss is too noisy to show it (0.29 to 0.36 throughout,
batches are random gauge subsets). Note the convergence template's "IQR trend" label reads "expanding" because it
compares epoch 1 to 30; read the per-epoch values.

## 7. What changed, and what we can and cannot attribute

Relative to the previous best (precip-disagg + L1, learned p), three things changed at once: p pinned at 21, the
`nse-batch` loss with Adam and gradient accumulation, and the gages_3000 population (which includes the Juniata). The
clean twin (`2026-09-08T14-06-12Z-train-and-test`, p = 21 on the 1,841-gauge area-balanced population) scored
0.700 / 0.736 against its learned-p sibling's 0.707 / 0.738, so pinning p by itself is neutral at the median. The
population is not comparable across those two runs and this one. What this run establishes:

1. Routing at daily resolution over UH-routed inflow does add generalizable held-out skill, and the skill lives in
   basins above 5,000 km².
2. The KGE regression that every L1 and NSE-loss run showed (the variance ratio α falling below 1) does not appear
   here at the median, although FHV moved from +4 % to −5 %, so peaks are now slightly under-predicted.
3. A p = 21 head with the NSE-batch objective converges in about 10 epochs on this population.

What it does not establish: which of the three changes carries the +0.04, and whether the result holds across seeds.

## 8. Caveats

- One seed. The CUDA scatter-add nondeterminism alone moves parameter distributions by 2 to 5 %.
- The eval population (2,365) overlaps the training population; this is the standard temporal split, not a spatial
  hold-out, the same as every benchmark row it is compared against.
- Newport full-period NSE here (0.858) is well above the WY2000-window values in the landscape study (about 0.70); the
  windows and warmup differ, so the two are not in conflict, but they should not be quoted interchangeably.
- The basemap convention changed during this analysis (CartoDB tiles are watermarked since 2026-09-08; the skill now
  uses Esri WorldGrayCanvas). Cosmetic only.

## 9. Follow-ups, in order

1. Seed replicate at identical settings to put an error bar on +0.042 / +0.037.
2. A 10-epoch run at the same settings: if it reproduces the score, the sweep cost drops threefold.
3. Ablate the loss: same population and p = 21 with L1, to separate the objective from the pinning.
4. Correlate per-reach n with the baseline NSE of the gauges downstream, and run the `channel_geometry` plot family
   to check the implied width-depth exponents against Leopold and Maddock, to decide whether the East-West n split is
   physics or inflow compensation.
5. Check whether the FHV sign flip is the NSE-batch objective attenuating peaks, the mechanism previously diagnosed for
   L1 (`src/training/loss.rs` header comment).
