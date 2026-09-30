# Reservoir benchmark: up to ten dams per HUC2

**Date:** 2026-09-26
**Script:** `experiments/reservoir/benchmark/select_dam_benchmark.py` (run
`experiments/reservoir/regulation_sizing.py` first). **Outputs:** `experiments/reservoir/benchmark/dam_benchmark.csv`
(121 rows), `dam_benchmark_summary.json`; every candidate with its tags in
`output/dam_sandbox/dam_benchmark_candidates.csv`.
**Follows:** `2026-09-25-reservoir-representation-options.md` (option C is implemented; options B, D and E need
dams with release data to test against).
**Purpose (user):** a fixed benchmarking set of about ten dams per HUC so that any reservoir representation is
scored by its median NSE at a known sample size per region.

## 1. Data used

| What | Where | Fetched |
|---|---|---|
| GRanD dams mapped to MERIT COMIDs (the NWM / RFC-DA set) | `/mnt/ssd1/data/resops/derived/grand_to_merit_comid.csv` | 2026-09-25 |
| ResOpsUS v2 daily inflow, outflow, storage | `/mnt/ssd1/data/resops/resopsus/` | 2026-09-25 |
| ISTARF-CONUS release rules; GRanD and GloFAS attributes (ResOpsUS+CARS) | `/mnt/ssd1/data/resops/{istarf_conus,resopsus_cars}/` | 2026-09-25 |
| NWIS annual peak-flow qualification codes, 2,365 eval gauges | `/mnt/ssd1/data/usgs_regulation/derived/peak_regulation_summary.csv` | 2026-09-26 |
| GAGES-II basin characteristics (dams, classification) | `/mnt/ssd1/data/usgs_regulation/derived/gagesii_regulation.csv` | 2026-09-26 |

Fetch scripts and provenance: `~/projects/remote_sensing_extraction/{resops,usgs_regulation}/`.

USGS publishes no release attributes per gauge (no release capacity, maximum release or operating rule). What it
publishes is evidence of regulation: the peak-flow qualification code 6 ("discharge affected by regulation or
diversion") per site and year, GAGES-II dam counts and storage (static, about 2009), and free-text station
remarks that no API exposes. Release information per dam comes from ResOpsUS (observed), ISTARF-CONUS (fitted or
extrapolated STARFIT rules) and GloFAS (`Qmin`, `Qn`, `Qf`).

## 2. Design

- **Pool:** the 2,177 GRanD dams of the NWM / RFC-DA reservoir table that map to a MERIT COMID. This table misses
  dams outside the NWM set (Alamo Dam, for one).
- **Gauge:** the nearest downstream gauge of the 2,365-gauge eval population (fewest upstream reaches among the
  gauges whose subgraph contains the dam's COMID), with gauge drainage area at most 1.5 times the dam's HydroLAKES
  watershed area.
- **Regulated:** the gauge carries peak code 6 in at least one year of WY1996-2010.
- **One dam per gauge:** the dam nearest the gauge (area ratio closest to 1), whose release the gauge records. The
  largest upstream volume among the gauge's candidate dams is kept as a tag.
- **HUC2:** HUC02 of the gauge (GAGES-II).
- **Selection per HUC2 (user):** ResOpsUS dams first, then fill to ten evenly spaced in gauge drainage-area rank,
  within three tiers: 1 = ResOpsUS with at least 5 years of overlapping daily inflow and outflow in WY1996-2010,
  2 = other ResOpsUS dams, 3 = the rest. Regions with fewer than ten eligible dams keep all of them, and the sample
  size is reported next to the median (user).
- **Skill:** trained run `2026-09-17T16-38-16Z-train-and-test` and the summed-Q' baseline, WY1997-2010, as in
  `regulation_sizing.py`. Bootstrap 95 % interval on each median (2,000 resamples, seed 42).

Funnel: 2,177 dams, 959 with a downstream eval gauge, 274 within the area ratio, 217 of those with code 6,
204 distinct gauges, **121 benchmark dams**.

## 3. The benchmark and its current skill

| HUC2 | n | tier 1 | median NSE trained [95 % CI] | median NSE summed Q' | median KGE trained | median gauge DOR |
|---|---:|---:|---|---:|---:|---:|
| 01 New England | 10 | 0 | 0.630 [0.358, 0.850] | 0.553 | 0.761 | 0.50 |
| 02 Mid-Atlantic | 10 | 2 | 0.535 [0.272, 0.657] | 0.326 | 0.701 | 0.49 |
| 03 South Atlantic-Gulf | 10 | 3 | 0.291 [0.002, 0.602] | -0.026 | 0.598 | 0.84 |
| 04 Great Lakes | 6 | 0 | 0.461 [0.253, 0.740] | 0.312 | 0.608 | 1.02 |
| 05 Ohio | 10 | 3 | 0.725 [0.533, 0.830] | 0.681 | 0.805 | 0.48 |
| 06 Tennessee | 0 | | | | | |
| 07 Upper Mississippi | 9 | 0 | 0.630 [0.509, 0.852] | 0.633 | 0.761 | 0.35 |
| 08 Lower Mississippi | 0 | | | | | |
| 09 Souris-Red-Rainy | 1 | 0 | 0.581 | 0.664 | 0.300 | 0.52 |
| 10 Missouri | 10 | 6 | 0.331 [0.128, 0.498] | -0.029 | 0.503 | 1.64 |
| 11 Arkansas-White-Red | 4 | 1 | 0.439 [0.166, 0.577] | 0.322 | 0.539 | 1.36 |
| 12 Texas-Gulf | 3 | 0 | 0.458 [0.424, 0.705] | 0.549 | 0.669 | 1.79 |
| 13 Rio Grande | 6 | 1 | 0.080 [-0.116, 0.231] | -0.057 | 0.395 | 4.90 |
| 14 Upper Colorado | 10 | 10 | 0.161 [0.027, 0.551] | 0.076 | 0.514 | 1.08 |
| 15 Lower Colorado | 2 | 0 | -6.285 [-12.87, 0.301] | -21.84 | -0.598 | 3.19 |
| 16 Great Basin | 10 | 4 | 0.369 [0.256, 0.509] | 0.380 | 0.490 | 1.34 |
| 17 Pacific Northwest | 10 | 3 | 0.394 [0.336, 0.469] | 0.397 | 0.461 | 0.41 |
| 18 California | 10 | 10 | 0.516 [0.274, 0.610] | 0.472 | 0.542 | 2.85 |
| **All** | **121** | **43** | **0.458 [0.387, 0.508]** | **0.371** | **0.579** | |

By tier: tier 1 (43 dams) 0.432, tier 2 (39) 0.331, tier 3 (39) 0.638. Tier 3 is mostly eastern dams with lower
DOR, so tier and region are confounded; do not read the tier medians as an effect of having ResOpsUS data.

Release data on the 121: an ISTARF rule fitted to observations at 76, extrapolated from another dam at 43,
storage-only at 2; GloFAS `Qf` at 22; ResOpsUS records at 82 (tiers 1 and 2); GRanD purpose at 82 (flood control
38, irrigation 24, hydropower 11, water supply 7, other 2). The four sandbox dams: Raystown (HUC 02, tier 3: it is
not in ResOpsUS), Abiquiu and Santa Rosa (HUC 13) are in; Alamo is not in the NWM table.

## 4. Reading it

- **Nine regions have ten dams** (01, 02, 03, 05, 10, 14, 16, 17, 18). 04 and 07 nearly do (6 and 9). 09, 11, 12,
  13 and 15 have one to six, and 06 and 08 none; their medians carry the sample size and should not be compared
  with the others as equals. HUC 15's -6.3 is two dams, one of them Coolidge Dam at -12.9.
- **At n = 10 a regional median is uncertain by about 0.2 to 0.6 NSE** (the bootstrap intervals above). A change
  in reservoir representation will only show in a regional median if it is large. The stronger comparison is
  paired: the same 121 gauges before and after, the per-gauge NSE difference, and its sign test or median with an
  interval.
- **The regions where the trained model is worst below dams are the arid West and the plains**: 13 (0.08),
  14 (0.16), 03 (0.29), 10 (0.33), with gauge DOR above 1 except in 03. That matches the population result that
  the skill deficit sits above DOR 0.5 (options doc §3).
- **The trained model beats the summed Q' at the median in most regions** (overall 0.458 against 0.371), so
  routing already recovers part of the timing below dams; the benchmark measures what a reservoir term adds on top.

## 5. Caveats

- Code 6 comes from annual peaks only. A gauge whose peaks are regulated but whose daily record is not dominated
  by the dam still passes.
- HydroLAKES `Wshd_area` stands in for the dam catchment; the benchmark pairs have area ratio 0.88 to 1.5 (median
  1.02).
- One trained run and one window. The benchmark definition does not depend on the run; its current numbers do.
- The NWM / RFC-DA reservoir table is incomplete; the lake name is blank for some dams (El Vado) in the crosswalk.

## 6. Next

1. Fit a linear-reservoir `T` from ResOpsUS inflow and outflow for the 43 tier-1 dams (as in
   `linear_reservoir_diagnostics.py`, same-day inflow), write the `data_sources.reservoirs` table, and evaluate
   option C on the benchmark gauges against the current numbers, paired per gauge.
2. Use the ISTARF rules at the 76 dams where they are fitted to observations for option E.
