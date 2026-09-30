# Reservoir representation in ddrs: the linear reservoir, the literature, and options

**Date:** 2026-09-25
**Scripts:** `experiments/reservoir/linear_reservoir_diagnostics.py`,
`experiments/reservoir/threshold_reservoir.py [--inflow modelled]` (offline, no ddrs code, same
inputs as `dam_sandbox.py`), `experiments/reservoir/regulation_sizing.py`, then
`experiments/reservoir/release_gauge_coverage.py` (needs the sizing CSV). Run under DDR's venv,
e.g. `~/projects/ddr/.venv/bin/python experiments/reservoir/threshold_reservoir.py`. Results in
`experiments/reservoir/results/{linear_diagnostics,threshold_reservoir,threshold_reservoir_modelled,regulation_sizing,release_gauge_coverage}.json`;
figures `output/dam_sandbox/threshold_hydrographs{,_modelled}.png`; per-gauge table
`output/dam_sandbox/regulation_by_gauge.csv`.
**Follows:** `2026-09-22-raystown-storage-law-sandbox.md`, `2026-09-25-arid-dam-sandboxes.md`.
**Goal (user, this day):** a better representation of reservoir flows in river routing. Step 1 is
the linear reservoir that exists, step 2 a literature search, step 3 options.

## Summary

1. **The linear reservoir is better than the sandbox reported, and it is already expressible in
   ddrs.** Driven by same-day inflow instead of the previous day's, it scores 0.800 / 0.732
   (train / test NSE) at Raystown, not 0.737 / 0.650. It is exactly a Muskingum row with `X = 0`
   and `K = T`, and a free Muskingum fit picks `X = 0` at every dam, so ddrs could host it by
   overriding one row's coefficients.
2. **Its main error is a release cap.** Raystown and Abiquiu release up to a ceiling whatever the
   inflow. `min(S / T, Q_max)` lifts test NSE by 0.13 at Raystown and 0.10 at Abiquiu given
   observed inflow, but only 0.02 at Raystown given the trained model's inflow: a parametric
   node is only as good as the inflow reaching it. The arid dams stay at NSE 0 under every
   storage-state rule; their release is a schedule.
3. **At population scale the cost is concentrated above DOR 0.5**: 347 of 2,365 eval gauges (a
   lower bound), median NSE 0.495 against 0.739 unregulated, 0.12 to 0.38 below unregulated gauges
   of the same size in every area bin, and mostly timing rather than volume.
4. **The literature agrees on the two points that matter.** The laws tried so far are lake laws,
   and operating rules are not identifiable from downstream discharge; the error they leave lands
   in other parameters, Manning's n included.
5. **Recommendation:** first, mask the DOR > 0.5 gauges from the loss and measure whether the
   learned `n` elsewhere moves (A). Then build one identity-row mechanism and use it for observed
   releases (B, up to 216 of the 347) and, where there is no gauge, a linear or capped reservoir
   with prescribed `Q_max` (C, D). Fit operating rules offline to ResOpsUS only if the scheduled
   dams earn it (E).

## 1. What exists

| Where | What | Status |
|---|---|---|
| ddrs `src/` | Nothing. A reach inside a reservoir is routed as a prismatic MC channel. | |
| `experiments/reservoir/dam_sandbox.py` | Linear reservoir `Q = S / T`, implicit Euler, daily, `T` fitted; dMC-dev fill-fraction law `Q = Q0 (S / S0)^b`; pass-through. Observed inflow (upstream gauges + local Q'), observed release. | Offline sandbox, four dams |
| dMC-dev (PR #67, #79, 2024) | The fill-fraction law with `S0` = total capacity. | Refuted at all four dams: response time bounded below by `S0 / (b theta Q_ref)` (Raystown doc §3, §5); outside the model class at the arid dams (arid doc §2) |
| DDR #137 to #139 (2026-02) | Level pool: weir + orifice outflow from HydroLAKES-derived geometry (orifice at 15 % of depth, weir at 90 %, weir length 1 % of shoreline, orifice area back-calculated from `Dis_avg`), no learned parameters; its doc counts 46,095 CONUS reaches intersecting HydroLAKES, while the parameter table left on disk covers 2,178 COMIDs. Integrated as an identity row in the sparse solve (`c1 = 0`, `b = Q_out(H_t)`), pool elevation carried as state, forward-Euler pool update clamped to `[OE, WE + D]`. | Reverted in `b3f122e` (#143) with eight other features as "unstable". No gauge-level evaluation is recorded in the PRs or the revert. Parameter tables survive in `~/projects/ddr/data/` (`hydrolakes_rfc_da.csv` 6,797 lakes with `Grand_id`, `merit_reservoir_params.csv` 2,178 COMIDs) |

The DDR integration pattern is worth keeping even though the physics was not validated: an identity
row whose right-hand side is the reservoir's release keeps one sparse solve per timestep, keeps the
CSR pattern fixed (only values change), and propagates the release downstream in the same forward
substitution. Every option in §5 that is not a pure parameter override uses it.

## 2. The linear reservoir, examined

### 2.1 It is a Muskingum row with X = 0

`Q = S / T` with inflow `I` is the Muskingum storage relation `S = K [X I + (1 - X) Q]` at `X = 0`,
`K = T`. Discretised by Muskingum's trapezoid it gives `c1 = c2 = dt / (2T + dt)`,
`c3 = (2T - dt) / (2T + dt)`, which is what `src/routing/mmc.rs` computes for a reach whose
`k = L / c` equals `T` and whose `x_storage` is 0. Fitting Muskingum `(K, X)` freely at the dam
selects `X = 0` at all four dams (`linear_diagnostics.json`, `muskingum_KX`). So a linear
reservoir at a dam needs no new solver machinery in ddrs: override `k` and `x_storage` on the
dam's row. At the hourly step `dt = 3600 s` and the fitted `T` of 1 to 6 days, `c3 > 0` and all
coefficients are non-negative.

### 2.2 `dam_sandbox.py` understated it

`simulate_linear` drives the reservoir with the previous day's inflow (`inflow[t - 1]`), while
pass-through uses the same day's. The lag costs the linear reservoir 0.06 train and 0.08 test NSE
at Raystown:

| Dam | Pass-through | Linear, inflow lag 1 (sandbox) | Linear, same-day inflow | Muskingum (K, X) fitted |
|---|---|---|---|---|
| Raystown | 0.650 / 0.532 | 0.737 / 0.650, T 0.67 d | **0.800 / 0.732, T 1.23 d** | 0.797 / 0.715, X = 0 |
| Abiquiu | 0.646 / 0.514 | 0.730 / 0.662, T 5.5 d | **0.743 / 0.677, T 5.8 d** | 0.738 / 0.671, X = 0 |
| Alamo | −155 / −2.05 | 0.02 / 0.01, T 2,153 d | 0.02 / 0.01, T 2,153 d | same |
| Santa Rosa | −0.16 / −0.39 | 0.01 / 0.01, T 458 d | 0.01 / −0.04, T 47 d | same |

Train WY1997-2001 / test WY2002-2010, daily NSE. The arid doc's table reports the lagged column.
Its ordering holds (the storage law never beats the linear reservoir) and the gap widens.

### 2.3 What T is, and how well it is identified

- **Not capacity.** The fitted `T` times mean inflow is an effective buffer of 3.8 MCM at Raystown
  and 6.4 MCM at Abiquiu, 0.4 % of total capacity at both. Capacity from GRanD or HydroLAKES
  cannot set `T`; the Raystown doc's "operational buffer, not total volume" reading holds for the
  linear form too.
- **Identified where the dam passes events through.** The range of `T` within 0.02 train NSE of
  the optimum is 0.64 to 2.03 d at Raystown and 2.5 to 10.7 d at Abiquiu, a factor of 3 to 4.
- **Unidentified or degenerate at the scheduled dams.** 20 to 2,400 d at Santa Rosa. At Alamo `T`
  runs to about 2,000 d, where the output is a constant near the mean and NSE is 0.
- **Mass balance is not what is missing.** A fitted volume factor moves NSE by < 0.01 at Raystown,
  Abiquiu and Alamo. At Santa Rosa release is 0.72 of inflow volume (evaporation and diversion),
  and scaling still leaves NSE at 0.

### 2.4 What it cannot do: capped releases

The largest test-period inflow peaks against the maximum release over the following five days:

| Dam | Inflow peaks, m³/s | Observed release peaks, m³/s | Linear reservoir peaks, m³/s |
|---|---|---|---|
| Raystown (8 events) | 338 to 1,031 | 292 to 413, and 413 for the 1,031 event (Ivan, 2004-09-18) | 213 to 534 |
| Abiquiu (8 events) | 52 to 156 | 41 to 53 | 41 to 96 |

Release saturates at a ceiling regardless of inflow, which is how flood-control dams are operated
(release up to the downstream channel's safe capacity, store the rest). The linear reservoir
scales every peak with inflow, so it underestimates the moderate events and overshoots the
largest. Even the two dams that looked "near pass-through" are threshold-operated.

### 2.5 One feature closes most of the gap: a release cap

`threshold_reservoir.py` fits nested extensions of the same-day linear reservoir,
`Q = min(max(S − S_d, 0) / T, Q_max)` with a constant withdrawal `E` from storage:

| Dam | Linear | + cap | + cap + withdrawal | + pool + cap + withdrawal |
|---|---|---|---|---|
| Raystown | 0.799 / 0.727 | **0.877 / 0.856**, `Q_max` 288 [244, 339], T 0.33 d | 0.879 / 0.854 | 0.879 / 0.854 |
| Abiquiu | 0.743 / 0.673 | 0.744 / **0.774**, `Q_max` 54 [46, inf] | 0.744 / 0.774 | same |
| Alamo | −0.40 / 0.04 | −0.38 / −0.01 | −0.09 / −0.01 | −0.08 / −0.01 |
| Santa Rosa | 0.01 / 0.01 | 0.01 / 0.01 | 0.02 / 0.01 | 0.02 / 0.01 |

Brackets: range within 0.02 train NSE of the optimum, on a 40-point log grid in `Q_max`.

- **Raystown:** the cap adds 0.13 test NSE and is identified to about ±17 % from five years of
  release record. The capped model reproduces the clipped peaks and the flat evacuation after
  Ivan.
- **Abiquiu:** the cap adds 0.10 test NSE and lands on the observed 48 to 53 m³/s snowmelt
  plateau, but the training years barely reach it, so the band includes `Q_max = inf`: the cap is
  right but not identified from WY1997-2001 alone. The observed plateau also lasts three weeks
  longer than the model's, so the dam holds more water than `T = 5.5 d` implies.
- **Withdrawal and pool add nothing** at either dam. (With `E = 0` the pool is a no-op by
  construction, since storage starts at it.)
- **The arid dams do not move under any storage-state rule.** Alamo evacuates at about
  200 m³/s weeks after each flood; Santa Rosa's 39 m³/s September 2004 block has no inflow behind
  it. The release is a schedule, not a function of storage.

![threshold hydrographs](../../output/dam_sandbox/threshold_hydrographs.png)

### 2.6 What the trained model does at these gauges

Trained ddrs (`2026-09-17T16-38-16Z-train-and-test`, summed Q' inflow, not observed inflow),
NSE WY1997-2001 / WY2002-2010:

| Gauge | Trained ddrs | Volume ratio, WY1996-2010 |
|---|---|---|
| Raystown Branch at Saxton, 01562000 (above the lake) | 0.893 / 0.842 | 0.89 |
| below Raystown Dam, 01563200 | 0.692 / 0.622 | 0.90 |
| Rio Chama above Abiquiu, 08286500 | 0.139 / 0.195 | 1.17 |
| below Abiquiu Dam, 08287000 | 0.074 / 0.180 | 1.14 |
| below Alamo Dam, 09426000 | −18.1 / 0.42 | 1.28 |
| Pecos above Santa Rosa Lake, 08382650 | 0.364 / 0.379 | 1.26 |
| below Santa Rosa Dam, 08382830 | −0.08 / −0.24 | 1.48 |

The routing loses 0.22 NSE across Raystown Dam, which the capped reservoir recovers in the
sandbox given observed inflow (and mostly does not given modelled inflow, §2.7). Abiquiu's error is inherited: the gauge above it is already at
0.2, because the Rio Chama above Abiquiu is itself regulated (El Vado, and the San Juan-Chama
imports through Heron), so a reservoir at Abiquiu alone cannot fix it. Alamo's 0.42 is set by a
few evacuation events and changes sign between windows; NSE carries no information there.

### 2.7 Does the gain survive modelled inflow?

The fits above feed the reservoir observed upstream flow. Inside ddrs the dam sees routed Q'.
`threshold_reservoir.py --inflow modelled` replaces the upstream gauges' observations with the
trained model's predictions there:

| Dam | Linear, observed inflow | Linear, modelled inflow | + cap, observed | + cap, modelled | Trained ddrs at the dam |
|---|---|---|---|---|---|
| Raystown | 0.799 / 0.727 | 0.764 / 0.683 | 0.877 / 0.856 | 0.844 / 0.703, `Q_max` 365 [310, 365] | 0.692 / 0.622 |
| Abiquiu | 0.743 / 0.673 | 0.237 / 0.300 | 0.744 / 0.774 | 0.237 / 0.276 | 0.074 / 0.180 |

- A linear-reservoir node at Raystown adds about 0.06 test NSE over the trained model; `T` is
  identified to the same band as with observed inflow.
- The cap's test gain shrinks from +0.13 to +0.02. It still fits train, but once flood inflows are
  wrong in magnitude a fixed ceiling no longer generalises, and its band runs into the grid.
- At Abiquiu nothing helps: upstream regulation dominates the inflow error.

A parametric reservoir node is only as good as the inflow reaching it. The robust gains are the
ones that do not depend on it: prescribing observed release, or keeping regulated gauges out of
the parameters' way.

## 3. Population scale

`experiments/reservoir/regulation_sizing.py`: degree of regulation (DOR, Lehner et al. 2011,
upstream reservoir storage over mean annual flow volume) for the 2,365 eval gauges, from DDR's
reservoir table (2,178 COMIDs, the NWM / RFC-DA set, 2,177 with a GRanD id) joined to HydroLAKES
`Vol_total`. The table misses dams outside the NWM set (Alamo has no mapped reservoir), so
regulated counts are lower bounds. Trained run as §2.6, WY1997-2010.

| Class | Gauges | NSE trained / base | KGE trained / base | NSE if volume were perfect | r | median abs(1 − β) | α | NSE < 0 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| no mapped reservoir | 1,456 | 0.739 / 0.708 | 0.760 / 0.718 | 0.759 | 0.880 | 0.095 | 0.894 | 3 % |
| DOR ≤ 0.1 | 218 | 0.814 / 0.735 | 0.842 / 0.795 | 0.823 | 0.913 | 0.050 | 0.963 | 0 % |
| 0.1 < DOR ≤ 0.5 | 344 | 0.765 / 0.677 | 0.799 / 0.745 | 0.779 | 0.895 | 0.073 | 0.926 | 2 % |
| **DOR > 0.5** | **347** | **0.495 / 0.418** | **0.579 / 0.518** | 0.553 | 0.769 | 0.148 | 0.836 | 9 % |

Median trained NSE by drainage area:

| Area, km² | none | DOR ≤ 0.1 | 0.1 to 0.5 | DOR > 0.5 |
|---|---|---|---|---|
| < 316 | 0.718 (426) | 0.798 (4) | 0.696 (34) | **0.338** (48) |
| 316 to 1k | 0.754 (563) | 0.781 (29) | 0.727 (72) | **0.489** (91) |
| 1k to 3.2k | 0.756 (355) | 0.811 (78) | 0.768 (96) | **0.533** (104) |
| 3.2k to 10k | 0.757 (98) | 0.827 (74) | 0.798 (89) | **0.529** (66) |
| > 10k | 0.644 (14) | 0.851 (33) | 0.802 (53) | **0.523** (38) |

- **The cost is concentrated above DOR 0.5**, 347 gauges (14.7 % of the eval set), and it holds in
  every area bin: 0.12 to 0.38 NSE below unregulated gauges of the same size. Below DOR 0.5 there
  is no deficit; the higher medians there are large rivers.
- **It is mostly timing, not volume.** A perfect volume correction lifts the DOR > 0.5 median from
  0.495 to 0.553; correlation is 0.77 against 0.88 unregulated. Routing is mass-conserving, so
  volume was never going to be fixed by a reservoir node anyway (the Santa Rosa loss, the Abiquiu
  imports); timing is what a reservoir representation can address.
- **Removing them moves the population median from 0.732 to 0.751.** Every trained number in
  `research-status.md` includes these gauges.
- These gauges are also training targets. Their loss asks the upstream channel parameters for
  attenuation and timing that roughness cannot give. That this distorts the learned `n` is an
  inference, not a ddrs measurement; the literature reports exactly that compensation (§4.3), and
  option A in §5 is the test.
- **Where a boundary condition could go** (`release_gauge_coverage.py`, over the 8,945 gauges in
  the gauges adjacency store): 216 of the 347 (62 %) have another gauge between them and their
  largest upstream reservoir, median trained NSE 0.557; the other 131 are the first gauge below it,
  median 0.401. Observation coverage at the intermediate gauges is not checked, so 216 is an upper
  bound. The largest reservoir holds a median 100 % of a gauge's upstream storage: one dam per
  gauge, not cascades, is the typical case.

## 4. Literature

A search agent covered large-scale reservoir schemes, generic operating rules, differentiable and
hybrid reservoir modules, boundary conditions and data assimilation, and identifiability. The 22
DOIs this section leans on were checked against Crossref (author, year, venue, title); equations
were checked in full text for WaterGAP 2.2d, Vanderkelen 2022 and Casado-Rodríguez 2026 only.
Other numbers are as reported by the papers.

### 4.1 The two laws already tried are lake laws

- **The dMC fill-fraction law is WaterGAP's natural-lake outflow**, `Q = k S (S / S_max)^a`
  (Döll et al. 2003, 10.1016/S0022-1694(02)00283-4; Müller Schmied et al. 2021, GMD,
  10.5194/gmd-14-1037-2021, Eq. 27), with `k = 0.01 d⁻¹`, `a = 1.5` and `S_max` = lake area × 2 m,
  an active layer, not dam capacity. WaterGAP switched reservoirs to Hanasaki's scheme in 2.1g.
  dMC applied the lake law to reservoirs with total capacity as `S_max`, which is the 67-day
  floor of the Raystown doc.
- **NWM / WRF-Hydro level pool, which DDR ported, is passive by design**: weir plus orifice, no
  management. Level pool mis-simulates operated reservoirs and the error propagates downstream
  (Kim et al. 2020, HSJ, 10.1080/02626667.2020.1757677); NWM's skill at managed reservoirs
  comes from persisting observed USGS / USACE outflow and ingesting RFC forecasts, not from the
  level-pool physics (Cosgrove et al. 2024, JAWRA, 10.1111/1752-1688.13184).

### 4.2 Caps and plateaus are standard; zero-then-rectangle is a schedule

- **A flood-control release plateau is what the current global schemes do.** CaMa-Flood
  (Hanazaki et al. 2022, JAMES, 10.1029/2021MS002944) holds release near a flood discharge `Q_f`
  while inflow exceeds it, stores the rest, and passes inflow only above an emergency level; it
  improved NSE at 62 % of gauges over 2,169 dams and replaces the LISFLOOD routine in GloFAS v5.
  LISFLOOD's three-zone scheme (Zajac et al. 2017, J. Hydrol., 10.1016/j.jhydrol.2017.03.022)
  has a normal-release plateau and a floor. §2.5's `min(S / T, Q_max)` is the simplest member of
  this family.
- **On 164 ResOpsUS reservoirs** (Casado-Rodríguez et al. 2026, HESS,
  10.5194/hess-30-4629-2026), default outflow KGE' is 0.44 LISFLOOD, 0.52 CaMa-Flood, 0.60 mHM;
  STARFIT did no better than CaMa-Flood at the daily scale.
- **Zero release, then a fixed rate** is HYPE's regulated lake (production flow cut to zero below
  a level), ParFlow's threshold rule (West et al. 2025, HESS, 10.5194/hess-29-245-2025), and
  GDROM's constant / piecewise-constant modules (Li et al. 2024, WRR, 10.1029/2023WR036686: 1 to 8
  modules of five types describe all 452 reservoirs). The shape has names; the *timing* is not a
  function of inflow: about 80 % of 300+ US dams release on medium- to long-range forecasts
  (Turner, Xu & Voisin 2020, HESS, 10.5194/hess-24-1275-2020). That is Alamo and Santa Rosa.
- **STARFIT / ISTARF-CONUS** (Turner et al. 2021, J. Hydrol., 10.1016/j.jhydrol.2021.126843):
  harmonic normal-operating range with release clamps, fitted offline to storage *and* release
  records at 595 reservoirs and extrapolated to 1,930 CONUS GRanD dams.

### 4.3 What downstream discharge can and cannot identify

- **Outflow-only calibration does not identify storage dynamics** (Casado-Rodríguez 2026:
  calibrating to outflow degrades storage, calibrating to storage keeps outflow skill;
  Hosseini-Moghari & Döll 2025, HESS, 10.5194/hess-29-4073-2025: storage anomalies make storage
  skilful at 64 to 68 of 100 reservoirs against 16 with defaults, outflow calibration does not).
- **Missing or wrong reservoir physics is absorbed by other parameters.** VIC calibrated without
  reservoirs matches the daily skill of VIC with them by distorting runoff and baseflow parameters
  (Dang, Chowdhury & Galelli 2020, HESS, 10.5194/hess-24-397-2020). A fully differentiable
  HBV + reservoir model trained on outflow fit outflow but biased recession and baseflow
  parameters; a loosely coupled version did better ungauged (Chen et al. 2025, WRR,
  10.1029/2025WR041684). In LISFLOOD, reservoir-parameter sensitivity fades downstream in favour of
  Manning's n (Zajac 2017).
- **A timescale can be learned where the dam is near pass-through.** Two-parameter water-supply
  rules calibrated at downstream gauges improved skill across Great Britain, with transfer
  functions from attributes working in about half the catchments (Salwey et al. 2024, HESS,
  10.5194/hess-28-4203-2024). This matches Raystown's identified `T`.
- **Inflow bias caps any rule's gain.** Empirical rules help with observed inflow and not with
  biased model inflow (Turner, Doering & Voisin 2020, WRR, 10.1029/2020WR027902); in global
  mizuRoute runs, runoff bias swamped the reservoir scheme (Vanderkelen et al. 2022, GMD,
  10.5194/gmd-15-4163-2022). That is §2.7.

### 4.4 Regulation thresholds and the population pattern

DOR ≥ 0.02 is the conventional "affected" cut (Lehner et al. 2011, 10.1890/100125); DOR ≥ 0.08
with depth-of-disruption ≥ 0.06 m marks "disruptive" dams (Shrestha et al. 2024, WRR,
10.1029/2023WR035433). For LSTMs over 3,557 CONUS basins, median NSE was 0.72 / 0.79 / 0.64 for
zero / small (≤ about a month of flow) / large DOR (Ouyang et al. 2021, J. Hydrol.,
10.1016/j.jhydrol.2021.126455). ddrs reproduces the shape (0.739 / 0.814 / 0.495, §3) with the
deficit starting later, at DOR 0.5 rather than about 0.08.

### 4.5 Data that exist for CONUS

ResOpsUS (Steyaert et al. 2022, Sci. Data, 10.1038/s41597-022-01134-7): daily inflow, outflow and
storage for 679 reservoirs. ISTARF-CONUS: STARFIT parameters for 1,930 GRanD dams. GDROM v2
(Zheng et al. 2025, Sci. Data, 10.1038/s41597-025-06162-7): reconstructed daily series for 2,017
reservoirs. Caravan-format ResOpsUS used by Casado-Rodríguez 2026: 10.5281/zenodo.15978041.
Satellite storage and area: GRSAD, GloLakes, Global Water Watch. When this section was written
none of these was on disk. **Update, same day:** ResOpsUS v2 (Zenodo 10.5281/zenodo.6612040),
ISTARF-CONUS (10.5281/zenodo.4602277) and the ResOpsUS+CARS attribute tables are now at
`/mnt/ssd1/data/resops/`, fetched by `~/projects/remote_sensing_extraction/resops/fetch_resops.sh`,
with a GRanD to MERIT COMID crosswalk (2,177 rows, 663 in ResOpsUS) and a coverage inventory under
`derived/`: 298 reservoirs have at least 5 years of overlapping daily inflow and outflow, 289 of
them mapped to a COMID. GDROM v2 and the satellite products are still not on disk.

## 5. Options

### 5.1 The menu

| | Option | Parameters | Data | Where it helps | Evidence here | Cost |
|---|---|---|---|---|---|---|
| **A** | Stratify, then mask DOR > 0.5 gauges from the training loss | 0 | DOR per gauge (§3) | Protects `n` elsewhere from the dams' shadow; honest reporting | 347 gauges at median NSE 0.495 inside every trained number | A gauge list; one training run for the A/B |
| **B** | Observed release as a boundary condition at a gauge below the dam | 0 | USGS gauge below the dam; ResOpsUS / GDROM v2 outflow where no gauge | Every gauge downstream of a gauged release: up to 216 of the 347 | Exact by construction; Raystown loses 0.22 NSE across a dam whose release is gauged | Identity-row mechanism, an obs reader for non-target gauges |
| **C** | Linear reservoir at the dam row: `k := T`, `x_storage := 0` | 1 per dam | Dam → COMID table | Near-pass-through flood-control dams | +0.06 test NSE over the trained model at Raystown with modelled inflow (§2.7); `T` identified to a factor of 3 | Coefficient override only; no new state |
| **D** | Capped linear reservoir, `R = min((S + dt I) / (T + dt), Q_max)`, CaMa-Flood-like | 2 per dam | `Q_max` from data (a release-record quantile, or USACE / USBR channel capacity) | Flood-control dams with a known safe release | +0.13 test NSE with observed inflow, +0.02 with modelled inflow (§2.5, §2.7) | Identity row + carried storage state |
| **E** | Offline-fitted operating rule (ISTARF / STARFIT, or GDROM modules), frozen in the router | 12 to 19 per dam, prescribed | ResOpsUS / ISTARF / GDROM v2 downloads | Scheduled irrigation and water-supply dams, only where a record exists to fit | Arid dams: no storage-state rule works (§2.5); Casado-Rodríguez 2026: STARFIT ≈ CaMa-Flood at daily scale | Same mechanism as D + a data pipeline |
| **F** | Level pool (DDR #137 to #139) for **natural lakes only** (`Lake_type` 1 / 3) | 0 learned, 5 from HydroLAKES | HydroLAKES | Lake-controlled outlets | None measured; the literature uses weir laws for lakes | Port DDR's code |

**Not recommended**, with reasons in hand: the fill-fraction law (a lake law, refuted at all four
dams); level pool for operated reservoirs (passive by design, Kim 2020); learning any operating
rule end to end from downstream discharge (§4.3: not identified, and the error lands in other
parameters).

### 5.2 Recommendation

1. **A first, as the experiment that decides how much the rest matters.** Retrain with the DOR > 0.5
   gauges masked from the loss (still evaluated), compare learned `n` on reaches that drain to no
   DOR > 0.5 gauge (contamination would travel through the shared head, not only the local
   reaches), and skill on unregulated gauges. If masking moves `n` and lifts unregulated
   skill, the dams have been contaminating the parameters everywhere, and the curvature paper has a
   second exhibit next to leakage: parameters that should not be learned through a dam. If it moves
   nothing, reservoirs are a local skill problem and B to D are priced by the 347 gauges alone.
2. **B and D share one mechanism; build it once, B first.** An identity row whose right-hand side
   is the release (DDR's pattern, §1). B is exact, has zero parameters, and does not depend on
   inflow quality, which §2.7 shows limits every parametric node. Where a release gauge exists,
   nothing parametric beats it.
3. **C where there is no gauge and the dam is near pass-through**; it is a coefficient override,
   so it is cheap to try, and `T` is the one reservoir parameter that looked identifiable. Treat D's
   cap as prescribed data, not a learned parameter: it overfits under modelled inflow.
4. **E only if the scheduled dams earn it.** Their share of the 347 is not yet counted; their
   release timing is set by demand and forecasts, so a rule needs storage and release records to
   fit, offline.

### 5.3 How B, C and D enter the solve

```
 One timestep t -> t+1 (dt = 3600 s), one lower-triangular solve  A q_{t+1} = b   (src/routing/mmc.rs)

 ordinary reach i (MC)                          dam row d (B, D, E)
 A[i,:] = e_i - c1_i N[i,:]                     A[d,:] = e_d             c1_d := 0: pattern fixed, value zeroed
 b_i    = c2_i (N q_t)_i + c3_i q_t,i           b_d    = R_d(S_t, I_t)   the release, computed before the solve
          + c4_i q'_i
                                                   B:  R_d = Q_obs(t+1)            (modelled value where Q_obs is NaN)
 option C needs no identity row:                   D:  R_d = min((S_t + dt I_t) / (T_d + dt), Q_max,d)
   k_d := T_d, x_d := 0 on the MC row              E:  R_d = f(S_t / S_cap, day of year; fitted offline)
   (Muskingum X = 0 is the linear reservoir, §2.1)
                                                 after the solve (D, E only):
         forward substitution carries              I_{t+1} = (N q_{t+1})_d + q'_d
         q_{t+1,d} = R_d to every row below        S_{t+1} = S_t + dt (I_{t+1} - R_d)

 upstream of d ─▶ [reach] ─▶ [reach] ─▶ ║dam row d║ ─▶ [reach] ─▶ (gauge)
                                         ║R_d     ║
                                         ╚════════╝
```

Using `I_t` from the previous solve is a one-hour lag at `dt = 3600 s`, where the daily sandbox's
equivalent choice cost 0.08 NSE (§2.2) because the lag there was a day.

### 5.4 Blast radius (B, D; C is a subset, A is config only)

| Area | Change | Risk |
|---|---|---|
| `src/routing/mmc.rs` | `route_timestep`: zero `c1` on dam rows, override `b`, update storage after the solve | Touches the hot path; must be a no-op when the feature is off |
| new `src/routing/reservoir.rs` | Release functions, storage state, dam → row mapping | New |
| `src/sparse/` | none: values change, the CSR pattern and `CsrSolveOp` backward do not | Invariant 4 kept |
| `src/config.rs` | `params.reservoirs` block, `data_sources.reservoirs` table | Validation: B needs observations for the boundary gauges |
| `src/data/` | Dam table reader; observation series for boundary gauges on the routing axis (today only target gauges are read) | The dataset's batch window must carry the boundary series |
| `src/training/` | Boundary gauges removed from the targets | Otherwise a gauge is trained on its own input |
| `src/cuda_graph/` | Reject the combination at config load, or capture the override | CPU is the reference backend anyway |
| Eval, manifest | Record the boundary-condition mode; its metrics are conditioned on observed releases | Not comparable with any unconditioned number |
| Tests | Gradcheck through an identity row and the storage recursion; mass balance; `ddr_sandbox_match` unchanged | Invariant 1: off by default, DDR has no reservoirs on master |

### 5.5 Concerns

- **B conditions predictions on observations.** Downstream metrics then measure routing below a
  known release, not prediction in ungauged conditions. It needs its own eval mode and label, and a
  boundary gauge must never be a training target.
- **Daily observations in an hourly solve.** An observed release enters as a 24-hour step, the
  way daily Q' does (repeat-24), so B fixes daily volume and timing, not sub-daily shape.
- **Gauge ≠ dam.** A "gauge below the dam" can be kilometres downstream with local inflow between;
  the boundary goes at the gauge's COMID, and the reaches between the dam and the gauge are then
  routed as usual.
- **The reservoir table is the NWM set.** It misses Alamo; any population claim is a lower bound
  until GRanD or Global Dam Watch is mapped to MERIT.
- **D's cap is fragile under modelled inflow** (§2.7), and `T` absorbs upstream timing error. A
  learned `T` at a dam is a compensating parameter unless the curvature instrument says otherwise.
- **f32 storage.** Keep `S` as the active buffer above the pool, a few MCM, not total volume:
  f32 resolves 1e9 m³ only to 128 m³, against an hourly inflow increment of order 1e5 m³.
- **Min/max gradients.** The cap and any pool clamp have zero gradient on the flat side (the
  literature's warning for STARFIT clamps, §4.3). Fine for a prescribed `Q_max`; a learned one
  needs a smooth clamp.

### 5.6 Assumptions

- The trained run `2026-09-17T16-38-16Z-train-and-test` stands for the current model. The DOR
  pattern is structural (routing cannot hold water for weeks), so it should hold for the other
  runs, but it is measured on one.
- HydroLAKES `Vol_total` (from GRanD for GRanD-linked lakes) is total volume; DOR on active storage
  would be smaller. The 0.5 break is therefore on total-volume DOR.
- Daily NSE on the eval window is the metric. At Alamo it is dominated by a few evacuations and
  should not be read.
- Four dams are the whole sandbox. Raystown and Abiquiu are flood control; the two arid dams are
  scheduled. The DOR > 0.5 population has not been split by purpose.

**Status (same day):** option C is implemented, off by default: `params.use_reservoirs` plus a
`data_sources.reservoirs` CSV (`COMID,T_days`), commits `368786a`, `c10da38`, `9d5bab8`; see
`.claude/RESERVOIRS.md`. It has been exercised only on the Juniata bundle with Raystown at
`T = 1.23 d`. The next step for it is a `T` table fitted per dam from ResOpsUS inflow and outflow
(§4.5), which is also the data option E needs.

### 5.7 Next measurements, in order

1. The A/B of option A: one training run (the reference run took about 13 h wall-clock).
2. Split the 347 by GRanD purpose and by whether the intermediate gauge has observations over the
   eval window; that turns the 216 upper bound into a count and prices option E.
3. At the 216, an offline estimate of B's gain: replace modelled flow at the intermediate gauge with
   observed flow and re-route the reaches below it.
