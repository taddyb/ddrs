# Arid-region dam sandboxes: Alamo, Santa Rosa, Abiquiu

**Date:** 2026-09-25
**Scripts:** `experiments/reservoir/find_arid_dams.py` (candidate search),
`experiments/reservoir/dam_sandbox.py` (generic above/below sandbox), results in
`experiments/reservoir/results/{alamo,santa_rosa,abiquiu}.json`, figures under
`output/dam_sandbox/<tag>/sandbox.png`, candidate table
`experiments/reservoir/arid_dam_candidates.csv`.
**Follows:** `2026-09-22-raystown-storage-law-sandbox.md`, where the dMC fill-fraction law
failed at a humid flood-control dam because its response time is bounded by capacity.

## 1. Candidates

67 dam-named gauges in the training population have aridity >= 2.5. Twenty of them have an
upstream gauge on the same river so the inflow is observed rather than modelled. Three were
chosen for gauged-inflow share and purpose:

| Dam | Release gauge | Inflow gauges | Gauged share of summed Q' | Mean inflow / release, m³/s | Summed-Q' volume ratio |
|---|---|---|---:|---|---:|
| Alamo Dam, Bill Williams River AZ (flood control, aridity 7.5) | 09426000 | 09424450 Big Sandy + 09424900 Santa Maria | 0.96 | 3.39 / 2.48 | 1.28 |
| Santa Rosa Dam, Pecos River NM (flood control, aridity 3.8) | 08382830 | 08382650 above the lake | 1.00 | 2.80 / 2.38 | 1.70 |
| Abiquiu Dam, Rio Chama NM (flood control + storage, aridity 3.9) | 08287000 | 08286500 above the reservoir | 0.96 | 12.76 / 13.00 | 1.17 |

Full observation coverage 1995-2010 at all six gauges. Assumed capacities (from memory, and
bracketed by a 3,000× sweep): Alamo 1,290 MCM, Santa Rosa 550 MCM, Abiquiu 1,480 MCM.

## 2. Result

Train WY1997-2001, test WY2002-2010, daily NSE. "Storage law" is dMC's
`_first_order_euler_storage` with `b = storage_n · alpha` and `theta` fitted on a 28 × 28
grid; "best over capacity" is the best cell over `S0 × {0.001, 0.01, 0.1, 0.3, 1, 3}`.

| Dam | Pass-through | Linear, T fitted | Storage law at physical S0 | Storage law, best over capacity |
|---|---|---|---|---|
| Alamo | −155 / −2.05 | −4.1 / 0.01 (T 3,000 d) | −0.39 / −0.04 (corner) | 0.04 / −0.01 |
| Santa Rosa | −0.16 / −0.39 | 0.00 / 0.00 (T 1,240 d) | −0.01 / 0.00 (corner) | 0.02 / −0.00 |
| Abiquiu | 0.65 / 0.51 | 0.73 / 0.66 (T 5.5 d) | 0.11 / 0.42 (corner) | 0.73 / 0.67 (S0 × 0.001) |

Values are train / test.

**Alamo and Santa Rosa: nothing explains the release.** Every model, at every capacity, sits
at NSE ≈ 0 or below, meaning no storage law does better than predicting the mean release.
The hydrographs say why. At Alamo the release is zero for months, then a flat-topped
evacuation at about 180 m³/s that starts weeks after the flood and holds until the flood pool
is empty. At Santa Rosa the release is zero most of the year and then a rectangular block
(39 m³/s for two weeks in September 2004) with no inflow behind it: an irrigation delivery
from storage. These are operating rules with thresholds and fixed rates. A power law of fill
fraction is smooth and monotone in storage and cannot produce a rectangle. The landscape at
Alamo is a monotone slope to the box corner; at Santa Rosa a third of the grid is within
0.02 NSE of the best cell, which is a flat plateau at zero skill.

**Abiquiu: Raystown again.** Pass-through already scores 0.65 train, the fitted linear
reservoir with a 5.5-day timescale 0.73 / 0.66, and the storage law only matches that when
capacity is shrunk a thousand-fold to 1.5 MCM. At the physical capacity it is at the corner
with 0.11 train. The dam is operated near pass-through with a few days of buffer, and the
mass-conserving law can do nothing about the 17 % inflow-volume excess the summed Q' carries.

## 3. Reading

The arid flood-control dams are the case the user asked for, and they are the clearest
refutation yet of learning reservoir behaviour from a storage law: the release is a policy,
decoupled from inflow at the daily scale, and its shape (zero, then a rectangle) is outside
the model class regardless of parameters. The equifinality symptoms at Raystown (corner
optimum, indefinite curvature) were misspecification of scale; at Alamo and Santa Rosa it is
misspecification of form, and no capacity rescues it.

This also settles what a downstream gauge can teach in these basins. Below Alamo and Santa
Rosa the hydrograph carries almost no information about inflow timing, so any routing or
leakance parameter above the dam is unconstrained by that gauge, and the "volume-deficit"
regulated gauges of the regional census should be dropped from the training population
rather than modelled.

## 4. What would work

1. **Prescribe observed release** at dam rows as a boundary condition where it exists
   (these six gauges all have it). Zero parameters, exact by construction, and it removes the
   dam's shadow from every gauge downstream.
2. **A rule with thresholds** where release is not observed: release = minimum release when
   storage is below the flood pool, evacuation at a fixed rate when above, seasonal delivery
   blocks from storage targets. That is the STARFIT form fitted to storage records
   (ResOpsUS), and it is not learnable from discharge alone because the storage record is what
   sets the thresholds.
3. **Do not port the fill-fraction law** in any capacity parameterisation. Three dams, three
   purposes, one humid and two arid, and it never beat a one-parameter linear reservoir.
