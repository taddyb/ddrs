# Option C on the dam benchmark: linear reservoirs with ResOpsUS-fitted T

**Date:** 2026-09-26
**Scripts** (in `experiments/reservoir/benchmark/`, run in this order, Python under `~/projects/ddr/.venv`):
`fit_reservoir_T.py` (per-dam `T`, 2.2 s), `audit_benchmark_attributes.py` (reservoir attributes per
dam and per option), `prepare_option_c_eval.py` (gauge CSV + three eval configs),
`run_option_c_eval.sh` (three ddrs evals, CPU), `score_option_c.py` (paired scoring).
**Outputs:** `reservoir_T_fits.csv`, `reservoirs_T_{fit,prior}.csv`, `attribute_audit{.csv,_summary.json}` next to
the scripts; `experiments/reservoir/results/option_c_benchmark.json`; per gauge
`output/reservoir_benchmark/option_c_by_gauge.csv` (gitignored).
**Follows:** `2026-09-26-dam-benchmark.md` §6 item 1.
**Asked (user):** a sandbox test of the reservoir code on the benchmark dams: NSE, timing, and which reservoir
attributes are missing.

## Summary

1. **Fitted `T` does nothing at the median.** At the 44 dams with a ResOpsUS fit, paired ΔNSE is +0.015
   [−0.082, +0.066], 23 up and 21 down (sign test p = 0.88). The spread is wide, from +1.03 at Falls Lake to
   −0.57 at Oroville. The benchmark median over 121 gauges goes 0.442 → 0.422.
2. **It helps where the linear reservoir describes the dam and hurts where it does not.** `T` ≤ 60 d
   (32 dams): +0.040 [−0.007, +0.090], 20 up and 12 down. `T` > 60 d (12 dams): −0.145, 3 up and 9 down. The losers are
   western storage projects (Oroville, Englebright, Berryessa, Island Park) where the fitted reservoir scores
   ≤ 0.23 on the dam's own record even with observed inflow. Both splits are post hoc.
3. **KGE falls at nearly every dam, because the reservoir double-counts attenuation.** ΔKGE −0.081
   [−0.164, −0.047], 36 of 44 down (p = 2.5e-5). α drops from 0.885 to 0.607 while β does not move. The
   head was trained without reservoirs, so its channel parameters already supply part of the dam's
   attenuation. Option C cannot be judged fairly until a head is trained with it on.
4. **A generic `T` at dams without data is harmful.** Median fitted `T` (22.6 d) at the 77 dams with no fit:
   ΔNSE −0.143 [−0.179, −0.052], 24 up and 53 down (p = 0.001); ΔKGE −0.198. Do not route a dam as a
   reservoir without a dam-specific `T`.
5. **Timing: no measurable cost.** 461 s with reservoirs off, 461 s with 44 dams, 447 s with 121 dams (CPU,
   11,522 reaches, 131,496 hourly steps, about 3.5 ms per step). The override is two O(N) masked selects per
   timestep ahead of an unchanged sparse solve.
6. **Reservoir attributes: `T` is the one that is missing.** 44 of 121 dams have one. Another 33 could
   get one by reconstructing inflow from ResOpsUS outflow and storage. For the last 44 there is no source, and no
   published attribute predicts `T` (R² 0.06 on log residence time). HUC 01 and 04 have no dam-specific
   release data of any kind.
7. **The benchmark's gauge predictions depend on which other gauges are evaluated.** `FLOW_SCALE` rescales
   the lateral inflow of every eval gauge's outlet reach, and that inflow is routed downstream. A 121-gauge
   eval differs from the 2,365-gauge eval at 73 of 121 gauges (NSE by up to 1.58). Paired comparisons within one
   gauge set are valid. Absolute numbers from different gauge sets are not comparable. New trap T19.

## 1. Design

- **Host head:** `2026-09-12T23-39-03Z-train-and-test` (`sr_n0_gamma`: learned `n_0` and `gamma`, `p` = 21 and
  `q` = 0.65 fixed, no leakance), checkpoint `epoch_50_mb_9`, the checkpoint its own eval used. The benchmark's
  numbers come from `2026-09-17T16-38-16Z` (`sr_n0_gamma_leakance`), but `validate_reservoirs` rejects
  `use_reservoirs` with `use_leakance` (the leakance op has no K/X override), so the reservoir arms need a head
  trained without leakance, and `sr_n0_gamma` is that run's matched control. The two agree on the 121 gauges
  (median NSE 0.457 and 0.458 in their stored full-population evals).
- **`T` per dam:** Muskingum at `K = T`, `X = 0`, trapezoid rule, same-day inflow, which is what a ddrs dam row
  computes. Fitted on the daily ResOpsUS record of the dam's own inflow and release, restricted to days
  outside WY1997-2010 so the gauge score stays out of sample for `T` (all 44 fits had at least three years
  there). 200-point log grid on [1/24, 3000] d. Median `T` 22.6 d, IQR 7.1 to 69.8 d; two dams on the
  3000 d wall (Island Park and one unnamed Upper Colorado dam, scheduled releases). On the eval window, with
  observed inflow, the fitted reservoir scores median NSE 0.469 at the dam, against −0.121 for pass-through.
- **Arms**, identical but for the table: `off`; `fit` (44 dams); `prior` (all 121, median `T` where no fit).
  Legacy eval binary, `--backend cpu`, 15-day chunks with cross-chunk state (at `X = 0` the reservoir storage
  is `T·Q`, so it carries across chunks). The logs confirm 44 of 44 and 121 of 121 table COMIDs matched.
- **Scoring:** daily NSE and KGE on WY1997-2010, as in the benchmark. Paired difference against `off` per gauge,
  median with a 2,000-resample bootstrap interval (seed 42), sign test over the gauges that changed.
- **Sanity check:** the `off` arm reproduces the host run's stored CUDA eval to within 1e-4 m³/s at the
  47 gauges with no other scaled eval gauge upstream. Everywhere else the gauge set differs (§6).

## 2. NSE

| | off | fit | prior |
|---|---:|---:|---:|
| median NSE, 121 gauges | 0.442 [0.386, 0.499] | 0.422 [0.370, 0.491] | 0.334 [0.282, 0.373] |
| median KGE, 121 gauges | 0.563 | 0.535 | 0.388 |

| Paired, against `off` | n | median ΔNSE [95 % CI] | up / down | sign p | median ΔKGE |
|---|---:|---|---|---:|---:|
| fitted `T`, fitted dams | 44 | +0.015 [−0.082, +0.066] | 23 / 21 | 0.88 | −0.081 |
| median `T`, dams without a fit | 77 | −0.143 [−0.179, −0.052] | 24 / 53 | 0.001 | −0.198 |
| post hoc: fitted, `T` ≤ 60 d | 32 | +0.040 [−0.007, +0.090] | 20 / 12 | 0.22 | |
| post hoc: fitted, `T` > 60 d | 12 | −0.145 [−0.525, +0.080] | 3 / 9 | 0.15 | |
| post hoc: fitted, fit-window NSE ≥ 0.5 | 21 | +0.023 [−0.057, +0.074] | 12 / 9 | 0.66 | |
| post hoc: fitted, flood control | 16 | +0.015 [−0.073, +0.071] | 9 / 7 | 0.80 | |

The fitted arm changes one gauge besides the 44 (02129000, below a fitted dam, +0.001).

Fitted dams by region (medians):

| HUC2 | n | NSE off | NSE fit | ΔNSE | up | ΔKGE |
|---|---:|---:|---:|---:|---:|---:|
| 02 Mid-Atlantic | 2 | 0.382 | 0.542 | +0.160 | 1 | −0.012 |
| 03 South Atlantic-Gulf | 3 | 0.152 | 0.515 | +0.364 | 3 | +0.113 |
| 05 Ohio | 3 | 0.541 | 0.469 | +0.071 | 2 | −0.129 |
| 10 Missouri | 6 | 0.392 | 0.438 | +0.034 | 4 | −0.026 |
| 11 Arkansas-White-Red | 1 | 0.483 | 0.506 | +0.023 | 1 | −0.086 |
| 13 Rio Grande | 2 | −0.182 | −0.006 | +0.176 | 2 | −0.066 |
| 14 Upper Colorado | 10 | 0.126 | 0.184 | +0.087 | 7 | −0.139 |
| 16 Great Basin | 4 | 0.411 | 0.256 | −0.155 | 0 | −0.224 |
| 17 Pacific Northwest | 3 | 0.410 | 0.283 | −0.127 | 1 | −0.198 |
| 18 California | 10 | 0.513 | 0.280 | −0.160 | 2 | −0.140 |

Largest changes:

| Dam (gauge) | `T`, d | NSE off → fit | reservoir alone, observed inflow |
|---|---:|---|---:|
| Falls Lake (02087183) | 13.6 | −0.534 → 0.495 | 0.495 |
| Stockton Lake (06919020) | 52.4 | −0.011 → 0.448 | 0.461 |
| Flaming Gorge (09234500) | 397 | −0.271 → 0.139 | 0.327 |
| J. Strom Thurmond (02197000) | 10.3 | 0.152 → 0.515 | 0.598 |
| Beltzville (01449800) | 8.7 | 0.214 → 0.540 | 0.623 |
| Lake Oroville (11407000) | 65.7 | 0.423 → −0.150 | 0.229 |
| Englebright (11421000) | 283 | 0.785 → 0.235 | 0.121 |
| Island Park (13042500) | 3000 | 0.530 → −0.020 | −0.610 |
| Lake Berryessa (11454000) | 588 | 0.517 → 0.016 | 0.090 |

The last column is the fitted reservoir on the dam's own ResOpsUS record in WY1997-2010. Where it beats the
no-reservoir model at the gauge (26 of 44 dams), option C gains a median +0.072 (19 of 26 up); where it does not
(18), −0.120 (4 of 18 up). That split uses eval-window dam records, so it explains the result but cannot select
dams. A usable version would compare the fitted reservoir against the trained model on a window outside eval.

## 3. Why KGE falls

At the 44 fitted dams the KGE terms move r 0.719 → 0.670, α 0.885 → 0.607, β 1.096 → 1.104. At the 77 prior dams:
r 0.727 → 0.580, α 0.898 → 0.584. The loss is variability. The no-reservoir model is already below α = 1
at these gauges (the options doc measured 0.836 over all DOR > 0.5 gauges). A reservoir fitted to the dam's
inflow and release adds the dam's attenuation on top of whatever the trained channel parameters already
supply. That compensation is the one the literature describes (options doc §4.3): error from a missing reservoir
lands in the upstream `n`. With observed inflow the same `T` does not over-attenuate (median 0.469 at the
dam), so the over-attenuation comes from the routed inflow, not from `T`.

## 4. Timing

| Arm | Reservoir rows | Wall, s | per hourly step |
|---|---:|---:|---:|
| off | 0 | 461 | 3.5 ms |
| fit | 44 | 461 | 3.5 ms |
| prior | 121 | 447 | 3.4 ms |

CPU (NdArray), 11,522 reaches in the union of the 121 subgraphs, 5,479 days (131,496 hourly steps) as 366
fifteen-day chunks. Wall time is measured from each arm's log creation to the next (file birth times; the
runner's own timer used `bc`, which is not installed, and has been changed to `awk`). One run per arm on a
shared workstation, with short Python jobs running alongside, so the 3 % spread is within noise and not
attributable to the reservoir rows. The `T` fit for all 121 dams takes 2.2 s.

## 5. Reservoir attributes: what the 121 dams have

| Attribute | Needed by | Dams | Source |
|---|---|---:|---|
| Capacity | E, level pool | 121 | ISTARF `GRanD_CAP_MCM`; HydroLAKES `Vol_total` |
| Full-volume residence time | (not usable as `T`) | 121 | HydroLAKES `Res_time` |
| Level-pool geometry (weir, orifice) | DDR #137 to #139 | 121 | NWM / RFC-DA `merit_reservoir_params.csv` |
| Main purpose, year | stratification | 82 | GRanD via ResOpsUS-CARS (the local GRanD copy holds only ResOpsUS dams) |
| **Linear-reservoir `T`** | **C, D** | **44** | fitted to ResOpsUS inflow and release |
| `T` reconstructable (outflow + storage ≥ 3 y, no inflow) | C, D | +33 | inflow = outflow + dS/dt, not done |
| `T`, no source | C, D | 44 | 39 not in ResOpsUS, 5 too short |
| `Q_max`, ISTARF `Release_max` (fitted rule) | D | 76 | `mean inflow × (1 + Release_max)` |
| `Q_max`, observed 99th-percentile release | D | 79 | ResOpsUS outflow |
| `Q_max`, GloFAS `Qf` | D | 22 | ResOpsUS-CARS `glofas.csv` |
| `T` and an observed `Q_max` together | D | 44 | limited by `T` |
| Operating rule fitted to observations | E | 76 | ISTARF `fit = full` (43 more are borrowed from another dam, 2 storage-only) |
| Observed release ≥ 3 y inside WY1997-2010 | B | 77 | ResOpsUS outflow |

By tier: tier 1 (43) has `T` at all 43; tier 2 (39) has `T` at 1 and reconstructable inflow at 33; tier 3 (39)
has nothing dam-specific but ISTARF's borrowed rules. By region, the dams with a fitted `T`: 01 0/10, 02 2/10,
03 3/10, 04 0/6, 05 3/10, 07 0/9, 09 0/1, 10 6/10, 11 1/4, 12 0/3, 13 2/6, 14 10/10, 15 0/2, 16 4/10,
17 3/10, 18 10/10. HUC 01 and 04 have no `T`, no observed `Q_max`, no fitted rule and no release record; HUC 07
has release records at 3 of 9.

**`T` is not recoverable from published attributes.** Over the 44 fits, log `T` correlates with HydroLAKES
residence time, GRanD DOR, and capacity over mean flow at Spearman +0.34 each, and with ISTARF `Release_max`
at −0.44. A log-log fit on residence time explains 6 % of the variance and leaves a factor-6 residual; residence
time runs a median 14 times the fitted `T`. By purpose the median `T` is 6.6 d for hydropower (5 dams), 14 d for
water supply (6), 22.6 d for flood control (16), and 52 d for irrigation (15). Purpose and `Release_max`
are the only weak priors, and §2 shows a weak prior does harm.

Not available on this workstation at all: a maximum non-damaging release or spillway capacity per dam (the
National Inventory of Dams is not on disk; USACE water-control manuals publish no dataset), and GRanD attributes
for the 39 dams outside ResOpsUS (GRanD v1.3 itself is not on disk).

The KAN head's own inputs are fine: every reach of the 121 subgraphs is in the attribute file with length and
slope; 147 of 12,759 reaches (1.2 %) have a NaN `NDVI` or `Porosity`, which ddrs fills with the network mean.

## 6. Gauge predictions depend on the eval gauge set

`src/data/collate.rs::build_flow_scale` sets, for every gauge in the batch, the `q'` scale of that gauge's
outlet reach to its `FLOW_SCALE` (DDR `readers.py:270-330`). The scaled inflow is routed, so it reaches every
gauge downstream. A gauge's prediction therefore changes with the set of other gauges evaluated with it.
Measured here: all 47 benchmark gauges with no non-benchmark eval gauge of `FLOW_SCALE` ≠ 1 upstream match the
2,365-gauge CUDA eval to 1e-4 m³/s (so CPU and CUDA agree); 73 of the 74 with one differ, by a per-gauge median
relative difference of up to 29 % (09510000) and, at single steps, up to 16 times the gauge's mean flow. NSE
moves by up to 1.58 (San Carlos Lake, −29.9 against −28.4). The benchmark table in `2026-09-26-dam-benchmark.md` is a full-population eval
and stays self-consistent; a subset eval must be compared only with the same subset.

## 7. Caveats

- One head, trained without reservoirs, so §3's over-attenuation is built in. This measures option C bolted
  onto an existing model, not option C.
- The `T` ≤ 60 d, fit-window NSE and purpose splits were chosen after seeing the per-gauge table.
- ResOpsUS release and the benchmark gauge are not the same series where the dam is not on the gauge reach
  (17 of the 44). Englebright's gauge (Yuba River near Marysville) sits below other inflow.
- Lake Sumner's reservoir-alone NSE in the eval window is −357: its release has almost no variance there.
- `T` is fitted daily and applied hourly. For `T` of a day or more the two agree; below half a day the daily
  trapezoid has `c3 < 0` and the hourly one does not.

## 8. Next

1. Train a head with option C on at the dams with `T` ≤ 60 d (or all 44), and score it paired against
   `sr_n0_gamma` on the full population. That tests whether the head stops supplying the dam's attenuation and
   KGE recovers. Measure `n` upstream of those dams before and after.
2. Reconstruct inflow from outflow and storage at the 33 tier-2 dams and refit `T`, taking the fitted set to 77.
3. Before any absolute comparison with the benchmark table, run on the full 2,365-gauge population, or apply
   `FLOW_SCALE` to the gauge output instead of the reach inflow (a DDR-parity question).
4. Option D at the 44: `Q_max` from ISTARF `Release_max` (fitted rules) or the observed 99th-percentile
   release.

## 9. Addendum: release tuned on routed inflow only (stage 1, offline)

The design the user asked for (2026-09-26): no dam data as input. The release law is driven by the trained
model's routed flow, its parameters are tuned against the gauge below on training years only, and test years
score it. `experiments/reservoir/benchmark/release_fit_routed.py`, run `2026-09-17T16-38-16Z` (the benchmark's
own model, full-population eval, so no T19 issue), daily implicit Euler
`Q = min((S + I)/(T + 1), Q_max)`, `T_t = T0 exp(a sin w + b cos w)`, grid search on WY1997-2001 NSE, scored
WY2002-2010. The law acts on the gauge's routed series, which equals a dam row inside ddrs at the 80 dams on the
gauge reach. Controls: 121 undammed gauges (no NWM reservoir, no peak code 6 in WY1996-2010, no major GAGES-II
dam), area-matched one to one.

| Law | Free numbers | Dams: median ΔNSE [95 % CI], up / down | Controls: median ΔNSE, up / down |
|---|---:|---|---|
| linear `T0` | 1 | +0.027 [+0.010, +0.045], 89 / 32 (p = 2e-7) | +0.000, 65 / 56 |
| + season `a, b` | 3 | +0.040 [+0.019, +0.066], 86 / 35 | −0.001, 55 / 66 |
| + cap `Q_max` | 2 | +0.023 [+0.004, +0.060], 80 / 41 | −0.004, 49 / 72 |
| + cap + season | 4 | +0.034 [+0.007, +0.078], 79 / 42 | −0.006, 42 / 79 |

- Dam-specific: the controls do not move, and the fitted `T0` is a median 2.1 d at dams against 0.08 d at controls
  (0.05 d is the grid floor, pass-through). Median test NSE at dams 0.420 → 0.507 (linear), 0.516 (seasonal).
- On the dam reach (80): +0.033 linear, +0.055 seasonal; further down (41): +0.015.
- The cap overfits with routed inflow (hurts controls), as §2.7 of the options doc found at Raystown.
- KGE is flat (median −0.001, dams and controls); the fit optimises NSE.
- The fitted `T0` (2.1 d) is far shorter than the ResOpsUS dam-record `T` (22.6 d): the routed inflow is
  already attenuated by the trained channels, so the tuned law supplies only the residual storage. This is
  the opposite failure to §3, where a record-based `T` added the full dam on top and over-attenuated.
- Four laws on the same test years; the one-parameter law clears its interval alone.
- Explainer page (function, inputs, parameters, tuning, interactive toy):
  https://claude.ai/artifact/KSP5GLgFfW44SmZUFyZgeZ
