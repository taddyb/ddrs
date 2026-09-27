# Dam-release smoke-test set

A small, fast gauge set with known expected results, built 2026-09-26 **before** the learned dam release exists in
ddrs, so the implementation can be checked against it in minutes. Explainer with hydrograph viewer:
https://claude.ai/artifact/KSP5GLgFfW44SmZUFyZgeZ (§6).

## The set

50 gauges from the 2,365 training / eval gauges, every HUC2 region covered (`select_smoke_gauges.py`):

- **32 dam gauges**: exactly one NID dam >= 10 MCM upstream (`../nid/nid_dams_in_eval_network.csv`), completed by
  1980, snapped by drainage-area match (class A, B, D); gauge area <= 1.5x the dam reach's; >= 80 % observed days in
  WY1982-1995 and WY1996-2010; gauge area <= 25,000 km2. Two per region (most regulated, then most regulated with a
  different purpose). HUC 08 and 09 had no such gauge and got one each with area ratio up to 10 (`relaxed`:
  08015500 Bundick Creek Dam, 05078000 Clearwater). 24 of the 32 dams are on the gauge's own reach.
- **18 controls**, one per region: no NID dam upstream, no NWIS peak code 6, same coverage, closest drainage area.

Files: `gages_smoke.csv` (gages_3000 format, for `data_sources.gages`), `smoke_gauges.csv` (the set with dam
attributes), `smoke_dams.csv` (the 32 dams >= 10 MCM in the smoke network, with NID features), `smoke_summary.json`.
Network: 1,431 reaches.

## Expected results (no release code involved)

`run_smoke_eval.sh`: the no-dam head `2026-09-12T23-39-03Z` (sr_n0_gamma) @ `epoch_50_mb_9` routed over
1981-10-01..2010-09-30 in one pass on the smoke gauges, CPU, 380 s -> `output/reservoir_smoke/pred_1981_2010.zarr`.
`expected_release_fit.py`: on that routed flow, a storage release law `S = T Q` (plain, and with seasonal
`T_t = T0 exp(a sin w + b cos w)`) fitted per gauge on WY1983-1995 (WY1982 spin-up), scored WY1996-2010 ->
`expected_release_fit.{csv,json}` and `output/reservoir_smoke/smoke_series.json` (the viewer's data).

| | no dam | plain bucket | seasonal bucket |
|---|---:|---:|---:|
| median test NSE, 32 dam gauges | 0.295 | 0.300 | 0.306 |
| median test NSE, 18 controls | 0.710 | 0.729 | 0.729 |
| dam gauges up / down vs no dam | | 21 / 11 | 20 / 12 |
| controls up / down | | 10 / 8 | 13 / 5 |

Fitted T0: median 1.1 d (plain) / 2.0 d (seasonal) at dams, 0.09 / 0.16 d (pass-through) at controls. KGE flat.
Seasonal-bucket wins: Shelbyville +0.38, Big Stone Lake +0.22, Alum Creek +0.21, Barre Falls +0.16, Buford +0.15,
Alamo +0.12.
Failures: Courtright (seasonal T0 hits the 1,000 d grid wall, test NSE -1.62 -> -3.47), Downsville -0.15, Eagle Nest
-0.10, Chimney Dam -0.07.

## What the implementation must show on this set

1. **Pass-through start.** Every smoke dam at T0 = 1 hour reproduces `pred_1981_2010.zarr` at all 50 gauges to
   1e-4 m3/s.
2. **Engine matches the fit.** KAN head frozen, each dam's T0, a, b set to `expected_release_fit.csv` (`seas_*`):
   ddrs matches the tuned hydrographs at the 24 on-reach dams, NSE between the two > 0.99 (the fixed-T engine
   managed this at 27 dams, `research/findings/2026-09-26-option-c-dam-benchmark-findings.md`).
3. **Gradients.** T0, a, b pass a finite-difference check.
4. **Learning.** Training on 1981-1995 from pass-through moves T0 off 1 h where the fit found storage (Shelbyville,
   Buford, Big Stone, Barre Falls) and leaves it near 1 h where it found none (Lake Anna, Nantahala, Clearwater).
5. **No harm.** Median test NSE at the controls within 0.01 of 0.710; at the dam gauges not below 0.295.

Cap T0 well below 1,000 days (Courtright). Part of the gain is generic smoothing (controls +0.005), so the joint run
has to beat the controls, not zero. Evaluate with this gauge CSV only when comparing to these numbers (trap T19).
