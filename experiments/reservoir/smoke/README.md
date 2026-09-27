# Dam-release smoke-test set

A gauge set with known expected results, built 2026-09-26 **before** the learned dam release exists in ddrs, so the
implementation can be checked against it. Results page (maps, drainage areas, hydrographs, all 628 gauges):
https://claude.ai/artifact/UbLNfpeMRS4k44vGXM6uV2. The first 50-gauge version (two dam gauges per region) is commit
`5a47623`.

## The set (`select_smoke_gauges.py`)

628 gauges from the 2,365 training / eval gauges, every HUC2 region:

- **314 dam gauges**: at least one NID dam >= 10 MCM upstream (`../nid/nid_dams_in_eval_network.csv`); the nearest of
  them (upstream area closest to the gauge's) has gauge area <= 1.5x its reach's, was completed by 1980, and was
  snapped by drainage-area match (class A, B, D); >= 80 % observed days in WY1982-1995 and WY1996-2010; gauge area
  <= 25,000 km2. 214 of those dams are on the gauge's own reach; 148 gauges sit below a chain of large dams. HUC 08
  (08015500) and 09 (05046000) had none and got one each with area ratio up to 10 (`relaxed`).
- **314 controls**, one per dam gauge: no NID dam upstream, no NWIS peak code 6, same coverage, same HUC2, closest
  drainage area without replacement; 17 regions ran out and took the closest area from **any** region.

Files: `gages_smoke.csv` (gages_3000 format, for `data_sources.gages`), `smoke_gauges.csv` (the set with the nearest
dam's attributes, `control_for`, `cascade`, `relaxed`), `smoke_dams.csv` (the 571 NID dams >= 10 MCM in the network,
with NID features), `smoke_summary.json`. Network: 23,024 reaches.

## Expected results (no release code involved)

`run_smoke_eval.sh`: the no-dam head `2026-09-12T23-39-03Z` (sr_n0_gamma) @ `epoch_50_mb_9` routed over
1981-10-01..2010-09-30 in one pass on the set, CPU, 1,468 s -> `output/reservoir_smoke/pred_1981_2010.zarr`.
`expected_release_fit.py`: on that routed flow, a storage release `S = T Q`, plain and with seasonal
`T_t = T0 exp(a sin w + b cos w)`, fitted per gauge on WY1983-1995 (WY1982 spin-up), scored WY1996-2010 ->
`expected_release_fit.{csv,json}`, `output/reservoir_smoke/web/` (the page's data).

| Test WY1996-2010, median NSE | no dam | plain bucket | seasonal bucket |
|---|---:|---:|---:|
| 314 dam gauges | 0.526 | 0.585 | 0.601 |
| 314 controls | 0.761 | 0.766 | 0.765 |

- Per-gauge change, seasonal: dam gauges +0.017 [+0.007, +0.026], 210 up / 104 down (sign p = 2e-9); controls 0.000,
  155 / 159 (p = 0.87), T0 at the pass-through floor at 185 of 314 controls.
- Dam gauge minus its matched control: +0.014 [+0.007, +0.028], dam ahead in 202 of 314 pairs (p = 4e-7).
- Plain bucket: +0.009 at dam gauges (218 / 96), 0.000 at controls.
- KGE falls slightly in both groups (-0.002): the fit maximises NSE.
- Gain by degree of regulation (NID storage / annual flow): +0.012 below 0.1, rising to +0.030 at 1 to 2, +0.004
  above 2. On the gauge's reach +0.020, further up +0.011; below a chain of dams +0.024, a single dam +0.014.
- By region the dam median beats the control median in 15 of 18. Largest: Rio Grande +0.16, Upper Colorado +0.10,
  Texas-Gulf +0.08. California: 0.000 over 53 gauges (23 up, 30 down; 24 of them below hydropower dams).

## Known issues

1. The drainage-area match is loose for large basins: controls median 724 km2 against 1,225 km2 for dam gauges;
   47 % of pairs within 1.5x; above 5,000 km2 the median control is 0.15x its dam gauge. Well-matched pairs give the
   same answer (+0.012 [+0.003, +0.030]).
2. The 17 cross-region controls were taken from anywhere (California dam gauges got Florida, Illinois and Appalachian
   controls), not from neighbouring regions.
3. Rio Hondo below Diamond A Dam (08390800): no-dam NSE -17.5, change +6.3, dry 85 % of days. Summaries are medians.
4. The grid: 3 dams hit the 1,000-day T0 wall (Courtright -1.62 -> -3.47, Sumner, Lake Almanor); 102 of 246 dam
   gauges with an active bucket have a or b on the +/-2 edge.

## What the implementation must show on this set

1. **Pass-through start.** Every dam at T0 = 1 hour reproduces `pred_1981_2010.zarr` at all 628 gauges to 1e-4 m3/s.
2. **Engine matches the fit.** KAN head frozen, each gauge's nearest dam set to `expected_release_fit.csv` (`seas_T0`,
   `seas_a`, `seas_b`): ddrs matches the tuned hydrographs at the 214 on-reach dams, NSE between the two > 0.99.
3. **Gradients.** T0, a, b pass a finite-difference check.
4. **Learning.** Training on 1981-1995 from pass-through moves T0 off 1 hour where this fit found storage and leaves it
   near 1 hour where it found none.
5. **Beat the controls.** Median per-gauge change in test NSE at dam gauges above zero with its 95 % interval clear of
   zero, dam gain minus matched-control gain positive, controls' median test NSE within 0.01 of 0.761. The offline fit
   (+0.017, +0.014 over controls) is roughly the ceiling for a per-dam bucket on this inflow.

Cap T0 well below 1,000 days. Compare only against results on this gauge CSV (trap T19).

## Rebuilding the page

`page/`: `page_src.html` (template), `prose.json` (the brief texts and the four hydrograph picks), `map.json` (HUC2
outlines in CONUS Albers, from `build_map.py`, source USGS 1:2M HUC2 via CAMELS `huc_02.zip`), `build_page.py` (fills
the template from `output/reservoir_smoke/web/index.json`, writes `output/reservoir_smoke/web/publish/`),
`to_b64.py` (artifacts do not serve `.bin`: series are published as `series/hucNN.b64.txt`). Publish
`publish/dam_release_smoke.html` with `index.json` and `publish/series/*.b64.txt` alongside.
