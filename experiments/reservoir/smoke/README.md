# Dam-release smoke-test set

A gauge set with known expected results, built 2026-09-26 **before** the learned dam release exists in ddrs, so the
implementation (branch `dam-release-head`) can be checked against it. Results page (maps, drainage areas,
hydrographs, every gauge): https://claude.ai/artifact/UbLNfpeMRS4k44vGXM6uV2. Earlier versions: 50 gauges at commit
`5a47623`, 628 gauges (1.5x area rule) at `f5dfd27`.

## The set (`select_smoke_gauges.py`)

916 gauges from the 2,365 training / eval gauges, every HUC2 region:

- **458 dam gauges**: at least one NID dam >= 10 MCM upstream (`../nid/nid_dams_in_eval_network.csv`); the nearest of
  them (upstream area closest to the gauge's) has gauge area **<= 3x** its reach's (user decision, 2026-09-26), was
  completed by 1980, and was snapped by drainage-area match (class A, B, D); >= 80 % observed days in WY1982-1995 and
  WY1996-2010; gauge area <= 25,000 km2. 214 of those dams are on the gauge's own reach; 223 gauges sit below a chain
  of large dams. HUC 08 had none and got 08015500 (area ratio up to 10, `relaxed`).
- **458 controls**, one per dam gauge: no NID dam upstream, no NWIS peak code 6, same coverage, same HUC2, closest
  drainage area without replacement. 54 came from outside the region: the unused gauge minimising
  |log area ratio| + distance / 1,000 km.

Files: `gages_smoke.csv` (gages_3000 format, for `data_sources.gages`), `smoke_gauges.csv` (the set with the nearest
dam's attributes, `control_for`, `cascade`, `relaxed`), `smoke_dams.csv` (the 723 NID dams >= 10 MCM in the network,
with NID features), `smoke_summary.json`. Network: 29,395 reaches.

## Expected results (no release code involved)

`run_smoke_eval.sh`: the no-dam head `2026-09-12T23-39-03Z` (sr_n0_gamma) @ `epoch_50_mb_9` routed over
1981-10-01..2010-09-30 in one pass on the set, CPU, 1,880 s -> `output/reservoir_smoke/pred_1981_2010.zarr`.
`expected_release_fit.py`: on that routed flow, a storage release `S = T Q`, plain and with seasonal
`T_t = T0 exp(a sin w + b cos w)`, fitted per gauge on WY1983-1995 (WY1982 spin-up), scored WY1996-2010 ->
`expected_release_fit.{csv,json}`, `output/reservoir_smoke/web/` (the page's data).

| Test WY1996-2010 | no dam | plain bucket | seasonal bucket |
|---|---:|---:|---:|
| median NSE, 458 dam gauges | 0.619 | 0.674 | 0.670 |
| median NSE, 458 controls | 0.767 | 0.771 | 0.769 |
| median KGE, dam gauges | 0.685 | 0.692 | 0.692 |
| median KGE, controls | 0.774 | 0.772 | 0.772 |

- Per-gauge change, seasonal: dam gauges +0.0068 [+0.0041, +0.0131], 291 up / 167 down (sign p = 7e-9); controls
  +0.0001, 233 / 225 (p = 0.74). Plain bucket: dam gauges +0.0055 [+0.0029, +0.0095], 307 / 151 (p = 3e-13).
- Dam gauge minus its matched control: +0.0085 [+0.0050, +0.0125], dam ahead in 282 of 458 pairs (p = 8e-7).
- Dam on the gauge's own reach +0.020 (214); further up +0.004 (244). The 1.5x-3x gauges see a diluted release,
  which is why the per-gauge median is smaller than on the 1.5x set (+0.017).
- Chain of dams +0.010 (223), single dam +0.006 (235). Dam median above control median in 13 of 18 regions.
- By degree of regulation (NID storage / annual flow): +0.005 below 0.1, +0.007 at 0.1-0.5, +0.011 at 0.5-1, +0.024
  at 1-2, 0.000 above 2.
- Fitted T0 (seasonal): dams median 0.8 d (IQR 0.05-3.7), 123 of 458 at the pass-through floor; controls at the floor
  in 267 of 458.

## Known issues

1. The drainage-area match is loose: median 1,400 km2 at dam gauges against 641 km2 at controls, since few large
   undammed basins exist. On the 1.5x set, well-matched pairs gave the same answer as the rest.
2. Rio Hondo below Diamond A Dam (08390800): no-dam NSE -17.5, change +6.3, dry most days. Summaries are medians.
3. The grid: 3 dams hit the 1,000-day T0 wall (Courtright -1.62 -> -3.47, Sumner, Lake Almanor); 142 of the 335 dam
   gauges with an active bucket have a or b on the +/-2 edge.

## What the implementation must show on this set

1. **Off = identical.** With `use_reservoirs: false`, output is bitwise identical to `pred_1981_2010.zarr` on this
   gauge CSV. With every smoke dam at fixed T = 1/24 d, a = b = 0, daily flows match the no-dam run with NSE > 0.999
   at every gauge. A one-hour bucket at an hourly step is nearly, not exactly, pass-through.
2. **Engine matches the fit.** KAN head frozen, each gauge's nearest dam set to `expected_release_fit.csv`
   (`seas_T0`, `seas_a`, `seas_b`): ddrs matches the tuned hydrographs at the 214 on-reach dams, NSE between the two
   > 0.99.
3. **Gradients.** T0, a, b pass a finite-difference check.
4. **Learning.** Training on 1981-1995 from near pass-through moves T0 up where this fit found storage and leaves it
   low where it found none.
5. **Beat the controls.** Median per-gauge change in test NSE at dam gauges above zero with its 95 % interval clear of
   zero, dam gain minus matched-control gain positive, controls' median test NSE within 0.01 of the no-dam arm's. The
   offline fit is roughly the ceiling for a per-dam bucket on this inflow.

Cap T0 well below 1,000 days. Compare only against results on this gauge CSV (trap T19).

## Rebuilding the page

`page/`: `page_src.html` (template), `prose.json` (the brief texts and the four hydrograph picks), `map.json` (HUC2
outlines in CONUS Albers, from `build_map.py`, source USGS 1:2M HUC2 via CAMELS `huc_02.zip`), `build_page.py` (fills
the template from `output/reservoir_smoke/web/index.json`, writes `output/reservoir_smoke/web/publish/`),
`to_b64.py` (artifacts do not serve `.bin`: series are published as `series/hucNN.b64.txt`). Publish
`publish/dam_release_smoke.html` with `index.json` and `publish/series/*.b64.txt` alongside.
