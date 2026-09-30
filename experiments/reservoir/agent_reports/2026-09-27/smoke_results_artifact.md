<!-- agent aac2767b32551014c | Build smoke-test results artifact | handback | 2026-09-27T03:32:35.631Z -->
The Dam Release Smoke Test page is rebuilt for the 916-gauge (3x) set and republished to the same URL as Version 2: **https://claude.ai/artifact/UbLNfpeMRS4k44vGXM6uV2**. The supporting files kept the same names (`index.json` and `series/huc01` to `huc18.b64.txt`), so there was nothing to remove with null. Before publishing I rendered it once and fixed what showed up; the series load and every figure draws. Nothing is committed.

## What changed on the page
- **Set description:** the 3x rule is written out, with the earlier two-thirds (1.5x) rule and the dilution it trades for. Only HUC 08 is relaxed now. The distance-aware borrowing of controls (|log area ratio| + distance / 1,000 km, 54 borrowed) is described.
- **Checks:** check 1 now reads "Off is identical": bitwise identical with `use_reservoirs: false`, and a fixed 1-hour bucket only nearly pass-through (NSE > 0.999). Check 4 starts from "near pass-through".
- **Figure 3 (distribution of the change in NSE):** now split by default into dam on the gauge's own reach (solid ochre, +0.020, n 214), dam further up (dashed ochre, +0.004, n 244) and controls. A picker switches back to all dam gauges together. The results and conclusion carry the diluted-release reading: +0.017 at the 312 gauges within 1.5x of the dam's area against +0.003 at the 145 gauges between 1.5x and 3x, which is why the per-gauge median fell from +0.017 to +0.007.
- **New hydrograph examples (WY2005, a typical year at all four gauges):**
  - Big Muddy River at Plumfield below Rend Dam, a gauge the 3x rule added: test NSE 0.003 to 0.872.
  - Cannonsville on the West Branch Delaware, typical: 0.540 to 0.546. The model misses the summer releases, and a bucket cannot add water.
  - Courtright, largest loss: −1.62 to −3.47.
  - Little Fishing Creek, NC, a control: no change.
  - The viewer still opens on Shelbyville in WY2008 (its numbers did not change).
- **Rest of the prose:** every number is recomputed from the new `index.json` and `smoke_summary.json`, including:
  - regions: dam median above control median in 13 of 18;
  - California −0.001 and the Great Basin −0.009, explained as Truckee and Weber operations;
  - East and West both +0.007;
  - regulation bands: +0.005, +0.007, +0.011, +0.024, then 0.000 above 2;
  - chain of dams +0.010 against single +0.006;
  - both KGE readings.
- **Map:** `map.json` was rebuilt because gauge and dam points are baked into it. All 916 gauges fall inside their own HUC2 polygon, and the 723 dams are shown.
- **Small fixes from the render:**
  - Figure 3's "below −0.3" label no longer crosses the curves.
  - Tied dots in Figure 5 (Great Basin, Great Lakes) are offset so the teal one shows.
  - Up/down counts drawn inside the charts are removed. The rounded values in `index.json` gave 231 instead of 233 for controls, and the text carries the exact counts.

## Anything that looked wrong
1. **The drainage-area match is looser than on the 1.5x set.**
   - Pairs within 1.5x fell from 47 % to 36 %, and within 2x from 57 % to 45 %.
   - Above 5,000 km², only 1 % of pairs are within 1.5x.
   - On the 1.5x set, well-matched and loose pairs gave the same answer. Now they differ: well-matched pairs give dam minus control +0.005 [+0.001, +0.016], loose pairs +0.011 [+0.006, +0.018].
   - The gap comes from the dam side (dam gains rise with area, Spearman 0.12; control gains do not, 0.02), so the loose match doesn't create the effect. But +0.005 is the conservative figure, and the README's "well-matched pairs gave the same answer" is no longer true on this set. The page says this.
2. **For the 244 dams further up, the offline fit is not what ddrs will do.** The fit puts the tributaries between dam and gauge through the bucket too, which a dam-level release will not. The caveats say this, and it supports keeping check 2 to on-reach dams.
3. **Some controls gain because of natural storage.** 7 of the 15 controls that gain more than 0.1 are South Atlantic-Gulf swamp and spring-fed rivers (Santa Fe, Myakka, Econfina, Pearl), where the bucket stands in for storage the no-dam model lacks.
4. **Group medians and per-gauge medians disagree.**
   - NSE: the plain bucket beats the seasonal one on the dam-gauge group median (0.674 against 0.670), though the seasonal one is higher per gauge (+0.007 against +0.006).
   - KGE: the dam-gauge group median rises (0.685 to 0.692) while the per-gauge change is −0.002.

## Repo files changed (uncommitted)
All under `experiments/reservoir/smoke/page/`:
- `page_src.html`: set text, checks 1 and 4, Figure 3 split, Figure 7 brief, label fixes.
- `prose.json`: all texts and the four hydrograph examples.
- `map.json`: regenerated.
- `build_page.py`: small fix. The preview folder's `series` link pointed at the `.bin` folder, but the page fetches `.b64.txt`, so the preview would have failed. It now links the published copies and also copies `index.json` into `publish/`, so the publish root is self-contained.

`output/reservoir_smoke/web/publish/` was regenerated (page, `index.json`, `series/*.b64.txt`, about 40 MB in total).
