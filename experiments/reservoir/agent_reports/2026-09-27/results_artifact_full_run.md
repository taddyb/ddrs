<!-- agent a6bdb75019c4d6001 | Eval plots and results artifact | handback | 2026-09-27T13:04:45.775Z -->
**URL:** https://claude.ai/artifact/HhcubFbHpBdmpKykN7PeUT. I published version 1 with seed 42, then version 2 at the same URL once both seed-43 runs finished (manifest status ok, predictions.zarr written). The page is private until the user shares it.

## Plots produced (ddrs-eval-plots, executed with the ddrs-py venv)
- **Parameter dumps:** I ran `dump_parameters --backend cpu` on the dam-release branch binary for the final checkpoint and for epochs 1, 10, 25 and 50 of both seed-42 arms, plus the final checkpoint of both seed-43 runs. It accepted the learned config as-is, so no throwaway config was needed. Output is in each run's `plot/`.
- **Notebooks:** each of the two seed-42 run dirs (`2026-09-27T07-29-47Z-train-and-test`, `2026-09-27T07-29-55Z-train-and-test`) has three executed notebooks in `plots/`:
  - `metrics.ipynb`: six-metric box plot, NSE and KGE CDFs, drainage-area box plots and gauge map, all against summed Q′. It also draws two maps of the per-gauge change in NSE: dam release minus no dam, and each arm minus Q′.
  - `hydrographs.ipynb`: four gauges, each in its typical and its wettest test water year.
  - `parameter_maps.ipynb`: n₀ and γ maps (declared range and a shared stretched scale), histograms, parameter against drainage area, convergence, and in the dam run the change maps with the dams marked.
- **Stage roughness:** `n_of_d_wy2000_{traces,maps}.png` in both run dirs.
- **Colours:** the diverging teal-grey-ochre steps and an ochre sequential ramp both pass the dataviz validator in light and dark.

## What is on the page
Each figure has the hypothesis, plain-words, what-is-drawn, result and conclusion brief.
- **Header:** tiles, five numbered claims linked to their evidence, the 15-year test split stated plainly, and a score table (summed Q′, no dam, dam release).
- **Paired comparisons:**
  - Forest plots with bootstrap intervals and up/down counts: each arm against Q′, and dam release against no dam. The groups are all, dammed, undammed, dam on the gauge reach, dam further up, and the five regulation bands.
  - The distribution of per-gauge changes on a symmetric log axis.
  - An interactive CONUS map (HUC2 basemap, dammed shown as triangles, hover, three comparisons to choose from).
  - A region-by-region chart, and a scatter of the change against degree of regulation.
- **Release head:** an interactive map of T₀ and seasonal swing at all 1,024 dams, T₀ against residence time and against the offline fitted T₀, and the month T is longest.
- **Channel parameters:** n₀ and γ maps for both arms, the change between arms, and convergence.
- **Hydrographs:** four interactive panels, and a viewer covering all 2,365 gauges and all 15 test years. Series ship as gzip+base64 text, about 42 MB.
- **Also:** a gauge table, a limits section, the replicate section, and a gallery of all 41 eval PNGs.

Sources are in `experiments/reservoir/full_run/page/` (uncommitted):
- `analysis.py`
- `replicate.py`
- `build_page.py`
- `prepare_images.py`
- `page_src.html`
- `picks.json`

Built outputs are in `output/reservoir_full_run/web/` (gitignored).

## Seed replicate (seed 43)
| | Seed 42 | Seed 43 | Two no-dam seeds differ by |
|---|---|---|---|
| Dammed gauges, change in NSE | +0.0014 | +0.0039 [+0.0027, +0.0055] | −0.0005 |
| Dam on the gauge reach | +0.0027 | +0.0049 | −0.0003 |
| Undammed gauges | −0.0007 | −0.0004 | −0.0008 |
| Population median NSE, no dam → dam release | 0.7391 → 0.7379 | 0.7341 → 0.7396 | |

- **The gain below dams reproduces.**
  - Every regulation band from 0.1 to 2 has its interval clear of zero in both seeds.
  - It holds for either cross-seed pairing: +0.0031 and +0.0020.
  - Which gauges gain repeats between seeds: rank correlation 0.63 (0.74 on the gauge reach, 0.82 at regulation 1 to 2).
  - The typical per-gauge change at dammed gauges is about 3 times the no-dam seed-to-seed spread (median |change| 0.009–0.010 against 0.0030).
- **The undammed loss is at the noise floor.**
  - Its sign agrees in the two seeds, but it is as large as the median difference between the two no-dam runs.
  - The per-gauge pattern does not repeat (ρ −0.15), and the cross-seed pairings disagree (+0.0002 and −0.0012).
  - Verdict: two seeds cannot separate it from chance.
- **Population median NSE** moves by less than the spread between seeds, so there is no CONUS-wide gain or loss.
- **KGE** falls at dammed gauges in both seeds (−0.0009 and −0.0007), consistent with the bucket smoothing peaks.

## Things that looked wrong or need attention
1. **γ is not identified.** Median γ is 0.067 without dams and 0.168 with them in seed 42, but 0.272 and 0.043 in seed 43. The order between arms reverses with the seed, and γ was still moving at epoch 50 in every run (15–17 % of its range over the last 25 epochs).
   - My seed-42 text blamed the undammed loss on the γ difference. After the replicate I revised that on the page.
   - Any γ or stage-roughness claim from single runs, including the existing research notes, is unreliable.
2. **The release head's seasonal phase flips between seeds.** In seed 42 T is longest from August to February; in seed 43 from March to September. Per-dam T₀ ranks reproduce (ρ 0.82 between seeds); the phase does not (ρ 0.34). The page now says a and b should not be read as operating rules.
3. **Learned T₀ values are hours to days.** The median is 8.5 h, against a median residence time of about 149 days at dams on a gauge reach. The ordering is physical (ρ 0.57 with residence time, 0.66 with storage), but these are flood-attenuation times, not residence times.
4. **The largest loss, Feather River at Oroville, is a spill case.** Observed January 1997 peak 3,568 m³/s is more than the unrouted inflow (3,146). The release, with T₀ of 16 days, cuts the peak to 783, and NSE within that year falls from 0.70 to 0.35. A storage-dependent release term is the likely next step.
5. **The dam table ignores completion dates.** By the NID year of each reach's largest dam, 75 dam reaches were completed after 1981 and 19 after 1995, and 27 have no year, yet every bucket is active in every year.
6. **Minor:**
   - The summed-Q′ median differs by metric code: 0.6785 recomputed, 0.6771 in `full_pairing.json`, 0.6781 quoted in the skill. The page uses 0.6785 throughout, scored with the same code as the arms.
   - The dam feature table used by the runs sits only in the `agent-a92e512a7c47c97b4` worktree (`experiments/reservoir/release_head/dam_features.csv`). The configs reference it by a relative path.

