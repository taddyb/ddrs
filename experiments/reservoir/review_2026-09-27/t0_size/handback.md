<!-- agent a8cb23abbf34756f5 | Fable: learned T0 vs dam size | handback | 2026-09-27T13:50:43.504Z -->
> The agent's final handback message, verbatim; the full report is `report.md` beside it. Saved 2026-09-27.

Report: /home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size/report.md (tables, scripts assemble.py / analyze.py, joined table t0_size_table.csv, results.json in the same directory).

**Hypothesis.** If the release head has learned a physically meaningful time scale, T0 should rise with dam size, and more with relative size (storage / mean flow, i.e. residence time) than with absolute storage; and it should rank the dams the way independent bucket fits do.

**Plain-English summary.** T0 tracks absolute storage strongly and reproducibly, especially NID max (crest) storage, and is raised for flood-control dams. Relative size (residence time, storage per unit drainage area) carries only a weak secondary signal, and drainage area alone carries none. A surrogate shows T0 is essentially a function of max storage plus the flood flag, so the storage correlation is what the head chose to use, not evidence of physics. Against independent estimates the head agrees at rank level with offline bucket fits on the same routed inflow (rho about 0.5) but not with ResOpsUS fits from real inflow and release (rho about 0), which run 7 to 8 times longer; T0 is roughly 1/300 of hydraulic residence time.

**Figures.**
- /home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size/fig1_t0_vs_size.png: log-log T0 vs normal storage, residence time, height, drainage area, storage per area (both seeds, rho with CI); T0 by purpose. Storage panel is a clean band; drainage panel is a cloud; flood control sits highest, navigation collapsed at the 0.1 d floor.
- /home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size/fig2_surrogate.png: boosted-tree surrogate from the 19 head inputs (CV R2 0.95 / 0.93): log10_storage_max dominates (importance 0.53 / 0.58), purpose_flood second (0.19 / 0.24), storage_per_area third (0.06 / 0.12); height last. Partial dependence on max storage is monotone, factor 6 to 7 across the range.
- /home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size/fig3_independent.png: learned T0 vs offline seasonal fit (202 on-reach dams), ResOpsUS T (41), residence time (1,024), with 1:1 lines.

**Result** (Spearman rho [95 % bootstrap CI], seed 42 / seed 43, n = 1024 unless noted):

| relation | seed 42 | seed 43 |
|---|---|---|
| normal storage | 0.67 [0.63, 0.70] | 0.60 [0.55, 0.64] |
| max storage (n=988) | 0.78 [0.75, 0.80] | 0.72 [0.68, 0.75] |
| residence time (gauge-flow proxy) | 0.36 [0.30, 0.42] | 0.42 [0.37, 0.48] |
| residence time (GRanD flow, n=757) | 0.25 [0.18, 0.32] | 0.33 [0.26, 0.40] |
| height | 0.55 [0.51, 0.59] | 0.40 [0.35, 0.45] |
| drainage area | 0.10 [0.04, 0.17] | 0.01 [-0.06, 0.08] |
| partial: residence time given storage | 0.22 | 0.30 |
| offline seasonal fit, on-reach (n=202) | 0.46 [0.35, 0.57], median ratio 0.56 | 0.50 [0.39, 0.59], ratio 0.54 |
| offline linear fit, on-reach (n=202) | 0.44 [0.32, 0.56], ratio 1.03 | 0.46 [0.34, 0.56], ratio 1.05 |
| ResOpsUS own-record fit (n=39) | 0.12 [-0.25, 0.45], ratio 0.13 | 0.03 [-0.31, 0.36], ratio 0.14 |
| HydroLAKES Res_time (n=109) | 0.56 [0.40, 0.69], ratio 0.006 | 0.57 [0.41, 0.70], ratio 0.006 |

Seed rank agreement 0.82 [0.80, 0.85]; seed 43 longer by median factor 1.33; all size correlations agree in order between seeds. Within flood-control dams the residence-time correlation vanishes (0.10 / 0.14) while storage holds (0.70 / 0.57). Mean flow proxy: nearest-gauge specific discharge (median 13 km) times dam area, cross-checked against GRanD (rho 0.93, median ratio 0.97).

**Conclusion.** Yes, T0 correlates with dam size, but with absolute storage and the flood flag, which is what the head was given and chose; it does not mean T0 is a residence time (it is 0.3 % of one) and the relative-size signal expected from a storage-flow argument is weak. Agreement with independent fits is rank-level only where they share the routed inflow, and absent for ResOpsUS, whose long T (Oroville 66 d vs learned 16, Navajo 161 vs 13) the head compresses: routed inflow is pre-attenuated, 90-day windows cannot express multi-month memory, and half the ResOps fits have NSE < 0.5. Caveats: two seeds; offline fits are independent of the head but not of the routing; proxies have isolated bad gauge picks (Flaming Gorge). Follow-ups: (1) retrain with rho windows of 365 d or a state carried across windows and see whether ResOps agreement improves for the long-T dams; (2) an ablation dropping log10_storage_max and purpose_flood from the head inputs to test whether gauge loss actually needs them or the head merely found them convenient.
