# Adjoint influence map — GAGES-II nested-reference population — findings

**Spec:** `research/specs/2026-09-03-ddrs-experiment-adjoint-design.md`
**Prior finding:** `research/findings/2026-09-04-adjoint-influence-poc-findings.md` (Juniata pair)
**Bundle:** `experiments/adjoint-conus/`
**Output:** `.ddrs/experiments/adjoint-conus/2026-09-05T16-28-57Z/` — figures, `README.md`,
`STATS.md` (tables T1–T10) and `HANDOFF.md` (paper-structured handoff) in `figures/`
**Binary:** `target/release/ddrs` at commit `91b006a` (working tree), cpu backend, 5 arm threads

**Verdict: INCONCLUSIVE pending a replicate seed; direction established.** Across 41
gauges (20 GAGES-II Ref downstream + 21 nested upstream) the five inflow-source arms
disagree on effective wave celerity by a median factor of 2.2 per gauge, with a
consistent ordering (dhbv2-lumped slowest at 26/38 gauges, dhbv2-dist fastest at
19–26/38) and a scale dependence (ratio 1.25 in the 520-reach basin, > 3.3 in
8–48-reach subgraphs). Upstream-gauge volume bias transfers downstream with
coefficient 1.00 in every arm, but downstream bias is mostly generated locally
(median inherited share 0.27–0.39). Most apparent "mass loss" in the volume
functional is truncation by the 90-day window in basins > 300 km; genuine
unexplained mass loss is 13–69 reaches per arm (~1–2 %), clustered in Northern
Plains basins.

> **Correction (2026-09-07, checks 3–4 on the UH arm,
> `research/findings/2026-09-07-adjoint-volume-functional-checks-3-4-findings.md`):** the "mass loss" /
> "unexplained mass loss" / "clamped negative solves" reading of low volume sensitivity below is
> superseded. Low `volume_sens` is (1) inflow intermittency — the gradient is exactly zero at source
> hours with inflow at the clamp floor, so the raw time-mean collapses to the wet-hour fraction — and
> (2) slow low-flow transport in far semi-arid reaches. Pulse traces deliver 97–98 % of injected water
> from "zero-kernel" reaches; no clamped reach lies on any losing path. Use `volume_sens_wet`.
> The transfer coefficient (perennial upstream gauge reaches) and all kernel/celerity results stand.

## 1. Pre-registered hypotheses (spec §0, scope design §2)

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H-celerity | Learned routing timing is inflow-source dependent (arms differ in effective celerity on the same network) | **SUPPORTED** (single seed) | per-gauge max/min ratio median 2.16 low / 2.32 high; ordering consistent at ≥ 2/3 of gauges |
| H-volume | Volume bias cannot be absorbed: transfer coefficient ≈ 1 | **SUPPORTED** | transfer median 1.00 every arm; IQR 0.74–1.01 (daily-lstm) to 0.92–1.00 |
| H-inherited | Downstream bias is mostly inherited from upstream gauges (Juniata LSTM arms: 0.9) | **REFUTED** as a generalization | median inherited share 0.27–0.39; pooled slope 0.02–0.19 |

## 2. Methods

Same study code as the PoC (`src/experiment/adjoint/`), plus: GAGES-II
nested-reference gauge selection (`gauges.rs`), one thread per arm, per-gauge
failures non-fatal, and a hard check that all arms share `data_sources.gages`. The
dhbv2-dist arm is `2026-08-09T03-05-54Z` (the 2026-08-17 run was trained on
`gages_3000.csv` and was refused). All arms `epoch_30_mb_1`. Windows: 90 d hourly,
warmup 5 d, tail 7 d, lag 30 d, tau 9; anchors 2 high + 2 low ≥ 60 d apart; volume +
residual on WY2000 seasonal windows. Residual functional = squared error. Gate:
+5 % inflow at 3 reaches, tol 5 %.

Wall time: daily-lstm 862 s, uh-retro 665 s, dhbv2-lumped 1001 s, dhbv2-dist 1004 s,
hourly-lstm 2226 s, concurrent; 41 gauges × 9 backwards each per arm.

## 3. Results (all numbers from `figures/STATS.md`)

**Gate (T1).** Passed in all 5 arms; max rel. err 0.02 % (hourly-lstm) to 0.17 % (uh-retro).

**Celerity (T3, T9).** Per-gauge medians at low flow: daily-lstm 0.18, hourly-lstm
0.29, uh-retro 0.28, dhbv2-lumped 0.16, dhbv2-dist 0.26 m/s (n = 38). Pooled
origin fits: 0.26 / 0.30 / 0.26 / 0.23 / 0.39 m/s. Rank counts (slowest, low flow):
dhbv2-lumped 26, daily-lstm 9; (fastest, low): dhbv2-dist 19, hourly-lstm 11,
uh-retro 6. Cross-arm ratio smallest in the largest basins (06452000: 1.25;
06360500, 06354000: 1.27) and largest in small ones (12413000: 3.71; 08165500: 3.55;
08165300: 3.30). Within-arm IQR across gauges spans a factor 2–3: basin identity is
the first-order control; the source is a consistent second-order bias.

**Volume sensitivity (T5, T10).** Per-gauge median 1.00 in every arm. Reaches with
sensitivity < 0.5: 1230 / 692 / 1002 / 681 / 413 per arm (daily-lstm … dhbv2-dist)
of ~3,100, in 14–20 gauges. **Mostly truncation:** those reaches lie at median
243–334 km with low-flow kernel lag 15–22 d (vs 113–141 km, 3–7 d for the rest);
Spearman(volume_sens, distance) −0.45 to −0.61 in every arm. Unexplained mass loss
(lag < 3 d and inflow ≥ 1e-3 m³/s): 69 / 60 / 31 / 21 / 13 reaches, in 14 / 8 / 9 / 7 / 3
gauges, concentrated in 06354000 (38 reach-arms), 06447000 (30), 06353000 (27),
06360500 (23), 06359500 (15). Inflow-floor reaches (< 1e-3 m³/s) account for 86 / 73 /
377 / 40 / 164 of the low-sensitivity reaches.

**Inheritance (T6).** Inherited share median 0.27 / 0.34 / 0.28 / 0.39 / 0.34; 1
gauge-arm > 1. Pooled inherited-vs-total slope 0.02–0.19, r 0.08–0.70. WY-mean
downstream bias median −0.35 / +0.21 / +0.30 / −1.41 / +1.06 m³/s; dhbv2-dist
positive at 14/20 gauges, dhbv2-lumped negative at 13/20.

**Residual attribution sign (T7).** One sign over > 90 % of reaches in 65–95 % of
gauges; pooled positive fraction 0.36 / 0.37 / 0.62 / 0.38 / 0.72.

## 4. Conclusions

1. The inflow product imprints a consistent, ordered bias on learned wave celerity
   across the population (lumped dHBV2 slowest, distributed fastest), reproducing
   the Juniata pair. Magnitude is ~2× at the typical gauge, smaller than the
   basin-to-basin spread within one arm.
2. Celerity agreement across arms improves with basin size: the gauge constrains
   travel time when travel time is a large fraction of the hydrograph timescale.
   This is the "which places can be trusted" signal the paper needs, on 5 extreme
   gauges so far.
3. Volume bias transfers with unit coefficient; the routing reshapes timing only.
   The downstream bias sign is set by the product. Most downstream bias is local.
4. The volume functional is truncated by the 90-day window in basins > ~300 km; the
   PoC's "mass loss" reading must be qualified. Real unexplained mass loss is rare
   and regional (Northern Plains); mechanism (clamped negative solves) unverified.

## 5. Next steps

1. Replicate seed per arm; report the seed-to-seed cross-arm ratio as the noise floor.
2. Fix the volume functional: mean over source hours whose kernel lag fits the
   remaining window (per-reach lag is already stored), or lengthen the window.
3. Negative-discharge tracking on WY2000 window 1 for 06354000 / 06447000 / 06353000.
4. Enlarge the population: relax the downstream Ref requirement (Ref upstream +
   reservoir mask) to test the scale dependence formally.
5. Investigate the 3 upstream gauges without a celerity fit (02017500, 08377900, 09492400).
6. Dropped: per-gauge kernel-by-lag panels at population scale (unreadable; the
   pooled lag-vs-distance figure replaces them).

## 6. Reproduce

```bash
cargo build --release --bin ddrs
target/release/ddrs --workspace .ddrs experiment adjoint-conus --backend cpu
~/projects/ddr/.venv/bin/python experiments/adjoint/plots.py .ddrs/experiments/adjoint-conus/<ts> --maps 6
```

## 7. Corrected volume statistics (2026-09-07 rerun, `.ddrs/experiments/adjoint-conus/2026-09-07T18-18-39Z/`)

Rerun of all five arms with `volume_sens_wet` (mean over source hours with inflow above the clamp
floor; see `research/findings/2026-09-07-adjoint-volume-functional-checks-3-4-findings.md`). 3,146 reaches per arm.

| arm | median raw | median wet-hour | reaches raw < 0.5 | reaches wet < 0.5 | gauges with any wet < 0.5 | reaches with wet-hour fraction < 0.5 |
|---|---|---|---|---|---|---|
| daily-lstm | 0.809 | 0.995 | 1230 | 311 | 9 | 1063 |
| hourly-lstm | 0.931 | 0.999 | 692 | 57 | 9 | 661 |
| uh-retro | 0.927 | 1.000 | 1002 | 130 | 6 | 879 |
| dhbv2-lumped | 0.901 | 0.975 | 681 | 263 | 7 | 455 |
| dhbv2-dist | 0.985 | 0.999 | 413 | 34 | 3 | 373 |

Mass is conserved in every arm (wet-hour median 0.975–1.000). The residue below 0.5 follows arm speed:
the two slowest arms (daily-lstm, dhbv2-lumped; §3.2) retain the most reaches whose water is still in
transit at the window end, the fastest (dhbv2-dist) the fewest. The inflow products differ strongly in
intermittency (373–1063 reaches dry more than half the time), which is what the raw statistic was
measuring. §3.1's "unexplained mass loss" numbers and the clamped-negative-solve hypothesis are withdrawn.
