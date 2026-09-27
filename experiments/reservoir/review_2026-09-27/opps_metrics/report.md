<!-- agent aaa05a448bc22c7bb | Fable: metrics, loss, evaluation design | handback | 2026-09-27T13:49:22.440Z -->
> Recovered verbatim from the agent's final handback message (it could not write its own report file). Saved 2026-09-27. Synthesis: `research/findings/2026-09-27-dam-release-review-synthesis.md`.

# Summary (metrics, loss, evaluation review of the learned dam release)

Note: the harness refused to write `report.md`; the full report is below this summary. Computed tables, per-gauge metrics and scripts are on disk: `/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_metrics/{analysis.py,analysis2.py,analysis3.py,per_gauge_metrics.csv,tables.md,tables.json,tables2.md,tables2.json}`.

## Metric findings (paired learned minus off, median over gauges; seed 42 / seed 43)

1. The law is a sub-weekly smoother. NSE of daily differences at the 917 dammed gauges: +0.014 / +0.027 (undammed -0.003 / -0.002; seed noise -0.006), ten times the NSE change (+0.0014 / +0.0039). Mann-Whitney dammed vs undammed p = 1e-37 vs 2e-23 for NSE. All of the gain is in the sub-seasonal anomaly (+0.0018 / +0.0051); the NSE of the monthly climatology does not move (+0.0000 / -0.0004, p 0.54). The seasonal deficit at DOR > 0.5 is untouched. Daily cross-correlation lag is 0 in every arm (timing at daily resolution is saturated).
2. KGE components: r +0.0005 / +0.0022, alpha -0.0085 / -0.0082 (749 of 917 gauges down), beta 0. The KGE drop is entirely alpha. Gupta decomposition: at the median dammed gauge the NSE gain is the r term (+0.0010 of +0.0014; +0.0036 of +0.0039); above DOR 1 the alpha-shrink term is the mechanism for 21-40 % of the gainers (DOR 1-2 seed 42: alpha term +0.0035 of +0.0047).
3. Every amplitude metric worsens at dammed gauges: FHV -5.2 to -8.8 % (DOR 1-2: -11 to -19 %), peak ratio 0.84 to 0.78, flashiness ratio 0.77 to 0.68, recession ratio 0.64 to 0.60, lag-1 autocorrelation 0.921 to 0.943 against 0.915 observed; the fraction of dammed predictions smoother than the observation goes 69 % to 80 %. FLV +34 to +36 % (+150 to +165 % at DOR > 2). Undammed: none of these moves beyond seed noise.
4. Bimodal effect: 32-38 % of dammed gauges gain > 0.01, 13-15 % lose > 0.01; Hodges-Lehmann +0.008 / +0.016 vs median +0.0014 / +0.0039; gains sit in the worst-fit tercile (off NSE 0.38: +0.010) and where the learned T0 > 3 d (+0.041 / +0.043 vs +0.001 below 0.3 d). Group and paired medians disagree because of this skew.
5. Seed consistency is the reproducibility statistic: Spearman of per-gauge changes across seeds 0.63 dammed (0.74 on-reach, 0.80 flashiness) vs -0.15 to -0.25 undammed. T0 rank 0.82 across seeds but the population median differs 1.5x (0.35 vs 0.53 d); sign of b agrees at 43 % (chance).
6. Outliers: Gila below Coolidge +9.35 / -0.18 and Diamond A +1.76 / -8.20 (NSE < -16 in all arms) set the manifest mean (0.589). The next 13 largest changes are seed-consistent wins of +0.4 to +0.97 at on-reach flood-control dams. Area is not the confound (DiD positive in all five area quintiles).
7. Protocol: the undammed paired change (-0.0007, p 2e-15) is the null, so the sign test vs zero in `paired_full_run.py` is wrong; DiD per seed +0.0021 / +0.0043 (sd 0.0016) would need ~10 seed pairs for SE 0.0005.

## Top 5 opportunities

1. Frozen-routing ablation arm. Train the release head from the off checkpoint with the routing head frozen: undammed gauges become bitwise identical, the confound and the seed requirement vanish (release init is deterministic). Go: undammed change 0, dammed NSE >= +0.0014 and nse_diff >= +0.014, interval clear of zero. No-go: below the joint arm (co-training does the work). Cost: one ~2.5 h run.
2. Co-primary metrics: nse_diff, r, alpha, FHV/peak ratio in `paired_full_run.py` and the page. Criterion for future runs: dammed nse_diff > 0 with alpha change >= -0.003 and FHV not worse. Cost: hours (`analysis.py` is the implementation).
3. `nse-batch-deriv` (exists in `loss.rs`) for the release phase with routing frozen: it scores what the release changes and penalises smoothing past observed flashiness. Go on the smoke set: dammed NSE >= nse-batch arm, alpha >= -0.003, FHV not worse, controls unchanged. Cost: config + ~1 h run.
4. Freeze a, b and lengthen the window (`state_cache` or rho 365) for the release phase. Go: seasonal-cycle NSE at DOR > 0.5 rises above noise (0.0012) and T0 vs residence-time rank rises above 0.57. Cost: one run.
5. Hygiene: Mann-Whitney DiD + seed-consistency Spearman; Hodges-Lehmann and fraction beyond +/-0.01; clip to -1 before means; drop/clip `mean_nse_finite` (`src/cli/run.rs`); f64 accumulators in `src/training/metrics.rs` (f32 error up to 3e-4 NSE at 3 gauges); log-form low-flow metric; pre-registered "dam expected to act" stratum. Cost: hours.

---

# Full report

Date 2026-09-27. Inputs: seed 42 `2026-09-27T07-29-47Z` off / `07-29-55Z` learned; seed 43 `10-31-30Z` / `10-31-50Z`; `eval/predictions.zarr` (2,365 gauges x 5,477 test days WY1996-2010); seed-42 summed-Q' baseline; `experiments/reservoir/nid/nid_dams_by_gauge.csv`; `experiments/reservoir/smoke/smoke_gauges.csv`; both `release_params.csv`. Paired numbers: learned minus off at the same gauge and seed, median with a 2,000-draw bootstrap interval; "seed noise" is off43 minus off42.

## 1. What the extra metrics show

### 1.1 Daily-difference NSE sees the law ten times more clearly than NSE

| paired median | seed 42 | seed 43 | seed noise | median abs noise |
|---|---:|---:|---:|---:|
| NSE, dammed (917) | +0.0014 [+0.0007, +0.0020] | +0.0039 [+0.0027, +0.0055] | -0.0005 | 0.0030 |
| NSE, undammed (1,448) | -0.0007 [-0.0009, -0.0004] | -0.0004 [-0.0005, -0.0001] | -0.0008 | 0.0036 |
| nse_diff, dammed | +0.0143 [+0.0094, +0.0184] | +0.0267 [+0.0181, +0.0339] | -0.0057 | 0.0211 |
| nse_diff, undammed | -0.0032 | -0.0017 | -0.0050 | 0.0138 |
| nse_diff, dam on reach (377) | +0.0223 | +0.0386 | -0.0028 | 0.0152 |
| nse_diff, DOR 0.5-1 / 1-2 | +0.034 / +0.036 | +0.053 / +0.047 | -0.004 / -0.005 | |
| anomaly NSE (monthly climatology removed), dammed | +0.0018 | +0.0051 | -0.0007 | |
| seasonal-cycle NSE (12 monthly means), dammed | +0.0000 | -0.0004 | +0.0002 | 0.0012 |
| KGE, dammed | -0.0009 | -0.0007 | -0.0001 | |

Best cross-correlation lag (-5..5 d) is 0 at the median of every group and arm; r gain at the best lag 0.000. Annual-peak lag does not change and is not a usable per-gauge statistic (median mean |lag| 25 d: the model picks a different event). Event-based timing needs observed-event matching; not built.

### 1.2 KGE components and the Gupta decomposition

Dammed, signed paired medians: r +0.0005 / +0.0022 (level 0.872); alpha -0.0085 / -0.0082 (level 0.924; 749 / 679 of 917 down); beta +0.0002 / +0.0022; KGE -0.0009 / -0.0007.

`dNSE = 2 alpha_o dr + (2 r_l - alpha_l - alpha_o) dalpha - d(bias^2)`, medians (seed 42 / 43):

| group | dNSE | r term | alpha term | gainers (> 0.005) with alpha dominant |
|---|---:|---:|---:|---:|
| dammed | +0.0014 / +0.0039 | +0.0010 / +0.0036 | +0.0002 / +0.0002 | 16 % / 12 % |
| dam on reach | +0.0027 / +0.0049 | +0.0022 / +0.0041 | +0.0002 / +0.0002 | 13 % / 9 % |
| DOR 1-2 | +0.0047 / +0.0046 | +0.0023 / +0.0027 | +0.0035 / +0.0006 | 26 % / 21 % |
| DOR > 2 | +0.0009 / +0.0011 | -0.0003 / +0.0005 | +0.0020 / +0.0003 | 40 % / 36 % |

### 1.3 Amplitude metrics (group medians off -> learned, seed 42; paired change 42 / 43; undammed paired 42 / 43)

| metric | dammed | DOR 1-2 | paired | undammed |
|---|---|---|---:|---:|
| FHV % | -5.2 -> -8.8 | -11.1 -> -18.7 | -0.91 / -0.91 | -0.03 / +0.22 |
| peak ratio (max pred within 3 d of obs annual peak / obs peak) | 0.836 -> 0.782 | 0.671 -> 0.577 | -0.013 / -0.016 | -0.000 / +0.001 |
| flashiness ratio (Richards-Baker sim/obs) | 0.767 -> 0.681 | 0.499 -> 0.419 | -0.037 / -0.043 | -0.000 / +0.003 |
| recession ratio (median log decline sim/obs, observed falling days above the median) | 0.639 -> 0.600 | 0.337 -> 0.276 | -0.013 / -0.013 | -0.001 / +0.003 |
| lag-1 autocorrelation of prediction (obs 0.915; DOR 1-2 obs 0.954) | 0.921 -> 0.943 | 0.964 -> 0.991 | +0.005 / +0.008 | -0.000 / +0.001 |
| FLV % | +33.7 -> +35.7 | +86 -> +92 | +0.30 / +0.45 | -0.01 / -0.02 |
| fraction with prediction smoother than obs | 69 % -> 80 % | 77 % -> 87 % | | 85 % -> 85 % |

The 106 / 124 dammed gauges that crossed from under- to over-smoothed gained the most (median +0.037 / +0.040). Q95 log ratio (+0.42 dammed) is unchanged; FLV's form (percent bias of the lowest 30 % of the sorted FDC) is dominated by near-zero observed flows at regulated gauges.

### 1.4 Heterogeneity

Dammed NSE change: gain > 0.01 32 % / 38 %, loss > 0.01 15 % / 13 %, gain > 0.05 13 % / 15 %; median +0.0014 / +0.0039, Hodges-Lehmann +0.0084 / +0.0159, mean +0.033 / +0.014, mean without the top gauge +0.022 / +0.023. By tercile of off NSE (0.38 / 0.72 / 0.85): paired +0.0097 / +0.0007 / -0.0000 (seed 42), group-median shift +0.049 / +0.005 / +0.004. By learned T0 of the nearest dam (458 smoke dam gauges): < 0.3 d +0.0014 / +0.0006; 0.3-1 d +0.0028 / +0.0020; 1-3 d +0.013 / +0.020; > 3 d +0.041 / +0.043.

### 1.5 Seed consistency, parameters, outliers, area

Spearman of per-gauge changes across seeds: dammed NSE 0.63, nse_diff 0.67, KGE 0.65, alpha 0.69, flashiness 0.80, FHV 0.71; on-reach 0.74 to 0.85; undammed -0.03 to -0.25. Release parameters (1,024 dams): T0 rank 0.82, medians 0.35 vs 0.53 d, median |log ratio| 0.38; amplitude rank 0.46; sign agreement a 74 %, b 43 %.

Outliers: 09469500 Gila below Coolidge (off NSE -21.2 / -16.6) +9.35 / -0.18; 08390800 Rio Hondo below Diamond A (-17.6 / -16.2) +1.76 / -8.20. Next: seed-consistent +0.4 to +0.97 at Maumelle (07263300), Neuse near Falls (02087183), Pomme de Terre (06921350), Salt River near Center / New London (05507800, 05508000), Big Muddy at Plumfield, Cowlitz below Mayfield, Etowah at Allatoona, Sac River below Stockton, East Fork Little Chariton.

Area: DiD positive in all five dammed-area quintiles (+0.0011, +0.0031, +0.0021, +0.0044, +0.0106 seed 42; +0.0019 to +0.0036 seed 43); Spearman(dNSE, log area) at dammed -0.02 / +0.08.

## 2. Loss

`nse-batch` (per-day squared error / sigma_train^2) rewards a low-pass filter whenever r < 1, since its optimum sits at alpha = r: alpha falls at 3 of 4 dammed gauges, FHV worsens 3.6 points, the peak ratio drops 0.05, and 80 % of dammed predictions end up smoother than the observed record. The Gupta split makes this secondary at the median gauge (the r term carries the gain) and primary above DOR 1 (26-40 % of gainers). `kge` with alpha_weight acts on the routing head everywhere; `nnse-kge` has the sqrt cusp the code flags.

The larger limit is structural: `rho: 90` with a steady-state hotstart caps learnable T0 near days (learned median 0.35-0.53 d vs residence time 149 d, offline fit 1.75 d); the gain is 30x larger where T0 > 3 d was learned. `experiment.state_cache` or rho 365 for the release phase is the route; no loss substitutes.

Options that reward correct behaviour: (1) `nse-batch-deriv` (`loss.rs`, `deriv_weight` 0.5): scores the quantity the release changes and penalises smoothing past observed flashiness; risk: it changes the routing head too and is noisy at flashy small basins, so run with the routing head frozen first. (2) Per-gauge loss weights / a dammed-only KGE-alpha term (`batch_loss` has no weight vector; small change); risk: upstream n0 still feels it. (3) A one-sided smoothness hinge on (ac1_sim - ac1_obs)+ from daily differences; risk: new term, needs a gradcheck. (4) Freeze or L2-regularise a, b (b sign at chance; seasonal NSE unchanged).

## 3. Evaluation protocol

The sign test against zero in `paired_full_run.py` has the wrong null: the undammed paired change is -0.0007 (p 2e-15) / -0.0004 with no dam in the network, the trajectory confound of joint training. Use the DiD against the undammed population (Mann-Whitney: NSE p 2e-23 / 3e-23, nse_diff 1e-37 / 8e-34, KGE 0.012 / 0.002 with dammed worse) and the seed-consistency Spearman (0.63 vs -0.15).

Seeds: DiD per seed +0.0021 / +0.0043, sd 0.0016; SE 0.0005 needs ~10 joint seed pairs (~5 h CPU each). Instead: a frozen-routing arm (release head only, from the off checkpoint) makes undammed gauges bitwise identical, isolates the law, and a joint arm then measures the co-training cost as (joint - frozen) at undammed gauges. Keep one gauge CSV per comparison (T19: FLOW_SCALE changes downstream flows); smoke-CSV (+0.0056) and full-CSV (+0.0040 / +0.0057) numbers at the same 458 gauges are different experiments, not replicates.

Design: the 458 matched controls add nothing over the 1,448 undammed gauges (identical paired change); DOR and on-reach strata are the useful design; add a pre-registered "dam expected to act" stratum (nearest dam on reach, DOR >= 0.1) since ~60 % of dammed gauges have a sub-daily T0 and dilute the median. The smoke README's "controls within 0.01" criterion is 10x too loose for the confound (-0.0007 to -0.0011).

Headline: DiD at dammed gauges (NSE and nse_diff), a no-harm bound at undammed gauges (|change| < 0.001; exactly 0 under the frozen design), and the amplitude cost (alpha, FHV, peak ratio); the population median stays a gate (must not fall by more than the seed spread 0.005). Outliers: clip NSE/KGE to -1 before any mean, report medians and Hodges-Lehmann, name the NSE < -1 gauges.

## 4. Wrong or misleading now

1. `paired_full_run.py::paired` tests against zero (null is the undammed change).
2. `src/cli/run.rs` manifest `mean_nse_finite` (0.589) is set by two gauges with NSE < -16.
3. `src/training/metrics.rs::Metrics::compute` accumulates SSE/SSO sequentially in f32: max NSE error 3.0e-4 (3 gauges, NSE < -2), 99th pct 2.3e-5; KGE NaN when the prediction is constant while the script uses r = 0; the script drops gauges with < 365 finite days and Rust does not (0.73778 vs 0.73788).
4. FLV form reads +86 to +165 % at regulated gauges; use a log-form low-flow bias or the Q95 log ratio.
5. The page's "+0.0014" tile hides a bimodal effect (32-38 % gain > 0.01); the reproducibility statistic (rho 0.63 vs -0.15) is mentioned only in passing.
6. Nowhere is it said that the seasonal cycle did not change or that amplitude got worse; "small gain below dams" without "smooths peaks that were already too smooth" misdescribes what was learned.
7. Annual-peak lag and cross-correlation lag are uninformative at daily resolution for this law; event-based scoring needs observed-event matching.

## 5. Ranked opportunities

As in the summary above (frozen-routing arm; co-primary nse_diff / r / alpha / FHV; nse-batch-deriv with routing frozen; freeze a, b and lengthen the window; statistics and metric hygiene), each with its go/no-go criterion.

Full computed tables: `/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_metrics/tables.md` (levels per arm and group for 22 metrics; paired deltas per seed for 19 metrics x 11 groups with seed noise; KGE components; seed consistency; DiD; outliers; terciles; release parameters across seeds; dNSE vs T0; power) and `tables2.md` (Gupta decomposition; signed FDC deltas; over-smoothing; Mann-Whitney; heavy tails; largest changes with names; per-metric seed Spearman).
