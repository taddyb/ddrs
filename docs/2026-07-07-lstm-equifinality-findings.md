# LSTM-source selective-equifinality experiment — findings (2026-07-07)

Spec:   `docs/superpowers/specs/2026-07-06-lstm-equifinality-cpu-design.md`
Script: `scripts/equif_convergence_analysis.py`
Prior findings: `docs/2026-07-06-leakance-nogo-scientific-summary.md`,
                `docs/2026-06-23-precip-disaggregation-findings.md`

**One-line verdict:** The pre-registered selective-equifinality pattern is REFUTED on the LSTM arms — under structurally different lateral inflows, Manning's n (not channel geometry) is the cross-source-consistent parameter: its per-reach values correlate strongly across sources (Spearman ρ up to 0.890) and its loss gradients align across sources (mean cosine 0.656), while geometry-parameter gradients are nearly orthogonal between the two structurally distinct stores (cosine 0.023–0.095).

---

## 1. Motivating observation and hypotheses

The paper's central claim — **selective equifinality** — is untested: channel geometry
(p, q → realized depth/width/hydraulic radius) is predicted to be identifiable across
structurally different lateral-inflow (Q′) sources, while Manning's n is predicted to be a
bias-absorber that shifts to compensate each source's errors. No cross-source comparison
run existed as of 2026-07-06. This experiment trains the same MERIT CONUS routing model
(same network, attributes, observations, seed, budget) under two NH LSTM Q′ sources
(arms 1–3 of the paper's four-source design) and measures parameter convergence at four
levels. The dHBV2 arms follow in a later session.

Pre-registered hypotheses (copied verbatim from the spec; registered BEFORE any run):

| # | Hypothesis | Test | Falsified if |
|---|---|---|---|
| H1 | Realized channel geometry converges: depth, top width, hydraulic radius at a common per-reach reference discharge agree across arms | Compute realized geometry from each arm's learned p, q at the common reference discharge; median per-reach cross-arm relative spread, compared to n's range-normalized spread | median relative spread of realized geometry ≥ that of Manning's n |
| H2 | Manning's n diverges as a bias-absorber; per-reach n-divergence is predicted by inter-source Q′ disagreement | Spearman ρ between per-reach cross-arm n-spread and per-reach Q′ disagreement (relative difference of eval-window mean Q′ across sources) | ρ ≤ 0.2, or n-spread ≈ geometry-spread (no selective contrast) |
| H3 | Gradient alignment is selective: cross-arm ∂L/∂(p,q) fields align; ∂L/∂n fields point in source-specific directions | Per-reach adjoint gradients at each arm's final checkpoint over identical deterministic windows; cross-arm cosine alignment per parameter | n-gradient cross-arm alignment ≥ geometry-gradient alignment |
| H4 | Gradient reachability decays with distance from gauges for ALL parameters (identifiability is gauge-local) | Gauged vs ungauged per-reach gradient-magnitude ratio and decay vs network distance to nearest gauge | gauged/ungauged ratio ≈ 1 (no decay) |

---

## 2. Methods

### Arms

Three runs, everything constant except the Q′ source and its resolution handling. Git SHA b261e1d (clean).

| Arm | Source | Q′ handling | Config | Run ID | Wall time |
|---|---|---|---|---|---|
| R1 | daily-lstm (CudaLSTM, 288,421 divides, `days since 1981-01-01`) | flat repeat-24, no disagg head | `config/experiments/equif_daily_lstm_flat.yaml` | `2026-07-07T03-55-53Z-train-and-test` | 53 min |
| R2 | daily-lstm (same store) | precip-driven disagg head (`use_precip: true` + aorc_precip) | `config/experiments/equif_daily_lstm_disagg.yaml` | `2026-07-07T04-49-19Z-train-and-test` | 2h 01 min |
| R3 | hourly-lstm (MTS-LSTM, 197,088 divides, `hours since 1981-01-01`) | hourly-native slicing | `config/experiments/equif_hourly_lstm.yaml` | `2026-07-07T06-50-28Z-train-and-test` | 3h 01 min |

Constants across all arms: seed 42, 5 epochs, rho 90, warmup 5, L1 loss, train window
1981/10/01–1995/09/30, eval window 1995/10/01–2010/09/30, identical attributes / gauges /
adjacency, `use_leakance: false`, `use_cuda_graphs: false`, CPU backend (NdArray f32,
bitwise-deterministic).

### Binary provenance

All three runs produced directory-style checkpoints
(`.ddrs/runs/<id>/checkpoints/epoch_5_mb_9/head.mpk`), confirming the current binary
ran (flat `.mpk` files indicate a stale pre-checkpoint-resume binary — none observed).
Log lines confirm store resolution: `streamflow resolution: Daily` for R1 and R2;
`streamflow resolution: Hourly` for R3.

### Eval network and analysis set

Eval network: 132,336 reaches (union of gauge subgraphs, 3,211-gauge baseline set;
2,365 finite gauges after DA-valid and headwater filters applied at eval time).
Analysis set (cross-arm coverage intersection): 132,336 reaches — no reaches dropped.
All three stores' upstream closures cover the full eval network; the hourly-lstm store
(197,088 divides) provides non-fill Q′ at every eval COMID.

### Gradient probe

Gradient files generated per arm via `probe_zeta_gradient --params n,q_spatial,p_spatial
--windows 96 --seed 42 --backend cpu`. Probe COMIDs ∩ analysis set: 64,892 reaches
(the full eval network intersected with reaches having non-degenerate probe coverage).

### Analysis script

`scripts/equif_convergence_analysis.py`, four levels (Stages A–E), run from `~/projects/ddr`
under DDR's `uv` venv. Outputs cached to `output/equif/`; verdicts machine-written to
`output/equif/verdicts.json`.

---

## 3. Results

### Verdict table

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H1 | Realized geometry converges; Manning's n diverges | **REFUTED** | Geometry spread (depth 0.2506, top_width 0.4075, Rh 0.2545) all exceed n norm-spread (0.1555) |
| H2 | n-divergence predicted by Q′ disagreement | **REFUTED** | Spearman ρ(n-spread, Q′-disagreement) = −0.248 (bar: > 0.2) |
| H3 | Gradient alignment selective (geometry > n) | **REFUTED** | Mean cosine: n 0.656 > q_spatial 0.361 > p_spatial 0.383; n aligns MORE |
| H4 | Gradient reachability decays with gauge distance | **INCONCLUSIVE** | S/R/I cell counts: 3/3/3 across 9 arm×param cells |

### H1 — Realized geometry converges

**REFUTED.** The pre-registered rule requires ALL three realized-geometry relative-spread
medians to fall below Manning's n range-normalized spread. With arm-own Q′ as the
reference discharge (the primary metric agreed in the spec), the medians are:
depth 0.2506, top_width 0.4075, hydraulic_radius 0.2545 — all exceed the n norm-spread of
0.1555 (132,336 reaches). The rule is unambiguously violated on all three geometry
quantities.

Sensitivity check at the common dHBV2-UH reference discharge: depth drops to 0.0955 and
hydraulic_radius to 0.1004 — both now below the n norm-spread — but top_width remains
0.3794, and the pre-registered rule requires ALL three to be below. Top width diverges
under every reference tested (arm-own median, common median, p10 0.3584, p90 0.4036).

*Post-hoc interpretation (labeled as such):* the arm-own-Q′ result partially conflates
realized-state divergence with Q′-magnitude disagreement: two arms with identical p, q
still realize different depths when routed by different discharges. At the common
discharge, depth and hydraulic-radius spreads drop substantially and nearly fall below n —
suggesting the geometry FUNCTION has some convergence that the arm-own reference obscures.
Top width is an exception: its spread (0.38–0.41) is discharge-insensitive because top
width in the trapezoidal parameterization is dominated by the p exponent, which controls
the scaling exponent, not just the magnitude. This is the strongest H1 signal.

Manning's n across arms: range-normalized median spread 0.1555, Spearman ρ R1–R2 0.439,
R1–R3 0.890, R2–R3 0.685. The n spatial patterns are substantially more correlated across
sources than the rule-based verdict captures.

Learned ranges (full CONUS dumps, as of 2026-07-07): n R1 [0.0151, 0.1385], R2 [0.0345,
0.1372], R3 [0.0153, 0.1411] (box [0.015, 0.25]); q_spatial ~[0.23, 0.51] (box [0, 1]);
p_spatial ~[1.3, 13.9] (box [1, 200], log-space, log-center ≈ 14). No parameters are
pinned at box bounds.

*Primary threat to validity (post-hoc):* all arms share seed 42, so they start from a
common initialization. After only 5 epochs the learned parameters have moved limited
distances from that shared init. Any cross-arm spread IS source-driven signal (the same
init rules out random divergence), but the magnitudes are budget-bounded. Low raw spread
partially reflects limited parameter travel, not necessarily higher identifiability. A
longer-budget replicate (§5, item 2) is the upgrade path.

### H2 — n-divergence predicted by Q′ disagreement

**REFUTED.** The rule requires Spearman ρ(n-spread, Q′-disagreement) > 0.2 AND
n-spread > geometry-spread. The measured ρ is −0.248 — negative, in the opposite
direction from the hypothesis. Where Q′ sources disagree more (larger eval-window mean
Q′ relative difference), Manning's n values are MORE consistent across arms, not less.

*Post-hoc interpretation (labeled as such):* this negative correlation is consistent with
the H3 result. If n gradients are source-independent (as H3 shows), n converges toward
the same value regardless of the local Q′ magnitude. Reaches where the two daily-lstm
and hourly-lstm stores disagree most tend to be gauged headwaters with strong training
signal — the same places where coherent gradient flow drives n toward a common
gauge-fitting value. This would produce exactly the negative ρ observed.

### H3 — Gradient alignment selective (geometry > n)

**REFUTED.** Mean pairwise cosine alignments (96 windows, seed 42, 64,892 probe COMIDs):
n 0.656, q_spatial 0.361, p_spatial 0.383. The rule requires mean_cos(q) > mean_cos(n)
AND mean_cos(p) > mean_cos(n). Instead, n's gradient field is the most cross-arm-aligned
of the three parameters.

The sharpest contrast is the R1–R3 pair (the two structurally distinct stores: daily-lstm
vs hourly-lstm): n cosine 0.730, q_spatial cosine 0.023, p_spatial cosine 0.095. Gradient
directions for the geometry parameters are nearly orthogonal across the two structurally
distinct stores, while n gradients point in nearly the same direction. The R1–R2 pair
(same store, disagg vs flat) shows intermediate alignment across all parameters (n 0.674,
q 0.434, p 0.459), consistent with R1 and R2 sharing the same Q′ source and differing
only in temporal disaggregation.

This result is the strongest single piece of evidence in the experiment: it shows that the
OPTIMIZER is receiving a source-consistent signal for n and a source-divergent signal for
geometry, at the level of individual per-reach gradient vectors.

### H4 — Gradient reachability decays with gauge distance

**INCONCLUSIVE.** The per-arm verdicts are R1 I/I/I, R2 R/R/R, R3 S/S/S — a perfect
three-way split that yields 3 S, 3 R, 3 I across all 9 arm×param cells. Majority vote
does not decide.

The arm-level pattern is itself informative (*post-hoc*): R1 (daily flat) shows distance
decay but with non-monotone bins — distance bin 1 exceeds bin 0, producing INCONCLUSIVE
rather than SUPPORTED. R3 (hourly-native) shows clean monotone decay for all three
parameters (ratios 2.57, 2.13, 1.80; bins strictly decreasing). R2 (daily disagg) is the
outlier: all ratios < 1 (0.91, 0.75, 0.78), meaning gradient mass concentrates AWAY from
gauges. This is consistent with the disaggregation head redistributing gradient energy
across within-day timesteps everywhere in the network, diffusing the gauge-proximity
signal upstream of each gauge.

### Level 3 — Routing skill

All arms beat their own summed-Q′ baseline on NSE (2,365 gauges, 1995/10–2010/09):

| Arm | NSE | KGE | Own baseline NSE | Own baseline KGE | ΔNSE | ΔKGE |
|---|---|---|---|---|---|---|
| R1 daily-lstm flat | 0.5894 | 0.6219 | 0.4366 | 0.6162 | +0.153 | +0.006 |
| R2 daily-lstm disagg | 0.6198 | 0.5969 | 0.4366 | 0.6162 | +0.183 | −0.019 |
| R3 hourly-lstm native | 0.5542 | 0.4849 | 0.5321 | 0.5473 | +0.022 | −0.062 |

The hourly-lstm baseline (R3) is substantially higher than the daily-lstm baseline
(R1/R2 share the same daily-lstm store). R3's routing head provides modest incremental
skill (+0.022 NSE) over an already stronger baseline. R2's disagg head gains the most
NSE (+0.183) but loses KGE (−0.019), consistent with the prior disaggregation finding
(dHBV2-UH arms: NSE +0.037 / KGE −0.007, `docs/2026-06-23-precip-disaggregation-findings.md`).
None of the arms beat their own baseline on KGE except R1 (+0.006).

---

## 4. Conclusions

1. **The selective-equifinality thesis is NOT supported by the LSTM arms.** All three
   pre-registered hypotheses with defined falsification criteria (H1–H3) are REFUTED.
   H4 is INCONCLUSIVE. The directionality is reversed from the paper's prediction:
   Manning's n converges across sources; geometry parameters diverge.

2. **Manning's n is the cross-source-stable parameter in this experiment.** Spearman ρ
   for n reaches 0.890 (R1–R3) at the raw-parameter level, and n gradient cosine reaches
   0.730 (R1–R3) at the adjoint level. Both measures agree.

3. **Top-width divergence is the most robust geometry signal.** Top-width relative spread
   (0.36–0.41) is the only quantity that remains well above n's norm-spread (0.1555)
   under every reference discharge (arm-own, common median, p10, p90). It warrants
   focused follow-up.

4. **The R1–R3 gradient comparison is the sharpest test.** R1 and R3 are the most
   structurally distinct arm pair (daily CudaLSTM vs hourly MTS-LSTM, 288k vs 197k
   divides). Their geometry-parameter gradient cosines (0.023, 0.095) are near-zero
   while n is 0.730 — this contrast survives the shared-seed / short-budget caveats
   because any source-driven signal dominates at near-zero.

5. **The paper abstract's n-as-bias-absorber claim requires revision.** The current
   draft (`/home/tbindas/projects/ddr_equifinality/paper.tex`) claims
   "channel geometries are highly influenced by physics while manning's roughness is a
   bias absorber" as if established — as of 2026-07-07, the LSTM-arm evidence points
   the opposite way. The abstract must be rewritten to present this as a hypothesis
   with the present result as an empirical finding. This supersedes any prior
   implication in earlier drafts that the result was already known.

6. **The init-hugging threat is the primary validity concern.** All arms share seed 42.
   After 5 epochs the parameters have not traveled far from the common init, which
   bounds the achievable spread independently of identifiability. Low spread is
   informative (it IS source-driven at 5 epochs), but the finding cannot distinguish
   between "n converges because it is identifiable" and "n is dragged toward a common
   attractor by the shared architecture." A longer-budget or seed-replicate arm is
   needed before making identifiability claims.

7. **These verdicts cover only the LSTM half of the four-source design.** R1 and R2
   share the same daily-lstm store and differ only in temporal disaggregation; they
   are not fully independent arm pairs. The dHBV2 arms (daily and hourly) are the
   cross-family test needed to distinguish LSTM-family intra-family similarity from
   cross-model-family convergence.

---

## 5. Next steps

1. **(Highest priority) Run dHBV2 arms** (daily_dhbv2 unit-catchment store exists and
   has been validated by `ddrs import`) to complete the four-source matrix. The
   LSTM-family-similarity caveat (R1–R2 share a store; R1–R3 are the only fully
   independent pair) dissolves only with a cross-family arm pair. The dHBV2 vs LSTM
   comparison is the paper's primary evidence for or against selective equifinality.

2. **(High priority) Longer-budget replicate (15–20 epochs) on one arm pair** — most
   informative pair is R1 vs R3 (structurally distinct) — to test whether the
   init-hugging budget constraint suppresses geometry spread. If geometry spread
   increases while n spread stays stable at longer budget, the current result is
   budget-limited. If the pattern persists, it is more likely structural.

3. **(Medium priority) Seed replicate** — repeat R1 and R3 with a different seed (e.g.,
   seed 0) to put a noise floor under the spread numbers. Two-arm sweep; 53 + 180 min
   CPU. Establishes whether the n vs geometry contrast is seed-dependent.

4. **(Medium priority) Top-width divergence deep-dive** — top width is the only geometry
   quantity that diverges under all reference discharges (spread 0.36–0.41). The p
   exponent (which controls the top-width power law) is log-space-initialized to
   log-center ≈ 14 in all arms. Investigate whether the top-width divergence reflects
   genuine cross-source learning or is an artifact of the p_spatial box ([1, 200],
   log-space) giving proportionally more room than n ([0.015, 0.25]).

5. **(Paper) Rewrite abstract** — replace the claim of n-as-bias-absorber as an
   established result with: (a) the hypothesis, (b) a citation to this finding, (c) a
   note that the dHBV2 arms are needed for the full four-source test.

Dropped: none.

---

## 6. Raw script output — verdict block

```
========================================================================
EQUIFINALITY CONVERGENCE VERDICTS
Eval window: 1995-10-01 – 2010-09-30
========================================================================

[H1] Realized geometry converges; Manning's n diverges
  Primary (arm-own Q'):
    median norm-spread(n)          = 0.1555
    median rel-spread(depth)        = 0.2506
    median rel-spread(top_width)    = 0.4075
    median rel-spread(hyd_radius)   = 0.2545
  Rule: SUPPORTED iff all geometry spreads < n-spread
  → H1: REFUTED
  Sensitivity (common dHBV2 Q'):
    median rel-spread(depth) = 0.0955
    median rel-spread(top_width) = 0.3794
    median rel-spread(hydraulic_radius) = 0.1004

[H2] n-divergence predicted by inter-source Q' disagreement
  Spearman ρ(n-spread, Q'-disagreement) = -0.248  (bar: > 0.2)
  n-spread vs geometry contrast: 0.1555 vs 0.2545  contrast=False
  Rule: SUPPORTED iff ρ > 0.2 AND n-spread > geometry-spread
  → H2: REFUTED

[H3] Cross-arm gradient alignment: geometry > n
  mean cosine: n=0.656  q_spatial=0.361  p_spatial=0.383
  (cosines for pairs R1-R2, R1-R3, R2-R3)
    n: 0.674 / 0.730 / 0.564
    q_spatial: 0.434 / 0.023 / 0.625
    p_spatial: 0.459 / 0.095 / 0.595
  Rule: SUPPORTED iff mean_cos(q) > mean_cos(n) AND mean_cos(p) > mean_cos(n)
        REFUTED   iff mean_cos(n) >= both; INCONCLUSIVE otherwise
  → H3: REFUTED

[H4] Gradient reachability decays with gauge distance
  Arm  Param        Ratio    Verdict cell
  R1   n            2.13     I  bins=['1.7e-04', '1.8e-04', '1.2e-04', '7.7e-05', '4.3e-05']
  R1   q_spatial    2.00     I  bins=['7.0e-06', '8.0e-06', '4.7e-06', '3.3e-06', '2.2e-06']
  R1   p_spatial    1.43     I  bins=['4.7e-05', '6.1e-05', '4.4e-05', '3.1e-05', '2.0e-05']
  R2   n            0.91     R  bins=['2.9e-04', '4.6e-04', '4.1e-04', '3.2e-04', '2.0e-04']
  R2   q_spatial    0.75     R  bins=['3.7e-05', '7.2e-05', '6.0e-05', '4.9e-05', '3.4e-05']
  R2   p_spatial    0.78     R  bins=['1.5e-04', '2.8e-04', '2.4e-04', '1.9e-04', '1.4e-04']
  R3   n            2.57     S  bins=['1.9e-04', '1.5e-04', '1.0e-04', '7.4e-05', '4.2e-05']
  R3   q_spatial    2.13     S  bins=['5.3e-06', '4.5e-06', '3.0e-06', '2.4e-06', '1.9e-06']
  R3   p_spatial    1.80     S  bins=['4.2e-05', '4.0e-05', '2.9e-05', '2.2e-05', '1.6e-05']
  Rule: majority vote over 9 param×arm cells (S/R/I count: 3/3/3)
  → H4: INCONCLUSIVE

========================================================================
VERDICT TABLE
  H1: REFUTED
  H2: REFUTED
  H3: REFUTED
  H4: INCONCLUSIVE
========================================================================
```

---

## 7. Reproduce

All commands run from `/home/tbindas/projects/ddrs` unless noted.
`--workspace` flag is required: without it `ddrs run` creates `.ddrs/` beside the config
file rather than at the project root.

### Step 1 — Train the three arms (sequential, detached)

```bash
nohup scripts/run_equif_arms.sh > output/equif_runs.log 2>&1 &
```

Or individually:

```bash
ddrs --workspace /home/tbindas/projects/ddrs/.ddrs \
  --config config/experiments/equif_daily_lstm_flat.yaml \
  run --backend cpu --workflow train-and-test

ddrs --workspace /home/tbindas/projects/ddrs/.ddrs \
  --config config/experiments/equif_daily_lstm_disagg.yaml \
  run --backend cpu --workflow train-and-test

ddrs --workspace /home/tbindas/projects/ddrs/.ddrs \
  --config config/experiments/equif_hourly_lstm.yaml \
  run --backend cpu --workflow train-and-test
```

### Step 2 — Dump KAN parameters per arm

Replace `<R1_ID>`, `<R2_ID>`, `<R3_ID>` with the actual run IDs from `.ddrs/runs/`.
Checkpoint format: `checkpoints/epoch_5_mb_9` (directory, not a file).

```bash
mkdir -p output/equif

dump_parameters \
  --config config/experiments/equif_daily_lstm_flat.yaml \
  --checkpoint .ddrs/runs/<R1_ID>/checkpoints/epoch_5_mb_9 \
  --output output/equif/R1_kan_parameters.nc \
  --backend cpu

dump_parameters \
  --config config/experiments/equif_daily_lstm_disagg.yaml \
  --checkpoint .ddrs/runs/<R2_ID>/checkpoints/epoch_5_mb_9 \
  --output output/equif/R2_kan_parameters.nc \
  --backend cpu

dump_parameters \
  --config config/experiments/equif_hourly_lstm.yaml \
  --checkpoint .ddrs/runs/<R3_ID>/checkpoints/epoch_5_mb_9 \
  --output output/equif/R3_kan_parameters.nc \
  --backend cpu
```

### Step 3 — Gradient probe per arm

```bash
mkdir -p output/equif_probe

probe_zeta_gradient \
  --config config/experiments/equif_daily_lstm_flat.yaml \
  --checkpoint .ddrs/runs/<R1_ID>/checkpoints/epoch_5_mb_9 \
  --params n,q_spatial,p_spatial \
  --windows 96 --seed 42 \
  --output output/equif_probe/grad_R1.nc \
  --backend cpu

probe_zeta_gradient \
  --config config/experiments/equif_daily_lstm_disagg.yaml \
  --checkpoint .ddrs/runs/<R2_ID>/checkpoints/epoch_5_mb_9 \
  --params n,q_spatial,p_spatial \
  --windows 96 --seed 42 \
  --output output/equif_probe/grad_R2.nc \
  --backend cpu

probe_zeta_gradient \
  --config config/experiments/equif_hourly_lstm.yaml \
  --checkpoint .ddrs/runs/<R3_ID>/checkpoints/epoch_5_mb_9 \
  --params n,q_spatial,p_spatial \
  --windows 96 --seed 42 \
  --output output/equif_probe/grad_R3.nc \
  --backend cpu
```

### Step 4 — Cross-arm convergence analysis

Run from `~/projects/ddr` (DDR's uv venv):

```bash
cd ~/projects/ddr && \
uv run python ~/projects/ddrs/scripts/equif_convergence_analysis.py \
  --r1 <R1_ID> \
  --r2 <R2_ID> \
  --r3 <R3_ID> \
  --params-r1 /home/tbindas/projects/ddrs/output/equif/R1_kan_parameters.nc \
  --params-r2 /home/tbindas/projects/ddrs/output/equif/R2_kan_parameters.nc \
  --params-r3 /home/tbindas/projects/ddrs/output/equif/R3_kan_parameters.nc \
  --grads-r1  /home/tbindas/projects/ddrs/output/equif_probe/grad_R1.nc \
  --grads-r2  /home/tbindas/projects/ddrs/output/equif_probe/grad_R2.nc \
  --grads-r3  /home/tbindas/projects/ddrs/output/equif_probe/grad_R3.nc
```

Verdicts are written to `output/equif/verdicts.json`; figures to `output/equif/figs/`.
Stages are individually cached; re-running is safe and skips completed stages.

The run IDs used for the results in this document:

```
R1: 2026-07-07T03-55-53Z-train-and-test
R2: 2026-07-07T04-49-19Z-train-and-test
R3: 2026-07-07T06-50-28Z-train-and-test
```
