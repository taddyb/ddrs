# LSTM-source selective-equifinality experiment (CPU arms 1–3) — design

Date: 2026-07-06. Branch: `unit_catchments` (contains master's leakance merge
`ea5fb39`; PR #24 head).
Prior findings: `docs/2026-07-06-leakance-nogo-scientific-summary.md` (leakance
= the reference unidentifiable pole), `docs/2026-06-23-precip-disaggregation-findings.md`.
Paper: `/home/tbindas/projects/ddr_equifinality/paper.tex` ("Beyond
Equifinality in Differentiable River Routing", Bindas & Shen).

## Problem

The paper's central claim — **selective equifinality** — is untested: channel
geometry (p, q → realized depth/width/hydraulic radius) is predicted to be
identifiable across structurally different lateral-inflow (Q′) sources, while
Manning's n is predicted to be a bias-absorber that shifts to compensate each
source's errors. No cross-source comparison run exists as of 2026-07-06. This
experiment trains the SAME MERIT CONUS routing model (same network,
attributes, observations, seed, budget) under the two NH LSTM Q′ sources
registered by PR #24 and measures parameter convergence at four levels. These
are arms 1–3 of the paper's four-source design; the dHBV2 arms
(`daily_dhbv2_merit_unit_catchments.ic`, `merit_dhbv2_UH_retrospective.ic`)
follow in a later session.

## Hypotheses and tests

Pre-registered BEFORE any run (guards against HARKing).

| # | Hypothesis | Test | Falsified if |
|---|---|---|---|
| H1 | Realized channel geometry converges: depth, top width, hydraulic radius at a common per-reach reference discharge agree across arms | Compute realized geometry from each arm's learned p, q at the common reference discharge; median per-reach cross-arm relative spread, compared to n's range-normalized spread | median relative spread of realized geometry ≥ that of Manning's n |
| H2 | Manning's n diverges as a bias-absorber; per-reach n-divergence is predicted by inter-source Q′ disagreement | Spearman ρ between per-reach cross-arm n-spread and per-reach Q′ disagreement (relative difference of eval-window mean Q′ across sources) | ρ ≤ 0.2, or n-spread ≈ geometry-spread (no selective contrast) |
| H3 | Gradient alignment is selective: cross-arm ∂L/∂(p,q) fields align; ∂L/∂n fields point in source-specific directions | Per-reach adjoint gradients at each arm's final checkpoint over identical deterministic windows; cross-arm cosine alignment per parameter | n-gradient cross-arm alignment ≥ geometry-gradient alignment |
| H4 | Gradient reachability decays with distance from gauges for ALL parameters (identifiability is gauge-local) | Gauged vs ungauged per-reach gradient-magnitude ratio and decay vs network distance to nearest gauge | gauged/ungauged ratio ≈ 1 (no decay) |

Verdicts use SUPPORTED / REFUTED / INCONCLUSIVE only.

**Divergence null (decided 2026-07-06):** relative — each parameter's
cross-arm spread is normalized by its physical range
(`params.parameter_ranges`), and n is judged against geometry. "Spread" =
per-reach max−min across arms, divided by the parameter's range width (for
realized geometry, divided by the per-reach cross-arm mean, i.e. a relative
range); the reported statistic is the median over analysis reaches. No seed
replicates in this pass; within-source variance is NOT measured (see
Concerns).

## Arms

Three runs. Everything constant except the Q′ source and its resolution
handling.

| Run | Source (store) | Q′ handling | Config (tracked) |
|---|---|---|---|
| R1 | daily-lstm — CudaLSTM, 288,421 divides, `days since 1981-01-01`, `/mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic` | flat repeat-24, no disagg head | `config/experiments/equif_daily_lstm_flat.yaml` |
| R2 | daily-lstm (same store) | precip-driven disagg head (`kan_head.disaggregation.use_precip: true` + `aorc_precip`) | `config/experiments/equif_daily_lstm_disagg.yaml` |
| R3 | hourly-lstm — MTS-LSTM, 197,088 divides, `hours since 1981-01-01`, `/mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic` | hourly-native slicing (disagg + hourly-native is a config error — PR #24 guard) | `config/experiments/equif_hourly_lstm.yaml` |

Constants across arms (from `config/merit_training.yaml`): seed 42 /
np_seed 42, 5 epochs, rho 90, warmup 5, L1 loss, train 1981/10/01–1995/09/30,
eval 1995/10/01–2010/09/30, identical attributes / gauges / adjacency,
`use_leakance: false`, `use_cuda_graphs: false`, CPU backend (NdArray f32,
deterministic — bitwise-reproducible arms). Both stores start 1981-01-01, so
the standard windows never reach into 1980.

R1↔R2 doubles as a controlled disagg-head ablation on a new source (prior
disagg evidence is dHBV2-UH only: NSE +0.037 / KGE −0.007,
`docs/2026-06-23-precip-disaggregation-findings.md`).

## Phase 1 — infrastructure (before any run)

1. **`--backend {cuda,cpu}` on `ddrs run`.** Refactor
   `src/cli/run.rs::dispatch`: workflow internals (`training_train`,
   `bootstrap_head_and_state`, `evaluate`) are already backend-generic; extract
   the dispatch body into a generic fn and match on the new flag — the same
   pattern as `src/bin/train.rs:60-75` (NdArray arm forces sparse_solver to
   cpu). Full test suite + `compare_ddr_sandbox` ABSOLUTE MATCH must stay
   green; `cargo install --path .` afterwards (STALE-BINARY TRAP).
2. **Parameter-gradient probe.** Adapt the `probe_zeta_gradient` adjoint
   machinery (master, `src/bin/probe_zeta_gradient.rs`) to emit per-reach
   ∂L/∂n, ∂L/∂p_spatial, ∂L/∂q_spatial at a given checkpoint over 96
   deterministic training-style windows (CPU; matches the leakance probe's
   window budget for comparability) → one NetCDF per arm. Read-only
   w.r.t. the routing core; no gated invariant files touched.
3. **Per-arm configs** under `config/experiments/` (tracked), derived from
   `config/merit_training.yaml` + the `daily-lstm` / `hourly-lstm` source
   groups (`config/sources/*.yaml`, shipped by PR #24).

## Phase 2 — runs and measurement

1. `ddrs plan` per arm (adjacency/baseline caches are content-addressed —
   first plan per source computes that source's summed-Q′ baseline).
2. **Smoke-time 2 mini-batches per arm on CPU** (`--max-mini-batches 2`)
   before committing to full runs.
3. Launch R1, R2, R3 **sequentially** via detached scripts
   (`nohup`, survive agent/session death; no agent babysitting;
   never `TaskStop` a task whose command IS the compute).
4. Per checkpoint: eval (in `train-and-test`), `dump_parameters --backend cpu`
   → `kan_parameters.nc`, gradient probe → `gradients.nc`.

## Phase 3 — cross-arm analysis (script in `ddrs-py`, uv-run)

Four levels, all restricted to the **intersection of real-coverage reaches**
(non-fill Q′ in ALL arms; coverage from `ddrs import --dry-run` reports):

1. **Raw parameters:** per-reach n, p_spatial, q_spatial; cross-arm Spearman ρ
   and range-normalized spread.
2. **Realized geometry:** depth, top width, hydraulic radius from learned p, q
   at a common per-reach reference discharge — median summed upstream Q′ over
   the eval window from the dHBV2-UH retrospective (arm-independent);
   sensitivity at the 10th/90th percentile flows.
3. **Routing skill:** median NSE/KGE per arm vs each arm's own summed-Q′
   baseline (CONUS reference: 0.6781 NSE / 0.7172 KGE, 2365 gauges,
   1995/10–2010/09) — gauge count and window reported with every number.
4. **Gradients:** cross-arm cosine alignment per parameter (H3);
   gauged/ungauged magnitude ratios and decay vs distance (H4).

## Deliverables

1. `--backend` flag on `ddrs run` (+ tests), merged on this branch.
2. Parameter-gradient probe binary/mode (+ smoke test).
3. Three tracked configs under `config/experiments/equif_*.yaml`.
4. Three completed runs under `.ddrs/runs/` with manifests, checkpoints,
   `kan_parameters.nc`, eval outputs, `gradients.nc`.
5. Cross-arm analysis script in `ddrs-py` + figures (parameter maps,
   convergence scatter, gradient-alignment maps).
6. Findings doc `docs/2026-07-XX-lstm-equifinality-findings.md` with the H1–H4
   verdict table, feeding the paper's Results section.

## Concerns / assumptions (per planning rules)

- **Concern — hourly CPU cost unknown:** ~85 min is the reference for a
  5-epoch daily CONUS CPU run; hourly was 2.1× daily per batch on GPU but the
  CPU ratio could be worse (24× routing timesteps). *Mitigation:* smoke-time
  first; if a full R3 epoch exceeds ~4 h, WAIT for GPU rather than cutting
  epochs — unequal training budgets would confound the convergence comparison.
- **Concern — COMID coverage differs across stores** (288,421 vs 197,088
  divides; uncovered reaches are 0.001-filled at read). Divergence at
  fill-in-one-arm reaches is an artifact. *Mitigation:* intersection-only
  analysis (Phase 3 preamble).
- **Concern — relative null is weak:** with no seed replicates, a
  geometry-vs-n contrast could in principle arise even if all parameters were
  noise-dominated. *Accepted* (user decision 2026-07-06); flagged as a paper
  limitation; seed replicates are the upgrade path.
- **Concern — CLI refactor risk:** `src/cli/run.rs` only (no gated invariant
  files), but any `src/` change re-arms the STALE-BINARY TRAP. *Mitigation:*
  full suite + sandbox parity gate + `cargo install` before runs;
  directory-style checkpoints as the runtime self-check.
- **Assumption — 5 epochs suffices** for parameter comparison: matches every
  prior benchmarked run; comparability outweighs asymptotic convergence.
- **Assumption — dHBV2-UH-derived reference discharge is fair to LSTM arms:**
  it is arm-independent, which is what the comparison requires; absolute
  realism matters less than commonality.
- **Assumption — `workflow: train-and-test` per arm** yields training + eval +
  parameter dump in one lifecycle; no bespoke per-arm steps.
- **Why this change:** produces the paper's first citable results (research
  questions 1–2) and the first quantitative test of the selective-equifinality
  thesis, with full provenance via the CLI manifest path.
