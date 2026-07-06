# Zeta gradient-sensitivity probe — design

Date: 2026-07-02. Branch: `worktree-zeta-sensitivity` (worktree off `master`
@ `1aa9278`, post PR #23 merge — includes the zeta/depth eval instrumentation).
Prior findings this follows from:
`docs/2026-07-02-leakance-diagnosis-findings.md` (§5 next-steps item 1).

## Problem

The low-zeta diagnosis concluded that the learned leakance flux is small
because the optimizer chooses small — the driving head is throttled
(`d_gw ≈ depth`) and zeta tracks gauge proximity/river size instead of
aridity (H4 gauge bias / gradient starvation, SUPPORTED). But H4 as measured
is correlational: zeta is small where gauges aren't. Two rival mechanisms
produce that pattern and demand different remedies:

- **Starvation** — the gradient `∂Loss/∂zeta_r` is effectively DEAD away from
  gauges; the optimizer never receives a signal there. Remedy: auxiliary
  supervision (losing-potential maps, water-table attributes).
- **Rejection** — the gradient is alive off-gauge but consistently points
  toward smaller zeta; the discharge objective actively dislikes leakance on
  those reaches. Remedy: the objective (or the physics) is wrong for those
  reaches; auxiliary supervision would fight the loss rather than fill a gap.

A full twin (recover-planted-zeta-by-retraining) experiment cannot separate
these: the parameter triple `(K_D, d_gw, factor)` is internally degenerate
(only the flux is even in principle identifiable), routing params absorb
planted losses (the H5 finding), and training stochasticity smears the
endpoint. This spec replaces it with two **no-training** instruments.

## Stage 1 — adjoint reachability map

**Question:** where can the gauges "see" zeta at all, and which way does the
signal point?

**Instrument:** per-reach gradients of the actual training objective with
respect to the per-reach leakance parameter vectors, read via autograd at two
parameter points. No training step is ever taken.

**Mechanics** (new binary `src/bin/probe_zeta_gradient.rs`):

1. Load config `config/experiments/leakance_hourly_on.yaml` and a checkpoint,
   exactly like `bin/eval`.
2. Sample N training-style batches (default N=32, CLI-flag, seed-fixed):
   `batch_size` 64 gauges × rho=90-day windows drawn across the training
   period — the same sampling distribution the optimizer saw.
3. Per batch: KAN head inference on the inner backend → detach the three
   denormalized per-reach leakance vectors (`K_D`, `d_gw`, `leakance_factor`)
   and re-lift them as `require_grad` Autodiff leaves (the
   `tests/leakance_gradcheck.rs` pattern; the analytical `TimestepLeakanceOp`
   backward already supplies exact grads for these parents — **no change to
   any Backward impl**) → route with leakance → tau-trim + daily downsample →
   **L1 loss vs USGS observations** (the objective this run trained with;
   the config has no `loss:` block ⇒ default L1) → `backward()` → read the
   three leaf gradients.
4. Accumulate per reach across batches: `Σ|g|`, `Σg` (signed), and a
   window-coverage count. A reach appears in a batch iff it is in one of the
   batch gauges' subgraphs; coverage counts make under-sampled reaches
   visible instead of silently noisy.
5. Write netCDF (append-or-create, same idiom as `write_zeta_netcdf`):
   dimension `COMID_probe` (union of covered reaches), variables
   `grad_factor_abs`, `grad_factor_net`, `grad_dgw_abs`, `grad_dgw_net`,
   `grad_kd_abs`, `grad_kd_net` (all mean-per-covered-window), `n_windows`,
   plus global attrs (checkpoint, N, seed, ddrs version).
6. Run twice with **identical window samples**:
   - point (a) — the trained hourly-ON checkpoint
     (`.ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9`):
     the gradient landscape at convergence;
   - point (b) — the untrained head (fresh seed-42 init, no checkpoint):
     the signal that shaped epoch 0.
   Two output files. The pair separates "converged-so-flat" from
   "never-saw-a-signal".

**Guards:** `leakance_gradcheck` (8/8) already proves gradient exactness. New
`tests/zeta_gradient_probe.rs` on the mock network: (i) leaf grads finite and
nonzero on a losing chain; (ii) accumulating two batches equals the sum of two
single-batch runs; (iii) the routed discharge is byte-identical to the
non-probed eval path (lifting leaves must not perturb the forward).
`compare_ddr_sandbox` ABSOLUTE MATCH after any `src/` change, as always.

## Stage 2 — detectability bound

**Question:** is a literature-magnitude reach loss even measurable at the
gauges that would have to teach it, relative to the noise the optimizer
trains through?

**Instrument:** forward-only perturbation of the lateral-inflow forcing —
**zero routing-code changes**. Adding `+δ` m³/s of `q_prime` at reach r poses
the same detectability question as removing it (and avoids the
`clamp_min(discharge)` floor on small reaches), with δ ∈ {0.01, 0.1} m³/s
spanning the Shanafield & Cook transmission-loss range.

**Site selection (the dam/lake control):** join `gages_3000.csv` STAIDs to
GAGES-II `CLASS` from
`/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.dbf` (fields verified
present: `STAID`, `CLASS`, `HCDN_2009`). Primary population = subgraphs of
`CLASS == Ref` gauges only; `Non-ref` kept as a labeled contrast group whose
detections are reported as upper bounds (regulation storage/release is
indistinguishable from GW–SW exchange at the hydrograph level). Sample ~1000
probe reaches stratified by upstream area tercile × aridity tercile ×
stage-1 reachability tercile.

**Mechanics:** batch probes so no two perturbed reaches share any downstream
gauge → ~10–20 full-eval forwards (inner backend, no tape), plus two
unperturbed identical runs to measure the CUDA rerun-noise floor empirically.
Detection criterion per probe, at the nearest downstream gauge: ΔQ
(eval-window mean and peak-day) vs (a) the measured noise floor, (b) 5% and
10% observational-uncertainty bands on that gauge's flow, (c) the ΔL1-loss.

## Analysis, verdicts, and maps

`scripts/zeta_gradient_analysis.py` (run under ddr's uv venv), pre-registered
criteria:

- **H4-starvation SUPPORTED** iff median `|∂L/∂factor|` on ungauged (and
  separately, arid-tercile) reaches sits ≥1–2 orders of magnitude below
  gauged reaches, at BOTH parameter points.
- **H4-rejection SUPPORTED** iff off-gauge magnitudes are comparable but the
  signed gradient consistently pushes zeta down (positive `∂L/∂factor`) on
  arid/ungauged reaches.
- **Detectability NO-GO** iff <10% of Ref-basin probes at δ = 0.01 m³/s clear
  both the noise floor and the 5% observational band — no gauge-only
  objective can teach real-magnitude leakance, and auxiliary supervision
  (diagnosis findings §5 item 2) becomes the only viable rescue.
- **Cross-check:** stage-1 reachability must rank-predict stage-2
  detectability; if it doesn't, the adjoint map is suspect and the findings
  say so.

**CONUS gradient maps (via the `ddrs-eval-plots` skill, parameter_map +
gauge-scatter conventions):** notebooks written to the probe output dir's
`plots/`, executed from `./ddrs-py`'s venv, PNGs at dpi 300, CONUS bounds
`(-125, -66) × (24, 53)`, CartoDB.Positron basemap:

1. **Per-gauge scatter map** — one point per gauge at `LAT_GAGE`/`LNG_GAGE`,
   colored by `log10` median `|∂L/∂factor|` over the gauge's subgraph, with a
   companion sign map (net gradient direction per basin). Four small
   multiples: trained-|g|, trained-sign, cold-|g|, cold-sign — the
   trained-vs-cold national contrast is the H4 picture in one figure.
2. **Per-reach MERIT-polygon map** of `log10|∂L/∂factor|` on the covered
   network — shows the within-basin decay away from gauges that per-gauge
   aggregation hides.

The findings doc quotes these figures directly.

## Deliverables

1. This spec.
2. `src/bin/probe_zeta_gradient.rs` + `tests/zeta_gradient_probe.rs`.
3. Two stage-1 reachability netCDFs (trained + cold) + stage-2 probe results
   (netCDF or CSV) in a run-dir-style output directory.
4. Gradient-map notebooks + PNGs per the `ddrs-eval-plots` conventions.
5. `scripts/zeta_gradient_analysis.py`.
6. Findings report `docs/<run-date>-zeta-gradient-probe-findings.md` (dated
   the day the battery completes) in the established format (hypotheses /
   methods / results / conclusions / next steps).

## Concerns / assumptions (per planning rules)

- **Concern — memory.** A rho-90 hourly window × 64 gauge subgraphs under
  autograd equals training's existing footprint, so it fits; the probe must
  never put the 15-year eval window on tape. Enforced by construction (the
  probe reuses the training batch shape).
- **Concern — optimum flatness.** Per-reach grads at convergence are small by
  construction (only the KAN-weight projection is exactly zero, but
  magnitudes shrink). The cold-start point is the control; conclusions about
  starvation must hold at both points.
- **Concern — grads w.r.t. denormalized params.** Leaf grads are in
  denormalized units (factor is [0,1]-native; d_gw in m; K_D in 1/s,
  log-space range). Cross-parameter magnitude comparisons are therefore
  unit-laden — the analysis compares WITHIN a parameter across space, never
  across parameters.
- **Assumption — q′-vs-b_rhs equivalence for stage 2.** A q′ perturbation
  differs from a b_rhs zeta injection by a per-step `c4` factor; second-order
  for a detectability *bound*, and it buys zero routing-code risk.
- **Assumption — N=32 windows suffice.** ~2k gauge-draws cover the eval
  network via subgraph unions; `n_windows` coverage counts verify, and N is a
  CLI flag if holes appear.
- **Assumption — GAGES-II point-shapefile CLASS is an adequate regulation
  flag.** Ref = least-disturbed per USGS; stricter HCDN_2009 available as a
  sensitivity check. No NID download needed.
- **Why this experiment.** It is the cheapest decisive test of the
  diagnosis's central mechanism, it disambiguates two remedies that would
  otherwise be pursued blind, and its detectability bound gates whether ANY
  leakance rescue on gauged discharge alone is worth attempting.
