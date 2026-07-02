# Leakance low-zeta diagnosis — design

Date: 2026-07-01. Branch: `worktree-leakance-diagnosis` (worktree off `master`
@ `25f93aa`, post PR #22 merge).
Lit review: `docs/2026-07-01-leakance-litreview.md`.
Prior findings: `docs/2026-07-01-leakance-hourly-findings.md`.

## Problem

The 2×2 experiment returned GO-but-marginal: leakance helps skill under hourly
forcing, but the learned exchange is tiny — median |zeta| 6.4e-4 m³/s,
|zeta| > 0.01 m³/s on only 10.4% of the 64,892 eval reaches, `K_D` pinned at
its `1e-6 s⁻¹` ceiling on 100% of reaches. The goal of this experiment is to
**explain why zeta is so small**, with hypotheses each falsifiable by a cheap
test, and only then decide whether/how to fix it.

Back-of-envelope anchor: at the ceiling, `zeta_max ≈ 0.33 · 10⁴ m² · 10⁻⁶ s⁻¹
· 0.3 m ≈ 1e-3 m³/s` for a typical reach — right at the observed median. The
literature (litreview §A1) says the ceiling itself corresponds to a clogged
silt/clay bed; sand/gravel losing streambeds sit 2–4 orders of magnitude
higher. But the identifiability literature (§B) predicts that even an
unbounded `K_D` may not move the median: gauge bias and equifinality can hold
zeta near zero regardless. The diagnosis must separate these.

## Hypotheses and tests

All tests run on existing artifacts plus the Phase-1 exports. Each has a
prediction and a falsification criterion; the deliverable is a ranked-verdict
report.

| # | Hypothesis | Test | Falsified if |
|---|---|---|---|
| H1 | **Structural ceiling**: `K_D ≤ 1e-6` dimensionally caps zeta below 0.01 m³/s for most reach geometries | Per-reach closed-form `zeta_max = 1.0 · area_z · 1e-6 · (depth_mean + 2)` using learned geometry + eval mean depth; fraction of reaches that could exceed 0.01 anywhere in the box | most reaches could exceed the bar within the current box |
| H2 | **Driving-head starvation**: learned `d_gw` sits near typical depths so `(depth − d_gw)` is tiny/sign-flipping | Distribution of `(depth_mean − d_gw)`; fraction ≤ 0 and fraction < 0.1 m | driving head ≫ 0 on the large majority of reaches |
| H3 | **KAN variance collapse** (user's hypothesis): head maps attributes to near-constant leakance params | Dispersion (IQR/median, CV) of `K_D`, `d_gw`, `leakance_factor` vs routing params (`n`, `q_spatial`) from the same head; attribute R² (aridity/permeability proxies from `merit_global_attributes_v2.nc`) | leakance params show spatial structure comparable to routing params |
| H4 | **Gauge bias / gradient starvation**: loss only samples large perennial gauged rivers; no gradient pressure where leakance matters | Stratify `zeta`, `K_D` by gauged-vs-ungauged reach, upstream area, aridity (litreview §A4: losing potential concentrates in dry/flat regions) | zeta is equally small on arid ungauged headwaters and gauged perennial rivers, with no stratification signal |
| H5 | **Equifinality**: Manning's n / storage absorb the attenuation leakance would provide | Paired ON−OFF comparison of learned `n`, `x_storage` distributions (dump_parameters exists for both hourly and daily pairs) | routing params indistinguishable between ON/OFF |
| H6 | **Wrong yardstick**: 0.01 m³/s absolute is meaningless on headwaters; fractional loss may be non-trivial | Distribution of `|zeta| / q_mean` per reach; compare against the 5–10% gauge-detectability band (litreview §B3) | fractional losses also negligible everywhere |
| H7 | **Model-form error**: linear `(depth − d_gw)` is a connected-regime law; strongest-losing reaches are disconnected (flux saturates) | Map-based: where does Brunner-disconnection plausibly hold (arid, deep water table)? Is learned `d_gw` boundary-pinned there? Analytic note on the saturating-flux alternative | `d_gw` interior everywhere including arid regions, and connected-regime assumption defensible for the eval network |

Note H1–H2 are "the box is wrong", H3–H5 are "the gradient is wrong", H6–H7
are "the question is wrong". They are not mutually exclusive; the report
assigns each a verdict (supported / refuted / inconclusive) with effect sizes.

## Phase 1 — instrumentation (Rust, small)

H1, H2, H6 need per-reach eval-window **mean depth** and **mean discharge**,
which the zeta accumulator does not export. Extend it in place, same pattern:

- `MuskingumCunge`: alongside `zeta_abs_sum`/`zeta_net_sum`, accumulate
  `depth_sum` and `q_sum` (inner-backend tensors, no tape) when
  `enable_zeta_accumulation` is on. Depth is already computed per step in the
  leakance branch (`depth_p` saved primitive); q is `q_next`.
- `ZetaSums` (src/training/forward.rs): add `depth_sum`, `q_sum` fields to
  `new()`/`merge()`.
- `EvalOutput`/`evaluate` (src/training/eval.rs): carry `depth_mean`,
  `q_mean` next to the zeta means.
- `write_zeta_netcdf` (src/dump_parameters.rs): two new variables
  `depth_mean` (m) and `q_mean` (m³/s) on the existing `COMID_eval`
  dimension, append-or-create semantics unchanged.
- `tests/zeta_accum.rs`: extend — accumulated depth/q means must equal the
  hand-computed per-step means on the mock network; accumulation still must
  not perturb discharge (byte-equal).

Guards that must stay green (unchanged bar):
`cargo test --test zeta_accum --test leakance_gradcheck --test
leakance_off_parity` and `cargo run --release --example compare_ddr_sandbox`
→ ABSOLUTE MATCH. Training path untouched (accumulation is eval-only;
invariant 4 intact).

Then re-run the two leakance-ON evals (~10 min each, no retrain) with the
legacy eval binary `--zeta-output`, appending the new variables to each run's
`kan_parameters.nc`:
`.ddrs/runs/2026-07-01T13-43-32Z-train-and-test` (hourly-ON) and
`.ddrs/runs/2026-07-01T21-20-27Z-train-and-test` (daily-ON).
Evals run from the MAIN working tree's `.ddrs` (run dirs live there), with a
binary built from this worktree.

## Phase 2 — hypothesis battery (Python)

One script, `scripts/leakance_diagnosis.py`, run under ddr's uv venv (netCDF4/
zarr/xarray available there), consuming:

- `<run_dir>/kan_parameters.nc` (full-CONUS learned params on `COMID` +
  eval-network `zeta`, `zeta_net`, `depth_mean`, `q_mean` on `COMID_eval`),
- `merit_global_attributes_v2.nc` (aridity/permeability proxies),
- the gauge list + gauge-subgraph adjacency (gauged-reach mask),
- geometry relations from `src/geometry.rs` re-expressed in numpy
  (`area_z = (p·depth)^q_eps · length`).

It prints one section per hypothesis (effect sizes + verdict) and writes the
report skeleton. Findings land in
`docs/2026-07-02-leakance-diagnosis-findings.md` with a ranked table.

## Phase 3 — gated fix (at most one retrain)

**Gate: H1 supported AND the gradient is alive** (H3/H4 do not show total
collapse — i.e. leakance params carry spatial structure and/or stratify with
losing-ness). Then:

- Widen `K_D` to `[1e-8, 1e-4]` (litreview §A1: includes sand-bed regime;
  stop short of 1e-3 to limit the risk of runaway loss at f32), new experiment
  config `config/experiments/leakance_hourly_on_kd4.yaml` cloned from
  `leakance_hourly_on.yaml`.
- Retrain the hourly-ON arm only (same seed 42, same window); hourly-OFF
  control is unchanged and reused.
- Re-run `scripts/leakance_subset_analysis.py` + zeta export; update findings.

If the gate fails (H4/H5 dominate), a widening retrain is predicted useless:
the report instead recommends the literature remedies — synthetic
losing-reach experiment to prove the gradient path, auxiliary constraints
(baseflow/recharge), or a saturating flux form (H7) — and stops without
spending GPU.

## Deliverables

1. `docs/2026-07-01-leakance-litreview.md` (done, committed with this spec).
2. Phase-1 instrumentation + extended `tests/zeta_accum.rs` on this branch.
3. Updated `kan_parameters.nc` in both ON run dirs (depth_mean/q_mean added).
4. `scripts/leakance_diagnosis.py` + `docs/2026-07-02-leakance-diagnosis-findings.md`.
5. If gated in: the widened-K_D run, updated findings, GO/NO-GO refresh.

## Concerns / assumptions (per planning rules)

- **Concern — mean depth flattens dynamics.** A 15-year mean depth hides the
  sub-daily range that makes hourly leakance work; H2 could look healthier or
  sicker than it is at the mean. *Mitigation:* H2's verdict is stated at the
  mean with this caveat; if it is the deciding hypothesis, a follow-up
  accumulator for `mean((depth − d_gw)⁺)` is a 10-line extension.
- **Concern — H5 confounded by stochasticity.** ON/OFF runs differ by CUDA
  scatter-add nondeterminism too; routing-param shifts are suggestive, not
  decisive. Treated as such in the report.
- **Concern — netcdf append clobber.** `write_zeta_netcdf` appends to files
  that already contain full-CONUS dump variables; the existing
  add-or-overwrite semantics were built for exactly this, but both files get
  a /tmp backup before re-eval anyway (they were expensive to produce).
- **Assumption — the two 2026-07-01 ON checkpoints are the subjects.** No
  Phase-1/2 retrains; the diagnosis explains *those* runs.
- **Assumption — attribute proxies suffice for "losing-ness".** We have no
  water-table observations; aridity + permeability-class attributes stand in
  (consistent with Jasechko 2021's drivers).
- **Why this change.** The GO was marginal (10.4% vs 10% bar). This diagnosis
  decides — with physical justification — whether leakance gets promoted,
  re-parameterized (wider K_D / saturating form), or documented NO-GO.
