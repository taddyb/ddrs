# Synthetic losing-reach recoverability (positive control) — design

Date: 2026-07-03
Worktree: `zeta-sensitivity` (branch `worktree-zeta-sensitivity`)
Prior instruments:
`docs/2026-07-02-leakance-diagnosis-findings.md` (H1–H7 diagnosis),
`docs/2026-07-03-zeta-gradient-probe-findings.md` (P1/P2 refuted, P3 NO-GO).

## 1. Question

The gradient probe closed two links of the causal chain: the leakance
gradient reaches every reach (P1 refuted), and the trained objective is not
anti-leakance (P2 refuted) — but a real-magnitude loss is invisible at the
gauge (P3 NO-GO: the median Ref gauge's 5% discharge-uncertainty band is 53×
a literature-magnitude 0.01 m³/s loss). One link remains unmeasured:

> **When the gauge signal IS visible, does training attribute the missing
> water to the leakance term — or absorb it into routing parameters
> (Manning's n, q_spatial; the diagnosis's H5 mechanism)?**

This experiment plants *detectable-scale* losses in a synthetic world with a
known answer key and measures where the optimizer puts the water. It is a
**positive control**: if recovery fails here, no gauge-supervised rescue
(including the auxiliary-constraint experiment) can assume the recovery
machinery works, and the auxiliary design must also constrain routing
parameters.

Two commitments answer the original objection to a naive twin experiment
("too many variables — how could we even recover a learned zeta?"):

1. **Recovery target is the flux field** (per-reach `zeta_net`), never the
   internally degenerate triple `(K_D, d_gw, leakance_factor)`.
2. **Warm-start attribution**: the student initializes from the *same
   weights that generated the observations*, so its step-0 residual is
   exactly the planted signal routed to gauges — every gradient is
   attributable to the plant, with no cold-start stochasticity.

## 2. Architecture

```
trained hourly-ON checkpoint
(.ddrs/runs/2026-07-01T13-43-32Z-train-and-test/checkpoints/epoch_5_mb_9)
        │
        ▼
TEACHER  eval-mode forward, leakance ON, per-COMID overrides of the
         NORMALIZED leakance params at planted reaches
         (K_D → ceiling 1e-5, d_gw → floor −2, factor → target/ceiling)
         window: training period 1981-10-01 .. 1995-09-30
        │
        ├──► SYNTHETIC OBS   daily gauge discharge, zarr-v2 group in the
        │                    ObservationsStore layout (one f64 array per
        │                    STAID, NaN outside the simulated window)
        └──► ANSWER KEY      zeta accumulator netCDF (Σzeta / Σ|zeta| per
                             reach == exactly what was subtracted from b;
                             identity proven by tests/zeta_accum.rs)
        │
        ▼   students train on synthetic obs instead of USGS; all three
        │   run IN PARALLEL on CPU (NdArray, sparse_solver forced cpu,
        │   nice -n 10; RAM per run is small)
        ▼
  A  warm-start (epoch_5_mb_9 head, weights only), leakance ON
  B  warm-start (same head),                       leakance OFF
  C  cold init  (seed-42 head),                    leakance ON
        │
        ▼
MEASURE  per-student forward pass over the SAME training window with zeta
         accumulation → learned zeta_net field, compared reach-by-reach
         against the answer key
```

Training recipe for all students mirrors `config/experiments/
leakance_hourly_on.yaml` exactly (5 epochs, batch 64, rho 90, warmup 5,
seed 42, L1 loss, lr schedule 1e-3 → 5e-4 at epoch 3, hourly
precip-disaggregation head), with three derived-config edits only:
`data_sources.observations` → the synthetic store, `params.sparse_solver:
cpu`, and the per-run init/leakance settings below.

### Run roster

| Run | Init | Leakance | Tests |
|---|---|---|---|
| A | `epoch_5_mb_9` head, **weights only** (fresh Adam, fresh schedule) | ON | Positive control proper: does zeta rise toward the key at planted reaches? |
| B | same head, weights only | OFF (routing term disabled; head still emits the 3 params, ignored) | Pure H5 absorption: how well do n/q_spatial alone fit the planted losses? |
| C | cold seed-42 init | ON | Does the cold-point suppression regime (P2: 80.5% push-down) prevent zeta from emerging when the signal is real? |

B keeps the leakance-ON head architecture so the checkpoint loads and its
step-0 routing residual matches A's planted signal (minus the caveat in
§6.2). Only the routing-side subtraction is disabled.

### Plant sites and magnitudes

- **Sites**: the gradient probe's GAGES-II `Ref` reaches
  (`output/zeta_probe/probe_plan.csv`, 96 Ref reaches on least-disturbed
  gauges, stratified by uparea × aridity × reachability). **No round
  packing** — all plants coexist in one teacher world, because truth is
  per-reach from the accumulator, not a differential measurement.
- **Target magnitude** per site: `min(2 × band5, 0.5 × ceiling_flux)` where
  `band5` = 5% of the measurement gauge's mean observed flow (the probe's
  detectability band, already in `detectability_rows.csv`) and
  `ceiling_flux = area_z · K_D_max · (depth_mean − d_gw_min)` is the reach's
  maximum expressible leakance flux (`K_D_max = 1e-5`, `d_gw_min = −2`;
  `depth_mean` and `area_z_mean` from the diagnosis run's zeta diagnostic
  netCDF exports). (`K_D_max` widened from 1e-6 by user decision 2026-07-03: the original ceiling left only 23/96 sites expressible; teacher and students share the widened range, preserving expressibility-by-construction and step-0 attribution.)
- **Implementation**: overrides set `K_D = 1e-5`, `d_gw = −2`,
  `factor = target / ceiling_flux` (clamped to [0, 1]) — magnitude is tuned
  through `factor` alone. Sites where `ceiling_flux < 0.25 × band5` are
  **dropped and logged** (the term cannot express a detectable loss there;
  planting anyway would bake failure into the design). The drop count and
  dropped-site list go in the findings report — no silent truncation.
- Magnitude imprecision cannot corrupt the truth: the answer key is whatever
  the accumulator says was actually subtracted, not the target.

## 3. Pre-registered verdicts

Computed by the analysis script; thresholds fixed here, before any run.

| # | Metric | Definition | Bar |
|---|---|---|---|
| R1 | **Recovery ratio** | median over planted reaches of (run-A `zeta_net` / answer-key `zeta_net`), both accumulated over the same window | ≥ 0.5 RECOVERED; ≤ 0.1 FAILED; else PARTIAL |
| R2 | **Spatial precision** | median run-A `zeta_net` over NON-planted reaches vs the unmodified checkpoint's `zeta_net` field on the same window | < 2× baseline = precise; else SMEARED |
| R3 | **Absorption gap** | final-epoch mean training loss, A vs B, identical obs and batch draws (same seed) | A < B ⇒ leakance term needed; A ≈ B (within 5% relative) ⇒ H5 absorption confirmed at detectable scale |
| R4 | **Absorption map** | per-reach Δn (B minus checkpoint) around planted basins | descriptive — where the water went; no bar |
| R5 | **Cold emergence** | median run-C `zeta_net` at planted reaches vs run-C's own non-planted median | > 3× ⇒ signal overcomes cold suppression |

**Headline: the positive control PASSES iff R1 ≥ 0.5 AND R3 shows A < B.**
Everything else is mechanism description. If R1 fails while R3 shows A ≈ B,
absorption — not detectability — is the primary obstacle, and the
auxiliary-constraint experiment must constrain routing parameters as well as
zeta.

## 4. Components to build

1. **Teacher mode** — extend `src/bin/probe_zeta_gradient.rs` with
   `--mode teacher --plant-file <csv> --obs-output <zarr dir>
   --zeta-output <nc>`:
   - Plant file: CSV `comid,k_d_norm,d_gw_norm,factor_norm` (normalized
     [0,1] values; the override happens on the KAN head's output vectors
     BEFORE denormalization inside `setup_inputs`, so ranges/log-space
     handling stays untouched).
   - Forward-only chunked eval over the training window (replica of the
     existing perturb-mode loop) with zeta accumulation enabled.
   - Obs writer: zarr-v2 group in the `ObservationsStore` layout — root
     `.zgroup`, one child directory per gauge with `.zarray` + raw chunks,
     f64 daily m³/s, array length = the store's implicit
     1980-01-01–2020-12-31 axis (14,976 days) with NaN outside the teacher's
     simulated window, array names = the exact STAID keys the training
     dataset looks up (verified by the roundtrip test, §7). The hand-written
     zarr-v2 test fixture in `src/data/store/zarr_obs.rs` is the format
     template.
2. **Weights-only warm-start** — new optional config key
   `experiment.init_head: <path to head.mpk base>` in
   `src/training/bootstrap.rs`: loads KAN weights only (no `optim.mpk`, no
   `state.json`), starting a fresh run at epoch 1 with a fresh Adam.
   Mutually exclusive with `experiment.checkpoint:` (config load rejects
   both set).
3. **Run-B config compatibility** — allow `use_leakance: false` while the
   head still lists `K_D`/`d_gw`/`leakance_factor` in
   `learnable_parameters` (head emits them; routing ignores them). If
   current config validation rejects this, relax it with a warning log line
   rather than an error. Fallback if the relaxation turns invasive: B
   warm-starts from the 2×2's hourly-OFF checkpoint
   (`2026-06-23T02-49-12Z-conus-hourly-train-and-test`) — accepted and
   recorded as a dirtier step-0 residual.
4. **Site/plant script** — `scripts/recoverability_sites.py`: joins
   `output/zeta_probe/probe_plan.csv` (Ref rows) with `band5` from
   `detectability_rows.csv` and `depth_mean`/`area_z_mean` from the
   diagnosis zeta netCDF; computes `ceiling_flux`, applies the target rule
   and the drop rule; emits the plant CSV plus a sites report (kept,
   dropped, target vs ceiling distributions).
5. **Analysis script** — `scripts/recoverability_analysis.py`: loads answer
   key + three student zeta netCDFs + the checkpoint-baseline zeta field,
   computes R1–R5 exactly as §3 defines them, prints a VERDICTS block
   (same style as `zeta_gradient_analysis.py`), writes per-reach rows CSV.
6. **Recovery map notebook** — `scripts/notebooks/recovery_maps.ipynb` per
   `ddrs-eval-plots` conventions: CONUS map of recovered/planted ratio at
   planted reaches (run A), plus B's Δn absorption map.
7. **Findings report** — `docs/<run-date>-synthetic-recoverability-findings.md`
   in the hypotheses / changes / experiment / pass-fail / next-steps
   structure.

## 5. Execution sequence and compute

All CPU (`NdArray<f32>`, deterministic; GPU owned by another training job).
Everything `nice -n 10`.

1. **Timing gate** (before committing): one teacher chunk and one training
   mini-batch on the derived config. Probe data predicts ~20–25 s per
   training window ⇒ a 5-epoch × 96-batch student ≈ 3–4 h; the 14-year
   teacher forward ≈ 4–6 h (3-year eval measured ~1 h). If the gate
   measures > 3× these estimates, stop and re-scope (shorter comparison
   window) before launching.
2. **Teacher pass** → synthetic obs + answer key. Also run the
   **unmodified checkpoint** over the same window with accumulation → the
   baseline `zeta_net` field for R2 (can run in parallel with the teacher).
3. **Students A, B, C in parallel** (~3–4 h wall).
4. **Measurement passes** A/B/C over the training window with zeta
   accumulation, in parallel (B's is trivially zero — run it anyway as a
   consistency check that OFF-mode accumulation reports zeros).
5. **Analysis + maps + findings.**

Per-epoch checkpoints are kept for all students so a "moving but
unconverged" R1 can be diagnosed from the trajectory and run A extended
cheaply if needed (§6.3).

## 6. Concerns

1. **Run-B config relaxation could ripple.** `use_leakance: false` with
   leakance params listed may hit validation or head-output-slicing
   assumptions beyond one check. Why it matters: B is half the headline
   verdict. Mitigation: the fallback in §4.3 keeps the experiment alive at
   the cost of a noisier B.
2. **B's residual is not purely the plant.** Disabling the routing term also
   removes the checkpoint's base zeta field (median |zeta| 6.4e-4 m³/s
   everywhere) — B fights a small extra misfit A doesn't have. Why
   accepted: planted magnitudes are ~2 orders larger; correcting it would
   require a second teacher, doubling cost for a second-order effect.
   Recorded as a caveat on R3.
3. **Warm-start + L1 may under-move in 5 epochs.** A converged Adam start
   with a small relative residual may travel slowly; a FAILED R1 could
   partly mean "unconverged", not "unattributable". Mitigation: per-epoch
   zeta trajectories distinguish direction from distance; extend A before
   declaring failure.
4. **CPU timing estimates are extrapolations.** The optimizer step and the
   synthetic-obs read path are unmeasured. Mitigation: the §5 timing gate is
   a hard stop.
5. **STAID key mismatch between synthetic store and dataset lookup.** The
   CONUS dataset resolves observations by STAID; the zarr obs store was
   built for `Provider__GageId` names. Why it matters: a silent key miss
   would NaN-out gauges and quietly shrink the training set. Mitigation:
   the roundtrip test (§7) asserts every training gauge resolves, and the
   student driver logs the resolved-gauge count, which must equal the real
   run's count.
6. **Parallel students share the box with the GPU job's CPU threads.**
   Mitigation: `nice -n 10` everywhere; serialize if the machine gets
   tight.

## 7. Testing

- **Override injection** (`tests/` new): a small losing chain routed with
  plant-file overrides equals the same chain routed with the params
  substituted manually into the head output — byte-identical.
- **Synthetic-obs roundtrip**: write a store with the teacher's writer, open
  via `ObservationsStore::open`, assert values and dates round-trip exactly
  and every gauge STAID in a sample gauge list resolves.
- **Answer-key identity** already holds (`tests/zeta_accum.rs`); teacher
  mode reuses `enable_zeta_accumulation` unchanged.
- **Guard suites stay green**: `leakance_gradcheck`, `leakance_off_parity`,
  `zeta_accum`, `compare_ddr_sandbox` ABSOLUTE MATCH. No autograd Backward
  impl is touched (repo invariant 4); `init_head` touches bootstrap only.

## 8. Out of scope

- Observation noise / robustness (synthetic obs are noise-free by design —
  this tests optimizer attribution, not detectability; P3 already measured
  detectability).
- Recovery of the parameter triple (degenerate; flux only).
- The auxiliary-constraint experiment (next rung; inherits this verdict).
- Any GPU runs, promotion of leakance out of experimental status, or DDR
  backports.
