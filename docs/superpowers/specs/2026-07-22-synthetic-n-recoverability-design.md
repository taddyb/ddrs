# Synthetic n-recoverability across real Q' sources — design

Date: 2026-07-22
Prior instruments:
`docs/2026-07-16-aorc2f-wave1-findings.md`, `docs/2026-07-16-wave2-cross-wave-findings.md`
(this campaign's 4 real-Q'-source arms),
`docs/2026-07-04-synthetic-recoverability-findings.md` +
`docs/superpowers/specs/2026-07-03-synthetic-recoverability-design.md`
(the leakance positive-control pattern this design generalizes),
`/tmp/experiment-handoff-lstm-equifinality-parameterization-patterns.md`
(the standing selective-equifinality hypothesis this experiment tests).

## 1. Question

The separate, pre-registered LSTM-equifinality campaign found — in REAL data,
with REAL (noisy, incomplete) USGS observations — that Manning's n diverges
substantially across differently-sourced Q' inputs (relative spread 0.4512,
the largest of any learned quantity) while channel geometry (q_spatial,
p_spatial) is comparatively stable, and that two of three arms recovered a
roughness-INCREASES-downstream slope opposite classical Leopold-Maddock
hydraulic geometry. That campaign's verdict was INCONCLUSIVE: real
observation noise and real physical heterogeneity across basins are
confounds that cannot be ruled out from real data alone.

> **Can we build a ground-truth-anchored control: prescribe a KNOWN Manning's
> n field, hold geometry fixed to what real training already agrees on, route
> a known "true" Q' through the real MC solver to generate noise-free
> synthetic gauge observations, then train fresh KAN heads against those
> synthetic observations using each of this campaign's 4 REAL Q' stores as
> the forcing — and see whether recovered n diverges by Q'-source (while
> geometry does not), even though the true n is identical across all four?**

If recovered n systematically deviates from the known truth in a way that
tracks each store's departure from the true Q' (while q_spatial/p_spatial
stay close to truth), that is a direct, ground-truth-anchored demonstration
of the bias-absorption mechanism — no real roughness observations needed,
because the roughness observations are built by construction.

Unlike the leakance recoverability experiment (which planted a *subtle*,
sub-percent per-reach flux perturbation that turned out to be invisible below
the windowed training objective's ~130× noise floor), this design swaps the
*entire* Q' forcing across arms — a global, large-magnitude (O(10-100%) at
many reaches) discharge difference, well above that floor. That prior
finding is noted as a risk to watch, not assumed away.

## 2. Architecture

```
                         TRUTH INGREDIENTS (fixed, no training)
                         ────────────────────────────────────
  Q'_true = merit_dhbv2_UH_retrospective.ic
            (the real USGS-validated standard benchmark store)

  geometry truth (q_spatial, p_spatial) = per-reach MEDIAN across the 4
            already-converged real checkpoints from this campaign:
              .ddrs/runs/2026-07-16T02-22-14Z-train-and-test/checkpoints/epoch_5_mb_35  (AORC2F distributed)
              .ddrs/runs/2026-07-16T02-23-20Z-train-and-test/checkpoints/epoch_5_mb_35  (AORC2F lumped)
              .ddrs/runs/2026-07-16T11-31-50Z-train-and-test/checkpoints/epoch_5_mb_35  (daily-lstm)
              .ddrs/runs/2026-07-16T11-31-52Z-train-and-test/checkpoints/epoch_5_mb_35  (hourly-lstm)
            via dump_parameters on each, then per-COMID median. Data-driven
            consensus of "whatever the most common trained value trend is" —
            cancels any one arm's idiosyncratic noise, and is consistent
            with the equifinality campaign's own finding that geometry is
            the more identifiable quantity.

  n truth (prescribed, per-COMID, two variants):
    Phase 1 (primary): Leopold-Maddock power law,
            n = clip(0.08 * (uparea / uparea_median)^(-b), 0.015, 0.15),
            DEcreasing downstream (classical hydraulic-geometry direction —
            the opposite sign of the equifinality campaign's anomalous
            trained-model finding). `b` is calibrated empirically against
            the real `log10_uparea` distribution so the field spans
            roughly the full [0.015, 0.15] range across CONUS (start from
            b=0.15 and adjust once the truth-n script runs against the
            actual attribute distribution) — this is a tuning detail, not
            a design fork.
    Phase 2 (confirmatory, run only if Phase 1 shows a clear signal):
            Gaussian-noise field, mean 0.08, spread within [0.015, 0.15],
            IID per reach — no spatial structure at all. Any recovered
            structure in this case is pure training-side artifact.

                                    │
                                    ▼
                         TEACHER (new eval-only mode, --backend cpu)
                         frozen consensus geometry + per-COMID n OVERRIDE
                         (new eval-path seam in src/training/forward.rs,
                         same pattern as LeakanceOverride but for `n`),
                         forced Q' = Q'_true, full simulation window
                         1981/10/01-2010/09/30, no gradient computed
                                    │
                                    ▼
                         synthetic gauge observations (zarr-v2, REUSES
                         obs_writer.rs unchanged) at the 2,365 USGS gauge
                         locations, daily, noise-free
                                    │
        ┌───────────────┬──────────┼──────────┬───────────────┐
        ▼               ▼          ▼          ▼               │
   student:        student:    student:    student:             │
   AORC2F          AORC2F      daily-lstm  hourly-lstm           │
   distributed     lumped                                       │
   (cold seed-42, --backend cpu, learns n/q_spatial/p_spatial     │
    from attributes; forcing = each real campaign Q' store;       │
    everything else = EXACT copy of this campaign's own configs,  │
    only data_sources.observations repointed to the synthetic     │
    store)                                                        │
        │               │          │          │                  │
        └───────────────┴──────────┼──────────┴──────────────────┘
                                    ▼
                         dump_parameters (--backend cpu) → recovered
                         n/q_spatial/p_spatial per COMID, per student
                                    │
                                    ▼
                         COMPARE vs known truth:
                           n        → vs the prescribed truth field
                           q/p      → vs the 4-checkpoint consensus median
                         per-reach error, drainage-area-binned slope fit,
                         cross-arm divergence (does n diverge by Q'-source
                         while q/p do not?)
```

## 3. Pre-registered verdicts

Computed by the analysis script; thresholds fixed here, before any run.

| # | Metric | Definition | Bar |
|---|---|---|---|
| S1 | **n recovery error** | per-arm median absolute error, recovered n vs truth n, all reaches | report per arm; compare across arms |
| S2 | **n slope fidelity** | per-arm fitted slope of recovered n vs log10(uparea), vs the true (negative) slope | slope flips sign (positive) ⇒ reproduces the equifinality anomaly |
| S3 | **Geometry recovery error** | per-arm median absolute error, recovered q_spatial/p_spatial vs the 4-checkpoint consensus median | should be small and CONSISTENT across arms |
| S4 | **Cross-arm divergence ratio** | (max−min across the 4 arms) of S1, divided by (max−min across the 4 arms) of S3 | ≥ 3× ⇒ n diverges by Q'-source substantially more than geometry does — headline PASS criterion |
| S5 | **Divergence-vs-bias correlation** | per-arm S1 vs a scalar measure of that arm's Q'-source departure from Q'_true (e.g. mean daily volume ratio) | positive correlation ⇒ divergence tracks bias magnitude, not just arbitrary arm-to-arm noise |

**Headline: PASS iff S4 ≥ 3× AND S2 shows at least one arm's slope flipping
sign.** S5 is supporting evidence, not a hard bar (only 4 data points).

## 4. Components to build

1. **n-override eval-path seam** — new struct in `src/training/forward.rs`,
   same shape as `LeakanceOverride` (per-COMID normalized-value substitution
   before denormalization), scoped to `n` only. Eval-only; no `Backward` impl
   touched (repo invariant 4 untouched — this mirrors the leakance override's
   proven-safe pattern).
2. **Teacher mode** — reuses the chunked forward-eval loop
   (`probe_zeta_gradient --mode teacher` is the direct template): loads the
   consensus-geometry checkpoint, applies the n-override CSV, forces
   `streamflow = merit_dhbv2_UH_retrospective.ic`, runs `--backend cpu` over
   the full 1981/10/01-2010/09/30 window, writes synthetic gauge obs via the
   unmodified `obs_writer.rs`.
3. **Consensus-geometry script** (Python, `ddrs-py`) — runs `dump_parameters`
   (existing binary, `--backend cpu`) against each of the 4 real checkpoints'
   `epoch_5_mb_35/head`, loads the 4 resulting NetCDFs, computes per-COMID
   median `q_spatial`/`p_spatial`, writes a single consensus NetCDF/CSV.
4. **Truth-n generator script** (Python, `ddrs-py`) — computes the
   Leopold-Maddock CSV and the Gaussian-noise CSV from
   `merit_global_attributes_v2.nc`'s `log10_uparea` attribute. Zero repo
   risk.
5. **4 student configs** — each is the campaign's own existing config
   (`aorc2f_distributed_frozen_chunk1.yaml`, `aorc2f_lumped_frozen_chunk1.yaml`,
   `lstm_daily_frozen_chunk1.yaml`, `lstm_hourly_native.yaml`) with only
   `data_sources.observations` repointed to the synthetic obs store. Cold
   seed-42 init (no `experiment.checkpoint`), `params.sparse_solver: cpu`,
   `--backend cpu` at launch — matching this campaign's own convention.
6. **Analysis script** — `scripts/synthetic_n_recoverability_analysis.py`:
   loads the 4 students' `dump_parameters` outputs, the truth-n CSV, and the
   consensus-geometry NetCDF; computes S1-S5 exactly as §3 defines; prints a
   VERDICTS block; writes per-reach rows CSV.
7. **Findings report** — `docs/2026-07-22-synthetic-n-recoverability-findings.md`
   once run.

## 5. Execution sequence and compute

All CPU (`NdArray<f32>`, deterministic, `--backend cpu` everywhere per
explicit instruction).

1. **Consensus geometry**: 4 `dump_parameters` calls (fast, single forward
   pass each) + median script.
2. **Truth-n generation**: 2 scripts (Leopold-Maddock + Gaussian), no
   training required.
3. **Teacher pass** (Phase 1, Leopold-Maddock n): one full-window forward-eval
   over 1981/10/01-2010/09/30 → synthetic obs store.
4. **4 students in parallel** (Phase 1): full CONUS 5-epoch train-and-test,
   `--backend cpu`, cold seed-42, comparable wall-time to this campaign's own
   CPU arms (the lumped arm took ~8h26m end-to-end).
5. **Measurement**: `dump_parameters` on each student's final checkpoint +
   analysis script.
6. **Phase 2 (contingent)**: if S4 ≥ 3× in Phase 1, re-run the teacher with
   the Gaussian-noise n truth, then re-run only the 2 arms with the largest
   Phase-1 n-recovery error, then re-measure.

## 6. Concerns

1. **Disagg-head architecture is NOT shared between the consensus-geometry
   source (this campaign's 4 checkpoints, capacity-boosted frozen chunk1
   head) and the teacher's forward pass.** Why accepted: geometry
   (q_spatial/p_spatial) is independent of the disagg head's daily→hourly
   precip translation; the disagg head only affects Q' timing, not the main
   KAN head's attribute→n/q/p mapping. All 4 real checkpoints already share
   the SAME disagg-head architecture as each other (and as the students that
   will reuse it), so this only matters for the teacher's own forward pass
   (which needs *some* disagg head to translate daily Q'_true → hourly if the
   config calls for it) — using the same frozen capacity-boosted chunk1 head
   as the students is the natural, zero-new-code choice.
2. **Cold-start students could reintroduce the leakance experiment's ~130×
   windowed-objective noise floor.** Why judged low-risk here: that floor
   swamped a SMALL per-reach flux signal; a full Q'-source swap changes
   discharge by O(10-100%) at many reaches, well above the floor. Flagged as
   a risk to check in the findings (e.g. via each student's step-0 vs
   converged loss), not assumed away.
3. **8 full-CONUS training runs (4 arms × 2 truth-n fields) is expensive.**
   Mitigated by staging: Phase 2 only runs if Phase 1 shows a signal, and
   only on the 2 most-divergent arms, cutting Phase 2 cost by half.
4. **Consensus geometry is itself derived from real, noisy-observation-trained
   checkpoints**, not a "true" physical geometry. Why accepted: the
   experiment's target claim is about n divergence RELATIVE to a held-fixed
   geometry, not an absolute physical-truth claim about channel shape — using
   the real training's own consensus is the least-arbitrary fixed reference
   available, and ties the control directly to what real training already
   agrees on.

## 7. Testing

- **n-override injection test**: a small losing/gaining chain routed with an
  n-override CSV equals the same chain routed with n substituted manually
  into the head output — byte-identical (mirrors the existing
  `LeakanceOverride` injection test pattern).
- **Synthetic-obs roundtrip**: reuses the existing `obs_writer.rs` roundtrip
  test infra unchanged (no new obs-writer code).
- **Guard suites stay green**: `compare_ddr_sandbox` ABSOLUTE MATCH (nothing
  in `src/routing/`/`src/sparse.rs` touched). No autograd `Backward` impl
  touched (repo invariant 4).

## 8. Out of scope

- Observation noise / robustness (synthetic obs are noise-free by design —
  this tests parameter-recovery attribution, not real-world detectability).
- A hand-fit smooth trend for q_spatial/p_spatial truth (rejected in favor of
  the per-reach consensus median — see §1 decision log / brainstorming
  transcript).
- Any GPU runs.
- Promotion of any finding here into the standing, pre-registered
  equifinality campaign's own verdict — this experiment is a separate,
  non-registered synthetic control that INFORMS that campaign's open
  question; it does not amend its registered results.
