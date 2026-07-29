# Synthetic-n recoverability across real Q' sources — findings (INTERIM)

**Status: INTERIM — 1 of 4 arms complete.** The experiment was paused on
2026-07-29 after arm 1 (`distributed`) when the machine was reallocated to new
unit-catchment runs. The pre-registered S1–S5 verdicts (design spec §3) require
all 4 arms and are **not yet computable**. This doc records execution facts and
the arm-1 preview so nothing is lost; it must be finalized when the remaining
arms (`lumped`, `daily_lstm`, `hourly_lstm`) complete.

Spec: `docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md`
Plan: `docs/superpowers/plans/2026-07-22-synthetic-n-recoverability.md`
PR: https://github.com/taddyb/ddrs/pull/29

## What this tests

Whether learned Manning's n absorbs Q'-source bias while channel geometry
(q_spatial/p_spatial) does not — in a world where the truth is KNOWN. A teacher
(standard benchmark Q' store + prescribed Leopold-Maddock n + consensus
geometry) generates noise-free synthetic gauge observations; 4 students train
against those observations, each forced by one of the campaign's real Q'
stores. Recovered n/q/p (via `dump_parameters`) is compared to the truth donor.

## Execution notes

| Step | Outcome |
|---|---|
| Teacher (Task 4 Step 5) | First launch 2026-07-23 **OOM-killed** at chunk 1/30: the 365-day chunk peaks ~65 GB RSS on the 64,892-reach network (debugging-playbook T14). Fixed by new `--chunk-days` flag; relaunched 2026-07-27 with 180-day chunks (peak ~45 GB, ~4 GB at boundaries). Completed 59/59 chunks in ~14 h wall: synthetic obs for 2,365 gauges × 10,592 days from 1981-10-01, 94.3% finite (NaN-padded before 1981-10-02). |
| Donor parity gate (Task 1) | `tests/teacher_donor_override_parity.rs` passes: own-donor teacher run within 6e-5 m³/s of no-donor (tolerance 1e-3). |
| Student `distributed` | Training complete (5 epochs, `epoch_5_mb_35`, ~7 h CPU). Eval died at chunk 364/366 on a **transient** icechunk `object not found` (playbook T15 — store probes clean; divide-major chunking means the same objects were read 363 times before). Eval diagnostics lost; `recovered_distributed.nc` dumped from the completed checkpoint. |
| Students `lumped`/`daily_lstm`/`hourly_lstm` | Launched sequentially (serialized per user request); **stopped at user request 2026-07-29** during `lumped` epoch 1 (machine reallocated). Resume: `output/synthetic_n/run_students_sequential.sh lumped daily_lstm hourly_lstm`. |
| Original campaign eval caveat | The Jul 16 `distributed` campaign arm's own eval crashed at chunk 4/366 (GPU cubecl OOM panic), so the full 1995–2010 eval window had never been exercised against the aorc2f store before this experiment. |

## Arm-1 preview (distributed) — NOT a verdict

Computed over all 346,321 CONUS reaches vs `truth_leopold_maddock.nc`
(notebook: `output/synthetic_n/plots/synthetic_n_recovery_distributed.ipynb`):

| Quantity | Value |
|---|---|
| n median abs error | 0.0354 |
| corr(truth n, recovered n) | 0.736 |
| true n slope vs log10_uparea | −0.0421 |
| recovered n slope | −0.0193 (correct sign, attenuated ~54%) |
| q_spatial median abs error | 0.0274 |
| p_spatial median abs error | 5.95 |

Reading: under the distributed (aorc2f) forcing, the student recovers the
Leopold-Maddock downstream-decreasing n structure with the correct sign but
under-estimates its magnitude; roughly a third of reaches carry |n error| >
0.05. Whether this error varies systematically ACROSS Q' sources (S1/S4) and
whether any source flips the slope sign (S2) is exactly what the remaining 3
arms decide — no conclusion is drawn here.

## Required caveats (design spec §6 concern 1, §1 naming note)

1. **Disagg-head confound on S3.** Small/consistent geometry recovery error
   across arms is NOT independent proof that geometry is physically
   identifiable regardless of Q'-source bias: teacher and all 4 students share
   the SAME frozen capacity-boosted chunk1 disagg head, so a latent disagg-head
   effect on q/p would look identical to genuine geometry robustness. Any S3
   result must be reported as "consistent under a shared, frozen disagg head."

2. **Campaign-naming provenance.** This experiment's 4 arms
   (`aorc2f_distributed`/`aorc2f_lumped`/`daily_lstm`/`hourly_lstm`, from the
   2026-07-16 AORC2F/LSTM wave) are a DIFFERENT arm set from the
   pre-registered LSTM-equifinality campaign's R1/R2/R3 naming (the paper's
   `tab:arms`). The two numbering schemes must not bleed together in any table
   or cross-reference.

## Remaining work to finalize

1. Resume and complete the 3 remaining students + dumps (~24 h CPU sequential).
2. Run `scripts/synthetic_n_recoverability_analysis.py` → S1–S5 verdicts +
   `recoverability_rows.csv`.
3. Extend the recovery notebook to all 4 arms; decide Phase 2 (Gaussian-null
   confirmatory run) per plan Task 7 Step 5.
4. Replace this doc's INTERIM header with the final verdicts.
