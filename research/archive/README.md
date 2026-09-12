# Archive

This directory holds scripts and examples from closed research campaigns.
They are retired, not deleted, so the history and the exact commands stay
available if a campaign ever needs to be revisited.

## Admission rule

An artifact lands in this archive when the findings document or plan that
cites it records its campaign as closed, or when nothing in the repository
cites it at all.

This is **not** name-matching. A filename that merely resembles a closed
campaign's vocabulary is not grounds for archiving it: `examples/leak_probe.rs`
matches the word "leak" but is the autograd-tape-leak repro that
`.claude/skills/ddrs-dev/references/traps.md` names as a live discriminating
test, and `src/experiment/landscape/objective.rs` cites it directly. It stays
in `examples/`. Every artifact below was checked against its claimed closing
document (or checked to have no inbound reference anywhere) before it moved.

A second near-miss turned up during this pass, the same shape as
`leak_probe.rs`: `scripts/sp8_check_scatter.sh` and
`scripts/sp10_check_launches.sh` were proposed for archiving under the
SP-8/SP-10 spike campaigns, and their closing design documents do exist.
But `tests/sp8_v7_profile.rs` (an `#[ignore]`d test, run manually with
`cargo test --release --test sp8_v7_profile -- --ignored --nocapture`)
spawns `scripts/sp8_check_scatter.sh` directly via `Command::new`, and
`docs/book/reference/perf.md` documents `scripts/sp10_check_launches.sh`
as the V10 gate, with an explicit instruction to run it "when investigating
launch-count regressions or the SP-11 backward-capture work." Both are
still cited by live code or a still-current verification instruction, so
neither moved. They stay in `scripts/`.

## Warning: archived examples no longer build

`cargo check --examples` only compiles the four examples left in
`examples/`. Nothing under `research/archive/examples/` is compiled, tested,
or gated by CI any more. An archived example may not build against the
current API: routing, dataset, or KAN-head signatures can change after an
example is archived, and nothing will tell you. The "commit at which it last
built" column below is the last commit where the file is known to have
compiled and run, not a promise that it still does. Treat every archived
example as a historical artifact to read or resurrect with care, not as a
binary you can run today.

## Scripts (`research/archive/scripts/`)

| Artifact | Closing document | Commit last built |
|---|---|---|
| `leakance_diagnosis.py` | `research/findings/2026-07-02-leakance-diagnosis-findings.md` | `a575064` |
| `leakance_subset_analysis.py` | `research/findings/2026-07-01-leakance-hourly-experiment-handoff.md` | `6b02cc7` |
| `zeta_gradient_analysis.py` | `research/findings/2026-07-03-zeta-gradient-probe-findings.md` | `53b7e1b` |
| `zeta_probe_sites.py` | `research/findings/2026-07-03-zeta-gradient-probe-findings.md` | `42b38aa` |
| `floor_analysis.py` | `research/plans/2026-07-04-phase-b2-state-cache.md` | `ff7c59a` |
| `equif_convergence_analysis.py` | `research/findings/2026-07-07-lstm-equifinality-findings.md` | `5aaf237` |
| `run_equif_arms.sh` | `research/findings/2026-07-07-lstm-equifinality-findings.md` | `59b67c6` |
| `h5_h6_audit_analysis.py` | `research/findings/2026-07-09-h5-h6-equifinality-v2-findings.md` | `a4c910f` |
| `wave_comparison_plots.py` | `research/findings/2026-07-16-wave2-cross-wave-findings.md` | `5c28893` |
| `recoverability_analysis.py` | `research/findings/2026-07-04-synthetic-recoverability-findings.md` | `a218546` |
| `recoverability_sites.py` | `research/findings/2026-07-04-synthetic-recoverability-findings.md` | `82447b6` |
| `synthetic_n_recoverability_analysis.py` | `research/findings/2026-07-22-synthetic-n-recoverability-findings.md` | `5af6c4f` |
| `synthetic_n_consensus_geometry.py` | `research/plans/2026-07-22-synthetic-n-recoverability.md` | `310cc5d` |
| `synthetic_n_truth_fields.py` | `research/plans/2026-07-22-synthetic-n-recoverability.md` | `b5af06d` |
| `tau_sweep.py` | `research/findings/2026-08-06-tau-sweep-pilot-findings.md` | `54cd386` |
| `run_tau9_hourly_eval_chain.sh` | `research/findings/2026-08-06-tau-sweep-pilot-findings.md` | `b7cbaef` |
| `run_tau9_remaining.sh` | `research/findings/2026-08-06-tau-sweep-pilot-findings.md` | `b7cbaef` |
| `run_tau9_source_trains.sh` | `research/findings/2026-08-06-tau-sweep-pilot-findings.md` | `54cd386` |
| `run_tau_interp_arms.sh` | `research/findings/2026-08-06-tau-sweep-pilot-findings.md` | `5240fb1` |
| `run_tau_source_arms.sh` | `research/findings/2026-08-06-tau-sweep-pilot-findings.md` | `5240fb1` |
| `parameter_landscape.py` | none: no inbound reference anywhere in the repository. This is the H6-era equifinality instrument (R1/R2/R3 arms, `output/equif/`), not the current `experiments/landscape/` study. | `4e25bc7` |
| `generate_run_notebooks.py` | none: no inbound reference anywhere. | `5ee6cbc` |
| `hydrograph_comparison.py` | none: no inbound reference anywhere. | `5ee6cbc` |
| `merge_merit_basins.py` | none: no inbound reference anywhere. | `4bc8ea4` |

## Examples (`research/archive/examples/`)

| Artifact | Closing document | Commit last built |
|---|---|---|
| `disagg_boundary_verification.rs` | none beyond this archiving task's own plan; the only other citation was the `docs/book/usage/running.md` example table, now trimmed. | `334f0fe` |
| `disagg_precip_normalization_discriminator.rs` | none beyond the `src/data/dataset.rs` doc comment, repointed to `research/archive/examples/disagg_precip_normalization_discriminator.rs` in this same change. | `73a39c8` |
| `disagg_transfer_diagnostic.rs` | none beyond this archiving task's own plan. | `9178934` |
| `kan_disagg_mass_balance_real.rs` | `research/findings/2026-07-16-disagg-head-sensitivity-findings.md` | `26c3645` |
| `kan_disagg_real_storm_shift.rs` | `research/findings/2026-07-16-disagg-head-sensitivity-findings.md` (sibling probe of `kan_disagg_trained_sensitivity.rs`, same campaign) | `98f464c` |
| `kan_disagg_trained_sensitivity.rs` | `research/findings/2026-07-16-disagg-head-sensitivity-findings.md` | `26c3645` |
| `pretrain_disagg_capacity_storm_compare.rs` | none beyond this archiving task's own plan. | `73a39c8` |
| `pretrain_disagg_storm_compare.rs` | none beyond this archiving task's own plan. | `73a39c8` |
| `pretrain_disagg_verify.rs` | none beyond this archiving task's own plan. | `73a39c8` |
| `pretrain_disagg_window72_storm_compare.rs` | none beyond this archiving task's own plan. | `73a39c8` |
| `pretrain_reconciliation_check.rs` | none beyond the `src/pretrain/mod.rs` doc comment, repointed to `research/archive/examples/pretrain_reconciliation_check.rs` in this same change. | `73a39c8` |
| `save_random_kan.rs` | none: no inbound reference anywhere. | `beddc16` |
| `kan_sensitivity_sweep.rs` | none beyond this archiving task's own plan; also named in `kan_disagg_trained_sensitivity.rs`'s own doc comment as the synthetic-mechanism sibling, both archived together here. | `334f0fe` |

## One caveat worth reading before trusting the table above at face value

`synthetic_n_recoverability_analysis.py`'s closing document,
`research/findings/2026-07-22-synthetic-n-recoverability-findings.md`, calls
its own status "INTERIM, 1 of 4 arms complete" and says the campaign was
paused, not formally closed, on 2026-07-29. As of this archiving pass the
campaign has had no further activity for well over a month and the script
has no other inbound reference, so it is archived under the rule's second
clause as much as the first. If this campaign resumes, restore the script
from this directory rather than rewriting it.
