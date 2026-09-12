# `ddrs experiment adjoint` — Implementation Plan (PoC on the Juniata pair)

**Spec:** `research/specs/2026-09-03-ddrs-experiment-adjoint-design.md`
**Status (2026-09-04):** all nine tasks done; findings in `research/findings/2026-09-04-adjoint-influence-poc-findings.md`.

**Goal:** one `ddrs experiment adjoint` invocation on the pair
01563500 → 01567000 across the five trained arms, producing the four figure
families, with the finite-difference gate passing.

## Tasks

1. **`src/experiment/mod.rs`** — `ExperimentSpec` (serde), `ArmSpec`,
   `resolve_arm(ws, arm)` (latest dir checkpoint; refuse flat `.mpk`),
   `ExperimentRun::create(ws, name)` (ts dir), `ExperimentManifest`.
   → verify: unit tests for checkpoint selection + refusal.
2. **`src/experiment/adjoint/influence.rs`** — `InfluenceContext` (dataset,
   head, cfg per arm), `run_functional(...) -> Grad{T,N}`, functionals
   `Kernel{t0}`, `Volume`, `Residual{obs}`; `dist_to_gauge`.
   → verify: `tests/adjoint_influence.rs` gradcheck on the 4-reach chain.
3. **`src/experiment/adjoint/output.rs`** — netCDF + summary CSV writers.
4. **`src/experiment/adjoint/validate.rs`** — FD gate.
5. **`src/experiment/adjoint/mod.rs`** — anchors, windows, sweep loop,
   manifest fill.
6. **`src/cli/experiment.rs` + `Cmd::Experiment`** — wiring, tee, exit codes.
7. **`experiments/adjoint/{experiment.yaml,README.md,plots.py}`**.
8. **Run** on the pair, cpu; iterate until the gate passes and the figures
   render. Tier C gates (`cargo test --lib`, `compare_ddr_sandbox`).
9. **Findings doc** `research/findings/2026-09-04-adjoint-influence-poc-findings.md`;
   update `ddrs-dev` skill (new subcommand, cost numbers); commit.

## Concerns / assumptions

See spec §5–6. Plan-specific: `dataset.collate` single-gauge in Testing
mode is untested; if it fails, fall back to `collate_window` + selection.
