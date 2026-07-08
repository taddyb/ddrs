---
name: ddrs-dev
description: Use at the start of any ddrs task to route to the right sub-skill. Load this first when the task type is unclear, when onboarding to the project, or when multiple skill areas may apply.
---

# ddrs skill index

**Project:** Differentiable Muskingum-Cunge routing solver — BURN 0.21 Rust port of DDR (Python/PyTorch). Gradient-exact. Trains a KAN head to emit per-reach hydraulic parameters.

**Live research goal (as of 2026-07-06):** Selective-equifinality paper — train on 4 NH inflow sources (daily-lstm, hourly-lstm, dHBV2.0-lumped, dHBV2.0-UH), compare convergence of geometry (identifiable) vs Manning's n (equifinal). Paper draft: `/home/tbindas/projects/ddr_equifinality/paper.tex`.

---

## Route to the right skill

| I need to… | Use skill |
|---|---|
| Understand what changes are safe to merge | `ddrs-change-control` |
| Debug a failed / wrong-result run | `ddrs-debugging-playbook` |
| Check whether a problem was already solved | `ddrs-failure-archaeology` |
| Understand a load-bearing design decision | `ddrs-architecture-contract` |
| Understand hydrology concepts (NSE/KGE, routing, equifinality) | `ddrs-hydrology-reference` |
| Add, change, or audit a YAML config key | `ddrs-config-and-flags` |
| Set up or rebuild the dev environment | `ddrs-build-and-env` |
| Install the CLI, run a train/eval job, resume a checkpoint | `ddrs-run-and-operate` |
| Measure a result, interpret zeta/gradient/metric diagnostics | `ddrs-diagnostics-and-tooling` |
| Decide whether a change is safe / what tests to run | `ddrs-validation-and-qa` |
| Write a findings doc, spec, plan, or paper section | `ddrs-docs-and-writing` |
| Make an external claim about novelty or results | `ddrs-external-positioning` |
| Plan or execute the selective-equifinality experiment | `ddrs-identifiability-campaign` |
| Verify gradient correctness / design a controlled ablation | `ddrs-proof-and-analysis-toolkit` |
| Identify the next high-value research direction | `ddrs-research-frontier` |
| Design an experiment / enforce evidence standards | `ddrs-research-methodology` |

---

## Critical facts every agent must know

- **Stale binary trap:** `cargo build` does NOT update `~/.cargo/bin/ddrs`. After any `src/` change: `cargo install --path .`
- **V1 gate:** `cargo run --release --example compare_ddr_sandbox` must print `ABSOLUTE MATCH` (max abs < 1e-3 m³/s) after any change to `src/routing/`, `src/geometry.rs`, or `src/sparse.rs`.
- **CUDA graphs mask NaN:** validate new forward paths with `use_cuda_graphs: false` before enabling graphs.
- **Leakance is CLOSED (NO-GO, 2026-07-06):** do not re-open without reading `docs/2026-07-06-leakance-nogo-scientific-summary.md`.
- **Best CONUS result (2026-06-23):** NSE 0.715 / KGE 0.711 (precip-driven disagg + L1, 2,365 gauges). KGE does NOT beat the summed-Q' baseline (0.7172) in any config as of 2026-07-06.

---

## Provenance and maintenance

Re-verify routing after any new master merge:
```bash
git log --oneline origin/master..HEAD   # commits ahead
cargo run --release --example compare_ddr_sandbox  # V1 gate
```
Update this index when a new skill is added to `.claude/skills/`.
