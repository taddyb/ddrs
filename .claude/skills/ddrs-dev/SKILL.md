---
name: ddrs-dev
description: Use for any ddrs building, coding, configuring, testing, running, or debugging task — build failures, config keys, which tests gate a change, launching train/eval jobs, resuming checkpoints, diagnosing a wrong or failed run, and research-status facts needed to avoid re-opening closed questions. Trigger on "build ddrs", "cargo test", "which tests do I run", "config key", "ddrs plan/run", "resume from checkpoint", "training is broken", "wrong result", "OOM", "is X still true". For plotting or interpreting eval output, use ddrs-eval-plots instead.
---

# ddrs development

BURN-0.21 Rust port of DDR's differentiable Muskingum-Cunge routing. Gradient-exact
against the Python reference at `~/projects/ddr/`. A KAN head emits per-reach
hydraulic parameters; a sparse triangular solve routes discharge over the MERIT
network.

**CLAUDE.md is already in your context and is the source of truth for**: the seven
invariants, CLI lifecycle, workspace layout, data sources, source groups, and
checkpoint resume. This skill does **not** repeat them. It carries what CLAUDE.md
does not: traps with their discriminating tests, the change→gate matrix, the full
config reference, test-authoring patterns, and the current research status.

## Route

| I need to… | Read |
|---|---|
| Fix a build failure (cmake, netcdf, fork resolution, fixtures) | `references/build-and-env.md` |
| Look up a config key, its default, or a load-time guard | `references/config.md` |
| Know which tests gate the change I just made | `references/testing.md` §Tier gates |
| Write a new gradcheck / parity / fixture test | `references/testing.md` §Authoring patterns |
| Diagnose a failed, hung, or wrong-result run | `references/traps.md` §Symptom → trap |
| Check whether a question is already settled | `references/research-status.md` |
| Build or change the training gauge CSV / population | `references/gauge-population.md` |
| Launch / watch / resume / audit a training run, or use any `ddrs` subcommand | skill `ddrs-run` |
| Plot or interpret eval output | skill `ddrs-eval-plots` |

## Contents

**This file** — Five costly facts · Change→gate table · Commands · Verifying a run ·
Maintenance

**`references/build-and-env.md`** (80 lines)
Hard prerequisites (cmake; CUDA toolkit is required even for CPU-only builds) ·
Fork pins (13 burn, 11 cubecl, rskan tag) + resolution failure modes · Fixtures
(V1 sandbox, KAN parity, the wrong-reference caveat) · Gitignored artifacts ·
Cargo features · Worktree gotchas

**`references/config.md`** (194 lines)
Top level · `data_sources:` (8 fields, adjacency rule) · `experiment:` (incl.
`optimizer`, grad-accum) → `experiment.loss:` (l1 / nnse-kge / kge / nse-batch) ·
`testing:` overlay (batch_size shifts meaning) · `kan_head:` →
`kan_head.disaggregation:` (**the real fields — `use_precip` does not exist**) ·
`params:` (incl. what `tau` actually is) → `parameter_ranges` → `attribute_minimums` ·
Load-time guards + their error substrings · Adding a routing parameter · Adding a
boolean flag · Enabling leakance

**`references/testing.md`** (127 lines)
Tier gates A/B/C/D with exact commands · What covers what (test → area map) ·
Acceptance thresholds · Authoring patterns: gradcheck (**ε depends on the parent's
nonlinearity**), parity (must be bidirectional), fixtures · Why the zeta_accum
headwater identity works · Checkpoint f16 drift

**`references/traps.md`** (212 lines)
Symptom → trap table · T1 stale binary · T2 DDR sandbox mismatch · T3 CUDA graphs
mask NaN · T4 phantom-zero baseline · T5 flat training loss · T6 GPU eval OOM that
never propagates · T7 silent kernel OOM on long CPU forwards · T8 transient icechunk
read · T9 `.ddrs/` beside the config · T10 `--checkpoint` differs per binary ·
Exit codes · Pre-flight checklist

**`references/research-status.md`** (213 lines)
Gauge-set definitions (2,365 vs 2,698 vs 3,211 vs 5,224) · Benchmarks + the KGE
claim restated · Closed campaigns: leakance NO-GO, selective equifinality H1–H6,
Q′-store waves, synthetic-n interim · **Do-not-use list** · Structural constants ·
Evidence standard · Doc conventions · Open questions

**`references/gauge-population.md`** (100 lines)
Regenerating `gages_2000_area_balanced.csv` (one command, seed 42, all-local
inputs) · Relative `DA_VALID` (`ABS_DIFF/DRAIN_SQKM ≤ 10%`) vs the scale-biased
absolute criterion · The filter funnel (coverage in both configured windows,
non-headwater subgraph) · Consequences of changing the population: baseline
cache invalidation, incomparable metrics, run-log verification string

## The five facts that cause the most wasted time

1. **Stale binary.** `cargo build` / `cargo run` do **not** update `~/.cargo/bin/ddrs`.
   The manifest stamps `git.sha` from `.git` at runtime, so a run *looks* current
   while a weeks-old binary executed. After any `src/` change: `cargo install --path .`.
   Self-check: checkpoints must be **directories** (`epoch_E_mb_M/head.mpk`); flat
   `epoch_E_mb_M.mpk` files mean a stale binary ran. Full incident: `references/traps.md` T1.

2. **`--workspace` is not optional when you pass `--config`.** The default is
   `Workspace::beside(config)`, so `--config config/experiments/x.yaml` silently
   creates `config/experiments/.ddrs/`. Always pass
   `--workspace /home/tbindas/projects/ddrs/.ddrs`.

3. **Use the right baseline number.** The CONUS summed-Q′ bar is
   **0.6781 NSE / 0.7172 KGE on 2,365 gauges**. The widely-copied `0.689 / 0.723`
   is a *global* MERIT number on a different 5,224-gauge network and must never be
   used as a CONUS bar. See `references/research-status.md`.

4. **CUDA graphs mask NaN.** `use_cuda_graphs: true` replays the captured graph and
   returns a stale finite loss when the real computation went NaN. Validate every
   new data path with `use_cuda_graphs: false` first.

5. **A gauge observes a network sum.** Any per-reach quantity whose only supervision
   is downstream discharge is structurally non-identifiable — a sum is not invertible
   for its addends. This is why leakance is CLOSED (NO-GO, 2026-07-06) and it
   generalizes to any new per-reach term you are tempted to add.

## Change → gate, in one table

Full commands and rationale in `references/testing.md`.

| You touched | Tier | Must pass |
|---|---|---|
| `src/routing/`, `src/geometry.rs`, `src/sparse/` | **A** | `--test ddr_sandbox_match` (machine-enforced invariant 1), `compare_ddr_sandbox` = ABSOLUTE MATCH (exits 1 on mismatch since 2026-09-02), `cargo test --lib`, `--test mmc`, `--test sparse_gradcheck`, all three leakance tests |
| `src/nn/`, `Cargo.toml` rskan tag | **B** | 4-test KAN fixture sweep, then Tier A |
| `src/config.rs`, `src/training/`, other `src/` | **C** | `cargo test --lib`, `cargo test`, `compare_ddr_sandbox` |
| `config/**/*.yaml` only | **D** | `ddrs plan --config … --workspace …` exits 0, no drift |
| `.github/workflows/ci.yml`, `.githooks/**` | — | push to a branch and let the PR run vet it; there is no local gate for workflow changes |
| `examples/juniata/**` (bundle, config, README) | — | `cargo test --test juniata_bundle` (never skips — bundle is committed), `cargo test --release --test juniata_acceptance` (end-to-end metric floors), and `ddrs --config examples/juniata/ddrs.yaml plan` exits 0 from the repo root |
| `src/adjacency/gridded.rs`, `resolve_or_build_gridded`, `examples/juniata_gridded/**`, the `Qr` axis-order sniff in `src/data/store/icechunk.rs` | **C** | `cargo test --test gridded_bundle` (never skips), `cargo test --test hourly_streamflow` (time-major fixture), `cargo test --release --test gridded_acceptance`, then Tier C |
| Plotting / analysis scripts only | — | no gate |
| `epochs`, `learning_rate`, `batch_size`, loss weights within documented ranges | — | no gate |

`mkdir -p output` before `compare_ddr_sandbox` — the example calls
`File::create("output/…")` with no `create_dir_all` and panics on a fresh clone.

## Commands you actually need

```bash
cargo install --path .                              # refresh the PATH binary — after every src/ change
mkdir -p output && cargo run --release --example compare_ddr_sandbox   # V1 gate
cargo test                                          # full suite, 2–5 min
cargo test --release --test juniata_acceptance      # end-to-end metric-floor acceptance (~20 s; skips in debug)
ddrs plan                                           # GPU probe + smoke + baseline (cached)
ddrs run --workflow train-and-test --backend cpu    # train then eval
ddrs run --workflow train --backend cpu --max-mini-batches 2   # mechanics smoke
ddrs show <run-id>; ddrs status; ddrs gc --keep 5 --keep-successful
ddrs --workspace .ddrs experiment adjoint --backend cpu   # paper study over trained runs (see below)
gh pr checks --watch                                # CI status for the current PR
```

**`ddrs run --workflow eval` does not work** — it returns
`"standalone --workflow eval needs a --from-run <run-id> flag"`, and `--from-run`
is unimplemented (`src/cli/run.rs:322`). Use `--workflow train-and-test`, or the
legacy `eval` binary against an existing checkpoint. **`ddrs init` is a dead stub**
(exits 2, `src/bin/ddrs.rs:167`); use `ddrs plan`. Both are still documented as
working in README.md and `docs/` — see `docs/2026-07-30-docs-and-skills-audit.md`.

## Verifying a run did what you think

Grep the run log — these strings are real (`src/data/dataset.rs`, `src/training/bootstrap.rs`):

```
streamflow resolution: Daily|Hourly
AORC precip store: N catchments, hourly …
gages_adjacency filter: kept X gauges (dropped Y missing, Z headwater)
warm start: loaded KAN head from …
warm start: no <path>.mpk — Adam starts cold
```

There is no `"precip loading"` string — two retired skills told you to grep for it.

## `ddrs experiment <name>` — paper studies over trained runs (2026-09-04)

A study is a checked-in bundle `experiments/<name>/experiment.yaml` (arms as
**run ids** under `.ddrs/runs/`, plus study parameters) and a Rust module under
`src/experiment/`. Arms resolve to the run's `config.yaml` snapshot and its
latest `checkpoints/epoch_E_mb_M/` (max by `(E, M)`; flat `.mpk` refused).
Output: `.ddrs/experiments/<name>/<UTC ts>/` with `manifest.json`, `run.log`,
per-arm outputs, and `figures/` from the bundle's `plots.py`. Training,
baselines, and `status`/`gc` are deliberately not integrated. Spec:
`docs/superpowers/specs/2026-09-03-ddrs-experiment-adjoint-design.md`.

**`adjoint`** (the only study so far): gradient of routed gauge discharge w.r.t.
hourly lateral inflow at every upstream reach — read from the existing routing
backward by lifting the inflow tensor as a `require_grad` leaf
(`src/experiment/adjoint/influence.rs`; no `Backward` impl touched, Tier C).
Three functionals: kernel `dQ_g(t0)/dq'` at high/low anchors, volume
`d[ΣQ_g]/dq'` (≈1 by mass conservation, the sanity gate), squared-error
sensitivity `d[mean(Q̄−obs)²]/dq'` (NOT the signed mean residual — its gradient
is independent of the observations). A finite-difference gate runs first and
aborts on > 5 % relative error. PoC on the Juniata pair 01563500 → 01567000:
gate 0.1–0.9 %, ~15 s per gauge-arm for daily stores (56 s hourly-lstm), 9
backwards per gauge-arm. Findings: `docs/2026-09-04-adjoint-influence-poc-findings.md`.
Gates: `cargo test --release --test adjoint_influence`, `cargo test --lib experiment`.
Figures: `~/projects/ddr/.venv/bin/python experiments/adjoint/plots.py <out dir>`.

## Juniata single-catchment sample (`examples/juniata/`)

The fastest full end-to-end exercise of the CLI, and the mirror of DDR's
`examples/juniata` (DeepGroundwater/ddr PR #193): one gauge (USGS 01567000,
8,657 km², 213 reaches), 8.9 MB committed bundle, no external stores or CUDA.
Run **from the repo root** (data paths in the config are repo-root-relative);
the workspace intentionally lands beside the config at `examples/juniata/.ddrs/`
(gitignored) — the one sanctioned exception to fact 2 above.

```bash
target/release/ddrs --config examples/juniata/ddrs.yaml plan
target/release/ddrs --config examples/juniata/ddrs.yaml run --workflow train-and-test --backend cpu
```

Verified 2026-08-19: `plan` baseline NSE 0.695 / KGE 0.819 (matches DDR's
Python readers to rounding — a live cross-implementation check); 30-epoch CPU
train-and-test finishes in ~21 s at routed NSE 0.790 / KGE 0.881 vs DDR-Python's
0.784 / 0.877 (residual = window-sampling RNG streams only; exact match is
impossible by construction; ddrs 4-seed spread NSE 0.790–0.800). Both examples
run the corrected physics — since 2026-08-19 `ddr_match` is DEPRECATED and
defaults to `false`, matching DDR post-#192. (Historical: with legacy physics
this gauge scored 0.840 / 0.913 — *better* here, but not the same model.)
Two deviations from DDR's bundle: `data/statistics/*.json` is **committed**
(ddrs never recomputes statistics), and `.gitignore` carries
`!examples/juniata/ddrs.yaml` so the example config survives the global
`ddrs.yaml` ignore. Regenerate the bundle in the ddr repo
(`extract_bundle.py`), then re-copy `data/` plus the generated statistics JSON.

## Gridded (ISIMIP DDM30) routing — `examples/juniata_gridded/` (2026-09-09)

ddrs consumes DDR PR #194's gridded products as a **data source**; it ports
none of the raster/regridding engine. `data_sources.gridded_network` points at
DDR's sub-reach adjacency zarr (`order`, `parent_cell`, `indices_0/1`,
`length_m`, `slope`) and `ddrs plan` relabels it into the existing subdivided
store layout (parent = cell, pieces = the cell's contiguous sub-reaches, gauge
read at the last piece) via `adjacency::gridded::GriddedNetwork` +
`cache::resolve_or_build_gridded`. Attributes and Q′ are keyed on the cell id,
so `parent_order` lookups and `pieces_per_row_divisor` (Q′/k) need no change.
The gauge CSV's `cell` column is a serde alias of `COMID`. Guards: exclusive
with `geospatial_fabric` and the explicit zarr pair; `params.subdivision.enabled`
is rejected with it. Spec:
`docs/superpowers/specs/2026-09-08-ddrs-gridded-routing-design.md`.

```bash
target/release/ddrs --config examples/juniata_gridded/ddrs.yaml plan
target/release/ddrs --config examples/juniata_gridded/ddrs.yaml run --workflow train-and-test --backend cpu
```

Verified 2026-09-09 (4 cells / 27 sub-reaches, 30 epochs, CPU, ~15 s): baseline
NSE 0.594 / KGE 0.672 (DDR 0.594 / 0.671); routed NSE 0.751 / KGE 0.730 at seed
42, seeds {42, 7, 123} span 0.744–0.758 / 0.728–0.732 (DDR 0.795 / 0.737 — RNG
windows plus ddrs's per-cell KAN head vs DDR's per-sub-reach `log10_uparea`).
CONUS: `config/sources/conus-gridded.yaml` + `config/experiments/gridded_conus.yaml`
(620 gauges from `scripts/snap_gridded_gauges.py`, DDR's population); `plan`
builds 5,198 cells → 91,867 sub-reaches in 0.6 s, baseline median NSE 0.354 /
KGE 0.505 over 1995-10-01..1997-09-30 (DDR's 0.390 includes its `da_ratio`
output correction, which ddrs does not apply).

Two things this work found, both fixed: (1) `GageSubgraph::upstream_comids`
returned one entry per piece on subdivided stores, so the summed-Q′ baseline
counted each divide `m` times — now deduplicated by id (also affected MERIT
`params.subdivision` baselines); (2) DDR's gridded Q′ stores are
`Qr(time, divide_id)`, and zarrs returns fill values for an out-of-range subset
instead of erroring, so the old fixed-axis reader silently read NaN — the
reader now sniffs the axis order (traps.md T11).

## Maintenance

Three skills live in this repo: this one (build / code / configure / test / debug),
`ddrs-run` (launch, watch, resume, and audit runs; the full `ddrs` command
reference), and `ddrs-eval-plots` (visualize and interpret run output). When a run,
eval, or export completes, update the relevant section here in the same session that
produced the knowledge — do not leave it only in a findings doc. If a rule here is
superseded, correct it in place with the new nuance rather than deleting it.

**The mdBook under `docs/` is now the canonical prose documentation** — it is a
strict superset of the deleted `.claude/references/` copies. The old
`regenerate-docs` skill was removed: its input contract pointed at
`.claude/references/*.md` frontmatter that no longer exists, its
`.regenerate-state.json` was never created, and its dataflow diagram published an
"MLP head" node that violates invariant 5. If you rebuild that mechanism, keep its
one genuinely good rule: **never invent a function signature, file path, or API
detail — if a claim is not backed by a cited source, flag it rather than writing it
into a chapter.** This audit found that rule violated repeatedly.

Every factual claim in this skill was verified against source on 2026-07-30. Claims
carrying a date are volatile; re-verify before citing externally. The audit that
produced this consolidation is `docs/2026-07-30-docs-and-skills-audit.md`.
