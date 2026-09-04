# `ddrs experiment` + the adjoint influence-map study — Design

**Date:** 2026-09-03
**Status:** Sections 1–2 approved in brainstorming; sections 3–4 drafted for
review. Proof of concept authorized on the Juniata pair before full review
(user directive 2026-09-03, overnight) — **implemented**, see
`docs/2026-09-04-adjoint-influence-poc-findings.md`. One correction from the
PoC (§2.3): the residual functional is the **squared error**, not the mean
signed residual — the latter is linear in the prediction, so the observations
cancel out of its gradient.
**Paper:** `~/papers/ddr_equifinality/paper.tex` (AGU H069 abstract;
"Revisiting Beven's Equifinality and Uncertainty Thesis").
**Related:** `docs/superpowers/specs/2026-08-06-ddr-equifinality-paper-scope-design.md`,
`docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md` (the
`lift_leaf` pattern this reuses).

## 0. Purpose

Two things, in one PR:

1. **`ddrs experiment <name>`** — a new CLI entry point that runs a *paper
   study* over already-trained runs, from a checked-in bundle of
   pre-defined configs, and writes a self-describing output directory. It
   replaces the pattern of adding a seventh `--mode` to the 3,133-line
   `probe_zeta_gradient` binary plus a shell loop.
2. **The first study, `adjoint`** — the adjoint influence map: the gradient
   of routed gauge discharge with respect to *lateral inflow* at every
   upstream reach and hour, for several trained arms that differ only in
   their inflow source. It answers, per gauge and per arm, how an error
   injected at reach *i* on day *d* reaches the gauge (the propagation
   kernel), how much of a reach's inflow volume the gauge sees (the
   mass-conservation check), and which reaches the gauge's residual bias is
   attributed to.

The scientific question is the one recorded in memory
`research-direction-bias-propagation`: how lateral-inflow bias propagates
upstream→downstream through a large network, across inflow sources.
Leakance is out of scope.

## 1. The `ddrs experiment` framework — APPROVED

A study is a checked-in **bundle** (data) plus a Rust **module** (code).
Nothing in the bundle is executable.

```
experiments/                          # tracked, repo root
└── adjoint/
    ├── experiment.yaml               # which study, which arms, study parameters
    ├── README.md                     # question, how to run, what it writes
    └── plots.py                      # figure generator (section 3)

src/experiment/
├── mod.rs                            # ExperimentSpec load+validate; Arm resolution;
│                                     # output dir + manifest
└── adjoint/                          # the first study (section 2)

src/cli/experiment.rs                 # `ddrs experiment <name>` wiring
```

`experiment.yaml`:

```yaml
study: adjoint
arms:
  - { name: daily-lstm,   run: 2026-08-09T12-05-08Z-train-and-test }
  - { name: hourly-lstm,  run: 2026-08-09T14-55-05Z-train-and-test }
  - { name: uh-retro,     run: 2026-08-09T09-30-39Z-train-and-test }
  - { name: dhbv2-lumped, run: 2026-08-09T22-13-43Z-train-and-test }
  - { name: dhbv2-dist,   run: 2026-08-17T02-03-02Z-train-and-test }
adjoint:
  gauges:
    pairs: [["01563500", "01567000"]]   # [upstream, downstream]; PoC scope
  window_days: 90
  anchors: { high: 2, low: 2 }
  lag_days: 30
  water_year: 2000                       # volume + residual windows
  functionals: [kernel, volume, residual]
```

**Arm resolution.** An arm is resolved from its run id under
`<workspace>/runs/<run>/`: the config is that run's `config.yaml` snapshot;
the checkpoint is the latest `checkpoints/epoch_E_mb_M/` directory (max by
`(E, M)`). Resolution fails loudly if the run, the snapshot, or a directory
checkpoint is missing. Flat `epoch_E_mb_M.mpk` files are **refused** (stale
binary, CLAUDE.md). `manifest.json` is *not* required — a run whose eval
phase was resumed elsewhere still has a valid config + checkpoint.

**Invocation and outputs.**

```
ddrs --workspace .ddrs experiment adjoint [--backend cpu] [--arms a,b]
                                          [--max-gauges N] [--skip-validate]
                                          [--bundle experiments/adjoint]
→ .ddrs/experiments/adjoint/<UTC ts>/
    manifest.json     experiment.yaml copy · resolved arms (run id, checkpoint,
                      config path) · git sha/dirty · gauge list · validation
                      result · wall time
    run.log           fd-level tee (cli::tee), same as `ddrs run`
    <arm>/gauges/<staid>.nc   study outputs (section 2)
    <arm>/summary.csv
    figures/          written by plots.py (section 3)
```

`--backend` defaults to **cpu** (standing rule for diagnostics). `--bundle`
defaults to `experiments/<name>` relative to the current directory. The
subcommand uses the same `Workspace` as `run`, so `--workspace` semantics
are unchanged.

**Out of scope:** training arms, source-group switching, baselines,
`ddrs status`/`gc` integration (the experiments directory is manually
managed; README says so).

## 2. The adjoint study (`src/experiment/adjoint/`) — APPROVED

Four units.

### 2.1 `gauges.rs` — population

**Full study (deferred):** a gauge is selected when its GAGES-II CLASS is
`Ref` (read from `/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.dbf`
via the `dbase` crate) and at least one other training gauge's COMID lies in
its subgraph. **PoC:** the `pairs:` list is explicit; every listed staid must
exist in the arm's dataset (all arms share
`gages_2000_area_balanced.csv`, asserted at resolution). Every arm runs the
same list, so cross-arm comparisons are population-matched by construction.
Writes `gauges.csv` (staid, comid, role upstream|downstream, pair id).

### 2.2 `influence.rs` — one gauge, one arm, one functional

1. Open the arm's dataset with the run config in **Testing** mode (eval
   window 1995/10–2010/09; `sparse_solver: cpu`, `use_cuda_graphs: false`
   under the cpu backend).
2. Load the KAN head from the checkpoint (`kan_config(...).init` +
   `load_kan_head(head_base(ckpt))`). `use_leakance` must be `false`.
3. `dataset.collate(&[staid], &RhoWindow{..})` for a single gauge → its own
   subgraph over a `window_days` hourly window (`(window_days − 1)·24` h).
4. Mirror `forward_eval_core`: head → `gather_params_to_subreaches` →
   `n`, `q_spatial`, `p_spatial`, `x_storage`; hourly inflow via the disagg
   head when the head carries one, else the flat repeat-24 `q_prime`.
5. **Lift the hourly inflow as a gradient leaf**:
   `Tensor::<Autodiff<I>,2>::from_inner(q_hourly).require_grad()` (the 2-D
   analogue of `probe::lift_leaf`). Parameters are wrapped with
   `from_inner` without grad.
6. `setup_inputs(..., carry_state=false, initial_state=None)` (hotstart
   heuristic, as training), `forward()` → `(N, T)`, `scatter_add_by_group`
   → gauge series `(1, T)`.
7. Evaluate the functional as a scalar, `backward()`, read
   `q_leaf.grad(&grads)` → `(T, N)`. The first `warmup` days are excluded
   from every functional.

Nothing in `src/routing/`, `src/sparse/`, or any `Backward` impl changes.
The gradient parents already exist (`TimestepOp` parent 5 at
`src/routing/mmc_op.rs:895`; `CsrSolveOp` parent `b` at
`src/sparse/mod.rs:456`); they were simply never attached to a leaf.

### 2.3 Functionals, windows, and reductions

| Functional | Scalar | Windows | Stored per reach | Also stored |
|---|---|---|---|---|
| **kernel** | `Q_g(t0)`, `t0` = noon of the anchor day | one per anchor: `window_days` ending 7 days after the anchor | `kernel(anchor, reach, lag_day)` = Σ over the 24 source hours at that daily lag, lag 0..`lag_days` | `kernel_hourly(anchor, lag_hour)` = Σ over reaches, lag 0..`24·lag_days`; `anchor_date`, `anchor_obs` |
| **volume** | `Σ_{t ≥ warmup} Q_g(t)` | 1 fixed (first seasonal window) | `volume_sens(reach)` = mean over source hours in `[warmup·24, T − 7·24)` (tail dropped: truncation) | `volume_profile(hour)` = mean over reaches |
| **residual** | `mean_{valid days d ≥ warmup} (Q̄_g(d) − obs(d))²`, `Q̄` via `tau_trim_and_downsample(cfg.params.tau)`, obs day `d` ↔ pooled day `d` (2026-08-08 convention). Gradient `(2/n)Σ_d (Q̄_d − obs_d)·dQ̄_d/dq'`; positive ⇒ the reach's inflow arrives when the gauge over-predicts | 4 seasonal windows of the configured water year (starts +0, +92, +182, +273 days from Oct 1) | `residual_attr(reach)` = mean over windows of the time-mean sensitivity over `[warmup·24, T − 7·24)` | `residual_gauge_mean(window)`, `residual_gauge_mse(window)`, `residual_n_valid(window)` |

**Anchors** are chosen from the gauge's observed series over the eval axis:
the `high` highest and `low` lowest strictly-positive valid days, each at
least 60 days from every other anchor, and positioned so the window fits the
axis. **Nine backwards per gauge-arm** (4 kernel + 1 volume + 4 residual).

Per-reach coordinates written alongside: `COMID`, `dist_to_gauge_m`
(along-channel distance from the reach's outlet to the gauge outlet,
computed on the batch adjacency: `dist[gauge]=0`,
`dist[upstream] = dist[downstream] + length_m[downstream]`), and
`q_prime_mean` (window-mean hourly inflow, m³/s) as the size proxy.

### 2.4 `output.rs` — schema

One netCDF per gauge per arm at `<arm>/gauges/<staid>.nc`, dimensions
`reach`, `anchor`, `lag_day`, `lag_hour`, `window`, `hour`. Global attrs:
`staid`, `arm`, `run_id`, `checkpoint`, `window_days`, `tau`, `warmup`,
functional definitions verbatim from the table above.

`<arm>/summary.csv`, one row per (gauge, reach): `arm, staid, comid,
dist_to_gauge_m, q_prime_mean, volume_sens, residual_attr, kernel_mass,
kernel_mean_lag_days` (kernel scalars averaged over anchors; also written per
anchor as `kernel_mass_a{k}`).

The **inherited-versus-local** decomposition needs nothing new: the
downstream gauge's `volume_sens` at the upstream gauge's COMID is the
transfer coefficient, and the upstream gauge's own `residual_gauge_mean` is
the inherited bias source.

### 2.5 `validate.rs` — the finite-difference gate (approach C)

Runs before the sweep on the first gauge and first arm, first anchor window.
Perturbs inflow by +5 % of the reach's window-mean over all source hours in
`[t0 − 24·lag_days, t0)` at three reaches — the gauge reach, the reach at
median `dist_to_gauge_m`, and the farthest headwater — reruns the forward,
and compares `ΔQ_g(t0)` against `Σ_t' grad[t', i]·δ`. Passes when the
relative error is under 5 % at all three (ε is loose because celerity depends
on discharge). Result goes into the manifest; failure aborts with exit 1.
`--skip-validate` for reruns.

## 3. Figures (`experiments/adjoint/plots.py`) — DRAFT, not yet reviewed

Run under the DDR venv (`~/projects/ddr/.venv/bin/python`, has xarray,
netCDF4, matplotlib, geopandas). Reads only the study output directory and
writes to `<out>/figures/`. Four families, as chosen in brainstorming:

1. **Kernel by lag, per arm** — `kernel_hourly` for each anchor (high vs
   low flow panels), one line per arm; companion scatter of per-reach
   `kernel_mass` and `kernel_mean_lag_days` vs `dist_to_gauge_m`.
2. **Influence map** — per arm, reach polygons from the pfaf-7 MERIT
   catchment shapefile (`/mnt/ssd1/data/merit/cat_pfaf_7_…shp`) filtered
   to the gauge's COMIDs, colored by `residual_attr` (diverging) and
   `volume_sens` (sequential). Gauges marked.
3. **Inherited vs local bias** — for each pair and window: downstream
   `residual_gauge_mean`, the inherited part
   (upstream `residual_gauge_mean` × downstream `volume_sens` at the
   upstream COMID), and the remainder; grouped bars per arm.
4. **Volume-sensitivity check** — histogram of `volume_sens` per arm with
   the 1.0 line; `volume_sens` vs `dist_to_gauge_m`.

Figures are PNG; the notebook form (ddrs-eval-plots convention) is a
follow-up once the figures stabilize.

## 4. Testing — DRAFT, not yet reviewed

| Layer | Test | Gate |
|---|---|---|
| lifted-leaf gradient | `tests/adjoint_influence.rs`: 4-reach linear chain (mock config), `Q_outlet(t0)` gradient wrt the lifted `q'` leaf vs central finite difference, ε=1e-2 on q', rel tol 1e-2 | must pass; proves the leaf attaches to the existing parents |
| spec loading / arm resolution | unit tests in `src/experiment/mod.rs`: latest-checkpoint selection, flat `.mpk` refusal, missing run error | `cargo test --lib` |
| FD gate on real data | `validate.rs` (section 2.5) | recorded in manifest |
| Tier C (other `src/`) | `cargo test --lib`, `cargo test`, `compare_ddr_sandbox` ABSOLUTE MATCH | must pass before merge |

## 5. Concerns for the user

- **Cost scales as 9 backwards × gauges × arms.** PoC is 2 gauges × 5
  arms = 90 backwards over a 5,300–8,700 km² subgraph on CPU. Each is a
  2,136-step tape; expect minutes each, so hours total. The full nested-Ref
  population (hundreds of gauges) is a multi-day CPU job or needs the GPU
  path; measure on the PoC before promising the population run.
- **Hotstart transient.** Each window cold-starts from the hotstart
  heuristic. The kernel anchor sits 83 days in, so the transient is long
  gone, but `volume_sens` and `residual_attr` exclude only the 5 warmup
  days. If the transient is longer for large basins (the floor work measured
  this), the first days bias the residual functional. Mitigation: the
  reductions drop `warmup` days; the floor curves say what warmup to use.
- **The kernel is state-dependent.** Celerity depends on discharge, so the
  high-flow and low-flow anchors will differ. That is a feature (it is what
  the arms' learned parameters change), but a reader may mistake it for
  noise. Report both.
- **Distance is along-channel, not travel time.** Comparing kernels by
  distance across arms ignores that arms learn different celerities;
  `kernel_mean_lag_days` is the travel-time view.
- **Observations at daily resolution, kernel at hourly.** The residual
  functional uses the same tau/pool convention training used, so its
  attribution inherits any residual day-boundary misalignment; tau=9 is
  considered settled.
- **Double-routing confound on the dHBV2 distributed arm** (scope design
  B4) is unresolved; its kernel may reflect a pre-routed store.

## 6. Assumptions

- The August 9–17 epoch-9 runs are the current trained arms for the
  five stores (recorded in `experiment.yaml`; the manifest records the
  checkpoint actually used). If they are superseded, edit the bundle.
- Mass conservation of the gauge prediction (`tests/gauge_mass_conservation.rs`)
  means `volume_sens ≈ 1` upstream; deviations are truncation, the
  `discharge` clamp floor, and f32.
- `dataset.collate` works on a single-gauge batch in Testing mode. If it
  does not (e.g. the training sampler assumes ≥ 2 gauges), the fallback is
  `collate_window` on the all-gauges network with per-gauge selection, at
  higher cost.
- Both PoC gauges are in `gages_2000_area_balanced.csv` (verified
  2026-09-03: 01563500 Mapleton Depot 5,262 km²; 01567000 Newport 8,657 km²).
