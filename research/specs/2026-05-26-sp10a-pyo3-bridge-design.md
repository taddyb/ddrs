# SP-10a — PyO3 inference bridge (`ddrs-py`)

**Date:** 2026-05-26
**Branch:** `python-hooks`
**Status:** Design — awaiting review

Parent epic SP-10 ("Python interop layer") decomposes into:

- **10a (this spec):** Rust↔Python FFI crate exposing checkpoint loading, MLP
  inference, denormalization, and an optional full forward pass.
- 10b (future): Pure-Python `ddrs` package + `xarray` results reader + example
  notebooks.
- 10c (future): Map visualization over MERIT catchments.

10a is the foundation; 10b/10c import from it.

## Goal

Make the trained MLP and the MC forward solver callable from Python so a data
scientist can — from a notebook on a CPU laptop — load a `.mpk` checkpoint,
hand it a batch of MERIT attributes, and get back denormalized spatial
parameters `(n, q_spatial, p_spatial)` per COMID. As a stretch within the same
crate, expose the full MC forward so a single-batch hydrograph can be produced
end-to-end without invoking the `eval` CLI.

The Rust `ddrs` crate stays Python-agnostic. The bridge is a separate crate.

## Reference workflows in DDR

The Python source the bridge is reproducing (and the configs that drive it)
live at `~/projects/ddr/config/`:

| DDR config | DDR mode | What it does | ddrs analog |
|---|---|---|---|
| `merit_training_config.yaml` | training | Gauge-subgraph training, per-gauge loss | mirrored verbatim by `config/merit_training.yaml`; drives `src/bin/train.rs` |
| `merit_testing_config.yaml` | testing | Per-gauge eval on a held-out window, writes a predictions zarr | `src/bin/eval.rs` + the testing-overlay block in `config/merit_training.yaml` |
| `merit_geometry_config.yaml` | routing | **CONUS-wide inference** — loads a pretrained checkpoint, runs MLP over every MERIT catchment's attributes, produces a per-COMID parameter field. No gauges. No training. | **This is the workflow 10a unlocks from Python.** No CLI equivalent exists in ddrs yet — the bridge fills that gap. |
| `merit_plots_config.yaml` | routing | DDR's plotting harness for hydrographs + maps | reference for what 10b/10c need to render |

`merit_geometry_config.yaml` is the most important of these for SP-10a — it
is exactly the "load checkpoint → run inference over all COMIDs → get a
spatial parameter field" call we want exposed to Python. Its data path is
`data_sources.conus_adjacency` (the 346k-reach `merit_conus_adjacency.zarr`),
not `gages_adjacency`. The bridge mirrors that distinction — see
`run_inference_over_conus` below.

Its parameter block also pins the concrete bounds the bridge must reproduce:

```yaml
params:
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
  log_space_parameters: [p_spatial]
```

Note: `config/merit_training.yaml` lists `n` in `log_space_parameters`;
`merit_geometry_config.yaml` lists `p_spatial`. The bridge does not hard-code
either set — it reads from whatever YAML the user passes. This is the same
flexibility `denormalize` already has in `src/routing/utils.rs`.

A reference DDR checkpoint exists at
`~/projects/ddr/examples/merit/ddr-v0.5.2-merit-geometry-weights.pt`. It is
**not directly loadable** by the bridge (different runtime, KAN not MLP) but
is useful as a cross-validation target: running ddrs's MLP checkpoint and
DDR's KAN checkpoint over the same attrs batch should give parameter fields
of the same shape and broadly similar geography. Stretch goal for 10b's
notebook examples; not a 10a deliverable.

## Non-goals

- GPU backends from Python. CPU (`NdArray`) only in 10a.
- Training from Python. Read-only inference path.
- Reading `.mpk` files outside of BURN. Checkpoints flow through the bridge.
- Re-exposing every Rust type. Only the four entry points listed below.
- Distribution as a published wheel. Local `maturin develop` workflow only.

## Public API surface

Five functions, one struct. Everything else stays Rust-private.

```python
import ddrs_py as dd
import numpy as np

# 1. Load checkpoint + config (config provides MlpConfig + parameter bounds)
model = dd.load_mlp(
    checkpoint="output/saved_models/epoch_1_mb_35",
    config_path="config/merit_training.yaml",
)
# returns dd.PyMlp wrapping Mlp<NdArray>

# 2. Inference on an arbitrary attrs batch — returns parameters in [0,1]
attrs = np.load(...)                          # shape (R, F=10), float32
raw = model.forward(attrs)                    # dict[str, np.ndarray (R,)]
#   keys per merit_training.yaml's `mlp.learnable_parameters`:
#     "n", "q_spatial", "p_spatial"

# 3. Denormalize to physical units. Bounds come from the config; callers can
#    pass them by hand or use the helper below.
n_physical = dd.denormalize(raw["n"], bounds=(0.015, 0.25), log_space=True)

bounds = dd.parameter_bounds("config/merit_training.yaml")
# bounds == {"n": ((0.015, 0.25), True),
#            "q_spatial": ((0.0, 1.0), False),
#            "p_spatial": ((1.0, 200.0), False)}

# 4. CONUS-wide inference (parameter map for SP-10c).
#    Mirrors DDR's `merit_geometry_config.yaml` workflow: load every COMID's
#    attributes from the netcdf, run MLP, return one row per COMID.
params_per_comid = dd.run_inference_over_conus(
    attrs_nc="/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc",
    conus_adjacency_zarr="/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr",
    checkpoint="output/saved_models/epoch_1_mb_35",
    config_path="config/merit_training.yaml",
)
# returns dict[str, np.ndarray]:
#   "comid":      (N,) int64
#   "n":          (N,) float32 — physical units, already denormalized
#   "q_spatial":  (N,) float32
#   "p_spatial":  (N,) float32

# 5. (stretch) End-to-end MC forward for one gauge subgraph.
#    Useful for "what does my model predict at gauge X for this window?" in
#    a notebook, without spinning up `eval`.
discharge = dd.run_forward(
    gages_adjacency_zarr="/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr",
    gauge_id="01013500",
    attrs=attrs,                # (R, F) for the subgraph
    forcing=forcing_np,         # (T, R) hourly Q' per reach
    checkpoint="output/saved_models/epoch_1_mb_35",
    config_path="config/merit_training.yaml",
)
# returns np.ndarray, shape (T, R), float32 m³/s
```

Notes on the shapes and the F=10 contract:

- `attrs`: `np.float32`, `(R, F)`. F is fixed at **10** by
  `mlp.input_var_names` in `config/merit_training.yaml`:
  `SoilGrids1km_clay, aridity, meanelevation, meanP, NDVI, meanslope,
  log10_uparea, SoilGrids1km_sand, ETPOT_Hargr, Porosity`. The bridge
  validates F against `Config.mlp.input_var_names.len()` and raises
  `ValueError` on mismatch.
- `forward` returns a dict keyed by `mlp.learnable_parameters` (currently
  `n, q_spatial, p_spatial`).
- `denormalize` mirrors `routing::utils::denormalize`. Always takes a 1-D
  `np.float32`, returns the same shape.
- `parameter_bounds` reads `params.parameter_ranges` and
  `params.log_space_parameters` from the YAML. This is the same data
  consumed by `denormalize` calls inside `MuskingumCunge::setup_inputs`.
- `run_inference_over_conus` is the workflow gap noted in the table above —
  DDR has it as a config + CLI invocation; ddrs gets it as a single Python
  call. This is the function that powers the SP-10c choropleth.

## Crate layout

```
ddrs-py/                       # new sibling crate, NOT a workspace member of ddrs
├── Cargo.toml                 # crate-type = ["cdylib"], pyo3 0.21, ddrs path-dep
├── pyproject.toml             # maturin build backend, package name "ddrs_py"
├── README.md                  # vendored-cubecl warning + setup steps
├── src/
│   └── lib.rs                 # #[pymodule] fn ddrs_py(...) — all 4 fns + PyMlp
└── tests/
    └── smoke.py               # uv-run pytest, ~3 assertions
```

`ddrs-py` is a separate crate (not in `ddrs`'s workspace) because:

- Keeps PyO3 + maturin out of the routing crate's dependency closure. Anyone
  building `ddrs` for training doesn't pay for Python.
- Lets `ddrs-py` have its own feature flags and Cargo lockfile.
- BURN backend choice is forced (NdArray) without polluting `ddrs`'s
  generic-over-`Backend` design.

## Build + install workflow

```bash
cd ddrs-py
uv venv
uv pip install maturin pytest numpy
uv run maturin develop --release   # builds, installs into the venv
uv run pytest tests/smoke.py
```

The release flag matters — debug PyO3 + BURN is unusably slow for any
real-sized attrs batch.

## What changes in the existing `ddrs` crate

Minimal. The audit:

| Item | Current visibility | Needed |
|---|---|---|
| `nn::mlp::Mlp` | `pub` | ok |
| `nn::mlp::Mlp::forward` | `pub` | ok |
| `nn::mlp::Mlp::learnable_parameters` | `pub` | ok |
| `training::checkpoint::load_mlp` | `pub` | ok |
| `config::Config::from_yaml_file` | `pub` | ok |
| `config::ParameterRanges` fields | check | likely pub already; verify |
| `routing::utils::denormalize` | `pub` | ok |
| `data::store::zarr::GagesAdjacencyStore` | pub | ok, needed for `run_forward` |
| `routing::mmc::MuskingumCunge::setup_inputs` / `forward` | `pub` | ok |

Expected: 0–2 small `pub` widenings, no behavioral changes. The 5-reach
sandbox regression test stays the single source of truth — if any of these
flips break it, we revert and rework.

## Backend choice

NdArray for 10a. Concrete reasoning:

- A notebook on a CPU laptop is the target workflow.
- CUDA from Python adds: `burn-cuda` build complexity, the vendored
  cubecl/burn forks (already documented as fragile), CUDA driver assumptions
  in the wheel. None of that is worth it for interpretation.
- If someone *does* want GPU inference later, add a `cuda` Cargo feature on
  `ddrs-py` that swaps the backend and ships a second wheel. Defer.

## Testing

Two layers.

1. **Rust unit tests inside `ddrs-py`** — round-trip: build a small synthetic
   `Mlp<NdArray>`, save it, load it through the bridge, run inference on a
   fixed-seed attrs tensor, assert shape + range `[0, 1]`.

2. **Python smoke test (`tests/smoke.py`)** — `uv run pytest`:
   - Import succeeds.
   - `load_mlp` on a real checkpoint from `output/saved_models/` returns a
     `PyMlp`.
   - `forward` on a synthetic `(10, F)` attrs batch returns a dict whose
     values are 1-D `np.float32` of length 10 with all values in `[0, 1]`.
   - `denormalize([0.0, 1.0], (0.01, 0.3), log_space=True)` matches the Rust
     `denormalize` output to within 1e-6.

10b/10c add notebook-level integration tests; 10a does not.

CI: a follow-up will add a `cargo test -p ddrs-py` + smoke `pytest` job. Out
of scope for the design but flagged so it doesn't get forgotten.

## Concerns (per CLAUDE.md planning rules)

- **PyO3 ↔ BURN tensor conversion.** `Tensor<NdArray, 2>` has to be built from
  a `&numpy::PyArray2<f32>` and converted back. Plan: `to_data()` /
  `TensorData::from` on the way in; `into_data().to_vec()` on the way out,
  reshape via `numpy`. Straightforward but the BURN 0.21 API is the same
  surface that bit us in SP-6/SP-7 — pin the snippet in the design before
  implementation starts.
- **Checkpoint config drift.** `load_mlp` needs the `MlpConfig` that produced
  the checkpoint. Currently `eval.rs` reads it from `Config.mlp` in the YAML.
  If a notebook user points at a checkpoint trained with a different YAML
  they'll get a silent dimension mismatch or a panic. **Mitigation:** when
  saving, also drop a sibling `<checkpoint>.config.json` capturing the
  MlpConfig + parameter list. The bridge prefers that file over a passed-in
  YAML. Adds one file write to `training::checkpoint::save_mlp` — small
  diff, ten lines.
- **Vendored cubecl/burn forks.** `Cargo.toml` `[patch.crates-io]` block
  pins ~16 crates to local paths under `/home/tbindas/projects/`. `ddrs-py`
  inherits this through its path-dep on `ddrs`. Anyone setting up the bridge
  on a fresh machine needs the same clones at the same paths. Document in
  `ddrs-py/README.md`; revisit when SP-8 upstream PRs land.
- **`Autodiff` wrapping in `MuskingumCunge::forward`.** Returns
  `Tensor<Autodiff<I>, 2>`. For `run_forward` we don't need gradients —
  either call with the plain backend (no autodiff wrapper) if the API
  allows, or strip gradients on the way out. Spike this in implementation;
  if it's invasive, drop `run_forward` from 10a and ship it in 10b.
- **Notebook reproducibility.** Out of scope for 10a but: when 10b lands,
  pin `pyproject.toml`, never commit `.ipynb` outputs without a `jupytext`
  mirror.

## Assumptions (per CLAUDE.md planning rules)

- Notebook users run on CPU. GPU inference is a later add-on, not a current
  requirement.
- "Weights across a map" means *predicted physical parameters per COMID*
  (after MLP forward + denormalize), not raw linear-layer weights.
- A standalone `ddrs-py` crate is acceptable. The alternative (PyO3 feature
  flag inside `ddrs`) was rejected because it would force `pyo3` into the
  training binaries' link path.
- Existing checkpoints in `output/saved_models/` are usable from the bridge.
  This assumes their `MlpConfig` matches the current `merit_training.yaml`.
  If they don't, the user re-trains a small one for testing — not a blocker.
- The `python-hooks` branch is the integration target. No `main` merge until
  10a + 10b are both green.

## Out of scope (explicitly deferred)

- The Python `ddrs` package (10b).
- Map visualization, geopandas, MERIT catchment polygons (10c). The
  shapefile already exists at
  `~/projects/ddr/data/merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp`
  (referenced by `merit_geometry_config.yaml`), so 10c has a known polygon
  source — no distribution concern.
- Wheel distribution / `pip install ddrs-py` from PyPI.
- GPU inference path.
- Loss-curve / training-diagnostic notebooks.
- Loading DDR's pretrained KAN checkpoints
  (`ddr-v0.5.2-merit-geometry-weights.pt`). KAN ≠ MLP. Cross-validation
  against DDR happens in 10b notebooks by running each runtime separately
  and comparing the resulting parameter fields.

## Success criteria

- `cd ddrs-py && uv run maturin develop --release` succeeds on this machine.
- `uv run pytest tests/smoke.py` is green.
- `cargo run --release --example compare_ddr_sandbox` still reports
  "ABSOLUTE MATCH" — the bridge has not perturbed the routing core.
- A 5-line notebook snippet (in the spec's "Public API surface" section
  above) executes end-to-end against a real `.mpk` checkpoint.
