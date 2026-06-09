# Formatting inputs

ddrs configs are YAML to mirror DDR's tooling
(`~/projects/ddr/config/merit_training_config.yaml` is the upstream
reference). They are loaded via `Config::from_yaml_file(path)`
(back-compat, training mode) or
`Config::from_yaml_file_with_mode(path, ConfigMode::Testing)` in
`src/config.rs`. Deserialization runs through `serde_yaml` into a
`ConfigRaw` intermediate and then into the public `Config` via
`From<ConfigRaw>`. Every optional field has a default, so
`Config::default()` still constructs for call sites that only need the
solver core (the V1 sandbox example does this).

This chapter walks through every YAML key the config understands —
top-level fields, the `data_sources:` and `experiment:` sections, the
`kan_head:` head config, the `params:` block that drives the routing
core, the `testing:` overlay, and how to add a new parameter without
breaking the existing tests. Every key documented here is verified
against the serde structs in `src/config.rs` and the shipped
`config/merit_training.yaml`.

## What it is

A ddrs config is a single YAML file. The canonical one ships at
`config/merit_training.yaml`; at runtime the `ddrs` CLI bootstraps a
working copy to `ddrs.yaml` (see [Running the code](running.md)). The
file has a small set of top-level scalars plus four object sections —
`data_sources`, `experiment`, `kan_head`, and `params` — and an
optional `testing` overlay.

```
mode: training              # str, "training" or "testing"
workflow: train-and-test    # optional enum, cross-validated against mode
geodataset: merit           # str, dataset family name
device: 0                   # usize, CUDA device ordinal
seed: 42                    # u64, Rust-side RNG seed
np_seed: 42                 # u64, mirrors DDR's numpy seed
data_sources: { ... }       # paths the dataloader reads in place
experiment: { ... }         # training schedule
kan_head:    { ... }        # KAN routing-head shape (alias: `mlp`)
params:      { ... }         # routing-engine knobs
testing:     { ... }         # optional overlay; applied when mode == testing
```

| Key | Type | Role |
|---|---|---|
| `mode` | string | Run mode, `training` or `testing`. Defaults to `training` when absent. The `testing` overlay is only applied in testing mode. |
| `workflow` | enum (optional) | `train`, `eval`, or `train-and-test` (kebab-case). Cross-validated against `mode`: training implies `train`/`train-and-test`, testing implies `eval`. A mismatch is a load-time error. Absent → `None`. |
| `geodataset` | string | Free-form dataset tag (`merit` for the CONUS adjacency set). Defaults to `merit`. |
| `device` | usize | CUDA device ordinal, mirrors DDR's `device:` key (`device: 2` → `cuda:2`). Defaults to `0`. |
| `seed`, `np_seed` | u64 | Two seeds — DDR draws both because numpy and torch RNGs are seeded independently. Both default to `42`. |
| `data_sources` | section | Paths read in place; see [Reading inputs](inputs-reading.md) for what each feeds. Optional section, but validated when present. |
| `experiment` | section | Training schedule (`batch_size`, `start_time`, `end_time`, `epochs`, `rho`, `shuffle`, `warmup`, `learning_rate`, `grad_clip_max_norm`, `checkpoint`). |
| `kan_head` | section | KAN head shape. Accepts the legacy key `mlp` as a serde alias. |
| `params` | section | Routing engine knobs (see [`params` section](#params-section)). |
| `testing` | section | Overlay applied to `experiment` when `mode == testing`. |

The defining types are in `src/config.rs`: `Config` /
`ConfigRaw` / `From<ConfigRaw>` for the root, and the section structs
`DataSources`, `Experiment`, `KanHeadConfigSection`, and `Params`.

## How to use it

### A complete example: `config/merit_training.yaml`

The current shipped MERIT training config:

```yaml
mode: training
workflow: train-and-test    # ddrs plan/run picks this up; override with --workflow X
geodataset: merit
device: 0                   # CUDA device ordinal (mirrors DDR's `device:` key)
seed: 42
np_seed: 42

# Source paths — read in place by ddrs's Rust loaders.
data_sources:
  attributes: /home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc
  # geospatial_fabric triggers the managed adjacency build on first `ddrs plan`.
  geospatial_fabric: /projects/mhpi/data/MERIT/raw/continent/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp
  streamflow: /mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic
  observations: /mnt/ssd1/data/icechunk/usgs_daily_observations
  gages: /home/tbindas/projects/ddr/references/gage_info/gages_3000.csv

experiment:
  batch_size: 64
  start_time: 1981/10/01
  end_time: 1995/09/30
  epochs: 5
  rho: 90
  shuffle: true
  warmup: 5
  learning_rate:
    1: 0.001
    3: 0.0005
  grad_clip_max_norm: 1.0

kan_head:
  hidden_size: 21
  num_hidden_layers: 2
  grid: 50      # B-spline grid intervals (`num` in pykan)
  k: 2          # B-spline order
  input_var_names:
    - SoilGrids1km_clay
    - aridity
    - meanelevation
    - meanP
    - NDVI
    - meanslope
    - log10_uparea
    - SoilGrids1km_sand
    - ETPOT_Hargr
    - Porosity
  learnable_parameters:
    - n
    - q_spatial
    - p_spatial

params:
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
  attribute_minimums:
    discharge: 1.0e-4
    slope: 1.0e-3
    velocity: 0.01
    depth: 0.01
    bottom_width: 0.01
  defaults:
    p_spatial: 21.0
  log_space_parameters:
    - p_spatial
  sparse_solver: cuda    # opt-in for GPU cuSPARSE solve
  use_cuda_graphs: true  # SP-10: forward CUDA Graph capture+replay

testing:
  start_time: 1995/10/01
  end_time: 2010/09/30
  batch_size: 15      # DAYS, not gauges
  rho: null           # disabled in test mode
```

### `data_sources` — adjacency strategy

`DataSources` has six path fields plus a layer selector. `attributes`,
`streamflow`, `observations`, and `gages` are always required when the
section is present; the adjacency inputs follow one of two strategies
(validated by `validate_data_sources` at load time):

```yaml
# Strategy A — managed build (the shipped default):
data_sources:
  attributes: ...
  geospatial_fabric: .../riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp
  streamflow: ...
  observations: ...
  gages: ...

# Strategy B — pre-built zarr stores (drop geospatial_fabric, set both):
data_sources:
  attributes: ...
  conus_adjacency: /path/to/merit_conus_adjacency.zarr
  gages_adjacency: /path/to/merit_gages_conus_adjacency.zarr
  streamflow: ...
  observations: ...
  gages: ...
```

The validation rules are strict:

- Both `conus_adjacency` **and** `gages_adjacency` present → OK.
- Neither adjacency key present, but `geospatial_fabric` present → OK
  (managed build by `ddrs plan`).
- Exactly one of the two adjacency keys → error (partial adjacency).
- Neither adjacency key and no `geospatial_fabric` → error.
- `geospatial_fabric_layer` set while the fabric is not a `.gpkg` →
  error (the layer concept only applies to GeoPackage fabrics).

`geospatial_fabric` accepts `.shp` (the sibling `.dbf` is read), `.dbf`,
or `.gpkg`; geometry is never opened in any format. Set
`geospatial_fabric_layer` only for multi-layer `.gpkg` fabrics. See
[Reading inputs](inputs-reading.md) for what each path actually feeds.

### `kan_head` — the routing head

`KanHeadConfigSection` configures the KAN head shape. The section may be
named `kan_head:` (the v1 key) or `mlp:` (a backward-compat serde alias
retained for older configs).

| Key | Type | Default | Role |
|---|---|---|---|
| `hidden_size` | usize | required | KAN hidden width |
| `num_hidden_layers` | usize | required | Number of inner `KanLayer` blocks |
| `grid` | usize | `5` | B-spline grid intervals (`num` in pykan); merit YAML sets `50` |
| `k` | usize | `3` | B-spline order; merit YAML sets `2` (DDR's production override) |
| `input_var_names` | `Vec<String>` | required | Attribute columns fed to the head |
| `learnable_parameters` | `Vec<String>` | required | Which routing parameters the head produces |

`grid` and `k` default to `5` and `3` (pykan defaults) when absent; the
merit config overrides them to `50` and `2` to match DDR production.

### `experiment` — training schedule

| Key | Type | Notes |
|---|---|---|
| `batch_size` | usize | Number of gauges per mini-batch (training). Required. |
| `start_time`, `end_time` | string | `YYYY/MM/DD` window bounds. Required. |
| `epochs` | usize | Required. |
| `rho` | usize (optional) | Rho-window length; `None` if absent. |
| `shuffle` | bool | Defaults to `false`. |
| `warmup` | usize | Spin-up steps. Required. |
| `learning_rate` | `BTreeMap<usize, f32>` | Epoch → LR schedule. Defaults to empty. |
| `grad_clip_max_norm` | f32 (optional) | Gradient clip norm. |
| `checkpoint` | path (optional) | Resume directory. |

## `params` section

`Params` is the routing-core configuration. YAML enters via `ParamsRaw`
and is folded into the typed `Params` by `From<ParamsRaw>`.

### `parameter_ranges`

Physical `[min, max]` ranges used to denormalize the NN's `[0,1]`
outputs into real channel-routing parameters.

| Key | YAML shape | Default | Used by |
|---|---|---|---|
| `n` | `[min, max]` f32 pair | `[0.015, 0.25]` | Manning's roughness |
| `q_spatial` | `[min, max]` f32 pair | `[0.0, 1.0]` | Discharge spatial term |
| `p_spatial` | `[min, max]` f32 pair | `[1.0, 200.0]` | Pressure spatial term |

YAML is a dict-of-2-tuples (`HashMap<String, [f32; 2]>`); the parse
block reads only these three known keys.

### `attribute_minimums`

Physical lower bounds clamped during routing to keep the math stable.
Every clamp in [Algorithm](../algorithm.md) (depth, bottom_width,
velocity, discharge) comes from this block.

| Key | Default | Units |
|---|---|---|
| `discharge` | `1.0e-4` | m³/s |
| `slope` | `1.0e-3` | unitless |
| `velocity` | `0.01` | m/s |
| `depth` | `0.01` | m |
| `bottom_width` | `0.01` | m |

### `log_space_parameters`

A `Vec<String>` listing parameter names whose denormalization happens in
log10-space rather than linear (see `src/routing/utils.rs::denormalize`).
The Rust default is `["p_spatial"]`, and the merit YAML sets the same
value (`["p_spatial"]`).

If the YAML list is non-empty it **replaces** the default entirely;
an empty/absent list keeps the default.

### `defaults`

A `HashMap<String, f32>` of fallback values for parameters not produced
by the NN head. Both the Rust default and the merit YAML set
`p_spatial: 21.0`. As with `log_space_parameters`, a non-empty YAML
value overrides the default; an empty/absent one keeps it.

### Solver toggles

| Key | Type | YAML value | Rust default | Effect |
|---|---|---|---|---|
| `tau` | u32 | unset → 3 | 3 | Hours-per-substep when subdividing the daily forcing |
| `sparse_solver` | `"cpu"` \| `"cuda"` | `cuda` | `Cpu` | Picks the CSR triangular solve backend |
| `use_cuda_graphs` | bool | `true` | `false` | Enables per-timestep CUDA-graph capture+replay |

Parsing of `sparse_solver` accepts both lower and upper case (`cpu`,
`CPU`, `cuda`, `CUDA`); anything else panics with
`unknown sparse_solver: "..."`. `use_cuda_graphs` silently has no effect
on the CPU path. On a non-CUDA backend, `sparse_solver: cuda` falls back
to `Cpu` (logged once at WARN).

## Defaults

The YAML in `config/merit_training.yaml` is **CUDA-on**:

```yaml
params:
  sparse_solver: cuda    # SP-9 (commit dbcf6e6) — was cpu before
  use_cuda_graphs: true  # SP-10 (commit e35af29) — was false before
```

The Rust-side `Params::default()` is still `Cpu` +
`use_cuda_graphs: false`, because the routing solver constructs a
sensible default without a YAML — but every code path that loads
`merit_training.yaml` opts into the GPU.

CPU-only override is one line each:

```yaml
params:
  sparse_solver: cpu
  use_cuda_graphs: false
```

## Adding a new parameter

Three coordinated edits in `src/config.rs`. Example: adding an
`enable_foo: bool` toggle to `params`.

1. **Extend `Params` + its `Default` impl:**

   ```rust
   pub struct Params {
       // ...existing fields...
       pub enable_foo: bool,
   }

   impl Default for Params {
       fn default() -> Self {
           Self {
               // ...existing fields...
               enable_foo: false,
           }
       }
   }
   ```

2. **Add an `Option<T>` to `ParamsRaw`:**

   ```rust
   struct ParamsRaw {
       // ...existing fields...
       enable_foo: Option<bool>,
   }
   ```

3. **Wire it into the `From<ParamsRaw>` parse block:**

   ```rust
   if let Some(b) = r.enable_foo {
       p.enable_foo = b;
   }
   ```

Then add an assertion to the `loads_merit_training_yaml` test so the
default behavior is locked. For root-level fields the pattern is the
same but in `Config`, `ConfigRaw`, and the `From<ConfigRaw>` block.

## Reference

### Top-level keys and their defaults

| Key | Struct field | Default when absent |
|---|---|---|
| `mode` | `Config::mode` | `"training"` |
| `workflow` | `Config::workflow` | `None` |
| `geodataset` | `Config::geodataset` | `"merit"` |
| `device` | `Config::device` | `0` |
| `seed` | `Config::seed` | `42` |
| `np_seed` | `Config::np_seed` | `42` |

### Gotchas

- **Unknown YAML keys are silently dropped.** `ParamsRaw` and
  `ConfigRaw` both use `#[serde(default)]` and do *not* use
  `#[serde(deny_unknown_fields)]`. A typo (`use_cuda_graph` instead of
  `use_cuda_graphs`) compiles, runs, and silently uses the default.
  Check that the `loads_merit_training_yaml` assertions match what you
  wrote.
- **`log_space_parameters` entries are bare strings.** A typo (`m` for
  `n`) parses fine and silently changes the denorm formula for whatever
  matched. There's no compile-time check; the only guard is the merit
  YAML test asserting the exact list (currently `["p_spatial"]`).
- **YAML defaults moved across SPs.** `sparse_solver` flipped to `cuda`
  in SP-9 (commit `dbcf6e6`); `use_cuda_graphs` flipped to `true` in
  SP-10 (commit `e35af29`). Don't hard-code the assumption that either
  is `false` in tests — read the YAML or set them explicitly.
- **`kan_head` vs `mlp`.** The section is `kan_head:`; `mlp:` is kept
  only as a serde alias for older configs. Prefer `kan_head:` in new
  files.
- **`mode` and `workflow` are cross-validated.** A `mode: training` /
  `workflow: eval` combination (or `mode: testing` / `workflow: train`)
  is rejected at load time as a `DataError::Yaml` with a "conflicting
  top-level keys" message.
- **`testing.batch_size` semantically shifts.** In `experiment` it's the
  number of *gauges* per mini-batch; in `testing` it's the number of
  *days* per chunk. The overlay copies the value verbatim, so the unit
  change is invisible — the YAML comment is your only warning.
- **`testing.rho: null` is distinct from absent.** A custom serde shim
  (double-`Option`) lets `null` explicitly clear `rho`; leaving the key
  out preserves the training-side value.
- **`sparse_solver` rejects unknowns with a panic, not an error.**
  Typos like `sparse_solver: gpu` crash with
  `unknown sparse_solver: "gpu"` — not a clean `DataError::Yaml`. Don't
  hand this YAML to an end user uninspected.

### Verification

```bash
cargo test --lib config::
```

Covers the critical assertions:

| Test | Locks |
|---|---|
| `loads_merit_training_yaml` | YAML round-trip, every default in `params`, the `kan_head` section, top-level `seed`/`mode`/`workflow`/`device` |
| `default_config_still_constructs` | `Config::default()` keeps working for the routing-only path |
| `testing_mode_overlays_apply_to_experiment` | Testing overlay copies fields and clears `rho` |
| `training_mode_does_not_apply_overlays` | Training mode leaves `experiment` untouched |
| `mode_workflow_conflict_rejected` / `*_data_sources*` | mode/workflow and adjacency validation matrix |

If a new YAML key is added, extend `loads_merit_training_yaml` with an
explicit assertion — silent serde defaults are the gotcha above.

## See also

- [Reading inputs](inputs-reading.md) — what the `data_sources:` paths
  point at and how each is read.
- [Running the code](running.md) — how `--config` and `ddrs init/plan/run`
  wire the YAML through the CLI.
- [Setup](../setup.md) — the canonical data-source paths and how to flip
  the CUDA defaults to CPU.
- [Algorithm](../algorithm.md) — why every key in `attribute_minimums`
  matters.
- [Performance & CUDA Graphs](../reference/perf.md) — what the
  `sparse_solver` and `use_cuda_graphs` toggles actually do.
