# Formatting inputs

ddrs configs are YAML to mirror DDR's tooling
(`~/projects/ddr/config/merit_training_config.yaml` is the upstream
reference). They are loaded via `Config::from_yaml_file(path)`
(back-compat, training mode) or
`Config::from_yaml_file_with_mode(path, ConfigMode::Testing)` in
`src/config.rs`. Deserialization runs through `serde_yaml` into a
`ConfigRaw` intermediate and then into the public `Config` via
`From<ConfigRaw>`. Every optional field has a default, so
`Config::default()` still constructs for call sites that only need
the solver core (the V1 sandbox example does this).

This chapter walks through every YAML key the config understands —
top-level fields, the `params:` block that drives the routing core,
solver toggles, defaults, and how to add a new parameter without
breaking the existing tests.

## Top-level structure

```
mode: training              # str, "training" or "testing"
geodataset: merit           # str, dataset family name
seed: 42                    # u64, Rust-side RNG seed
np_seed: 42                 # u64, mirrors DDR's numpy seed
data_sources: { ... }       # required for the SP-3 dataloader
experiment: { ... }         # required for the SP-3 dataloader
mlp:        { ... }         # required when training the MLP head
params:     { ... }         # required for the routing core
testing:    { ... }         # optional overlay; applied when mode==testing
```

| Key | Type | Role |
|---|---|---|
| `mode` | string | Selects the run mode; `Config::from_yaml_file` ignores it, the binary reads it as the default for `--mode`. Defaults to `training`. |
| `geodataset` | string | Free-form dataset tag (`merit` for the CONUS adjacency set). Defaults to `merit`. |
| `seed`, `np_seed` | u64 | Two seeds — DDR draws both because numpy and torch RNGs are seeded independently. Both default to `42`. |
| `data_sources` | section | Five `PathBuf` fields + a gauges CSV; see [Reading inputs](inputs-reading.md) for what each path feeds. No defaults — required to construct the dataset. |
| `experiment` | section | Training schedule. `batch_size`, `start_time`, `end_time`, `epochs`, optional `rho`, `shuffle`, `warmup`, `learning_rate` map, optional `grad_clip_max_norm`, optional `checkpoint`. |
| `mlp` | section | `hidden_size`, `num_hidden_layers`, `input_var_names`, `learnable_parameters`. |
| `params` | section | Routing engine knobs (next section). |
| `testing` | section | Overlay applied to `experiment` when `mode == Testing`. |

The defining file is `src/config.rs` — `Config` struct around line
222, defaults around line 275.

## A complete example: `config/merit_training.yaml`

The current shipped MERIT training config:

```yaml
mode: training
geodataset: merit
seed: 42
np_seed: 42

# Source paths — read in place by ddrs's Rust loaders.
data_sources:
  attributes: /home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc
  conus_adjacency: /home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr
  gages_adjacency: /home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr
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

mlp:
  hidden_size: 21
  num_hidden_layers: 2
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
    - n
  sparse_solver: cuda    # opt-in for GPU cuSPARSE solve
  use_cuda_graphs: true  # SP-10: forward CUDA Graph capture+replay

testing:
  start_time: 1995/10/01
  end_time: 2010/09/30
  batch_size: 15      # DAYS, not gauges
  rho: null           # disabled in test mode
```

## `params` section

`Params` is the routing-core configuration. YAML enters via
`ParamsRaw` and is folded into the typed `Params` by
`From<ParamsRaw>`.

### `parameter_ranges`

Physical `[min, max]` ranges used to denormalize the NN's `[0,1]`
outputs into real channel-routing parameters.

| Key | YAML shape | Default | Used by |
|---|---|---|---|
| `n` | `[min, max]` f32 pair | `[0.015, 0.25]` | Manning's roughness |
| `q_spatial` | `[min, max]` f32 pair | `[0.0, 1.0]` | Discharge spatial term |
| `p_spatial` | `[min, max]` f32 pair | `[1.0, 200.0]` | Pressure spatial term |

YAML is a dict-of-2-tuples (`HashMap<String, [f32; 2]>`); the parse
block reads only the three known keys.

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

A `Vec<String>` listing parameter names whose denormalization happens
in log10-space rather than linear (see
`src/routing/utils.rs::denormalize`). The merit YAML overrides the
default (`["p_spatial"]`) with `["n"]`.

If the YAML list is non-empty it **replaces** the default entirely;
otherwise the default survives.

### `defaults`

A `HashMap<String, f32>` of fallback values for parameters not
produced by the NN head. Merit YAML sets `p_spatial: 21.0`. As with
`log_space_parameters`, non-empty overrides the default.

### Solver toggles

| Key | Type | YAML default | Rust default | Effect |
|---|---|---|---|---|
| `tau` | u32 | unset → 3 | 3 | Hours-per-substep when subdividing the daily forcing |
| `sparse_solver` | `"cpu"` \| `"cuda"` | `cuda` | `Cpu` | Picks the CSR triangular solve backend |
| `use_cuda_graphs` | bool | `true` | `false` | Enables per-timestep CUDA-graph capture+replay |

Parsing of `sparse_solver` accepts both lower and upper case; anything
else panics with `unknown sparse_solver: "..."`. `use_cuda_graphs`
silently has no effect on the CPU path.

## Defaults

The YAML default in `config/merit_training.yaml` is **CUDA-on**:

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

1. **Extend `Params` + default** (`src/config.rs` around lines 122-150):

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

2. **Add an `Option<T>` to `ParamsRaw`** (`src/config.rs` around lines 156-166):

   ```rust
   struct ParamsRaw {
       // ...existing fields...
       enable_foo: Option<bool>,
   }
   ```

3. **Wire it into the parse block** (`src/config.rs` around lines 168-215):

   ```rust
   if let Some(b) = r.enable_foo {
       p.enable_foo = b;
   }
   ```

Then add an assertion to `loads_merit_training_yaml`
(`src/config.rs` around lines 339-368) so the default behavior is
locked.

For root-level fields the pattern is the same but in `Config`,
`ConfigRaw`, and the `From<ConfigRaw>` block.

## Gotchas

- **Unknown YAML keys are silently dropped.** `ParamsRaw` and
  `ConfigRaw` both use `#[serde(default)]` and do *not* use
  `#[serde(deny_unknown_fields)]`. A typo (`use_cuda_graph` instead of
  `use_cuda_graphs`) compiles, runs, and silently uses the default.
  Watch for this when editing YAML; check that the
  `loads_merit_training_yaml` assertions match what you wrote.
- **`log_space_parameters` entries are bare strings.** A typo (`m`
  for `n`) parses fine and silently changes the denorm formula for
  whatever matched. There's no compile-time check; the only guard is
  the merit YAML test asserting the exact list.
- **YAML defaults moved across SPs.** `sparse_solver` flipped to
  `cuda` in SP-9 (commit `dbcf6e6`); `use_cuda_graphs` flipped to
  `true` in SP-10 (commit `e35af29`). Don't hard-code the assumption
  that either is `false` in tests — read the YAML or set them
  explicitly.
- **`testing.batch_size` semantically shifts.** In `experiment` it's
  the number of *gauges* per mini-batch; in `testing` it's the number
  of *days* per chunk. The overlay copies the value verbatim, so the
  unit change is invisible — the YAML comment is your only warning.
- **`testing.rho: null` is distinct from absent.** A custom serde
  shim lets `null` explicitly clear `rho`; leaving the key out
  preserves the training-side value.
- **`sparse_solver` rejects unknowns with a panic, not an error.**
  Typos like `sparse_solver: gpu` crash with
  `unknown sparse_solver: "gpu"` — not a clean `DataError::Yaml`.
  Don't hand this YAML to an end user uninspected.

## Verification

```bash
cargo test --lib config::
```

Covers the four critical assertions:

| Test | Locks |
|---|---|
| `loads_merit_training_yaml` | YAML round-trip, every default in `params`, mlp section, top-level seed/mode |
| `default_config_still_constructs` | `Config::default()` keeps working for the routing-only path |
| `testing_mode_overlays_apply_to_experiment` | Testing overlay copies fields and clears `rho` |
| `training_mode_does_not_apply_overlays` | Training mode leaves `experiment` untouched |

If a new YAML key is added, extend `loads_merit_training_yaml` with
an explicit assertion — silent serde defaults are the gotcha above.

## See also

- [Reading inputs](inputs-reading.md) — what the `data_sources:`
  paths point at.
- [Running the code](running.md) — how `--config` is wired through
  the binaries.
- [Algorithm](../algorithm.md) — why every key in
  `attribute_minimums` matters.
- [Performance & CUDA Graphs](../reference/perf.md) — what the
  `sparse_solver` and `use_cuda_graphs` toggles actually do.
