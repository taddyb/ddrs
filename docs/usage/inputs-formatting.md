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
| `experiment` | section | Training schedule (`batch_size`, `start_time`, `end_time`, `epochs`, `rho`, `shuffle`, `warmup`, `learning_rate`, `grad_clip_max_norm`, `checkpoint`, `state_cache`, `loss`). |
| `kan_head` | section | KAN head shape, plus the optional `disaggregation` sub-section. Accepts the legacy key `mlp` as a serde alias. |
| `params` | section | Routing engine knobs (see [`params` section](#params-section)). |
| `testing` | section | Overlay applied to `experiment` when `mode == testing`; eight overridable keys. |

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
  # Training objective. Omit this block for the historical L1 loss.
  #   kind: l1        → mean(|p - o|)  (default; rewards peak attenuation)
  #   kind: nnse-kge  → nnse-weight·(1 - NNSE) + kge-weight·(1 - KGE), per gauge.
  #                     The KGE term's (alpha-1)^2 restores the hydrograph
  #                     variance that L1/NSE shrink away (fixes the KGE
  #                     regression vs the summed-Q' baseline). See
  #                     src/training/loss.rs.
  # loss:
  #   kind: nnse-kge
  #   nnse-weight: 1.0
  #   kge-weight: 1.0
  #   eps: 0.1          # stabilizes variance/mean denominators

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

The shipped file carries the `loss:` block **commented out** — that is
deliberate (an absent block means `LossKind::L1`, the historical
behavior), and it doubles as the in-file reference for the option. Note
also what the shipped config does *not* contain: neither adjacency key,
no `aorc_precip`, no `kan_head.disaggregation`, and no leakance keys.

### `data_sources` — paths and the adjacency strategy

`DataSources` has **eight** path fields plus a layer selector:

| Field | Rust type | Required? |
|---|---|---|
| `attributes` | `Vec<PathBuf>` | yes |
| `streamflow` | `PathBuf` | yes |
| `observations` | `PathBuf` | yes |
| `gages` | `PathBuf` | yes |
| `conus_adjacency` | `Option<PathBuf>` | adjacency Strategy B |
| `gages_adjacency` | `Option<PathBuf>` | adjacency Strategy B |
| `geospatial_fabric` | `Option<PathBuf>` | adjacency Strategy A |
| `aorc_precip` | `Option<PathBuf>` | only with `kan_head.disaggregation` |
| `geospatial_fabric_layer` | `Option<String>` | multi-layer `.gpkg` only |

#### `attributes` accepts one path or many

`attributes` is a `Vec<PathBuf>` behind a custom
`deserialize_one_or_many_paths` shim, so both spellings parse:

```yaml
# scalar — becomes a one-element Vec, byte-identical to the old behavior
attributes: /home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc

# list — routes to AttributesStore::open_multi
attributes:
  - /path/to/merit_global_attributes_v2.nc
  - /path/to/streamcat_corridor.nc
```

With two or more paths the files are **feature-concatenated on COMID**,
not row-concatenated: each requested variable in
`kan_head.input_var_names` must live in exactly one file (present in two
⇒ load error; present in none ⇒ load error), and a COMID missing from
one file gets `NaN` for that file's variables. An **empty list** is a
hard deserialize error (`data_sources.attributes: list must not be
empty`). Tests: `attributes_bare_path_parses_as_single_element_vec`,
`attributes_yaml_list_parses_as_multi_element_vec`,
`attributes_empty_list_rejected`.

#### Adjacency: two strategies

`attributes`, `streamflow`, `observations`, and `gages` are always
required when the section is present; the adjacency inputs follow one of
two strategies (validated by `validate_data_sources` at load time):

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

#### `aorc_precip`

Optional path to the hourly AORC precipitation store
(`merit_unit_catchments.zarr`, zarr v3, CONUS-only). It is only read when
a `kan_head.disaggregation:` block is present, and in that case it is
**mandatory**: `MeritGagesDataset::open` errors if the head wants precip
and no `aorc_precip` source is configured, so a missing precip store can
never silently degrade to flat daily forcing. Note this is a *runtime*
check in the dataset, not one of the four config-load validators.

### Load-time validation

`Config::from_yaml_file_with_mode` runs **four** validators in order,
each wrapping its message in a `DataError::Yaml`:

| Validator | Rejects | Error substring |
|---|---|---|
| `validate_mode_workflow` | `mode`/`workflow` disagreement | `conflicting top-level keys` |
| `validate_data_sources` | the adjacency matrix below; `geospatial_fabric_layer` on a non-`.gpkg` fabric | `data_sources:` |
| `validate_leakance` | `use_leakance: true` together with `use_cuda_graphs: true` | `` `use_leakance: true` requires `use_cuda_graphs: false` `` |
| `validate_disagg_pretrained` | `disaggregation.freeze: true` without `pretrained_checkpoint` | `` `freeze: true` requires `pretrained_checkpoint` `` |

The `testing:` overlay is applied *after* all four run, so validation
always sees the training-mode view of the config.

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
| `disaggregation` | section (optional) | absent | Enables the learnable daily→hourly forcing disaggregation head; absent ⇒ flat `repeat-24` |

`grid` and `k` default to `5` and `3` (pykan defaults) when absent; the
merit config overrides them to `50` and `2` to match DDR production.

#### `kan_head.disaggregation` — the daily→hourly head

The **presence of the block** turns the head on; there is no `enabled`
flag and — important — **no `use_precip` key**. The head always consumes
`(daily Q′, that day's 24 h precip)`, which is why the block's presence
makes `data_sources.aorc_precip` mandatory. Eight keys, all optional:

| Key | Type | Default | Role |
|---|---|---|---|
| `hidden_size` | usize | `16` | Hidden width of the disagg KAN |
| `num_hidden_layers` | usize | `1` | Inner `KanLayer` count |
| `grid` | usize | `3` | B-spline grid intervals |
| `k` | usize | `3` | B-spline order |
| `boundary_blend` | f32 | `0.0` | Day-boundary shape-continuity blend λ ∈ [0,1]; `0.0` ⇒ fully independent per-day shapes. Only used when `chunk_days <= 1` |
| `chunk_days` | usize | `1` | Mass-balance chunk size in days. `1` keeps the per-calendar-day-exact contract (byte-identical to pre-field checkpoints); `> 1` relaxes conservation to the chunk aggregate so storms spanning midnight stay one event |
| `pretrained_checkpoint` | path (optional) | `None` | Warm-start from a standalone-pretrained `DisaggHead` (`CompactRecorder` `.mpk`). The five architecture fields above MUST match what the checkpoint was trained with — `load_record` fails loudly otherwise |
| `freeze` | bool | `false` | `Module::no_grad()` after loading. **Requires** `pretrained_checkpoint` — rejected at load time otherwise |

`pretrained_checkpoint` and `freeze` are operational, not architectural:
deliberately not threaded through `kan_config`, mirroring how
`experiment.checkpoint` bypasses it. Tests:
`disagg_freeze_without_pretrained_checkpoint_fails_to_load`,
`disagg_pretrained_fields_default_to_none_and_false`.

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
| `state_cache` | path (optional) | Day-boundary discharge cache; see below. |
| `loss` | section | Training objective; defaults to L1. See below. |

### `experiment.state_cache`

Path to a day-boundary discharge state cache — a **netCDF** file (dims
`(day, COMID)`, var `q_state` f32, var `COMID` i64, global attr `day0` as
an ISO date string), produced by `--mode state-cache` in
`probe_zeta_gradient`. Not zarr.

When set, `collate` attaches the window-start per-reach Q vector as
`RoutingBatch::initial_state`, and `setup_inputs` uses it instead of the
cold-start hotstart solve. Absent ⇒ `None` ⇒ every code path is
byte-identical to the no-cache behavior. Reads are lazy (one row, ~256 KB,
per `row_for_day` call) because the full array is ~1.3 GB at CONUS scale.
Tests: `state_cache_absent_yields_none`, `state_cache_path_parses`.

### `experiment.loss` — the training objective

Omit the block and you get `kind: l1`, the historical objective, exactly.
The section deserializes with `#[serde(default, rename_all =
"kebab-case")]`, so every sub-key is **kebab-case in YAML** and every one
has a default:

| YAML key | Rust field | Default | Used by |
|---|---|---|---|
| `kind` | `kind` | `l1` | — |
| `nnse-weight` | `nnse_weight` | `1.0` | `nnse-kge`, and the optional NNSE guard of `kge` |
| `kge-weight` | `kge_weight` | `1.0` | `nnse-kge` only |
| `r-weight` | `r_weight` | `1.0` | `kge` only — the `(r-1)²` correlation term |
| `alpha-weight` | `alpha_weight` | `1.0` | `kge` only — the `(α-1)²` variance-ratio term; the anti-attenuation restoring force |
| `beta-weight` | `beta_weight` | `1.0` | `kge` only — the `(β-1)²` mean-ratio term |
| `kge-clamp` | `kge_clamp` | `10.0` | `kge` only — per-gauge upper bound on the weighted component sum before averaging |
| `eps` | `eps` | `0.1` | all non-L1 kinds — stabilizes variance/mean denominators (matches DDR `hydrograph_loss`) |

`kind` accepts **three** values on this commit (`LossKind`, also
kebab-case):

- `l1` — `mean(|p - o|)`. The default; rewards peak attenuation.
- `nnse-kge` — per gauge `nnse_weight·(1 - NNSE) + kge_weight·(1 - KGE)`.
  The KGE term's `(α-1)²` restores the hydrograph variance L1/NSE shrink
  away.
- `kge` — per gauge `r_w·(r-1)² + α_w·(α-1)² + β_w·(β-1)² +
  nnse_w·(1-NNSE)`. The KGE components are weighted individually (no
  sqrt Euclidean term, hence no gradient singularity at perfect KGE), so
  `alpha-weight` can up-weight the anti-attenuation force.

`kge-clamp` exists because gauges with near-constant observed flow have a
collapsing `std_o`/`mean_o` denominator, so `(α-1)²`/`(β-1)²` can explode
and hijack the batch gradient (a single gauge drove batch loss to ~1e4 in
testing). Tests: `loss_config_defaults_to_l1`,
`loss_config_parses_nnse_kge_kebab_case`.

### `testing` — the eval overlay

`TestingOverridesRaw` overlays **eight** `experiment` keys when the
config is loaded with `ConfigMode::Testing`. Every field is optional, so
an absent key inherits the training value:

| `testing` key | Overrides | Note |
|---|---|---|
| `start_time` | `experiment.start_time` | |
| `end_time` | `experiment.end_time` | |
| `batch_size` | `experiment.batch_size` | **semantics shift**: DAYS per eval chunk, not gauges |
| `rho` | `experiment.rho` | double-`Option`: `rho: null` explicitly clears it; an absent key inherits |
| `warmup` | `experiment.warmup` | |
| `epochs` | `experiment.epochs` | |
| `grad_clip_max_norm` | `experiment.grad_clip_max_norm` | |
| `checkpoint` | `experiment.checkpoint` | parsed as a string, converted to `PathBuf` |

Anything *not* in that list — `shuffle`, `learning_rate`, `state_cache`,
`loss` — cannot be overridden per-mode; the training value always
applies. Tests: `testing_mode_overlays_apply_to_experiment`,
`training_mode_does_not_apply_overlays`.

## `params` section

`Params` is the routing-core configuration. YAML enters via `ParamsRaw`
and is folded into the typed `Params` by `From<ParamsRaw>`.

### `parameter_ranges`

Physical `[min, max]` ranges used to denormalize the NN's `[0,1]`
outputs into real channel-routing parameters.

| YAML key | Rust field | Default | Used by |
|---|---|---|---|
| `n` | `n` | `[0.015, 0.25]` | Manning's roughness |
| `q_spatial` | `q_spatial` | `[0.0, 1.0]` | Leopold & Maddock width–depth exponent |
| `p_spatial` | `p_spatial` | `[1.0, 200.0]` | Leopold & Maddock width coefficient |
| `x_storage` | `x_storage` | `[0.0, 0.5]` | Muskingum storage weight X |
| `K_D` | `k_d` | `[1e-8, 1e-6]` | Leakance hydraulic exchange rate (1/s) |
| `d_gw` | `d_gw` | `[-2.0, 2.0]` | Leakance groundwater depth offset (m) |
| `leakance_factor` | `leakance_factor` | `[0.0, 1.0]` | Leakance dimensionless scale |

YAML is a dict-of-2-tuples (`HashMap<String, [f32; 2]>`); the
`From<ParamsRaw>` block reads **seven** known keys, and silently ignores
anything else.

Two traps in that table:

- **`K_D` is uppercase in YAML, `k_d` in Rust.** The lookup is a literal
  string match on `"K_D"` (`src/config.rs`), spelled that way because DDR
  spells it uppercase while every sibling key is lowercase. Writing
  `k_d:` in YAML parses fine and silently leaves the default range in
  place. The same uppercase spelling is what `log_space_parameters` and
  `kan_head.learnable_parameters` must use.
- **A range is only consumed if the head emits the parameter.**
  `x_storage`'s range is used only when `x_storage` is listed in
  `kan_head.learnable_parameters`; otherwise routing uses a constant
  `0.3`. Likewise `K_D` / `d_gw` / `leakance_factor` are consumed only
  when `params.use_leakance: true` **and** all three are in
  `learnable_parameters` (the leakance params are all-or-nothing: any one
  missing routes the non-leakance path).

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

| Key | Type | Merit YAML | Rust default | Effect |
|---|---|---|---|---|
| `tau` | u32 | unset → 3 | 3 | UTC→local phase offset of the daily-aggregation trim window (see below) |
| `sparse_solver` | `"cpu"` \| `"cuda"` | `cuda` | `Cpu` | Picks the CSR triangular solve backend |
| `use_cuda_graphs` | bool | `true` | `false` | Enables per-timestep CUDA-graph capture+replay |
| `use_leakance` | bool | unset → `false` | `false` | Enables the GW–SW water-loss term in routing |
| `leakance_losing_only` | bool | unset → `true` | **`true`** | Clamps the leakance head term to `max(0, depth − d_gw)` so gaining reaches produce `zeta ≡ 0` |
| `leakance_impervious_threshold` | f32 | unset → `0.7` | `0.7` | Reaches with `corridor_impervious` **strictly greater than** this get `zeta ≡ 0` and zero gradient to their leakance params |

Parsing of `sparse_solver` accepts both lower and upper case (`cpu`,
`CPU`, `cuda`, `CUDA`); anything else panics with
`unknown sparse_solver: "..."`. `use_cuda_graphs` silently has no effect
on the CPU path. On a non-CUDA backend, `sparse_solver: cuda` falls back
to `Cpu` (logged once at WARN).

#### `tau` is not a substep count

`tau` does **not** subdivide the forcing. The routing core never reads it
— `grep -rn '\btau\b' src/routing/` returns nothing. It is consumed only
by `tau_trim_and_downsample` (`src/training/loss.rs`), which trims the
hourly prediction window before daily area-pooling:

```rust
let start = 13 + tau as usize;
let end   = t_hours - 11 + tau as usize;
```

Those constants come straight from DDR's `compute_daily_runoff` slice
`[13 + tau : -11 + tau]`. In other words `tau` is the **UTC→local phase
offset** that aligns the model's hourly axis with the observation
product's local-day boundaries. Raising it shifts *which* hours land in
each daily bin; it changes nothing about the routing timestep, which is
fixed at `DT_SECONDS = 3600.0` in `src/routing/mmc.rs`.

#### Leakance toggles

`leakance_losing_only` and `leakance_impervious_threshold` are Phase-C
physical guards; both are inert unless `use_leakance: true`. Note the two
defaults that are easy to get backwards:

- `leakance_losing_only` defaults to **`true`** (guard ON). Set it to
  `false` to recover the prior unclamped behavior byte-identically.
- The impervious comparison is strict: `corridor_impervious > threshold`
  zeroes the reach, so a reach exactly at `0.7` is **not** masked. NaN
  (no StreamCat coverage) is treated as not-impervious — absence of data
  is not concrete. The mask is only built when leakance is on *and*
  `corridor_impervious` is among `kan_head.input_var_names`.

Turning leakance on takes three coordinated edits, not one:

```yaml
kan_head:
  learnable_parameters: [n, q_spatial, p_spatial, K_D, d_gw, leakance_factor]
params:
  use_leakance: true
  use_cuda_graphs: false        # REQUIRED — see below
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-6]
    d_gw: [-2, 2]
    leakance_factor: [0, 1]
```

Two combinations are rejected at load time rather than silently
misbehaving:

1. `use_leakance: true` with `use_cuda_graphs: true` → error (the
   CUDA-graph capture path bakes the non-leakance `b_rhs` into the
   graph). Test: `leakance_with_cuda_graphs_rejected`.
2. `kan_head.disaggregation.freeze: true` without
   `pretrained_checkpoint` → error (freezing a randomly-initialized head
   would train nothing). Test:
   `disagg_freeze_without_pretrained_checkpoint_fails_to_load`.

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
- **`K_D` is the only uppercase parameter name.** `parameter_ranges`,
  `log_space_parameters`, and `kan_head.learnable_parameters` all match
  it literally. `k_d:` in YAML is silently ignored (see the
  unknown-keys gotcha above) and you get the default `[1e-8, 1e-6]`.
- **There is no `use_precip` key.** The disaggregation head is always
  precip-driven; what switches it on is the *presence* of the
  `kan_head.disaggregation:` block, and that in turn makes
  `data_sources.aorc_precip` mandatory at dataset-open time. Any doc or
  config snippet with `use_precip:` is stale — serde drops it silently.
- **`leakance_losing_only` defaults to `true`.** Unlike every other
  leakance key it is a guard that is ON by default. Omitting it does not
  reproduce pre-Phase-C behavior; you must write
  `leakance_losing_only: false` for that.
- **A `parameter_ranges` entry for a parameter the head doesn't emit is
  dead config.** `x_storage` falls back to a constant `0.3` and the
  leakance trio falls back to the non-leakance routing path. Setting the
  range without adding the name to `kan_head.learnable_parameters`
  changes nothing.

### Verification

```bash
cargo test --lib config::
```

Covers the critical assertions:

| Test | Locks |
|---|---|
| `loads_merit_training_yaml` | YAML round-trip, every default in `params`, the `kan_head` section, top-level `seed`/`mode`/`workflow`/`device`; also asserts the shipped config has **no** adjacency keys |
| `default_config_still_constructs` | `Config::default()` keeps working for the routing-only path |
| `testing_mode_overlays_apply_to_experiment` | Testing overlay copies fields and clears `rho` |
| `training_mode_does_not_apply_overlays` | Training mode leaves `experiment` untouched |
| `mode_workflow_conflict_rejected`, `mode_testing_with_train_workflow_rejected` | mode/workflow cross-validation |
| `both_adjacency_paths_valid`, `fabric_only_valid`, `both_adjacency_and_fabric_valid`, `neither_adjacency_nor_fabric_rejected`, `partial_adjacency_conus_only_rejected`, `partial_adjacency_gages_only_rejected`, `gpkg_fabric_with_layer_valid`, `layer_without_gpkg_fabric_rejected`, `no_data_sources_section_valid` | the adjacency / fabric validation matrix |
| `attributes_bare_path_parses_as_single_element_vec`, `attributes_yaml_list_parses_as_multi_element_vec`, `attributes_empty_list_rejected` | `attributes` one-or-many deserializer |
| `loss_config_defaults_to_l1`, `loss_config_parses_nnse_kge_kebab_case` | `experiment.loss` defaults and kebab-case sub-keys |
| `state_cache_absent_yields_none`, `state_cache_path_parses` | `experiment.state_cache` |
| `use_leakance_defaults_false`, `leakance_flag_and_ranges_parse`, `leakance_losing_only_defaults_true`, `leakance_losing_only_parses_false`, `leakance_losing_only_absent_defaults_true`, `leakance_impervious_threshold_defaults_to_0_7`, `leakance_impervious_threshold_parses`, `leakance_with_cuda_graphs_rejected` | the leakance toggles and their load-time rejection |
| `disagg_freeze_without_pretrained_checkpoint_fails_to_load`, `disagg_pretrained_fields_default_to_none_and_false` | `kan_head.disaggregation` pretrain/freeze rules |

`cargo test --lib config::` runs 35 tests on this commit.

If a new YAML key is added, extend `loads_merit_training_yaml` with an
explicit assertion — silent serde defaults are the gotcha above.

## See also

- [Reading inputs](inputs-reading.md) — what the `data_sources:` paths
  point at and how each is read.
- [Running the code](running.md) — how `--config` and `ddrs plan/run`
  wire the YAML through the CLI.
- [Setup](../setup.md) — the canonical data-source paths and how to flip
  the CUDA defaults to CPU.
- [Algorithm](../algorithm.md) — why every key in `attribute_minimums`
  matters.
- [Performance & CUDA Graphs](../reference/perf.md) — what the
  `sparse_solver` and `use_cuda_graphs` toggles actually do.
