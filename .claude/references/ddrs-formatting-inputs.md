---
name: ddrs-formatting-inputs
description: How to write or modify ddrs's config YAML — parameter ranges, attribute minimums, log_space_parameters, sparse_solver and use_cuda_graphs toggles, kan_head head config, experiment/seed/mode/workflow/device top-levels.
output: usage/inputs-formatting.md
sources:
  - src/config.rs
  - config/merit_training.yaml
---

# ddrs-formatting-inputs

> Canonical agent-readable skill. Published chapter at `docs/usage/inputs-formatting.md`
> is regenerated from this file by `/regenerate-docs`.

## What to know

ddrs configs are YAML to mirror DDR's tooling (`~/projects/ddr/config/
merit_training_config.yaml` is the upstream reference). They are loaded
via `Config::from_yaml_file(path)` (back-compat, training mode) or
`Config::from_yaml_file_with_mode(path, ConfigMode::Testing)` in
`src/config.rs`. Deserialization runs through `serde_yaml` into a
`ConfigRaw` intermediate and then into the public `Config` via
`From<ConfigRaw>`. Every optional field has a default, so
`Config::default()` still constructs for call sites that only need the
solver core (the V1 sandbox example does this).

## Top-level structure

```
mode: training              # str, "training" or "testing"
workflow: train-and-test    # optional enum, cross-validated against mode
geodataset: merit           # str, dataset family name
device: 0                   # usize, CUDA device ordinal
seed: 42                    # u64, Rust-side RNG seed
np_seed: 42                 # u64, mirrors DDR's numpy seed
data_sources: { ... }       # required for the SP-3 dataloader
experiment: { ... }         # required for the SP-3 dataloader
kan_head:   { ... }         # KAN routing-head shape (alias: `mlp`)
params:     { ... }         # required for the routing core
testing:    { ... }         # optional overlay; applied when mode==testing
```

| Key | Type | Role |
|---|---|---|
| `mode` | string | Selects the run mode; `Config::from_yaml_file` ignores it, the binary reads it as the default for `--mode`. Defaults to `training`. |
| `workflow` | enum (optional) | `train`, `eval`, or `train-and-test` (kebab-case). Cross-validated against `mode`: training implies `train`/`train-and-test`, testing implies `eval`; a mismatch is a load-time `DataError::Yaml`. Absent → `None`. |
| `geodataset` | string | Free-form dataset tag (`merit` for the CONUS adjacency set). Defaults to `merit`. |
| `device` | usize | CUDA device ordinal, mirrors DDR's `device:` key (`device: 2` → `cuda:2`). Defaults to `0`. |
| `seed`, `np_seed` | u64 | Two seeds — DDR draws both because numpy and torch RNGs are seeded independently. Both default to `42`. |
| `data_sources` | section | Five `PathBuf` fields + a gauges CSV; see `ddrs-reading-inputs` for what each path feeds. No defaults — required to construct the dataset. |
| `experiment` | section | Training schedule. `batch_size`, `start_time`, `end_time`, `epochs`, optional `rho`, `shuffle`, `warmup`, `learning_rate` map, optional `grad_clip_max_norm`, optional `checkpoint`. |
| `kan_head` | section | KAN head shape: `hidden_size`, `num_hidden_layers`, `grid`, `k`, `input_var_names`, `learnable_parameters`. Accepts the legacy key `mlp` as a serde alias. |
| `params` | section | Routing engine knobs (next section). |
| `testing` | section | Overlay applied to `experiment` when `mode == Testing`. |

The defining types are in `src/config.rs`: `Config` (`:268-282`),
`ConfigRaw` (`:312-329`), and `From<ConfigRaw>` (`:331-346`) for the
root, plus the section structs `DataSources`, `Experiment`,
`KanHeadConfigSection`, and `Params`.

## `kan_head` section

`KanHeadConfigSection` (`src/config.rs:91-103`) configures the KAN
routing-head shape. The YAML key is `kan_head:`; the legacy key `mlp:` is
accepted as a serde alias (`#[serde(alias = "mlp")]` on
`ConfigRaw::kan_head`, `src/config.rs:326-327`) so older configs still
parse. Prefer `kan_head:` in new files.

| Key | Type | Default | Role |
|---|---|---|---|
| `hidden_size` | usize | required | KAN hidden width |
| `num_hidden_layers` | usize | required | Number of inner `KanLayer` blocks |
| `grid` | usize | `5` | B-spline grid intervals (`num` in pykan); merit YAML sets `50` |
| `k` | usize | `3` | B-spline order; merit YAML sets `2` (DDR's production override) |
| `input_var_names` | `Vec<String>` | required | Attribute columns fed to the head |
| `learnable_parameters` | `Vec<String>` | required | Routing parameters the head produces |

`grid` and `k` default via `default_grid`/`default_k`
(`src/config.rs:105-110`) when absent.

## `params` section

`Params` is the routing-core configuration (`src/config.rs:122-150`).
YAML enters via `ParamsRaw` (`src/config.rs:156-166`) and is folded into
the typed `Params` by `From<ParamsRaw>` (`src/config.rs:168-215`).

### `parameter_ranges`

Physical `[min, max]` ranges used to denormalize the NN's `[0,1]` outputs
into real channel-routing parameters.

| Key | YAML shape | Default | Used by |
|---|---|---|---|
| `n` | `[min, max]` f32 pair | `[0.015, 0.25]` | Manning's roughness |
| `q_spatial` | `[min, max]` f32 pair | `[0.0, 1.0]` | Discharge spatial term |
| `p_spatial` | `[min, max]` f32 pair | `[1.0, 200.0]` | Pressure spatial term |

YAML is a dict-of-2-tuples (`HashMap<String, [f32; 2]>`); the parse
block reads only the three known keys.

### `attribute_minimums`

Physical lower bounds clamped during routing to keep the math stable.

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
Both the Rust default and the merit YAML are `["p_spatial"]`
(`src/config.rs:189`, `config/merit_training.yaml:91-92`).

If the YAML list is non-empty it **replaces** the default entirely;
otherwise the default survives (`src/config.rs:201-203`).

### `defaults`

A `HashMap<String, f32>` of fallback values for parameters not produced
by the NN head. Merit YAML sets `p_spatial: 21.0`. As with
`log_space_parameters`, non-empty overrides the default
(`src/config.rs:198-200`).

### Solver toggles

| Key | Type | YAML default | Rust default | Effect |
|---|---|---|---|---|
| `tau` | u32 | unset → 3 | 3 | Hours-per-substep when subdividing the daily forcing |
| `sparse_solver` | `"cpu"` \| `"cuda"` | `cuda` | `Cpu` | Picks the CSR triangular solve backend |
| `use_cuda_graphs` | bool | `true` | `false` | Enables per-timestep CUDA-graph capture+replay |

Parsing of `sparse_solver` accepts both lower and upper case; anything
else panics (`src/config.rs:205-209`). `use_cuda_graphs` silently has no
effect on the CPU path.

## Defaults

The YAML default in `config/merit_training.yaml` is **CUDA-on**:

```yaml
params:
  sparse_solver: cuda    # SP-9 (commit dbcf6e6) — was cpu before
  use_cuda_graphs: true  # SP-10 (commit e35af29) — was false before
```

The Rust-side `Params::default()` is still `Cpu` + `use_cuda_graphs: false`
(`src/config.rs:140-149`), because the routing solver constructs a sensible
default without a YAML — but every code path that loads
`merit_training.yaml` opts into the GPU.

CPU-only override is one line each:

```yaml
params:
  sparse_solver: cpu
  use_cuda_graphs: false
```

## Adding a new parameter

Three coordinated edits in `src/config.rs`. Example: adding a
`enable_foo: bool` toggle to `params`.

1. **Extend `Params` + default** (`src/config.rs:122-150`):

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

2. **Add an `Option<T>` to `ParamsRaw`** (`src/config.rs:156-166`):

   ```rust
   struct ParamsRaw {
       // ...existing fields...
       enable_foo: Option<bool>,
   }
   ```

3. **Wire it into the parse block** (`src/config.rs:168-215`):

   ```rust
   if let Some(b) = r.enable_foo {
       p.enable_foo = b;
   }
   ```

Then add an assertion to `loads_merit_training_yaml`
(`src/config.rs:339-368`) so the default behavior is locked.

For root-level fields the pattern is the same but in `Config`,
`ConfigRaw`, and the `From<ConfigRaw>` block (`src/config.rs:275-288`).

## Gotchas

- **Unknown YAML keys are silently dropped.** `ParamsRaw` and `ConfigRaw`
  both use `#[serde(default)]` and do *not* use
  `#[serde(deny_unknown_fields)]`. A typo (`use_cuda_graph` instead of
  `use_cuda_graphs`) compiles, runs, and silently uses the default.
  Watch for this when editing YAML and check the `loads_merit_training_yaml`
  assertions match what you wrote.
- **`log_space_parameters` entries are bare strings.** A typo
  (`p_spatail` for `p_spatial`) parses fine and silently changes the
  denorm formula for whatever matched. There's no compile-time check; the
  only guard is the merit YAML test asserting the exact list — currently
  `["p_spatial"]` (`src/config.rs:506`).
- **YAML defaults moved across SPs.** `sparse_solver` flipped to `cuda`
  in SP-9 (commit `dbcf6e6`); `use_cuda_graphs` flipped to `true` in
  SP-10 (commit `e35af29`). Don't hard-code the assumption that either
  is `false` in tests — read the YAML or set them explicitly.
- **`testing.batch_size` semantically shifts.** In `experiment` it's the
  number of *gauges* per mini-batch; in `testing` it's the number of
  *days* per chunk. The overlay copies the value verbatim, so the unit
  change is invisible — the YAML comment is your only warning
  (`config/merit_training.yaml:79-86`).
- **`testing.rho: null` is distinct from absent.** Custom serde shim
  (`src/config.rs:253-258`) lets `null` explicitly clear `rho`; leaving
  the key out preserves the training-side value.
- **`sparse_solver` rejects unknowns with a panic, not an error.**
  Typos like `sparse_solver: gpu` crash with
  `unknown sparse_solver: "gpu"` (`src/config.rs:208`) — not a clean
  `DataError::Yaml`. Don't hand this YAML to an end user uninspected.

## Verification

```bash
cargo test --lib config::
```

Covers the four critical assertions:

| Test | Locks |
|---|---|
| `loads_merit_training_yaml` | YAML round-trip, every default in `params`, `kan_head` section, top-level seed/mode/workflow/device |
| `default_config_still_constructs` | `Config::default()` keeps working for the routing-only path |
| `testing_mode_overlays_apply_to_experiment` | Testing overlay copies fields and clears `rho` |
| `training_mode_does_not_apply_overlays` | Training mode leaves `experiment` untouched |

If a new YAML key is added, extend `loads_merit_training_yaml` with an
explicit assertion — silent serde defaults are the gotcha above.
