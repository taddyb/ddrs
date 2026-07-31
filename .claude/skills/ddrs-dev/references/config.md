# Config reference

Struct: `src/config.rs::Config`. Loaded via
`Config::from_yaml_file_with_mode(path, ConfigMode::Training|Testing)`.
Six top-level sections. Verified against source 2026-07-30.

**No `deny_unknown_fields` anywhere** — a typo'd key silently takes its default
instead of erroring. This is the single most common cause of "my config change did
nothing".

## Top level

| Key | Default | Notes |
|---|---|---|
| `mode` | `training` | `training` \| `testing` |
| `workflow` | none | Cross-validated against `mode`: `training` ↔ `{train, train-and-test}`, `testing` ↔ `eval` |
| `geodataset` | `merit` | |
| `device` | `0` | CUDA ordinal |
| `seed` / `np_seed` | `42` / `42` | |

`kan_head:` accepts `mlp:` as a serde alias (backward compat with pre-KAN configs).

## `data_sources:` — 8 path fields

`attributes` is a `Vec<PathBuf>` accepting a **single path or a list**
(feature-concatenated on COMID, NaN-filled, `deserialize_one_or_many_paths`).
An empty list is a hard error.

Adjacency rule: provide **either** both `conus_adjacency` + `gages_adjacency`,
**or** `geospatial_fabric` (managed build into `.ddrs/adjacency/<key>/`).
Exactly one of the pair ⇒ error; neither source ⇒ `"adjacency sources are missing"`.
For multi-layer gpkg set `geospatial_fabric_layer` (participates in the cache key).

`aorc_precip` is required whenever `kan_head.disaggregation:` is present — see below.

## `experiment:`

| Key | Production value | Notes |
|---|---|---|
| `batch_size` | 64 | **GAUGES** in training |
| `start_time` / `end_time` | 1981/10/01 – 1995/09/30 | |
| `epochs` | 5 | Resume requires raising this past the checkpoint's epoch or zero batches train |
| `rho` | 90 | Training window length in days |
| `warmup` | 5 | |
| `shuffle` | true | |
| `learning_rate` | `{1: 0.001, 3: 0.0005}` | Epoch→lr schedule. **Ignored under `optimizer: adadelta`** — AdaDelta derives its step from RMS[Δx]/RMS[g] |
| `grad_clip_max_norm` | 1.0 | |
| `checkpoint` | none | Directory `epoch_E_mb_M/` holding `head.mpk`, `optim.mpk`, `state.json` |
| `state_cache` | none | **netCDF** (not zarr) day-boundary discharge state, from `probe_zeta_gradient --mode state-cache` |
| `optimizer` | `adam` | `adam` \| `adadelta`. A checkpoint refuses to load across kinds rather than reinterpreting moment tensors |
| `use_grad_accum` | false | Master switch for optimizer micro-batching |
| `grad_accum_steps` | none | Micro-batches accumulated into one optimizer step, with exact valid-count weighting |

> `optimizer` / `use_grad_accum` / `grad_accum_steps` / `loss.kind: nse-batch` land
> with the gradient-accumulation work (PR #31, branch `exp_train`). If you are on a
> commit without them, `nse-batch` fails config load with
> `unknown variant 'nse-batch'`.

Under `use_grad_accum: true`, `--max-mini-batches` counts optimizer **STEPS**, not
mini-batches.

### `experiment.loss:`

`LossConfig::default()`: `kind: l1`, `nnse_weight 1.0`, `kge_weight 1.0`,
`r_weight 1.0`, `alpha_weight 1.0`, `beta_weight 1.0`, `kge_clamp 10.0`, `eps 0.1`.
Sub-keys are kebab-case in YAML.

| `kind` | Objective |
|---|---|
| `l1` | `mean(\|p − o\|)`. Historical default |
| `nnse-kge` | Per-gauge `nnse_weight·(1−NNSE) + kge_weight·(1−KGE)` |
| `kge` | KGE term alone |
| `nse-batch` | dHBV `NSELossBatch`: mean over valid (day, gauge) of `(sim−obs)²/(σ_gauge+eps)²`, σ fixed over the training period |

`kge_clamp` exists because a single near-constant gauge once drove batch loss to
~1e4.

Why the loss menu exists: L1 and NSE are both maximized at a simulated variance
*below* observed (NSE's optimum is at `α = r < 1`), so they reward the router for
over-attenuating peaks. Note this did **not** turn out to be the binding constraint —
see `research-status.md`.

## `testing:` overlay

Replaces matching `experiment:` keys; absent keys inherit. Covers `start_time`,
`end_time`, `batch_size`, `rho`, `warmup`, `epochs`, `grad_clip_max_norm`,
`checkpoint`.

- **`batch_size` semantically shifts**: GAUGES in training, **DAYS** in testing.
- `rho: null` is a *double-Option* (`deserialize_option_option`) — "present and null"
  (disabled) is distinct from "absent" (inherit).

## `kan_head:`

Code defaults `grid: 5`, `k: 3`; production overrides to `grid: 50`, `k: 2` for DDR
parity. `hidden_size: 21`, `num_hidden_layers: 2`.

Ten production `input_var_names`: `SoilGrids1km_clay`, `aridity`, `meanelevation`,
`meanP`, `NDVI`, `meanslope`, `log10_uparea`, `SoilGrids1km_sand`, `ETPOT_Hargr`,
`Porosity`.

### `kan_head.disaggregation:` — the real fields

> **There is no `use_precip`, `use_attributes`, or `use_temp` key.** They were
> removed in `334f0fe` ("rework disaggregation head to KAN + basin-normalized
> precip"). CLAUDE.md, `src/config.rs:113`'s comment, and
> `config/sources/conus-hourly.yaml:5` all still reference the phantom key.
> **Current contract:** presence of the `disaggregation:` block ⇒ the head always
> consumes precip ⇒ `data_sources.aorc_precip` is mandatory, else
> `MeritGagesDataset::open` errors. It cannot silently degrade to flat repeat-24.

| Key | Default | Notes |
|---|---|---|
| `hidden_size` | 16 | |
| `num_hidden_layers` | 1 | |
| `grid` | 3 | |
| `k` | 3 | |
| `boundary_blend` | 0.0 | Day-boundary shape continuity λ; only used when `chunk_days <= 1` |
| `chunk_days` | 1 | `> 1` relaxes mass balance to the chunk aggregate, letting storms span day boundaries |
| `pretrained_checkpoint` | none | CompactRecorder `.mpk`. The five architecture fields above MUST match or `load_record` fails loudly |
| `freeze` | false | `Module::no_grad()`. Requires `pretrained_checkpoint` |

## `params:`

| Key | Default | Notes |
|---|---|---|
| `sparse_solver` | `cpu` | On a non-CUDA backend, `cuda` **silently WARN-falls-back** to cpu. An unrecognized value **panics** |
| `use_cuda_graphs` | **false** | `config/merit_training.yaml` *sets* true — that is a YAML value, not the code default |
| `use_leakance` | false | |
| `leakance_losing_only` | **true** | Clamps `head = max(0, depth − d_gw)`, so gaining reaches produce `zeta ≡ 0` |
| `leakance_impervious_threshold` | 0.7 | Masks reaches whose `corridor_impervious` is **`>`** this value (not `≥`) |
| `tau` | 3 | **Not** a routing sub-step count. It is the hourly→daily trim phase offset in `tau_trim_and_downsample`: DDR's slice `[13 + tau : -11 + tau]`, then area-pool to days (`src/training/loss.rs:17-45`). Nothing in `src/routing/` reads it |
| `log_space_parameters` | `["p_spatial"]` | |
| `defaults` | `{p_spatial: 21.0}` | Value used when a parameter is not in `learnable_parameters` |

### `parameter_ranges` (7 keys parsed)

`n [0.015, 0.25]`, `q_spatial [0, 1]`, `p_spatial [1, 200]`, `x_storage [0, 0.5]`,
`k_d [1e-8, 1e-6]`, `d_gw [-2, 2]`, `leakance_factor [0, 1]`.

Case quirk: **`K_D` is uppercase in YAML, `k_d` in Rust.** `x_storage` is only
consumed when listed in `learnable_parameters`; otherwise routing uses a constant 0.3.
`p_spatial` is the Leopold-Maddock **coefficient**; `q_spatial` is the exponent
(`docs/algorithm.md` has this backwards).

### `attribute_minimums`

`discharge 1e-4`, `slope 1e-3`, `velocity 0.01`, `depth 0.01`, `bottom_width 0.01`.

## Load-time guards

Four validators run at `Config::from_yaml_file`, plus one at dataset open.

| Guard | Rejects | Error substring |
|---|---|---|
| `validate_mode_workflow` | `mode`/`workflow` disagreement | `"conflicting top-level keys"` |
| `validate_data_sources` | one of the adjacency pair | `` "`gages_adjacency` is missing" `` |
| | neither adjacency nor fabric | `"adjacency sources are missing"` |
| | `geospatial_fabric_layer` on a non-gpkg | `"geospatial_fabric_layer"` + `".gpkg"` |
| `validate_leakance` | `use_leakance` + `use_cuda_graphs` | both key names |
| `validate_disagg_pretrained` | `freeze: true` without `pretrained_checkpoint` | `"freeze: true requires pretrained_checkpoint"` |
| `validate_grad_accum` | `grad_accum_steps: 0` | `"grad_accum_steps: 0"` |
| | `use_grad_accum: true` with steps < 2 | `"requires grad_accum_steps: N with N >= 2"` |
| `validate_disagg_vs_resolution` (runtime, `src/data/dataset.rs`) | `disaggregation:` + hourly-native streamflow store | hard error |

## Adding a new routing parameter

1. Add the range to `ParameterRanges` (`src/config.rs`) and to
   `config/merit_training.yaml`'s `params.parameter_ranges`.
2. Add it to `kan_head.learnable_parameters` in the experiment config so the head
   emits it.
3. Thread it through `denormalize` (`src/routing/utils.rs`) and
   `SpatialParameters` (`src/routing/mmc.rs`).
4. If it enters the timestep, add it as a `Backward` parent and widen the op's `N`.
5. Add a gradcheck case (`references/testing.md`) and an OFF-parity test.
6. Update this file and `config/merit_training.yaml`'s comments.

## Adding a new boolean flag

1. Field on `Params` with `#[serde(default)]` and an explicit `Default` impl —
   absent must be byte-identical to the old behavior.
2. If it is incompatible with another flag, add a validator and a test asserting the
   error substring.
3. Add an OFF-parity test that asserts **bit-exact** equality with the pre-feature
   expected array, and an ON test that asserts the output actually changes. Only the
   second catches a silent no-op.

## Leakance: enabling it

Three changes are required together — `params.use_leakance: true` (which forces
`use_cuda_graphs: false`), `K_D`/`d_gw`/`leakance_factor` in
`kan_head.learnable_parameters`, and matching `parameter_ranges`. The term is
CLOSED (NO-GO) as a research direction but remains code-complete and gradient-exact;
see `research-status.md` before touching it.
