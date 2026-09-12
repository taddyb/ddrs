# Config reference

Struct: `src/config.rs::Config`. Loaded via
`Config::from_yaml_file_with_mode(path, ConfigMode::Training|Testing)`.
Six top-level sections. Verified against source 2026-07-30.

**Every section sets `deny_unknown_fields`** as of 2026-09-09 (`ConfigRaw`,
`DataSources`, `Experiment`, `KanHeadConfigSection`, `LossConfig`, `ParamsRaw`,
`TestingOverridesRaw`, plus `DisaggregationSection` and `Subdivision`, which had it
already), so a typo'd key is a load error naming the key. Historically only the last
two had it and everywhere else a typo silently took its default, which was the
single most common cause of "my config change did nothing". On a binary older than
2026-09-09, that silence is still the first thing to suspect.

## Contents

§Top level · §`data_sources:` · §`experiment:` → §`experiment.loss:` ·
§`testing:` overlay · §`kan_head:` → §`kan_head.disaggregation:` ·
§`params:` → §`parameter_ranges` → §`attribute_minimums` · §Load-time guards ·
§Adding a new routing parameter · §Adding a new boolean flag · §Leakance: enabling it

Jump straight to §Load-time guards if a config was **rejected**; to
§`kan_head.disaggregation:` if anything mentions `use_precip` (it does not exist);
to §`params:` if you are looking for `tau` (it is not a routing sub-step count).

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

Adjacency rule: provide **exactly one** of (a) both `conus_adjacency` +
`gages_adjacency`, (b) `geospatial_fabric` (MERIT managed build into
`.ddrs/adjacency/<key>/`), (c) `gridded_network` (DDR's DDM30 sub-reach
adjacency zarr, managed build via `resolve_or_build_gridded`; added 2026-09-09).
Exactly one of the pair ⇒ error; neither source ⇒ `"adjacency sources are missing"`;
`gridded_network` with (a) or (b) ⇒ error. For multi-layer gpkg set
`geospatial_fabric_layer` (participates in the cache key). `params.subdivision.enabled`
is rejected alongside `gridded_network` (the store is already split by DDR).

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

Under `use_grad_accum: true`, `--max-mini-batches` counts optimizer **STEPS**, not
mini-batches.

### `experiment.loss:`

`LossConfig::default()`: `kind: l1`, `nnse_weight 1.0`, `kge_weight 1.0`,
`r_weight 1.0`, `alpha_weight 1.0`, `beta_weight 1.0`, `kge_clamp 10.0`, `eps 0.1`,
`deriv_weight 0.5`. Sub-keys are kebab-case in YAML (`deriv-weight:`).

| `kind` | Objective |
|---|---|
| `l1` | `mean(\|p − o\|)`. Historical default |
| `nnse-kge` | Per-gauge `nnse_weight·(1−NNSE) + kge_weight·(1−KGE)` |
| `kge` | KGE term alone |
| `nse-batch` | dHBV `NSELossBatch`: mean over valid (day, gauge) of `(sim−obs)²/(σ_gauge+eps)²`, σ fixed over the training period |
| `nse-batch-deriv` | `nse-batch` + `deriv_weight` × the same mean over ADJACENT valid (day, gauge) pairs of `((dsim−dobs)²/(σ_d,gauge+eps)²)`, `σ_d` = per-gauge observed consecutive-day-difference std, fixed over the training period |

`nse-batch-deriv` exists because the landscape curvature probe measured the loss
curvature in Manning's `n` to be governed by the mean square of the hydrograph's
TIME DERIVATIVE (findings §25); §26 predicted a derivative term deepens that
valley ~3.7× at `deriv-weight: 0.5` (the default). `deriv-weight: 0.0` reduces it
to exactly `nse-batch` (guarded by
`nse_batch_deriv_at_zero_weight_is_identical_to_nse_batch`); a negative or
non-finite weight is a config-load error. It is the trainable twin of the
LANDSCAPE study's `nse-deriv` measurement objective — same adjacent-pair rule
(a gap BREAKS the chain), same population `σ_d`. It costs one extra full-window
observation read at open (`MeritGagesDataset::gauge_obs_diff_std`, gated so no
other kind pays for it).

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
> precip").
> **Current contract:** presence of the `disaggregation:` block ⇒ the head always
> consumes precip ⇒ `data_sources.aorc_precip` is mandatory, else
> `MeritGagesDataset::open` errors. It cannot silently degrade to flat repeat-24.
> **Exception (2026-08-03):** `disaggregation.enabled: false` strips the block at
> load time, making it inert — the sanctioned way to A/B the head vs nearest
> (repeat-24) without deleting the block. (This replaced the short-lived
> `experiment.use_frozen_kan_head`, which never ran an experiment.)
> The section is `#[serde(deny_unknown_fields)]` (2026-08-03): phantom keys like
> `use_precip` now FAIL LOAD with "unknown field" instead of silently taking
> defaults — the one section where a typo'd key cannot silently no-op.

| Key | Default | Notes |
|---|---|---|
| `enabled` | **true** | `false` ⇒ block stripped at load ⇒ flat repeat-24 (nearest) upsampling; the one-line ablation switch. `tests/disagg_enabled.rs` |
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
| `use_cuda_graphs` | false | Requires the DEPRECATED `ddr_match: true` (the captured kernel hardcodes the legacy 5/3 celerity); rejected alongside the corrected-physics default. `config/merit_training.yaml` set it true until 2026-08-19 |
| `ddr_match` | **false** (since 2026-08-19) | DEPRECATED. `true` = legacy pre-#192 DDR physics (5/3 celerity, X ≡ 0.3, upstream-cols readout) — parses with a WARN, kept only for pre-#192 reproduction and CUDA graphs. DDR itself runs the corrected physics since DeepGroundwater/ddr#192. See `.claude/PHYSICS-CORRECTIONS.md` |
| `use_leakance` | false | |
| `leakance_losing_only` | **true** | Clamps `head = max(0, depth − d_gw)`, so gaining reaches produce `zeta ≡ 0` |
| `leakance_impervious_threshold` | 0.7 | Masks reaches whose `corridor_impervious` is **`>`** this value (not `≥`) |
| `tau` | 9 | **Not** a routing sub-step count. Since 2026-08-08: hours the routed output is ADVANCED before daily scoring (translation-only inverse routing, dMC-Juniata's sign). Slice `[tau : -(24-tau)]`, pooled day i ↔ obs day i, valid range [0, 24) (`src/training/loss.rs`). Nothing in `src/routing/` reads it. Default 9 = the measured CONUS optimum (findings §5g). **Legacy scale (pre-2026-08-08): old = new + 11**, slice `[13+tau : -11+tau]`, day i ↔ obs day i+1; DDR-Python still uses it, all older configs/checkpoints carry it (old shipped 3 ≡ new −8, wrong direction). Never copy a `tau:` value across the convention boundary. |
| `log_space_parameters` | `["p_spatial"]` | |
| `defaults` | `{p_spatial: 21.0}` | Value used when a parameter is not in `learnable_parameters` |

### `parameter_ranges` (7 keys parsed)

`n [0.015, 0.25]`, `q_spatial [0, 1]`, `p_spatial [1, 200]`, `x_storage [0, 0.5]`,
`k_d [1e-8, 1e-6]`, `d_gw [-2, 2]`, `leakance_factor [0, 1]`.

Case quirk: **`K_D` is uppercase in YAML, `k_d` in Rust.** `x_storage` is only
consumed when listed in `learnable_parameters`; otherwise routing uses a constant 0.3.
`p_spatial` is the Leopold-Maddock **coefficient**; `q_spatial` is the exponent
(`docs/book/algorithm.md` has this backwards).

### `attribute_minimums`

`discharge 1e-4`, `slope 1e-3`, `velocity 0.01`, `depth 0.01`, `bottom_width 0.01`.

## Load-time guards

Ten validators run directly at `Config::from_yaml_file`, plus a nested
`validate_subdivision_reaches_the_builder` (called from `validate_subdivision`
when subdivision is enabled), plus one at dataset open.

| Guard | Rejects | Error substring |
|---|---|---|
| `validate_mode_workflow` | `mode`/`workflow` disagreement | `"conflicting top-level keys"` |
| `validate_data_sources` | one of the adjacency pair | `` "`gages_adjacency` is missing" `` |
| | neither adjacency nor fabric | `"adjacency sources are missing"` |
| | `geospatial_fabric_layer` on a non-gpkg | `"geospatial_fabric_layer"` + `".gpkg"` |
| | `gridded_network` + `geospatial_fabric` | `"gridded_network"` + `"geospatial_fabric"` |
| | `gridded_network` + explicit adjacency pair | `"gridded_network"` + `"conus_adjacency"` |
| `validate_subdivision` | `subdivision.enabled: true` + `gridded_network` | `"params.subdivision"` + `"gridded_network"` |
| | nested `validate_subdivision_reaches_the_builder`: `enabled: true` + explicit `conus_adjacency`/`gages_adjacency` pointing at a non-subdivided store | `"params.subdivision"` + `"conflicts with the explicit"` |
| `validate_geodataset` | `geodataset:` contradicting the adjacency source (`ddm30` with `geospatial_fabric`, `merit` with `gridded_network`) | `"geodataset"` + the source key. Absent ⇒ inferred; explicit adjacency paths ⇒ any label allowed |
| `validate_leakance` | `use_leakance` + `use_cuda_graphs` | both key names |
| `validate_ddr_match` | `use_cuda_graphs: true` without the deprecated `ddr_match: true` | `"use_cuda_graphs: true` requires the DEPRECATED `ddr_match: true"` |
| `validate_enforce_positivity` | `enforce_positivity: true` + `ddr_match: true` | `"requires \`ddr_match: false\`"` |
| `validate_disagg_pretrained` | `freeze: true` without `pretrained_checkpoint` | `"freeze: true requires pretrained_checkpoint"` |
| `validate_grad_accum` | `grad_accum_steps: 0` | `"grad_accum_steps: 0"` |
| | `use_grad_accum: true` with steps < 2 | `"requires grad_accum_steps: N with N >= 2"` |
| `validate_loss` | `loss.deriv-weight` non-finite or negative | `"deriv-weight"` |
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

To produce the `zeta` / `zeta_net` eval diagnostic for an EXISTING checkpoint
without retraining (`ddrs run --workflow train-and-test` writes it automatically
in Phase 2, so this is only for a checkpoint from an earlier run):

```bash
cargo build --release --bin eval
target/release/eval --config config/experiments/leakance_hourly_on.yaml \
  --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
  --output /tmp/eval.zarr \
  --zeta-output .ddrs/runs/<id>/kan_parameters.nc
```

Takes about 10 minutes, no retrain.
