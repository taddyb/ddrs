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
§Adding a new routing parameter · §Adding a new boolean flag · §Leakance: enabling it ·
§Reservoirs

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

## `data_sources:` — 10 path fields

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

`reservoirs` (added 2026-09-25) is the reservoir table CSV for `params.use_reservoirs`:
header `COMID,T_days`, extra columns allowed and ignored. Required when
`use_reservoirs: true`, ignored otherwise (but still fingerprinted into `sources.lock`
when set). See §Reservoirs.

`pins:` (optional map, source name → icechunk snapshot id) opens a source at
`VersionInfo::SnapshotId` instead of the `main` branch tip, so a run is
reproducible after the store is appended to. Only the two icechunk-backed
sources — `streamflow` and `observations` (`config.rs::ICECHUNK_PINNABLE`) —
may be pinned; a pin on any other name is a load-time error
(`config.rs::validate_pins`), and a pin on a path that turns out to be the
zarr-v2 global product is refused at open (`store/mod.rs::reject_pin_on_non_icechunk`).
Omitting the block changes nothing: every source opens at the tip, exactly as
before. Either way the id actually read is recorded — dataset open logs
`streamflow snapshot: <id> (pinned|main tip)` next to `streamflow resolution:`,
and `ddrs plan` writes it to `sources.lock` and the run manifest as
`sources.<name>.snapshot` with `fp = icechunk:<id>`. Because the fingerprint IS
the snapshot id, appending to an unpinned store shows up as drift on the next
`plan` (one `field: locked <fp> -> current <fp>` line per source); a pinned
source never drifts.
Changing a pin also invalidates the cached summed-Q' baseline:
`baseline/cache.rs::cache_key` hashes the `pins` block after the window fields,
so re-pinning recomputes the baseline instead of scoring the model against a
reference built from a different snapshot. The block is hashed only when present
and non-empty, so an unpinned config keeps the key it always had and no existing
cache under `.ddrs/baselines/` is orphaned.

```yaml
data_sources:
  streamflow: /mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic
  observations: /mnt/ssd1/data/icechunk/usgs_daily_observations
  pins:
    streamflow: E0M3W6W1881H868V2KRG
```

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

### Head topology knobs (added 2026-09-11) — all default to the current head

These exist to test §31 of `research/findings/2026-09-08-landscape-hypothesis-tests-findings.md`:
the head's learnable outputs came out as one latent direction relabelled
(`logit(q) = 3.979 · logit(n) + 2.752`, R² = 0.987 over 346,321 reaches), which
is also why `q_spatial` saturated both bounds while `n` never touched its own.

| Key | Default | Meaning |
|---|---|---|
| `parameter_groups` | `[]` | Partition `learnable_parameters` into independently parameterized trunks, e.g. `[[n], [p_spatial, q_spatial]]`. Must cover every name exactly once. Group 0 keeps the config `seed`; later groups are offset. |
| `input_layer_kan` | `false` | `Linear(F,H)` → `KanLayer(F,H)`: per-attribute splines instead of a linear mixing applied before any nonlinearity. |
| `output_layer_kan` | `false` | `Linear(H,P)` → `KanLayer(H,P)`: each output gets its own spline coefficients per edge. |
| `kan_grid_range` | `[-3.0, 3.0]` | B-spline grid range for the two boundary KanLayers only. Inner trunk layers keep rskan's `[-1, 1]`. Widened because the boundary layers see z-scored attributes (±4) and unnormalised activations; outside the grid the spline term flattens and only `scale_base · SiLU` survives, which is affine and would quietly reinstate the coupling `output_layer_kan` exists to remove. |

**Both KanLayer knobs break DDR `kan.py` parity by construction** (invariant 5).
They are experiment arms, not defaults; a config that sets either one cannot be
compared to a DDR parity fixture.

**The mechanism, stated correctly.** A `Linear(H,P)` read-out does not force
affinity between outputs on its own — it does so only when `h` is effectively
rank 1 across reaches, which is the measured §31 regime. And `output_layer_kan`
breaks the *affinity* but not the *functional dependence*: two different
nonlinear functions of one scalar latent still move in lockstep (rank
correlation stays at 1). Decoupling the outputs needs a trunk that carries more
than one direction, which is what `parameter_groups` tests. Both halves are
pinned in `tests/kan_head_groups.rs`.

**Checkpoint compatibility.** burn's `Module` derive emits a serde record with
no `#[serde(default)]`, so these three fields broke every pre-2026-09-11
checkpoint with `missing field \`extra\``. `load_kan_head` now falls back to an
explicit `LegacyKanHead` mirror of the old layout, and refuses that fallback for
a split-trunk or KAN-boundary template (the old file carries no weights for
those, and leaving them at init would be worse than failing). Gate:

```bash
DDRS_LEGACY_HEAD_CKPT=.ddrs/runs/<pre-change-id>/checkpoints/epoch_9_mb_9 \
  cargo test --test kan_head_record_compat -- --nocapture
```

Remember this whenever you add a field to any `Module` — the same trap applies
to `DisaggHead` and to anything else with checkpoints on disk.

### Screening a head topology without training

`src/bin/head_arch_screen.rs` compares topologies in minutes rather than ~1.8 h
of CONUS training per arm, because the head is a pure per-reach function of the
attributes. It writes init fields over all of CONUS, the trunk activations, and
a supervised capacity control against two targets that are uncorrelated by
construction and exactly recoverable from the inputs.

```bash
cargo run --release --bin head_arch_screen -- \
  --config .ddrs/runs/<id>/config.yaml \
  --out-dir .ddrs/experiments/head-arch/<ts> --sample 4000 --fit-steps 400
experiments/head_arch/analyze.py .ddrs/experiments/head-arch/<ts>
```

Budget: roughly 40 s per arm per 200 Adam steps at 4,000 reaches on CPU, plus
~25 s to load attributes. `--sample 20000 --fit-steps 2000` is far too slow to
sweep seven arms; it is not more informative, because the targets are exact
functions of the inputs and the fit converges early.

`experiments/head_arch/attribute_rank.py <run-config.yaml>` needs no Rust at all
and bounds every topology from above.

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
| `use_reservoirs` | false | Route the COMIDs in `data_sources.reservoirs` as linear reservoirs `S = T·Q` (option C). `false` is bit-identical to no reservoir code. Rejected with leakance, CUDA graphs, `ddr_match`, subdivision, `gridded_network`; see §Reservoirs |
| `reservoir_release` | `fixed` | `fixed`: `T` (and optional seasonal `a`, `b`) from the table. `learned`: `(T0, a, b)` per dam from the `release_head:` block, trained jointly with the routing head. `learned` requires `use_reservoirs`, `kan_head`, `release_head`; see §Reservoirs |
| `leakance_losing_only` | **true** | Clamps `head = max(0, depth − d_gw)`, so gaining reaches produce `zeta ≡ 0` |
| `leakance_impervious_threshold` | 0.7 | Masks reaches whose `corridor_impervious` is **`>`** this value (not `≥`) |
| `leakance_gate` | absent | Temperature-annealed 0/1 gate on the head's `leakance_factor` output; see §Leakance gate. Absent ⇒ byte-identical to the ungated readers. Requires `use_leakance: true` |
| `tau` | 9 | **Not** a routing sub-step count. Since 2026-08-08: hours the routed output is ADVANCED before daily scoring (translation-only inverse routing, dMC-Juniata's sign). Slice `[tau : -(24-tau)]`, pooled day i ↔ obs day i, valid range [0, 24) (`src/training/loss.rs`). Nothing in `src/routing/` reads it. Default 9 = the measured CONUS optimum (findings §5g). **Legacy scale (pre-2026-08-08): old = new + 11**, slice `[13+tau : -11+tau]`, day i ↔ obs day i+1; DDR-Python still uses it, all older configs/checkpoints carry it (old shipped 3 ≡ new −8, wrong direction). Never copy a `tau:` value across the convention boundary. |
| `log_space_parameters` | `["p_spatial"]` | |
| `defaults` | `{p_spatial: 21.0}` | Value used when a parameter is not in `learnable_parameters` |

### `parameter_ranges` (11 keys parsed)

`n [0.015, 0.25]`, `q_spatial [0, 1]`, `p_spatial [1, 200]`, `x_storage [0, 0.5]`,
`k_d [1e-8, 1e-6]`, `d_gw [-2, 2]`, `leakance_factor [0, 1]`, `gamma [0, 0.5]`, and the learned
dam release's `reservoir_T0 [1/24, 365]` (days, always LOG space), `reservoir_a [-2, 2]`,
`reservoir_b [-2, 2]` (linear). Unknown keys are silently ignored (`parameter_ranges` is a map).

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
| `validate_pins` (called from `validate_data_sources`) | `pins:` naming anything but `streamflow`/`observations` | `"pins"` + the offending source name |
| `validate_subdivision` | `subdivision.enabled: true` + `gridded_network` | `"params.subdivision"` + `"gridded_network"` |
| | nested `validate_subdivision_reaches_the_builder`: `enabled: true` + explicit `conus_adjacency`/`gages_adjacency` pointing at a non-subdivided store | `"params.subdivision"` + `"conflicts with the explicit"` |
| `validate_geodataset` | `geodataset:` contradicting the adjacency source (`ddm30` with `geospatial_fabric`, `merit` with `gridded_network`) | `"geodataset"` + the source key. Absent ⇒ inferred; explicit adjacency paths ⇒ any label allowed |
| `validate_reservoirs` | `use_reservoirs: true` without `data_sources.reservoirs` | `"use_reservoirs"` + `"data_sources.reservoirs"` |
| | `use_reservoirs: true` + `use_leakance`, `use_cuda_graphs`, `ddr_match`, `subdivision.enabled` or `gridded_network` | `"use_reservoirs"` + the offending key. Runs before `validate_ddr_match`, so `use_cuda_graphs` reports the reservoir error first |
| `validate_leakance` | `use_leakance` + `use_cuda_graphs` | both key names |
| `validate_leakance_gate` | `leakance_gate` block without `use_leakance: true` | `"leakance_gate"` + `"use_leakance"` |
| | empty `leakance_gate.temperature` | `"temperature"` |
| | a temperature that is not positive and finite | `"epoch N"` + `"positive"` |
| `validate_ddr_match` | `use_cuda_graphs: true` without the deprecated `ddr_match: true` | `"use_cuda_graphs: true` requires the DEPRECATED `ddr_match: true"` |
| `validate_enforce_positivity` | `enforce_positivity: true` + `ddr_match: true` | `"requires \`ddr_match: false\`"` |
| `validate_disagg_pretrained` | `freeze: true` without `pretrained_checkpoint` | `"freeze: true requires pretrained_checkpoint"` |
| `validate_grad_accum` | `grad_accum_steps: 0` | `"grad_accum_steps: 0"` |
| | `use_grad_accum: true` with steps < 2 | `"requires grad_accum_steps: N with N >= 2"` |
| `validate_loss` | `loss.deriv-weight` non-finite or negative | `"deriv-weight"` |
| `validate_disagg_vs_resolution` (runtime, `src/data/dataset.rs`) | `disaggregation:` + hourly-native streamflow store | hard error |
| `read_reservoir_table` (runtime, dataset open) | a reservoir CSV without `COMID`/`T_days`, an unparseable row, `T_days` non-finite or `< 1/24`, a duplicate COMID, zero rows | `DataError` naming the CSV path |
| `validate_reservoirs` (learned release) | `release_head:` without `reservoir_release: learned`; `learned` without `use_reservoirs`, `release_head` or `kan_head`; empty `release_head.input_var_names`; `reservoir_T0` lower bound below 1/24 or inverted; `reservoir_a`/`_b` inverted | `"release_head"` / `"reservoir_release: learned"` + the missing key / the range name |
| `read_fixed_release_table` (runtime) | only one of `a`/`b`; non-finite `a`/`b` | `"`a` and `b`"` / `"row N"` |
| `validate_reservoirs` (flood pool) | `params.reservoir_flood_pool` with `learned` or without `use_reservoirs`; either pool key with the carried dam floor; an unknown `flood_pool` mode | `"reservoir_flood_pool"` / `"release_head.flood_pool"` + `"dam_floor: carry"` |
| dataset open / `read_fixed_release_table` (flood pool) | a pool without `inflow_mean_m3s`; `flood_control` without `purpose_flood`; `reservoir_flood_pool: true` without `kc, phi, z`; not all three of `kc, phi, z`; `kc < 0`, `phi` outside `[0, 1]`, `z < 0` | `"inflow_mean_m3s"` / `"purpose_flood"` / `"kc"` / `"all three"` |
| `read_dam_features` (runtime, `learned`) | a listed feature column missing, a non-finite value, a duplicate COMID, zero rows | the column name / `"row N"` / `"listed twice"` |

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

## Leakance gate (`params.leakance_gate`, added 2026-09-16)

`zeta = leakance_factor · area_z · K_D · (depth − d_gw)` multiplies
`leakance_factor` and `K_D`, so only their product reaches the physics: an
exact scaling degeneracy (measured `rho(leakance_factor, K_D) = +0.9986` over
346,321 CONUS reaches on a trained arm, with 97 % of the four outputs' variance
in one direction). Physically a reach either sits above a losing aquifer or it
does not, so `leakance_factor` is meant to be a SELECTOR of where leakance
acts, not a continuous multiplier. The gate makes it one without a
straight-through estimator: the head's normalized output `u ∈ (0, 1)` becomes

```text
g = sigmoid( logit(clamp(u, 1e-6, 1 − 1e-6)) / tau ),   logit(x) = ln(x / (1 − x))
```

`tau = 1` is the identity (`g = u`; the code returns `u` untouched, so it is
BIT-exact, not round-tripped through `ln`/`exp`). Smaller `tau` sharpens toward
a step at `u = 0.5`; `0.5` is a fixed point at every temperature. The clamp
margin `1e-6` sits ~17 f32 ulps below 1.0 so `1 − u` keeps ~3 % relative
accuracy at the edge, and it only alters values the head has already saturated
past `|pre-activation| > 13.8`. Autograd through `clamp → log → div → sigmoid`
gives the exact gradient; `TimestepLeakanceOp` and `src/routing/` are untouched
because the gate is applied to the head OUTPUT before `setup_inputs`
denormalizes it (`src/training/gate.rs::leakance_gate`).

```yaml
params:
  use_leakance: true
  leakance_gate:
    temperature: {1: 1.0, 6: 0.5, 11: 0.2, 16: 0.05}   # keyed by 1-indexed epoch
```

| Key | Default | Notes |
|---|---|---|
| `temperature` | required | Epoch-keyed schedule resolved like `experiment.learning_rate` (`LeakanceGate::resolve`: largest key `<= epoch`, first value before it). Every value must be positive and finite; `1.0` = identity, `> 1` softens |

Which temperature each reader applies:

| Reader | Temperature | Log line |
|---|---|---|
| `src/training/forward.rs::forward` (training) | `resolve(epoch)`, threaded by the driver as `gate_tau` | `epoch E lr=… leakance_gate_tau=…` |
| `src/training/forward.rs::forward_eval_core` (eval) | `final_temperature()` | `leakance gate: final temperature tau=…` |
| `src/training/probe.rs::probe_forward` | `final_temperature()` (after lifting, so leaf grads carry the gate's Jacobian) | same, from `probe_zeta_gradient` |
| `src/dump_parameters.rs::dump` / `dump_init` | `final_temperature()` (the dumped factor is the gated value) | same |

`forward` asserts `gate_tau.is_some() == params.leakance_gate.is_some()`, so a
caller cannot train an ungated model against a gated config or vice versa. In
`forward_eval_core` the gate is applied BEFORE `LeakanceOverride`, so an
override still replaces the value that reaches denormalization. Tests:
`tests/leakance_gate.rs` (tau = 1 bit-exact identity through all three readers,
central-difference gradcheck at five temperatures, monotonicity, limits,
saturation without NaN/Inf) and the gated rows of `tests/gamma_eval_parity.rs`.
This does not re-open the leakance verdict in `research-status.md`; it removes
one degeneracy so a future arm can be judged on the gate, not on the product.

## Reservoirs (`params.use_reservoirs` + `data_sources.reservoirs`, added 2026-09-25)

Option C of `.claude/RESERVOIRS.md`: each dam reach listed in the table is routed as
a linear reservoir `S = T·Q`, which is Muskingum at `K = T`, `X = 0`. `T` is
prescribed data, not learned; a dam row's `n`/`q_spatial`/`p_spatial` get exactly
zero gradient. Read `.claude/RESERVOIRS.md` before extending it.

```yaml
params:
  use_reservoirs: true
data_sources:
  reservoirs: examples/juniata/data/juniata_reservoirs.csv   # COMID,T_days[,…]
```

CSV contract (`src/data/store/reservoirs.rs::read_reservoir_table`): a header
naming `COMID` (MERIT reach id) and `T_days` (residence time, days); other columns
ignored. Every row must parse, `T_days` must be finite and `>= 1/24` (one hour, the
routing `dt`; `c3 >= 0` at `X = 0` needs `T >= dt/2`), COMIDs unique, at least one
row. Any violation is a `DataError` naming the file, raised at dataset open.

Wiring: `MeritGagesDataset::open` reads the table once; every batch (`collate`) and
the eval static network map it onto their COMID order with
`src/data/store/reservoirs.rs::reservoir_rows` (table COMIDs outside the network are
skipped) and carry the result as `RoutingBatch::reservoir_rows`. Every engine site
(`forward`, `forward_eval_core`, `forward_with_frozen_params`, `probe_forward`) calls
`src/training/forward.rs::apply_reservoir_rows` right after `setup_inputs`, which
calls `MuskingumCunge::set_reservoir_rows`. The eval network's match is logged once:
`reservoirs: <k> of <m> table COMIDs are in the network`. Training batches do not
log the match, so dataset open also logs `reservoirs: table <path> has <m> COMIDs`,
which a train-only run carries in `run.log` too. `src/bin/probe_courant.rs` refuses
`use_reservoirs: true`: it drives `forward_chain_inner` with no reservoir override.

Rejected at load (`validate_reservoirs`): `use_reservoirs: true` without
`data_sources.reservoirs`, or with `use_leakance: true`, `use_cuda_graphs: true`,
`ddr_match: true`, `params.subdivision.enabled: true`, or
`data_sources.gridded_network`. The engine panics if reservoir rows reach the
leakance or CUDA-graph op, so these guards are what keeps a run from starting in a
combination the override does not cover. The paper studies (`ddrs experiment`)
refuse arms with `use_reservoirs: true`.

Committed fixture: `examples/juniata/data/juniata_reservoirs.csv`, one row, Raystown
Lake (COMID 73005301, GRanD 1613, `T_days = 1.23` from the sandbox fit in
`research/findings/2026-09-25-reservoir-representation-options.md` §2.2), row 177
of the 213-reach Juniata network, upstream of the Newport gauge. No CONUS table is
committed yet: it is to be fitted from ResOpsUS, keyed through the GRanD → MERIT
COMID crosswalk at `/mnt/ssd1/data/resops/derived/grand_to_merit_comid.csv`
(workstation data, not a test input).
Tests: `src/config.rs` (`use_reservoirs_*`), `src/data/store/reservoirs.rs`,
`tests/reservoir_override.rs`, and the release-only
`tests/juniata_acceptance.rs::juniata_reservoir_is_matched_logged_and_changes_the_gauge_series`.

### Seasonal fixed table

A `fixed` table may add BOTH `a` and `b` columns (`COMID,T_days,a,b`): each dam is then a
prescribed seasonal bucket `T(t) = max(T_days·exp(a·sin ω_t + b·cos ω_t), 1 h)`, routed through
`MuskingumCunge::set_dam_release` with constant tensors. `a = b = 0` is bitwise option C. Dataset
open logs `reservoirs: table <path> has <m> COMIDs (fixed, seasonal a/b)`. A fixed table may also
carry a rule curve (`c1s, c1c, c2s, c2c` + `inflow_mean_m3s`) and route on the additive row with
`params.reservoir_dam_row: additive` (see `dam_row` below).

### Learned release (`reservoir_release: learned`, added 2026-09-26)

```yaml
params:
  use_reservoirs: true
  reservoir_release: learned
  parameter_ranges:
    reservoir_T0: [0.041666668, 365.0]   # days, log space
    reservoir_a: [-2.0, 2.0]
    reservoir_b: [-2.0, 2.0]
data_sources:
  reservoirs: experiments/reservoir/release_head/dam_features.csv   # COMID + feature columns
release_head:            # top-level block, deny_unknown_fields
  hidden_size: 8         # default 8
  num_hidden_layers: 1   # default 1
  grid: 5                # default 5
  k: 3                   # default 3
  seasonal: true         # default true; false = constant T0 per dam (one output)
  input_var_names: [log10_storage, ..., purpose_other]   # feature-table columns, required
  routing_checkpoint: /abs/.ddrs/runs/<id>/checkpoints/epoch_E_mb_M   # default absent
  freeze_routing: false  # default false; true requires routing_checkpoint
  dam_row: replace       # default replace; additive adds T·Q to the reach's channel storage
  dam_floor: forgive     # default forgive; carry makes the dam-row clamp mass-conserving (owed volume)
  dam_row_positivity: false  # default false; true caps the additive dam row's wedge so c1 > 0 (needs dam_row: additive)
  rule_curve: false      # per-dam harmonic rule curve S0_d(t); needs inflow_mean_m3s in the table
  rule_curve_max: 1.0    # c = rule_curve_max·tanh(θ); > 0
  per_dam_t0: false      # T0_d = T0_head,d·exp(δ_d)
  per_dam_lr: 0.05       # constant lr of the per-dam parameters' own Adam; > 0
  per_dam_l2: 0.0        # per_dam_l2·Σ(θ² + δ²) over the optimizer step's dams (union over its
                         # micro-batches), added ONCE per step; >= 0
  rule_curve_penalty: 0.0  # feasibility penalty weight (off at 0); >= 0; > 0 needs rule_curve
  rule_curve_alpha: 0.9    # share of the dam's inflow the flux may store before the hinge; (0, 1]
  flood_pool: none         # none | flood_control | all: the per-dam flood pool (law FA); needs inflow_mean_m3s
```

**Rule curve and per-dam parameters (added 2026-09-27).** `rule_curve: true` makes the storage
law `S = T·Q + S0_d(t)` on either dam row, the flux
`r_d = Ibar_d·Σ_{k=1,2}(c_{k,s} sin kω + c_{k,c} cos kω)` taken off the dam row's lateral inflow
as `(S0_{t+1} − S0_t)/dt` after the `discharge` floor on `q'` (derivation and units in
`src/routing/release.rs`). Its phase is CONTINUOUS (`rule_curve_phase_start`: `ω` advances
`Ω·dt` every hourly step, `2π/365.25` at 1970-01-01), not the `T` law's day-of-year phase, so
the Dec 31 → Jan 1 step carries one hour of phase (the first build's day-of-year phase gave that
step +7 h, or −17 h in a leap year, of flux). The coefficients are per-dam FREE parameters
(`src/nn/dam_params.rs`, one row per feature-table dam, init 0 = no rule curve bit for bit),
not head outputs: offline they are not predictable from the NID features. `per_dam_t0` adds a
per-dam `exp(δ)` on the head's `T0`. Both train with their own ROW-SPARSE Adam
(`src/training/lazy_adam.rs`, since the v3 fixes: only rows with a nonzero gradient in a step
update, each with its own bias-correction counter, so a dam never in a batch stays exactly at
its init; the first build's dense Adam kept drifting rows on stale moments) at the constant
`per_dam_lr` (a dam gets a gradient only when a gauge below it is in the batch), clipped on
their own norm, saved as `release_dams.mpk` (FULL precision) + `release_dams_optim.json` (f32
bit patterns, bitwise resume) in each checkpoint and restored on resume. A checkpoint from the
first build carries a dense `release_dams_optim.mpk` instead; its per-dam optimizer restarts
cold (logged). The gradient reaches `θ` through the timestep op's `q'`
parent (no op change). `Ibar_d` is the table's raw `inflow_mean_m3s` column
(`experiments/reservoir/release_head/build_dam_inflow_clim.py`); `rule_curve: true` without it
fails at dataset open. The resolved test-phase table carries `c1s, c1c, c2s, c2c`, the effective
`T0` and `inflow_mean_m3s` per dam (also in `release_params.csv`), and a `fixed` table may carry
the same optional columns (all four `c` or none; `c` requires `inflow_mean_m3s`). Training logs
`rule_curve_|c|_median=<v> (<k> dams with gradient)` per optimizer step and, whenever dams are
armed, one `dam clamp, step <N>: dam-row steps at the discharge clamp <k>/<m> (<p>%);
clamp-created volume <c> m3 (storage <s>); dam inflow <v> m3 (summed per dam); created share per
dam-window: median, p90, max, <j> of <n> >= 0.5%, <l> >= 5%` line per optimizer step (training;
over every micro-batch's dams, each over its own window). The created volume (corrected in v4,
review v3 finding 1) is `Σ max(lb − x, 0)·dt/c4` over the dam rows' pre-clamp solves `x`:
`dt/c4 = D/2 = K(1 − X) + T_{t+1} + dt/2`, so it is the storage the clamp forgives
(`(K(1 − X) + T)·δ`, the `storage` part) plus the below-floor outflow the solve passed down
(`dt/2·δ`), and equals the lateral volume that would have held the row at `lb`. It is exactly
what the dam row's volume balance lacks with its routed series as the outflow (derivation in
`src/routing/mmc.rs` `DamAccount`; `tests/reservoir_rule_curve.rs` mass-balance tests). The v3
build logged `Σ max(lb − x, 0)·dt`, a one-step rate deficit, too small by `D/(2·dt)` (3-8x at
`T` of 0.1-0.3 d; more with the channel's `K`); v3 run logs and `release_clamp.csv` files carry
that. The inflow is `Σ (I_t + q')·dt` (routed upstream inflow plus the reach's own `q'`, before
the flux); the engine keeps all three per dam (`MuskingumCunge::dam_account`). There is no
pooled share: a dam below another counts the upper dam's outflow in its own inflow (review v3,
finding 6), so shares are per dam. The test phase sums the account over every chunk and writes
`<run>/release_clamp.csv` (`COMID,created_m3,storage_m3,inflow_m3,created_share,clamp_steps,steps`,
every dam armed at least once, any dam table), logs `release clamp (test phase, <n> dams): ...`
and records `metrics.release_clamp` (`n_dams`, `clamp_steps`, `steps`, `created_m3`, `storage_m3`,
`created_share_{median,p90,max}`, `n_dams_created_share_ge_{0p5,5}pct`). Read an S3/S4-type arm
only after excluding or flagging every scored dam whose created share is above about 0.5 %.
Tests:
`tests/reservoir_rule_curve.rs`, `tests/release_freeze_routing.rs` (frozen + rule curve),
`src/nn/dam_params.rs`, `src/data/store/reservoirs.rs`, `src/config.rs` (`rule_curve_*`,
`*_per_dam_*`).

**Rule-curve feasibility penalty (added 2026-09-27, v3).** The flux can store more than the dam
receives; the S28 clamp then creates water and zeroes the dam row's gradient, so nothing pushes
back (review v2). `rule_curve_penalty: λ > 0` adds, ONCE per optimizer step,
`P = λ·Σ relu(r_d(t) − α·Qin_d(t))² / Σ Qin_d(t)²` over the step's distinct (dam, window) pairs,
`r = (S0_{t+1} − S0_t)/dt` the flux (positive = storing), `Qin = I_t + q'` the dam's inflow from
the forward itself, DETACHED (`src/training/dam_terms.rs`). Differentiable in `θ` only. It is a
hinge: exactly 0 while the flux stays below `α·Qin`, so it is 0 at `θ = 0` (the start of every
run). Training logs `rule_curve_penalty=<P> (hinge active on <k>/<m> dam-steps)` and
`per_dam_l2_term=<v>` on each step's `mb=` line. The hinge does not cover every clamp: an
additive row with the channel's negative `c1` can clamp with the flux below `α·Qin` (seen on
the Juniata test fixture), so read the created share, not only the hinge count.

**`dam_row` (added 2026-09-27).** `replace` (default, every earlier run) makes the dam row the
reservoir alone (`K := T`, `X := 0`, S19''/S19'''), so `T = 1 h` is faster than no dam on a reach
whose `K_r` is hours, and `reservoir_T0` must start at `>= 1/24` d. `additive` keeps the reach's
own Muskingum `K_r`, `X_r` and adds the reservoir: `S = K_r[X_r I + (1 − X_r) Q] + T·Q` (S19''''
/ B19'''' in `mmc_op`; coefficients in `src/routing/release.rs`). `T = 0` is the channel row bit
for bit, `T` has no floor, and `reservoir_T0` needs only `0 < lo` (log space); the dam reach's
`n`/`q_spatial`/`p_spatial` keep their gradient through `K_r`, `X_r`. `c3 >= 0` needs
`K_r(1 − X_r) + T_t >= dt/2`, met wherever the channel's own `c3 >= 0`. The test phase routes the
resolved table through the same row. A `fixed` table has no `release_head:` block (rejected at
load); it selects its row with `params.reservoir_dam_row: replace | additive` (added 2026-09-27,
v3; absent = `replace`, every earlier config). That key is rejected with `reservoir_release:
learned` (the block's `dam_row` owns it) and without `use_reservoirs`. `Config::dam_row` reads
the block for a learned release, the params key otherwise. A fixed additive table may carry the
rule-curve columns (`c1s, c1c, c2s, c2c` + `inflow_mean_m3s`), so an offline fit replays in the
engine (`experiments/reservoir/smoke/replay_offline_fits.py` builds `fixed_L2.csv` /
`fixed_L4.csv`; `config/experiments/dam_release_smoke_replay_L{2,4}.yaml` route them with the
smoke off arm's head, zero training steps, traps.md T10). The engine API
(`MuskingumCunge::set_reservoir_rows_as`, `DamRelease::dam_row`) supports both.
Tests: `tests/reservoir_additive.rs` (T = 0 and K_r = 0 identities, per-step storage balance,
gradchecks in both `c1` regimes, opt-out on Juniata), the additive cases of
`tests/reservoir_release_training.rs`, `src/config.rs` (`dam_row_*`).

**`dam_floor` (added 2026-09-28, v4).** What the S28 clamp does with the water it creates on a
dam row. `forgive` (default; every earlier run, bitwise) leaves it in the river. `carry` keeps
the floor mass-conserving: each armed dam carries an owed volume (m³, >= 0); after a solve whose
dam row clamped, `owed += Σ max(lb − x, 0)·dt/c4` (the created volume above); before every solve
the dam repays `r = min(owed/dt, max(Qin_t − lb, 0))` m³/s out of its effective lateral inflow,
`Qin_t = (N·Q_t)_d + q'_d` (step-start routed inflow plus its own `q'`, before the rule-curve
flux, the penalty's `Qin`), and `owed −= r·dt`. `created − repaid − owed = 0` at every step, so
the dam row's routed series passes on its inflow less its storage change, plus only what is still
owed (`tests/reservoir_dam_floor.rs`: one year, outflow = inflow to 1.3e-4 where `forgive` creates
6-11 %). It covers every dam-row clamp, the additive row's negative `c1` at low flow included.
What it conserves is the dam row's ROUTED series (what an on-reach gauge reads): the rows below
read the pre-clamp `x` in a clamped step, so over a period they receive the dam's inflow less `ΔS`
less the below-floor part `created_m3 − storage_m3`, which is owed and repaid too (in the one-year
replace-row test that is 7 % of the inflow; roughly `dt/D` of the in-debt flux per step). Owing
only `storage_m3` would conserve their receipt instead; holding the dam's in-step outflow at `lb`
would conserve both (not built).
**Known failure, do not read carry runs on the additive row (2026-09-28, v4 re-evaluations).**
Once a dam owes more than its storage, the repayment (all of `Qin − lb`) pins its outflow at the
floor, where the additive row's channel `K`, `X` (Cunge, evaluated at the dam's own `Q_t`) go to
days and 0.5: `c1 ≈ −K·X/(K(1 − X) + T)`, about −0.5. Every rising step of the upstream inflow
then clamps and owes about `K·X·ΔI`, while the wedge's release on falling steps (`x > lb`)
passes downstream instead of repaying: a debt pump. Measured: replay L2 carry
(`2026-09-28T01-11-35Z`), 21 of 202 dams end owing > 0.5 % of their inflow; COMID 74038104
(T 1.67 d) clamps 25,813 of 131,130 test steps (forgive: 2,219), owes 3.7x its 15-year inflow
and repays all of it, i.e. passes nothing downstream for the whole period. S3v3 carry
(`2026-09-28T01-11-45Z`): 25 of 675 dams repay > 50 % of their inflow; COMID 74037973 goes from
13 forgive clamps to 26,699. Median NSE falls (L2 0.7135 -> 0.7047, L4 0.7165 -> 0.7059, S3v3
0.7184 -> 0.7047). A synthetic chain reproduces it (30 days, diurnal upstream inflow, 1e7 m³
initial debt at T 0.05 d: 232 clamp steps, 1.95e7 m³ new debt, the whole inflow repaid);
without the diurnal swing, or with a debt smaller than the storage, there is none. The replace
row (`K = T`, `X = 0`, `c1 = dt/D > 0`) and the unit tests' constant-`K` additive row do not
pump. The fix built (v5) is `dam_row_positivity` (below), which removes the negative `c1`
itself; reclaiming the wedge's release while in debt (retain `x − lb` after the solve, or re-solve
with the dam's in-step outflow held at `lb`) was the alternative and is not built.
The owed state is DETACHED (inner backend, a constant cut to `q'`): training sees neither the
debt a flux incurs nor its repayment as a function of `θ`, `T0` or the routing parameters; the
gradient through a clamped step is still zero, the repayment acts like a change to the forcing,
and the feasibility penalty stays the only restoring gradient. It restarts at 0 every training
window and runs across the test phase's chunks (`training::forward::carry_dam_owed`,
`DamClampSums::merge` keeps each dam's closing owed; a 15-day-chunked year matches one engine
bitwise). A dam not yet completed has no owed state and starts at 0 when it switches on. With
`carry`, `created` counts every step at the floor, including those spent repaying (the law still
asks for more than the dam has), so it is several times the `forgive` figure; read `owed_m3`
for what the floor finally adds. Logs: the training `dam clamp` line and the test phase's line
gain `dam_floor carry: repaid <R> m3, owed at end <O> m3 (max <p>% of a dam's inflow)`;
`release_clamp.csv` has `repaid_m3`, `owed_m3` (0 with `forgive`); `metrics.release_clamp` has
`dam_floor`, `repaid_m3`, `owed_m3`, `owed_share_max`; `metrics.release_training.dam_floor`.
A `fixed` table sets it with `params.reservoir_dam_floor: forgive | carry` (absent = forgive),
rejected with `reservoir_release: learned` (the block's `dam_floor` owns it) and without
`use_reservoirs`, exactly like `reservoir_dam_row`. `Config::dam_floor` reads the block for a
learned release, the params key otherwise; the engine takes it at construction
(`MuskingumCunge::set_dam_floor` / `set_dam_owed` / `dam_rows` for tests and chunk threading).
Tests: `tests/reservoir_dam_floor.rs`, `src/config.rs` (`*dam_floor*`),
`src/training/release_eval.rs`.

**`dam_row_positivity` (added 2026-09-28, v5).** Keeps the additive dam row's inflow coefficient
`c1 >= 0`. The additive row keeps the reach's channel wedge `K_r·X_r·I` in its storage, so
`c1 = (dt/2 − K_r X_r)/D` is negative wherever `K_r·X_r > dt/2`; at low outflow the Cunge `K_r`
is days and `X_r` is 0.5, rising inflow then drives the pre-clamp solve below the floor, `forgive`
creates water and `carry` pumps debt (above). With `true`, on the dam rows only (channel rows are
untouched), `X_eff = min(X_r, 0.5·(1 − δ)·dt/K_r)`, `δ = mmc_op::POSITIVITY_DELTA = 1e-2`, and
`D`, `c1..c4` all read `X_eff` (S19p / B19p in `mmc_op`), so `c1 >= δ·dt/D > 0`. It is S19''s
`hi_a` branch on the dam rows alone: `δ` because at `δ = 0` the cap lands on `c1 = 0` and f32
roundoff crosses it; no K floor or `hi_b` cap, because `T_t` keeps c3's numerator up and the cap
only raises `2K(1 − X)`. Backward: where the cap binds (`x_eff == cap`, recomputed from the
saved K), `∂L/∂x_eff` goes into K through `∂cap/∂K = −cap/K` and nothing reaches the Cunge `X_r`
chain; elsewhere the row is bitwise the additive row, outputs and gradients. Consequence: with
it, `T = 0` is the channel row only where the channel's own `K_r·X_r <= (1 − δ)·dt/2`. Measured
on the v4 synthetic pump (30 days, diurnal upstream swing, 1e7 m³ opening debt, T 0.05 d, carry):
new debt 1.95e7 m³ -> 0, cleared on day 11, and the carried balance closes; with the cap binding
on every step and `K`, `X` constant the dam row's routed-series balance over a rule-curve year
closes to 3e-7 of the created volume. `release_head.dam_row_positivity` for a learned release;
`params.reservoir_dam_row_positivity: true | false` for a `fixed` table (absent = false),
rejected with `reservoir_release: learned` and without `use_reservoirs` (like
`reservoir_dam_floor`); `true` is also rejected without `dam_row: additive` (the replace row's
`c1 = dt/D > 0` already) and with `params.enforce_positivity` (S19' caps every row at the same
bound). `Config::dam_row_positivity` reads the block or the key; the engine takes it at
construction (`MuskingumCunge::set_dam_row_positivity` for tests). The engine's account and the
test phase count negative-`c1` dam-row steps: `DamClampAccount::{c1_min, neg_c1_steps}`,
`release_clamp.csv` column `neg_c1_steps`, `metrics.release_clamp.{neg_c1_steps, n_dams_neg_c1}`,
and `dam-row steps with c1 < 0 <k>/<m> (<j> dams)` in the clamp log lines;
`metrics.release_training.dam_row_positivity`. Tests: `tests/reservoir_dam_positivity.rs`,
`src/config.rs` (`*dam_row_positivity*`).

**`flood_pool` (added 2026-09-29, v6; law FA of `experiments/reservoir/laws_v6`, report §2/§5).**
A per-dam flood pool on top of the dam row (whatever it is: option C, the bucket, a rule curve):
each pooled dam carries a pool `F` (m³, >= 0) and, before each solve, from its step-start inflow
`I = (N·Q_t)_d + q'_d` (the penalty's `Qin`: routed upstream inflow plus its own `q'`, before the
rule-curve flux), captures `Vc = min(phi·max(I − Qc, 0)·dt, Fmax − F)` and evacuates
`Ve = min(F, max(Qc − p, 0)·dt)` at the release target (`p = I − Vc/dt`), both on the dam row's
`q'` (`(Ve − Vc)/dt`), then `F += Vc − Ve`. `Qc = kc·Ibar`, `Fmax = z·Ibar·86400`, `Ibar` the
table's `inflow_mean_m3s` (required, detached). Mass is conserved by construction (captures only
what enters, evacuates only what it holds; `F >= 0` exactly in f32) and the discharge floor is
never used (one-year flood test: balance residual 6e-7 of inflow, zero clamps). A, the CSR pattern
and the backward are unchanged (`src/routing/mmc.rs` `FloodPool`). `none` (default) routes bitwise
as before; `flood_control` pools the dams whose table row has `purpose_flood` (the raw 0/1 column,
read whether or not the head uses it; required at dataset open); `all` pools every dam. `kc`,
`phi`, `z` are per-dam free parameters (`src/nn/dam_params.rs` `pool` `[n_dams, 3]`, raw 0 at
init) trained with the other per-dam parameters (row-sparse Adam, `per_dam_lr`, `per_dam_l2`
anchors at the init): `kc = clamp(3·exp(r), 0.5, 20)`, `phi = sigmoid(r)`,
`z = 120·sigmoid(4·r + logit(0.05/120))` days (0.05 d at init: off in effect but with a gradient
through the cap; the logistic rate 4 lets ~1.5 raw units of per-dam Adam travel span 0.05 to ~20
d). GRADIENT: `F` stays on the tape across a window; `I` is NOT detached (the pool's response to
its inflow reaches upstream routing parameters, gradchecked on an upstream reach's `n`). `F`
starts at 0 every training window and runs across the test phase's chunks
(`training::forward::carry_dam_state`, `DamClampSums::merge_pool`/`pool_f_for`; a 15-day-chunked
period equals one engine bitwise); a dam not yet completed has no pool. The pool's timescale is
~60 days: pool configs use `experiment.rho: 180`, `warmup: 30`, and `testing.warmup: 5` (the test
overlay otherwise inherits 30 and the test metrics would skip 25 more days than every earlier
arm). A `fixed` table sets it with `params.reservoir_flood_pool: true | false` (absent = false) and
the columns `kc`, `phi`, `z` (days; all three or none; `z = 0` = no pool at that dam) plus
`inflow_mean_m3s`; true without the columns fails at dataset open, columns without it are logged
and not routed. The key is rejected with `learned` and without `use_reservoirs`; either key is
rejected with the carried dam floor (`carry` repays out of the same inflow). The resolved test
table carries `kc, phi, z` (z = 0 at dams training does not pool) and `release_params.csv` gains
`kc,phi,z` (before `inflow_mean_m3s`). Logs: dataset open `reservoirs: flood pool on (...): <k> of
<m> table dams pooled`; each training step ` flood_pool_median kc=… phi=… z=…d (<k> dams with
gradient)` on the `mb=` line and a `flood pool, step <N>: …` line (fill F/Fmax at the end and
peak, pool size, most held, captured/evacuated/held volumes); test phase `flood pool (test phase):
…`, `<run>/release_pool.csv` (`COMID,captured_m3,evacuated_m3,F_end_m3,max_F_days,Fmax_days,
inflow_mean_m3s,inflow_m3,steps`) and `metrics.release_pool`; `metrics.release_training.flood_pool`.
Tests: `tests/reservoir_flood_pool.rs`, `src/config.rs` (`*flood_pool*`), `src/nn/dam_params.rs`,
`src/training/lazy_adam.rs`, `src/data/store/reservoirs.rs`, `src/training/release_eval.rs`.

**Release-only training (added 2026-09-27).** `routing_checkpoint` is a checkpoint DIRECTORY
whose `head.mpk` initialises the ROUTING head: weights only, never its `optim.mpk` or
`state.json` (the routing optimizer starts cold, the run at epoch 1). `freeze_routing: true`
detaches the routing head (`Module::no_grad`, like the disaggregation freeze): the backward
spends nothing on its parameters, the driver takes no routing optimizer step, and only the
release head trains, its gradient still flowing through the routing solve into `T`. The saved
`head.mpk` of every checkpoint is then the frozen weights, which the test phase loads. A batch
with no active dam has a constant loss and takes no step. `freeze_routing: true` without
`routing_checkpoint` is rejected at load (`validate_reservoirs`). `run.log` carries
`routing warm start: ... FROZEN` and `routing head frozen: training the release head only`;
the manifest's `metrics.release_training` records `routing_checkpoint` and `freeze_routing`.
`experiment.checkpoint` (a full resume) is applied after `routing_checkpoint` and wins; a frozen
head stays frozen across it. Tests: `tests/release_freeze_routing.rs`,
`src/config.rs` (`freeze_routing_*`, `routing_checkpoint_*`).

The table is `experiments/reservoir/release_head/dam_features.csv` (1,024 COMIDs, NID >= 10 MCM,
19 normalised features), rebuilt by `build_dam_features.py` under the DDR venv. Dataset open logs
`... (learned release, dam features)`; training logs `release_T0_median=<d>d (<k> dams)` per
mini-batch; the test phase logs `release head: resolved <m> dams from <ckpt>/release_head.mpk` and
writes `<run>/release_params.csv` (the eval binary writes `<output>.release_params.csv`).
Checkpoints carry `release_head.mpk` + `release_optim.mpk`; resuming a learned config from a
checkpoint without them starts the release head cold (logged). Arms of record:
`config/experiments/dam_release_smoke_{off,learned}.yaml`,
`config/experiments/dam_release_full_{off,learned}.yaml`.

**Head-less forwards panic, by design.** `training::forward::apply_reservoir_rows` panics when a
learned table's feature rows reach a forward that has no release head: `probe_forward` (the probe
binaries), the `Frozen` test-phase path (`eval --frozen`), and any caller of plain
`forward_eval*` before `training::release_eval::resolve_learned_release` has turned the table into
a fixed seasonal one. The alternative, routing the dams as channels, would silently score a
different model. The paper studies refuse `use_reservoirs` arms at load.

**Untyped ranges.** `params.parameter_ranges` is a map; a misspelt `reservoir_T0` /
`reservoir_a` / `reservoir_b` key is ignored and the default box is used (the pre-existing
pattern for every range key). Check the spelling against this file.
