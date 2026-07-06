---
name: ddrs-config-and-flags
description: "Use when you need to add, change, or audit any ddrs YAML configuration key; diagnose a config-load error; understand what a parameter controls; add a new routing parameter or training flag; or decide which experiment config to use as a starting point. Also use when modifying params.use_leakance, params.use_cuda_graphs, kan_head.disaggregation, or experiment.loss."
---

# ddrs Config and Flags Reference

**Jargon primer (defined once):**
- **BURN** — Rust deep-learning framework (like PyTorch for Rust). Used instead of PyTorch here.
- **KAN head** — Kolmogorov-Arnold Network; maps per-reach catchment attributes to routing parameters. Replaces an MLP.
- **MC routing** — Muskingum-Cunge, a 1-D river routing solver. The `params:` block controls it.
- **CONUS** — Contiguous US; the default training domain (346,321 reaches).
- **Q'** — lateral inflow (m³/s) per reach; the forcing signal from dHBV2.
- **CSR** — Compressed Sparse Row; the sparse matrix format used for the triangular network solve.
- **CUDA graph** — GPU kernel sequence baked at compile time and replayed cheaply each timestep.
- **ddrs.yaml** — the live workspace config; `ddrs plan` generates it from a template. NEVER committed.
- **config/merit_training.yaml** — the canonical production template; committed and kept in sync with DDR-Python.

---

## When NOT to use this skill

| Situation | Use instead |
|---|---|
| Debugging NaN loss or gradient explosion | `ddrs-systematic-debugging` |
| Adding a new data source format (new zarr reader, new fabric) | `ddrs-data-sources` |
| Understanding the sparse backward / autograd tape | `.claude/references/ddrs-burn-autograd.md` |
| Per-timestep routing math | `.claude/ARCHITECTURE.md` |
| CLI lifecycle (`ddrs plan`, `ddrs run`, `ddrs gc`) | `CLAUDE.md` §"ddrs CLI" |

---

## Overview: config file anatomy

A ddrs config is a single YAML file with six top-level sections:

```
mode / workflow / geodataset / device / seed / np_seed   ← top-level scalars
data_sources:     ← where inputs live on disk
experiment:       ← training-loop hyperparameters
kan_head:         ← KAN head architecture + which parameters it predicts
params:           ← routing engine settings
testing:          ← overlay applied in eval mode (overrides experiment: keys)
```

The Rust struct is `src/config.rs::Config`. Deserialization uses
`Config::from_yaml_file_with_mode(path, ConfigMode::Training|Testing)`.

---

## Top-level scalars

| Key | Type | Default | Notes |
|---|---|---|---|
| `mode` | `"training"` \| `"testing"` | `"training"` | Must agree with `workflow:` (see guard below). |
| `workflow` | `train` \| `eval` \| `train-and-test` | absent (None) | `train-and-test` runs both phases and computes the baseline comparison. |
| `geodataset` | `"merit"` | `"merit"` | Only value supported as of 2026-07-05. |
| `device` | integer | `0` | CUDA device ordinal. On multi-GPU hosts, pick a non-display GPU. |
| `seed` | integer | `42` | Controls KAN weight initialization. |
| `np_seed` | integer | `42` | Controls per-epoch gauge shuffle order. |

**Guard:** `mode: training` requires `workflow ∈ {train, train-and-test}`. `mode: testing` requires `workflow: eval`. A contradiction is rejected at load time with a message containing `"conflicting top-level keys"`.

---

## `data_sources:` section

All data is read in place — no export step. Every path is a `PathBuf`.

| Key | Required | Notes |
|---|---|---|
| `attributes` | Yes | NetCDF catchment attributes. Columns must match `kan_head.input_var_names`. |
| `streamflow` | Yes | dHBV2 lateral inflow Q'. Icechunk (`.ic`) for CONUS; zarr-v2 for global. |
| `observations` | Yes | USGS (or global) daily observed discharge; training targets. |
| `gages` | Yes | CSV with STAID and COMID columns. |
| `geospatial_fabric` | Conditional | `.shp`/`.dbf`/`.gpkg`; triggers managed adjacency build into `.ddrs/adjacency/<key>/`. |
| `geospatial_fabric_layer` | Optional | Layer name inside a multi-layer `.gpkg`; invalid for `.shp`/`.dbf`. |
| `conus_adjacency` | Conditional | Pre-built zarr. Must be paired with `gages_adjacency`. |
| `gages_adjacency` | Conditional | Pre-built zarr. Must be paired with `conus_adjacency`. |
| `aorc_precip` | Optional | Hourly AORC precip zarr v3 (`merit_unit_catchments.zarr`). Required when `kan_head.disaggregation.use_precip: true`. |

**Adjacency rule (enforced at load time):** provide EITHER both `conus_adjacency` + `gages_adjacency`, OR `geospatial_fabric` (managed build). Providing only one of the two adjacency zarrs is rejected. Providing none of the three is rejected.

**Production path (CONUS workstation, as of 2026-07-05):**
```yaml
data_sources:
  attributes: /home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc
  geospatial_fabric: /projects/mhpi/data/MERIT/raw/continent/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp
  streamflow: /mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic
  observations: /mnt/ssd1/data/icechunk/usgs_daily_observations
  gages: /home/tbindas/projects/ddr/references/gage_info/gages_3000.csv
```

---

## `experiment:` section (training mode)

| Key | Type | Default | Production value | Notes |
|---|---|---|---|---|
| `batch_size` | integer | none | `64` | **Gauges** per mini-batch during training. Meaning shifts in `testing:` — see below. |
| `start_time` | `"YYYY/MM/DD"` | none | `"1981/10/01"` | Training window start. |
| `end_time` | `"YYYY/MM/DD"` | none | `"1995/09/30"` | Training window end (water year 1995). |
| `epochs` | integer | none | `5` | Total training epochs. |
| `rho` | integer \| null | none | `90` | Sequence length in days per mini-batch. Set `null` in testing overlay. |
| `shuffle` | bool | `false` | `true` | Re-shuffle gauge order each epoch (seeded by `np_seed`). |
| `warmup` | integer | none | `5` | Days excluded from loss at sequence start (routing spin-up). |
| `learning_rate` | map epoch→f32 | `{}` | `{1: 0.001, 3: 0.0005}` | Step decay; applies from that epoch onward. |
| `grad_clip_max_norm` | float \| absent | absent | `1.0` | Global gradient-norm clip. Omit to disable. |
| `checkpoint` | path \| absent | absent | absent | Directory path to resume from (e.g. `.ddrs/runs/<id>/checkpoints/epoch_5_mb_9`). |
| `loss` | block \| absent | L1 (see below) | absent | Training objective. Omit for historical L1. |

### `experiment.loss:` sub-block

Omit the entire `loss:` block to use the historical L1 objective. Including the block does NOT change behavior if `kind: l1`.

| Key | Type | Default | Notes |
|---|---|---|---|
| `kind` | `l1` \| `nnse-kge` \| `kge` | `l1` | `l1` = mean absolute error. `nnse-kge` = composite NNSE + KGE. `kge` = component-weighted KGE (r, alpha, beta terms individually weighted). |
| `nnse_weight` | float | `1.0` | Weight on `1 - NNSE` term (all non-L1 kinds). |
| `kge_weight` | float | `1.0` | Weight on `1 - KGE` Euclidean term (`nnse-kge` only). |
| `r_weight` | float | `1.0` | Weight on `(r-1)²` correlation term (`kge` kind only). |
| `alpha_weight` | float | `1.0` | Weight on `(alpha-1)²` variance ratio (`kge` kind only). This is the restoring force against MC over-attenuation. |
| `beta_weight` | float | `1.0` | Weight on `(beta-1)²` mean ratio (`kge` kind only). |
| `kge_clamp` | float | `10.0` | Per-gauge upper bound on weighted KGE-component sum before averaging. Prevents near-constant gauges from hijacking the batch gradient. |
| `eps` | float | `0.1` | Stabilizes variance/mean denominators. Matches DDR `hydrograph_loss` default. |

**Why L1 and NSE both fail KGE:** L1 and NSE are both maximized when simulated variance is below observed (NSE optimum is at `alpha = r < 1`). This rewards MC for over-attenuating flood peaks — the diagnosed cause of KGE regression vs the summed-Q' baseline in CONUS runs (median KGE 0.723→0.701 while NSE improved 0.639→0.684). The `(alpha-1)²` term in `nnse-kge` / `kge` supplies a restoring gradient.

---

## `testing:` section (eval-mode overlay)

These keys **replace** the matching `experiment:` keys when `mode: testing` is loaded. Absent keys inherit from `experiment:`.

| Key | Default in testing | Notes |
|---|---|---|
| `start_time` | `"1995/10/01"` | Eval window start (water year 1996). |
| `end_time` | `"2010/09/30"` | Eval window end. |
| `batch_size` | `15` | **DAYS** per evaluation chunk — semantic shift from training's gauges-per-batch. |
| `rho` | `null` | Explicitly clears sequence sampling (null is distinct from absent). |

**CAUTION:** `batch_size` changes meaning between modes. Training: gauges. Testing: days. This is not a typo — it's in the YAML comments.

---

## `kan_head:` section

The KAN head architecture: `Linear(F,H) → KanLayer(H,H) × num_hidden_layers → Linear(H,P) → Sigmoid`.

| Key | Type | Code default | Production value | Notes |
|---|---|---|---|---|
| `hidden_size` | integer | none | `21` | Hidden dimension H. |
| `num_hidden_layers` | integer | none | `2` | Inner KanLayer repetitions. ALL receive the SAME seed (DDR `kan.py` quirk, preserved for parity). |
| `grid` | integer | `5` | `50` | B-spline grid intervals per KAN edge (`num` in pykan). Production overrides the code default. |
| `k` | integer | `3` | `2` | B-spline order. DDR overrides pykan's default of 3 to 2 in production; keep 2 for parity. |
| `input_var_names` | list of strings | none | 10 attributes (see below) | Column names in `attributes` NetCDF. |
| `learnable_parameters` | list of strings | none | `[n, q_spatial, p_spatial]` | Parameters the KAN head emits. Must have matching entries in `params.parameter_ranges`. |
| `disaggregation` | block \| absent | absent | absent | Enables the daily→hourly disaggregation head (see sub-block below). Absent = flat repeat-24. |

**Production `input_var_names` (10 attributes):**
```
SoilGrids1km_clay, aridity, meanelevation, meanP, NDVI,
meanslope, log10_uparea, SoilGrids1km_sand, ETPOT_Hargr, Porosity
```

### `kan_head.disaggregation:` sub-block

Presence of this block enables the learnable daily→hourly disaggregation head (`src/nn/disagg_head.rs`). Absence = flat `repeat-24` (backward-compatible default).

| Key | Type | Default | Notes |
|---|---|---|---|
| `hidden_size` | integer | `16` | Hidden dimension of the disagg MLP. |
| `use_attributes` | bool | `true` | Condition on catchment attributes. |
| `use_precip` | bool | `false` | Condition on 72-h AORC precip window `[d-1, d, d+1]`. Requires `data_sources.aorc_precip`. |
| `use_temp` | bool | `false` | Condition on 72-h AORC temperature window. Also requires `data_sources.aorc_precip`. |

**If `use_precip: true` and `data_sources.aorc_precip` is absent:** `MeritGagesDataset::open` errors at runtime (not at config load time). The missing precip source cannot silently degrade.

---

## `params:` section (routing engine)

| Key | Type | Code default | Production value | Notes |
|---|---|---|---|---|
| `sparse_solver` | `cpu` \| `cuda` | `cpu` | `cuda` | `cuda` uses cuSPARSE for the triangular solve. Falls back to `cpu` on non-CUDA backends with a WARN log. |
| `use_cuda_graphs` | bool | `false` | `true` | Capture the routing forward as a CUDA graph; faster replay each timestep. **See guards below.** |
| `use_leakance` | bool | `false` | `false` | Enable the GW–SW water-loss term. **See guards below.** Experimental as of 2026-07-05. |
| `tau` | integer | `3` | `3` (not set in YAML) | Muskingum routing sub-step count. Rarely changed. |
| `log_space_parameters` | list of strings | `["p_spatial"]` | `["p_spatial"]` | Parameters whose range spans decades; KAN output is exp-scaled before routing. |
| `defaults` | map str→f32 | `{p_spatial: 21.0}` | `{p_spatial: 21.0}` | Fixed values for parameters NOT in `learnable_parameters`. |

### `params.parameter_ranges:` sub-block

Physical `[min, max]` each sigmoid-normalized KAN output maps onto. All defaults are defined in `src/config.rs::ParameterRanges::default()`.

| Key | Default range | Log-space | Notes |
|---|---|---|---|
| `n` | `[0.015, 0.25]` | No | Manning's roughness coefficient. |
| `q_spatial` | `[0.0, 1.0]` | No | Leopold & Maddock width–depth exponent (`top_width = p·depth^q`). |
| `p_spatial` | `[1.0, 200.0]` | Yes | Leopold & Maddock width coefficient. In log-space by default. |
| `x_storage` | `[0.0, 0.5]` | No | Muskingum storage weight X. Only consumed when listed in `learnable_parameters`; otherwise routing uses constant 0.3. |
| `K_D` | `[1e-8, 1e-6]` | Yes (add to `log_space_parameters`) | Hydraulic exchange rate (1/s). Leakance only. Note: uppercase in YAML. |
| `d_gw` | `[-2.0, 2.0]` | No | Groundwater depth offset (m). Leakance only. |
| `leakance_factor` | `[0.0, 1.0]` | No | Dimensionless leakance scale. Leakance only. |

### `params.attribute_minimums:` sub-block

Physical floor applied during routing for numerical stability.

| Key | Default | Units |
|---|---|---|
| `discharge` | `1.0e-4` | m³/s |
| `slope` | `1.0e-3` | m/m |
| `velocity` | `0.01` | m/s |
| `depth` | `0.01` | m |
| `bottom_width` | `0.01` | m |

---

## Guards enforced at `Config::from_yaml_file` (load-time errors)

| Guard | Trigger | Error message substring |
|---|---|---|
| Mode/workflow conflict | `mode: training` + `workflow: eval`, or `mode: testing` + `workflow: train` | `"conflicting top-level keys"` |
| Partial adjacency | Only one of `conus_adjacency` / `gages_adjacency` set | `"gages_adjacency\` is missing"` or `"conus_adjacency\` is missing"` |
| No adjacency sources | Neither adjacency zarrs nor `geospatial_fabric` | `"adjacency sources are missing"` |
| Fabric layer on non-gpkg | `geospatial_fabric_layer` set with `.shp`/`.dbf` fabric | `"geospatial_fabric_layer"` and `".gpkg"` |
| Leakance + CUDA graphs | `use_leakance: true` and `use_cuda_graphs: true` | `"use_leakance"` and `"use_cuda_graphs"` |

---

## Production configs vs experimental configs

| Config file | Status | Key differences from production |
|---|---|---|
| `config/merit_training.yaml` | Production template | `grid:50`, `k:2`, `use_cuda_graphs:true`, managed adjacency via `geospatial_fabric`, no leakance, no disaggregation, L1 loss |
| `config/experiments/leakance_hourly_on.yaml` | Experimental (2026-07-01, GO-marginal) | `use_leakance:true`, `use_cuda_graphs:false`, `K_D`/`d_gw`/`leakance_factor` in head, precip disaggregation enabled, explicit zarr adjacency paths |
| `config/experiments/leakance_daily_on.yaml` | Experimental (2026-07-01) | Same as `leakance_hourly_on.yaml` but NO disaggregation block, NO `aorc_precip` |
| `config/sources/conus.yaml` | Source group | CONUS workstation paths without AORC precip |
| `config/sources/conus-hourly.yaml` | Source group | CONUS + `aorc_precip: /mnt/ssd1/data/aorc/merit_unit_catchments.zarr` |
| `config/sources/global.yaml` | Source group | GPFS global paths |

**Source groups** are text-spliced into `data_sources:` by `ddrs sources use <name>`. They do not set `kan_head` or `params`.

---

## Critical runtime traps

### STALE-BINARY TRAP
`cargo build` does NOT update `~/.cargo/bin/ddrs`. After ANY change to `src/`, run:
```bash
cargo install --path .
# or faster if target/release is current:
cargo build --release --bin ddrs && cp target/release/ddrs ~/.cargo/bin/ddrs
```
Self-check: current checkpoints are **directories** (`.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/head.mpk`). Flat `.mpk` files mean a stale binary ran.

### CUDA graphs mask NaN
`use_cuda_graphs: true` captures a finite forward pass graph. If a NaN appears in a subsequent forward (different inputs), the graph replays stale finite values — you get a finite loss with no error. To validate forwards, test with `use_cuda_graphs: false`. This is why `use_leakance: true` + `use_cuda_graphs: true` is rejected at config load time.

### Leakance status (as of 2026-07-05)
The leakance 2×2 (forcing × leakance) returned a **GO-marginal** verdict:
- `|zeta| > 0.01 m³/s` on 10.4% of 64,892 eval reaches (meets ≥10% bar)
- Leakance helps under hourly forcing on the losing-stream subset (ΔNSE +0.0005, ΔKGE +0.0018, 55.5% of gauges improve)
- Leakance hurts under daily forcing (ΔNSE −0.0017, ΔKGE −0.0009, 35.6%)
- `K_D` pinned at ceiling `1e-6` (binding constraint; widening NOT recommended as of 2026-07-05)

Diagnosis hypotheses (2026-07-02):
- H2 (head throttling): SUPPORTED
- H4 (gauge bias): SUPPORTED
- H5 (equifinality with `n`): SUPPORTED
- H1 (K_D box too narrow): REFUTED
- H3 (KAN capacity): REFUTED
- H6, H7: REFUTED

**Leakance identifiability is NOT proven.** The positive-control synthetic recovery experiment (2026-07-04) FAILED: recovery ratio 0.009 vs ≥0.5 bar. Root cause: windowed training objective has ~130x hotstart-transient noise floor. Phase B (state-cache hotstart, ≤0.25 mean L1 noise floor target) is required before any identifiability claim.

---

## How to add a new routing parameter (checklist)

A "routing parameter" is a per-reach scalar the KAN head predicts and the MC solver consumes. Example: adding a new parameter `my_param`.

- [ ] **1. Add to `ParameterRanges` struct** (`src/config.rs`):
  ```rust
  pub my_param: [f32; 2],
  ```
  Add a default in `ParameterRanges::default()`.

- [ ] **2. Add YAML key parsing** in `From<ParamsRaw> for Params` (`src/config.rs`):
  ```rust
  if let Some(v) = r.parameter_ranges.get("my_param") {
      p.parameter_ranges.my_param = *v;
  }
  ```

- [ ] **3. Add to `config/merit_training.yaml`** under `params.parameter_ranges:` (if it has a production-relevant range).

- [ ] **4. Wire into routing** (`src/routing/mmc.rs` or a new module): consume `Params.parameter_ranges.my_param` in `setup_inputs` or `route_timestep`. Follow the `denormalize` pattern in `src/routing/utils.rs`.

- [ ] **5. Add to `kan_head.learnable_parameters:`** in any experiment config that uses it.

- [ ] **6. Add to `params.log_space_parameters:`** if the range spans decades.

- [ ] **7. If log-space:** add to `log_space_parameters` list in `params.log_space_parameters` in YAML and ensure `denormalize` handles it.

- [ ] **8. Write a gradient-exactness test** if the new parameter enters a custom backward op. Run:
  ```bash
  cargo test --test <gradcheck_test>
  cargo run --release --example compare_ddr_sandbox
  ```
  The sandbox must still report `ABSOLUTE MATCH`.

---

## How to add a new training-mode boolean flag (checklist)

Example: adding `use_my_feature: bool` under `params:`.

- [ ] **1. Add field to `Params` struct** (`src/config.rs`):
  ```rust
  pub use_my_feature: bool,
  ```
  Add `use_my_feature: false` to `Params::default()`.

- [ ] **2. Add to `ParamsRaw`** and parse in `From<ParamsRaw> for Params`:
  ```rust
  // in ParamsRaw:
  use_my_feature: Option<bool>,
  // in From impl:
  if let Some(b) = r.use_my_feature { p.use_my_feature = b; }
  ```

- [ ] **3. Add validation if needed** (`validate_*` functions in `src/config.rs`). Call from `from_yaml_file_with_mode`. Validation errors must include the YAML key name and the reason.

- [ ] **4. Add a test** in the `#[cfg(test)]` block at the bottom of `src/config.rs` covering: flag defaults to false, flag parses true, any guard is rejected.

- [ ] **5. Thread through call sites**: training bootstrap (`src/training/bootstrap.rs`), eval (`src/cli/eval.rs`), and any other entrypoints that construct `MuskingumCunge` or read `Params`.

- [ ] **6. Document in `config/merit_training.yaml`** as a commented-out key with explanation if it has production relevance.

---

## Quick reference: minimal leakance-on config diff

Starting from `config/merit_training.yaml`, three changes activate leakance:

```yaml
# 1. Under params:
params:
  use_leakance: true
  use_cuda_graphs: false  # REQUIRED — leakance + cuda_graphs is rejected at load time
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-6]
    d_gw: [-2.0, 2.0]
    leakance_factor: [0.0, 1.0]
  log_space_parameters:
    - p_spatial
    - K_D  # ADD — K_D range spans decades

# 2. Under kan_head.learnable_parameters:
  learnable_parameters:
    - n
    - q_spatial
    - p_spatial
    - K_D
    - d_gw
    - leakance_factor
```

See `config/experiments/leakance_hourly_on.yaml` for the full working example.

---

## Provenance and maintenance

Ground truth files read to produce this skill (verify before re-editing):
```bash
# Config struct (all fields, defaults, guards):
grep -n "pub use_" /home/tbindas/projects/ddrs/src/config.rs

# Production defaults verified from:
grep -n "fn default" /home/tbindas/projects/ddrs/src/config.rs

# Production YAML (single source of truth for hyperparameter values):
cat /home/tbindas/projects/ddrs/config/merit_training.yaml

# Leakance experiment config:
cat /home/tbindas/projects/ddrs/config/experiments/leakance_hourly_on.yaml

# Load-time guard tests (exhaustive):
grep -n "#\[test\]" /home/tbindas/projects/ddrs/src/config.rs | head -40
```

Config struct location: `/home/tbindas/projects/ddrs/src/config.rs`
Production template: `/home/tbindas/projects/ddrs/config/merit_training.yaml`
Experiment configs: `/home/tbindas/projects/ddrs/config/experiments/`
Source group configs: `/home/tbindas/projects/ddrs/config/sources/`

Volatile facts date-stamped: 2026-07-05. Re-verify leakance status, CONUS metric baselines, and K_D ceiling diagnosis before citing in new experiments.
