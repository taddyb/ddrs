//! `MeritGagesDataset` + `RoutingBatch`.
//!
//! Mirrors `~/projects/ddr/src/ddr/geodatazoo/merit.py::Merit` for the
//! training-mode (`_init_training` + `_collate_gages`) path. Other modes
//! (target_catchments, all_catchments) are out of scope for SP-3.

use std::cell::OnceCell;
use std::sync::Arc;

use ndarray::{Array1, Array2};

use crate::config::Config;
use crate::data::collate::{build_flow_scale, compress, union_subgraphs};
use crate::data::dates::{Frequency, RhoWindow, TimeAxis};
use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, Staid};
use crate::data::statistics::{fill_nans, AttrStats};
use crate::data::store::{
    AorcPrecipStore, AttributesStore, ConusAdjacencyStore, GageMetadata, GagesAdjacencyStore,
    ObservationsStore, StateCache, StreamflowSource,
};
use crate::sparse::SparseAdjacency;

/// One batch of inputs for the MC routing engine + MLP head.
///
/// All tensors are plain `ndarray::Array` here — SP-4's training loop
/// materializes them onto a BURN backend at the device boundary.
#[derive(Debug)]
pub struct RoutingBatch {
    pub adjacency: SparseAdjacency,
    /// Normalized attributes, shape `(N, F)`. Caller-major to match the
    /// KAN head input contract (`src/nn/kan_head.rs::KanHead::forward`).
    pub spatial_attributes_normalized: Array2<f32>,
    /// q' streamflow forcing, shape `(T_hours, N)`. Already multiplied by
    /// `flow_scale` per column. The flat `repeat-24` upsampling of
    /// `q_prime_daily`; used by the no-disaggregation path.
    pub q_prime: Array2<f32>,
    /// Daily q' forcing, shape `(D, N)` (pre-trim, flow-scaled). Input to the
    /// learnable disaggregation head; `None` of it is upsampled here.
    pub q_prime_daily: Array2<f32>,
    /// Hourly AORC precip, shape `(T_hours, N)`, **already normalized** per
    /// reach (per-window z-score of `log1p(precip)`, see `normalize_precip`).
    /// Empty `(0, N)` when no precip store is wired. Conditions the
    /// disaggregation head.
    pub precip_hourly: Array2<f32>,
    /// Hourly AORC temperature, shape `(T_hours, N)`, **already normalized**
    /// per reach (z-score). Empty `(0, N)` when temp conditioning is off.
    pub temp_hourly: Array2<f32>,
    /// USGS observations, shape `(T_days, G)`. NaN-tolerant.
    pub observations: Array2<f32>,
    /// Per-gauge observed std over the training period, aligned to
    /// `gauge_staids`. Empty unless `loss.kind: nse-batch`. See
    /// `MeritGagesDataset::gauge_obs_std`.
    pub gauge_obs_std: Vec<f32>,
    /// For each gauge in `gauge_staids`, list of compressed-cols whose row
    /// equals the gauge's outlet position. SP-4 reads gauge predictions
    /// out of the engine's `(N, T)` output via these indices.
    pub outflow_idx: Vec<Vec<usize>>,
    pub gauge_staids: Vec<Staid>,
    /// Compressed COMIDs in topological position order, length `N`.
    pub divide_comids: Vec<Comid>,
    /// Per-segment flow scaling factors, length `N`. Already applied to
    /// `q_prime` — kept here for diagnostics / loss reconstruction.
    pub flow_scale: Vec<f32>,
    pub window: RhoWindow,
    /// Per-reach discharge at window start (m³/s), batch-network order,
    /// length `N`. Populated when `experiment.state_cache` is configured;
    /// `None` otherwise → Task 3's forward pass falls back to the hotstart
    /// heuristic (byte-identical to no-cache behavior).
    pub initial_state: Option<Array1<f32>>,
    /// Precomputed per-reach impervious hard-zero mask (0.0 = impervious,
    /// 1.0 = leakance allowed), shape `(N,)`. Built from `corridor_impervious`
    /// attribute column using `cfg.params.leakance_impervious_threshold`.
    /// `None` when the column is absent from `attr_names` or leakance is off —
    /// both are byte-identical no-ops (`SpatialParameters::impervious_mask`
    /// stays `None`). NaN (no StreamCat coverage) → 1.0 (absence of
    /// imperviousness data ≠ concrete; leakance allowed).
    pub impervious_mask: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// StaticNetworkCache — lazy, all-gauges network for test mode
// ---------------------------------------------------------------------------

/// Lazy cache of the all-gauges static network, computed on first call to
/// `MeritGagesDataset::collate_window`. Reused across every test-mode chunk.
///
/// All fields are computed once and never mutated thereafter.
struct StaticNetworkCache {
    adjacency: SparseAdjacency,
    outflow_idx: Vec<Vec<usize>>,
    flow_scale: Vec<f32>,
    /// Normalized attributes, shape `(N_active, F)`.
    spatial_attributes_normalized: Array2<f32>,
    /// Full-period observations `(n_days_full, G)`. Sliced per `collate_window` call.
    full_observations: Array2<f32>,
    /// All filtered gauge STAIDs (in subgraph union order).
    gauge_staids: Vec<Staid>,
    /// Active COMID list in topological order, length `N_active`.
    divide_comids: Vec<Comid>,
    /// Precomputed impervious mask (same as `RoutingBatch::impervious_mask`).
    /// Built once at static-network construction; cloned into every eval batch.
    impervious_mask: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// RoutingTensors + to_tensors
// ---------------------------------------------------------------------------

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

/// BURN-tensor-lifted version of `RoutingBatch`. Produced via
/// `RoutingBatch::to_tensors`. `observations` stays on CPU as it's only
/// used for masking + comparison at loss time.
pub struct RoutingTensors<B: Backend> {
    pub adjacency: SparseAdjacency,
    /// Normalized attributes, shape `(N, F)`.
    pub spatial_attributes: Tensor<B, 2>,
    /// q' streamflow, shape `(T_hours, N)`. Not yet Autodiff-wrapped.
    pub q_prime: Tensor<B, 2>,
    /// Daily q' forcing, shape `(D, N)`. Input to the disaggregation head.
    pub q_prime_daily: Tensor<B, 2>,
    /// Hourly normalized precip, shape `(T_hours, N)` (empty `(0, N)` when no
    /// precip store). Input to the precip-driven disaggregation head.
    pub precip_hourly: Tensor<B, 2>,
    /// Hourly normalized temperature, shape `(T_hours, N)` (empty when off).
    pub temp_hourly: Tensor<B, 2>,
    /// Observations stay on CPU.
    pub observations: Array2<f32>,
    /// Flat concat of `outflow_idx`, shape `(sum_g len(outflow_idx[g]),)`.
    pub flat_indices: Tensor<B, 1, Int>,
    /// Per-flat-index gauge group id, same shape as `flat_indices`.
    pub group_ids: Tensor<B, 1, Int>,
    pub num_gauges: usize,
    pub gauge_staids: Vec<Staid>,
    pub window: RhoWindow,
    /// Per-reach window-start discharge, shape `(N,)`.
    /// `None` when `experiment.state_cache` is not configured.
    /// Task 3 consumes this to seed the router instead of the hotstart
    /// heuristic. It is a constant tensor (no grad); autograd topology
    /// is unchanged relative to the no-cache path.
    pub initial_state: Option<Tensor<B, 1>>,
    /// Per-reach impervious hard-zero mask, shape `(N,)`. Constant — no
    /// autograd. `None` when `corridor_impervious` is absent from attributes
    /// or leakance is disabled (byte-identical back-compat with PC1).
    pub impervious_mask: Option<Tensor<B, 1>>,
}

impl RoutingBatch {
    /// Lift plain ndarray buffers onto a BURN device.
    ///
    /// Pre-computes `flat_indices` + `group_ids` from `outflow_idx` here,
    /// mirrors DDR `mmc.py:347-358`.
    pub fn to_tensors<B: Backend>(self, device: &B::Device) -> RoutingTensors<B> {
        // 1. Pre-compute flat/group from outflow_idx.
        let mut flat: Vec<i32> = Vec::new();
        let mut group: Vec<i32> = Vec::new();
        for (g_idx, segs) in self.outflow_idx.iter().enumerate() {
            flat.extend(segs.iter().map(|&s| s as i32));
            group.extend(std::iter::repeat_n(g_idx as i32, segs.len()));
        }

        // 2. Lift spatial_attributes (N, F) — already owned + contiguous after reversed_axes().into_owned().
        let (rows, cols) = (
            self.spatial_attributes_normalized.shape()[0],
            self.spatial_attributes_normalized.shape()[1],
        );
        let attrs_vec: Vec<f32> = self
            .spatial_attributes_normalized
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset()
            .0;
        let spatial_attributes =
            Tensor::<B, 2>::from_data(TensorData::new(attrs_vec, [rows, cols]), device);

        // 3. Lift q_prime (T_hours, N).
        let (t_hours, n) = (self.q_prime.shape()[0], self.q_prime.shape()[1]);
        let q_vec: Vec<f32> = self
            .q_prime
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset()
            .0;
        let q_prime = Tensor::<B, 2>::from_data(TensorData::new(q_vec, [t_hours, n]), device);

        // 3b. Lift q_prime_daily (D, N).
        let (d_days, dn) = (self.q_prime_daily.shape()[0], self.q_prime_daily.shape()[1]);
        let qd_vec: Vec<f32> = self
            .q_prime_daily
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset()
            .0;
        let q_prime_daily = Tensor::<B, 2>::from_data(TensorData::new(qd_vec, [d_days, dn]), device);

        // 3c. Lift precip_hourly (T_hours, N) — empty (0, N) when no store.
        let (p_hours, pn) = (self.precip_hourly.shape()[0], self.precip_hourly.shape()[1]);
        let p_vec: Vec<f32> = self
            .precip_hourly
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset()
            .0;
        let precip_hourly = Tensor::<B, 2>::from_data(TensorData::new(p_vec, [p_hours, pn]), device);

        // 3d. Lift temp_hourly (T_hours, N) — empty (0, N) when temp off.
        let (te_hours, ten) = (self.temp_hourly.shape()[0], self.temp_hourly.shape()[1]);
        let te_vec: Vec<f32> = self
            .temp_hourly
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset()
            .0;
        let temp_hourly = Tensor::<B, 2>::from_data(TensorData::new(te_vec, [te_hours, ten]), device);

        // 4. Lift flat_indices + group_ids as Int tensors.
        let flat_indices = Tensor::<B, 1, Int>::from_data(TensorData::from(flat.as_slice()), device);
        let group_ids = Tensor::<B, 1, Int>::from_data(TensorData::from(group.as_slice()), device);

        let num_gauges = self.gauge_staids.len();

        // 5. Lift initial_state (N,) — constant, no grad.
        let initial_state = self.initial_state.map(|arr| {
            let n = arr.len();
            let vec: Vec<f32> = arr.into_raw_vec_and_offset().0;
            Tensor::<B, 1>::from_data(TensorData::new(vec, [n]), device)
        });

        // 6. Lift impervious_mask (N,) — constant, no grad.
        let impervious_mask = self.impervious_mask.map(|mask| {
            let n = mask.len();
            Tensor::<B, 1>::from_data(TensorData::new(mask, [n]), device)
        });

        RoutingTensors {
            adjacency: self.adjacency,
            spatial_attributes,
            q_prime,
            q_prime_daily,
            precip_hourly,
            temp_hourly,
            observations: self.observations,
            flat_indices,
            group_ids,
            num_gauges,
            gauge_staids: self.gauge_staids,
            window: self.window,
            initial_state,
            impervious_mask,
        }
    }
}

// ---------------------------------------------------------------------------
// MeritGagesDataset
// ---------------------------------------------------------------------------

pub struct MeritGagesDataset {
    pub(crate) conus: Arc<ConusAdjacencyStore>,
    pub(crate) gages_adj: Arc<GagesAdjacencyStore>,
    pub(crate) attrs: Arc<AttributesStore>,
    #[allow(dead_code)] // retained for diagnostics; means/stds are pre-derived at open() time
    pub(crate) stats: Arc<AttrStats>,
    pub(crate) gages: Arc<GageMetadata>,
    pub(crate) streamflow: Arc<StreamflowSource>,
    /// Optional hourly AORC store (precip + temperature) for the disaggregation
    /// head. `None` ⇒ the head conditions on daily Q' only (or disagg is off).
    pub(crate) precip: Option<Arc<AorcPrecipStore>>,
    /// Whether to read+carry the precip / temperature channels (mirrors the
    /// head's `use_precip` / `use_temp`). Avoids reading a channel the head
    /// would ignore.
    pub(crate) want_precip: bool,
    pub(crate) want_temp: bool,
    pub(crate) observations: Arc<ObservationsStore>,
    pub(crate) time_axis: TimeAxis,
    pub(crate) attr_names: Vec<String>,
    pub(crate) means: Array1<f32>,
    pub(crate) stds: Array1<f32>,
    /// Filtered training gauges (DA_VALID + adjacency presence + non-headwater).
    pub(crate) gauges: Vec<Staid>,
    /// Lazily-built all-gauges static network; populated on first call to
    /// `collate_window`. Interior mutability via `OnceCell` (single-threaded
    /// dataset access — the training loop drives it from one thread).
    static_network: OnceCell<StaticNetworkCache>,
    /// Per-gauge observed-discharge std over the FULL training period, keyed
    /// by STAID. Built once on first use; only populated when the configured
    /// objective needs it (`want_gauge_std`), since it costs a full-window
    /// observation read. Fixed across windows on purpose — a per-window std
    /// would make the objective drift between micro-batches and break
    /// gradient-accumulation exactness. Mirrors dHBV's `stdarray`.
    gauge_std: OnceCell<std::collections::HashMap<Staid, f32>>,
    /// Whether the configured loss needs `gauge_std` (`loss.kind: nse-batch`).
    want_gauge_std: bool,
    /// `params.ddr_match`, captured at open. Selects the `outflow_idx`
    /// convention in `collate::compress`: `true` (default) reproduces DDR's
    /// upstream-cols behaviour, `false` reads the gauge's own reach.
    ddr_match: bool,
    /// Optional day-boundary discharge state cache. `None` ⇒ every code path
    /// byte-identical to no-cache behavior (`RoutingBatch::initial_state = None`).
    state_cache: Option<crate::data::store::StateCache>,
    /// When `Some(threshold)`, `corridor_impervious` is in `attr_names` AND
    /// leakance is enabled — `collate`/`build_static_network` will call
    /// `build_impervious_mask`. `None` ⇒ mask is never built (back-compat no-op).
    leakance_impervious_threshold: Option<f32>,
}

/// Reject the disaggregation head when the streamflow store is hourly-native:
/// disaggregating an already-hourly signal is a config contradiction, and
/// after the 2026-07-01 stale-binary incident nothing in the forcing path is
/// allowed to silently degrade.
fn validate_disagg_vs_resolution(
    resolution: Frequency,
    has_disagg: bool,
    streamflow_path: &std::path::Path,
) -> Result<()> {
    if resolution == Frequency::Hourly && has_disagg {
        return Err(DataError::Malformed {
            path: streamflow_path.to_path_buf(),
            message: "kan_head.disaggregation is set but the streamflow store is \
                      hourly-native; remove the disaggregation block (an hourly \
                      store needs no daily→hourly head)"
                .into(),
        });
    }
    Ok(())
}

impl MeritGagesDataset {
    /// Open all five stores and apply the training-mode filter pipeline.
    /// Mirrors `Merit.__init__` + `_init_training` in `geodatazoo/merit.py`.
    pub fn open(cfg: &Config) -> Result<Self> {
        let ds = cfg.data_sources.as_ref().ok_or_else(|| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: "data_sources section missing".into(),
        })?;
        let exp = cfg.experiment.as_ref().ok_or_else(|| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: "experiment section missing".into(),
        })?;
        let head_cfg = cfg.kan_head.as_ref().ok_or_else(|| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: "kan_head section missing".into(),
        })?;

        // ---------- 1. Adjacency + gage CSV ----------
        // Defensive: `ddrs plan` resolves adjacency (explicit or managed build)
        // and materializes the paths into the config before the dataset opens.
        let conus_path = ds.conus_adjacency.as_ref().ok_or_else(|| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: "conus_adjacency not resolved — open the dataset via `ddrs run` \
                      (which resolves adjacency), or set conus_adjacency/gages_adjacency \
                      explicitly".into(),
        })?;
        let conus = Arc::new(ConusAdjacencyStore::open(conus_path)?);
        let gage_meta = GageMetadata::open(&ds.gages)?;

        // Filter 1: DA_VALID drop. Rows without the column (e.g. the v3.1
        // global per-zone CSVs carry no DA_VALID) pass — only an explicit
        // False drops the gauge.
        let pre_filter = gage_meta.rows.len();
        let da_valid: Vec<Staid> = gage_meta
            .rows
            .iter()
            .filter(|r| r.da_valid.unwrap_or(true))
            .map(|r| r.staid.clone())
            .collect();
        eprintln!(
            "DA_VALID filter: kept {}/{} gauges",
            da_valid.len(),
            pre_filter
        );

        // Open the gages adjacency store with the DA_VALID set.
        let gages_adj_path = ds.gages_adjacency.as_ref().ok_or_else(|| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: "gages_adjacency not resolved — open the dataset via `ddrs run` \
                      (which resolves adjacency), or set conus_adjacency/gages_adjacency \
                      explicitly".into(),
        })?;
        let gages_adj = Arc::new(GagesAdjacencyStore::open(gages_adj_path, &da_valid)?);

        // Filter 2 + 3: adjacency presence + headwater drop.
        let mut gauges: Vec<Staid> = Vec::new();
        let mut n_missing = 0usize;
        let mut n_headwater = 0usize;
        for s in &da_valid {
            let Some(g) = gages_adj.get(s) else {
                n_missing += 1;
                continue;
            };
            if g.indices_0.is_empty() {
                n_headwater += 1;
                continue;
            }
            gauges.push(s.clone());
        }
        eprintln!(
            "gages_adjacency filter: kept {} gauges (dropped {} missing, {} headwater)",
            gauges.len(),
            n_missing,
            n_headwater
        );

        // ---------- 2. Attributes + statistics ----------
        let attr_names: Vec<String> = head_cfg.input_var_names.clone();
        let (attrs, stats, means, stds) = if ds.attributes.len() == 1 {
            // Single-path: byte-identical to the pre-C0 behavior.
            let attrs = AttributesStore::open(&ds.attributes[0], &attr_names, &conus.order)?;
            let stats_path = stats_path_from_attrs(&ds.attributes[0]);
            let stats = AttrStats::open(&stats_path)?;
            let means = stats.means_f32(&attr_names);
            let stds = stats.stds_f32(&attr_names);
            (Arc::new(attrs), Arc::new(stats), means, stds)
        } else {
            // Multi-path: COMID-aligned merge across stores; NaN-fill per-store gaps.
            let attrs = AttributesStore::open_multi(&ds.attributes, &attr_names, &conus.order)?;
            let (means, stds) = load_merged_stats(&ds.attributes, &attr_names)?;
            // stats field is diagnostics-only (dead_code); use an empty placeholder.
            let placeholder_stats = AttrStats {
                path: ds.attributes[0].clone(),
                by_name: std::collections::HashMap::new(),
            };
            (Arc::new(attrs), Arc::new(placeholder_stats), means, stds)
        };

        // ---------- 3. Icechunk stores ----------
        let streamflow = Arc::new(StreamflowSource::open(&ds.streamflow)?);
        // The smoke-train self-check line: proves which read path executed.
        eprintln!("streamflow resolution: {:?}", streamflow.resolution());
        validate_disagg_vs_resolution(
            streamflow.resolution(),
            head_cfg.disaggregation.is_some(),
            &ds.streamflow,
        )?;
        let observations = Arc::new(ObservationsStore::open(&ds.observations)?);

        // Optional hourly precip store for the disaggregation head. The head
        // always requires precip when enabled — no separate toggle. (`temp`
        // conditioning was removed from the disagg head; `want_temp` stays
        // hardcoded false so `read_temp_window`/`read_temp_test_window` keep
        // compiling as dead-but-harmless empty-array paths rather than a
        // wider dataset.rs refactor out of scope for this change.)
        let want_precip = head_cfg.disaggregation.is_some();
        let want_temp = false;
        let precip = match (&ds.aorc_precip, want_precip) {
            (Some(p), _) => {
                let store = AorcPrecipStore::open(p)?;
                eprintln!(
                    "AORC precip store: {} catchments, hourly {}..",
                    store.n_catchments(),
                    store.time_start
                );
                Some(Arc::new(store))
            }
            (None, true) => {
                return Err(DataError::Malformed {
                    path: std::path::PathBuf::from("<config>"),
                    message: "kan_head.disaggregation is set but data_sources.aorc_precip \
                              is not — the disaggregation head always requires precip".into(),
                });
            }
            (None, false) => None,
        };

        // Filter 4: drop gauges the observation store has no series for —
        // observation reads hard-error on missing STAIDs, and the global
        // v3.1 gage CSVs list a few dozen gauges absent from the obs zarr.
        let pre_obs = gauges.len();
        gauges.retain(|s| observations.contains(s));
        if gauges.len() != pre_obs {
            eprintln!(
                "observations filter: kept {}/{} gauges (dropped {} absent from {})",
                gauges.len(),
                pre_obs,
                pre_obs - gauges.len(),
                ds.observations.display()
            );
        }

        // ---------- 4. Time axis from experiment dates ----------
        let time_axis = parse_experiment_axis(&exp.start_time, &exp.end_time)?;

        // ---------- 5. Optional state cache ----------
        let state_cache = match &exp.state_cache {
            Some(p) => {
                eprintln!("state cache: opening {}", p.display());
                Some(StateCache::open(p)?)
            }
            None => None,
        };

        // Build the impervious threshold when leakance is on AND the attribute
        // is present. `None` ⇒ no mask built (all-ones back-compat no-op).
        let leakance_impervious_threshold = if cfg.params.use_leakance
            && attr_names.iter().any(|n| n == "corridor_impervious")
        {
            Some(cfg.params.leakance_impervious_threshold)
        } else {
            None
        };

        Ok(Self {
            conus,
            gages_adj,
            attrs,
            stats,
            gages: Arc::new(gage_meta),
            streamflow,
            precip,
            want_precip,
            want_temp,
            observations,
            time_axis,
            attr_names,
            means,
            stds,
            gauges,
            static_network: OnceCell::new(),
            gauge_std: OnceCell::new(),
            want_gauge_std: cfg
                .experiment
                .as_ref()
                .map(|e| e.loss.kind == crate::config::LossKind::NseBatch)
                .unwrap_or(false),
            ddr_match: cfg.params.ddr_match,
            state_cache,
            leakance_impervious_threshold,
        })
    }

    pub fn len(&self) -> usize {
        self.gauges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gauges.is_empty()
    }

    pub fn staids(&self) -> &[Staid] {
        &self.gauges
    }

    /// Per-gauge observed-discharge standard deviation over the full training
    /// period, in the order of `staids`. Empty vec when the configured loss
    /// doesn't need it. Gauges with <2 finite observations get 0.0 (the loss
    /// adds `eps` to the denominator, so they stay finite).
    ///
    /// Fixed across windows on purpose — a per-window std would make the
    /// objective drift between micro-batches and break gradient-accumulation
    /// exactness. Mirrors dHBV's `stdarray` (hydroDL `crit.py::NSELossBatch`).
    pub fn gauge_obs_std(&self, staids: &[Staid]) -> Result<Vec<f32>> {
        if !self.want_gauge_std {
            return Ok(Vec::new());
        }
        if self.gauge_std.get().is_none() {
            let full = crate::data::dates::RhoWindow {
                start_day_idx: 0,
                rho_days: self.time_axis.num_days,
                window_start: self.time_axis.start,
            };
            let obs = self.observations.read_window(&full, &self.gauges)?;
            let mut m = std::collections::HashMap::with_capacity(self.gauges.len());
            for (j, s) in self.gauges.iter().enumerate() {
                let vals: Vec<f32> =
                    obs.column(j).iter().copied().filter(|v| v.is_finite()).collect();
                let std = if vals.len() < 2 {
                    0.0
                } else {
                    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                    (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32)
                        .sqrt()
                };
                m.insert(s.clone(), std);
            }
            let mut sorted: Vec<f32> = m.values().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!(
                "gauge obs std (training period, {} days): {} gauges, median {:.3} m3/s",
                self.time_axis.num_days,
                m.len(),
                sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
            );
            let _ = self.gauge_std.set(m);
        }
        let map = self.gauge_std.get().expect("populated above");
        Ok(staids.iter().map(|s| map.get(s).copied().unwrap_or(0.0)).collect())
    }

    pub fn time_axis(&self) -> &TimeAxis {
        &self.time_axis
    }

    /// Returns the cached full-period observations `(n_days_full, n_all_gauges)`.
    ///
    /// Triggers static-network build if not already cached. Reads observations
    /// once during cache build; does NOT touch the streamflow store.
    pub fn full_observations(&self) -> Result<&Array2<f32>> {
        Ok(&self.get_or_build_static_network()?.full_observations)
    }

    /// Build one `RoutingBatch` from a STAID subset + a time window.
    ///
    /// Mirrors `Merit._collate_gages` in `geodatazoo/merit.py:245-330`.
    pub fn collate(
        &self,
        batch_staids: &[Staid],
        window: &RhoWindow,
    ) -> Result<RoutingBatch> {
        // ----- 1. Subgraph union + compression -----
        let unioned = union_subgraphs(batch_staids, &self.gages_adj);
        if unioned.gauges.is_empty() {
            return Err(DataError::Malformed {
                path: std::path::PathBuf::from("<collate>"),
                message: format!(
                    "no batch gauges present in gages_adjacency (asked for {})",
                    batch_staids.len()
                ),
            });
        }
        // Pull out present STAIDs (caller-order, skipping missing) before
        // moving `unioned` into compress().
        let gauge_staids: Vec<Staid> =
            unioned.gauges.iter().map(|(s, _, _)| s.clone()).collect();

        let compressed = compress(&unioned, &self.conus.order, self.ddr_match)?;
        let n = compressed.divide_comids.len();

        // ----- 2. SparseAdjacency: rows/cols + length/slope sliced -----
        let mut length_m: Vec<f32> = Vec::with_capacity(n);
        let mut slope: Vec<f32> = Vec::with_capacity(n);
        for c in &compressed.divide_comids {
            let pos = self.conus.index.position(c).ok_or_else(|| DataError::Malformed {
                path: self.conus.path.clone(),
                message: format!("compressed COMID {c:?} not found in CONUS order"),
            })?;
            length_m.push(self.conus.length_m[pos]);
            slope.push(self.conus.slope[pos]);
        }
        let values: Vec<f32> = vec![1.0; compressed.rows.len()];
        let adjacency = SparseAdjacency {
            n,
            rows: compressed.rows.clone(),
            cols: compressed.cols.clone(),
            values,
            length_m,
            slope,
        };

        // ----- 3. flow_scale + q_prime read & fuse -----
        let flow_scale = build_flow_scale(
            &gauge_staids,
            &compressed.gauge_compressed,
            &self.gages,
            n,
        );
        let mut q_prime = self
            .streamflow
            .read_window(window, &compressed.divide_comids)?;
        // q_prime: (T_hours, N). Multiply each column by flow_scale[col].
        let t_hours = q_prime.shape()[0];
        for col in 0..n {
            let s = flow_scale[col];
            if (s - 1.0).abs() < 1e-9 {
                continue;
            }
            for t in 0..t_hours {
                q_prime[(t, col)] *= s;
            }
        }

        // Daily q' (pre-trim) for the disaggregation head; same per-column
        // flow_scale (a constant scale is mass-consistent at both resolutions).
        let mut q_prime_daily = self.streamflow.read_window_daily(
            window.window_start,
            window.rho_days,
            &compressed.divide_comids,
        )?;
        let n_days_q = q_prime_daily.shape()[0];
        for col in 0..n {
            let s = flow_scale[col];
            if (s - 1.0).abs() < 1e-9 {
                continue;
            }
            for t in 0..n_days_q {
                q_prime_daily[(t, col)] *= s;
            }
        }

        // ----- 3c. Hourly precip + temperature for the disagg head -----
        let precip_hourly = self.read_precip_window(window, &compressed.divide_comids, n)?;
        let temp_hourly = self.read_temp_window(window, &compressed.divide_comids, n)?;

        // ----- 4. Attributes: slice + fill_nans + normalize + transpose -----
        let spatial_attributes_normalized = self.finalize_attrs(&compressed.divide_comids, n);

        // ----- 5. Observations (present-in-adjacency STAIDs; missing→error) -----
        let observations = self.observations.read_window(window, &gauge_staids)?;

        // ----- 6. Window-start state from the cache (if configured) -----
        let initial_state = match &self.state_cache {
            Some(cache) => Some(cache.row_for_day(window.window_start, &compressed.divide_comids)?),
            None => None,
        };

        // ----- 7. Impervious mask (None when absent or leakance off) -----
        let impervious_mask = self.build_impervious_mask(&compressed.divide_comids);

        // ----- 8. Assemble -----
        Ok(RoutingBatch {
            adjacency,
            spatial_attributes_normalized,
            q_prime,
            q_prime_daily,
            precip_hourly,
            temp_hourly,
            gauge_obs_std: self.gauge_obs_std(&gauge_staids)?,
            observations,
            outflow_idx: compressed.outflow_idx,
            gauge_staids,
            divide_comids: compressed.divide_comids,
            flow_scale,
            window: *window,
            initial_state,
            impervious_mask,
        })
    }

    // -----------------------------------------------------------------------
    // Precip read helpers (precip-driven disaggregation head)
    // -----------------------------------------------------------------------

    /// Read + normalize hourly precip for a training rho-window. Returns
    /// `(n_hourly, N)`, or empty `(0, N)` when precip conditioning is off.
    fn read_precip_window(
        &self,
        window: &RhoWindow,
        comids: &[Comid],
        n: usize,
    ) -> Result<Array2<f32>> {
        match (&self.precip, self.want_precip) {
            (Some(store), true) => Ok(normalize_precip(store.read_window(window, comids)?)),
            _ => Ok(Array2::<f32>::zeros((0, n))),
        }
    }

    /// Read + normalize hourly precip for a test window.
    fn read_precip_test_window(
        &self,
        window: &crate::data::TestWindow,
        comids: &[Comid],
        n: usize,
    ) -> Result<Array2<f32>> {
        match (&self.precip, self.want_precip) {
            (Some(store), true) => Ok(normalize_precip(store.read_test_window(window, comids)?)),
            _ => Ok(Array2::<f32>::zeros((0, n))),
        }
    }

    /// Read + normalize hourly temperature for a training rho-window. Returns
    /// `(n_hourly, N)`, or empty `(0, N)` when temperature conditioning is off.
    fn read_temp_window(
        &self,
        window: &RhoWindow,
        comids: &[Comid],
        n: usize,
    ) -> Result<Array2<f32>> {
        match (&self.precip, self.want_temp) {
            (Some(store), true) => Ok(normalize_temp(store.read_temp_window(window, comids)?)),
            _ => Ok(Array2::<f32>::zeros((0, n))),
        }
    }

    /// Read + normalize hourly temperature for a test window.
    fn read_temp_test_window(
        &self,
        window: &crate::data::TestWindow,
        comids: &[Comid],
        n: usize,
    ) -> Result<Array2<f32>> {
        match (&self.precip, self.want_temp) {
            (Some(store), true) => Ok(normalize_temp(store.read_temp_test_window(window, comids)?)),
            _ => Ok(Array2::<f32>::zeros((0, n))),
        }
    }

    // -----------------------------------------------------------------------
    // Shared attribute helper
    // -----------------------------------------------------------------------

    /// Slice + fill_nans + normalize + transpose the attribute matrix for a
    /// given COMID list. Returns `(N, F)` normalized attributes.
    ///
    /// Extracted from `collate`'s attribute block so that both `collate` and
    /// `build_static_network` share a single implementation.
    fn finalize_attrs(&self, divide_comids: &[Comid], n: usize) -> Array2<f32> {
        let f = self.attr_names.len();
        let mut attrs_present: Array2<f32> = Array2::zeros((f, n));
        for (out_col, comid) in divide_comids.iter().enumerate() {
            if let Some(src_col) = self.attrs.index.position(comid) {
                for fi in 0..f {
                    attrs_present[(fi, out_col)] = self.attrs.attrs[(fi, src_col)];
                }
            } else {
                // Missing — fill with NaN so fill_nans handles it via row_means.
                for fi in 0..f {
                    attrs_present[(fi, out_col)] = f32::NAN;
                }
            }
        }
        fill_nans(attrs_present.view_mut(), &self.attrs.row_means);
        // Normalize: (attrs - means) / stds, broadcast along axis 1.
        for fi in 0..f {
            let mean = self.means[fi];
            let std = self.stds[fi];
            for col in 0..n {
                attrs_present[(fi, col)] = (attrs_present[(fi, col)] - mean) / std;
            }
        }
        // Transpose to (N, F) for the MLP head's input contract.
        attrs_present.reversed_axes().into_owned()
    }

    // -----------------------------------------------------------------------
    // Impervious mask helper
    // -----------------------------------------------------------------------

    /// Build the per-reach impervious hard-zero mask for `divide_comids`.
    ///
    /// Returns `None` when `leakance_impervious_threshold` is `None` (i.e.,
    /// leakance is off or `corridor_impervious` is not among the attributes).
    ///
    /// NaN (no StreamCat coverage for a reach) → 1.0: absence of imperviousness
    /// data is NOT the same as impervious concrete; leakance is allowed.
    fn build_impervious_mask(&self, divide_comids: &[Comid]) -> Option<Vec<f32>> {
        let threshold = self.leakance_impervious_threshold?;
        let fi = self
            .attr_names
            .iter()
            .position(|n| n == "corridor_impervious")
            .expect("leakance_impervious_threshold is Some only when corridor_impervious is in attr_names");
        let raw: Vec<f32> = divide_comids
            .iter()
            .map(|comid| {
                if let Some(src_col) = self.attrs.index.position(comid) {
                    self.attrs.attrs[(fi, src_col)]
                } else {
                    f32::NAN // missing COMID → NaN → treated as not impervious
                }
            })
            .collect();
        Some(build_mask_from_raw_col(&raw, threshold))
    }

    // -----------------------------------------------------------------------
    // Test-mode collation (SP-5)
    // -----------------------------------------------------------------------

    /// Test-mode collation. Lazily builds and caches the all-gauges static
    /// network on first call; subsequent calls slice `q_prime` + `observations`
    /// for the given `TestWindow`.
    ///
    /// Mirrors DDR's `_test` per-batch RoutingDataclass construction with the
    /// simplification that the network is the full filtered-gauge union.
    ///
    /// `window` is a `TestWindow` (contiguous-hourly) rather than `RhoWindow`
    /// because chunked test time must tile without DDR's `inclusive='left'`
    /// trim. See `src/data/test_window.rs` and SP-5 design spec.
    pub fn collate_window(
        &self,
        window: &crate::data::TestWindow,
    ) -> Result<RoutingBatch> {
        self.collate_window_inner(window, false)
    }

    /// Like `collate_window`, but reads one extra daily Q' row beyond the window
    /// end so that the disaggregation head can use the NEXT chunk's first day as
    /// the "next" context for the last day of this chunk. Without lookahead the
    /// disagg right-clamps (next[d-1] = center[d-1]), which shifts sub-daily Q'
    /// and causes systematic routing errors at every 15-day chunk boundary — the
    /// diagnosed root cause of the floor validation failing the 0.25 m³/s bar.
    ///
    /// Pass `true` only for teacher/state-cache generation where the disagg must
    /// match the floor's continuous-window disagg. Pass `false` (= `collate_window`)
    /// everywhere else (eval, floor, training) where the current lookahead behaviour
    /// is appropriate.
    pub fn collate_window_with_daily_lookahead(
        &self,
        window: &crate::data::TestWindow,
    ) -> Result<RoutingBatch> {
        self.collate_window_inner(window, true)
    }

    fn collate_window_inner(
        &self,
        window: &crate::data::TestWindow,
        daily_q_lookahead: bool,
    ) -> Result<RoutingBatch> {
        let cache = self.get_or_build_static_network()?;

        // Read q_prime for this contiguous window and apply cached flow_scale.
        let mut q_prime = self
            .streamflow
            .read_test_window(window, &cache.divide_comids)?;
        let t_hours = q_prime.shape()[0];
        let n = cache.adjacency.n;
        for col in 0..n {
            let s = cache.flow_scale[col];
            if (s - 1.0).abs() < 1e-9 {
                continue;
            }
            for t in 0..t_hours {
                q_prime[(t, col)] *= s;
            }
        }

        // Daily q' for the disaggregation head.
        // With daily_q_lookahead=true, read one extra day so that the disagg
        // "next" context for the last day of this chunk is the first day of the
        // next chunk rather than the last day repeated (right-clamp). The extra
        // row only participates in the 3-tap window; it does NOT extend routing
        // (n_hourly = window.n_days * 24 is unchanged). d_use < d condition in
        // disagg_head::forward then avoids the right-clamp unconditionally.
        let n_days_q = if daily_q_lookahead {
            window.n_days + 1
        } else {
            window.n_days
        };
        let mut q_prime_daily = self.streamflow.read_window_daily(
            window.window_start,
            n_days_q,
            &cache.divide_comids,
        )?;
        let actual_n_days_q = q_prime_daily.shape()[0];
        for col in 0..n {
            let s = cache.flow_scale[col];
            if (s - 1.0).abs() < 1e-9 {
                continue;
            }
            for t in 0..actual_n_days_q {
                q_prime_daily[(t, col)] *= s;
            }
        }

        // Hourly precip + temperature for the disagg head (test window).
        let precip_hourly = self.read_precip_test_window(window, &cache.divide_comids, n)?;
        let temp_hourly = self.read_temp_test_window(window, &cache.divide_comids, n)?;

        // Slice observations from the cached full-period array along axis 0.
        let obs = cache
            .full_observations
            .slice(ndarray::s![window.daily_range(), ..])
            .to_owned();

        // Window-start state from the cache (if configured). Same logic as
        // `collate`: look up the window's start date in the cache.
        let initial_state = match &self.state_cache {
            Some(sc) => Some(sc.row_for_day(window.window_start, &cache.divide_comids)?),
            None => None,
        };

        Ok(RoutingBatch {
            adjacency: cache.adjacency.clone(),
            spatial_attributes_normalized: cache.spatial_attributes_normalized.clone(),
            q_prime,
            q_prime_daily,
            precip_hourly,
            temp_hourly,
            observations: obs,
            gauge_obs_std: Vec::new(),
            outflow_idx: cache.outflow_idx.clone(),
            gauge_staids: cache.gauge_staids.clone(),
            divide_comids: cache.divide_comids.clone(),
            flow_scale: cache.flow_scale.clone(),
            // RoutingBatch.window carries RhoWindow for diagnostics only.
            // Construct one with rho_days == n_days; the engine doesn't read it.
            window: RhoWindow {
                start_day_idx: window.start_day_idx,
                rho_days: window.n_days,
                window_start: window.window_start,
            },
            initial_state,
            impervious_mask: cache.impervious_mask.clone(),
        })
    }

    fn get_or_build_static_network(&self) -> Result<&StaticNetworkCache> {
        if let Some(c) = self.static_network.get() {
            return Ok(c);
        }
        let cache = self.build_static_network()?;
        let _ = self.static_network.set(cache);
        Ok(self.static_network.get().expect("just set"))
    }

    fn build_static_network(&self) -> Result<StaticNetworkCache> {
        let all_staids: Vec<Staid> = self.staids().to_vec();

        // 1. Subgraph union + compression over ALL filtered gauges.
        let unioned = union_subgraphs(&all_staids, &self.gages_adj);
        if unioned.gauges.is_empty() {
            return Err(DataError::Malformed {
                path: std::path::PathBuf::from("<collate_window:build_static_network>"),
                message: "no gauges present in gages_adjacency for static network".into(),
            });
        }
        let gauge_staids: Vec<Staid> =
            unioned.gauges.iter().map(|(s, _, _)| s.clone()).collect();
        let compressed = compress(&unioned, &self.conus.order, self.ddr_match)?;
        let n = compressed.divide_comids.len();

        // 2. SparseAdjacency.
        let mut length_m: Vec<f32> = Vec::with_capacity(n);
        let mut slope: Vec<f32> = Vec::with_capacity(n);
        for c in &compressed.divide_comids {
            let pos = self.conus.index.position(c).ok_or_else(|| DataError::Malformed {
                path: self.conus.path.clone(),
                message: format!("compressed COMID {c:?} not found in CONUS order"),
            })?;
            length_m.push(self.conus.length_m[pos]);
            slope.push(self.conus.slope[pos]);
        }
        let values: Vec<f32> = vec![1.0; compressed.rows.len()];
        let adjacency = SparseAdjacency {
            n,
            rows: compressed.rows.clone(),
            cols: compressed.cols.clone(),
            values,
            length_m,
            slope,
        };

        // 3. flow_scale.
        let flow_scale = build_flow_scale(
            &gauge_staids,
            &compressed.gauge_compressed,
            &self.gages,
            n,
        );

        // 4. Normalized attributes (N, F).
        let spatial_attributes_normalized = self.finalize_attrs(&compressed.divide_comids, n);

        // 5. Full-period observations: read the entire time axis at once.
        let full_rho = RhoWindow {
            start_day_idx: 0,
            rho_days: self.time_axis.num_days,
            window_start: self.time_axis.start,
        };
        let full_observations = self.observations.read_window(&full_rho, &gauge_staids)?;

        // 6. Impervious mask (None when corridor_impervious absent or leakance off).
        let impervious_mask = self.build_impervious_mask(&compressed.divide_comids);

        Ok(StaticNetworkCache {
            adjacency,
            outflow_idx: compressed.outflow_idx,
            flow_scale,
            spatial_attributes_normalized,
            full_observations,
            gauge_staids,
            divide_comids: compressed.divide_comids,
            impervious_mask,
        })
    }
}

/// Load and merge per-variable normalization stats from a list of attribute
/// store paths. Each path's adjacent statistics JSON is checked in order; the
/// first file that contains a given variable's stats wins. Hard-errors if any
/// requested variable has no stats in any of the files.
///
/// Used by the multi-path `data_sources.attributes` case in [`MeritGagesDataset::open`].
fn load_merged_stats(
    attr_paths: &[std::path::PathBuf],
    attr_names: &[String],
) -> Result<(Array1<f32>, Array1<f32>)> {
    let mut means_vec: Vec<f32> = vec![0.0; attr_names.len()];
    let mut stds_vec: Vec<f32> = vec![1.0; attr_names.len()];
    let mut found: Vec<bool> = vec![false; attr_names.len()];

    for path in attr_paths {
        let stats_path = stats_path_from_attrs(path);
        let Ok(stats) = AttrStats::open(&stats_path) else {
            continue;
        };
        for (fi, name) in attr_names.iter().enumerate() {
            if !found[fi] {
                if let Some(row) = stats.by_name.get(name) {
                    means_vec[fi] = row.mean as f32;
                    stds_vec[fi] = row.std as f32;
                    found[fi] = true;
                }
            }
        }
    }

    for (fi, name) in attr_names.iter().enumerate() {
        if !found[fi] {
            return Err(DataError::Malformed {
                path: attr_paths[0].clone(),
                message: format!(
                    "no statistics found for attribute '{name}' in any stats file \
                     adjacent to the attribute stores"
                ),
            });
        }
    }

    Ok((Array1::from(means_vec), Array1::from(stds_vec)))
}

/// Pre-head precip normalization: per-reach (column), per-WINDOW z-score of
/// `log1p(precip)` — mean-centered, unit-variance, recomputed fresh over
/// whatever window of hours is passed in (the training rho-window or the eval
/// chunk). NOT normalized by a fixed climatological constant.
///
/// Why a per-window z-score instead of a fixed-divisor basin normalization
/// (`log1p(precip / meanP_hourly)`): a fixed-divisor feature is always
/// non-negative and mostly exact-zero (dry hours), so every hidden unit's
/// weight gradient shares the same sign and weights for rarely-wet hours get
/// no gradient signal on dry samples — well-documented to slow convergence
/// (LeCun et al., *Efficient BackProp*). A mean-centered feature trains every
/// weight every step. Verified via the synthetic discriminator
/// `examples/disagg_precip_normalization_discriminator.rs`: at a matched
/// short training budget the z-scored feature reached 88% precip-timing
/// accuracy vs 77% for the fixed-divisor feature (both reach 100% given 20x
/// more steps — a convergence-speed gap, not missing information). This is
/// the design that produced the documented +0.037 NSE disaggregation win.
///
/// Near-constant windows (std < 1e-6, e.g. all-dry) map to all zeros
/// (neutral), so the head falls back to the daily-Q shape.
///
/// `pub(crate)` (not private) so `src/pretrain/` can call the EXACT same
/// production transform on real hourly ground truth, rather than
/// reimplementing it and risking drift between pretraining and production
/// features.
pub(crate) fn normalize_precip(mut precip: Array2<f32>) -> Array2<f32> {
    let (t, n) = precip.dim();
    if t == 0 {
        return precip;
    }
    for col in 0..n {
        // log1p in place.
        for row in 0..t {
            precip[(row, col)] = precip[(row, col)].ln_1p();
        }
        let mean: f32 = (0..t).map(|r| precip[(r, col)]).sum::<f32>() / t as f32;
        let var: f32 =
            (0..t).map(|r| (precip[(r, col)] - mean).powi(2)).sum::<f32>() / t as f32;
        let std = var.sqrt();
        if std < 1e-6 {
            for row in 0..t {
                precip[(row, col)] = 0.0;
            }
        } else {
            for row in 0..t {
                precip[(row, col)] = (precip[(row, col)] - mean) / std;
            }
        }
    }
    precip
}

/// Pre-head temperature normalization: per-reach (column) z-score over the
/// window's hours — like [`normalize_precip`] but WITHOUT `log1p` (temperature
/// in K is not a non-negative flux). No-coverage reaches read as a constant 0
/// (std≈0) and map to 0 (neutral), so they fall back to the daily-Q shape.
fn normalize_temp(mut temp: Array2<f32>) -> Array2<f32> {
    let (t, n) = temp.dim();
    if t == 0 {
        return temp;
    }
    for col in 0..n {
        let mean: f32 = (0..t).map(|r| temp[(r, col)]).sum::<f32>() / t as f32;
        let var: f32 = (0..t).map(|r| (temp[(r, col)] - mean).powi(2)).sum::<f32>() / t as f32;
        let std = var.sqrt();
        if std < 1e-6 {
            for row in 0..t {
                temp[(row, col)] = 0.0;
            }
        } else {
            for row in 0..t {
                temp[(row, col)] = (temp[(row, col)] - mean) / std;
            }
        }
    }
    temp
}

/// Default statistics JSON path: `<attrs_dir>/statistics/merit_attribute_statistics_<attrs_filename>.json`.
fn stats_path_from_attrs(attrs_path: &std::path::Path) -> std::path::PathBuf {
    let dir = attrs_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let fname = attrs_path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();
    dir.join("statistics")
        .join(format!("merit_attribute_statistics_{fname}.json"))
}

/// Parse two `"YYYY/MM/DD"` strings (DDR convention) into a `TimeAxis`.
fn parse_experiment_axis(start: &str, end: &str) -> Result<TimeAxis> {
    let start_date =
        chrono::NaiveDate::parse_from_str(start, "%Y/%m/%d").map_err(|e| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: format!("invalid experiment.start_time {start:?}: {e}"),
        })?;
    let end_date =
        chrono::NaiveDate::parse_from_str(end, "%Y/%m/%d").map_err(|e| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: format!("invalid experiment.end_time {end:?}: {e}"),
        })?;
    Ok(TimeAxis::new(start_date, end_date))
}

/// Convert a raw `corridor_impervious` column (fraction [0, 1]) to a 0/1 mask.
///
/// `mask[i] = if raw[i] > threshold { 0.0 } else { 1.0 }`
///
/// NaN → 1.0: no StreamCat coverage means absence of imperviousness data, which is
/// NOT the same as impervious concrete. Leakance is allowed where data is missing.
pub(crate) fn build_mask_from_raw_col(raw: &[f32], threshold: f32) -> Vec<f32> {
    raw.iter()
        .map(|&v| if v.is_nan() || v <= threshold { 1.0_f32 } else { 0.0_f32 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    // -----------------------------------------------------------------------
    // build_mask_from_raw_col unit tests (no dataset required)
    // -----------------------------------------------------------------------

    #[test]
    fn mask_from_raw_col_basic_threshold() {
        // Values below/at threshold → 1.0 (leakance allowed).
        // Values above threshold → 0.0 (impervious hard-zero).
        let raw = [0.0_f32, 0.5, 0.7, 0.8, 1.0];
        let mask = build_mask_from_raw_col(&raw, 0.7);
        assert_eq!(mask, vec![1.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn mask_from_raw_col_nan_treated_as_not_impervious() {
        // NaN (no StreamCat coverage) → 1.0: absence ≠ concrete.
        let raw = [f32::NAN, 0.9, f32::NAN, 0.3];
        let mask = build_mask_from_raw_col(&raw, 0.7);
        assert_eq!(mask, vec![1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn mask_from_raw_col_all_below_threshold_is_all_ones() {
        let raw = [0.1_f32, 0.2, 0.3];
        let mask = build_mask_from_raw_col(&raw, 0.7);
        assert!(mask.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn mask_from_raw_col_all_above_threshold_is_all_zeros() {
        let raw = [0.8_f32, 0.9, 1.0];
        let mask = build_mask_from_raw_col(&raw, 0.7);
        assert!(mask.iter().all(|&v| v == 0.0));
    }

    // -----------------------------------------------------------------------
    // normalize_precip: per-reach, per-window z-score of log1p(precip)
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_precip_storm_hour_is_positive_outlier() {
        // Within-window RELATIVE contrast: one storm hour amid mostly-dry
        // hours must z-score to a clear positive outlier, while the dry
        // hours land below the window mean (negative). This is the timing
        // signal the disagg head trains on.
        let t = 24;
        let raw = Array2::<f32>::from_shape_fn((t, 1), |(row, _)| {
            if row == 7 { 8.6 } else { 0.0 }
        });
        let out = normalize_precip(raw);
        assert!(
            out[(7, 0)] > 2.0,
            "storm hour z-score {} should be a large positive outlier",
            out[(7, 0)]
        );
        for row in (0..t).filter(|&r| r != 7) {
            assert!(
                out[(row, 0)] < 0.0,
                "dry hour {row} z-scored to {} — expected below the window mean",
                out[(row, 0)]
            );
        }
    }

    #[test]
    fn normalize_precip_near_constant_window_maps_to_zeros() {
        // A (near-)zero-variance window (e.g. all-dry, or a data gap read
        // as a constant) must map to all zeros — neutral input, not NaN/Inf
        // from dividing by std ≈ 0.
        let raw = Array2::<f32>::from_elem((4, 2), 0.3);
        let out = normalize_precip(raw);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "near-constant window produced non-finite output: {out:?}"
        );
        assert!(
            out.iter().all(|&v| v == 0.0),
            "near-constant window should map to all zeros: {out:?}"
        );
    }

    #[test]
    fn normalize_precip_empty_window_is_noop() {
        let raw = Array2::<f32>::zeros((0, 3));
        let out = normalize_precip(raw);
        assert_eq!(out.dim(), (0, 3));
    }

    // -----------------------------------------------------------------------
    // Back-compat: when corridor_impervious is absent, mask is None
    // -----------------------------------------------------------------------

    #[test]
    fn leakance_threshold_none_when_attr_absent() {
        // A config with use_leakance=true but corridor_impervious NOT in attr_names
        // must leave leakance_impervious_threshold = None (back-compat no-op).
        //
        // We can't open a full MeritGagesDataset without live data, so test
        // the threshold-selection logic directly.
        let attr_names = ["n", "q_spatial", "p_spatial"];
        let has_corridor = attr_names.iter().any(|&n| n == "corridor_impervious");
        let threshold: Option<f32> = if has_corridor { Some(0.7) } else { None };
        assert!(threshold.is_none(), "back-compat guard: threshold must be None when attr absent");
    }

    #[test]
    fn open_dataset_against_live_yaml() {
        let cfg_path = "config/merit_training.yaml";
        if !std::path::Path::new(cfg_path).exists() {
            eprintln!("skipping: {cfg_path} not present");
            return;
        }
        let cfg = match Config::from_yaml_file(cfg_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: config load failed: {e}");
                return;
            }
        };
        // Skip if any required data path is absent.
        if let Some(ds) = cfg.data_sources.as_ref() {
            for p in &ds.attributes {
                if !p.exists() {
                    eprintln!("skipping: {} not present", p.display());
                    return;
                }
            }
            for p in [&ds.streamflow, &ds.observations, &ds.gages] {
                if !p.exists() {
                    eprintln!("skipping: {} not present", p.display());
                    return;
                }
            }
            // Adjacency zarr required to open MeritGagesDataset; skip if not
            // configured (fabric-only configs resolve managed paths via
            // `ddrs plan`, which this test doesn't run).
            for opt in &[&ds.conus_adjacency, &ds.gages_adjacency] {
                match opt {
                    None => {
                        eprintln!("skipping: adjacency zarr path not configured (test does not run the managed build)");
                        return;
                    }
                    Some(p) if !p.exists() => {
                        eprintln!("skipping: {} not present", p.display());
                        return;
                    }
                    _ => {}
                }
            }
        } else {
            return;
        }
        let ds = MeritGagesDataset::open(&cfg).expect("open dataset");
        assert!(ds.len() > 100, "expected many filtered gauges, got {}", ds.len());
        assert_eq!(ds.attr_names.len(), 10);
        assert_eq!(ds.means.len(), 10);
        assert_eq!(ds.stds.len(), 10);
    }

    #[test]
    fn collate_one_batch_against_live_yaml() {
        let cfg_path = "config/merit_training.yaml";
        if !std::path::Path::new(cfg_path).exists() {
            eprintln!("skipping: {cfg_path} not present");
            return;
        }
        let cfg = match Config::from_yaml_file(cfg_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        if let Some(ds) = cfg.data_sources.as_ref() {
            for p in &ds.attributes {
                if !p.exists() {
                    eprintln!("skipping: {} not present", p.display());
                    return;
                }
            }
            for p in [&ds.streamflow, &ds.observations, &ds.gages] {
                if !p.exists() {
                    eprintln!("skipping: {} not present", p.display());
                    return;
                }
            }
            for opt in &[&ds.conus_adjacency, &ds.gages_adjacency] {
                match opt {
                    None => {
                        eprintln!("skipping: adjacency zarr path not configured (managed build not yet available)");
                        return;
                    }
                    Some(p) if !p.exists() => {
                        eprintln!("skipping: {} not present", p.display());
                        return;
                    }
                    _ => {}
                }
            }
        } else {
            return;
        }
        let dataset = MeritGagesDataset::open(&cfg).expect("open dataset");

        // Pick the first 4 gauges and a 90-day window.
        let staids: Vec<_> = dataset.staids().iter().take(4).cloned().collect();
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let window = dataset.time_axis().sample_rho_window(&mut rng, 90);

        let batch = dataset.collate(&staids, &window).expect("collate");

        assert_eq!(batch.gauge_staids.len(), staids.len());
        assert_eq!(batch.adjacency.n, batch.divide_comids.len());
        assert_eq!(batch.spatial_attributes_normalized.shape()[0], batch.adjacency.n);
        assert_eq!(batch.spatial_attributes_normalized.shape()[1], dataset.attr_names.len());
        assert_eq!(batch.q_prime.shape(), &[window.n_hourly(), batch.adjacency.n]);
        assert_eq!(batch.observations.shape(), &[window.rho_days, batch.gauge_staids.len()]);
        // Lower-triangular invariant.
        for k in 0..batch.adjacency.nnz() {
            assert!(
                batch.adjacency.rows[k] >= batch.adjacency.cols[k],
                "lower-triangular violated at k={k}"
            );
        }
        assert_eq!(batch.outflow_idx.len(), batch.gauge_staids.len());
        assert_eq!(batch.flow_scale.len(), batch.adjacency.n);
    }

    #[test]
    fn collate_window_static_network_reuses_across_calls() {
        let cfg_path = "config/merit_training.yaml";
        if !std::path::Path::new(cfg_path).exists() {
            eprintln!("skipping: {cfg_path} not present");
            return;
        }
        let cfg = match Config::from_yaml_file(cfg_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let Some(ds_cfg) = cfg.data_sources.as_ref() else { return };
        for p in &ds_cfg.attributes {
            if !p.exists() {
                eprintln!("skipping: {} not present", p.display());
                return;
            }
        }
        for p in [&ds_cfg.streamflow, &ds_cfg.observations, &ds_cfg.gages] {
            if !p.exists() {
                eprintln!("skipping: {} not present", p.display());
                return;
            }
        }
        for opt in &[&ds_cfg.conus_adjacency, &ds_cfg.gages_adjacency] {
            match opt {
                None => {
                    eprintln!("skipping: adjacency zarr path not configured (managed build not yet available)");
                    return;
                }
                Some(p) if !p.exists() => {
                    eprintln!("skipping: {} not present", p.display());
                    return;
                }
                _ => {}
            }
        }

        let ds = MeritGagesDataset::open(&cfg).expect("open");
        let axis = ds.time_axis().clone();

        let w1 = crate::data::TestWindow::new(&axis, 0, 15);
        let w2 = crate::data::TestWindow::new(&axis, 15, 15);

        let b1 = ds.collate_window(&w1).expect("w1");
        let b2 = ds.collate_window(&w2).expect("w2");

        assert_eq!(b1.adjacency.n, b2.adjacency.n,
            "static network changed between calls");
        assert_eq!(b1.gauge_staids, b2.gauge_staids);
        assert_eq!(b1.outflow_idx, b2.outflow_idx);

        assert_eq!(b1.q_prime.nrows(), 15 * 24, "TestWindow contiguous hourly");
        assert_eq!(b2.q_prime.nrows(), 15 * 24);
        assert_eq!(b1.observations.nrows(), 15, "observations sliced per window");
        assert_eq!(b2.observations.nrows(), 15);
    }

    #[test]
    fn disagg_rejected_on_hourly_native_source() {
        use crate::data::dates::Frequency;
        let p = std::path::Path::new("/mnt/fake/qr_hourly.ic");
        let err = validate_disagg_vs_resolution(Frequency::Hourly, true, p).unwrap_err();
        assert!(err.to_string().contains("hourly-native"), "got: {err}");
        assert!(validate_disagg_vs_resolution(Frequency::Hourly, false, p).is_ok());
        assert!(validate_disagg_vs_resolution(Frequency::Daily, true, p).is_ok());
        assert!(validate_disagg_vs_resolution(Frequency::Daily, false, p).is_ok());
    }
}
