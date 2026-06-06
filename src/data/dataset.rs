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
use crate::data::dates::{RhoWindow, TimeAxis};
use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, Staid};
use crate::data::statistics::{fill_nans, AttrStats};
use crate::data::store::{
    AttributesStore, ConusAdjacencyStore, GageMetadata, GagesAdjacencyStore, StreamflowStore,
    UsgsObservationsStore,
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
    /// `flow_scale` per column.
    pub q_prime: Array2<f32>,
    /// USGS observations, shape `(T_days, G)`. NaN-tolerant.
    pub observations: Array2<f32>,
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
    /// Observations stay on CPU.
    pub observations: Array2<f32>,
    /// Flat concat of `outflow_idx`, shape `(sum_g len(outflow_idx[g]),)`.
    pub flat_indices: Tensor<B, 1, Int>,
    /// Per-flat-index gauge group id, same shape as `flat_indices`.
    pub group_ids: Tensor<B, 1, Int>,
    pub num_gauges: usize,
    pub gauge_staids: Vec<Staid>,
    pub window: RhoWindow,
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

        // 4. Lift flat_indices + group_ids as Int tensors.
        let flat_indices = Tensor::<B, 1, Int>::from_data(TensorData::from(flat.as_slice()), device);
        let group_ids = Tensor::<B, 1, Int>::from_data(TensorData::from(group.as_slice()), device);

        let num_gauges = self.gauge_staids.len();

        RoutingTensors {
            adjacency: self.adjacency,
            spatial_attributes,
            q_prime,
            observations: self.observations,
            flat_indices,
            group_ids,
            num_gauges,
            gauge_staids: self.gauge_staids,
            window: self.window,
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
    pub(crate) streamflow: Arc<StreamflowStore>,
    pub(crate) observations: Arc<UsgsObservationsStore>,
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

        // Filter 1: DA_VALID drop.
        let pre_filter = gage_meta.rows.len();
        let da_valid: Vec<Staid> = gage_meta
            .rows
            .iter()
            .filter(|r| r.da_valid == Some(true))
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
        let attrs = Arc::new(AttributesStore::open(
            &ds.attributes,
            &attr_names,
            &conus.order,
        )?);

        let stats_path = stats_path_from_attrs(&ds.attributes);
        let stats = Arc::new(AttrStats::open(&stats_path)?);
        let means = stats.means_f32(&attr_names);
        let stds = stats.stds_f32(&attr_names);

        // ---------- 3. Icechunk stores ----------
        let streamflow = Arc::new(StreamflowStore::open(&ds.streamflow)?);
        let observations = Arc::new(UsgsObservationsStore::open(&ds.observations)?);

        // ---------- 4. Time axis from experiment dates ----------
        let time_axis = parse_experiment_axis(&exp.start_time, &exp.end_time)?;

        Ok(Self {
            conus,
            gages_adj,
            attrs,
            stats,
            gages: Arc::new(gage_meta),
            streamflow,
            observations,
            time_axis,
            attr_names,
            means,
            stds,
            gauges,
            static_network: OnceCell::new(),
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

        let compressed = compress(&unioned, &self.conus.order)?;
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

        // ----- 4. Attributes: slice + fill_nans + normalize + transpose -----
        let spatial_attributes_normalized = self.finalize_attrs(&compressed.divide_comids, n);

        // ----- 5. Observations (present-in-adjacency STAIDs; missing→error) -----
        let observations = self.observations.read_window(window, &gauge_staids)?;

        // ----- 6. Assemble -----
        Ok(RoutingBatch {
            adjacency,
            spatial_attributes_normalized,
            q_prime,
            observations,
            outflow_idx: compressed.outflow_idx,
            gauge_staids,
            divide_comids: compressed.divide_comids,
            flow_scale,
            window: *window,
        })
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

        // Slice observations from the cached full-period array along axis 0.
        let obs = cache
            .full_observations
            .slice(ndarray::s![window.daily_range(), ..])
            .to_owned();

        Ok(RoutingBatch {
            adjacency: cache.adjacency.clone(),
            spatial_attributes_normalized: cache.spatial_attributes_normalized.clone(),
            q_prime,
            observations: obs,
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
        let compressed = compress(&unioned, &self.conus.order)?;
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

        Ok(StaticNetworkCache {
            adjacency,
            outflow_idx: compressed.outflow_idx,
            flow_scale,
            spatial_attributes_normalized,
            full_observations,
            gauge_staids,
            divide_comids: compressed.divide_comids,
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

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
            for p in &[&ds.attributes, &ds.streamflow, &ds.observations, &ds.gages] {
                if !p.exists() {
                    eprintln!("skipping: {} not present", p.display());
                    return;
                }
            }
            // Adjacency zarr required to open MeritGagesDataset; skip if not
            // configured (fabric-only config awaits Task 7 managed build).
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
            for p in &[&ds.attributes, &ds.streamflow, &ds.observations, &ds.gages] {
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
        for p in &[&ds_cfg.attributes, &ds_cfg.streamflow, &ds_cfg.observations, &ds_cfg.gages] {
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
}
