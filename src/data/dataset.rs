//! `MeritGagesDataset` + `RoutingBatch`.
//!
//! Mirrors `~/projects/ddr/src/ddr/geodatazoo/merit.py::Merit` for the
//! training-mode (`_init_training` + `_collate_gages`) path. Other modes
//! (target_catchments, all_catchments) are out of scope for SP-3.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use crate::config::Config;
use crate::data::dates::{RhoWindow, TimeAxis};
use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, Staid};
use crate::data::statistics::AttrStats;
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
    /// MLP head input contract (`src/nn/mlp.rs::Mlp::forward`).
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
// MeritGagesDataset
// ---------------------------------------------------------------------------

pub struct MeritGagesDataset {
    pub(crate) conus: Arc<ConusAdjacencyStore>,
    pub(crate) gages_adj: Arc<GagesAdjacencyStore>,
    pub(crate) attrs: Arc<AttributesStore>,
    #[allow(dead_code)] // used by Task 8 (collate)
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
        let mlp = cfg.mlp.as_ref().ok_or_else(|| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: "mlp section missing".into(),
        })?;

        // ---------- 1. Adjacency + gage CSV ----------
        let conus = Arc::new(ConusAdjacencyStore::open(&ds.conus_adjacency)?);
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
        let gages_adj = Arc::new(GagesAdjacencyStore::open(&ds.gages_adjacency, &da_valid)?);

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
        let attr_names: Vec<String> = mlp.input_var_names.clone();
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
        // Skip if any data path is absent.
        if let Some(ds) = cfg.data_sources.as_ref() {
            for p in &[
                &ds.attributes,
                &ds.conus_adjacency,
                &ds.gages_adjacency,
                &ds.streamflow,
                &ds.observations,
                &ds.gages,
            ] {
                if !p.exists() {
                    eprintln!("skipping: {} not present", p.display());
                    return;
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
}
