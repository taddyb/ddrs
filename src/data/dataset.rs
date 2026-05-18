//! `MeritGagesDataset` + `RoutingBatch`.
//!
//! Mirrors `~/projects/ddr/src/ddr/geodatazoo/merit.py::Merit` for the
//! training-mode (`_init_training` + `_collate_gages`) path. Other modes
//! (target_catchments, all_catchments) are out of scope for SP-3.

use ndarray::Array2;

use crate::data::dates::RhoWindow;
use crate::data::ids::{Comid, Staid};
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
