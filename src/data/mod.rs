//! Data layer: read DDR's live training sources in place (no export/conversion).
//!
//! ## Design notes
//!
//! The five data sources have different I/O models:
//!
//!   - `merit_conus_adjacency.zarr`           (plain zarr v3, sync)
//!   - `merit_gages_conus_adjacency.zarr`     (plain zarr v3, sync)
//!   - `merit_global_attributes_v2.nc`        (netCDF4/HDF5, sync)
//!   - `merit_dhbv2_UH_retrospective.ic`      (icechunk, async-first)
//!   - `usgs_daily_observations`              (icechunk, async-first)
//!
//! No `trait Store` unifying them — the call sites diverge too much. Each
//! store has a small focused module under `store/` returning `ndarray::Array`
//! buffers indexed by `Comid`/`Staid` newtypes.
//!
//! The dataset (`dataset.rs`) owns a single `tokio::runtime::Runtime` and
//! calls `block_on(...)` at the icechunk boundary — the rest of ddrs stays
//! sync. See `mlp_router`-style training loop in `harness/` for how the
//! pieces compose.

pub mod collate;
pub use collate::{compress, union_subgraphs, CompressedAdj, UnionedCoo};
pub mod dataset;
pub mod dates;
pub mod error;
pub mod ids;
pub mod sampler;
pub mod store;
pub mod statistics;
pub mod test_window;

pub use dataset::{MeritGagesDataset, RoutingBatch};
pub use dates::{Frequency, RhoWindow, TimeAxis};
pub use error::{DataError, Result};
pub use ids::{Comid, IdIndex, Staid};
pub use sampler::{RandomSampler, SequentialSampler};
pub use store::{
    AttributesStore, ConusAdjacencyStore, GageMetadata, GageRow, GageSubgraph,
    GagesAdjacencyStore, GlobalObservationsStore, ObservationsStore, StreamflowStore,
    UsgsObservationsStore,
};
pub use statistics::{fill_nans, fill_nans_1d, naninfmean, AttrStatRow, AttrStats};
pub use test_window::TestWindow;
