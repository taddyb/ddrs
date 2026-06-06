//! Managed adjacency builds — construct MERIT CONUS and gauge subgraph
//! adjacency stores from raw geospatial fabric files rather than requiring
//! pre-built zarr exports.
//!
//! ## Pipeline overview
//!
//! ```text
//!   .shp/.dbf  ──► dbf::read_flowpath_records()
//!                       │
//!                       ▼
//!               Vec<FlowpathRecord>
//!                       │
//!          (Task 3)     ▼
//!              build::build_conus_adjacency()
//!                       │
//!          (Task 4)     ▼
//!              gauges::build_gauge_subgraphs()
//!                       │
//!          (Task 5)     ▼
//!              zarr_write::write_adjacency_zarr()
//! ```

/// Bump on any algorithm change that would invalidate previously-cached
/// adjacency zarr outputs.
pub const BUILDER_VERSION: u32 = 1;

pub mod build;
pub mod dbf;
pub mod gauges;
