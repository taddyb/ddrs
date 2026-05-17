//! Per-source store modules. Each is a small focused reader over one of the
//! DDR data sources, returning `ndarray` buffers + domain-typed metadata.
//! Backend types (`zarrs::Array`, `netcdf::Variable`, `icechunk::Session`)
//! never escape the modules — callers see only `ndarray` and `data::ids`
//! types.
//!
//! Per the design notes in `src/data/mod.rs`: no `trait Store`, no
//! `Box<dyn Store>` — premature unification across three different I/O
//! models. Composition over abstraction at this layer.

pub mod gage_csv;
pub mod icechunk;
pub mod netcdf;
pub mod zarr;

pub use gage_csv::{GageMetadata, GageRow};
pub use netcdf::AttributesStore;
pub use zarr::{ConusAdjacencyStore, GageSubgraph, GagesAdjacencyStore};
