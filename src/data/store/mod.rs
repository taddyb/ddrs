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
pub mod zarr_obs;
pub mod zarr_qprime;

pub use gage_csv::{GageMetadata, GageRow};
pub use icechunk::{StreamflowStore, UsgsObservationsStore};
pub use netcdf::AttributesStore;
pub use zarr::{ConusAdjacencyStore, GageSubgraph, GagesAdjacencyStore};
pub use zarr_obs::GlobalObservationsStore;
pub use zarr_qprime::GlobalStreamflowStore;

use ndarray::Array2;

use crate::data::dates::RhoWindow;
use crate::data::error::Result;
use crate::data::ids::{Comid, Staid};

/// Format-dispatching observations reader. The `observations` data source is
/// either an icechunk repo (`usgs_daily_observations`, CONUS) or a plain
/// zarr v2 directory group (`dMC_global_v3.1`, global); both expose the same
/// `(n_days, G)` f32 daily read contract. An enum, not a trait — per the
/// no-`Box<dyn Store>` rule, this is closed-set static dispatch.
pub enum ObservationsStore {
    /// Icechunk-backed `usgs_daily_observations` repo.
    Usgs(UsgsObservationsStore),
    /// `dMC_global_v3.1`-style zarr v2 group, one array per gage.
    Global(GlobalObservationsStore),
}

impl ObservationsStore {
    /// Open `path`, sniffing the format: a `.zgroup` at the root means a
    /// plain zarr v2 group; anything else is treated as an icechunk repo.
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let path = path.into();
        if GlobalObservationsStore::sniff(&path) {
            Ok(Self::Global(GlobalObservationsStore::open(path)?))
        } else {
            Ok(Self::Usgs(UsgsObservationsStore::open(path)?))
        }
    }

    pub fn read_window_daily(
        &self,
        window_start: chrono::NaiveDate,
        n_days: usize,
        staids: &[Staid],
    ) -> Result<Array2<f32>> {
        match self {
            Self::Usgs(s) => s.read_window_daily(window_start, n_days, staids),
            Self::Global(s) => s.read_window_daily(window_start, n_days, staids),
        }
    }

    pub fn read_window(&self, window: &RhoWindow, staids: &[Staid]) -> Result<Array2<f32>> {
        match self {
            Self::Usgs(s) => s.read_window(window, staids),
            Self::Global(s) => s.read_window(window, staids),
        }
    }

    /// Whether the store has a series for `staid`. Used to pre-filter gauge
    /// sets (gage CSVs can list gauges the observation product lacks — 63 of
    /// 5,975 in the v3.1 global set) so reads never hit `MissingIds`.
    pub fn contains(&self, staid: &Staid) -> bool {
        match self {
            Self::Usgs(s) => s.index.contains(staid),
            Self::Global(s) => s.index.contains(staid),
        }
    }
}

/// Format-dispatching Q' (unit-catchment streamflow) reader. The
/// `streamflow` data source is either an icechunk repo
/// (`merit_dhbv2_UH_retrospective`, CONUS) or a `merit_global_v2.x`-style
/// multi-zone zarr v2 store (global). Both are daily m³/s per COMID with
/// the same read contract (missing COMIDs → 0.001 fill).
pub enum StreamflowSource {
    /// Icechunk-backed CONUS retrospective.
    Icechunk(StreamflowStore),
    /// Multi-zone zarr v2 global predictions (one group per pfaf-2 zone).
    GlobalZarr(GlobalStreamflowStore),
}

impl StreamflowSource {
    /// Open `path`, sniffing the format: zone groups (`.zgroup` +
    /// `streamflow/.zarray`, at the root or one level down) mean the
    /// global zarr v2 layout; anything else is treated as an icechunk repo.
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let path = path.into();
        if GlobalStreamflowStore::sniff(&path) {
            Ok(Self::GlobalZarr(GlobalStreamflowStore::open(path)?))
        } else {
            Ok(Self::Icechunk(StreamflowStore::open(path)?))
        }
    }

    pub fn read_window_daily(
        &self,
        window_start: chrono::NaiveDate,
        n_days: usize,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        match self {
            Self::Icechunk(s) => s.read_window_daily(window_start, n_days, comids),
            Self::GlobalZarr(s) => s.read_window_daily(window_start, n_days, comids),
        }
    }

    pub fn read_window(&self, window: &RhoWindow, comids: &[Comid]) -> Result<Array2<f32>> {
        match self {
            Self::Icechunk(s) => s.read_window(window, comids),
            Self::GlobalZarr(s) => s.read_window(window, comids),
        }
    }

    pub fn read_test_window(
        &self,
        window: &crate::data::TestWindow,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        match self {
            Self::Icechunk(s) => s.read_test_window(window, comids),
            Self::GlobalZarr(s) => s.read_test_window(window, comids),
        }
    }
}
