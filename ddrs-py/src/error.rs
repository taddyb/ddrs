//! Bridge error type. Converts ddrs + serde + io errors into PyErr.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("config load failed at {path:?}: {source}")]
    Config {
        path: std::path::PathBuf,
        #[source]
        source: ddrs::data::error::DataError,
    },
    #[error("config is missing the `kan_head:` section ({path:?})")]
    MissingKanHeadSection { path: String },
    #[error("attrs shape ({rows}, {cols}) mismatches kan_head.input_var_names.len() = {expected_cols}")]
    AttrShapeMismatch { rows: usize, cols: usize, expected_cols: usize },
    #[error("checkpoint load failed at {path:?}: {source}")]
    Checkpoint {
        path: std::path::PathBuf,
        #[source]
        source: ddrs::data::error::DataError,
    },
    #[error("netcdf attribute read failed: {0}")]
    Netcdf(#[source] ddrs::data::error::DataError),
    #[error("zarr adjacency read failed: {0}")]
    Zarr(#[source] ddrs::data::error::DataError),
}

impl From<BridgeError> for PyErr {
    fn from(e: BridgeError) -> Self {
        match e {
            BridgeError::Config { .. }
            | BridgeError::Checkpoint { .. }
            | BridgeError::Netcdf(_)
            | BridgeError::Zarr(_) => PyIOError::new_err(e.to_string()),
            BridgeError::MissingKanHeadSection { .. }
            | BridgeError::AttrShapeMismatch { .. } => {
                PyValueError::new_err(e.to_string())
            }
        }
    }
}
