//! Unified data-layer error type.
//!
//! Every store variant carries source-path context so failures point to the
//! actual file/group that broke. DDR's stack traces ("KeyError: 'gage_id'")
//! are notoriously hard to debug — we're paying the extra fields once here so
//! callers don't have to wrap every read with their own context.
use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum DataError {
    #[error("zarr read failed at {path}: {source}")]
    Zarr {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("netcdf read failed at {path}: {source}")]
    NetCdf {
        path: PathBuf,
        #[source]
        source: netcdf::Error,
    },

    #[error("icechunk read failed at {path}: {source}")]
    IceChunk {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("missing {missing}/{total} {kind} in store at {path}")]
    MissingIds {
        path: PathBuf,
        kind: &'static str,
        missing: usize,
        total: usize,
    },

    #[error("malformed store at {path}: {message}")]
    Malformed { path: PathBuf, message: String },

    #[error("yaml parse error at {path}: {source}")]
    Yaml {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },

    #[error("csv parse error at {path}: {source}")]
    Csv {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },
}

pub type Result<T> = std::result::Result<T, DataError>;
