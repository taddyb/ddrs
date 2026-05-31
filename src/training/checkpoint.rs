//! Checkpoint save/load via BURN's CompactRecorder.
//!
//! Mirrors `~/projects/ddr/src/ddr/validation/utils.py::save_state` and
//! `~/projects/ddr/src/ddr/scripts_utils.py::load_checkpoint`. Cross-runtime
//! checkpoint compatibility with DDR's .pt files is NOT supported (different
//! recorder formats).
//!
//! # File extension
//!
//! `CompactRecorder` (= `NamedMpkFileRecorder<HalfPrecisionSettings>`) appends
//! `.mpk` to the caller-supplied base path. Pass `/path/to/checkpoint` and the
//! file written on disk will be `/path/to/checkpoint.mpk`.

use std::path::Path;

use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::Backend;

use crate::data::error::{DataError, Result};
use crate::nn::kan_head::KanHead;

/// Save KAN head weights to `path` (`.mpk` extension appended by the recorder).
pub fn save_kan_head<B: Backend>(path: &Path, head: &KanHead<B>) -> Result<()> {
    CompactRecorder::new()
        .record(head.clone().into_record(), path.to_path_buf())
        .map_err(|e| DataError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")),
        })
}

/// Load KAN head weights from `path` (`.mpk` extension appended by the recorder).
///
/// `head_template` must be constructed with the same architecture as the
/// saved checkpoint; its parameter values are discarded.
pub fn load_kan_head<B: Backend>(
    path: &Path,
    head_template: KanHead<B>,
    device: &B::Device,
) -> Result<KanHead<B>> {
    let record = CompactRecorder::new()
        .load(path.to_path_buf(), device)
        .map_err(|e| DataError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")),
        })?;
    Ok(head_template.load_record(record))
}
