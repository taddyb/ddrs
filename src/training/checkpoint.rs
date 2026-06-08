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
//!
//! # Checkpoint directory layout (full resume)
//!
//! A checkpoint is a DIRECTORY `epoch_E_mb_M/` holding three fixed-name files
//! so `experiment.checkpoint` resumes are exact, not weights-only:
//!
//! ```text
//! epoch_E_mb_M/
//! ├── head.mpk      KAN weights      (CompactRecorder)
//! ├── optim.mpk     Adam moments     (CompactRecorder)
//! └── state.json    [`TrainCkptState`]: epoch, next mini-batch, serialized
//!                   rng, sampler permutation + cursor
//! ```
//!
//! `experiment.checkpoint:` points at the directory; the inner filenames are
//! hardcoded. [`head_base`]/[`optim_base`] return the recorder *bases*
//! (`dir/head`, `dir/optim`) — `CompactRecorder` appends `.mpk` itself.
//! [`state_path`] returns the full `dir/state.json` path. Each file gets its
//! own name in its own directory, so nothing can clobber the head and the
//! whole checkpoint copies/deletes as one unit.

use std::path::{Path, PathBuf};

use burn::module::{AutodiffModule, Module};
use burn::optim::Optimizer;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

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

// ---------------------------------------------------------------------------
// Sidecars: optimizer record + train-loop state
// ---------------------------------------------------------------------------

/// Recorder base for the KAN head inside a checkpoint dir: `dir/head`
/// (`CompactRecorder` appends `.mpk` → `dir/head.mpk`).
pub fn head_base(ckpt_dir: &Path) -> PathBuf {
    ckpt_dir.join("head")
}

/// Recorder base for the Adam moments inside a checkpoint dir: `dir/optim`
/// (`CompactRecorder` appends `.mpk` → `dir/optim.mpk`).
pub fn optim_base(ckpt_dir: &Path) -> PathBuf {
    ckpt_dir.join("optim")
}

/// Train-loop state path inside a checkpoint dir: `dir/state.json`.
pub fn state_path(ckpt_dir: &Path) -> PathBuf {
    ckpt_dir.join("state.json")
}

/// Train-loop position saved alongside each head checkpoint so a resumed run
/// continues with the SAME gauge batches and rho-windows the original run
/// would have drawn.
///
/// `rng` is the training rng as of just after this mini-batch's window draw —
/// restoring it makes every subsequent `sample_rho_window` and per-epoch
/// `reshuffle` identical to an uninterrupted run. The sampler permutation +
/// cursor reproduce the remainder of the in-flight epoch (the shuffle that
/// produced them consumed the rng at epoch start, so it can't be re-derived
/// from `rng` alone).
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainCkptState {
    /// Epoch this checkpoint was saved in (1-based, matches the filename).
    pub epoch: usize,
    /// Mini-batch the resumed run should execute next.
    pub next_mini_batch: usize,
    /// Serialized training rng (ChaCha12 — identical stream to rand 0.8's
    /// `StdRng::seed_from_u64`).
    pub rng: ChaCha12Rng,
    /// Shuffled dataset-row permutation for the in-flight epoch.
    pub sampler_indices: Vec<usize>,
    /// Sampler cursor (already advanced past this checkpoint's batch).
    pub sampler_cursor: usize,
}

/// Save the optimizer state (Adam moments) to `base` (`.mpk` appended).
pub fn save_optimizer<B, M, O>(base: &Path, optimizer: &O) -> Result<()>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    CompactRecorder::new()
        .record(optimizer.to_record(), base.to_path_buf())
        .map_err(|e| DataError::Io {
            path: base.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")),
        })
}

/// Load the optimizer state from `base` (`.mpk` appended) into `optimizer`.
pub fn load_optimizer<B, M, O>(base: &Path, optimizer: O, device: &B::Device) -> Result<O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let record = CompactRecorder::new()
        .load(base.to_path_buf(), device)
        .map_err(|e| DataError::Io {
            path: base.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")),
        })?;
    Ok(optimizer.load_record(record))
}

/// Write the train-loop state sidecar as JSON.
pub fn save_train_state(path: &Path, state: &TrainCkptState) -> Result<()> {
    let json = serde_json::to_string(state).map_err(|e| DataError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")),
    })?;
    std::fs::write(path, json).map_err(|e| DataError::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Read a train-loop state sidecar.
pub fn load_train_state(path: &Path) -> Result<TrainCkptState> {
    let json = std::fs::read_to_string(path).map_err(|e| DataError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    serde_json::from_str(&json).map_err(|e| DataError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")),
    })
}
