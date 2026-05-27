//! SP-10: CUDA Graphs for per-timestep launch-overhead collapse.
//!
//! Captures the per-timestep kernel sequence (forward + backward) into a
//! `CUgraphExec` once during `MuskingumCunge::setup_inputs`, then replays it
//! per timestep so the CPU issues 1 `cuGraphLaunch` instead of ~100
//! `cuLaunchKernel`s.
//!
//! See `.claude/specs/2026-05-26-sp10-cuda-graphs-design.md`.

pub mod capture;
pub mod scratch;

pub use capture::{CaptureError, CudaGraph, capture_on_stream, capture_status};
pub use scratch::PersistentScratch;
