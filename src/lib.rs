//! ddrs — Distributed Differentiable Routing in Rust.
//!
//! BURN port of the Muskingum-Cunge solver from `~/projects/ddr` (`src/ddr/`).
//! See `~/projects/ddr/CLAUDE.md` for the reference algorithm.

pub mod config;
pub mod data;
pub mod geometry;
pub mod nn;
pub mod routing;
pub mod sparse;
