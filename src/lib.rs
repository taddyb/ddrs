//! ddrs — Distributed Differentiable Routing in Rust.
//!
//! BURN port of the Muskingum-Cunge solver from `~/projects/ddr` (`src/ddr/`).
//! See `~/projects/ddr/CLAUDE.md` for the reference algorithm.

pub mod adjacency;
pub mod baseline;
pub mod cli;
pub mod config;
pub mod cuda_graph;
pub mod data;
pub mod dump_parameters;
pub mod error;
pub mod geometry;
pub mod nn;
pub mod routing;
pub mod sandbox;
pub mod sparse;
pub mod training;
