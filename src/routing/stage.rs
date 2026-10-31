//! Per-network constants for the stage routing block (host side, f32).
//!
//! The stage set is a downstream-closed set of reaches that drains to exactly
//! one network outlet. This module builds the set, the bed elevations, the
//! edge lengths, the edge masks on the `CsrPattern` layout and the dam rows.

use crate::sparse::tree_lu::assert_tree;
use crate::sparse::{CsrPattern, SparseAdjacency};

const SECONDS_PER_DAY: f32 = 86_400.0;

/// Constants of one routed network for the stage step.
#[derive(Clone, Debug)]
pub struct StageNetwork {
    pub n: usize,
    /// Downstream reach of each reach, -1 at a network outlet.
    pub down: Vec<i32>,
    /// The stage set's one outlet (C1). It is a network outlet, because the set is downstream-closed.
    pub outlet: usize,
    /// Downstream-closed stage membership.
    pub in_stage: Vec<bool>,
    /// Size of the stage set.
    pub n_stage: usize,
    /// Reaches added by the downstream closure.
    pub n_added: usize,
    /// Bed at the reach outlet; `anchor_m` at every network outlet.
    pub z_out: Vec<f32>,
    /// `z_out + slope * length / 2` (raw slope, no floor).
    pub z_mid: Vec<f32>,
    /// `(L_i + L_down)/2`; at a network outlet `L_i / 2` (C21).
    pub edge_length: Vec<f32>,
    /// Length nnz, 1.0 where the column is MC and the row is stage.
    pub adj_mc_to_stage: Vec<f32>,
    /// Length nnz, 1.0 where both ends are stage.
    pub adj_stage_to_stage: Vec<f32>,
    /// Dams inside the stage set.
    pub dam_rows: Vec<usize>,
    /// Dam residence time in seconds, converted from days (C30).
    pub dam_t_seconds: Vec<f32>,
}

fn down_of(adj: &SparseAdjacency) -> Vec<i32> {
    let mut down = vec![-1i32; adj.n];
    for (&r, &c) in adj.rows.iter().zip(&adj.cols) {
        if r != c {
            down[c as usize] = r;
        }
    }
    down
}

/// Mark `seeds` and every reach downstream of them. Returns the mask and the
/// number of reaches the closure added beyond the seeds.
pub fn downstream_closure(adj: &SparseAdjacency, seeds: &[usize]) -> (Vec<bool>, usize) {
    let down = down_of(adj);
    let mut in_stage = vec![false; adj.n];
    for &s in seeds {
        in_stage[s] = true;
    }
    let seed_count = in_stage.iter().filter(|&&b| b).count();
    // rows >= cols: a downstream reach has a higher index than its upstream
    // neighbour, so an ascending sweep marks the whole path in one pass.
    for i in 0..adj.n {
        if in_stage[i] && down[i] >= 0 {
            in_stage[down[i] as usize] = true;
        }
    }
    let total = in_stage.iter().filter(|&&b| b).count();
    (in_stage, total - seed_count)
}

/// Bed elevation at each reach outlet and midpoint. Integrates `slope * length`
/// upstream from `anchor_m` at every network outlet.
pub fn bed_elevation(adj: &SparseAdjacency, anchor_m: f32) -> (Vec<f32>, Vec<f32>) {
    let down = down_of(adj);
    let mut z_out = vec![anchor_m; adj.n];
    for i in (0..adj.n).rev() {
        if down[i] >= 0 {
            let d = down[i] as usize;
            z_out[i] = z_out[d] + adj.slope[d] * adj.length_m[d];
        }
    }
    let z_mid = (0..adj.n)
        .map(|i| z_out[i] + adj.slope[i] * adj.length_m[i] / 2.0)
        .collect();
    (z_out, z_mid)
}

impl StageNetwork {
    /// Build the constants. `seeds = None` means every reach. `dams` holds
    /// `(row, T in days)`. A dam row outside the stage set is skipped. Returns
    /// `Err` when the pattern is not a tree or the stage set has 0 or 2+ outlets.
    pub fn build(
        adj: &SparseAdjacency,
        pattern: &CsrPattern,
        seeds: Option<&[usize]>,
        anchor_m: f32,
        dams: &[(usize, f32)],
    ) -> Result<Self, String> {
        assert_tree(pattern)?;
        let n = adj.n;
        let down = down_of(adj);
        let all: Vec<usize>;
        let seed_slice = match seeds {
            Some(s) => s,
            None => {
                all = (0..n).collect();
                &all
            }
        };
        if let Some(&bad) = seed_slice.iter().find(|&&s| s >= n) {
            return Err(format!("stage seed row {bad} is out of range (n = {n})"));
        }
        let (in_stage, n_added) = downstream_closure(adj, seed_slice);
        let n_stage = in_stage.iter().filter(|&&b| b).count();
        let outlets: Vec<usize> = (0..n).filter(|&i| in_stage[i] && down[i] < 0).collect();
        if outlets.len() != 1 {
            return Err(format!(
                "the stage set must drain to exactly 1 outlet, found {} outlet(s) {:?}",
                outlets.len(),
                outlets
            ));
        }
        let outlet = outlets[0];

        let (z_out, z_mid) = bed_elevation(adj, anchor_m);
        let edge_length = (0..n)
            .map(|i| {
                if down[i] < 0 {
                    adj.length_m[i] / 2.0
                } else {
                    (adj.length_m[i] + adj.length_m[down[i] as usize]) / 2.0
                }
            })
            .collect();

        let nnz = pattern.nnz();
        let mut adj_mc_to_stage = vec![0.0f32; nnz];
        let mut adj_stage_to_stage = vec![0.0f32; nnz];
        for s in 0..nnz {
            let r = pattern.row_for_nnz[s] as usize;
            let c = pattern.col[s] as usize;
            if r == c {
                continue;
            }
            if in_stage[r] && in_stage[c] {
                adj_stage_to_stage[s] = 1.0;
            } else if in_stage[r] {
                adj_mc_to_stage[s] = 1.0;
            }
        }

        let mut dam_rows = Vec::new();
        let mut dam_t_seconds = Vec::new();
        for &(row, t_days) in dams {
            if row < n && in_stage[row] {
                dam_rows.push(row);
                dam_t_seconds.push(t_days * SECONDS_PER_DAY);
            }
        }

        Ok(Self {
            n,
            down,
            outlet,
            in_stage,
            n_stage,
            n_added,
            z_out,
            z_mid,
            edge_length,
            adj_mc_to_stage,
            adj_stage_to_stage,
            dam_rows,
            dam_t_seconds,
        })
    }
}
