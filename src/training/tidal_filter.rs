//! Godin tidal filter on the hourly routed discharge of gauges on stage rows.
//!
//! A daily mean of an hourly M2 tide (12.42 h) aliases to a 14.8-day period,
//! the same period as the spring-neap cycle. The Godin filter (running means
//! of 24, 24 and 25 h in sequence) removes the semidiurnal and diurnal bands
//! first, so the daily pooling that follows keeps no tidal alias. It runs in
//! training and in eval, on the hourly series before `tau_trim_and_downsample`.
//!
//! The filter uses Burn tensor ops only, so on the training path
//! (`B = Autodiff<I>`) it stays on the tape and the loss gradient flows
//! through it. It sums shifted slices (24, 24 and 25 terms) and never takes a
//! global cumulative sum: an f32 cumsum over a 60,624-hour eval series drifts
//! by about 1 m3/s on a late daily mean.
//!
//! Edge handling: the series is padded with its first and last hourly value
//! for 36 h at each end, so the shape is unchanged. This is an approximation
//! for the first and last 36 h of a window. The training warmup covers the
//! front. The last 1.5 days of each training window carry a slightly biased
//! filter output.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ndarray::Array2;

use crate::config::Config;
use crate::data::{RoutingBatch, Staid};
use crate::routing::stage::downstream_closure;

/// Padding at each end, hours. The three windows reach 35 h each way.
const PAD: usize = 36;

/// Running mean of `w` consecutive hours: output `i` is the mean of
/// `x[i..i + w]`. The series is `w - 1` shorter.
fn running_mean<B: Backend>(x: Tensor<B, 2>, w: usize) -> Tensor<B, 2> {
    let [k, len] = x.dims();
    let out_len = len - (w - 1);
    let mut acc = x.clone().slice([0..k, 0..out_len]);
    for j in 1..w {
        acc = acc + x.clone().slice([0..k, j..j + out_len]);
    }
    acc / (w as f32)
}

/// Godin filter of every row of `x` (shape `(k, T)`), same shape out.
fn godin_rows<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let [k, t] = x.dims();
    let first = x.clone().slice([0..k, 0..1]).repeat_dim(1, PAD);
    let last = x.clone().slice([0..k, t - 1..t]).repeat_dim(1, PAD);
    let padded = Tensor::cat(vec![first, x, last], 1); // (k, T + 72)
    // Output index n of the third pass sits at padded position n + 35, so the
    // series starts at n = 1. The window pairs 24 + 24 + 25 are centred overall.
    let y = running_mean(running_mean(running_mean(padded, 24), 24), 25);
    y.slice([0..k, 1..1 + t])
}

/// Filter the listed gauge rows of `hourly` (shape `(G, T)`) with the Godin
/// filter. Other rows are copied bitwise. `rows` must be strictly ascending.
pub fn godin_filter<B: Backend>(hourly: Tensor<B, 2>, rows: &[usize]) -> Tensor<B, 2> {
    if rows.is_empty() {
        return hourly;
    }
    let [g, t] = hourly.dims();
    assert!(
        rows.windows(2).all(|w| w[0] < w[1]) && rows[rows.len() - 1] < g,
        "godin_filter: rows must be ascending and below {g}, got {rows:?}"
    );
    assert!(t >= 1, "godin_filter: empty time axis");
    let picked: Vec<Tensor<B, 2>> =
        rows.iter().map(|&r| hourly.clone().slice([r..r + 1, 0..t])).collect();
    let filtered = godin_rows(Tensor::cat(picked, 0));
    let mut parts: Vec<Tensor<B, 2>> = Vec::new();
    let mut next = 0usize;
    for (i, &r) in rows.iter().enumerate() {
        if r > next {
            parts.push(hourly.clone().slice([next..r, 0..t]));
        }
        parts.push(filtered.clone().slice([i..i + 1, 0..t]));
        next = r + 1;
    }
    if next < g {
        parts.push(hourly.slice([next..g, 0..t]));
    }
    Tensor::cat(parts, 0)
}

/// Gauge rows the filter acts on: gauges whose reach is in the stage set, and
/// that have at least one finite value in their column of `observations`.
///
/// A NaN-only gauge (such as a pseudo-gauge at the network outlet) is left out.
/// `observations` is `(T_days, G)` with columns in gauge order. Training passes
/// `batch.observations`; eval passes the full test-period observations. Returns
/// an empty list when the stage block is off or the batch has no stage seed.
pub fn stage_gauge_rows(
    cfg: &Config,
    batch: &RoutingBatch,
    observations: &Array2<f32>,
) -> Vec<usize> {
    let n_gauges = batch.gauge_staids.len();
    assert_eq!(
        observations.ncols(),
        n_gauges,
        "stage_gauge_rows: observations have {} columns, the batch has {n_gauges} gauges",
        observations.ncols()
    );
    if !cfg.stage_routing_on() {
        return Vec::new();
    }
    let Some(stage) = batch.stage.as_ref() else {
        return Vec::new();
    };
    if stage.seed_rows.as_ref().is_some_and(|s| s.is_empty()) {
        return Vec::new();
    }
    let all: Vec<usize>;
    let seeds: &[usize] = match stage.seed_rows.as_deref() {
        Some(s) => s,
        None => {
            all = (0..batch.adjacency.n).collect();
            &all
        }
    };
    let (in_stage, _) = downstream_closure(&batch.adjacency, seeds);
    (0..n_gauges)
        .filter(|&gi| {
            let on_stage = batch.outflow_idx[gi]
                .iter()
                .any(|&r| in_stage.get(r).copied().unwrap_or(false));
            on_stage && observations.column(gi).iter().any(|v| v.is_finite())
        })
        .collect()
}

/// Run-log line for the gauges the filter acts on. `phase` is `train` or `eval`.
pub fn godin_log_line(phase: &str, ids: &[Staid]) -> String {
    let names: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    let noun = if ids.len() == 1 { "gauge" } else { "gauges" };
    format!("{phase}: tidal filter: godin at {} {noun} ({})", ids.len(), names.join(", "))
}
