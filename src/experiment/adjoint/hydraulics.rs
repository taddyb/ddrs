//! Hydraulic travel time from the trained geometry, for validating the kernel.
//!
//! The Muskingum storage constant `K = L / c` of a reach equals the mean lag
//! of its linear impulse response (for any X), so the kernel-weighted mean lag
//! from reach `i` to the gauge should equal the path sum `Σ_j K_j` over reaches
//! `j` from `i` (inclusive) down to the gauge (inclusive), evaluated at the
//! flow state of the window. `reach_k_hours` mirrors `forward_chain_inner`
//! S1–S18 for `ddr_match: false` (exact trapezoidal celerity).

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::config::Config;
use crate::sparse::SparseAdjacency;

/// Per-reach Muskingum `K` in hours at the given discharge (`q_t`, m³/s).
/// `n`, `p`, `q` are the *denormalized* channel parameters; `slope` the
/// clamped slope; `length` in metres.
pub fn reach_k_hours<I: Backend>(
    cfg: &Config,
    n: &Tensor<I, 1>,
    p: &Tensor<I, 1>,
    q: &Tensor<I, 1>,
    q_t: Tensor<I, 1>,
    slope: &Tensor<I, 1>,
    length: &Tensor<I, 1>,
    // Per-reach learned stage-roughness exponent (physical, `d_ref = 1`);
    // `None` falls back to the config scalar, exactly as the solver does.
    gamma_field: Option<&Tensor<I, 1>>,
) -> Vec<f32> {
    assert!(!cfg.params.ddr_match, "hydraulic K mirrors the ddr_match=false celerity");
    let depth_lb = cfg.params.attribute_minimums.depth;
    let bottom_width_lb = cfg.params.attribute_minimums.bottom_width;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let discharge_lb = cfg.params.attribute_minimums.discharge;

    // Stage-dependent roughness, if configured. This file MUST track
    // `mmc_op.rs` S5/S17 exactly: it is what the landscape instrument measures
    // K on, so a divergence here would silently probe a different surface than
    // the one the solver routes.
    let (gamma, d_ref) = match gamma_field {
        Some(_) => (0.0, 1.0), // learned: the field carries it, d_ref is 1
        None => cfg.params.stage_roughness_params(),
    };

    let q_t = q_t.clamp_min(discharge_lb);
    let q_eps = q.clone() + 1e-6_f32;
    let numerator = q_t * n.clone() * (q_eps.clone() + 1.0);
    let numerator = if gamma != 0.0 && d_ref != 1.0 {
        numerator * d_ref.powf(gamma)
    } else {
        numerator
    };
    let denominator = p.clone() * slope.clone().sqrt() + 1e-8_f32;
    let ratio = numerator / denominator;
    let exponent = match gamma_field {
        Some(g) => (q_eps.clone() * 3.0 + g.clone() * 3.0 + 5.0).recip() * 3.0,
        None => (q_eps.clone() * 3.0 + (5.0 + 3.0 * gamma)).recip() * 3.0,
    };
    let depth = ratio.powf(exponent).clamp_min(depth_lb);
    let top_width = p.clone() * depth.clone().powf(q_eps.clone());
    let side_slope = (top_width.clone() * q_eps / (depth.clone() * 2.0)).clamp(0.5, 50.0);
    let bottom_width = (top_width.clone() - side_slope.clone() * depth.clone() * 2.0).clamp_min(bottom_width_lb);
    let area = (top_width.clone() + bottom_width.clone()) * depth.clone() / 2.0;
    let root = (side_slope.powf_scalar(2.0) + 1.0).sqrt();
    let wp = bottom_width + depth.clone() * root.clone() * 2.0;
    let hyd_radius = area.clone() / wp.clone();
    // Stage-dependent roughness rides on the velocity too, not only on the
    // depth inversion — must match mmc_op.rs S15 exactly.
    let n_recip = match gamma_field {
        Some(g) => n.clone().recip() * depth.clone().powf(g.clone()),
        None if gamma != 0.0 => n.clone().recip() * (depth.clone() / d_ref).powf_scalar(gamma),
        None => n.clone().recip(),
    };
    let velocity = (n_recip * hyd_radius.powf_scalar(2.0 / 3.0) * slope.clone().sqrt()).clamp(velocity_lb, 15.0);
    let beta = -(area.clone() * root) / (top_width.clone() * wp) * (4.0 / 3.0) + (5.0 / 3.0);
    let beta = match gamma_field {
        Some(g) => beta + area * g.clone() / (top_width * depth),
        None if gamma != 0.0 => beta + area * gamma / (top_width * depth),
        None => beta,
    };
    let celerity = velocity * beta;
    let k_seconds = length.clone() / celerity;
    let k: Vec<f32> = k_seconds.into_data().to_vec::<f32>().expect("f32");
    k.into_iter().map(|s| s / 3600.0).collect()
}

/// Mean over hours `[from, to)` of `reach_k_hours` evaluated at each hour's
/// routed discharge (`runoff` is row-major `(N, T)`).
#[allow(clippy::too_many_arguments)]
pub fn mean_reach_k_hours<I: Backend>(
    cfg: &Config,
    n: &Tensor<I, 1>,
    p: &Tensor<I, 1>,
    q: &Tensor<I, 1>,
    runoff: &Tensor<I, 2>,
    slope: &Tensor<I, 1>,
    length: &Tensor<I, 1>,
    from: usize,
    to: usize,
    gamma_field: Option<&Tensor<I, 1>>,
) -> Vec<f32> {
    let [n_reach, _t] = runoff.dims();
    let mut acc = vec![0.0f64; n_reach];
    let mut count = 0usize;
    for h in from..to {
        let q_t = runoff.clone().slice([0..n_reach, h..h + 1]).reshape([n_reach]);
        let k = reach_k_hours::<I>(cfg, n, p, q, q_t, slope, length, gamma_field);
        for (a, v) in acc.iter_mut().zip(k) {
            *a += v as f64;
        }
        count += 1;
    }
    acc.into_iter().map(|a| (a / count.max(1) as f64) as f32).collect()
}

/// Path sum of per-reach `K` (hours) from each reach (inclusive) down to the
/// gauge reach (inclusive). NaN where the reach does not drain to the gauge.
pub fn path_travel_time_hours(adj: &SparseAdjacency, gauge_row: usize, k_hours: &[f32]) -> Vec<f32> {
    let n = adj.n;
    let mut downstream: Vec<Option<usize>> = vec![None; n];
    for (r, c) in adj.rows.iter().zip(&adj.cols) {
        let (r, c) = (*r as usize, *c as usize);
        if r != c {
            downstream[c] = Some(r);
        }
    }
    (0..n)
        .map(|start| {
            let mut t = 0.0f32;
            let mut i = start;
            let mut hops = 0;
            loop {
                t += k_hours[i];
                if i == gauge_row {
                    return t;
                }
                match downstream[i] {
                    Some(d) if hops < n => {
                        i = d;
                        hops += 1;
                    }
                    _ => return f32::NAN,
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_sum_walks_to_gauge_inclusive() {
        // chain 0 → 1 → 2 (gauge at 2); K = 1, 2, 3 h
        let dense = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let adj = SparseAdjacency::from_dense(3, &dense, vec![100.0; 3], vec![0.001; 3]);
        let t = path_travel_time_hours(&adj, 2, &[1.0, 2.0, 3.0]);
        assert_eq!(t, vec![6.0, 5.0, 3.0]);
    }
}
