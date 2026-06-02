//! NSE / RMSE / KGE per-gauge metrics.
//!
//! Mirrors `~/projects/ddr/src/ddr/validation/metrics.py::Metrics` (the subset
//! SP-4 logs per batch). NaN-tolerant per DDR semantics: pairs with NaN in
//! either pred or target are masked out of the accumulators.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub nse: Vec<f32>,
    pub rmse: Vec<f32>,
    pub kge: Vec<f32>,
    pub bias: Vec<f32>,
    pub fhv: Vec<f32>,
    pub flv: Vec<f32>,
}

impl Metrics {
    /// Compute per-gauge NSE, RMSE, and KGE.
    ///
    /// `pred` and `target` are `[gauges, timesteps]`. Non-finite pairs are
    /// masked before accumulation (mirrors DDR's NaN masking).
    pub fn compute(pred: &Array2<f32>, target: &Array2<f32>) -> Self {
        let (g, _t) = pred.dim();
        let mut nse = Vec::with_capacity(g);
        let mut rmse = Vec::with_capacity(g);
        let mut kge = Vec::with_capacity(g);
        let mut bias = Vec::with_capacity(g);
        let mut fhv = Vec::with_capacity(g);
        let mut flv = Vec::with_capacity(g);
        for j in 0..g {
            let p = pred.row(j);
            let o = target.row(j);
            let pairs: Vec<(f32, f32)> = p
                .iter()
                .zip(o.iter())
                .filter_map(|(&pi, &oi)| {
                    if pi.is_finite() && oi.is_finite() {
                        Some((pi, oi))
                    } else {
                        None
                    }
                })
                .collect();
            if pairs.is_empty() {
                nse.push(f32::NAN);
                rmse.push(f32::NAN);
                kge.push(f32::NAN);
                bias.push(f32::NAN);
                fhv.push(f32::NAN);
                flv.push(f32::NAN);
                continue;
            }
            let n = pairs.len() as f32;
            let p_mean = pairs.iter().map(|x| x.0).sum::<f32>() / n;
            let o_mean = pairs.iter().map(|x| x.1).sum::<f32>() / n;
            let sse = pairs
                .iter()
                .map(|(p, o)| (p - o) * (p - o))
                .sum::<f32>();
            let sso = pairs
                .iter()
                .map(|(_, o)| (o - o_mean) * (o - o_mean))
                .sum::<f32>();
            nse.push(if sso > 0.0 {
                1.0 - sse / sso
            } else {
                f32::NAN
            });
            rmse.push((sse / n).sqrt());

            let p_var = pairs
                .iter()
                .map(|(p, _)| (p - p_mean) * (p - p_mean))
                .sum::<f32>()
                / n;
            let o_var = pairs
                .iter()
                .map(|(_, o)| (o - o_mean) * (o - o_mean))
                .sum::<f32>()
                / n;
            let p_std = p_var.sqrt();
            let o_std = o_var.sqrt();
            let cov = pairs
                .iter()
                .map(|(p, o)| (p - p_mean) * (o - o_mean))
                .sum::<f32>()
                / n;
            let r = if p_std > 0.0 && o_std > 0.0 {
                cov / (p_std * o_std)
            } else {
                f32::NAN
            };
            let alpha = if o_std > 0.0 {
                p_std / o_std
            } else {
                f32::NAN
            };
            let beta = if o_mean.abs() > 0.0 {
                p_mean / o_mean
            } else {
                f32::NAN
            };
            let kge_val = 1.0
                - ((r - 1.0).powi(2) + (alpha - 1.0).powi(2) + (beta - 1.0).powi(2)).sqrt();
            kge.push(kge_val);

            // Mirrors DDR validation/metrics.py:46,92-95.
            // bias: mean(pred - obs). FHV/FLV: FDC-volume biases — sort each
            // series independently, take top 2% / bottom 30% slices,
            // 100 * sum(p-o)/sum(o) over each slice. NOT timestep-paired.
            bias.push(pairs.iter().map(|(p, o)| p - o).sum::<f32>() / n);

            let mut p_sorted: Vec<f32> = pairs.iter().map(|x| x.0).collect();
            let mut o_sorted: Vec<f32> = pairs.iter().map(|x| x.1).collect();
            p_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            o_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n_us = pairs.len();
            let idx_low = (0.3 * n_us as f32).round() as usize;
            let idx_high = (0.98 * n_us as f32).round() as usize;

            let p_bias = |p: &[f32], o: &[f32]| -> f32 {
                let so: f32 = o.iter().sum();
                if so.abs() > 0.0 {
                    100.0 * p.iter().zip(o).map(|(a, b)| a - b).sum::<f32>() / so
                } else {
                    f32::NAN
                }
            };
            flv.push(p_bias(&p_sorted[..idx_low], &o_sorted[..idx_low]));
            fhv.push(p_bias(&p_sorted[idx_high..], &o_sorted[idx_high..]));
        }
        Self { nse, rmse, kge, bias, fhv, flv }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn nse_one_for_perfect_match() {
        // n>=50 so the FHV slice (top 2% = round(0.98*N)..N) is non-empty.
        let row: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let p = Array2::from_shape_vec((1, 100), row.clone()).unwrap();
        let o = Array2::from_shape_vec((1, 100), row).unwrap();
        let m = Metrics::compute(&p, &o);
        assert!((m.nse[0] - 1.0).abs() < 1e-6);
        assert!(m.rmse[0] < 1e-6);
        assert!(m.bias[0].abs() < 1e-6);
        assert!(m.fhv[0].abs() < 1e-3);
        assert!(m.flv[0].abs() < 1e-3);
    }

    #[test]
    fn bias_matches_mean_diff() {
        let p = array![[2.0_f32, 4.0, 6.0, 8.0]];
        let o = array![[1.0_f32, 2.0, 3.0, 4.0]];
        let m = Metrics::compute(&p, &o);
        // mean(p - o) = mean(1,2,3,4) = 2.5
        assert!((m.bias[0] - 2.5).abs() < 1e-6);
    }
}
