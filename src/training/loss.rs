//! Daily downsample + L1 loss with NaN mask.
//!
//! Mirrors `~/projects/ddr/src/ddr/scripts_utils.py::compute_daily_runoff`
//! and `scripts/train.py:62-86` (NaN-filter + L1 + warmup trim).

use burn::tensor::{backend::Backend, Tensor};
use ndarray::{s, Array2};

/// Tau-trim then mean-pool 24 hourly samples → 1 daily sample.
///
/// Input shape `(G, T_hours)`. Slicing convention from DDR
/// `compute_daily_runoff`: `[13 + tau : -11 + tau]`. After the slice
/// `T_hours_trimmed` must be a multiple of 24 (asserted).
///
/// Returns `(G, T_days)` where `T_days = T_hours_trimmed / 24`.
pub fn tau_trim_and_downsample<B: Backend>(
    predictions_hourly: Tensor<B, 2>,
    tau: u32,
) -> Tensor<B, 2> {
    let dims = predictions_hourly.dims();
    let (g, t_hours) = (dims[0], dims[1]);
    let start = 13 + tau as usize;
    // DDR's Python end index is `-11 + tau`, i.e., position `t_hours + (-11 + tau)`
    // from the start, which equals `t_hours - 11 + tau`. Use that form to avoid
    // underflow risk for any tau >= 0.
    let end = t_hours - 11 + tau as usize;
    let t_trimmed = end - start;
    assert!(
        t_trimmed.is_multiple_of(24),
        "tau-trim left {t_trimmed} hours, not a multiple of 24 (tau={tau})"
    );
    let t_days = t_trimmed / 24;
    let sliced = predictions_hourly.slice([0..g, start..end]);
    let reshaped = sliced.reshape([g, t_days, 24]);
    reshaped.mean_dim(2).squeeze::<2>()
}

pub struct FilteredPair {
    pub predictions: Array2<f32>,  // (T_days, G_kept)
    pub observations: Array2<f32>, // (T_days, G_kept)
    pub mask: Vec<bool>,           // length original G; true = kept
}

/// Filter gauges whose observations contain any NaN in the window.
pub fn filter_nan_gauges(
    daily_predictions: &Array2<f32>, // (G, T_days)
    observations: &Array2<f32>,      // (T_days, G)
) -> FilteredPair {
    let (g, t_days_p) = daily_predictions.dim();
    let (t_days_o, g2) = observations.dim();
    assert_eq!(g, g2);
    assert_eq!(t_days_p, t_days_o);
    let mask: Vec<bool> = (0..g)
        .map(|j| !observations.column(j).iter().any(|v| v.is_nan()))
        .collect();
    let n_kept = mask.iter().filter(|&&v| v).count();
    let mut pred_kept = Array2::<f32>::zeros((t_days_p, n_kept));
    let mut obs_kept = Array2::<f32>::zeros((t_days_o, n_kept));
    let mut col_idx = 0usize;
    for j in 0..g {
        if !mask[j] {
            continue;
        }
        for t in 0..t_days_p {
            pred_kept[(t, col_idx)] = daily_predictions[(j, t)];
        }
        for t in 0..t_days_o {
            obs_kept[(t, col_idx)] = observations[(t, j)];
        }
        col_idx += 1;
    }
    FilteredPair { predictions: pred_kept, observations: obs_kept, mask }
}

/// L1 loss over `(T_days_post_warmup, G_kept)`.
///
/// Mirrors `~/projects/ddr/scripts/train.py:75-85`:
///   1. Drop gauges with any NaN.
///   2. Truncate to `[warmup..]` along the time axis.
///   3. Mean of absolute differences.
pub fn l1_loss_post_warmup(
    predictions: &Array2<f32>,
    observations: &Array2<f32>,
    warmup: usize,
) -> f32 {
    let (t_days, _g) = predictions.dim();
    assert!(warmup < t_days, "warmup={warmup} >= T_days={t_days}");
    let p = predictions.slice(s![warmup.., ..]);
    let o = observations.slice(s![warmup.., ..]);
    let diff = &p - &o;
    diff.iter().map(|v| v.abs()).sum::<f32>() / (diff.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn l1_loss_post_warmup_basic() {
        let pred = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]]; // (T=3, G=2)
        let obs = array![[1.0_f32, 2.0], [4.0, 4.0], [5.0, 7.0]];
        // warmup=0: 6 entries, sum of |diff| = 0+0+1+0+0+1 = 2; mean = 2/6.
        let l = l1_loss_post_warmup(&pred, &obs, 0);
        assert!((l - 2.0 / 6.0).abs() < 1e-6);
        // warmup=1: 4 entries, sum = 1+0+0+1 = 2; mean = 0.5.
        let l = l1_loss_post_warmup(&pred, &obs, 1);
        assert!((l - 0.5).abs() < 1e-6);
    }

    #[test]
    fn filter_nan_gauges_drops_columns() {
        let pred = array![[1.0_f32, 1.5], [2.0, 2.5], [3.0, 3.5]]; // (G=3, T=2)
        let obs = array![[10.0_f32, f32::NAN, 30.0], [11.0, 21.0, 31.0]]; // (T=2, G=3)
        let f = filter_nan_gauges(&pred, &obs);
        assert_eq!(f.mask, vec![true, false, true]);
        assert_eq!(f.predictions.shape(), &[2, 2]);
        assert_eq!(f.observations.shape(), &[2, 2]);
    }
}
