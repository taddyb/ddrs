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

/// Construct the area-mode pooling weight matrix `W ∈ R^{M × L}` such that
/// `W[i, j] = overlap(input_cell_j, output_bin_i) / s` where `s = L / M`.
///
/// Each row sums to 1. Mirrors `torch.nn.functional.interpolate(mode="area")`
/// for the 1D case (DDR uses this at `ddr/io/functions.py:22`). The result
/// is a constant matrix that depends only on shape — compute once per
/// (L, M) pair and reuse.
///
/// Sparsity: each row has at most `ceil(L/M) + 1` nonzeros for `s > 1`.
#[allow(dead_code)] // wired into tau_trim_and_downsample in next commit
fn area_pool_weights<B: Backend>(
    l: usize,
    m: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(l > 0 && m > 0 && l >= m, "need L >= M > 0; got L={l}, M={m}");
    let s = l as f32 / m as f32;
    let mut data: Vec<f32> = vec![0.0; m * l];

    for i in 0..m {
        let left = (i as f32) * s;
        let right = ((i + 1) as f32) * s;
        let j_lo = left.floor() as usize;
        let j_hi = (right.ceil() as usize).min(l);
        for j in j_lo..j_hi {
            let cell_left = (j as f32).max(left);
            let cell_right = ((j + 1) as f32).min(right);
            let weight = (cell_right - cell_left) / s;
            data[i * l + j] = weight;
        }
    }

    Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(data, [m * l]),
        device,
    )
    .reshape([m, l])
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
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    type Bp = NdArray<f32>;

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

    #[test]
    fn area_pool_weights_rows_sum_to_one() {
        let device = Default::default();
        let w = area_pool_weights::<Bp>(2139, 89, &device);
        let row_sums: Tensor<Bp, 1> = w.sum_dim(1).squeeze();
        for v in row_sums.into_data().to_vec::<f32>().unwrap() {
            assert!((v - 1.0).abs() < 1e-5, "row sum {v} != 1");
        }
    }

    #[test]
    fn area_pool_matches_block_mean_when_divisible() {
        let device = Default::default();
        // Input: 1..=48 over 48 hours, single gauge.
        let v: Vec<f32> = (1..=48).map(|x| x as f32).collect();
        let input: Tensor<Bp, 2> = Tensor::<Bp, 1>::from_data(
            burn::tensor::TensorData::new(v, [48]),
            &device,
        )
        .reshape([1, 48]);

        let w = area_pool_weights::<Bp>(48, 2, &device);
        let out: Tensor<Bp, 2> = input.matmul(w.transpose());
        let got: Vec<f32> = out.into_data().to_vec().unwrap();
        // Block 1 = mean(1..=24)  = 12.5
        // Block 2 = mean(25..=48) = 36.5
        assert!((got[0] - 12.5).abs() < 1e-5, "got {}", got[0]);
        assert!((got[1] - 36.5).abs() < 1e-5, "got {}", got[1]);
    }

    #[test]
    fn area_pool_handles_non_divisible_input() {
        let device = Default::default();
        let w = area_pool_weights::<Bp>(2139, 89, &device);
        let data: Vec<f32> = w.into_data().to_vec().unwrap();

        // Row 0 covers input range [0, 24.0337...). Cells 0-23 contribute
        // their full weight 1/s, cell 24 contributes the fractional piece.
        let s = 2139.0_f32 / 89.0;
        for j in 0..24 {
            let expected = 1.0 / s;
            assert!(
                (data[j] - expected).abs() < 1e-6,
                "row 0 col {j}: got {} want {expected}",
                data[j]
            );
        }
        let frac = (s - 24.0) / s;
        assert!(
            (data[24] - frac).abs() < 1e-4,
            "row 0 col 24: got {} want ~{frac}",
            data[24]
        );
        for j in 25..2139 {
            assert!(data[j].abs() < 1e-6, "row 0 col {j} should be 0; got {}", data[j]);
        }
    }
}
