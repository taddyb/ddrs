//! Pre-computed attribute statistics + NaN-handling helpers.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/statistics.py::set_statistics` (read
//! path only) and the `naninfmean` / `fill_nans` helpers in
//! `~/projects/ddr/src/ddr/io/readers.py:315-368`.
//!
//! We do **not** recompute statistics. DDR caches them as JSON next to the
//! attributes file; we load that JSON. If a user changes the attribute list
//! they regenerate the cache under DDR's `uv` venv and re-point
//! `config.data_sources.statistics` here.

use ndarray::{Array1, ArrayViewMut1, ArrayViewMut2};

/// Replace `NaN` entries in a 1D array with `row_mean`. Mirrors `fill_nans`
/// in `readers.py:332-368` for the 1D case.
pub fn fill_nans_1d(mut attr: ArrayViewMut1<f32>, row_mean: f32) {
    for v in attr.iter_mut() {
        if v.is_nan() {
            *v = row_mean;
        }
    }
}

/// Replace `NaN` entries in a `(F, N)` array with the per-row mean. `row_means`
/// has length `F`. Mirrors `fill_nans` (readers.py:332-368) for the 2D case
/// — specifically the branch that broadcasts a length-F vector across N
/// columns.
pub fn fill_nans(mut attr: ArrayViewMut2<f32>, row_means: &Array1<f32>) {
    let (f, _n) = attr.dim();
    assert_eq!(
        f,
        row_means.len(),
        "fill_nans: row_means length {} does not match F={}",
        row_means.len(),
        f
    );
    for (i, mut row) in attr.outer_iter_mut().enumerate() {
        let m = row_means[i];
        for v in row.iter_mut() {
            if v.is_nan() {
                *v = m;
            }
        }
    }
}

/// Mean over the finite values of an array. Returns `f32::NAN` if no finite
/// values exist. Mirrors `naninfmean` (readers.py:315-330).
pub fn naninfmean(arr: &[f32]) -> f32 {
    let mut sum = 0.0_f64;
    let mut n = 0_usize;
    for &x in arr {
        if x.is_finite() {
            sum += x as f64;
            n += 1;
        }
    }
    if n == 0 {
        f32::NAN
    } else {
        (sum / n as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naninfmean_mixed() {
        let v: Vec<f32> = vec![1.0, 2.0, f32::NAN, 3.0, f32::INFINITY, -f32::INFINITY];
        assert_eq!(naninfmean(&v), 2.0);
    }

    #[test]
    fn naninfmean_all_nonfinite_returns_nan() {
        let v: Vec<f32> = vec![f32::NAN, f32::INFINITY, -f32::INFINITY];
        assert!(naninfmean(&v).is_nan());
    }

    #[test]
    fn naninfmean_empty_returns_nan() {
        assert!(naninfmean(&[]).is_nan());
    }

    #[test]
    fn fill_nans_1d_replaces_nan_with_row_mean() {
        use ndarray::Array1;
        let mut a: Array1<f32> = Array1::from(vec![1.0, f32::NAN, 3.0, f32::NAN]);
        fill_nans_1d(a.view_mut(), 2.5);
        assert_eq!(a.as_slice().unwrap(), &[1.0, 2.5, 3.0, 2.5]);
    }

    #[test]
    fn fill_nans_2d_broadcasts_row_means_across_columns() {
        use ndarray::{Array1, Array2};
        let mut a: Array2<f32> = Array2::from_shape_vec(
            (2, 3),
            vec![1.0, f32::NAN, 3.0,
                 f32::NAN, 5.0, f32::NAN],
        )
        .unwrap();
        let row_means: Array1<f32> = Array1::from(vec![10.0, 20.0]);
        fill_nans(a.view_mut(), &row_means);
        assert_eq!(
            a.as_slice().unwrap(),
            &[1.0, 10.0, 3.0,  20.0, 5.0, 20.0]
        );
    }

    #[test]
    fn fill_nans_2d_wrong_row_means_length_panics() {
        use ndarray::{Array1, Array2};
        let mut a: Array2<f32> = Array2::zeros((2, 3));
        let row_means: Array1<f32> = Array1::from(vec![1.0]);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fill_nans(a.view_mut(), &row_means);
        }));
        assert!(r.is_err());
    }
}
