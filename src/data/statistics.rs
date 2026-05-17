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
}
