//! Python entry for `routing::utils::denormalize`.
//!
//! Re-implemented as a pure ndarray op (rather than routing through a BURN
//! tensor) because the input is already on the host as a numpy array — the
//! BURN round-trip would just be allocation churn.

use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn denormalize<'py>(
    py: Python<'py>,
    values: PyReadonlyArrayDyn<'py, f32>,
    bounds: (f32, f32),
    log_space: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let arr = values.as_array();
    if arr.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "denormalize expects a 1-D array, got shape {:?}",
            arr.shape()
        )));
    }
    let (lo, hi) = bounds;
    let input = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("denormalize expects a contiguous array"))?;
    let out: Vec<f32> = if log_space {
        if hi <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "denormalize log_space=True requires hi > 0, got hi={hi}"
            )));
        }
        let log_min = (lo + 1e-6_f32).ln();
        let log_max = hi.ln();
        let scale = log_max - log_min;
        input.iter().map(|&v| (v * scale + log_min).exp()).collect()
    } else {
        let scale = hi - lo;
        input.iter().map(|&v| v * scale + lo).collect()
    };
    Ok(PyArray1::from_vec_bound(py, out))
}
