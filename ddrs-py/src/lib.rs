//! ddrs-py: PyO3 bindings for ddrs.
//!
//! See `.claude/specs/2026-05-26-sp10a-pyo3-bridge-design.md` for design.

use pyo3::prelude::*;

mod config;
mod conus;
mod denormalize;
mod error;
mod mlp;

#[pymodule]
fn ddrs_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(config::parameter_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(conus::run_inference_over_conus, m)?)?;
    m.add_function(wrap_pyfunction!(denormalize::denormalize, m)?)?;
    m.add_function(wrap_pyfunction!(mlp::load_mlp, m)?)?;
    m.add_class::<mlp::PyMlp>()?;
    Ok(())
}
