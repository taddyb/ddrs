//! ddrs-py: PyO3 bindings for ddrs.
//!
//! See `.claude/specs/2026-05-26-sp10a-pyo3-bridge-design.md` for design.

use pyo3::prelude::*;

#[pymodule]
fn ddrs_py(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
