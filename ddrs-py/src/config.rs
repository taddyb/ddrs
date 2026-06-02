//! Config helpers exposed to Python and used internally to build KAN-head templates.

use std::path::Path;

use ddrs::config::{Config, KanHeadConfigSection};
use ddrs::nn::kan_head::KanHeadConfig;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::BridgeError;

/// Load `Config` from a YAML path with consistent error wrapping.
pub(crate) fn load_config(path: &str) -> Result<Config, BridgeError> {
    Config::from_yaml_file(Path::new(path)).map_err(|source| BridgeError::Config {
        path: path.into(),
        source,
    })
}

/// Pull `cfg.kan_head` or return a typed error if absent.
pub(crate) fn require_kan_head_section<'a>(
    cfg: &'a Config,
    path: &str,
) -> Result<&'a KanHeadConfigSection, BridgeError> {
    cfg.kan_head
        .as_ref()
        .ok_or_else(|| BridgeError::MissingKanHeadSection {
            path: path.to_string(),
        })
}

/// Convert a ddrs YAML `KanHeadConfigSection` into the ddrs `KanHeadConfig`
/// used to build a `KanHead<B>` template. `seed` is the top-level `cfg.seed`
/// — KanHeadConfig requires it because every inner KanLayer init draws from
/// it (DDR-Python `kan.py:24-34` quirk).
pub(crate) fn kan_head_config_from_section(
    section: &KanHeadConfigSection,
    seed: u64,
) -> KanHeadConfig {
    KanHeadConfig::new(
        section.input_var_names.clone(),
        section.learnable_parameters.clone(),
        seed,
    )
    .with_hidden_size(section.hidden_size)
    .with_num_hidden_layers(section.num_hidden_layers)
    .with_grid(section.grid)
    .with_k(section.k)
}

/// Python entry point.
///
/// Returns `dict[str, tuple[tuple[float, float], bool]]`. Keys are the
/// three parameter names; the bool flag is `True` iff the parameter is in
/// `log_space_parameters`.
#[pyfunction]
pub fn parameter_bounds<'py>(
    py: Python<'py>,
    config_path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = load_config(config_path)?;
    let log_set: std::collections::HashSet<&str> = cfg
        .params
        .log_space_parameters
        .iter()
        .map(String::as_str)
        .collect();

    let ranges = &cfg.params.parameter_ranges;
    let entries: [(&str, [f32; 2]); 3] = [
        ("n", ranges.n),
        ("q_spatial", ranges.q_spatial),
        ("p_spatial", ranges.p_spatial),
    ];

    let out = PyDict::new_bound(py);
    for (name, [lo, hi]) in entries {
        let bounds_tup = (lo as f64, hi as f64);
        let log = log_set.contains(name);
        out.set_item(name, (bounds_tup, log))?;
    }
    Ok(out)
}
