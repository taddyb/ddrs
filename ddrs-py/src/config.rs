//! Config helpers exposed to Python and used internally to build MLP templates.

use std::path::Path;

use ddrs::config::{Config, MlpConfigSection};
use ddrs::nn::mlp::MlpConfig;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::BridgeError;

/// Load `Config` from a YAML path with consistent error wrapping.
pub fn load_config(path: &str) -> Result<Config, BridgeError> {
    Config::from_yaml_file(Path::new(path)).map_err(|source| BridgeError::Config {
        path: path.into(),
        source,
    })
}

/// Pull `cfg.mlp` or return a typed error if absent.
#[allow(dead_code)] // used in Task 3 (mlp.rs) and Task 5 (conus.rs)
pub fn require_mlp_section<'a>(
    cfg: &'a Config,
    path: &str,
) -> Result<&'a MlpConfigSection, BridgeError> {
    cfg.mlp.as_ref().ok_or_else(|| BridgeError::MissingMlpSection {
        path: path.to_string(),
    })
}

/// Convert a ddrs YAML `MlpConfigSection` into the ddrs `MlpConfig` used to
/// build an `Mlp<B>` template.
#[allow(dead_code)] // used in Task 3 (mlp.rs)
pub fn mlp_config_from_section(section: &MlpConfigSection) -> MlpConfig {
    MlpConfig::new(section.input_var_names.clone(), section.learnable_parameters.clone())
        .with_hidden_size(section.hidden_size)
        .with_num_hidden_layers(section.num_hidden_layers)
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
