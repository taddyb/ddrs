//! Python class wrapping `Mlp<NdArray>` + the `load_mlp` constructor function.

use std::path::Path;

use burn::backend::NdArray;
use burn::tensor::Device;
use ddrs::nn::mlp::Mlp;
use ddrs::training::checkpoint::load_mlp as load_mlp_impl;
use pyo3::prelude::*;

use crate::config::{load_config, mlp_config_from_section, require_mlp_section};
use crate::error::BridgeError;

type Backend = NdArray<f32>;

/// Opaque container for a loaded MLP.
///
/// The backend is fixed to `NdArray<f32>` — see the design doc for the
/// rationale (CPU-only in 10a). A future GPU variant would be a sibling
/// pyclass selected behind a Cargo feature.
#[pyclass(module = "ddrs_py")]
pub struct PyMlp {
    pub(crate) inner: Mlp<Backend>,
    pub(crate) device: Device<Backend>,
}

#[pymethods]
impl PyMlp {
    /// Names of the output parameters in column order.
    #[getter]
    fn learnable_parameters(&self) -> Vec<String> {
        self.inner.learnable_parameters().to_vec()
    }

    /// Number of input attribute columns this MLP expects.
    ///
    /// To find out which attributes these are, read the
    /// `mlp.input_var_names` list in your YAML config — the names are
    /// not stored in the checkpoint after `MlpConfig::init`.
    #[getter]
    fn input_var_names_len(&self) -> usize {
        // Inferred from the first Linear layer's weight rows.
        self.inner.input.weight.val().dims()[0]
    }
}

/// Load an MLP checkpoint.
///
/// `checkpoint` is the BASE path (no `.mpk` extension — `CompactRecorder`
/// appends it).
#[pyfunction]
#[pyo3(signature = (checkpoint, config_path))]
pub fn load_mlp(checkpoint: &str, config_path: &str) -> PyResult<PyMlp> {
    let cfg = load_config(config_path)?;
    let mlp_section = require_mlp_section(&cfg, config_path)?;
    let mlp_cfg = mlp_config_from_section(mlp_section);
    let device = Device::<Backend>::default();
    let template = mlp_cfg.init::<Backend>(&device);

    let inner = load_mlp_impl::<Backend>(Path::new(checkpoint), template, &device).map_err(
        |source| BridgeError::Checkpoint {
            path: checkpoint.into(),
            source,
        },
    )?;
    Ok(PyMlp { inner, device })
}
