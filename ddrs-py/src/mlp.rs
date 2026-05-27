//! Python class wrapping `Mlp<NdArray>` + the `load_mlp` constructor function.

use std::path::Path;

use burn::backend::NdArray;
use burn::tensor::{Device, Tensor, TensorData};
use ddrs::nn::mlp::Mlp;
use ddrs::training::checkpoint::load_mlp as load_mlp_impl;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

    /// Run inference on a `(R, F)` `float32` attrs batch.
    ///
    /// Returns a dict keyed by `learnable_parameters`; each value is a 1-D
    /// `float32` numpy array of length R, with values in `[0, 1]`.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        attrs: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let arr = attrs.as_array();
        let shape = arr.shape();
        let rows = shape[0];
        let cols = shape[1];
        let expected_cols = self.inner.input.weight.val().dims()[0];
        if cols != expected_cols {
            return Err(BridgeError::AttrShapeMismatch {
                rows,
                cols,
                expected_cols,
            }
            .into());
        }

        // Numpy → BURN tensor. Use as_array() indirection (matches denormalize.rs pattern)
        // since as_slice() returns NotContiguousError rather than PyErr.
        let slice: &[f32] = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "forward expects a C-contiguous (row-major) attrs array",
            )
        })?;
        let data = TensorData::new(slice.to_vec(), [rows, cols]);
        let input: Tensor<Backend, 2> = Tensor::from_data(data, &self.device);

        let raw = self.inner.forward(input);

        let out = PyDict::new_bound(py);
        // Iterate in `learnable_parameters` order so the dict key order is
        // deterministic for callers that turn it into a DataFrame.
        for key in self.inner.learnable_parameters() {
            let tensor = raw
                .get(key)
                .expect("MLP returned no entry for declared learnable_parameter");
            let vec: Vec<f32> = tensor.clone().into_data().to_vec().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "BURN tensor → Vec<f32> failed for `{key}`: {e:?}"
                ))
            })?;
            out.set_item(key, PyArray1::from_vec_bound(py, vec))?;
        }
        Ok(out)
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
