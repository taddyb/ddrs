//! CONUS-wide inference: load every MERIT COMID's attributes, run the MLP,
//! denormalize. Mirrors the workflow in DDR's `merit_geometry_config.yaml`.

use std::collections::HashSet;
use std::path::Path;

use burn::tensor::{Tensor, TensorData};
use ddrs::config::Params;
use ddrs::data::store::netcdf::AttributesStore;
use ddrs::data::store::zarr::ConusAdjacencyStore;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::config::{load_config, require_mlp_section};
use crate::error::BridgeError;
use crate::mlp::Backend;

/// Load checkpoint → walk every COMID in the CONUS adjacency → return a
/// per-COMID dict of physical-unit parameters.
///
/// All arrays returned have length `N` (the number of COMIDs the
/// attributes file had data for — typically equal to the adjacency's
/// reach count). The arrays are aligned: row `i` of every key refers to
/// the COMID at `result["comid"][i]`.
#[pyfunction]
#[pyo3(signature = (attrs_nc, conus_adjacency_zarr, checkpoint, config_path))]
pub fn run_inference_over_conus<'py>(
    py: Python<'py>,
    attrs_nc: &str,
    conus_adjacency_zarr: &str,
    checkpoint: &str,
    config_path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = load_config(config_path)?;
    let mlp_section = require_mlp_section(&cfg, config_path)?;
    let attr_names = mlp_section.input_var_names.clone();

    // 1. Adjacency → ordered COMID list (in topological order).
    let adj = ConusAdjacencyStore::open(Path::new(conus_adjacency_zarr))
        .map_err(BridgeError::Zarr)?;

    // 2. Attributes → (F, N) matrix aligned to the subset of COMIDs that the
    //    netcdf actually contains. May be shorter than adj.order if the
    //    netcdf is missing rows; AttributesStore handles that internally.
    let attrs_store = AttributesStore::open(Path::new(attrs_nc), &attr_names, &adj.order)
        .map_err(BridgeError::Netcdf)?;

    // 3. Pull the resolved COMID list out of attrs_store.index.
    //    ids() returns the slice in insertion order, which matches attrs columns.
    let n_reaches = attrs_store.index.len();
    let f = attr_names.len();
    // Hard assert: a layout mismatch would silently scramble the input tensor
    // in release builds, producing finite but wrong parameter maps.
    assert_eq!(attrs_store.attrs.dim(), (f, n_reaches),
        "AttributesStore returned shape {:?}, expected (F={}, N={})",
        attrs_store.attrs.dim(), f, n_reaches);
    let resolved_comids: Vec<i64> = attrs_store.index.ids().iter().map(|c| c.0).collect();

    // 4. Reload the MLP. (We do this here rather than accepting a PyMlp
    //    parameter so the caller has a single-call API.) Note: this
    //    re-parses the YAML internally; acceptable for a one-shot
    //    CONUS-wide inference call.
    let model = crate::mlp::load_mlp(checkpoint, config_path)?;

    // 5. Build the BURN input tensor. AttributesStore stores attrs as
    //    (F, N); MLP wants (N, F). Transpose into a Vec<f32> in row-major.
    //    Impute NaN/Inf with per-attribute row_means (mirrors DDR's fill_nans).
    //
    // DDR's `fill_nans` in `~/projects/ddr/src/ddr/io/readers.py` uses
    // `torch.isnan`, which catches NaN only. We use `is_finite()` to catch
    // both NaN and ±Inf — strictly more defensive. In MERIT data observed
    // so far (9,291 non-finite values across 346k reaches × 10 attributes)
    // all non-finite values were NaN, so this is a no-op divergence. If
    // Inf values ever appear in future attribute files, DDR would pass
    // them through to the MLP (producing Inf outputs) while ddrs-py would
    // impute them.
    let mut input_buf = vec![0.0_f32; n_reaches * f];
    for row in 0..n_reaches {
        for col in 0..f {
            let v = attrs_store.attrs[(col, row)];
            input_buf[row * f + col] = if v.is_finite() {
                v
            } else {
                attrs_store.row_means[col]
            };
        }
    }
    let input: Tensor<Backend, 2> =
        Tensor::from_data(TensorData::new(input_buf, [n_reaches, f]), &model.device);

    // 6. MLP forward → raw [0, 1] parameter dict.
    let raw = model.run(input);

    // 7. Denormalize each parameter per params.parameter_ranges.
    let params: &Params = &cfg.params;
    let log_set: HashSet<&str> = params
        .log_space_parameters
        .iter()
        .map(String::as_str)
        .collect();

    let out = PyDict::new_bound(py);
    out.set_item("comid", PyArray1::from_vec_bound(py, resolved_comids))?;

    for name in model.param_order() {
        let bounds = match name.as_str() {
            "n" => params.parameter_ranges.n,
            "q_spatial" => params.parameter_ranges.q_spatial,
            "p_spatial" => params.parameter_ranges.p_spatial,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unrecognized learnable parameter `{other}` (expected n, q_spatial, or p_spatial)"
                )));
            }
        };
        let raw_t = raw
            .get(name)
            .expect("MLP returned no entry for declared learnable_parameter");
        let raw_vec: Vec<f32> = raw_t.to_data().to_vec().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "BURN tensor → Vec<f32> failed for `{name}`: {e:?}"
            ))
        })?;
        let log_space = log_set.contains(name.as_str());
        let denorm = denormalize_vec(&raw_vec, bounds, log_space);
        out.set_item(name, PyArray1::from_vec_bound(py, denorm))?;
    }

    Ok(out)
}

/// Same math as `ddrs::routing::utils::denormalize`, on a host Vec<f32>.
/// Private: callers outside this module should use `crate::denormalize` for
/// the public path; this mirror avoids allocating a PyArray for intermediate
/// values.
fn denormalize_vec(values: &[f32], bounds: [f32; 2], log_space: bool) -> Vec<f32> {
    let [lo, hi] = bounds;
    if log_space {
        debug_assert!(
            hi > 0.0,
            "denormalize_vec log_space=true requires hi > 0, got hi={hi}"
        );
        let log_min = (lo + 1e-6_f32).ln();
        let log_max = hi.ln();
        let scale = log_max - log_min;
        values.iter().map(|&v| (v * scale + log_min).exp()).collect()
    } else {
        let scale = hi - lo;
        values.iter().map(|&v| v * scale + lo).collect()
    }
}
