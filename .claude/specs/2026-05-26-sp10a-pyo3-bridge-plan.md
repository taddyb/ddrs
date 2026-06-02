# SP-10a — PyO3 inference bridge: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `ddrs-py` crate exposing the ddrs MLP + denormalize + a CONUS-wide inference helper to Python, so notebook users can load a `.mpk` checkpoint and produce a per-COMID parameter field on CPU.

**Architecture:** New sibling crate `ddrs-py/` (NOT a workspace member) with `crate-type = ["cdylib"]`, depending on `ddrs` as a path dep + `burn-ndarray` for the backend. Wraps four pure functions and one `PyMlp` class. Built with `maturin develop --release` into a `uv` venv. No changes to the `ddrs` crate.

**Tech Stack:** Rust 2021, `pyo3 = "0.22"`, `numpy = "0.22"`, `burn` 0.21 with `ndarray` + `std` features, `serde_json` (already a ddrs dep), `maturin` build backend, `uv` for the Python side.

**Spec reference:** `.claude/specs/2026-05-26-sp10a-pyo3-bridge-design.md`. Read it first.

---

## File Structure

All new files. Nothing in `ddrs/` is modified.

```
ddrs-py/                       # NEW — sibling crate, outside the ddrs workspace
├── .gitignore                 # target/, *.egg-info, __pycache__, .venv
├── Cargo.toml                 # cdylib, pyo3 + numpy + burn-ndarray + ddrs path dep
├── pyproject.toml             # maturin build backend, package name "ddrs_py"
├── README.md                  # setup instructions, vendored-cubecl warning
└── src/
    ├── lib.rs                 # #[pymodule] root, re-exports submodules
    ├── error.rs               # BridgeError -> PyErr conversion
    ├── config.rs              # parameter_bounds(), MlpConfigSection→MlpConfig
    ├── denormalize.rs         # denormalize() pyfunction
    ├── mlp.rs                 # PyMlp class, load_mlp(), forward()
    └── conus.rs               # run_inference_over_conus()
tests/
└── smoke.py                   # uv-run pytest end-to-end
```

`lib.rs` is a 30-line `#[pymodule]` that re-exports from the focused submodules. Each submodule has one responsibility — easier for the implementer to hold one in context at a time, and easier to test in isolation.

`run_forward` (full MC, stretch goal in the spec) is **deferred** out of 10a. The `Autodiff<I>` wrapping on `MuskingumCunge::forward` makes it non-trivial and 10a's three other public functions already unblock the "weights across a map" workflow. Tracked as a 10b follow-up.

---

## Task 0: Scaffolding — empty crate that builds

**Files:**
- Create: `ddrs-py/Cargo.toml`
- Create: `ddrs-py/pyproject.toml`
- Create: `ddrs-py/src/lib.rs`
- Create: `ddrs-py/.gitignore`

- [ ] **Step 1: Create the `.gitignore`**

```bash
mkdir -p /home/tbindas/projects/ddrs/ddrs-py/src
mkdir -p /home/tbindas/projects/ddrs/ddrs-py/tests
```

Write `/home/tbindas/projects/ddrs/ddrs-py/.gitignore`:

```gitignore
target/
*.egg-info/
__pycache__/
.venv/
.pytest_cache/
```

- [ ] **Step 2: Write `Cargo.toml`**

Write `/home/tbindas/projects/ddrs/ddrs-py/Cargo.toml`:

```toml
[package]
name = "ddrs-py"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
description = "Python bindings for ddrs (read-only inference + parameter export)."

[lib]
name = "ddrs_py"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py39"] }
numpy = "0.22"
ddrs = { path = ".." }
burn = { version = "0.21", default-features = false, features = ["std", "ndarray"] }
burn-ndarray = { version = "0.21", default-features = false, features = ["std"] }
serde_yaml = "0.9"
thiserror = "1"

# Match ddrs's vendored fork patches so cargo resolves a single set of burn-* crates.
[patch.crates-io]
cubecl         = { path = "/home/tbindas/projects/cubecl/crates/cubecl" }
cubecl-cuda    = { path = "/home/tbindas/projects/cubecl/crates/cubecl-cuda" }
cubecl-common  = { path = "/home/tbindas/projects/cubecl/crates/cubecl-common" }
cubecl-core    = { path = "/home/tbindas/projects/cubecl/crates/cubecl-core" }
cubecl-cpp     = { path = "/home/tbindas/projects/cubecl/crates/cubecl-cpp" }
cubecl-ir      = { path = "/home/tbindas/projects/cubecl/crates/cubecl-ir" }
cubecl-macros  = { path = "/home/tbindas/projects/cubecl/crates/cubecl-macros" }
cubecl-opt     = { path = "/home/tbindas/projects/cubecl/crates/cubecl-opt" }
cubecl-runtime = { path = "/home/tbindas/projects/cubecl/crates/cubecl-runtime" }
cubecl-std     = { path = "/home/tbindas/projects/cubecl/crates/cubecl-std" }
cubecl-zspace  = { path = "/home/tbindas/projects/cubecl/crates/cubecl-zspace" }
burn-cubecl    = { path = "/home/tbindas/projects/burn/crates/burn-cubecl" }
burn-autodiff  = { path = "/home/tbindas/projects/burn/crates/burn-autodiff" }
burn-backend   = { path = "/home/tbindas/projects/burn/crates/burn-backend" }
burn-core      = { path = "/home/tbindas/projects/burn/crates/burn-core" }
burn-cuda      = { path = "/home/tbindas/projects/burn/crates/burn-cuda" }
burn-derive    = { path = "/home/tbindas/projects/burn/crates/burn-derive" }
burn-ir        = { path = "/home/tbindas/projects/burn/crates/burn-ir" }
burn-ndarray   = { path = "/home/tbindas/projects/burn/crates/burn-ndarray" }
burn-nn        = { path = "/home/tbindas/projects/burn/crates/burn-nn" }
burn-optim     = { path = "/home/tbindas/projects/burn/crates/burn-optim" }
burn-std       = { path = "/home/tbindas/projects/burn/crates/burn-std" }
burn-tensor    = { path = "/home/tbindas/projects/burn/crates/burn-tensor" }
burn-fusion    = { path = "/home/tbindas/projects/burn/crates/burn-fusion" }

[profile.release]
opt-level = 3
lto = "thin"
```

- [ ] **Step 3: Write `pyproject.toml`**

Write `/home/tbindas/projects/ddrs/ddrs-py/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "ddrs-py"
version = "0.1.0"
description = "Python bindings for ddrs (read-only inference + parameter export)."
requires-python = ">=3.9"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
test = ["pytest>=8"]

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "ddrs_py"
```

- [ ] **Step 4: Write the `lib.rs` stub**

Write `/home/tbindas/projects/ddrs/ddrs-py/src/lib.rs`:

```rust
//! ddrs-py: PyO3 bindings for ddrs.
//!
//! See `.claude/specs/2026-05-26-sp10a-pyo3-bridge-design.md` for design.

use pyo3::prelude::*;

#[pymodule]
fn ddrs_py(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
```

- [ ] **Step 5: Build to verify scaffolding compiles**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
cargo build --release 2>&1 | tail -20
```

Expected: `Compiling ddrs-py v0.1.0` then `Finished release [optimized]`. No warnings about the empty `#[pymodule]`. If cargo complains about the patches, double-check the paths in `[patch.crates-io]` match `ddrs/Cargo.toml` exactly.

- [ ] **Step 6: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add ddrs-py/
git commit -m "SP-10a Task 0: scaffold ddrs-py crate (empty pymodule, builds clean)"
```

---

## Task 1: `parameter_bounds(config_path)` — pure config read

This is the simplest function. No BURN tensors cross the FFI boundary — it's just reading YAML and returning a dict. Doing it first gives us a working Python ↔ Rust roundtrip to build on.

**Files:**
- Create: `ddrs-py/src/error.rs`
- Create: `ddrs-py/src/config.rs`
- Modify: `ddrs-py/src/lib.rs`
- Modify: `ddrs-py/tests/smoke.py` (created here for the first test)

- [ ] **Step 1: Write the failing Python test**

Write `/home/tbindas/projects/ddrs/ddrs-py/tests/smoke.py`:

```python
"""End-to-end smoke test for the ddrs_py bridge.

Run from the ddrs-py directory with a venv that has ddrs_py installed:
    uv run pytest tests/smoke.py -v
"""

import pytest
import ddrs_py


REPO_ROOT = "/home/tbindas/projects/ddrs"
CONFIG_PATH = f"{REPO_ROOT}/config/merit_training.yaml"


def test_parameter_bounds_from_merit_training_yaml():
    bounds = ddrs_py.parameter_bounds(CONFIG_PATH)
    assert set(bounds.keys()) == {"n", "q_spatial", "p_spatial"}

    n_bounds, n_log = bounds["n"]
    assert n_bounds == (pytest.approx(0.015), pytest.approx(0.25))
    assert n_log is True  # log_space_parameters lists "n" in this yaml

    q_bounds, q_log = bounds["q_spatial"]
    assert q_bounds == (pytest.approx(0.0), pytest.approx(1.0))
    assert q_log is False

    p_bounds, p_log = bounds["p_spatial"]
    assert p_bounds == (pytest.approx(1.0), pytest.approx(200.0))
    assert p_log is False
```

- [ ] **Step 2: Set up the test venv and verify the test fails**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv venv
uv pip install maturin pytest numpy
uv run maturin develop --release 2>&1 | tail -5
uv run pytest tests/smoke.py -v 2>&1 | tail -10
```

Expected: `AttributeError: module 'ddrs_py' has no attribute 'parameter_bounds'`.

- [ ] **Step 3: Write `src/error.rs`**

Write `/home/tbindas/projects/ddrs/ddrs-py/src/error.rs`:

```rust
//! Bridge error type. Converts ddrs + serde + io errors into PyErr.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("config load failed at {path:?}: {source}")]
    Config {
        path: String,
        #[source]
        source: ddrs::data::error::DataError,
    },
    #[error("config is missing the `mlp:` section ({path:?})")]
    MissingMlpSection { path: String },
    #[error("attrs shape ({rows}, {cols}) mismatches mlp.input_var_names.len() = {expected_cols}")]
    AttrShapeMismatch { rows: usize, cols: usize, expected_cols: usize },
    #[error("checkpoint load failed at {path:?}: {source}")]
    Checkpoint {
        path: String,
        #[source]
        source: ddrs::data::error::DataError,
    },
    #[error("netcdf attribute read failed: {0}")]
    Netcdf(#[source] ddrs::data::error::DataError),
    #[error("zarr adjacency read failed: {0}")]
    Zarr(#[source] ddrs::data::error::DataError),
}

impl From<BridgeError> for PyErr {
    fn from(e: BridgeError) -> Self {
        match e {
            BridgeError::Config { .. }
            | BridgeError::Checkpoint { .. }
            | BridgeError::Netcdf(_)
            | BridgeError::Zarr(_) => PyIOError::new_err(e.to_string()),
            BridgeError::MissingMlpSection { .. } | BridgeError::AttrShapeMismatch { .. } => {
                PyValueError::new_err(e.to_string())
            }
        }
    }
}
```

- [ ] **Step 4: Write `src/config.rs`**

Write `/home/tbindas/projects/ddrs/ddrs-py/src/config.rs`:

```rust
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
        path: path.to_string(),
        source,
    })
}

/// Pull `cfg.mlp` or return a typed error if absent.
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
```

- [ ] **Step 5: Wire into `lib.rs`**

Replace `/home/tbindas/projects/ddrs/ddrs-py/src/lib.rs`:

```rust
//! ddrs-py: PyO3 bindings for ddrs.
//!
//! See `.claude/specs/2026-05-26-sp10a-pyo3-bridge-design.md` for design.

use pyo3::prelude::*;

mod config;
mod error;

#[pymodule]
fn ddrs_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(config::parameter_bounds, m)?)?;
    Ok(())
}
```

- [ ] **Step 6: Rebuild and run the test**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run maturin develop --release 2>&1 | tail -5
uv run pytest tests/smoke.py::test_parameter_bounds_from_merit_training_yaml -v 2>&1 | tail -10
```

Expected: `1 passed`.

- [ ] **Step 7: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add ddrs-py/src/error.rs ddrs-py/src/config.rs ddrs-py/src/lib.rs ddrs-py/tests/smoke.py
git commit -m "SP-10a Task 1: add parameter_bounds() + bridge error type"
```

---

## Task 2: `denormalize(values, bounds, log_space)` — first numpy round-trip

Same `denormalize` math as `routing::utils::denormalize`. The point of doing it as a separate function (instead of just calling Rust from inside `forward`) is to give notebooks a way to apply the bounds to any array, including ones that didn't come from the bridge.

**Files:**
- Create: `ddrs-py/src/denormalize.rs`
- Modify: `ddrs-py/src/lib.rs`
- Modify: `ddrs-py/tests/smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/tbindas/projects/ddrs/ddrs-py/tests/smoke.py`:

```python
import numpy as np


def test_denormalize_linear_matches_formula():
    # bounds (1.0, 200.0), log_space=False  →  v · (200 - 1) + 1
    values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    result = ddrs_py.denormalize(values, (1.0, 200.0), False)
    expected = np.array([1.0, 100.5, 200.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_denormalize_log_matches_formula():
    # bounds (0.015, 0.25), log_space=True  →  exp(v · (ln(0.25) - ln(0.015+1e-6)) + ln(0.015+1e-6))
    values = np.array([0.0, 1.0], dtype=np.float32)
    result = ddrs_py.denormalize(values, (0.015, 0.25), True)
    lo_eff = np.log(0.015 + 1e-6)
    expected = np.array([np.exp(lo_eff), 0.25], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_denormalize_rejects_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        ddrs_py.denormalize(np.zeros((2, 2), dtype=np.float32), (0.0, 1.0), False)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run pytest tests/smoke.py -v -k denormalize 2>&1 | tail -10
```

Expected: `AttributeError: module 'ddrs_py' has no attribute 'denormalize'`.

- [ ] **Step 3: Implement `denormalize.rs`**

Write `/home/tbindas/projects/ddrs/ddrs-py/src/denormalize.rs`:

```rust
//! Python entry for `routing::utils::denormalize`.
//!
//! Re-implemented as a pure ndarray op (rather than routing through a BURN
//! tensor) because the input is already on the host as a numpy array — the
//! BURN round-trip would just be allocation churn.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn denormalize<'py>(
    py: Python<'py>,
    values: PyReadonlyArrayDyn<'py, f32>,
    bounds: (f32, f32),
    log_space: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if values.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "denormalize expects a 1-D array, got shape {:?}",
            values.shape()
        )));
    }
    let (lo, hi) = bounds;
    let input = values.as_slice()?;
    let out: Vec<f32> = if log_space {
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
```

- [ ] **Step 4: Wire into `lib.rs`**

Edit `/home/tbindas/projects/ddrs/ddrs-py/src/lib.rs` — add `mod denormalize;` and register the function:

```rust
mod config;
mod denormalize;
mod error;

#[pymodule]
fn ddrs_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(config::parameter_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(denormalize::denormalize, m)?)?;
    Ok(())
}
```

- [ ] **Step 5: Rebuild and run**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run maturin develop --release 2>&1 | tail -5
uv run pytest tests/smoke.py -v 2>&1 | tail -15
```

Expected: 4 passed (the new 3 + the original 1).

- [ ] **Step 6: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add ddrs-py/src/denormalize.rs ddrs-py/src/lib.rs ddrs-py/tests/smoke.py
git commit -m "SP-10a Task 2: add denormalize() with 1-D shape check"
```

---

## Task 3: `PyMlp` + `load_mlp(checkpoint, config_path)`

This is the first task that touches BURN. We build an `Mlp<NdArray>` template from the YAML, fill it from the `.mpk` file, and store it inside a `PyMlp` pyclass. No tensors cross the FFI boundary yet — `forward` lands in Task 4.

The test needs a real checkpoint. We generate one in-process by training one mini-batch — but that's expensive. Cheaper path: there are already checkpoints in `output/saved_models/` from a previous run (`epoch_1_mb_35.mpk` exists). Use one of those. If a future caller starts fresh and `output/saved_models/` is empty, the test skips with a clear message.

**Files:**
- Create: `ddrs-py/src/mlp.rs`
- Modify: `ddrs-py/src/lib.rs`
- Modify: `ddrs-py/tests/smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/tbindas/projects/ddrs/ddrs-py/tests/smoke.py`:

```python
import os

CHECKPOINT_DIR = f"{REPO_ROOT}/output/saved_models"


def _first_available_checkpoint() -> str | None:
    """Return base path (no .mpk extension) of any checkpoint in the saved_models dir."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    mpks = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".mpk"))
    if not mpks:
        return None
    return f"{CHECKPOINT_DIR}/{mpks[0][:-len('.mpk')]}"


def test_load_mlp_returns_pymlp_with_param_names():
    ckpt = _first_available_checkpoint()
    if ckpt is None:
        pytest.skip(f"no checkpoints in {CHECKPOINT_DIR}; train one first")

    model = ddrs_py.load_mlp(checkpoint=ckpt, config_path=CONFIG_PATH)
    assert model.learnable_parameters == ["n", "q_spatial", "p_spatial"]
    assert model.input_var_names_len == 10  # matches merit_training.yaml
```

- [ ] **Step 2: Verify the test fails**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run pytest tests/smoke.py -v -k load_mlp 2>&1 | tail -10
```

Expected: `AttributeError: module 'ddrs_py' has no attribute 'load_mlp'`.

- [ ] **Step 3: Implement `mlp.rs`**

Write `/home/tbindas/projects/ddrs/ddrs-py/src/mlp.rs`:

```rust
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

    /// Number of input attribute columns this MLP expects. The bridge
    /// doesn't expose the names (use `parameter_bounds`/config for that)
    /// because the only thing the forward call needs is the length.
    #[getter]
    fn input_var_names_len(&self) -> usize {
        // Inferred from the first Linear layer's weight rows. We don't
        // store the original names on the BURN module.
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
            path: checkpoint.to_string(),
            source,
        },
    )?;
    Ok(PyMlp { inner, device })
}
```

- [ ] **Step 4: Wire into `lib.rs`**

Edit `/home/tbindas/projects/ddrs/ddrs-py/src/lib.rs`:

```rust
mod config;
mod denormalize;
mod error;
mod mlp;

#[pymodule]
fn ddrs_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(config::parameter_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(denormalize::denormalize, m)?)?;
    m.add_function(wrap_pyfunction!(mlp::load_mlp, m)?)?;
    m.add_class::<mlp::PyMlp>()?;
    Ok(())
}
```

- [ ] **Step 5: Rebuild and run**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run maturin develop --release 2>&1 | tail -5
uv run pytest tests/smoke.py -v -k load_mlp 2>&1 | tail -10
```

Expected: PASS. If it fails with "input.weight dims" indexing, the `weight` accessor on BURN `Linear` may need `.val().shape().dims` instead — adjust to the working form, then commit.

- [ ] **Step 6: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add ddrs-py/src/mlp.rs ddrs-py/src/lib.rs ddrs-py/tests/smoke.py
git commit -m "SP-10a Task 3: add PyMlp + load_mlp() on NdArray backend"
```

---

## Task 4: `PyMlp.forward(attrs)` — numpy ↔ BURN tensor round-trip

The core inference path. `attrs: np.ndarray[float32, (R, F)]` → BURN `Tensor<NdArray, 2>` → `Mlp::forward` → `HashMap<String, Tensor<NdArray, 1>>` → `dict[str, np.ndarray]`.

**Files:**
- Modify: `ddrs-py/src/mlp.rs`
- Modify: `ddrs-py/tests/smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/tbindas/projects/ddrs/ddrs-py/tests/smoke.py`:

```python
def test_forward_returns_param_dict_in_unit_interval():
    ckpt = _first_available_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint")
    model = ddrs_py.load_mlp(checkpoint=ckpt, config_path=CONFIG_PATH)

    rng = np.random.default_rng(seed=0)
    attrs = rng.standard_normal((7, model.input_var_names_len)).astype(np.float32)
    out = model.forward(attrs)

    assert set(out.keys()) == {"n", "q_spatial", "p_spatial"}
    for key, arr in out.items():
        assert arr.dtype == np.float32, f"{key} dtype is {arr.dtype}"
        assert arr.shape == (7,), f"{key} shape is {arr.shape}"
        assert np.all((arr >= 0.0) & (arr <= 1.0)), f"{key} values out of [0,1]: min={arr.min()} max={arr.max()}"


def test_forward_rejects_wrong_feature_count():
    ckpt = _first_available_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint")
    model = ddrs_py.load_mlp(checkpoint=ckpt, config_path=CONFIG_PATH)
    bad = np.zeros((4, 99), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatches"):
        model.forward(bad)
```

- [ ] **Step 2: Verify failure**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run pytest tests/smoke.py -v -k forward 2>&1 | tail -10
```

Expected: `AttributeError: 'builtins.PyMlp' object has no attribute 'forward'`.

- [ ] **Step 3: Add `forward` to `PyMlp`**

Edit `/home/tbindas/projects/ddrs/ddrs-py/src/mlp.rs` — add these imports at the top of the file:

```rust
use burn::tensor::{Tensor, TensorData};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::types::PyDict;
```

And add a new method inside the existing `#[pymethods] impl PyMlp { ... }` block, after `input_var_names_len`:

```rust
    /// Run inference on a `(R, F)` `float32` attrs batch.
    ///
    /// Returns a dict keyed by `learnable_parameters`; each value is a 1-D
    /// `float32` numpy array of length R, with values in `[0, 1]`.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        attrs: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dims = attrs.shape();
        let rows = dims[0];
        let cols = dims[1];
        let expected_cols = self.inner.input.weight.val().dims()[0];
        if cols != expected_cols {
            return Err(BridgeError::AttrShapeMismatch {
                rows,
                cols,
                expected_cols,
            }
            .into());
        }

        // Numpy → BURN tensor. attrs is C-contiguous via numpy's default;
        // as_slice() returns the underlying f32 buffer in row-major order,
        // which matches BURN's [rows, cols] layout.
        let slice: &[f32] = attrs.as_slice()?;
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
```

Also add a helper to silence the now-unused `PyArray2` import if you didn't end up using it — easiest is to drop `PyArray2` from the `numpy::` import line above. Keep only `PyArray1`, `PyArrayMethods`, `PyReadonlyArray2`.

- [ ] **Step 4: Rebuild and run**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run maturin develop --release 2>&1 | tail -5
uv run pytest tests/smoke.py -v 2>&1 | tail -20
```

Expected: all 6 tests pass (4 prior + the 2 new forward tests).

- [ ] **Step 5: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add ddrs-py/src/mlp.rs ddrs-py/tests/smoke.py
git commit -m "SP-10a Task 4: PyMlp.forward() with shape validation"
```

---

## Task 5: `run_inference_over_conus(...)` — the CONUS parameter-map workflow

This is the function that unlocks "interpret weights across a map". It loads the full MERIT attribute matrix for every COMID in the CONUS adjacency, runs `PyMlp.forward` over it, denormalizes, and returns one row per COMID plus the COMID array.

**Files:**
- Create: `ddrs-py/src/conus.rs`
- Modify: `ddrs-py/src/lib.rs`
- Modify: `ddrs-py/src/mlp.rs` (expose a private helper, see Step 3)
- Modify: `ddrs-py/tests/smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `/home/tbindas/projects/ddrs/ddrs-py/tests/smoke.py`:

```python
DDR_DATA = "/home/tbindas/projects/ddr/data"
ATTRS_NC = f"{DDR_DATA}/merit_global_attributes_v2.nc"
CONUS_ZARR = f"{DDR_DATA}/merit_conus_adjacency.zarr"


def test_run_inference_over_conus_returns_per_comid_params():
    ckpt = _first_available_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint")
    if not os.path.exists(ATTRS_NC) or not os.path.exists(CONUS_ZARR):
        pytest.skip(f"MERIT data not present at {DDR_DATA}")

    result = ddrs_py.run_inference_over_conus(
        attrs_nc=ATTRS_NC,
        conus_adjacency_zarr=CONUS_ZARR,
        checkpoint=ckpt,
        config_path=CONFIG_PATH,
    )

    assert set(result.keys()) == {"comid", "n", "q_spatial", "p_spatial"}
    n_reaches = result["comid"].shape[0]
    assert n_reaches > 100_000, f"expected ~346k MERIT reaches, got {n_reaches}"

    # COMID column is int64.
    assert result["comid"].dtype == np.int64

    # Parameter columns are float32, same length as comid, physical-units range.
    for key in ("n", "q_spatial", "p_spatial"):
        arr = result[key]
        assert arr.dtype == np.float32
        assert arr.shape == (n_reaches,)
        assert np.all(np.isfinite(arr)), f"{key} has non-finite values"

    # Physical bounds sanity per merit_training.yaml's parameter_ranges:
    #   n in [0.015, 0.25] (log space → exp() output strictly positive)
    #   q_spatial in [0.0, 1.0]
    #   p_spatial in [1.0, 200.0]
    assert result["n"].min() > 0.0 and result["n"].max() <= 0.25 + 1e-3
    assert result["q_spatial"].min() >= 0.0 and result["q_spatial"].max() <= 1.0 + 1e-5
    assert result["p_spatial"].min() >= 1.0 - 1e-3 and result["p_spatial"].max() <= 200.0 + 1e-3
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run pytest tests/smoke.py -v -k run_inference_over_conus 2>&1 | tail -10
```

Expected: `AttributeError: module 'ddrs_py' has no attribute 'run_inference_over_conus'`.

- [ ] **Step 3: Expose an internal forward helper on `PyMlp`**

The `conus.rs` module needs to call `Mlp::forward` on a BURN tensor directly (no numpy round-trip). Add this helper inside the existing `impl PyMlp { ... }` block in `ddrs-py/src/mlp.rs` — but OUTSIDE the `#[pymethods]` block. Add the block at the end of the file:

```rust
// Internal helpers used by sibling modules; NOT exposed to Python.
impl PyMlp {
    pub(crate) fn run(&self, input: Tensor<Backend, 2>) -> std::collections::HashMap<String, Tensor<Backend, 1>> {
        self.inner.forward(input)
    }

    pub(crate) fn param_order(&self) -> &[String] {
        self.inner.learnable_parameters()
    }
}
```

Also make the `Backend` type alias `pub(crate)` so `conus.rs` can use it. Change the existing line in `mlp.rs`:

```rust
type Backend = NdArray<f32>;
```

to:

```rust
pub(crate) type Backend = NdArray<f32>;
```

- [ ] **Step 4: Implement `conus.rs`**

Write `/home/tbindas/projects/ddrs/ddrs-py/src/conus.rs`:

```rust
//! CONUS-wide inference: load every MERIT COMID's attributes, run the MLP,
//! denormalize. Mirrors the workflow in DDR's `merit_geometry_config.yaml`.

use std::collections::HashSet;
use std::path::Path;

use burn::tensor::{Tensor, TensorData};
use ddrs::config::Params;
use ddrs::data::store::netcdf::AttributesStore;
use ddrs::data::store::zarr::ConusAdjacencyStore;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::config::{load_config, require_mlp_section};
use crate::error::BridgeError;
use crate::mlp::{Backend, PyMlp};

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

    // 1. Adjacency → ordered COMID list.
    let adj = ConusAdjacencyStore::open(Path::new(conus_adjacency_zarr))
        .map_err(BridgeError::Zarr)?;

    // 2. Attributes → (F, N) matrix aligned to a subset of COMIDs that the
    //    netcdf actually contains. May be shorter than adj.order if the
    //    netcdf is missing rows; AttributesStore handles that internally.
    let attrs_store = AttributesStore::open(Path::new(attrs_nc), &attr_names, &adj.order)
        .map_err(BridgeError::Netcdf)?;

    // 3. Pull the resolved COMID list out of attrs_store.index. We need
    //    to keep its order because attrs[:, i] aligns to position i in
    //    that list, not the adjacency's order.
    let resolved_comids: Vec<i64> = attrs_store.index.ids().iter().map(|c| c.0).collect();
    let n_reaches = resolved_comids.len();
    let f = attr_names.len();
    debug_assert_eq!(attrs_store.attrs.dim(), (f, n_reaches));

    // 4. Reload the MLP. (We do this here rather than accepting a PyMlp
    //    parameter so the caller has a single-call API.)
    let model = crate::mlp::load_mlp(checkpoint, config_path)?;

    // 5. Build the BURN input tensor. AttributesStore stores attrs as
    //    (F, N); MLP wants (N, F). Transpose into a Vec<f32> in row-major.
    let mut input_buf = vec![0.0_f32; n_reaches * f];
    for row in 0..n_reaches {
        for col in 0..f {
            input_buf[row * f + col] = attrs_store.attrs[(col, row)];
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
        let raw_vec: Vec<f32> = raw_t.clone().into_data().to_vec().map_err(|e| {
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
/// Kept private because callers should go through `crate::denormalize` for
/// the public path.
fn denormalize_vec(values: &[f32], bounds: [f32; 2], log_space: bool) -> Vec<f32> {
    let [lo, hi] = bounds;
    if log_space {
        let log_min = (lo + 1e-6_f32).ln();
        let log_max = hi.ln();
        let scale = log_max - log_min;
        values.iter().map(|&v| (v * scale + log_min).exp()).collect()
    } else {
        let scale = hi - lo;
        values.iter().map(|&v| v * scale + lo).collect()
    }
}
```

Note: `AttributesStore.index` exposes the COMID list via `IdIndex::ids()`. If that method has a different name in the current code (e.g. `as_slice` or a public field), check `src/data/ids.rs` and adjust the call. The intent is: get the COMIDs in the same order they appear as columns of `attrs_store.attrs`.

- [ ] **Step 5: Verify `IdIndex::ids()` exists, or adjust**

```bash
cd /home/tbindas/projects/ddrs
grep -n "pub fn\|pub struct\|pub.*Vec<T>" src/data/ids.rs
```

If there's no `ids()` accessor, add one or use the existing public field. If `IdIndex` has a `pub fn iter` or holds a `pub order: Vec<T>`, use that instead. Update the line in `conus.rs`:

```rust
let resolved_comids: Vec<i64> = attrs_store.index.ids().iter().map(|c| c.0).collect();
```

to whatever the actual API requires (likely `.order` or `.iter()`).

- [ ] **Step 6: Wire into `lib.rs`**

Edit `/home/tbindas/projects/ddrs/ddrs-py/src/lib.rs`:

```rust
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
```

- [ ] **Step 7: Rebuild and run the full suite**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run maturin develop --release 2>&1 | tail -5
uv run pytest tests/smoke.py -v 2>&1 | tail -20
```

Expected: 7 passed (1 + 3 + 1 + 1 + 1).

- [ ] **Step 8: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add ddrs-py/src/conus.rs ddrs-py/src/mlp.rs ddrs-py/src/lib.rs ddrs-py/tests/smoke.py
git commit -m "SP-10a Task 5: run_inference_over_conus() — CONUS parameter map workflow"
```

---

## Task 6: README + regression-test guard

Document the setup and confirm we have NOT broken anything in the main `ddrs` crate.

**Files:**
- Create: `ddrs-py/README.md`

- [ ] **Step 1: Write `README.md`**

Write `/home/tbindas/projects/ddrs/ddrs-py/README.md`:

````markdown
# ddrs-py

Python bindings for the [ddrs](../) Muskingum-Cunge routing crate. Read-only
inference path on CPU (`NdArray` backend). Intended for notebook-based
interpretation, not training.

## Setup

Requires the same vendored `cubecl` + `burn` fork clones as the main `ddrs`
crate — see `[patch.crates-io]` in `Cargo.toml`. If those paths don't exist
on your machine, the build will fail with "no matching package found".

```bash
cd ddrs-py
uv venv
uv pip install maturin pytest numpy
uv run maturin develop --release
uv run pytest tests/smoke.py -v
```

`--release` is required — debug builds of PyO3 + BURN are unusably slow for
any real-sized attribute batch.

## API

```python
import ddrs_py
import numpy as np

# 1. Inspect parameter bounds defined by the training YAML.
bounds = ddrs_py.parameter_bounds("config/merit_training.yaml")
# {"n": ((0.015, 0.25), True), "q_spatial": ((0.0, 1.0), False), ...}

# 2. Denormalize MLP outputs to physical units.
physical = ddrs_py.denormalize(
    np.array([0.0, 0.5, 1.0], dtype=np.float32),
    bounds=(1.0, 200.0),
    log_space=False,
)

# 3. Load a trained MLP checkpoint.
model = ddrs_py.load_mlp(
    checkpoint="output/saved_models/epoch_1_mb_35",   # NO .mpk suffix
    config_path="config/merit_training.yaml",
)

# 4. Inference on an arbitrary (R, F=10) attrs batch.
out = model.forward(attrs)
# {"n": np.ndarray (R,), "q_spatial": ..., "p_spatial": ...}

# 5. Full CONUS parameter map — drives the SP-10c choropleth.
result = ddrs_py.run_inference_over_conus(
    attrs_nc="~/projects/ddr/data/merit_global_attributes_v2.nc",
    conus_adjacency_zarr="~/projects/ddr/data/merit_conus_adjacency.zarr",
    checkpoint="output/saved_models/epoch_1_mb_35",
    config_path="config/merit_training.yaml",
)
# {"comid": (N,) int64, "n": (N,) float32, ...}
```

## Out of scope for 10a

- GPU inference (use the `eval` binary in the parent crate for that).
- Full Muskingum-Cunge forward pass from Python — coming in 10b.
- DDR `.pt` checkpoint loading — never supported (different runtime).
````

- [ ] **Step 2: Verify nothing in `ddrs` regressed**

```bash
cd /home/tbindas/projects/ddrs
cargo build --release 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: build clean; final line of `compare_ddr_sandbox` reports `ABSOLUTE MATCH`. If it doesn't, something in the bridge accidentally touched the routing core — investigate (should NOT be possible since we made no edits to `ddrs/src/`, but verify before committing).

- [ ] **Step 3: Run the full Python smoke suite one last time**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
uv run pytest tests/smoke.py -v 2>&1 | tail -15
```

Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add ddrs-py/README.md
git commit -m "SP-10a Task 6: README + verify ddrs regression suite still green"
```

---

## Self-Review

**Spec coverage:**
- Goal (load checkpoint + run MLP on CPU from Python) → Tasks 3, 4 ✓
- `parameter_bounds` → Task 1 ✓
- `denormalize` → Task 2 ✓
- `load_mlp` → Task 3 ✓
- `PyMlp.forward` → Task 4 ✓
- `run_inference_over_conus` → Task 5 ✓
- `run_forward` → explicitly deferred (called out in File Structure section) ✓
- Maturin/uv workflow → Task 0 + README ✓
- Vendored cubecl/burn forks documented → Task 0 (Cargo.toml comment) + README ✓
- No changes to `ddrs` crate → confirmed in Task 6 regression check ✓
- Smoke test green on real checkpoint → Task 5 ✓

**Placeholder scan:** every step shows the actual file content, command, or expected output. No TODOs. The one "if API differs, adjust" note in Task 5 Step 5 is a pre-empted real concern (`IdIndex::ids()` was inferred, not verified) and includes the grep + concrete remediation, not a vague directive.

**Type consistency:** `PyMlp` defined Task 3, used in Tasks 4 + 5. `Backend = NdArray<f32>` defined Task 3, promoted to `pub(crate)` and used in Task 5. `load_config`, `require_mlp_section`, `mlp_config_from_section`: defined Task 1, reused Tasks 3 + 5. `BridgeError` variants defined Task 1, reused Tasks 3 + 5. `denormalize_vec` in `conus.rs` Task 5 is a private mirror of the public `denormalize` (called out in a doc comment so the duplication is intentional, not a leak).

One small consistency note worth flagging to the implementer: the bounds tuples returned by `parameter_bounds` are `(float, float)` (numpy-friendly), but the bounds taken by `denormalize` are also a 2-tuple — they're API-compatible. The test in Task 2 passes `(1.0, 200.0)` directly; passing `bounds["p_spatial"][0]` from a `parameter_bounds` result would also work.

---

**Plan complete and saved to `.claude/specs/2026-05-26-sp10a-pyo3-bridge-plan.md`.**
