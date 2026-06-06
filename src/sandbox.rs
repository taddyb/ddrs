//! Sandbox fixture loader + functional smoke test.
//!
//! 5-reach RAPID sandbox at `fixtures/sandbox/*.csv`. Used by:
//!   - `examples/compare_ddr_sandbox.rs` (DDR-parity regression)
//!   - `cli::system` smoke test (does routing work on this machine?)

use std::path::Path;

use burn::backend::Autodiff;
use burn::tensor::{backend::Backend, Tensor};

use crate::error::CliError;
use crate::config::Config;
use crate::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use crate::sparse::SparseAdjacency;

/// Number of reaches in the RAPID sandbox.
pub const N_REACHES: usize = 5;

/// Loaded sandbox inputs ready to feed into `MuskingumCunge`.
pub struct SandboxInputs {
    /// Lateral inflow in topological order, flattened `[n_timesteps, N_REACHES]`.
    pub qprime_flat: Vec<f32>,
    /// Dense adjacency matrix, lower-triangular, flattened `[N_REACHES, N_REACHES]`.
    pub adjacency_flat: Vec<f32>,
    /// Reach IDs in topological order (length `N_REACHES`).
    pub topo_order: Vec<i32>,
    /// Reach IDs in RAPID2 order (length `N_REACHES`).
    pub rapid2_order: Vec<i32>,
    /// Number of timesteps in `qprime_flat`.
    pub n_timesteps: usize,
    /// Routing config built from `config.csv`.
    pub config: Config,
}

// ---------------------------------------------------------------------------
// Private parse helpers
// ---------------------------------------------------------------------------

fn parse_matrix_csv(src: &str, path_hint: &str, expect_rows: usize, expect_cols: usize) -> Result<Vec<f32>, CliError> {
    let mut data = Vec::with_capacity(expect_rows * expect_cols);
    let mut rows = 0usize;
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<f32> = line
            .split(',')
            .map(|x| {
                x.trim()
                    .parse::<f32>()
                    .map_err(|e| CliError::Runtime(format!("{path_hint}: {e}")))
            })
            .collect::<Result<_, _>>()?;
        if cols.len() != expect_cols {
            return Err(CliError::Runtime(format!(
                "{path_hint}: expected {expect_cols} cols, got {}",
                cols.len()
            )));
        }
        data.extend(cols);
        rows += 1;
    }
    if rows != expect_rows {
        return Err(CliError::Runtime(format!(
            "{path_hint}: expected {expect_rows} rows, got {rows}"
        )));
    }
    Ok(data)
}

fn parse_int_csv(src: &str, path_hint: &str) -> Result<Vec<i32>, CliError> {
    src.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            l.parse::<i32>()
                .map_err(|e| CliError::Runtime(format!("{path_hint}: {e}")))
        })
        .collect()
}

/// Parse `config.csv` key-value rows into a `Config`.
///
/// Only the fields that `ddr_config()` sets are honoured; everything else
/// stays at `Config::default()`.
fn parse_config_csv(src: &str) -> Result<Config, CliError> {
    let mut cfg = Config::default();
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "range_n" if parts.len() == 3 => {
                let lo: f32 = parts[1].trim().parse().map_err(|e| CliError::Runtime(format!("config.csv range_n lo: {e}")))?;
                let hi: f32 = parts[2].trim().parse().map_err(|e| CliError::Runtime(format!("config.csv range_n hi: {e}")))?;
                cfg.params.parameter_ranges.n = [lo, hi];
            }
            "range_q_spatial" if parts.len() == 3 => {
                let lo: f32 = parts[1].trim().parse().map_err(|e| CliError::Runtime(format!("config.csv range_q_spatial lo: {e}")))?;
                let hi: f32 = parts[2].trim().parse().map_err(|e| CliError::Runtime(format!("config.csv range_q_spatial hi: {e}")))?;
                cfg.params.parameter_ranges.q_spatial = [lo, hi];
            }
            "log_space_parameters" if parts.len() >= 2 => {
                cfg.params.log_space_parameters = parts[1..]
                    .iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "p_spatial_default" if parts.len() >= 2 => {
                let v: f32 = parts[1].trim().parse().map_err(|e| CliError::Runtime(format!("config.csv p_spatial_default: {e}")))?;
                cfg.params.defaults.insert("p_spatial".to_string(), v);
            }
            // n_reaches, n_timesteps, length_m, slope, x_storage, dt_seconds
            // are embedded in SandboxInputs fields; no need to set on Config.
            _ => {}
        }
    }
    // Hardcoded minimums matching the sandbox (same as ddr_config() in the example).
    cfg.params.attribute_minimums.discharge = 1e-4;
    cfg.params.attribute_minimums.slope = 1e-3;
    cfg.params.attribute_minimums.velocity = 0.01;
    cfg.params.attribute_minimums.depth = 0.01;
    cfg.params.attribute_minimums.bottom_width = 0.1;
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Build SandboxInputs from parsed text slices
// ---------------------------------------------------------------------------

fn build_inputs(
    topo_src: &str,
    rapid2_src: &str,
    qprime_src: &str,
    adjacency_src: &str,
    config_src: &str,
) -> Result<SandboxInputs, CliError> {
    let topo_order = parse_int_csv(topo_src, "topo_order.csv")?;
    let rapid2_order = parse_int_csv(rapid2_src, "rapid2_order.csv")?;
    if topo_order.len() != N_REACHES {
        return Err(CliError::Runtime(format!(
            "topo_order.csv: expected {N_REACHES} entries, got {}",
            topo_order.len()
        )));
    }
    if rapid2_order.len() != N_REACHES {
        return Err(CliError::Runtime(format!(
            "rapid2_order.csv: expected {N_REACHES} entries, got {}",
            rapid2_order.len()
        )));
    }

    // qprime: (T, N) — count rows first, then validate columns.
    let qprime_rows: Vec<Vec<f32>> = qprime_src
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            l.split(',')
                .map(|x| {
                    x.trim()
                        .parse::<f32>()
                        .map_err(|e| CliError::Runtime(format!("qprime_topo.csv: {e}")))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<_, _>>()?;
    let n_timesteps = qprime_rows.len();
    for (i, row) in qprime_rows.iter().enumerate() {
        if row.len() != N_REACHES {
            return Err(CliError::Runtime(format!(
                "qprime_topo.csv row {i}: expected {N_REACHES} cols, got {}",
                row.len()
            )));
        }
    }
    let qprime_flat: Vec<f32> = qprime_rows.into_iter().flatten().collect();

    let adjacency_flat = parse_matrix_csv(adjacency_src, "adjacency_topo.csv", N_REACHES, N_REACHES)?;
    let config = parse_config_csv(config_src)?;

    Ok(SandboxInputs {
        qprime_flat,
        adjacency_flat,
        topo_order,
        rapid2_order,
        n_timesteps,
        config,
    })
}

// ---------------------------------------------------------------------------
// Public loaders
// ---------------------------------------------------------------------------

/// Load sandbox fixtures from a directory containing the CSV files.
pub fn load_from_dir(dir: &Path) -> Result<SandboxInputs, CliError> {
    let read = |name: &str| -> Result<String, CliError> {
        let p = dir.join(name);
        std::fs::read_to_string(&p).map_err(|_| CliError::DataSourceMissing { path: p })
    };

    build_inputs(
        &read("topo_order.csv")?,
        &read("rapid2_order.csv")?,
        &read("qprime_topo.csv")?,
        &read("adjacency_topo.csv")?,
        &read("config.csv")?,
    )
}

/// Load sandbox fixtures from compiled-in bytes (used by installed binaries).
pub fn load_embedded() -> Result<SandboxInputs, CliError> {
    build_inputs(
        include_str!("../fixtures/sandbox/topo_order.csv"),
        include_str!("../fixtures/sandbox/rapid2_order.csv"),
        include_str!("../fixtures/sandbox/qprime_topo.csv"),
        include_str!("../fixtures/sandbox/adjacency_topo.csv"),
        include_str!("../fixtures/sandbox/config.csv"),
    )
}

// ---------------------------------------------------------------------------
// Smoke test
// ---------------------------------------------------------------------------

/// Result of the functional smoke test.
#[derive(Debug)]
pub struct SmokeResult {
    pub passed: bool,
    pub max_q: f32,
    pub n_reaches: usize,
    pub n_nan: usize,
    pub n_negative: usize,
}

/// Run a single MC forward pass on the sandbox and check well-formedness.
///
/// `passed` is `true` iff:
/// - all output discharge values are finite (no NaN, no Inf),
/// - all output discharge values are >= 0,
/// - at least one output discharge value is > 0.
pub fn smoke<I>(
    inputs: &SandboxInputs,
    device: &I::Device,
) -> Result<SmokeResult, CliError>
where
    I: Backend,
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    type Bv<I> = Autodiff<I>;

    let qprime: Tensor<Bv<I>, 2> =
        Tensor::<Bv<I>, 1>::from_floats(inputs.qprime_flat.as_slice(), device)
            .reshape([inputs.n_timesteps, N_REACHES]);

    let adjacency = SparseAdjacency::from_dense(
        N_REACHES,
        &inputs.adjacency_flat,
        vec![5000.0; N_REACHES],
        vec![0.001; N_REACHES],
    );
    let routing_inputs = RoutingInputs::<I> {
        adjacency,
        x_storage: Tensor::ones([N_REACHES], device) * 0.25,
    };
    let params = SpatialParameters::<I> {
        n: Tensor::ones([N_REACHES], device) * 0.5,
        q_spatial: Tensor::ones([N_REACHES], device) * 0.5,
        p_spatial: None,
    };

    let mut mc = MuskingumCunge::<I>::new(inputs.config.clone(), device.clone());
    mc.setup_inputs(routing_inputs, qprime, params, false);
    let out: Vec<f32> = mc.forward().into_data().to_vec().map_err(|e| {
        CliError::Runtime(format!("sandbox smoke: failed to extract output tensor: {e}"))
    })?;

    let n_nan = out.iter().filter(|v| !v.is_finite()).count();
    let n_negative = out.iter().filter(|&&v| v < 0.0).count();
    let max_q = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let has_positive = out.iter().any(|&v| v > 0.0);

    let passed = n_nan == 0 && n_negative == 0 && has_positive;

    Ok(SmokeResult {
        passed,
        max_q,
        n_reaches: N_REACHES,
        n_nan,
        n_negative,
    })
}
