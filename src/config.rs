//! Subset of the DDR `Config` schema needed by the routing core and dataset.
//!
//! Mirrors `~/projects/ddr/src/ddr/validation/configs.py`. The routing core
//! reads `params`; SP-3 dataset code reads `data_sources`, `experiment`, and
//! `mlp`. Higher-level fields are kept optional so `Config::default()` still
//! works for code that only needs the solver.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::Deserialize;

use crate::data::error::{DataError, Result};

// ---------------------------------------------------------------------------
// ConfigMode
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConfigMode {
    Training,
    Testing,
}

// ---------------------------------------------------------------------------
// New top-level sections (SP-3)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct DataSources {
    pub attributes: std::path::PathBuf,
    pub conus_adjacency: std::path::PathBuf,
    pub gages_adjacency: std::path::PathBuf,
    pub streamflow: std::path::PathBuf,
    pub observations: std::path::PathBuf,
    pub gages: std::path::PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Experiment {
    pub batch_size: usize,
    pub start_time: String,
    pub end_time: String,
    pub epochs: usize,
    pub rho: Option<usize>,
    #[serde(default)]
    pub shuffle: bool,
    pub warmup: usize,
    #[serde(default)]
    pub learning_rate: BTreeMap<usize, f32>,
    #[serde(default)]
    pub grad_clip_max_norm: Option<f32>,
    #[serde(default)]
    pub checkpoint: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MlpConfigSection {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub input_var_names: Vec<String>,
    pub learnable_parameters: Vec<String>,
}

// ---------------------------------------------------------------------------
// Routing parameter types (pre-existing, kept as named-field structs)
// ---------------------------------------------------------------------------

/// Physical lower bounds applied during routing to keep the math stable.
#[derive(Debug, Clone)]
pub struct AttributeMinimums {
    pub discharge: f32,
    pub slope: f32,
    pub velocity: f32,
    pub depth: f32,
    pub bottom_width: f32,
}

impl Default for AttributeMinimums {
    fn default() -> Self {
        // Matches `Params.attribute_minimums` defaults in DDR.
        Self {
            discharge: 1e-4,
            slope: 1e-3,
            velocity: 0.01,
            depth: 0.01,
            bottom_width: 0.01,
        }
    }
}

/// Physical bounds `[min, max]` used to denormalize NN [0,1] outputs.
#[derive(Debug, Clone)]
pub struct ParameterRanges {
    pub n: [f32; 2],
    pub q_spatial: [f32; 2],
    pub p_spatial: [f32; 2],
}

impl Default for ParameterRanges {
    fn default() -> Self {
        Self {
            n: [0.015, 0.25],
            q_spatial: [0.0, 1.0],
            p_spatial: [1.0, 200.0],
        }
    }
}

/// Selects the backend implementation of the CSR triangular solve in
/// `MuskingumCunge`. `Cuda` opts into the cuSPARSE path when the runtime
/// backend is `burn::backend::Cuda`; on other backends the solver silently
/// falls back to `Cpu` (logged once at WARN).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SparseSolver {
    #[default]
    Cpu,
    Cuda,
}

/// Routing parameter configuration.
#[derive(Debug, Clone)]
pub struct Params {
    pub parameter_ranges: ParameterRanges,
    pub log_space_parameters: Vec<String>,
    pub defaults: HashMap<String, f32>,
    pub attribute_minimums: AttributeMinimums,
    pub tau: u32,
    pub sparse_solver: SparseSolver,
    /// SP-10: enable per-timestep CUDA-graph capture/replay on the CUDA
    /// path. No effect on the CPU path. Defaults to `false`; flipped to
    /// `true` in `config/merit_training.yaml` only after V9/V10/V7a pass.
    pub use_cuda_graphs: bool,
}

impl Default for Params {
    fn default() -> Self {
        let mut defaults = HashMap::new();
        defaults.insert("p_spatial".to_string(), 21.0);
        Self {
            parameter_ranges: ParameterRanges::default(),
            log_space_parameters: vec!["p_spatial".to_string()],
            defaults,
            attribute_minimums: AttributeMinimums::default(),
            tau: 3,
            sparse_solver: SparseSolver::default(),
            use_cuda_graphs: false,
        }
    }
}

// ---------------------------------------------------------------------------
// YAML intermediate for Params (dict-shaped in YAML, named fields in Rust)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ParamsRaw {
    parameter_ranges: HashMap<String, [f32; 2]>,
    attribute_minimums: HashMap<String, f32>,
    defaults: HashMap<String, f32>,
    log_space_parameters: Vec<String>,
    tau: Option<u32>,
    sparse_solver: Option<String>,
    use_cuda_graphs: Option<bool>,
}

impl From<ParamsRaw> for Params {
    fn from(r: ParamsRaw) -> Self {
        let mut p = Params::default();
        // parameter_ranges — named field mapping.
        if let Some(v) = r.parameter_ranges.get("n") {
            p.parameter_ranges.n = *v;
        }
        if let Some(v) = r.parameter_ranges.get("q_spatial") {
            p.parameter_ranges.q_spatial = *v;
        }
        if let Some(v) = r.parameter_ranges.get("p_spatial") {
            p.parameter_ranges.p_spatial = *v;
        }
        // attribute_minimums — named field mapping.
        if let Some(&v) = r.attribute_minimums.get("discharge") {
            p.attribute_minimums.discharge = v;
        }
        if let Some(&v) = r.attribute_minimums.get("slope") {
            p.attribute_minimums.slope = v;
        }
        if let Some(&v) = r.attribute_minimums.get("velocity") {
            p.attribute_minimums.velocity = v;
        }
        if let Some(&v) = r.attribute_minimums.get("depth") {
            p.attribute_minimums.depth = v;
        }
        if let Some(&v) = r.attribute_minimums.get("bottom_width") {
            p.attribute_minimums.bottom_width = v;
        }
        // defaults and log_space_parameters override if non-empty.
        if !r.defaults.is_empty() {
            p.defaults = r.defaults;
        }
        if !r.log_space_parameters.is_empty() {
            p.log_space_parameters = r.log_space_parameters;
        }
        p.tau = r.tau.unwrap_or(3);
        p.sparse_solver = match r.sparse_solver.as_deref() {
            Some("cuda") | Some("CUDA") => SparseSolver::Cuda,
            Some("cpu") | Some("CPU") | None => SparseSolver::Cpu,
            Some(other) => panic!("unknown sparse_solver: {other:?} (expected \"cpu\" or \"cuda\")"),
        };
        if let Some(b) = r.use_cuda_graphs {
            p.use_cuda_graphs = b;
        }
        p
    }
}

// ---------------------------------------------------------------------------
// Root Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Config {
    pub params: Params,
    pub data_sources: Option<DataSources>,
    pub experiment: Option<Experiment>,
    pub mlp: Option<MlpConfigSection>,
    pub mode: String,
    pub geodataset: String,
    pub seed: u64,
    pub np_seed: u64,
}

/// Overlay section from `testing:` in the YAML.
/// Fields are all optional so absent keys inherit from `experiment:`.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(default)]
struct TestingOverridesRaw {
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub batch_size: Option<usize>,
    /// Double Option: serde-yaml distinguishes "key absent" (outer None)
    /// from "key present with value null" (Some(None)). The latter
    /// explicitly clears rho even if experiment had it set.
    #[serde(default, deserialize_with = "deserialize_option_option")]
    pub rho: Option<Option<usize>>,
    pub warmup: Option<usize>,
    pub epochs: Option<usize>,
    pub grad_clip_max_norm: Option<f32>,
    pub checkpoint: Option<String>,
}

/// Allows `rho: null` in YAML to be distinct from `rho` being absent.
fn deserialize_option_option<'de, D>(d: D) -> std::result::Result<Option<Option<usize>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Some(Option::<usize>::deserialize(d)?))
}

/// YAML-shaped intermediate; the public `Config` has nicer types.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ConfigRaw {
    mode: Option<String>,
    geodataset: Option<String>,
    seed: Option<u64>,
    np_seed: Option<u64>,
    params: ParamsRaw,
    data_sources: Option<DataSources>,
    experiment: Option<Experiment>,
    mlp: Option<MlpConfigSection>,
    testing: TestingOverridesRaw,
}

impl From<ConfigRaw> for Config {
    fn from(r: ConfigRaw) -> Self {
        Self {
            params: r.params.into(),
            data_sources: r.data_sources,
            experiment: r.experiment,
            mlp: r.mlp,
            mode: r.mode.unwrap_or_else(|| "training".to_string()),
            geodataset: r.geodataset.unwrap_or_else(|| "merit".to_string()),
            seed: r.seed.unwrap_or(42),
            np_seed: r.np_seed.unwrap_or(42),
        }
    }
}

impl Config {
    /// Back-compat: defaults to Training mode (no overlay applied).
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_yaml_file_with_mode(path, ConfigMode::Training)
    }

    /// Load the YAML, optionally apply the `testing:` section as an overlay
    /// onto `experiment:`. Testing-mode batch_size has semantic shift:
    /// represents number of DAYS per chunk (not gauges).
    pub fn from_yaml_file_with_mode(
        path: impl AsRef<Path>,
        mode: ConfigMode,
    ) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| DataError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let raw: ConfigRaw = serde_yaml::from_slice(&bytes).map_err(|e| DataError::Yaml {
            path: path.to_path_buf(),
            source: e,
        })?;
        let testing_raw = raw.testing.clone();
        let mut cfg: Self = raw.into();
        if mode == ConfigMode::Testing {
            apply_testing_overlay(&mut cfg, testing_raw);
        }
        Ok(cfg)
    }
}

fn apply_testing_overlay(cfg: &mut Config, overrides: TestingOverridesRaw) {
    let Some(exp) = cfg.experiment.as_mut() else { return; };
    if let Some(v) = overrides.start_time { exp.start_time = v; }
    if let Some(v) = overrides.end_time { exp.end_time = v; }
    if let Some(v) = overrides.batch_size { exp.batch_size = v; }
    if let Some(v) = overrides.rho { exp.rho = v; }
    if let Some(v) = overrides.warmup { exp.warmup = v; }
    if let Some(v) = overrides.epochs { exp.epochs = v; }
    if let Some(v) = overrides.grad_clip_max_norm { exp.grad_clip_max_norm = Some(v); }
    if let Some(v) = overrides.checkpoint {
        exp.checkpoint = Some(std::path::PathBuf::from(v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_merit_training_yaml() {
        let path = "config/merit_training.yaml";
        let cfg = Config::from_yaml_file(path).expect("load yaml");
        assert_eq!(cfg.experiment.as_ref().unwrap().batch_size, 64);
        assert_eq!(cfg.experiment.as_ref().unwrap().rho, Some(90));
        assert_eq!(cfg.experiment.as_ref().unwrap().warmup, 5);
        let ds = cfg.data_sources.as_ref().unwrap();
        assert!(ds.conus_adjacency.file_name().unwrap().to_str().unwrap() == "merit_conus_adjacency.zarr");
        assert!(ds.streamflow.extension().map(|e| e == "ic").unwrap_or(false));
        // params still readable.
        let pr = &cfg.params.parameter_ranges;
        assert!((pr.n[0] - 0.015).abs() < 1e-9);
        assert!((pr.p_spatial[1] - 200.0).abs() < 1e-9);
        // log_space_parameters from YAML overrides default.
        assert_eq!(cfg.params.log_space_parameters, vec!["n".to_string()]);
        // mlp section.
        let mlp = cfg.mlp.as_ref().unwrap();
        assert_eq!(mlp.hidden_size, 21);
        assert_eq!(mlp.input_var_names.len(), 10);
        // tau defaults to 3 when not set in YAML.
        assert_eq!(cfg.params.tau, 3);
        // sparse_solver is set to Cuda by merit_training.yaml (since SP-9).
        assert_eq!(cfg.params.sparse_solver, SparseSolver::Cuda);
        // SP-10: use_cuda_graphs defaults to false when not set in YAML.
        assert!(!cfg.params.use_cuda_graphs);
        // top-level scalars.
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.mode, "training");
    }

    #[test]
    fn default_config_still_constructs() {
        // Sanity: existing call sites that use Config::default() still work.
        let cfg = Config::default();
        assert!(cfg.params.parameter_ranges.n[0] > 0.0);
        assert!(cfg.data_sources.is_none());
        assert!(cfg.experiment.is_none());
    }

    #[test]
    fn testing_mode_overlays_apply_to_experiment() {
        let cfg = Config::from_yaml_file_with_mode(
            "config/merit_training.yaml",
            ConfigMode::Testing,
        ).expect("yaml");
        let exp = cfg.experiment.as_ref().unwrap();
        assert_eq!(exp.batch_size, 15);
        assert_eq!(exp.start_time, "1995/10/01");
        assert_eq!(exp.end_time, "2010/09/30");
        assert!(exp.rho.is_none(), "rho should be cleared by testing overlay");
    }

    #[test]
    fn training_mode_does_not_apply_overlays() {
        let cfg = Config::from_yaml_file_with_mode(
            "config/merit_training.yaml",
            ConfigMode::Training,
        ).expect("yaml");
        let exp = cfg.experiment.as_ref().unwrap();
        assert_eq!(exp.batch_size, 64, "training default preserved");
        assert_eq!(exp.rho, Some(90), "training default rho preserved");
        assert_eq!(exp.start_time, "1981/10/01");
    }
}
