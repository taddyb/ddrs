//! Subset of the DDR `Config` schema needed by the routing core and dataset.
//!
//! Mirrors `~/projects/ddr/src/ddr/validation/configs.py`. The routing core
//! reads `params`; SP-3 dataset code reads `data_sources`, `experiment`, and
//! `mlp`. Higher-level fields are kept optional so `Config::default()` still
//! works for code that only needs the solver.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::de::Error as _;
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
// Workflow
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum,
    serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
#[clap(rename_all = "kebab-case")]
pub enum Workflow {
    /// Train the KAN head (requires `mode: training`).
    Train,
    /// Evaluate a trained checkpoint over the testing window (requires `mode: testing`).
    Eval,
    /// Train, then evaluate, then compare against the summed-Q' baseline.
    TrainAndTest,
}

// ---------------------------------------------------------------------------
// New top-level sections (SP-3)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct DataSources {
    pub attributes: std::path::PathBuf,
    /// Pre-built CONUS adjacency zarr store. Either both adjacency keys must
    /// be present, or `geospatial_fabric` must be provided for managed builds.
    #[serde(default)]
    pub conus_adjacency: Option<std::path::PathBuf>,
    /// Pre-built per-gauge adjacency zarr store. Either both adjacency keys must
    /// be present, or `geospatial_fabric` must be provided for managed builds.
    #[serde(default)]
    pub gages_adjacency: Option<std::path::PathBuf>,
    pub streamflow: std::path::PathBuf,
    pub observations: std::path::PathBuf,
    pub gages: std::path::PathBuf,
    /// Path to the MERIT flowlines fabric: `.shp` (sibling `.dbf` read),
    /// `.dbf`, or `.gpkg` (attribute columns read via SQL; geometry never
    /// opened in any format). Matches DDR's `geospatial_fabric_gpkg` artifact.
    /// Required when `conus_adjacency` and `gages_adjacency` are absent;
    /// used by `ddrs plan` to build them.
    #[serde(default)]
    pub geospatial_fabric: Option<std::path::PathBuf>,
    /// Feature layer to read when `geospatial_fabric` is a `.gpkg` with more
    /// than one feature layer (e.g. a file holding both flowlines and
    /// catchments). Optional for single-layer gpkg files; invalid for
    /// `.shp`/`.dbf` fabrics.
    #[serde(default)]
    pub geospatial_fabric_layer: Option<String>,
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
    /// Training objective. Defaults to L1 (the historical loss) so configs
    /// without a `loss:` block are byte-for-byte unchanged in behavior.
    #[serde(default)]
    pub loss: LossConfig,
}

/// Selects the training objective and (for the composite objective) its
/// component weights. See `src/training/loss.rs`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct LossConfig {
    /// Which objective to optimize.
    pub kind: LossKind,
    /// Weight on the `1 - NNSE` term (composite objective only).
    pub nnse_weight: f32,
    /// Weight on the `1 - KGE` term (composite objective only).
    pub kge_weight: f32,
    /// Stabilization constant added to variance denominators / mean-bias
    /// denominator so near-constant gauges don't produce NaN gradients.
    /// Matches DDR `hydrograph_loss`'s `eps=0.1`.
    pub eps: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        // L1 fallback preserves the prior training behavior exactly.
        Self { kind: LossKind::L1, nnse_weight: 1.0, kge_weight: 1.0, eps: 0.1 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum LossKind {
    /// `mean(|p - o|)` — the historical objective; rewards peak attenuation.
    #[default]
    L1,
    /// `λ_nnse·(1 - NNSE) + λ_kge·(1 - KGE)`, per gauge. The KGE term's
    /// `(α-1)²` restores the hydrograph variance L1/NSE shrink away.
    NnseKge,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KanHeadConfigSection {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// B-spline grid intervals (`num` in pykan). Default 5 matches DDR.
    #[serde(default = "default_grid")]
    pub grid: usize,
    /// B-spline order. Default 3 (cubic).
    #[serde(default = "default_k")]
    pub k: usize,
    pub input_var_names: Vec<String>,
    pub learnable_parameters: Vec<String>,
}

fn default_grid() -> usize {
    5
}
fn default_k() -> usize {
    3
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
    pub kan_head: Option<KanHeadConfigSection>,
    pub mode: String,
    pub geodataset: String,
    pub seed: u64,
    pub np_seed: u64,
    pub workflow: Option<Workflow>,
    /// CUDA device ordinal (mirrors DDR's top-level `device:` key,
    /// e.g. `device: 2` → `cuda:2`). Defaults to 0.
    pub device: usize,
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
    workflow: Option<Workflow>,
    device: Option<usize>,
    params: ParamsRaw,
    data_sources: Option<DataSources>,
    experiment: Option<Experiment>,
    /// `kan_head:` is the v1 YAML key; `mlp:` is accepted as a backward-compat
    /// alias so existing YAML configs still parse during the migration.
    #[serde(alias = "mlp")]
    kan_head: Option<KanHeadConfigSection>,
    testing: TestingOverridesRaw,
}

impl From<ConfigRaw> for Config {
    fn from(r: ConfigRaw) -> Self {
        Self {
            params: r.params.into(),
            data_sources: r.data_sources,
            experiment: r.experiment,
            kan_head: r.kan_head,
            mode: r.mode.unwrap_or_else(|| "training".to_string()),
            geodataset: r.geodataset.unwrap_or_else(|| "merit".to_string()),
            seed: r.seed.unwrap_or(42),
            np_seed: r.np_seed.unwrap_or(42),
            workflow: r.workflow,
            device: r.device.unwrap_or(0),
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
        validate_mode_workflow(&cfg).map_err(|msg| DataError::Yaml {
            path: path.to_path_buf(),
            source: serde_yaml::Error::custom(msg),
        })?;
        validate_data_sources(&cfg).map_err(|msg| DataError::Yaml {
            path: path.to_path_buf(),
            source: serde_yaml::Error::custom(msg),
        })?;
        if mode == ConfigMode::Testing {
            apply_testing_overlay(&mut cfg, testing_raw);
        }
        Ok(cfg)
    }
}

/// Validate the adjacency / geospatial-fabric combination in `data_sources`.
///
/// Rules:
/// - Both `conus_adjacency` **and** `gages_adjacency` present → OK (explicit zarr).
/// - Neither adjacency key present, but `geospatial_fabric` present → OK (managed build).
/// - Neither adjacency key and no `geospatial_fabric` → error: name missing keys.
/// - Exactly one adjacency key → error: partial adjacency.
/// - `geospatial_fabric_layer` set while the fabric is not a `.gpkg` → error
///   (the layer concept only exists for GeoPackage).
fn validate_data_sources(cfg: &Config) -> std::result::Result<(), String> {
    let ds = match cfg.data_sources.as_ref() {
        None => return Ok(()), // no data_sources section at all — allowed for default/test configs
        Some(ds) => ds,
    };
    let has_conus = ds.conus_adjacency.is_some();
    let has_gages = ds.gages_adjacency.is_some();
    let has_fabric = ds.geospatial_fabric.is_some();

    // Layer selection is a gpkg-only concept; reject it for dBASE fabrics
    // (or with no fabric at all) at load time rather than silently ignoring.
    if ds.geospatial_fabric_layer.is_some() {
        let is_gpkg = ds
            .geospatial_fabric
            .as_ref()
            .and_then(|p| p.extension())
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gpkg"));
        if !is_gpkg {
            return Err(
                "data_sources: `geospatial_fabric_layer` is set but `geospatial_fabric` \
                 is not a .gpkg file — the layer key only applies to GeoPackage fabrics."
                    .to_string(),
            );
        }
    }

    match (has_conus, has_gages, has_fabric) {
        // Both explicit zarr paths: valid.
        (true, true, _) => Ok(()),
        // Managed build: neither zarr, fabric present: valid.
        (false, false, true) => Ok(()),
        // Partial adjacency: exactly one of the two zarr keys provided.
        (true, false, _) => Err(
            "data_sources: `conus_adjacency` is set but `gages_adjacency` is missing; \
             provide both adjacency paths or remove both and set `geospatial_fabric`."
            .to_string(),
        ),
        (false, true, _) => Err(
            "data_sources: `gages_adjacency` is set but `conus_adjacency` is missing; \
             provide both adjacency paths or remove both and set `geospatial_fabric`."
            .to_string(),
        ),
        // Neither zarr and no fabric.
        (false, false, false) => Err(
            "data_sources: adjacency sources are missing — either set both \
             `conus_adjacency` and `gages_adjacency`, or set `geospatial_fabric` \
             for a managed adjacency build."
            .to_string(),
        ),
    }
}

fn validate_mode_workflow(cfg: &Config) -> std::result::Result<(), String> {
    use Workflow::*;
    let Some(wf) = cfg.workflow else { return Ok(()); };
    let ok = match (cfg.mode.as_str(), wf) {
        ("training", Train | TrainAndTest) => true,
        ("testing", Eval) => true,
        _ => false,
    };
    if !ok {
        return Err(format!(
            "conflicting top-level keys — mode: {} but workflow: {} \
             (mode=training implies workflow ∈ {{train, train-and-test}}; \
              mode=testing implies workflow=eval)",
            cfg.mode,
            match wf { Train => "train", Eval => "eval", TrainAndTest => "train-and-test" },
        ));
    }
    Ok(())
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
        // After the managed-adjacency change, merit_training.yaml uses
        // geospatial_fabric instead of the two explicit zarr paths.
        assert!(ds.conus_adjacency.is_none(), "conus_adjacency should be absent after Task 1");
        assert!(ds.gages_adjacency.is_none(), "gages_adjacency should be absent after Task 1");
        assert!(ds.geospatial_fabric.is_some(), "geospatial_fabric must be set");
        assert!(ds.streamflow.extension().map(|e| e == "ic").unwrap_or(false));
        // params still readable.
        let pr = &cfg.params.parameter_ranges;
        assert!((pr.n[0] - 0.015).abs() < 1e-9);
        assert!((pr.p_spatial[1] - 200.0).abs() < 1e-9);
        // log_space_parameters from YAML overrides default.
        assert_eq!(cfg.params.log_space_parameters, vec!["p_spatial".to_string()]);
        // kan_head section.
        let kan_head = cfg.kan_head.as_ref().unwrap();
        assert_eq!(kan_head.hidden_size, 21);
        assert_eq!(kan_head.input_var_names.len(), 10);
        // tau defaults to 3 when not set in YAML.
        assert_eq!(cfg.params.tau, 3);
        // sparse_solver is set to Cuda by merit_training.yaml (since SP-9).
        assert_eq!(cfg.params.sparse_solver, SparseSolver::Cuda);
        // SP-10: merit_training.yaml now sets use_cuda_graphs: true
        // (flipped by commit e35af29 after V7a=0.385 landed).
        assert!(cfg.params.use_cuda_graphs);
        // top-level scalars.
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.mode, "training");
        assert_eq!(cfg.workflow, Some(Workflow::TrainAndTest));
        assert_eq!(cfg.device, 0);
    }

    #[test]
    fn device_parses_and_defaults_to_zero() {
        // Explicit `device:` key parses (mirrors DDR's top-level `device: 2`).
        let yaml = "mode: training\ngeodataset: merit\nseed: 1\nnp_seed: 1\ndevice: 3\n";
        let path = std::env::temp_dir().join("ddrs_config_device_test.yaml");
        std::fs::write(&path, yaml).unwrap();
        let cfg = Config::from_yaml_file(&path).expect("load yaml");
        assert_eq!(cfg.device, 3);

        // Absent key defaults to device 0.
        let yaml = "mode: training\ngeodataset: merit\nseed: 1\nnp_seed: 1\n";
        let path = std::env::temp_dir().join("ddrs_config_no_device_test.yaml");
        std::fs::write(&path, yaml).unwrap();
        let cfg = Config::from_yaml_file(&path).expect("load yaml");
        assert_eq!(cfg.device, 0);
        assert_eq!(Config::default().device, 0);
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
    fn loss_config_defaults_to_l1() {
        // An experiment block without `loss:` must default to L1 so existing
        // configs train identically.
        let exp: Experiment = serde_yaml::from_str(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\n",
        )
        .expect("parse experiment");
        assert_eq!(exp.loss.kind, LossKind::L1);
        assert!((exp.loss.eps - 0.1).abs() < 1e-9);
    }

    #[test]
    fn loss_config_parses_nnse_kge_kebab_case() {
        let lc: LossConfig = serde_yaml::from_str(
            "kind: nnse-kge\nnnse-weight: 0.5\nkge-weight: 2.0\neps: 0.05\n",
        )
        .expect("parse loss");
        assert_eq!(lc.kind, LossKind::NnseKge);
        assert!((lc.nnse_weight - 0.5).abs() < 1e-9);
        assert!((lc.kge_weight - 2.0).abs() < 1e-9);
        assert!((lc.eps - 0.05).abs() < 1e-9);
    }

    #[test]
    fn loads_workflow_from_yaml() {
        let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: train-and-test
"#;
        let path = std::env::temp_dir().join("ddrs_config_workflow_test.yaml");
        std::fs::write(&path, yaml).unwrap();
        let cfg = Config::from_yaml_file(&path).expect("load yaml");
        assert_eq!(cfg.workflow, Some(Workflow::TrainAndTest));
    }

    #[test]
    fn workflow_absent_is_none() {
        let yaml = "mode: training\ngeodataset: merit\nseed: 1\nnp_seed: 1\n";
        let path = std::env::temp_dir().join("ddrs_config_no_workflow_test.yaml");
        std::fs::write(&path, yaml).unwrap();
        let cfg = Config::from_yaml_file(&path).expect("load yaml");
        assert_eq!(cfg.workflow, None);
    }

    #[test]
    fn mode_workflow_conflict_rejected() {
        let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: eval
"#;
        let path = std::env::temp_dir().join("ddrs_config_conflict_test.yaml");
        std::fs::write(&path, yaml).unwrap();
        let err = Config::from_yaml_file(&path).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("conflicting") && msg.contains("mode: training") && msg.contains("workflow: eval"),
            "expected conflict message, got: {msg}"
        );
    }

    #[test]
    fn mode_testing_with_train_workflow_rejected() {
        let yaml = r#"
mode: testing
geodataset: merit
seed: 1
np_seed: 1
workflow: train
"#;
        let path = std::env::temp_dir().join("ddrs_config_conflict2_test.yaml");
        std::fs::write(&path, yaml).unwrap();
        let err = Config::from_yaml_file(&path).unwrap_err();
        assert!(format!("{}", err).contains("conflicting"));
    }

    // ── data_sources validation matrix ──────────────────────────────────────

    fn write_yaml_with_data_sources(name: &str, ds_block: &str) -> std::path::PathBuf {
        let yaml = format!(
            "mode: training\ngeodataset: merit\nseed: 1\nnp_seed: 1\n{ds_block}"
        );
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, yaml).unwrap();
        path
    }

    #[test]
    fn both_adjacency_paths_valid() {
        // Both conus_adjacency + gages_adjacency present → valid regardless of fabric.
        let ds_block = r#"
data_sources:
  attributes: /dev/null/attrs.nc
  conus_adjacency: /dev/null/conus.zarr
  gages_adjacency: /dev/null/gages.zarr
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_ds_both_adj.yaml", ds_block);
        let cfg = Config::from_yaml_file(&path).expect("both adjacency paths should be valid");
        let ds = cfg.data_sources.as_ref().unwrap();
        assert!(ds.conus_adjacency.is_some());
        assert!(ds.gages_adjacency.is_some());
        assert!(ds.geospatial_fabric.is_none());
    }

    #[test]
    fn fabric_only_valid() {
        // Neither adjacency path, geospatial_fabric set → valid (managed build).
        let ds_block = r#"
data_sources:
  attributes: /dev/null/attrs.nc
  geospatial_fabric: /dev/null/rivers.shp
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_ds_fabric_only.yaml", ds_block);
        let cfg = Config::from_yaml_file(&path).expect("fabric-only should be valid");
        let ds = cfg.data_sources.as_ref().unwrap();
        assert!(ds.conus_adjacency.is_none());
        assert!(ds.gages_adjacency.is_none());
        assert!(ds.geospatial_fabric.is_some());
    }

    #[test]
    fn neither_adjacency_nor_fabric_rejected() {
        // Neither adjacency keys nor geospatial_fabric → config error.
        let ds_block = r#"
data_sources:
  attributes: /dev/null/attrs.nc
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_ds_none_adj.yaml", ds_block);
        let err = Config::from_yaml_file(&path).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("adjacency sources are missing"),
            "expected missing-adjacency error, got: {msg}"
        );
    }

    #[test]
    fn partial_adjacency_conus_only_rejected() {
        // Only conus_adjacency, no gages_adjacency → config error.
        let ds_block = r#"
data_sources:
  attributes: /dev/null/attrs.nc
  conus_adjacency: /dev/null/conus.zarr
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_ds_conus_only.yaml", ds_block);
        let err = Config::from_yaml_file(&path).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("gages_adjacency` is missing"),
            "expected partial-adjacency error, got: {msg}"
        );
    }

    #[test]
    fn partial_adjacency_gages_only_rejected() {
        // Only gages_adjacency, no conus_adjacency → config error.
        let ds_block = r#"
data_sources:
  attributes: /dev/null/attrs.nc
  gages_adjacency: /dev/null/gages.zarr
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_ds_gages_only.yaml", ds_block);
        let err = Config::from_yaml_file(&path).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("conus_adjacency` is missing"),
            "expected partial-adjacency error, got: {msg}"
        );
    }

    #[test]
    fn gpkg_fabric_with_layer_valid() {
        // .gpkg fabric + explicit layer → valid managed-build config.
        let ds_block = r#"
data_sources:
  attributes: /dev/null/attrs.nc
  geospatial_fabric: /dev/null/global_merit_riv.gpkg
  geospatial_fabric_layer: flowlines
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_ds_gpkg_layer.yaml", ds_block);
        let cfg = Config::from_yaml_file(&path).expect("gpkg fabric + layer should be valid");
        let ds = cfg.data_sources.as_ref().unwrap();
        assert_eq!(ds.geospatial_fabric_layer.as_deref(), Some("flowlines"));
    }

    #[test]
    fn layer_without_gpkg_fabric_rejected() {
        // geospatial_fabric_layer alongside a .shp fabric → config error.
        let ds_block = r#"
data_sources:
  attributes: /dev/null/attrs.nc
  geospatial_fabric: /dev/null/rivers.shp
  geospatial_fabric_layer: flowlines
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_ds_layer_no_gpkg.yaml", ds_block);
        let err = Config::from_yaml_file(&path).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("geospatial_fabric_layer") && msg.contains(".gpkg"),
            "expected layer-without-gpkg error, got: {msg}"
        );
    }

    #[test]
    fn both_adjacency_and_fabric_valid() {
        // Both zarr paths + fabric present → valid (fabric is informational/extra).
        let ds_block = r#"
data_sources:
  attributes: /dev/null/attrs.nc
  conus_adjacency: /dev/null/conus.zarr
  gages_adjacency: /dev/null/gages.zarr
  geospatial_fabric: /dev/null/rivers.shp
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_ds_all.yaml", ds_block);
        let cfg = Config::from_yaml_file(&path).expect("both adjacency + fabric should be valid");
        let ds = cfg.data_sources.as_ref().unwrap();
        assert!(ds.conus_adjacency.is_some());
        assert!(ds.gages_adjacency.is_some());
        assert!(ds.geospatial_fabric.is_some());
    }

    #[test]
    fn no_data_sources_section_valid() {
        // Config without data_sources (e.g. routing-only usage) → valid.
        let yaml = "mode: training\ngeodataset: merit\nseed: 1\nnp_seed: 1\n";
        let path = std::env::temp_dir().join("ddrs_ds_absent.yaml");
        std::fs::write(&path, yaml).unwrap();
        let cfg = Config::from_yaml_file(&path).expect("no data_sources should be valid");
        assert!(cfg.data_sources.is_none());
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
