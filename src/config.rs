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

/// Deserialize `attributes` as either a bare path string or a YAML list of
/// path strings. A bare string maps to a single-element `Vec`; a list maps
/// one-to-one. An empty list is a hard error. Preserved so all existing
/// configs (which use the bare-path form) parse unchanged.
fn deserialize_one_or_many_paths<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<std::path::PathBuf>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};

    struct OneOrManyPaths;

    impl<'de> Visitor<'de> for OneOrManyPaths {
        type Value = Vec<std::path::PathBuf>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a path string or a list of path strings")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
            Ok(vec![std::path::PathBuf::from(v)])
        }

        fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Self::Value, E> {
            Ok(vec![std::path::PathBuf::from(v)])
        }

        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut paths: Vec<std::path::PathBuf> = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                paths.push(std::path::PathBuf::from(s));
            }
            if paths.is_empty() {
                return Err(A::Error::custom(
                    "data_sources.attributes: list must not be empty",
                ));
            }
            Ok(paths)
        }
    }

    deserializer.deserialize_any(OneOrManyPaths)
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataSources {
    #[serde(deserialize_with = "deserialize_one_or_many_paths")]
    pub attributes: Vec<std::path::PathBuf>,
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
    /// Optional hourly AORC precipitation store (`merit_unit_catchments.zarr`,
    /// zarr v3). CONUS-only; drives the precip-conditioned disaggregation head
    /// (`kan_head.disaggregation.use_precip`). Absent ⇒ the head conditions on
    /// daily Q' only (or, with disaggregation off, flat repeat-24).
    #[serde(default)]
    pub aorc_precip: Option<std::path::PathBuf>,
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
    /// Optional path to a day-boundary discharge state cache (netCDF produced
    /// by `--mode state-cache` in the probe binary). When set, `collate`
    /// attaches the window-start per-reach Q vector as
    /// `RoutingBatch::initial_state`. Absent → `None` → all behavior
    /// byte-identical to the no-cache path (the hotstart heuristic remains in
    /// effect until Task 3 wires the state into the forward pass).
    #[serde(default)]
    pub state_cache: Option<std::path::PathBuf>,
    /// Training objective. Defaults to L1 (the historical loss) so configs
    /// without a `loss:` block are byte-for-byte unchanged in behavior.
    #[serde(default)]
    pub loss: LossConfig,
    /// Master switch for optimizer micro-batching (gradient accumulation),
    /// following the `use_leakance`/`use_cuda_graphs` flag convention.
    /// Default `false` ⇒ byte-identical single-batch training regardless of
    /// `grad_accum_steps` (the tuning stays in the file, inert). `true`
    /// requires `grad_accum_steps >= 2` — rejected at load otherwise, so the
    /// switch can never be a silent no-op.
    /// Which optimizer to use. `adam` (default) preserves every historical
    /// run; `adadelta` matches dHBV's `torch.optim.Adadelta(params)` — a
    /// scale-free update that needs no lr schedule (`lr` is ignored) and can
    /// take meaningful steps when gradient accumulation reduces the run to a
    /// few dozen updates, where Adam's per-step movement is capped at `lr`.
    #[serde(default)]
    pub optimizer: OptimizerKind,
    #[serde(default)]
    pub use_grad_accum: bool,
    /// Group size for `use_grad_accum: true`: number of micro-batches (each
    /// `batch_size` gauges) whose gradients are summed into ONE optimizer
    /// step — effective batch `N × batch_size` at the peak memory of a single
    /// micro-batch. Each micro-batch loss is weighted by its share of the
    /// pooled valid-residual denominator, so the accumulated gradient equals
    /// the true large-batch gradient (see `tests/grad_accum_equivalence.rs`).
    /// When accumulating, the sampler keeps the partial tail batch
    /// (`drop_last = false`) instead of silently dropping
    /// `n_gauges % batch_size` gauges each epoch.
    /// NOTE: iterations-per-epoch shrink by ~N — keep total update count in
    /// mind when comparing to a single-batch baseline.
    #[serde(default)]
    pub grad_accum_steps: Option<usize>,
}

impl Experiment {
    /// The micro-batch group size the trainer actually runs: `1` (single
    /// batch) unless `use_grad_accum: true`, in which case the validated
    /// `grad_accum_steps`. Single source of truth for the driver.
    pub fn effective_grad_accum_steps(&self) -> usize {
        if self.use_grad_accum {
            self.grad_accum_steps.unwrap_or(1).max(1)
        } else {
            1
        }
    }
}

/// Which optimizer the training loop constructs. See `experiment.optimizer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizerKind {
    /// `torch.optim.Adam` defaults (β=0.9/0.999, ε=1e-8) with the YAML
    /// `learning_rate` schedule. The historical choice.
    #[default]
    Adam,
    /// `torch.optim.Adadelta` defaults (ρ=0.9, ε=1e-6, lr=1.0). Ignores the
    /// `learning_rate` schedule by design — AdaDelta derives its own step
    /// size from the ratio of accumulated updates to accumulated gradients.
    Adadelta,
}

/// Selects the training objective and (for the composite objective) its
/// component weights. See `src/training/loss.rs`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct LossConfig {
    /// Which objective to optimize.
    pub kind: LossKind,
    /// Weight on the `1 - NNSE` term (composite `nnse-kge` objective, and the
    /// optional NNSE guard of the component-weighted `kge` objective).
    pub nnse_weight: f32,
    /// Weight on the `1 - KGE` Euclidean term (composite `nnse-kge` only).
    pub kge_weight: f32,
    /// Weight on the `(r - 1)²` correlation term (component-weighted `kge` only).
    pub r_weight: f32,
    /// Weight on the `(α - 1)²` variance-ratio term (component-weighted `kge`
    /// only). This is the restoring force against MC over-attenuation — raise
    /// it to push `σ_sim/σ_obs` back toward 1.
    pub alpha_weight: f32,
    /// Weight on the `(β - 1)²` mean-ratio term (component-weighted `kge` only).
    pub beta_weight: f32,
    /// Per-gauge upper bound on the weighted KGE-component sum before averaging
    /// (component-weighted `kge` only). Gauges with near-constant observed flow
    /// have a collapsing `std_o`/`mean_o` denominator, so `(α-1)²`/`(β-1)²` can
    /// explode and hijack the batch gradient (a single gauge drove batch loss to
    /// ~1e4 in testing). Clamping each gauge's contribution — DDR's
    /// `hydrograph_loss` uses the same trick at 10 — keeps the restoring signal
    /// from well-posed gauges while bounding the pathological ones.
    pub kge_clamp: f32,
    /// Stabilization constant added to variance denominators / mean-bias
    /// denominator so near-constant gauges don't produce NaN gradients.
    /// Matches DDR `hydrograph_loss`'s `eps=0.1`.
    pub eps: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        // L1 fallback preserves the prior training behavior exactly.
        Self {
            kind: LossKind::L1,
            nnse_weight: 1.0,
            kge_weight: 1.0,
            r_weight: 1.0,
            alpha_weight: 1.0,
            beta_weight: 1.0,
            kge_clamp: 10.0,
            eps: 0.1,
        }
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
    /// Per gauge `r_w·(r-1)² + α_w·(α-1)² + β_w·(β-1)² + nnse_w·(1-NNSE)`.
    /// The KGE components are weighted individually (no sqrt Euclidean term,
    /// so no gradient singularity at perfect KGE), letting `α_w` up-weight the
    /// anti-attenuation restoring force. See `src/training/loss.rs`.
    Kge,
    /// dHBV's batch-NSE (hydroDL `NSELossBatch`, Frederick 2019):
    /// `mean over valid (day, gauge) of (sim − obs)² / (σ_gauge + eps)²`,
    /// where `σ_gauge` is the gauge's observed-discharge standard deviation
    /// over the **training period** (fixed, not per window). Normalizing by
    /// σ stops high-variance basins from dominating the batch gradient the
    /// way raw L1 does — the conditioning half of the dHBV recipe that
    /// pairs with the AdaDelta optimizer.
    NseBatch,
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
    /// Optional learnable daily→hourly forcing disaggregation. Absent ⇒ flat
    /// `repeat-24` (current behavior). See `src/nn/disagg_head.rs`.
    #[serde(default)]
    pub disaggregation: Option<DisaggregationSection>,
}

/// YAML `kan_head.disaggregation:` block (presence enables the head).
/// The head always consumes `(daily Q', that day's 24h precip)` — requires
/// `data_sources.aorc_precip` to be set. See `src/nn/disagg_head.rs`.
#[derive(Debug, Clone, Deserialize)]
pub struct DisaggregationSection {
    #[serde(default = "default_disagg_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_disagg_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_disagg_grid")]
    pub grid: usize,
    #[serde(default = "default_disagg_k")]
    pub k: usize,
    /// Day-boundary shape-continuity blend factor λ ∈ [0, 1]. Default 0.0 ⇒
    /// fully independent per-day shapes (no continuity enforcement). Only
    /// used when `chunk_days <= 1`.
    #[serde(default)]
    pub boundary_blend: f32,
    /// Mass-balance chunk size, in days. Default `1` (absent ⇒
    /// byte-identical to every checkpoint trained before this field
    /// existed) preserves the original per-calendar-day-exact contract.
    /// `> 1` relaxes mass conservation to the CHUNK aggregate (see
    /// `DisaggHeadConfig::chunk_days` in `src/nn/disagg_head.rs`), letting
    /// storms that span a day boundary be represented as one continuous
    /// event instead of being clipped at hour 0/23.
    #[serde(default = "default_disagg_chunk_days")]
    pub chunk_days: usize,
    /// Path to a standalone-pretrained DisaggHead checkpoint (CompactRecorder
    /// `.mpk`, e.g. from `pretrain_disagg_capacity`) to warm-start from
    /// instead of a fresh random init. Absent (default) = current behavior
    /// exactly. The architecture fields above (hidden_size /
    /// num_hidden_layers / grid / k / chunk_days) MUST match what the
    /// checkpoint was trained with — `load_record` fails loudly otherwise.
    /// Operational, not architecture: deliberately NOT threaded into
    /// [`kan_config`], mirroring how `experiment.checkpoint` bypasses it.
    #[serde(default)]
    pub pretrained_checkpoint: Option<std::path::PathBuf>,
    /// Freeze this head's params after loading (`Module::no_grad()`, no
    /// further training). Default false. Requires `pretrained_checkpoint` to
    /// be set — enforced by `validate_disagg_pretrained` at config load.
    #[serde(default)]
    pub freeze: bool,
}

/// Build a [`KanHeadConfig`] from a parsed YAML section + seed, threading the
/// optional disaggregation settings. Single source of truth so every call site
/// (train bootstrap, eval, dump) builds an identical head template — required
/// for `load_record` to match a disagg-trained checkpoint.
pub fn kan_config(
    section: &KanHeadConfigSection,
    seed: u64,
) -> crate::nn::kan_head::KanHeadConfig {
    let cfg = crate::nn::kan_head::KanHeadConfig::new(
        section.input_var_names.clone(),
        section.learnable_parameters.clone(),
        seed,
    )
    .with_hidden_size(section.hidden_size)
    .with_num_hidden_layers(section.num_hidden_layers)
    .with_grid(section.grid)
    .with_k(section.k);
    match &section.disaggregation {
        Some(d) => cfg
            .with_disagg_enabled(true)
            .with_disagg_hidden_size(d.hidden_size)
            .with_disagg_num_hidden_layers(d.num_hidden_layers)
            .with_disagg_grid(d.grid)
            .with_disagg_k(d.k)
            .with_disagg_boundary_blend(d.boundary_blend)
            .with_disagg_chunk_days(d.chunk_days),
        None => cfg,
    }
}

fn default_grid() -> usize {
    5
}
fn default_k() -> usize {
    3
}
fn default_disagg_hidden() -> usize {
    16
}
fn default_disagg_chunk_days() -> usize {
    1
}
fn default_disagg_num_hidden_layers() -> usize {
    1
}
fn default_disagg_grid() -> usize {
    3
}
fn default_disagg_k() -> usize {
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
    /// Muskingum storage weight X. Only consumed when `x_storage` is listed in
    /// `kan_head.learnable_parameters`; otherwise routing uses a constant 0.3.
    pub x_storage: [f32; 2],
    /// Hydraulic diffusivity coefficient — verbatim from DDR `configs.py`
    /// (commit c2bd0f9). Only consumed when `Params.use_leakance` is true and
    /// `K_D` is listed in `kan_head.learnable_parameters`.
    pub k_d: [f32; 2],
    /// Groundwater depth parameter. Companion to `k_d`; same activation guard.
    pub d_gw: [f32; 2],
    /// Fractional leakance weight [0, 1]. Companion to `k_d`; same activation guard.
    pub leakance_factor: [f32; 2],
}

impl Default for ParameterRanges {
    fn default() -> Self {
        Self {
            n: [0.015, 0.25],
            q_spatial: [0.0, 1.0],
            p_spatial: [1.0, 200.0],
            x_storage: [0.0, 0.5],
            k_d: [1e-8, 1e-6],
            d_gw: [-2.0, 2.0],
            leakance_factor: [0.0, 1.0],
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
    /// Enable the leakance (GW–SW water-loss) term in routing. Off by default;
    /// when on, `K_D`/`d_gw`/`leakance_factor` must be in
    /// `kan_head.learnable_parameters`, and `use_cuda_graphs` must be false.
    pub use_leakance: bool,
    /// Phase C: clamp the leakance head term to `max(0, depth − d_gw)` so
    /// gaining reaches (depth ≤ d_gw) produce zeta ≡ 0. Defaults to `true`
    /// (Phase C on). Set to `false` to recover the prior unclamped behavior
    /// byte-identically (e.g. for the recovery control answer key).
    pub leakance_losing_only: bool,
    /// Phase C: impervious hard-zero threshold. Reaches with
    /// `corridor_impervious > threshold` get `zeta ≡ 0` and zero gradient to
    /// their leakance params. Only applied when an impervious mask tensor is
    /// supplied at routing setup (the mask is precomputed by the caller).
    /// Default 0.7 (70% impervious surface ≈ concrete-lined channel).
    pub leakance_impervious_threshold: f32,
    /// When `true` (default) the routing core reproduces DDR's formulation
    /// bit-for-bit, including two known physical approximations:
    ///   * celerity `c = v · 5/3` (the wide-rectangular Kleitz-Seddon limit,
    ///     ~22-27% high for the trapezoid this code actually builds), and
    ///   * Muskingum `X ≡ 0.3` (constant, NOT Cunge-derived, giving a median
    ///     10-30x excess numerical diffusion).
    ///
    /// Set `false` to enable the corrected physics. This CHANGES FORWARD
    /// OUTPUT and will break `examples/compare_ddr_sandbox`'s ABSOLUTE MATCH
    /// (invariant 1) — which is why the default preserves DDR behaviour.
    /// See `.claude/PHYSICS-CORRECTIONS.md`.
    pub ddr_match: bool,
}

fn default_ddr_match() -> bool {
    true
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
            use_leakance: false,
            leakance_losing_only: true,
            leakance_impervious_threshold: 0.7,
            ddr_match: default_ddr_match(),
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
    use_leakance: Option<bool>,
    leakance_losing_only: Option<bool>,
    leakance_impervious_threshold: Option<f32>,
    ddr_match: Option<bool>,
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
        if let Some(v) = r.parameter_ranges.get("x_storage") {
            p.parameter_ranges.x_storage = *v;
        }
        // DDR spells this uppercase (K_D); siblings are lowercase.
        if let Some(v) = r.parameter_ranges.get("K_D") {
            p.parameter_ranges.k_d = *v;
        }
        if let Some(v) = r.parameter_ranges.get("d_gw") {
            p.parameter_ranges.d_gw = *v;
        }
        if let Some(v) = r.parameter_ranges.get("leakance_factor") {
            p.parameter_ranges.leakance_factor = *v;
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
        if let Some(b) = r.use_leakance {
            p.use_leakance = b;
        }
        if let Some(b) = r.leakance_losing_only {
            p.leakance_losing_only = b;
        }
        if let Some(v) = r.leakance_impervious_threshold {
            p.leakance_impervious_threshold = v;
        }
        if let Some(b) = r.ddr_match {
            p.ddr_match = b;
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
        validate_leakance(&cfg).map_err(|msg| DataError::Yaml {
            path: path.to_path_buf(),
            source: serde_yaml::Error::custom(msg),
        })?;
        validate_disagg_pretrained(&cfg).map_err(|msg| DataError::Yaml {
            path: path.to_path_buf(),
            source: serde_yaml::Error::custom(msg),
        })?;
        validate_grad_accum(&cfg).map_err(|msg| DataError::Yaml {
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

fn validate_leakance(cfg: &Config) -> std::result::Result<(), String> {
    if cfg.params.use_leakance && cfg.params.use_cuda_graphs {
        return Err(
            "params: `use_leakance: true` requires `use_cuda_graphs: false` — the \
             CUDA-graph capture path bakes the non-leakance b_rhs into the graph."
                .to_string(),
        );
    }
    Ok(())
}

/// `freeze: true` without a `pretrained_checkpoint` would permanently pin the
/// disaggregation head at its random init — reject at load time.
fn validate_disagg_pretrained(cfg: &Config) -> std::result::Result<(), String> {
    if let Some(d) = cfg.kan_head.as_ref().and_then(|k| k.disaggregation.as_ref()) {
        if d.freeze && d.pretrained_checkpoint.is_none() {
            return Err(
                "kan_head.disaggregation: `freeze: true` requires `pretrained_checkpoint` \
                 — freezing a randomly-initialized disaggregation head would train nothing."
                    .to_string(),
            );
        }
    }
    Ok(())
}

/// `use_grad_accum: true` without a meaningful group size would silently run
/// single-batch training — reject at load time (mirrors the disaggregation
/// freeze/pretrained contradiction check). `grad_accum_steps: 0` is likewise
/// meaningless in any combination.
fn validate_grad_accum(cfg: &Config) -> std::result::Result<(), String> {
    if let Some(exp) = cfg.experiment.as_ref() {
        if exp.grad_accum_steps == Some(0) {
            return Err(
                "experiment: `grad_accum_steps: 0` is invalid — set 2 or more \
                 micro-batches per optimizer step (with `use_grad_accum: true`), \
                 or omit the key for single-batch training."
                    .to_string(),
            );
        }
        if exp.use_grad_accum && exp.grad_accum_steps.unwrap_or(1) < 2 {
            return Err(
                "experiment: `use_grad_accum: true` requires `grad_accum_steps: N` \
                 with N >= 2 — accumulating a single micro-batch is identical to \
                 single-batch training and would be a silent no-op."
                    .to_string(),
            );
        }
    }
    Ok(())
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
    fn optimizer_defaults_to_adam_and_parses_adadelta() {
        // Absent key ⇒ Adam ⇒ every historical run unchanged.
        let exp: Experiment = serde_yaml::from_str(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\n",
        )
        .expect("parse experiment");
        assert_eq!(exp.optimizer, OptimizerKind::Adam);

        let exp: Experiment = serde_yaml::from_str(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\noptimizer: adadelta\n",
        )
        .expect("parse experiment");
        assert_eq!(exp.optimizer, OptimizerKind::Adadelta);

        // A typo must fail loudly, not silently fall back to Adam.
        assert!(serde_yaml::from_str::<Experiment>(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\noptimizer: adadelta_typo\n",
        )
        .is_err());
    }

    #[test]
    fn nse_batch_loss_kind_parses() {
        let exp: Experiment = serde_yaml::from_str(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\nloss:\n  kind: nse-batch\n",
        )
        .expect("parse experiment");
        assert_eq!(exp.loss.kind, LossKind::NseBatch);
    }

    #[test]
    fn grad_accum_defaults_off_and_parses() {
        // Absent keys ⇒ toggle off ⇒ byte-identical single-batch training.
        let exp: Experiment = serde_yaml::from_str(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\n",
        )
        .expect("parse experiment");
        assert!(!exp.use_grad_accum);
        assert_eq!(exp.grad_accum_steps, None);
        assert_eq!(exp.effective_grad_accum_steps(), 1);

        // Toggle on with a step count.
        let exp: Experiment = serde_yaml::from_str(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\nuse_grad_accum: true\ngrad_accum_steps: 4\n",
        )
        .expect("parse experiment");
        assert!(exp.use_grad_accum);
        assert_eq!(exp.effective_grad_accum_steps(), 4);
    }

    #[test]
    fn grad_accum_steps_inert_when_toggle_off() {
        // Keeping the tuning in the file with the switch off must fall back
        // to single-batch training — the off switch, not an error.
        let exp: Experiment = serde_yaml::from_str(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\nuse_grad_accum: false\ngrad_accum_steps: 4\n",
        )
        .expect("parse experiment");
        assert!(!exp.use_grad_accum);
        assert_eq!(exp.effective_grad_accum_steps(), 1);
    }

    #[test]
    fn grad_accum_enabled_without_steps_rejected_at_load() {
        // `use_grad_accum: true` with no (or degenerate) step count would be
        // a silent no-op — reject at load, mirroring the freeze/pretrained
        // contradiction check.
        for steps_line in ["", "  grad_accum_steps: 1\n", "  grad_accum_steps: 0\n"] {
            let yaml = format!(
                "mode: training\ngeodataset: merit\nseed: 1\nnp_seed: 1\n\
                 experiment:\n  batch_size: 4\n  start_time: 2000/01/01\n\
                 \x20 end_time: 2000/01/02\n  epochs: 1\n  rho: 10\n  warmup: 1\n\
                 \x20 use_grad_accum: true\n{steps_line}"
            );
            let path = std::env::temp_dir().join("ddrs_config_grad_accum_bad.yaml");
            std::fs::write(&path, yaml).unwrap();
            let err = Config::from_yaml_file(&path).unwrap_err();
            assert!(
                err.to_string().contains("grad_accum"),
                "steps_line={steps_line:?}: expected grad_accum load error, got: {err}"
            );
        }
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
    fn disagg_freeze_without_pretrained_checkpoint_fails_to_load() {
        let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
kan_head:
  hidden_size: 8
  num_hidden_layers: 1
  input_var_names: [a, b]
  learnable_parameters: [n]
  disaggregation:
    freeze: true
"#;
        let path = std::env::temp_dir().join("ddrs_config_disagg_freeze_test.yaml");
        std::fs::write(&path, yaml).unwrap();
        let err = Config::from_yaml_file(&path).expect_err("freeze without checkpoint must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("pretrained_checkpoint"),
            "error should name the missing key, got: {msg}"
        );
    }

    #[test]
    fn disagg_pretrained_fields_default_to_none_and_false() {
        // An existing tracked disagg config (predates the two fields) must
        // parse to (pretrained_checkpoint: None, freeze: false).
        let cfg = Config::from_yaml_file("config/experiments/kan_disagg_conus_no_leakance.yaml")
            .expect("load tracked disagg config");
        let d = cfg
            .kan_head
            .as_ref()
            .unwrap()
            .disaggregation
            .as_ref()
            .expect("disaggregation block present");
        assert!(d.pretrained_checkpoint.is_none());
        assert!(!d.freeze);

        // And the new frozen-arm config round-trips both fields.
        let cfg = Config::from_yaml_file("config/experiments/kan_disagg_conus_frozen_chunk1.yaml")
            .expect("load frozen-arm config");
        let d = cfg.kan_head.as_ref().unwrap().disaggregation.as_ref().unwrap();
        assert!(d.freeze);
        assert_eq!(
            d.pretrained_checkpoint.as_deref(),
            Some(std::path::Path::new(
                "/home/tbindas/projects/ddrs/output/disagg_pretrain/capacity_chunk1.mpk"
            ))
        );
        assert_eq!((d.hidden_size, d.num_hidden_layers, d.grid, d.k), (16, 2, 20, 3));

        // Fine-tune arm: same checkpoint, freeze: false.
        let cfg =
            Config::from_yaml_file("config/experiments/kan_disagg_conus_finetune_chunk1.yaml")
                .expect("load finetune-arm config");
        let d = cfg.kan_head.as_ref().unwrap().disaggregation.as_ref().unwrap();
        assert!(!d.freeze);
        assert!(d.pretrained_checkpoint.is_some());
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
    fn leakance_flag_and_ranges_parse() {
        let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
params:
  use_leakance: true
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-6]
    d_gw: [-2.0, 2.0]
    leakance_factor: [0.0, 1.0]
  log_space_parameters: [p_spatial, K_D]
"#;
        let path = std::env::temp_dir().join("ddrs_leakance_cfg.yaml");
        std::fs::write(&path, yaml).unwrap();
        let cfg = Config::from_yaml_file(&path).expect("load yaml");
        assert!(cfg.params.use_leakance);
        assert!((cfg.params.parameter_ranges.k_d[0] - 1e-8).abs() < 1e-12);
        assert!((cfg.params.parameter_ranges.k_d[1] - 1e-6).abs() < 1e-12);
        assert_eq!(cfg.params.parameter_ranges.d_gw, [-2.0, 2.0]);
        assert_eq!(cfg.params.parameter_ranges.leakance_factor, [0.0, 1.0]);
        assert!(cfg.params.log_space_parameters.iter().any(|s| s == "K_D"));
    }

    #[test]
    fn use_leakance_defaults_false() {
        assert!(!Params::default().use_leakance);
    }

    #[test]
    fn leakance_losing_only_defaults_true() {
        assert!(Params::default().leakance_losing_only,
            "leakance_losing_only must default to true for Phase C");
    }

    #[test]
    fn leakance_losing_only_parses_false() {
        // Explicit false preserves the prior unclamped behavior.
        let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
params:
  use_leakance: true
  leakance_losing_only: false
"#;
        let path = std::env::temp_dir().join("ddrs_leakance_losing_only_false.yaml");
        std::fs::write(&path, yaml).unwrap();
        let cfg = Config::from_yaml_file(&path).expect("load yaml");
        assert!(!cfg.params.leakance_losing_only);
    }

    #[test]
    fn leakance_losing_only_absent_defaults_true() {
        // Absent key must default to true so Phase C configs without it are on.
        let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
params:
  use_leakance: true
"#;
        let path = std::env::temp_dir().join("ddrs_leakance_losing_only_absent.yaml");
        std::fs::write(&path, yaml).unwrap();
        let cfg = Config::from_yaml_file(&path).expect("load yaml");
        assert!(cfg.params.leakance_losing_only);
    }

    #[test]
    fn leakance_impervious_threshold_defaults_to_0_7() {
        assert!(
            (Params::default().leakance_impervious_threshold - 0.7).abs() < 1e-9,
            "leakance_impervious_threshold must default to 0.7"
        );
    }

    #[test]
    fn leakance_impervious_threshold_parses() {
        let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
params:
  use_leakance: true
  leakance_impervious_threshold: 0.5
"#;
        let path = std::env::temp_dir().join("ddrs_leakance_impervious_threshold.yaml");
        std::fs::write(&path, yaml).unwrap();
        let cfg = Config::from_yaml_file(&path).expect("load yaml");
        assert!(
            (cfg.params.leakance_impervious_threshold - 0.5).abs() < 1e-9,
            "leakance_impervious_threshold should parse to 0.5"
        );
    }

    #[test]
    fn leakance_with_cuda_graphs_rejected() {
        let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
params:
  use_leakance: true
  use_cuda_graphs: true
"#;
        let path = std::env::temp_dir().join("ddrs_leakance_graphs.yaml");
        std::fs::write(&path, yaml).unwrap();
        let err = Config::from_yaml_file(&path).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("use_leakance") && msg.contains("use_cuda_graphs"),
            "expected leakance/graphs conflict, got: {msg}"
        );
    }

    #[test]
    fn state_cache_absent_yields_none() {
        // experiment block without state_cache → None (critical byte-identity invariant).
        let exp: Experiment = serde_yaml::from_str(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\n",
        )
        .expect("parse experiment");
        assert!(exp.state_cache.is_none(), "state_cache must default to None");
    }

    #[test]
    fn state_cache_path_parses() {
        let exp: Experiment = serde_yaml::from_str(
            "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
             epochs: 1\nrho: 10\nwarmup: 1\n\
             state_cache: /tmp/state_cache.nc\n",
        )
        .expect("parse experiment with state_cache");
        assert_eq!(
            exp.state_cache.as_deref(),
            Some(std::path::Path::new("/tmp/state_cache.nc"))
        );
    }

    // ── attributes: one-or-many deserialization ──────────────────────────────

    #[test]
    fn attributes_bare_path_parses_as_single_element_vec() {
        let ds_block = r#"
data_sources:
  attributes: /dev/null/attrs.nc
  geospatial_fabric: /dev/null/rivers.shp
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_attrs_bare_path.yaml", ds_block);
        let cfg = Config::from_yaml_file(&path).expect("bare path should parse");
        let ds = cfg.data_sources.as_ref().unwrap();
        assert_eq!(ds.attributes.len(), 1);
        assert_eq!(ds.attributes[0], std::path::PathBuf::from("/dev/null/attrs.nc"));
    }

    #[test]
    fn attributes_yaml_list_parses_as_multi_element_vec() {
        let ds_block = r#"
data_sources:
  attributes:
    - /dev/null/global_attrs.nc
    - /dev/null/channel_attrs.nc
  geospatial_fabric: /dev/null/rivers.shp
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_attrs_list.yaml", ds_block);
        let cfg = Config::from_yaml_file(&path).expect("list should parse");
        let ds = cfg.data_sources.as_ref().unwrap();
        assert_eq!(ds.attributes.len(), 2);
        assert_eq!(ds.attributes[0], std::path::PathBuf::from("/dev/null/global_attrs.nc"));
        assert_eq!(ds.attributes[1], std::path::PathBuf::from("/dev/null/channel_attrs.nc"));
    }

    #[test]
    fn attributes_empty_list_rejected() {
        let ds_block = r#"
data_sources:
  attributes: []
  geospatial_fabric: /dev/null/rivers.shp
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
"#;
        let path = write_yaml_with_data_sources("ddrs_attrs_empty.yaml", ds_block);
        let err = Config::from_yaml_file(&path).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("must not be empty") || msg.contains("attributes"),
            "expected empty-list error, got: {msg}"
        );
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
