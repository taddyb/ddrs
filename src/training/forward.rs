//! One forward pass — direct-param path (verification) and MLP path
//! (training). Plus the scatter_add-by-group helper that turns the MC
//! engine's `(N, T)` output into per-gauge `(G, T)` via the
//! `outflow_idx`-derived `flat_indices` + `group_ids`.

use burn::backend::Autodiff;
use burn::tensor::{backend::Backend, IndexingUpdateOp, Int, Tensor, TensorData};

use crate::config::Config;
use crate::data::dataset::RoutingTensors;
use crate::routing::mmc::{MuskingumCunge, RoutingInputs, SpatialParameters};
use crate::routing::utils::denormalize;

/// Gather + grouped sum: `output[g, t] = sum_{k : group_ids[k] == g} runoff[flat_indices[k], t]`.
///
/// Mirrors DDR `~/projects/ddr/src/ddr/routing/mmc.py:401-410`. Used to
/// extract per-gauge predictions from the engine's all-segments output.
pub fn scatter_add_by_group<B: Backend>(
    runoff: Tensor<B, 2>,             // (N, T)
    flat_indices: Tensor<B, 1, Int>,  // (K,)
    group_ids: Tensor<B, 1, Int>,     // (K,)
    num_gauges: usize,
) -> Tensor<B, 2> {
    // 1. Gather rows: (K, T).
    let gathered = runoff.select(0, flat_indices);
    let [k, t] = gathered.dims();

    // 2. Expand group_ids from (K,) to (K, T) so scatter indices match values shape.
    let group_2d = group_ids.unsqueeze_dim::<2>(1).expand([k, t]);

    // 3. Scatter-add into (G, T) output.
    let zeros = Tensor::<B, 2>::zeros([num_gauges, t], &gathered.device());
    zeros.scatter(0, group_2d, gathered, IndexingUpdateOp::Add)
}

// ---------------------------------------------------------------------------
// FrozenParams + forward_with_frozen_params
// ---------------------------------------------------------------------------

/// Scalar constants applied uniformly across every reach. Used for V1/V2
/// verification tests. **The numeric values are mirrored in
/// `scripts/dump_ddr_loss.py` — keep both in sync.**
pub struct FrozenParams {
    pub n: Vec<f32>,         // length N
    pub q_spatial: Vec<f32>, // length N
    pub p_spatial: Vec<f32>, // length N
}

/// V1/V2 verification constants. Uniform across all reaches.
pub const FROZEN_N: f32 = 0.05;
pub const FROZEN_Q_SPATIAL: f32 = 0.5;
pub const FROZEN_P_SPATIAL: f32 = 21.0;

impl FrozenParams {
    pub fn constant(n_reaches: usize) -> Self {
        Self {
            n:         std::iter::repeat_n(FROZEN_N,         n_reaches).collect(),
            q_spatial: std::iter::repeat_n(FROZEN_Q_SPATIAL, n_reaches).collect(),
            p_spatial: std::iter::repeat_n(FROZEN_P_SPATIAL, n_reaches).collect(),
        }
    }
}

/// Inverse of `denormalize` in `src/routing/utils.rs:32-43`.
///
/// Converts a physical parameter value back to normalized [0, 1] so it can
/// be fed into `MuskingumCunge::setup_inputs`. Must exactly mirror the
/// `+1e-6` epsilon used in `denormalize`'s log-space branch.
///
/// `pub` (not `pub(crate)`): callers in `src/bin/*.rs` are separate crates
/// that depend on `ddrs` externally, so crate-visibility isn't enough.
pub fn physical_to_normalized(values: &[f32], range: [f32; 2], log_space: bool) -> Vec<f32> {
    let [lo, hi] = range;
    if log_space {
        let log_lo = (lo + 1e-6).ln(); // matches denormalize's epsilon
        let log_hi = hi.ln();
        values
            .iter()
            .map(|&v| (v.ln() - log_lo) / (log_hi - log_lo))
            .collect()
    } else {
        values
            .iter()
            .map(|&v| (v - lo) / (hi - lo))
            .collect()
    }
}

/// Direct-param forward pass for V1/V2 verification. No MLP, no autograd
/// retention. Takes frozen physical parameters, runs the MC engine over the
/// full window, and returns per-gauge hourly predictions `(num_gauges, T_hours)`.
///
/// Parameterized on the *inner* backend `I` (e.g. `NdArray<f32>` or `LibTorch<f32>`).
/// `tensors` must have been built on backend `I`; the engine runs on `Autodiff<I>`
/// internally and `.inner()` strips the graph before the scatter call.
///
/// `engine.forward()` returns `(N, T_hours)` (segment × time); after stripping
/// autograd with `.inner()` we scatter-add by gauge group to `(G, T_hours)`.
pub fn forward_with_frozen_params<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    frozen: &FrozenParams,
    device: &I::Device,
    carry_state: bool,
) -> Tensor<I, 2> {
    let n_active = tensors.adjacency.n;
    let ranges = &cfg.params.parameter_ranges;
    let log_space = &cfg.params.log_space_parameters;

    // Normalize physical values → [0, 1] using the exact inverse of `denormalize`.
    let n_norm = physical_to_normalized(
        &frozen.n, ranges.n, log_space.iter().any(|s| s == "n"),
    );
    let q_norm = physical_to_normalized(
        &frozen.q_spatial, ranges.q_spatial, log_space.iter().any(|s| s == "q_spatial"),
    );
    let p_norm = physical_to_normalized(
        &frozen.p_spatial, ranges.p_spatial, log_space.iter().any(|s| s == "p_spatial"),
    );

    // Wrap as Autodiff<I> tensors (inner backend's device == Autodiff device).
    let n_t: Tensor<Autodiff<I>, 1> = Tensor::from_floats(n_norm.as_slice(), device);
    let q_t: Tensor<Autodiff<I>, 1> = Tensor::from_floats(q_norm.as_slice(), device);
    let p_t: Tensor<Autodiff<I>, 1> = Tensor::from_floats(p_norm.as_slice(), device);
    let x_storage: Tensor<Autodiff<I>, 1> = Tensor::full([n_active], 0.3_f32, device);

    // Wrap q_prime (Tensor<I,2>) into Tensor<Autodiff<I>,2> via from_inner —
    // the clean BURN 0.21 API for promoting a plain tensor to the autodiff backend.
    let q_prime_autodiff: Tensor<Autodiff<I>, 2> =
        Tensor::from_inner(tensors.q_prime.clone());

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    // Lift initial_state from inner backend to Autodiff (same pattern as q_prime).
    let initial_state_ad = tensors.initial_state.clone().map(Tensor::<Autodiff<I>, 1>::from_inner);
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage },
        q_prime_autodiff,
        SpatialParameters {
            n: n_t,
            q_spatial: q_t,
            p_spatial: Some(p_t),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
        },
        carry_state,
        initial_state_ad,
    );

    // engine.forward() → (N, T_hours) on Autodiff<I>.
    // Drop autograd graph immediately — this is a verification path with no backward.
    let runoff: Tensor<I, 2> = engine.forward().inner();

    // Scatter-add (N, T_hours) → (G, T_hours).
    scatter_add_by_group(
        runoff,
        tensors.flat_indices.clone(),
        tensors.group_ids.clone(),
        tensors.num_gauges,
    )
}

// ---------------------------------------------------------------------------
// MLP-integrated production forward
// ---------------------------------------------------------------------------

use crate::nn::kan_head::KanHead;

/// Summary of the physical Manning's `n` the head currently emits for this
/// batch's network: `(median, fraction within 1% of the range floor)`.
///
/// Diagnostic only. Runs the head a second time and reads the result straight
/// back to the host, so nothing survives on the autograd tape and the training
/// gradient is untouched. Cost is one KAN forward over the micro-batch's
/// reaches (milliseconds) against a ~27 s routing forward — it does not move
/// the wall clock.
///
/// Why this exists: the loss curve cannot distinguish a healthy run from one
/// whose `n` field has collapsed onto the lower bound of
/// `params.parameter_ranges.n`, where the sigmoid derivative vanishes and
/// those reaches stop learning for good. The 2026-08-01 `lr: 0.01` run ended
/// with 47% of CONUS pinned at `n = 0.015`, and that was only visible after a
/// 14 h run plus a `dump_parameters` pass. Logging it per micro-batch turns a
/// post-mortem into a first-epoch observation.
pub fn manning_n_stats<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<Autodiff<I>>,
    head: &KanHead<Autodiff<I>>,
) -> (f32, f32) {
    let params_map = head.forward(tensors.spatial_attributes.clone());
    let n_norm = params_map.get("n").expect("KAN head missing n").clone();
    let n_phys = denormalize(
        n_norm,
        cfg.params.parameter_ranges.n,
        cfg.params.log_space_parameters.iter().any(|s| s == "n"),
    )
    .detach();

    let mut vals: Vec<f32> = n_phys.into_data().into_vec().unwrap();
    vals.retain(|v| v.is_finite());
    if vals.is_empty() {
        return (f32::NAN, f32::NAN);
    }

    let [lo, hi] = cfg.params.parameter_ranges.n;
    // "At the floor" = within 1% of the declared span. Matches the threshold
    // the parameter-map notebooks use, so log and plot agree.
    let floor_tol = lo + 0.01 * (hi - lo);
    let at_floor = vals.iter().filter(|&&v| v <= floor_tol).count() as f32 / vals.len() as f32;

    vals.sort_unstable_by(|a, b| a.partial_cmp(b).expect("finite"));
    let mid = vals.len() / 2;
    let median = if vals.len() % 2 == 0 {
        0.5 * (vals[mid - 1] + vals[mid])
    } else {
        vals[mid]
    };
    (median, at_floor)
}

/// One training-step forward pass. Computes MLP outputs from normalized
/// attributes, denormalizes through the engine's `setup_inputs`, runs MC,
/// and scatter-adds to per-gauge predictions. Returns `(num_gauges, T_hours)`
/// with autograd alive on the engine path.
///
/// Mirrors `~/projects/ddr/scripts/train.py:67-73` (MLP forward + dmc forward).
pub fn forward<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<Autodiff<I>>,
    head: &KanHead<Autodiff<I>>,
    device: &I::Device,
    carry_state: bool,
) -> Tensor<Autodiff<I>, 2> {
    let params_map = head.forward(tensors.spatial_attributes.clone());

    let n_param = params_map.get("n").expect("MLP missing n").clone();
    let q_param = params_map.get("q_spatial").expect("MLP missing q_spatial").clone();
    let p_param = params_map.get("p_spatial").cloned();

    let n_active = tensors.adjacency.n;
    // Learnable Muskingum X: when the KAN emits `x_storage`, denormalize its
    // [0,1] output to the configured range so the routing learns its own
    // attenuation-vs-translation per reach (gradient already flows via the
    // custom sparse backward in mmc_op.rs). Otherwise hold the constant 0.3.
    let x_storage: Tensor<Autodiff<I>, 1> = match params_map.get("x_storage") {
        Some(x_norm) => denormalize(
            x_norm.clone(),
            cfg.params.parameter_ranges.x_storage,
            cfg.params.log_space_parameters.iter().any(|s| s == "x_storage"),
        ),
        None => Tensor::full([n_active], 0.3_f32, device),
    };

    // Forcing: learnable mass-preserving disaggregation of the daily Q' when a
    // disagg head is attached, else the flat repeat-24 already in `q_prime`.
    let n_hourly = tensors.q_prime.dims()[0];
    let q_prime_hourly = match &head.disagg {
        Some(d) => d.forward(tensors.q_prime_daily.clone(), tensors.precip_hourly.clone(), n_hourly),
        None => tensors.q_prime.clone(),
    };

    let (k_d, d_gw, leakance_factor) = if cfg.params.use_leakance {
        for key in &["K_D", "d_gw", "leakance_factor"] {
            if !params_map.contains_key(*key) {
                panic!(
                    "use_leakance=true but KAN head is missing key '{key}' — \
                     add it to kan_head.learnable_parameters in your config"
                );
            }
        }
        (
            params_map.get("K_D").cloned(),
            params_map.get("d_gw").cloned(),
            params_map.get("leakance_factor").cloned(),
        )
    } else {
        (None, None, None)
    };

    // Extract the inner-backend mask (no grad) from the Autodiff tensor.
    // `None` when the attribute was absent or leakance is off — byte-identical
    // to the pre-mask behavior.
    let impervious_mask: Option<Tensor<I, 1>> =
        tensors.impervious_mask.as_ref().map(|m| m.clone().inner());

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage },
        q_prime_hourly,
        SpatialParameters {
            n: n_param,
            q_spatial: q_param,
            p_spatial: p_param,
            k_d,
            d_gw,
            leakance_factor,
            impervious_mask,
        },
        carry_state,
        tensors.initial_state.clone(),
    );

    let runoff = engine.forward(); // (N, T_hours)

    // Scatter-add (N, T_hours) → (G, T_hours) with autograd alive.
    scatter_add_by_group(
        runoff,
        tensors.flat_indices.clone(),
        tensors.group_ids.clone(),
        tensors.num_gauges,
    )
}

/// Running zeta accumulation across chunked `forward_eval` calls (eval builds
/// a fresh engine per chunk, so the sums merge here). `steps` counts routed
/// timesteps; mean |zeta| per reach = `abs_sum / steps`.
pub struct ZetaSums<I: Backend> {
    pub abs_sum: Option<Tensor<I, 1>>,
    pub net_sum: Option<Tensor<I, 1>>,
    pub depth_sum: Option<Tensor<I, 1>>,
    pub area_z_sum: Option<Tensor<I, 1>>,
    pub q_sum: Option<Tensor<I, 1>>,
    pub steps: usize,
}

impl<I: Backend> ZetaSums<I> {
    pub fn new() -> Self {
        Self {
            abs_sum: None,
            net_sum: None,
            depth_sum: None,
            area_z_sum: None,
            q_sum: None,
            steps: 0,
        }
    }

    fn merge(&mut self, sums: crate::routing::ZetaSumTensors<I>) {
        fn add<I: Backend>(slot: &mut Option<Tensor<I, 1>>, v: Tensor<I, 1>) {
            *slot = Some(match slot.take() {
                Some(s) => s + v,
                None => v,
            });
        }
        add(&mut self.abs_sum, sums.abs);
        add(&mut self.net_sum, sums.net);
        add(&mut self.depth_sum, sums.depth);
        add(&mut self.area_z_sum, sums.area_z);
        add(&mut self.q_sum, sums.q);
        self.steps += sums.steps;
    }
}

impl<I: Backend> Default for ZetaSums<I> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-reach override of the NORMALIZED leakance head outputs, applied inside
/// `forward_eval` between `head.forward` and denormalization (which happens
/// in `setup_inputs`). Vectors are dense over the network's reach columns
/// (same order as `divide_comids`): `mask[i] == 1.0` replaces reach i's
/// normalized K_D/d_gw/factor with the corresponding value; `mask[i] == 0.0`
/// leaves the head's output untouched. Eval-path only — the training
/// `forward` never sees this type.
pub struct LeakanceOverride {
    pub mask: Vec<f32>,
    pub k_d: Vec<f32>,
    pub d_gw: Vec<f32>,
    pub factor: Vec<f32>,
}

impl LeakanceOverride {
    fn apply<I: Backend>(
        &self,
        param: Tensor<I, 1>,
        vals: &[f32],
        device: &I::Device,
    ) -> Tensor<I, 1> {
        debug_assert_eq!(
            vals.len(),
            self.mask.len(),
            "override vals length {} != mask length {}",
            vals.len(),
            self.mask.len()
        );
        let n = self.mask.len();
        let mask_t: Tensor<I, 1> =
            Tensor::from_data(TensorData::new(self.mask.clone(), [n]), device);
        let vals_t: Tensor<I, 1> =
            Tensor::from_data(TensorData::new(vals.to_vec(), [n]), device);
        param * (mask_t.ones_like() - mask_t.clone()) + vals_t * mask_t
    }
}

/// Per-reach override of a subset of the NORMALIZED `n`/`q_spatial`/
/// `p_spatial` head outputs, applied inside `forward_eval_core` right after
/// `head.forward` — same seam as `LeakanceOverride`, but a WHOLE-BATCH
/// replace rather than a masked blend: every populated field replaces every
/// reach in the batch's column order (same order as `divide_comids`);
/// `None` passes the head's own output through unchanged. Built for the
/// H5/H6 selective-equifinality parameter-swap tests
/// (`docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md`)
/// — eval-path only, the training `forward` never sees this type.
#[derive(Default)]
pub struct RoutingParamOverride {
    pub n: Option<Vec<f32>>,
    pub q_spatial: Option<Vec<f32>>,
    pub p_spatial: Option<Vec<f32>>,
}

impl RoutingParamOverride {
    /// Replace `param` wholesale with `vals` when `Some`, else pass through.
    /// Panics (not a silent no-op) on a length mismatch — a mismatched
    /// override vector means the donor NetCDF's gather-by-COMID produced the
    /// wrong reach count for this batch, which would silently corrupt the
    /// routing solve if allowed through.
    fn replace_or_passthrough<I: Backend>(
        param: Tensor<I, 1>,
        vals: &Option<Vec<f32>>,
        device: &I::Device,
    ) -> Tensor<I, 1> {
        match vals {
            Some(v) => {
                assert_eq!(
                    v.len(),
                    param.dims()[0],
                    "RoutingParamOverride length {} != batch reach count {}",
                    v.len(),
                    param.dims()[0]
                );
                Tensor::from_data(TensorData::new(v.clone(), [v.len()]), device)
            }
            None => param,
        }
    }
}

/// MLP inference forward — no autograd anywhere. Used by `bin/eval` and
/// the KanHead arm of `EvalParams`.
///
/// Mirrors `forward` (production training path) but operates on the inner
/// backend `I` throughout. Caller passes an `KanHead<I>` loaded via
/// `checkpoint::load_kan_head`.
///
/// `zeta` — optional leakance-diagnostic sink; when `Some` and leakance is
/// active, the engine's per-timestep zeta sums are merged into it.
///
/// `overrides` — optional per-reach override of the NORMALIZED leakance head
/// outputs; applied after `head.forward` and before denormalization. `None`
/// is the normal eval path (no override).
///
/// `param_overrides` — optional per-reach override of a subset of the
/// NORMALIZED `n`/`q_spatial`/`p_spatial` head outputs (see
/// `RoutingParamOverride`); `None` is the normal eval path (no override).
/// Orthogonal to `overrides`/leakance and to the disaggregation head — both
/// are computed independently of `n`/`q_spatial`/`p_spatial`.
pub fn forward_eval<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    head: &KanHead<I>,
    device: &I::Device,
    carry_state: bool,
    zeta: Option<&mut ZetaSums<I>>,
    overrides: Option<&LeakanceOverride>,
    param_overrides: Option<&RoutingParamOverride>,
) -> Tensor<I, 2> {
    let runoff = forward_eval_core(
        cfg,
        tensors,
        head,
        device,
        carry_state,
        zeta,
        overrides,
        param_overrides,
    );
    scatter_add_by_group(
        runoff,
        tensors.flat_indices.clone(),
        tensors.group_ids.clone(),
        tensors.num_gauges,
    )
}

/// Like `forward_eval` but returns per-reach `(n_reaches, T_hours)` before
/// gauge aggregation. Used by the state-cache writer to capture day-boundary
/// discharge states and by the teacher-obs writer, which needs the per-reach
/// final column for cross-chunk state injection (gauge aggregation happens at
/// the call site via `scatter_add_by_group`).
pub fn forward_eval_reaches<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    head: &KanHead<I>,
    device: &I::Device,
    carry_state: bool,
    zeta: Option<&mut ZetaSums<I>>,
    overrides: Option<&LeakanceOverride>,
    param_overrides: Option<&RoutingParamOverride>,
) -> Tensor<I, 2> {
    forward_eval_core(
        cfg,
        tensors,
        head,
        device,
        carry_state,
        zeta,
        overrides,
        param_overrides,
    )
}

/// Shared body of `forward_eval` and `forward_eval_reaches`. Returns
/// per-reach `(n_reaches, T_hours)` on the inner backend before the
/// `scatter_add_by_group` that produces gauge-aggregated output.
fn forward_eval_core<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    head: &KanHead<I>,
    device: &I::Device,
    carry_state: bool,
    zeta: Option<&mut ZetaSums<I>>,
    overrides: Option<&LeakanceOverride>,
    param_overrides: Option<&RoutingParamOverride>,
) -> Tensor<I, 2> {
    let params_map = head.forward(tensors.spatial_attributes.clone());

    let n_param = params_map.get("n").expect("MLP missing n").clone();
    let q_param = params_map.get("q_spatial").expect("MLP missing q_spatial").clone();
    let p_param = params_map.get("p_spatial").cloned();

    // H5/H6 parameter-swap override: whole-batch replace of n/q_spatial/
    // p_spatial (normalized space), independent of leakance/disagg below.
    let (n_param, q_param, p_param) = if let Some(po) = param_overrides {
        let n_param = RoutingParamOverride::replace_or_passthrough(n_param, &po.n, device);
        let q_param = RoutingParamOverride::replace_or_passthrough(q_param, &po.q_spatial, device);
        let p_param = match (p_param, &po.p_spatial) {
            (Some(p), ov) => Some(RoutingParamOverride::replace_or_passthrough(p, ov, device)),
            (None, Some(_)) => panic!(
                "RoutingParamOverride requested a p_spatial override but this KAN head has no \
                 p_spatial output — add p_spatial to kan_head.learnable_parameters"
            ),
            (None, None) => None,
        };
        (n_param, q_param, p_param)
    } else {
        (n_param, q_param, p_param)
    };

    let n_active = tensors.adjacency.n;
    // Learnable Muskingum X (eval path mirrors `forward`): denormalize the
    // KAN's `x_storage` output when present, else the constant 0.3.
    let x_storage: Tensor<I, 1> = match params_map.get("x_storage") {
        Some(x_norm) => denormalize(
            x_norm.clone(),
            cfg.params.parameter_ranges.x_storage,
            cfg.params.log_space_parameters.iter().any(|s| s == "x_storage"),
        ),
        None => Tensor::full([n_active], 0.3_f32, device),
    };

    // Forcing (inner backend): disaggregate when a head is attached, else the
    // flat repeat-24. Mirrors `forward`.
    let n_hourly = tensors.q_prime.dims()[0];
    // Forcing: disaggregate when a head is attached (mirrors `forward`). Each
    // day is disaggregated independently from its own (daily Q', 24h precip)
    // — no cross-day lookahead/window, so no special handling is needed for
    // the teacher/state-cache path's lookahead-batch shape here.
    let q_prime_hourly: Tensor<I, 2> = match &head.disagg {
        Some(d) => d.forward(tensors.q_prime_daily.clone(), tensors.precip_hourly.clone(), n_hourly),
        None => tensors.q_prime.clone(),
    };

    let (k_d_inner, d_gw_inner, leakance_factor_inner) = if cfg.params.use_leakance {
        for key in &["K_D", "d_gw", "leakance_factor"] {
            if !params_map.contains_key(*key) {
                panic!(
                    "use_leakance=true but KAN head is missing key '{key}' — \
                     add it to kan_head.learnable_parameters in your config"
                );
            }
        }
        let (mut k_d_t, mut d_gw_t, mut factor_t) = (
            params_map.get("K_D").expect("checked above").clone(),
            params_map.get("d_gw").expect("checked above").clone(),
            params_map.get("leakance_factor").expect("checked above").clone(),
        );
        if let Some(ov) = overrides {
            assert_eq!(
                ov.mask.len(),
                k_d_t.dims()[0],
                "LeakanceOverride mask length doesn't match network reaches"
            );
            k_d_t = ov.apply::<I>(k_d_t, &ov.k_d, device);
            d_gw_t = ov.apply::<I>(d_gw_t, &ov.d_gw, device);
            factor_t = ov.apply::<I>(factor_t, &ov.factor, device);
        }
        (Some(k_d_t), Some(d_gw_t), Some(factor_t))
    } else {
        (None, None, None)
    };

    // Wrap to Autodiff at the engine boundary (engine requires Autodiff
    // even for forward-only). Drop the graph immediately after with .inner().
    let q_prime_ad: Tensor<Autodiff<I>, 2> = Tensor::from_inner(q_prime_hourly);
    let n_ad = Tensor::<Autodiff<I>, 1>::from_inner(n_param);
    let q_ad = Tensor::<Autodiff<I>, 1>::from_inner(q_param);
    let p_ad = p_param.map(Tensor::<Autodiff<I>, 1>::from_inner);
    let x_ad = Tensor::<Autodiff<I>, 1>::from_inner(x_storage);
    let k_d_ad = k_d_inner.map(Tensor::<Autodiff<I>, 1>::from_inner);
    let d_gw_ad = d_gw_inner.map(Tensor::<Autodiff<I>, 1>::from_inner);
    let leakance_factor_ad = leakance_factor_inner.map(Tensor::<Autodiff<I>, 1>::from_inner);

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    // Lift initial_state from inner backend to Autodiff (same pattern as q_prime_ad).
    let initial_state_ad = tensors.initial_state.clone().map(Tensor::<Autodiff<I>, 1>::from_inner);
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage: x_ad },
        q_prime_ad,
        SpatialParameters {
            n: n_ad,
            q_spatial: q_ad,
            p_spatial: p_ad,
            k_d: k_d_ad,
            d_gw: d_gw_ad,
            leakance_factor: leakance_factor_ad,
            impervious_mask: tensors.impervious_mask.clone(),
        },
        carry_state,
        initial_state_ad,
    );
    if zeta.is_some() {
        engine.enable_zeta_accumulation();
    }
    let runoff_ad = engine.forward();
    let runoff = runoff_ad.inner();
    if let Some(sink) = zeta {
        if let Some(sums) = engine.zeta_sums() {
            sink.merge(sums);
        }
    }

    runoff
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;
    type Dev = <B as burn::tensor::backend::BackendTypes>::Device;

    #[test]
    fn physical_to_normalized_round_trips_through_denormalize() {
        use crate::routing::utils::denormalize;
        type AB = Autodiff<NdArray<f32>>;

        let device = <AB as burn::tensor::backend::BackendTypes>::Device::default();

        // Linear range (n parameter).
        let range = [0.015_f32, 0.25];
        let physical = vec![0.05_f32];
        let norm = physical_to_normalized(&physical, range, false);
        let norm_t: Tensor<AB, 1> = Tensor::from_floats(norm.as_slice(), &device);
        let denorm = denormalize(norm_t, range, false);
        let recovered: Vec<f32> = denorm.into_data().into_vec().unwrap();
        assert!(
            (recovered[0] - 0.05).abs() < 1e-6,
            "linear round-trip failed: {} != 0.05",
            recovered[0]
        );

        // Log-space range (p_spatial). p_spatial range is [1.0, 200.0]; the
        // +1e-6 epsilon barely matters here but the test still verifies the
        // formula matches denormalize's branch exactly.
        let p_range = [1.0_f32, 200.0];
        let p_physical = vec![21.0_f32];
        let p_norm = physical_to_normalized(&p_physical, p_range, true);
        let p_norm_t: Tensor<AB, 1> = Tensor::from_floats(p_norm.as_slice(), &device);
        let p_denorm = denormalize(p_norm_t, p_range, true);
        let p_recovered: Vec<f32> = p_denorm.into_data().into_vec().unwrap();
        assert!(
            (p_recovered[0] - 21.0).abs() < 1e-4,
            "log round-trip failed: {} != 21.0",
            p_recovered[0]
        );
    }

    #[test]
    fn leakance_override_apply_replaces_masked_entries_only() {
        let device = <B as burn::tensor::backend::BackendTypes>::Device::default();
        let ov = LeakanceOverride {
            mask: vec![0.0, 1.0, 0.0, 1.0],
            k_d: vec![0.0, 1.0, 0.0, 0.5],
            d_gw: vec![0.0, 0.0, 0.0, 0.25],
            factor: vec![0.0, 0.9, 0.0, 0.1],
        };
        let param: Tensor<B, 1> =
            Tensor::from_floats([0.7, 0.7, 0.7, 0.7], &device);
        let out: Vec<f32> = ov
            .apply(param.clone(), &ov.k_d, &device)
            .into_data()
            .into_vec()
            .unwrap();
        assert_eq!(out, vec![0.7, 1.0, 0.7, 0.5]);
        let out_f: Vec<f32> = ov
            .apply(param, &ov.factor, &device)
            .into_data()
            .into_vec()
            .unwrap();
        assert_eq!(out_f, vec![0.7, 0.9, 0.7, 0.1]);

        // d_gw: masked positions (1, 3) take the d_gw values; unmasked stay 0.7.
        let param2: Tensor<B, 1> = Tensor::from_floats([0.7, 0.7, 0.7, 0.7], &device);
        let out_dgw: Vec<f32> = ov
            .apply(param2, &ov.d_gw, &device)
            .into_data()
            .into_vec()
            .unwrap();
        assert_eq!(out_dgw, vec![0.7, 0.0, 0.7, 0.25]);

        // All-zero mask is identity: output equals input param exactly.
        let ov_zero = LeakanceOverride {
            mask: vec![0.0; 4],
            k_d: vec![1.0, 2.0, 3.0, 4.0],
            d_gw: vec![1.0, 2.0, 3.0, 4.0],
            factor: vec![1.0, 2.0, 3.0, 4.0],
        };
        let param3: Tensor<B, 1> = Tensor::from_floats([0.1, 0.2, 0.3, 0.4], &device);
        let out_id: Vec<f32> = ov_zero
            .apply(param3, &ov_zero.k_d, &device)
            .into_data()
            .into_vec()
            .unwrap();
        assert_eq!(out_id, vec![0.1, 0.2, 0.3, 0.4]);
    }

    // -----------------------------------------------------------------------
    // RoutingParamOverride (H5/H6 parameter-swap) — unit test + parity test
    // -----------------------------------------------------------------------

    #[test]
    fn routing_param_override_replace_or_passthrough() {
        let device = Dev::default();
        let param: Tensor<B, 1> = Tensor::from_floats([0.1, 0.2, 0.3], &device);

        // None passes the input straight through, unchanged.
        let passthrough: Vec<f32> =
            RoutingParamOverride::replace_or_passthrough(param.clone(), &None, &device)
                .into_data()
                .into_vec()
                .unwrap();
        assert_eq!(passthrough, vec![0.1, 0.2, 0.3]);

        // Some(vals) replaces every entry (whole-batch, not masked).
        let replaced: Vec<f32> = RoutingParamOverride::replace_or_passthrough(
            param,
            &Some(vec![0.9, 0.8, 0.7]),
            &device,
        )
        .into_data()
        .into_vec()
        .unwrap();
        assert_eq!(replaced, vec![0.9, 0.8, 0.7]);
    }

    #[test]
    #[should_panic(expected = "RoutingParamOverride length")]
    fn routing_param_override_length_mismatch_panics() {
        let device = Dev::default();
        let param: Tensor<B, 1> = Tensor::from_floats([0.1, 0.2, 0.3], &device);
        RoutingParamOverride::replace_or_passthrough(param, &Some(vec![0.9, 0.8]), &device);
    }

    /// Build a minimal `RoutingTensors<B>` (inner backend, no Autodiff) for a
    /// linear chain of `n` reaches with `t` hourly steps and a single gauge
    /// at the outlet. Mirrors `tests/leakance_off_parity.rs::minimal_routing_tensors`
    /// but on the inner backend, matching `forward_eval`'s signature.
    fn minimal_eval_tensors(n: usize, t: usize, f_attrs: usize, device: &Dev) -> RoutingTensors<B> {
        use crate::data::{RhoWindow, Staid};
        use crate::sparse::SparseAdjacency;
        use chrono::NaiveDate;

        let adjacency = {
            let mut dense = vec![0.0_f32; n * n];
            for i in 0..n - 1 {
                dense[(i + 1) * n + i] = 1.0;
            }
            SparseAdjacency::from_dense(n, &dense, vec![1000.0; n], vec![0.001; n])
        };

        let attrs_vec = vec![0.5_f32; n * f_attrs];
        let spatial_attributes =
            Tensor::<B, 2>::from_data(TensorData::new(attrs_vec, [n, f_attrs]), device);

        let mut qp_data = vec![0.0_f32; t * n];
        for ti in 0..t {
            let phase = (ti as f32) / (t.max(2) - 1) as f32 * 4.0 * std::f32::consts::PI;
            for ri in 0..n {
                qp_data[ti * n + ri] = (5.0 + phase.sin() * 2.0).max(0.1);
            }
        }
        let q_prime = Tensor::<B, 2>::from_data(TensorData::new(qp_data, [t, n]), device);

        let q_prime_daily =
            Tensor::<B, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, n]), device);
        let precip_hourly =
            Tensor::<B, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, n]), device);
        let temp_hourly =
            Tensor::<B, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, n]), device);

        let flat_indices = Tensor::<B, 1, Int>::from_data(
            TensorData::from([(n - 1) as i32].as_slice()),
            device,
        );
        let group_ids =
            Tensor::<B, 1, Int>::from_data(TensorData::from([0i32].as_slice()), device);

        RoutingTensors {
            adjacency,
            spatial_attributes,
            q_prime,
            q_prime_daily,
            precip_hourly,
            temp_hourly,
            observations: ndarray::Array2::zeros((1, 1)),
            flat_indices,
            group_ids,
            num_gauges: 1,
            gauge_staids: vec![Staid::new("dummy")],
            window: RhoWindow {
                start_day_idx: 0,
                rho_days: 1,
                window_start: NaiveDate::from_ymd_opt(1990, 1, 1).unwrap(),
            },
            initial_state: None,
            impervious_mask: None,
        }
    }

    /// Build a KAN head with `learnable_parameters = [n, q_spatial, p_spatial]`
    /// on the inner backend (no leakance, no disagg) — matches `forward_eval`'s
    /// `head: &KanHead<I>` signature.
    fn minimal_eval_head(f_attrs: usize, device: &Dev) -> KanHead<B> {
        use crate::nn::kan_head::KanHeadConfig;
        KanHeadConfig::new(
            (0..f_attrs).map(|i| format!("attr_{i}")).collect(),
            vec!["n".to_string(), "q_spatial".to_string(), "p_spatial".to_string()],
            42,
        )
        .with_hidden_size(8)
        .with_num_hidden_layers(1)
        .init::<B>(device)
    }

    /// The parity gate H5/H6 must pass before any swap result is trusted:
    /// round-tripping a head's own `n`/`q_spatial`/`p_spatial` output through
    /// denormalize -> `physical_to_normalized` -> `RoutingParamOverride` (the
    /// exact path a real donor-NetCDF injection takes) must reproduce the
    /// un-overridden forward to the same f32 floor already established by
    /// `compare_ddr_sandbox` (CLAUDE.md invariant 1, < 1e-3 m3/s absolute).
    #[test]
    fn routing_param_override_own_dump_round_trip_matches_baseline() {
        let device = Dev::default();
        let n = 5usize;
        let t = 24usize;
        let f = 4usize;

        let tensors = minimal_eval_tensors(n, t, f, &device);
        let head = minimal_eval_head(f, &device);
        let mut cfg = Config::default();
        cfg.params.parameter_ranges.n = [0.01, 0.25];
        cfg.params.parameter_ranges.q_spatial = [0.1, 0.9];
        cfg.params.parameter_ranges.p_spatial = [1.0, 200.0];

        // Baseline: no override.
        let baseline = forward_eval::<B>(&cfg, &tensors, &head, &device, false, None, None, None);
        let baseline_vals: Vec<f32> = baseline.into_data().into_vec().unwrap();

        // Round-trip the head's own normalized output through physical space
        // and back — mirrors dump_parameters::write_netcdf (denormalize) +
        // param_dump::load_comid_field + physical_to_normalized (read side).
        let params_map = head.forward(tensors.spatial_attributes.clone());
        let log_space = |name: &str| cfg.params.log_space_parameters.iter().any(|s| s == name);
        let round_trip = |name: &str, range: [f32; 2]| -> Vec<f32> {
            let norm = params_map.get(name).expect("head missing param").clone();
            let physical = denormalize(norm, range, log_space(name));
            let physical_vals: Vec<f32> = physical.into_data().into_vec().unwrap();
            physical_to_normalized(&physical_vals, range, log_space(name))
        };
        let ov = RoutingParamOverride {
            n: Some(round_trip("n", cfg.params.parameter_ranges.n)),
            q_spatial: Some(round_trip("q_spatial", cfg.params.parameter_ranges.q_spatial)),
            p_spatial: Some(round_trip("p_spatial", cfg.params.parameter_ranges.p_spatial)),
        };

        let overridden =
            forward_eval::<B>(&cfg, &tensors, &head, &device, false, None, None, Some(&ov));
        let overridden_vals: Vec<f32> = overridden.into_data().into_vec().unwrap();

        assert_eq!(baseline_vals.len(), overridden_vals.len());
        let max_abs_diff = baseline_vals
            .iter()
            .zip(overridden_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff < 1e-3,
            "own-dump round-trip diverged from baseline: max abs diff {max_abs_diff} m3/s"
        );
    }

    #[test]
    fn scatter_add_three_gauges_two_groups() {
        // 4 segments × 2 timesteps.
        //   runoff = [[1, 10], [2, 20], [3, 30], [4, 40]]
        // outflow_idx = [[0, 1], [2], [3]]
        //   → flat_indices = [0, 1, 2, 3], group_ids = [0, 0, 1, 2]
        // expected (G=3, T=2): [[1+2=3, 10+20=30], [3, 30], [4, 40]]
        let device = Dev::default();
        let runoff = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], [4, 2]),
            &device,
        );
        let flat = Tensor::<B, 1, Int>::from_data(
            TensorData::from([0i32, 1, 2, 3].as_slice()),
            &device,
        );
        let group = Tensor::<B, 1, Int>::from_data(
            TensorData::from([0i32, 0, 1, 2].as_slice()),
            &device,
        );
        let out = scatter_add_by_group(runoff, flat, group, 3);
        let v: Vec<f32> = out.into_data().into_vec().unwrap();
        assert_eq!(v, vec![3.0, 30.0, 3.0, 30.0, 4.0, 40.0]);
    }
}
