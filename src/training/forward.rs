//! One forward pass — direct-param path (verification) and MLP path
//! (training). Plus the scatter_add-by-group helper that turns the MC
//! engine's `(N, T)` output into per-gauge `(G, T)` via the
//! `outflow_idx`-derived `flat_indices` + `group_ids`.

use burn::backend::Autodiff;
use burn::tensor::{backend::Backend, IndexingUpdateOp, Int, Tensor};

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
fn physical_to_normalized(values: &[f32], range: [f32; 2], log_space: bool) -> Vec<f32> {
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
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage },
        q_prime_autodiff,
        SpatialParameters { n: n_t, q_spatial: q_t, p_spatial: Some(p_t) },
        carry_state,
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
        Some(d) => d.forward(
            tensors.q_prime_daily.clone(),
            tensors.spatial_attributes.clone(),
            tensors.precip_hourly.clone(),
            tensors.temp_hourly.clone(),
            n_hourly,
        ),
        None => tensors.q_prime.clone(),
    };

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage },
        q_prime_hourly,
        SpatialParameters { n: n_param, q_spatial: q_param, p_spatial: p_param },
        carry_state,
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

/// MLP inference forward — no autograd anywhere. Used by `bin/eval` and
/// the KanHead arm of `EvalParams`.
///
/// Mirrors `forward` (production training path) but operates on the inner
/// backend `I` throughout. Caller passes an `KanHead<I>` loaded via
/// `checkpoint::load_kan_head`.
pub fn forward_eval<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<I>,
    head: &KanHead<I>,
    device: &I::Device,
    carry_state: bool,
) -> Tensor<I, 2> {
    let params_map = head.forward(tensors.spatial_attributes.clone());

    let n_param = params_map.get("n").expect("MLP missing n").clone();
    let q_param = params_map.get("q_spatial").expect("MLP missing q_spatial").clone();
    let p_param = params_map.get("p_spatial").cloned();

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
    let q_prime_hourly: Tensor<I, 2> = match &head.disagg {
        Some(d) => d.forward(
            tensors.q_prime_daily.clone(),
            tensors.spatial_attributes.clone(),
            tensors.precip_hourly.clone(),
            tensors.temp_hourly.clone(),
            n_hourly,
        ),
        None => tensors.q_prime.clone(),
    };

    // Wrap to Autodiff at the engine boundary (engine requires Autodiff
    // even for forward-only). Drop the graph immediately after with .inner().
    let q_prime_ad: Tensor<Autodiff<I>, 2> = Tensor::from_inner(q_prime_hourly);
    let n_ad = Tensor::<Autodiff<I>, 1>::from_inner(n_param);
    let q_ad = Tensor::<Autodiff<I>, 1>::from_inner(q_param);
    let p_ad = p_param.map(Tensor::<Autodiff<I>, 1>::from_inner);
    let x_ad = Tensor::<Autodiff<I>, 1>::from_inner(x_storage);

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage: x_ad },
        q_prime_ad,
        SpatialParameters { n: n_ad, q_spatial: q_ad, p_spatial: p_ad },
        carry_state,
    );
    let runoff_ad = engine.forward();
    let runoff = runoff_ad.inner();

    scatter_add_by_group(
        runoff,
        tensors.flat_indices.clone(),
        tensors.group_ids.clone(),
        tensors.num_gauges,
    )
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
