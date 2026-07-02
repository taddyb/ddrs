//! `forward_chain_inner` with `leakance = None` must produce byte-identical
//! `q_next` to the pre-leakance code path. The deterministic linear-chain
//! routing fixture below is captured from the unmodified routing core (the
//! `forward()` of a 5-reach chain over 24 steps with the shared `mock_*`
//! fixtures); after the Phase-3 changes it must reproduce the SAME hydrograph
//! bit-for-bit, proving the leakance-off path is untouched.
//!
//! Also contains a head-driven smoke test (`head_driven_leakance`) verifying
//! that when `use_leakance=true` the head's `K_D/d_gw/leakance_factor` keys
//! are threaded into `SpatialParameters` and change the routed output.

mod common;

use burn::backend::Autodiff;
use burn::tensor::{Int, Tensor, TensorData};
use common::{
    mock_config, mock_routing_inputs, mock_spatial_parameters, mock_streamflow, InnerBackend,
    TestDevice,
};
use ddrs::data::{RhoWindow, Staid};
use ddrs::data::dataset::RoutingTensors;
use ddrs::nn::kan_head::KanHeadConfig;
use ddrs::routing::{MuskingumCunge, SpatialParameters};
use ddrs::sparse::SparseAdjacency;
use ndarray::Array2;

/// Committed expected hydrograph for the 5-reach × 24-step linear chain,
/// captured from the unmodified `forward()` (row-major `[n, t]`). This is the
/// regression guard for the leakance `None` path.
const EXPECTED: [f32; 120] = [
    5.0, 5.0, 6.138457, 6.8463597, 7.0130415, 6.5882154,
    5.7050047, 4.6247187, 3.6582274, 3.076246, 3.0397897, 3.5674682,
    4.5232778, 5.629129, 6.5485415, 7.0054984, 6.86994, 6.1897984,
    5.170149, 4.108406, 3.3059778, 2.9851596, 3.23788, 4.006502,
    10.0, 10.000001, 11.722974, 13.438426, 13.989628, 13.383512,
    11.793261, 9.70426, 7.718887, 6.3883276, 6.071085, 6.868001,
    8.595534, 10.770285, 12.72397, 13.853659, 13.830936, 12.685504,
    10.774359, 8.660961, 6.943698, 6.090408, 6.334672, 7.6389866,
    15.0, 15.0, 17.058432, 19.608543, 20.869438, 20.271181,
    18.17088, 15.175259, 12.1706085, 9.986137, 9.200596, 10.035779,
    12.324955, 15.464205, 18.498262, 20.46777, 20.780603, 19.382988,
    16.730556, 13.618612, 10.931847, 9.395909, 9.416386, 11.025755,
    20.0, 20.0, 22.269648, 25.437702, 27.495834, 27.198418,
    24.744701, 20.985073, 16.993822, 13.897171, 12.501091, 13.176022,
    15.8137665, 19.7648, 23.860197, 26.788347, 27.635294, 26.195568,
    22.968067, 18.942514, 15.274255, 12.95232, 12.574909, 14.276365,
    25.0, 25.0, 27.41432, 31.02724, 33.820404, 34.047234,
    31.466799, 27.064842, 22.167454, 18.13102, 16.023022, 16.371897,
    19.157803, 23.740326, 28.819405, 32.771797, 34.323105, 33.04499,
    29.419636, 24.590364, 19.96302, 16.790401, 15.8782215, 17.484375,
];

/// Build losing-config leakance tensors for `n` reaches on `device`.
/// Normalized values chosen so denormalization produces a strong losing regime:
///   K_D   = 1.0 → top of log range → ~1e-6 m/s/m
///   d_gw  = 0.0 → bottom of linear range [-2, 2] → -2.0 m  (always below any depth)
///   factor = 1.0 → top of linear range [0, 1]  → 1.0 (full gate)
fn losing_config_params(
    n: usize,
    device: &TestDevice,
) -> SpatialParameters<InnerBackend> {
    use burn::backend::Autodiff;
    SpatialParameters {
        n: Tensor::<Autodiff<InnerBackend>, 1>::ones([n], device) * 0.5,
        q_spatial: Tensor::<Autodiff<InnerBackend>, 1>::ones([n], device) * 0.5,
        p_spatial: None,
        k_d: Some(Tensor::<Autodiff<InnerBackend>, 1>::ones([n], device) * 1.0),
        d_gw: Some(Tensor::<Autodiff<InnerBackend>, 1>::zeros([n], device)),
        leakance_factor: Some(Tensor::<Autodiff<InnerBackend>, 1>::ones([n], device) * 1.0),
    }
}

#[test]
fn leakance_removes_water_on_losing_config() {
    let device = TestDevice::default();
    let n = 5usize;
    let t = 24usize;

    // Run 1: no leakance.
    let cfg = mock_config();
    let mut mc_no_leak = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
    mc_no_leak.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        mock_spatial_parameters(n, &device),
        false,
    );
    let out_no_leak = mc_no_leak.forward();
    let sum_no_leak: f32 = out_no_leak.into_data().to_vec::<f32>().unwrap().iter().sum();

    // Run 2: losing-config leakance (K_D high, d_gw at floor, factor=1).
    let mut mc_leak = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
    mc_leak.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        losing_config_params(n, &device),
        false,
    );
    let out_leak = mc_leak.forward();
    let sum_leak: f32 = out_leak.into_data().to_vec::<f32>().unwrap().iter().sum();

    assert!(sum_leak.is_finite(), "with-leakance output is not finite");
    assert!(sum_no_leak.is_finite(), "no-leakance output is not finite");
    assert!(
        sum_leak < sum_no_leak,
        "expected leakance to remove water (sum_leak={sum_leak} < sum_no_leak={sum_no_leak})"
    );
}

#[test]
fn leakance_none_matches_baseline_chain() {
    let device = TestDevice::default();
    let n = 5usize;
    let t = 24usize;
    let cfg = mock_config();
    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());

    mc.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        mock_spatial_parameters(n, &device),
        false,
    );
    let out = mc.forward();
    assert_eq!(out.dims(), [n, t], "output shape");
    let routed: Vec<f32> = out.into_data().to_vec().unwrap();
    assert_eq!(routed.len(), EXPECTED.len());
    for (i, (got, exp)) in routed.iter().zip(EXPECTED.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "leakance-off routing diverged at idx {i}: got {got}, expected {exp}"
        );
    }
}

// ---------------------------------------------------------------------------
// Head-driven smoke test: K_D/d_gw/leakance_factor threaded from KAN head
// ---------------------------------------------------------------------------

/// Build a minimal `RoutingTensors<Autodiff<InnerBackend>>` for a linear chain
/// of `n` reaches with `t` hourly steps and a single gauge at the outlet.
/// This is a self-contained construction — no live data stores required.
fn minimal_routing_tensors(
    n: usize,
    t: usize,
    f_attrs: usize,
    device: &TestDevice,
) -> RoutingTensors<Autodiff<InnerBackend>> {
    use chrono::NaiveDate;
    type AB = Autodiff<InnerBackend>;

    let adjacency = {
        let mut dense = vec![0.0_f32; n * n];
        for i in 0..n - 1 {
            dense[(i + 1) * n + i] = 1.0;
        }
        SparseAdjacency::from_dense(n, &dense, vec![1000.0; n], vec![0.001; n])
    };

    // spatial_attributes: (N, F) normalized, all 0.5
    let attrs_vec = vec![0.5_f32; n * f_attrs];
    let spatial_attributes =
        Tensor::<AB, 2>::from_data(TensorData::new(attrs_vec.clone(), [n, f_attrs]), device);

    // q_prime: (T, N) — a mild sin sweep so depth > 0
    let mut qp_data = vec![0.0_f32; t * n];
    for ti in 0..t {
        let phase = (ti as f32) / (t.max(2) - 1) as f32 * 4.0 * std::f32::consts::PI;
        for ri in 0..n {
            qp_data[ti * n + ri] = (5.0 + phase.sin() * 2.0).max(0.1);
        }
    }
    let q_prime =
        Tensor::<AB, 2>::from_data(TensorData::new(qp_data.clone(), [t, n]), device);

    // q_prime_daily: empty (0, N) — no disagg head
    let q_prime_daily =
        Tensor::<AB, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, n]), device);

    // precip_hourly / temp_hourly: empty (0, N)
    let precip_hourly =
        Tensor::<AB, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, n]), device);
    let temp_hourly =
        Tensor::<AB, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, n]), device);

    // flat_indices + group_ids: outlet reach (n-1) feeds gauge 0
    let flat_indices = Tensor::<AB, 1, Int>::from_data(
        TensorData::from([(n - 1) as i32].as_slice()),
        device,
    );
    let group_ids = Tensor::<AB, 1, Int>::from_data(
        TensorData::from([0i32].as_slice()),
        device,
    );

    // observations: dummy (1, 1) — not used by forward()
    let observations = Array2::zeros((1, 1));

    let gauge_staids = vec![Staid::new("dummy")];
    let window = RhoWindow {
        start_day_idx: 0,
        rho_days: 1,
        window_start: NaiveDate::from_ymd_opt(1990, 1, 1).unwrap(),
    };

    RoutingTensors {
        adjacency,
        spatial_attributes,
        q_prime,
        q_prime_daily,
        precip_hourly,
        temp_hourly,
        observations,
        flat_indices,
        group_ids,
        num_gauges: 1,
        gauge_staids,
        window,
    }
}

/// Build a KAN head with `learnable_parameters = [n, q_spatial, p_spatial, K_D, d_gw, leakance_factor]`.
fn leakance_head(f_attrs: usize, device: &TestDevice) -> ddrs::nn::kan_head::KanHead<Autodiff<InnerBackend>> {
    KanHeadConfig::new(
        (0..f_attrs).map(|i| format!("attr_{i}")).collect(),
        vec![
            "n".to_string(),
            "q_spatial".to_string(),
            "p_spatial".to_string(),
            "K_D".to_string(),
            "d_gw".to_string(),
            "leakance_factor".to_string(),
        ],
        42,
    )
    .with_hidden_size(8)
    .with_num_hidden_layers(1)
    .init::<Autodiff<InnerBackend>>(device)
}


/// Head-driven smoke: when `use_leakance=true`, the `K_D/d_gw/leakance_factor`
/// keys from the head HashMap are threaded into `SpatialParameters`, so the
/// same head run with `use_leakance=true` must produce different output than
/// the same head run with `use_leakance=false`.
///
/// **Pre-Task 9 behaviour:** `forward.rs` hardcodes `k_d: None` regardless of
/// the flag — both runs are identical → `assert_ne!` FAILS.
/// **Post-Task 9:** the flag gates look up the keys and thread them in → PASS.
#[test]
fn head_driven_leakance_changes_output() {
    let device = TestDevice::default();
    let n = 5usize;
    let t = 24usize;
    let f = 4usize; // number of spatial attributes

    let tensors = minimal_routing_tensors(n, t, f, &device);

    // Same head for both runs — includes K_D / d_gw / leakance_factor keys.
    let head = leakance_head(f, &device);

    // Config A: use_leakance = true — head params should be threaded in.
    let mut cfg_leak = mock_config();
    cfg_leak.params.use_leakance = true;
    cfg_leak.params.use_cuda_graphs = false;

    // Config B: use_leakance = false — leakance keys in the map are ignored.
    let cfg_no_leak = mock_config();

    // Run A: with leakance.
    let out_leak = ddrs::training::forward::forward(
        &cfg_leak,
        &tensors,
        &head,
        &device,
        false,
    );
    let sum_leak: f32 = out_leak.into_data().to_vec::<f32>().unwrap().iter().sum();

    // Run B: without leakance (same head, different config).
    let out_no_leak = ddrs::training::forward::forward(
        &cfg_no_leak,
        &tensors,
        &head,
        &device,
        false,
    );
    let sum_no_leak: f32 = out_no_leak.into_data().to_vec::<f32>().unwrap().iter().sum();

    assert!(sum_leak.is_finite(), "leakance run output is not finite");
    assert!(sum_no_leak.is_finite(), "no-leakance run output is not finite");
    assert_ne!(
        sum_leak, sum_no_leak,
        "leakance run must differ from no-leakance run — \
         K_D/d_gw/leakance_factor not threaded into SpatialParameters \
         (got sum_leak={sum_leak}, sum_no_leak={sum_no_leak})"
    );
}
