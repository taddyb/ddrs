//! `forward_chain_inner` with `leakance = None` must produce byte-identical
//! `q_next` to the pre-leakance code path. The deterministic linear-chain
//! routing fixture below is captured from the unmodified routing core (the
//! `forward()` of a 5-reach chain over 24 steps with the shared `mock_*`
//! fixtures); after the Phase-3 changes it must reproduce the SAME hydrograph
//! bit-for-bit, proving the leakance-off path is untouched.

mod common;

use burn::tensor::Tensor;
use common::{
    mock_config, mock_routing_inputs, mock_spatial_parameters, mock_streamflow, InnerBackend,
    TestDevice,
};
use ddrs::routing::{MuskingumCunge, SpatialParameters};

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
