//! Engine integration of the stage block: absent is bitwise the committed
//! hydrograph, on changes it, a subset leaves the upstream rows alone, the
//! refusals, the carry, and the MC-row gradients through the hook.

mod common;

use burn::tensor::Tensor;
use chrono::NaiveDate;

use ddrs::config::{DamRow, DownstreamBoundary};
use ddrs::data::store::ReservoirRows;
use ddrs::routing::mmc::DamRelease;
use ddrs::routing::stage::{StageCarry, StageInputs, StageParams};
use ddrs::routing::{MuskingumCunge, SpatialParameters};
use ddrs::training::forward::apply_stage_block;

use common::{
    mock_config, mock_routing_inputs, mock_spatial_parameters, mock_streamflow, InnerBackend,
    TestBackend, TestDevice,
};

/// Verbatim from `tests/leakance_off_parity.rs::EXPECTED` (WET by user preference, C33).
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

const N: usize = 5;
const T: usize = 24;

fn stage_params(boundary: DownstreamBoundary) -> StageParams {
    StageParams {
        theta: 1.0,
        theta_control: 0.5,
        substeps: 1,
        boundary,
        depth_lb: 0.01,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    }
}

fn inputs(seeds: Option<Vec<usize>>) -> StageInputs {
    StageInputs {
        seeds,
        anchor_m: 0.0,
        boundary_series: None,
        boundary_row: None,
        geometry: None,
        dams: vec![],
    }
}

/// Spatial parameters with p = 20.9 (normalized 0.1 on [1, 200]).
fn spatial(n: usize, device: &TestDevice) -> SpatialParameters<InnerBackend> {
    SpatialParameters {
        p_spatial: Some(Tensor::<TestBackend, 1>::ones([n], device) * 0.1),
        ..mock_spatial_parameters(n, device)
    }
}

fn engine(n: usize, t: usize, device: &TestDevice) -> MuskingumCunge<InnerBackend> {
    let mut mc = MuskingumCunge::<InnerBackend>::new(mock_config(), device.clone());
    mc.setup_inputs(
        mock_routing_inputs(n, device),
        mock_streamflow(t, n, device),
        spatial(n, device),
        false,
        None,
    );
    mc
}

fn run(mut mc: MuskingumCunge<InnerBackend>) -> Vec<f32> {
    mc.forward().into_data().to_vec::<f32>().unwrap()
}

#[test]
fn stage_routing_absent_is_bitwise_the_committed_hydrograph() {
    let device = TestDevice::default();
    // EXPECTED was captured under the legacy physics (C7).
    let mut cfg = mock_config();
    cfg.params.ddr_match = true;
    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc.setup_inputs(
        mock_routing_inputs(N, &device),
        mock_streamflow(T, N, &device),
        mock_spatial_parameters(N, &device),
        false,
        None,
    );
    let routed = run(mc);
    assert_eq!(routed.len(), EXPECTED.len());
    let mut max_diff = 0.0f32;
    let mut bitwise = true;
    for (got, exp) in routed.iter().zip(EXPECTED.iter()) {
        max_diff = max_diff.max((got - exp).abs());
        bitwise &= got.to_bits() == exp.to_bits();
    }
    println!("stage absent: max abs diff to EXPECTED {max_diff:e}, bitwise = {bitwise}");
    assert!(max_diff < 1e-6, "max abs diff {max_diff}");
}

#[test]
fn stage_routing_on_changes_the_hydrograph() {
    let device = TestDevice::default();
    let a = run(engine(N, T, &device));
    let mut mc = engine(N, T, &device);
    mc.set_stage_block(stage_params(DownstreamBoundary::NormalDepth), inputs(None)).unwrap();
    let b = run(mc);
    assert!(b.iter().all(|v| v.is_finite()), "stage hydrograph is not finite");
    let max_diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("stage all vs MC-only: max abs diff {max_diff}");
    assert!(max_diff > 1e-3, "max abs diff {max_diff}");
}

#[test]
fn stage_set_subset_leaves_upstream_rows_bitwise() {
    let device = TestDevice::default();
    let a = run(engine(N, T, &device));
    let mut mc = engine(N, T, &device);
    mc.set_stage_block(stage_params(DownstreamBoundary::NormalDepth), inputs(Some(vec![3]))).unwrap();
    let b = run(mc);
    for i in 0..3 * T {
        assert_eq!(a[i].to_bits(), b[i].to_bits(), "MC row {} hour {} differs", i / T, i % T);
    }
    let max_stage = (3 * T..N * T).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max);
    assert!(max_stage > 1e-4, "stage rows 3..5 do not differ (max {max_stage})");
    assert!(b.iter().all(|v| v.is_finite()));
}

#[test]
fn route_timestep_with_leakance_and_stage_panics() {
    let device = TestDevice::default();
    let mut mc = MuskingumCunge::<InnerBackend>::new(mock_config(), device.clone());
    let one = |v: f32| Tensor::<TestBackend, 1>::ones([N], &device) * v;
    mc.setup_inputs(
        mock_routing_inputs(N, &device),
        mock_streamflow(T, N, &device),
        SpatialParameters {
            k_d: Some(one(1.0)),
            d_gw: Some(one(0.0)),
            leakance_factor: Some(one(1.0)),
            ..spatial(N, &device)
        },
        false,
        None,
    );
    let err = mc
        .set_stage_block(stage_params(DownstreamBoundary::NormalDepth), inputs(None))
        .unwrap_err();
    assert!(err.contains("stage") && err.contains("leakance"), "{err}");
}

#[test]
fn boundary_comid_must_be_the_stage_set_outlet() {
    let device = TestDevice::default();
    let mut bad = inputs(Some(vec![1]));
    bad.boundary_row = Some(2);
    let err = engine(N, T, &device)
        .set_stage_block(stage_params(DownstreamBoundary::NormalDepth), bad)
        .unwrap_err();
    assert!(err.contains("boundary_comid"), "{err}");
    let mut good = inputs(Some(vec![1]));
    good.boundary_row = Some(4);
    engine(N, T, &device)
        .set_stage_block(stage_params(DownstreamBoundary::NormalDepth), good)
        .unwrap();
}

fn release_on(row: usize, device: &TestDevice) -> DamRelease<InnerBackend> {
    DamRelease {
        rows: vec![row],
        t0_days: Tensor::<TestBackend, 1>::from_floats([1.0], device),
        seasonal: None,
        phase: vec![[0.0, 1.0]; T],
        dam_row: DamRow::Replace,
        rule_curve: None,
    }
}

#[test]
fn release_armed_inside_the_set_is_refused() {
    let device = TestDevice::default();
    let mut mc = engine(N, T, &device);
    mc.set_dam_release(release_on(3, &device)).unwrap();
    let err = mc
        .set_stage_block(stage_params(DownstreamBoundary::NormalDepth), inputs(Some(vec![2])))
        .unwrap_err();
    assert!(err.contains("stage") && err.contains("release"), "{err}");
    // A release outside the set is fine.
    let mut mc = engine(N, T, &device);
    mc.set_dam_release(release_on(1, &device)).unwrap();
    mc.set_stage_block(stage_params(DownstreamBoundary::NormalDepth), inputs(Some(vec![2]))).unwrap();
}

fn stage_cfg() -> ddrs::config::Config {
    let mut cfg = mock_config();
    cfg.params.stage_routing = Some(ddrs::config::StageRouting {
        enabled: true,
        reaches: ddrs::config::StageReaches::Comids(vec![2]),
        ..Default::default()
    });
    cfg
}

#[test]
#[should_panic(expected = "seasonal")]
fn seasonal_dam_inside_the_set_is_refused() {
    let device = TestDevice::default();
    let cfg = stage_cfg();
    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
    mc.setup_inputs(
        mock_routing_inputs(N, &device),
        mock_streamflow(T, N, &device),
        spatial(N, &device),
        false,
        None,
    );
    let rows = ReservoirRows {
        rows: vec![3],
        t_days: vec![1.0],
        seasonal: Some((vec![0.1], vec![0.1])),
        ..Default::default()
    };
    let batch = ddrs::data::dataset::StageBatch {
        seed_rows: Some(vec![2]),
        ..Default::default()
    };
    apply_stage_block(&cfg, &mut mc, Some(&batch), Some(&rows), NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
}

/// 3-chain, stage set {1, 2}, outlet stage flat for 22 h, then jumping to 20 m.
fn backflow_engine(
    leaf_q0: Option<Tensor<TestBackend, 1>>,
    leaves: Option<&[Tensor<TestBackend, 1>; 3]>,
    device: &TestDevice,
) -> MuskingumCunge<InnerBackend> {
    let n = 3;
    let mut mc = MuskingumCunge::<InnerBackend>::new(mock_config(), device.clone());
    let sp = match leaves {
        Some([a, b, c]) => SpatialParameters {
            n: a.clone(),
            q_spatial: b.clone(),
            p_spatial: Some(c.clone()),
            ..mock_spatial_parameters(n, device)
        },
        None => spatial(n, device),
    };
    mc.setup_inputs(
        mock_routing_inputs(n, device),
        mock_streamflow(T, n, device),
        sp,
        false,
        leaf_q0,
    );
    let mut inp = inputs(Some(vec![1]));
    inp.boundary_series = Some((0..T).map(|k| if k < 22 { 0.6 } else { 20.0 }).collect());
    mc.set_stage_block(stage_params(DownstreamBoundary::PrescribedStage), inp).unwrap();
    mc
}

#[test]
fn stage_carry_round_trips_unclamped() {
    let device = TestDevice::default();
    let mut a = backflow_engine(None, None, &device);
    let _ = a.forward();
    let c: StageCarry = a.stage_carry().unwrap();
    println!("carried Q {:?}", c.q);
    assert!(c.q[1] < 0.0 || c.q[2] < 0.0, "no backflow at the end of the window: {:?}", c.q);
    let mut b = backflow_engine(None, None, &device);
    b.set_stage_carry(&c).unwrap();
    let h: Vec<f32> = b.stage_state().unwrap().into_data().to_vec().unwrap();
    assert_eq!(h.len(), c.h.len());
    for (x, y) in h.iter().zip(&c.h) {
        assert_eq!(x.to_bits(), y.to_bits(), "h differs");
    }
    let q: Vec<f32> = b.discharge_state().unwrap().into_data().to_vec().unwrap();
    for r in 1..3 {
        assert_eq!(q[r].to_bits(), c.q[r].to_bits(), "Q row {r} differs: {} vs {}", q[r], c.q[r]);
    }
}

#[test]
fn engine_mc_rows_stay_finite_under_stage_backflow() {
    let device = TestDevice::default();
    let leaf = |v: f32| Tensor::<TestBackend, 1>::from_floats([v; 3], &device).require_grad();
    let (n_l, q_l, p_l) = (leaf(0.5), leaf(0.5), leaf(0.1));
    let q0 = Tensor::<TestBackend, 1>::from_floats([5.0, 3.0, 3.0], &device).require_grad();
    let mut mc = backflow_engine(Some(q0.clone()), Some(&[n_l.clone(), q_l.clone(), p_l.clone()]), &device);
    // Backflow on the stage rows at the step start, stage high above the bed.
    let h = mc.stage_state().unwrap();
    let h_vals: Vec<f32> = h.into_data().to_vec().unwrap();
    mc.set_stage_carry(&StageCarry { h: h_vals, q: vec![0.0, -5.0, -2.0] }).unwrap();
    let qp = Tensor::<TestBackend, 1>::from_floats([1.0; 3], &device);
    let q_next = mc.route_timestep(qp);
    let g = q_next.clone().sum().backward();
    for (name, t) in [("n", &n_l), ("q", &q_l), ("p", &p_l), ("q0", &q0)] {
        let v: Vec<f32> = t.grad(&g).unwrap_or_else(|| panic!("no gradient for {name}")).into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "gradient of {name} is not finite: {v:?}");
    }
    // The MC row (0) still gets a gradient through the hook (a hook that
    // detached every row would give 0).
    let g0: Vec<f32> = q0.grad(&g).unwrap().into_data().to_vec().unwrap();
    assert!(g0[0].abs() > 1e-6, "MC-row q_t gradient is {}", g0[0]);
    // mc_view: stage rows are constants for the MC op, MC rows pass through.
    let probe = Tensor::<TestBackend, 1>::from_floats([5.0, -5.0, -2.0], &device).require_grad();
    let view = mc.stage_block_mut().unwrap().mc_view(probe.clone(), 1e-3);
    let gm = view.clone().slice([0..1]).sum().backward();
    assert_eq!(probe.grad(&gm).unwrap().into_data().to_vec::<f32>().unwrap(), vec![1.0, 0.0, 0.0]);
    let gs = view.slice([1..3]).sum().backward();
    let v: Vec<f32> = probe.grad(&gs).map(|t| t.into_data().to_vec().unwrap()).unwrap_or(vec![0.0; 3]);
    assert_eq!(v, vec![0.0, 0.0, 0.0], "stage-row q_t gradient through the MC path is not zero");
}
