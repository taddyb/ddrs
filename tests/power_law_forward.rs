//! The fused forward on `geometry_form: power_law`: saved state against the
//! reference section functions, the change from the trapezoid, the stage
//! block across forms, and the entry guards for paths that have no power-law
//! support.

mod common;

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::{Config, DownstreamBoundary, GeometryForm, StageRoughnessSection};
use ddrs::geometry::compute_power_law_geometry;
use ddrs::routing::mmc_op::{
    __spike_forward_chain_k1_outputs, timestep_forward, timestep_forward_leakance,
    timestep_forward_release, timestep_forward_via_graph,
};
use ddrs::routing::stage::{StageInputs, StageParams};
use ddrs::routing::{MuskingumCunge, SpatialParameters};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

use common::{
    mock_config, mock_routing_inputs, mock_spatial_parameters, mock_streamflow, InnerBackend,
    TestBackend, TestDevice,
};

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 4;

fn mock_cfg(form: GeometryForm) -> Config {
    let mut cfg = Config::default();
    cfg.params.ddr_match = false;
    cfg.params.parameter_ranges.n = [0.01, 0.1];
    cfg.params.parameter_ranges.q_spatial = [0.1, 0.9];
    cfg.params.parameter_ranges.p_spatial = [1.0, 200.0];
    cfg.params.attribute_minimums.velocity = 0.1;
    cfg.params.attribute_minimums.depth = 0.01;
    cfg.params.attribute_minimums.discharge = 0.001;
    cfg.params.attribute_minimums.bottom_width = 0.1;
    cfg.params.attribute_minimums.slope = 0.001;
    cfg.params.defaults.insert("p_spatial".to_string(), 1.0);
    cfg.params.log_space_parameters = vec![];
    cfg.params.geometry_form = form;
    cfg
}

fn chain(length_m: f32) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![length_m; N], vec![1e-3; N])
}

const QT: [f32; N] = [100.0, 120.0, 140.0, 160.0];
const QPT: [f32; N] = [10.0, 12.0, 14.0, 16.0];

fn device() -> <I as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

fn t1(v: &[f32]) -> Tensor<I, 1> {
    Tensor::<I, 1>::from_floats(v, &device())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn vec_of(t: Tensor<I, 1>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

fn k1(cfg: &Config, length_m: f32) -> Vec<Vec<f32>> {
    let adj = chain(length_m);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    __spike_forward_chain_k1_outputs::<I>(
        cfg,
        &pattern,
        t1(&[0.035; N]),
        t1(&[0.4; N]),
        t1(&[20.0; N]),
        t1(&QT),
        t1(&QPT),
        t1(&adj.length_m),
        t1(&adj.slope),
        t1(&[0.3; N]),
    )
}

fn q_next(cfg: &Config, length_m: f32) -> Vec<f32> {
    let adj = chain(length_m);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device());
    let out = timestep_forward::<I>(
        cfg,
        &pattern,
        &assembler,
        at(&[0.035; N]),
        at(&[0.4; N]),
        at(&[20.0; N]),
        at(&QT),
        at(&QPT),
        at(&adj.length_m),
        at(&adj.slope),
        at(&[0.3; N]),
        false,
        None,
        None,
    );
    out.into_data().to_vec::<f32>().unwrap()
}

fn check_saved_state(gamma: f32) {
    let mut cfg = mock_cfg(GeometryForm::PowerLaw);
    if gamma != 0.0 {
        cfg.params.stage_roughness = Some(StageRoughnessSection { gamma, d_ref: 1.0 });
    }
    let length = 10_000.0_f32;
    let out = k1(&cfg, length);
    let adj = chain(length);
    let g = compute_power_law_geometry::<I>(
        t1(&[0.035; N]),
        t1(&[20.0; N]),
        t1(&[0.4; N]),
        t1(&QT),
        t1(&adj.slope),
        0.01,
        gamma,
        1.0,
    );
    let v_cl = g.velocity.clone().clamp(0.1, 15.0);
    let celerity = v_cl.clone() * g.celerity_ratio.clone();
    let k_mus = t1(&adj.length_m) / celerity.clone();

    assert_eq!(bits(&out[0]), bits(&vec_of(g.depth.clone())), "DEPTH");
    assert_eq!(bits(&out[1]), bits(&vec_of(g.top_width.clone())), "TOP_WIDTH");
    assert_eq!(bits(&out[4]), bits(&vec_of(g.hydraulic_radius.clone())), "HYDRAULIC_RADIUS");
    assert_eq!(bits(&out[5]), bits(&vec_of(g.velocity.clone())), "VELOCITY_UNCLAMPED");
    assert_eq!(bits(&out[6]), bits(&vec_of(v_cl)), "VELOCITY_CLAMPED");
    assert_eq!(bits(&out[7]), bits(&vec_of(celerity)), "CELERITY");
    assert_eq!(bits(&out[8]), bits(&vec_of(k_mus)), "K_MUSKINGUM");
    assert_eq!(bits(&out[14]), bits(&vec_of(g.ln_ratio.clone())), "RATIO");
    // Trapezoid-only slots hold exact zeros.
    for (slot, name) in [
        (2, "SIDE_SLOPE"),
        (3, "BOTTOM_WIDTH"),
        (15, "DENOMINATOR"),
        (17, "SIDE_SLOPE_RAW"),
        (18, "BW_RAW"),
    ] {
        assert!(out[slot].iter().all(|v| *v == 0.0), "{name} is not zero");
    }
}

#[test]
fn saved_state_matches_the_reference_bitwise() {
    check_saved_state(0.0);
    check_saved_state(0.35);
}

#[test]
fn power_law_changes_q_next() {
    let a = q_next(&mock_cfg(GeometryForm::Trapezoid), 10_000.0);
    let b = q_next(&mock_cfg(GeometryForm::PowerLaw), 10_000.0);
    assert!(b.iter().all(|v| v.is_finite()));
    let max_diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("power_law vs trapezoid q_next: max abs diff {max_diff}");
    assert!(max_diff > 1e-3, "max abs diff {max_diff}");
}

// ---------------------------------------------------------------------------
// Stage block across forms (WET copy of the `stage_off_parity` helpers).
// ---------------------------------------------------------------------------

const SN: usize = 5;
const ST: usize = 24;

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

fn spatial(n: usize, device: &TestDevice) -> SpatialParameters<InnerBackend> {
    SpatialParameters {
        p_spatial: Some(Tensor::<TestBackend, 1>::ones([n], device) * 0.1),
        ..mock_spatial_parameters(n, device)
    }
}

fn stage_on_config(form: GeometryForm) -> Config {
    let mut cfg = mock_config();
    cfg.params.stage_routing = Some(ddrs::config::StageRouting {
        enabled: true,
        ..Default::default()
    });
    cfg.params.geometry_form = form;
    cfg
}

fn engine(form: GeometryForm, device: &TestDevice) -> MuskingumCunge<InnerBackend> {
    let mut mc = MuskingumCunge::<InnerBackend>::new(stage_on_config(form), device.clone());
    mc.setup_inputs(
        mock_routing_inputs(SN, device),
        mock_streamflow(ST, SN, device),
        spatial(SN, device),
        false,
        None,
    );
    mc
}

fn run(mut mc: MuskingumCunge<InnerBackend>) -> Vec<f32> {
    mc.forward().into_data().to_vec::<f32>().unwrap()
}

#[test]
fn full_stage_block_is_bitwise_across_forms() {
    let device = TestDevice::default();
    let go = |form: GeometryForm| {
        let mut mc = engine(form, &device);
        mc.set_stage_block(stage_params(DownstreamBoundary::NormalDepth), inputs(None))
            .unwrap();
        run(mc)
    };
    let a = go(GeometryForm::Trapezoid);
    let b = go(GeometryForm::PowerLaw);
    assert_eq!(bits(&a), bits(&b), "stage rows differ across forms");
}

#[test]
fn stage_subset_changes_mc_rows_only_by_form() {
    let device = TestDevice::default();
    let go = |form: GeometryForm| {
        let mut mc = engine(form, &device);
        mc.set_stage_block(
            stage_params(DownstreamBoundary::NormalDepth),
            inputs(Some(vec![3])),
        )
        .unwrap();
        run(mc)
    };
    let a = go(GeometryForm::Trapezoid);
    let b = go(GeometryForm::PowerLaw);
    assert!(b.iter().all(|v| v.is_finite()));
    let max_mc = (0..3 * ST).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max);
    assert!(max_mc > 1e-4, "MC rows do not differ across forms (max {max_mc})");
}

// ---------------------------------------------------------------------------
// Entry guards.
// ---------------------------------------------------------------------------

fn at(v: &[f32]) -> Tensor<AB, 1> {
    Tensor::<AB, 1>::from_floats(v, &device())
}

#[test]
#[should_panic(expected = "geometry_form")]
fn leakance_entry_rejects_power_law() {
    let cfg = mock_cfg(GeometryForm::PowerLaw);
    let adj = chain(10_000.0);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device());
    let _ = timestep_forward_leakance::<I>(
        &cfg,
        &pattern,
        &assembler,
        at(&[0.035; N]),
        at(&[0.4; N]),
        at(&[20.0; N]),
        at(&QT),
        at(&QPT),
        at(&adj.length_m),
        at(&adj.slope),
        at(&[0.3; N]),
        at(&[5e-7; N]),
        at(&[0.0; N]),
        at(&[0.5; N]),
        None,
        None,
        false,
        None,
        None,
    );
}

#[test]
#[should_panic(expected = "geometry_form")]
fn release_entry_rejects_power_law() {
    let cfg = mock_cfg(GeometryForm::PowerLaw);
    let adj = chain(10_000.0);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device());
    let _ = timestep_forward_release::<I>(
        &cfg,
        &pattern,
        &assembler,
        at(&[0.035; N]),
        at(&[0.4; N]),
        at(&[20.0; N]),
        at(&QT),
        at(&QPT),
        at(&adj.length_m),
        at(&adj.slope),
        at(&[0.3; N]),
        vec![1],
        at(&[3600.0; N]),
        at(&[3600.0; N]),
    );
}

#[test]
#[should_panic(expected = "geometry_form")]
fn via_graph_entry_rejects_power_law() {
    let cfg = mock_cfg(GeometryForm::PowerLaw);
    let adj = chain(10_000.0);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device());
    let _ = timestep_forward_via_graph::<I>(
        &cfg,
        &pattern,
        &assembler,
        at(&[0.035; N]),
        at(&[0.4; N]),
        at(&[20.0; N]),
        at(&QT),
        at(&QPT),
        at(&adj.length_m),
        at(&adj.slope),
        at(&[0.3; N]),
    );
}
