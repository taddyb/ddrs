//! `reach_k_hours` must give the same K as the fused solver on every
//! geometry form, so the adjoint and landscape studies probe the surface the
//! solver routes on.

use std::sync::Arc;

use burn::backend::NdArray;
use burn::tensor::Tensor;

use ddrs::config::{Config, GeometryForm, StageRoughnessSection};
use ddrs::experiment::adjoint::hydraulics::reach_k_hours;
use ddrs::geometry::compute_power_law_geometry_field;
use ddrs::routing::mmc_op::__spike_forward_chain_k1_outputs;
use ddrs::sparse::{CsrPattern, SparseAdjacency};

type I = NdArray<f32>;

const N: usize = 4;
const L: f32 = 10_000.0;
const QT: [f32; N] = [100.0, 120.0, 140.0, 160.0];
const QPT: [f32; N] = [10.0, 12.0, 14.0, 16.0];

fn mock_cfg(form: GeometryForm, gamma: f32) -> Config {
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
    if gamma != 0.0 {
        cfg.params.stage_roughness = Some(StageRoughnessSection { gamma, d_ref: 1.0 });
    }
    cfg
}

fn chain() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![L; N], vec![1e-3; N])
}

fn t1(v: &[f32]) -> Tensor<I, 1> {
    Tensor::<I, 1>::from_floats(v, &Default::default())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// K in hours from the fused solver (k1 hook, K_MUSKINGUM at index 8).
fn solver_k_hours(cfg: &Config) -> Vec<f32> {
    let adj = chain();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let out = __spike_forward_chain_k1_outputs::<I>(
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
    );
    out[8].iter().map(|s| s / 3600.0).collect()
}

fn helper_k_hours(cfg: &Config, gamma_field: Option<&Tensor<I, 1>>) -> Vec<f32> {
    let adj = chain();
    reach_k_hours::<I>(
        cfg,
        &t1(&[0.035; N]),
        &t1(&[20.0; N]),
        &t1(&[0.4; N]),
        t1(&QT),
        &t1(&adj.slope),
        &t1(&adj.length_m),
        gamma_field,
    )
}

#[test]
fn reach_k_hours_matches_the_solver_k() {
    // Trapezoid: bitwise, or the recorded exception below.
    let cfg = mock_cfg(GeometryForm::Trapezoid, 0.0);
    let (h, s) = (helper_k_hours(&cfg, None), solver_k_hours(&cfg));
    for (a, b) in h.iter().zip(&s) {
        assert!(((a - b) / b).abs() < 1e-6, "trapezoid {a} vs {b}");
    }
    if bits(&h) != bits(&s) {
        eprintln!("trapezoid K is not bitwise; within rel 1e-6");
    }

    // Power law, no stage roughness.
    let cfg = mock_cfg(GeometryForm::PowerLaw, 0.0);
    assert_eq!(bits(&helper_k_hours(&cfg, None)), bits(&solver_k_hours(&cfg)));

    // Power law, global gamma.
    let cfg = mock_cfg(GeometryForm::PowerLaw, 0.35);
    assert_eq!(bits(&helper_k_hours(&cfg, None)), bits(&solver_k_hours(&cfg)));

    // Power law, per-reach gamma field (physical, d_ref = 1).
    let cfg = mock_cfg(GeometryForm::PowerLaw, 0.0);
    let gamma = t1(&[0.2; N]);
    let adj = chain();
    let g = compute_power_law_geometry_field::<I>(
        t1(&[0.035; N]),
        t1(&[20.0; N]),
        t1(&[0.4; N]),
        t1(&QT),
        t1(&adj.slope),
        cfg.params.attribute_minimums.depth,
        gamma.clone(),
        1.0,
    );
    let celerity = g.velocity.clamp(cfg.params.attribute_minimums.velocity, 15.0) * g.celerity_ratio;
    let expect: Vec<f32> = (t1(&adj.length_m) / celerity)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|s| s / 3600.0)
        .collect();
    assert_eq!(bits(&helper_k_hours(&cfg, Some(&gamma))), bits(&expect));
}
