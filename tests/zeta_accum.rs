//! Eval-time zeta accumulation (leakance diagnostics).
//!
//! Verifies the `enable_zeta_accumulation` path on `MuskingumCunge`:
//!   1. Accumulation yields `None` when leakance is off or accumulation is
//!      not enabled, and never perturbs the routed discharge.
//!   2. The accumulated zeta is EXACTLY what was subtracted from `b_rhs`:
//!      for a headwater reach (no upstream), `A·x = b` has `x[0] = b[0]`, so
//!      `q_no_leak[0] − q_leak[0] == zeta[0]` on a single routed timestep.
//!   3. zeta is linear in `leakance_factor` on a single timestep (depth at
//!      t=1 depends only on the hotstart Q0, which is leakance-independent).

mod common;

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use common::{
    mock_config, mock_routing_inputs, mock_spatial_parameters, mock_streamflow, InnerBackend,
    TestDevice,
};
use ddrs::routing::{MuskingumCunge, SpatialParameters};

type AB = Autodiff<InnerBackend>;

/// Losing-regime leakance params (normalized): K_D at the top of its log
/// range, d_gw at the bottom of [-2, 2] (always below any depth), factor as
/// given. n/q_spatial match `mock_spatial_parameters` so geometry is shared.
fn leakance_params(n: usize, factor_norm: f32, device: &TestDevice) -> SpatialParameters<InnerBackend> {
    SpatialParameters {
        n: Tensor::<AB, 1>::ones([n], device) * 0.5,
        q_spatial: Tensor::<AB, 1>::ones([n], device) * 0.5,
        p_spatial: None,
        k_d: Some(Tensor::<AB, 1>::ones([n], device)),
        d_gw: Some(Tensor::<AB, 1>::zeros([n], device)),
        leakance_factor: Some(Tensor::<AB, 1>::ones([n], device) * factor_norm),
    }
}

fn forward_vec(mc: &mut MuskingumCunge<InnerBackend>) -> Vec<f32> {
    mc.forward().into_data().to_vec::<f32>().unwrap()
}

#[test]
fn zeta_sums_none_when_leakance_off_or_not_enabled() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 6usize);
    let cfg = mock_config();

    // Leakance off, accumulation enabled → None.
    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
    mc.enable_zeta_accumulation();
    mc.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        mock_spatial_parameters(n, &device),
        false,
    );
    let _ = mc.forward();
    assert!(mc.zeta_sums().is_none(), "no leakance params ⇒ no zeta sums");

    // Leakance on, accumulation NOT enabled → None.
    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        leakance_params(n, 1.0, &device),
        false,
    );
    let _ = mc.forward();
    assert!(mc.zeta_sums().is_none(), "accumulation off ⇒ no zeta sums");
}

#[test]
fn accumulation_does_not_perturb_discharge() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 24usize);
    let cfg = mock_config();

    let mut mc_plain = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
    mc_plain.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        leakance_params(n, 1.0, &device),
        false,
    );
    let out_plain = forward_vec(&mut mc_plain);

    let mut mc_accum = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc_accum.enable_zeta_accumulation();
    mc_accum.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        leakance_params(n, 1.0, &device),
        false,
    );
    let out_accum = forward_vec(&mut mc_accum);

    assert_eq!(out_plain, out_accum, "accumulation must not change routing");

    let sums = mc_accum.zeta_sums().expect("zeta sums present");
    assert_eq!(sums.steps, t - 1, "one accumulated step per routed timestep");
    let abs_v: Vec<f32> = sums.abs.into_data().to_vec().unwrap();
    let net_v: Vec<f32> = sums.net.into_data().to_vec().unwrap();
    assert_eq!(abs_v.len(), n);
    // Losing regime (d_gw = −2 m < depth) ⇒ zeta > 0 everywhere ⇒ |Σzeta| = Σ|zeta|.
    for (a, m) in abs_v.iter().zip(net_v.iter()) {
        assert!(a.is_finite() && *a > 0.0, "expected positive finite zeta, got {a}");
        assert!((a - m).abs() < 1e-9, "losing regime: net ({m}) must equal abs ({a})");
    }
}

#[test]
fn accumulated_zeta_equals_headwater_qnext_difference() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 2usize); // single routed timestep

    // Same n/q_spatial with and without leakance ⇒ identical c1..c4 and depth
    // at t=1 (hotstart Q0 is leakance-independent). Reach 0 is a headwater in
    // the linear chain, so x_sol[0] = b_rhs[0] and the discharge difference
    // is exactly zeta[0].
    let cfg = mock_config();

    let mut mc_off = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
    mc_off.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        mock_spatial_parameters(n, &device),
        false,
    );
    let out_off = forward_vec(&mut mc_off); // [n, 2] row-major

    let mut mc_on = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc_on.enable_zeta_accumulation();
    mc_on.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        leakance_params(n, 1.0, &device),
        false,
    );
    let out_on = forward_vec(&mut mc_on);

    let sums = mc_on.zeta_sums().expect("zeta sums present");
    assert_eq!(sums.steps, 1);
    let zeta: Vec<f32> = sums.abs.into_data().to_vec().unwrap();

    // Column t=1 of reach 0 lives at index 0*t + 1.
    let diff = out_off[1] - out_on[1];
    assert!(
        (diff - zeta[0]).abs() < 1e-6 * zeta[0].abs().max(1.0),
        "headwater q_next difference ({diff}) must equal accumulated zeta[0] ({})",
        zeta[0]
    );
}

#[test]
fn zeta_is_linear_in_leakance_factor_on_single_step() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 2usize);
    let cfg = mock_config();

    let run = |factor_norm: f32| -> Vec<f32> {
        let mut mc = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
        mc.enable_zeta_accumulation();
        mc.setup_inputs(
            mock_routing_inputs(n, &device),
            mock_streamflow(t, n, &device),
            leakance_params(n, factor_norm, &device),
            false,
        );
        let _ = mc.forward();
        let sums = mc.zeta_sums().expect("zeta sums present");
        sums.abs.into_data().to_vec().unwrap()
    };

    let z_full = run(1.0);
    let z_half = run(0.5);
    for (f, h) in z_full.iter().zip(z_half.iter()) {
        assert!(
            (f - 2.0 * h).abs() < 1e-6 * f.abs().max(1e-12),
            "zeta must be linear in leakance_factor: full={f}, half={h}"
        );
    }
}

#[test]
fn q_mean_matches_routed_discharge() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 24usize);
    let cfg = mock_config();

    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc.enable_zeta_accumulation();
    mc.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        leakance_params(n, 1.0, &device),
        false,
    );
    let out = forward_vec(&mut mc); // [n, t] row-major

    // q_sum accumulates the SAME q_next tensors that become output columns
    // 1..t, in the same order, so the sums match to f32 addition noise.
    let sums = mc.zeta_sums().expect("zeta sums present");
    assert_eq!(sums.steps, t - 1);
    let q_sum: Vec<f32> = sums.q.into_data().to_vec().unwrap();
    assert_eq!(q_sum.len(), n);
    for i in 0..n {
        let expected: f32 = (1..t).map(|j| out[i * t + j]).sum();
        assert!(
            (q_sum[i] - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "reach {i}: q_sum ({}) must equal summed routed discharge ({expected})",
            q_sum[i]
        );
    }
}

#[test]
fn depth_and_area_z_are_leakance_independent_primitives() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 2usize); // single routed timestep

    // Depth at t=1 is a function of the hotstart Q0 only, so depth and area_z
    // must be identical across leakance_factor values while zeta scales.
    let cfg = mock_config();
    let run = |factor_norm: f32| {
        let mut mc = MuskingumCunge::<InnerBackend>::new(cfg.clone(), device.clone());
        mc.enable_zeta_accumulation();
        mc.setup_inputs(
            mock_routing_inputs(n, &device),
            mock_streamflow(t, n, &device),
            leakance_params(n, factor_norm, &device),
            false,
        );
        let _ = mc.forward();
        mc.zeta_sums().expect("zeta sums present")
    };

    let full = run(1.0);
    let half = run(0.5);

    let depth_f: Vec<f32> = full.depth.into_data().to_vec().unwrap();
    let depth_h: Vec<f32> = half.depth.into_data().to_vec().unwrap();
    let area_f: Vec<f32> = full.area_z.into_data().to_vec().unwrap();
    let area_h: Vec<f32> = half.area_z.into_data().to_vec().unwrap();
    assert_eq!(depth_f, depth_h, "depth must not depend on leakance_factor");
    assert_eq!(area_f, area_h, "area_z must not depend on leakance_factor");

    // Structural identity: zeta = factor·area_z·K_D·(depth − d_gw). With
    // uniform factor/K_D and d_gw = −2 m (leakance_params denormalizes to the
    // bottom of [-2, 2]), zeta/(area_z·(depth+2)) is the SAME for every reach.
    let abs_f: Vec<f32> = full.abs.into_data().to_vec().unwrap();
    let ratios: Vec<f32> = (0..n)
        .map(|i| abs_f[i] / (area_f[i] * (depth_f[i] + 2.0)))
        .collect();
    for r in &ratios {
        assert!(r.is_finite() && *r > 0.0, "ratio must be positive finite, got {r}");
        assert!(
            (r - ratios[0]).abs() <= 1e-5 * ratios[0],
            "zeta/(area_z·(depth−d_gw)) must be uniform across reaches: {ratios:?}"
        );
    }
}
