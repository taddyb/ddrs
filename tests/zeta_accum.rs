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
//!   4. The accumulated `q` sum equals the summed routed discharge columns
//!      (`q_next` is the same tensor that becomes the output column).
//!   5. `depth`/`area_z` are the raw geometry primitives: invariant to
//!      `leakance_factor`, and `zeta/(area_z·(depth − d_gw))` is uniform.
//!   6. Volume accounting over a multi-timestep window (see the section at the
//!      bottom of this file): the accumulated flux is the per-step flux times
//!      the window, and the routed discharge deficit accounts for all of it.

mod common;

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use common::{
    mock_config, mock_routing_inputs, mock_spatial_parameters, mock_streamflow, InnerBackend,
    TestDevice,
};
use ddrs::routing::{MuskingumCunge, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

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
        impervious_mask: None,
        gamma: None,
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
        false, None
    );
    let _ = mc.forward();
    assert!(mc.zeta_sums().is_none(), "no leakance params ⇒ no zeta sums");

    // Leakance on, accumulation NOT enabled → None.
    let mut mc = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        leakance_params(n, 1.0, &device),
        false, None
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
        false, None
    );
    let out_plain = forward_vec(&mut mc_plain);

    let mut mc_accum = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc_accum.enable_zeta_accumulation();
    mc_accum.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        leakance_params(n, 1.0, &device),
        false, None
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
        false, None
    );
    let out_off = forward_vec(&mut mc_off); // [n, 2] row-major

    let mut mc_on = MuskingumCunge::<InnerBackend>::new(cfg, device.clone());
    mc_on.enable_zeta_accumulation();
    mc_on.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        leakance_params(n, 1.0, &device),
        false, None
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
            false, None
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
        false, None
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
            false, None
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

// ---------------------------------------------------------------------------
// PC2: Impervious mask tests on eval-time zeta accumulation
// ---------------------------------------------------------------------------

/// With mask[2]=0, reach 2's accumulated |zeta| must be exactly 0.
/// Other reaches (losing config, d_gw=−2 m < depth) must have zeta > 0.
#[test]
fn zeta_accum_masked_reach_zero() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 2usize);
    let cfg = mock_config();

    // Build a mask tensor (inner backend) with mask[2] = 0.
    use burn::backend::NdArray;
    let mask_data = vec![1.0f32, 1.0, 0.0, 1.0, 1.0];
    let mask: burn::tensor::Tensor<NdArray<f32>, 1> =
        burn::tensor::Tensor::from_floats(mask_data.as_slice(), &device);

    let mut params = leakance_params(n, 1.0, &device);
    params.impervious_mask = Some(mask);

    let mut mc = MuskingumCunge::new(cfg, device.clone());
    mc.enable_zeta_accumulation();
    mc.setup_inputs(
        mock_routing_inputs(n, &device),
        mock_streamflow(t, n, &device),
        params,
        false,
        None,
    );
    let _ = mc.forward();
    let sums = mc.zeta_sums().expect("zeta sums present");
    let abs_v: Vec<f32> = sums.abs.into_data().to_vec().unwrap();

    assert_eq!(abs_v.len(), n);
    assert!(
        abs_v[2].abs() < 1e-9,
        "masked reach 2: accumulated |zeta| must be 0, got {:.3e}", abs_v[2]
    );
    for i in [0usize, 1, 3, 4] {
        assert!(
            abs_v[i] > 0.0,
            "unmasked losing reach {i}: accumulated |zeta| must be positive, got {:.3e}", abs_v[i]
        );
    }
}

/// All-ones mask must produce the same accumulated zeta as no-mask (None).
#[test]
fn zeta_accum_all_ones_mask_same_as_no_mask() {
    let device = TestDevice::default();
    let (n, t) = (5usize, 6usize);
    let cfg = mock_config();

    use burn::backend::NdArray;
    let all_ones: burn::tensor::Tensor<NdArray<f32>, 1> =
        burn::tensor::Tensor::ones([n], &device);

    let run = |mask: Option<burn::tensor::Tensor<NdArray<f32>, 1>>| -> Vec<f32> {
        let mut params = leakance_params(n, 1.0, &device);
        params.impervious_mask = mask;
        let mut mc = MuskingumCunge::new(cfg.clone(), device.clone());
        mc.enable_zeta_accumulation();
        mc.setup_inputs(
            mock_routing_inputs(n, &device),
            mock_streamflow(t, n, &device),
            params,
            false,
            None,
        );
        let _ = mc.forward();
        mc.zeta_sums().expect("zeta sums present").abs.into_data().to_vec().unwrap()
    };

    let abs_no_mask = run(None);
    let abs_ones_mask = run(Some(all_ones));
    assert_eq!(
        abs_no_mask, abs_ones_mask,
        "all-ones mask must produce byte-identical zeta accumulation as no-mask"
    );
}

// ---------------------------------------------------------------------------
// Volume accounting over a multi-timestep window.
//
// `accumulated_zeta_equals_headwater_qnext_difference` above proves the
// SINGLE-step identity: one routed step removes exactly `zeta[0]` from the
// headwater's discharge. Over a window that stops being true step for step,
// because the Muskingum scheme releases only part of each step's deficit and
// carries the rest forward as channel storage. The books still have to
// balance, and these tests close them.
//
// The accounting needs the routing to be linear and time-invariant over the
// window, which it is not in general: `c1..c4` depend on the depth (S6) and,
// through the Cunge `X` (S19 of `src/routing/mmc_op.rs::forward_chain_inner`),
// on the discharge itself. Both dependencies are removed deliberately here:
//
//   * `attribute_minimums.depth` is raised far above the depth this discharge
//     would produce, so S6 saturates at the floor and the whole trapezoidal
//     geometry, the celerity, and `zeta` itself become constants.
//   * the reach length and slope are chosen so `X = 0.5·(1 − Q/(B·S·c·L))`
//     saturates at its lower clamp of 0 for every reach and every timestep.
//
// Neither is assumed. `depth_is_pinned_so_the_window_is_lti` checks the first
// directly, and `window_discharge_deficit_accounts_for_the_summed_flux`
// checks the second the strong way, by showing the measured deficit obeys a
// single-coefficient recursion across the whole window.
// ---------------------------------------------------------------------------

/// Depth floor for the LTI window, metres. `mock_spatial_parameters` +
/// `mock_streamflow` route at O(10) m³/s, whose unpinned depth is O(1) m, so
/// S6 saturates at every reach and every timestep by a wide margin.
const LTI_DEPTH_LB: f32 = 50.0;
/// Reach length, metres. At the pinned depth the celerity is ≈ 0.33 m/s, so
/// `K = L/c ≈ 5400 s = 1.5·dt`, putting `c3 = (2K − dt)/(2K + dt)` near 0.5 —
/// far enough from 0 that the storage term in the identity below is not
/// vacuous, and far enough from 1 that the window is well conditioned.
const LTI_LENGTH_M: f32 = 1790.0;
/// Reach slope. Together with the length this keeps `Q/(B·S·c·L) > 1` — hence
/// the Cunge `X` at its lower clamp — for the whole `mock_streamflow` sweep.
/// `attribute_minimums.slope` has to come down with it or the routing would
/// substitute the 1e-3 floor and `X` would float.
const LTI_SLOPE: f32 = 1.0e-4;
const LTI_STEPS: usize = 25;

fn lti_config() -> ddrs::config::Config {
    let mut cfg = mock_config();
    cfg.params.attribute_minimums.depth = LTI_DEPTH_LB;
    cfg.params.attribute_minimums.slope = 1.0e-6;
    cfg
}

fn lti_routing_inputs(n: usize, device: &TestDevice) -> ddrs::routing::RoutingInputs<InnerBackend> {
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    ddrs::routing::RoutingInputs {
        adjacency: SparseAdjacency::from_dense(
            n,
            &dense,
            vec![LTI_LENGTH_M; n],
            vec![LTI_SLOPE; n],
        ),
        x_storage: Tensor::<AB, 1>::ones([n], device) * 0.2,
    }
}

/// Route the chain for `t` timesteps and return `(hydrograph [n, t] row-major,
/// zeta sums)`. `leakance = false` routes without the term and yields no sums.
fn run_lti_window(
    n: usize,
    t: usize,
    leakance: bool,
    device: &TestDevice,
) -> (Vec<f32>, Option<ddrs::routing::ZetaSumTensors<InnerBackend>>) {
    let params = if leakance {
        leakance_params(n, 1.0, device)
    } else {
        mock_spatial_parameters(n, device)
    };
    let mut mc = MuskingumCunge::<InnerBackend>::new(lti_config(), device.clone());
    if leakance {
        mc.enable_zeta_accumulation();
    }
    mc.setup_inputs(
        lti_routing_inputs(n, device),
        mock_streamflow(t, n, device),
        params,
        false,
        None,
    );
    let out = forward_vec(&mut mc);
    (out, mc.zeta_sums())
}

/// First premise: with the floor raised, the shared depth saturates for every
/// reach on every accumulated step, so the summed depth is exactly
/// `steps · depth_lb` and the per-step flux cannot vary in time.
#[test]
fn depth_is_pinned_so_the_window_is_lti() {
    let device = TestDevice::default();
    let n = 5usize;
    let (_, sums) = run_lti_window(n, LTI_STEPS, true, &device);
    let sums = sums.expect("zeta sums present");
    assert_eq!(sums.steps, LTI_STEPS - 1);
    let depth: Vec<f32> = sums.depth.into_data().to_vec().unwrap();
    let expected = (LTI_STEPS - 1) as f32 * LTI_DEPTH_LB;
    for (i, d) in depth.iter().enumerate() {
        assert_eq!(
            *d, expected,
            "reach {i}: summed depth {d} != steps·depth_lb {expected}; the depth \
             floor did not saturate, so the window is not LTI and the volume \
             identities here do not apply"
        );
    }
}

/// The accumulator is the summed flux, not a sample of it: over the window it
/// holds exactly `steps` times the single-step flux, for every reach. This is
/// the "summed over reaches and timesteps" half of the accounting, and it is
/// what converts the diagnostic into a volume (`Σ zeta · dt`).
#[test]
fn accumulated_flux_is_the_per_step_flux_times_the_window() {
    let device = TestDevice::default();
    let n = 5usize;

    // One routed step: `sums.net` IS the per-step flux.
    let (_, one) = run_lti_window(n, 2, true, &device);
    let one = one.expect("zeta sums present");
    assert_eq!(one.steps, 1);
    let z: Vec<f32> = one.net.into_data().to_vec().unwrap();

    let (_, many) = run_lti_window(n, LTI_STEPS, true, &device);
    let many = many.expect("zeta sums present");
    let steps = many.steps;
    assert_eq!(steps, LTI_STEPS - 1);
    let net: Vec<f32> = many.net.into_data().to_vec().unwrap();

    let mut volume_m3 = 0.0_f64;
    for i in 0..n {
        assert!(z[i] > 0.0, "reach {i}: losing regime expected, flux is {}", z[i]);
        let expected = steps as f32 * z[i];
        assert!(
            (net[i] - expected).abs() <= 1e-6 * expected,
            "reach {i}: accumulated flux {} != steps ({steps}) × per-step flux {} = {expected}",
            net[i], z[i]
        );
        volume_m3 += net[i] as f64 * ddrs::routing::mmc::DT_SECONDS as f64;
    }
    eprintln!(
        "zeta volume over {steps} steps × {n} reaches: {volume_m3:.6e} m³ \
         (per-step flux {:.6e} m³/s per reach)",
        z[0]
    );
}

/// Volume accounting proper. In the LTI window the headwater's discharge
/// deficit `e_k = q_off[k] − q_on[k]` obeys `e_{k+1} = c3·e_k + z` exactly:
/// the first row of `A` is the identity (no upstream), so `q_next[0]` is
/// `b_rhs[0] = c3·q_t + c4·q' − zeta`, and everything but `zeta` cancels
/// between the two runs. `e_0 = 0` because the hot start is
/// leakance-independent. Summing that recursion over the window closes the
/// books:
///
/// ```text
///   (1 − c3)·Σ_k e_k  +  c3·e_last  =  steps·z  =  the accumulated flux
/// ```
///
/// Read physically: every unit of flux removed over the window leaves the
/// system either as reduced discharge (`Σ_k e_k`, weighted by the fraction
/// `1 − c3` each step actually releases) or as the storage deficit still in
/// transit when the window ends (`c3·e_last`). Nothing else absorbs it.
///
/// `c3` is not assumed and not taken from the solver. It is recovered from the
/// first two deficits and then has to reproduce every remaining step of the
/// window — which is also the check that the Cunge `X` stayed pinned, since a
/// floating `X` makes `c3` drift step to step (it does, visibly, at the stock
/// 1e-3 slope floor).
#[test]
fn window_discharge_deficit_accounts_for_the_summed_flux() {
    let device = TestDevice::default();
    let n = 5usize;
    let t = LTI_STEPS;

    let (off, _) = run_lti_window(n, t, false, &device);
    let (on, sums) = run_lti_window(n, t, true, &device);
    let sums = sums.expect("zeta sums present");
    let steps = sums.steps;
    let flux = sums.net.into_data().to_vec::<f32>().unwrap()[0];
    let z = flux / steps as f32;

    // Reach 0 is the head of the chain, so its row of `A` is [1, 0, …].
    let e: Vec<f32> = (0..t).map(|k| off[k] - on[k]).collect();
    assert_eq!(e[0], 0.0, "hot start must be leakance-independent");
    assert!(
        (e[1] - z).abs() <= 1e-6 * z,
        "first routed step must remove exactly the flux: e[1] = {}, z = {z}",
        e[1]
    );

    // Second premise, checked rather than assumed: one coefficient explains
    // the whole window.
    let c3 = (e[2] - z) / e[1];
    assert!(
        c3.abs() < 1.0,
        "recovered c3 = {c3} is outside (−1, 1); the window is not decaying"
    );
    let mut max_c3_spread = 0.0_f32;
    for k in 1..t - 1 {
        let c3_k = (e[k + 1] - z) / e[k];
        max_c3_spread = max_c3_spread.max((c3_k - c3).abs() / c3.abs());
    }
    assert!(
        max_c3_spread < 1e-4,
        "c3 drifts by {max_c3_spread:.3e} across the window, so the routing is \
         not time-invariant here (the Cunge X is floating) and the accounting \
         identity does not hold"
    );

    let sum_e: f32 = e.iter().sum();
    let released = (1.0 - c3) * sum_e;
    let in_transit = c3 * e[t - 1];
    let accounted = released + in_transit;
    let rel = ((accounted - flux) / flux).abs();

    eprintln!(
        "headwater volume accounting over {steps} steps: released {released:.6e} + \
         in transit {in_transit:.6e} = {accounted:.6e} m³/s·steps vs accumulated \
         flux {flux:.6e} (c3 = {c3:.6}, spread {max_c3_spread:.3e}, rel diff {rel:.3e})"
    );
    // The f32 floor, and nothing above it: `sum_e` adds 25 same-sign terms of
    // O(e) and the weights are O(1), so a few tens of roundings bound the
    // identity at ~1e-6 relative. Measured 1.2e-7, i.e. single-rounding; the
    // gate below sits at 1e-6 so any structural leak shows up rather than
    // hiding under a wide bar.
    assert!(
        rel <= 1e-6,
        "volume not conserved: the discharge deficit accounts for {accounted:.9e} \
         m³/s·steps but {flux:.9e} was removed as flux (rel diff {rel:.3e})"
    );
}
