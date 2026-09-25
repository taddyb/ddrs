//! Temperature-annealed binary gate on the head's `leakance_factor` output
//! (`src/training/gate.rs::leakance_gate`, config `params.leakance_gate`).
//!
//! Three layers, in priority order:
//!
//! 1. `tau = 1` is a BYTE-exact identity, both on the bare function and end
//!    to end through all three head readers (`forward`, `forward_eval`,
//!    `probe_forward`) against the same config with the block absent. This
//!    is the guarantee that lets the block ship absent-by-default.
//! 2. The gate's gradient is the exact autograd gradient of
//!    `sigmoid(logit(clamp(u)) / tau)`, checked against central finite
//!    differences (of the formula in f64, and of the f32 implementation at
//!    its noise floor) at several temperatures, including a sharp one. No
//!    straight-through estimator anywhere.
//! 3. Numerical safety: monotone in `u`, the `tau -> 0` limits are 0 below
//!    0.5 and 1 above, and a temperature so small the sigmoid saturates
//!    yields finite values and finite gradients even for `u` exactly 0 or 1.

mod common;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use chrono::NaiveDate;
use common::{mock_config, InnerBackend, TestDevice};
use ddrs::config::{Config, LeakanceGate};
use ddrs::data::dataset::RoutingTensors;
use ddrs::data::{RhoWindow, Staid};
use ddrs::nn::kan_head::{KanHead, KanHeadConfig};
use ddrs::sparse::SparseAdjacency;
use ddrs::training::forward::{forward, forward_eval};
use ddrs::training::gate::{leakance_gate, GATE_EPS};
use ddrs::training::probe::probe_forward;
use ndarray::Array2;
use std::collections::BTreeMap;

type I = NdArray<f32>;
type AB = Autodiff<InnerBackend>;

// Same acceptance rule as the repo's other gradchecks (`references/testing.md`).
const REL_TOL: f32 = 5e-3;
const ABS_TOL: f32 = 1e-4;

fn gate_vec(u: &[f32], tau: f32) -> Vec<f32> {
    let device = Default::default();
    let t = Tensor::<I, 1>::from_data(TensorData::new(u.to_vec(), [u.len()]), &device);
    leakance_gate(t, tau).into_data().to_vec::<f32>().unwrap()
}

/// f64 evaluation of the same formula with the implementation's f32 clamp
/// bounds (`1.0_f32 - GATE_EPS` is ~17 ulps below 1, not the real number
/// `1 - 1e-6`). Test-side only: the routing core and the gate stay f32.
fn reference(u: f64, tau: f64) -> f64 {
    let lo = GATE_EPS as f64;
    let hi = (1.0_f32 - GATE_EPS) as f64;
    let u = u.clamp(lo, hi);
    let logit = (u / (1.0 - u)).ln();
    1.0 / (1.0 + (-logit / tau).exp())
}

fn sweep() -> Vec<f32> {
    let mut u: Vec<f32> = (0..=1000).map(|k| k as f32 / 1000.0).collect();
    u.extend_from_slice(&[1e-7, 1e-6, 0.5 - 1e-6, 0.5 + 1e-6, 1.0 - 1e-6, 0.999_999_9]);
    u
}

// ---------------------------------------------------------------------------
// 1. Identity at tau = 1
// ---------------------------------------------------------------------------

#[test]
fn tau_one_is_bit_exact_identity() {
    let u = sweep();
    let g = gate_vec(&u, 1.0);
    for (a, b) in u.iter().zip(&g) {
        assert_eq!(a.to_bits(), b.to_bits(), "tau=1 must return u untouched: {a} -> {b}");
    }
}

/// Without the special case, `sigmoid(logit(u))` would NOT be bit-exact in
/// f32: pin that the round trip really does perturb values, so the identity
/// test above is testing something.
#[test]
fn round_trip_without_special_case_is_not_exact() {
    let u = sweep();
    // tau = 1 + 1 ulp takes the general branch with a negligible sharpening.
    let g = gate_vec(&u, f32::from_bits(1.0_f32.to_bits() + 1));
    let differs = u.iter().zip(&g).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert!(differs > 0, "expected the ln/exp round trip to perturb some values");
}

// ---------------------------------------------------------------------------
// 2. Values and gradients
// ---------------------------------------------------------------------------

#[test]
fn matches_reference_formula() {
    let u = sweep();
    for tau in [2.0_f32, 0.5, 0.1, 0.02] {
        let g = gate_vec(&u, tau);
        for (a, b) in u.iter().zip(&g) {
            let r = reference(*a as f64, tau as f64) as f32;
            assert!((r - b).abs() < 2e-6, "tau={tau} u={a}: got {b}, reference {r}");
        }
    }
}

/// The autograd gradient of the f32 gate against central finite differences.
///
/// Two references. (a) The formula evaluated in f64 with a step of 1e-6: the
/// standard gradcheck practice, and the one held to the repo's tolerance
/// (`rel < 5e-3` or `abs < 1e-4`). (b) The f32 implementation itself with a
/// step of 2.5e-4: what the gate really computes, but a difference of two
/// values near 1.0 is quantized at `ulp(1) / 2h ≈ 1.2e-4`, so it is held to an
/// absolute floor of 1e-3 (eight times that noise) where the relative test
/// fails. The step is below the repo's usual 1e-3 because the sharpest
/// temperature has `O((4h/tau)^2 / 12)` truncation at `u = 0.5`.
#[test]
fn gradient_matches_central_differences() {
    let device = Default::default();
    // Interior of (0, 1), on both sides of and at the switch point.
    let u: Vec<f32> = vec![0.2, 0.35, 0.45, 0.5, 0.55, 0.65, 0.8];
    let h32 = 2.5e-4_f32;
    let h64 = 1e-6_f64;
    const F32_FD_ABS_FLOOR: f32 = 1e-3;
    for tau in [1.0_f32, 0.5, 0.2, 0.05, 0.01] {
        let leaf = Tensor::<Autodiff<I>, 1>::from_data(TensorData::new(u.clone(), [u.len()]), &device)
            .require_grad();
        let loss = leakance_gate(leaf.clone(), tau).sum();
        let grads = loss.backward();
        let analytic: Vec<f32> = leaf.grad(&grads).unwrap().into_data().to_vec().unwrap();

        for (k, &uk) in u.iter().enumerate() {
            let a = analytic[k];
            assert!(a.is_finite(), "tau={tau} u={uk}: non-finite gradient");

            // (a) f64 central difference of the formula.
            let fd64 = ((reference(uk as f64 + h64, tau as f64) - reference(uk as f64 - h64, tau as f64))
                / (2.0 * h64)) as f32;
            let abs = (a - fd64).abs();
            let rel = abs / fd64.abs().max(1e-12);
            assert!(
                rel < REL_TOL || abs < ABS_TOL,
                "tau={tau} u={uk}: analytic {a:.6e} vs f64 central-difference {fd64:.6e} (rel {rel:.2e}, abs {abs:.2e})"
            );

            // (b) f32 central difference of the implementation itself.
            let plus = gate_vec(&[uk + h32], tau)[0];
            let minus = gate_vec(&[uk - h32], tau)[0];
            let fd32 = (plus - minus) / (2.0 * h32);
            let abs = (a - fd32).abs();
            let rel = abs / fd32.abs().max(1e-12);
            assert!(
                rel < REL_TOL || abs < F32_FD_ABS_FLOOR,
                "tau={tau} u={uk}: analytic {a:.6e} vs f32 central-difference {fd32:.6e} (rel {rel:.2e}, abs {abs:.2e})"
            );
        }
        if tau == 1.0 {
            // The identity branch: gradient is exactly one everywhere.
            assert!(analytic.iter().all(|g| *g == 1.0), "tau=1 gradient must be exactly 1: {analytic:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Monotonicity, limits, saturation
// ---------------------------------------------------------------------------

#[test]
fn monotone_in_u_and_fixed_at_half() {
    let u: Vec<f32> = (1..100).map(|k| k as f32 / 100.0).collect();
    for tau in [1.0_f32, 0.5, 0.1, 0.01, 1e-4] {
        let g = gate_vec(&u, tau);
        for w in g.windows(2) {
            assert!(w[1] >= w[0], "tau={tau}: not monotone at {w:?}");
        }
        assert!(g.iter().all(|v| (0.0..=1.0).contains(v)), "tau={tau}: outside [0, 1]");
        // logit(0.5) = ln(0.5) - ln(0.5) is exactly 0 in f32, so 0.5 is a
        // fixed point at every temperature.
        assert_eq!(gate_vec(&[0.5], tau)[0], 0.5, "tau={tau}");
    }
}

#[test]
fn small_temperature_limits_are_zero_below_half_and_one_above() {
    let below: Vec<f32> = vec![0.01, 0.2, 0.4, 0.49];
    let above: Vec<f32> = vec![0.51, 0.6, 0.8, 0.99];
    // logit(0.49) = -0.04, so tau = 1e-3 already puts it at sigmoid(-40).
    for tau in [1e-3_f32, 1e-4, 1e-5] {
        for v in gate_vec(&below, tau) {
            assert!(v < 1e-3, "tau={tau}: below-half value {v} did not go to 0");
        }
        for v in gate_vec(&above, tau) {
            assert!(v > 1.0 - 1e-3, "tau={tau}: above-half value {v} did not go to 1");
        }
    }
    // Sharpening is monotone in tau: the same u moves further from 0.5 as
    // tau falls.
    let u = 0.6_f32;
    let seq: Vec<f32> = [1.0_f32, 0.5, 0.2, 0.1, 0.05].iter().map(|&t| gate_vec(&[u], t)[0]).collect();
    for w in seq.windows(2) {
        assert!(w[1] >= w[0], "sharpening not monotone in tau: {seq:?}");
    }
}

#[test]
fn saturating_temperature_is_finite_forward_and_backward() {
    let device = Default::default();
    // Includes exact 0 and 1, which a saturated head sigmoid can emit and
    // which are only safe because of the clamp before the logit.
    let u: Vec<f32> = vec![0.0, 1e-8, 0.3, 0.5, 0.7, 1.0 - 1e-8, 1.0];
    for tau in [1e-6_f32, 1e-12, 1e-30, f32::MIN_POSITIVE] {
        let g = gate_vec(&u, tau);
        assert!(g.iter().all(|v| v.is_finite()), "tau={tau}: non-finite forward {g:?}");
        assert!(g.iter().all(|v| (0.0..=1.0).contains(v)), "tau={tau}: outside [0, 1] {g:?}");
        assert!(g[0] == 0.0 && g[6] == 1.0 && g[3] == 0.5, "tau={tau}: limits {g:?}");

        let leaf = Tensor::<Autodiff<I>, 1>::from_data(TensorData::new(u.clone(), [u.len()]), &device)
            .require_grad();
        let grads = leakance_gate(leaf.clone(), tau).sum().backward();
        let grad: Vec<f32> = leaf.grad(&grads).unwrap().into_data().to_vec().unwrap();
        assert!(grad.iter().all(|v| v.is_finite()), "tau={tau}: non-finite gradient {grad:?}");
    }
}

// ---------------------------------------------------------------------------
// 4. End to end through the three head readers
// ---------------------------------------------------------------------------

/// Linear chain of `n` reaches, `t` hourly steps, one gauge at the outlet.
/// Mirrors `tests/gamma_eval_parity.rs::minimal_routing_tensors`.
fn minimal_routing_tensors<B: Backend>(
    n: usize,
    t: usize,
    f_attrs: usize,
    device: &B::Device,
) -> RoutingTensors<B> {
    let adjacency = {
        let mut dense = vec![0.0_f32; n * n];
        for i in 0..n - 1 {
            dense[(i + 1) * n + i] = 1.0;
        }
        SparseAdjacency::from_dense(n, &dense, vec![1000.0; n], vec![0.001; n])
    };
    let attrs_vec: Vec<f32> = (0..n * f_attrs)
        .map(|k| 0.2 + 0.6 * ((k * 7) % 11) as f32 / 10.0)
        .collect();
    let spatial_attributes =
        Tensor::<B, 2>::from_data(TensorData::new(attrs_vec, [n, f_attrs]), device);
    let mut qp = vec![0.0_f32; t * n];
    for ti in 0..t {
        let phase = (ti as f32) / (t.max(2) - 1) as f32 * 4.0 * std::f32::consts::PI;
        for ri in 0..n {
            qp[ti * n + ri] = (5.0 + phase.sin() * 2.0).max(0.1);
        }
    }
    let q_prime = Tensor::<B, 2>::from_data(TensorData::new(qp, [t, n]), device);
    let empty = |d: &B::Device| Tensor::<B, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, n]), d);
    RoutingTensors {
        adjacency,
        spatial_attributes,
        q_prime,
        q_prime_daily: empty(device),
        precip_hourly: empty(device),
        temp_hourly: empty(device),
        observations: Array2::zeros((1, 1)),
        flat_indices: Tensor::<B, 1, Int>::from_data(TensorData::from([(n - 1) as i32].as_slice()), device),
        group_ids: Tensor::<B, 1, Int>::from_data(TensorData::from([0i32].as_slice()), device),
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

const LEAK_OUTPUTS: [&str; 6] = ["n", "q_spatial", "p_spatial", "K_D", "d_gw", "leakance_factor"];

fn leak_head(f_attrs: usize, device: &TestDevice) -> KanHead<AB> {
    KanHeadConfig::new(
        (0..f_attrs).map(|i| format!("attr_{i}")).collect(),
        LEAK_OUTPUTS.iter().map(|s| s.to_string()).collect(),
        42,
    )
    .with_hidden_size(8)
    .with_num_hidden_layers(1)
    .init::<AB>(device)
}

fn leak_config(schedule: Option<&[(usize, f32)]>) -> Config {
    let mut cfg = mock_config();
    cfg.params.use_leakance = true;
    cfg.params.use_cuda_graphs = false;
    cfg.params.leakance_gate = schedule.map(|s| LeakanceGate {
        temperature: s.iter().copied().collect::<BTreeMap<usize, f32>>(),
    });
    cfg
}

/// `(train, eval, probe)` gauge hydrographs for one config. Training takes
/// the schedule's FINAL temperature so the three readers see the same model.
fn three_readers(cfg: &Config) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = TestDevice::default();
    let (n, t, f) = (5usize, 48usize, 4usize);
    let head_ad = leak_head(f, &device);
    let head_inner: KanHead<InnerBackend> = head_ad.valid();
    let tensors_ad = minimal_routing_tensors::<AB>(n, t, f, &device);
    let gate_tau = cfg.params.leakance_gate.as_ref().map(|g| g.final_temperature());

    let train = forward::<InnerBackend>(cfg, &tensors_ad, &head_ad, &device, false, gate_tau).inner();
    let eval = forward_eval::<InnerBackend>(
        cfg,
        &minimal_routing_tensors::<InnerBackend>(n, t, f, &device),
        &head_inner,
        &device,
        false,
        None,
        None,
        None,
    );
    let (probe, _leaves) = probe_forward::<InnerBackend>(cfg, &tensors_ad, &head_ad, &device, &[]);
    let v = |t: Tensor<InnerBackend, 2>| t.into_data().to_vec::<f32>().unwrap();
    (v(train), v(eval), v(probe.inner()))
}

fn assert_bits_equal(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: shape");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}: bit mismatch at {k}: {x} vs {y}");
    }
}

/// THE parity test: a config whose gate block schedules `tau = 1` must route
/// byte-identically to the same config without the block, in all three
/// readers. The block absent is the historical code path.
#[test]
fn gate_at_tau_one_is_byte_exact_through_all_readers() {
    let (train_off, eval_off, probe_off) = three_readers(&leak_config(None));
    let (train_on, eval_on, probe_on) = three_readers(&leak_config(Some(&[(1, 1.0)])));
    assert!(train_off.iter().all(|v| v.is_finite()));
    assert_bits_equal("forward", &train_off, &train_on);
    assert_bits_equal("forward_eval", &eval_off, &eval_on);
    assert_bits_equal("probe_forward", &probe_off, &probe_on);
}

/// Teeth for the parity test above: a sharp final temperature must actually
/// move the routed hydrograph, in all three readers, or the identity could be
/// vacuous (the readers silently dropping the gate would pass it).
#[test]
fn gate_at_sharp_tau_changes_output_in_all_readers() {
    let (train_off, eval_off, probe_off) = three_readers(&leak_config(None));
    let (train_on, eval_on, probe_on) = three_readers(&leak_config(Some(&[(1, 1.0), (2, 0.05)])));
    let max_diff = |a: &[f32], b: &[f32]| {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    };
    assert!(train_on.iter().all(|v| v.is_finite()), "gated forward produced non-finite values");
    let d_train = max_diff(&train_off, &train_on);
    let d_eval = max_diff(&eval_off, &eval_on);
    let d_probe = max_diff(&probe_off, &probe_on);
    assert!(d_train > 0.0, "forward: gate at tau=0.05 was a no-op");
    assert!(d_eval > 0.0, "forward_eval: gate at tau=0.05 was a no-op");
    assert!(d_probe > 0.0, "probe_forward: gate at tau=0.05 was a no-op");
    // And the three readers still agree with each other on the gated model.
    assert_bits_equal("gated: forward vs probe", &train_on, &probe_on);
    let worst = train_on
        .iter()
        .zip(&eval_on)
        .map(|(x, y)| (x - y).abs() / x.abs().max(1e-6))
        .fold(0.0_f32, f32::max);
    assert!(worst < 1e-5, "gated: forward vs forward_eval diverge, worst rel diff {worst:.3e}");
}

/// `forward` refuses a temperature that contradicts the config, in both
/// directions, so a caller cannot train the wrong model silently.
#[test]
#[should_panic(expected = "gate_tau")]
fn forward_rejects_temperature_without_gate_block() {
    let device = TestDevice::default();
    let cfg = leak_config(None);
    let head = leak_head(4, &device);
    let tensors = minimal_routing_tensors::<AB>(5, 48, 4, &device);
    let _ = forward::<InnerBackend>(&cfg, &tensors, &head, &device, false, Some(0.5));
}

#[test]
#[should_panic(expected = "gate_tau")]
fn forward_rejects_missing_temperature_with_gate_block() {
    let device = TestDevice::default();
    let cfg = leak_config(Some(&[(1, 0.5)]));
    let head = leak_head(4, &device);
    let tensors = minimal_routing_tensors::<AB>(5, 48, 4, &device);
    let _ = forward::<InnerBackend>(&cfg, &tensors, &head, &device, false, None);
}
