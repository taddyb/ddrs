//! A learned stage-roughness exponent must reach BOTH routing entry points.
//!
//! The training path (`training::forward::forward`) reads `gamma` off the
//! head's output map and hands it to the solver, which routes it through the
//! six-parent `TimestepGammaOp`. The eval path (`forward_eval`) is a separate
//! function that mirrors `forward` by hand — so a KAN output that `forward`
//! threads through and `forward_eval` does not is silently scored at
//! `gamma = 0`: the model trained with stage-dependent roughness, the metrics
//! reported for it belong to a different model, and nothing errors.
//!
//! That is exactly what happened on the first learned-gamma CONUS arm
//! (`2026-09-12T13-38-27Z-train-and-test`, caught before its eval finished).
//! This test pins the property directly: for a head that emits `gamma`, the
//! two entry points must produce the same routed hydrograph.
//!
//! The three readers of the head's output map — `forward`, `forward_eval_core`
//! and `probe_forward` — are kept as separate hand-written copies on purpose.
//! The price of that is this table: every optional head output is routed
//! through all three and must agree. Add a row when you add an output.

mod common;

use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use chrono::NaiveDate;
use common::{mock_config, InnerBackend, TestDevice};
use ddrs::data::dataset::RoutingTensors;
use ddrs::data::{RhoWindow, Staid};
use ddrs::nn::kan_head::{KanHead, KanHeadConfig};
use ddrs::sparse::SparseAdjacency;
use ddrs::training::forward::{forward, forward_eval};
use ddrs::training::probe::probe_forward;
use ndarray::Array2;

type AB = Autodiff<InnerBackend>;

/// Linear chain of `n` reaches, `t` hourly steps, one gauge at the outlet.
/// Generic over the backend so the same fixture feeds both entry points.
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
    // Attributes vary per reach so the head emits a per-reach gamma field,
    // not one constant that a scalar fallback could happen to reproduce.
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

fn head(f_attrs: usize, outputs: &[&str], device: &TestDevice) -> KanHead<AB> {
    KanHeadConfig::new(
        (0..f_attrs).map(|i| format!("attr_{i}")).collect(),
        outputs.iter().map(|s| s.to_string()).collect(),
        42,
    )
    .with_hidden_size(8)
    .with_num_hidden_layers(1)
    .init::<AB>(device)
}

fn to_vec<B: Backend>(t: Tensor<B, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// Route the same head through the three readers and return
/// `(train, eval, probe)` gauge hydrographs.
fn all_paths(outputs: &[&str], cfg: &ddrs::config::Config) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = TestDevice::default();
    let (n, t, f) = (5usize, 48usize, 4usize);
    let head_ad = head(f, outputs, &device);
    let head_inner: KanHead<InnerBackend> = head_ad.valid();
    let tensors_ad = minimal_routing_tensors::<AB>(n, t, f, &device);

    let train = forward::<InnerBackend>(cfg, &tensors_ad, &head_ad, &device, false).inner();
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
    // No lifted leaves: the probe must then be the training forward exactly.
    let (probe, _leaves) = probe_forward::<InnerBackend>(cfg, &tensors_ad, &head_ad, &device, &[]);
    (to_vec(train), to_vec(eval), to_vec(probe.inner()))
}

fn both_paths(outputs: &[&str]) -> (Vec<f32>, Vec<f32>) {
    // ddr_match: false, no stage_roughness block, gamma range [0, 0.5]
    let (train, eval, _) = all_paths(outputs, &mock_config());
    (train, eval)
}

fn assert_same(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: shape");
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| ((x - y).abs() / x.abs().max(1e-6)) as f64)
        .fold(0.0_f64, f64::max);
    assert!(worst < 1e-5, "{label}: train and eval paths diverge, worst rel diff {worst:.3e}");
}

/// Control: without gamma the two paths already agree (the fixture is sound).
#[test]
fn train_and_eval_agree_without_gamma() {
    let (train, eval) = both_paths(&["n", "q_spatial", "p_spatial"]);
    assert!(train.iter().all(|v| v.is_finite()));
    assert_same("no gamma", &train, &eval);
}

/// The property: a head that emits `gamma` must be routed identically by
/// training and eval. Before the fix, `forward_eval_core` passed
/// `gamma: None` and this diverged by ~1e-2 relative.
#[test]
fn train_and_eval_agree_with_learned_gamma() {
    let (train, eval) = both_paths(&["n", "q_spatial", "p_spatial", "gamma"]);
    assert!(train.iter().all(|v| v.is_finite()));
    assert_same("learned gamma", &train, &eval);
}

/// Teeth: parity would be vacuous if the head's gamma field were identically
/// zero (the scalar fallback is gamma = 0). A sigmoid read-out cannot emit an
/// exact 0, but pin it so a future read-out change cannot silence the test.
#[test]
fn learned_gamma_field_is_not_zero_at_init() {
    let device = TestDevice::default();
    let f = 4usize;
    let h = head(f, &["n", "q_spatial", "p_spatial", "gamma"], &device);
    let attrs = minimal_routing_tensors::<AB>(5, 2, f, &device).spatial_attributes;
    let out = h.forward(attrs);
    let g: Vec<f32> = out["gamma"].clone().inner().into_data().to_vec().unwrap();
    // Normalised head output in (0, 1); denormalised into [0, 0.5]. A sigmoid
    // read-out cannot produce exactly 0, and the spread must be non-trivial.
    assert!(g.iter().all(|v| *v > 0.05 && *v < 0.95), "gamma at init: {g:?}");
}

/// The table. One row per optional head output; each routes through all three
/// readers. `use_leakance` rows need the flag, the other rows must not set it
/// (the readers panic if the head lacks the leakance keys while it is on).
#[test]
fn every_optional_head_output_reaches_all_three_readers() {
    let base = mock_config();
    let mut leak = mock_config();
    leak.params.use_leakance = true;
    leak.params.use_cuda_graphs = false;
    let mut fixed_q = mock_config();
    fixed_q.params.defaults.insert("q_spatial".to_string(), 0.65);
    let rows: [(&str, &[&str], &ddrs::config::Config); 8] = [
        ("n only (p, q fixed)", &["n"], &fixed_q),
        ("n + gamma (p, q fixed)", &["n", "gamma"], &fixed_q),
        ("n, q only (p fixed)", &["n", "q_spatial"], &base),
        ("p_spatial", &["n", "q_spatial", "p_spatial"], &base),
        ("x_storage", &["n", "q_spatial", "p_spatial", "x_storage"], &base),
        ("gamma", &["n", "q_spatial", "p_spatial", "gamma"], &base),
        (
            "leakance",
            &["n", "q_spatial", "p_spatial", "K_D", "d_gw", "leakance_factor"],
            &leak,
        ),
        (
            "everything",
            &["n", "q_spatial", "p_spatial", "x_storage", "gamma", "K_D", "d_gw", "leakance_factor"],
            &leak,
        ),
    ];
    for (label, outputs, cfg) in rows {
        let (train, eval, probe) = all_paths(outputs, cfg);
        assert!(train.iter().all(|v| v.is_finite()), "{label}: non-finite");
        assert_same(&format!("{label}: eval"), &train, &eval);
        assert_same(&format!("{label}: probe"), &train, &probe);
    }
}
