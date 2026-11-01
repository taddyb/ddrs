//! Tie rule on the MC path: a clamp bound is INACTIVE at an exact tie, so the
//! gradient passes. These tests pin the predicates, the Burn autodiff clamp
//! they agree with, and the fused op at exact ties on four clamp sites.
//!
//! Fixture: linear chain of 4 reaches, 10 km each, the shared mock config.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::clamp_rule::{ceiling_active, floor_active};
use ddrs::routing::mmc_op::{
    __spike_forward_chain_k1_outputs, __spike_forward_chain_k23_outputs, timestep_forward,
};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;
type B = Autodiff<I>;

const N: usize = 4;
const LENGTH_M: f32 = 10_000.0;
const Q_PRIME: [f32; N] = [10.0, 12.0, 14.0, 16.0];

fn mock_cfg() -> Config {
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
    cfg
}

fn chain() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![LENGTH_M; N], vec![1e-3; N])
}

const Q_T: [f32; N] = [100.0, 120.0, 140.0, 160.0];
const Q_T_SMALL: [f32; N] = [0.05, 120.0, 140.0, 160.0];

fn device() -> <I as burn::tensor::backend::BackendTypes>::Device {
    <I as burn::tensor::backend::BackendTypes>::Device::default()
}

/// The k1 hook outputs for the fixture at the given inflow. Index 0 is depth,
/// 5 velocity_unclamped, 18 bw_raw.
fn k1(cfg: &Config, q_t: &[f32; N]) -> Vec<Vec<f32>> {
    let adj = chain();
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let mk = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &dev) };
    __spike_forward_chain_k1_outputs::<I>(
        cfg,
        &pattern,
        mk(&[0.035; N]),
        mk(&[0.4; N]),
        mk(&[20.0; N]),
        mk(q_t),
        mk(&Q_PRIME),
        mk(&adj.length_m),
        mk(&adj.slope),
        mk(&[0.3; N]),
    )
}

/// The solved flow before the discharge floor.
fn x_sol_of(cfg: &Config) -> Vec<f32> {
    let adj = chain();
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let mk = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &dev) };
    __spike_forward_chain_k23_outputs::<I>(
        cfg,
        &pattern,
        mk(&[0.035; N]),
        mk(&[0.4; N]),
        mk(&[20.0; N]),
        mk(&Q_T),
        mk(&Q_PRIME),
        mk(&adj.length_m),
        mk(&adj.slope),
        mk(&[0.3; N]),
    )
    .2
}

/// Gradients of the loss with respect to the five parents. `only` selects one
/// output reach (narrow then sum); `None` sums every reach.
fn parent_grads(cfg: &Config, q_t: &[f32; N], only: Option<usize>) -> Vec<Vec<f32>> {
    let adj = chain();
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let tracked =
        |v: &[f32]| -> Tensor<B, 1> { Tensor::<B, 1>::from_floats(v, &dev).require_grad() };
    let constant = |v: &[f32]| -> Tensor<B, 1> { Tensor::<B, 1>::from_floats(v, &dev) };

    let n = tracked(&[0.035; N]);
    let qsp = tracked(&[0.4; N]);
    let psp = tracked(&[20.0; N]);
    let qt = tracked(q_t);
    let qpt = tracked(&Q_PRIME);
    let q_next = timestep_forward::<I>(
        cfg,
        &pattern,
        &assembler,
        n.clone(),
        qsp.clone(),
        psp.clone(),
        qt.clone(),
        qpt.clone(),
        constant(&adj.length_m),
        constant(&adj.slope),
        constant(&[0.3; N]),
        false,
        None,
        None,
    );
    let loss = match only {
        Some(i) => q_next.narrow(0, i, 1).sum(),
        None => q_next.sum(),
    };
    let grads = loss.backward();
    [&n, &qsp, &psp, &qt, &qpt]
        .iter()
        .map(|t| t.grad(&grads).expect("tracked parent").into_data().to_vec::<f32>().unwrap())
        .collect()
}

fn assert_bitwise_eq(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    for (p, (x, y)) in a.iter().zip(b).enumerate() {
        for (i, (u, v)) in x.iter().zip(y).enumerate() {
            assert_eq!(u.to_bits(), v.to_bits(), "{what}: parent {p} entry {i}: {u} vs {v}");
        }
    }
}

fn bools(t: Tensor<I, 1, burn::tensor::Bool>) -> Vec<bool> {
    t.into_data().to_vec::<bool>().unwrap()
}

#[test]
fn predicates_follow_the_tie_rule() {
    let dev = device();
    let probe = |b: f32| -> Tensor<I, 1> {
        Tensor::from_floats([b.next_down(), b, b.next_up(), f32::NAN], &dev)
    };
    assert_eq!(bools(floor_active(probe(0.5), 0.5)), [true, false, false, true]);
    assert_eq!(bools(floor_active(probe(0.0), 0.0)), [true, false, false, true]);
    assert_eq!(bools(ceiling_active(probe(50.0), 50.0)), [false, false, true, true]);
    assert_eq!(bools(ceiling_active(probe(15.0), 15.0)), [false, false, true, true]);
    assert_eq!(bools(ceiling_active(probe(0.5), 0.5)), [false, false, true, true]);
}

/// Pins the Burn fork: a clamp passes the gradient at an exact tie.
#[test]
fn burn_autodiff_clamp_passes_gradient_at_exact_tie() {
    let dev = device();
    let grad_of = |x: f32, f: &dyn Fn(Tensor<B, 1>) -> Tensor<B, 1>| -> f32 {
        let t = Tensor::<B, 1>::from_floats([x], &dev).require_grad();
        let g = f(t.clone()).sum().backward();
        t.grad(&g).unwrap().into_data().to_vec::<f32>().unwrap()[0]
    };
    assert_eq!(grad_of(1.0, &|t| t.clamp_min(1.0)), 1.0);
    assert_eq!(grad_of(0.0, &|t| t.clamp(0.0, 1.0)), 1.0);
    assert_eq!(grad_of(1.0, &|t| t.clamp(0.0, 1.0)), 1.0);
}

#[test]
fn discharge_floor_tie_passes_and_one_ulp_below_cuts() {
    let x_sol = x_sol_of(&mock_cfg());
    let (i_min, r) = x_sol
        .iter()
        .copied()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    assert!(r > 0.0);
    let run = |lb: f32| {
        let mut cfg = mock_cfg();
        cfg.params.attribute_minimums.discharge = lb;
        parent_grads(&cfg, &Q_T, Some(i_min))
    };
    let a = run(r / 2.0);
    let b = run(r);
    let c = run(r.next_up());
    assert_bitwise_eq(&a, &b, "discharge tie vs inactive");
    assert!(a.iter().flatten().any(|g| *g != 0.0), "reference gradient must be nonzero");
    for (p, g) in c.iter().enumerate() {
        assert!(g.iter().all(|v| *v == 0.0), "clamped entry must cut parent {p}: {g:?}");
    }
}

#[test]
fn depth_floor_tie_passes() {
    let mut cfg = mock_cfg();
    cfg.params.attribute_minimums.depth = 1e-6;
    let r = k1(&cfg, &Q_T_SMALL)[0][0];
    let run = |lb: f32| {
        let mut cfg = mock_cfg();
        cfg.params.attribute_minimums.depth = lb;
        parent_grads(&cfg, &Q_T_SMALL, None)
    };
    assert_bitwise_eq(&run(r / 2.0), &run(r), "depth floor tie");
}

#[test]
fn bottom_width_floor_tie_passes() {
    let mut cfg = mock_cfg();
    cfg.params.attribute_minimums.bottom_width = 1e-6;
    let r = k1(&cfg, &Q_T_SMALL)[18][0];
    let run = |lb: f32| {
        let mut cfg = mock_cfg();
        cfg.params.attribute_minimums.bottom_width = lb;
        parent_grads(&cfg, &Q_T_SMALL, None)
    };
    assert_bitwise_eq(&run(r / 2.0), &run(r), "bottom width floor tie");
}

#[test]
fn velocity_floor_tie_passes() {
    let mut cfg = mock_cfg();
    cfg.params.attribute_minimums.velocity = 1e-6;
    let r = k1(&cfg, &Q_T_SMALL)[5][0];
    let run = |lb: f32| {
        let mut cfg = mock_cfg();
        cfg.params.attribute_minimums.velocity = lb;
        parent_grads(&cfg, &Q_T_SMALL, None)
    };
    assert_bitwise_eq(&run(r / 2.0), &run(r), "velocity floor tie");
}
