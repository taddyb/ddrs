//! A single stage pool node with release `R = S/T` reproduces the Muskingum
//! option C reservoir (`MuskingumCunge::set_reservoir_rows`: `K = T`, `X = 0`).
//!
//! Setup: reach 0 is an MC headwater (q' = a sine sweep), reach 1 is the pool
//! (q' = 0, `q_spatial = 0` so the storage law is exact, `p = 200`, `L = 5000 m`,
//! `h_ref` = bed so `S_active = S`). Run A is the full MC engine with option C
//! on row 1. Run B is a `StageBlock` on closure([1]) driven hour by hour from
//! run A's own MC hydrograph as `q_next_mc` (reach 0 never reads reach 1, so
//! this equals the engine hook).
//!
//! `theta_control = 0.5` and the fixed 1/2 MC inflow weight are required: MC
//! option C is the trapezoid rule (`.claude/RESERVOIRS.md` §Learned dam
//! release), and `substeps: 1` is required for the exact match. The dam row
//! has no friction edge, so `theta` must not matter.

mod common;

use std::sync::Arc;

use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::routing::{MuskingumCunge, RoutingInputs};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

use common::{mock_config, mock_spatial_parameters, InnerBackend, TestBackend, TestDevice};

const HOURS: usize = 72;
const T_DAYS: f32 = 1.23;
const N_MAN: f32 = 0.035;
const P_POOL: f32 = 200.0;

fn adjacency() -> SparseAdjacency {
    // 0 -> 1, L = 5000 m, slope 0.001.
    SparseAdjacency::from_dense(2, &[0.0, 0.0, 1.0, 0.0], vec![5000.0; 2], vec![0.001; 2])
}

/// `[HOURS + 1, 2]` row-major: col 0 a sine sweep, col 1 (the pool) zero.
fn q_prime() -> Vec<f32> {
    let mut v = vec![0.0f32; (HOURS + 1) * 2];
    for t in 0..=HOURS {
        let ph = t as f32 / HOURS as f32 * 4.0 * std::f32::consts::PI;
        v[t * 2] = 5.0 + 2.0 * ph.sin();
    }
    v
}

fn tv(v: &[f32], d: &TestDevice) -> Tensor<TestBackend, 1> {
    Tensor::<TestBackend, 1>::from_floats(v, d)
}

/// Run A: `[2][HOURS + 1]` from the MC engine with option C on row 1.
fn run_a(adj: &SparseAdjacency, device: &TestDevice) -> Vec<Vec<f32>> {
    let t = HOURS + 1;
    let sf = tv(&q_prime(), device).reshape([t, 2]);
    let mut mc = MuskingumCunge::<InnerBackend>::new(mock_config(), device.clone());
    mc.setup_inputs(
        RoutingInputs {
            adjacency: adj.clone(),
            x_storage: Tensor::<TestBackend, 1>::ones([2], device) * 0.2,
        },
        sf,
        mock_spatial_parameters(2, device),
        false,
        None,
    );
    mc.set_reservoir_rows(&[1], &[T_DAYS]).expect("valid reservoir rows");
    let flat: Vec<f32> = mc.forward().into_data().to_vec().unwrap();
    (0..2).map(|i| flat[i * t..(i + 1) * t].to_vec()).collect()
}

struct RunB {
    /// `[2][HOURS + 1]`, column 0 is the initial state.
    q: Vec<Vec<f32>>,
    rel_mass: f32,
}

fn run_b(adj: &SparseAdjacency, mc: &[Vec<f32>], theta: f32, theta_c: f32, device: &TestDevice) -> RunB {
    let pattern = Arc::new(CsrPattern::from_sparse(adj));
    let net = StageNetwork::build(adj, &pattern, Some(&[1]), 0.0, &[(1, T_DAYS)]).unwrap();
    let sp = StageParams {
        theta,
        theta_control: theta_c,
        substeps: 1,
        boundary: DownstreamBoundary::NormalDepth,
        depth_lb: 0.01,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    };
    let mut block = StageBlock::<InnerBackend>::new(net, sp, device);
    let nn = tv(&[N_MAN; 2], device);
    let pp = tv(&[P_POOL; 2], device);
    let qq = tv(&[0.0; 2], device);
    let slope = tv(&[0.001; 2], device);
    let mut q_t: Vec<f32> = (0..2).map(|i| mc[i][0]).collect();
    block.hotstart(tv(&q_t, device), nn.clone(), pp.clone(), qq.clone(), slope);
    // The engine clamps q' to `attribute_minimums.discharge` before routing
    // (mmc.rs `q_prime_clamped`), so the pool's zero q' reaches MC as 0.001.
    // Feed the block the same clamped values, as the engine hook must.
    let lb = mock_config().params.attribute_minimums.discharge;
    let qp: Vec<f32> = q_prime().into_iter().map(|v| v.max(lb)).collect();
    let mut q = vec![vec![q_t[0]], vec![q_t[1]]];
    for t in 1..=HOURS {
        let q_next_mc: Vec<f32> = (0..2).map(|i| mc[i][t]).collect();
        let qn = block.step(
            &pattern,
            nn.clone(),
            pp.clone(),
            qq.clone(),
            tv(&q_t, device),
            tv(&q_next_mc, device),
            tv(&qp[(t - 1) * 2..t * 2], device),
        );
        q_t = qn.into_data().to_vec().unwrap();
        q[0].push(q_t[0]);
        q[1].push(q_t[1]);
    }
    RunB { q, rel_mass: block.mass_balance().rel() }
}

fn max_rel(a: &[f32], b: &[f32]) -> f64 {
    let scale = a[1..].iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
    (1..=HOURS).fold(0.0f64, |m, t| m.max((a[t] - b[t]).abs() as f64 / scale))
}

#[test]
fn single_pool_node_reproduces_option_c() {
    let device = TestDevice::default();
    let adj = adjacency();
    let a = run_a(&adj, &device);
    let b1 = run_b(&adj, &a, 1.0, 0.5, &device);
    let b2 = run_b(&adj, &a, 0.55, 0.5, &device);
    let (r1, r2) = (max_rel(&a[1], &b1.q[1]), max_rel(&a[1], &b2.q[1]));
    println!("option C vs stage pool: max rel {r1:.3e} (theta 1.0), {r2:.3e} (theta 0.55)");
    assert!(r1 < 1e-5, "theta 1.0: {r1:.3e}");
    assert!(r2 < 1e-5, "theta 0.55: {r2:.3e}");
    for t in 0..=HOURS {
        assert_eq!(b1.q[1][t].to_bits(), b2.q[1][t].to_bits(), "theta changed the dam row at t {t}");
        assert_eq!(b1.q[0][t].to_bits(), a[0][t].to_bits(), "reach 0 differs at t {t}");
    }
}

#[test]
fn pool_node_at_theta_control_one_differs_but_conserves() {
    let device = TestDevice::default();
    let adj = adjacency();
    let a = run_a(&adj, &device);
    let b = run_b(&adj, &a, 1.0, 1.0, &device);
    let r = max_rel(&a[1], &b.q[1]);
    println!("theta_control 1.0: max rel vs option C {r:.3e}, mass rel {:.3e}", b.rel_mass);
    assert!(r > 1e-3, "theta_control 1.0 should differ from option C, got {r:.3e}");
    assert!(b.rel_mass <= 1e-4, "mass balance {}", b.rel_mass);
}
