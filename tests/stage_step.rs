//! The stage block step: assembly, tree solve, readout (design: Physics,
//! Time weights per edge, Coupling and state). Tests drive `StageBlock::step`
//! from a precomputed Muskingum-Cunge hydrograph: MC rows never read stage
//! rows (the stage set is downstream-closed), so this equals the engine.

mod common;

use std::sync::Arc;

use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::data::store::ChannelGeometryRows;
use ddrs::data::ConusAdjacencyStore;
use ddrs::routing::mmc_op::timestep_forward;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::routing::{MuskingumCunge, RoutingInputs};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

use common::{
    linear_chain_sparse, mock_config, mock_spatial_parameters, mock_streamflow, InnerBackend,
    TestBackend, TestDevice,
};

const N_MAN: f32 = 0.035;
const P_SP: f32 = 21.0;
const Q_SP: f32 = 0.5;
const DEPTH_LB: f32 = 0.01;

fn params(theta: f32, theta_control: f32, substeps: u32) -> StageParams {
    StageParams {
        theta,
        theta_control,
        substeps,
        boundary: DownstreamBoundary::NormalDepth,
        depth_lb: DEPTH_LB,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    }
}

fn tv(v: &[f32], device: &TestDevice) -> Tensor<TestBackend, 1> {
    Tensor::<TestBackend, 1>::from_floats(v, device)
}

fn to_vec1<B: burn::tensor::backend::Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

/// Cumulative upstream sum of `q'` on a tree (rows >= cols).
fn cumulative(adj: &SparseAdjacency, qp: &[f32]) -> Vec<f32> {
    let mut q = qp.to_vec();
    let mut down = vec![usize::MAX; adj.n];
    for (&r, &c) in adj.rows.iter().zip(&adj.cols) {
        down[c as usize] = r as usize;
    }
    for i in 0..adj.n {
        if down[i] != usize::MAX {
            let v = q[i];
            q[down[i]] += v;
        }
    }
    q
}

/// MC hydrograph `[n][T]` (no stage block) for a streamflow `[T, n]`.
fn mc_hydrograph(adj: &SparseAdjacency, sf: &Tensor<TestBackend, 2>, device: &TestDevice) -> Vec<Vec<f32>> {
    let n = adj.n;
    let t = sf.dims()[0];
    let mut mc = MuskingumCunge::<InnerBackend>::new(mock_config(), device.clone());
    mc.setup_inputs(
        RoutingInputs {
            adjacency: adj.clone(),
            x_storage: Tensor::<TestBackend, 1>::ones([n], device) * 0.2,
        },
        sf.clone(),
        mock_spatial_parameters(n, device),
        false,
        None,
    );
    let flat: Vec<f32> = mc.forward().into_data().to_vec().unwrap();
    (0..n).map(|i| flat[i * t..(i + 1) * t].to_vec()).collect()
}

struct Run {
    /// Per hour `q_next` (len n).
    q: Vec<Vec<f32>>,
    /// Stage before the first hour, then after each hour.
    h: Vec<Vec<f32>>,
    /// Per hour `D_s` of the last substep.
    d_s: Vec<Vec<f32>>,
    block: StageBlock<InnerBackend>,
}

/// Run `hours` hours. `q_prime` is `[T][n]` row-major time-major; hour `t`
/// (1-based) uses `q_prime[t-1]` and `mc[.][t]`, as `MuskingumCunge::forward`.
#[allow(clippy::too_many_arguments)]
fn drive(
    adj: &SparseAdjacency,
    pattern: &Arc<CsrPattern>,
    net: StageNetwork,
    sp: StageParams,
    mc: &[Vec<f32>],
    q_prime: &[f32],
    q_init: &[f32],
    hours: usize,
    device: &TestDevice,
) -> Run {
    drive_with(adj, pattern, net, sp, mc, q_prime, q_init, hours, device, &|_| {})
}

/// `drive`, with `setup` run on the new block before the hotstart.
#[allow(clippy::too_many_arguments)]
fn drive_with(
    adj: &SparseAdjacency,
    pattern: &Arc<CsrPattern>,
    net: StageNetwork,
    sp: StageParams,
    mc: &[Vec<f32>],
    q_prime: &[f32],
    q_init: &[f32],
    hours: usize,
    device: &TestDevice,
    setup: &dyn Fn(&mut StageBlock<InnerBackend>),
) -> Run {
    let n = adj.n;
    let mut block = StageBlock::<InnerBackend>::new(net, sp, device);
    setup(&mut block);
    let nn = tv(&vec![N_MAN; n], device);
    let pp = tv(&vec![P_SP; n], device);
    let qq = tv(&vec![Q_SP; n], device);
    let slope_f: Vec<f32> = adj.slope.iter().map(|s| s.max(1e-3)).collect();
    block.hotstart(tv(q_init, device), nn.clone(), pp.clone(), qq.clone(), tv(&slope_f, device));
    let mut q_t = q_init.to_vec();
    let mut out = Run { q: vec![], h: vec![to_vec1(block.stage().unwrap())], d_s: vec![], block: block };
    for t in 1..=hours {
        let q_next_mc: Vec<f32> = (0..n).map(|i| mc[i].get(t).copied().unwrap_or(0.0)).collect();
        let qp = &q_prime[(t - 1) * n..t * n];
        let qn = out.block.step(
            pattern,
            nn.clone(),
            pp.clone(),
            qq.clone(),
            tv(&q_t, device),
            tv(&q_next_mc, device),
            tv(qp, device),
        );
        q_t = to_vec1(qn);
        out.q.push(q_t.clone());
        out.h.push(to_vec1(out.block.stage().unwrap()));
        out.d_s.push(to_vec1(out.block.last_coefficients().unwrap().d_s));
    }
    out
}

fn flat(sf: &Tensor<TestBackend, 2>) -> Vec<f32> {
    sf.clone().into_data().to_vec().unwrap()
}

#[test]
fn stage_rows_replace_mc_rows_only() {
    let device = TestDevice::default();
    let n = 5;
    let t = 25;
    let adj = linear_chain_sparse(n);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let sf = mock_streamflow(t, n, &device);
    let mc = mc_hydrograph(&adj, &sf, &device);
    let net = StageNetwork::build(&adj, &pattern, Some(&[3]), 0.0, &[]).unwrap();
    let q_init: Vec<f32> = (0..n).map(|i| mc[i][0]).collect();
    let run = drive(&adj, &pattern, net, params(1.0, 0.5, 1), &mc, &flat(&sf), &q_init, t - 1, &device);
    for (h, q) in run.q.iter().enumerate() {
        for i in 0..3 {
            assert_eq!(q[i].to_bits(), mc[i][h + 1].to_bits(), "hour {} row {i}", h + 1);
        }
        assert!(q[3].is_finite() && q[4].is_finite());
    }
}

#[test]
fn step_conserves_linearised_storage() {
    let device = TestDevice::default();
    let n = 5;
    let t = 25;
    for substeps in [1u32, 4] {
        let adj = linear_chain_sparse(n);
        let pattern = Arc::new(CsrPattern::from_sparse(&adj));
        let sf = mock_streamflow(t, n, &device);
        let mc = mc_hydrograph(&adj, &sf, &device);
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let q_init: Vec<f32> = (0..n).map(|i| mc[i][0]).collect();
        let run = drive(&adj, &pattern, net, params(0.7, 0.5, substeps), &mc, &flat(&sf), &q_init, 24, &device);
        let mb = run.block.mass_balance();
        assert_eq!(mb.steps, 24);
        assert!(mb.gross_flux_m3 > 0.0);
        assert!(mb.rel() <= 1e-4, "substeps {substeps}: rel {} ({mb:?})", mb.rel());
    }
}

fn juniata() -> (SparseAdjacency, Arc<CsrPattern>) {
    let store = ConusAdjacencyStore::open("examples/juniata/data/juniata_conus_adjacency.zarr")
        .expect("open bundle conus adjacency");
    let adj = SparseAdjacency {
        n: store.n,
        rows: store.indices_0.clone(),
        cols: store.indices_1.clone(),
        values: vec![1.0; store.nnz],
        length_m: store.length_m.to_vec(),
        slope: store.slope.to_vec(),
        parent_offset: None,
    };
    let pat = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, pat)
}

#[test]
fn step_is_finite_on_juniata_network() {
    let device = TestDevice::default();
    let (adj, pattern) = juniata();
    let n = adj.n;
    let hours = 48;
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = net.z_mid.clone();
    let mc = vec![vec![0.0f32; hours + 1]; n];
    let q_prime = vec![1.0f32; hours * n];
    let q_init = cumulative(&adj, &vec![1.0; n]);
    let run = drive(&adj, &pattern, net, params(1.0, 0.5, 1), &mc, &q_prime, &q_init, hours, &device);
    for (h, q) in run.q.iter().enumerate() {
        assert!(q.iter().all(|v| v.is_finite()), "q hour {h}");
    }
    let hl = run.h.last().unwrap();
    assert!(hl.iter().all(|v| v.is_finite()));
    let wet = (0..n).filter(|&i| hl[i] - z_mid[i] > DEPTH_LB + 1e-4).count();
    assert!(wet >= 200, "only {wet} reaches deeper than depth_lb");
    // No growth: the start state is the steady state of q' = 1, so |Q| must stay
    // within 1.5x the steady outlet discharge (reference: max abs Q 235, no growth).
    let steady_outlet = q_init.iter().cloned().fold(0.0f32, f32::max);
    let max_abs = run.q.iter().flat_map(|q| q.iter()).fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(max_abs <= 1.5 * steady_outlet, "max |Q| {max_abs} vs steady outlet {steady_outlet}");
    let mb = run.block.mass_balance();
    assert!(mb.rel() <= 1e-4, "{mb:?}");
}

#[test]
fn mc_interface_conserves_volume_for_any_theta() {
    let device = TestDevice::default();
    let n = 3;
    let hours = 72;
    let t = hours + 1;
    for theta in [0.55f32, 1.0] {
        for substeps in [1u32, 4] {
            let adj = linear_chain_sparse(n);
            let pattern = Arc::new(CsrPattern::from_sparse(&adj));
            let sf = mock_streamflow(t, n, &device);
            let mc = mc_hydrograph(&adj, &sf, &device);
            let net = StageNetwork::build(&adj, &pattern, Some(&[1]), 0.0, &[]).unwrap();
            let q_init: Vec<f32> = (0..n).map(|i| mc[i][0]).collect();
            let run = drive(&adj, &pattern, net, params(theta, 0.5, substeps), &mc, &flat(&sf), &q_init, hours, &device);
            let want: f64 = (0..hours)
                .map(|k| 3600.0 * (mc[0][k] as f64 + mc[0][k + 1] as f64) / 2.0)
                .sum();
            let mb = run.block.mass_balance();
            let rel = ((mb.mc_inflow_m3 as f64) - want).abs() / want;
            assert!(rel <= 1e-5, "theta {theta} substeps {substeps}: mc_inflow {} want {want} rel {rel}", mb.mc_inflow_m3);
            assert!(mb.rel() <= 1e-4, "theta {theta} substeps {substeps}: {mb:?}");
        }
    }
}

#[test]
fn control_edges_count_their_old_flux_once() {
    let device = TestDevice::default();
    let n = 3;
    let hours = 48;
    let t = hours + 1;
    let (theta, theta_c) = (0.55f32, 0.5f32);
    let adj = linear_chain_sparse(n);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let sf = mock_streamflow(t, n, &device);
    let qp = flat(&sf);
    let mc = vec![vec![0.0f32; t + 1]; n];
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[(0, 1.23)]).unwrap();
    // Start the chain in balance with the dam: Q_dam = q'_0 and the rest cumulative.
    let q_init = cumulative(&adj, &qp[0..n]);
    let run = drive(&adj, &pattern, net, params(theta, theta_c, 1), &mc, &qp, &q_init, hours, &device);
    let mut q_prev = q_init.clone();
    for k in 0..hours {
        let q_new = &run.q[k];
        let (h0, h1) = (&run.h[k], &run.h[k + 1]);
        let d_s = &run.d_s[k];
        let qp_k = &qp[k * n..(k + 1) * n];
        let bar = |a: &[f32]| 1e-5 * a.iter().map(|v| v.abs()).sum::<f32>();
        // Row 1: dam edge in (weight theta_c), friction edge out (weight theta).
        let lhs1 = d_s[1] * (h1[1] - h0[1]);
        let rhs1 = theta_c * q_new[0] + (1.0 - theta_c) * q_prev[0]
            - theta * q_new[1]
            - (1.0 - theta) * q_prev[1]
            + qp_k[1];
        let b1 = bar(&[q_prev[0], q_new[0], q_prev[1], q_new[1], qp_k[1]]);
        assert!((lhs1 - rhs1).abs() <= b1, "hour {k} row 1: {lhs1} vs {rhs1} (bar {b1}) h0 {h0:?} h1 {h1:?} q {q_new:?} prev {q_prev:?}");
        // Row 2: friction edge in (theta), normal-depth outlet control out (theta).
        let lhs2 = d_s[2] * (h1[2] - h0[2]);
        let rhs2 = theta * q_new[1] + (1.0 - theta) * q_prev[1]
            - theta * q_new[2]
            - (1.0 - theta) * q_prev[2]
            + qp_k[2];
        let b2 = bar(&[q_prev[1], q_new[1], q_prev[2], q_new[2], qp_k[2]]);
        assert!((lhs2 - rhs2).abs() <= b2, "hour {k} row 2: {lhs2} vs {rhs2} (bar {b2}) h0 {h0:?} h1 {h1:?} q {q_new:?} prev {q_prev:?} ds {d_s:?} mb {:?}", run.block.mass_balance());
        q_prev = q_new.clone();
    }
    let mb = run.block.mass_balance();
    assert!(mb.rel() <= 1e-4, "{mb:?}");
    assert_eq!(mb.floor_created_m3, 0.0, "{mb:?}");
}

#[test]
fn substeps_four_approaches_substeps_one_at_steady_state() {
    let device = TestDevice::default();
    let n = 5;
    let hours = 240;
    let adj = linear_chain_sparse(n);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let qp = vec![1.0f32; hours * n];
    let mc = vec![vec![0.0f32; hours + 1]; n];
    let steady = cumulative(&adj, &vec![1.0; n]);
    let q_init: Vec<f32> = steady.iter().map(|v| 0.9 * v).collect();
    let mut last = vec![];
    for substeps in [1u32, 4] {
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let run = drive(&adj, &pattern, net, params(1.0, 0.5, substeps), &mc, &qp, &q_init, hours, &device);
        last.push((run.q.last().unwrap().clone(), run.h.last().unwrap().clone()));
    }
    for i in 0..n {
        let (a, b) = (last[0].0[i], last[1].0[i]);
        assert!((a - b).abs() <= 1e-4 * a.abs().max(b.abs()), "Q row {i}: {a} vs {b}");
        let (ha, hb) = (last[0].1[i], last[1].1[i]);
        assert!((ha - hb).abs() <= 1e-4 * ha.abs().max(hb.abs()), "h row {i}: {ha} vs {hb}");
        assert!((a - steady[i]).abs() <= 1e-3 * steady[i], "row {i} not at steady Q: {a} vs {}", steady[i]);
    }
}

#[test]
fn gradcheck_through_one_step() {
    let device = TestDevice::default();
    let n = 5;
    let adj = linear_chain_sparse(n);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let net0 = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = net0.z_mid.clone();
    let n0: Vec<f32> = vec![0.03, 0.035, 0.04, 0.045, 0.05];
    let q0: Vec<f32> = vec![0.45, 0.5, 0.55, 0.5, 0.6];
    let p0: Vec<f32> = vec![18.0, 21.0, 24.0, 20.0, 22.0];
    let h0: Vec<f32> = (0..n).map(|i| z_mid[i] + 0.4 + 0.07 * (i as f32) + 0.03 * ((i * i) as f32 % 3.0)).collect();
    let qt0: Vec<f32> = vec![1.1, 2.3, 3.2, 4.6, 5.1];
    let qprime = vec![1.0f32; n];
    let zeros = vec![0.0f32; n];

    for substeps in [1u32, 2] {
        // loss(n, q, p, h, qt) -> (loss, grads of each input)
        let eval = |x: [&Vec<f32>; 5], with_grad: bool| -> (f32, Vec<Vec<f32>>) {
            let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
            let mut block = StageBlock::<InnerBackend>::new(net, params(0.8, 0.5, substeps), &device);
            let ts: Vec<Tensor<TestBackend, 1>> = x.iter().map(|v| tv(v, &device).require_grad()).collect();
            block.set_stage(ts[3].clone());
            let out = block.step(
                &pattern,
                ts[0].clone(),
                ts[2].clone(),
                ts[1].clone(),
                ts[4].clone(),
                tv(&zeros, &device),
                tv(&qprime, &device),
            );
            let loss = out.sum();
            let l = loss.clone().into_scalar();
            if !with_grad {
                return (l, vec![]);
            }
            let g = loss.backward();
            let grads = ts.iter().map(|t| to_vec1(t.grad(&g).expect("grad"))).collect();
            (l, grads)
        };
        let base = [&n0, &q0, &p0, &h0, &qt0];
        let (_, grads) = eval(base, true);
        let eps_abs = [1e-3f32, 1e-3, 1e-3, 1e-2, 1e-2];
        let names = ["n", "q", "p", "h_t", "q_t"];
        for (pi, name) in names.iter().enumerate() {
            for i in 0..n {
                let x = base[pi][i];
                let eps = if pi < 3 { (1e-3 * x.abs()).max(1e-3) } else { eps_abs[pi] };
                let cd = |e: f32| -> f32 {
                    let mut up = base[pi].clone();
                    let mut dn = base[pi].clone();
                    up[i] = x + e;
                    dn[i] = x - e;
                    let mut bu = base;
                    bu[pi] = &up;
                    let mut bd = base;
                    bd[pi] = &dn;
                    (eval(bu, false).0 - eval(bd, false).0) / (2.0 * e)
                };
                let fd = cd(eps);
                let an = grads[pi][i];
                let err = (fd - an).abs();
                assert!(
                    err <= 1e-4 || err <= 5e-3 * fd.abs().max(an.abs()),
                    "substeps {substeps} d loss/d {name}[{i}]: fd {fd} analytic {an}"
                );
            }
        }
    }
}

#[test]
fn newton_coefficients_match_the_implicit_edge_law() {
    let device = TestDevice::default();
    let n = 3;
    let adj = linear_chain_sparse(n);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = net.z_mid.clone();
    let theta = 0.55f32;
    let mut block = StageBlock::<InnerBackend>::new(net, params(theta, 0.5, 1), &device);
    let h: Vec<f32> = [0.6f32, 1.8, 0.8].iter().zip(&z_mid).map(|(d, z)| z + d).collect();
    block.set_stage(tv(&h, &device));
    let q_t = [-3.0f32, 4.0, 5.0];
    let _ = block.step(
        &pattern,
        tv(&[N_MAN; 3], &device),
        tv(&[P_SP; 3], &device),
        tv(&[Q_SP; 3], &device),
        tv(&q_t, &device),
        tv(&[0.0; 3], &device),
        tv(&[1.0; 3], &device),
    );
    let c = block.last_coefficients().unwrap();
    let (alpha, beta, kappa, a_up, b_down, dt_phi) = (
        to_vec1(c.alpha),
        to_vec1(c.beta),
        to_vec1(c.kappa),
        to_vec1(c.a_up),
        to_vec1(c.b_down),
        to_vec1(c.dt_phi),
    );
    // Test-side f64 copy of src/geometry.rs::trapezoid_from_depth (area, wetted perimeter).
    let section = |d: f64| -> (f64, f64) {
        let qe = Q_SP as f64 + 1e-6;
        let tw = P_SP as f64 * d.powf(qe);
        let ss = (tw * qe / (2.0 * d)).clamp(0.5, 50.0);
        let bw = (tw - 2.0 * ss * d).max(0.1);
        ((tw + bw) * d / 2.0, bw + 2.0 * d * (ss * ss + 1.0).sqrt())
    };
    let (g, dt, l_edge, nm) = (9.81f64, 3600.0f64, 1000.0f64, N_MAN as f64);
    let f = |qn: f64, qt: f64, de: f64, dh: f64| -> f64 {
        let (a, p) = section(de);
        let phi_conv = a * (a / p).powf(4.0 / 3.0) / (nm * nm);
        qn - qt + dt * (g * qn * qn.abs() / phi_conv - g * a * dh / l_edge)
    };
    let eps = 1e-6;
    // (edge, d_e, dh^t, s_up, s_dn)
    let edges = [(0usize, 0.8f64, (h[0] - h[1]) as f64, 0.0f64, 1.0f64), (1, 1.8, (h[1] - h[2]) as f64, 1.0, 0.0)];
    for (e, de, dh, s_up, s_dn) in edges {
        let qt = q_t[e] as f64;
        let dq = (f(qt + eps, qt, de, dh) - f(qt - eps, qt, de, dh)) / (2.0 * eps);
        let ddh = (f(qt, qt, de, dh + eps) - f(qt, qt, de, dh - eps)) / (2.0 * eps);
        let dde = (f(qt, qt, de + eps, dh) - f(qt, qt, de - eps, dh)) / (2.0 * eps);
        let (b_ref, k_ref) = (-ddh / dq, -dde / dq);
        let rel = |x: f32, y: f64| ((x as f64 - y) / y.abs().max(1e-9)).abs();
        assert!(rel(beta[e], b_ref) <= 1e-4, "edge {e} beta {} vs {b_ref}", beta[e]);
        assert!(rel(kappa[e], k_ref) <= 1e-4, "edge {e} kappa {} vs {k_ref}", kappa[e]);
        let a_alpha = ((1.0 + dt_phi[e]) / (1.0 + 2.0 * dt_phi[e])) as f64;
        assert!(rel(alpha[e], a_alpha) <= 1e-6, "edge {e} alpha {} vs {a_alpha}", alpha[e]);
        let a_ref = (theta as f64 * beta[e] as f64 + kappa[e] as f64 * s_up).max(0.0);
        let b_ref2 = (theta as f64 * beta[e] as f64 - kappa[e] as f64 * s_dn).max(0.0);
        assert!(rel(a_up[e], a_ref) <= 1e-5, "edge {e} a {} vs {a_ref}", a_up[e]);
        assert!(rel(b_down[e], b_ref2) <= 1e-5, "edge {e} b {} vs {b_ref2}", b_down[e]);
    }
    // f64 reference values from the plan (coef_state.py), looser: they pin the whole chain.
    let want = [(0.5107f32, 8.477f32, -8.361f32, 4.662f32, 13.023f32), (0.5692, 185.80, 312.36, 414.55, 102.19)];
    for (e, w) in want.iter().enumerate() {
        for (got, exp, name) in [
            (alpha[e], w.0, "alpha"),
            (beta[e], w.1, "beta"),
            (kappa[e], w.2, "kappa"),
            (a_up[e], w.3, "a"),
            (b_down[e], w.4, "b"),
        ] {
            assert!(((got - exp) / exp).abs() <= 2e-3, "edge {e} {name}: {got} vs {exp}");
        }
    }
}

#[test]
fn perturbation_decays_on_short_reaches() {
    let device = TestDevice::default();
    let n = 5;
    let hours = 48;
    for theta in [1.0f32, 0.55] {
        let adj = linear_chain_sparse(n);
        let pattern = Arc::new(CsrPattern::from_sparse(&adj));
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let steady = cumulative(&adj, &vec![1.0; n]);
        let q_init: Vec<f32> = steady.iter().map(|v| 0.9 * v).collect();
        let mc = vec![vec![0.0f32; hours + 1]; n];
        let qp = vec![1.0f32; hours * n];
        let run = drive(&adj, &pattern, net, params(theta, 0.5, 1), &mc, &qp, &q_init, hours, &device);
        let smax = steady.iter().cloned().fold(0.0f32, f32::max);
        // e[t] is the error after t hours (e[0] is the start state).
        let e: Vec<f32> = std::iter::once(&q_init)
            .chain(run.q.iter())
            .map(|q| q.iter().zip(&steady).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / smax)
            .collect();
        assert!(e[12] <= 1e-4, "theta {theta}: e(12) = {}", e[12]);
        assert!(e[24] <= 1e-5, "theta {theta}: e(24) = {}", e[24]);
        for t in 2..hours {
            assert!(e[t + 1] <= e[t] + 1e-5, "theta {theta}: e grows at hour {t}: {} -> {}", e[t], e[t + 1]);
        }
    }
}

// =============================================================================
// Task 7: edge cases of the step
// =============================================================================

/// Chain of `len.len()` reaches with per-reach length and slope.
fn chain(len: &[f32], slope: &[f32]) -> (SparseAdjacency, Arc<CsrPattern>) {
    let n = len.len();
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(n, &dense, len.to_vec(), slope.to_vec());
    let pat = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, pat)
}

fn prescribed(theta: f32, substeps: u32) -> StageParams {
    StageParams { boundary: DownstreamBoundary::PrescribedStage, ..params(theta, 0.5, substeps) }
}

/// One step with uniform physical parameters. Returns `q_next`.
#[allow(clippy::too_many_arguments)]
fn one_step(
    block: &mut StageBlock<InnerBackend>,
    pattern: &Arc<CsrPattern>,
    n_man: f32,
    q_t: &[f32],
    q_next_mc: &[f32],
    qp: &[f32],
    device: &TestDevice,
) -> Vec<f32> {
    let n = q_t.len();
    to_vec1(block.step(
        pattern,
        tv(&vec![n_man; n], device),
        tv(&vec![P_SP; n], device),
        tv(&vec![Q_SP; n], device),
        tv(q_t, device),
        tv(q_next_mc, device),
        tv(qp, device),
    ))
}

#[test]
fn stage_depth_floor_is_counted() {
    let device = TestDevice::default();
    let n = 3;
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = net.z_mid.clone();
    let mut block = StageBlock::<InnerBackend>::new(net, prescribed(1.0, 1), &device);
    // The outlet stage sits 5 m below every bed.
    block.set_boundary_series(vec![-5.0; 7]).unwrap();
    let h0: Vec<f32> = z_mid.iter().map(|z| z + 0.05).collect();
    block.set_stage(tv(&h0, &device));
    let mut q_t = vec![0.0f32; n];
    for _ in 0..6 {
        q_t = one_step(&mut block, &pattern, N_MAN, &q_t, &vec![0.0; n], &vec![0.0; n], &device);
    }
    let h = to_vec1(block.stage().unwrap());
    for i in 0..n {
        assert!(h[i] - z_mid[i] >= DEPTH_LB - 1e-6, "row {i} below the floor: {}", h[i] - z_mid[i]);
    }
    let mb = block.mass_balance();
    assert!(mb.floor_created_m3 > 0.0, "{mb:?}");
    let resid = (mb.storage_change_m3 - mb.net_flux_m3 - mb.floor_created_m3).abs();
    assert!(resid <= 1e-4 * mb.gross_flux_m3, "resid {resid} {mb:?}");
}

#[test]
fn beta_is_finite_at_zero_flow() {
    let device = TestDevice::default();
    let n = 3;
    let theta = 0.7f32;
    let (adj, pattern) = chain(&[1000.0; 3], &[0.0; 3]);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = net.z_mid.clone();
    let edge_len = net.edge_length.clone();
    let mut block = StageBlock::<InnerBackend>::new(net, prescribed(theta, 1), &device);
    let flat_h = z_mid[0] + 1.0;
    assert!(z_mid.iter().all(|&z| z == z_mid[0]), "slope 0 means a flat bed");
    block.set_boundary_series(vec![flat_h; 2]).unwrap();
    block.set_stage(tv(&vec![flat_h; n], &device));
    let q = one_step(&mut block, &pattern, N_MAN, &[0.0; 3], &[0.0; 3], &[0.0; 3], &device);
    let c = block.last_coefficients().unwrap();
    let (alpha, beta, kappa, a_up, b_down, area) = (
        to_vec1(c.alpha),
        to_vec1(c.beta),
        to_vec1(c.kappa),
        to_vec1(c.a_up),
        to_vec1(c.b_down),
        to_vec1(c.area_edge),
    );
    for i in 0..n {
        let want = 9.81 * area[i] * 3600.0 / edge_len[i];
        assert!(beta[i].is_finite() && ((beta[i] - want) / want).abs() <= 1e-6, "beta[{i}] {} vs {want}", beta[i]);
        assert_eq!(alpha[i], 1.0, "alpha[{i}]");
        assert_eq!(kappa[i], 0.0, "kappa[{i}]");
        assert_eq!(a_up[i], theta * beta[i], "a[{i}]");
        assert_eq!(b_down[i], theta * beta[i], "b[{i}]");
        assert_eq!(q[i], 0.0, "Q_next[{i}]");
    }
    let h = to_vec1(block.stage().unwrap());
    for i in 0..n {
        assert!((h[i] - flat_h).abs() <= 1e-6, "stage[{i}] moved: {}", h[i]);
    }
}

/// Test-side f64 trapezoid section (area, wetted perimeter) at depth `d`.
fn section64(d: f64) -> (f64, f64) {
    let qe = Q_SP as f64 + 1e-6;
    let tw = P_SP as f64 * d.powf(qe);
    let ss = (tw * qe / (2.0 * d)).clamp(0.5, 50.0);
    let bw = (tw - 2.0 * ss * d).max(0.1);
    ((tw + bw) * d / 2.0, bw + 2.0 * d * (ss * ss + 1.0).sqrt())
}

/// Conveyance `K = A R^(2/3) / n` at depth `d` (f64).
fn conveyance64(d: f64, n_man: f64) -> f64 {
    let (a, p) = section64(d);
    a * (a / p).powf(2.0 / 3.0) / n_man
}

#[test]
fn friction_limit_recovers_the_diffusive_conveyance() {
    // In this limit one substep is one Newton step on the diffusive law
    // Q = K(d_e) sqrt(dh / L_edge), which is the limit case of the local-inertial law.
    let device = TestDevice::default();
    let n = 5;
    let (n_man, slope, len) = (0.1f32, 5e-3f32, 1000.0f32);
    let (adj, pattern) = chain(&[len; 5], &[slope; 5]);
    let k = conveyance64(0.5, n_man as f64);
    let q_eq = (k * (slope as f64).sqrt()) as f32;
    assert!((q_eq - 2.0414).abs() < 1e-3, "Q_eq {q_eq}");
    // dK/dd, central difference in f64.
    let e = 1e-6;
    let dk = (conveyance64(0.5 + e, n_man as f64) - conveyance64(0.5 - e, n_man as f64)) / (2.0 * e);
    for scale in [1.0f32, 1.2] {
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let z_mid = net.z_mid.clone();
        let mut block = StageBlock::<InnerBackend>::new(net, params(1.0, 0.5, 1), &device);
        let h: Vec<f32> = z_mid.iter().map(|z| z + 0.5).collect();
        block.set_stage(tv(&h, &device));
        let mut qp = vec![0.0f32; n];
        qp[0] = q_eq;
        let q = one_step(&mut block, &pattern, n_man, &vec![q_eq * scale; n], &vec![0.0; n], &qp, &device);
        let c = block.last_coefficients().unwrap();
        let (alpha, beta, kappa) = (to_vec1(c.alpha), to_vec1(c.beta), to_vec1(c.kappa));
        for i in 1..4 {
            if scale == 1.0 {
                let want_beta = k / (2.0 * len as f64 * slope.sqrt() as f64 * 1.0);
                let want_beta = want_beta * 1.0;
                assert!(((beta[i] as f64 - want_beta) / want_beta).abs() <= 0.02, "beta[{i}] {} vs {want_beta}", beta[i]);
                assert!((alpha[i] - 0.5).abs() <= 0.01, "alpha[{i}] {}", alpha[i]);
                let want_kappa = (slope as f64).sqrt() * dk;
                assert!(((kappa[i] as f64 - want_kappa) / want_kappa).abs() <= 0.02, "kappa[{i}] {} vs {want_kappa}", kappa[i]);
                assert!(((q[i] - q_eq) / q_eq).abs() <= 1e-4, "Q_next[{i}] {} vs {q_eq}", q[i]);
            } else {
                assert!((q[i] / q_eq - 1.0).abs() <= 0.03, "scale {scale}: Q_next[{i}]/Q_eq {}", q[i] / q_eq);
            }
        }
    }
}

#[test]
fn mc_rows_have_finite_gradients_under_stage_backflow() {
    // C3: the MC op reads stage rows of q_t through mc_view (clamp_min(discharge_lb).detach()).
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let cfg = mock_config();
    let lb = cfg.params.attribute_minimums.discharge;
    let assembler = AValuesAssembler::<InnerBackend>::new(&pattern, &device);
    let mut finite_with_raw = None;
    for use_view in [true, false] {
        let net = StageNetwork::build(&adj, &pattern, Some(&[1]), 0.0, &[]).unwrap();
        let z_mid = net.z_mid.clone();
        let mut block = StageBlock::<InnerBackend>::new(net, prescribed(1.0, 1), &device);
        block.set_boundary_series(vec![0.0, 0.0]).unwrap();
        let h: Vec<f32> = z_mid.iter().map(|z| z + 0.5).collect();
        block.set_stage(tv(&h, &device));
        let nn = tv(&[0.035; 3], &device).require_grad();
        let qq = tv(&[0.4; 3], &device).require_grad();
        let pp = tv(&[20.0; 3], &device).require_grad();
        let q_t = tv(&[5.0, -5.0, -2.0], &device).require_grad();
        let qp = tv(&[1.0; 3], &device);
        let q_t_mc = if use_view { block.mc_view(q_t.clone(), lb) } else { q_t.clone() };
        if use_view {
            let v = to_vec1(q_t_mc.clone().inner().into());
            assert_eq!(v[0], 5.0);
            assert_eq!(v[1], lb);
            assert_eq!(v[2], lb);
        }
        let q_next_mc = timestep_forward::<InnerBackend>(
            &cfg,
            &pattern,
            &assembler,
            nn.clone(),
            qq.clone(),
            pp.clone(),
            q_t_mc,
            qp.clone(),
            tv(&adj.length_m, &device),
            tv(&adj.slope, &device),
            tv(&[0.3; 3], &device),
            false,
            None,
        );
        let q_next = block.step(&pattern, nn.clone(), pp.clone(), qq.clone(), q_t.clone(), q_next_mc, qp);
        let g = q_next.sum().backward();
        let all_finite = [&nn, &qq, &pp, &q_t].iter().all(|t| match t.grad(&g) {
            Some(v) => to_vec1(v).iter().all(|x| x.is_finite()),
            None => false,
        });
        if use_view {
            assert!(all_finite, "a gradient is not finite with mc_view");
        } else {
            finite_with_raw = Some(all_finite);
        }
    }
    println!("raw q_t into the MC op: all gradients finite = {finite_with_raw:?}");
}

#[test]
fn flow_reversal_is_not_clamped() {
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 2], &[1e-3; 2]);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = net.z_mid.clone();
    let mut block = StageBlock::<InnerBackend>::new(net, prescribed(1.0, 1), &device);
    let h: Vec<f32> = z_mid.iter().map(|z| z + 0.5).collect();
    // The outlet stage starts at the outlet reach's stage and rises 1 m in one hour.
    block.set_boundary_series(vec![h[1], h[1] + 1.0]).unwrap();
    block.set_stage(tv(&h, &device));
    let q = one_step(&mut block, &pattern, N_MAN, &[0.0; 2], &[0.0; 2], &[0.0; 2], &device);
    assert!(q[1].is_finite() && q[1] < 0.0, "outlet Q {}", q[1]);
    assert!(q[1] < -DEPTH_LB, "outlet Q {} is clamped near zero", q[1]);
}

/// Central-difference check of `sum(q_next)` through one step, every input.
/// Steps: n 1e-3, q 1e-2, p 0.2, h_t 1e-2, q_t 1e-2.
#[allow(clippy::too_many_arguments)]
fn gradcheck_state(
    label: &str,
    adj: &SparseAdjacency,
    pattern: &Arc<CsrPattern>,
    dams: &[(usize, f32)],
    depth0: &[f32],
    qt0: &[f32],
    params: StageParams,
    device: &TestDevice,
) {
    let n = adj.n;
    let net0 = StageNetwork::build(adj, pattern, None, 0.0, dams).unwrap();
    let h0: Vec<f32> = (0..n).map(|i| net0.z_mid[i] + depth0[i]).collect();
    let n0: Vec<f32> = (0..n).map(|i| 0.03 + 0.005 * i as f32).collect();
    let q0: Vec<f32> = (0..n).map(|i| 0.45 + 0.05 * (i % 3) as f32).collect();
    let p0: Vec<f32> = (0..n).map(|i| 18.0 + 2.0 * (i % 4) as f32).collect();
    let qt0 = qt0.to_vec();
    let qprime = vec![1.0f32; n];
    let zeros = vec![0.0f32; n];
    let eval = |x: [&Vec<f32>; 5], with_grad: bool| -> (f32, Vec<Vec<f32>>) {
        let net = StageNetwork::build(adj, pattern, None, 0.0, dams).unwrap();
        let mut block = StageBlock::<InnerBackend>::new(net, params, device);
        let ts: Vec<Tensor<TestBackend, 1>> = x.iter().map(|v| tv(v, device).require_grad()).collect();
        block.set_stage(ts[3].clone());
        let out = block.step(
            pattern,
            ts[0].clone(),
            ts[2].clone(),
            ts[1].clone(),
            ts[4].clone(),
            tv(&zeros, device),
            tv(&qprime, device),
        );
        let loss = out.sum();
        let l = loss.clone().into_scalar();
        if !with_grad {
            return (l, vec![]);
        }
        let g = loss.backward();
        (l, ts.iter().map(|t| to_vec1(t.grad(&g).expect("grad"))).collect())
    };
    let base = [&n0, &q0, &p0, &h0, &qt0];
    let (l0, grads) = eval(base, true);
    println!("{label}: loss {l0}");
    let names = ["n", "q", "p", "h_t", "q_t"];
    // f32 noise on the loss (about 1e-5) sets the lower bound; curvature the upper.
    const EPS: [f32; 5] = [1e-3, 1e-2, 0.2, 1e-2, 1e-2];
    for (pi, name) in names.iter().enumerate() {
        for i in 0..n {
            let x = base[pi][i];
            let eps = EPS[pi];
            let mut up = base[pi].clone();
            let mut dn = base[pi].clone();
            up[i] = x + eps;
            dn[i] = x - eps;
            let mut bu = base;
            bu[pi] = &up;
            let mut bd = base;
            bd[pi] = &dn;
            let fd = (eval(bu, false).0 - eval(bd, false).0) / (2.0 * eps);
            let an = grads[pi][i];
            let err = (fd - an).abs();
            assert!(
                an.is_finite() && (err <= 1e-4 || err <= 5e-3 * fd.abs().max(an.abs())),
                "{label}: d loss/d {name}[{i}]: fd {fd} analytic {an}"
            );
        }
    }
}

#[test]
fn gradcheck_through_a_reversed_edge_and_negative_flow() {
    // Edge 0 is reversed (h_1 > h_0 by 0.2 m, s_dn = 1: the b = max(theta*beta - kappa*s_dn, 0)
    // branch) and row 1 carries a negative q_t (negative kappa). Away from every kink.
    // A gentle bed (2e-4) keeps the surface slopes small, so the f32 central
    // differences stay below the 5e-3 bar.
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 5], &[2e-4; 5]);
    let depth0 = [0.3f32, 0.7, 0.6, 0.5, 0.5];
    let qt0 = [0.8f32, -1.2, 1.5, 1.8, 2.0];
    // Pin which branches the state takes (theta 0.8, one substep).
    {
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let h: Vec<f32> = net.z_mid.iter().zip(&depth0).map(|(z, d)| z + d).collect();
        assert!(h[1] > h[0], "edge 0 must be reversed");
        let mut block = StageBlock::<InnerBackend>::new(net, params(0.8, 0.5, 1), &device);
        block.set_stage(tv(&h, &device));
        let _ = one_step(&mut block, &pattern, N_MAN, &qt0, &[0.0; 5], &[1.0; 5], &device);
        let c = block.last_coefficients().unwrap();
        let (beta, kappa, b_down) = (to_vec1(c.beta), to_vec1(c.kappa), to_vec1(c.b_down));
        println!("reversed state: beta {beta:?} kappa {kappa:?} b {b_down:?}");
        assert!(0.8 * beta[0] - kappa[0] > 1e-3 || b_down[0] == 0.0, "b[0] sits on the kink");
        assert!(kappa[1] != 0.0);
    }
    for substeps in [1u32, 2] {
        gradcheck_state(
            &format!("reversed edge, substeps {substeps}"),
            &adj,
            &pattern,
            &[],
            &depth0,
            &qt0,
            params(0.8, 0.5, substeps),
            &device,
        );
    }
}

#[test]
fn gradcheck_through_a_dam_control_edge() {
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 5], &[1e-3; 5]);
    let depth0 = [0.45f32, 0.55, 0.8, 0.6, 0.7];
    let qt0 = [1.1f32, 2.3, 3.2, 4.6, 5.1];
    for substeps in [1u32, 2] {
        gradcheck_state(
            &format!("dam at row 2, substeps {substeps}"),
            &adj,
            &pattern,
            &[(2, 1.23)],
            &depth0,
            &qt0,
            params(0.8, 0.5, substeps),
            &device,
        );
    }
}

#[test]
fn normal_depth_outlet_gradient_is_finite() {
    // The outlet R' path: dL/dn at the outlet row of a 2-chain, REL 5e-3.
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 2], &[1e-3; 2]);
    let net0 = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let h0: Vec<f32> = (0..2).map(|i| net0.z_mid[i] + 0.6 + 0.2 * i as f32).collect();
    let eval = |n_out: f32, with_grad: bool| -> (f32, f32) {
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let mut block = StageBlock::<InnerBackend>::new(net, params(1.0, 0.5, 1), &device);
        block.set_stage(tv(&h0, &device));
        let nn = tv(&[0.035, n_out], &device).require_grad();
        let out = block.step(
            &pattern,
            nn.clone(),
            tv(&[P_SP; 2], &device),
            tv(&[Q_SP; 2], &device),
            tv(&[2.0, 3.0], &device),
            tv(&[0.0; 2], &device),
            tv(&[1.0; 2], &device),
        );
        let loss = out.sum();
        let l = loss.clone().into_scalar();
        if !with_grad {
            return (l, 0.0);
        }
        let g = loss.backward();
        (l, to_vec1(nn.grad(&g).expect("grad"))[1])
    };
    let n0 = 0.04f32;
    let (_, an) = eval(n0, true);
    let eps = 1e-3;
    let fd = (eval(n0 + eps, false).0 - eval(n0 - eps, false).0) / (2.0 * eps);
    assert!(an.is_finite() && an != 0.0, "analytic {an}");
    assert!(((fd - an) / fd.abs().max(an.abs())).abs() <= 5e-3, "fd {fd} analytic {an}");
}

/// Hour-to-hour changes of `x`, the longest run of sign alternations (both
/// changes above `thresh`), and the largest |change| from index `skip` on.
fn ring_stats(x: &[f32], thresh: f32, skip: usize) -> (usize, f32) {
    let d: Vec<f32> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let (mut run, mut best) = (0usize, 0usize);
    for k in 1..d.len() {
        let alt = d[k] * d[k - 1] < 0.0 && d[k].abs() > thresh && d[k - 1].abs() > thresh;
        run = if alt { run + 1 } else { 0 };
        best = best.max(run);
    }
    let tail = d.iter().skip(skip).fold(0.0f32, |m, v| m.max(v.abs()));
    (best, tail)
}

#[test]
fn stiff_short_reach_does_not_ring() {
    let device = TestDevice::default();
    let n = 5;
    let (pre, post) = (24usize, 48usize);
    let hours = pre + post;
    let (adj, pattern) = chain(&[5000.0, 5000.0, 93.0, 5000.0, 5000.0], &[3e-3, 3e-3, 0.0, 3e-3, 3e-3]);
    let step = 2.5f32;
    let mut qp = vec![5.0f32; hours * n];
    for t in pre..hours {
        qp[t * n + 2] += step;
    }
    let steady = cumulative(&adj, &vec![5.0; n]);
    let mc = vec![vec![0.0f32; hours + 1]; n];
    let mut stats = vec![];
    for theta in [1.0f32, 0.5] {
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        let run = drive(&adj, &pattern, net, params(theta, 0.5, 1), &mc, &qp, &steady, hours, &device);
        // Q at reach 2 from the hour before the step on.
        let series: Vec<f32> = run.q[pre - 1..].iter().map(|q| q[2]).collect();
        assert!(series.iter().all(|v| v.is_finite()));
        let (alt, tail) = ring_stats(&series, 1e-3 * step, 1 + 6);
        println!("stiff reach 2, theta {theta}: longest alternation {alt} h, max |dQ| after hour 6 = {tail} ({:.2e} x step)", tail / step);
        stats.push((theta, alt, tail));
    }
    let (_, alt1, tail1) = stats[0];
    assert!(alt1 <= 2, "theta 1.0 rings: run {alt1} h");
    assert!(tail1 <= 1e-2 * step, "theta 1.0 tail {tail1}");
    let (_, alt5, tail5) = stats[1];
    assert!(alt5 >= 10, "theta 0.5 should ring (run {alt5} h): the test cannot fail otherwise");
    assert!(tail5 > 1e-2 * step, "theta 0.5 tail {tail5}");
}

#[test]
fn adverse_flow_keeps_the_m_matrix() {
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = net.z_mid.clone();
    let theta = 1.0f32;
    let mut block = StageBlock::<InnerBackend>::new(net, params(theta, 0.5, 1), &device);
    let h: Vec<f32> = z_mid.iter().zip([0.3f32, 1.5, 1.0]).map(|(z, d)| z + d).collect();
    block.set_stage(tv(&h, &device));
    let q = one_step(&mut block, &pattern, N_MAN, &[10.0, 5.0, 6.0], &[0.0; 3], &[1.0; 3], &device);
    let c = block.last_coefficients().unwrap();
    let (beta, kappa, a_up, b_down) = (to_vec1(c.beta), to_vec1(c.kappa), to_vec1(c.a_up), to_vec1(c.b_down));
    assert!(theta * beta[0] - kappa[0] < 0.0, "clamp not exercised: {} - {}", theta * beta[0], kappa[0]);
    assert_eq!(b_down[0], 0.0);
    assert!(a_up.iter().all(|&v| v >= 0.0) && b_down.iter().all(|&v| v >= 0.0));
    assert!(q.iter().all(|v| v.is_finite()), "{q:?}");
    let mb = block.mass_balance();
    assert!(mb.rel() <= 1e-4, "{mb:?}");
    println!("adverse flow: Q_next {q:?}, floor_created_m3 {}", mb.floor_created_m3);
}

// =============================================================================
// Task 10b: table geometry (channel_geometry)
// =============================================================================

fn table(rows: &[usize], tw: f32, bed: &[f32], n: f32, pool: &[Option<f32>]) -> ChannelGeometryRows {
    ChannelGeometryRows {
        rows: rows.to_vec(),
        top_width: vec![tw; rows.len()],
        bed_z: bed.to_vec(),
        n: vec![n; rows.len()],
        pool_z: pool.to_vec(),
    }
}

#[test]
fn table_geometry_must_cover_the_stage_set_in_the_block() {
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let mut block = StageBlock::<InnerBackend>::new(net, prescribed(1.0, 1), &device);
    let rows = table(&[0, 2], 100.0, &[0.0, 0.0], 0.03, &[None, None]);
    let err = block.set_table_geometry(rows.clone(), true).unwrap_err();
    assert!(err.contains("row 1"), "{err}");
    block.set_table_geometry(rows, false).unwrap();
}

#[test]
fn mixed_geometry_uses_the_head_where_the_table_has_no_row() {
    let device = TestDevice::default();
    let n = 3;
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let mk = || StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = mk().z_mid.clone();
    let h0: Vec<f32> = z_mid.iter().map(|z| z + 1.0).collect();
    let q_t = [3.0f32, 5.0, 7.0];
    let qp = [0.5f32, 0.4, 0.3];
    let run = |tab: Option<ChannelGeometryRows>| {
        let mut block = StageBlock::<InnerBackend>::new(mk(), prescribed(1.0, 1), &device);
        block.set_boundary_series(vec![-5.0; 2]).unwrap();
        if let Some(t) = tab {
            block.set_table_geometry(t, false).unwrap();
        }
        block.set_stage(tv(&h0, &device));
        let q = one_step(&mut block, &pattern, N_MAN, &q_t, &[0.0; 3], &qp, &device);
        (q, block.last_coefficients().unwrap())
    };
    let (qa, ca) = run(None);
    // Row 2 table: width 80 m, fixed n 0.02, bed at the head's z_mid so row 1's
    // downstream bed is unchanged.
    let (qb, cb) = run(Some(table(&[2], 80.0, &[z_mid[2]], 0.02, &[None])));
    let bits = |t: Tensor<InnerBackend, 1>| to_vec1(t).iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for (name, a, b) in [
        ("beta", bits(ca.beta), bits(cb.beta)),
        ("kappa", bits(ca.kappa), bits(cb.kappa)),
        ("alpha", bits(ca.alpha), bits(cb.alpha)),
        ("area_edge", bits(ca.area_edge), bits(cb.area_edge)),
        ("radius_edge", bits(ca.radius_edge), bits(cb.radius_edge)),
        ("d_s", bits(ca.d_s), bits(cb.d_s)),
    ] {
        assert_eq!(a[0], b[0], "{name} row 0");
        assert_eq!(a[1], b[1], "{name} row 1");
        assert_ne!(a[2], b[2], "{name} row 2 should use the rectangle");
    }
    // Discharge is not compared: the tree solve couples every row to the table row.
    assert!(qa.iter().chain(qb.iter()).all(|v| v.is_finite()));
}

#[test]
fn table_rows_use_the_rectangle_and_fixed_n() {
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = net.z_mid.clone();
    let edge_len = net.edge_length.clone();
    let (tw, bed, n_tab) = (80.0f64, 0.3f32, 0.02f64);
    let mut block = StageBlock::<InnerBackend>::new(net, prescribed(1.0, 1), &device);
    block.set_boundary_series(vec![-5.0; 2]).unwrap();
    block
        .set_table_geometry(table(&[2], tw as f32, &[bed], n_tab as f32, &[None]), false)
        .unwrap();
    let mut h0: Vec<f32> = z_mid.iter().map(|z| z + 1.0).collect();
    h0[2] = 1.5;
    block.set_stage(tv(&h0, &device));
    let (q_t2, dt, g) = (7.0f64, 3600.0f64, 9.81f64);
    let nn = |v: f32| tv(&[v; 3], &device);
    // Differentiable inputs, to check the head's gradient on the table row.
    let (nh, qh, ph) = (nn(0.035).require_grad(), nn(0.5).require_grad(), nn(21.0).require_grad());
    let out = block.step(
        &pattern,
        nh.clone(),
        ph.clone(),
        qh.clone(),
        tv(&[3.0, 5.0, q_t2 as f32], &device),
        tv(&[0.0; 3], &device),
        tv(&[0.5, 0.4, 0.3], &device),
    );
    let c = block.last_coefficients().unwrap();
    let (area, rad, d_s, kappa) = (
        to_vec1(c.area_edge)[2] as f64,
        to_vec1(c.radius_edge)[2] as f64,
        to_vec1(c.d_s)[2] as f64,
        to_vec1(c.kappa)[2] as f64,
    );
    // Outlet edge: d_edge = max(h, h_bc) - max(bed, z_down) = 1.5 - max(0.3, 0).
    let d = 1.5f64 - bed as f64;
    let a = tw * d;
    let p = tw + 2.0 * d;
    let rel = |x: f64, y: f64| ((x - y) / y).abs();
    assert!(rel(area, a) <= 1e-6, "area {area} vs {a}");
    assert!(rel(rad, a / p) <= 1e-6, "radius {rad} vs {}", a / p);
    assert!(rel(d_s, tw * 1000.0 / dt) <= 1e-6, "d_s {d_s}");
    // kappa, section derivatives A_d = TW, P_d = 2 (revision 4).
    let phi_cap = a * (a / p).powf(4.0 / 3.0);
    let phi = g * n_tab * n_tab * q_t2.abs() / phi_cap;
    let phid_over = tw / a + (4.0 / 3.0) * (tw / a - 2.0 / p);
    let dh = 1.5f64 - (-5.0);
    let want_k = (dt * phi * q_t2 * phid_over + g * tw * dt / edge_len[2] as f64 * dh) / (1.0 + 2.0 * dt * phi);
    assert!(rel(kappa, want_k) <= 1e-5, "kappa {kappa} vs {want_k}");
    // The head's n, p, q are not read on the table row: exactly zero gradient.
    let g_all = out.sum().backward();
    for (name, t) in [("n", &nh), ("q", &qh), ("p", &ph)] {
        let gv = to_vec1(t.grad(&g_all).expect("grad"));
        assert_eq!(gv[2], 0.0, "d loss/d {name}[2]");
    }
}

#[test]
fn pool_z_sets_the_release_reference() {
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let (tw, len, t_days, theta_c) = (400.0f64, 1000.0f64, 1.5f32, 1.0f32);
    let run = |stage_above_bed: f32| {
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[(0, t_days)]).unwrap();
        let mut sp = prescribed(1.0, 1);
        sp.theta_control = theta_c;
        let mut block = StageBlock::<InnerBackend>::new(net, sp, &device);
        block.set_boundary_series(vec![-5.0; 2]).unwrap();
        // Bed 10 m, pool 2 m above it, on every row so the others stay simple.
        block
            .set_table_geometry(table(&[0, 1, 2], tw as f32, &[10.0, 10.0, 10.0], 0.03, &[Some(12.0), None, None]), true)
            .unwrap();
        block.set_stage(tv(&[10.0 + stage_above_bed, 12.0, 12.0], &device));
        let q = one_step(&mut block, &pattern, N_MAN, &[0.0; 3], &[0.0; 3], &[0.0; 3], &device);
        let c = block.last_coefficients().unwrap();
        (q[0] as f64, to_vec1(c.r_prime)[0] as f64, to_vec1(c.d_s)[0] as f64)
    };
    let (q_lo, rp_lo, _) = run(1.0);
    assert_eq!(q_lo, 0.0, "R below the pool");
    assert_eq!(rp_lo, 0.0, "R' below the pool");
    let (q_hi, rp_hi, ds) = run(3.0);
    let t_s = t_days as f64 * 86_400.0;
    let want_rp = tw * len / t_s;
    assert!(((rp_hi - want_rp) / want_rp).abs() <= 1e-6, "R' {rp_hi} vs {want_rp}");
    // theta_c = 1 and no inflow: dh = -R/(D_s + R'), Q = R + R'dh = R*D_s/(D_s + R').
    let r = q_hi * (ds + rp_hi) / ds;
    let want_r = tw * len * 1.0 / t_s;
    assert!(((r - want_r) / want_r).abs() <= 1e-5, "R {r} vs {want_r}");
}

#[test]
fn table_rows_hotstart_from_the_boundary_or_normal_depth() {
    let device = TestDevice::default();
    let (adj, pattern) = chain(&[1000.0; 3], &[1e-3; 3]);
    let (tw, n_tab, bed) = (1000.0f64, 0.022f64, -10.0f32);
    let q0 = [100.0f32, 2000.0, 2000.0];
    let slope_f = [1e-3f32; 3];
    let all = |pool: [Option<f32>; 3]| table(&[0, 1, 2], tw as f32, &[bed; 3], n_tab as f32, &pool);
    let start = |dams: &[(usize, f32)], bc: DownstreamBoundary, pool: [Option<f32>; 3]| {
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, dams).unwrap();
        let sp = StageParams { boundary: bc, ..params(1.0, 0.5, 1) };
        let mut block = StageBlock::<InnerBackend>::new(net, sp, &device);
        if bc == DownstreamBoundary::PrescribedStage {
            block.set_boundary_series(vec![0.4; 3]).unwrap();
        }
        block.set_table_geometry(all(pool), true).unwrap();
        let v = |x: f32| tv(&[x; 3], &device);
        block.hotstart(tv(&q0, &device), v(0.035), v(21.0), v(0.5), tv(&slope_f, &device));
        to_vec1(block.stage().unwrap())
    };
    // PrescribedStage: the boundary at t0 on non-dam rows, exactly.
    let h = start(&[], DownstreamBoundary::PrescribedStage, [None; 3]);
    for i in 0..3 {
        assert_eq!(h[i], 0.4, "row {i}");
    }
    // NormalDepth: the wide-channel normal depth of the rectangle.
    let h = start(&[], DownstreamBoundary::NormalDepth, [None; 3]);
    for i in 1..3 {
        let dn = (n_tab * q0[i] as f64 / (tw * (1e-3f64).sqrt())).powf(0.6);
        let got = (h[i] - bed) as f64;
        assert!(((got - dn) / dn).abs() <= 1e-5, "row {i}: {got} vs {dn}");
    }
    // A dam row: pool_z + T*Q0/(TW*L).
    let t_days = 1.23f32;
    let h = start(&[(0, t_days)], DownstreamBoundary::NormalDepth, [Some(-2.0), None, None]);
    let want = -2.0 + t_days as f64 * 86_400.0 * 100.0 / (tw * 1000.0);
    assert!(((h[0] as f64 - want) / want).abs() <= 1e-6, "dam {} vs {want}", h[0]);
}

#[test]
fn geometry_head_is_bitwise_the_block_without_a_table() {
    let device = TestDevice::default();
    let n = 5;
    let hours = 24;
    let t = hours + 1;
    let adj = linear_chain_sparse(n);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let sf = mock_streamflow(t, n, &device);
    let mc = mc_hydrograph(&adj, &sf, &device);
    let q_init: Vec<f32> = (0..n).map(|i| mc[i][0]).collect();
    let go = |with_table: bool| {
        let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
        drive_with(&adj, &pattern, net, params(0.7, 0.5, 2), &mc, &flat(&sf), &q_init, hours, &device, &|b| {
            if with_table {
                b.set_table_geometry(ChannelGeometryRows::default(), false).unwrap();
            }
        })
    };
    let (a, b) = (go(false), go(true));
    for h in 0..hours {
        for i in 0..n {
            assert_eq!(a.q[h][i].to_bits(), b.q[h][i].to_bits(), "hour {h} row {i}");
        }
    }
}
