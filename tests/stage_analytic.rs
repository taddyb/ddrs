//! Analytic checks of the stage step (Task 8): the steady M1 backwater profile
//! against the standard-step method, and the closed-end standing tidal wave
//! against the linear solution. Head geometry with physical `p`, `q` (C4).
//! The references run in f64 inside the test.

mod common;

use std::sync::Arc;

use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

use common::{InnerBackend, TestBackend, TestDevice};

const G: f64 = 9.81;

fn tv(v: &[f32], device: &TestDevice) -> Tensor<TestBackend, 1> {
    Tensor::<TestBackend, 1>::from_floats(v, device)
}

fn to_vec1<B: burn::tensor::backend::Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

fn chain(n: usize, len: f32, slope: f32) -> (SparseAdjacency, Arc<CsrPattern>) {
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(n, &dense, vec![len; n], vec![slope; n]);
    let pat = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, pat)
}

fn stage_params(theta: f32, substeps: u32) -> StageParams {
    StageParams {
        theta,
        theta_control: 0.5,
        substeps,
        boundary: DownstreamBoundary::PrescribedStage,
        depth_lb: 0.01,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    }
}

/// Head geometry section of `trapezoid_from_depth` for `p`, `q = 0`:
/// returns `(A, R)` at depth `d` (f64).
fn section(p: f64, d: f64) -> (f64, f64) {
    let qe = 1e-6;
    let tw = p * d.powf(qe);
    let ss = (tw * qe / (2.0 * d)).clamp(0.5, 50.0);
    let bw = (tw - 2.0 * ss * d).max(0.1);
    let a = (tw + bw) * d / 2.0;
    let pw = bw + 2.0 * d * (1.0 + ss * ss).sqrt();
    (a, a / pw)
}

fn conveyance(p: f64, n: f64, d: f64) -> f64 {
    let (a, r) = section(p, d);
    a * r.powf(2.0 / 3.0) / n
}

fn bisect(f: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(lo) * f(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

#[test]
fn steady_m1_profile_matches_standard_step() {
    let device = TestDevice::default();
    let (n, len, slope, nman, p, qin) = (40usize, 1000.0f32, 2e-4f32, 0.03f64, 50.0f64, 100.0f64);
    let (adj, pattern) = chain(n, len, slope);
    let net = StageNetwork::build(&adj, &pattern, None, 0.0, &[]).unwrap();
    let z_mid = net.z_mid.clone();
    let z_out_outlet = net.z_out[net.outlet] as f64;

    let d_n = bisect(|d| conveyance(p, nman, d) * (slope as f64).sqrt() - qin, 0.02, 30.0);
    let h_bc = z_out_outlet + 1.5 * d_n;

    // Standard step: dh/dx = -Q^2 / K(d)^2, integrated upstream from x = 40 km (RK4, 100 m).
    let x_end = n as f64 * len as f64;
    let bed = |x: f64| z_out_outlet + (slope as f64) * (x_end - x);
    let sf = |x: f64, h: f64| {
        let d = (h - bed(x)).max(0.01);
        (qin / conveyance(p, nman, d)).powi(2)
    };
    let dx = len as f64 / 10.0;
    let mut x = x_end;
    let mut h = h_bc;
    let mut depth_ref = vec![0.0f64; n];
    for step in 0..(n * 10) {
        // going upstream: dh/d(-x) = +Sf
        let k1 = sf(x, h);
        let k2 = sf(x - dx / 2.0, h + dx / 2.0 * k1);
        let k3 = sf(x - dx / 2.0, h + dx / 2.0 * k2);
        let k4 = sf(x - dx, h + dx * k3);
        h += dx / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        x -= dx;
        // after the 5th sub-step of each reach (counted from the outlet) x is a reach midpoint
        if step % 10 == 4 {
            let reach = n - 1 - step / 10;
            depth_ref[reach] = h - bed(x);
        }
    }

    let hours = 2000usize;
    let mut block = StageBlock::<InnerBackend>::new(net, stage_params(0.55, 1), &device);
    block.set_boundary_series(vec![h_bc as f32; hours + 1]).unwrap();
    let nn = tv(&vec![nman as f32; n], &device);
    let pp = tv(&vec![p as f32; n], &device);
    let qq = tv(&vec![0.0f32; n], &device);
    let slope_f = vec![slope.max(1e-3); n];
    let q0 = vec![qin as f32; n];
    block.hotstart(tv(&q0, &device), nn.clone(), pp.clone(), qq.clone(), tv(&slope_f, &device));
    let mut qp = vec![0.0f32; n];
    qp[0] = qin as f32;
    let zeros = vec![0.0f32; n];
    let mut q_t = q0.clone();
    let mut h_prev = to_vec1(block.stage().unwrap());
    let mut converged_at = None;
    for t in 1..=hours {
        let qn = block.step(&pattern, nn.clone(), pp.clone(), qq.clone(), tv(&q_t, &device), tv(&zeros, &device), tv(&qp, &device));
        q_t = to_vec1(qn);
        let h_now = to_vec1(block.stage().unwrap());
        let dmax = h_now.iter().zip(&h_prev).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        h_prev = h_now;
        if dmax < 1e-5 {
            converged_at = Some(t);
            break;
        }
    }
    let t_conv = converged_at.expect("did not reach max |dh| < 1e-5 m in 2000 hours");

    let mut max_rel = 0.0f64;
    let mut max_q = 0.0f64;
    for i in 0..n {
        let d_model = (h_prev[i] - z_mid[i]) as f64;
        let rel = (d_model - depth_ref[i]).abs() / depth_ref[i];
        max_rel = max_rel.max(rel);
        max_q = max_q.max((q_t[i] as f64 - qin).abs());
        assert!(rel <= 0.02, "reach {i}: depth {d_model} vs ref {} (rel {rel})", depth_ref[i]);
    }
    println!(
        "M1: normal depth {d_n:.3} m, outlet depth ref {:.3} m, converged hour {t_conv}, max depth rel err {:.3e}, max |Q-100| {max_q:.3e}",
        depth_ref[n - 1], max_rel
    );
    assert!(max_q <= 0.1, "max |Q - 100| = {max_q}");
}

struct Wave {
    /// Stage amplitude / phase (rad) at reach 0 and reach n/2.
    amp: [f64; 2],
    phase: [f64; 2],
    /// Fitted velocity amplitude at mid-channel (m/s).
    u_mid: f64,
}

const N_W: usize = 30;
const L_W: f64 = 5000.0;
const OMEGA: f64 = 2.0 * std::f64::consts::PI / (12.42 * 3600.0);
const A_TIDE: f64 = 0.7;
const N_MAN_W: f64 = 0.010;

fn solve3(mut m: [[f64; 4]; 3]) -> [f64; 3] {
    for c in 0..3 {
        let piv = (c..3).max_by(|&a, &b| m[a][c].abs().partial_cmp(&m[b][c].abs()).unwrap()).unwrap();
        m.swap(c, piv);
        for r in 0..3 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..4 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]]
}

/// Least squares `y ~ s sin(wt) + c cos(wt) + k`. Returns (amplitude, phase = atan2(c, s)).
fn fit(ts: &[f64], y: &[f64]) -> (f64, f64) {
    let mut m = [[0.0f64; 4]; 3];
    for (t, &v) in ts.iter().zip(y) {
        let b = [(OMEGA * t).sin(), (OMEGA * t).cos(), 1.0];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += b[i] * b[j];
            }
            m[i][3] += b[i] * v;
        }
    }
    let s = solve3(m);
    (s[0].hypot(s[1]), s[1].atan2(s[0]))
}

fn run_wave(substeps: u32, device: &TestDevice) -> Wave {
    let n = N_W;
    let (adj, pattern) = chain(n, L_W as f32, 0.0);
    let net = StageNetwork::build(&adj, &pattern, None, -10.0, &[]).unwrap();
    let z_mid = net.z_mid.clone();
    let hours = (16.0f64 * 12.42).round() as usize;
    let fit_hours = (4.0f64 * 12.42).round() as usize;
    let series: Vec<f32> = (0..=hours).map(|k| (A_TIDE * (OMEGA * k as f64 * 3600.0).sin()) as f32).collect();
    let mut block = StageBlock::<InnerBackend>::new(net, stage_params(0.55, substeps), device);
    block.set_boundary_series(series).unwrap();
    block.set_stage(tv(&vec![0.0f32; n], device));
    let nn = tv(&vec![N_MAN_W as f32; n], device);
    let pp = tv(&vec![1000.0f32; n], device);
    let qq = tv(&vec![0.0f32; n], device);
    let zeros = vec![0.0f32; n];
    let mut q_t = zeros.clone();
    let mid = n / 2;
    let (mut ts, mut h0, mut hm, mut qm) = (vec![], vec![], vec![], vec![]);
    for k in 0..hours {
        let qn = block.step(&pattern, nn.clone(), pp.clone(), qq.clone(), tv(&q_t, device), tv(&zeros, device), tv(&zeros, device));
        q_t = to_vec1(qn);
        let h = to_vec1(block.stage().unwrap());
        assert!(h.iter().all(|v| v.is_finite()), "non-finite stage at hour {k}");
        if k >= hours - fit_hours {
            ts.push((k + 1) as f64 * 3600.0);
            h0.push(h[0] as f64);
            hm.push(h[mid] as f64);
            qm.push(q_t[mid] as f64);
        }
    }
    let _ = z_mid;
    let (a0, p0) = fit(&ts, &h0);
    let (am, pm) = fit(&ts, &hm);
    let (aq, _) = fit(&ts, &qm);
    Wave { amp: [a0, am], phase: [p0, pm], u_mid: aq / (1000.0 * 10.0) }
}

/// `a cos(k x) / cos(k L)` with `k^2 = (w^2 - i w r) / (g d)`; returns (|H|, arg H).
fn analytic(x: f64, r: f64, d: f64) -> (f64, f64) {
    let k2 = ((OMEGA * OMEGA) / (G * d), -OMEGA * r / (G * d));
    let m = k2.0.hypot(k2.1).sqrt();
    let ang = k2.1.atan2(k2.0) / 2.0;
    let k = (m * ang.cos(), m * ang.sin());
    let cosc = |z: f64| {
        let (a, b) = (k.0 * z, k.1 * z);
        (a.cos() * b.cosh(), -a.sin() * b.sinh())
    };
    let (n, dd) = (cosc(x), cosc(N_W as f64 * L_W));
    let den = dd.0 * dd.0 + dd.1 * dd.1;
    let re = (n.0 * dd.0 + n.1 * dd.1) / den;
    let im = (n.1 * dd.0 - n.0 * dd.1) / den;
    (A_TIDE * re.hypot(im), im.atan2(re))
}

fn wrap(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

#[test]
fn standing_wave_in_a_closed_channel_matches_the_linear_solution() {
    let device = TestDevice::default();
    let w = run_wave(4, &device);
    let r = G * N_MAN_W * N_MAN_W * w.u_mid / 10.0f64.powf(4.0 / 3.0);
    let mid = N_W / 2;
    for (slot, reach, name) in [(0usize, 0usize, "closed end"), (1, mid, "mid-channel")] {
        let x = (reach as f64 + 0.5) * L_W;
        let (a_ref, p_ref) = analytic(x, r, 10.0);
        let amp_err = 100.0 * (w.amp[slot] / a_ref - 1.0);
        let ph_err = wrap(w.phase[slot] - p_ref).to_degrees();
        println!(
            "standing wave {name}: amp {:.4} m (analytic {a_ref:.4}), error {amp_err:+.2} %, phase error {ph_err:+.2} deg, u_ref {:.3} m/s, r {r:.2e} 1/s",
            w.amp[slot], w.u_mid
        );
        assert!(amp_err.abs() <= 10.0, "{name}: amplitude error {amp_err} %");
        assert!(ph_err.abs() <= 10.0, "{name}: phase error {ph_err} deg");
    }
}

#[test]
fn hourly_substep_error_is_as_measured() {
    let device = TestDevice::default();
    let ref24 = run_wave(24, &device).amp[0];
    let e1 = 100.0 * (run_wave(1, &device).amp[0] / ref24 - 1.0);
    let e4 = 100.0 * (run_wave(4, &device).amp[0] / ref24 - 1.0);
    println!("closed-end amplitude vs substeps 24: substeps 1 {e1:+.2} %, substeps 4 {e4:+.2} %");
    assert!((e1 - (-7.7)).abs() <= 2.0, "substeps 1: {e1} % (expected -7.7)");
    assert!((e4 - (-1.3)).abs() <= 2.0, "substeps 4: {e4} % (expected -1.3)");
}
