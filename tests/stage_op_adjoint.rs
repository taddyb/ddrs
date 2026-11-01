//! Adjoints of the elementwise blocks of the fused stage op (Task 16b).
//!
//! Each test builds the block's inputs as `require_grad()` leaves on
//! `Autodiff<NdArray<f32>>`, runs a Burn copy of the block (copied op for op
//! from `StageBlock::step_reference`; the comment names the block) or the
//! library fns, takes `L = sum(bar * outputs)` with seeded cotangents, and
//! compares Burn's input gradients with the host adjoint of
//! `src/routing/stage_op.rs`. The reference is Burn autodiff.
//!
//! Gradient bar: rel 1e-5 or abs 1e-6 * max(1, max|g_ref|). Each test prints
//! the worst error per block.

mod common;

use std::sync::Arc;

use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::geometry::{trapezoid_depth_derivatives, trapezoid_from_depth};
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::routing::stage_op::{
    control_backward_host, control_host, depth_backward_host, depth_host, newton_backward_host, newton_host,
    section_backward_host, section_host, ControlBar, DepthOut, NewtonBar, SectionBar, SectionHost,
    StageOpConsts,
};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

use common::{mock_config, InnerBackend, TestBackend, TestDevice};

type T = Tensor<TestBackend, 1>;

fn dev() -> TestDevice {
    Default::default()
}

fn cst(v: &[f32]) -> T {
    T::from_floats(v, &dev())
}

fn leaf(v: &[f32]) -> T {
    cst(v).require_grad()
}

fn host(t: Tensor<InnerBackend, 1>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

fn val(t: &T) -> Vec<f32> {
    host(t.clone().inner())
}

/// Deterministic cotangents in magnitude [0.5, 1.5], alternating sign when `signed`.
fn bar(seed: u32, len: usize, signed: bool) -> Vec<f32> {
    let mut s = 0x9E3779B9u32.wrapping_mul(seed.wrapping_add(1));
    (0..len)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let m = 0.5 + (s >> 8) as f32 / (1u32 << 24) as f32;
            if signed && (s >> 7) & 1 == 1 && i % 2 == 0 {
                -m
            } else {
                m
            }
        })
        .collect()
}

/// `sum(w * t)`.
fn wsum(w: &[f32], t: T) -> T {
    (t * cst(w)).sum()
}

fn grad(g: &burn::backend::autodiff::grads::Gradients, t: &T, n: usize) -> Vec<f32> {
    t.grad(g).map(host).unwrap_or_else(|| vec![0.0; n])
}

/// Gradient bar. Returns the worst `err / allowed` (<= 1 passes) and the worst
/// `err / max|ref|`.
fn check(what: &str, got: &[f32], refg: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), refg.len(), "{what}: length");
    let scale = refg.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let rmax = refg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (mut worst, mut inf_rel) = (0.0f32, 0.0f32);
    for (i, (a, r)) in got.iter().zip(refg).enumerate() {
        assert!(a.is_finite() && r.is_finite(), "{what}[{i}]: non-finite {a} vs {r}");
        let err = (a - r).abs();
        let allowed = (1e-5 * r.abs()).max(1e-6 * scale);
        assert!(err <= allowed, "{what}[{i}]: host {a} vs autodiff {r} (err {err:e}, allowed {allowed:e})");
        worst = worst.max(err / allowed);
        if rmax > 0.0 {
            inf_rel = inf_rel.max(err / rmax);
        }
    }
    (worst, inf_rel)
}

fn check_fwd(what: &str, got: &[f32], refv: &[f32]) {
    assert_eq!(got.len(), refv.len(), "{what}: length");
    let scale = refv.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (i, (a, r)) in got.iter().zip(refv).enumerate() {
        assert!((a - r).abs() <= 1e-6 * scale.max(f32::MIN_POSITIVE), "{what}[{i}] forward: {a} vs {r}");
    }
}

struct Report {
    name: String,
    worst: f32,
    inf_rel: f32,
}

impl Report {
    fn new(name: &str) -> Self {
        Report { name: name.to_string(), worst: 0.0, inf_rel: 0.0 }
    }
    fn add(&mut self, what: &str, got: &[f32], refg: &[f32]) {
        let (w, r) = check(&format!("{}: {what}", self.name), got, refg);
        self.worst = self.worst.max(w);
        self.inf_rel = self.inf_rel.max(r);
    }
    fn print(&self) {
        println!(
            "{}: worst err/allowed {:.3}, max err/max|ref| {:.3e}",
            self.name, self.worst, self.inf_rel
        );
    }
}

// ---------------------------------------------------------------- fixtures

fn chain(n: usize) -> (SparseAdjacency, Arc<CsrPattern>) {
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    let adj = SparseAdjacency::from_dense(n, &dense, vec![1000.0; n], vec![0.001; n]);
    let pat = Arc::new(CsrPattern::from_sparse(&adj));
    (adj, pat)
}

fn consts(n: usize, theta: f32, prescribed: bool) -> StageOpConsts {
    let (adj, pat) = chain(n);
    let net = StageNetwork::build(&adj, &pat, None, 0.0, &[]).unwrap();
    let params = StageParams {
        theta,
        theta_control: 0.5,
        substeps: 1,
        boundary: if prescribed { DownstreamBoundary::PrescribedStage } else { DownstreamBoundary::NormalDepth },
        depth_lb: 0.01,
        bottom_width_lb: mock_config().params.attribute_minimums.bottom_width,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    };
    let mut block = StageBlock::<InnerBackend>::new(net, params, &dev());
    (*block.op_consts(&pat)).clone()
}

fn section_cases() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pq = [(21.0f32, 0.5f32), (1000.0, 0.0), (200.0, 0.9), (5.0, 0.9)];
    let depths = [0.02f32, 0.3, 1.0, 5.0];
    let (mut p, mut q, mut d) = (vec![], vec![], vec![]);
    for (pp, qq) in pq {
        for dd in depths {
            p.push(pp);
            q.push(qq);
            d.push(dd);
        }
    }
    (p, q, d)
}

fn branches(s: &SectionHost, bwlb: f32) -> Vec<&'static str> {
    (0..s.ss_raw.len())
        .map(|i| {
            if s.ss_raw[i] < 0.5 {
                "side-slope floor"
            } else if s.ss_raw[i] > 50.0 {
                "side-slope cap"
            } else if s.bw_raw[i] < bwlb {
                "bottom-width floor"
            } else {
                "interior"
            }
        })
        .collect()
}

/// Section head plus the optional table mix, as `StageBlock::section`. Returns
/// the six outputs TW, A, P, R, A_d, P_d.
fn section_burn(c: &StageOpConsts, p: &T, q: &T, d: &T) -> [T; 6] {
    let bw_lb = c.bw_lb;
    let g = trapezoid_from_depth(p.clone(), q.clone(), d.clone(), bw_lb);
    let (a_d, p_d) = trapezoid_depth_derivatives(p.clone(), q.clone(), d.clone(), &g, bw_lb);
    if !c.has_table {
        return [g.top_width, g.area, g.wetted_perimeter, g.hydraulic_radius, a_d, p_d];
    }
    let mask = cst(&c.t_mask);
    let tw = cst(&c.t_tw);
    let keep = mask.clone().neg() + 1.0;
    let area_r = tw.clone() * d.clone();
    let perim_r = tw.clone() + d.clone() * 2.0;
    let radius_r = area_r.clone() / perim_r.clone();
    let mix = |head: T, tab: T| head * keep.clone() + tab * mask.clone();
    [
        mix(g.top_width, tw.clone()),
        mix(g.area, area_r),
        mix(g.wetted_perimeter, perim_r),
        mix(g.hydraulic_radius, radius_r),
        mix(a_d, tw.clone()),
        mix(p_d, T::ones_like(&tw) * 2.0),
    ]
}

fn section_run(label: &str, with_table: bool) {
    let (p, q, d) = section_cases();
    let n = p.len();
    let mut c = StageOpConsts { n, bw_lb: mock_config().params.attribute_minimums.bottom_width, ..Default::default() };
    if with_table {
        c.has_table = true;
        c.t_mask = (0..n).map(|i| (i % 2) as f32).collect();
        c.t_tw = vec![500.0; n];
    }
    let q_eps: Vec<f32> = q.iter().map(|v| v + 1e-6f32).collect();
    let s = section_host(&c, &p, &q_eps, &d);
    let br = branches(&s, c.bw_lb);
    for i in 0..n {
        println!("{label} point {i}: p {} q {} d {} -> {}", p[i], q[i], d[i], br[i]);
    }
    for want in ["interior", "side-slope floor", "side-slope cap", "bottom-width floor"] {
        assert!(br.iter().any(|b| *b == want), "{label}: branch '{want}' not exercised");
    }
    let (lp, lq, ld) = (leaf(&p), leaf(&q), leaf(&d));
    let out = section_burn(&c, &lp, &lq, &ld);
    let bars: Vec<Vec<f32>> = (0..6).map(|k| bar(10 + k as u32, n, true)).collect();
    let mut loss = wsum(&bars[0], out[0].clone());
    for k in 1..6 {
        loss = loss + wsum(&bars[k], out[k].clone());
    }
    let g = loss.backward();
    let (gp, gq, gd) = (grad(&g, &lp, n), grad(&g, &lq, n), grad(&g, &ld, n));
    // forward equality of the host block
    check_fwd("TW", &s.out_tw, &val(&out[0]));
    check_fwd("A", &s.out_area, &val(&out[1]));
    check_fwd("P", &s.out_perim, &val(&out[2]));
    check_fwd("R", &s.out_radius, &val(&out[3]));
    check_fwd("A_d", &s.out_a_d, &val(&out[4]));
    check_fwd("P_d", &s.out_p_d, &val(&out[5]));
    let sbar = SectionBar {
        top_width: bars[0].clone(),
        area: bars[1].clone(),
        perim: bars[2].clone(),
        radius: bars[3].clone(),
        a_d: bars[4].clone(),
        p_d: bars[5].clone(),
    };
    let got = section_backward_host(&c, &p, &q_eps, &d, &s, &sbar);
    let mut rep = Report::new(label);
    rep.add("p", &got.p, &gp);
    rep.add("q_eps", &got.q_eps, &gq);
    rep.add("d", &got.d, &gd);
    if with_table {
        for i in (1..n).step_by(2) {
            assert_eq!(got.p[i], 0.0, "table row {i}: p gradient must be exactly 0");
            assert_eq!(got.q_eps[i], 0.0, "table row {i}: q gradient must be exactly 0");
            assert_eq!(gp[i], 0.0);
        }
    }
    rep.print();
}

#[test]
fn section_adjoint_matches_burn_autodiff() {
    section_run("section", false);
}

#[test]
fn section_adjoint_with_a_table_matches_burn_autodiff() {
    section_run("section+table", true);
}

// ------------------------------------------------------------------- Newton

/// Block 3 (3a to 3f) of `StageBlock::step_reference`. Returns (Q*, a, b).
#[allow(clippy::too_many_arguments)]
fn newton_burn(
    c: &StageOpConsts,
    dt_s: f32,
    h: &T,
    hd: &T,
    q: &T,
    n2: &T,
    area: &T,
    perim: &T,
    radius: &T,
    a_d: &T,
    p_d: &T,
    s_up: &T,
    s_dn: &T,
) -> (T, T, T) {
    let (g, th) = (c.g, c.theta);
    let fr = cst(&c.friction);
    let le = cst(&c.edge_length);
    let phi = n2.clone() * q.clone().abs() * g / (area.clone() * radius.clone().powf_scalar(4.0 / 3.0));
    let dt_phi = phi * dt_s;
    let den = dt_phi.clone() * 2.0 + 1.0;
    let alpha = (dt_phi.clone() + 1.0) / den.clone() * fr.clone();
    let beta = area.clone() * (g * dt_s) / (le.clone() * den.clone()) * fr.clone();
    let phi_d = a_d.clone() / area.clone()
        + (a_d.clone() / area.clone() - p_d.clone() / perim.clone()) * (4.0 / 3.0);
    let dh_edge = h.clone() - hd.clone();
    let kappa = (dt_phi.clone() * q.clone() * phi_d + a_d.clone() * dh_edge.clone() * (g * dt_s) / le)
        / den
        * fr.clone();
    let q_star = alpha * q.clone() + beta.clone() * dh_edge;
    let a_up = (beta.clone() * th + kappa.clone() * s_up.clone()).clamp_min(0.0) * fr.clone();
    let b_down = (beta * th - kappa * s_dn.clone()).clamp_min(0.0) * fr;
    (q_star, a_up, b_down)
}

#[test]
fn newton_adjoint_matches_burn_autodiff() {
    let n = 8;
    let mut rep = Report::new("newton");
    let mut saw_ra_neg = false;
    let mut saw_rb_neg = false;
    for theta in [0.55f32, 1.0] {
        for dt_s in [3600.0f32, 900.0] {
            let mut c = consts(n, theta, false);
            c.edge_length = vec![1000.0; n];
            c.friction = vec![1.0; n];
            c.friction[7] = 0.0;
            let h = [3.0f32, 3.0, 2.0, 2.0, 2.0, 3.0, 5.0, 2.5];
            let hd = [2.9f32, 2.7, 1.9, 2.1, 2.05, 5.0, 3.0, 2.0];
            let q = [0.8f32, -0.6, 0.0, 0.5, 0.3, 0.01, 0.01, 0.4];
            let s_up = [1.0f32, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
            let s_dn = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
            let n2 = [0.0016f32, 0.0020, 0.0016, 0.0025, 0.0016, 0.0016, 0.0016, 0.0016];
            let area = [20.0f32, 25.0, 15.0, 30.0, 12.0, 1.0, 1.0, 18.0];
            let perim = [22.0f32, 24.0, 18.0, 26.0, 14.0, 3.0, 3.0, 20.0];
            let radius: Vec<f32> = (0..n).map(|i| area[i] / perim[i]).collect();
            let a_d = [10.0f32, 12.0, 9.0, 14.0, 8.0, 3.0, 3.0, 9.0];
            let p_d = [2.1f32, 2.0, 2.2, 2.4, 2.0, 2.0, 2.0, 2.1];
            let dep = DepthOut { h_down: hd.to_vec(), s_up: s_up.to_vec(), s_dn: s_dn.to_vec(), ..Default::default() };
            let edge = SectionHost {
                out_area: area.to_vec(),
                out_perim: perim.to_vec(),
                out_radius: radius.clone(),
                out_a_d: a_d.to_vec(),
                out_p_d: p_d.to_vec(),
                ..Default::default()
            };
            let o = newton_host(&c, dt_s, &h, &q, &n2, &dep, &edge);
            saw_ra_neg |= o.ra[5] < 0.0;
            saw_rb_neg |= o.rb[6] < 0.0;
            assert!(o.ra[5] < 0.0 && o.rb[6] < 0.0, "theta {theta} dt {dt_s}: clamp rows do not bind");

            let (lh, lhd, lq, ln2) = (leaf(&h), leaf(&hd), leaf(&q), leaf(&n2));
            let (la, lp, lr, lad, lpd) = (leaf(&area), leaf(&perim), leaf(&radius), leaf(&a_d), leaf(&p_d));
            let (qs, a, b) = newton_burn(
                &c, dt_s, &lh, &lhd, &lq, &ln2, &la, &lp, &lr, &lad, &lpd, &cst(&s_up), &cst(&s_dn),
            );
            check_fwd("q_star", &o.q_star, &val(&qs));
            check_fwd("a_up", &o.a_up, &val(&a));
            check_fwd("b_down", &o.b_down, &val(&b));
            let bars = [bar(1, n, true), bar(2, n, true), bar(3, n, true)];
            let loss = wsum(&bars[0], qs) + wsum(&bars[1], a) + wsum(&bars[2], b);
            let g = loss.backward();
            let got = newton_backward_host(
                &c,
                dt_s,
                &h,
                &q,
                &n2,
                &dep,
                &edge,
                &o,
                &NewtonBar { q_star: bars[0].clone(), a_up: bars[1].clone(), b_down: bars[2].clone() },
            );
            let tag = format!("theta {theta} dt {dt_s}");
            rep.add(&format!("{tag} qc"), &got.qc, &grad(&g, &lq, n));
            rep.add(&format!("{tag} n2"), &got.n2, &grad(&g, &ln2, n));
            rep.add(&format!("{tag} h"), &got.h, &grad(&g, &lh, n));
            rep.add(&format!("{tag} h_down"), &got.h_down, &grad(&g, &lhd, n));
            rep.add(&format!("{tag} A_e"), &got.edge.area, &grad(&g, &la, n));
            rep.add(&format!("{tag} P_e"), &got.edge.perim, &grad(&g, &lp, n));
            rep.add(&format!("{tag} R_e"), &got.edge.radius, &grad(&g, &lr, n));
            rep.add(&format!("{tag} Ad_e"), &got.edge.a_d, &grad(&g, &lad, n));
            rep.add(&format!("{tag} Pd_e"), &got.edge.p_d, &grad(&g, &lpd, n));
            // Q = 0 exactly: the |Q| kink gives the phi path no gradient to Q.
            assert!(got.edge.top_width.is_empty());
        }
    }
    assert!(saw_ra_neg && saw_rb_neg);
    rep.print();
}

// ------------------------------------------------------------------ control

/// Block 4 (4a to 4d). Returns (R, R').
#[allow(clippy::too_many_arguments)]
fn control_burn(
    c: &StageOpConsts,
    dt_s: f32,
    h: &T,
    p: &T,
    q: &T,
    n_eff: &T,
    depth: &T,
    d_s: &T,
    a_s: &T,
    p_s: &T,
    r_s: &T,
    ad_s: &T,
    pd_s: &T,
) -> (T, T) {
    let q_eps = q.clone() + 1e-6;
    let len = cst(&c.length);
    let dam_t = cst(&c.dam_t);
    let e1 = q_eps + 1.0;
    let mut r_dam = len.clone() * p.clone() * depth.clone().powf(e1.clone()) / e1 / dam_t.clone();
    let mut rp_dam = d_s.clone() * dt_s / dam_t.clone();
    if c.has_table {
        let mask = cst(&c.t_mask);
        let tw = cst(&c.t_tw);
        let pool = cst(&c.t_pool);
        let excess = h.clone() - pool;
        let active = excess.clone().detach().greater_elem(0.0).float();
        let r_tab = tw * len * excess.clamp_min(0.0) / dam_t;
        let rp_tab = rp_dam.clone() * active;
        let keep = mask.clone().neg() + 1.0;
        r_dam = r_dam * keep.clone() + r_tab * mask.clone();
        rp_dam = rp_dam * keep + rp_tab * mask;
    }
    let kk = a_s.clone() * r_s.clone().powf_scalar(2.0 / 3.0) / n_eff.clone();
    let s0_sqrt = cst(&c.s0_sqrt);
    let r_nd = kk.clone() * s0_sqrt.clone();
    let dk_dd = kk * (ad_s.clone() / a_s.clone() * (5.0 / 3.0) - pd_s.clone() / p_s.clone() * (2.0 / 3.0));
    let rp_nd = dk_dd * s0_sqrt;
    let dam = cst(&c.dam);
    let nd = cst(&c.nd);
    let r_t = dam.clone() * r_dam + nd.clone() * r_nd;
    let r_prime = (dam * rp_dam + nd * rp_nd).clamp_min(0.0);
    (r_t, r_prime)
}

fn control_run(with_table: bool) {
    let n = 8;
    let mut c = consts(n, 0.6, false);
    let dt_s = 3600.0f32;
    c.length = vec![2000.0; n];
    c.dam_t = vec![106272.0; n];
    c.s0_sqrt = vec![0.03; n];
    c.dam = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
    c.nd = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    c.has_table = with_table;
    if with_table {
        c.t_mask = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        c.t_tw = vec![60.0; n];
        c.t_pool = vec![0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0];
    } else {
        c.t_mask = vec![];
        c.t_tw = vec![];
        c.t_pool = vec![];
    }
    let h = [3.0f32, 2.0, 1.5, 3.5, 1.5, 2.5, 2.2, 3.1];
    let p = [21.0f32, 25.0, 18.0, 30.0, 22.0, 19.0, 24.0, 26.0];
    let q = [0.5f32, 0.4, 0.6, 0.5, 0.45, 0.55, 0.35, 0.65];
    let n_eff = [0.03f32, 0.035, 0.04, 0.03, 0.05, 0.045, 0.03, 0.04];
    let depth = [1.2f32, 0.8, 0.5, 1.5, 0.4, 1.0, 0.7, 1.3];
    let d_s = [50.0f32, 40.0, 30.0, 60.0, 25.0, 45.0, 35.0, -20.0];
    let a_s = [40.0f32, 30.0, 20.0, 50.0, 15.0, 35.0, 10.0, 28.0];
    let p_s = [30.0f32, 25.0, 20.0, 35.0, 18.0, 28.0, 20.0, 26.0];
    let r_s: Vec<f32> = (0..n).map(|i| a_s[i] / p_s[i]).collect();
    let ad_s = [20.0f32, 18.0, 15.0, 25.0, 12.0, 22.0, 4.0, 17.0];
    let pd_s = [2.1f32, 2.0, 2.0, 2.2, 2.0, 2.1, 40.0, 2.0];
    let q_eps: Vec<f32> = q.iter().map(|v| v + 1e-6f32).collect();
    let store = SectionHost {
        out_area: a_s.to_vec(),
        out_perim: p_s.to_vec(),
        out_radius: r_s.clone(),
        out_a_d: ad_s.to_vec(),
        out_p_d: pd_s.to_vec(),
        ..Default::default()
    };
    let o = control_host(&c, dt_s, &h, &p, &q_eps, &n_eff, &depth, &store, &d_s);
    assert_eq!(o.r_prime[6], 0.0, "nd clamp row must bind");
    assert!(o.rpr[6] < 0.0 && o.rpr[7] < 0.0);
    if with_table {
        assert_eq!(o.ex[5], 0.0, "tie row: h == pool exactly");
        assert_eq!(o.r_prime[5], 0.0);
        assert!(o.ex[3] > 0.0 && o.ex[4] < 0.0);
    }
    let (lh, lp, lq, ln, ldep) = (leaf(&h), leaf(&p), leaf(&q), leaf(&n_eff), leaf(&depth));
    let (lds, la, lpp, lr, lad, lpd) =
        (leaf(&d_s), leaf(&a_s), leaf(&p_s), leaf(&r_s), leaf(&ad_s), leaf(&pd_s));
    let (rt, rpr) = control_burn(&c, dt_s, &lh, &lp, &lq, &ln, &ldep, &lds, &la, &lpp, &lr, &lad, &lpd);
    check_fwd("R", &o.r_t, &val(&rt));
    check_fwd("R'", &o.r_prime, &val(&rpr));
    let bars = [bar(4, n, true), bar(5, n, false)];
    let loss = wsum(&bars[0], rt) + wsum(&bars[1], rpr);
    let g = loss.backward();
    let got = control_backward_host(
        &c,
        dt_s,
        &h,
        &p,
        &q_eps,
        &n_eff,
        &depth,
        &store,
        &d_s,
        &o,
        &ControlBar { r_t: bars[0].clone(), r_prime: bars[1].clone() },
    );
    let mut rep = Report::new(if with_table { "control+table" } else { "control" });
    rep.add("h", &got.h, &grad(&g, &lh, n));
    rep.add("depth", &got.depth, &grad(&g, &ldep, n));
    rep.add("p", &got.p, &grad(&g, &lp, n));
    rep.add("q_eps", &got.q_eps, &grad(&g, &lq, n));
    rep.add("n_eff", &got.n_eff, &grad(&g, &ln, n));
    rep.add("d_s", &got.d_s, &grad(&g, &lds, n));
    rep.add("A_s", &got.store.area, &grad(&g, &la, n));
    rep.add("P_s", &got.store.perim, &grad(&g, &lpp, n));
    rep.add("R_s", &got.store.radius, &grad(&g, &lr, n));
    rep.add("Ad_s", &got.store.a_d, &grad(&g, &lad, n));
    rep.add("Pd_s", &got.store.p_d, &grad(&g, &lpd, n));
    assert!(got.store.top_width.is_empty(), "the caller adds the TW_s term");
    if with_table {
        // The tie row passes tw*L/T to h while R' stays 0.
        let gh = grad(&g, &lh, n);
        assert!(gh[5] != 0.0, "tie row: h gradient from the release must be nonzero");
    }
    rep.print();
}

#[test]
fn control_adjoint_matches_burn_autodiff() {
    control_run(false);
    control_run(true);
}

// -------------------------------------------------------------------- depth

/// Block 1 (1a to 1d): returns (depth, h_down, d_edge).
fn depth_burn(c: &StageOpConsts, h: &T, hb0: f32) -> (T, T, T) {
    let lb = c.depth_lb;
    let z_mid = cst(&c.z_mid);
    let z_down = cst(&c.z_down);
    let down: Vec<i32> = c.down.iter().map(|&d| d as i32).collect();
    let idx = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints(down.as_slice(), &dev());
    let outlet = cst(&c.outlet);
    let depth = (h.clone() - z_mid.clone()).clamp_min(lb);
    let mut h_down = h.clone().gather(0, idx);
    if c.prescribed {
        h_down = h_down * (outlet.clone().neg() + 1.0) + outlet * hb0;
    }
    let d_edge_raw = h.clone().max_pair(h_down.clone()) - z_mid.max_pair(z_down);
    let d_edge = d_edge_raw.clamp_min(lb);
    (depth, h_down, d_edge)
}

fn depth_run(label: &str, c: &StageOpConsts, h: &[f32], hb0: f32, rep: &mut Report) {
    let n = c.n;
    let o = depth_host(c, h, hb0);
    let lh = leaf(h);
    let (dep, hd, de) = depth_burn(c, &lh, hb0);
    check_fwd("depth", &o.depth, &val(&dep));
    check_fwd("h_down", &o.h_down, &val(&hd));
    check_fwd("d_edge", &o.d_edge, &val(&de));
    let bars = [bar(6, n, true), bar(7, n, true), bar(8, n, true)];
    let loss = wsum(&bars[0], dep) + wsum(&bars[1], de) + wsum(&bars[2], hd);
    let g = loss.backward();
    let got = depth_backward_host(c, h, &o, &bars[0], &bars[1], &bars[2]);
    rep.add(label, &got, &grad(&g, &lh, n));
}

#[test]
fn depth_adjoint_matches_burn_autodiff() {
    let n = 5;
    let mut rep = Report::new("depth");
    // generic states on a NormalDepth and a PrescribedStage chain
    for prescribed in [false, true] {
        let c = consts(n, 0.6, prescribed);
        let h: Vec<f32> = (0..n).map(|i| c.z_mid[i] + 0.4 + 0.3 * ((i * 7) % 5) as f32).collect();
        depth_run(&format!("generic prescribed={prescribed}"), &c, &h, 0.5, &mut rep);
    }
    // exact ties, built in f32
    for prescribed in [false, true] {
        let mut c = consts(n, 0.6, prescribed);
        c.z_mid = vec![0.0, 0.0, 0.3, 0.2, 0.1];
        c.z_down = vec![0.2, 0.0, 0.1, 0.1, 0.0];
        let lb = c.depth_lb;
        // row 0: x == lb. row 1: x == lb and der == lb (z_mid = z_down = 0, h[1] > h[2]).
        // row 3: h == h[down]. row 4: the outlet (self-gather) tie, or PrescribedStage.
        let h = vec![lb, lb, 0.005, 1.5, 1.5];
        let o = depth_host(&c, &h, 1.0);
        assert_eq!(o.x[0], lb);
        assert_eq!(o.x[1], lb);
        assert_eq!(o.der[1], lb, "der == lb tie");
        assert_eq!(h[3], h[c.down[3]], "max-pair tie");
        depth_run(&format!("ties prescribed={prescribed}"), &c, &h, 1.0, &mut rep);
    }
    rep.print();
}

// ------------------------------------------------- readout, solve, matrix, RHS

/// Blocks 5 to 8 of one substep, copied from `StageBlock::step_reference` with
/// `tree_lu_solve`. Returns `(h_new, q_new, dh)`.
#[allow(clippy::too_many_arguments)]
fn assemble_solve_burn(
    c: &StageOpConsts,
    pat: &Arc<CsrPattern>,
    hb: (f32, f32),
    leaves: &[T; 10], // d_s, a_up, b_down, r_prime, r_t, q_star, qc, qp, i_mid, h
) -> (T, T, T) {
    use burn::tensor::{IndexingUpdateOp, Int};
    let idx = |v: &[usize]| {
        let w: Vec<i32> = v.iter().map(|&x| x as i32).collect();
        Tensor::<TestBackend, 1, Int>::from_ints(w.as_slice(), &dev())
    };
    let (row_idx, col_idx, down_idx) = (idx(&c.row), idx(&c.col), idx(&c.down));
    let scatter = |vals: T| T::zeros([c.n], &dev()).scatter(0, row_idx.clone(), vals, IndexingUpdateOp::Add);
    let [d_s, a_up, b_down, r_prime, r_t, q_star, qc, qp, i_mid, h] = leaves.clone();
    let (th, thc) = (c.theta, c.theta_c);
    let (hbc0, hbc1) = hb;
    let one_minus = |t: T| t.neg().add_scalar(1.0);
    let (in_stage, w_ctrl, friction, ctrl, outlet) =
        (cst(&c.in_stage), cst(&c.w_ctrl), cst(&c.friction), cst(&c.ctrl), cst(&c.outlet));
    let (adj_friction, adj_stage, dam_slot, diag_slot) =
        (cst(&c.adj_friction), cst(&c.adj_stage), cst(&c.dam_slot), cst(&c.diag_slot));
    // 5
    let a_edge = a_up.clone().gather(0, col_idx.clone()) * adj_friction.clone();
    let b_edge = b_down.clone().gather(0, col_idx.clone()) * adj_friction;
    let lower_f = a_edge.neg() * th;
    let upper_f = b_edge.clone().neg() * th;
    let off_dam = r_prime.clone().gather(0, col_idx) * dam_slot * (-thc);
    let sum_in_b = scatter(b_edge);
    let diag_stage = d_s + (a_up.clone() + sum_in_b) * th;
    let diag = diag_stage * in_stage.clone() + w_ctrl.clone() * r_prime.clone() + one_minus(in_stage);
    let lower = lower_f + off_dam + diag.gather(0, row_idx.clone()) * diag_slot;
    let a_values = Tensor::cat(vec![lower, upper_f], 0);
    // 6
    let flux_old = friction * ((q_star.clone() * th) + qc.clone() * (1.0 - th));
    let ctrl_old = w_ctrl.clone() * r_t.clone() + (ctrl.clone() - w_ctrl) * qc;
    let e_old = flux_old + ctrl_old;
    let in_e = scatter(e_old.clone().gather(0, idx(&c.col)) * adj_stage);
    let mut r = qp + i_mid + in_e - e_old;
    if c.prescribed {
        r = r + outlet.clone() * b_down.clone() * (th * (hbc1 - hbc0));
    }
    // 7
    let dh = ddrs::sparse::tree_lu::tree_lu_solve(pat, a_values, r, false);
    let h_solved = h.clone() + dh.clone();
    let h_floor = cst(&c.z_mid) + c.depth_lb;
    let h_new = h_solved.max_pair(h_floor);
    // 8
    let mut dh_down = dh.clone().gather(0, down_idx);
    if c.prescribed {
        dh_down = dh_down * (outlet.clone().neg() + 1.0) + outlet * (hbc1 - hbc0);
    }
    let q_fr = q_star + a_up * dh.clone() - b_down * dh_down;
    let q_ctrl = (r_t + r_prime * dh.clone()) * ctrl;
    (h_new, q_fr + q_ctrl, dh)
}

fn readout_solve_run(label: &str, floored: bool, rep: &mut Report) {
    use ddrs::routing::stage_op::{
        matrix_backward_host, matrix_host, readout_backward_host, rhs_backward_host, rhs_host,
        stage_readout_host, Assembled, ControlOut, HourHost, NewtonOut,
    };
    use ddrs::sparse::tree_lu::{
        tree_lu_factor, tree_lu_grad_a_host, tree_lu_solve_host, tree_lu_solve_transposed_host,
    };
    let n = 5;
    let (adj, pat) = chain(n);
    // MC headwater (row 0), a dam on row 2, PrescribedStage
    let net = StageNetwork::build(&adj, &pat, Some(&[1]), 0.0, &[(2, 1.23)]).unwrap();
    let params = StageParams {
        theta: 0.7,
        theta_control: 0.5,
        substeps: 1,
        boundary: DownstreamBoundary::PrescribedStage,
        depth_lb: 0.01,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    };
    let mut block = StageBlock::<InnerBackend>::new(net, params, &dev());
    let c = (*block.op_consts(&pat)).clone();
    let pos = |seed, lo: f32, hi: f32| -> Vec<f32> { bar(seed, n, false).iter().map(|v| lo + (v - 0.5) * (hi - lo)).collect() };
    let d_s = pos(1, 1.0, 2.0);
    let a_up = pos(2, 0.0, 0.3);
    let b_down = pos(3, 0.0, 0.3);
    let r_prime = pos(4, 0.0, 0.3);
    let r_t = bar(5, n, true);
    let q_star = bar(6, n, true);
    let qc = bar(7, n, true);
    let qp: Vec<f32> = bar(8, n, false).iter().zip(&c.in_stage).map(|(v, m)| v * m).collect();
    let i_mid = bar(9, n, false);
    let h: Vec<f32> = if floored {
        c.z_mid.iter().map(|z| z - 3.0).collect()
    } else {
        c.z_mid.iter().enumerate().map(|(i, z)| z + 0.5 + 0.1 * i as f32).collect()
    };
    let (hb0, hb1) = (0.4f32, 0.9f32);
    let host_leaves: [&Vec<f32>; 10] = [&d_s, &a_up, &b_down, &r_prime, &r_t, &q_star, &qc, &qp, &i_mid, &h];
    let lv: [T; 10] = std::array::from_fn(|k| leaf(host_leaves[k]));
    let (h_new_b, q_new_b, dh_b) = assemble_solve_burn(&c, &pat, (hb0, hb1), &lv);
    let (wq, wv) = (bar(20, n, true), bar(21, n, true));
    let loss = wsum(&wq, q_new_b.clone()) + wsum(&wv, h_new_b.clone());
    let g = loss.backward();

    // host forward
    let a_values = matrix_host(&c, &d_s, &a_up, &b_down, &r_prime);
    let r = rhs_host(&c, &qc, &q_star, &r_t, &b_down, &qp, &i_mid, hb0, hb1);
    let f = tree_lu_factor(&pat, &a_values);
    let dh = tree_lu_solve_host(&pat, &f, &r);
    let hour = HourHost { qp_stage: qp.clone(), ..Default::default() };
    let asm = Assembled {
        i_mid: i_mid.clone(),
        hb0,
        hb1,
        dt_s: 3600.0,
        d_s: d_s.clone(),
        newton: NewtonOut { q_star: q_star.clone(), a_up: a_up.clone(), b_down: b_down.clone(), ..Default::default() },
        control: ControlOut { r_t: r_t.clone(), r_prime: r_prime.clone(), ..Default::default() },
        ..Default::default()
    };
    let ro = stage_readout_host(&c, &hour, &h, &qc, &asm, &dh);
    check_fwd("dh", &dh, &val(&dh_b));
    check_fwd("h_new", &ro.h_new, &val(&h_new_b));
    check_fwd("q_new", &ro.q_new, &val(&q_new_b));
    if floored {
        let hf: Vec<f32> = c.z_mid.iter().map(|z| z + c.depth_lb).collect();
        assert_eq!(ro.h_new, hf, "every row is on the floor");
    }

    // host adjoint
    let rg = readout_backward_host(&c, &h, &asm, &dh, &wv, &wq);
    let g_r = tree_lu_solve_transposed_host(&pat, &f, &rg.dh);
    let g_a = tree_lu_grad_a_host(&pat, &g_r, &dh);
    let rhs = rhs_backward_host(&c, &asm, &g_r);
    let mat = matrix_backward_host(&c, &g_a);
    let add = |xs: &[&Vec<f32>]| -> Vec<f32> { (0..n).map(|i| xs.iter().map(|x| x[i]).sum()).collect() };
    let names = ["d_s", "a_up", "b_down", "r_prime", "r_t", "q_star", "qc", "qp", "i_mid", "h"];
    let totals: [Vec<f32>; 10] = [
        mat.d_s.clone(),
        add(&[&rg.a_up, &mat.a_up]),
        add(&[&rg.b_down, &rhs.b_down, &mat.b_down]),
        add(&[&rg.r_prime, &mat.r_prime]),
        add(&[&rg.r_t, &rhs.r_t]),
        add(&[&rg.q_star, &rhs.q_star]),
        rhs.qc.clone(),
        rhs.qp.clone(),
        rhs.i_mid.clone(),
        rg.h.clone(),
    ];
    for k in 0..10 {
        rep.add(&format!("{label} {}", names[k]), &totals[k], &grad(&g, &lv[k], n));
    }
}

#[test]
fn readout_solve_matrix_rhs_adjoint_matches_burn_autodiff() {
    let mut rep = Report::new("readout+solve+matrix+rhs");
    readout_solve_run("generic", false, &mut rep);
    readout_solve_run("floor binds", true, &mut rep);
    rep.print();
}
