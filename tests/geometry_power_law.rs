//! Self-consistency checks for the power-law channel section in
//! `src/geometry.rs`: `T = p d^q`, `A = T d/(q+1)`, `R = d/(q+1)`, `v = Q/A`.
//!
//! References are computed in f64 inside the tests. The routing code is f32.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::geometry::{
    compute_power_law_geometry, compute_power_law_geometry_field, compute_trapezoidal_geometry,
    power_law_depth_derivatives, power_law_from_depth,
};

type B = Autodiff<NdArray<f32>>;

const DEPTH_LB: f32 = 0.01;

#[derive(Clone, Copy)]
struct Case {
    q_in: f64,
    n: f64,
    p: f64,
    q: f64,
    s: f64,
    gamma: f64,
    d_ref: f64,
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for &q_in in &[0.5, 10.0, 100.0, 5000.0] {
        for &n in &[0.02, 0.05] {
            for &p in &[5.0, 21.0, 80.0] {
                for &q in &[0.05, 0.084, 0.4, 0.65, 0.95] {
                    for &s in &[1e-3, 1e-2] {
                        for &gamma in &[0.0, 0.35] {
                            v.push(Case { q_in, n, p, q, s, gamma, d_ref: 1.0 });
                        }
                    }
                }
            }
        }
    }
    v.push(Case { q_in: 100.0, n: 0.035, p: 21.0, q: 0.65, s: 1e-3, gamma: 0.2, d_ref: 2.0 });
    v
}

fn dev() -> burn::backend::ndarray::NdArrayDevice {
    Default::default()
}

fn t1(v: Vec<f32>) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats(v.as_slice(), &dev())
}

fn vals(t: &Tensor<B, 1>) -> Vec<f64> {
    t.clone().into_data().to_vec::<f32>().unwrap().into_iter().map(f64::from).collect()
}

struct Row {
    depth: f64,
    area: f64,
    velocity: f64,
    celerity_ratio: f64,
}

/// Run the scalar-gamma function once per distinct `(gamma, d_ref)` group.
fn run(cs: &[Case]) -> Vec<Row> {
    let mut rows: Vec<Option<Row>> = (0..cs.len()).map(|_| None).collect();
    let mut groups: Vec<(f64, f64)> = Vec::new();
    for c in cs {
        if !groups.contains(&(c.gamma, c.d_ref)) {
            groups.push((c.gamma, c.d_ref));
        }
    }
    for (g, dr) in groups {
        let idx: Vec<usize> =
            (0..cs.len()).filter(|&i| cs[i].gamma == g && cs[i].d_ref == dr).collect();
        let pick = |f: &dyn Fn(&Case) -> f64| t1(idx.iter().map(|&i| f(&cs[i]) as f32).collect());
        let out = compute_power_law_geometry::<B>(
            pick(&|c| c.n),
            pick(&|c| c.p),
            pick(&|c| c.q),
            pick(&|c| c.q_in),
            pick(&|c| c.s),
            DEPTH_LB,
            g as f32,
            dr as f32,
        );
        let (d, a, v, b) =
            (vals(&out.depth), vals(&out.area), vals(&out.velocity), vals(&out.celerity_ratio));
        for (k, &i) in idx.iter().enumerate() {
            rows[i] = Some(Row { depth: d[k], area: a[k], velocity: v[k], celerity_ratio: b[k] });
        }
    }
    rows.into_iter().map(|r| r.unwrap()).collect()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-30)
}

fn closed_form_depth(c: &Case) -> f64 {
    let qe = c.q + 1e-6;
    let g = c.gamma;
    (c.q_in * c.n * (qe + 1.0).powf(5.0 / 3.0) * c.d_ref.powf(g) / (c.p * c.s.sqrt()))
        .powf(3.0 / (5.0 + 3.0 * qe + 3.0 * g))
}

fn manning_v(c: &Case, d: f64) -> f64 {
    let qe = c.q + 1e-6;
    (1.0 / c.n) * (d / c.d_ref).powf(c.gamma) * (d / (qe + 1.0)).powf(2.0 / 3.0) * c.s.sqrt()
}

fn area64(c: &Case, d: f64) -> f64 {
    let qe = c.q + 1e-6;
    c.p * d.powf(qe + 1.0) / (qe + 1.0)
}

#[test]
fn depth_matches_the_closed_form() {
    let cs = cases();
    let rows = run(&cs);
    let mut worst = 0.0_f64;
    let mut floored = 0;
    for (c, r) in cs.iter().zip(&rows) {
        let d = closed_form_depth(c);
        if d > DEPTH_LB as f64 {
            let e = rel(r.depth, d);
            worst = worst.max(e);
            assert!(
                e < 1e-5,
                "depth rel err {e} at q_in {} n {} p {} q {} s {} g {}",
                c.q_in, c.n, c.p, c.q, c.s, c.gamma
            );
        } else {
            floored += 1;
            assert_eq!(r.depth as f32, DEPTH_LB, "floored depth must equal depth_lb");
        }
    }
    println!("worst depth rel err {worst:e}, floored cases {floored}");
}

#[test]
fn discharge_equals_manning_velocity_times_area() {
    let cs = cases();
    let rows = run(&cs);
    let (mut worst64, mut worst32) = (0.0_f64, 0.0_f64);
    for (c, r) in cs.iter().zip(&rows) {
        let e32 = (r.velocity * r.area - c.q_in).abs() / c.q_in;
        worst32 = worst32.max(e32);
        assert!(e32 < 1e-6, "f32 v*A vs Q: {e32}");
        if r.depth as f32 > DEPTH_LB {
            let e = (manning_v(c, r.depth) * area64(c, r.depth) - c.q_in).abs() / c.q_in;
            worst64 = worst64.max(e);
            assert!(e < 1e-5, "Manning v*A vs Q: {e}");
        }
    }
    println!("worst manning rel err {worst64:e}, worst f32 v*A rel err {worst32:e}");
}

#[test]
fn celerity_equals_dq_da_by_central_difference() {
    let mut worst = 0.0_f64;
    for cc in cases() {
        let rows = run(&[cc]);
        let r = &rows[0];
        if r.depth as f32 <= DEPTH_LB {
            continue;
        }
        let d = r.depth;
        let h = 1e-6 * d;
        let rating = |x: f64| manning_v(&cc, x) * area64(&cc, x);
        let c_fd = (rating(d + h) - rating(d - h)) / (area64(&cc, d + h) - area64(&cc, d - h));
        let e = rel(r.velocity * r.celerity_ratio, c_fd);
        worst = worst.max(e);
        assert!(e < 1e-4, "celerity rel err {e}");
    }
    println!("worst celerity rel err {worst:e}");
}

#[test]
fn area_derivative_is_top_width() {
    let mut worst = 0.0_f64;
    for (p, q, d) in
        [(21.0_f64, 0.65_f64, 1.3_f64), (5.0, 0.05, 0.2), (80.0, 0.95, 4.0), (21.0, 0.084, 0.7)]
    {
        let c = Case { q_in: 1.0, n: 0.03, p, q, s: 1e-3, gamma: 0.0, d_ref: 1.0 };
        let h = 1e-6 * d;
        let fd = (area64(&c, d + h) - area64(&c, d - h)) / (2.0 * h);
        let tw = p * d.powf(q + 1e-6);
        let e = rel(fd, tw);
        worst = worst.max(e);
        assert!(e < 1e-6, "dA/dd vs T: {e}");

        let sec =
            power_law_from_depth::<B>(t1(vec![p as f32]), t1(vec![q as f32]), t1(vec![d as f32]));
        let (da, dr) = power_law_depth_derivatives::<B>(t1(vec![q as f32]), &sec);
        let da = da.into_data().to_vec::<f32>().unwrap()[0];
        let t = sec.top_width.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(da.to_bits(), t.to_bits(), "dA/dd must be top_width bitwise");
        let dr = dr.into_data().to_vec::<f32>().unwrap()[0] as f64;
        assert!(rel(dr, 1.0 / (q + 1e-6 + 1.0)) < 1e-6, "dR/dd");
        let r = sec.hydraulic_radius.into_data().to_vec::<f32>().unwrap()[0] as f64;
        assert!(rel(r, d / (q + 1e-6 + 1.0)) < 1e-6, "R = d/(q+1)");
    }
    println!("worst dA/dd vs T rel err {worst:e}");
}

#[test]
fn depth_floor_cuts_n() {
    let n = t1(vec![0.035]).require_grad();
    let out = compute_power_law_geometry::<B>(
        n.clone(),
        t1(vec![21.0]),
        t1(vec![0.65]),
        t1(vec![1e-6]),
        t1(vec![1e-3]),
        DEPTH_LB,
        0.0,
        1.0,
    );
    let d = out.depth.clone().into_data().to_vec::<f32>().unwrap()[0];
    assert_eq!(d, DEPTH_LB);
    let grads = out.depth.sum().backward();
    let g = n.grad(&grads).map(|g| g.into_data().to_vec::<f32>().unwrap()[0]).unwrap_or(0.0);
    assert_eq!(g, 0.0, "floored depth must not pass gradient to n");
}

#[test]
fn field_matches_scalar() {
    let cs: Vec<Case> = cases().into_iter().filter(|c| c.gamma == 0.35).collect();
    let pick = |f: &dyn Fn(&Case) -> f64| t1(cs.iter().map(|c| f(c) as f32).collect());
    let a = compute_power_law_geometry::<B>(
        pick(&|c| c.n),
        pick(&|c| c.p),
        pick(&|c| c.q),
        pick(&|c| c.q_in),
        pick(&|c| c.s),
        DEPTH_LB,
        0.35,
        1.0,
    );
    let b = compute_power_law_geometry_field::<B>(
        pick(&|c| c.n),
        pick(&|c| c.p),
        pick(&|c| c.q),
        pick(&|c| c.q_in),
        pick(&|c| c.s),
        DEPTH_LB,
        Tensor::<B, 1>::full([cs.len()], 0.35, &dev()),
        1.0,
    );
    let fields: [(&str, &Tensor<B, 1>, &Tensor<B, 1>); 8] = [
        ("ln_ratio", &a.ln_ratio, &b.ln_ratio),
        ("exponent", &a.exponent, &b.exponent),
        ("depth", &a.depth, &b.depth),
        ("top_width", &a.top_width, &b.top_width),
        ("area", &a.area, &b.area),
        ("hydraulic_radius", &a.hydraulic_radius, &b.hydraulic_radius),
        ("velocity", &a.velocity, &b.velocity),
        ("celerity_ratio", &a.celerity_ratio, &b.celerity_ratio),
    ];
    let mut worst = 0.0_f64;
    for (name, x, y) in fields {
        for (u, w) in vals(x).into_iter().zip(vals(y)) {
            let e = (u - w).abs() / w.abs().max(1e-12);
            worst = worst.max(e);
            assert!(e < 1e-6, "{name}: scalar {u} vs field {w}");
        }
    }
    println!("worst field vs scalar rel err {worst:e}");
}

#[test]
fn trapezoid_is_not_self_consistent() {
    // Pins the defect of the trapezoid path: v * A is not Q at the computed depth.
    let qs = [1.0_f32, 10.0, 100.0, 1000.0];
    let g = compute_trapezoidal_geometry::<B>(
        t1(vec![0.035; 4]),
        t1(vec![21.0; 4]),
        t1(vec![0.65; 4]),
        t1(qs.to_vec()),
        t1(vec![1e-3; 4]),
        DEPTH_LB,
        0.1,
    );
    let v = vals(&g.velocity);
    let a = vals(&g.cross_sectional_area);
    for i in 0..4 {
        let ratio = v[i] * a[i] / qs[i] as f64;
        println!("trapezoid v*A/Q at Q={}: {ratio}", qs[i]);
        assert!(ratio > 0.80 && ratio < 0.90, "ratio {ratio}");
    }
}

/// Prints the `vA/Q` spread of the trapezoid path over a grid on the box of
/// `config/merit_training.yaml` (`n` 0.015 to 0.25 and `p` 1 to 200, both
/// log-spaced; `q` 0.05 to 0.95; `Q` 0.5 to 5000 m3/s, log-spaced; slope
/// 1e-3 to 3e-2, log-spaced). It is the source of the figures in
/// `.claude/PHYSICS-CORRECTIONS.md`. Run:
/// `cargo test --test geometry_power_law trapezoid_va_over_q_spread -- --ignored --nocapture`
#[test]
#[ignore]
fn trapezoid_va_over_q_spread() {
    fn logspace(lo: f64, hi: f64, k: usize) -> Vec<f64> {
        (0..k).map(|i| (lo.ln() + (hi.ln() - lo.ln()) * i as f64 / (k - 1) as f64).exp()).collect()
    }
    let ns = logspace(0.015, 0.25, 6);
    let ps = logspace(1.0, 200.0, 7);
    let qs: Vec<f64> = (0..7).map(|i| 0.05 + 0.15 * i as f64).collect();
    let flows = logspace(0.5, 5000.0, 7);
    let ss = logspace(1e-3, 3e-2, 4);
    let (mut n_v, mut p_v, mut q_v, mut f_v, mut s_v) = (vec![], vec![], vec![], vec![], vec![]);
    for &n in &ns {
        for &p in &ps {
            for &q in &qs {
                for &f in &flows {
                    for &s in &ss {
                        n_v.push(n as f32);
                        p_v.push(p as f32);
                        q_v.push(q as f32);
                        f_v.push(f as f32);
                        s_v.push(s as f32);
                    }
                }
            }
        }
    }
    let g = compute_trapezoidal_geometry::<B>(
        t1(n_v),
        t1(p_v),
        t1(q_v),
        t1(f_v.clone()),
        t1(s_v),
        DEPTH_LB,
        0.1,
    );
    let (v, a) = (vals(&g.velocity), vals(&g.cross_sectional_area));
    let mut ratio: Vec<f64> = (0..v.len()).map(|i| v[i] * a[i] / f_v[i] as f64).collect();
    ratio.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let pct = |p: f64| ratio[((ratio.len() - 1) as f64 * p).round() as usize];
    println!(
        "trapezoid vA/Q over {} grid points: median {:.3}, 5% {:.3}, 95% {:.3}",
        ratio.len(),
        pct(0.5),
        pct(0.05),
        pct(0.95)
    );
}
