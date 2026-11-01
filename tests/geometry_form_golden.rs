//! Bitwise pins for the trapezoid Muskingum-Cunge op and the frozen geometry
//! functions.
//!
//! Each pin stores the `f32::to_bits` value of every output and every parent
//! gradient. A pin fails on any one-bit change. The pins prove that the default
//! (trapezoid) path of the fused op, and the four frozen functions in
//! `src/geometry.rs`, give the same bits as the commit named above `GOLDEN`.
//!
//! Fused-op cases use the 4-reach chain `0 -> 1 -> 2 -> 3`. The loss is
//! `q_next.sum()` and every parent is tracked in one backward.
//! Geometry cases take the gradient of the SUM of each output field with
//! respect to every input, with one backward per field.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::{Config, StageRoughnessSection};
use ddrs::geometry::{
    compute_trapezoidal_geometry, compute_trapezoidal_geometry_gamma, trapezoid_depth_derivatives,
    trapezoid_from_depth,
};
use ddrs::routing::mmc_op::{timestep_forward, timestep_forward_leakance};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;
type B = Autodiff<I>;

const N: usize = 4;

type Entries = Vec<(String, Vec<f32>)>;

// ---------------------------------------------------------------------------
// Fused-op cases
// ---------------------------------------------------------------------------

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

fn chain(length_m: f32) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![length_m; N], vec![1e-3; N])
}

#[derive(Copy, Clone)]
enum Variant {
    Plain,
    Gamma,
    Leakance,
}

fn grad_of(
    t: &Tensor<B, 1>,
    grads: &<B as burn::tensor::backend::AutodiffBackend>::Gradients,
) -> Vec<f32> {
    t.grad(grads)
        .expect("grad on tracked parent")
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

fn fused_case(name: &str, length_m: f32, cfg: Config, variant: Variant) -> Entries {
    let adj = chain(length_m);
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);

    let tracked = |v: Vec<f32>| -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(v.as_slice(), &device).require_grad()
    };
    let constant =
        |v: Vec<f32>| -> Tensor<B, 1> { Tensor::<B, 1>::from_floats(v.as_slice(), &device) };

    let n = tracked(vec![0.035; N]);
    let qsp = tracked(vec![0.4; N]);
    let psp = tracked(vec![20.0; N]);
    let qt = tracked(vec![100.0, 120.0, 140.0, 160.0]);
    let qpt = tracked(vec![10.0, 12.0, 14.0, 16.0]);
    let length = constant(adj.length_m.clone());
    let slope = constant(adj.slope.clone());
    let xst = constant(vec![0.3; N]);

    let mut extra: Vec<(&str, Tensor<B, 1>)> = vec![];
    let q_next = match variant {
        Variant::Plain => timestep_forward::<I>(
            &cfg,
            &pattern,
            &assembler,
            n.clone(),
            qsp.clone(),
            psp.clone(),
            qt.clone(),
            qpt.clone(),
            length,
            slope,
            xst,
            false,
            None,
            None,
        ),
        Variant::Gamma => {
            let gamma = tracked(vec![0.2; N]);
            extra.push(("gamma", gamma.clone()));
            timestep_forward::<I>(
                &cfg,
                &pattern,
                &assembler,
                n.clone(),
                qsp.clone(),
                psp.clone(),
                qt.clone(),
                qpt.clone(),
                length,
                slope,
                xst,
                false,
                None,
                Some(gamma),
            )
        }
        Variant::Leakance => {
            let kd = tracked(vec![5e-7; N]);
            let dgw = tracked(vec![0.0; N]);
            let fac = tracked(vec![0.5; N]);
            extra.push(("k_d", kd.clone()));
            extra.push(("d_gw", dgw.clone()));
            extra.push(("leakance_factor", fac.clone()));
            timestep_forward_leakance::<I>(
                &cfg,
                &pattern,
                &assembler,
                n.clone(),
                qsp.clone(),
                psp.clone(),
                qt.clone(),
                qpt.clone(),
                length,
                slope,
                xst,
                kd,
                dgw,
                fac,
                None,
                None,
                false,
                None,
                None,
            )
        }
    };

    let out = q_next.clone().into_data().to_vec::<f32>().unwrap();
    let grads = q_next.sum().backward();

    let mut e: Entries = vec![(format!("{name}/q_next"), out)];
    for (key, t) in [
        ("n", &n),
        ("q_spatial", &qsp),
        ("p_spatial", &psp),
        ("q_t", &qt),
        ("q_prime_t", &qpt),
    ] {
        e.push((format!("{name}/{key}"), grad_of(t, &grads)));
    }
    for (key, t) in &extra {
        e.push((format!("{name}/{key}"), grad_of(t, &grads)));
    }
    e
}

fn case_g1() -> Entries {
    fused_case("g1", 1000.0, mock_cfg(), Variant::Plain)
}
fn case_g1x() -> Entries {
    fused_case("g1x", 10_000.0, mock_cfg(), Variant::Plain)
}
fn case_g2() -> Entries {
    let mut cfg = mock_cfg();
    cfg.params.stage_roughness = Some(StageRoughnessSection {
        gamma: 0.35,
        d_ref: 1.0,
    });
    fused_case("g2", 10_000.0, cfg, Variant::Plain)
}
fn case_g3() -> Entries {
    fused_case("g3", 10_000.0, mock_cfg(), Variant::Gamma)
}
fn case_g4() -> Entries {
    let mut cfg = mock_cfg();
    cfg.params.enforce_positivity = true;
    fused_case("g4", 10_000.0, cfg, Variant::Plain)
}
fn case_g5() -> Entries {
    let mut cfg = mock_cfg();
    cfg.params.ddr_match = true;
    fused_case("g5", 1000.0, cfg, Variant::Plain)
}
fn case_g6() -> Entries {
    fused_case("g6", 10_000.0, mock_cfg(), Variant::Leakance)
}

// ---------------------------------------------------------------------------
// Frozen geometry interface cases
// ---------------------------------------------------------------------------

fn tr(v: &[f32]) -> Tensor<B, 1> {
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    Tensor::<B, 1>::from_floats(v, &device).require_grad()
}

const Q_SP: [f32; 4] = [0.084, 0.4, 0.65, 0.999];
const DISCHARGE: [f32; 4] = [1000.0, 1e-5, 100.0, 1e-5];
const DEPTH: [f32; 4] = [9.0, 0.01, 2.5, 0.01];

/// Push the forward value and the gradient wrt each input for one output field.
fn field(
    e: &mut Entries,
    case: &str,
    name: &str,
    out: Tensor<B, 1>,
    inputs: &[(&str, &Tensor<B, 1>)],
) {
    e.push((
        format!("{case}/{name}"),
        out.clone().into_data().to_vec::<f32>().unwrap(),
    ));
    let grads = out.sum().backward();
    for (iname, t) in inputs {
        let g = t
            .grad(&grads)
            .map(|g| g.into_data().to_vec::<f32>().unwrap())
            .unwrap_or_default();
        e.push((format!("{case}/{name}.d_{iname}"), g));
    }
}

const TRAP_FIELDS: usize = 8;

fn trap_field(g: ddrs::geometry::TrapezoidalGeometry<B>, f: usize) -> (&'static str, Tensor<B, 1>) {
    match f {
        0 => ("depth", g.depth),
        1 => ("top_width", g.top_width),
        2 => ("bottom_width", g.bottom_width),
        3 => ("side_slope", g.side_slope),
        4 => ("cross_sectional_area", g.cross_sectional_area),
        5 => ("wetted_perimeter", g.wetted_perimeter),
        6 => ("hydraulic_radius", g.hydraulic_radius),
        _ => ("velocity", g.velocity),
    }
}

fn case_f1() -> Entries {
    let mut e = vec![];
    // Rebuild the graph for every field so each backward is independent.
    for f in 0..TRAP_FIELDS {
        let n = tr(&[0.035; 4]);
        let p = tr(&[21.0; 4]);
        let q = tr(&Q_SP);
        let qd = tr(&DISCHARGE);
        let s = tr(&[1e-3; 4]);
        let g = compute_trapezoidal_geometry::<B>(
            n.clone(),
            p.clone(),
            q.clone(),
            qd.clone(),
            s.clone(),
            0.01,
            0.01,
        );
        let (name, out) = trap_field(g, f);
        field(
            &mut e,
            "f1",
            name,
            out,
            &[("n", &n), ("p", &p), ("q", &q), ("Q", &qd), ("s", &s)],
        );
    }
    e
}

fn case_f2() -> Entries {
    let mut e = vec![];
    for f in 0..TRAP_FIELDS {
        let n = tr(&[0.035; 4]);
        let p = tr(&[21.0; 4]);
        let q = tr(&Q_SP);
        let qd = tr(&DISCHARGE);
        let s = tr(&[1e-3; 4]);
        let g = compute_trapezoidal_geometry_gamma::<B>(
            n.clone(),
            p.clone(),
            q.clone(),
            qd.clone(),
            s.clone(),
            0.01,
            0.01,
            0.35,
            1.0,
        );
        let (name, out) = trap_field(g, f);
        field(
            &mut e,
            "f2",
            name,
            out,
            &[("n", &n), ("p", &p), ("q", &q), ("Q", &qd), ("s", &s)],
        );
    }
    e
}

fn case_f3() -> Entries {
    let mut e = vec![];
    for f in 0..6 {
        let p = tr(&[21.0; 4]);
        let q = tr(&Q_SP);
        let d = tr(&DEPTH);
        let g = trapezoid_from_depth::<B>(p.clone(), q.clone(), d.clone(), 0.01);
        let (name, out) = match f {
            0 => ("top_width", g.top_width),
            1 => ("bottom_width", g.bottom_width),
            2 => ("side_slope", g.side_slope),
            3 => ("area", g.area),
            4 => ("wetted_perimeter", g.wetted_perimeter),
            _ => ("hydraulic_radius", g.hydraulic_radius),
        };
        field(
            &mut e,
            "f3",
            name,
            out,
            &[("p", &p), ("q", &q), ("depth", &d)],
        );
    }
    e
}

fn case_f4() -> Entries {
    let mut e = vec![];
    for f in 0..2 {
        let p = tr(&[21.0; 4]);
        let q = tr(&Q_SP);
        let d = tr(&DEPTH);
        let g = trapezoid_from_depth::<B>(p.clone(), q.clone(), d.clone(), 0.01);
        let (area_d, per_d) = trapezoid_depth_derivatives::<B>(q.clone(), d.clone(), &g, 0.01);
        let (name, out) = if f == 0 {
            ("area_d", area_d)
        } else {
            ("per_d", per_d)
        };
        field(
            &mut e,
            "f4",
            name,
            out,
            &[("p", &p), ("q", &q), ("depth", &d)],
        );
    }
    e
}

// ---------------------------------------------------------------------------
// Capture tool and pins
// ---------------------------------------------------------------------------

fn all_cases() -> Vec<(&'static str, fn() -> Entries)> {
    vec![
        ("g1", case_g1 as fn() -> Entries),
        ("g1x", case_g1x),
        ("g2", case_g2),
        ("g3", case_g3),
        ("g4", case_g4),
        ("g5", case_g5),
        ("g6", case_g6),
        ("f1", case_f1),
        ("f2", case_f2),
        ("f3", case_f3),
        ("f4", case_f4),
    ]
}

/// Capture tool. Run it ONCE at the commit named above `GOLDEN`, paste its
/// output over `GOLDEN`, and never re-run it to "fix" a failing pin.
///
/// `cargo test --test geometry_form_golden print_golden -- --ignored --nocapture`
#[test]
#[ignore]
fn print_golden() {
    println!("const GOLDEN: &[(&str, &[u32])] = &[");
    for (_, f) in all_cases() {
        for (key, vals) in f() {
            let bits: Vec<String> = vals
                .iter()
                .map(|v| format!("0x{:08x}", v.to_bits()))
                .collect();
            println!("    (\"{key}\", &[{}]),", bits.join(", "));
        }
    }
    println!("];");
}

fn check(case: &str, entries: Entries) {
    let mut seen = 0;
    for (key, got) in &entries {
        let want = GOLDEN
            .iter()
            .find(|(k, _)| k == key)
            .unwrap_or_else(|| panic!("no golden entry for {key}"))
            .1;
        assert_eq!(got.len(), want.len(), "{key}: length differs");
        let mut max_diff = 0.0_f32;
        let mut first_bad: Option<(usize, f32, f32)> = None;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if g.to_bits() != *w {
                let wf = f32::from_bits(*w);
                max_diff = max_diff.max((g - wf).abs());
                first_bad.get_or_insert((i, *g, wf));
            }
        }
        if let Some((i, g, w)) = first_bad {
            panic!("{key}[{i}]: got {g:e}, want {w:e}, max abs diff {max_diff:e}");
        }
        seen += 1;
    }
    let expected = GOLDEN
        .iter()
        .filter(|(k, _)| k.starts_with(&format!("{case}/")))
        .count();
    assert_eq!(seen, expected, "{case}: entry count differs from GOLDEN");
}

#[test]
fn golden_g1() {
    check("g1", case_g1());
}
#[test]
fn golden_g1x() {
    check("g1x", case_g1x());
}
#[test]
fn golden_g2() {
    check("g2", case_g2());
}
#[test]
fn golden_g3() {
    check("g3", case_g3());
}
#[test]
fn golden_g4() {
    check("g4", case_g4());
}
#[test]
fn golden_g5() {
    check("g5", case_g5());
}
#[test]
fn golden_g6() {
    check("g6", case_g6());
}
#[test]
fn golden_f1() {
    check("f1", case_f1());
}
#[test]
fn golden_f2() {
    check("f2", case_f2());
}
#[test]
fn golden_f3() {
    check("f3", case_f3());
}
#[test]
fn golden_f4() {
    check("f4", case_f4());
}

// Captured at commit 90cb7a7 (`git rev-parse --short HEAD`).
const GOLDEN: &[(&str, &[u32])] = &[
    (
        "g1/q_next",
        &[0x3a83126f, 0x3a83126f, 0x41dcbea7, 0x4277b150],
    ),
    ("g1/n", &[0x442d1826, 0x4438b009, 0x444327cd, 0x43b60ea2]),
    (
        "g1/q_spatial",
        &[0x41acfd40, 0x41b9bae0, 0x41c49ca6, 0x41373528],
    ),
    (
        "g1/p_spatial",
        &[0x3ed6bd2a, 0x3edd181a, 0x3ee1f6e1, 0x3e4c42bc],
    ),
    ("g1/q_t", &[0x3ec3df72, 0x3efaed24, 0xbec70219, 0xbf29c23f]),
    (
        "g1/q_prime_t",
        &[0x3fee4f7a, 0x40166b6c, 0x403bc390, 0x3fd08605],
    ),
    (
        "g1x/q_next",
        &[0x41e4a756, 0x42e470e7, 0x4306ecd5, 0x431c6d26],
    ),
    ("g1x/n", &[0x4468dc50, 0x44131dca, 0x42d8ecff, 0x429c73aa]),
    (
        "g1x/q_spatial",
        &[0x41c4f8d9, 0x41bb2100, 0x405e6c88, 0x40263f9c],
    ),
    (
        "g1x/p_spatial",
        &[0x3eb23c6c, 0x3f0fc8a6, 0x3d82bfa2, 0x3d45bd44],
    ),
    ("g1x/q_t", &[0x3f710c8e, 0x3f6ced30, 0x3f7861cd, 0x3e16dbdb]),
    (
        "g1x/q_prime_t",
        &[0x3f490809, 0x3f53c3cf, 0x3f5cc978, 0x3f583ece],
    ),
    (
        "g2/q_next",
        &[0x411db46b, 0x42c6048b, 0x430205d3, 0x4319aea9],
    ),
    ("g2/n", &[0x446f456a, 0x44538770, 0x436fc01e, 0x42d23234]),
    (
        "g2/q_spatial",
        &[0x420cf85b, 0x42182131, 0x4129b536, 0x409436d9],
    ),
    (
        "g2/p_spatial",
        &[0x3f4e54d7, 0x3f84bb14, 0x3e8a67b5, 0x3de71f7e],
    ),
    ("g2/q_t", &[0x3f624867, 0x3f5441f6, 0x3f44bfa5, 0xbdd1109d]),
    (
        "g2/q_prime_t",
        &[0x3f960200, 0x3f9faaf2, 0x3fa2dbce, 0x3f8af256],
    ),
    (
        "g3/q_next",
        &[0x418abcf0, 0x42d402a3, 0x4304b017, 0x431b1414],
    ),
    ("g3/n", &[0x446dd77e, 0x443adb3a, 0x43335050, 0x42b6a621]),
    (
        "g3/q_spatial",
        &[0x41f8e9e8, 0x41fd9e30, 0x40e63708, 0x406807ad],
    ),
    (
        "g3/p_spatial",
        &[0x3f1ca54d, 0x3f509d23, 0x3e2ad22d, 0x3da3f52d],
    ),
    ("g3/q_t", &[0x3f68d376, 0x3f61762a, 0x3f5d499f, 0xbb2f7b02]),
    (
        "g3/q_prime_t",
        &[0x3f7f9455, 0x3f87c736, 0x3f8c80a3, 0x3f7d90f2],
    ),
    (
        "g3/gamma",
        &[0xc2648dfa, 0xc2369908, 0xc1377faf, 0xc0c1429e],
    ),
    (
        "g4/q_next",
        &[0x41fd44f2, 0x42e29d32, 0x4306e921, 0x431c6d01],
    ),
    ("g4/n", &[0x44a9e018, 0x43070b4b, 0x42e5ea9a, 0x429ca5e1]),
    (
        "g4/q_spatial",
        &[0x4229c5b0, 0x4087ce69, 0x4070761e, 0x4026870a],
    ),
    (
        "g4/p_spatial",
        &[0x3f52bed8, 0x3da1aa28, 0x3d90db50, 0x3d462a46],
    ),
    ("g4/q_t", &[0x3f585bc7, 0x3f83d19d, 0x3f7816bd, 0x3e16d7db]),
    (
        "g4/q_prime_t",
        &[0x3f432f92, 0x3f50c725, 0x3f5cc978, 0x3f583ece],
    ),
    (
        "g5/q_next",
        &[0x3a83126f, 0x3a83126f, 0x41656734, 0x42440498],
    ),
    ("g5/n", &[0x44004f62, 0x444873b6, 0x4452cd5a, 0x43c5dbec]),
    (
        "g5/q_spatial",
        &[0x41252ac7, 0x4184bd68, 0x418e5eb0, 0x41077d4c],
    ),
    (
        "g5/p_spatial",
        &[0x3ead66fa, 0x3f04136c, 0x3f07aef5, 0x3e792d11],
    ),
    ("g5/q_t", &[0x3ed830a2, 0x3ef7ea54, 0xbf081176, 0xbf4bd548]),
    (
        "g5/q_prime_t",
        &[0x4006ced4, 0x4026f6e1, 0x404c9802, 0x3fe10e22],
    ),
    (
        "g6/q_next",
        &[0x41e4636a, 0x42e45ddd, 0x4306e204, 0x431c6110],
    ),
    ("g6/n", &[0x4468b3a9, 0x4412fd76, 0x42d7c7c1, 0x429b4272]),
    (
        "g6/q_spatial",
        &[0x41c3f97a, 0x41ba0a21, 0x40547ed7, 0x401bc301],
    ),
    (
        "g6/p_spatial",
        &[0x3eb27816, 0x3f0ffb89, 0x3d848d3d, 0x3d4992e9],
    ),
    ("g6/q_t", &[0x3f70fdfc, 0x3f6cdbf7, 0x3f785107, 0x3e169d90]),
    (
        "g6/q_prime_t",
        &[0x3f490809, 0x3f53c3cf, 0x3f5cc978, 0x3f583ece],
    ),
    ("g6/k_d", &[0xc78034c9, 0xc794feab, 0xc7a8f77f, 0xc7b22036]),
    ("g6/d_gw", &[0x3c47ad3a, 0x3c54759b, 0x3c5f9ef8, 0x3c5cfe14]),
    (
        "g6/leakance_factor",
        &[0xbd866f17, 0xbd9c3b7d, 0xbdb12cac, 0xbdbac74a],
    ),
    (
        "f1/depth",
        &[0x41214ff5, 0x3c23d70a, 0x4022cc1e, 0x3c23d70a],
    ),
    (
        "f1/depth.d_n",
        &[0x43248aa9, 0x00000000, 0x41faf8e0, 0x00000000],
    ),
    (
        "f1/depth.d_p",
        &[0xbe8c68ac, 0x80000000, 0xbd5629b4, 0x80000000],
    ),
    (
        "f1/depth.d_q",
        &[0xc0ffd5c4, 0x00000000, 0xbeb82600, 0x00000000],
    ),
    (
        "f1/depth.d_Q",
        &[0x3bbcb5a3, 0x00000000, 0x3c33e598, 0x00000000],
    ),
    (
        "f1/depth.d_s",
        &[0xc533f7a9, 0x80000000, 0xc409401a, 0x80000000],
    ),
    (
        "f1/top_width",
        &[0x41cbfd5f, 0x40550236, 0x421a1c66, 0x3e580814],
    ),
    (
        "f1/top_width.d_n",
        &[0x420bd3b5, 0x00000000, 0x439a6d8a, 0x00000000],
    ),
    (
        "f1/top_width.d_p",
        &[0x3f93f69b, 0x3e224ad4, 0x3fa8f24d, 0x3c249871],
    ),
    (
        "f1/top_width.d_q",
        &[0x4264e41a, 0xc1753c3a, 0x4201b7c0, 0xbf78b72d],
    ),
    (
        "f1/top_width.d_Q",
        &[0x3aa05d62, 0x00000000, 0x3ddd6348, 0x00000000],
    ),
    (
        "f1/top_width.d_s",
        &[0xc418ef8e, 0x80000000, 0xc5a8e7cd, 0x80000000],
    ),
    (
        "f1/bottom_width",
        &[0x4176aac9, 0x40150236, 0x4157c134, 0x3c23d70a],
    ),
    (
        "f1/bottom_width.d_n",
        &[0xc30195bc, 0x00000000, 0x42d832cd, 0x00000000],
    ),
    (
        "f1/bottom_width.d_p",
        &[0x3fb710c6, 0x3e224ad4, 0x3eec8640, 0x00000000],
    ),
    (
        "f1/bottom_width.d_q",
        &[0x42826f69, 0xc1753c3a, 0xc1d96b71, 0x00000000],
    ),
    (
        "f1/bottom_width.d_Q",
        &[0xbb949e4a, 0x00000000, 0x3d1af896, 0x00000000],
    ),
    (
        "f1/bottom_width.d_s",
        &[0x450dbbc5, 0x80000000, 0xc4ec778d, 0x80000000],
    ),
    (
        "f1/side_slope",
        &[0x3f000000, 0x42480000, 0x409d8588, 0x41289b27],
    ),
    (
        "f1/side_slope.d_n",
        &[0x00000000, 0x00000000, 0xc1a9fca1, 0x00000000],
    ),
    (
        "f1/side_slope.d_p",
        &[0x00000000, 0x00000000, 0x3e8a25fd, 0x3f007635],
    ),
    (
        "f1/side_slope.d_q",
        &[0x00000000, 0x00000000, 0x414699d3, 0xc217ebd5],
    ),
    (
        "f1/side_slope.d_Q",
        &[0x00000000, 0x00000000, 0xbbf3b171, 0x00000000],
    ),
    (
        "f1/side_slope.d_s",
        &[0x80000000, 0x80000000, 0x43b9ec4e, 0x80000000],
    ),
    (
        "f1/cross_sectional_area",
        &[0x434e4149, 0x3ce7b0e9, 0x42844dec, 0x3a90d055],
    ),
    (
        "f1/cross_sectional_area.d_n",
        &[0x453491fa, 0x00000000, 0x44a8451f, 0x00000000],
    ),
    (
        "f1/cross_sectional_area.d_p",
        &[0x40eda6f8, 0x3acfbbf0, 0x3f6810ac, 0x3852aea5],
    ),
    (
        "f1/cross_sectional_area.d_q",
        &[0x43e2d55d, 0xbe1cf358, 0xc02b1ed0, 0xbb9f2d8d],
    ),
    (
        "f1/cross_sectional_area.d_Q",
        &[0x3dcf17a5, 0x00000000, 0x3ef13b5c, 0x00000000],
    ),
    (
        "f1/cross_sectional_area.d_s",
        &[0xc7457fa7, 0x80000000, 0xc6b80b98, 0x80000000],
    ),
    (
        "f1/wetted_perimeter",
        &[0x4217d7d7, 0x4055057c, 0x421c2832, 0x3e63067b],
    ),
    (
        "f1/wetted_perimeter.d_n",
        &[0x436e5766, 0x00000000, 0x439eaa01, 0x00000000],
    ),
    (
        "f1/wetted_perimeter.d_p",
        &[0x3f512630, 0x3e224ad4, 0x3fa41568, 0x3c23b216],
    ),
    (
        "f1/wetted_perimeter.d_q",
        &[0x423d5cc2, 0xc1753c3a, 0x41f8beab, 0xbf4196e8],
    ),
    (
        "f1/wetted_perimeter.d_Q",
        &[0x3c08acac, 0x00000000, 0x3de375f8, 0x00000000],
    ),
    (
        "f1/wetted_perimeter.d_s",
        &[0xc58257cb, 0x80000000, 0xc5ad89ef, 0x80000000],
    ),
    (
        "f1/hydraulic_radius",
        &[0x40adde2e, 0x3c0b37db, 0x3fd8e586, 0x3ba34bc8],
    ),
    (
        "f1/hydraulic_radius.d_n",
        &[0x4227f9df, 0x00000000, 0x41a5ab17, 0x00000000],
    ),
    (
        "f1/hydraulic_radius.d_p",
        &[0x3da12f09, 0x38961118, 0xbd04cbbf, 0x36055406],
    ),
    (
        "f1/hydraulic_radius.d_q",
        &[0x40a599c9, 0xbbe2c2fa, 0xbfb583f6, 0xbba0fd42],
    ),
    (
        "f1/hydraulic_radius.d_Q",
        &[0x3ac0a5fd, 0x00000000, 0x3bed808a, 0x00000000],
    ),
    (
        "f1/hydraulic_radius.d_s",
        &[0xc437b94c, 0x80000000, 0xc3b5331e, 0x80000000],
    ),
    (
        "f1/velocity",
        &[0x4032b6c0, 0x3d1a1a33, 0x3fa46022, 0x3cd7f163],
    ),
    (
        "f1/velocity.d_n",
        &[0xc282ca26, 0xbf899764, 0xc1d1d38a, 0xbf40ce62],
    ),
    (
        "f1/velocity.d_p",
        &[0x3cdce6d2, 0x395d7b2b, 0xbc862fa8, 0x36eb1582],
    ),
    (
        "f1/velocity.d_q",
        &[0x3fe2f480, 0xbca7563f, 0xbf376a73, 0xbc8deda9],
    ),
    (
        "f1/velocity.d_Q",
        &[0x3a04030a, 0x00000000, 0x3b6ffd14, 0x00000000],
    ),
    (
        "f1/velocity.d_s",
        &[0x448f0d19, 0x41967d95, 0x43e57f5e, 0x4152e1ba],
    ),
    (
        "f2/depth",
        &[0x40db8789, 0x3c23d70a, 0x401005b5, 0x3c2abc73],
    ),
    (
        "f2/depth.d_n",
        &[0x42ba9d85, 0x00000000, 0x41c0e30e, 0x3dca338d],
    ),
    (
        "f2/depth.d_p",
        &[0xbe1f3ebc, 0x80000000, 0xbd2498cb, 0xb92c8b90],
    ),
    (
        "f2/depth.d_q",
        &[0xc051afd2, 0x00000000, 0xbe312b30, 0x3c8f5ba9],
    ),
    (
        "f2/depth.d_Q",
        &[0x3b56067f, 0x00000000, 0x3c0a42e8, 0x43acc78d],
    ),
    (
        "f2/depth.d_s",
        &[0xc4cc1c47, 0x80000000, 0xc3d2f856, 0xbfdd2860],
    ),
    (
        "f2/top_width",
        &[0x41c57f82, 0x40550236, 0x420e4fe9, 0x3e611d7e],
    ),
    (
        "f2/top_width.d_n",
        &[0x41e1a49d, 0x00000000, 0x4377c6a2, 0x40052af8],
    ),
    (
        "f2/top_width.d_p",
        &[0x3f907524, 0x3e224ad4, 0x3fa3ff6c, 0x3be56568],
    ),
    (
        "f2/top_width.d_q",
        &[0x423a3457, 0xc1753c3a, 0x41d8a166, 0xbf22705e],
    ),
    (
        "f2/top_width.d_Q",
        &[0x3a81648a, 0x00000000, 0x3db19b14, 0x45e394ef],
    ),
    (
        "f2/top_width.d_s",
        &[0xc3f6cc0a, 0x80000000, 0xc587809e, 0xc211a6fe],
    ),
    (
        "f2/bottom_width",
        &[0x418e9da0, 0x40150236, 0x41473c88, 0x3c23d70a],
    ),
    (
        "f2/bottom_width.d_n",
        &[0xc282345e, 0x00000000, 0x42ad714f, 0x00000000],
    ),
    (
        "f2/bottom_width.d_p",
        &[0x3fa45cfc, 0x3e224ad4, 0x3ee598a2, 0x00000000],
    ),
    (
        "f2/bottom_width.d_q",
        &[0x42474f54, 0xc1753c3a, 0xc1d0cdc8, 0x00000000],
    ),
    (
        "f2/bottom_width.d_Q",
        &[0xbb15543a, 0x00000000, 0x3cf8a5b8, 0x00000000],
    ),
    (
        "f2/bottom_width.d_s",
        &[0x448e6945, 0x80000000, 0xc4bdb3ec, 0x80000000],
    ),
    (
        "f2/side_slope",
        &[0x3f000000, 0x42480000, 0x40a46ca5, 0x41289960],
    ),
    (
        "f2/side_slope.d_n",
        &[0x00000000, 0x00000000, 0xc19a25bc, 0xbdcc432b],
    ),
    (
        "f2/side_slope.d_p",
        &[0x00000000, 0x00000000, 0x3e8db7d6, 0x3f007fc0],
    ),
    (
        "f2/side_slope.d_q",
        &[0x00000000, 0x00000000, 0x41435f9c, 0xc2163f77],
    ),
    (
        "f2/side_slope.d_Q",
        &[0x00000000, 0x00000000, 0xbbdcfc58, 0xc3ae8a65],
    ),
    (
        "f2/side_slope.d_s",
        &[0x80000000, 0x80000000, 0x43a89944, 0x3fdf6975],
    ),
    (
        "f2/cross_sectional_area",
        &[0x4311d47f, 0x3ce7b0e9, 0x42582b7d, 0x3a9cf7a3],
    ),
    (
        "f2/cross_sectional_area.d_n",
        &[0x44e81c78, 0x00000000, 0x446ed934, 0x3cb5c323],
    ),
    (
        "f2/cross_sectional_area.d_p",
        &[0x409f0a89, 0x3acfbbf0, 0x3f7b2b32, 0x37935b00],
    ),
    (
        "f2/cross_sectional_area.d_q",
        &[0x43827932, 0xbe1cf358, 0xc0437898, 0xbaa9c0a5],
    ),
    (
        "f2/cross_sectional_area.d_Q",
        &[0x3d851a13, 0x00000000, 0x3eab34e2, 0x429b507f],
    ),
    (
        "f2/cross_sectional_area.d_s",
        &[0xc6fddf21, 0x80000000, 0xc6829ec7, 0xbec6cd6e],
    ),
    (
        "f2/wetted_perimeter",
        &[0x4204ab10, 0x4055057c, 0x42100c36, 0x3e6c2407],
    ),
    (
        "f2/wetted_perimeter.d_n",
        &[0x430f8a3c, 0x00000000, 0x437e0595, 0x4005a23d],
    ),
    (
        "f2/wetted_perimeter.d_p",
        &[0x3f6fb4ab, 0x3e224ad4, 0x3fa011c0, 0x3be31f84],
    ),
    (
        "f2/wetted_perimeter.d_q",
        &[0x422a0158, 0xc1753c3a, 0x41d0443d, 0xbed15ac2],
    ),
    (
        "f2/wetted_perimeter.d_Q",
        &[0x3ba49f86, 0x00000000, 0x3db61535, 0x45e460c3],
    ),
    (
        "f2/wetted_perimeter.d_s",
        &[0xc51cff32, 0x80000000, 0xc58aeb0c, 0xc2122971],
    ),
    (
        "f2/hydraulic_radius",
        &[0x408cb2df, 0x3c0b37db, 0x3fc0165e, 0x3baa2b0d],
    ),
    (
        "f2/hydraulic_radius.d_n",
        &[0x4213d45f, 0x00000000, 0x417f1c2f, 0x3d497f77],
    ),
    (
        "f2/hydraulic_radius.d_p",
        &[0x3cd2b2c8, 0x38961118, 0xbccbb7a1, 0xb8a7977c],
    ),
    (
        "f2/hydraulic_radius.d_q",
        &[0x400ef02c, 0xbbe2c2fa, 0xbf95b7ac, 0x3b6b6791],
    ),
    (
        "f2/hydraulic_radius.d_Q",
        &[0x3aa98af4, 0x00000000, 0x3bb6dce8, 0x432c2dac],
    ),
    (
        "f2/hydraulic_radius.d_s",
        &[0xc421b047, 0x80000000, 0xc38b8369, 0xbf5c636a],
    ),
    (
        "f2/velocity",
        &[0x4098406c, 0x3bf5facc, 0x3fc95a08, 0x3bb3b8d1],
    ),
    (
        "f2/velocity.d_n",
        &[0xc2ad41b8, 0xbe5b9fec, 0xc1df3a12, 0xbdd4bf60],
    ),
    (
        "f2/velocity.d_p",
        &[0xbc9d3d2c, 0x3830c3e0, 0xbcdee6fa, 0xb8b591de],
    ),
    (
        "f2/velocity.d_q",
        &[0x3f50def6, 0xbb858d69, 0xbf5c1668, 0x3bbc8179],
    ),
    (
        "f2/velocity.d_Q",
        &[0x3ae236a8, 0x00000000, 0x3bc3713e, 0x4338e2b2],
    ),
    (
        "f2/velocity.d_s",
        &[0x44bd7fde, 0x407036ea, 0x43f42781, 0x3fe8b152],
    ),
    (
        "f3/top_width",
        &[0x41ca0db9, 0x40550236, 0x4218626c, 0x3e580814],
    ),
    (
        "f3/top_width.d_p",
        &[0x3f99f213, 0x3e224ad4, 0x3fe83474, 0x3c249871],
    ),
    (
        "f3/top_width.d_q",
        &[0x425dfa85, 0xc1753c3a, 0x420ba0e4, 0xbf78b72d],
    ),
    (
        "f3/top_width.d_depth",
        &[0x3e7163bc, 0x43052178, 0x411e7ae6, 0x41a89b26],
    ),
    (
        "f3/bottom_width",
        &[0x41820db9, 0x40150236, 0x41555670, 0x3c23d70a],
    ),
    (
        "f3/bottom_width.d_p",
        &[0x3f99f213, 0x3e224ad4, 0x3f228b00, 0x00000000],
    ),
    (
        "f3/bottom_width.d_q",
        &[0x425dfa85, 0xc1753c3a, 0xc1cf077e, 0x80000000],
    ),
    (
        "f3/bottom_width.d_depth",
        &[0xbf43a711, 0x420485e0, 0x405ddf1e, 0x00000000],
    ),
    (
        "f3/side_slope",
        &[0x3f000000, 0x42480000, 0x409e7ae6, 0x41289b27],
    ),
    (
        "f3/side_slope.d_p",
        &[0x00000000, 0x00000000, 0x3e717e54, 0x3f007635],
    ),
    (
        "f3/side_slope.d_q",
        &[0x00000000, 0x00000000, 0x414283b6, 0xc217ebd5],
    ),
    (
        "f3/side_slope.d_depth",
        &[0x00000000, 0x00000000, 0xbf317f46, 0xbf86c400],
    ),
    ("f3/area", &[0x433acf70, 0x3ce7b0e9, 0x42809305, 0x3a90d055]),
    (
        "f3/area.d_p",
        &[0x412d3055, 0x3acfbbf0, 0x4043ec39, 0x3852aea5],
    ),
    (
        "f3/area.d_q",
        &[0x43f9b9d5, 0xbe1cf358, 0x413491b4, 0xbb9f2d8d],
    ),
    (
        "f3/area.d_depth",
        &[0x419306bc, 0x406a365a, 0x4229b7da, 0x3e5d0b2c],
    ),
    (
        "f3/wetted_perimeter",
        &[0x42118677, 0x4055057c, 0x421a622a, 0x3e63067b],
    ),
    (
        "f3/wetted_perimeter.d_p",
        &[0x3f99f213, 0x3e224ad4, 0x3fe53813, 0x3c23b216],
    ),
    (
        "f3/wetted_perimeter.d_q",
        &[0x425dfa85, 0xc1753c3a, 0x4206d188, 0xbf4196e8],
    ),
    (
        "f3/wetted_perimeter.d_depth",
        &[0x3fbc63f2, 0x43052696, 0x4122c699, 0x41a93221],
    ),
    (
        "f3/hydraulic_radius",
        &[0x40a4502d, 0x3c0b37db, 0x3fd533fa, 0x3ba34bc8],
    ),
    (
        "f3/hydraulic_radius.d_p",
        &[0x3e02d7f4, 0x38961118, 0x3b0547f2, 0x36055406],
    ),
    (
        "f3/hydraulic_radius.d_q",
        &[0x40bcaac6, 0xbbe2c2fa, 0xbf94c196, 0xbba0fd42],
    ),
    (
        "f3/hydraulic_radius.d_depth",
        &[0x3e9848ce, 0x3f427265, 0x3f290821, 0x3eff1c0d],
    ),
    (
        "f4/area_d",
        &[0x419306bc, 0x406a365a, 0x4229b7da, 0x3e5d0b2d],
    ),
    (
        "f4/area_d.d_p",
        &[0x3fa6e08f, 0x3e63359a, 0x40014f21, 0x3c248365],
    ),
    (
        "f4/area_d.d_q",
        &[0x42aad36e, 0xc19109ea, 0x4204a746, 0xbf5d965e],
    ),
    (
        "f4/area_d.d_depth",
        &[0xbf3e9558, 0x42acc42e, 0x413081d7, 0x41a88598],
    ),
    (
        "f4/per_d",
        &[0x3fbc63f2, 0x43052696, 0x4122c699, 0x41a93224],
    ),
    (
        "f4/per_d.d_p",
        &[0x3c37ea76, 0x40caddab, 0x3eeb2483, 0x3f7f8444],
    ),
    (
        "f4/per_d.d_q",
        &[0x4054c09e, 0xc38c21c2, 0x41bbd4df, 0xc25a40b2],
    ),
    (
        "f4/per_d.d_depth",
        &[0xbcc48b66, 0xc5f99ea7, 0xbfacd44e, 0xc006013e],
    ),
];
