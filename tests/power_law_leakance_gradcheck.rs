//! Gradient check of `TimestepLeakanceOp` on `geometry_form: power_law`.
//! WET copy of `tests/leakance_gradcheck.rs` (same parents, same tolerances,
//! same 4-reach chain) with the power-law section. Zeta reads the shared
//! power-law depth and `q_eps`, so its depth, `q_eps` and `p_spatial`
//! gradients must join `power_law.rs::section_backward`.
//!
//! The sweep covers {n, q_spatial, p_spatial, q_t, q_prime_t, K_D, d_gw,
//! leakance_factor} on a LOSING base point (depth > d_gw), as the trapezoid
//! file does. `x_storage` is a constant of the op, not a parent.
//!
//! FD step: the rule of `tests/power_law_gradcheck.rs`. `h = max(h0, h_rt)`
//! with `h0 = 1e-3 * max(1, |x|)` and
//! `h_rt = 4 * eps_f32 * sum|q_next| / (REL_TOL * |g|)`, capped at
//! `0.1 * max(1, |x|)`. A slot where the cap binds is printed as `(capped)`.
//!
//! Exception: K_D, d_gw and leakance_factor. Zeta is exactly linear in each,
//! so a central difference has no truncation error at any step, and the step
//! must be large to lift the signal above the f32 round-off of the loss. The
//! rule's own steps do not work for them: `h0 = 1e-3` is 2000 times the base
//! K_D and drives the discharge through its floor, and the cap of 0.1 on d_gw
//! leaves the FD at the round-off floor (measured: abs error 2e-4 on a
//! gradient of 1.4e-2). These three keep the steps of
//! `tests/leakance_gradcheck.rs`: 4e-7, 1.5 and 0.4. The kink-free check
//! still guards every one of them.
//!
//! Every gradcheck first asserts that no clamp selection (depth floor,
//! velocity band, Cunge X band, losing switch `depth > d_gw`) changes between
//! `x - h` and `x + h`. A stencil that crosses a switch is refused, not passed.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::{Config, GeometryForm};
use ddrs::routing::mmc_op::{__spike_forward_chain_k1_outputs, timestep_forward_leakance};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 4;
const LEN: f32 = 10_000.0;
const EPS: f32 = 1e-3;
/// Largest FD step, as a fraction of `max(1, |x|)`.
const STEP_CAP: f32 = 0.1;
const REL_TOL: f32 = 5e-3;
const ABS_TOL: f32 = 1e-4;

// Interior leakance values (losing config: depth > d_gw, factor > 0, K_D > 0).
const K_D: f32 = 5e-7;
const D_GW: f32 = 0.0;
const LEAK_FAC: f32 = 0.5;

#[derive(Copy, Clone, Debug, PartialEq)]
enum Parent {
    N,
    QSpatial,
    PSpatial,
    QT,
    QPrimeT,
    KD,
    DGW,
    LeakFactor,
}

const ALL: [Parent; 8] = [
    Parent::N,
    Parent::QSpatial,
    Parent::PSpatial,
    Parent::QT,
    Parent::QPrimeT,
    Parent::KD,
    Parent::DGW,
    Parent::LeakFactor,
];

fn chain() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![LEN; N], vec![1e-3; N])
}

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
    cfg.params.geometry_form = GeometryForm::PowerLaw;
    cfg
}

#[derive(Clone)]
struct Inputs {
    n: Vec<f32>,
    qsp: Vec<f32>,
    psp: Vec<f32>,
    qt: Vec<f32>,
    qpt: Vec<f32>,
    kd: Vec<f32>,
    dgw: Vec<f32>,
    fac: Vec<f32>,
}

fn base_inputs(dgw: Vec<f32>) -> Inputs {
    Inputs {
        n: vec![0.035; N],
        qsp: vec![0.4; N],
        psp: vec![20.0; N],
        qt: vec![100.0, 120.0, 140.0, 160.0],
        qpt: vec![10.0, 12.0, 14.0, 16.0],
        kd: vec![K_D; N],
        dgw,
        fac: vec![LEAK_FAC; N],
    }
}

/// d_gw = 100 m, far above any routing depth: zeta is zero everywhere.
fn dgw_gaining() -> Vec<f32> {
    vec![100.0; N]
}

/// Reaches 0 and 1 gaining, reaches 2 and 3 losing.
fn dgw_mixed() -> Vec<f32> {
    vec![100.0, 100.0, 0.0, 0.0]
}

fn device() -> <I as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

fn field(inp: &Inputs, p: Parent) -> &Vec<f32> {
    match p {
        Parent::N => &inp.n,
        Parent::QSpatial => &inp.qsp,
        Parent::PSpatial => &inp.psp,
        Parent::QT => &inp.qt,
        Parent::QPrimeT => &inp.qpt,
        Parent::KD => &inp.kd,
        Parent::DGW => &inp.dgw,
        Parent::LeakFactor => &inp.fac,
    }
}

fn field_mut(inp: &mut Inputs, p: Parent) -> &mut Vec<f32> {
    match p {
        Parent::N => &mut inp.n,
        Parent::QSpatial => &mut inp.qsp,
        Parent::PSpatial => &mut inp.psp,
        Parent::QT => &mut inp.qt,
        Parent::QPrimeT => &mut inp.qpt,
        Parent::KD => &mut inp.kd,
        Parent::DGW => &mut inp.dgw,
        Parent::LeakFactor => &mut inp.fac,
    }
}

/// Forward through `timestep_forward_leakance`. Parents in `tracked` require
/// grad. Returns `q_next` and one gradient per entry of `tracked` when
/// `backward` is set.
fn run(
    cfg: &Config,
    inp: &Inputs,
    tracked: &[Parent],
    mask: Option<Vec<f32>>,
    backward: bool,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let adj = chain();
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let mk = |data: &[f32], p: Parent| -> Tensor<AB, 1> {
        let t: Tensor<AB, 1> = Tensor::from_floats(data, &dev);
        if tracked.contains(&p) {
            t.require_grad()
        } else {
            t
        }
    };
    let n = mk(&inp.n, Parent::N);
    let qsp = mk(&inp.qsp, Parent::QSpatial);
    let psp = mk(&inp.psp, Parent::PSpatial);
    let qt = mk(&inp.qt, Parent::QT);
    let qpt = mk(&inp.qpt, Parent::QPrimeT);
    let kd = mk(&inp.kd, Parent::KD);
    let dgw = mk(&inp.dgw, Parent::DGW);
    let fac = mk(&inp.fac, Parent::LeakFactor);
    let mask_t: Option<Tensor<I, 1>> = mask.map(|m| Tensor::from_floats(m.as_slice(), &dev));
    let q_next = timestep_forward_leakance::<I>(
        cfg,
        &pattern,
        &assembler,
        n.clone(),
        qsp.clone(),
        psp.clone(),
        qt.clone(),
        qpt.clone(),
        Tensor::from_floats(adj.length_m.as_slice(), &dev),
        Tensor::from_floats(adj.slope.as_slice(), &dev),
        Tensor::from_floats(vec![0.3f32; N].as_slice(), &dev),
        kd.clone(),
        dgw.clone(),
        fac.clone(),
        mask_t,
        None,
        false,
        None, // skip_rows
        None, // gamma
    );
    let q_host = q_next.clone().into_data().to_vec::<f32>().unwrap();
    if !backward {
        return (q_host, vec![]);
    }
    let grads = q_next.sum().backward();
    let gs = tracked
        .iter()
        .map(|p| {
            let g = match p {
                Parent::N => n.grad(&grads),
                Parent::QSpatial => qsp.grad(&grads),
                Parent::PSpatial => psp.grad(&grads),
                Parent::QT => qt.grad(&grads),
                Parent::QPrimeT => qpt.grad(&grads),
                Parent::KD => kd.grad(&grads),
                Parent::DGW => dgw.grad(&grads),
                Parent::LeakFactor => fac.grad(&grads),
            };
            g.unwrap_or_else(|| panic!("no grad on {p:?}"))
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        })
        .collect();
    (q_host, gs)
}

fn analytical(cfg: &Config, inp: &Inputs, parent: Parent) -> Vec<f32> {
    run(cfg, inp, &[parent], None, true).1.remove(0)
}

fn loss_value(cfg: &Config, inp: &Inputs) -> f32 {
    run(cfg, inp, &[], None, false).0.iter().sum()
}

/// FD step for one slot (see the module doc).
fn step_of(parent: Parent, x: f32, g: f32, q_abs_sum: f32) -> (f32, bool) {
    match parent {
        Parent::KD => return (4e-7, false),
        Parent::DGW => return (1.5, false),
        Parent::LeakFactor => return (0.4, false),
        _ => {}
    }
    let scale = x.abs().max(1.0);
    let h0 = EPS * scale;
    // An exactly zero analytical gradient has nothing to resolve: use h0.
    if g == 0.0 {
        return (h0, false);
    }
    let h_rt = 4.0 * f32::EPSILON * q_abs_sum / (REL_TOL * g.abs().max(1e-12));
    let cap = STEP_CAP * scale;
    let h = h0.max(h_rt);
    if h > cap {
        (cap, true)
    } else {
        (h, false)
    }
}

fn fd(cfg: &Config, inp: &Inputs, parent: Parent, steps: &[f32]) -> Vec<f32> {
    let base = field(inp, parent).clone();
    (0..N)
        .map(|i| {
            let eps = steps[i];
            let mut plus = inp.clone();
            field_mut(&mut plus, parent)[i] = base[i] + eps;
            let mut minus = inp.clone();
            field_mut(&mut minus, parent)[i] = base[i] - eps;
            (loss_value(cfg, &plus) - loss_value(cfg, &minus)) / (2.0 * eps)
        })
        .collect()
}

/// Clamp selections per reach, read from the saved forward state: depth
/// floor, velocity band, Cunge X band and the losing switch `depth > d_gw`.
/// Two stencil ends with equal codes are on the same smooth branch.
fn selections(cfg: &Config, inp: &Inputs) -> Vec<[u8; 4]> {
    let adj = chain();
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let t = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &dev) };
    let k = __spike_forward_chain_k1_outputs::<I>(
        cfg,
        &pattern,
        t(&inp.n),
        t(&inp.qsp),
        t(&inp.psp),
        t(&inp.qt),
        t(&inp.qpt),
        t(&adj.length_m),
        t(&adj.slope),
        t(&[0.3; N]),
    );
    let depth_lb = cfg.params.attribute_minimums.depth;
    (0..N)
        .map(|i| {
            let (depth, top, v_un, v_cl, c) = (k[0][i], k[1][i], k[5][i], k[6][i], k[7][i]);
            let depth_floor = (depth <= depth_lb) as u8;
            let v_band = if v_cl > v_un { 1 } else if v_cl < v_un { 2 } else { 0 };
            let w = inp.qt[i] / (top * 1e-3 * c * LEN + 1e-12);
            let x_raw = (-w + 1.0) * 0.5;
            let x_band = if x_raw < 0.0 { 1 } else if x_raw > 0.5 { 2 } else { 0 };
            let losing = (depth > inp.dgw[i]) as u8;
            [depth_floor, v_band, x_band, losing]
        })
        .collect()
}

fn assert_kink_free(cfg: &Config, inp: &Inputs, parent: Parent, steps: &[f32]) {
    let base = field(inp, parent).clone();
    for i in 0..N {
        let h = steps[i];
        let mut plus = inp.clone();
        field_mut(&mut plus, parent)[i] = base[i] + h;
        let mut minus = inp.clone();
        field_mut(&mut minus, parent)[i] = base[i] - h;
        let (sp, sm, s0) = (selections(cfg, &plus), selections(cfg, &minus), selections(cfg, inp));
        assert!(
            sp == sm && sp == s0,
            "{parent:?}[{i}]: a clamp selection changes inside the FD stencil (minus {sm:?}, base {s0:?}, plus {sp:?})"
        );
    }
}

fn slot_ok(a: f32, f: f32) -> bool {
    let abs = (a - f).abs();
    let rel = abs / a.abs().max(f.abs()).max(1e-12);
    rel < REL_TOL || abs < ABS_TOL
}

/// Returns `(worst rel, worst abs)` and asserts every slot is in tolerance.
fn check(label: &str, inp: &Inputs, parent: Parent) -> (f32, f32) {
    let cfg = mock_cfg();
    let a = analytical(&cfg, inp, parent);
    let q_abs_sum: f32 = run(&cfg, inp, &[], None, false).0.iter().map(|q| q.abs()).sum();
    let picked: Vec<(f32, bool)> = (0..N)
        .map(|i| step_of(parent, field(inp, parent)[i], a[i], q_abs_sum))
        .collect();
    let steps: Vec<f32> = picked.iter().map(|p| p.0).collect();
    assert_kink_free(&cfg, inp, parent, &steps);
    let f = fd(&cfg, inp, parent, &steps);
    let rel = |x: f32, y: f32| (x - y).abs() / x.abs().max(y.abs()).max(1e-12);
    let (mut worst_rel, mut worst_abs, mut ok) = (0.0f32, 0.0f32, true);
    for i in 0..N {
        println!(
            "  {label} {parent:?}[{i}] h={:.3e}{} analytical={:.6e} fd={:.6e} rel={:.3e} abs={:.3e}",
            steps[i],
            if picked[i].1 { " (capped)" } else { "" },
            a[i],
            f[i],
            rel(a[i], f[i]),
            (a[i] - f[i]).abs()
        );
        worst_rel = worst_rel.max(rel(a[i], f[i]));
        worst_abs = worst_abs.max((a[i] - f[i]).abs());
        ok &= slot_ok(a[i], f[i]);
    }
    println!("{label} {parent:?}: worst rel {worst_rel:.3e} worst abs {worst_abs:.3e}");
    assert!(ok, "{label} {parent:?}: gradcheck failed (worst rel {worst_rel:.3e})");
    (worst_rel, worst_abs)
}

#[test]
fn base_point_is_losing_and_x_is_interior() {
    let cfg = mock_cfg();
    let inp = base_inputs(vec![D_GW; N]);
    let sel = selections(&cfg, &inp);
    for (i, s) in sel.iter().enumerate() {
        assert_eq!(*s, [0, 0, 0, 1], "reach {i}: expected no clamp active and losing, got {s:?}");
    }
}

#[test]
fn gradcheck_n() {
    check("losing", &base_inputs(vec![D_GW; N]), Parent::N);
}
#[test]
fn gradcheck_q_spatial() {
    check("losing", &base_inputs(vec![D_GW; N]), Parent::QSpatial);
}
#[test]
fn gradcheck_p_spatial() {
    check("losing", &base_inputs(vec![D_GW; N]), Parent::PSpatial);
}
#[test]
fn gradcheck_q_t() {
    check("losing", &base_inputs(vec![D_GW; N]), Parent::QT);
}
#[test]
fn gradcheck_q_prime_t() {
    check("losing", &base_inputs(vec![D_GW; N]), Parent::QPrimeT);
}
#[test]
fn gradcheck_k_d() {
    check("losing", &base_inputs(vec![D_GW; N]), Parent::KD);
}
#[test]
fn gradcheck_d_gw() {
    check("losing", &base_inputs(vec![D_GW; N]), Parent::DGW);
}
#[test]
fn gradcheck_leakance_factor() {
    check("losing", &base_inputs(vec![D_GW; N]), Parent::LeakFactor);
}

/// zeta is zero when d_gw is far above the depth: the three leakance
/// gradients are exactly zero, analytically and by finite difference.
#[test]
fn gaining_chain_leakance_gradients_are_zero() {
    let cfg = mock_cfg();
    let inp = base_inputs(dgw_gaining());
    for p in [Parent::KD, Parent::DGW, Parent::LeakFactor] {
        let a = analytical(&cfg, &inp, p);
        let steps: Vec<f32> = (0..N).map(|i| step_of(p, field(&inp, p)[i], a[i], 0.0).0).collect();
        let f = fd(&cfg, &inp, p, &steps);
        for i in 0..N {
            assert!(a[i].abs() < 1e-9, "gaining: {p:?}[{i}] must be 0, got {:.3e}", a[i]);
            assert!(f[i].abs() < 1e-5, "gaining: {p:?}[{i}] FD must be 0, got {:.3e}", f[i]);
        }
    }
}

#[test]
fn gaining_chain_routing_params() {
    let inp = base_inputs(dgw_gaining());
    for p in [Parent::N, Parent::QSpatial, Parent::PSpatial, Parent::QT, Parent::QPrimeT] {
        check("gaining", &inp, p);
    }
}

/// Mixed chain: gradients are right per reach, and zero on the gaining reaches.
#[test]
fn mixed_chain_all_parents() {
    let inp = base_inputs(dgw_mixed());
    for p in ALL {
        check("mixed", &inp, p);
    }
    let cfg = mock_cfg();
    for p in [Parent::KD, Parent::DGW, Parent::LeakFactor] {
        let a = analytical(&cfg, &inp, p);
        for i in 0..2 {
            assert!(a[i].abs() < 1e-9, "mixed: {p:?}[{i}] (gaining) must be 0, got {:.3e}", a[i]);
        }
        for i in 2..N {
            assert!(a[i] != 0.0 && a[i].is_finite(), "mixed: {p:?}[{i}] (losing) is {}", a[i]);
        }
    }
}

/// Impervious reach 2: its three leakance gradients are exactly zero; the
/// others stay finite and non-zero.
#[test]
fn impervious_mask_reach2_zero() {
    let cfg = mock_cfg();
    let inp = base_inputs(vec![D_GW; N]);
    let mask = Some(vec![1.0f32, 1.0, 0.0, 1.0]);
    for p in [Parent::KD, Parent::DGW, Parent::LeakFactor] {
        let g = run(&cfg, &inp, &[p], mask.clone(), true).1.remove(0);
        assert!(g[2].abs() < 1e-9, "{p:?}[2] on the impervious reach is {:.3e}", g[2]);
        for i in [0usize, 1, 3] {
            assert!(g[i].is_finite() && g[i] != 0.0, "{p:?}[{i}] = {}", g[i]);
        }
    }
}

/// An all-ones mask is bitwise the no-mask gradient.
#[test]
fn all_ones_mask_equals_no_mask() {
    let cfg = mock_cfg();
    let inp = base_inputs(vec![D_GW; N]);
    for p in ALL {
        let none = run(&cfg, &inp, &[p], None, true).1.remove(0);
        let ones = run(&cfg, &inp, &[p], Some(vec![1.0; N]), true).1.remove(0);
        assert_eq!(none, ones, "{p:?}: all-ones mask differs from no mask");
    }
}

/// The depth floor is ACTIVE on a losing reach. Reach 0 has a tiny Q_t, a
/// small n and a large p, so its raw depth is well below
/// `attribute_minimums.depth`. The depth is then constant: dL/dn through the
/// depth is cut, including zeta's depth term, so dL/dn on reach 0 is exactly
/// 0. dL/dp and dL/dq_spatial are not cut (the width term and the section's
/// `p d^q` read them directly) and are checked against FD. This pins the
/// order in which zeta's depth gradient joins: before the depth-floor mask.
#[test]
fn floored_losing_reach_cuts_the_depth_gradient() {
    let cfg = mock_cfg();
    let mut inp = base_inputs(vec![D_GW; N]);
    inp.qt[0] = 1e-3;
    inp.n[0] = 0.01;
    inp.psp[0] = 150.0;

    // Raw depth of reach 0 from the forward's own formula.
    let q_eps = inp.qsp[0] + 1e-6;
    let ratio = inp.qt[0] * inp.n[0] * (q_eps + 1.0) / (inp.psp[0] * 1e-3_f32.sqrt() + 1e-8);
    let raw_depth = ratio.powf(3.0 / (3.0 * q_eps + 5.0));
    let floor = cfg.params.attribute_minimums.depth;
    println!("floored reach 0: raw depth {raw_depth:.4e} m, floor {floor:.4e} m");
    assert!(raw_depth < 0.5 * floor, "raw depth {raw_depth} is not clearly below the floor {floor}");

    let sel = selections(&cfg, &inp);
    assert_eq!(sel[0][0], 1, "depth floor must be active on reach 0: {:?}", sel[0]);
    assert_eq!(sel[0][3], 1, "reach 0 must be losing: {:?}", sel[0]);

    let gn = analytical(&cfg, &inp, Parent::N);
    assert_eq!(gn[0], 0.0, "dL/dn on the floored reach must be exactly 0, got {}", gn[0]);
    let gp = analytical(&cfg, &inp, Parent::PSpatial);
    assert!(gp[0] != 0.0 && gp[0].is_finite(), "dL/dp on the floored reach is {}", gp[0]);

    for p in [Parent::N, Parent::PSpatial, Parent::QSpatial] {
        check("floored", &inp, p);
    }
}
