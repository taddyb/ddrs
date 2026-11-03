//! Gradient check of the fused TimestepOp on `geometry_form: power_law`.
//! Same harness as `tests/sp8_gradcheck.rs`: analytical gradient against
//! central finite differences, per tracked parent, on the 4-reach chain with
//! `L = 10_000` m so the Cunge X is interior (B19's terms run).

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::{Config, GeometryForm, StageRoughnessSection};
use ddrs::routing::mmc_op::{__spike_forward_chain_k1_outputs, timestep_forward};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 4;
const LEN: f32 = 10_000.0;
const EPS: f32 = 1e-3;
const REL_TOL: f32 = 5e-3;
const ABS_TOL: f32 = 1e-4;
const GAMMA: f32 = 0.35;
const LEARNED_GAMMA: f32 = 0.2;

#[derive(Copy, Clone, Debug, PartialEq)]
enum Parent {
    N,
    QSpatial,
    PSpatial,
    QT,
    QPrimeT,
    Gamma,
}

const ALL: [Parent; 6] = [
    Parent::N,
    Parent::QSpatial,
    Parent::PSpatial,
    Parent::QT,
    Parent::QPrimeT,
    Parent::Gamma,
];

fn chain() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![LEN; N], vec![1e-3; N])
}

#[derive(Clone, Copy)]
enum Mode {
    Plain,
    GlobalGamma,
    LearnedGamma,
    Positivity,
}

fn mock_cfg(mode: Mode) -> Config {
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
    match mode {
        Mode::GlobalGamma => {
            cfg.params.stage_roughness = Some(StageRoughnessSection {
                gamma: GAMMA,
                d_ref: 1.0,
            });
        }
        Mode::Positivity => cfg.params.enforce_positivity = true,
        _ => {}
    }
    cfg
}

#[derive(Clone)]
struct Inputs {
    n: Vec<f32>,
    qsp: Vec<f32>,
    psp: Vec<f32>,
    qt: Vec<f32>,
    qpt: Vec<f32>,
    gamma: Option<Vec<f32>>,
}

fn base_inputs(mode: Mode) -> Inputs {
    Inputs {
        n: vec![0.035; N],
        qsp: vec![0.4; N],
        psp: vec![20.0; N],
        qt: vec![100.0, 120.0, 140.0, 160.0],
        qpt: vec![10.0, 12.0, 14.0, 16.0],
        gamma: matches!(mode, Mode::LearnedGamma).then(|| vec![LEARNED_GAMMA; N]),
    }
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
        Parent::Gamma => inp.gamma.as_ref().expect("no learned gamma"),
    }
}

fn field_mut(inp: &mut Inputs, p: Parent) -> &mut Vec<f32> {
    match p {
        Parent::N => &mut inp.n,
        Parent::QSpatial => &mut inp.qsp,
        Parent::PSpatial => &mut inp.psp,
        Parent::QT => &mut inp.qt,
        Parent::QPrimeT => &mut inp.qpt,
        Parent::Gamma => inp.gamma.as_mut().expect("no learned gamma"),
    }
}

/// Run the forward and backward with the parents in `tracked` requiring grad.
/// Returns one gradient per entry of `tracked`, in order.
fn analytical(
    cfg: &Config,
    inp: &Inputs,
    tracked: &[Parent],
    loss_of: impl Fn(Tensor<AB, 1>) -> Tensor<AB, 1>,
) -> Vec<Vec<f32>> {
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
    let gamma = inp.gamma.as_ref().map(|g| mk(g, Parent::Gamma));
    let q_next = timestep_forward::<I>(
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
        false,
        None,
        gamma.clone(),
    );
    let grads = loss_of(q_next).backward();
    tracked
        .iter()
        .map(|p| {
            let g = match p {
                Parent::N => n.grad(&grads),
                Parent::QSpatial => qsp.grad(&grads),
                Parent::PSpatial => psp.grad(&grads),
                Parent::QT => qt.grad(&grads),
                Parent::QPrimeT => qpt.grad(&grads),
                Parent::Gamma => gamma.as_ref().unwrap().grad(&grads),
            };
            g.unwrap_or_else(|| panic!("no grad on {p:?}"))
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        })
        .collect()
}

fn loss_value(cfg: &Config, inp: &Inputs) -> f32 {
    let adj = chain();
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let t = |d: &[f32]| -> Tensor<AB, 1> { Tensor::from_floats(d, &dev) };
    let q_next = timestep_forward::<I>(
        cfg,
        &pattern,
        &assembler,
        t(&inp.n),
        t(&inp.qsp),
        t(&inp.psp),
        t(&inp.qt),
        t(&inp.qpt),
        t(&adj.length_m),
        t(&adj.slope),
        t(&vec![0.3f32; N]),
        false,
        None,
        inp.gamma.as_ref().map(|g| t(g)),
    );
    q_next.sum().into_data().to_vec::<f32>().unwrap()[0]
}

fn fd(cfg: &Config, inp: &Inputs, parent: Parent, rel_step: f32) -> Vec<f32> {
    let base = field(inp, parent).clone();
    (0..N)
        .map(|i| {
            let eps = (rel_step * base[i].abs()).max(rel_step);
            let mut plus = inp.clone();
            field_mut(&mut plus, parent)[i] = base[i] + eps;
            let mut minus = inp.clone();
            field_mut(&mut minus, parent)[i] = base[i] - eps;
            (loss_value(cfg, &plus) - loss_value(cfg, &minus)) / (2.0 * eps)
        })
        .collect()
}

fn slot_ok(a: f32, f: f32) -> bool {
    let abs = (a - f).abs();
    let rel = abs / a.abs().max(f.abs()).max(1e-12);
    rel < REL_TOL || abs < ABS_TOL
}

/// One FD step (`EPS`), as in `tests/sp8_gradcheck.rs`. A slot passes when the
/// analytical and FD values agree within `REL_TOL` or `ABS_TOL`.
fn check(mode: Mode, parent: Parent) {
    let cfg = mock_cfg(mode);
    let inp = base_inputs(mode);
    let a = analytical(&cfg, &inp, &[parent], |q| q.sum()).remove(0);
    let f = fd(&cfg, &inp, parent, EPS);
    let rel = |x: f32, y: f32| (x - y).abs() / x.abs().max(y.abs()).max(1e-12);
    let mut worst = 0.0f32;
    let mut ok = true;
    for i in 0..N {
        println!(
            "  {parent:?}[{i}] analytical={:.6e} fd={:.6e} rel={:.3e}",
            a[i],
            f[i],
            rel(a[i], f[i])
        );
        worst = worst.max(rel(a[i], f[i]));
        ok &= slot_ok(a[i], f[i]);
    }
    println!("{parent:?}: worst rel {worst:.3e}");
    assert!(ok, "{parent:?}: gradcheck failed (worst rel {worst:.3e})");
}

#[test]
fn x_is_interior_at_the_base_point() {
    let cfg = mock_cfg(Mode::Plain);
    let inp = base_inputs(Mode::Plain);
    let adj = chain();
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let t = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &dev) };
    let k1 = __spike_forward_chain_k1_outputs::<I>(
        &cfg,
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
    for i in 0..N {
        let w = inp.qt[i] / (k1[1][i] * 1e-3 * k1[7][i] * LEN);
        assert!(w > 0.0 && w < 1.0, "reach {i}: w = {w} is not in (0, 1)");
    }
}

#[test]
fn gradcheck_n() {
    check(Mode::Plain, Parent::N);
}
#[test]
fn gradcheck_q_spatial() {
    check(Mode::Plain, Parent::QSpatial);
}
#[test]
fn gradcheck_p_spatial() {
    check(Mode::Plain, Parent::PSpatial);
}
#[test]
fn gradcheck_q_t() {
    check(Mode::Plain, Parent::QT);
}
#[test]
fn gradcheck_q_prime_t() {
    check(Mode::Plain, Parent::QPrimeT);
}

#[test]
fn gradcheck_n_with_global_gamma() {
    check(Mode::GlobalGamma, Parent::N);
}
#[test]
fn gradcheck_q_spatial_with_global_gamma() {
    check(Mode::GlobalGamma, Parent::QSpatial);
}
#[test]
fn gradcheck_p_spatial_with_global_gamma() {
    check(Mode::GlobalGamma, Parent::PSpatial);
}
#[test]
fn gradcheck_q_t_with_global_gamma() {
    check(Mode::GlobalGamma, Parent::QT);
}

#[test]
fn gradcheck_n_with_learned_gamma() {
    check(Mode::LearnedGamma, Parent::N);
}
#[test]
fn gradcheck_q_spatial_with_learned_gamma() {
    check(Mode::LearnedGamma, Parent::QSpatial);
}
#[test]
fn gradcheck_p_spatial_with_learned_gamma() {
    check(Mode::LearnedGamma, Parent::PSpatial);
}
#[test]
fn gradcheck_q_t_with_learned_gamma() {
    check(Mode::LearnedGamma, Parent::QT);
}
#[test]
fn gradcheck_gamma_with_learned_gamma() {
    check(Mode::LearnedGamma, Parent::Gamma);
}

#[test]
fn gradcheck_n_with_enforce_positivity() {
    check(Mode::Positivity, Parent::N);
}
#[test]
fn gradcheck_q_spatial_with_enforce_positivity() {
    check(Mode::Positivity, Parent::QSpatial);
}
#[test]
fn gradcheck_p_spatial_with_enforce_positivity() {
    check(Mode::Positivity, Parent::PSpatial);
}
#[test]
fn gradcheck_q_t_with_enforce_positivity() {
    check(Mode::Positivity, Parent::QT);
}

/// A tracked parent's gradient must not change with the parent mask.
#[test]
fn parent_gradients_do_not_depend_on_which_other_parents_are_tracked() {
    let cfg = mock_cfg(Mode::LearnedGamma);
    let inp = base_inputs(Mode::LearnedGamma);
    let all = analytical(&cfg, &inp, &ALL, |q| q.sum());
    for (k, parent) in ALL.iter().enumerate() {
        let alone = analytical(&cfg, &inp, &[*parent], |q| q.sum()).remove(0);
        for (i, (a, b)) in alone.iter().zip(&all[k]).enumerate() {
            assert!(
                a == b,
                "{parent:?}[{i}]: gradient changed with the parent mask ({a} alone vs {b} all tracked)"
            );
        }
    }
}

#[test]
fn n_gradient_is_exactly_zero_where_the_depth_floor_binds() {
    let cfg = mock_cfg(Mode::Plain);
    let mut inp = base_inputs(Mode::Plain);
    inp.qt = vec![1e-6, 120.0, 140.0, 160.0];
    let g = analytical(&cfg, &inp, &[Parent::N], |q| q.sum()).remove(0);
    assert_eq!(g[0], 0.0, "dL/dn at the floored reach is {}", g[0]);
}

#[test]
fn power_law_backward_is_finite_at_zero_discharge() {
    let cfg = mock_cfg(Mode::LearnedGamma);
    let mut inp = base_inputs(Mode::LearnedGamma);
    inp.qt = vec![0.0, 1e-9, 120.0, 140.0];
    let gs = analytical(&cfg, &inp, &ALL, |q| q.sum());
    for (parent, g) in ALL.iter().zip(&gs) {
        for (i, v) in g.iter().enumerate() {
            assert!(v.is_finite(), "{parent:?}[{i}] = {v} is not finite");
        }
    }
}
