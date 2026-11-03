//! Gradient check of the fused TimestepOp on `geometry_form: power_law`.
//! Same harness as `tests/sp8_gradcheck.rs`: analytical gradient against
//! central finite differences, per tracked parent, on the 4-reach chain with
//! `L = 10_000` m so the Cunge X is interior (B19's terms run).
//!
//! FD step: one rule for every slot, declared in `step_of`.
//! `h = max(h0, h_rt)`, with `h0 = 1e-3 * max(1, |x|)` and
//! `h_rt = 4 * eps_f32 * sum|q_next| / (REL_TOL * |g|)`, capped at
//! `0.1 * max(1, |x|)`. The loss is an f32 sum, so the FD carries a roundoff
//! error of about `eps_f32 * sum|q_next| / h`. Where `|g|` is small next to
//! `sum|q_next|`, a fixed step puts the FD at that floor and it cannot resolve
//! `REL_TOL`. `h_rt` is the step at which the floor is `REL_TOL / 4` of `|g|`.
//! A slot where the cap binds is printed as `(capped)` and relies on
//! `ABS_TOL`. `check` prints `h` and `roundoff_est` per slot.
//!
//! `enforce_positivity` base point: every positivity gradcheck first asserts
//! that no clamp selection (depth floor, velocity band, K floor, Cunge X band,
//! positivity X cap) changes between `x - h` and `x + h`. With
//! `Q_t = [100, 120, 140, 160]` the X cap on reach 1 binds exactly at the base
//! point, so the stencil straddled a kink (n gradient 140 vs FD 326). The
//! positivity tests therefore use `Q_t = [100, 150, 160, 170]`, the nearest
//! variant found that is kink-free for all four parents.

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
/// Largest FD step, as a fraction of `max(1, |x|)`.
const STEP_CAP: f32 = 0.1;
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
        qt: match mode {
            // Positivity base point: see the module doc.
            Mode::Positivity => vec![100.0, 150.0, 160.0, 170.0],
            _ => vec![100.0, 120.0, 140.0, 160.0],
        },
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

fn q_next_of(cfg: &Config, inp: &Inputs) -> Vec<f32> {
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
    q_next.into_data().to_vec::<f32>().unwrap()
}

fn loss_value(cfg: &Config, inp: &Inputs) -> f32 {
    q_next_of(cfg, inp).iter().sum()
}

/// FD step for one slot, one rule for every slot (see the module doc):
/// `h = max(h0, h_rt)`, capped at `CAP * max(1, |x|)`.
/// Returns `(h, capped)`.
fn step_of(x: f32, g: f32, q_abs_sum: f32) -> (f32, bool) {
    let scale = x.abs().max(1.0);
    let h0 = EPS * scale;
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


/// Clamp selections per reach, read from the saved forward state (k1 hook):
/// depth floor, velocity band, K floor, Cunge X band and the positivity X cap.
/// Each entry is a small code; two stencil ends with equal codes are on the
/// same smooth branch. Mirrors S6, S16, S18', S19 and S19' in `mmc_op.rs`.
fn selections(cfg: &Config, inp: &Inputs) -> Vec<[u8; 5]> {
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
    let dt = ddrs::routing::mmc::DT_SECONDS;
    let delta = ddrs::routing::mmc_op::POSITIVITY_DELTA;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let enforce = cfg.params.enforce_positivity;
    (0..N)
        .map(|i| {
            let (depth, top, v_un, v_cl, c, k_m) = (k[0][i], k[1][i], k[5][i], k[6][i], k[7][i], k[8][i]);
            let depth_floor = (depth <= depth_lb) as u8;
            let v_band = if v_cl > v_un { 1 } else if v_cl < v_un { 2 } else { 0 };
            let k_raw = LEN / c;
            let k_floor = (enforce && k_m > k_raw) as u8;
            let w = inp.qt[i] / (top * 1e-3 * c * LEN + 1e-12);
            let x_raw = (-w + 1.0) * 0.5;
            let x_band = if x_raw < 0.0 { 1 } else if x_raw > 0.5 { 2 } else { 0 };
            let x_cap = if enforce {
                let x_c = x_raw.clamp(0.0, 0.5);
                let cr = k_m.recip() * dt;
                let hi_a = cr * (0.5 * (1.0 - delta));
                let hi_b = (-cr * 0.5 + 1.0) * (1.0 - delta);
                if hi_a < x_c && hi_a <= hi_b { 1 } else if hi_b < x_c { 2 } else { 0 }
            } else {
                0
            };
            [depth_floor, v_band, k_floor, x_band, x_cap]
        })
        .collect()
}

/// Panics when any clamp selection differs between `x - h` and `x + h` of the
/// perturbed slot. A differing selection puts a kink inside the FD stencil.
fn assert_kink_free(cfg: &Config, inp: &Inputs, parent: Parent, steps: &[f32]) {
    let base = field(inp, parent).clone();
    for i in 0..N {
        let h = steps[i];
        let mut plus = inp.clone();
        field_mut(&mut plus, parent)[i] = base[i] + h;
        let mut minus = inp.clone();
        field_mut(&mut minus, parent)[i] = base[i] - h;
        let mut mid = inp.clone();
        field_mut(&mut mid, parent)[i] = base[i];
        let (sp, sm, s0) = (selections(cfg, &plus), selections(cfg, &minus), selections(cfg, &mid));
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

/// A slot passes when the analytical and FD values agree within `REL_TOL` or
/// `ABS_TOL`. The FD step comes from `step_of`.
fn check(mode: Mode, parent: Parent) {
    let cfg = mock_cfg(mode);
    let inp = base_inputs(mode);
    let a = analytical(&cfg, &inp, &[parent], |q| q.sum()).remove(0);
    let q_abs_sum: f32 = q_next_of(&cfg, &inp).iter().map(|q| q.abs()).sum();
    let picked: Vec<(f32, bool)> = (0..N)
        .map(|i| step_of(field(&inp, parent)[i], a[i], q_abs_sum))
        .collect();
    let steps: Vec<f32> = picked.iter().map(|p| p.0).collect();
    assert_kink_free(&cfg, &inp, parent, &steps);
    let f = fd(&cfg, &inp, parent, &steps);
    let rel = |x: f32, y: f32| (x - y).abs() / x.abs().max(y.abs()).max(1e-12);
    let mut worst = 0.0f32;
    let mut ok = true;
    for i in 0..N {
        println!(
            "  {parent:?}[{i}] h={:.3e}{} analytical={:.6e} fd={:.6e} rel={:.3e} roundoff_est={:.3e}",
            steps[i],
            if picked[i].1 { " (capped)" } else { "" },
            a[i],
            f[i],
            rel(a[i], f[i]),
            f32::EPSILON * q_abs_sum / steps[i] / a[i].abs().max(1e-12)
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
