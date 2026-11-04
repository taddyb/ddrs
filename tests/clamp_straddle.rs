//! Straddle tests: the gradient at every Muskingum-Cunge clamp site, for both
//! channel-section forms (`trapezoid` and `power_law`).
//!
//! For each (site, form) the raw value sits at five places against the bound:
//! far below, just below, an exact tie, just above and far above. At each place
//! the tests check four things on the straddling reach.
//!
//! 1. The clamp ledger is active exactly where the raw value is strictly
//!    outside the bound. A tie is inactive (`src/routing/clamp_rule.rs`).
//! 2. The clamp probe cuts the gradient to exactly 0.0 where the site is
//!    active. Where it is inactive, the gradient passes bit for bit and is not
//!    zero.
//! 3. At the two far places only, a central finite difference of the parent
//!    through the site matches the analytical gradient. The ledger state at
//!    `x - h` and `x + h` must equal the state at `x`, so the step does not
//!    cross the bound.
//!
//! A bound that is a config value (depth, bottom width, velocity floor and
//! discharge floor) is moved around a fixed raw value `r`. A bound that is a
//! literal (side slope 0.5 and 50, velocity 15, X band 0 and 0.5, K floor, X
//! cap) is crossed by bisecting one input knob, then walking the knob's f32
//! neighbours (up to 64 each way) to the closest raw values on each side and
//! an exact tie if one exists. When no exact tie exists the test prints
//! `no exact tie reachable at <site>` and skips only that place.
//!
//! 4. At every place, the fused gradient of all five parents matches the
//!    gradient of `src/routing/reference_step.rs::reference_timestep` (plain Burn
//!    autodiff): the same zero set, and nonzero entries at rel < 1e-3 or
//!    abs < 1e-6. This covers the three far places where the FD cannot
//!    resolve the gradient, and the ties.
//!
//! An FD is skipped, with a printed reason, where the round-off step `h_rt` is
//! more than 10 % of the parameter.
//!
//! The leakance head has no ledger sites. Four tests at the end check its two
//! tie cases (head floor, bed cap) against the head with that clamp removed.
//!
//! Fixture: a 4-reach chain of 10 km reaches (a copy of the one in
//! `tests/clamp_ledger.rs`), so the Cunge X is interior. Loss: `q_next.sum()`.
//! The tests collect every failed check and assert at the end, so one failure
//! does not hide the other places.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::{Config, GeometryForm};
use ddrs::routing::clamp_ledger::{ClampLedger, ClampProbe, ClampSite, ProbeSite};
use ddrs::routing::mmc::DT_SECONDS;
use ddrs::routing::reference_step::reference_timestep;
use ddrs::routing::mmc_op::{
    __spike_forward_chain_k1_outputs, __spike_forward_chain_k23_outputs, timestep_forward_diag,
    timestep_forward_leakance, POSITIVITY_DELTA,
};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 4;
const EPS: f32 = 1e-3;
const STEP_CAP: f32 = 0.1;
const REL_TOL: f32 = 5e-3;
const ABS_TOL: f32 = 1e-4;

// Slots of the k1 hook output (`forward_saved_idx` minus the four slots k1 skips).
const K1_DEPTH: usize = 0;
const K1_TOP: usize = 1;
const K1_VUN: usize = 5;
const K1_CEL: usize = 7;
const K1_K: usize = 8;
const K1_SSRAW: usize = 17;
const K1_BWRAW: usize = 18;

const TRAP: GeometryForm = GeometryForm::Trapezoid;
const POWER: GeometryForm = GeometryForm::PowerLaw;

// ---------------------------------------------------------------- fixture

#[derive(Clone)]
struct Inputs {
    n: Vec<f32>,
    qsp: Vec<f32>,
    psp: Vec<f32>,
    qt: Vec<f32>,
    qpt: Vec<f32>,
    length: Vec<f32>,
    slope: Vec<f32>,
}

fn f4_10k() -> Inputs {
    Inputs {
        n: vec![0.035; N],
        qsp: vec![0.4; N],
        psp: vec![20.0; N],
        qt: vec![100.0, 120.0, 140.0, 160.0],
        qpt: vec![10.0, 12.0, 14.0, 16.0],
        length: vec![10_000.0; N],
        slope: vec![1e-3; N],
    }
}

fn chain(f: &Inputs) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, f.length.clone(), f.slope.clone())
}

fn cfg(form: GeometryForm, enforce_positivity: bool) -> Config {
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
    cfg.params.geometry_form = form;
    cfg.params.enforce_positivity = enforce_positivity;
    cfg
}

fn host(t: Tensor<I, 1>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

fn device() -> <I as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

// ---------------------------------------------------------------- parents

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Wrt {
    N,
    QSp,
    PSp,
    QT,
}

fn field(inp: &Inputs, w: Wrt) -> &Vec<f32> {
    match w {
        Wrt::N => &inp.n,
        Wrt::QSp => &inp.qsp,
        Wrt::PSp => &inp.psp,
        Wrt::QT => &inp.qt,
    }
}

fn field_mut(inp: &mut Inputs, w: Wrt) -> &mut Vec<f32> {
    match w {
        Wrt::N => &mut inp.n,
        Wrt::QSp => &mut inp.qsp,
        Wrt::PSp => &mut inp.psp,
        Wrt::QT => &mut inp.qt,
    }
}

// ---------------------------------------------------------------- runs

struct Out {
    q_next: Vec<f32>,
    active: BTreeMap<ClampSite, Vec<bool>>,
    /// Per probe site: `(g_in, g_out)`.
    probe: BTreeMap<ProbeSite, (Vec<f32>, Vec<f32>)>,
    grads: BTreeMap<Wrt, Vec<f32>>,
}

/// One routed step through `timestep_forward_diag`, with the ledger and the
/// probe. With `backward`, also the loss `q_next.sum()` backward.
fn run(cfg: &Config, f: &Inputs, backward: bool) -> Out {
    let adj = chain(f);
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let leaf = |d: &[f32]| -> Tensor<AB, 1> { Tensor::<AB, 1>::from_floats(d, &dev).require_grad() };
    let n = leaf(&f.n);
    let qsp = leaf(&f.qsp);
    let psp = leaf(&f.psp);
    let qt = leaf(&f.qt);
    let qpt = leaf(&f.qpt);
    let length: Tensor<AB, 1> = Tensor::from_floats(f.length.as_slice(), &dev);
    let slope: Tensor<AB, 1> = Tensor::from_floats(f.slope.as_slice(), &dev);
    let xs: Tensor<AB, 1> = Tensor::from_floats(vec![0.3f32; N].as_slice(), &dev);

    let mut ledger = ClampLedger::default();
    let probe = Arc::new(Mutex::new(ClampProbe::default()));
    let q = timestep_forward_diag::<I>(
        cfg,
        &pattern,
        &assembler,
        n.clone(),
        qsp.clone(),
        psp.clone(),
        qt.clone(),
        qpt.clone(),
        length,
        slope,
        xs,
        None,
        Some(&mut ledger),
        Some(probe.clone()),
    );
    let q_next = host(q.clone().inner());
    assert_eq!(ledger.steps.len(), 1);
    let active = ledger.steps[0].active.clone();
    let mut grads = BTreeMap::new();
    let mut probe_out = BTreeMap::new();
    if backward {
        let g = q.sum().backward();
        for (w, t) in [(Wrt::N, &n), (Wrt::QSp, &qsp), (Wrt::PSp, &psp), (Wrt::QT, &qt)] {
            grads.insert(w, host(t.grad(&g).expect("tracked parent has a gradient")));
        }
        let p = probe.lock().unwrap().clone();
        assert_eq!(p.steps.len(), 1, "one backward call, one probe step");
        for (site, sg) in &p.steps[0].sites {
            probe_out.insert(*site, (sg.g_in.clone(), sg.g_out.clone()));
        }
    }
    Out { q_next, active, probe: probe_out, grads }
}

/// The 19 outputs of the k1 hook (see `__spike_forward_chain_k1_outputs`).
fn k1(cfg: &Config, f: &Inputs) -> Vec<Vec<f32>> {
    let adj = chain(f);
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let t = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &dev) };
    __spike_forward_chain_k1_outputs::<I>(
        cfg,
        &pattern,
        t(&f.n),
        t(&f.qsp),
        t(&f.psp),
        t(&f.qt),
        t(&f.qpt),
        t(&f.length),
        t(&f.slope),
        t(&[0.3; N]),
    )
}

/// The solved flow before the discharge floor (k23 hook `x_sol`).
fn x_sol(cfg: &Config, f: &Inputs) -> Vec<f32> {
    let adj = chain(f);
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let t = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &dev) };
    __spike_forward_chain_k23_outputs::<I>(
        cfg,
        &pattern,
        t(&f.n),
        t(&f.qsp),
        t(&f.psp),
        t(&f.qt),
        t(&f.qpt),
        t(&f.length),
        t(&f.slope),
        t(&[0.3; N]),
    )
    .2
}

/// Pre-clamp Cunge X of reach `i`: the forward's own ops (`mmc_op.rs`, S19).
fn x_raw_of(k: &[Vec<f32>], f: &Inputs, i: usize) -> f32 {
    let w = f.qt[i] / (k[K1_TOP][i] * f.slope[i] * k[K1_CEL][i] * f.length[i] + 1e-12);
    (-w + 1.0) * 0.5
}

// ---------------------------------------------------------------- points

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pos {
    FarBelow,
    JustBelow,
    Tie,
    JustAbove,
    FarAbove,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// The bound is a lower bound: active where `raw < bound`.
    Floor,
    /// The bound is an upper bound: active where `raw > bound`.
    Ceiling,
}

struct Point {
    pos: Pos,
    cfg: Config,
    inp: Inputs,
    raw: f32,
    bound: f32,
}

struct Spec {
    site: ClampSite,
    form: GeometryForm,
    kind: Kind,
    /// The straddling reach.
    reach: usize,
    /// The parent for the finite difference at the far places.
    fd: Wrt,
}

/// A config-value bound moved around a fixed raw value `r` (floors only).
fn moved_floor_points(
    base_cfg: &Config,
    inp: &Inputs,
    r: f32,
    set: impl Fn(&mut Config, f32),
) -> Vec<Point> {
    assert!(r > 0.0 && r.is_finite(), "raw value {r} is not usable");
    [
        (Pos::FarBelow, 2.0 * r),
        (Pos::JustBelow, r.next_up()),
        (Pos::Tie, r),
        (Pos::JustAbove, r.next_down()),
        (Pos::FarAbove, r / 2.0),
    ]
    .into_iter()
    .map(|(pos, bound)| {
        let mut c = base_cfg.clone();
        set(&mut c, bound);
        Point { pos, cfg: c, inp: inp.clone(), raw: r, bound }
    })
    .collect()
}

/// A literal bound crossed by one positive input knob. `eval(x)` returns the
/// `(raw, bound)` pair at knob value `x`; `build(x)` returns the config and
/// inputs at `x`. `lo` and `hi` are the two far places and must lie on
/// opposite sides of the bound. Returns the places found and the places
/// skipped.
fn knob_points(
    site: ClampSite,
    lo: f32,
    hi: f32,
    eval: &dyn Fn(f32) -> (f32, f32),
    build: &dyn Fn(f32) -> (Config, Inputs),
) -> Vec<Point> {
    let less = |x: f32| {
        let (r, b) = eval(x);
        r < b
    };
    let (less_lo, less_hi) = (less(lo), less(hi));
    assert_ne!(
        less_lo, less_hi,
        "{}: knob ends {lo} and {hi} are on the same side of the bound (eval {:?} and {:?})",
        site.name(),
        eval(lo),
        eval(hi)
    );
    // Bisect on the f32 bit pattern: adjacent floats at the end.
    let (mut a, mut b) = (lo, hi);
    while (a.to_bits() as i64 - b.to_bits() as i64).abs() > 1 {
        let mid = f32::from_bits(((a.to_bits() as u64 + b.to_bits() as u64) / 2) as u32);
        if less(mid) == less_lo {
            a = mid;
        } else {
            b = mid;
        }
    }
    // Walk 64 neighbours away from the crossing on each side.
    let a_dir_down = a < b;
    let mut xs = Vec::new();
    let mut x = a;
    for _ in 0..=64 {
        xs.push(x);
        x = if a_dir_down { x.next_down() } else { x.next_up() };
    }
    let mut x = b;
    for _ in 0..=64 {
        xs.push(x);
        x = if a_dir_down { x.next_up() } else { x.next_down() };
    }
    let evals: Vec<(f32, f32, f32)> = xs
        .iter()
        .map(|&x| {
            let (r, b) = eval(x);
            (x, r, b)
        })
        .collect();
    let best = |want_below: bool| {
        evals
            .iter()
            .filter(|(_, r, b)| if want_below { r < b } else { r > b })
            .min_by(|p, q| (p.2 - p.1).abs().partial_cmp(&(q.2 - q.1).abs()).unwrap())
            .copied()
    };
    let tie = evals.iter().find(|(_, r, b)| r == b).copied();
    let (lo_pos, hi_pos) = if less_lo {
        (Pos::FarBelow, Pos::FarAbove)
    } else {
        (Pos::FarAbove, Pos::FarBelow)
    };
    let mut places: Vec<(Pos, f32)> = vec![(lo_pos, lo), (hi_pos, hi)];
    match best(true) {
        Some((x, _, _)) => places.push((Pos::JustBelow, x)),
        None => println!("no point just below reachable at {}", site.name()),
    }
    match tie {
        Some((x, _, _)) => places.push((Pos::Tie, x)),
        None => println!("no exact tie reachable at {}", site.name()),
    }
    match best(false) {
        Some((x, _, _)) => places.push((Pos::JustAbove, x)),
        None => println!("no point just above reachable at {}", site.name()),
    }
    places
        .into_iter()
        .map(|(pos, x)| {
            let (c, i) = build(x);
            let (raw, bound) = eval(x);
            Point { pos, cfg: c, inp: i, raw, bound }
        })
        .collect()
}

// ---------------------------------------------------------------- checks

fn expect_active(kind: Kind, pos: Pos) -> bool {
    match kind {
        Kind::Floor => matches!(pos, Pos::FarBelow | Pos::JustBelow),
        Kind::Ceiling => matches!(pos, Pos::JustAbove | Pos::FarAbove),
    }
}

/// FD step for one slot. `h = max(h0, h_rt)` capped at `0.1 * max(1, |x|)`
/// (the rule of `tests/power_law_gradcheck.rs::step_of`). An exactly zero
/// analytical gradient has nothing to resolve: use `h0`. The cap also keeps
/// `x - h` positive (`h <= 0.5 |x|`), because the rule's cap of 0.1 is larger
/// than a small parameter such as `n = 0.035` and would give a negative `n`.
fn step_of(x: f32, g: f32, q_abs_sum: f32) -> (f32, bool) {
    let scale = x.abs().max(1.0);
    let h0 = EPS * scale;
    if g == 0.0 {
        return (h0, false);
    }
    let h_rt = 4.0 * f32::EPSILON * q_abs_sum / (REL_TOL * g.abs().max(1e-12));
    let cap = (STEP_CAP * scale).min(0.5 * x.abs());
    let h = h0.max(h_rt);
    if h > cap {
        (cap, true)
    } else {
        (h, false)
    }
}

fn slot_ok(a: f32, f: f32) -> bool {
    let abs = (a - f).abs();
    let rel = abs / a.abs().max(f.abs()).max(1e-12);
    rel < REL_TOL || abs < ABS_TOL
}

fn differing_sites(a: &Out, b: &Out, reach: usize) -> Vec<&'static str> {
    ClampSite::ALL
        .iter()
        .filter(|s| a.active.get(s).map(|v| v[reach]) != b.active.get(s).map(|v| v[reach]))
        .map(|s| s.name())
        .collect()
}

/// Fused gradient against the autodiff reference for all five parents, with
/// the loss `q_next.sum()`. Pushes one message per mismatch.
fn autodiff_check(tag: &str, cfg: &Config, f: &Inputs, fails: &mut Vec<String>) {
    let adj = chain(f);
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let one = |fused: bool| -> Vec<Vec<f32>> {
        let leaf = |d: &[f32]| -> Tensor<AB, 1> { Tensor::<AB, 1>::from_floats(d, &dev).require_grad() };
        let c = |d: &[f32]| -> Tensor<AB, 1> { Tensor::<AB, 1>::from_floats(d, &dev) };
        let (n, qsp, psp, qt, qpt) = (leaf(&f.n), leaf(&f.qsp), leaf(&f.psp), leaf(&f.qt), leaf(&f.qpt));
        let (length, slope, xs) = (c(&f.length), c(&f.slope), c(&[0.3; N]));
        let q = if fused {
            timestep_forward_diag::<I>(
                cfg, &pattern, &assembler, n.clone(), qsp.clone(), psp.clone(), qt.clone(),
                qpt.clone(), length, slope, xs, None, None, None,
            )
        } else {
            reference_timestep::<I>(
                cfg, &pattern, &assembler, n.clone(), qsp.clone(), psp.clone(), qt.clone(),
                qpt.clone(), length, slope, xs, None,
            )
        };
        let g = q.sum().backward();
        [&n, &qsp, &psp, &qt, &qpt].iter().map(|t| host(t.grad(&g).expect("tracked parent"))).collect()
    };
    let (gf, gr) = (one(true), one(false));
    let names = ["n", "q_spatial", "p_spatial", "q_t", "q_prime_t"];
    for (p, (a, b)) in gf.iter().zip(&gr).enumerate() {
        if !(a.iter().chain(b).all(|x| x.is_finite())) {
            fails.push(format!("{tag}: autodiff check, non-finite gradient for {}: fused {a:?} reference {b:?}", names[p]));
            continue;
        }
        for i in 0..N {
            let (x, y) = (a[i], b[i]);
            if (x == 0.0) != (y == 0.0) {
                fails.push(format!("{tag}: autodiff zero-set mismatch {}[{i}]: fused {x:e}, reference {y:e}", names[p]));
            } else if x != 0.0 {
                let abs = (x - y).abs();
                let rel = abs / x.abs().max(y.abs());
                if !(rel < 1e-3 || abs < 1e-6) {
                    fails.push(format!("{tag}: autodiff value mismatch {}[{i}]: fused {x:e}, reference {y:e}, rel {rel:e}", names[p]));
                }
            }
        }
    }
}

fn check_point(spec: &Spec, p: &Point, fails: &mut Vec<String>) {
    let tag = format!("{} {:?} {:?}", spec.site.name(), spec.form, p.pos);
    let out = run(&p.cfg, &p.inp, true);
    let r = spec.reach;
    let act = out.active.get(&spec.site).unwrap_or_else(|| panic!("{tag}: site has no ledger entry"))[r];
    let want = expect_active(spec.kind, p.pos);
    let ps = spec.site.probe_site().expect("site has a probe site");
    let (g_in, g_out) = {
        let (a, b) = out.probe.get(&ps).unwrap_or_else(|| panic!("{tag}: no probe entry for {ps:?}"));
        (a[r], b[r])
    };
    let any_active = ClampSite::ALL
        .iter()
        .filter(|s| s.probe_site() == Some(ps))
        .any(|s| out.active.get(s).is_some_and(|v| v[r]));
    println!(
        "[{tag}] raw={:.9e} bound={:.9e} raw_bits={:#010x} bound_bits={:#010x} ledger_active={act} (want {want}) g_in={g_in:.6e} g_out={g_out:.6e}",
        p.raw,
        p.bound,
        p.raw.to_bits(),
        p.bound.to_bits(),
    );
    if act != want {
        fails.push(format!(
            "{tag}: ledger active={act}, want {want} (raw {:.9e}, bound {:.9e})",
            p.raw, p.bound
        ));
    }
    autodiff_check(&tag, &p.cfg, &p.inp, fails);
    if act != any_active {
        fails.push(format!(
            "{tag}: site {} ledger active={act}, but a site sharing probe site {ps:?} is active (any_active={any_active}); the pass-through check would run as a zero check",
            spec.site.name()
        ));
    }
    if any_active {
        if g_out != 0.0 {
            fails.push(format!("{tag}: probe site {ps:?} active but g_out={g_out:e} (g_in={g_in:e})"));
        }
        if g_in == 0.0 {
            println!("  note: g_in is 0 at an active place, so the cut proves nothing here");
        }
    } else {
        if g_out.to_bits() != g_in.to_bits() {
            fails.push(format!("{tag}: probe site {ps:?} inactive but g_out={g_out:e} != g_in={g_in:e}"));
        }
        if g_in == 0.0 {
            fails.push(format!(
                "{tag}: probe site {ps:?} inactive and g_in == 0 (g_out={g_out:e}); zero for a reason other than this site"
            ));
        }
    }

    if matches!(p.pos, Pos::FarBelow | Pos::FarAbove) {
        let x = field(&p.inp, spec.fd)[r];
        let a = out.grads[&spec.fd][r];
        let qsum: f32 = out.q_next.iter().map(|q| q.abs()).sum();
        // The FD needs a step that lifts the signal above f32 round-off. When
        // the parent has a small effect on the loss compared with the loss
        // itself, that step is a large part of the parameter, and truncation
        // error dominates the central difference. Such a point cannot be
        // checked by FD in f32, so only the FD comparison is skipped. The
        // ledger and probe checks above still ran.
        if a != 0.0 {
            let h_rt = 4.0 * f32::EPSILON * qsum / (REL_TOL * a.abs().max(1e-12));
            if h_rt > 0.1 * x.abs() {
                println!(
                    "FD not resolvable at {} {:?} {:?}: h_rt={h_rt:e} x={x:e} (the round-off step is more than 10% of the parameter)",
                    spec.site.name(),
                    spec.form,
                    p.pos
                );
                return;
            }
        }
        let (h, capped) = step_of(x, a, qsum);
        let shifted = |d: f32| {
            let mut i = p.inp.clone();
            field_mut(&mut i, spec.fd)[r] = x + d;
            i
        };
        let (plus, minus) = (run(&p.cfg, &shifted(h), false), run(&p.cfg, &shifted(-h), false));
        let s_base = out.active[&spec.site][r];
        let s_plus = plus.active[&spec.site][r];
        let s_minus = minus.active[&spec.site][r];
        let others: Vec<&str> = differing_sites(&out, &plus, r)
            .into_iter()
            .chain(differing_sites(&out, &minus, r))
            .collect();
        // The loss is the sum of `q_next`. The difference of the two losses is
        // formed per reach and added in f64, so the sum itself adds no
        // round-off. Each `q_next` is still an f32 value.
        let fd_of = |p: &Out, m: &Out, h: f32| -> f32 {
            let d: f64 = p.q_next.iter().zip(&m.q_next).map(|(a, b)| *a as f64 - *b as f64).sum();
            (d / (2.0 * h as f64)) as f32
        };
        let fd = fd_of(&plus, &minus, h);
        let rel = (a - fd).abs() / a.abs().max(fd.abs()).max(1e-12);
        println!(
            "  fd {:?}[{r}] x={x:.6e} h={h:.3e}{} analytical={a:.6e} fd={fd:.6e} rel={rel:.3e} ledger x-h/x/x+h={s_minus}/{s_base}/{s_plus} flipped_sites={others:?}",
            spec.fd,
            if capped { " (capped)" } else { "" },
        );
        if !(s_base == s_plus && s_base == s_minus) {
            fails.push(format!(
                "{tag}: the FD step crosses the bound (ledger {s_minus}/{s_base}/{s_plus}, h={h:e})"
            ));
        }
        if !slot_ok(a, fd) {
            // Convergence table: does the FD approach the analytical value as h shrinks?
            for div in [2.0f32, 4.0, 8.0] {
                let hh = h / div;
                let (p2, m2) = (run(&p.cfg, &shifted(hh), false), run(&p.cfg, &shifted(-hh), false));
                println!(
                    "    fd at h={hh:.3e}: {:.6e} (roundoff est {:.3e})",
                    fd_of(&p2, &m2, hh),
                    f32::EPSILON * qsum / hh
                );
            }
            fails.push(format!(
                "{tag}: FD mismatch on {:?}[{r}]: analytical {a:e}, fd {fd:e}, rel {rel:e}, h {h:e}, other sites flipping in the stencil: {others:?}",
                spec.fd
            ));
        }
    }
}

fn check_all(spec: &Spec, points: Vec<Point>) {
    println!("== {} {:?} ==", spec.site.name(), spec.form);
    let mut fails = Vec::new();
    for p in &points {
        check_point(spec, p, &mut fails);
    }
    for f in &fails {
        println!("FAILED: {f}");
    }
    assert!(
        fails.is_empty(),
        "{} {:?}: {} failed checks:\n{}",
        spec.site.name(),
        spec.form,
        fails.len(),
        fails.join("\n")
    );
}

// ---------------------------------------------------------------- moved bounds

fn depth_floor(form: GeometryForm) {
    let mut c = cfg(form, false);
    c.params.attribute_minimums.depth = 1e-6;
    let mut inp = f4_10k();
    inp.qt[0] = 0.05;
    let r = k1(&c, &inp)[K1_DEPTH][0];
    let pts = moved_floor_points(&c, &inp, r, |c, b| c.params.attribute_minimums.depth = b);
    let spec = Spec { site: ClampSite::DepthFloor, form, kind: Kind::Floor, reach: 0, fd: Wrt::N };
    check_all(&spec, pts);
}

fn bottom_width_floor(form: GeometryForm) {
    let c = cfg(form, false);
    let inp = f4_10k();
    let r = k1(&c, &inp)[K1_BWRAW][0];
    let pts = moved_floor_points(&c, &inp, r, |c, b| c.params.attribute_minimums.bottom_width = b);
    let spec = Spec { site: ClampSite::BottomWidthFloor, form, kind: Kind::Floor, reach: 0, fd: Wrt::PSp };
    check_all(&spec, pts);
}

fn velocity_low(form: GeometryForm) {
    let c = cfg(form, false);
    let mut inp = f4_10k();
    inp.qt[0] = 0.05;
    let r = k1(&c, &inp)[K1_VUN][0];
    let pts = moved_floor_points(&c, &inp, r, |c, b| c.params.attribute_minimums.velocity = b);
    let spec = Spec { site: ClampSite::VelocityLow, form, kind: Kind::Floor, reach: 0, fd: Wrt::N };
    check_all(&spec, pts);
}

fn discharge_floor(form: GeometryForm) {
    let c = cfg(form, false);
    let inp = f4_10k();
    let r = x_sol(&c, &inp)[0];
    let pts = moved_floor_points(&c, &inp, r, |c, b| c.params.attribute_minimums.discharge = b);
    let spec = Spec { site: ClampSite::DischargeFloor, form, kind: Kind::Floor, reach: 0, fd: Wrt::QT };
    check_all(&spec, pts);
}

#[test]
fn straddle_depth_floor_trapezoid() {
    depth_floor(TRAP);
}
#[test]
fn straddle_depth_floor_power_law() {
    depth_floor(POWER);
}
#[test]
fn straddle_bottom_width_floor_trapezoid() {
    bottom_width_floor(TRAP);
}
#[test]
fn straddle_velocity_low_trapezoid() {
    velocity_low(TRAP);
}
#[test]
fn straddle_velocity_low_power_law() {
    velocity_low(POWER);
}
#[test]
fn straddle_discharge_floor_trapezoid() {
    discharge_floor(TRAP);
}
#[test]
fn straddle_discharge_floor_power_law() {
    discharge_floor(POWER);
}

// ---------------------------------------------------------------- literal bounds

fn side_slope(form: GeometryForm, site: ClampSite, bound: f32, kind: Kind, lo: f32, hi: f32) {
    let c = cfg(form, false);
    let build = |p: f32| {
        let mut i = f4_10k();
        i.psp[0] = p;
        (c.clone(), i)
    };
    let eval = |p: f32| {
        let (c, i) = build(p);
        (k1(&c, &i)[K1_SSRAW][0], bound)
    };
    let pts = knob_points(site, lo, hi, &eval, &build);
    let spec = Spec { site, form, kind, reach: 0, fd: Wrt::PSp };
    check_all(&spec, pts);
}

#[test]
fn straddle_side_slope_low_trapezoid() {
    side_slope(TRAP, ClampSite::SideSlopeLow, 0.5, Kind::Floor, 1.0, 20.0);
}
#[test]
fn straddle_side_slope_high_trapezoid() {
    side_slope(TRAP, ClampSite::SideSlopeHigh, 50.0, Kind::Ceiling, 20.0, 1000.0);
}

fn velocity_high(form: GeometryForm) {
    let c = cfg(form, false);
    let build = |s: f32| {
        let mut i = f4_10k();
        i.n = vec![0.01; N];
        i.qt = vec![500.0; N];
        i.slope = vec![s; N];
        (c.clone(), i)
    };
    let eval = |s: f32| {
        let (c, i) = build(s);
        (k1(&c, &i)[K1_VUN][0], 15.0)
    };
    let pts = knob_points(ClampSite::VelocityHigh, 1e-3, 0.1, &eval, &build);
    let spec = Spec { site: ClampSite::VelocityHigh, form, kind: Kind::Ceiling, reach: 0, fd: Wrt::N };
    check_all(&spec, pts);
}

#[test]
fn straddle_velocity_high_trapezoid() {
    velocity_high(TRAP);
}
#[test]
fn straddle_velocity_high_power_law() {
    velocity_high(POWER);
}

fn x_band_low(form: GeometryForm) {
    let c = cfg(form, false);
    let build = |l: f32| {
        let mut i = f4_10k();
        i.length[0] = l;
        (c.clone(), i)
    };
    let eval = |l: f32| {
        let (c, i) = build(l);
        (x_raw_of(&k1(&c, &i), &i, 0), 0.0)
    };
    let pts = knob_points(ClampSite::XBandLow, 500.0, 10_000.0, &eval, &build);
    let spec = Spec { site: ClampSite::XBandLow, form, kind: Kind::Floor, reach: 0, fd: Wrt::QT };
    check_all(&spec, pts);
}

#[test]
fn straddle_x_band_low_trapezoid() {
    x_band_low(TRAP);
}
#[test]
fn straddle_x_band_low_power_law() {
    x_band_low(POWER);
}

/// X > 0.5 needs a negative flow, so there are no places above the bound.
/// The tie is the natural one at `Q_t[0] = 1e-9`. Just below is the smallest
/// `Q_t[0]` whose `x_raw` is strictly below 0.5.
fn x_band_high(form: GeometryForm) {
    let c = cfg(form, false);
    let build = |q: f32| {
        let mut i = f4_10k();
        i.qt[0] = q;
        (c.clone(), i)
    };
    let x_at = |q: f32| {
        let (c, i) = build(q);
        x_raw_of(&k1(&c, &i), &i, 0)
    };
    let mut pts = Vec::new();
    let mk = |pos: Pos, q: f32| {
        let (c, i) = build(q);
        Point { pos, cfg: c, inp: i, raw: x_at(q), bound: 0.5 }
    };
    pts.push(mk(Pos::FarBelow, 100.0));
    // Smallest Q_t with x_raw < 0.5, by bisection on the bit pattern.
    let (mut a, mut b) = (1e-9_f32, 100.0_f32);
    assert!(x_at(a) >= 0.5 && x_at(b) < 0.5);
    while b.to_bits() - a.to_bits() > 1 {
        let mid = f32::from_bits((a.to_bits() + b.to_bits()) / 2);
        if x_at(mid) < 0.5 {
            b = mid;
        } else {
            a = mid;
        }
    }
    pts.push(mk(Pos::JustBelow, b));
    if x_at(1e-9) == 0.5 {
        pts.push(mk(Pos::Tie, 1e-9));
    } else {
        println!("no exact tie reachable at {}", ClampSite::XBandHigh.name());
    }
    println!("x_band_high {form:?}: just below is Q_t[0] = {b:e}, x_raw = {:e}", x_at(b));
    let spec = Spec { site: ClampSite::XBandHigh, form, kind: Kind::Ceiling, reach: 0, fd: Wrt::QT };
    check_all(&spec, pts);
}

#[test]
fn straddle_x_band_high_trapezoid() {
    x_band_high(TRAP);
}
#[test]
fn straddle_x_band_high_power_law() {
    x_band_high(POWER);
}

fn k_floor(form: GeometryForm) {
    let c = cfg(form, true);
    let build = |l: f32| {
        let mut i = f4_10k();
        i.length[0] = l;
        (c.clone(), i)
    };
    let eval = |l: f32| {
        let (c, i) = build(l);
        let k = k1(&c, &i);
        (l / k[K1_CEL][0], DT_SECONDS * (1.0 + POSITIVITY_DELTA) / 2.0)
    };
    let pts = knob_points(ClampSite::KFloor, 500.0, 10_000.0, &eval, &build);
    let spec = Spec { site: ClampSite::KFloor, form, kind: Kind::Floor, reach: 0, fd: Wrt::N };
    check_all(&spec, pts);
}

#[test]
fn straddle_k_floor_trapezoid() {
    k_floor(TRAP);
}
#[test]
fn straddle_k_floor_power_law() {
    k_floor(POWER);
}

fn x_cap(form: GeometryForm) {
    let c = cfg(form, true);
    let build = |l: f32| {
        let mut i = f4_10k();
        i.length[0] = l;
        (c.clone(), i)
    };
    // Raw: the Cunge X after its band. Bound: the smaller of the two caps of
    // the same reach (S19'), because the ledger is active when the X exceeds
    // either. Below about 8 km `hi_b` is the smaller and the cap binds through
    // it, so the low end of the knob starts at 8 km, where the X is below both.
    // Above that, `hi_a` is the smaller. Both move with the knob.
    let eval = |l: f32| {
        let (c, i) = build(l);
        let k = k1(&c, &i);
        let x_cunge = x_raw_of(&k, &i, 0).clamp(0.0, 0.5);
        let cr = k[K1_K][0].recip() * DT_SECONDS;
        let hi_a = cr * (0.5 * (1.0 - POSITIVITY_DELTA));
        let hi_b = (-cr * 0.5 + 1.0) * (1.0 - POSITIVITY_DELTA);
        (x_cunge, hi_a.min(hi_b))
    };
    let pts = knob_points(ClampSite::XCap, 8_000.0, 30_000.0, &eval, &build);
    let spec = Spec { site: ClampSite::XCap, form, kind: Kind::Ceiling, reach: 0, fd: Wrt::N };
    check_all(&spec, pts);
}

#[test]
fn straddle_x_cap_trapezoid() {
    x_cap(TRAP);
}
#[test]
fn straddle_x_cap_power_law() {
    x_cap(POWER);
}

// ---------------------------------------------------------------- leakance ties

const K_D: f32 = 5e-7;
const LEAK_FAC: f32 = 0.5;
const BED_M: f32 = 5.0;
/// The reach whose `d_gw` sits at the tie.
const TIE_REACH: usize = 1;

const LEAK_PARENTS: [&str; 7] = ["n", "q_spatial", "p_spatial", "q_t", "K_D", "d_gw", "leakance_factor"];

struct LeakOut {
    q_next: Vec<f32>,
    grads: Vec<Vec<f32>>,
}

/// One step through `timestep_forward_leakance`; gradients of `q_next.sum()`
/// for the seven parents in `LEAK_PARENTS`.
fn leak_run(cfg: &Config, f: &Inputs, dgw: &[f32]) -> LeakOut {
    let adj = chain(f);
    let dev = device();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let leaf = |d: &[f32]| -> Tensor<AB, 1> { Tensor::<AB, 1>::from_floats(d, &dev).require_grad() };
    let n = leaf(&f.n);
    let qsp = leaf(&f.qsp);
    let psp = leaf(&f.psp);
    let qt = leaf(&f.qt);
    let qpt = leaf(&f.qpt);
    let kd = leaf(&[K_D; N]);
    let dg = leaf(dgw);
    let fac = leaf(&[LEAK_FAC; N]);
    let q = timestep_forward_leakance::<I>(
        cfg,
        &pattern,
        &assembler,
        n.clone(),
        qsp.clone(),
        psp.clone(),
        qt.clone(),
        qpt.clone(),
        Tensor::from_floats(f.length.as_slice(), &dev),
        Tensor::from_floats(f.slope.as_slice(), &dev),
        Tensor::from_floats(vec![0.3f32; N].as_slice(), &dev),
        kd.clone(),
        dg.clone(),
        fac.clone(),
        None,
        None,
        false,
        None,
        None,
    );
    let q_next = host(q.clone().inner());
    let g = q.sum().backward();
    let grads = [&n, &qsp, &psp, &qt, &kd, &dg, &fac]
        .iter()
        .map(|t| host(t.grad(&g).expect("tracked parent has a gradient")))
        .collect();
    LeakOut { q_next, grads }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `a` and `b` must be bitwise equal, in `q_next` and in every parent gradient.
/// Collects the largest difference when they are not.
fn compare_bitwise(tag: &str, a: &LeakOut, b: &LeakOut, fails: &mut Vec<String>) {
    if bits(&a.q_next) != bits(&b.q_next) {
        fails.push(format!("{tag}: q_next differs: {:?} vs {:?}", a.q_next, b.q_next));
    }
    for (k, name) in LEAK_PARENTS.iter().enumerate() {
        let (x, y) = (&a.grads[k], &b.grads[k]);
        if bits(x) != bits(y) {
            let worst = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
            let rel = x
                .iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs() / p.abs().max(q.abs()).max(1e-30))
                .fold(0.0f32, f32::max);
            fails.push(format!(
                "{tag}: gradient of {name} is not bitwise equal: max abs diff {worst:e}, max rel diff {rel:e}; clamped {x:?} vs removed {y:?}"
            ));
        }
    }
}

fn finish(tag: &str, fails: Vec<String>) {
    for f in &fails {
        println!("FAILED: {f}");
    }
    assert!(fails.is_empty(), "{tag}: {} failed checks:\n{}", fails.len(), fails.join("\n"));
}

/// Tie of the losing head floor: `depth == d_gw` on one reach. The tie passes
/// the gradient, so the clamped run equals the run with `losing_only: false`.
fn leakance_head_floor(form: GeometryForm) {
    let tag = format!("leakance_head_floor {form:?}");
    let inp = f4_10k();
    let mut clamped = cfg(form, false);
    clamped.params.leakance_losing_only = true;
    clamped.params.leakance_bed_thickness = None;
    let mut removed = clamped.clone();
    removed.params.leakance_losing_only = false;

    let depth = k1(&clamped, &inp)[K1_DEPTH][TIE_REACH];
    let mut dgw = vec![0.0f32; N];
    dgw[TIE_REACH] = depth;
    let a = leak_run(&clamped, &inp, &dgw);
    let b = leak_run(&removed, &inp, &dgw);
    println!("[{tag}] depth = d_gw = {depth:.9e} (bits {:#010x})", depth.to_bits());
    println!("  q_next clamped  {:?}", a.q_next);
    println!("  d_gw grad clamped {:?}", a.grads[5]);
    println!("  d_gw grad removed {:?}", b.grads[5]);

    let mut fails = Vec::new();
    compare_bitwise(&tag, &a, &b, &mut fails);
    let g_tie = a.grads[5][TIE_REACH];
    if g_tie == 0.0 {
        fails.push(format!("{tag}: the d_gw gradient on the tie reach is 0 (the tie must pass)"));
    }
    // One step past the tie on the cut side: d_gw above the depth, a gaining reach.
    let mut dgw_cut = dgw.clone();
    dgw_cut[TIE_REACH] = depth.next_up();
    let cut = leak_run(&clamped, &inp, &dgw_cut);
    println!("  cut side d_gw = {:.9e}: d_gw grad {:?}", dgw_cut[TIE_REACH], cut.grads[5]);
    if cut.grads[5][TIE_REACH] != 0.0 {
        fails.push(format!(
            "{tag}: one step past the tie (d_gw = next_up(depth)) the d_gw gradient is {:e}, want exactly 0",
            cut.grads[5][TIE_REACH]
        ));
    }
    finish(&tag, fails);
}

/// Tie of the bed cap: `d_gw == -M` on one reach, so `depth - d_gw` and
/// `depth + M` are equal bit for bit. The tie goes to the first operand of
/// `min_pair`, so the clamped run equals the run with `bed_thickness: None`.
fn leakance_bed_cap(form: GeometryForm) {
    let tag = format!("leakance_bed_cap {form:?}");
    let inp = f4_10k();
    let mut clamped = cfg(form, false);
    clamped.params.leakance_losing_only = true;
    clamped.params.leakance_bed_thickness = Some(BED_M);
    let mut removed = clamped.clone();
    removed.params.leakance_bed_thickness = None;

    let depth = k1(&clamped, &inp)[K1_DEPTH][TIE_REACH];
    assert!(depth + BED_M > 0.0);
    let mut dgw = vec![0.0f32; N];
    dgw[TIE_REACH] = -BED_M;
    let a = leak_run(&clamped, &inp, &dgw);
    let b = leak_run(&removed, &inp, &dgw);
    println!("[{tag}] depth = {depth:.9e}, d_gw = -M = {:.9e}", dgw[TIE_REACH]);
    println!("  q_next clamped  {:?}", a.q_next);
    println!("  d_gw grad clamped {:?}", a.grads[5]);
    println!("  d_gw grad removed {:?}", b.grads[5]);

    let mut fails = Vec::new();
    compare_bitwise(&tag, &a, &b, &mut fails);
    if a.grads[5][TIE_REACH] == 0.0 {
        fails.push(format!("{tag}: the d_gw gradient on the tie reach is 0 (the tie must pass)"));
    }
    // One step past the tie on the cut side: d_gw below -M, the stream is disconnected.
    let mut dgw_cut = dgw.clone();
    dgw_cut[TIE_REACH] = (-BED_M).next_down();
    let cut = leak_run(&clamped, &inp, &dgw_cut);
    println!("  cut side d_gw = {:.9e}: d_gw grad {:?}", dgw_cut[TIE_REACH], cut.grads[5]);
    if cut.grads[5][TIE_REACH] != 0.0 {
        fails.push(format!(
            "{tag}: one step past the tie (d_gw = next_down(-M)) the d_gw gradient is {:e}, want exactly 0",
            cut.grads[5][TIE_REACH]
        ));
    }
    finish(&tag, fails);
}

#[test]
fn straddle_leakance_head_floor_trapezoid() {
    leakance_head_floor(TRAP);
}
#[test]
fn straddle_leakance_head_floor_power_law() {
    leakance_head_floor(POWER);
}
#[test]
fn straddle_leakance_bed_cap_trapezoid() {
    leakance_bed_cap(TRAP);
}
#[test]
fn straddle_leakance_bed_cap_power_law() {
    leakance_bed_cap(POWER);
}
