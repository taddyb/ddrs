//! CUDA-backend verification of the `ddr_match: false` physics backwards.
//!
//! Build with: `cargo test --features cuda --test cuda_backward_parity`
//!
//! # Why this file exists
//!
//! Three physics corrections landed with hand-written BURN-0.21
//! `Backward<I, N>` implementations rather than autograd-tape unrolling
//! (invariant 4, `docs/reference/burn-autograd.md`):
//!
//!   1. trapezoidal celerity `c = v·β`, `β = 5/3 − (4/3)·A·√(1+z²)/(T·P)`  (S17)
//!   2. Cunge `X = clamp(0.5(1 − Q/(B·S·c·L)), 0, 0.5)`                    (S19)
//!   3. the positivity clamp — K floor + three-way X cap                   (S18'/S19')
//!
//! Every existing gradcheck for them (`tests/celerity_beta.rs`,
//! `tests/cunge_x.rs`, `tests/positivity_clamp.rs`, `tests/sparse_gradcheck.rs`)
//! declares `type I = NdArray<f32>` — **CPU only**. But `ddr_match: false` +
//! `use_cuda_graphs: false` + a CUDA backend is a legal, actively-used
//! configuration, so those backwards ship un-exercised on the hardware that
//! actually runs them. This file closes that gap.
//!
//! The CUDA-graph path is separately walled off and is NOT this file's problem:
//! `validate_ddr_match` rejects `ddr_match: false` + `use_cuda_graphs: true`,
//! and `validate_enforce_positivity` requires `enforce_positivity ⟹ !ddr_match`,
//! so transitively `enforce_positivity ⟹ !use_cuda_graphs`. That transitive
//! implication is asserted in Part D — it was previously only *implied* by two
//! independent validators and never tested.
//!
//! # Structure
//!
//! * **Part A** — native central-difference gradcheck with `Cuda<f32, i32>` as
//!   the inner backend. This is the load-bearing evidence: it shows CUDA
//!   gradients are *correct*, not merely *consistent with CPU*.
//! * **Part B** — CUDA-vs-CPU analytic gradient parity on an identical fixture.
//! * **Part C** — non-vacuity guards, evaluated *on the CUDA backend*, so a
//!   fixture that went degenerate only on GPU would be caught.
//! * **Part D** — the transitive `enforce_positivity ⟹ !use_cuda_graphs` guard.
//!
//! Every numeric helper below is generic in the inner backend `I` and is called
//! with BOTH `NdArray<f32>` and `Cuda<f32, i32>`. That is deliberate: a parity
//! claim is only meaningful if both sides execute the same source.
//!
//! # Falsifiability
//!
//! This repo has a burned-in lesson: `tests/cunge_x.rs` was originally VACUOUS
//! (1000 m reaches saturated X's clamp on every reach, so all four tests passed
//! with the backward terms DELETED). The guards in Part C are the standing
//! defense, and they run as a PRECONDITION of every gradcheck rather than as
//! standalone tests that a future retune could leave behind.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::mmc_op::{timestep_forward, POSITIVITY_DELTA};
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

/// CPU reference backend.
type Cpu = NdArray<f32>;
/// GPU backend under test. `burn::backend::Cuda` needs the umbrella crate's
/// "cuda" feature; the direct crate is the convention here — see
/// `tests/kan_head_fixture_forward.rs` and `tests/cusparse_ptr_spike.rs`.
type Gpu = burn_cuda::Cuda<f32, i32>;

/// `crate::routing::mmc::DT_SECONDS`, restated so a change there fails loudly
/// here rather than silently retuning the fixture.
const DT: f32 = 3600.0;

fn device<I: Backend>() -> I::Device {
    <I as burn::tensor::backend::BackendTypes>::Device::default()
}

// ===========================================================================
// Fixture — lifted from `tests/positivity_clamp.rs`, unchanged.
//
// The reach lengths span 400 m .. 150 km so `Cr = Δt·c/L` sweeps ~0.05 .. ~7.9:
// the short end sits deep in the `c3 < 0` regime, the long end deep in the
// `c1 < 0` regime, and the middle keeps at least one reach where Cunge X wins
// the three-way min strictly inside `[0, 0.5]`.
// ===========================================================================

struct Fixture {
    length: Vec<f32>,
    slope: Vec<f32>,
    n: Vec<f32>,
    qsp: Vec<f32>,
    psp: Vec<f32>,
    qt: Vec<f32>,
    qpt: Vec<f32>,
}

/// The gradcheck fixture. `q_t` is GRADED along the chain, and that is
/// load-bearing, not decoration — see `tests/positivity_clamp.rs::grad_fixture`:
/// with a flat `q_t`, the partition identity `c1+c2+c3 = 1` on a chain where
/// `x_up ≈ i_t ≈ q_t` makes `q_next ≈ q_t` regardless of `(K, X)`, every
/// S17/S19/S18' effect cancels to f32 noise, and the reaches would be
/// gradient-DEAD. On GPU that would show up as "CUDA and CPU agree perfectly"
/// — both computing nothing.
///
/// `q'` is small (0.01 m³/s) so the `c3 < 0` reaches are not masked by `c4·q'`.
fn grad_fixture() -> Fixture {
    let length = vec![
        400.0f32, 800.0, 1500.0, 2500.0, 3600.0, 5000.0, 8000.0, 20000.0, 60000.0, 150000.0,
    ];
    let n = length.len();
    Fixture {
        length,
        slope: vec![0.001; n],
        n: vec![0.035; n],
        qsp: vec![0.4; n],
        psp: vec![20.0; n],
        qt: vec![6.0, 8.0, 15.0, 50.0, 120.0, 200.0, 300.0, 40.0, 400.0, 30.0],
        qpt: vec![0.01; n],
    }
}

/// `enforce_positivity` is the only knob: `false` gives β celerity + Cunge X,
/// `true` adds S18'/S19'. There is deliberately no config where β is on and
/// Cunge X is off — both are gated on the single `ddr_match: false` flag — so
/// two configs cover all three patterns.
fn stress_cfg(enforce_positivity: bool) -> Config {
    let mut cfg = Config::default();
    cfg.params.ddr_match = false;
    cfg.params.enforce_positivity = enforce_positivity;
    cfg.params.parameter_ranges.n = [0.01, 0.3];
    cfg.params.parameter_ranges.q_spatial = [0.1, 0.9];
    cfg.params.parameter_ranges.p_spatial = [1.0, 200.0];
    cfg.params.attribute_minimums.velocity = 0.01;
    cfg.params.attribute_minimums.depth = 0.001;
    cfg.params.attribute_minimums.discharge = 1e-4;
    cfg.params.attribute_minimums.bottom_width = 0.01;
    cfg.params.attribute_minimums.slope = 0.0001;
    cfg.params.defaults.insert("p_spatial".to_string(), 1.0);
    cfg.params.log_space_parameters = vec![];
    cfg
}

fn chain(f: &Fixture) -> SparseAdjacency {
    let n = f.length.len();
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, f.length.clone(), f.slope.clone())
}

/// The S1..S23 quantities the non-vacuity guards need, read out of the REAL
/// forward chain on backend `I` (not recomputed from a parallel model).
struct ChainOutputs {
    side_slope: Vec<f32>,
    velocity_clamped: Vec<f32>,
    celerity: Vec<f32>,
    k_muskingum: Vec<f32>,
    top_width: Vec<f32>,
    c1: Vec<f32>,
    c2: Vec<f32>,
    c3: Vec<f32>,
}

fn run_chain<I: Backend + 'static>(f: &Fixture, enforce_positivity: bool) -> ChainOutputs
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let adj = chain(f);
    let dev = device::<I>();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let mk = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &dev) };
    let outs = ddrs::routing::mmc_op::__spike_forward_chain_k1_outputs::<I>(
        &stress_cfg(enforce_positivity),
        &pattern,
        mk(&f.n),
        mk(&f.qsp),
        mk(&f.psp),
        mk(&f.qt),
        mk(&f.qpt),
        mk(&f.length),
        mk(&f.slope),
        mk(&vec![0.3f32; f.length.len()]),
    );
    // k1 output order: [depth, top_width, side_slope, bottom_width, hyd_radius,
    //                   velocity_un, velocity_cl, celerity, k_muskingum, denom,
    //                   c1, c2, c3, c4, ...]
    ChainOutputs {
        top_width: outs[1].clone(),
        side_slope: outs[2].clone(),
        velocity_clamped: outs[6].clone(),
        celerity: outs[7].clone(),
        k_muskingum: outs[8].clone(),
        c1: outs[10].clone(),
        c2: outs[11].clone(),
        c3: outs[12].clone(),
    }
}

fn run_solve<I: Backend + 'static>(f: &Fixture, enforce_positivity: bool) -> Vec<f32>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let adj = chain(f);
    let dev = device::<I>();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let mk = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &dev) };
    let (_b, _i, x_sol, _q) = ddrs::routing::mmc_op::__spike_forward_chain_k23_outputs::<I>(
        &stress_cfg(enforce_positivity),
        &pattern,
        mk(&f.n),
        mk(&f.qsp),
        mk(&f.psp),
        mk(&f.qt),
        mk(&f.qpt),
        mk(&f.length),
        mk(&f.slope),
        mk(&vec![0.3f32; f.length.len()]),
    );
    x_sol
}

// ===========================================================================
// Non-vacuity instrumentation (shared by Part A and Part C)
// ===========================================================================

/// Which branch of `x_eff = min(x_cunge, hi_a, hi_b)` won on a reach. The
/// tie-break cascade `Cunge > hi_a > hi_b` mirrors the backward's mask cascade
/// in `mmc_op.rs` B19' exactly.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Branch {
    Cunge,
    HiA,
    HiB,
}

struct Report {
    branches: Vec<Branch>,
    floored: Vec<bool>,
    cr_raw: Vec<f32>,
    x_cunge: Vec<f32>,
    /// `β = c / v_clamped`, recovered from the chain outputs.
    beta: Vec<f32>,
    side_slope: Vec<f32>,
}

fn report<I: Backend + 'static>(f: &Fixture, enforce_positivity: bool) -> Report
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let out = run_chain::<I>(f, enforce_positivity);
    let k_floor = DT * (1.0 + POSITIVITY_DELTA) / 2.0;
    let mut r = Report {
        branches: Vec::new(),
        floored: Vec::new(),
        cr_raw: Vec::new(),
        x_cunge: Vec::new(),
        beta: Vec::new(),
        side_slope: out.side_slope.clone(),
    };
    for i in 0..f.length.len() {
        let k_raw = f.length[i] / out.celerity[i];
        r.floored.push(k_raw < k_floor);
        r.cr_raw.push(DT / k_raw);
        r.beta.push(out.celerity[i] / out.velocity_clamped[i]);
        let w = f.qt[i] / (out.top_width[i] * f.slope[i] * out.celerity[i] * f.length[i] + 1e-12);
        let x_cunge = (0.5 * (1.0 - w)).clamp(0.0, 0.5);
        r.x_cunge.push(x_cunge);
        // `cr` uses the POST-floor K (S18'), matching how the forward composes
        // S18' into S19'.
        let cr = DT / out.k_muskingum[i];
        let hi_a = cr * 0.5 * (1.0 - POSITIVITY_DELTA);
        let hi_b = (1.0 - 0.5 * cr) * (1.0 - POSITIVITY_DELTA);
        r.branches.push(if x_cunge <= hi_a && x_cunge <= hi_b {
            Branch::Cunge
        } else if hi_a <= hi_b {
            Branch::HiA
        } else {
            Branch::HiB
        });
    }
    r
}

/// STANDING GUARD, run as a PRECONDITION of every gradcheck below.
///
/// Five independent ways this file could go vacuous:
///
/// 1. `β ≡ 5/3` (or constant) → the S17 backward terms `∂β/∂A,T,P,z` carry no
///    signal and could be deleted;
/// 2. `side_slope` saturated on its `[0.5, 50]` clamp everywhere → `∂β/∂z` is
///    masked to zero on every reach;
/// 3. every Cunge-branch win sitting on X's own `[0, 0.5]` clamp → exactly the
///    failure that made `tests/cunge_x.rs` vacuous at 1000 m;
/// 4. one branch of the three-way min always winning → the other two masks are
///    never exercised (positivity config only);
/// 5. every reach on the same side of the K floor → S18' is a no-op or a
///    constant (positivity config only).
fn assert_fixture_exercises_everything<I: Backend + 'static>(label: &str, enforce_positivity: bool)
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let f = grad_fixture();
    let r = report::<I>(&f, enforce_positivity);
    println!("--- non-vacuity [{label}] enforce_positivity={enforce_positivity} ---");
    for i in 0..r.branches.len() {
        println!(
            "  [{i}] L={:>8.0} q_t={:>6.1} beta={:.4} z={:>8.4} Cr_raw={:>7.3} \
             floored={:<5} x_cunge={:.4} branch={:?}",
            f.length[i],
            f.qt[i],
            r.beta[i],
            r.side_slope[i],
            r.cr_raw[i],
            r.floored[i],
            r.x_cunge[i],
            r.branches[i]
        );
    }

    // (1) β must be genuinely trapezoidal and must VARY. `5/3` is the
    //     wide-rectangular limit; a fixture sitting there tests nothing.
    let beta_min = r.beta.iter().cloned().fold(f32::INFINITY, f32::min);
    let beta_max = r.beta.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  beta span: {beta_min:.4} .. {beta_max:.4}  (5/3 = {:.4})", 5.0 / 3.0);
    assert!(
        beta_max < 5.0 / 3.0 - 0.05,
        "[{label}] beta ~ 5/3 everywhere ({beta_max:.4}) — the S17 correction is untested"
    );
    assert!(
        beta_max - beta_min > 1e-3,
        "[{label}] beta is constant ({beta_min:.5}..{beta_max:.5}) — \
         d(beta)/d(A,T,P,z) carry no signal"
    );

    // (2) `∂β/∂z` is masked wherever S9's clamp saturates.
    let z_interior = r
        .side_slope
        .iter()
        .filter(|&&z| z > 0.5 + 1e-3 && z < 50.0 - 1e-3)
        .count();
    assert!(
        z_interior > 0,
        "[{label}] side_slope saturated on [0.5, 50] on every reach — d(beta)/dz is masked off"
    );

    // (3) Cunge X must be strictly interior somewhere, or `gX` is masked to
    //     zero and the S19 terms could be deleted (the cunge_x.rs trap).
    let x_interior = r.x_cunge.iter().filter(|&&x| x > 0.02 && x < 0.48).count();
    assert!(
        x_interior > 0,
        "[{label}] every x_cunge sits on its own [0, 0.5] clamp — same vacuity trap as cunge_x.rs"
    );

    if !enforce_positivity {
        return;
    }

    // (4) three-way min branch mix.
    let n = r.branches.len() as f32;
    let frac = |b: Branch| r.branches.iter().filter(|&&x| x == b).count() as f32 / n;
    let (fc, fa, fb) = (frac(Branch::Cunge), frac(Branch::HiA), frac(Branch::HiB));
    println!("  branch mix: cunge={fc:.2} hi_a={fa:.2} hi_b={fb:.2}");
    assert!(
        fc > 0.05 && fa > 0.05 && fb > 0.05,
        "[{label}] fixture is vacuous: branch mix cunge={fc:.2} hi_a={fa:.2} hi_b={fb:.2}"
    );

    // (5) the K floor must BIND on some reaches and not on others.
    let f_floored = r.floored.iter().filter(|&&x| x).count() as f32 / n;
    println!("  K floored fraction: {f_floored:.2}");
    assert!(
        (0.05..0.95).contains(&f_floored),
        "[{label}] fixture must STRADDLE the K floor, got {f_floored:.2}"
    );

    // The clamp must actually be doing work: with it ON no coefficient may be
    // negative, and with it OFF the same fixture must violate both bounds.
    let on = run_chain::<I>(&f, true);
    let off = run_chain::<I>(&f, false);
    let min_of = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "  min c1/c3: clamp ON ({:e}, {:e})  OFF ({:e}, {:e})",
        min_of(&on.c1),
        min_of(&on.c3),
        min_of(&off.c1),
        min_of(&off.c3)
    );
    assert!(
        min_of(&on.c1) >= 0.0 && min_of(&on.c3) >= 0.0,
        "[{label}] clamp ON still produced a negative coefficient"
    );
    assert!(
        min_of(&off.c1) < 0.0 && min_of(&off.c3) < 0.0,
        "[{label}] fixture must violate BOTH bounds when unclamped — the clamp is untested"
    );
    // Mass conservation survives (the reason the clamp targets K/X, not c1/c3).
    for i in 0..on.c1.len() {
        let s = on.c1[i] + on.c2[i] + on.c3[i];
        assert!(
            (s - 1.0).abs() < 1e-5,
            "[{label}] reach {i}: c1+c2+c3 = {s} on the clamped path"
        );
    }
}

// ===========================================================================
// Gradient machinery — generic in the inner backend.
// ===========================================================================

#[derive(Copy, Clone, Debug)]
enum Parent {
    N,
    QSpatial,
    PSpatial,
    QT,
}

impl Parent {
    fn name(self) -> &'static str {
        match self {
            Parent::N => "n",
            Parent::QSpatial => "q_spatial",
            Parent::PSpatial => "p_spatial",
            Parent::QT => "q_t",
        }
    }
}

const ALL_PARENTS: [Parent; 4] = [Parent::N, Parent::QSpatial, Parent::PSpatial, Parent::QT];

/// FD step as a pure FRACTION of the base value.
///
/// An ABSOLUTE floor (as the sibling gradchecks use) would be a 2.9%
/// perturbation of `n = 0.035`, large enough to move a reach across the
/// `x_cunge` / `hi_b` min boundary — central differences would then average two
/// different slopes and disagree with the analytical gradient by ~30%, an FD
/// artifact rather than a backward bug. Matches `tests/positivity_clamp.rs`.
const REL_STEP: f32 = 3e-3;
/// FD-vs-analytic tolerance. Same value as the CPU gradchecks; NOT widened for
/// CUDA — see `NOISE_ULPS` for how GPU round-off is accounted for instead.
const REL_TOL: f32 = 5e-3;

/// Ulps of the loss allowed for accumulated round-off in the weighted sum plus
/// the FD subtraction.
///
/// `tests/positivity_clamp.rs` uses 16 on CPU. 64 here because the GPU's
/// transcendental ops are looser than glibc's: cubecl lowers `powf(x, y)` to
/// `exp2(y·log2(x))` on hardware SFU instructions with ~2 ulp each, versus
/// libm's ~0.5-1 ulp, and the S1..S17 chain contains 4 `powf`/`powf_scalar`,
/// 2 `sqrt` and 2 `recip`. 4x the CPU allowance is a conservative bound on
/// that (it would cover ~8 ulp of extra error per transcendental).
///
/// This constant only *widens the band in which a disagreement is forgiven as
/// unmeasurable*; it does NOT relax `REL_TOL` for reaches whose gradient is
/// above the floor, and `compare_grads` fails outright if fewer than 4 reaches
/// remain above it. So it cannot be abused to make a broken backward pass —
/// it would instead trip the `resolved >= 4` power check.
const NOISE_ULPS: f32 = 64.0;

struct GradTensors<I: Backend> {
    n: Tensor<Autodiff<I>, 1>,
    qsp: Tensor<Autodiff<I>, 1>,
    psp: Tensor<Autodiff<I>, 1>,
    qt: Tensor<Autodiff<I>, 1>,
}

/// Per-reach loss weights `w[i] = 1/max(q_next_base[i], 1)`, so every reach
/// contributes O(1) to `loss = Σ w[i]·q_next[i]` instead of the O(180) the
/// downstream reaches would otherwise contribute. Without this, `l₊ − l₋` at a
/// 3e-3 relative step lands at the f32 round-off floor of an O(600) loss.
///
/// Computed ONCE on the CPU backend and reused for both backends: the weights
/// must be bit-identical across backends or Part B would be comparing two
/// different loss functions.
fn conditioning_weights(enforce_positivity: bool) -> Vec<f32> {
    let base = run_solve::<Cpu>(&grad_fixture(), enforce_positivity);
    base.iter().map(|&q| 1.0 / q.max(1.0)).collect()
}

#[allow(clippy::too_many_arguments)]
fn run_forward_loss<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    assembler: &AValuesAssembler<I>,
    dev: &I::Device,
    n_vec: &[f32],
    qsp_vec: &[f32],
    psp_vec: &[f32],
    qt_vec: &[f32],
    qpt_vec: &[f32],
    length_vec: &[f32],
    slope_vec: &[f32],
    weights: &[f32],
    require_grad_parent: Option<Parent>,
) -> (Tensor<Autodiff<I>, 1>, GradTensors<I>)
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let mk = |data: &[f32], req: bool| -> Tensor<Autodiff<I>, 1> {
        let t: Tensor<Autodiff<I>, 1> = Tensor::from_floats(data, dev);
        if req {
            t.require_grad()
        } else {
            t
        }
    };
    let n_t = mk(n_vec, matches!(require_grad_parent, Some(Parent::N)));
    let qsp_t = mk(qsp_vec, matches!(require_grad_parent, Some(Parent::QSpatial)));
    let psp_t = mk(psp_vec, matches!(require_grad_parent, Some(Parent::PSpatial)));
    let qt_t = mk(qt_vec, matches!(require_grad_parent, Some(Parent::QT)));
    let qpt_t = mk(qpt_vec, false);
    let length_t = mk(length_vec, false);
    let slope_t = mk(slope_vec, false);
    let xst_t = mk(&vec![0.3f32; n_vec.len()], false);

    let q_next = timestep_forward::<I>(
        cfg,
        pattern,
        assembler,
        n_t.clone(),
        qsp_t.clone(),
        psp_t.clone(),
        qt_t.clone(),
        qpt_t.clone(),
        length_t,
        slope_t,
        xst_t,
        false,
    );

    let loss = q_next * mk(weights, false);

    (
        loss,
        GradTensors {
            n: n_t,
            qsp: qsp_t,
            psp: psp_t,
            qt: qt_t,
        },
    )
}

fn analytical_grad<I: Backend + 'static>(parent: Parent, enforce_positivity: bool) -> Vec<f32>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    analytical_grad_cfg::<I>(&stress_cfg(enforce_positivity), parent, enforce_positivity)
}

fn analytical_grad_cfg<I: Backend + 'static>(
    cfg: &Config,
    parent: Parent,
    enforce_positivity: bool,
) -> Vec<f32>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let f = grad_fixture();
    let adj = chain(&f);
    let dev = device::<I>();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let weights = conditioning_weights(enforce_positivity);

    let (loss, parents) = run_forward_loss::<I>(
        cfg, &pattern, &assembler, &dev, &f.n, &f.qsp, &f.psp, &f.qt, &f.qpt, &f.length, &f.slope,
        &weights, Some(parent),
    );

    let grads = loss.sum().backward();
    let g = match parent {
        Parent::N => parents.n.grad(&grads).expect("grad on n"),
        Parent::QSpatial => parents.qsp.grad(&grads).expect("grad on q_spatial"),
        Parent::PSpatial => parents.psp.grad(&grads).expect("grad on p_spatial"),
        Parent::QT => parents.qt.grad(&grads).expect("grad on q_t"),
    };
    g.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

/// `(fd_grad, fd_noise_floor)`. The second vector is the smallest gradient
/// difference f32 central differences can resolve at each reach's step size:
/// `NOISE_ULPS · ulp(loss) / (2·eps_i)`. Anything below it is unmeasurable, not
/// evidence about the backward.
fn fd_grad<I: Backend + 'static>(
    parent: Parent,
    enforce_positivity: bool,
) -> (Vec<f32>, Vec<f32>)
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    fd_grad_cfg::<I>(&stress_cfg(enforce_positivity), parent, enforce_positivity)
}

fn fd_grad_cfg<I: Backend + 'static>(
    cfg: &Config,
    parent: Parent,
    enforce_positivity: bool,
) -> (Vec<f32>, Vec<f32>)
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let f = grad_fixture();
    let adj = chain(&f);
    let dev = device::<I>();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &dev);
    let weights = conditioning_weights(enforce_positivity);

    let eval_loss = |n: &[f32], qsp: &[f32], psp: &[f32], qt: &[f32]| -> f32 {
        let (loss, _) = run_forward_loss::<I>(
            cfg, &pattern, &assembler, &dev, n, qsp, psp, qt, &f.qpt, &f.length, &f.slope,
            &weights, None,
        );
        loss.sum().into_data().convert::<f32>().into_vec::<f32>().unwrap()[0]
    };
    let loss_mag = eval_loss(&f.n, &f.qsp, &f.psp, &f.qt).abs().max(1.0);

    let n_reach = f.length.len();
    let mut grad = vec![0.0f32; n_reach];
    let mut noise = vec![0.0f32; n_reach];
    for i in 0..n_reach {
        let (mut pn, mut pq, mut pp, mut pt) =
            (f.n.clone(), f.qsp.clone(), f.psp.clone(), f.qt.clone());
        let (mut mn, mut mq, mut mp, mut mt) =
            (f.n.clone(), f.qsp.clone(), f.psp.clone(), f.qt.clone());
        let (plus, minus, base) = match parent {
            Parent::N => (&mut pn, &mut mn, &f.n),
            Parent::QSpatial => (&mut pq, &mut mq, &f.qsp),
            Parent::PSpatial => (&mut pp, &mut mp, &f.psp),
            Parent::QT => (&mut pt, &mut mt, &f.qt),
        };
        let eps = REL_STEP * base[i].abs();
        assert!(eps > 0.0, "reach {i}: zero FD step (base = {})", base[i]);
        plus[i] = base[i] + eps;
        minus[i] = base[i] - eps;
        grad[i] = (eval_loss(&pn, &pq, &pp, &pt) - eval_loss(&mn, &mq, &mp, &mt)) / (2.0 * eps);
        noise[i] = NOISE_ULPS * loss_mag * f32::EPSILON / (2.0 * eps);
    }
    (grad, noise)
}

/// `(worst_rel, worst_abs, resolved, pass)`. A reach passes when it either
/// agrees to `REL_TOL` or disagrees by less than what f32 central differences
/// can resolve at that step size — the second clause is a statement about
/// measurability, not a relaxed tolerance.
///
/// Split out from `compare_grads` so the negative control can read the verdict
/// without the panic.
fn gradcheck_verdict(
    name: &str,
    analytical: &[f32],
    fd: &[f32],
    noise: &[f32],
    verbose: bool,
) -> (f32, f32, usize, bool) {
    assert_eq!(analytical.len(), fd.len());
    if verbose {
        println!("--- {name} ---");
    }
    let (mut worst_rel, mut worst_abs) = (0.0f32, 0.0f32);
    let mut resolved = 0usize;
    let mut pass = true;
    for i in 0..analytical.len() {
        let (a, d) = (analytical[i], fd[i]);
        let abs_diff = (a - d).abs();
        let rel_diff = abs_diff / a.abs().max(d.abs()).max(1e-12);
        let informative = a.abs().max(d.abs()) > noise[i];
        if informative {
            resolved += 1;
            worst_abs = worst_abs.max(abs_diff);
            worst_rel = worst_rel.max(rel_diff);
        }
        if !(rel_diff < REL_TOL || abs_diff < noise[i]) {
            pass = false;
        }
        if verbose {
            println!(
                "  [{i}] analytical={a:.6e}  fd={d:.6e}  abs={abs_diff:.3e}  rel={rel_diff:.3e}  \
                 fd_noise={:.3e}{}",
                noise[i],
                if informative { "" } else { "  (below FD resolution)" }
            );
        }
    }
    if verbose {
        println!("  resolved reaches: {resolved}/{}", analytical.len());
        println!("  worst abs={worst_abs:.3e}  worst rel={worst_rel:.3e}");
    }
    (worst_rel, worst_abs, resolved, pass)
}

fn compare_grads(name: &str, analytical: &[f32], fd: &[f32], noise: &[f32]) {
    let (worst_rel, worst_abs, resolved, pass) =
        gradcheck_verdict(name, analytical, fd, noise, true);
    assert!(
        resolved >= 4,
        "{name}: only {resolved} reaches are above the FD noise floor — \
         the gradcheck has no power left"
    );
    assert!(
        pass,
        "{name}: gradcheck failed (worst rel={worst_rel:.3e}, abs={worst_abs:.3e})"
    );
}

// ===========================================================================
// Part A — native finite-difference gradcheck ON CUDA.
//
// The strongest evidence in this file: the analytical CUDA gradient is checked
// against central differences of the CUDA forward. This tests CORRECTNESS, not
// merely agreement with the CPU backend (two backends can be identically
// wrong, e.g. if a mask were inverted in shared source).
//
// Two configs cover all three physics patterns:
//   * enforce_positivity=false -> beta celerity (S17) + Cunge X (S19)
//   * enforce_positivity=true  -> the above + the positivity clamp (S18'/S19')
// ===========================================================================

fn cuda_gradcheck(parent: Parent, enforce_positivity: bool) {
    let label = if enforce_positivity {
        "beta + cunge-X + positivity clamp"
    } else {
        "beta + cunge-X"
    };
    assert_fixture_exercises_everything::<Gpu>("cuda", enforce_positivity);
    let (fd, noise) = fd_grad::<Gpu>(parent, enforce_positivity);
    let analytic = analytical_grad::<Gpu>(parent, enforce_positivity);
    compare_grads(
        &format!("CUDA FD gradcheck: {} [{label}]", parent.name()),
        &analytic,
        &fd,
        &noise,
    );
}

#[test]
fn cuda_gradcheck_beta_and_cunge_x_n() {
    cuda_gradcheck(Parent::N, false);
}

#[test]
fn cuda_gradcheck_beta_and_cunge_x_q_spatial() {
    cuda_gradcheck(Parent::QSpatial, false);
}

#[test]
fn cuda_gradcheck_beta_and_cunge_x_p_spatial() {
    cuda_gradcheck(Parent::PSpatial, false);
}

/// `q_t` carries the Cunge `∂X/∂Q` term on top of its pre-existing S25 RHS,
/// S24 SpMV and S2 depth paths — the most heavily multiplexed parent.
#[test]
fn cuda_gradcheck_beta_and_cunge_x_q_t() {
    cuda_gradcheck(Parent::QT, false);
}

#[test]
fn cuda_gradcheck_positivity_clamp_n() {
    cuda_gradcheck(Parent::N, true);
}

#[test]
fn cuda_gradcheck_positivity_clamp_q_spatial() {
    cuda_gradcheck(Parent::QSpatial, true);
}

#[test]
fn cuda_gradcheck_positivity_clamp_p_spatial() {
    cuda_gradcheck(Parent::PSpatial, true);
}

/// Under the clamp, `q_t`'s Cunge path is switched OFF on every reach where
/// `hi_a`/`hi_b` win the three-way min — the mask cascade B19' is at its most
/// load-bearing here.
#[test]
fn cuda_gradcheck_positivity_clamp_q_t() {
    cuda_gradcheck(Parent::QT, true);
}

// ===========================================================================
// Part B — CUDA-vs-CPU analytic gradient parity.
// ===========================================================================

/// Cross-backend relative tolerance for the ANALYTIC gradient.
///
/// Derived, not tuned:
///
/// * The two backends run the SAME source, so any difference is pure op-level
///   round-off. The gradient of one reach is a product/sum of roughly 25 f32
///   chain-rule factors. Under linear error propagation with ≤1 ulp
///   (`f32::EPSILON ≈ 1.2e-7`) per elementary op and no catastrophic
///   cancellation — verified: every component compared here is O(1e-3)..O(25),
///   none is a near-cancelling difference — the worst-case accumulated relative
///   error is bounded by `25 · 1.2e-7 ≈ 3e-6`.
/// * `1e-5` is ~3x that analytic bound, which absorbs the extra slack in the
///   two transcendentals the BACKWARD adds on top of the forward (`ratio.log()`
///   in B6, `powf` in B15): cubecl lowers these onto the SFU (~2 ulp) whereas
///   `NdArray` calls libm (~0.5-1 ulp).
/// * MEASURED on this fixture: worst = 2.784e-7 over all four parents and both
///   configs (and exactly 0.0 for two of the four). So the tolerance sits ~36x
///   above observation — enough headroom for a driver or architecture change,
///   and 36x is small enough that it is still an assertion rather than a
///   formality. `negative_control_*` below proves it: a 1e-4 relative
///   perturbation — 360x smaller than any structural error could be — trips it.
/// * Sharpness: a wrong branch mask, a dropped `∂β/∂z` term or an inverted
///   K-floor mask changes the gradient by O(10%)-O(100%) on the affected
///   reaches, i.e. 1e4-1e5 times this tolerance.
const XBACKEND_REL_TOL: f32 = 1e-5;

/// Reaches whose |gradient| is below this (relative to the largest component)
/// are excluded from the relative comparison — a relative metric on a component
/// that is ~0 on both backends is meaningless, and an ABSOLUTE check catches
/// the real failure there (one backend producing 0 where the other does not).
const XBACKEND_REL_FLOOR: f32 = 1e-6;

/// Worst relative disagreement between two gradient vectors, restricted to the
/// components that carry signal. Also asserts the ZERO PATTERN matches: a
/// component that is exactly 0 on one backend and not the other is a masking
/// disagreement, never round-off.
///
/// Split out from `compare_backends` so the negative controls can call the
/// metric without the tolerance assertion.
fn worst_backend_rel(name: &str, cpu: &[f32], gpu: &[f32], verbose: bool) -> f32 {
    assert_eq!(cpu.len(), gpu.len());
    let scale = cpu
        .iter()
        .chain(gpu.iter())
        .fold(0.0f32, |m, v| m.max(v.abs()));
    if verbose {
        println!("--- {name} (scale = {scale:.6e}) ---");
    }
    let mut worst_rel = 0.0f32;
    let mut compared = 0usize;
    for i in 0..cpu.len() {
        let abs_diff = (cpu[i] - gpu[i]).abs();
        let denom = cpu[i].abs().max(gpu[i].abs());
        let rel = abs_diff / denom.max(1e-30);
        let significant = denom > XBACKEND_REL_FLOOR * scale;
        // A component that is EXACTLY zero on one backend and not the other is
        // a masking disagreement, not round-off — always a failure.
        assert!(
            (cpu[i] == 0.0) == (gpu[i] == 0.0),
            "{name} reach {i}: gradient is zero on exactly one backend \
             (cpu={:e}, gpu={:e}) — a mask disagrees across backends",
            cpu[i],
            gpu[i]
        );
        if significant {
            compared += 1;
            worst_rel = worst_rel.max(rel);
        }
        if verbose {
            println!(
                "  [{i}] cpu={:.7e}  gpu={:.7e}  abs={abs_diff:.3e}  rel={rel:.3e}{}",
                cpu[i],
                gpu[i],
                if significant { "" } else { "  (below relative floor)" }
            );
        }
    }
    if verbose {
        println!("  compared {compared}/{} components, worst rel = {worst_rel:.3e}", cpu.len());
    }
    assert!(
        compared >= 4,
        "{name}: only {compared} components are above the relative floor — no power left"
    );
    worst_rel
}

fn compare_backends(name: &str, cpu: &[f32], gpu: &[f32]) -> f32 {
    let worst_rel = worst_backend_rel(name, cpu, gpu, true);
    assert!(
        worst_rel < XBACKEND_REL_TOL,
        "{name}: CUDA/CPU analytic gradient parity failed, worst rel = {worst_rel:.3e} \
         (tol {XBACKEND_REL_TOL:.1e})"
    );
    worst_rel
}

fn parity_for(enforce_positivity: bool) -> f32 {
    assert_fixture_exercises_everything::<Cpu>("cpu", enforce_positivity);
    assert_fixture_exercises_everything::<Gpu>("cuda", enforce_positivity);
    let mut worst = 0.0f32;
    for p in ALL_PARENTS {
        let cpu = analytical_grad::<Cpu>(p, enforce_positivity);
        let gpu = analytical_grad::<Gpu>(p, enforce_positivity);
        worst = worst.max(compare_backends(
            &format!(
                "d(loss)/d({}) [enforce_positivity={enforce_positivity}]",
                p.name()
            ),
            &cpu,
            &gpu,
        ));
    }
    println!("WORST cross-backend rel (enforce_positivity={enforce_positivity}) = {worst:.3e}");
    worst
}

#[test]
fn analytic_gradients_match_across_backends_beta_and_cunge_x() {
    parity_for(false);
}

#[test]
fn analytic_gradients_match_across_backends_positivity_clamp() {
    parity_for(true);
}

/// The production CUDA configuration is `sparse_solver: cuda`, which swaps the
/// host-side forward substitution for the cuSPARSE triangular solve — a
/// DIFFERENT `Backward` impl in `src/sparse/` sitting directly downstream of
/// the new physics terms. Everything above runs with the default
/// `sparse_solver: cpu` (`Config::default()`), so without this test the new
/// backwards would still be unverified in the combination that actually ships.
///
/// Checked two ways: analytic-vs-FD **entirely inside the cuSPARSE path**
/// (so a solver-side gradient error cannot be cancelled by a matching forward
/// error), and analytic-vs-analytic against the host-solve path.
///
/// The cuSPARSE path is provably taken rather than silently falling back:
/// `dispatch::effective_use_cuda::<B>` degrades to CPU on exactly one
/// condition — `B != Cuda<f32, i32>` — which is a compile-time-known `TypeId`
/// comparison, true here; and `cusparse_forward` itself has no fallback (it
/// `.expect()`s every cuSPARSE call). Note the observed cuSPARSE-vs-host
/// gradients are BIT-identical on this fixture: the chain has one off-diagonal
/// per row, so forward substitution reduces to the same `b[i] − a·x[i−1]` in
/// both implementations. The FD half of this test is therefore the one
/// carrying the discriminating power.
#[test]
fn cusparse_solver_path_carries_the_same_gradients() {
    let mut cfg = stress_cfg(true);
    cfg.params.sparse_solver = ddrs::config::SparseSolver::Cuda;

    for p in ALL_PARENTS {
        let (fd, noise) = fd_grad_cfg::<Gpu>(&cfg, p, true);
        let analytic = analytical_grad_cfg::<Gpu>(&cfg, p, true);
        compare_grads(
            &format!("cuSPARSE FD gradcheck: {} [positivity clamp]", p.name()),
            &analytic,
            &fd,
            &noise,
        );
        // Same physics, different triangular solve: the two must agree to the
        // cross-backend tolerance, since both run on the same device and only
        // the solve implementation differs.
        let host_solve = analytical_grad::<Gpu>(p, true);
        compare_backends(
            &format!("cuSPARSE vs host-solve: d(loss)/d({})", p.name()),
            &host_solve,
            &analytic,
        );
    }
}

/// NEGATIVE CONTROL for Part B. A parity test that passes because the
/// tolerance is loose is worthless, so this pins the tolerance's SHARPNESS:
/// inject a 1e-4 relative error into a single significant component of the
/// CUDA gradient and confirm the harness rejects it.
///
/// 1e-4 is deliberately far smaller than any structural defect could produce
/// (a dropped or mis-masked backward term moves a reach by 10%-100%), so
/// passing this control means the parity test would catch a real divergence
/// with ~3 orders of magnitude to spare.
#[test]
fn negative_control_backend_parity_rejects_a_tiny_perturbation() {
    let cpu = analytical_grad::<Cpu>(Parent::QT, true);
    let idx = (0..cpu.len())
        .max_by(|&a, &b| cpu[a].abs().partial_cmp(&cpu[b].abs()).unwrap())
        .unwrap();
    let mut perturbed = cpu.clone();
    perturbed[idx] *= 1.0 + 1e-4;
    let worst = worst_backend_rel("negative control (perturbed)", &cpu, &perturbed, false);
    println!(
        "negative control: perturbing reach {idx} by 1e-4 gives worst rel = {worst:.3e} \
         (tol {XBACKEND_REL_TOL:.1e})"
    );
    assert!(
        worst >= XBACKEND_REL_TOL,
        "the cross-backend tolerance {XBACKEND_REL_TOL:.1e} is too loose to detect a \
         1e-4 relative perturbation (measured {worst:.3e})"
    );
    // And the zero-pattern assertion inside `worst_backend_rel` must fire when
    // a mask is dropped on one backend only — the failure mode that matters
    // most, since a mask error zeroes a whole reach rather than nudging it.
    let mut zeroed = cpu.clone();
    zeroed[idx] = 0.0;
    let caught = std::panic::catch_unwind(|| {
        worst_backend_rel("negative control (masked)", &cpu, &zeroed, false)
    })
    .is_err();
    assert!(caught, "a gradient zeroed on exactly one backend must be rejected");
}

/// NEGATIVE CONTROL for Part A. Same idea one level up: perturb the ANALYTIC
/// gradient and confirm the FD comparison rejects it.
///
/// The perturbation is 2%, chosen to sit just above the `REL_TOL` band while
/// still being an order of magnitude below what a deleted backward term does.
/// It is applied to the reach with the largest gradient, which is also the one
/// furthest above the FD noise floor — i.e. the harness cannot dodge it via
/// the `abs_diff < noise` escape clause.
#[test]
fn negative_control_fd_gradcheck_rejects_a_two_percent_error() {
    let (fd, noise) = fd_grad::<Gpu>(Parent::QT, true);
    let analytic = analytical_grad::<Gpu>(Parent::QT, true);
    let (_, _, _, clean_pass) =
        gradcheck_verdict("negative control (clean)", &analytic, &fd, &noise, false);
    assert!(clean_pass, "the unperturbed CUDA gradcheck must pass first");

    let idx = (0..analytic.len())
        .max_by(|&a, &b| analytic[a].abs().partial_cmp(&analytic[b].abs()).unwrap())
        .unwrap();
    let mut broken = analytic.clone();
    broken[idx] *= 1.02;
    let (worst, _, _, pass) =
        gradcheck_verdict("negative control (broken)", &broken, &fd, &noise, false);
    println!(
        "negative control: 2% error on reach {idx} gives worst rel = {worst:.3e} — \
         gradcheck pass = {pass}"
    );
    assert!(
        !pass,
        "a 2% analytic-gradient error slipped through the FD gradcheck \
         (worst rel = {worst:.3e}, REL_TOL = {REL_TOL:.1e})"
    );
}

/// Forward parity anchor: if the two backends' FORWARDS disagreed materially,
/// the gradient parity above would be comparing derivatives at two different
/// operating points and its tolerance would be uninterpretable.
#[test]
fn forward_matches_across_backends() {
    for enforce in [false, true] {
        let f = grad_fixture();
        let cpu = run_solve::<Cpu>(&f, enforce);
        let gpu = run_solve::<Gpu>(&f, enforce);
        let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = cpu
            .iter()
            .zip(&gpu)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "x_sol forward [enforce_positivity={enforce}]: max abs diff = {worst:.3e} \
             (scale {scale:.3e}, rel {:.3e})",
            worst / scale
        );
        assert!(
            worst / scale < 1e-5,
            "forward diverges across backends: {worst:.3e} on scale {scale:.3e}"
        );
    }
}

// ===========================================================================
// Part C — CUDA-side sanity of the individual tensor ops the new code uses.
//
// The new forward/backward paths lean on: `min_pair`, `clamp`, `clamp_min`,
// `greater_elem`, `lower_elem`, `lower_equal`, `bool_and`, `bool_not`,
// `mask_fill`, `powf`, `powf_scalar`, `recip`, `sqrt`. Part A/B exercise them
// in composition; this test pins each one INDIVIDUALLY on CUDA so that a
// failure there localises immediately instead of surfacing as a mysterious
// gradient discrepancy.
//
// The values are chosen to hit the exact edges the physics hits: `min_pair`
// with ties, `clamp` saturating on both ends, `greater_elem` / `lower_elem`
// exactly ON the threshold (strict inequality — a `>=` would flip the mask),
// and `powf` with a non-integer exponent.
// ===========================================================================

#[test]
fn cuda_tensor_ops_used_by_the_new_physics_are_correct() {
    let dev = device::<Gpu>();
    let mk = |d: &[f32]| -> Tensor<Gpu, 1> { Tensor::from_floats(d, &dev) };
    let get = |t: Tensor<Gpu, 1>| -> Vec<f32> { t.into_data().convert::<f32>().into_vec().unwrap() };
    let getb = |t: Tensor<Gpu, 1, burn::tensor::Bool>| -> Vec<bool> {
        t.into_data().convert::<bool>().into_vec().unwrap()
    };

    // min_pair — including an exact tie (index 2) and negatives (index 3).
    let a = mk(&[1.0, 5.0, 3.0, -2.0]);
    let b = mk(&[4.0, 2.0, 3.0, -7.0]);
    assert_eq!(get(a.clone().min_pair(b.clone())), vec![1.0, 2.0, 3.0, -7.0]);

    // clamp / clamp_min, saturating on both ends.
    assert_eq!(
        get(mk(&[-1.0, 0.25, 0.9]).clamp(0.0, 0.5)),
        vec![0.0, 0.25, 0.5]
    );
    assert_eq!(get(mk(&[1.0, 5.0]).clamp_min(2.0)), vec![2.0, 5.0]);

    // greater_elem / lower_elem: STRICT. The value exactly on the threshold
    // must be false on both — the K-floor and clamp masks rely on that to zero
    // the gradient at a saturated point.
    assert_eq!(getb(mk(&[1.0, 2.0, 3.0]).greater_elem(2.0)), vec![false, false, true]);
    assert_eq!(getb(mk(&[1.0, 2.0, 3.0]).lower_elem(2.0)), vec![true, false, false]);

    // lower_equal (used by the three-way branch cascade): NON-strict.
    assert_eq!(
        getb(mk(&[1.0, 2.0, 3.0]).lower_equal(mk(&[2.0, 2.0, 2.0]))),
        vec![true, true, false]
    );

    // bool_and / bool_not, the mask algebra of B19'.
    let m1 = mk(&[1.0, 1.0, 0.0, 0.0]).greater_elem(0.5);
    let m2 = mk(&[1.0, 0.0, 1.0, 0.0]).greater_elem(0.5);
    assert_eq!(getb(m1.clone().bool_and(m2.clone())), vec![true, false, false, false]);
    assert_eq!(getb(m1.clone().bool_not()), vec![false, false, true, true]);

    // mask_fill — the gradient-zeroing primitive every mask above feeds.
    assert_eq!(
        get(mk(&[7.0, 8.0, 9.0, 10.0]).mask_fill(m1.bool_not(), 0.0)),
        vec![7.0, 8.0, 0.0, 0.0]
    );

    // powf (elementwise exponent), powf_scalar, recip, sqrt — checked against
    // f64 references at tolerances tight enough to catch a wrong op, loose
    // enough for SFU-precision transcendentals.
    let p = get(mk(&[2.0, 9.0, 100.0]).powf(mk(&[0.5, 0.5, 0.6])));
    for (got, want) in p.iter().zip([2f64.sqrt(), 3.0, 100f64.powf(0.6)]) {
        assert!(
            (*got as f64 - want).abs() / want < 1e-5,
            "powf: got {got}, want {want}"
        );
    }
    let ps = get(mk(&[8.0, 27.0]).powf_scalar(2.0 / 3.0));
    for (got, want) in ps.iter().zip([4.0f64, 9.0]) {
        assert!(
            (*got as f64 - want).abs() / want < 1e-5,
            "powf_scalar: got {got}, want {want}"
        );
    }
    let rc = get(mk(&[2.0, 4.0, 1e-3]).recip());
    for (got, want) in rc.iter().zip([0.5f64, 0.25, 1000.0]) {
        assert!(
            (*got as f64 - want).abs() / want < 1e-6,
            "recip: got {got}, want {want}"
        );
    }
    let sq = get(mk(&[1e-6, 2.0, 1e6]).sqrt());
    for (got, want) in sq.iter().zip([1e-3f64, 2f64.sqrt(), 1e3]) {
        assert!(
            (*got as f64 - want).abs() / want < 1e-6,
            "sqrt: got {got}, want {want}"
        );
    }
}

// ===========================================================================
// Part D — the TRANSITIVE config guard.
//
// `enforce_positivity: true` + `use_cuda_graphs: true` must be impossible.
// Nothing asserts this today: it falls out of two independent validators, and
// which one fires depends on `ddr_match`.
//
//   ddr_match omitted (default true) -> validate_enforce_positivity fires
//                                       (enforce_positivity requires !ddr_match)
//   ddr_match: false                 -> validate_ddr_match fires
//                                       (!ddr_match requires !use_cuda_graphs)
//
// If EITHER validator were relaxed the combination would become reachable, and
// the CUDA-graph kernel — which implements DDR's formulation only, by design —
// would silently run the uncorrected forward against the corrected backward.
// This test does not depend on CUDA hardware, but it belongs here: it is the
// wall that makes "the graph kernel need not support the new physics" true.
// ===========================================================================

fn load_params_yaml(name: &str, params: &str) -> ddrs::data::error::Result<Config> {
    let yaml = format!(
        "mode: training\ngeodataset: merit\nseed: 1\nnp_seed: 1\nparams:\n{params}"
    );
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, yaml).unwrap();
    Config::from_yaml_file(&path)
}

#[test]
fn enforce_positivity_can_never_reach_the_cuda_graph_path() {
    // Case 1: ddr_match left at its default (true). The enforce_positivity
    // validator rejects before cuda_graphs is even relevant.
    let err = load_params_yaml(
        "ddrs_ep_graphs_default_ddr_match.yaml",
        "  enforce_positivity: true\n  use_cuda_graphs: true\n",
    )
    .expect_err("enforce_positivity + cuda_graphs (default ddr_match) must be rejected");
    let msg = format!("{err}");
    println!("case 1 rejection: {msg}");
    assert!(
        msg.contains("enforce_positivity") && msg.contains("ddr_match"),
        "expected the enforce_positivity/ddr_match conflict, got: {msg}"
    );

    // Case 2: ddr_match: false, which is the ONLY way enforce_positivity can
    // load at all. Now the ddr_match validator rejects the graphs.
    let err = load_params_yaml(
        "ddrs_ep_graphs_ddr_match_false.yaml",
        "  ddr_match: false\n  enforce_positivity: true\n  use_cuda_graphs: true\n",
    )
    .expect_err("ddr_match:false + enforce_positivity + cuda_graphs must be rejected");
    let msg = format!("{err}");
    println!("case 2 rejection: {msg}");
    assert!(
        msg.contains("ddr_match") && msg.contains("use_cuda_graphs"),
        "expected the ddr_match/cuda_graphs conflict, got: {msg}"
    );

    // Case 3: ddr_match: true + enforce_positivity: true, graphs OFF. Still
    // rejected — the clamp changes forward output and would break the DDR
    // sandbox ABSOLUTE MATCH (invariant 1).
    let err = load_params_yaml(
        "ddrs_ep_ddr_match_true.yaml",
        "  ddr_match: true\n  enforce_positivity: true\n  use_cuda_graphs: false\n",
    )
    .expect_err("ddr_match:true + enforce_positivity must be rejected");
    println!("case 3 rejection: {err}");

    // NON-VACUITY: the guard must not be rejecting everything. The one legal
    // corner — corrected physics, clamp on, graphs off — must LOAD, and must
    // land with the flags actually set (a validator that silently cleared
    // `enforce_positivity` would pass every assertion above).
    let cfg = load_params_yaml(
        "ddrs_ep_legal.yaml",
        "  ddr_match: false\n  enforce_positivity: true\n  use_cuda_graphs: false\n",
    )
    .expect("ddr_match:false + enforce_positivity:true + cuda_graphs:false must load");
    assert!(!cfg.params.ddr_match);
    assert!(cfg.params.enforce_positivity);
    assert!(!cfg.params.use_cuda_graphs);

    // And `enforce_positivity` must default OFF, so the guard is not merely
    // describing a flag nobody can set.
    let cfg = load_params_yaml("ddrs_ep_default.yaml", "  ddr_match: false\n")
        .expect("plain ddr_match:false must load");
    assert!(!cfg.params.enforce_positivity, "enforce_positivity must default to false");
}
