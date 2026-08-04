//! Positivity clamp (`params.enforce_positivity`) — S18' K floor and S19' X cap.
//!
//! # The theorem
//!
//! With `Cr = Δt/K` and `denom = 2K(1−X) + Δt > 0`:
//!
//! ```text
//! c2 = (2KX + Δt)/denom   > 0        always  (K>0, X>=0, Δt>0)
//! c4 = 2Δt/denom          > 0        always
//! c1 = (Δt − 2KX)/denom   >= 0  <=>  Cr >= 2X
//! c3 = (2K(1−X) − Δt)/denom >= 0 <=> Cr <= 2(1−X)
//! ```
//!
//! S27 is forward substitution in topological order,
//! `x[i] = b[i] + c1[i]·Σ_{j∈up(i)} x[j]`, with
//! `b[i] = c2·(N q_t)[i] + c3·q_t[i] + c4·q'[i]`. Since `q_t > 0` (S28's
//! `clamp_min(1e-4)` and the hotstart) and `q' > 0`, `c1, c3 >= 0` gives
//! `b >= 0`, and induction over the topological order gives `x >= 0` at every
//! reach: headwaters have no upstream so `x = b >= 0`, and each subsequent
//! `x[i]` is a non-negative combination of `b[i]` and already-non-negative
//! upstream values.
//!
//! The clamp therefore targets the INPUTS (`K`, `X`), never the coefficients:
//! `c1+c2+c3 = 1` holds identically for any `(K, X)`, so mass is preserved
//! exactly — clamping `c3` would break that.
//!
//! # Non-vacuity
//!
//! `fixture_is_not_vacuous` is a standing guard, not decoration. The Cunge-X
//! gradcheck in `tests/cunge_x.rs` was initially VACUOUS because a 1000 m
//! fixture saturated `X`'s clamp on every reach, so all four tests passed with
//! the backward terms deleted. Here the analogue would be a fixture where one
//! branch of the three-way `min` always wins, or where every reach sits on the
//! same side of the K floor. The fixture below spans `Cr` from 0.051 to 7.9,
//! straddles the K floor 5/10, and lets all three `min` branches win.

use std::sync::Arc;

use burn::backend::NdArray;
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::mmc_op::POSITIVITY_DELTA;
use ddrs::sparse::{CsrPattern, SparseAdjacency};

type I = NdArray<f32>;

/// `crate::routing::mmc::DT_SECONDS`, restated so a change there fails loudly
/// here rather than silently retuning the fixture.
const DT: f32 = 3600.0;

/// `x_sol` from the PRE-CHANGE code path, captured bit-for-bit by running this
/// exact fixture against `src/routing/mmc_op.rs` at commit fa5bcb4 (the config
/// flag, before the forward gained S18'/S19'). This is what makes
/// `off_parity_bit_identical_when_disabled` a real regression test rather than
/// a self-consistency tautology.
const GOLDEN_X_SOL_BITS: [u32; 10] = [
    0xc0645cca, // -3.5681634
    0xbf9f4594, // -1.2443109
    0x3f511646, //  0.8167461
    0xbf7f58e2, // -0.99745
    0x4143d4d4, // 12.23946
    0x42857d42, // 66.744644
    0x433334a2, // 179.2056
    0x428d80f3, // 70.751854
    0x42f2ee24, // 121.46512
    0x42a20a08, // 81.01959
];

// ===========================================================================
// Fixture
// ===========================================================================

/// Which branch of `x_eff = min(x_cunge, hi_a, hi_b)` won on a reach.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Branch {
    /// The Cunge-derived X was already inside the stability window.
    Cunge,
    /// `hi_a = 0.5·Cr·(1−δ)` — the `c1 >= 0` constraint bound.
    HiA,
    /// `hi_b = (1 − 0.5·Cr)·(1−δ)` — the `c3 >= 0` constraint bound.
    HiB,
}

struct Fixture {
    length: Vec<f32>,
    slope: Vec<f32>,
    n: Vec<f32>,
    qsp: Vec<f32>,
    psp: Vec<f32>,
    qt: Vec<f32>,
    qpt: Vec<f32>,
}

/// A 10-reach linear chain deliberately spanning the whole Courant range.
///
/// `Cr = Δt·c/L`, so length is the primary lever: 400 m gives `Cr ≈ 7.9`
/// (deep in the `c3 < 0` regime that produces negative solves) and 150 km
/// gives `Cr ≈ 0.051` (deep in the `c1 < 0` regime). `q_t` is graded so the
/// short reaches do not saturate Cunge `X` at its lower clamp, which keeps at
/// least one Cunge-branch win in the interior of `[0, 0.5]`.
///
/// `q'` is small (0.01 m³/s) so that on the `c3 < 0` reaches the negative
/// `c3·q_t` term is not masked by `c4·q'` — that is what makes the positive
/// control produce actual negatives.
fn fixture() -> Fixture {
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
        qt: vec![6.0, 8.0, 15.0, 50.0, 120.0, 200.0, 100.0, 100.0, 100.0, 100.0],
        qpt: vec![0.01; n],
    }
}

fn stress_cfg(enforce_positivity: bool) -> Config {
    let mut cfg = Config::default();
    // The clamp only exists on the corrected-physics path (and `Config`'s
    // loader rejects `ddr_match: true` + `enforce_positivity: true` outright).
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

/// The S1..S23 quantities the fixture assertions need, read out of the real
/// forward chain (not recomputed from a parallel model).
struct ChainOutputs {
    celerity: Vec<f32>,
    k_muskingum: Vec<f32>,
    top_width: Vec<f32>,
    c1: Vec<f32>,
    c2: Vec<f32>,
    c3: Vec<f32>,
}

fn run_chain(f: &Fixture, enforce_positivity: bool) -> ChainOutputs {
    let adj = chain(f);
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let mk = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &device) };
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
        celerity: outs[7].clone(),
        k_muskingum: outs[8].clone(),
        c1: outs[10].clone(),
        c2: outs[11].clone(),
        c3: outs[12].clone(),
    }
}

/// `x_sol` — the raw S27 solve, BEFORE S28's `clamp_min` rewrites negatives to
/// `+1e-4`. This is the quantity the whole task is about.
fn run_solve(f: &Fixture, enforce_positivity: bool) -> Vec<f32> {
    let adj = chain(f);
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let mk = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &device) };
    let (_b_rhs, _i_t, x_sol, _q_next) =
        ddrs::routing::mmc_op::__spike_forward_chain_k23_outputs::<I>(
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

/// `(negatives, total)` in the raw solve.
fn run_stress_network(enforce_positivity: bool) -> (usize, usize) {
    let x_sol = run_solve(&fixture(), enforce_positivity);
    (x_sol.iter().filter(|&&v| v < 0.0).count(), x_sol.len())
}

/// Which `min` branch won on each reach, and whether the K floor bound, both
/// derived from the *actual* chain outputs. `x_cunge` uses the pre-floor
/// `celerity` (S17) while `cr` uses the post-floor `k_muskingum` (S18'), which
/// is exactly how the forward composes them.
fn branch_report(f: &Fixture) -> (Vec<Branch>, Vec<bool>, Vec<f32>, Vec<f32>) {
    let out = run_chain(f, true);
    let k_floor = DT * (1.0 + POSITIVITY_DELTA) / 2.0;
    let mut branches = Vec::new();
    let mut floored = Vec::new();
    let mut cr_raw = Vec::new();
    let mut x_cunge_all = Vec::new();
    for i in 0..f.length.len() {
        let k_raw = f.length[i] / out.celerity[i];
        floored.push(k_raw < k_floor);
        cr_raw.push(DT / k_raw);
        let cr = DT / out.k_muskingum[i];
        let w = f.qt[i] / (out.top_width[i] * f.slope[i] * out.celerity[i] * f.length[i] + 1e-12);
        let x_cunge = (0.5 * (1.0 - w)).clamp(0.0, 0.5);
        x_cunge_all.push(x_cunge);
        let hi_a = cr * 0.5 * (1.0 - POSITIVITY_DELTA);
        let hi_b = (1.0 - 0.5 * cr) * (1.0 - POSITIVITY_DELTA);
        branches.push(if x_cunge <= hi_a && x_cunge <= hi_b {
            Branch::Cunge
        } else if hi_a <= hi_b {
            Branch::HiA
        } else {
            Branch::HiB
        });
    }
    (branches, floored, cr_raw, x_cunge_all)
}

// ===========================================================================
// Tests
// ===========================================================================

/// STANDING GUARD against a fixture that would let every test above pass with
/// the clamp deleted. Four independent ways this file could go vacuous:
///
/// 1. one `min` branch always wins → the other two are never exercised;
/// 2. every reach on the same side of the K floor → S18' is a no-op or a
///    constant;
/// 3. `Cr` never leaves `[0.98, 1.02]` → nothing to clamp;
/// 4. every Cunge-branch win sits on `x_cunge`'s own `[0, 0.5]` clamp → the
///    branch is "active" but carries no signal (this is precisely the failure
///    mode that made `tests/cunge_x.rs` vacuous at 1000 m reaches).
#[test]
fn fixture_is_not_vacuous() {
    let f = fixture();
    let (branches, floored, cr_raw, x_cunge) = branch_report(&f);
    let n = branches.len() as f32;
    let frac = |b: Branch| branches.iter().filter(|&&x| x == b).count() as f32 / n;
    let (fc, fa, fb) = (frac(Branch::Cunge), frac(Branch::HiA), frac(Branch::HiB));
    let f_floored = floored.iter().filter(|&&x| x).count() as f32 / n;
    let cr_min = cr_raw.iter().cloned().fold(f32::INFINITY, f32::min);
    let cr_max = cr_raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("branch mix: cunge={fc:.2} hi_a={fa:.2} hi_b={fb:.2}");
    println!("K floored: {f_floored:.2}   Cr_raw span: {cr_min:.3} .. {cr_max:.3}");
    for i in 0..branches.len() {
        println!(
            "  [{i}] L={:>8.0} Cr_raw={:>7.3} floored={:<5} x_cunge={:.4} branch={:?}",
            f.length[i], cr_raw[i], floored[i], x_cunge[i], branches[i]
        );
    }

    assert!(fc > 0.05, "Cunge branch never wins ({fc:.2}) — S19' cap is unfalsifiable");
    assert!(fa > 0.05, "hi_a branch never wins ({fa:.2}) — the c1>=0 bound is untested");
    assert!(fb > 0.05, "hi_b branch never wins ({fb:.2}) — the c3>=0 bound is untested");
    assert!(
        (0.05..0.95).contains(&f_floored),
        "fixture must STRADDLE the K floor, got {f_floored:.2} floored"
    );
    assert!(cr_min < 0.5, "fixture must reach well below Cr=1, got min {cr_min:.3}");
    assert!(cr_max > 2.0, "fixture must reach well above Cr=2, got max {cr_max:.3}");
    // At least one Cunge win must be strictly interior to X's own [0, 0.5]
    // clamp, or the "Cunge branch wins" reaches carry no gradient signal.
    let interior_cunge = branches
        .iter()
        .zip(&x_cunge)
        .filter(|(&b, &x)| b == Branch::Cunge && x > 0.02 && x < 0.48)
        .count();
    assert!(
        interior_cunge > 0,
        "every Cunge-branch win sits on X's own clamp — same vacuity trap as cunge_x.rs"
    );
}

/// POSITIVE CONTROL. If this stops producing negatives the fixture is broken
/// and every other test in this file becomes meaningless.
#[test]
fn positive_control_negatives_exist_without_the_clamp() {
    let (neg, tot) = run_stress_network(false);
    println!("clamp OFF: {neg}/{tot} negative solves");
    assert!(neg > 0, "fixture must actually produce negatives, got {neg}/{tot}");
}

#[test]
fn clamp_drives_negatives_to_exactly_zero() {
    let (neg, tot) = run_stress_network(true);
    println!("clamp ON: {neg}/{tot} negative solves");
    assert_eq!(neg, 0, "expected exactly zero negative solves, got {neg}/{tot}");
    assert!(tot > 0, "fixture must have solved something");
}

/// Mass conservation. The whole reason the clamp targets `K` and `X` rather
/// than `c1`/`c3` is that this identity is preserved for free.
#[test]
fn partition_identity_survives_the_clamp() {
    let out = run_chain(&fixture(), true);
    for i in 0..out.c1.len() {
        let s = out.c1[i] + out.c2[i] + out.c3[i];
        assert!(
            (s - 1.0).abs() < 1e-5,
            "reach {i}: c1+c2+c3 = {s} (c1={}, c2={}, c3={})",
            out.c1[i],
            out.c2[i],
            out.c3[i]
        );
    }
}

/// The δ margin is what makes this hold in f32: at δ = 0 the cap lands exactly
/// on `c1 = 0` / `c3 = 0` and roundoff crosses it.
#[test]
fn coefficients_are_non_negative_with_margin() {
    let out = run_chain(&fixture(), true);
    let min_c1 = out.c1.iter().cloned().fold(f32::INFINITY, f32::min);
    let min_c3 = out.c3.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("min c1 = {min_c1:e}, min c3 = {min_c3:e}");
    assert!(min_c1 >= 0.0, "min c1 = {min_c1:e} < 0");
    assert!(min_c3 >= 0.0, "min c3 = {min_c3:e} < 0");
    // The same coefficients WITHOUT the clamp must go negative, or this test
    // would pass on an unclamped build.
    let off = run_chain(&fixture(), false);
    let off_c1 = off.c1.iter().cloned().fold(f32::INFINITY, f32::min);
    let off_c3 = off.c3.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        off_c1 < 0.0 && off_c3 < 0.0,
        "fixture must violate BOTH bounds when unclamped (min c1={off_c1:e}, min c3={off_c3:e})"
    );
}

/// `enforce_positivity: false` must be a BIT-IDENTICAL no-op. `GOLDEN_X_SOL_BITS`
/// was captured from the code at commit fa5bcb4, i.e. before S18'/S19' existed.
#[test]
fn off_parity_bit_identical_when_disabled() {
    let x_sol = run_solve(&fixture(), false);
    assert_eq!(x_sol.len(), GOLDEN_X_SOL_BITS.len());
    for (i, (&v, &want)) in x_sol.iter().zip(&GOLDEN_X_SOL_BITS).enumerate() {
        assert_eq!(
            v.to_bits(),
            want,
            "reach {i}: disabled clamp changed x_sol — got {v:e} ({:#010x}), \
             pre-change {:e} ({want:#010x})",
            v.to_bits(),
            f32::from_bits(want)
        );
    }
    // Non-vacuity: the clamp must actually move this fixture, otherwise
    // "byte-identical when disabled" is trivially true.
    let on = run_solve(&fixture(), true);
    assert!(
        on.iter().zip(&x_sol).any(|(a, b)| a != b),
        "clamp ON produced identical output — the off-parity test proves nothing"
    );
}

/// S18' floors K at `Δt(1+δ)/2`, which is exactly the condition
/// `Cr <= 2/(1+δ) < 2`, which is what makes `hi_b = (1 − 0.5·Cr)(1−δ) > 0` and
/// therefore makes the three-way `min` positive without a `clamp_min`.
#[test]
fn k_floor_keeps_courant_below_two() {
    let f = fixture();
    let out = run_chain(&f, true);
    let off = run_chain(&f, false);
    let k_floor = DT * (1.0 + POSITIVITY_DELTA) / 2.0;
    let cr_cap = 2.0 / (1.0 + POSITIVITY_DELTA);
    let mut any_floored = false;
    for i in 0..f.length.len() {
        assert!(
            out.k_muskingum[i] >= k_floor * (1.0 - 1e-6),
            "reach {i}: K = {} below floor {k_floor}",
            out.k_muskingum[i]
        );
        let cr = DT / out.k_muskingum[i];
        assert!(cr <= cr_cap * (1.0 + 1e-6), "reach {i}: Cr = {cr} exceeds {cr_cap}");
        if off.k_muskingum[i] < k_floor {
            any_floored = true;
            // Where the floor binds it must bind to exactly the floor value.
            assert!(
                (out.k_muskingum[i] - k_floor).abs() < 1e-3,
                "reach {i}: floored K = {} != {k_floor}",
                out.k_muskingum[i]
            );
        } else {
            // Where it does not bind, K must be untouched bit-for-bit.
            assert_eq!(
                out.k_muskingum[i].to_bits(),
                off.k_muskingum[i].to_bits(),
                "reach {i}: unfloored K was perturbed"
            );
        }
    }
    assert!(any_floored, "no reach hit the K floor — the S18' branch is untested");
}
