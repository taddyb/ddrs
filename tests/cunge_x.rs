//! Cunge-derived Muskingum storage weight `X` (`ddr_match: false`).
//!
//! Part 1 — the physics. Cunge picks `X` so the Muskingum scheme's *numerical*
//! diffusion equals the channel's *physical* hydraulic diffusivity:
//!
//!     D_num  = c·Δx·(0.5 − X)          D_phys = Q/(2·B·S₀)
//!     =>  X  = clamp( 0.5·(1 − Q/(B·S₀·c·Δx)), 0, 0.5 )
//!
//! A constant `X` (DDR / `ddr_match: true`; `forward.rs` supplies 0.3) severs
//! that link — it is Cunge-optimal only on the measure-zero locus where
//! `Q/(B·S₀·c·Δx) == 0.4`, and over-diffuses by ~30x at the CONUS-median
//! operating point (documented median `D_num/D_phys` = 28x).
//!
//! Part 2 — the gradcheck. `X` now depends on `Q`, `B` (top width) and `c`
//! (celerity), so the analytical backward in `src/routing/mmc_op.rs` gains
//! three hand-derived terms. `q_t` in particular acquires a SECOND gradient
//! path (it already entered the RHS at S25 and the depth chain at S2).
//! Validated against central finite differences below with `mock_cfg()`
//! setting `ddr_match = false`.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::mmc_op::timestep_forward;
use ddrs::sparse::{AValuesAssembler, CsrPattern, SparseAdjacency};

// ===========================================================================
// Part 1: the X formula, in f64, independent of any ddrs code.
// ===========================================================================

/// `X = clamp( 0.5·(1 − Q/(B·S·c·Δx)), 0, 0.5 )`.
fn cunge_x(q: f64, b: f64, s: f64, c: f64, dx: f64) -> f64 {
    (0.5 * (1.0 - q / (b * s * c * dx))).clamp(0.0, 0.5)
}

/// Muskingum numerical diffusion `D_num = c·Δx·(0.5 − X)`.
fn d_num(c: f64, dx: f64, x: f64) -> f64 {
    c * dx * (0.5 - x)
}

/// Physical hydraulic diffusivity `D_phys = Q/(2·B·S₀)`.
fn d_phys(q: f64, b: f64, s: f64) -> f64 {
    q / (2.0 * b * s)
}

/// A CONUS-median-ish reach: `W = Q/(B·S·c·Δx) ≈ 0.0135` → Cunge `X ≈ 0.493`,
/// reproducing the documented "`X ≈ 0.49` almost everywhere" measurement.
///
/// Self-consistency check on Q: `c = 1.4` under `β = 5/3` implies `v ≈ 0.84`
/// m/s, so `A = Q/v ≈ 11.9 m²`, i.e. ~0.3 m mean depth over a 40 m top width —
/// a plausible Manning `n ≈ 0.024` at `S = 2e-3`.
const MEDIAN_REACH: (f64, f64, f64, f64, f64) = (10.0, 40.0, 2e-3, 1.4, 6598.0);

#[test]
fn cunge_x_matches_numerical_to_physical_diffusion() {
    // Unclamped regime: the identity D_num == D_phys is exact by construction.
    for &(q, b, s, c, dx) in &[
        MEDIAN_REACH,
        (300.0_f64, 40.0_f64, 2e-3_f64, 1.4_f64, 6598.0_f64),
        (5.0, 12.0, 5e-4, 0.9, 2100.0),
    ] {
        let x = cunge_x(q, b, s, c, dx);
        assert!(x > 0.0 && x < 0.5, "operating point must be unclamped, got X={x}");
        let (dn, dp) = (d_num(c, dx, x), d_phys(q, b, s));
        assert!(
            (dn / dp - 1.0).abs() < 1e-9,
            "D_num={dn} != D_phys={dp} at (q={q}, b={b}, s={s}, c={c}, dx={dx})"
        );
    }
}

#[test]
fn constant_x_030_over_diffuses_at_the_median_reach() {
    // Regression guard on the magnitude of the defect this task fixes.
    let (q, b, s, c, dx) = MEDIAN_REACH;
    let ratio = d_num(c, dx, 0.3) / d_phys(q, b, s);
    assert!(
        ratio > 2.0,
        "expected heavy over-diffusion from constant X=0.3, got {ratio:.2}x"
    );
    // The documented CONUS median is 28x; this operating point sits at ~30x.
    assert!(
        (20.0..40.0).contains(&ratio),
        "expected the documented order-30x over-diffusion, got {ratio:.2}x"
    );
}

#[test]
fn constant_x_030_is_correct_only_on_a_measure_zero_locus() {
    // D_num(0.3)/D_phys = 0.4/W, so the constant is Cunge-optimal iff W = 0.4.
    // That is why (300, 40, 2e-3, 1.4, 6598) — W = 0.406 — makes X=0.3 look
    // fine: it is one point on that locus, NOT a representative reach.
    let (b, s, c, dx) = (40.0_f64, 2e-3_f64, 1.4_f64, 6598.0_f64);
    let q_tuned = 0.4 * b * s * c * dx;
    let ratio = d_num(c, dx, 0.3) / d_phys(q_tuned, b, s);
    assert!(
        (ratio - 1.0).abs() < 1e-9,
        "X=0.3 must be exact at W=0.4, got {ratio}"
    );
    // Move Q by one order of magnitude in either direction and it breaks by
    // the same order (D_num(0.3)/D_phys = 0.4/W is exactly inverse in Q).
    for &f in &[0.1_f64, 10.0] {
        let r = d_num(c, dx, 0.3) / d_phys(q_tuned * f, b, s);
        assert!(
            r.max(1.0 / r) > 9.0,
            "X=0.3 should be ~{f}x wrong at {f}x the tuned Q, got {r}"
        );
    }
}

#[test]
fn cunge_x_clamps_into_zero_half() {
    let (_, b, s, c, dx) = MEDIAN_REACH;
    // Huge Q -> raw X goes negative -> clamped to 0 (pure advection).
    assert_eq!(cunge_x(1e9, b, s, c, dx), 0.0);
    // Tiny Q -> raw X approaches 0.5 from below, never exceeds it.
    let x_tiny = cunge_x(1e-9, b, s, c, dx);
    assert!(x_tiny <= 0.5, "X must not exceed 0.5, got {x_tiny}");
    assert!(x_tiny > 0.499, "tiny Q should push X to the 0.5 limit, got {x_tiny}");
    // Sweep: X stays inside [0, 0.5] over 12 decades of Q.
    for e in -6..6 {
        let x = cunge_x(10f64.powi(e), b, s, c, dx);
        assert!((0.0..=0.5).contains(&x), "X={x} out of [0,0.5] at Q=1e{e}");
    }
}

#[test]
fn muskingum_coefficients_sum_to_one_for_any_x() {
    // Mass conservation is independent of X: the three numerators sum to the
    // shared denominator identically.
    for &x in &[0.0_f64, 0.3, 0.49, 0.5] {
        let (k, dt) = (3295.0_f64, 3600.0_f64);
        let denom = 2.0 * k * (1.0 - x) + dt;
        let c1 = (dt - 2.0 * k * x) / denom;
        let c2 = (dt + 2.0 * k * x) / denom;
        let c3 = (2.0 * k * (1.0 - x) - dt) / denom;
        assert!((c1 + c2 + c3 - 1.0).abs() < 1e-12, "x={x}");
    }
}

#[test]
fn cunge_x_narrows_the_non_negative_coefficient_window() {
    // Non-negative Muskingum coefficients require 2X <= Cr <= 2(1-X).
    // X=0.3 -> [0.6, 1.4]; Cunge X~0.49 -> ~[0.98, 1.02]. This is the
    // documented reason Task 5 (Courant sub-stepping) exists.
    let (q, b, s, c, dx) = MEDIAN_REACH;
    let x = cunge_x(q, b, s, c, dx);
    let (lo, hi) = (2.0 * x, 2.0 * (1.0 - x));
    assert!(hi - lo < 0.1, "Cunge window [{lo}, {hi}] should be narrow");
    assert!(
        hi - lo < (2.0 * (1.0 - 0.3) - 2.0 * 0.3) / 10.0,
        "Cunge window must be >10x narrower than the X=0.3 window"
    );
}

// ===========================================================================
// Part 2: gradcheck of the Cunge-X backward (`ddr_match: false`).
// Scaffolding mirrors `tests/celerity_beta.rs` Part 2 (itself from
// `tests/sp8_gradcheck.rs`). `mock_cfg()` sets `ddr_match = false`, so BOTH
// the trapezoidal celerity (S17, Task 3) and the Cunge X (S19, this task) are
// live — the two corrections are coupled through `celerity`, and X's
// `∂X/∂c` term feeds the very `gcelerity` accumulator B17 consumes.
// ===========================================================================

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 4;
const EPS: f32 = 1e-3;
const REL_TOL: f32 = 5e-3;
const ABS_TOL: f32 = 1e-4;

#[derive(Copy, Clone, Debug)]
enum Parent {
    N,
    QSpatial,
    PSpatial,
    QT,
}

/// NOTE the 5000 m reach length — it is load-bearing, not cosmetic.
///
/// `tests/celerity_beta.rs` uses 1000 m. At that length the fixture's
/// `W = Q/(B·S·c·L) ≈ 1.6`, so Cunge `X` saturates at the lower clamp on every
/// reach, `gX` is masked to zero, and the whole Part-2 gradcheck would pass
/// with the S19 backward terms DELETED (verified: it does). 5000 m puts
/// `W ≈ 0.32..0.51`, i.e. `X ≈ 0.25..0.34`, strictly interior — so the three
/// new terms are actually exercised. `cunge_x_operating_point_is_unclamped`
/// below is the standing guard on this.
fn linear_chain_sparse() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![5000.0; N], vec![0.001; N])
}

fn mock_cfg() -> Config {
    let mut cfg = Config::default();
    // The whole point of this file: exercise the corrected-physics branch.
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

fn default_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_vec = vec![0.035f32; N];
    let qsp_vec = vec![0.4f32; N];
    let psp_vec = vec![20.0f32; N];
    let qt_vec = vec![100.0f32, 120.0, 140.0, 160.0];
    let qpt_vec = vec![10.0f32, 12.0, 14.0, 16.0];
    (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec)
}

struct GradTensors {
    n: Tensor<AB, 1>,
    qsp: Tensor<AB, 1>,
    psp: Tensor<AB, 1>,
    qt: Tensor<AB, 1>,
}

#[allow(clippy::too_many_arguments)]
fn run_forward_loss(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    assembler: &AValuesAssembler<I>,
    device: &<I as burn::tensor::backend::BackendTypes>::Device,
    n_vec: &[f32],
    qsp_vec: &[f32],
    psp_vec: &[f32],
    qt_vec: &[f32],
    qpt_vec: &[f32],
    length_vec: &[f32],
    slope_vec: &[f32],
    x_storage_vec: &[f32],
    require_grad_parent: Option<Parent>,
) -> (Tensor<AB, 1>, GradTensors) {
    let mk = |data: &[f32], req: bool| -> Tensor<AB, 1> {
        let t: Tensor<AB, 1> = Tensor::from_floats(data, device);
        if req { t.require_grad() } else { t }
    };
    let n_t = mk(n_vec, matches!(require_grad_parent, Some(Parent::N)));
    let qsp_t = mk(qsp_vec, matches!(require_grad_parent, Some(Parent::QSpatial)));
    let psp_t = mk(psp_vec, matches!(require_grad_parent, Some(Parent::PSpatial)));
    let qt_t = mk(qt_vec, matches!(require_grad_parent, Some(Parent::QT)));
    let qpt_t = mk(qpt_vec, false);
    let length_t = mk(length_vec, false);
    let slope_t = mk(slope_vec, false);
    let xst_t = mk(x_storage_vec, false);

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

    (
        q_next,
        GradTensors {
            n: n_t,
            qsp: qsp_t,
            psp: psp_t,
            qt: qt_t,
        },
    )
}

fn compute_analytical_grad(parent: Parent) -> Vec<f32> {
    let cfg = mock_cfg();
    let adj = linear_chain_sparse();
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);

    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let length_vec = adj.length_m.clone();
    let slope_vec = adj.slope.clone();
    let x_storage_vec = vec![0.3f32; N];

    let (q_next, parents) = run_forward_loss(
        &cfg,
        &pattern,
        &assembler,
        &device,
        &n_vec,
        &qsp_vec,
        &psp_vec,
        &qt_vec,
        &qpt_vec,
        &length_vec,
        &slope_vec,
        &x_storage_vec,
        Some(parent),
    );

    let loss = q_next.sum();
    let grads = loss.backward();

    let g = match parent {
        Parent::N => parents.n.grad(&grads).expect("grad on n"),
        Parent::QSpatial => parents.qsp.grad(&grads).expect("grad on q_spatial"),
        Parent::PSpatial => parents.psp.grad(&grads).expect("grad on p_spatial"),
        Parent::QT => parents.qt.grad(&grads).expect("grad on q_t"),
    };
    g.into_data().to_vec::<f32>().unwrap()
}

fn compute_fd_grad(parent: Parent) -> Vec<f32> {
    let cfg = mock_cfg();
    let adj = linear_chain_sparse();
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);

    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let length_vec = adj.length_m.clone();
    let slope_vec = adj.slope.clone();
    let x_storage_vec = vec![0.3f32; N];

    let eval_loss = |n: &[f32], qsp: &[f32], psp: &[f32], qt: &[f32], qpt: &[f32]| -> f32 {
        let (q_next, _) = run_forward_loss(
            &cfg,
            &pattern,
            &assembler,
            &device,
            n,
            qsp,
            psp,
            qt,
            qpt,
            &length_vec,
            &slope_vec,
            &x_storage_vec,
            None,
        );
        let v: Vec<f32> = q_next.sum().into_data().to_vec::<f32>().unwrap();
        v[0]
    };

    let mut grad = vec![0.0f32; N];
    for i in 0..N {
        let mut plus_n = n_vec.clone();
        let mut plus_qsp = qsp_vec.clone();
        let mut plus_psp = psp_vec.clone();
        let mut plus_qt = qt_vec.clone();
        let mut minus_n = n_vec.clone();
        let mut minus_qsp = qsp_vec.clone();
        let mut minus_psp = psp_vec.clone();
        let mut minus_qt = qt_vec.clone();

        let (plus, minus, base) = match parent {
            Parent::N => (&mut plus_n, &mut minus_n, &n_vec),
            Parent::QSpatial => (&mut plus_qsp, &mut minus_qsp, &qsp_vec),
            Parent::PSpatial => (&mut plus_psp, &mut minus_psp, &psp_vec),
            Parent::QT => (&mut plus_qt, &mut minus_qt, &qt_vec),
        };
        let eps = (EPS * base[i].abs()).max(EPS);
        plus[i] = base[i] + eps;
        minus[i] = base[i] - eps;

        let l_plus = eval_loss(&plus_n, &plus_qsp, &plus_psp, &plus_qt, &qpt_vec);
        let l_minus = eval_loss(&minus_n, &minus_qsp, &minus_psp, &minus_qt, &qpt_vec);
        grad[i] = (l_plus - l_minus) / (2.0 * eps);
    }
    grad
}

fn compare_grads(name: &str, analytical: &[f32], fd: &[f32]) {
    assert_eq!(analytical.len(), fd.len());
    println!("--- {name} ---");
    let mut worst_rel = 0.0f32;
    let mut worst_abs = 0.0f32;
    for i in 0..analytical.len() {
        let a = analytical[i];
        let f = fd[i];
        let abs_diff = (a - f).abs();
        let denom = a.abs().max(f.abs()).max(1e-12);
        let rel_diff = abs_diff / denom;
        worst_abs = worst_abs.max(abs_diff);
        worst_rel = worst_rel.max(rel_diff);
        println!("  [{i}] analytical={a:.6e}  fd={f:.6e}  abs={abs_diff:.3e}  rel={rel_diff:.3e}");
    }
    println!("  worst abs={worst_abs:.3e}  worst rel={worst_rel:.3e}");
    let pass = analytical.iter().zip(fd).all(|(&a, &f)| {
        let abs_diff = (a - f).abs();
        let denom = a.abs().max(f.abs()).max(1e-12);
        let rel_diff = abs_diff / denom;
        rel_diff < REL_TOL || abs_diff < ABS_TOL
    });
    assert!(
        pass,
        "{name}: gradcheck failed (worst rel={worst_rel:.3e}, abs={worst_abs:.3e})"
    );
}

#[test]
fn gradcheck_cunge_x_n() {
    let a = compute_analytical_grad(Parent::N);
    let fd = compute_fd_grad(Parent::N);
    compare_grads("n (cunge X)", &a, &fd);
}

#[test]
fn gradcheck_cunge_x_q_spatial() {
    let a = compute_analytical_grad(Parent::QSpatial);
    let fd = compute_fd_grad(Parent::QSpatial);
    compare_grads("q_spatial (cunge X)", &a, &fd);
}

#[test]
fn gradcheck_cunge_x_p_spatial() {
    let a = compute_analytical_grad(Parent::PSpatial);
    let fd = compute_fd_grad(Parent::PSpatial);
    compare_grads("p_spatial (cunge X)", &a, &fd);
}

/// The genuinely new gradient path this task adds: `q_t` now enters `X` at
/// S19 on top of the S25 RHS, the S24 SpMV and the S2 depth chain.
#[test]
fn gradcheck_cunge_x_q_t() {
    let a = compute_analytical_grad(Parent::QT);
    let fd = compute_fd_grad(Parent::QT);
    compare_grads("q_t (cunge X)", &a, &fd);
}

/// Standing guard against a VACUOUS gradcheck.
///
/// Every S19 gradient term is multiplied by the `[0, 0.5]` clamp mask, so if
/// the fixture's operating point saturates `X`, all four gradcheck tests above
/// pass trivially whether or not the backward terms exist. This test pins the
/// fixture into the interior of the clamp using the top width and celerity the
/// forward chain actually produced.
#[test]
fn cunge_x_operating_point_is_unclamped() {
    let cfg = mock_cfg();
    let adj = linear_chain_sparse();
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));

    let (n_vec, qsp_vec, psp_vec, qt_vec, qpt_vec) = default_inputs();
    let mk = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &device) };

    let outs = ddrs::routing::mmc_op::__spike_forward_chain_k1_outputs::<I>(
        &cfg,
        &pattern,
        mk(&n_vec),
        mk(&qsp_vec),
        mk(&psp_vec),
        mk(&qt_vec),
        mk(&qpt_vec),
        mk(&adj.length_m),
        mk(&adj.slope),
        mk(&vec![0.3f32; N]),
    );
    // k1 output order: [depth, top_width, side_slope, bottom_width, hyd_radius,
    //                   velocity_un, velocity_cl, celerity, ...]
    let top_width = &outs[1];
    let celerity = &outs[7];

    for i in 0..N {
        let (q, b, s, c, dx) = (
            qt_vec[i] as f64,
            top_width[i] as f64,
            adj.slope[i] as f64,
            celerity[i] as f64,
            adj.length_m[i] as f64,
        );
        let w = q / (b * s * c * dx);
        let x = cunge_x(q, b, s, c, dx);
        println!("  [{i}] B={b:.3} c={c:.4} W={w:.4} X={x:.4}");
        assert!(
            x > 0.02 && x < 0.48,
            "reach {i}: Cunge X={x:.4} (W={w:.4}) is at/near a clamp bound — \
             the gradcheck would be vacuous. Retune the fixture."
        );
    }
}
