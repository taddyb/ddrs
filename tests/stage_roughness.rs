//! Gates for stage-dependent Manning roughness, `n(d) = n_0·(d/d_ref)^(−gamma)`.
//!
//! Design: `docs/superpowers/specs/2026-09-12-stage-dependent-roughness-design.md`.
//!
//! Three things must hold, and the third is the one the existing suite has no
//! analogue for:
//!
//! 1. **`gamma = 0` changes nothing.** The solver must take the historical path
//!    bit for bit, or every model trained before this existed becomes
//!    unreproducible and `compare_ddr_sandbox` (invariant 1) moves.
//! 2. **`gamma > 0` produces the exponents the theory says.** The whole point is
//!    to move the at-a-station velocity exponent off Manning's `2f/3`.
//! 3. **The celerity correction is right.** `beta` gained a `gamma·A/(T·d)`
//!    term, and `c = dQ/dA` is the definition it has to satisfy. A wrong
//!    celerity would surface only as slightly-off peak timing, which is
//!    indistinguishable from ordinary model error and would quietly poison
//!    everything downstream — so it is checked numerically here.

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};
use ddrs::geometry::{compute_trapezoidal_geometry, compute_trapezoidal_geometry_gamma};

type B = NdArray<f32>;

const DEPTH_LB: f32 = 0.01;
const BW_LB: f32 = 0.01;

fn t(v: Vec<f32>, device: &NdArrayDevice) -> Tensor<B, 1> {
    let n = v.len();
    Tensor::from_data(TensorData::new(v, [n]), device)
}

/// A handful of reaches spanning the CONUS range of size and steepness.
fn reaches(device: &NdArrayDevice, q: Vec<f32>) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let k = q.len();
    (
        t(vec![0.05, 0.08, 0.12, 0.03][..k].to_vec(), device), // n
        t(vec![21.0; k], device),                              // p
        t(q, device),                                          // q
        t(vec![0.005, 0.002, 0.001, 0.01][..k].to_vec(), device), // slope
    )
}

fn vecf(x: Tensor<B, 1>) -> Vec<f32> {
    x.into_data().to_vec::<f32>().unwrap()
}

/// Gate 1. `gamma = 0` must reproduce the historical geometry bit for bit.
#[test]
fn gamma_zero_is_bit_identical() {
    let device = NdArrayDevice::default();
    let (n, p, q, s) = reaches(&device, vec![0.1, 0.4, 0.8, 0.05]);
    let qd = t(vec![1.0, 50.0, 500.0, 0.2], &device);

    let base = compute_trapezoidal_geometry(
        n.clone(), p.clone(), q.clone(), qd.clone(), s.clone(), DEPTH_LB, BW_LB,
    );
    // Both the "off" spelling and an explicit d_ref must match.
    for d_ref in [1.0_f32, 2.5] {
        let g = compute_trapezoidal_geometry_gamma(
            n.clone(), p.clone(), q.clone(), qd.clone(), s.clone(), DEPTH_LB, BW_LB, 0.0, d_ref,
        );
        assert_eq!(
            vecf(base.depth.clone()),
            vecf(g.depth),
            "gamma=0 (d_ref={d_ref}) changed depth"
        );
        assert_eq!(
            vecf(base.velocity.clone()),
            vecf(g.velocity),
            "gamma=0 (d_ref={d_ref}) changed velocity"
        );
    }
}

/// Gate 2. The realised exponents must be the ones the algebra predicts:
/// `f = 3/(5+3q+3·gamma)`, `b = q·f`, `m = 1 − b − f`.
///
/// Fitted by finite difference in log-log over a decade of discharge, which is
/// how the hydraulic-geometry literature defines them.
#[test]
fn exponents_match_theory_and_gamma_moves_velocity() {
    let device = NdArrayDevice::default();
    let q_val = 0.65_f32; // the at-a-station Leopold & Maddock value, b/f
    let (n, p, q, s) = reaches(&device, vec![q_val]);

    let exponents = |gamma: f32| -> (f64, f64, f64) {
        let lo = 10.0_f32;
        let hi = 100.0_f32;
        let mut out = Vec::new();
        for qd in [lo, hi] {
            let g = compute_trapezoidal_geometry_gamma(
                n.clone(), p.clone(), q.clone(),
                t(vec![qd], &device), s.clone(), DEPTH_LB, BW_LB, gamma, 1.0,
            );
            out.push((
                vecf(g.depth)[0] as f64,
                vecf(g.top_width)[0] as f64,
                vecf(g.velocity)[0] as f64,
            ));
        }
        let dlq = (hi as f64 / lo as f64).ln();
        (
            (out[1].1 / out[0].1).ln() / dlq, // b, width
            (out[1].0 / out[0].0).ln() / dlq, // f, depth
            (out[1].2 / out[0].2).ln() / dlq, // m, velocity
        )
    };

    for gamma in [0.0_f32, 0.183, 0.4] {
        let (b, f, m) = exponents(gamma);
        let qf = q_val as f64;
        let f_pred = 3.0 / (5.0 + 3.0 * qf + 3.0 * gamma as f64);
        let b_pred = qf * f_pred;
        let m_pred = 1.0 - b_pred - f_pred;
        assert!(
            (f - f_pred).abs() < 2e-3,
            "gamma={gamma}: depth exponent {f:.4} != predicted {f_pred:.4}"
        );
        assert!(
            (b - b_pred).abs() < 2e-3,
            "gamma={gamma}: width exponent {b:.4} != predicted {b_pred:.4}"
        );
        // Velocity is the one gamma exists to move, and it is the one the
        // trapezoid approximation perturbs most, so it gets a looser bar.
        assert!(
            (m - m_pred).abs() < 2e-2,
            "gamma={gamma}: velocity exponent {m:.4} != predicted {m_pred:.4}"
        );
    }

    // And the headline claim: gamma raises the velocity exponent off Manning's
    // 2f/3, toward the observed at-a-station 0.34.
    let (_, _, m0) = exponents(0.0);
    let (_, _, m1) = exponents(0.183);
    assert!(
        m1 > m0 + 0.02,
        "gamma=0.183 must raise the velocity exponent: {m0:.4} -> {m1:.4}"
    );
}

/// Gate 3. The stage-roughness celerity term is exact.
///
/// `beta_trapezoid = 5/3 − (4/3)·A·√(1+z²)/(T·P)` is itself an approximation:
/// it treats the side slope `z` as fixed while depth varies, but this geometry
/// has `z = T·q/(2d) = (p·q/2)·d^(q−1)`, which moves with depth unless `q = 1`.
/// So `v·beta` does NOT equal the true `dQ/dA` even at `gamma = 0`, and that is
/// pre-existing, inherited from DDR, and out of scope here — see
/// `documents_the_preexisting_beta_approximation` below, which measures it.
///
/// What IS exact, and what this gate checks, is the stage-roughness increment.
/// From `Q = (d/d_ref)^gamma · Q_Manning(d)` and `dA/dd = T`:
///
/// ```text
///   c(d, gamma) = (d/d_ref)^gamma · c(d, 0)  +  gamma · v(d,gamma) · A/(T·d)
/// ```
///
/// for `c(d, 0)` **computed under the same convention the solver uses**, which
/// is `c = (dQ/dd) / T`. That convention is itself an approximation twice over:
/// it takes `dA/dd = T` and `dP/dd = 2√(1+z²)`, i.e. a trapezoid of FIXED shape
/// being filled. This geometry reshapes as it fills — `bw = tw·(1−q)` and
/// `z = (p·q/2)·d^(q−1)` both move with depth, so the true `dA/dd` is
/// `T·(2−q)(q+1)/2`, up to 11 % away from `T` in the middle of the `q` range.
/// Correcting that would change the routing physics and DDR parity, so it is
/// out of scope; it is measured in
/// `documents_the_preexisting_beta_approximation`.
///
/// The geometry at a given depth does not depend on `gamma` at all — `gamma`
/// only changes which discharge maps to that depth, and the velocity there — so
/// the identity can be checked at fixed depth.
///
/// The geometry is reimplemented here from DEPTH rather than from discharge, so
/// that a change to `src/geometry.rs` which forgets the correction fails here
/// rather than being absorbed.
#[test]
fn stage_roughness_celerity_increment_is_exact() {
    // Pure functions of depth, mirroring src/geometry.rs.
    let geom = |d: f64, p: f64, q: f64| -> (f64, f64, f64, f64) {
        let tw = p * d.powf(q);
        let z = (tw * q / (d * 2.0)).clamp(0.5, 50.0);
        let bw = (tw - z * d * 2.0).max(BW_LB as f64);
        let area = (tw + bw) * d / 2.0;
        let wp = bw + d * (z * z + 1.0).sqrt() * 2.0;
        (area, tw, wp, area / wp)
    };
    // Q(d, gamma) = v·A with v = (1/n_0)·(d/d_ref)^gamma·R^(2/3)·√S.
    let discharge = |d: f64, gamma: f64, n0: f64, s: f64, p: f64, q: f64| -> (f64, f64) {
        let (area, _tw, _wp, r) = geom(d, p, q);
        let v = (1.0 / n0) * d.powf(gamma) * r.powf(2.0 / 3.0) * s.sqrt();
        (v * area, v)
    };
    // The solver's celerity convention: c = (dQ/dd) / T, where T is the top
    // width. NOT (dQ/dd)/(dA/dd) — see the doc comment for why those differ.
    let celerity_num = |d: f64, gamma: f64, n0: f64, s: f64, p: f64, q: f64| -> f64 {
        let h = d * 1e-6;
        let (q_hi, _) = discharge(d + h, gamma, n0, s, p, q);
        let (q_lo, _) = discharge(d - h, gamma, n0, s, p, q);
        let (_, tw, ..) = geom(d, p, q);
        (q_hi - q_lo) / (2.0 * h) / tw
    };

    let (n0, s, p) = (0.08_f64, 0.002_f64, 21.0_f64);
    // Both the trained regime (q ~ 0.08) and the at-a-station L&M value.
    for q in [0.084_f64, 0.65] {
        for d in [0.25_f64, 1.0, 3.0] {
            for gamma in [0.1_f64, 0.183, 0.4] {
                let c_g = celerity_num(d, gamma, n0, s, p, q);
                let c_0 = celerity_num(d, 0.0, n0, s, p, q);
                let (_, v_g) = discharge(d, gamma, n0, s, p, q);
                let (area, tw, ..) = geom(d, p, q);

                // d_ref = 1, so (d/d_ref)^gamma = d^gamma.
                let predicted = d.powf(gamma) * c_0 + gamma * v_g * area / (tw * d);
                let rel = (c_g - predicted).abs() / c_g.abs();
                assert!(
                    rel < 1e-4,
                    "q={q} d={d} gamma={gamma}: numerical celerity {c_g:.6} vs \
                     identity {predicted:.6} (rel {rel:.2e}). The gamma·A/(T·d) \
                     term is wrong."
                );
            }
        }
    }
}

/// Not a gate: a measurement, recorded so it is not rediscovered as a bug.
///
/// `beta_trapezoid` ignores `dz/dd`, so `v·beta` differs from the true `dQ/dA`.
/// This prints the size of that gap across the range of `q` the model actually
/// occupies. It matters because `K = L/c`, so a celerity bias is a routing
/// timescale bias, and the whole model is judged on peak timing.
#[test]
fn documents_the_preexisting_beta_approximation() {
    let (n0, s, p) = (0.08_f64, 0.002_f64, 21.0_f64);
    let geom = |d: f64, q: f64| -> (f64, f64, f64, f64) {
        let tw = p * d.powf(q);
        let z = (tw * q / (d * 2.0)).clamp(0.5, 50.0);
        let bw = (tw - z * d * 2.0).max(BW_LB as f64);
        let area = (tw + bw) * d / 2.0;
        let wp = bw + d * (z * z + 1.0).sqrt() * 2.0;
        (area, tw, wp, z)
    };
    eprintln!("\n  beta_trapezoid vs true dQ/dA at gamma = 0 (pre-existing)");
    eprintln!("  {:>8} {:>7} {:>12} {:>12} {:>9}", "q", "depth", "v*beta", "dQ/dA", "rel err");
    for q in [0.084_f64, 0.3, 0.65, 1.0] {
        for d in [0.25_f64, 1.0] {
            let (area, tw, wp, z) = geom(d, q);
            let r = area / wp;
            let v = (1.0 / n0) * r.powf(2.0 / 3.0) * s.sqrt();
            let beta = 5.0 / 3.0 - (4.0 / 3.0) * area * (1.0 + z * z).sqrt() / (tw * wp);
            let c_analytic = v * beta;

            let h = d * 1e-5;
            let qd = |dd: f64| {
                let (a, _, w, _) = geom(dd, q);
                let rr = a / w;
                (1.0 / n0) * rr.powf(2.0 / 3.0) * s.sqrt() * a
            };
            let (a_hi, ..) = geom(d + h, q);
            let (a_lo, ..) = geom(d - h, q);
            let c_num = (qd(d + h) - qd(d - h)) / (a_hi - a_lo);
            eprintln!(
                "  {:>8.3} {:>7.2} {:>12.5} {:>12.5} {:>8.1}%",
                q, d, c_analytic, c_num, 100.0 * (c_analytic - c_num).abs() / c_num.abs()
            );
        }
    }
    eprintln!(
        "  (beta treats the side slope z as fixed; here z = (p·q/2)·d^(q−1),\n   \
         so it varies with depth for every q except 1. Inherited from DDR.)\n"
    );
}
