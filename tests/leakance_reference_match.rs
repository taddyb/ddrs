//! Cross-implementation acceptance for the leakance (GW–SW exchange) term:
//! `src/routing/leakance.rs::zeta_forward` against DDR's `_compute_zeta`.
//!
//! Every other part of the routing core is anchored to the Python reference by
//! `examples/compare_ddr_sandbox` / `tests/ddr_sandbox_match.rs`. Leakance had
//! no such anchor, because DDR reverted the feature on master — its reference
//! lives only in commit `c2bd0f9` ("feat: add leakance (GW-SW exchange) to
//! routing (#130)", 2026-02-13). `scripts/export_ddr_leakance_reference.py`
//! extracts that function from git history, runs it on 29 hand-chosen reaches,
//! and writes `fixtures/leakance/ddr_reference_zeta.csv`, which this test reads.
//!
//! # The three known, deliberate ddrs deviations
//!
//! 1. `losing_only` (`src/config.rs::Params::leakance_losing_only`, default
//!    `true`) clamps the head at `max(0, depth − d_gw)`. The reference is
//!    sign-symmetric: negative flux means a gaining reach. The parity tests
//!    therefore run with `losing_only = false`, which IS the reference's form;
//!    `losing_only_clamp_is_a_ddrs_only_deviation` pins the clamp separately.
//! 2. The impervious hard-zero mask is a ddrs Phase C addition with no
//!    reference counterpart. The parity tests pass `mask = None`.
//! 3. ddrs raises the width to `q_eps = q_spatial + 1e-6` where the reference
//!    uses `q_spatial`. This is the same stabilisation the base solver already
//!    applies (`src/geometry.rs::compute_trapezoidal_geometry_gamma`), which
//!    the sandbox comparison already accepts.
//!
//! # Where the tolerance comes from (deviation 3, derived not tuned)
//!
//! `zeta = f·L·K_D·(d − d_gw)·(p·d)^q`, so perturbing the width exponent by
//! `eps = 1e-6` changes the flux by a relative `eps·ln(p·d)`. Over this fixture
//! `p·d` peaks at `21 · 21.75 ≈ 4.6e2` (the `q_1e4` reach), giving
//! `|ln(p·d)| <= 6.2` and a predicted relative difference of `6.2e-6`.
//! The full-chain test additionally carries `eps` through the depth inversion:
//! the numerator's `(q+1)` contributes `eps/(1+q) <= 1e-6` and the exponent
//! `3/(5+3q)` shifts by `-9·eps/(5+3q)^2 ≈ -3.6e-7`, which multiplies
//! `ln(ratio) = ln(depth)/exponent`, at most ≈ 9.4 here, for ≈ `3.4e-6` in
//! depth; the flux amplifies that by `d/(d − d_gw) + q <= 2.8` on this fixture.
//! Both paths therefore predict single-digit `1e-6` relative differences, on
//! top of an f32 evaluation floor of ~1e-7 per op. `REL_TOL = 2e-5` is that
//! prediction with roughly 3x headroom. It is NOT a bar chosen to make the
//! test pass: if a measured difference approaches it, the q_eps stabilisation
//! is no longer the explanation and the discrepancy is real.
//!
//! # Not covered by parity, on purpose
//!
//! The reference does not bound `depth`; ddrs shares the routing core's
//! `depth = max(ratio^exponent, attribute_minimums.depth)` (S6 of
//! `src/routing/mmc_op.rs::forward_chain_inner`). Reaches whose reference
//! depth falls below that floor are excluded from the full-chain parity max
//! and asserted separately by `depth_floor_reach_is_a_ddrs_only_deviation`.

use std::path::Path;

use burn::backend::NdArray;
use burn::tensor::backend::BackendTypes;
use burn::tensor::Tensor;

use ddrs::geometry::compute_trapezoidal_geometry;
use ddrs::routing::leakance::zeta_forward;

type B = NdArray<f32>;
type D = <B as BackendTypes>::Device;

/// Derived in the module docs from the `q_eps = q_spatial + 1e-6` stabilisation.
const REL_TOL: f32 = 2e-5;
/// Fluxes smaller than this are compared on absolute difference only; the
/// smallest non-zero reference flux in the fixture is 6.7e-8 m³/s, so this
/// only ever catches the `leakance_factor = 0` reach (exactly 0 on both sides).
const ABS_FLOOR: f32 = 1e-12;

/// `ddrs`'s `attribute_minimums.bottom_width`. Does not enter the depth
/// inversion; passed only because `compute_trapezoidal_geometry` needs it.
const BOTTOM_WIDTH_LB: f32 = 0.01;

struct Fixture {
    case: Vec<String>,
    q_t: Vec<f32>,
    n: Vec<f32>,
    q_spatial: Vec<f32>,
    s0: Vec<f32>,
    p_spatial: Vec<f32>,
    length: Vec<f32>,
    k_d: Vec<f32>,
    d_gw: Vec<f32>,
    leakance_factor: Vec<f32>,
    ref_depth: Vec<f32>,
    ref_zeta: Vec<f32>,
    depth_lb: f32,
}

impl Fixture {
    fn len(&self) -> usize {
        self.case.len()
    }
}

fn load() -> Fixture {
    let path = Path::new("fixtures/leakance/ddr_reference_zeta.csv");
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "read {path:?}: {e} — fixtures/leakance is committed; a missing file \
             means a broken checkout, not a skippable test. Regenerate with \
             `cd ~/projects/ddr && uv run python \
             ~/projects/ddrs/scripts/export_ddr_leakance_reference.py`"
        )
    });

    let mut f = Fixture {
        case: vec![],
        q_t: vec![],
        n: vec![],
        q_spatial: vec![],
        s0: vec![],
        p_spatial: vec![],
        length: vec![],
        k_d: vec![],
        d_gw: vec![],
        leakance_factor: vec![],
        ref_depth: vec![],
        ref_zeta: vec![],
        depth_lb: f32::NAN,
    };
    let mut seen_header = false;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("# ddrs_depth_lb = ") {
            f.depth_lb = rest.parse().expect("parse ddrs_depth_lb");
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !seen_header {
            assert!(
                line.starts_with("case,"),
                "unexpected header in {path:?}: {line}"
            );
            seen_header = true;
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        assert_eq!(c.len(), 14, "wrong column count in {path:?}: {line}");
        let num =
            |i: usize| -> f32 { c[i].parse().unwrap_or_else(|e| panic!("parse {:?}: {e}", c[i])) };
        f.case.push(c[0].to_string());
        f.q_t.push(num(1));
        f.n.push(num(2));
        f.q_spatial.push(num(3));
        f.s0.push(num(4));
        f.p_spatial.push(num(5));
        f.length.push(num(6));
        f.k_d.push(num(7));
        f.d_gw.push(num(8));
        f.leakance_factor.push(num(9));
        f.ref_depth.push(num(10));
        f.ref_zeta.push(num(11));
    }
    assert!(seen_header && f.len() > 0, "no rows parsed from {path:?}");
    assert!(
        f.depth_lb.is_finite(),
        "fixture header is missing `# ddrs_depth_lb`"
    );
    f
}

fn t(v: &[f32], device: &D) -> Tensor<B, 1> {
    Tensor::from_floats(v, device)
}

/// `zeta_forward` on every fixture reach, in the reference's form: the clamp
/// disabled and no impervious mask. `depth` is supplied by the caller.
fn ddrs_zeta(f: &Fixture, depth: &[f32], device: &D) -> Vec<f32> {
    let q_eps: Vec<f32> = f.q_spatial.iter().map(|q| q + 1e-6).collect();
    let (_w, _a, zeta) = zeta_forward::<B>(
        t(depth, device),
        t(&f.p_spatial, device),
        t(&q_eps, device),
        t(&f.length, device),
        t(&f.k_d, device),
        t(&f.d_gw, device),
        t(&f.leakance_factor, device),
        false, // deviation 1: reference form, sign-symmetric
        // deviation 3: no disconnection cap. The DDR c2bd0f9 reference has a
        // purely linear head, so the cap MUST be off for this comparison; with
        // it on the two would diverge wherever d_gw < -M, by design.
        None,
        None,  // deviation 2: no impervious mask
    );
    zeta.into_data().to_vec().unwrap()
}

/// Max abs / max rel difference plus the worst reach's index, over `keep`.
fn compare(f: &Fixture, got: &[f32], keep: &dyn Fn(usize) -> bool) -> (f32, f32, usize) {
    let (mut max_abs, mut max_rel, mut worst) = (0.0_f32, 0.0_f32, 0usize);
    for i in 0..f.len() {
        if !keep(i) {
            continue;
        }
        let (a, b) = (f.ref_zeta[i], got[i]);
        assert!(b.is_finite(), "reach {} ({}): non-finite zeta {b}", i, f.case[i]);
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        if a.abs() > ABS_FLOOR {
            max_rel = max_rel.max(d / a.abs());
        } else {
            assert_eq!(
                a, b,
                "reach {} ({}): reference flux is 0, ddrs is {b}",
                i, f.case[i]
            );
        }
    }
    (max_abs, max_rel, worst)
}

/// Tail chain only: the reference's own unclamped depth feeds `zeta_forward`,
/// so `width -> area -> zeta` is isolated from the depth inversion. The only
/// expected difference is the `q_eps` width exponent (deviation 3).
#[test]
fn zeta_forward_matches_ddr_reference_on_reference_depth() {
    let f = load();
    let device = D::default();
    let got = ddrs_zeta(&f, &f.ref_depth, &device);
    let (max_abs, max_rel, worst) = compare(&f, &got, &|_| true);

    eprintln!(
        "leakance tail parity (reference depth): max abs diff {max_abs:.6e} m³/s, \
         max rel diff {max_rel:.6e} (worst reach {worst} `{}`), {} reaches",
        f.case[worst],
        f.len()
    );
    eprintln!(
        "  verdict: {}",
        if max_rel < REL_TOL {
            "REFERENCE MATCH (max rel < 2e-5, the q_eps width-exponent bound)"
        } else {
            "MISMATCH"
        }
    );
    assert!(
        max_rel < REL_TOL,
        "leakance tail chain disagrees with DDR `_compute_zeta` @ c2bd0f9: max rel \
         diff {max_rel:.6e} (reach {worst} `{}`, abs {max_abs:.6e} m³/s) >= {REL_TOL:.1e}. \
         That is beyond the q_eps stabilisation's predicted ~6.2e-6 — treat it as a \
         real discrepancy in `src/routing/leakance.rs::zeta_forward`, not a tolerance \
         to widen.",
        f.case[worst]
    );
}

/// Full chain: ddrs's own depth inversion (`src/geometry.rs::
/// compute_trapezoidal_geometry`, the S2..S6 spelling shared with
/// `src/routing/mmc_op.rs::forward_chain_inner`) feeds `zeta_forward`.
/// Reaches saturated by the depth floor are excluded here and asserted by
/// `depth_floor_reach_is_a_ddrs_only_deviation`.
#[test]
fn full_chain_matches_ddr_reference() {
    let f = load();
    let device = D::default();
    let geom = compute_trapezoidal_geometry::<B>(
        t(&f.n, &device),
        t(&f.p_spatial, &device),
        t(&f.q_spatial, &device),
        t(&f.q_t, &device),
        t(&f.s0, &device),
        f.depth_lb,
        BOTTOM_WIDTH_LB,
    );
    let depth: Vec<f32> = geom.depth.into_data().to_vec().unwrap();
    let got = ddrs_zeta(&f, &depth, &device);

    let floored: Vec<usize> = (0..f.len()).filter(|&i| f.ref_depth[i] < f.depth_lb).collect();
    let (max_abs, max_rel, worst) = compare(&f, &got, &|i| !floored.contains(&i));

    eprintln!(
        "leakance full-chain parity (ddrs depth): max abs diff {max_abs:.6e} m³/s, \
         max rel diff {max_rel:.6e} (worst reach {worst} `{}`), {} of {} reaches \
         ({} excluded at the depth floor)",
        f.case[worst],
        f.len() - floored.len(),
        f.len(),
        floored.len()
    );
    eprintln!(
        "  verdict: {}",
        if max_rel < REL_TOL {
            "REFERENCE MATCH (max rel < 2e-5, the q_eps width-exponent bound)"
        } else {
            "MISMATCH"
        }
    );
    assert!(
        max_rel < REL_TOL,
        "leakance full chain disagrees with DDR `_compute_zeta` @ c2bd0f9: max rel \
         diff {max_rel:.6e} (reach {worst} `{}`, abs {max_abs:.6e} m³/s) >= {REL_TOL:.1e}. \
         That is beyond the q_eps stabilisation's predicted ~6.2e-6 — treat it as a \
         real discrepancy, not a tolerance to widen.",
        f.case[worst]
    );
}

/// The reference is sign-symmetric; the fixture's gaining reaches carry a
/// negative flux and ddrs reproduces the sign under `losing_only = false`.
#[test]
fn gaining_reaches_reproduce_the_reference_sign() {
    let f = load();
    let device = D::default();
    let got = ddrs_zeta(&f, &f.ref_depth, &device);
    let gaining: Vec<usize> = (0..f.len()).filter(|&i| f.ref_zeta[i] < 0.0).collect();
    assert!(
        gaining.len() >= 2,
        "fixture must exercise gaining reaches (depth < d_gw); found {}",
        gaining.len()
    );
    for i in gaining {
        assert!(
            got[i] < 0.0,
            "reach {i} (`{}`): reference flux {:.6e} is gaining, ddrs gave {:.6e}",
            f.case[i],
            f.ref_zeta[i],
            got[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Deliberate ddrs-only behaviour. These are NOT parity: they pin differences
// from the reference that exist on purpose. A change here is a change of
// intent, not a regression against DDR.
// ---------------------------------------------------------------------------

/// `losing_only = true` (the ddrs default) zeroes every gaining reach while
/// leaving losing reaches bit-identical to the reference form.
#[test]
fn losing_only_clamp_is_a_ddrs_only_deviation() {
    let f = load();
    let device = D::default();
    let unclamped = ddrs_zeta(&f, &f.ref_depth, &device);

    let q_eps: Vec<f32> = f.q_spatial.iter().map(|q| q + 1e-6).collect();
    let (_w, _a, z) = zeta_forward::<B>(
        t(&f.ref_depth, &device),
        t(&f.p_spatial, &device),
        t(&q_eps, &device),
        t(&f.length, &device),
        t(&f.k_d, &device),
        t(&f.d_gw, &device),
        t(&f.leakance_factor, &device),
        true, // the ddrs default
        None, // no disconnection cap: this test pins the losing-only clamp alone
        None,
    );
    let clamped: Vec<f32> = z.into_data().to_vec().unwrap();

    let mut n_zeroed = 0;
    for i in 0..f.len() {
        if f.ref_zeta[i] < 0.0 {
            assert_eq!(
                clamped[i], 0.0,
                "reach {i} (`{}`) is gaining; `losing_only` must zero it, got {:.6e}",
                f.case[i], clamped[i]
            );
            n_zeroed += 1;
        } else {
            assert_eq!(
                clamped[i], unclamped[i],
                "reach {i} (`{}`) is losing; `losing_only` must not change it",
                f.case[i]
            );
        }
    }
    assert!(
        n_zeroed >= 2,
        "expected the clamp to zero the fixture's gaining reaches"
    );
    eprintln!(
        "losing_only (ddrs default, NOT the reference): {n_zeroed} of {} reaches \
         zeroed; the reference would return a negative (gaining) flux there",
        f.len()
    );
}

/// The depth floor has no reference counterpart, so the floored reach's flux
/// is larger than the reference's by exactly `(depth_lb/depth_ref)^(1+q)`.
#[test]
fn depth_floor_reach_is_a_ddrs_only_deviation() {
    let f = load();
    let device = D::default();
    let geom = compute_trapezoidal_geometry::<B>(
        t(&f.n, &device),
        t(&f.p_spatial, &device),
        t(&f.q_spatial, &device),
        t(&f.q_t, &device),
        t(&f.s0, &device),
        f.depth_lb,
        BOTTOM_WIDTH_LB,
    );
    let depth: Vec<f32> = geom.depth.into_data().to_vec().unwrap();
    let got = ddrs_zeta(&f, &depth, &device);

    let floored: Vec<usize> = (0..f.len()).filter(|&i| f.ref_depth[i] < f.depth_lb).collect();
    assert!(
        !floored.is_empty(),
        "fixture must contain at least one reach below the depth floor ({} m)",
        f.depth_lb
    );
    for i in floored {
        assert_eq!(depth[i], f.depth_lb, "reach {i} (`{}`) must saturate", f.case[i]);
        // zeta ∝ (p·d)^q · (d − d_gw); with d_gw = 0 the ratio is (d_lb/d_ref)^(1+q).
        assert_eq!(f.d_gw[i], 0.0, "ratio identity below assumes d_gw = 0");
        let expected = (f.depth_lb / f.ref_depth[i]).powf(1.0 + f.q_spatial[i]);
        let ratio = got[i] / f.ref_zeta[i];
        assert!(
            (ratio - expected).abs() / expected < 1e-4,
            "reach {i} (`{}`): flux ratio vs reference {ratio:.6} should be \
             (d_lb/d_ref)^(1+q) = {expected:.6}",
            f.case[i]
        );
        eprintln!(
            "depth floor (ddrs-only): reach {i} (`{}`) reference depth {:.6e} m < \
             floor {} m, so ddrs's flux is {ratio:.3}x the reference's",
            f.case[i], f.ref_depth[i], f.depth_lb
        );
    }
}
