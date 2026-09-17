//! Leakance (GW–SW water-loss) term `zeta`, ported from DDR `_compute_zeta`
//! (`~/projects/ddr/src/ddr/routing/mmc.py:146-197`, commit c2bd0f9).
//!
//! `zeta = leakance_factor · area_z · K_D · head`, where
//! `head = max(0, depth − d_gw)` (losing-only clamp, default) or
//! `head = depth − d_gw` (unclamped, back-compat when `losing_only=false`).
//! With `bed_thickness = Some(M)` the head is additionally capped at
//! `depth + M`, the point at which the stream disconnects from the aquifer and
//! the flux stops depending on how far below the bed the water table sits.
//! `width_z = (p·depth)^q_eps`, `area_z = width_z · length`, and `depth` is the
//! SHARED power-law depth already computed by `forward_chain_inner` (S6).
//! Subtracted from `b_rhs`. Positive ⇒ losing stream. All ops are plain inner-
//! backend `Tensor<I,1>` (no autograd tape).

use burn::tensor::{backend::Backend, Tensor};

/// `(width_z, area_z, zeta)` from the shared `depth` and the three leakance
/// params. `q_eps = q_spatial + 1e-6` (consistency with the shared depth).
///
/// `losing_only=true` (Phase C default): clamps the head term to
/// `max(0, depth − d_gw)` so gaining reaches (depth ≤ d_gw) produce zeta ≡ 0.
/// `losing_only=false`: unclamped `depth − d_gw` (back-compat with prior runs).
///
/// `mask`: optional per-reach 0/1 constant (not autograd-tracked). When
/// `mask[i] = 0.0`, the reach is treated as impervious and `zeta[i] ≡ 0`.
/// `None` (or all-ones mask) ⇒ no-op, byte-identical to the no-mask path.
pub fn zeta_forward<I: Backend>(
    depth: Tensor<I, 1>,
    p_spatial: Tensor<I, 1>,
    q_eps: Tensor<I, 1>,
    length: Tensor<I, 1>,
    k_d: Tensor<I, 1>,
    d_gw: Tensor<I, 1>,
    leakance_factor: Tensor<I, 1>,
    losing_only: bool,
    bed_thickness: Option<f32>,
    mask: Option<Tensor<I, 1>>,
) -> (Tensor<I, 1>, Tensor<I, 1>, Tensor<I, 1>) {
    let p_depth = p_spatial * depth.clone();
    let width_z = p_depth.powf(q_eps);
    let area_z = width_z.clone() * length;
    // DISCONNECTION CAP. Once the water table falls below the streambed by more
    // than the bed thickness M, the stream and the aquifer are no longer
    // hydraulically connected: an unsaturated zone opens beneath the bed, and
    // the driving head is the head across the BED LAYER alone, `depth + M`, not
    // the distance down to the table. Flux stops depending on `d_gw` entirely.
    // This is the MODFLOW river package's RBOT behaviour.
    //
    // Without the cap the linear head `depth − d_gw` grows without bound, which
    // is both wrong and harmful: at depth ≈ 1.5 m a `d_gw` of −80 m gives 12x
    // the flux of −5 m for the same conductance, and a large head with a small
    // `K_D` is indistinguishable from a small head with a large one. Widening
    // the `d_gw` box under a linear head therefore buys nothing but degeneracy.
    //
    // `None` ⇒ no cap, byte-identical to every run before 2026-09-17 and to the
    // DDR c2bd0f9 reference that `tests/leakance_reference_match.rs` pins.
    let m_raw = depth.clone() - d_gw;
    let m = match bed_thickness {
        Some(bed_m) => m_raw.min_pair(depth.clone() + bed_m),
        None => m_raw,
    };
    let m = if losing_only { m.clamp_min(0.0) } else { m };
    let zeta = leakance_factor * area_z.clone() * k_d * m;
    let zeta = match mask {
        Some(msk) => zeta * msk,
        None => zeta,
    };
    (width_z, area_z, zeta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    fn t(v: &[f32]) -> Tensor<B, 1> {
        Tensor::from_floats(v, &Default::default())
    }

    #[test]
    fn zeta_matches_hand_computed_value() {
        // depth=2, p=10, q_eps=0.5, length=1000, K_D=1e-6, d_gw=1, factor=0.5
        // width_z = (10·2)^0.5 = sqrt(20) = 4.472136
        // area_z  = 4.472136·1000 = 4472.136
        // m       = 2−1 = 1  (losing: depth > d_gw, clamp inactive)
        // zeta    = 0.5·4472.136·1e-6·1 = 0.002236068
        let (w, a, z) = zeta_forward::<B>(
            t(&[2.0]), t(&[10.0]), t(&[0.5]), t(&[1000.0]),
            t(&[1e-6]), t(&[1.0]), t(&[0.5]),
            false,  // unclamped path for hand-computed test
            None,   // no disconnection cap
            None,   // no impervious mask
        );
        assert!((w.into_scalar() - 4.472_136).abs() < 1e-4);
        assert!((a.into_scalar() - 4472.136).abs() < 1e-1);
        assert!((z.into_scalar() - 0.002_236_068).abs() < 1e-7);
    }

    #[test]
    fn gaining_stream_is_negative_unclamped() {
        // depth < d_gw ⇒ m < 0 ⇒ zeta < 0 (unclamped path).
        let (_, _, z) = zeta_forward::<B>(
            t(&[1.0]), t(&[10.0]), t(&[0.5]), t(&[1000.0]),
            t(&[1e-6]), t(&[3.0]), t(&[1.0]),
            false,
            None,
            None,
        );
        assert!(z.into_scalar() < 0.0);
    }

    #[test]
    fn gaining_stream_clamped_to_zero() {
        // depth < d_gw with losing_only=true ⇒ zeta ≡ 0.
        let (_, _, z) = zeta_forward::<B>(
            t(&[1.0]), t(&[10.0]), t(&[0.5]), t(&[1000.0]),
            t(&[1e-6]), t(&[3.0]), t(&[1.0]),
            true,
            None,
            None,
        );
        assert_eq!(z.into_scalar(), 0.0);
    }

    #[test]
    fn losing_stream_clamped_unchanged() {
        // depth > d_gw with losing_only=true ⇒ same as unclamped.
        let (_, _, z_clamped) = zeta_forward::<B>(
            t(&[2.0]), t(&[10.0]), t(&[0.5]), t(&[1000.0]),
            t(&[1e-6]), t(&[1.0]), t(&[0.5]),
            true,
            None,
            None,
        );
        let (_, _, z_unclamped) = zeta_forward::<B>(
            t(&[2.0]), t(&[10.0]), t(&[0.5]), t(&[1000.0]),
            t(&[1e-6]), t(&[1.0]), t(&[0.5]),
            false,
            None,
            None,
        );
        assert!((z_clamped.into_scalar() - z_unclamped.into_scalar()).abs() < 1e-9);
    }
}

/// Per-parent gradient contributions of `zeta`. `g_b` is ∂L/∂b_rhs; since
/// `b_rhs = … − zeta`, `gzeta = −g_b`. Returns grads for the three leakance
/// params plus zeta's contributions into `depth`, `p_spatial`, `q_eps`.
///
/// `losing_only=true`: the forward used `head = max(0, depth − d_gw)`, so the
/// backward gates all grads to zero where `depth ≤ d_gw` (gaining reaches).
/// Subgradient at the kink (depth == d_gw) is defined as 0 (measure-zero;
/// finite-difference safe). `losing_only=false`: unclamped backward.
///
/// `mask`: same 0/1 constant passed to the forward. `mask[i]=0` ⇒ the
/// forward multiplied zeta by 0 at reach i, so `∂L/∂(leakance_params[i]) = 0`
/// here too. Applied as a multiplier on `gzeta` before all grad computations.
/// `None` ⇒ no-op (no mask was applied in the forward).
pub struct ZetaGrads<I: Backend> {
    pub g_k_d: Tensor<I, 1>,
    pub g_d_gw: Tensor<I, 1>,
    pub g_leakance_factor: Tensor<I, 1>,
    pub g_depth: Tensor<I, 1>,
    pub g_p_spatial: Tensor<I, 1>,
    pub g_q_eps: Tensor<I, 1>,
}

#[allow(clippy::too_many_arguments)]
pub fn zeta_backward<I: Backend>(
    g_b: Tensor<I, 1>,
    depth: Tensor<I, 1>,
    p_spatial: Tensor<I, 1>,
    q_eps: Tensor<I, 1>,
    area_z: Tensor<I, 1>,
    k_d: Tensor<I, 1>,
    d_gw: Tensor<I, 1>,
    leakance_factor: Tensor<I, 1>,
    losing_only: bool,
    bed_thickness: Option<f32>,
    mask: Option<Tensor<I, 1>>,
) -> ZetaGrads<I> {
    // When losing_only: gate gzeta to zero where depth ≤ d_gw (gaining reaches).
    // Since every grad is a product of gzeta, gating it zeros all outputs at once.
    // Subgradient at the kink (depth == d_gw) = 0 by convention.
    let gzeta = if losing_only {
        let gate = depth.clone().greater(d_gw.clone()); // true where depth > d_gw
        (-g_b).mask_fill(gate.bool_not(), 0.0)
    } else {
        -g_b
    };
    // Impervious mask: forward multiplied zeta by mask, so chain rule multiplies
    // gzeta by mask too. mask[i]=0 ⇒ all leakance-param grads at reach i are 0.
    let gzeta = match mask {
        Some(msk) => gzeta * msk,
        None => gzeta,
    };
    // The head the forward actually used. Under the disconnection cap this is
    // NOT `depth − d_gw`: every grad below scales with the CAPPED head, and the
    // old code could get away with the raw difference only because
    // `losing_only` zeroes `gzeta` exactly where the raw and clamped heads
    // differ. The cap binds where `gzeta` is nonzero, so it must be applied here.
    let m_raw = depth.clone() - d_gw.clone();
    let (m, connected) = match bed_thickness {
        Some(bed_m) => {
            let cap = depth.clone() + bed_m;
            // Connected ⇔ the cap is NOT binding ⇔ depth − d_gw < depth + M
            // ⇔ d_gw > −M. At the kink the subgradient is taken as
            // disconnected (∂h/∂d_gw = 0), matching the `losing_only`
            // convention of resolving kinks to zero.
            let conn = m_raw.clone().lower(cap.clone());
            (m_raw.min_pair(cap), Some(conn))
        }
        None => (m_raw, None),
    };
    let g_leakance_factor = gzeta.clone() * area_z.clone() * k_d.clone() * m.clone();
    let g_k_d = gzeta.clone() * leakance_factor.clone() * area_z.clone() * m.clone();
    // ∂h/∂d_gw is −1 while connected and 0 once disconnected: that is the whole
    // point of the cap, and it is the ONLY gradient the cap changes. ∂h/∂depth
    // is 1 in BOTH branches (`d(depth+M)/d(depth) = 1`), so `g_depth` below is
    // untouched.
    let g_d_gw = -(gzeta.clone() * leakance_factor.clone() * area_z.clone() * k_d.clone());
    let g_d_gw = match &connected {
        Some(conn) => g_d_gw.mask_fill(conn.clone().bool_not(), 0.0),
        None => g_d_gw,
    };
    let common = gzeta * leakance_factor * k_d; // = ∂zeta/∂area_z (× m below)
    let common_m = common.clone() * m.clone();
    let g_p_spatial = common_m.clone() * area_z.clone() * q_eps.clone() / p_spatial.clone();
    let g_q_eps = common_m.clone() * area_z.clone() * (p_spatial * depth.clone()).log();
    // ∂zeta/∂depth = factor·K_D·area_z (direct m) + factor·K_D·m·(area_z·q_eps/depth)
    let g_depth = common.clone() * area_z.clone()
        + common_m * area_z * q_eps / depth;
    ZetaGrads { g_k_d, g_d_gw, g_leakance_factor, g_depth, g_p_spatial, g_q_eps }
}

#[cfg(test)]
mod cap_tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    fn s1(v: f32) -> Tensor<B, 1> { Tensor::from_floats(&[v][..], &Default::default()) }
    fn val(t: Tensor<B, 1>) -> f32 { t.into_scalar() }

    #[allow(clippy::too_many_arguments)]
    fn zeta(depth: f32, d_gw: f32, bed: Option<f32>) -> f32 {
        let (_w, _a, z) = zeta_forward::<B>(
            s1(depth), s1(10.0), s1(0.5), s1(1000.0),
            s1(1e-6), s1(d_gw), s1(0.5),
            true, bed, None,
        );
        val(z)
    }

    /// `None` must leave the head exactly as it was. This is what keeps the
    /// DDR c2bd0f9 reference match and every pre-2026-09-17 leakance run valid.
    #[test]
    fn absent_bed_thickness_is_bit_identical() {
        for d_gw in [-80.0, -5.0, -1.0, 0.0, 0.5] {
            let (_w, _a, uncapped) = zeta_forward::<B>(
                s1(2.0), s1(10.0), s1(0.5), s1(1000.0),
                s1(1e-6), s1(d_gw), s1(0.5), true, None, None,
            );
            // Recompute the pre-cap expression by hand.
            let area_z = (10.0f32 * 2.0).powf(0.5) * 1000.0;
            let expect = 0.5 * area_z * 1e-6 * (2.0f32 - d_gw).max(0.0);
            assert_eq!(val(uncapped), expect, "d_gw = {d_gw}");
        }
    }

    /// Once the table is more than M below the bed the stream is disconnected
    /// and the flux STOPS depending on how far down it is. Two very different
    /// `d_gw` values must give the same zeta.
    #[test]
    fn disconnected_flux_is_independent_of_how_far_the_table_sits() {
        let m = 1.0;
        let a = zeta(2.0, -5.0, Some(m));
        let b = zeta(2.0, -80.0, Some(m));
        let c = zeta(2.0, -150.0, Some(m));
        assert_eq!(a, b, "-5 m and -80 m must give the same disconnected flux");
        assert_eq!(b, c, "-80 m and -150 m must give the same disconnected flux");
        // ...and it equals the head across the bed layer alone, depth + M.
        let area_z = (10.0f32 * 2.0).powf(0.5) * 1000.0;
        assert!((a - 0.5 * area_z * 1e-6 * (2.0 + m)).abs() < 1e-6 * a.abs().max(1.0));
    }

    /// Without the cap that same widening inflates the flux without bound,
    /// which is the behaviour the cap exists to remove. Guards against the cap
    /// being silently disabled.
    #[test]
    fn uncapped_flux_grows_without_bound_as_the_table_drops() {
        let shallow = zeta(2.0, -5.0, None);
        let deep = zeta(2.0, -80.0, None);
        assert!(
            deep > 10.0 * shallow,
            "uncapped head should grow roughly 12x from -5 m to -80 m, got {deep} vs {shallow}"
        );
    }

    /// The connection test is `d_gw > -M` and does NOT involve depth: the cap
    /// `depth + M` and the head `depth - d_gw` both carry depth linearly, so it
    /// cancels. Depth changing must not flip a reach between regimes.
    #[test]
    fn connection_depends_on_d_gw_and_m_only_not_on_depth() {
        let m = 1.0;
        for depth in [0.5, 2.0, 8.0] {
            // d_gw = -0.5 > -M: connected, so the cap is inactive and the
            // capped result equals the uncapped one.
            assert_eq!(zeta(depth, -0.5, Some(m)), zeta(depth, -0.5, None), "depth {depth}");
            // d_gw = -2.0 < -M: disconnected at every depth.
            assert!(zeta(depth, -2.0, Some(m)) < zeta(depth, -2.0, None), "depth {depth}");
        }
    }

    /// The gradient consequence, and the reason the box must be centred in the
    /// connected region: `∂zeta/∂d_gw` is zero once disconnected, so a head
    /// initialised below `-M` would receive no gradient on `d_gw` at all.
    #[test]
    fn d_gw_gradient_vanishes_once_disconnected_and_survives_while_connected() {
        let m = 1.0;
        let g = |d_gw: f32| {
            let zg = zeta_backward::<B>(
                s1(1.0), s1(2.0), s1(10.0), s1(0.5),
                s1((10.0f32 * 2.0).powf(0.5) * 1000.0),
                s1(1e-6), s1(d_gw), s1(0.5), true, Some(m), None,
            );
            val(zg.g_d_gw)
        };
        assert_eq!(g(-2.0), 0.0, "disconnected reach must have zero d_gw gradient");
        assert_eq!(g(-80.0), 0.0, "deeply disconnected reach likewise");
        assert!(g(-0.5).abs() > 0.0, "connected reach must still carry a d_gw gradient");
        assert!(g(0.5).abs() > 0.0, "connected reach must still carry a d_gw gradient");
    }

    /// Central differences through the capped forward, in BOTH regimes and for
    /// every parameter whose gradient the cap touches. `d_gw` is the only one
    /// whose derivative changes; `K_D` and `leakance_factor` still scale with
    /// the (now capped) head, so they must use the capped value, not the raw
    /// difference.
    #[test]
    fn capped_gradients_match_central_differences() {
        let m = 1.0;
        let depth = 2.0_f64;
        let area_z = (10.0f64 * depth).powf(0.5) * 1000.0;
        let fwd = |k_d: f64, d_gw: f64, fac: f64| -> f64 {
            let head = (depth - d_gw).min(depth + m as f64).max(0.0);
            fac * area_z * k_d * head
        };
        for d_gw0 in [-0.5_f64, -2.0] {
            let (k0, f0) = (1e-6_f64, 0.5_f64);
            let zg = zeta_backward::<B>(
                s1(-1.0), s1(depth as f32), s1(10.0), s1(0.5), s1(area_z as f32),
                s1(k0 as f32), s1(d_gw0 as f32), s1(f0 as f32), true, Some(m), None,
            );
            // g_b = -1 ⇒ gzeta = +1, so grads are +∂zeta/∂param.
            let h = 1e-4;
            let fd_dgw = (fwd(k0, d_gw0 + h, f0) - fwd(k0, d_gw0 - h, f0)) / (2.0 * h);
            let fd_kd = (fwd(k0 * 1.5, d_gw0, f0) - fwd(k0 * 0.5, d_gw0, f0)) / (k0);
            let fd_fac = (fwd(k0, d_gw0, f0 + h) - fwd(k0, d_gw0, f0 - h)) / (2.0 * h);
            let near = |a: f32, b: f64, what: &str| {
                let d = (a as f64 - b).abs() / b.abs().max(1e-12);
                assert!(d < 5e-3, "{what} @ d_gw={d_gw0}: analytical {a:e} vs fd {b:e} (rel {d:e})");
            };
            near(val(zg.g_d_gw), fd_dgw, "g_d_gw");
            near(val(zg.g_k_d), fd_kd, "g_k_d");
            near(val(zg.g_leakance_factor), fd_fac, "g_leakance_factor");
        }
    }
}

#[cfg(test)]
mod grad_tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    fn s(v: f32) -> Tensor<B, 1> { Tensor::from_floats(&[v][..], &Default::default()) }
    fn val(t: Tensor<B, 1>) -> f32 { t.into_scalar() }

    // Scalar zeta as a plain f64 closure for central differences (unclamped).
    #[allow(clippy::too_many_arguments)]
    fn zeta_scalar(depth: f64, p: f64, q_eps: f64, length: f64, k_d: f64, d_gw: f64, factor: f64) -> f64 {
        let width_z = (p * depth).powf(q_eps);
        let area_z = width_z * length;
        factor * area_z * k_d * (depth - d_gw)
    }

    #[test]
    fn zeta_grads_match_central_differences() {
        // Base point: depth > d_gw (losing), unclamped path.
        let (depth, p, q_eps, length, k_d, d_gw, factor) =
            (2.0_f64, 10.0, 0.5, 1000.0, 1e-6, 1.0, 0.5);
        let area_z = (p * depth).powf(q_eps) * length;
        // g_b = 1 ⇒ gzeta = −1, so analytical grads below are −∂zeta/∂param.
        let g = zeta_backward::<B>(
            s(1.0), s(depth as f32), s(p as f32), s(q_eps as f32),
            s(area_z as f32), s(k_d as f32), s(d_gw as f32), s(factor as f32),
            false,  // unclamped — matches the zeta_scalar FD helper
            None,   // no disconnection cap
            None,   // no impervious mask
        );
        let h = 1e-4;
        let cd = |f: &dyn Fn(f64) -> f64, x: f64| (f(x + h) - f(x - h)) / (2.0 * h);

        // Each analytical grad equals −∂zeta/∂param (because gzeta = −1).
        let d_kd = cd(&|x| zeta_scalar(depth, p, q_eps, length, x, d_gw, factor), k_d);
        assert!(((val(g.g_k_d) as f64) - (-d_kd)).abs() / d_kd.abs().max(1.0) < 1e-3);

        let d_dgw = cd(&|x| zeta_scalar(depth, p, q_eps, length, k_d, x, factor), d_gw);
        assert!(((val(g.g_d_gw) as f64) - (-d_dgw)).abs() < 1e-7);

        let d_fac = cd(&|x| zeta_scalar(depth, p, q_eps, length, k_d, d_gw, x), factor);
        assert!(((val(g.g_leakance_factor) as f64) - (-d_fac)).abs() / d_fac.abs() < 1e-3);

        let d_p = cd(&|x| zeta_scalar(depth, x, q_eps, length, k_d, d_gw, factor), p);
        assert!(((val(g.g_p_spatial) as f64) - (-d_p)).abs() / d_p.abs() < 1e-2);

        let d_q = cd(&|x| zeta_scalar(depth, p, x, length, k_d, d_gw, factor), q_eps);
        assert!(((val(g.g_q_eps) as f64) - (-d_q)).abs() / d_q.abs() < 1e-2);

        let d_depth = cd(&|x| zeta_scalar(x, p, q_eps, length, k_d, d_gw, factor), depth);
        assert!(((val(g.g_depth) as f64) - (-d_depth)).abs() / d_depth.abs() < 1e-2);
    }
}
