//! Leakance (GW–SW water-loss) term `zeta`, ported from DDR `_compute_zeta`
//! (`~/projects/ddr/src/ddr/routing/mmc.py:146-197`, commit c2bd0f9).
//!
//! `zeta = leakance_factor · area_z · K_D · (depth − d_gw)`, where
//! `width_z = (p·depth)^q_eps`, `area_z = width_z · length`, and `depth` is the
//! SHARED power-law depth already computed by `forward_chain_inner` (S6).
//! Subtracted from `b_rhs`. Positive ⇒ losing stream. All ops are plain inner-
//! backend `Tensor<I,1>` (no autograd tape).

use burn::tensor::{backend::Backend, Tensor};

/// `(width_z, area_z, zeta)` from the shared `depth` and the three leakance
/// params. `q_eps = q_spatial + 1e-6` (consistency with the shared depth).
pub fn zeta_forward<I: Backend>(
    depth: Tensor<I, 1>,
    p_spatial: Tensor<I, 1>,
    q_eps: Tensor<I, 1>,
    length: Tensor<I, 1>,
    k_d: Tensor<I, 1>,
    d_gw: Tensor<I, 1>,
    leakance_factor: Tensor<I, 1>,
) -> (Tensor<I, 1>, Tensor<I, 1>, Tensor<I, 1>) {
    let p_depth = p_spatial * depth.clone();
    let width_z = p_depth.powf(q_eps);
    let area_z = width_z.clone() * length;
    let m = depth - d_gw;
    let zeta = leakance_factor * area_z.clone() * k_d * m;
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
        // m       = 2−1 = 1
        // zeta    = 0.5·4472.136·1e-6·1 = 0.002236068
        let (w, a, z) = zeta_forward::<B>(
            t(&[2.0]), t(&[10.0]), t(&[0.5]), t(&[1000.0]),
            t(&[1e-6]), t(&[1.0]), t(&[0.5]),
        );
        assert!((w.into_scalar() - 4.472_136).abs() < 1e-4);
        assert!((a.into_scalar() - 4472.136).abs() < 1e-1);
        assert!((z.into_scalar() - 0.002_236_068).abs() < 1e-7);
    }

    #[test]
    fn gaining_stream_is_negative() {
        // depth < d_gw ⇒ m < 0 ⇒ zeta < 0 (gaining stream).
        let (_, _, z) = zeta_forward::<B>(
            t(&[1.0]), t(&[10.0]), t(&[0.5]), t(&[1000.0]),
            t(&[1e-6]), t(&[3.0]), t(&[1.0]),
        );
        assert!(z.into_scalar() < 0.0);
    }
}

/// Per-parent gradient contributions of `zeta`. `g_b` is ∂L/∂b_rhs; since
/// `b_rhs = … − zeta`, `gzeta = −g_b`. Returns grads for the three leakance
/// params plus zeta's contributions into `depth`, `p_spatial`, `q_eps`.
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
) -> ZetaGrads<I> {
    let gzeta = -g_b;
    let m = depth.clone() - d_gw;
    let g_leakance_factor = gzeta.clone() * area_z.clone() * k_d.clone() * m.clone();
    let g_k_d = gzeta.clone() * leakance_factor.clone() * area_z.clone() * m.clone();
    let g_d_gw = -(gzeta.clone() * leakance_factor.clone() * area_z.clone() * k_d.clone());
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
mod grad_tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    fn s(v: f32) -> Tensor<B, 1> { Tensor::from_floats(&[v][..], &Default::default()) }
    fn val(t: Tensor<B, 1>) -> f32 { t.into_scalar() }

    // Scalar zeta as a plain f64 closure for central differences.
    #[allow(clippy::too_many_arguments)]
    fn zeta_scalar(depth: f64, p: f64, q_eps: f64, length: f64, k_d: f64, d_gw: f64, factor: f64) -> f64 {
        let width_z = (p * depth).powf(q_eps);
        let area_z = width_z * length;
        factor * area_z * k_d * (depth - d_gw)
    }

    #[test]
    fn zeta_grads_match_central_differences() {
        // Base point.
        let (depth, p, q_eps, length, k_d, d_gw, factor) =
            (2.0_f64, 10.0, 0.5, 1000.0, 1e-6, 1.0, 0.5);
        let area_z = (p * depth).powf(q_eps) * length;
        // g_b = 1 ⇒ gzeta = −1, so analytical grads below are −∂zeta/∂param.
        let g = zeta_backward::<B>(
            s(1.0), s(depth as f32), s(p as f32), s(q_eps as f32),
            s(area_z as f32), s(k_d as f32), s(d_gw as f32), s(factor as f32),
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
