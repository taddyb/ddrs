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
