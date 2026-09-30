//! Per-dam free parameters of the learned dam release
//! (`release_head.rule_curve`, `release_head.per_dam_t0`,
//! `release_head.flood_pool`).
//!
//! Offline, the harmonic rule curve's coefficients are not predictable from
//! the NID features (5-fold CV R² −0.13 to +0.15), so they cannot come from
//! the release head: each dam carries its own, calibrated through the gauge
//! loss. One row per dam of the FEATURE TABLE (`data_sources.reservoirs`, in
//! table order), so a checkpoint's rows are stable across batches:
//!
//! - `theta` `[n_dams, 4]`: the rule-curve logits. The coefficients are
//!   `c = rule_curve_max · tanh(θ)`, columns `(c1s, c1c, c2s, c2c)` of
//!   `r_d(t) = Ibar_d · Σ_{k=1,2} (c_{k,s} sin kω_t + c_{k,c} cos kω_t)`
//!   (`crate::routing::release`). `θ = 0` is no rule curve, bit for bit.
//! - `delta` `[n_dams]`: a per-dam log multiplier on the head's `T0`,
//!   `T0_d = T0_head,d · exp(δ_d)`. `δ = 0` is the head's `T0`.
//! - `pool` `[n_dams, 3]`: the flood pool's raw parameters `(r_kc, r_phi,
//!   r_z)` (law FA, `crate::routing::mmc::FloodPool`), through
//!   [`flood_pool_transform`]:
//!
//!   ```text
//!   kc  = clamp(3 · exp(r_kc), 0.5, 20)            release target Qc = kc·Ibar
//!   phi = sigmoid(r_phi)                           capture share, (0, 1)
//!   z   = 120 · sigmoid(4·r_z + logit(0.05/120))   pool size Fmax = z·Ibar, days
//!   ```
//!
//!   `r = 0` is the init: `kc = 3`, `phi = 0.5`, `z = 0.05 d` (72 minutes of
//!   mean inflow). That pool is OFF in effect (it holds at most 0.05 d of
//!   mean inflow per flood, against the offline fits' 5.7 to 59 d), but not
//!   dead: in the first flood hour above `Qc` it caps, so `z` gets a gradient
//!   through the cap, and wherever the capture or the evacuation does not
//!   cap, `kc` and `phi` get one too. `z` is logistic in `r_z` with rate 4:
//!   below ~30 d it is `0.05·exp(4·r_z)`, log-linear, so the per-dam Adam's
//!   travel of about 1.5 raw units in a 200-step smoke run (0.02 per update,
//!   ~75 updates per dam) spans 0.05 d to ~20 d, the offline median; with
//!   rate 1 it would reach 0.22 d. The logistic bounds `z` smoothly at 120 d
//!   (no dead zone), where the spec's `softplus` would need a hard clamp.
//!   `kc` is hard-clamped at its box (zero gradient outside it, ~1.8 raw
//!   units from the init either way). Since `r = 0` is the init,
//!   `per_dam_l2` pulls each pool back towards OFF.
//!
//! All start at zero. They have their own optimizer, a row-sparse Adam
//! (`crate::training::lazy_adam`: only rows with a nonzero gradient in a step
//! move, so a dam never in a batch stays exactly at zero), and a constant
//! learning rate (`release_head.per_dam_lr`): a dam's parameters get a
//! gradient only when a gauge below it is in the batch, so the heads'
//! schedule would barely move them. Saved as `release_dams.mpk` (+
//! `release_dams_optim.json`) next to `release_head.mpk`, so the release
//! head's own record and every older checkpoint are unchanged.

use burn::module::{Module, Param};
use burn::tensor::activation::sigmoid;
use burn::tensor::{backend::Backend, Int, Tensor};

/// Flood pool release target multiplier at `r_kc = 0` (`Qc = 3·Ibar`).
pub const POOL_KC_INIT: f32 = 3.0;
/// Box of the release target multiplier `kc`.
pub const POOL_KC_RANGE: [f32; 2] = [0.5, 20.0];
/// Flood pool size at `r_z = 0`, days of mean inflow.
pub const POOL_Z_INIT_DAYS: f32 = 0.05;
/// Upper bound of the pool size, days of mean inflow.
pub const POOL_Z_MAX_DAYS: f32 = 120.0;
/// Rate of the pool size's logistic in `r_z` (see the module docs).
pub const POOL_Z_RATE: f32 = 4.0;

/// See the module docs. A field is `None` when its option is off.
#[derive(Module, Debug)]
pub struct DamParams<B: Backend> {
    /// `[n_dams, 4]` rule-curve logits, `(c1s, c1c, c2s, c2c)`.
    pub theta: Option<Param<Tensor<B, 2>>>,
    /// `[n_dams]` log multiplier on `T0`.
    pub delta: Option<Param<Tensor<B, 1>>>,
    /// `[n_dams, 3]` flood pool raw parameters `(r_kc, r_phi, r_z)`.
    pub pool: Option<Param<Tensor<B, 2>>>,
}

/// The flood pool's physical parameters of a set of dams, each `[k]`.
pub struct PoolParams<B: Backend> {
    /// Release target multiplier, `Qc = kc·Ibar`.
    pub kc: Tensor<B, 1>,
    /// Capture share of the inflow above `Qc`.
    pub phi: Tensor<B, 1>,
    /// Pool size in days of mean inflow, `Fmax = z·Ibar·86400` m³.
    pub z_days: Tensor<B, 1>,
}

/// `(kc, phi, z)` from the raw pool parameters `r` `[k, 3]` (module docs).
pub fn flood_pool_transform<B: Backend>(r: Tensor<B, 2>) -> PoolParams<B> {
    let [k, _] = r.dims();
    let col = |j: usize| r.clone().slice([0..k, j..j + 1]).reshape([k]);
    let [lo, hi] = POOL_KC_RANGE;
    let kc = (col(0) + POOL_KC_INIT.ln()).exp().clamp(lo, hi);
    let phi = sigmoid(col(1));
    let q = POOL_Z_INIT_DAYS / POOL_Z_MAX_DAYS;
    let z_days = sigmoid(col(2) * POOL_Z_RATE + (q / (1.0 - q)).ln()) * POOL_Z_MAX_DAYS;
    PoolParams { kc, phi, z_days }
}

impl<B: Backend> DamParams<B> {
    /// Zero-initialised parameters for `n_dams` table dams (no flood pool).
    pub fn zeros(n_dams: usize, rule_curve: bool, per_dam_t0: bool, device: &B::Device) -> Self {
        Self::zeros_with_pool(n_dams, rule_curve, per_dam_t0, false, device)
    }

    /// [`Self::zeros`] with the flood pool's raw parameters when `flood_pool`.
    pub fn zeros_with_pool(
        n_dams: usize,
        rule_curve: bool,
        per_dam_t0: bool,
        flood_pool: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            theta: rule_curve.then(|| Param::from_tensor(Tensor::zeros([n_dams, 4], device))),
            delta: per_dam_t0.then(|| Param::from_tensor(Tensor::zeros([n_dams], device))),
            pool: flood_pool.then(|| Param::from_tensor(Tensor::zeros([n_dams, 3], device))),
        }
    }

    /// Number of table dams the parameters cover (0 when every option is off).
    pub fn n_dams(&self) -> usize {
        match (&self.theta, &self.delta, &self.pool) {
            (Some(t), _, _) => t.val().dims()[0],
            (None, Some(d), _) => d.val().dims()[0],
            (None, None, Some(p)) => p.val().dims()[0],
            (None, None, None) => 0,
        }
    }

    /// The device the parameters live on; `None` when every option is off.
    pub fn device(&self) -> Option<B::Device> {
        self.theta
            .as_ref()
            .map(|t| t.val().device())
            .or_else(|| self.delta.as_ref().map(|d| d.val().device()))
            .or_else(|| self.pool.as_ref().map(|p| p.val().device()))
    }

    /// Rule-curve coefficients `rule_curve_max · tanh(θ)` of the table rows
    /// `idx`, `[idx.len(), 4]`; `None` without a rule curve.
    pub fn coefficients(&self, idx: Tensor<B, 1, Int>, rule_curve_max: f32) -> Option<Tensor<B, 2>> {
        self.theta.as_ref().map(|t| t.val().select(0, idx).tanh() * rule_curve_max)
    }

    /// `exp(δ)` of the table rows `idx`; `None` without per-dam `T0`.
    pub fn t0_factor(&self, idx: Tensor<B, 1, Int>) -> Option<Tensor<B, 1>> {
        self.delta.as_ref().map(|d| d.val().select(0, idx).exp())
    }

    /// The flood pool's `(kc, phi, z)` of the table rows `idx`
    /// ([`flood_pool_transform`]); `None` without a flood pool.
    pub fn pool_params(&self, idx: Tensor<B, 1, Int>) -> Option<PoolParams<B>> {
        self.pool.as_ref().map(|p| flood_pool_transform(p.val().select(0, idx)))
    }

    /// `Σ θ² + Σ δ² + Σ r_pool²` over the table rows `idx` (the
    /// `per_dam_l2` penalty's sum), `[1]`; `None` when every option is off.
    pub fn sum_sq(&self, idx: Tensor<B, 1, Int>) -> Option<Tensor<B, 1>> {
        let theta = self.theta.as_ref().map(|t| t.val().select(0, idx.clone()).powi_scalar(2).sum());
        let delta = self.delta.as_ref().map(|d| d.val().select(0, idx.clone()).powi_scalar(2).sum());
        let pool = self.pool.as_ref().map(|p| p.val().select(0, idx).powi_scalar(2).sum());
        [theta, delta, pool].into_iter().flatten().reduce(|a, b| a + b)
    }
}

/// Table rows as an index tensor.
pub fn table_index<B: Backend>(rows: &[usize], device: &B::Device) -> Tensor<B, 1, Int> {
    let v: Vec<i64> = rows.iter().map(|&r| r as i64).collect();
    Tensor::from_data(burn::tensor::TensorData::new(v, [rows.len()]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn zeros_are_no_rule_curve_and_unit_t0_factor() {
        let d = Default::default();
        let p = DamParams::<B>::zeros(5, true, true, &d);
        assert_eq!(p.n_dams(), 5);
        assert!(p.pool.is_none());
        let idx = table_index::<B>(&[4, 1], &d);
        let c: Vec<f32> = p.coefficients(idx.clone(), 0.7).unwrap().into_data().to_vec().unwrap();
        assert!(c.iter().all(|&v| v == 0.0));
        let f: Vec<f32> = p.t0_factor(idx.clone()).unwrap().into_data().to_vec().unwrap();
        assert_eq!(f, vec![1.0, 1.0]);
        assert_eq!(p.sum_sq(idx).unwrap().into_scalar(), 0.0);
    }

    #[test]
    fn options_off_give_none() {
        let d = Default::default();
        let p = DamParams::<B>::zeros(3, false, false, &d);
        let idx = table_index::<B>(&[0], &d);
        assert_eq!(p.n_dams(), 0);
        assert!(p.coefficients(idx.clone(), 1.0).is_none());
        assert!(p.t0_factor(idx.clone()).is_none());
        assert!(p.pool_params(idx.clone()).is_none());
        assert!(p.sum_sq(idx).is_none());
        assert!(p.device().is_none());
    }

    #[test]
    fn coefficients_are_bounded_by_the_max() {
        let d = Default::default();
        let mut p = DamParams::<B>::zeros(2, true, false, &d);
        p.theta = Some(Param::from_tensor(Tensor::from_floats([[40.0, -40.0, 0.5, 0.0], [1.0, 2.0, 3.0, -1.0]], &d)));
        let c: Vec<f32> =
            p.coefficients(table_index::<B>(&[0, 1], &d), 0.8).unwrap().into_data().to_vec().unwrap();
        assert!((c[0] - 0.8).abs() < 1e-6 && (c[1] + 0.8).abs() < 1e-6);
        assert!((c[2] - 0.8 * 0.5_f32.tanh()).abs() < 1e-6 && c[3] == 0.0);
        assert!(c.iter().all(|v| v.abs() <= 0.8));
    }

    fn host(t: Tensor<B, 1>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }

    #[test]
    fn pool_zeros_are_the_init_pool() {
        let d = Default::default();
        let p = DamParams::<B>::zeros_with_pool(4, false, false, true, &d);
        assert_eq!(p.n_dams(), 4);
        assert!(p.device().is_some());
        let idx = table_index::<B>(&[3, 0], &d);
        let pp = p.pool_params(idx.clone()).unwrap();
        let (kc, phi, z) = (host(pp.kc), host(pp.phi), host(pp.z_days));
        for i in 0..2 {
            assert!((kc[i] - POOL_KC_INIT).abs() < 1e-6, "kc {}", kc[i]);
            assert_eq!(phi[i], 0.5);
            assert!((z[i] - POOL_Z_INIT_DAYS).abs() < 1e-6, "z {}", z[i]);
        }
        assert_eq!(p.sum_sq(idx).unwrap().into_scalar(), 0.0);
    }

    #[test]
    fn pool_transform_is_bounded_and_log_linear_in_z() {
        let d = Default::default();
        let r = Tensor::<B, 2>::from_floats(
            [[-50.0, -50.0, -50.0], [50.0, 50.0, 50.0], [1.0, 0.0, 0.5], [0.0, 2.0, 1.5]],
            &d,
        );
        let pp = flood_pool_transform(r);
        let (kc, phi, z) = (host(pp.kc), host(pp.phi), host(pp.z_days));
        assert_eq!((kc[0], kc[1]), (POOL_KC_RANGE[0], POOL_KC_RANGE[1]), "kc clamped to its box");
        assert!((kc[2] - 3.0 * 1.0_f32.exp()).abs() < 1e-5);
        assert!(phi[0] >= 0.0 && phi[0] < 1e-6 && phi[1] <= 1.0 && phi[1] > 1.0 - 1e-6);
        assert!(z[0] >= 0.0 && z[0] < 1e-6, "z -> 0 far below");
        assert!(z[1] <= POOL_Z_MAX_DAYS && z[1] > POOL_Z_MAX_DAYS * (1.0 - 1e-6), "z -> 120 d far above");
        // Log-linear below the bound: z(r) ~= 0.05·exp(4r)·(1 − z/120)/(1 − 0.05/120).
        let expect = |r: f32| {
            let q = POOL_Z_INIT_DAYS / POOL_Z_MAX_DAYS;
            let a = (q / (1.0 - q)).ln() + POOL_Z_RATE * r;
            POOL_Z_MAX_DAYS / (1.0 + (-a).exp())
        };
        assert!((z[2] - expect(0.5)).abs() < 1e-4 * expect(0.5));
        assert!((z[3] - expect(1.5)).abs() < 1e-4 * expect(1.5));
        assert!(z[3] > 15.0 && z[3] < 20.0, "1.5 raw units reach ~17 d, got {}", z[3]);
    }
}
