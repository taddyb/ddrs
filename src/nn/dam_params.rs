//! Per-dam free parameters of the learned dam release
//! (`release_head.rule_curve`, `release_head.per_dam_t0`).
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
//!
//! Both start at zero. They have their own optimizer, a row-sparse Adam
//! (`crate::training::lazy_adam`: only rows with a nonzero gradient in a step
//! move, so a dam never in a batch stays exactly at zero), and a constant
//! learning rate (`release_head.per_dam_lr`): a dam's parameters get a
//! gradient only when a gauge below it is in the batch, so the heads'
//! schedule would barely move them. Saved as `release_dams.mpk` (+
//! `release_dams_optim.json`) next to `release_head.mpk`, so the release
//! head's own record and every older checkpoint are unchanged.

use burn::module::{Module, Param};
use burn::tensor::{backend::Backend, Int, Tensor};

/// See the module docs. A field is `None` when its option is off.
#[derive(Module, Debug)]
pub struct DamParams<B: Backend> {
    /// `[n_dams, 4]` rule-curve logits, `(c1s, c1c, c2s, c2c)`.
    pub theta: Option<Param<Tensor<B, 2>>>,
    /// `[n_dams]` log multiplier on `T0`.
    pub delta: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> DamParams<B> {
    /// Zero-initialised parameters for `n_dams` table dams.
    pub fn zeros(n_dams: usize, rule_curve: bool, per_dam_t0: bool, device: &B::Device) -> Self {
        Self {
            theta: rule_curve.then(|| Param::from_tensor(Tensor::zeros([n_dams, 4], device))),
            delta: per_dam_t0.then(|| Param::from_tensor(Tensor::zeros([n_dams], device))),
        }
    }

    /// Number of table dams the parameters cover (0 when both are off).
    pub fn n_dams(&self) -> usize {
        match (&self.theta, &self.delta) {
            (Some(t), _) => t.val().dims()[0],
            (None, Some(d)) => d.val().dims()[0],
            (None, None) => 0,
        }
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

    /// `Σ θ² + Σ δ²` over the table rows `idx` (the `per_dam_l2` penalty's
    /// sum), `[1]`; `None` when both options are off.
    pub fn sum_sq(&self, idx: Tensor<B, 1, Int>) -> Option<Tensor<B, 1>> {
        let theta = self.theta.as_ref().map(|t| t.val().select(0, idx.clone()).powi_scalar(2).sum());
        let delta = self.delta.as_ref().map(|d| d.val().select(0, idx).powi_scalar(2).sum());
        match (theta, delta) {
            (Some(a), Some(b)) => Some(a + b),
            (a, b) => a.or(b),
        }
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
        assert!(p.sum_sq(idx).is_none());
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
}
