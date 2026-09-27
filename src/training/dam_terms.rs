//! Per-dam terms of the learned release's objective that are added ONCE per
//! optimizer step, over the step's dams, not once per micro-batch.
//!
//! The data loss is a pooled mean over the step's micro-batches (each
//! micro-batch loss times its valid count, the sum divided by the step's
//! total; `training::driver`). A per-dam term added inside each micro-batch
//! would instead count a dam once per micro-batch it appears in and be
//! rescaled by that micro-batch's share of the valid count, so its strength
//! would depend on how the batch was split (review v2, finding 4). Here every
//! micro-batch only RECORDS its dams ([`DamBatchRecord`]); the driver folds
//! the records into a [`DamStepTerms`] and, after the last micro-batch,
//! evaluates the terms once on the per-dam parameters, backpropagates them
//! on their own, and adds their gradient to the step's (already pooled) data
//! gradient.
//!
//! - `release_head.per_dam_l2`: `λ·Σ_{d ∈ D}(θ_d² + δ_d²)`, `D` the union of
//!   the step's active dams (feature-table rows), each dam once.

use std::collections::BTreeSet;

use burn::prelude::ElementConversion;
use burn::tensor::{backend::AutodiffBackend, Tensor};
use chrono::NaiveDate;

use crate::config::ReleaseHeadSection;
use crate::nn::dam_params::{table_index, DamParams};

/// One micro-batch's dams, as its forward armed them.
#[derive(Clone, Debug, PartialEq)]
pub struct DamBatchRecord {
    /// The micro-batch's window start (it decided which dams are built).
    pub window_start: NaiveDate,
    /// Feature-table row of each armed dam, in the engine's dam order.
    pub table_index: Vec<usize>,
}

/// The step's per-dam terms, accumulated over its micro-batches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamStepTerms {
    /// Union of the step's dams (feature-table rows).
    rows: BTreeSet<usize>,
}

/// The step terms evaluated on the parameters: the tracked total to
/// backpropagate and each term's value, for the log.
pub struct DamTermsLoss<B: AutodiffBackend> {
    pub total: Tensor<B, 1>,
    pub l2: f32,
}

impl DamStepTerms {
    /// Add one micro-batch's dams.
    pub fn add(&mut self, rec: &DamBatchRecord) {
        self.rows.extend(rec.table_index.iter().copied());
    }

    /// The union of the step's dams, ascending.
    pub fn rows(&self) -> Vec<usize> {
        self.rows.iter().copied().collect()
    }

    /// The step's terms on `params`, or `None` when every term is off or the
    /// step saw no dam.
    pub fn loss<B: AutodiffBackend>(
        &self,
        params: &DamParams<B>,
        rh: &ReleaseHeadSection,
    ) -> Option<DamTermsLoss<B>> {
        if rh.per_dam_l2 <= 0.0 || self.rows.is_empty() {
            return None;
        }
        let device = params
            .theta
            .as_ref()
            .map(|t| t.val().device())
            .or_else(|| params.delta.as_ref().map(|d| d.val().device()))?;
        let idx = table_index::<B>(&self.rows(), &device);
        let l2 = params.sum_sq(idx)? * rh.per_dam_l2;
        let l2_value: f32 = l2.clone().into_scalar().elem();
        Some(DamTermsLoss { total: l2, l2: l2_value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::module::Param;

    type AB = Autodiff<NdArray<f32>>;

    pub(super) fn section(l2: f32) -> ReleaseHeadSection {
        let mut rh: ReleaseHeadSection = serde_yaml::from_str("input_var_names: [f1]\nrule_curve: true\nper_dam_t0: true\n").unwrap();
        rh.per_dam_l2 = l2;
        rh
    }

    pub(super) fn params() -> DamParams<AB> {
        let d = Default::default();
        let mut p = DamParams::<AB>::zeros(5, true, true, &d);
        let th: Vec<f32> = (0..20).map(|i| 0.1 * (i as f32 - 7.0)).collect();
        p.theta = Some(Param::from_tensor(Tensor::<AB, 1>::from_floats(th.as_slice(), &d).reshape([5, 4])));
        p.delta = Some(Param::from_tensor(Tensor::from_floats([0.3, -0.2, 0.0, 0.5, -0.4], &d)));
        p
    }

    fn day(d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(1990, 5, d).unwrap()
    }

    /// Value and (θ, δ) gradient of the step terms.
    pub(super) fn eval(terms: &DamStepTerms, rh: &ReleaseHeadSection) -> (f32, Vec<f32>, Vec<f32>) {
        let p = params();
        let out = terms.loss(&p, rh).expect("terms on");
        let v: f32 = out.total.clone().into_scalar();
        let g = burn::optim::GradientsParams::from_grads(out.total.backward(), &p);
        let gt: Vec<f32> =
            g.get::<NdArray<f32>, 2>(p.theta.as_ref().unwrap().id).unwrap().into_data().to_vec().unwrap();
        let gd: Vec<f32> =
            g.get::<NdArray<f32>, 1>(p.delta.as_ref().unwrap().id).unwrap().into_data().to_vec().unwrap();
        (v, gt, gd)
    }

    #[test]
    fn l2_is_over_the_union_of_the_steps_dams_whatever_the_split() {
        let rh = section(0.01);
        // One batch with dams {0, 2, 4}, or split into micro-batches {0, 2}
        // and {2, 4} (dam 2 is below gauges in both), in either order.
        let whole = DamBatchRecord { window_start: day(1), table_index: vec![4, 0, 2] };
        let a = DamBatchRecord { window_start: day(1), table_index: vec![0, 2] };
        let b = DamBatchRecord { window_start: day(9), table_index: vec![2, 4] };
        let mut one = DamStepTerms::default();
        one.add(&whole);
        let mut split = DamStepTerms::default();
        split.add(&a);
        split.add(&b);
        let mut rev = DamStepTerms::default();
        rev.add(&b);
        rev.add(&a);
        let (v1, gt1, gd1) = eval(&one, &rh);
        for (label, t) in [("split", &split), ("reversed", &rev)] {
            let (v, gt, gd) = eval(t, &rh);
            assert_eq!(v.to_bits(), v1.to_bits(), "{label}: value");
            assert_eq!(gt, gt1, "{label}: theta gradient");
            assert_eq!(gd, gd1, "{label}: delta gradient");
        }
        // Hand value: 0.01 · Σ_{d ∈ {0,2,4}} (Σ θ_d² + δ_d²).
        let th = |i: usize| 0.1 * (i as f64 - 7.0);
        let delta = [0.3, -0.2, 0.0, 0.5, -0.4];
        let hand: f64 = [0usize, 2, 4]
            .iter()
            .map(|&d| (0..4).map(|j| th(4 * d + j).powi(2)).sum::<f64>() + f64::powi(delta[d], 2))
            .sum::<f64>()
            * 0.01;
        assert!((v1 as f64 - hand).abs() < 1e-6, "{v1} vs {hand}");
        // Dams outside the step get no gradient; a dam's gradient is 2λθ.
        assert!(gt1[4..8].iter().chain(&gt1[12..16]).all(|&g| g == 0.0));
        assert!((gt1[0] - 2.0 * 0.01 * th(0) as f32).abs() < 1e-7);
        assert_eq!((gd1[1], gd1[3]), (0.0, 0.0));
    }

    #[test]
    fn terms_off_or_no_dams_give_none() {
        let p = params();
        let mut t = DamStepTerms::default();
        assert!(t.loss(&p, &section(0.01)).is_none(), "no dams");
        t.add(&DamBatchRecord { window_start: day(1), table_index: vec![1] });
        assert!(t.loss(&p, &section(0.0)).is_none(), "l2 off");
    }
}
