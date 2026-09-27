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
//! - `release_head.rule_curve_penalty` (the rule curve's feasibility
//!   penalty):
//!
//!   ```text
//!   P = λ_P · Σ_{(d,w)} Σ_t relu(r_d(t) − α·Qin_d(t))² / Σ_{(d,w)} Σ_t Qin_d(t)²
//!   r_d(t) = (S0_{t+1} − S0_t)/dt = (Ibar_d/dt)·(c_d · ΔH_t)    (positive = storing)
//!   ```
//!
//!   over the step's distinct (dam, window) pairs `(d, w)` (a dam below
//!   gauges in two micro-batches of the same window counts once), with
//!   `Qin_d(t)` the dam's inflow at step `t` from the model's own forward
//!   (routed upstream inflow plus the reach's own `q'`, before the flux;
//!   `MuskingumCunge::dam_inflow_record`), DETACHED. `P` is differentiable in
//!   the rule-curve coefficients `c = rule_curve_max·tanh(θ)` only, through
//!   `r`. It supplies the restoring gradient the S28 clamp removes when the
//!   flux stores more than the dam receives (review v2, finding 1): on a
//!   clamped step the clamp zeroes the dam row's gradient and creates water.
//!   `r` is recomputed here from `θ` with the window's phase increments `ΔH`
//!   (the engine's own table), so it is the flux the forward applied, to f32
//!   rounding.
//!
//! Every sum runs over ordered maps, so the terms are a function of the SET
//! of the step's (dam, window) pairs: the same pairs give bitwise the same
//! value and gradient however they were split into micro-batches.

use std::collections::{BTreeMap, BTreeSet};

use burn::prelude::ElementConversion;
use burn::tensor::{backend::AutodiffBackend, Tensor, TensorData};
use chrono::NaiveDate;

use crate::config::ReleaseHeadSection;
use crate::nn::dam_params::{table_index, DamParams};
use crate::routing::mmc::DT_SECONDS;

/// One micro-batch's dams, as its forward armed them.
#[derive(Clone, Debug, PartialEq)]
pub struct DamBatchRecord {
    /// The micro-batch's window start (it decided which dams are built).
    pub window_start: NaiveDate,
    /// Feature-table row of each armed dam, in the engine's dam order.
    pub table_index: Vec<usize>,
    /// The rule curve's penalty inputs, when the penalty is on.
    pub rule_curve: Option<RuleCurveRecord>,
}

/// One micro-batch's rule-curve penalty inputs, dams in the engine's order
/// (aligned with [`DamBatchRecord::table_index`]).
#[derive(Clone, Debug, PartialEq)]
pub struct RuleCurveRecord {
    /// `Ibar` per dam, m³/s.
    pub inflow_mean: Vec<f32>,
    /// The window's per-step phase increments `ΔH_t` (seconds),
    /// `MuskingumCunge::rule_curve_increments`.
    pub dh: Vec<[f32; 4]>,
    /// `Qin` per step and dam, m³/s, row-major `[dh.len(), n_dams]`
    /// (`MuskingumCunge::dam_inflow_record`), detached.
    pub qin: Vec<f32>,
}

/// One window's penalty inputs in the step: its `ΔH` and, per dam
/// (feature-table row), `Ibar` and the `Qin` column.
#[derive(Clone, Debug, PartialEq)]
struct WindowTerms {
    dh: Vec<[f32; 4]>,
    dams: BTreeMap<usize, (f32, Vec<f32>)>,
}

/// The step's per-dam terms, accumulated over its micro-batches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamStepTerms {
    /// Union of the step's dams (feature-table rows).
    rows: BTreeSet<usize>,
    /// The penalty's (dam, window) pairs, by window start.
    windows: BTreeMap<NaiveDate, WindowTerms>,
}

/// The step terms evaluated on the parameters: the tracked total to
/// backpropagate and each term's value, for the log.
pub struct DamTermsLoss<B: AutodiffBackend> {
    pub total: Tensor<B, 1>,
    /// `per_dam_l2·Σ(θ² + δ²)`, when on.
    pub l2: Option<f32>,
    /// The feasibility penalty, when on.
    pub penalty: Option<PenaltyValue>,
}

/// The feasibility penalty's value and how much of the step it binds on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PenaltyValue {
    /// `P`.
    pub value: f32,
    /// (dam, step) pairs with `r > α·Qin` (the hinge active).
    pub active: usize,
    /// (dam, step) pairs in the step.
    pub total: usize,
}

impl DamStepTerms {
    /// Add one micro-batch's dams. A (dam, window) pair already in the step
    /// keeps its first record.
    pub fn add(&mut self, rec: &DamBatchRecord) {
        self.rows.extend(rec.table_index.iter().copied());
        if let Some(rc) = rec.rule_curve.as_ref() {
            let n = rec.table_index.len();
            assert_eq!(rc.inflow_mean.len(), n, "one Ibar per dam");
            assert_eq!(rc.qin.len(), rc.dh.len() * n, "Qin is [n_steps, n_dams]");
            let w = self
                .windows
                .entry(rec.window_start)
                .or_insert_with(|| WindowTerms { dh: rc.dh.clone(), dams: BTreeMap::new() });
            assert_eq!(w.dh.len(), rc.dh.len(), "micro-batches of one window route the same steps");
            for (j, &row) in rec.table_index.iter().enumerate() {
                w.dams.entry(row).or_insert_with(|| {
                    (rc.inflow_mean[j], (0..rc.dh.len()).map(|t| rc.qin[t * n + j]).collect())
                });
            }
        }
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
        let device = params
            .theta
            .as_ref()
            .map(|t| t.val().device())
            .or_else(|| params.delta.as_ref().map(|d| d.val().device()))?;
        let l2 = (rh.per_dam_l2 > 0.0 && !self.rows.is_empty())
            .then(|| params.sum_sq(table_index::<B>(&self.rows(), &device)))
            .flatten()
            .map(|s| s * rh.per_dam_l2);
        let penalty = (rh.rule_curve_penalty > 0.0).then(|| self.penalty(params, rh, &device)).flatten();
        let l2_value = l2.as_ref().map(|t| t.clone().into_scalar().elem::<f32>());
        let (total, penalty_value) = match (l2, penalty) {
            (None, None) => return None,
            (Some(a), None) => (a, None),
            (None, Some((p, v))) => (p, Some(v)),
            (Some(a), Some((p, v))) => (a + p, Some(v)),
        };
        Some(DamTermsLoss { total, l2: l2_value, penalty: penalty_value })
    }

    /// The feasibility penalty (module docs), `None` without (dam, window)
    /// pairs or with no inflow at all.
    fn penalty<B: AutodiffBackend>(
        &self,
        params: &DamParams<B>,
        rh: &ReleaseHeadSection,
        device: &B::Device,
    ) -> Option<(Tensor<B, 1>, PenaltyValue)> {
        let alpha = rh.rule_curve_alpha;
        let mut num: Option<Tensor<B, 1>> = None;
        let mut den = 0.0_f64;
        let (mut active, mut total) = (0usize, 0usize);
        for w in self.windows.values() {
            let rows: Vec<usize> = w.dams.keys().copied().collect();
            let (n_steps, n) = (w.dh.len(), rows.len());
            if n == 0 || n_steps == 0 {
                continue;
            }
            let c = params.coefficients(table_index::<B>(&rows, device), rh.rule_curve_max)?; // [n, 4]
            let dh_flat: Vec<f32> = w.dh.iter().flatten().copied().collect();
            let dh = Tensor::<B, 1>::from_floats(dh_flat.as_slice(), device).reshape([n_steps, 4]);
            let scale: Vec<f32> = w.dams.values().map(|(ibar, _)| ibar / DT_SECONDS).collect();
            let scale = Tensor::<B, 1>::from_floats(scale.as_slice(), device).reshape([1, n]);
            let r = dh.matmul(c.transpose()) * scale; // [n_steps, n], m³/s
            let mut qin = vec![0.0_f32; n_steps * n];
            for (j, (_, col)) in w.dams.values().enumerate() {
                for (t, &q) in col.iter().enumerate() {
                    qin[t * n + j] = q;
                    den += (q as f64) * (q as f64);
                }
            }
            let qin = Tensor::<B, 2>::from_data(TensorData::new(qin, [n_steps, n]), device);
            let excess = (r - qin * alpha).clamp_min(0.0);
            active += excess
                .clone()
                .inner()
                .greater_elem(0.0)
                .int()
                .sum()
                .into_scalar()
                .elem::<i64>() as usize;
            total += n_steps * n;
            let sq = excess.powi_scalar(2).sum();
            num = Some(match num {
                Some(acc) => acc + sq,
                None => sq,
            });
        }
        let num = num?;
        if den <= 0.0 {
            return None;
        }
        let p = num * (rh.rule_curve_penalty as f64 / den) as f32;
        let value: f32 = p.clone().into_scalar().elem();
        Some((p, PenaltyValue { value, active, total }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::module::Param;
    use chrono::Datelike;

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
        let rec = |d: u32, rows: Vec<usize>| DamBatchRecord { window_start: day(d), table_index: rows, rule_curve: None };
        let whole = rec(1, vec![4, 0, 2]);
        let a = rec(1, vec![0, 2]);
        let b = rec(9, vec![2, 4]);
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
        t.add(&DamBatchRecord { window_start: day(1), table_index: vec![1], rule_curve: None });
        assert!(t.loss(&p, &section(0.0)).is_none(), "l2 off");
        // The penalty on, but no rule-curve record: nothing to penalise.
        let mut rh = section(0.0);
        rh.rule_curve_penalty = 1.0;
        assert!(t.loss(&p, &rh).is_none(), "penalty without records");
    }

    // -------------------------------------------------------------------
    // The rule curve's feasibility penalty.
    // -------------------------------------------------------------------

    const N_STEPS: usize = 96;

    fn penalty_section(lambda: f32, alpha: f32) -> ReleaseHeadSection {
        let mut rh = section(0.0);
        rh.rule_curve_penalty = lambda;
        rh.rule_curve_alpha = alpha;
        rh.rule_curve_max = 0.8;
        rh
    }

    /// `Ibar` of table dam `d`.
    fn ibar(d: usize) -> f32 {
        10.0 + 5.0 * d as f32
    }

    /// `Qin` of table dam `d` at step `t` of a window starting on `start`:
    /// a smooth daily cycle around a quarter of `Ibar`, so the rule curve's
    /// flux sometimes exceeds `α·Qin` and sometimes not.
    fn qin(d: usize, start: NaiveDate, t: usize) -> f32 {
        let phase = (start.ordinal() as f32) * 0.1 + d as f32;
        ibar(d) * (0.25 + 0.2 * ((t as f32) * 2.0 * std::f32::consts::PI / 24.0 + phase).sin())
    }

    /// A micro-batch record for dams `rows` (table rows) of the window
    /// starting on `start`, from the engine's own phase increments.
    fn rc_record(start: NaiveDate, rows: &[usize]) -> DamBatchRecord {
        let w0 = crate::routing::release::rule_curve_phase_start(start);
        let dh: Vec<[f32; 4]> = crate::routing::release::rule_curve_increments(w0, N_STEPS + 1)
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let n = rows.len();
        let mut q = vec![0.0_f32; N_STEPS * n];
        for t in 0..N_STEPS {
            for (j, &d) in rows.iter().enumerate() {
                q[t * n + j] = qin(d, start, t);
            }
        }
        DamBatchRecord {
            window_start: start,
            table_index: rows.to_vec(),
            rule_curve: Some(RuleCurveRecord { inflow_mean: rows.iter().map(|&d| ibar(d)).collect(), dh, qin: q }),
        }
    }

    /// October: dam 0's flux (`c` of [`params`]) stores ~0.8·Ibar, dam 1's
    /// ~0.2·Ibar, the others release, so the hinge binds on part of the steps.
    fn oct(d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(1990, 10, d).unwrap()
    }

    /// `P` in f64 from the same inputs, `c = 0.8·tanh(θ)` of [`params`].
    fn hand_penalty(recs: &[(NaiveDate, Vec<usize>)], lambda: f64, alpha: f64) -> (f64, usize) {
        let th = |i: usize| 0.1 * (i as f64 - 7.0);
        let (mut num, mut den, mut active) = (0.0, 0.0, 0);
        for (start, rows) in recs {
            let rec = rc_record(*start, rows);
            let dh = &rec.rule_curve.as_ref().unwrap().dh;
            for &d in rows {
                let c: Vec<f64> = (0..4).map(|j| 0.8 * th(4 * d + j).tanh()).collect();
                for (t, h) in dh.iter().enumerate() {
                    let r = ibar(d) as f64 / DT_SECONDS as f64 * (0..4).map(|j| c[j] * h[j] as f64).sum::<f64>();
                    let q = qin(d, *start, t) as f64;
                    let e = (r - alpha * q).max(0.0);
                    active += (e > 0.0) as usize;
                    num += e * e;
                    den += q * q;
                }
            }
        }
        (lambda * num / den, active)
    }

    #[test]
    fn penalty_matches_a_hand_computation_and_finite_differences() {
        let rh = penalty_section(2.0, 0.9);
        let mut terms = DamStepTerms::default();
        terms.add(&rc_record(oct(10), &[0, 3]));
        terms.add(&rc_record(oct(25), &[1]));
        let p = params();
        let out = terms.loss(&p, &rh).expect("penalty on");
        let pv = out.penalty.expect("penalty value");
        assert!(out.l2.is_none());
        let (hand, hand_active) = hand_penalty(&[(oct(10), vec![0, 3]), (oct(25), vec![1])], 2.0, 0.9);
        println!(
            "penalty {:.6e} (hand {hand:.6e}), hinge active on {}/{} dam-steps (hand {hand_active})",
            pv.value, pv.active, pv.total
        );
        assert!(pv.active > 0 && pv.active < pv.total, "the case must bind on part of the steps");
        assert_eq!(pv.total, 3 * N_STEPS);
        assert_eq!(pv.active, hand_active);
        assert!((pv.value as f64 - hand).abs() <= 1e-4 * hand, "{} vs {hand}", pv.value);

        // Gradient: every theta entry of the three dams against central
        // differences of the hand value (smooth: relu squared); none elsewhere.
        let g = burn::optim::GradientsParams::from_grads(out.total.backward(), &p);
        let gt: Vec<f32> =
            g.get::<NdArray<f32>, 2>(p.theta.as_ref().unwrap().id).unwrap().into_data().to_vec().unwrap();
        let hand_at = |k: usize, eps: f64| -> f64 {
            // Same hand computation with theta entry k shifted.
            let th = |i: usize| 0.1 * (i as f64 - 7.0) + if i == k { eps } else { 0.0 };
            let (mut num, mut den) = (0.0, 0.0);
            for (start, rows) in [(oct(10), vec![0usize, 3]), (oct(25), vec![1])] {
                let rec = rc_record(start, &rows);
                for &d in &rows {
                    let c: Vec<f64> = (0..4).map(|j| 0.8 * th(4 * d + j).tanh()).collect();
                    for (t, h) in rec.rule_curve.as_ref().unwrap().dh.iter().enumerate() {
                        let r = ibar(d) as f64 / DT_SECONDS as f64
                            * (0..4).map(|j| c[j] * h[j] as f64).sum::<f64>();
                        let q = qin(d, start, t) as f64;
                        let e = (r - 0.9 * q).max(0.0);
                        num += e * e;
                        den += q * q;
                    }
                }
            }
            2.0 * num / den
        };
        for k in 0..20 {
            let fd = (hand_at(k, 1e-4) - hand_at(k, -1e-4)) / 2e-4;
            let a = gt[k] as f64;
            if [0usize, 1, 3].contains(&(k / 4)) {
                let rel = (a - fd).abs() / a.abs().max(fd.abs()).max(1e-12);
                println!("theta[{k}]: analytical {a:.6e} fd {fd:.6e} rel {rel:.2e}");
                assert!(rel < 5e-3 || (a - fd).abs() < 1e-7, "theta[{k}]: {a} vs {fd}");
            } else {
                assert_eq!(a, 0.0, "theta[{k}] of a dam outside the step");
            }
        }
        // delta does not enter the flux.
        assert!(g.get::<NdArray<f32>, 1>(p.delta.as_ref().unwrap().id).is_none());
    }

    #[test]
    fn penalty_and_l2_do_not_depend_on_the_micro_batch_split() {
        let mut rh = penalty_section(1.0, 0.9);
        rh.per_dam_l2 = 1e-3;
        // One micro-batch with dams {0, 2, 4} on one window, plus one on a
        // second window; or the first split into {0, 2} and {2, 4} (dam 2 below
        // gauges in both: the same (dam, window) pair, counted once), in any order.
        let whole = [rc_record(oct(10), &[4, 0, 2]), rc_record(oct(25), &[1, 2])];
        let split = [rc_record(oct(10), &[0, 2]), rc_record(oct(25), &[1, 2]), rc_record(oct(10), &[2, 4])];
        let mut orders: Vec<Vec<&DamBatchRecord>> = vec![whole.iter().collect(), split.iter().collect()];
        orders.push(split.iter().rev().collect());
        let mut results = Vec::new();
        for recs in &orders {
            let mut t = DamStepTerms::default();
            for r in recs {
                t.add(r);
            }
            let p = params();
            let out = t.loss(&p, &rh).unwrap();
            let v: f32 = out.total.clone().into_scalar();
            let pv = out.penalty.unwrap();
            let g = burn::optim::GradientsParams::from_grads(out.total.backward(), &p);
            let gt: Vec<u32> = g
                .get::<NdArray<f32>, 2>(p.theta.as_ref().unwrap().id)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            results.push((v.to_bits(), pv.value.to_bits(), pv.active, pv.total, gt));
        }
        println!("split-independence: P = {}", f32::from_bits(results[0].1));
        assert!(f32::from_bits(results[0].1) > 0.0, "the case must bind");
        assert_eq!(results[0].3, 5 * N_STEPS, "5 distinct (dam, window) pairs");
        for (i, r) in results.iter().enumerate().skip(1) {
            assert_eq!(r, &results[0], "grouping {i}: penalty / L2 value or gradient differs");
        }
    }

    #[test]
    fn zero_rule_curve_has_zero_penalty_and_gradient() {
        let rh = penalty_section(1.0, 0.9);
        let mut t = DamStepTerms::default();
        t.add(&rc_record(oct(10), &[0, 1]));
        let d = Default::default();
        let p = DamParams::<AB>::zeros(5, true, false, &d);
        let out = t.loss(&p, &rh).unwrap();
        assert_eq!(out.penalty.unwrap().value, 0.0);
        assert_eq!(out.penalty.unwrap().active, 0);
        let g = burn::optim::GradientsParams::from_grads(out.total.backward(), &p);
        let gt: Vec<f32> =
            g.get::<NdArray<f32>, 2>(p.theta.as_ref().unwrap().id).unwrap().into_data().to_vec().unwrap();
        assert!(gt.iter().all(|&x| x == 0.0));
    }
}
