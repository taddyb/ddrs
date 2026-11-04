//! Does each active clamp have its physical reason?
//!
//! A clamp is active when the raw value is strictly outside the bound (see
//! [`crate::routing::clamp_ledger`]). For three sites the cause has a closed
//! form, so the ledger can be checked against the routed values. This is a
//! diagnostic. It reads routed values and feeds nothing back, so it uses f64.
//!
//! Notation: `qe = q + 1e-6`, `e = 3 / (5 + 3 qe + 3 gamma)`, `d_ref = 1`,
//! `A(d) = p d^(qe+1) / (qe+1)` (the power-law section, see
//! `src/geometry.rs::power_law_from_depth`).
//!
//! Rules, each with slack `1e-5` (relative):
//!
//! - `DepthFloor` active means `Q_t < Q_floor`. The depth inversion gives
//!   `d_raw < d_lb` exactly when `Q_t < Q_floor`, with
//!   power law `Q_floor = p sqrt(s) d_lb^(1/e) / (n (qe+1)^(5/3))` and
//!   trapezoid `Q_floor = p sqrt(s) d_lb^(1/e) / (n (qe+1))`. The trapezoid
//!   adds `1e-8` to its denominator `p sqrt(s)`. That change is far inside the
//!   slack.
//! - `VelocityLow` active (power law only) means `Q_t < v_lb A(max(d_v, d_lb))`.
//!   `d_v` is the depth where the Manning velocity equals `v_lb`:
//!   `d_v = [v_lb n (qe+1)^(2/3) / sqrt(s)]^(1/(gamma + 2/3))`. The velocity
//!   `Q/A(d)` rises with `Q`. Where `d_v < d_lb` the depth is floored, the
//!   velocity is `Q / A(d_lb)`, and the site is active up to `v_lb A(d_lb)`.
//!   On the trapezoid this site is counted, not checked.
//! - `DischargeFloor` active with `x_sol[i] < 0` means: `OutsideCrWindow` at
//!   `i`, or `x_sol[j] < 0` for some direct upstream reach `j`. This is an
//!   induction over the topological order, not a direct-upstream window test.
//!   With no leakance and no reservoir rows, `b = c2 i_t + c3 q_t + c4 q'` with
//!   `c2, c4 > 0` and `i_t, q_t, q' >= 0`. The solve gives
//!   `x[i] = b[i] + c1[i] * sum_j x[j]`. With `c1[i] >= 0` and `c3[i] >= 0`,
//!   `x[i] < 0` needs a negative `x[j]` upstream. A reach two levels below a
//!   window violation can go negative while inside the window itself. Products
//!   and sums of non-negative f32 values stay non-negative, so the rule is
//!   exact in f32. The rule does not hold with leakance or reservoir rows. The
//!   caller must not use this function there. `probe_clamp_parity` refuses both.
//!
//! Every other site is counted only.

use std::collections::BTreeMap;

use crate::config::{AttributeMinimums, GeometryForm};
use crate::routing::clamp_ledger::{ClampSite, StepLedger};

/// Relative slack on the closed-form bounds.
const SLACK: f64 = 1e-5;

/// What the reason check found.
///
/// `checked[site]` is the number of active entries a rule was applied to for
/// `DepthFloor`, power-law `VelocityLow`, and `DischargeFloor` with
/// `x_sol < 0`. For every other site it is the number of active entries
/// (count only). `DischargeFloor` entries with `x_sol >= 0` are not counted.
#[derive(Clone, Debug, Default)]
pub struct ReasonReport {
    pub checked: BTreeMap<ClampSite, u64>,
    /// True when the `DischargeFloor` rule was not applied (count only)
    /// because the config has leakance or reservoir rows.
    pub discharge_rule_skipped: bool,
    /// `(site, reach, message)`. The message holds the raw inputs, the bound
    /// and the margin.
    pub violations: Vec<(ClampSite, usize, String)>,
}

impl ReasonReport {
    /// Add another report (another step) to this one.
    pub fn merge(&mut self, other: ReasonReport) {
        for (s, c) in other.checked {
            *self.checked.entry(s).or_insert(0) += c;
        }
        self.violations.extend(other.violations);
        self.discharge_rule_skipped |= other.discharge_rule_skipped;
    }
}

/// The closed forms assume `d_ref = 1`. Return an error naming
/// `params.stage_roughness.d_ref` when `gamma != 0` and `d_ref != 1`.
pub fn check_preconditions(gamma: f32, d_ref: f32) -> Result<(), String> {
    if gamma != 0.0 && d_ref != 1.0 {
        return Err(format!(
            "the clamp-reason closed forms assume d_ref = 1, but params.stage_roughness.d_ref is {d_ref} with gamma {gamma}"
        ));
    }
    Ok(())
}

fn is_active(step: &StepLedger, site: ClampSite, i: usize) -> bool {
    step.active.get(&site).map(|v| v[i]).unwrap_or(false)
}

/// `Q_floor`: the discharge where the raw depth equals `d_lb`.
fn q_floor(form: GeometryForm, gamma: f64, d_lb: f64, n: f64, p: f64, q: f64, s: f64) -> f64 {
    let qe = q + 1e-6;
    let inv_e = (5.0 + 3.0 * qe + 3.0 * gamma) / 3.0;
    let qp1 = qe + 1.0;
    let denom_pow = match form {
        GeometryForm::PowerLaw => qp1.powf(5.0 / 3.0),
        GeometryForm::Trapezoid => qp1,
    };
    p * s.sqrt() * d_lb.powf(inv_e) / (n * denom_pow)
}

/// Power-law section area `A(d) = p d^(qe+1) / (qe+1)`.
fn area(p: f64, q: f64, d: f64) -> f64 {
    let qe = q + 1e-6;
    p * d.powf(qe + 1.0) / (qe + 1.0)
}

/// Check the active clamps of one step against their reasons.
///
/// `gamma` is `cfg.params.stage_roughness_params().0`. `upstream[i]` lists the
/// direct upstream reaches of `i`. `q_t` is the discharge the step started from
/// and `x_sol` is the pre-clamp solve of the same step. Call
/// [`check_preconditions`] first. `discharge_rule_exact` is false when the
/// config has leakance or reservoir rows: `DischargeFloor` is then count only.
#[allow(clippy::too_many_arguments)]
pub fn clamp_reasons(
    form: GeometryForm,
    gamma: f32,
    step: &StepLedger,
    q_t: &[f32],
    n: &[f32],
    p: &[f32],
    q: &[f32],
    slope: &[f32],
    x_sol: &[f32],
    upstream: &[Vec<usize>],
    mins: &AttributeMinimums,
    discharge_rule_exact: bool,
) -> ReasonReport {
    let mut rep = ReasonReport::default();
    rep.discharge_rule_skipped = !discharge_rule_exact;
    let gamma = gamma as f64;
    let d_lb = mins.depth as f64;
    let v_lb = mins.velocity as f64;

    for site in ClampSite::ALL {
        let Some(active) = step.active.get(&site) else { continue };
        for i in 0..step.n {
            if !active[i] {
                continue;
            }
            let (qt, ni, pi, qi, si) =
                (q_t[i] as f64, n[i] as f64, p[i] as f64, q[i] as f64, slope[i] as f64);
            let inputs = format!("Q_t {qt:e} n {ni:e} p {pi:e} q {qi:e} s {si:e}");
            match site {
                ClampSite::DepthFloor => {
                    *rep.checked.entry(site).or_insert(0) += 1;
                    let bound = q_floor(form, gamma, d_lb, ni, pi, qi, si);
                    if !(qt < bound * (1.0 + SLACK)) {
                        rep.violations.push((
                            site,
                            i,
                            format!(
                                "{inputs} d_lb {d_lb:e}; bound Q_floor {bound:e}, margin Q_t - Q_floor {:e}",
                                qt - bound
                            ),
                        ));
                    }
                }
                ClampSite::VelocityLow if form == GeometryForm::PowerLaw => {
                    *rep.checked.entry(site).or_insert(0) += 1;
                    let qe = qi + 1e-6;
                    let d_v = (v_lb * ni * (qe + 1.0).powf(2.0 / 3.0) / si.sqrt())
                        .powf(1.0 / (gamma + 2.0 / 3.0));
                    let bound = v_lb * area(pi, qi, d_v.max(d_lb));
                    if !(qt < bound * (1.0 + SLACK)) {
                        rep.violations.push((
                            site,
                            i,
                            format!(
                                "{inputs} v_lb {v_lb:e} d_v {d_v:e} d_lb {d_lb:e}; bound v_lb A(max(d_v, d_lb)) {bound:e}, margin Q_t - bound {:e}",
                                qt - bound
                            ),
                        ));
                    }
                }
                ClampSite::DischargeFloor if !discharge_rule_exact => {
                    *rep.checked.entry(site).or_insert(0) += 1;
                }
                ClampSite::DischargeFloor if x_sol[i] < 0.0 => {
                    *rep.checked.entry(site).or_insert(0) += 1;
                    let in_window = is_active(step, ClampSite::OutsideCrWindow, i);
                    let neg_up = upstream[i].iter().any(|&j| x_sol[j] < 0.0);
                    if !in_window && !neg_up {
                        rep.violations.push((
                            site,
                            i,
                            format!(
                                "{inputs} x_sol {:e}; no outside_cr_window on the reach and no negative x_sol upstream (upstream {:?})",
                                x_sol[i], upstream[i]
                            ),
                        ));
                    }
                }
                // No rule: `DischargeFloor` with `x_sol >= 0` (the solve sits in
                // `[0, discharge_lb)`), and every other site. The active
                // counts of the parity report already hold these.
                ClampSite::DischargeFloor => {}
                _ => {
                    *rep.checked.entry(site).or_insert(0) += 1;
                }
            }
        }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mins() -> AttributeMinimums {
        AttributeMinimums { discharge: 1e-4, slope: 1e-3, velocity: 0.01, depth: 0.01, bottom_width: 0.01 }
    }

    fn led(n: usize, sites: &[(ClampSite, &[usize])]) -> StepLedger {
        let mut st = StepLedger { n, active: Default::default() };
        for (s, idx) in sites {
            let mut v = vec![false; n];
            for &i in *idx {
                v[i] = true;
            }
            st.active.insert(*s, v);
        }
        st
    }

    const N: f32 = 0.035;
    const P: f32 = 20.0;
    const Q: f32 = 0.4;
    const S: f32 = 1e-3;

    fn one(form: GeometryForm, site: ClampSite, q_t: f32) -> ReasonReport {
        let step = led(1, &[(site, &[0])]);
        clamp_reasons(form, 0.0, &step, &[q_t], &[N], &[P], &[Q], &[S], &[1.0], &[vec![]], &mins(), true)
    }

    fn floor_of(form: GeometryForm) -> f32 {
        q_floor(form, 0.0, 0.01, N as f64, P as f64, Q as f64, S as f64) as f32
    }

    #[test]
    fn depth_floor_violation_both_forms() {
        for form in [GeometryForm::PowerLaw, GeometryForm::Trapezoid] {
            let f = floor_of(form);
            let ok = one(form, ClampSite::DepthFloor, 0.5 * f);
            assert!(ok.violations.is_empty(), "{form:?} below the floor is fine");
            assert_eq!(ok.checked[&ClampSite::DepthFloor], 1);
            let bad = one(form, ClampSite::DepthFloor, 2.0 * f);
            assert_eq!(bad.violations.len(), 1, "{form:?} above the floor is a violation");
            assert_eq!(bad.violations[0].0, ClampSite::DepthFloor);
            assert_eq!(bad.violations[0].1, 0);
        }
    }

    /// The closed form agrees with the real power-law depth.
    #[test]
    fn q_floor_matches_the_power_law_depth() {
        use burn::backend::NdArray;
        use burn::tensor::Tensor;
        type B = NdArray<f32>;
        let dev = Default::default();
        let f = floor_of(GeometryForm::PowerLaw);
        let depth_at = |scale: f32| -> f32 {
            let t = |v: f32| Tensor::<B, 1>::from_floats([v], &dev);
            let g = crate::geometry::compute_power_law_geometry::<B>(
                t(N), t(P), t(Q), t(f * scale), t(S), 0.0, 0.0, 1.0,
            );
            g.depth.into_data().to_vec::<f32>().unwrap()[0]
        };
        // With depth_lb = 0 the raw depth shows: it equals d_lb = 0.01 at Q_floor.
        assert!((depth_at(1.0) - 0.01).abs() < 1e-4);
        assert!(depth_at(0.5) < 0.01 && depth_at(2.0) > 0.01);
    }

    fn vel_bound(d: f64) -> f64 {
        0.01 * area(P as f64, Q as f64, d)
    }

    #[test]
    fn velocity_low_violation_power_law() {
        // d_v is above d_lb here: a large v_lb.
        let mut m = mins();
        m.velocity = 0.5;
        let qe = Q as f64 + 1e-6;
        let d_v = (0.5 * N as f64 * (qe + 1.0).powf(2.0 / 3.0) / (S as f64).sqrt()).powf(1.5);
        assert!(d_v > 0.01);
        let bound = (0.5 * area(P as f64, Q as f64, d_v)) as f32;
        let run = |q_t: f32| {
            let step = led(1, &[(ClampSite::VelocityLow, &[0])]);
            clamp_reasons(GeometryForm::PowerLaw, 0.0, &step, &[q_t], &[N], &[P], &[Q], &[S], &[1.0], &[vec![]], &m, true)
        };
        assert!(run(0.5 * bound).violations.is_empty());
        let bad = run(2.0 * bound);
        assert_eq!(bad.violations.len(), 1);
        assert_eq!(bad.violations[0].0, ClampSite::VelocityLow);
    }

    /// Where `d_v < d_lb` the bound uses `d_lb`. The naive `A(d_v)` bound would
    /// report this entry. It must not be reported.
    #[test]
    fn velocity_low_with_d_v_below_d_lb_is_not_reported() {
        let qe = Q as f64 + 1e-6;
        let d_v = (0.01 * N as f64 * (qe + 1.0).powf(2.0 / 3.0) / (S as f64).sqrt()).powf(1.5);
        assert!(d_v < 0.01, "setup: d_v {d_v}");
        let q_t = (0.5 * vel_bound(0.01)) as f32;
        assert!(q_t as f64 > vel_bound(d_v), "setup: the naive bound would flag it");
        let rep = one(GeometryForm::PowerLaw, ClampSite::VelocityLow, q_t);
        assert!(rep.violations.is_empty(), "{:?}", rep.violations);
        assert_eq!(rep.checked[&ClampSite::VelocityLow], 1);
        // Above the true bound it is a violation.
        let bad = one(GeometryForm::PowerLaw, ClampSite::VelocityLow, (2.0 * vel_bound(0.01)) as f32);
        assert_eq!(bad.violations.len(), 1);
    }

    #[test]
    fn trapezoid_velocity_low_is_counted_only() {
        let rep = one(GeometryForm::Trapezoid, ClampSite::VelocityLow, 1e9);
        assert!(rep.violations.is_empty());
        assert_eq!(rep.checked[&ClampSite::VelocityLow], 1);
    }

    fn chain_upstream(n: usize) -> Vec<Vec<usize>> {
        (0..n).map(|i| if i == 0 { vec![] } else { vec![i - 1] }).collect()
    }

    fn floor_run(n: usize, x_sol: &[f32], window: &[usize]) -> ReasonReport {
        floor_run_with(n, x_sol, window, true)
    }

    fn floor_run_with(n: usize, x_sol: &[f32], window: &[usize], exact: bool) -> ReasonReport {
        let all: Vec<usize> = (0..n).collect();
        let step = led(n, &[(ClampSite::DischargeFloor, &all), (ClampSite::OutsideCrWindow, window)]);
        let ones = vec![1.0_f32; n];
        clamp_reasons(
            GeometryForm::PowerLaw, 0.0, &step, &ones, &vec![N; n], &vec![P; n], &vec![Q; n],
            &vec![S; n], x_sol, &chain_upstream(n), &mins(), exact,
        )
    }

    /// Reach 0 breaks the window. Reaches 1 and 2 are inside it and negative
    /// only because of the negative reach upstream, two levels up for reach 2.
    #[test]
    fn discharge_floor_chain_through_in_window_reaches_is_not_reported() {
        let rep = floor_run(4, &[-1.0, -0.5, -0.2, 3.0], &[0]);
        assert!(rep.violations.is_empty(), "{:?}", rep.violations);
        assert_eq!(rep.checked[&ClampSite::DischargeFloor], 3);
    }

    #[test]
    fn discharge_floor_negative_without_cause_is_reported() {
        let rep = floor_run(2, &[1.0, -1.0], &[]);
        assert_eq!(rep.violations.len(), 1);
        assert_eq!(rep.violations[0].0, ClampSite::DischargeFloor);
        assert_eq!(rep.violations[0].1, 1);
    }

    #[test]
    fn discharge_floor_non_negative_has_no_rule() {
        let rep = floor_run(2, &[1.0, 5e-5], &[]);
        assert!(rep.violations.is_empty());
        assert!(!rep.checked.contains_key(&ClampSite::DischargeFloor));
    }

    #[test]
    fn discharge_floor_is_count_only_with_leakance_or_reservoirs() {
        let rep = floor_run_with(2, &[1.0, -1.0], &[], false);
        assert!(rep.violations.is_empty());
        assert!(rep.discharge_rule_skipped);
        assert_eq!(rep.checked[&ClampSite::DischargeFloor], 2);
    }

    #[test]
    fn d_ref_other_than_one_is_refused() {
        let e = check_preconditions(0.18, 2.0).unwrap_err();
        assert!(e.contains("params.stage_roughness.d_ref"), "{e}");
        assert!(check_preconditions(0.18, 1.0).is_ok());
        assert!(check_preconditions(0.0, 2.0).is_ok());
    }
}
