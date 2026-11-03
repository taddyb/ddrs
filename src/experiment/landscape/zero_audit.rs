//! Audit of the per-reach gradients that are exactly 0.
//!
//! `Objective::reach_grad` gives `dL/d ln x_i` for each reach `i` and axis
//! `n`, `p`, `q`. Many entries are exactly 0.0. This module gives each zero
//! one cause and counts the zeros that have none. The gate is that every zero
//! is either explained or, if it is not, flat under a +-0.05 perturbation of
//! its own leaf (a multi-reach cut that the table below does not model).
//!
//! # Classes, in order
//!
//! Each zero entry gets the first class that holds.
//!
//! 1. `no_path`: the reach is not upstream of the gauge (`dist_to_gauge` is
//!    NaN), so the gauge cannot see it.
//! 2. `range_clamp`: the parameter is outside its range at this reach in
//!    every window, so the range clamp passes no gradient.
//! 3. `masked_row`: a dam row or a stage-table row. The class exists for
//!    completeness. The paper studies refuse reservoir and stage arms, so it
//!    counts 0 there.
//! 4. `gauge_floor`: `DischargeFloor` is active at the gauge row at every
//!    step of every window, so the gauge discharge is the constant floor.
//! 5. `clamp_cut`: a cut formula (below) holds for the reach at every step of
//!    every window.
//! 6. `unexplained`: none of the above.
//!
//! # The cut table
//!
//! The formulas are sufficient conditions for a zero local gradient at one
//! step. They are not necessary ones. They hold for `ddr_match: false`, no
//! leakance and no reservoirs. "Velocity" means `VelocityLow` or
//! `VelocityHigh`. "X band" means `XBandLow` or `XBandHigh`. `KFloor` and
//! `XCap` exist only with `enforce_positivity`; without it they are never
//! active.
//!
//! | Form | n | p | q |
//! |---|---|---|---|
//! | power_law | `DepthFloor`, or velocity and (X band or `XCap`) | velocity and (X band or `XCap`) | velocity and `KFloor` and (X band or `XCap`) |
//! | trapezoid | `DepthFloor` and velocity, or velocity and `KFloor` and (X band or `XCap`) | velocity and `KFloor` and (X band or `XCap`) | velocity and `KFloor` and (X band or `XCap`) |
//!
//! Why, for power_law: `n` reaches the next discharge only through the depth
//! (`T`, `A`, `v = Q/A`, `c = v * beta(q)`). `DepthFloor` fixes the depth, so
//! `n` is cut. A velocity site fixes `c = v_bound * beta(q)`, so `K` is free
//! of `n` and `p`. Only the Cunge X path (through `T`) is left, and an X band
//! site or `XCap` cuts it. `q` is also in `beta`, so `K` needs `KFloor` too.
//!
//! Why, for trapezoid: with `DepthFloor` alone, `v = (1/n) R^(2/3) sqrt(s)`
//! still carries `n`, so `n` needs a velocity site as well. `beta` depends on
//! the whole section, so without `DepthFloor` the `K` path needs `KFloor`.
//!
//! With the four paper configs (`enforce_positivity` off) `KFloor` and `XCap`
//! never fire. Then the rows that can hold are: power_law `n` (`DepthFloor`,
//! or velocity and X band), power_law `p` (velocity and X band), and
//! trapezoid `n` (`DepthFloor` and velocity). The `q` columns and the
//! trapezoid `p` column cannot hold.

use std::path::Path;

use burn::tensor::backend::Backend;

use super::objective::{Objective, ReachGrad};
use super::output::{append_zero_audit, ZeroAuditRow};
use crate::config::GeometryForm;
use crate::experiment::BoxError;
use crate::routing::clamp_ledger::{ClampLedger, ClampSite, StepLedger};

/// Size of the single-leaf perturbation for an `unexplained` entry.
pub const PERTURB: f32 = 0.05;
/// An unexplained entry is flat when `|dL| < FLAT_REL * |L|`.
pub const FLAT_REL: f32 = 1e-6;

/// The audited axes, in table order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    N,
    P,
    Q,
}

impl Axis {
    /// The axis for a `landscape.axes` name. `None` for any other name.
    pub fn from_param(name: &str) -> Option<Axis> {
        match name {
            "n" => Some(Axis::N),
            "p_spatial" => Some(Axis::P),
            "q_spatial" => Some(Axis::Q),
            _ => None,
        }
    }
}

/// The cause of one zero entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroClass {
    NoPath,
    RangeClamp,
    MaskedRow,
    GaugeFloor,
    ClampCut,
    Unexplained,
}

fn on(step: &StepLedger, site: ClampSite, i: usize) -> bool {
    step.active.get(&site).map(|v| v[i]).unwrap_or(false)
}

/// True when the cut formula for `form` and `axis` holds at one step on
/// reach `i`. See the module doc for the table and the reasons.
pub fn cut_holds(form: GeometryForm, axis: Axis, step: &StepLedger, i: usize) -> bool {
    let velocity = on(step, ClampSite::VelocityLow, i) || on(step, ClampSite::VelocityHigh, i);
    let x_cut = on(step, ClampSite::XBandLow, i) || on(step, ClampSite::XBandHigh, i) || on(step, ClampSite::XCap, i);
    let depth = on(step, ClampSite::DepthFloor, i);
    let k_floor = on(step, ClampSite::KFloor, i);
    match (form, axis) {
        (GeometryForm::PowerLaw, Axis::N) => depth || (velocity && x_cut),
        (GeometryForm::PowerLaw, Axis::P) => velocity && x_cut,
        (GeometryForm::PowerLaw, Axis::Q) => velocity && k_floor && x_cut,
        (GeometryForm::Trapezoid, Axis::N) => (depth && velocity) || (velocity && k_floor && x_cut),
        (GeometryForm::Trapezoid, Axis::P) | (GeometryForm::Trapezoid, Axis::Q) => velocity && k_floor && x_cut,
    }
}

/// What the classifier reads. `range_active[k]` has `n_windows * n_reach`
/// entries, windows first (the layout of `Objective::range_active`).
pub struct AuditInput<'a> {
    pub form: GeometryForm,
    pub ledger: &'a ClampLedger,
    /// Routed steps per window, in window order. Sums to the ledger length.
    pub steps_per_window: &'a [usize],
    /// Gauge row of each window.
    pub gauge_rows: &'a [usize],
    pub range_active: &'a [Vec<bool>; 3],
    /// `dist_to_gauge` per reach. NaN means no path to the gauge.
    pub dist: &'a [f32],
    /// Dam rows and stage-table rows. All false in the paper studies.
    pub masked: &'a [bool],
}

impl AuditInput<'_> {
    pub fn n_reach(&self) -> usize {
        self.dist.len()
    }

    /// Stop when the ledger does not have one step per routed step of every
    /// window. A cut that holds "at every step" over a short ledger would be a
    /// false explanation.
    pub fn check_coverage(&self) -> Result<(), String> {
        let want: usize = self.steps_per_window.iter().sum();
        if self.ledger.steps.len() != want {
            return Err(format!("clamp ledger has {} steps, the windows route {want}", self.ledger.steps.len()));
        }
        if self.ledger.steps.is_empty() {
            return Err("clamp ledger is empty".into());
        }
        if self.steps_per_window.len() != self.gauge_rows.len() {
            return Err("steps_per_window and gauge_rows differ in length".into());
        }
        if let Some(s) = self.ledger.steps.iter().find(|s| s.n != self.n_reach()) {
            return Err(format!("ledger step has {} reaches, the network has {}", s.n, self.n_reach()));
        }
        Ok(())
    }

    /// DischargeFloor is active at the gauge row at every step of every window.
    fn gauge_floor(&self) -> bool {
        let mut step = 0usize;
        for (w, &len) in self.steps_per_window.iter().enumerate() {
            for s in &self.ledger.steps[step..step + len] {
                if !on(s, ClampSite::DischargeFloor, self.gauge_rows[w]) {
                    return false;
                }
            }
            step += len;
        }
        true
    }

    fn range_clamped(&self, slot: usize, i: usize) -> bool {
        let n = self.n_reach();
        let v = &self.range_active[slot];
        let n_windows = self.steps_per_window.len();
        (0..n_windows).all(|w| v[w * n + i])
    }

    /// The class of a zero entry. `slot` is the alpha slot, which indexes
    /// `range_active`. Call it only for entries whose gradient is exactly 0.
    pub fn classify(&self, slot: usize, axis: Axis, i: usize, gauge_floor: bool) -> ZeroClass {
        if self.dist[i].is_nan() {
            ZeroClass::NoPath
        } else if self.range_clamped(slot, i) {
            ZeroClass::RangeClamp
        } else if self.masked[i] {
            ZeroClass::MaskedRow
        } else if gauge_floor {
            ZeroClass::GaugeFloor
        } else if self.ledger.steps.iter().all(|s| cut_holds(self.form, axis, s, i)) {
            ZeroClass::ClampCut
        } else {
            ZeroClass::Unexplained
        }
    }
}

/// Counts for one axis at one point.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AxisCounts {
    pub n_reach: usize,
    pub n_zero: usize,
    pub no_path: usize,
    pub range_clamp: usize,
    pub masked_row: usize,
    pub gauge_floor: usize,
    pub clamp_cut: usize,
    pub unexplained: usize,
    /// Reaches of the `unexplained` class, for the perturbation check.
    pub unexplained_reaches: Vec<usize>,
}

/// Classify every exactly-zero entry of `grad` (one axis, one value per
/// reach). Panics on a coverage error: call `check_coverage` first to get it
/// as a `Result`.
pub fn audit_axis(input: &AuditInput, slot: usize, axis: Axis, grad: &[f32]) -> AxisCounts {
    input.check_coverage().expect("ledger coverage");
    assert_eq!(grad.len(), input.n_reach(), "one gradient per reach");
    let gauge_floor = input.gauge_floor();
    let mut c = AxisCounts { n_reach: grad.len(), ..Default::default() };
    for (i, g) in grad.iter().enumerate() {
        if *g != 0.0 {
            continue;
        }
        c.n_zero += 1;
        match input.classify(slot, axis, i, gauge_floor) {
            ZeroClass::NoPath => c.no_path += 1,
            ZeroClass::RangeClamp => c.range_clamp += 1,
            ZeroClass::MaskedRow => c.masked_row += 1,
            ZeroClass::GaugeFloor => c.gauge_floor += 1,
            ZeroClass::ClampCut => c.clamp_cut += 1,
            ZeroClass::Unexplained => {
                c.unexplained += 1;
                c.unexplained_reaches.push(i);
            }
        }
    }
    c
}

/// How many steps have `site` active on reach `i`, for the report.
pub fn active_steps(ledger: &ClampLedger, site: ClampSite, i: usize) -> usize {
    ledger.steps.iter().filter(|s| on(s, site, i)).count()
}

/// True when the arm can be audited: a plain Muskingum-Cunge path (no
/// leakance, no reservoirs, no stage routing) and axes that are all `n`, `p`
/// or `q`. The clamp ledger covers that path only.
pub fn auditable<I: Backend + 'static>(obj: &Objective<I>) -> bool
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let p = &obj.ctx.cfg.params;
    !p.use_leakance && !p.use_reservoirs && obj.axes.iter().all(|a| Axis::from_param(a).is_some())
}

/// Audit one gauge at one point and append its rows to `zero_audit.csv`.
/// Returns the gate violations: `unexplained` entries whose perturbation is
/// not flat. Prints one line per such entry with the number of steps each
/// clamp site is active on that reach.
#[allow(clippy::too_many_arguments)]
pub fn audit_gauge<I: Backend + 'static>(
    obj: &Objective<I>,
    arm: &str,
    staid: &str,
    arm_dir: &Path,
    point: &str,
    alpha: [f32; 3],
    rg: &ReachGrad,
    dist: &[f32],
) -> Result<usize, BoxError>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let (e, ledger) = obj.eval_with_ledger(alpha, false);
    let steps_per_window = obj.routed_steps_per_window();
    let gauge_rows: Vec<usize> = obj.windows.iter().map(|w| w.gauge_row).collect();
    let range_active = obj.range_active(alpha);
    let masked = vec![false; obj.n_reach()];
    let input = AuditInput {
        form: obj.ctx.cfg.params.geometry_form,
        ledger: &ledger,
        steps_per_window: &steps_per_window,
        gauge_rows: &gauge_rows,
        range_active: &range_active,
        dist,
        masked: &masked,
    };
    input.check_coverage()?;
    let l0 = e.loss;
    let mut rows = Vec::new();
    let mut counts_all = Vec::new();
    let mut violations = 0usize;
    let mut checked_zero_perturbation = false;
    for slot in 0..3 {
        let axis = Axis::from_param(&obj.axes[slot]).expect("auditable arm");
        let counts = audit_axis(&input, slot, axis, &rg.slots[slot]);
        let mut flat = 0usize;
        for &i in &counts.unexplained_reaches {
            if !checked_zero_perturbation {
                let again = obj.reach_loss(alpha, None);
                assert_eq!(again.to_bits(), l0.to_bits(), "reach_loss with no perturbation must equal eval bitwise");
                checked_zero_perturbation = true;
            }
            let d = [PERTURB, -PERTURB].iter().map(|dx| (obj.reach_loss(alpha, Some((slot, i, *dx))) - l0).abs()).fold(0.0f32, f32::max);
            let is_flat = d < FLAT_REL * l0.abs();
            if is_flat {
                flat += 1;
            } else {
                violations += 1;
            }
            let sites: Vec<String> = super::fd_sweep::clamp_sites().map(|s| format!("{}={}", s.name(), active_steps(&ledger, s, i))).collect();
            println!(
                "  [{arm}] {staid} zero_audit UNEXPLAINED ({point}, axis {}, reach {i}): |dL| {d:e}, |dL|/|L| {:e}, flat {is_flat}; ledger steps {}; active steps per site: {}",
                super::axis_label(&obj.axes[slot]),
                d / l0.abs(),
                ledger.steps.len(),
                sites.join(" ")
            );
        }
        counts_all.push((slot, counts, flat));
    }
    for (slot, counts, flat) in &counts_all {
        rows.push(ZeroAuditRow { point, axis: super::axis_label(&obj.axes[*slot]), counts, unexplained_flat: *flat });
    }
    append_zero_audit(arm_dir, arm, staid, &rows)?;
    for r in &rows {
        println!(
            "  [{arm}] {staid} zero_audit ({point}, {}): n_zero {} no_path {} range_clamp {} masked_row {} gauge_floor {} clamp_cut {} unexplained {} (flat {})",
            r.axis, r.counts.n_zero, r.counts.no_path, r.counts.range_clamp, r.counts.masked_row, r.counts.gauge_floor, r.counts.clamp_cut, r.counts.unexplained, r.unexplained_flat
        );
    }
    Ok(violations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const N: usize = 4;

    /// One-window ledger of `steps` steps. `set` lists (site, reach) pairs
    /// that are active at every step.
    fn ledger(steps: usize, set: &[(ClampSite, usize)]) -> ClampLedger {
        let mut active: BTreeMap<ClampSite, Vec<bool>> = BTreeMap::new();
        for (s, i) in set {
            active.entry(*s).or_insert_with(|| vec![false; N])[*i] = true;
        }
        ClampLedger { steps: vec![StepLedger { n: N, active }; steps] }
    }

    struct Fx {
        ledger: ClampLedger,
        range: [Vec<bool>; 3],
        dist: Vec<f32>,
        masked: Vec<bool>,
        gauge_row: usize,
    }

    impl Fx {
        fn new(ledger: ClampLedger) -> Fx {
            Fx { ledger, range: [vec![false; N], vec![false; N], vec![false; N]], dist: vec![0.0, 1.0, 2.0, 3.0], masked: vec![false; N], gauge_row: 0 }
        }
        fn run(&self, form: GeometryForm, axis: Axis, grad: &[f32]) -> AxisCounts {
            let n_steps = self.ledger.steps.len();
            let rows = [self.gauge_row];
            let steps = [n_steps];
            let input = AuditInput {
                form,
                ledger: &self.ledger,
                steps_per_window: &steps,
                gauge_rows: &rows,
                range_active: &self.range,
                dist: &self.dist,
                masked: &self.masked,
            };
            audit_axis(&input, 0, axis, grad)
        }
    }

    // Reach 1 is the zero under test. Reaches 0, 2, 3 have nonzero gradient.
    const G: [f32; N] = [1.0, 0.0, 1.0, 1.0];

    #[test]
    fn no_path_when_distance_is_nan() {
        let mut fx = Fx::new(ledger(3, &[]));
        fx.dist[1] = f32::NAN;
        let c = fx.run(GeometryForm::PowerLaw, Axis::N, &G);
        assert_eq!((c.n_zero, c.no_path, c.unexplained), (1, 1, 0));
    }

    #[test]
    fn range_clamp_needs_every_window() {
        let mut fx = Fx::new(ledger(3, &[]));
        fx.range[0][1] = true;
        let c = fx.run(GeometryForm::PowerLaw, Axis::N, &G);
        assert_eq!((c.range_clamp, c.unexplained), (1, 0));
        // Two windows, the clamp holds in only one of them: not a range clamp.
        let l = ledger(2, &[]);
        let range = [vec![false, true, false, false, false, false, false, false], vec![false; 2 * N], vec![false; 2 * N]];
        let input = AuditInput {
            form: GeometryForm::PowerLaw,
            ledger: &l,
            steps_per_window: &[1, 1],
            gauge_rows: &[0, 0],
            range_active: &range,
            dist: &fx.dist,
            masked: &fx.masked,
        };
        let c = audit_axis(&input, 0, Axis::N, &G);
        assert_eq!((c.range_clamp, c.unexplained), (0, 1));
    }

    #[test]
    fn masked_row_is_counted() {
        let mut fx = Fx::new(ledger(3, &[]));
        fx.masked[1] = true;
        let c = fx.run(GeometryForm::PowerLaw, Axis::N, &G);
        assert_eq!((c.masked_row, c.unexplained), (1, 0));
    }

    #[test]
    fn gauge_floor_needs_every_step() {
        let fx = Fx::new(ledger(3, &[(ClampSite::DischargeFloor, 0)]));
        let c = fx.run(GeometryForm::PowerLaw, Axis::N, &G);
        assert_eq!((c.gauge_floor, c.unexplained), (1, 0));
        // Active at the gauge row at only some steps: not a gauge floor.
        let mut l = ledger(3, &[(ClampSite::DischargeFloor, 0)]);
        l.steps[1].active.get_mut(&ClampSite::DischargeFloor).unwrap()[0] = false;
        let fx = Fx::new(l);
        let c = fx.run(GeometryForm::PowerLaw, Axis::N, &G);
        assert_eq!((c.gauge_floor, c.unexplained), (0, 1));
    }

    #[test]
    fn clamp_cut_follows_the_table() {
        // power_law n: DepthFloor alone cuts it.
        let fx = Fx::new(ledger(3, &[(ClampSite::DepthFloor, 1)]));
        assert_eq!(fx.run(GeometryForm::PowerLaw, Axis::N, &G).clamp_cut, 1);
        // trapezoid n: DepthFloor alone does not; with a velocity site it does.
        assert_eq!(fx.run(GeometryForm::Trapezoid, Axis::N, &G).unexplained, 1);
        let fx = Fx::new(ledger(3, &[(ClampSite::DepthFloor, 1), (ClampSite::VelocityHigh, 1)]));
        assert_eq!(fx.run(GeometryForm::Trapezoid, Axis::N, &G).clamp_cut, 1);
        // power_law p: velocity plus an X site; velocity alone does not.
        let fx = Fx::new(ledger(3, &[(ClampSite::VelocityLow, 1)]));
        assert_eq!(fx.run(GeometryForm::PowerLaw, Axis::P, &G).unexplained, 1);
        let fx = Fx::new(ledger(3, &[(ClampSite::VelocityLow, 1), (ClampSite::XBandHigh, 1)]));
        assert_eq!(fx.run(GeometryForm::PowerLaw, Axis::P, &G).clamp_cut, 1);
        // power_law q also needs KFloor.
        assert_eq!(fx.run(GeometryForm::PowerLaw, Axis::Q, &G).unexplained, 1);
        let fx = Fx::new(ledger(3, &[(ClampSite::VelocityLow, 1), (ClampSite::XCap, 1), (ClampSite::KFloor, 1)]));
        assert_eq!(fx.run(GeometryForm::PowerLaw, Axis::Q, &G).clamp_cut, 1);
        assert_eq!(fx.run(GeometryForm::Trapezoid, Axis::P, &G).clamp_cut, 1);
    }

    #[test]
    fn clamp_cut_needs_every_step() {
        let mut l = ledger(3, &[(ClampSite::DepthFloor, 1)]);
        l.steps[2].active.get_mut(&ClampSite::DepthFloor).unwrap()[1] = false;
        let c = Fx::new(l).run(GeometryForm::PowerLaw, Axis::N, &G);
        assert_eq!((c.clamp_cut, c.unexplained, c.unexplained_reaches), (0, 1, vec![1]));
    }

    #[test]
    fn unexplained_when_nothing_holds() {
        let c = Fx::new(ledger(3, &[])).run(GeometryForm::PowerLaw, Axis::N, &G);
        assert_eq!((c.n_zero, c.unexplained), (1, 1));
    }

    #[test]
    fn classes_apply_in_the_documented_order() {
        // Reach 1 is no_path, range-clamped, masked, under a gauge floor and
        // cut. The first class wins.
        let mut fx = Fx::new(ledger(2, &[(ClampSite::DischargeFloor, 0), (ClampSite::DepthFloor, 1)]));
        fx.masked[1] = true;
        fx.range[0][1] = true;
        assert_eq!(fx.run(GeometryForm::PowerLaw, Axis::N, &G).range_clamp, 1);
        fx.dist[1] = f32::NAN;
        assert_eq!(fx.run(GeometryForm::PowerLaw, Axis::N, &G).no_path, 1);
        fx.dist[1] = 1.0;
        fx.range[0][1] = false;
        assert_eq!(fx.run(GeometryForm::PowerLaw, Axis::N, &G).masked_row, 1);
        fx.masked[1] = false;
        assert_eq!(fx.run(GeometryForm::PowerLaw, Axis::N, &G).gauge_floor, 1);
    }

    #[test]
    fn nonzero_gradients_are_not_audited() {
        let c = Fx::new(ledger(2, &[])).run(GeometryForm::PowerLaw, Axis::N, &[1.0, -2.0, 1e-30, 1.0]);
        assert_eq!(c.n_zero, 0);
    }

    #[test]
    fn coverage_check_rejects_a_short_ledger() {
        let l = ledger(2, &[]);
        let range = [vec![false; N], vec![false; N], vec![false; N]];
        let input = AuditInput {
            form: GeometryForm::PowerLaw,
            ledger: &l,
            steps_per_window: &[3],
            gauge_rows: &[0],
            range_active: &range,
            dist: &[0.0; N],
            masked: &[false; N],
        };
        assert!(input.check_coverage().unwrap_err().contains("2 steps"));
    }

    #[test]
    fn axis_names_map_to_table_columns() {
        assert_eq!(Axis::from_param("n"), Some(Axis::N));
        assert_eq!(Axis::from_param("p_spatial"), Some(Axis::P));
        assert_eq!(Axis::from_param("q_spatial"), Some(Axis::Q));
        assert_eq!(Axis::from_param("gamma"), None);
    }
}
