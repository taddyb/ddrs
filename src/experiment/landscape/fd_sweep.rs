//! Finite-difference step sweep of the landscape Hessian.
//!
//! `Objective::hessian` takes a central difference of adjoint gradients at one
//! step `h`. The routing core has clamps, so the loss is piecewise smooth. A
//! difference that straddles a clamp switch mixes two pieces. This module
//! repeats the difference at a list of steps. For each step and each active
//! axis it counts the clamp states that differ between the plus and the minus
//! evaluation, using the clamp ledger (`routing/clamp_ledger.rs`). A frozen
//! axis has no rows: its Hessian column is zero by construction.

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use super::objective::{eig_active, Objective};
use crate::routing::clamp_ledger::{ClampLedger, ClampSite};

/// The `landscape.fd_sweep` block. Present means on.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FdSweepSpec {
    /// Steps `h` in log-multiplier units. Strictly decreasing.
    #[serde(default = "d_sweep_steps")]
    pub steps: Vec<f32>,
    /// Also sweep at the Newton optimum, not only at the trained point.
    #[serde(default)]
    pub at_optimum: bool,
}

fn d_sweep_steps() -> Vec<f32> {
    vec![0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
}

impl FdSweepSpec {
    /// Steps must be non-empty, finite, positive and strictly decreasing.
    pub fn validate(&self) -> Result<(), crate::experiment::BoxError> {
        if self.steps.is_empty() {
            return Err("landscape.fd_sweep.steps is empty".into());
        }
        if self.steps.iter().any(|h| !h.is_finite()) {
            return Err("landscape.fd_sweep.steps must be finite".into());
        }
        if self.steps.iter().any(|h| *h <= 0.0) {
            return Err("landscape.fd_sweep.steps must be positive".into());
        }
        if self.steps.windows(2).any(|w| w[1] >= w[0]) {
            return Err("landscape.fd_sweep.steps must be strictly decreasing".into());
        }
        Ok(())
    }
}

/// One step `h` along one active axis `k`.
#[derive(Debug, Clone)]
pub struct FdSweepRow {
    pub h: f32,
    pub axis: usize,
    /// Column `k` of the unsymmetrized Hessian: `(g(a+h e_k) - g(a-h e_k)) / 2h`.
    pub h_col: [f32; 3],
    /// Ledger steps times reaches.
    pub reach_steps: u64,
    /// Entries where each clamp site differs between the plus and minus runs.
    pub switches: BTreeMap<ClampSite, u64>,
    /// Entries where the parameter-range clamp differs, all axes and windows.
    pub range_switches: u64,
    /// (step, reach) entries where at least one clamp site differs.
    pub switched_entries: u64,
}

/// Eigenvalues of the symmetrized Hessian at one `h`. A frozen axis is NaN.
#[derive(Debug, Clone)]
pub struct FdSweepEig {
    pub h: f32,
    pub eig: [f32; 3],
}

/// The 11 real clamp sites, in `ClampSite::ALL` order. `OutsideCrWindow` is
/// information about the coefficients, not a clamp.
pub fn clamp_sites() -> impl Iterator<Item = ClampSite> {
    ClampSite::ALL.into_iter().filter(|s| *s != ClampSite::OutsideCrWindow)
}

/// Number of (step, reach) entries where at least one of the 11 clamp sites
/// differs between the two ledgers. A site missing on one side counts as
/// inactive there. Each entry counts once.
pub fn switched_entries(a: &ClampLedger, b: &ClampLedger) -> u64 {
    assert_eq!(a.steps.len(), b.steps.len(), "ledgers have different step counts");
    let mut total = 0u64;
    for (x, y) in a.steps.iter().zip(&b.steps) {
        assert_eq!(x.n, y.n, "ledgers differ in reach count");
        for i in 0..x.n {
            let differs = clamp_sites().any(|s| {
                let p = x.active.get(&s).map(|v| v[i]).unwrap_or(false);
                let q = y.active.get(&s).map(|v| v[i]).unwrap_or(false);
                p != q
            });
            if differs {
                total += 1;
            }
        }
    }
    total
}

/// Symmetrized Hessian at step `h` from the rows of one sweep. Same algebra as
/// `Objective::hessian`, so it matches that function bit for bit at the same
/// `h` and `alpha`.
pub fn hessian_from_rows(rows: &[FdSweepRow], h: f32) -> [[f32; 3]; 3] {
    let mut hm = [[0.0f32; 3]; 3];
    for r in rows.iter().filter(|r| r.h == h) {
        for j in 0..3 {
            hm[j][r.axis] = r.h_col[j];
        }
    }
    let mut s = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            s[i][j] = 0.5 * (hm[i][j] + hm[j][i]);
        }
    }
    s
}

/// Sweep the FD step at `alpha`. Two evaluations per active axis per step.
pub fn fd_step_sweep<I: Backend + 'static>(obj: &Objective<I>, alpha: [f32; 3], steps: &[f32]) -> (Vec<FdSweepRow>, Vec<FdSweepEig>)
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let active = obj.active();
    let mut rows = Vec::new();
    let mut eigs = Vec::new();
    for &h in steps {
        for k in 0..3 {
            if !active[k] {
                continue;
            }
            let mut ap = alpha;
            ap[k] += h;
            let mut am = alpha;
            am[k] -= h;
            let (e_p, l_p) = obj.eval_with_ledger(ap, true);
            let (e_m, l_m) = obj.eval_with_ledger(am, true);
            let (g_p, g_m) = (e_p.grad.unwrap(), e_m.grad.unwrap());
            let mut h_col = [0.0f32; 3];
            for j in 0..3 {
                h_col[j] = (g_p[j] - g_m[j]) / (2.0 * h);
            }
            let reach_steps: u64 = l_p.steps.iter().map(|s| s.n as u64).sum();
            let switches: BTreeMap<ClampSite, u64> = clamp_sites().map(|s| (s, l_p.switches(&l_m, s))).collect();
            let (r_p, r_m) = (obj.range_active(ap), obj.range_active(am));
            let range_switches: u64 = (0..3).map(|a| r_p[a].iter().zip(&r_m[a]).filter(|(x, y)| x != y).count() as u64).sum();
            rows.push(FdSweepRow { h, axis: k, h_col, reach_steps, switches, range_switches, switched_entries: switched_entries(&l_p, &l_m) });
        }
        let (eig, _) = eig_active(hessian_from_rows(&rows, h), active);
        eigs.push(FdSweepEig { h, eig });
    }
    (rows, eigs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::clamp_ledger::{ClampLedger, ClampSite, StepLedger};

    fn ledger(bits: &[bool]) -> ClampLedger {
        let mut active = std::collections::BTreeMap::new();
        active.insert(ClampSite::DepthFloor, bits.to_vec());
        ClampLedger { steps: vec![StepLedger { n: bits.len(), active }] }
    }

    #[test]
    fn switches_of_a_ledger_with_itself_is_zero() {
        let a = ledger(&[true, false, true]);
        assert_eq!(a.switches(&a, ClampSite::DepthFloor), 0);
        assert_eq!(switched_entries(&a, &a), 0);
    }

    #[test]
    fn one_changed_entry_is_one_switch() {
        let a = ledger(&[true, false, true]);
        let b = ledger(&[true, true, true]);
        assert_eq!(a.switches(&b, ClampSite::DepthFloor), 1);
        assert_eq!(switched_entries(&a, &b), 1);
    }
}
