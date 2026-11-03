//! Zero-pattern parity of the fused MC timestep gradient against plain Burn
//! autodiff, on a routed trajectory.
//!
//! The loop shape follows [`crate::routing::courant_probe::run_courant_probe`]:
//! forcing clamp, subdivision divisor, and `q_t` fed back from the previous
//! step. The trajectory itself runs through `forward_chain_inner`. At every
//! sampled step the fused op
//! ([`crate::routing::mmc_op::timestep_forward_diag`]) and the autodiff
//! reference ([`crate::routing::reference_step::reference_timestep`]) both run
//! from the same `q_t`. The loss is `sum(q_next * w)` with `w_i = 1 + (i % 7)`.
//! Five parents are tracked: n, q_spatial, p_spatial, q_t, q_prime_t.
//!
//! Per parent, the set of entries equal to 0.0 must match. A non-finite entry
//! on either side counts as a zero-set mismatch. Nonzero entries that differ
//! by more than rel 1e-3 and abs 1e-6 are reported as value outliers only.

use std::collections::BTreeMap;

use burn::backend::Autodiff;
use burn::tensor::{backend::Backend, Tensor, TensorPrimitive};

use crate::config::Config;
use crate::routing::clamp_ledger::{ClampLedger, ClampSite};
use crate::routing::courant_probe::ProbeInputs;
use crate::routing::mmc_op::{forward_chain_inner, forward_saved_idx, timestep_forward_diag};
use crate::routing::reference_step::reference_timestep;
use crate::sparse::AValuesAssembler;

type AB<I> = Autodiff<I>;

/// Parent names, in the order the report keys use.
pub const PARENTS: [&str; 5] = ["n", "q_spatial", "p_spatial", "q_t", "q_prime_t"];

const REL_TOL: f32 = 1e-3;
const ABS_TOL: f32 = 1e-6;
const MAX_FIRST_MISMATCHES: usize = 20;

/// What the comparison found. Map keys are the entries of [`PARENTS`].
#[derive(Clone, Debug, Default)]
pub struct ClampParityReport {
    pub n_reaches: usize,
    pub n_sampled_steps: usize,
    /// Zero-set mismatches per parent (reach entries, summed over sampled steps).
    pub mismatches: BTreeMap<String, u64>,
    /// Entries compared per parent.
    pub compared: BTreeMap<String, u64>,
    /// Active (step, reach) entries per clamp site, from the fused ledger.
    pub active: BTreeMap<ClampSite, u64>,
    /// Per parent: entries nonzero on both sides that fail
    /// `rel < 1e-3 || abs < 1e-6`. Reported, not gated.
    pub value_outliers: BTreeMap<String, u64>,
    /// The first 20 zero-set mismatches, with the ledger row of the reach.
    pub first_mismatches: Vec<String>,
    /// Largest `|q_ref - q_fused| / max(|q_fused|, 1e-6)` over sampled steps.
    pub max_q_rel_diff: f32,
    /// Per parent, 8 cells indexed by [`cell_index`]. Diagnostic only.
    pub cells: BTreeMap<String, [CellStats; 8]>,
    /// Zero-set mismatches on clamp-active reaches, counted per parent entry,
    /// keyed by (active clamp sites joined by `+`, steadiness band name).
    pub clamp_groups: BTreeMap<(String, String), u64>,
    /// Full dumps of reaches with a zero-set mismatch in the `>1e-4` band.
    pub unsteady_details: Vec<String>,
}

/// Steadiness bands of a reach at one step, from the trajectory values.
pub const BANDS: [&str; 4] = ["bitwise", "<=1e-6", "<=1e-4", ">1e-4"];

/// Counts for one (steadiness band, clamp attribution) cell of one parent.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellStats {
    /// All compared entries in the cell (the base rate).
    pub compared: u64,
    pub mismatches: u64,
    pub outliers: u64,
    /// Largest |g| on the nonzero side of a zero-set mismatch.
    pub max_mismatch_abs_g: f32,
    /// Largest |fused - reference| over value outliers.
    pub max_outlier_abs_diff: f32,
}

/// Index into [`ClampParityReport::cells`]: `band * 2 + clamped as usize`.
pub fn cell_index(band: usize, clamped: bool) -> usize {
    band * 2 + clamped as usize
}

/// Steadiness `s = max(|q_next - q_t|, |I_next - I_t|) / max(|q_t|, |I_t|, 1e-6)`
/// to a band index in [`BANDS`]. `bitwise` needs both pairs equal bit for bit.
fn band_of(q_t: f32, q_next: f32, i_t: f32, i_next: f32) -> usize {
    if q_next.to_bits() == q_t.to_bits() && i_next.to_bits() == i_t.to_bits() {
        return 0;
    }
    let s = (q_next - q_t).abs().max((i_next - i_t).abs()) / q_t.abs().max(i_t.abs()).max(1e-6);
    if s <= 1e-6 {
        1
    } else if s <= 1e-4 {
        2
    } else {
        3
    }
}

/// `I = N q`: sum of the upstream discharges of each reach.
fn inflow(pattern: &crate::sparse::CsrPattern, q: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; pattern.n];
    for r in 0..pattern.n {
        for k in pattern.crow[r] as usize..pattern.crow[r + 1] as usize {
            out[r] += pattern.adj_values[k] * q[pattern.col[k] as usize];
        }
    }
    out
}

impl ClampParityReport {
    pub fn total_mismatches(&self) -> u64 {
        self.mismatches.values().sum()
    }
}

/// The gate: the gradient pair `(a, b)` differs in its zero set, or either
/// side is not finite.
pub fn zero_set_mismatch(a: f32, b: f32) -> bool {
    !(a.is_finite() && b.is_finite()) || (a == 0.0) != (b == 0.0)
}

fn vec1<I: Backend>(t: Tensor<I, 1>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("f32 tensor")
}

fn ledger_row(l: &ClampLedger, i: usize) -> String {
    let st = &l.steps[0];
    let on: Vec<&str> = ClampSite::ALL
        .iter()
        .filter(|s| st.active.get(s).map(|v| v[i]).unwrap_or(false))
        .map(|s| s.name())
        .collect();
    if on.is_empty() {
        "no site active".to_string()
    } else {
        format!("active: {}", on.join(" "))
    }
}

/// Route `n_steps` hourly steps and compare gradients every `sample_every`.
/// `gamma_at` is `None` on both sides: `ProbeInputs` carries no gamma.
pub fn run_clamp_parity<I: Backend + 'static>(
    cfg: &Config,
    inp: &ProbeInputs<I>,
    n_steps: usize,
    sample_every: usize,
) -> ClampParityReport
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let n = inp.n_segments;
    let discharge_lb = cfg.params.attribute_minimums.discharge;
    let device = inp.n.device();
    let assembler = AValuesAssembler::<I>::new(&inp.pattern, &device);

    let t_avail = inp.q_prime.dims()[0];
    let steps = n_steps.min(t_avail.saturating_sub(1));
    // Same forcing treatment as `run_courant_probe`.
    let q_prime = inp.q_prime.clone().clamp_min(discharge_lb);
    let q_prime = match inp.pieces_per_row.as_ref() {
        Some(d) => q_prime / d.clone().unsqueeze_dim::<2>(0),
        None => q_prime,
    };
    let mut q_t = inp.q0.clone().clamp_min(discharge_lb);

    let weights: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32).collect();

    let mut rep = ClampParityReport { n_reaches: n, ..Default::default() };
    for p in PARENTS {
        rep.mismatches.insert(p.to_string(), 0);
        rep.compared.insert(p.to_string(), 0);
        rep.value_outliers.insert(p.to_string(), 0);
        rep.cells.insert(p.to_string(), [CellStats::default(); 8]);
    }

    for t in 1..=steps {
        let q_prime_t: Tensor<I, 1> = q_prime.clone().slice([(t - 1)..t, 0..n]).reshape([n]);
        let mut x_eff_out = None;
        let mut leak_out = None;
        let (q_next, saved) = forward_chain_inner::<I>(
            cfg,
            &inp.pattern,
            inp.n.clone(),
            inp.q_spatial.clone(),
            inp.p_spatial.clone(),
            q_t.clone(),
            q_prime_t.clone(),
            inp.length.clone(),
            inp.slope.clone(),
            inp.x_storage.clone(),
            None,
            &mut leak_out,
            &mut x_eff_out,
            /* track_neg */ false,
            /* skip_rows */ None,
            /* gamma_in */ None,
            /* reservoir */ None,
            /* ledger */ None,
        );
        let q_next_t: Tensor<I, 1> = Tensor::from_primitive(TensorPrimitive::Float(q_next));

        if t % sample_every == 0 {
            sample_step::<I>(
                cfg, inp, &assembler, &q_t, &q_prime_t, &q_next_t, &weights, t, &mut rep,
                &saved, x_eff_out.clone(),
            );
            rep.n_sampled_steps += 1;
        }
        q_t = q_next_t;
    }
    rep
}

#[allow(clippy::too_many_arguments)]
fn sample_step<I: Backend + 'static>(
    cfg: &Config,
    inp: &ProbeInputs<I>,
    assembler: &AValuesAssembler<I>,
    q_t: &Tensor<I, 1>,
    q_prime_t: &Tensor<I, 1>,
    trajectory_next: &Tensor<I, 1>,
    weights: &[f32],
    step: usize,
    rep: &mut ClampParityReport,
    saved: &[I::FloatTensorPrimitive],
    x_eff: Option<I::FloatTensorPrimitive>,
) where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let n = inp.n_segments;
    let device = inp.n.device();
    let tracked = |t: &Tensor<I, 1>| -> Tensor<AB<I>, 1> {
        Tensor::<AB<I>, 1>::from_inner(t.clone()).require_grad()
    };
    let constant = |t: &Tensor<I, 1>| -> Tensor<AB<I>, 1> { Tensor::<AB<I>, 1>::from_inner(t.clone()) };
    let w: Tensor<AB<I>, 1> = Tensor::<AB<I>, 1>::from_floats(weights, &device);

    // Fused op, with a ledger.
    let (n_f, qs_f, ps_f, qt_f, qp_f) = (
        tracked(&inp.n),
        tracked(&inp.q_spatial),
        tracked(&inp.p_spatial),
        tracked(q_t),
        tracked(q_prime_t),
    );
    let mut ledger = ClampLedger::default();
    let qn_f = timestep_forward_diag::<I>(
        cfg,
        &inp.pattern,
        assembler,
        n_f.clone(),
        qs_f.clone(),
        ps_f.clone(),
        qt_f.clone(),
        qp_f.clone(),
        constant(&inp.length),
        constant(&inp.slope),
        constant(&inp.x_storage),
        None,
        Some(&mut ledger),
        None,
    );
    let fused_next = vec1(qn_f.clone().inner());
    // Same operating point: the fused forward must equal the trajectory forward.
    let traj_next = vec1(trajectory_next.clone());
    assert!(
        fused_next.iter().zip(&traj_next).all(|(a, b)| a.to_bits() == b.to_bits()),
        "step {step}: fused forward differs from the trajectory forward"
    );
    let g_f = (qn_f * w.clone()).sum().backward();
    let grads_f: Vec<Vec<f32>> = [&n_f, &qs_f, &ps_f, &qt_f, &qp_f]
        .iter()
        .map(|p| vec1(p.grad(&g_f).expect("fused gradient")))
        .collect();

    // Autodiff reference.
    let (n_r, qs_r, ps_r, qt_r, qp_r) = (
        tracked(&inp.n),
        tracked(&inp.q_spatial),
        tracked(&inp.p_spatial),
        tracked(q_t),
        tracked(q_prime_t),
    );
    let qn_r = reference_timestep::<I>(
        cfg,
        &inp.pattern,
        assembler,
        n_r.clone(),
        qs_r.clone(),
        ps_r.clone(),
        qt_r.clone(),
        qp_r.clone(),
        constant(&inp.length),
        constant(&inp.slope),
        constant(&inp.x_storage),
        None,
    );
    let ref_next = vec1(qn_r.clone().inner());
    for (r, f) in ref_next.iter().zip(&fused_next) {
        let d = (r - f).abs() / f.abs().max(1e-6);
        if !(d <= rep.max_q_rel_diff) {
            rep.max_q_rel_diff = d;
        }
    }
    let g_r = (qn_r * w).sum().backward();
    let grads_r: Vec<Vec<f32>> = [&n_r, &qs_r, &ps_r, &qt_r, &qp_r]
        .iter()
        .map(|p| vec1(p.grad(&g_r).expect("reference gradient")))
        .collect();

    for s in ClampSite::ALL {
        *rep.active.entry(s).or_insert(0) += ledger.count(s);
    }

    // Steadiness band and clamp attribution of each reach at this step.
    let q_t_v = vec1(q_t.clone());
    let (i_t_v, i_next_v) = (inflow(&inp.pattern, &q_t_v), inflow(&inp.pattern, &traj_next));
    let band: Vec<usize> =
        (0..n).map(|i| band_of(q_t_v[i], traj_next[i], i_t_v[i], i_next_v[i])).collect();
    let clamped: Vec<bool> = (0..n)
        .map(|i| {
            ClampSite::ALL
                .iter()
                .filter(|s| **s != ClampSite::OutsideCrWindow)
                .any(|s| ledger.steps[0].active.get(s).map(|v| v[i]).unwrap_or(false))
        })
        .collect();

    let sites_of = |i: usize| -> Vec<&'static str> {
        ClampSite::ALL
            .iter()
            .filter(|s| ledger.steps[0].active.get(s).map(|v| v[i]).unwrap_or(false))
            .map(|s| s.name())
            .collect()
    };
    let is_mismatch = |pi: usize, i: usize| {
        zero_set_mismatch(grads_f[pi][i], grads_r[pi][i])
    };
    // Full dump of reaches with any mismatch in the >1e-4 band.
    for i in 0..n {
        if band[i] == 3 && (0..PARENTS.len()).any(|pi| is_mismatch(pi, i)) {
            let get = |idx: usize| vec1(Tensor::<I, 1>::from_primitive(TensorPrimitive::Float(saved[idx].clone())));
            let (c1, c2, c3, c4, k) = (
                get(forward_saved_idx::C1),
                get(forward_saved_idx::C2),
                get(forward_saved_idx::C3),
                get(forward_saved_idx::C4),
                get(forward_saved_idx::K_MUSKINGUM),
            );
            let x = x_eff
                .clone()
                .map(|p| vec1(Tensor::<I, 1>::from_primitive(TensorPrimitive::Float(p))));
            // Solve adjoint lambda = A^-T w, A = I - diag(c1) N, by a reverse sweep.
            let mut lam: Vec<f32> = weights.to_vec();
            for r in (0..n).rev() {
                for kk in inp.pattern.crow[r] as usize..inp.pattern.crow[r + 1] as usize {
                    let c = inp.pattern.col[kk] as usize;
                    lam[c] += c1[r] * inp.pattern.adj_values[kk] * lam[r];
                }
            }
            let qpv = vec1(q_prime_t.clone());
            let (dq_i, di) = (q_t_v[i] - i_t_v[i] - qpv[i], i_next_v[i] - i_t_v[i]);
            let s = (traj_next[i] - q_t_v[i]).abs().max(di.abs()) / q_t_v[i].abs().max(i_t_v[i].abs()).max(1e-6);
            let vals: Vec<String> = PARENTS
                .iter()
                .enumerate()
                .map(|(pi, nm)| format!("{nm}: fused {:e} ref {:e}{}", grads_f[pi][i], grads_r[pi][i], if is_mismatch(pi, i) { " MISMATCH" } else { "" }))
                .collect();
            rep.unsteady_details.push(format!(
                "step {step} reach {i} band {} active [{}]\n    {}\n    s {s:e} dI {di:e} dQ {dq_i:e} q_t {:e} I_t {:e} I_next {:e} q_prime_t {:e} q_next {:e}\n    c1 {:e} c2 {:e} c3 {:e} c4 {:e} K {:e} X {}  (saved state of the fused forward)\n    w {} lambda=(A^-T w)[reach] {:e}  (reverse sweep over the pattern with c1)",
                BANDS[band[i]],
                sites_of(i).join(" "),
                vals.join("\n    "),
                q_t_v[i], i_t_v[i], i_next_v[i], qpv[i], traj_next[i],
                c1[i], c2[i], c3[i], c4[i], k[i],
                x.as_ref().map(|v| format!("{:e}", v[i])).unwrap_or_else(|| "n/a".into()),
                weights[i], lam[i],
            ));
        }
    }

    for (pi, name) in PARENTS.iter().enumerate() {
        let (gf, gr) = (&grads_f[pi], &grads_r[pi]);
        *rep.compared.get_mut(*name).unwrap() += n as u64;
        for i in 0..n {
            let (a, b) = (gf[i], gr[i]);
            let cell = &mut rep.cells.get_mut(*name).unwrap()[cell_index(band[i], clamped[i])];
            cell.compared += 1;
            if zero_set_mismatch(a, b) {
                cell.mismatches += 1;
                if clamped[i] {
                    let key = (
                        ClampSite::ALL
                            .iter()
                            .filter(|s| ledger.steps[0].active.get(s).map(|v| v[i]).unwrap_or(false))
                            .map(|s| s.name())
                            .collect::<Vec<_>>()
                            .join("+"),
                        BANDS[band[i]].to_string(),
                    );
                    *rep.clamp_groups.entry(key).or_insert(0) += 1;
                }
                let g = if a == 0.0 { b.abs() } else { a.abs() };
                if g.is_nan() || g > cell.max_mismatch_abs_g {
                    cell.max_mismatch_abs_g = g;
                }
                *rep.mismatches.get_mut(*name).unwrap() += 1;
                if rep.first_mismatches.len() < MAX_FIRST_MISMATCHES {
                    rep.first_mismatches.push(format!(
                        "step {step} parent {name} reach {i}: fused {a:e} reference {b:e}; {}",
                        ledger_row(&ledger, i)
                    ));
                }
                continue;
            }
            if a != 0.0 {
                let abs = (a - b).abs();
                let rel = abs / a.abs().max(b.abs());
                if !(rel < REL_TOL || abs < ABS_TOL) {
                    cell.outliers += 1;
                    if abs > cell.max_outlier_abs_diff {
                        cell.max_outlier_abs_diff = abs;
                    }
                    *rep.value_outliers.get_mut(*name).unwrap() += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CsrPattern, SparseAdjacency};
    use std::sync::Arc;
    use burn::backend::NdArray;

    type I = NdArray<f32>;

    fn mock(n: usize, t: usize) -> (Config, ProbeInputs<I>) {
        let dev: <I as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let mut cfg = Config::default();
        cfg.params.ddr_match = false;
        cfg.params.parameter_ranges.n = [0.01, 0.1];
        cfg.params.parameter_ranges.q_spatial = [0.1, 0.9];
        cfg.params.parameter_ranges.p_spatial = [1.0, 200.0];
        cfg.params.attribute_minimums.velocity = 0.1;
        cfg.params.attribute_minimums.depth = 0.01;
        cfg.params.attribute_minimums.discharge = 0.001;
        cfg.params.attribute_minimums.bottom_width = 0.1;
        cfg.params.attribute_minimums.slope = 0.001;
        cfg.params.defaults.insert("p_spatial".to_string(), 1.0);
        cfg.params.log_space_parameters = vec![];

        let mut dense = vec![0.0_f32; n * n];
        for i in 0..n - 1 {
            dense[(i + 1) * n + i] = 1.0;
        }
        let adj = SparseAdjacency::from_dense(n, &dense, vec![10_000.0; n], vec![1e-3; n]);
        let pattern = Arc::new(CsrPattern::from_sparse(&adj));
        let full = |v: f32| Tensor::<I, 1>::full([n], v, &dev);
        let q_prime: Vec<f32> = (0..t * n).map(|k| 5.0 + (k / n) as f32 * 0.1).collect();
        let inp = ProbeInputs {
            pattern,
            n: full(0.035),
            q_spatial: full(0.4),
            p_spatial: full(20.0),
            length: full(10_000.0),
            slope: full(1e-3),
            x_storage: full(0.3),
            q_prime: Tensor::<I, 1>::from_floats(q_prime.as_slice(), &dev).reshape([t, n]),
            pieces_per_row: None,
            q0: full(10.0),
            n_segments: n,
        };
        (cfg, inp)
    }

    #[test]
    fn five_reach_mock_has_no_zero_set_mismatch() {
        let (cfg, inp) = mock(5, 40);
        let rep = run_clamp_parity::<I>(&cfg, &inp, 30, 5);
        assert_eq!(rep.n_sampled_steps, 6);
        for p in PARENTS {
            assert_eq!(rep.compared[p], 30, "compared {p}");
            assert!(rep.compared[p] > 0, "compared {p}");
            assert_eq!(rep.mismatches[p], 0, "mismatches {p}: {:?}", rep.first_mismatches);
        }
    }

    /// Negative control: the gate counts a pair that differs in its zero set,
    /// and a non-finite value, as one mismatch each.
    #[test]
    fn gate_counts_zero_set_difference() {
        assert!(zero_set_mismatch(1e-8, 0.0));
        assert!(zero_set_mismatch(0.0, -3.0));
        assert!(zero_set_mismatch(f32::NAN, 1.0));
        assert!(!zero_set_mismatch(0.0, 0.0));
        assert!(!zero_set_mismatch(1.0, 2.0));
        let a = [0.0_f32, 1.0, 0.0, 2.0];
        let b = [0.0_f32, 1.0, 1e-9, 2.0];
        let n = a.iter().zip(&b).filter(|(x, y)| zero_set_mismatch(**x, **y)).count();
        assert_eq!(n, 1);
    }
}
