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
use std::sync::Arc;

use burn::backend::Autodiff;
use burn::tensor::{backend::Backend, Tensor, TensorPrimitive};

use crate::config::Config;
use crate::routing::clamp_ledger::{ClampLedger, ClampSite};
use crate::routing::courant_probe::ProbeInputs;
use crate::routing::mmc_op::{forward_chain_inner, timestep_forward_diag};
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
}

impl ClampParityReport {
    pub fn total_mismatches(&self) -> u64 {
        self.mismatches.values().sum()
    }
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
    }

    for t in 1..=steps {
        let q_prime_t: Tensor<I, 1> = q_prime.clone().slice([(t - 1)..t, 0..n]).reshape([n]);
        let mut x_eff_out = None;
        let mut leak_out = None;
        let (q_next, _saved) = forward_chain_inner::<I>(
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

    for (pi, name) in PARENTS.iter().enumerate() {
        let (gf, gr) = (&grads_f[pi], &grads_r[pi]);
        *rep.compared.get_mut(*name).unwrap() += n as u64;
        for i in 0..n {
            let (a, b) = (gf[i], gr[i]);
            let finite = a.is_finite() && b.is_finite();
            if !finite || (a == 0.0) != (b == 0.0) {
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
            assert_eq!(rep.mismatches[p], 0, "mismatches {p}: {:?}", rep.first_mismatches);
        }
    }
}
