//! Courant / Muskingum-window diagnostic for the S18'/S19' positivity clamp.
//!
//! Measures — on the REAL routing chain, not a re-derivation — what
//! `params.enforce_positivity` actually does to a CONUS mini-batch:
//!
//!   * `negative solves before clamp` (the atomic counters in `mmc_op`),
//!   * the Courant number `Cr = dt / K`, before and after the K floor,
//!   * the effective Muskingum `X` the chain used (`x_eff_out`),
//!   * `k_musk / k_raw`, i.e. how much the floor inflates travel time,
//!   * `min c1` / `min c3`, the quantities the clamp claims to keep >= 0.
//!
//! It drives [`forward_chain_inner`] directly in the same loop shape as
//! `MuskingumCunge::forward` (q_next fed back as q_t), so every number below
//! comes out of the identical kernel sequence training runs. Nothing here
//! touches the tape or the numerics.

use std::sync::Arc;

use burn::tensor::{backend::Backend, Tensor, TensorPrimitive};

use crate::config::Config;
use crate::routing::mmc_op::{
    forward_chain_inner, forward_saved_idx, negative_solve_stats, reset_negative_solve_stats,
};
use crate::sparse::CsrPattern;

/// Inner-backend snapshot of everything `forward_chain_inner` consumes.
/// Produced by [`crate::routing::MuskingumCunge::probe_inputs`] after
/// `setup_inputs` (so the hot-start `q0` and the CSR pattern are the real
/// ones, not a re-derivation).
pub struct ProbeInputs<I: Backend> {
    pub pattern: Arc<CsrPattern>,
    pub n: Tensor<I, 1>,
    pub q_spatial: Tensor<I, 1>,
    pub p_spatial: Tensor<I, 1>,
    pub length: Tensor<I, 1>,
    pub slope: Tensor<I, 1>,
    pub x_storage: Tensor<I, 1>,
    /// `(T_hours, N)` hourly forcing, unclamped (the probe applies S0's clamp).
    pub q_prime: Tensor<I, 2>,
    /// Per-row parent piece count, or `None` when the network is un-subdivided.
    /// The probe applies it exactly as `MuskingumCunge::forward` does — AFTER
    /// the `discharge` clamp — so a subdivided network is routed with the same
    /// `q'/m` lateral inflow training would use, not `m×` too much water.
    pub pieces_per_row: Option<Tensor<I, 1>>,
    /// Window-start discharge as `setup_inputs` left it.
    pub q0: Tensor<I, 1>,
    pub n_segments: usize,
}

/// Everything the probe measured. Sampled vectors hold one entry per
/// (reach, sampled timestep); the solve counters are exact over ALL timesteps.
pub struct CourantReport {
    pub n_reaches: usize,
    pub n_steps: usize,
    pub n_sampled_steps: usize,
    /// Exact, over every routed timestep.
    pub neg_solves: u64,
    pub total_solves: u64,
    /// `dt / k_raw` — the Courant number the physics asks for.
    pub cr_raw: Vec<f32>,
    /// `dt / k_musk` — after the S18' floor (identical to `cr_raw` when off).
    pub cr: Vec<f32>,
    /// The `x_eff` the chain fed into c1..c4.
    pub x_eff: Vec<f32>,
    /// `x_cunge`, the S19 Cunge X BEFORE the S19' stability cap, recomputed
    /// from the same saved `top_width`/`celerity` the chain used. Equals
    /// `x_eff` when the clamp is off.
    pub x_cunge: Vec<f32>,
    /// `k_musk / k_raw` — 1.0 wherever the S18' floor did not bite.
    pub k_ratio: Vec<f32>,
    pub c1: Vec<f32>,
    pub c3: Vec<f32>,
    /// Total network discharge `Σ_i q_t[i]` at each of the first
    /// `trace_steps` timesteps (index 0 = the hot-start `Q_0` itself).
    /// Used to measure how long an inflated cold start takes to wash out.
    pub trace_total_q: Vec<f64>,
}

fn to_vec<I: Backend>(p: I::FloatTensorPrimitive) -> Vec<f32> {
    Tensor::<I, 1>::from_primitive(TensorPrimitive::Float(p))
        .into_data()
        .to_vec::<f32>()
        .expect("f32 tensor")
}

/// Route `n_steps` timesteps, harvesting diagnostics every `sample_every`
/// steps. `cfg` decides `enforce_positivity` — pass two configs to compare.
pub fn run_courant_probe<I: Backend + 'static>(
    cfg: &Config,
    inp: &ProbeInputs<I>,
    n_steps: usize,
    sample_every: usize,
    trace_steps: usize,
) -> CourantReport
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let dt = crate::routing::mmc::DT_SECONDS;
    let n = inp.n_segments;
    let discharge_lb = cfg.params.attribute_minimums.discharge;

    let t_avail = inp.q_prime.dims()[0];
    let steps = n_steps.min(t_avail.saturating_sub(1));
    // Mirrors `MuskingumCunge::forward`: one clamp on the whole forcing block,
    // THEN the subdivision divisor (dividing first would floor each of the `m`
    // pieces independently and create mass), and the initial state clamped once.
    let q_prime = inp.q_prime.clone().clamp_min(discharge_lb);
    let q_prime = match inp.pieces_per_row.as_ref() {
        Some(d) => q_prime / d.clone().unsqueeze_dim::<2>(0),
        None => q_prime,
    };
    let mut q_t = inp.q0.clone().clamp_min(discharge_lb);

    let length: Vec<f32> = inp.length.clone().into_data().to_vec::<f32>().expect("f32");
    let slope: Vec<f32> = inp.slope.clone().into_data().to_vec::<f32>().expect("f32");

    reset_negative_solve_stats();

    let mut rep = CourantReport {
        n_reaches: n,
        n_steps: steps,
        n_sampled_steps: 0,
        neg_solves: 0,
        total_solves: 0,
        cr_raw: Vec::new(),
        cr: Vec::new(),
        x_eff: Vec::new(),
        x_cunge: Vec::new(),
        k_ratio: Vec::new(),
        c1: Vec::new(),
        c3: Vec::new(),
        trace_total_q: Vec::new(),
    };

    let total_q = |q: &Tensor<I, 1>| -> f64 {
        q.clone()
            .into_data()
            .to_vec::<f32>()
            .expect("f32")
            .iter()
            .map(|&v| v as f64)
            .sum()
    };
    if trace_steps > 0 {
        rep.trace_total_q.push(total_q(&q_t));
    }

    for t in 1..=steps {
        let q_prime_t = q_prime
            .clone()
            .slice([(t - 1)..t, 0..n])
            .reshape([n]);
        let mut x_eff_out: Option<I::FloatTensorPrimitive> = None;
        let mut leak_out = None;
        let (q_next, saved) = forward_chain_inner::<I>(
            cfg,
            &inp.pattern,
            inp.n.clone(),
            inp.q_spatial.clone(),
            inp.p_spatial.clone(),
            q_t.clone(),
            q_prime_t,
            inp.length.clone(),
            inp.slope.clone(),
            inp.x_storage.clone(),
            None,
            &mut leak_out,
            &mut x_eff_out,
            /* track_neg */ true,
        );

        if t % sample_every == 0 {
            let k = to_vec::<I>(saved[forward_saved_idx::K_MUSKINGUM].clone());
            let cel = to_vec::<I>(saved[forward_saved_idx::CELERITY].clone());
            let c1 = to_vec::<I>(saved[forward_saved_idx::C1].clone());
            let c3 = to_vec::<I>(saved[forward_saved_idx::C3].clone());
            let x = to_vec::<I>(x_eff_out.clone().expect("x_eff always written"));
            let tw = to_vec::<I>(saved[forward_saved_idx::TOP_WIDTH].clone());
            let qt = q_t.clone().into_data().to_vec::<f32>().expect("f32");
            for i in 0..n {
                let k_raw = length[i] / cel[i];
                rep.cr_raw.push(dt / k_raw);
                rep.cr.push(dt / k[i]);
                rep.k_ratio.push(k[i] / k_raw);
                rep.x_eff.push(x[i]);
                // Mirrors S19 exactly (`forward_chain_inner`).
                let w = qt[i] / (tw[i] * slope[i] * cel[i] * length[i] + 1e-12);
                rep.x_cunge.push((0.5 * (1.0 - w)).clamp(0.0, 0.5));
                rep.c1.push(c1[i]);
                rep.c3.push(c3[i]);
            }
            rep.n_sampled_steps += 1;
        }

        q_t = Tensor::from_primitive(TensorPrimitive::Float(q_next));
        if t < trace_steps {
            rep.trace_total_q.push(total_q(&q_t));
        }
    }

    let (neg, total) = negative_solve_stats();
    rep.neg_solves = neg;
    rep.total_solves = total;
    rep
}
