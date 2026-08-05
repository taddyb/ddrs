//! Static reach subdivision so Cr = c*dt/dx lands near 1.
//!
//! HEC-HMS picks the space step as `dx = c*dt` (Technical Reference Manual,
//! Muskingum-Cunge Model). ddrs historically used the full MERIT reach length,
//! giving median Cr = 0.226 — reaches ~4.4x too long.
//!
//! This module is a pure preprocessing step: plain `Vec`s in, plain `Vec`s out.
//! No BURN, no I/O, no dependence on training state.

use crate::config::Subdivision;

/// Reference celerity (m/s) used ONLY to choose the piece count.
///
/// Mirrors the solver's S15/S17 chain with the wide-channel ratio c = (5/3)*v
/// instead of the exact trapezoidal beta: this only sets `m`, the cap dominates
/// the result, and beta needs the learned p_spatial/q_spatial.
pub fn reference_celerity(uparea_km2: f32, slope: f32, cfg: &Subdivision) -> f32 {
    // Slope floor mirrors `attribute_minimums.slope` (mmc.rs:208).
    let s = slope.max(1e-3);
    let q_ref = cfg.reference_discharge_coefficient
        * uparea_km2.max(0.0).powf(cfg.reference_discharge_exponent);
    // Hydraulic radius via a wide-channel regime relation; the exponent 0.4 is
    // the Leopold & Maddock downstream depth exponent.
    let r = (q_ref.max(1e-3)).powf(0.4).max(0.01);
    let v = (1.0 / cfg.reference_n) * r.powf(2.0 / 3.0) * s.sqrt();
    (v * (5.0 / 3.0)).clamp(0.01, 15.0)
}

/// Both sides of the two-sided rule. `pieces[i]` is how many sub-reaches parent
/// `i` becomes; `length_m[i]` is its (possibly clamped) total length, which
/// Task 3's expansion then divides by `pieces[i]`.
pub struct ReachPlan {
    /// Piece count per parent reach; always `>= 1`.
    pub pieces: Vec<u32>,
    /// Total (possibly clamped-up) length per parent reach, in metres; always
    /// `> 0` — a zero-length reach would give `K = 0` and `c1 = 1`.
    pub length_m: Vec<f32>,
}

/// The two-sided rule. Long reaches split; short reaches have their length
/// clamped UP to `dx_target`. Merging short reaches is deliberately not done:
/// a short reach can carry two upstream tributaries or have a parallel
/// tributary joining below it, so collapsing it would destroy junctions.
pub fn plan_reaches(
    length_m: &[f32],
    slope: &[f32],
    uparea_km2: &[f32],
    dt_seconds: f32,
    cfg: &Subdivision,
) -> ReachPlan {
    if !cfg.enabled {
        return ReachPlan {
            pieces: vec![1; length_m.len()],
            length_m: length_m.to_vec(),
        };
    }
    let mut pieces = Vec::with_capacity(length_m.len());
    let mut lengths = Vec::with_capacity(length_m.len());
    for ((&l, &s), &a) in length_m.iter().zip(slope).zip(uparea_km2) {
        let dx_target = reference_celerity(a, s, cfg) * dt_seconds;
        // Short reach: stretch it so Cr ~ 1 at the reference flow. This is a
        // STATIC constant, unlike the runtime K floor in `enforce_positivity`
        // — no gradient path, so it cannot pull `n` toward its bound.
        let l_eff = (l.max(0.0)).max(dx_target * cfg.min_length_fraction);
        // Long reach: split.
        let m = ((l_eff / dx_target).ceil().max(1.0) as u32).min(cfg.max_pieces as u32);
        pieces.push(m);
        lengths.push(l_eff);
    }
    ReachPlan {
        pieces,
        length_m: lengths,
    }
}
