//! Static reach subdivision so Cr = c*dt/dx lands near 1.
//!
//! HEC-HMS picks the space step as `dx = c*dt` (Technical Reference Manual,
//! Muskingum-Cunge Model). ddrs historically used the full MERIT reach length,
//! giving median Cr = 0.226 — reaches ~4.4x too long.
//!
//! This module is a pure preprocessing step: plain `Vec`s in, plain `Vec`s out.
//! No BURN, no I/O, no dependence on training state.

use crate::adjacency::build::ConusAdjacency;
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
    // Clamped to a physical FLOOD-WAVE celerity band, deliberately tighter than
    // the solver's own [0.01, 15.0] velocity clamp. The upper bound matters:
    // `dx_target = c * dt`, so at dt = 3600 s a celerity of 8.9 m/s (which this
    // relation reaches at slope 1e-2) implies a 32 km space step, and every
    // shorter reach would be stretched to it. 5 m/s caps `dx_target` at 18 km,
    // and `max_clamp_factor` bounds the residual distortion.
    (v * (5.0 / 3.0)).clamp(0.05, 5.0)
}

/// Absolute lower bound on a reach length, in metres. Independent of the clamp
/// so that `min_length_fraction: 0.0` on a zero-length reach still cannot
/// produce `K = L/c = 0`, which would make `c1 = 1` and break the solve.
/// MERIT contains sub-10 m reaches, so this is reachable in practice.
const ABS_MIN_LENGTH_M: f32 = 1.0;

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
        let l_raw = l.max(0.0);
        let want = dx_target * cfg.min_length_fraction;
        // Never stretch a reach more than `max_clamp_factor` times its true
        // length. A zero-length reach has no meaningful factor, so it takes the
        // target directly — it is degenerate geometry either way.
        let ceiling = if l_raw > 0.0 {
            l_raw * cfg.max_clamp_factor
        } else {
            f32::INFINITY
        };
        // ABS_MIN_LENGTH_M is applied LAST so it survives min_length_fraction: 0.0.
        let l_eff = l_raw.max(want.min(ceiling)).max(ABS_MIN_LENGTH_M);
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

/// Cost accounting for a [`ReachPlan`], measured against the raw fabric lengths.
///
/// The length clamp is the price of the short-reach branch: a reach modelled
/// `f×` longer than reality has an `f×` longer travel time, which is a real
/// physical distortion traded for numerical stability. `max_clamp_factor`
/// bounds `f`, and `n_at_clamp_ceiling` counts the reaches pinned at that bound
/// — those stay over-Courant by design and will still produce negative
/// coefficients.
#[derive(Debug, Clone)]
pub struct PlanStats {
    /// Parent reaches considered.
    pub n: usize,
    /// `Σ pieces` — the sub-reach count the solver will see.
    pub sum_pieces: u64,
    /// Reaches with `m > 1`.
    pub n_split: usize,
    /// Reaches whose length was stretched (`length_m > raw + 1e-3`).
    pub n_clamped: usize,
    /// Reaches pinned at `raw * max_clamp_factor` — the clamp wanted more.
    pub n_at_clamp_ceiling: usize,
    /// Clamp factor `length_m / raw` over the CLAMPED reaches only: p50/p95/p99/max.
    pub clamp_factor_p50: f32,
    pub clamp_factor_p95: f32,
    pub clamp_factor_p99: f32,
    pub clamp_factor_max: f32,
    /// Total network channel length (m) before and after the clamp.
    pub total_length_before_m: f64,
    pub total_length_after_m: f64,
    /// Histogram of piece counts, `pieces_hist[m]` for `m` in `0..=max_pieces`.
    pub pieces_hist: Vec<usize>,
}

impl PlanStats {
    /// Fractional inflation of total channel length, e.g. `0.171` for +17.1 %.
    pub fn length_inflation(&self) -> f64 {
        self.total_length_after_m / self.total_length_before_m - 1.0
    }
}

/// Measure a [`ReachPlan`] against the RAW fabric lengths it was built from.
///
/// `raw_length_m` must be `ConusAdjacency::length_m` — the un-clamped input to
/// [`plan_reaches`] — not `plan.length_m`.
pub fn plan_stats(raw_length_m: &[f32], plan: &ReachPlan, cfg: &Subdivision) -> PlanStats {
    let n = raw_length_m.len();
    let mut factors: Vec<f32> = Vec::new();
    let mut n_at_ceiling = 0usize;
    let mut before = 0f64;
    let mut after = 0f64;
    let mut hist = vec![0usize; cfg.max_pieces.max(1) + 1];
    let mut sum_pieces = 0u64;
    let mut n_split = 0usize;

    for i in 0..n {
        let raw = raw_length_m[i].max(0.0);
        let eff = plan.length_m[i];
        before += raw as f64;
        after += eff as f64;
        if eff > raw + 1e-3 {
            let f = if raw > 0.0 { eff / raw } else { f32::INFINITY };
            factors.push(f);
            // Pinned at the ceiling: the clamp wanted `dx_target *
            // min_length_fraction` but got `raw * max_clamp_factor`.
            if raw > 0.0 && (f - cfg.max_clamp_factor).abs() <= 1e-3 * cfg.max_clamp_factor {
                n_at_ceiling += 1;
            }
        }
        let m = plan.pieces[i] as usize;
        sum_pieces += m as u64;
        if m > 1 {
            n_split += 1;
        }
        if m < hist.len() {
            hist[m] += 1;
        }
    }

    factors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| -> f32 {
        if factors.is_empty() {
            return f32::NAN;
        }
        factors[(((factors.len() - 1) as f64) * p).round() as usize]
    };

    PlanStats {
        n,
        sum_pieces,
        n_split,
        n_clamped: factors.len(),
        n_at_clamp_ceiling: n_at_ceiling,
        clamp_factor_p50: q(0.50),
        clamp_factor_p95: q(0.95),
        clamp_factor_p99: q(0.99),
        clamp_factor_max: factors.last().copied().unwrap_or(f32::NAN),
        total_length_before_m: before,
        total_length_after_m: after,
        pieces_hist: hist,
    }
}

/// The expanded network: one row per sub-reach, plus the map back to parents.
///
/// Two index spaces coexist. **Parent space** (`parent_order`, length `N`) is
/// where COMID lookups live — `order` gains duplicates after expansion, so a
/// COMID→row lookup on it would be ambiguous. **Sub-reach space** (`order`,
/// `rows`, `cols`, `length_m`, `slope`, length `N' = Σ m_p`) is what the solver
/// sees. Parent `p` owns the contiguous rows `parent_offset[p]..parent_offset[p+1]`,
/// ordered upstream→downstream, so its inlet is the first and its outlet the last.
///
/// ```text
///    BEFORE                       AFTER (m = 3)
///    U ──> P ──> D                U₂ ──> P₀ ──> P₁ ──> P₂ ──> D₀
///                                        └─ q'/3   q'/3   q'/3
///    len(P) = L                   len(Pᵢ) = L/3,  slope unchanged
/// ```
#[derive(Debug, Clone)]
pub struct SubdividedAdjacency {
    /// COMID per sub-reach row — the parent's COMID, repeated `m_p` times.
    pub order: Vec<i32>,
    /// Original COMID topological order, length `N`. COMID→position indexes
    /// (`IdIndex<Comid>`) must be built from THIS, not from `order`.
    pub parent_order: Vec<i32>,
    /// Length `N + 1`. Rows `[parent_offset[p], parent_offset[p + 1])` belong to
    /// parent `p`, ordered upstream→downstream.
    pub parent_offset: Vec<i32>,
    /// COO `indices_0` — downstream sub-reach row.
    pub rows: Vec<i32>,
    /// COO `indices_1` — upstream sub-reach row.
    pub cols: Vec<i32>,
    /// Per-piece length in metres: the parent's (possibly clamped) length / `m`.
    pub length_m: Vec<f32>,
    /// Per-piece slope — inherited unchanged from the parent.
    pub slope: Vec<f32>,
}

impl SubdividedAdjacency {
    /// First (most upstream) sub-reach row of `parent`.
    #[inline]
    pub fn inlet(&self, parent: usize) -> usize {
        self.parent_offset[parent] as usize
    }

    /// Last (most downstream) sub-reach row of `parent`. A gauge on this reach
    /// must be read here: any earlier piece omits the downstream fraction of the
    /// reach's own lateral inflow.
    #[inline]
    pub fn outlet(&self, parent: usize) -> usize {
        self.parent_offset[parent + 1] as usize - 1
    }

    /// Number of sub-reaches owned by `parent`.
    #[inline]
    pub fn pieces(&self, parent: usize) -> usize {
        (self.parent_offset[parent + 1] - self.parent_offset[parent]) as usize
    }
}

/// Expand `adj` into the sub-reach graph described by `plan`.
///
/// **Topology rules.** Parent `p` owns rows `[off[p], off[p+1])` ordered
/// upstream→downstream; consecutive pieces are joined by internal chain edges.
/// An external edge `u -> p` (`u` upstream of `p`) becomes
/// `outlet(u) -> inlet(p)`. Each piece gets `plan.length_m[p] / m` — the
/// **clamped** length from `plan_reaches`, not `adj.length_m[p]`. Slope is
/// inherited unchanged.
///
/// **Why the result is still lower-triangular** (routing invariant 3, which the
/// forward-substitution solver in `src/sparse/` depends on and which fails
/// silently rather than loudly):
/// - Internal edges run `base + k - 1 -> base + k`, strictly increasing.
/// - External edges: the input satisfies `r > c` (`build.rs` rejects `r < c`,
///   and the real CONUS COO carries no `r == c`), so `r >= c + 1` and therefore
///   `inlet(r) = off[r] >= off[c + 1] > off[c + 1] - 1 = outlet(c)`.
///
/// Because `rows`/`cols` are `downstream`/`upstream` (`zarr.rs:39-42`,
/// `build.rs:231-232`), reversing the mapping to `inlet(u) -> outlet(p)` would
/// still be lower-triangular and would NOT be caught by that argument — the
/// direction is pinned by tests instead.
pub fn subdivide(adj: &ConusAdjacency, plan: &ReachPlan) -> SubdividedAdjacency {
    let n = adj.order.len();
    assert_eq!(plan.pieces.len(), n, "pieces must be one per parent reach");
    assert_eq!(plan.length_m.len(), n, "lengths must be one per parent reach");

    let mut parent_offset = Vec::with_capacity(n + 1);
    let mut acc: i32 = 0;
    parent_offset.push(0);
    for &m in &plan.pieces {
        acc += m.max(1) as i32;
        parent_offset.push(acc);
    }
    let n_sub = acc as usize;

    let mut order = Vec::with_capacity(n_sub);
    let mut length_m = Vec::with_capacity(n_sub);
    let mut slope = Vec::with_capacity(n_sub);
    let mut rows = Vec::with_capacity(adj.rows.len() + n_sub - n);
    let mut cols = Vec::with_capacity(adj.cols.len() + n_sub - n);

    for p in 0..n {
        let m = plan.pieces[p].max(1) as usize;
        let base = parent_offset[p] as usize;
        // plan.length_m, NOT adj.length_m: short reaches were clamped UP in the
        // two-sided rule above, and the expansion must honour that.
        let piece_len = plan.length_m[p] / m as f32;
        for k in 0..m {
            order.push(adj.order[p]);
            length_m.push(piece_len);
            slope.push(adj.slope[p]);
            if k > 0 {
                // Internal chain link: piece k-1 flows into piece k.
                cols.push((base + k - 1) as i32);
                rows.push((base + k) as i32);
            }
        }
    }

    // External edges: upstream parent's OUTLET -> downstream parent's INLET.
    for (&r, &c) in adj.rows.iter().zip(adj.cols.iter()) {
        // A self-edge (r == c) would map to outlet(p) -> inlet(p), i.e.
        // off[p]+m-1 -> off[p], which is UPPER triangular and would break the
        // forward-substitution invariant *silently*. `build.rs:236-240` only
        // rejects r < c, so r == c is permitted by that check — but the real
        // CONUS COO (346,321 reaches / 338,814 edges) has zero of them, the
        // diagonal being synthesized later by `CsrPattern::from_sparse`. This is
        // therefore an assertion documenting that fact, not a filter.
        assert!(r != c, "self-edge on parent {r}: subdivision cannot expand it");
        let up_outlet = parent_offset[c as usize + 1] - 1;
        let down_inlet = parent_offset[r as usize];
        cols.push(up_outlet);
        rows.push(down_inlet);
    }

    SubdividedAdjacency {
        order,
        parent_order: adj.order.clone(),
        parent_offset,
        rows,
        cols,
        length_m,
        slope,
    }
}
