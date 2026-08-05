//! `params.subdivision` — static reach subdivision (variable Δx).
//!
//! Defaults to disabled so every existing config keeps its current behaviour.
use ddrs::adjacency::build::ConusAdjacency;
use ddrs::adjacency::subdivide::{plan_reaches, reference_celerity, subdivide, ReachPlan};
use ddrs::config::{Config, Subdivision};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A minimal valid training config, with `extra` spliced into `params:`.
/// `extra` must already be indented two spaces (it sits at `params:` depth).
fn yaml_with_params(extra: &str) -> String {
    format!(
        r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
params:
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
{extra}"#
    )
}

/// Inline YAML → tempfile → `Config::from_yaml_file`, the style used by
/// `tests/ddr_match_flag.rs`. The filename is unique per call because cargo
/// runs the tests in this file concurrently.
fn try_load_cfg(yaml: &str) -> Result<Config, String> {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "ddrs_subdivision_{}_{}.yaml",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, yaml).unwrap();
    Config::from_yaml_file(&path).map_err(|e| e.to_string())
}

fn load_cfg(yaml: &str) -> Config {
    try_load_cfg(yaml).expect("parse")
}

#[test]
fn subdivision_defaults_to_disabled() {
    let cfg = load_cfg(&yaml_with_params(""));
    assert!(!cfg.params.subdivision.enabled);
    assert_eq!(cfg.params.subdivision.max_pieces, 8);
}

#[test]
fn subdivision_rejects_max_pieces_below_one() {
    let err = try_load_cfg(&yaml_with_params(
        "  subdivision:\n    enabled: true\n    max_pieces: 0\n",
    ))
    .expect_err("must reject");
    assert!(err.to_string().contains("max_pieces"), "got: {err}");
}

#[test]
fn subdivision_enabled_loads() {
    let cfg = load_cfg(&yaml_with_params(
        "  subdivision:\n    enabled: true\n    max_pieces: 4\n",
    ));
    assert!(cfg.params.subdivision.enabled);
    assert_eq!(cfg.params.subdivision.max_pieces, 4);
}

// ---------------------------------------------------------------------------
// Task 2: reference celerity + the two-sided reach plan
// ---------------------------------------------------------------------------

fn cfg(max_pieces: usize) -> Subdivision {
    Subdivision {
        enabled: true,
        max_pieces,
        ..Default::default()
    }
}

#[test]
fn celerity_rises_with_slope_and_area() {
    let c = cfg(8);
    let lo = reference_celerity(100.0, 1e-3, &c);
    assert!(
        reference_celerity(100.0, 1e-2, &c) > lo,
        "steeper must be faster"
    );
    assert!(
        reference_celerity(10_000.0, 1e-3, &c) > lo,
        "bigger must be faster"
    );
    assert!(
        lo > 0.0 && lo < 15.0,
        "celerity {lo} outside physical range"
    );
}

#[test]
fn long_reaches_split_and_are_capped() {
    let c = cfg(4);
    let p = plan_reaches(&[200_000.0], &[1e-4], &[30.0], 3600.0, &c);
    assert_eq!(p.pieces[0], 4, "must clamp to max_pieces");
    assert_eq!(
        p.length_m[0], 200_000.0,
        "long reaches keep their true length"
    );
}

#[test]
fn short_reaches_are_length_clamped_not_split() {
    let c = cfg(8);
    let dx = reference_celerity(10_000.0, 1e-2, &c) * 3600.0;
    // The fixture must sit INSIDE `max_clamp_factor`, or that bound binds first
    // and we would be testing the bound instead of the clamp. Half of dx_target
    // is a 2x stretch, comfortably under the default 4x ceiling.
    let raw = dx / 2.0;
    let p = plan_reaches(&[raw], &[1e-2], &[10_000.0], 3600.0, &c);
    assert_eq!(p.pieces[0], 1, "short reach must not split");
    assert!(
        (p.length_m[0] - dx).abs() < 1e-3,
        "expected clamp to dx_target {dx}, got {}",
        p.length_m[0]
    );
    assert!(p.length_m[0] > raw, "clamp must lengthen, not shorten");
}

#[test]
fn min_length_fraction_zero_disables_the_clamp() {
    let mut c = cfg(8);
    c.min_length_fraction = 0.0;
    let p = plan_reaches(&[50.0], &[1e-2], &[10_000.0], 3600.0, &c);
    assert_eq!(p.length_m[0], 50.0, "clamp must be off");
}

#[test]
fn disabled_is_an_exact_no_op() {
    let mut c = cfg(8);
    c.enabled = false;
    let p = plan_reaches(
        &[200_000.0, 50.0],
        &[1e-4, 1e-2],
        &[30.0, 10_000.0],
        3600.0,
        &c,
    );
    assert_eq!(p.pieces, vec![1, 1]);
    assert_eq!(
        p.length_m,
        vec![200_000.0, 50.0],
        "lengths must be untouched"
    );
}

#[test]
fn degenerate_input_never_yields_zero_pieces_or_zero_length() {
    let c = cfg(8);
    let p = plan_reaches(&[5000.0, 0.0], &[0.0, 0.0], &[0.0, 0.0], 3600.0, &c);
    assert!(
        p.pieces.iter().all(|&v| v >= 1),
        "pieces must be >= 1, got {:?}",
        p.pieces
    );
    assert!(
        p.length_m.iter().all(|&v| v > 0.0),
        "length must be > 0 (a 0 m reach gives K = 0 and c1 = 1), got {:?}",
        p.length_m
    );
}

// ── Bounds added after Task 2 review ────────────────────────────────────────
// Two holes the original six tests did not cover.

#[test]
fn zero_length_reach_survives_min_length_fraction_zero() {
    // The combination `min_length_fraction: 0.0` + a 0 m reach previously gave
    // l_eff = 0, hence K = L/c = 0 and c1 = 1, breaking the solve. MERIT
    // contains sub-10 m reaches, so this is reachable, not hypothetical.
    let mut c = cfg(8);
    c.min_length_fraction = 0.0;
    let p = plan_reaches(&[0.0], &[1e-3], &[100.0], 3600.0, &c);
    assert!(
        p.length_m[0] >= 1.0,
        "absolute floor must apply even with the clamp disabled, got {}",
        p.length_m[0]
    );
    assert_eq!(p.pieces[0], 1);
}

#[test]
fn clamp_cannot_stretch_a_reach_beyond_max_clamp_factor() {
    // A steep headwater: dx_target is large, but a 100 m reach must not be
    // rewritten into a multi-km channel. Unbounded, measured clamp factors
    // reached p99 = 36x and max = 48,597x.
    let mut c = cfg(8);
    c.max_clamp_factor = 4.0;
    let p = plan_reaches(&[100.0], &[1e-2], &[100.0], 3600.0, &c);
    assert!(
        p.length_m[0] <= 100.0 * 4.0 + 1e-3,
        "stretched {}x, exceeding max_clamp_factor",
        p.length_m[0] / 100.0
    );
    assert!(p.length_m[0] >= 100.0, "clamp must never shorten a reach");
}

#[test]
fn reference_celerity_stays_in_a_physical_flood_wave_band() {
    let c = cfg(8);
    // Steep + large: the case that previously produced 8.9 m/s and a 32 km dx.
    let fast = reference_celerity(10_000.0, 1e-2, &c);
    assert!(
        (0.05..=5.0).contains(&fast),
        "celerity {fast} m/s outside the physical flood-wave band"
    );
    assert!(
        fast * 3600.0 <= 18_000.0 + 1.0,
        "dx_target {} m is larger than any plausible MERIT reach",
        fast * 3600.0
    );
}

// ---------------------------------------------------------------------------
// Task 3: graph expansion
// ---------------------------------------------------------------------------

/// Chain of 3 reaches: 0 -> 1 -> 2 (rows = downstream, cols = upstream,
/// so rows[k] >= cols[k]).
fn chain3() -> ConusAdjacency {
    ConusAdjacency {
        order: vec![100, 200, 300],
        rows: vec![1, 2],
        cols: vec![0, 1],
        length_m: vec![3000.0, 6000.0, 900.0],
        slope: vec![1e-3, 2e-3, 3e-3],
        dropped_comids: vec![],
    }
}

/// Explicit plan so these tests exercise expansion alone, independent of the
/// celerity heuristic in the two-sided rule above.
fn plan(pieces: Vec<u32>, length_m: Vec<f32>) -> ReachPlan {
    ReachPlan { pieces, length_m }
}

/// Row → parent position, derived from `parent_offset`. Used to collapse the
/// expanded COO back to parent space.
fn row_parents(offset: &[i32]) -> Vec<usize> {
    let mut out = Vec::new();
    for p in 0..offset.len() - 1 {
        for _ in offset[p]..offset[p + 1] {
            out.push(p);
        }
    }
    out
}

#[test]
fn expansion_preserves_total_length_and_slope() {
    let s = subdivide(&chain3(), &plan(vec![3, 2, 1], chain3().length_m.clone()));
    assert_eq!(s.length_m.len(), 6);
    for (p, &m) in [3u32, 2, 1].iter().enumerate() {
        let lo = s.parent_offset[p] as usize;
        let hi = s.parent_offset[p + 1] as usize;
        assert_eq!(hi - lo, m as usize, "parent {p} piece count");
        let total: f32 = s.length_m[lo..hi].iter().sum();
        assert!(
            (total - chain3().length_m[p]).abs() < 1e-3,
            "parent {p} length not conserved: {total}"
        );
        assert!(
            s.slope[lo..hi].iter().all(|&v| v == chain3().slope[p]),
            "slope must be inherited unchanged"
        );
    }
}

#[test]
fn expansion_stays_lower_triangular_and_topological() {
    let s = subdivide(&chain3(), &plan(vec![3, 2, 1], chain3().length_m.clone()));
    for (&r, &c) in s.rows.iter().zip(s.cols.iter()) {
        assert!(r > c, "edge {c}->{r} violates strict lower-triangular ordering");
    }
}

#[test]
fn expansion_edge_count_is_original_plus_internal_links() {
    let s = subdivide(&chain3(), &plan(vec![3, 2, 1], chain3().length_m.clone()));
    // 2 original edges + (3-1) + (2-1) + (1-1) internal = 5
    assert_eq!(s.rows.len(), 5, "rows: {:?} cols: {:?}", s.rows, s.cols);
}

#[test]
fn external_edges_land_on_parent_outlet_and_inlet() {
    let s = subdivide(&chain3(), &plan(vec![3, 2, 1], chain3().length_m.clone()));
    // parent0 rows 0..3, parent1 rows 3..5, parent2 row 5.
    // edge 0->1 becomes outlet(0)=2 -> inlet(1)=3
    assert!(
        s.rows.iter().zip(&s.cols).any(|(&r, &c)| c == 2 && r == 3),
        "missing 2->3; rows {:?} cols {:?}",
        s.rows,
        s.cols
    );
    // edge 1->2 becomes outlet(1)=4 -> inlet(2)=5
    assert!(
        s.rows.iter().zip(&s.cols).any(|(&r, &c)| c == 4 && r == 5),
        "missing 4->5; rows {:?} cols {:?}",
        s.rows,
        s.cols
    );
}

#[test]
fn all_ones_is_an_exact_identity() {
    let a = chain3();
    let s = subdivide(&a, &plan(vec![1, 1, 1], a.length_m.clone()));
    assert_eq!(s.order, a.order);
    assert_eq!(s.rows, a.rows);
    assert_eq!(s.cols, a.cols);
    assert_eq!(s.length_m, a.length_m);
    assert_eq!(s.parent_offset, vec![0, 1, 2, 3]);
}

#[test]
fn expansion_uses_the_clamped_length_not_the_raw_one() {
    let a = chain3(); // reach 2 is only 900 m
    let clamped = vec![3000.0, 6000.0, 4000.0]; // reach 2 stretched to 4 km
    let s = subdivide(&a, &plan(vec![1, 1, 1], clamped));
    assert_eq!(
        s.length_m[2], 4000.0,
        "must use ReachPlan.length_m, not ConusAdjacency.length_m"
    );
}

// ── Bounds added after Task 3 review ────────────────────────────────────────
// A 3-reach chain is too easy a fixture to trust for the invariant that the
// forward-substitution solver silently depends on: it has no junction, and its
// piece counts happen to be monotonically decreasing. These three tests use a
// branching network instead.

/// Two headwaters into a confluence, then an outlet — the shape a pure chain
/// cannot exercise. Topologically ordered, strictly lower triangular.
///
/// ```text
///   COMID 10 (p0) ─┐
///                  ├─> COMID 30 (p2) ─> COMID 40 (p3)
///   COMID 20 (p1) ─┘
/// ```
fn junction4() -> ConusAdjacency {
    ConusAdjacency {
        order: vec![10, 20, 30, 40],
        // rows = downstream, cols = upstream.
        rows: vec![2, 2, 3],
        cols: vec![0, 1, 2],
        length_m: vec![1000.0, 2000.0, 4000.0, 8000.0],
        slope: vec![5e-3, 4e-3, 3e-3, 1e-3],
        dropped_comids: vec![],
    }
}

#[test]
fn junction_expansion_stays_strictly_lower_triangular() {
    // Piece counts deliberately non-monotonic (2, 3, 1, 4) so the invariant
    // cannot pass by an accident of ordering, and include an unsplit reach.
    let a = junction4();
    let s = subdivide(&a, &plan(vec![2, 3, 1, 4], a.length_m.clone()));
    assert_eq!(s.parent_offset, vec![0, 2, 5, 6, 10]);
    assert_eq!(s.order.len(), 10);
    for (&r, &c) in s.rows.iter().zip(s.cols.iter()) {
        assert!(
            r > c,
            "edge {c}->{r} violates strict lower-triangular ordering; \
             rows {:?} cols {:?}",
            s.rows,
            s.cols
        );
    }
    // 3 external + (2-1) + (3-1) + (1-1) + (4-1) = 3 + 6 = 9 edges.
    assert_eq!(s.rows.len(), 9);
    // The confluence must survive: the inlet of parent 2 (row 5) still has two
    // distinct upstream rows — the outlets of parents 0 and 1.
    let ups: Vec<i32> = s
        .rows
        .iter()
        .zip(&s.cols)
        .filter(|(&r, _)| r == 5)
        .map(|(_, &c)| c)
        .collect();
    assert_eq!(ups, vec![s.outlet(0) as i32, s.outlet(1) as i32]);
}

#[test]
fn collapsing_subedges_to_parents_reproduces_the_parent_edge_set() {
    // The direction check that lower-triangularity CANNOT make: mapping
    // `inlet(u) -> outlet(p)` instead of `outlet(u) -> inlet(p)` is still
    // lower triangular, and reversing rows/cols on a general DAG can be too.
    // Collapsing every sub-edge back to (parent_of_row, parent_of_col) must
    // reproduce the original (rows, cols) pairs exactly, in order.
    let a = junction4();
    let s = subdivide(&a, &plan(vec![2, 3, 1, 4], a.length_m.clone()));
    let owner = row_parents(&s.parent_offset);

    let mut external: Vec<(usize, usize)> = Vec::new();
    for (&r, &c) in s.rows.iter().zip(s.cols.iter()) {
        let (pr, pc) = (owner[r as usize], owner[c as usize]);
        if pr == pc {
            // Internal chain link: must join consecutive pieces of one parent.
            assert_eq!(r, c + 1, "internal link {c}->{r} is not consecutive");
        } else {
            external.push((pr, pc));
            // And it must be anchored at the true outlet/inlet, not any
            // interior piece — otherwise part of the reach is bypassed.
            assert_eq!(c as usize, s.outlet(pc), "upstream end is not the outlet");
            assert_eq!(r as usize, s.inlet(pr), "downstream end is not the inlet");
        }
    }
    let expected: Vec<(usize, usize)> = a
        .rows
        .iter()
        .zip(&a.cols)
        .map(|(&r, &c)| (r as usize, c as usize))
        .collect();
    assert_eq!(
        external, expected,
        "collapsed edge set does not match the parent graph — the flow \
         direction was inverted somewhere"
    );
}

#[test]
fn every_subreach_row_carries_its_parents_comid() {
    let a = junction4();
    let s = subdivide(&a, &plan(vec![2, 3, 1, 4], a.length_m.clone()));
    assert_eq!(s.parent_order, a.order, "parent space must be untouched");
    for (row, &p) in row_parents(&s.parent_offset).iter().enumerate() {
        assert_eq!(
            s.order[row], a.order[p],
            "row {row} should carry COMID of parent {p}"
        );
        assert_eq!(s.slope[row], a.slope[p], "row {row} slope");
    }
    for p in 0..a.order.len() {
        assert_eq!(s.inlet(p), s.parent_offset[p] as usize);
        assert_eq!(s.outlet(p), s.parent_offset[p + 1] as usize - 1);
        assert_eq!(s.pieces(p), s.outlet(p) - s.inlet(p) + 1);
    }
}
