//! End-to-end checks for reach subdivision inside the routing core.
//!
//! A reach split into `m` pieces must have its lateral inflow split `m` ways:
//! the pieces are `L/m` long and chain in series, so each carries `q'/m` and
//! the parent's outlet piece still discharges the whole reach's runoff. This
//! mirrors HEC-HMS, whose lateral term is `C4·(q_L·Δx)` with `q_L` an inflow
//! per unit length.
//!
//! ```text
//!   1 piece                    4 pieces
//!   ┌──────────────┐           ┌────┬────┬────┬────┐
//!   │      q'      │──> Q      │q'/4│q'/4│q'/4│q'/4│──> Q
//!   └──────────────┘           └────┴────┴────┴────┘
//!        L = 4 km                 each 1 km, Σq' = q'
//! ```
//!
//! Both configurations must reach the SAME steady-state outflow. If the split
//! is missing, the 4-piece network manufactures 4× the mass.

use std::collections::HashMap;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, TensorData};

use ddrs::adjacency::build::ConusAdjacency;
use ddrs::adjacency::subdivide::{subdivide, ReachPlan};
use ddrs::adjacency::zarr_write::write_conus_store_subdivided;
use ddrs::config::Config;
use ddrs::data::collate::UnionedCoo;
use ddrs::data::dataset::slice_reach_geometry;
use ddrs::data::{compress, ConusAdjacencyStore, Staid};
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;
use ddrs::training::forward::gather_params_to_subreaches;

type I = NdArray<f32>;
type AB = Autodiff<I>;

/// Total length of the single parent reach, metres. Split evenly across pieces.
const PARENT_LENGTH_M: f32 = 4000.0;
/// Long enough for a 4 km / 0.1%-slope chain at dt = 3600 s to settle.
const T: usize = 256;
/// Constant lateral inflow for the parent reach, m³/s.
const Q_PRIME: f32 = 10.0;

/// Same knobs as `tests/gauge_mass_conservation.rs::mock_cfg` — puts
/// velocity/depth in well-conditioned, non-saturated regimes for this network.
fn mock_cfg() -> Config {
    let mut cfg = Config::default();
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
    cfg
}

/// One parent reach expanded into `pieces` sub-reaches chained
/// upstream→downstream: `0 → 1 → ... → pieces-1`. Lower-triangular and
/// topologically ordered, exactly as `adjacency::subdivide` emits.
///
/// `with_parent_map` controls whether the engine is told about the expansion;
/// `false` is the "forgot to split q'" control.
fn subdivided_reach(pieces: usize, with_parent_map: bool) -> SparseAdjacency {
    let n = pieces;
    let mut dense = vec![0.0_f32; n * n];
    for i in 1..n {
        dense[i * n + (i - 1)] = 1.0; // adj[i, i-1]: piece i-1 flows into piece i
    }
    let piece_len = PARENT_LENGTH_M / pieces as f32;
    let mut adj = SparseAdjacency::from_dense(n, &dense, vec![piece_len; n], vec![0.001; n]);
    if with_parent_map {
        // A single parent owning all `pieces` rows.
        adj.parent_offset = Some(vec![0, pieces as i32]);
    }
    adj
}

/// Route to steady state and return the outlet piece's discharge (m³/s).
fn steady_state_outflow(pieces: usize, with_parent_map: bool) -> f32 {
    let row = outlet_series(pieces, with_parent_map, None);
    row[T - 1]
}

/// The outlet piece's full discharge series (length `T`). Column 0 is the
/// cold-start `Q_0` that `setup_inputs` solved, so `[0]` measures the initial
/// condition and `[T-1]` the steady state.
///
/// `divide_hotstart` overrides `MuskingumCunge::divide_hotstart_by_pieces`;
/// `None` leaves the shipped default in place.
fn outlet_series(pieces: usize, with_parent_map: bool, divide_hotstart: Option<bool>) -> Vec<f32> {
    let n = pieces;
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();

    // Constant q_prime on every sub-reach row, shape (T, N). The engine is what
    // divides it by the piece count — the caller always supplies the parent's
    // full lateral inflow, exactly as the q'-store read does.
    let q_prime: Tensor<AB, 2> =
        Tensor::from_data(TensorData::new(vec![Q_PRIME; T * n], [T, n]), &device);

    // Mid-range normalized params → n ≈ 0.055, q_spatial ≈ 0.5, p_spatial ≈ 100.
    let mk = |v: f32| -> Tensor<AB, 1> { Tensor::from_floats(vec![v; n].as_slice(), &device) };
    let mut engine = MuskingumCunge::<I>::new(mock_cfg(), device.clone());
    if let Some(d) = divide_hotstart {
        engine.divide_hotstart_by_pieces = d;
    }
    engine.setup_inputs(
        RoutingInputs {
            adjacency: subdivided_reach(pieces, with_parent_map),
            x_storage: mk(0.3),
        },
        q_prime,
        SpatialParameters {
            n: mk(0.5),
            q_spatial: mk(0.5),
            p_spatial: Some(mk(0.5)),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
        },
        false,
        None,
    );

    // (N, T) routed discharge; the parent's outlet is the LAST piece.
    let runoff: Tensor<I, 2> = engine.forward().inner();
    let all: Vec<f32> = runoff.into_data().to_vec::<f32>().expect("runoff to host");
    all[(n - 1) * T..n * T].to_vec()
}

/// A 1-reach network with constant q' must reach the SAME steady-state outflow
/// whether or not it is subdivided — the pieces split the inflow m ways but
/// chain in series, so the outlet still carries the whole reach's runoff.
#[test]
fn subdivision_conserves_mass_at_steady_state() {
    let un_split = steady_state_outflow(1, true);
    let split = steady_state_outflow(4, true);
    println!("1 piece = {un_split:.6} m3/s   4 pieces = {split:.6} m3/s   q' = {Q_PRIME}");

    assert!(
        (un_split - Q_PRIME).abs() / Q_PRIME < 1e-3,
        "control drifted: 1-piece steady state {un_split} != q' {Q_PRIME}"
    );
    assert!(
        (split - un_split).abs() / un_split < 1e-3,
        "mass not conserved: 1 piece = {un_split}, 4 pieces = {split} \
         (ratio {:.4}; a ratio near 4 means q' was not divided by the piece count)",
        split / un_split,
    );
}

/// Proves the test above discriminates: without the parent map the engine
/// cannot know the reach was split, so every piece receives the parent's full
/// q' and the outlet manufactures `pieces ×` the mass.
#[test]
fn without_the_parent_map_a_split_reach_manufactures_mass() {
    let split_unaware = steady_state_outflow(4, false);
    println!("4 pieces, no parent map = {split_unaware:.6} m3/s");
    assert!(
        (split_unaware - 4.0 * Q_PRIME).abs() / (4.0 * Q_PRIME) < 1e-3,
        "expected 4x mass ({}), got {split_unaware} — if this now conserves \
         mass the divisor is being applied without a parent map",
        4.0 * Q_PRIME,
    );
}

/// The disabled path must be a true no-op: an identity parent map (one row per
/// parent, `0..=n`) is byte-identical to no parent map at all, because
/// `pieces_per_row_divisor` returns `None` rather than a tensor of ones.
#[test]
fn identity_parent_map_is_bit_identical_to_none() {
    let n = 4;
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let run = |parent_offset: Option<Vec<i32>>| -> Vec<f32> {
        let q_prime: Tensor<AB, 2> =
            Tensor::from_data(TensorData::new(vec![Q_PRIME; T * n], [T, n]), &device);
        let mk = |v: f32| -> Tensor<AB, 1> { Tensor::from_floats(vec![v; n].as_slice(), &device) };
        let mut adj = subdivided_reach(n, false);
        adj.parent_offset = parent_offset;
        let mut engine = MuskingumCunge::<I>::new(mock_cfg(), device.clone());
        engine.setup_inputs(
            RoutingInputs { adjacency: adj, x_storage: mk(0.3) },
            q_prime,
            SpatialParameters {
                n: mk(0.5),
                q_spatial: mk(0.5),
                p_spatial: Some(mk(0.5)),
                k_d: None,
                d_gw: None,
                leakance_factor: None,
                impervious_mask: None,
            },
            false,
            None,
        );
        engine
            .forward()
            .inner()
            .into_data()
            .to_vec::<f32>()
            .expect("runoff to host")
    };

    // Here the four rows are four independent PARENTS that happen to chain.
    let identity = run(Some((0..=n as i32).collect()));
    let none = run(None);
    assert_eq!(
        identity.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        none.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "identity parent map must not perturb a single bit"
    );
}

// ---------------------------------------------------------------------------
// Per-row channel geometry under subdivision
// ---------------------------------------------------------------------------

/// Chain of 3 reaches `0 → 1 → 2` with deliberately DISTINCT lengths and
/// slopes, so a row that reads the wrong index is visible rather than lucky.
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

/// `chain3` subdivided into `pieces`, written to a zarr store and reloaded —
/// the same round trip `ddrs plan` performs, so `parent_order` / `parent_offset`
/// come back through the real reader.
fn subdivided_store(pieces: &[u32]) -> (ConusAdjacencyStore, tempfile::TempDir) {
    let a = chain3();
    let s = subdivide(
        &a,
        &ReachPlan {
            pieces: pieces.to_vec(),
            length_m: a.length_m.clone(),
        },
    );
    let dir = tempfile::tempdir().expect("tempdir");
    write_conus_store_subdivided(&s, dir.path()).expect("write");
    let store = ConusAdjacencyStore::open(dir.path()).expect("open");
    (store, dir)
}

/// Every sub-reach row must carry its OWN geometry: length `L/m` (of its
/// parent's possibly-clamped total) and its parent's slope.
///
/// This is the regression for the parent/sub-reach index mix-up: the geometry
/// arrays are in SUB-REACH space, but `ConusAdjacencyStore::index` resolves a
/// COMID to a PARENT position. Slicing with the latter gives row `i` the
/// geometry of parent-position `i`, which is only correct when the two spaces
/// coincide — i.e. everywhere except under subdivision, which is what made the
/// bug silent.
#[test]
fn subdivided_rows_get_their_own_length_and_slope() {
    let pieces = [3u32, 2, 1];
    let (store, _dir) = subdivided_store(&pieces);
    assert_eq!(store.parent_offset, vec![0, 3, 5, 6]);

    // Activate the whole network: every subdivided edge plus a gauge on the
    // outlet piece of the last parent.
    let edges: Vec<(usize, usize)> = store
        .indices_0
        .iter()
        .zip(store.indices_1.iter())
        .map(|(&r, &c)| (r as usize, c as usize))
        .collect();
    let unioned = UnionedCoo {
        edges,
        gauges: vec![(Staid::new("gauge"), store.outlet_row(2), "300".to_string())],
    };
    let compressed = compress(&unioned, &store.order, false, Some(&store.parent_offset))
        .expect("compress");
    assert_eq!(
        compressed.divide_comids.len(),
        6,
        "all six sub-reaches must be active for this fixture to be conclusive"
    );

    let (length_m, slope) = slice_reach_geometry(&store, &compressed);
    println!("length_m = {length_m:?}\nslope    = {slope:?}");

    let parent = chain3();
    for (p, &m) in pieces.iter().enumerate() {
        let lo = store.parent_offset[p] as usize;
        let hi = store.parent_offset[p + 1] as usize;
        let expect_len = parent.length_m[p] / m as f32;
        for row in lo..hi {
            assert!(
                (length_m[row] - expect_len).abs() < 1e-3,
                "row {row} (parent {p}, {m} pieces): length {} != L/m {expect_len}",
                length_m[row],
            );
            assert!(
                (slope[row] - parent.slope[p]).abs() < 1e-9,
                "row {row} (parent {p}): slope {} != parent slope {}",
                slope[row],
                parent.slope[p],
            );
        }
    }

    // Total length is conserved per parent: splitting must not create channel.
    for (p, _) in pieces.iter().enumerate() {
        let lo = store.parent_offset[p] as usize;
        let hi = store.parent_offset[p + 1] as usize;
        let total: f32 = length_m[lo..hi].iter().sum();
        assert!(
            (total - parent.length_m[p]).abs() < 1e-2,
            "parent {p}: pieces sum to {total}, parent is {}",
            parent.length_m[p],
        );
    }
}

// ---------------------------------------------------------------------------
// Parent -> sub-reach KAN parameter gather
// ---------------------------------------------------------------------------

/// Gather `parents` onto the rows described by `parent_offset`, forward only.
fn gather_for_test(parents: &[f32], parent_offset: &[i32]) -> Vec<f32> {
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let n_rows = *parent_offset.last().expect("non-empty offset") as usize;
    let mut params: HashMap<String, Tensor<I, 1>> = HashMap::new();
    params.insert("n".to_string(), Tensor::from_floats(parents, &device));
    let out = gather_params_to_subreaches(
        params,
        Some(&parent_offset.to_vec()),
        n_rows,
        &device,
    );
    out["n"].clone().into_data().into_vec().expect("to host")
}

/// `d(Σ gathered)/d(parent p)`. Proves `select`'s backward is a scatter-add.
fn gather_grad_for_test(parents: &[f32], parent_offset: &[i32]) -> Vec<f32> {
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let n_rows = *parent_offset.last().expect("non-empty offset") as usize;
    let leaf: Tensor<AB, 1> = Tensor::from_floats(parents, &device).require_grad();
    let mut params: HashMap<String, Tensor<AB, 1>> = HashMap::new();
    params.insert("n".to_string(), leaf.clone());
    let out = gather_params_to_subreaches(
        params,
        Some(&parent_offset.to_vec()),
        n_rows,
        &device,
    );
    let grads = out["n"].clone().sum().backward();
    leaf.grad(&grads)
        .expect("parent leaf must receive a gradient")
        .into_data()
        .into_vec()
        .expect("to host")
}

/// Sub-reaches inherit their parent's hydraulics verbatim — MERIT carries no
/// within-reach variation, so there is nothing better to give them.
#[test]
fn every_piece_inherits_its_parents_parameters() {
    let gathered = gather_for_test(&[0.02, 0.05, 0.10], &[0, 3, 5, 6]);
    println!("gathered = {gathered:?}");
    assert_eq!(gathered, vec![0.02, 0.02, 0.02, 0.05, 0.05, 0.10]);
}

/// The gradient half of the shared-parameter contract: a parent that feeds `m`
/// pieces must receive the SUM of their gradients, not one piece's. Anything
/// else (a broadcast, or a gather that forgets the tape) makes long reaches
/// learn at a different rate than the objective actually implies.
#[test]
fn gradient_sums_back_to_the_parent() {
    let g = gather_grad_for_test(&[0.02, 0.05, 0.10], &[0, 3, 5, 6]);
    println!("d(sum)/d(parent) = {g:?}");
    assert_eq!(
        g,
        vec![3.0, 2.0, 1.0],
        "scatter-add must sum piece gradients (expected the piece counts)"
    );
}

/// The subdivision-off path must not perturb a value.
///
/// Honest about its reach: this is a VALUE-level check. It would still pass if
/// the `n_parent == n_rows` short-circuit were deleted and the gather ran with
/// an identity index (that property is structural — `gather_params_to_subreaches`
/// returns the map itself, recording nothing on the tape). What it does catch is
/// a mis-built row→parent index, which in the identity case would reorder or
/// truncate the parameters.
#[test]
fn identity_parent_offset_leaves_values_untouched() {
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let vals = [0.02_f32, 0.05, 0.10];
    let mut params: HashMap<String, Tensor<I, 1>> = HashMap::new();
    params.insert("n".to_string(), Tensor::from_floats(vals.as_slice(), &device));

    for offset in [None, Some(vec![0i32, 1, 2, 3])] {
        let out = gather_params_to_subreaches(params.clone(), offset.as_ref(), 3, &device);
        let got: Vec<f32> = out["n"].clone().into_data().into_vec().expect("to host");
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "identity map must not perturb a single bit (offset {offset:?})"
        );
    }
}

// ── cold start under subdivision ─────────────────────────────────────────────
//
// `setup_inputs` cold-starts with `(I − N)·Q_0 = q'_0`. The `q'/m` split lives
// in `forward`, so without `divide_hotstart_by_pieces` the cold start feeds the
// UNDIVIDED `q'_0` into a chain of `m` pieces and parent `p`'s outlet begins at
// `m_p ×` its true steady state.
//
// Measured on the real network (2026-08-05, `probe_courant --max-pieces 8`,
// 1,841 CONUS gauges / 184,676 rows): the undivided start put 2.94× the correct
// total discharge into the network and needed **221 hourly steps to decay
// below a 10 % difference** (282 below 5 %, still >1 % at 500), against a
// configured `warmup` of 5 days = 120 steps — at which point it was still
// 41.7 % off. Hence the division is on by default.

/// The cold start must be mass-consistent: subdividing a reach must not change
/// the discharge the solver starts that reach's outlet at.
#[test]
fn divided_hotstart_gives_a_subdivided_reach_the_same_initial_condition() {
    let un_split = outlet_series(1, true, None)[0];
    let split = outlet_series(4, true, None)[0];
    println!("Q_0 outlet: 1 piece = {un_split:.6}, 4 pieces = {split:.6}");
    assert!(
        (split - un_split).abs() / un_split < 1e-3,
        "cold start not mass-consistent: 1 piece = {un_split}, 4 pieces = {split} \
         (ratio {:.4}; a ratio near 4 means the hot-start q'_0 was not divided)",
        split / un_split
    );
}

/// The discriminating control: with the division off, the cold start really is
/// inflated `m×`. This is the state the real-network wash-out was measured on.
#[test]
fn undivided_hotstart_inflates_the_initial_condition_by_the_piece_count() {
    let un_split = outlet_series(1, true, Some(false))[0];
    let split = outlet_series(4, true, Some(false))[0];
    println!("Q_0 outlet (undivided hot start): 1 piece = {un_split:.6}, 4 pieces = {split:.6}");
    assert!(
        (split / un_split - 4.0).abs() < 1e-2,
        "expected a 4x inflated cold start, got ratio {:.4}",
        split / un_split
    );
}

/// Steady state is reached either way — the cold start only sets how long the
/// spin-up takes — so the fix must not move the converged answer.
#[test]
fn hotstart_division_does_not_move_the_steady_state() {
    let divided = outlet_series(4, true, Some(true))[T - 1];
    let undivided = outlet_series(4, true, Some(false))[T - 1];
    println!("steady state: divided = {divided:.6}, undivided = {undivided:.6}");
    assert!(
        (divided - undivided).abs() / divided < 1e-3,
        "the cold start changed the steady state: {divided} vs {undivided}"
    );
}

/// An un-subdivided network has no divisor, so the default must be an EXACT
/// no-op there — this is what keeps `params.subdivision.enabled: false`
/// byte-identical (and `compare_ddr_sandbox` an ABSOLUTE MATCH).
#[test]
fn hotstart_division_is_a_no_op_without_subdivision() {
    assert_eq!(
        outlet_series(1, true, Some(true)),
        outlet_series(1, true, Some(false)),
        "the hot-start divisor must not touch an un-subdivided network"
    );
}

// ── the non-negativity window ────────────────────────────────────────────────

/// **Why "Cr ≈ 1 ⇒ non-negative coefficients" does not survive contact with
/// MERIT.** `c1 ≥ 0 ⟺ Cr ≥ 2X` and `c3 ≥ 0 ⟺ Cr ≤ 2(1−X)`, so BOTH hold only
/// inside a window of width `2(1−2X)`. That width collapses as `X → 0.5`, and
/// on the real network the Cunge `X` sits at a median of 0.492–0.497 — a window
/// **1.3–3.1 % wide** (measured 2026-08-05, `probe_courant`, 1,841 gauges:
/// `2(1−2X)` p50 = 0.0134 un-split, 0.0310 at `max_pieces: 8`).
///
/// A build-time piece count sets `Δx` from a *reference* flow, while `Cr`
/// tracks the *routed* celerity, which varies several-fold within a single
/// storm. Landing inside a 1–3 % window is therefore not achievable by
/// subdivision, whatever the cap. Measured `frac c1 ≥ 0 AND c3 ≥ 0` on CONUS:
/// 3.1 % un-split → 0.9 % at cap 8.
#[test]
fn both_coefficients_are_non_negative_only_inside_a_window_that_collapses_at_x_half() {
    // Muskingum coefficients, mirroring `mmc_op.rs:1077-1080` with Cr = dt/K.
    fn coeffs(cr: f32, x: f32) -> (f32, f32) {
        let k = 1.0 / cr; // dt = 1 in these units
        let denom = 2.0 * k * (1.0 - x) + 1.0;
        ((1.0 - 2.0 * k * x) / denom, (2.0 * k * (1.0 - x) - 1.0) / denom)
    }

    for &x in &[0.30_f32, 0.45, 0.4966] {
        let lo = 2.0 * x;
        let hi = 2.0 * (1.0 - x);
        let width = hi - lo;
        assert!(
            (width - 2.0 * (1.0 - 2.0 * x)).abs() < 1e-6,
            "window width must be 2(1-2X)"
        );
        // Inside the window both are non-negative...
        let (c1, c3) = coeffs(0.5 * (lo + hi), x);
        assert!(c1 >= 0.0 && c3 >= 0.0, "X={x}: mid-window gave c1={c1}, c3={c3}");
        // ...and stepping outside it, either side, breaks one of them.
        let (c1_lo, _) = coeffs(lo * 0.98, x);
        assert!(c1_lo < 0.0, "X={x}: Cr below 2X must give c1 < 0, got {c1_lo}");
        let (_, c3_hi) = coeffs(hi * 1.02, x);
        assert!(c3_hi < 0.0, "X={x}: Cr above 2(1-X) must give c3 < 0, got {c3_hi}");
    }

    // The measured CONUS median X leaves a window barely 1 % wide.
    let x = 0.4966_f32;
    assert!(
        2.0 * (1.0 - 2.0 * x) < 0.02,
        "the CONUS-median non-negativity window must be under 2% wide"
    );
}
