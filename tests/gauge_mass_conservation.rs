//! End-to-end mass-conservation check on the per-gauge prediction path.
//!
//! A USGS gauge measures ALL drainage area above it, and we do not know where
//! along its reach the gauge physically sits. The Muskingum-Cunge solve at the
//! gauge's own reach already accumulates every upstream contribution PLUS that
//! reach's own lateral inflow (mass conservation), so the gauge's extracted
//! prediction must equal the discharge at the gauge reach — which at steady
//! state equals the sum of `q_prime` over every reach in the subgraph.
//!
//! This is the test that would have caught the `outflow_idx` defect:
//! `collate::compress` returned the gauge's UPSTREAM neighbours instead of the
//! gauge reach itself, silently dropping the gauge reach's own local runoff
//! from every prediction. Real-world symptom: gauge `01457000` (366.8 km²,
//! whose own reach is 250.1 km² = 68% of the basin) predicted 1.58 m³/s
//! against an observed 7.60 and a summed-Q' baseline of 7.38 — a constant
//! 0.215× suppression across 15 straight eval years.
//!
//! The defect is preserved (DDR parity) under `params.ddr_match: true` and
//! corrected under `false`, so both directions are asserted here.
//!
//! Network (mirrors 01457000's topology exactly): two headwaters draining into
//! the gauge reach.
//!
//!     0 ──┐
//!         ├──> 2  (gauge)
//!     1 ──┘
//!
//! `ddr_match: true` sums reaches {0, 1} — 2/3 of the mass. `ddr_match: false`
//! reads reach {2} alone, which carries 3/3.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Int, Tensor, TensorData};

use ddrs::config::Config;
use ddrs::data::collate::{compress, UnionedCoo};
use ddrs::data::ids::{Comid, Staid};
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;
use ddrs::training::forward::scatter_add_by_group;

type I = NdArray<f32>;
type AB = Autodiff<I>;

/// Parent (MERIT) reaches in the fixture network. The *row* count grows when
/// the gauge reach is subdivided, but the mass balance is set by this.
const N: usize = 3;
/// Long enough for a 1 km / 0.1%-slope network at the configured dt to settle.
const T: usize = 64;
/// Constant lateral inflow per reach, m³/s.
const Q_PRIME: f32 = 10.0;

/// Same knobs as `tests/sp8_gradcheck.rs::mock_cfg` — puts velocity/depth in
/// well-conditioned, non-saturated regimes for this network.
fn mock_cfg(ddr_match: bool) -> Config {
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
    cfg.params.ddr_match = ddr_match;
    cfg
}

/// Two headwaters (0, 1) → gauge reach (2), with the gauge reach split into
/// `pieces` sub-reaches chained in series (reach subdivision, Task 3):
///
/// ```text
///   0 ─┐
///      ├─> 2 → 3 → … → (1 + pieces)     (gauge reach; outlet is the last)
///   1 ─┘
/// ```
///
/// Total gauge-reach length is held at 1000 m regardless of `pieces`, so the
/// steady state is unchanged — the physics is identical, only the discretization
/// differs. Returns the adjacency plus its CONUS-space `parent_offset`.
fn confluence_sparse(pieces: usize) -> (SparseAdjacency, Vec<i32>) {
    assert!(pieces >= 1);
    let n = 2 + pieces;
    let mut rows: Vec<i32> = vec![2, 2];
    let mut cols: Vec<i32> = vec![0, 1];
    for k in 1..pieces {
        cols.push((2 + k - 1) as i32);
        rows.push((2 + k) as i32);
    }
    let piece_len = 1000.0 / pieces as f32;
    let mut length_m = vec![1000.0_f32; 2];
    length_m.extend(std::iter::repeat_n(piece_len, pieces));
    let parent_offset: Vec<i32> = vec![0, 1, 2, n as i32];
    let adj = SparseAdjacency {
        n,
        values: vec![1.0; rows.len()],
        rows,
        cols,
        length_m,
        slope: vec![0.001; n],
        parent_offset: Some(parent_offset.clone()),
    };
    (adj, parent_offset)
}

/// Run `collate::compress` over the same topology to get `outflow_idx` from
/// the REAL production code path (not a hand-rolled copy).
fn outflow_idx_from_collate(ddr_match: bool, pieces: usize) -> Vec<Vec<usize>> {
    let gauge_comid = Comid(73005764);
    let mut conus_order = vec![Comid(73006562), Comid(73006585)];
    conus_order.extend(std::iter::repeat_n(gauge_comid, pieces));
    let (adj, parent_offset) = confluence_sparse(pieces);
    let edges: Vec<(usize, usize)> = adj
        .rows
        .iter()
        .zip(adj.cols.iter())
        .map(|(&r, &c)| (r as usize, c as usize))
        .collect();
    // The subgraph builder resolves a gauge COMID to its parent's LAST row, so
    // `gage_idx` is the outlet piece (`cache.rs::resolve_or_build`).
    let unioned = UnionedCoo {
        edges,
        gauges: vec![(Staid::new("01457000"), adj.n - 1, "73005764".to_string())],
    };
    compress(&unioned, &conus_order, ddr_match, Some(&parent_offset))
        .expect("compress")
        .outflow_idx
}

/// Route the network to steady state and extract the gauge series exactly as
/// training does. Returns `(gauge_series, all_reach_discharge_row_major_NxT)`.
fn route_and_extract(ddr_match: bool, pieces: usize) -> (Vec<f32>, Vec<f32>) {
    let cfg = mock_cfg(ddr_match);
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let (adjacency, _) = confluence_sparse(pieces);
    let n = adjacency.n;

    // Constant q_prime on every ROW — exactly what `StreamflowStore::read_window`
    // hands back, since every piece of a parent carries the parent's COMID. The
    // engine is what divides by the piece count (`mmc.rs`, Task 5).
    let q_prime: Tensor<AB, 2> =
        Tensor::from_data(TensorData::new(vec![Q_PRIME; T * n], [T, n]), &device);

    // Mid-range normalized params → n ≈ 0.055, q_spatial ≈ 0.5, p_spatial ≈ 100.
    let mk = |v: f32| -> Tensor<AB, 1> { Tensor::from_floats(vec![v; n].as_slice(), &device) };
    let mut engine = MuskingumCunge::<I>::new(cfg, device.clone());
    engine.setup_inputs(
        RoutingInputs {
            adjacency,
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

    // (N, T) routed discharge.
    let runoff: Tensor<I, 2> = engine.forward().inner();

    let outflow_idx = outflow_idx_from_collate(ddr_match, pieces);
    let flat: Vec<i32> = outflow_idx[0].iter().map(|&c| c as i32).collect();
    let groups: Vec<i32> = vec![0; flat.len()];
    let flat_t: Tensor<I, 1, Int> =
        Tensor::from_data(TensorData::from(flat.as_slice()), &device);
    let group_t: Tensor<I, 1, Int> =
        Tensor::from_data(TensorData::from(groups.as_slice()), &device);
    let gauge_q: Tensor<I, 2> = scatter_add_by_group(runoff.clone(), flat_t, group_t, 1);

    let series: Vec<f32> = gauge_q
        .into_data()
        .to_vec::<f32>()
        .expect("gauge series to host");
    let all: Vec<f32> = runoff.into_data().to_vec::<f32>().expect("runoff to host");
    (series, all)
}

/// Total lateral inflow entering the whole subgraph, m³/s. At steady state
/// every drop of it must appear at the gauge.
const EXPECTED: f32 = Q_PRIME * N as f32; // 30.0

#[test]
fn gauge_prediction_conserves_mass_when_not_ddr_match() {
    let (series, all) = route_and_extract(false, 1);
    let final_q = series[T - 1];
    println!(
        "ddr_match=false: gauge={final_q:.4} m3/s  expected={EXPECTED:.4}  \
         ratio={:.4}",
        final_q / EXPECTED
    );

    assert!(
        (final_q - EXPECTED).abs() / EXPECTED < 1e-2,
        "gauge prediction at steady state = {final_q} m³/s, expected {EXPECTED} \
         m³/s (sum of q_prime over all {N} reaches). Ratio {:.3}. A ratio near \
         {:.3} means outflow_idx is pointing at the gauge's upstream neighbours \
         instead of the gauge reach itself.",
        final_q / EXPECTED,
        (N as f32 - 1.0) / N as f32,
    );

    // And the extracted prediction must literally BE the gauge reach's own
    // routed discharge, at every timestep — not a sum over other reaches.
    for t in 0..T {
        let reach2 = all[2 * T + t];
        assert!(
            (series[t] - reach2).abs() <= 1e-4 * reach2.abs().max(1.0),
            "t={t}: gauge prediction {} != gauge reach discharge {reach2}",
            series[t],
        );
    }
}

#[test]
fn ddr_match_gauge_prediction_omits_the_gauge_reach() {
    // Pins the defect that `ddr_match: true` faithfully reproduces, and proves
    // the mass check above actually discriminates between the two conventions.
    // Summing the two headwaters yields 2/3 of the network's lateral inflow;
    // the gauge reach's own 10 m³/s is silently dropped.
    let (series, all) = route_and_extract(true, 1);
    let final_q = series[T - 1];
    let ddr_expected = Q_PRIME * (N as f32 - 1.0); // 20.0
    println!(
        "ddr_match=true : gauge={final_q:.4} m3/s  mass-conserving={EXPECTED:.4}  \
         ratio={:.4}",
        final_q / EXPECTED
    );

    assert!(
        (final_q - ddr_expected).abs() / ddr_expected < 1e-2,
        "ddr_match=true must sum the two headwaters only: got {final_q}, \
         expected {ddr_expected}"
    );
    assert!(
        (final_q - EXPECTED).abs() / EXPECTED > 0.2,
        "ddr_match=true must NOT conserve mass — if it does, the test network \
         no longer discriminates the two outflow_idx conventions"
    );
    // The gauge reach itself carries the full mass; it is simply not read.
    let reach2_final = all[2 * T + (T - 1)];
    assert!(
        (reach2_final - EXPECTED).abs() / EXPECTED < 1e-2,
        "gauge reach discharge {reach2_final} should still conserve mass \
         ({EXPECTED}); the defect is in extraction, not routing"
    );
}

/// Steady-state gauge discharge with the gauge's own reach split `pieces` ways.
fn gauge_steady_state(pieces: usize) -> f32 {
    route_and_extract(false, pieces).0[T - 1]
}

#[test]
fn gauge_conserves_mass_when_its_reach_is_subdivided() {
    // Same topology as `gauge_prediction_conserves_mass_when_not_ddr_match`,
    // but the gauge's own reach is split 4 ways. The answer must not change:
    // the MC solve is mass-conserving down the internal chain, so the whole
    // reach's runoff — upstream network plus its own `q'`, split `q'/4` across
    // the pieces — arrives at the LAST piece.
    let un_split = gauge_steady_state(1);
    let split = gauge_steady_state(4);
    println!(
        "subdivision: 1 piece = {un_split:.6} m3/s   4 pieces = {split:.6} m3/s   \
         expected = {EXPECTED:.6}"
    );
    assert!((un_split - EXPECTED).abs() < 1e-3, "control changed: {un_split}");
    assert!(
        (split - EXPECTED).abs() < 1e-3,
        "subdivided gauge lost mass: got {split}, expected {EXPECTED}. Reading \
         any piece other than the outlet drops the downstream fraction of the \
         gauge reach's own lateral inflow — the inlet piece carries only \
         {:.1} m3/s here.",
        Q_PRIME * (N as f32 - 1.0) + Q_PRIME / 4.0,
    );
}

#[test]
fn subdivided_interior_pieces_carry_less_than_the_outlet() {
    // Proves the test above actually discriminates: the four pieces of the
    // gauge reach form a strictly increasing ramp (22.5, 25.0, 27.5, 30.0), so
    // pointing `outflow_idx` at anything but the last piece is detectable.
    let (_, all) = route_and_extract(false, 4);
    let n = 2 + 4;
    let piece_q: Vec<f32> = (2..n).map(|r| all[r * T + (T - 1)]).collect();
    println!("gauge-reach pieces at steady state: {piece_q:?}");
    for w in piece_q.windows(2) {
        assert!(
            w[1] > w[0] + 1e-3,
            "pieces must accumulate downstream, got {piece_q:?}"
        );
    }
    let expected_inlet = Q_PRIME * (N as f32 - 1.0) + Q_PRIME / 4.0; // 22.5
    assert!(
        (piece_q[0] - expected_inlet).abs() < 1e-2,
        "inlet piece {} != {expected_inlet} (two headwaters + one quarter of \
         the gauge reach's own lateral inflow)",
        piece_q[0]
    );
    assert!((piece_q[3] - EXPECTED).abs() < 1e-2, "outlet {} != {EXPECTED}", piece_q[3]);
}
