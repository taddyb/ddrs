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

/// Two headwaters (0, 1) → gauge reach (2). Lower-triangular, topological.
fn confluence_sparse() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    dense[2 * N] = 1.0; // adj[2, 0]
    dense[2 * N + 1] = 1.0; // adj[2, 1]
    SparseAdjacency::from_dense(N, &dense, vec![1000.0; N], vec![0.001; N])
}

/// Run `collate::compress` over the same topology to get `outflow_idx` from
/// the REAL production code path (not a hand-rolled copy).
fn outflow_idx_from_collate(ddr_match: bool) -> Vec<Vec<usize>> {
    let conus_order = vec![Comid(73006562), Comid(73006585), Comid(73005764)];
    let unioned = UnionedCoo {
        edges: vec![(2, 0), (2, 1)],
        gauges: vec![(Staid::new("01457000"), 2, "73005764".to_string())],
    };
    compress(&unioned, &conus_order, ddr_match)
        .expect("compress")
        .outflow_idx
}

/// Route the network to steady state and extract the gauge series exactly as
/// training does. Returns `(gauge_series, all_reach_discharge_row_major_NxT)`.
fn route_and_extract(ddr_match: bool) -> (Vec<f32>, Vec<f32>) {
    let cfg = mock_cfg(ddr_match);
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();

    // Constant q_prime everywhere, shape (T, N).
    let q_prime: Tensor<AB, 2> =
        Tensor::from_data(TensorData::new(vec![Q_PRIME; T * N], [T, N]), &device);

    // Mid-range normalized params → n ≈ 0.055, q_spatial ≈ 0.5, p_spatial ≈ 100.
    let mk = |v: f32| -> Tensor<AB, 1> { Tensor::from_floats([v; N], &device) };
    let mut engine = MuskingumCunge::<I>::new(cfg, device.clone());
    engine.setup_inputs(
        RoutingInputs {
            adjacency: confluence_sparse(),
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

    let outflow_idx = outflow_idx_from_collate(ddr_match);
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
    let (series, all) = route_and_extract(false);
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
    let (series, all) = route_and_extract(true);
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
