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

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, TensorData};

use ddrs::config::Config;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

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
    all[(n - 1) * T + (T - 1)]
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
