//! Machine-enforced version of invariant 1: the 5-reach RAPID sandbox must
//! route to an ABSOLUTE MATCH (max abs diff < 1e-3 m³/s) against DDR's
//! committed output.
//!
//! `examples/compare_ddr_sandbox` prints the same verdict for humans (plus
//! CSV/PNG diagnostics); this test is the CI-gateable form. Both consume the
//! committed `fixtures/sandbox/` set. The MC forward here mirrors the
//! example's `run_mc` (integration tests are separate crates; the ~30 lines
//! are duplicated rather than reshaping `src/sandbox.rs`).

use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::BackendTypes;
use burn::tensor::Tensor;

use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sandbox::{self, N_REACHES};
use ddrs::sparse::SparseAdjacency;

type Inner = NdArray<f32>;
type B = Autodiff<Inner>;
type D = <Inner as BackendTypes>::Device;

/// DDR's routed output, shape `(N_REACHES, n_timesteps)` in RAPID2 order.
fn read_ddr_reference(path: &Path, expect_rows: usize, expect_cols: usize) -> Vec<f32> {
    let s = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path:?}: {e} — fixtures/sandbox is committed; a missing file means a broken checkout, not a skippable test"));
    let mut data = Vec::with_capacity(expect_rows * expect_cols);
    let mut rows = 0;
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<f32> = line.split(',').map(|x| x.trim().parse().unwrap()).collect();
        assert_eq!(cols.len(), expect_cols, "wrong col count in {path:?}");
        data.extend(cols);
        rows += 1;
    }
    assert_eq!(rows, expect_rows, "wrong row count in {path:?}");
    data
}

#[test]
fn sandbox_routes_to_absolute_match_against_ddr() {
    let fixtures = Path::new("fixtures/sandbox");
    let inputs = sandbox::load_from_dir(fixtures).expect("load sandbox fixtures");
    let n_timesteps = inputs.n_timesteps;

    let ddr_rapid2 = read_ddr_reference(
        &fixtures.join("ddr_discharge_rapid2.csv"),
        N_REACHES,
        n_timesteps,
    );

    // Same forward as examples/compare_ddr_sandbox.rs::run_mc (CPU path).
    let device = D::default();
    let qprime: Tensor<B, 2> = Tensor::<B, 1>::from_floats(inputs.qprime_flat.as_slice(), &device)
        .reshape([n_timesteps, N_REACHES]);
    let adjacency = SparseAdjacency::from_dense(
        N_REACHES,
        &inputs.adjacency_flat,
        vec![5000.0; N_REACHES],
        vec![0.001; N_REACHES],
    );
    let routing_inputs = RoutingInputs::<Inner> {
        adjacency,
        x_storage: Tensor::ones([N_REACHES], &device) * 0.25,
    };
    let params = SpatialParameters::<Inner> {
        n: Tensor::ones([N_REACHES], &device) * 0.5,
        q_spatial: Tensor::ones([N_REACHES], &device) * 0.5,
        p_spatial: None,
        k_d: None,
        d_gw: None,
        leakance_factor: None,
        impervious_mask: None,
    };
    let mut mc = MuskingumCunge::<Inner>::new(inputs.config.clone(), device.clone());
    mc.setup_inputs(routing_inputs, qprime, params, false, None);
    let topo_data: Vec<f32> = mc.forward().into_data().to_vec().unwrap();

    // Reorder topo -> RAPID2 to line up with the reference.
    let rapid2_idx_in_topo: Vec<usize> = inputs
        .rapid2_order
        .iter()
        .map(|rid| {
            inputs
                .topo_order
                .iter()
                .position(|t| t == rid)
                .expect("RAPID2 id missing from topo")
        })
        .collect();

    let mut overall_max_abs = 0.0_f32;
    let mut overall_max_rel = 0.0_f32;
    let mut worst_reach = 0_i32;
    for (r2_pos, &rid) in inputs.rapid2_order.iter().enumerate() {
        let topo_pos = rapid2_idx_in_topo[r2_pos];
        for t in 0..n_timesteps {
            let a = ddr_rapid2[r2_pos * n_timesteps + t];
            let b = topo_data[topo_pos * n_timesteps + t];
            let d = (a - b).abs();
            if d > overall_max_abs {
                overall_max_abs = d;
                worst_reach = rid;
            }
            if a.abs() > 1e-6 {
                overall_max_rel = overall_max_rel.max(d / a.abs());
            }
        }
    }

    eprintln!(
        "sandbox parity: max abs diff {overall_max_abs:.6e} m³/s, max rel diff {overall_max_rel:.6e} (worst reach {worst_reach})"
    );
    assert!(
        overall_max_abs < 1e-3,
        "invariant 1 broken: max abs diff {overall_max_abs:.6e} m³/s (reach {worst_reach}) >= 1e-3 — \
         run `cargo run --release --example compare_ddr_sandbox` for per-reach diagnostics"
    );
}
