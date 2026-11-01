//! Autodiff tape size per routing hour: the fused stage op against the
//! reference step.
//!
//! `NodeId::new()` draws from one process-wide counter, so the ids drawn
//! during an hour count the autodiff tensors created in it. That counter is
//! shared by every test in a binary, so this file holds ONE test.

mod common;

use std::sync::Arc;

use burn::backend::autodiff::NodeId;
use burn::tensor::Tensor;

use ddrs::config::DownstreamBoundary;
use ddrs::data::ConusAdjacencyStore;
use ddrs::routing::stage::{StageBlock, StageNetwork, StageParams};
use ddrs::sparse::{CsrPattern, SparseAdjacency};

use common::{linear_chain_sparse, InnerBackend, TestBackend, TestDevice};

fn tv(v: &[f32], device: &TestDevice) -> Tensor<TestBackend, 1> {
    Tensor::<TestBackend, 1>::from_floats(v, device)
}

fn leaf(v: &[f32], device: &TestDevice) -> Tensor<TestBackend, 1> {
    tv(v, device).require_grad()
}

/// Ids drawn per hour, averaged over hours 2 to 24, with tracked leaves.
fn ids_per_hour(adj: &SparseAdjacency, substeps: u32, reference: bool, device: &TestDevice) -> f64 {
    let pattern = Arc::new(CsrPattern::from_sparse(adj));
    let n = adj.n;
    let net = StageNetwork::build(adj, &pattern, None, 0.0, &[]).unwrap();
    let params = StageParams {
        theta: 1.0,
        theta_control: 0.5,
        substeps,
        boundary: DownstreamBoundary::NormalDepth,
        depth_lb: 0.01,
        bottom_width_lb: 0.1,
        dt: 3600.0,
        g: 9.81,
        use_cuda: false,
    };
    let mut block = StageBlock::<InnerBackend>::new(net, params, device);
    block.set_reference_step(reference);
    // Hotstart from the cumulative upstream sum of q' = 1.
    let mut q_init = vec![1.0f32; n];
    let mut down = vec![usize::MAX; n];
    for (&r, &c) in adj.rows.iter().zip(&adj.cols) {
        down[c as usize] = r as usize;
    }
    for i in 0..n {
        if down[i] != usize::MAX {
            let v = q_init[i];
            q_init[down[i]] += v;
        }
    }
    let slope_f: Vec<f32> = adj.slope.iter().map(|s| s.max(1e-3)).collect();
    block.hotstart(
        tv(&q_init, device),
        tv(&vec![0.035; n], device),
        tv(&vec![21.0; n], device),
        tv(&vec![0.5; n], device),
        tv(&slope_f, device),
    );
    let h0: Vec<f32> = block.stage().unwrap().into_data().to_vec().unwrap();
    block.set_stage(leaf(&h0, device));
    let (nn, pp, qq) = (leaf(&vec![0.035; n], device), leaf(&vec![21.0; n], device), leaf(&vec![0.5; n], device));
    let mut q_t = leaf(&q_init, device);
    let (mut total, mut counted) = (0u64, 0u64);
    for hour in 0..24 {
        let mc = leaf(&vec![0.0; n], device);
        let qp = leaf(&vec![1.0; n], device);
        let before = NodeId::new().value;
        let out = block.step(&pattern, nn.clone(), pp.clone(), qq.clone(), q_t.clone(), mc, qp);
        let after = NodeId::new().value;
        if hour >= 1 {
            total += after - before - 1;
            counted += 1;
        }
        q_t = out;
    }
    total as f64 / counted as f64
}

#[test]
fn node_ids_per_hour_fused_vs_reference() {
    let device = TestDevice::default();
    let chain = linear_chain_sparse(5);
    let store = ConusAdjacencyStore::open("examples/juniata/data/juniata_conus_adjacency.zarr")
        .expect("open bundle conus adjacency");
    let juniata = SparseAdjacency {
        n: store.n,
        rows: store.indices_0.clone(),
        cols: store.indices_1.clone(),
        values: vec![1.0; store.nnz],
        length_m: store.length_m.to_vec(),
        slope: store.slope.to_vec(),
        parent_offset: None,
    };
    for (name, adj) in [("5-chain", &chain), ("juniata", &juniata)] {
        for substeps in [1u32, 4] {
            let fused = ids_per_hour(adj, substeps, false, &device);
            let reference = ids_per_hour(adj, substeps, true, &device);
            println!("{name} substeps {substeps}: fused {fused:.1} ids/hour, reference {reference:.1} ids/hour");
            assert!(fused <= 8.0, "{name} substeps {substeps}: fused op draws {fused} ids per hour");
            assert!(reference >= 100.0, "{name} substeps {substeps}: reference draws only {reference} ids per hour");
        }
    }
}
