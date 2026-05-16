//! Shared test fixtures — Rust mirror of `tests/routing/test_utils.py`.

#![allow(dead_code)]

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::{RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

pub type InnerBackend = NdArray<f32>;
pub type TestBackend = Autodiff<InnerBackend>;
pub type TestDevice = <InnerBackend as burn::tensor::backend::BackendTypes>::Device;

/// Mirror of `create_mock_config()`. The Python mock overrides discharge_lb=0.001
/// and bottom_width=0.1; reflect that here.
pub fn mock_config() -> Config {
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
    cfg
}

/// Build a linear-chain adjacency `N` of size `n` with `N[i+1, i] = 1` — matches
/// `MockRoutingDataclass.adjacency_matrix`. Returned as a dense tensor for tests
/// that still want one (the gradient-flow test prints/inspects shapes).
pub fn linear_chain_adjacency(n: usize, device: &TestDevice) -> Tensor<TestBackend, 2> {
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        data[(i + 1) * n + i] = 1.0;
    }
    Tensor::<TestBackend, 1>::from_floats(data.as_slice(), device).reshape([n, n])
}

/// Build a `SparseAdjacency` for a linear chain of length `n`, length 1000 m
/// and slope 0.001 per reach. Uses `from_dense` for clarity — the chain is
/// small enough that the O(n²) scan is irrelevant.
pub fn linear_chain_sparse(n: usize) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n * n];
    for i in 0..n - 1 {
        dense[(i + 1) * n + i] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, vec![1000.0; n], vec![0.001; n])
}

pub fn mock_routing_inputs(n: usize, device: &TestDevice) -> RoutingInputs<InnerBackend> {
    RoutingInputs {
        adjacency: linear_chain_sparse(n),
        x_storage: Tensor::ones([n], device) * 0.2,
    }
}

/// Mock streamflow: a base of 5 plus a sin sweep across time. Shape [T, n].
pub fn mock_streamflow(t: usize, n: usize, device: &TestDevice) -> Tensor<TestBackend, 2> {
    let mut data = vec![0.0_f32; t * n];
    for ti in 0..t {
        let phase = (ti as f32) / (t.max(2) - 1) as f32 * 4.0 * std::f32::consts::PI;
        for ri in 0..n {
            let v = 5.0 + phase.sin() * 2.0;
            data[ti * n + ri] = v.max(0.1);
        }
    }
    Tensor::<TestBackend, 1>::from_floats(data.as_slice(), device).reshape([t, n])
}

/// Mock spatial parameters (normalized to [0,1]).
pub fn mock_spatial_parameters(n: usize, device: &TestDevice) -> SpatialParameters<InnerBackend> {
    SpatialParameters {
        n: Tensor::ones([n], device) * 0.5,
        q_spatial: Tensor::ones([n], device) * 0.5,
        p_spatial: None,
    }
}
