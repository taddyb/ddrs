//! Stage-1 adjoint reachability probe (spec:
//! docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md).
//!
//! Gradients of the training objective w.r.t. the per-reach NORMALIZED
//! leakance parameters, read at a FIXED head (no optimizer step ever).
//! `lift_leaf` detaches a head output from its graph and re-registers it as
//! an autograd leaf; the analytical `TimestepLeakanceOp` backward already
//! provides exact grads for these parents (tests/leakance_gradcheck.rs), so
//! no Backward impl is touched.

use std::collections::HashMap;

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::config::Config;
use crate::data::dataset::RoutingTensors;
use crate::nn::kan_head::KanHead;
use crate::routing::utils::denormalize;
use crate::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use crate::training::forward::scatter_add_by_group;

/// Detach `t` from its autograd graph and re-lift it as a `require_grad`
/// leaf. Values are bit-identical; only the tape topology changes.
pub fn lift_leaf<I: Backend>(t: Tensor<Autodiff<I>, 1>) -> Tensor<Autodiff<I>, 1> {
    Tensor::<Autodiff<I>, 1>::from_inner(t.inner()).require_grad()
}

/// The three lifted per-reach leaves (normalized [0,1] space).
pub struct ProbeLeaves<I: Backend> {
    pub k_d: Tensor<Autodiff<I>, 1>,
    pub d_gw: Tensor<Autodiff<I>, 1>,
    pub factor: Tensor<Autodiff<I>, 1>,
}

/// Training-path forward (mirrors `forward`, src/training/forward.rs:169-252)
/// with the leakance vectors lifted as leaves. Returns gauge-hourly
/// predictions plus the leaves to read grads from after `loss.backward()`.
pub fn probe_forward<I: Backend>(
    cfg: &Config,
    tensors: &RoutingTensors<Autodiff<I>>,
    head: &KanHead<Autodiff<I>>,
    device: &I::Device,
) -> (Tensor<Autodiff<I>, 2>, ProbeLeaves<I>) {
    assert!(cfg.params.use_leakance, "probe requires params.use_leakance");
    let params_map = head.forward(tensors.spatial_attributes.clone());

    let n_param = params_map.get("n").expect("head missing n").clone();
    let q_param = params_map.get("q_spatial").expect("head missing q_spatial").clone();
    let p_param = params_map.get("p_spatial").cloned();

    let n_active = tensors.adjacency.n;
    let x_storage: Tensor<Autodiff<I>, 1> = match params_map.get("x_storage") {
        Some(x_norm) => denormalize(
            x_norm.clone(),
            cfg.params.parameter_ranges.x_storage,
            cfg.params.log_space_parameters.iter().any(|s| s == "x_storage"),
        ),
        None => Tensor::full([n_active], 0.3_f32, device),
    };

    let n_hourly = tensors.q_prime.dims()[0];
    let q_prime_hourly = match &head.disagg {
        Some(d) => d.forward(
            tensors.q_prime_daily.clone(),
            tensors.spatial_attributes.clone(),
            tensors.precip_hourly.clone(),
            tensors.temp_hourly.clone(),
            n_hourly,
        ),
        None => tensors.q_prime.clone(),
    };

    for key in &["K_D", "d_gw", "leakance_factor"] {
        assert!(
            params_map.contains_key(*key),
            "probe: head missing '{key}' — use a leakance experiment config"
        );
    }
    let leaves = ProbeLeaves {
        k_d: lift_leaf::<I>(params_map.get("K_D").unwrap().clone()),
        d_gw: lift_leaf::<I>(params_map.get("d_gw").unwrap().clone()),
        factor: lift_leaf::<I>(params_map.get("leakance_factor").unwrap().clone()),
    };

    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage },
        q_prime_hourly,
        SpatialParameters {
            n: n_param,
            q_spatial: q_param,
            p_spatial: p_param,
            k_d: Some(leaves.k_d.clone()),
            d_gw: Some(leaves.d_gw.clone()),
            leakance_factor: Some(leaves.factor.clone()),
        },
        false,
        tensors.initial_state.clone(),
    );
    let runoff = engine.forward();

    (
        scatter_add_by_group(
            runoff,
            tensors.flat_indices.clone(),
            tensors.group_ids.clone(),
            tensors.num_gauges,
        ),
        leaves,
    )
}

/// COMID-keyed gradient accumulation across probe batches. Batches route
/// different subnetworks (unions of 64 gauge subgraphs), so per-batch grads
/// are folded into a CPU map keyed by COMID.
pub struct GradAccum {
    map: HashMap<i64, (f64, f64, u32)>, // (Σ|g|, Σg, n_windows)
}

impl GradAccum {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    pub fn add(&mut self, comids: &[i64], grad: &[f32], grad_signed: &[f32]) {
        assert_eq!(comids.len(), grad.len());
        assert_eq!(comids.len(), grad_signed.len());
        for i in 0..comids.len() {
            let e = self.map.entry(comids[i]).or_insert((0.0, 0.0, 0));
            e.0 += grad[i].abs() as f64;
            e.1 += grad_signed[i] as f64;
            e.2 += 1;
        }
    }

    /// `(comid, abs_sum, net_sum, count)` sorted by COMID.
    pub fn into_sorted_rows(self) -> Vec<(i64, f64, f64, u32)> {
        let mut rows: Vec<_> = self.map.into_iter().map(|(c, (a, s, n))| (c, a, s, n)).collect();
        rows.sort_by_key(|r| r.0);
        rows
    }
}

impl Default for GradAccum {
    fn default() -> Self {
        Self::new()
    }
}
