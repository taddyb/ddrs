//! Fused MC timestep custom autodiff op. SP-8.
//!
//! Replaces the ~33 BURN-tensor-op chain in `MuskingumCunge::route_timestep`
//! with a single autograd node. Pattern mirrors `CsrSolveOp` in
//! `src/sparse/mod.rs:415-462`: a `Backward<B, N>` impl with a saved-state
//! struct holding backend primitives (no autograd participation).
//!
//! Parents in fixed order: [n, q_spatial, p_spatial, q_t, q_prime_t].

use std::sync::Arc;

use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::{backend::Backend, Tensor, TensorPrimitive};

use crate::config::Config;
use crate::sparse::{AValuesAssembler, CsrPattern};

/// Saved primitives used by `TimestepOp::backward`.
///
/// Forward inputs (the 5 autograd-tracked parents + the 3 constants) plus the
/// intermediates needed to evaluate the analytical chain rule.
#[derive(Clone, Debug)]
pub(crate) struct TimestepState<B: Backend> {
    pub pattern: Arc<CsrPattern>,
    // Inputs (autograd-tracked parents — read by backward to compute scalars
    // that flow into gradient register calls).
    pub n: B::FloatTensorPrimitive,
    pub q_spatial: B::FloatTensorPrimitive,
    pub p_spatial: B::FloatTensorPrimitive,
    pub q_t: B::FloatTensorPrimitive,
    pub q_prime_t: B::FloatTensorPrimitive,
    // Constants (not parents — used by backward but not differentiated through).
    pub length: B::FloatTensorPrimitive,
    pub slope: B::FloatTensorPrimitive,
    pub x_storage: B::FloatTensorPrimitive,
    // Forward intermediates (saved for backward).
    pub depth: B::FloatTensorPrimitive,
    pub top_width: B::FloatTensorPrimitive,
    pub side_slope: B::FloatTensorPrimitive,
    pub bottom_width: B::FloatTensorPrimitive,
    pub hydraulic_radius: B::FloatTensorPrimitive,
    pub velocity_unclamped: B::FloatTensorPrimitive,
    pub velocity_clamped: B::FloatTensorPrimitive,
    pub celerity: B::FloatTensorPrimitive,
    pub k_muskingum: B::FloatTensorPrimitive,
    pub denom: B::FloatTensorPrimitive,
    pub c1: B::FloatTensorPrimitive,
    pub c2: B::FloatTensorPrimitive,
    pub c3: B::FloatTensorPrimitive,
    pub c4: B::FloatTensorPrimitive,
    pub a_values: B::FloatTensorPrimitive,
    pub b_rhs: B::FloatTensorPrimitive,
    pub i_t: B::FloatTensorPrimitive, // N · Q_t  (SpMV result)
    pub x_sol: B::FloatTensorPrimitive, // pre-clamp solve output
    // Bookkeeping (small floats).
    pub depth_lb: f32,
    pub bottom_width_lb: f32,
    pub velocity_lb: f32,
    pub discharge_lb: f32,
    pub dt: f32,
}

#[derive(Debug)]
pub(crate) struct TimestepOp;

impl<B: Backend + 'static> Backward<B, 5> for TimestepOp
where
    B::FloatTensorPrimitive: 'static,
{
    type State = TimestepState<B>;

    fn backward(
        self,
        _ops: Ops<Self::State, 5>,
        _grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        // Task 3 fills this in. For Task 2 we leave the body empty — the op
        // compiles but registering it on the tape will not propagate any
        // gradients yet. V1/V2/V4 will fail until Task 3 lands; that's the
        // exact gate Task 3 must clear.
        unimplemented!("Task 3 — analytical backward not yet implemented");
    }
}

/// Forward + register-on-tape entry point. Called from
/// `MuskingumCunge::route_timestep` (Task 4). Returns Q_{t+1} as an
/// autograd-tracked rank-1 tensor.
///
/// Parent order: [n, q_spatial, p_spatial, q_t, q_prime_t].
#[allow(clippy::too_many_arguments)]
pub(crate) fn timestep_forward<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    assembler: &AValuesAssembler<I>,
    n: Tensor<Autodiff<I>, 1>,
    q_spatial: Tensor<Autodiff<I>, 1>,
    p_spatial: Tensor<Autodiff<I>, 1>,
    q_t: Tensor<Autodiff<I>, 1>,
    q_prime_t: Tensor<Autodiff<I>, 1>,
    length: Tensor<Autodiff<I>, 1>,
    slope: Tensor<Autodiff<I>, 1>,
    x_storage: Tensor<Autodiff<I>, 1>,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    // Task 2 returns a placeholder by panicking. The actual fused forward +
    // saved state lands in Task 4. Touch every parameter so they're not
    // dead-code-warned during this skeleton commit.
    let _ = (cfg, pattern, assembler, n.clone(), q_spatial.clone(),
             p_spatial.clone(), q_t.clone(), q_prime_t.clone(),
             length.clone(), slope.clone(), x_storage.clone());
    let _ = TimestepOp; // touch the symbol so it isn't dead-code-warned
    unimplemented!("Task 4 fills the fused forward and the OpsKind branches");
}
