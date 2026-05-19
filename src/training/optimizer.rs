//! Adam optimizer + lr schedule + gradient clipping.
//!
//! Mirrors `~/projects/ddr/scripts/train.py:34-39` (Adam construction)
//! and lines 64-71 (grad-clip via torch.nn.utils.clip_grad_norm_).

use std::collections::BTreeMap;

use burn::module::{AutodiffModule, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::ElementConversion;
use burn::tensor::{Tensor, backend::AutodiffBackend};

/// Resolve the lr to use for `epoch` from the YAML schedule
/// (`experiment.learning_rate: {1: 0.001, 3: 0.0005}`).
///
/// Largest key <= epoch wins. Falls back to the first entry if epoch
/// is below all keys. Mirrors `~/projects/ddr/src/ddr/scripts_utils.py::resolve_learning_rate`.
pub fn resolve_lr(schedule: &BTreeMap<usize, f32>, epoch: usize) -> f32 {
    schedule
        .range(..=epoch)
        .next_back()
        .map(|(_, &lr)| lr)
        .unwrap_or_else(|| schedule.values().next().copied().unwrap_or(0.001))
}

/// Build a fresh Adam optimizer with PyTorch-matching defaults:
/// beta1=0.9, beta2=0.999, eps=1e-8.
///
/// Note: BURN 0.21 `AdamConfig` defaults to eps=1e-5 (not 1e-8), so we
/// override explicitly to match `torch.optim.Adam` defaults.
/// Mirrors DDR's `torch.optim.Adam(params=nn.parameters(), lr=lr)`.
///
/// The returned type is `impl Optimizer<M, B>` — callers only need the
/// trait. The concrete type is `OptimizerAdaptor<Adam, M, B>` but we
/// don't expose it so this function remains module-agnostic.
pub fn build_adam<M, B>() -> impl Optimizer<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .init::<B, M>()
}

// ── gradient clipping ──────────────────────────────────────────────────────

/// Apply global-norm gradient clipping (consume + return pattern).
///
/// Walks every float parameter of `module` to read its gradient out of
/// `grads`, computes the global L2 norm, and scales every gradient by
/// `max_norm / norm` when `norm > max_norm`.  Mirrors PyTorch's
/// `torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=1.0)`.
///
/// # Why consume + return
/// `GradientsParams` does not expose mutable iteration over its type-erased
/// tensor store.  The cleanest BURN 0.21 API is to remove each grad,
/// optionally scale it, then re-register it — which requires consuming the
/// struct and returning a new one.
pub fn clip_grad_norm<M, B>(grads: GradientsParams, module: &M, max_norm: f32) -> GradientsParams
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    // --- pass 1: compute global squared-norm --------------------------------
    let mut norm_collector = NormCollector::<M, B> {
        grads: &grads,
        sum_sq: 0.0,
        _phantom: std::marker::PhantomData,
    };
    module.visit(&mut norm_collector);
    let global_norm = norm_collector.sum_sq.sqrt();

    if global_norm <= max_norm || global_norm == 0.0 {
        return grads;
    }

    // --- pass 2: scale each gradient ----------------------------------------
    let scale = max_norm / global_norm;
    let mut scaler = GradScaler::<M, B> {
        grads,
        scale,
        _phantom: std::marker::PhantomData,
    };
    // `module.map` visits every Param and calls `map_float`, which is where
    // we swap the scaled grad back.  We discard the returned module (params
    // are unchanged; only the GradientsParams side is rebuilt).
    let _ = module.clone().map(&mut scaler);
    scaler.grads
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Visitor that accumulates the sum of squared L2 norms of all gradients.
struct NormCollector<'a, M, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    sum_sq: f32,
    _phantom: std::marker::PhantomData<(M, B)>,
}

impl<M, B> ModuleVisitor<B> for NormCollector<'_, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) else {
            return;
        };
        // sum of squares for this param's gradient
        let ss: f32 = grad
            .powf_scalar(2.0_f32)
            .sum()
            .into_scalar()
            .elem::<f32>();
        self.sum_sq += ss;
    }
}

/// Mapper that removes each gradient from the inner `GradientsParams`,
/// scales it, and re-registers it.
///
/// The module parameters themselves are passed through unchanged.
struct GradScaler<M, B: AutodiffBackend> {
    grads: GradientsParams,
    scale: f32,
    _phantom: std::marker::PhantomData<(M, B)>,
}

impl<M, B> ModuleMapper<B> for GradScaler<M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id: ParamId = param.id;
        if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) {
            let scaled = grad.mul_scalar(self.scale);
            self.grads.register::<B::InnerBackend, D>(id, scaled);
        }
        param
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn resolve_lr_picks_largest_key_leq_epoch() {
        let mut sched: BTreeMap<usize, f32> = BTreeMap::new();
        sched.insert(1, 0.001);
        sched.insert(3, 0.0005);
        assert!((resolve_lr(&sched, 1) - 0.001).abs() < 1e-9);
        assert!((resolve_lr(&sched, 2) - 0.001).abs() < 1e-9);
        assert!((resolve_lr(&sched, 3) - 0.0005).abs() < 1e-9);
        assert!((resolve_lr(&sched, 100) - 0.0005).abs() < 1e-9);
    }
}
