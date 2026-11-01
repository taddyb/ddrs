//! The tie rule for clamps on the Muskingum-Cunge path.
//!
//! A clamp bound is INACTIVE at an exact tie, so the gradient passes. For a
//! `min` of two live operands, the tie goes to the first operand. This
//! matches Burn autodiff (`clamp_min` masks with a strict comparison, so a tie
//! passes) and DDR's `torch.clamp`.
//!
//! The backward in `mmc_op.rs::timestep_backward_core` and
//! `leakance.rs::zeta_backward` apply the rule at these sites:
//!
//! - discharge floor on the solved flow (B28);
//! - K floor (B18'), velocity band (B16), bottom width floor (B11), side
//!   slope band (B9) and depth floor (B6);
//! - Cunge X band (B19), read from the PRE-clamp X;
//! - dam-row X cap (B19p): the cap wins only when strictly smaller than the
//!   channel X, so a tie goes to the channel X;
//! - leakance losing-only gate and bed cap: a tie is losing and connected.
//!
//! Two sites read the saved CLAMPED value today (depth and X). A `>=` mask on
//! a clamped value would pass the gradient through every clamped entry, so
//! those masks read the pre-clamp value, recomputed with the forward's ops.

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

/// True where a lower bound `lb` cuts the gradient: `raw < lb`. A tie
/// (`raw == lb`) is inactive. NaN counts as active.
pub fn floor_active<B: Backend>(raw: Tensor<B, 1>, lb: f32) -> Tensor<B, 1, Bool> {
    raw.greater_equal_elem(lb).bool_not()
}

/// True where an upper bound `ub` cuts the gradient: `raw > ub`. A tie
/// (`raw == ub`) is inactive. NaN counts as active.
pub fn ceiling_active<B: Backend>(raw: Tensor<B, 1>, ub: f32) -> Tensor<B, 1, Bool> {
    raw.lower_equal_elem(ub).bool_not()
}
