//! Training driver for the BURN MC engine + MLP head.
//!
//! Mirrors `~/projects/ddr/scripts/train.py:23-128` for the per-batch
//! forward/loss/backward step and `_test` in
//! `~/projects/ddr/scripts/train_and_test.py:43-119` for inference.
//!
//! Verification ladder (see `.claude/specs/2026-05-17-sp4-training-design.md`):
//!   V1 — single small batch, frozen scalar params, loss matches DDR.
//!   V2 — all filtered gauges in one batch, same frozen params.
//!   V3 — full training loop runs end-to-end without divergence.

pub mod checkpoint;
pub mod driver;
pub mod forward;
pub mod loss;
pub mod metrics;
pub mod optimizer;

pub use checkpoint::{load_mlp, save_mlp};
pub use forward::{
    scatter_add_by_group, forward_with_frozen_params, FrozenParams,
    FROZEN_N, FROZEN_Q_SPATIAL, FROZEN_P_SPATIAL,
};
pub use loss::{tau_trim_and_downsample, filter_nan_gauges, l1_loss_post_warmup, FilteredPair};
pub use driver::{train, TrainState};
pub use metrics::Metrics;
pub use optimizer::{resolve_lr, build_adam, clip_grad_norm};
