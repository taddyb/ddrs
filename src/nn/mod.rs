//! Neural-network heads that emit routing parameters for the MC engine.

pub mod dam_params;
pub mod disagg_head;
pub mod init;
pub mod kan_head;
pub mod release_head;

pub use dam_params::DamParams;
pub use disagg_head::{DisaggHead, DisaggHeadConfig};
pub use kan_head::{KanHead, KanHeadConfig};
