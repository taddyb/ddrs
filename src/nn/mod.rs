//! Neural-network heads that emit routing parameters for the MC engine.

pub mod disagg_head;
pub mod init;
pub mod kan_head;

pub use disagg_head::{DisaggHead, DisaggHeadConfig};
pub use kan_head::{KanHead, KanHeadConfig};
