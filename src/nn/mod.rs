//! Neural-network heads that emit routing parameters for the MC engine.

pub mod init;
pub mod kan_head;

pub use kan_head::{KanHead, KanHeadConfig};
