//! Non-routing baselines for sanity-checking trained KAN performance.
//!
//! `summed_q_prime`: per-gauge sum of upstream divide Qr over the testing
//! eval window, compared against USGS daily observations. Mirrors
//! `~/projects/ddr/scripts/summed_q_prime.py`. If the trained KAN's
//! median NSE doesn't beat this baseline, the routing isn't earning its
//! keep — check loss curves and KAN-head gradient stats first, not the
//! sparse solver.

pub mod cache;
pub mod print;
pub mod summed_q_prime;

pub use cache::{cache_dir, cache_key, compute_or_load_cached, load_cached, save_cached};
pub use print::{print_metrics_summary, write_metrics_summary};
pub use summed_q_prime::{compute, BaselineError, SummedQPrime};
