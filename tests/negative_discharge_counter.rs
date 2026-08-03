//! The S28 clamp silently turns negative solves into +1e-4. This counter is
//! the only way to see how often Muskingum's non-negative-coefficient
//! condition (2X <= Cr <= 2(1-X)) is violated in a real run.
use ddrs::routing::mmc_op::{negative_solve_stats, reset_negative_solve_stats};

#[test]
fn counter_starts_at_zero_and_resets() {
    reset_negative_solve_stats();
    let (neg, total) = negative_solve_stats();
    assert_eq!(neg, 0);
    assert_eq!(total, 0);
}
