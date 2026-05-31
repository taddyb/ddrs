//! End-to-end `--plot` test. Requires:
//!   - merit data sources reachable
//!   - a trained checkpoint produced by an earlier real run (the v1
//!     dispatch stub does not produce a real checkpoint)
//!
//! Until the dispatch stub is replaced with real training, this test
//! is `#[ignore]`-gated. Run with `cargo test --test cli_plot -- --ignored`
//! on a host that meets those prereqs.

#[test]
#[ignore = "needs real training dispatch (Task 16's v1 stub doesn't produce checkpoints)"]
fn run_with_plot_writes_kan_parameters_nc() {
    // Intentionally empty; this test exists so the file compiles and
    // future work has a place to land.
    //
    // Sketch when real dispatch lands:
    //   1. init Phase A+B as in cli_run_drift.
    //   2. `run --workflow train --max-mini-batches 1 --plot`.
    //   3. assert run_dir/plot/kan_parameters.nc exists and has > 0 bytes.
    //   4. assert manifest.outputs.plot == Some("plot/kan_parameters.nc").
}
