//! Layer-2 parity gate for the H5/H6 `eval-loss` probe mode
//! (docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md).
//!
//! Injecting a checkpoint's OWN `n`/`q_spatial`/`p_spatial` dump
//! (`R1_kan_parameters.nc`, produced by `dump_parameters::write_netcdf`) as a
//! "full-swap" donor must reproduce the same per-window loss as running with
//! no override at all — this exercises the real donor-file I/O path (COMID
//! keying, f32 storage round-trip) that the synthetic unit test in
//! `src/training/forward.rs` doesn't. This is the actual gate the campaign
//! must pass before trusting any real n-swap/geo-swap/full-swap result.
//!
//! Skips gracefully if the real checkpoint/dump aren't present (machine-local,
//! gitignored) so CI on a clean checkout doesn't break — same pattern as
//! `tests/data_zarr_store.rs`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const CONFIG: &str = "/home/tbindas/projects/ddrs/.ddrs/runs/2026-07-07T03-55-53Z-train-and-test/config.yaml";
const CHECKPOINT: &str = "/home/tbindas/projects/ddrs/.ddrs/runs/2026-07-07T03-55-53Z-train-and-test/checkpoints/epoch_5_mb_35";
const OWN_DUMP: &str = "/home/tbindas/projects/ddrs/output/equif/R1_kan_parameters.nc";

fn skip_if_missing(path: &str) -> Option<PathBuf> {
    let p = PathBuf::from(path);
    p.exists().then_some(p)
}

/// Run `probe_zeta_gradient --mode eval-loss` and return the per-window mean
/// losses for the (single) composition it was asked to evaluate, keyed by
/// window index.
fn run_eval_loss(
    config: &Path,
    checkpoint: &Path,
    compositions: &str,
    donor: Option<&Path>,
    windows: u32,
    seed: u32,
    output: &Path,
) -> HashMap<usize, f32> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_probe_zeta_gradient"));
    cmd.args([
        "--mode",
        "eval-loss",
        "--backend",
        "cpu",
        "--config",
    ])
    .arg(config)
    .arg("--checkpoint")
    .arg(checkpoint)
    .args(["--compositions", compositions])
    .args(["--windows", &windows.to_string()])
    .args(["--seed", &seed.to_string()])
    .arg("--loss-output")
    .arg(output);
    if let Some(d) = donor {
        cmd.arg("--donor-params-nc").arg(d);
    }

    let out = cmd.output().expect("probe_zeta_gradient binary should run");
    assert!(
        out.status.success(),
        "eval-loss run failed (compositions={compositions}): {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let csv = std::fs::read_to_string(output).expect("read loss-output CSV");
    let mut losses = HashMap::new();
    for line in csv.lines().skip(1) {
        // composition,window,mean_loss
        let mut parts = line.split(',');
        let _composition = parts.next().unwrap();
        let window: usize = parts.next().unwrap().parse().unwrap();
        let loss: f32 = parts.next().unwrap().parse().unwrap();
        losses.insert(window, loss);
    }
    losses
}

#[test]
fn own_dump_full_swap_matches_no_override_baseline() {
    let Some(config) = skip_if_missing(CONFIG) else {
        eprintln!("skip: {CONFIG} not present");
        return;
    };
    let Some(checkpoint) = skip_if_missing(CHECKPOINT) else {
        eprintln!("skip: {CHECKPOINT} not present");
        return;
    };
    let Some(own_dump) = skip_if_missing(OWN_DUMP) else {
        eprintln!("skip: {OWN_DUMP} not present");
        return;
    };

    let dir = std::env::temp_dir().join("eval_loss_own_parity_test");
    std::fs::create_dir_all(&dir).unwrap();
    let baseline_csv = dir.join("baseline.csv");
    let swap_csv = dir.join("full_swap.csv");

    // Same --seed/--windows for both runs — the plan (window/gauge sequence)
    // is a pure function of (dataset, seed, windows), independent of the
    // composition requested, so both invocations draw identical windows.
    let baseline =
        run_eval_loss(&config, &checkpoint, "own", None, 4, 42, &baseline_csv);
    let swapped = run_eval_loss(
        &config,
        &checkpoint,
        "full-swap",
        Some(&own_dump),
        4,
        42,
        &swap_csv,
    );

    assert_eq!(baseline.len(), 4, "expected 4 windows in baseline");
    assert_eq!(swapped.len(), 4, "expected 4 windows in full-swap");

    for (window, base_loss) in &baseline {
        let swap_loss = swapped
            .get(window)
            .unwrap_or_else(|| panic!("window {window} missing from full-swap output"));
        let abs_diff = (base_loss - swap_loss).abs();
        let rel_diff = abs_diff / base_loss.abs().max(1e-6);
        // Relative, not absolute: this is a full-CONUS-scale network loss
        // (O(1-10) m3/s), not the 5-reach sandbox fixture — the same
        // f32-accumulation-at-scale relaxation already established in
        // tests/training_verification.rs's V4 test (1e-3 RELATIVE, "f32
        // accumulation drift through the triangular solve and per-timestep
        // geometry recomputation exceeds the V2 1e-4 bound at this scale").
        assert!(
            rel_diff < 1e-3,
            "window {window}: own-dump full-swap diverged from no-override baseline \
             (own={base_loss}, full-swap={swap_loss}, rel diff={rel_diff:.3e})"
        );
    }
}
