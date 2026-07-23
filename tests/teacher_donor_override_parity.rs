//! Parity gate for teacher mode's new n/q_spatial/p_spatial donor override
//! (docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md).
//!
//! Injecting a checkpoint's OWN dump_parameters output as the teacher's
//! donor field must reproduce the SAME synthetic gauge observations as
//! running teacher mode with no donor override at all — this exercises the
//! real donor-file I/O path (COMID keying, f32 round-trip, gather-by-comid)
//! that `RoutingParamOverride`'s unit tests in `src/training/forward.rs`
//! don't cover, mirroring `tests/eval_loss_own_parity.rs`'s pattern applied
//! to teacher mode.
//!
//! Skips gracefully if the real checkpoint/dump aren't present (machine-local,
//! gitignored) so CI on a clean checkout doesn't break.

use std::path::{Path, PathBuf};
use std::process::Command;

const CONFIG: &str = "/home/tbindas/projects/ddrs/config/experiments/synthetic_n_teacher.yaml";
const CHECKPOINT: &str =
    "/home/tbindas/projects/ddrs/.ddrs/runs/2026-07-16T02-22-14Z-train-and-test/checkpoints/epoch_5_mb_35";
const OWN_DUMP: &str = "/home/tbindas/projects/ddrs/output/synthetic_n/own_dump_for_parity_test.nc";

fn skip_if_missing(path: &str) -> Option<PathBuf> {
    let p = PathBuf::from(path);
    p.exists().then_some(p)
}

fn run_teacher(donor: Option<&Path>, obs_output: &Path) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_probe_zeta_gradient"));
    cmd.args(["--mode", "teacher", "--backend", "cpu", "--config", CONFIG])
        .arg("--checkpoint")
        .arg(CHECKPOINT)
        .args(["--eval-days", "10"])
        .arg("--obs-output")
        .arg(obs_output);
    if let Some(d) = donor {
        cmd.arg("--donor-params-nc").arg(d);
    }
    let status = cmd.status().expect("run probe_zeta_gradient");
    assert!(status.success(), "teacher mode run failed (donor={donor:?})");
}

#[test]
fn own_donor_reproduces_no_donor_synthetic_obs() {
    let Some(config) = skip_if_missing(CONFIG) else {
        eprintln!("skipping: {CONFIG} not present on this machine");
        return;
    };
    let Some(checkpoint_head) = skip_if_missing(&format!("{CHECKPOINT}/head.mpk")) else {
        eprintln!("skipping: {CHECKPOINT}/head.mpk not present on this machine");
        return;
    };
    let Some(own_dump) = skip_if_missing(OWN_DUMP) else {
        eprintln!(
            "skipping: {OWN_DUMP} not present — generate via \
             `cargo run --release --bin dump_parameters -- --backend cpu \
             --config {CONFIG} --checkpoint {CHECKPOINT}/head --output {OWN_DUMP}`"
        );
        return;
    };
    drop((config, checkpoint_head));

    let tmp = std::env::temp_dir().join(format!(
        "teacher_donor_parity_{}",
        std::process::id()
    ));
    let no_donor = tmp.join("no_donor");
    let with_donor = tmp.join("with_donor");
    std::fs::create_dir_all(&tmp).unwrap();

    run_teacher(None, &no_donor);
    run_teacher(Some(&own_dump), &with_donor);

    // Compare the raw zarr-v2 chunk bytes for a sample of gauges — the
    // writer's chunk layout is deterministic given identical inputs, so
    // byte-identical chunks prove the donor path reproduced the exact same
    // routed discharge as the no-donor path.
    let mut compared = 0;
    for entry in std::fs::read_dir(&no_donor).unwrap() {
        let entry = entry.unwrap();
        if !entry.path().is_dir() {
            continue;
        }
        let gauge = entry.file_name();
        let a = std::fs::read(entry.path().join("0")).unwrap();
        let b = std::fs::read(with_donor.join(&gauge).join("0")).unwrap();
        assert_eq!(a, b, "gauge {gauge:?}: own-donor chunk diverged from no-donor chunk");
        compared += 1;
    }
    assert!(compared > 0, "no gauge chunks found to compare — obs writer produced nothing");
    eprintln!("compared {compared} gauges — own-donor teacher run byte-identical to no-donor");

    std::fs::remove_dir_all(&tmp).ok();
}
