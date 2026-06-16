use ddrs::cli::manifest::{GitInfo, Manifest, RunOutputs, SourceLockRef, SystemProbe};
use ddrs::cli::types::{RunStatus, Workflow};
use std::collections::BTreeMap;

#[test]
fn manifest_round_trips_via_serde_json() {
    let m = Manifest {
        run_id: "2026-05-30T00-00-00-train".into(),
        ddrs_version: "0.1.0".into(),
        git: GitInfo { sha: "abc".into(), dirty: false, branch: "main".into() },
        workflow: Workflow::Train,
        config_path: ".ddrs/runs/.../config.yaml".into(),
        started_at: "2026-05-30T00:00:00Z".into(),
        finished_at: Some("2026-05-30T01:00:00Z".into()),
        status: RunStatus::Ok,
        exit_reason: None,
        system: SystemProbe::default(),
        sources: BTreeMap::new(),
        resolved_adjacency: None,
        source_lock: SourceLockRef {
            lockfile: ".ddrs/sources.lock".into(),
            matched: true,
            drift: vec![],
        },
        outputs: RunOutputs {
            checkpoints: vec![],
            plot: None,
            eval_zarr: None,
            baseline_predictions: None,
            baseline_observations: None,
            baseline_manifest: None,
            run_log: None,
        },
        metrics: serde_json::json!({"final_loss": 0.385}),
        max_mini_batches: None,
    };
    let s = serde_json::to_string(&m).unwrap();
    let m2: Manifest = serde_json::from_str(&s).unwrap();
    assert_eq!(m, m2);
}
