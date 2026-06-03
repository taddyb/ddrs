#[test]
fn run_train_requires_gpu_when_none_probed() {
    // Soft test: if a GPU IS present, skip. Otherwise assert the pre-flight fires.
    if ddrs::cli::system::probe().ok().flatten()
        .map(|p| !p.gpu.is_empty()).unwrap_or(false)
    {
        eprintln!("skipping — GPU present on this host");
        return;
    }
    use std::fs;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = tmp.path().join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: train
data_sources:
  attributes: /dev/null
  conus_adjacency: /dev/null
  gages_adjacency: /dev/null
  streamflow: /dev/null
  observations: /dev/null
  gages: /dev/null
"#).unwrap();
    let ws_root = tmp.path().join(".ddrs");
    fs::create_dir_all(ws_root.join("runs")).unwrap();
    let lock = ddrs::cli::lockfile::Lockfile {
        ddrs_version: "test".into(),
        created_at: "0".into(),
        sources: std::collections::BTreeMap::new(),
    };
    lock.write_atomic(&ws_root.join("sources.lock")).unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let err = ddrs::cli::run::run(ddrs::cli::run::RunInput {
        workspace: ws,
        config_path: cfg_path,
        workflow: None,
        plot: false,
        strict: false,
        max_mini_batches: Some(1),
    }).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("requires a CUDA GPU") && msg.contains("train"),
        "expected GPU pre-flight, got: {msg}"
    );
}
