use ddrs::cli::plan_bootstrap::{bootstrap, BootstrapInput, BootstrapSource};
use std::fs;

#[test]
fn bootstrap_copies_template_when_no_history() {
    let d = tempfile::tempdir().unwrap();
    let target = d.path().join("ddrs.yaml");
    let template = d.path().join("template.yaml");
    fs::write(&template, "workflow: train\n").unwrap();
    let input = BootstrapInput {
        target: target.clone(),
        runs_dir: d.path().join(".ddrs/runs"),
        bundled_template: template,
        editor_cmd: Some("true".into()), // shell `true` exits 0 immediately
        interactive: false,
    };
    let chosen = bootstrap(input).unwrap();
    assert!(matches!(chosen, BootstrapSource::Template));
    assert!(target.is_file());
}

#[test]
fn bootstrap_uses_latest_successful_run_when_present() {
    let d = tempfile::tempdir().unwrap();
    let runs = d.path().join(".ddrs/runs/2026-05-30T00-00-00-train");
    fs::create_dir_all(&runs).unwrap();
    fs::write(
        runs.join("manifest.json"),
        r#"{"status":"ok","workflow":"train","run_id":"x","ddrs_version":"x","git":{"sha":"x","dirty":false,"branch":"x"},"config_path":"x","started_at":"x","finished_at":null,"exit_reason":null,"system":{},"sources":{},"source_lock":{"lockfile":"x","matched":true,"drift":[]},"outputs":{"checkpoints":[],"plot":null},"metrics":{}}"#,
    )
    .unwrap();
    fs::write(runs.join("config.yaml"), "workflow: train\nfrom_last: true\n").unwrap();

    let target = d.path().join("ddrs.yaml");
    let template = d.path().join("template.yaml");
    fs::write(&template, "workflow: train\n").unwrap();
    let input = BootstrapInput {
        target: target.clone(),
        runs_dir: d.path().join(".ddrs/runs"),
        bundled_template: template,
        editor_cmd: Some("true".into()),
        interactive: false,
    };
    let chosen = bootstrap(input).unwrap();
    assert!(matches!(chosen, BootstrapSource::LastSuccessful(_)));
    let copied = fs::read_to_string(&target).unwrap();
    assert!(copied.contains("from_last: true"));
}
