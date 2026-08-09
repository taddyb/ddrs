//! `kan_head.disaggregation.enabled` defaults to true so every existing config
//! keeps its behaviour: a bare block enables the head. `enabled: false` strips
//! the block at load, giving the flat repeat-24 (nearest) daily→hourly
//! fallback with the block left inert in YAML — the one-line ablation switch.
//! The section is `deny_unknown_fields`: phantom keys (e.g. the removed
//! `use_precip`) fail loudly instead of silently building a different head.
use ddrs::config::Config;

const BASE: &str = r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
experiment:
  batch_size: 4
  start_time: 1981/10/01
  end_time: 1982/09/30
  epochs: 1
  warmup: 5
kan_head:
  hidden_size: 21
  num_hidden_layers: 2
  input_var_names: [aridity]
  learnable_parameters: [n]
  disaggregation:
    hidden_size: 16
"#;

const PARAMS: &str = r#"
params:
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
"#;

fn load(name: &str, extra_disagg: &str) -> ddrs::data::error::Result<Config> {
    let yaml = format!("{BASE}{extra_disagg}{PARAMS}");
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, yaml).unwrap();
    Config::from_yaml_file(&path)
}

#[test]
fn bare_block_defaults_to_enabled() {
    let cfg = load("ddrs_disagg_enabled_default_test.yaml", "").expect("parse");
    assert!(
        cfg.kan_head.unwrap().disaggregation.is_some(),
        "a bare disaggregation block must enable the head"
    );
}

#[test]
fn enabled_false_strips_block() {
    let cfg = load("ddrs_disagg_enabled_off_test.yaml", "    enabled: false\n").expect("parse");
    assert!(
        cfg.kan_head.unwrap().disaggregation.is_none(),
        "enabled: false must strip the disaggregation block"
    );
}

#[test]
fn enabled_true_keeps_block() {
    let cfg = load("ddrs_disagg_enabled_on_test.yaml", "    enabled: true\n").expect("parse");
    assert!(cfg.kan_head.unwrap().disaggregation.is_some());
}

#[test]
fn unknown_key_is_rejected() {
    let err = load("ddrs_disagg_unknown_key_test.yaml", "    use_precip: true\n")
        .expect_err("phantom keys in the disaggregation block must fail load");
    let msg = err.to_string();
    assert!(
        msg.contains("unknown field") && msg.contains("use_precip"),
        "error should name the unknown field, got: {msg}"
    );
}
