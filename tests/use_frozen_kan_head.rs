//! `experiment.use_frozen_kan_head` defaults to true so every existing config
//! keeps its current behaviour: presence of `kan_head.disaggregation:` enables
//! the head. Setting it false strips the block at load, giving the flat
//! repeat-24 (nearest) daily→hourly fallback with the block left inert in YAML.
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
"#;

const HEAD: &str = r#"
kan_head:
  hidden_size: 21
  num_hidden_layers: 2
  input_var_names: [aridity]
  learnable_parameters: [n]
  disaggregation:
    hidden_size: 16
params:
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
"#;

fn load(name: &str, extra_experiment: &str) -> Config {
    let yaml = format!("{BASE}{extra_experiment}{HEAD}");
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, yaml).unwrap();
    Config::from_yaml_file(&path).expect("parse")
}

#[test]
fn defaults_to_true_and_keeps_disagg_block() {
    let cfg = load("ddrs_frozen_head_default_test.yaml", "");
    assert!(cfg.experiment.as_ref().unwrap().use_frozen_kan_head);
    assert!(
        cfg.kan_head.unwrap().disaggregation.is_some(),
        "absent flag must leave the disaggregation block enabled"
    );
}

#[test]
fn false_strips_disagg_block() {
    let cfg = load(
        "ddrs_frozen_head_off_test.yaml",
        "  use_frozen_kan_head: false\n",
    );
    assert!(!cfg.experiment.as_ref().unwrap().use_frozen_kan_head);
    assert!(
        cfg.kan_head.unwrap().disaggregation.is_none(),
        "use_frozen_kan_head: false must strip the disaggregation block"
    );
}

#[test]
fn explicit_true_keeps_disagg_block() {
    let cfg = load(
        "ddrs_frozen_head_on_test.yaml",
        "  use_frozen_kan_head: true\n",
    );
    assert!(cfg.kan_head.unwrap().disaggregation.is_some());
}
