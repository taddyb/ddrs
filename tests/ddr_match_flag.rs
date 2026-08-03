//! `ddr_match` defaults to true so every existing config and the DDR sandbox
//! parity example keep their current behaviour (invariant 1).
use ddrs::config::Config;

#[test]
fn ddr_match_defaults_to_true() {
    let yaml = r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
params:
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
"#;
    let path = std::env::temp_dir().join("ddrs_ddr_match_default_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("parse");
    assert!(cfg.params.ddr_match, "ddr_match must default to true");
}

#[test]
fn ddr_match_can_be_disabled() {
    let yaml = r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
params:
  ddr_match: false
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
"#;
    let path = std::env::temp_dir().join("ddrs_ddr_match_disabled_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("parse");
    assert!(!cfg.params.ddr_match);
}
