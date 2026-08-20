//! `ddr_match` is DEPRECATED and defaults to false (corrected physics) since
//! 2026-08-19 — DDR removed its legacy path in DeepGroundwater/ddr#192, so
//! both implementations now share the corrected formulation by default. The
//! legacy path stays reachable via an explicit (warned) `ddr_match: true`.
use ddrs::config::Config;

#[test]
fn ddr_match_defaults_to_false() {
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
    assert!(
        !cfg.params.ddr_match,
        "ddr_match must default to false (corrected physics, DDR post-#192)"
    );
}

#[test]
fn ddr_match_legacy_path_still_loads() {
    // Deprecated but not removed: explicit true must still parse (it emits a
    // WARN on stderr) so pre-#192 results stay reproducible and CUDA graphs
    // stay usable.
    let yaml = r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
params:
  ddr_match: true
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
"#;
    let path = std::env::temp_dir().join("ddrs_ddr_match_disabled_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("parse");
    assert!(cfg.params.ddr_match);
}

#[test]
fn enforce_positivity_defaults_to_false() {
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
    let path = std::env::temp_dir().join("ddrs_enforce_positivity_default_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("parse");
    assert!(
        !cfg.params.enforce_positivity,
        "enforce_positivity must default to false so existing runs are unchanged"
    );
}

#[test]
fn enforce_positivity_requires_corrected_physics() {
    // ddr_match: true + enforce_positivity: true must be rejected at load: the
    // clamp changes K and X, which would break compare_ddr_sandbox.
    let yaml = r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
params:
  ddr_match: true
  enforce_positivity: true
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
"#;
    let path = std::env::temp_dir().join("ddrs_enforce_positivity_reject_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let err = Config::from_yaml_file(&path).expect_err("must reject");
    let msg = err.to_string();
    assert!(
        msg.contains("enforce_positivity"),
        "error must name the offending key, got: {msg}"
    );
}

#[test]
fn enforce_positivity_loads_with_corrected_physics() {
    let yaml = r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
params:
  ddr_match: false
  use_cuda_graphs: false
  enforce_positivity: true
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
"#;
    let path = std::env::temp_dir().join("ddrs_enforce_positivity_accept_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("ddr_match:false + enforce_positivity:true must load");
    assert!(cfg.params.enforce_positivity);
    assert!(!cfg.params.ddr_match);
}
