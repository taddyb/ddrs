//! `params.subdivision` — static reach subdivision (variable Δx).
//!
//! Defaults to disabled so every existing config keeps its current behaviour.
use ddrs::config::Config;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A minimal valid training config, with `extra` spliced into `params:`.
/// `extra` must already be indented two spaces (it sits at `params:` depth).
fn yaml_with_params(extra: &str) -> String {
    format!(
        r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
params:
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
{extra}"#
    )
}

/// Inline YAML → tempfile → `Config::from_yaml_file`, the style used by
/// `tests/ddr_match_flag.rs`. The filename is unique per call because cargo
/// runs the tests in this file concurrently.
fn try_load_cfg(yaml: &str) -> Result<Config, String> {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "ddrs_subdivision_{}_{}.yaml",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, yaml).unwrap();
    Config::from_yaml_file(&path).map_err(|e| e.to_string())
}

fn load_cfg(yaml: &str) -> Config {
    try_load_cfg(yaml).expect("parse")
}

#[test]
fn subdivision_defaults_to_disabled() {
    let cfg = load_cfg(&yaml_with_params(""));
    assert!(!cfg.params.subdivision.enabled);
    assert_eq!(cfg.params.subdivision.max_pieces, 8);
}

#[test]
fn subdivision_rejects_max_pieces_below_one() {
    let err = try_load_cfg(&yaml_with_params(
        "  subdivision:\n    enabled: true\n    max_pieces: 0\n",
    ))
    .expect_err("must reject");
    assert!(err.to_string().contains("max_pieces"), "got: {err}");
}

#[test]
fn subdivision_enabled_loads() {
    let cfg = load_cfg(&yaml_with_params(
        "  subdivision:\n    enabled: true\n    max_pieces: 4\n",
    ));
    assert!(cfg.params.subdivision.enabled);
    assert_eq!(cfg.params.subdivision.max_pieces, 4);
}
