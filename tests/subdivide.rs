//! `params.subdivision` — static reach subdivision (variable Δx).
//!
//! Defaults to disabled so every existing config keeps its current behaviour.
use ddrs::adjacency::subdivide::{plan_reaches, reference_celerity};
use ddrs::config::{Config, Subdivision};
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

// ---------------------------------------------------------------------------
// Task 2: reference celerity + the two-sided reach plan
// ---------------------------------------------------------------------------

fn cfg(max_pieces: usize) -> Subdivision {
    Subdivision {
        enabled: true,
        max_pieces,
        ..Default::default()
    }
}

#[test]
fn celerity_rises_with_slope_and_area() {
    let c = cfg(8);
    let lo = reference_celerity(100.0, 1e-3, &c);
    assert!(
        reference_celerity(100.0, 1e-2, &c) > lo,
        "steeper must be faster"
    );
    assert!(
        reference_celerity(10_000.0, 1e-3, &c) > lo,
        "bigger must be faster"
    );
    assert!(
        lo > 0.0 && lo < 15.0,
        "celerity {lo} outside physical range"
    );
}

#[test]
fn long_reaches_split_and_are_capped() {
    let c = cfg(4);
    let p = plan_reaches(&[200_000.0], &[1e-4], &[30.0], 3600.0, &c);
    assert_eq!(p.pieces[0], 4, "must clamp to max_pieces");
    assert_eq!(
        p.length_m[0], 200_000.0,
        "long reaches keep their true length"
    );
}

#[test]
fn short_reaches_are_length_clamped_not_split() {
    let c = cfg(8);
    // 50 m reach with a fast celerity: far below dx_target, so it stretches.
    let p = plan_reaches(&[50.0], &[1e-2], &[10_000.0], 3600.0, &c);
    assert_eq!(p.pieces[0], 1, "short reach must not split");
    let dx = reference_celerity(10_000.0, 1e-2, &c) * 3600.0;
    assert!(
        (p.length_m[0] - dx).abs() < 1e-3,
        "expected clamp to dx_target {dx}, got {}",
        p.length_m[0]
    );
    assert!(p.length_m[0] > 50.0, "clamp must lengthen, not shorten");
}

#[test]
fn min_length_fraction_zero_disables_the_clamp() {
    let mut c = cfg(8);
    c.min_length_fraction = 0.0;
    let p = plan_reaches(&[50.0], &[1e-2], &[10_000.0], 3600.0, &c);
    assert_eq!(p.length_m[0], 50.0, "clamp must be off");
}

#[test]
fn disabled_is_an_exact_no_op() {
    let mut c = cfg(8);
    c.enabled = false;
    let p = plan_reaches(
        &[200_000.0, 50.0],
        &[1e-4, 1e-2],
        &[30.0, 10_000.0],
        3600.0,
        &c,
    );
    assert_eq!(p.pieces, vec![1, 1]);
    assert_eq!(
        p.length_m,
        vec![200_000.0, 50.0],
        "lengths must be untouched"
    );
}

#[test]
fn degenerate_input_never_yields_zero_pieces_or_zero_length() {
    let c = cfg(8);
    let p = plan_reaches(&[5000.0, 0.0], &[0.0, 0.0], &[0.0, 0.0], 3600.0, &c);
    assert!(
        p.pieces.iter().all(|&v| v >= 1),
        "pieces must be >= 1, got {:?}",
        p.pieces
    );
    assert!(
        p.length_m.iter().all(|&v| v > 0.0),
        "length must be > 0 (a 0 m reach gives K = 0 and c1 = 1), got {:?}",
        p.length_m
    );
}
