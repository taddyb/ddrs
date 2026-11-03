//! `params.geometry_form` config surface: parsing, default and load-time guards.
use ddrs::config::{Config, GeometryForm};

const RANGES: &str = "  parameter_ranges:\n    n: [0.015, 0.25]\n    q_spatial: [0.0, 1.0]\n    p_spatial: [1.0, 200.0]\n";

fn load(tag: &str, params_extra: &str) -> Result<Config, String> {
    let yaml = format!(
        "mode: training\ngeodataset: merit\nseed: 42\nnp_seed: 42\nparams:\n{params_extra}{RANGES}"
    );
    let path = std::env::temp_dir().join(format!("ddrs_geometry_form_{tag}.yaml"));
    std::fs::write(&path, yaml).unwrap();
    Config::from_yaml_file(&path).map_err(|e| e.to_string())
}

fn expect_err(tag: &str, extra: &str, key: &str) {
    let msg = load(tag, extra).expect_err("must be rejected");
    assert!(msg.contains("geometry_form"), "message lacks geometry_form: {msg}");
    assert!(msg.contains(key), "message lacks {key}: {msg}");
}

#[test]
fn absent_is_trapezoid() {
    let cfg = load("absent", "").expect("parse");
    assert_eq!(cfg.params.geometry_form, GeometryForm::Trapezoid);
    assert_eq!(cfg.params.geometry_form.name(), "trapezoid");
}

#[test]
fn power_law_parses() {
    let cfg = load("power_law", "  geometry_form: power_law\n").expect("parse");
    assert_eq!(cfg.params.geometry_form, GeometryForm::PowerLaw);
    assert_eq!(cfg.params.geometry_form.name(), "power_law");
}

#[test]
fn typo_is_rejected() {
    let msg = load("typo", "  geometry_form: power-law\n").expect_err("typo");
    assert!(msg.contains("power_law"), "message should list valid values: {msg}");
}

#[test]
fn power_law_rejects_ddr_match() {
    expect_err("ddr_match", "  geometry_form: power_law\n  ddr_match: true\n", "ddr_match");
}

#[test]
fn power_law_rejects_cuda_graphs() {
    expect_err(
        "cuda_graphs",
        "  geometry_form: power_law\n  ddr_match: false\n  use_cuda_graphs: true\n",
        "use_cuda_graphs",
    );
}

#[test]
fn power_law_loads_with_leakance() {
    let cfg = load("leakance", "  geometry_form: power_law\n  use_leakance: true\n").expect("loads");
    assert!(cfg.params.use_leakance);
}

#[test]
fn power_law_loads_with_reservoirs() {
    // `use_reservoirs` needs a dam table in `data_sources.reservoirs`.
    let yaml = format!(
        "mode: training\ngeodataset: merit\nseed: 42\nnp_seed: 42\ndata_sources:\n  attributes: [a.nc]\n  streamflow: s\n  observations: o\n  gages: g.csv\n  geospatial_fabric: f.shp\n  reservoirs: /tmp/ddrs_dams.csv\nparams:\n  geometry_form: power_law\n  use_reservoirs: true\n{RANGES}"
    );
    let path = std::env::temp_dir().join("ddrs_geometry_form_reservoirs.yaml");
    std::fs::write(&path, yaml).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("loads");
    assert!(cfg.params.use_reservoirs);
}

#[test]
fn power_law_rejects_zero_slope_minimum() {
    expect_err(
        "slope_min",
        "  geometry_form: power_law\n  attribute_minimums:\n    slope: 0.0\n",
        "attribute_minimums.slope",
    );
}

#[test]
fn power_law_rejects_nonpositive_p_spatial_lower_bound() {
    let yaml = "mode: training\ngeodataset: merit\nseed: 42\nnp_seed: 42\nparams:\n  geometry_form: power_law\n  parameter_ranges:\n    n: [0.015, 0.25]\n    q_spatial: [0.0, 1.0]\n    p_spatial: [0.0, 200.0]\n";
    let path = std::env::temp_dir().join("ddrs_geometry_form_p_lower.yaml");
    std::fs::write(&path, yaml).unwrap();
    let msg = Config::from_yaml_file(&path).map(|_| ()).expect_err("must be rejected").to_string();
    assert!(msg.contains("geometry_form"), "message lacks geometry_form: {msg}");
    assert!(msg.contains("parameter_ranges.p_spatial"), "message lacks the key: {msg}");
}

#[test]
fn power_law_rejects_nonpositive_p_spatial_default() {
    expect_err(
        "p_default",
        "  geometry_form: power_law\n  defaults:\n    p_spatial: 0.0\n",
        "defaults.p_spatial",
    );
}

#[test]
fn trapezoid_still_loads_with_zero_slope_minimum() {
    load("trap_slope_min", "  attribute_minimums:\n    slope: 0.0\n").expect("trapezoid is unchanged");
}

#[test]
fn power_law_loads_with_the_hudson_stage_config() {
    let text = std::fs::read_to_string("examples/hudson/ddrs.yaml").expect("hudson yaml");
    assert_eq!(text.lines().filter(|l| *l == "params:").count(), 1);
    let patched = text.replace("\nparams:\n", "\nparams:\n  geometry_form: power_law\n");
    let path = std::env::temp_dir().join("ddrs_geometry_form_hudson.yaml");
    std::fs::write(&path, patched).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("hudson loads with power_law");
    assert_eq!(cfg.params.geometry_form, GeometryForm::PowerLaw);
    assert!(cfg.stage_routing_on());
}
