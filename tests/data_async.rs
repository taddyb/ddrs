//! Integration tests for SP-2 icechunk readers, exercised against the
//! production stores under `/mnt/ssd1/data/icechunk/`.
//!
//! Tests skip with `eprintln!` if the production stores are absent so a
//! clean machine still passes.

use std::path::Path;

use chrono::NaiveDate;
use rand::SeedableRng;

use ddrs::data::dates::TimeAxis;
use ddrs::data::error::DataError;
use ddrs::data::ids::{Comid, Staid};
use ddrs::data::{ConusAdjacencyStore, GageMetadata, StreamflowStore, UsgsObservationsStore};

const STREAMFLOW_IC: &str =
    "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic";
const CONUS_ADJ: &str =
    "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr";

#[test]
fn streamflow_store_reads_known_window() {
    if !Path::new(STREAMFLOW_IC).exists() || !Path::new(CONUS_ADJ).exists() {
        eprintln!("skipping: production streamflow / adjacency files absent");
        return;
    }
    let conus = ConusAdjacencyStore::open(CONUS_ADJ).expect("conus");
    let comids: Vec<Comid> = conus.order.iter().take(50).copied().collect();

    let store = StreamflowStore::open(STREAMFLOW_IC).expect("open streamflow");

    // Window over 1981-10-01 .. 1981-12-31 (within MERIT training period),
    // sample a 90-day rho window.
    let axis = TimeAxis::new(
        NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
        NaiveDate::from_ymd_opt(1981, 12, 31).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let window = axis.sample_rho_window(&mut rng, 90);

    let q_prime = store.read_window(&window, &comids).expect("read window");

    assert_eq!(q_prime.shape(), &[window.n_hourly(), 50]);
    // All values finite (either real q' or the 0.001 fill).
    for &v in q_prime.iter() {
        assert!(v.is_finite(), "got non-finite q': {v}");
    }
    // The 0.001 fill should not dominate — most of the first 50 COMIDs
    // from CONUS adjacency have DHBv2 coverage.
    let nonfill = q_prime.iter().filter(|&&v| (v - 0.001).abs() > 1e-9).count();
    assert!(
        nonfill > q_prime.len() / 2,
        "too many fill values: nonfill={nonfill}, total={}",
        q_prime.len()
    );
}

#[test]
fn streamflow_missing_divides_get_filled() {
    if !Path::new(STREAMFLOW_IC).exists() {
        eprintln!("skipping: streamflow not present");
        return;
    }
    let store = StreamflowStore::open(STREAMFLOW_IC).expect("open");

    // A real COMID from the design-time probe and a fake one.
    let comids = vec![Comid(71024425), Comid(-1)];
    let axis = TimeAxis::new(
        NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
        NaiveDate::from_ymd_opt(1981, 12, 31).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let window = axis.sample_rho_window(&mut rng, 90);

    let q = store.read_window(&window, &comids).expect("read");
    // Column 1 (Comid(-1)) is entirely the 0.001 fill.
    for h in 0..window.n_hourly() {
        assert!(
            (q[(h, 1)] - 0.001).abs() < 1e-9,
            "expected 0.001 fill at ({h},1), got {}",
            q[(h, 1)]
        );
    }
}

const OBS_IC: &str = "/mnt/ssd1/data/icechunk/usgs_daily_observations";
const GAGES_CSV: &str =
    "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";

#[test]
fn observations_store_reads_known_window() {
    if !Path::new(OBS_IC).exists() || !Path::new(GAGES_CSV).exists() {
        eprintln!("skipping: observations / gages files absent");
        return;
    }
    let gages = GageMetadata::open(GAGES_CSV).expect("gages");

    // The first 10 STAIDs from gages_3000.csv may or may not be in the
    // obs store. Filter to the ones that are.
    let store = UsgsObservationsStore::open(OBS_IC).expect("open obs");
    let staids: Vec<Staid> = gages
        .staids()
        .into_iter()
        .filter(|s| store.index.contains(s))
        .take(10)
        .collect();
    assert!(
        !staids.is_empty(),
        "expected at least one gages_3000 STAID present in obs store"
    );

    let axis = TimeAxis::new(
        NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
        NaiveDate::from_ymd_opt(1981, 12, 31).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let window = axis.sample_rho_window(&mut rng, 90);

    let obs = store.read_window(&window, &staids).expect("read obs");

    assert_eq!(obs.shape(), &[window.rho_days, staids.len()]);
    // At least one finite value across the requested gauges × 90 days.
    let finite_count = obs.iter().filter(|v| v.is_finite()).count();
    assert!(
        finite_count > 0,
        "expected some finite obs values, got 0 across {} cells",
        obs.len()
    );
}

#[test]
fn observations_missing_gauges_errors() {
    if !Path::new(OBS_IC).exists() {
        eprintln!("skipping: observations not present");
        return;
    }
    let store = UsgsObservationsStore::open(OBS_IC).expect("open");

    let axis = TimeAxis::new(
        NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
        NaiveDate::from_ymd_opt(1981, 12, 31).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let window = axis.sample_rho_window(&mut rng, 90);

    let bogus = vec![Staid::new("99999999")];
    let err = store.read_window(&window, &bogus).unwrap_err();
    match err {
        DataError::MissingIds {
            kind,
            missing,
            total,
            ..
        } => {
            assert_eq!(kind, "gage_id");
            assert_eq!(missing, 1);
            assert_eq!(total, 1);
        }
        other => panic!("expected MissingIds, got {other:?}"),
    }
}
