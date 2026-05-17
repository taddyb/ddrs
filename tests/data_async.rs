//! Integration tests for SP-2 icechunk readers, exercised against the
//! production stores under `/mnt/ssd1/data/icechunk/`.
//!
//! Tests skip with `eprintln!` if the production stores are absent so a
//! clean machine still passes.

use std::path::Path;

use chrono::NaiveDate;
use rand::SeedableRng;

use ddrs::data::dates::TimeAxis;
use ddrs::data::ids::Comid;
use ddrs::data::{ConusAdjacencyStore, StreamflowStore};

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
