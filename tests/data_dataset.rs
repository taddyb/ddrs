//! V2 integration test for SP-3: open all five stores via the production
//! YAML, sample one batch via RandomSampler + RhoWindow, run `collate`,
//! sanity-check shapes + the lower-triangular invariant.
//!
//! Skip cleanly if any production data file is absent.

use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;

use ddrs::config::Config;
use ddrs::data::{MeritGagesDataset, RandomSampler};

#[test]
fn collate_one_batch_against_live_stores() {
    let cfg_path = "config/merit_training.yaml";
    if !Path::new(cfg_path).exists() {
        eprintln!("skipping: {cfg_path} not present");
        return;
    }
    let cfg = Config::from_yaml_file(cfg_path).expect("load yaml");
    let ds_paths = cfg.data_sources.as_ref().expect("data_sources");
    for p in &[
        &ds_paths.attributes,
        &ds_paths.conus_adjacency,
        &ds_paths.gages_adjacency,
        &ds_paths.streamflow,
        &ds_paths.observations,
        &ds_paths.gages,
    ] {
        if !p.exists() {
            eprintln!("skipping: {} not present", p.display());
            return;
        }
    }

    let dataset = MeritGagesDataset::open(&cfg).expect("open dataset");
    assert!(
        dataset.len() > 100,
        "expected many filtered gauges, got {}",
        dataset.len()
    );

    let mut rng = StdRng::seed_from_u64(42);
    let mut sampler = RandomSampler::new(dataset.len(), 8, true);
    sampler.reshuffle(&mut rng);
    let batch_idx = sampler.next_batch().expect("batch");
    let staids: Vec<_> = batch_idx
        .iter()
        .map(|&i| dataset.staids()[i].clone())
        .collect();
    let window = dataset.time_axis().sample_rho_window(&mut rng, 90);

    let batch = dataset.collate(&staids, &window).expect("collate");

    // Adjacency: at least as many segments as gauges (ancestry expanded),
    // but compressed below CONUS scale.
    assert!(batch.adjacency.n >= staids.len());
    assert!(batch.adjacency.n < 200_000);

    // q' shape: (T_hours, N), where T_hours = window.n_hourly().
    assert_eq!(batch.q_prime.shape(), &[window.n_hourly(), batch.adjacency.n]);

    // Normalized attrs shape: (N, F).
    assert_eq!(
        batch.spatial_attributes_normalized.shape()[0],
        batch.adjacency.n
    );

    // Observations shape: (rho_days, G).
    assert_eq!(
        batch.observations.shape(),
        &[window.rho_days, batch.gauge_staids.len()]
    );

    // Lower-triangular invariant.
    for k in 0..batch.adjacency.nnz() {
        assert!(
            batch.adjacency.rows[k] >= batch.adjacency.cols[k],
            "lower-triangular violated at nnz={k}"
        );
    }

    // outflow_idx has one entry per gauge.
    assert_eq!(batch.outflow_idx.len(), batch.gauge_staids.len());

    // flow_scale length matches adjacency.
    assert_eq!(batch.flow_scale.len(), batch.adjacency.n);

    // Compressed COMID count matches adjacency size.
    assert_eq!(batch.divide_comids.len(), batch.adjacency.n);
}
