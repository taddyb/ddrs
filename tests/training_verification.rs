//! SP-4 verification ladder: V1 (small batch) + V2 (all gauges) + V3 (full loop).
//!
//! V1: 8-gauge × 90-day batch with frozen scalar parameters. Asserts ddrs's
//! per-batch L1 loss matches DDR to f32 floor (1e-5 relative). The exact
//! staids and start_day_idx are read directly from the fixture — cross-runtime
//! PRNG streams diverge, so we pin inputs to DDR's batch selection.

use std::path::Path;

use ndarray::{s, Array2};
use serde::Deserialize;

use burn::backend::NdArray;

use ddrs::config::Config;
use ddrs::data::dataset::MeritGagesDataset;
use ddrs::data::{RhoWindow, Staid};
use ddrs::training::{
    filter_nan_gauges, forward_with_frozen_params, l1_loss_post_warmup, tau_trim_and_downsample,
    FrozenParams,
};

// ---------------------------------------------------------------------------
// Fixture schema
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct DdrLossFixture {
    #[allow(dead_code)]
    seed: u64,
    #[allow(dead_code)]
    batch_size: usize,
    rho: usize,
    start_day_idx: usize,
    n_active: usize,
    num_gauges: usize,
    loss: f32,
    staids: Vec<String>,
}

fn load_fixture(path: &str) -> Option<DdrLossFixture> {
    if !Path::new(path).exists() {
        eprintln!("skipping: {path} not present");
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn all_paths_exist(cfg: &Config) -> bool {
    let Some(ds) = cfg.data_sources.as_ref() else {
        return false;
    };
    [
        &ds.attributes,
        &ds.conus_adjacency,
        &ds.gages_adjacency,
        &ds.streamflow,
        &ds.observations,
        &ds.gages,
    ]
    .iter()
    .all(|p| p.exists())
}

// ---------------------------------------------------------------------------
// V1 test
// ---------------------------------------------------------------------------

#[test]
fn v1_loss_matches_ddr_for_frozen_constant_params_small_batch() {
    // 1. Load fixture (skip if absent — not always present in CI).
    let fixture = match load_fixture("fixtures/sp4/v1_ddr_loss.json") {
        Some(f) => f,
        None => return,
    };

    // 2. Load config (skip if absent or data paths missing).
    let cfg_path = "config/merit_training.yaml";
    if !Path::new(cfg_path).exists() {
        eprintln!("skipping: {cfg_path} not present");
        return;
    }
    let cfg = match Config::from_yaml_file(cfg_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: config load failed: {e}");
            return;
        }
    };
    if !all_paths_exist(&cfg) {
        eprintln!("skipping: one or more data paths absent");
        return;
    }

    // 3. Open dataset.
    let dataset = MeritGagesDataset::open(&cfg).expect("open MeritGagesDataset");

    // 4. Convert fixture staids.
    let batch_staids: Vec<Staid> = fixture
        .staids
        .iter()
        .map(|s| Staid::new(s))
        .collect();

    // 5. Build RhoWindow directly from fixture (DO NOT call sample_rho_window).
    let time_axis = dataset.time_axis();
    let window = RhoWindow {
        start_day_idx: fixture.start_day_idx,
        rho_days: fixture.rho,
        window_start: time_axis.start
            + chrono::Duration::days(fixture.start_day_idx as i64),
    };

    // 6. Collate the batch.
    let batch = dataset
        .collate(&batch_staids, &window)
        .expect("collate batch");

    // Sanity assertions — if these fail, dataset filtering changed since fixture was made.
    assert_eq!(
        batch.adjacency.n, fixture.n_active,
        "n_active mismatch: ddrs={}, fixture={}",
        batch.adjacency.n, fixture.n_active
    );
    assert_eq!(
        batch.gauge_staids.len(),
        fixture.num_gauges,
        "num_gauges mismatch: ddrs={}, fixture={}",
        batch.gauge_staids.len(),
        fixture.num_gauges
    );

    // 7. Lift onto NdArray backend.
    let device = <NdArray<f32> as burn::tensor::backend::BackendTypes>::Device::default();
    let tensors = batch.to_tensors::<NdArray<f32>>(&device);

    let num_gauges = tensors.num_gauges;

    // 8. Build FrozenParams (uniform constants across all reaches).
    let frozen = FrozenParams::constant(tensors.adjacency.n);

    // 9. Forward pass → (num_gauges, T_hours).
    let pred_hourly =
        forward_with_frozen_params::<NdArray<f32>>(&cfg, &tensors, &frozen, &device, false);

    // 10. Tau-trim + daily downsample → (num_gauges, T_days).
    let pred_daily = tau_trim_and_downsample(pred_hourly, cfg.params.tau);
    let [_g, t_days] = pred_daily.dims();

    // 11. Convert BURN tensor → ndarray::Array2 (row-major, shape (G, T_days)).
    let v: Vec<f32> = pred_daily.into_data().into_vec().unwrap();
    let daily_arr = Array2::from_shape_vec((num_gauges, t_days), v)
        .expect("reshape daily predictions");

    // 12. Trim observations: (rho_days, G) → (rho_days - 2, G) = (T_days, G).
    //     Mirrors DDR's obs[:, 1:-1] semantics (transposed: our axis 0 is time).
    let observations_trimmed = tensors
        .observations
        .slice(s![1..-1_isize, ..])
        .to_owned();

    assert_eq!(
        observations_trimmed.shape()[0],
        t_days,
        "trimmed obs T_days mismatch: obs={}, pred={}",
        observations_trimmed.shape()[0],
        t_days
    );

    // 13. Filter NaN gauges.
    //     daily_arr: (G, T_days), observations_trimmed: (T_days, G).
    let filtered = filter_nan_gauges(&daily_arr, &observations_trimmed);

    // 14. L1 loss post warmup.
    let warmup = cfg.experiment.as_ref().unwrap().warmup;
    let loss_ddrs = l1_loss_post_warmup(&filtered.predictions, &filtered.observations, warmup);

    // 15. Compare.
    let rel_diff = (loss_ddrs - fixture.loss).abs() / fixture.loss.abs();
    eprintln!(
        "V1: ddrs={loss_ddrs}, DDR={}, rel={rel_diff}",
        fixture.loss
    );
    assert!(
        rel_diff < 1e-5,
        "V1 loss diverged: ddrs={loss_ddrs}, DDR={}, rel={rel_diff}",
        fixture.loss
    );
}

// ---------------------------------------------------------------------------
// V2 test
// ---------------------------------------------------------------------------

#[test]
fn v2_loss_matches_ddr_for_frozen_constant_params_all_gauges() {
    // 1. Load fixture (skip if absent — not always present in CI).
    let fixture = match load_fixture("fixtures/sp4/v2_ddr_loss.json") {
        Some(f) => f,
        None => return,
    };

    // 2. Load config (skip if absent or data paths missing).
    let cfg_path = "config/merit_training.yaml";
    if !Path::new(cfg_path).exists() {
        eprintln!("skipping: {cfg_path} not present");
        return;
    }
    let cfg = match Config::from_yaml_file(cfg_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: config load failed: {e}");
            return;
        }
    };
    if !all_paths_exist(&cfg) {
        eprintln!("skipping: one or more data paths absent");
        return;
    }

    // 3. Open dataset.
    let dataset = MeritGagesDataset::open(&cfg).expect("open MeritGagesDataset");

    // Sanity assertion: filter pipeline must not have drifted since fixture was made.
    assert_eq!(
        dataset.len(),
        fixture.batch_size,
        "dataset.len() mismatch: ddrs={}, fixture={}; SP-1/SP-2/SP-3 filter pipeline has drifted",
        dataset.len(),
        fixture.batch_size
    );

    // 4. Convert ALL fixture staids (the full filtered gauge list).
    let batch_staids: Vec<Staid> = fixture
        .staids
        .iter()
        .map(|s| Staid::new(s))
        .collect();

    // 5. Build RhoWindow directly from fixture (DO NOT call sample_rho_window).
    let time_axis = dataset.time_axis();
    let window = RhoWindow {
        start_day_idx: fixture.start_day_idx,
        rho_days: fixture.rho,
        window_start: time_axis.start
            + chrono::Duration::days(fixture.start_day_idx as i64),
    };

    // 6. Collate the batch.
    let batch = dataset
        .collate(&batch_staids, &window)
        .expect("collate batch");

    // Sanity assertions — if these fail, dataset filtering changed since fixture was made.
    assert_eq!(
        batch.adjacency.n, fixture.n_active,
        "n_active mismatch: ddrs={}, fixture={}",
        batch.adjacency.n, fixture.n_active
    );
    assert_eq!(
        batch.gauge_staids.len(),
        fixture.num_gauges,
        "num_gauges mismatch: ddrs={}, fixture={}",
        batch.gauge_staids.len(),
        fixture.num_gauges
    );

    // 7. Lift onto NdArray backend.
    let device = <NdArray<f32> as burn::tensor::backend::BackendTypes>::Device::default();
    let tensors = batch.to_tensors::<NdArray<f32>>(&device);

    let num_gauges = tensors.num_gauges;

    // 8. Build FrozenParams (uniform constants across all reaches).
    let frozen = FrozenParams::constant(tensors.adjacency.n);

    // 9. Forward pass → (num_gauges, T_hours).
    let pred_hourly =
        forward_with_frozen_params::<NdArray<f32>>(&cfg, &tensors, &frozen, &device, false);

    // 10. Tau-trim + daily downsample → (num_gauges, T_days).
    let pred_daily = tau_trim_and_downsample(pred_hourly, cfg.params.tau);
    let [_g, t_days] = pred_daily.dims();

    // 11. Convert BURN tensor → ndarray::Array2 (row-major, shape (G, T_days)).
    let v: Vec<f32> = pred_daily.into_data().into_vec().unwrap();
    let daily_arr = Array2::from_shape_vec((num_gauges, t_days), v)
        .expect("reshape daily predictions");

    // 12. Trim observations: (rho_days, G) → (rho_days - 2, G) = (T_days, G).
    //     Mirrors DDR's obs[:, 1:-1] semantics (transposed: our axis 0 is time).
    let observations_trimmed = tensors
        .observations
        .slice(s![1..-1_isize, ..])
        .to_owned();

    assert_eq!(
        observations_trimmed.shape()[0],
        t_days,
        "trimmed obs T_days mismatch: obs={}, pred={}",
        observations_trimmed.shape()[0],
        t_days
    );

    // 13. Filter NaN gauges.
    //     daily_arr: (G, T_days), observations_trimmed: (T_days, G).
    let filtered = filter_nan_gauges(&daily_arr, &observations_trimmed);

    // 14. L1 loss post warmup.
    let warmup = cfg.experiment.as_ref().unwrap().warmup;
    let loss_ddrs = l1_loss_post_warmup(&filtered.predictions, &filtered.observations, warmup);

    // 15. Compare — tolerance is 1e-4 to allow CONUS-scale f32 accumulation drift.
    let rel_diff = (loss_ddrs - fixture.loss).abs() / fixture.loss.abs();
    eprintln!(
        "V2: ddrs={loss_ddrs}, DDR={}, rel={rel_diff}",
        fixture.loss
    );
    assert!(
        rel_diff < 1e-4,
        "V2 loss diverged: ddrs={loss_ddrs}, DDR={}, rel={rel_diff}",
        fixture.loss
    );
}

// ---------------------------------------------------------------------------
// V3 test
// ---------------------------------------------------------------------------

#[test]
fn v3_train_one_epoch_runs_end_to_end() {
    use burn::backend::{Autodiff, NdArray};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use ddrs::nn::mlp::MlpConfig;
    use ddrs::training::{TrainState, train, build_adam};

    type I = NdArray<f32>;
    type AB = Autodiff<I>;

    let cfg_path = "config/merit_training.yaml";
    if !std::path::Path::new(cfg_path).exists() {
        eprintln!("skipping: {cfg_path} not present");
        return;
    }
    let mut cfg = match Config::from_yaml_file(cfg_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: config load failed: {e}");
            return;
        }
    };
    if !all_paths_exist(&cfg) {
        eprintln!("skipping: one or more data paths absent");
        return;
    }

    // Force CI-friendly knobs.
    {
        let exp = cfg.experiment.as_mut().expect("experiment");
        exp.epochs = 1;
        exp.batch_size = 4;
    }

    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let dataset = MeritGagesDataset::open(&cfg).expect("open dataset");

    let mlp_section = cfg.mlp.as_ref().expect("mlp config");
    let mlp_cfg = MlpConfig::new(
        mlp_section.input_var_names.clone(),
        mlp_section.learnable_parameters.clone(),
    )
    .with_hidden_size(mlp_section.hidden_size)
    .with_num_hidden_layers(mlp_section.num_hidden_layers);
    let mlp = mlp_cfg.init::<AB>(&device);

    let mut state = TrainState::<I> {
        mlp,
        epoch: 1,
        mini_batch: 0,
        rng: StdRng::seed_from_u64(42),
    };
    let mut optimizer = build_adam::<ddrs::nn::mlp::Mlp<AB>, AB>();

    let ckpt_dir = std::path::PathBuf::from("/tmp/ddrs_v3_ckpts");
    let _ = std::fs::remove_dir_all(&ckpt_dir);
    std::fs::create_dir_all(&ckpt_dir).expect("ckpt dir");

    // Run only 3 mini-batches so the test finishes in seconds rather than hours.
    const MAX_MB_FOR_V3: usize = 3;
    train::<I>(&cfg, &dataset, &mut state, &mut optimizer, &device, &ckpt_dir, Some(MAX_MB_FOR_V3))
        .expect("V3 train run");

    // Bar 1: training advanced state past start.
    assert!(
        state.epoch >= 2 || state.mini_batch > 0,
        "training loop didn't advance state (epoch={}, mb={})",
        state.epoch,
        state.mini_batch
    );
    // Bar 2: at least one checkpoint exists.
    let entries: Vec<_> = std::fs::read_dir(&ckpt_dir)
        .expect("ckpt dir missing")
        .collect();
    assert!(!entries.is_empty(), "no checkpoints written");
}
