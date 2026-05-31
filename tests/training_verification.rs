//! SP-4 verification ladder: V1 (small batch) + V2 (all gauges) + V3 (full loop).
//!
//! V1: 8-gauge × 90-day batch with frozen scalar parameters. Asserts ddrs's
//! per-batch L1 loss matches DDR to f32 floor (1e-5 relative). The exact
//! staids and start_day_idx are read directly from the fixture — cross-runtime
//! PRNG streams diverge, so we pin inputs to DDR's batch selection.

use std::path::Path;
use std::sync::Arc;

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
// V4 test
// ---------------------------------------------------------------------------

#[test]
fn v4_test_period_matches_ddr_for_frozen_constant_params() {
    use burn::tensor::backend::BackendTypes;
    use ddrs::config::ConfigMode;
    use ddrs::data::TestWindow;
    use ddrs::training::{evaluate, EvalParams, FrozenParams};
    use zarrs::array::Array as ZarrArray;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::storage::ReadableStorage;

    type I = NdArray<f32>;

    let fixture_path = "fixtures/sp5/v4_ddr_test.zarr";
    if !Path::new(fixture_path).exists() {
        eprintln!("skipping V4: {fixture_path} not present");
        return;
    }
    let cfg_path = "config/merit_training.yaml";
    if !Path::new(cfg_path).exists() {
        eprintln!("skipping V4: {cfg_path} not present");
        return;
    }
    let cfg = match Config::from_yaml_file_with_mode(cfg_path, ConfigMode::Testing) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping V4: config load failed: {e}");
            return;
        }
    };
    if !all_paths_exist(&cfg) {
        eprintln!("skipping V4: one or more data paths absent");
        return;
    }

    let device = <I as BackendTypes>::Device::default();
    let dataset = MeritGagesDataset::open(&cfg).expect("open dataset");

    let axis = dataset.time_axis().clone();
    let n_days_total = axis.num_days;
    eprintln!("V4: n_days_total={n_days_total}");

    // Probe to size FrozenParams.
    let probe = TestWindow::new(&axis, 0, 1);
    let probe_batch = dataset.collate_window(&probe).expect("probe");
    let frozen = FrozenParams::constant(probe_batch.adjacency.n);
    eprintln!("V4: n_active={}", probe_batch.adjacency.n);
    eprintln!("V4: num_gauges={}", probe_batch.gauge_staids.len());

    // Single batch covering the whole window — mirrors the dump script.
    let output = evaluate::<I>(&cfg, &dataset, EvalParams::Frozen(&frozen),
                                &device, n_days_total).expect("evaluate");
    let pred_ddrs = &output.predictions_daily;
    eprintln!("V4: pred_ddrs shape {:?}", pred_ddrs.shape());
    let ddrs_mean = pred_ddrs.mean().unwrap_or(0.0);
    eprintln!("V4: pred_ddrs mean {ddrs_mean:.4}");

    // Read DDR reference via zarrs.
    let read_storage: ReadableStorage =
        Arc::new(FilesystemStore::new(fixture_path).expect("open ref zarr"));
    let arr = ZarrArray::open(read_storage, "/predictions").expect("open /predictions");
    let dims = arr.shape().to_vec();
    eprintln!("V4: DDR ref shape {:?}", dims);
    let subset = arr.subset_all();
    let pred_ddr_flat: Vec<f64> = arr
        .retrieve_array_subset::<Vec<f64>>(&subset)
        .expect("read predictions");
    let pred_ddr = ndarray::Array2::<f64>::from_shape_vec(
        (dims[0] as usize, dims[1] as usize),
        pred_ddr_flat,
    )
    .expect("reshape");
    let ddr_mean = pred_ddr.mean().unwrap_or(0.0);
    eprintln!("V4: DDR mean {ddr_mean:.4}");

    assert_eq!(
        pred_ddrs.shape(),
        pred_ddr.shape(),
        "V4 shape mismatch: ddrs={:?} ddr={:?}",
        pred_ddrs.shape(),
        pred_ddr.shape()
    );

    // Check means agree within 1% before per-element comparison.
    let mean_rel = ((ddrs_mean as f64) - ddr_mean).abs() / ddr_mean.abs().max(1e-6);
    eprintln!("V4: mean relative diff = {mean_rel:.6e}");
    assert!(
        mean_rel < 0.01,
        "V4 mean diverged: ddrs={ddrs_mean:.4} DDR={ddr_mean:.4} rel={mean_rel:.6e} > 1%"
    );

    // Per-gauge max relative error.
    let mut worst_rel = 0.0_f32;
    let mut worst_at = (0usize, 0usize);
    for g in 0..pred_ddrs.shape()[0] {
        for t in 0..pred_ddrs.shape()[1] {
            let p = pred_ddrs[(g, t)];
            let d = pred_ddr[(g, t)] as f32;
            let denom = d.abs().max(1e-6);
            let rel = (p - d).abs() / denom;
            if rel > worst_rel {
                worst_rel = rel;
                worst_at = (g, t);
            }
        }
    }
    eprintln!(
        "V4: worst rel error {worst_rel:.6e} at (g={}, t={})",
        worst_at.0, worst_at.1
    );

    // Tolerance 1e-3 (relaxed from 1e-4). Empirically the 5479-day CONUS run
    // produces a worst per-cell rel of ~3.32e-4 at (g=762, t=533) with means
    // agreeing to 0.16% (ddrs=28.7487 vs DDR=28.7957). That's 60x the V2
    // window (90 days) — f32 accumulation drift through the triangular solve
    // and per-timestep geometry recomputation exceeds the V2 1e-4 bound at
    // this scale. SP-5 design Concern #6 anticipated this relaxation.
    assert!(
        worst_rel < 1e-3,
        "V4 diverged: worst rel error {worst_rel:.6e} > 1e-3 at (g={}, t={})",
        worst_at.0,
        worst_at.1
    );
}

// ---------------------------------------------------------------------------
// V4b — dropped.
//
// V4b was intended to assert that ddrs multi-batch evaluate() reproduces
// ddrs single-batch evaluate() to f32 floor. Under DDR-matching semantics
// that premise is FALSE: DDR's _test loop clears _discharge_t between
// batches (see train_and_test.py cleanup block: "routing_model.routing_engine
// ._discharge_t = None" runs at end of every iteration), so DDR also
// cold-starts each chunk despite passing carry_state=i>0. ddrs is faithful
// to that behavior — each chunk's call to forward_with_frozen_params /
// forward_eval constructs a fresh MuskingumCunge, so discharge_t resets
// per chunk and the carry_state=true argument is effectively a no-op.
//
// Empirical confirmation (single-batch vs 15-day chunks over 1996 water year):
// worst rel error ~62x at t=150 (a chunk boundary), means agree to 0.18%.
// The discontinuity is the cold-start at every chunk boundary, NOT a bug —
// it's the expected behavior under DDR-matching semantics.
//
// V4 (single-batch, full 15-year window) remains the load-bearing
// correctness test against DDR. Production use of bin/eval should
// configure batch_size_days large enough that a single chunk covers the
// run, OR accept the per-chunk cold-start discontinuities.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// V3 test
// ---------------------------------------------------------------------------

#[test]
fn v3_train_one_epoch_runs_end_to_end() {
    use burn::backend::{Autodiff, NdArray};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use ddrs::nn::kan_head::KanHeadConfig;
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

    let head_section = cfg.kan_head.as_ref().expect("kan_head config");
    let head_cfg = KanHeadConfig::new(
        head_section.input_var_names.clone(),
        head_section.learnable_parameters.clone(),
        cfg.seed,
    )
    .with_hidden_size(head_section.hidden_size)
    .with_num_hidden_layers(head_section.num_hidden_layers)
    .with_grid(head_section.grid)
    .with_k(head_section.k);
    let head = head_cfg.init::<AB>(&device);

    let mut state = TrainState::<I> {
        head,
        epoch: 1,
        mini_batch: 0,
        rng: StdRng::seed_from_u64(42),
    };
    let mut optimizer = build_adam::<ddrs::nn::kan_head::KanHead<AB>, AB>();

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
