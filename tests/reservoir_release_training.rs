//! The learned dam release wired into training and the test phase.
//!
//! 1. The training path (`apply_reservoir_rows` with the release head on the
//!    feature rows) and the test-phase path (the head resolved once into a
//!    fixed seasonal table, `release_eval::resolve_release_table`, routed
//!    through the ordinary `fixed` path) give bitwise identical routing on
//!    NdArray, for a seasonal and a non-seasonal head. (On CUDA the one-matmul
//!    test-phase resolution may differ from per-batch training at the ulp level.)
//! 2. A learned table reaching a forward that has no release head is refused
//!    instead of silently routing the dams as channels.
//! 4. Activation year: a table whose dams were all completed by the window
//!    start routes bitwise like a table without years; a dam completed after
//!    the window start is not a dam row (its reach is a channel), for the
//!    training path and for the resolved test-phase table, which carries the
//!    years.
//! 3. Bootstrap builds the release head and its optimizer for a learned
//!    config, restores both from `release_head.mpk` / `release_optim.mpk`,
//!    and starts the head cold (logged) from a checkpoint that has none.
//!    A config without the learned release gets no release head.

use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::Tensor;
use chrono::NaiveDate;

use ddrs::config::{Config, ReleaseHeadSection, ReservoirRelease};
use ddrs::data::ids::Comid;
use ddrs::data::store::{map_reservoir_rows, DamFeatures, ReservoirTable};
use ddrs::nn::release_head::init_release_head;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;
use ddrs::training::checkpoint::{release_head_base, release_optim_base};
use ddrs::training::forward::apply_reservoir_rows;
use ddrs::training::release_eval::resolve_release_table;
use ddrs::training::{bootstrap_head_and_state, save_kan_head, save_optimizer};

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const N_REACH: usize = 5;
const STEPS: usize = 96;

fn sandbox5() -> SparseAdjacency {
    let n = N_REACH;
    let mut dense = vec![0.0_f32; n * n];
    for (up, down) in [(1, 3), (2, 3), (0, 4), (3, 4)] {
        dense[down * n + up] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, vec![5000.0; n], vec![0.001; n])
}

fn network() -> Vec<Comid> {
    (0..N_REACH as i64).map(|i| Comid(100 + i)).collect()
}

fn section(seasonal: bool) -> ReleaseHeadSection {
    ReleaseHeadSection {
        hidden_size: 6,
        num_hidden_layers: 1,
        grid: 5,
        k: 3,
        input_var_names: vec!["f1".into(), "f2".into()],
        seasonal,
        routing_checkpoint: None,
        freeze_routing: false,
    }
}

fn learned_cfg(seasonal: bool) -> Config {
    let mut cfg = Config::default();
    cfg.params.use_reservoirs = true;
    cfg.params.reservoir_release = ReservoirRelease::Learned;
    cfg.release_head = Some(section(seasonal));
    cfg
}

/// Dams on reaches 3 and 4 (COMIDs 103, 104) plus one outside the network.
fn features() -> DamFeatures {
    features_with_years(vec![None, None, None])
}

/// `features()` with completion years (rows: COMID 104, 999, 103).
fn features_with_years(years: Vec<Option<i32>>) -> DamFeatures {
    DamFeatures {
        comids: vec![Comid(104), Comid(999), Comid(103)],
        names: vec!["f1".into(), "f2".into()],
        values: ndarray::array![[0.8_f32, -1.2], [0.0, 0.0], [-0.5, 1.7]],
        years,
    }
}

/// A trained-looking head: nonzero read-out so the dams differ and a, b != 0.
fn trained_head(seasonal: bool) -> ddrs::nn::KanHead<AB> {
    let device = Device::default();
    let cfg = learned_cfg(seasonal);
    let mut head = init_release_head::<AB>(&section(seasonal), &cfg.params.parameter_ranges, 42, &device);
    let [h, p] = head.output.weight.val().dims();
    let w: Vec<f32> = (0..h * p).map(|i| ((i * 7 % 5) as f32 - 2.0) * 0.3).collect();
    head.output.weight = burn::module::Param::from_tensor(
        Tensor::<AB, 1>::from_floats(w.as_slice(), &device).reshape([h, p]),
    );
    head
}

fn engine(cfg: &Config) -> MuskingumCunge<I> {
    let device = Device::default();
    let leaf = |v: f32| Tensor::<AB, 1>::from_floats(vec![v; N_REACH].as_slice(), &device);
    let mut q = Vec::new();
    for t in 0..=STEPS {
        let s = (t as f32 / 24.0 * std::f32::consts::PI).sin();
        q.extend((0..N_REACH).map(|r| (11.0 + 3.0 * r as f32) * (1.0 + s * s)));
    }
    let q = Tensor::<AB, 1>::from_floats(q.as_slice(), &device).reshape([STEPS + 1, N_REACH]);
    let mut mc = MuskingumCunge::<I>::new(cfg.clone(), device);
    mc.setup_inputs(
        RoutingInputs { adjacency: sandbox5(), x_storage: Tensor::ones([N_REACH], &device) * 0.3 },
        q,
        SpatialParameters {
            n: leaf(0.5),
            q_spatial: leaf(0.5),
            p_spatial: Some(leaf(0.575)),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        },
        false,
        None,
    );
    mc
}

fn window_start() -> NaiveDate {
    NaiveDate::from_ymd_opt(1990, 4, 20).unwrap()
}

fn assert_bitwise(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "idx {i}: {x} vs {y}");
    }
}

fn training_vs_resolved(seasonal: bool) {
    let cfg = learned_cfg(seasonal);
    let head = trained_head(seasonal);

    // Training path: the release head runs on the batch's feature rows.
    let rows = map_reservoir_rows(&ReservoirTable::Learned(features()), &network());
    assert_eq!(rows.rows, vec![3, 4]);
    let mut a = engine(&cfg);
    apply_reservoir_rows(&cfg, &mut a, Some(&rows), window_start(), STEPS + 1, Some(&head));
    let train: Vec<f32> = a.forward().into_data().to_vec().unwrap();

    // Test-phase path: resolve once on the inner backend, route as fixed.
    let table = resolve_release_table::<I>(&head.valid(), &features(), &cfg);
    assert_eq!(table.seasonal, seasonal);
    assert_eq!(table.dams.len(), 3, "every table dam is resolved, in the table's order");
    let rows_b = map_reservoir_rows(&ReservoirTable::Fixed(table), &network());
    let mut b = engine(&cfg);
    apply_reservoir_rows(&cfg, &mut b, Some(&rows_b), window_start(), STEPS + 1, None);
    let resolved: Vec<f32> = b.forward().into_data().to_vec().unwrap();

    assert_bitwise(&train, &resolved);
    let mut plain = engine(&Config::default());
    let none: Vec<f32> = plain.forward().into_data().to_vec().unwrap();
    assert!(train != none, "the learned release must change the routing");
}

#[test]
fn training_and_resolved_release_route_identically_seasonal() {
    training_vs_resolved(true);
}

#[test]
fn training_and_resolved_release_route_identically_constant() {
    training_vs_resolved(false);
}

/// Route the sandbox with the learned release on `table` through the training
/// path, window starting at `window_start()` (1990-04-20).
fn route_learned(table: DamFeatures) -> Vec<f32> {
    let cfg = learned_cfg(true);
    let head = trained_head(true);
    let rows = map_reservoir_rows(&ReservoirTable::Learned(table), &network());
    let mut e = engine(&cfg);
    apply_reservoir_rows(&cfg, &mut e, Some(&rows), window_start(), STEPS + 1, Some(&head));
    e.forward().into_data().to_vec().unwrap()
}

#[test]
fn dams_completed_by_the_window_start_route_bitwise_like_a_table_without_years() {
    let no_years = route_learned(features());
    // 1990 is the window's own year: active from 1990-01-01.
    let built = route_learned(features_with_years(vec![Some(1990), Some(2020), Some(1937)]));
    assert_bitwise(&no_years, &built);
}

#[test]
fn a_dam_completed_after_the_window_start_routes_as_a_channel() {
    // Dam 104 (row 4) completed 1991: not yet built in a window starting 1990-04-20.
    let partial = route_learned(features_with_years(vec![Some(1991), None, Some(1970)]));
    let only_103 = route_learned(DamFeatures {
        comids: vec![Comid(103)],
        names: vec!["f1".into(), "f2".into()],
        values: ndarray::array![[-0.5_f32, 1.7]],
        years: vec![None],
    });
    assert_bitwise(&partial, &only_103);

    // No dam built yet: exactly the engine with no dams at all.
    let none_built = route_learned(features_with_years(vec![Some(1991), Some(1991), Some(2001)]));
    let mut plain = engine(&Config::default());
    let no_dams: Vec<f32> = plain.forward().into_data().to_vec().unwrap();
    assert_bitwise(&none_built, &no_dams);
    assert!(partial != no_dams, "the built dam must still act");
}

#[test]
fn the_resolved_test_phase_table_carries_the_years_and_filters_the_same_way() {
    let cfg = learned_cfg(true);
    let head = trained_head(true);
    let table = resolve_release_table::<I>(
        &head.valid(),
        &features_with_years(vec![Some(1991), None, Some(1970)]),
        &cfg,
    );
    assert_eq!(
        table.dams.iter().map(|d| d.year_completed).collect::<Vec<_>>(),
        vec![Some(1991), None, Some(1970)]
    );
    let rows = map_reservoir_rows(&ReservoirTable::Fixed(table), &network());
    let mut e = engine(&cfg);
    apply_reservoir_rows(&cfg, &mut e, Some(&rows), window_start(), STEPS + 1, None);
    let resolved: Vec<f32> = e.forward().into_data().to_vec().unwrap();
    assert_bitwise(&resolved, &route_learned(features_with_years(vec![Some(1991), None, Some(1970)])));
}

#[test]
#[should_panic(expected = "release head")]
fn learned_rows_without_a_release_head_are_refused() {
    let cfg = learned_cfg(true);
    let rows = map_reservoir_rows(&ReservoirTable::Learned(features()), &network());
    let mut e = engine(&cfg);
    apply_reservoir_rows(&cfg, &mut e, Some(&rows), window_start(), STEPS + 1, None);
}

// ---------------------------------------------------------------------------
// 3. Bootstrap + checkpoint.
// ---------------------------------------------------------------------------

/// A loadable config (paths are never opened by bootstrap).
fn write_config(dir: &Path, learned: bool, checkpoint: Option<&Path>) -> Config {
    let release = if learned {
        "  use_reservoirs: true\n  reservoir_release: learned\n"
    } else {
        ""
    };
    let head = if learned {
        "release_head:\n  hidden_size: 6\n  input_var_names: [f1, f2]\n"
    } else {
        ""
    };
    let ckpt = checkpoint.map(|p| format!("  checkpoint: {}\n", p.display())).unwrap_or_default();
    let yaml = format!(
        "mode: training\nseed: 42\ndata_sources:\n  attributes: /dev/null/a.nc\n  \
         conus_adjacency: /dev/null/c.zarr\n  gages_adjacency: /dev/null/g.zarr\n  \
         streamflow: /dev/null/s.ic\n  observations: /dev/null/o.ic\n  gages: /dev/null/g.csv\n  \
         reservoirs: /dev/null/r.csv\n\
         experiment:\n  batch_size: 2\n  start_time: 1981/10/01\n  end_time: 1982/09/30\n  \
         epochs: 3\n  rho: 30\n  warmup: 2\n{ckpt}\
         kan_head:\n  hidden_size: 4\n  num_hidden_layers: 1\n  input_var_names: [aridity]\n  \
         learnable_parameters: [n, q_spatial]\n\
         {head}params:\n{release}"
    );
    let path = dir.join("cfg.yaml");
    std::fs::write(&path, yaml).unwrap();
    Config::from_yaml_file(&path).expect("config loads")
}

fn weights<B: burn::tensor::backend::Backend>(h: &ddrs::nn::KanHead<B>) -> Vec<f32> {
    h.output.weight.val().into_data().to_vec().unwrap()
}

#[test]
fn bootstrap_without_learned_release_has_no_release_head() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = write_config(dir.path(), false, None);
    let (_, state, _) = bootstrap_head_and_state::<I>(&cfg, &Device::default()).unwrap();
    assert!(state.release.is_none());
}

#[test]
fn bootstrap_restores_the_release_head_and_optimizer() {
    let device = Device::default();
    let dir = tempfile::tempdir().unwrap();
    let cfg = write_config(dir.path(), true, None);
    let (_, state, _) = bootstrap_head_and_state::<I>(&cfg, &device).unwrap();
    let mut release = state.release.expect("learned config builds a release head");
    let init = weights(&release.head);
    assert!(init.iter().all(|&w| w == 0.0), "fresh read-out is zero");

    // One optimizer step so both the weights and the Adam moments move.
    let feats = Tensor::<AB, 2>::from_floats([[0.8_f32, -1.2], [0.1, 0.4]], &device);
    let loss = release.head.forward(feats)["T0"].clone().sum();
    let grads = GradientsParams::from_grads(loss.backward(), &release.head);
    release.head = release.optimizer.step(1e-2, release.head.clone(), grads);
    let stepped = weights(&release.head);
    assert!(stepped.iter().any(|&w| w != 0.0));

    let ckpt = dir.path().join("epoch_1_mb_0");
    std::fs::create_dir_all(&ckpt).unwrap();
    save_kan_head(&release_head_base(&ckpt), &release.head.clone().valid()).unwrap();
    save_optimizer(&release_optim_base(&ckpt), &release.optimizer).unwrap();
    // The routing head's files, so the resume finds a complete checkpoint.
    save_kan_head(&ddrs::training::head_base(&ckpt), &state.head.clone().valid()).unwrap();

    let cfg2 = write_config(dir.path(), true, Some(&ckpt));
    let (_, state2, _) = bootstrap_head_and_state::<I>(&cfg2, &device).unwrap();
    let restored = state2.release.expect("release head restored");
    let got = weights(&restored.head);
    for (g, w) in got.iter().zip(&stepped) {
        // CompactRecorder stores f16.
        assert!((g - w).abs() <= 1e-3 * w.abs().max(1.0), "restored {g} vs saved {w}");
    }
    // The optimizer moments came back too: a second step with a DIFFERENT
    // gradient (with an identical one, a cold Adam steps the same) moves the
    // restored head like the original, and unlike a cold optimizer.
    let second_step = |head: ddrs::nn::KanHead<AB>,
                       mut opt: ddrs::training::HeadOptimizer<ddrs::nn::KanHead<AB>, AB>| {
        let feats = Tensor::<AB, 2>::from_floats([[-1.5_f32, 0.3], [2.0, 2.0]], &device);
        let loss = (head.forward(feats)["a"].clone() * 3.0 - head.forward(
            Tensor::<AB, 2>::from_floats([[0.2_f32, -0.9]], &device),
        )["T0"].clone().sum())
        .sum();
        let grads = GradientsParams::from_grads(loss.backward(), &head);
        weights(&opt.step(1e-2, head, grads))
    };
    let warm = second_step(release.head.clone(), release.optimizer.clone());
    let resumed = second_step(restored.head.clone(), restored.optimizer.clone());
    let cold_opt = ddrs::training::build_head_optimizer::<ddrs::nn::KanHead<AB>, AB>(
        ddrs::config::OptimizerKind::Adam,
    );
    let cold = second_step(restored.head.clone(), cold_opt);
    let dist = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    println!("resumed vs warm {:.3e}, cold vs warm {:.3e}", dist(&resumed, &warm), dist(&cold, &warm));
    assert!(dist(&resumed, &warm) < 1e-3, "restored optimizer does not step like the saved one");
    assert!(dist(&cold, &warm) > 10.0 * dist(&resumed, &warm), "check is not discriminating");
}

#[test]
fn bootstrap_from_a_checkpoint_without_release_files_starts_the_head_cold() {
    let device = Device::default();
    let dir = tempfile::tempdir().unwrap();
    let plain = write_config(dir.path(), false, None);
    let (_, state, _) = bootstrap_head_and_state::<I>(&plain, &device).unwrap();
    let ckpt = dir.path().join("epoch_2_mb_0");
    std::fs::create_dir_all(&ckpt).unwrap();
    save_kan_head(&ddrs::training::head_base(&ckpt), &state.head.clone().valid()).unwrap();

    let cfg = write_config(dir.path(), true, Some(&ckpt));
    let (_, state2, _) = bootstrap_head_and_state::<I>(&cfg, &device).unwrap();
    let release = state2.release.expect("learned config builds a release head");
    assert!(weights(&release.head).iter().all(|&w| w == 0.0), "cold release head");
}
