//! Release-only training: `release_head.routing_checkpoint` +
//! `release_head.freeze_routing` (the review synthesis' item 4).
//!
//! On the committed Juniata bundle (213 reaches, one gauge, Raystown Dam as
//! the one release-head dam), through the real driver (`training::train`):
//!
//! 1. The routing head starts from another run's `head.mpk` (weights only),
//!    is detached, and after several optimizer steps is BITWISE the
//!    checkpoint, while the release head's read-out has moved off its zero
//!    init. Checked on the single-batch path and the accumulation path, and
//!    the frozen run's own saved checkpoints carry the same routing weights
//!    (the test phase loads those).
//! 2. With no dam armed (Raystown's completion year moved past the windows),
//!    the loss is a constant, the driver takes no step at all, and the frozen
//!    run routes BITWISE like a plain forward of the checkpoint under a
//!    config without reservoirs.
//!
//! Windows are 20 days so the debug build stays fast.

use std::path::{Path, PathBuf};

use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{backend::Backend, Tensor};
use rand::SeedableRng;

use ddrs::config::{kan_config, Config};
use ddrs::data::MeritGagesDataset;
use ddrs::nn::KanHead;
use ddrs::training::forward::{forward, forward_with_release};
use ddrs::training::{bootstrap_head_and_state, head_base, load_kan_head, save_kan_head, train};

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const JUNIATA_CONFIG: &str = "examples/juniata/ddrs.yaml";
const DAM_TABLE: &str = "examples/juniata/data/juniata_dam_features.csv";

/// The 19 dam-feature columns of the Juniata dam table.
const DAM_FEATURES: [&str; 19] = [
    "log10_storage",
    "log10_storage_max",
    "log10_surface",
    "log10_drainage",
    "log10_storage_per_area",
    "log10_max_discharge",
    "height",
    "year",
    "log10_storage_max_missing",
    "log10_surface_missing",
    "log10_max_discharge_missing",
    "year_missing",
    "purpose_flood",
    "purpose_hydro",
    "purpose_supply",
    "purpose_irrigation",
    "purpose_recreation",
    "purpose_navigation",
    "purpose_other",
];

fn juniata_yaml() -> serde_yaml::Value {
    let mut cfg: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(JUNIATA_CONFIG).unwrap()).unwrap();
    // Three one-gauge mini-batches (one per epoch), short windows.
    cfg["experiment"]["epochs"] = 3.into();
    cfg["experiment"]["rho"] = 20.into();
    cfg["experiment"]["warmup"] = 2.into();
    cfg
}

fn write(dir: &Path, name: &str, cfg: &serde_yaml::Value) -> Config {
    let path = dir.join(name);
    std::fs::write(&path, serde_yaml::to_string(cfg).unwrap()).unwrap();
    Config::from_yaml_file(&path).expect("config loads")
}

/// The Juniata bundle with the learned release on `dam_table`, the routing
/// head frozen at `routing_ckpt`.
fn frozen_cfg(dir: &Path, name: &str, dam_table: &str, routing_ckpt: &Path, accum: bool) -> Config {
    let mut cfg = juniata_yaml();
    cfg["params"]["use_reservoirs"] = true.into();
    cfg["params"]["reservoir_release"] = "learned".into();
    cfg["data_sources"]["reservoirs"] = dam_table.into();
    if accum {
        cfg["experiment"]["use_grad_accum"] = true.into();
        cfg["experiment"]["grad_accum_steps"] = 2.into();
    }
    let mut rh = serde_yaml::Mapping::new();
    rh.insert(
        "input_var_names".into(),
        serde_yaml::Value::Sequence(DAM_FEATURES.iter().map(|s| (*s).into()).collect()),
    );
    rh.insert("routing_checkpoint".into(), routing_ckpt.display().to_string().into());
    rh.insert("freeze_routing".into(), true.into());
    cfg["release_head"] = serde_yaml::Value::Mapping(rh);
    write(dir, name, &cfg)
}

/// A routing checkpoint directory whose head is NOT the seed-42 init the
/// frozen run would otherwise start from (seed 7), so a missing load shows.
fn routing_checkpoint(dir: &Path) -> PathBuf {
    let plain = write(dir, "plain.yaml", &juniata_yaml());
    let head = kan_config(plain.kan_head.as_ref().unwrap(), 7).init::<I>(&Device::default());
    let ckpt = dir.join("routing_ckpt").join("epoch_9_mb_0");
    std::fs::create_dir_all(&ckpt).unwrap();
    save_kan_head(&head_base(&ckpt), &head).unwrap();
    ckpt
}

/// The checkpoint's routing head, loaded the way the test phase loads one.
fn reference_head(cfg: &Config, ckpt: &Path) -> KanHead<AB> {
    let template = kan_config(cfg.kan_head.as_ref().unwrap(), cfg.seed).init::<AB>(&Device::default());
    load_kan_head::<AB>(&head_base(ckpt), template, &Device::default()).unwrap()
}

/// Every float parameter of a module, in visit order.
struct Collect(Vec<f32>);

impl<B: Backend> ModuleVisitor<B> for Collect {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.0.extend(param.val().into_data().convert::<f32>().to_vec::<f32>().unwrap());
    }
}

fn params<B: Backend, M: Module<B>>(m: &M) -> Vec<f32> {
    let mut c = Collect(Vec::new());
    m.visit(&mut c);
    c.0
}

fn assert_bitwise(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}: element {i}: {x} vs {y}");
    }
}

fn frozen_routing_stays_at_the_checkpoint(accum: bool) {
    let device = Device::default();
    let tmp = tempfile::tempdir().unwrap();
    let ckpt = routing_checkpoint(tmp.path());
    let cfg = frozen_cfg(tmp.path(), "frozen.yaml", DAM_TABLE, &ckpt, accum);
    assert!(cfg.routing_frozen());
    let reference = params(&reference_head(&cfg, &ckpt));
    let seed42 = params(&kan_config(cfg.kan_head.as_ref().unwrap(), 42).init::<AB>(&device));
    assert_ne!(reference, seed42, "the checkpoint must differ from the run's own init");

    let dataset = MeritGagesDataset::open(&cfg).unwrap();
    let (_, mut state, mut optimizer) = bootstrap_head_and_state::<I>(&cfg, &device).unwrap();
    assert_bitwise("routing head after bootstrap", &params(&state.head), &reference);
    let release_init = params(&state.release.as_ref().unwrap().head.output);

    let run_ckpts = tmp.path().join("checkpoints");
    train::<I>(&cfg, &dataset, &mut state, &mut optimizer, &device, &run_ckpts, None, None).unwrap();

    assert_bitwise("routing head after training", &params(&state.head), &reference);
    let release_now = params(&state.release.as_ref().unwrap().head.output);
    let moved = release_init.iter().zip(&release_now).filter(|(a, b)| a != b).count();
    assert!(moved > 0, "the release head did not train");
    println!("accum={accum}: {moved} of {} release read-out values moved", release_now.len());

    // The run's own checkpoints (what the test phase loads) hold the frozen weights.
    let mut saved: Vec<PathBuf> = std::fs::read_dir(&run_ckpts)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    saved.sort();
    assert!(!saved.is_empty(), "the frozen run wrote no checkpoint");
    for dir in &saved {
        assert_bitwise(
            &format!("saved {}", dir.display()),
            &params(&reference_head(&cfg, dir)),
            &reference,
        );
    }
}

#[test]
fn frozen_routing_head_stays_at_its_checkpoint_while_the_release_head_trains() {
    frozen_routing_stays_at_the_checkpoint(false);
}

#[test]
fn frozen_routing_head_stays_at_its_checkpoint_with_gradient_accumulation() {
    frozen_routing_stays_at_the_checkpoint(true);
}

#[test]
fn frozen_run_with_no_dam_armed_routes_like_the_checkpoint() {
    let device = Device::default();
    let tmp = tempfile::tempdir().unwrap();
    let ckpt = routing_checkpoint(tmp.path());

    // Raystown completed "2050": never a dam row in any window.
    let table = std::fs::read_to_string(DAM_TABLE).unwrap();
    let unbuilt = table.replacen(",1973", ",2050", 1);
    assert_ne!(unbuilt, table, "the fixture's completion year moved");
    let unbuilt_path = tmp.path().join("unbuilt_dam.csv");
    std::fs::write(&unbuilt_path, unbuilt).unwrap();

    let cfg = frozen_cfg(tmp.path(), "frozen.yaml", unbuilt_path.to_str().unwrap(), &ckpt, false);
    let dataset = MeritGagesDataset::open(&cfg).unwrap();
    let (_, mut state, mut optimizer) = bootstrap_head_and_state::<I>(&cfg, &device).unwrap();
    let release_init = params(&state.release.as_ref().unwrap().head);
    // No tracked parameter reaches the loss: the driver must skip, not panic.
    train::<I>(&cfg, &dataset, &mut state, &mut optimizer, &device, &tmp.path().join("ck"), None, None)
        .unwrap();
    assert_bitwise(
        "release head without an armed dam",
        &params(&state.release.as_ref().unwrap().head),
        &release_init,
    );

    // One batch through the frozen run's forward...
    let rho = cfg.experiment.as_ref().unwrap().rho.unwrap();
    let window = dataset
        .time_axis()
        .sample_rho_window(&mut rand_chacha::ChaCha12Rng::seed_from_u64(3), rho);
    let staids = dataset.staids().to_vec();
    let tensors = dataset.collate(&staids, &window).unwrap().to_tensors::<AB>(&device);
    let release = state.release.as_ref().map(|r| &r.head);
    let frozen: Vec<f32> = forward_with_release::<I>(&cfg, &tensors, &state.head, release, &device, false, None)
        .into_data()
        .to_vec()
        .unwrap();

    // ...and a plain forward of the checkpoint under the bundle's own config.
    let plain_cfg = write(tmp.path(), "plain.yaml", &juniata_yaml());
    assert!(!plain_cfg.params.use_reservoirs);
    let plain_ds = MeritGagesDataset::open(&plain_cfg).unwrap();
    let plain_tensors = plain_ds.collate(&staids, &window).unwrap().to_tensors::<AB>(&device);
    let plain: Vec<f32> = forward::<I>(&plain_cfg, &plain_tensors, &reference_head(&plain_cfg, &ckpt), &device, false, None)
        .into_data()
        .to_vec()
        .unwrap();
    assert!(!plain.is_empty() && plain.iter().all(|v| v.is_finite()));
    assert_bitwise("frozen run vs plain forward of the checkpoint", &frozen, &plain);
}
