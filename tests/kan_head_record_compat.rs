//! Checkpoints written before `parameter_groups` / `input_layer_kan` /
//! `output_layer_kan` existed must still load.
//!
//! Those three knobs added fields to the `KanHead` `Module` record
//! (`input_kan`, `output_kan`, `extra`). A burn record is matched by field, so
//! an additive change is only safe if the recorder tolerates keys the saved
//! file does not carry. Every landscape study in
//! `.ddrs/experiments/` re-loads old run checkpoints, so a silent break here
//! would invalidate the whole instrument, not just future training.
//!
//! Point `DDRS_LEGACY_HEAD_CKPT` at a pre-change checkpoint directory (the one
//! holding `head.mpk`) to run it; the test skips when the variable is unset,
//! since CI has no `.ddrs/` workspace.
//!
//! ```bash
//! DDRS_LEGACY_HEAD_CKPT=.ddrs/runs/<id>/checkpoints/epoch_9_mb_9 \
//!   cargo test --test kan_head_record_compat -- --nocapture
//! ```

use std::path::PathBuf;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use ddrs::config::Config;
use ddrs::nn::kan_head::KanHead;
use ddrs::training::checkpoint::{head_base, load_kan_head};

type B = NdArray<f32>;

#[test]
fn a_pre_parameter_groups_checkpoint_still_loads() {
    let Ok(ckpt) = std::env::var("DDRS_LEGACY_HEAD_CKPT") else {
        eprintln!("skipped: set DDRS_LEGACY_HEAD_CKPT to a pre-change checkpoint dir");
        return;
    };
    let ckpt = PathBuf::from(ckpt);
    assert!(ckpt.is_dir(), "not a directory: {}", ckpt.display());

    // The run's own config snapshot is the only thing that reproduces the
    // template the checkpoint was written against.
    let cfg_path = ckpt
        .parent()
        .and_then(|p| p.parent())
        .expect("checkpoint dir is <run>/checkpoints/<epoch>")
        .join("config.yaml");
    let cfg = Config::from_yaml_file(&cfg_path).expect("load run config snapshot");
    let section = cfg.kan_head.as_ref().expect("kan_head section");

    let device = NdArrayDevice::default();
    let template: KanHead<B> = ddrs::config::kan_config(section, cfg.seed).init(&device);

    assert!(
        template.extra.is_empty() && template.input_kan.is_none() && template.output_kan.is_none(),
        "a legacy config must still build the shared, all-Linear head"
    );

    let loaded = load_kan_head::<B>(&head_base(&ckpt), template, &device)
        .expect("legacy checkpoint failed to load into the post-change head");

    assert_eq!(
        loaded.learnable_parameters(),
        section.learnable_parameters.as_slice()
    );
    // A loaded head must not be the freshly-initialised one.
    let w = loaded
        .output
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        w.iter().any(|v| v.abs() > 1e-12),
        "loaded output weights are all zero — the record did not apply"
    );
    eprintln!(
        "legacy checkpoint loaded: {} params, output weight[0] = {}",
        loaded.learnable_parameters().len(),
        w[0]
    );
}
