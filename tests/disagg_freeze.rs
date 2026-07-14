//! Freeze semantics for a pretrained disaggregation head loaded into the
//! production `KanHead` (mirrors `bootstrap_head_and_state`'s
//! `pretrained_checkpoint` + `freeze` path, `src/training/bootstrap.rs`).
//!
//! 1. `load_disagg_head` + `Module::no_grad()` → the disagg submodule's
//!    params are byte-identical after an optimizer step whose loss touches
//!    BOTH the disagg and routing heads (and the routing params DO move, so
//!    the step wasn't a global no-op).
//! 2. Without `.no_grad()` (freeze: false) the same setup leaves the
//!    pretrained disagg head trainable — negative control for test 1.

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::optim::{GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::BackendTypes;
use burn::tensor::{Tensor, TensorData};

use ddrs::nn::{DisaggHead, DisaggHeadConfig, KanHead, KanHeadConfig};
use ddrs::training::{build_adam, load_disagg_head};

type AB = Autodiff<NdArray<f32>>;

/// Tiny disagg architecture shared by the standalone save and the KanHead
/// template — MUST match for `load_record` to succeed.
const DISAGG_HIDDEN: usize = 4;
const DISAGG_LAYERS: usize = 1;
const DISAGG_GRID: usize = 3;

/// Seed for the STANDALONE pretrained head — different from the KanHead's
/// seed so a successful load is distinguishable from the fresh init.
const PRETRAIN_SEED: u64 = 7;
const HEAD_SEED: u64 = 42;

fn disagg_cfg(seed: u64) -> DisaggHeadConfig {
    DisaggHeadConfig::new(seed)
        .with_hidden_size(DISAGG_HIDDEN)
        .with_num_hidden_layers(DISAGG_LAYERS)
        .with_grid(DISAGG_GRID)
}

fn make_head(device: &<AB as BackendTypes>::Device) -> KanHead<AB> {
    KanHeadConfig::new(
        (0..4).map(|i| format!("attr_{i}")).collect(),
        vec!["n".to_string()],
        HEAD_SEED,
    )
    .with_hidden_size(8)
    .with_num_hidden_layers(1)
    .with_disagg_enabled(true)
    .with_disagg_hidden_size(DISAGG_HIDDEN)
    .with_disagg_num_hidden_layers(DISAGG_LAYERS)
    .with_disagg_grid(DISAGG_GRID)
    .init::<AB>(device)
}

/// Save a standalone pretrained DisaggHead, then load it into `head.disagg`
/// via `load_disagg_head` — the exact bootstrap logic, inlined (the full
/// `bootstrap_head_and_state` needs a complete `Config`).
fn head_with_pretrained_disagg(
    freeze: bool,
    device: &<AB as BackendTypes>::Device,
    dir: &std::path::Path,
) -> KanHead<AB> {
    let pretrained: DisaggHead<AB> = disagg_cfg(PRETRAIN_SEED).init(device);
    let ckpt_base = dir.join("pretrained_disagg");
    CompactRecorder::new()
        .record(pretrained.into_record(), ckpt_base.clone())
        .expect("save standalone disagg checkpoint");

    let mut head = make_head(device);
    let template = head.disagg.take().expect("disagg enabled");
    let mut disagg =
        load_disagg_head::<AB>(&ckpt_base, template, device).expect("load pretrained disagg");
    if freeze {
        disagg = disagg.no_grad();
    }
    head.disagg = Some(disagg);
    head
}

fn disagg_input_weight_bytes(head: &KanHead<AB>) -> Vec<f32> {
    head.disagg
        .as_ref()
        .unwrap()
        .input
        .weight
        .val()
        .into_data()
        .to_vec()
        .unwrap()
}

fn routing_output_weight_bytes(head: &KanHead<AB>) -> Vec<f32> {
    head.output.weight.val().into_data().to_vec().unwrap()
}

/// One Adam step against a loss that flows through BOTH the disagg head's
/// forward (shape-sensitive via an hourly ramp — a plain sum is constant
/// under mass conservation and would give the disagg params zero gradient)
/// AND the routing head's output layer.
fn one_step_touching_both(
    head: KanHead<AB>,
    device: &<AB as BackendTypes>::Device,
) -> KanHead<AB> {
    // Disagg forward: 2 days × 2 reaches, 24 forwarded hours (d_use = 1).
    let daily_q = Tensor::<AB, 2>::from_data(
        TensorData::new(vec![5.0f32, 1.0, 20.0, 3.0], [2, 2]),
        device,
    );
    let precip: Vec<f32> = (0..48).map(|i| if i / 2 == 13 { 4.0 } else { 0.1 }).collect();
    let precip = Tensor::<AB, 2>::from_data(TensorData::new(precip, [24, 2]), device);
    let hourly = head.disagg.as_ref().unwrap().forward(daily_q, precip, 24); // (24, 2)
    let ramp = Tensor::<AB, 1>::from_data(
        TensorData::new((0..24).map(|i| i as f32).collect::<Vec<_>>(), [24]),
        device,
    )
    .reshape([24, 1]);
    let disagg_loss = (hourly * ramp).sum();

    // Routing forward: (N=3, F=4) attributes.
    let attrs: Vec<f32> = (0..12).map(|i| ((i as f32) * 0.3 - 1.0).sin()).collect();
    let attrs = Tensor::<AB, 2>::from_data(TensorData::new(attrs, [3, 4]), device);
    let routing_loss = head.forward(attrs)["n"].clone().sum();

    let loss = disagg_loss + routing_loss;
    let grads = GradientsParams::from_grads(loss.backward(), &head);
    let mut opt = build_adam::<KanHead<AB>, AB>();
    opt.step(0.01, head, grads)
}

#[test]
fn frozen_disagg_params_are_byte_identical_after_optimizer_step() {
    let device = Default::default();
    let tmp = tempfile::tempdir().unwrap();
    let head = head_with_pretrained_disagg(true, &device, tmp.path());

    // Sanity: the loaded values are the PRETRAINED ones, not the fresh init.
    let fresh = make_head(&device);
    assert_ne!(
        disagg_input_weight_bytes(&head),
        disagg_input_weight_bytes(&fresh),
        "loaded disagg weights should differ from the seed-{HEAD_SEED} fresh init"
    );

    let disagg_before = disagg_input_weight_bytes(&head);
    let routing_before = routing_output_weight_bytes(&head);

    let head = one_step_touching_both(head, &device);

    assert_eq!(
        disagg_input_weight_bytes(&head),
        disagg_before,
        "frozen disagg input.weight must be byte-identical after the step"
    );
    assert_ne!(
        routing_output_weight_bytes(&head),
        routing_before,
        "routing output.weight must move — otherwise the step was a global no-op \
         and the frozen assertion above is vacuous"
    );
}

#[test]
fn freeze_false_leaves_pretrained_disagg_trainable() {
    let device = Default::default();
    let tmp = tempfile::tempdir().unwrap();
    let head = head_with_pretrained_disagg(false, &device, tmp.path());

    let disagg_before = disagg_input_weight_bytes(&head);
    let head = one_step_touching_both(head, &device);

    assert_ne!(
        disagg_input_weight_bytes(&head),
        disagg_before,
        "without freeze the pretrained disagg head must keep training"
    );
}
