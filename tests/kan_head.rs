//! Rust mirror of `~/projects/ddr/tests/nn/test_kan.py`.
//!
//! DDR's KAN tests cover:
//!   - output shape per learnable parameter
//!   - sigmoid bounds (output ∈ [0, 1])
//!   - deterministic output for the same input
//!   - gradient flow to module parameters
//!   - varying `num_hidden_layers`
//!
//! We port all five and add two ddrs-specific tests:
//!   - input + output biases are exactly zero post-init (matches DDR init recipe)
//!   - end-to-end MLP → MuskingumCunge → loss → backward, gradients reach MLP params

mod common;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

use ddrs::nn::{KanHead, KanHeadConfig};

use common::{InnerBackend, TestBackend, TestDevice};

/// Default seed for `KanHead` test factories. Made explicit so any future
/// gradient/parity regression has a stable RNG fingerprint to compare against.
const HEAD_TEST_SEED: u64 = 42;

fn make_head<B: Backend>(
    input_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    learnable: &[&str],
    device: &B::Device,
) -> KanHead<B> {
    let cfg = KanHeadConfig::new(
        (0..input_size).map(|i| format!("attr_{i}")).collect(),
        learnable.iter().map(|s| s.to_string()).collect(),
        HEAD_TEST_SEED,
    )
    .with_hidden_size(hidden_size)
    .with_num_hidden_layers(num_hidden_layers);
    cfg.init::<B>(device)
}

/// Port of `test_kan_output_shape`.
#[test]
fn kan_head_output_shape() {
    let device = TestDevice::default();
    let model = make_head::<TestBackend>(5, 11, 1, &["n", "q_spatial"], &device);
    let inputs: Tensor<TestBackend, 2> = Tensor::random([10, 5], Distribution::Default, &device);
    let output = model.forward(inputs);

    assert!(output.contains_key("n"), "missing 'n' key");
    assert!(output.contains_key("q_spatial"), "missing 'q_spatial' key");
    assert_eq!(output["n"].dims(), [10]);
    assert_eq!(output["q_spatial"].dims(), [10]);
}

/// Port of `test_kan_sigmoid_bounds`.
#[test]
fn kan_head_sigmoid_bounds() {
    let device = TestDevice::default();
    let model = make_head::<TestBackend>(5, 11, 1, &["n", "q_spatial"], &device);
    let inputs: Tensor<TestBackend, 2> = Tensor::random([20, 5], Distribution::Default, &device);
    let output = model.forward(inputs);

    for key in ["n", "q_spatial"] {
        let v: Vec<f32> = output[key].clone().into_data().to_vec().unwrap();
        for x in v {
            assert!(
                (0.0..=1.0).contains(&x),
                "{key} contains {x} outside [0, 1]"
            );
        }
    }
}

/// Port of `test_kan_deterministic`. The forward pass has no internal RNG —
/// calling twice on the same input yields bit-identical output.
#[test]
fn kan_head_deterministic() {
    let device = TestDevice::default();
    let model = make_head::<TestBackend>(5, 11, 1, &["n", "q_spatial"], &device);
    let inputs: Tensor<TestBackend, 2> = Tensor::random([5, 5], Distribution::Default, &device);

    let out1 = model.forward(inputs.clone());
    let out2 = model.forward(inputs);

    for key in ["n", "q_spatial"] {
        let a: Vec<f32> = out1[key].clone().into_data().to_vec().unwrap();
        let b: Vec<f32> = out2[key].clone().into_data().to_vec().unwrap();
        assert_eq!(a, b, "{key} not deterministic across two forward calls");
    }
}

/// Port of `test_kan_gradient_flow`. Verify that backward from a scalar formed
/// out of the MLP outputs reaches at least one parameter of the module.
#[test]
fn kan_head_gradient_flow() {
    use burn::module::AutodiffModule;
    let device = TestDevice::default();
    let model_ad = make_head::<TestBackend>(5, 11, 1, &["n", "q_spatial"], &device);

    let inputs: Tensor<TestBackend, 2> = Tensor::random([5, 5], Distribution::Default, &device);
    let output = model_ad.forward(inputs);

    let loss = output["n"].clone().sum() + output["q_spatial"].clone().sum();
    let grads = loss.backward();

    // Check at least one parameter has a non-empty gradient buffer.
    let input_weight = model_ad.input.weight.val();
    let g = input_weight.grad(&grads);
    assert!(
        g.is_some(),
        "input.weight has no gradient — autograd is not wired through MLP"
    );
    let _ = model_ad.valid(); // confirm AutodiffModule trait is satisfied
}

/// Port of `test_kan_multiple_hidden_layers`. Three hidden layers still
/// produces the correct output shape.
#[test]
fn kan_head_multiple_hidden_layers() {
    let device = TestDevice::default();
    let model = make_head::<TestBackend>(5, 11, 3, &["n", "q_spatial"], &device);
    let inputs: Tensor<TestBackend, 2> = Tensor::random([5, 5], Distribution::Default, &device);
    let output = model.forward(inputs);

    assert_eq!(output["n"].dims(), [5]);
    assert_eq!(output["q_spatial"].dims(), [5]);
}

/// DDR-specific: confirm the init recipe zeros the input + output Linear
/// biases (matches `torch.nn.init.zeros_(self.{input,output}.bias)` in
/// `nn/kan.py:47-48`). Hidden KanLayers carry no bias — their per-edge
/// `scale_base` and `scale_sp` Params replace the per-neuron bias term.
#[test]
fn kan_head_biases_zero_at_init() {
    let device = TestDevice::default();
    let model = make_head::<TestBackend>(10, 21, 2, &["n", "q_spatial", "p_spatial"], &device);

    for (idx, layer) in [&model.input, &model.output].into_iter().enumerate() {
        let b = layer
            .bias
            .as_ref()
            .expect("Linear layer has a bias by default")
            .val();
        let data: Vec<f32> = b.into_data().to_vec().unwrap();
        for v in data {
            assert_eq!(v, 0.0, "Linear {idx} bias not zero at init");
        }
    }
}

/// End-to-end: KanHead emits SpatialParameters, MuskingumCunge routes, sum
/// of output is differentiable w.r.t. head parameters AND through the inner
/// KanLayer's spline coefficients.
#[test]
fn kan_head_to_muskingum_cunge_gradient_flow() {
    use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};

    let device = TestDevice::default();
    let n_segments = 5usize;
    let n_timesteps = 6usize;
    let n_attrs = 4usize;

    // Random per-reach attributes feed the head.
    let attrs: Tensor<TestBackend, 2> =
        Tensor::random([n_segments, n_attrs], Distribution::Default, &device);

    // Three learnable routing parameters → matches DDR's merit config.
    let learnable = ["n", "q_spatial", "p_spatial"];
    let head = make_head::<TestBackend>(n_attrs, 8, 1, &learnable, &device);
    let params_map = head.forward(attrs);

    let params = SpatialParameters::<InnerBackend> {
        n: params_map["n"].clone(),
        q_spatial: params_map["q_spatial"].clone(),
        p_spatial: Some(params_map["p_spatial"].clone()),
        k_d: None,
        d_gw: None,
        leakance_factor: None,
    };

    let mut mc = MuskingumCunge::<InnerBackend>::new(common::mock_config(), device.clone());
    let inputs = RoutingInputs::<InnerBackend> {
        adjacency: common::linear_chain_sparse(n_segments),
        x_storage: Tensor::<TestBackend, 1>::ones([n_segments], &device) * 0.2,
    };
    let streamflow = common::mock_streamflow(n_timesteps, n_segments, &device);

    mc.setup_inputs(inputs, streamflow, params, false);
    let out = mc.forward();
    let loss = out.sum();
    let grads = loss.backward();

    // Gradient must reach (a) the head's input Linear weight (the deepest
    // standard Param), and (b) the spline coefficients of the first inner
    // KanLayer (the v1 differentiability claim end-to-end).
    let g_in = head.input.weight.val().grad(&grads);
    assert!(
        g_in.is_some(),
        "no gradient reached head input weight — end-to-end backward broken"
    );
    let g_data: Vec<f32> = g_in.unwrap().into_data().to_vec().unwrap();
    let any_nonzero = g_data.iter().any(|&v: &f32| v != 0.0);
    let all_finite = g_data.iter().all(|v: &f32| v.is_finite());
    assert!(all_finite, "head input weight grad has non-finite values");
    assert!(any_nonzero, "head input weight grad is all zeros");

    let g_coef = head
        .hidden
        .first()
        .expect("test uses num_hidden_layers=1")
        .coef
        .val()
        .grad(&grads);
    assert!(
        g_coef.is_some(),
        "no gradient reached the inner KanLayer's coef Param — rskan autodiff broken at FFI"
    );
    let g_coef_data: Vec<f32> = g_coef.unwrap().into_data().to_vec().unwrap();
    let coef_finite = g_coef_data.iter().all(|v: &f32| v.is_finite());
    let coef_nonzero = g_coef_data.iter().any(|&v: &f32| v != 0.0);
    assert!(coef_finite, "KanLayer coef grad has non-finite values");
    assert!(coef_nonzero, "KanLayer coef grad is all zeros");
}

// Silence the unused-Autodiff/NdArray import warnings — keeping the imports
// explicit makes the test file self-documenting about its backend choice.
#[allow(dead_code)]
type _Backends = (Autodiff<NdArray<f32>>, NdArray<f32>);
