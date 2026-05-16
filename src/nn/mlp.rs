//! Feed-forward MLP head that produces routing parameters.
//!
//! Drop-in replacement for `~/projects/ddr/src/ddr/nn/kan.py`'s `kan` module.
//! Same external I/O contract:
//!
//! - Input: `Tensor<B, 2>` of shape `[N, F]` (N = num reaches, F = num attributes).
//! - Architecture:
//!   `Linear(F, H) → ReLU → (Linear(H, H) → ReLU) × num_hidden_layers → Linear(H, P) → Sigmoid`
//! - Output: `HashMap<String, Tensor<B, 1>>` of length `P`, with `output[key].shape == [N]`,
//!   keyed by the entries of `learnable_parameters` in order.
//!
//! Init recipe is copied from `kan.__init__` (lines 45-48 of `nn/kan.py`):
//! - input weight: `kaiming_normal_(nonlinearity="relu")`
//! - output weight: `xavier_normal_(gain=0.1)`
//! - hidden weights: same as input (matches the spirit of `nonlinearity="relu"`
//!   for a plain MLP; DDR's KAN layers had their own init that we don't need
//!   to mirror here)
//! - all biases: zero
//!
//! Defaults (`hidden_size=21`, `num_hidden_layers=2`) match `merit_training_config.yaml`.

use std::collections::HashMap;

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{backend::Backend, Tensor};

/// Kaiming normal gain for ReLU nonlinearity: `sqrt(2)`.
const KAIMING_GAIN_RELU: f64 = std::f64::consts::SQRT_2;
/// Xavier gain applied to the output layer (matches DDR `kan.py:46`).
const XAVIER_GAIN_OUTPUT: f64 = 0.1;

/// Configuration for the MLP head.
#[derive(Config, Debug)]
pub struct MlpConfig {
    /// Names of input attributes. The MLP only uses the length to size the
    /// input layer; names are stored for traceability and to match the DDR
    /// `kan` constructor signature.
    pub input_var_names: Vec<String>,
    /// Names of output parameters, e.g. `["n", "q_spatial", "p_spatial"]`.
    /// The forward pass returns a `HashMap` keyed by these names.
    pub learnable_parameters: Vec<String>,
    /// Hidden layer width. `21` per `merit_training_config.yaml`.
    #[config(default = 21)]
    pub hidden_size: usize,
    /// Number of `Linear(H, H) → ReLU` blocks between input and output. `2`
    /// per `merit_training_config.yaml`. Set to `0` for a 1-hidden-layer MLP.
    #[config(default = 2)]
    pub num_hidden_layers: usize,
}

impl MlpConfig {
    /// Build the MLP, initializing parameters per the DDR `kan` recipe.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        assert!(
            !self.input_var_names.is_empty(),
            "input_var_names must be non-empty"
        );
        assert!(
            !self.learnable_parameters.is_empty(),
            "learnable_parameters must be non-empty"
        );

        let kaiming = Initializer::KaimingNormal {
            gain: KAIMING_GAIN_RELU,
            fan_out_only: false,
        };
        let xavier = Initializer::XavierNormal {
            gain: XAVIER_GAIN_OUTPUT,
        };

        let input = LinearConfig::new(self.input_var_names.len(), self.hidden_size)
            .with_initializer(kaiming.clone())
            .init(device);

        let hidden: Vec<Linear<B>> = (0..self.num_hidden_layers)
            .map(|_| {
                LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_initializer(kaiming.clone())
                    .init(device)
            })
            .collect();

        let output = LinearConfig::new(self.hidden_size, self.learnable_parameters.len())
            .with_initializer(xavier)
            .init(device);

        // Zero all biases (matches `torch.nn.init.zeros_(self.input.bias)` and
        // `zeros_(self.output.bias)` in DDR's kan.py:47-48; hidden biases too
        // for consistency).
        let input = zero_bias(input, device);
        let hidden = hidden
            .into_iter()
            .map(|l| zero_bias(l, device))
            .collect::<Vec<_>>();
        let output = zero_bias(output, device);

        Mlp {
            input,
            hidden,
            output,
            learnable_parameters: self.learnable_parameters.clone(),
        }
    }
}

/// Feed-forward MLP producing routing parameters from per-reach attributes.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub input: Linear<B>,
    pub hidden: Vec<Linear<B>>,
    pub output: Linear<B>,
    /// Names of output parameters in column order — used to build the output
    /// HashMap. Carried as state so `Mlp` round-trips through a record without
    /// requiring callers to re-supply the keys.
    learnable_parameters: Vec<String>,
}

impl<B: Backend> Mlp<B> {
    /// Forward pass.
    ///
    /// Mirrors `kan.forward` in DDR (`nn/kan.py:50-62`): produces a sigmoid
    /// output of shape `[N, P]`, transposes to `[P, N]`, and splits row-by-row
    /// into the `learnable_parameters` slots of the returned HashMap.
    pub fn forward(&self, inputs: Tensor<B, 2>) -> HashMap<String, Tensor<B, 1>> {
        let mut x = relu(self.input.forward(inputs));
        for layer in &self.hidden {
            x = relu(layer.forward(x));
        }
        let logits = self.output.forward(x); // [N, P]
        let probs = sigmoid(logits); // [N, P] ∈ (0, 1)

        let dims = probs.dims();
        let n = dims[0];
        let p = dims[1];
        debug_assert_eq!(
            p,
            self.learnable_parameters.len(),
            "output width {} does not match learnable_parameters.len() {}",
            p,
            self.learnable_parameters.len()
        );

        let transposed: Tensor<B, 2> = probs.swap_dims(0, 1); // [P, N]
        let mut out = HashMap::with_capacity(p);
        for (idx, key) in self.learnable_parameters.iter().enumerate() {
            let row: Tensor<B, 1> = transposed
                .clone()
                .slice([idx..idx + 1, 0..n])
                .reshape([n]);
            out.insert(key.clone(), row);
        }
        out
    }

    /// Names of output parameters in column order. Useful for tests + callers
    /// that need to iterate consistently.
    pub fn learnable_parameters(&self) -> &[String] {
        &self.learnable_parameters
    }
}

/// Replace a `Linear`'s bias with a zero tensor of the same shape.
///
/// `LinearConfig::init` uses the *same* `Initializer` for both weight and bias,
/// so a `KaimingNormal` or `XavierNormal` config produces a random bias. DDR's
/// recipe zeros all biases explicitly; we do the same here.
fn zero_bias<B: Backend>(layer: Linear<B>, device: &B::Device) -> Linear<B> {
    let Linear { weight, bias } = layer;
    let bias = bias.map(|b| {
        let shape = b.shape();
        let zero: Tensor<B, 1> = Tensor::zeros(shape, device);
        Param::initialized(b.id, zero)
    });
    Linear { weight, bias }
}
