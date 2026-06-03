//! KAN-based routing-parameter head.
//!
//! Matches DDR-Python's `~/projects/ddr/src/ddr/nn/kan.py` architecture exactly:
//! a `Linear(F, H)` input projection feeds a stack of `rskan::KanLayer(H, H)`
//! blocks (no inter-block ReLU — see migration spec §8.2), followed by
//! `Linear(H, P) → Sigmoid` and a per-parameter HashMap split. Replaces the
//! prior MLP placeholder.
//!
//! - Input: `Tensor<B, 2>` of shape `[N, F]` (N = num reaches, F = num attributes).
//! - Architecture:
//!   `Linear(F, H) → KanLayer(H, H) × num_hidden_layers → Linear(H, P) → Sigmoid`
//! - Output: `HashMap<String, Tensor<B, 1>>` of length `P`, with `output[key].shape == [N]`,
//!   keyed by the entries of `learnable_parameters` in order.
//!
//! Init recipe (matches DDR `kan.py:45-48` element-for-element, with
//! `StdRng`-based sampling instead of PyTorch global MT — see C4):
//! - input Linear weight:  Kaiming-normal, `std = sqrt(2)/sqrt(F)`.
//! - output Linear weight: Xavier-normal, `std = 0.1 * sqrt(2/(H+P))`.
//! - both biases:          zero.
//! - hidden KanLayers:     `rskan::KanLayerConfig::new(H, H, seed)` with
//!                         `num=grid`, `k=k`, `noise_scale=0.3`. Same
//!                         `seed` for every inner KanLayer.
//! See `src/nn/init.rs` for the actual sampling code.
//!
//! Defaults (`hidden_size=21`, `num_hidden_layers=2`, `grid=5`, `k=3`) match
//! `config/merit_training.yaml`.

use std::collections::HashMap;

use burn::config::Config;
use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::activation::sigmoid;
use burn::tensor::{backend::Backend, Tensor};
use rand::SeedableRng;
use rskan::{KanLayer, KanLayerConfig};

/// Xavier gain applied to the output layer (matches DDR `kan.py:46`).
const XAVIER_GAIN_OUTPUT: f64 = 0.1;
/// pykan MultKAN default; matches DDR's `KAN([H, H], ...)` noise_scale default.
const KAN_NOISE_SCALE: f64 = 0.3;

/// Configuration for the KAN head.
#[derive(Config, Debug)]
pub struct KanHeadConfig {
    /// Names of input attributes. The head only uses the length to size the
    /// input layer; names are stored for traceability and to mirror the DDR
    /// `kan` constructor signature.
    pub input_var_names: Vec<String>,
    /// Names of output parameters, e.g. `["n", "q_spatial", "p_spatial"]`.
    /// The forward pass returns a `HashMap` keyed by these names.
    pub learnable_parameters: Vec<String>,
    /// Seed for KanLayer initialization. REQUIRED — no default. Passed to
    /// **every** inner KanLayer (DDR-Python quirk: same seed all blocks,
    /// `kan.py:24-34`).
    pub seed: u64,

    /// Hidden layer width. `21` per `config/merit_training.yaml`.
    #[config(default = 21)]
    pub hidden_size: usize,
    /// Number of `KanLayer(H, H)` blocks between input and output. `2`
    /// per `config/merit_training.yaml`.
    #[config(default = 2)]
    pub num_hidden_layers: usize,
    /// B-spline grid intervals (`num` in pykan). pykan MultKAN default = 3;
    /// `config/merit_training.yaml` overrides to 5 to match DDR.
    #[config(default = 5)]
    pub grid: usize,
    /// B-spline order. `3` per cubic-spline default.
    #[config(default = 3)]
    pub k: usize,
}

impl KanHeadConfig {
    /// Build the KAN head, initializing parameters per the DDR `kan.py` recipe
    /// using a project-controlled `StdRng` seeded from `self.seed`. See
    /// `src/nn/init.rs` for the sampling formulas.
    ///
    /// The same `self.seed` is also passed to every inner `KanLayer` — see
    /// the module-level docstring for why.
    pub fn init<B: Backend>(&self, device: &B::Device) -> KanHead<B> {
        assert!(
            !self.input_var_names.is_empty(),
            "input_var_names must be non-empty"
        );
        assert!(
            !self.learnable_parameters.is_empty(),
            "learnable_parameters must be non-empty"
        );

        let f = self.input_var_names.len();
        let h = self.hidden_size;
        let p = self.learnable_parameters.len();

        // Single StdRng controls both Linears so their bytes are reproducible
        // at fixed `seed`. The inner KanLayers each get the same `seed`
        // directly (rskan reseeds internally) — they do NOT consume from this
        // RNG.
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);

        let input_weight = crate::nn::init::sample_kaiming_normal_relu(&mut rng, f, h);
        let output_weight =
            crate::nn::init::sample_xavier_normal(&mut rng, h, p, XAVIER_GAIN_OUTPUT as f32);

        let input = burn::nn::Linear {
            weight: crate::nn::init::to_param_weight::<B>(input_weight, device),
            bias: Some(crate::nn::init::zero_bias_tensor::<B>(h, device)),
        };
        let output = burn::nn::Linear {
            weight: crate::nn::init::to_param_weight::<B>(output_weight, device),
            bias: Some(crate::nn::init::zero_bias_tensor::<B>(p, device)),
        };

        // DDR-Python quirk: same `seed` passed to every inner `KAN([H, H])`
        // constructor. See migration spec §8.3.
        let hidden: Vec<KanLayer<B>> = (0..self.num_hidden_layers)
            .map(|_| {
                KanLayerConfig::new(self.hidden_size, self.hidden_size, self.seed)
                    .with_num(self.grid)
                    .with_k(self.k)
                    .with_noise_scale(KAN_NOISE_SCALE)
                    .init(device)
            })
            .collect();

        KanHead {
            input,
            hidden,
            output,
            learnable_parameters: self.learnable_parameters.clone(),
        }
    }
}

/// KAN-based head producing routing parameters from per-reach attributes.
#[derive(Module, Debug)]
pub struct KanHead<B: Backend> {
    pub input: Linear<B>,
    pub hidden: Vec<KanLayer<B>>,
    pub output: Linear<B>,
    /// Names of output parameters in column order — used to build the output
    /// HashMap. Carried as state so the head round-trips through a record
    /// without requiring callers to re-supply the keys.
    learnable_parameters: Vec<String>,
}

impl<B: Backend> KanHead<B> {
    /// Forward pass.
    ///
    /// Mirrors `kan.forward` in DDR (`nn/kan.py:50-62`): chains the input
    /// `Linear`, the KAN blocks (each block applies SiLU+spline edge
    /// activations internally), then the output `Linear` and a sigmoid;
    /// transposes to `[P, N]` and splits row-by-row into the
    /// `learnable_parameters` slots of the returned HashMap.
    ///
    /// **No inter-block ReLU**, matching DDR's `kan.py:53` direct chaining
    /// of `KAN([H, H])` modules.
    pub fn forward(&self, inputs: Tensor<B, 2>) -> HashMap<String, Tensor<B, 1>> {
        let mut x = self.input.forward(inputs);
        for layer in &self.hidden {
            x = layer.forward(x);
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
