//! AdaDelta optimizer (Zeiler 2012), matching `torch.optim.Adadelta`.
//!
//! Why this exists: with gradient accumulation the whole epoch collapses to a
//! handful of optimizer steps, and Adam's per-step parameter movement is
//! bounded by `lr` — a 30-update run at lr 5e-4 can move a weight at most
//! ~0.017 total, which is why the 2026-07-30 all-basins run finished with the
//! KAN still emitting a near-constant field (n spanning 7.5% of its range,
//! flat vs drainage area) and losing to the summed-Q' baseline.
//!
//! AdaDelta is **scale-free**: the step is `RMS[Δx]_{t-1} / RMS[g]_t · g`, so
//! the update carries the units of the parameter itself and needs no lr
//! schedule (`lr = 1.0`). dHBV uses `torch.optim.Adadelta(model.parameters())`
//! with all defaults for exactly this reason
//! (`water_loss/.../train_gradaccum.py:93`).
//!
//! Update rule (PyTorch `_single_tensor_adadelta`, defaults ρ=0.9, ε=1e-6):
//! ```text
//! square_avg = ρ·square_avg + (1-ρ)·g²
//! std        = sqrt(square_avg + ε)
//! delta      = sqrt(acc_delta + ε) / std · g
//! acc_delta  = ρ·acc_delta + (1-ρ)·delta²
//! param     -= lr · delta
//! ```
//! Both accumulators start at zero, so the first step is
//! `sqrt(ε)/sqrt((1-ρ)g² + ε) · g` — bounded by `sqrt(ε)` regardless of how
//! large the gradient is, then grows as `acc_delta` fills.

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{LearningRate, SimpleOptimizer};
use burn::record::Record;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ops::Device, Tensor};

/// Per-parameter AdaDelta state: the two running averages.
#[derive(Record, Clone)]
pub struct AdaDeltaState<B: Backend, const D: usize> {
    /// `E[g²]` — running average of squared gradients.
    square_avg: Tensor<B, D>,
    /// `E[Δx²]` — running average of squared updates.
    acc_delta: Tensor<B, D>,
}

impl<B: Backend, const D: usize> AdaDeltaState<B, D> {
    pub fn new(square_avg: Tensor<B, D>, acc_delta: Tensor<B, D>) -> Self {
        Self {
            square_avg,
            acc_delta,
        }
    }
}

/// AdaDelta optimizer. Build via [`AdaDeltaConfig`].
#[derive(Clone)]
pub struct AdaDelta {
    rho: f32,
    epsilon: f32,
}

/// Configuration for [`AdaDelta`]. Defaults match `torch.optim.Adadelta`
/// (ρ = 0.9, ε = 1e-6) — the values dHBV runs with.
#[derive(Debug, Clone)]
pub struct AdaDeltaConfig {
    pub rho: f32,
    pub epsilon: f32,
}

impl Default for AdaDeltaConfig {
    fn default() -> Self {
        Self {
            rho: 0.9,
            epsilon: 1e-6,
        }
    }
}

impl AdaDeltaConfig {
    pub fn build(&self) -> AdaDelta {
        AdaDelta {
            rho: self.rho,
            epsilon: self.epsilon,
        }
    }

    /// Initialize an AdaDelta optimizer for module `M`.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<AdaDelta, M, B> {
        OptimizerAdaptor::from(self.build())
    }
}

impl<B: Backend> SimpleOptimizer<B> for AdaDelta {
    type State<const D: usize> = AdaDeltaState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let device = tensor.device();
        let shape = tensor.shape();
        let (square_avg, acc_delta) = match state {
            Some(s) => (s.square_avg, s.acc_delta),
            None => (
                Tensor::zeros(shape.clone(), &device),
                Tensor::zeros(shape, &device),
            ),
        };

        // square_avg = ρ·square_avg + (1-ρ)·g²
        let square_avg = square_avg.mul_scalar(self.rho)
            + (grad.clone() * grad.clone()).mul_scalar(1.0 - self.rho);
        // delta = sqrt(acc_delta + ε) / sqrt(square_avg + ε) · g
        let std = square_avg.clone().add_scalar(self.epsilon).sqrt();
        let delta = acc_delta.clone().add_scalar(self.epsilon).sqrt() / std * grad;
        // acc_delta = ρ·acc_delta + (1-ρ)·delta²
        let acc_delta = acc_delta.mul_scalar(self.rho)
            + (delta.clone() * delta.clone()).mul_scalar(1.0 - self.rho);

        let updated = tensor - delta.mul_scalar(lr as f32);
        (updated, Some(AdaDeltaState::new(square_avg, acc_delta)))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.square_avg = state.square_avg.to_device(device);
        state.acc_delta = state.acc_delta.to_device(device);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::TensorData;

    type B = Autodiff<NdArray<f32>>;
    type I = NdArray<f32>;

    fn t(vals: Vec<f32>) -> Tensor<I, 1> {
        let n = vals.len();
        Tensor::from_data(TensorData::new(vals, [n]), &Default::default())
    }

    #[test]
    fn first_step_matches_pytorch_hand_computation() {
        // From zero state with ρ=0.9, ε=1e-6, lr=1.0, g=1.0:
        //   square_avg = 0.1·1 = 0.1
        //   std        = sqrt(0.1 + 1e-6) = 0.31623...
        //   delta      = sqrt(1e-6)/std · 1 = 0.001/0.316229 = 3.16228e-3
        //   param      = 1.0 - 3.16228e-3
        let opt = AdaDeltaConfig::default().build();
        let param = t(vec![1.0]);
        let grad = t(vec![1.0]);
        let (out, state) = SimpleOptimizer::<I>::step(&opt, 1.0, param, grad, None);
        let got: f32 = out.into_data().to_vec::<f32>().unwrap()[0];
        let expected_delta = (1e-6f32).sqrt() / (0.1f32 + 1e-6).sqrt();
        assert!(
            (got - (1.0 - expected_delta)).abs() < 1e-7,
            "got {got}, want {}",
            1.0 - expected_delta
        );
        // State must be populated for the next step.
        let s = state.expect("state returned");
        let sq: f32 = s.square_avg.into_data().to_vec::<f32>().unwrap()[0];
        let ad: f32 = s.acc_delta.into_data().to_vec::<f32>().unwrap()[0];
        assert!((sq - 0.1).abs() < 1e-7, "square_avg {sq}");
        assert!(
            (ad - 0.1 * expected_delta.powi(2)).abs() < 1e-12,
            "acc_delta {ad}"
        );
    }

    #[test]
    fn step_is_scale_free_in_gradient_magnitude() {
        // The headline property vs Adam-at-fixed-lr: multiplying the gradient
        // by 1000 does NOT multiply the step by 1000 — from zero state the
        // step saturates at sqrt(ε) as |g| grows (delta = sqrt(ε)·g/sqrt(0.1g²+ε)
        // → sqrt(ε)/sqrt(0.1) = sqrt(10ε)).
        let opt = AdaDeltaConfig::default().build();
        let small: f32 = {
            let (o, _) = SimpleOptimizer::<I>::step(&opt, 1.0, t(vec![0.0]), t(vec![1.0]), None);
            -o.into_data().to_vec::<f32>().unwrap()[0]
        };
        let huge: f32 = {
            let (o, _) = SimpleOptimizer::<I>::step(&opt, 1.0, t(vec![0.0]), t(vec![1000.0]), None);
            -o.into_data().to_vec::<f32>().unwrap()[0]
        };
        assert!(huge / small < 1.01, "step scaled with |g|: {small} vs {huge}");
        let saturation = (10.0f32 * 1e-6).sqrt();
        assert!((huge - saturation).abs() < 1e-6, "huge step {huge}");
    }

    #[test]
    fn accumulated_steps_grow_beyond_the_first() {
        // AdaDelta's whole point for a low-update-count run: once acc_delta
        // fills, later steps are LARGER than the first (unlike Adam, whose
        // step is pinned at ~lr). Feed a constant gradient and check the step
        // sequence is increasing.
        let opt = AdaDeltaConfig::default().build();
        let mut param = t(vec![0.0]);
        let mut state = None;
        let mut steps = Vec::new();
        for _ in 0..10 {
            let prev: f32 = param.clone().into_data().to_vec::<f32>().unwrap()[0];
            let (p, s) = SimpleOptimizer::<I>::step(&opt, 1.0, param, t(vec![1.0]), state);
            let now: f32 = p.clone().into_data().to_vec::<f32>().unwrap()[0];
            steps.push(prev - now);
            param = p;
            state = s;
        }
        assert!(
            steps[9] > steps[0],
            "steps did not grow: first {} last {}",
            steps[0],
            steps[9]
        );
    }

    #[test]
    fn zero_gradient_leaves_parameter_unchanged() {
        let opt = AdaDeltaConfig::default().build();
        let (out, _) = SimpleOptimizer::<I>::step(&opt, 1.0, t(vec![0.5]), t(vec![0.0]), None);
        let got: f32 = out.into_data().to_vec::<f32>().unwrap()[0];
        assert!((got - 0.5).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn optimizes_a_kan_head_end_to_end() {
        // Smoke: the adaptor wires up and a real head's loss decreases.
        use crate::nn::kan_head::KanHeadConfig;
        use burn::module::Module;
        use burn::optim::{GradientsParams, Optimizer};
        let device = Default::default();
        let mut head = KanHeadConfig::new(
            (0..3).map(|i| format!("attr_{i}")).collect(),
            vec!["n".into()],
            42,
        )
        .with_hidden_size(4)
        .with_num_hidden_layers(1)
        .init::<B>(&device);
        let mut optim = AdaDeltaConfig::default().init::<B, _>();
        let x: Tensor<B, 2> =
            Tensor::random([8, 3], burn::tensor::Distribution::Default, &device);

        let loss_of = |h: &crate::nn::kan_head::KanHead<B>| -> Tensor<B, 1> {
            let out = h.forward(x.clone());
            let n = out.get("n").expect("n").clone();
            (n.clone() * n).mean()
        };
        let before: f32 = loss_of(&head).into_scalar();
        for _ in 0..5 {
            let loss = loss_of(&head);
            let grads = GradientsParams::from_grads(loss.backward(), &head);
            head = optim.step(1.0, head, grads);
        }
        let after: f32 = loss_of(&head).into_scalar();
        assert!(after < before, "loss did not decrease: {before} -> {after}");
    }
}
