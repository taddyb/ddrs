//! Learnable mass-preserving daily→hourly disaggregation head.
//!
//! Replaces the flat `repeat-24` upsampling (`daily_to_hourly_trim`,
//! `src/data/store/icechunk.rs`) with a learned within-day shape so that
//! routing's sub-daily lag/attenuation is no longer averaged into the
//! daily-mean loss's null-space (the diagnosed cause of the routing-parameter
//! gradient vanishing — X stuck at its 0.246 init across three runs).
//!
//! For reach `r`, day `d`, hour `k ∈ 0..24`:
//! ```text
//!   hourly[d·24 + k, r] = daily[d, r] · 24 · softmax(logits[d, r, :])[k]
//! ```
//! The 24-hour mean equals `daily[d, r]` exactly (softmax sums to 1), so mass
//! is conserved and the daily level is untouched — the head only redistributes
//! within the day. Output is non-negative (daily Q' ≥ 0).
//!
//! **Shape is dynamic, conditioned on the forcing**: a small MLP sees a 3-tap
//! window of the reach's own daily log-Q' `[d-1, d, d+1]` (edge-clamped) so it
//! can read the rising/falling limb, optionally concatenated with the reach's
//! static attributes. There is no hourly precip globally, and the daily Q' is
//! already post-dHBV-UH — so this is a learned *prior*, not recovery of the
//! true sub-daily signal; its job is to unstick the gradient and let routing
//! act. The output layer is zero-initialized → uniform softmax → **byte-exact
//! `repeat-24` at init** (parity preserved until the head learns).

use burn::config::Config;
use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{backend::Backend, Tensor};
use rand::SeedableRng;

/// log(Q' + EPS) floor, matching the discharge fill/floor used elsewhere.
const LOG_EPS: f32 = 1.0e-3;
/// Fixed temporal window: [d-1, d, d+1].
const WINDOW: usize = 3;
/// Xavier gain on the output (logit) layer. NON-zero so the within-day shape
/// starts genuinely non-flat — this gives the routing parameters (Muskingum X,
/// Manning n) a usable gradient from step 1 instead of the chicken-and-egg
/// slow start a zero-init (exact repeat-24) would impose. Daily mass is
/// conserved at any init, so the summed-Q' baseline is untouched.
const DISAGG_OUTPUT_GAIN: f32 = 1.0;

/// Configuration for the disaggregation head.
#[derive(Config, Debug)]
pub struct DisaggHeadConfig {
    /// Number of static attributes (only the count is used, to size the input
    /// layer when `use_attributes` is set). Mirrors `KanHeadConfig`.
    pub num_attributes: usize,
    /// Seed for deterministic init of the input layer.
    pub seed: u64,
    /// Hidden width of the MLP.
    #[config(default = 16)]
    pub hidden_size: usize,
    /// Concatenate the reach's static attributes with the windowed log-Q'.
    #[config(default = true)]
    pub use_attributes: bool,
    /// Condition the within-day shape on the hourly AORC precip window
    /// `[d-1,d,d+1]` (`WINDOW·24` = 72 features per reach-day). The precip is
    /// normalized in the data-batching layer before it reaches the head; this
    /// head only does the temporal windowing. Off ⇒ daily-Q-only head (parity).
    #[config(default = false)]
    pub use_precip: bool,
}

/// Per-reach-day precip features when `use_precip`: `[d-1,d,d+1]` × 24 hours.
const PRECIP_FEATS: usize = WINDOW * 24;

impl DisaggHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DisaggHead<B> {
        let f = WINDOW
            + if self.use_precip { PRECIP_FEATS } else { 0 }
            + if self.use_attributes { self.num_attributes } else { 0 };
        let h = self.hidden_size;

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let input_weight = crate::nn::init::sample_kaiming_normal_relu(&mut rng, f, h);
        let output_weight =
            crate::nn::init::sample_xavier_normal(&mut rng, h, 24, DISAGG_OUTPUT_GAIN);

        let input = Linear {
            weight: crate::nn::init::to_param_weight::<B>(input_weight, device),
            bias: Some(crate::nn::init::zero_bias_tensor::<B>(h, device)),
        };
        // Output layer small-but-nonzero (xavier) → mildly non-uniform shape at
        // init (bias zero). Mass is still conserved (softmax sums to 1).
        let output = Linear {
            weight: crate::nn::init::to_param_weight::<B>(output_weight, device),
            bias: Some(crate::nn::init::zero_bias_tensor::<B>(24, device)),
        };

        DisaggHead {
            input,
            output,
            use_attributes: self.use_attributes,
            use_precip: self.use_precip,
        }
    }
}

/// Mass-preserving daily→hourly disaggregation head (MLP over windowed log-Q'
/// and, optionally, the hourly precip window).
#[derive(Module, Debug)]
pub struct DisaggHead<B: Backend> {
    pub input: Linear<B>,
    pub output: Linear<B>,
    use_attributes: bool,
    use_precip: bool,
}

impl<B: Backend> DisaggHead<B> {
    /// Disaggregate `daily_q` `(D, N)` into hourly `(n_hourly, N)`, matching
    /// `daily_to_hourly_trim`'s hour→day mapping (`day = h / 24`). `n_hourly`
    /// must be a multiple of 24; `d_use = n_hourly / 24` days are disaggregated
    /// (train: `(rho_days-1)·24` uses `D-1` days; test: `n_days·24` uses all
    /// `D` days). Both window edges are clamped, so `d_use ∈ {D-1, D}` are safe.
    ///
    /// `attrs` is `(N, F)` static per-reach attributes (ignored when
    /// `use_attributes` is false). `precip_hourly` is `(n_hourly, N)` already
    /// normalized in the data-batching layer (ignored, may be empty, when
    /// `use_precip` is false). The daily mean of each output day equals the
    /// corresponding `daily_q` value by construction — regardless of the
    /// conditioning inputs, since the softmax shape only redistributes within
    /// the day.
    pub fn forward(
        &self,
        daily_q: Tensor<B, 2>,
        attrs: Tensor<B, 2>,
        precip_hourly: Tensor<B, 2>,
        n_hourly: usize,
    ) -> Tensor<B, 2> {
        let [d, n] = daily_q.dims();
        debug_assert_eq!(n_hourly % 24, 0, "n_hourly {n_hourly} not a multiple of 24");
        let d_use = n_hourly / 24;
        debug_assert!(d_use >= 1 && d_use <= d, "d_use {d_use} out of [1,{d}]");

        // Windowed daily values, each (d_use, N), both edges clamped.
        let center = daily_q.clone().slice([0..d_use, 0..n]);
        let prev = Tensor::cat(
            vec![
                daily_q.clone().slice([0..1, 0..n]),
                daily_q.clone().slice([0..d_use - 1, 0..n]),
            ],
            0,
        );
        let next = if d_use < d {
            daily_q.clone().slice([1..d_use + 1, 0..n])
        } else {
            // d_use == d: last day's d+1 clamps to the final daily value.
            Tensor::cat(
                vec![
                    daily_q.clone().slice([1..d, 0..n]),
                    daily_q.clone().slice([d - 1..d, 0..n]),
                ],
                0,
            )
        };

        // Log-transform window taps → features (d_use·N, WINDOW), row-major
        // index = day·N + reach.
        let logf = |t: Tensor<B, 2>| t.add_scalar(LOG_EPS).log().reshape([d_use * n, 1]);
        let mut feats = Tensor::cat(vec![logf(prev), logf(center.clone()), logf(next)], 1);

        // Precip window: [d-1,d,d+1] × 24 hours → (d_use·N, 72), same row-major
        // (day·N + reach) ordering as the flow features. Inserted between the
        // log-Q taps and attrs to match `DisaggHeadConfig::init`'s feature order.
        if self.use_precip {
            // (n_hourly, N) → (d_use, 24, N).
            let p = precip_hourly.reshape([d_use, 24, n]);
            let prev_d = Tensor::cat(
                vec![p.clone().slice([0..1, 0..24, 0..n]), p.clone().slice([0..d_use - 1, 0..24, 0..n])],
                0,
            );
            let next_d = Tensor::cat(
                vec![p.clone().slice([1..d_use, 0..24, 0..n]), p.clone().slice([d_use - 1..d_use, 0..24, 0..n])],
                0,
            );
            // (d_use, 72, N) → (d_use, N, 72) → (d_use·N, 72).
            let precip_feats = Tensor::cat(vec![prev_d, p, next_d], 1)
                .swap_dims(1, 2)
                .reshape([d_use * n, PRECIP_FEATS]);
            feats = Tensor::cat(vec![feats, precip_feats], 1);
        }

        if self.use_attributes {
            let fdim = attrs.dims()[1];
            // (N, F) → (1, N, F) → (d_use, N, F) → (d_use·N, F), same row-major
            // (day·N + reach) ordering as the flow features.
            let attr_tiled = attrs
                .unsqueeze_dim::<3>(0)
                .expand([d_use, n, fdim])
                .reshape([d_use * n, fdim]);
            feats = Tensor::cat(vec![feats, attr_tiled], 1);
        }

        // MLP → 24 logits per (day, reach) → softmax over the 24-hour axis.
        let h = silu(self.input.forward(feats));
        let logits = self.output.forward(h).reshape([d_use, n, 24]);
        let shape = softmax(logits, 2); // (d_use, N, 24), sums to 1 over hours

        // hourly[d, r, k] = daily[d, r] · 24 · shape[d, r, k]
        let hourly = center.reshape([d_use, n, 1]) * shape.mul_scalar(24.0);

        // (d_use, N, 24) → (d_use, 24, N) → (d_use·24, N) so row h = day·24 + k,
        // matching daily_to_hourly_trim's hour→day mapping.
        hourly.swap_dims(1, 2).reshape([d_use * 24, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::TensorData;
    use ndarray::Array2;
    type Bp = NdArray<f32>;
    type Ad = Autodiff<NdArray<f32>>;

    fn daily<B: Backend>(rows: &[[f32; 2]]) -> Tensor<B, 2> {
        let d = rows.len();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::<B, 1>::from_data(TensorData::new(flat, [d * 2]), &Default::default())
            .reshape([d, 2])
    }

    #[test]
    fn mass_is_conserved_for_arbitrary_shape() {
        // Non-zero output weights → non-uniform shape; daily mean must still
        // match the input daily value for every (day, reach).
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(4, 7).with_use_attributes(false);
        let mut head = cfg.init::<Bp>(&device);
        // Perturb the output layer away from zero so the shape is non-uniform.
        head.output.weight = crate::nn::init::to_param_weight::<Bp>(
            Array2::<f32>::from_shape_fn((cfg.hidden_size, 24), |(i, j)| {
                0.1 * ((i + 2 * j) as f32).sin()
            }),
            &device,
        );
        let q = daily::<Bp>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5], [2.0, 9.0]]); // D=4
        let attrs = Tensor::<Bp, 2>::zeros([2, 4], &device);
        let hourly = head.forward(q.clone(), attrs, Tensor::<Bp, 2>::zeros([72, 2], &device), 72); // train trim: 3 days
        // Per-day mean over its 24 hours == daily value (days 0..2).
        for d in 0..3usize {
            let block = hourly.clone().slice([d * 24..(d + 1) * 24, 0..2]);
            let day_mean = block.mean_dim(0).reshape([2]); // (2,)
            let got: Vec<f32> = day_mean.into_data().to_vec().unwrap();
            let want: Vec<f32> = q.clone().slice([d..d + 1, 0..2]).reshape([2]).into_data().to_vec().unwrap();
            assert!((got[0] - want[0]).abs() < 1e-4, "day {d} reach0 {got:?} vs {want:?}");
            assert!((got[1] - want[1]).abs() < 1e-4, "day {d} reach1 {got:?} vs {want:?}");
        }
    }

    #[test]
    fn test_mode_uses_all_days_with_edge_clamp() {
        // Test-mode trim: n_hourly = D·24 (d_use == D); the last day's right
        // window tap clamps to itself. Mass must still be conserved on all D days.
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(4, 3).with_use_attributes(false);
        let mut head = cfg.init::<Bp>(&device);
        head.output.weight = crate::nn::init::to_param_weight::<Bp>(
            Array2::<f32>::from_elem((cfg.hidden_size, 24), 0.07),
            &device,
        );
        let q = daily::<Bp>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5]]); // D=3
        let attrs = Tensor::<Bp, 2>::zeros([2, 3], &device);
        let hourly = head.forward(q.clone(), attrs, Tensor::<Bp, 2>::zeros([72, 2], &device), 72); // d_use = 3 == D
        for d in 0..3usize {
            let day_mean = hourly.clone().slice([d * 24..(d + 1) * 24, 0..2]).mean_dim(0).reshape([2]);
            let got: Vec<f32> = day_mean.into_data().to_vec().unwrap();
            let want: Vec<f32> = q.clone().slice([d..d + 1, 0..2]).reshape([2]).into_data().to_vec().unwrap();
            assert!((got[0] - want[0]).abs() < 1e-4 && (got[1] - want[1]).abs() < 1e-4, "day {d}: {got:?} vs {want:?}");
        }
    }

    #[test]
    fn default_init_is_nonflat_but_conserves_mass() {
        // The default (xavier) init must (a) NOT be flat — within-day variation
        // exists so routing gets a gradient — yet (b) preserve each day's mean.
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(4, 1);
        let head = cfg.init::<Bp>(&device);
        let q = daily::<Bp>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5], [2.0, 9.0]]);
        let attrs = Tensor::<Bp, 2>::from_data(
            TensorData::new((0..8).map(|x| x as f32).collect::<Vec<_>>(), [2, 4]),
            &device,
        );
        let hourly = head.forward(q.clone(), attrs, Tensor::<Bp, 2>::zeros([72, 2], &device), 72); // (72, 2)
        let v: Vec<f32> = hourly.clone().into_data().to_vec().unwrap();
        // (a) Day 0 reach0 hours must NOT all equal 5.0 (non-flat).
        let day0_reach0: Vec<f32> = (0..24).map(|k| v[k * 2]).collect();
        let spread = day0_reach0.iter().cloned().fold(f32::MIN, f32::max)
            - day0_reach0.iter().cloned().fold(f32::MAX, f32::min);
        assert!(spread > 1e-3, "default init is flat (spread {spread}) — no gradient to routing");
        // (b) Per-day mean still equals the daily value (mass conserved).
        for d in 0..3usize {
            let day_mean = hourly.clone().slice([d * 24..(d + 1) * 24, 0..2]).mean_dim(0).reshape([2]);
            let got: Vec<f32> = day_mean.into_data().to_vec().unwrap();
            let want: Vec<f32> = q.clone().slice([d..d + 1, 0..2]).reshape([2]).into_data().to_vec().unwrap();
            assert!((got[0] - want[0]).abs() < 1e-4 && (got[1] - want[1]).abs() < 1e-4, "day {d}: {got:?} vs {want:?}");
        }
    }

    #[test]
    fn gradient_flows_to_output_layer() {
        // Backprop a daily-mean-sensitive scalar through the head; the output
        // weight must receive a nonzero gradient (the property repeat-24 lacks).
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(4, 2).with_use_attributes(false);
        let mut head = cfg.init::<Ad>(&device);
        // Non-zero output so the softmax Jacobian is non-degenerate.
        head.output.weight = crate::nn::init::to_param_weight::<Ad>(
            Array2::<f32>::from_elem((cfg.hidden_size, 24), 0.05),
            &device,
        );
        let q = daily::<Ad>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5]]); // D=3 → 2 days
        let attrs = Tensor::<Ad, 2>::zeros([2, 4], &device);
        let hourly = head.forward(q, attrs, Tensor::<Ad, 2>::zeros([48, 2], &device), 48); // (48, 2)
        // A loss that depends on within-day distribution, not just daily mean:
        // weight hours linearly so redistribution changes the value.
        let t = hourly.dims()[0];
        let ramp = Tensor::<Ad, 1>::from_data(
            TensorData::new((0..t).map(|i| i as f32).collect::<Vec<_>>(), [t]),
            &device,
        )
        .reshape([t, 1]);
        let loss = (hourly * ramp).sum();
        let grads = loss.backward();
        let g = head.output.weight.val().grad(&grads).unwrap();
        let gsum: f32 = g.abs().sum().into_scalar();
        assert!(gsum > 1e-6, "output-layer gradient vanished: {gsum}");
    }

    /// Build a `(n_hourly, 2)` precip tensor from a per-(hour,reach) closure.
    fn precip<B: Backend>(n_hourly: usize, f: impl Fn(usize, usize) -> f32) -> Tensor<B, 2> {
        let mut v = Vec::with_capacity(n_hourly * 2);
        for h in 0..n_hourly {
            for r in 0..2 {
                v.push(f(h, r));
            }
        }
        Tensor::<B, 1>::from_data(TensorData::new(v, [n_hourly * 2]), &Default::default())
            .reshape([n_hourly, 2])
    }

    #[test]
    fn precip_head_conserves_mass() {
        // With use_precip, mass must STILL be conserved exactly (the softmax
        // shape only redistributes within the day) for arbitrary precip input.
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(4, 11)
            .with_use_attributes(true)
            .with_use_precip(true);
        let head = cfg.init::<Bp>(&device);
        let q = daily::<Bp>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5], [2.0, 9.0]]); // D=4
        let attrs = Tensor::<Bp, 2>::from_data(
            TensorData::new((0..8).map(|x| 0.1 * x as f32).collect::<Vec<_>>(), [2, 4]),
            &device,
        );
        // A storm pulse mid-window so the shape is genuinely non-flat.
        let p = precip::<Bp>(72, |h, r| if h % 24 == 13 { 5.0 + r as f32 } else { 0.1 });
        let hourly = head.forward(q.clone(), attrs, p, 72);
        for d in 0..3usize {
            let day_mean = hourly.clone().slice([d * 24..(d + 1) * 24, 0..2]).mean_dim(0).reshape([2]);
            let got: Vec<f32> = day_mean.into_data().to_vec().unwrap();
            let want: Vec<f32> =
                q.clone().slice([d..d + 1, 0..2]).reshape([2]).into_data().to_vec().unwrap();
            assert!(
                (got[0] - want[0]).abs() < 1e-4 && (got[1] - want[1]).abs() < 1e-4,
                "precip head broke mass on day {d}: {got:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn precip_drives_shape_and_gradient() {
        // The within-day shape must respond to the precip input, and a
        // within-day-sensitive loss must backprop a nonzero gradient through
        // the precip-fed input layer.
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(4, 5)
            .with_use_attributes(false)
            .with_use_precip(true);
        let mut head = cfg.init::<Ad>(&device);
        // Non-degenerate output layer so the softmax Jacobian is non-zero.
        head.output.weight = crate::nn::init::to_param_weight::<Ad>(
            Array2::<f32>::from_shape_fn((cfg.hidden_size, 24), |(i, j)| 0.05 * ((i + j) as f32).cos()),
            &device,
        );
        let q = daily::<Ad>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5]]); // D=3 → 2 days
        let attrs = Tensor::<Ad, 2>::zeros([2, 4], &device);

        // Two different precip patterns → different within-day shapes.
        let p_morning = precip::<Ad>(48, |h, _| if h % 24 == 3 { 9.0 } else { 0.0 });
        let p_evening = precip::<Ad>(48, |h, _| if h % 24 == 20 { 9.0 } else { 0.0 });
        let out_m = head.forward(q.clone(), attrs.clone(), p_morning.clone(), 48);
        let out_e = head.forward(q.clone(), attrs.clone(), p_evening, 48);
        let diff: f32 = (out_m.clone() - out_e).abs().sum().into_scalar();
        assert!(diff > 1e-4, "precip pattern did not change the shape (diff {diff})");

        // Gradient flows through the input layer (which the precip feeds).
        let t = out_m.dims()[0];
        let ramp = Tensor::<Ad, 1>::from_data(
            TensorData::new((0..t).map(|i| i as f32).collect::<Vec<_>>(), [t]),
            &device,
        )
        .reshape([t, 1]);
        let loss = (out_m * ramp).sum();
        let grads = loss.backward();
        let g = head.input.weight.val().grad(&grads).unwrap();
        let gsum: f32 = g.abs().sum().into_scalar();
        assert!(gsum > 1e-6, "input-layer gradient vanished with precip: {gsum}");
    }

    /// Mass balance across a 7-day window: the routing forcing carries the same
    /// total water whether it is upsampled by the flat `repeat-24`
    /// interpolation (no NN) or by the disaggregation head — with OR without
    /// the precip-timing NN. The head only *redistributes* water within each
    /// day; it must neither add nor remove any over the week.
    #[test]
    fn seven_day_mass_balance_interp_vs_disagg() {
        use crate::data::store::icechunk::daily_to_hourly_trim;
        let device = Default::default();

        // 8 days of daily Q' for 2 reaches (day 7 only feeds the `d+1` window
        // tap; `n_hourly = 7·24` disaggregates days 0..6).
        let rows: [[f32; 2]; 8] = [
            [5.0, 1.0], [20.0, 3.0], [8.0, 0.5], [2.0, 9.0],
            [12.0, 4.0], [7.0, 6.0], [3.0, 2.0], [9.0, 1.5],
        ];
        let q = daily::<Bp>(&rows); // (8, 2)
        let q_nd = Array2::<f32>::from_shape_vec(
            (8, 2),
            rows.iter().flatten().copied().collect(),
        )
        .unwrap();
        let n_hourly = 7 * 24; // 168

        // --- Path A: flat repeat-24 interpolation (no NN) ---
        let interp = daily_to_hourly_trim(&q_nd, n_hourly); // (168, 2)
        let interp_tot: [f64; 2] = [
            (0..n_hourly).map(|h| interp[(h, 0)] as f64).sum(),
            (0..n_hourly).map(|h| interp[(h, 1)] as f64).sum(),
        ];
        // Sanity: equals 24 · (7-day daily sum) per reach.
        for r in 0..2 {
            let want = 24.0 * (0..7).map(|d| rows[d][r] as f64).sum::<f64>();
            assert!(
                (interp_tot[r] - want).abs() < 1e-3,
                "interp 7-day mass reach{r}: {} vs {want}",
                interp_tot[r]
            );
        }

        // --- Path B: disaggregation head, precip-OFF then precip-ON ---
        for use_precip in [false, true] {
            let cfg = DisaggHeadConfig::new(4, 13)
                .with_use_attributes(false)
                .with_use_precip(use_precip);
            let head = cfg.init::<Bp>(&device);
            // Non-trivial precip so the within-day shape is genuinely non-flat
            // (ignored when use_precip is false).
            let p = precip::<Bp>(n_hourly, |h, r| if h % 24 == 3 + 5 * r { 6.0 } else { 0.2 });
            let attrs = Tensor::<Bp, 2>::zeros([2, 4], &device);
            let hourly = head.forward(q.clone(), attrs, p, n_hourly); // (168, 2)
            let v: Vec<f32> = hourly.into_data().to_vec().unwrap(); // row-major (168, 2)
            for r in 0..2 {
                let tot: f64 = (0..n_hourly).map(|h| v[h * 2 + r] as f64).sum();
                let rel = (tot - interp_tot[r]).abs() / interp_tot[r];
                assert!(
                    rel < 1e-4,
                    "7-day mass mismatch reach{r} (precip={use_precip}): \
                     disagg {tot} vs interp {} (rel {rel:.2e})",
                    interp_tot[r]
                );
            }
        }
    }
}
