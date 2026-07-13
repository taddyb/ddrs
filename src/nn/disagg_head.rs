//! Learnable mass-preserving daily→hourly disaggregation head (KAN-based).
//!
//! Replaces the flat `repeat-24` upsampling (`daily_to_hourly_trim`,
//! `src/data/store/icechunk.rs`) with a learned within-day shape so that
//! routing's sub-daily lag/attenuation is no longer averaged into the
//! daily-mean loss's null-space.
//!
//! For reach `r`, day `d`, hour `k ∈ 0..24`:
//! ```text
//!   hourly[d·24 + k, r] = daily[d, r] · 24 · softmax(logits[d, r, :])[k]
//! ```
//! The 24-hour mean equals `daily[d, r]` exactly (softmax sums to 1), so mass
//! is conserved and the daily level is untouched — the head only redistributes
//! within the day, exactly like the flat `repeat-24` baseline it replaces
//! (mass-preserving is a property of ANY within-day redistribution, not
//! specific to this head — the head only changes the SHAPE, never the mean).
//!
//! **Input contract (deliberately narrow): `(daily Q′ value, that same day's
//! 24-hour precip) → 24 hourly values`.** No cross-day window, no static
//! attributes, no temperature — those are the routing KAN head's job
//! (`src/nn/kan_head.rs`, which consumes catchment attributes to predict
//! Manning's n / q_spatial / p_spatial). This head does exactly one thing:
//! redistribute a daily rate across its own 24 hours, conditioned on that
//! day's own precip timing.
//!
//! **The interior network is a KAN** (`rskan::KanLayer`, the same primitive
//! the routing head uses): `Linear(25, H) → KanLayer(H, H) × num_hidden_layers
//! → Linear(H, 24)`, no inter-block activation (matches `KanHead`'s
//! convention). The output layer is Xavier-init with a small non-zero gain
//! (not zero) so the within-day shape starts genuinely non-flat — this gives
//! the routing parameters (Muskingum X, Manning n) a usable gradient from
//! step 1 instead of the chicken-and-egg slow start a zero-init (exact
//! repeat-24) would impose. Mass is conserved at any init.
//!
//! **Day-boundary continuity (`boundary_blend`, λ):** since each day is
//! disaggregated independently (no cross-day input at all in this narrow
//! contract), nothing otherwise prevents day `d`'s and day `d+1`'s
//! independently-softmaxed shapes from disagreeing sharply at the hour
//! 23→0 seam. AFTER the softmax, the last-hour shape probability of day `d`
//! and the first-hour shape probability of day `d+1` are blended toward
//! their mean by λ ∈ [0,1] (`0` = fully independent per-day shapes, today's
//! baseline behavior; `1` = the two boundary shape fractions are forced
//! exactly equal), with the day's other 22 hours explicitly rescaled so the
//! day's shape still sums to exactly 1. This is a **shape-level** continuity
//! guarantee — if two adjacent days have genuinely different daily means,
//! the raw disaggregated hourly VALUE can still jump at the boundary (that's
//! mass conservation doing its job correctly, not a bug); only the
//! within-day redistribution FRACTION is smoothed.
//!
//! **Why post-softmax, not pre-softmax.** An earlier version of this blend
//! operated on the raw LOGITS before the softmax, reasoning that "softmax
//! always sums to 1 regardless of its input" made mass conservation safe.
//! That's true, but it doesn't make the blend do what it's supposed to:
//! equalizing two raw logits across two INDEPENDENT per-day softmaxes does
//! NOT equalize the resulting probabilities in general, because each day's
//! softmax normalizes against its own, unrelated other-23-hour scale — two
//! days with different overall logit magnitudes can have identical boundary
//! logits yet meaningfully different boundary probabilities. Blending the
//! post-softmax probabilities directly, then explicitly renormalizing the
//! remaining hours, is what actually gives the guarantee: the boundary
//! probability gap scales by exactly `(1 - λ)` regardless of the day's other
//! values (`boundary_blend_gap_scales_exactly_by_one_minus_lambda`), and mass
//! conservation is enforced by construction (explicit rescale back to sum-1)
//! rather than merely inherited from softmax.

use burn::config::Config;
use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::activation::softmax;
use burn::tensor::{backend::Backend, Tensor};
use rand::SeedableRng;
use rskan::{KanLayer, KanLayerConfig};

/// log(Q' + EPS) floor, matching the discharge fill/floor used elsewhere.
const LOG_EPS: f32 = 1.0e-3;
/// Input features: 1 (log daily Q') + 24 (that day's hourly precip).
const NUM_FEATURES: usize = 25;
/// Xavier gain on the output (logit) layer. NON-zero so the within-day shape
/// starts genuinely non-flat (see module docs). Daily mass is conserved at
/// any init.
const DISAGG_OUTPUT_GAIN: f32 = 1.0;

/// Configuration for the disaggregation head.
#[derive(Config, Debug)]
pub struct DisaggHeadConfig {
    /// Seed for deterministic init (both `Linear` layers and every inner
    /// `KanLayer` — same seed everywhere, matching `KanHead`'s convention).
    pub seed: u64,
    /// Hidden width of the KAN.
    #[config(default = 16)]
    pub hidden_size: usize,
    /// Number of `KanLayer(H, H)` blocks between input and output.
    #[config(default = 1)]
    pub num_hidden_layers: usize,
    /// B-spline grid intervals (`num` in pykan/rskan).
    #[config(default = 3)]
    pub grid: usize,
    /// B-spline order. `3` per cubic-spline default.
    #[config(default = 3)]
    pub k: usize,
    /// Day-boundary shape-continuity blend factor λ ∈ [0, 1]. `0` (default)
    /// reproduces fully independent per-day shapes (no continuity
    /// enforcement); `1` forces the boundary shape fractions exactly equal.
    /// Only used when `chunk_days <= 1` — see `chunk_days` doc.
    #[config(default = 0.0)]
    pub boundary_blend: f32,
    /// Mass-balance granularity, in days. `1` (default) is the ORIGINAL
    /// contract: each individual calendar day gets its own independent
    /// softmax and must sum to exactly that day's `daily_q` (config absent
    /// ⇒ byte-identical to every checkpoint trained before this field
    /// existed). `> 1` relaxes the constraint to a JOINT softmax over
    /// `chunk_days` consecutive days at once: only the WHOLE chunk's mean
    /// must equal the chunk's mean `daily_q` — an individual day within the
    /// chunk is free to receive more or less than its own observed value,
    /// which is what lets a storm that spans a day boundary be represented
    /// as one continuous rise instead of being artificially clipped at
    /// hour 0/23. Verified (`docs/2026-07-1x-disagg-72h-window-findings.md`):
    /// a standalone 3-day-chunk pretrain on real USGS hourly data showed
    /// smooth cross-day storm shapes (vs. climatology's persistent blocky
    /// steps at the old day boundaries) and a real peak-timing improvement
    /// (median error 6h vs. climatology's 9h on a 72h scale), whereas
    /// `boundary_blend` alone (post-hoc smoothing, no relaxed constraint)
    /// left peak timing exactly tied with climatology. `d_use` need not be
    /// a multiple of `chunk_days` — a shorter remainder chunk at the end is
    /// handled by padding (repeating the last real day) up to `chunk_days`,
    /// forwarding, then keeping only the real days' hours.
    #[config(default = 1)]
    pub chunk_days: usize,
}

impl DisaggHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DisaggHead<B> {
        let h = self.hidden_size;
        let chunk_days = self.chunk_days.max(1);
        // chunk_days=1: 1 (log daily_q) + 24 (precip) = 25, matching the
        // original NUM_FEATURES exactly. chunk_days>1: chunk_days (one log
        // daily_q per day) + chunk_days*24 (precip) features; output width
        // is chunk_days*24 (one joint softmax over the whole chunk).
        let input_dim = if chunk_days <= 1 { NUM_FEATURES } else { chunk_days * NUM_FEATURES };
        let output_dim = if chunk_days <= 1 { 24 } else { chunk_days * 24 };

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let input_weight = crate::nn::init::sample_kaiming_normal_relu(&mut rng, input_dim, h);
        let output_weight =
            crate::nn::init::sample_xavier_normal(&mut rng, h, output_dim, DISAGG_OUTPUT_GAIN);

        let input = Linear {
            weight: crate::nn::init::to_param_weight::<B>(input_weight, device),
            bias: Some(crate::nn::init::zero_bias_tensor::<B>(h, device)),
        };
        // Output layer small-but-nonzero (xavier) → mildly non-uniform shape at
        // init (bias zero). Mass is still conserved (softmax sums to 1).
        let output = Linear {
            weight: crate::nn::init::to_param_weight::<B>(output_weight, device),
            bias: Some(crate::nn::init::zero_bias_tensor::<B>(output_dim, device)),
        };

        // Same seed passed to every inner KanLayer (matches KanHead's
        // documented quirk — reproducibility over exact pykan parity).
        let hidden: Vec<KanLayer<B>> = (0..self.num_hidden_layers)
            .map(|_| {
                KanLayerConfig::new(h, h, self.seed)
                    .with_num(self.grid)
                    .with_k(self.k)
                    .init(device)
            })
            .collect();

        DisaggHead {
            input,
            hidden,
            output,
            boundary_blend: self.boundary_blend,
            chunk_days,
        }
    }
}

/// Mass-preserving daily→hourly disaggregation head. KAN interior
/// (`Linear → KanLayer × L → Linear`) over `(log daily Q', that day's 24h
/// precip)`.
#[derive(Module, Debug)]
pub struct DisaggHead<B: Backend> {
    pub input: Linear<B>,
    pub hidden: Vec<KanLayer<B>>,
    pub output: Linear<B>,
    boundary_blend: f32,
    chunk_days: usize,
}

impl<B: Backend> DisaggHead<B> {
    /// Disaggregate `daily_q` `(D, N)` into hourly `(n_hourly, N)`, matching
    /// `daily_to_hourly_trim`'s hour→day mapping (`day = h / 24`). `n_hourly`
    /// must be a multiple of 24; `d_use = n_hourly / 24` days are
    /// disaggregated (train: `(rho_days-1)·24` uses `D-1` days; test:
    /// `n_days·24` uses all `D` days).
    ///
    /// `precip_hourly` is `(n_hourly, N)` — that same `d_use`-day window's
    /// hourly precip, already normalized in the data-batching layer. The
    /// daily mean of each output day equals the corresponding `daily_q`
    /// value by construction, regardless of the precip conditioning or the
    /// boundary blend — both act strictly upstream of the per-day softmax.
    pub fn forward(
        &self,
        daily_q: Tensor<B, 2>,
        precip_hourly: Tensor<B, 2>,
        n_hourly: usize,
    ) -> Tensor<B, 2> {
        let [d, n] = daily_q.dims();
        debug_assert_eq!(n_hourly % 24, 0, "n_hourly {n_hourly} not a multiple of 24");
        let d_use = n_hourly / 24;
        debug_assert!(d_use >= 1 && d_use <= d, "d_use {d_use} out of [1,{d}]");
        debug_assert_eq!(
            precip_hourly.dims(),
            [n_hourly, n],
            "precip_hourly must be exactly (n_hourly, N)"
        );

        if self.chunk_days <= 1 {
            return self.forward_per_day(daily_q, precip_hourly, d_use, n);
        }
        self.forward_chunked(daily_q, precip_hourly, d_use, n)
    }

    /// Original per-day-independent path (`chunk_days <= 1`) — UNCHANGED
    /// from before `chunk_days` existed, byte-identical to every checkpoint
    /// trained under the old contract.
    fn forward_per_day(
        &self,
        daily_q: Tensor<B, 2>,
        precip_hourly: Tensor<B, 2>,
        d_use: usize,
        n: usize,
    ) -> Tensor<B, 2> {
        let center = daily_q.slice([0..d_use, 0..n]); // (d_use, N)

        // Log daily-Q feature: (d_use·N, 1), row-major index = day·N + reach.
        let logq = center
            .clone()
            .add_scalar(LOG_EPS)
            .log()
            .reshape([d_use * n, 1]);

        // That day's own 24h precip: (n_hourly, N) -> (d_use, 24, N) ->
        // (d_use·N, 24), same row-major (day·N + reach) ordering.
        let precip_feat = precip_hourly
            .reshape([d_use, 24, n])
            .swap_dims(1, 2)
            .reshape([d_use * n, 24]);

        let feats = Tensor::cat(vec![logq, precip_feat], 1); // (d_use·N, 25)

        // Linear -> KanLayer x L (no inter-block activation) -> Linear.
        let mut x = self.input.forward(feats);
        for layer in &self.hidden {
            x = layer.forward(x);
        }
        let logits = self.output.forward(x).reshape([d_use, n, 24]);

        // Softmax over the 24-hour axis -> shape sums to 1 by construction.
        let mut shape = softmax(logits, 2); // (d_use, N, 24)

        if self.boundary_blend > 0.0 && d_use > 1 {
            shape = apply_boundary_blend(shape, self.boundary_blend, d_use, n);
        }

        // hourly[d, r, k] = daily[d, r] * 24 * shape[d, r, k]
        let hourly = center.reshape([d_use, n, 1]) * shape.mul_scalar(24.0);

        // (d_use, N, 24) -> (d_use, 24, N) -> (d_use·24, N) so row h = day·24 + k,
        // matching daily_to_hourly_trim's hour->day mapping.
        hourly.swap_dims(1, 2).reshape([d_use * 24, n])
    }

    /// `chunk_days > 1` path: split `d_use` days into non-overlapping
    /// chunks of `chunk_days` (a shorter remainder chunk at the end, if
    /// any, is zero-padded up to `chunk_days` by repeating the last real
    /// day, forwarded normally, then only its real days' hours are kept —
    /// see `chunk_days` doc comment on `DisaggHeadConfig`). Each chunk gets
    /// ONE joint softmax over `chunk_days*24` hours: the chunk's mean must
    /// equal the mean of its `chunk_days` `daily_q` values, but an
    /// individual day within the chunk is free to deviate from its own
    /// observed value — this is the deliberate mass-balance relaxation.
    fn forward_chunked(
        &self,
        daily_q: Tensor<B, 2>,
        precip_hourly: Tensor<B, 2>,
        d_use: usize,
        n: usize,
    ) -> Tensor<B, 2> {
        let cd = self.chunk_days;
        let n_chunks = d_use.div_ceil(cd);
        let mut outputs = Vec::with_capacity(n_chunks);

        for c in 0..n_chunks {
            let start_day = c * cd;
            let real_days = (d_use - start_day).min(cd);

            let daily_real = daily_q.clone().slice([start_day..start_day + real_days, 0..n]); // (real_days, N)
            let precip_real = precip_hourly
                .clone()
                .slice([start_day * 24..(start_day + real_days) * 24, 0..n]); // (real_days*24, N)

            if real_days < cd {
                // Remainder chunk: pad up to `cd` days (repeating the last
                // real day) so the network can be evaluated, but the mass
                // constraint must be against the REAL `real_days`, not the
                // padded `cd` (the padded fake day must not leak volume
                // into the kept output). Slice the joint shape down to the
                // real hours, RENORMALIZE it to sum to 1 over just those
                // hours, then scale by the REAL mean -- not the shape
                // computed over the padded window directly.
                let pad_days = cd - real_days;
                let last_daily = daily_real.clone().slice([real_days - 1..real_days, 0..n]); // (1, N)
                let daily_pad = last_daily.repeat_dim(0, pad_days); // (pad_days, N)
                let last_precip_day = precip_real.clone().slice([(real_days - 1) * 24..real_days * 24, 0..n]); // (24, N)
                let precip_pad = last_precip_day.repeat_dim(0, pad_days); // (pad_days*24, N)
                let daily_chunk = Tensor::cat(vec![daily_real.clone(), daily_pad], 0);
                let precip_chunk = Tensor::cat(vec![precip_real, precip_pad], 0);

                let shape_full = self.forward_joint_shape(daily_chunk, precip_chunk, n); // (N, cd*24)
                let real_hours = real_days * 24;
                let shape_real = shape_full.slice([0..n, 0..real_hours]); // (N, real_hours) -- does NOT sum to 1
                let row_sum = shape_real.clone().sum_dim(1); // (N, 1)
                let shape_renorm = shape_real / row_sum; // renormalized to sum to 1 over just the real hours

                let real_mean = daily_real.mean_dim(0).reshape([n, 1]); // (N, 1), REAL mean only
                let hourly = (shape_renorm * real_mean * (real_hours as f32)).transpose(); // (real_hours, N)
                outputs.push(hourly);
            } else {
                let shape = self.forward_joint_shape(daily_real.clone(), precip_real, n); // (N, cd*24)
                let chunk_mean = daily_real.mean_dim(0).reshape([n, 1]); // (N, 1)
                let hourly = (shape * chunk_mean * ((cd * 24) as f32)).transpose(); // (cd*24, N)
                outputs.push(hourly);
            }
        }

        Tensor::cat(outputs, 0) // (d_use*24, N)
    }

    /// ONE joint softmax over a full `chunk_days`-day window, returned as
    /// the raw shape (sums to 1 per row over the WHOLE `chunk_days*24`
    /// axis) — scaling to physical hourly values is the caller's job (it
    /// differs for full vs. padded-remainder chunks, see `forward_chunked`).
    /// `daily_q_chunk`: (chunk_days, N). `precip_chunk`: (chunk_days*24, N).
    /// Returns (N, chunk_days*24).
    fn forward_joint_shape(&self, daily_q_chunk: Tensor<B, 2>, precip_chunk: Tensor<B, 2>, _n: usize) -> Tensor<B, 2> {
        // Per-day log-Q features, one column per day: (N, cd).
        let logq = daily_q_chunk.add_scalar(LOG_EPS).log().transpose(); // (N, cd)
        // Whole-chunk precip, one column per hour: (N, cd*24).
        let precip_feat = precip_chunk.transpose(); // (N, cd*24)
        let feats = Tensor::cat(vec![logq, precip_feat], 1); // (N, cd*(1+24))

        let mut x = self.input.forward(feats);
        for layer in &self.hidden {
            x = layer.forward(x);
        }
        let logits = self.output.forward(x); // (N, cd*24)
        softmax(logits, 1) // (N, cd*24), sums to 1 per row over the WHOLE chunk
    }
}

/// Blend the last-hour SHAPE PROBABILITY of day `d` and the first-hour shape
/// probability of day `d+1` toward their mean by `lambda`, for every
/// adjacent day pair, then rescale each affected day's remaining 22 hours
/// (1..23) so the day's shape still sums to exactly 1.
///
/// Operates on POST-softmax probabilities, not logits — equalizing two
/// *raw logits* across independent per-day softmaxes does NOT equalize the
/// resulting probabilities in general (each day's softmax normalizes over
/// its own, unrelated other 23 hours, so equal boundary logits land at
/// different probabilities whenever the two days' overall logit scales
/// differ). Blending the probabilities directly and explicitly
/// renormalizing is the only way to get an exact guarantee: the boundary
/// probability gap scales by exactly `(1 - lambda)` regardless of the day's
/// other 23 values (see `boundary_blend_monotonically_reduces_discontinuity`),
/// and mass conservation is enforced by construction (the day's shape is
/// explicitly rescaled back to sum to 1), not merely inherited from softmax.
/// Caller guarantees `d_use >= 2`.
fn apply_boundary_blend<B: Backend>(
    shape: Tensor<B, 3>, // (d_use, N, 24), each day already sums to 1
    lambda: f32,
    d_use: usize,
    n: usize,
) -> Tensor<B, 3> {
    debug_assert!(d_use >= 2, "boundary blend requires at least 2 days");
    const EPS: f32 = 1e-6;

    let p0 = shape.clone().slice([0..d_use, 0..n, 0..1]); // (d_use, N, 1), each day's hour-0 prob
    let p23 = shape.clone().slice([0..d_use, 0..n, 23..24]); // (d_use, N, 1), each day's hour-23 prob
    let rest = shape.slice([0..d_use, 0..n, 1..23]); // (d_use, N, 22), hours 1..22

    // Blend adjacent (day d's p23, day d+1's p0) pairs toward their mean.
    let last = p23.clone().slice([0..d_use - 1, 0..n, 0..1]); // days [0, d_use-2]
    let first = p0.clone().slice([1..d_use, 0..n, 0..1]); // days [1, d_use-1]
    let mean = (last.clone() + first.clone()) / 2;
    let blended_last = last.clone() + (mean.clone() - last) * lambda;
    let blended_first = first.clone() + (mean - first) * lambda;

    // Day (d_use-1) has no next day -> its p23 (q23) is unblended. Day 0 has
    // no previous day -> its p0 (q0) is unblended.
    let last_unblended = p23.clone().slice([d_use - 1..d_use, 0..n, 0..1]);
    let first_unblended = p0.clone().slice([0..1, 0..n, 0..1]);
    let q23 = Tensor::cat(vec![blended_last, last_unblended], 0); // (d_use, N, 1)
    let q0 = Tensor::cat(vec![first_unblended, blended_first], 0); // (d_use, N, 1)

    // Rescale hours 1..22 so each day still sums to exactly 1:
    // new_total = q0 + q23 + s * (old_total - p0 - p23) = 1, old_total = 1.
    let old_other = p0.neg().add_scalar(1.0) - p23; // (d_use, N, 1) = 1 - p0 - p23
    let new_other = q0.clone().neg().add_scalar(1.0) - q23.clone(); // 1 - q0 - q23
    let scale = new_other / (old_other + EPS); // (d_use, N, 1), broadcasts over the 22-dim below
    let new_rest = rest * scale;

    Tensor::cat(vec![q0, new_rest, q23], 2) // (d_use, N, 24), hour order [0, 1..22, 23]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::TensorData;
    use ndarray::Array2;
    type Bp = NdArray<f32>;
    type Ad = Autodiff<NdArray<f32>>;

    /// Isolated test of `apply_boundary_blend` alone (no KAN/network in the
    /// loop), against a case specifically constructed to break the
    /// "equalize the raw logit" bug: the two days' non-boundary hours have
    /// very different overall scale (day1's hour5 = 7.0 vs day0's flat 0.1
    /// background), so a naive pre-softmax logit blend would NOT equalize
    /// the resulting probabilities (each day's softmax normalizes against
    /// its own, different, other-23-hour scale). Verifies the exact
    /// analytic prediction: the boundary probability gap scales by exactly
    /// `(1 - lambda)`, and each day's shape still sums to 1 at every lambda.
    #[test]
    fn boundary_blend_gap_scales_exactly_by_one_minus_lambda() {
        let device = Default::default();
        let mut day0 = vec![0.1f32; 24];
        day0[23] = 5.0;
        let mut day1 = vec![0.1f32; 24];
        day1[0] = 9.0;
        day1[5] = 7.0; // asymmetric overall scale vs day0
        let flat: Vec<f32> = day0.iter().chain(day1.iter()).copied().collect();
        let logits = Tensor::<Bp, 1>::from_data(TensorData::new(flat, [48]), &device).reshape([2, 1, 24]);
        let shape0 = softmax(logits, 2);
        let v0: Vec<f32> = shape0.clone().into_data().to_vec().unwrap();
        let gap0 = (v0[23] - v0[24]).abs();
        assert!(gap0 > 0.01, "test construction should show a real unblended gap: {gap0}");

        for lam in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let blended = apply_boundary_blend(shape0.clone(), lam, 2, 1);
            let v: Vec<f32> = blended.into_data().to_vec().unwrap();
            let gap = (v[23] - v[24]).abs();
            let expected = gap0 * (1.0 - lam);
            assert!(
                (gap - expected).abs() < 1e-5,
                "lambda={lam}: gap {gap} != predicted {expected} (unblended gap {gap0})"
            );
            let day0_sum: f32 = v[0..24].iter().sum();
            let day1_sum: f32 = v[24..48].iter().sum();
            assert!((day0_sum - 1.0).abs() < 1e-5, "lambda={lam}: day0 shape sum {day0_sum} != 1");
            assert!((day1_sum - 1.0).abs() < 1e-5, "lambda={lam}: day1 shape sum {day1_sum} != 1");
        }
    }

    fn daily<B: Backend>(rows: &[[f32; 2]]) -> Tensor<B, 2> {
        let d = rows.len();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::<B, 1>::from_data(TensorData::new(flat, [d * 2]), &Default::default())
            .reshape([d, 2])
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

    fn assert_mass_conserved(hourly: &Tensor<Bp, 2>, q: &Tensor<Bp, 2>, n_days: usize) {
        for d in 0..n_days {
            let day_mean = hourly
                .clone()
                .slice([d * 24..(d + 1) * 24, 0..2])
                .mean_dim(0)
                .reshape([2]);
            let got: Vec<f32> = day_mean.into_data().to_vec().unwrap();
            let want: Vec<f32> = q
                .clone()
                .slice([d..d + 1, 0..2])
                .reshape([2])
                .into_data()
                .to_vec()
                .unwrap();
            assert!(
                (got[0] - want[0]).abs() < 1e-4 && (got[1] - want[1]).abs() < 1e-4,
                "day {d}: {got:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn mass_is_conserved_at_default_init() {
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(7);
        let head = cfg.init::<Bp>(&device);
        let q = daily::<Bp>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5], [2.0, 9.0]]); // D=4
        let p = precip::<Bp>(72, |h, r| if h % 24 == 13 { 5.0 + r as f32 } else { 0.1 });
        let hourly = head.forward(q.clone(), p, 72); // train trim: 3 days
        assert_mass_conserved(&hourly, &q, 3);
    }

    #[test]
    fn test_mode_uses_all_days() {
        // Test-mode trim: n_hourly = D·24 (d_use == D).
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(3);
        let head = cfg.init::<Bp>(&device);
        let q = daily::<Bp>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5]]); // D=3
        let p = precip::<Bp>(72, |h, _| if h % 24 == 8 { 4.0 } else { 0.2 });
        let hourly = head.forward(q.clone(), p, 72); // d_use = 3 == D
        assert_mass_conserved(&hourly, &q, 3);
    }

    #[test]
    fn default_init_is_nonflat_but_conserves_mass() {
        // The default (xavier) init must (a) NOT be flat — within-day variation
        // exists so routing gets a gradient — yet (b) preserve each day's mean.
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(1);
        let head = cfg.init::<Bp>(&device);
        let q = daily::<Bp>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5], [2.0, 9.0]]);
        let p = precip::<Bp>(72, |h, r| if h % 24 == 5 { 3.0 + r as f32 } else { 0.05 });
        let hourly = head.forward(q.clone(), p, 72); // (72, 2)
        let v: Vec<f32> = hourly.clone().into_data().to_vec().unwrap();
        // Day 0 reach0 hours must NOT all equal 5.0 (non-flat).
        let day0_reach0: Vec<f32> = (0..24).map(|k| v[k * 2]).collect();
        let spread = day0_reach0.iter().cloned().fold(f32::MIN, f32::max)
            - day0_reach0.iter().cloned().fold(f32::MAX, f32::min);
        assert!(spread > 1e-3, "default init is flat (spread {spread}) — no gradient to routing");
        assert_mass_conserved(&hourly, &q, 3);
    }

    #[test]
    fn gradient_flows_to_output_and_kan_layers() {
        // Backprop a shape-sensitive scalar through the head; both the
        // output Linear AND the inner KanLayer's spline coefficients must
        // receive a nonzero gradient.
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(2);
        let head = cfg.init::<Ad>(&device);
        let q = daily::<Ad>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5]]); // D=3 -> 2 days
        let p = precip::<Ad>(48, |h, _| if h % 24 == 3 { 9.0 } else { 0.0 });
        let hourly = head.forward(q, p, 48); // (48, 2)
        let t = hourly.dims()[0];
        let ramp = Tensor::<Ad, 1>::from_data(
            TensorData::new((0..t).map(|i| i as f32).collect::<Vec<_>>(), [t]),
            &device,
        )
        .reshape([t, 1]);
        let loss = (hourly * ramp).sum();
        let grads = loss.backward();
        let g_out = head.output.weight.val().grad(&grads).unwrap();
        assert!(g_out.abs().sum().into_scalar() > 1e-6, "output-layer gradient vanished");
        let g_coef = head.hidden[0].coef.val().grad(&grads).unwrap();
        assert!(g_coef.abs().sum().into_scalar() > 1e-6, "KanLayer coef gradient vanished");
    }

    #[test]
    fn precip_drives_shape_and_gradient() {
        // The within-day shape must respond to the precip input, and a
        // within-day-sensitive loss must backprop a nonzero gradient through
        // the precip-fed input layer.
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(5);
        let head = cfg.init::<Ad>(&device);
        let q = daily::<Ad>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5]]); // D=3 -> 2 days

        // Two different precip patterns -> different within-day shapes.
        let p_morning = precip::<Ad>(48, |h, _| if h % 24 == 3 { 9.0 } else { 0.0 });
        let p_evening = precip::<Ad>(48, |h, _| if h % 24 == 20 { 9.0 } else { 0.0 });
        let out_m = head.forward(q.clone(), p_morning, 48);
        let out_e = head.forward(q, p_evening, 48);
        let diff: f32 = (out_m.clone() - out_e).abs().sum().into_scalar();
        assert!(diff > 1e-4, "precip pattern did not change the shape (diff {diff})");

        let t = out_m.dims()[0];
        let ramp = Tensor::<Ad, 1>::from_data(
            TensorData::new((0..t).map(|i| i as f32).collect::<Vec<_>>(), [t]),
            &device,
        )
        .reshape([t, 1]);
        let loss = (out_m * ramp).sum();
        let grads = loss.backward();
        let g = head.input.weight.val().grad(&grads).unwrap();
        assert!(g.abs().sum().into_scalar() > 1e-6, "input-layer gradient vanished with precip");
    }

    #[test]
    fn seven_day_mass_balance_interp_vs_disagg() {
        // The routing forcing carries the same total water whether it is
        // upsampled by the flat `repeat-24` interpolation (no NN) or by the
        // disaggregation head. The head only *redistributes* water within
        // each day; it must neither add nor remove any over the week.
        use crate::data::store::icechunk::daily_to_hourly_trim;
        let device = Default::default();

        let rows: [[f32; 2]; 8] = [
            [5.0, 1.0], [20.0, 3.0], [8.0, 0.5], [2.0, 9.0],
            [12.0, 4.0], [7.0, 6.0], [3.0, 2.0], [9.0, 1.5],
        ];
        let q = daily::<Bp>(&rows); // (8, 2)
        let q_nd = Array2::<f32>::from_shape_vec((8, 2), rows.iter().flatten().copied().collect()).unwrap();
        let n_hourly = 7 * 24; // 168

        let interp = daily_to_hourly_trim(&q_nd, n_hourly); // (168, 2)
        let interp_tot: [f64; 2] = [
            (0..n_hourly).map(|h| interp[(h, 0)] as f64).sum(),
            (0..n_hourly).map(|h| interp[(h, 1)] as f64).sum(),
        ];
        for r in 0..2 {
            let want = 24.0 * (0..7).map(|d| rows[d][r] as f64).sum::<f64>();
            assert!((interp_tot[r] - want).abs() < 1e-3, "interp 7-day mass reach{r}: {} vs {want}", interp_tot[r]);
        }

        let cfg = DisaggHeadConfig::new(13);
        let head = cfg.init::<Bp>(&device);
        let p = precip::<Bp>(n_hourly, |h, r| if h % 24 == 3 + 5 * r { 6.0 } else { 0.2 });
        let hourly = head.forward(q, p, n_hourly); // (168, 2)
        let v: Vec<f32> = hourly.into_data().to_vec().unwrap();
        for r in 0..2 {
            let tot: f64 = (0..n_hourly).map(|h| v[h * 2 + r] as f64).sum();
            let rel = (tot - interp_tot[r]).abs() / interp_tot[r];
            assert!(rel < 1e-4, "7-day mass mismatch reach{r}: disagg {tot} vs interp {} (rel {rel:.2e})", interp_tot[r]);
        }
    }

    // -------------------------------------------------------------------
    // Boundary continuity (lambda)
    // -------------------------------------------------------------------

    /// Build a head with hand-set output weights guaranteed to produce
    /// clearly disagreeing boundary shapes at lambda=0 (day 0's precip spike
    /// at hour 23, day 1's spike at hour 0 — maximally adversarial seam).
    fn adversarial_head(lambda: f32) -> (DisaggHead<Bp>, Tensor<Bp, 2>, Tensor<Bp, 2>) {
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(21).with_boundary_blend(lambda);
        let mut head = cfg.init::<Bp>(&device);
        // Strong, non-degenerate output map so precip timing clearly drives
        // which hour's logit is largest.
        head.output.weight = crate::nn::init::to_param_weight::<Bp>(
            Array2::<f32>::from_shape_fn((cfg.hidden_size, 24), |(i, j)| {
                if i % 24 == j { 3.0 } else { 0.05 * ((i + j) as f32).sin() }
            }),
            &device,
        );
        let q = daily::<Bp>(&[[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]); // D=3 -> 2 days, flat daily means
        // Day 0: storm at hour 23. Day 1: storm at hour 0. Adversarial seam.
        let p = precip::<Bp>(48, |h, r| {
            let hod = h % 24;
            let day = h / 24;
            let is_storm = (day == 0 && hod == 23) || (day == 1 && hod == 0);
            if is_storm { 8.0 + r as f32 } else { 0.1 }
        });
        (head, q, p)
    }

    fn seam_gap(head: &DisaggHead<Bp>, q: Tensor<Bp, 2>, p: Tensor<Bp, 2>) -> f32 {
        let hourly = head.forward(q, p, 48); // (48, 2)
        let v: Vec<f32> = hourly.into_data().to_vec().unwrap();
        // hour 23 of day 0 = row 23; hour 0 of day 1 = row 24. reach 0 (col 0).
        (v[23 * 2] - v[24 * 2]).abs()
    }

    #[test]
    fn boundary_blend_zero_is_byte_identical_to_no_blend() {
        let (head0, q0, p0) = adversarial_head(0.0);
        let out0 = head0.forward(q0, p0, 48);

        let device = Default::default();
        let cfg = DisaggHeadConfig::new(21); // boundary_blend defaults to 0.0
        let mut head_default = cfg.init::<Bp>(&device);
        head_default.output.weight = crate::nn::init::to_param_weight::<Bp>(
            Array2::<f32>::from_shape_fn((cfg.hidden_size, 24), |(i, j)| {
                if i % 24 == j { 3.0 } else { 0.05 * ((i + j) as f32).sin() }
            }),
            &device,
        );
        let q1 = daily::<Bp>(&[[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]);
        let p1 = precip::<Bp>(48, |h, r| {
            let hod = h % 24;
            let day = h / 24;
            let is_storm = (day == 0 && hod == 23) || (day == 1 && hod == 0);
            if is_storm { 8.0 + r as f32 } else { 0.1 }
        });
        let out1 = head_default.forward(q1, p1, 48);

        let diff: f32 = (out0 - out1).abs().sum().into_scalar();
        assert!(diff < 1e-6, "boundary_blend=0.0 (explicit) must match the default (implicit 0.0): diff {diff}");
    }

    #[test]
    fn boundary_blend_one_forces_exact_shape_equality() {
        let (head, q, p) = adversarial_head(1.0);
        let hourly = head.forward(q, p, 48);
        let v: Vec<f32> = hourly.into_data().to_vec().unwrap();
        // At lambda=1 the boundary SHAPE fractions are forced equal; daily
        // means are equal here too (both 10.0), so the raw hourly values at
        // the seam must also be equal.
        let gap0 = (v[23 * 2] - v[24 * 2]).abs();
        let gap1 = (v[23 * 2 + 1] - v[24 * 2 + 1]).abs();
        assert!(gap0 < 1e-4, "reach0 seam gap at lambda=1: {gap0}");
        assert!(gap1 < 1e-4, "reach1 seam gap at lambda=1: {gap1}");
    }

    #[test]
    fn boundary_blend_monotonically_reduces_discontinuity() {
        let lambdas = [0.0, 0.25, 0.5, 0.75, 1.0];
        let gaps: Vec<f32> = lambdas
            .iter()
            .map(|&lam| {
                let (head, q, p) = adversarial_head(lam);
                seam_gap(&head, q, p)
            })
            .collect();
        assert!(gaps[0] > 1e-3, "adversarial construction should show a real gap at lambda=0: {gaps:?}");
        for w in gaps.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-5,
                "seam gap must not increase as lambda grows: {gaps:?}"
            );
        }
        assert!(gaps[4] < 1e-4, "gap must vanish at lambda=1: {gaps:?}");
    }

    #[test]
    fn boundary_blend_preserves_mass_conservation() {
        let (head, q, p) = adversarial_head(0.7);
        let hourly = head.forward(q.clone(), p, 48);
        assert_mass_conserved(&hourly, &q, 2);
    }

    #[test]
    fn boundary_blend_preserves_mass_conservation_across_lambda_sweep() {
        // The single-lambda check above (0.7) is a spot check; this sweeps
        // the full registered range to confirm the post-softmax rescale
        // (scale = (1 - q0 - q23) / (1 - p0 - p23)) holds at every lambda,
        // not just one — including the lambda=1 edge case where the
        // boundary hours are forced fully equal and the rescale factor is
        // furthest from 1.0.
        for lambda in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let (head, q, p) = adversarial_head(lambda);
            let hourly = head.forward(q.clone(), p, 48);
            assert_mass_conserved(&hourly, &q, 2);
        }
    }

    // -----------------------------------------------------------------------
    // chunk_days > 1: relaxed (chunk-aggregate, not per-day) mass balance
    // -----------------------------------------------------------------------

    /// Asserts the MEAN over `chunk_days` consecutive days of `hourly`
    /// equals the mean of the corresponding `chunk_days` `daily_q` values —
    /// the chunk-aggregate invariant, deliberately weaker than per-day
    /// exactness (see `assert_mass_conserved` for the `chunk_days<=1` one).
    fn assert_chunk_mass_conserved(hourly: &Tensor<Bp, 2>, q: &Tensor<Bp, 2>, day_start: usize, chunk_days: usize) {
        let n = hourly.dims()[1];
        let chunk_mean_hourly = hourly
            .clone()
            .slice([day_start * 24..(day_start + chunk_days) * 24, 0..n])
            .mean_dim(0)
            .reshape([n]);
        let got: Vec<f32> = chunk_mean_hourly.into_data().to_vec().unwrap();
        let chunk_mean_q = q
            .clone()
            .slice([day_start..day_start + chunk_days, 0..n])
            .mean_dim(0)
            .reshape([n]);
        let want: Vec<f32> = chunk_mean_q.into_data().to_vec().unwrap();
        for i in 0..n {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "chunk [{day_start}, {}): reach {i}: {} vs {}",
                day_start + chunk_days,
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn chunk_days_two_conserves_chunk_mass_exactly() {
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(7).with_chunk_days(2);
        let head = cfg.init::<Bp>(&device);
        let q = daily::<Bp>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5], [2.0, 9.0]]); // D=4
        let p = precip::<Bp>(96, |h, r| if h % 24 == 13 { 5.0 + r as f32 } else { 0.1 });
        let hourly = head.forward(q.clone(), p, 96); // 4 days = 2 chunks of 2
        assert_eq!(hourly.dims(), [96, 2]);
        assert_chunk_mass_conserved(&hourly, &q, 0, 2);
        assert_chunk_mass_conserved(&hourly, &q, 2, 2);
    }

    #[test]
    fn chunk_days_relaxes_per_day_exactness() {
        // With chunk_days=2 and two days with very different daily_q, an
        // individual day's OWN 24h mean is free to deviate from its own
        // daily_q (only the 2-day chunk aggregate must match) -- verify
        // this is genuinely happening, not accidentally still per-day-exact.
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(3).with_chunk_days(2);
        let mut head = cfg.init::<Bp>(&device);
        // Bias the output layer heavily toward early hours so mass shifts
        // away from an even per-day split.
        let biased = Array2::<f32>::from_shape_fn((cfg.hidden_size, 48), |(_, j)| if j < 10 { 3.0 } else { -1.0 });
        head.output.weight = crate::nn::init::to_param_weight::<Bp>(biased, &device);

        let q = daily::<Bp>(&[[1.0, 1.0], [50.0, 50.0]]); // D=2, one chunk of 2, very different days
        let p = precip::<Bp>(48, |_, _| 0.1);
        let hourly = head.forward(q.clone(), p, 48);
        let v: Vec<f32> = hourly.clone().into_data().to_vec().unwrap();

        // Reach 0's values are at even indices (row-major (h, reach)).
        let day0_mean: f32 = (0..24).map(|h| v[h * 2]).sum::<f32>() / 24.0;
        let day1_mean: f32 = (24..48).map(|h| v[h * 2]).sum::<f32>() / 24.0;
        assert!(
            (day0_mean - 1.0).abs() > 0.5 || (day1_mean - 50.0).abs() > 0.5,
            "per-day means [{day0_mean}, {day1_mean}] suspiciously close to [1, 50] -- relaxation may not be active"
        );
        // But the 2-day chunk aggregate must still be exact.
        assert_chunk_mass_conserved(&hourly, &q, 0, 2);
    }

    #[test]
    fn chunk_days_remainder_chunk_is_handled_correctly() {
        // d_use=5, chunk_days=3 -> chunks of [3, 2] (a genuine remainder,
        // internally padded then trimmed back). Both chunks' own aggregate
        // mass must still conserve, and the output must have exactly 5*24
        // hours (no leftover padding leaking into the result).
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(11).with_chunk_days(3);
        let head = cfg.init::<Bp>(&device);
        let q = daily::<Bp>(&[[4.0, 4.0], [9.0, 9.0], [2.0, 2.0], [30.0, 30.0], [15.0, 15.0]]); // D=5
        let p = precip::<Bp>(120, |h, _| if h % 24 == 6 { 6.0 } else { 0.1 });
        let hourly = head.forward(q.clone(), p, 120); // 5 days -> chunks [3, 2]
        assert_eq!(hourly.dims(), [120, 2], "remainder chunk must not leak padded hours into the output");
        assert_chunk_mass_conserved(&hourly, &q, 0, 3);
        assert_chunk_mass_conserved(&hourly, &q, 3, 2);
    }

    #[test]
    fn chunk_days_one_matches_per_day_path_exactly() {
        // chunk_days=1 must dispatch to forward_per_day and reproduce the
        // ORIGINAL per-day-exact invariant -- the byte-identical-when-absent
        // guarantee for every checkpoint trained before chunk_days existed.
        let device = Default::default();
        let cfg = DisaggHeadConfig::new(7).with_chunk_days(1);
        let head = cfg.init::<Bp>(&device);
        let q = daily::<Bp>(&[[5.0, 1.0], [20.0, 3.0], [8.0, 0.5], [2.0, 9.0]]);
        let p = precip::<Bp>(72, |h, r| if h % 24 == 13 { 5.0 + r as f32 } else { 0.1 });
        let hourly = head.forward(q.clone(), p, 72);
        assert_mass_conserved(&hourly, &q, 3);
    }
}
