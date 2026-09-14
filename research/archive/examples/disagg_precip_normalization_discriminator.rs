//! Cheap discriminator: is the precip-normalization SCHEME the bottleneck for
//! the disagg head learning within-day TIMING, or is 5 epochs / hidden_size=16
//! simply not enough capacity/budget regardless of normalization?
//!
//! Frozen-routing, head-only synthetic overfit test: generate reach-days with
//! a KNOWN target shape (a Gaussian bump in the softmax logits, centered at
//! that day's storm hour + a fixed lag), train `DisaggHead` DIRECTLY against
//! that target (MSE on hourly values — no routing, no daily-downsampled loss,
//! no real data) under TWO precip-feature encodings:
//!   - "basin_norm": log1p(precip / FIXED per-reach constant)  (current prod)
//!   - "z_score":    z-score of log1p(precip) over the 24h window (original)
//! at two step budgets: SHORT (180 steps, matches the real run's 5 epochs x
//! 36 minibatches) and LONG (20x). If z-score succeeds short where
//! basin_norm fails, normalization is the bottleneck; if both fail short and
//! succeed long, budget/capacity is.
//!
//!   cargo run --release --example disagg_precip_normalization_discriminator

use burn::backend::{Autodiff, NdArray};
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use ddrs::nn::{DisaggHead, DisaggHeadConfig};
use ddrs::training::build_adam;

type I = NdArray<f32>;
type AutoI = Autodiff<I>;

const N_SAMPLES: usize = 256;
const BATCH_SIZE: usize = 64;
const SEED: u64 = 42;
const LAG_HOURS: f32 = 1.0; // true response peaks 1h after the storm hour
const SIGMA: f32 = 1.5; // spread of the true response bump, hours

/// Deterministic xorshift-style PRNG so this example has no external `rand`
/// dependency and is reproducible run-to-run.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(2685821657736338717).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
    fn range_usize(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_f32() * (hi - lo) as f32) as usize
    }
}

/// Circular distance between two hour-of-day values in [0, 24).
fn circ_dist(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(24.0);
    d.min(24.0 - d)
}

struct Sample {
    daily_q: f32,
    storm_hour: usize,
    meanp_hourly: f32, // this reach's fixed climatological divisor, basin_norm variant
    raw_precip: [f32; 24],
    target_hourly: [f32; 24], // sums to daily_q * 24 by construction
}

fn make_samples(rng: &mut Rng) -> Vec<Sample> {
    (0..N_SAMPLES)
        .map(|_| {
            let daily_q = 10f32.powf(rng.range_f32(-0.3, 1.7)); // ~0.5 to ~50
            let storm_hour = rng.range_usize(0, 24);
            let intensity = rng.range_f32(1.0, 8.0); // mm/hr peak, realistic range
            let meanp_hourly = rng.range_f32(0.05, 0.6); // matches real CONUS meanP/8760 range
            let background = 0.05f32;

            let mut raw_precip = [background; 24];
            raw_precip[storm_hour] = intensity;

            // True target: Gaussian bump in logit-space centered at
            // storm_hour + LAG, softmax-normalized, scaled to daily_q * 24.
            let mut logits = [0f32; 24];
            for (h, logit) in logits.iter_mut().enumerate() {
                let d = circ_dist(h as f32, (storm_hour as f32 + LAG_HOURS).rem_euclid(24.0));
                *logit = -(d * d) / (2.0 * SIGMA * SIGMA);
            }
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
            let sum_exp: f32 = exp.iter().sum();
            let mut target_hourly = [0f32; 24];
            for h in 0..24 {
                target_hourly[h] = daily_q * 24.0 * exp[h] / sum_exp;
            }

            Sample { daily_q, storm_hour, meanp_hourly, raw_precip, target_hourly }
        })
        .collect()
}

/// Basin-norm variant: log1p(precip / FIXED per-reach meanp_hourly).
fn feature_basin_norm(s: &Sample) -> [f32; 24] {
    let mut out = [0f32; 24];
    for h in 0..24 {
        out[h] = (s.raw_precip[h] / s.meanp_hourly).ln_1p();
    }
    out
}

/// Z-score variant: log1p(precip), then per-window (this sample's 24h) z-score.
fn feature_z_score(s: &Sample) -> [f32; 24] {
    let mut logp = [0f32; 24];
    for h in 0..24 {
        logp[h] = s.raw_precip[h].ln_1p();
    }
    let mean: f32 = logp.iter().sum::<f32>() / 24.0;
    let var: f32 = logp.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 24.0;
    let std = var.sqrt();
    let mut out = [0f32; 24];
    if std < 1e-6 {
        return out; // all-zero, matches normalize_temp's no-coverage fallback
    }
    for h in 0..24 {
        out[h] = (logp[h] - mean) / std;
    }
    out
}

fn train_and_eval(
    samples: &[Sample],
    feature_fn: impl Fn(&Sample) -> [f32; 24],
    n_steps: usize,
    label: &str,
) {
    let device = Default::default();
    <AutoI as Backend>::seed(&device, SEED);
    let mut head: DisaggHead<AutoI> = DisaggHeadConfig::new(SEED).with_hidden_size(16).init(&device);
    let mut optimizer = build_adam::<DisaggHead<AutoI>, AutoI>();

    let mut rng = Rng::new(SEED ^ 0x5EED);
    let n = samples.len();

    for step in 0..n_steps {
        // Random mini-batch with replacement.
        let idx: Vec<usize> = (0..BATCH_SIZE).map(|_| rng.range_usize(0, n)).collect();

        let mut q_buf = Vec::with_capacity(BATCH_SIZE);
        let mut precip_buf = Vec::with_capacity(BATCH_SIZE * 24);
        let mut target_buf = Vec::with_capacity(BATCH_SIZE * 24);
        for &i in &idx {
            let s = &samples[i];
            q_buf.push(s.daily_q);
            let feat = feature_fn(s);
            for h in 0..24 {
                precip_buf.push(feat[h]);
            }
            for h in 0..24 {
                target_buf.push(s.target_hourly[h]);
            }
        }

        let daily_q = Tensor::<AutoI, 2>::from_data(TensorData::new(q_buf, [1, BATCH_SIZE]), &device);
        // precip_buf is (BATCH, 24) row-major -> transpose to (24, BATCH) for forward()'s (n_hourly, N) contract.
        let precip = Tensor::<AutoI, 2>::from_data(TensorData::new(precip_buf, [BATCH_SIZE, 24]), &device)
            .transpose();
        let target = Tensor::<AutoI, 2>::from_data(TensorData::new(target_buf, [BATCH_SIZE, 24]), &device)
            .transpose(); // (24, BATCH)

        let pred = head.forward(daily_q, precip, 24); // (24, BATCH)
        let diff = pred - target;
        let loss = diff.clone().powf_scalar(2.0).mean();

        let grads = GradientsParams::from_grads(loss.backward(), &head);
        head = optimizer.step(1e-3, head, grads);

        if step == n_steps - 1 {
            let final_loss: f32 = loss.into_scalar();
            eprintln!("  [{label}] step {step}: mse={final_loss:.5}");
        }
    }

    // Evaluate peak-tracking accuracy on the SAME pool (overfit test — the
    // question is architectural/optimization capacity, not generalization).
    let mut within_1h = 0usize;
    let mut within_2h = 0usize;
    let mut circ_errs: Vec<f32> = Vec::with_capacity(n);
    for s in samples {
        let feat = feature_fn(s);
        let daily_q = Tensor::<AutoI, 2>::from_data(TensorData::new(vec![s.daily_q], [1, 1]), &device);
        let precip = Tensor::<AutoI, 2>::from_data(TensorData::new(feat.to_vec(), [24, 1]), &device);
        let pred = head.forward(daily_q, precip, 24);
        let pred_vec: Vec<f32> = pred.into_data().to_vec().unwrap();
        let pred_hour = pred_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(h, _)| h as f32)
            .unwrap();
        let true_peak = (s.storm_hour as f32 + LAG_HOURS).rem_euclid(24.0);
        let err = circ_dist(pred_hour, true_peak);
        circ_errs.push(err);
        if err <= 1.0 {
            within_1h += 1;
        }
        if err <= 2.0 {
            within_2h += 1;
        }
    }
    let mean_err: f32 = circ_errs.iter().sum::<f32>() / n as f32;
    println!(
        "{label:28}  mean_circ_err={mean_err:5.2}h  within_1h={:5.1}%  within_2h={:5.1}%",
        100.0 * within_1h as f32 / n as f32,
        100.0 * within_2h as f32 / n as f32,
    );
}

fn main() {
    let mut rng = Rng::new(SEED);
    let samples = make_samples(&mut rng);

    // Matches the real CONUS run: batch_size=64, ~36 minibatches/epoch, 5
    // epochs = 180 steps. LONG = 20x that.
    const SHORT: usize = 180;
    const LONG: usize = 180 * 20;

    println!("Disagg-head precip-normalization discriminator ({N_SAMPLES} synthetic reach-days, batch={BATCH_SIZE})");
    println!("A random/untrained baseline would show mean_circ_err ~6h, within_1h ~8%, within_2h ~17% (uniform over 24h).\n");

    train_and_eval(&samples, feature_basin_norm, SHORT, "basin_norm (SHORT=180)");
    train_and_eval(&samples, feature_z_score, SHORT, "z_score (SHORT=180)");
    train_and_eval(&samples, feature_basin_norm, LONG, "basin_norm (LONG=3600)");
    train_and_eval(&samples, feature_z_score, LONG, "z_score (LONG=3600)");
}
