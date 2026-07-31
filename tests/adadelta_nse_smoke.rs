//! Pre-flight smoke gate for the AdaDelta + batch-NSE recipe.
//!
//! These exist because the 2026-07-30 30-update run burned ~19 h of CPU to
//! discover the KAN had barely moved (n spanning 7.5% of its range, flat vs
//! drainage area, median NSE 0.621 vs a 0.674 no-routing baseline). The root
//! cause was arithmetic, not hydrology: Adam's per-step parameter movement is
//! bounded by `lr`, so 30 updates at 1e-3/5e-4 can move a weight at most
//! ~0.017 total.
//!
//! Each test below asserts a property that, had it been checked first, would
//! have predicted that failure — run them before committing GPU/CPU hours:
//!   1. the config key actually selects the optimizer (not a silent no-op),
//!   2. AdaDelta moves parameters ORDERS more than Adam in an equal, small
//!      number of updates,
//!   3. the batch-NSE loss is wired end-to-end through `batch_loss` and its
//!      gradient survives,
//!   4. batch-NSE stays exact under gradient accumulation (unequal micros),
//!   5. an AdaDelta+NSE optimizer checkpoint round-trips, and a cross-kind
//!      resume is refused rather than silently reinterpreted.

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};

use ddrs::config::{LossConfig, LossKind, OptimizerKind};
use ddrs::nn::{KanHead, KanHeadConfig};
use ddrs::training::{
    batch_loss, build_head_optimizer, loss_denominator, nse_batch_loss, scale_grads, HeadOptimizer,
};

type B = Autodiff<NdArray<f32>>;

const G: usize = 6;
const F: usize = 4;
const T: usize = 5;

fn head() -> KanHead<B> {
    KanHeadConfig::new(
        (0..F).map(|i| format!("attr_{i}")).collect(),
        vec!["n".into(), "q_spatial".into()],
        42,
    )
    .with_hidden_size(8)
    .with_num_hidden_layers(2)
    .init::<B>(&Default::default())
}

fn attributes() -> Tensor<B, 2> {
    let vals: Vec<f32> = (0..G * F).map(|i| ((i * 37 + 11) % 97) as f32 / 97.0).collect();
    Tensor::<B, 1>::from_data(TensorData::new(vals, [G * F]), &Default::default()).reshape([G, F])
}

fn observations() -> Tensor<B, 2> {
    let vals: Vec<f32> = (0..G * T).map(|i| 1.0 + ((i * 13 + 5) % 41) as f32 / 10.0).collect();
    Tensor::<B, 1>::from_data(TensorData::new(vals, [G * T]), &Default::default()).reshape([G, T])
}

fn sigma(vals: Vec<f32>) -> Tensor<B, 1> {
    let n = vals.len();
    Tensor::from_data(TensorData::new(vals, [n]), &Default::default())
}

/// Differentiable (G, T) predictions from the head, row-independent so a
/// pooled forward equals the concatenation of micro forwards.
fn predictions(head: &KanHead<B>, x: Tensor<B, 2>) -> Tensor<B, 2> {
    let g = x.dims()[0];
    let out = head.forward(x);
    let n = out.get("n").expect("n").clone().reshape([g, 1]);
    let q = out.get("q_spatial").expect("q_spatial").clone().reshape([g, 1]);
    let dev = Default::default();
    let b1: Vec<f32> = (0..T).map(|t| 1.0 + t as f32 * 0.5).collect();
    let b2: Vec<f32> = (0..T).map(|t| 2.0 - t as f32 * 0.3).collect();
    let b1 = Tensor::<B, 1>::from_data(TensorData::new(b1, [T]), &dev).reshape([1, T]);
    let b2 = Tensor::<B, 1>::from_data(TensorData::new(b2, [T]), &dev).reshape([1, T]);
    n.matmul(b1) + q.matmul(b2)
}

/// Mean |Δ| of the head's emitted `n` field after `steps` updates — the
/// quantity that actually matters for the science (parameter movement), not
/// the loss value.
fn n_movement_after(kind: OptimizerKind, lr: f64, steps: usize) -> f32 {
    let mut h = head();
    let x = attributes();
    let o = observations();
    let sig = sigma(vec![1.0; G]);
    let cfg = LossConfig { kind: LossKind::NseBatch, ..LossConfig::default() };

    let n_of = |h: &KanHead<B>| -> Vec<f32> {
        h.forward(x.clone()).get("n").unwrap().clone().into_data().to_vec().unwrap()
    };
    let before = n_of(&h);
    let mut optim: HeadOptimizer<KanHead<B>, B> = build_head_optimizer(kind);
    for _ in 0..steps {
        let loss = batch_loss(predictions(&h, x.clone()), o.clone(), &cfg, Some(sig.clone()));
        let grads = GradientsParams::from_grads(loss.backward(), &h);
        h = optim.step(lr, h, grads);
    }
    let after = n_of(&h);
    before
        .iter()
        .zip(&after)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / before.len() as f32
}

#[test]
fn config_kind_actually_selects_the_optimizer() {
    // Guards against the failure mode that cost the last run: a config key
    // that parses but doesn't change what runs.
    let adam: HeadOptimizer<KanHead<B>, B> = build_head_optimizer(OptimizerKind::Adam);
    let ada: HeadOptimizer<KanHead<B>, B> = build_head_optimizer(OptimizerKind::Adadelta);
    assert!(matches!(adam, HeadOptimizer::Adam(_)), "Adam kind built wrong variant");
    assert!(
        matches!(ada, HeadOptimizer::AdaDelta(_)),
        "Adadelta kind built wrong variant"
    );
}

#[test]
fn adadelta_moves_parameters_far_more_than_adam_in_few_updates() {
    // THE pre-flight check. At the previous run's settings (30 updates,
    // lr 5e-4) Adam barely moves the emitted field; AdaDelta must move it by
    // at least an order of magnitude more, or the next run repeats history.
    let steps = 30;
    let adam = n_movement_after(OptimizerKind::Adam, 5e-4, steps);
    let ada = n_movement_after(OptimizerKind::Adadelta, 1.0, steps);
    eprintln!("mean |Δn| after {steps} updates: adam={adam:.6e}  adadelta={ada:.6e}  ratio={:.1}x", ada / adam.max(1e-12));
    assert!(adam > 0.0 && ada > 0.0, "one optimizer did not move at all");
    assert!(
        ada > 10.0 * adam,
        "AdaDelta moved {ada:.3e} vs Adam {adam:.3e} — less than 10x, the \
         low-update-count problem is NOT solved"
    );
}

#[test]
fn batch_nse_loss_is_wired_through_batch_loss_and_differentiable() {
    let h = head();
    let x = attributes();
    let o = observations();
    let cfg = LossConfig { kind: LossKind::NseBatch, ..LossConfig::default() };
    let p = predictions(&h, x);
    let loss = batch_loss(p, o.clone(), &cfg, Some(sigma(vec![1.0; G])));
    let v: f32 = loss.clone().into_scalar();
    assert!(v.is_finite() && v > 0.0, "loss not finite/positive: {v}");
    let grads = GradientsParams::from_grads(loss.backward(), &h);
    assert!(grads.len() > 0, "batch-NSE produced no gradients");
}

#[test]
fn batch_nse_matches_the_dhbv_formula_elementwise() {
    // loss = mean over (day, gauge) of (sim - obs)^2 / (sigma_gauge + eps)^2
    let dev = Default::default();
    let p = Tensor::<B, 1>::from_data(TensorData::new(vec![3.0_f32, 3.0, 6.0, 6.0], [4]), &dev)
        .reshape([2, 2]);
    let o = Tensor::<B, 1>::from_data(TensorData::new(vec![1.0_f32, 1.0, 2.0, 2.0], [4]), &dev)
        .reshape([2, 2]);
    let s = sigma(vec![1.0, 3.0]);
    let got: f32 = nse_batch_loss(p, o, s, 0.1).into_scalar();
    // gauge 0: 2 residuals of 2 -> 4/(1.1^2)=3.305785 each
    // gauge 1: 2 residuals of 4 -> 16/(3.1^2)=1.664932 each
    let want = (2.0 * 4.0 / 1.1_f32.powi(2) + 2.0 * 16.0 / 3.1_f32.powi(2)) / 4.0;
    assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
}

#[test]
fn batch_nse_stays_exact_under_gradient_accumulation() {
    // The new loss must keep the accumulation identity the driver relies on
    // (weights = valid-element counts), with UNEQUAL micro-batches.
    let h = head();
    let x = attributes();
    let o = observations();
    let sig_all = sigma(vec![1.0, 2.0, 0.5, 4.0, 1.5, 3.0]);
    let cfg = LossConfig { kind: LossKind::NseBatch, ..LossConfig::default() };

    let pooled = batch_loss(
        predictions(&h, x.clone()),
        o.clone(),
        &cfg,
        Some(sig_all.clone()),
    );
    let pooled_v: f32 = pooled.clone().into_scalar();
    let ref_grads = GradientsParams::from_grads(pooled.backward(), &h);

    let mut acc = GradientsAccumulator::<KanHead<B>>::new();
    let mut total_n = 0usize;
    let mut lw = 0.0f64;
    for (lo, hi) in [(0usize, 4usize), (4, G)] {
        let gm = hi - lo;
        let p = predictions(&h, x.clone().slice([lo..hi, 0..F]));
        let sg = sig_all.clone().slice([lo..hi]);
        let l = batch_loss(p, o.clone().slice([lo..hi, 0..T]), &cfg, Some(sg));
        let lv: f32 = l.clone().into_scalar();
        let n_i = loss_denominator(&cfg, gm, T);
        acc.accumulate(&h, GradientsParams::from_grads(l.mul_scalar(n_i as f32).backward(), &h));
        total_n += n_i;
        lw += lv as f64 * n_i as f64;
    }
    let acc_grads = scale_grads(acc.grads(), &h, 1.0 / total_n as f32);

    let recombined = (lw / total_n as f64) as f32;
    assert!(
        (recombined - pooled_v).abs() / pooled_v.abs() < 1e-6,
        "recombined {recombined} != pooled {pooled_v}"
    );
    // Gradient equality, param by param.
    use burn::module::{ModuleVisitor, Param};
    struct Cmp<'a> {
        a: &'a GradientsParams,
        b: &'a GradientsParams,
        worst: f32,
    }
    impl ModuleVisitor<B> for Cmp<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            use burn::tensor::backend::AutodiffBackend;
            let (Some(x), Some(y)) = (
                self.a.get::<<B as AutodiffBackend>::InnerBackend, D>(param.id),
                self.b.get::<<B as AutodiffBackend>::InnerBackend, D>(param.id),
            ) else {
                return;
            };
            let xv: Vec<f32> = x.into_data().to_vec().unwrap();
            let yv: Vec<f32> = y.into_data().to_vec().unwrap();
            let scale = xv.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-30);
            for (a, b) in xv.iter().zip(&yv) {
                self.worst = self.worst.max((a - b).abs() / scale);
            }
        }
    }
    let mut cmp = Cmp { a: &ref_grads, b: &acc_grads, worst: 0.0 };
    h.visit(&mut cmp);
    let worst = cmp.worst;
    eprintln!("batch-NSE accumulation worst max-rel grad diff: {worst:.3e}");
    assert!(worst <= 1e-4, "accumulated grad differs by {worst} (> 1e-4)");
}

#[test]
fn adadelta_optimizer_state_round_trips_and_refuses_cross_kind_resume() {
    use burn::record::{CompactRecorder, Recorder};
    let dir = std::env::temp_dir().join("ddrs_adadelta_smoke");
    let _ = std::fs::create_dir_all(&dir);
    let base = dir.join("optim");

    let mut h = head();
    let x = attributes();
    let o = observations();
    let cfg = LossConfig { kind: LossKind::NseBatch, ..LossConfig::default() };
    let sig = sigma(vec![1.0; G]);
    let mut optim: HeadOptimizer<KanHead<B>, B> = build_head_optimizer(OptimizerKind::Adadelta);
    for _ in 0..3 {
        let loss = batch_loss(predictions(&h, x.clone()), o.clone(), &cfg, Some(sig.clone()));
        let grads = GradientsParams::from_grads(loss.backward(), &h);
        h = optim.step(1.0, h, grads);
    }
    CompactRecorder::new()
        .record(optim.to_record(), base.clone())
        .expect("record adadelta state");

    // Same kind: loads.
    let fresh: HeadOptimizer<KanHead<B>, B> = build_head_optimizer(OptimizerKind::Adadelta);
    let rec = CompactRecorder::new()
        .load(base.clone(), &Default::default())
        .expect("load adadelta state");
    let _restored = fresh.load_record(rec);

    // Cross-kind: must panic rather than reinterpret moment tensors.
    let adam: HeadOptimizer<KanHead<B>, B> = build_head_optimizer(OptimizerKind::Adam);
    let rec2 = CompactRecorder::new()
        .load(base, &Default::default())
        .expect("reload adadelta state");
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = adam.load_record(rec2);
    }));
    assert!(res.is_err(), "cross-kind optimizer resume was silently accepted");
}
