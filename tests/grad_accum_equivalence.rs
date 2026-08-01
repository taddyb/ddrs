//! Gradient-equivalence gate for optimizer micro-batching.
//!
//! Port of water_loss's `diag_gradaccum_equivalence.py` (see
//! `/tmp/experiment-handoff-gradaccum-for-ddrs.md`): with stochastic layers
//! absent and one fixed draw, the accumulated gradient must equal the true
//! large-batch gradient. Composes the exact production pieces the driver
//! uses — `batch_loss`, `loss_denominator`, loss×n_i before `backward()`,
//! `GradientsAccumulator`, `scale_grads(1/Σn)` — through a real `KanHead`
//! on the CPU backend, with deliberately UNEQUAL micro-batches (the NaN
//! filter makes that the production common case).
//!
//! Pass thresholds (from the handoff's Controls table, observed fp32 noise
//! 5.96e-08 loss / cosine 1.000000 / max-rel 6.4e-05):
//!   - recombined loss vs pooled loss: |diff| ≤ 1e-6 (relative)
//!   - per-tensor gradient cosine ≥ 0.999999
//!   - per-tensor max relative difference ≤ 1e-4

use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::{GradientsAccumulator, GradientsParams};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};

use ddrs::config::LossConfig;
use ddrs::nn::{KanHead, KanHeadConfig};
use ddrs::training::{batch_loss, loss_denominator, scale_grads};

type B = Autodiff<NdArray<f32>>;

const G: usize = 6; // total gauges
const F: usize = 4; // attributes
const T: usize = 5; // post-warmup days
const SPLIT: usize = 4; // micro A = gauges 0..4, micro B = 4..6 (unequal)

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

/// Deterministic attributes: G rows of F features in [0, 1).
fn attributes() -> Tensor<B, 2> {
    let vals: Vec<f32> = (0..G * F).map(|i| ((i * 37 + 11) % 97) as f32 / 97.0).collect();
    Tensor::<B, 1>::from_data(TensorData::new(vals, [G * F]), &Default::default())
        .reshape([G, F])
}

/// Deterministic "observations": G × T strictly positive values.
fn observations() -> Tensor<B, 2> {
    let vals: Vec<f32> = (0..G * T).map(|i| 1.0 + ((i * 13 + 5) % 41) as f32 / 10.0).collect();
    Tensor::<B, 1>::from_data(TensorData::new(vals, [G * T]), &Default::default())
        .reshape([G, T])
}

/// Differentiable (G_slice, T) "predictions" from the head's outputs for a
/// row slice of the attributes: pred(g, t) = n[g]·b1[t] + q[g]·b2[t].
/// Rows are independent, so a pooled forward equals the micro forwards —
/// the same property the real per-gauge routing loss has after collate.
fn predictions(head: &KanHead<B>, x: Tensor<B, 2>) -> Tensor<B, 2> {
    let g = x.dims()[0];
    let out = head.forward(x);
    let n = out.get("n").expect("head emits n").clone().reshape([g, 1]);
    let q = out
        .get("q_spatial")
        .expect("head emits q_spatial")
        .clone()
        .reshape([g, 1]);
    let b1: Vec<f32> = (0..T).map(|t| 1.0 + t as f32 * 0.5).collect();
    let b2: Vec<f32> = (0..T).map(|t| 2.0 - t as f32 * 0.3).collect();
    let dev = Default::default();
    let b1 = Tensor::<B, 1>::from_data(TensorData::new(b1, [T]), &dev).reshape([1, T]);
    let b2 = Tensor::<B, 1>::from_data(TensorData::new(b2, [T]), &dev).reshape([1, T]);
    n.matmul(b1) + q.matmul(b2)
}

/// Collect every param's gradient as (flattened values), keyed by ParamId.
struct GradCollector<'a> {
    grads: &'a GradientsParams,
    out: Vec<(String, Vec<f32>)>,
}

impl<'a> ModuleVisitor<B> for GradCollector<'a> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(g) = self
            .grads
            .get::<<B as AutodiffBackend>::InnerBackend, D>(param.id)
        {
            self.out
                .push((format!("{:?}", param.id), g.into_data().to_vec().unwrap()));
        }
    }
}

fn collect(head: &KanHead<B>, grads: &GradientsParams) -> Vec<(String, Vec<f32>)> {
    let mut c = GradCollector { grads, out: vec![] };
    head.visit(&mut c);
    let mut out = c.out;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Run the equivalence check for one loss config. Returns
/// (loss_rel_diff, worst_cosine, worst_max_rel).
fn check_equivalence(loss_cfg: &LossConfig) -> (f32, f32, f32) {
    let head = head();
    let x = attributes();
    let o = observations();

    // ── reference: one pooled batch ────────────────────────────────────────
    let p_all = predictions(&head, x.clone());
    let pooled_loss = batch_loss(p_all, o.clone(), loss_cfg, None);
    let pooled_f32: f32 = pooled_loss.clone().into_scalar();
    let grads_ref = GradientsParams::from_grads(pooled_loss.backward(), &head);
    let ref_tensors = collect(&head, &grads_ref);
    assert!(!ref_tensors.is_empty(), "reference produced no gradients");

    // ── accumulated: two UNEQUAL micro-batches, production combination ─────
    let mut accumulator = GradientsAccumulator::<KanHead<B>>::new();
    let mut total_n = 0usize;
    let mut loss_weighted = 0.0f64;
    for (lo, hi) in [(0, SPLIT), (SPLIT, G)] {
        let g_micro = hi - lo;
        let x_micro = x.clone().slice([lo..hi, 0..F]);
        let o_micro = o.clone().slice([lo..hi, 0..T]);
        let p_micro = predictions(&head, x_micro);
        let micro_loss = batch_loss(p_micro, o_micro, loss_cfg, None);
        let micro_f32: f32 = micro_loss.clone().into_scalar();
        let n_i = loss_denominator(loss_cfg, g_micro, T);
        let grads =
            GradientsParams::from_grads(micro_loss.mul_scalar(n_i as f32).backward(), &head);
        accumulator.accumulate(&head, grads);
        total_n += n_i;
        loss_weighted += micro_f32 as f64 * n_i as f64;
    }
    let grads_acc = scale_grads(accumulator.grads(), &head, 1.0 / total_n as f32);
    let acc_tensors = collect(&head, &grads_acc);

    // ── compare ────────────────────────────────────────────────────────────
    let recombined = (loss_weighted / total_n as f64) as f32;
    let loss_rel_diff = (recombined - pooled_f32).abs() / pooled_f32.abs().max(1e-12);

    assert_eq!(
        ref_tensors.len(),
        acc_tensors.len(),
        "accumulated grads cover {} tensors, reference {}",
        acc_tensors.len(),
        ref_tensors.len()
    );
    let mut worst_cosine = 1.0f32;
    let mut worst_max_rel = 0.0f32;
    for ((id_r, r), (id_a, a)) in ref_tensors.iter().zip(acc_tensors.iter()) {
        assert_eq!(id_r, id_a, "param id order mismatch");
        assert_eq!(r.len(), a.len());
        let dot: f32 = r.iter().zip(a).map(|(x, y)| x * y).sum();
        let nr: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        if nr == 0.0 && na == 0.0 {
            continue; // both zero — e.g. a frozen/unused param
        }
        let cosine = dot / (nr * na).max(1e-30);
        worst_cosine = worst_cosine.min(cosine);
        let scale = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-30);
        for (x, y) in r.iter().zip(a) {
            worst_max_rel = worst_max_rel.max((x - y).abs() / scale);
        }
    }
    (loss_rel_diff, worst_cosine, worst_max_rel)
}

#[test]
fn accumulated_gradient_equals_pooled_l1() {
    let cfg = LossConfig::default(); // L1: mean over elements
    let (loss_diff, cosine, max_rel) = check_equivalence(&cfg);
    eprintln!("L1: loss_rel_diff={loss_diff:.3e} cosine={cosine:.8} max_rel={max_rel:.3e}");
    assert!(loss_diff <= 1e-6, "loss rel diff {loss_diff} > 1e-6");
    assert!(cosine >= 0.999_999, "worst cosine {cosine} < 0.999999");
    assert!(max_rel <= 1e-4, "worst max-rel {max_rel} > 1e-4");
}

#[test]
fn accumulated_gradient_equals_pooled_nnse_kge() {
    // Per-gauge objective: denominator is GAUGES, not elements. This is the
    // case a naive 1/N average gets wrong with unequal micro-batches.
    let mut cfg = LossConfig::default();
    cfg.kind = ddrs::config::LossKind::NnseKge;
    let (loss_diff, cosine, max_rel) = check_equivalence(&cfg);
    eprintln!("NnseKge: loss_rel_diff={loss_diff:.3e} cosine={cosine:.8} max_rel={max_rel:.3e}");
    assert!(loss_diff <= 1e-6, "loss rel diff {loss_diff} > 1e-6");
    assert!(cosine >= 0.999_999, "worst cosine {cosine} < 0.999999");
    assert!(max_rel <= 1e-4, "worst max-rel {max_rel} > 1e-4");
}

#[test]
fn naive_equal_weighting_is_not_equivalent() {
    // Negative control: with unequal micro-batches, the naive mean-of-means
    // (weight 1/2 each) must NOT reproduce the pooled loss — proving the
    // test can actually detect wrong weighting.
    let cfg = LossConfig::default();
    let head = head();
    let x = attributes();
    let o = observations();
    let pooled: f32 = batch_loss(predictions(&head, x.clone()), o.clone(), &cfg, None).into_scalar();
    let mut naive = 0.0f32;
    for (lo, hi) in [(0, SPLIT), (SPLIT, G)] {
        let p = predictions(&head, x.clone().slice([lo..hi, 0..F]));
        let l: f32 = batch_loss(p, o.clone().slice([lo..hi, 0..T]), &cfg, None).into_scalar();
        naive += 0.5 * l;
    }
    let rel = (naive - pooled).abs() / pooled.abs().max(1e-12);
    assert!(
        rel > 1e-4,
        "naive weighting accidentally matched pooled loss (rel {rel}) — \
         fixture no longer discriminates; make the micro losses more unequal"
    );
}
