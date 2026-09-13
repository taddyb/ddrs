//! Parameter-group trunks and KAN boundary layers (`kan_head.parameter_groups`,
//! `input_layer_kan`, `output_layer_kan`).
//!
//! These knobs exist to test the §31 finding in
//! `docs/2026-09-08-landscape-hypothesis-tests-findings.md`: with a
//! `Linear(H, P)` read-out every output is an affine functional of one shared
//! latent, so when that latent's informative part is effectively one direction
//! any two parameters are forced into an exact affine relationship
//! (`logit(q) = 3.979 · logit(n) + 2.752`, R² = 0.987 over 346,321 reaches).
//!
//! The load-bearing property is that **the defaults change nothing**: every
//! config written before these fields existed must build the identical head
//! and load its existing checkpoints.

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::Tensor;
use ddrs::nn::kan_head::KanHeadConfig;

type B = NdArray<f32>;

const F: usize = 4;
const N: usize = 32;

fn attrs(device: &NdArrayDevice) -> Tensor<B, 2> {
    // Deterministic pseudo-attributes on roughly a z-score scale.
    let data: Vec<f32> = (0..N * F)
        .map(|i| ((i as f32 * 0.37).sin() * 1.7) + ((i % 7) as f32 - 3.0) * 0.2)
        .collect();
    Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(data, [N, F]), device)
}

fn base(params: &[&str]) -> KanHeadConfig {
    KanHeadConfig::new(
        (0..F).map(|i| format!("attr{i}")).collect::<Vec<_>>(),
        params.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        42,
    )
    .with_hidden_size(8)
    .with_num_hidden_layers(2)
    .with_grid(5)
    .with_k(3)
}

fn field(head: &ddrs::nn::kan_head::KanHead<B>, x: Tensor<B, 2>, key: &str) -> Vec<f32> {
    head.forward(x)
        .remove(key)
        .expect("key present")
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

/// A single group holding every parameter must reproduce the shared-trunk head
/// exactly — this is what makes `parameter_groups` a safe additive field.
#[test]
fn one_group_is_bit_identical_to_no_groups() {
    let device = NdArrayDevice::default();
    let x = attrs(&device);

    let shared = base(&["n", "q_spatial"]).init::<B>(&device);
    let grouped = base(&["n", "q_spatial"])
        .with_parameter_groups(vec![vec!["n".into(), "q_spatial".into()]])
        .init::<B>(&device);

    for key in ["n", "q_spatial"] {
        assert_eq!(
            field(&shared, x.clone(), key),
            field(&grouped, x.clone(), key),
            "one-group head diverged from the shared head on `{key}`"
        );
    }
    assert!(
        grouped.extra.is_empty(),
        "a single group must not allocate an extra trunk"
    );
}

/// Splitting into two groups must leave group 0 untouched (same seed, same
/// trunk) while giving group 1 genuinely different weights — otherwise the
/// "separate KAN" arm would be a relabelling of the shared one.
#[test]
fn split_leaves_group_zero_intact_and_group_one_independent() {
    let device = NdArrayDevice::default();
    let x = attrs(&device);

    let shared = base(&["n", "q_spatial"]).init::<B>(&device);
    let split = base(&["n", "q_spatial"])
        .with_parameter_groups(vec![vec!["n".into()], vec!["q_spatial".into()]])
        .init::<B>(&device);

    assert_eq!(split.extra.len(), 1, "expected one extra trunk");

    // Group 0 is `n` alone, so its output Linear is [H, 1] rather than [H, 2]:
    // the trunk is the same but the read-out width differs, so we assert on the
    // trunk activations, which are what "same trunk" actually means.
    let h_shared = shared
        .trunk_activations(x.clone())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let h_split = split
        .trunk_activations(x.clone())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(h_shared, h_split, "group 0's trunk must be unchanged");

    // The second trunk must not be a copy of the first.
    let t1 = &split.extra[0];
    let w0 = split
        .input
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let w1 = t1.input.weight.val().into_data().to_vec::<f32>().unwrap();
    assert_ne!(w0, w1, "extra trunk must be independently seeded");

    // And the split head must actually produce both parameters.
    let out = split.forward(x);
    assert!(out.contains_key("n") && out.contains_key("q_spatial"));
}

/// Group order, not config order, sets the output column order. The HashMap
/// keys must still be right, which is what every caller depends on.
#[test]
fn group_order_reorders_columns_without_losing_keys() {
    let device = NdArrayDevice::default();
    let x = attrs(&device);

    let head = base(&["n", "p_spatial", "q_spatial"])
        .with_parameter_groups(vec![
            vec!["p_spatial".into(), "q_spatial".into()],
            vec!["n".into()],
        ])
        .init::<B>(&device);

    assert_eq!(
        head.learnable_parameters(),
        ["p_spatial", "q_spatial", "n"],
        "column order must follow the group concatenation"
    );
    let out = head.forward(x);
    assert_eq!(out.len(), 3);
    for key in ["n", "p_spatial", "q_spatial"] {
        assert_eq!(out[key].dims(), [N], "`{key}` has the wrong shape");
    }
}

/// What a KAN read-out does and does not fix.
///
/// With `Linear(H, P)` every output is `w_j · h + b_j`. Two outputs are
/// therefore *exactly* affinely related whenever `h` is effectively rank 1
/// across reaches — which is the §31 situation (R² = 0.987 between
/// `logit(n)` and `logit(q)` over 346,321 CONUS reaches, gain 3.979).
/// `hidden_size = 1` reproduces that regime exactly and in miniature.
///
/// A `KanLayer(H, P)` read-out gives every output its own spline coefficients
/// on every edge, so the two columns stop being affine. It does **not** make
/// them independent: two different functions of the same scalar latent are
/// still in lockstep, which is why this test also pins the rank correlation
/// staying at 1. Breaking the functional dependence needs a trunk that carries
/// more than one direction, not a different read-out.
#[test]
fn kan_readout_breaks_affinity_but_not_functional_dependence() {
    let device = NdArrayDevice::default();
    let x = attrs(&device);

    // logit(param) = log(p / (1 - p)) recovers the pre-sigmoid column.
    let logit = |head: &ddrs::nn::kan_head::KanHead<B>, key: &str| -> Vec<f64> {
        field(head, x.clone(), key)
            .into_iter()
            .map(|v| (v as f64 / (1.0 - v as f64)).ln())
            .collect()
    };
    // Residual std of an OLS fit of `b` on `a`, relative to the std of `b`.
    // 0 = exactly affine.
    let rel_resid = |a: &[f64], b: &[f64]| -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let sab: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let saa: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
        let sbb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
        let slope = sab / saa;
        let resid: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (y - mb - slope * (x - ma)).powi(2))
            .sum();
        (resid / sbb).sqrt()
    };
    // Spearman |rho| via rank transform; ties are not possible here.
    let abs_spearman = |a: &[f64], b: &[f64]| -> f64 {
        let rank = |v: &[f64]| -> Vec<f64> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
            let mut r = vec![0.0; v.len()];
            for (pos, &i) in idx.iter().enumerate() {
                r[i] = pos as f64;
            }
            r
        };
        let (ra, rb) = (rank(a), rank(b));
        let n = ra.len() as f64;
        let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
        let num: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let da: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
        let db: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum::<f64>().sqrt();
        (num / (da * db)).abs()
    };

    // hidden_size = 1 forces a rank-1 latent: the §31 regime in miniature.
    let rank1 = |f: fn(KanHeadConfig) -> KanHeadConfig| {
        f(base(&["n", "q_spatial"]).with_hidden_size(1)).init::<B>(&device)
    };
    let linear_head = rank1(|c| c);
    let kan_head = rank1(|c| c.with_output_layer_kan(true));

    let (ln, lq) = (logit(&linear_head, "n"), logit(&linear_head, "q_spatial"));
    let (kn, kq) = (logit(&kan_head, "n"), logit(&kan_head, "q_spatial"));

    let lin = rel_resid(&ln, &lq);
    let kan = rel_resid(&kn, &kq);
    // Tolerance is the f32 sigmoid round-trip floor, not zero: the fields come
    // back as f32 probabilities and are pushed back through `logit` in f64.
    assert!(
        lin < 1e-3,
        "a Linear read-out on a rank-1 latent must make the two logit columns \
         affine to the f32 floor, got relative residual {lin}"
    );
    assert!(
        kan > 50.0 * lin.max(1e-6),
        "a KAN read-out must break the affine tie by a wide margin; \
         got kan={kan} against linear={lin}"
    );

    // The caveat, pinned: both remain deterministic functions of one scalar.
    assert!(
        abs_spearman(&kn, &kq) > 0.999,
        "a KAN read-out is not expected to decouple the outputs of a rank-1 \
         trunk — if this ever passes, the mechanism has changed"
    );
}

/// `input_layer_kan` must change the trunk (per-attribute splines instead of a
/// linear mixing) and must not silently fall back to the Linear path.
#[test]
fn kan_embedding_changes_the_trunk() {
    let device = NdArrayDevice::default();
    let x = attrs(&device);

    let linear_head = base(&["n", "q_spatial"]).init::<B>(&device);
    let kan_head = base(&["n", "q_spatial"])
        .with_input_layer_kan(true)
        .init::<B>(&device);

    assert!(kan_head.input_kan.is_some());
    let a = linear_head
        .trunk_activations(x.clone())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let b = kan_head
        .trunk_activations(x)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_ne!(a, b, "KAN embedding produced the Linear trunk's activations");
}

#[test]
#[should_panic(expected = "not assigned to any group")]
fn groups_must_cover_every_learnable_parameter() {
    let device = NdArrayDevice::default();
    base(&["n", "q_spatial"])
        .with_parameter_groups(vec![vec!["n".into()]])
        .init::<B>(&device);
}

#[test]
#[should_panic(expected = "not in learnable_parameters")]
fn groups_must_not_name_unknown_parameters() {
    let device = NdArrayDevice::default();
    base(&["n"])
        .with_parameter_groups(vec![vec!["n".into()], vec!["x_storage".into()]])
        .init::<B>(&device);
}
