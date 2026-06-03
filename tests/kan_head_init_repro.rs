//! Confirms DDRS's `KanHead` produces bit-identical parameter tensors when
//! built twice with the same seed — a baseline correctness requirement
//! before any DDR fixture comparison is meaningful.

use burn::backend::NdArray;
use ddrs::nn::KanHeadConfig;

type B = NdArray<f32>;

fn make_cfg(seed: u64) -> KanHeadConfig {
    KanHeadConfig::new(
        (0..10).map(|i| format!("attr_{i}")).collect(),
        vec!["n".into(), "q_spatial".into(), "p_spatial".into()],
        seed,
    )
    .with_hidden_size(21)
    .with_num_hidden_layers(2)
    .with_grid(50)
    .with_k(2)
}

fn flatten<const D: usize>(t: burn::tensor::Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

#[test]
fn kan_head_init_is_bit_reproducible_at_fixed_seed() {
    let device = Default::default();
    let cfg = make_cfg(42);
    let h1 = cfg.init::<B>(&device);
    let h2 = cfg.init::<B>(&device);

    assert_eq!(
        flatten(h1.input.weight.val()),
        flatten(h2.input.weight.val()),
        "input.weight differs between two builds at seed=42"
    );
    assert_eq!(
        flatten(h1.output.weight.val()),
        flatten(h2.output.weight.val()),
        "output.weight differs between two builds at seed=42"
    );
    // Inner KAN blocks must also be reproducible — this is what rskan
    // guarantees, but we re-check here at the head level.
    for (idx, (a, b)) in h1.hidden.iter().zip(h2.hidden.iter()).enumerate() {
        assert_eq!(
            flatten(a.coef.val()),
            flatten(b.coef.val()),
            "hidden[{idx}].coef differs between two builds at seed=42"
        );
        assert_eq!(
            flatten(a.scale_base.val()),
            flatten(b.scale_base.val()),
            "hidden[{idx}].scale_base differs"
        );
    }
}

#[test]
fn kan_head_inner_blocks_have_identical_init_per_ddr_quirk() {
    // DDR creates a fresh `KAN([H,H], seed=seed)` per outer hidden layer.
    // Each call reseeds Torch+NumPy globals to the same seed, so the two
    // inner blocks end up with identical params. DDRS mirrors this by
    // re-using `self.seed` for every inner KanLayer. Validate that here so
    // any regression is loud.
    let device = Default::default();
    let head = make_cfg(42).init::<B>(&device);
    assert_eq!(head.hidden.len(), 2, "expected 2 inner KanLayers");

    let coef0 = flatten(head.hidden[0].coef.val());
    let coef1 = flatten(head.hidden[1].coef.val());
    assert_eq!(coef0, coef1, "hidden[0].coef != hidden[1].coef — DDR quirk lost");

    let sb0 = flatten(head.hidden[0].scale_base.val());
    let sb1 = flatten(head.hidden[1].scale_base.val());
    assert_eq!(sb0, sb1, "hidden[0].scale_base != hidden[1].scale_base");
}

#[test]
fn different_seeds_produce_different_inits() {
    let device = Default::default();
    let h1 = make_cfg(42).init::<B>(&device);
    let h2 = make_cfg(43).init::<B>(&device);
    assert_ne!(
        flatten(h1.input.weight.val()),
        flatten(h2.input.weight.val()),
        "input.weight identical for seeds 42 and 43 — RNG not actually consumed"
    );
}
