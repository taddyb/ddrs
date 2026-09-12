# DDR ↔ DDRS KAN parity testing — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a five-layer testing scaffold that (1) realigns DDRS's KAN
head config and initialization to match DDR's exactly, (2) lets a Rust test
assert bit-identical forward + backward output against a fixture dumped from
DDR-Python, and (3) confirms the CONUS-wide init distribution of Manning's
`n` agrees between the two implementations — answering whether the
0.02–0.03 saturation symptom is a head divergence or a training pathology.

**Architecture:** Three artefact families.
1. **Init enforcement** — replace burn's RNG-opaque `Initializer` for the
   embedding + head Linears with a project-controlled `StdRng`-seeded
   sampler that uses pytorch's Kaiming/Xavier formulas. Makes DDRS init
   reproducible at fixed seed AND statistically identical to DDR.
2. **Fixture exchange** — Python script under DDR's `uv` venv dumps an
   `.npz` containing every parameter tensor + sample inputs + expected
   forward outputs + expected gradients. New `KanHead::from_npz` loader
   reads it through `ndarray-npy`. Three integration tests (init stats,
   forward, backward) consume the fixture.
3. **CONUS init comparison** — a Rust example + a DDR-side Python
   mirror, both sweep all 346 321 MERIT reaches through a fresh head
   and write `kan_init_params.nc`; a notebook in the `ddrs-eval-plots`
   skill plots histograms side by side.

**Tech Stack:** Rust 2021 + Burn 0.21 + rskan 0.1.0 (existing). Python 3.13
under DDR's `uv` venv at `~/projects/ddr/.venv/` (existing). New Rust dep
`ndarray-npy = "0.9"` for `.npz` I/O. No new Python deps (pykan, numpy,
torch already installed under DDR).

**Spec source of truth:** `docs/superpowers/specs/2026-06-02-ddr-ddrs-kan-parity-design.md`.
Do not relitigate decisions made there; if a question arises, re-read the
spec and decide consistently with it.

---

## File structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `config/merit_training.yaml` | modify | Move `kan_head.grid: 5 → 50`, `kan_head.k: 3 → 2` (spec §5 Option A). |
| `src/nn/kan_head.rs` | modify | (a) replace `Initializer::KaimingNormal`/`XavierNormal` for the two Linears with `sample_kaiming_normal_relu` / `sample_xavier_normal` helpers built on `StdRng`; (b) add `#[cfg(feature = "fixtures")] impl<B: Backend> KanHead<B> { pub fn from_npz(...) }`. |
| `src/nn/init.rs` | create | Project-controlled seeded init helpers (`sample_kaiming_normal_relu`, `sample_xavier_normal`, `make_zero_bias`) that match PyTorch's formulas exactly. Uses `rand::rngs::StdRng` — the same RNG family rskan uses. |
| `src/nn/mod.rs` | modify | `pub mod init;`. |
| `Cargo.toml` | modify | Add `ndarray-npy = "0.9"` under `[dev-dependencies]`; add `[features] fixtures = []`. |
| `tests/kan_head_init_repro.rs` | create | Asserts: building `KanHead` twice with the same `seed=42` produces bit-identical Linear weights AND bit-identical inner KanLayer coefs. |
| `tests/kan_head_init_parity.rs` | create | Reads `tests/fixtures/kan_init_stats_ddr.csv`, asserts per-tensor mean/std rel-err ≤ 5e-2 and KS-statistic < 0.05. |
| `tests/kan_head_fixture_forward.rs` | create | Reads `tests/fixtures/kan_head_init_seed42.npz`, builds `KanHead::from_npz`, runs forward, asserts max-abs-diff ≤ 1e-6 (NdArray) / 1e-4 (CUDA, gated). |
| `tests/kan_head_fixture_backward.rs` | create | Same fixture; uses `Autodiff` wrapper; computes loss = sum of all outputs, calls backward, asserts per-param max-abs-diff ≤ 1e-5 (NdArray) / 1e-3 (CUDA). |
| `tests/fixtures/kan_init_stats_ddr.csv` | create (committed) | One row per parameter tensor: `name,shape,mean,std,min,max,abs_mean`. |
| `tests/fixtures/kan_head_init_seed42.npz` | create (committed) | All param tensors + `inputs[64,10]` + 3× `expected_<param>[64]` + 3× `expected_grad_<param_name>`. |
| `tests/fixtures/README.md` | create | Documents how each fixture was generated (which script under DDR's venv, which seed). |
| `scripts/dump_kan_init_stats.py` | create | Python (run under DDR's venv) — emits the CSV above. |
| `scripts/dump_kan_fixture.py` | create | Python (run under DDR's venv) — emits the `.npz` above, including the gradient-fixture extension. |
| `examples/dump_init_params.rs` | create | Sweeps all CONUS attributes through a fresh DDRS `KanHead`, denormalises, writes `kan_init_params.nc`. |
| `scripts/dump_ddr_init_params.py` | create | Python mirror of the above using `ddr.nn.kan.kan`. |
| `.claude/skills/ddrs-eval-plots/references/parity_init.md` | create | New reference cell-by-cell instructions for the side-by-side init histogram notebook. |

---

## Task 1: Config alignment + saturation-hypothesis check

**Spec ref:** §5 (Option A) and §6 step 1.

**Files:**
- Modify: `config/merit_training.yaml:42-45`
- Run: `cargo test --test kan_head`
- Run: training (background; not blocking)

- [ ] **Step 1: Edit the config**

In `config/merit_training.yaml`, change:

```yaml
kan_head:
  hidden_size: 21
  num_hidden_layers: 2
  # B-spline grid intervals (`num` in pykan). Match DDR's default.
  grid: 5
  # B-spline order. Cubic.
  k: 3
```

to:

```yaml
kan_head:
  hidden_size: 21
  num_hidden_layers: 2
  # B-spline grid intervals (`num` in pykan). Matches DDR's
  # ~/projects/ddr/config/merit_training_config.yaml::kan.grid.
  grid: 50
  # B-spline order. Matches DDR's ::kan.k. (pykan's KANLayer default is 3,
  # but DDR overrides to 2 for production.)
  k: 2
```

- [ ] **Step 2: Verify existing tests still pass with new config**

Run: `cargo test --test kan_head`
Expected: PASS (all five existing tests are config-agnostic — they use their
own factory).

Run: `cargo test --test training_verification loads_merit_training_yaml -- --nocapture`
Expected: This test was failing BEFORE this change (per session notes). It
may still fail. Capture the failure output; if the failure mode references
`grid` or `k`, update the test's expected values. Otherwise leave alone —
out of scope per spec §6 step 1.

- [ ] **Step 3: Commit**

```bash
git add config/merit_training.yaml
git commit -m "config: align kan_head grid/k with DDR production (50/2)

Brings DDRS's merit_training.yaml into actual parity with DDR's
config/merit_training_config.yaml (kan.grid=50, kan.k=2). The previous
comment falsely claimed parity while shipping 5/3 — see parity spec §5
Option A.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Kick off a fresh training run in the background**

This is the saturation-hypothesis test. If grid/k was the root cause, the
new run's `n` distribution will not saturate. The run takes ~30 min on
this box; do it now so the data is ready when Layer 4 is built.

Run: `cargo run --release --bin ddrs -- run --workflow train-and-test`

Capture the new run-id (it appears in `.ddrs/runs/`).
Note the run-id in the plan execution log; do NOT block on completion.

---

## Task 2: Layer 0 audit — fill the spec tables

**Spec ref:** §4 Layer 0, sub-tables 0.1–0.5.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-02-ddr-ddrs-kan-parity-design.md` (replace `(verify)` placeholders with ✓ or ✗)

- [ ] **Step 1: Inspect each spec table row marked `(verify)`**

For each `(verify)` cell, read the DDR / DDRS source line cited in the
adjacent column and confirm the value. Replace `(verify)` with ✓ if the
value matches, or with the actual mismatched value followed by ✗.

Concrete checks:
- 0.1 `input_size`: `grep -c '^    - ' ~/projects/ddr/config/merit_training_config.yaml` under `kan.input_var_names`, and same for ddrs's yaml. Both should be 10.
- 0.1 `hidden_size`, `num_hidden_layers`, `output_size`: walk the yamls.
- 0.1 ordered name lists: a 2-column `diff <(yq '.kan.input_var_names[]' ddr.yaml) <(yq '.kan_head.input_var_names[]' ddrs.yaml)` returns nothing.
- 0.2 weight shape: trivially true given input_size + hidden_size match.
- 0.2 bias shape: trivially true given hidden_size matches.
- 0.4 weight + bias shape: trivially true given hidden_size + output_size match.

- [ ] **Step 2: Identify any remaining `investigate` rows**

The 0.5 "Are `input.weight` and `output.weight` reproducible…" row is
flagged `investigate`. Resolve it now: write a five-line throwaway program
that builds `KanHead<NdArrayBackend>` twice with the same `seed=42` and
prints `head.input.weight.val().to_data()` for both. Open `bash` in the
repo root and run:

```bash
cat <<'EOF' > /tmp/probe_init_repro.rs
use burn::backend::NdArray;
use ddrs::nn::KanHeadConfig;

fn main() {
    let device = Default::default();
    let cfg = KanHeadConfig::new(
        (0..10).map(|i| format!("attr_{i}")).collect(),
        vec!["n".into(), "q_spatial".into(), "p_spatial".into()],
        42,
    );
    let h1 = cfg.init::<NdArray<f32>>(&device);
    let h2 = cfg.init::<NdArray<f32>>(&device);
    let w1: Vec<f32> = h1.input.weight.val().into_data().to_vec().unwrap();
    let w2: Vec<f32> = h2.input.weight.val().into_data().to_vec().unwrap();
    println!("first 5 of h1.input.weight: {:?}", &w1[..5]);
    println!("first 5 of h2.input.weight: {:?}", &w2[..5]);
    println!("equal? {}", w1 == w2);
}
EOF
```

(Probe is for the author; convert to a proper test in Task 4. Don't commit
`/tmp/probe_init_repro.rs`.)

The probe answers the C7 question. Document the outcome inline in the spec:
either ✓ (burn's initializer IS seedable and DDRS init IS reproducible at
fixed seed — record which RNG path produced it) or ✗ (burn's initializer
is NOT seedable — Task 3 is required to fix).

- [ ] **Step 3: Commit the audited spec**

```bash
git add docs/superpowers/specs/2026-06-02-ddr-ddrs-kan-parity-design.md
git commit -m "docs/spec: complete Layer 0 hyperparameter audit

Replaces (verify) placeholders with concrete ✓/✗ verdicts after reading
both DDR and DDRS sources. Records the result of the C7 RNG-repro probe.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Replace burn's Linear initializer with a seeded `StdRng` sampler

**Spec ref:** §2 C7 and §4 Layer 0 sub-table 0.5 "Are Linears reproducible".
This is the **enforcement** of identical initialization, not just audit.

**Files:**
- Create: `src/nn/init.rs`
- Modify: `src/nn/mod.rs`
- Modify: `src/nn/kan_head.rs`
- Test: `tests/kan_head_init_repro.rs` (Task 4)

**Why both Linears use a project-controlled `StdRng`:** PyTorch samples
Kaiming-normal via its global Mersenne-Twister; burn 0.21's
`Initializer::KaimingNormal` samples via burn-internal RNG that is NOT
trivially seeded from the `seed` field on `KanHeadConfig`. To make DDRS
init reproducible at fixed `seed` (a baseline correctness requirement
before any DDR comparison is meaningful), we replace burn's `Initializer`
calls with explicit `StdRng`-seeded sampling that follows the same
formulas. Bonus: the inner KAN blocks already use `StdRng` via rskan, so
this unifies the entire head's RNG family.

- [ ] **Step 1: Write the failing test for `sample_kaiming_normal_relu`**

Create `src/nn/init.rs`:

```rust
//! Project-controlled seeded initialization for `Linear` weights.
//!
//! These helpers replace `burn::nn::Initializer::{KaimingNormal, XavierNormal}`
//! in the KAN head so that:
//!   (a) DDRS head init is reproducible across runs at a fixed seed,
//!   (b) it uses the same `rand::rngs::StdRng` family as `rskan::KanLayer`,
//!       removing one cross-module RNG source as a parity-test variable.
//!
//! Formulas mirror PyTorch (`torch/nn/init.py:578` for Kaiming,
//! `torch/nn/init.py:469` for Xavier) so the distributions match DDR's
//! `nn/kan.py:45-46` calls element-for-element (modulo RNG bytes, per
//! spec C4).

use burn::module::Param;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

/// Sample a Kaiming-normal `[out_dim, in_dim]` weight matrix for a
/// `Linear(in_dim, out_dim)` whose downstream nonlinearity is ReLU.
///
/// `std = sqrt(2) / sqrt(in_dim)` — equivalent to PyTorch's
/// `kaiming_normal_(mode="fan_in", nonlinearity="relu")`.
pub fn sample_kaiming_normal_relu(
    rng: &mut StdRng,
    in_dim: usize,
    out_dim: usize,
) -> Array2<f32> {
    let std = (2.0_f32 / in_dim as f32).sqrt();
    let normal = StandardNormal;
    Array2::from_shape_fn((out_dim, in_dim), |_| {
        normal.sample::<f32, _>(rng) * std
    })
}

/// Sample a Xavier-normal `[out_dim, in_dim]` weight matrix for a Linear
/// whose downstream nonlinearity is sigmoid/tanh-like.
///
/// `std = gain * sqrt(2 / (in_dim + out_dim))` — equivalent to PyTorch's
/// `xavier_normal_(gain=gain)`.
pub fn sample_xavier_normal(
    rng: &mut StdRng,
    in_dim: usize,
    out_dim: usize,
    gain: f32,
) -> Array2<f32> {
    let std = gain * (2.0_f32 / (in_dim + out_dim) as f32).sqrt();
    let normal = StandardNormal;
    Array2::from_shape_fn((out_dim, in_dim), |_| {
        normal.sample::<f32, _>(rng) * std
    })
}

/// Promote an `ndarray::Array2<f32>` into a Burn `Param<Tensor<B, 2>>`.
pub fn to_param_weight<B: Backend>(
    arr: Array2<f32>,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let data = TensorData::new(arr.as_slice().unwrap().to_vec(), [rows, cols]);
    Param::from_tensor(Tensor::from_data(data, device))
}

/// Construct a zero-initialised bias tensor.
pub fn zero_bias_tensor<B: Backend>(
    dim: usize,
    device: &B::Device,
) -> Param<Tensor<B, 1>> {
    Param::from_tensor(Tensor::<B, 1>::zeros([dim], device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn kaiming_normal_relu_is_reproducible() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let a = sample_kaiming_normal_relu(&mut rng_a, 10, 21);
        let b = sample_kaiming_normal_relu(&mut rng_b, 10, 21);
        assert_eq!(a, b);
    }

    #[test]
    fn xavier_normal_is_reproducible() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let a = sample_xavier_normal(&mut rng_a, 21, 3, 0.1);
        let b = sample_xavier_normal(&mut rng_b, 21, 3, 0.1);
        assert_eq!(a, b);
    }

    #[test]
    fn kaiming_std_matches_formula_at_large_n() {
        let mut rng = StdRng::seed_from_u64(0);
        let arr = sample_kaiming_normal_relu(&mut rng, 100, 5_000);
        let mean = arr.mean().unwrap();
        let var: f32 = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap();
        let std = var.sqrt();
        let expected = (2.0_f32 / 100.0).sqrt();
        assert!(
            (std - expected).abs() < 1e-3,
            "std={std}, expected≈{expected}"
        );
    }

    #[test]
    fn xavier_std_matches_formula_at_large_n() {
        let mut rng = StdRng::seed_from_u64(0);
        let arr = sample_xavier_normal(&mut rng, 100, 5_000, 0.1);
        let mean = arr.mean().unwrap();
        let var: f32 = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap();
        let std = var.sqrt();
        let expected = 0.1 * (2.0_f32 / 5_100.0).sqrt();
        assert!(
            (std - expected).abs() < 1e-4,
            "std={std}, expected≈{expected}"
        );
    }
}
```

- [ ] **Step 2: Wire the module + add the `rand_distr` dependency**

Edit `src/nn/mod.rs`:

```rust
pub mod init;
pub mod kan_head;
pub use kan_head::{KanHead, KanHeadConfig};
```

In `Cargo.toml`, add to `[dependencies]`:

```toml
rand_distr = "0.4"
```

(`rand_distr` is the standard companion to `rand 0.8` for `StandardNormal`.)

- [ ] **Step 3: Run the init unit tests**

Run: `cargo test --lib nn::init`
Expected: 4 passed.

- [ ] **Step 4: Refactor `kan_head.rs` to use the new init helpers**

Replace `kan_head.rs:74-126` (the body of `KanHeadConfig::init`) with:

```rust
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
        let output_weight = crate::nn::init::sample_xavier_normal(
            &mut rng, h, p, XAVIER_GAIN_OUTPUT as f32,
        );

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
```

Delete the now-unused `KAIMING_GAIN_RELU` const and the `zero_bias`
function. Delete the now-unused imports `Initializer`, `LinearConfig` (but
keep `Linear` — we construct it directly). Imports become:

```rust
use std::collections::HashMap;

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::tensor::activation::sigmoid;
use burn::tensor::{backend::Backend, Tensor};
use rand::SeedableRng;
use rskan::{KanLayer, KanLayerConfig};
```

Update the module docstring near the top (`kan_head.rs:14-23`) to read:

```rust
//! Init recipe (matches DDR `kan.py:45-48` element-for-element, with
//! `StdRng`-based sampling instead of PyTorch global MT — see C4):
//! - input Linear weight:  Kaiming-normal, `std = sqrt(2)/sqrt(F)`.
//! - output Linear weight: Xavier-normal, `std = 0.1 * sqrt(2/(H+P))`.
//! - both biases:          zero.
//! - hidden KanLayers:     `rskan::KanLayerConfig::new(H, H, seed)` with
//!                         `num=grid`, `k=k`, `noise_scale=0.3`. Same
//!                         `seed` for every inner KanLayer.
//! See `src/nn/init.rs` for the actual sampling code.
```

- [ ] **Step 5: Verify all existing KAN head tests still pass**

Run: `cargo test --test kan_head`
Expected: 6 passed (all five DDR ports + biases-zero test). The new
initialization path returns tensors of the same shape with statistically
identical distributions, so every existing assertion still holds.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml src/nn/init.rs src/nn/mod.rs src/nn/kan_head.rs
git commit -m "feat(nn): project-controlled StdRng init for KanHead Linears

Replaces burn::nn::Initializer::{KaimingNormal, XavierNormal} with
sample_{kaiming,xavier}_normal helpers seeded from a single StdRng
derived from KanHeadConfig::seed. Makes DDRS head init reproducible
across runs at a fixed seed (was implicitly using burn-internal RNG,
non-deterministic across runs — see parity spec C7), and unifies the
RNG family with rskan's inner KanLayer sampling.

The sampling formulas exactly mirror PyTorch's kaiming_normal_(relu)
and xavier_normal_(gain) — i.e. statistically identical to DDR's
nn/kan.py:45-46 modulo RNG bytes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Reproducibility integration test

**Spec ref:** §2 C7 follow-up.

**Files:**
- Create: `tests/kan_head_init_repro.rs`

- [ ] **Step 1: Write the test**

```rust
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
```

- [ ] **Step 2: Run the new test**

Run: `cargo test --test kan_head_init_repro`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/kan_head_init_repro.rs
git commit -m "test: assert KanHead init is bit-reproducible at fixed seed

Three tests: (1) seed=42 produces identical bytes across two builds for
both Linears + all inner KanLayers; (2) DDR's MultKAN-reseeding quirk
where both inner blocks end up with identical params is preserved;
(3) different seeds actually produce different bytes (sanity check
that the RNG is being consumed).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: DDR-side script — dump init statistics CSV

**Spec ref:** §4 Layer 1 step 1.

**Files:**
- Create: `scripts/dump_kan_init_stats.py`
- Create: `tests/fixtures/kan_init_stats_ddr.csv` (committed output)

- [ ] **Step 1: Write the dumper**

Create `scripts/dump_kan_init_stats.py`:

```python
"""Dump per-tensor init statistics for DDR's `ddr.nn.kan.kan` head.

Run under DDR's uv venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_kan_init_stats.py

Output: ~/projects/ddrs/tests/fixtures/kan_init_stats_ddr.csv
        with one row per parameter tensor.

Schema: name,shape,mean,std,min,max,abs_mean

Note: pykan's `KAN([H, H], seed=seed)` calls `torch.manual_seed(seed)`,
`np.random.seed(seed)`, and `random.seed(seed)` (MultKAN.__init__) BEFORE
the two outer Linears are initialised. That global-state side effect is
why DDR's `input.weight` and `output.weight` end up reproducible at fixed
seed — we preserve the same construction order here.
"""

import csv
from pathlib import Path
import sys

# Match DDRS's config/merit_training.yaml exactly.
SEED = 42
INPUT_VAR_NAMES = [
    "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
    "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
]
LEARNABLE = ["n", "q_spatial", "p_spatial"]
HIDDEN_SIZE = 21
NUM_HIDDEN_LAYERS = 2
GRID = 50
K = 2

OUT_CSV = Path("~/projects/ddrs/tests/fixtures/kan_init_stats_ddr.csv").expanduser()


def main() -> None:
    sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
    from ddr.nn.kan import kan as DdrKan  # type: ignore

    model = DdrKan(
        input_var_names=INPUT_VAR_NAMES,
        learnable_parameters=LEARNABLE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        grid=GRID,
        k=K,
        seed=SEED,
        device="cpu",
    )

    rows: list[dict[str, object]] = []
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().numpy().astype("float32")
        rows.append({
            "name":    name,
            "shape":   "x".join(str(d) for d in arr.shape),
            "mean":    float(arr.mean()),
            "std":     float(arr.std()),
            "min":     float(arr.min()),
            "max":     float(arr.max()),
            "abs_mean": float(abs(arr).mean()),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows → {OUT_CSV}")
    for r in rows:
        print(f"  {r['name']:40s} shape={r['shape']:20s} mean={r['mean']:+.4e} std={r['std']:.4e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the dumper**

Run:
```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_kan_init_stats.py
```

Expected: console prints ~10 rows (`input.weight`, `input.bias`,
`output.weight`, `output.bias`, and for each of 2 hidden blocks:
`layers.0.act_fun.0.{grid,coef,scale_base,scale_sp,mask}` and the same
for `layers.1`). File written to `tests/fixtures/kan_init_stats_ddr.csv`.

Sanity-check the CSV: `input.weight` should have `std ≈ sqrt(2/10) ≈ 0.447`;
`output.weight` should have `std ≈ 0.1*sqrt(2/24) ≈ 0.029`; both biases
should be exactly zero.

- [ ] **Step 3: Commit the script + the generated CSV**

```bash
git add scripts/dump_kan_init_stats.py tests/fixtures/kan_init_stats_ddr.csv
git commit -m "scripts: dump DDR KAN per-tensor init statistics for parity tests

Python helper (run under ~/projects/ddr/.venv) that builds DDR's
ddr.nn.kan.kan with the exact same hyperparameters as DDRS's
merit_training.yaml + seed=42 and writes per-tensor (mean, std, min,
max, abs_mean) to tests/fixtures/kan_init_stats_ddr.csv. Consumed by
tests/kan_head_init_parity.rs in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Rust statistical-init parity test

**Spec ref:** §4 Layer 1 step 2-3.

**Files:**
- Create: `tests/kan_head_init_parity.rs`

- [ ] **Step 1: Write the test**

```rust
//! Layer 1 of the DDR↔DDRS KAN parity plan: assert DDRS's per-parameter
//! init *distributions* match DDR's, even though RNG bytes differ.
//!
//! Reads tests/fixtures/kan_init_stats_ddr.csv (produced by
//! scripts/dump_kan_init_stats.py under DDR's uv venv), computes the
//! corresponding statistics from a freshly-initialised DDRS KanHead, and
//! asserts mean/std relative error ≤ 5%.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use burn::backend::NdArray;
use ddrs::nn::{KanHead, KanHeadConfig};

type B = NdArray<f32>;

#[derive(Debug, Clone)]
struct DdrStats {
    shape: String,
    mean: f64,
    std: f64,
}

fn load_ddr_stats() -> HashMap<String, DdrStats> {
    let path = Path::new("tests/fixtures/kan_init_stats_ddr.csv");
    let file = File::open(path).unwrap_or_else(|e| {
        panic!("missing fixture {path:?}: {e}. Re-run scripts/dump_kan_init_stats.py")
    });
    let mut reader = csv::Reader::from_reader(file);
    let mut out = HashMap::new();
    for record in reader.records() {
        let r = record.unwrap();
        let name = r[0].to_string();
        out.insert(name, DdrStats {
            shape: r[1].to_string(),
            mean: r[2].parse().unwrap(),
            std:  r[3].parse().unwrap(),
        });
    }
    out
}

fn stats_of(values: &[f32]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = values.iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>() / n;
    (mean, var.sqrt())
}

fn assert_stat_close(name: &str, got: f64, want: f64, tol: f64) {
    if want.abs() < 1e-6 {
        // Compare absolute when expected value is near zero (e.g. biases).
        assert!(
            got.abs() < 1e-6,
            "{name}: expected ≈0 got {got:+e}"
        );
    } else {
        let rel = ((got - want) / want).abs();
        assert!(
            rel < tol,
            "{name}: got {got:+e}, want {want:+e}, rel_err {rel:.4} > {tol}"
        );
    }
}

fn make_parity_head() -> KanHead<B> {
    let device = Default::default();
    let cfg = KanHeadConfig::new(
        vec![
            "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
            "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
        ].into_iter().map(String::from).collect(),
        vec!["n".into(), "q_spatial".into(), "p_spatial".into()],
        42,
    )
    .with_hidden_size(21)
    .with_num_hidden_layers(2)
    .with_grid(50)
    .with_k(2);
    cfg.init::<B>(&device)
}

#[test]
fn ddrs_init_matches_ddr_within_5pct() {
    let ddr = load_ddr_stats();
    let head = make_parity_head();
    let tol = 0.05_f64;

    // Map DDR's state-dict keys → (DDRS-side tensor extractor, expected key).
    // DDR keys:
    //   input.weight, input.bias, output.weight, output.bias,
    //   layers.<i>.act_fun.0.{grid,coef,scale_base,scale_sp,mask}
    let mut compared = 0;

    let probe = [
        ("input.weight",   head.input.weight.val().into_data().to_vec::<f32>().unwrap()),
        ("input.bias",     head.input.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap()),
        ("output.weight",  head.output.weight.val().into_data().to_vec::<f32>().unwrap()),
        ("output.bias",    head.output.bias.as_ref().unwrap().val().into_data().to_vec::<f32>().unwrap()),
    ];
    for (key, vals) in &probe {
        let want = ddr.get(*key).unwrap_or_else(|| panic!("DDR fixture missing {key}"));
        let (m, s) = stats_of(vals);
        assert_stat_close(&format!("{key}.mean"), m, want.mean, tol);
        assert_stat_close(&format!("{key}.std"),  s, want.std,  tol);
        compared += 1;
    }

    for (block_idx, layer) in head.hidden.iter().enumerate() {
        let pairs: Vec<(String, Vec<f32>)> = vec![
            (format!("layers.{block_idx}.act_fun.0.grid"),
             layer.grid.val().into_data().to_vec::<f32>().unwrap()),
            (format!("layers.{block_idx}.act_fun.0.coef"),
             layer.coef.val().into_data().to_vec::<f32>().unwrap()),
            (format!("layers.{block_idx}.act_fun.0.scale_base"),
             layer.scale_base.val().into_data().to_vec::<f32>().unwrap()),
            (format!("layers.{block_idx}.act_fun.0.scale_sp"),
             layer.scale_sp.val().into_data().to_vec::<f32>().unwrap()),
            (format!("layers.{block_idx}.act_fun.0.mask"),
             layer.mask.val().into_data().to_vec::<f32>().unwrap()),
        ];
        for (key, vals) in pairs {
            let want = ddr.get(&key)
                .unwrap_or_else(|| panic!("DDR fixture missing {key}"));
            let (m, s) = stats_of(&vals);
            assert_stat_close(&format!("{key}.mean"), m, want.mean, tol);
            assert_stat_close(&format!("{key}.std"),  s, want.std,  tol);
            compared += 1;
        }
    }

    assert_eq!(compared, 14, "expected 4 + 5*2 = 14 tensor comparisons");
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test --test kan_head_init_parity`
Expected: 1 passed. If it fails, the failure message names exactly which
tensor's mean or std diverges from DDR's — that's the bug to fix next.

- [ ] **Step 3: Commit**

```bash
git add tests/kan_head_init_parity.rs
git commit -m "test: assert DDRS KAN init distributions match DDR within 5% rel-err

Reads the per-tensor statistics CSV dumped from DDR-Python and compares
mean + std for every parameter tensor in a freshly-initialised DDRS
KanHead. 14 tensors checked (4 Linear weights/biases + 5 fields per
inner KanLayer × 2 blocks). Failure messages name the diverging tensor
to localize the bug.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: DDR-side script — dump full fixture .npz

**Spec ref:** §4 Layer 2 step 1.

**Files:**
- Create: `scripts/dump_kan_fixture.py`
- Create: `tests/fixtures/kan_head_init_seed42.npz` (committed output)

- [ ] **Step 1: Write the dumper**

Create `scripts/dump_kan_fixture.py`:

```python
"""Dump a full DDR KAN init+forward+backward fixture for DDRS parity tests.

Run under DDR's uv venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_kan_fixture.py

Output: ~/projects/ddrs/tests/fixtures/kan_head_init_seed42.npz

Contents:
    inputs              [64, 10] float32 — sampled with seed=0 in this script
    expected_n          [64]     float32 — DDR forward output for `n`
    expected_q_spatial  [64]     float32
    expected_p_spatial  [64]     float32
    input_weight        [21, 10] float32
    input_bias          [21]     float32
    output_weight       [3, 21]  float32
    output_bias         [3]      float32

    block_0_grid        [21, knots]      float32  (knots = grid+1+2k = 55)
    block_0_coef        [21, 21, n_basis] float32 (n_basis = grid+k = 52)
    block_0_scale_base  [21, 21] float32
    block_0_scale_sp    [21, 21] float32
    block_0_mask        [21, 21] float32
    block_1_*           (same shapes)

    grad_input_weight   [21, 10] float32
    grad_input_bias     [21]     float32
    grad_output_weight  [3, 21]  float32
    grad_output_bias    [3]      float32
    grad_block_<b>_<f>  (only for the trainable params per layer:
                         coef, scale_base, scale_sp; not grid or mask)

    meta                json blob (version, hyperparams)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 42
INPUTS_SEED = 0
BATCH = 64
INPUT_VAR_NAMES = [
    "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
    "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
]
LEARNABLE = ["n", "q_spatial", "p_spatial"]
HIDDEN_SIZE = 21
NUM_HIDDEN_LAYERS = 2
GRID = 50
K = 2

OUT_NPZ = Path("~/projects/ddrs/tests/fixtures/kan_head_init_seed42.npz").expanduser()


def main() -> None:
    sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
    from ddr.nn.kan import kan as DdrKan  # type: ignore

    model = DdrKan(
        input_var_names=INPUT_VAR_NAMES,
        learnable_parameters=LEARNABLE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        grid=GRID,
        k=K,
        seed=SEED,
        device="cpu",
    )
    model.eval()  # disable dropout etc., though kan.py has none

    torch.manual_seed(INPUTS_SEED)
    inputs = torch.randn(BATCH, len(INPUT_VAR_NAMES), dtype=torch.float32)

    # Forward (with grad to capture the backward later).
    inputs_v = inputs.detach().clone().requires_grad_(False)
    out = model(inputs=inputs_v)
    expected = {k: v.detach().cpu().numpy().astype("float32") for k, v in out.items()}

    # Backward: scalar loss = sum of all three output parameters.
    loss = sum(out[k].sum() for k in LEARNABLE)
    loss.backward()

    # Param dict (named state).
    payload: dict[str, np.ndarray] = {
        "inputs": inputs.numpy().astype("float32"),
        **{f"expected_{k}": v for k, v in expected.items()},

        "input_weight":  model.input.weight.detach().cpu().numpy().astype("float32"),
        "input_bias":    model.input.bias.detach().cpu().numpy().astype("float32"),
        "output_weight": model.output.weight.detach().cpu().numpy().astype("float32"),
        "output_bias":   model.output.bias.detach().cpu().numpy().astype("float32"),

        "grad_input_weight":  model.input.weight.grad.detach().cpu().numpy().astype("float32"),
        "grad_input_bias":    model.input.bias.grad.detach().cpu().numpy().astype("float32"),
        "grad_output_weight": model.output.weight.grad.detach().cpu().numpy().astype("float32"),
        "grad_output_bias":   model.output.bias.grad.detach().cpu().numpy().astype("float32"),
    }

    for block_idx, layer in enumerate(model.layers):
        # MultKAN.act_fun is a ModuleList; for width=[H, H] it has length 1.
        inner = layer.act_fun[0]
        prefix = f"block_{block_idx}"
        payload[f"{prefix}_grid"]       = inner.grid.detach().cpu().numpy().astype("float32")
        payload[f"{prefix}_coef"]       = inner.coef.detach().cpu().numpy().astype("float32")
        payload[f"{prefix}_scale_base"] = inner.scale_base.detach().cpu().numpy().astype("float32")
        payload[f"{prefix}_scale_sp"]   = inner.scale_sp.detach().cpu().numpy().astype("float32")
        payload[f"{prefix}_mask"]       = inner.mask.detach().cpu().numpy().astype("float32")

        # Gradients only for trainable tensors (coef, scale_base, scale_sp).
        # grid and mask carry requires_grad=False.
        payload[f"grad_{prefix}_coef"]       = inner.coef.grad.detach().cpu().numpy().astype("float32")
        payload[f"grad_{prefix}_scale_base"] = inner.scale_base.grad.detach().cpu().numpy().astype("float32")
        payload[f"grad_{prefix}_scale_sp"]   = inner.scale_sp.grad.detach().cpu().numpy().astype("float32")

    meta = {
        "version": 1,
        "seed": SEED,
        "inputs_seed": INPUTS_SEED,
        "batch": BATCH,
        "in": len(INPUT_VAR_NAMES),
        "hidden": HIDDEN_SIZE,
        "out": len(LEARNABLE),
        "grid": GRID,
        "k": K,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "learnable_parameters": LEARNABLE,
    }
    payload["meta"] = np.array(json.dumps(meta), dtype=object)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **payload)
    print(f"wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size/1024:.1f} KiB)")
    print(f"  keys = {sorted(payload.keys())}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the dumper**

Run:
```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_kan_fixture.py
```

Expected: prints the output path + the full key list. The file is ~200 KiB.

- [ ] **Step 3: Commit script + fixture**

```bash
git add scripts/dump_kan_fixture.py tests/fixtures/kan_head_init_seed42.npz
git commit -m "scripts: dump full DDR KAN init+forward+backward fixture

NPZ contains all parameter tensors (Linears + KanLayer fields × 2
blocks), sample inputs [64, 10] sampled with torch seed 0, expected
forward outputs per learnable parameter, and expected gradients per
trainable tensor. Consumed by the fixture-forward + fixture-backward
tests in the next two tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Add npz dep + fixtures feature + `KanHead::from_npz` loader

**Spec ref:** §4 Layer 2 step 2.

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/nn/kan_head.rs`
- Create: `tests/fixtures/README.md`

- [ ] **Step 1: Add the feature and dep**

In `Cargo.toml`:

```toml
[dependencies]
# ... existing entries ...
ndarray-npy = { version = "0.9", optional = true }

[features]
fixtures = ["dep:ndarray-npy"]
```

- [ ] **Step 2: Add the loader**

Append to `src/nn/kan_head.rs`:

```rust
#[cfg(feature = "fixtures")]
mod fixture {
    use super::*;
    use ndarray::{Array1, Array2, Array3};
    use ndarray_npy::NpzReader;
    use std::fs::File;
    use std::io;
    use std::path::Path;

    fn err_other(msg: impl Into<String>) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, msg.into())
    }

    fn linear_from_parts<B: Backend>(
        weight: Array2<f32>,
        bias: Array1<f32>,
        device: &B::Device,
    ) -> Linear<B> {
        let (out_dim, in_dim) = (weight.shape()[0], weight.shape()[1]);
        let bias_dim = bias.shape()[0];
        let w_data = TensorData::new(
            weight.as_slice().unwrap().to_vec(),
            [out_dim, in_dim],
        );
        let b_data = TensorData::new(bias.to_vec(), [bias_dim]);
        Linear {
            weight: Param::from_tensor(Tensor::from_data(w_data, device)),
            bias:   Some(Param::from_tensor(Tensor::from_data(b_data, device))),
        }
    }

    impl<B: Backend> KanHead<B> {
        /// Build a `KanHead` from a `.npz` fixture (Python-side dump). All
        /// initializers are bypassed — every tensor is loaded byte-for-byte
        /// from the fixture file.
        pub fn from_npz(
            path: &Path,
            device: &B::Device,
            cfg: &KanHeadConfig,
        ) -> io::Result<Self> {
            let file = File::open(path)?;
            let mut npz = NpzReader::new(file).map_err(|e| err_other(e.to_string()))?;

            let read_2 = |npz: &mut NpzReader<File>, k: &str| -> io::Result<Array2<f32>> {
                npz.by_name::<f32, _>(k).map_err(|e| err_other(format!("read {k}: {e}")))
                    .and_then(|a: ndarray::ArrayD<f32>| {
                        a.into_dimensionality::<ndarray::Ix2>()
                            .map_err(|e| err_other(format!("{k}: not 2D: {e}")))
                    })
            };
            let read_1 = |npz: &mut NpzReader<File>, k: &str| -> io::Result<Array1<f32>> {
                npz.by_name::<f32, _>(k).map_err(|e| err_other(format!("read {k}: {e}")))
                    .and_then(|a: ndarray::ArrayD<f32>| {
                        a.into_dimensionality::<ndarray::Ix1>()
                            .map_err(|e| err_other(format!("{k}: not 1D: {e}")))
                    })
            };
            let read_3 = |npz: &mut NpzReader<File>, k: &str| -> io::Result<Array3<f32>> {
                npz.by_name::<f32, _>(k).map_err(|e| err_other(format!("read {k}: {e}")))
                    .and_then(|a: ndarray::ArrayD<f32>| {
                        a.into_dimensionality::<ndarray::Ix3>()
                            .map_err(|e| err_other(format!("{k}: not 3D: {e}")))
                    })
            };

            let input = linear_from_parts::<B>(
                read_2(&mut npz, "input_weight")?,
                read_1(&mut npz, "input_bias")?,
                device,
            );
            let output = linear_from_parts::<B>(
                read_2(&mut npz, "output_weight")?,
                read_1(&mut npz, "output_bias")?,
                device,
            );

            let mut hidden = Vec::with_capacity(cfg.num_hidden_layers);
            for b in 0..cfg.num_hidden_layers {
                let grid       = read_2(&mut npz, &format!("block_{b}_grid"))?;
                let coef       = read_3(&mut npz, &format!("block_{b}_coef"))?;
                let scale_base = read_2(&mut npz, &format!("block_{b}_scale_base"))?;
                let scale_sp   = read_2(&mut npz, &format!("block_{b}_scale_sp"))?;
                let mask       = read_2(&mut npz, &format!("block_{b}_mask"))?;

                let to_t2 = |a: Array2<f32>| {
                    let (r, c) = (a.shape()[0], a.shape()[1]);
                    Tensor::<B, 2>::from_data(
                        TensorData::new(a.as_slice().unwrap().to_vec(), [r, c]),
                        device,
                    )
                };
                let to_t3 = |a: Array3<f32>| {
                    let (d0, d1, d2) = (a.shape()[0], a.shape()[1], a.shape()[2]);
                    Tensor::<B, 3>::from_data(
                        TensorData::new(a.as_slice().unwrap().to_vec(), [d0, d1, d2]),
                        device,
                    )
                };

                let layer = KanLayerConfig::new(cfg.hidden_size, cfg.hidden_size, cfg.seed)
                    .with_num(cfg.grid)
                    .with_k(cfg.k)
                    .with_noise_scale(KAN_NOISE_SCALE)
                    .init_from_parts::<B>(
                        device,
                        to_t2(grid),
                        to_t3(coef),
                        to_t2(scale_base),
                        to_t2(scale_sp),
                        to_t2(mask),
                    );
                hidden.push(layer);
            }

            Ok(KanHead {
                input,
                hidden,
                output,
                learnable_parameters: cfg.learnable_parameters.clone(),
            })
        }
    }
}
```

Also, at the top of `kan_head.rs`, after the existing imports, add the
extra `TensorData` import (needed in the fixture module via `super::*`):

```rust
#[cfg(feature = "fixtures")]
use burn::tensor::TensorData;
```

- [ ] **Step 3: Build with the new feature on**

Run: `cargo build --features fixtures`
Expected: clean build, no warnings about unused `ndarray_npy` etc.

- [ ] **Step 4: Document the fixture lifecycle**

Create `tests/fixtures/README.md`:

```markdown
# DDR ↔ DDRS KAN parity fixtures

These are byte-for-byte snapshots of DDR-Python's `ddr.nn.kan.kan` at a
fixed seed, consumed by the parity integration tests under `tests/`.

## Regenerating

All scripts run under DDR's `uv` venv (`~/projects/ddr/.venv/`):

```bash
cd ~/projects/ddr
# per-tensor (mean, std) for Layer 1
uv run python ~/projects/ddrs/scripts/dump_kan_init_stats.py
# full param + forward + backward fixture for Layers 2-3
uv run python ~/projects/ddrs/scripts/dump_kan_fixture.py
```

Regenerate any time DDR's `nn/kan.py` or `pykan` is updated, then re-run
the parity test suite:

```bash
cargo test --features fixtures --test kan_head_init_parity \
                              --test kan_head_fixture_forward \
                              --test kan_head_fixture_backward
```

## Files

| File | Producer | Consumer |
|------|----------|----------|
| `kan_init_stats_ddr.csv` | `dump_kan_init_stats.py` | `kan_head_init_parity.rs` |
| `kan_head_init_seed42.npz` | `dump_kan_fixture.py` | `kan_head_fixture_forward.rs`, `kan_head_fixture_backward.rs` |
```

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/nn/kan_head.rs tests/fixtures/README.md
git commit -m "feat(nn): KanHead::from_npz loader behind \`fixtures\` feature

Adds a feature-gated constructor that builds a KanHead from a .npz
dumped by scripts/dump_kan_fixture.py — every parameter tensor is
loaded byte-for-byte, bypassing all initializers. Enables bitwise
forward + backward parity assertions vs DDR-Python.

Behind \`fixtures\` so production builds don't carry the ndarray-npy
dependency.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Fixture forward parity test (NdArray + CUDA)

**Spec ref:** §4 Layer 2 steps 3-4.

**Files:**
- Create: `tests/kan_head_fixture_forward.rs`

- [ ] **Step 1: Write the test**

```rust
//! Layer 2 of the DDR↔DDRS KAN parity plan: assert bit-identical forward
//! pass on both backends given a fixture-loaded head.
//!
//! Build with: `cargo test --features fixtures --test kan_head_fixture_forward`

#![cfg(feature = "fixtures")]

use std::path::Path;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use ndarray::Array2;
use ndarray_npy::NpzReader;

use ddrs::nn::{KanHead, KanHeadConfig};

type B = NdArray<f32>;

const FIXTURE: &str = "tests/fixtures/kan_head_init_seed42.npz";

fn parity_cfg() -> KanHeadConfig {
    KanHeadConfig::new(
        vec![
            "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
            "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
        ].into_iter().map(String::from).collect(),
        vec!["n".into(), "q_spatial".into(), "p_spatial".into()],
        42,
    )
    .with_hidden_size(21)
    .with_num_hidden_layers(2)
    .with_grid(50)
    .with_k(2)
}

fn read_array2(key: &str) -> Array2<f32> {
    let mut npz = NpzReader::new(std::fs::File::open(FIXTURE).unwrap()).unwrap();
    let a: ndarray::ArrayD<f32> = npz.by_name(key).unwrap();
    a.into_dimensionality::<ndarray::Ix2>().unwrap()
}

fn read_vec(key: &str) -> Vec<f32> {
    let mut npz = NpzReader::new(std::fs::File::open(FIXTURE).unwrap()).unwrap();
    let a: ndarray::ArrayD<f32> = npz.by_name(key).unwrap();
    a.into_raw_vec_and_offset().0
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "shape mismatch");
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn forward_matches_ddr_fixture_ndarray() {
    let device = Default::default();
    let cfg = parity_cfg();
    let head: KanHead<B> = KanHead::<B>::from_npz(Path::new(FIXTURE), &device, &cfg).unwrap();

    let inputs_arr = read_array2("inputs");
    let (n, f) = (inputs_arr.shape()[0], inputs_arr.shape()[1]);
    let inputs: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(inputs_arr.as_slice().unwrap().to_vec(), [n, f]),
        &device,
    );

    let out = head.forward(inputs);
    for key in ["n", "q_spatial", "p_spatial"] {
        let got: Vec<f32> = out[key].clone().into_data().to_vec().unwrap();
        let want = read_vec(&format!("expected_{key}"));
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff <= 1e-6,
            "{key}: max abs diff {diff} > 1e-6 on NdArray backend"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn forward_matches_ddr_fixture_cuda() {
    use burn::backend::Cuda;
    type Bc = Cuda<f32>;

    let device = Default::default();
    let cfg = parity_cfg();
    let head: KanHead<Bc> = KanHead::<Bc>::from_npz(Path::new(FIXTURE), &device, &cfg).unwrap();

    let inputs_arr = read_array2("inputs");
    let (n, f) = (inputs_arr.shape()[0], inputs_arr.shape()[1]);
    let inputs: Tensor<Bc, 2> = Tensor::from_data(
        TensorData::new(inputs_arr.as_slice().unwrap().to_vec(), [n, f]),
        &device,
    );

    let out = head.forward(inputs);
    for key in ["n", "q_spatial", "p_spatial"] {
        let got: Vec<f32> = out[key].clone().into_data().to_vec().unwrap();
        let want = read_vec(&format!("expected_{key}"));
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff <= 1e-4,
            "{key}: max abs diff {diff} > 1e-4 on CUDA backend"
        );
    }
}
```

- [ ] **Step 2: Add a `cuda` feature**

In `Cargo.toml`:

```toml
[features]
fixtures = ["dep:ndarray-npy"]
cuda = []  # enables the GPU branch of the fixture tests
```

(The CUDA backend is always compiled in via `burn-cuda` — this feature
flag only gates the CUDA-specific *test*. If the project already has a
gating convention, follow it instead.)

- [ ] **Step 3: Run on NdArray**

Run: `cargo test --features fixtures --test kan_head_fixture_forward forward_matches_ddr_fixture_ndarray`
Expected: 1 passed.

- [ ] **Step 4: Run on CUDA**

Run: `cargo test --features fixtures,cuda --test kan_head_fixture_forward forward_matches_ddr_fixture_cuda`
Expected: 1 passed on a CUDA-equipped box; skipped (compile-gated) otherwise.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml tests/kan_head_fixture_forward.rs
git commit -m "test: bit-identical forward parity vs DDR fixture on NdArray + CUDA

Loads the .npz fixture into KanHead::from_npz on both backends and
asserts max abs diff ≤ 1e-6 (NdArray) / 1e-4 (CUDA) for each of the
three learnable parameters. CUDA branch is feature-gated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Fixture backward parity test

**Spec ref:** §4 Layer 3.

**Files:**
- Create: `tests/kan_head_fixture_backward.rs`

- [ ] **Step 1: Write the test**

```rust
//! Layer 3 of the DDR↔DDRS KAN parity plan: assert gradient parity on
//! both backends given a fixture-loaded head. Wraps everything in
//! `Autodiff` and reproduces the loss DDR's `dump_kan_fixture.py` uses:
//!     loss = out["n"].sum() + out["q_spatial"].sum() + out["p_spatial"].sum()

#![cfg(feature = "fixtures")]

use std::path::Path;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::tensor::{Tensor, TensorData};
use ndarray::Array2;
use ndarray_npy::NpzReader;

use ddrs::nn::{KanHead, KanHeadConfig};

type B = Autodiff<NdArray<f32>>;

const FIXTURE: &str = "tests/fixtures/kan_head_init_seed42.npz";

fn parity_cfg() -> KanHeadConfig {
    KanHeadConfig::new(
        vec![
            "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
            "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
        ].into_iter().map(String::from).collect(),
        vec!["n".into(), "q_spatial".into(), "p_spatial".into()],
        42,
    )
    .with_hidden_size(21)
    .with_num_hidden_layers(2)
    .with_grid(50)
    .with_k(2)
}

fn read_array2(key: &str) -> Array2<f32> {
    let mut npz = NpzReader::new(std::fs::File::open(FIXTURE).unwrap()).unwrap();
    let a: ndarray::ArrayD<f32> = npz.by_name(key).unwrap();
    a.into_dimensionality::<ndarray::Ix2>().unwrap()
}

fn read_vec(key: &str) -> Vec<f32> {
    let mut npz = NpzReader::new(std::fs::File::open(FIXTURE).unwrap()).unwrap();
    let a: ndarray::ArrayD<f32> = npz.by_name(key).unwrap();
    a.into_raw_vec_and_offset().0
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "shape mismatch (got {}, want {})", got.len(), want.len());
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn backward_matches_ddr_fixture_ndarray() {
    let device = Default::default();
    let cfg = parity_cfg();
    let head: KanHead<B> = KanHead::<B>::from_npz(Path::new(FIXTURE), &device, &cfg).unwrap();

    let inputs_arr = read_array2("inputs");
    let (n, f) = (inputs_arr.shape()[0], inputs_arr.shape()[1]);
    let inputs: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(inputs_arr.as_slice().unwrap().to_vec(), [n, f]),
        &device,
    );

    let out = head.forward(inputs);
    let loss = out["n"].clone().sum()
        + out["q_spatial"].clone().sum()
        + out["p_spatial"].clone().sum();
    let grads = loss.backward();

    let tol = 1e-5_f32;

    // Embedding + head Linears.
    let pairs: Vec<(&str, Vec<f32>)> = vec![
        ("grad_input_weight",  head.input.weight.val().grad(&grads).unwrap().into_data().to_vec().unwrap()),
        ("grad_input_bias",    head.input.bias.as_ref().unwrap().val().grad(&grads).unwrap().into_data().to_vec().unwrap()),
        ("grad_output_weight", head.output.weight.val().grad(&grads).unwrap().into_data().to_vec().unwrap()),
        ("grad_output_bias",   head.output.bias.as_ref().unwrap().val().grad(&grads).unwrap().into_data().to_vec().unwrap()),
    ];
    for (key, got) in &pairs {
        let want = read_vec(key);
        let diff = max_abs_diff(got, &want);
        assert!(
            diff <= tol,
            "{key}: max abs grad diff {diff} > {tol}"
        );
    }

    // Inner KanLayer trainables.
    for (b, layer) in head.hidden.iter().enumerate() {
        for (field, grad_vec) in [
            ("coef",       layer.coef.val().grad(&grads).unwrap().into_data().to_vec().unwrap()),
            ("scale_base", layer.scale_base.val().grad(&grads).unwrap().into_data().to_vec().unwrap()),
            ("scale_sp",   layer.scale_sp.val().grad(&grads).unwrap().into_data().to_vec().unwrap()),
        ] {
            let key = format!("grad_block_{b}_{field}");
            let want = read_vec(&key);
            let diff = max_abs_diff(&grad_vec, &want);
            assert!(
                diff <= tol,
                "{key}: max abs grad diff {diff} > {tol}"
            );
        }
    }

    // Sanity — confirm valid module conversion works (consistency with
    // the existing kan_head.rs `_ = model_ad.valid()` check).
    let _ = head.valid();
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test --features fixtures --test kan_head_fixture_backward`
Expected: 1 passed. Any failure pinpoints exactly which gradient tensor
diverges from DDR's, which routes to either burn-autodiff or rskan's
custom Backward as the culprit.

- [ ] **Step 3: Commit**

```bash
git add tests/kan_head_fixture_backward.rs
git commit -m "test: gradient parity vs DDR fixture on Autodiff<NdArray>

Per-tensor max abs diff ≤ 1e-5 for all four Linear gradients plus the
three trainable fields (coef, scale_base, scale_sp) of each inner
KanLayer. Failure messages name the diverging tensor, routing the bug
to either burn-autodiff or rskan's custom Backward.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: CONUS init-distribution example + DDR mirror

**Spec ref:** §4 Layer 4.

**Files:**
- Create: `examples/dump_init_params.rs`
- Create: `scripts/dump_ddr_init_params.py`

- [ ] **Step 1: Write the DDRS example**

Create `examples/dump_init_params.rs`:

```rust
//! Sweep all CONUS MERIT reaches through a freshly-initialised DDRS
//! `KanHead` (no checkpoint, no training) and write per-COMID denormalised
//! parameters to a NetCDF. Used by the Layer 4 init-distribution
//! comparison against DDR.
//!
//! Run:
//!     cargo run --release --example dump_init_params -- \
//!         --config config/merit_training.yaml \
//!         --out    /tmp/kan_init_params_ddrs.nc
//!
//! Mirrors `scripts/dump_ddr_init_params.py` on the DDR side.

use std::path::PathBuf;

use burn::backend::NdArray;
use clap::Parser;
use ddrs::config::Config;
use ddrs::nn::KanHeadConfig;

type B = NdArray<f32>;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "config/merit_training.yaml")]
    config: PathBuf,
    #[arg(long, default_value = "/tmp/kan_init_params_ddrs.nc")]
    out: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let cfg = Config::from_yaml(&cli.config)?;

    let device = Default::default();
    let head_cfg = KanHeadConfig::new(
        cfg.kan_head.input_var_names.clone(),
        cfg.kan_head.learnable_parameters.clone(),
        cfg.seed,
    )
    .with_hidden_size(cfg.kan_head.hidden_size)
    .with_num_hidden_layers(cfg.kan_head.num_hidden_layers)
    .with_grid(cfg.kan_head.grid)
    .with_k(cfg.kan_head.k);
    let head = head_cfg.init::<B>(&device);

    // Pull every CONUS COMID + its attribute row.
    let comids = ddrs::data::store::netcdf::read_all_comids(
        &cfg.data_sources.attributes,
    )?;
    let attrs  = ddrs::data::store::netcdf::read_attribute_matrix(
        &cfg.data_sources.attributes,
        &cfg.kan_head.input_var_names,
        &comids,
    )?;
    // attrs shape: [n_reaches, n_attrs]
    let (n_reaches, n_attrs) = (attrs.shape()[0], attrs.shape()[1]);

    let inputs = burn::tensor::Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(
            attrs.as_slice().unwrap().to_vec(),
            [n_reaches, n_attrs],
        ),
        &device,
    );
    let raw = head.forward(inputs);  // sigmoid-bounded (0,1)

    // Denormalise per parameter via the existing denormalize helper.
    let denorm_map: std::collections::HashMap<String, Vec<f32>> = raw.iter()
        .map(|(k, t)| {
            let (lo, hi) = cfg.params.parameter_ranges[k];
            let log_space = cfg.params.log_space_parameters.contains(k);
            let v: Vec<f32> = t.clone().into_data().to_vec().unwrap();
            let dn = ddrs::config::denormalize(&v, (lo, hi), log_space);
            (k.clone(), dn)
        })
        .collect();

    // Write NetCDF: dim COMID, vars [n, q_spatial, p_spatial], coord comid.
    ddrs::data::write_netcdf::write_init_params(
        &cli.out,
        &comids,
        &denorm_map,
    )?;
    eprintln!("wrote {} reaches → {}", n_reaches, cli.out.display());
    Ok(())
}
```

(If `ddrs::data::store::netcdf::read_attribute_matrix` or
`ddrs::data::write_netcdf::write_init_params` does not yet exist in
the codebase, this Task should be reduced in scope: write a shim that
reuses `dump_parameters` if present (the session notes reference one),
or skip the dump step and instead make the example write CSV. The
plan author will know which path exists after one `find src/ -name
'*.rs' | xargs grep -l 'read_attribute_matrix\|dump_parameters'`.)

- [ ] **Step 2: Run it**

Run:
```bash
cargo run --release --example dump_init_params -- \
    --config config/merit_training.yaml \
    --out /tmp/kan_init_params_ddrs.nc
```

Expected: NetCDF file created with 346 321 reaches.

- [ ] **Step 3: Write the DDR mirror**

Create `scripts/dump_ddr_init_params.py`:

```python
"""DDR-side mirror of `examples/dump_init_params.rs`.

Builds DDR's `ddr.nn.kan.kan` at seed=42 with identical hyperparameters,
sweeps all CONUS MERIT reaches through it, writes per-COMID denormalised
parameters to a NetCDF.

Run under DDR's uv venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_ddr_init_params.py \
        --out /tmp/kan_init_params_ddr.nc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

SEED = 42
INPUT_VAR_NAMES = [
    "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
    "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
]
LEARNABLE = ["n", "q_spatial", "p_spatial"]
HIDDEN_SIZE = 21
NUM_HIDDEN_LAYERS = 2
GRID = 50
K = 2

# Verbatim from config/merit_training.yaml::params.parameter_ranges
PARAM_RANGES = {
    "n":         (0.015, 0.25),
    "q_spatial": (0.0, 1.0),
    "p_spatial": (1.0, 200.0),
}
LOG_SPACE = {"n"}
ATTRS_NC = Path("~/projects/ddr/data/merit_global_attributes_v2.nc").expanduser()


def denormalize(sigmoid_out: np.ndarray, lo: float, hi: float, log_space: bool) -> np.ndarray:
    if log_space:
        # Matches src/config.rs::denormalize log branch (eps = 1e-6 on lo).
        lo_eff = np.log(lo + 1e-6)
        hi_eff = np.log(hi)
        return np.exp(sigmoid_out * (hi_eff - lo_eff) + lo_eff)
    return sigmoid_out * (hi - lo) + lo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/kan_init_params_ddr.nc"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
    from ddr.nn.kan import kan as DdrKan  # type: ignore

    model = DdrKan(
        input_var_names=INPUT_VAR_NAMES,
        learnable_parameters=LEARNABLE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        grid=GRID,
        k=K,
        seed=SEED,
        device="cpu",
    )
    model.eval()

    with xr.open_dataset(ATTRS_NC) as ds:
        comids = ds["COMID"].values.astype("int64")
        attr_block = np.stack(
            [ds[name].values.astype("float32") for name in INPUT_VAR_NAMES],
            axis=1,
        )
    n_reaches = attr_block.shape[0]

    with torch.no_grad():
        out = model(inputs=torch.from_numpy(attr_block))
        raw = {k: v.cpu().numpy().astype("float32") for k, v in out.items()}

    denorm = {
        k: denormalize(raw[k], *PARAM_RANGES[k], k in LOG_SPACE)
        for k in LEARNABLE
    }

    xr.Dataset(
        {k: ("COMID", denorm[k]) for k in LEARNABLE},
        coords={"COMID": comids},
        attrs={"seed": SEED, "source": "ddr.nn.kan.kan @ seed=42"},
    ).to_netcdf(args.out)
    print(f"wrote {n_reaches} reaches → {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the DDR mirror**

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_init_params.py \
    --out /tmp/kan_init_params_ddr.nc
```

Expected: NetCDF file with 346 321 reaches.

- [ ] **Step 5: Commit**

```bash
git add examples/dump_init_params.rs scripts/dump_ddr_init_params.py
git commit -m "feat: dump fresh KAN init parameters over all CONUS reaches

Two parallel artefacts:
  - examples/dump_init_params.rs sweeps all CONUS MERIT attributes
    through a fresh DDRS KanHead (no checkpoint, no training) and
    writes per-COMID denormalised params to NetCDF.
  - scripts/dump_ddr_init_params.py does the same for DDR's
    ddr.nn.kan.kan under DDR's uv venv.

Both pin seed=42 + identical hyperparameters. Consumed by the
parity_init notebook in the next commit to produce side-by-side init
histograms for Manning's n / q_spatial / p_spatial.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Side-by-side init-distribution notebook + skill backport

**Spec ref:** §4 Layer 4 step 3.

**Files:**
- Create: `.claude/skills/ddrs-eval-plots/references/parity_init.md`

- [ ] **Step 1: Write the skill reference**

Create `.claude/skills/ddrs-eval-plots/references/parity_init.md`:

````markdown
# Layer 4 — DDR ↔ DDRS init-distribution side-by-side

## When to use

After running both `examples/dump_init_params` and
`scripts/dump_ddr_init_params.py`. The two NetCDFs they produce are the
inputs to this plot family.

## Inputs

| File | Producer |
|------|----------|
| `/tmp/kan_init_params_ddrs.nc` | `cargo run --release --example dump_init_params` |
| `/tmp/kan_init_params_ddr.nc`  | `cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_init_params.py` |

## Notebook cells

Place the notebook at `<RUN_DIR>/plots/parity_init.ipynb` even though
this plot is run-independent — keeps the artefacts colocated with
whichever run is being investigated.

### Cell 1 — load
```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DDRS_NC = Path("/tmp/kan_init_params_ddrs.nc")
DDR_NC  = Path("/tmp/kan_init_params_ddr.nc")
PLOT_DIR = Path(".").resolve()  # set to <RUN_DIR>/plots/ if needed

dssrs = xr.open_dataset(DDRS_NC)
ddr   = xr.open_dataset(DDR_NC)
assert (dssrs["COMID"].values == ddr["COMID"].values).all()
print(f"{dssrs.sizes['COMID']:,} reaches in both files")
```

### Cell 2 — histograms
```python
from scipy.stats import ks_2samp

PARAMS = ["n", "q_spatial", "p_spatial"]
RANGES = {"n": (0.015, 0.25), "q_spatial": (0, 1), "p_spatial": (1, 200)}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, key in zip(axes, PARAMS):
    lo, hi = RANGES[key]
    bins = np.linspace(lo, hi, 60)
    ax.hist(ddr[key].values,   bins=bins, alpha=0.5, label="DDR",  density=True)
    ax.hist(dssrs[key].values, bins=bins, alpha=0.5, label="DDRS", density=True)
    ks = ks_2samp(ddr[key].values, dssrs[key].values)
    ax.set_title(f"{key}  KS={ks.statistic:.3f} (p={ks.pvalue:.2g})")
    ax.set_xlabel(key)
    ax.legend()
fig.suptitle("DDR ↔ DDRS init distributions — seed=42, no training", fontsize=16)
fig.tight_layout()
fig.savefig(PLOT_DIR / "parity_init.png", dpi=200, bbox_inches="tight")
plt.show()
```

### Cell 3 — pass/fail
```python
TOL = 0.05  # spec A5
verdict = {}
for key in PARAMS:
    ks = ks_2samp(ddr[key].values, dssrs[key].values).statistic
    verdict[key] = ("✓" if ks < TOL else "✗", float(ks))
print("DDR ↔ DDRS init parity (Layer 4):")
for k, (v, ks) in verdict.items():
    print(f"  {k:12s}  {v}  KS={ks:.4f}")
```

## Pass criterion

KS < 0.05 on all three parameters. If `n` fails specifically, the
saturation symptom is a head-init divergence; if all three fail, look
upstream (denormalize, attribute normalisation). If all three pass yet
the trained `n` saturates after 5 epochs, the bug is in training
dynamics (optimizer / batching / loss) — outside the scope of the
parity plan.
````

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/ddrs-eval-plots/references/parity_init.md
git commit -m "ddrs-eval-plots: add Layer 4 init-distribution parity reference

Cell-by-cell notebook recipe for the DDR ↔ DDRS init histogram +
KS-test comparison, the test that disambiguates head-init divergence
from training-dynamics pathology for the Manning's n saturation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Final integration check

**Files:** none — verification only.

- [ ] **Step 1: Run the full parity suite**

```bash
cargo test --features fixtures --test kan_head_init_repro \
                              --test kan_head_init_parity \
                              --test kan_head_fixture_forward \
                              --test kan_head_fixture_backward
```

Expected: 3 + 1 + 1 + 1 = 6 tests passed.

- [ ] **Step 2: Regenerate the eval-plots notebook bundle for the new run**

The training run from Task 1 step 4 should be complete by now. Locate its
run-id:

```bash
ls -t .ddrs/runs/ | head -3
```

Use it with the existing `ddrs-eval-plots` skill to produce the standard
plot bundle PLUS the new `parity_init.ipynb` from Task 12.

- [ ] **Step 3: Compare the new training run's Manning's n histogram against the saturated one**

Open the new `parameter_map_n_conus.png` (skill output) and the saturated
one from `.ddrs/runs/2026-06-02T02-10-30Z-train-and-test/plots/`. The
saturation either disappeared (root cause was the grid/k config drift) or
persisted (root cause is downstream — open a new spec).

- [ ] **Step 4: Update CLAUDE.md with the parity test contract**

In `CLAUDE.md`, append a new invariant to the "Critical invariants" list:

```markdown
7. **KAN head parity vs DDR must pass on every PR that touches `src/nn/`,
   `Cargo.toml`'s rskan pin, or DDR's `nn/kan.py`.** Run:
   `cargo test --features fixtures --test kan_head_init_repro --test kan_head_init_parity --test kan_head_fixture_forward --test kan_head_fixture_backward`
   If a DDR change breaks the fixture, regenerate via
   `~/projects/ddr/.venv` + `scripts/dump_kan_*.py` and re-validate.
```

- [ ] **Step 5: Commit and ship**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE.md): add KAN parity test contract as invariant 7

Documents the rskan-head-swap parity invariant: any change to src/nn/,
rskan pin, or DDR's nn/kan.py must keep the four parity tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Spec coverage map

| Spec section | Plan task(s) |
|--------------|-------------|
| §1 motivation | (covered by plan goal) |
| §2 C1 fixture-schema drift | Task 7 (writes `meta` JSON), Task 8 (loader validates names) |
| §2 C2 fixture API gate | Task 8 (`fixtures` feature flag) |
| §2 C3 grid/k drift | Task 1 |
| §2 C4 RNG non-bit-parity | Task 3 (acceptance via statistical helpers), Tasks 5–6 (assert STAT only) |
| §2 C5 MultKAN seed quirk | Task 4 (`hidden[0].coef == hidden[1].coef`) |
| §2 C6 backend-specific divergence | Task 9 (NdArray + CUDA branches) |
| §2 C7 burn-initializer non-determinism | Task 3 (replaces it), Task 4 (proves repro) |
| §3 A1–A6 | All used as background; Task 5–7 cite specific assumptions in Python scripts |
| §4 Layer 0 audit | Task 2 |
| §4 Layer 1 statistical init parity | Tasks 5–6 |
| §4 Layer 2 fixture forward parity | Tasks 7–9 |
| §4 Layer 3 fixture backward parity | Tasks 7 + 10 |
| §4 Layer 4 CONUS init distribution | Tasks 11–12 |
| §5 Option A grid/k decision | Task 1 |
| §6 implementation order | Plan task order |
| §7 success criteria | Task 13 |

---

## Plan complete — execution choice

**Plan saved to `docs/superpowers/plans/2026-06-02-ddr-ddrs-kan-parity.md`.
Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?
