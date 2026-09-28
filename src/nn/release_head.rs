//! Learned dam release head (`params.reservoir_release: learned`).
//!
//! A second, independent [`KanHead`] instance: the same
//! `Linear(F, H) → KanLayer(H, H) × N → Linear(H, P) → Sigmoid` topology as
//! the routing head, reading the dam feature table
//! (`experiments/reservoir/release_head/build_dam_features.py`) and emitting,
//! per dam, `T0` (and, when `release_head.seasonal`, `a` and `b`) of
//!
//! ```text
//! T_d(t) = max(T0_d · exp(a_d · sin ω_t + b_d · cos ω_t), 1/24 d)
//! ```
//!
//! The routing head is untouched (invariant 5): this module only builds and
//! reads a separate instance, with its own checkpoint files and optimizer.
//!
//! **Denormalisation.** `T0` is always LOG space over
//! `params.parameter_ranges.reservoir_T0` (default `[1/24, 365]` d): residence
//! times span five decades, and a linear map would put the whole sub-day range
//! into the bottom 0.1 % of the sigmoid. `a`, `b` are linear over
//! `reservoir_a`, `reservoir_b` (default `[-2, 2]`), so a zero logit is
//! `a = b = 0` for a symmetric box.
//!
//! **Initialisation.** Near pass-through without a saturated sigmoid: the
//! output bias for `T0` is the logit that denormalises to
//! [`INIT_T0_HOURS`] = 4.5 h (sigmoid 0.166 on the default box, slope 0.138,
//! 55 % of the maximum 0.25), and 0 for `a`, `b` (sigmoid 0.5). The read-out
//! weights are zeroed, so every dam starts at exactly `T0 = 4.5 h`,
//! `a = b = 0`; the trunk keeps the routing head's Kaiming/KAN init, so the
//! read-out gradient `h ⊗ g` is nonzero and dam-specific from the first step.
//! (With the Xavier read-out kept instead, 40 synthetic feature rows started
//! as far as `T0` = 5.5 h and `|a|` or `|b|` = 0.11.) The seed is the config seed
//! XOR [`RELEASE_SEED_SALT`], so the release head never copies the routing
//! head's weights, and neither head's init consumes the other's RNG.

use std::collections::HashMap;

use burn::module::Param;
use burn::tensor::{backend::Backend, Tensor};

use crate::config::{ParameterRanges, ReleaseHeadSection};
use crate::nn::kan_head::{KanHead, KanHeadConfig};
use crate::routing::utils::denormalize;

/// XOR-ed into the config seed for the release head's weights.
pub const RELEASE_SEED_SALT: u64 = 0xDA7E_5EA5_0000_0001;

/// `T0` the release head emits at initialisation (hours), before the small
/// read-out weights add their spread.
pub const INIT_T0_HOURS: f32 = 4.5;

/// Output names of the release head, in column order.
pub fn release_outputs(seasonal: bool) -> Vec<String> {
    if seasonal {
        vec!["T0".into(), "a".into(), "b".into()]
    } else {
        vec!["T0".into()]
    }
}

/// `KanHeadConfig` for the release head.
pub fn release_head_config(section: &ReleaseHeadSection, seed: u64) -> KanHeadConfig {
    KanHeadConfig::new(
        section.input_var_names.clone(),
        release_outputs(section.seasonal),
        seed ^ RELEASE_SEED_SALT,
    )
    .with_hidden_size(section.hidden_size)
    .with_num_hidden_layers(section.num_hidden_layers)
    .with_grid(section.grid)
    .with_k(section.k)
}

/// Logit of `x` in `(0, 1)`.
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

/// Output-layer bias giving `T0 = INIT_T0_HOURS` (log space over `t0_range`)
/// and `a = b = 0` (linear over `a_range`, `b_range`). A box that does not
/// contain the init value is clamped to 1 % inside it, so the logit stays
/// finite.
pub fn release_init_bias(
    seasonal: bool,
    ranges: &ParameterRanges,
) -> Vec<f32> {
    let unit = |u: f64| u.clamp(0.01, 0.99);
    let [lo, hi] = ranges.reservoir_t0;
    let t0 = (INIT_T0_HOURS / 24.0) as f64;
    let u_t0 = (t0.ln() - (lo as f64).ln()) / ((hi as f64).ln() - (lo as f64).ln());
    let mut bias = vec![logit(unit(u_t0)) as f32];
    if seasonal {
        for [lo, hi] in [ranges.reservoir_a, ranges.reservoir_b] {
            let u = (0.0 - lo as f64) / (hi as f64 - lo as f64);
            bias.push(logit(unit(u)) as f32);
        }
    }
    bias
}

/// Build the release head with the pass-through initialisation described in
/// the module docs.
pub fn init_release_head<B: Backend>(
    section: &ReleaseHeadSection,
    ranges: &ParameterRanges,
    seed: u64,
    device: &B::Device,
) -> KanHead<B> {
    let mut head = release_head_config(section, seed).init::<B>(device);
    let bias = release_init_bias(section.seasonal, ranges);
    head.output.bias = Some(Param::from_tensor(Tensor::<B, 1>::from_floats(
        bias.as_slice(),
        device,
    )));
    // Zero read-out: every dam starts at exactly the bias values. The trunk
    // keeps its random init, so `∂L/∂W_out = h ⊗ g` is nonzero from the first
    // step and differs between dams through their features.
    let dims = head.output.weight.val().dims();
    head.output.weight = Param::from_tensor(Tensor::<B, 2>::zeros(dims, device));
    head
}

/// Per-dam release parameters in physical units.
pub struct ReleaseParams<B: Backend> {
    /// `[n_dams]`, days.
    pub t0_days: Tensor<B, 1>,
    /// `(a, b)`, each `[n_dams]`; `None` for a non-seasonal head.
    pub seasonal: Option<(Tensor<B, 1>, Tensor<B, 1>)>,
}

/// Run the release head on `features` (`[n_dams, F]`) and denormalise.
/// Autodiff flows through when `B` is an autodiff backend.
pub fn release_params<B: Backend>(
    head: &KanHead<B>,
    features: Tensor<B, 2>,
    ranges: &ParameterRanges,
) -> ReleaseParams<B> {
    let out: HashMap<String, Tensor<B, 1>> = head.forward(features);
    let t0 = out.get("T0").expect("release head emits T0").clone();
    let t0_days = denormalize(t0, ranges.reservoir_t0, true);
    let seasonal = match (out.get("a"), out.get("b")) {
        (Some(a), Some(b)) => Some((
            denormalize(a.clone(), ranges.reservoir_a, false),
            denormalize(b.clone(), ranges.reservoir_b, false),
        )),
        (None, None) => None,
        _ => panic!("a release head emits both a and b or neither"),
    };
    ReleaseParams { t0_days, seasonal }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn section(seasonal: bool) -> ReleaseHeadSection {
        ReleaseHeadSection {
            hidden_size: 8,
            num_hidden_layers: 1,
            grid: 5,
            k: 3,
            input_var_names: vec!["f1".into(), "f2".into(), "f3".into()],
            seasonal,
            dam_row: crate::config::DamRow::Replace,
            dam_floor: crate::config::DamFloor::Forgive,
            routing_checkpoint: None,
            freeze_routing: false,
            rule_curve: false,
            rule_curve_max: 1.0,
            per_dam_t0: false,
            per_dam_lr: 0.05,
            per_dam_l2: 0.0,
            rule_curve_penalty: 0.0,
            rule_curve_alpha: 0.9,
        }
    }

    fn features(n: usize) -> Tensor<B, 2> {
        // z-scored-looking inputs spanning ±2.5.
        let v: Vec<f32> = (0..n * 3).map(|i| ((i * 37 % 11) as f32 - 5.0) / 2.0).collect();
        Tensor::<B, 1>::from_floats(v.as_slice(), &Default::default()).reshape([n, 3])
    }

    #[test]
    fn init_is_near_pass_through_and_unsaturated() {
        let ranges = ParameterRanges::default();
        let head = init_release_head::<B>(&section(true), &ranges, 42, &Default::default());
        let p = release_params(&head, features(40), &ranges);
        let t0: Vec<f32> = p.t0_days.into_data().to_vec().unwrap();
        let (a, b) = p.seasonal.expect("seasonal");
        let a: Vec<f32> = a.into_data().to_vec().unwrap();
        let b: Vec<f32> = b.into_data().to_vec().unwrap();
        for &t in &t0 {
            let h = t * 24.0;
            assert!((h - INIT_T0_HOURS).abs() < 1e-3, "init T0 {h} h, want {INIT_T0_HOURS} h");
        }
        for &v in a.iter().chain(&b) {
            assert_eq!(v, 0.0, "init a/b must be exactly 0");
        }
    }

    #[test]
    fn init_bias_denormalises_to_the_documented_values() {
        let ranges = ParameterRanges::default();
        let bias = release_init_bias(true, &ranges);
        let t0 = denormalize(
            Tensor::<B, 1>::from_floats([1.0 / (1.0 + (-bias[0]).exp())], &Default::default()),
            ranges.reservoir_t0,
            true,
        )
        .into_scalar();
        assert!((t0 * 24.0 - INIT_T0_HOURS).abs() < 1e-3, "T0 at zero weights = {} h", t0 * 24.0);
        assert_eq!(&bias[1..], &[0.0, 0.0], "a = b = 0 at the centre of a symmetric box");
        // Sigmoid well away from saturation.
        let s = 1.0 / (1.0 + (-bias[0]).exp());
        assert!(s > 0.05 && s < 0.95, "sigmoid {s}");
    }

    #[test]
    fn t0_is_log_space_and_stays_in_its_box() {
        let ranges = ParameterRanges::default();
        let mut head = init_release_head::<B>(&section(true), &ranges, 7, &Default::default());
        // Saturate the read-out both ways.
        for (bias, lo_hi) in [(-40.0_f32, 0), (40.0, 1)] {
            head.output.bias = Some(Param::from_tensor(Tensor::from_floats(
                [bias, bias, bias],
                &Default::default(),
            )));
            let p = release_params(&head, features(5), &ranges);
            let t0: Vec<f32> = p.t0_days.into_data().to_vec().unwrap();
            let want = ranges.reservoir_t0[lo_hi];
            for t in t0 {
                assert!((t - want).abs() / want < 1e-3, "saturated T0 {t} vs bound {want}");
            }
        }
        // Mid-sigmoid is the geometric mean of the box.
        let mid = denormalize(
            Tensor::<B, 1>::from_floats([0.5], &Default::default()),
            ranges.reservoir_t0,
            true,
        )
        .into_scalar();
        let geo = (ranges.reservoir_t0[0] * ranges.reservoir_t0[1]).sqrt();
        assert!((mid - geo).abs() / geo < 1e-4, "log-space midpoint {mid} vs {geo}");
    }

    #[test]
    fn non_seasonal_head_emits_only_t0() {
        let ranges = ParameterRanges::default();
        let head = init_release_head::<B>(&section(false), &ranges, 42, &Default::default());
        assert_eq!(head.learnable_parameters(), &["T0".to_string()]);
        let p = release_params(&head, features(3), &ranges);
        assert!(p.seasonal.is_none());
        assert_eq!(p.t0_days.dims(), [3]);
    }

    #[test]
    fn release_seed_differs_from_the_routing_seed() {
        let cfg = release_head_config(&section(true), 42);
        assert_ne!(cfg.seed, 42);
        assert_eq!(cfg.learnable_parameters, release_outputs(true));
    }
}
