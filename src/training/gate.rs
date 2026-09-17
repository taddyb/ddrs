//! Temperature-annealed binary gate for the leakance factor.
//!
//! `zeta = leakance_factor · area_z · K_D · (depth − d_gw)` multiplies
//! `leakance_factor` and `K_D`, so only their product reaches the physics and
//! the two are exactly degenerate (measured `rho = +0.9986` over CONUS on a
//! trained arm). Physically a reach either sits above a losing aquifer or it
//! does not, so `leakance_factor` should be a 0/1 SELECTOR of where leakance
//! acts, not a continuous multiplier. A gate and a conductance are not
//! degenerate with each other.
//!
//! A hard step has zero gradient almost everywhere, so the gate is a
//! sharpening sigmoid applied to the head's NORMALIZED `(0, 1)` output `u`:
//!
//! ```text
//! g = sigmoid( logit(clamp(u, eps, 1 − eps)) / tau ),   logit(x) = ln(x / (1 − x))
//! ```
//!
//! At `tau = 1` this is `g = u` exactly (sigmoid ∘ logit is the identity), and
//! the code special-cases it to return `u` untouched so the identity is
//! BIT-exact rather than round-tripped through `ln`/`exp`. As `tau` falls the
//! transform sharpens toward a step at `u = 0.5`; `tau` is annealed downward
//! over training by `params.leakance_gate.temperature` (`src/config.rs::LeakanceGate`).
//!
//! The transform is differentiable at every `tau > 0`, so autograd through
//! `clamp → log → div → sigmoid` gives the exact gradient; there is no
//! straight-through estimator. It is applied to the head OUTPUT before the
//! value reaches `setup_inputs`, so `TimestepLeakanceOp` and the sparse
//! backward are untouched. Pinned by `tests/leakance_gate.rs`.

use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Clamp margin applied to `u` before the logit.
///
/// The head ends in a sigmoid, so `u` can be exactly `0.0` or `1.0` in f32
/// once the pre-activation passes about ±17 (`sigmoid(17) == 1.0_f32`). An
/// unclamped logit there is `±inf`, and its backward `1/(u(1−u))` is `inf`,
/// which turns the gradient into NaN. `1e-6` is chosen because:
///
/// * `1 − 1e-6` is ~17 ulps below `1.0_f32` (the ulp there is `2^-24 ≈ 6e-8`),
///   so `1 − u` at the clamp is resolved to about 3 % relative accuracy;
///   `1e-7` would sit only 2 ulps away and `1 − u` would carry ~20 % error.
/// * `logit(1e-6) ≈ −13.8`, well inside f32 range, and it only alters values
///   the head has already saturated past `|pre-activation| > 13.8`.
/// * The routing core is f32 throughout (invariant 2); no f64 is introduced.
pub const GATE_EPS: f32 = 1e-6;

/// Apply the temperature-annealed gate to a normalized head output `u`.
///
/// `tau == 1.0` returns `u` untouched (bit-exact identity). Any other
/// positive, finite `tau` evaluates the transform above. Generic over the
/// backend so the same function serves the training (`Autodiff<I>`), eval,
/// probe and `dump_parameters` readers.
pub fn leakance_gate<B: Backend>(u: Tensor<B, 1>, tau: f32) -> Tensor<B, 1> {
    if tau == 1.0 {
        return u;
    }
    debug_assert!(tau.is_finite() && tau > 0.0, "gate temperature must be positive and finite, got {tau}");
    let u = u.clamp(GATE_EPS, 1.0 - GATE_EPS);
    // logit(u) = ln(u) − ln(1 − u)
    let logit = u.clone().log() - u.neg().add_scalar(1.0).log();
    sigmoid(logit.div_scalar(tau))
}
