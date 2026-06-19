//! Daily downsample + training objective with NaN mask.
//!
//! Mirrors `~/projects/ddr/src/ddr/scripts_utils.py::compute_daily_runoff`
//! and `scripts/train.py:62-86` (NaN-filter + warmup trim). The objective
//! is selectable (`config::LossKind`): the historical L1, or a per-gauge
//! `λ_nnse·(1 - NNSE) + λ_kge·(1 - KGE)` composite (`nnse_kge_loss`).

use burn::tensor::{backend::Backend, Tensor};
use ndarray::{s, Array2};

use crate::config::{LossConfig, LossKind};

/// Tau-trim then daily downsample via area-mode adaptive average pooling.
///
/// Mirrors DDR's `~/projects/ddr/src/ddr/io/functions.py:22`:
/// `F.interpolate(data.unsqueeze(1), size=(rho,), mode="area").squeeze(1)`.
///
/// Input shape `(G, T_hours)`. Slicing convention from DDR
/// `compute_daily_runoff`: `[13 + tau : -11 + tau]`. The trimmed length
/// does NOT need to be a multiple of 24 — fractional boundary hours are
/// handled by area-mode pooling.
///
/// Returns `(G, T_days)` where `T_days = T_hours_trimmed // 24` (matching
/// DDR's `num_days` computation at `scripts/train.py:78`).
pub fn tau_trim_and_downsample<B: Backend>(
    predictions_hourly: Tensor<B, 2>,
    tau: u32,
) -> Tensor<B, 2> {
    let dims = predictions_hourly.dims();
    let (g, t_hours) = (dims[0], dims[1]);
    let start = 13 + tau as usize;
    let end = t_hours - 11 + tau as usize;
    assert!(start < end, "tau-trim window degenerate: [{start}, {end})");
    let t_trimmed = end - start;
    let t_days = t_trimmed / 24;
    assert!(
        t_days > 0,
        "trimmed window too short: T_trimmed={t_trimmed}, T_days={t_days}"
    );

    let device = predictions_hourly.device();
    let sliced = predictions_hourly.slice([0..g, start..end]); // (G, L)
    let weights = area_pool_weights::<B>(t_trimmed, t_days, &device); // (M, L)
    // (G, L) @ (L, M) = (G, M)
    sliced.matmul(weights.transpose())
}

/// Construct the area-mode pooling weight matrix `W ∈ R^{M × L}` such that
/// `W[i, j] = overlap(input_cell_j, output_bin_i) / s` where `s = L / M`.
///
/// Each row sums to 1. Mirrors `torch.nn.functional.interpolate(mode="area")`
/// for the 1D case (DDR uses this at `ddr/io/functions.py:22`). The result
/// is a constant matrix that depends only on shape — compute once per
/// (L, M) pair and reuse.
///
/// Sparsity: each row has at most `ceil(L/M) + 1` nonzeros for `s > 1`.
fn area_pool_weights<B: Backend>(
    l: usize,
    m: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(l > 0 && m > 0 && l >= m, "need L >= M > 0; got L={l}, M={m}");
    let s = l as f32 / m as f32;
    let mut data: Vec<f32> = vec![0.0; m * l];

    for i in 0..m {
        let left = (i as f32) * s;
        let right = ((i + 1) as f32) * s;
        let j_lo = left.floor() as usize;
        let j_hi = (right.ceil() as usize).min(l);
        for j in j_lo..j_hi {
            let cell_left = (j as f32).max(left);
            let cell_right = ((j + 1) as f32).min(right);
            let weight = (cell_right - cell_left) / s;
            data[i * l + j] = weight;
        }
    }

    Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(data, [m * l]),
        device,
    )
    .reshape([m, l])
}

pub struct FilteredPair {
    pub predictions: Array2<f32>,  // (T_days, G_kept)
    pub observations: Array2<f32>, // (T_days, G_kept)
    pub mask: Vec<bool>,           // length original G; true = kept
}

/// Filter gauges whose observations contain any NaN in the window.
pub fn filter_nan_gauges(
    daily_predictions: &Array2<f32>, // (G, T_days)
    observations: &Array2<f32>,      // (T_days, G)
) -> FilteredPair {
    let (g, t_days_p) = daily_predictions.dim();
    let (t_days_o, g2) = observations.dim();
    assert_eq!(g, g2);
    assert_eq!(t_days_p, t_days_o);
    let mask: Vec<bool> = (0..g)
        .map(|j| !observations.column(j).iter().any(|v| v.is_nan()))
        .collect();
    let n_kept = mask.iter().filter(|&&v| v).count();
    let mut pred_kept = Array2::<f32>::zeros((t_days_p, n_kept));
    let mut obs_kept = Array2::<f32>::zeros((t_days_o, n_kept));
    let mut col_idx = 0usize;
    for j in 0..g {
        if !mask[j] {
            continue;
        }
        for t in 0..t_days_p {
            pred_kept[(t, col_idx)] = daily_predictions[(j, t)];
        }
        for t in 0..t_days_o {
            obs_kept[(t, col_idx)] = observations[(t, j)];
        }
        col_idx += 1;
    }
    FilteredPair { predictions: pred_kept, observations: obs_kept, mask }
}

/// L1 loss over `(T_days_post_warmup, G_kept)`.
///
/// Mirrors `~/projects/ddr/scripts/train.py:75-85`:
///   1. Drop gauges with any NaN.
///   2. Truncate to `[warmup..]` along the time axis.
///   3. Mean of absolute differences.
pub fn l1_loss_post_warmup(
    predictions: &Array2<f32>,
    observations: &Array2<f32>,
    warmup: usize,
) -> f32 {
    let (t_days, _g) = predictions.dim();
    assert!(warmup < t_days, "warmup={warmup} >= T_days={t_days}");
    let p = predictions.slice(s![warmup.., ..]);
    let o = observations.slice(s![warmup.., ..]);
    let diff = &p - &o;
    diff.iter().map(|v| v.abs()).sum::<f32>() / (diff.len() as f32)
}

/// Dispatch a mini-batch to the configured training objective.
///
/// `p` / `o` are `(G, T_post_warmup)` with autograd alive on `p`. `o` must
/// be NaN-free (the driver drops gauges with any NaN in the window before
/// calling). Returns the scalar batch loss with the autograd graph intact.
pub fn batch_loss<B: Backend>(
    p: Tensor<B, 2>,
    o: Tensor<B, 2>,
    cfg: &LossConfig,
) -> Tensor<B, 1> {
    match cfg.kind {
        LossKind::L1 => (p - o).abs().mean(),
        LossKind::NnseKge => nnse_kge_loss(p, o, cfg.nnse_weight, cfg.kge_weight, cfg.eps),
        LossKind::Kge => kge_component_loss(
            p,
            o,
            cfg.r_weight,
            cfg.alpha_weight,
            cfg.beta_weight,
            cfg.nnse_weight,
            cfg.kge_clamp,
            cfg.eps,
        ),
    }
}

/// Per-gauge `λ_nnse·(1 - NNSE) + λ_kge·(1 - KGE)`, averaged over gauges.
///
/// Both metrics are computed per gauge along the time axis, then averaged —
/// so large basins don't dominate. All moments use the population form
/// (divide by `T`); the `r` and `α` ratios are invariant to that choice.
///
/// Why this exists: L1 and NSE are both maximized at a simulated variance
/// *below* the observed (NSE's optimum sits at `α = r < 1`), so they reward
/// the Muskingum-Cunge routing for over-attenuating flood peaks. KGE's
/// `(α - 1)²` term, with `α = σ_sim/σ_obs`, supplies the missing restoring
/// gradient; NNSE guards correlation and volume. See the loss-decomposition
/// analysis in the 2026-06 KGE-regression investigation.
///
/// `eps` stabilizes the variance/mean denominators (DDR `hydrograph_loss`
/// uses `0.1`) so a near-constant gauge can't produce a NaN gradient.
///
/// KGE = 1 - √((r-1)² + (α-1)² + (β-1)²), so `1 - KGE` is exactly that
/// Euclidean distance. NNSE = 1/(2 - NSE) ∈ (0, 1], `1 - NNSE` its loss.
pub fn nnse_kge_loss<B: Backend>(
    p: Tensor<B, 2>, // (G, T), autograd alive
    o: Tensor<B, 2>, // (G, T), NaN-free
    nnse_weight: f32,
    kge_weight: f32,
    eps: f32,
) -> Tensor<B, 1> {
    // Per-gauge means, kept as (G, 1) for broadcasting back over time.
    let mean_p = p.clone().mean_dim(1);
    let mean_o = o.clone().mean_dim(1);

    // Centered series and the raw residual (for NSE's SSE).
    let pc = p.clone() - mean_p.clone(); // (G, T)
    let oc = o.clone() - mean_o.clone();
    let resid = p - o; // (G, T); consumes p, o (last use)

    // Second moments (population) along time → (G, 1).
    let var_p = (pc.clone() * pc.clone()).mean_dim(1);
    let var_o = (oc.clone() * oc.clone()).mean_dim(1);
    let std_p = var_p.add_scalar(eps).sqrt();
    let std_o = var_o.add_scalar(eps).sqrt();
    let cov = (pc * oc.clone()).mean_dim(1);

    // KGE components and `1 - KGE` = Euclidean distance of (r, α, β) from 1.
    let r = cov / (std_p.clone() * std_o.clone());
    let alpha = std_p / std_o;
    let beta = mean_p / mean_o.add_scalar(eps);
    let dr = r.sub_scalar(1.0);
    let da = alpha.sub_scalar(1.0);
    let db = beta.sub_scalar(1.0);
    let one_minus_kge = (dr.clone() * dr + da.clone() * da + db.clone() * db).sqrt();
    let kge_loss = one_minus_kge.mean();

    // NSE = 1 - SSE/SSO; NNSE = 1/(2 - NSE); loss = 1 - NNSE.
    let sse = (resid.clone() * resid).sum_dim(1); // (G, 1)
    let sso = (oc.clone() * oc).sum_dim(1).add_scalar(eps);
    let nse = (sse / sso).neg().add_scalar(1.0); // 1 - sse/sso
    let nnse = nse.neg().add_scalar(2.0).recip(); // 1/(2 - nse)
    let nnse_loss = nnse.neg().add_scalar(1.0).mean(); // 1 - nnse

    nnse_loss.mul_scalar(nnse_weight) + kge_loss.mul_scalar(kge_weight)
}

/// Component-weighted KGE loss:
/// per gauge `r_w·(r-1)² + α_w·(α-1)² + β_w·(β-1)² + nnse_w·(1-NNSE)`, averaged.
///
/// Unlike [`nnse_kge_loss`] (which sums the KGE components under one square
/// root, `1-KGE = √((r-1)²+(α-1)²+(β-1)²)`), this weights each squared
/// component independently. Two reasons:
///
/// 1. **No gradient singularity.** `√(·)` has an infinite-slope cusp as the
///    prediction approaches perfect KGE (the argument → 0); the squared form
///    is smooth there, so late-training gradients stay well-behaved.
/// 2. **Tunable restoring force.** `α_w` independently up-weights the
///    `(α-1)²` variance-ratio term — the direct counter-pressure to MC
///    over-attenuation (the diagnosed `α: 0.96 → 0.85` regression). Set
///    `α_w > 1` to prioritize restoring `σ_sim/σ_obs → 1`.
///
/// `nnse_w` keeps the optional NNSE guard (correlation + volume); set it to 0
/// for a pure component-weighted KGE objective. All moments use the population
/// form along time; `eps` stabilizes the variance/mean denominators exactly as
/// in [`nnse_kge_loss`].
#[allow(clippy::too_many_arguments)]
pub fn kge_component_loss<B: Backend>(
    p: Tensor<B, 2>, // (G, T), autograd alive
    o: Tensor<B, 2>, // (G, T), NaN-free
    r_weight: f32,
    alpha_weight: f32,
    beta_weight: f32,
    nnse_weight: f32,
    clamp: f32,
    eps: f32,
) -> Tensor<B, 1> {
    // Per-gauge means, kept as (G, 1) for broadcasting back over time.
    let mean_p = p.clone().mean_dim(1);
    let mean_o = o.clone().mean_dim(1);

    // Centered series and the raw residual (for NSE's SSE).
    let pc = p.clone() - mean_p.clone(); // (G, T)
    let oc = o.clone() - mean_o.clone();
    let resid = p - o; // (G, T); consumes p, o (last use)

    // Second moments (population) along time → (G, 1).
    let var_p = (pc.clone() * pc.clone()).mean_dim(1);
    let var_o = (oc.clone() * oc.clone()).mean_dim(1);
    let std_p = var_p.add_scalar(eps).sqrt();
    let std_o = var_o.add_scalar(eps).sqrt();
    let cov = (pc * oc.clone()).mean_dim(1);

    // KGE components: r (correlation), α (variance ratio), β (mean ratio).
    let r = cov / (std_p.clone() * std_o.clone());
    let alpha = std_p / std_o;
    let beta = mean_p / mean_o.add_scalar(eps);
    let dr = r.sub_scalar(1.0);
    let da = alpha.sub_scalar(1.0);
    let db = beta.sub_scalar(1.0);

    // Weighted sum of squared component deviations, clamped per gauge so a
    // collapsed-variance gauge can't hijack the batch gradient, then averaged.
    let kge_term = (dr.clone() * dr).mul_scalar(r_weight)
        + (da.clone() * da).mul_scalar(alpha_weight)
        + (db.clone() * db).mul_scalar(beta_weight);
    let kge_loss = kge_term.clamp_max(clamp).mean();

    if nnse_weight == 0.0 {
        return kge_loss;
    }

    // NSE = 1 - SSE/SSO; NNSE = 1/(2 - NSE); loss = 1 - NNSE.
    let sse = (resid.clone() * resid).sum_dim(1); // (G, 1)
    let sso = (oc.clone() * oc).sum_dim(1).add_scalar(eps);
    let nse = (sse / sso).neg().add_scalar(1.0); // 1 - sse/sso
    let nnse = nse.neg().add_scalar(2.0).recip(); // 1/(2 - nse)
    let nnse_loss = nnse.neg().add_scalar(1.0).mean(); // 1 - nnse

    kge_loss + nnse_loss.mul_scalar(nnse_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::{Tensor, TensorData};
    type Bp = NdArray<f32>;
    type Ad = Autodiff<NdArray<f32>>;

    fn mk<B: Backend>(rows: &[[f32; 4]]) -> Tensor<B, 2> {
        let g = rows.len();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::<B, 1>::from_data(TensorData::new(flat, [g * 4]), &Default::default())
            .reshape([g, 4])
    }

    #[test]
    fn nnse_kge_loss_matches_hand_computation() {
        // pred = obs centered ×0.5 + mean → α=0.5, r=1, β=1 (one gauge).
        // obs:  [1,3,1,3] (mean 2, σ²=1);  pred: [1.5,2.5,1.5,2.5] (σ²=0.25)
        let p = mk::<Bp>(&[[1.5, 2.5, 1.5, 2.5]]);
        let o = mk::<Bp>(&[[1.0, 3.0, 1.0, 3.0]]);
        // eps=0 to check exact metric algebra.
        let loss = nnse_kge_loss(p, o, 1.0, 1.0, 0.0);
        let v: f32 = loss.into_scalar();
        // 1-KGE = √((1-1)²+(0.5-1)²+(1-1)²) = 0.5
        // NSE = 1 - SSE/SSO = 1 - 1/4 = 0.75; NNSE = 1/1.25 = 0.8; 1-NNSE = 0.2
        assert!((v - (0.2 + 0.5)).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn kge_term_prefers_unattenuated_amplitude() {
        // Pure KGE term (nnse_weight=0): an attenuated prediction must score
        // a HIGHER loss than the perfect one.
        let o = mk::<Bp>(&[[1.0, 3.0, 1.0, 3.0]]);
        let perfect = mk::<Bp>(&[[1.0, 3.0, 1.0, 3.0]]);
        let attenuated = mk::<Bp>(&[[1.5, 2.5, 1.5, 2.5]]);
        let l_perfect: f32 =
            nnse_kge_loss(perfect, o.clone(), 0.0, 1.0, 0.0).into_scalar();
        let l_atten: f32 =
            nnse_kge_loss(attenuated, o, 0.0, 1.0, 0.0).into_scalar();
        assert!(l_perfect < 1e-5, "perfect KGE loss should be ~0, got {l_perfect}");
        assert!(l_atten > l_perfect, "attenuated {l_atten} !> perfect {l_perfect}");
    }

    #[test]
    fn kge_gradient_points_toward_de_attenuation() {
        // With an attenuated, perfectly-correlated prediction, the gradient
        // must push peaks UP and troughs DOWN (restore amplitude):
        //   ∂loss/∂p < 0 at peak timesteps, > 0 at trough timesteps.
        let p = mk::<Ad>(&[[1.5, 2.5, 1.5, 2.5]]).require_grad();
        let o = mk::<Ad>(&[[1.0, 3.0, 1.0, 3.0]]);
        let loss = nnse_kge_loss(p.clone(), o, 0.0, 1.0, 0.1);
        let grads = loss.backward();
        let g = p.grad(&grads).unwrap();
        let gv: Vec<f32> = g.into_data().to_vec().unwrap();
        // indices 1,3 are peaks (pred 2.5 < obs 3); 0,2 are troughs.
        assert!(gv[1] < 0.0 && gv[3] < 0.0, "peak grads not negative: {gv:?}");
        assert!(gv[0] > 0.0 && gv[2] > 0.0, "trough grads not positive: {gv:?}");
    }

    #[test]
    fn kge_component_loss_matches_hand_computation() {
        // Same fixture as nnse_kge: pred = obs centered ×0.5 + mean
        //   → α=0.5, r=1, β=1 (one gauge). obs:[1,3,1,3] pred:[1.5,2.5,1.5,2.5].
        let p = mk::<Bp>(&[[1.5, 2.5, 1.5, 2.5]]);
        let o = mk::<Bp>(&[[1.0, 3.0, 1.0, 3.0]]);
        // Pure KGE components (nnse_weight=0), eps=0, default unit weights.
        // r_w·(0)² + α_w·(0.5-1)² + β_w·(0)² = 0.25.
        let v: f32 = kge_component_loss(p, o, 1.0, 1.0, 1.0, 0.0, 1e9, 0.0).into_scalar();
        assert!((v - 0.25).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn kge_component_alpha_weight_scales_attenuation_penalty() {
        // Doubling α_w must exactly double the loss for an α-only error.
        let p = mk::<Bp>(&[[1.5, 2.5, 1.5, 2.5]]);
        let o = mk::<Bp>(&[[1.0, 3.0, 1.0, 3.0]]);
        let l1: f32 = kge_component_loss(p.clone(), o.clone(), 1.0, 1.0, 1.0, 0.0, 1e9, 0.0).into_scalar();
        let l2: f32 = kge_component_loss(p, o, 1.0, 2.0, 1.0, 0.0, 1e9, 0.0).into_scalar();
        assert!((l2 - 2.0 * l1).abs() < 1e-5, "α_w=2 gave {l2}, expected 2×{l1}");
    }

    #[test]
    fn kge_component_clamp_bounds_collapsed_variance_gauge() {
        // Gauge 2 has near-constant obs (var_o≈0) → without the clamp the α
        // term explodes; clamp at 5.0 must bound its contribution. Two gauges:
        // gauge 0 is well-posed (small loss), gauge 1 is the pathological one.
        let p = mk::<Bp>(&[[1.5, 2.5, 1.5, 2.5], [10.0, 90.0, 10.0, 90.0]]);
        let o = mk::<Bp>(&[[1.0, 3.0, 1.0, 3.0], [1.0, 1.0, 1.0, 1.0001]]);
        // eps small so the collapsed denominator really does blow up unclamped.
        let unclamped: f32 =
            kge_component_loss(p.clone(), o.clone(), 1.0, 2.0, 1.0, 0.0, 1e9, 1e-6).into_scalar();
        let clamped: f32 =
            kge_component_loss(p, o, 1.0, 2.0, 1.0, 0.0, 5.0, 1e-6).into_scalar();
        assert!(unclamped > 100.0, "expected blowup without clamp, got {unclamped}");
        // Mean of two gauges, each ≤ 5.0 after clamp → batch ≤ 5.0.
        assert!(clamped <= 5.0 + 1e-4, "clamp did not bound the loss: {clamped}");
    }

    #[test]
    fn kge_component_gradient_points_toward_de_attenuation() {
        // With an attenuated, perfectly-correlated prediction, the α term's
        // gradient must push peaks UP and troughs DOWN (restore amplitude).
        let p = mk::<Ad>(&[[1.5, 2.5, 1.5, 2.5]]).require_grad();
        let o = mk::<Ad>(&[[1.0, 3.0, 1.0, 3.0]]);
        let loss = kge_component_loss(p.clone(), o, 1.0, 1.0, 1.0, 0.0, 1e9, 0.1);
        let grads = loss.backward();
        let g = p.grad(&grads).unwrap();
        let gv: Vec<f32> = g.into_data().to_vec().unwrap();
        // indices 1,3 are peaks (pred 2.5 < obs 3); 0,2 are troughs.
        assert!(gv[1] < 0.0 && gv[3] < 0.0, "peak grads not negative: {gv:?}");
        assert!(gv[0] > 0.0 && gv[2] > 0.0, "trough grads not positive: {gv:?}");
    }

    #[test]
    fn l1_loss_post_warmup_basic() {
        let pred = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]]; // (T=3, G=2)
        let obs = array![[1.0_f32, 2.0], [4.0, 4.0], [5.0, 7.0]];
        // warmup=0: 6 entries, sum of |diff| = 0+0+1+0+0+1 = 2; mean = 2/6.
        let l = l1_loss_post_warmup(&pred, &obs, 0);
        assert!((l - 2.0 / 6.0).abs() < 1e-6);
        // warmup=1: 4 entries, sum = 1+0+0+1 = 2; mean = 0.5.
        let l = l1_loss_post_warmup(&pred, &obs, 1);
        assert!((l - 0.5).abs() < 1e-6);
    }

    #[test]
    fn filter_nan_gauges_drops_columns() {
        let pred = array![[1.0_f32, 1.5], [2.0, 2.5], [3.0, 3.5]]; // (G=3, T=2)
        let obs = array![[10.0_f32, f32::NAN, 30.0], [11.0, 21.0, 31.0]]; // (T=2, G=3)
        let f = filter_nan_gauges(&pred, &obs);
        assert_eq!(f.mask, vec![true, false, true]);
        assert_eq!(f.predictions.shape(), &[2, 2]);
        assert_eq!(f.observations.shape(), &[2, 2]);
    }

    #[test]
    fn area_pool_weights_rows_sum_to_one() {
        let device = Default::default();
        let w = area_pool_weights::<Bp>(2139, 89, &device);
        let row_sums: Tensor<Bp, 1> = w.sum_dim(1).squeeze();
        for v in row_sums.into_data().to_vec::<f32>().unwrap() {
            assert!((v - 1.0).abs() < 1e-5, "row sum {v} != 1");
        }
    }

    #[test]
    fn area_pool_matches_block_mean_when_divisible() {
        let device = Default::default();
        // Input: 1..=48 over 48 hours, single gauge.
        let v: Vec<f32> = (1..=48).map(|x| x as f32).collect();
        let input: Tensor<Bp, 2> = Tensor::<Bp, 1>::from_data(
            burn::tensor::TensorData::new(v, [48]),
            &device,
        )
        .reshape([1, 48]);

        let w = area_pool_weights::<Bp>(48, 2, &device);
        let out: Tensor<Bp, 2> = input.matmul(w.transpose());
        let got: Vec<f32> = out.into_data().to_vec().unwrap();
        // Block 1 = mean(1..=24)  = 12.5
        // Block 2 = mean(25..=48) = 36.5
        assert!((got[0] - 12.5).abs() < 1e-5, "got {}", got[0]);
        assert!((got[1] - 36.5).abs() < 1e-5, "got {}", got[1]);
    }

    #[test]
    fn area_pool_handles_non_divisible_input() {
        let device = Default::default();
        let w = area_pool_weights::<Bp>(2139, 89, &device);
        let data: Vec<f32> = w.into_data().to_vec().unwrap();

        // Row 0 covers input range [0, 24.0337...). Cells 0-23 contribute
        // their full weight 1/s, cell 24 contributes the fractional piece.
        let s = 2139.0_f32 / 89.0;
        for j in 0..24 {
            let expected = 1.0 / s;
            assert!(
                (data[j] - expected).abs() < 1e-6,
                "row 0 col {j}: got {} want {expected}",
                data[j]
            );
        }
        let frac = (s - 24.0) / s;
        assert!(
            (data[24] - frac).abs() < 1e-4,
            "row 0 col 24: got {} want ~{frac}",
            data[24]
        );
        for j in 25..2139 {
            assert!(data[j].abs() < 1e-6, "row 0 col {j} should be 0; got {}", data[j]);
        }
    }

    #[test]
    fn n_gauges_one_does_not_panic() {
        let device = Default::default();
        // 2160 hourly input → 89 daily output for tau=3.
        let input: Tensor<Bp, 2> = Tensor::zeros([1, 2160], &device);
        let out = tau_trim_and_downsample(input, 3);
        assert_eq!(out.dims(), [1, 89]);
    }

    #[test]
    fn tau_trim_matches_old_block_mean_on_divisible_input() {
        // Verify the new area-pool body reduces to block-mean whenever
        // the trimmed window IS a multiple of 24. tau=11, T=72 →
        // trimmed window is hours [24..72] (length 48 = 2 days exactly).
        let device = Default::default();
        let v: Vec<f32> = (0..72).map(|x| x as f32).collect();
        let input: Tensor<Bp, 2> = Tensor::<Bp, 1>::from_data(
            burn::tensor::TensorData::new(v, [72]),
            &device,
        )
        .reshape([1, 72]);
        let out = tau_trim_and_downsample(input, 11);
        let got: Vec<f32> = out.into_data().to_vec().unwrap();
        // Sliced = hours 24..72 (48 values: 24..=71).
        // Day 1 = mean(24..=47) = 35.5
        // Day 2 = mean(48..=71) = 59.5
        assert!((got[0] - 35.5).abs() < 1e-4, "got {}", got[0]);
        assert!((got[1] - 59.5).abs() < 1e-4, "got {}", got[1]);
    }
}
