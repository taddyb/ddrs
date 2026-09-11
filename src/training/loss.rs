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
/// Pooling mirrors DDR's `~/projects/ddr/src/ddr/io/functions.py:22`:
/// `F.interpolate(data.unsqueeze(1), size=(rho,), mode="area").squeeze(1)`.
///
/// Input shape `(G, T_hours)`. Slicing convention (since 2026-08-08):
/// `[tau : -(24 - tau)]` — pooled day `i` covers hours
/// `[tau + 24i, tau + 24(i+1))` and is scored against OBSERVATION DAY `i`.
/// `tau` is the number of hours the routed output is advanced before
/// scoring (a translation-only inverse-routing shift, same sign and
/// magnitude as dMC-Juniata's tau): `tau = 0` is exactly day-aligned,
/// `tau = 9` pairs obs day `i` with routed hours `[24i + 9, 24i + 33)`.
///
/// Legacy mapping: the pre-2026-08-08 slice was `[13+tau : -11+tau]` with
/// pooled day `i` scored against obs day `i+1`; `tau_new = tau_old - 11`
/// (so old shipped 3 ≡ new −8, old optimum 20 ≡ new 9). DDR-Python's
/// `compute_daily_runoff` still uses the legacy form. The total trim is
/// 24 h under both conventions, so `T_days = T_hours/24 - 1` is unchanged.
///
/// Why the last day is excluded: a window advanced by `tau` needs `tau`
/// hours of routed output past the final store day, which were never
/// routed (the q' feeding them is outside the window). Dropping exactly
/// one day — front trim `tau` + back trim `24 - tau` — makes the SAME
/// `T_days` days part of the training/testing set at every tau: tensor
/// shapes, the obs pairing, and NSE(tau) sweeps all score an identical
/// day sample regardless of the shift, and the trimmed length stays a
/// multiple of 24 so pooling is an exact block mean. The alternative
/// (score all days by routing one extra day of q' per window) is ~1% more
/// training signal and needs +1-day window plumbing; deliberately not
/// done — see the 2026-08-09 discussion in
/// `docs/2026-08-06-tau-sweep-pilot-findings.md` §5i.
///
/// Returns `(G, T_days)` where `T_days = T_hours_trimmed // 24`.
pub fn tau_trim_and_downsample<B: Backend>(
    predictions_hourly: Tensor<B, 2>,
    tau: u32,
) -> Tensor<B, 2> {
    let dims = predictions_hourly.dims();
    let (g, t_hours) = (dims[0], dims[1]);
    assert!(tau < 24, "tau must be in [0, 24) hours; got {tau}");
    let start = tau as usize;
    let end = t_hours - (24 - tau as usize);
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

/// The pooled-mean denominator of [`batch_loss`] for a micro-batch of
/// `g_kept` surviving gauges × `t_post` post-warmup days.
///
/// This is the exact-recombination weight for gradient accumulation: a
/// micro-batch loss `L_i` (a mean over this count) scaled by `n_i` turns
/// back into a SUM, so `Σ_i (L_i·n_i) / Σ_i n_i` equals the loss of the
/// pooled batch — and, by linearity of the gradient, the accumulated
/// scaled gradients equal the true large-batch gradient. A naive 1/N
/// average is wrong whenever micro-batches differ in surviving-gauge
/// count (the NaN filter makes that the common case).
///
/// L1 averages over ELEMENTS (`(p - o).abs().mean()`); the composite
/// objectives compute per-gauge scores then average over GAUGES.
pub fn loss_denominator(cfg: &LossConfig, g_kept: usize, t_post: usize) -> usize {
    match cfg.kind {
        // `nse-batch-deriv` shares `nse-batch`'s denominator: its level term
        // IS that mean, and its derivative term is a mean over the
        // `g_kept * (t_post - 1)` adjacent pairs, which scales with the same
        // count. Recombination is therefore exact to `O(1/t_post)` rather
        // than exact, so prefer single-batch training with this objective if
        // micro-batches differ in surviving-gauge count.
        LossKind::L1 | LossKind::NseBatch | LossKind::NseBatchDeriv => g_kept * t_post,
        LossKind::NnseKge | LossKind::Kge => g_kept,
    }
}

/// Dispatch a mini-batch to the configured training objective.
///
/// `p` / `o` are `(G, T_post_warmup)` with autograd alive on `p`. `o` must
/// be NaN-free (the driver drops gauges with any NaN in the window before
/// calling). Returns the scalar batch loss with the autograd graph intact.
/// `gauge_std` is the per-gauge observed-discharge standard deviation over the
/// TRAINING period, shape `(G,)`, aligned to `p`'s rows. Required by
/// [`LossKind::NseBatch`] and [`LossKind::NseBatchDeriv`]; ignored by every
/// other objective (pass `None`). `gauge_diff_std` is its consecutive-day
/// DIFFERENCE counterpart, required by [`LossKind::NseBatchDeriv`] alone.
pub fn batch_loss<B: Backend>(
    p: Tensor<B, 2>,
    o: Tensor<B, 2>,
    cfg: &LossConfig,
    gauge_std: Option<Tensor<B, 1>>,
    gauge_diff_std: Option<Tensor<B, 1>>,
) -> Tensor<B, 1> {
    match cfg.kind {
        LossKind::L1 => (p - o).abs().mean(),
        LossKind::NseBatch => nse_batch_loss(
            p,
            o,
            gauge_std.expect(
                "loss.kind: nse-batch requires per-gauge std — the driver must pass \
                 RoutingBatch::gauge_obs_std for the surviving gauges",
            ),
            cfg.eps,
        ),
        LossKind::NseBatchDeriv => nse_batch_deriv_loss(
            p,
            o,
            gauge_std.expect(
                "loss.kind: nse-batch-deriv requires per-gauge std — the driver must pass \
                 RoutingBatch::gauge_obs_std for the surviving gauges",
            ),
            gauge_diff_std.expect(
                "loss.kind: nse-batch-deriv requires per-gauge observed-DIFFERENCE std — \
                 the driver must pass RoutingBatch::gauge_obs_diff_std for the surviving gauges",
            ),
            cfg.deriv_weight,
            cfg.eps,
        ),
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

/// dHBV's batch-NSE loss (hydroDL `crit.py::NSELossBatch`, Frederick 2019).
///
/// `mean over (gauge, day) of (p − o)² / (σ_gauge + eps)²` — a squared-error
/// loss with each gauge's residuals normalized by that gauge's observed
/// standard deviation over the **training period** (a fixed vector, NOT
/// recomputed per window; recomputing per window would make the objective
/// drift between micro-batches and break accumulation exactness).
///
/// Why: raw L1/L2 weight every m³/s equally, so a handful of large,
/// high-variance basins dominate the batch gradient and the head optimizes
/// them at the expense of everything else. Dividing by σ puts all gauges on a
/// comparable scale — the same normalization NSE itself uses, hence the name.
/// `eps` (dHBV default 0.1) keeps a near-constant gauge from exploding.
///
/// `p` / `o` are `(G, T)` with autograd alive on `p` and `o` NaN-free (the
/// driver's NaN-gauge filter runs first); `sigma` is `(G,)`.
pub fn nse_batch_loss<B: Backend>(
    p: Tensor<B, 2>,
    o: Tensor<B, 2>,
    sigma: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 1> {
    let g = p.dims()[0];
    let denom = sigma.add_scalar(eps).reshape([g, 1]); // (G, 1), broadcasts over time
    let resid = p - o;
    let sq = resid.clone() * resid;
    (sq / (denom.clone() * denom)).mean()
}

/// [`nse_batch_loss`] plus `deriv_weight` times the same quantity evaluated on
/// CONSECUTIVE-DAY DIFFERENCES:
///
/// ```text
/// L = L_nse_batch + lambda * mean over valid adjacent (day, gauge) pairs of
///       ((sim_{t+1} - sim_t) - (obs_{t+1} - obs_t))^2 / (sigma_d_gauge + eps)^2
/// ```
///
/// Why: the landscape study's curvature probe found the loss curvature in
/// Manning's `n` to be governed by the mean square of the hydrograph's TIME
/// DERIVATIVE (findings §25), so a level-only objective leaves the `n` valley
/// nearly flat and the aggregate gradient vanishes long before `n` is
/// identified. Scoring the derivative directly deepens that valley (§26
/// predicted ~3.7x at `lambda = 0.5`).
///
/// This mirrors the landscape MEASUREMENT objective
/// `experiment::landscape::objective::deriv_window_loss` term for term — same
/// adjacent-pair rule, same population `sigma_d`, same dense masking — so a
/// model trained here is optimizing exactly what that instrument plots.
///
/// `sigma` / `sigma_d` are `(G,)`, fixed over the training period (see
/// `MeritGagesDataset::gauge_obs_std` / `gauge_obs_diff_std`); `p` / `o` are
/// `(G, T)` with autograd alive on `p`. At `deriv_weight == 0.0` this is
/// numerically identical to [`nse_batch_loss`] — the derivative term is
/// always finite (never NaN/Inf), so `x + 0.0 * term == x` exactly.
pub fn nse_batch_deriv_loss<B: Backend>(
    p: Tensor<B, 2>,
    o: Tensor<B, 2>,
    sigma: Tensor<B, 1>,
    sigma_d: Tensor<B, 1>,
    deriv_weight: f32,
    eps: f32,
) -> Tensor<B, 1> {
    let level = nse_batch_loss(p.clone(), o.clone(), sigma, eps);
    let deriv = obs_diff_term(p, o, sigma_d, eps);
    level + deriv.mul_scalar(deriv_weight)
}

/// The derivative term of [`nse_batch_deriv_loss`], alone:
/// `sum over adjacent valid pairs (g, j) of w * (dsim - dobs)^2 /
/// (sigma_d_g + eps)^2` with `w = 1 / n_pairs_total` — i.e. one global mean
/// over every valid pair in the batch, exactly as [`nse_batch_loss`]'s
/// `.mean()` is one global mean over every valid (day, gauge).
///
/// A pair `(j, j+1)` counts for gauge `g` only when BOTH days are valid for
/// that gauge, under the same rule `nse_batch_loss` relies on (the driver's
/// per-gauge NaN filter). A gap therefore BREAKS the chain: `N` valid days
/// with a hole in the middle give strictly fewer than `N - 1` pairs. Mirrors
/// `landscape::objective::adjacent_valid_pairs`.
///
/// Masking is dense — a `(G, T-1)` weight tensor that is zero at invalid
/// pairs, and an observed-difference tensor that is zero there too — so no
/// gather is needed, the term stays differentiable in `p`, and a NaN
/// observation can never reach the arithmetic. Returns an exact zero when the
/// batch has no valid pair at all (`T < 2`, or every pair broken), so the
/// composite degrades to the level term instead of to NaN.
///
/// A gauge with constant observations has `sigma_d = 0`; as in
/// `nse_batch_loss` the `+ eps` is what keeps it finite (dHBV's convention,
/// and `gauge_obs_diff_std` returns `0.0` for degenerate gauges on purpose).
fn obs_diff_term<B: Backend>(
    p: Tensor<B, 2>,
    o: Tensor<B, 2>,
    sigma_d: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 1> {
    let dims = p.dims();
    let (g, t) = (dims[0], dims[1]);
    let device = p.device();
    if t < 2 {
        return Tensor::<B, 1>::zeros([1], &device);
    }
    let n_pair = t - 1;
    // Observations carry no autograd, so reading them host-side to build the
    // mask is free of graph consequences (same trick the driver's NaN filter
    // and `landscape::objective` use).
    let ov: Vec<f32> = o.into_data().into_vec().unwrap(); // row-major (G, T)
    let mut wts = vec![0.0f32; g * n_pair];
    let mut dobs = vec![0.0f32; g * n_pair];
    let mut n_valid = 0usize;
    for gi in 0..g {
        for j in 0..n_pair {
            let a = ov[gi * t + j];
            let b = ov[gi * t + j + 1];
            if a.is_finite() && b.is_finite() {
                wts[gi * n_pair + j] = 1.0;
                dobs[gi * n_pair + j] = b - a;
                n_valid += 1;
            }
        }
    }
    if n_valid == 0 {
        return Tensor::<B, 1>::zeros([1], &device);
    }
    let scale = 1.0 / n_valid as f32;
    for w in wts.iter_mut() {
        *w *= scale;
    }
    let w_t = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(wts, [g * n_pair]),
        &device,
    )
    .reshape([g, n_pair]);
    let dobs_t = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(dobs, [g * n_pair]),
        &device,
    )
    .reshape([g, n_pair]);
    let dsim = p.clone().slice([0..g, 1..t]) - p.slice([0..g, 0..n_pair]);
    let resid = dsim - dobs_t;
    let denom = sigma_d.add_scalar(eps).reshape([g, 1]); // (G, 1), broadcasts over pairs
    ((resid.clone() * resid) * w_t / (denom.clone() * denom)).sum()
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
    fn nse_batch_loss_matches_hand_computation() {
        // dHBV NSELossBatch (hydroDL crit.py:102-122):
        //   mean over valid (day, gauge) of (sim - obs)^2 / (sigma_gauge + eps)^2
        // Two gauges, 4 days. sigma = [1.0, 2.0], eps = 0.1.
        //   gauge 0 residuals: [1, -1, 1, -1] -> sq 1 each; /(1.1^2)=0.826446 each
        //   gauge 1 residuals: [2, 2, 2, 2]   -> sq 4 each; /(2.1^2)=0.907029 each
        //   mean over 8 elements = (4*0.826446 + 4*0.907029)/8 = 0.866738
        let p = mk::<Bp>(&[[2.0, 0.0, 2.0, 0.0], [5.0, 5.0, 5.0, 5.0]]);
        let o = mk::<Bp>(&[[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]]);
        let sigma = Tensor::<Bp, 1>::from_data(
            burn::tensor::TensorData::new(vec![1.0_f32, 2.0], [2]),
            &Default::default(),
        );
        let v: f32 = nse_batch_loss(p, o, sigma, 0.1).into_scalar();
        let want = (4.0 * 1.0 / 1.1_f32.powi(2) + 4.0 * 4.0 / 2.1_f32.powi(2)) / 8.0;
        assert!((v - want).abs() < 1e-5, "got {v}, want {want}");
    }

    #[test]
    fn nse_batch_loss_downweights_high_variance_gauges() {
        // The whole point vs L1: the same absolute residual costs LESS at a
        // flashy (high-sigma) gauge than at a steady one, so big basins stop
        // dominating the batch gradient.
        let p = mk::<Bp>(&[[2.0, 2.0, 2.0, 2.0]]);
        let o = mk::<Bp>(&[[1.0, 1.0, 1.0, 1.0]]);
        let dev = Default::default();
        let lo = Tensor::<Bp, 1>::from_data(burn::tensor::TensorData::new(vec![0.5_f32], [1]), &dev);
        let hi = Tensor::<Bp, 1>::from_data(burn::tensor::TensorData::new(vec![50.0_f32], [1]), &dev);
        let l_lo: f32 = nse_batch_loss(p.clone(), o.clone(), lo, 0.1).into_scalar();
        let l_hi: f32 = nse_batch_loss(p, o, hi, 0.1).into_scalar();
        assert!(l_hi < l_lo * 0.01, "high-sigma loss {l_hi} not << low-sigma {l_lo}");
    }

    #[test]
    fn nse_batch_gradient_points_toward_observations() {
        let p = mk::<Ad>(&[[2.0, 0.0, 2.0, 0.0]]).require_grad();
        let o = mk::<Ad>(&[[1.0, 1.0, 1.0, 1.0]]);
        let sigma = Tensor::<Ad, 1>::from_data(
            burn::tensor::TensorData::new(vec![1.0_f32], [1]),
            &Default::default(),
        );
        let loss = nse_batch_loss(p.clone(), o, sigma, 0.1);
        let grads = loss.backward();
        let gv: Vec<f32> = p.grad(&grads).unwrap().into_data().to_vec().unwrap();
        // over-prediction -> positive grad (push down); under -> negative.
        assert!(gv[0] > 0.0 && gv[2] > 0.0, "over-pred grads not positive: {gv:?}");
        assert!(gv[1] < 0.0 && gv[3] < 0.0, "under-pred grads not negative: {gv:?}");
    }

    /// `(G, T)` from ragged rows (unlike `mk`, which is fixed at 4 days).
    fn mk_n<B: Backend>(rows: &[&[f32]]) -> Tensor<B, 2> {
        let g = rows.len();
        let t = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::<B, 1>::from_data(TensorData::new(flat, [g * t]), &Default::default())
            .reshape([g, t])
    }

    fn sig<B: Backend>(v: &[f32]) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_data(TensorData::new(v.to_vec(), [v.len()]), &Default::default())
    }

    /// CRITICAL GUARD: `deriv_weight = 0.0` must make `nse-batch-deriv`
    /// numerically IDENTICAL to `nse-batch` — not merely close. This is what
    /// lets an existing `nse-batch` result be reproduced bit-for-bit under
    /// the new kind, and it only holds because `obs_diff_term` is always
    /// finite (`0.0 * NaN` would be NaN).
    #[test]
    fn nse_batch_deriv_at_zero_weight_is_identical_to_nse_batch() {
        let p = mk::<Bp>(&[[2.0, 0.0, 2.0, 0.0], [5.0, 5.0, 5.0, 5.0]]);
        let o = mk::<Bp>(&[[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]]);
        let sigma = sig::<Bp>(&[1.0, 2.0]);
        let sigma_d = sig::<Bp>(&[0.5, 3.0]);
        let base: f32 = nse_batch_loss(p.clone(), o.clone(), sigma.clone(), 0.1).into_scalar();
        let composite: f32 =
            nse_batch_deriv_loss(p.clone(), o.clone(), sigma, sigma_d.clone(), 0.0, 0.1)
                .into_scalar();
        assert_eq!(composite, base, "lambda = 0 must reproduce nse-batch exactly");
        // ...and the same through the config dispatcher.
        let mut cfg = LossConfig { kind: LossKind::NseBatchDeriv, ..LossConfig::default() };
        cfg.deriv_weight = 0.0;
        let dispatched: f32 =
            batch_loss(p, o, &cfg, Some(sig::<Bp>(&[1.0, 2.0])), Some(sigma_d)).into_scalar();
        assert_eq!(dispatched, base);
        // A nonzero lambda must actually change the answer (no silent no-op).
        let mut cfg2 = cfg.clone();
        cfg2.deriv_weight = 0.5;
        let with_deriv: f32 = batch_loss(
            mk::<Bp>(&[[2.0, 0.0, 2.0, 0.0], [5.0, 5.0, 5.0, 5.0]]),
            mk::<Bp>(&[[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]]),
            &cfg2,
            Some(sig::<Bp>(&[1.0, 2.0])),
            Some(sig::<Bp>(&[0.5, 3.0])),
        )
        .into_scalar();
        assert!(with_deriv > base, "lambda 0.5 did nothing: {with_deriv} vs {base}");
    }

    /// Hand-computed two-gauge case (eps = 0, lambda = 0.5).
    ///
    /// gauge 0: obs [1, 2, 4, 4]  sim [1.5, 2.0, 3.0, 5.0]
    /// gauge 1: obs [0, 1, 3, 6]  sim [0.0, 2.0, 2.0, 6.0]
    /// sigma = [2, 1], sigma_d = [1, 0.5]
    ///
    /// LEVEL (mean over 8 (day, gauge) entries of resid^2 / sigma_g^2):
    ///   g0 resid [0.5, 0, -1, 1] -> sq sum 2.25; / 2^2 = 0.5625
    ///   g1 resid [0, 1, -1, 0]   -> sq sum 2.00; / 1^2 = 2.0
    ///   L_nse = (0.5625 + 2.0) / 8 = 2.5625 / 8 = 0.3203125
    ///
    /// DERIV (mean over all 6 adjacent pairs of (dsim - dobs)^2 / sigma_d_g^2):
    ///   g0 dsim [0.5, 1, 2], dobs [1, 2, 0] -> diff [-0.5, -1, 2] -> sq sum 5.25; / 1^2   = 5.25
    ///   g1 dsim [2, 0, 4],   dobs [1, 2, 3] -> diff [1, -2, 1]    -> sq sum 6.00; / 0.5^2 = 24.0
    ///   L_deriv = (5.25 + 24.0) / 6 = 29.25 / 6 = 4.875
    ///
    /// TOTAL: 0.3203125 + 0.5 * 4.875 = 0.3203125 + 2.4375 = 2.7578125
    #[test]
    fn nse_batch_deriv_matches_hand_computation() {
        let p = mk::<Bp>(&[[1.5, 2.0, 3.0, 5.0], [0.0, 2.0, 2.0, 6.0]]);
        let o = mk::<Bp>(&[[1.0, 2.0, 4.0, 4.0], [0.0, 1.0, 3.0, 6.0]]);
        let sigma = sig::<Bp>(&[2.0, 1.0]);
        let sigma_d = sig::<Bp>(&[1.0, 0.5]);
        let level: f32 = nse_batch_loss(p.clone(), o.clone(), sigma.clone(), 0.0).into_scalar();
        assert!((level - 0.3203125).abs() < 1e-5, "L_nse = {level}");
        let deriv: f32 = obs_diff_term(p.clone(), o.clone(), sigma_d.clone(), 0.0).into_scalar();
        assert!((deriv - 4.875).abs() < 1e-5, "L_deriv = {deriv}");
        let total: f32 = nse_batch_deriv_loss(p, o, sigma, sigma_d, 0.5, 0.0).into_scalar();
        assert!((total - 2.7578125).abs() < 1e-5, "L = {total}");
    }

    /// A gap BREAKS the pair chain: 4 valid days split 2+2 give 2 adjacent
    /// pairs, not `n_valid - 1 = 3` — and the NaN never reaches the
    /// arithmetic. Mirrors `landscape::objective::adjacent_valid_pairs`.
    #[test]
    fn obs_diff_term_counts_adjacent_valid_pairs_only() {
        // obs day 2 missing -> pairs {(0,1), (3,4)} only.
        let o = mk_n::<Bp>(&[&[1.0, 2.0, f32::NAN, 4.0, 6.0]]);
        let p = mk_n::<Bp>(&[&[1.0, 3.0, 10.0, 4.0, 5.0]]);
        let sigma_d = sig::<Bp>(&[1.0]);
        // pair 0: dsim 2, dobs 1 -> 1 ; pair 3: dsim 1, dobs 2 -> 1
        // mean over the 2 valid pairs = 1.0 (the wild day-2 prediction is
        // invisible, because every pair touching it is masked out).
        let v: f32 = obs_diff_term(p.clone(), o, sigma_d.clone(), 0.0).into_scalar();
        assert!(v.is_finite(), "NaN obs leaked into the term: {v}");
        assert!((v - 1.0).abs() < 1e-5, "got {v}, want 1.0");
        // Same series with the hole filled: 4 pairs, and day 2's prediction
        // now costs a great deal — proof the mask is doing the work.
        let o_full = mk_n::<Bp>(&[&[1.0, 2.0, 3.0, 4.0, 6.0]]);
        let v_full: f32 = obs_diff_term(p, o_full, sigma_d, 0.0).into_scalar();
        assert!((v_full - 21.75).abs() < 1e-4, "got {v_full}, want 21.75");
    }

    /// A gauge with constant observations has `sigma_d = 0`. As in
    /// `nse_batch_loss`, `+ eps` is what keeps the denominator finite — the
    /// term must not be NaN or Inf.
    #[test]
    fn zero_sigma_d_gauge_stays_finite() {
        let p = mk::<Bp>(&[[1.0, 2.0, 4.0, 8.0]]);
        let o = mk::<Bp>(&[[5.0, 5.0, 5.0, 5.0]]);
        // dsim [1, 2, 4], dobs [0, 0, 0] -> mean sq 21/3 = 7; / (0 + 0.1)^2 = 700
        let v: f32 = obs_diff_term(p.clone(), o.clone(), sig::<Bp>(&[0.0]), 0.1).into_scalar();
        assert!(v.is_finite(), "sigma_d = 0 produced {v}");
        assert!((v - 700.0).abs() < 1e-2, "got {v}, want 700");
        // ...and the composite, with a degenerate sigma too (the same gauge
        // is constant, so `gauge_obs_std` would return 0.0 as well).
        let total: f32 =
            nse_batch_deriv_loss(p, o, sig::<Bp>(&[0.0]), sig::<Bp>(&[0.0]), 0.5, 0.1)
                .into_scalar();
        assert!(total.is_finite(), "composite was {total}");
    }

    /// Gradient sanity: central-difference check of `dL/dp` on a small
    /// two-gauge tensor. The loss is quadratic in `p`, so the central
    /// difference has zero truncation error and `h = 1e-2` sits well above
    /// the f32 round-off floor of the loss (~1e-7). Tolerances follow the
    /// repo's gradcheck convention: accept on rel < 5e-3 OR abs < 1e-4.
    #[test]
    fn nse_batch_deriv_gradient_matches_finite_differences() {
        const H: f32 = 1e-2;
        const REL_TOL: f32 = 5e-3;
        const ABS_TOL: f32 = 1e-4;
        let base = [[1.5f32, 2.0, 3.0, 5.0], [0.0, 2.0, 2.0, 6.0]];
        let obs = [[1.0f32, 2.0, 4.0, 4.0], [0.0, 1.0, 3.0, 6.0]];
        let sigma = [2.0f32, 1.0];
        let sigma_d = [1.0f32, 0.5];
        let lam = 0.5;

        let p_ad = mk::<Ad>(&base).require_grad();
        let loss = nse_batch_deriv_loss(
            p_ad.clone(),
            mk::<Ad>(&obs),
            sig::<Ad>(&sigma),
            sig::<Ad>(&sigma_d),
            lam,
            0.1,
        );
        let grads = loss.backward();
        let analytic: Vec<f32> = p_ad.grad(&grads).unwrap().into_data().to_vec().unwrap();

        let eval = |rows: &[[f32; 4]; 2]| -> f32 {
            nse_batch_deriv_loss(
                mk::<Bp>(rows),
                mk::<Bp>(&obs),
                sig::<Bp>(&sigma),
                sig::<Bp>(&sigma_d),
                lam,
                0.1,
            )
            .into_scalar()
        };

        for gi in 0..2 {
            for ti in 0..4 {
                let mut up = base;
                up[gi][ti] += H;
                let mut dn = base;
                dn[gi][ti] -= H;
                let fd = (eval(&up) - eval(&dn)) / (2.0 * H);
                let got = analytic[gi * 4 + ti];
                let abs = (got - fd).abs();
                let rel = abs / fd.abs().max(1e-12);
                assert!(
                    rel < REL_TOL || abs < ABS_TOL,
                    "grad mismatch at ({gi}, {ti}): analytic {got}, fd {fd} (rel {rel}, abs {abs})"
                );
                assert!(got != 0.0, "gradient is dead at ({gi}, {ti})");
            }
        }
    }

    #[test]
    fn loss_denominator_matches_each_objective_mean() {
        use crate::config::LossConfig;
        // L1 is a mean over (gauge, timestep) ELEMENTS.
        let l1 = LossConfig::default();
        assert_eq!(loss_denominator(&l1, 3, 5), 15);
        // The composite objectives are per-gauge means (mean over GAUGES).
        let mut nk = LossConfig::default();
        nk.kind = LossKind::NnseKge;
        assert_eq!(loss_denominator(&nk, 3, 5), 3);
        let mut kge = LossConfig::default();
        kge.kind = LossKind::Kge;
        assert_eq!(loss_denominator(&kge, 3, 5), 3);
        // Batch-NSE is a mean over ELEMENTS, like L1.
        let mut nb = LossConfig::default();
        nb.kind = LossKind::NseBatch;
        assert_eq!(loss_denominator(&nb, 3, 5), 15);
    }

    #[test]
    fn scaled_micro_losses_recombine_to_pooled_l1() {
        // The accumulation identity the driver relies on:
        //   Σ_i (L_i · n_i) / Σ_i n_i  ==  L over the pooled batch,
        // with n_i = loss_denominator. Unequal micro-batch sizes on purpose.
        use crate::config::LossConfig;
        let cfg = LossConfig::default();
        let p_all = mk::<Bp>(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [0.0, 0.0, 0.0, 0.0]]);
        let o_all = mk::<Bp>(&[[1.0, 1.0, 1.0, 1.0], [8.0, 8.0, 8.0, 8.0], [1.0, 2.0, 3.0, 4.0]]);
        let pooled: f32 = batch_loss(p_all.clone(), o_all.clone(), &cfg, None, None).into_scalar();

        // Micro 1 = gauges {0, 1}; micro 2 = gauge {2}.
        let p1 = p_all.clone().slice([0..2, 0..4]);
        let o1 = o_all.clone().slice([0..2, 0..4]);
        let p2 = p_all.slice([2..3, 0..4]);
        let o2 = o_all.slice([2..3, 0..4]);
        let n1 = loss_denominator(&cfg, 2, 4) as f32;
        let n2 = loss_denominator(&cfg, 1, 4) as f32;
        let l1_: f32 = batch_loss(p1, o1, &cfg, None, None).into_scalar();
        let l2_: f32 = batch_loss(p2, o2, &cfg, None, None).into_scalar();
        let recombined = (l1_ * n1 + l2_ * n2) / (n1 + n2);
        assert!(
            (recombined - pooled).abs() <= 1e-6 * pooled.abs().max(1.0),
            "recombined {recombined} != pooled {pooled}"
        );
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
        // Verify the area-pool body reduces to block-mean whenever the
        // trimmed window IS a multiple of 24. tau=0, T=72 → trimmed window
        // is hours [0..48) (length 48 = 2 days exactly, day-aligned).
        let device = Default::default();
        let v: Vec<f32> = (0..72).map(|x| x as f32).collect();
        let input: Tensor<Bp, 2> = Tensor::<Bp, 1>::from_data(
            burn::tensor::TensorData::new(v, [72]),
            &device,
        )
        .reshape([1, 72]);
        let out = tau_trim_and_downsample(input, 0);
        let got: Vec<f32> = out.into_data().to_vec().unwrap();
        // Sliced = hours 0..48.
        // Day 0 = mean(0..=23) = 11.5
        // Day 1 = mean(24..=47) = 35.5
        assert!((got[0] - 11.5).abs() < 1e-4, "got {}", got[0]);
        assert!((got[1] - 35.5).abs() < 1e-4, "got {}", got[1]);
    }

    #[test]
    fn new_tau_equals_legacy_tau_plus_eleven_shifted_one_day() {
        // Convention-change equivalence: new tau=t reproduces the legacy
        // slice [13+(t+11) : -11+(t+11)] exactly, offset by one pooled day
        // (legacy day i was scored against obs day i+1; new day i against
        // obs day i). new[:, j+1] == legacy[:, j] for all j.
        let device = Default::default();
        let t_hours = 24 * 10;
        let v: Vec<f32> = (0..t_hours).map(|x| ((x * 37) % 101) as f32).collect();
        let input: Tensor<Bp, 2> = Tensor::<Bp, 1>::from_data(
            burn::tensor::TensorData::new(v.clone(), [t_hours]),
            &device,
        )
        .reshape([1, t_hours]);
        let tau_new: u32 = 9; // ≡ legacy tau 20
        let new_out: Vec<f32> = tau_trim_and_downsample(input, tau_new)
            .into_data()
            .to_vec()
            .unwrap();
        // Legacy formula, computed by hand: start 13+20=33, end T-11+20=T+9… the
        // legacy end offset (-11+tau) only stays in-bounds for tau<=11, so build
        // the expected bins directly from the window definition instead: legacy
        // day j covered hours [33 + 24j, 33 + 24(j+1)).
        let n_days = new_out.len();
        for j in 0..n_days - 1 {
            let s = 33 + 24 * j;
            let expect: f32 = v[s..s + 24].iter().sum::<f32>() / 24.0;
            assert!(
                (new_out[j + 1] - expect).abs() < 1e-4,
                "day {j}: legacy {expect} vs new[j+1] {}",
                new_out[j + 1]
            );
        }
    }
}
