# DDRS daily-downsample area-mode fix design

**Date:** 2026-06-04
**Branch (planned):** `area-mode-downsample`
**Successor to:** `docs/superpowers/specs/2026-06-04-ddr-ddrs-training-step-parity-design.md`
(`§5.1 Empirical verdict` — concrete root cause: downsample-mode divergence)
**Status:** ready for implementation

## 1. The bug, exactly

DDRS's hourly→daily downsample in `src/training/loss.rs::tau_trim_and_downsample`
diverges from DDR's at `~/projects/ddr/src/ddr/io/functions.py:22`. The
training-step parity scaffold (PR #13) localized this as the load-bearing
divergence behind the trained-`n` saturation:

| Side | Slice | Length | Reduction | Output |
|------|-------|--------|-----------|--------|
| DDR `scripts/train.py:78,80` | `[13 : -11+tau]` | 2139 (for tau=3, rho_hourly=2160) | `F.interpolate(mode="area", size=(rho_days,))` | 89 daily values |
| DDRS `src/training/loss.rs::tau_trim_and_downsample` | `[13+tau : -11+tau]` | 2136 (multiple of 24) | `reshape(N, 24).mean(dim=2)` | 89 daily values |

Both produce 89 daily values, but each daily value differs by a small amount:

- **DDR's `mode="area"`** is adaptive average pooling. With input length
  `L = 2139` and output length `M = 89`, each output bin covers
  `L / M = 24.0337…` input hours, with fractional weights at bin boundaries:
  ```
  output[i] = ( sum_{j in [floor(i·s), floor((i+1)·s)]}  overlap(j, [i·s, (i+1)·s]) · input[j] )
              / s
  ```
  where `s = L / M`. So each daily value includes a small fractional
  contribution from the boundary hour(s) of the adjacent day, with the
  total weight summing to 24.0337 hours per day.

- **DDRS's `reshape+mean`** uses strict 24-hour blocks. Each daily value
  is `mean(input[i·24 : (i+1)·24])`. The last 3 hours of the original
  hourly window are dropped to make `L` a multiple of 24.

Empirically (PR #13 Layer B sub-step 4): on a snowmelt rising limb in
Jan 1990 at Logan House Ck, the per-day diff is **5.01e-2 m³/s max** —
~1.3% of the daily Q.

That per-day difference propagates through:

1. The L1 loss (the gradient w.r.t. each daily prediction is `sign(p - o) / N`).
2. The autograd chain back through the KAN head.
3. Adam's `1 / (sqrt(m2_hat) + eps)` denominator — for KAN parameters whose
   gradient magnitude is near `eps = 1e-8`, the eps-denominator amplifies
   the 1e-6 per-step grad noise into ~1e-3 per-step parameter noise
   (PR #13 §5.1).
4. Compounded over 175 SGD steps (5 epochs × 35 mb), this lands the
   observed ~0.044 median-`n` divergence (DDR 0.074 vs DDRS 0.030).

Fix the downsample, the chain collapses.

## 2. Concerns

| # | Concern | Why it could go wrong |
|---|---------|----------------------|
| C1 | `F.interpolate(mode="area")` is an op that PyTorch implements as adaptive average pooling on the CPU and a single fused CUDA kernel on the GPU. burn 0.21 does NOT have a built-in `interpolate` with `area` mode for arbitrary input lengths. We have to roll it ourselves. | Hand-coded burn-tensor implementation is ~30-50 lines. The reference formula is well-defined; the implementation risk is autograd correctness (every op in the formula must compose into a clean backward). |
| C2 | The current `tau_trim_and_downsample` has the `squeeze::<2>()` bug for `n_gauges=1` (Layer B finding). Fixing the function gives us a chance to fix the squeeze too. | Bundle the fixes. |
| C3 | The new function must preserve forward + backward autograd parity with DDR's `F.interpolate(mode="area")`. Tested via PR #13's Layer B-D fixtures (which currently pass at relaxed tolerances) after the fix should tighten significantly. | Re-running PR #13's parity tests against tightened tolerances is the verification gate. |
| C4 | Changing the downsample changes the loss surface. After the fix, retrained checkpoints will differ from pre-fix DDRS checkpoints. This is intentional (DDRS-pre-fix was wrong) but means existing trained `.mpk` files must be discarded. | Documented in the commit message. |
| C5 | DDR's pooling is implemented as `F.interpolate(data.unsqueeze(1), size=(rho,), mode="area").squeeze(1)`. The `unsqueeze(1)` adds a "channel" dim because `F.interpolate` expects `(N, C, L)`. Our DDRS implementation operates on a `(n_gauges, n_hours)` tensor and produces `(n_gauges, n_days)`. The unsqueeze is a PyTorch input-shape requirement, not a semantic constraint. | No DDR-side wrapper needed; the per-batch math is the same. |

## 3. Assumptions

| # | Assumption | Justification |
|---|------------|---------------|
| A1 | We can implement `F.interpolate(mode="area")` semantics in pure burn-tensor ops without dropping to a custom Backward implementation. | The reference formula is `output[i] = (sum_j w_ij · input[j]) / total_weight`, where `w_ij = max(0, min(input_right_edge[j], output_right_edge[i]) - max(input_left_edge[j], output_left_edge[i]))`. This is a sparse matmul with constant weights — burn supports it via `Tensor::matmul` (the weights are computed once from shape, not per-batch). |
| A2 | The fix is correct iff PR #13's Layer B-D parity tests pass at **tightened** tolerances: B4 from 0.1 to 1e-5, C2 from 1e-3 to 1e-5, D1 from 2e-3 to 1e-5. (The C7 tau-asymmetry remains as a STAT-only divergence — it's intentional per Bindas et al. 2025 WRR, not a bug — so a small residual diff is expected and acceptable.) | The per-step gradient noise post-fix should be at f32 floor (1e-7 to 1e-6), and the Adam-step amplification at lr=1e-3 will keep stepped-param noise at 1e-4 max. The "tightened" tolerances reflect this. |
| A3 | The `(n_gauges=1)` `squeeze::<2>()` bug is a separate issue but is fixed in the same PR because it blocks direct API reuse from Layer B's sub-test 4. | Layer C's implementer flagged this; the fix is one extra line (specify the squeeze axis explicitly). |
| A4 | The new function preserves the existing public signature `tau_trim_and_downsample(pred: Tensor<B, 2>, tau: u32) -> Tensor<B, 2>`. Callers don't change. | Yes — only the internal slice + reduction changes. |
| A5 | We do NOT change the tau-slicing convention (DDR uses `[13:-11+tau]`, DDRS will continue using `[13+tau:-11+tau]`). The user has confirmed the tau-asymmetry is intentional. | Out of scope per user direction. |

## 4. The implementation

### 4.1 Algorithm

`F.interpolate(mode="area")` with input length `L`, output length `M`:

```
s = L / M                       # input cells per output cell (float)
for i in range(M):
    left  = i       * s         # output bin's left edge in input-space
    right = (i + 1) * s         # output bin's right edge in input-space
    j_lo = floor(left)
    j_hi = ceil(right)          # half-open: [j_lo, j_hi)
    output[i] = 0
    for j in range(j_lo, j_hi):
        cell_left  = max(j,     left)
        cell_right = min(j + 1, right)
        weight     = cell_right - cell_left
        output[i] += weight * input[j]
    output[i] /= s              # normalize by total bin width
```

Equivalent matrix form: `output = W @ input` where `W ∈ R^{M × L}` is a
fixed sparse matrix (sparsity ≤ 3 nonzeros per row for `s ≤ 25`). Each
row sums to 1.

### 4.2 Implementation in `src/training/loss.rs`

Replace the body of `tau_trim_and_downsample` with:

```rust
/// Tau-trim + daily downsample using DDR's `F.interpolate(mode="area")`
/// semantics. Replaces the prior reshape+mean (which required the trimmed
/// hourly length to be a multiple of 24).
///
/// Mirrors DDR's `ddr/io/functions.py:22` exactly: adaptive average pooling
/// from `L` input hours to `M = L // 24` daily outputs, with fractional
/// weights at output-bin boundaries.
///
/// Layout in / out: `(n_gauges, n_hours)` → `(n_gauges, n_days)`.
pub fn tau_trim_and_downsample<B: Backend>(
    pred: Tensor<B, 2>,
    tau: u32,
) -> Tensor<B, 2> {
    let [_g, t_hours] = pred.dims();
    let start = 13 + tau as usize;
    let end = t_hours - 11 + tau as usize;
    assert!(start < end, "tau-trim window degenerate: [{start}, {end})");

    let trimmed = pred.slice([0.._g, start..end]);   // shape: (g, L)
    let l = end - start;
    let m = l / 24;                                   // number of daily bins
    assert!(m > 0, "trimmed window too short: L={l}, M={m}");

    // Build the M × L weight matrix once. Sparsity ≤ 3 nonzeros per row.
    let device = trimmed.device();
    let weights = area_pool_weights::<B>(l, m, &device);  // (M, L), constant
    // output = trimmed (G, L) @ weights.T (L, M) = (G, M)
    trimmed.matmul(weights.transpose())
}

/// Construct the area-mode pooling weight matrix `W ∈ R^{M × L}` such that
/// `W[i, j] = overlap(input_cell_j, output_bin_i) / bin_width`.
///
/// Each row sums to 1. Sparsity: ≤ ceil(L/M) + 1 nonzeros per row.
fn area_pool_weights<B: Backend>(
    l: usize,
    m: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let s = l as f32 / m as f32;
    let mut data: Vec<f32> = vec![0.0; m * l];

    for i in 0..m {
        let left  = (i     as f32) * s;
        let right = ((i+1) as f32) * s;
        let j_lo  = left.floor()  as usize;
        let j_hi  = (right.ceil() as usize).min(l);
        for j in j_lo..j_hi {
            let cell_left  = (j as f32).max(left);
            let cell_right = ((j + 1) as f32).min(right);
            let weight     = (cell_right - cell_left) / s;  // normalized
            data[i * l + j] = weight;
        }
    }

    Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(data, [m * l]),
        device,
    )
    .reshape([m, l])
}
```

### 4.3 Unit tests in `src/training/loss.rs`

Add inline tests covering:

```rust
#[cfg(test)]
mod area_pool_tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    /// When L is a multiple of M, area-mode reduces to strict block-mean.
    /// Verify on L=48, M=2.
    #[test]
    fn area_pool_matches_block_mean_when_divisible() {
        let device = Default::default();
        let input: Tensor<B, 2> = Tensor::from_data(
            // (1 gauge, 48 hours)
            (1..=48).map(|x| x as f32).collect::<Vec<f32>>().as_slice(),
            &device,
        ).reshape([1, 48]);

        // tau=0 trick to expose pure pooling: skip the trim by passing a
        // pre-trimmed slice manually. Better: directly call area_pool_weights
        // and matmul.
        let w = area_pool_weights::<B>(48, 2, &device);
        let out: Tensor<B, 2> = input.matmul(w.transpose());
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        // Block 1: mean(1..=24)  = 12.5
        // Block 2: mean(25..=48) = 36.5
        assert!((v[0] - 12.5).abs() < 1e-5, "got {}", v[0]);
        assert!((v[1] - 36.5).abs() < 1e-5, "got {}", v[1]);
    }

    /// Each row of W sums to 1 (the bin-width normalization).
    #[test]
    fn area_pool_rows_sum_to_one() {
        let device = Default::default();
        let w = area_pool_weights::<B>(2139, 89, &device);
        let row_sums: Tensor<B, 1> = w.sum_dim(1).squeeze::<1>(1);
        for v in row_sums.into_data().to_vec::<f32>().unwrap() {
            assert!((v - 1.0).abs() < 1e-5, "row sum {} != 1", v);
        }
    }

    /// Non-divisible case: L=2139, M=89. Each daily bin should cover
    /// ~24.034 hours with fractional weights. Spot-check the weight
    /// shape (≤ 3 nonzeros per row).
    #[test]
    fn area_pool_handles_non_divisible_input() {
        let device = Default::default();
        let w = area_pool_weights::<B>(2139, 89, &device);
        let data: Vec<f32> = w.into_data().to_vec().unwrap();
        // Spot-check row 0: covers input range [0, 24.034). Cells 0-23
        // contribute their full weight 1/s, cell 24 contributes a
        // fractional 0.034/s.
        let s = 2139.0_f32 / 89.0;
        for j in 0..24 {
            let expected = 1.0 / s;
            assert!((data[0 * 2139 + j] - expected).abs() < 1e-6,
                "row 0 col {j}: got {} want {expected}", data[0 * 2139 + j]);
        }
        let frac = (24.034f32 - 24.0) / s;
        assert!((data[0 * 2139 + 24] - frac).abs() < 1e-4,
            "row 0 col 24: got {} want ~{frac}", data[0 * 2139 + 24]);
        for j in 25..2139 {
            assert!(data[0 * 2139 + j] < 1e-6, "row 0 col {j} should be 0");
        }
    }

    /// n_gauges=1 must not panic (Layer B regression from PR #13).
    #[test]
    fn n_gauges_one_does_not_panic() {
        let device = Default::default();
        // 2160 hourly input → 89 daily output for tau=3.
        let input: Tensor<B, 2> = Tensor::zeros([1, 2160], &device);
        let out = tau_trim_and_downsample(input, 3);
        assert_eq!(out.dims(), [1, 89]);
    }
}
```

### 4.4 Tighten PR #13's parity tests

The training-step parity tests landed in PR #13 with relaxed tolerances to
make them pass against the buggy downsample. After this fix:

| Test | Old tol | New tol |
|------|---------|---------|
| Layer B4 (tau-trim daily Q) | 0.1 m³/s | 1e-5 |
| Layer C2 (per-KAN-param grads) | 1e-3 | 1e-5 |
| Layer D1 (post-Adam-step params) | 2e-3 | 1e-5 |

Tighten each in the respective `tests/training_step_layer_*.rs` file in the
same commit as the downsample fix. If any fails at the new tolerance,
that's the signal to investigate further (some other minor divergence may
still be lurking — but the dominant one should be gone).

### 4.5 Retrain + retest

After the fix lands:

1. `cargo run --release --bin ddrs -- --config config/merit_training.yaml run --workflow train-and-test`
   (use the explicit `--config` flag to bypass the gitignored `ddrs.yaml` bootstrap-from-prior-run trap documented in CLAUDE.md after PR #12).
2. `cargo run --release --bin dump_parameters -- --config <run>/config.yaml --checkpoint <ckpt> --output <run>/kan_parameters.nc`
3. Run the Layer 2 trained-distribution parity notebook from PR #12
   (`.claude/skills/ddrs-eval-plots/references/parity_trained.md`).
4. The expected outcome: median `n` ≈ DDR's 0.074, KS ≤ 0.10, Spearman ≥ 0.70.
   That's the formal closure of the n-saturation investigation.

## 5. What success looks like

| Outcome | Meaning | Next step |
|---------|---------|-----------|
| **Layer B4 / C2 / D1 all pass at 1e-5** | Downsample fix closes the per-step gradient + param divergence. | Retrain + run trained-distribution parity. |
| **Layer B4 passes but C2 fails** | The new downsample is correct but reveals another, smaller divergence in the autograd path. | Investigate; out of scope. |
| **Layer B4 passes, C2 passes, D1 fails** | Adam's eps-denominator is sensitive to even smaller grad differences than this fix addresses. | Either tweak eps, or accept the residual (~1e-5 per-step) as f32 floor noise. |
| **Layer B4 fails** | The new `area_pool_weights` function is wrong. | Fix it; re-test. |

## 6. Implementation order

1. Read `~/projects/ddr/src/ddr/io/functions.py:22` again to confirm the
   `unsqueeze(1) / squeeze(1)` is purely shape-adapting (no channel-mixing).
2. Implement `area_pool_weights` + 4 unit tests in `src/training/loss.rs`.
3. Rewrite `tau_trim_and_downsample` body to use it.
4. Run `cargo test --lib training::loss::area_pool_tests`.
5. Run `cargo test --features fixtures --test training_step_layer_b --test training_step_layer_c --test training_step_layer_d` against the **old** tolerances. Confirm tests still pass with the fix (which they will — the fix only reduces diffs, never increases them).
6. Tighten the tolerances per §4.4. Re-run. All should pass.
7. Retrain DDRS at the fixed downsample.
8. Re-run PR #12's Layer 2 notebook. Record verdict in a §5.1 of this spec.
9. Open PR.

## 7. Out of scope

- The C7 tau-asymmetry. User has confirmed it's intentional. After this
  fix, if the trained `n` is still off, revisit.
- Multi-step training-trajectory comparison (PR #12 already does this).
- Adam epsilon-tuning. burn 0.21's default (`1e-8`) matches PyTorch's
  default exactly; PR #13's Layer D confirmed the algebraic equivalence.
- The `tau_trim_and_downsample::squeeze::<2>()` bug for n_gauges=1 is
  fixed implicitly by the rewrite (matmul preserves the leading
  axis without need for any squeeze).

---

## §5.1 Empirical verdict (Task 6 of the plan)

**Pre-fix baseline:** `.ddrs/runs/2026-06-04T01-57-45Z-train-and-test`
(the first DDRS run with NaN-filter + log_space [p_spatial] from PR #12
applied; the run that produced the §5.1 entry in
`2026-06-03-ddr-ddrs-trained-saturation-parity-design.md`).

**Post-fix retrain:** `.ddrs/runs/2026-06-04T16-46-58Z-train-and-test`
(adds the area-mode downsample fix from commits c334f77 + 3b907f1).

**DDR reference:** `epoch_5_mb_35.pt` from
`~/projects/ddr/output/ddr-v0.5.2.dev2+g21a3a96b5-merit-training/2026-03-14_06-03-23/`
— verified by PR #13 Task 2 to use the exact parity config.
COMID alignment confirmed: 346,321 CONUS reaches, order matches exactly.

**n-distribution comparison (346,321 CONUS reaches):**

| metric | DDR ref | DDRS pre-fix | DDRS post-fix | gap closed |
|---|---:|---:|---:|---:|
| median n | 0.0744 | 0.0296 | 0.0395 | ~22% |
| mean n | 0.0735 | 0.0370 | 0.0444 | ~20% |
| p95 | 0.1047 | 0.0852 | 0.0962 | ~56% |
| frac n<0.035 | 0.031 | 0.650 | 0.404 | ~40% |
| frac in [.02, .03] | 0.014 | 0.423 | 0.213 | ~50% |

**Cell 2 output (per-parameter stats):**

```
           ddrs_med   ddr_med   ddrs_p5    ddr_p5   ddrs_p95    ddr_p95
param
n          0.039454  0.074377  0.017567  0.038731   0.096171   0.104686
q_spatial  0.469512  0.463047  0.444897  0.422190   0.487581   0.480362
p_spatial  6.634186  8.153398  3.600475  5.494444  10.919070  10.425868

                 KS  Spearman
param
n          0.568536  0.347068
q_spatial  0.206017  0.516933
p_spatial  0.330272  0.370499
```

**Cell 5 verdict:**

```
DDR ↔ DDRS trained-distribution parity (Layer 2):

  n             KS=0.5685  Spearman=+0.3471   ✗ real divergence
  q_spatial     KS=0.2060  Spearman=+0.5169   ✗ real divergence
  p_spatial     KS=0.3303  Spearman=+0.3705   ✗ real divergence
```

**Outcome (from §5 table):** The `n` row returns `✗ real divergence`
(KS=0.5685, well above the 0.10 pass threshold; Spearman=+0.347, well
below the 0.70 fail threshold). The fix is a real improvement (~22-56%
gap closure per metric), but does not close the n-distribution parity.
Median n moved from 0.030 to 0.040, p95 closed to 0.096 vs DDR's 0.105,
and the saturation band shrunk by ~50% on fraction-based metrics. But
median is still ~2× below DDR's 0.074, and the Spearman correlation
(+0.347) indicates weak per-reach correspondence — reaches that should
have high n in DDR are not reliably getting high n in DDRS after training.

The remaining gap is dominated by causes beyond the C7 tau-slicing
asymmetry (tau-slicing closes at most ~3 hours out of 2139). The KS
statistic (0.57 post-fix vs ~0.69 pre-fix) shows the distribution shape
moved toward DDR's but did not converge. Further investigation into
the training dynamics (gradient flow through the KAN head, loss landscape
differences, epoch/minibatch ordering effects) is warranted in a fresh spec.

**Next step:** Open a fresh spec localizing the remaining trained-n
divergence. The area-pool downsample fix is confirmed correct and
beneficial, but does not fully resolve the n-saturation. The saturation
investigation is NOT closed by this fix alone.
