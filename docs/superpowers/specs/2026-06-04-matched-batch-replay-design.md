# Matched-batch replay parity experiment design

**Date:** 2026-06-04
**Branch:** `training-step-parity` (PR #13)
**Successor to:** `docs/superpowers/specs/2026-06-04-ddrs-area-mode-downsample-fix-design.md`
(`§5.1 Empirical verdict` — area-pool fix closed ~50% of the saturation
band fraction, but trained-`n` per-reach Spearman is still only 0.347)

## 1. The question

After the area-pool fix (`c334f77`), DDRS's trained `n` distribution moved
substantially toward DDR's (median 0.030 → 0.040, frac<0.035 0.65 → 0.40)
but the per-reach correspondence is still weak (KS = 0.5685, Spearman =
+0.347). PR #11/#12/#13 proved every per-stage operation is bit-identical
to DDR at fixed inputs (forward 5.96e-8, backward 1.91e-6, Adam 1.49e-3).
The unresolved divergence sits between "same operations" and "different
trained outputs."

The leading hypothesis (`docs/.../2026-06-04-ddrs-area-mode-downsample-fix-design.md`
§5.1 + chat 2026-06-04): the remaining gap is **mini-batch ordering**
compounded over 175 SGD steps on a highly non-convex loss surface
(~93k KAN params, ~210k data points — many near-equivalent local minima).
PyTorch's MT19937 vs Rust's `StdRng` produce different shuffles at the
same nominal seed; same operations + different inputs = different
trajectories.

**Question:** if DDR and DDRS see **exactly the same gauges in exactly the
same mini-batch order**, do they converge to the same per-reach `n`?

## 2. Why this matters

If the answer is yes (Spearman > 0.9 after matched batches), the parity
work is **done** — DDRS is a faithful port; the remaining trained-output
divergence at default config is inherent SGD-trajectory variance. The user
can ship the port + close the n-saturation investigation.

If the answer is no (Spearman still < 0.7), there is a systematic
divergence beyond batch ordering that the existing parity scaffold hasn't
caught — and a fresh investigation is warranted (likely targeting the
tau-asymmetry, GPU non-determinism, or some unaudited path through the
training data loader).

The experiment is cheap (~half a day) and produces a binary verdict.

## 3. Concerns

| # | Concern | Why it could go wrong |
|---|---------|----------------------|
| C1 | DDR's training script doesn't expose batch-order as a hook. Capturing the order requires modifying DDR's `scripts/train.py` (10 lines) to log per-`(epoch, mb_idx)`. | Invasive, but reversible. Use a `--dump-batch-order <path>` CLI flag; default behavior preserved. |
| C2 | DDRS's training driver builds batches via `RandomSampler` equivalents in `src/data/dataset.rs`. Replaying an external order requires a new `--batch-order-from <path>` flag plus a sampling-bypass code path. | New `pub fn` on `MeritGagesDataset` that takes a `Vec<Vec<StaId>>` (epochs × mb) instead of sampling. ~30 lines. |
| C3 | If DDR's batch sequence references gauges that don't appear in DDRS's `gages_3000.csv` (e.g. DDR has filtered some out for missing data), the replay fails. | Pre-flight: assert the two gauge sets agree set-wise. They should — both load from the same CSV. |
| C4 | Mini-batch SIZE must match too (64 gauges/mb in production), not just the order. If DDR's last mb of an epoch has < 64 gauges (drop_last=False), DDRS must match. | DDR-side dumper records the actual per-mb count; DDRS replay respects it. |
| C5 | Even with matched batches, the tau-asymmetry residual (~5e-2 m³/s daily Q on rising limbs) will produce SOME per-step gradient divergence. Spearman won't be 1.0. | Acceptable per spec §1. The threshold is "qualitatively different from the current 0.347" — 0.85+ is success; 0.95+ would be a strong signal. |
| C6 | GPU non-determinism alone (cuBLAS atomic accumulation orders) means DDRS twice at the same seed produces slightly different outputs. We can't subtract that out without a CPU-only training run. | A CPU-only DDRS run is ~10× slower (~5 hours). If the GPU-run answer is ambiguous (0.7 < Spearman < 0.9), fall back to CPU. Otherwise skip. |
| C7 | DDR's `RandomSampler(replacement=False)` is per-epoch — it shuffles each epoch independently. DDRS may do the same or may use one shuffle for the whole training run. Verify before replay. | Inspect both implementations; record actual behavior in the audit. |

## 4. Assumptions

| # | Assumption | Justification |
|---|------------|---------------|
| A1 | The replay test uses the SAME area-pool-fixed DDRS (commit `c334f77`) and the SAME PR-#12-fixed DDR YAML (`log_space_parameters: [p_spatial]`). Tau-slicing convention unchanged on both sides (intentional per user). | Want to isolate batch-order from the other already-resolved bugs. |
| A2 | Spearman ≥ 0.85 on matched-batch trained `n` closes the investigation. 0.70-0.85 = "improved but not closed, requires deeper analysis." < 0.70 = "matched batches don't help, the bug is elsewhere." | Spearman 0.85 corresponds roughly to the noise floor of independent GPU runs of the same model (per published reproducibility studies of similar-scale DL training). |
| A3 | Capturing DDR's full batch order = ~ (5 epochs × ~35 mb × 64 gauges × 9-char STAID) ≈ 100 KB of JSON. Trivially serializable. | Yes — even at 100 mb of 256 gauges, it's well under 1 MB. |
| A4 | DDRS's training driver can be retrofitted with a `--batch-order-from <path>` flag without disrupting other code paths. The shuffler logic lives in `src/data/dataset.rs`; the change is a conditional `match` over the source. | The dataset module is well-isolated; this is a new code path, not a rewrite. |
| A5 | The DDR-side dump uses STAID (not COMID, not row-index) as the canonical gauge identifier. DDRS's replay maps STAID → batch row at replay time. | STAID is the only identifier both sides agree on byte-for-byte. |

## 5. Implementation outline

### 5.1 DDR-side dumper

Modify `~/projects/ddr/scripts/train.py`. Find the per-mini-batch loop
(around line 50-70). Where the sampler yields `batch_gauges`, also append
to a running log:

```python
if cfg.params.get("dump_batch_order_to"):
    _batch_order_log.append({
        "epoch": epoch_idx,
        "mb": mb_idx,
        "staids": [str(s) for s in batch_gauges],
    })
```

At end of training, dump `_batch_order_log` as JSON to the configured path.
Add a CLI/config flag `dump_batch_order_to: /tmp/ddr_batch_order.json`.

Run DDR for 5 epochs at parity config (seed=42), capture the file.

### 5.2 DDRS-side replay

Add to `src/data/dataset.rs`:

```rust
pub enum BatchSource {
    Shuffled { seed: u64 },
    Replay { batches: Vec<Vec<StaId>> },
}
```

Plumb through `MeritGagesDataset::new` and the training driver's
mini-batch iterator. When `Replay`, the dataset yields batches in the
provided order instead of shuffling.

Add a CLI flag `--batch-order-from <path>` on `ddrs run`. Parse the
JSON, build the `Vec<Vec<StaId>>`, pass to the dataset.

### 5.3 Run + verify

Both DDR and DDRS train at seed=42 with the SAME batch order. Both produce
checkpoints. Run PR #12's `parity_trained` notebook against the two
resulting NetCDFs. The Cell 5 verdict is the test:

- KS(`n`) ≤ 0.10 AND Spearman(`n`) ≥ 0.85: **investigation closed.**
- KS(`n`) ≤ 0.20 AND Spearman(`n`) in 0.70-0.85: improvement, but a deeper
  divergence remains. Open a new spec to investigate (likely targets:
  tau-asymmetry quantification, GPU determinism, forcing data path).
- Spearman(`n`) < 0.70: matched batches didn't help. The bug is in a path
  the existing parity scaffold hasn't covered.

## 6. Out of scope

- Fixing whatever remains if Spearman < 0.85. This spec only diagnoses.
- Multi-seed comparison (DDR seed=42 vs DDR seed=43). Useful as a control
  but adds another training run; defer unless the matched-batch result is
  ambiguous.
- CPU-only run. Only if GPU result is in the 0.70-0.85 ambiguous zone.
- Changing the loss / model. Out of scope for parity work.
- The `q_spatial` / `p_spatial` per-reach correspondence. The original
  symptom is `n`; report the other two for context but don't gate the
  verdict on them.

## 7. Estimated effort

- DDR-side dumper: 1 hour (10 lines + a flag + one retrain to capture).
- DDRS-side replay: 2 hours (new BatchSource enum + dataset method + CLI flag).
- DDR retrain (for batch capture): ~45 min.
- DDRS retrain (with replay): ~45 min.
- Layer-2 notebook + verdict: 30 min.
- Total: ~5 hours wall-clock, ~3 hours active.
