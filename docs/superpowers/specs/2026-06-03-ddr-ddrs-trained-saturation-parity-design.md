# DDR ↔ DDRS trained-`n` saturation parity testing plan

**Date:** 2026-06-03
**Branch (planned):** `trained-parity` (sibling to merged `kan_improvements`)
**Successor to:** `docs/superpowers/specs/2026-06-02-ddr-ddrs-kan-parity-design.md`
**Symptom:** After 5 epochs of `train-and-test` at `seed=42, grid=50, k=2`, DDRS's
trained Manning's `n` distribution is centred ≈ 0.030 with 47 % of CONUS reaches
in the band `[0.020, 0.030]` and a p95 of 0.044 — i.e. the lower ~10 % of the
log-space range `[0.015, 0.25]`. The previous spec proved this is **not** a head
divergence: at fixed init parameters, DDRS's KAN is bit-identical to DDR's
forward (5.96e-8) and backward (1.91e-6) on NdArray, and the CONUS init mean
matches DDR's to < 1 %. So the saturation must come from one of: data, loss,
optimizer, batch construction, training loop.

This spec asks one question: **does DDR train to the same saturated `n`
distribution as DDRS?**

- **If yes:** DDRS is correctly reproducing DDR. The saturation is a property
  of the model + loss + data, not a port bug. The next investigation is in DDR
  itself (or "is saturation actually the right answer?") — outside scope.
- **If no:** there is a real divergence in `src/training/`. The KS gap and the
  per-gauge differences localize which training-loop component to audit next.

---

## 1. Why this matters / why now

The previous parity scaffold's `parity_init.md` Layer 4 result (KS = 0.137 /
0.426 / 0.640 on init outputs) is the **expected** signature of two different
RNG families (StdRng vs Mersenne-Twister) producing draws from the same
distribution and is therefore unobservable as a divergence — means agree, only
shape differs by sub-pct on the relevant metrics. So Layer 4 cannot
disambiguate "DDRS saturates because DDR saturates" from "DDRS saturates due
to a training-loop bug."

We need a head-to-head comparison of the **trained** `n` distributions, run on
the same data, the same time period, and the same gauge set. The parity
scaffold built in `kan_improvements` already gives us the machinery
(`scripts/dump_ddr_init_params.py` mirror, `parity_init.md` notebook recipe);
this spec adapts it to compare trained models.

---

## 2. Concerns

| # | Concern | Why it could go wrong |
|---|---------|----------------------|
| C1 | DDR's `~/projects/ddr/scripts/train.py` may have stale deps or drifted config since the last successful run (March 2026). | Run it once on a fresh venv resolve before committing to a full re-run. If it fails on imports, this entire spec stalls — but the failure mode is loud and immediate. |
| C2 | DDR's training is non-deterministic on GPU even at fixed `seed=42` (cuBLAS reductions, atomic adds in CUDA kernels, optimizer fused kernels). DDRS is also non-deterministic on GPU for the same reasons. So trained outputs can never be bit-identical, only distributionally close. The KS-test pass threshold must reflect this. | Use the conventional 0.10 KS bound at n ≈ 346 k samples — looser than the 0.05 used at init. If both DDR and DDRS hit median `n` ≈ 0.030 with KS ≤ 0.10 and per-gauge correlation > 0.90, "same saturation" is confirmed. |
| C3 | DDR's `merit_training_config.yaml` uses `kan.grid: 50, kan.k: 2`; DDRS now matches. But other knobs (loss function name, optimizer, learning-rate schedule, batch_size semantics, grad_clip_max_norm, warmup, rho) may differ silently — Task 0 of the parity plan didn't audit them. | Spec §4 Layer 0 adds a training-config audit pass before the parity comparison. Surfaces drift before it pollutes the trained outputs. |
| C4 | "Saturation may be the correct answer." If DDR also produces `n ≈ 0.030`, then the user's original question ("the trained n looks too low") is best answered by DDR's published-paper expected `n` distribution or by hydrological reasoning, neither of which we can settle in this spec. | Acceptable — that's the answer. The spec scope ends at "DDR matches" / "DDR doesn't match"; the follow-up investigation into whether the matched value is *physically right* is out of scope and would belong in a hydrology spec, not a port-parity one. |
| C5 | Existing DDR outputs at `~/projects/ddr/output/ddr-v0.5.2.dev2+g21a3a96b5-merit-training/` are from March 14 2026 (3 months old). Re-using them avoids a fresh GPU run but risks comparing against a DDR config / data-source state that drifted since. | Default to **re-running DDR fresh** with the parity config to avoid this. Use the existing checkpoint only as a coarse "is this hypothesis even worth running" sanity check (Layer 0.5 below). |
| C6 | `dump_ddr_init_params.py` (Task 11 of the previous plan) reads its own normalization stats and may not match what `~/projects/ddr/scripts/train.py` uses at training time. If the trained-vs-init dump scripts disagree, the post-training comparison would be polluted by a normalization mismatch unrelated to the saturation question. | Spec mandates: the trained-`n` dump for DDR and DDRS must use the SAME normalization the training loop used (= the existing `merit_attribute_statistics_*.json`). Verify in §4 Layer 1. |

---

## 3. Assumptions

| # | Assumption | Justification |
|---|------------|---------------|
| A1 | DDR's `scripts/train.py` is runnable on this machine with its existing `uv` venv. | DDR's venv was confirmed live in the previous parity plan (Tasks 5, 7, 11). The training script is the same family; should work. |
| A2 | One DDR training run (5 epochs, seed=42, same gauges + dates as DDRS) is sufficient to characterise DDR's trained-`n` distribution. We do not need an ensemble across seeds — saturation either appears at seed=42 or it doesn't. | Asymmetric saturation across seeds would itself be a discovery; if it shows up here we widen scope. The user's symptom is observed at seed=42 specifically. |
| A3 | The DDR training is deterministic up to GPU non-determinism (~1e-4 on outputs after 5 epochs) at fixed config + seed. | DDR's documented behavior; consistent with PyTorch + Adam + grad_clip default. |
| A4 | KS ≤ 0.10 on 346 k-reach trained-`n` is "indistinguishable saturation"; KS ≥ 0.20 is "real divergence to investigate." Between 0.10 and 0.20 → DONE_WITH_CONCERNS and bring to user. | KS ≈ 0.05 is "indistinguishable" at large n; we relax to 0.10 because GPU non-determinism widens the noise floor for trained outputs vs init outputs. |
| A5 | Per-gauge Spearman correlation > 0.90 is the second pass criterion. Below 0.70 → real divergence. | Distribution-level KS can pass while per-gauge predictions are scrambled; the correlation guards against that. |
| A6 | Re-using DDRS's existing training run `.ddrs/runs/2026-06-03T09-45-09Z-train-and-test/` is fine — no need to re-train. Its `kan_parameters.nc` was already dumped during Task 13. | The DDRS side is fixed; the question is purely whether DDR produces the same thing. |

---

## 4. Layered test plan

Three layers, ordered cheapest → most decisive. Pause between layers and read
the result before continuing.

### Layer 0 — Training-config audit (no code; 1 hour)

**Question:** Do DDR's and DDRS's training loops use the same loss, optimizer,
LR schedule, batch construction, gradient clipping, warmup, rho, and shuffling?

**Procedure:** for each table row below, read the cited DDR file and the cited
DDRS file. Fill in the "Match?" column with ✓ or ✗ (actual: …, want: …).

| Field | DDR source | DDRS source | Match? |
|-------|------------|-------------|--------|
| Loss function | `~/projects/ddr/src/ddr/training/loss.py` (or wherever the trainer imports from) | `src/training/loss.rs` (if present) or wherever `driver.rs` computes loss | (audit) |
| Optimizer | DDR's train.py `torch.optim.*` construction | `src/training/driver.rs` `Optimizer::*` construction | (audit) |
| LR schedule | DDR YAML's `experiment.learning_rate` mapping | DDRS YAML's `experiment.learning_rate` mapping | already audited as equal in previous plan; re-confirm |
| Adam betas / eps / weight_decay | torch defaults: `betas=(0.9, 0.999), eps=1e-8, weight_decay=0` | burn's `AdamConfig::new()` defaults — confirm match | (audit) |
| Batch size (training) | DDR YAML `experiment.batch_size: 64` | DDRS YAML `experiment.batch_size: 64` | already ✓ |
| Batch construction (per epoch) | `RandomSampler` over gauge IDs, seeded by `np_seed` | `src/data/dataset.rs::shuffle_*` — confirm equivalent sampler logic | (audit; expected STAT only per spec C5 of previous plan) |
| Shuffle PRNG | `numpy.random.default_rng(np_seed)` | Rust `rand::SeedableRng::seed_from_u64(np_seed)` | STAT only — different sequences |
| `grad_clip_max_norm` | DDR YAML `experiment.grad_clip_max_norm: 1.0` | DDRS YAML same | already ✓ |
| Warmup | DDR YAML `experiment.warmup: 5` | DDRS YAML same | already ✓ |
| Rho (training window length) | DDR YAML `experiment.rho: 90` | DDRS YAML same | already ✓ |
| Epoch count | DDR YAML `experiment.epochs: 5` | DDRS YAML same | already ✓ |
| Start / end time | DDR YAML | DDRS YAML | already ✓ |
| Gauge list | `references/gage_info/gages_3000.csv` | same path | already ✓ |
| Streamflow source | `merit_dhbv2_UH_retrospective.ic` | same | already ✓ |
| Observations source | `usgs_daily_observations` | same | already ✓ |
| Routing engine f32 throughput | DDR f32-by-construction | DDRS f32-by-construction | already ✓ |

**Pass criterion:** every row ✓ or STAT-only (a deliberate, documented
divergence with known statistical equivalence). Any unexpected ✗ → STOP and
report; that ✗ is more likely the saturation culprit than anything found in
Layer 2.

### Layer 0.5 — Sanity check against existing DDR output (~30 min, no GPU run)

**Question:** Before committing to a fresh DDR training run, does any existing
DDR run have a `kan_parameters.nc` or equivalent we can quickly inspect?

**Procedure:**
1. Search for `kan_parameters.nc` / `kan_parameters_*.nc` under
   `~/projects/ddr/output/`.
2. If found, open the most recent one, compute median + 5th/95th percentiles
   of `n`.
3. **Read-only quick verdict:** Does it look saturated (median ≈ 0.03,
   p95 < 0.05) or healthy (median ≈ 0.06, p95 ≈ 0.10)?

**Pass criterion:** none — this is a 30-minute "is the saturation hypothesis
even worth a full retrain" check. Whatever the result is, Layer 1 still runs.

**Failure routing:**
- If DDR's existing `n` is also saturated → strong evidence the saturation is
  shared. Layer 1 will confirm. Move with confidence.
- If DDR's existing `n` is not saturated → strong evidence of a DDRS-only bug.
  Layer 1 + Layer 2 become more urgent.
- If no usable existing artifact → skip to Layer 1.

### Layer 1 — Fresh DDR training run + dump (~30-60 min)

**Question:** With DDR's training loop running at the same `seed=42` config on
the same data, what does its trained `n` distribution look like over CONUS?

**Procedure:**
1. Ensure `~/projects/ddr/.venv` is resolved and importable.
2. Confirm `~/projects/ddr/config/merit_training_config.yaml` has not
   drifted from the parity assumption (`seed: 42`, `np_seed: 42`,
   `kan.grid: 50`, `kan.k: 2`, the 10 attribute names matching DDRS's
   `kan_head.input_var_names` in the same order, the 3 learnable params).
   If any of these have drifted on the DDR side, fix the YAML to match the
   parity baseline AND record the edit in the spec.
3. Launch DDR training:
   ```bash
   cd ~/projects/ddr && uv run python scripts/train_and_test.py
   ```
   Capture stdout to `~/projects/ddr/output/<latest>/training.log`.
   Monitor: loss curve should fall smoothly; if it diverges to NaN, STOP.
4. After completion, locate the saved checkpoint (under
   `output/<run>/saved_models/`). Pick the final epoch's checkpoint
   (`epoch_5_mb_*` or whatever the run naming convention is).
5. Run DDR's parameter-dump equivalent — likely
   `scripts/predictions/geometry_predictor.py` or
   `scripts/predictions/kan_predictor.py`. (Find the analog of DDRS's
   `dump_parameters`; if none exists, write a tiny script that does the same:
   load checkpoint → forward over CONUS attributes → denormalise → write
   NetCDF.)
6. Output path: `/tmp/kan_params_trained_ddr.nc`.

**Pass criterion:** the run completes; the NetCDF is written with 346 321
reaches; `n` distribution is finite.

**Failure routing:**
- DDR training NaNs out → blocked, escalate. Possibly a known DDR issue.
- DDR training takes much longer than DDRS (> 4×) → DONE_WITH_CONCERNS;
  may indicate a config divergence.

### Layer 2 — Trained-`n` parity comparison (~1 hour)

**Question:** Are DDR's and DDRS's trained `n` distributions statistically
equivalent?

**Procedure:** new notebook `parity_trained.ipynb` at
`<latest-ddrs-run>/plots/` (or anywhere; not run-specific). Five cells:

1. **Load.** Open `<latest-ddrs-run>/kan_parameters.nc` and
   `/tmp/kan_params_trained_ddr.nc`. Intersect to common CONUS COMIDs.
2. **Per-distribution stats.** Print median, mean, p5, p95, std for `n`,
   `q_spatial`, `p_spatial` from both. Compute KS statistic for each.
3. **Side-by-side histograms.** Same template as
   `.claude/skills/ddrs-eval-plots/references/parity_init.md` Cell 2.
4. **Per-gauge scatter.** For each COMID, plot DDRS's trained `n` against
   DDR's. Compute Spearman correlation. Save `parity_trained_scatter.png`.
5. **Verdict cell.** Apply pass criteria from A4 + A5:
   - KS ≤ 0.10 AND Spearman > 0.90 → "DDR and DDRS train to the same `n`
     saturation." The port is faithful; the saturation is a model+data
     property.
   - KS ≥ 0.20 OR Spearman < 0.70 → "Real divergence in `src/training/`."
   - Anything between → DONE_WITH_CONCERNS, bring to user.

**Pass criterion:** the verdict cell prints a clear verdict (one of the three
above). The verdict itself is the deliverable.

---

## 5. What success looks like

Three possible outcomes after Layer 2:

| Outcome | Meaning | Next step |
|---------|---------|-----------|
| **Same saturation** (KS ≤ 0.10, Spearman > 0.90, both median ≈ 0.030) | Port is faithful. Saturation is a property of the routing-MC + summed-Q loss + USGS data at this scale. | Out of scope for the port. If the saturation is hydrologically wrong, the next spec is a model/loss redesign — same team, different repo. |
| **DDR healthier** (DDR median > 0.05, DDRS median ≈ 0.030, KS > 0.20) | DDRS has a training-loop bug. The Layer 0 audit's ✗ rows are the prime suspects. | New spec to localize the bug in `src/training/`. The five candidate causes (loss bias, Adam hyperparams, batch shuffle, grad clip/sigmoid saturation, NaN masking) become the test plan. |
| **Both saturated, different shapes** (KS 0.10–0.20, Spearman ~0.70–0.90) | Likely a minor optimizer or batch-order divergence — both heading to the same attractor but at different rates. | DONE_WITH_CONCERNS. Probably worth one targeted fix (e.g., port DDR's exact Adam invocation byte-for-byte) but does not necessarily unblock anything. |

---

## 6. Implementation order

1. Layer 0 audit (1 hour, no code). Surface ✗ rows.
2. Layer 0.5 quick check on existing DDR outputs (30 min). May change priority.
3. Layer 1 fresh DDR training (30–60 min) + DDR-side dump script if one
   doesn't already exist.
4. Layer 2 notebook + KS/Spearman verdict.
5. Spec append: record the verdict in §5.

---

## 7. Out of scope

- Whether DDR's `n` distribution is *physically correct*. That is a
  hydrological question, not a port-parity one.
- Changing the training loop. This spec only diagnoses; the fix (if needed)
  belongs in a separate spec.
- Anything beyond `n`. `q_spatial` and `p_spatial` may also diverge but the
  user's stated symptom is `n` saturation; reporting their KS values is
  background information.
- Ensemble across seeds. The user observed the symptom at `seed=42`; that's
  the single seed we compare.
- Comparing against published DDR papers. Out of scope for parity work.
