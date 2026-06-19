# ddrs Routing Experiments — Handoff Journal (2026-06-19)

Investigation into **why the trained KAN + Muskingum-Cunge router does not beat
the summed-Q′ baseline** on the global MERIT dataset, and what was tried to fix
it. Reference numbers throughout are on the **matched gauge set** (the 5,224
gauges the trained model actually predicts, intersected with the baseline — the
only fair comparison):

| Reference | median NSE | median KGE | α = σ_sim/σ_obs |
|---|---|---|---|
| summed-Q′ baseline (no routing) | **0.689** | **0.723** | **0.96** |

The arc below establishes three layered findings, in order: the **loss** was
never the limiter → the daily→hourly **interpolation** was the gradient limiter
→ beneath it sits a **structural ceiling**. Code changes live on branch
`routing` (this PR contains only the journal).

---

# Experiment 0 — Baseline characterization (L1 vs NNSE-KGE)

## Experiment
Two pre-existing full runs: L1 loss (`2026-06-16T01-25-11Z`) and the new
`nnse-kge` loss (`2026-06-16T09-48-44Z`), both with fixed Muskingum X = 0.3.

## Hypothesis
L1 in raw m³/s rewards over-attenuation (its optimum sits at α < 1), so a
KGE-aware loss whose `(α−1)²` term penalizes variance loss should restore
amplitude and beat the baseline.

## Changes
None to the router; loss selected via `experiment.loss.kind`.

## Results
- L1: NSE 0.684 / KGE 0.701 / α 0.85.
- NNSE-KGE: NSE 0.684 / KGE 0.699.
- Both **below baseline** (0.689 / 0.723). Routing over-attenuates: α drops
  0.96 → 0.85 in 99.7% of gauges; corr(trained, baseline) = 0.97 (routing barely
  changes shape, mostly shrinks amplitude).

## Notes
Switching the loss did essentially nothing → the objective is not the binding
constraint. Training loss was **flat across all 10 epochs** in both runs — the
first hint the model wasn't learning at all.

---

# Experiment 1 — Component-weighted KGE loss + learnable Muskingum X

## Experiment
Full run `2026-06-17T10-07-51Z-global-train-and-test`. New `kge` loss with
independent r/α/β weights (α-weight = 2 to emphasize amplitude restoration) +
make Muskingum X a learnable per-reach KAN output (range [0, 0.5]; 0 = max
attenuation, 0.5 = pure lag).

## Hypothesis
The loss needs a **mechanism** to act on. Learnable X is the physical knob for
attenuation-vs-translation; an α-weighted KGE loss driving a learnable X should
de-attenuate and lift KGE.

## Changes
- `src/config.rs`: `LossKind::Kge`; `LossConfig` gains `r_weight`, `alpha_weight`,
  `beta_weight`, `kge_clamp`; `ParameterRanges.x_storage`.
- `src/training/loss.rs`: `kge_component_loss` — per-gauge
  `r_w·(r−1)² + α_w·(α−1)² + β_w·(β−1)² + nnse_w·(1−NNSE)`, with a per-gauge
  `kge_clamp` (default 10) — needed because collapsed-variance gauges (var_o≈0,
  eps=0.1) spiked batch loss to ~1.3e4 on the first attempt.
- `src/training/forward.rs`: source `x_storage` from the KAN's `x_storage`
  output (denormalized) when learnable; the custom sparse backward
  (`src/routing/mmc_op.rs`) already differentiates `x_storage`.

## Results
NSE 0.676 / KGE 0.698 — slightly **worse** than baseline and prior runs. Loss
stayed flat (epoch means ~1.2, no descent). α = 0.847 (unchanged from L1's 0.85).

## Notes
Clamp made the KGE loss numerically stable (no spikes), but the loss + knob did
not help. The flat loss + unchanged α said the parameters still weren't moving —
pointing the finger at the gradient itself, not the objective.

---

# Experiment 2 — Diagnostic: dump the learned X field

## Experiment
Extended `dump_parameters` to emit `x_storage`, dumped the learned X over all
2.94M reaches from the Experiment-1 final checkpoint.

## Hypothesis
If X is stuck near its sigmoid-init (~0.25), the gradient to the routing
parameters is ≈ 0 (an optimization failure); if X moved but α stayed bad, X is
structurally the wrong knob.

## Changes
- `src/dump_parameters.rs`: compute + write `x_storage` to the NetCDF and print a
  percentile summary.

## Results
X **stuck at init**: median 0.246, p10–p90 = 0.214–0.253, range [0.135, 0.257];
0% of reaches reached either bound despite α-weight = 2.

## Notes
Decisive: ∂loss/∂(routing params) ≈ 0. Combined with a literature sweep
(MC-LSTM, Harder et al. 2023 constraint layers, average-pooling gradient
dilution), this localized the cause to the forcing pipeline: daily Q′ is
upsampled to hourly by a **flat `repeat-24`** (`src/data/store/icechunk.rs::
daily_to_hourly_trim`), routed, then **averaged back to daily** for the loss —
so routing's within-day effect lands in the daily-mean's null-space and the
gradient vanishes.

---

# Experiment 3 — Learnable mass-preserving daily→hourly disaggregation head

## Experiment
Full run `2026-06-18T02-33-13Z-global-train-and-test`. Replace flat `repeat-24`
with a learnable head that gives each day a non-flat 24-hour shape while
preserving the daily mean (so routing has within-day structure to act on and the
gradient flows).

## Hypothesis
The daily→hourly interpolation + daily-mean aggregation is the gradient
bottleneck. A mass-preserving, learnable disaggregation will unstick it (loss
will descend, X will move) and let routing improve held-out skill.

## Changes
- `src/nn/disagg_head.rs` (new): `DisaggHead` — 3-tap windowed log-Q′ `[d−1,d,d+1]`
  (+ static attributes) → MLP → softmax over 24 hours → `daily·24·shape`. Daily
  mean conserved by construction; non-negative. Output init **non-flat** (xavier)
  so X gets gradient from step 1 (a zero-init = exact repeat-24 caused a
  chicken-and-egg slow start in an earlier smoke). Handles train trim
  `(rho−1)·24` and test `n_days·24` via an `n_hourly` arg with edge-clamped
  windows.
- `src/nn/kan_head.rs`: embed `disagg: Option<DisaggHead>` inside `KanHead` so the
  optimizer / checkpoint / eval / dump generics flow unchanged (no
  `CombinedHead` cascade). `KanHeadConfig` gains `disagg_*` fields.
- `src/data/dataset.rs`: carry daily Q′ (`q_prime_daily`) through
  `RoutingBatch`/`RoutingTensors` (flow-scaled), keeping the hourly `repeat-24`
  for the no-disagg path.
- `src/training/forward.rs`: when `head.disagg` is present, disaggregate
  `q_prime_daily` → hourly; else use the existing `q_prime`.
- `src/config.rs`: `kan_head.disaggregation` YAML section + a `kan_config` helper
  so every head-build site (train/eval/dump) builds an identical template.

## Results
- **Training loss DESCENDED for the first time ever** (epoch mean 1.224 → 1.02,
  median 1.220 → 0.987) vs dead-flat ~1.2 in every prior run.
- **X MOVED for the first time**: median 0.246 → 0.217 (range now [0.093, 0.246]).
- **But held-out metrics regressed**: NSE 0.680 (≈ prior), KGE **0.624** (down
  from ~0.70), α **0.761** (worse; 81% over-attenuated). X moved the *wrong* way
  (↓ = more attenuation) despite α-weight = 2.

## Notes
Hypothesis **confirmed**: the interpolation was the gradient bottleneck —
disaggregation is the only change that ever made the loss descend and X move.
But loss-down + test-metric-down = **overfitting**: the expressive,
sub-daily-unsupervised disagg learns training-period within-day shapes that drive
more attenuation and don't generalize (the risk flagged in the design plan).
Parity preserved throughout — with disagg disabled the path is byte-identical
(`compare_ddr_sandbox` unchanged; KAN-head parity tests pass).

---

# Experiment 4 — Early-stopping sweep

## Experiment
Evaluated end-of-epoch checkpoints (epochs 1–5, plus the epoch-10 final) of the
Experiment-3 run on the held-out test window to find where KGE peaked.

## Hypothesis
If Experiment 3's regression is overfitting, KGE was higher early and degraded
with training; an early checkpoint should recover it (and maybe beat baseline).

## Changes
None (evaluation only, via the `eval` binary on each checkpoint).

## Results
| epoch | NSE | KGE |
|---|---|---|
| 1 | 0.648 | 0.665 |
| **2** | **0.681** | **0.671** ← KGE peak |
| 3 | 0.686 | 0.644 |
| 4 | 0.687 | 0.650 |
| 5 | 0.684 | 0.639 |
| 10 | 0.680 | 0.624 |

NSE plateaus (~0.685) after epoch 2 while KGE degrades monotonically — textbook
overfitting. Best checkpoint = `epoch_2_mb_80` (0.681 / 0.671).

## Notes
Early stopping recovers KGE (+0.047 over epoch 10) and confirms overfitting, but
**even the best checkpoint loses to both** the baseline (0.689 / 0.723) and the
simpler no-disagg run (0.684 / ~0.70) on both metrics. So fixing the optimization
barrier exposed a **structural ceiling**: at daily resolution over already-UH-
routed forcing, learnable channel routing has no generalizable held-out skill to
add beyond summed-Q′ — more model capacity just overfits faster.

---

# Summary & next steps

**All three diagnostic layers are now empirically pinned:**
1. The **loss** was never the limiter (L1, NNSE-KGE, component-KGE → same flat
   result).
2. The daily→hourly **interpolation** was the gradient limiter (disaggregation →
   loss descends, X moves — both firsts). ✓ fixed.
3. Underneath sits a **structural ceiling**: daily-resolution routing over
   pre-UH-routed forcing yields no generalizable skill beyond the no-routing
   baseline.

**To genuinely beat summed-Q′, change the problem, not the model capacity:**
- **Sub-daily observations** (even a CONUS subset) to supervise the
  disaggregation and make routing's timing matter — the single change that turns
  the now-working gradient into generalizable skill.
- **Less-pre-routed forcing** (hillslope runoff) so MC does the channel routing
  instead of double-routing already-UH-routed Q′.
- If keeping the disagg head: **regularize** it (attribute-static shape +
  smoothness/flatness prior) and expect parity-at-best, not a win.

**Code state:** all of the above (KGE loss, learnable X, disaggregation head,
x_storage dump) is committed on branch `routing`. Enabled locally via
`ddrs.yaml` (`experiment.loss.kind: kge`, `x_storage` in
`kan_head.learnable_parameters`, a `kan_head.disaggregation` block). Best
generalizing checkpoint from the disagg run: `epoch_2_mb_80`.
