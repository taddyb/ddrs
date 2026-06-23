# Precip-driven disaggregation — first CONUS train-and-test findings (2026-06-23)

First end-to-end run of the **precip-driven mass-preserving daily→hourly
disaggregation head** (real hourly AORC precip drives the within-day shape;
daily mean conserved exactly). Branch `hourly-forcings`; source group
`conus-hourly`; loss **L1**; `cuda_graphs: false`. Design:
`docs/superpowers/specs/2026-06-22-precip-driven-disaggregation-design.md`.

## Result (run `2026-06-23T02-49-12Z-conus-hourly-train-and-test`)

Matched set = the **2365 CONUS gauges the model predicts**, trained vs **this
run's own summed-Q′ baseline**, identical metric code, common test window
(1995-10..2010-09):

| reference | median NSE | median KGE |
|---|---|---|
| **Trained (precip-disagg, L1)** | **0.7152** | **0.7106** |
| summed-Q′ baseline (no routing) | 0.6781 | 0.7172 |
| **Δ (trained − baseline)** | **+0.037** | **−0.007** |

- Trained **beats baseline on NSE** (+0.037; 57% of gauges improve).
- Trained **ties/slightly regresses on KGE** (−0.007; 47% improve).
- 2365/2365 gauges finite; predictions 0% NaN.
- Training loss **descended** (epoch-mean L1 9.15→8.30→7.89) then rose
  (8.60→9.36) — the disaggregation unsticks the gradient (the journal's
  predicted effect), with epoch-4/5 the same overfitting Exp-3/4 showed.

## Interpretation

This **reproduces the exact signature the 2026-06-19 journal diagnosed**:
routing improves NSE but loses on KGE via the variance-ratio term, because the
**L1 objective rewards a simulated variance below observed** (over-attenuation;
NSE's optimum sits at α<1). Real precip *timing* unstuck the gradient and
delivered a clean NSE win, but did **not** break the KGE ceiling — the binding
constraint there is the *loss*, not the forcing. The journal's prescription for
KGE (the `kge`/`nnse-kge` α-restoring loss) was deliberately excluded here for a
clean L1 comparison; combining precip timing **with** that loss is the natural
next experiment.

## Precip-off control (run `2026-06-23T13-03-35Z-train-and-test`)

Identical config except `use_precip: false` + no `aorc_precip` (daily-Q-only
disagg, the journal's Exp-3 mechanism); same seed → **same gauge batches**, so
it's a paired comparison. Three-way, 2365 matched gauges, same metric code:

| config | median NSE | median KGE |
|---|---|---|
| **precip-ON** (disagg + precip) | **0.7152** | **0.7106** |
| **precip-OFF** (disagg only) | 0.6957 | 0.6926 |
| baseline (summed-Q′) | 0.6781 | 0.7172 |

Decomposition:

| effect | Δ NSE | Δ KGE |
|---|---|---|
| **precip contribution** (ON − OFF) | **+0.0196** | **+0.0180** |
| disagg alone vs baseline (OFF − base) | +0.0176 | **−0.0245** |
| net vs baseline (ON − base) | +0.0372 | −0.0065 |

Per-gauge **paired** precip effect: NSE median +0.0049 (**58%** of gauges
improve), KGE median +0.0058 (**60%** improve) — a real, directional effect on
both metrics, not median noise.

### What the control reveals

- **Real precip timing helps on BOTH metrics** (+0.020 NSE, +0.018 KGE). It is
  not just the disaggregation mechanism doing the work.
- **The bare daily-Q disaggregation trades KGE for NSE** (−0.025 KGE / +0.018
  NSE vs baseline) — the journal's exact over-attenuation signature, from an
  *invented* within-day shape.
- **Precip rescues the KGE the bare disagg destroys** (0.6926 → 0.7106, +0.018),
  bringing net KGE to within −0.007 of baseline. So the residual KGE gap is
  **entirely the disaggregation's over-attenuation, not precip** — precip is the
  one component pushing KGE the right way.
- **Conclusion:** precip timing is genuinely valuable. To also beat baseline on
  KGE, pair precip with the α-restoring `kge`/`nnse-kge` loss to remove the
  disagg's residual over-attenuation — now strongly motivated.

## Critical bug found & fixed en route (commit a5972d9)

The **first** run produced **0/2365 finite NSE (all NaN)**. Root cause: the AORC
`total_precipitation` array carries **real NaN** (~14% of values: ocean /
no-coverage catchments / missing hours) despite a `0.0` `fill_value` — only
never-written chunks materialize the fill. NaN flowed through
`normalize_precip`'s `log1p` → NaN softmax → NaN forcing → NaN routing.

**`use_cuda_graphs: true` masked it**: printed training losses looked finite
(graph replay returns a stale loss scalar) while NaN gradients silently
corrupted the weights, surfacing only at eval. Diagnosis chain: baseline finite
→ training "finite" → checkpoints NaN from mb_4 → with graphs **off**, loss is
NaN from mb_0 → AORC scan shows `min=max=nan`, 7.07M NaNs in one sampled week.

Fix: zero-fill non-finite precip at `AorcPrecipStore` read time (matches the
coverage-gap intent) + defensive `max(0.0)` before `log1p`. **Lesson:** with
`cuda_graphs: true`, a finite printed loss is **not** proof of a finite forward
— validate with graphs off, or trust the eval/checkpoints.

## Next steps

1. **precip-off CONUS control** (`use_precip: false`, else identical) — isolates
   precip's contribution to the +0.037 NSE.
2. **precip timing + `kge`/`nnse-kge` loss** — the α-restoring term to convert
   the timing signal into a KGE win, not just NSE.
3. Early-stopping sweep (epoch-mean rose after epoch 3 → likely an earlier
   checkpoint generalizes better, à la the journal's Exp-4).
