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

## precip + nnse-kge loss (run `2026-06-24T00-03-01Z-conus-hourly-train-and-test`)

Tested the prediction above (#2): swap L1→`nnse-kge` (balanced, nnse_weight =
kge_weight = 1.0) to recover KGE via the `(α−1)²` restoring gradient. 4-way,
2365 matched gauges, identical metric code:

| config | median NSE | median KGE |
|---|---|---|
| precip + L1 | **0.7152** | **0.7106** |
| precip + nnse-kge | 0.7095 | 0.7100 |
| disagg-only + L1 (precip-off) | 0.6957 | 0.6926 |
| baseline (summed-Q′) | 0.6771 | 0.7171 |

precip+nnse-kge vs baseline: NSE **+0.032**, KGE **−0.007** — **not a dual win**.

### The L1-over-attenuation hypothesis was REFUTED

The prediction was that nnse-kge's α-term would lift KGE over baseline. It did
not: KGE moved essentially nowhere (0.7106 → 0.7100) and NSE dropped slightly.
For the precip-driven setup, **L1 stays the better objective** — nnse-kge did not
earn its keep on either metric.

**Corrected interpretation.** The residual ~−0.007 KGE gap is **not loss-fixable**
(at least not with balanced nnse-kge) — it is **baseline-dominated / structural**:
the summed-Q′ reference already has the highest KGE (0.717) of any config, and
*any* routing trades a large NSE gain (+0.032…+0.038) for a small KGE give-back.
This is the journal's "structural ceiling" showing up on the KGE axis: routing
already-dHBV-UH-smoothed Q′ cannot beat the no-routing baseline's KGE. The win
that *is* real and robust is **NSE via precip-driven timing** (+0.037), with the
learned roughness rising 0.050→0.068 (precip-off→on) to attenuate precip's
sharper sub-daily pulses.

The one untested loss lever is the **α-weighted `kge` component loss**
(`alpha_weight: 2`, the journal's Exp-1 setting) — now that the gradient actually
flows (precip), it *might* push α/KGE where balanced nnse-kge could not. But the
evidence leans toward the KGE ceiling being structural, not objective-driven.

## Next steps

1. **Per-gauge local-midnight daily aggregation.** `tau_trim_and_downsample`
   uses a fixed `13+τ:−11+τ` offset for *all* gauges; USGS daily is local
   standard time, so model vs obs daily bins are misaligned by the per-gauge
   UTC offset (−5…−8 h), smearing the cross-day routing timing. Likely the
   biggest remaining timing lever.
2. **Temperature as a 2nd disagg channel** (AORC already carries `temperature`)
   — snowmelt timing in western basins is temperature-, not precip-, driven.
3. **Hourly USGS IV** for a CONUS subset — the only way to *directly* supervise
   sub-daily timing (daily obs marginalize it away).
4. α-weighted `kge` loss (`alpha_weight: 2`) — last objective lever before
   accepting the KGE ceiling as structural.
5. Early-stopping sweep (epoch-mean L1 rose after epoch 3).
