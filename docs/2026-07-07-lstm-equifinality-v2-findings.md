# LSTM-source selective-equifinality — v2 findings (audit-corrected rules)

Date: 2026-07-07. Branch: `unit_catchments`. Governed by
`docs/superpowers/specs/2026-07-07-lstm-equifinality-v2-analysis-design.md`.
Supersedes ONLY the analysis rules of the original findings doc
(`docs/2026-07-07-lstm-equifinality-findings.md`) — no retraining, no new
arms. That doc's Methods (§2), arms, run IDs, and raw §6 script output remain
the reference for setup; this doc reports what changed under the v2 rules
and the two analyses the audit flagged as untested.

**No new compute.** Same three checkpoints (`2026-07-07T03-55-53Z /
04-49-19Z / 06-50-28Z-train-and-test`, sha `b261e1d`), same
`R{1,2,3}_kan_parameters.nc`, same `grad_R{1,2,3}.nc` (seed 42) +
`grad_R{1,2,3}_seed123.nc` (noise ceiling). Only
`scripts/equif_convergence_analysis.py` changed (+561/−54 lines: network-scale
H2, Level 1.5, N-arm-ready helper refactor — `git log` this branch for the
diff).

## 1. What this pass adds over the v1 findings doc

The v1-registered verdicts (H1–H3 REFUTED, H4 INCONCLUSIVE) and the audit's
items 1–3 (dual-metric H1/H2, timing-axis H2, CM-removed H3 + noise ceiling)
were already complete before this pass — see v1 doc §8. This pass answers
the audit's two remaining open items:

1. **Network-scale H2** — does the bias-absorber mechanism test hold up when
   Q′ disagreement and n-divergence are aggregated over each gauge's full
   upstream network, rather than judged reach-by-reach?
2. **Level 1.5** — per-arm parameter distributions and drainage-area (DA)
   conditioning, closing the descriptive gap the audit's addendum opened by
   hand (`log10_uparea` join, done here as a cached pipeline stage instead of
   an ad hoc recomputation).

## 2. Verdict table (unchanged v1-registered + new v2 tests)

| # | Hypothesis | v1-registered verdict | v2 status |
|---|---|---|---|
| H1 | Geometry converges, n diverges | **REFUTED** (unchanged) | Value-level reverses under like-for-like metric (§3); mechanism untouched |
| H2 | n-divergence tracks Q′ disagreement | **REFUTED** (unchanged) | REFUTED at 2 axes (volume, timing) × 2 scales (reach, gauge-network) — strengthened, not just replicated |
| H3 | Selective gradient alignment | **REFUTED** (unchanged) | n's alignment shown entirely common-mode; instrument noise-limited (v1 §8.3, unchanged this pass) |
| H4 | Gauge-distance decay, all params | **INCONCLUSIVE** (unchanged) | unchanged this pass |
| H2-network (new) | Same H2 mechanism at gauge-network scale | — | **REFUTED**, ρ = **−0.380** (6,888/8,945 gauges) |

`verdicts.json`'s top-level `verdicts` block is byte-identical to the v1 run
(`H1/H2/H3: REFUTED`, `H4: INCONCLUSIVE`) — confirmed by direct read, not
just re-derivation. `audit.H2_network` and `audit.Level1_5` are new keys;
nothing under `audit.H1`/`H2`/`H3` (the v1 audit items) changed.

## 3. Network-scale H2 (new)

Per-gauge aggregation: for each of 8,945 gauges, `n_disagreement(g)` = median
n rel-spread over the gauge's full upstream network; `timing_disagreement(g)`
= median `(1 − pearson_r)` over the same network (daily-lstm vs hourly-lstm
summed hydrographs). 6,888/8,945 gauges had enough upstream reaches to
compute both.

```
Spearman ρ(n_disagreement, timing_disagreement) = -0.380   (bar: > 0.2)
n rel-spread vs geometry rel-spread contrast: 0.4512 vs 0.2545  contrast=True
→ H2_network: REFUTED
```

This is the more negative of the four ρ values now on record (reach-volume
−0.248, reach-timing −0.233/−0.211, gauge-network −0.380). Aggregating to the
scale a bias-absorber mechanism would actually operate at (a gauge integrates
its whole upstream network's timing error into one calibration target) makes
the refutation MORE decisive, not less — this rules out "the reach-scale test
was too fine-grained to see the effect" as a rescue for H2. Whatever produces
n's ~40% level divergence across arms (§8.1 of the v1 doc), it is not
per-reach or per-gauge-network compensation proportional to inflow
disagreement, at any axis or scale tested so far.

## 4. Level 1.5 — distributions and drainage-area conditioning (new)

### 4.1 Per-arm percentiles

| Arm | n (p5/p50/p95) | q_spatial (p5/p50/p95) | p_spatial (p5/p50/p95) |
|---|---|---|---|
| R1 (daily flat) | 0.024 / 0.084 / 0.125 | 0.484 / 0.490 / 0.496 | 4.14 / 8.62 / 12.20 |
| R2 (daily disagg) | 0.077 / 0.100 / 0.117 | 0.431 / 0.457 / 0.480 | 5.74 / 8.21 / 10.98 |
| R3 (hourly native) | 0.025 / 0.065 / 0.107 | 0.366 / 0.431 / 0.475 | 3.05 / 6.02 / 10.41 |

R2's n distribution is visibly narrower than R1/R3 (p5–p95 span 0.040 vs
0.101/0.082) — the disaggregation head is the distributional outlier, not
the hourly-native arm, consistent with the v1 audit addendum's manual
recomputation. This reproduces exactly (percentiles match the addendum's
0.077–0.117 for R2 vs 0.024–0.125-ish spans for R1/R3).

### 4.2 Drainage-area-conditioned OLS slopes (ln-quantity vs log10-DA)

| Quantity | R1 | R2 | R3 |
|---|---|---|---|
| n | **+0.184** | **−0.027** | **+0.145** |
| q_spatial | +0.004 | −0.008 | +0.031 |
| p_spatial | +0.146 | −0.020 | +0.155 |
| depth | +1.361 | +1.362 | +1.301 |
| top_width | +0.814 | +0.602 | +0.715 |
| hydraulic_radius | +1.315 | +1.262 | +1.193 |

n's slopes reproduce the pre-registered targets exactly (+0.1838 / −0.0265 /
+0.1453 vs targets +0.184 / −0.027 / +0.145). R1 and R3 — the two
structurally distinct stores — share a near-parallel, POSITIVE n-vs-DA
relationship (roughness increasing downstream); R2 is flat. This is the
"shared slope, shifted intercept" signature the audit addendum described by
hand, now confirmed by the pipeline: R1/R3 agree functionally despite
different Q′ physics, while R2's disaggregation head appears to absorb
sub-daily timing error internally and relieve n of that role (consistent
with R2's H4 gradient anomaly, v1 doc §H4, and R2's narrow n distribution
above). The shared R1/R3 slope is opposite the classical downstream-smoothing
expectation (Leopold–Maddock-style hydraulic geometry predicts roughness
*decreasing* with drainage area) — geometry (depth, top_width,
hydraulic_radius) DOES follow the classical positive-with-DA direction and
agrees closely across all three arms (slopes 1.19–1.36 for depth/Rh, all
arms), which is itself a point in favor of geometry identifiability at the
functional-form level, even though its cross-arm SPREAD (§H1) is not smaller
than n's on the like-for-like metric.

### 4.3 Spread-vs-DA profile

Cross-arm rel-spread declines monotonically from headwaters to outlets for
every quantity:

| Quantity | headwater decile | outlet decile |
|---|---|---|
| n | 0.599 | 0.288 |
| q_spatial | 0.151 | 0.093 |
| p_spatial | 0.455 | 0.335 |
| depth | 0.278 | 0.176 |
| top_width | 0.458 | 0.376 |
| hydraulic_radius | 0.297 | 0.151 |

Cross-arm disagreement (in every quantity, not just n) concentrates in small,
ungauged headwater reaches and shrinks toward gauged outlets — consistent
with H4's gauge-locality finding (INCONCLUSIVE but trending toward gauge-near
identifiability for S/I cells) and a reminder that the median statistics
reported for H1/H2 average over a network where identifiability is
DA-dependent, not uniform.

## 5. Revised overall interpretation

**Still INCONCLUSIVE, unchanged from v1 §8.4** — this pass did not run the
compute (dHBV2 arms, longer-budget replicate, seed replicate) that would
resolve the shared-init/single-model-family confounds. What changed:

- **H2's REFUTED verdict is now materially stronger.** Four independent
  operationalizations (2 axes × 2 scales) all give ρ < 0, three of four with
  |ρ| > 0.2 in the refuting direction. The audit's "was the reach-scale test
  too fine-grained" question is answered no — if anything, network-scale
  aggregation sharpens the refutation. This is the paper's most defensible
  single result from the LSTM arms.
- **H1 remains the paper's least resolved claim.** Level 1.5 adds texture
  (R2 is the outlier, R1/R3 share a backward-signed but parallel n–DA slope,
  disagreement concentrates in headwaters) without resolving the direction:
  value-level evidence still leans toward the original thesis (n diverges
  ~40% in level; depth/hydraulic-radius converge within ~10% at common
  reference), while the functional-form evidence (§4.2, geometry's
  consistent classical-direction DA slope across arms) is a NEW point mildly
  favoring geometry identifiability that v1 did not have.
- **The two audit-robust facts from v1 stand unchanged:** the negative-ρ
  mechanism result (now quadruple-replicated across axis/scale, §3 above)
  and geometry-gradient orthogonality across distinct stores (raw R1–R3
  cosines 0.023–0.095, still far below noise ceilings 0.39–0.59).

The paper must still adopt NEITHER "n converges" nor "geometry converges" as
an established claim from the LSTM arms alone. What has changed since v1 is
that the H2 refutation is no longer a single-axis result open to a
scale-artifact objection — that objection is now closed. The remaining path
to resolution is unchanged: dHBV2 cross-family arms, a longer-budget
replicate, and a seed replicate (v1 doc §5 items 1–3; as of this writing none
of the three has been started — no `config/sources/*dhbv*` or `*seed*`/
`*budget*` experiment configs exist on this branch).

## 6. Reproduce

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/equif_convergence_analysis.py \
  --r1 2026-07-07T03-55-53Z-train-and-test \
  --r2 2026-07-07T04-49-19Z-train-and-test \
  --r3 2026-07-07T06-50-28Z-train-and-test \
  --params-r1 ~/projects/ddrs/output/equif/R1_kan_parameters.nc \
  --params-r2 ~/projects/ddrs/output/equif/R2_kan_parameters.nc \
  --params-r3 ~/projects/ddrs/output/equif/R3_kan_parameters.nc \
  --grads-r1  ~/projects/ddrs/output/equif_probe/grad_R1.nc \
  --grads-r2  ~/projects/ddrs/output/equif_probe/grad_R2.nc \
  --grads-r3  ~/projects/ddrs/output/equif_probe/grad_R3.nc \
  --grads-r1-rep ~/projects/ddrs/output/equif_probe/grad_R1_seed123.nc \
  --grads-r2-rep ~/projects/ddrs/output/equif_probe/grad_R2_seed123.nc \
  --grads-r3-rep ~/projects/ddrs/output/equif_probe/grad_R3_seed123.nc
```

All stages cache-hit except the first run after this pass's edits (Stage C
needed one `--force` to pick up the N-arm-ready helper refactor; subsequent
runs, including the new `stage_h2_network.npz` and `stage_g.npz` caches, are
plain reruns). Full stdout of the verifying run is quoted in the delegated
implementation's session log; the numbers above were read directly from
`output/equif/verdicts.json` and the script's own printed verdict block, not
retyped from memory.

## 7. Next steps (unchanged priority order, v1 doc §5)

1. dHBV2 cross-family arms — no configs exist yet; the only cross-model-family
   test.
2. Longer-budget replicate (15–20 epochs, R1 vs R3) — tests init-hugging.
3. Seed replicate (different seed, R1 + R3) — noise floor under every
   convergence statistic; upgraded from optional to essential by the v1
   audit.
4. Top-width/p box-sensitivity deep-dive — is the one robustly divergent
   geometry quantity (top_width 0.38–0.41 under every reference) genuine
   disagreement or an artifact of p's log-space `[1, 200]` box?
