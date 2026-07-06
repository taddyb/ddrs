# Riverbed Leakance is Not Identifiable from Gauged Discharge — Scientific Summary

Date: 2026-07-06
Branch: `worktree-zeta-sensitivity` (worktree off master).
Campaign docs (chronological):
`docs/2026-07-01-leakance-hourly-findings.md` (2×2),
`docs/2026-07-02-leakance-diagnosis-findings.md` (diagnosis),
`docs/2026-07-03-zeta-gradient-probe-findings.md` (gradient probe),
`docs/2026-07-04-synthetic-recoverability-findings.md` (recovery control),
`docs/2026-07-06-phase-c-findings.md` (promotion gate).
Literature: `docs/2026-07-04-leakance-literature-review.md` (32 verified citations).

**One-line verdict: riverbed leakance (the GW–SW exchange term zeta) is NOT
identifiable from gauged discharge and is NOT promotable — established by
removing, one experiment at a time, every alternative explanation (gradient
starvation, objective noise, uninformative inputs, sign ambiguity). It remains
active and marginally skill-improving in aggregate, but its learned spatial
field is anti-correlated with independent groundwater data. This is the
extreme, fully-controlled case of the paper's selective-equifinality thesis.**

---

## 1. Claim and its place in the thesis

The differentiable routing paper ("Beyond Equifinality in Differentiable River
Routing," Bindas & Shen) advances **selective equifinality**: in a
differentiable Muskingum-Cunge router, some learned parameters are identifiable
(shaped by geomorphic physics — channel geometry p, q, top width) and others
are compensatory bias-absorbers (Manning's n, which shifts to offset errors in
lateral inflow rather than representing roughness truth). The forthcoming core
experiment tests this by training on four structurally different lateral-inflow
sources (two LSTM, two dHBV2.0) and asking which parameters converge across
sources (identifiable) versus diverge (compensatory).

Riverbed leakance is the **limiting unidentifiable case** that anchors one pole
of that spectrum. Unlike n (which is at least gauge-constrained enough to
absorb bias), zeta cannot be constrained by gauges at all — because the
observation operator that maps per-reach flux to gauge discharge is a network
sum, and a sum is not invertible for its addends. Establishing this rigorously,
with every confound removed, is what makes the identifiability axis of the
thesis credible: we show a term can be physically real, well-parameterized,
and still fundamentally unlearnable from the available observations.

## 2. The term

Ported from a losing-stream correction (MODFLOW-family Darcy conductance;
Harbaugh 2005 RIV package; Niswonger & Prudic 2005 SFR2), subtracted from the
routing right-hand side each timestep:

```
zeta = leakance_factor · area_z · K_D · (depth − d_gw)     [m³/s, positive = losing]
```

Functional form lit-verified as standard (§Literature). Gradient-exact backward
op (`TimestepLeakanceOp`), matched to finite differences.

## 3. Evidence chain — every alternative explanation removed

Each row is an experiment that eliminated a rival account of "why zeta is
small / unrecovered." Verdicts are SUPPORTED / REFUTED per the house standard.

| # | Rival explanation for non-identifiability | Experiment | Verdict | Key number (units, gauges) |
|---|---|---|---|---|
| A | The parameter box (K_D ceiling) is too small | Diagnosis (2026-07-02) | **REFUTED** | 71.5% of reaches can exceed 0.01 m³/s inside the box; median utilization 3.4% |
| B | The KAN head collapsed (no learned variance) | Diagnosis | **REFUTED** | K_D–aridity ρ +0.61, d_gw–meanP ρ +0.71 (strong learned structure) |
| C | Gradients never reach ungauged/losing reaches (starvation) | Gradient probe (2026-07-03) | **REFUTED** | gauged/ungauged \|∂L/∂factor\| ratio 1.5× (trained), not ≥10× |
| D | Real losses are large enough to see at gauges | Detectability probe (2026-07-03) | **REFUTED** | 0.01 m³/s loss is 53× below the median 5% discharge-uncertainty band; 4.2% of Ref probes detectable |
| E | The training objective's noise floor masks the signal | Recovery control on noisy vs clean objective (2026-07-04/05) | **REFUTED as the cause** | floor fixed on losing gauges (1.5→0.11 m³/s) yet recovery unchanged: R1 = 0.009 → 0.008 |
| F | Uninformative inputs starve d_gw of a groundwater signal | Phase C enriched inputs (2026-07-06) | **REFUTED as sufficient** | 17 inputs incl. channel bed-relative WTD, permeability; field still anti-physical |
| G | Sign ambiguity lets zeta double-count baseflow | Phase C losing-only clamp + impervious hard-zero | controlled | clamp/hard-zero active; did not rescue identifiability |

**The surviving explanation** (SUPPORTED, by elimination and by direct
measurement): the gauge observation operator is a network integral. A gauge
observes Σ(zeta) over its entire upstream network, so training constrains the
*aggregate* upstream loss but carries no information about its *per-reach
distribution*. Many spatial fields yield the same gauge signal and hence the
same loss; the optimizer selects a smeared minimum-norm field, not the true
localized one.

## 4. Direct measurements of the smearing (the mechanism, not just the outcome)

**Recovery control, clean objective (synthetic obs with 58 planted losing
reaches; state-cache hotstart objective, floor ≤0.11 m³/s on losing gauges):**
- Recovery ratio R1 = **0.008** (median learned/planted zeta_net at planted
  reaches; bar for recovery ≥0.5). REFUTED.
- The term is heavily USED, not suppressed: active on **78.2%** of 64,892 eval
  reaches, Σ\|zeta\| = **1485 m³/s** network-wide; loss gap A(ON) 0.50 vs
  B(OFF) 1.74 (+71%).
- Yet the 58 planted reaches hold **0.1%** of that flux; the 10 largest-zeta
  reaches include **zero** plants. The optimizer fit the integrated loss by
  spreading, not by localizing.

**Promotion gate on real USGS gauges (2365 gauges, eval window
1995/10–2010/09; ON vs OFF, same enriched inputs):**
- Leg 1 (skill): losing-subset ΔKGE **+0.006** (73.4% of affected gauges
  improve), ΔNSE +0.0015 — below the +0.01 promotion bar; overall improves
  (ΔNSE +0.0011, ΔKGE +0.0046), no harm. Leakance genuinely helps aggregate
  fit, ~3× the 2×2's +0.0018, but marginally.
- Leg 2 (equifinality): Δn IQR **0.0143** (bar <0.1; daily anti-pattern 0.59),
  ρ(Δn, zeta_net) +0.079. PASS — leakance did NOT cannibalize Manning's n.
- Leg 3 (external consistency): ρ(\|zeta\|, bed-relative water-table depth) =
  **−0.355** (bar >0.3, and negative); only 15.8% of active-zeta reaches are
  losing-possible by the water table. FAIL — the learned field is
  anti-correlated with groundwater reality.

Gate verdict: **NO-GO for promotion** (Leg 1 sub-threshold, Leg 3 fails).
Reference context: no trained ddrs configuration beats the summed-Q′ CONUS
baseline (0.6781 NSE / 0.7172 KGE, 2365 gauges) on KGE as of 2026-07-06;
leakance does not change that.

## 5. What this supersedes

- The 2×2 "GO — marginal" verdict (`docs/2026-07-01-leakance-hourly-findings.md`)
  stands as a description of the term's *aggregate skill* (active, non-collapsed,
  marginally helpful) but is **superseded as an identifiability or promotion
  claim**: the term is active and skill-positive, and simultaneously
  unidentifiable and not promotable.
- The "Phase B objective fix is required before an identifiability claim"
  status (skill/CLAUDE guidance, 2026-07-05) is **resolved**: Phase B was
  built and run; the identifiability control still failed on the clean
  objective. Identifiability is REFUTED, not pending.
- The "widen K_D past 1e-6" follow-up remains superseded (diagnosis §4.2):
  K_D was widened to [1e-8, 1e-5] in Phase C and the field is still anti-physical.

## 6. Implication and the path that is NOT gauge supervision

Because the obstruction is the observation operator (a network sum), no
objective computed from gauged discharge — L1, NSE, KGE, or otherwise — can
identify per-reach zeta. This is a property of the data, not the optimizer or
the parameterization, and it is invariant to more training data (contra the
scaling-effect optimism of Tsai et al. 2021; Feng et al. 2022). Promotion of
leakance as a physical parameterization would require supervision **outside**
gauge discharge — a spatial prior constraining zeta_net / d_gw against
independent groundwater data (Jasechko et al. 2021 well-vs-stream levels; Zell
& Sanford 2020 CONUS depth-to-water; ParFlow GW–SW flux, Yang et al. 2025).
The channel/groundwater attribute layer built in Phase A already supplies the
fields such an auxiliary loss would need. Whether to pursue that is a research
decision; the present result argues for documenting the identifiability limit
rather than engineering around it.

## 7. For the paper

Leakance is the **negative control that validates the identifiability axis**:
a physically-motivated term, correctly parameterized, given informative inputs
and a de-noised objective, that gauge supervision provably cannot localize.
Placed against the forthcoming four-inflow experiment, it defines the
unidentifiable pole — the extreme that Manning's n approaches (bias-absorber)
and that geometry departs from (geomorphically constrained). The general
statement: **identifiability in differentiable routing is bounded by the
observation operator, not the optimizer or data volume.** Report as REFUTED
identifiability with the mechanism, not as a failed feature.

House-style note for downstream drafts: write "leakance is active and
non-collapsed under hourly forcing but not identifiable from gauged discharge
(recovery ratio 0.008; field–WTD ρ −0.36)" — never "leakance is identifiable,"
and never frame the NO-GO as "we chose not to promote it."
