# H5/H6 parameter-transfer and loss-landscape — v2 findings (audit-corrected analysis)

Date: 2026-07-09. Supersedes ONLY the analysis and interpretation of
`docs/2026-07-09-h5-h6-equifinality-findings.md` — **no new Rust runs, no new
compute**. Same raw CSVs (`output/equif/h5/registered/*.csv`,
`output/equif/h6/*_{surface,barrier}.csv`, seed 42), re-analyzed with
`scripts/h5_h6_audit_analysis.py`. Spec:
`docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md`.

**Verdicts unchanged: H5 INCONCLUSIVE, H6 INCONCLUSIVE.** What changes is the
evidentiary basis: H5's INCONCLUSIVE is now *certified* (the registered
split-half noise floor was computable from the existing per-window CSVs all
along, and SUPPORTED definitively fails on the R3 direction), and H6's surface
is a strongly anisotropic n–p valley whose registered sublevel metric
saturated — not the "sharp, roughly isotropic bowl" the v1 doc described.

## 1. What this pass corrects

1. **v1 §3.1's noise comparison is a statistical error.** It compared P_n
   (~0.1) against the ~4.2–4.5 m³/s *window-to-window* std. But every
   composition was evaluated on identical windows (the seeded-plan-replay
   design guarantees it), so the correct test is paired: per-window
   differences, whose variance is 15–40× smaller. Three of the four P_n values
   are statistically significant (§2).
2. **The "missing" split-half noise floor was never missing.** The spec
   registers it as "split-half over windows" — computable from the per-window
   CSVs, no seed-123 rerun required. Computed here (§2): H5's SUPPORTED bar
   fails definitively (R3 direction: P_n negative, 1.9× floor).
3. **The registered control comparison was never reported.** The
   source-disagreement effect — the design's own reason for running R1↔R2 —
   comes out ≈ 0 or negative: the low-disagreement control penalty (+0.125)
   *exceeds* the primary penalty (+0.095) (§3).
4. **v1 §3.2's surface characterization is wrong.** The 5%-sublevel was read
   as loss ≤ min×1.05; 5% of a ~10 m³/s loss (0.49) is ~30% of the entire
   grid's loss range, so the "sublevel set" swallowed 100–105/121 grid points,
   and the PCA aspect of a nearly full square grid is ~1 by construction. The
   instrument saturated; the underlying surface is a valley with 44–65×
   curvature anisotropy — stiff along α (n-scale), sloppy along β (p-scale)
   (§4).

## 2. H5 — paired per-window statistics and the registered noise floor

Paired differences d = L(composition) − L(own) per window, 96 windows
(`scripts/h5_h6_audit_analysis.py`, section 1):

| Run | P_n | paired SE | t | P_geo | paired SE | t | floor(P_n)† | P_n/floor |
|---|---|---|---|---|---|---|---|---|
| R1←R3 (primary) | +0.0953 | 0.0261 | **+3.65** | −0.0059 | 0.0124 | −0.48 | 0.0189 | 5.1× |
| R3←R1 (primary) | −0.0279 | 0.0188 | −1.49 | +0.0091 | 0.0035 | **+2.60** | 0.0149 | 1.9× |
| R1←R2 (control) | +0.1251 | 0.0202 | **+6.20** | −0.0095 | 0.0096 | −0.99 | 0.0366 | 3.4× |
| R2←R1 (control) | +0.0961 | 0.0317 | **+3.03** | −0.0065 | 0.0101 | −0.64 | 0.0387 | 2.5× |

† |P_n(even windows) − P_n(odd windows)|; the spec registered "split-half over
windows" without fixing the split, so first/second-half floors are also
reported by the script (0.0308 / 0.0326 / 0.0656 / 0.0900 — same conclusions).
Additivity holds throughout (P_full ≈ P_n + P_geo within 0.02).

**Bar application.** SUPPORTED requires f_n ≥ 2/3 under BOTH forcings AND
P_n ≥ 3× the noise floor. The R3 direction fails both ways: P_n is *negative*
(R1's n slightly improves R3's loss; t = −1.49, not significant) and only
1.9× its floor. SUPPORTED is therefore definitively closed, not pending.
REFUTED (f_n ≤ 1/2 under either forcing) does not trigger literally —
f_n(R3←R1) = 1.485 — but only through a sign pathology the registered formula
did not anticipate: with P_n = −0.0279 and P_geo = +0.0091, both numerator and
denominator are negative, so f_n > 1 despite geometry carrying *more* transfer
damage than n under R3's forcing, which is the bar's stated refutation intent
verbatim ("geometry carries at least as much transfer damage as n").
Per pre-registration discipline the literal verdict stands:
**H5 INCONCLUSIVE**, with the formula pathology disclosed. Any v2 registration
must put bars on P_n and P_geo separately (or use a sign-aware f_n).

## 3. H5 — the control result and gauge concentration (new)

The design's rationale for the R1↔R2 pair was that a shared daily-LSTM store
implies low Q′ disagreement, so its swap penalties should be small and the
R1↔R3 excess would isolate the source-disagreement effect. Observed: the
control n-swap penalty into R1 (+0.1251, t = +6.2) **exceeds** the primary
(+0.0953, t = +3.7). The source-disagreement effect is ≈ 0 or negative — any
foreign n hurts R1 by roughly the same amount regardless of whether the
donor's forcing agrees. This is the opposite of the forcing-bound-compensator
prediction and is the single sharpest fact in the H5 data.

Per-gauge structure (paired per-gauge means, 2,340 gauges joined to DA via
`~/projects/ddr/references/gage_info/gages_3000.csv`):

| Run | median per-gauge d_n | frac gauges hurt | top-10-gauge share of Σd_n | ρ(log10DA, d_n) |
|---|---|---|---|---|
| R1←R3 | −0.0000 | 0.45 | **82%** | +0.050 |
| R3←R1 | +0.0001 | 0.57 | sign-mixed (same gauges, negative) | +0.015 |
| R1←R2 (control) | +0.0021 | 0.63 | 41% | +0.058 |

The primary pair's network-mean P_n is an outlier-gauge statistic: 10 of
2,340 gauges (0.4%) carry 82% of the summed penalty, and the same large-river
gauges recur across runs and directions with flipped signs (03374000,
03341500, 08025360, 06926000, 03320000, 05464500 — the gauges R3's n hurts
under R1's forcing are the ones R1's n *helps* under R3's forcing). The
typical gauge is indifferent to the swap (median ≈ 0). The control swap, by
contrast, inflicts a broader-based penalty (63% of gauges hurt, positive
median) — consistent with R2's collapsed n distribution (v2 LSTM findings
§4.1) being a poor fit for most reaches, and again unrelated to forcing
disagreement. DA-stratification shows no meaningful gradient (|ρ| ≤ 0.06;
what little penalty the median gauge shows concentrates in the largest-DA
quintile).

## 4. H6 — corrected surface geometry

From the raw per-window grid (`scripts/h5_h6_audit_analysis.py`, section 3):

| Metric | R1 forcing | R3 forcing |
|---|---|---|
| grid min (α*, β*) | 9.8523 at (+0.3, +0.9) | 10.6732 at (−0.3, −0.9) |
| grid loss range | 1.7057 (17.3% of min) | 1.3536 (12.7% of min) |
| sublevel ≤ min×1.05 (v1's reading) | 100/121 pts, aspect 1.22 | 105/121 pts, aspect 1.15 |
| sublevel ≤ min + 5% of range | 35/121 pts, aspect **3.22** | 36/121 pts, aspect **2.92** |
| L*(β) range over 8× p-scaling | 0.0277 | 0.0302 |
| curvature ratio, stiff α : floor β | **44×** | **65×** |
| valley-floor slope dα*/dβ | +0.164 | +0.109 |
| split-half minima displacement | 0.424 | 0.000 |

The surface is a valley: the profile L*(β) is flat to within 0.03 m³/s across
the full 8-fold range of global p-scaling under both forcings (β is the sloppy
direction — the loss barely constrains global top-width scale), while L*(α)
rises ~0.5 across the same span (α, the n-scale, is the stiff direction).
The 44–65× curvature anisotropy corresponds to a sublevel-ellipse aspect of
~7–8:1 — far past the 3:1 bar — and even the milder range-based sublevel gives
3.22/2.92, straddling it. The registered min×1.05 reading saturates the grid
and returns aspect ≈ 1 by construction. So the degeneracy leg of H6 did not
"fail outright" (v1 §3.2); it was measured with a metric that cannot detect a
valley whose depth is small relative to the absolute loss level. Because the
registered operationalization is what it is, **H6 remains INCONCLUSIVE** —
but the failure is in the instrument definition, not evidence of an isotropic
bowl, and v1's descriptive reading ("sharp basin whose floor moves →
forcing-specific identification") is correct only along the α axis.

The displacement claim inverts accordingly. Of the 1.8974 log2-unit
cross-forcing displacement, Δβ = 1.80 lies along the flat axis, where an
argmin is ill-conditioned by definition (R1's own split-half β* wandered by
0.3–0.4 grid units; the loss differences involved are ~0.003). The
loss-relevant component is Δα = 0.60: R1's forcing prefers global n scaled
2^{+0.3} above the R1/R3 anchor and R3's 2^{−0.3} below — a real,
forcing-indexed ~1.5× difference in preferred n level, independently matching
the ~40–50% n level divergence of the learned fields (H1). Cross-evaluating
each forcing's grid minimum under the other forcing (paired over the 16 shared
windows): +0.0408 (t = +1.10, not significant) under R1, +0.0642 (t = +2.43,
marginal) under R3. The floor moves, but at small and only marginally
resolvable loss cost at this sample size.

**Per-axis reading of the registered 2×2** (descriptive, not a verdict):
along α (n), comparatively sharp basin + floor moves ~1.5× with forcing →
"forcing-specific identification (structural input error)". Along β (p), flat
valley + floor position meaningless → p is **sloppy** in the Chis et al.
(2016) sense. This is the first landscape-level explanation for the one
robustly divergent geometry quantity in the H1–H4 campaign: top_width's
0.38–0.41 cross-arm spread (v1 LSTM findings §5 item 4) diverges because the
loss surface does not constrain global p — the suspected p [1, 200] log-box
artifact is better described as genuine p-insensitivity of the routing loss.

## 5. Internal consistency checks (all pass)

- Barrier endpoint delta under R1 forcing on the 16-window plan (+0.0910)
  reproduces H5's P_full on the independent 96-window plan (+0.0908); R3
  direction −0.0223 vs −0.0138, same sign and magnitude class.
- H6's Δα = 0.6 (factor 1.52) independently matches H1's ~40–50% n level
  divergence between the learned R1/R3 fields.
- v1 doc's f_n values, minima locations, and B = 0 barriers all reproduce
  exactly from the raw CSVs.
- The anchor (0,0) sits within +0.038/+0.012 of each surface minimum — the
  arm-mean anchor construction is sound.

## 6. Revised interpretation

The compensator reading of Manning's n has now failed every direct test:
H2 refuted at two axes × two scales (v2 LSTM findings §3), and H5 shows no
source-disagreement effect (control ≥ primary), asymmetric transfer (R3
accepts R1's n for free), and outlier-gauge concentration in place of a
distributed forcing-specific signal. What the landscape adds is a positive
statement: n's global level is the *stiff*, comparatively identified direction
of this model, and its optimum is forcing-indexed by ~1.5× — structural input
error shifting a well-defined minimum, not a wide basin hiding a shared one.
The genuinely sloppy direction is geometry's p (global top-width scale),
loss-flat over 8× — inverting the campaign's original framing, in which
geometry was the hypothesized-identifiable class and n the suspect. The
sharper statement for the paper: identifiability lives in realized conveyance
(depth, hydraulic radius — which do converge, H1 value-level), while the
(n, p) parameterization contains one stiff combination and one nearly free
direction. Neither H5 nor H6 should be cited for or against the compensator
thesis (both INCONCLUSIVE as registered); the p-sloppiness and the
forcing-indexed n level are citable as measured facts with the caveats above.

## 7. Next steps (revised from v1 §5)

1. **Finer-α landscape, not wider-β** (replaces v1 item 3): β is flat —
   scanning it wider adds nothing. Scan α at 0.1 steps over [−0.6, +0.6] at
   96 windows to pin each forcing's α* with a real error bar; that directly
   measures the forcing-indexed n level, the one loss-relevant displacement.
2. **Robust-penalty H5 re-analysis** (new): per-gauge median / trimmed-mean
   penalties to test whether any broad-based transfer effect survives removal
   of the ~10 dominant gauges; also inspect those gauges (all large rivers)
   directly.
3. **Seed-123 replicate** (v1 item 1, demoted from gate to confirmation): the
   registered split-half floor already exists (§2); an independent-seed floor
   is confirmatory.
4. **Bar repairs before any v2 registration** (new): sign-aware f_n or
   separate P_n/P_geo bars; range- or curvature-based degeneracy metric with
   the sublevel definition fixed in the spec text.
5. **Per-reach landscape** (v1 item 6, upgraded): the gauge concentration in
   §3 says whatever transfer structure exists is spatially localized —
   invisible to global (α, β) scaling by construction. Better motivated now
   than v1's list suggested.
6. **dHBV2 cross-family arms** — unchanged, still the campaign bottleneck.

## 8. Reproduce

```bash
cd ~/projects/ddrs/ddrs-py && uv run python ../scripts/h5_h6_audit_analysis.py
```

Reads only the registered raw CSVs; every number in §§2–5 appears in its
stdout. The v1 doc's §7 commands regenerate the CSVs themselves if needed.
