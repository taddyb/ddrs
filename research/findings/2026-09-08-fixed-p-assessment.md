# Prescribing the width coefficient p: assessment and plan (2026-09-08)

**Prompt (user):** p (the Leopold–Maddock width coefficient) has little effect; make it a simple function, learn only
n and q, and draw the loss landscapes over (n, q). q is an exponent and therefore the more effective lever.
**Literature and provenance:** `docs/book/reference/hydraulic-geometry-literature-2026-09-08.md` (verified metadata only).

## 1. What p and q do in this model (`src/geometry.rs`)

```
depth     = ( Q · n · (q+1) / (p · √s) )^( 3 / (5 + 3q) )
top_width = p · depth^q
```
Log-sensitivities at the trained q ≈ 0.35: ∂ln d/∂ln p = −3/(5+3q) = −0.50, ∂ln w/∂ln p = 5/(5+3q) = 0.83.
Doubling p widens the channel by 77 % and shallows it by 29 %, so p is not physically weak. But n and p enter depth
only as the ratio n/p. What a gauge sees (celerity, attenuation) comes mostly through depth, so to first order a
gauge identifies n/p and not n and p separately. The Newport Hessian (`research/findings/2026-09-07-landscape-uh-juniata-findings.md`)
shows exactly this: stiff eigenvector ≈ n^0.8 p^−0.5, sloppy directions where n and p move together.

The exponent's leverage on width is `∂ln w/∂q = ln d + q ∂ln d/∂q`: it scales with ln(depth), so it is a river-size
lever (and changes sign for depth below 1 m). This is the user's point, and it is right in natural units: a log
multiplier on q ∈ [0, 1] understates q. Converting the Newport curvature `H_qq(ln q) = 0.0134` to natural units,
`H_qq(q) = 0.0134 / 0.35² = 0.11 per unit q²`, comparable to n's 0.158 per unit (ln n)².

| Newport, seed 42 | curvature at optimum | at trained point | trained field, 213 reaches |
|---|---|---|---|
| ln n | 0.158 | 0.114 | 0.098 to 0.108 |
| ln p | 0.065 | 0.022 | 12.1 to 13.2 |
| ln q | 0.013 | 0.027 | 0.31 to 0.37 |

The head emits an almost constant field over the basin: the learned p is already a basin constant (≈ 12.7), not a
spatial pattern. Prescribing p costs little spatial information and removes the n–p degeneracy that lets every seed
land at a different point on the valley floor (seed 43: p × 0.63 relative to seed 42 with the same loss).

## 2. Provenance of p = 21 and the literature

- **p = 21 is a Juniata field fit, not a literature constant.** The dissertation (ch. 2) states p = 21 came from
  "preliminary data fitting to USGS hydraulic geometries from field surveys of gages in the JRB"; the general form
  w = p d^q is credited there to Leopold & Maddock (1953), Gleason (2015), Orlandini & Rosso (1998). Bindas et al.
  (2024, WRR, doi 10.1029/2023WR035337) reports the geometry parameter unidentifiable in synthetic tests. DDR's
  `configs.py` hardcodes 21 as the fallback with no numeric citation.
- **Verified downstream hydraulic geometry:** Moody & Troutman (2002): `w = 7.2 Q^0.5`, `d = 0.27 Q^0.3`
  (coefficients recovered from open-access citing texts and Frasson et al. 2019's validation table; the primary PDF
  was paywalled). Yamazaki et al. (2011, CaMa-Flood) prescribe `W = max(1.00 R_up^0.7, 10)` from 30-day upstream
  runoff, calibrated to their router, not a field regression. Andreadis et al. (2013) and Neal et al. (2012) could
  not be fetched; nothing from them is used.
- **A consistency check the coefficients allow.** Downstream, `w ∝ d^(0.5/0.3) = d^1.67`. The model's w = p d^q is
  applied at each reach for all flows (an at-a-station relation) with q ∈ [0, 1]. A constant p therefore cannot
  represent downstream widening: bigger rivers must be wider at the same depth, which only p can carry. This is the
  physical argument for p as a function of river size rather than a constant.

## 3. Candidate functions for p

1. **Constant 21** (2024 setting). Arm `config/experiments/uh_retro_pfixed21.yaml`, training started 14:06Z
   2026-09-08 (unit `ddrs-train-p21`). Baseline for everything below.
2. **Moody–Troutman consistent p.** From w = 7.2 Q^0.5 and d = 0.27 Q^0.3 with w = p d^q:
   `p_i = 7.2 · 0.27^(−q_i) · Q_ref,i^(0.5 − 0.3 q_i)`; at q = 0.35, `p ≈ 11.4 Q_ref^0.395`
   (Q_ref = 1 m³/s → 11; 100 → 71; 1000 → 176). Q_ref must be a per-reach reference discharge. Two choices:
   (a) the long-term mean routed discharge from the arm's own inflow (self-consistent but arm-dependent, which
   confounds the cross-input comparison); (b) drainage area through a Q_ref(A) relation fitted once on CONUS from
   the UH Q′ store and MERIT `log10_uparea` (arm-independent, reproducible in-house, but a fit we make ourselves).
   Recommendation: (b), with the fit reported.
3. **Learned p, one scalar per basin** is what the head already does in effect; not a simplification.

## 4. Plan

- Constant-21 arm: landscape over (n, q) at the eight validation gauges when training finishes (objective now
  handles fixed parameters, `b52d966`); compare gauge optima, NSE at optimum, and the census statistics with the
  learned-p arm. If the constant-p arm's gauge optima have smaller |α*| and the seeds' optima agree on n*, the
  degeneracy argument is confirmed.
- Then the Moody–Troutman p(A) arm: needs a config option (`params.p_spatial_function`) and a one-off Q_ref(A) fit.
  User decision before spending the training time.

## 5. Concerns

- q's natural unit is additive; the landscape's log multiplier on q is fine for plots (relabel the axis in q) but the
  behavioural half-widths along q should be quoted in Δq.
- Prescribing p from bankfull relations at a per-reach reference discharge assumes the at-a-station exponent q is
  the model's q; the two exponents differ in the literature, and the model conflates them.
- Only Moody & Troutman's coefficients are verified; Andreadis (2013) is the natural global source and is paywalled.
