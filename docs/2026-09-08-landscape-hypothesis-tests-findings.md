# Tests of the batch-compromise hypothesis (landscape spec §7) — findings, 2026-09-08

**Spec:** `docs/superpowers/specs/2026-09-07-adjoint-landscape-design.md` §7.
**Hypothesis (user):** each gauge's loss under large-batch training sits at a batch compromise, not the gauge's own
minimum; the adjoint landscape over (n, p, q) at a gauge tells whether the model is trained correctly, and the same
map under different inputs tells whether inputs move the optimum or only the batch solution.
**Scripts:** `experiments/landscape/{census,trajectory,arms,surface,reachgrad}.py`.

## 1. Displacement census (§7.1): eight validation gauges, UH seeds 42 and 43

Run `.ddrs/experiments/landscape-uh-census/2026-09-08T13-52-27Z/` (72 min, cpu; `figures/CENSUS.md`, `census.png`).
Range-bounded Newton (`max_clamped 0.05`), grid 13², box ±ln 10.

| Gauge | NSE trained (s42 / s43) | Gain to own optimum | Optimum multipliers (n, p, q), s42 | s43 |
|---|---|---|---|---|
| 01563500 Mapleton Depot | 0.55 / 0.56 | 0.10 / 0.09 | 0.30, 0.26, 0.10 | 0.31, 0.40, 0.10 |
| 01567000 Newport | 0.69 / 0.70 | 0.08 / 0.07 | 0.36, 0.25, 0.63 | 0.37, 0.46, 0.39 |
| 06452000 White River (582 reaches) | −0.38 / −0.37 | 0.05 / 0.01 | 0.84, 1.08, 1.12 | 0.99, 1.35, 1.60 |
| 06449100 | −1.37 / −1.35 | 0.02 / 0.02 | 0.45, 2.9, 2.1 | 0.53, 1.8, 2.2 |
| 06447000 | 0.09 / 0.06 | 0.00 / 0.02 | range-bound at trained point | 1.05, 2.2, 0.19 |
| 06350000 Cannonball | −29 / −37 | 0.01 / 1.9 | no descent (iter 1) | no descent |
| 06354000 | −18 / −19 | 0 / 0 | no descent (iter 0) | range-bound |
| 06353000 | −11 / −12 | 1.6 / 0 | 1.6, 9.6, 0.10 (creased) | range-bound |

Three groups, and the grouping is the result.

1. **Well-fit gauges (Juniata).** Both seeds agree on the move: n × 0.30 to 0.37, p × 0.25 to 0.46, q down. Gains of
   0.07 to 0.10 NSE. The batch solution is displaced from the gauge optimum in a direction the two seeds reproduce
   (seed angle small), so this is a property of the training, not seed noise. It is the "route faster" displacement
   found in the sample case, and it is regional: the Plains gauges do not share it.
2. **Poorly-fit gauges (White River, NSE −1.4 to 0.09).** The routing optimum is close to the trained point in n
   (× 0.84 to 1.05 at the two larger basins) and the gain is at most 0.05 NSE. The landscape says: no combination of
   channel parameters fixes these gauges. The error is in the inflow, not the channel. The Hessians here are steep
   (λ₁ up to 37, from small σ in the NSE normalisation) and the Newton hit the range bound at three of six arms.
3. **Un-fittable gauges (Cannonball, NSE −11 to −37).** Newton finds no descent direction at the trained point at
   four of six arm × gauge cases; where it moves, it runs to extreme corners (p × 9.6, q × 0.1) for a "gain" that
   leaves NSE at −10. These are the intermittent-inflow gauges of checks 3 and 4; the landscape is creased and the
   quadratic model does not apply. Reported as "no descent", not as an optimum.

**Reading against the hypothesis.** The batch compromise is real and directionally consistent where the model fits
(Juniata: 0.07 to 0.10 NSE left on the table, all in the "faster" direction). It is not a CONUS-wide offset: the
White River optima sit near the trained point. So the census does not indicate a training defect (which would show
as the same displacement everywhere); it indicates a regional compromise, and a second use of the instrument: at
gauges with NSE < 0 the landscape is flat or creased in every channel direction, which localises the error to the
inflow. That is the bias-propagation question answered per gauge: routing parameters cannot absorb the Plains
inflow error.

**Limits.** Eight gauges, two of them well fit; the set was chosen for the adjoint mass checks, not for landscapes.
The 41-gauge nested-reference census (Newton and Hessian only) is queued to test whether "faster" is Appalachian or
general among well-fit gauges. Direction statistics (mean resultant length) are in `figures/CENSUS.md`.

## 2. Training trajectory (§7.2): UH seed 42 at init, epochs 1, 5, 10, 20, 30

Run `.ddrs/experiments/landscape-uh-trajectory/2026-09-08T15-04-23Z/` (11 min, cpu, Newton and Hessian only;
`figures/TRAJECTORY.md`, `trajectory.png`). Every checkpoint's trained point is expressed in the epoch-30 physical
frame (reach-mean log field ratio) and then in epoch 30's eigenbasis at epoch 30's per-gauge optimum, so c = 0 is that
optimum and the epoch-30 row equals the census `coord_trained`.

| | init | epoch 5 | epoch 10 | epoch 20 | epoch 30 | epoch-30 half-width (5 %) |
|---|---|---|---|---|---|---|
| Newport c₁ (stiff) | 0.05 | 0.01 | −0.02 | −0.00 | −0.02 | 0.14 |
| Newport c₂ | 2.13 | 2.03 | 1.91 | 1.81 | 1.80 | 1.89 |
| Newport NSE at trained point | 0.598 | 0.639 | 0.667 | 0.687 | 0.692 | optimum 0.769 |
| Mapleton c₁ (stiff) | 0.36 | 0.30 | 0.23 | 0.23 | 0.21 | 0.19 |
| Mapleton c₂, c₃ | 2.88, 1.69 | 2.78, 1.64 | 2.65, 1.51 | 2.54, 1.47 | 2.53, 1.44 | 5.5, 12.7 |
| Mapleton NSE at trained point | 0.463 | 0.497 | 0.524 | 0.544 | 0.548 | optimum 0.652 |
| Basin-median fields (n, p, q) | 0.133, 14.0, 0.50 | | | | 0.103, 12.8, 0.35 | |

Readings.

1. **The stiff coordinate converges first and fully at the downstream gauge.** Newport's c₁ is inside the 5 %
   half-width from epoch 5 on and hovers at 0.1 half-widths thereafter. Along the one direction the gauge constrains,
   large-batch training is correct and fast.
2. **The sloppy coordinates move slowly and were still moving at epoch 30, but decelerating.** Newport c₂ fell 0.33
   in 30 epochs, 0.01 of it in the last 10; Mapleton's coordinates the same. The remaining gain (0.08 to 0.10 NSE)
   lies along these directions. Part of it is under-training (the batch gradient along a sloppy axis is small), but
   the deceleration says the asymptote is a compromise, not the gauge optimum: more epochs recover little.
3. **Mapleton, upstream, loses the compromise.** Its stiff coordinate stalls at 1.1 half-widths out (0.36 → 0.21).
   The two gauges want nearly the same physical move (both n × 0.3), so the compromise is not between them; it is with
   the rest of CONUS through the head's attribute mapping, and the downstream gauge with the larger observed variance
   is the one the batch satisfies.
4. **The head changes a basin almost uniformly.** The across-reach spread of the per-checkpoint field change falls
   from 0.07 (init to epoch 30) to 0 monotonically and is nearly identical across the eight gauges (panel d): within
   a basin, training moves n, p, q by a common factor. This is why the basin-uniform multiplier parametrization
   (Phase A) captures what training can do, and why "p is effectively a basin constant" (`docs/2026-09-08-fixed-p-assessment.md`).
5. **Plains gauges do not move in loss space.** NSE at the trained point is flat at −11 to −29 from init to epoch 30;
   training changed their parameters but not their loss. Consistent with §1: the channel is not where their error is.

**Reading against the hypothesis.** Training is correct along the constrained direction at the gauge that wins the
compromise, and slow but not wrong along the unconstrained ones. The instrument separates the two. "Are we training
correctly" gets a graded answer: yes for the stiff coordinate at well-fit downstream gauges; the upstream gauge is
left 1 half-width short along its stiff axis; the sloppy coordinates are wherever the shared head puts them.
## 3. Inputs (§7.3): pending (queued)
## 4. Dense terrain (n-p, stiff-sloppy; n-q with p at trained): pending (queued after §3)
## 5. Per-reach gradient map: pending (queued)
