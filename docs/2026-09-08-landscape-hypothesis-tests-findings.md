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
White River optima sit near the trained point. So the census does not indicate an optimizer defect (which would show
as the same displacement everywhere); it indicates a regional compromise (see §3 for its input-independence), and a second use of the instrument: at
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
## 3. Inputs (§7.3): five inflow arms, eight gauges

Run `.ddrs/experiments/landscape-arms/2026-09-08T15-15-36Z/` (13 min, cpu, Newton and Hessian only;
`figures/ARMS.md`, `arms.png`, `arms_compact.png`). Basin-median values; the optimum is median(n0)·exp(α*_n).

| Newport 01567000 | trained n | gauge-optimal n | NSE trained → optimum | note |
|---|---|---|---|---|
| daily-lstm | 0.129 | 0.031 | 0.608 → 0.805 | range-bound after 2 iterations |
| hourly-lstm | 0.120 | 0.026 | 0.725 → 0.868 | range-bound after 3 |
| uh-retro | 0.103 | 0.037 | 0.692 → 0.769 | interior |
| dhbv2-lumped | 0.170 | 0.050 | 0.501 → 0.644 | interior |
| dhbv2-dist | 0.103 | 0.052 | 0.701 → 0.743 | interior |

| Mapleton Depot 01563500 | trained n | gauge-optimal n | NSE trained → optimum |
|---|---|---|---|
| daily-lstm | 0.132 | 0.030 | 0.609 → 0.782 |
| hourly-lstm | 0.121 | 0.018 | 0.683 → 0.831 |
| uh-retro | 0.103 | 0.031 | 0.548 → 0.652 |
| dhbv2-lumped | 0.170 | 0.084 | 0.539 → 0.628 |
| dhbv2-dist | 0.103 | 0.020 | 0.553 → 0.630 |

White River 06452000: the two LSTM arms find no descent direction (NSE −1.8, −0.6); uh moves n 0.12 → 0.10 for
+0.05; dhbv2-lumped n 0.16 → 0.04 for −2.3 → −1.2; dhbv2-dist −0.07 → 0.02. The Cannonball gauges as in §1.

Readings.

1. **Every input wants a faster Juniata, by a factor 2 to 6 in n.** The trained n across arms spans 0.10 to 0.17;
   the gauge-optimal n spans 0.018 to 0.084, all far below trained, and NSE at the optimum is 0.04 to 0.20 higher.
   The two LSTM arms hit the range bound on the way down, so their true optima are lower and better still
   (hourly-lstm reaches 0.868 at Newport before the bound). Gauge-optimal n of 0.02 to 0.05 is a plausible
   main-stem roughness; trained 0.10 to 0.17 is not.
2. **The optimum does move with the input, but less than the trained point suggests.** Spread of the gauge-optimal
   ln n across arms at Newport 0.71 (× 2.0) versus 0.51 (× 1.7) for the trained n; the fastest inflow in the
   population study (dhbv2-dist) wants the least reduction, the slowest (dhbv2-lumped) the most in absolute n.
   That is compensation: the channel optimum partly absorbs the inflow's timing. But it is second order next to
   reading 1.
3. **Together with §1 and §2: a regional, input-independent bias in trained n.** The same displacement appears in
   both seeds (§1), in all five inputs (§3), and is where training was still heading at epoch 30 (§2). At the
   Plains gauges it is absent (the optimum is near the trained point or undefined). So the batch pushes Juniata's n
   up for a reason that is not the inflow product and not the seed: other gauges in the batch need the delay that
   a high n provides (inflow that arrives too early, hillslope timing the routing is asked to absorb), and the
   attribute-conditioned head cannot give the Juniata a low n while giving those basins a high one. This revises
   §1's "not a training defect": it is not a defect of the optimizer, it is a compromise imposed by the shared
   head across regions, and its cost at the Juniata is 0.08 to 0.20 NSE depending on the inflow.

**Reading against the hypothesis.** Confirmed in the strong form for the well-fit gauges: the batch solution is a
compromise, its direction is set by the batch and not by the input, and the inflow moves the gauge optimum only at
second order. The landscape has told us where the model is trained wrong (Juniata n too high for every input),
where it is trained right (the stiff coordinate at Newport under the UH arm reaches the optimum by epoch 5), and
where the parameters cannot matter (Plains).

**Limits.** Basin-uniform multipliers; the eight gauges; the Newton range bound on the LSTM arms; `p` and `q`
optima are along sloppy directions and their cross-arm spread (× 4 in p) is not informative.
## 4. Dense terrain, per-reach gradient map, 41-gauge census: NOT RUN (stopped 15:45Z; leak diagnosed and fixed 16:15Z, `658cbfc`, see ddrs-dev traps.md T11)

The 41×41 dense-grid landscape (`landscape-uh-surface`) leaks memory in the slice loop: the process grew from
31 GB to 39 GB in its first 20 minutes alongside another run, and to 77 GB when re-run alone, at which point it was
stopped to protect the GPU training. The 13×13 runs (169 cells per plane) complete; the 41×41 runs (1,681 cells per
plane, two planes, four windows) do not. Suspects: per-cell tensors retained through the window loop in
`Objective::forward_loss`/`eval` under the ndarray backend, or the per-cell `Eval` values accumulating something
larger than three floats. Diagnosed afterwards: the autodiff tape of every forward-only eval is retained until a backward consumes it
(`examples/leak_probe.rs`); `Objective::eval` now runs a backward per eval, RSS is flat and numerics unchanged. Under the user's three-strike rule this was the third failed assumption
of the day (nohup survival, two-process concurrency, serial dense run), so no further runs were launched.

Not run for that reason: `landscape-uh-surface` (dense n-p and stiff-sloppy terrain), `landscape-uh-surface-nq`
(dense n-q with p at trained), `landscape-uh-reachgrad` (per-reach loss gradients), `landscape-uh-census41`
(41 gauges, both seeds, Newton and Hessian only; would have been cheap since grid 0 does not enter the leaking loop).
The fixed-p arm training (`ddrs-train-p21`) was left running; it does not use this code path.

**Next session, in order:** (1) find the leak with a 13×13 vs 25×25 run under `/usr/bin/time -v` and a heap
profile, or reduce peak memory by evaluating the grid one window at a time; (2) run census41 (safe: grid 0);
(3) the three dense/gradient bundles; (4) the fixed-p arm's landscape.

## 5. The p = 21 model: first results (user run `2026-09-08T15-55-52Z-conus-train-and-test`)

Trained from the workspace `ddrs.yaml` with `p_spatial` removed from the learnable list (p = 21 everywhere, the 2024
setting), n and q learned, seed 42, 30 epochs, **gages_3000 (2,859 gauges after the drainage-area filter), which
includes the Juniata gauges in training**. The learned-p arm was trained on gages_2000_area_balanced, which does not
contain them. Two things changed at once; the clean single-change comparison is the `uh_retro_pfixed21` run
(same population as the learned-p arm), training at epoch 16 of 30 when this was written.

Bundle `landscape-p21-reachgrad` (Newton, Hessian, per-reach gradients; 2 min):

| | learned p (seed 42) | p = 21 (this run) |
|---|---|---|
| Basin-median n at the Juniata (both gauges) | 0.103 | **0.040** (p10 to p90: 0.033 to 0.044) |
| Basin-median q | 0.35 | 0.10 |
| Newport NSE at trained point (WY2000 windows) | 0.692 | 0.703 |
| Newport gain to own optimum | +0.077 | none found (Newton: no descent; Hessian indefinite, λ₃ = −0.03) |
| Mapleton NSE at trained point | 0.548 | 0.644 |
| Mapleton gain to own optimum | +0.104 | +0.013 (INVALID: the 3-D search moved p, which is not a parameter of this model; redo in 2-D) |

With p pinned, training put the Juniata's n at 0.040, inside the 0.03 to 0.05 band that yesterday's landscapes
identified as the gauge optimum under every inflow, and the remaining per-gauge gain collapsed from 0.08 to 0.10 NSE
to at most 0.013. Whether that is the pinning of p (removing the n/p degeneracy) or the gauges being in the training
set is exactly what the `uh_retro_pfixed21` run will separate.

**Correction (17:40Z).** The p = 21 model has two channel parameters, n and q. The study treated p as a third axis
(the multiplier on the constant 21) and let Newton move it; for this model the landscape must be two-dimensional.
The running 41-gauge census on this model was stopped and the study is being restricted to the learnable parameters
(inactive axes masked out of Newton, Hessian, eigenvectors, half-widths, and slices). The reach-gradient rows for p
below are a sensitivity to a constant, not to a parameter, and are kept only as a diagnostic.

**Per-reach gradient map at Newport** (`figures/reachgrad_p21-conus_01567000.png`). At the trained point, 95 % of
reaches have dL/d ln n < 0 (loss falls if n rises: the network is now slightly too fast), with the largest magnitudes
spread along the whole main stem; the handful of positive values sit within 30 km of the gauge. So the gauge would
raise n far upstream and lower it near the gauge, a spatial pattern the basin-uniform multiplier cannot express and the
uniform gradient (−0.047) averages away. Half of the total |dL/d ln n| lies within about 130 km of the gauge (of
320 km), half of the q sensitivity within about 60 km, p's further out. The ten most sensitive reaches carry 47 %
(Newport) and 63 % (Mapleton) of the total n sensitivity. Per-reach sums match the basin-uniform gradients to 1e-7
(chain-rule check).

**Instrument note.** At Newport the Hessian at the trained point is indefinite, so the damped Newton direction is not
a descent direction and the search stops at iteration 0 even though the gradient is nonzero. A gradient-descent
fallback when Newton fails at the first iteration is needed (small change in `run_gauge`).

## 6. 2-D census on the p = 21 model: 84 nested-reference gauges (run `landscape-p21-census41/2026-09-08T18-06-55Z`)

Newton and Hessian over (n, q) only (p inactive, `692c00f`), box ±ln 10, 30 min on cpu. The nested-reference
selection on this run's gauge list (gages_3000) yields 84 gauges rather than the 41 of the area-balanced list.

| | count | median gain to own optimum | median optimal n / trained n (IQR) | median q multiplier |
|---|---|---|---|---|
| Well fit (NSE at trained point > 0.3) | 20 | 0.001 | 1.00 (0.88 to 1.03) | 1.00 |
| Poorly fit (NSE ≤ 0.3) | 64 | 0.000 | | |

Five well-fit gauges gain more than 0.02 NSE, in mixed directions: 02110500 wants slower (n × 2.8, q × 3.2),
03069500 slower (n × 1.5), 03161000 faster (n × 0.5), the two Kansas gauges 06889500 and 06889200 faster with q at
the range ceiling. No common direction: the resultant of the unit displacement vectors is small.

**Reading.** With p pinned, the trained model sits at or within noise of the per-gauge optimum at the median well-fit
gauge, and the systematic "route faster" displacement of the learned-p arm (Juniata n × 0.3, all inputs, both seeds)
is gone. What remains is a scatter of small, gauge-specific corrections in both directions, which is what a genuine
batch compromise looks like. Poorly fit gauges are unchanged: nothing in the channel helps them.

**Caveats.** (i) This model trained on gages_3000, so every gauge here is in its training set; the clean twin on the
area-balanced list (`uh_retro_pfixed21`, training) will show whether the collapse of the per-gauge gain survives
out-of-sample gauges. (ii) Nine of the 20 well-fit gauges hit the range bound, five of them with zero Newton
iterations: they have 3 to 5 reaches, where a single clamped reach exceeds the 5 % `max_clamped` rule, so their
"gain 0" is the instrument refusing to step, not an optimum. The rule needs a per-reach floor (at least one reach
allowed) for small basins. (iii) The census figure with 84 gauges is dense; the per-gauge CSV is the reference.

