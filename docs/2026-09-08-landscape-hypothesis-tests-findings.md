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

## 7. Dense (n, q) terrain on the p = 21 model, 2-D optimum, depth axis (run `landscape-p21-surface-nq/2026-09-08T18-20-07Z`)

25 × 25 grid over n and q with p = 21 pinned, slices centred at the trained point, Newton in 2-D. Figures in
`figures/` (q axis) and `figures/depth/` (depth at the gauge under mean flow in place of q; `surface.py --depth-axis`).

| Newport | n | q | depth at gauge (m) | width (m) | NSE |
|---|---|---|---|---|---|
| trained | 0.040 | 0.10 | 1.8 | 22 | 0.703 |
| 2-D optimum (q at the box edge) | 0.082 | 0.023 | about 2.8 | 21 | 0.742 |

Geometry at the gauge reach: observed mean flow 97 m³/s over 340 valid days, slope 0.0048, p = 21. Mapleton: n
0.040 → 0.073, q 0.10 → 0.022, NSE 0.644 → 0.602 (loss fell, window-mean NSE rose then fell: the two metrics weight
the four windows differently; the loss is the objective).

**Shape.** The trench in n is at 0.04 for q ≥ 0.1, exactly where training put it. Toward small q the trench floor
keeps dropping and drifts to higher n: the gauge prefers a nearly rectangular channel (q → 0, width fixed at p = 21 m)
with twice the roughness. n and q trade through depth, and q sits at the box edge, so the true 2-D optimum is at
even smaller q.

**Why.** With p = 21 the modelled channel at Newport is 21 to 22 m wide at a mean flow of 97 m³/s and about 1.8 m
deep. The Juniata at Newport is far wider than that by any map. An under-wide channel is too deep and too fast for
its discharge, and the two levers the gauge has left, n and q, are being used to compensate: more roughness to slow
it, smaller q to stop width growing with depth. This is the case for p as a function of river size (assessment
doc §3.2) made by the landscape itself: at a 20 m headwater p = 21 is right, at an 8,700 km² main stem it is not, and
a constant p pushes the compensation into n and q, which is the same kind of confounding the learned p had, moved
one parameter over.

**Readability.** The depth-axis surface is a narrow curled ribbon because depth is exponentially sensitive to q; it
shows the valley floor at 2 to 3 m depth but hides the n structure. The q-axis figure with the depth annotation in
the corner is the more legible of the two.

## 8. The clean twin: p = 21 on the area-balanced population (run `2026-09-08T14-06-12Z-train-and-test`)

Same config, seed, and 1,841-gauge population as the learned-p seed-42 arm; only `p_spatial` removed from the
learnable list. Eval on the same test population: median NSE 0.700, KGE 0.736 (learned p: 0.707 / 0.738; seed 43:
0.710 / 0.743). Pinning p costs nothing measurable at the population level. Study run `.ddrs/experiments/landscape-pfixed-reachgrad/2026-09-08T20-43-34Z/`.

| Juniata, out of sample for this model | learned p (seed 42) | p = 21, area-balanced (this run) | p = 21, gages_3000 (user run, in sample) |
|---|---|---|---|
| Basin-median trained n | 0.103 | **0.100** | 0.040 |
| Basin-median trained q | 0.35 | 0.385 | 0.10 |
| Newport NSE trained → own optimum | 0.692 → 0.770 | 0.661 → 0.742 (n × 0.71, q to the low box edge) | 0.703 → 0.742 |
| Mapleton NSE trained → own optimum | 0.548 → 0.652 | 0.514 → 0.607 (n × 0.36, q × 2.5) | 0.644 → 0.658 |

**This refutes the reading in §5 and §6 that pinning p moved n to the gauge value.** With p pinned and the Juniata out
of sample, training leaves n at 0.100, the same as with p learned, and the per-gauge gain (0.08 to 0.09 NSE) is
unchanged. The 0.040 in the user's run came from the training population (gages_3000, 2,859 gauges, Juniata
included), not from pinning p. Which part of the population change did it (the Juniata gauges themselves being in the
batch, or the wider set shifting the batch's preferred n everywhere) is testable from the two 41-gauge censuses:
compare trained n gauge by gauge between the two p = 21 runs, for gauges in both populations versus gauges only in
gages_3000.

**What survives.** The n/p ratio identifiability is confirmed once more from the other side: the gauge-optimal n
scales with p. Learned p 12.7 → optimal n 0.037 (n/p = 0.0029); p = 21 → optimal n 0.071 (n/p = 0.0034); the two
models' optima differ in n by the factor their p differ by, to within the q trade. What does not survive is the
hope that fixing p alone repairs the batch compromise at gauges the batch does not contain: n stays where the
batch's 1,841 gauges leave it, near 0.10, whether or not p is free.

**Corrections to earlier sections.** §5 "with p pinned, training put the Juniata's n at 0.040" should read "with p
pinned and the Juniata in the training population". §6's "pinning p removed the batch compromise on n" should read
"in a model trained on these gauges, the per-gauge gain is small"; the out-of-sample statement is this section.

## 9. Population, not membership: the two p = 21 models compared gauge by gauge

`experiments/landscape/population_compare.py` on the two 2-D censuses (`landscape-p21-census41/2026-09-08T18-06-55Z`,
84 gauges, gages_3000 model; `landscape-pfixed-census41/2026-09-08T20-45-04Z`, 41 gauges, area-balanced model).
The 30 gauges present in both censuses are in BOTH training lists.

| | gages_3000 model | area-balanced model |
|---|---|---|
| Trained n at the 30 shared gauges, median ratio | 0.48 × | 1 |
| Well-fit gauges (NSE > 0.3) in its census | 20 of 84 | 6 of 41 |
| Median gain to own optimum at well-fit gauges | 0.001 | about 0 (window-mean NSE), loss falls |
| Direction of the well-fit optima in n | scattered, resultant 0.15 | all 6 want n × 0.57 to 0.67 |

The configs differ only in the gauge list (plus an unused precip path and the sparse-solver device, which the cpu
backend overrides). So the factor-two difference in trained n is a property of the training population and it is
CONUS-wide: gauges that both models trained on shift by the same factor as the Juniata. Membership of the Juniata in
the batch is not the cause.

**Which property of the population?** Two candidates, not yet separated.
1. Composition. gages_3000 has 3,211 gauges with median area 700 km² (14 % above 5,000 km²); the area-balanced list
   has 1,841 with median 1,249 km² (32 % above 5,000 km²). The list with MORE small basins trained the LOWER n.
   A roughness of 0.10 over-attenuates daily peaks in small basins, which the daily NSE does see; 1,400 more of them
   would supply a consistent gradient toward lower n that the area-balanced batch lacks.
2. Step count. With `batch_size` 64, gages_3000 gives about 45 optimizer steps per epoch against 29, so 30 epochs is
   1.55 × more updates. The area-balanced trajectory (§2) was still drifting toward lower n at epoch 30.
The p = 21 trajectory on the gages_3000 run (bundle `landscape-p21-trajectory`, running) separates them: if its n
reaches 0.04 within the first few epochs the composition sets the target; if it drifts there over 30 epochs, the
area-balanced run is simply under-trained along the sloppy direction.

**Standing result either way.** The gages_3000 model sits at the per-gauge optimum at its well-fit gauges (median
gain 0.001) and is the only model so far that does. Its population median NSE is 0.720 on its own 2,365-gauge test set.

## 10. Composition, not step count: the gages_3000 model's trajectory (run `landscape-p21-trajectory/2026-09-08T21-00-25Z`)

Basin-median trained n and q at the Juniata (Newport; Mapleton identical) per checkpoint of the gages_3000 p = 21 run,
with the NSE at the trained point over the WY2000 windows:

| epoch | init | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| n | 0.133 | 0.129 | 0.093 | 0.040 | 0.040 | 0.040 |
| q | 0.50 | 0.48 | 0.34 | 0.12 | 0.10 | 0.10 |
| Newport NSE | 0.562 | 0.575 | 0.710 | 0.708 | 0.703 | 0.703 |

n reaches 0.040 by epoch 10 (about 450 optimizer steps at 45 per epoch) and does not move for the remaining 20
epochs, through two learning-rate decays. The area-balanced run made about 870 steps in its 30 epochs and left n at
0.100. So the difference is not step count: the gages_3000 batch's gradient drives n to 0.04 and holds it there; the
area-balanced batch's gradient does not. **Composition of the training population sets the batch's roughness.**

Across the four gauges checked (Juniata pair, White River pair) the trained n at epoch 30 is 0.040 to 0.058: the
head's output is close to CONUS-uniform, so "the batch's n" is nearly a single number set by the aggregate gradient.
Newport's NSE peaks at epoch 5 (0.710, n = 0.093) and settles at 0.703; the last 0.05 of n came from other gauges.

**Which gauges supply the gradient** is the remaining question. The lists differ mainly in small basins (gages_3000
adds 1,370 gauges of median area 333 km²). The direct test is a size-stratified training experiment (or the
per-gauge dL/d ln n at α = 0 on the area-balanced model, tabulated against drainage area across the 84 gauges of
the gages_3000 census: the gauges with a strong negative gradient are the ones that pull n down). Not run; user
decision.

**Consequence for the thesis.** The per-gauge landscape is doing what the hypothesis wanted: it identified that the
area-balanced model's n was a batch compromise (§1 to §3), predicted the gauge value (0.03 to 0.05), and the model
trained on the other population landed there and shows no residual per-gauge gain (§6). The same instrument now says
the area-balanced population under-determines n, and which gauges hold the information.

## 11. Full water-year window (user decision 2026-09-09): p = 21 model, epoch 30

All bundles now use one 365-day window over WY2000 (`window_days: 365`, `n_windows: 1`; code defaults changed,
`7352665`). The p = 21 studies were rerun on `2026-09-08T15-55-52Z-conus-train-and-test/epoch_30_mb_1`; the clean-twin
steps were dropped at the user's request (only this model from here on).

**Juniata (n, q) terrain** (`landscape-p21-surface-nq/2026-09-09T02-14-52Z`, 25 × 25, p pinned at 21):

| full year | NSE trained | NSE at 2-D optimum | optimum move |
|---|---|---|---|
| Newport | 0.790 | 0.859 | n 0.040 → 0.082, q 0.10 → 0.023 (box edge) |
| Mapleton Depot | 0.840 | 0.858 | n 0.040 → 0.073, q 0.10 → 0.022 (box edge) |

The shape is the seasonal-window shape: a trench in n at 0.04 for q ≥ 0.1 whose floor drops and drifts to n ≈ 0.08 as q
→ 0. NSE at the trained point is much higher over a full year (0.79 vs 0.70 at Newport) because the annual variance is
dominated by the spring floods the model gets right; 90-day windows scored each season's small variance alone. The
gain to the gauge optimum is larger over the year (+0.07 at Newport) and points the same way: with a 21 m channel the
gauge wants roughly double the roughness and a near-rectangular section. q sits on the box edge, so the true 2-D
optimum is at smaller q still.

**84-gauge census, full year, before the line-search fix** (`landscape-p21-census41/2026-09-09T01-52-19Z`): 58 of 84
gauges well fit (NSE > 0.3; 20 with seasonal windows), median gain 0.000, 66 % within a factor 1.25 of their optimum,
but 41 of the 58 range-bound and 26 with zero Newton iterations: the unbounded Newton step for small basins landed on
the box corner every trial (fixed in `964f062`: step cap 1 log unit, gradient fallback, 16 halvings; verified on
01436000, 01435000, 01452000, which now take 5 to 12 steps). The "at optimum" share from this census is inflated and is
superseded by the all-gauge sharded census (§12, pending).

**CONUS map** (`experiments/landscape/conus_map.py`, `figures/conus_n_gap.png`): per gauge |ln(n_optimal / n_trained)|
blue (0) to red (ln 3), sign in a second panel, hollow grey where NSE < 0.3, black edge for range-bound optima. The
preview on the 84 gauges is too blue for the reason above.



## 12. Juniata pair on the full 15-year test period (run `landscape-p21-fulltest-juniata/2026-09-09T02-55-05Z`, 2 shards)

One window over the whole test axis (5,479 days, 1995-10-01 to 2010-09-30; `window_days: 0`), Newton and Hessian
only, about 20 minutes per gauge on one core.

| gauge | NSE trained (landscape) | NSE trained (run's own eval, same period) | NSE at 2-D optimum | optimum n, q |
|---|---|---|---|---|
| Mapleton Depot | 0.841 | 0.847 | 0.845 | 0.050, 0.37 (n x1.24, q x3.67) |
| Newport | 0.853 | 0.858 | 0.883 | 0.067, 0.36 (n x1.68, q x3.57) |

The landscape's NSE at the trained point agrees with the run's own eval to the second decimal, which validates the
objective (hourly routing, daily pooling under the training `tau`, warm-up excluded) against the production eval
path. Over 15 years the per-gauge gain is +0.03 at Newport and +0.004 at Mapleton, smaller than over WY2000 alone
(+0.07 and +0.02): a single year overstates what the gauge could gain.

**q flips sign between windows.** Over WY2000 the optimum pushed q to its floor (0.02); over 15 years it pushes q up
(0.36). n moves up in both (× 1.7 to 2 over one year, × 1.2 to 1.7 over 15). So the n direction is robust across
windows and the q direction is not: q is the sloppy coordinate, and its "optimum" is whatever the particular
sequence of floods rewards. For the paper this is the operational definition of a poorly constrained parameter:
its optimum changes sign with the evaluation window while the loss barely moves.

## 13. All 2,365 test gauges, WY2000, p = 21 model, epoch 30 (run `landscape-p21-all/merged`, 24 shards, 1 h 47 min)

First census with the fixed line search (§11) and the full population. Newton and Hessian over (n, q), p pinned;
0.4 % of gauges range-bound, none with zero iterations, 4 % used the gradient fallback. Map:
`docs/figures/2026-09-09-conus_n_gap_p21_wy2000.png` (copy of `merged/figures/conus_n_gap.png`).

| WY2000 | count | median gain to own optimum | gain > 0.02 | median |ln(n*/n)| | within × 1.25 | wants slower (n* > 1.25 n) | wants faster (n* < 0.8 n) |
|---|---|---|---|---|---|---|---|
| well fit, NSE > 0.3 | 1,710 | 0.014 | 42 % | 0.67 (factor 1.95) | 19 % | 61 % | 19 % |
| NSE > 0.6 | 1,261 | 0.013 | 40 % | 0.62 | 22 % | 62 % | 16 % |
| poorly fit, NSE ≤ 0.3 | 655 | 0.027 | 55 % | 0.92 | 11 % | 55 % | 34 % |

By basin size among the well fit (reaches in the subgraph): ≤ 10 reaches, 578 gauges, median gain 0.01; 10 to 50,
717, 0.01; 50 to 200, 308, 0.02; > 200, 107, 0.03. Median |ln(n*/n)| is 0.4 to 0.65 in every size class.

**Reading.** Two facts that look contradictory and are not.
1. **The trained n is far from most gauges' optima.** Only 19 % of well-fit gauges have their optimum within a factor
   1.25 of the trained n; the median distance is a factor 2, and the direction is systematic: 61 % want slower
   (higher n, median optimum about × 1.65), 19 % faster. The gages_3000 batch put n at 0.04 to 0.06 CONUS-wide (§10),
   and the typical well-fit gauge would rather have 0.07 to 0.10. The signed median ln(n*/n) is +0.50 among the well fit.
2. **Moving there buys almost nothing.** The median gain is 0.014 NSE, and 58 % of well-fit gauges gain under 0.02.
   Only in basins over 200 reaches does the median gain reach 0.03. The landscape is flat in n at most gauges: a factor 2
   in roughness changes the daily NSE by about a hundredth.

Together: **n is weakly identifiable at the daily scale for most of CONUS**, which is why the batch's n is set by
population composition (§9 and §10) rather than by any gauge, and why two seeds or two populations can land a factor 2
apart in n at the same skill. The gauges where n is identifiable, and where the map is actionable, are the large basins
(gain 0.03 median above 200 reaches) and the poorly fit gauges, where the "gain" of 0.027 is a fraction of a large
deficit and the inflow, not the channel, is the problem. This is the population-scale statement of the equifinality
thesis, measured with the gradient rather than assumed: the map of |ln(n*/n)| (panel a) is red over much of the East
and the West Coast, the map of what that costs (gain) is blue almost everywhere except the large rivers.

**Regional pattern** (panel b): the Appalachians, the Southeast, the Northeast, and the Pacific coast want slower
routing (red); the Upper Midwest and the northern Rockies are mixed; the Plains are hollow (poorly fit). The eastern
"slower" band is the population that a gages_3000 batch under-weights relative to its own count (§10 hypothesis 1);
a size- or region-stratified training run is the direct test.

**Next in the chain** (running): the same census over five water years (1996 to 2000, 16 shards, about 4 h) and then
the full 15-year test period (12 shards, about a day), to check that the direction and the flatness hold across years.
The Juniata full-period result (§12) says the n direction holds and the q direction does not.

### 13b. By region (GAGES-II aggregated ecoregions; `experiments/landscape/region_breakdown.py`)

`docs/figures/2026-09-09-conus_n_gap_by_ecoregion_wy2000.png`; tables in `merged/figures/region-ecoregion/REGION.md`
and `region-huc02/`. The seven PUR regions of Feng et al. (2021, GRL, doi 10.1029/2021GL092999) are defined in that
paper's Table S4 (neighbouring HUC2 units combined); the SI was not retrievable, so a provisional geographic grouping
is in the script, flagged, pending the published table.

| Ecoregion | gauges | well fit | median gain | gain > 0.02 | wants slower | wants faster | median |ln(n*/n)| |
|---|---|---|---|---|---|---|---|
| WestPlains | 55 | 15 | 0.049 | 67 % | 47 % | 33 % | 0.60 |
| CntlPlains | 409 | 300 | 0.022 | 51 % | 55 % | 27 % | 0.60 |
| EastHghlnds | 327 | 291 | 0.020 | 50 % | 68 % | 16 % | 0.63 |
| SEPlains | 384 | 297 | 0.020 | 50 % | 56 % | 21 % | 0.57 |
| NorthEast | 291 | 274 | 0.017 | 46 % | 65 % | 11 % | 0.63 |
| WestXeric | 171 | 85 | 0.014 | 34 % | 48 % | 38 % | 0.96 |
| SECstPlain | 69 | 37 | 0.011 | 38 % | 51 % | 35 % | 0.86 |
| MxWdShld | 52 | 33 | 0.010 | 27 % | 33 % | 39 % | 0.71 |
| WestMnts | 607 | 378 | 0.005 | 21 % | 70 % | 12 % | 0.76 |

Readings. (i) The "wants slower" majority holds in every region except the Mixed Wood Shield; the bias in trained n is
CONUS-wide, strongest in the Western Mountains (70 %) and the Northeast (65 %). (ii) Distance and gain anti-correlate
across regions: the Western Mountains sit farthest from their optima (median factor 2.1) and gain least (0.005);
these are snowmelt basins where daily NSE is set by the melt season the inflow controls. (iii) Gains concentrate in
the Plains and the Eastern Highlands (median 0.02 to 0.05, half the gauges above 0.02): the rain-driven, flashier
basins where routing timing matters at the daily step. (iv) The Xeric West and the coastal plain want to move
farthest with the most balanced directions: routing is poorly identified there rather than biased. (v) HUC2 06
(Tennessee) has the largest positive median displacement and the highest share of gauges with gain above 0.02 (79 %).

## 14. Five water years (WY1996 to WY2000), all 2,365 gauges (run `landscape-p21-all-5yr/merged`, 16 shards, 12.7 h)

Maps: `docs/figures/2026-09-09-conus_n_gap_p21_wy1996-2000.png`, `docs/figures/2026-09-09-conus_n_gap_by_ecoregion_wy1996-2000.png`.
The 15-year all-gauge census was held at launch (12 shards would need about 140 GB); scope pending the user.

| | WY2000 (365 d) | WY1996 to 2000 (1,826 d) |
|---|---|---|
| well fit (NSE > 0.3) | 1,710 | 2,124 |
| median NSE at trained point, well fit | 0.733 | 0.754 |
| median gain to own optimum | 0.014 | 0.010 |
| gain > 0.02 | 42 % | 35 % |
| median |ln(n*/n)| | 0.67 | 0.72 |
| within a factor 1.25 | 19 % | 18 % |
| wants slower / faster | 61 % / 19 % | 65 % / 17 % |
| range-bound | 10 | 15 |

Across the 1,675 gauges well fit in both windows the direction of the n displacement agrees 79 % of the time and
ln(n*/n) correlates at 0.63 between windows: the direction is stable, the magnitude is noisy at the gauge level.
The longer window makes more gauges "well fit" (a five-year record is dominated by the floods the model captures)
and shrinks the gain available (0.010 median), which strengthens the §13 reading: n is weakly identifiable at most
daily gauges, and the systematic "slower" bias is CONUS-wide (65 %).

By ecoregion on five years: SEPlains (364 well fit) 74 % want slower with median gain 0.019; CntlPlains 64 % / 0.014;
EastHghlnds 69 % / 0.013; NorthEast 70 % / 0.010; WestMnts 63 % / 0.004 (far from optimum, flat); the two small
regions (WestPlains 44, MxWdShld 47) are balanced in direction. Gains above 0.02 concentrate in the Plains and the
Southeast: 44 to 52 % of gauges there versus 15 % in the Western Mountains.

**Ten most egregious gauges** (largest five-year gain among the well fit; `experiments/landscape-p21-top10_selection.csv`):
05464500 and 05465000 (Cedar and Iowa Rivers, IA; 377 and 453 reaches), 05458500 (Shell Rock, IA), 05594100
(Kaskaskia, IL), 02223500 (Oconee, GA), 02425000 (Cahaba, AL), 01674500 (Mattaponi, VA), 03381500 (Little Wabash,
IL), 06810000 (Nishnabotna, IA), 03371500 (East Fork White, IN). NSE at the trained point 0.40 to 0.55, at the
optimum 0.77 to 0.90 (gains 0.35 to 0.45). Every one wants n × 3.9 to 8.6: slower, by a lot, and most push q to
the floor. Low-gradient agricultural and coastal-plain rivers of the Midwest and Southeast, 100 to 450 reaches.
The routed flow arrives far too early at these gauges; whether the inflow or the channel is at fault is what the
hydrographs (§15, pending) will show. Their (n, q) terrains and hydrographs run next, NSE then KGE.

## 15. The ten most egregious gauges: terrains and hydrographs, NSE objective (run `landscape-p21-top10-nse/merged`)

Five-year window, 25 × 25 (n, q) grid with p = 21, `series: true`; reports in `merged/figures/report_<staid>.png`,
overview `docs/figures/2026-09-09-top10_nse_overview.png`, example `docs/figures/2026-09-09-top10_nse_report_05465000.png`.
Timing below is the cross-correlation lag of each series against the observed daily flow (negative = arrives early).

| gauge | NSE 0 → * | KGE 0 → * | n* / n | inflow vol / obs | inflow early (d) | routed early (d) | routed* early (d) | peak ratio 0 → * |
|---|---|---|---|---|---|---|---|---|
| 01674500 Mattaponi VA | 0.47 → 0.81 | 0.69 → 0.85 | 4.5 (box) | 1.16 | 3 | 3 | 1 | 1.28 → 1.14 |
| 02223500 Oconee GA | 0.47 → 0.88 | 0.69 → 0.81 | 3.9 | 1.08 | 3 | 2 | 0 | 0.87 → 0.78 |
| 02425000 Cahaba AL | 0.51 → 0.89 | 0.76 → 0.94 | 4.3 | 1.03 | 2 | 1 | 0 | 1.26 → 1.01 |
| 03371500 East Fork White IN | 0.55 → 0.89 | 0.78 → 0.93 | 4.5 (box) | 1.06 | 3 | 2 | 0 | 1.08 → 0.93 |
| 03381500 Little Wabash IL | 0.44 → 0.66 | 0.71 → 0.82 | 4.5 (box) | 1.10 | 3 | 3 | 2 | 1.46 → 1.31 |
| 05458500 Shell Rock IA | 0.41 → 0.84 | 0.64 → 0.81 | 4.5 (box) | 1.14 | 2 | 2 | 0 | 1.30 → 1.16 |
| 05464500 Cedar IA | 0.44 → 0.86 | 0.70 → 0.85 | 4.5 (box) | 1.06 | 4 | 3 | 1 | 1.28 → 1.16 |
| 05465000 Iowa IA | 0.44 → 0.81 | 0.72 → 0.89 | 4.5 (box) | 1.03 | 6 | 5 | 1 | 1.25 → 1.12 |
| 05594100 Kaskaskia IL | 0.40 → 0.77 | 0.71 → 0.88 | 4.5 (box) | 1.04 | 2 | 2 | 0 | 1.61 → 1.28 |
| 06810000 Nishnabotna IA | 0.41 → 0.74 | 0.70 → 0.86 | 4.5 (box) | 0.99 | 2 | 2 | 0 | 1.13 → 1.05 |

Readings.
1. **The routed flow sits on the summed inflow.** At every gauge the trained channel removes at most one day of the
   inflow's two-to-six-day lead; the routed hydrograph is the inflow hydrograph. With p = 21 and n = 0.04 these
   100-to-450-reach rivers are routed with almost no delay and almost no attenuation (peaks 1.1 to 1.6 times observed).
2. **Slower routing fixes most of it.** At the optimum the lead falls to zero or one day and the peaks to 1.0 to 1.3
   times observed; NSE rises by 0.3 to 0.4 and KGE by 0.1 to 0.2. Volume is not the problem (inflow within 3 to
   16 % of observed).
3. **The optimum is a bound.** Nine of ten sit at n × 4.48 = exp(1.5), the box edge of these bundles (alpha_max 1.5,
   inherited from the surface bundle); the census with a wider box (§14) found n × 5 to 9 at the same gauges. The
   true optima want slower still.
4. **Inflow early or channel too fast.** From the gauge alone the two are the same statement: the summed inflow reaches
   the gauge two to six days before the observed flow, and the only lever the model has is the channel. What decides
   between them is physics: a 21 m wide channel on a 10,000 km² river is too narrow, so its celerity at n = 0.04 is
   far too high; a width of 100 to 200 m at these rivers would slow the wave without n leaving its physical range.
   This is the p-from-river-size experiment (why-not-at-optimum findings §4, change 1).

Caveat: the (n, q) terrains for these gauges show the trench at the box edge in n; the KGE run (same box) is in
progress and will show whether the KGE optimum agrees in direction. Rerun both with alpha_max 2.3 to locate the optima.

## 16. The same ten gauges with 1 − KGE as the objective (run `landscape-p21-top10-kge/merged`)

Same window, grid and box as §15; objective `kge`. Overview `docs/figures/2026-09-09-top10_kge_overview.png`.

| gauge | NSE-optimum: n* / n, q* / q, NSE*, KGE* | KGE-optimum: n* / n, q* / q, NSE*, KGE* |
|---|---|---|
| 01674500 | 4.48 (box), 4.48, 0.811, 0.847 | 4.48 (box), 0.22, 0.794, 0.840 |
| 02223500 | 3.94, 0.22, 0.884, 0.811 | 3.12, 0.22, 0.849, 0.819 |
| 02425000 | 4.31, 0.22, 0.886, 0.939 | 3.98, 0.25, 0.886, 0.942 |
| 03371500 | 4.48, 0.25, 0.889, 0.925 | 4.48, 1.26, 0.876, 0.923 |
| 03381500 | 4.48, 0.42, 0.661, 0.815 | 4.48, 0.79, 0.658, 0.814 |
| 05458500 | 4.48, 0.76, 0.839, 0.808 | 4.48, 1.30, 0.838, 0.807 |
| 05464500 | 4.48, 0.22, 0.859, 0.853 | 4.48, 0.46, 0.854, 0.851 |
| 05465000 | 4.48, 0.87, 0.809, 0.890 | 4.48, 0.87, 0.809, 0.890 |
| 05594100 | 4.48, 0.22, 0.765, 0.881 | 4.48, 1.17, 0.762, 0.881 |
| 06810000 | 4.48, 0.22, 0.741, 0.864 | 4.48, 1.42, 0.718, 0.852 |

Readings. (i) The two objectives agree on n at all ten gauges: the same direction, and the same box-edge value at
eight of them (3.1 to 4.0 at the other two). (ii) They disagree on q, freely: the q multiplier at the KGE optimum
differs from the NSE one by factors up to 20 (01674500: 4.48 vs 0.22) with NSE and KGE at the two optima within 0.02
of each other. q is the sloppy direction under both objectives. (iii) Optimising KGE costs at most 0.035 NSE and
optimising NSE costs at most 0.012 KGE at these gauges; the two scores move together because the fix is timing and
attenuation, which both measure. Conclusion: the choice of objective does not change the diagnosis at the worst
gauges; it only relocates the optimum along the flat q direction, which is another demonstration that q is not
identified. The wide-box reruns (n up to × 10) will give the true optima for both objectives.

## 17. The ten gauges with the box widened to a factor ten in n (run `landscape-p21-top10-nse/merged`, box ln 10)

All ten optima are now interior (none at the box edge, none range-bound). Overview
`docs/figures/2026-09-10-top10_nse_wide_overview.png`; the box-1.5 results of §15 are kept at `merged-box1.5/`.

| gauge | n* / n | q* / q | NSE 0 → * | KGE 0 → * |
|---|---|---|---|---|
| 01674500 Mattaponi | 4.7 | 6.2 | 0.470 → 0.837 | 0.691 → 0.862 |
| 02223500 Oconee | 3.9 | 0.10 (floor) | 0.471 → 0.884 | 0.686 → 0.811 |
| 02425000 Cahaba | 4.1 | 0.11 | 0.509 → 0.887 | 0.756 → 0.940 |
| 03371500 East Fork White | 5.2 | 0.10 | 0.549 → 0.903 | 0.778 → 0.923 |
| 03381500 Little Wabash | 8.6 | 0.24 | 0.439 → 0.803 | 0.711 → 0.882 |
| 05458500 Shell Rock | 5.4 | 3.1 | 0.409 → 0.842 | 0.642 → 0.824 |
| 05464500 Cedar | 5.3 | 0.10 | 0.438 → 0.885 | 0.700 → 0.871 |
| 05465000 Iowa | 5.6 | 0.10 | 0.438 → 0.881 | 0.723 → 0.929 |
| 05594100 Kaskaskia | 8.1 | 0.93 | 0.403 → 0.818 | 0.705 → 0.868 |
| 06810000 Nishnabotna | 6.0 | 0.10 | 0.407 → 0.770 | 0.703 → 0.879 |

The gauge-optimal roughness at these rivers is 4 to 9 times the trained value: with the trained basin median near
0.03, that is n ≈ 0.12 to 0.26, at or beyond the physical range for a river channel (the parameter range ceiling is
0.25). The median NSE at the optimum rises from 0.825 (box 1.5) to 0.861. q still runs to its floor at six of ten and
to 3 to 6 at two others: sloppy, as before. Reading: a roughness of 0.2 on a main stem is not a channel property; it is
the amount of delay and attenuation the model needs and cannot get from a 21 m wide channel by any other means. The
target for the p-from-river-size experiment is precisely this: recover the same NSE with n in the 0.03 to 0.05 range
and the width doing the work.

## 18. Wide box, both objectives, at the ten gauges (runs `landscape-p21-top10-{nse,kge}/merged`, box ln 10)

| gauge | n*/n (NSE) | n*/n (KGE) | q*/q (NSE) | q*/q (KGE) | NSE* nse-opt / kge-opt | KGE* nse-opt / kge-opt |
|---|---|---|---|---|---|---|
| 01674500 | 4.68 | 4.90 | 6.22 | 0.11 | 0.837 / 0.814 | 0.862 / 0.850 |
| 02223500 | 3.87 | 3.08 | 0.10 | 0.10 | 0.884 / 0.852 | 0.811 / 0.820 |
| 02425000 | 4.06 | 3.96 | 0.11 | 0.16 | 0.887 / 0.887 | 0.940 / 0.942 |
| 03371500 | 5.25 | 4.90 | 0.10 | 0.73 | 0.903 / 0.896 | 0.923 / 0.928 |
| 03381500 | 8.62 | 8.62 | 0.24 | 0.54 | 0.803 / 0.801 | 0.882 / 0.882 |
| 05458500 | 5.40 | 6.40 | 3.07 | 1.03 | 0.842 / 0.803 | 0.824 / 0.832 |
| 05464500 | 5.34 | 8.19 | 0.10 | 3.56 | 0.885 / 0.862 | 0.871 / 0.887 |
| 05465000 | 5.60 | 5.94 | 0.10 | 0.25 | 0.881 / 0.878 | 0.929 / 0.930 |
| 05594100 | 8.07 | 5.31 | 0.93 | 2.13 | 0.818 / 0.785 | 0.868 / 0.887 |
| 06810000 | 6.01 | 5.80 | 0.10 | 0.10 | 0.770 / 0.770 | 0.879 / 0.881 |

All twenty optima are interior. The two objectives agree on roughness: median multiplier 5.4 (NSE) and 5.6 (KGE),
per-gauge ratio median 0.99 across a 0.66 to 1.53 range, and all ten want slower under both. They disagree on the
width exponent by a median factor 1.8 and by up to 35, while the scores at the two optima differ by at most 0.039 NSE
and 0.018 KGE. Choosing the objective relocates the optimum along q and leaves n where it was: the same conclusion as
the narrow box (§16), now with interior optima. The routing correction these gauges need is objective-independent;
the width exponent is not identified by either score.

## 19. Did training converge? The aggregate gradient at the trained point (2026-09-10)

The per-gauge landscape gradient `grad0_n` = dL_gauge / d ln(n multiplier) at the trained point is stored for every
gauge by the five-year census. If training had reached a stationary point for a global scaling of n, these gradients
would cancel across the population: some gauges pulling n up, others down, mean near zero relative to the spread.

| well-fit gauges, five-year window, n = 2,124 | value |
|---|---|
| share with `grad0_n` < 0 (loss falls if n rises) | 78.2 % |
| mean gradient | −0.0519 |
| mean absolute gradient | 0.0639 |
| alignment, abs(mean) / mean(abs) | **0.81** |
| significance of the mean | 3 sigma |
| basins > 200 reaches (n = 140): share negative, alignment | 90 %, **0.97** |
| basins <= 50 reaches (n = 1,580): share negative, alignment | 77 %, 0.56 |

An alignment of 0 means the gauges disagree and their gradients cancel, which is what a converged batch compromise
looks like. An alignment of 1 means they all pull the same way, which is what an unfinished descent looks like. The
population sits at 0.81, and at 0.97 among the large basins. **The aggregate gradient on a global n scaling has not
vanished: it still points toward higher n.**

This is consistent with the optimizer budget. Gradient accumulation over 20 micro-batches at 2 updates per epoch gives
**60 optimizer updates in the whole 30-epoch run**, and the trajectory study (§10, the same run) found n stationary
from update 20 onward while the learning rate decayed 0.005 to 0.001 to 0.0005. So n stopped moving when the steps
became small, not when the gradient became small.

**Confound to eliminate.** These gradients are evaluated on WY1996 to 2000, the test period; training used 1981 to
1995. A model converged on its training data could still show a gradient on a later period, which would be
nonstationarity rather than undertraining. The decisive test is the same aggregate on the training window, and it
needs the study to be able to score on the training period (`period: training`, in progress). If the training-window
alignment is also near 1, training genuinely stopped early and the fix is optimizer steps, not physics. If it is near
0 while the test-window alignment is 0.81, the model converged on what it was shown and the test period wants a
different channel.

**If undertraining is confirmed**, it reorders the ranked changes of `docs/2026-09-09-why-not-at-optimum-findings.md`
§4: more optimizer updates (smaller accumulation, more epochs, or a flatter learning-rate schedule) moves ahead of the
attribute and architecture changes, and the width experiment becomes a test of whether the converged n is physical
rather than a test of whether n can move at all. It does not overturn the equifinality results: q stays unidentified
at 85 % of gauges and n stays unidentifiable below one day of travel time whatever the optimizer does.


---

## 20. How much skill is actually recoverable from the displacement

§19 established that the per-gauge gradients do not cancel: the population wants more roughness than it has. That
raises the obvious question for training. If we could act on it, what would it buy?

Both calculations below use a per-gauge quadratic surrogate in the log-multiplier `c` on Manning's n,

    NSE(c) = NSE* - k (c - a*)^2 ,    k = (NSE* - NSE_0) / a*^2

which is exact at the two points the census measures (the trained point `c = 0` and the Newton optimum `c = a*`)
and interpolates between them. Gauges on the search-box edge are excluded, leaving 1,884 of the 2,365. **Read
§20.4 before quoting any number from §20.1 or §20.2**: the surrogate has a failure mode that widens both answers by
roughly a factor of two, and the ranges given there are the honest ones. Reproduce with
`experiments/landscape/recoverability.py`.

### 20.1 One global roughness level recovers a minority of the gain

Scale every reach's trained n by a single common factor and sweep it:

| global log-multiplier c | n factor | median NSE, all gauges | median NSE, &#124;a*&#124; >= 0.05 |
|---|---|---|---|
| -0.50 | 0.61 | 0.7041 | |
| -0.25 | 0.78 | 0.7272 | |
| **0.00 (trained)** | 1.00 | **0.7507** | **0.7487** |
| +0.275 / +0.455 (best) | 1.32 / 1.58 | **0.7533** | **0.7601** |
| +0.50 | 1.65 | 0.7519 | 0.7595 |
| +1.00 | 2.72 | 0.7414 | 0.7510 |
| +1.25 | 3.49 | 0.7330 | |

Against a per-gauge ceiling of 0.7786, the best single global number captures **9 % of the gain on all gauges and
38 % once the unstable gauges of §20.4 are removed.** A minority either way.

The robust part of this table is the flatness. **A factor 2.7 on the roughness of every channel in the CONUS moves
the continental median efficiency by less than 0.01 in either direction** (-0.009 all gauges, +0.002 filtered).
That statement survives every variant tried and is the one to quote.

A surrogate-free version of the same point, using no quadratic at all, just counting who ends up nearer their own
optimum:

| global shift | moved closer to own optimum | moved further |
|---|---|---|
| +0.25 | 68 % | 32 % |
| +0.50 | 62 % | 38 % |
| +0.75 | 58 % | 42 % |
| +1.00 | 52 % | 48 % |

The population is not globally displaced in a way one number can fix, even though 73 % of gauges want more
roughness and the median gauge optimum is n x 1.71 (p25 x 0.98, p75 x 2.70, p95 x 5.27). The displacement is real
in direction and diffuse in magnitude. Only the largest basins behave like a coherent group: at `n_reach > 200`
the best global multiplier is +0.735 and it lifts the median from 0.778 to 0.823, which is 65 % of that group's
per-gauge gain. That is the same size dependence as §19's alignment statistic, and it is the one place where a
global fix clearly works.

A caution on the mean. Mean NSE collapses under any global shift (0.714 at `c = 0`, 0.351 at +0.25, negative
beyond), but that is extrapolation-driven: a downward parabola has no floor while a real NSE curve flattens. Read
the median rows only.

### 20.2 A per-gauge optimum estimated on one year transfers at 31 to 59 %

The census was also run on WY2000 alone (`landscape-p21-all`, 365 days). Comparing the two on the 1,351 gauges
well-fit and off the box edge in both:

| quantity | value |
|---|---|
| Pearson r of `a*` between windows | 0.646 |
| Spearman r | 0.672 |
| sign agreement | 78.8 % |
| median `a*`, one year | +0.491 |
| median `a*`, five years | +0.524 |
| median &#124;a*_1y - a*_5y&#124; | 0.248 log units (median &#124;a*_5y&#124; = 0.714) |

None of that row block uses the surrogate, so it stands as measured: the direction is a stable property of the
gauge, not window noise. The two medians agree to 0.03 log units and four gauges in five agree on the sign.

Stability of direction is not usable precision, though. Taking the WY2000 optimum and scoring it on the five-year
window, through the surrogate:

| placement | all 1,351 | &#124;a*&#124; >= 0.05 (1,273) |
|---|---|---|
| trained point | 0.7771 | 0.7744 |
| WY2000 optimum, scored on WY1996 to 2000 | 0.7847 (+0.0076) | 0.7898 (+0.0154) |
| WY1996 to 2000 optimum (in sample) | 0.8015 (+0.0244) | 0.8006 (+0.0262) |
| **fraction of the in-sample gain kept** | **31 %** | **59 %** |
| share of gauges improved | 67.7 % | 71.3 % |

So between a third and three fifths of the in-sample gain survives a five-fold change in record length. Both
figures are optimistic: WY2000 sits *inside* WY1996 to 2000, so the two estimates share a fifth of their data. A
genuinely disjoint transfer would be worse.

### 20.3 What this means

Neither route to the displacement is clean. It is not fixable by a global constant (9 to 38 %), and a per-gauge
estimate from a normal length of record loses between two fifths and two thirds of its own promise. The "0.03 NSE
gain" reported in §11 and in `docs/2026-09-09-why-not-at-optimum-findings.md` is an **in-sample upper bound on a
quantity that is only partly recoverable**, and the recoverable part is of order 0.01, not 0.03.

The identifiability statement is the durable one, and it does not depend on the surrogate at all: a large,
systematic, reproducible displacement in Manning's n costs very little in discharge skill, and a factor 2.7 on
every channel in the continent moves the median efficiency by under 0.01. The loss surface has a well-defined
minimum at each gauge; that minimum is shallow and window-dependent, so daily discharge cannot pin the parameter
down even where it does constrain it. This is Beven's argument stated as a measurement rather than an assertion.

For training the priority is unchanged in direction and softened in magnitude. Chasing the aggregate gradient of
§19 is worth a few thousandths to perhaps 0.01 NSE outside the largest basins, so "train longer" is mainly about
making the model's parameters defensible. The exception is `n_reach > 200`, where the displacement is coherent, a
global fix works, and the gain is 0.045.

### 20.4 Why the ranges, and what would close them

`k = (NSE* - NSE_0) / a*^2` is heavy-tailed by construction. As a gauge's optimum approaches its trained point,
`a*` goes to zero and `k` diverges. Measured: median k = 0.027, p99 = 30, max = 5,633, and **the top 1 % of
gauges hold 95 % of the total k**. Those are gauges already at their optimum, where the census measured a tiny
displacement and a tiny gain and the ratio is numerically meaningless. Extrapolating their parabola out to
`c = 0.5` predicts an NSE collapse the gauge would not actually suffer.

Consequences, measured:

| variant | best c* | captured |
|---|---|---|
| as first published | +0.275 | 9 % |
| drop the top 1 % of k | +0.270 | 14 % |
| drop the top 5 % of k | +0.395 | 37 % |
| drop &#124;a*&#124; < 0.05 | +0.455 | 38 % |
| floor the parabola at NSE_0 - 0.3 or - 1.0 | +0.275 | 9 % (unchanged) |

Flooring does nothing because the median gauge never reaches the floor; the sensitivity is entirely in which
gauge sits at the median once the anchored ones are removed. An earlier attempt to compute a curvature-weighted
mean displacement gave +0.004 and was discarded for the same reason: with 95 % of the weight on 1 % of the
gauges, that number describes the tail and not the population.

What closes it: a real grid, not a two-point fit. `landscape-p21-report38` runs 25 x 25 grids at 37 gauges
spanning the gain range. **The first six have landed and they confirm the diagnosis directly.** Taking the n axis
of each grid at the trained q and comparing it against that gauge's two-point parabola:

| staid | n_reach | a*, census | a*, grid | k | NSE at c=+0.5, parabola | grid | NSE at c=+1.0, parabola | grid |
|---|---|---|---|---|---|---|---|---|
| 01362500 | 13 | +1.436 | +1.535 | 0.011 | 0.674 | 0.670 | 0.681 | 0.679 |
| 01447720 | 5 | +1.011 | +1.151 | 0.010 | 0.771 | 0.771 | 0.774 | 0.773 |
| 04160600 | 7 | +0.941 | +0.959 | 0.013 | 0.556 | 0.544 | 0.559 | 0.552 |
| 05458000 | 13 | +1.273 | +1.343 | 0.195 | 0.545 | 0.416 | 0.647 | 0.559 |
| 01532000 | 9 | +0.249 | +0.192 | 0.008 | 0.585 | 0.583 | 0.581 | 0.571 |
| **02120780** | 7 | **-0.022** | **+0.384** | **25.7** | **-6.29** | **0.702** | **-26.13** | **0.640** |

For the five gauges with a well-separated optimum the surrogate is good: median absolute error 0.008 NSE at a
half-log-unit shift, worst 0.13 at the one gauge with strong curvature. At 02120780, the single gauge in this
sample from the pathological class (`|a*| < 0.05`, k in the hundreds or beyond), the parabola is wrong by 7.0 NSE
at `c = +0.5` and by 26.8 at `c = +1.0`, and wrong in the pessimistic direction every time. The grid also puts that
gauge's true optimum at +0.384 with a gain of 0.0044, not at -0.022 with a gain of 0.0125: on a nearly flat
surface the Newton search settled on a spurious nearby point, which is what manufactured the divergent k.

**This makes the filtered column the better estimate, not merely the upper end of a range.** The unfiltered 9 %
and 31 % are artifacts of parabolas that predict an NSE collapse the model does not actually suffer. Pending the
other 31 grids, quote **38 % for the global fix and 59 % for the one-year transfer**, and treat 9 % / 31 % as
superseded. One pathological gauge is a small sample, but the mechanism was predicted in advance from the
definition of k and the measurement matches it in sign, location and magnitude.

**Remaining caveats.** The transfer test in 20.2 is nested and therefore optimistic. Both calculations are on
`nse-batch` optima with NSE scoring; KGE was checked separately (§16) and agrees on the direction. The stiff
direction is n-dominated at every basin size, so both bounds were computed on n alone.
