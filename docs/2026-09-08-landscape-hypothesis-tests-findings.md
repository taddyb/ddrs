# Tests of the batch-compromise hypothesis (landscape spec §7) - findings, 2026-09-08

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

n reaches 0.040 by epoch 10 and does not move for the remaining 20 epochs, through two learning-rate decays.

**Correction 2026-09-10.** This section originally read "about 450 optimizer steps at 45 per epoch" for the
gages_3000 run and "about 870 steps" for the area-balanced one. Both assumed one step per micro-batch and ignored
gradient accumulation. Counted from the checkpoint directories, which are written one per optimizer step, **each
run made exactly 60 updates**: 2,365 gauges give 37 micro-batches per epoch and 1,841 give 29, and at
`grad_accum_steps: 20` both yield 2 steps per epoch over 30 epochs. So epoch 10 is **update 20**, not update 450.

The conclusion is unchanged and in fact strengthened: the two runs made the *identical* number of updates, so step
count cannot be the difference between them. The gages_3000 batch's gradient drives n to 0.04 and holds it there;
the area-balanced batch's gradient does not. **Composition of the training population sets the batch's roughness.**

What the corrected count does change is the reading of "holds it there". n stops moving at epoch 10, which is
exactly where the learning rate drops from 0.005 to 0.001 (`learning_rate: {1: 0.005, 11: 0.001, 21: 0.0005}`).
With only 20 updates spent before that drop and 40 much smaller ones after, "the gradient went to zero" and "the
steps became too small to see" are not distinguishable from this trajectory alone. §21 settles it from the other
side: the gradient had not gone to zero, on either window.

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

Against a per-gauge ceiling of 0.7786, the best single global number captures **about a third of the available
gain**: 32 to 38 % under the three principled repairs of §20.4, at a multiplier of n x 1.5, worth roughly +0.009
median NSE out of an available +0.028. The 9 % from the unbounded parabola is an artifact and is superseded.

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

### 20.2 A per-gauge optimum estimated on one year transfers at about 59 %

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

By the same argument as §20.4 the filtered column is the one to quote: about three fifths of the in-sample gain
survives a five-fold change in record length. Both figures are optimistic: WY2000 sits *inside* WY1996 to 2000, so the two estimates share a fifth of their data. A
genuinely disjoint transfer would be worse.

### 20.3 What this means

Neither route to the displacement is clean. A single global constant recovers about a third of it, and a
per-gauge estimate from a normal length of record loses about two fifths of its own promise. The "0.03 NSE
gain" reported in §11 and in `docs/2026-09-09-why-not-at-optimum-findings.md` is an **in-sample upper bound on a
quantity that is only partly recoverable**, and the recoverable part is of order 0.01, not 0.03.

The identifiability statement is the durable one, and it does not depend on the surrogate at all: a large,
systematic, reproducible displacement in Manning's n costs very little in discharge skill, and a factor 2.7 on
every channel in the continent moves the median efficiency by under 0.01. The loss surface has a well-defined
minimum at each gauge; that minimum is shallow and window-dependent, so daily discharge cannot pin the parameter
down even where it does constrain it. This is Beven's argument stated as a measurement rather than an assertion.

For training the priority is unchanged in direction and softened in magnitude. Chasing the aggregate gradient of
§19 is worth about 0.009 NSE at the population median, so "train longer" is mainly about
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
and 31 % are artifacts of parabolas that predict an NSE collapse the model does not actually suffer.

With 32 grids in hand (26 from `landscape-p21-report38` plus the six large basins of
`landscape-p21-report6-large`) the picture is complete enough to settle it. Two of the 32 are in the pathological
class
(02120780 k = 25.7, 02151500 k = 29.6) and both behave identically: wrong by about 7 NSE at `c = +0.5` and about
29 at `c = +1.0`, pessimistic every time. Away from that class the parabola errs the other way, being mildly
optimistic in 23 of 32 gauges, so the two biases partly cancel in a median. On the 21-gauge subsample the
parabola reproduces the grid-derived captured fraction to one point (84 % against 83 %), which is why the
population sweep is worth repairing rather than abandoning.

What the grids bound directly is how far a gauge ever really falls. Scaling roughness up across all 32:

| shift | median drop in NSE | p90 | max |
|---|---|---|---|
| c = +0.25 (n x 1.28) | -0.012 (a gain) | +0.000 | +0.017 |
| c = +0.50 (n x 1.65) | -0.016 (a gain) | +0.003 | +0.089 |
| c = +1.00 (n x 2.72) | -0.013 (a gain) | +0.083 | +0.150 |

No gauge in a sample deliberately loaded with the worst cases loses more than 0.15 NSE, and the median gauge
gains at every shift. Adding the six large basins raised the worst case from 0.089 to 0.150 (at 01646500, the
Potomac at Little Falls) without changing the medians. A parabola predicting a
fall of 6 or 26 is measurably wrong, and the correct repair is not a floor (the curves are nearly flat, not
steeply bounded) but a sane curvature. Three independent repairs of the 120 affected gauges, 6.4 % of the
population:

| treatment of &#124;a*&#124; < 0.05 | best c* | median NSE | captured |
|---|---|---|---|
| unbounded parabola (as first published) | +0.275 | 0.7533 | 9 % |
| impute the population median k = 0.027 | +0.440 | 0.7604 | 35 % |
| impute k = 0.07, the grid-measured value | +0.390 | 0.7598 | 32 % |
| drop them entirely | +0.455 | 0.7601 | 38 % |

They converge on **a third, at n x 1.5, worth about +0.009 median NSE**. Flooring the parabola does not work
(0.15 gives 12 %, 0.10 gives 19 %) because a floor still places the anchored gauges well below the median while
the grids say their curves barely move. The 9 % figure is the outlier and is withdrawn.

**Remaining caveats.** The transfer test in 20.2 is nested and therefore optimistic. Both calculations are on
`nse-batch` optima with NSE scoring; KGE was checked separately (§16) and agrees on the direction. The stiff
direction is n-dominated at every basin size, so both bounds were computed on n alone.

---

## 21. The training-window control: the model had not converged on its own training data

§19 measured the per-gauge loss gradient at the trained point on the test window (WY1996 to 2000) and found the
population gradients do not cancel: 78 % point the same way and the alignment `|mean(g)| / mean(|g|)` is 0.81. The
confound named there was nonstationarity. Training used 1981 to 1995; a model perfectly converged on that period
could still show a gradient on a later one, and that would say nothing about the optimizer.

`landscape.period: training` (commit 674d50b) lets the study score on the training window. `landscape-p21-all-trainwin-diag`
re-runs the trained-point gradient over WY1991 to WY1995, the last five training years, matching the test run's
window length exactly.

**Final result, 2,170 well-fit gauges of 2,365 scored, all 14 HUC regions, run complete.** An interim read at
806 eastern gauges gave the same answer, and adding the West and the arid interior moved the headline shares by
under a point, so the conclusion is not a regional artifact. The tables below use the robust statistics of §21.3;
the mean-based `alignment` values that appeared in earlier drafts of this section are superseded.

| statistic | testing window (WY1996-2000) | training window (WY1991-1995) |
|---|---|---|
| well-fit gauges | 2,124 | 2,170 |
| median dL/d ln n | -0.0130 | -0.0104 |
| **share where loss falls if n rises** | **78.2 %** (z = 26.0) | **78.0 %** (z = 26.1) |
| median-based alignment | 0.749 | 0.694 |
| 10 % trimmed alignment | 0.940 | 0.953 |
| share negative, n_reach <= 50 | 76.8 % | 76.6 % |
| share negative, 51 to 200 | 80.0 % | 80.5 % |
| share negative, n_reach > 200 | 90.0 % | 87.1 % |
| trimmed alignment, n_reach > 200 | 1.000 | 0.998 |

Every share is more than eight standard deviations from the 50 % a converged optimum would give (p < 1e-12).

Paired, on the 2,059 gauges well-fit in both windows, which holds the population fixed:

| | testing | training |
|---|---|---|
| share where loss falls if n rises | 78.4 % (z = 25.8) | **78.9 %** (z = 26.2) |
| median-based alignment | 0.762 | 0.711 |
| 10 % trimmed alignment | 0.941 | **0.959** |
| Spearman r between the windows | - | 0.691 |
| sign agreement | - | **84.2 %** |

### 21.1 What it settles

**Nonstationarity is refuted as the explanation.** On the same 2,059 gauges the training window is if anything
marginally more unanimous than the test window: 78.9 % against 78.4 % on the sign share, 0.959 against 0.941 on the
trimmed alignment. The training period wants more roughness exactly as the test period does. The trained point is not a stationary point of the
objective the model was actually fitted to.

**The user's hypothesis is supported.** Stated on 2026-09-07 as "each gauge's loss isn't at its absolute minimum but
a local minimum achieved when training in a large batch", and on 2026-09-10 as "we didn't finish training and we
never correctly converged on our parameters". The second reading is the one the data support: this is not a batch
compromise among gauges pulling in different directions, because they are not pulling in different directions. At
large basins 98 % of them pull the same way and the aggregate is 99 % as large as the mean absolute gradient. A
converged batch optimum would show alignment near zero. Descent simply stopped.

The mechanism was identified in §19 and needs no new evidence: gradient accumulation over 20 micro-batches at 2
updates per epoch gives **60 optimizer updates in the entire 30-epoch run**, and §10's trajectory shows n stationary
from update 20 while the learning rate decayed 0.005 to 0.001 to 0.0005. n stopped moving when the steps became
small, not when the gradient did.

Pearson 0.22 against Spearman 0.75 is not a contradiction: the test-window gradients have a heavy tail (sd 0.736
versus 0.059 on the training window), so a handful of extreme gauges destroy the linear correlation while the rank
structure holds. Sign agreement of 88 % is the practical statement.

### 21.2 What it does not settle, and what it is worth

It does not make the model meaningfully more skillful. §20 bounds the prize: the recoverable part of the
displacement is of order 0.01 NSE, and a single global roughness level captures about 38 % of it. So "run more
optimizer updates" is the right fix for the right reason, and the reason is that the learned parameters should be
the ones the objective actually asks for, not that the hydrographs will visibly improve. The exception remains
`n_reach > 200`, where alignment is 0.99, the displacement is coherent, and the gain is about 0.045.

It also does not touch the identifiability results. q stays unidentified at 85 % of gauges and n stays
unidentifiable below one day of channel travel time whatever the optimizer does. Those are properties of daily
discharge as an observation, not of the fit.

**Ranked consequence.** More optimizer updates (a smaller accumulation factor, more epochs, or a flatter learning
rate schedule) moves to the top of `docs/2026-09-09-why-not-at-optimum-findings.md` §4, ahead of the attribute and
architecture changes. The width-from-river-size retrain becomes a test of whether a converged n is physical rather
than a test of whether n can move at all.

**Confirmed continentally.** The concern that the eastern sample was where earlier sections put the largest
displacement did not materialise: adding HUC 04 through 14 changed the paired sign share from 79.6 % to 78.9 % and
the trimmed alignment from 0.948 to 0.959. Reproduce with
`experiments/landscape/trainwin_compare.py <testing-merged> <training-merged> --out <dir>`.

### 21.3 Correction: the alignment statistic was not robust, and the robust version is stronger

At 806 gauges the training-window alignment `|mean(g)| / mean(|g|)` read 0.781 overall and 0.990 at basins over
200 reaches. At 1,508 gauges the same statistic reads 0.409 and **0.149**, while the median gradient and the share
of negative gradients barely moved. A handful of gauges with extreme positive gradients entered the large-basin
bin (sd 1.38 on 98 gauges) and destroyed a ratio of means. The test-window figures quoted in §19 and §21 have the
same weakness: sd 2.78 on the 140 large basins there.

`|mean| / mean|g|` was the wrong statistic. Per-gauge gradients span orders of magnitude with heavy tails, so any
ratio of means is set by a few gauges. Three robust replacements, all measured on both windows:

| | share where loss falls if n rises | median-based alignment | 10 % trimmed alignment |
|---|---|---|---|
| **training WY1991-1995, all** | 79.1 % | 0.747 | 0.945 |
| testing WY1996-2000, all | 78.2 % | 0.749 | 0.940 |
| **training, n_reach <= 50** | 76.7 % | 0.698 | 0.890 |
| testing, n_reach <= 50 | 76.8 % | 0.696 | 0.908 |
| **training, 51 to 200** | 84.7 % | 0.906 | 0.988 |
| testing, 51 to 200 | 80.0 % | 0.789 | 0.966 |
| **training, n_reach > 200** | 90.8 % | 0.973 | 1.000 |
| testing, n_reach > 200 | 90.0 % | 0.962 | 1.000 |

Every share is more than seven standard deviations from the 50 % a converged optimum would give (p < 1e-12 on a
binomial test against 0.5), at every basin size, on both windows.

**The conclusion of §21 is unchanged and better supported.** The two windows now agree to within a percentage
point on the sign share and to within 0.05 on both robust alignments, which is a far cleaner refutation of
nonstationarity than the mean-based numbers ever were. After trimming the extreme tenth of each tail, essentially
the whole gradient mass points one way (0.94 overall, 1.00 at large basins). Descent stopped; the gauges are not
pulling against each other.

Paired on the 1,458 gauges well-fit in both windows, which is the cleanest form of the statement because it holds
the gauge population fixed:

| | testing WY1996-2000 | training WY1991-1995 |
|---|---|---|
| share where loss falls if n rises | 78.7 % (z = 21.9) | 79.6 % (z = 22.6) |
| median-based alignment | 0.788 | 0.753 |
| 10 % trimmed alignment | 0.933 | 0.948 |
| sign agreement between the two windows | - | 85.5 % |

`experiments/landscape/trainwin_compare.py` prints all of these (commit ace9411); the mean-based line is retained
and labelled as not robust.

**What to quote from here on:** the sign share and the trimmed alignment. Treat every `|mean| / mean|g|` figure in
§19 and in §21's first two tables as superseded, including the 0.813 and 0.974 that motivated this whole line of
work. Their qualitative reading survives; their values do not.

### 21.4 What the training-window run cannot answer, and the trap in trying

`landscape-p21-all-trainwin-diag` is a gradient-only run: `grid: 0`, `newton_iters: 0`. It measures the gradient,
Hessian and daily series at the trained point and performs **no optimum search**, so its `alpha_n_star` column is
identically 0 for every gauge. That is by design and it is what §21 needs, but it means the run cannot answer the
natural follow-up question, "where would a single global roughness multiplier go if it were chosen on the training
window rather than the test window?"

Attempting it anyway produces confident nonsense: reading the zero column as a per-gauge optimum gives a median
displacement of 0.000, a cross-window sign agreement of 27.7 %, and the conclusion that the training-window
optimum realises 0 % of the achievable test gain at 0 % of gauges. All four numbers are artifacts of an unpopulated
column. Anything joining the two censuses on `alpha_n_star` must check that both sides ran a Newton search.

**What is answerable from the gradient alone, and is already in §21.3:** direction. Both windows point the same way
at 79 to 80 % of gauges and agree with each other gauge by gauge 85.5 % of the time, so more optimizer updates move
roughness toward what the test period also wants at roughly six gauges in seven. That is enough to justify the
training change.

**What would be needed for the quantitative version:** the same all-gauge bundle with `period: training` and
`newton_iters: 12`, which costs what the five-year testing census cost (about 40 core-hours) rather than what the
gradient-only run costs. Worth running only if the transfer magnitude, not the direction, becomes load-bearing for
the paper.


---

## 22. Correction: the two-model skill comparison, done on a common gauge population

§8 and §9 report the two p = 21 models as scoring median NSE 0.700 (area-balanced population) and 0.720
(gages_3000 population), and the abstract draft built its strongest sentence on those being indistinguishable.
**Those two numbers are on different test populations** and must not be compared: 0.700 is the area-balanced
model on its own 1,841-gauge test set and 0.720 is the gages_3000 model on its own 2,365-gauge set. Each model's
test set is derived from its own training list, so the two differ in composition and difficulty.

Both runs' eval stores cover the identical window, 1995-10-01 to 2010-09-30, 5,477 days
(`.ddrs/runs/<id>/eval/predictions.zarr`), so the paired comparison is directly computable on the 1,323 gauges
present in both.

| on the 1,323 shared gauges, same 15 years | area-balanced model | gages_3000 model |
|---|---|---|
| median NSE | **0.7384** | **0.7330** |
| median KGE | 0.7695 | 0.7711 |

| paired per-gauge difference, gages_3000 minus area-balanced | |
|---|---|
| median | -0.0042 |
| mean | -0.0169 |
| median absolute difference | 0.0214 |
| gages_3000 better at | 37.5 % of gauges |
| sign test | z = -9.1 |

**What changes.** The 0.020 NSE gap in §8 and §9 was an artifact of the differing test populations, and it ran the
wrong way: on a common set the area-balanced model is marginally *better*, not 0.020 worse. The gap between the
models is 0.005 in the median, which is a quarter of the median gauge-to-gauge difference between them (0.021), so
in magnitude their skill is practically identical. But the difference is consistent in sign, with the area-balanced
model ahead at 62.5 % of gauges at z = -9.1, so **"statistically indistinguishable" is not the right phrase**.
Write "practically identical median skill, 0.738 against 0.733" and, if the direction matters, note that the
smaller, larger-basin population is very slightly ahead.

**What survives, and it is the part that matters.** The equifinality argument never needed the two models to be
exactly equal, only close. Two models whose median skill differs by 0.005 carry basin-median roughness differing by
about a factor of two and sit on opposite sides of the per-gauge optima, and they differ only in the population of
gauges each was shown. That claim stands.

**Caveat on the roughness factor, which is thinner than the skill comparison.** It rests on the 30 gauges present
in both 41-gauge censuses of §9 (median trained-n ratio 0.48, so a factor 2.1) plus the Juniata pair (0.100 against
0.040, a factor 2.5). Thirty gauges is a small basis for a headline number. Before the abstract is submitted this
should be recomputed across the full shared population, which is cheap: both runs' trained fields are already on
disk and only the basin-median n per gauge is needed, not a landscape.

---

## 23. The factor-two roughness difference, measured across all 1,323 shared gauges

§22 flagged that the headline roughness difference between the two p = 21 models rested on the 30 gauges common
to the two 41-gauge censuses plus the Juniata pair. Bundle `landscape-pfixed-all-nfield` harvests the trained
field at every gauge of the area-balanced model. The trained field is a KAN forward pass and does not depend on
the evaluation window, so the bundle runs a one-year window with no Newton search and no series purely to collect
`n0` and `q0` cheaply; its loss and NSE columns are one water year and must not be compared to the five-year
censuses. 1,821 of 1,841 gauges produced a field, 20 skipped for having no observations in the one-year slice.

Basin-median trained Manning n, area-balanced model over gages_3000 model, on the 1,323 shared gauges. That is the
same population as the skill comparison of §22, which is a useful consistency check: the eval overlap and the
trained-field overlap are the same 1,323 gauges.

| quantile of the ratio | value |
|---|---|
| p10 | 1.15 |
| p25 | 1.57 |
| **p50** | **2.39** |
| p75 | 2.65 |
| p90 | 3.32 |
| geometric mean | 2.10 |
| share where the area-balanced model is rougher | **96.6 %** |

Median basin-median n: **0.1036 area-balanced against 0.0435 gages_3000**. The width exponent differs by more:
median q 0.402 against 0.121, a ratio of 3.33.

The ratio is almost independent of basin size, which is what makes it a population effect rather than a
composition artifact:

| basin size | gauges | median n ratio |
|---|---|---|
| n_reach <= 50 | 924 | 2.38 |
| 51 to 200 | 246 | 2.40 |
| > 200 | 153 | 2.39 |

**This confirms the original claim and strengthens its basis by a factor of 44.** The "factor 2.4" quoted from the
30-gauge comparison was accurate: the population figure is 2.39, the two models disagree in the same direction at
96.6 % of shared gauges, and the ratio is flat to within 0.03 across two orders of magnitude in basin size. Unlike
the skill comparison of §22, this number needed no correction.

An interim read on the eastern 570 gauges gave 2.46 and 99.1 %. Adding the West lowered the sign agreement to
96.6 % and widened the lower tail (p10 from 1.21 to 1.15, p25 from 2.21 to 1.57) while leaving the median
essentially unchanged, so the arid interior holds a minority of gauges where the two models nearly agree.

**Why the first attempt aborted, and the fix.** `src/experiment/landscape/objective.rs` panicked when *every*
window for a gauge came up empty, which killed the whole arm thread and lost every remaining gauge in that shard:
720 of 1,841 were produced. The run-level gauge filter guarantees coverage over the configured training and
testing window, not over whatever shorter sub-window a landscape study selects, so a gauge can pass the filter and
still be empty in the slice. That is an expected per-gauge data condition, not a bug in the data, and a whole-arm
panic is the wrong response. The study now skips the gauge, logs
`[arm] <staid> skipped: no valid observations in the window <start> .. <end>`, records the count in
`manifest.notes`, and continues (commit eb3f159). **Any earlier study using a sub-window narrower than its run's
configured window could have been truncated the same way**, and would have looked like a crashed shard rather than
a short population.

---

## 24. The NSE optimum does not cost KGE

A standing worry about acting on the displacement is that the per-gauge optima were found on an `nse-batch`
objective, and NSE is maximised at a simulated variance below observed, so chasing it could degrade KGE. Computed
directly from the stored daily series at the 32 gauges that have grids, comparing the trained point against the
gauge's own optimum:

| | value |
|---|---|
| median NSE change | **+0.0267** |
| median KGE change | **+0.0121** |
| NSE improves at | 100 % of gauges |
| KGE improves at | **81 %** of gauges |
| NSE up while KGE falls | 6 of 32 |

| basin size | gauges | median dNSE | median dKGE | KGE falls at |
|---|---|---|---|---|
| n_reach <= 50 | 13 | +0.012 | +0.007 | 3 of 13 |
| 51 to 200 | 9 | +0.325 | +0.078 | 1 of 9 |
| > 200 | 10 | +0.049 | +0.025 | 2 of 10 |

So the two objectives broadly agree about which way roughness should move, and the "train longer" recommendation
of §21 does not trade one metric for the other at four gauges in five. The exception is real but small: at 6 of 32
gauges, including the Potomac at Little Falls (NSE 0.877 to 0.912 while KGE falls 0.839 to 0.824), the NSE optimum
costs KGE. This agrees with §16, which found the two objectives pick the same direction at the population level.

**Caveat.** These 32 gauges were selected to span the gain range (14 worst, 8 largest, 8 mid, 8 near-optimal), so
they over-represent large displacements. The median changes here are not population estimates; the population
figures are in §20.


---

## 25. It is not a weak gradient, it is a flat valley: where roughness curvature lives

"The gradient with respect to roughness is weak" is imprecise and sends the wrong fixes. Measured on the
2,124 well-fit gauges of the five-year census:

**The batch gradient is not noisy.** Per-gauge `dL/d ln n` has a 5-95 % trimmed effect size
`|mean| / sd` of 0.734, so a batch of 256 gauges sees the direction at a signal-to-noise ratio of about **12**.
Adam normalises by gradient magnitude, so a uniformly small but consistent gradient is not a problem for it.

**Restricting to identifiable gauges does not help.** Keeping only travel times over one day changes the effect
size from 0.734 to 0.717. Short-travel-time gauges actually have the *highest* effect size (1.66 below half a day,
1.04 from half a day to one day, against 0.39 beyond four days): they sit far from their optima and push
consistently, on a surface where moving buys nothing.

**The width exponent is not stealing the stiffness.** Curvature along `ln n` with q frozen (`H_nn`, median
0.00109) against q free to re-optimise (the Schur complement `H_nn - H_nq^2/H_qq`, median 0.00128) differ by a
factor of **0.99**. The n-q coupling is negligible at every basin size, so pinning q the way p was pinned would
not sharpen n.

So the direction is clear and the valley is flat. The quantity to attack is the **curvature**, not the gradient.

### 25.1 Ninety percent of the curvature is in one percent of the gauges

| | share of the population's total &#124;H_nn&#124; |
|---|---|
| top 1 % of gauges (21 of 2,124) | **89.9 %** |
| top 5 % (106) | 95.3 % |
| top 10 % (212) | 96.7 % |
| top 50 % (1,062) | 99.6 % |

| basin size | gauges | median &#124;H_nn&#124; | share of total curvature |
|---|---|---|---|
| n_reach <= 50 | 1,580 | 0.0127 | 6.6 % |
| 51 to 200 | 404 | 0.0293 | 67.1 % |
| > 200 | 140 | 0.0734 | 26.3 % |

The continental channel parameter is, in effect, determined by a few dozen gauges. Everything else contributes a
confident push on a surface too flat to care.

### 25.2 What creates curvature: the hydrograph's time derivative

Spearman rank correlations with `|H_nn|`:

| predictor | rho |
|---|---|
| gauge reach slope | **-0.435** |
| flashiness (daily dQ/dt magnitude) | **+0.314** |
| total channel length | +0.314 |
| drainage area | +0.270 |
| reach count | +0.261 |
| mean flow | +0.183 |

Within small basins alone (`n_reach <= 50`, n = 1,580), flashiness against `|H_nn|` is **+0.408**, stronger than
in the pooled set, so this is not a size proxy.

**The mechanism this implies.** Roughness acts on the hydrograph almost entirely through travel time: for Manning
flow at fixed discharge, velocity goes as `n^-3/5`, so travel time goes as `n^3/5`. To first order a change in
roughness shifts the routed series in time, and a pure time shift changes the series by `dQ = -(dQ/dt) dtau`. The
loss therefore picks up curvature in proportion to **the mean square of the hydrograph's time derivative**, scaled
by the channel's travel time. That predicts exactly what is measured: stiffness rises with flashiness and with
channel length, and falls with slope (steep reaches are fast, so `dtau/d ln n` is small).

It also explains the flatness directly. Daily averaging is a low-pass filter applied to precisely the quantity
that gives roughness its leverage. Whatever timing information lives below the daily scale, which is most of it at
a basin whose wave crosses in hours, is removed before the loss ever sees it.

### 25.3 What follows for strengthening it

Ranked by the mechanism above, not by convenience:

1. **Sub-daily observations.** The only change that attacks `⟨(dQ/dt)^2⟩` at its source. The hourly store, the
   AORC precip source and the disaggregation head already exist. This is the principled fix and the strongest
   available test of whether the flatness is physics or sampling.
2. **A time-derivative term in the objective.** Even at daily resolution, adding a penalty on `d/dt` mismatch
   projects the loss onto what roughness controls, instead of onto the volume and correlation that the inflow
   already supplies. Cheap to implement in `src/training/loss.rs` alongside the existing kinds.
3. **Curvature-weighted or size-restricted training.** Legitimate, and revealing rather than merely tuning, since
   §9 and §23 showed the training population sets the roughness. But note what 25.1 means: weighting by curvature
   is close to training the channel on a few dozen gauges, with the variance that implies.
4. **Not worth doing:** pinning q to sharpen n (factor 0.99), or dropping short-travel-time gauges to clean up the
   gradient (effect size 0.734 to 0.717). Both are ruled out above.

**Caveat.** The mechanism in 25.2 is inferred from correlations plus the Manning scaling, not from a controlled
experiment. The direct test is to recompute `H_nn` on an hourly-resolution objective at the same gauges and check
that it rises, most at the flashy small basins where daily curvature is lowest.

---

## 26. Pre-registered prediction: what a time-derivative loss term should do to the curvature

Written **before** the probe was run, so the comparison is a test and not a post-hoc fit.

§25 established that curvature in `ln n` is governed by the mean square of the hydrograph's time derivative:
`d2L/d(ln n)^2 = 2 (0.6 tau)^2 <(dQ/dt)^2> / sigma^2`. Tested against the measured Hessian at 2,365 gauges
(training window, series and Hessian from the same run) this parameter-free expression gives **Spearman +0.710**
with a log-log slope of **0.82**, against a predicted slope of 1.0. The absolute scale is off by about 12x
(median measured/predicted 0.086), which is expected: a pure time shift ignores attenuation, and the travel time
is an order-of-magnitude proxy. The ranking is what the mechanism claims, and the ranking holds. Both factors
carry weight independently: `<(dQ/dt)^2>/sigma^2` alone gives +0.533, `tau^2` alone +0.301.

### The prediction

Add to the objective a term on the series' time derivative,

    L' = L_nse + lambda * mean_pairs( (dsim/dt - dobs/dt)^2 ) / sigma_d^2

with `sigma_d` the standard deviation of the observed daily differences. Under the same time-shift argument the
derivative term contributes curvature through `<(d2Q/dt2)^2>/sigma_d^2` where the level term contributes
`<(dQ/dt)^2>/sigma^2`, so

    stiffening = 1 + lambda * R,      R = [<Q''^2>/sigma_d^2] / [<Q'^2>/sigma^2]

Measured from the stored daily series at 2,365 gauges: **R has median 5.33** (p25 2.79, p75 13.25), and it rises
with basin size (4.35 at `n_reach <= 50`, 9.23 at 51 to 200, 15.05 above 200).

**At `lambda = 0.5` the prediction is therefore:**

| quantity | predicted |
|---|---|
| median stiffening of &#124;H_nn&#124; | **3.7x** (p25 2.4x, p75 7.6x) |
| stiffening at `n_reach <= 50` | 3.2x |
| stiffening at 51 to 200 | 5.6x |
| stiffening at `n_reach > 200` | 8.5x |
| direction | curvature rises at essentially every gauge; the gain rises with basin size |

### What would refute it

- Median stiffening below about 1.5x: the derivative term does not reach the mechanism the correlations implied.
- Stiffening that does not increase with basin size: `tau^2` is not really the second factor, so the time-shift
  picture is wrong even if the curvature happens to rise.
- Curvature rising while the per-gauge optimum `alpha_n_star` moves a long way: then the term is not sharpening
  the existing valley, it is choosing a different one, and the two objectives disagree about the answer rather
  than about the confidence. Worth knowing either way, but it is a different claim from the one registered here.

### Design of the probe

`objective: nse-deriv`, `deriv_weight: 0.5`, run at the **same checkpoint, same gauges and same window** as
`landscape-p21b-all-testwin-diag`, which supplies the `nse-batch` baseline. No retraining: this measures whether
the valley is deeper under the candidate objective before any training run is spent on it. If it passes, the term
goes into `src/training/loss.rs` and gets a retrain; if it fails, the cost was a few hours.

---

## 27. The retrain converged: §21 is closed

§21 measured that the 60-update model was not at a stationary point, on either window. The fix it implied was
optimizer budget, and §20 predicted the prize was about +0.009 median NSE concentrated in large basins. Both were
tested by the 2026-09-10 retrain (`grad_accum_steps` 20 to 4, `epochs` 30 to 50, lr flattened to 0.005 through
epoch 20, giving **500 optimizer updates against 60**). Everything else held fixed: same data, architecture, seed
and population.

### The convergence test, before and after

Paired on the gauges well-fit in both windows, using the robust statistics of §21.3:

| | 60 updates (§21) | **500 updates** |
|---|---|---|
| shared gauges | 2,059 | 2,076 |
| share where the loss falls if n rises, **testing** | 78.4 % (z = 25.8) | **53.6 %** (z = 3.2) |
| share where the loss falls if n rises, **training** | 78.9 % (z = 26.2) | **53.1 %** (z = 2.8) |
| 10 % trimmed alignment, testing | 0.941 | **0.008** |
| 10 % trimmed alignment, training | 0.959 | **0.057** |

Per window across all well-fit gauges, by basin size (testing window):

| basin size | share negative | trimmed alignment | z |
|---|---|---|---|
| all (2,138) | 53.9 % | 0.017 | 3.6 |
| n_reach <= 50 | 53.9 % | 0.047 | 3.1 |
| 51 to 200 | 53.9 % | 0.223 | 1.6 |
| > 200 | 53.1 % | 0.355 | **0.7** |

**The aggregate gradient has essentially vanished.** The population sign share moved from 78 % to 54 %, and the
trimmed alignment from 0.94 to 0.008, which is what a batch optimum looks like: the per-gauge gradients now
cancel. At basins over 200 reaches, where the 60-update model was most lopsided (90 % negative, alignment 1.000),
the retrained model is at z = 0.7, statistically indistinguishable from a stationary point.

A residual tilt remains, 54 % rather than 50 %, at z = 3.2. On 2,076 gauges that is detectable but it is 1/8 the
effect size of before, and it is the scale at which "not exactly stationary" stops being the dominant story.

### What this settles

1. **The undertraining diagnosis was right, and the fix worked.** This is the instrument's first genuine
   out-of-sample prediction: the landscape said the model had not converged, the prescribed change was made, and
   the landscape now says it has.
2. **§20's magnitude prediction also held.** Predicted about +0.009 median NSE with the gain concentrated in large
   basins; measured +0.0062 paired median, +0.018 in the medians (0.7200 to 0.7376), improved at 70.7 % of gauges,
   with the gain rising from +0.005 below 1,000 km2 to +0.020 at 10,000 to 30,000 km2.
3. **It changes what the remaining displacement means.** Before, "gauges sit away from their own optima" was
   ambiguous between an unfinished descent and a genuine batch compromise. That ambiguity is now resolved: the
   descent is finished, so what remains **is** the compromise. Selective equifinality is the correct reading of
   the residual, not undertraining.
4. **It reopens §10.** "Composition, not step count" was concluded from two runs that had both made exactly 60
   updates (see the §10 correction). With 500 updates the same gages_3000 population lands at median trained
   n = 0.080 rather than 0.049, wandering between 0.073 and 0.100 over the last 40 epochs without the loss
   objecting. Whether the two populations still differ by a factor of two once BOTH are properly trained is now
   an open question, and it bears directly on the abstract's central claim (§22, §23).

### Caveat

The residual z of 3.2 is small but real, and the trimmed alignment at large basins (0.355 testing, 0.172 training)
is higher than the population value, so the biggest basins are the least settled even though their sign share is
closest to 50 %. That is consistent with them having the most curvature and therefore the most to say.

---

## 28. The curvature probe: the derivative term deepens the valley, as predicted

The prediction registered in §26 was tested by running the landscape with `objective: nse-deriv`,
`deriv_weight: 0.5` at the **same checkpoint, gauges and window** as `landscape-p21b-all-testwin-diag`, so the
only difference between the two censuses is the objective. 2,365 paired gauges.

### Result

| quantity | predicted (§26) | **measured** |
|---|---|---|
| median stiffening of &#124;H_nn&#124; | 3.7x | **3.04x** |
| p25 / p75 | 2.4x / 7.6x | 1.89x / 5.98x |
| share of gauges stiffer | essentially all | **92.8 %** |
| `n_reach <= 50` | 3.2x | 2.71x |
| 51 to 200 | 5.6x | 4.16x |
| `n_reach > 200` | 8.5x | 4.92x |

Absolute curvature, median `|H_nn|`: **0.0455 to 0.1670**. The share of gauges flat enough that a factor of two in
roughness costs under 0.005 of loss falls from **35 % to 16 %**.

**The registered prediction passes.** The median is 82 % of the predicted value, comfortably above the 1.5x
refutation bar, and the direction of the size scaling is right. The magnitude of that scaling is over-predicted at
the largest basins (4.92x measured against 8.5x), which is the one place the simple time-shift argument is weakest:
at long travel times the routing also attenuates rather than purely translating, so `<Q''^2>` over-states the
available leverage.

### The term sharpens the existing valley, it does not choose a different one

This was the third refutation condition in §26, and it is the one that decides whether the term is usable. Gradient
at the identical parameter point under each objective:

| | share where the loss falls if n rises | 10 % trimmed alignment |
|---|---|---|
| nse-batch | 54.1 % | 0.059 |
| nse-deriv | 54.5 % | 0.024 |

Sign agreement between the two objectives, gauge by gauge: **88.7 %**. Both objectives agree that the converged
model is close to stationary, and they agree gauge by gauge about which way to move. So the derivative term is
adding **confidence, not disagreement**: the minimum stays where it was and the surface around it becomes steeper.
That is exactly the property needed for it to sharpen identifiability without changing the answer.

### Caveats

- Both censuses ran `newton_iters: 0`, so this compares curvature and gradient at the trained point, not the
  location of the two optima directly. The gradient agreement above is the proxy, and it is a good one at a point
  this close to stationary, but a Newton run under each objective would be the direct measurement.
- The probe measures the objective's curvature at a point reached by training on a *different* objective. For a
  quadratic the Hessian does not depend on where it is evaluated, and the model is near-stationary, so this is a
  small correction, but it is not zero.
- `lambda = 0.5` makes the two terms roughly co-equal in magnitude, since the median gauge's realised derivative
  contribution is 2.5x its level contribution. That is a deliberate choice and not a tuned one; no other value was
  tried.
- The derivative term concentrates the batch somewhat more than the level term (effective sample size 151 of 2,365
  gauges against 295 by realised contribution), which is the same regime, not a new pathology.

### 28.1 The same question at one gauge: does a derivative-trained model end up better placed?

The probe measures the curvature of the new objective. The complementary question is whether a model *trained*
with the term ends up nearer its own optimum. `experiments/juniata-deriv-compare` answers it at the Juniata
(USGS 01567000, 213 reaches): two models identical apart from `experiment.loss.kind`, both at 300 optimizer steps,
both scored under the **same** `nse-batch` objective with a 25 x 25 grid, so the comparison is of trained points
and not of two different surfaces.

| | trained NSE | NSE at its own optimum | displacement `alpha_n_star` | n factor from optimum |
|---|---|---|---|---|
| trained on `nse-batch` | 0.8653 | 0.8670 | **-0.268** | 1.31x |
| trained on `nse-batch-deriv` | 0.8671 | 0.8672 | **-0.049** | **1.05x** |

The derivative-trained model sits **5.4 times closer to its own optimum**, with essentially nothing left to gain
(+0.0001 against +0.0017). Its skill is also marginally higher, though at 0.002 that is not the point and should
not be quoted as one.

This is one gauge, so it is an illustration rather than evidence: a single basin cannot show a population effect,
and the Juniata is large enough (213 reaches) to be among the better-constrained gauges to begin with. The
population version of this table is what the retrain now running will produce.

**One number that does not fit the simple story.** The stiff eigenvalue at the optimum, measured under
`nse-batch`, is *lower* for the derivative-trained model (0.457 against 0.593). That is not a contradiction, since
the probe measured curvature under the derivative objective while this measures the `nse-batch` surface at a
different point, but it is a reminder that "better placed" and "in a sharper basin" are separate claims. Only the
first is demonstrated here.

---

## 29. Why the width exponent collapses to its bounds

Training on `nse-batch-deriv` drove q to the boundaries: reaches with `q < 0.01` went from 3.5 % to 64.2 %, the
share inside the Leopold and Maddock band 0.1 to 0.6 fell from 33.0 % to 9.3 %, and 7.9 % sit at the upper bound,
so **72 % of reaches are pinned at one end or the other**. Manning's n, by contrast, stayed off both bounds
entirely. Two candidate causes: p being fixed at 21 forcing q to compensate, or q sitting on a weak gradient path
of the kind n was on.

### The q direction is concave, not merely flat

Measured at the same parameter point under both objectives (2,365 paired gauges, the `nse-batch`-trained
500-update model):

| | nse-batch | nse-deriv |
|---|---|---|
| share where the loss falls if q RISES | 54.2 % | 58.6 % |
| trimmed alignment along q | 0.257 | 0.426 |
| median &#124;H_qq&#124; | 0.00107 | 0.00366 |
| **share with `H_qq <= 0`, i.e. NO interior minimum in q** | **50 %** | **53 %** |

Sign agreement between the two objectives along q is 87.8 %, so they largely agree about direction.

**At half the gauges the q direction is a ridge, not a valley.** That is true under the ordinary `nse-batch`
objective and predates the derivative term entirely. A parameter sitting on a concave direction does not converge;
gradient descent pushes it away from the stationary point toward whichever bound it started nearest.

The 25 x 25 grids say the same thing from the other side: scanning q at the trained n, the minimum sits at an
**edge of the search box at 62.5 % of gauges** (59.4 % upper, 3.1 % lower) and is interior at only 37.5 %, while
the loss changes by a median of just **5.4 %** across a factor of ten either way in q.

### The derivative term did not create this, it amplified it

The term multiplies the q gradient by **2.77x** and the q curvature by 2.89x. For comparison it multiplies the n
gradient by 2.68x. **It is not q-specific**: it amplifies both directions by essentially the same factor. On n,
which has a genuine interior minimum at most gauges, amplification sharpens convergence. On q, which is concave
at half of them, the same amplification accelerates divergence.

The bimodal outcome is the signature. If q were simply being pushed down by a monotone preference, the result
would pile up at one bound. Instead it splits, 64.2 % at the lower bound and 7.9 % at the upper, which is what
divergence from a ridge looks like: each reach falls off toward whichever side it started on. The skew toward
zero follows from where training starts and where it had already moved: q initialises near 0.49 and the
`nse-batch` model had already carried it down to a median of 0.084 with a long low tail, so most reaches were
below the crest before the amplified gradient arrived.

### Verdict on the two hypotheses

- **"Weak gradient path like n": SUPPORTED, with a correction.** It is not weak-but-monotone, it is concave. That
  distinction matters, because a weak monotone gradient is fixed by more optimizer steps whereas a concave
  direction is made worse by them.
- **"Caused by fixing p at 21": NOT SUPPORTED by this evidence, and not refuted either.** The concavity is present
  under `nse-batch` with p fixed, so it is not something the derivative term introduced, but every measurement
  here has p fixed, so there is no control that isolates p's role. §8 established that what a gauge identifies is
  the n/p ratio and that the gauge-optimal n scales with p, which makes p worth suspecting: with p frozen, the
  width degree of freedom has only q to move in.

**The experiment that settles it:** a landscape census on the learned-p arm with p active, comparing `H_qq` and the
share of concave gauges against the p-fixed model. If concavity in q largely disappears when p is free, fixing p
is the cause and the fix is to prescribe p from river size rather than to constrain q. If it persists, q is
intrinsically unidentifiable from daily discharge and should be fixed rather than learned.

**Practical consequence either way:** do not ship `nse-batch-deriv` with q learnable. The term's benefit is on n,
and n is where the curvature result applies; q should be pinned the way p is until this is resolved.

---

## 30. n and q live in different information channels, which decides what objective can identify what

### The measurement

What predicts each parameter's identifiability, Spearman rank correlation against the curvature at the trained
point (2,124 well-fit gauges):

| predictor | &#124;H_nn&#124; (roughness) | &#124;H_qq&#124; (width exponent) |
|---|---|---|
| flashiness (daily dQ/dt) | **+0.314** | **-0.045** |
| gauge reach slope | **-0.435** | -0.109 |
| depth at mean flow | | **+0.312** |
| width | | +0.242 |
| mean flow | +0.183 | +0.221 |
| channel length | +0.314 | +0.213 |
| peak ratio | +0.116 | +0.158 |

**They are governed by different things.** Roughness identifiability is a TIMING quantity: it tracks the
hydrograph's time derivative and is destroyed by steep, fast reaches. Width-exponent identifiability is a SCALE
quantity: it tracks depth, width and size, and flashiness tells you nothing about it (-0.045).

Two supporting numbers: `|H_qq|` is a median **6.2 %** of `|H_nn|` at the same gauge, so q is roughly sixteen
times less determined than n; and the two curvatures correlate only +0.353 across gauges, so a gauge that pins
down roughness is not thereby pinning down width.

### Why this decides the objective question

The derivative term of §26 to §28 works by amplifying the `dQ/dt` channel. That is exactly the channel roughness
lives in, which is why it deepened the n valley 3.04x. It is exactly the channel the width exponent does **not**
live in, which is why it amplified q's gradient by a nearly identical 2.77x while leaving q no better determined,
and so drove it off the ridge of §29 into the bounds.

**RMSE and MSE would not help either, and for a more basic reason.** `nse-batch` already is a mean squared error,
normalised per gauge by that gauge's observed variance. RMSE is its monotone square root: identical minimiser,
different gradient scaling. Unnormalised MSE differs only in how gauges are weighted against each other, trading
the per-gauge normalisation for a size weighting. None of the three opens a new information channel about channel
geometry; they reweight the same residuals. An objective can only identify a parameter if the data carries
information about it, and for q, daily discharge barely does.

**What would actually identify q** is an observation of the thing q parameterises. `W = p * d^q` is a width-depth
relation, and remotely sensed river widths (Landsat-derived, SWOT) constrain it directly, at the scale where the
measurement above says the signal already lives: large, deep rivers. Short of that, a flow-stratified objective
is the best proxy available from discharge alone, since q controls how celerity changes between low and high
flow; but note that the dynamic-range proxy we have (peak ratio) correlates with `|H_qq|` at only +0.158, so
expectations should be modest.

### On learning p and q together rather than n

This changes what is learned but not how much is identified, and it is still the better choice.

- **It does not add constraint.** §8 established that a gauge identifies the ratio n/p, not n and p separately, so
  "learn p with n fixed" and "learn n with p fixed" are the same model reparameterised. The stiff direction is the
  same direction either way.
- **It relocates the free parameter to something checkable.** Reach-scale Manning's n cannot be measured. Channel
  width can. Putting the learned degree of freedom in `p` and `q` makes the learned field falsifiable against
  external width data instead of being an unfalsifiable friction factor, which is worth more than the identifiability
  it does not gain.
- **The pair is genuinely coupled, and the standard fix applies.** `log W = log p + q log d` makes p and q the
  intercept and slope of a line, and intercept and slope are strongly correlated unless the predictor is centred.
  The reparameterisation `W = p_ref * (d / d_ref)^q`, with `d_ref` a per-reach reference depth (bankfull, or the
  median routed depth), decorrelates them by construction and makes `p_ref` directly comparable to a measured
  width. Without that centring, learning p and q together will reproduce the ridge behaviour of §29 in a rotated
  frame.
- **Measured coupling, for what it is worth:** the n-q Hessian correlation has median 0.336 with 21 % of gauges
  above 0.9, so n and q are not generally degenerate. p could not be measured here because no landscape census has
  p as an active axis, and no learned-p model is on disk; that is the gap to close.

### Recommended next experiments, in order

1. **Census with p active** on a p-learnable arm, to measure `H_pp`, `H_pq` and the p-q coupling directly. Nothing
   in this document constrains p, and every claim about it above is inference from §8 plus the algebra.
2. **Centred geometry**: implement `W = p_ref * (d/d_ref)^q` and repeat the census. Prediction to register in
   advance: the p-q Hessian correlation drops well below 0.9 at most gauges, and q stops running to its bounds
   under any objective.
3. **Fix n, learn p and q** with the centred parameterisation, against the current model. Expect equal skill
   (§20: the whole channel is worth about 0.09 NSE and roughness about a third of it), and judge it on whether the
   learned widths agree with observed river widths, not on NSE.

---

## 31. The KAN head is emitting one latent direction relabelled as two parameters

> **PARTLY SUPERSEDED BY §32 (2026-09-11).** The measurements below on the *trained* fields stand. Three of the
> inferences drawn from them do not, and should not be cited:
> (a) that the head is structurally unable to represent independent patterns for its outputs — every topology,
> the current one included, decorrelates two supervised targets to affine R^2 = 0.0000 (§32.3);
> (b) that the trunk delivers "essentially one direction" — measured directly, its effective rank is 4.36 of 21
> (§32.2);
> (c) that the 10 attributes "carry roughly one usable direction" — their effective rank is 6.11 of 10 (§32.1).
> The correct statement is that the collapse is an inductive bias under the routing gradient acting on a capable
> architecture, and that the trunk is narrower than its inputs. See §32.5 for the three questions re-answered.

The architecture is `Linear(F, H) -> KanLayer(H, H) x 2 -> Linear(H, P) -> Sigmoid` with `F = 10` attributes and
`H = 21`. All learned parameters come from **one shared trunk**, separated only by the final `Linear(H, P)`.
Measured on the learned fields over all 346,321 CONUS reaches (500-update `nse-batch` model):

| quantity | value |
|---|---|
| Spearman rho(n, q_spatial) | **0.9967** |
| Pearson rho(n, q_spatial) | 0.9640 |
| Pearson on the PRE-SIGMOID (logit) scale | **0.9935** |
| variance of q's pre-activation explained by a line in n's | **98.7 %** |
| fitted relation | `logit(q) = 3.98 * logit(n) + 2.75` |

The two learnable outputs are **the same spatial field read twice with different gains**. If the trunk's output
were full rank, two independent rows of the final Linear layer would produce two unrelated combinations of 21
hidden units. A near-perfect line on the pre-sigmoid scale means the trunk is delivering essentially one
direction. **(Superseded: the trunk's effective rank, measured directly at initialisation rather than inferred
from outputs, is 4.36 of 21 — §32.2. The near-perfect line is a property of the trained state, not of the
architecture's capacity.)**

Across models: rho(n, q) is 0.727 at epoch 1 (near initialisation), **0.999** after 60 updates, **0.997** after
500, and 0.869 for the derivative-loss model. Training makes the collapse worse, not better.

### This explains the boundary behaviour of §29 without any appeal to the loss

The fitted gain is **3.98**. Whatever spread the trunk produces in `logit(n)`, q's pre-activation spans about four
times as much, so q saturates its sigmoid at both ends while n stays comfortably interior. That is exactly what is
measured: n occupies 0.031 to 0.213 inside a declared range of 0.01 to 0.35 and never reaches a bound, while q
runs 0.0006 to 0.998 and piles up at both. **q hits its bounds because it is n's field amplified fourfold, not
because of anything the objective does to q.** The derivative term made it worse by widening n's spread, which q
then multiplies.

### What this means for the three questions

1. **"Is our KAN the problem?"** ~~Yes, demonstrably, and this is the first architectural defect the study has
   found that is not about identifiability. A model whose two geometry-and-friction outputs are 98.7 % the same
   latent direction cannot represent independent spatial patterns for them, whatever the data or the loss
   allows.~~ **WITHDRAWN — see §32.3.** The head *can* represent independent patterns; it does not under this
   gradient. Read §32.5.
2. **"A separate KAN for p and q?"** This is now the evidence-backed fix rather than an intuition. Separate trunks
   (or at minimum a wider trunk with a decorrelation penalty on the output heads) are what allow the fields to
   differ at all. Note the likely cause of the collapse: 10 attributes with a gradient-boosted ceiling of
   R^2 = 0.160 on the targets carry roughly one usable direction of information, and a 21-unit trunk trained on
   near-unanimous gradients has no reason to preserve more than that.
3. **"A loss that assists p and q outside n?"** Still worth having, for the reasons in §30, but it is now clearly
   the SECOND problem. ~~**No objective can separate two outputs that are reading the same latent direction.**~~
   **The premise is withdrawn (§32.2): at initialisation the outputs are not reading the same direction, because
   the trunk carries 4.36 of them.** The §30 case for such an objective stands on its own merits. Fix the
   architecture first, then re-ask whether the loss needs changing.

### Caveats

- This measures the learned output fields, not the trunk activations directly. The rank-1 reading is an inference
  from a 98.7 % linear relation on the pre-sigmoid scale, which is strong but indirect. Dumping the penultimate
  activations and taking their singular values would settle it outright and is cheap.
- `p_spatial` is constant at 21 and `x_storage` constant at 0.30 in this configuration, so this is a statement
  about the two genuinely learned outputs. Whether a three-output head collapses the same way is untested.
- The collapse is present but weaker at initialisation (0.727), so it is partly inherited from the initial
  weights and partly learned. Which dominates is untested.

### Suggested order of work

**Steps 1 and 2 were carried out on 2026-09-11; see §32. Step 1 refuted the rank-1 reading.**

1. Dump the trunk activations and compute their singular-value spectrum. One dominant singular value confirms
   rank-1 outright. **DONE — it does not: effective rank 4.36 of 21 (§32.2).**
2. Separate heads or trunks per parameter group, retrain, and re-measure rho(n, q). Registered prediction: rho
   falls well below 0.9 and q stops reaching its bounds under the ordinary objective, with no skill change
   (§20 bounds the whole channel at about 0.09 NSE).
3. Only then revisit the objective question of §30.

## 32. The head is capable; the collapse is an inductive bias, not a wall

§31 concluded "is our KAN the problem? Yes, demonstrably". That was too strong, and this section corrects it
with three measurements §31 called for and one it did not anticipate. The apparatus is
`src/bin/head_arch_screen.rs` plus `experiments/head_arch/analyze.py`, which compare head topologies in minutes
rather than the ~2 h a CONUS training arm costs, because the head is a pure per-reach function of the attributes:
no routing, no observations, no optimizer schedule. Seven topologies, `n` + `p_spatial` + `q_spatial` emitted by
each, evaluated over the real CONUS attribute distribution.

### 32.1 The inputs are not one-dimensional

§31 guessed that "10 attributes with a gradient-boosted ceiling of R^2 = 0.160 carry roughly one usable
direction". They do not. PCA of the ten z-scored production attributes over all 2,939,404 MERIT reaches
(`experiments/head_arch/attribute_rank.py`, no Rust needed):

| quantity | value |
|---|---|
| effective rank (participation ratio) | **6.11 of 10** |
| directions holding 90 % of variance | 6 |
| directions holding 99 % of variance | 10 |
| PC1 share | 0.274 |

PC1 is a wetness-and-vegetation axis (`meanP` −0.540, `SoilGrids1km_clay` −0.512, `NDVI` −0.503; the strongest
pair is `meanP`–`NDVI` at r = +0.795). The two attributes §30 identified as carrying the identifiable channels
load essentially zero on it: `log10_uparea` at −0.000 and `meanslope` at −0.040. **The scale channel and the
timing channel are available to the head as separate directions.** So a collapse to one direction is not
inherited from the data.

The GBM ceiling of R^2 = 0.160 is a statement about how much of the *target* the attributes explain, not about
how many directions they span. Conflating the two was the error.

### 32.2 The trunk is rank 4.4, not rank 1

§31's stated caveat was that it inferred the trunk's rank from the output fields rather than measuring it.
`KanHead::trunk_activations` now exposes the penultimate activations directly. Singular spectrum of the centred
`[N, H]` activation matrix, at initialisation, over a 4,027-reach stride sample of CONUS:

| topology | H | effective rank | PC1 share | dims for 90 % |
|---|---|---|---|---|
| shared, Linear read-out (current) | 21 | **4.36** | 0.412 | 6 |
| separate trunk per group | 21 | 4.36 | 0.412 | 6 |
| KAN read-out | 21 | 4.36 | 0.412 | 6 |
| KAN embedding + KAN read-out | 21 | **3.03** | 0.524 | 4 |
| **depth 2 -> 4** | 21 | **3.41** | 0.450 | 5 |
| **H 21 -> 64** | 64 | **6.41** | 0.249 | 10 |

Two facts follow immediately, and neither was expected.

**The H = 21 trunk throws away input directions.** The attributes carry 6.11; the trunk delivers 4.36. Three
outputs reading a rank-4.4 latent through three rows of a `Linear(21, 3)` have to overlap. At H = 64 the trunk
carries 6.41, essentially everything the inputs have.

**Depth is the wrong knob, and it is actively harmful.** Two extra `KanLayer` blocks *reduce* effective rank from
4.36 to 3.41. Each block applies per-edge splines on a `[-1, 1]` grid and sums over inputs; stacking them
contracts the representation rather than enriching it. The KAN embedding arm is worse still at 3.03, because a
`KanLayer(F, H)` on z-scored attributes puts much of its input outside the spline grid where only the
`scale_base · SiLU` path survives.

### 32.3 Every topology passes a supervised capacity control, including the current one

This is the measurement that overturns §31's verdict. Two targets were built from the real attributes, one from
`meanslope` and one from `log10_uparea` Gram-Schmidt orthogonalised against the first, then squashed into the
head's own output range: each is exactly recoverable from the inputs and their mutual correlation is
+1.3e-3 by construction. Each topology was fitted to both, supervised, with Adam for 400 steps.

| topology | loss, start -> end | rank corr between outputs | affine R^2 |
|---|---|---|---|
| shared, Linear read-out (current) | 0.0767 -> 0.000016 | +0.028 | **0.0000** |
| separate trunk for {n} vs {p, q} | 0.0746 -> 0.000011 | +0.028 | 0.0000 |
| one trunk per parameter | 0.0752 -> 0.000012 | +0.028 | 0.0000 |
| KAN read-out | 0.0746 -> 0.000004 | +0.028 | 0.0000 |
| KAN embedding + KAN read-out | 0.0769 -> 0.000001 | +0.027 | 0.0000 |
| depth 2 -> 4 | 0.0767 -> 0.000002 | +0.028 | 0.0000 |
| H 21 -> 64 | 0.0768 -> 0.000005 | +0.028 | 0.0000 |

**The current head can emit two independent spatial fields.** It is not structurally incapable, and a separate
KAN adds no capability it lacks. §31's claim that "a model whose two geometry-and-friction outputs are 98.7 %
the same latent direction cannot represent independent spatial patterns for them" is false as stated: it cannot
*under the routing gradient*, which is a different and weaker claim.

The control does not discriminate between topologies, and that is the finding. It converts the question from
"what restores a missing capability" into "what removes a bias", which is a question only a training arm can
answer.

### 32.4 What a KAN read-out does and does not fix

Worth stating precisely, because the intuition that a nonlinear read-out decouples the outputs is half right.

With `Linear(H, P)` the outputs are `logit(param_j) = w_j · h + b_j`. Two of them are *exactly* affinely related
if and only if `h` is effectively rank 1 across reaches. A Linear read-out therefore does not force affinity on
its own; it forces it in combination with a rank-1 latent, which is the §31 regime and not a property of the
layer. `tests/kan_head_groups.rs` pins this at `hidden_size = 1`, where the relative residual of one logit column
regressed on the other is at the f32 floor.

With `KanLayer(H, P)` each output carries its own spline coefficients on every edge
(`y[o] = sum_i sb[i,o]·SiLU(h[i]) + sp[i,o]·spline_{i,o}(h[i])`), so the affinity breaks. **It does not break the
functional dependence.** Two different nonlinear functions of one scalar latent remain in lockstep: the same test
asserts the rank correlation stays above 0.999 in the rank-1 case. A KAN read-out changes the shape of the
coupling, not the fact of it.

Decoupling the outputs requires a trunk that carries more than one direction. That is why `H 21 -> 64` is now an
arm, and why it is the one to watch.

### 32.5 What this licenses, and what still needs a training arm

Answering the three questions of §31 again, corrected:

1. **"Is our KAN the problem?"** Partly, and not in the way §31 said. It is not incapable. It is *narrower than
   its inputs* (rank 4.36 against 6.11) and it hands training a coupling to start from. "Architectural defect"
   should read "architectural bias".
2. **"A separate KAN for p and q?"** It removes the inherited coupling but adds no capability. Whether removing
   the bias changes where the routing gradient lands is exactly the open question, and it needs the CONUS arms.
3. **"A loss that assists p and q outside n?"** §31 said no objective can separate two outputs reading the same
   latent direction. With the trunk at rank 4.4 rather than 1, they are *not* reading the same direction at
   initialisation, so that argument does not hold as stated. The §30 reasons for wanting such an objective
   stand on their own.
4. **"Is the KAN deep enough?"** No, and depth is the wrong question. Depth reduces the rank the trunk carries
   (4.36 -> 3.41) and raises the coupling it inherits. Width does the opposite.

### 32.6 The seed sweep: splitting and widening both decouple, depth does not

A single initialisation cannot rank these topologies. For two random read-out rows over a latent of effective
rank r, chance alone gives |rho| of order 1/sqrt(r). Eight seeds per arm, Spearman |rho(n, q)| on the
4,027-reach sample, with each arm's own chance line from its own measured rank:

| topology | median \|rho\| | IQR | chance line | median affine R^2 | share of seeds \|rho\| > 0.5 |
|---|---|---|---|---|---|
| shared, Linear read-out (current) | 0.466 | [0.328, 0.505] | **0.479** | 0.2065 | 0.375 |
| depth 2 -> 4 | 0.488 | [0.160, 0.602] | **0.541** | 0.2476 | 0.500 |
| KAN embedding + KAN read-out | 0.297 | [0.125, 0.750] | 0.575 | 0.0525 | 0.375 |
| KAN read-out | 0.217 | [0.154, 0.406] | 0.479 | 0.0333 | 0.125 |
| H 21 -> 64 | **0.105** | [0.077, 0.328] | 0.395 | 0.0413 | 0.125 |
| separate trunk for {n} vs {p, q} | **0.091** | [0.068, 0.322] | 0.479 | 0.0334 | 0.250 |
| one trunk per parameter | **0.086** | [0.037, 0.219] | 0.479 | 0.0139 | 0.250 |

**The current head sits exactly on its chance line: 0.466 against 0.479.** Its output coupling at
initialisation is fully accounted for by two arbitrary rows reading a rank-4.4 latent. There is nothing
pathological in the initialisation, and equally nothing working in its favour.

**Depth is at chance too, with the worst affine R^2 of any arm (0.2476) and the highest share of seeds above
0.5.** Combined with its lower trunk rank this is the clearest negative result of the screen: adding
`KanLayer` blocks does not help and plausibly hurts.

**Splitting trunks and widening the trunk both land far below chance**, and they are not distinguishable from
each other at eight seeds: medians 0.086 to 0.105 with heavily overlapping IQRs. They get there by different
mechanisms, which is why both are training arms. Splitting gives each output group its own latent, so the
shared-rank argument stops applying. Widening lowers the chance line itself, from 0.479 to 0.395, by carrying
more of the input's 6.11 directions, and then lands well under it.

**The KAN read-out roughly halves the coupling (0.217 against a 0.479 chance line) without touching the
trunk**, which is consistent with §32.4: it breaks the affine tie but leaves the outputs reading the same
latent. The KAN embedding arm is the least reliable of all, with an IQR reaching 0.750, and it has the lowest
trunk rank; it is not promoted.

### Registered predictions for the CONUS arms

Four arms, differing only in `kan_head`, all learning `n` + `p_spatial` + `q_spatial`, all derived from the
500-update `nse-batch` baseline `2026-09-10T21-21-48Z-conus-train-and-test`
(`config/experiments/head_{shared_linear,split_trunk,wider,kan_readout}.yaml`):

- **`head_shared_linear`** reproduces the §31 collapse with three outputs: Spearman rho(n, q) above 0.95 after
  500 updates, and `q_spatial` piling up at both bounds.
- **`head_wider` and `head_split_trunk` both fall well below the control.** §32.6 cannot separate them at
  initialisation, and that is precisely what the training arms are for: they decouple by different mechanisms
  (more directions to read, versus each group reading its own latent) and the routing gradient may reward one
  and not the other.
- **`head_kan_readout`** barely moves rho(n, q) at all. This is the discriminating prediction: if it *does* move,
  the read-out was the binding constraint after all; if it does not, the trunk's capacity is.
- **No arm changes median skill by more than about 0.01 NSE**, since §20 bounds the whole channel at roughly
  0.09 NSE and §24 showed the NSE optimum costs no KGE.

### Caveats

- Every number in §32.1 to §32.4 is at **initialisation**. §31 measured rho(n, q) rising from 0.727 near
  initialisation to 0.999 at 60 updates, so the trained outcome is not implied by the prior. The arms are the
  test.
- The capacity control's targets are near-linear functions of the inputs, which is a far easier ask than the
  routing problem. It bounds capability from below; it says nothing about what a weak gradient will find.
- The trunk spectra of §32.2 are measured at one seed. The eight-seed sweep of §32.6 corroborates the ordering
  indirectly but does not re-measure rank per seed.
- Eight seeds is enough to separate the control from the split and wide arms (0.466 against 0.086 to 0.105) and
  not enough to separate those two from each other.
- The trunk spectrum is measured on a 4,027-reach stride sample, not all 346,321 reaches.


## 33. Making `p_spatial` learnable is worth about +0.008 NSE (one seed, one arm)

The first of the four head-topology arms finished before the chain was stopped, and it carries a result that
is not about head topology at all.

`head_shared_linear` is the **matched control**: the current architecture, one shared trunk, Linear read-out,
everything taken from the 500-update `nse-batch` baseline `2026-09-10T21-21-48Z-conus-train-and-test` except
that it learns `n` + `p_spatial` + `q_spatial` where the baseline learned `n` + `q_spatial` and pinned
`p_spatial` at 21.

| | NSE | KGE |
|---|---|---|
| baseline, `p` fixed at 21 | 0.7376 | 0.7600 |
| **`head_shared_linear`, `p` learnable** | **0.7458** | **0.7619** |
| difference | **+0.0082** | +0.0019 |

Same 2,365 gauges, same eval window, same optimizer budget, same seed, same everything else — the arm config is
that run's own `config.yaml` with only the `kan_head` block changed.

**The predicted mechanism is REFUTED — see §33.1.** §32.5 argued that with `p` constant the downstream width
exponent is capped at `q·f`, and that `p` is the only parameter able to supply the rest through how it scales
with river size. The skill gain is real, but it did not come from that: `p` scales *negatively* with discharge
and the downstream geometry got worse, not better.

### What this is not, yet

- **One seed.** NdArray is deterministic, so re-running reproduces the number exactly and tells us nothing.
  There is no spread estimate for this configuration. +0.008 is seven times smaller than the +0.059 the
  optimizer-budget fix delivered (§27), and that one moved 70.7 % of gauges.
- ~~**Not run on a plain-master binary.**~~ **RESOLVED 2026-09-12.** A binary built from `3412a78` (plain
  master) and one built from this branch — carrying both the `kan_head` restructure and the stage-roughness
  code — produce **identical per-micro-batch losses** on this exact config, to all six printed decimals:

  ```
    micro 1/4  loss=0.141474  n=4980  median_n=0.13343      master 3412a78
    micro 1/4  loss=0.141474  n=4980  median_n=0.13343      this branch
    micro 2/4  loss=0.153518  n=5312  median_n=0.13331      both
    micro 3/4  loss=0.092350  n=5312  median_n=0.13324      both
  ```

  The full comparison runs **100 mini-batches over 50 epochs and the two logs are identical line for
  line, all 500 lines**, so the equivalence is not just at initialisation: it survives fifty epochs of
  accumulated drift, where any real numerical difference would have amplified. Both sets of changes are
  inert at their defaults, and the +0.008 is attributable to `p_spatial` becoming learnable rather than
  to any code change. Reproduce with
  `experiments/head_arch/` — the check is a `run --workflow train --max-mini-batches 2` on each binary with
  the same config (note `--max-mini-batches` caps mini-batches PER EPOCH, not in total).
- **The mechanism was measured and is refuted.** See §33.1.

### Also measured: the celerity convention is approximate

Building the stage-roughness gates surfaced a property of the existing solver worth recording, since it is
pre-existing and easy to rediscover as a bug. `beta = 5/3 − (4/3)·A·√(1+z²)/(T·P)` is derived for a trapezoid
of FIXED shape being filled, i.e. `dA/dd = T` and `dP/dd = 2√(1+z²)`. This geometry reshapes as it fills:
`bw = tw·(1−q)` and `z = (p·q/2)·d^(q−1)` both move with depth, so the true `dA/dd` is `T·(2−q)(q+1)/2`.

Measured gap between `v·beta` and the true `dQ/dA`:

| q | 0.084 | 0.30 | 0.65 | 1.00 |
|---|---|---|---|---|
| relative error | 0.0–0.8 % | 0.8–1.0 % | **2.1 %** | 0.0 % |

It vanishes at `q = 0` and `q = 1` (where `(2−q)(q+1)/2 = 1`) and peaks in between. At the trained
`q ≈ 0.084` it is under 1 %, so it is not a live problem for current results, but it scales with `q` and would
matter if `q` were moved toward the Leopold & Maddock band. Inherited from DDR; changing it would move
invariant 1, so it is documented rather than fixed. Test:
`tests/stage_roughness.rs::documents_the_preexisting_beta_approximation`.

### 33.1 Skill went up and the channel geometry went down

`experiments/head_arch/downstream_geometry.py` fits the downstream exponents the way
`channel_geometry.md` defines them, at a common baseflow specific discharge across all 346,321 reaches. The
model reaches Leopold & Maddock's `b ≈ 0.50` through

```
  b = beta + q · f          beta = dlog(p) / dlog(Q)
```

so with `p` pinned, `beta = 0` and `b` cannot exceed `q·f`. That cap was the whole argument for freeing `p`.

| model | median q | median p | beta | q·f | **b** | f |
|---|---|---|---|---|---|---|
| `p` fixed at 21 (`2026-09-10T21-21-48Z`) | 0.0843 | 21.00 | 0.000 | 0.046 | **0.099** | 0.550 |
| `p` learnable (`2026-09-11T23-24-04Z`) | 0.2951 | 5.85 | **−0.144** | 0.163 | **0.004** | 0.551 |
| Leopold & Maddock | — | — | — | — | **0.50** | 0.40 |

`q` more than tripled, which lifted `q·f` from 0.046 to 0.163 exactly as intended. But **`p` shrinks as rivers
grow** (`beta = −0.144`), which cancels that and more. The net downstream width exponent fell from 0.099 to
**0.004**: channel width is now essentially constant from headwater to main stem, where reality grows it as
`Q^0.5`.

**So freeing `p` bought +0.008 NSE and made the channel geometry worse.** The two moved in opposite directions.

### 33.2 What the gain actually came from

Measuring the trained trunk for this model (`--checkpoint` mode, same method as §32.2):

| | trunk eff. rank | PC1 share | rho(n, q) | affine R² |
|---|---|---|---|---|
| two parameters (`n`, `q`) | **1.38** | 0.846 | 0.997 | **0.9924** |
| three parameters (`n`, `p`, `q`) | **1.64** | 0.768 | 0.681 | **0.4580** |

Adding a third output **partly breaks the rank-1 collapse**. The trunk carries a little more (1.38 → 1.64) and,
far more visibly, the outputs stop being the same field: the share of `q`'s pre-activation explained by `n`'s
falls from 99.2 % to 45.8 %. The residual is not noise either — it correlates +0.601 with `meanslope`.

The fields also separate onto different attributes for the first time. `q` is now slope-driven (rho = +0.665
with `meanslope`, against +0.325 with elevation), while `n` stays soil- and size-driven (+0.624 sand, −0.494
`log10_uparea`). In the two-parameter model every field had the same correlation profile to within a few
hundredths (§32's table).

So the honest account of the +0.008 is: **not the geometry mechanism, but the head escaping some of its own
collapse**, which is a different and more interesting reason than the one predicted.

### 33.3 Why this matters more than the skill number

This is selective equifinality caught in the act. Given freedom, the model used `p` for whatever helped the
daily hydrograph and spent none of it on making channels physically sensible, because the objective cannot see
channel width at all (§30: the width exponent lives in the scale channel, where flashiness is null at −0.045).
A model that fits better and describes the river worse is exactly what the paper claims a distributed
differentiable model does when its parameters are unidentifiable.

It also means **`b` should be reported alongside NSE for every future arm**. Skill alone would have recorded
this run as a straightforward improvement.

## 34. The matched control: the +0.000 displacement was the budget, not the loss

PR #42 reported that the derivative-loss model's 400-gauge stratified census gave a **median signed roughness
displacement of +0.000**, gauges sitting exactly at their own optimum, and said plainly that without a matched
control the number could not be attributed to the loss rather than to the optimizer budget. Both models had 500
updates; only one had the time-derivative term. The control has now finished: the same 400 gauges stratified by
basin size, the same objective, window and Newton search, run against the `nse-batch` model.

Well-fit gauges (`nse0 > 0.3`, finite optimum), 400 per arm:

| arm | n | median a* | median \|a*\| | \|a*\| < 0.10 | \|a*\| < 0.25 |
|---|---|---|---|---|---|
| `nse-batch` (control) | 358 | **−0.001** | 0.383 | 17.0 % | 33.5 % |
| `nse-deriv` | 359 | **+0.000** | 0.376 | **24.8 %** | 37.0 % |

**The headline number is not the loss.** The control lands at −0.001, indistinguishable from the derivative
model's +0.000. Sitting at the median optimum is a property of running 500 optimizer updates, which is §27's
result, not of the derivative term. Any reading of PR #42 that credits the loss for it should be corrected.

**But the derivative term does help, modestly and significantly.** Paired on the 357 gauges both arms resolved:

| | median a* | median \|a*\| |
|---|---|---|
| `nse-batch` | −0.001 | 0.385 |
| `nse-deriv` | +0.000 | **0.366** |

- median change in \|a*\|: **−0.058**
- the derivative model is closer to its own optimum at **57.7 %** of gauges
- Wilcoxon on the paired \|a*\|: **p = 0.0038**

So the term tightens identifiability rather than relocating the optimum, which is exactly what §28's curvature
probe predicted: it sharpens the valley (3.04x deeper, 92.8 % of gauges) without moving where the valley sits
(sign agreement 88.7 %). The census now shows that sharpening translating into gauges actually sitting nearer
their optima, with the share within 0.10 log units rising from 17.0 % to 24.8 %.

### What this settles and what it does not

Settled: the derivative term is a real if modest improvement in how well a gauge determines its own roughness,
it is statistically significant on a paired test, and it is not responsible for the +0.000 median that PR #42
led with.

Not settled: whether that improvement is worth its cost. §20 bounds the entire roughness channel at about 0.09
NSE, the derivative model scored a null on skill (+0.0007 NSE, +0.0012 KGE, journal entry for
`2026-09-11T06-38-49Z`), and §29 showed it drove `q` to its bounds at 64 % of reaches. A better-identified
roughness that buys no skill and wrecks the width exponent is not obviously a good trade. The recommendation in
PR #42 stands: do not make `nse-batch-deriv` the default.

## 35. Stage-dependent roughness on CONUS: the single-basin screen did not generalise

`n(d) = n_0·(d/d_ref)^(−gamma)` (design doc `2026-09-12-stage-dependent-roughness-design.md`, Phase 1:
`gamma` is a fixed global value, not a KAN output). Juniata screened it at 20 s per arm and gave a smooth
single-peaked response, +0.091 NSE at `gamma = 0.35` against a control that reproduced the documented
0.7903 / 0.8810 exactly. The matched CONUS pair, identical in every other respect and both on the same binary:

| | `gamma = 0` | `gamma = 0.35` | change |
|---|---|---|---|
| **median NSE** | **0.7458** | **0.7362** | **−0.0096** |
| median KGE | 0.7619 | 0.7588 | −0.0031 |
| downstream `b` (L&M 0.50) | 0.004 | **0.098** | **+0.094** |
| downstream `f` (L&M 0.40) | 0.551 | **0.484** | −0.067, toward L&M |
| median `q` | 0.295 | 0.408 | +0.113 |
| `beta` = dlog p / dlog Q | −0.144 | −0.082 | +0.062, less wrong |
| trunk effective rank | 1.64 | **1.36** | −0.28 |
| affine R² (`n`, `q`) | 0.458 | **0.976** | +0.518 |
| `n(d)` breathing, per reach (median) | 1.000x | **1.910x** | — |

**Juniata was misleading, by a lot and with the wrong sign.** +0.091 on one gauge became −0.0096 on 2,365.
The Juniata sample is the documented fast end-to-end check and it is genuinely useful for mechanics, but it
does not predict CONUS for this parameter. `gamma = 0.35` was chosen from a smooth interior optimum on a
single basin, which looked like exactly the kind of evidence that should generalise, and did not.

This was foreseeable and was partly foreseen. §2 of the design doc and the commit that recorded the Juniata
sweep both flagged that the optimum sat at 0.35 against a literature-motivated 0.183, and that at the trained
`q ≈ 0.084` the model already had an at-a-station velocity exponent of 0.381 against an observed 0.34 — so
raising `gamma` was moving the static exponents *away* from observation while improving single-basin skill.
The CONUS result is what that warning looks like when it comes true.

The roughness really does breathe: over water year 1996 the typical live CONUS reach swings its Manning's
`n` by **1.91x** between its driest and wettest day (p90 3.26x), against a network-median swing of only 1.29x
— the spatial average is small because reaches peak on different days, so the per-reach number is the one to
quote. That measurement requires excluding **188,646 reaches (54.5 %)** that carry no Q' prediction or sit at
physically meaningless flows; they plot as flat lines and bias the statistic toward 1. See
`.claude/skills/ddrs-eval-plots/references/stage_roughness.md`.

### 35.1 Skill and physical plausibility are anti-correlated, in both directions

Put §33.1 and this section side by side. Both are matched single-variable changes off the same baseline:

| change | Δ median NSE | Δ downstream `b` |
|---|---|---|
| free `p_spatial` (§33.1) | **+0.0082** | **−0.095** |
| add `gamma = 0.35` (§35) | **−0.0096** | **+0.094** |

The two are near-perfect mirror images: roughly **0.01 NSE traded against 0.095 in the width exponent**, in
whichever direction the change happens to push. One change bought skill by making the channel less physical;
the other bought physics by giving up skill; neither bought both.

That is a stronger statement than §30's "the objective cannot see channel width". It is not indifference. The
daily-discharge objective **actively prefers** the physically wrong channel, and the preference is measurable
and roughly symmetric. Any arm tuned on skill alone will drift away from defensible geometry, which is exactly
why §33.3 now requires `b` next to every NSE.

### 35.2 `gamma` deepened the output collapse

Unexpected, and it cuts against the §32 story. Adding `gamma` moved the trained trunk from rank 1.64 back down
to 1.36, and the share of `q`'s pre-activation explained by `n`'s from 45.8 % up to 97.6 % — almost all the way
back to the two-parameter model's 99.2 %.

A plausible reading, untested: stage-dependent roughness gives the *router* a flow-dependent travel time for
free, so the network no longer needs spread in its parameter fields to produce the same routing behaviour, and
collapses further. If that is right, it is a general caution: adding physical capacity to the solver can
reduce what the learned parameters have to carry, and therefore make the parameters less identifiable rather
than more. Worth testing directly before it is believed.

### 35.3 Verdict

**Do not adopt `gamma` at 0.35.** It costs about 0.01 NSE, and while it improves the downstream exponents
substantially it also deepens the head collapse.

What is not settled is whether some *smaller* `gamma` sits on the good side of the trade — `b` improved by
0.094 for a skill cost of 0.0096, and a value near the literature-motivated 0.183 was never run on CONUS. The
right next experiment is a CONUS sweep of `gamma ∈ {0.1, 0.183}`, scored on **both** NSE and `b`, not a
re-tune on Juniata. Two arms, about 4.5 h.

The implementation stands and is verified: `gamma = 0` is bit-identical to the historical solver
(`tests/stage_roughness.rs`), the backward is gradient-exact at `gamma = 0.35` across all five parents
(`tests/sp8_gradcheck.rs`), and the control arm here reproduced `head_shared_linear` to four decimals on a
different binary.

## 36. Learned stage-dependent roughness: code audit, prior art, and the CONUS read-out

Reads the learned-`gamma` arm (`config/experiments/sr_gamma_learned.yaml`) against its matched control
`head_shared_linear` (`2026-09-12T03-53-34Z`, 0.7458 / 0.7619) and the global-constant arm `gamma = 0.35`
(`2026-09-12T06-06-19Z`, 0.7362 / 0.7588, §35). Before the read-out, the implementation was audited end to end,
which is where the section starts, because the audit changed what got run.

### 36.1 The equations, checked by hand

The model family is `n(d) = n_0 · (d/d_ref)^(−gamma)` substituted into Manning for a section whose top width
follows `w = p·d^q`. Three consequences, all verified against `src/geometry.rs`, `src/routing/mmc_op.rs` S2–S17
and `src/experiment/adjoint/hydraulics.rs`:

1. **Depth inversion stays closed form.** `Q = (1/n_0)·d^gamma·R^(2/3)·√S·A` with the power-law section gives
   `d = (Q·n_0·(q+1)·d_ref^gamma / (p·√S))^(3/(5+3q+3·gamma))`. The code applies exactly this: the exponent
   denominator gains `3·gamma` and the numerator gains `d_ref^gamma` (which is `1.0` for the learned field, since
   config load requires no `stage_roughness` block there and `stage_roughness_params()` then returns
   `d_ref = 1`).
2. **Velocity carries the factor too.** `v = (1/n_0)·(d/d_ref)^gamma·R^(2/3)·√S`. Applying the stage law to
   the depth inversion but not to the velocity would move the depth exponent while leaving the velocity exponent
   at Manning's `2f/3`; `tests/stage_roughness.rs::exponents_match_theory_and_gamma_moves_velocity` pins the
   realised exponents `f = 3/(5+3q+3·gamma)`, `b = q·f`, `m = 1 − b − f` at `gamma ∈ {0, 0.183, 0.4}`.
3. **Celerity gains one term.** With `dA/dd = T` for any section, `c = dQ/dA = v·[beta_trap + gamma·A/(T·d)]`,
   where `beta_trap = 5/3 − (4/3)·A·√(1+z²)/(T·P)` is the pre-existing fixed-shape trapezoid convention. The
   increment is exact under that convention
   (`tests/stage_roughness.rs::stage_roughness_celerity_increment_is_exact`, rel. error < 1e-4); the convention
   itself is approximate by up to 2.25 % because the section reshapes as it fills, which is inherited from DDR,
   measured, and deliberately unfixed (`documents_the_preexisting_beta_approximation`).

`gamma = 0` takes the historical code path bit for bit (`compare_ddr_sandbox` still reports ABSOLUTE MATCH at
1.5e-5 m³/s; `gamma_zero_is_bit_identical`).

### 36.2 The gradient, checked by hand and by finite differences

A learned `gamma` enters the forward in three places and the hand-written backward (invariant 4) has one term
for each, all read from the same saved primitives:

| forward site | term | backward |
|---|---|---|
| S5 exponent `3/(5+3q+3·gamma)` | `∂exp/∂gamma = −9/(5+3q+3·gamma)²`, the same expression as `∂exp/∂q` | B5 |
| S15 velocity `(d/d_ref)^gamma` | `∂v/∂gamma = v·ln(d/d_ref)`; also a new explicit `∂v/∂d = gamma·v/d` | B15 |
| S17 celerity `+gamma·A/(T·d)` | `∂c/∂gamma = v·A/(T·d)`; also `∂/∂A, ∂/∂T, ∂/∂d` of the new term | B17 |

The `d_ref^gamma` numerator factor contributes nothing because `d_ref = 1` whenever `gamma` is learned. The two
new depth paths (B15, B17) join the depth accumulator before the `depth_lb` clamp mask, which is right because
the clamped depth is a constant there. `tests/sp8_gradcheck.rs` compares all six parents against central finite
differences with `gamma = 0.35` both as a global constant and as a learned per-reach tensor (19 tests, all pass).
`gamma` is a real autograd parent through `TimestepGammaOp`, the six-parent sibling of `TimestepOp`; a tensor
that is not a parent never receives a gradient, which is the trap the sibling exists to avoid.

### 36.3 What the audit found: the eval path never saw the learned gamma

`src/training/forward.rs` has three hand-written readers of the head's output map — `forward` (training),
`forward_eval_core` (eval, behind `forward_eval`), and `probe_forward` — and only the first had been taught about
`gamma`. The other two built `SpatialParameters { gamma: None }`, so the solver fell back to the config scalar,
which is 0 for a learned-gamma run. **The first learned-gamma CONUS arm (`2026-09-12T13-38-27Z`) therefore
trained the right model and was about to score a different one**, at `gamma = 0`, with no error and plausible
numbers. It was stopped at eval chunk 80/366.

The fix threads `gamma` through all three readers, makes the landscape objective refuse a learned-gamma arm
instead of routing it at 0, exports the per-reach `gamma` field to `plot/kan_parameters.nc`, and adds
`tests/gamma_eval_parity.rs`: one head routed through `forward`, `forward_eval` and `probe_forward` must give the
same hydrograph. Before the fix it diverged by 2.1e-2 relative; after, they agree to f32 round-off, and a table
over every optional head output (`p_spatial`, `x_storage`, `gamma`, the three leakance fields) now runs on every
`cargo test`. The three readers stay separate on purpose (WET); the table is the price. Trap T14 in
`ddrs-dev/references/traps.md`.

The relaunch (`2026-09-12T16-30-14Z`, binary `0844cf8`) reproduces the killed arm's training bit for bit — its
third accumulated batch loss, 0.294693, is identical — so nothing about the training result is in question;
only the eval numbers below are new.

The same pass found five test crates (`celerity_beta`, `cunge_x`, `negative_discharge_counter`,
`positivity_clamp`, `cuda_backward_parity`) that no longer compiled against the 13-argument
`timestep_forward`. "All 7 gate suites green" in the previous handoff was true of the suites it named and false
of `cargo test`.

### 36.4 Prior art and physical realism, for the reviewer

Roughness that falls with stage is standard river hydraulics, not a modelling convenience:

- **Semi-logarithmic resistance laws.** Keulegan-type relations give `1/√f ∝ log(R/k_s)`, so Manning's `n`
  (which absorbs `R^(1/6)/√f`) falls as relative submergence `R/k_s` rises. Limerinos (1970) fitted exactly this
  to natural channels: `n = 0.0926·R^(1/6) / (1.16 + 2.0·log10(R/d_84))`.
- **Power laws in depth.** Jarrett (1984), for high-gradient streams, `n = 0.39·S^0.38·R^(−0.16)`: a stage
  exponent of −0.16 on hydraulic radius, which is the same form as this model with `gamma ≈ 0.16` and sits next
  to the 0.183 the at-a-station exponents imply (§2 of the design doc). Ferguson's (2007) variable-power equation
  reproduces the same decline across the shallow-to-deep transition; Bjerklie et al. (2005) compare these forms
  on natural rivers and find the depth dependence necessary.
- **Hydraulic geometry.** Leopold & Maddock's at-a-station velocity exponent (`m ≈ 0.34`) exceeds the `2f/3`
  that constant-`n` Manning allows; the gap is precisely a stage-dependent roughness, which is the argument the
  design doc made from the exponents alone.

So the functional form has a literature and a physical mechanism (drowning of the bed material). What a reviewer
will push on, and what the numbers below have to answer:

1. **Monotone only in-bank.** Every relation above is for in-channel flow. Overbank flow raises composite
   roughness sharply (which is why operational routing such as the National Water Model carries a separate,
   larger compound-channel `n`). A single decreasing power law is wrong above bankfull; on daily CONUS routing
   the exposure is the largest floods at the largest rivers, exactly where §35's improvement in the width
   exponent came from. This is a modelling limitation to state, not hide.
2. **`n_0` is not Manning's `n`.** It is roughness at `d_ref = 1 m`. Published `n` maps from `gamma = 0` runs
   are not comparable; the parameter dump names the variable `gamma` and this document names the change.
3. **Identifiability.** A gauge observes a network sum (fact 5 in the `ddrs-dev` skill), and a per-reach
   `gamma` adds one more field with only that supervision. The registered prediction in the config banner —
   `rho(n, gamma) > 0.9`, `gamma` as another relabelled copy of `n` — is the thing §36.5 measures, and a
   "yes" would argue for `gamma` as a physical constant, not a learned field.
4. **The celerity convention** is approximate at the 1–2 % level (36.1). `K = L/c` and `c ∝ 1/n`, so the
   learned roughness absorbs the smooth part; it does not affect the comparison between arms, which share it.
5. **It deviates from DDR**, so the KAN-head parity fixtures do not cover it; the manifest's config snapshot
   records the deviation.

### 36.5 Read-out: the registered prediction failed, and gamma bought nothing

Run `2026-09-12T16-30-14Z-train-and-test`, binary `0844cf8`, 50 epochs / 500 updates, 2,365 gauges, CPU.

| | control `gamma = 0` | constant `gamma = 0.35` | **learned `gamma`** |
|---|---|---|---|
| median NSE | 0.7458 | 0.7362 | **0.7420** (−0.0038) |
| median KGE | 0.7619 | 0.7588 | **0.7624** (+0.0005) |
| downstream `b` (L&M 0.50) | 0.004 | 0.098 | **−0.017** |
| downstream `f` (L&M 0.40) | 0.551 | 0.484 | 0.550 |
| `beta` = dlog p / dlog Q | −0.144 | −0.082 | −0.157 |
| median `q` / `p` | 0.295 / 5.85 | 0.408 / 9.99 | 0.261 / 8.22 |
| trunk effective rank (dims for 90 %) | 1.64 | 1.36 | **1.96** (3) |
| affine R² (`n`, `q`) | 0.458 | 0.976 | 0.704 |
| rho(`n`, `gamma`) | — | — | **+0.30** |

**The learned field.** `gamma` is median 0.221, p10 0.159, p90 0.312, min 0.064, max 0.427, with 0.0 % of
reaches at either bound of the [0, 0.5] box; the sigmoid initialises at 0.25, so the head moved it down and
spread it. It is ordered by river size: median 0.236 below 100 km², 0.217 at 100–1,000, 0.179 at
1,000–10,000, 0.149 above 10,000 km². That is the direction the resistance literature gives (relative
submergence grows with size, so the stage dependence weakens), and the large-river value sits at Jarrett's
0.16. Figure: `plots/gamma_readout.png`.

**The registered prediction — rho(n, gamma) > 0.9, gamma as a relabelled `n` — did not hold.** rho(n, gamma)
is +0.30. What `gamma` correlates with is the *width* channel: rho(p, gamma) = +0.89, rho(q, gamma) = +0.76
(and rho(q, p) = +0.97). The trunk did not collapse further either; its effective rank rose from 1.64 to
1.96 and it takes three directions to explain 90 % of the latent, against two for every earlier arm. §35.2's
reading ("more physics in the solver ⇒ less for the parameters to carry ⇒ deeper collapse") was drawn from
the constant arm and does not transfer to the learned one; treat it as refuted in that general form.

**But it bought nothing.** Skill is inside noise of the control (−0.004 NSE, +0.0005 KGE) and better than the
constant arm; the width exponent `b` is back at the control's value (−0.017 vs 0.004), so the geometric
improvement that made the constant arm interesting (§35, `b` → 0.098) is gone once `gamma` is free to move.
Compare the two `gamma` arms: the constant forced every reach to 0.35 and the head answered by raising `q`
(0.295 → 0.408) and `p`; the learned arm settled `gamma` lower (0.22) and `q` *fell* (0.261), and the
downstream exponent went with it. The daily objective, given a free `gamma`, uses it as a third direction
that is correlated with the width parameters and orthogonal to skill.

**Verdict on the handoff's question** (learned field or fixed constant): on skill grounds undecided (both
within 0.01 of the control); on geometry grounds the constant is the only arm that moved `b`, and only at a
skill cost. Neither arm is promotable as-is. What is now settled: (i) `gamma` is not `n` relabelled, so the
§31/§32 "one latent direction" story does not automatically extend to a fourth output; (ii) the learned
field is physically ordered, which is worth one figure in the paper as a positive example next to the
`q`/`p` negative ones; (iii) the open experiment remains the small-constant sweep, `gamma ∈ {0.1, 0.183}`,
scored on both NSE and `b`, which the learned arm's median (0.22) now brackets from above.

**Breathing.** Over water year 2000, with 192,152 dead reaches (55.5 %) excluded and accumulated Q′
as the discharge, the typical live reach swings its Manning's `n` by a median **1.46x** between its driest and
wettest day (p90 2.24x), against 1.91x (p90 3.26x) on the constant `gamma = 0.35` arm over water year 1996;
the learned field's lower median (0.22) and its fall with river size both damp the swing where the flow
range is largest. Figures: `plots/n_of_d_wy2000_area.gif` (log drainage area vs n(d), one frame per day),
`n_of_d_wy2000_3d.png` / `_3d_heatmap.png`, `n_of_d_wy2000_traces.png`.

**Routing lag** (§36.6 method) on this arm: identical to the control below 10,000 km²; above it the router
adds a median 2 days at 10,000–30,000 km² and 3 above (control 1 and 2; the gauges ask for 1 and 2), on 138
gauges — a hint that the learned stage law slows the largest rivers slightly too much, consistent with
`gamma` being lowest but not zero there. Correlation with observations is 0.886 routed vs 0.880 summed.

### 36.6 Routing lag against the summed Q', on the two finished arms

Asked directly: does routing add a delay, and is it the delay the gauges ask for?
`experiments/stage_roughness/routing_lag.py` cross-correlates daily anomalies over the full 15-year eval window
and takes the lag (whole days) that maximises the correlation, per gauge, for three pairs.

| drainage area (km²) | n | summed Q' → routed | summed Q' → observed | routed → observed | r(summed, obs) | r(routed, obs) |
|---|---|---|---|---|---|---|
| < 300 | 491 | 0 [0, 0] | 0 [0, 1] | 0 [0, 1] | 0.865 | 0.856 |
| 300–1,000 | 776 | 0 [0, 0] | 0 [0, 1] | 0 [0, 1] | 0.888 | 0.887 |
| 1,000–3,000 | 617 | 0 [0, 1] | 0 [0, 2] | 0 [0, 1] | 0.882 | 0.896 |
| 3,000–10,000 | 343 | 1 [0, 2] | 1 [0, 3] | 0 [0, 1] | 0.885 | 0.911 |
| 10,000–30,000 | 126 | 1 [1, 3] | 1 [0, 4] | 0 [−1, 2] | 0.883 | 0.922 |
| > 30,000 | 12 | 2 [1, 4] | 2 [1, 8] | 0 [0, 4] | 0.850 | 0.891 |

(control arm `gamma = 0`; median [p10, p90]. The `gamma = 0.35` arm is the same to within one gauge class.)

The router adds the lag the observations ask for — 0 days below ~1,500 km², one day through 30,000 km², two
above — at 69.9 % of gauges, and the residual routed-to-observed lag is 0 at 75.1 %. The gain in correlation
from routing grows with basin size (+0.03 to +0.04 above 3,000 km²) and is nil below 1,000 km², where the
summed Q' already has the right timing at daily resolution. Two cautions: daily output cannot resolve a
sub-day lag, so most of CONUS reads as 0 by construction; and the examples
(`plots/routing_lag_examples_wy2000.png`) show that for basins under a few thousand km² the routed and summed
series are nearly on top of each other — routing's visible work there is peak attenuation, not delay.

### 36.7 The small-constant sweep: a monotone trade, no interior optimum

Two more constant arms, `gamma = 0.1` (`2026-09-12T20-38-08Z`) and `gamma = 0.183`
(`2026-09-12T20-36-00Z`), each a one-line change from the `0.35` config, run in parallel on CPU from binary
`7d3bd49`. With the control and the learned arm that is five points on one axis:

| gamma | run | NSE | KGE | `b` (L&M 0.50) | `f` (0.40) | median `q` | negative solves | per-gauge dNSE, % up |
|---|---|---|---|---|---|---|---|---|
| 0 (control) | `03-53-34Z` | 0.7458 | 0.7619 | 0.004 | 0.551 | 0.295 | 0.0137 % | — |
| 0.1 | `20-38-08Z` | 0.7456 | 0.7620 | −0.017 | 0.543 | 0.289 | 0.0211 % | −0.0001, 48 % |
| 0.183 | `20-36-00Z` | 0.7416 | 0.7611 | 0.065 | 0.532 | 0.351 | 0.0295 % | −0.0005, 43 % |
| 0.35 | `06-06-19Z` | 0.7362 | 0.7588 | 0.098 | 0.484 | 0.408 | 0.0381 % | −0.0015, 38 % |
| learned (median 0.22) | `16-30-14Z` | 0.7420 | 0.7624 | −0.017 | 0.550 | 0.261 | 0.0309 % | −0.0006, 43 % |

Three readings.

1. **The trade is monotone and roughly linear.** Skill falls and the width exponent rises together as the
   constant grows: about 0.03 of `b` per 0.01 of NSE across 0.183 and 0.35. There is no interior value that
   keeps the geometry and the skill, which answers §35.3's open question in the negative. At 0.1 the term is
   inert on every axis, including geometry.
2. **The head compensates the same way at every constant.** Median `q` rises with gamma (0.29, 0.35, 0.41)
   and the peak bias grows slightly more negative; the roughness law makes floods faster and sharper in
   isolation, and the head answers by widening the channel and damping them back. `b` improves as a side
   effect of that compensation, not because the objective asked for geometry.
3. **The learned field is the 0.183 arm's skill with the control's geometry.** Same NSE to within 0.001,
   `b` back at zero, `q` at its lowest. Given gamma as a free direction the head lowers `q` and lets gamma
   carry the stage dependence, which is the degeneracy argued in §36.5.

Negative pre-clamp solves rise with gamma, from 0.014 % to 0.038 % of reach-timesteps: still one solve in
several thousand, floored to 1e-4 m³/s, and every micro-batch has a few. It is the Muskingum coefficient
window (`.claude/REACH-SUBDIVISION.md`), not an instability, but it is a real cost of the term and the
stage law roughly doubles it.

**Verdict on the constant-versus-learned question.** Neither is promotable on skill. If the paper wants
the geometry argument, the constant is the only form that delivers it, at a known price; if it wants a
learned physical field as a positive example next to the `q`/`p` negatives, the learned gamma is that,
ordered by river size and uncorrelated with `n`, and it costs nothing. The experiment that would separate
"gamma is degenerate with q" from "gamma is unidentifiable" is to learn gamma with `q` held fixed; it has
not been run.

## 37. The head that learns only n_0 and gamma: with the channel fixed, gamma is n_0 relabelled

The decision after §36.7 was to make n_0 and gamma the only KAN outputs, with the channel shape prescribed
(p = 21, the DDR default coefficient; q = 0.65, the Leopold & Maddock at-a-station b/f). Two reasons: it removes
the width channel that a free gamma had been riding, so identifiability can be read without the degeneracy; and
a two-output head is what the landscape and adjoint instruments can probe directly. Config
`config/experiments/sr_n0_gamma.yaml`, matched control `sr_n0_only.yaml` (n_0 alone, same fixed channel).
Both 500 updates, seed 42, 2,365 gauges, CPU, binary `becc4b4`.

### 37.1 Skill

| arm | run | NSE / KGE | per-gauge dNSE vs its control | b (by construction) | negative solves |
|---|---|---|---|---|---|
| free channel control (§33) | `03-53-34Z` | 0.7458 / 0.7619 | | 0.004 | 0.014 % |
| n_0 only, channel fixed | `23-39-06Z` | 0.7408 / 0.7612 | −0.0002, 46 % up (vs free) | 0.284 | 0.020 % |
| **n_0 + gamma, channel fixed** | `23-39-03Z` | 0.7391 / 0.7592 | −0.0004, 42 % up (vs n_0 only) | 0.285 | 0.031 % |

Prescribing the channel costs about 0.004 NSE, which the head mostly absorbs through n_0 (median 0.064
against 0.10 with the channel free: the prescribed channel is wider and shallower, so the head smooths it).
Adding gamma on top costs nothing and buys nothing: 48 gauges move by more than 0.05 NSE, 19 of them up.
The downstream width exponent is 0.284 in both by construction (q·f with p constant), incidentally the
closest any arm has come to Leopold & Maddock's 0.50, and it came from prescribing, not learning.

### 37.2 The learned gamma collapsed onto n_0

| | free channel (§36.5) | **channel fixed** |
|---|---|---|
| gamma median (p10, p90) | 0.221 (0.159, 0.312) | **0.067 (0.028, 0.189)** |
| rho(n_0, gamma) | +0.30 | **+0.90** |
| gamma by size, <100 km² to >10,000 km² | 0.236 to 0.149 | 0.084 to 0.053 |
| per-reach breathing, water year 2000 | 1.46x | **1.09x** (n_low/n_high median 1.05) |

The registered prediction of §36 (rho above 0.9) that the free-channel arm refuted holds here. The gamma
against n_0 panel of `plots/gamma_readout.png` is a single tight monotone curve: gamma is a function of n_0.
The size ordering and the physically plausible median of the free-channel arm are therefore not properties
of the stage law the gauges learned; they were gamma tracking p and q (rho 0.89 and 0.76 there). Take the
width parameters away and the head has one roughness direction, which it emits twice.

What the gauges do constrain is the composite: roughness at the depths the reach actually runs. n_0 is
roughness at 1 m and most reaches run shallower, so n_0 and gamma trade along the curve that keeps
n(d) fixed at the typical depth. That is the same non-identifiability as §5 fact 5 in the skill (a gauge
sees a network sum) one level down: even at one reach, daily discharge sees n at one effective depth,
not the slope of n against depth.

### 37.3 What this settles and what the landscape adds

- The learned stage exponent is not identifiable from daily gauges in this model: unchanged skill, and a
  field that is a relabelling of n_0 once the channel cannot absorb it. The constant-gamma sweep (§36.7)
  remains the only way the term moved anything, and it traded geometry for skill linearly.
- The "physically ordered gamma" of §36.5 is withdrawn as evidence of identifiability. It is on the
  do-not-use list with that note.
- Open: the per-gauge (n_0, gamma) landscape (`experiments/landscape-n0-gamma`, §37.4 when it finishes)
  gives the curvature along gamma at fixed n_0 and the orientation of the sloppy valley. The prediction
  from the collapse is a valley along the n(d)-preserving curve, i.e. H_gamma,gamma small against H_nn
  and the stiff eigenvector nearly along n_0.

### 37.5 Do dams explain where roughness is high? A small, real, Midwest-concentrated signal

User hypothesis (2026-09-13, from the low/high-flow roughness maps of the n_0 + gamma arm): the slow, rough
stretches of the Mississippi basin in the Midwest coincide with dams, since an impounded reach runs deep and slow
at every discharge and a model with a prescribed channel can only say so through n_0.

Test (`experiments/stage_roughness/dam_roughness.py`): the 2,178 MERIT reaches hosting a HydroLAKES/GRanD
reservoir outlet (DDR's `merit_reservoir_params.csv`; 2,144 reservoirs, 34 dam-controlled natural lakes), plus
their neighbours one to three hops upstream (impounded) and downstream (regulated), against every other live
reach in the same 0.25-dex drainage-area bin. Residual = n_0 minus the bin median of unaffected reaches.

| n_0 residual, median | CONUS, n_0 + gamma arm | Midwest box (98–84 W, 36–47 N) |
|---|---|---|
| dam reach | +0.0025 (n = 2,178, p = 6e-6) | −0.005 (n = 338) |
| 1–3 hops upstream | +0.0013 to +0.0006 | −0.009 to −0.010 |
| 1–3 hops downstream | +0.0002 to +0.0013 (n.s.) | −0.002 to +0.004 |
| all other reaches | 0 by construction | **−0.014** (n = 27,905) |

Reading. CONUS-wide the dam effect is a few per cent of n_0 (0.057 against 0.058 at the median; the p-value is
small because n is large). Inside the Midwest box the picture the user saw is real but inverted from the naive
reading: Midwest reaches as a population are *smoother* than same-size reaches elsewhere (residual −0.014 on a
base of 0.044), and it is the dam reaches and their regulated downstream neighbours that stand *out* of that
background by +0.009 to +0.018, i.e. 20–40 % rougher than the surrounding unaffected reaches. The same ordering
appears in the free-channel arm (`16-30-14Z`: Midwest dam reaches −0.010 against other −0.015). So dams do mark
the rougher reaches within the Midwest, but they do not explain the region's roughness level, which is low, and the
CONUS-wide association is weak.

Caveats. The reservoir set is 0.6 % of reaches, and a reservoir's attenuation has to be absorbed over many reaches
in a model with no storage term, so three hops is a short reach of the effect; the residual is against a size bin,
not a regional baseline, which is why the Midwest column is negative for everyone; and n_0 in this arm is
roughness at 1 m with a prescribed channel, so a deep impounded reach is rough by construction only if the model
routes it at the wrong depth. Figure `plots/dam_roughness.png` in both run directories (n_0 against size by group,
residual box plots, a map of dam reaches coloured by residual with the Midwest box drawn).

### 37.4 The (n_0, gamma) landscape: a valley that runs straight along gamma

`experiments/landscape-n0-gamma` on the finished checkpoint (`epoch_50_mb_9` of `23-39-03Z`), first use of
`landscape.axes: [n, gamma, q_spatial]` (q fixed, so slot 2 is inert). 15 gauges spanning 5 to 213 reaches,
one 5-year testing window (1995-10-01 to 2000-09-29), central-difference Hessian at h = 0.05, damped Newton
(8 iterations, alpha capped at ±1.1 = a factor 3), 11×11 grids on the n-gamma and stiff-sloppy planes. Five
shards, about 20 min per gauge. One listed staid is not in the eval population; 13 of the remaining 14 are in
as this is written, the 14th (13317000, the largest) still running. Pre-registered bars, as for q in §29: median
|H_gg| / |H_nn| ≥ 0.25 and H_gg > 0 at ≥ 70 % of well-fit gauges.

| gauge | reaches | NSE trained → optimum | alpha* (n_0, gamma) | H_nn at trained | H_gg at trained | ratio |
|---|---|---|---|---|---|---|
| 01047000 | 17 | 0.815 → 0.843 | −0.41, **−1.10** (bound) | 7.4e-1 | −4.1e-3 | 0.006 |
| 02177000 | 8 | 0.846 → 0.857 | −0.64, **−1.10** | 9.0e-2 | −2.4e-3 | 0.026 |
| 01055000 | 5 | 0.680 → 0.688 | −0.45, **−1.10** | 2.3e-1 | −5.5e-3 | 0.024 |
| 03155000 | 77 | 0.861 → 0.902 | −0.43, +0.72 | 9.6e-1 | −8.3e-3 | 0.009 |
| 11274500 | 7 | 0.849 → 0.866 | +0.61, −0.19 | 5.3e-2 | 4.7e-2 | 0.882 |
| 01449000 | 31 | 0.694 → 0.745 | +0.74, −0.25 | −4.6e-2 | 2.4e-3 | 0.052 |
| 05451500 | 89 | 0.824 → 0.836 | −0.66, **−1.10** | 2.8e-2 | −7.9e-4 | 0.029 |
| 12451000 | 23 | 0.732 → 0.732 | +0.14, −0.02 | −2.5e-3 | 8.6e-4 | 0.340 |
| 01563500 | 127 | 0.824 → 0.825 | −0.04, **−1.10** | 1.4e-1 | 7.7e-4 | 0.005 |
| 06934000 | 163 | 0.876 → 0.894 | +0.15, −0.68 | 4.3e-1 | 2.7e-3 | 0.006 |
| 01567000 | 213 | 0.867 → 0.867 | +0.00, +0.01 | 4.0e-1 | 2.3e-3 | 0.006 |
| 07068000 | 113 | 0.772 → 0.772 | −0.00, −0.00 | 1.1e-1 | 2.6e-3 | 0.023 |
| 14166000 | 181 | 0.864 → 0.866 | +0.10, −0.04 | 4.5e-2 | 6.0e-3 | 0.134 |

Summary over the 13: median |H_gg| / |H_nn| **0.024** at the trained point (0.034 at the optimum); H_gg > 0
at 8 of 13 trained points and 12 of 13 optima; gamma driven to the alpha bound (a factor 3 smaller) at 5 of 13
optima with essentially no change in loss at three of them; median NSE gain to the per-gauge optimum 0.011.
Figure `n_gamma_planes.png` in the study directory: at every gauge but the two smallest basins the NSE contours
on the n-gamma plane are vertical stripes, i.e. the loss is a function of n_0 alone across a factor of three in
gamma either way.

**Both bars fail.** The magnitude bar by a factor of ten (0.024 against 0.25; q managed 0.062), the sign bar at
the trained point (62 %, against 70 %). The two exceptions to the stripe pattern, 11274500 (ratio 0.88) and
12451000 (0.34), are the two basins where n_0's own curvature is smallest (5e-2 and −2.5e-3), so the ratio is
large because the denominator is tiny, not because gamma is constrained; 12451000's whole plane spans 0.006 NSE.

**What it means.** With the channel prescribed, the per-gauge loss has one stiff direction and it is n_0. gamma
is flatter than q was in §29 by a factor of two to three, and the training result of §37.2 (gamma collapsed
onto n_0 at rho 0.90) is the optimizer walking that valley: any gamma is as good as any other once n_0 has been
set for the depth the reach usually runs at, so the head emits whatever the trunk's one direction gives it. This
is the cleanest statement of the paper's thesis produced so far: the daily hydrograph identifies Manning's
roughness at one effective depth per gauge and nothing about its dependence on stage, even when that
dependence is the only physical degree of freedom left in the channel.

Caveats: 13 gauges, one seed, one window; the trained point is not at the optimum in n_0 at most gauges (the
500-update convergence issue of §21, visible as the black dot sitting off the stripe's centre), and gamma's
multiplicative perturbation of a field whose median is 0.067 spans 0.02–0.2, inside the box but on the low
side. A stratified-400 census with the same axes is the natural next run (about 6 h sharded).
