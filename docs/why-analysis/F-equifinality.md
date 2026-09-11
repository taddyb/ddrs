# Analyst F: equifinality or error? Where the displacement from the gauge optimum is free

**Data.** Five-year census (`landscape-p21-all-5yr/merged`, p = 21 model, epoch 30, 2-D Newton over basin-uniform
multipliers on n and q), 2,124 well-fit gauges (NSE at the trained point > 0.3), covariate table
`merged/figures/covariates.csv`. Per gauge: the 2-D Hessian at the optimum, its eigenpairs, the trained point's
coordinates in that eigenbasis, and the behavioural half-widths at a 10 % loss tolerance. Script and per-gauge
table: `merged/figures/why/F_classes.csv`; figures `F_classes.png`, `F_eigenbasis.png`.

**Definitions.** Stiff axis = eigenvector of the larger eigenvalue λ₁ at the optimum; sloppy axis = the other, λ₂.
`r_k = |c_k| / w_k` is the trained point's distance along axis k in units of the 10 % behavioural half-width
(quadratic approximation). Classes: (i) at optimum, r₁ and r₂ below 0.25; (ii) inside the behavioural set, both
below 1 (equifinal: the batch's answer is as good as the gauge's to within 10 % of the loss); (iii) outside along the
sloppy axis only (cheap displacement); (iv) outside along the stiff axis (costly displacement, real error);
(v) degenerate (optimum on the box edge, range-bound, or λ₁ ≤ 0).

## Observations

**1. The sloppy axis is q, the stiff axis is n, at every basin size.** Angle of the stiff eigenvector from the n axis
(0 = pure n, 90 = pure q), well-fit gauges:

| reaches | count | p25 | median | p75 |
|---|---|---|---|---|
| ≤ 10 | 688 | 1.0 | 12.6 | 33.5 |
| 11 to 50 | 892 | 1.2 | 10.1 | 30.2 |
| 51 to 200 | 404 | 1.4 | 8.4 | 23.5 |
| > 200 | 140 | 1.3 | 6.8 | 26.7 |

Anisotropy λ₁/λ₂ (median) grows with size: 71, 162, 186, 301. Large basins have a sharper, more n-aligned valley;
small basins a mixed n–q trade with a wider spread of orientations.

**2. The Hessian at the reported optimum is indefinite at 53 % of well-fit gauges (1,131 of 2,124).** In 86 cases
λ₁ ≤ 0; in the rest λ₂ < 0 along the sloppy (q-dominated) axis. The negative curvature is not always small:
|λ₂|/λ₁ has median 0.29, p75 0.97, p90 6.9; only 27 % of these cases have |λ₂| below 5 % of λ₁ (flat within the
finite-difference noise). Read strictly, most "optima" are saddles along q: the damped Newton stops where the
gradient vanishes but the loss still falls if q moves away in either direction. Read leniently, q is unconstrained
there and only the n coordinate carries information. Both readings are reported below.

**3. Classification.** Strict (λ₂ ≤ 0 counts as degenerate) and lenient (λ₂ ≤ 0 or |λ₂|/λ₁ < 0.05 means the sloppy
axis is flat and the gauge is judged on the stiff axis alone):

| class | strict n (share) | lenient n (share) | median gain (lenient) | gain > 0.02 | median n*/n | wants slower |
|---|---|---|---|---|---|---|
| i at optimum | 184 (9 %) | 470 (22 %) | 0.001 | 10 % | 0.99 | 48 % |
| ii inside the set | 324 (15 %) | 76 (4 %) | 0.005 | 15 % | 0.86 | 43 % |
| ii inside the set, flat sloppy axis | | 490 (23 %) | 0.006 | 16 % | 1.35 | 68 % |
| iii sloppy only | 5 (0 %) | 5 (0 %) | 0.011 | 0 % | 0.56 | 20 % |
| iv stiff | 335 (16 %) | 749 (35 %) | 0.029 | 64 % | 2.52 | 93 % |
| v degenerate | 1,276 (60 %) | 334 (16 %) | 0.014 | 41 % | 2.15 | 87 % |

Under the lenient reading: 45 % of well-fit gauges are at their optimum or inside its 10 % behavioural set
(classes i and ii): equifinality, not error. 35 % are displaced along the stiff axis by a median 2.3 half-widths and
would gain a median 0.029 NSE (64 % above 0.02); 93 % of those want slower routing, median n × 2.5. Class iii is
empty: when a gauge is displaced, it is displaced along the axis it can see.

**4. Class iv (real error) by ecoregion and size.** Share of well-fit gauges in class iv: Eastern Highlands 51 %,
Southeast Plains 49 %, West Plains 41 %, Northeast 40 %, Central Plains 39 %, Southeast Coastal Plain 35 %, Xeric
West 26 %, Western Mountains 14 %, Mixed Wood Shield 13 %. By size the class-iv share is flat (32 to 41 %); the
"at optimum" share is flat too (21 to 24 %). Class-iv gauges: median 19 reaches, 820 km², eval NSE 0.74 over the
15 years (they are ordinary, well-simulated basins, not outliers).

**5. q-flat gauges have a well-defined n optimum.** Among the 531 gauges with the smallest |H_qq| in natural units
(bottom quartile), 95 % have λ₁ > 0 and 83 % have the stiff axis within 30° of n (median 8.6°); their median gain is
0.002 and 43 % are at optimum or inside the set. Flatness in q does not blur the n optimum.

**6. The trained point itself.** At the trained point the 2-D Hessian has H_nn ≤ 0 at 46 % and H_qq ≤ 0 at 69 % of
well-fit gauges (determinant ≤ 0 at 48 %). The batch solution sits on a ridge or shoulder of the per-gauge loss more
often than in a bowl; combined with observation 2, the per-gauge loss surface over (n, q) is non-convex over the box
at most gauges, with a single clear valley in n and an ambiguous, often two-sided, behaviour in q.

## Interpretation

- **Equifinality, in the strict sense, covers about 45 % of well-fit gauges.** Their trained parameters are within
  10 % of the loss of the gauge's own best, and for most of them the q coordinate is unconstrained. Nothing about
  training is wrong at these gauges; the map's red there (§13) is distance without cost.
- **Real error covers about 35 %**, concentrated east of the Rockies in rain-driven basins (Eastern Highlands,
  Southeast, Northeast, Central Plains). At those gauges the displacement is along the axis the gauge sees, almost
  always toward slower routing, and it costs 0.03 NSE at the median. This is the population the training-side and
  inflow-bias analyses (D, E) should focus on; the equifinal 45 % should be excluded from those correlations.
- **The saddles are the instrument's honest report of a non-convex q direction, not a bug**, but they mean the
  reported q* is often not a minimum. The 2-D optimum's n coordinate is trustworthy (λ₁ > 0 in 96 % of well-fit
  cases); the q coordinate is not, and half-widths along q should not be quoted. The Newton search should be followed
  by a 1-D line scan in q at n* to find the true q minimum (or confirm flatness) before any q statement is made.
- **The n–q trade is weak.** Stiff eigenvectors lie within 13° of the n axis at the median in every size class, so
  with p pinned the gauge constrains n almost on its own, and q is either flat or non-convex. This supports treating q
  as a nuisance parameter for the routing question and either fixing it or reparametrising it (analyst A).

## One decisive follow-up test

A 1-D line scan in q at fixed n* for a stratified sample of 60 saddle gauges (20 each from |λ₂|/λ₁ < 0.05, 0.05 to 1,
> 1), 25 points over the box, five-year window: about 2 minutes per gauge on one core, 2 hours total as 6 shards. If
the loss is flat to within the FD noise, the saddles are numerical and the lenient classification stands; if the
loss drops away from q* on one or both sides, q has a second regime the Newton search cannot reach, and the class-ii
"flat sloppy" gauges need reclassifying by the loss at the q minimum. Either outcome fixes how q is reported.

## Out of scope, noted

The `grad_star` variable is all zeros in the census files even where the search stopped on a bound; it is not
usable as a convergence diagnostic. Not investigated here.
