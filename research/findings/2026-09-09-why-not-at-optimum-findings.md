# Why is a gauge not at its optimum? Synthesis of the six-analyst swarm (2026-09-09)

**Executive summary.** We asked, for every one of 2,365 test gauges, why the roughness the model was trained with
differs from the roughness that gauge alone would choose. About one gauge in five is already at its optimum and
another two in five sit inside the optimum's behavioural set: the difference there is free, because at small or
steep basins the flood wave crosses the channel within one daily time step and the daily score cannot see roughness
at all. The remaining third, mostly rain-fed rivers east of the Rockies with a day or more of travel time, carry a
real, systematic error: the model's channel adds almost no delay, the routed flow arrives as early as the raw inflow,
and the gauge wants routing two to three times slower, worth about 0.03 NSE. Clamping, low flow, loss weighting,
and too few optimizer steps are all ruled out; what is left is a channel geometry that is too fast on main stems,
and a head whose attributes carry only a tenth of the information needed to place n per gauge. The width exponent q
is unidentifiable from daily discharge at 85 % of gauges and should be prescribed or reparametrised, not learned.

Reports: `research/why-analysis/A-flat-q.md`, `B-clamping.md`, `C-weak-gradient.md`, `D-inflow-bias.md`,
`E-training-side.md`, `F-equifinality.md`. Census facts: `research/findings/2026-09-08-landscape-hypothesis-tests-findings.md` §13, §14.

## 1. Question and data

The model is the user's p = 21 run (`2026-09-08T15-55-52Z-conus-train-and-test`, epoch 30, trained on gages_3000; width
coefficient fixed at 21, Manning n and width exponent q learned per reach by the attribute-conditioned head). For each of
the 2,365 gauges the run's eval scores, the five-year census (`landscape-p21-all-5yr/merged`, water years 1996 to 2000)
gives the 2-D Newton optimum over basin-uniform multipliers on (n, q), the Hessian and its eigenpairs at the optimum,
the behavioural half-widths, and the gain `NSE(optimum) − NSE(trained)`. The diagnostic pass
(`landscape-p21-all-5yr-diag/merged`) adds the gradient and Hessian at the trained point and the daily series
(observed, routed at the trained point, summed inflow without routing). The covariate table
(`landscape-p21-all-5yr/merged/figures/covariates.csv`, 69 columns) joins these to GAGES-II, MERIT attributes, and the
15-year eval. "Well fit" means NSE above 0.3 at the trained point over the five years: 2,124 gauges. Displacement is
`ln(n*/n)`, positive when the gauge wants slower routing.

## 2. Verdict per hypothesis

**A. Flat q: weak gradient by construction; not low flow, not clamping.** q is flat (a factor-2 move changes the loss by
under 1 %) at 85 % of well-fit gauges, n at 11 %, and no gauge is flat in n but curved in q. Flatness in q falls
monotonically with depth at the gauge under mean flow, 94 % below 0.3 m to 28 % above 4 m, so the symmetric-leverage
prediction (flat only near 1 m) fails; in a partial regression depth drops out once slope, area, and the trained q are
in. Low-flow and zero-flow fractions do not correlate with flatness (rho −0.04, −0.01). The 15 % where q is identifiable
are deep, large, low-slope main stems (median 1.5 m, 2,600 km²). q curvature at the trained point is negative at 69 % of
gauges, so the trained q sits on a low crest: the optimum along q is noise and flips sign between windows.

**B. Clamping: refuted for the fitted population.** No well-fit gauge has a trained n or q within 1 % of a range edge.
Routed floor days occur at 2.2 % of well-fit gauges, observed zero-flow days at 8.2 %; the gauges with any such days are
steeper, not flatter (|H_nn| 0.078 against 0.014 for the clean group), and want no change in n. The 1.9 % of landscapes
flat in both parameters have no clamping and are tiny basins (median 7 reaches). Clamping marks only about 60 arid
intermittent gauges among the 241 poorly fit, where the inflow, not the channel, is the problem.

**C. Weak gradient scales with travel time, and separates cheap from costly displacement.** Curvature in n rises with
channel length (rho +0.31), area (+0.27), and the travel-time proxy (+0.26), and falls with slope (−0.43); median |H_nn|
is 0.006 to 0.008 below one day of travel and 0.023 to 0.025 at one to four days, so identifiability sets in near one
day. Displacement runs the other way (median n*/n 4.4 under half a day, 1.0 beyond eight days). Quadrants of the well
fit: far-and-flat 45 % (gain 0.009), far-and-curved 37 % (gain 0.028, 76 % want slower, median 1,226 km²), near 18 %.
Note that 63 % of gauge reaches sit at or just above the routing's slope floor of 1e-3, so "slope" mostly separates
floor-slope rivers from steeper ones.

**D. The displacement is the channel being too fast, not compensation for wrong inflow.** With fractional lags, the
Spearman correlation of the displacement with "routed flow arrives early" is +0.73; a joint linear model explains 45 %
of the variance and a boosted model 78 %, with timing dominant and volume bias under 1 %. The trained channel adds a
median 0.00 days of delay (68 % of gauges under 0.1 day), so the routed flow inherits the summed inflow's timing. Where
the routed flow leads the observation by more than half a day (483 gauges), 94 % want slower and the median gain is
0.045; where it is on time (1,114 gauges) the gain is 0.005; where it is late (161 gauges) 90 % or more want faster
with gain 0.005. At the five most displaced gauges the routed curve lies on the summed inflow, both 1.2 to 3 days early
with peaks 20 to 60 % high, and the observed hydrograph is a delayed, smoothed version of both.

**E. Training side: not weighting, not step count; information and capacity.** The NSE-batch normalisation makes per-day
weights nearly uniform, and the gauges that weigh most sit farther from their optima (rho +0.24), so reweighting would
move the median displacement little (partial contribution 0.03). The run made 60 optimizer updates and n was settled by
the twentieth. Attribute nearest neighbours want n values a factor 1.75 apart (random pairs 2.2), and a 10-neighbour
attribute regression explains 12 % of the variance of the gauge-optimal ln n; the head's trained ln n has half the
spread the gauges want. In a joint model with region controls, the routed-flow timing dominates the displacement
(drop-one dR² 0.14 of 0.31).

**F. Equifinality structure: a valley in n, a saddle in q.** The stiff eigenvector lies within 13° of the n axis at
every basin size, anisotropy grows from 71 to 301 with size, and the n coordinate of the optimum is trustworthy
(λ₁ > 0 at 96 %). The Hessian at the optimum is indefinite at 53 % of well-fit gauges, almost always along q, so the
reported q* is often a saddle. Lenient classification: 22 % at optimum, 27 % inside the 10 % behavioural set, 35 %
displaced along the stiff axis (real error: median 2.3 half-widths, gain 0.029, 93 % want slower, n × 2.5), 16 %
degenerate. q-flat gauges still have a well-defined n optimum (95 % with λ₁ > 0).

### Reconciliations

*D against E.* Both see the same fact: the routed hydrograph sits on the summed inflow, both early relative to the
gauge, and within this model only a slower channel fixes it. D calls it routing bias because the summed inflow is a
hillslope response without channel travel and is expected to lead the gauge by the true travel time, which the channel
should supply and does not (0.00 days of delay at n ≈ 0.04, p = 21, q ≈ 0.1). E calls the same thing "n absorbing a
timing error the attributes do not encode", which is the mechanism by which the head fails to supply it. The two are
consistent; what neither can settle from the gauge alone is whether the inflow product is itself early by a fraction of
a day. Two things would separate the cases: independent width or travel-time data at the main stems (a channel 21 m
wide at Newport is not physical, findings §7), and the ten-gauge hydrographs at the optimum: if the routed flow at the
gauge optimum arrives on time with a peak ratio near one, the inflow needed no correction and the miss was the channel.
Compensation for inflow error exists, but at the "wants faster" tail (17 % of gauges, gain 0.005), where the inflow is
late or misses peaks and no channel setting can help.

*A against F.* Consistent: q is flat at 85 % of gauges (A) and the stiff axis is n regardless (F), so a flat q does not
blur the n optimum. Where q is identifiable (deep main stems, 15 %) it is the same population where n is most curved.

*C against F.* Both partition the same 2,124 gauges. C's "near" 18 % matches F's "at optimum" 22 %; C's "far-and-flat"
45 % is F's "inside the set" 27 % plus most of F's degenerate 16 % (box-edge and saddle optima on flat ground); C's
"far-and-curved" 37 % is F's stiff-axis class 35 %. D's timing classes give the same split from the hydrology side:
on time 52 % (gain 0.005), early by more than a quarter day 40 % (gain 0.025 to 0.045), late 8 %.

## 3. Consolidated partition of the 2,124 well-fit gauges

| class | share | median gain | who they are | evidence |
|---|---|---|---|---|
| at the optimum | about 20 % | 0.001 | all sizes and regions; includes the 24 gauges with more than eight days of travel time, which sit at their optima with ten times the curvature | C near, F class i |
| equifinal: inside the behavioural set or on flat ground | about 45 % | 0.005 to 0.009 | small or steep basins, travel time under a day, routed flow on time within a quarter day; the Western Mountains are the largest regional block (63 % want slower there, gain 0.004) | A flat q, C far-and-flat, D on-time class, F classes ii and most of v |
| real error along the stiff axis | about 35 % | 0.028 to 0.029 | larger, floor-slope, rain-fed rivers with one to four days of travel; routed flow early by a quarter day or more; 76 to 93 % want slower, n × 2.5; class-iv shares by ecoregion: Eastern Highlands 51 %, Southeast Plains 49 %, Northeast 40 %, Central Plains 39 %, Western Mountains 14 % | C far-and-curved, D early classes, F class iv, E timing dominance |
| degenerate optimum | 16 % (overlaps the two rows above) | 0.014 | q on the box edge (11.6 %) or λ₁ ≤ 0 (86 gauges) | F |

The poorly fit 241 gauges are outside this partition: their channel landscape is flat or creased in every direction,
and the intermittent Xeric and Plains subset is clamp-marked (B); the Western Mountains subset (104 gauges) has the
right inflow volume and the wrong snowmelt timing (B, D).

## 4. What this says about training

Ranked by expected effect, each with the evidence and the confirming experiment.

1. **Give the channel a physical travel time on main stems: width as a function of river size.** Evidence: the
   trained channel adds 0.00 days of delay at two thirds of gauges (D); the class-iv gauges want n × 2.5 and the ten
   most egregious n × 4 to 9 with q at the floor (§14), which is the gauge compensating a 21 m channel on an
   8,700 km² river (findings §7); the largest basins want slower at 77 % and gain most (E). Experiment: the
   Moody and Troutman p(Q_ref) or p(A) arm from `research/findings/2026-09-08-fixed-p-assessment.md`, retrained on gages_3000, then
   this census. Success: the early-arrival class shrinks from 40 % toward the on-time share, class-iv gain falls
   below 0.01, small-basin displacement unchanged. Expected effect: the largest of any single change, because it
   addresses the 35 % that carry nearly all the available NSE.
2. **A learnable per-basin timing term (inflow delay or unit-hydrograph scale, or a learnable tau), trained jointly.**
   Evidence: routed-flow timing is the strongest covariate of both displacement and gain (D rho +0.73; E drop-one
   dR² 0.14). Experiment: retrain with the term; success: displacement loses its correlation with the lag and the
   "wants slower" majority (65 %) drops toward 50 %. Overlaps with change 1; run after it to measure the residual.
3. **Attributes that carry routing-timing information** (channel width or width-to-depth from GRWL, sinuosity,
   floodplain or wetland fraction, tile drainage or cropland, reservoir and lake storage). Evidence: attribute
   neighbours disagree by a factor 1.75 in n*; the attribute regression explains 12 % of ln n* (E). Experiment:
   retrain with the augmented attributes; success: the kNN R² of ln n* rises well above 0.12 and the median
   |ln(n*/n)| among well-fit gauges falls below 0.5.
4. **Stop learning q from daily discharge where it is unidentifiable.** Evidence: q flat at 85 % of gauges (A), a saddle
   at 53 % of optima (F), and its optimum flips sign between windows (§12). Options: prescribe q or tie it to depth by
   a downstream hydraulic-geometry relation, or learn it only where the hourly test (below) shows curvature.
5. **Sub-daily or timing-aware objective.** Evidence: identifiability of n sets in at about one day of travel (C); the
   daily NSE is blind to attenuation below that. Experiment first as a diagnostic: the same landscape at the hourly step
   at 60 gauges spanning the depth and travel-time bins (`pool: hourly`, USGS instantaneous data; A and C). If |H_nn|
   and q curvature rise by a factor 2 to 5 at short-travel-time gauges, the daily objective is the cause and an hourly
   or timing-aware loss makes n and q learnable there; if not, the flatness is physical and item 4 stands.
6. **Area or sigma reweighting** and **more optimizer steps**: expected effect small (partial contributions 0.03; n
   converged by update 20, E). Cheap to test after 1 to 3; more steps would move q rather than n.

## 5. What is equifinality rather than error

- **q is unidentifiable from daily discharge at 85 % of gauges** (A): any q in the range gives the same five-year loss
  within 1 %. Learning q per reach there learns noise; the head propagates a value from the deep-river minority.
- **n is unidentifiable below about one day of channel travel time** (C): median |H_nn| 0.006 to 0.008, gain 0.009
  despite a median displacement of a factor 4.4. That is 45 % of well-fit gauges, and the map's red there is distance
  without cost.
- **The trained point sits on a ridge or shoulder of the per-gauge surface** at most gauges: H_nn ≤ 0 at 46 %, H_qq ≤ 0
  at 69 % (F), while the optimum is convex in n at 96 %. The batch compromise lands between conflicting gauges, not in
  any one gauge's bowl, so trained-point curvature understates identifiability.
- **The saddle in q** at 53 % of optima (F) is the instrument reporting a non-convex direction honestly; q half-widths
  and q optima should not be quoted until the 1-D scan (§6) has run.
- **Real error is the complement**: about 35 % of gauges, one direction (slower), concentrated east of the Rockies,
  worth 0.03 NSE at the median and up to 0.45 at the ten most egregious rivers.

## 6. Instrument notes: fix before the next census

- q saddles: follow the 2-D Newton with a 1-D line scan in q at n* (25 points over the box) and report q* as a minimum
  only when the scan confirms one; F's stratified 60-gauge scan (2 h as 6 shards) decides whether the saddles are
  numerical or a second regime.
- `grad_star` is written as zeros even where the search stopped on a bound; record the true gradient at the reported
  optimum.
- The covariate table's `lag_days_*` are integer days; replace them with the fractional lags of
  `D_wellfit_with_fractional_lags.csv` (`early_r`, `early_q`, `delay_channel`, positive = early) and add the routed
  timing and peak ratio at the optimum (`routed_daily_star`) to the diagnostic pass.
- Box-edge optima (11.6 % of q*, 8.7 % at the floor) are bounds, not optima; flag them as such in maps and tables
  (the census netCDF has no flag beyond `hit_range_bound`).
- Use the Hessian at the optimum, not at the trained point, for any statement about identifiability.
- The slope floor of 1e-3 binds at 63 % of gauge reaches; slope covariates are effectively binary.
- Add a per-gauge clamp accumulator (floor hours, negative pre-clamp hours, depth and width floor reaches) to the
  diagnostic pass to close the clamping question at the hourly level (B).

## 7. Open questions and the next experiment

Open: whether the UH inflow product is itself early by a fraction of a day everywhere (not measurable from the gauge
alone); whether the Western Mountains failures are inflow snowmelt timing only; which gauges set the batch's n
(size- or region-stratified training); the seven PUR regions for the regional breakdown.

**Single highest-value next experiment:** retrain with width as a function of river size (item 1), on gages_3000 with
everything else unchanged, then rerun the five-year census and the diagnostic pass. It targets the 35 % of gauges that
hold nearly all the recoverable NSE, its prediction is sharp (the early-arrival class collapses, the small-basin
picture is unchanged), and it costs one training run of about two hours plus a 13-hour census. The cheap companion,
runnable today, is the hourly landscape at 60 gauges, which decides whether the 45 % equifinal share is a property of
the daily objective or of the physics.
