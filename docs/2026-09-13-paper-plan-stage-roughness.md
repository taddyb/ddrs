# What the n(d) experiments mean for the paper and the AGU abstract, and a plan (2026-09-13)

Written overnight for the morning of 2026-09-13. Sources: findings §32–§37.5
(`docs/2026-09-08-landscape-hypothesis-tests-findings.md`), the experiment handoff
`/tmp/experiment-handoff-learned-stage-roughness.md`, the abstract draft
`~/papers/ddr_equifinality/abstract_draft_2026-09-10.md`, and `paper.tex` as of 2026-09-05.

## 1. Where the paper and the abstract stand

The manuscript ("Beyond Equifinality in Differentiable River Routing") still carries the July design:
inflow-source arms R1–R5, hypotheses H1–H4 about bias absorption, and an abstract that ends by *asking*
whether the loss landscapes settle on one set of channel parameters. The 2026-09-10 abstract revision
answers that question with the adjoint landscape census: every gauge constrains one combination of the
channel parameters, dominated by roughness; the width exponent is flat at 85 % of gauges; the displacement
from the per-gauge optimum is reproducible and nearly free; two models of equal skill carry roughness a factor
2.4 apart. The thesis is "equifinality, computed rather than sampled, is a property of what daily discharge
can say about a channel".

## 2. What the stage-roughness work adds, in the order it strengthens that thesis

**A. The collapse is in the weights, not only in the loss surface (§32).** The KAN trunk starts at effective
rank 4.4 of 21 over inputs of rank 6.1 of 10, and the gradient collapses it to 1.4 inside the first epoch;
every topology screened can represent independent fields when asked to. This is the abstract's claim shown
inside the learned representation, and it corrects §31's "the head is defective" reading.

**B. The objective actively prefers the physically wrong channel (§33.1, §35.1, §36.7).** Two matched changes
are mirror images: freeing p buys +0.008 NSE and −0.095 in the downstream width exponent; a constant stage law
buys +0.094 in the exponent and −0.010 NSE. The constant-gamma sweep makes this a line: 0.1 inert, 0.183 and
0.35 trading geometry for skill roughly linearly, no interior optimum. "Equifinal" undersells it: skill and
plausibility are anti-correlated, at a measured exchange rate.

**C. Roughness is identifiable only at one effective depth (§36, §37).** Stage-dependent roughness
n(d) = n_0 (d/d_ref)^(−gamma) is the one physical term that can move the at-a-station velocity exponent off
Manning's 2f/3, and it has textbook support (Limerinos, Jarrett, Ferguson). Learned with the channel free it
looked identifiable: median 0.22, ordered by river size, rho(n, gamma) 0.30. Learned with the channel fixed
(the design adopted 2026-09-13) it collapses: median 0.07 and rho(n_0, gamma) 0.90, a single monotone curve.
The earlier ordering was gamma riding p and q. Skill is unchanged in every arm. The (n_0, gamma) landscape on
14 gauges confirms the geometry of the valley (§37.4): NSE contours on the n-gamma plane are vertical
stripes at every basin larger than a few reaches, median |H_gg|/|H_nn| 0.024 against the 0.25 bar, gamma
driven to its bound at six optima at no cost. This is the
sharpest form of the thesis: the daily hydrograph identifies Manning's n at the depth the reach usually runs,
and nothing about how n changes with depth.

**D. What roughness absorbs (§37.5).** Dams: dam reaches and their regulated neighbours are 20–40 % rougher
than surrounding reaches inside the Midwest, a few per cent CONUS-wide. Small, real, and a concrete example of
n_0 standing in for a missing process (storage), which is the "bias absorber" of the paper's H2 in a new guise.

**E. Instrument work that the methods section can now claim.** Learned gamma is a real autograd parent
(six-parent op, gradient-exact); the landscape can take any parameter as an axis; the parity test that caught
the eval-path bug is the kind of check a reviewer will want to hear about.

## 3. Decisions to make in the morning

1. **What the headline is.** Three candidates, one figure each:
   - the trunk-rank collapse trajectory (novel, visual, weights-level equifinality);
   - the skill–plausibility line (five arms on one axis: NSE against width exponent), the most quotable;
   - the gamma collapse pair (rho 0.30 with the channel free, 0.90 with it fixed), the cleanest
     identifiability statement.
   My recommendation: lead the abstract with B (the exchange rate), put C as the closing sentence, use A as
   the mechanism paragraph in the discussion.
2. **Whether the paper keeps the R1–R5 inflow arms as its spine.** They are the H1–H4 experiment and still
   unfinished (R4/R5). Option: demote them to one results subsection ("bias in the inflow is absorbed by
   roughness") and make the landscape and the stage-roughness identifiability the spine. That matches the
   revised abstract and everything we have numbers for.
3. **The single sentence on stage roughness for the AGU abstract** (2,000-character limit is tight). Proposed:
   "Adding the one physical degree of freedom that could reconcile the model with observed at-a-station hydraulic
   geometry, a stage-dependent roughness, changes skill by less than 0.005 and, once channel shape is
   prescribed, collapses onto Manning's n: daily discharge identifies roughness at one depth and nothing about
   its variation with stage."

## 4. Concrete plan

### Presentation (AGU, 12–15 minutes; figures already exist unless marked)

1. Beven's question, one slide.
2. The instrument: adjoint gradient and curvature at every gauge (existing figure from the census).
3. Selective equifinality map (existing: per-gauge displacement and cost).
4. Collapse in the weights: trunk rank trajectory (existing, `experiments/head_arch/trunk_trajectory.py`).
5. The exchange rate: NSE vs width exponent, five arms on one line (NEW, one script, half a day).
6. Stage roughness: what it is (the n(d) basin GIF, existing) and what it learned (the gamma-against-n_0
   panel, existing, two arms side by side).
7. The (n_0, gamma) landscape at a gauge: n-gamma plane with the flat valley (NEW from §37.4 output,
   `experiments/landscape/plot3d.py` or `surface.py`, half a day).
8. Close: what a gauge can say about a channel; the trust map.

### Paper (order of work)

1. Rewrite the abstract per §3.1–3.3 (one sitting).
2. New results subsection "The channel parameters a gauge can see": census numbers, the exchange rate,
   the gamma collapse pair, the (n_0, gamma) landscape. Pull numbers only from
   `research-status.md` and findings §§32–37.
3. New methods paragraph: stage-dependent roughness (equations §36.1), the six-parent op, gradient checks,
   the landscape axes.
4. Decide the fate of R1–R5 (§3.2). If demoted, the H1–H4 verdict table becomes one paragraph.
5. Corrections carried from the handoff: retire §31's architectural verdict in the text; do not cite the
   free-channel gamma's size ordering as identifiability; rewrite the Leopold & Maddock spec's range
   recommendation (docs only).

### Experiments that would strengthen the paper (ranked, each with cost)

1. **A second seed of the n_0 + gamma arm** (2.5 h CPU). Turns "gamma collapses onto n_0" into a reproducible
   statement and gives the seed-to-seed floor for the landscape.
2. **The (n_0, gamma) landscape on the stratified 400** (about 6 h sharded). Turns 15 gauges into a census
   with the same bars used for q, so the abstract can say "at X % of gauges".
3. **The exchange-rate figure with error bars**: rerun the 0.183 and 0.35 constants with a second seed
   (5 h). Only if the line is the headline.
4. **Prescribed channel by river size** (p growing with drainage area, the Leopold & Maddock spec's surviving
   idea) with n_0 + gamma. Makes the prescribed geometry defensible rather than a constant; one arm, 2.5 h.
5. **Not worth it now:** more constant-gamma values, Juniata screens, further head-topology arms.

## 5. What is in the repo for all this

- PR #44 (draft): everything since master, CI green.
- Figures per run under `.ddrs/runs/<id>/plots/`; the n(d) family via `experiments/stage_roughness/animate_n_of_d.py`
  (`--view area|3d|maps|scatter`), the basin GIF, `gamma_readout.py`, `routing_lag.py`, `arm_delta_by_gauge.py`,
  `dam_roughness.py`; landscape output under `.ddrs/experiments/landscape-n0-gamma/`.
- Journal entries for every run; research-status table with all seven arms.
