# Leakance & differentiable routing — literature grounding (2026-07-04)

Purpose: ground the ddrs leakance term and our four experiments
(2×2 identifiability, low-zeta diagnosis, gradient probe, synthetic
recoverability) in the peer-reviewed record, and confirm we correctly
understand how gradients flow through a differentiable routing model.
Prompted by the recoverability FAIL and its auxiliary-supervision
recommendation — we need to know that recommendation rests on established
science, not just our own runs.

Every citation below was web-verified (title + authors + venue + DOI) by a
research pass; confidence is HIGH unless flagged. 26 papers across five themes.

The term under review:
```
zeta   = leakance_factor · area_z · K_D · (depth − d_gw)      [m³/s]
area_z = (p · depth)^q · length                               [m²]
b ← b − zeta                                                   (losing reach)
```

---

## 1. The functional form is standard MODFLOW-family GW–SW physics

Our `zeta` is the Darcy exchange `Q = C·(h_stream − h_aquifer)` with streambed
conductance `C = K·(bed area)/thickness` — the exact basis of MODFLOW's River
and Streamflow-Routing packages.

1. **Harbaugh (2005)**, *MODFLOW-2005 … Ground-Water Flow Process*, USGS
   Techniques & Methods 6-A16. DOI:10.3133/tm6A16. — The RIV package:
   `QRIV = CRIV·(HRIV − h)`, `CRIV = K·L·W/M`. This IS our form: `K·(L·W)/M`
   maps to `K_D · area_z` (K_D absorbs the `1/M` bed-thickness), `L·W` is the
   plan-view bed area our `(p·depth)^q · length` generalizes to stage-dependent
   width, and `(HRIV − h)` is `(depth − d_gw)`. Also documents the RBOT cutoff
   where flux saturates — the first sign the linear head-difference is bounded.
2. **Niswonger & Prudic (2005)**, *Streamflow-Routing (SFR2) Package*, USGS
   TM 6-A13. DOI:10.3133/tm6A13. — Same Darcy leakage `Q = C·(h_str − h_aq)`,
   `C = K·W·L/M`, but coupled to streamflow routing (leakage limited by
   available flow) — the closest published analog to what ddrs does
   (loss inside a routing scheme). SFR2's unsaturated column decouples seepage
   from water-table head once the stream disconnects.
3. **Brunner, Cook & Simmons (2009)**, "Hydrogeologic controls on disconnection
   between surface water and groundwater," *WRR* 45, W01422.
   DOI:10.1029/2008WR006953. — The connected→transitional→disconnected regime
   theory: below a critical water-table depth, infiltration becomes
   **independent of `h_aquifer`** and saturates. **Direct caveat to our form:**
   `zeta ∝ (depth − d_gw)` grows unboundedly as `d_gw` drops; physics says it
   should plateau. An uncapped linear leakance over-predicts loss on
   deep-water-table reaches.
4. **Brunner, Simmons, Cook & Therrien (2011)**, "Disconnected Surface Water and
   Groundwater: From Theory to Practice," *Groundwater* 49(4), 460–467.
   DOI:10.1111/j.1745-6584.2010.00752.x. — Practitioner companion; explains why
   linear-conductance (RIV-type) packages misrepresent maximum seepage under
   disconnection. (Author-list ordering MEDIUM confidence.)
5. **Rushton (2007)**, "Representation in regional models of saturated
   river–aquifer interaction for gaining/losing rivers," *J. Hydrology*
   334(1–2), 262–281. DOI:10.1016/j.jhydrol.2006.10.008. — A single lumped
   conductance conflates **streambed and aquifer resistance**, so `C` is not a
   pure streambed property and is poorly identifiable / scale-dependent.
   **This is very likely the physical explanation for our K_D pinning at its
   range ceiling** (2×2 result) — a lumped conductance absorbing aquifer-side
   resistance the single parameter cannot separate.

**Takeaway:** the form is well-grounded, but two established caveats must live in
our docs — (a) the linear head-difference breaks under disconnection (a
saturation cap is the physically-correct fix), and (b) lumped conductance is
non-identifiable, which independently predicts the K_D ceiling behavior we saw.

## 2. Integrated models (ParFlow) are the "physics truth" our term approximates — and the source of auxiliary supervision

2. **Kollet & Maxwell (2006)**, "Integrated surface–groundwater flow modeling:
   A free-surface overland flow boundary condition …," *Adv. Water Resour.*
   29(7), 945–958. DOI:10.1016/j.advwatres.2005.08.006. — ParFlow's coupling:
   overland flow is a pressure-continuity boundary condition on 3D Richards'
   flow; losing/gaining exchange is an **emergent** vertical flux, not a
   parameterized term. Their pressure-continuity replaces our `(depth − d_gw)`;
   their subsurface K replaces our `K_D`. This is what our lumped term
   approximates by hand.
3. **Maxwell (2013)**, "A terrain-following grid transform and preconditioner
   …," *Adv. Water Resour.* 53, 109–117.
   DOI:10.1016/j.advwatres.2012.10.001. — The numerical enabler for
   continental-scale coupled runs.
4. **Maxwell, Condon & Kollet (2015)**, "A high-resolution simulation of
   groundwater and surface water over most of the continental US … ParFlow
   v3," *GMD* 8, 923–937. DOI:10.5194/gmd-8-923-2015. — PFCONUS: coupled GW–SW
   at 1 km over the same CONUS/MERIT footprint ddrs routes. Establishes that
   emergent stream–aquifer exchange is resolvable at our scale.
5. **Condon & Maxwell (2015)**, "Evaluating the relationship between topography
   and groundwater …," *WRR* 51(8), 6602–6621. DOI:10.1002/2014WR016774. —
   Gridded CONUS water-table depth; shows **topography alone does not predict
   water-table depth**. Direct cautionary evidence for `d_gw`: a DEM/topographic
   proxy is a poor groundwater-depth predictor, so `d_gw` needs a real
   groundwater field, not an attribute the KAN can derive from elevation.
6. **Yang, Condon & Maxwell (2025)**, "Unravelling groundwater–stream
   connections over the continental United States," *Nature Water*.
   DOI:10.1038/s44221-024-00366-8. — Backward particle tracking on PFCONUS
   **quantifies per-reach groundwater discharge to streams** (deep GW
   contributes >half of baseflow in 56% of subbasins). This is the most
   directly usable **reach-level auxiliary-supervision label** for the sign and
   magnitude our `zeta` predicts.

**Takeaway:** ParFlow-derived water-table depth and GW–SW flux maps are a
defensible, physically-consistent, continental-coverage auxiliary target for
supervising `d_gw`/`zeta_net` — exactly the "signal outside the gauge-discharge
loss" our recoverability FAIL says is required.

## 3. Routing: the matrix-Muskingum core and the precedent for channel loss

7. **David et al. (2011)**, "River Network Routing on the NHDPlus Dataset,"
   *J. Hydrometeorology* 12(5), 913–934. DOI:10.1175/2011JHM1345.1. — The RAPID
   paper. Matrix-Muskingum: `(I − C₁N)Q_{t+1} = …` solved as a sparse linear
   system over network adjacency `N`. **This is exactly ddrs's `(I−C)Q_{t+1}=b`
   structure.** No loss term — loss must be added to the RHS, as ddrs does.
8. **David et al. (2015)**, "Enhanced fixed-size parallel speedup with the
   Muskingum method …," *WRR* 51. DOI:10.1002/2014WR016650. — States the system
   matrix is **lower unit triangular** solved by substitution — precisely
   ddrs's invariant #3 (topologically-ordered lower-triangular adjacency) and
   the forward-sub solver in `src/sparse.rs`. Cite for the triangular-solve
   justification.
9. **Cunge (1969)**, "On the subject of a flood propagation computation method
   (Muskingum method)," *J. Hydraulic Research* 7(2), 205–230.
   DOI:10.1080/00221686909500264. — Derives Muskingum coefficients from channel
   hydraulics so the scheme is a numerical analog of the diffusive-wave
   (convection-diffusion) equation. Basis for ddrs computing C-coefficients from
   trapezoidal geometry (`src/geometry.rs`).
10. **Ponce & Yevjevich (1978)**, "Muskingum-Cunge Method with Variable
    Parameters," *J. Hydraulics Div. (ASCE)* 104(12).
    DOI:10.1061/JYCEAJ.0005119. — Variable K/X from Courant and cell-Reynolds
    numbers; justifies recomputing coefficients each timestep from local
    hydraulic state. Notes the small mass-conservation cost of variable
    parameters — relevant to how a subtracted `zeta` sink interacts with the
    volume budget.
11. **Elfeki, Ewea, Bahrawi & Al-Amri (2015)**, "Incorporating transmission
    losses in flash flood routing in ephemeral streams by using the
    three-parameter Muskingum method," *Arabian J. Geosciences* 8, 5153–5165.
    DOI:10.1007/s12517-014-1511-y. — **The direct precedent for our leakance
    term.** Adds a third Muskingum parameter for lateral outflow = channel
    transmission loss (bed seepage to aquifer), subtracted from the routing
    balance — structurally identical to ddrs subtracting `zeta` from `b`.
    Establishes channel-aquifer loss *inside* a Muskingum scheme as published,
    defensible practice.

**Takeaway:** the matrix-Muskingum core is RAPID (David 2011/2015), the geometry
is Cunge/Ponce-Yevjevich, and subtracting a channel-loss term from the RHS is
not novel — Elfeki et al. (2015) formalized exactly this. The Ponce-Yevjevich
caveat applies: the loss is an intentional, quantified departure from the
conservative core (which is why our `|zeta|` eval diagnostic exists).

## 4. Differentiable modeling — and how gradients actually work here

12. **Shen (2018)**, "A Transdisciplinary Review of Deep Learning Research …,"
    *WRR* 54(11), 8558–8593. DOI:10.1029/2018WR022643. — Framing review;
    equifinality, regionalization, scaling — the problems dPL was built to solve.
13. **Tsai, Feng, Pan, Beck, Lawson, Yang, Liu & Shen (2021)**, "From
    calibration to parameter learning …," *Nature Communications* 12, 5988.
    DOI:10.1038/s41467-021-26107-z. — The canonical **differentiable parameter
    learning (dPL)** paper: an NN learns a global attributes→parameters map,
    trained end-to-end against observations. Direct conceptual ancestor of the
    ddrs KAN head.
14. **Feng, Liu, Lawson & Shen (2022)**, "Differentiable, Learnable,
    Regionalized Process-Based Models …," *WRR* 58(10), e2022WR032404.
    DOI:10.1029/2022WR032404. — Differentiable process model with regionalized
    NN parameterization approaches LSTM accuracy while keeping physical outputs —
    the "NN inside physics, not replacing it" thesis ddrs inherits.
15. **Shen, Appling, Gentine, … Lawson (2023)**, "Differentiable modelling to
    unify machine learning and physical models for geosciences," *Nature Reviews
    Earth & Environment* 4(8), 552–567. DOI:10.1038/s43017-023-00450-9. — The
    definitional reference for "differentiable modeling"; cite for what the
    paradigm means.
16. **Bindas, Tsai, Liu, Rahmani, Feng, Bian, Lawson & Shen (2024)**, "Improving
    River Routing Using a Differentiable Muskingum-Cunge Model and
    Physics-Informed Machine Learning," *WRR* 60(1), e2023WR035337.
    DOI:10.1029/2023WR035337. — **The direct DDR/ddrs predecessor.** NN infers
    Manning's n + channel geometry from attributes into a differentiable MC
    solver, trained on downstream hydrographs. Its synthetic experiments found
    **channel geometry unidentifiable and n only moderately recoverable** — the
    exact identifiability subtlety our leakance recoverability control
    re-encountered a layer deeper.
17. **Rackauckas, Ma, Martensen, … Edelman (2020)**, "Universal Differential
    Equations for Scientific Machine Learning," *arXiv:2001.04385* (preprint).
    — SciML theory for embedding NNs in ODE/PDE solvers and training via
    reverse-mode AD / adjoint sensitivity; the general justification for our
    hand-written O(nnz) sparse backward through the routing solve.

### How gradients work in ddrs (grounded in the above)

**Differentiable parameter learning (Tsai 2021).** A KAN head maps reach
attributes → per-reach physical parameters (Manning's n; leakance K_D, d_gw,
factor). These parameters are *not* the prediction — they are inputs to the
Muskingum-Cunge physics (Bindas 2024). The solver routes runoff to gauge
discharge; an L1/KGE loss compares against USGS observations; reverse-mode
autodiff propagates ∂Loss/∂discharge back *through the physics* into the KAN
weights. Gradients flow through both halves end-to-end (Shen 2023).

**Why the solver must be differentiable.** Each timestep is a sparse
lower-triangular forward-substitution solve. Naive autograd would tape O(n²)
operations; ddrs instead supplies a hand-written adjoint (`CsrSolveOp`,
`TimestepLeakanceOp`) — the reverse-mode transpose-solve of the linear system —
giving O(nnz) tape entries per step (the UDE/adjoint discipline of
Rackauckas 2020; ddrs invariant #4). "Gradient-exact" means this analytical
backward reproduces the true Jacobian-vector product: ∂Loss/∂factor matches
finite differences to machine precision. Our gradient probe confirmed the
signal **reaches essentially every reach**.

**The subtlety our experiments hit — and why it is not a bug.** Gradient
exactness is necessary but *not sufficient* for learnability. A parameter is
recoverable only if its influence on the loss exceeds the objective's **noise
floor**. Two floors bury the leakance signal: (a) gauge observation uncertainty
(§5), and (b) hotstart transients in windowed training (our recoverability
finding — a ~130× loss floor from rho-90/warmup-5 windows started off developed
storage). So "the gradient reaches the reach" (probe: confirmed) does **not**
imply "the parameter is recoverable from gauge loss" (recoverability control:
refuted). This is the differentiable-modeling analog of equifinality —
precisely the regime Bindas (2024) already flagged for channel geometry, and it
is a property of the *objective*, not the autodiff.

## 5. Losing streams are real and widespread; reach-scale loss is below the gauge floor

18. **Jasechko, Seybold, Perrone, Fan & Kirchner (2021)**, "Widespread
    potential loss of streamflow into underlying aquifers across the USA,"
    *Nature* 591, 391–395. DOI:10.1038/s41586-021-03311-x. — ~4.2M wells;
    **~64% sit below adjacent stream stage** → downward seepage wherever the bed
    is permeable. Strongest large-scale evidence that losing streams are real
    and common, and a mappable well-vs-stream head prior. (This is the
    "Jasechko-style" signal our findings docs already name as the auxiliary
    target — now with the exact citation.)
19. **Fan, Li & Miguez-Macho (2013)**, "Global Patterns of Groundwater Table
    Depth," *Science* 339(6122), 940–943. DOI:10.1126/science.1229881. —
    Continuous global water-table-depth map from >1M wells + model. A directly
    usable spatial prior for `d_gw` — an independent groundwater field, exactly
    what Condon & Maxwell (2015) says topography cannot supply.
20. **McCallum, Cook, Berhane, Rumpf & McMahon (2012)**, "Quantifying
    groundwater flows to streams using differential flow gaugings and water
    chemistry," *J. Hydrology* 416–417, 118–132.
    DOI:10.1016/j.jhydrol.2011.11.040. — Differential gauging resolves only
    *net* exchange, and the difference of two large discharges carries
    uncertainty that swamps small exchange — the authors add tracer chemistry
    *specifically because* gauging alone can't separate the flux. **Empirical
    grounding for our detectability NO-GO** (P3 gradient-probe result).
21. **Shanafield & Cook (2014)**, "Transmission losses, infiltration and
    groundwater recharge through ephemeral and intermittent streambeds: A review
    of applied methods," *J. Hydrology* 511, 518–529.
    DOI:10.1016/j.jhydrol.2014.01.068. — Review; discharge-based loss estimates
    are limited by measurement uncertainty, point methods don't scale — no
    single measurement class cleanly resolves reach-scale loss.
22. **Sauer & Meyer (1992)**, "Determination of Error in Individual Discharge
    Measurements," USGS Open-File Report 92-144. — The MEASERR decomposition of
    current-meter discharge error; source of the **~5% per-gauging uncertainty**
    band a ~0.01 m³/s reach loss falls 1–2 orders of magnitude beneath. This is
    the literature anchor for the exact band our detectability probe used.

**Takeaway:** both of our load-bearing claims are backed. Losing streams are
real and widespread (Jasechko 2021; Shanafield & Cook 2014) with a mappable
independent prior (Fan 2013), AND gauge discharge cannot resolve individual-reach
loss (McCallum 2012; Sauer & Meyer 1992) — so auxiliary spatial supervision,
not discharge, is the identifying signal.

---

## What this means for our findings (synthesis)

1. **Our functional form is defensible** — it is the MODFLOW RIV/SFR Darcy
   conductance (Harbaugh 2005; Niswonger & Prudic 2005) and its
   subtract-from-routing-RHS placement is published Muskingum practice
   (Elfeki 2015). The matrix-Muskingum core is RAPID (David 2011/2015).
2. **Two caveats we should add to the leakance docs:** the linear
   `(depth − d_gw)` should saturate under disconnection (Brunner 2009/2011), and
   lumped conductance is non-identifiable (Rushton 2007) — the latter
   independently predicts the K_D ceiling pinning we observed, and tempers the
   "widen K_D" follow-up: widening the range doesn't fix a structurally
   non-identifiable parameter, it just moves the ceiling.
3. **Our gradient understanding is correct and matches the paradigm** (Tsai 2021;
   Shen 2023; Rackauckas 2020; Bindas 2024): exact reverse-mode gradients through
   the differentiable solver, with learnability gated by the objective's noise
   floor — the equifinality regime Bindas (2024) already reported for channel
   geometry. Our probe/recoverability results are a sharper, quantified instance
   of a known phenomenon, not an anomaly.
4. **The auxiliary-supervision recommendation is now literature-backed, not just
   run-backed.** The detectability floor is real (McCallum 2012; Sauer &
   Meyer 1992); the losing-stream signal is real and independently mappable
   (Jasechko 2021; Fan 2013); and a continental, physically-consistent
   supervision target exists (Yang, Condon & Maxwell 2025 GW–SW flux;
   Condon & Maxwell 2015 water-table depth). The next experiment's auxiliary
   term should supervise `d_gw`/`zeta_net` against one of these fields, injected
   outside the gauge-discharge loss.

Confidence note: all 26 citations web-verified (title+authors+venue+DOI);
Brunner 2011 author-ordering is MEDIUM; Rackauckas 2020 is a widely-cited
preprint, never formally journal-published. No citation here is unverified —
anything the research pass could not confirm was dropped rather than guessed.
