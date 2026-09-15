# Where our stage-roughness work sits against the SMASH/INRAE lineage (2026-09-14)

Prompted by a manuscript paragraph citing Jay-Allemand et al. (2020), Garambois et al. (2025),
Huynh et al. (2024), Ettalbi et al. (2025), and Huynh et al. (2025, 2026) as the "advances in
addressing computational challenges" for fully distributed physics-based models. That cluster is
one research programme (SMASH, INRAE Aix-en-Provence) and it is the closest prior art to this
work. Every quote below was verified against the publisher's page, not a secondary source.

## What each paper actually does

| paper | model and domain | what is learned | how the gradient is obtained |
|---|---|---|---|
| Jay-Allemand 2020, HESS 24, 5519 | GRD, 3 params per 1 km cell, Gardon d'Anduze 540 km2, 5 gauges, hourly | 1,620 distributed parameters direct, Tikhonov prior to a uniform solution | Tapenade source-to-source adjoint, hand-optimised, gradient-tested |
| Garambois 2025, J. Hydrol. Reg. Stud. 60, 102485 | GR4-like gridded, 235 French catchments, 1 km, hourly | distributed calibration plus parametric sensitivity | same platform adjoint |
| Huynh 2024, WRR 60, e2024WR037544 | SMASH, 126 Mediterranean catchments | ANN maps 7 descriptors to cp, ct, kexc, llr. **Routing excluded** | adjoint chained analytically through the ANN |
| Ettalbi 2025, J. Hydrol. 660, 133300 | SMASH, southern France | same regionalisation, plus satellite soil-moisture assimilation | as above |
| Huynh 2025, HESS 29, 3589 | SMASH, 235 + 21 catchments | two nets: flux corrections f_q,1..4, and regionalisation of cp, ct, kexc, **a_kw, b_kw** | Tapenade adjoint, both nets trained jointly |
| Huynh 2026, GMD 19, 1055 | SMASH as a universal differential equation, Aude basin, 25 gauges | state-dependent net inside the ODE right-hand side, plus regionalisation net | implicit Euler with analytic Jacobians, adjoint preserved |

**A mis-citation to flag.** The paragraph attributes "spatiotemporal flux corrections within
kinematic wave routing structures" to Huynh 2025, 2026. The corrections are not in the routing.
In Huynh 2025 they are applied to infiltration, actual evapotranspiration, the slow/fast transfer
partition, and the non-conservative exchange flux, all inside the GR-like production and transfer
reservoirs. The kinematic-wave parameters appear only in the regionalisation vector: "The vector
of conceptual spatialized parameters, mapped by phi_2, is theta = (cp, ct, kexc, a_kw, b_kw)".
Nobody in this lineage has put a learned correction inside the routing operator itself. We have.

## Three ways their results corroborate ours

1. **Routing parameters are the least sensitive of the control vector.** Garambois 2025, over 235
   catchments: "water balance parameters, especially production capacity (cp) and exchange (kex),
   are most sensitive, while routing parameters are less so." That is a national-scale, independent
   confirmation of what our per-gauge landscapes measure at the level of individual gauges.
2. **Parameter fields track the data and the structure, not the catchment.** Jay-Allemand 2020 on
   fields differing between calibration subperiods: "This clearly indicates that the corresponding
   estimates are determined by the observed data used in each case." Huynh 2026 reports that
   different network architectures and model structures yield notably different parameter patterns.
   Our five-arm result is the same phenomenon isolated on one variable: hold network, gauges,
   observations and budget fixed, change only the inflow product, and the roughness field moves.
3. **The compensation worry is live in peer review, and unanswered.** The HESS 2025 reviewers asked
   whether small skill gains from per-pixel, per-timestep corrections reflect "compensatory
   adjustments rather than genuine process learning". The authors answered by inspecting patterns,
   with no identifiability test. Huynh 2026 states that testing whether the learned corrections are
   physically real is future work, to be done against satellite water-budget products.

## What is ours and not theirs

- **A controlled perturbation instead of a plausibility argument.** Their evidence for physical
  meaning is that the learned fields look coherent. Ours is an experiment: five inflow products
  through one fixed routing model. The learned roughness field is a fingerprint of the product, and
  what it absorbs is product-wide timing, not gauge-wise error.
- **A curvature instrument at the gauge.** Per-gauge (n_0, gamma) loss surfaces with pre-registered
  bars give a number, not an impression, and the number orders cleanly by the inflow's timing lead:
  0.008 lumped dHBV2, 0.018 hourly LSTM, 0.024 retrospective, 0.10 and 0.11 the daily LSTMs,
  0.229 distributed dHBV2. The stage exponent becomes visible only on the product that needed no
  timing repair.
- **A learned correction inside the routing operator, at continental scale.** 346,321 MERIT reaches
  on a vectorised river network, against their 1 km gridded French domains. Hand-written sparse
  backward, O(nnz) tape entries per timestep, rather than source-to-source AD of Fortran.

## Two things to borrow

1. **Say that fixing the channel geometry is choosing a prior, in their words.** Jay-Allemand 2020,
   section 3.3: adding a regularisation term "makes the problem formally well-posed but, in
   practical terms, transforms the non-uniqueness issue into the issue of choosing the prior." Our
   decision to fix p = 21 and q = 0.65 at Leopold and Maddock values, taken because the landscape is
   flat and pushes to the bounds, is exactly that move. Framing it as a declared prior rather than a
   neutral simplification is both more honest and better supported.
2. **An independent observation is the way out, and Ettalbi 2025 is the template.** They break the
   ill-posedness by assimilating satellite soil moisture alongside discharge. The routing analogue
   is water-surface elevation and width from altimetry and SWOT, which constrain depth and width
   directly rather than through the discharge sum. This is the concrete recommendation our
   discussion and the talk's closing should carry.

## One tension to resolve honestly in the paper

Jay-Allemand 2020 found the routing velocity "the most stable among all estimated parameters",
while Garambois 2025 finds routing parameters the least sensitive, and we find the stage exponent
unidentifiable but the roughness level well determined. These are not contradictory, and the
resolution is worth a paragraph. In a three-parameter model on one 540 km2 catchment, a single
velocity is the only lever on timing, so discharge determines it sharply. In a richer model with
production and exchange terms, routing is a small lever by comparison. Our case adds a third
regime: when the inflow itself arrives early, the roughness level is determined sharply because it
is absorbing that lead, which makes it look identifiable while being a property of the product.
Stability, sensitivity, and physical meaning are three different claims, and this lineage plus our
result separates them.

## Actions

- [x] Bibliography entries added to `~/papers/ddr_equifinality/references.bib`
- [ ] Introduction topic 3 ("what has changed since Beven") to cite this lineage as the
      state of the art in gradient-based distributed calibration, with Castaings 2009 as ancestor
- [ ] Fixed-geometry justification to adopt the "choosing the prior" framing
- [ ] Discussion to carry the independent-observation recommendation
- [ ] Check the flux-correction mis-citation with whoever wrote that paragraph
