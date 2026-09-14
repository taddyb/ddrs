# ddr_equifinality Paper Scope — Design

**Date:** 2026-08-06
**Status:** Approved (brainstorming), pending user review of this spec
**Paper:** `/home/tbindas/papers/ddr_equifinality/paper.tex`
**Submitted AGU H069 abstract:** anchors this scope; already spliced into `paper.tex` (title, abstract, authors, Beven 2001 citation).

## Purpose

Scope the reframed `ddr_equifinality` paper so that a fresh writer knows what the
paper claims, what to keep from the existing draft, what to rewrite, and which
experiments and fixes must land before the Results can be written. This spec
covers **paper scope**; each prerequisite experiment gets its own findings doc in
`docs/` when it runs.

## 1. Contribution and framing

The paper is an **empirical answer to Beven (2001), "How far can we go in
distributed hydrological modelling?"**: can a distributed, differentiable routing
model overcome the *uncertainty* and *equifinality* obstacles Beven raised, using
tools he lacked (differentiable modeling, cloud-scale multi-source datasets)?

The payload is **normative**, engaging Beven's closing *Alternative Blueprint*
(future-proof modelling that gradually learns the idiosyncrasies of individual
places under acknowledged uncertainty): the answer becomes **criteria for judging
future differentiable models** — which learned parameters, and which places, can
be trusted versus which remain equifinal.

This is **not** a methods/protocol contribution. The input-perturbation design is
how the question is answered, not the novelty.

## 2. The two questions (open, non-directional)

Framed as open questions, not directional predictions. The paper does **not** bet
on the old "Manning's roughness is a bias absorber" mechanism (that direction is
unsupported in the group's own prior analysis).

- **Q-uncertainty:** Does trained routing *skill* converge across structurally
  different inflow sources — can the model absorb source-specific input bias?
- **Q-equifinality:** Do learned *channel parameters* (and their loss landscapes)
  settle on one consistent set across sources, or do some remain equifinal — which
  ones, and where?

## 3. Experiment matrix (full two-axis)

Same MERIT CONUS network, catchment attributes, USGS observations, and training
budget across all arms; swap **only** the lateral-inflow source. Two structural
axes:

- **Spatial axis:** lumped vs distributed dHBV runoff.
- **Temporal axis:** daily vs hourly LSTM forecasts (daily includes the
  precip-driven disaggregation variant as the bridge).

Parameter-convergence + skill analysis runs on **every** arm. Current state: the
temporal (LSTM) arms have full parameter analysis; the spatial (dHBV lumped/
distributed) arms have **routing-skill numbers only** — their parameter analysis
is prerequisite B2 below.

## 4. Analysis levels

| Level | What it measures | Answers |
|---|---|---|
| L1 routing skill | per-arm median NSE/KGE vs summed-inflow baseline | Q-uncertainty (does skill converge / bias get absorbed) |
| L2 raw parameters | cross-arm spread and correlation of learned n, p, q | Q-equifinality |
| L3 realized geometry | depth, top width, hydraulic radius at reference discharge, cross-arm | Q-equifinality (the physically interpretable quantities) |
| L4 loss-landscape / identifiability | which parameters are stiff vs sloppy around the trained optimum | Q-equifinality |

Every convergence number is reported against a **replicate-seed noise floor**
(prerequisite B3): a spread smaller than the seed-to-seed spread is not
convergence.

## 5. Keep vs rewrite in the existing draft

**Keep, lightly revise:**
- **Introduction** — repoint the thesis paragraph from "selective equifinality /
  bias absorber" to Beven's five challenges and the two open questions. The
  opening (routing, ungauged reaches, lookup-table parameters, National Water
  Model) stands.
- **Methods** — the differentiable Muskingum-Cunge routing math, the lateral
  inflow sources, the data, and the experimental-design subsections are reusable
  as-is. Update the inflow-source table to the final two-axis arm list. Replace
  the "Cross-Arm Convergence Analysis" framing to match the L1-L4 levels above
  and the noise-floor requirement.

**Rewrite:**
- **Results** — currently a skeleton. Populate with L1-L4 once the arms and the
  prerequisites land.
- **Discussion** — drop the bias-absorber "division of labor" narrative. Frame
  around the two questions and the judgment-criteria payload (what to trust,
  transfer, and interpret physically; which places).
- **Conclusion** — replace the "selective equifinality confirmed" arc with the
  honest answer to Beven plus the evaluation-criteria contribution.

## 6. Prerequisites that gate the Results

Intro and Methods can be written now. Results cannot be written until:

- **B1 — timezone / day-boundary alignment (THE GATE, do first).** USGS daily
  observations are midnight-to-midnight in **local standard time**; AORC forcing
  and likely the Q' stores are **UTC**. That is a 5-8 h, longitude-dependent,
  per-gauge offset. Verify how `src/data/dates.rs` and `src/data/dataset.rs` build
  daily windows and index each store. If the stores disagree on day convention,
  cross-source differences are confounded and every convergence number is suspect.
  Fix (reconcile all sources to one boundary) or, if already reconciled, document
  it as a common-mode caveat. Handoff: `/tmp/handoff-aorc-usgs-recording-times.md`.
- **B2 — spatial-axis parameter analysis.** Run L2-L4 on the lumped and
  distributed dHBV arms (checkpoints/parameter dumps exist; the analysis does
  not).
- **B3 — replicate seed.** At least one additional seed per arm, to put a noise
  floor under every convergence statistic. The group's evidence standard treats
  this as essential, not optional.
- **B4 — double-routing confound.** Resolve whether the distributed dHBV store is
  already pre-routed / pre-smoothed (open question from the AORC2F wave-1
  findings). If it is, lumped-vs-distributed differences are store-provenance
  artifacts, not model-structure effects. Fastest resolution: inspect the export
  script that generated the distributed store for whether it exports per-unit
  runoff or a routed/aggregated product.

## 7. Concerns and assumptions

- **The honest answer may be "partially."** The defensible result could be "skill
  converges, so the model largely absorbs input bias (uncertainty answered), but
  channel parameters only partly converge, and here is the noise floor
  (equifinality persists for some parameters/places)." The paper structure must
  make that a *satisfying answer to Beven*, feeding the judgment-criteria
  contribution, not read as a null result. **Assumption:** a nuanced, honest
  answer is publishable precisely because it is honest and it yields evaluation
  criteria.
- **Compute is substantial** — full two-axis matrix + replicate seed + reruns
  after the B1 fix. This is the schedule risk against the December AGU talk.
- **Risk:** if B4 cannot be resolved, the spatial axis weakens and the paper leans
  on the temporal axis, partly undercutting the full-matrix promise. Mitigation:
  the timezone gate and the parameter analysis are worth doing regardless; decide
  spatial-axis weight after B4.
- **Assumption:** authors are Bindas, Song, Shen (as now in `paper.tex`); author
  order and affiliations confirmed separately.

## 8. Out of scope

- Global scale-out (the global MERIT fabric) — a separate effort.
- Leakance / water-loss terms — closed NO-GO, ζ=0 in every arm.
- Any new routing-core or KAN-head development — the model is fixed; this paper
  measures what it learns.
