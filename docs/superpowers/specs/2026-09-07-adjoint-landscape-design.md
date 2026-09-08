# Per-gauge loss landscape in channel-parameter space via the adjoint — Design

**Date:** 2026-09-07
**Status:** approved in conversation (training loss for now; other losses later). **Sample case run 2026-09-07:** `docs/2026-09-07-landscape-uh-juniata-findings.md` (instrument PASS; box must be bounded by the parameter ranges, see §5).
**Study name:** `landscape` (`ddrs experiment <bundle>` with `study: landscape`)
**Depends on:** `src/experiment/` framework, `adjoint::influence::InfluenceContext` (dataset/head/collate),
the seed-43 replicate (`config/experiments/uh_retro_seed43.yaml`) for the tolerance.

## 0. Purpose

**Hypothesis (user, 2026-09-07).** Each gauge's loss under large-batch training (256 to 2,000 gauges per
update) is not at that gauge's own minimum; it is the compromise the batch reaches. Taking a trained model (or an
untrained one) and mapping, via the adjoint, the loss at each gauge over Manning's `n`, `p`, and `q` as the three
axes tells us whether we are training the model correctly: how far the batch solution sits from the gauge's own
optimum, and along which directions. The same maps computed with different inflow inputs (arms) show whether the
inputs change where the gauge's optimum is or only where the batch solution lands.

For each gauge, determine which combinations of the channel parameters (Manning's `n`, Leopold–Maddock
`p`, `q`) produce near-optimal routed discharge at that gauge, and locate where the values learned by the
large-scale (all-gauge, attribute-conditioned KAN) training sit relative to that set. This is GLUE's
behavioural set (Beven & Binley, 1992) computed per gauge with gradients instead of Monte Carlo, and it
is the sharp form of the abstract's equifinality question: the directions the gauge constrains (stiff)
should agree across inflow-source arms; the directions it does not (sloppy) are free to differ.

## 1. Parametrization (Phase A: basin-uniform log-multipliers)

Let `n₀, p₀, q₀` be the trained physical per-reach fields over the gauge's subgraph (head output,
denormalized with the arm's `parameter_ranges`). Define three scalars `α = (α_n, α_p, α_q)` and
```
n_i = n₀_i · e^{α_n},   p_i = p₀_i · e^{α_p},   q_i = q₀_i · e^{α_q},
```
each clamped to the arm's parameter range before re-normalization (`physical_to_normalized`, inverse of
`routing::utils::denormalize`, including the `+1e-6` epsilon on log-space `p`). The fraction of reaches
clamped at a range edge is recorded per evaluation; landscape values where > 5 % of reaches are clamped are
flagged. `α = 0` is the trained point. Domain: `α ∈ [−ln 3, ln 3]³` (factor 3 down/up).

Phase B (deferred): per-reach directions via randomized Lanczos on the 3N-dimensional Gauss–Newton
Hessian (Hessian-vector products = one tangent FD forward + one weighted backward).

## 2. Objective

The training objective restricted to gauge `g`: dHBV's NSE-batch loss over one 90-day window
(`experiment.loss.kind: nse-batch`, `src/training/loss.rs::nse_batch_loss`),
```
L_g(α) = mean_{valid days d ≥ warmup} (Q̄_g(d; α) − O_g(d))² / (σ_g + ε)²,
```
`Q̄` daily-pooled under the training `tau` convention, `σ_g` the training-period observed std
(`MeritGagesDataset::gauge_obs_std`), `ε = 0.1`. Windows: the four WY2000 seasonal windows already used
by the adjoint study; `L_g` is their mean (4 forwards per evaluation). Reported alongside: NSE of the
same series, for readability.

## 3. Quantities computed per gauge × arm

1. **Gradient** `∂L_g/∂α` at any `α`: one backward with `α` lifted as three scalar leaves (the fields
   are tensor functions of `α`, so autograd carries the multiplier through the head's normalized
   outputs into the engine). Cost: 4 forwards + 4 backwards (one per window).
2. **Hessian** `H(α)` (3×3): central finite differences of the gradient, `h = 0.05` in log space
   (6 gradient evaluations), symmetrized. Eigen-decomposition → eigenvalues `λ₁ ≥ λ₂ ≥ λ₃` and
   eigenvectors `v_k` (unit vectors in `(α_n, α_p, α_q)`), i.e. the stiff and sloppy combinations
   `n^{v_k[0]} p^{v_k[1]} q^{v_k[2]}`.
3. **Per-gauge optimum** `α*`: damped Newton from `α = 0` (`H + μI`, backtracking on `L_g`, ≤ 15
   iterations, stop at `‖Δα‖ < 1e-3`), then `H(α*)`, `L_g(α*)`, `NSE(α*)`.
4. **Behavioural set**: at `α*`, half-widths along each eigenvector `w_k = sqrt(2 ε_L / λ_k)` for a loss
   tolerance `ε_L` (quadratic approximation), and the trained point's coordinates in the eigenbasis,
   `c_k = v_kᵀ(0 − α*)`. `|c_k| / w_k` says whether the large-scale solution lies inside the
   behavioural set along axis `k`. Tolerance: `ε_L` = the seed-to-seed loss difference at that gauge
   (from the seed-42/43 pair) once available; until then report `ε_L ∈ {0.05, 0.10}·L_g(α*)`.
5. **Landscape slices**: three 2-D grids through `α*`, 11×11 over `[−ln 3, ln 3]`, in the `(α_n, α_p)`,
   `(α_n, α_q)`, `(α_p, α_q)` planes (forward-only, 363 evaluations × 4 windows), plus one 11×11 grid in
   the plane of `(v₁, v₃)` (stiffest, sloppiest). Each grid cell stores `L_g`, `NSE`, clamped fraction.
6. **Trained-point diagnostics**: `L_g(0) − L_g(α*)` (how much the gauge would gain from its own
   optimum), `‖α*‖`, and the per-parameter multipliers `e^{α*}`.

Cost per gauge × arm ≈ 484 forward-only + ~130 forward+backward evaluations; with 4 windows and the
Juniata network (~0.4 s per forward) ≈ 10 min; the 520-reach White River ≈ 1 h. Arms run in threads as
in the adjoint study.

## 4. Sample case (this PR)

Bundle `experiments/landscape-uh-juniata`: UH arm (seed 42), gauges 01563500 and 01567000, Phase A
quantities 1–6, single arm. Figures (`experiments/landscape/plots.py`): the three slices per gauge as
filled contours of `NSE` with the trained point (×), the per-gauge optimum (★), the eigenvectors of
`H(α*)` as arrows, and the behavioural ellipse at the two tolerances; a table of `α*`, `e^{α*}`,
eigenvalues/eigenvectors, and `|c_k|/w_k`. Once the seed-43 arm exists the same bundle adds it as a
second arm and the seed-to-seed distance in `(v₁, v₂, v₃)` coordinates becomes the noise floor.

**Pass criteria for the sample case** (instrument, not science): (i) the FD Hessian is symmetric to
< 5 % and positive semi-definite at `α*`; (ii) the gradient at `α*` has norm < 1e-3 of its norm at
`α = 0`; (iii) the leading eigenvector is dominated by the combination that changes celerity (checked
against the analytic dependence of `K = L/c` on `n`, `p`, `q` at the window-mean discharge, computed
with `hydraulics::reach_k_hours`); (iv) `L_g(α*) ≤ L_g(0)`.

## 5. Concerns

- **Clamp kinks.** Check 1 showed the model is non-differentiable at flood peaks and dry transitions;
  the landscape will have creases. The grids see them; the Hessian at `α*` may not. Report the clamped
  fraction and inspect grids before trusting eigenvalues.
- **Basin-uniform multipliers are a projection.** A gauge may be able to trade `n` upstream against
  `p` downstream in ways three scalars cannot express; Phase A finds the projected behavioural set,
  which is a subset. Phase B addresses this.
- **One window family.** WY2000 only; the landscape is state-dependent (celerity depends on flow).
  Same caveat as the anchors; the seasonal mean spans the state range of one year.
- **NSE-batch normalises by observed variance**, so gauges with tiny `σ_g` (dry Plains) have steep,
  noisy landscapes. Mask gauges with `σ_g < 1 m³/s` as in the adjoint study.

## 6. Out of scope

Other losses (KGE-based) — later, by user decision. Per-reach (Phase B) — after the sample case.
Cross-arm population run — after the seed replicate defines the tolerance.

## 7. Tests of the hypothesis (added 2026-09-08)

The hypothesis in §0 makes three claims the sample case (§4, one gauge pair) did not test. Each is one bundle.

**7.1 Displacement census** (`experiments/landscape-uh-census`, UH seeds 42 and 43, the eight validation
gauges, full slices). For each gauge: the gain available to the gauge from its own optimum, `NSE(α*) − NSE(0)`;
the trained point's stiff coordinate `c₁/w₁`; the physical optimum `(n*, p*, q*)`. The discriminating
statistic is the **direction** of `α*` across gauges. If every gauge wants the same move (the sample said
"everything smaller, n × 0.36" at both Juniata gauges), the batch solution is not a compromise between gauges
but a systematic offset, which points at training (learning rate, range floors, the σ-normalised batch loss
weighting some gauges over others) rather than at equifinality. If the directions scatter, the batch solution
is a genuine compromise and the per-gauge gains are the cost of sharing one head. The seed pair gives the
noise on each gauge's `α*`.

**7.2 Training trajectory** (`experiments/landscape-uh-trajectory`, UH seed 42 at init, epoch 1, 5, 10, 20,
30; no slices). "Are we training correctly" in its direct form: at each checkpoint, each gauge's stiff
coordinate `c₁` and loss `L_g`. With the trained fields stored per checkpoint, all coordinates are expressed
in the epoch-30 physical frame. Correct training moves `c₁` to zero for every gauge and leaves the sloppy
coordinates wherever the shared head happens to put them; a gauge whose `c₁` stalls away from zero, or whose
optimum sits at a range floor, is a gauge the batch cannot fit with this parametrisation.

**7.3 Inputs** (`experiments/landscape-arms`, the five inflow arms, eight gauges, no slices). For each gauge,
the physical optimum and `NSE(α*)` per arm. If the optimum in physical `(n, p, q)` moves with the input, the
channel parameters are absorbing inflow bias (the channel compensates for the runoff product), and the
cross-arm celerity spread measured by the adjoint study is that compensation. If the physical optimum is
the same across arms and only the batch solution moves, the gauge's constraint is input-independent and the
arms' disagreement is a training artefact the census (7.1) should also show.

Prerequisites (implemented with this section): Newton bounded by the clamped fraction (`max_clamped`),
per-arm `checkpoint:` override including `init`, and `grid: 0` to skip slices. Cost on cpu: 7.1 about 2 h,
7.2 and 7.3 about 1 h each with slices off.

**Concerns.** (i) The basin-uniform multipliers cannot express a per-reach compromise; a gauge that "wants
everything smaller" may want only its headwaters smaller. Phase B remains the fix. (ii) The eight gauges
include three Northern Plains gauges with intermittent inflow, where the landscape is state-dependent and
creased; their `α*` may be unstable between seeds. (iii) `init` is the head at its seed initialisation, not
a physically meaningful channel; its landscape is a reference for the trajectory only.
