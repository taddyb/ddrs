# Giving the width exponent a Leopold & Maddock prior

**Status:** design, not implemented. Written 2026-09-12.
**Motivated by:** `docs/2026-09-08-landscape-hypothesis-tests-findings.md` §30 (what daily
discharge can identify), §32 (the head is capable; the trunk collapses to rank 1.38),
and `.claude/skills/ddrs-eval-plots/references/channel_geometry.md` (the width law
cannot reach L&M's exponent inside the declared range).

---

## 1. The problem, stated exactly

`q_spatial` is the exponent in the **width-depth** law `w = p · d^q`. Leopold &
Maddock's `b ≈ 0.50` is the exponent in the **width-discharge** law `w ∝ Q^b`. They
are not the same number. Substituting the depth solution
`d ∝ Q^(3/(5+3q))` into `w = p·d^q`, at constant `p`:

```
        3q                                   5b
  b = --------            <=>        q = ----------
      5 + 3q                             3 (1 - b)
```

Which gives the mapping that decides everything below:

```
    q      0.00   0.08   0.30   0.50   1.00   1.67   3.00   4.00
    b      0.00   0.05   0.15   0.23   0.375  0.50   0.64   0.71
                    ^                    ^      ^
                    |                    |      L&M b = 0.50
                    |                    q's CURRENT UPPER BOUND
                    trained median q = 0.084
```

Three consequences, and each is a separate problem:

1. **The declared range cannot express plausible geometry.**
   `parameter_ranges.q_spatial: [0.0, 1.0]` caps the achievable width exponent at
   `b = 0.375`, below L&M's 0.50. Reaching 0.50 needs `q = 5/3`, outside the box.
   This was already established in `channel_geometry.md` and never acted on.

2. **The trained model sits at `b ≈ 0.05`, not merely low but nearly rectangular.**
   The 500-update `nse-batch` model has median `q = 0.0843`, so width grows as
   `Q^0.05`: essentially constant width from headwater to main stem. The older
   `2026-08-03` run fitted `b = 0.226` with `p` learnable; the current model has `p`
   pinned at 21, so it has no second channel to make up the difference.

3. **We have been scoring this against the wrong band.** `LM_BAND = (0.1, 0.6)`
   applied to `q` (introduced in `experiments/head_arch/compare_arms.py`, and used in
   this session's reporting of the derivative-loss model as "Leopold-Maddock band
   0.1-0.6, 33.0% -> 9.3%") treats `q` as if it were `b`. In `b` terms that band is
   `[0.057, 0.257]`, so it was rewarding the model for being in the wrong place. It
   must be corrected before any arm is scored with it.

---

## 2. What this change is and is not for

**It does not fix the rank-1 collapse, and must not be sold as doing so.** §32 measured
the trunk collapsing from effective rank 4.36 to 1.38 in the first epoch, so `q` will
remain a stretched copy of `n` whatever range it is given. These are two independent
defects that happen to meet in the same variable:

| defect | evidence | what fixes it |
|---|---|---|
| `q` carries no independent information | trunk rank 1.38, affine R² 0.9924 (§32) | an observation that sees width, or an objective with leverage on shape (§30) |
| `q` cannot express plausible geometry even in principle | `b` capped at 0.375 by the box | **this design** |

The benefit is **physical defensibility and falsifiability**, not skill. §20 bounds the
whole channel-parameter question at about 0.09 NSE, and §24 showed the NSE optimum
costs no KGE, so the honest prediction is that skill barely moves. What changes is that
the learned quantity becomes a number a hydrologist can argue with, and one that
remotely sensed width (SWOT, Landsat) could falsify. §30 named exactly that as the
thing that would identify `q`.

---

## 3. The design

### 3.1 Phase 1 is one config line, and it does most of the work

```yaml
params:
  parameter_ranges:
    q_spatial: [0.3, 3.0]     # was [0.0, 1.0]
```

The sigmoid is centred at its midpoint, so this **also sets the prior**. Midpoint
`q = 1.65` gives `b = 0.497`. An untrained head therefore starts at Leopold & Maddock
instead of at an arbitrary point, and departures from it are what training has to earn.

```
  CURRENT  q in [0, 1]                   PROPOSED  q in [0.3, 3.0]

  b  0.375 +---------------+ q=1         b  0.71 +----------------+ q=4
           |               |                     |                |
           |         .     |                     |                |
  L&M 0.50 - - - - - - - - - -  unreachable      0.50 - - -*- - - - -  <- sigmoid centre
           |               |                     |                |
           |    *  trained |                     |                |
    0.05   |    (q=0.084)  |                0.15 +----------------+ q=0.3
       0.0 +---------------+ q=0
```

No Rust changes at all. `denormalize` (`src/routing/utils.rs:32`) already maps the
sigmoid onto `[lo, hi]`, and `q_spatial` is not in `log_space_parameters`.

### 3.2 Phase 0 gates Phase 1, and is free

Widening `q` changes channel geometry non-obviously, because baseflow depth is below
1 m over most of CONUS and `d^q` therefore *shrinks* as `q` grows. At the trained
median depth of 0.25 m:

```
  q = 0.084 :  d^q = 0.890  ->  w = p·d^q = 18.7 m   (p = 21)
  q = 1.65  :  d^q = 0.101  ->  w =          2.1 m
```

A 2 m wide main stem is not plausible, so **the range cannot be chosen without
checking the geometry it implies**. Depth also moves, since its exponent is
`3/(5+3q)`, and the two effects partly cancel; the net has to be computed, not argued.

Phase 0 is a pure-numpy sweep over `q` using the existing notebook template in
`channel_geometry.md`, on the real trained `n`, `p` and post-clamp slope, reporting
width, depth and `w:d` percentiles plus the fitted `b` at each candidate range. It
costs minutes and it decides the numbers in Phase 1. **If no range gives both
`b ≈ 0.5` and plausible widths, the width law itself is the problem and Phase 1 is
abandoned rather than tuned.** That is the real decision gate.

### 3.3 Phase 2, only if Phase 0 says the box is not enough

If a simple range cannot centre the prior without distorting the geometry, add an
explicit prior instead of relying on the sigmoid midpoint: keep a wide box for
expressiveness and add a penalty pulling the network-scale fitted `b` toward 0.5.
That is a training-loss change, not a geometry change, and it is deliberately deferred
until Phase 0 shows it is needed.

---

## 4. Blast radius

| file | change | risk |
|---|---|---|
| `config/experiments/*.yaml` | one range line | none, config only |
| `experiments/head_arch/compare_arms.py` | correct `LM_BAND`; report `b`, not raw `q` | none, analysis only |
| `.claude/skills/ddrs-eval-plots/references/channel_geometry.md` | record the corrected band and the `q <-> b` table | none, docs |
| `src/routing/utils.rs`, `src/geometry.rs` | **no change in Phase 1** | — |
| `src/training/loss.rs` | Phase 2 only, new optional penalty term | autograd touched; needs a gradcheck |

**Invariant 1 is safe in Phase 1**: `compare_ddr_sandbox` uses its own fixture config,
so a range change in the CONUS experiment configs cannot move it. Phase 2 would add an
opt-in loss term, which is a drop-in scalar on the routed predictions and leaves the
sparse backward alone (as `nse-batch-deriv` already did).

**Checkpoints do not transfer.** Every learned parameter was fitted against the old
box, so a range change invalidates all existing heads for this parameter. Retraining is
mandatory, not optional.

---

## 5. Concerns

1. **Skill may fall.** Wider, shallower channels change hydraulic radius, hence
   velocity, hence celerity and the whole routing timescale. This is the one change here
   that touches the physics the model is scored on. Mitigation: Phase 0 predicts the
   geometry before any training; the arm is compared against a matched control on the
   same population.
2. **`side_slope` clamping may start binding.** `clamp(w·q/(2d), 0.5, 50)` grows with
   `q`. If a large share of reaches pin at 50 the trapezoid degenerates and the
   geometry is no longer what the equations say. Phase 0 must report the clamp share.
3. **This deviates from DDR.** DDR uses `q ∈ [0, 1]`. The port stays gradient-exact,
   but a CONUS model trained on a different box is no longer parameter-comparable to
   DDR's. Flag in the run manifest and the findings doc.
4. **`b = 3q/(5+3q)` assumes `p` is constant.** With `p` learnable (which the four arms
   currently training all are) the network-scale exponent becomes `b = β + 3q/(5+3q)`,
   where `β` is how `p` itself scales with discharge. The clean test therefore holds
   `p` fixed at 21. Running it with `p` learnable confounds the two.
5. **L&M's `b ≈ 0.5` is not universal.** It varies with region, climate and channel
   type. Treat 0.5 as a central prior to be departed from, never as a target to be hit,
   and report the spread rather than the mean alone.
6. **It will not make `q` identifiable.** See §2. If we retrain and `q` still tracks
   `n` at ρ > 0.95, that is the expected outcome and not a failure of this change.

---

## 6. Assumptions

1. **Baseflow specific discharge `Q_SPEC = 0.005 m³/s/km²`** for the geometry sweep,
   per `channel_geometry.md`. The fitted exponents are invariant to it (it is a
   constant inside a log-log fit), so only absolute widths and depths depend on the
   choice. Sweeping it 0.002 / 0.005 / 0.01 is the built-in check that the fit is
   wired correctly.
2. **The post-clamp `slope` variable is the right slope**, not the raw fabric slope.
   33.2% of CONUS reaches are pinned at `attribute_minimums.slope`.
3. **`p` held at 21** for the clean arm, per concern 4.
4. **Downstream, not at-a-station, hydraulic geometry** is the relevant comparison,
   since the fit runs across reaches in a network rather than through time at one site.

---

## 7. Success criteria

Run in order; each gates the next.

1. **Phase 0** -> a range exists giving fitted `b ∈ [0.45, 0.55]` with median width and
   depth inside the plausibility bands in `channel_geometry.md` and `side_slope`
   clamping below 5% of reaches. **If not, stop and report that the width law, not the
   box, is the binding constraint.**
2. **Phase 1** -> retrain on that range with `p` fixed, matched control on the same
   population. Fitted `b` moves from ~0.05 toward 0.5, and median NSE changes by less
   than 0.02 against the control.
3. **Registered prediction** -> `q` still tracks `n` at ρ > 0.95 and the trunk still
   collapses below rank 2. If either fails, §32's account of the collapse is wrong and
   that is the more important result.

---

## 8. What this is worth

It converts `q_spatial` from an unfalsifiable number between 0 and 1 into a stated
departure from a literature relation, in a range that can actually express that
relation. It does not make the parameter identifiable from discharge, and it should not
be described as doing so. For the paper it turns "the width exponent is unconstrained"
into the sharper and more useful "the width exponent is unconstrained, and the
parameterisation could not have reached the physically expected value even if it were".
