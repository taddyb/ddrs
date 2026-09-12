# Stage-dependent roughness: `n(d) = n₀ · (d / d_ref)^(−γ)`

**Status:** design, not implemented. Written 2026-09-12.
**Gate:** no learnable parameter is added until a landscape probe shows `γ` is
identifiable. See §5, which amends the "probe before any code" framing because a
probe needs the axis to exist in the forward model first.

**Background:** `docs/2026-09-08-landscape-hypothesis-tests-findings.md` §29
(the width exponent is concave at half of gauges), §30 (roughness is a *timing*
channel, width a *scale* channel), §32 (the trunk collapses to rank 1.38 under a
near one-dimensional gradient), and
`.claude/skills/ddrs-eval-plots/references/channel_geometry.md` (measured
downstream width exponent 0.226 against Leopold & Maddock's 0.50).

---

## 1. The problem

At one reach, `p`, `q`, `n` and slope are all constant in time. So as a flood
passes, the geometry response is not free: Manning plus continuity lock it onto a
one-parameter curve indexed by `q` alone.

```
  Manning (constant n):   m = (2/3) · f
  continuity:             b + f + m = 1
  =>                      b = 1 − (5/3) f        one degree of freedom
```

Substituting the model's own `f = 3/(5+3q)` returns `b = 3q/(5+3q)`, which is
exactly what `src/geometry.rs` implements. The formulation is already the general
solution of that constraint, which is why adding `b` and `f` as separate learnable
parameters would be redundant: they are two numbers pinned to a curve.

**The observed exponents are not on the curve.** For a 100x rise in discharge at
one cross section:

```
                          b       f       m        width   depth   velocity
  observed at-a-station  0.26    0.40    0.34      x3.3    x6.3    x4.8
  model curve, q=0.65    0.281   0.432   0.288     x2.9    x5.4    x6.4
  model curve, q=0.833   0.333   0.400   0.267     x4.6    x6.3    x5.8
  MODEL AS TRAINED       0.048   0.571   0.381     x1.25   x13.9   x5.8
   (q = 0.084)
```

No value of `q` reaches the observed triple. Match `f` and velocity is too fast;
match `b` and depth is too large. And the trained model is far outside either:
its floods rise almost vertically instead of spreading, then travel too fast
because depth drives hydraulic radius drives velocity.

**The one physical ingredient that can move `m` off `2f/3` at a fixed cross
section is roughness that changes with stage.** Real channels get smoother as
they fill, because relative roughness falls once the bed material is drowned.
The model has no such term: `n` is learned per reach and then held constant for
the whole simulation.

---

## 2. The proposal

```
  n(d) = n₀ · (d / d_ref)^(−γ)
```

`n₀` and `γ` are **static per-reach fields**, emitted once by the KAN exactly as
`n`, `p` and `q` are today. The stage dependence is applied in closed form inside
the timestep. The KAN is *not* given discharge as an input (see §7, concern 2).

### 2.1 The depth inversion stays closed form

This is what makes the change small. Starting from the relation the current code
already implements, `Q·n·(q+1)/(p√S) = d^((5+3q)/3)`, and substituting
`n = n₀(d/d_ref)^(−γ)`:

```
                                              3
                 Q · n₀ · d_ref^γ · (q+1)   -------
     depth  =  ( ------------------------ ) 5+3q+3γ
                        p · √S
```

At `γ = 0` this is byte-identical to the current formula: `d_ref^0` is exactly
`1.0` in f32 and the exponent reduces to `3/(5+3q)`. No iteration, no solver, one
extra factor and one extra term in an exponent.

### 2.2 The exponents it produces

```
        3                                              2 + 3γ
  f = ---------        b = q · f        m = 1 − b − f = ---------
      5+3q+3γ                                          5+3q+3γ
```

which is `m = (γ + 2/3) · f`: exactly Manning's `2f/3` shifted by `γ`.

**Solving for the observed at-a-station target** `b = 0.26, f = 0.40, m = 0.34`:

```
  q = b/f = 0.65          γ = 0.183
```

Those two values reproduce observed at-a-station hydraulic geometry exactly. The
current model is the `γ = 0` slice of this family, and that slice does not
contain the observation.

### 2.3 The celerity correction, which is the hard part

`c = dQ/dA` is no longer `v·β_trapezoid`, because `Q` now depends on `d` through
`n` as well as through the geometry. Deriving it exactly, with
`Q = (d/d_ref)^γ · Q_Manning(d)` and `dA/dd = T` for any cross section:

```
  c = v · [ β_trapezoid  +  γ · A / (T · d) ]
```

where `A` is cross-sectional area, `T` top width, `d` depth, and

```
  β_trapezoid = 5/3 − (4/3) · A·√(1+z²) / (T · P)     src/routing/mmc_op.rs:1160
```

Checks: the correction vanishes at `γ = 0`; and for the pure power-law section
`A/(T·d) = 1/(q+1)`, recovering `β = (5+3q+3γ)/(3(q+1))`, which at `q = 0, γ = 0`
is the wide-rectangular `5/3`. All three quantities are already computed in
`compute_trapezoidal_geometry`, so no new state is needed.

**This term must also be added to the hand-written backward.** See §6.

---

## 3. Why this parameter and not another

| | `q` (width exponent) | `γ` (stage-roughness) |
|---|---|---|
| information channel (§30) | **scale** (flashiness null at −0.045) | **timing** (flashiness +0.314, slope −0.435) |
| curvature vs `n` (§29/§30) | \|H_qq\| median **6.2 %** of \|H_nn\| | unknown, this is the gate |
| Hessian sign (§29) | **concave at 50–53 %** of gauges, a saddle | unknown |
| what it changes in the hydrograph | channel shape at a given flow | **how travel time varies between low and high flow** |

`γ` changes celerity's dependence on stage, so it changes when the peak arrives
relative to the recession. That is timing, which §30 established is the channel
daily discharge can actually see. That is the reason to expect it to be
identifiable where `q` was not, and it is a prediction to be tested, not assumed.

---

## 4. What this does NOT do

- **It does not fix the rank-1 collapse (§32).** A fourth output emitted by the
  same trunk will be as correlated with `n` as everything else is. `γ` is
  physically a roughness property, so correlation with `n` is expected and not
  by itself evidence of collapse.
- **It does not fix the downstream width exponent.** That needs `p` to grow with
  river size, which is a separate change
  (`2026-09-12-leopold-maddock-q-prior-design.md`).
- **It is not expected to move skill much.** §20 bounds the whole channel
  parameter question at about 0.09 NSE. The argument for this change is that it
  makes flood-wave timing physically right, not that it buys efficiency.

---

## 5. Staging, and an amendment to the gate

A landscape probe perturbs an axis, so the axis has to exist in the forward
model. "Probe before any code" is therefore not achievable literally. The
achievable and equally safe version is **probe before any LEARNABLE code**:

```
  Phase 1  forward model only. γ is a fixed scalar/field from config, NOT a KAN
           output. Includes the depth change, the celerity correction, and its
           gradcheck. Nothing learns it.
                    |
                    v
  Phase 2  LANDSCAPE PROBE with γ as an axis.  <-- THE GATE
           Measure |H_γγ| / |H_nn| and the sign of H_γγ, on the same stratified
           400-gauge population used for the p21b/p21c census.
                    |
          +---------+---------+
          |                   |
      gate passes         gate fails
          |                   |
          v                   v
  Phase 3  KAN emits      STOP. Record γ as measured-unidentifiable, alongside
           n₀ and γ.      q (§29). That is a publishable result: it would say
           Retrain.       the daily hydrograph cannot see stage-dependent
                          roughness either, which sharpens the paper's thesis.
```

Phase 1 is self-contained and reversible. Phase 3 is the only phase that costs
training time.

### Gate criteria, pre-registered

Both must hold on well-fit gauges (`nse0 > 0.3`), against the `q` baseline of
§29/§30:

1. **Magnitude**: median `|H_γγ| / |H_nn| ≥ 0.25`. `q` managed 0.062.
2. **Sign**: `H_γγ > 0` (a genuine minimum, not a saddle) at **≥ 70 %** of
   gauges. `q` was concave at 50–53 %, which is the more damning of its two
   failures and the one that makes an "optimum" meaningless.

Failing either stops the work at Phase 2.

---

## 6. Blast radius

| file | change | risk |
|---|---|---|
| `src/geometry.rs:40-47` | `d_ref^γ` factor, exponent `3/(5+3q+3γ)`, new `gamma` argument | **invariant 1** |
| `src/routing/mmc_op.rs:1160` | forward `β += γ·A/(T·d)` | **invariant 4** |
| `src/routing/mmc_op.rs:612`, `BetaGrads` (`:191`) | the matching **backward** term | **invariant 4, highest risk** |
| `src/experiment/adjoint/hydraulics.rs:49` | same `β` correction, or the landscape probe silently measures the wrong surface | high, and easy to forget |
| `src/routing/mmc.rs` | thread `gamma` through `setup_inputs` / `route_timestep` | low |
| `src/config.rs` | `parameter_ranges.gamma`, `d_ref`, validation | low |
| `src/nn/kan_head.rs` | none (Phase 3 just adds a name to `learnable_parameters`) | none |

**The backward is the real cost.** `src/sparse/` and `mmc_op.rs` carry a
hand-written `Backward` impl precisely so the tape stays O(nnz) per timestep
(invariant 4). `β` already contributes four chain-rule terms through
`BetaGrads`; `γ·A/(T·d)` adds a fifth path plus a new gradient into `γ` itself.
This is not a drop-in the way `nse-batch-deriv` was, and it is where I would
expect this to go wrong.

**Byte-identity at `γ = 0` is the safety net.** Implement `gamma` as
`Option<Tensor>` so that `None` takes the literal existing code path rather than
relying on `powf(0.0) == 1.0`. Then `compare_ddr_sandbox` cannot move.

### Gates for this change

```bash
cargo test --test ddr_sandbox_match          # invariant 1, must not move
cargo run --release --example compare_ddr_sandbox   # ABSOLUTE MATCH
cargo test --test sparse_gradcheck --test mmc
cargo test --test gamma_gradcheck            # NEW: analytic vs finite-difference
cargo test --test gamma_off_parity           # NEW: γ=None byte-identical
cargo test --release --test juniata_acceptance --test gridded_acceptance
```

Plus one gate this design needs that the existing suite has no analogue for:

**A numerical `c = dQ/dA` check.** Perturb area, recompute discharge through the
full geometry, and assert the analytic `β` matches the finite difference with
`γ ≠ 0`. Without it, a wrong celerity would show up only as slightly wrong peak
timing, which is indistinguishable from ordinary model error and would poison
every subsequent result.

---

## 7. Concerns

1. **The backward is hand-written and gradient-exact, and this touches it.**
   Highest risk item in the design. Mitigation: `Option` gating, a dedicated
   gradcheck, and the `dQ/dA` finite-difference check above.
2. **Do not feed discharge into the KAN.** The head runs once per forward pass
   and emits static fields; making `n` a function of `Q` inside the routing loop
   means 346,321 KAN evaluations per timestep, all on the autograd tape. That is
   exactly what invariant 4 exists to prevent. Emitting `γ` and applying the law
   in closed form keeps the head where it is.
3. **`n₀` is no longer Manning's `n`.** It is roughness at `d = d_ref`. With
   `d_ref = 1 m` and most CONUS baseflow depths below 1 m, `n = n₀·d^(−γ) > n₀`,
   so `parameter_ranges.n` cannot be carried over unchanged and every published
   `n` number becomes incomparable. Rename the field in the dump to avoid a silent
   unit change in the parameter maps.
4. **`n` blows up as depth goes to zero.** At `depth_lb = 0.01 m` and
   `γ = 0.183`, `n/n₀ = 2.3`. Tolerable, but `γ` must be bounded and `n` clamped;
   an unbounded `γ` on a drying reach is a NaN source.
5. **It deviates from DDR.** DDR has no stage-dependent roughness, so a CONUS
   model trained with `γ` is no longer parameter-comparable to DDR's, and the
   KAN-head parity fixtures do not cover it. Flag in the manifest.
6. **It cannot be validated directly.** Our inputs carry no at-a-station
   hydraulic geometry. The only checks available are indirect: whether routed
   peak timing improves, and whether the fitted `m` moves toward 0.34.
7. **A fourth learnable parameter reintroduces §32's question.** Expect `γ` to
   correlate with `n`. Unlike `q`, that is physically defensible, so the
   discriminating test is seed reproducibility, not correlation: train two seeds
   and check whether the `γ` field agrees.
8. **`γ` may be absorbed by `n`.** Over a narrow range of stages, `n₀·d^(−γ)` is
   nearly a constant, so the two could trade off almost freely. §30's curvature
   correlation between `n` and `γ` is the thing to measure at the Phase 2 gate,
   alongside the magnitude and sign.

---

## 8. Assumptions

1. **Observed at-a-station exponents are `b = 0.26, f = 0.40, m = 0.34`**, against
   downstream `0.50 / 0.40 / 0.10`. Verified 2026-09-12 against the hydraulic
   geometry literature, not quoted from memory. They are averages and vary by
   channel type, so 0.34 is a central prior, not a target to hit.
2. **`d_ref = 1.0 m`, global.** A per-reach `d_ref` would be redundant with `n₀`
   (only the product `n₀·d_ref^γ` enters), so a per-reach `d_ref` is not
   identifiable and must not be added.
3. **`γ ≥ 0`.** Channels get smoother as they fill, not rougher. Proposed range
   `[0.0, 0.4]`, which spans `m` from `2f/3` to roughly the observed value with
   room above.
4. **`w = p·d^q` is retained.** This design changes roughness only. The width law
   and the `q` question are the other spec's business.
5. **The trapezoid correction `γ·A/(T·d)` is exact**, derived from `dA/dd = T`
   which holds for any cross section, not just the power-law one.

---

## 9. Success criteria

1. **Phase 1** -> `compare_ddr_sandbox` still ABSOLUTE MATCH with `γ = None`;
   `gamma_gradcheck` and the `dQ/dA` check pass at `γ ∈ {0.1, 0.2, 0.4}`.
2. **Phase 2 gate** -> median `|H_γγ|/|H_nn| ≥ 0.25` and `H_γγ > 0` at ≥ 70 % of
   well-fit gauges, on the stratified 400-gauge population.
3. **Phase 3** -> retrain; fitted at-a-station `m` moves from 0.381 toward 0.34,
   median NSE within 0.02 of a matched `γ = 0` control, and the `γ` field
   reproduces across two seeds.

**Registered prediction.** `γ` passes the gate where `q` failed, because it acts
on travel time rather than on channel shape. If it fails, that is the more
valuable result: it would say the daily hydrograph cannot see stage-dependent
roughness either, and the paper's claim strengthens from "the width exponent is
unconstrained" to "everything except a single roughness level is unconstrained".

---

## 10. Estimated cost

| phase | work | compute |
|---|---|---|
| 1 | geometry + celerity forward and backward, 3 new tests | none |
| 2 | one landscape axis, one sharded study | ~6 h CPU, 400 gauges |
| 3 | config + retrain + matched control + second seed | ~8 h CPU |

Phase 2 is the decision point and costs a sixth of the total.
