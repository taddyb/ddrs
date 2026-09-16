# Depth-dependent leakance: does a volume term absorb a runoff product's volume bias?

**Status:** design, not implemented. Written 2026-09-15.
**Gate:** no stage exponent is added to leakance until the delta-free re-run
(Phase 1) shows the term absorbs *aggregate* volume bias at all. See §8.
**Sibling design:** `research/specs/2026-09-12-stage-dependent-roughness-design.md`,
whose structure and gates this document mirrors.

**Background this builds on, not around:**

1. Leakance is NOT identifiable per reach from gauged discharge
   (`research/findings/2026-07-06-leakance-nogo-scientific-summary.md`). A gauge
   observes the sum of zeta over its upstream network; a sum does not determine
   its addends. Nothing here lifts that limit, and this design claims nothing
   about the per-reach field.
2. The log-space denormalisation guard collapsed `K_D`'s box. Verified by reading
   `src/routing/utils.rs::denormalize` in this worktree: `log_min = (lo + 1e-6).ln()`
   with `lo = 1e-8` gives `log_min = ln(1.01e-6)`, which sits *above*
   `log_max = ln(1e-6)`, so the map inverts and `K_D` spans `[1e-6, 1.01e-6]`
   whatever the head emits. The 2x2's "K_D at ceiling, 1.003e-6, frac@ceil 100 %"
   (`2026-07-01-leakance-hourly-findings.md` §3) is that band, not a preference.
   Phase C's widened `[1e-8, 1e-5]` became `[1.01e-6, 1e-5]`, one decade instead
   of three. The fix in flight (`log_space_lower`, sibling worktree
   `agent-a4e2dc3cb12fb10d5`, read but not merged here) uses `lo.ln()` whenever
   `lo > 0`. Assumed to land before anything below runs.
3. Roughness is a *timing* absorber (`2026-09-08-landscape-hypothesis-tests-findings.md`
   §37, §38): the learned `n_0` tracks each product's inflow lead over the
   observation, and the hourly MTS-LSTM arm marks the limit, since its timing
   was repaired but its peaks stayed 44 % too small (KGE alpha 0.65, routing
   +0.007 NSE). Roughness can delay and attenuate; it cannot create or destroy
   volume. Leakance is the only term in the model that can.

---

## 1. The question, and what the data already say about it

**Hypothesis.** Each learned term absorbs the bias it is dimensionally able to
absorb: roughness the timing bias, leakance the volume bias. If that holds, the
paper's thesis closes in one sentence.

Before designing the term, the volume-bias spectrum of the available products
was measured (`experiments/stage_roughness/cross_arm_fields.py` output,
`.ddrs/experiments/beven-inflow-arms/bias_absorber_per_gauge.csv`, 2,361 gauges,
`vol_ratio` = summed unrouted Q' volume over observed volume on the eval window,
computed 2026-09-15 with a stdlib script; the medians are new numbers, the FHV
column matches §38.2):

| product | vol_ratio median (p10, p90) | gauges < 0.9 | gauges > 1.1 | FHV of summed Q' | routed KGE alpha (§38.2) |
|---|---|---|---|---|---|
| retrospective (reference) | **1.116** (0.91, 1.39) | 9 % | 54 % | −10 % | 0.91 |
| dHBV2 distributed | 1.141 (0.96, 1.47) | 6 % | 61 % | −9 % | 0.91 |
| NH daily LSTM | 1.037 (0.75, 1.49) | 22 % | 36 % | −18 % | 0.85 |
| dHBV2 lumped | **0.968** (0.77, 1.66) | 34 % | 31 % | −27 % | 0.85 |
| hydroDL LSTM | 0.884 (0.68, 1.57) | 54 % | 23 % | −35 % | 0.75 |
| NH hourly MTS-LSTM | **0.975** (0.75, 1.53) | 31 % | 28 % | −39 % | **0.65** |

Three things follow and they shape everything below.

1. **The reference product over-delivers volume by 12 % at the median, at 54 %
   of gauges.** That is a net surplus a sink can absorb. It is also the largest
   systematic volume bias in the set, which no one had put next to the leakance
   results before.
2. **The hourly MTS-LSTM's volume is right (0.975).** Its error is not net
   volume; it is the *distribution* of volume across stage: too little at the
   peak, correspondingly too much on the recession (alpha 0.65 with beta near 1).
   A losing term removes water; to repair this product one would have to *add*
   water at high stage and remove it at low stage, which is a negative
   conductance. No physical leakance has that sign. So the arm that motivated
   the question is the arm the term is predicted NOT to help (§7, prediction P3).
3. **The lumped dHBV2 has no net bias but the widest per-gauge spread** (p10
   0.77, p90 1.66). Its volume error is gauge-specific and of both signs. That
   is the arm on which "leakance tracks the product's volume error gauge by
   gauge" is most testable, and it needs the *unclamped* (gaining allowed) form.

The hypothesis therefore has to be stated more carefully than "leakance absorbs
volume bias": **a signed, stage-profiled sink or source absorbs the signed net
volume error each gauge sees, and the stage exponent decides which part of the
hydrograph pays.** The identifiable object is the per-gauge network sum, which is
exactly what the NO-GO says a gauge constrains. The design stays on that side of
the line.

---

## 2. The equation

### 2.1 What exists

`src/routing/leakance.rs::zeta_forward` (verified):

```
  zeta = leakance_factor · area_z · K_D · head
  area_z = (p · d)^q_eps · length            q_eps = q_spatial + 1e-6
  head   = max(0, d − d_gw)   (losing_only, default)   or   d − d_gw
```

subtracted from `b_rhs` at S25 in `mmc_op.rs::forward_chain_inner`. `d` is the
shared power-law depth from S6, computed from `q_t` (the previous step's
discharge), so zeta does not feed back into depth or celerity within a step.
With `p = 21` and `q = 0.65` prescribed (the inflow arms), the stage profile of
the loss at one reach is already `d^0.65 · (d − d_gw)`, roughly `d^1.65` above
the threshold.

### 2.2 The proposed form

```
  K_D(d) = K_D0 · (d / d_ref)^delta            d_ref = 1 m, fixed
  zeta   = C · L · p^q_eps · d^(q_eps + delta) · head(d)        C = leakance_factor · K_D0
```

At `delta = 0` this is byte-identical to the current term (`(d/1)^0 = 1.0` in
f32, but the implementation gates on `delta.is_none()` rather than relying on
that, as the gamma work did). The direct analogue of `n(d) = n_0 (d/d_ref)^(−gamma)`:
a static per-reach exponent applied in closed form inside the timestep, no
discharge fed to the KAN, no new state.

**What delta does physically in the model.** The stage exponent of the loss is
`s = q_eps + delta`. `s` decides where in the hydrograph volume is removed:

```
  s ≈ 0     (delta ≈ −0.65):  loss ∝ (d − d_gw), nearly flat in stage above
                               threshold: a baseflow abstraction
  s = 0.65  (delta = 0):       the current form, loss grows ~d^1.65
  s ≈ 1.5   (delta ≈ +0.85):   loss ∝ d^2.5, concentrated at the peak
```

That is the exact counterpart of gamma, which decides where in the hydrograph
delay is applied. The paper's sentence becomes: roughness level sets the timing
correction and gamma its stage profile; conductance sets the volume correction
and delta its stage profile.

### 2.3 The confound with `q_eps`, and how it is resolved

`q_eps` and `delta` are both exponents on the same `d` inside `area_z`. Only
their sum `s` enters the physics, so with both learned the pair is an exact
flat direction. Resolution: **`q_spatial` stays fixed at 0.65**, as it already
is on every inflow arm, and `delta` is the single learned stage exponent. This
is precisely what was done for roughness (fix `p`, `q`; learn `n_0`, `gamma`),
for the same reason: §29 and §30 showed the width channel is a saddle and the
head rode it. Config load must reject `q_spatial` and `delta` learned together
(§6). A cleaner parameterisation would learn `s` directly and drop `q_eps` from
`area_z`; I keep `delta` on top of the fixed `q` so that `delta = 0` recovers
the existing term and its recorded runs bit for bit.

### 2.4 Alternatives considered

| form | what it is | why not |
|---|---|---|
| **Wetted-perimeter conductance** `zeta = C · P_wetted · L · head` | SFR2's ICALC=2 form (Niswonger & Prudic 2005): `K` constant, conductance follows the wetted perimeter of an eight-point section as stage changes. `P` is already computed in `compute_trapezoidal_geometry`. | Zero new parameters, so it cannot absorb a product-specific stage profile; the stage law is fixed by geometry. **Keep as the physics-only control arm** (§7): it is the form a reviewer will ask for. |
| **Disconnection cap** `head = min(d − d_gw, h_sat)` | Brunner et al. 2009: below a critical water-table depth the flux saturates and stops depending on aquifer head. | Modifies the head term, not the stage dependence of conductance; adds a parameter with the same per-reach non-identifiability and no volume-profile lever. |
| **Dynamic bank storage** (Birkhead & James 2002, J. Hydrol. 264) | A per-reach storage that fills on the rising limb and returns on the recession. | Volume-neutral over an event by construction, so it cannot absorb a net volume bias; and it adds a state variable to the timestep, a tape entry per reach per step, and a checkpoint field. It attenuates peaks, which is the wrong direction for every product here. |
| **Loss as a power of discharge** `zeta = a · Q^m` (transmission-loss tradition) | | Through Manning, `Q^m` is a power of `d`, so it is the proposed form with a different exponent bookkeeping and no head term. Rejected as redundant. |
| **Signed conductance** `C ∈ [−C_max, C_max]` | Lets a reach add water at high stage. | Not physics. Named here because it is the only form that could repair the hourly arm (§1), and that makes it a useful *fitting-device discriminator*, not a model term. Optional ablation in §7, never a default. |

---

## 3. Physical basis

Checked by web search on 2026-09-15; abstract-level verification only, none of
the full texts were read for this document (two fetches returned 403). The
existing 32-citation review `research/findings/2026-07-04-leakance-literature-review.md`
covers the base term; the searches below are specifically about stage dependence.

**What supports a stage-dependent conductance.**

- SFR2 (Niswonger & Prudic 2005, USGS TM 6-A13) makes conductance stage
  dependent through the wetted perimeter of a Manning-derived cross-section;
  `K` itself is held constant. The MODFLOW-family precedent for "conductance
  rises with stage" is therefore *geometric*, which our `area_z ∝ d^q` already
  encodes.
- Streambed `K` is transient during flood waves: Gianni et al. 2016 (WRR,
  "Rapid identification of transience in streambed conductance by inversion of
  floodwave responses") and Xian, Jin & Zhan 2019 (J. Hydrol., the buffer-effect
  follow-up) invert flood-wave responses for a time-varying `K`. Naganna et al.
  2017 (Environ. Sci. Pollut. Res. 24:24765) review the mechanisms: scour on
  the rising limb restores `K`, fine-sediment deposition on the recession
  lowers it, and `K` shifts sharply when stage crosses the aquifer head. This is
  real, but it is *hysteretic*, not a monotone function of stage.
- Bank and floodplain connection: Doble et al. 2012 (Groundwater 50:77) find
  sloping banks raise bank infiltration by 98 % and storage by 40 % against
  vertical banks, i.e. the newly wetted bank area at high stage matters; a 2026
  sensitivity study (MDPI Hydrology 13(7):193) reports floodplain width raising
  infiltration and bank storage with peak attenuation of 2 to 6 %; dryland
  transmission losses at Walnut Gulch are reported as "unpredictably high" when
  flow exceeds channel capacity (Wohl 2021 review of floodplain storage). These
  support a *jump* in loss at overbank stage (`delta > 0` above bankfull).
- Hyporheic exchange increases with discharge (MDPI Water 2019, 11(7):1436,
  boreal stream, temperature tracing across low, base and high flow), but
  hyporheic flow returns to the channel within the reach and is not a net sink.

**What argues against it, or against a single sign.**

- Vertical and lateral profiles of `K`: along the Elkhorn River (Nebraska) `K`
  decreases with depth into the bed; on the Beiluo River (Hydrogeology J. 2016)
  `K` is higher in the submerged bed than on the exposed banks. As stage rises
  onto the banks, the *marginal* wetted surface is less conductive, which is
  `delta < 0`.
- Rushton 2007 (already in the review): a lumped conductance conflates bed and
  aquifer resistance, so `K_D0` is not a bed property even before a stage law
  is put on it.

**Verdict on the physics.** The support for "conductance varies with stage" is
real but thin and *of indeterminate sign*: geometry and overbank connection say
it rises, bank-versus-bed profiles say it falls per unit wetted area, and flood
transience says it is hysteretic. A monotone power law with a learned exponent
spans those regimes without committing to any of them. **As physics it would
not survive review if `delta` is presented as a streambed property. As a
parsimonious stage-profile parameterisation, presented next to the SFR2
wetted-perimeter form as the physics baseline and with the sign left free, it
would.** The honest framing is the one the roughness work already uses: the
learned exponent is a diagnostic of what the gauges ask for, not a measurement
of the bed.

---

## 4. Degeneracies, stated up front

1. **`leakance_factor · K_D0` is an exact flat direction.** Only the product
   enters `zeta_forward`; verified. The 2x2 confirms the optimiser used it: with
   `K_D` frozen at 1e-6 by the bug, `leakance_factor` sat interior at 0.33
   (§3 of the hourly findings), so the effective conductance was
   `0.33e-6` and free. **Recommendation: collapse to one effective conductance
   `C` in log space over `[1e-9, 1e-5] s⁻¹`.** What it costs: the DDR-schema
   keys `K_D` and `leakance_factor` stop being the learned quantities (keep
   accepting them for the old op so recorded runs stay reproducible); the
   linear-space `factor` gave the head an easy path to an exact zero, which a
   log box does not have (at 1e-9 with `area_z ~ 1e4 m²` and 1 m head, zeta is
   1e-5 m³/s, which is zero for every purpose here); and Phase C's
   `leakance_impervious_threshold` mask stays as the hard zero. Note the fix to
   `denormalize` changes the *floor* of `K_D`, not the ceiling: the reachable
   product range was already `[0, 1e-6]` through `factor`, so a re-run with the
   fix and the old two-key form would mostly re-parameterise the low end. The
   ceiling question ("wants more exchange") needs the wider box above, not the
   fix.

2. **`delta` against `d_gw` at a fixed `q_eps`.** Over the range of stages a
   reach actually runs, `d^s · (d − d_gw)` with a larger `d_gw` and larger `s`
   is nearly indistinguishable from a smaller `d_gw` with smaller `s`: both
   steepen the profile. This is §37.2's degeneracy one level down (daily
   discharge sees the composite at the effective depth, not the slope), and it
   is predicted to reproduce: `rho(delta, d_gw)` and `rho(delta, C)` high, the
   `(C, delta)` landscape a valley (§7, P4). The `losing_only` clamp partly
   breaks it, because `d_gw` alone sets the threshold below which zeta is
   exactly zero, and that threshold is visible in the low-flow volume.

3. **`delta` against `q_eps`.** Exact when both are learned; removed by fixing
   `q` (§2.3).

4. **`d_ref`.** With per-reach `C_i`, `d_ref^(delta_i)` is absorbed into `C_i`
   reach by reach, so `d_ref` is never identifiable. Fixed at 1 m, never a
   parameter. (For gamma the same argument only held for a global gamma; here
   it holds always because `C` is per reach.)

5. **`delta` against `gamma`.** Both are static per-reach stage exponents
   emitted by the same trunk, and §31 and §37.2 say the trunk emits one latent
   direction. Expect `rho(delta, gamma)` near ±0.9 whatever the physics. The
   discriminating read-out is not correlation but whether the *network-median*
   `delta` differs between products in the direction their volume profile
   predicts (§7, P4).

6. **Per-reach zeta against the gauge sum.** Unchanged from the NO-GO and not
   addressed. Every read-out below is on the per-gauge upstream sum.

---

## 5. Gradient design

### 5.1 Where delta enters and where it does not

Verified in `mmc_op.rs::timestep_backward_core` and `TimestepLeakanceOp::backward`:
zeta touches the routing only through `b_rhs`, and its backward is entirely
inside the `zeta_hook` closure, which receives `gb_rhs` right after B27 and
returns `ZetaGeomGrads {g_depth, g_q_eps, g_p_spatial}` for the core to add at
three accumulation points (`gd_total` before the depth-clamp mask, `gq_spatial`
at B1, `gp` at final assembly). The three leakance-parent grads are captured out
of the hook and registered by the op after the core returns.

Because `delta` changes zeta and nothing else, **no new B-path appears in the
core.** This is the main way the change is cheaper than gamma, which touched B5
(depth exponent), B15 (velocity) and B17 (celerity). Everything new lives in
`leakance.rs`:

```
  zeta = C · L · (p·d)^q_eps · (d/d_ref)^delta · head

  ∂zeta/∂delta  = zeta · ln(d / d_ref)                       NEW parent grad
  ∂zeta/∂d      = C·L·w·(d/d_ref)^delta · [ 1 + head·(q_eps + delta)/d ]
                                                             existing g_depth, one extra term
  ∂zeta/∂q_eps, ∂zeta/∂p, ∂zeta/∂C, ∂zeta/∂d_gw               existing, each picks up the (d/d_ref)^delta factor
```

`ZetaGrads` gains `g_delta`; `ZetaGeomGrads` is unchanged in shape, so
`timestep_backward_core`'s signature and the three fold-in points do not move.
The `losing_only` gate and the impervious mask apply to `g_delta` exactly as
they apply to the other three (they multiply `gzeta` before every product).
`ln(d/d_ref)` is finite because `d ≥ depth_lb = 0.01 m` (clamped at S6), so
`ln` is bounded below by −4.6 at `d_ref = 1`.

### 5.2 The operator

Follow the sibling pattern (`TimestepGammaOp` was added beside `TimestepOp`,
not by widening it): **a new `TimestepLeakanceStageOp`, `Backward<I, 9>`**,
parents `[n, q_spatial, p_spatial, q_t, q_prime_t, gamma, C, d_gw, delta]`.
The existing 8-parent `TimestepLeakanceOp` is left untouched so every recorded
leakance run stays reproducible through its own op.

Why nine and not eight: the inflow arms' controls are `n_0 + gamma` heads
(`experiments/beven-inflow-arms/experiment.yaml`), and the experiment must use
the same head plus leakance. Today that combination is rejected by
`config.rs::validate_learned_gamma` because `TimestepLeakanceOp` has no gamma
parent and `timestep_forward_leakance` builds its state with `gamma_t: None`
(verified). The new op carries the gamma parent, and its backward simply passes
`mask.gamma = ids[5].is_some()` to the core, which already sums B5, B15 and B17
into `ParentGrads::gamma` when `state.gamma_t` is `Some`. The forward is a
copy of `timestep_forward_leakance` that threads `gamma_p` into the state the
way `timestep_forward` does. Config selects the op by the presence of
`conductance` in `learnable_parameters` (§6), so old configs never reach it.

Dispatch in `mmc.rs::route_timestep` (verified: the leakance branch is chosen
when all three leakance tensors are `Some`) gains a third branch, chosen when
`self.conductance` is `Some`, ahead of the old leakance branch. The zeta
diagnostic sink (`ZetaStepDiag`, `zeta_abs_sum`, `zeta_net_sum`) is reused as
is; it is what the read-out in §7 reads.

Blast radius:

| file | change | risk |
|---|---|---|
| `src/routing/leakance.rs` | `zeta_forward`/`zeta_backward` take `delta: Option<Tensor>`; `g_delta` | low, unit-tested in place |
| `src/routing/mmc_op.rs` | new 9-parent op + forward, `LeakanceSaved` gains `conductance`, `delta` | **invariant 4**, but no core change |
| `src/routing/mmc.rs` | `SpatialParameters` gains `conductance`, `delta`; denormalise; dispatch | low |
| `src/config.rs` | keys, ranges, guards (§6) | low |
| `src/training/forward.rs` | thread the two new head keys; `physical_to_normalized` uses the fixed lower bound (the in-flight fix already shares `log_space_lower`) | low |
| `src/dump_parameters.rs` | denormalise the new keys for the parameter dump | low |
| `src/experiment/landscape/objective.rs` | only if a landscape probe is wanted: `AXIS_PARAMS` gains `conductance`, `delta`; the hard-coded `k_d: None, d_gw: None, leakance_factor: None` (verified) must carry the trained leakance fields | medium, and easy to forget: today the landscape silently probes a leakance arm *without* its leakance |
| `src/experiment/adjoint/influence.rs` | none; it already refuses leakance arms (verified) | none |

### 5.3 Tests

Existing `tests/leakance_gradcheck.rs` (verified) calls
`timestep_forward_leakance` directly with physical tensors, `EPS = 1e-3`
relative for routing parents and absolute steps `4e-7` for `K_D`, `1.5` for
`d_gw`, `0.4` for `leakance_factor`, tolerance 5e-3 relative. It sweeps the
op, not the head, so it never sees `denormalize`: **the K_D collapse was
invisible to every leakance gradient check ever run.** `tests/leakance_off_parity.rs::head_driven_leakance_changes_output`
does cross the head but asserts only that the output differs and is finite.

New tests, all on the 4-reach chain the existing file uses:

1. `leakance_stage_gradcheck`: the nine parents of the new op against central
   differences. Steps: routing parents as today; `C` relative `1e-1` in log
   space (perturb `ln C` by ±0.1, since the parameter is log-box); `d_gw` ±0.5
   (the existing 1.5 spans most of the box; zeta is linear in `d_gw` above
   threshold so either is exact, but 0.5 keeps the losing/gaining gate fixed
   across the stencil); `delta` ±1e-2 absolute at base points
   `delta ∈ {−0.5, 0, +0.5}`; `gamma` as in `sp8_gradcheck.rs::gradcheck_gamma_learned`.
   Tolerance 5e-3 relative, 1e-4 absolute. Run on losing, gaining and mixed
   `d_gw` vectors as the existing file does, so the clamp gate is exercised on
   `g_delta`.
2. `leakance_stage_off_parity`: `delta = None` and `C = factor · K_D`
   reproduces the 8-parent op's forward and all shared parent grads to 1e-6
   relative. This is the byte-identity net for recorded runs.
3. **`leakance_box_round_trip`, the normalisation-crossing test.** For every
   log-space parameter in the config, assert `denormalize(0) == lo` and
   `denormalize(1) == hi` to 1e-3 relative, and that `physical_to_normalized`
   inverts it. One assertion each; either would have caught the collapse on
   the day the box was written. Then a head-level gradcheck: perturb the head's
   *normalised* output `u` for `C` by ±1e-2 and compare the autograd
   `dL/du` through `denormalize` and the op against central differences. This
   crosses the layer the existing checks are blind to.
4. `leakance_volume_accounting`: on a single-gauge chain, the eval-window
   volume of routed outflow minus the volume of lateral inflow equals minus the
   window sum of `zeta_net · dt` to 1e-4 relative. Without this the volume
   read-out in §7 rests on an untested identity. `tests/gauge_mass_conservation.rs`
   covers the no-leakance identity today and hard-codes `leakance_factor: None`
   (verified), so this is a new case, not a modification.
5. `gamma_with_leakance_gradcheck`: gamma's gradient through the new op equals
   its gradient through `TimestepGammaOp` at `C → 1e-9`. Guards the claim that
   the nine-parent op changes nothing for the six gamma-paths.

Gates, mirroring the roughness design:

```bash
cargo test --test ddr_sandbox_match                    # invariant 1, must not move
cargo run --release --example compare_ddr_sandbox      # ABSOLUTE MATCH
cargo test --test leakance_gradcheck --test leakance_off_parity   # existing, unchanged
cargo test --test leakance_stage_gradcheck --test leakance_stage_off_parity \
           --test leakance_box_round_trip --test leakance_volume_accounting   # NEW
cargo test --test sp8_gradcheck                        # gamma paths unchanged
cargo test --release --test juniata_acceptance
```

---

## 6. Config and guards

Keys (YAML, under `params`):

```yaml
params:
  use_leakance: true                       # existing
  leakance_losing_only: false              # existing; MUST be false for the volume test (§7)
  parameter_ranges:
    conductance: [1.0e-9, 1.0e-5]          # NEW, replaces K_D x leakance_factor; s^-1
    d_gw: [-2.0, 2.0]                      # existing
    leakance_exponent: [-1.0, 1.0]         # NEW, delta
  log_space_parameters: [p_spatial, conductance]
  defaults:
    p_spatial: 21.0
    q_spatial: 0.65
kan_head:
  learnable_parameters: [n, gamma, conductance, d_gw, leakance_exponent]
```

Config load must reject:

1. `conductance` together with `K_D` or `leakance_factor` in
   `learnable_parameters` (two parameterisations of one quantity).
2. `leakance_exponent` without `conductance` and `d_gw` (nothing to scale), or
   without `use_leakance: true`.
3. `leakance_exponent` together with `q_spatial` in `learnable_parameters`
   (§2.3, exact confound).
4. `conductance` not in `log_space_parameters`, or a `conductance` box whose
   lower bound is not strictly positive (a linear box over four decades puts
   the whole head output in the top decade).
5. `leakance_exponent` range outside `[−1, 1]` or with `lo >= hi`. At `delta = −1`
   and `q = 0.65` the loss still grows as `d^0.65` above threshold; at
   `delta = +1` it grows as `d^2.65`, and `(d/d_ref)^delta` at `depth_lb` is
   `0.01`, harmless. Below −1 the loss would fall with stage faster than the
   head term grows, which is the regime with no physical reading at all.
6. `use_cuda_graphs: true` with leakance (existing `validate_leakance`, keep).
7. A missing `params.defaults.q_spatial` when `q_spatial` is fixed (existing
   `validate_fixed_q_spatial`, keep).

**The learned-gamma-with-leakance guard.** `validate_learned_gamma` currently
rejects `gamma` in `learnable_parameters` with `use_leakance: true`, with the
message that the eight-parent op has no gamma parent so gamma would silently
receive no gradient (verified). **Keep it for the old op and lift it only when
`conductance` is the leakance key**, i.e. when the nine-parent op is selected.
The reason it exists is exactly right and stays true for `TimestepLeakanceOp`;
lifting it globally would let an old-style config (`K_D` + `leakance_factor`)
train gamma with a dead gradient. The guard message should name the new key as
the way to combine them.

A global `params.stage_roughness.gamma` with leakance is already allowed and
threaded (`timestep_forward_leakance` reads `stage_roughness_params()`,
verified), so a fixed-gamma control needs no guard change.

---

## 7. The experiment

### 7.1 Arms

All arms: the inflow-arm recipe (`experiments/beven-inflow-arms/experiment.yaml`:
500 Adam updates, seed 42, `nse-batch`, 2,365 gauges, CPU) with the `n_0 + gamma`
head, channel fixed at `p = 21`, `q = 0.65`. Leakance-off controls exist for all
three products.

| arm | product | leakance form | purpose | control |
|---|---|---|---|---|
| A1 | retrospective | `C`, `d_gw`, **`delta = 0`** (fixed) | does a volume term absorb the 12 % surplus at all | `2026-09-12T23-39-03Z` |
| A2 | retrospective | `C`, `d_gw`, **`delta` learned** | does the stage profile matter beyond A1 | A1 |
| B | dHBV2 lumped | `C`, `d_gw`, `delta` learned, **unclamped** | per-gauge, signed volume tracking (the sharpest test) | `2026-09-13T13-56-50Z` |
| H | NH hourly MTS-LSTM | `C`, `d_gw`, `delta` learned | the predicted null: volume is right, variance is wrong | `2026-09-13T14-14-46Z` |
| P | retrospective | wetted-perimeter conductance, `C` only (§2.4 row 1) | the physics-only baseline a reviewer will ask for | A1 |
| B' | dHBV2 lumped | as B, seed 43 | reproducibility of the P1 correlation, since one seed decides the thesis | B |

`leakance_losing_only: false` on every arm. The clamp would forbid the gaining
direction and therefore forbid the test on B (34 % of its gauges are volume
deficient). The Phase C reasons for the clamp (sign ambiguity with baseflow,
per-reach realism) are per-reach concerns that the NO-GO already settled; the
read-outs here are per-gauge sums where the sign is the signal. The optional
signed-conductance ablation on H (§2.4, last row) is not in the base set; run
it only if H comes out as predicted and the "fitting device" question is worth
one more 18-hour run.

### 7.2 Read-out

Per gauge, on the eval window (1995-10 to 2010-09, the arms' testing window):

1. **Volume absorbed:** `V_zeta = Σ_t Σ_{i ∈ upstream(g)} zeta_net[i, t] · dt`,
   from the eval-phase zeta diagnostic (`src/training/eval.rs` writes
   `zeta_net_mean` per eval reach into the run's `kan_parameters.nc`; the
   upstream set is the gauge subgraph). Express as a fraction of observed volume.
2. **Volume error of the product:** `vol_ratio − 1` from
   `cross_arm_fields.py::gauge_inflow_error` (summed Q' over observed).
3. **Skill deltas:** `arm_delta_by_gauge.py <control> <arm>` for NSE, KGE, FHV,
   alpha, lag; add KGE `beta` (routed volume over observed) to its output,
   which it computes and does not print (verified in `metrics`).
4. **Stage split:** the same volume ratio on the days the observation is below
   and above its median, to see which part of the hydrograph paid. New, small
   script; needs only the routed and observed daily series the read-out already
   loads.
5. **Fields:** `gamma_readout.py` for `n_0`, `gamma`; extend to `C`, `d_gw`,
   `delta` with the same box-fraction and correlation table. Roughness
   cannibalisation (`Δn_0` IQR against the control; the promotion gate's Leg 2
   bounded it at 0.014).
6. **Landscape (Phase 2 only):** `experiments/landscape` with axes
   `[conductance, leakance_exponent, n]` on the §37.4 14-gauge set, after the
   `objective.rs` change in §5.2.

### 7.3 Per-product predictions

- **Retrospective (A1, A2):** the term is used as a sink. Median `V_zeta`
  negative in the flow, about half the surplus (median KGE beta from ~1.12 to
  under 1.06); `rho(V_zeta fraction, vol_ratio − 1)` ≥ +0.3 across gauges;
  NSE +0.01 to +0.02 over the control (the 2x2 and Phase C gave +0.001 to
  +0.006 with the clamp on and the box collapsed). A2 over A1: the learned
  `delta` goes *negative* at the network median (loss taken from the recession,
  because this product's surplus sits there: FHV −10 % with volume +12 %),
  with a further small gain in alpha and beta. P (perimeter form) between the
  control and A1.
- **Lumped dHBV2 (B, B'):** the sharp test. `rho(V_zeta fraction, vol_ratio − 1)`
  ≥ +0.5, the sign of `V_zeta` agreeing with the sign of the volume error at
  ≥ 70 % of gauges with `|vol_ratio − 1| > 0.1`, reproduced on seed 43 within
  0.1 in rho. NSE +0.02 or more over the control (this product has the most
  volume error to give back).
- **Hourly MTS-LSTM (H):** the predicted null. `|V_zeta|` fraction median under
  0.02, `C` at the box floor at the majority of reaches, FHV within 5 points of
  −44 %, alpha within 0.05 of 0.65, NSE within 0.01 of the control. The
  product's volume is right and its error is variance, which a same-sign sink
  or source cannot supply.
- **Across products:** the network-median `C` orders the products by net
  surplus (retrospective ≈ distributed > daily LSTM > lumped ≈ hourly > hydroDL),
  the volume counterpart of §38.2's "median `n_0` tracks the median lead".

---

## 8. Registered prediction

Stated so that it can be wrong.

**P1 (aggregate volume absorption, the thesis).** On the lumped dHBV2 arm, the
per-gauge net leakance volume over the eval window correlates with the
product's per-gauge volume error at Spearman rho ≥ +0.5, with matching sign at
≥ 70 % of gauges whose volume error exceeds 10 %, and this reproduces on a
second seed within 0.1 in rho. On the retrospective arm the routed KGE beta
moves at least half the way from its control value to 1.

**P2 (product fingerprint).** The network-median effective conductance orders
the six products by their median summed-Q' volume surplus, monotonically.

**P3 (the limit).** On the hourly MTS-LSTM arm leakance changes FHV by less
than 5 points and alpha by less than 0.05, and routing still adds under 0.02 NSE.

**P4 (the stage exponent is a profile, not a property).** `delta` is not
identifiable per reach: `|rho(delta, C)| > 0.8` and, on the landscape,
median `|H_δδ| / |H_CC| < 0.1` with `H_δδ > 0` at fewer than 70 % of gauges,
which is the same failure gamma showed in §37.4. Its *network median* is
nevertheless product-specific: negative on the retrospective arm.

**What refutes the hypothesis.** If on the lumped arm rho < +0.3, or the sign
agreement is under 55 % (chance is 50 %), and on the retrospective arm beta
closes less than a quarter of its gap, then a physically signed leakance does
not absorb volume bias the way roughness absorbs timing bias, and the paper's
sentence must be cut to its first half: roughness absorbs timing; volume bias is
not absorbed by a physical sink even where the sink is free to act. That would
also be worth reporting, because it would say the optimiser prefers to leave a
12 % volume surplus in place rather than spend a term that can remove it, which
is a statement about the objective (NSE's weak sensitivity to constant bias
against its strong sensitivity to timing, §25.2), not about the physics.

**What would make the result uninterpretable rather than negative.** Roughness
cannibalisation (`Δn_0` IQR > 0.1 against the control), or `C` pinned at the
box ceiling at more than 10 % of reaches (the box is too small, not the
hypothesis wrong; widen and re-run before reading anything).

---

## 9. Honest verdict

**Against building it.**

- It adds a parameter to a term proven non-identifiable per reach, and P4
  predicts the new parameter inherits the same fate. A reviewer can fairly ask
  why a second unidentifiable exponent was added after the first one (gamma)
  collapsed onto its level parameter in §37.
- The physics for a monotone `K(d)` is thin and of undetermined sign (§3). The
  standard stage dependence (SFR2) is geometric and already present.
- The arm that motivated the question (hourly MTS-LSTM) is the arm the term is
  predicted not to help, because its bias is variance, not volume. The
  "completes the thesis in one sentence" framing survives only in the narrower
  form of §1: a signed sink absorbs signed net volume error.
- The reference product's 12 % surplus may be a Q' problem better fixed
  upstream than absorbed by the router; absorbing it makes the routed model
  right for the wrong reason, which is the paper's own complaint about `n`.
- Every leakance experiment so far has been a NO-GO on promotion, and re-running
  one with a widened box invites the reading that the question is being
  re-opened. It is not: the per-reach limit is accepted as final (§1 fact 1).

**For building it.**

- The read-out lives on the identifiable side of the NO-GO. The network sum
  per gauge is exactly what a gauge sees, and P1 asks only about that sum. The
  NO-GO says nothing about whether the sum tracks the product's volume error;
  that question has never been asked, because every previous run was on the
  reference product only and, as it turns out, with `K_D` frozen.
- The term is code-complete and gradient-exact. Phase 1 (A1, B, H) needs no
  new physics at all: the collapse to `C`, the nine-parent op so the existing
  `n_0 + gamma` controls can be reused, the box round-trip test, and a read-out
  script. `delta` is a small addition confined to `leakance.rs`, with no new
  core path (§5.1).
- The controls exist and the read-out scripts exist. The compute is CPU hours,
  not GPU days.
- The result is useful in either direction. A positive P1 closes the thesis;
  a negative P1 says something specific about the objective. A positive P3 is
  the cleanest demonstration in the paper that "bias absorber" has a
  dimensional grammar: the hourly product's error is in a dimension no term
  in the model owns.
- A volume term is the natural place to ask the question the roughness work
  could not: §38.2 found roughness correlates with peak bias at only −0.12
  pooled, "not their volume or peak error". That was measured on a model with
  no volume term. Whether volume bias is absorbable at all has not been tested.

**Recommendation: build it, staged, and stop early if Phase 1 is null.**

```
  Phase 1  collapse to C; nine-parent op (gamma parent, delta = None);
           box round-trip + volume-accounting tests; arms A1, B, H;
           read-outs 1 to 5.                                 <-- THE GATE
                    |
          +---------+---------+
          |                   |
     P1 holds on B       P1 fails on B and A1
          |                   |
          v                   v
  Phase 2  delta in          STOP. Record: a physical sink does not absorb
           leakance.rs;      volume bias from daily gauges. Write it up next
           arms A2, B', P;   to §37 as the volume-side result. No delta code
           landscape axes    is ever written.
           [C, delta, n].
```

Phase 1 answers the thesis question with zero new physics. `delta` is only
worth its code if there is a volume correction whose stage profile can be
asked about. This is the same discipline as the roughness design's "probe
before any learnable code", applied one level earlier: **absorb before any
stage law.**

Do not present `delta` as a streambed property anywhere. Present `C` as the
volume-side counterpart of `n_0` and `delta` as the counterpart of `gamma`, both
diagnostics of what the gauges ask of the router given a particular runoff
product, and put the SFR2 wetted-perimeter arm (P) in the table so the physics
baseline is visible.

### Cost

| phase | engineering | compute |
|---|---|---|
| 1 | 2.5 days: op + forward (1), config/guards/plumbing/dump (0.5), tests 2 to 5 (0.5), read-out script and `beta` in `arm_delta_by_gauge.py` (0.5) | A1 ~8 h, B ~8 h, H ~18 h CPU (per-arm times from the journal, leakance path adds a few percent); ~35 CPU-hours plus ~2 h of read-outs |
| 2 | 2 days: `delta` in `leakance.rs` + gradcheck (1), landscape axes and the `objective.rs` leakance carry (0.5), perimeter arm (0.5) | A2, B', P ~24 h; landscape 14 gauges x 3 arms ~9 h; ~35 CPU-hours |

About five engineering days and 70 CPU-hours in total; Phase 1 is half of each
and is the decision point.

---

## 10. Concerns

1. **The unclamped form re-admits the sign ambiguity Phase C closed.** A
   gaining reach adds water that the product's Q' may already contain as
   baseflow. Per reach that is unresolvable and accepted; per gauge it is the
   signal. The mitigation is the read-out, not the clamp: report the sign
   agreement with the product's volume error, and report `Δn_0` so a volume
   term that is really a roughness proxy is caught.
2. **The box.** `[1e-9, 1e-5]` is chosen so that the 2x2's effective `0.33e-6`
   sits mid-box in log space and the Phase C ceiling `1e-5` is the ceiling.
   If `C` pins at `1e-5` the read-out is void (§8). Diagnosis §3 measured that
   at `K_D = 1e-4` every reach could exceed 0.01 m³/s, so a ceiling of `1e-4`
   is the fallback.
3. **Volume accounting must be exact for the read-out to mean anything.** Test 4
   is not optional. `zeta` is subtracted from `b_rhs` before the solve and after
   the `q_prime` clamp, so `V_zeta` is what left the network, but the routed
   output is also clamped at `discharge_lb`; on a reach driven to the floor the
   accounting breaks. Report the fraction of reach-steps at the floor next to
   `V_zeta`.
4. **`nse-batch` is weakly sensitive to constant bias.** §25.2 found curvature
   lives in the hydrograph's time derivative. A constant 12 % surplus costs
   little NSE, so the optimiser may not spend the term. This is a genuine
   alternative outcome and is written into §8 as the interpretation of a null,
   not as a reason to switch objectives beforehand. If Phase 1 is null, one
   follow-up with `nnse-kge` on A1 is justified (KGE's beta term supplies the
   restoring gradient directly), and only then.
5. **The hourly arm costs 18 hours per run.** H is in Phase 1 because P3 is the
   prediction most likely to be quoted; if compute is short, run A1 and B first
   and H only if P1 holds, since P3's refutation without P1 is not interesting.
6. **The landscape probe silently drops leakance today** (`objective.rs`
   hard-codes `k_d: None`). Any landscape run on a leakance arm before the §5.2
   change measures a different model than was trained. Add a hard error there
   in Phase 1 even if the axes wait for Phase 2.
7. **Deviation from DDR.** DDR's Python still applies `bounds[0] + 1e-6`
   unconditionally (per the fix's doc comment), so a DDR leakance run and a ddrs
   one would disagree on `K_D` after the fix lands. Leakance is outside the
   parity fixtures, so no invariant moves, but the manifest should flag the
   collapsed key and the fixed guard.

---

## 11. Assumptions

1. The `denormalize` fix lands as read in the sibling worktree: `lo.ln()` for
   `lo > 0`, the old nudge for `lo <= 0`. If it lands differently, test 3 is the
   thing that notices.
2. `vol_ratio` in `bias_absorber_per_gauge.csv` is summed unrouted Q' volume
   over observed volume on the same eval window the arms were scored on.
   Inferred from `cross_arm_fields.py::gauge_inflow_error`'s docstring, not
   re-derived.
3. The eval-phase zeta diagnostic covers every reach in the eval gauges'
   subgraphs, so `V_zeta` per gauge is a complete upstream sum. Inferred from
   `eval.rs` field names and the CLAUDE.md description; the read-out script
   should assert it against the subgraph size.
4. Fixing `q = 0.65` and `p = 21` on all arms, as the inflow arms already do.
5. `d_ref = 1 m`, never learned (§4, item 4).
6. The nine-parent op's gamma paths are exactly the six-parent op's; test 5
   is the check.

## 12. What was verified by reading code, and what was inferred

Verified: the `denormalize` guard and its effect on `[1e-8, 1e-6]`; the fix in
the sibling worktree; `zeta_forward`/`zeta_backward` and their inputs; the hook
structure and the three fold-in points in `timestep_backward_core`; the three
`Backward` impls and their parent lists; `timestep_forward_leakance` reading the
scalar gamma and setting `gamma_t: None`; the dispatch order in
`mmc.rs::route_timestep`; every guard named in §6; the existing leakance tests'
entry points and step sizes; `objective.rs` dropping leakance and its
`AXIS_PARAMS`; `influence.rs` refusing leakance arms; the metrics computed by
`arm_delta_by_gauge.py`; the per-product volume ratios (computed here from the
CSV); the 2x2's `K_D`/`leakance_factor` medians.

Inferred: the per-arm CPU times for a leakance-on run (scaled from the journal's
leakance-off times); that the eval zeta diagnostic is complete over each
gauge's subgraph; the literature findings, which rest on search snippets and
abstracts, not full texts.
