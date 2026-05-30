# Algorithm

This chapter describes the Muskingum-Cunge routing math implemented
per reach per timestep across a CONUS-scale river network. The physics
is fixed (trapezoidal channels, Leopold-Maddock hydraulic geometry,
Manning's equation, the standard four-coefficient Muskingum update).
What is learned is the spatial parameter field
`(n, q_spatial, p_spatial)` — emitted by an MLP head from catchment
attributes — and what makes the whole thing differentiable is that
every algebraic op in the per-step chain is autograd-traced through
`TimestepOp`'s analytical backward.

The result: gradients of any loss on routed discharge flow back to the
MLP weights in one `Backward<I, 5>` node per timestep (not ~33), and
ddrs matches DDR gradient-for-gradient at the f32 precision floor.

## Trapezoidal geometry

For each reach at each timestep, with discharge $Q_t$, Manning's $n$,
Leopold-Maddock exponents $(p, q)$, and slope $s$, the chain S1..S17
in `src/routing/mmc_op.rs` (mirrored by `src/geometry.rs` for the
standalone diagnostic path) computes the channel geometry.

A small `q_eps = q_spatial + 1e-6` is added to keep the exponent away
from a zero denominator further down (S1):

$$q_\varepsilon = q_{\text{spatial}} + 10^{-6}$$

Invert Manning's equation for depth. The combined Leopold-Maddock
power law $T = p \cdot d^{q_\varepsilon}$ for the top width plus the
trapezoidal area-of-flow gives a closed-form depth in terms of
discharge:

$$d = \max\!\left(\!\left(\frac{Q_t \cdot n \cdot (q_\varepsilon + 1)}{p \cdot \sqrt{s} + 10^{-8}}\right)^{\!\!3/(5 + 3 q_\varepsilon)}\!\!,\ d_{\text{lb}}\right)$$

The `clamp_min(..., depth_lb)` floor (from
`cfg.params.attribute_minimums.depth`) protects the downstream side
slope and bottom width formulas when the MLP head is mid-training and
the unclamped geometry would otherwise turn non-physical.

Top width from the Leopold-Maddock power law:

$$T = p \cdot d^{q_\varepsilon}$$

Side slope $z$ (H:V) — the ratio that defines how much wider the top
is than the bottom per unit depth — derived from the relationship
between top width and depth in the Leopold-Maddock geometry, then
clamped to a physically plausible band `[0.5, 50]`:

$$z = \mathrm{clamp}\!\left(\frac{T \cdot q_\varepsilon}{2 d},\ 0.5,\ 50\right)$$

Bottom width: with top width and side slope known, the trapezoidal
geometry gives the bottom width directly. Clamped to
`attribute_minimums.bottom_width` so a near-degenerate trapezoid
doesn't blow up:

$$b = \max(T - 2 z d,\ b_{\text{lb}})$$

Cross-sectional area, wetted perimeter, hydraulic radius — pure
trapezoidal area + perimeter:

$$A = \frac{(T + b) \cdot d}{2}, \quad
P_w = b + 2 d \sqrt{z^2 + 1}, \quad
R = \frac{A}{P_w}$$

Manning's velocity, then clamp to $[v_{\text{lb}},\ 15]$ m/s. The
hard upper cap on velocity prevents non-physical celerities from
locking up the downstream Muskingum coefficient computation:

$$v = \mathrm{clamp}\!\left(\frac{1}{n} \cdot R^{2/3} \cdot \sqrt{s},\ v_{\text{lb}},\ 15\right)$$

Every clamp comes from `cfg.params.attribute_minimums` — see
[Formatting inputs](usage/inputs-formatting.md) for the YAML keys.

## Muskingum-Cunge coefficients

The kinematic-wave assumption sets the wave celerity at $5/3$ of the
mean velocity (S17). Then the storage time $k$ — how long water
spends in the reach in transit — is just `length / celerity`:

$$c = \tfrac{5}{3} v, \qquad k = \frac{L}{c}$$

With storage weight $x \in [0, 0.5]$ and timestep $\Delta t =
3600\,\text{s}$, the standard Muskingum coefficients
(S18..S23, mirrored by `calculate_muskingum_coefficients` in
`src/routing/mmc.rs`) are:

$$\mathrm{denom} = 2k(1 - x) + \Delta t$$

$$c_1 = \frac{-2kx + \Delta t}{\mathrm{denom}}, \quad
c_2 = \frac{2kx + \Delta t}{\mathrm{denom}}, \quad
c_3 = \frac{2k(1 - x) - \Delta t}{\mathrm{denom}}, \quad
c_4 = \frac{2 \Delta t}{\mathrm{denom}}$$

By construction $c_1 + c_2 + c_3 = 1$ (modulo f32 round-off), which
is what makes the Muskingum step conservative for the routed
component. The fourth coefficient $c_4$ scales the lateral inflow
forcing $q'_t$ — the runoff that joins the channel between gauges.

## The sparse system

The per-timestep routing equation, when written for every reach
simultaneously with an inflow $i_t = N \cdot Q_t$ from the upstream
adjacency $N$ (CSR, topologically ordered, lower-triangular per
CLAUDE.md invariant 3), is:

$$Q_{t+1} = c_1 \cdot i_{t+1} + c_2 \cdot i_t + c_3 \cdot Q_t + c_4 \cdot q'_t$$

The trick is that $i_{t+1} = N \cdot Q_{t+1}$ — the next-step inflow
depends on the not-yet-known next-step discharge. Rearranging gives a
sparse linear system whose left-hand matrix is the identity minus a
scaled adjacency:

$$(I - c_1 \cdot N) \cdot Q_{t+1} = c_2 \cdot i_t + c_3 \cdot Q_t + c_4 \cdot q'_t$$

In the code (`src/routing/mmc_op.rs`), the SpMV-assemble-solve
sequence is:

- **S24** — `i_t = N · q_t` via `spmv_primitive` (cuSPARSE on GPU,
  scatter-add on CPU).
- **S25** — `b_rhs = c2·i_t + c3·q_t + c4·q_prime_t` assembled
  element-wise.
- **S26** — `a_values = assemble_primitive(c1)` builds the CSR values
  of $A = I - c_1 \cdot N$ in-place over the shared `Arc<CsrPattern>`.
  Diagonal slots get $1$ from $I$; off-diagonal slots get $-c_1
  \cdot N[i, j]$.
- **S27** — `x_sol = triangular_csr_solve(a_values, b_rhs)` runs
  forward-substitution on CPU or cuSPARSE SpSV on GPU. Both rely on
  the lower-triangular topological ordering — by construction, the
  diagonal of $A$ is $1$, every off-diagonal sits in a lower-row,
  lower-column position, and a single forward-substitution sweep
  computes the solution exactly.
- **S28** — `q_next = clamp_min(x_sol, discharge_lb)` enforces the
  non-negativity floor from `cfg.params.attribute_minimums.discharge`.

Cold start at $t = 0$ solves $(I - N) \cdot Q_0 = q'_0$ through the
same triangular path — see `src/routing/utils.rs::compute_hotstart_discharge`.
On a linear chain this reduces to a simple cumulative sum.

## Why differentiable

`TimestepOp` in `src/routing/mmc_op.rs` registers a single
`Backward<I, 5>` node per timestep whose **five parents**, in fixed
order, are:

1. `n` — Manning's roughness (registered, learned).
2. `q_spatial` — Leopold-Maddock width exponent (registered, learned).
3. `p_spatial` — Leopold-Maddock width coefficient (registered, learned).
4. `q_t` — previous-step discharge (tape link to the prior
   `TimestepOp`).
5. `q_prime_t` — lateral inflow forcing (registered upstream from the
   streamflow forcing reader).

`TimestepState` saves 23 forward intermediates: depth, top_width,
side_slope, bottom_width, hyd_radius, two velocities, celerity, $k$,
denom, $c_1..c_4$, $a_\text{values}$, $b_\text{rhs}$, $i_t$,
$x_\text{sol}$, plus pre-clamp ratio, denominator, $q_\varepsilon$,
side_slope_raw, bw_raw.

The backward then consumes `∂L/∂q_next`, walks S28→S1 in reverse with
closed-form partials at each algebraic step, calls into `CsrSolveOp`'s
own `impl Backward` for the triangular-solve adjoint, and pushes
accumulated gradients onto the five parent tapes. See the
[BURN autograd recipe](reference/burn-autograd.md) for the
`Backward`/`Ops`/`OpsKind` plumbing both ops follow.

The net effect: tape size is **O(parents + saved_state)** per step
instead of the **O(n²)** blowup an autograd-tape unrolling of the
whole sparse solve would produce. On a CONUS-scale network with
n=346,321 reaches and ~50 timesteps, this is the difference between
trainable and "out of GPU memory at batch 1".

## Putting it together — one timestep, end to end

Here is the chain for one reach, walked end-to-end. Given:

- Manning's $n = 0.035$.
- Leopold-Maddock $p = 21$, $q = 0.5$.
- Slope $s = 10^{-3}$.
- Reach length $L = 5{,}000\,\text{m}$.
- Storage weight $x = 0.25$.
- Previous discharge $Q_t = 100\,\text{m}^3/\text{s}$.
- Lateral inflow $q' = 0.5\,\text{m}^3/\text{s}$.
- Upstream inflow $i_t = 80\,\text{m}^3/\text{s}$ (i.e. one parent reach contributing $N \cdot Q_t$).

The chain computes:

1. **Geometry** — depth from Manning's inversion, then top width,
   side slope, bottom width, area, wetted perimeter, hydraulic
   radius, velocity.
2. **Celerity** — $c = \tfrac{5}{3} v$, then $k = L / c$.
3. **Muskingum coefficients** — $c_1, c_2, c_3, c_4$ from $k$, $x$,
   $\Delta t$.
4. **Sparse system** — assemble row $i$ of $(I - c_1 N)$ and row $i$
   of the RHS, then forward-substitute (or SpSV) to get $Q_{t+1}$.
5. **Clamp** — $Q_{t+1} = \max(Q_{t+1}, 10^{-4}\,\text{m}^3/\text{s})$.

The reverse pass uses the closed-form partial of each step (Manning
inversion → power-law top width → algebraic side slope → trapezoidal
$A, P_w, R$ → Manning velocity → kinematic celerity → Muskingum
coefficients → linear combination → sparse-solve adjoint via
$A^T \cdot \text{grad}_b = \text{grad}_{Q_{t+1}}$).

## Gotchas

1. **f32 precision floor.** The DDR comparison sits at ~1e-7 relative
   diff per reach. Any cast to f64/bf16 inside the timestep chain
   breaks bit-for-bit reproducibility against the reference and
   forfeits the ABSOLUTE MATCH invariant from `CLAUDE.md`.
2. **Clamps are load-bearing — and all come from
   `cfg.params.attribute_minimums`** (`src/routing/mmc_op.rs`):
   `depth_lb`, `bottom_width_lb`, `velocity_lb`, `discharge_lb`, plus
   the hard-coded `[0.5, 50]` band on side slope and `[v_lb, 15]` cap
   on velocity. Changing any of these without re-validating against
   DDR breaks the V1 invariant; they exist precisely because the
   unclamped geometry can go non-physical when the MLP head is
   mid-training.
3. **Gradient-exact match against DDR is the bar.** Forward parity
   alone isn't sufficient — `sp8_gradcheck` exists because a subtly
   wrong analytical backward will silently miscompute parameter
   updates. If you touch S1..S28 or any saved-state index, re-run
   both the forward and gradient checks before claiming the change
   works.
4. **`q_eps = q_spatial + 1e-6` and `denominator + 1e-8` are not
   cosmetic.** They keep the depth formula well-defined when
   `q_spatial` or the squared-slope term hits zero. Removing them
   produces NaNs that propagate through the entire backward.

## Verification

The two gates for any change to the routing-core math:

```bash
# Finite-difference check on the analytical backward.
cargo test --test sp8_gradcheck -- --ignored

# V1 ABSOLUTE MATCH against the 5-reach RAPID sandbox.
cargo run --release --example compare_ddr_sandbox
```

`sp8_gradcheck` is the gradient gate: each of the five parents'
analytical gradients must agree with central differences over the
chain. `compare_ddr_sandbox` is the forward gate: max abs diff <
`1e-3 m³/s` over the entire 5-reach × 238-step run.

If `sp8_gradcheck` fails but V1 passes, the analytical backward is
wrong and silently miscomputing parameter updates — a worse failure
than a forward mismatch because nothing in the training loop will
catch it. If both fail, the change to the chain is more fundamental.

## See also

- [Architecture](architecture.md) — module map and the per-timestep
  dataflow this chapter implements.
- [Graph objects](usage/graph-objects.md) — `CsrPattern`,
  `AValuesAssembler`, and how `setup_inputs` builds the sparse system
  the algorithm runs against.
- [BURN autograd recipe](reference/burn-autograd.md) — the
  `Backward<I, N>` plumbing that the single-node-per-timestep design
  depends on.
- [Comparing to DDR](reference/ddr-comparison.md) — the V1 regression
  details and what to inspect when it drifts.
