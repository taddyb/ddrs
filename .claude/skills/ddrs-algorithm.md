---
name: ddrs-algorithm
description: The Muskingum-Cunge routing math implemented by ddrs — trapezoidal channel geometry (Leopold-Maddock), Muskingum coefficient derivation, the sparse linear system per timestep, why the whole chain is differentiable.
output: algorithm.md
sources:
  - src/geometry.rs
  - src/routing/mmc.rs
  - src/routing/mmc_op.rs
---

# ddrs-algorithm

> Canonical agent-readable skill. Published chapter at `docs/algorithm.md`
> is regenerated from this file by `/regenerate-docs`.

## What to know

`ddrs` solves the Muskingum-Cunge routing equation per reach per timestep
on a CONUS-scale river network. The physics is fixed (trapezoidal channels,
Leopold-Maddock hydraulic geometry, Manning's equation, the standard
4-coefficient Muskingum update). What is **learned** is the spatial
parameter field `(n, q_spatial, p_spatial)` — a per-reach Manning's
roughness and the two Leopold-Maddock width/depth exponents — emitted by
an MLP head from catchment attributes. Every algebraic op in the per-step
chain is autograd-traced through `TimestepOp`'s analytical backward, so
gradients of any loss on routed discharge flow back to the MLP weights in
one `Backward<I, 5>` node per step (not ~33). That is what makes the
solver "differentiable" in the DDR sense and what lets ddrs match DDR
gradient-for-gradient at the f32 precision floor.

## Trapezoidal geometry

For each reach at each timestep, with discharge $Q_t$, Manning's $n$,
Leopold-Maddock exponents $(p, q)$, and slope $s$, the chain S1..S17 in
`src/routing/mmc_op.rs:580-615` (mirrored by `src/geometry.rs:37-67` for
the standalone diagnostic path) computes:

$$q_\varepsilon = q_{\text{spatial}} + 10^{-6}$$

Invert Manning's equation for depth (powf at line 591):

$$d = \max\!\left(\!\left(\frac{Q_t \cdot n \cdot (q_\varepsilon + 1)}{p \cdot \sqrt{s} + 10^{-8}}\right)^{\!\!3/(5 + 3 q_\varepsilon)}\!\!,\ d_{\text{lb}}\right)$$

Top width from the Leopold-Maddock power law (line 593):

$$T = p \cdot d^{q_\varepsilon}$$

Side slope $z$ (H:V), clamped to a physically plausible band (lines 595-597):

$$z = \mathrm{clamp}\!\left(\frac{T \cdot q_\varepsilon}{2 d},\ 0.5,\ 50\right)$$

Bottom width (line 599-601):

$$b = \max(T - 2 z d,\ b_{\text{lb}})$$

Cross-sectional area, wetted perimeter, hydraulic radius (lines 603-608):

$$A = \frac{(T + b) \cdot d}{2}, \quad
P_w = b + 2 d \sqrt{z^2 + 1}, \quad
R = \frac{A}{P_w}$$

Manning's velocity, then clamp to $[v_{\text{lb}},\ 15]$ m/s (lines 610-613):

$$v = \mathrm{clamp}\!\left(\frac{1}{n} \cdot R^{2/3} \cdot \sqrt{s},\ v_{\text{lb}},\ 15\right)$$

## Muskingum-Cunge coefficients

Celerity from the kinematic-wave assumption (S17, line 615), then storage
time $k$ from reach length $L$:

$$c = \tfrac{5}{3} v, \qquad k = \frac{L}{c}$$

With storage weight $x \in [0, 0.5]$ and timestep $\Delta t = 3600\,\text{s}$,
the standard Muskingum coefficients (S18..S23, `src/routing/mmc_op.rs:618-627`,
mirrored by `calculate_muskingum_coefficients` at `src/routing/mmc.rs:245-265`):

$$\mathrm{denom} = 2k(1 - x) + \Delta t$$

$$c_1 = \frac{-2kx + \Delta t}{\mathrm{denom}}, \quad
c_2 = \frac{2kx + \Delta t}{\mathrm{denom}}, \quad
c_3 = \frac{2k(1 - x) - \Delta t}{\mathrm{denom}}, \quad
c_4 = \frac{2 \Delta t}{\mathrm{denom}}$$

By construction $c_1 + c_2 + c_3 = 1$ (modulo f32 round-off), which is
what makes the Muskingum step conservative for the routed component.

## The sparse system

The per-timestep routing equation, when written for every reach
simultaneously with an inflow $i_t = N \cdot Q_t$ from the upstream
adjacency $N$ (CSR, topologically ordered, lower-triangular per CLAUDE.md
invariant 3), is:

$$Q_{t+1} = c_1 \cdot i_{t+1} + c_2 \cdot i_t + c_3 \cdot Q_t + c_4 \cdot q'_t$$

Since $i_{t+1} = N \cdot Q_{t+1}$ (the next-step inflow depends on the
not-yet-known next-step discharge), rearranging gives a sparse linear
system whose left-hand matrix is the identity minus a scaled adjacency:

$$(I - c_1 \cdot N) \cdot Q_{t+1} = c_2 \cdot i_t + c_3 \cdot Q_t + c_4 \cdot q'_t$$

In the code (`src/routing/mmc_op.rs:629-654`):

- **S24** — `i_t = N · q_t` via `spmv_primitive` (cuSPARSE on GPU,
  scatter-add on CPU).
- **S25** — `b_rhs = c2·i_t + c3·q_t + c4·q_prime_t` assembled
  element-wise.
- **S26** — `a_values = assemble_primitive(c1)` builds the CSR values of
  $A = I - c_1 \cdot N$ in-place over the shared `Arc<CsrPattern>`.
- **S27** — `x_sol = triangular_csr_solve(a_values, b_rhs)` runs
  forward-substitution on CPU or cuSPARSE SpSV on GPU. Both rely on the
  lower-triangular topological ordering.
- **S28** — `q_next = clamp_min(x_sol, discharge_lb)` enforces the
  non-negativity floor from `cfg.params.attribute_minimums.discharge`.

Cold start at $t = 0$ solves $(I - N) \cdot Q_0 = q'_0$ through the same
triangular path — see `src/routing/utils.rs::compute_hotstart_discharge`.

## Why differentiable

`TimestepOp` in `src/routing/mmc_op.rs` registers a single `Backward<I, 5>`
node per timestep whose **five parents**, in fixed order, are:

1. `n` — Manning's roughness (registered, learned).
2. `q_spatial` — Leopold-Maddock width exponent (registered, learned).
3. `p_spatial` — Leopold-Maddock width coefficient (registered, learned).
4. `q_t` — previous-step discharge (tape link to the prior `TimestepOp`).
5. `q_prime_t` — lateral inflow forcing (registered upstream from the
   streamflow forcing reader).

`TimestepState` saves 23 forward intermediates (depth, top_width,
side_slope, bottom_width, hyd_radius, two velocities, celerity,
$k$, denom, $c_1..c_4$, $a_\text{values}$, $b_\text{rhs}$, $i_t$,
$x_\text{sol}$, plus pre-clamp ratio, denominator, $q_\varepsilon$,
side_slope_raw, bw_raw — see `forward_saved_idx` at line 502). Backward
consumes `∂L/∂q_next`, walks S28→S1 in reverse with closed-form partials
at each algebraic step, calls into `CsrSolveOp`'s own `impl Backward` for
the triangular-solve adjoint (`src/sparse/mod.rs`), and pushes
accumulated gradients onto the five parent tapes. See the
`ddrs-burn-autograd` skill for the BURN-0.21 `Backward`/`Ops` recipe both
ops follow.

The net effect: tape size is O(parents + saved_state) per step instead of
the O(n²) blowup an autograd-tape unrolling of the whole sparse solve
would produce.

## Gotchas

1. **f32 precision floor.** The DDR comparison sits at ~1e-7 relative
   diff per reach. Any cast to f64/bf16 inside the timestep chain breaks
   bit-for-bit reproducibility against the reference and forfeits the
   ABSOLUTE MATCH invariant from `CLAUDE.md`.
2. **Clamps are load-bearing — and all come from
   `cfg.params.attribute_minimums`** (`src/routing/mmc_op.rs:561-564`):
   `depth_lb`, `bottom_width_lb`, `velocity_lb`, `discharge_lb`, plus the
   hard-coded `[0.5, 50]` band on side slope and `[v_lb, 15]` cap on
   velocity. Changing any of these without re-validating against DDR
   breaks the V1 invariant; they exist precisely because the unclamped
   geometry can go non-physical when the MLP head is mid-training.
3. **Gradient-exact match against DDR is the bar.** Forward parity alone
   isn't sufficient — `sp8_gradcheck` exists because a subtly wrong
   analytical backward will silently miscompute parameter updates. If
   you touch S1..S28 or any saved-state index, re-run both the forward
   and gradient checks before claiming the change works.

## Verification

- `cargo test --test sp8_gradcheck -- --ignored` — finite-difference
  check on the `TimestepOp` analytical backward (the five parents'
  gradients must agree with central differences over the chain).
- `cargo run --release --example compare_ddr_sandbox` — V1 ABSOLUTE
  MATCH (max abs < 1e-3 m³/s) against the 5-reach RAPID sandbox fixture
  exported from DDR.
