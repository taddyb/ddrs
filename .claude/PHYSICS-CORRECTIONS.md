# `ddr_match` — the two physics paths

`params.ddr_match: bool` (default **true**) selects which forward chain runs.
`true` reproduces DDR bit-for-bit and keeps `compare_ddr_sandbox` an ABSOLUTE
MATCH (invariant 1). `false` enables the corrected physics.

```
                       forward_chain_inner  (src/routing/mmc_op.rs)
                       ─────────────────────────────────────────────
  S1..S14  geometry    depth → top_width → side_slope → bottom_width
  (IDENTICAL in both)  → area → wetted_perimeter → hydraulic_radius
                                     │
                                     ▼
  S15/S16  velocity    v = n⁻¹·R^(2/3)·√S ; v_cl = clamp(v, 0.01, 15)
  (IDENTICAL in both)                │
                                     ▼
  S17  celerity        ┌─────────────────────────────────────────────┐
                       │ ddr_match = true   c = v_cl · 5/3           │  wide-rectangular
                       │                                             │  limit; +22..27%
                       │ ddr_match = false  c = v_cl · β             │  exact for the
                       │   β = 5/3 − (4/3)·A·√(1+z²)/(T·P)           │  trapezoid built
                       └─────────────────────────────────────────────┘  above
                                     │
                                     ▼
  S18  K = L / c       (same in both — already correct, uses c not v)
                                     │
                                     ▼
  S19  X               ┌─────────────────────────────────────────────┐
                       │ ddr_match = true   X ≡ 0.3 (constant)       │  D_num/D_phys
                       │                                             │  median 28x
                       │ ddr_match = false  X = clamp(               │  → Cunge:
                       │     0.5·(1 − Q/(T·S·c·L)), 0, 0.5)          │  diffusion matched
                       └─────────────────────────────────────────────┘
                                     │
                                     ▼
  S20..S23 coefficients  denom = 2K(1−X) + Δt
                         c1 = (Δt − 2KX)/denom      c2 = (Δt + 2KX)/denom
                         c3 = (2K(1−X) − Δt)/denom  c4 = 2Δt/denom
                         c1 + c2 + c3 = 1  EXACTLY  (both paths)
                                     │
                                     ▼
  S24..S27 solve        (I − c1·N)·Q_{t+1} = c2·(N·Q_t) + c3·Q_t + c4·q'
                                     │
                                     ▼
  S28  clamp           ┌─────────────────────────────────────────────┐
                       │ both paths:  q = clamp_min(x_sol, 1e-4)     │
                       │ NEW (both):  count x_sol < 0 and report     │  never measured
                       └─────────────────────────────────────────────┘  before
```

## Why the corrections are coupled

Muskingum non-negative coefficients require `2X ≤ Cr ≤ 2(1−X)` with
`Cr = Δt/K`. Measured on CONUS at mean flow: **69.8% of reaches fall outside**
the `X = 0.3` window `[0.6, 1.4]` (28.4% give `c1 < 0`, 41.4% give `c3 < 0`).

Cunge `X ≈ 0.49` almost everywhere, which **narrows** the admissible window to
roughly `[0.98, 1.02]`. So enabling Cunge X without sub-stepping makes the
Courant violation worse, not better. **Tasks 4 and 5 must land together.**

```
   Cr window vs X                 0        0.6      1.0      1.4        2.0
   X = 0.3   [2X, 2(1-X)]         |---------[=========|=========]--------|
   X = 0.49  [2X, 2(1-X)]         |--------------[====|====]-------------|
   measured Cr  p25 0.54 ── median 1.09 ── p75 2.46 ── p95 10.2
```

## Blast radius

| Change | Forward | Backward | Breaks parity | Gate |
|---|---|---|---|---|
| Task 1 flag | plumbing only | none | no | config test |
| Task 2 counter | read-only | none | no | none |
| Task 3 β | S17 | new gβ → gA, gT, gP, gz | yes (flag-gated) | `celerity_beta_gradcheck` |
| Task 4 Cunge X | new S19 | new gX → gq_t, gT, gc | yes (flag-gated) | `cunge_x_gradcheck` |
| Task 5 sub-step | timestep loop | tape depth ×n_sub | yes (flag-gated) | `substep_courant` |
