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

## `enforce_positivity` — provably zero negative solves

`params.enforce_positivity: bool` (default **false**, rejected at load unless
`ddr_match: false`). Inserts two clamps at S18′/S19′ so the solve output cannot
be negative. `δ = POSITIVITY_DELTA = 1e-2`.

```
  S18′ K floor         k_raw   = length / celerity
                       k_musk  = max(k_raw, dt·(1+δ)/2)      ⇒ Cr ≤ 2/(1+δ)
                                     │
  S19′ X stability cap cr      = dt / k_musk
                       hi_a    = (1−δ)·0.5·cr                ⇒ c1 ≥ 0
                       hi_b    = (1−δ)·(1 − 0.5·cr)          ⇒ c3 ≥ 0
                       x_eff   = min(x_cunge, hi_a, hi_b)
```

### Why this is a proof, not a heuristic

`c2, c4 > 0` always. `c1 ≥ 0 ⟺ Cr ≥ 2X`; `c3 ≥ 0 ⟺ Cr ≤ 2(1−X)` — i.e. exactly
the classical window `2X ≤ Cr ≤ 2(1−X)` (0 mismatches in 200k random draws).
The solve is forward substitution in topological order,
`x[i] = b[i] + c1[i]·Σ_{j∈up(i)} x[j]`, with `q_t > 0` (S28 clamp + hotstart,
`utils.rs:97`) and `q′ > 0` (`mmc.rs:453`), so `b ≥ 0`. Induction over the
topological order gives `x ≥ 0` everywhere. ∎

**Clamp the INPUTS (K, X), never the coefficients.** `c1+c2+c3 = 1` holds for
any `(K, X)`, so clamping K and X preserves mass exactly; clamping `c3` to zero
would break it.

**δ is mandatory, not cosmetic.** At δ=0 the clamp lands exactly on `c1=0`/`c3=0`
and f32 roundoff crosses it (7,149 `c1<0`, 964 `c3<0` in a 400k f32 sweep).
Sign safety comes from the *numerator*: `dt − 2KX ≥ δ·dt = 36 s` independent of
K, ~1e5× the f32 representation error at that magnitude. The *value* of `c1` can
still be small (measured min +2.5e−6) because `denom ≈ 2K` grows without bound
for slow reaches — small, but never negative.

### Verified on real CONUS data (`src/bin/probe_courant.rs`)

Trained head, 1,841 gauges → 92,488 reaches, 2,135 hourly steps:

| | OFF | ON |
|---|---|---|
| negative solves | 55,181 / 197,461,880 (0.0279 %) | **0 / 197,461,880** |
| min c1 / min c3 | −9.99e−1 / −9.97e−1 | +2.50e−6 / +5.00e−5 |
| cells c1<0 / c3<0 | 8,081,351 / 1,351,260 | 0 / 0 |

Replicated at a second time window, on a 256-gauge batch, on the NdArray/CPU
backend, and against a second (differently trained) head — **exactly 0 in all
five**.

### The cost — larger than first estimated

Measured `Cr = dt/K` on the full network: p5 0.0142 · p25 0.0741 · **p50 0.226**
· p75 0.603 · p95 2.724. Only **7.3 %** of reach-timesteps have `Cr > 2` and get
K floored (median inflation 1.77×, p95 9.6×).

The X cap binds on **95.3 %** of reach-timesteps. Median X used falls
**0.4976 → 0.0794, a 6.3× reduction**:

| | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| X_cunge (pre-cap) | 0.330 | 0.485 | 0.4976 | 0.4997 | 0.5000 |
| X_eff (capped) | 0.0055 | 0.0219 | **0.0794** | 0.1936 | 0.398 |

So `enforce_positivity` does not merely *shade* the Cunge X of `54ec215` — it
**replaces it almost everywhere**. Numerical diffusion becomes stability-set
rather than hydraulic-diffusivity-matched. **Skill impact is unmeasured; do not
promote this flag on the positivity guarantee alone.**

> **Erratum.** An earlier version of this section and of
> `docs/superpowers/plans/2026-08-04-positivity-clamp.md` reported X falling only
> to ~0.45 at the median. That was a methodological error: `X_max(Cr)` is
> non-monotone (it peaks at `Cr = 1`), so evaluating it *at* each Cr percentile
> does not yield the percentiles of X. The tell was that the p95 entry came out
> *below* the p75 entry. The measured numbers above supersede it.

### Why sub-stepping still does not substitute

The Task-5 abandonment analysed landing *inside* `[2X, 2(1−X)]` at fixed X≈0.49.
Here the K constraint is one-sided (`K ≥ dt/2n_sub`), which shrinking `dt` does
satisfy — but it also drives `Cr` further from 1 for the ~93 % of reaches that
are already `Cr < 2`, shrinking `X_max` and making the cap bite *harder*.

### Reach subdivision does NOT substitute either — measured, 2026-08-05

> **Erratum.** This section previously argued that the measured Cr distribution
> (median 0.226 — the typical MERIT reach is ~4.4× too *long* for an hourly
> step) "reframes the variable-Δx fix": split each reach into `m ≈ 4` pieces,
> land the network at `Cr ≈ 1`, and both `X_max → 0.5` and non-negative
> coefficients follow, making `enforce_positivity` unnecessary. **It was built
> and measured. It does not work, and the underlying claim was wrong.**
>
> The claim held only at `Cr` **exactly** 1. Both coefficients are non-negative
> only inside `[2X, 2(1−X)]`, a window of width `2(1−2X)` — 80 % wide at
> `X = 0.3`, but **1.4 % wide at the measured CONUS median `X = 0.4966`**. A
> build-time piece count fixes Δx from a *reference* flow while `Cr` tracks the
> *routed* celerity, which varies severalfold within one storm. No cap lands a
> flow-varying Cr inside a 1.4 % window.
>
> Measured on 1,841 gauges with `enforce_positivity` OFF (`probe_courant`):
>
> | arm | rows | Cr p50 | Cr > 2 | **c1 < 0** | c3 < 0 | both ≥ 0 | neg solves | ms/step |
> |---|---|---|---|---|---|---|---|---|
> | off | 92,488 | 0.096 | 2.10 % | **93.0 %** | 3.93 % | 3.1 % | 0.1356 % | 1.90 |
> | cap 4 | 171,381 | 0.115 | 0.18 % | **98.75 %** | 0.33 % | 0.9 % | 0.0945 % | 2.73 |
> | cap 8 | 184,676 | 0.123 | 0.16 % | **98.79 %** | 0.31 % | 0.9 % | 0.0876 % | 2.87 |
>
> `frac c1 < 0` gets **worse**; negative solves fall only 35 % for a 2.05×
> network, 1.5× step time and **+23.9 % total channel length**.
>
> What it *does* buy: `frac Cr > 2` 2.10 % → 0.16 %, nearly eliminating
> `c3 < 0` (3.93 % → 0.31 %) — and a clamp-off control shows it is the
> short-reach **length clamp**, not the splitting, that does this. `c3 < 0` is
> the smaller population.
>
> **A second erratum, same section.** Subdivision also does not restore the
> Cunge X's dynamic range: raw `X_cunge` median moves only **0.4973 → 0.4815**
> even uncapped. `D = q/(So·c·Δx)` is small because of the
> `attribute_minimums.slope = 1e-3` floor and large top width `B`, not because
> Δx is long.
>
> Root cause of the collapsed window: the cell Reynolds number on MERIT is
> `D ≈ 0.012` — advection-dominated, so Cunge correctly returns near-pure
> translation and `X → 0.5`. This **retroactively vindicates the constant
> `X = 0.3`** as a deliberate stability trade (80 % window width), not an
> oversight.
>
> Full write-up, cost tables and the code that stays in-tree:
> `.claude/REACH-SUBDIVISION.md`. Plan of record:
> `docs/superpowers/plans/2026-08-05-reach-subdivision.md`.

## Outside the forward chain: per-gauge extraction (`outflow_idx`)

`ddr_match` also gates `collate::compress`'s `outflow_idx` — WHICH reaches are
summed to form a gauge's prediction. This is downstream of the solver, so it
does **not** affect `compare_ddr_sandbox` (which never builds `outflow_idx`).

```
  gauge 01457000        73006562 ──┐
  (the 26-gauge case)              ├──> 73005764  (gauge reach, 250.1 km²
                        73006585 ──┘                 = 68% of the 366.8 km² basin)

  ddr_match = true    outflow_idx = upstream cols  [73006562, 73006585]
                      → drops the gauge reach's own local drainage
                      → predicted 1.58 vs observed 7.60, summed-Q' 7.38
                        (0.215x, constant across all 15 eval years)

  ddr_match = false   outflow_idx = [73005764]  — the gauge reach itself
                      → mass-conserving: the MC solve there already carries
                        everything upstream PLUS its own lateral inflow
```

A USGS gauge measures all drainage above it and we do not know where along its
reach it sits, so `false` is the physical answer. `true` reproduces DDR's
`geodatazoo/merit.py:226-234`; DDR's Lynker path validates `outflow_idx`
against the flowpath `toid` column (`lynker_hydrofabric.py:239-250`), the MERIT
path does not — that is where this would have been caught upstream.

Impact: 26 of 1841 gauges below 0.5x baseline (all small basins, 139-453 km²,
3-9 reaches; median ratio over all gauges 0.952). The omitted mass is always
positive, so `true` biases EVERY ddrs-vs-baseline comparison against ddrs.
Gate: `tests/gauge_mass_conservation.rs` (steady-state mass check, both flag
values) + `collate.rs::outflow_idx_includes_the_gauge_reach_when_not_ddr_match`.

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
| `outflow_idx` | extraction only (not the solver) | none | no (sandbox untouched) | `gauge_mass_conservation` |
| `enforce_positivity` | S18′ K floor, S19′ X cap | gk_musk mask + new cr→k path; XGrads masked by mask_cunge | no (default off; requires `ddr_match: false`) | `positivity_clamp` |
| `params.subdivision` | build-time graph only (larger N, `q'/m`, KAN gather, hot-start divisor) | **none** — no gradient path | no (default off) | `subdivide`, `subdivision_integration`, `adjacency_parity` |

## CUDA backend coverage

Every gate above declares `type I = NdArray<f32>` — **CPU only**. Since
`ddr_match: false` + `use_cuda_graphs: false` + a CUDA backend is a legal and
actively-used configuration, `tests/cuda_backward_parity.rs` re-runs the same
gradchecks with `burn_cuda::Cuda<f32, i32>` as the inner backend:

```bash
cargo test --features cuda --test cuda_backward_parity
```

Run it alongside `positivity_clamp` / `cunge_x` / `celerity_beta` on any change
to `src/routing/mmc_op.rs`. What it covers:

* native central-difference gradcheck **on CUDA** for β, Cunge X and the
  positivity clamp (correctness, not just CPU agreement);
* CUDA-vs-CPU analytic gradient parity, plus a zero-pattern assertion that
  catches a mask disagreeing across backends;
* the transitive guard `enforce_positivity ⟹ !use_cuda_graphs`, which nothing
  else asserts — it falls out of `validate_enforce_positivity` and
  `validate_ddr_match` separately, and which one fires depends on `ddr_match`.

Measured 2026-08-04 on an RTX 4080 SUPER (driver 610.43.02, nvcc 13.2, burn
0.21 fork `a033dc8`, cubecl `d562ab9`), `sparse_solver: cpu`:

* forward `x_sol` is **bit-identical** CPU vs CUDA on the 10-reach fixture
  (max abs diff 0.0, both clamp settings) — the S1..S23 geometry chain's
  `powf`/`sqrt`/`recip`/`min_pair` agree exactly at these operating points;
* analytic gradients differ by at most **2.784e-7 relative** (~2 f32 ulp),
  from the extra transcendentals the backward adds (`ratio.log()` in B6);
* CUDA analytic-vs-FD worst relative error **7.5e-4** (β + Cunge X + clamp).

Falsification results (each mutation applied to `mmc_op.rs`, run, reverted):

| Mutation | CUDA FD gradcheck | CPU↔CUDA parity |
|---|---|---|
| drop `∂β/∂z` | FAIL, rel 5.4e-2 (all 8) | passes |
| drop `∂X/∂B` | FAIL, rel 9.4e-2 … 1.6e0 (all 8) | passes |
| invert the B18′ K-floor mask | FAIL, rel 1.0 (4 clamp cases) | passes |
| swap the B19′ `hi_a`/`hi_b` tie-break | FAIL, rel 7.8e-1 (4 clamp cases) | passes |
| drop the new `x_eff→cr→k_musk` term | FAIL, rel 3.9e-1 (4 clamp cases) | passes |

The right-hand column is not a defect: both backends run the same source, so a
physics error is invisible to a cross-backend comparison **by construction**.
The FD gradcheck is the falsifier for wrong physics; the parity test is the
falsifier for a backend-specific divergence. Do not treat either as a
substitute for the other.
