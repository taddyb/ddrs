# `ddr_match` — the two physics paths

**DEPRECATED (2026-08-19): default flipped to `false`.** DDR ported these
corrections in DeepGroundwater/ddr#192 as a clean break (no flag), so the
corrected physics is now what BOTH implementations run by default, the sandbox
fixture is regenerated from post-#192 DDR, and `compare_ddr_sandbox` holds its
ABSOLUTE MATCH on the corrected path (verified 2026-08-19: max abs 1.5e-5 m³/s,
f32 floor). Setting `ddr_match: true` emits a deprecation WARN; the legacy path
remains only to reproduce pre-#192 results and for `use_cuda_graphs: true`
(whose captured kernel hardcodes the legacy 5/3 celerity), and will be removed.

`params.ddr_match: bool` selects which forward chain runs.
`true` reproduces PRE-#192 DDR bit-for-bit. `false` (default) enables the
corrected physics.

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
> `research/plans/2026-08-04-positivity-clamp.md` reported X falling only
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
> `research/plans/2026-08-05-reach-subdivision.md`.

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
| `geometry_form` | S2..S17 arm | early-return power-law tail | no, default trapezoid | `power_law_gradcheck`, `geometry_form_golden` |

## geometry_form: one consistent section (power_law)

`params.geometry_form` picks the channel section for Muskingum-Cunge rows.
`trapezoid` is the default and an absent key means it. Forward values and parent
gradients of the default are bitwise unchanged, except at exact clamp ties (see
the tie rule below).

### The defect on the trapezoid

The trapezoid path gets its depth from a power-law inversion with `R = d`. It then
builds a tangent trapezoid on that depth. The two sections differ, so `Q != vA`.
Over the parameter box the ratio `vA/Q` has median 0.83 and a 5 to 95 % range of
0.14 to 1.17. At p 21, q 0.65 it is 0.84 to 0.86.

The side-slope clamp makes it worse at small q. At q about 0.084 the clamp binds
above about 1.9 m depth. It flips the sign of `d ln c / dq`: +0.19 on the
trapezoid, against -0.80 for a consistent section, at Q 100 m3/s.

### The power-law section

```
T = p d^q          A = T d/(q+1)          R = d/(q+1)
d = [ Q n (q+1)^(5/3) d_ref^g / (p sqrt(s)) ]^( 3/(5 + 3q + 3g) )
v = Q / A          c = v (5 + 3q + 3g) / (3 (q+1))      (= dQ/dA exactly)
```

- The depth runs in log space. The depth floor `attribute_minimums.depth` stays.
- `dA/dd = T` holds exactly, so the celerity `beta` has no approximation.
- Cunge X uses `T`. The path has no side slope, no bottom width, and no clamps C4
  and C5.
- The velocity clamp stays as `clamp(Q/A, v_lb, 15)`. It acts as a celerity band.
  `Q = vA` holds wherever the clamp is inactive. At the depth floor `v = Q/A` is
  kept (continuity, not Manning).

### Where it applies

- Supported: `ddr_match: false` only. `use_cuda_graphs`, `use_leakance` and
  `use_reservoirs` are rejected at load (`config.rs::validate_geometry_form`).
- Stage routing is allowed. The load prints a WARN: stage rows keep their own
  section, so `geometry_form` changes Muskingum-Cunge rows only.
- The load prints an INFO line that points at this file.
- The learned `n` shifts. At fixed trained parameters the median celerity rises 3.0 %
  on Juniata and 3.6 % on Hudson (see Measured). Runs and values of `n` are not
  comparable across forms.
- Do not resume a checkpoint across forms, and do not evaluate one across forms.
  A trapezoid checkpoint loads under `power_law` with no warning, and its weights
  were trained against the other section.

Code: forward in `src/routing/mmc_op.rs` (the `geometry_form` arm of the forward
chain), backward in `src/routing/power_law.rs::section_backward`, reference
functions in `src/geometry.rs`.

### Tie rule on the MC path

A clamp bound is inactive at an exact tie, so the gradient passes. For a min of
two live operands the tie goes to the first operand (Burn `min_pair`). This
matches Burn autodiff and DDR's `torch.clamp`. The helpers are in
`src/routing/clamp_rule.rs`. Two sites read the saved clamped value before. A `>=`
mask on a clamped value would pass every clamped entry, so they now read the
pre-clamp value, rebuilt with the forward's own ops.

| Site | Mask | Tensor read | Before | After |
|---|---|---|---|---|
| B28 solve output floor | `x_sol` vs discharge floor | `x_sol` | `x > lb` | `x >= lb` |
| B18' K floor | `k_raw` vs K floor | `k_raw` | `k > floor` | `k >= floor` |
| B16 velocity band | `v_un` in `[v_lb, 15]` | `v_un` | strict `<` and `>` | `<=` and `>=` |
| B11 bottom-width floor | `bw_raw` vs floor | `bw_raw` | `bw > lb` | `bw >= lb` |
| B9 side-slope band | `z_raw` in `[0.5, 50]` | `z_raw` | strict | closed |
| B6 depth floor | `ratio^e` vs `depth_lb` | pre-clamp depth | saved post-clamp depth `> lb` | pre-clamp `>= lb` |
| B19 Cunge X band | `x_raw` in `[0, 0.5]` | pre-clamp `x_raw` | saved clamped X, strict | pre-clamp, closed |
| S19p dam-row X cap | cap vs channel X | recomputed cap and `X_r` | tie counted as the cap | cap wins only if `cap < X_r` |
| Leakance losing-only gate | `depth` vs `d_gw` | `depth` | `depth > d_gw` | `depth >= d_gw` |
| Leakance bed cap | `m_raw` vs cap | `m_raw` | `m_raw < cap` | `m_raw <= cap` |

A natural X = 0.5 rounding tie on the default path now passes the gradient.
Forward values stay bitwise. A training trajectory can move slightly from older
runs. The pins are in `tests/clamp_tie_rule.rs`.

### Measured

Seed 42, CPU, 30 epochs, committed bundles. This is a record, not a gate. One run per
cell, so differences of a few hundredths in NSE are within seed noise. The harness is
`tests/geometry_form_measure.rs` (ignored, release, manual).

Routed metrics (median over gauges on Juniata, per gauge on Hudson):

| Bundle | Gauge | Trapezoid NSE / KGE | power_law NSE / KGE | Baseline NSE / KGE |
|---|---|---|---|---|
| Juniata | median (1 gauge) | 0.790 / 0.881 | 0.836 / 0.910 | 0.695 / 0.819 |
| Hudson MC only | 01372058 Poughkeepsie | 0.145 / 0.604 | 0.208 / 0.621 | 0.144 / 0.561 |
| Hudson MC only | 01358000 Green Island | 0.781 / 0.876 | 0.810 / 0.882 | 0.688 / 0.801 |
| Hudson MC only | 01375000 Croton | 0.820 / 0.856 | 0.823 / 0.857 | 0.835 / 0.867 |
| Hudson stage | 01372058 Poughkeepsie | 0.308 / 0.625 | 0.317 / 0.623 | 0.144 / 0.561 |
| Hudson stage | 01358000 Green Island | 0.780 / 0.876 | 0.801 / 0.881 | 0.688 / 0.801 |
| Hudson stage | 01375000 Croton | 0.819 / 0.856 | 0.821 / 0.856 | 0.835 / 0.867 |

Celerity from `probe_courant` (`enforce_positivity: false` pass, rho 90, 2000 steps,
every 10th step). The paired ratio is the power_law celerity over the trapezoid
celerity, on the trapezoid run's trained checkpoint, joined on reach and step. This is
the pure form effect at fixed trained parameters. The trained-vs-trained ratio
compares the median celerity of each form's own trained run.

| Bundle | Paired ratio median | Paired ratio 5 to 95 % | Trained-vs-trained ratio |
|---|---|---|---|
| Juniata (213 reaches) | 1.030 | 1.024 to 1.102 | 1.027 |
| Hudson MC only (756 reaches) | 1.036 | 1.026 to 1.183 | 1.054 |

Median learned parameters. Juniata (from the plot output): n 0.1258 trapezoid and
0.1275 power_law, p 12.47 and 12.26, q 0.483 and 0.481. Hudson runs write no
parameter file; the last training log line gives median n: MC only 0.1258 and 0.1262,
stage 0.1268 and 0.1306.

Hudson stage run: the `stage routing: 41 of 756 reaches in the stage block` line
confirms 41 stage rows. Under power_law the section changes the 715 Muskingum-Cunge
rows only (756 minus 41). Stage rows keep the table section. Poughkeepsie sits on a
stage row, so it changes only through its Muskingum-Cunge inflows.

`probe_courant --celerity-csv <path>` writes `reach,step,celerity_m_s` for the
`enforce_positivity: false` pass.

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
