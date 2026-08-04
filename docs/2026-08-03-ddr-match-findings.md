# `ddr_match` physics corrections — findings

**Date:** 2026-08-03
**Branch:** `ddr-match-physics`
**Run:** `.ddrs/runs/2026-08-03T13-11-00Z-train-and-test`

## Summary

Three defects were found in the Muskingum-Cunge core, two were corrected behind
`params.ddr_match: bool` (default `true`, preserving DDR parity), and the third
was instrumented. The first run using the corrected physics produced **the
largest margin over the summed-Q' baseline this project has recorded** —
+0.0296 median NSE, three times the previous best — together with the most
physically defensible parameter field yet produced.

The result is **not attributable to the physics alone**: four variables changed
simultaneously. See §Caveats.

## The three defects

### 1. Celerity used the wide-rectangular limit (corrected)

`c = v · 5/3` is `dQ/dA` for a wide rectangular channel. The solver builds a
trapezoid (S7-S13). Correct kinematic celerity:

```
c = v · β,    β = 5/3 − (4/3)·A·√(1+z²)/(T·P)
```

Derived from `c = dQ/dA = (dQ/dy)/T`, verified against finite-difference `dQ/dA`
to 1e-6. Limits: `β → 5/3` as `b/y → ∞`; `β → 4/3` as `b → 0` at fixed `z`.

**β is not bounded below by 4/3** — it is non-monotone in `κ = b/y` and reaches
~1.07 for narrow sections. This code's channels have `κ ≈ 0.7-1.8`, giving
`β ≈ 1.30-1.36`, so the hardcoded `5/3` was **22-27% too high**.

### 2. Muskingum X was a constant, not Cunge-derived (corrected)

`X ≡ 0.3` severs the link between the scheme's numerical diffusion and the
channel's physical hydraulic diffusivity — the defining feature of
Muskingum-**Cunge**. Corrected:

```
X = clamp( 0.5·(1 − Q/(B·S·c·L)), 0, 0.5 )
```

making `D_num = c·L·(0.5−X)` equal `D_phys = Q/(2·B·S)`. On CONUS the Cunge
value is ≈0.49 almost everywhere; the constant 0.3 injected a **median 28×
excess numerical diffusion**, over-attenuating 3-6 h waves by 13-34%.

### 3. Negative discharge was clamped without measurement (instrumented)

`q_next = x_sol.clamp_min(1e-4)` at S28 silently rewrote negative solve output
to `+1e-4`, **creating mass**, hiding Courant instability, and zeroing gradients
where saturated. Now counted before the clamp.

**First measurement:**

| | mb1 | mb2 | mb3 | mb4 |
|---|---|---|---|---|
| `ddr_match: true` | 0.004% | 0.006% | 0.007% | 0.012% |
| `ddr_match: false` | 0.027% | 0.082% | 0.025% | 0.048% |

This **corrects the audit's framing**: ~70% of reaches carry a negative
Muskingum coefficient, but only ~0.01-0.05% of reach-timesteps produce negative
discharge. Negative coefficients cause an initial dip or recession oscillation —
artifacts, not divergence.

## Courant sub-stepping: attempted, abandoned, documented

Cunge `X ≈ 0.49` narrows the non-negative window `2X ≤ Cr ≤ 2(1−X)` from
`[0.6, 1.4]` to `[0.98, 1.02]`. Sub-stepping was planned to bring `Cr` in range.
**It cannot work:**

```
K spans p5 = 425 s to p95 = 18,551 s — a 44x range
best GLOBAL n_sub:      1.4% of reaches in window
ideal PER-REACH n_sub:  6.3%      <- even the unattainable ideal fails
```

Shrinking Δt globally slides every reach's `Cr` down together; it cannot compress
the spread, and integer sub-stepping quantizes too coarsely. **The correct fix is
variable Δx — subdividing reaches so `Δx ≈ c·Δt`, which is what HEC-HMS does.**
That changes adjacency topology, the CSR pattern, and per-reach parameter fields.
Out of scope; recorded here so it is not re-attempted.

This also reframes `X = 0.3`: window width is `2 − 4X`, so smaller X gives a
*wider* stability window. The constant 0.3 (width 0.8, 61.7% of reaches
admissible) is the most numerically forgiving choice available, and may have been
a deliberate stability trade rather than an oversight.

## Result

Run: 1,841 area-balanced gauges, Adam, no gradient accumulation (280 optimizer
steps in 10 epochs), `nse-batch`, disagg head OFF, `ddr_match: false`.
1.52 h train + 2.29 h eval on CPU.

```
metric     baseline       ddrs     delta
nse          0.6440     0.6736   +0.0296
kge          0.6956     0.6963   +0.0007
corr         0.8503     0.8512   +0.0010
bias         1.3969     0.4999   -0.8970
rmse        11.4719    11.1571   -0.3149
fhv          5.3840    -6.0287  -11.4126
flv         52.7742    38.8074  -13.9668
```

Observations byte-identical between the two series (max diff 0.000e+00 over
10,026,921 finite cells), so this is not a join artifact.

### Area-stratified NSE

| area km² | n | baseline | ddrs | Δ |
|---|---|---|---|---|
| <1k | 841 | **0.720** | 0.686 | −0.035 |
| 1k–5k | 418 | 0.662 | **0.711** | +0.048 |
| 5k–10k | 244 | 0.490 | **0.626** | **+0.135** |
| ≥10k | 338 | 0.352 | **0.545** | **+0.193** |

The rebalanced gauge set puts **582 of 1,841 (32%)** at or above 5,000 km²,
versus 295 of 2,365 (12.5%) in `gages_3000`. That is why the pooled median moved
when it never had before — the metric finally samples basins where routing can
physically act. The small-basin loss also shrank from ~−0.10 in earlier runs to
−0.035.

### Learned parameter field (346,321 CONUS reaches)

| | median n | @floor | ρ(n, log10_uparea) |
|---|---|---|---|
| **this run** | **0.0467** | 6.56% | **+0.323** |
| 50ep `gages_3000` | 0.0402 | 0.73% | +0.076 |
| lr 1e-2 `gages_3000` | 0.0177 | **47.28%** | +0.205 |

**76.3% of reaches fall inside the NLCD natural-channel band 0.025-0.15.** All
three learnable parameters show their strongest-ever scale dependence (ρ ≈ +0.33
for `n`, `q_spatial`, `p_spatial`).

`n_mean` converged cleanly: per-epoch 0.1262 → 0.1074 → 0.0768 → 0.0612 → 0.0546
→ 0.0528 → 0.0500 → 0.0499 → 0.0485, with the LR halvings at epochs 5 and 8
arresting the descent as designed.

## Caveats

**Four variables changed at once** versus every prior run: corrected celerity,
Cunge X, disagg head off, and the rebalanced gauge set. The +0.0296 is **not
attributable to the physics on this evidence**. The matched `ddr_match: true`
control on the same gauge set is required and has not been run.

**The baseline differs** (0.6440 on this 1,841-gauge population vs 0.6754 on
`gages_3000`). None of these absolutes compare to earlier runs.

**FHV moved +5.4 → −6.0** — peaks now under-predicted ~6%. Mechanistically
consistent: corrected celerity is ~20% lower, so `K = L/c` is longer and the
router attenuates more.

**The floor fraction rose to 6.56%** from 0.73% in the 50-epoch run. Far below
the 47.3% collapse, but the pinned reaches should be checked for concentration in
small headwaters (identity-routing pressure) versus scatter (a training defect).

## Not changed (found, deferred)

- **`attribute_minimums.slope: 1e-3`** clamps 33.2% of reaches (4.03% have slope
  exactly 0). This is an **exact invariance** — scaling `n` by `√f` restores
  depth, `R` and velocity identically, because `n` and `√S` enter depth only as
  the ratio `n/√S`. Undoing it puts 94.2% of implied physical `n` inside the NLCD
  band. Requires a retrain.
- **`leakance.rs:35-36`** uses `(p·d)^q` instead of `p·d^q` — dimensionally
  incoherent, inherited from DDR. Leakance is closed/NO-GO, so informational.
- **`mmc.rs::calculate_muskingum_coefficients`** takes a parameter named
  `velocity` that is actually the celerity (dead in production).

## Verification

`compare_ddr_sandbox` ABSOLUTE MATCH (1.53e-5 m³/s) · `cunge_x` 11/11 ·
`celerity_beta` 9/9 · `sp8_gradcheck` 5/5 · `sparse_gradcheck` 1/1 · `mmc` 13/13
· `leakance_gradcheck` 16/16 · `zeta_accum` 8/8 · `cargo test --lib` 260 passed.

**Both new backward branches were verified falsifiable**, not vacuous: they FAIL
with their terms disabled (celerity rel 1.3e-1 vs 1.44e-3 passing; X rel
8.9e-2…5.7e-1 vs 2.84e-3 passing). The Cunge-X gradcheck fixture was raised from
1000 m to 5000 m reaches because at 1000 m `W ≈ 1.6` saturates the clamp on every
reach, masking `gX` to zero and letting all four tests pass with the terms
deleted.

**Config guard:** `ddr_match: false` + `use_cuda_graphs: true` is rejected at load
(`validate_ddr_match`), because `cuda_graph/geometry_kernel.rs:296` hardcodes
DDR's 5/3 and would replay a DDR forward against a corrected backward.

## Next

1. **Matched control** — same config, `ddr_match: true`. The only way to attribute
   the +0.0296.
2. **Disagg ablation** — `enabled: true` vs the `false` used here. Never run in
   any configuration.
3. **Slope floor** — the third correction, needs its own retrain.

Artifacts: `.ddrs/runs/2026-08-03T13-11-00Z-train-and-test/plots/` (23 PNGs,
three notebooks, `_make_notebooks.py`). Diagram: `.claude/PHYSICS-CORRECTIONS.md`.
