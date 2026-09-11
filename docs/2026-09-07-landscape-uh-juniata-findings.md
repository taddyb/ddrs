# Per-gauge loss landscape in (n, p, q) log-multiplier space — sample case (UH arm, Juniata pair) — findings

**Spec:** `docs/superpowers/specs/2026-09-07-adjoint-landscape-design.md`
**Bundles:** `experiments/landscape-uh-juniata` (box ±ln 3), `experiments/landscape-uh-juniata-wide` (box ±ln 10; arms uh-seed42 + uh-seed43)
**Outputs:** `.ddrs/experiments/landscape-uh-juniata/2026-09-07T19-04-07Z/`,
`.ddrs/experiments/landscape-uh-juniata-wide/2026-09-07T19-13-30Z/` (seed 42 only),
`.ddrs/experiments/landscape-uh-juniata-wide/2026-09-07T21-15-36Z/` (both seeds; `figures/LANDSCAPE.md`, `SEED_COMPARE.md`)
**Code:** `src/experiment/landscape/` (`97d362b`, `054d47e`), `experiments/landscape/plots.py`
**Arm:** UH retrospective, seed 42 (`2026-08-09T09-30-39Z`, `epoch_30_mb_1`); objective = NSE-batch training loss
restricted to the gauge, mean over the four WY2000 seasonal 90-day windows; cpu; ~4–6 min per gauge.

**Verdict (instrument): PASS on the sample.** The FD Hessian is symmetric to machine precision, positive
definite at the interior optimum found for Newport, the Newton descent converges (loss path monotone,
flat at the end), and `L(α*) ≤ L(0)` at both gauges. The stiff eigenvector aligns with the hydraulic
travel-time direction at |cos| = 0.65, the remainder being the width term's effect on attenuation
(Cunge X), so criterion (iii) is met in the sense intended: the gauge constrains a celerity-like
combination. One instrument lesson: when the multiplier box exceeds the parameter ranges the fields clamp
and the landscape flattens artificially (Mapleton Depot, below); the box must be limited by the ranges.

**Hypothesis under test (user framing).** Each gauge's loss under large-batch training is a batch compromise, not
the gauge's own minimum. The adjoint landscape over (n, p, q) at a gauge measures how far, and in which directions,
the batch solution sits from the gauge's optimum, which is the test of whether the model is being trained correctly;
the same map under different inputs tests whether the inputs move the optimum or only the batch solution.

**Verdict (science, one arm, one seed):** at Newport the large-scale training sits **on the gauge's stiff
axis** (offset 0.00 half-widths) and **1.8 log-units along a sloppy axis** (0.86 of the 5 % behavioural
half-width): inside the behavioural set, displaced only along a direction the gauge does not constrain.
In the hypothesis's terms: the batch compromise at Newport costs 0.08 NSE against the gauge's own optimum, and all of
that cost lies along directions the gauge does not constrain. Along the one direction it does constrain, large-batch
training reached the gauge's optimum. This is selective equifinality measured directly. Both gauges would be routed ~3× faster than the
large-scale training does (n × 0.36), gaining NSE 0.69 → 0.77 (Newport) and 0.55 → 0.66 (Mapleton).

## 1. Newport (01567000, 213 reaches)

Optimum (±ln 10 box, interior, 0 % of reach-parameters clamped): multipliers **n × 0.36, p × 0.26,
q × 0.56**; loss 0.0518 → 0.0454; NSE 0.692 → 0.770. Hessian at α*:

| k | λ_k | v_k = (n, p, q) exponents | |cos(v_k, ∇travel-time)| | trained-point coordinate c_k | 5 % half-width w_k | |c_k| / w_k |
|---|---|---|---|---|---|---|
| 1 (stiff) | 2.34e-1 | (+0.82, −0.52, −0.24) | 0.65 | −0.00 | 0.14 | **0.03** |
| 2 | 1.03e-3 | (+0.57, +0.76, +0.30) | 0.75 | +1.80 | 2.10 | **0.86** |
| 3 (sloppy) | 2.38e-4 | (+0.02, −0.38, +0.92) | 0.10 | +0.04 | 4.37 | 0.01 |

Hydraulic travel-time gradient direction at the trained point: (n +0.97, p +0.20, q +0.16): travel time
is set by n, as the Manning relation implies (v ∝ n^−0.7 at fixed Q). Anisotropy λ₁/λ₃ ≈ 1,000.

Reading: the gauge fixes one combination, roughly `n^0.8 p^−0.5 q^−0.25` (raise roughness, narrow the
channel), which changes both celerity and attenuation. Along it the large-scale training is at the
optimum. The whole 0.08 NSE gain available to Newport lies along `v₂ ∝ n^0.57 p^0.76 q^0.30` (make
everything smaller together), a direction with 200× less curvature: the valley floor. The landscape slices
(`figures/landscape_01567000.png`) show a single near-vertical valley in n, broad in p and q, with a
p–q saddle at fixed n (high NSE at low-p/low-q and at high-p/high-q).

## 2. Mapleton Depot (01563500, 127 reaches)

In the ±ln 3 box the optimum is at the corner (n × 0.36, p × 0.33, q × 0.33; NSE 0.548 → 0.651;
0 % clamped). In the ±ln 10 box the descent runs to n, p, q × 0.1 with **100 % of reach-parameters
clamped at the range floors** (n ≥ 0.015, p ≥ 1, q ≥ 0) and only +0.01 further NSE; the eigenvectors
there are meaningless (n fully clamped ⇒ zero derivative). The loss is therefore monotone decreasing in
the "all smaller" direction down to the physically allowed floor: no interior optimum exists for this
gauge within the parameter ranges. Hydraulic direction (0.95, 0.27, 0.17); ln-3 stiff vector
(0.82, −0.56, −0.12), |cos| = 0.61; stiff/sloppy anisotropy 100:1 at the ln-3 corner.

## 2b. Seed replicate (seed 43 added as a second arm, run `2026-09-07T21-15-36Z`)

The netCDF now stores the trained per-reach fields (`n0`, `p0`, `q0`, `comid`), so seed 43's trained point can be
expressed in seed 42's multiplier coordinates as the reach-mean log field ratio Δα, and then in seed 42's eigenbasis
(`experiments/landscape/seed_compare.py`). Newport, where both seeds have interior optima with nothing clamped:

| | seed 42 | seed 43 |
|---|---|---|
| Trained fields relative to seed 42, Δα = (n, p, q), reach-mean ± reach-std | 0 | (−0.05 ± 0.04, −0.46 ± 0.04, −0.00 ± 0.05) |
| Trained point in seed 42's eigenbasis, c = (stiff, middle, sloppy) | (−0.00, 1.80, 0.04) | (0.20, 1.43, 0.20) |
| |c_k| / half-width at 5 % of L* | (0.03, 0.86, 0.01) | (1.41, 0.68, 0.05) |
| Own optimum, NSE | 0.770 | 0.772 |
| Own optimum relative to seed 42's, multipliers (n, p, q) | 1 | (0.98, 1.24, 0.44) |

Three readings.

1. **The seed difference is almost purely a width shift.** Seed 43's channels are 37 % narrower (p × 0.63) nearly
   uniformly across the 213 reaches (std 0.04 in log), with n 5 % smaller and q unchanged. The basin-uniform
   projection is faithful for this pair.
2. **The seed-defined tolerance is about 10 % of L\*, not 5 %.** Along the stiff axis the two seeds differ by
   Δc₁ = 0.20 log-units, which at λ₁ = 0.234 is a loss difference of ½λ₁Δc₁² ≈ 0.0047 ≈ 0.10 L\*. With ε_L = 0.10 L\*
   the half-widths become (0.20, 2.96, 6.2) and both seeds lie inside the behavioural set on every axis
   (seed 43 at (1.0, 0.48, 0.03) half-widths). Two seeds give one draw of this number; it is the tolerance the spec
   asked for, and it replaces the placeholder.
3. **The optimum is a valley, and two searches land at different points along it.** The two seeds' Newport optima
   agree on n\* to 2 % and on p\* to 24 %, but differ on q\* by a factor 2.3, with NSE 0.770 vs 0.772. In seed 42's
   eigenbasis the q-dominated direction is the sloppy one (half-width 4.4 log-units), so a factor 2.3 in q\* is
   0.19 half-widths: no loss consequence. The gauge determines the stiff coordinate of its optimum and nothing else.

Mapleton Depot: the seed-42 reference optimum is the clamped corner (§2), so its eigenbasis is degenerate and the
comparison is not meaningful there; seed 43 finds an interior optimum (n × 0.31, p × 0.40, q at the floor;
NSE 0.558 → 0.652) and the two seeds' NSE at optimum agree (0.661 vs 0.652). Range-bounded Newton (follow-up 1)
is required before this gauge's optima can be compared.

## 3. What this says, and does not say

- The routing that fits these two Juniata gauges best is about three times faster than the UH arm's
  large-scale training gives them (n × 0.36 at both; the trained median n across CONUS is 0.13, so the
  gauge-optimal n here is ~0.04–0.05, textbook natural-channel values). Consistent with the population
  celerity (UH arm pooled 0.26 m/s at low flow; ×3 ≈ 0.8 m/s).
- The large-scale training is not wrong along the axis the gauge can see; it is displaced along an axis
  the gauge cannot see. Whether that displacement is "wrong" is exactly what an independent geometry
  observation (width, depth, travel time between gauges) would decide; the gauge alone cannot.
- One arm, two seeds, two gauges. The seed-to-seed displacement along Newport's stiff axis sets the behavioural
  tolerance at about 10 % of L* (§2b); one draw of that number, so treat it as the order of magnitude.
- The multiplier parametrization is basin-uniform (Phase A); per-reach directions (Phase B) may reveal
  trades the gauge can make between upstream and downstream reaches.

## 4. Follow-ups

1. Bound the Newton descent and the grids by the parameter ranges (reject steps with > 5 % clamped),
   or report the box-edge optimum as "monotone to the floor" explicitly. Implement before the 8-gauge run.
2. Run the 8 validation gauges (`experiments/adjoint-uh-validate` set) on the UH arm; then both seeds.
3. Cross-arm: the five arms' trained points in each gauge's eigenbasis — the direct test of "arms agree
   on the stiff coordinate, differ on the sloppy ones".
4. Phase B per-reach Lanczos.

## 5. Reproduce

```bash
target/release/ddrs --workspace .ddrs experiment landscape-uh-juniata-wide --backend cpu
~/projects/ddr/.venv/bin/python experiments/landscape/plots.py .ddrs/experiments/landscape-uh-juniata-wide/<ts>
```
