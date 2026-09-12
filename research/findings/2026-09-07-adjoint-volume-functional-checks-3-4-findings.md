# Adjoint influence map — checks 3 and 4 on the UH arm: volume functional and the "mass loss" — findings

**Plan:** validation plan of 2026-09-05 (checks 3, 4); prior: `docs/2026-09-06-adjoint-method-validation-uh-findings.md` (checks 1, 2)
**Bundles:** `experiments/adjoint-uh-w90`, `experiments/adjoint-uh-w180`, `experiments/adjoint-uh-trace` (UH retrospective arm, 8 gauges)
**Outputs:** `.ddrs/experiments/adjoint-uh-w90/2026-09-07T18-13-59Z/`, `.ddrs/experiments/adjoint-uh-w180/2026-09-07T18-14-59Z/`
(figures + `CHECK34.md` in the w180 `figures/`), `.ddrs/experiments/adjoint-uh-trace/2026-09-07T18-08-*/uh-retro/trace/*.nc`
**Code:** `volume_sens_lag_aware`, `floor_frac`, `q_prime_wet_frac`, `volume_sens_wet`, `downstream_row`, `adjoint.traces`
(`src/experiment/adjoint/{mod,output}.rs`, commits `7adbceb`…`4c57558`); `experiments/adjoint/check34_analysis.py`

**Verdict.** The "mass loss" reported by the volume functional in the population findings is **not mass loss**.
It is, in order of importance: (1) **inflow intermittency** — at every source hour where the lateral inflow
sits at the `discharge` clamp floor the gradient is exactly zero, so the time-mean volume sensitivity of an
intermittent reach cannot exceed its wet-hour fraction (0.13–0.21 at the Cannonball gauges); (2) **slow
travel** in semi-arid far-field reaches (effective 0.05–0.1 m/s), whose water is still in transit when the
window closes. Averaged over wet source hours only, the volume sensitivity has a median of **0.88–1.02 at
all eight gauges** (raw: 0.11–1.02), and the fraction of reaches below 0.5 falls from 0.39–0.82 to
0.00–0.29, the remainder being the slow far reaches. Forward pulse traces confirm: a 30-day 1 m³/s pulse
at two "zero-kernel" reaches arrives at the gauge at **97 % and 98 %**; nothing is destroyed. **Check 4
refutes the clamped-negative-solve mechanism** for wet reaches: no mass-losing reach has a clamped reach
anywhere on its flow path unless it is itself dry (own-floor reaches are dry reaches). The one genuine
defect found is not in mass but in the linearization: the 180-day window exposes **explosive
sensitivities (up to 600) at the February–March wet-up** in the Plains basins, an open item.

## 1. What was run

Same eight UH-arm gauges as checks 1–2 (Juniata pair; White River 06447000, 06449100 → 06452000;
Cannonball 06350000, 06353000 → 06354000). Volume functional over the first WY2000 seasonal window
(start 1999-10-01), warmup 5 d, tail 7 d; kernel and residual functionals as before (no full-map FD).
Windows: 90 d (2,136 h) and 180 d (4,296 h). New per-reach outputs: `volume_sens_lag_aware` (source hours
restricted so the reach's kernel lag fits inside the window), `floor_frac` (fraction of routed hours with
discharge at the clamp floor), `q_prime_wet_frac` (fraction of source hours with inflow above the floor),
`volume_sens_wet` (mean gradient over wet source hours only), `downstream_row` (flow path). Traces: +1 m³/s
sustained over window days 10–40 at six reaches, 180-day window, `dq(reach, hour)` stored.

## 2. Check 3 — truncation of the volume functional

| gauge | reaches | median `volume_sens` (raw) | median `volume_sens_wet` | frac < 0.5 raw | frac < 0.5 wet | frac < 0.5 wet, 180 d | median wet-hour fraction |
|---|---|---|---|---|---|---|---|
| 01563500 | 127 | 1.018 | 1.018 | 0.00 | 0.00 | 0.00 | 1.00 |
| 01567000 | 213 | 1.014 | 1.014 | 0.00 | 0.00 | 0.00 | 1.00 |
| 06449100 | 45 | 1.000 | 1.000 | 0.02 | 0.00 | 0.00 | 1.00 |
| 06350000 | 41 | 0.903 | 0.952 | 0.20 | 0.00 | 0.00 | 1.00 |
| 06447000 | 298 | 0.714 | 0.897 | 0.46 | 0.00 | 0.02 | 0.81 |
| 06452000 | 582 | 0.647 | 0.877 | 0.39 | 0.02 | 0.12 | 1.00 |
| 06353000 | 105 | 0.107 | 0.962 | 0.82 | 0.19 | 0.08 | 0.13 |
| 06354000 | 235 | 0.162 | 0.880 | 0.79 | 0.29 | 0.27 | 0.21 |

- **Intermittency is the dominant driver.** At the Cannonball gauges the inflow is above the floor for only
  13–21 % of source hours; the raw medians (0.11, 0.16) are the wet fraction times the transmitted
  fraction. Spearman(raw `volume_sens`, wet fraction) = 0.74 (06354000), 0.87 (06353000); raw/wet-fraction
  median ratio 0.81–0.94 in the Plains, 1.01 in the Juniata.
- **The lag-aware correction (spec follow-up) is the wrong fix.** It uses the kernel mean lag, which is
  undefined (mass 0) exactly for the intermittent reaches, so it only lowered the pooled low-sensitivity
  fraction from 0.39 to 0.26. The wet-hour average is the correct normalisation and lowers it to ≤ 0.29
  everywhere, ≤ 0.02 outside the Cannonball.
- **The 180-day control is contaminated.** Reach-mean sensitivity by source day is flat at 0.5–0.7 through
  day 140 and then swings to +10.6 and −377 (06447000) and −8.8 (06452000) around source days 150–165
  (mid-February to early March 2000): 99 of 298 reaches at 06447000 and 74 of 582 at 06452000 acquire
  `volume_sens` > 2 (max 595). These reaches carry 0.001–0.07 m³/s and are not at the floor. This is a
  blow-up of the *linearisation* at the transition from near-dry to flowing, in the regime where check 1
  found the model non-differentiable (clamp kinks). It makes the 180-day raw functional unusable as an
  aggregate in the Plains; the wet-hour statistic at 180 d (column 7) is reported but should be read
  with that caveat.
- **Residual truncation is real for far reaches.** After the wet-hour correction, 19–29 % of Cannonball
  reaches and 0–2 % of White River reaches remain below 0.5. Their hydraulic path travel times are 30–50 d
  (median 41.5 d at 06354000 vs 17.1 d for passing reaches), and the traces below show the model moving
  their water at ~0.08 m/s, so they are still in transit when the window closes.

## 3. Check 4 — mechanism of the residual loss

**Hypothesis under test:** clamped negative Muskingum solves (S28 `clamp_min(1e-4)`) destroy mass at or
downstream of the losing reaches.

**Own-floor test.** Pooled over wet reaches (window-mean inflow ≥ 1e-3 m³/s), 90-day run: loss (raw
`volume_sens` < 0.5) vs own `floor_frac` > 0.5 — 141 / 159 / 0 / 1043 (loss & floor / loss & no floor /
no loss & floor / no loss & no floor). Specificity 1.00 (every floor reach loses), sensitivity 0.47.
Own-floor reaches have low wet fractions: they are dry reaches, not clamped negative solves.

**Path test** (`downstream_row`, path-maximum `floor_frac`): among the 223 wet losing reaches not at
their own floor, **zero** have a clamped reach anywhere downstream on the path to the gauge, in every
basin. The clamp hypothesis is refuted as the cause of loss at wet reaches.

**Hydraulic travel time.** Losing reaches not at the floor have hydraulic path lags of 14.6 d (06447000),
28.6 d (06452000), 41.5 d (06354000), 21.3 d (06353000); the passing reaches 16.2 / 19.0 / 17.1 / 11.3 d.
At the White River gauges they are indistinguishable in distance and hydraulic lag from the passing
reaches, yet have kernel mass 0.00 — the signature of the inflow clamp, not of transport.

**Pulse traces** (+1 m³/s for 30 days, 180-day window, `uh-retro/trace/*.nc`):

| gauge | reach (COMID) | distance | role | delivered at gauge in 170 d | 50 % by day | 90 % by day |
|---|---|---|---|---|---|---|
| 06447000 | 74019234 | 72 km | loss (kernel mass 0, raw 0.39) | **98 %** | 29 | 41 |
| 06447000 | 74017934 | 173 km | loss (kernel mass 0, raw 0.39, hydraulic lag 9.7 d) | **97 %** | 34 | 46 |
| 06447000 | 74019402 | 259 km | control (raw 0.91) | 69 % | 46 | not reached |
| 06447000 | 74024884 | 392 km | loss (raw 0.17) | 9 % | not reached | not reached |
| 06354000 | 74007259 | 111 km | control (raw 0.91) | 92 % | 34 | 47 |
| 06354000 | 74007891 | 316 km | loss (raw 0.38, hydraulic lag 49 d) | 73 % | 54 | not reached |

The two zero-kernel reaches deliver essentially all their water; the pulse lifts every source hour above
the floor, whereas the infinitesimal perturbation the gradient measures is zeroed at the dry hours. The
far reaches deliver 9–73 %: along the 53-hop path from 74024884 the delivered fraction falls 0.94, 0.87,
0.79 … 0.09 hop by hop with no discontinuity — water in transit, not destroyed. Effective speed from the
173-km trace: 50 % arrival 24 days after injection start, ≈ 0.08 m/s, 2.5× slower than the hydraulic
estimate at the window-mean discharge.

**Mechanism verdict:** REFUTED for clamped negative solves. The residual loss is (a) the inflow clamp
zeroing gradients at dry hours and (b) slow low-flow transport in the far semi-arid field. Neither is a
routing mass-balance defect.

## 4. Consequences for earlier documents

1. `docs/2026-09-05-adjoint-influence-conus-findings.md` §3.1 and the population handoff: "genuine
   unexplained mass loss … clamped negative solves are the candidate mechanism" must be replaced by
   intermittency + slow transport; the "13–69 reaches per arm" count was computed with a window-mean
   wetness criterion that hides intermittency. The population run is being re-executed with
   `volume_sens_wet` (2026-09-07) to give the corrected per-arm numbers.
2. `docs/2026-09-04-adjoint-influence-poc-findings.md` §3.1: the "dhbv2-lumped destroys mass in a
   12-reach tributary" reading must be re-examined under that arm with `q_prime_wet_frac` before it is
   repeated; the mechanism it hypothesised is refuted here on the UH arm.
3. Spec §2.3 volume functional: define the reported statistic as the wet-hour mean; keep the raw mean as
   a diagnostic of intermittency (raw / wet ≈ wet-hour fraction).
4. The transfer coefficient in the inherited-bias decomposition (population §3.4) is `volume_sens` at the
   upstream gauge's reach; upstream gauge reaches are perennial (wet fraction 1.0 at all six here), so
   that result stands.

## 5. Open items

- The February–March explosive linearisation in the 180-day Plains windows: characterise the state
  (depth/velocity clamps, X = 0.5 cap, K ≫ Δt) at the offending hours; decide whether to mask source
  hours where |∂ΣQ/∂q'| > 5 in aggregate statistics.
- Effective low-flow celerity in the Plains (≈ 0.08 m/s from the trace) vs the hydraulic estimate
  (0.2 m/s): the discrepancy is the kernel–hydraulic mismatch of check 2 at high flow in reverse; a
  time-integrated hydraulic reference would resolve both.
- Remaining checks of the plan: 5 (anchor robustness), 6 (window/warm-up), 7 (descent direction), 8 (three
  gauges without a celerity fit).

## 6. Reproduce

```bash
target/release/ddrs --workspace .ddrs experiment adjoint-uh-w90  --backend cpu
target/release/ddrs --workspace .ddrs experiment adjoint-uh-w180 --backend cpu
target/release/ddrs --workspace .ddrs experiment adjoint-uh-trace --backend cpu
~/projects/ddr/.venv/bin/python experiments/adjoint/check34_analysis.py <w90 dir> <w180 dir>
```
