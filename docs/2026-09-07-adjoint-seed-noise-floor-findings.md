# Seed-to-seed noise floor of the adjoint influence-map statistics (UH arm, seeds 42 and 43) — findings

**Bundle:** `experiments/adjoint-uh-seeds` (arms `uh-seed42` = `2026-08-09T09-30-39Z`, `uh-seed43` =
`2026-09-07T18-45-57Z-train-and-test`; both UH retrospective inflow, `gages_2000_area_balanced.csv`, 30 epochs,
`epoch_30_mb_1`; identical config except `seed`).
**Output:** `.ddrs/experiments/adjoint-uh-seeds/2026-09-07T20-48-11Z/` (figures + `SEED_NOISE.md` in `figures/`).
**Script:** `experiments/adjoint/seed_noise.py <seed-run> <cross-arm-run> --out <dir>` (reuses `plots.py` definitions).
**Cross-arm reference:** `.ddrs/experiments/adjoint-conus/2026-09-07T18-18-39Z/` (5 arms, same 41 gauges).
Gate: finite-difference check passed on both arms; 41 of 41 gauges completed per arm; cpu; 203 s per arm.

**Question.** The population study found the five inflow arms disagree on effective channel celerity by a factor
of about 2.2 at the median gauge (max/min over arms). Is that a property of the arms, or of the training run?

**Answer.** It is a property of the arms. Retraining the same arm with a different seed moves per-gauge celerity by
a median factor of 1.09 at low flow and 1.01 at high flow (IQR about 0.14 in the ratio), against a cross-arm
max/min ratio of 2.16 (low) and 2.32 (high). The two log-ratio distributions are almost disjoint
(`figures/seed_noise.png`, middle panel). The population celerity result moves from INCONCLUSIVE to
**SUPPORTED at the population level**: the inflow source, not the seed, determines the routing speed the head learns.

| statistic (38 gauges with a celerity fit) | seed 43 / seed 42 | cross-arm max/min (5 arms) |
|---|---|---|
| celerity, low anchor: median ratio, IQR | 1.09, 0.14 | 2.16, 0.69 |
| celerity, high anchor: median ratio, IQR | 1.01, 0.15 | 2.32, 0.87 |
| max abs log10 ratio, low / high | 0.19 / 0.65 | 0.57 / 1.10 |

Mass-type statistics are seed-stable at the median to 1e-5 or better: kernel mass at the gauge reach, median
wet-hour volume sensitivity, the upstream→downstream transfer, and the inherited share (20 downstream gauges) all
have median seed differences below 1e-4 with IQR below 3e-3. Their mass-conservation interpretation (transfer 1.00,
inherited share set by network position) is not a seed artefact either.

## Caveats

1. **Two seeds of one arm.** The noise floor of the LSTM-driven and dHBV2 arms could differ; the UH arm is the
   smoothest inflow and likely the quietest. Extending to one replicate per arm is the direct fix.
2. **A small systematic low-flow shift.** Seed 43 is faster than seed 42 at low flow at the median gauge (ratio 1.09,
   about 0.7 IQR). Two seeds cannot tell a systematic offset from noise; a third would.
3. **Outliers.** One high-anchor gauge changes celerity by a factor 4.5 between seeds (max |log10| 0.65) and
   06350000 (Plains) changes kernel mass at the high anchor from 1.09 to 2.62. These are the intermittent, slow
   transport cases identified in checks 3 and 4, where the linearization is fragile; they do not move the medians.
4. **Three gauges have no celerity fit** in either seed (02017500, 08377900, 09492400: three upstream reaches,
   below the five-reach minimum of the origin fit). Shared limitation, not seed noise.

## Consequence for the landscape study

The landscape spec uses `ε_L` = seed-to-seed loss difference at the gauge as the behavioural tolerance. Both
seeds are now available as arms; the Juniata bundle can be re-run with `uh-seed43` added and `c_k` of seed 43
read in seed 42's eigenbasis. That replaces the placeholder 5 % tolerance.

## Reproduce

```bash
target/release/ddrs --workspace .ddrs experiment adjoint-uh-seeds --backend cpu
~/projects/ddr/.venv/bin/python experiments/adjoint/seed_noise.py \
  .ddrs/experiments/adjoint-uh-seeds/<ts> .ddrs/experiments/adjoint-conus/2026-09-07T18-18-39Z \
  --out .ddrs/experiments/adjoint-uh-seeds/<ts>/figures
```
