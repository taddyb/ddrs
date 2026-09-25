# Pre-registration: what the two-way leakance arm has to do

**Written:** 2026-09-17, while the run was training, BEFORE any result.
**Run:** `2026-09-17T16-38-16Z-train-and-test`
**Config:** `config/experiments/sr_n0_gamma_leakance.yaml`
**Read-out:** `experiments/leakance/validate_water_table.py <run-id>`

## Why register anything

Every leakance arm so far has been judged after the fact, on whichever number
looked best. The term has a documented history of absorbing bias while meaning
nothing, and two-way exchange makes that risk worse, not better: a source term
can manufacture water wherever the model runs short. So the bar is written down
first.

**Skill is explicitly NOT the read-out.** If NSE or KGE improves, that is the
expected behaviour of a more flexible error-absorber and is not evidence for
anything.

## The baseline to beat

Measured with the same script on the previous losing-only arm
(`2026-09-17T05-11-55Z`), 154,145 reaches, all ten head inputs as controls:

| test | prior arm |
|---|---|
| magnitude, raw correlation | +0.212 |
| magnitude, controlling for river size | +0.166 |
| **magnitude, controlling for all ten head inputs** | **+0.027** |
| sign agreement (gaining vs losing) | 60.8% |
| majority-class baseline | 63.6% |
| Matthews correlation | +0.139 |
| **regime partial correlation, all controls** | **+0.011** |

The prior arm fails both. Its learned `d_gw` also spans only p10 0.21 to p90
0.89 against a reference spanning -4.46 to 2.18, i.e. the field barely varies.

Note on the sign test for the prior arm: under `leakance_losing_only: true` a
reach the model places as gaining receives zero flux and therefore zero
gradient, so its `d_gw` was never trained. Scoring at the noise level is the
expected result, not a surprise.

## Registered predictions

**SUCCESS** requires BOTH:

1. Magnitude partial correlation against all ten head inputs **> +0.10**
   (prior: +0.027).
2. Sign agreement **above the 63.6% majority-class baseline**, AND regime
   partial correlation against all ten controls **> +0.10** (prior: +0.011).

**PARTIAL** if the sign test passes and the magnitude test does not. That
outcome would be interesting rather than disappointing: it would say the model
locates the gaining/losing boundary without pinning the depth, which is the
weaker but more identifiable claim.

**FAILURE** if the full-control partial correlations stay below +0.10 on both.
The honest conclusion then is that `d_gw` is not recovering a water table
independent of its own predictors, and that the two-way freedom bought skill
without physics.

## What would make me distrust a PASS

- Skill improving a lot while the partial correlations move a little. That
  pattern says the extra freedom went into absorbing inflow error.
- `leakance_max_rhs_fraction` binding on a large share of reach-timesteps: the
  bound would then be setting the answer rather than the physics.
- `reaches with negative discharges pre clamp` climbing above ~0.1%. The
  no-leakance baseline is 0.013-0.114%; the first two mini-batches of this run
  read 0.045%.
- The learned `d_gw` distribution staying as compressed as the prior arm's. A
  field that does not vary cannot be recovering anything, whatever its
  correlation.

## Caveat that applies even to a clean PASS

The reference field (`channel_wtd_bed_rel`, from a national groundwater model
interpolated to channels) is itself MODELLED, not measured. No per-reach
measured distribution of water-table depth below streambeds exists. A positive
result means this model agrees with another model, which is weaker than it
sounds, and should be stated that way.
