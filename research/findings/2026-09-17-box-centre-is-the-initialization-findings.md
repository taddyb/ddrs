# The box centre is the initialization

**Date:** 2026-09-17
**Branch:** `leakance-gamma-coexist`
**Status:** measured, single-variable, reproduced

## Summary

Widening a parameter's range in `params.parameter_ranges` does not merely extend
what the model can represent. Because the KAN head ends in a sigmoid with a
near-zero output bias, every parameter **starts training at its box centre**, so
widening a box relocates the starting point. For a parameter in
`log_space_parameters` that centre is the GEOMETRIC mean, which moves much
faster than the arithmetic one.

This caused a CONUS run to diverge to NaN on its first optimizer step, and the
same mechanism would have silently disabled a second parameter.

## The measurement

Run `2026-09-17T15-47-11Z-train-and-test` (leakance + learned `n(d)`, 2,365
gauges) produced a finite loss through all four micro-batches of `mb=0`
(0.513, 0.399, 0.421, 0.810) and NaN for every value after the first optimizer
step. Finite forward, non-finite gradient.

Re-run with the `K_D` box as the ONLY difference, same seed:

| `K_D` box | geometric centre = init | initial loss | reaches negative pre-clamp |
|---|---|---|---|
| `[1e-8, 1e-6]` | 1e-7 | 0.149 | 2.9% |
| `[1e-8, 1e-3]` | 3.16e-6 | 0.513 | **30.8%** |

`zeta` is linear in `K_D`, so a 31.6x larger starting conductance removed
roughly 31.6x more water at epoch zero. That drove 30.8% of all reaches
negative, the S28 `clamp_min(discharge_lb)` rewrote every one of them, and the
gradient through that many saturated clamps went non-finite immediately.
`grad_clip_max_norm: 1.0` does not save this: a NaN gradient norm yields a NaN
scale factor, which propagates rather than bounds.

The narrow-box variant carried BOTH the widened gamma box `[-0.2, 0.85]` and
the new disconnection cap and was stable through the optimizer step, so those
two changes are exonerated by the same run.

## The same mistake, three parameters

| parameter | proposed box | centre it would have started at | consequence |
|---|---|---|---|
| `K_D` | `[1e-8, 1e-3]` | 3.16e-6 (geometric) | 30.8% of reaches negative, NaN on step 1 |
| `d_gw` | deep, e.g. `[-80, 1]` | −39.5 (arithmetic) | below `-M`, where the disconnection cap sets `∂zeta/∂d_gw ≡ 0`, so the parameter would never have moved |
| `gamma` | `[-0.2, 0.85]` | 0.325 | harmless, and closer to the literature values than the old 0.25 |

The fix in both live cases is to choose the box so its centre lands where
training should start, rather than accepting whatever the centre falls out as:

- `K_D: [1e-9, 1e-5]` — geometric centre exactly 1e-7, the historically stable
  init, four decades of span, covers the measured streambed range
  (8.8e-8 to 2.1e-5 1/s, Abimbola at M = 1 m).
- `d_gw: [-2, 1]` with `leakance_bed_thickness: 1.0` — centre −0.5, which is
  connected (`d_gw > -M`), so `d_gw` receives gradient at epoch 1.

## A second, deeper finding

**Nothing bounds `zeta` by the water actually in the reach.** `leakance_losing_only`
constrains the SIGN of the exchange (no gaining reaches) but never its
MAGNITUDE, so a sufficiently large conductance removes more water than the
reach contains. The routing then produces negative discharge and the S28 clamp
manufactures mass to hide it.

This was unobservable for the entire history of the feature: the
`log_space_lower` bug (fixed 2026-09-16) collapsed `K_D`'s box to a span of
~1e-8 with the log range inverted, so `K_D` was a frozen constant near 1e-7 in
every leakance run ever executed. Unfreezing it is what exposed the flaw.

Not fixed. Currently bounded only by choosing a box ceiling that keeps the
model out of trouble. The principled fix is to cap `zeta` at a fraction of the
available RHS (`c2·i_t + c3·q_t + c4·q'_t`), which is a forward and backward
change in invariant-4 code.

**Instrument:** `reaches with negative discharges pre clamp` in the run log.
The no-leakance rate is 2 to 4%. A sustained climb well above that band means
the missing bound is being hit.

## Consequences for reading any learned parameter

A sigmoid head with near-zero output bias reports the box centre when it has
learned nothing. So a learned value AT the box centre is not evidence of
anything, and the `|u_med - 0.5|` statistic must be reported alongside any
parameter value. This was already known (2026-09-17 parameter-range audit) but
the failure above makes the converse point too: the centre is not a neutral
observation post, it is the state the optimizer must climb out of, and it can
be a state the model cannot survive.

## Related

- `research/findings/2026-09-17-parameter-range-audit.md` — the box
  recommendations this session acted on; its `K_D` recommendation of
  `[1e-8, 1e-3]` is the one that diverged, and should be read with this
  centre constraint attached.
- `src/routing/utils.rs::log_space_lower` — the denormalize bug that froze
  `K_D`, fixed 2026-09-16 (PR #46).
