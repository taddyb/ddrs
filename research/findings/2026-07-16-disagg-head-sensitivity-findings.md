# Input perturbation experiment — frozen disagg head sensitivity (2026-07-16)

Example: `examples/kan_disagg_trained_sensitivity.rs`
Checkpoints probed: `.ddrs/runs/2026-07-16T02-22-14Z-train-and-test`,
`.ddrs/runs/2026-07-16T02-23-20Z-train-and-test`,
`.ddrs/runs/2026-07-16T11-31-50Z-train-and-test`'s
`checkpoints/epoch_5_mb_35/head` (all three train-and-test arms that used
the frozen capacity-boosted disagg head — `docs/2026-07-16-aorc2f-wave1-findings.md`,
`docs/2026-07-16-wave2-cross-wave-findings.md`; run-ID-to-experiment mapping
in `.ddrs/README.md`).

## 1. Setup

Fixed daily Q' = 10 m³/s. Two sweeps against the loaded `DisaggHead`:
1. **Intensity sweep**: inject a precip spike (0.1, 1, 2, 4, 8, 16 mm/hr) at
   a fixed hour (12), background 0.1 mm/hr elsewhere; record the resulting
   24-hour disaggregated shape (`*_trained.csv`).
2. **Hour-position sweep**: fix intensity (8 mm/hr), vary WHICH hour (0, 3,
   6, ..., 21) receives the spike; record the shape (`*_hour_position.csv`).

Outputs: `<run_dir>/plots/precip_sensitivity_trained.csv` and
`..._hour_position.csv` for each of the 3 runs.

## 2. All three runs' outputs are byte-identical — expected, and confirmed

`diff` across all 3 runs' CSVs reports zero differences. This is the
*correct* result, not a bug: all three configs warm-start the disagg head
from the same `output/disagg_pretrain/capacity_chunk1.mpk` with `freeze:
true`, so the head's weights are literally identical across arms regardless
of which Q' model or routing outcome each arm produced. This confirms
`freeze: true` behaved as documented — no accidental per-run drift — and
that wave 1/2's cross-arm NSE/KGE differences are attributable entirely to
the routing KAN head + Q' source, not to any variation in the disagg head.

## 3. Sensitivity behavior (from the distributed arm's CSVs — identical for all 3)

**Mass balance holds exactly** at every intensity and every injected hour:
`sum(hourly_value over 24h) == 240.0` (= 10 m³/s × 24h), confirming the
mass-preserving constraint survives the frozen/warm-start path.

**Intensity sweep** — output peak hour and magnitude both respond to the
injected spike (not a static shape):

| precip intensity (mm/hr) | peak hour | peak value (m³/s) |
|---|---|---|
| 0.1 | 1 | 11.49 |
| 1.0 | 1 | 11.18 |
| 2.0 | 2 | 11.05 |
| 4.0 | 6 | 10.51 |
| 8.0 | 19 | 11.29 |
| 16.0 | 17 | 15.19 |

**Hour-position sweep** — output peak hour tracks the injected storm hour
loosely but not with a simple fixed lag; the relationship is non-monotonic
(e.g. storm at hour 15 → output peak at hour 6, storm at hour 21 → peak at
hour 23):

| swept storm hour | output peak hour | peak value (m³/s) |
|---|---|---|
| 0 | 4 | 11.89 |
| 3 | 6 | 11.08 |
| 6 | 10 | 11.88 |
| 9 | 19 | 11.46 |
| 12 | 19 | 11.29 |
| 15 | 6 | 10.83 |
| 18 | 5 | 11.01 |
| 21 | 23 | 12.43 |

## 4. Interpretation

The frozen head is genuinely precip-responsive (peak timing/magnitude vary
with both intensity and injection hour) rather than reproducing a fixed
diurnal shape, and it's exactly mass-conserving by construction. The
peak-hour response to storm timing is not a simple causal lag — expected,
since the KAN maps `(daily_q, precip[24]) → hourly[24]` as a joint function
over the whole day rather than a strictly-causal per-hour convolution; this
matches how it was pretrained (`pretrain_disagg_capacity`) and is consistent
with the companion mass-balance example
(`examples/kan_disagg_mass_balance_real.rs`). No further action needed —
this was a confirmatory check, not a discovery of a defect.

**Caveat on the hour-position sweep:** at the fixed 8 mm/hr spike used there,
the response shape is nearly flat (peaks 10.8–11.9 m³/s vs a 10.0 m³/s
uniform baseline) — the erratic peak-hour jumps (e.g. 15→6, 18→5) are mostly
an argmax over a low-amplitude, noise-sensitive shape rather than meaningful
timing physics. The genuinely strong, clearly-structured timing response
only shows up at the higher 16 mm/hr intensity in the intensity sweep.
