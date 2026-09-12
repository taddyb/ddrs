# Adjoint influence map — proof of concept on the Juniata pair — findings

**Spec:** `research/specs/2026-09-03-ddrs-experiment-adjoint-design.md`
**Plan:** `research/plans/2026-09-03-ddrs-experiment-adjoint.md`
**Bundle:** `experiments/adjoint/` (`experiment.yaml`, `plots.py`, README)
**Output:** `.ddrs/experiments/adjoint/2026-09-04T13-25-01Z/` (figures in `figures/`)
**Binary:** `target/release/ddrs` built from the working tree at commit `cfdd7c8`+ (this PR), cpu backend

**Verdict:** the machinery works end to end. `ddrs experiment adjoint` resolves
five trained arms, lifts the hourly lateral inflow as a gradient leaf, passes
the finite-difference gate (relative error ≤ 0.9 % at three reaches), and
produces all four figure families for the nested pair in about two minutes on
CPU. Scientifically, the five arms learned **travel times that differ by a
factor of about 2.3 on the same network** (effective celerity 0.21–0.47 m/s at
low flow), and the downstream gauge's seasonal bias decomposes into an
inherited part routed from the upstream gauge and a local remainder whose
split differs by arm. These are one-pair, one-seed numbers: illustrations of
what the instrument measures, not population results.

## 0. Correction (2026-09-05): dhbv2-dist arm replaced

The dhbv2-dist arm used below (`2026-08-17T02-03-02Z`) was trained on
`gages_3000.csv` (2,365 gauges after filters), not the
`gages_2000_area_balanced.csv` (1,841) population the other four arms share.
The bundle now points at `2026-08-09T03-05-54Z` (same store, same CSV, also
`epoch_30_mb_1`), and `run_adjoint` refuses arms with differing gauge CSVs.
Rerun output: `.ddrs/experiments/adjoint/2026-09-05T16-29-39Z/`. Corrected
dhbv2-dist numbers for Newport (other arms unchanged):

| quantity | 08-17 arm (tables below) | 08-09 arm (corrected) |
|---|---|---|
| far-reach mean lag, high / low (d) | 2.9 / 7.4 | 3.5 / 8.9 |
| effective celerity, low flow (m/s) | 0.47 | 0.40 |
| kernel mass median, high / low | 1.16 / 0.96 | 1.36 / 0.97 |
| `volume_sens` median; reaches < 0.5 | 1.010; 0 | 1.006; 0 |
| Newport mean(pred−obs) by window (m³/s) | +15.9, +20.8, +18.8, +2.4 | +15.2, +20.1, +18.8, +2.5 |
| inherited / local, inherited share | +7.3,+8.0,+7.2,+2.4 / +8.6,+12.8,+11.6,0.0; 0.43 | +6.8,+7.7,+7.2,+2.4 / +8.4,+12.4,+11.5,+0.1; 0.43 |
| `residual_attr` median | +0.014 | +0.013 |

The celerity spread across arms becomes 0.21–0.41 m/s (factor ≈ 2.0, was 2.3).
Every other conclusion stands.

## 1. What ran

| Item | Value |
|---|---|
| Gauges | 01563500 Juniata R. at Mapleton Depot (5,262 km², 127 reaches) → 01567000 Juniata R. at Newport (8,657 km², 213 reaches) |
| Arms (run id → checkpoint) | daily-lstm `2026-08-09T12-05-08Z`, hourly-lstm `2026-08-09T14-55-05Z`, uh-retro `2026-08-09T09-30-39Z`, dhbv2-lumped `2026-08-09T22-13-43Z`, dhbv2-dist `2026-08-17T02-03-02Z`; all resolved to `epoch_30_mb_1` |
| Windows | 90 days hourly (2,136 steps), warmup 5 d, tail 7 d, tau 9 (from each run's config) |
| Kernel anchors (Newport) | high 1996-01-20 (obs 2,614 m³/s), 2004-09-19 (2,410); low 2007-10-16 (13), 2002-09-17 (14) |
| Volume / residual windows | WY2000 seasonal starts 1999-10-01, 2000-01-01, 2000-03-31, 2000-06-30 |
| Backwards per gauge-arm | 9 (4 kernel, 1 volume, 4 residual) |
| Wall time | 13–19 s per arm for daily stores; 56 s for hourly-lstm; ~2 min total |

## 2. Gate

Finite-difference gate on daily-lstm / 01563500, anchor 2004-09-18, +5 % of
window-mean inflow over the 30 days before the anchor:

| reach | distance | δ (m³/s) | ΔQ adjoint | ΔQ actual | rel. err |
|---|---|---|---|---|---|
| gauge reach | 0 km | 0.0143 | 0.01669 | 0.01685 | 0.9 % |
| median | 103 km | 0.0700 | 0.29332 | 0.29248 | 0.3 % |
| farthest headwater | 204 km | 0.0497 | 0.16743 | 0.16766 | 0.1 % |

The unit-level gradcheck (`tests/adjoint_influence.rs`, 4-reach chain) also
passes. Tier C: `cargo test --release --lib` 279 passed; `compare_ddr_sandbox`
ABSOLUTE MATCH (max abs diff 1.5e-5 m³/s). No routing or sparse code changed.

## 3. Results (Newport, 01567000, unless stated)

### 3.1 Volume sensitivity ≈ 1: mass is conserved to 1–3 %

| arm | median `volume_sens` | reaches with `volume_sens` < 0.5 (of 213) |
|---|---|---|
| daily-lstm | 1.019 | 3 |
| hourly-lstm | 1.015 | 2 |
| uh-retro | 1.014 | 0 |
| dhbv2-lumped | 1.035 | 12 |
| dhbv2-dist | 1.010 | 0 |

The systematic 1–3 % excess over 1.0 is the celerity feedback, not a
truncation artifact: adding inflow raises depth, raises celerity, and moves
pre-existing water out of the finite window sooner. Low-flow kernel masses
(next section) sit at 0.90–1.00, high-flow at 1.2–2.5, consistent with that
reading.

**The dhbv2-lumped arm has a cluster of 12 reaches in one tributary
(≈70–130 km upstream) with `volume_sens` in −0.05…0.35.** Their window-mean
inflow is 0.2–1.6 m³/s, far above the `discharge` clamp floor (1e-4), so this
is not a clamped-gradient artifact. Water added there does not reach the
gauge within 83 days under this arm's learned parameters. Hypothesis: the
learned parameters put those reaches in the negative-Muskingum-coefficient
regime and the resulting negative solves are clamped, destroying mass (the
mechanism `.claude/REACH-SUBDIVISION.md` documents). Not verified here; the
adjoint locates the reaches, negative-discharge tracking on that window would
confirm.

One low-flow anchor at Mapleton Depot (2002-09-14, drought) shows exactly zero
kernel mass at the gauge's own reach while the other 126 reaches are normal.
The gauge reach's 90-day mean inflow in that window is 0.0076 m³/s; the
likely cause is the last 30 days sitting at or below the 1e-4 clamp, which
zeroes the gradient by construction. The `q_prime_mean_by_anchor` variable
was added so this can be checked per anchor window.

### 3.2 Kernel: the arms learned different travel times

Kernel-weighted mean lag for reaches more than 280 km upstream of Newport
(the Raystown/Frankstown headwaters), averaged over anchors of each kind:

| arm | mean lag, high flow (d) | mean lag, low flow (d) | effective celerity at low flow (m/s) | kernel mass median, high | low |
|---|---|---|---|---|---|
| daily-lstm | 6.2 | 13.6 | 0.26 | 2.38 | 0.98 |
| hourly-lstm | 5.0 | 8.5 | 0.41 | 1.69 | 1.00 |
| uh-retro | 3.6 | 9.1 | 0.39 | 1.45 | 0.97 |
| dhbv2-lumped | 8.4 | 17.0 | 0.21 | 2.48 | 0.90 |
| dhbv2-dist | 2.9 | 7.4 | 0.47 | 1.16 | 0.96 |

Same network, same attributes, same observations, same seed and budget; only
the inflow store differed. The lumped dHBV2 arm routes a headwater signal to
Newport in 17 days at low flow, the distributed dHBV2 arm in 7.4. The
`kernel_vs_distance` figure shows mean lag increasing linearly with distance
in every arm with arm-specific slope, i.e. each arm learned a different
effective celerity across the whole network, not a local anomaly. This is the
compensation Beven (2001) describes: the channel parameters absorbed the
timing character of each inflow source.

High-flow kernels are oscillatory at lags under 3 days (dips to −10, spikes
to +18 in the reach-summed hourly kernel) and integrate to more than 1. Two
mechanisms, both real model behaviour rather than error: negative
Muskingum coefficients at short lags (documented CONUS-wide), and the
celerity feedback on a rising limb, where earlier arrival of the flood wave
moves a large discharge across the anchor hour. Low-flow kernels are smooth
and positive beyond the first two hours.

### 3.3 Residual attribution (squared-error sensitivity)

Spec §2.3 defined the residual functional as the mean signed residual. Its
gradient is linear in the prediction, so the observations cancel and the
attribution was uniform (checked: 0.0004–0.0005 everywhere). Corrected to
the squared-error functional `d[mean (Q̄−obs)²]/dq' = (2/n)Σ(Q̄−obs)·dQ̄/dq'`,
the classical adjoint cost-function sensitivity. Sign: positive means the
reach's inflow arrives when the gauge over-predicts.

| arm | `residual_attr` median (m³/s) | sign | Newport mean(pred−obs) by window (m³/s) |
|---|---|---|---|
| daily-lstm | −0.008 | under-predicts | −6.7, −6.0, −1.2, −8.1 |
| hourly-lstm | −0.005 | under-predicts | −3.5, −5.0, −6.4, +4.9 |
| uh-retro | +0.010 | over-predicts | +9.9, +14.0, +20.3, +1.5 |
| dhbv2-lumped | −0.020 | under-predicts | −11.9, −19.2, −23.6, −8.1 |
| dhbv2-dist | +0.014 | over-predicts | +15.9, +20.8, +18.8, +2.4 |

The maps are near-uniform in sign within an arm because a single gauge's
squared error weights every upstream reach by the same residual series; the
spatial structure comes from `volume_sens` and lag. The dhbv2-lumped arm is
the exception: the 12-reach tributary cluster carries the opposite sign
(+0.02) to the rest of the basin (−0.02), because its volume sensitivity is
near zero or negative.

### 3.4 Inherited vs local bias

Downstream bias at Newport per seasonal window, split into the part routed
from Mapleton Depot (upstream gauge's mean residual × the downstream volume
sensitivity at the upstream gauge's reach, 1.00–1.03 in every arm) and the
remainder generated in the intervening 3,400 km²:

| arm | total (m³/s) | inherited | local | inherited share of Σ|total| |
|---|---|---|---|---|
| daily-lstm | −6.7, −6.0, −1.2, −8.1 | −1.3, −9.1, −7.4, −3.1 | −5.4, +3.1, +6.2, −5.0 | 0.95 |
| hourly-lstm | −3.5, −5.0, −6.4, +4.9 | +1.7, −4.2, −6.9, +5.3 | −5.2, −0.8, +0.5, −0.4 | 0.92 |
| uh-retro | +9.9, +14.0, +20.3, +1.5 | +6.9, +6.1, +8.3, +2.8 | +3.1, +7.8, +12.0, −1.4 | 0.53 |
| dhbv2-lumped | −11.9, −19.2, −23.6, −8.1 | −4.6, −15.6, −17.2, −3.6 | −7.2, −3.6, −6.4, −4.5 | 0.65 |
| dhbv2-dist | +15.9, +20.8, +18.8, +2.4 | +7.3, +8.0, +7.2, +2.4 | +8.6, +12.8, +11.6, 0.0 | 0.43 |

The LSTM arms' downstream bias is small and almost entirely inherited. The
dHBV2 and UH arms carry 10–24 m³/s biases of opposite sign (lumped under,
distributed over), and roughly half is generated between the two gauges. The
transfer coefficient is ≈1 by mass conservation, so this decomposition is
exact up to the 1–3 % volume excess. Because routing conserves volume, none
of these biases can be removed by channel parameters; they are inflow-source
biases and the routing can only reshape their timing.

## 4. Conclusions

1. The adjoint influence map is computable from the existing backward with no
   solver change, validated against finite differences, and cheap: seconds per
   gauge-arm on CPU for basins of this size.
2. On this pair the learned routing differs across inflow sources by a factor
   of about 2.3 in effective celerity. That is a direct, per-reach measurement
   of the compensation Beven predicted, on the network the arms were trained
   on.
3. Volume bias is inherited through the network almost exactly (transfer
   1.00–1.03), so the question "can the model shrug off input bias" has a
   structural answer for the volume component: no. It can only reshape timing.
4. The adjoint also finds where the model destroys mass (the dhbv2-lumped
   tributary cluster), which no gauge metric would reveal.

## 5. Next steps

- Population: replace `gauges.pairs` with the GAGES-II reference-class nested
  selection (spec §2.1). Cost scales linearly; at ~15 s per gauge-arm for
  daily stores, 300 gauges × 5 arms ≈ 6 h CPU.
- Replicate seeds per arm before any cross-arm claim (scope design B3). The
  celerity spread above is one seed.
- Verify the dhbv2-lumped mass-destruction hypothesis with negative-discharge
  tracking on the WY2000 first window.
- Normalize the squared-error sensitivity (by σ_obs or by the arm's RMSE) so
  arms with different bias magnitudes are comparable on one color scale.
- Notebook form of `plots.py` per the ddrs-eval-plots convention once the
  figures stabilize.
- Dropped: kernel at hourly-reach resolution in the netCDF (spec §2.3
  storage argument); the daily-lag × reach and hourly × reach-sum tables were
  sufficient for every figure.

## 6. Reproduce

```bash
cargo build --release --bin ddrs
target/release/ddrs --workspace .ddrs experiment adjoint --backend cpu
~/projects/ddr/.venv/bin/python experiments/adjoint/plots.py .ddrs/experiments/adjoint/<ts>
```
