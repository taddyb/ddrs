# Raystown Dam sandbox: does downstream discharge identify a storage-release law?

**Date:** 2026-09-22
**Script:** `experiments/reservoir/raystown_sandbox.py` (offline, no ddrs code; reads the two-way
arm's baseline files for the observed series and the summed-Q' product)
**Outputs:** `output/raystown_sandbox/` (`raystown_S0_*.json`, `raystown_landscape.png`,
`raystown_hydrograph.png`); the S0 = 940 result is also committed as
`experiments/reservoir/raystown_S0_940.json`
**Motivation:** before porting dMC-dev's reservoir operations (PR #67/#79 review, this day), test
whether the storage law they use has a real, well-conditioned gradient at one dam where inflow
and release are both observed.

## 1. Setup

- **Inflow:** observed Raystown Branch at Saxton (01562000, 1,944 km²) plus the summed-Q' of the
  ungauged 540 km² between Saxton and the dam (baseline(01563200) − baseline(01562000)). Mean
  35.4 m³/s, of which 7.6 is the ungauged part.
- **Release:** observed Raystown Branch below Raystown Dam (01563200, 2,485 km²). Mean 34.9 m³/s.
  The summed-Q' volume ratio at this gauge over 1995-2010 is 1.006, so the inflow product is
  unbiased here and the dam is a pure timing problem.
- **Law:** dMC-dev's `_first_order_euler_storage`, reproduced exactly: `Q = Q0 (S/S0)^b`,
  `Q0 = theta · median inflow`, linearised implicit Euler, daily step. dMC's `(storage_n, alpha)`
  enter only as `b = storage_n · alpha`, so `b` is fitted as one parameter over dMC's box
  `[0.01, 3]`; `theta` over `[0.3, 3]` (dMC box `[1, 3]`). Storage spun up from `S0/2` over
  WY1996, never scored. Capacity `S0 = 940 MCM` (GRanD value from memory, bracketed ×0.5 and ×2).
- **Baselines:** pass-through (`Q = I`); linear reservoir with residence time `S0 / mean inflow`
  (zero parameters); linear reservoir with residence time fitted (one parameter).
- **Windows:** train WY1997-2001, test WY2002-2010. Daily NSE.

## 2. Result

| Model | Params | Train NSE | Test NSE | Fitted values |
|---|---:|---:|---:|---|
| pass-through | 0 | 0.650 | 0.532 | |
| linear reservoir, T = S0 / mean inflow | 0 | 0.061 | 0.057 | T = 307 d |
| linear reservoir, T fitted | 1 | **0.737** | **0.649** | T = 0.66 d |
| dMC storage law, theta = 1, b fitted | 1 | 0.135 | 0.115 | b = 3 (box wall) |
| dMC storage law, (theta, b) fitted | 2 | 0.175 | 0.151 | theta = 3, b = 3 (box corner) |
| ddrs routing, no reservoir term (census, WY2000 only) | | | 0.754 | |

Sensitivity: `S0 × 0.5` lifts the two-parameter law to 0.270 / 0.247; `S0 × 2` drops it to
0.104 / 0.089. The fitted linear reservoir is unchanged (0.649 test) because its timescale is
fitted directly.

Landscape over `(theta, b)`: no interior optimum. NSE rises monotonically toward the
`(3, 3)` corner; 1.3 % of the grid is within 0.02 NSE of the best cell and that region hugs the
box edge. Autograd at dMC's default `(1.5, 0.8)`: gradient `(−0.037, −0.059)` in
`(theta, ln b)`, Hessian eigenvalues `(−0.046, +0.020)`, a saddle. At the grid optimum the
Hessian is still indefinite `(−0.047, +0.006)`.

Hydrograph (test years WY2003-2004): the observed release tracks the inflow day by day,
with peaks clipped by a day or two and released over the following week. The fitted storage
law produces a slow seasonal wave with no events at all; the fixed-timescale linear reservoir
is a nearly flat line. The storage trace under the fitted law oscillates around and above
`S0` all year.

## 3. Reading

1. **There is a gradient, and it points at the wall.** The loss does respond to both
   parameters, but only by pushing them to the fastest response the box allows. The law's
   response time is bounded below by `S0 / (b · theta · Q_ref)`, about 45 days at the corner
   with `S0 = 940 MCM`, while the observed release responds in under a day. Capacity-as-`S0`,
   which is dMC's design, makes the law structurally unable to represent this dam.
2. **The best storage model is no storage.** A linear reservoir with a fitted 0.66-day
   timescale beats pass-through by 0.09 to 0.12 NSE and beats the dMC law by 0.5. Raystown is
   a flood-control reservoir operated near pass-through with a few days of event storage;
   its release is a function of inflow and the rule curve, not of fill fraction.
3. **Making `S0` learnable would not help.** Only the ratio `S0 / (b · theta)` sets the
   response time, so a learnable capacity buys a three-way ridge, the same equifinality
   as `(n, p, q)` and `(n, gamma)`, with no physical anchor left.
4. **The routing without a reservoir already does better** (0.754 in WY2000 in the census)
   than anything the storage law reaches, because Muskingum-Cunge attenuation over the 63
   reaches above the gauge is a closer model of "a day or two of clipping" than a
   fill-fraction law is.

## 4. Verdict and what it changes

The dMC storage law is a poor model class for an operated flood-control dam, and its two
parameters are not identifiable from downstream discharge because the law cannot get close
enough to the data for a bowl to form. This is misspecification, not the sum-observation
problem that closed leakance, but the symptom is the same: parameters at the box wall,
indefinite curvature, and a "learned" value that means nothing.

Do not port the dMC reservoir operations to ddrs. If reservoirs are ever revisited, the
model class has to be a release rule (a function of inflow, storage, and season, as the
data-driven reservoir-operation literature does), and the object set has to be GRanD dams
whose timing signature is visible below them, which at Raystown it barely is.

One dam, one law, one capacity assumption. The capacity sensitivity spans a factor of four
and does not change the ordering.

## 5. Verification (2026-09-25, after the user questioned the flat hydrographs)

`experiments/reservoir/raystown_verify.py`, three checks:

1. **Equivalence with dMC's own function.** dMC-dev's `_first_order_euler_storage` (copied
   verbatim from `methods.py` at the PR #79 merge) run step by step in torch float64 on the
   same inflow and initial state agrees with the sandbox loop to a maximum of 2e-5 m³/s over
   5,479 days at three parameter settings, with identical NSE to four decimals. The sandbox
   is the dMC law.
2. **Step response against the analytic timescale.** Constant inflow 20 m³/s for five years,
   then 60 m³/s. At the box corner (theta 3, b 3) the release e-folds in 93 days against an
   analytic linearised timescale of 62 days at equilibrium; at dMC's default (theta 1.5,
   b 0.8) it e-folds in 509 days against 608, and the equilibrium storage sits at 2.6 times
   capacity. The law is that slow at this capacity by construction.
3. **Capacity sweep.** Best train NSE on a 12 × 12 grid as `S0` shrinks:

   | S0, MCM | best train NSE | at (theta, b) | response time S0 / (b · theta · q_ref) |
   |---:|---:|---|---:|
   | 1 | 0.754 | (2.5, 0.63) | 0.4 d |
   | 3 | 0.735 | (2.5, 1.06) | 0.7 d |
   | 10 | 0.711 | (3.0, 1.79) | 1.2 d |
   | 30 | 0.682 | (3.0, 3.0) | 2.1 d |
   | 100 | 0.545 | (3.0, 3.0) | 7.1 d |
   | 300 | 0.341 | (3.0, 3.0) | 21 d |
   | 940 | 0.175 | (3.0, 3.0) | 67 d |

   Pass-through is 0.650. The law collapses onto and then past pass-through as capacity goes
   to zero, exactly as it must if the code is right, and at 1 to 3 MCM it matches the fitted
   0.66-day linear reservoir (0.737). Monotone in capacity, optimum on the box wall from
   30 MCM upward.

So the flatness is not the implementation. It is the design choice, in dMC, of total lake
volume as `S0`: the law's response time is bounded below by `S0 / (b · theta · q_ref)`, and
with 940 MCM that floor is 67 days at the edge of the parameter box. The dam's daily
operations use an effective buffer of about 2 MCM (fitted timescale 0.66 d times mean
inflow), 0.2 % of capacity. A law that used the operational buffer instead of total volume
would work about as well as a one-day linear reservoir, which is to say it would beat
pass-through by 0.1 NSE and still not represent the events. Making `S0` learnable would
recover that, but only the ratio `S0 / (b · theta)` is identifiable, so it trades one
unidentifiable pair for a three-way ridge. A second dam with a different purpose (hydropower or
irrigation, where releases are decorrelated from inflow) would be the natural extension if
the question is reopened.
