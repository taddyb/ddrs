# Leakance (water-loss term) × hourly disaggregation — feasibility design (2026-06-29)

## Purpose

Test one hypothesis: **the hourly disaggregation signal makes DDR's reverted
leakance (GW–SW water-loss) term identifiable and helpful, where daily forcing
made it neither.**

This is a **feasibility + experiment-design** spec, not a product commitment.
It defines (a) the minimal, flag-gated leakance port needed as a testbed and
(b) a pre-registered experiment with a go/no-go gate. A GO result justifies a
later full port; a NO-GO cheaply falsifies the idea against a fixed bar.

## Background

### The water-loss term = leakance

In DDR (`~/projects/ddr/src/ddr/routing/mmc.py`, commit `c2bd0f9`,
`_compute_zeta`), the GW–SW exchange enters routing as:

```
numerator   = q_t · n · (q_spatial + 1)
denominator = p_spatial · sqrt(s0)
depth       = (numerator / (denominator + 1e-8)) ^ (3 / (5 + 3·q_spatial))
width       = (p_spatial · depth) ^ q_spatial
area        = width · length
zeta        = leakance_factor · area · K_D · (depth − d_gw)

b           = c2·I + c3·Q + c4·Q' − zeta        # subtracted from routing RHS
```

`depth` is a function of **instantaneous discharge** `q_t`. Positive `zeta` =
losing stream (water leaves channel → groundwater); negative = gaining stream.
Reference: Song et al. (2025), Eq. 12–14.

Learned spatial parameters (denormalized from the head, DDR ranges verbatim):

| param | range | meaning | space |
|---|---|---|---|
| `K_D` | `[1e-8, 1e-6]` | hydraulic exchange rate (1/s) | log |
| `d_gw` | `[-2.0, 2.0]` | groundwater depth threshold (m) | linear |
| `leakance_factor` | `[0.0, 1.0]` | gating/scaling factor (-) | linear |

### Why it was reverted in DDR

On daily forcing, leakance was **unidentifiable and did not help metrics**: the
learned exchange collapsed to **sub-0.01 m³/s** net flux — physically negligible
values that added nothing — so it was reverted (the `Revert "..."` chain on DDR
master, originals #130/#131/#133/#135).

### Why the hourly signal might fix it

`zeta ∝ (depth − d_gw)` and `depth = f(q_t)` is **convex** in discharge. Daily
forcing flattens `q_t`, compressing the depth range and averaging away the
high-flow regime where leakance is largest. The hourly disaggregation head
(`src/nn/disagg_head.rs`, precip-driven, mass-preserving) restores sub-daily
depth swings, re-exposing that regime and giving `K_D`/`d_gw` a gradient with
real dynamic range. There is also a clean division of labor: the disagg head is
**mass-preserving within a day**; leakance is the **only** deliberately
mass-*reducing* term — and the summed-Q′ baseline over-predicts downstream
(gage ratio ~1.05), a losing-stream signature and a KGE-β lever neither the loss
function nor geometry tuning has moved (a *structural* ceiling per
`docs/2026-06-23-precip-disaggregation-findings.md`).

## Part A — The testbed (minimal, flag-gated leakance port)

Approach: a faithful port of DDR's `zeta`, gated behind
`experiment.use_leakance` (default `false`). Default behavior is byte-identical;
leakance code paths exist only when the flag is on.

### Dataflow

```
KAN head (per reach)                 hourly disagg head
  n, p, q, X  ──┐                      daily Q' ──► hourly Q'  (precip-driven,
  K_D, d_gw, ───┤                                    mass-preserving)
  leakance_factor                                        │
                │                                        ▼
                │              route_timestep (per hour, per reach):
                │                depth = f(Q_t, n, p, q, s0)   ← instantaneous Q_t
                └──────────────► zeta  = leakance_factor · area · K_D · (depth − d_gw)
                                 b     = c2·I + c3·Q + c4·Q' − zeta
                                 Q_t+1 = triangular_sparse_solve(A, b)
```

### Components touched (blast radius)

| File | Change | Invariant guard |
|---|---|---|
| `src/config.rs` | add 3 param ranges + log-space flag for `K_D`, verbatim from DDR | additive; off by default |
| `src/nn/kan_head.rs` | head output `P → P+3` **only when `use_leakance`** | default head byte-exact ⇒ KAN parity (#5/#7) pass |
| `src/routing/mmc.rs` | port `_compute_zeta`; subtract `zeta` from `b` in `route_timestep` | `use_leakance=false` ⇒ RHS identical ⇒ DDR sandbox (#1) ABSOLUTE MATCH |
| `src/training/loss.rs` | none — `zeta` acts through routed Q | — |

### Invariants preserved

- **#1 DDR sandbox** — leakance off → `b` unchanged → still ABSOLUTE MATCH
  (`< 1e-3 m³/s`). Re-run `examples/compare_ddr_sandbox` after the routing edit.
- **#2 f32 throughout** — `zeta` is f32; `depth`/`area` reuse the existing
  geometry path.
- **#4 hand-written sparse backward** — `zeta` is an additive RHS term; autograd
  tapes it through the existing `b`. The `CsrSolveOp` backward in `src/sparse.rs`
  is **not** touched. A dedicated `tests/` finite-difference gradcheck covers the
  new `zeta` autograd path.
- **#5/#7 KAN parity** — the default 4-output head is unchanged; the 7-output
  head is a new config with **no DDR fixture to violate** (DDR reverted
  leakance), so it needs only internal gradient/finite-diff checks.

### Assumptions (explicit)

1. **`K_D` is log-space in ddrs** — its range spans two decades and DDR
   denormalizes it through `log_space_params`. Port the log-space flag from DDR
   rather than guess.
2. **Depth expression is ported exactly from DDR's `_compute_zeta`**, not
   silently reused from ddrs's existing geometry, unless the two are verified
   equal. A subtle `depth` mismatch would invalidate the entire test. Finite-diff
   the ported `zeta` (including its `depth`) against DDR before trusting any run.

## Part B — The experiment

The hypothesis is an **interaction**: leakance is identifiable *and* helpful
under hourly forcing but not daily. The test is a 2×2; only **2 new runs** are
needed because the leakance-OFF cells already exist.

### Run matrix

| forcing | leakance | run | role |
|---|---|---|---|
| hourly (precip-disagg) | OFF | existing — 2026-06-23 precip+L1 | hourly control |
| hourly (precip-disagg) | **ON** | **NEW #1** | the candidate |
| daily (repeat-24) | OFF | existing — pre-disagg trained daily routing run (2026-06-19 journal), leakance OFF | daily control |
| daily (repeat-24) | **ON** | **NEW #2** | reproduces DDR's failure regime in ddrs |

Note: the daily-OFF cell is the **trained daily routing** run (the 2026-06-19
journal runs where Muskingum X stuck at init), *not* the summed-Q′ no-routing
baseline.

- **NEW #1** = the precip+L1 config + `use_leakance: true`, **same seed** → same
  gauge batches → paired per-gauge comparison vs the existing hourly-OFF run.
  This is the **decisive paired comparison**.
- **NEW #2** = repeat-24 (flat) upsampling + `use_leakance: true`. Reproduces
  "daily can't learn leakance" inside ddrs's own code and metric pipeline,
  making the hourly-vs-daily interaction airtight rather than borrowed from DDR's
  anecdote. Loss is L1 throughout (matches the existing controls). If no
  same-seed daily-OFF run exists for an exact pairing, run a paired daily-OFF
  alongside #2 (the hourly arm still carries the decisive paired weight).

### Evaluation cohort

One CONUS train-and-test run per ON cell, then slice the **losing-stream
subset** — gauges where the summed-Q′ baseline **over-predicts** (baseline ratio
> 1, the losing-stream signature) — for the decisive metrics. Also report the
full per-gauge distribution (consistent with prior runs; the CONUS median is
known to dilute minority-reach effects, as it did for the temperature channel).

### Decisive metrics (pre-registered, on the losing-stream subset)

1. **Identifiability / magnitude** — did the term learn a non-trivial flux?
   - **Learned net `|zeta|` per reach (m³/s)** — the headline identifiability
     metric. The revert was caused by sub-0.01 m³/s collapse; the term must
     reach **> 0.01 m³/s** on a meaningful set of losing-stream reaches.
   - Δ from init for `K_D / d_gw / leakance_factor`; `K_D` distribution vs its
     `[1e-8, 1e-6]` bounds (a pile-up at the `1e-8` floor is a config-floor
     diagnostic, not automatically a NO-GO).
   - leakance-param gradient norm across training (nonzero, non-vanishing).
2. **Skill** — paired ON−OFF per gauge on the subset:
   - **NSE and/or KGE improvement** (either metric).
   - KGE-β (bias ratio): median `|β−1|` reduction — the volume-closure lever.
3. **Physical plausibility** — physics vs fudge factor:
   - learned losing streams spatially coherent (arid west / karst)?
   - `zeta ≈ 0` where bias ≈ 0 (acts only where needed) vs deleting water
     everywhere.

### Pre-registered go/no-go

- **GO** (hourly unlocks leakance) — on the losing-stream subset, hourly+ON shows
  **(a)** an NSE *or* KGE improvement over hourly+OFF, **AND** **(b)** learned net
  `|zeta| > 0.01 m³/s` on a meaningful set of reaches (clears the revert
  threshold), **AND (c)** this is **absent or much weaker** in daily+ON (the
  interaction). Justifies a full port through the KAN head as a real feature.
- **NO-GO** — leakance collapses to sub-0.01 m³/s under hourly too (forcing is
  not the blocker), **or** `|zeta|` clears 0.01 only by deleting water everywhere
  with no spatial coherence / no skill gain (fudge factor), **or** no NSE/KGE
  improvement on the subset. Falsifies the hypothesis at the pre-registered bar.

## Risks (what could go wrong, and why)

1. **Magnitude bar met for the wrong reason.** `|zeta| > 0.01 m³/s` could be
   reached by leakance acting as a fudge factor (fitting noise) rather than
   physical loss — the opposite failure from the revert. *Guard:* the plausibility
   checks; and the interaction — if daily+ON also clears 0.01 but worsens skill,
   magnitude alone isn't the signal, only the hourly-improves / daily-doesn't
   contrast is.
2. **Depth-expression mismatch.** If ddrs's geometry `depth` ≠ the `depth` DDR's
   `zeta` expects, the physics is wrong and silently invalidates the test.
   *Guard:* port DDR's exact `_compute_zeta` depth expression and finite-diff it
   against DDR before any run (assumption 2).
3. **Attribution ambiguity from joint training.** Geometry (`n`, `X`) can
   compensate for `zeta`. *Guard:* paired seeds (ON vs OFF see identical batches)
   + compare the learned `n` distribution ON vs OFF (the temperature run already
   showed `n` shifts under new forcing).
4. **`K_D` range floor as a false negative.** Learned `K_D` pinning to `1e-8`
   may be range-limited, not data-limited — looks like the revert pathology but is
   a config artifact. *Guard:* report `K_D` vs its bounds; a floor pile-up is
   diagnostic.

## Why this is worth doing

Leakance attacks the one axis — volume bias / KGE-β on losing streams — that the
findings doc identified as a *structural* ceiling, unmoved by either the loss
function or geometry tuning. If hourly forcing unlocks it (GO), it is a
genuinely new lever; if not (NO-GO), we have cheaply falsified it against a fixed
bar, with the testbed code preserved behind a default-off flag.

## Out of scope

- Full productionization of leakance (only the flag-gated testbed is built here).
- d_gw external data sources (it is a learned spatial parameter, not read data).
- Any change to the disagg head, the loss functions, or the summed-Q′ baseline.
- Sub-daily *supervision* (hourly USGS IV) — a separate future lever.
