# Reservoirs (not implemented; options of record 2026-09-25)

> ## STATUS: **no reservoir code in ddrs.** Options ranked, none built.
>
> A reach inside a reservoir is routed as an MC channel. The dMC fill-fraction law is **closed**
> (a natural-lake law; never beat a one-parameter linear reservoir at four dams). DDR's level pool
> (#137 to #139) was reverted in #143 without a gauge evaluation. Read the options doc before
> building anything:
> `research/findings/2026-09-25-reservoir-representation-options.md` (§5 options, blast radius,
> concerns). Numbers of record: `skills/ddrs-dev/references/research-status.md` §Reservoirs.

---

## Where a reservoir can enter the per-timestep solve

Everything below would live in `src/routing/mmc.rs::route_timestep`; the CSR
pattern and the hand-written sparse backward in `src/sparse/` are untouched because only values
change.

```
 A q_{t+1} = b,  lower-triangular, one solve per hour (dt = 3600 s)

 ordinary reach i                             dam row d  (identity row: DDR #138's pattern)
 A[i,:] = e_i - c1_i N[i,:]                   A[d,:] = e_d          c1_d := 0
 b_i    = c2_i (N q_t)_i + c3_i q_t,i         b_d    = R_d(S_t, I_t)
          + c4_i q'_i                                  │
                                                       ├─ B observed release: Q_obs(t+1), fallback modelled
 C  linear reservoir, no identity row:                 ├─ D capped linear:   min((S_t + dt I_t)/(T_d + dt), Q_max,d)
    k_d := T_d,  x_storage_d := 0                      └─ E offline rule:    f(S_t / S_cap, day of year)
    (Muskingum with X = 0 IS S = K Q)
                                              after the solve (D, E):
                                                I_{t+1} = (N q_{t+1})_d + q'_d
                                                S_{t+1} = S_t + dt (I_{t+1} - R_d)

   ...─▶ [reach] ─▶ [reach] ─▶ ╔═ dam row d ═╗ ─▶ [reach] ─▶ ◉ gauge
                               ║ q_d = R_d   ║     forward substitution carries R_d
                               ╚═════════════╝     to every row below in the same solve
```

## What decides between them

```
                    is there a gauge between the dam and the scored gauge?
                         │ yes (≤ 216 of 347 DOR > 0.5 gauges)      │ no (131)
                         ▼                                          ▼
                B: prescribe observed release             is the dam near pass-through?
                (0 parameters, exact, but                 │ yes (flood control, Raystown)   │ no (scheduled, Alamo)
                 conditions eval on obs)                  ▼                                  ▼
                                                C (T) or D (T + Q_max as data)     E if ResOpsUS / ISTARF has it,
                                                                                   otherwise mask the gauge (A)
```

## Traps

- **A parametric node is only as good as its inflow.** The cap's +0.13 test NSE at Raystown with
  observed inflow is +0.02 with the trained model's inflow. Any learned `T` or `Q_max` absorbs
  upstream error.
- **A boundary gauge must not be a training target**, and B's metrics are conditioned on observed
  releases: record the mode in the manifest.
- **The reservoir table is the NWM / RFC-DA set** (`~/projects/ddr/data/merit_reservoir_params.csv`,
  2,178 COMIDs). It misses dams outside it (Alamo). Counts built on it are lower bounds.
- **f32:** carry `S` as the active buffer (a few MCM), not total volume.
- **Invariant 1:** off by default; DDR master has no reservoirs, so `ddr_sandbox_match` must not
  see any change.
