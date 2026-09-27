# Reservoirs (option C and the learned dam release implemented; options of record 2026-09-25)

> ## STATUS: **option C and the learned dam release implemented, both off by default.** Options B, D, E not built.
>
> **Learned dam release (2026-09-26, branch `dam-release-head`).** `params.reservoir_release: learned`
> routes every dam of a feature table (`experiments/reservoir/release_head/dam_features.csv`, NID
> >= 10 MCM) as a seasonal linear reservoir `S = T_d(t)·Q`,
> `T_d(t) = max(T0_d·exp(a_d sin ω_t + b_d cos ω_t), 1 h)`, with `(T0, a, b)` from a release head
> (a second `KanHead`, `src/nn/release_head.rs`) fed NID dam features and trained jointly with the
> routing head from gauge observations only. `T` is an autodiff parent of the timestep op
> (`TimestepReleaseOp` / `TimestepReleaseGammaOp`, S19''' / B19''' in `src/routing/mmc_op.rs`), gradchecked
> in `tests/reservoir_release_gradcheck.rs`. A `fixed` table may carry `a`, `b` columns for a
> prescribed seasonal bucket. Section below; config contract in
> `skills/ddrs-dev/references/config.md` §Reservoirs; smoke-set results in
> `research/findings/2026-09-27-learned-dam-release-findings.md` (dam gauges ΔNSE +0.0049
> [+0.0027, +0.0072], controls −0.0001, dam minus control +0.0058, first build). The dam row is the
> storage-conserving trapezoid on `S = T·Q` since 2026-09-27 (the first build's `K := T(t)` alone did
> not conserve storage; Traps below). Full population (2,365 gauges, one seed): ΔNSE +0.0014
> [+0.0007, +0.0020] at the 917 gauges below a NID dam >= 10 MCM, −0.0007 at the 1,448 without;
> median NSE 0.7378 learned vs 0.7391 off.
>
> `params.use_reservoirs: true` + `data_sources.reservoirs: <csv>` (header `COMID,T_days`) routes
> each listed dam reach as a linear reservoir `S = T·Q` through
> `MuskingumCunge::set_reservoir_rows` (Muskingum `K := T`, `X := 0` on those rows only).
> Committed fixture: `examples/juniata/data/juniata_reservoirs.csv` (Raystown Lake, COMID
> 73005301, `T_days = 1.23`). Config contract, rejected combinations and wiring:
> `skills/ddrs-dev/references/config.md` §Reservoirs. With the flag off (the default) every reach
> is routed as an MC channel. **Benchmark result (2026-09-26), C bolted onto a head trained
> without it:** ΔNSE +0.015 (null) at the 44 dams with a ResOpsUS-fitted `T`, ΔKGE −0.081
> (over-attenuation), a median `T` at the other 77 hurts (−0.143), no runtime cost; a head
> trained with C on is the untested fair comparison
> (`research/findings/2026-09-26-option-c-dam-benchmark-findings.md`). The dMC fill-fraction law is **closed**
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

## The learned dam release (`reservoir_release: learned`)

```
 per dam d, per hourly step t (release head shared across dams)

 NID features_d ──▶ release head ──▶ (T0_d, a_d, b_d)          T0 log space [1/24, 365] d, a, b in [-2, 2]
                    KanHead, P = 3        │
                                          ▼
 ω_t = 2π·doy(t)/365.25 ──▶ T_d(t) = max(T0_d·exp(a_d sin ω_t + b_d cos ω_t), 1/24 d)   ordinary Burn autodiff
                                          │  seconds, at BOTH ends of each step: T_t (row t−1), T_{t+1} (row t)
                                          ▼
 timestep op (TimestepReleaseOp): S19'' K := T_{t+1}, X := 0 on dam rows, and S19''' c3's numerator
   reads T_t:  c1 = c2 = dt/(2T_{t+1}+dt), c3 = (2T_t − dt)/(2T_{t+1}+dt), c4 = 2dt/(2T_{t+1}+dt)
   = the trapezoid on S = T·Q, storage-conserving for any T(t); one lower-triangular solve
 backward (B19'''): ∂L/∂T_{t+1} = the dam row's ∂L/∂K without c3's numerator term, ∂L/∂T_t = 2·gc3/denom;
   B19'' still masks the dam row's K and X away from n, q, p
```

- **Where the code is.** Engine: `src/routing/mmc.rs::set_dam_release` (`DamRelease`), the
  per-step `T` in `MuskingumCunge::forward`, `src/routing/release.rs` (phase table, clamp,
  constants). Op: `ReleaseParent`, `TimestepReleaseOp`, `TimestepReleaseGammaOp`, the
  `t_release` branch of `timestep_backward_core` in `src/routing/mmc_op.rs`. Head:
  `src/nn/release_head.rs`. Training: `forward_with_release` and `apply_reservoir_rows` in
  `src/training/forward.rs`, `ReleaseTrainer` in `src/training/driver.rs`, bootstrap and
  checkpoint files `release_head.mpk` / `release_optim.mpk`. Test phase:
  `src/training/release_eval.rs` resolves the head ONCE into a fixed seasonal table
  (`MeritGagesDataset::resolve_learned_release`) and writes `<run>/release_params.csv`
  (`COMID, T0_days, a, b, T_min_days, T_max_days`). That path routes bitwise like training on
  NdArray; on CUDA ulp-level `T0` differences are plausible (one matmul over all table dams vs
  per-batch subsets).
- **Phase convention.** `doy` is 1-based like pandas `dayofyear` (the offline fit's). The step
  producing routed column `t` reads `T` at phase rows `t − 1` (start) and `t` (end), both from the
  closed form, so a window or test-phase chunk start needs no carried state.
- **Constant `T` is option C bit for bit.** `c3`'s dam-row expression is the generic one op for op
  (`T·2` against `(K·2)·(1 − 0)`, exact in f32), pinned by
  `tests/reservoir_release.rs::zero_seasonal_coefficients_are_bitwise_option_c`.
- **The clamp.** `T >= 1 h` keeps `c3 >= 0`. Where it binds, `T0`, `a`, `b` get exactly zero
  gradient for that step (`clamped_release_has_exactly_zero_gradient`).
- **Init.** Read-out weights zero, bias at `T0 = 4.5 h`, `a = b = 0` for every dam; sigmoid slope
  0.138 at init.
- **Joint optimizer.** Same kind and lr as the routing head, separate moments, each head clipped
  on its own gradient norm.
- **Additive dam row (2026-09-27, `release_head.dam_row: additive`).** The replace row (the
  default) drops the reach's channel storage, so a 1 h bucket is faster than no dam and the head
  cannot opt out. The additive row keeps the reach's `K_r`, `X_r` and adds `T·Q`:
  `D = K_r(1 − X_r) + T_{t+1} + dt/2`, `c1 = (dt/2 − K_r X_r)/D`, `c2 = (dt/2 + K_r X_r)/D`,
  `c3 = (K_r(1 − X_r) + T_t − dt/2)/D`, `c4 = dt/D` (S19'''' / B19''''). `T = 0` is the channel
  row bit for bit, `K_r = 0` is the replace row, `T` has no floor. `tests/reservoir_additive.rs`.
- **Harmonic rule curve (2026-09-27, `release_head.rule_curve`, `per_dam_t0`).** `S = T·Q + S0_d(t)`,
  flux `r_d = Ibar_d·Σ_{k=1,2}(c_{k,s} sin kω + c_{k,c} cos kω)` off the dam row's `q'`; per-dam
  free coefficients `c = rule_curve_max·tanh(θ)` (and `T0·exp(δ)`) calibrated through the gauge
  loss at their own constant lr, because offline they are not predictable from NID features
  (rule-curve report: +0.049 median NSE at DOR > 0.5 on-reach with per-dam calibration).
  `Ibar_d` = `inflow_mean_m3s` (`build_dam_inflow_clim.py`). Flaming Gorge (COMID 77013090) is
  mis-snapped to a 32 km² reach, so its `Ibar` is ~0 and its rule curve inert. The rule curve's
  phase is continuous (`release::rule_curve_phase_start`, one hour of phase per hourly step,
  within ±0.75 d of day-of-year), unlike the `T` law's calendar day of year: the first build
  used the latter, whose restart at 1 January gave the year-boundary step +7 h (−17 h in a leap
  year) of flux. `tests/reservoir_rule_curve.rs`.
- **Release-only training (2026-09-27).** `release_head.routing_checkpoint` loads the routing
  head's weights (only `head.mpk`) from another run's checkpoint directory, and
  `release_head.freeze_routing: true` detaches it: the routing head stays bitwise at the
  checkpoint (training and test phase), and only the release head trains, through the solve.
  This removes the co-training confound (undammed gauges are then unchanged by construction).
  `tests/release_freeze_routing.rs`; config contract in `skills/ddrs-dev/references/config.md`.
- **Off is identical.** `use_reservoirs: false`, or a `fixed` table without `a`/`b`, runs the
  historical ops and nodes; `a = b = 0` is bitwise option C (`tests/reservoir_release.rs`).
- **Refusals.** A learned table reaching a forward with no release head panics (probe and frozen
  paths); the paper studies refuse any `use_reservoirs` arm.

## Traps

- **A parametric node is only as good as its inflow.** The cap's +0.13 test NSE at Raystown with
  observed inflow is +0.02 with the trained model's inflow. Any learned `T` or `Q_max` absorbs
  upstream error.
- **A boundary gauge must not be a training target**, and B's metrics are conditioned on observed
  releases: record the mode in the manifest.
- **The NWM / RFC-DA set** (`~/projects/ddr/data/merit_reservoir_params.csv`, 2,178 COMIDs) misses
  dams (Alamo). Since 2026-09-26 the fuller list is the NID snapped to MERIT:
  `experiments/reservoir/nid/nid_dams_in_eval_network.csv` (5,935 dams inside eval networks,
  1,099 >= 10 MCM); see `research/findings/2026-09-26-nid-dams-merit-findings.md`.
- **f32:** carry `S` as the active buffer (a few MCM), not total volume.
- **Invariant 1:** off by default; DDR master has no reservoirs, so `ddr_sandbox_match` must not
  see any change.
- **A time-varying `T` in a plain Muskingum row does not conserve storage (fixed 2026-09-27).** The first
  build set only `K := T_d(t)`, `X := 0`, which carries `Q` across a change in `T`, so `S = T·Q` jumped:
  `dS/dt = I − Q + Q·dT/dt` (mass ratio 2.49 at `T0` 776 d with amplitude 2). The dam row now reads
  `T_t` in c3's numerator (S19'''), the trapezoid on `S`; pinned by
  `tests/reservoir_release.rs::seasonal_release_conserves_storage`. Any future time-varying dam law
  must keep the storage form. See the findings doc, check 2.
- **A one-hour bucket is not pass-through relative to the no-dam model.** A dam row replaces its reach's channel
  routing (`K = L/c`, ~4 h at the median MERIT reach), so the change grows with the dam reach's length.
- **Long `T` against `rho`.** Training windows are 90 days and start from a hotstart guess, so a
  `T0` of weeks or more is learned from a bucket that never fills from its steady state inside the
  window. The `experiment.state_cache` fix is later work.
- **Finite differences at large `T`.** With `T0 = 20` d the dam's `c3` sits within 2e-3 of 1 and a
  1e-2 step in `a` or `b` moves it by a few hundred f32 ulps: the FD scatters around the analytical
  value until the step reaches ~0.1 (`tests/reservoir_release_gradcheck.rs` module docs).
