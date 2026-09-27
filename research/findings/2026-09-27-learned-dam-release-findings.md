# Learned dam release in ddrs: implementation, gates, smoke checks

**Date:** 2026-09-26 to 27. **Branch:** `dam-release-head` (cut from `reservoir-options`). **Design brief:**
`docs/superpowers/specs/2026-09-26-learned-dam-release-design.md` (local, not committed). **Status:** built, gated,
smoke-checked; off by default. First build (§1 to §5): checks 1a, 3, 4 and 5 passed; 1b failed on a wrong
premise; 2 failed and exposed a storage-conservation flaw in the specified dam row. The flaw is fixed (§6: the
storage-conserving dam row, commit `0ac6f2e`), after which check 2 reaches median NSE 0.998, 1b passes under the
adopted criterion, and 4 and 5 pass again. Full-population arms: §7. Follow-ups from the code review: §8.

## 1. What was built

Every large dam in a batch network is routed as a seasonal linear reservoir whose release parameters are learned
jointly with the KAN routing head, from gauge observations only:

```
T_d(t) = max(T0_d · exp(a_d · sin ω_t + b_d · cos ω_t), 1/24 d),   ω_t = 2π · doy(t) / 365.25
dam row: K := T_d(t), X := 0        (Muskingum = linear reservoir S = T·Q)
(T0_d, a_d, b_d) = release_head(NID features_d)
```

No observed dam data enters: the inputs are static NID attributes (storage, surface, drainage, storage per area,
spillway capacity, height, year, primary purpose); the dam inflow is the model's own routed flow; gauge
observations are only the loss target.

| Piece | Where |
|---|---|
| Dam feature table (1,099 NID dams >= 10 MCM in the eval networks, 1,024 COMIDs, 19 normalised columns) | `experiments/reservoir/release_head/build_dam_features.py` → `dam_features.csv`, `dam_features_stats.json`; Juniata fixture `examples/juniata/data/juniata_dam_features.csv` |
| Readers: seasonal fixed table (`COMID,T_days[,a,b]`), feature table | `src/data/store/reservoirs.rs::read_fixed_release_table`, `::read_dam_features`, `::map_reservoir_rows` |
| Config: `params.reservoir_release: fixed \| learned`, `release_head:`, `reservoir_T0/_a/_b` ranges, guards | `src/config.rs::ReservoirRelease`, `::ReleaseHeadSection`, `::validate_reservoirs` |
| Engine: `T` per step in autodiff, handed to the op | `src/routing/mmc.rs::set_dam_release` (`DamRelease`), `src/routing/release.rs` |
| Op: `T` as an autodiff parent | `src/routing/mmc_op.rs::ReleaseParent`, `TimestepReleaseOp` (6 parents), `TimestepReleaseGammaOp` (7), `t_release` in `timestep_backward_core` (B19''') |
| Release head | `src/nn/release_head.rs` (a second `KanHead`; the routing head is untouched) |
| Joint training, checkpoint, resume | `src/training/forward.rs::forward_with_release`, `::apply_reservoir_rows`; `src/training/driver.rs::ReleaseTrainer`; `src/training/bootstrap.rs`; `release_head.mpk` / `release_optim.mpk` |
| Test phase and output | `src/training/release_eval.rs` (resolve the head once into a fixed seasonal table; `release_params.csv`) |

### Gradient

S19'' already replaced a dam row's Muskingum `K` with `T` and `X` with 0, and B19'' masked the dam row's `∂L/∂K`
and `∂L/∂X` so no gradient reaches its channel parameters. B19''' reads that `∂L/∂K`, after the c1..c4 chain and
before the B19'' mask, and returns it as `∂L/∂T` (seconds). `T_d(t)` is built from `(T0, a, b)` in ordinary Burn
autodiff before each step, so the chain to the release head is automatic. The sparse backward is extended, not
replaced (invariant 4).

### Choices recorded

- **Phase.** `doy` 1-based (pandas `dayofyear`, as the offline fit), fractional by hour; the step producing routed
  column `t` uses the phase of hour `t`.
- **Clamp.** At 1/24 d after the seasonal factor; where it binds the step passes exactly zero gradient.
- **Init.** Read-out weights zero, bias at `T0 = 4.5 h`, `a = b = 0`, for every dam; sigmoid slope 0.138 (of 0.25).
  Keeping the Xavier read-out instead spread the init to `T0` 5.5 h and `|a|` 0.11 on synthetic rows.
- **`T0` space.** Always log over `[1/24, 365]` d; `a`, `b` linear over `[-2, 2]`.
- **Joint optimizer.** Same kind and learning rate as the routing head, own moments; each head's gradient clipped
  on its own norm, so a large release gradient early on cannot throttle the routing head.
- **Test phase.** The head is a per-dam function of static features, so it is evaluated once over the whole table
  and routed through the `fixed` seasonal path; `tests/reservoir_release_training.rs` pins that this is bitwise the
  training path on the deterministic NdArray backend (CUDA may differ at the ulp level: one matmul over every table
  dam against per-batch subsets).
- **Multiple dams per COMID** (51 COMIDs): storage, surface and spillway capacity summed, height max, drainage,
  year and purpose from the largest dam.

## 2. Gates

All on commit `772247a` (the routing core has not changed since `854da96`), CPU, deterministic NdArray.

| Gate | Result |
|---|---|
| `cargo test --test ddr_sandbox_match` | pass |
| `compare_ddr_sandbox` | ABSOLUTE MATCH, max abs 1.53e-4 m³/s, max rel 2.41e-6 (unchanged by this branch) |
| `cargo test --lib` | pass (432 at the engine commit, all again in the full run) |
| `--test mmc`, `--test sparse_gradcheck`, `--test sp8_gradcheck` | pass |
| `leakance_gradcheck`, `leakance_off_parity`, `zeta_accum`, `leakance_reference_match` | pass |
| `--test reservoir_override` | pass (option C unchanged) |
| `--test reservoir_release` (a = b = 0 is bitwise option C; seasonal recurrence; clamp; phase; validation) | pass |
| `--test reservoir_release_gradcheck` | pass; worst relative error 1.7e-3 (`T0` at 0.1 d); head weight end to end 1.2e-4; clamped steps give exactly 0 |
| `--test reservoir_release_training` | pass; training and resolved test paths bitwise equal (NdArray); restored optimizer steps like the saved one (8.8e-5 vs 1.5e-2 cold) |
| config tests (`cargo test --lib config`) | pass |
| Tier B KAN fixture sweep | pass |
| `cargo test --no-fail-fast` (full suite, data-dependent tests included) | 883 passed, 0 failed, 26 ignored, 114 binaries |
| `cargo test --release --test juniata_acceptance` | 3 of 3 pass. Plain: NSE 0.7903 / KGE 0.8810, baseline 0.6947, the reference values to four decimals. Option C test passes. Learned: Raystown `T0` 4.50 → 4.30 h after 30 updates, NSE 0.7938 / KGE 0.8848 |

Gradcheck steps: relative `max(1e-2·|x|, 1e-2)` on `T0`, `a`, `b`, except `a`, `b` at `T0 = 20` d, where the dam's
`c3` sits within 2e-3 of 1 and a 1e-2 step is at the f32 floor: the FD scattered non-monotonically (b: 8.6e-3,
8.0e-3, 3.8e-3 at steps 3e-3, 1e-2, 3e-2) and converged at 0.1 (3.9e-4), so that case uses 0.1.

## 3. Smoke checks

Set: `experiments/reservoir/smoke/` at the 3x area rule, 916 gauges (458 below a NID dam >= 10 MCM, 458 matched
controls), 29,395 reaches. Checks 1 and 2 re-route the no-dam head `2026-09-12T23-39-03Z` @ `epoch_50_mb_9` over
1981-10-01..2010-09-30 in one pass (legacy test-phase binary, CPU, ~2,900 s each); checks 4 and 5 are two
train-and-test runs. Scoring: `experiments/reservoir/release_head/smoke_checks_1_2.py`, `smoke_checks_4_5.py`,
`seasonal_mass_check.py`; tables and configs for checks 1 and 2 from `smoke_check_tables.py`. Outputs (JSON
summaries, per-gauge and per-dam CSVs, the learned arm's `release_params.csv`):
`experiments/reservoir/release_head/results/`.

### Check 1a: off = identical. PASS

`use_reservoirs` off on this branch against the reference `pred_1981_2010.zarr` (reservoir-options binary):
0 of 9,700,440 daily values differ, bitwise.

### Check 1b: every dam at `T` = 1 h, `a = b = 0`, NSE > 0.999 at every gauge. FAIL as specified; the premise is wrong

677 smoke dams (all in the network). Median NSE against the no-dam run 0.99999986, but 188 of 916 gauges fall below
0.999 (min 0.485 at 03225500); every one is a dam gauge (the 458 controls are untouched). A dam row does not add a
one-hour bucket to a channel: it REPLACES its MERIT reach's Muskingum-Cunge routing, whose `K = L/c` is about 4 h at
the median reach (median `Cr` 0.226, research-status). A one-hour bucket is therefore faster than the no-dam reach it
replaces. The loss tracks the dam reach's length: Spearman(1 − NSE, longest dam reach upstream) = 0.55
(p 3e-37); by quartile of summed dam-reach length over sqrt(area), median NSE 0.99995, 0.99977, 0.99890, 0.99647.
Separately, 60 gauges' no-dam series sit at the discharge floor (1e-4 m³/s) on some days (206 days at 03225500): the
channel reach's negative-coefficient oscillation, clamped. The bucket (all coefficients positive at `T` >= dt/2)
does not do this, which accounts for the worst few gauges.

Proposed replacement check: pass-through relative to the reach, i.e. the recurrence test already in
`tests/reservoir_release.rs` (a headwater dam follows the bucket recurrence to 2e-5), plus 1b's median and the
length dependence above as the expected signature.

### Check 2: engine vs offline seasonal fit at on-reach dams, NSE > 0.99. FAIL; exposes a design flaw

202 on-reach dams (dam COMID == gauge COMID), each at its gauge's fitted `seas_T0`, `seas_a`, `seas_b`, other dams
off. Test-window (WY1996-2010) NSE between ddrs and the fit's series: median 0.982, 85 of 202 above 0.99; the 169
with no other table dam upstream: median 0.985, 73 above 0.99. Worst: `T0` of 100 to 1,000 d with `|(a, b)|` 1 to
2 (08245000, `T0` 776 d: NSE −1.47).

Cause, confirmed: the brief's dam row, `K := T_d(t)`, `X := 0` in a Muskingum row, conserves the state `Q` across a
change in `T`, not the storage `S = T·Q`. Integrated, it is `dQ/dt = (I − Q)/T`, so `dS/dt = I − Q + Q·dT/dt`: a
spurious source `Q·dT/dt`. The offline law carries `S`. Simulated hourly on the no-dam flow at all 202 gauges:

| | NSE vs ddrs (median) | NSE vs offline fit | above 0.99 vs fit |
|---|---:|---:|---:|
| Muskingum `K := T(t+1)` (what ddrs does) | 0.9989 | | |
| storage-conserving trapezoid | 0.9826 | 0.9996 (min 0.987) | 197 of 202 |

Mass (test-window mean outflow over mean inflow): ddrs median 1.0044, 90th percentile 1.127, max 2.49; the
storage-conserving form 1.0000. ddrs's agreement with the fit falls with the size of the source term
`T0·|(a,b)|·ω`: median NSE 0.995 below 0.01 d/d, 0.990 at 0.01 to 0.1, 0.859 at 0.1 to 1 (0 of 52 above 0.99),
0.29 above 1. The engine implements the specified recurrence exactly; the specification is not mass-conserving
when `T` varies.

**Proposed fix (not implemented, pending approval).** On dam rows use the storage-conserving trapezoid

```
T_{t+1}·Q_{t+1} − T_t·Q_t = dt·[(I_t + I_{t+1})/2 + q' − (Q_t + Q_{t+1})/2]
⇒ c1 = c2 = dt/(2T_{t+1} + dt),  c3 = (2T_t − dt)/(2T_{t+1} + dt),  c4 = 2dt/(2T_{t+1} + dt)
```

Only `c3`'s numerator changes (the previous step's `T`); with constant `T` it is today's row bit for bit, so option
C, `a = b = 0` and every non-release path stay identical. The op then needs `T_t` as a second parent (the previous
step's `T` tensor): B19''' gains `∂c3/∂T_t = 2/(2T_{t+1} + dt)` and a changed `∂c3/∂T_{t+1}`; the gradcheck
covers it. Then checks 2, 4 and 5 re-run.

### Check 3: gradients. PASS

`tests/reservoir_release_gradcheck.rs` (Section 2): `T0`, `a`, `b` at `T0` = 0.1, 1.5 and 20 d, with the learned
gamma op, one release-head read-out weight end to end, and zero gradient under the clamp. (Of the formulation as
built; the fix above needs its own cases.)

### Check 4: learning. PASS in direction, partial in "stays low"

Arms `2026-09-27T04-29-33Z-train-and-test` (off) and `2026-09-27T04-29-43Z-train-and-test` (learned), the
`sr_n0_gamma` recipe on the smoke CSV, 50 epochs, 200 optimizer steps, seed 42, CPU; wall 3,981 s and 4,123 s
(train 2,832 / 2,961 s, test 1,145 / 1,157 s). Workspace: this worktree's `.ddrs`.

- The release head's batch-median `T0` rose from 0.19 d (4.5 h) to about 0.6 d by the first quarter of training
  and stayed there (0.62, 0.62, 0.71, 0.55 d at the log's quartiles and end).
- Resolved over all 1,024 table dams: `T0` median 0.56 d (IQR 0.33 to 1.03, max 8.3 d); `|a|` and `|b|` all below
  0.46, most within 0.1: the head learned almost no seasonality.
- At the 339 nearest dams of dam gauges: where the offline fit found storage (`seas_T0` above its 0.05 d floor, 251
  dams) learned `T0` median 0.71 d (fit 1.75 d), 95 % moved more than 10 % from init, 88 % above it; where the fit
  found none (88 dams) median 0.37 d, 77 % above init. Spearman(learned, fitted `T0`) 0.35 (p 3e-11).
- So `T0` moves up where the fit found storage and is lower where it found none, but "none" did not stay near
  pass-through (0.37 d against 0.19 d init). Learned `T0` sits well below the fit's; the 90-day hotstarted `rho`
  windows (brief §6) are one candidate cause, untested.

### Check 5: beat the controls. PASS

Paired per gauge, test WY1996-2010, learned minus off:

| | median ΔNSE [95 % CI] | up / down | offline ceiling |
|---|---:|---:|---:|
| 458 dam gauges | +0.0049 [+0.0027, +0.0072] | 300 / 158 (p 3e-11) | +0.0068 [+0.0041, +0.0131] |
| on the dam's reach (214) | +0.0055 [+0.0033, +0.0122] | 147 / 67 | |
| further down (244) | +0.0035 [+0.0018, +0.0075] | 153 / 91 | |
| 458 controls | −0.0001 [−0.0005, +0.0003] | 222 / 236 (p 0.54) | +0.0001 |
| dam minus matched control | +0.0058 [+0.0032, +0.0086] | 294 / 164 (p 1e-9) | +0.0085 |

Median test NSE: dam gauges 0.610 → 0.646, controls 0.759 → 0.763 (within 0.01). ΔKGE is null in both groups
(dam −0.0001, controls +0.0001). Whole-set median NSE 0.7091 → 0.7209 / KGE 0.7328 → 0.7337 (the runs' manifests;
summed-Q' baseline on these gauges and window 0.662 / 0.702). The learned bucket reaches about 70 % of the offline
ceiling. The mass flaw of check 2 barely
touches this arm: with the learned `a`, `b` the source bound `T_max·|(a,b)|·ω` is 8e-4 at the median dam and 4e-2
at the worst.

## 4. Deviations from the brief

- **Read-out init.** The brief asked for `T0` ≈ 3-6 h and `a = b = 0` at init; the read-out weights are zeroed so
  every dam starts at exactly 4.5 h and 0, 0 (the Xavier read-out spread `T0` to 5.5 h and `|a|` to 0.11).
- **Gradient clipping.** Each head clipped on its own norm (not specified in the brief).
- **Test phase.** The release head is not threaded through `training::eval`; it is resolved once into a fixed
  seasonal table (bitwise the training path on NdArray, pinned by a test; ulp-level differences are plausible on
  CUDA).
- **Feature table.** The brief's `max discharge plus a missing flag` generalised: every continuous column with a
  missing value gets a flag (storage max, surface, max discharge, year). Catchment attributes of the dam reach were
  not added (optional in v1).
- **Check 1b/2 tables.** Check 1b used `smoke_dams.csv` (677 COMIDs); check 2 used on-reach dams only (202), since
  the offline fit is per gauge on the no-dam flow and an off-reach bucket upstream would change another gauge's dam
  inflow in ddrs but not in the fit.
- **No `cargo install`.** Runs used the worktree's `target/release/ddrs` and test-phase binary directly; the global
  `~/.cargo/bin/ddrs` was left alone.

## 5. Open problems

1. **The storage flaw (check 2).** Resolved 2026-09-27, §6.
2. **Check 1b's criterion.** Replaced, §6 (the recurrence, the median, the reach-length dependence).
3. **Week-scale `T0`** is not learnable from 90-day hotstarted windows; `experiment.state_cache` is the known route.
4. **Seasonality was not learned** (`|a|, |b|` < 0.46, mostly < 0.1) where the offline fit put 142 of 335 active
   dams on the ±2 edge. Whether that is the flaw, the window length, the learning rate, or real, is open.
5. **No-dam floor days.** 60 of 916 no-dam smoke gauges hit the discharge floor on some days (channel reaches with
   negative Muskingum coefficients); unrelated to this feature, worth a look.

## 6. The storage-conserving dam row (2026-09-27, commit `0ac6f2e`)

Approved fix for the check-2 flaw. On dam rows only:

```
T_{t+1}·Q_{t+1} − T_t·Q_t = dt·[(I_t + I_{t+1})/2 + q' − (Q_t + Q_{t+1})/2]
c1 = c2 = dt/(2T_{t+1} + dt),  c3 = (2T_t − dt)/(2T_{t+1} + dt),  c4 = 2dt/(2T_{t+1} + dt)
```

`T` is evaluated from the closed form at both ends of every step (phase rows `t − 1` and `t`), so a window or
test-phase chunk start needs no carried state; the one-hour clamp applies to both. The step-start `T_t` is a second
autodiff parent (`TimestepReleaseOp` 7 parents, `TimestepReleaseGammaOp` 8). In B19''' c3's numerator term,
`gc3/denom`, no longer reaches K (= `T_{t+1}`) on dam rows and is returned as `∂L/∂T_t = 2·gc3/denom`; `∂L/∂T_{t+1}`
is the rest of the dam row's `∂L/∂K`. c3's dam-row expression is the generic one op for op, so a constant `T` is
option C bit for bit.

### Gates after the fix

| Gate | Result |
|---|---|
| `ddr_sandbox_match`, `compare_ddr_sandbox` | pass; ABSOLUTE MATCH, max abs 1.53e-4 m³/s (unchanged) |
| `--lib` | 437 pass |
| `mmc`, `sparse_gradcheck`, `sp8_gradcheck`, the four leakance tests, `reservoir_override` | pass |
| `reservoir_release` (7) | pass. `zero_seasonal_coefficients_are_bitwise_option_c` pins constant `T` = option C. New `seasonal_release_conserves_storage`: storage imbalance 7.6e-8 over 492 h with `T` swinging 0.7 to 13 d every 48 h and a pulsed inflow; the pre-fix row fails it at 1.2e-1 |
| `reservoir_release_gradcheck` (7) | pass. New one-step check through `mmc_op::timestep_forward_release`: `T_t` and `T_{t+1}` separately, worst rel error 1.0e-4, at 2 to 3 h and at 5 to 30 d; one leaf in both slots gets exactly the sum of the two (difference 0), and matches its FD to 5.9e-6 of the parts' scale (the two nearly cancel at constant `T`) |
| `reservoir_release_training` (6) | pass |
| full `cargo test --no-fail-fast` | 885 passed, 0 failed, 26 ignored, 114 binaries |
| `cargo test --release --test juniata_acceptance` | 3 of 3; plain NSE 0.7903 / KGE 0.8810, baseline 0.6947 (unchanged); learned NSE 0.7937 / KGE 0.8848, Raystown `T0` 4.30 h |

### Check 2 after the fix: median 0.998 (was 0.982)

Same 202 on-reach dams at their offline seasonal fit, test-window NSE between ddrs and the fit's series:

| | before | after |
|---|---:|---:|
| all 202: median, above 0.99 | 0.982, 85 | **0.998, 160** |
| 169 with no other table dam upstream | 0.985, 73 | **0.998, 140** |
| 33 below another table dam | 0.960, 12 | 0.992, 20 |
| minimum | −1.47 (`T0` 776 d) | 0.52 (cascade), 0.75 (isolated) |

The long-`T0` failures are gone: ddrs now matches the storage-conserving law it was compared with (median NSE 0.9990
against an hourly simulation of that law on the no-dam flow). The dams still below 0.99 are the check-1b gauges
(03225500, 08390500, 03228805, 02085500, 08177500 are the worst five): a dam row replaces its reach's channel routing
and the no-dam reference series hits the discharge floor, which the offline fit, run on the no-dam gauge series,
inherits. 08177500 is one of them at a constant `T` (`a = b = 0`), so the seasonal law is not the cause.

**Mass.** Plain ratio of test-window outflow to the no-dam inflow at the 202 dams: median 1.0008 (max 1.155 at
03225500, which has 206 floor days in the no-dam series). With the storage change `T_end·Q_end − T_start·Q_start`
added (`dam_mass_balance.py`, WY1983-2010): median 1.0002. Normalising by the one-hour pass-through run at the same
gauge, which carries the same reach-replacement and floor effects but no seasonal storage, isolates the dam row's
own balance: median 0.99998, 156 of 169 isolated dams within 1e-3 of 1, 167 within 5e-3 (outliers are `T0` of 60 to
1,000 d, where the daily-mean `ΔS` estimate is coarse). The exact check is the engine test (imbalance 7.6e-8).

### Check 1b: criterion adopted

Adopted from §3 (check 1b): the dam row follows the bucket recurrence exactly
(`headwater_dam_follows_the_storage_conserving_seasonal_recurrence`, rel 2e-5), the median over all 916 gauges is
above 0.999 (0.99999986), and the deviations at dam gauges follow the dam reach's length (Spearman 0.55,
p 3e-37). Under it, 1b passes. The fix does not change it: at `a = b = 0` the dam row is bitwise option C.

### Checks 4 and 5 after the fix (smoke learned arm re-run)

Run `2026-09-27T07-30-13Z-train-and-test` (main workspace, CPU, wall 3,577 s: train 2,496 s, test 1,075 s), paired
against the unchanged no-dam arm `2026-09-27T04-29-33Z-train-and-test` (the no-dam path is untouched by the fix).

| | first build | after the fix |
|---|---:|---:|
| dam gauges ΔNSE | +0.0049 [+0.0027, +0.0072], 300 / 158 | **+0.0056 [+0.0021, +0.0095], 280 / 178** |
| on the dam's reach (214) | +0.0055 | +0.0063 [+0.0017, +0.0153] |
| further down (244) | +0.0035 | +0.0038 [+0.0014, +0.0096] |
| controls ΔNSE | −0.0001 [−0.0005, +0.0003] | −0.0011 [−0.0018, −0.0002], 199 / 259 |
| dam minus matched control | +0.0058 [+0.0032, +0.0086] | **+0.0072 [+0.0044, +0.0108]** |
| dam ΔKGE | −0.0001 | −0.0014 [−0.0030, −0.0002] |
| median test NSE, dam / control | 0.646 / 0.763 | 0.643 / 0.761 (no-dam 0.610 / 0.759) |
| learned `T0` median (1,024 dams) | 0.56 d | 0.59 d (IQR 0.34 to 1.21, max 9.7) |
| seasonal amplitude `√(a² + b²)` median, max | < 0.1, 0.46 | **0.31, 0.74** |
| learned vs fitted `T0`, Spearman | 0.35 | 0.40 |
| `T0` where the fit found storage / none | 0.71 / 0.37 d | 0.87 / 0.44 d |

All three check-5 criteria pass again, and the dam-minus-control margin rises to about 85 % of the offline one
(+0.0085). With storage conserved the head now learns seasonality (median amplitude 0.31 against < 0.1), which the
flawed row penalised: under it a seasonal swing injected spurious water. Two costs are new and small: controls lose
0.0011 NSE per gauge (interval clear of zero; the control-group median NSE still moves by only 0.0016, well
within 0.01), and dam-gauge KGE falls by 0.0014. The controls have no dams; their change is the routing head
co-trained with the release head.

## 7. Full population (2,365 gauges), dams off against learned

Arms, both `train-and-test` at `0ac6f2e`, CPU, seed 42, 50 epochs, main workspace `/home/tbindas/projects/ddrs/.ddrs`:

| arm | run | wall | train / test |
|---|---|---:|---:|
| off (`config/experiments/dam_release_full_off.yaml`, the sr_n0_gamma recipe) | `2026-09-27T07-29-47Z-train-and-test` | 8,601 s | 6,558 / 2,043 s |
| learned (`dam_release_full_learned.yaml`, + the release on 1,024 NID dam COMIDs, all in the network) | `2026-09-27T07-29-55Z-train-and-test` | 9,003 s | 6,959 / 2,043 s |

The off arm reproduces `2026-09-12T23-39-03Z-train-and-test` exactly (median NSE 0.7391 / KGE 0.7592). The paired
analysis is the coordinator's, `experiments/reservoir/full_run/paired_full_run.{py,json,csv}` on branch
`reservoir-options`; `experiments/reservoir/release_head/full_population_pairing.py` on this branch reproduces its
paired numbers exactly (`results/full_pairing.json`).

| test WY1996-2010 | off | learned | summed Q' |
|---|---:|---:|---:|
| median NSE, 2,365 gauges | 0.7391 | 0.7378 | 0.6785 |
| median KGE | 0.7591 | 0.7579 | 0.7171 |
| median NSE, 917 gauges with a NID dam >= 10 MCM upstream | 0.7219 | 0.7286 | 0.6196 |
| median NSE, 1,448 without | 0.7458 | 0.7412 | 0.7068 |

Paired per gauge, learned minus off (median [95 % bootstrap], up / down):

| | ΔNSE | ΔKGE |
|---|---:|---:|
| all 2,365 | −0.0002 [−0.0004, −0.0001], 1,105 / 1,260 | −0.0003 [−0.0005, −0.0002] |
| 917 with a NID dam >= 10 MCM upstream | **+0.0014 [+0.0007, +0.0020], 532 / 385** (p 1e-6) | −0.0009 [−0.0016, −0.0004] |
| 1,448 without | **−0.0007 [−0.0009, −0.0004], 573 / 875** (p 2e-15) | −0.0002 [−0.0003, −0.0001] |
| dam on the gauge's reach (377) | +0.0027 [+0.0015, +0.0055] | |
| the 458 smoke dam gauges / 458 controls | +0.0040 [+0.0021, +0.0067] / −0.0006 [−0.0010, −0.0003] | |
| by NID storage / annual flow: <= 0.1, 0.1-0.5, 0.5-1, 1-2, > 2 | +0.0000, +0.0017, +0.0026, +0.0047, +0.0009 | |

**Learned release** (`2026-09-27T07-29-55Z-train-and-test/release_params.csv`, 1,024 dams; copy in
`results/release_params_full_learned.csv`): `T0` median 0.35 d (IQR 0.19 to 0.84, 5-95 % 0.12 to 3.9, max 17.4);
219 dams above 1 d, 13 above 10 d. Seasonal amplitude `√(a² + b²)` median 0.13 (5-95 % 0.06 to 0.35, max 0.56);
`a` median −0.08 (range −0.47 to +0.16), `b` +0.02 (−0.31 to +0.22). Seasonal `T` range over the year: minimum
median 0.32 d, maximum median 0.39 d (max 30 d). The full-population head sits lower and less seasonal than the
smoke re-run's (`T0` 0.59 d, amplitude 0.31), whose loss is half dam gauges.

**Reading.** The release helps below dams, most where storage is 0.5 to 2 years of flow and where the dam is on the
gauge's reach, the same pattern as the offline fit; it is small (+0.0014 median) because most of the 917 gauges see
little regulation. It is not a population gain: undammed gauges lose 0.0007, a cost that can only come through the
co-trained routing head, since their networks contain no dam. One seed per arm cannot separate that cost from a
different training trajectory. Seed-43 replicates of both arms are running (`2026-09-27T10-31-30Z` off,
`2026-09-27T10-31-50Z` learned); not waited for here.

## 8. Follow-ups (code review of `0ac6f2e`, nothing blocking)

a. **Bitwise equality of the test phase is an NdArray claim.** The resolved test-phase path routes bitwise like
   training on the deterministic NdArray backend (`tests/reservoir_release_training.rs`). On CUDA, ulp-level `T0`
   differences are plausible: the test phase runs the head on every table dam in one matmul, training on each
   batch's subset. The wording in this doc, `RESERVOIRS.md`, `testing.md` and the test's module docs now says so.
b. **`params.parameter_ranges` is an untyped map,** so a misspelt `reservoir_T0` / `reservoir_a` / `reservoir_b` is
   silently ignored and the default box is used (the pre-existing pattern for every range key). Proposed: a key
   whitelist in `From<ParamsRaw>`, rejecting unknown range keys. Not in this branch; documented in `config.md`.
c. **Head-less forwards panic with a learned config,** by design: `probe_forward`, the `Frozen` test-phase path
   (`eval --frozen`), and any `forward_eval*` caller before `resolve_learned_release`. Routing the dams as channels
   instead would silently score a different model. Documented in `config.md` §Reservoirs.
d. **Performance.** `ArmedRelease::t_step_at` evaluates the seasonal `T` at each hour twice (as one step's end and the
   next step's start); caching the previous step's tensor would halve those small ops. `release_t0_stats` runs the
   release head a second time per micro-batch for the log line; it could read the forward's own `T0`. Neither is
   material next to the routing forward (release-arm training was 6 % slower than off on the full population).
