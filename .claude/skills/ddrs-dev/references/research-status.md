# Research status and authoritative numbers

Re-verified 2026-07-30. This file replaces the leakance/equifinality status tables
that were duplicated across twelve retired skills — where the same superseded
paragraph appeared four to six times and none were updated when one findings doc
retired all of them.

**Cite from this file, not from memory, and not from a skill that predates it.**

## Contents

§Gauge-set definitions · §Benchmarks → §The KGE claim · §Closed campaigns
(leakance · selective equifinality H1–H6 · Q′-store waves · synthetic-n) ·
**§Do-not-use list** · §Structural constants · §Evidence standard ·
§Doc conventions · §Open, not closed

If you are about to cite a number, read §Gauge-set definitions and
§Do-not-use list first — most wrong numbers here are population confusions, not
arithmetic errors.

## Gauge-set definitions — memorize these

Most wrong numbers in this repo are population confusions, not arithmetic errors.

| Set | N | Definition |
|---|---|---|
| Raw gage list | 3,211 | `gages_3000.csv`. **Pre-fix baseline population** — includes 513 phantom-zero single-divide gauges |
| DA_VALID | 2,859 | after the drainage-area validity filter |
| **Training / eval set** | **2,365** | after the `gages_adjacency` filter (dropped 494 headwater). **Every trained median is on this set** |
| Post-fix baseline population | 2,698 | 3,211 − 513 headwater. `ddrs plan` baselines from 2026-07-29 onward |
| Global matched set | 5,224 | a **different network** (global MERIT). Only in the removed `6_19_26_journal.md` (git history: `git show 339da86:6_19_26_journal.md`) |
| Area-balanced set (2026-08-02) | 1,841 | `~/projects/ddr/references/gage_info/gages_2000_area_balanced.csv`, built by `scripts/build_gages_2000_area_balanced.py` (seed 42) from GAGES-II with `DA_VALID` recomputed as **relative** `ABS_DIFF/DRAIN_SQKM ≤ 10%`, ≥80% obs coverage in both the 1981-10→1995-09 and 1995-10→2010-09 windows, non-headwater subgraph required. All 582 basins ≥5,000 km² kept + 418 random from [1k,5k) + 841 (all available) <1,000 km² → 45.7%/54.3% either side of 1,000 km². **Metrics on this set are incomparable to every 2,365/2,698-gauge number**; switching `data_sources.gages` to it invalidates the cached summed-Q′ baseline (`ddrs plan` recomputes) |

## Benchmarks — CONUS, eval 1995-10-01 → 2010-09-30, 2,365 gauges

| Quantity | NSE | KGE | Source |
|---|---|---|---|
| **Summed-Q′ baseline (dHBV2-UH store) — the CONUS bar** | **0.6781** | **0.7172** | `research/findings/2026-06-23-precip-disaggregation-findings.md` |
| **Best documented trained result** — precip-driven disagg + L1, run `2026-06-23T02-49-12Z-conus-hourly-train-and-test` | **0.7152** | **0.7106** | same |
| Δ vs baseline | **+0.037** | **−0.007** | same |
| **p = 21 fixed, `nse-batch` + Adam, daily flat (no disagg), gages_3000 population**, run `2026-09-08T15-55-52Z-conus-train-and-test` | **0.7200** | **0.7537** | `.ddrs/runs/<id>/plots/metrics_summary.json`; own-baseline recompute on the same 2,365 gauges 0.6785 / 0.7171, so Δ **+0.042 / +0.037**, the first dual win on the dHBV2-UH store. Gain grows with drainage area: +0.02 below 1,000 km², +0.11 at 5k–10k, +0.16 at 10k–30k. First run whose parameters plateaued (n median 0.130 → 0.053 by epoch 10, then < 0.4 % of range per 10 epochs). Findings: `research/findings/2026-09-08-p21-nse-batch-conus-findings.md`. |
| Precip-disagg + `nnse-kge` (`2026-06-24T00-03-01Z`) | 0.710 | 0.710 | no dual win |
| Precip + temperature, L1 (`2026-06-24T02-10-49Z`) | 0.716 | 0.709 | temp does not earn its keep |
| daily-OFF flat repeat-24 (`2026-06-05T01-41-16Z`) | 0.700 | 0.724 | |

Precip contribution decomposition (ON−OFF / OFF−base / ON−base):
NSE +0.020 / +0.018 / +0.037; KGE +0.018 / **−0.025** / −0.007. Reading: bare disagg
trades KGE for NSE; real precip timing rescues what bare disagg destroys.

### The KGE claim — restate it carefully

The long-standing claim was "KGE does not beat the summed-Q′ baseline in any
config". That was true through 2026-07-06 for the dHBV2-UH store.

**As of 2026-07-30 it needs qualification.** Run
`2026-07-30T00-24-24Z-train-and-test` on the newer
`daily_dhbv2_distributed_aorc2f_merit_unit_catchments.ic` store scored median
NSE **0.6799** / KGE **0.7194** on 2,365 gauges against its own eval-subset baseline
of 0.6744 / 0.7082 — beating it on **both** metrics. But its absolute NSE is well
below the 0.7152 benchmark, and the store differs, so this is not a like-for-like
improvement. **No findings doc covers this run.** Treat it as unwritten evidence:
do not repeat the blanket claim undated, and do not upgrade it to a result either.

## Closed campaigns — do not re-open

### Leakance (GW–SW exchange): **CLOSED — NO-GO, 2026-07-06**

Authority: `research/findings/2026-07-06-leakance-nogo-scientific-summary.md`. Read §3 before
proposing any retry.

The term is code-complete and gradient-exact. **Do not remove it.** But it is not
promotable, and identifiability is **REFUTED — not "pending"**.

| Finding | Value |
|---|---|
| Recovery ratio R1, clean objective, 58 planted reaches | **0.008** (bar ≥ 0.5) |
| Noise floor after the Phase B fix | 1.5 → **0.11 m³/s** — and recovery did **not** improve ⇒ the noise floor is REFUTED as the cause |
| Term usage | active on 78.2% of 64,892 eval reaches, Σ\|zeta\| = 1485 m³/s — not collapsed |
| Planted-reach share of that flux | **0.1%**; zero plants in the top-10 zeta reaches ⇒ the optimizer smears rather than localizes |
| Phase C Leg 1 (skill) | losing-subset ΔKGE +0.006 (bar ≥ +0.01) — FAIL |
| Phase C Leg 2 (equifinality) | Δn IQR 0.0143, ρ(Δn, zeta_net) +0.079 — PASS |
| Phase C Leg 3 (external) | ρ(\|zeta\|, bed-relative WTD) **−0.355** (bar > +0.3, and wrong sign) — FAIL |

**The mechanism, which is the durable lesson:** a gauge observes Σ(flux) over its
entire upstream network, and a sum is not invertible for its addends. Any per-reach
term whose only supervision is downstream discharge is structurally
non-identifiable — invariant to optimizer, objective, input richness, and data
volume. Every rival explanation (gradient starvation, objective noise, uninformative
inputs, sign ambiguity) was individually REFUTED.

**Superseded — do NOT cite as current:** R1 = 0.009 · "~130× noise floor" as the
live blocker · "Phase B is required before an identifiability claim" · the 2×2
"GO — marginal" verdict as a promotion claim · "K_D widening is the top follow-up"
(it was widened to `[1e-8, 1e-5]` in Phase C and the field is still anti-physical) ·
"K_D widening is NOT recommended" (moot). **Never write "leakance is identifiable"
in any form.**

### Selective equifinality (H1–H6): **all INCONCLUSIVE or REFUTED-not-clean**

Authority: `research/findings/2026-07-07-lstm-equifinality-v2-findings.md` and
`research/findings/2026-07-09-h5-h6-equifinality-v2-findings.md`.

| # | Registered verdict | Key number |
|---|---|---|
| H1 geometry converges, n diverges | **REFUTED** | like-for-like n relative spread 0.4512 exceeds every geometry spread — the direction *reverses* |
| H2 n-divergence predicted by Q′ disagreement | **REFUTED** | ρ = −0.248 reach-scale, **−0.380** network-scale |
| H3 gradient alignment is selective | **REFUTED** | mean cosine n 0.656 > p 0.383 > q 0.361 — n aligns *more* |
| H4 gradient decays with gauge distance | **INCONCLUSIVE** | 3 S / 3 R / 3 I across 9 cells |
| H5 parameter-swap transfer | **INCONCLUSIVE** | the control (+0.1251) **exceeds** the primary (+0.0953) ⇒ source-disagreement effect ≈ 0; 10/2,340 gauges carry 82% of the penalty |
| H6 loss-landscape degeneracy | **INCONCLUSIVE** | anisotropy 44× (R1) / 65× (R3), but **n is the stiff axis and p the sloppy one — the inverse of the campaign's framing** |

**Cite NEITHER direction.** The one audit-robust fact worth keeping is
geometry-gradient orthogonality across distinct Q′ stores (raw cosine 0.023–0.095 vs
ceilings 0.39–0.59).

Do not cite v1's "isotropic bowl" or "neither swap moves the loss" framings — the v1
statistics were invalidated by the v2 audit (v1 compared against *unpaired* window
std; the correct paired test has 15–40× smaller variance, and the 5%-sublevel
contour saturated, swallowing 100–105 of 121 grid points).

### Q′-store waves (2026-07-16) — carry the population caveat

Trained medians on 2,365 gauges: AORC2F distributed 0.3437/0.3256 · AORC2F lumped
0.5259/0.5175 · daily-lstm 0.5674/0.6169 · hourly-lstm 0.5543/0.4852. None beats the
0.7152/0.7106 benchmark.

⚠️ **Every "Δ vs own baseline" figure in `research/findings/2026-07-16-*` and
`research/findings/2026-07-07-lstm-equifinality-findings.md` is population-inconsistent** — the
baseline column is the 3,211-gauge median (including phantom zeros) while the trained
column is 2,365 gauges. Recomputed population-matched, the hourly-lstm "+0.022 NSE
gain from routing" **reverses to −0.051**. Those docs need a correction note.

### Synthetic-n recoverability — INTERIM, 1 of 4 arms

`research/findings/2026-07-22-synthetic-n-recoverability-findings.md`. S1–S5 are **not yet
computable**. Arm-1 preview (explicitly not a verdict): n median abs err 0.0354,
corr(truth, recovered) 0.736, slope vs `log10_uparea` −0.0193 recovered vs −0.0421
true (right sign, ~54% attenuated). Two caveats that must travel with any S3 result:
teacher and all four students share the same **frozen** disagg head (report as
"consistent under a shared, frozen disagg head"), and these four arms are **not** the
paper's R1–R5.

## Do-not-use list

| Number / claim | Why |
|---|---|
| `0.689 / 0.723` as a CONUS bar | Global MERIT, 5,224 gauges, different network. Use 0.6781 / 0.7172 |
| `+0.026` NSE improvement | Computed against the global baseline. The correct value is **+0.037** |
| Any "own baseline" NSE from the 07-07 / 07-16 docs | 3,211-gauge population including 513 phantom zeros. KGE is unaffected (phantom gauges are NaN-KGE and were dropped) |
| R1 = 0.009 / "130× noise floor" as the live blocker | Superseded by R1 = 0.008 on the fixed objective |
| "leakance is identifiable" (any phrasing) | Explicitly forbidden by the NO-GO summary §7 |
| H1–H6 in either direction | INCONCLUSIVE |
| "KGE has never beaten the baseline", undated | Needs the 2026-07-30 qualification above |
| Dense-grid landscape runs on a binary before `658cbfc` | Leaked the autodiff tape per forward-only eval (77 GB); fixed 2026-09-08 by running backward in `Objective::eval`, see traps.md T13 |
| The 84-gauge full-year census (`landscape-p21-census41/2026-09-09T01-52-19Z`), or any census run on a binary before `964f062`, for "share of gauges at optimum" | Unbounded Newton step landed on the search-box corner and reported zero iterations, which read as already-at-optimum; superseded by the 2,365-gauge sharded census, §Landscape census above |
| The (n, q) landscape with a depth axis by default | User: the axis should show post-transformation q, not depth |
| alpha_q_star or q half-widths as a reported optimum | The Hessian at the optimum is a saddle along q at 53 % of well-fit gauges (§Why gauges are not at their roughness optimum, F). Confirm with the 1-D line scan at n\* before quoting a q optimum |
| The integer `lag_days_*` columns of `covariates.csv` | Coarse to the day; use the fractional lags in analyst D's `D_wellfit_with_fractional_lags.csv` (`early_r`, `early_q`, `delay_channel`, positive = early) instead |
| The 15-year all-gauge census, cited as a result | Held at launch (12 shards would need about 140 GB); scope pending the user, see §Landscape census, all 2,365 test gauges |
| `\|mean(g)\| / mean(\|g\|)` as a gradient-alignment statistic | Not robust on heavy-tailed per-gauge gradients: swung 0.990 to 0.149 at large basins between two samples of the same run while the sign share moved under a point. Use the share of gauges with `grad0_n < 0` and the 10 % trimmed alignment (findings §21.3) |
| The recoverability figures "9 % global fix" and "31 % one-year transfer" | Artifacts of the two-point curvature `k = gain / a*^2` diverging where a gauge's optimum sits on its trained point; 32 real 25 x 25 grids show those parabolas predicting NSE falls of 7 to 26 that do not occur. Use **about a third at n x 1.5, worth +0.009 median NSE** (findings §20.4) |
| "median NSE 0.700 against 0.720" for the two p = 21 models | Each model scored on its OWN test set, derived from its own training list. On the 1,323 shared gauges over the same 15 years it is **0.7384 area-balanced against 0.7330 gages_3000**, a 0.005 gap running the other way (findings §22) |
| "statistically indistinguishable skill" for those two models | The paired sign test is z = -9.1. Say "practically identical median skill" (findings §22) |
| A landscape summary.csv from a study whose `window_days` slice is narrower than the run's configured window, on a binary before `eb3f159` | A gauge empty in the slice panicked the whole arm thread, silently truncating the population (720 of 1,841 in one case). Now skipped with a logged count in `manifest.notes` |

## Structural constants (stable)

CONUS 346,321 reaches / 338,814 edges · eval network 64,892 reaches for the
2,365-gauge set (**132,336** for the 3,211-gauge LSTM set — these are different
numbers and were conflated) · global fabric 2,939,408 reaches, 6,051 gauges ·
BURN 0.21 · rskan tag `v0.1.3` · V1 gate < 1e-3 m³/s.

The sparse backward lives in **`src/sparse/`** (`mod.rs`, `dispatch.rs`,
`cusparse.rs`): four retired skills (and, until fixed, CLAUDE.md) cited it as
a file, src/sparse.rs, which does not exist. `TimestepLeakanceOp: Backward<I,8>` is defined in
`src/routing/mmc_op.rs`, not in `src/routing/leakance.rs` (which exports
`zeta_forward` / `zeta_backward` / `ZetaGrads`).

## Evidence standard

The house rules that produced the results above, worth keeping:

1. **Pre-register hypotheses in a spec before running anything.** The hypothesis
   table must derive from the spec, not be reverse-engineered from results.
2. **Three verdict states only: SUPPORTED / REFUTED / INCONCLUSIVE.** Never
   "confirmed", "partially supported", or "likely".
3. **Define the gate as a single boolean on a computable number, before the battery
   runs.** Spend GPU only if it opens. When a gate fails, write "the gate FAILED",
   not "we decided not to proceed".
4. **Order instruments cheapest-first**: adjoint reachability (no training) →
   detectability bound (forward-only; detect if `|mean ΔQ| > 99th-pct rerun noise`
   **and** `> 5% of the gauge's mean flow`) → synthetic recoverability (full
   training). **If detectability is NO-GO, stop — no training objective can learn the
   term.** Run the adjoint map at *both* cold and trained points to separate
   "converged-flat" from "never-saw-signal".
5. **Every numeric claim carries a unit, a gauge count, and an eval window.**
   "median NSE 0.715 (2,365 gauges, 1995/10–2010/09)", not "NSE was good".
6. **When a finding overturns a prior doc, name the prior doc and the exact item.**
   The NO-GO summary §5 is the model implementation.
7. **A positive control needs a continuous baseline eval before training starts**,
   and the recovery target must be the *flux field*, not a degenerate parameter
   triple.
8. Watch the dam/lake regulation confound in any differential-gauging argument —
   restrict detectability sites to GAGES-II Ref class.

## Doc conventions

| Doc type | Location | Naming |
|---|---|---|
| spec (before code runs) | `research/specs/` | `YYYY-MM-DD-<slug>-design.md` |
| plan (tasks from a spec) | `research/plans/` | `YYYY-MM-DD-<slug>.md` |
| findings (after it ran) | `docs/` | `YYYY-MM-DD-<slug>-findings.md` |
| handoff (mid-experiment) | `docs/` | `YYYY-MM-DD-<slug>-handoff.md` |
| reference (data contract, API) | `docs/book/reference/` or `docs/` | descriptive, no date |

A findings doc opens with the header block (spec / plan / script / prior finding),
then a **one-line verdict** before any section, then §1 pre-registered hypotheses,
§2 methods, §3 results with bold verdicts, §4 conclusions, §5 next steps (dropped
items labeled "Dropped — reason"), §6 raw output, §7 reproduce. If a findings doc for
that experiment exists, add a datestamped section rather than creating a duplicate.

Always document the **binary provenance** in a methods section — the 2026-07-01 2×2
was invalidated by a stale binary and the manifest did not reveal it.

## Adjoint influence map — pair PoC (2026-09-04) + 41-gauge nested-reference population (2026-09-05), one seed

Authority: `research/findings/2026-09-05-adjoint-influence-conus-findings.md` (population) and
`research/findings/2026-09-04-adjoint-influence-poc-findings.md` (Juniata pair, with the
dhbv2-dist correction). Handoff + tables:
`.ddrs/experiments/adjoint-conus/2026-09-05T16-28-57Z/figures/{HANDOFF,STATS,README}.md`.

Five tau=9 arms (daily-lstm, hourly-lstm, uh-retro, dhbv2-lumped, dhbv2-dist
= run 2026-08-09T03-05-54Z; all `epoch_30_mb_1`, all trained on
`gages_2000_area_balanced.csv`), 20 GAGES-II Ref downstream gauges with nested
training gauges + 21 upstream partners. Finite-difference gate passed in every
arm (max rel. err 0.02–0.17 %).

- **Celerity is inflow-source dependent, ordered, and scale dependent.** Per-gauge
  max/min cross-arm celerity ratio: median 2.16 (low flow), 2.32 (high). dhbv2-lumped
  slowest at 26/38 gauges; dhbv2-dist fastest at 19/38 (low) and 26/38 (high). Ratio
  1.25 in the 520-reach basin 06452000, > 3.3 in 8–48-reach subgraphs. Within-arm
  gauge-to-gauge spread (IQR factor 2–3) exceeds the between-arm spread.
- **Volume bias transfers with coefficient 1.00** (median, every arm). Downstream
  bias is mostly local: inherited share median 0.27–0.39 (the Juniata LSTM arms'
  ~0.9 is not typical). Downstream bias sign follows the product: dhbv2-dist
  over-predicts at 14/20 gauges, dhbv2-lumped under-predicts at 13/20.
- **"Mass loss" is NOT mass loss (checks 3–4, 2026-09-07,
  `research/findings/2026-09-07-adjoint-volume-functional-checks-3-4-findings.md`).** The inflow
  gradient is exactly zero at source hours where lateral inflow sits at the
  `discharge` clamp floor, so the raw time-mean volume sensitivity of an
  intermittent reach collapses to its wet-hour fraction (0.13–0.21 at the Cannonball
  gauges in the UH product). Use `volume_sens_wet` (mean over wet source hours):
  median 0.88–1.02 at all 8 UH validation gauges, frac < 0.5 falls from 0.39–0.82
  to ≤ 0.29 (remainder = slow far reaches, ~0.08 m/s, water in transit). Pulse
  traces: +1 m³/s for 30 d at two "zero-kernel" reaches arrives at the gauge at
  97–98 %. Clamped-negative-solve mechanism REFUTED for wet reaches (no clamped
  reach on any losing path). The 180-day window exposes explosive linearised
  sensitivities (to 600) at the Feb–Mar wet-up in Plains basins — open item; do not
  aggregate raw 180-d volume sensitivity there. Population numbers with
  `volume_sens_wet`: rerun 2026-09-07 (see that doc when written).
- **Do-not-use:** the PoC's "dhbv2-lumped destroys mass in a 12-reach tributary" as
  a general claim (the population shows this is rare and regional); the Juniata
  0.21–0.47 m/s celerity range as a population number (use the ratios above).
  **Correction (2026-09-07, seed-noise entry below): the cross-arm celerity spread
  (median 2.16 low / 2.32 high) is no longer INCONCLUSIVE pending seed. The UH
  arm's seed-42/43 replicate gives a seed-to-seed celerity ratio median 1.09 (low
  anchor) / 1.01 (high), IQR 0.14, against the cross-arm max/min 2.16 / 2.32,
  IQR 0.69 / 0.87. The two distributions are nearly disjoint, so the cross-arm
  celerity spread is SUPPORTED at the population level: inflow source, not seed,
  sets the learned routing speed. Caveat: only the UH arm has a replicate seed, so
  the other four arms' noise floors are still unmeasured.**

## Adjoint seed-to-seed noise floor — UH arm seeds 42/43 (2026-09-07)

Authority: `research/findings/2026-09-07-adjoint-seed-noise-floor-findings.md`. Bundle
`experiments/adjoint-uh-seeds` (arms `uh-seed42`, `uh-seed43`; UH retrospective
inflow, `gages_2000_area_balanced.csv`, 30 epochs, identical config except
seed), scored against the 5-arm cross-arm reference above (41 gauges, 38 with a
celerity fit). Celerity: seed-to-seed ratio median 1.09 (low anchor) / 1.01
(high), IQR 0.14, versus cross-arm max/min 2.16 / 2.32, IQR 0.69 / 0.87. This
moves the population celerity result from INCONCLUSIVE to **SUPPORTED at the
population level**. Mass-type statistics (kernel mass, wet-hour volume
sensitivity, upstream→downstream transfer, inherited share) are seed-stable at
the median to 1e-5 or better. Caveats: only the UH arm is replicated (LSTM and
dHBV2 arms could have a different noise floor); two seeds cannot separate the
1.09 low-flow systematic shift from noise; a few outlier gauges (intermittent,
slow-transport) swing more between seeds without moving the medians. Reproduce:

```bash
target/release/ddrs --workspace .ddrs experiment adjoint-uh-seeds --backend cpu
~/projects/ddr/.venv/bin/python experiments/adjoint/seed_noise.py \
  .ddrs/experiments/adjoint-uh-seeds/<ts> .ddrs/experiments/adjoint-conus/2026-09-07T18-18-39Z \
  --out .ddrs/experiments/adjoint-uh-seeds/<ts>/figures
```

## Per-gauge loss landscape — UH arm sample case, Newport + Mapleton Depot (2026-09-07), two seeds

Authority: `research/findings/2026-09-07-landscape-uh-juniata-findings.md` (see §2b for the
seed replicate), spec
`research/specs/2026-09-07-adjoint-landscape-design.md`. Measures NSE-batch
loss at one gauge over basin-uniform log-multipliers on (n, p, q), the FD Hessian
of the adjoint gradient, the damped Newton optimum, and behavioural half-widths
per eigenvector. **Verdict (instrument): PASS**, but the multiplier
box must be bounded by the parameter ranges, or fields clamp and the landscape
flattens artificially. Newport (01567000): optimum n×0.36, p×0.26, q×0.56, NSE
0.692 → 0.770; eigenvalues 2.34e-1 / 1.03e-3 / 2.38e-4; the trained point sits
0.03 half-widths off the stiff axis, 0.86 off a sloppy one. Mapleton Depot
(01563500): monotone to the parameter-range floor, no interior optimum.

**Seed replicate (§2b, run
`.ddrs/experiments/landscape-uh-juniata-wide/2026-09-07T21-15-36Z`, compare
script `experiments/landscape/seed_compare.py`).** Seed 43 added as a second
arm. At Newport, seed 43's trained fields differ from seed 42's by a
near-uniform width shift: p × 0.63 (reach-std 0.04 in log), n × 0.95, q
unchanged. In seed 42's eigenbasis, seed 43's trained point sits 0.20
log-units along the stiff axis, which sets the seed-defined behavioural
tolerance at about 10 % of L* (replacing the 5 % placeholder above). The two
seeds' optima agree on n* to 2 % and on p* to 24 %, but differ on q* by a
factor 2.3 at equal NSE (0.770 vs 0.772): the gauge determines only the stiff
coordinate of its optimum. At Mapleton Depot, seed 42's optimum is the
clamped corner, so the seed comparison is not meaningful there; seed 43 finds
an interior optimum (NSE 0.558 → 0.652), so range-bounded Newton (a
follow-up) is needed before this gauge's optima can be compared.

Reproduce (findings §5): `target/release/ddrs --workspace .ddrs experiment
landscape-uh-juniata-wide --backend cpu`. **Status: two seeds of one arm, two
gauges; instrument PASS; tolerance measured; science still sample-scale
pending the 8-gauge run and cross-arm placement.**

## Landscape hypothesis tests (spec §7): census, trajectory, inputs (2026-09-08)

Authority: `research/findings/2026-09-08-landscape-hypothesis-tests-findings.md`. Runs: census
`.ddrs/experiments/landscape-uh-census/2026-09-08T13-52-27Z/` (8 gauges, UH seeds 42/43);
trajectory `.ddrs/experiments/landscape-uh-trajectory/2026-09-08T15-04-23Z/` (seed 42, init
to epoch 30); inputs `.ddrs/experiments/landscape-arms/2026-09-08T15-15-36Z/` (5 inflow arms).
**Census:** well-fit Juniata gauges want n × 0.3, p × 0.25 to 0.45, gains 0.07 to 0.10 NSE in
both seeds; poorly-fit White River optima sit near the trained point, gains at most 0.05;
Cannonball (NSE −11 to −37) has no descent direction, channel parameters cannot fix an inflow
error. **Trajectory:** Newport's stiff coordinate is inside the 5 % half-width from epoch 5
on; sloppy coordinates are still moving at epoch 30 but decelerating; Mapleton's stiff
coordinate stalls at 1.1 half-widths; the head moves a basin by a near-uniform factor.
**Inputs:** across five arms every gauge-optimal n at Newport (0.026 to 0.052) sits well
below trained (0.10 to 0.17), gains 0.04 to 0.20 NSE; the LSTM arms hit the range bound; the
optimum moves with input only at second order.

**Verdict:** batch compromise confirmed at well-fit gauges, direction set by the batch not
the input, regional not CONUS-wide; INCONCLUSIVE at population scale pending the 41-gauge
census. The dense-terrain, per-reach-gradient, and census41 bundles did not run: the 41×41
slice loop leaks memory (process grew to 77 GB), stopped under the user's three-strike rule.

**Population result (§9, §10 of the findings doc), on the two fixed-p (p = 21) models.** At
the 30 gauges present in both models' 2-D censuses (both are in both training lists), the
gages_3000 model's trained n is 0.48× the area-balanced model's: a CONUS-wide factor, not a
Juniata-membership effect. The gages_3000 model reaches n = 0.040 by epoch 10 and holds it
flat through epoch 30 (450 optimizer steps at 45/epoch, two learning-rate decays) even though
the area-balanced run made more total steps (870) and still left n at 0.100; this is
composition of the training population setting the batch's roughness, not step count. The
gages_3000 model sits at the per-gauge optimum at its well-fit gauges (median gain 0.001 NSE,
20 of 84), while all six well-fit gauges shared with the area-balanced model still want n ×
0.6, unchanged from the learned-p reading. **Do not use:** do not attribute the 0.04 vs 0.10
difference in trained n to pinning p: the clean twin (§8 below) shows p pinned alone leaves
n at 0.100, unchanged from learned p, at the same out-of-sample Juniata. **Open question:**
which gauges by drainage area supply the gradient that pulls n down; gages_3000 adds 1,370
gauges of median area 333 km² over the area-balanced list, but this is not yet separated from
the aggregate. A size-stratified training experiment is the direct test (not run; user
decision).

**Landscape-study infrastructure, this session, one line each:**
- The landscape search is 2-D (n, q only) for fixed-p arms: inactive axes masked out of
  Newton, Hessian, eigenvectors, half-widths, and slices (`active mask`, commit `692c00f`).
- The 41×41 dense-grid autodiff-tape leak (§4 above) is fixed by running a backward per eval
  in `Objective::eval` (commit `658cbfc`, `src/experiment/landscape/objective.rs`).
- Small basins (3-5 reaches) no longer refuse to step under the 5 % `max_clamped` rule: a
  per-reach floor, `max_clamped_min_reaches` (default 2), sets the effective bound to
  `max(max_clamped, max_clamped_min_reaches / n_reach)` (`src/experiment/landscape/mod.rs`,
  `output.rs`).
- Cross-model and depth-axis comparison tools added: `experiments/landscape/population_compare.py`
  (gauge-by-gauge trained-n ratio between two censuses) and `surface.py --depth-axis` (n-q
  plane rendered over basin-median n and gauge-reach depth under mean flow instead of raw q).

## Fixed width coefficient p (2026-09-08)

Authority: `research/findings/2026-09-08-fixed-p-assessment.md`. p = 21 (the DDR default)
is a dissertation-stage field fit to Juniata gages, not a literature constant.
In this model n and p enter depth only as the ratio n/p, so a gauge identifies
that ratio and not n and p separately. The only verified downstream
coefficients are Moody & Troutman (2002): w = 7.2 Q^0.5, d = 0.27 Q^0.3; the
candidate spatial function is `p = 7.2 · 0.27^(−q) · Q_ref^(0.5 − 0.3q)`. Arm
`config/experiments/uh_retro_pfixed21.yaml` (constant p = 21, learn n and q
only) started training 14:06Z 2026-09-08 (unit `ddrs-train-p21`); the
landscape objective now supports fixed parameters (commit `b52d966`).
**Follow-up (2026-09-08): the landscape search itself is now mask-aware, not
just the constant-broadcast forward pass.** Before this, `run_gauge`'s Newton
step, Hessian, and eigen-decomposition still treated p as a free axis even
when it wasn't learned, moving `alpha_p` off zero for no physical reason.
`Objective::active()` (derived from `kan_head.learnable_parameters`, same
source as the arm-open log line) now gates: `eval`/`hessian` force the fixed
component's gradient/Hessian row+column to exactly 0; `solve_active` masks
the Newton step so `alpha_p` never moves; `eig_active` drops the fixed
component to a `NaN` eigenvalue placed last; axis planes naming the fixed
parameter are skipped and logged; `stiff-sloppy` uses the two active
eigenvectors. Verified against the real `p21-conus` arm
(`.ddrs/experiments/landscape-p21-reachgrad/2026-09-08T17-48-24Z/`):
`alpha_p_star = 0` exactly at both Juniata gauges, `active_params =
"n,q_spatial"` in the netCDF. Python readers (`experiments/landscape/*.py`,
excluding `plot3d.py`) updated to read the new `active` variable and drop the
fixed parameter from tables/panels; all-active output (`landscape-uh-census`)
verified byte-identical before/after. Decision on the Moody-Troutman p(A) arm
is pending the user.

**Both p = 21 runs finished (2026-09-08).** `2026-09-08T14-06-12Z-train-and-test`
(the clean twin: `uh_retro_pfixed21`, area-balanced 1,841-gauge population, same
config/seed as the learned-p arm otherwise) scored median NSE 0.700 / KGE 0.736
on the test population, against learned-p's 0.707 / 0.738, indistinguishable.
`2026-09-08T15-55-52Z-conus-train-and-test` (the user's run, gages_3000 population,
which includes the Juniata gauges in training) scored median NSE 0.720 / KGE 0.754
on its own 2,365-gauge test set. **Pinning p costs nothing at the population
median in either case.**

At the gauge level, pinning p does NOT move n toward the gauge optimum by itself.
On the clean twin, with the Juniata still out of sample, trained n stays at 0.100
(learned-p arm: 0.103) and the per-gauge gain to the own optimum is unchanged
(Newport 0.661 → 0.742, Mapleton 0.514 → 0.607, both close to the learned-p gaps
of 0.08-0.10 NSE). Only the user's gages_3000 run, which trains on the Juniata
directly, lands n at 0.040. The gauge-optimal n scales with p as the n/p ratio
identifiability predicts: n/p ≈ 0.0029-0.0034 at Newport under both the learned-p
model (p ≈ 12.7, optimal n 0.037) and the p = 21 clean twin (optimal n 0.071),
to within the q trade. This refutes the earlier reading (§5-§6 of the findings
doc, corrected there) that pinning p had fixed the batch compromise: the clean
twin isolates the single change (p pinned, same population) and shows the 0.040
was a property of the training population (gages_3000, which contains the Juniata),
not of removing the n/p degeneracy. See findings doc §8 for the full twin comparison.

## Landscape census, all 2,365 test gauges, p = 21 model (2026-09-09)

Authority: `research/findings/2026-09-08-landscape-hypothesis-tests-findings.md` §11-13.

**Window policy (user decision 2026-09-09).** Every landscape bundle now uses full water
years: `window_days: 365` (one water year, `n_windows: 1`) or `window_days: 0` (the whole
eval axis, one window). 90-day seasonal windows are retired; a season scores only that
season's small variance and understates the model's real skill (Newport NSE at the trained
point was 0.70 over 90 days, 0.79 over the full year). Code defaults changed in `7352665`.

**Sharded workflow.** A full-population census runs as K parallel shards:
`scripts/landscape_shards.sh <bundle> <K> [backend]` launches `ddrs experiment <bundle>
--shard I/K` as transient systemd --user units against gauge source `all` (all 2,365 test
gauges, not a fixed list); `scripts/landscape_merge.py` concatenates the shards' `summary.csv`,
hard-links their `gauges/*.nc`, and writes one merged manifest that `census.py`, `conus_map.py`,
and `plots.py` read like an ordinary unsharded run. 24 shards covered all 2,365 gauges on
WY2000 in 1 h 47 min.

**Newton line-search fix (`964f062`).** The Newton step is now capped at 1 log unit per
iteration with a gradient-descent fallback and up to 16 step halvings. Before this fix, small
basins' unbounded Newton step landed on the search-box corner on the first trial and reported
zero iterations, which the code then read as "already at the optimum," inflating the
"fraction of gauges at their optimum" statistic. Verified on 01436000, 01435000, 01452000,
which now take 5 to 12 Newton steps instead of 0.

**Results (`landscape-p21-all/merged`, 24 shards, WY2000, p = 21 model epoch 30, Newton +
Hessian over (n, q), p pinned): 0.4 % of gauges range-bound, none with zero iterations, 4 %
used the gradient fallback.**

| WY2000 | count | median gain to own optimum | gain > 0.02 | median \|ln(n*/n)\| | within × 1.25 | wants slower | wants faster |
|---|---|---|---|---|---|---|---|
| well fit, NSE > 0.3 | 1,710 | 0.014 | 42 % | 0.67 (factor 1.95) | 19 % | 61 % | 19 % |
| NSE > 0.6 | 1,261 | 0.013 | 40 % | 0.62 | 22 % | 62 % | 16 % |
| poorly fit, NSE ≤ 0.3 | 655 | 0.027 | 55 % | 0.92 | 11 % | 55 % | 34 % |

By basin size among the well fit, median gain only reaches 0.03 above 200 reaches (107
gauges); it is 0.01 to 0.02 in every smaller size class, while median |ln(n*/n)| stays 0.4 to
0.65 across every size class: the trained n is far from most gauges' optima (median distance
factor 2, 61 % want slower) but moving there buys almost nothing (58 % of well-fit gauges gain
under 0.02 NSE).

**Verdict: n is weakly identifiable at the daily scale over most of CONUS.** This is why the
batch's n is set by population composition (see the gages_3000 vs area-balanced comparison
above), not by any individual gauge: the loss surface in n is nearly flat at most gauges, so
the gradient that sets the batch optimum comes from wherever the population happens to weight
it, not from a well-conditioned per-gauge signal. Map: `research/figures/2026-09-09-conus_n_gap_p21_wy2000.png`
(`experiments/landscape/conus_map.py`): the |ln(n*/n)| panel is red over much of the East and
the West Coast (wants slower routing), the gain panel is blue almost everywhere except the
large rivers.

**Juniata, full 15-year test period** (`landscape-p21-fulltest-juniata/2026-09-09T02-55-05Z`,
`window_days: 0`, 5,479 days, 1995-10-01 to 2010-09-30). The landscape's NSE at the trained
point matches the run's own eval on the same period to the second decimal (Mapleton 0.841 vs
0.847, Newport 0.853 vs 0.858), which validates the landscape objective (hourly routing, daily
pooling under the training tau, warm-up excluded) against the production eval path. Per-gauge
gain over the full period is +0.004 at Mapleton and +0.03 at Newport, both smaller than the
WY2000-only gains (+0.02 and +0.07): a single year overstates what a gauge could gain. **The q
optimum flips sign between windows while n does not**: over WY2000 the optimum pushes q to its
floor, over 15 years it pushes q up (× 3.6 at both gauges); n moves the same direction in both
windows (up, × 1.2 to 2). This is the operational definition of a poorly constrained
parameter used in the paper: its optimum changes sign with the evaluation window while the
loss barely moves.

## Why gauges are not at their roughness optimum (2026-09-09, swarm synthesis)

Authority: `research/findings/2026-09-09-why-not-at-optimum-findings.md` (six-analyst swarm synthesis, reports
`research/why-analysis/A-flat-q.md`, `B-clamping.md`, `C-weak-gradient.md`, `D-inflow-bias.md`,
`E-training-side.md`, `F-equifinality.md`) and `research/findings/2026-09-08-landscape-hypothesis-tests-findings.md`
§14.

**Five-year census facts** (`landscape-p21-all-5yr/merged`, WY1996 to WY2000, all 2,365 gauges,
16 shards, 12.7 h): 2,124 gauges well fit (NSE > 0.3 at the trained point), median NSE 0.754,
median gain to own optimum 0.010, median |ln(n*/n)| 0.72, 65 % want slower routing and 17 %
want faster. The longer window raises the well-fit count and lowers the median gain versus
WY2000 alone (0.010 vs 0.014): a single year overstates what a gauge could gain, and the
systematic "wants slower" bias (65 %) is CONUS-wide, not a WY2000 artifact.

**Consolidated partition of the 2,124 well-fit gauges** (six independent analyses of clamping,
flat q, weak gradient vs travel time, inflow timing, training mechanics, and Hessian structure,
reconciled against each other in the synthesis):

| class | share | median gain | who they are |
|---|---|---|---|
| at the optimum | about 20 % | 0.001 | all sizes and regions, including the 24 gauges beyond eight days of travel time |
| equifinal: inside the behavioural set or on flat ground | about 45 % | 0.005 to 0.009 | small or steep basins, travel time under a day, routed flow on time within a quarter day; Western Mountains is the largest regional block (63 % want slower, gain 0.004) |
| real error along the stiff axis | about 35 % | 0.028 to 0.029 | larger, floor-slope, rain-fed rivers with one to four days of travel; routed flow early by a quarter day or more; 76 to 93 % want slower, n × 2.5; ecoregion shares Eastern Highlands 51 %, Southeast Plains 49 %, Northeast 40 %, Central Plains 39 %, Western Mountains 14 % |
| degenerate optimum | 16 % (overlaps the rows above) | 0.014 | q on the box edge (11.6 %) or λ₁ ≤ 0 (86 gauges) |

The poorly fit 241 gauges sit outside this partition: an intermittent Xeric/Plains subset
(about 60 gauges) is clamp-marked, and a Western Mountains subset (104 gauges) has the right
inflow volume and the wrong snowmelt timing.

**Verdicts.**

- **Clamping: refuted for the fitted population.** No well-fit gauge has a trained n or q
  within 1 % of a range edge. Routed floor days occur at 2.2 % of well-fit gauges, observed
  zero-flow days at 8.2 %, and those gauges are steeper (not flatter) and want no change in n.
  Clamping only marks the ~60 arid intermittent gauges among the 241 poorly fit.
- **Low flow: refuted as the cause of flat q.** Low-flow and zero-flow fractions do not
  correlate with q flatness (rho −0.04, −0.01). Flatness in q instead falls monotonically
  with depth at the gauge (94 % below 0.3 m, 28 % above 4 m), which rules out the
  symmetric-leverage prediction that flatness peaks near 1 m depth.
- **q is flat at 85 % of well-fit gauges by construction, not by accident.** A factor-2 move
  in q changes the loss by under 1 % at those gauges; n is flat at only 11 %, and no gauge is
  flat in n but curved in q. The 15 % where q is identifiable are deep, large, low-slope main
  stems (median depth 1.5 m, area 2,600 km²).
- **n is identifiable above about one day of travel time.** Curvature in n rises with channel
  length (rho +0.31), area (+0.27), and the travel-time proxy (+0.26); median |H_nn| is 0.006
  to 0.008 below one day of travel and 0.023 to 0.025 at one to four days.
- **The trained channel adds about zero days of delay, so routed flow inherits the inflow's
  early timing.** Median added delay is 0.00 days (68 % of gauges under 0.1 day). The
  displacement correlates with "routed flow arrives early" at Spearman +0.73 (fractional
  lags), and a joint linear model explains 45 % of the displacement variance (78 % boosted),
  with timing dominant and volume bias under 1 %.
- **Loss weighting and step count are refuted.** The NSE-batch per-day weights are nearly
  uniform; reweighting would move the median displacement by a partial contribution of only
  0.03. The run made 60 optimizer updates and n was settled by the twentieth.
- **Head attributes explain only 12 % of ln n\* variance.** Attribute nearest neighbours still
  disagree by a factor 1.75 in the gauge-optimal n (random pairs disagree by 2.2), and the
  head's trained ln n spread is half of what the gauges want.

**Ranked training changes** (highest expected effect first):

1. **Width as a function of river size (p(A) or p(Q_ref)), not a fixed coefficient.** Targets
   the 35 % real-error class directly: the class wants n × 2.5 to 9 with q pinned to the floor,
   consistent with a channel too narrow for large rivers (a 21 m channel at Newport on an
   8,700 km² basin). Expected effect is the largest of any single change.
2. **A learnable per-basin timing term** (inflow delay, unit-hydrograph scale, or a learnable
   tau), trained jointly. Routed-flow timing is the strongest covariate of both displacement
   (rho +0.73) and gain (drop-one dR² 0.14 of 0.31). Run after item 1 to measure the residual.
3. **Attributes that carry routing-timing information**: channel width or width-to-depth from
   GRWL, sinuosity, floodplain/wetland fraction, tile drainage/cropland, reservoir/lake storage.
   Targets the 12 % ceiling on attribute-explained ln n\*.
4. **Stop learning q from daily discharge where it is unidentifiable** (flat at 85 % of
   gauges, a saddle at 53 % of optima, sign flips between windows): prescribe q, tie it to a
   downstream hydraulic-geometry relation, or learn it only where an hourly test shows
   curvature.
5. **An hourly diagnostic at 60 gauges** spanning the depth and travel-time bins, run first as
   a cheap check: if |H_nn| and q curvature rise 2 to 5× at short-travel-time gauges under an
   hourly step, the daily objective (not the physics) is suppressing identifiability there and
   an hourly or timing-aware loss is warranted; if not, item 4 stands.

Area/sigma reweighting and more optimizer steps rank last: partial contributions of 0.03, and
n was already converged by update 20 of 60.

**Equifinality statements.**

- q is unidentifiable from daily discharge at 85 % of gauges: any q in the range gives the
  same five-year loss within 1 %, and learning it per reach there learns noise.
- n is unidentifiable below about one day of channel travel time (45 % of well-fit gauges):
  median |H_nn| 0.006 to 0.008, gain only 0.009 despite a median displacement of a factor 4.4.
- The trained point sits on a ridge or shoulder of the per-gauge surface at most gauges
  (H_nn ≤ 0 at 46 %, H_qq ≤ 0 at 69 %), while the optimum itself is convex in n at 96 % of
  gauges: the batch compromise lands between conflicting gauges, not inside any one gauge's
  bowl, so trained-point curvature understates identifiability.
- The Hessian at the optimum is indefinite (a saddle, almost always along q) at 53 % of
  well-fit optima: the reported q\* is often not a minimum.
- Real error is the complement: about 35 % of gauges, one direction (slower), concentrated
  east of the Rockies, worth 0.03 NSE at the median and up to 0.45 at the ten most egregious
  rivers (n × 3.9 to 8.6, low-gradient agricultural/coastal-plain rivers of the Midwest and
  Southeast).

## Open, not closed

- **tau is mis-set (pilot-strength, 2026-08-06).** WY1996 sweep on the epoch-30
  area-balanced checkpoint: NSE(tau) plateaus at tau ≈ 14–19 in every bin
  < 30,000 km²; a single global tau=19 gains +0.114 median NSE (0.546 → 0.660,
  1,841 gauges, WY1996) and tau=16 ties the summed-Q' baseline in the
  < 1,000 km² bin (0.677 vs 0.674). Sign convention: window offset vs the
  scored day's UTC midnight is (tau − 11) h; larger optimal tau ⇒ model LATE
  vs obs. **Adversarial-review correction (same day): this run had the disagg
  head OFF (flat repeat-24), so the hourly signal is 97.6% UTC-day-constant and
  the sweep resolves only a day-pairing + blend weight (~half-day), NOT
  sub-daily phase — do not read tau=16/19 as Eastern/Pacific midnight.** The
  robust covariate is drainage area (Spearman +0.16 uncensored, an
  accumulated-lag signature); the longitude/timezone fingerprint is
  absent-to-contradicted (sign flips uncensored). 596 of 661 tau=23 pins are
  real optima beyond the sweep edge ⇒ Phase 2 needs the ±1-day mapping
  extension, split-sample selection, and a re-sweep on a disagg-ON run to test
  for any hour-scale signal. Training also runs at tau=3, so gradients have
  always been ~half a day misaligned — retrain at corrected tau is the open
  test (freeze the tau protocol first). Instrument: `DDRS_HOURLY_DUMP` env var
  on `evaluate` + `scripts/tau_sweep.py`. Authority:
  `research/findings/2026-08-06-tau-sweep-pilot-findings.md` incl. §5a corrections.
  **Interpolation arms (§5c, same day):** replicated on the standard 2,365-gauge
  population — argmax tau=18–20, small-basin global tau=19 beats baseline
  0.674 vs 0.645 (WY1996). Linear/quadratic q' upsampling
  (`DDRS_QPRIME_INTERP`, commit e4fb66d) neither sharpens nor shifts the
  curves ⇒ the mis-set is not a step-function artifact, and interpolation is
  NOT the fix (nearest ≥ linear ≥ quadratic at each optimum). ~30% of gauges
  pin at tau=23 (optima beyond +12 h) and best_tau correlates with area, not
  longitude ⇒ likely convention offset + area-growing lag ("double routing" —
  DDR's own tau docstring). Mechanistic prior: USGS obs are LST
  midnight-to-midnight (no DST), AORC/Q' stores are UTC; `src/data/` has no
  timezone logic anywhere (§5b).
  **Cross-source arms (§5e/§5f, 2026-08-07):** same checkpoint/network,
  streamflow store swapped (`config/experiments/tau_src_*.yaml`). Four of
  five stores replicate the tau 17–21 optimum (aorc2f distributed 20, UH
  retro 21, daily LSTM 17, hourly-native LSTM 19; per-gauge best-tau median
  18–19 on all four) ⇒ the lag is shared upstream of store choice — and the
  routing-free discriminator (§5g) resolved the attribution: it is the
  common MC routing, NOT the day convention.
  **Exception: `aorc2f_lumped`** — obs-free cross-correlation shows its
  routed hydrographs LEAD the reference by ~23 h; CF metadata are
  byte-identical to the distributed store (convention candidate REFUTED), so
  the shift is in the data the lumped pipeline wrote. Do not use it for
  timing-sensitive comparisons. The hourly-native arm does NOT sharpen the
  curve (not a floor effect) and its longitude correlation is a censoring
  artifact (interior sign +0.075, wrong sign) — timezone fingerprint remains
  absent-to-contradicted even with native sub-daily data. Fable review
  (§5f): claims sound-with-corrections; extended sweep −13..47 shows the
  constant-tau optimum is interior (20) but 33% of per-gauge optima lie
  beyond tau=23. Plot: `output/tau_sweep/cross_source_nse_vs_tau.png`.
  **Routing-free summed-q' sweep (§5g, 2026-08-07) — the attribution
  answer.** Summed daily q' repeat-24'd in the arms' phase, same sweep
  machinery (tau=11 reproduces the cached baseline NSE, median |diff|
  0.0003). Optimum tau=6 global; per-bin 9 / 3 / −3 / −8 with area ⇒
  (1) **UTC-vs-LST convention REFUTED as dominant** (zero-area intercept
  ≈ tau 10–11 ⇒ convention offset ≈ 0–2 h; explains every failed longitude
  fingerprint); (2) summed q' LEADS the gauge by area-growing travel time
  (2→19 h), so day-aligned scoring understates baseline skill in large
  basins; (3) **MC routing over-delays by ≈2× the required travel time**
  (routed lateness = summed earliness bin-by-bin; added delay 10→38 h) —
  the measured "double routing" of DDR's tau docstring; tau≈19–20 is
  compensation, root cause is routing timing (double-carried travel time
  and/or slow trained celerity, median n 0.130 vs reference 0.05).
  Routed-at-optimum still beats summed-at-optimum in every bin (+0.013 to
  +0.051). Plot: `output/tau_sweep/summed_qprime_vs_routed_tau.png`.
  **Convention change SHIPPED (2026-08-08, findings §5i):** tau is now
  signed-at-zero hours of advance (`[tau : -(24-tau)]`, day i ↔ obs day i,
  default 9 ≡ old 20); old scale = new + 11. Old checkpoints trained at
  old-3 ≡ new −8. **Retrain COMPLETE (2026-08-10, all five stores, 30
  epochs, CPU, 1,841 gauges, full test window): aorc2f distributed
  0.620 → 0.706 median NSE (+0.086 from timing alone; now beats the
  summed-q' baseline 0.642 by +0.064), UH retro 0.707, daily LSTM 0.578,
  hourly LSTM 0.564, aorc2f lumped 0.483 (WORSE, as pre-registered — its
  optimum is ≈ −8; the negative control behaved).** Merged to master in
  PR #33. Gotcha: a train-and-test resume from a final checkpoint cannot
  re-enter Phase 2 (needs checkpoints from its own Phase 1) — finish a
  killed eval with the legacy eval binary instead.
  **Gamma-UH params pulled (§5h):** the distributed aorc2f store's q' was
  exported (2026-07-29, water_loss) with each divide routed through its own
  learned gamma UH; `scripts/dump_gamma_uh_params.py` (water_loss venv)
  reads the v3_gradaccum ep100 Ann head: median kernel mean 1.50 days per
  divide (IQR 0.87–4.22), spearman +0.383 with log uparea ⇒ the per-divide
  q' carries area-dependent NETWORK travel time before MC routes at all —
  double routing is structural in the store. Clean fix target: sub-grid-only
  UH on lateral inflows; col-7 (unrouted) is not it (0.29 routed,
  water_loss 2026-07-29 finding). Dump:
  `output/tau_sweep/gamma_uh_params.csv`.

- **Backward CUDA graphs (SP-11).** Forward capture landed (V7a 0.385, V10 29.2%
  launch reduction); the backward pass is not captured. Path: profile → fuse backward
  kernels → capture. Blocked for leakance configs (the leakance kernel has no capture
  path).
- **Global scale-out.** `ddrs sources use global && ddrs plan && ddrs run`. Needs a
  1-epoch smoke and per-provider eval before any full run.
- **The paper** (`/home/tbindas/projects/ddr_equifinality/paper.tex`, "Beyond
  Equifinality in Differentiable River Routing", Bindas & Shen). Five arms R1–R5, four
  pre-registered hypotheses, leakance disabled (ζ=0) in every arm. Abstract, intro,
  and methods are drafted; the Results section is still a `\tbd{}` skeleton. Only the
  dHBV2 cross-family arms remain unrun.
- **Five-year and 15-year all-gauge censuses (running).** `landscape-p21-all-5yr` (WY1996 to
  WY2000, 16 shards) and `landscape-p21-all-15yr` (the full test period, 12 shards) repeat the
  §Landscape census methodology to check whether the n direction and the flatness in §Landscape
  census hold across years the way the Juniata pair suggests, or whether they are a WY2000
  artifact.
- **Which gauges set the batch's n.** gages_3000 adds 1,370 gauges of median area 333 km² over
  the area-balanced list and pulls the batch's trained n down by roughly half; a size- or
  region-stratified training run is the direct test of which gauges in that addition supply the
  gradient (not run; user decision, see §Fixed width coefficient p above).
- **p as a function of river size.** The Moody & Troutman-derived candidate
  `p = 7.2 · 0.27^(−q) · Q_ref^(0.5 − 0.3q)` is a candidate spatial function for p, not yet
  fit or tested against a learned-p model at varying basin size.
- **The p(A) retrain on gages_3000 is the highest-value experiment (§Why gauges are not at
  their roughness optimum above).** Retrain with width as a function of river size, everything
  else unchanged, then rerun the five-year census and the diagnostic pass. Targets the 35 %
  real-error class directly; costs one ~2 h training run plus a ~13 h census. Success: the
  early-arrival class shrinks toward the on-time share and class-iv gain falls below 0.01.
- **The hourly landscape at 60 gauges** spanning the depth and travel-time bins is the cheap
  companion, runnable today: it decides whether the 45 % equifinal share is a property of the
  daily objective or of the physics (§Why gauges are not at their roughness optimum, item 5).
- **The q line scan on the 60 saddle gauges.** A stratified 1-D scan in q at n\* (25 points
  over the box, about 2 h as 6 shards) decides whether the 53 %-of-optima q saddles are
  numerical or a second regime, and is the prerequisite for quoting any q\* or q half-width.
- **A per-gauge clamp accumulator** (floor hours, negative pre-clamp hours, depth/width floor
  reaches) added to the diagnostic pass, to close the clamping question at the hourly level.
- **PUR-7 grouping pending Table S4.** The seven PUR regions of Feng et al. (2021, GRL, doi
  10.1029/2021GL092999) are defined in that paper's Table S4; the SI was not retrievable, so
  the regional breakdown currently uses a provisional geographic grouping
  (`experiments/landscape/region_breakdown.py`), flagged pending the published table.

## Training convergence: the p = 21 CONUS model had not converged (2026-09-10, SETTLED)

Authority: `research/findings/2026-09-08-landscape-hypothesis-tests-findings.md` §19, §21, §21.3, §21.4.
Reproduce: `experiments/landscape/trainwin_compare.py <testing-merged> <training-merged> --out <dir>`.

Run `2026-09-08T15-55-52Z-conus-train-and-test`, checkpoint `epoch_30_mb_1`. Measured at 2,124 (testing) and
2,170 (training) well-fit gauges; paired on the 2,059 well-fit in both.

| | testing WY1996-2000 | training WY1991-1995 |
|---|---|---|
| share where the loss falls if n rises | 78.4 % (z = 25.8) | **78.9 %** (z = 26.2) |
| 10 % trimmed gradient alignment | 0.941 | **0.959** |
| share negative, n_reach > 200 | 90.0 % | 87.1 % |
| sign agreement between the windows | - | 84.2 % |

**Nonstationarity is refuted**: the training period wants more roughness exactly as the test period does, so the
trained point is not a stationary point of the objective the model was fitted to. Mechanism: gradient accumulation
over 20 micro-batches at 2 updates per epoch gives **60 optimizer updates in the whole 30-epoch run**, and n was
stationary from update 20 while the learning rate decayed 0.005 to 0.001 to 0.0005.

**What it is worth (§20):** a single global multiplier of n x 1.5 recovers about a third of the available 0.028
NSE, worth **+0.009** at the population median; a factor 2.7 on every channel in the CONUS moves the median by
under 0.01 in either direction. So more optimizer updates is the right change for **parameter defensibility, not
skill**. The exception is `n_reach > 200`, where the displacement is coherent and the gain is about 0.045. Moving
to the NSE optimum also raises KGE at 81 % of gridded gauges (§24), so it does not trade one metric for the other.

**Ranked consequence:** more optimizer updates (smaller accumulation, more epochs, or a flatter lr schedule) moves
to the top of `research/findings/2026-09-09-why-not-at-optimum-findings.md` §4, ahead of the attribute and architecture changes.

**Gotcha:** `landscape-p21-all-trainwin-diag` runs `newton_iters: 0`, so its `alpha_n_star` is identically zero.
Never join it to another census on that column; doing so yields a confident-looking 27.7 % sign agreement and an
apparent 0 % transfer, both artifacts (§21.4). Any study needing per-gauge optima on the training window must run
`period: training` WITH `newton_iters: 12`, at roughly the cost of the five-year testing census.

## Two p = 21 models: identical skill, roughness a factor 2.4 apart (2026-09-10)

Authority: findings §22 (skill) and §23 (roughness). The paper's central equifinality claim.

Arms: `2026-09-08T14-06-12Z-train-and-test` (gages_2000_area_balanced, 1,841 gauges) and
`2026-09-08T15-55-52Z-conus-train-and-test` (gages_3000, 2,365). Same config, seed and architecture; only the
gauge list differs. **Both comparisons use the same 1,323 shared gauges**, which is what makes the pair citable.

| on the 1,323 shared gauges | area-balanced | gages_3000 |
|---|---|---|
| median NSE, 1995-10-01 to 2010-09-30 | 0.7384 | 0.7330 |
| median KGE | 0.7695 | 0.7711 |
| median basin-median Manning n | **0.1036** | **0.0435** |
| median basin-median width exponent q | 0.402 | 0.121 |

Roughness ratio: median **2.39**, geometric mean 2.10, same direction at **96.6 %** of gauges, and flat across
basin size (2.38 / 2.40 / 2.39 for <=50, 51-200, >200 reaches), which is what distinguishes a population effect
from a composition artifact. Skill gap 0.005 against a median gauge-to-gauge difference of 0.021.

Harvest the trained field cheaply with bundle `landscape-pfixed-all-nfield`: the field is a KAN forward pass and
does not depend on the evaluation window, so it runs a one-year window with `newton_iters: 0`, `grid: 0`,
`series: false`. **Its loss and NSE columns are one water year and must not be compared to the five-year
censuses**; that mismatch is exactly what produced the withdrawn 0.700-against-0.720 figure.
