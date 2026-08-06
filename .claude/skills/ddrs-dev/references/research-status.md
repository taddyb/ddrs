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
| Global matched set | 5,224 | a **different network** (global MERIT). Only in `6_19_26_journal.md` (repo root, not `docs/`) |
| Area-balanced set (2026-08-02) | 1,841 | `~/projects/ddr/references/gage_info/gages_2000_area_balanced.csv`, built by `scripts/build_gages_2000_area_balanced.py` (seed 42) from GAGES-II with `DA_VALID` recomputed as **relative** `ABS_DIFF/DRAIN_SQKM ≤ 10%`, ≥80% obs coverage in both the 1981-10→1995-09 and 1995-10→2010-09 windows, non-headwater subgraph required. All 582 basins ≥5,000 km² kept + 418 random from [1k,5k) + 841 (all available) <1,000 km² → 45.7%/54.3% either side of 1,000 km². **Metrics on this set are incomparable to every 2,365/2,698-gauge number**; switching `data_sources.gages` to it invalidates the cached summed-Q′ baseline (`ddrs plan` recomputes) |

## Benchmarks — CONUS, eval 1995-10-01 → 2010-09-30, 2,365 gauges

| Quantity | NSE | KGE | Source |
|---|---|---|---|
| **Summed-Q′ baseline (dHBV2-UH store) — the CONUS bar** | **0.6781** | **0.7172** | `docs/2026-06-23-precip-disaggregation-findings.md` |
| **Best documented trained result** — precip-driven disagg + L1, run `2026-06-23T02-49-12Z-conus-hourly-train-and-test` | **0.7152** | **0.7106** | same |
| Δ vs baseline | **+0.037** | **−0.007** | same |
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

Authority: `docs/2026-07-06-leakance-nogo-scientific-summary.md`. Read §3 before
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

Authority: `docs/2026-07-07-lstm-equifinality-v2-findings.md` and
`docs/2026-07-09-h5-h6-equifinality-v2-findings.md`.

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

⚠️ **Every "Δ vs own baseline" figure in `docs/2026-07-16-*` and
`docs/2026-07-07-lstm-equifinality-findings.md` is population-inconsistent** — the
baseline column is the 3,211-gauge median (including phantom zeros) while the trained
column is 2,365 gauges. Recomputed population-matched, the hourly-lstm "+0.022 NSE
gain from routing" **reverses to −0.051**. Those docs need a correction note.

### Synthetic-n recoverability — INTERIM, 1 of 4 arms

`docs/2026-07-22-synthetic-n-recoverability-findings.md`. S1–S5 are **not yet
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

## Structural constants (stable)

CONUS 346,321 reaches / 338,814 edges · eval network 64,892 reaches for the
2,365-gauge set (**132,336** for the 3,211-gauge LSTM set — these are different
numbers and were conflated) · global fabric 2,939,408 reaches, 6,051 gauges ·
BURN 0.21 · rskan tag `v0.1.3` · V1 gate < 1e-3 m³/s.

The sparse backward lives in **`src/sparse/`** (`mod.rs`, `dispatch.rs`,
`cusparse.rs`) — four retired skills and CLAUDE.md cite a non-existent
`src/sparse.rs`. `TimestepLeakanceOp: Backward<I,8>` is defined in
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
| spec (before code runs) | `docs/superpowers/specs/` | `YYYY-MM-DD-<slug>-design.md` |
| plan (tasks from a spec) | `docs/superpowers/plans/` | `YYYY-MM-DD-<slug>.md` |
| findings (after it ran) | `docs/` | `YYYY-MM-DD-<slug>-findings.md` |
| handoff (mid-experiment) | `docs/` | `YYYY-MM-DD-<slug>-handoff.md` |
| reference (data contract, API) | `docs/reference/` or `docs/` | descriptive, no date |

A findings doc opens with the header block (spec / plan / script / prior finding),
then a **one-line verdict** before any section, then §1 pre-registered hypotheses,
§2 methods, §3 results with bold verdicts, §4 conclusions, §5 next steps (dropped
items labeled "Dropped — reason"), §6 raw output, §7 reproduce. If a findings doc for
that experiment exists, add a datestamped section rather than creating a duplicate.

Always document the **binary provenance** in a methods section — the 2026-07-01 2×2
was invalidated by a stale binary and the manifest did not reveal it.

## Open, not closed

- **tau is mis-set (pilot-strength, 2026-08-06).** WY1996 sweep on the epoch-30
  area-balanced checkpoint: NSE(tau) plateaus at tau ≈ 14–19 (local-midnight
  pooling) in every bin < 30,000 km²; a single global tau=19 gains +0.114
  median NSE (0.546 → 0.660, 1,841 gauges, WY1996) and tau=16 ties the
  summed-Q' baseline in the < 1,000 km² bin (0.677 vs 0.674). Shipped tau=3
  pools from 16:00 UTC, ~11–16 h out of phase with the local obs day. Training
  also runs at tau=3, so gradients have always been misaligned — retrain at
  corrected tau is the open test. Instrument: `DDRS_HOURLY_DUMP` env var on
  `evaluate` + `scripts/tau_sweep.py` (one eval run, then exact offline sweep).
  Caveats: single year; per-gauge best is in-sample; best-tau histogram pinned
  at the {0,23} edges ⇒ Phase 2 needs a ±1-day mapping extension. Authority:
  `docs/2026-08-06-tau-sweep-pilot-findings.md`.

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
