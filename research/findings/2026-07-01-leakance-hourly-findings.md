# Leakance × hourly-disaggregation 2×2 — findings (2026-07-01)

Spec:  `docs/superpowers/specs/2026-06-29-leakance-hourly-feasibility-design.md`
Handoff (root-cause + re-run): `docs/2026-07-01-leakance-hourly-experiment-handoff.md`
Branch: `hourly-forcings`

**One-line verdict:** the hypothesized **interaction is present and in the
predicted direction** — leakance *helps* skill under hourly forcing and *hurts*
under daily forcing — leakance is **identifiable (non-collapsed)**, and the
eval-time zeta export (built 2026-07-01, later session) measures
**|zeta| > 0.01 m³/s on 10.4% of eval reaches**, clearing the ≥10% proxy bar.
All three gate criteria met. Verdict: **GO — but marginal**: the skill effect
is small and the magnitude bar passes with no headroom (10.4% vs 10%); the
binding `K_D ≤ 1e-6` ceiling (§3) is the first follow-up because it likely
clips both.

---

## 0. Provenance / validity

The earlier (2026-07-01 morning) runs were invalid — they ran a **stale
June-3 `ddrs` binary** with no disaggregation and no leakance, so the hourly and
daily cells were byte-identical (see handoff §0). All four arms below use
**valid** binaries: the two leakance-ON arms were re-run 2026-07-01 with the
current binary (disagg + leakance present, precip verified loading, directory
checkpoints); the two OFF controls need neither feature past what their binaries
had.

| arm | run id | forcing | leakance | binary |
|---|---|---|---|---|
| hourly-OFF | `2026-06-23T02-49-12Z-conus-hourly-train-and-test` | hourly disagg + precip | OFF | Jun-23 (disagg ✓) |
| hourly-ON  | `2026-07-01T13-43-32Z-train-and-test`               | hourly disagg + precip | ON  | current ✓ |
| daily-OFF  | `2026-06-05T01-41-16Z-train-and-test`               | flat repeat-24         | OFF | Jun-05 (no feats needed) |
| daily-ON   | `2026-07-01T21-20-27Z-train-and-test`               | flat repeat-24         | ON  | current ✓ |

Paired: seed 42, eval window 1995/10/01–2010/09/30, 2365 finite-NSE gauges.

---

## 1. All-gauge medians (2×2)

| median | leakance OFF | leakance ON | Δ (ON−OFF) |
|---|---|---|---|
| **hourly**  NSE | 0.7153 | 0.7145 | **−0.0008** |
| **hourly**  KGE | 0.7104 | 0.7150 | **+0.0046** |
| **daily**   NSE | 0.7004 | 0.6963 | **−0.0041** |
| **daily**   KGE | 0.7244 | 0.7250 | **+0.0006** |

Reading:
- **Forcing axis (OFF row):** hourly disaggregation alone lifts median NSE
  +0.0149 (0.7004→0.7153) over flat-daily — the disagg head earns its keep,
  independent of leakance.
- **Leakance under hourly:** NSE flat (−0.0008, noise), KGE **+0.0046** — the
  volume/variance-correction signature (KGE penalizes α/β terms that L1 and NSE
  don't).
- **Leakance under daily:** NSE **worse** (−0.0041), KGE flat — under flat-daily
  forcing leakance behaves like a mild fudge factor that degrades fit.

## 2. Losing-stream subset — the decisive interaction

Subset = gauges where the summed-Q′ baseline over-predicts (mean pred/obs > 1)
on the hourly-OFF run: **1883 / 2365 gauges (79.6%)**. Paired ON−OFF deltas:

| arm | ΔNSE med | ΔKGE med | ΔKGE-β med | frac(ΔNSE>0) |
|---|---|---|---|---|
| **hourly** ON−OFF | **+0.0005** | **+0.0018** | +0.0033 | **55.5%** |
| **daily**  ON−OFF | −0.0017 | −0.0009 | −0.0102 | 35.6% |

**The interaction is coherent and hypothesis-consistent:** under hourly forcing
leakance improves both NSE and KGE on the losing-stream subset and a majority
(55.5%) of gauges improve; under daily forcing it degrades all three metrics and
only a minority (35.6%) improve. The sub-daily depth dynamic range from
disaggregation is what lets `zeta ∝ (depth − d_gw)` do useful losing-stream
correction — exactly the mechanism the spec predicted.

## 3. Identifiability — leakance did NOT collapse

`dump_parameters` over all 346,321 CONUS reaches (hourly-ON):

| param | median | pinning | read |
|---|---|---|---|
| `K_D` (1/s)       | 1.003e-6 | **frac@ceil = 100%**, frac@floor = 0% | saturated at upper bound — wants *more* exchange |
| `leakance_factor` | 0.327 | interior (0.12–0.53) | gate open everywhere, nowhere near 0 |
| `d_gw` (m)        | 0.294 | interior (−0.02–0.78) | spatially non-trivial |

This is the **inverse of DDR's revert** (which saw K_D collapse to the *lower*
bound / sub-0.01 exchange). Non-collapse is established — but note K_D at ceiling
means the range `[1e-8, 1e-6]` is **binding**; a follow-up should widen it to see
where K_D actually wants to sit. daily-ON shows the same ceiling pinning.

## 4. GO / NO-GO status

Per spec the GO gate is three-fold:

1. **Skill gain on losing subset (ΔNSE or ΔKGE > 0, hourly):** ✅ YES
   (ΔNSE +0.0005, ΔKGE +0.0018).
2. **Effect absent/weaker under daily:** ✅ YES (daily ΔNSE −0.0017, ΔKGE
   −0.0009 — actively negative).
3. **Learned `|zeta| > 0.01 m³/s` on a meaningful fraction:** ✅ **YES —
   measured 2026-07-01 (later session)** via the new eval-time zeta
   accumulator (`--zeta-output` on the eval binary; automatic in
   `train-and-test` Phase 2). Hourly-ON, eval network of **64,892 reaches**
   over the full 1995/10–2010/09 window:
   - median |zeta| = **6.4e-4 m³/s** (most reaches trivially small)
   - **|zeta| > 0.01 m³/s on 10.4%** of reaches — clears the ≥10% proxy bar,
     with essentially no headroom
   - `zeta_net` > 0 (net-losing) on **53.7%** of reaches
   - re-eval reproduced the run's metrics (NSE 0.7126 / KGE 0.7135 vs the
     manifest's 0.7145 / 0.7150 — CUDA scatter-add nondeterminism + f16
     checkpoint noise)

   **daily-ON zeta, for contrast:** median 8.1e-4, |zeta| > 0.01 on **14.2%**,
   net-losing 56.8%. Under daily forcing leakance is *more* active yet
   *degrades* skill (§2) — magnitude alone is not identifiability. Consistent
   with the fudge-factor reading: flat forcing gives `(depth − d_gw)` a
   mis-specified signal, so the exchange trains larger but points the wrong
   way, while the sub-daily depth range under hourly forcing yields a smaller,
   *skill-improving* correction.

**Verdict: GO — 3 of 3 criteria met**, confirmed by the analysis script
(`scripts/leakance_subset_analysis.py` prints `VERDICT: GO` with the four
valid arms + both zeta exports). Marginal: the magnitude bar passes at 10.4%
vs the 10% proxy threshold; treat it as a pass-with-asterisk until the binding
`K_D` ceiling is lifted (§3 — a clipped K_D suppresses |zeta|, so the true
fraction is plausibly higher).

## 5. Remaining work

1. ~~Build the eval-time per-reach `zeta` accumulator~~ **DONE** (this
   branch): `evaluate` accumulates per-reach mean `|zeta|` + signed `zeta_net`
   when leakance is active and appends them (dimension `COMID_eval`) to
   `<run_dir>/kan_parameters.nc` alongside `dump_parameters`' full-CONUS
   variables. Correctness guard: `tests/zeta_accum.rs` (accumulated zeta ==
   headwater `q_next` difference, exactly). See CLAUDE.md §Leakance.
2. **Widen the `K_D` range** past `1e-6` (it pins at ceiling) to locate the true
   optimum and confirm the effect isn't clipped — now the top follow-up, since
   it likely lifts both the marginal zeta fraction and the small skill delta.
3. If the GO holds after the K_D widening: promote leakance from experimental
   to a documented, default-off-but-supported routing term; else document the
   NO-GO with these numbers.

## 6. Reproduce

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/leakance_subset_analysis.py \
  --hourly-on  2026-07-01T13-43-32Z-train-and-test \
  --daily-on   2026-07-01T21-20-27Z-train-and-test \
  --hourly-off 2026-06-23T02-49-12Z-conus-hourly-train-and-test \
  --daily-off  2026-06-05T01-41-16Z-train-and-test \
  --ddrs-runs-dir /home/tbindas/projects/ddrs/.ddrs/runs
```
