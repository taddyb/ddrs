<!-- agent a92e512a7c47c97b4 | Implement learned dam release in ddrs | handback | 2026-09-27T10:36:52.008Z -->
## Write-up done and pushed; checks 4 and 5 still pass with the fixed dam row

**Branch:** `dam-release-head`, last commit `61a4a50`, pushed. No `src/` changes since `0ac6f2e`; only one test-file doc comment was reworded. The seed-43 replicates (`10-31-30Z`, `10-31-50Z`) were not touched or waited for.

### Checks 4 and 5 after the fix
Smoke re-run `2026-09-27T07-30-13Z-train-and-test` (main workspace, CPU, wall 3,577 s: train 2,496 s, test 1,075 s), paired against the off arm `2026-09-27T04-29-33Z` with the same script as before (`smoke_checks_4_5.py`).

**Check 5: pass on all three criteria.**

| Measure (test WY1996-2010) | First build | After the fix |
|---|---|---|
| Dam gauges ΔNSE | +0.0049 | **+0.0056 [+0.0021, +0.0095]**, 280 up / 178 down, sign p 2e-6 |
| On-reach dams | | +0.0063 |
| Off-reach dams | | +0.0038 |
| Controls ΔNSE | −0.0001 | **−0.0011 [−0.0018, −0.0002]**, 199 / 259 |
| Dam minus matched control | +0.0058 | **+0.0072 [+0.0044, +0.0108]**, 295 / 163 (offline ceiling +0.0085, so about 85 %) |
| Control median NSE | | 0.759 → 0.761, within 0.01 |
| Dam gauges ΔKGE | | −0.0014 [−0.0030, −0.0002] |

**Check 4: passes in direction.**
- Release-head `T0` went from 0.19 d at init to a median of 0.59 d over the 1,024 dams (IQR 0.34 to 1.21, max 9.7).
- Where the offline fit found storage: 0.87 d (251 dams). Where it found none: 0.44 d (88 dams). Spearman with the fitted `T0` is 0.40.
- The head now learns seasonality: median amplitude √(a²+b²) is 0.31 (max 0.74), against under 0.1 before the fix.
- "Stays low where the fit found none" is still only partial: 0.44 d against the 0.19 d init.

The findings doc §6 also has check 2 after the fix (median 0.998, 160 of 202 dams above 0.99), the mass balance, and check 1b under the adopted criterion.

### What else was written
- **Findings doc, §7 (full population):** summarises your `paired_full_run.json` (cited, with the branch). My `full_population_pairing.py` reproduces its paired numbers exactly.
  - Arms: off `07-29-47Z` (8,601 s wall) and learned `07-29-55Z` (9,003 s).
  - Median NSE 0.7391 off vs 0.7378 learned; summed Q' 0.6785.
  - Paired ΔNSE: dammed +0.0014 [+0.0007, +0.0020] (532 / 385); undammed −0.0007 [−0.0009, −0.0004] (573 / 875).
  - Learned release from `release_params.csv`: `T0` median 0.35 d (IQR 0.19 to 0.84, max 17.4; 219 dams above 1 d, 13 above 10 d); amplitude median 0.13 (max 0.56).
  - It notes the seed-43 replicates are running.
- **Findings doc, §8 (Follow-ups):** the four review items.
  - (a) is fixed in the docs: the findings doc, `RESERVOIRS.md`, `testing.md` and the test's module docs now say the bitwise match is NdArray-only.
  - (b) and (c) are documented in `config.md` §Reservoirs.
  - (d) is recorded. The release arm's full-population training ran 6 % slower than the off arm.
- **Journal:** the three runs are filled in this branch's `research/journal/2026-09.md`, consistent with your reservoir-options entries. Status shows 0 open TODOs.
- **Other docs:** `RESERVOIRS.md` and `research-status.md` carry the headline full-population numbers. `verify_doc_paths` passes; the one citation to your branch's file has an ignore marker.
- **Result tables** copied into `experiments/reservoir/release_head/results/`.

### Things that looked off
- **Control and undammed losses grew with the fix.** The smoke controls lost −0.0001 NSE with the first build and −0.0011 with the fixed row; the full population's undammed gauges lose −0.0007. Their networks have no dams, so this can only come through the co-trained routing head. The seed-43 replicates are the test.
- **Two baseline numbers for the same gauges.** The run's cached `baseline/manifest.json` covers 2,698 gauges; my first pairing read its median (0.6607) by mistake, and the script now filters to the run's own gauges. On those 2,365 the cached baseline gives 0.6771 NSE, against your 0.6785 recomputed from the prediction days. I cite 0.6785. The gap is presumably day alignment, but I have not checked.
- **Manifests say "dirty".** That is doc edits made while the runs were going; `src/` equals `0ac6f2e`.
- **Harmless:** some of my wait scripts exited 1 because of a `tail` syntax error. The runs themselves all exited 0.

Findings: `/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/research/findings/2026-09-27-learned-dam-release-findings.md`
