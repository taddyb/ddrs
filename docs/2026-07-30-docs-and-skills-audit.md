# Documentation and skills audit — findings (2026-07-30)

Branch: `worktree-docs-audit` (worktree off `master` @ `b12c3ff`).
Scope: all 18 `.claude/skills/` packages (9,371 lines), all 32 `docs/` files,
`README.md`, `.claude/references/` (12 files), and `CLAUDE.md`.

**One-line verdict:** The skill library had accumulated ~35% duplication of
CLAUDE.md, ~25% closed-campaign narrative, and ~15% verified-wrong content — including
a config key that does not exist, two documented commands that hard-error, and a
baseline number from the wrong gauge network propagated into 13 of 18 skills — so it
was consolidated to two skills (`ddrs-dev`, `ddrs-eval-plots`) rebuilt from
source-verified facts.

---

## 1. Method

Five parallel auditors, each assigned a slice, instructed to verify every concrete
claim against the actual repository rather than assess plausibility: every file path
`ls`'d, every config key checked against `src/config.rs`, every `cargo test --test X`
checked against `tests/`, every CLI flag grepped in `src/cli/` and `src/bin/`, every
`#[test]` counted, every numeric claim cross-checked against the dated findings doc
that owns it. The highest-stakes findings were then re-verified directly.

Where docs disagreed with each other, one auditor recomputed metrics from
`.ddrs/runs/*/baseline/manifest.json` and `eval/predictions.zarr` rather than picking
a side.

---

## 2. Findings

### 2.1 A config key that does not exist is documented in three places

`kan_head.disaggregation.use_precip` is referenced by **CLAUDE.md** (twice),
`src/config.rs:113`'s doc comment, and `config/sources/conus-hourly.yaml:5`.

`DisaggregationSection` (`src/config.rs:249-288`) has exactly eight fields:
`hidden_size`, `num_hidden_layers`, `grid`, `k`, `boundary_blend`, `chunk_days`,
`pretrained_checkpoint`, `freeze`. No `use_precip`, no `use_attributes`, no
`use_temp`. They were removed in `334f0fe` ("rework disaggregation head to KAN +
basin-normalized precip").

The real contract: presence of the `disaggregation:` block ⇒ the head **always**
consumes precip ⇒ `data_sources.aorc_precip` is mandatory or `MeritGagesDataset::open`
errors. The severity is that CLAUDE.md is in every agent's context on every turn, so
this phantom key was being propagated faster than any single doc could be corrected.

### 2.2 Two documented entry-point commands hard-error

| Command | Reality |
|---|---|
| `ddrs init` | Stub. `src/bin/ddrs.rs:167` prints "merged into `ddrs plan`"; `tests/cli_init_stub.rs` asserts exit code **2**. Documented as the first step in `docs/setup.md` and `docs/usage/running.md` |
| `ddrs run --workflow eval` | `src/cli/run.rs:322` returns `"standalone --workflow eval needs a --from-run <run-id> flag"`. **`--from-run` does not exist anywhere in `src/`.** Documented in README.md, `docs/usage/running.md`, and CLAUDE.md as "equivalent to legacy `eval`" |

### 2.3 The baseline number was from the wrong network

`0.689 NSE / 0.723 KGE` appears in 13 of 18 skills as *the* bar CONUS results must
clear. It traces to `6_19_26_journal.md` (at the **repo root**, not `docs/`, which
four skills also got wrong), whose own header says it is about "the **global MERIT
dataset** … the 5,224 gauges".

The CONUS bar is **0.6781 / 0.7172 on 2,365 gauges**. Consequences:

- The widely-quoted `+0.026` NSE improvement is a cross-population subtraction. The
  correct delta against the CONUS baseline is **+0.037**.
- `ddrs-run-and-operate` and `ddrs-validation-and-qa` each paired the *global*
  baseline with a *CONUS* trained result and then quoted `+0.037` — three mutually
  inconsistent numbers in one paragraph.
- `ddrs-validation-and-qa`'s acceptance threshold "median NSE on 2365 CONUS gauges >
  0.689" was simply the wrong gate.
- `ddrs-research-frontier` contained **both** numbers, contradicting itself within ten
  lines.

**A second, distinct population error:** the "own baseline" columns in
`docs/2026-07-07-lstm-equifinality-findings.md` and `docs/2026-07-16-*` take the
3,211-gauge baseline median (including 513 single-divide gauges that scored phantom
zeros before the 2026-07-28 fix) and compare it against 2,365-gauge trained medians.
Recomputed population-matched, the hourly-lstm **"+0.022 NSE gain from routing"
reverses to −0.051** — a verdict-level sign flip. KGE is unaffected, because the
phantom gauges have zero variance and were dropped from KGE medians.

### 2.4 One superseding document invalidated the same paragraph in six skills

`docs/2026-07-06-leakance-nogo-scientific-summary.md` §5 explicitly retired the
"Phase B is required before any identifiability claim" framing: *"Phase B was built
and run; the identifiability control still failed on the clean objective.
Identifiability is REFUTED, not pending."*

All six research skills still asserted Phase B as a live precondition. Three of them
also still contained executable instructions for running it. `ddrs-identifiability-campaign`
had been patched with a "CLOSED" banner at line 53 while lines 45-54 and 246-361 still
described the campaign as blocked and pending — self-contradictory within one file.

This is the structural lesson: the duplication is *why* the staleness was uniform.
One doc invalidated one paragraph, that paragraph existed in six places, and none
were updated.

### 2.5 Test counts were wrong and are inherently brittle

| Claimed | Actual |
|---|---|
| `leakance_gradcheck` 8/8 | **16** |
| `zeta_accum` 6/6 | **8** |
| `leakance_off_parity` 3/3 | 3 ✓ |

The "8" almost certainly came from `Backward<I,8>` — the number of saved tensors, a
different quantity. Both counts had grown with the Phase C `losing_only` and
impervious-mask paths. The consolidated skill asserts "all pass" instead.

Two gate commands in CLAUDE.md and `docs/` **run zero tests and exit 0**:
`cargo test --test mmc mc_routes_linear_chain` (no such test exists) and
`cargo test --test sp8_gradcheck -- --ignored` (nothing in that file is `#[ignore]`) —
a gate that cannot fail.

### 2.6 The mdBook is a 2026-06-08 snapshot

Everything landed since is invisible to it: leakance, the precip disaggregation head
(`src/nn/disagg_head.rs`, 869 lines, in **no** chapter), the `nnse-kge`/`kge`
objectives, `ddrs sources`/`import`/`status`/`gc`, managed adjacency in the reader
chapters, the global data stores, and `--backend cpu`.

Additional verified errors: `docs/algorithm.md` says "MLP head" four times and
`docs/intro.md` three times, violating invariant 5 while `docs/architecture.md:71`
correctly says KAN; `docs/algorithm.md` calls `p_spatial` an exponent (it is the
coefficient; `q_spatial` is the exponent); `docs/usage/outputs.md` opens with "there
is no global results directory" when `src/cli/run.rs:98` puts everything under
`run_dir`; `docs/setup.md` claims the CPU path needs no CUDA when `burn-cuda` and
`cudarc` are non-optional dependencies, so a CPU-only reader fails at `cargo build`,
not at runtime; `docs/reference/perf.md` states the Rust defaults for
`sparse_solver`/`use_cuda_graphs` inverted (they are `Cpu`/`false`; merit YAML *sets*
cuda/true); `docs/reference/baseline.md`'s cache-key formula omits the
`BASELINE_ALGO_VERSION` salt that is hashed first, and its gauge filter predates the
headwater skip.

`docs/reference/burn-autograd.md` and `docs/nh-qprime-store-contract.md` are the two
cleanest files audited — the latter had zero factual errors.

### 2.7 `.claude/references/` is a stale strict subset of `docs/`

Provenance from git: `docs/` was generated from the skills on 2026-05-29 (`59f7bcb`),
**corrected against source** on 2026-06-08 (`e6dc3df` — "KAN head (not MLP),
directory-based checkpoints…"), and on 2026-06-10 (`f724913`) the skills were renamed
to `.claude/references/` with those same corrections back-ported. So the references'
newer mtime reflects a catch-up commit, not newer knowledge.

Content check across all 12 pairs found **zero substantive lines present in a
reference and absent from its `docs/` counterpart**. Coverage discriminator (grep
counts, refs vs docs): `ddrs sources` 0 vs 1, `ddrs import` 0 vs 4,
`geospatial_fabric` 2 vs 13, `ddrs run` 2 vs 18. At least one inbound pointer was
already broken — a skill sent readers to `ddrs-reading-outputs.md` for the "zeta
netcdf schema" in a file containing zero occurrences of "zeta".

---

## 3. What was done

**Skills: 18 → 2.** 9,371 lines → 2,359, a 75% reduction, with the load-bearing
content preserved and corrected.

| Skill | Content |
|---|---|
| `ddrs-dev` | Building, coding, configuring, testing, running, debugging. `SKILL.md` + `references/{build-and-env, config, testing, traps, research-status}.md` |
| `ddrs-eval-plots` | Evaluating and visualizing output. `SKILL.md` + `references/{hydrograph, metrics, parameter_map, parity}.md` + `scripts/` |

Deleted: `ddrs-architecture-contract`, `ddrs-build-and-env`, `ddrs-change-control`,
`ddrs-config-and-flags`, `ddrs-debugging-playbook`, `ddrs-diagnostics-and-tooling`,
`ddrs-docs-and-writing`, `ddrs-external-positioning`, `ddrs-failure-archaeology`,
`ddrs-hydrology-reference`, `ddrs-identifiability-campaign`,
`ddrs-proof-and-analysis-toolkit`, `ddrs-research-frontier`,
`ddrs-research-methodology`, `ddrs-run-and-operate`, `ddrs-validation-and-qa`,
`regenerate-docs.md`.

Within `ddrs-eval-plots`: `parameter_swap.md` and `loss_landscape_h6.md` deleted as
dead campaigns (both plot instruments the v2 audit refuted); `kan_interpretability.md`
deleted as mechanism-inspection rather than output-evaluation; the two orphaned
`parity_*.md` files (referenced from neither the routing table nor the file list) were
merged into `parity.md` and wired in.

**Editorial principles applied:**
1. **Do not duplicate CLAUDE.md.** It is in context on every turn; restating it is
   pure waste. The consolidated skills say so explicitly and carry only what CLAUDE.md
   lacks.
2. **Lead with the discriminating test, not the narrative.** A future session needs
   "checkpoints must be directories; flat `.mpk` means stale binary", not the story.
   The one incident retained in full is the 2026-07-01 stale binary, because the
   forensic detail is what makes the rule stick.
3. **One authoritative numbers table**, with gauge-set definitions first and an
   explicit do-not-use list, replacing five copies of H1–H7, four of P1–P3, and four
   of R1–R5.
4. **Assert "all pass", never a test count.**

**Verification:** every template in the rewritten `ddrs-eval-plots` references was
executed end-to-end against a real run, producing PNGs and reproducing the committed
`parameter_convergence_stats.json` values exactly.

---

## 4. Not done — recommended follow-ups, most-wrong first

Fixing `docs/` and `README.md` was out of scope for this change; this audit produced
the list, not the fix.

**P0 — commands that fail**
1. `ddrs init` → `ddrs plan` in `docs/setup.md`, `docs/usage/running.md`.
2. `ddrs run --workflow eval` → document `train-and-test` as the only path, in
   README.md, `docs/usage/running.md`, **and CLAUDE.md**.
3. Drop `-- --ignored` from `docs/algorithm.md`'s sp8_gradcheck gate.
4. Remove `mc_routes_linear_chain` from CLAUDE.md and `docs/usage/graph-objects.md`.
5. Fix `cargo test --lib data::store::zarr::tests::…` → `tests/data_zarr_store.rs`.
6. Fix `cargo test --lib training::checkpoint` → `tests/checkpoint_resume.rs`.

**P1 — actively misleading**
7. Rewrite `docs/usage/outputs.md` around `.ddrs/runs/<id>/`.
8. Reframe the CPU path as "CPU *execution*; a CUDA toolkit is still required to
   compile".
9. Fix the seven MLP mentions in `docs/algorithm.md` and `docs/intro.md`.
10. Add the headwater skip and the algorithm-version salt to
    `docs/reference/baseline.md` **and CLAUDE.md**.
11. `DDRS_FORCE_GRAPHS` selects the CUDA backend but does **not** enable graph
    capture — fix `docs/reference/ddr-comparison.md` and the dependent V9 claim in
    `perf.md`.
12. Correct the inverted defaults in `perf.md`; correct `tau` in
    `docs/usage/inputs-formatting.md`.

**P2 — CLAUDE.md corrections** (highest leverage; it is always in context)
- `use_precip` phantom key (also `src/config.rs:113`, `config/sources/conus-hourly.yaml:5`)
- `--workflow eval` "equivalent to legacy eval"
- baseline cache key missing `BASELINE_ALGO_VERSION`
- `mc_routes_linear_chain`
- data-source table still marks `netcdf`/`icechunk` **(TODO)**; both are implemented
- `src/sparse.rs` → `src/sparse/`
- `LossKind::kge` unmentioned
- `conus-hourly` described as "`conus` + aorc_precip"; they target different machines

**P3 — structural**
- Add `docs/nh-qprime-store-contract.md` to `docs/SUMMARY.md` (cited by README and
  CLAUDE.md but not published).
- Delete `.claude/references/` and repoint its ~25 inbound links (CLAUDE.md ×4,
  `src/sparse/mod.rs:11`) at `docs/`. Cheaper interim: replace each file with a
  two-line stub.
- Decide whether the 22 dated findings docs should live under `docs/` — with
  `src = "docs"`, mdBook copies them to the published output as static assets.
- Add correction notes to `docs/2026-07-16-wave2-cross-wave-findings.md` and
  `docs/2026-07-07-lstm-equifinality-findings.md` for the population-mismatched
  baseline columns (§2.3).
- Write a findings doc for the undocumented 2026-07-29/30/31 runs. One of them
  (`2026-07-30T00-24-24Z`, median NSE 0.6799 / KGE 0.7194 on 2,365 gauges vs its own
  baseline 0.6744 / 0.7082) beats its baseline on **both** metrics, which qualifies
  the most-repeated claim in the retired skills — though on a different Q′ store and
  at an absolute NSE well below the 0.7152 benchmark.

---

## 5. Reproduce

```bash
git worktree add .claude/worktrees/docs-audit -b worktree-docs-audit master
cd .claude/worktrees/docs-audit

# phantom config key
grep -rn use_precip src/ CLAUDE.md config/sources/
sed -n '249,290p' src/config.rs

# dead commands
sed -n '165,172p' src/bin/ddrs.rs
sed -n '320,328p' src/cli/run.rs

# test counts
for f in leakance_gradcheck zeta_accum leakance_off_parity; do
  printf '%s: ' "$f"; grep -c '#\[test\]' "tests/$f.rs"; done

# baseline population
python3 -c "import json;m=json.load(open('.ddrs/runs/2026-07-30T00-24-24Z-train-and-test/manifest.json'));print(m['metrics'])"
```
