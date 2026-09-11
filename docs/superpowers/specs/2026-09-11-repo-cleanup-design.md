# Repository cleanup: stale context, split docs, archived campaigns (design, 2026-09-11)

Branch: `worktree-repo-cleanup` (worktree off `origin/master` @ `3412a78`,
rebased from `571e2f6` after PR #42 merged).
Scope: `CLAUDE.md`, `.claude/` (skills, references, specs), the mdBook pages and
`README.md`, `scripts/`, `examples/`, and the eight files tracked inside the
gitignored `output/`.

**One-line goal:** make every claim in always-loaded agent context verifiable
against source, leave only the book under the book's source tree, and leave only
live tooling in `scripts/` and `examples/`.

This design is the successor to `docs/2026-07-30-docs-and-skills-audit.md`, whose
§4 recorded a P0-P3 follow-up list that was never worked. Part of that list is now
fixed; part is still broken; and six weeks of new work (gridded routing, the
landscape study, `nse-batch`, `ddrs experiment`, the research journal) added a new
layer of drift on top. This design covers both layers.

---

## 1. Success criteria

Three criteria, each mechanically checkable rather than judged:

1. **Always-loaded context is true.** Every path, test name, config key, CLI flag
   and source citation in `CLAUDE.md` and `.claude/skills/**` resolves against the
   tree it ships with. Enforced by a verifier script, run as a gate.
2. **The book source tree contains only the book.** `mdbook build` output contains
   no research narrative. Published URLs are unchanged, proven by diffing the
   built file list before and after.
3. **`scripts/` and `examples/` contain only live tooling.** Every remaining entry
   is reachable from a current workstream, an invariant gate, or a documented
   command. Everything else is in `research/archive/` with its closing findings doc
   named.

---

## 2. Findings that motivate the work

All entries below were verified against source in this worktree, not inherited
from the 2026-07-30 audit. Where that audit made a claim, it was re-checked; several
of its P0/P1 items have since been fixed and are excluded here.

### 2.1 `CLAUDE.md` (663 lines, loaded on every turn)

| Line | Claim | Verified reality |
|---|---|---|
| 294 | `src/sparse.rs` in the architecture diagram | `src/sparse/` is a directory (`mod.rs`, `cusparse.rs`, `dispatch.rs`). Invariants 1 and 4 say it correctly; only the diagram regressed. The 2026-07-30 audit §4 already recorded this exact fix as needed |
| 309 | `spike_backward/` exists, "ignore" | `ls spike_backward` fails. The directory is gone |
| 409-412 | `nse-batch`, `optimizer`, `use_grad_accum`, `grad_accum_steps` "land with the gradient-accumulation work (PR #31, branch `exp_train`)"; on a commit without them `nse-batch` fails config load | All four are on master. `LossKind::NseBatch` parses (`src/config.rs`, test `nse_batch_loss_kind_parses`); `OptimizerKind`, `use_grad_accum`, `grad_accum_steps` all validated at load with their own passing tests |
| 649 | "Three skills cover this repo" | Four. `ddrs-run` landed 2026-09-09 (`16989ea`) and is named nowhere in "When in doubt", despite being the runbook for "train a model" and "resume from epoch N" |
| 153 | `ddrs init` stub at `src/bin/ddrs.rs:167` | Line 167 is inside the `SourcesCmd` enum. The `Cmd::Init` arm is at 193-196. Exit code 2 is correct |
| 538 | `validate_subdivision_reaches_the_builder`, `src/config.rs:1066` | The function is at 1202. Line 1066 is an unrelated `geodataset:` validator's doc comment |
| 151 | `src/cli/run.rs:322` for the `--workflow eval` error | The string is at 324 |
| 163 | `conus`, `conus-hourly`, `global`, `daily-lstm`, `hourly-lstm` ship in-repo | Nine files in `config/sources/`. `conus-gridded` is mentioned elsewhere (line 343) but not here; `aorc_dhbv_distributed`, `conus-experimental`, `conus-hydrodl2` are unmentioned anywhere |
| 156 | "First study: `adjoint`" | `landscape` also shipped, dispatched alongside `adjoint` in `src/cli/experiment.rs`, with 30+ committed bundles under `experiments/landscape-*`. It is the user's current workstream |

**Omitted but load-bearing** (verified to exist, absent from `CLAUDE.md`):

- `vendor/README.md` explains why `[patch.crates-io]` points at the `taddyb/cubecl`
  (`ddrs-release`) and `taddyb/burn` (`ddrs-sp7-primitive-ctor`) forks: two `pub`
  accessors needed for the SP-7 cuSPARSE work. Anyone hitting a patch or version
  error during `cargo build` needs this, and `CLAUDE.md` never says `vendor/` exists.
- `ddrs-py/`, a maturin/PyO3 crate for read-only CPU inference from notebooks.
- `src/experiment/landscape/`, the second paper study.

**Section line budget**, largest first: Commands 231, Leakance 84, Data sources 82,
Reach subdivision 54, Critical invariants 40, Baseline 38, Research journal 31,
Architecture 29, Training objective 29, When in doubt 17, Conventions 14, What this
project is 10.

### 2.2 The four skills

| file:line | Claim | Verified reality |
|---|---|---|
| `ddrs-dev/SKILL.md:262` | the mdBook "is a strict superset of the **deleted** `.claude/references/` copies" | `.claude/references/` is not deleted. Twelve files, 4.5 KB to 14.5 KB each |
| `ddrs-dev/references/build-and-env.md:16` | CPU-only compilation is unsupported, "`docs/setup.md` claims otherwise — it is wrong" | `docs/setup.md:74-85` says the identical thing. The skill is correcting a doc that already agrees with it |
| `ddrs-dev/references/config.md:180` | "Four validators run at `Config::from_yaml_file`, plus one at dataset open" | Nine validators run directly, plus a nested `validate_subdivision_reaches_the_builder`, plus one at dataset open. The guards table below the prose also omits `validate_enforce_positivity` |
| `ddrs-run/SKILL.md:245` | `ddrs experiment <name>` is "**Not on master.** The subcommand and `src/experiment/` live on the `experiment-adjoint` branch" | `src/experiment/{adjoint,landscape}`, `src/cli/experiment.rs` and the `Experiment` subcommand are all present. PR #39 merged as `05f82f9`. `ddrs-dev/SKILL.md:161-184` documents it as working, so the library contradicts itself |
| `ddrs-dev/references/traps.md:200,243` | Two distinct traps both headed `## T11` | Numbering collision. The symptom index (lines 7-22) has no row for the autodiff-tape-leak trap, making it unreachable from the index |
| `ddrs-dev/references/gauge-population.md:52-53` | `is_headwater` at `src/data/store/zarr.rs:128`, "zarr group present AND `order` length > 1" | The function is at 208 and its body is `self.indices_0.is_empty()`. It tests edge count, not `order` length |
| `ddrs-dev/references/config.md:123-126` | `use_precip` is "still referenced" by `CLAUDE.md`, `src/config.rs:113` and `config/sources/conus-hourly.yaml:5` | All three were fixed. `CLAUDE.md` now states "There is no `use_precip` key"; the YAML no longer mentions it; `src/config.rs:113` is an unrelated `attributes` doc comment |

**Hard-coded counts that have drifted**, and that the library's own editorial rule
forbids:

- `testing.md:3` claims "70 test files, 233 `#[test]` fns". Actual: 88 files, 363
  `#[test]` in `tests/*.rs`. The rule against hard-coded counts is stated two lines
  below this line.
- `ddrs-dev/SKILL.md:38-77` gives per-reference line counts of 80 / 194 / 127 / 212 /
  213 / 100. Actual: 97 / 227 / 194 / 262 / **782** / 90. `research-status.md` grew
  3.7x during the landscape campaign without the index noticing.
- `ddrs-eval-plots` declares 211 and 455 for `metrics.md` and `parameter_map.md`;
  actual 221 and 473.

**Duplication of `CLAUDE.md`**, against the library's stated rule that it must not
duplicate always-loaded context:

- `config.md:73-92` restates `CLAUDE.md:402-412` near-verbatim, including the same
  wrong `exp_train` sentence. The duplication is the mechanism by which one stale
  paragraph became two.
- `ddrs-dev/SKILL.md:81-85` condenses the `CLAUDE.md:111-131` stale-binary callout,
  17 lines after declaring that it does not repeat `CLAUDE.md`. The same fact appears
  a third time in `ddrs-run/SKILL.md:21-44` (with a newer 2026-09-09 example, which
  is defensible) and a fourth in `traps.md` T1.

**Audited and sound**, so out of scope: `ddrs-journal/SKILL.md` (hooks,
`scripts/journal.py`, `docs/journal/*.md` all verified); `ddrs-run/SKILL.md` apart
from line 245; `traps.md` exit-code table against `src/cli/types.rs`;
`ddrs-eval-plots/references/channel_geometry.md` line citations, verified byte-exact;
all 23 dated doc paths cited by `research-status.md`.

### 2.3 Book pages and README

Still broken from the 2026-07-30 P1 list, both of which broke *after* that audit:

| file:line | Claim | Verified reality |
|---|---|---|
| `docs/reference/perf.md:48,59`, `docs/usage/inputs-formatting.md:476` | `config/merit_training.yaml` ships `use_cuda_graphs: true` | It is `false` (`config/merit_training.yaml:143`), flipped 2026-08-19 when the captured kernel became incompatible with the corrected physics that is now default. `docs/reference/ddr-comparison.md:155-164` documents the same event correctly; the fix never propagated |
| `docs/usage/inputs-formatting.md:474` | `tau` default is 3 | `src/config.rs:766` is `p.tau = r.tau.unwrap_or(9)` since 2026-08-08 (`54cd386`), and 3 is on the retired, wrong-direction scale. Test `cfg.params.tau == 9` |

New drift since 2026-07-30:

- `README.md:211-213` frames `nse-batch` and `optimizer: adadelta` as "once PR #31
  lands". Merged 2026-07-30 as `24cb4d0`.
- `docs/reference/baseline.md:147,386` still describe an `init -> plan -> run`
  lifecycle. The P0 fix landed in `setup.md` and `running.md` and stopped there.
- `docs/architecture.md:147-155`'s `src/adjacency/` file table omits `gridded.rs`
  and `subdivide.rs`.
- `docs/usage/running.md` documents `plan`/`run`/`show`/`status`/`gc`/`sources`/
  `import` exhaustively and never mentions `ddrs experiment`.
- `docs/usage/inputs-formatting.md`'s `params:` table omits `ddr_match`,
  `enforce_positivity` and `subdivision`. The `ddr_match` omission is the root cause
  of the `use_cuda_graphs` error above: the table has no concept of a default that
  depends on physics mode.
- `use_grad_accum` and `grad_accum_steps` appear in no book page.
- `docs/reference/perf.md:47-48` cites `src/config.rs:407-410` and `:454`; actual
  499-503 and 677.

Verified clean, so out of scope: `intro.md`, `setup.md`, `algorithm.md`,
`usage/inputs-reading.md`, `usage/graph-objects.md`, `usage/outputs.md`,
`reference/ddr-comparison.md`, `reference/burn-autograd.md`,
`nh-qprime-store-contract.md`. Every relative link in all 15 pages resolves; there
are no dead links.

### 2.4 Cross-workstream artifacts

- `.claude/2026-05-29-ddrs-docs-{design,plan}.md` are md5-identical to the copies in
  `.claude/specs/`.
- `.claude/references/` (12 files) duplicates `docs/`. Every file is between a
  quarter and a half the length of its counterpart. `src/sparse/mod.rs:11` and
  `docs/intro.md`'s mapping table point into it.
- `.claude/specs/` holds the sp1-sp10 design and plan pairs dated 2026-05-17 to
  2026-05-29. `docs/superpowers/{specs,plans}` holds the same series from 2026-05-30
  on. One series, split by date across two directories.
- `output/` is gitignored, yet eight files under `output/synthetic_n/plots/` are
  tracked, cited by `docs/2026-07-22-synthetic-n-recoverability-findings.md:36`
  from the campaign that stopped on 2026-07-29.
- `docs/images/` holds only `.gitkeep` and is referenced nowhere.
- `scripts/` holds roughly 21 one-off campaign scripts (leakance, zeta probe,
  equifinality, synthetic-n, tau sweep, the sp8/sp10 spike checks) whose only
  inbound reference is the findings doc that closed the campaign. Four have no
  inbound reference at all: `generate_run_notebooks.py`, `hydrograph_comparison.py`,
  `merge_merit_basins.py`, `parameter_landscape.py`.
- `examples/` holds roughly 13 probes from the disaggregation and leakance
  campaigns. `Cargo.toml` declares no `[[example]]` entries and no test depends on
  any of them, so removing them cannot break a gate.

### 2.5 The mdBook publishes research narrative

`book.toml` sets `src = "docs"` and `.github/workflows/docs.yml` deploys to GitHub
Pages from `master`. mdBook renders only files reachable from `SUMMARY.md` and copies
everything else in the source tree verbatim, so 36 findings docs (592 KB),
`docs/superpowers/` (1.5 MB) and `docs/figures/` (7.2 MB) ship as unlinked assets
beside a 14-page book. The repository is public, so this is not new exposure; it is
build-output clutter and a `docs/` root that reads as a junk drawer.

---

## 3. Sequencing: one pull request, two phases

An earlier revision of this design split the work across two pull requests, because
`origin/landscape-deriv-objective` was in flight with ten commits editing
`docs/2026-09-08-landscape-hypothesis-tests-findings.md`, `docs/journal/2026-09.md`
and `.claude/skills/ddrs-dev/references/config.md`, all of which the restructure
renames. A rename of about 100 files plus 414 link rewrites would have collided
with it.

**That branch merged as PR #42, so the split is unnecessary** and the work is one
pull request. `origin/skill/ddrs-eval-plots-global-runs` (2026-06-19) is the only
remaining unmerged branch and it is three months stale.

The phase boundary is kept as an ordering constraint rather than a PR boundary:

**Phase 1 changes content only and renames nothing.** Doing it first means the
citation verifier is in place, and every claim is already true, before any path moves.

**Phase 2 does the restructure**, so the link rewrite operates on corrected text
rather than racing it.

Re-verified against `origin/master` @ `3412a78` after the merge: `CLAUDE.md` is
unchanged at 663 lines with every cited line number intact, the verifier reports the
same nine strict failures, and the reference count is still 414. Of the nineteen new
commits, only `ddrs-dev/references/config.md` and four new `examples/juniata/*.yaml`
case configs fall inside this design's scope, and neither changes a finding.

---

## 4. Phase 1: truth pass

### 4.1 `CLAUDE.md`, 663 lines to roughly 480

Correct every row of §2.1. Compress the two closed workstreams to status blocks:

```
## Leakance (CLOSED, NOT PROMOTABLE, 2026-07-06)

A losing-stream term subtracted from the routing RHS `b`. Code-complete and
gradient-exact; `params.use_leakance` defaults false. Do NOT remove it, and do
NOT re-open the question: a gauge measures the SUM of zeta over its upstream
network, and that sum does not determine the per-reach distribution. Every rival
explanation (gradient starvation, objective noise, uninformative inputs, sign
ambiguity) was individually refuted.
Verdict and refutations: docs/2026-07-06-leakance-nogo-scientific-summary.md §3
Enable steps, parameter ranges, gate commands: ddrs-dev/references/config.md
```

Reach subdivision gets the same shape: the NO-GO verdict, the reason (both
Muskingum coefficients are non-negative only inside a window 1.4 % wide at the
measured CONUS median X), the do-not-reopen line, and pointers to
`.claude/REACH-SUBDIVISION.md` and the skill. Leakance goes from 84 lines to about
10, subdivision from 54 to about 8. The enable instructions, the seven subdivision
fields, the parameter ranges and the gate command lists move into
`ddrs-dev/references/config.md` and `testing.md`.

Add short entries for `vendor/` and `ddrs-py/` to the architecture section, and add
`landscape` to the paper-studies paragraph.

### 4.2 Citation policy

Six of the findings in §2.1 and §2.2 are drifted `file:line` citations. Re-pinning
the numbers resets a clock that will run out again, so `CLAUDE.md` and the skills
adopt a rule: **inside `src/`, cite file plus symbol**
(`src/config.rs::validate_subdivision_reaches_the_builder`), never a line number.
Line citations remain only for files that do not move, such as the DDR reference
tree at `~/projects/ddr/`, and for fixture or config files where the line is the
content. The rule is stated once, in `CLAUDE.md`'s conventions section, and the
verifier in §4.6 checks symbol citations resolve.

### 4.3 The four skills

Correct every row of §2.2. Delete all hard-coded line counts from
`ddrs-dev/SKILL.md`'s contents table and the `ddrs-eval-plots` file list, and
replace `testing.md:3`'s test-count sentence with the assertion the file's own rule
prescribes. Delete `config.md:73-92`, the paragraph duplicating `CLAUDE.md`, and
replace it with a one-line pointer. Delete `config.md:123-126`'s `use_precip`
passage, which now describes a fixed problem as live. Renumber the second `T11` in
`traps.md` and add it to the symptom index. Leave `ddrs-dev/SKILL.md:81-85` in
place: it is a condensation rather than a copy, and the stale-binary trap has cost
a whole experiment before.

### 4.4 Book pages and README

Correct every item in §2.3. Two of them are table cells with wrong values
(`use_cuda_graphs`, `tau`), so the surrounding `params:` table in
`inputs-formatting.md` gets the three missing keys at the same time, plus a note
that `ddr_match` changes other defaults. Add `ddrs experiment` to
`running.md`'s CLI reference, `gridded.rs` and `subdivide.rs` to
`architecture.md`'s `src/adjacency/` table, and `use_grad_accum` /
`grad_accum_steps` to the training-configuration section.

### 4.5 `.claude/` deduplication

Delete `.claude/2026-05-29-ddrs-docs-{design,plan}.md`, verified md5-identical to
the `.claude/specs/` copies.

Delete `.claude/references/` (12 files) and repoint `src/sparse/mod.rs:11` at
`docs/reference/burn-autograd.md` and remove `docs/intro.md`'s mapping table. This
makes `ddrs-dev/SKILL.md:262` true.

**Gate before deleting:** the claim that `docs/` is a strict superset comes from
the 2026-07-30 audit and `docs/` has been edited since. Before the delete, run a
per-pair comparison across all 12 pairs and confirm no substantive claim exists only
in a reference. If one does, it moves into the `docs/` counterpart first.

### 4.6 Phase 1 gates

| Gate | Catches |
|---|---|
| `cargo check --examples --tests` | any doc-comment edit that broke a file. Baseline green, 39.85 s |
| `mdbook build` | a dangling `SUMMARY.md` entry (`create-missing = false`) |
| path verifier (new, `scripts/verify_doc_paths.py`) | every path-shaped and `file::symbol`-shaped token in `CLAUDE.md`, `.claude/skills/**` and the book pages that does not resolve |
| `grep -rn '\.claude/references'` | a missed inbound link. Must return nothing |
| `cargo test --test ddr_sandbox_match --test gridded_bundle` | the pre-push hook's own gate. Nothing in Phase 1 touches the solver, so this is a sanity check rather than a risk |

---

## 5. Phase 2: restructure

### 5.1 Split the book from the research record

`book.toml` gets `src = "docs/book"` and `edit-url-template` gains the `docs/book`
prefix. `git mv` the 14 pages plus `SUMMARY.md`, `usage/` and `reference/` under
`docs/book/`. `.github/workflows/docs.yml`'s path filter tightens from `docs/**` to
`docs/book/**` plus `book.toml`, so research edits stop triggering a docs deploy.

Published URLs do not change: mdBook output paths are relative to the source root,
so `docs/usage/running.md` and `docs/book/usage/running.md` both render to
`/usage/running.html`. **Proof obligation:** capture `find target/book -type f | sort`
before and after, and diff. The diff must show only the removal of the copied
research assets.

### 5.2 `research/`

```
research/
  findings/      36 dated *-findings.md, *-handoff.md, *-litreview.md
  specs/         docs/superpowers/specs/ + the design half of .claude/specs/
  plans/         docs/superpowers/plans/ + the plan half of .claude/specs/
  why-analysis/  7 files, unchanged
  figures/       9 PNGs + synthetic-n/ (from output/)
  journal/       docs/journal/
  archive/       scripts/ + examples/ + README.md
```

`.claude/specs/` folds in because it is the same spec-and-plan series as
`docs/superpowers/`, split only by the date the convention changed.

### 5.3 The link rewrite

414 references across about 150 files, four anchored prefixes:

| From | To |
|---|---|
| `docs/2026-` | `research/findings/2026-` |
| `docs/superpowers/specs/` | `research/specs/` |
| `docs/superpowers/plans/` | `research/plans/` |
| `docs/why-analysis/`, `docs/journal/`, `docs/figures/` | `research/…` |

Referrers inside the moving set are rewritten in bulk, because they move together
and stay self-consistent. The roughly 130 referrers outside it are read individually,
in particular about 45 `config/experiments/*.yaml` comments and about 16 `src/*.rs`
doc comments.

### 5.4 Archive the closed campaigns

`research/archive/{scripts,examples}/` with a README stating the admission rule:
**an artifact lands here when the findings doc that cites it records its campaign as
closed**, and recording, per artifact, the closing doc and the commit at which it
last built.

Archived from `scripts/`: the leakance and zeta-probe set, the equifinality set, the
synthetic-n and recoverability set, the tau-sweep set, the sp8/sp10 spike checks,
and the four with no inbound reference at all.

Archived from `examples/`: the `disagg_*`, `pretrain_disagg_*`, `kan_disagg_*` probes
plus `leak_probe.rs`, `save_random_kan.rs`, `kan_sensitivity_sweep.rs`.

Retained in `examples/`: `compare_ddr_sandbox.rs` (invariant 1),
`benchmark_hydrograph.rs`, `dump_init_params.rs`, and the `juniata/` and
`juniata_gridded/` bundles. Retained in `scripts/`: the DDR-parity dump scripts, the
KAN fixture dumps, `export_ddr_sandbox.py`, the attribute and gauge builders,
`snap_gridded_gauges.py`, the landscape shard tooling, and the journal scripts.

`docs/book/usage/running.md`'s example table drops the archived rows.

### 5.5 `output/` and `docs/images/`

`git rm --cached` the eight tracked files under gitignored `output/`, moving the
seven PNGs and the notebook to `research/figures/synthetic-n/` and repointing
`docs/2026-07-22-synthetic-n-recoverability-findings.md:36`. Delete
`docs/images/.gitkeep`, referenced nowhere.

### 5.6 What Phase 2 does not touch

`experiments/` stays as it is. Its 37 bundles total 584 KB and the `landscape-*` and
`adjoint-*` families are the active workstream, not residue. `src/`, `tests/`,
`config/` and `fixtures/` change only where a doc comment names a moved path.

### 5.7 Phase 2 gates

| Gate | Catches |
|---|---|
| `cargo check --examples --tests` | a doc-comment rewrite that broke a source file |
| `mdbook build` plus built-file-list diff | a SUMMARY entry left behind, or a URL change |
| path verifier over the whole tree | any of the 414 rewrites that produced a path which does not resolve |
| `grep -rn 'docs/20\|docs/superpowers\|docs/why-analysis\|docs/journal'` | a missed rewrite. Must return nothing outside the cleanup's own findings doc |
| `cargo test --test ddr_sandbox_match --test gridded_bundle` | pre-push hook parity |

---

## 6. Concerns

**The 414-reference rewrite is the principal risk.** An anchored prefix substitution
cannot distinguish a live pointer from a historical quotation, and some findings docs
quote paths as they stood at the time of writing. Rewriting such a quotation makes
the historical record say something that was never true. Mitigation: the verifier
proves every resulting path resolves, and the out-of-moving-set referrers are read
rather than swept. Residual risk: a quotation that resolves after rewriting but has
been silently falsified. Accepted, because the alternative is leaving 414 dead links.

**Phase 2 edits `src/`, which was scoped out.** Sixteen source files cite
`docs/superpowers/` in doc comments. The edits are comment-only and `cargo check`
proves no semantic change, but `src/` will appear in that diff.

**Deleting `.claude/references/` rests on a six-week-old subset finding.** `docs/`
has been edited since, so a reference could now carry something its counterpart lost.
Mitigated by making the per-pair comparison a gate in §4.5 rather than an assumption.

**Archiving examples stops compiling them.** Nothing outside `examples/` references
the thirteen, so future rot becomes invisible. That is the intent, but it is a
one-way door, mitigated by recording the last-known-good commit per artifact.

**Compressing `CLAUDE.md` moves detail from always-loaded to trigger-loaded
context.** An agent that never triggers `ddrs-dev` could re-open a closed question.
Mitigated by keeping the verdict and the do-not-reopen sentence in `CLAUDE.md` and
moving only enable steps and gate commands.

**Phase 1 touches `.claude/skills/ddrs-dev/references/config.md`,** which
`landscape-deriv-objective` also edits. A small, hand-resolvable conflict, not zero.

**The verifier is new code that could itself be wrong.** A verifier with a false
negative gives false confidence. Mitigated by seeding it with the known-bad
citations from §2 and requiring it to flag all of them before any are fixed.

---

## 7. Assumptions

- **`experiments/` is live research input, not residue.** The landscape and adjoint
  bundles are the current workstream, so pruning them would destroy active work. This
  deliberately narrows the requested sweep.
- **The 36 findings docs are the scientific record.** They move, and their content is
  never edited. Only inbound paths change, and only where the path is a live pointer.
- **`docs/journal/` and `ddrs-journal` are live** as of 2026-09-10, so they move
  rather than archive.
- **`src/` is out of scope except for doc-comment path strings.** No behavior change
  is intended anywhere in this work, and `cargo check` is the proof.
- **The book's published URLs must not change.** Readers and external links depend on
  them, which is why §5.1 carries a proof obligation rather than an assertion.
- **The skills library must record what this session learned.** Per the repository's
  own rule, a findings doc and a `ddrs-dev` update are part of the deliverable, not a
  follow-up.

---

## 8. Deliverables

1. Phase 1: the truth pass of §4, plus `scripts/verify_doc_paths.py`.
2. Phase 2: the restructure of §5.
3. `research/findings/2026-09-11-repo-cleanup-findings.md`, recording what was
   verified wrong, what moved, and what the verifier now prevents.
4. `ddrs-dev` updated with the new layout and the citation policy of §4.2.
