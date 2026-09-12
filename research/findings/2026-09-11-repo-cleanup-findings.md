# Repository cleanup, phase 1 findings (2026-09-11)

Branch: `worktree-repo-cleanup`, off `origin/master` @ `3412a78` (post PR #42).
Scope: Tasks 1-9 of `docs/superpowers/plans/2026-09-11-repo-cleanup.md`, the
content-only phase: `CLAUDE.md`, the four `.claude/skills/` packages, the
mdBook pages under `docs/`, `README.md`, and the new citation verifier in
`scripts/`. No file moved and no path changed; Phase 2 (Tasks 10-15) does the
restructure on top of this corrected text.

**One-line verdict:** every claim the 2026-09-11 design (`docs/superpowers/specs/2026-09-11-repo-cleanup-design.md`)
flagged as verified-wrong in `CLAUDE.md`, the four skills, and the book pages
was corrected; a new gate (`scripts/verify_doc_paths.py`) now fails the build
if a future edit drifts a citation the same way; and the review loop that
executed the plan caught roughly seven errors that originated in the plan
itself against about one implementer slip, which is this run's own strongest
argument for keeping that loop in Phase 2.

---

## 1. Method

This is the successor to `docs/2026-07-30-docs-and-skills-audit.md`, whose
open follow-up list (its section 4) fed directly into the 2026-09-11 design's
section 2. That design re-verified every inherited claim against source
before relying on it, rather than trusting the six-week-old audit.

Execution ran as eight dispatched tasks (1 through 8), each following the
same loop: an implementer made the change and reported it; a controller
independently re-verified the implementer's claims against source, not
against the report; a reviewer then re-verified the same diff end to end,
weighted toward whichever claim carried the most risk (for example, whether
every *new* claim in a rewritten diagram was true, not just whether the old
false ones were gone); and any finding from either pass went through one or
more fix rounds, each itself re-reviewed, before the task was marked complete.
The highest-stakes claims were checked a third time by an independent
spot-check in the re-review (three per scoped re-review, chosen by the
re-reviewer, not the fixer).

This loop is the reason the process findings in section 4 exist: several of
its most valuable catches were errors in the plan and the design document
itself, not in any implementer's work, and they were only visible because
every claim, including the brief's own, was checked against source rather
than assumed.

---

## 2. What Phase 1 corrected

### 2.1 `CLAUDE.md`

All nine rows of the design's section 2.1 table were corrected (commit
`b85e45d`, Task 4): the `src/sparse.rs` diagram entry became `src/sparse/`;
the claimed `spike_backward/` entry was removed (the directory does not
exist); `nse-batch`, `optimizer`, `use_grad_accum` and `grad_accum_steps` are
now described as landed, not pending on PR #31; "three skills" became four
skills (`ddrs-run` added); the `ddrs init` stub citation, the
`validate_subdivision_reaches_the_builder` citation, and the
`--workflow eval` error citation were each re-pointed at their current lines;
the data-source-group line was expanded to name all nine files in
`config/sources/`; and "First study: `adjoint`" now also names `landscape`.

The diagram rebuild that fixed the `sparse.rs`/`spike_backward` pair also
added eight subsystems the old diagram omitted entirely: `routing/leakance.rs`,
`cuda_graph/`, `nn/disagg_head.rs`, `adjacency/`, `baseline/`, `training/`,
`pretrain/`, `experiment/`, plus `cli/`, `bin/`, `vendor/`, and `ddrs-py/`.
The fix for one defect (a stale entry) briefly reintroduced the same defect
class at a smaller scale: the first version of the rebuilt diagram named only
4 of the 12 files under `src/bin/`, which the Task 4 review caught and fixed
by naming the binary *families* (`dump_parameters`, four `pretrain_disagg*`,
three `probe_*`, etc.) rather than an enumerated subset.

Leakance and reach subdivision, the two closed campaigns, were compressed
from 84 and 54 lines to short status blocks (Task 5, commits `e2b95b8` through
`bc52656`): the verdict, the do-not-reopen sentence, and a pointer to the
owning reference, with the enable steps, parameter ranges, and gate commands
moved into `ddrs-dev/references/config.md` and `testing.md`. Two numeric
errors were introduced and caught inside this compression itself; see
section 3.2.

A new "cite symbols, not lines, inside `src/`" convention was added to the
Conventions section (Task 6, commit `baa97d6`): `file.rs::symbol` instead of
`file.rs:123`, because line numbers drift silently and a symbol citation can
be checked by a machine.

### 2.2 The four skills

Every row of the design's section 2.2 table was corrected:

- `ddrs-dev/SKILL.md:262`'s claim that the mdBook is "a strict superset of the
  **deleted** `.claude/references/` copies" is now true in both halves: the
  deletion happened (Task 2) and the superset claim was independently
  re-verified by re-reading all 12 deleted files against their `docs/`
  counterparts in full, not just diffing tokens (Task 2 review).
- `ddrs-dev/references/build-and-env.md:16`'s claim that `docs/setup.md`
  disagrees about CPU-only compilation was corrected: `docs/setup.md` already
  said the same thing, so the skill was "correcting" a doc that agreed with
  it.
- `ddrs-dev/references/config.md`'s validator count is now "ten validators
  run directly ... plus a nested `validate_subdivision_reaches_the_builder`,
  plus one at dataset open" (was "four"), with the missing
  `validate_enforce_positivity` and `validate_loss` rows added to the guards
  table (Task 7). The true count is ten, not the nine the design's own brief
  assumed; see the process finding in section 4.1.
- `ddrs-run/SKILL.md:245`'s claim that `ddrs experiment` is "Not on master"
  was corrected; `src/experiment/{adjoint,landscape}` and the `Experiment`
  subcommand are present, merged as PR #39 (Task 7). The location this claim
  was originally reported at was wrong; see section 3.6.
- `ddrs-dev/references/traps.md`'s duplicate `## T11` heading was renumbered
  to `T13`, and the symptom index now covers it (Task 7, commit `16adb9c`).
  The renumbering itself broke a cross-reference; see section 3.3.
- `ddrs-dev/references/gauge-population.md:52-53`'s citation for
  `is_headwater` was corrected from a nonexistent line-128 "order length > 1"
  description to `src/data/store/zarr.rs::GageSubgraph::is_headwater`, whose
  actual body is `self.indices_0.is_empty()` (Task 6/7).
- `ddrs-dev/references/config.md:123-126`'s `use_precip` passage, which
  described the phantom key as still live, was deleted; the key was already
  removed from `CLAUDE.md` and the YAML by an earlier, unrelated commit.

Hard-coded counts were removed throughout: `testing.md`'s "70 test files, 233
`#[test]` fns" claim (actual 88 files, 363 fns at the time this was checked;
commit `16adb9c`'s own message says "364 tests", but `ls tests/*.rs | wc -l`
gives 88 and `grep -rhE '^\s*#\[test\]' tests/*.rs | wc -l` gives 363, so 363
is the measured figure and the commit message is the one that is off by one),
`ddrs-dev/SKILL.md`'s per-reference line-count table, and `ddrs-eval-plots`'s
file-list counts were all replaced with the file names alone or with the
editorial rule the library already stated but did not follow.

Duplication of `CLAUDE.md` was trimmed in `config.md` (the stale `exp_train`
blockquote at lines 73-76, carefully scoped so the new `nse-batch-deriv`
documentation PR #42 had just added at lines 78-99 was not deleted alongside
it; see section 3.7 for why that distinction mattered).

### 2.3 Book pages and README

All of the design's section 2.3 items were fixed (Task 8, commits `61d774f`
and `4798253`): the `use_cuda_graphs: true` shipped-value claim, which spans
five files (`intro.md`, `setup.md`, `running.md`, `inputs-formatting.md`,
`perf.md`), was corrected everywhere; `tau`'s documented default went from
the stale 3 to the real 9, with a note that the struct literal's 3 is never
observed via a YAML load; `README.md`'s framing of `nse-batch`/`adadelta` as
"once PR #31 lands" was updated (it merged as `24cb4d0`); `docs/reference/baseline.md`'s
`init -> plan -> run` lifecycle description was brought in line with the
already-fixed `setup.md`/`running.md`; `docs/architecture.md`'s `src/adjacency/`
table gained `gridded.rs` and `subdivide.rs`; `running.md` gained `ddrs experiment`;
and `use_grad_accum`/`grad_accum_steps` were added to the training-configuration
section.

Fixing this pass surfaced a much larger defect class than the design had
catalogued: stale or undercounted enumerations. Section 4.7 records it in
full, because it is the single most common defect this run found.

### 2.4 `.claude/` deduplication

`.claude/2026-05-29-ddrs-docs-{design,plan}.md` were deleted as exact
duplicates of their `.claude/specs/` counterparts, confirmed by git blob
identity rather than a checksum tool (`3d8e7eb`, Task 3). `.claude/references/`
(12 files, 2,243 lines, measured via `git archive 927f755 .claude/references
| tar -xO | wc -l`) was deleted in full (commit `3522c38`'s own deletion count
of 2,268 lines also includes `docs/intro.md` and
`docs/reference/burn-autograd.md`, which the same commit trimmed; 2,243 is
the directory's own size) in Task 2 after
a complete re-read of every pair against its `docs/` counterpart, not a
sampled check. Section 4.4 names what that deleted directory was actively
propagating, not merely duplicating.

### 2.5 The citation gate

`scripts/verify_doc_paths.py` and its test, `scripts/test_verify_doc_paths.py`,
were built in Task 1 and strengthened in Tasks 6 and 7. The verifier walks
`CLAUDE.md`, `.claude/skills/**`, and the book pages for anything shaped like
a repo-rooted path, a markdown link target, or a `file.rs::symbol` citation,
and resolves it against the tree. A citation in `CLAUDE.md` or
`.claude/skills/**` that fails to resolve is a strict failure (the gate's exit
code); the same failure inside a `docs/` prose page is a warning only, because
Phase 1 deliberately left the book's drift toward Phase 2's restructure rather
than hand-fixing all of it. Section 4.8 records the rule's own history.

---

## 3. Findings the design document did not and could not know

These surfaced only while executing the plan, against source as it stood
during this run, not at design time.

### 3.1 The citation gate's real trajectory: 9, then 10, then a transient 21, then 12, then 0

The design's section 2.1/2.2 tables were built from a verifier that, on the
tree at `3412a78`, reported **9** strict failures (Task 1, commit `d48d143`,
matching the design's own predicted list exactly).

Task 2's deletion of `.claude/references/` raised the count to **10**: not a
regression, but a new, correct failure, because `ddrs-dev/SKILL.md:262`'s
backtick-quoted `.claude/references/` path became unresolvable the instant
the directory it named was deleted, even though the surrounding prose (past
tense, "the deleted ... copies") was accurate. Tasks 3, 4, and 5 held the
count at 10 while making unrelated corrections.

Task 6 landed the enforcement rule that rejects any `src/` line citation
outright (not just the ones already known to be wrong). Before any of the 13
newly-enforced citations were converted to `file.rs::symbol` form, the count
spiked to roughly **21** (prose-level warnings rose from 210 to 402 at the
same time). This was the gate firing correctly on citations nobody had
touched yet, not a regression; a first, uncommitted implementer attempt was
stopped by the user mid-task specifically because the controller misread this
spike as alarming before confirming it was expected. Once Task 6 converted 11
of the 13 (deliberately deferring 2 to Task 7, because the content around them
was already scheduled for deletion there), the count settled at **12**: the
10 pre-existing unrelated failures plus exactly the 2 deferred line citations.

Task 7 converted the last 2 and fixed the 10 pre-existing failures (a mix of
dead paths, a doubled `fixtures/fixtures/` typo, and stale test-file
citations). The gate reached **0** (commit `16adb9c`) and has held at 0
through Task 8 and this task's own re-run.

### 3.2 Compression manufactured a factual error out of hedged prose

The pre-compression reach-subdivision text in `CLAUDE.md` read "`Cr > 2` /
`c3 < 0` (3.93% to 0.31%)", vague in its attribution but not false: both
figures were present, unassigned to a single label. Task 5's first
compression pass dropped the second label to shorten the sentence and
produced a precise, wrong attribution: it assigned the `c3 < 0` figures
(3.93% to 0.31%) to `Cr > 2`, whose real figures (per
`.claude/REACH-SUBDIVISION.md`'s table) are 2.10% to 0.16%. The review caught
this and a second numeric error in the same commit: a claim that
`.claude/REACH-SUBDIVISION.md` lists "all seven" subdivision fields, when it
names five by name and two (`reference_discharge_coefficient`,
`reference_discharge_exponent`) appear nowhere in it, only as `coeff`/`exp` in
the governing formula; those two live in `src/config.rs`. Both were fixed in
commit `bc52656`. The lesson: shortening always-loaded context is not purely
subtractive, and needs the same numeric re-checking as writing new prose.

### 3.3 Renumbering a trap broke a cross-reference no gate and no keyword search could find

Task 7 renumbered the duplicate `## T11` heading in `traps.md` to `T13`. That
orphaned `research-status.md:154`, which said "see `traps.md` T11" while
describing the autodiff-tape-leak trap (the renumbered one is now T11's
replacement, a different trap, the axis-sniff one). A keyword grep for
"Forward-only" or "leak_probe" would not have found the orphaned line,
because its actual text was "Leaked the autodiff tape per forward-only eval",
phrased differently from either search term; `scripts/verify_doc_paths.py`
cannot see it either, because `T11` is a label, not a path. The fix (commit
`baf6253`) required a sweep of every trap-number cross-reference by subject:
does the trap this reference points at actually describe the same bug as the
sentence around it, not whether the pointer resolves.

### 3.4 `.claude/references/` was propagating at least eight wrong claims, and `docs/` frequently corrected it

The design's section 2.4 described `.claude/references/` as a stale subset of
`docs/`. Re-reading all 12 deleted files against their `docs/` counterparts in
full (the Task 2 review) found it was doing more than idling: it was actively
carrying wrong claims the `docs/` counterpart had already corrected.
Confirmed, by file:

1. `ddrs-setup.md:21`: "`use_cuda_graphs: true` defaults in
   `config/merit_training.yaml`". False since 2026-08-19; the shipped default
   is `false`.
2. `ddrs-setup.md:21`: "The CPU path needs none of this", meaning no CUDA.
   False: `burn-cuda` and `cudarc` are non-optional dependencies, so a CUDA
   toolkit is required to compile even for CPU execution.
3. `ddrs-setup.md:22`: an obsolete "desktop-only DDR working tree" caveat for
   V1 fixture regeneration, retired by `CLAUDE.md` invariant 1 on 2026-08-19.
4. `ddrs-architecture.md` (Pair B): named `merit_train` as a binary under
   `src/bin/`. No such file exists; `docs/architecture.md`'s equivalent row
   lists the real set.
5. `ddrs-graph-objects.md` (Pair G): cited `mc_routes_linear_chain` as a test
   in `tests/mmc.rs`. That test does not exist.
6. `ddrs-reading-inputs.md` and `ddrs-running-the-code.md` (Pairs I and K):
   both cited `src/sparse.rs` as a path. It is a directory.
7. `ddrs-algorithm.md` (Pair A): gave `cargo test --test sp8_gradcheck --
   --ignored` as the way to run the gradcheck. Nothing in that file is
   `#[ignore]`, so the command runs zero tests and exits 0, a gate that
   cannot fail.
8. `ddrs-perf-and-cuda-graphs.md` (Pair H): described `PersistentScratch` as
   32 handles over an incomplete grouping of the K1 inputs. The real count is
   39 handles, with `q_prime_t` and explicit `q_eps`/`side_slope` buffers the
   old grouping omitted; `docs/reference/perf.md`'s equivalent section is
   both correct and more detailed.

The two occurrences of claim 6 are counted once as a claim, not twice, which
is why this list reads as "eight" rather than nine rows.

### 3.5 `ddrs-dev/SKILL.md` carried the same two drifted citations as `CLAUDE.md`, because it was written by copying `CLAUDE.md`

Before Task 6, `ddrs-dev/SKILL.md:142` cited `src/cli/run.rs:322` for the
`--workflow eval` error string, and `:144` cited `src/bin/ddrs.rs:167` for the
`ddrs init` stub. Both citations had already drifted (the real lines are 324
and, for the `Cmd::Init` arm, 193-196) and both are the exact same two
citations Task 4 had already fixed in `CLAUDE.md` at lines 151 and 154. The
skill was not independently wrong; it had been written by copying
`CLAUDE.md`'s text, and inherited `CLAUDE.md`'s error along with everything
else. One stale line in always-loaded context became two stale lines the
moment it was duplicated into a skill. This is the mechanism the entire
cleanup is premised on (duplication propagates staleness), observed directly
rather than inferred from the design's reasoning about it. Both are now bare
`src/cli/run.rs` and `src/bin/ddrs.rs` citations with no line number, the same
form `CLAUDE.md` uses, so neither can drift the same way again.

### 3.6 Two citations converted in Task 6 were already pointing at the wrong code before conversion

Converting `CLAUDE.md` and the skills to `file.rs::symbol` citations was not
merely a style change; verifying each conversion against source (the Task 6
review, using `git show` to compare each new symbol's actual content against
what the old citation's prose claimed) found two citations that had silently
drifted onto unrelated code before anyone touched them:

- `.claude/ARCHITECTURE.md:255` cited `src/routing/mmc.rs:289-294` for the
  CUDA-graph capture gate. That range had drifted onto an unrelated
  `denormalize(params.n, ...)` block; the gate the prose actually describes
  (`use_cuda_graphs && sparse_solver == Cuda`) lives at lines 370-372, with
  its guarded call (`self.try_capture_forward_graph();`) at 374, inside
  `setup_inputs`. The citation now reads
  `src/routing/mmc.rs::setup_inputs`.
- `ddrs-dev/references/gauge-population.md:44` cited
  `src/data/store/gage_csv.rs:62` for the `RawRow` struct's `abs_diff` field.
  That line had drifted to 64. The citation now reads
  `src/data/store/gage_csv.rs::RawRow`.

Both were caught only because the symbol-conversion work forced someone to
open the cited line and check it against the surrounding prose, not because
any gate flagged them; a verifier that only checks whether a line number is
in range would have accepted both as "resolved". This is the concrete,
already-incurred cost of line-number drift that motivated the
cite-a-symbol-not-a-line policy in `CLAUDE.md`'s Conventions section, not a
hypothetical one.

### 3.7 An audit finding was misfiled: the location was wrong, the finding held

The design's section 2.2 table reported the `ddrs experiment` staleness claim
at `ddrs-run/references/commands.md:245`. `ddrs-run/references/commands.md`
is 178 lines long, so line 245 cannot exist in it. The text the finding
describes (`ddrs experiment <name>` characterized as "Not on master") is at
`ddrs-run/SKILL.md:245` instead, in a 261-line file. The underlying claim was
correct (`src/experiment/{adjoint,landscape}` and the `Experiment` subcommand
are present on master, merged as PR #39) and was fixed at the right location
in Task 7; only the citation in the design document pointed at the wrong
file. This is why every carried-forward finding in this run was re-checked by
hand against source rather than trusted on the strength of the document that
reported it, including the design document's own citations.

### 3.8 `examples/leak_probe.rs` was nearly archived as a leakance artifact

The design's section 5.4 (Phase 2, not yet executed) originally listed
`examples/leak_probe.rs` for archiving alongside the other leakance-campaign
probes, on the strength of the filename matching "leak". It is not a
leakance artifact: it is the autograd-tape-leak repro, the discriminating
test `ddrs-dev/references/traps.md` names for a live, still-relevant trap
(the one renumbered in section 3.3 above), and it is cited directly from
`src/experiment/landscape/objective.rs`, the active landscape workstream.
Name-matching on "leak" would have deleted a working diagnostic that a
current experiment depends on. The plan was corrected before any Phase 2 task
was dispatched, and Task 13's archive admission rule is now written to defer
to the findings document that cites an artifact rather than to filename
similarity, specifically because of this near-miss.

---

## 4. Process findings: the ten lessons this execution produced

These are general lessons about running this kind of cleanup, each grounded
in one specific, named incident from this branch's history so a reader can
check it independently. They are not in the design document, because the
design document could not have known them; they were discovered while
executing it.

1. **The expensive errors were in the specification, not the work.** Across
   eight tasks the review loop caught roughly seven false claims that
   originated in the plan or its briefs, against about one implementer slip.
   Examples: the verifier's docstring claimed it caught drifted line numbers
   when it discards the line suffix entirely; a predicted post-compression
   line count of 480 that was wrong by 80 and would have shipped in the PR
   description; "nine validators" when `Config::from_yaml_file` runs ten;
   `src/bin/` described as four binaries when it holds twelve. The plan was
   written from a six-week-old audit plus one reading of the tree, and the
   tree had moved. The practice that caught these was re-verifying each
   task's premises against source immediately before dispatching it, not the
   post-hoc review.

2. **Compression can manufacture a factual error out of hedged prose.** The
   pre-compression text read "`Cr > 2` / `c3 < 0` (3.93% to 0.31%)", vague but
   not false. Dropping the second label to shorten it produced a precise,
   wrong attribution: `Cr > 2` is 2.10% to 0.16%. Shortening always-loaded
   context is not purely subtractive and needs the same numeric re-checking
   as new writing.

3. **Renaming a label breaks references that no gate and no keyword search
   can find.** Renumbering a duplicate `## T11` trap heading to `T13`
   orphaned `research-status.md:154`, which said "see `traps.md` T11" while
   describing the autodiff-tape trap. A keyword grep missed it because the
   line described the bug in different words than the trap's title, and
   `verify_doc_paths.py` is blind to it because `T11` is a label, not a path.
   Cross-references keyed on numbers need a by-subject sweep after any
   renumbering.

4. **Duplication propagates staleness, observed directly rather than
   inferred.** `ddrs-dev/SKILL.md:142` and `:144` carried the same two
   drifted citations that `CLAUDE.md` carried, because the skill had been
   written by copying `CLAUDE.md`. One stale line became two. This is the
   mechanism the whole cleanup is premised on, caught in the act.

5. **An audit misses what it does not think to look for, and a default
   flip's blast radius goes uncounted.** The three-auditor sweep graded
   `docs/setup.md` and `docs/intro.md` CLEAN and `docs/usage/running.md`
   MINOR. All three were wrong about `use_cuda_graphs`, and `setup.md`
   carried five stale claims. The auditors checked against the prior
   follow-up list and for newly-landed features; nobody swept for the
   2026-08-19 `ddr_match` default flip, so every consequence of that one
   change went unexamined. When a default changes, enumerate everything that
   asserts the old value.

6. **Verifying that a pointer resolves is not verifying that it delivers.** A
   fix round confirmed each status-block pointer's target existed, and the
   claim "all seven fields" was still false, because
   `.claude/REACH-SUBDIVISION.md` names five. `reference_discharge_coefficient`
   and `reference_discharge_exponent` appear nowhere in it. Check the
   promise, not just the address.

7. **Counts of sets that grow are the single most common defect in this
   repository.** Ten instances found, and the class was only recognised
   because the third one prompted a deliberate sweep:

   | Claim | Reality |
   |---|---|
   | `CLAUDE.md` diagram: 4 `src/bin/` binaries named | 12 |
   | `docs/usage/running.md`: "ten binaries", twice | 12 |
   | `running.md`: "sixteen examples" | 17 |
   | `running.md`: "three groups" naming one `probe_*` binary | the `probe_*` family has 3 |
   | `docs/architecture.md`: "fifteen top-level modules" | `src/lib.rs` declares 16 |
   | `architecture.md` module map | **omitted `src/experiment/` entirely** |
   | `docs/usage/inputs-formatting.md`: "runs four validators" | 10 |
   | `inputs-formatting.md`: disagg has "no `enabled` flag", "Eight keys" | 9 fields, `enabled` is the first |
   | `ddrs-dev/SKILL.md` trap index: T1-T10 | T1-T13 |
   | `ddrs-dev/SKILL.md`: 4 loss kinds | 5 |
   | `README.md` data-source groups table | omitted `conus-gridded` |

   Two observations make this more than bookkeeping. First, in every single
   case the omitted members were the **newest**, so each index was stalest
   exactly where a reader most needed it. Second, the `architecture.md`
   module map had lost an entire subsystem, and that was discovered only
   because someone checked a count: the count was a symptom whose cause was a
   missing row in the document that exists to map the codebase.

   Two documents also drifted to the **same** wrong validator count of four
   independently, which is the signature of both being written from a shared
   stale reading rather than from source.

   The fix that holds is to name families, state a range, or defer to the
   authoritative list, never to correct one number to another. A legitimate
   exception is a table that IS the enumeration, which a reader can recount
   against source; a bare count floating in prose is not. The repository
   already had this rule for test counts and had not generalised it.

   The defect recurred during the task that wrote this finding: Task 9's own
   scratch report first said "twelve fixture-based cases" in
   `scripts/test_verify_doc_paths.py` where there are fourteen, caught only
   on a later review pass. A bare count is apparently easy to write and hard
   to self-audit even while cataloguing the exact failure mode it is an
   instance of.

8. **A gate's usefulness is set by its false-positive rate, not its
   coverage.** The first draft of `verify_doc_paths.py` reported 288 strict
   failures, almost all bare filenames and runtime artifacts. A gate that
   noisy gets switched off. Four changes took it to 9 actionable findings:
   requiring a repo-rooted path, skipping gitignored workspace paths,
   resolving against the citing file's directory, and testing existence
   rather than file-ness because an `.ic` store is a directory.

9. **Two citations were already pointing at the wrong code before
   conversion, which is the concrete cost of the drift.**
   `.claude/ARCHITECTURE.md`'s `src/routing/mmc.rs:289-294` had drifted onto
   an unrelated `denormalize` block, while the gate it described sits at
   370-372 inside `setup_inputs`. `gauge-population.md`'s
   `src/data/store/gage_csv.rs:62` had drifted to 64. The line-citation
   problem was not hypothetical at the time it was fixed.

10. **A test written against the state of a system mid-repair will fail when
    the repair completes.** The Task 6 brief told its implementer to "add a
    case that runs `python3 scripts/verify_doc_paths.py` with no arguments
    from the repository root and asserts the exit code is 1", because at the
    moment that instruction was written the real tree carried twelve strict
    failures and exit 1 was the observed, correct behaviour. Task 7 then
    drove the strict count to zero, which is the entire point of the work,
    and the frozen assertion started failing precisely because the project
    succeeded: a test that can only pass while its companion defect still
    exists. The fix (this task) does not flip the expectation from 1 to 0
    either, since that only moves the coupling to the other direction: the
    next contributor who introduces a bad citation would then get a failing
    unit-test suite instead of a clear citation report, and would go looking
    for a bug in the script rather than in their own edit. The case now
    asserts the interface instead of a verdict: the no-argument invocation
    exits 0 or 1 (anything else, a 2 or a traceback, is the only real
    failure) and its stdout reaches the `unresolved in agent context`
    summary line, with no assertion about how many failures it found. The
    instruction that created the bug came from the plan's own dispatcher, not
    from an implementer, which makes this another instance of finding 1
    above: the specification was where the expensive error lived, discovered
    here only because the gate set was run in full at the end of Phase 1
    rather than trusted from its last green run.

---

## 5. `CLAUDE.md` line count: the measured trajectory, not a predicted one

`CLAUDE.md` was 663 lines at branch start (`3412a78`). Task 4's diagram
rebuild, which named eight previously-omitted subsystems (section 2.1 above),
took it to 684, corrected from an initial estimate of 663 once the controller
actually counted the rebuilt file. The spec and plan had separately predicted
the *post-compression* target would land near 480; that number was wrong by
about 80 lines and was caught only because a later task was told to quote the
measured count rather than the predicted one (commit `72a0029`). It would
otherwise have shipped, unverified, in Task 15's pull-request description.

Task 5's compression of the leakance and reach-subdivision sections to status
blocks took the file from 684 to 577, then to 578 and 581 across two fix
rounds that corrected the numeric errors in section 3.2. Task 6 added the
seven-line "cite symbols, not lines" convention paragraph, landing at 588.
Tasks 7 and 8 touched other files and left `CLAUDE.md` unchanged.

**Measured now, by running `wc -l CLAUDE.md` against this branch's HEAD
(`f2ecf52`) on 2026-09-12, immediately before this task's own commit: 588
lines.** This task's commit touches only `docs/2026-09-11-repo-cleanup-findings.md`
and `.claude/skills/ddrs-dev/references/testing.md`, neither of which is
`CLAUDE.md`, so this count should still read 588 after this commit; a reader
checking it later should re-run `wc -l CLAUDE.md` rather than trust this
sentence, which is itself a count that will drift the moment someone next
edits the file.

---

## 6. What Phase 2 covers (not yet done)

Everything below is still open, per the design document's sections 5 and the
plan's Tasks 10-15. Recording it here so this findings document is a complete
phase boundary, not a partial one:

- **Split the book from the research record** (Task 10): move `book.toml`'s
  `src` to `docs/book/`, move the 14 book pages and `SUMMARY.md` with it, and
  prove published URLs are unchanged by diffing the built file list before
  and after.
- **Create `research/`** (Task 11): `findings/`, `specs/`, `plans/`,
  `why-analysis/`, `figures/`, `journal/`, `archive/`, consolidating
  `docs/superpowers/` and `.claude/specs/` (one series split by the date the
  naming convention changed) into one tree.
- **The link rewrite** (Task 12): roughly 414 references across roughly 150
  files get their `docs/2026-`, `docs/superpowers/specs/`,
  `docs/superpowers/plans/`, `docs/why-analysis/`, `docs/journal/`, and
  `docs/figures/` prefixes rewritten to their `research/` equivalents. The
  referrers inside the moving set are rewritten in bulk; the roughly 130
  referrers outside it (mostly `config/experiments/*.yaml` comments and
  `src/*.rs` doc comments) are read individually rather than swept, because an
  anchored substitution cannot tell a live pointer from a historical
  quotation.
- **Archive the closed campaigns** (Task 13): `research/archive/{scripts,examples}/`,
  with an admission rule written against the citing findings document (the
  lesson of section 3.8 above), not against filename similarity.
- **`output/` and `docs/images/`** (Task 14): move the eight tracked files
  under gitignored `output/synthetic_n/plots/` into `research/figures/`, and
  delete the unreferenced `docs/images/.gitkeep`.
- **The single pull request** (Task 15): opens once Phase 2's gates are
  green. This task does not push and does not open a PR; that is Task 15's
  Step 5.

**Update (2026-09-12):** all five items above are complete. Section 8 below
records what actually happened, including findings Phase 1 had no way to
predict. This section is left as written, since it is a correct record of
what was still open at the time Task 9 wrote it.

---

## 8. Phase 2: the book tree holds only the book

Tasks 10 through 14 did the restructure this document described in section 6
as not yet done. Measured at this task's HEAD:

- `book.toml`'s `src` now points at `docs/book/` (Task 10, commit `cd4ec55`).
  `find target/book -type f | sort` before and after the move differs by
  exactly 10 removals (the 9 `docs/figures/` PNGs plus `docs/images/.gitkeep`)
  and 0 additions; every one of the 18 `.html` output files is
  byte-identical, so no published URL moved. `docs/` now contains exactly one
  entry, `book`.
- `research/` now holds `archive`, `figures`, `findings`, `journal`, `plans`,
  `specs`, `why-analysis` (Task 11, commit `5f2771a`), 135 `.md` files total
  (38 findings, 40 specs, 44 plans, 7 why-analysis, 5 journal, 1 archive),
  consolidating `docs/superpowers/` and `.claude/specs/` by pure rename: the
  move's own commit reports 144 files changed, 0 insertions, 0 deletions, and
  `git log --follow` still traces every moved file through its pre-move
  history.
- 406 references were repointed at the new `research/` paths (Task 12,
  commit `ec7014d` plus four fix rounds ending `cafc6d2`); 68 were left
  because they are historical quotations of the pre-move layout inside
  documents describing that layout (the cleanup's own spec, plan, findings
  doc, and the 2026-07-30 audit), and one coincidental `docs/superpowers/`
  text match that is prose about a grep's expected output, not a path.
- `research/archive/{scripts,examples}/` holds 24 retired scripts and 13
  retired examples (Task 13, commit `8fd0c6d`), `examples/*.rs` is down to
  4 (`benchmark_hydrograph`, `compare_ddr_sandbox`, `dump_init_params`,
  `leak_probe`) from 17, under an admission rule keyed to a citing findings
  document or plan declaring the campaign closed, or to nothing citing the
  artifact at all, never to filename similarity.
- The 8 tracked files under gitignored `output/synthetic_n/plots/` moved to
  `research/figures/synthetic-n/` as `git mv` renames (Task 14, commit
  `ca68012`); `git ls-files output/` is empty.

Measured now: the citation gate is at 0 strict failures, 80 prose warnings
(`python3 scripts/verify_doc_paths.py`); `cargo check --examples --tests`
exits 0; `cargo test --test ddr_sandbox_match --test gridded_bundle` is 6/6;
`mdbook build` is clean.

---

## 9. Findings Phase 2 surfaced that Phase 1 had no way to know

### 9.1 A number we nearly shipped as an achievement was a scope change, not a cleanup

The citation gate's prose-warning count fell from roughly 516 mid-run to 80.
That looks like thorough hygiene. It is almost entirely Task 11 moving 135
documents from `docs/` (which `verify_doc_paths.py`'s `WARN_GLOBS` scans) to
`research/` (which it does not scan). The documents were not fixed; the
scanned set shrank out from under them. Stating the drop as a hygiene
improvement in the PR description would have been exactly the kind of false
claim this cleanup exists to remove.

**Ruling:** `research/**` stays unscanned, on purpose, by both tiers of the
gate. A findings document, a spec, or a plan is an immutable record: its path
citations are snapshots of what was true when it was written, not claims
about what is true now. Linting an immutable record against the current tree
is a category error, since a findings doc from three moves ago is *supposed*
to cite paths that no longer exist. Scanning `research/**` would emit
hundreds of warnings about correctly-historical paths and recreate the
288-noise problem the verifier was originally tuned to escape. This decision
is now a comment in `scripts/verify_doc_paths.py` next to `WARN_GLOBS`,
converting what started as an accident of the move into a documented
decision a future contributor can read without having to reconstruct it.

### 9.2 Archiving by filename would have silently broken a live test and orphaned a published gate

The Task 13 brief listed `scripts/sp8_check_scatter.sh` and
`scripts/sp10_check_launches.sh` for archiving, on the strength of both
looking like closed-campaign spike checks from the sp8/sp10 series. Both are
live. `tests/sp8_v7_profile.rs:21-24` spawns the first directly:
`Command::new("bash").arg("scripts/sp8_check_scatter.sh").status().expect("spawn sp8_check_scatter.sh")`.
`docs/book/reference/perf.md:194` and `:301` name the second as the current
V10 gate command, with a measured result (29.2%) in the gates table. The
implementer reversed both, against the brief, and was right to.

The test that spawns `sp8_check_scatter.sh` is `#[ignore]`d, so archiving the
script would not have failed CI. It would have failed silently, for whoever
next ran the profiling gate with `--ignored`, who would have seen a missing
file and had no reason to connect it to a repository cleanup weeks earlier.
This is the headline argument for writing the archive admission rule against
what a findings document or plan says about an artifact's campaign status,
never against whether its filename resembles a closed campaign's vocabulary,
alongside the `examples/leak_probe.rs` near-miss from Phase 1 (section 3.8
above: a "leak" filename that is an active autograd-tape-leak repro, not a
leakance artifact).

### 9.3 A citation broke twice in one cleanup, and only a widened gate caught the second break

`src/sparse/mod.rs:11` cited `.claude/references/ddrs-burn-autograd.md` at
branch start. Task 2 repointed it to `docs/reference/burn-autograd.md`, which
was correct at the time. Task 10's book move then relocated that file to
`docs/book/reference/burn-autograd.md`, silently re-breaking the citation
Task 2 had just fixed. Nothing in Task 10's own scope caught this, because
Task 10's gate run only covered the book pages it was moving, not doc
comments elsewhere in `src/`. The break was only found when Task 12's fix
round 1 widened `WARN_GLOBS` to `src/**/*.rs` for an unrelated reason (the
`.claude/specs/` reference cleanup) and the wider scan turned up this file
along with it. The citation now reads `docs/book/reference/burn-autograd.md`.

The lesson: a citation fixed early in a multi-step restructure is not fixed
for the rest of the restructure. It can be re-broken by a later, unrelated
step, and nothing catches that except running the gate, at its fullest
coverage, after every step, rather than trusting the last green run.

### 9.4 A commit message self-certified a false claim about its own diff

Commit `ec7014d` ("docs: repoint 406 references at research/") ends its
message with "Touches `src/` and `scripts/` doc comments/docstrings only; no
logic changed." That is false for `scripts/`: `scripts/journal.py`'s
`journal_dir()` (line 80-81) changed from returning
`root / "docs" / "journal"` to `root / "research" / "journal"`, a real
runtime behavior change, not a comment edit, plus two hardcoded paths in
`scripts/test_journal.py` changed alongside it. The change itself is correct
and necessary (`docs/journal/` no longer exists after Task 11), and
`python3 scripts/test_journal.py` passes. The defect is narrower and more
interesting than a bug: a task whose entire purpose was auditing the
precision of claims against source mischaracterized its own diff in its own
commit message. The commit is several back in the branch's history and is
not worth rewriting to fix prose in a message; recording it here is the
correction.

### 9.5 A sweep that returned zero was itself wrong, and nearly went unchecked

Checking whether any stale `docs/` citation remained inside the newly-moved
book pages, the controller ran:

```bash
grep -rn "docs/" docs/book | grep -v "docs/book"
```

and got zero hits, which was nearly reported as proof the book was clean.
The sweep was wrong, not the result: `grep -rn` prefixes every matching line
with its own filename, and every filename under `docs/book/` itself contains
the substring `docs/book`, so the second `grep -v` discarded every line the
first `grep` had found, including real hits. Re-running without the
self-defeating filter surfaced the genuine stale citation (fixed in commit
`cafc6d2`) and one false positive, a `docs/` segment inside a NASA URL
(`https://gmao.gsfc.nasa.gov/.../docs/yamazaki.pdf`) that is not a repository
path at all, which is why the verifier's own skip list starts with `http`.

This is the same failure shape as the defects this cleanup exists to find: a
check that looks correct and silently discards the evidence it exists to
surface, committed by the person checking for exactly that failure mode.
**Lesson: when a sweep returns zero, verify the sweep before trusting the
zero.**

### 9.6 Two smaller items, recorded briefly

**The 21-item pre-existing citation backlog.** Task 12 deliberately left 21
dead or convention-violating citations unfixed, named and measured rather
than silently absorbed into the "406 rewritten" figure: 9 unresolved paths
that never existed in this tree and predate this cleanup entirely
(`src/dag_algo/mod.rs`, `src/algo/mod.rs`, `scripts/build_subdivided_adjacency.py`
×2, `src/ddr/`, `scripts/train.py`, three `tests/routing/test_*.py` Python
mirrors), 9 "line citation into `src/` is not allowed" convention violations
in `src/` doc comments (a style issue, not a dead link), 1 wildcard-slug
placeholder for a campaign writeup that was never authored
(`research/findings/2026-07-1x-disagg-72h-window-findings.md`, cited from
`src/nn/disagg_head.rs:117`), and 2 `examples/*/README.md` references to a
nonexistent `extract_bundle.py`. All 21 are reproduced by the current
`python3 scripts/verify_doc_paths.py` run. A measured backlog someone chose
not to fix in this cleanup is a known quantity; an unexamined one is not.

**The five `*-handoff.md` documents split by accident.** `research/findings/`
holds three (`2026-06-07-checkpoint-resume-handoff.md`,
`2026-06-11-global-data-sources-handoff.md`,
`2026-07-01-leakance-hourly-experiment-handoff.md`) and `research/plans/`
holds two (`2026-06-06-gpu-device-config-handoff.md`,
`2026-06-06-sigfpe-wukong-debug-handoff.md`), all five of the same genre
(a mid-experiment handoff), split purely by which of `docs/` or
`.claude/specs/` each one started in, not by any ruling about where a
handoff document belongs. This is the one place the three-way
findings/specs/plans split does not explain itself: a reader hunting "the
SIGFPE handoff" by genre would check `research/findings/` first and miss it.
A renames-only task could not have fixed this without inventing a new
judgment call outside its scope, so it stands as a known wart rather than a
silently-accepted one.

---

## 10. Process findings, continued: five more lessons from Phase 2

Continuing the numbering from section 4 above (lessons 1 through 10 were
Phase 1's).

11. **A scope change can look like a quality improvement if you only read the
    summary number.** The citation gate's prose-warning count falling from
    roughly 516 to 80 reads as hygiene; it is almost entirely 135 documents
    leaving the scanned set. Any before/after count needs its denominator
    checked, not just its value. See section 9.1.

12. **An admission rule stated against a citing document, not a filename,
    is the whole point, and it only proves its worth when it overrides a
    plan that used filenames.** The sp8/sp10 scripts were listed for
    archiving by name-matching against "spike check"; both are live, one
    spawned directly by an `#[ignore]`d test whose failure mode would have
    been invisible to CI. See section 9.2.

13. **A fix made early in a multi-step restructure is not durable against the
    later steps.** `src/sparse/mod.rs:11` broke, was fixed, and broke again
    from an unrelated later move, and was only caught because an unrelated
    later gate-widening happened to re-scan it. The practice that generalizes:
    run the widest-coverage gate after every step, not just the step that
    seems related. See section 9.3.

14. **A commit message is a claim like any other, and needs the same
    verification.** "No logic changed" in `ec7014d`'s message is false for
    `scripts/journal.py`. The change itself was correct; the self-description
    of it was not checked before being written. See section 9.4.

15. **A sweep returning zero is a claim, not a proof, until the sweep itself
    is checked.** `grep -rn "docs/" docs/book | grep -v "docs/book"` returns
    zero by construction, not because the tree is clean: `grep -n` prefixes
    every line with a filename that itself contains the excluded string. See
    section 9.5.

---

## 11. `CLAUDE.md` line count, reconfirmed

`CLAUDE.md` is still **588** lines, unchanged since section 5's measurement
at the end of Phase 1. None of Tasks 10 through 14 touched `CLAUDE.md`; this
task's own edits (the "When in doubt" paths, the research-journal path, and
the doc-conventions table in `.claude/skills/ddrs-dev/references/research-status.md`)
changed text in place on existing lines rather than adding or removing any,
so the count held. Re-run `wc -l CLAUDE.md` rather than trust this sentence.

---

## 12. Final gate results (whole branch, 2026-09-12)

```
python3 scripts/test_verify_doc_paths.py        # 21/21 pass
python3 scripts/verify_doc_paths.py             # 0 unresolved in agent context, 80 in prose docs
cargo check --examples --tests                  # exit 0 (pre-existing warnings only)
cargo test --test ddr_sandbox_match --test gridded_bundle   # 6/6 pass
mdbook build                                    # clean
find target/book -type f | sort                 # differs from the pre-move snapshot by
                                                 #   exactly the 9 figures PNGs + images/.gitkeep;
                                                 #   all 18 .html outputs byte-identical
```

This is the state the branch's single pull request (Task 15, this task) is
opened against.

---

## 7. Reproduce

```bash
# the citation gate, now and its history
python3 scripts/test_verify_doc_paths.py
python3 scripts/verify_doc_paths.py   # expect: "0 unresolved in agent context, N in prose docs"

# CLAUDE.md's measured line count (re-run, do not trust a number in this doc)
wc -l CLAUDE.md

# the two citations that had drifted onto the wrong code before conversion (now fixed)
grep -n "setup_inputs" src/routing/mmc.rs | head -3
grep -n "struct RawRow" src/data/store/gage_csv.rs

# the duplicated-staleness mechanism: both now bare, no line number
grep -n "src/cli/run.rs\|src/bin/ddrs.rs" .claude/skills/ddrs-dev/SKILL.md CLAUDE.md

# the misfiled finding: confirm the real file is 261 lines, the claimed one 178
wc -l .claude/skills/ddrs-run/SKILL.md .claude/skills/ddrs-run/references/commands.md

# leak_probe.rs is a live citation, not leakance residue
grep -rn "leak_probe" src/ .claude/skills/

# the gate set this task ran
cargo check --examples --tests
mdbook build
cargo test --test ddr_sandbox_match --test gridded_bundle
```
