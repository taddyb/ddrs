# Research journal — design

**Date:** 2026-09-10
**Status:** implemented
**Problem:** 89 runs and ~20 experiment bundles exist under `.ddrs/`, and the
reasoning behind almost all of them lives only in conversation history, which is
compacted away. CLAUDE.md already requires that knowledge from a completed run be
committed rather than left in a findings doc or a transcript, but nothing enforced
it and nothing made it cheap.

## 1. What this is

A per-month, append-only notebook at `research/journal/YYYY-MM.md` with two parts:

- **Ledger** — one row per completed run or experiment. Machine-written.
- **Entries** — Question / What we did / Result / Conclusion. Human-written.

The split is the whole design. Everything a manifest knows is written for you;
everything it cannot know is left explicitly blank. A `_TODO_` field is an open
question, not a formatting defect.

## 2. Dataflow

```
   .ddrs/runs/<id>/manifest.json            .ddrs/experiments/<name>/<ts>/manifest.json
   run_id, workflow, status, git,           study, status, backend, git, arms,
   finished_at, max_mini_batches,           shard, shards, notes, finished_utc
   metrics{epochs, median_nse,
           median_kge, n_gauges, secs}
            |                                              |
            +----------------------+-----------------------+
                                   |
                         scripts/journal.py  scan()
                                   |
                    +--------------+--------------+
                    |                             |
              smoke test?                   real experiment?
         (--max-mini-batches set,          (everything else)
          epochs < 5, or a shard)                 |
                    |                             |
            ledger row only            ledger row  +  entry stub
                    |                             |         _TODO_ x4
                    +--------------+--------------+
                                   |
                     research/journal/YYYY-MM.md   (git-tracked)
                                   |
                                   |  <-- judgement written here, by hand
                                   v
                      Question / What we did / Result / Conclusion
```

Trigger surface:

```
  Bash call returns, cmd matches /ddrs (run|experiment)/  --> post-tool
        writes rows + stubs, tells Claude which need filling

  turn ends, a stub THIS session opened is still _TODO_   --> stop
        blocks once (stop_hook_active guard), names the ids

  session starts                                          --> session-start
        reports finished-but-unjournaled runs, WRITES NOTHING
```

## 3. Why a hook rather than a rule

A CLAUDE.md rule saying "write a journal entry after every run" is the design that
already failed: the existing findings docs exist for campaigns someone remembered
to write up, and nothing for the other 200 runs. The hook makes the factual half
unconditional and independent of whether the session compacts, which agent ran the
command, or whether anyone remembers.

Rejected alternatives:

- **`ddrs journal` Rust subcommand.** Most robust, but it puts documentation
  plumbing in the routing crate and buys a permanent tier-C gate on a concern that
  has nothing to do with the solver.
- **One file per entry.** Never merge-conflicts and matches the `docs/*-findings.md`
  convention, but loses the read-it-straight-through quality that makes a journal
  worth keeping. Monthly rotation is the compromise: conflicts are bounded and
  always resolved by taking both sides.
- **Full entry for every run.** At the observed run rate most entries would read
  "smoke test, mechanics OK, no conclusion" and bury the real findings.

## 4. Blast radius

| Touched | Risk | Mitigation |
|---|---|---|
| `.claude/settings.json` (new) | Hooks fire on **every** Bash call in the repo | Regex-gated to `ddrs run`/`ddrs experiment`; returns in ~30 ms otherwise; `DDRS_JOURNAL_OFF=1` kill switch; top-level `except` exits 0 so a journal bug can never fail the command it observes |
| `scripts/journal.py` (new) | Could corrupt the record it exists to keep | `scripts/test_journal.py` covers idempotency, smoke classification, and that judgement is never auto-written |
| `research/journal/**` (new) | Merge conflicts across parallel worktrees | Append-only, monthly, `---`-separated; conflict resolution is always "keep both" |
| `.ddrs/journal-state.json` (new) | none — gitignored under the existing `.ddrs/` rule | — |
| `CLAUDE.md`, `docs/SUMMARY.md` | Journal deliberately **not** added to `SUMMARY.md` | It is a working notebook; mdBook publishes reference docs |
| `src/**`, `Cargo.toml`, CI | **Untouched.** No Rust, no build, no gate. | Invariants 1–7 are not in scope |

## 5. Assumptions

- **The journal is git-tracked.** CLAUDE.md's reproducibility rule requires
  research knowledge to be committed; a gitignored journal would repeat the
  failure it exists to fix.
- **A NO-GO earns a full entry.** The leakance non-identifiability result is the
  repo's most valuable finding and it is a negative one. Failed and aborted runs
  get entries recording the failure mode.
- **Smoke tests have no question.** `--max-mini-batches` runs and experiment
  shards are mechanics checks and slices, not results. They keep a ledger row so
  provenance is complete, and nothing more.
- **History cannot be backfilled with judgement.** `--mode backfill` writes ledger
  rows only. The 224 pre-existing records got provenance, not invented hypotheses.

## 6. Known gaps

1. **A hook only sees commands Claude runs.** `ddrs run` typed into a terminal
   produces no `PostToolUse`. Caught at the next session start instead, so capture
   is eventual rather than immediate.
2. **Long runs are backgrounded.** `PostToolUse` fires when the tool call returns,
   which for a backgrounded run is before the manifest exists. Handled by scanning
   for *any* finished-and-unjournaled run on every matching command rather than
   only the run that command started, plus the `Stop` hook.
3. **The Stop hook can only nag the session that caused the stub.** By design: a
   session with no context on a run would otherwise be pressured into inventing a
   hypothesis, which is the one failure mode that makes the record worthless.
4. **Experiment ledger rows are thin** (`N arm(s), status`) because experiment
   manifests carry no scalar result. Their results live in `figures/`.

## 7. Outcome

Backfilled 224 historical records into ledger rows across `2026-06.md` through
`2026-09.md`; no judgement fields invented. First real entry written by hand for
the gridded DDM30 port. `scripts/test_journal.py`: all pass.
