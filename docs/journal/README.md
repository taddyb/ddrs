# Research journal

What we tried, what came out, and what we concluded — one file per month,
append-only, newest at the bottom.

Each month has two parts:

- **Ledger** — one row per completed run or experiment, including smoke tests
  and shards, so no run is ever lost. Written automatically from
  `.ddrs/runs/<id>/manifest.json` and `.ddrs/experiments/<name>/<ts>/manifest.json`.
- **Entries** — the four judgement fields (Question, What we did, Result,
  Conclusion) for runs that were actually answering something. Written by hand.

`scripts/journal.py` writes the facts; it never writes judgement. A stub with
`_TODO_` fields is an open question, not a finished entry.

```bash
python3 scripts/journal.py --mode status     # what is pending
python3 scripts/journal.py --mode backfill   # ledger rows for anything missed
python3 scripts/test_journal.py              # stdlib only, no deps
```

Hooks in `.claude/settings.json` run it after `ddrs run` / `ddrs experiment`, at
session start, and at turn end. `DDRS_JOURNAL_OFF=1` disables them for one
command.

Entries from before 2026-09-10 are ledger-only: the journal was introduced then,
and the reasoning behind earlier runs is not recoverable. It was not invented.

Rules for writing an entry: `.claude/skills/ddrs-journal/SKILL.md`.
Long-form results still get their own `docs/YYYY-MM-DD-<topic>-findings.md`; the
journal entry links to it rather than repeating it.

Not in `SUMMARY.md` on purpose — this is a working notebook, not published
documentation.
