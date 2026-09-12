---
name: ddrs-journal
description: Use when a ddrs run or experiment has finished, when a journal entry in docs/journal/ has unfilled TODO fields, or when recording what a training run or experiment concluded.
---

# ddrs research journal

`docs/journal/YYYY-MM.md` is the running record of what we tried and what came of <!-- verify-doc-paths: ignore -->
it. Facts are written for you by `scripts/journal.py` (hooked on `ddrs run` and
`ddrs experiment`). Judgement is yours.

## What fires, and when

| Event | What happens |
|---|---|
| A Bash call matching `ddrs run` / `ddrs experiment` returns | Any finished, unjournaled run gets a ledger row; non-smoke runs also get an entry stub. You are told which. |
| Your turn ends with a stub **this session** opened still unfilled | The Stop hook blocks and names it. |
| A session starts | Runs finished outside a session are reported. Nothing is written. |

Smoke tests (`--max-mini-batches`, fewer than 5 epochs) and experiment *shards*
get a ledger row only. There is no question for them to answer.

Manual: `python3 scripts/journal.py --mode status` (what is pending),
`--mode backfill` (ledger rows for history), `--mode backfill --with-stubs`
(also open entries — only for runs recent enough that you remember why).
`DDRS_JOURNAL_OFF=1` disables the hooks for one command.

## Filling an entry

Four fields. Keep the whole entry under ~15 lines; if the conclusion needs more,
it has earned a `docs/YYYY-MM-DD-<topic>-findings.md` and the entry links to it.

**Question** — what we wanted to find out, phrased so "no" is a possible answer.

- Good: "Does `nse-batch` + AdaDelta beat L1 on the 2,365-gauge CONUS set?"
- Bad: "Improve NSE." That cannot come out false, so it cannot be answered.

**What we did** — 2 to 4 bullets: the one thing under test, and what it was held
against. Name the arm it is being compared to. If nothing changed but the seed,
say that; seed spread is a result.

**Result** — the number *next to the number it must beat*. A bare "NSE 0.72"
is not a result. The comparison is the summed-Q' baseline for the **same gauge
population** (`references/research-status.md` in `ddrs-dev` has the right bars;
the CONUS one is 0.6781 NSE / 0.7172 KGE on 2,365 gauges). Record the metric the
experiment was designed to move, not the one that happens to look best.

**Conclusion** — supported, refuted, or inconclusive, and what changes next.
"Inconclusive" is a real and frequent answer; write it rather than rounding a
null result up into a weak positive.

## Rules

1. **Never invent a Question or Conclusion.** If you did not run the experiment
   and cannot tell why it was run, write `unknown — run predates this entry` and
   leave it. A journal that contains one confident guess is no longer evidence.
2. **A NO-GO is a result and gets a full entry.** The most valuable thing in this
   repo is the leakance non-identifiability finding, which closed a line of work.
   Failed and aborted runs get entries too: record the failure mode.
3. **Flag a dirty tree.** The facts block marks `**dirty**` when the run's git SHA
   did not describe the working tree. That run is not reproducible; say so in the
   conclusion rather than quietly citing its number later.
4. **Watch for the stale-binary trap.** A run whose SHA looks current can still
   have executed an old `~/.cargo/bin/ddrs`. Before trusting a surprising result,
   check the checkpoint layout (see `ddrs-dev` trap T1) and note what you checked.
5. **Link, don't duplicate.** If a findings doc, spec, or figure directory exists,
   point at it. The journal is the index of the research, not a second copy of it.
6. **Append only.** Never rewrite a past entry's conclusion. If it turns out to be
   wrong, write a new entry saying so and link back. The record of having been
   wrong is part of the research.

## What this is not

Not a changelog (that is git), not a run log (that is `.ddrs/runs/<id>/run.log`),
and not a substitute for updating `ddrs-dev` when a run establishes a durable
fact. CLAUDE.md's rule still holds: knowledge from a completed run belongs in the
skill library in the same session that produced it. The journal records *that we
learned it and when*; the skill records *what to do about it*.
