#!/usr/bin/env python3
"""Research journal: record what we ran, what came out, and what we concluded.

Machine facts come from run/experiment manifests. The judgement fields
(Question / What we did / Result / Conclusion) are left as TODO for a human or
for Claude to fill in -- this script never invents them.

Modes (all read a Claude Code hook payload on stdin except backfill/status):

  --mode post-tool     after a Bash call: journal any finished, unjournaled run
  --mode session-start at session start: report unjournaled runs, write nothing
  --mode stop          at turn end: nag only about stubs this session created
  --mode backfill      one-off: ledger rows for every historical run
  --mode status        human-readable summary, no writes

Stdlib only, no git subprocess: hooks must be fast and must not perturb the
worktree they run in.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# A run with fewer than this many epochs, or any run capped with
# --max-mini-batches, is a mechanics smoke test: it gets a ledger row but no
# stub, because there is no question for it to answer.
SMOKE_EPOCHS = 5

LEDGER_START = "<!-- ledger:start -->"
LEDGER_END = "<!-- ledger:end -->"
ENTRIES_MARK = "<!-- entries -->"
TODO = "_TODO_"


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def repo_root() -> Path:
    """Nearest ancestor holding Cargo.toml (works in a worktree)."""
    override = os.environ.get("DDRS_JOURNAL_ROOT")
    if override:
        return Path(override).resolve()
    for d in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (d / "Cargo.toml").is_file():
            return d
    return Path(__file__).resolve().parent.parent


def main_tree(root: Path) -> Path:
    """The primary checkout. In a worktree, .git is a file pointing into it."""
    dot = root / ".git"
    if dot.is_file():
        try:
            line = dot.read_text(encoding="utf-8").strip()
        except OSError:
            return root
        if line.startswith("gitdir:"):
            gitdir = line.split(":", 1)[1].strip()
            marker = "/.git/worktrees/"
            if marker in gitdir:
                return Path(gitdir.split(marker)[0])
    return root


def workspaces(root: Path) -> list[Path]:
    """Every .ddrs this repo might write runs into, nearest first."""
    out, seen = [], set()
    for cand in (root / ".ddrs", main_tree(root) / ".ddrs"):
        rp = cand.resolve()
        if cand.is_dir() and rp not in seen:
            seen.add(rp)
            out.append(cand)
    return out


def journal_dir(root: Path) -> Path:
    return root / "research" / "journal"


def state_path(root: Path) -> Path:
    """Gitignored: which stubs this session opened, so --mode stop is precise."""
    return root / ".ddrs" / "journal-state.json"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
class Record:
    def __init__(self, rid, kind, finished, status, git, facts, metrics, smoke, path):
        self.id = rid
        self.kind = kind          # "run" | "experiment"
        self.finished = finished  # str, UTC
        self.status = status
        self.git = git            # dict sha/branch/dirty
        self.facts = facts        # list[str] auto bullets
        self.metrics = metrics    # short one-line result summary
        self.smoke = smoke
        self.path = path

    def sort_key(self):
        return (self.finished or "", self.id)


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fmt_secs(s) -> str:
    if not isinstance(s, (int, float)) or s <= 0:
        return ""
    m = s / 60.0
    return f"{m:.0f} min" if m >= 1 else f"{s:.0f} s"


def _short(sha) -> str:
    return (sha or "")[:7] or "?"


def read_run(mp: Path) -> Record | None:
    d = _load(mp)
    if not d or not d.get("finished_at"):
        return None
    m = d.get("metrics") or {}
    git = d.get("git") or {}
    epochs = m.get("epochs_completed")
    smoke = d.get("max_mini_batches") is not None or (
        isinstance(epochs, int) and epochs < SMOKE_EPOCHS
    )

    nse, kge = m.get("median_nse_finite"), m.get("median_kge_finite")
    parts = []
    if isinstance(nse, (int, float)):
        parts.append(f"NSE {nse:.4f}")
    if isinstance(kge, (int, float)):
        parts.append(f"KGE {kge:.4f}")
    if parts:
        summary = " / ".join(parts)
    elif d.get("status") != "ok":
        summary = f"status `{d.get('status')}`"
    else:
        summary = "trained, no eval metrics"

    facts = [f"workflow `{d.get('workflow')}`, status `{d.get('status')}`"]
    if d.get("exit_reason"):
        facts.append(f"exit reason: {d['exit_reason']}")
    dirty = " **dirty**" if git.get("dirty") else " clean"
    facts.append(f"git `{_short(git.get('sha'))}`{dirty} on `{git.get('branch')}`")

    shape = []
    if isinstance(epochs, int):
        shape.append(f"{epochs} epochs")
    ng = m.get("n_gauges_finite_nse")
    nt = m.get("n_gauges_total")
    if isinstance(ng, int):
        shape.append(f"{ng:,} gauges" + (f" of {nt:,}" if isinstance(nt, int) and nt != ng else ""))
    t1, t2 = _fmt_secs(m.get("phase1_seconds")), _fmt_secs(m.get("phase2_seconds"))
    if t1 or t2:
        shape.append("train " + (t1 or "-") + " / eval " + (t2 or "-"))
    if shape:
        facts.append(", ".join(shape))
    if isinstance(nse, (int, float)):
        mean = m.get("mean_nse_finite")
        line = f"median NSE **{nse:.4f}**"
        if isinstance(kge, (int, float)):
            line += f", median KGE **{kge:.4f}**"
        if isinstance(mean, (int, float)):
            line += f" (mean NSE {mean:.4f})"
        facts.append(line)
    if d.get("source_lock", {}).get("drift"):
        facts.append("⚠ source drift recorded against the lockfile")

    return Record(d.get("run_id") or mp.parent.name, "run", d["finished_at"],
                  d.get("status"), git, facts, summary, smoke, mp.parent)


def read_experiment(mp: Path) -> Record | None:
    d = _load(mp)
    if not d or not d.get("finished_utc"):
        return None
    git = d.get("git") or {}
    study = d.get("study") or "?"
    # A shard is one slice of a study, not a result. Merged output is the result.
    smoke = d.get("shard") is not None
    rid = f"{mp.parent.parent.name}/{mp.parent.name}"

    arms = d.get("arms") or []
    facts = [f"study `{study}`, status `{d.get('status')}`, backend `{d.get('backend')}`"]
    dirty = " **dirty**" if git.get("dirty") else " clean"
    facts.append(f"git `{_short(git.get('sha'))}`{dirty} on `{git.get('branch')}`")
    if arms:
        names = ", ".join(f"`{a.get('name')}`" for a in arms[:4])
        more = f" (+{len(arms) - 4} more)" if len(arms) > 4 else ""
        facts.append(f"{len(arms)} arm(s): {names}{more}")
        for a in arms[:4]:
            if a.get("run_id"):
                facts.append(f"arm `{a.get('name')}` ← run `{a['run_id']}` @ `{a.get('checkpoint', '?')}`")
    if d.get("shards"):
        facts.append(f"merged from {len(d['shards'])} shards")
    for n in (d.get("notes") or [])[:5]:
        facts.append(f"note: {n}")
    if d.get("bundle_dir"):
        facts.append(f"bundle `{d['bundle_dir']}`")

    return Record(rid, "experiment", d["finished_utc"], d.get("status"), git,
                  facts, f"{len(arms)} arm(s), status `{d.get('status')}`",
                  smoke, mp.parent)


def scan(root: Path) -> list[Record]:
    recs, seen = [], set()
    for ws in workspaces(root):
        for mp in sorted(ws.glob("runs/*/manifest.json")):
            r = read_run(mp)
            if r and r.id not in seen:
                seen.add(r.id)
                recs.append(r)
        for mp in sorted(ws.glob("experiments/*/*/manifest.json")):
            r = read_experiment(mp)
            if r and r.id not in seen:
                seen.add(r.id)
                recs.append(r)
    recs.sort(key=Record.sort_key)
    return recs


# --------------------------------------------------------------------------
# Journal files
# --------------------------------------------------------------------------
def month_of(finished: str) -> str:
    m = re.match(r"(\d{4})-(\d{2})", finished or "")
    return f"{m.group(1)}-{m.group(2)}" if m else datetime.now(timezone.utc).strftime("%Y-%m")


LEDGER_ID = re.compile(r"^\|[^|]*\|\s*`([^`]+)`\s*\|", re.M)

# Only YYYY-MM.md files are data. README.md and anything else in the directory
# is prose and must never be parsed as ledger rows or entries.
MONTH_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9].md"


def recorded_ids(root: Path) -> set[str]:
    """Ids already in the journal, from ledger rows *and* entry markers.

    A ledger-only record has no `<!-- id -->` marker, so both sources are
    needed or every backfill duplicates the whole table.
    """
    ids = set()
    jd = journal_dir(root)
    if not jd.is_dir():
        return ids
    for f in jd.glob(MONTH_GLOB):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        ids.update(re.findall(r"<!-- id:([^\s>]+) -->", text))
        ids.update(LEDGER_ID.findall(text))
    return ids


def month_file(root: Path, month: str) -> Path:
    return journal_dir(root) / f"{month}.md"


def ensure_month(root: Path, month: str) -> Path:
    p = month_file(root, month)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"# Research journal — {month}\n\n"
        "Facts are written automatically from run manifests by `scripts/journal.py`.\n"
        "The Question / What we did / Result / Conclusion fields are judgement and are\n"
        "written by hand. See `.claude/skills/ddrs-journal/SKILL.md` for what makes a\n"
        "good entry. Append-only: on a merge conflict, keep both sides.\n\n"
        "## Ledger\n\n"
        "Every completed run, including smoke tests, so nothing is lost.\n\n"
        f"{LEDGER_START}\n"
        "| finished (UTC) | id | kind | result | git |\n"
        "|---|---|---|---|---|\n"
        f"{LEDGER_END}\n\n"
        "## Entries\n\n"
        f"{ENTRIES_MARK}\n",
        encoding="utf-8",
    )
    return p


def when_utc(finished: str) -> str:
    """Run ids use `T02-03-02Z`, run manifests use `T02:03:02Z`. Normalise."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})[T ](\d{2})[:-](\d{2})", finished or "")
    return f"{m.group(1)} {m.group(2)}:{m.group(3)}" if m else (finished or "?")


def ledger_row(r: Record) -> str:
    when = when_utc(r.finished)
    dirty = "+dirty" if r.git.get("dirty") else ""
    return (f"| {when} | `{r.id}` | {r.kind} | {r.metrics} | "
            f"`{_short(r.git.get('sha'))}`{dirty} |")


def rel_path(root: Path, p: Path) -> str:
    """Repo-relative where possible: absolute home paths in a committed doc are
    noise, and the journal is read from other checkouts."""
    for base in (root, main_tree(root)):
        try:
            return str(Path(p).resolve().relative_to(base.resolve()))
        except ValueError:
            continue
    return str(p)


def stub(root: Path, r: Record) -> str:
    when = (r.finished or "")[:10]
    facts = "\n".join(f"- {f}" for f in r.facts)
    return (
        f"\n---\n\n"
        f"### {when} · `{r.id}` <!-- id:{r.id} -->\n\n"
        f"**Facts** (auto)\n\n{facts}\n"
        f"- artifacts: `{rel_path(root, r.path)}`\n\n"
        f"**Question:** {TODO} — what were we trying to find out, stated so it could come out \"no\"\n\n"
        f"**What we did:** {TODO} — 2–4 bullets: the change under test, and what it was held against\n\n"
        f"**Result:** {TODO} — the number, next to the number it must beat\n\n"
        f"**Conclusion:** {TODO} — supported / refuted / inconclusive, and what it changes next\n"
    )


def append(root: Path, recs: list[Record], stubs: bool = True) -> list[tuple[Record, bool]]:
    """Write ledger rows for all; stubs for non-smoke. Returns (rec, got_stub).

    `stubs=False` is for backfill: the judgement behind a months-old run is not
    recoverable, and an entry nobody can honestly fill is worse than no entry.
    """
    out = []
    by_month: dict[str, list[Record]] = {}
    for r in recs:
        by_month.setdefault(month_of(r.finished), []).append(r)

    for month, rs in sorted(by_month.items()):
        p = ensure_month(root, month)
        text = p.read_text(encoding="utf-8")
        rows = "".join(ledger_row(r) + "\n" for r in rs)
        if LEDGER_END in text:
            text = text.replace(LEDGER_END, rows + LEDGER_END, 1)
        else:  # file was hand-edited past recognition; never lose the data
            text += "\n" + rows
        for r in rs:
            want = stubs and not r.smoke
            if want:
                text += stub(root, r)
            out.append((r, want))
        p.write_text(text, encoding="utf-8")
    return out


def open_todos(root: Path) -> list[str]:
    """Entry ids whose judgement fields are still unfilled."""
    ids = []
    for f in sorted(journal_dir(root).glob(MONTH_GLOB)) if journal_dir(root).is_dir() else []:
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for block in text.split("\n---\n"):
            m = re.search(r"<!-- id:([^\s>]+) -->", block)
            if m and TODO in block:
                ids.append(m.group(1))
    return ids


# --------------------------------------------------------------------------
# Session state (gitignored) -- so --mode stop only nags about our own stubs
# --------------------------------------------------------------------------
def load_state(root: Path) -> dict:
    return _load(state_path(root)) or {}


def save_state(root: Path, st: dict) -> None:
    p = state_path(root)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(st, indent=1), encoding="utf-8")
    except OSError:
        pass


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def emit(event: str, context: str) -> None:
    json.dump({"hookSpecificOutput": {"hookEventName": event,
                                      "additionalContext": context}}, sys.stdout)
    sys.stdout.write("\n")


TRIGGER = re.compile(r"\bddrs\b[^|;&]*\b(run|experiment)\b|--bin\s+(train|train_and_test|eval)\b")


def mode_post_tool(root: Path, payload: dict) -> int:
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not TRIGGER.search(cmd):
        return 0
    pending = [r for r in scan(root) if r.id not in recorded_ids(root)]
    if not pending:
        return 0
    written = append(root, pending)

    st = load_state(root)
    sid = payload.get("session_id") or "?"
    mine = st.setdefault("stubs_by_session", {}).setdefault(sid, [])
    mine.extend(r.id for r, got in written if got)
    save_state(root, st)

    stubs = [r.id for r, got in written if got]
    lines = [f"Research journal: recorded {len(written)} completed run(s) in research/journal/."]
    if stubs:
        lines.append("These need the judgement fields written before this turn ends "
                     "(Question / What we did / Result / Conclusion):")
        lines += [f"  - {i}" for i in stubs]
        lines.append("Fill them from what you actually know. If you do not know why the "
                     "run was made, say so in the entry rather than inventing a hypothesis.")
    else:
        lines.append("All were smoke tests: ledger rows only, no entry needed.")
    emit("PostToolUse", "\n".join(lines))
    return 0


def mode_session_start(root: Path, payload: dict) -> int:
    done = recorded_ids(root)
    pending = [r for r in scan(root) if r.id not in done and not r.smoke]
    todos = open_todos(root)
    if not pending and not todos:
        return 0
    lines = []
    if pending:
        lines.append(f"Research journal: {len(pending)} finished run(s) are not in the journal "
                     "(likely run outside a Claude session):")
        lines += [f"  - {r.id} — {r.metrics}" for r in pending[-8:]]
        lines.append("Run `python3 scripts/journal.py --mode backfill` to record them, "
                     "then write the judgement fields for any that answered a question.")
    if todos:
        lines.append(f"{len(todos)} journal entr(ies) still have TODO judgement fields: "
                     + ", ".join(todos[-6:]))
    emit("SessionStart", "\n".join(lines))
    return 0


def mode_stop(root: Path, payload: dict) -> int:
    if payload.get("stop_hook_active"):
        return 0  # already nagged once this turn; never loop
    sid = payload.get("session_id") or "?"
    st = load_state(root)
    mine = set((st.get("stubs_by_session") or {}).get(sid) or [])
    if not mine:
        return 0
    unfilled = [i for i in open_todos(root) if i in mine]
    if not unfilled:
        return 0
    json.dump({"decision": "block", "reason":
               "These journal entries were opened by runs in this session and still have "
               "TODO judgement fields: " + ", ".join(unfilled) +
               ". Fill Question / What we did / Result / Conclusion in research/journal/ now. "
               "Write what you actually know; if the run's purpose is unclear to you, "
               "record that plainly instead of inventing one."}, sys.stdout)
    sys.stdout.write("\n")
    return 0


def mode_backfill(root: Path, _payload: dict) -> int:
    """Ledger rows for history. Never stubs -- see append()'s docstring.

    Pass --with-stubs to open entries too; only sensible for runs recent enough
    that someone still remembers why they were made.
    """
    pending = [r for r in scan(root) if r.id not in recorded_ids(root)]
    if not pending:
        print("Journal is up to date.")
        return 0
    with_stubs = "--with-stubs" in sys.argv
    written = append(root, pending, stubs=with_stubs)
    stubs = [r.id for r, got in written if got]
    print(f"Recorded {len(written)} run(s) in the ledger"
          + (f"; {len(stubs)} opened as entries needing judgement." if stubs else
             " (ledger only -- rerun with --with-stubs to open entries)."))
    for i in stubs:
        print(f"  {i}")
    return 0


def mode_status(root: Path, _payload: dict) -> int:
    recs = scan(root)
    done = recorded_ids(root)
    todos = open_todos(root)
    print(f"workspaces : {', '.join(str(w) for w in workspaces(root))}")
    print(f"journal    : {journal_dir(root)}")
    print(f"runs found : {len(recs)}  ({sum(1 for r in recs if r.smoke)} smoke)")
    print(f"recorded   : {len(done)}")
    print(f"unrecorded : {sum(1 for r in recs if r.id not in done)}")
    print(f"open TODOs : {len(todos)}")
    for i in todos:
        print(f"  {i}")
    return 0


MODES = {"post-tool": mode_post_tool, "session-start": mode_session_start,
         "stop": mode_stop, "backfill": mode_backfill, "status": mode_status}


def main() -> int:
    mode = "status"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    if mode not in MODES:
        print(f"unknown mode {mode!r}; pick one of {', '.join(MODES)}", file=sys.stderr)
        return 2
    payload = {}
    if mode in ("post-tool", "session-start", "stop") and not sys.stdin.isatty():
        try:
            payload = json.loads(sys.stdin.read() or "{}")
        except ValueError:
            payload = {}
    root = repo_root()
    if os.environ.get("DDRS_JOURNAL_OFF"):
        return 0
    return MODES[mode](root, payload)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # a journal must never break the thing it observes
        print(f"journal.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(0)
