#!/usr/bin/env python3
"""Tests for scripts/journal.py. Stdlib only:  python3 scripts/test_journal.py

Covers the behaviours that would silently corrupt the research record:
idempotency, smoke-test classification, that judgement fields are never
invented, and that the Stop nag is scoped to the session that caused it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

JOURNAL = Path(__file__).resolve().parent / "journal.py"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def run(root: Path, mode: str, payload=None, extra=()):
    env = {**os.environ, "DDRS_JOURNAL_ROOT": str(root)}
    p = subprocess.run(
        [sys.executable, str(JOURNAL), "--mode", mode, *extra],
        input=json.dumps(payload or {}), capture_output=True, text=True, env=env)
    return p


def write_run(root: Path, rid: str, *, finished: str, epochs=30, nse=0.72,
              kge=0.75, status="ok", max_mb=None):
    d = root / ".ddrs" / "runs" / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "run_id": rid, "workflow": "train-and-test", "status": status,
        "finished_at": finished, "max_mini_batches": max_mb,
        "git": {"sha": "abc1234def", "dirty": False, "branch": "master"},
        "metrics": {"epochs_completed": epochs, "median_nse_finite": nse,
                    "median_kge_finite": kge, "n_gauges_finite_nse": 2365,
                    "phase1_seconds": 3888.0, "phase2_seconds": 2019.0},
    }))


def write_experiment(root: Path, study: str, ts: str, *, finished: str, shard=None):
    d = root / ".ddrs" / "experiments" / study / ts
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "study": study, "status": "success", "finished_utc": finished,
        "backend": "cpu", "shard": shard, "notes": [],
        "git": {"sha": "feed1234", "dirty": True, "branch": "master"},
        "arms": [{"name": "a1", "run_id": "r1", "checkpoint": "epoch_30_mb_1"}],
    }))


def journal_text(root: Path) -> str:
    jd = root / "docs" / "journal"
    return "".join(f.read_text() for f in sorted(jd.glob("*.md"))) if jd.is_dir() else ""


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "Cargo.toml").write_text("[package]\n")

        print("scan + classification")
        write_run(root, "2026-09-01T00-00-00Z-train-and-test", finished="2026-09-01T02:00:00Z")
        write_run(root, "2026-09-02T00-00-00Z-smoke", finished="2026-09-02T02:00:00Z",
                  epochs=2, max_mb=2)
        write_run(root, "2026-08-30T00-00-00Z-train-and-test", finished="2026-08-30T02:00:00Z")
        write_experiment(root, "landscape", "2026-09-03T00-00-00Z", finished="2026-09-03T01-00-00Z")
        write_experiment(root, "landscape", "2026-09-03T00-00-01Z",
                         finished="2026-09-03T01-00-01Z", shard="0-of-4")
        out = run(root, "status").stdout
        check("finds every record", "runs found : 5" in out, out)
        check("smoke + shard classified as smoke", "(2 smoke)" in out, out)

        print("backfill writes ledger only, never judgement")
        run(root, "backfill")
        text = journal_text(root)
        check("ledger rows written", text.count("| 2026-") == 5, str(text.count("| 2026-")))
        check("no invented judgement", "_TODO_" not in text)
        check("months split", (root / "docs/journal/2026-09.md").exists()
              and (root / "docs/journal/2026-08.md").exists())

        print("idempotency")
        run(root, "backfill")
        run(root, "backfill")
        text = journal_text(root)
        check("no duplicate rows after 3 backfills", text.count("| 2026-") == 5,
              str(text.count("| 2026-")))

        print("post-tool: trigger matching")
        write_run(root, "2026-09-05T00-00-00Z-train-and-test", finished="2026-09-05T02:00:00Z")
        p = run(root, "post-tool", {"session_id": "S1", "tool_input": {"command": "ls -la"}})
        check("ignores unrelated commands", p.stdout.strip() == "", p.stdout)
        p = run(root, "post-tool",
                {"session_id": "S1", "tool_input": {"command": "ddrs run --workflow train"}})
        check("fires on ddrs run", "Research journal" in p.stdout, p.stdout[:200])
        text = journal_text(root)
        check("opens a stub for a real run", "_TODO_" in text)
        check("stub carries an id marker", "<!-- id:2026-09-05T00-00-00Z-train-and-test -->" in text)

        print("post-tool: smoke runs get no stub")
        write_run(root, "2026-09-06T00-00-00Z-smoke", finished="2026-09-06T02:00:00Z",
                  epochs=1, max_mb=2)
        p = run(root, "post-tool",
                {"session_id": "S1", "tool_input": {"command": "ddrs run --max-mini-batches 2"}})
        check("smoke says so", "smoke tests" in p.stdout, p.stdout[:200])
        check("smoke has no stub", "2026-09-06T00-00-00Z-smoke -->" not in journal_text(root))

        print("stop: scoped to the session that opened the stub")
        p = run(root, "stop", {"session_id": "S2"})
        check("silent for a session that opened nothing", p.stdout.strip() == "", p.stdout)
        p = run(root, "stop", {"session_id": "S1"})
        check("blocks the session that did", '"decision": "block"' in p.stdout, p.stdout[:200])
        p = run(root, "stop", {"session_id": "S1", "stop_hook_active": True})
        check("never loops", p.stdout.strip() == "", p.stdout)

        print("stop: goes quiet once judgement is written")
        f = root / "docs/journal/2026-09.md"
        f.write_text(f.read_text().replace("_TODO_", "answered"))
        p = run(root, "stop", {"session_id": "S1"})
        check("quiet after fields filled", p.stdout.strip() == "", p.stdout)

        print("session-start reports work done outside a session")
        write_run(root, "2026-09-07T00-00-00Z-train-and-test", finished="2026-09-07T02:00:00Z")
        p = run(root, "session-start", {"session_id": "S3"})
        check("reports unjournaled run", "2026-09-07T00-00-00Z" in p.stdout, p.stdout[:300])
        check("writes nothing itself", "2026-09-07T00-00-00Z" not in journal_text(root))

        print("robustness")
        bad = root / ".ddrs" / "runs" / "corrupt"
        bad.mkdir(parents=True, exist_ok=True)
        (bad / "manifest.json").write_text("{not json")
        p = run(root, "status")
        check("survives a corrupt manifest", p.returncode == 0, p.stderr[:200])
        env = {**os.environ, "DDRS_JOURNAL_ROOT": str(root), "DDRS_JOURNAL_OFF": "1"}
        p = subprocess.run([sys.executable, str(JOURNAL), "--mode", "post-tool"],
                           input='{"tool_input":{"command":"ddrs run"}}',
                           capture_output=True, text=True, env=env)
        check("DDRS_JOURNAL_OFF disables it", p.stdout.strip() == "", p.stdout)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all journal tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
