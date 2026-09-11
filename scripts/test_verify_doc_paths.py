#!/usr/bin/env python3
"""Tests for scripts/verify_doc_paths.py. Stdlib only:
    python3 scripts/test_verify_doc_paths.py

Covers the failure modes that would make the verifier worse than useless:
a false negative (misses a dead path, so a broken citation ships with a
green gate), and a false positive (flags a runtime path or a placeholder,
so the gate gets disabled as noisy).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

VERIFY = Path(__file__).resolve().parent / "verify_doc_paths.py"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def run_on(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VERIFY), "--root", str(root)],
        capture_output=True, text=True,
    )


def scaffold(tmp: Path, body: str) -> Path:
    """A minimal repo: one real source file, one markdown file citing things."""
    (tmp / "src").mkdir(parents=True)
    (tmp / "src" / "config.rs").write_text(
        "pub fn validate_grad_accum() {}\npub struct Params;\n"
    )
    (tmp / "src" / "sparse").mkdir()
    (tmp / "src" / "sparse" / "mod.rs").write_text("// solver\n")
    (tmp / "CLAUDE.md").write_text(body)
    return tmp


def main() -> int:
    # each case gets its own tree, so one failure cannot mask another
    cases = [
        ("live path resolves", "See `src/config.rs` for ranges.\n", 0, ""),
        ("dead path is caught", "See `src/sparse.rs` for the solver.\n", 1, "src/sparse.rs"),
        ("dead dir is caught", "Notes in `.claude/references/`.\n", 1, ".claude/references/"),
        ("live dir resolves", "Solver in `src/sparse/`.\n", 0, ""),
        ("line suffix stripped", "See `src/config.rs:9999`.\n", 0, ""),
        ("live symbol resolves", "See `src/config.rs::validate_grad_accum`.\n", 0, ""),
        ("dead symbol is caught", "See `src/config.rs::validate_nothing`.\n", 1, "validate_nothing"),
        ("runtime path skipped", "Writes `.ddrs/runs/x/manifest.json`.\n", 0, ""),
        ("placeholder skipped", "Path `.ddrs/runs/<id>/head.mpk` holds weights.\n", 0, ""),
        ("external tree skipped", "Read `~/projects/ddr/src/ddr/routing/mmc.py`.\n", 0, ""),
        ("url skipped", "See `https://example.com/a.md`.\n", 0, ""),
        ("markdown link, dead target is caught",
         "See [gone](src/missing.rs) for detail.\n", 1, "src/missing.rs"),
        ("markdown link, live target resolves",
         "See [config](src/config.rs) for detail.\n", 0, ""),
    ]
    for i, (name, body, want_code, want_token) in enumerate(cases):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / f"case{i}"
            root.mkdir()
            scaffold(root, body)
            proc = run_on(root)
            check(f"{name}: exit {want_code}", proc.returncode == want_code,
                  f"got {proc.returncode}, stdout={proc.stdout!r}")
            if want_token:
                check(f"{name}: names the token",
                      want_token in proc.stdout, f"stdout={proc.stdout!r}")

    # Pins the real interface: later tasks invoke the verifier with no
    # arguments from the repository root, which run_on() above never
    # exercises because it always passes --root.
    repo_root = VERIFY.resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(VERIFY)], cwd=repo_root,
        capture_output=True, text=True,
    )
    check("real tree, no args: exit 1", proc.returncode == 1,
          f"got {proc.returncode}, stdout={proc.stdout!r}")
    check("real tree, no args: reports agent-context count",
          "unresolved in agent context" in proc.stdout,
          f"stdout={proc.stdout!r}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
