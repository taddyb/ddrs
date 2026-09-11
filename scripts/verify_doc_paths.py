#!/usr/bin/env python3
"""Verify that every citation in agent-loaded context resolves.

CLAUDE.md and .claude/skills/** are loaded into an agent's context, so a
dead citation there is the most expensive kind of wrong line in the repo.
This script extracts every backtick-quoted path, `file:line` citation,
markdown link target and `file.rs::symbol` citation from those files and
checks that the path exists and that every `file.rs::symbol` citation
names a real item. It deliberately does not validate the `:line` number
itself: a citation that drifted from line 1066 to line 1202 still points
at a line that exists, so no existence check can catch that drift. The
remedy is the symbol-citation policy, which Task 6 will make this script
enforce by refusing `src/` line citations outright. Book pages under docs/
are scanned in warn mode, because their line citations are a reading aid
rather than a contract.

    python3 scripts/verify_doc_paths.py          # strict, exit 1 on failure
    python3 scripts/verify_doc_paths.py --list    # print every citation found
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Agent context: a dead citation here fails the run.
STRICT_GLOBS = (
    "CLAUDE.md",
    ".claude/skills/**/*.md",
    ".claude/ARCHITECTURE.md",
    ".claude/PHYSICS-CORRECTIONS.md",
    ".claude/REACH-SUBDIVISION.md",
)
# A line carrying this marker is exempt: it cites a DDR-side path, a template,
# or a path that deliberately does not exist (a trap describing a wrong path).
IGNORE_MARKER = "verify-doc-paths: ignore"
# Prose documentation: reported, never fatal.
WARN_GLOBS = ("docs/**/*.md", "README.md")

FILE_EXT = r"rs|py|md|ya?ml|toml|sh|json|nc|ipynb|dbf|shp|gpkg|mpk|zarr|ic|csv"
PATH_RE = re.compile(rf"`([A-Za-z0-9_.][A-Za-z0-9_./-]*\.(?:{FILE_EXT}))(?::\d+(?:-\d+)?)?`")
DIR_RE = re.compile(r"`([A-Za-z0-9_.][A-Za-z0-9_./-]*/)`")
SYM_RE = re.compile(r"`([A-Za-z0-9_./-]+\.rs)::([A-Za-z0-9_]+)`")
LINK_RE = re.compile(rf"\]\(([A-Za-z0-9_.][A-Za-z0-9_./-]*\.(?:{FILE_EXT}))(?:#[A-Za-z0-9_-]+)?\)")

# A token is a repo citation only if it is rooted in a directory this
# repository actually has, or is a file at the repo root. Everything else is
# a bare filename ("kan.py"), a runtime artifact ("head.mpk"), a
# machine-specific data path, or an external tree, and flagging those would
# bury the real failures and get the gate switched off.
ROOTS = (
    "src/", "tests/", "docs/", "config/", "scripts/", "examples/",
    ".claude/", ".github/", ".githooks/", "fixtures/", "research/",
    "experiments/", "vendor/", "ddrs-py/",
)
# Gitignored or generated, so their absence proves nothing.
SKIP_PREFIXES = (".ddrs", "output/", "target/", "examples/fixtures/")
# Already excluded by every extraction regex's character class, so this
# check is unreachable today; kept as a guard in case those classes widen.
SKIP_CHARS = set("<>*{}$…")
# Rust items a `file.rs::symbol` citation may name.
ITEM_KINDS = ("fn", "struct", "enum", "trait", "type", "const", "static", "mod", "impl", "macro_rules!")


def is_citation(root: Path, token: str) -> bool:
    """True when the token claims to name something in this repository."""
    if SKIP_CHARS & set(token) or token.startswith(SKIP_PREFIXES):
        return False
    if "/.ddrs" in token:   # a gitignored workspace nested under a real dir
        return False
    if token.startswith(ROOTS):
        return True
    # A bare name counts only if it names a real file at the repo root, so
    # that CLAUDE.md and Cargo.toml are checked and kan.py is not.
    return "/" not in token.rstrip("/") and (root / token).exists()


def symbol_defined(source: Path, name: str) -> bool:
    text = source.read_text(encoding="utf-8", errors="replace")
    return any(re.search(rf"\b{kind}\s+{re.escape(name)}\b", text) for kind in ITEM_KINDS)


def citations(text: str):
    """Yield (line_no, kind, token, symbol_or_None) for every citation."""
    for n, line in enumerate(text.splitlines(), 1):
        for m in SYM_RE.finditer(line):
            yield n, "symbol", m.group(1), m.group(2)
        for m in PATH_RE.finditer(line):
            yield n, "path", m.group(1), None
        for m in LINK_RE.finditer(line):
            yield n, "path", m.group(1), None
        for m in DIR_RE.finditer(line):
            yield n, "dir", m.group(1), None


def check_file(root: Path, doc: Path, listing: bool) -> list[str]:
    failures = []
    text = doc.read_text(encoding="utf-8", errors="replace")
    exempt = {n for n, line in enumerate(text.splitlines(), 1) if IGNORE_MARKER in line}
    for line_no, kind, token, symbol in citations(text):
        if line_no in exempt:
            continue
        if not is_citation(root, token):
            continue
        target = root / token
        rel = doc.relative_to(root)
        if listing:
            print(f"{rel}:{line_no}: {kind} {token}" + (f"::{symbol}" if symbol else ""))
        # Skill references cite their own siblings, so try the citing file's
        # directory before giving up. An .ic or .zarr store is a directory, so
        # existence rather than file-ness is the test.
        for base in (root, doc.parent, doc.parent.parent):
            if (base / token).exists():
                target = base / token
                break
        if not target.exists():
            failures.append(f"{rel}:{line_no}: unresolved {token}")
        elif kind == "dir" and not target.is_dir():
            failures.append(f"{rel}:{line_no}: not a directory {token}")
        elif kind == "symbol" and not symbol_defined(target, symbol):
            failures.append(f"{rel}:{line_no}: unresolved {token}::{symbol}")
    return failures


def collect(root: Path, globs) -> list[Path]:
    seen = []
    for g in globs:
        seen.extend(sorted(p for p in root.glob(g) if p.is_file()))
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--list", action="store_true", help="print every citation found")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    strict = []
    for doc in collect(root, STRICT_GLOBS):
        strict.extend(check_file(root, doc, args.list))
    warn = []
    for doc in collect(root, WARN_GLOBS):
        warn.extend(check_file(root, doc, args.list))

    for f in warn:
        print(f"warn: {f}")
    for f in strict:
        print(f"FAIL: {f}")
    print(f"\n{len(strict)} unresolved in agent context, {len(warn)} in prose docs")
    return 1 if strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
