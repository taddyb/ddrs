# Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every claim in always-loaded agent context resolve against source, leave only the book under the book's source tree, and leave only live tooling in `scripts/` and `examples/`.

**Architecture:** One pull request, two phases. Phase 1 (Tasks 1-9) changes file *content* only and renames nothing. Phase 2 (Tasks 10-15) does the `docs/book` + `research/` restructure. The phase order is a requirement, not a preference: Phase 1 puts the citation verifier in place and makes every claim true, so Phase 2's 414-reference rewrite operates on corrected text instead of racing it. A new stdlib-Python verifier, built test-first in Task 1, is what makes "every citation resolves" mechanical rather than judged, and every subsequent task uses it as a gate.

An earlier revision split this across two PRs because `landscape-deriv-objective` was in flight over the same files. It merged as PR #42, so there is one PR.

**Tech Stack:** Rust (cargo, BURN 0.21), mdBook 0.4.52 with `mdbook-katex` 0.9.4 and `mdbook-mermaid` 0.16.0, Python 3 stdlib only for repo scripts, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-11-repo-cleanup-design.md`

## Global Constraints

- **Working directory is the worktree** `/home/tbindas/projects/ddrs/.claude/worktrees/repo-cleanup`, branch `worktree-repo-cleanup`, rebased onto `origin/master` @ `3412a78`. Never `cd` to the shared checkout. Every line number in this plan was re-verified against `3412a78` after that rebase.
- **`.cargo/config.toml` is required and gitignored.** It sets `CUDARC_CUDA_VERSION = "13020"` because the host runs CUDA 13.3.1 and `cudarc` 0.19.7 panics on an unknown `nvcc --version`. It is already in place in this worktree. A fresh worktree needs it copied before any cargo command.
- **No behavior change anywhere in `src/`.** The only permitted source edits are doc-comment path strings. `cargo check --examples --tests` is the proof, and its baseline is green at 39.85 s.
- **Python is stdlib only** and run with `python3`, matching `scripts/journal.py` and `scripts/test_journal.py`. Do not add dependencies and do not introduce `uv` for these two files, because the journal hooks that run alongside them are plain `python3`.
- **Never edit the content of a dated findings doc.** They are the scientific record. Only inbound paths may change, and only where the path is a live pointer rather than a historical quotation.
- **No em-dashes in prose.** Use commas, colons, or separate sentences. Existing quoted text keeps its original punctuation.
- **Commit messages end with:**
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
  ```
- **`experiments/` is out of scope.** Its 37 bundles are the active landscape and adjoint workstream.
- **Invariant gate.** Any task that touched anything must still pass `cargo test --test ddr_sandbox_match --test gridded_bundle`, the pre-push hook's own gate.

---

## File Structure

**Created in Phase 1:**

| Path | Responsibility |
|---|---|
| `scripts/verify_doc_paths.py` | Extract every path, `file:line`, and `file.rs::symbol` citation from a set of markdown files and report the ones that do not resolve. Strict mode for agent context, warn mode for book pages |
| `scripts/test_verify_doc_paths.py` | Tests for the above, stdlib only, run as `python3 scripts/test_verify_doc_paths.py` |

**Modified in Phase 1:** `CLAUDE.md`, `.claude/skills/ddrs-dev/SKILL.md`, `.claude/skills/ddrs-dev/references/{config,testing,traps,gauge-population,build-and-env}.md`, `.claude/skills/ddrs-run/SKILL.md`, `.claude/skills/ddrs-eval-plots/SKILL.md`, `README.md`, `docs/intro.md`, `docs/architecture.md`, `docs/usage/{running,inputs-formatting}.md`, `docs/reference/{perf,baseline}.md`, `src/sparse/mod.rs` (one doc-comment line).

**Deleted in Phase 1:** `.claude/references/` (12 files), `.claude/2026-05-29-ddrs-docs-design.md`, `.claude/2026-05-29-ddrs-docs-plan.md`.

**Created in Phase 2:** `docs/book/` (the moved book), `research/{findings,specs,plans,why-analysis,figures,journal,archive}/`, `research/archive/README.md`, `research/findings/2026-09-11-repo-cleanup-findings.md`.

---

# Phase 1: truth pass (Tasks 1-9)

### Task 1: The citation verifier

Built first because every later task uses it as its gate. The spec requires it be seeded with the known-bad citations before any of them are fixed, so that a verifier with a false negative cannot give false confidence.

**Files:**
- Create: `scripts/verify_doc_paths.py`
- Create: `scripts/test_verify_doc_paths.py`

**Interfaces:**
- Produces: CLI `python3 scripts/verify_doc_paths.py`, exit 0 when every strict citation resolves, exit 1 otherwise, printing one `file:line: unresolved <token>` per failure. `--root <dir>` retargets it (tests only) and `--list` prints every citation found. Later tasks invoke it with no arguments. There is deliberately no `--warn-only` flag: the strict and warn glob sets are constants in the script, and nothing needs them overridden.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_verify_doc_paths.py`:

```python
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

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python3 scripts/test_verify_doc_paths.py
```

Expected: every case FAILs, because `scripts/verify_doc_paths.py` does not exist yet and `subprocess.run` returns a non-zero code with an ImportError-style message on stderr.

- [ ] **Step 3: Write the verifier**

Create `scripts/verify_doc_paths.py`:

```python
#!/usr/bin/env python3
"""Verify that every citation in agent-loaded context resolves.

CLAUDE.md and .claude/skills/** are loaded into an agent's context, so a
dead path or a drifted line number there is the most expensive kind of
wrong line in the repo. This script extracts every backtick-quoted path,
`file:line` citation and `file.rs::symbol` citation from those files and
reports the ones that do not resolve. Book pages under docs/ are scanned
in warn mode, because their line citations are a reading aid rather than a
contract.

    python3 scripts/verify_doc_paths.py          # strict, exit 1 on failure
    python3 scripts/verify_doc_paths.py --list    # print every citation found
"""
from __future__ import annotations

import argparse
import re
import sys
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
```

- [ ] **Step 4: Run the tests and make sure they pass**

```bash
python3 scripts/test_verify_doc_paths.py
```

Expected: every case prints `ok`, final line `all passed`, exit 0.

- [ ] **Step 5: Seed against the real tree, and confirm the output is exactly this**

```bash
python3 scripts/verify_doc_paths.py | tee /tmp/verify-before.txt
```

This was run against `571e2f6` while writing the plan. Expected: exit 1, a last line reading `9 unresolved in agent context, <N> in prose docs`, and exactly these nine strict failures. Only the first number is a gate; the prose-doc count is advisory and drifts as docs are edited (it was around 195 when this plan was written):

| Failure | Disposition |
|---|---|
| `research-status.md:175: src/sparse.rs` | **Genuine.** The same bug as the `CLAUDE.md` diagram. Task 7 fixes it |
| `ARCHITECTURE.md:99,100: tests/routing/test_*.py` | DDR-side paths in a ddrs-vs-DDR comparison table |
| `parity.md:115: scripts/train_and_test.py` | DDR-side path |
| `ddrs-journal/SKILL.md:8: docs/journal/YYYY-MM.md` | A filename template |
| `ddrs-run/SKILL.md:120` and `commands.md:70: config/experiments/config/sources/` | A path the trap exists to say does NOT exist |
| `ddrs-run/SKILL.md:120: config/experiments/foo.yaml` | A placeholder in that same trap |
| `build-and-env.md:96: fixtures/fixtures/` | A wrong path quoted as an error example |

If the count differs, the tree has moved since the plan was written: re-derive the list before continuing, and do not proceed with a verifier that reports fewer than nine, because that means a false negative.

Everything after the first row is a citation of something that deliberately does not exist. Task 7 marks those eight lines with `verify-doc-paths: ignore` rather than deleting them, so the exemptions stay explicit and reviewable in the diff.

Note on tuning: an earlier draft of this verifier reported **288** strict failures, almost all of them bare filenames (`kan.py`, `head.mpk`) and runtime artifacts. A gate that noisy gets switched off. The `ROOTS` predicate, the `/.ddrs` skip, resolution against the citing file's directory and its parent, and `exists()` rather than `is_file()` (an `.ic` store is a directory) are what take it from 288 to 9.

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_doc_paths.py scripts/test_verify_doc_paths.py
git commit -m "$(cat <<'EOF'
scripts: verify_doc_paths, a gate for citations in agent-loaded context

CLAUDE.md and .claude/skills/** are in context on every turn, so a dead
path there is the most expensive wrong line in the repo. This catches the
four dead paths the 2026-09-11 audit found, and it verifies that every
file.rs::symbol citation names a real item.

What it deliberately does NOT do is validate a line number. A citation
that drifted from line 1066 to 1202 still points at a line that exists,
so existence checking cannot see the drift, and bounds checking would
only catch a citation past EOF while implying a guarantee it does not
give. The remedy for drift is the symbol-citation policy, which this
tool can and does enforce.

Book pages are scanned in warn mode: their line citations are a reading
aid, not a contract, and failing on them would get the gate switched off.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 2: Delete `.claude/references/`, after proving it is a subset

`ddrs-dev/SKILL.md:262` already claims these 12 files were deleted. The 2026-07-30 audit found `docs/` to be a strict superset, but `docs/` has been edited since, so that finding is re-established here rather than assumed.

**Files:**
- Delete: `.claude/references/` (12 files)
- Modify: `src/sparse/mod.rs:11` (doc comment), `docs/intro.md` (the mapping table at lines 11-19)

**Interfaces:**
- Consumes: `scripts/verify_doc_paths.py` from Task 1.
- Produces: nothing later tasks import. Task 7 depends on this having happened, because it makes `ddrs-dev/SKILL.md:262` true.

- [ ] **Step 1: Prove the subset claim, pair by pair**

```bash
for pair in "algorithm:docs/algorithm.md" "architecture:docs/architecture.md" \
            "baseline:docs/reference/baseline.md" "burn-autograd:docs/reference/burn-autograd.md" \
            "comparing-to-ddr:docs/reference/ddr-comparison.md" \
            "formatting-inputs:docs/usage/inputs-formatting.md" \
            "graph-objects:docs/usage/graph-objects.md" \
            "perf-and-cuda-graphs:docs/reference/perf.md" \
            "reading-inputs:docs/usage/inputs-reading.md" \
            "reading-outputs:docs/usage/outputs.md" \
            "running-the-code:docs/usage/running.md" "setup:docs/setup.md"; do
  ref=".claude/references/ddrs-${pair%%:*}.md"; doc="${pair#*:}"
  echo "=== $ref -> $doc ==="
  # sentences present in the reference and absent from the doc
  grep -oE '`[^`]+`' "$ref" | sort -u > /tmp/ref.toks
  grep -oE '`[^`]+`' "$doc" | sort -u > /tmp/doc.toks
  comm -23 /tmp/ref.toks /tmp/doc.toks
done 2>&1 | tee /tmp/subset-check.txt
```

This compares the code-quoted tokens, which is where the load-bearing content lives: paths, config keys, commands, test names.

- [ ] **Step 2: Read the output and rescue anything real**

For every token the check reports as reference-only, decide: is it a fact `docs/` lacks, or is it phrasing? A fact gets written into the `docs/` counterpart in this same commit. Record each rescue in the commit message. If the output is empty for all 12 pairs, say so in the commit message instead.

Known example to expect and handle: `.claude/references/ddrs-setup.md:22` carries a "desktop-only DDR working tree" caveat for fixture regeneration. `CLAUDE.md` invariant 1 already records this as obsolete since 2026-08-19, so it is NOT rescued. Delete it with the file.

- [ ] **Step 3: Repoint the two inbound links**

`src/sparse/mod.rs:11` currently reads:

```rust
//! See `.claude/references/ddrs-burn-autograd.md` for the BURN-0.21 custom-backward
```

Change to:

```rust
//! See `docs/reference/burn-autograd.md` for the BURN-0.21 custom-backward
```

In `docs/intro.md`, delete the sentence at line 11 ("condensed agent-readable notes under `.claude/references/ddrs-*.md` are …") and the whole mapping table beginning at line 17 ("If you arrive from a `.claude/references/` note, its counterpart chapter is:"), through the end of that table.

- [ ] **Step 4: Delete the directory**

```bash
git rm -r .claude/references/
```

- [ ] **Step 5: Verify nothing points at it any more**

```bash
grep -rn "\.claude/references" . --include=*.md --include=*.rs --include=*.toml \
  | grep -v "docs/2026-07-30-docs-and-skills-audit.md" \
  | grep -v "docs/superpowers/specs/2026-09-11-repo-cleanup-design.md" \
  | grep -v "docs/superpowers/plans/2026-09-11-repo-cleanup.md"
```

Expected: no output. The three excluded files are the audit and this cleanup's own spec and plan, which describe the deletion as history and must keep the name.

```bash
cargo check --examples --tests
```

Expected: `Finished` and exit 0. This proves the `src/sparse/mod.rs` doc-comment edit did not break the crate.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
.claude: delete references/, a stale subset of the mdBook

Twelve files, each a quarter to a half the length of its docs/ counterpart,
generated from the skills on 2026-05-29 and never independently maintained.
ddrs-dev/SKILL.md already claimed they were deleted; this makes that true.

The subset claim was re-established pair by pair rather than inherited from
the 2026-07-30 audit, because docs/ has been edited since.

Inbound links repointed: src/sparse/mod.rs at docs/reference/burn-autograd.md,
and docs/intro.md loses its mapping table.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 3: Delete the two byte-identical `.claude/` duplicates

**Files:**
- Delete: `.claude/2026-05-29-ddrs-docs-design.md`, `.claude/2026-05-29-ddrs-docs-plan.md`

- [ ] **Step 1: Re-confirm they are identical before deleting**

```bash
md5sum .claude/2026-05-29-ddrs-docs-design.md .claude/specs/2026-05-29-ddrs-docs-design.md
md5sum .claude/2026-05-29-ddrs-docs-plan.md   .claude/specs/2026-05-29-ddrs-docs-plan.md
```

Expected: the two hashes in each pair match. If either pair differs, stop: the copies have diverged and this is no longer a deduplication.

- [ ] **Step 2: Check nothing cites the root copies specifically**

```bash
grep -rn "\.claude/2026-05-29" . --include=*.md --include=*.rs | grep -v "^./.claude/specs/"
```

Expected: no output, or only hits inside this plan and the spec.

- [ ] **Step 3: Delete and verify**

```bash
git rm .claude/2026-05-29-ddrs-docs-design.md .claude/2026-05-29-ddrs-docs-plan.md
python3 scripts/verify_doc_paths.py
```

Expected: the unresolved count is no higher than before.

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
.claude: drop the two docs-design/plan copies duplicated at the root

Byte-identical (md5-verified) to the copies in .claude/specs/, where the
rest of that spec series lives.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 4: `CLAUDE.md` factual corrections

Every row here was verified against source. Line numbers are as of `571e2f6`; re-grep before editing, because Tasks 2 and 3 did not touch `CLAUDE.md` but your own edits shift later lines.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Confirm the verifier currently fails on the dead paths**

```bash
python3 scripts/verify_doc_paths.py | grep -E "spike_backward|sparse\.rs"
```

Expected: `spike_backward/` appears. (`sparse.rs` is inside a fenced diagram without backticks, so the verifier cannot see it. That is a known limitation; Step 3 fixes it by hand and Step 6 confirms by grep.)

- [ ] **Step 2: Fix the architecture diagram (lines 288-311)**

The diagram lists `sparse.rs` as a file when `src/sparse/` is a directory, claims `spike_backward/` exists when it does not, and omits eight subsystems that have shipped since it was written. Replace the fenced block with:

```
src/
├── routing/              Core MC solver (port of ddr/src/ddr/routing/)
│   ├── mmc.rs            MuskingumCunge<I>: setup_inputs, forward, route_timestep
│   ├── leakance.rs       Losing-stream zeta term (off by default, see below)
│   ├── utils.rs          denormalize, hotstart, dense helpers
│   └── mod.rs
├── sparse/               CSR pattern + triangular solve + custom Backward
├── geometry.rs           Trapezoidal channel geometry (Leopold & Maddock)
├── config.rs             Parameter ranges, attribute minimums, log-space flags
├── cuda_graph/           Captured-graph path for the CUDA backend
├── nn/
│   ├── kan_head.rs       KAN head via rskan v0.1.3: Linear→KanLayer×N→Linear
│   │                     →Sigmoid, no inter-block ReLU (matches DDR `kan.py`)
│   └── disagg_head.rs    Precip-conditioned daily→hourly disaggregation head
├── adjacency/            Managed adjacency builder (fabric → zarr), incl.
│                         gridded.rs (DDM30) and subdivide.rs (off by default)
├── baseline/             Summed-Q' reference, cached under .ddrs/baselines/
├── training/             Driver, loss, checkpointing, bootstrap
├── pretrain/             Disaggregation-head pretraining
├── experiment/           Paper studies: adjoint/ and landscape/
├── cli/                  plan, run, show, status, gc, sources, import, experiment
├── bin/                  ddrs (primary) + legacy train/eval/train_and_test
└── data/                 Live readers for DDR's training data (no export step)
    ├── ids.rs            Comid, Staid newtypes; IdIndex<T>
    ├── error.rs          DataError with source-path context
    ├── dates.rs          TimeAxis + rho-window sampler (mirrors DDR's Dates)
    └── store/zarr.rs     ConusAdjacencyStore, GagesAdjacencyStore via zarrs

tests/                    Integration tests; each file is its own crate
examples/                 compare_ddr_sandbox (regression), benchmark_hydrograph
scripts/                  Python helpers run under DDR's uv venv
vendor/                   Why [patch.crates-io] points at the taddyb/burn and
                          taddyb/cubecl forks (see vendor/README.md)
ddrs-py/                  maturin/PyO3 bindings, read-only CPU inference
```

- [ ] **Step 3: Fix the seven wrong citations and claims**

| Find | Replace with |
|---|---|
| `src/cli/run.rs:322` (line 151) | `src/cli/run.rs::standalone_eval_unsupported` if that is the enclosing fn, else `src/cli/run.rs` with no line |
| `src/bin/ddrs.rs:167` (line 154) | `src/bin/ddrs.rs` with no line. The `Cmd::Init` arm is at 193-196 and will drift again |
| `src/config.rs:1066` (line 538) | `src/config.rs::validate_subdivision_reaches_the_builder` |
| "First study: `adjoint` (inflow-gradient influence map)." (line 160) | "Two studies ship: `adjoint` (inflow-gradient influence map) and `landscape` (loss-landscape and watershed-perturbation probes, `src/experiment/landscape/`)." |
| "Three skills cover this repo." (line 649) | "Four skills cover this repo." plus a `ddrs-run` entry, see Step 5 |

For the first row, run `awk 'NR>=300 && NR<=330' src/cli/run.rs` to find the enclosing function name and use it. If the string sits in a match arm with no distinct function, cite the file with no suffix.

- [ ] **Step 4: Fix the Training-objective paragraph (lines 409-412)**

Delete the "Not on `master` yet" block entirely and replace with:

```markdown
  Pairs with `experiment.optimizer: adadelta`, which is scale-free and ignores
  the `learning_rate` schedule by design. Gradient accumulation
  (`experiment.use_grad_accum`, `grad_accum_steps`) is also available; it is
  rejected at load unless `grad_accum_steps >= 2`.
```

Verify first that the keys exist:

```bash
grep -n "use_grad_accum\|grad_accum_steps\|NseBatch\|OptimizerKind" src/config.rs | head
```

- [ ] **Step 5: Fix the "When in doubt" skill roster (lines 647-655)**

Replace the "Three skills" bullet with four entries. `ddrs-run` is the runbook for launching, watching, resuming and auditing a job, and is what an agent asked to "train a model" or "resume from epoch N" needs.

- [ ] **Step 6: Fix the data-source group list (line 163)**

`config/sources/` holds nine files, not five. Run `ls config/sources/` and list them all, noting which are the maintained groups and which are one-off snapshots.

- [ ] **Step 7: Verify**

```bash
grep -n "spike_backward\|sparse\.rs\|exp_train\|Three skills\|run\.rs:322\|ddrs\.rs:167\|config\.rs:1066" CLAUDE.md
```

Expected: no output.

```bash
python3 scripts/verify_doc_paths.py
```

Expected: unresolved count strictly lower than the Task 1 baseline, and no new failures naming `CLAUDE.md`.

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
CLAUDE.md: correct nine claims that no longer hold

Verified against source, not assessed for plausibility:
- the architecture diagram showed sparse.rs as a file (it is a directory)
  and spike_backward/ as present (it is gone), and omitted adjacency/,
  baseline/, cli/, cuda_graph/, experiment/, pretrain/, training/ and
  nn/disagg_head.rs
- nse-batch, optimizer, use_grad_accum and grad_accum_steps were described
  as unlanded work on branch exp_train; all four are on master with tests
- three skills were claimed, four exist; ddrs-run was unnamed
- adjoint was called the only paper study; landscape also ships
- five source groups were listed, config/sources/ holds nine
- three src/ line citations had drifted by 26, 2 and 136 lines

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 5: Compress the two closed workstreams in `CLAUDE.md`

Leakance is 84 lines and reach subdivision is 54, both describing campaigns that closed NO-GO, both loaded on every turn. The verdict stays in always-loaded context. The operational detail is almost entirely duplicated in the skills already, so this is mostly a deletion: see Step 1's table for what is already elsewhere and Step 2 for the only two items that genuinely move.

**Files:**
- Modify: `CLAUDE.md` (the Leakance and Reach subdivision sections; re-grep, they moved to 445 and 529 after Task 4)
- Modify: `.claude/skills/ddrs-dev/references/config.md` (receives ONLY the eval zeta recipe, extending its existing leakance section)
- Modify: `.claude/skills/ddrs-dev/references/testing.md` (receives ONLY a Subdivision coverage row)

**Interfaces:**
- Produces: two `## ` sections in `CLAUDE.md` of about 10 and 8 lines, each ending in a pointer to `ddrs-dev/references/config.md` and to the closing findings doc. Task 7 edits the same two reference files, so do Task 5 before Task 7 or resolve by hand.

- [ ] **Step 1: Establish what is ALREADY elsewhere before moving anything**

The plan originally said to append both sections' operational detail into the skills.
Checked during execution, and most of it is already there, often stated better. Do NOT
append a second copy: that would be duplication, which is the thing this task exists to
remove. Verify each row yourself before relying on it:

| CLAUDE.md content | Already lives at | So |
|---|---|---|
| The three leakance enable changes (`use_leakance: true` forcing `use_cuda_graphs: false`, the three `learnable_parameters`, matching `parameter_ranges`) | `ddrs-dev/references/config.md` §"Leakance: enabling it" | DELETE from CLAUDE.md, move nothing |
| The three leakance parameter ranges with units | `config.md` §`parameter_ranges`, and the `params:` table's `use_leakance` / `leakance_losing_only` / `leakance_impervious_threshold` rows | DELETE, move nothing |
| All four leakance gate commands | `testing.md` Tier A (`leakance_gradcheck`, `leakance_off_parity`, `zeta_accum`) and its "What covers what" Leakance row | DELETE, move nothing |
| The `zeta` / `zeta_net` netCDF schema and which writer produces it | `ddrs-eval-plots/SKILL.md` output table and `references/parameter_map.md`, both of which additionally give the `COMID_eval` dimension and its 64,892 reaches, which CLAUDE.md never stated | DELETE, move nothing |
| All seven subdivision fields, the `Δx_target` formula, and the three enable preconditions | `.claude/REACH-SUBDIVISION.md` | DELETE, move nothing |
| The subdivision NO-GO measurements | `.claude/REACH-SUBDIVISION.md` | DELETE, keep only the headline numbers in the status block |

- [ ] **Step 2: Move the two things that are genuinely unique, and only those**

1. **The eval-time zeta recipe for an existing checkpoint.** `ddrs-eval-plots` names
   the `eval --zeta-output` flag but not the full invocation. Move the complete
   command block (`cargo build --release --bin eval`, then `target/release/eval
   --config … --checkpoint … --output … --zeta-output …`) plus the "~10 min, no
   retrain" note into `config.md`, extending the EXISTING §"Leakance: enabling it"
   rather than opening a new section.
2. **The subdivision gate list.** Add `subdivide`, `subdivision_integration` and
   `gauge_mass_conservation` to `testing.md`'s "What covers what" table as a
   Subdivision row, noting `compare_ddr_sandbox` must stay an ABSOLUTE MATCH.
   Confirm first with `ls tests/` that all three files exist.

- [ ] **Step 3: Replace the `CLAUDE.md` leakance section**

```markdown
## Leakance (CLOSED, NOT PROMOTABLE, 2026-07-06)

A losing-stream term subtracted from the routing RHS `b`:
`zeta = leakance_factor · area_z · K_D · (depth − d_gw)`, positive zeta means a
losing reach. Code-complete and gradient-exact (`src/routing/leakance.rs`,
`TimestepLeakanceOp`); `params.use_leakance` defaults false. Do NOT remove it.

Do NOT re-open the question. A gauge measures the SUM of zeta over its upstream
network, and that sum does not determine the per-reach distribution, so training
constrains aggregate loss while carrying zero information about per-reach flux.
Every rival explanation (gradient starvation, objective noise, uninformative
inputs, sign ambiguity) was individually refuted.

Verdict and refutations: `docs/2026-07-06-leakance-nogo-scientific-summary.md` §3
Enable steps, ranges, gates, zeta diagnostic: `ddrs-dev/references/config.md`
```

- [ ] **Step 4: Replace the `CLAUDE.md` subdivision section**

```markdown
## Reach subdivision (`params.subdivision`, NO-GO, off by default)

A build-time normalization of reach length toward `Δx ≈ c_ref·Δt` inside the
managed adjacency builder. Built to make the Muskingum coefficients non-negative
by construction; measured on 1,841 CONUS gauges, `frac c1 < 0` got WORSE (93.0 %
to 98.79 % at `max_pieces: 8`). Both coefficients are non-negative only inside a
window `2(1−2X)` wide, which is 1.4 % at the measured CONUS median X = 0.4966,
and a static piece count cannot hold a flow-varying `Cr` inside it. It does
nearly eliminate `Cr > 2` (3.93 % to 0.31 %), via the length clamp.

Correct, gated off, stays in-tree as the measurement apparatus. Do not re-open
the "Cr ≈ 1 implies non-negative" argument without reading
`.claude/REACH-SUBDIVISION.md`. Enable steps and fields:
`ddrs-dev/references/config.md`
```

- [ ] **Step 5: Verify nothing was lost**

```bash
git show HEAD:CLAUDE.md | sed -n '426,563p' > /tmp/old-sections.txt
grep -oE '`[^`]+`' /tmp/old-sections.txt | sort -u > /tmp/old.toks
cat CLAUDE.md .claude/skills/ddrs-dev/references/config.md \
    .claude/skills/ddrs-dev/references/testing.md \
  | grep -oE '`[^`]+`' | sort -u > /tmp/new.toks
comm -23 /tmp/old.toks /tmp/new.toks
```

Expected: mostly empty. A token that appears only in the old CLAUDE.md sections and
nowhere else must be either restored or deliberately dropped with a note in the commit
message. Because Step 1 established that the skills already carry this material, widen
the comparison set to include the files that actually hold it:

```bash
cat CLAUDE.md \
    .claude/skills/ddrs-dev/references/config.md \
    .claude/skills/ddrs-dev/references/testing.md \
    .claude/REACH-SUBDIVISION.md \
    .claude/skills/ddrs-eval-plots/SKILL.md \
    .claude/skills/ddrs-eval-plots/references/parameter_map.md \
  | grep -oE '`[^`]+`' | sort -u > /tmp/new.toks
comm -23 /tmp/old.toks /tmp/new.toks
```

```bash
wc -l CLAUDE.md
```

Task 4 grew the file to **681** lines, because the rebuilt architecture diagram names
eight subsystems the old one omitted. Compressing leakance (84 lines to about 10) and
subdivision (54 to about 8) removes roughly 120, so expect about **555 to 575**.
Report the measured number: later steps quote it and must not quote a prediction.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md .claude/skills/ddrs-dev/references/
git commit -m "$(cat <<'EOF'
CLAUDE.md: compress leakance and reach subdivision to status blocks

138 of 663 always-loaded lines described two campaigns that closed NO-GO.
The verdict and the do-not-reopen reasoning stay in context; the enable
steps, parameter ranges and gate commands move to ddrs-dev/references/,
which loads on trigger. Verified no code-quoted token was lost in the move.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 6: Adopt the symbol-citation policy

Six of the audit's findings were drifted `file:line` citations. Re-pinning numbers resets a clock; citing symbols stops it.

**Files:**
- Modify: `CLAUDE.md` (the "Conventions specific to this repo" section)
- Modify: `.claude/ARCHITECTURE.md`, `.claude/REACH-SUBDIVISION.md`
- Modify: `.claude/skills/ddrs-dev/SKILL.md`, `.claude/skills/ddrs-dev/references/gauge-population.md`
- Modify: `.claude/skills/ddrs-eval-plots/references/channel_geometry.md`
- Modify: `scripts/verify_doc_paths.py`, `scripts/test_verify_doc_paths.py` (Step 4's enforcement)

- [ ] **Step 1: Find every remaining `src/…:NN` citation in agent context**

```bash
grep -rnoE 'src/[A-Za-z0-9_/]+\.rs:[0-9]+(-[0-9]+)?' \
  CLAUDE.md .claude/skills/ .claude/ARCHITECTURE.md \
  .claude/PHYSICS-CORRECTIONS.md .claude/REACH-SUBDIVISION.md
```

Enumerated during execution: **thirteen**, not the nine this plan originally estimated,
and they reach three files the estimate missed. Note the glob above includes the three
`.claude/*.md` agent notes, which the earlier narrower grep skipped.

| Citation | Disposition |
|---|---|
| `ARCHITECTURE.md:255` `src/routing/mmc.rs:289-294` | convert |
| `ARCHITECTURE.md:256` `src/sandbox.rs:88-89` | convert |
| `REACH-SUBDIVISION.md:17` and `:210` `src/config.rs:491-576` | convert (a range spanning a whole struct; cite the type, not the span) |
| `REACH-SUBDIVISION.md:142` `src/routing/mmc.rs:144` | convert |
| `REACH-SUBDIVISION.md:202` `src/data/store/zarr.rs:121-122` | convert |
| `REACH-SUBDIVISION.md:214` `src/routing/mmc.rs:267-273` | convert |
| `ddrs-dev/SKILL.md:142` `src/cli/run.rs:322` | convert to a bare `src/cli/run.rs`, no symbol. Same ruling as Task 4: the string sits in the generic `fn dispatch_backend<I>`, so a symbol would mislead |
| `ddrs-dev/SKILL.md:144` `src/bin/ddrs.rs:167` | convert to a bare `src/bin/ddrs.rs`. The real `Cmd::Init` arm is near 193 and will drift again |
| `gauge-population.md:44` `src/data/store/gage_csv.rs:62` | convert |
| `ddrs-eval-plots/references/channel_geometry.md:11` `src/geometry.rs:37-67` | convert. A reviewer verified this range is currently exact, which is precisely why it will drift unnoticed |
| `config.md:138` `src/config.rs:113` | **LEAVE IT.** Task 7 deletes the whole obsolete `use_precip` sentence containing it. Converting a citation that is about to be removed wastes work and collides |
| `gauge-population.md:53` `src/data/store/zarr.rs:128` | **LEAVE IT.** Task 7 owns this line: the citation is wrong AND the claim beside it is wrong (the body is `indices_0.is_empty()`, not an `order` length test). Task 7's content fix produces the symbol citation |

**Note what this enumeration proves.** `ddrs-dev/SKILL.md:142` and `:144` carry the
same two drifted citations Task 4 just corrected in `CLAUDE.md`. The skill had copied
`CLAUDE.md` and inherited its errors, which is the duplication-propagates-staleness
mechanism this whole cleanup is about, caught in the act.

- [ ] **Step 2: Convert each to `file.rs::symbol`**

For each hit, find the enclosing item:

```bash
awk -v n=<LINE> 'NR<=n && /^\s*(pub )?(fn|struct|enum|trait|impl|const) /{last=$0; ln=NR} END{print ln": "last}' <FILE>
```

Use the item name. If a line citation points at data rather than an item (a YAML line, a fixture value), leave it: the rule is about `src/` only.

- [ ] **Step 3: Write the rule into `CLAUDE.md`**

Add to "Conventions specific to this repo":

```markdown
- **Cite symbols, not lines, inside `src/`.** Write
  `src/config.rs::validate_subdivision_reaches_the_builder`, never
  `src/config.rs:1066`. Line numbers drift silently and a 2026-09-11 audit
  found six that had, by up to 136 lines. Line citations are fine for files
  that do not move: the DDR reference tree, fixtures, and config YAML where
  the line number is the content. `scripts/verify_doc_paths.py` checks that
  every `file.rs::symbol` citation resolves.
```

- [ ] **Step 4: Make the verifier enforce the policy, not merely tolerate it**

A policy nothing checks decays. Now that the strict files carry no `src/**.rs:NN`
citations, teach the verifier to refuse new ones. In `scripts/verify_doc_paths.py`,
`PATH_RE` already captures the line suffix; add a second capture group for it and,
in `check_file`, report a strict failure when a token under `src/` carries a `:NN`
suffix:

```python
    if kind == "path" and line_suffix and token.startswith("src/"):
        failures.append(
            f"{rel}:{line_no}: line citation into src/ is not allowed, "
            f"cite file.rs::symbol instead of {token}:{line_suffix}"
        )
        continue
```

Add a test case to `scripts/test_verify_doc_paths.py` asserting that
``See `src/config.rs:9999`.`` now FAILS in a strict file, and amend the existing
"line suffix stripped" case so it covers a non-`src/` path, where line citations
remain legal (a fixture or config line number is the content).

This is the step that makes the tool's claim about drift true: it cannot detect a
citation that drifted to a different valid line, so instead it forbids the form that
can drift.

- [ ] **Step 5: Verify**

```bash
grep -rnoE 'src/[A-Za-z0-9_/]+\.rs:[0-9]+' \
  CLAUDE.md .claude/skills/ .claude/ARCHITECTURE.md \
  .claude/PHYSICS-CORRECTIONS.md .claude/REACH-SUBDIVISION.md
python3 scripts/test_verify_doc_paths.py
python3 scripts/verify_doc_paths.py
```

Expected: the grep returns **exactly the two citations Step 1 told you to leave**
(`config.md:138` and `gauge-population.md:53`), both of which Task 7 removes; the test
suite passes including the two amended cases; and the verifier's strict count does not
rise. It cannot reach zero until Task 7 closes, because that task still owns
`ddrs-dev/SKILL.md:262` and the eight deliberate non-paths.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md .claude/skills/ scripts/
git commit -m "$(cat <<'EOF'
docs: cite symbols, not line numbers, inside src/

Six citations in agent-loaded context had drifted, by up to 136 lines, and
every one still looked authoritative. Symbol citations do not drift, and
verify_doc_paths.py now checks that each one resolves to a real item.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 7: Correct the four skills

**Files:**
- Modify: `.claude/skills/ddrs-dev/SKILL.md`, `.claude/skills/ddrs-dev/references/{build-and-env,config,testing,traps,gauge-population}.md`, `.claude/skills/ddrs-run/SKILL.md`, `.claude/skills/ddrs-eval-plots/SKILL.md`

- [ ] **Step 1: Fix the seven verified-false claims**

| File and anchor | Current | Correct to |
|---|---|---|
| `ddrs-dev/SKILL.md:262` | mdBook "is a strict superset of the **deleted** `.claude/references/` copies" | Now true after Task 2. Keep the sentence, drop any hedging that implies they might still exist |
| `ddrs-dev/SKILL.md:254-256` | "Three skills live in this repo" | Four. Add `ddrs-journal`, which is wired into `.claude/settings.json` hooks |
| `ddrs-dev/SKILL.md:172` | "`adjoint` (the only study so far)" | `adjoint` and `landscape` |
| `ddrs-dev/references/build-and-env.md:16` | "(`docs/setup.md` claims otherwise, it is wrong.)" | Delete the parenthetical. `docs/setup.md:74-85` says the same thing |
| `ddrs-dev/references/config.md:180` | "Four validators run at `Config::from_yaml_file`, plus one at dataset open" | Nine run directly, plus a nested `validate_subdivision_reaches_the_builder`, plus one at dataset open. Add the missing `validate_enforce_positivity` row to the guards table |
| `ddrs-dev/references/gauge-population.md:52-53` | `is_headwater` at `zarr.rs:128`, "`order` length > 1" | `src/data/store/zarr.rs::is_headwater`, whose body is `self.indices_0.is_empty()`: it tests edge count, not `order` length. **Task 6 deliberately left this citation for you**, because the claim beside it is wrong too and one edit should fix both |
| `ddrs-run/SKILL.md:245` | "`ddrs experiment <name>`: **Not on master.**" | It is on master. `src/experiment/{adjoint,landscape}` and the `Experiment` subcommand all exist. Delete the row from the does-not-work table and cross-reference `ddrs-dev/SKILL.md`'s working description |

Confirm each before editing:

```bash
grep -c "fn validate_" src/config.rs
grep -n "pub fn is_headwater" -A2 src/data/store/zarr.rs
ls src/experiment/
```

- [ ] **Step 2: Delete every hard-coded count**

`testing.md:3` claims "70 test files, 233 `#[test]` fns" (actual 88 and 363) two lines above its own rule against hard-coded counts. Replace with a sentence asserting the suites pass, not their size. Delete the per-file line counts from `ddrs-dev/SKILL.md`'s contents table and from the `ddrs-eval-plots` file list: `research-status.md` is declared 213 lines and is 782.

- [ ] **Step 3: Remove the paragraph duplicating `CLAUDE.md`**

`config.md:73-92` restates `CLAUDE.md`'s loss-kind list including the same wrong `exp_train` sentence. Replace the whole block with:

```markdown
`experiment.loss.kind` and the optimizer keys are documented in CLAUDE.md's
"Training objective" section. This file covers only what CLAUDE.md omits:
the validator list below, and the closed-workstream config in the leakance
and subdivision sections.
```

- [ ] **Step 4: Delete the obsolete `use_precip` passage**

`config.md:123-126` says `use_precip` is "still referenced" in three places. All three were fixed. Replace with one sentence: there is no `use_precip` key, the head always consumes precip, and the presence of the `disaggregation:` block is what makes `aorc_precip` mandatory.

- [ ] **Step 5: Fix the duplicate `T11` in `traps.md`**

Two distinct traps are both headed `## T11`: the time-major `Qr(time, divide_id)` axis sniff, and the forward-only autodiff tape retention. Renumber the second to the next free number, and add a row for it to the symptom index at lines 7-22, where it currently has none. Its discriminating test is
`cargo run --release --example leak_probe -- ad-notrack 40`.

- [ ] **Step 6: Fix the one real path bug and mark the eight deliberate non-paths**

`research-status.md:175` cites `src/sparse.rs`; it is the directory `src/sparse/`, the same bug Task 4 fixed in the `CLAUDE.md` diagram.

The other eight verifier failures are citations of things that deliberately do not exist. Append the marker comment to each line so the exemption is explicit in the diff rather than hidden inside the verifier:

| File and line | Why exempt |
|---|---|
| `.claude/ARCHITECTURE.md:99` and `:100` | DDR-side paths in the ddrs-vs-DDR comparison table |
| `ddrs-eval-plots/references/parity.md:115` | DDR's `scripts/train_and_test.py` |
| `ddrs-journal/SKILL.md:8` | `docs/journal/YYYY-MM.md` is a filename template |
| `ddrs-run/SKILL.md:120` (two tokens) and `ddrs-run/references/commands.md:70` | `config/experiments/config/sources/` is the path the trap exists to say does NOT exist, and `foo.yaml` is its placeholder |
| `ddrs-dev/references/build-and-env.md:96` | `fixtures/fixtures/` is quoted as a wrong path |

The marker is a trailing HTML comment, invisible in rendered markdown:

```markdown
| `tests/routing_utils.rs` | `tests/routing/test_routing_utils.py` | Denormalize + triangular solve | <!-- verify-doc-paths: ignore -->
```

- [ ] **Step 7: Verify**

```bash
python3 scripts/verify_doc_paths.py
grep -rn "Not on master\|the only study\|Three skills" .claude/skills/
grep -c "^## T11" .claude/skills/ddrs-dev/references/traps.md
```

Expected: verifier exits **0** with `0 unresolved in agent context`; no output from the grep; `1` from the T11 count.

- [ ] **Step 8: Commit**

```bash
git add .claude/skills/ .claude/ARCHITECTURE.md
git commit -m "$(cat <<'EOF'
skills: correct seven false claims and drop the drifted counts

Verified against source:
- ddrs-run said `ddrs experiment` is not on master; src/experiment/ is here
  and ddrs-dev documents the same command as working, so the library
  contradicted itself
- ddrs-dev said docs/setup.md is wrong about CPU compilation; setup.md now
  says the same thing
- config.md said four validators run at load; nine do, plus a nested one
- gauge-population.md put is_headwater 80 lines off and described it as an
  order-length test when it tests edge count
- three skills were claimed, four exist
- adjoint was called the only study; landscape also ships
- traps.md had two different traps numbered T11, the second missing from
  the symptom index
- research-status.md cited src/sparse.rs, which is a directory

Eight lines citing DDR-side paths, filename templates, or paths a trap
exists to say do NOT exist now carry a verify-doc-paths: ignore marker, so
the gate reaches zero without the exemptions being hidden in the verifier.

Hard-coded counts deleted: testing.md's "70 files, 233 tests" (actual 88 and
363) sat two lines above the rule forbidding them, and research-status.md was
declared 213 lines while being 782.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 8: Correct the book pages and README

**Files:**
- Modify: `docs/reference/perf.md`, `docs/reference/baseline.md`, `docs/usage/inputs-formatting.md`, `docs/usage/running.md`, `docs/architecture.md`, `docs/intro.md`, `docs/setup.md`, `README.md`

**Scope correction, made during execution.** The audit graded `docs/setup.md` and
`docs/intro.md` CLEAN and `docs/usage/running.md` MINOR. On the `use_cuda_graphs`
axis all three are wrong, and `setup.md` carries five stale claims. The auditors were
checking against the 2026-07-30 follow-up list and for newly-landed features; none of
them swept for the **2026-08-19 default flip**, so that correction's blast radius was
undercounted. Work from the verified table in Step 2 below, not from the audit's
per-file verdicts.

- [ ] **Step 1: Confirm the two wrong table cells against source**

```bash
grep -n "use_cuda_graphs" config/merit_training.yaml
grep -n "tau = r.tau.unwrap_or" src/config.rs
```

Expected: `use_cuda_graphs: false` at line 143, and `p.tau = r.tau.unwrap_or(9)`.

- [ ] **Step 2: Fix every stale `use_cuda_graphs` claim, in all five files**

`config/merit_training.yaml:143` is `use_cuda_graphs: false`. It flipped on 2026-08-19
with the `ddr_match` deprecation, because the captured kernel hardcodes the legacy
celerity and cannot be used with the corrected physics that is now default.
`docs/reference/ddr-comparison.md:155-164` documents that event correctly; the fix
never propagated anywhere else. Every row below was verified by grep against the YAML
during execution:

| File and line | Current claim | Correct to |
|---|---|---|
| `docs/intro.md:64-66` | "Defaults in `config/merit_training.yaml` are now `sparse_solver: cuda` + `use_cuda_graphs: true`" | `sparse_solver: cuda` with `use_cuda_graphs: false`, noting the 2026-08-19 flip |
| `docs/intro.md:69` | calls it "the `DDRS_FORCE_GRAPHS=1` CUDA-capture path" | It is the CUDA **backend** path. The env var selects `Cuda<f32, i32>` and does not enable capture |
| `docs/setup.md:67-68` | "current `config/merit_training.yaml` defaults to `sparse_solver: cuda` and `use_cuda_graphs: true`, so the GPU path is exercised on every default training run" | `use_cuda_graphs: false`; the cuSPARSE solver is still exercised, the capture path is not |
| `docs/setup.md:311-312` | "ships with `sparse_solver: cuda` and `use_cuda_graphs: true`" | same correction |
| `docs/setup.md:389-390` | "routes the example through the `Cuda<f32, i32>` inner backend with `use_cuda_graphs=true` regardless of the YAML" | Drop the `use_cuda_graphs=true` clause. The env var only swaps the inner backend |
| `docs/usage/running.md:12-14` | "ships with `sparse_solver: cuda` and `use_cuda_graphs: true`" | same correction |
| `docs/usage/inputs-formatting.md:147` | a `config/merit_training.yaml` excerpt showing `use_cuda_graphs: true  # SP-10: forward CUDA Graph capture+replay` | `use_cuda_graphs: false` with a comment saying why it flipped |
| `docs/usage/inputs-formatting.md:637-640` | "`use_cuda_graphs` flipped to `true` in SP-10 … Don't hard-code the assumption that either is `false`" | Now backwards. Record both flips: `true` in SP-10, back to `false` on 2026-08-19, and keep the read-the-YAML advice |
| `docs/reference/perf.md:48` | merit-YAML column says `true` (line 141) | `false` (line 143) |
| `docs/reference/perf.md:59` | `use_cuda_graphs: true  # YAML value since the SP-10 close commit (e35af29)` | `false`, with the 2026-08-19 reason, cross-referencing `docs/reference/ddr-comparison.md` |

Leave alone every other `use_cuda_graphs` mention the grep finds. Most describe the
flag's semantics or the capture precondition (`use_cuda_graphs && sparse_solver == Cuda`)
and are correct. Only a claim about the **shipped YAML value**, or about
`DDRS_FORCE_GRAPHS` enabling capture, is wrong.

- [ ] **Step 3: Fix the obsolete desktop-DDR caveat in `docs/setup.md`**

`docs/setup.md:91-93` and `:309-310` tell the reader a valid V1 fixture "currently
requires the desktop's DDR working tree" and point at
`docs/reference/ddr-comparison.md` for the caveat. That target marks the caveat
**OBSOLETE since 2026-08-19** at its line 155, and CLAUDE.md invariant 1 says any DDR
checkout at or past #192 is a valid reference. So setup.md is routing readers to an
obsolete warning as though it were live. Remove both, and point at
`docs/reference/ddr-comparison.md` §Regenerating fixtures instead.

- [ ] **Step 4: Fix the rest of `docs/reference/perf.md`**

Line 48's merit-YAML column says `true` (line 141); it is `false` at line 143, flipped 2026-08-19 when the captured kernel became incompatible with the corrected physics that is now default. Line 59's example block says `use_cuda_graphs: true  # YAML value since the SP-10 close commit (e35af29)`; replace the value with `false` and the comment with a note that it flipped on 2026-08-19 with the `ddr_match` deprecation, cross-referencing `docs/reference/ddr-comparison.md`, which documents that event correctly. Lines 47-48 also cite `src/config.rs:407-410` and `:454`; replace with symbol citations.

- [ ] **Step 5: Fix `docs/usage/inputs-formatting.md`**

Line 474: `tau` default is 9, not 3, since `54cd386` (2026-08-08); 3 is on the retired, wrong-direction scale. Line 476's `use_cuda_graphs` cell is already handled by Step 2; do not edit it twice. Then add the three missing rows to the `params:` table: `ddr_match`, `enforce_positivity`, `subdivision`. Add a note that `ddr_match` changes other defaults, since its absence is the reason the `use_cuda_graphs` cell went stale.

- [ ] **Step 6: Fix `docs/reference/baseline.md`**

Lines 147 and 386 describe an `init → plan → run` lifecycle. `ddrs init` is a stub that exits 2; `setup.md` and `running.md` already say so. Change both to `plan → run`.

- [ ] **Step 7: Fix `docs/architecture.md`**

The `src/adjacency/` file table at lines 147-155 omits `gridded.rs` (DDM30 sub-reach network relabelling) and `subdivide.rs` (the off-by-default length normalization). Add both rows.

Also fix its `src/bin/` inventory near line 100, found during Task 4's review. It says
`src/bin/` holds **ten** binaries and lists ten, but `ls src/bin/` returns **twelve**:
`probe_courant.rs` and `probe_n_slope.rs` are missing. Rather than correcting the count
to twelve, which drifts again on the next binary, name the families the way Task 4's
diagram now does: `ddrs` (primary CLI), the three legacy binaries
(`train`/`eval`/`train_and_test`), `dump_parameters`, the `pretrain_disagg*` family,
and the `probe_*` family. Adding a binary to an existing family then cannot falsify the
text, while a genuinely new family stays visible.

- [ ] **Step 8: Fix `docs/usage/running.md`**

The CLI reference documents `plan`, `run`, `show`, `status`, `gc`, `sources` and `import` exhaustively and never mentions `ddrs experiment`. Add it, listing the flags from `src/bin/ddrs.rs` (name, bundle, backend, arms, max_gauges, skip_validate, jobs, dry_run, shard) and naming the two studies. Also add `use_grad_accum` and `grad_accum_steps` wherever the training configuration is covered.

- [ ] **Step 9: Fix `README.md`**

Lines 211-213 frame `nse-batch` and `optimizer: adadelta` as "once PR #31 lands". Both merged 2026-07-30 as `24cb4d0`, with passing tests `nse_batch_loss_kind_parses` and `optimizer_defaults_to_adam_and_parses_adadelta`. Rewrite as shipped features.

- [ ] **Step 10: Verify**

```bash
mdbook build
python3 scripts/verify_doc_paths.py
grep -rn "PR #31\|init → plan → run\|unwrap_or(3)" docs/ README.md
```

Expected: `mdbook build` succeeds; the verifier's warn count drops; the grep returns nothing.

- [ ] **Step 11: Commit**

```bash
git add docs/ README.md
git commit -m "$(cat <<'EOF'
docs: fix the two config defaults that changed under the book

Both broke after the 2026-07-30 audit and were never propagated:
- perf.md and inputs-formatting.md said merit_training.yaml ships
  use_cuda_graphs: true; it flipped to false on 2026-08-19 when the
  captured kernel stopped being usable with the corrected physics
- inputs-formatting.md gave tau's default as 3; it has been 9 since
  54cd386 (2026-08-08), and 3 is on the retired scale

The params: table also gained ddr_match, enforce_positivity and
subdivision. The ddr_match omission is why the cuda-graphs cell went
stale: the table had no concept of a physics-mode-dependent default.

Also: baseline.md's leftover init → plan → run, architecture.md's
adjacency table missing gridded.rs and subdivide.rs, running.md missing
the ddrs experiment subcommand entirely, and README's nse-batch still
described as unlanded.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 9: Phase 1 gates and the findings doc

**Files:**
- Create: `docs/2026-09-11-repo-cleanup-findings.md` (Task 11 moves it to `research/findings/`)
- Modify: `.claude/skills/ddrs-dev/references/testing.md` (record the new gate)

- [ ] **Step 1: Run the full Phase 1 gate set**

```bash
python3 scripts/test_verify_doc_paths.py
python3 scripts/verify_doc_paths.py
cargo check --examples --tests
mdbook build
cargo test --test ddr_sandbox_match --test gridded_bundle
grep -rn "\.claude/references" . --include=*.md --include=*.rs \
  | grep -v 2026-07-30-docs-and-skills-audit \
  | grep -v 2026-09-11-repo-cleanup
```

Expected: all pass, verifier exits 0, the final grep is empty. If the verifier is not at zero, fix the remaining citations before proceeding: zero is the deliverable.

- [ ] **Step 2: Write the findings doc**

`docs/2026-09-11-repo-cleanup-findings.md`, following the shape of `docs/2026-07-30-docs-and-skills-audit.md`: method (three parallel auditors, every claim verified against source, highest-stakes findings re-verified by hand), the finding tables from the spec's §2, what was done, what Phase 2 will cover, and a reproduce section. Record two things the spec did not know:

- One audit finding was misfiled. The `ddrs experiment` claim was reported at `ddrs-run/references/commands.md:245`; that file is 178 lines long and the text is at `ddrs-run/SKILL.md:245`. The finding held, the location did not. This is why every finding was re-checked by hand.
- `examples/leak_probe.rs` was nearly archived as a leakance artifact. It is the autograd-tape-leak repro, named in `traps.md` as a live trap's discriminating test and cited from `src/experiment/landscape/objective.rs`. Name-matching would have removed a working diagnostic; the admission rule in Task 13 is written against the citing findings doc for exactly this reason.

- [ ] **Step 2b: Record the nine process findings this execution produced**

These are the durable output of the run and they are NOT in the spec, because they were
discovered while executing it. Each one is a general lesson, verified by a specific
incident in this branch's history. Write them as a section of their own, with the
incident named so a reader can check it.

1. **The expensive errors were in the specification, not the work.** Across eight tasks
   the review loop caught roughly seven false claims that originated in the plan or its
   briefs, against about one implementer slip. Examples: the verifier's docstring
   claimed it caught drifted line numbers when it discards the line suffix entirely; a
   predicted post-compression line count of 480 that was wrong by 80 and would have
   shipped in the PR description; "nine validators" when `Config::from_yaml_file` runs
   ten; `src/bin/` described as four binaries when it holds twelve. The plan was written
   from a six-week-old audit plus one reading of the tree, and the tree had moved. The
   practice that caught these was re-verifying each task's premises against source
   immediately before dispatching it, not the post-hoc review.

2. **Compression can manufacture a factual error out of hedged prose.** The
   pre-compression text read "`Cr > 2` / `c3 < 0` (3.93 % to 0.31 %)", vague but not
   false. Dropping the second label to shorten it produced a precise, wrong attribution:
   `Cr > 2` is 2.10 % to 0.16 %. Shortening always-loaded context is not purely
   subtractive and needs the same numeric re-checking as new writing.

3. **Renaming a label breaks references that no gate and no keyword search can find.**
   Renumbering a duplicate `## T11` trap heading to `T13` orphaned
   `research-status.md:154`, which said "see traps.md T11" while describing the
   autodiff-tape trap. A keyword grep missed it because the line described the bug in
   different words than the trap's title, and `verify_doc_paths.py` is blind to it
   because `T11` is a label, not a path. Cross-references keyed on numbers need a
   by-subject sweep after any renumbering.

4. **Duplication propagates staleness, observed directly rather than inferred.**
   `ddrs-dev/SKILL.md:142` and `:144` carried the same two drifted citations that
   `CLAUDE.md` carried, because the skill had been written by copying `CLAUDE.md`. One
   stale line became two. This is the mechanism the whole cleanup is premised on, caught
   in the act.

5. **An audit misses what it does not think to look for, and a default flip's blast
   radius goes uncounted.** The three-auditor sweep graded `docs/setup.md` and
   `docs/intro.md` CLEAN and `docs/usage/running.md` MINOR. All three were wrong about
   `use_cuda_graphs`, and `setup.md` carried five stale claims. The auditors checked
   against the prior follow-up list and for newly-landed features; nobody swept for the
   2026-08-19 `ddr_match` default flip, so every consequence of that one change went
   unexamined. When a default changes, enumerate everything that asserts the old value.

6. **Verifying that a pointer resolves is not verifying that it delivers.** A fix round
   confirmed each status-block pointer's target existed, and the claim "all seven fields"
   was still false, because `.claude/REACH-SUBDIVISION.md` names five.
   `reference_discharge_coefficient` and `reference_discharge_exponent` appear nowhere in
   it. Check the promise, not just the address.

7. **Counts of sets that grow are a systematic defect class, not isolated slips.** Four
   instances in one repository: `CLAUDE.md`'s diagram naming 4 of 12 binaries,
   `docs/usage/running.md` claiming 10 of 12, a skill index enumerating 10 of 13 traps,
   and a skill summary listing 4 of 5 loss kinds. In every case the omitted members were
   the newest, so each index was stalest exactly where a reader most needed it. The fix
   that holds is to name families or defer to the authoritative list, never to correct
   one number to another. The repository already had this rule for test counts; it was
   not generalized.

8. **A gate's usefulness is set by its false-positive rate, not its coverage.** The
   first draft of `verify_doc_paths.py` reported 288 strict failures, almost all bare
   filenames and runtime artifacts. A gate that noisy gets switched off. Four changes
   took it to 9 actionable findings: requiring a repo-rooted path, skipping gitignored
   workspace paths, resolving against the citing file's directory, and testing existence
   rather than file-ness because an `.ic` store is a directory.

9. **Two citations were already pointing at the wrong code before conversion, which is
   the concrete cost of the drift.** `.claude/ARCHITECTURE.md`'s `src/routing/mmc.rs:289-294`
   had drifted onto an unrelated `denormalize` block, while the gate it described sits at
   370-372 inside `setup_inputs`. `gauge-population.md`'s `src/data/store/gage_csv.rs:62`
   had drifted to 64. The line-citation problem was not hypothetical at the time it was
   fixed.

- [ ] **Step 3: Record the new gate in `ddrs-dev/references/testing.md`**

Add `python3 scripts/verify_doc_paths.py` to the gate list, with one line on what it catches and the note that it must exit 0 on any change to `CLAUDE.md` or `.claude/skills/`.

- [ ] **Step 4: Commit**

```bash
git add docs/2026-09-11-repo-cleanup-findings.md .claude/skills/
git commit -m "$(cat <<'EOF'
docs: repository cleanup findings, phase 1

What was verified wrong in always-loaded context and what was corrected,
plus the two things the audit itself got wrong: one finding was reported
against the wrong file, and examples/leak_probe.rs was nearly archived as
a leakance artifact when it is the autograd-tape-leak repro named by a
live trap. Phase 2 appends what moved.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

No push and no PR here. This is a phase boundary inside one branch, and Task 15 Step 5 opens the single PR once Phase 2 is done. Leaving the branch unpushed until then keeps a half-restructured tree off the remote.

---

# Phase 2: restructure (Tasks 10-15)

No external gate: `landscape-deriv-objective` merged as PR #42 and the branch is already rebased onto `3412a78`. The only ordering requirement is internal, that Phase 1 completes first, so the link rewrite in Task 12 operates on text whose claims are already correct.

### Task 10: Split the book from the research record

**Files:**
- Modify: `book.toml`, `.github/workflows/docs.yml`
- Move: `docs/{intro,setup,architecture,algorithm,nh-qprime-store-contract,SUMMARY}.md`, `docs/usage/`, `docs/reference/` into `docs/book/`

- [ ] **Step 1: Capture the published file list before touching anything**

```bash
mdbook build && find target/book -type f | sort > /tmp/book-before.txt
wc -l /tmp/book-before.txt
```

- [ ] **Step 2: Move the book**

```bash
mkdir -p docs/book
git mv docs/SUMMARY.md docs/intro.md docs/setup.md docs/architecture.md \
       docs/algorithm.md docs/nh-qprime-store-contract.md docs/book/
git mv docs/usage docs/reference docs/book/
```

- [ ] **Step 3: Point the book at its new source**

In `book.toml`, change `src = "docs"` to `src = "docs/book"`, and
`edit-url-template` from `.../edit/master/docs/{path}` to `.../edit/master/docs/book/{path}`.

In `.github/workflows/docs.yml`, change both `docs/**` path filters to `docs/book/**` so research edits stop triggering a docs deploy.

- [ ] **Step 4: Check the book's internal links survived the move**

`mdbook build` validates `SUMMARY.md` entries but not inline links, and
`scripts/verify_doc_paths.py` cannot see a bare sibling link either: `is_citation`
rejects a token with no `/` before the sibling-directory fallback runs, so the 44
`](algorithm.md)`-style chapter links in the book are unverified by either gate. The
move keeps whole directories together so they should stay valid, but prove it with a
throwaway script:

```bash
python3 - <<'PYEOF'
import pathlib, re
bad = []
for md in pathlib.Path("docs/book").rglob("*.md"):
    for n, line in enumerate(md.read_text().splitlines(), 1):
        for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", line):
            if not (md.parent / target).exists():
                bad.append(f"{md}:{n}: {target}")
print("\n".join(bad) if bad else "all intra-book links resolve")
PYEOF
```

Expected: `all intra-book links resolve`. Any output is a link the move broke.

- [ ] **Step 5: Prove the published URLs did not change**

```bash
mdbook build && find target/book -type f | sort > /tmp/book-after.txt
diff /tmp/book-before.txt /tmp/book-after.txt
```

Expected: the diff shows ONLY removals, and every removal is a copied research asset (`2026-*.md`, `superpowers/`, `why-analysis/`, `figures/`, `journal/`). If any `.html` file appears or disappears, the URL surface changed: stop and investigate before committing.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
book: move the book under docs/book/ so its source holds only the book

book.toml had src = "docs", so mdBook copied 36 findings docs (592 KB),
docs/superpowers/ (1.5 MB) and docs/figures/ (7.2 MB) verbatim into the
published site as unlinked assets beside a 14-page book.

Published URLs are unchanged: mdBook output paths are relative to the
source root. Proven by diffing `find target/book -type f` before and
after; the diff is removals of copied research assets only.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 11: Create `research/` and move the record into it

**Files:**
- Move: 36 dated `docs/*.md`, `docs/superpowers/{specs,plans}/`, `docs/why-analysis/`, `docs/journal/`, `docs/figures/`, `.claude/specs/`
- Delete: `docs/images/.gitkeep`

- [ ] **Step 1: Move, preserving history**

```bash
mkdir -p research/findings research/specs research/plans research/figures
git mv docs/20*.md research/findings/
git mv docs/superpowers/specs/* research/specs/
git mv docs/superpowers/plans/* research/plans/
rmdir docs/superpowers/specs docs/superpowers/plans docs/superpowers
git mv docs/why-analysis research/why-analysis
git mv docs/journal research/journal
git mv docs/figures/* research/figures/
rmdir docs/figures
git rm docs/images/.gitkeep && rmdir docs/images 2>/dev/null || true
```

- [ ] **Step 2: Fold `.claude/specs/` into the same series**

It holds the sp1 through sp10 design and plan pairs dated 2026-05-17 to 2026-05-29, the same series `docs/superpowers/` continued from 2026-05-30.

```bash
git mv .claude/specs/*-design.md research/specs/
git mv .claude/specs/*-plan.md research/plans/
ls .claude/specs/   # expect empty; anything left is neither design nor plan, decide by hand
rmdir .claude/specs
```

- [ ] **Step 3: Verify nothing was dropped**

```bash
ls research/findings | wc -l   # expect 36
ls research/specs research/plans | wc -l
git status --short | grep -c "^R"   # renames, not deletes-plus-adds
```

- [ ] **Step 4: Commit the moves alone, before any link rewriting**

A commit containing only renames lets git's rename detection work cleanly, which keeps `git log --follow` usable for every findings doc.

```bash
git commit -m "$(cat <<'EOF'
research: gather the record into research/, out of the book's source tree

findings/ (36 dated docs), specs/, plans/, why-analysis/, figures/,
journal/. .claude/specs/ folds in: it is the same spec-and-plan series as
docs/superpowers/, split only by the date the convention changed.

Renames only. Link rewriting is the next commit, so rename detection stays
clean and git log --follow keeps working for every findings doc.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 12: Rewrite the 414 references

**Files:** about 150, across `CLAUDE.md`, `.claude/`, `src/`, `tests/`, `config/experiments/`, `scripts/`, `experiments/`, `examples/`, `research/`, `docs/book/`, `README.md`

- [ ] **Step 1: Snapshot the reference set**

```bash
grep -rIn -e "docs/20" -e "docs/superpowers" -e "docs/why-analysis" \
          -e "docs/figures" -e "docs/journal" \
  --include=*.md --include=*.rs --include=*.yaml --include=*.toml \
  --include=*.sh --include=*.py . > /tmp/refs-before.txt
wc -l /tmp/refs-before.txt   # expect about 414
```

- [ ] **Step 2: Rewrite in bulk**

```bash
files=$(cut -d: -f1 /tmp/refs-before.txt | sort -u)
for f in $files; do
  sed -i \
    -e 's|docs/superpowers/specs/|research/specs/|g' \
    -e 's|docs/superpowers/plans/|research/plans/|g' \
    -e 's|docs/why-analysis/|research/why-analysis/|g' \
    -e 's|docs/journal/|research/journal/|g' \
    -e 's|docs/figures/|research/figures/|g' \
    -e 's|docs/\(20[0-9][0-9]-\)|research/findings/\1|g' \
    "$f"
done
```

- [ ] **Step 3: Read every rewrite outside the moving set by hand**

The bulk edit is safe inside `research/`, where everything moved together. Outside it, a prefix substitution cannot distinguish a live pointer from a historical quotation, so read these:

```bash
grep -vE '^\./research/' /tmp/refs-before.txt | cut -d: -f1 | sort | uniq -c | sort -rn
git diff -- CLAUDE.md .claude/ src/ tests/ config/ scripts/ experiments/ examples/ README.md docs/
```

Roughly 130 hits, concentrated in `.claude/skills/ddrs-dev/references/research-status.md` (26), `CLAUDE.md` (16), about 45 `config/experiments/*.yaml` comments and about 16 `src/*.rs` doc comments. For each, confirm it is a live pointer. If a findings doc quotes a path as it stood at the time, revert that one hunk and leave the historical text intact.

- [ ] **Step 4: Verify every rewritten path resolves**

```bash
python3 scripts/verify_doc_paths.py
grep -rIn -e "docs/20" -e "docs/superpowers" -e "docs/why-analysis" \
          -e "docs/figures" -e "docs/journal" \
  --include=*.md --include=*.rs --include=*.yaml --include=*.toml \
  --include=*.sh --include=*.py . \
  | grep -v 2026-09-11-repo-cleanup
cargo check --examples --tests
mdbook build
```

Expected: verifier exits 0; the grep returns nothing except this plan and spec, which quote the old paths deliberately; cargo and mdbook both succeed. The cargo run is what proves the 16 `src/` doc-comment edits were comment-only.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
docs: repoint 414 references at research/

Four anchored prefixes. Referrers inside research/ moved together and were
rewritten in bulk; the ~130 outside it were read individually, because a
prefix substitution cannot tell a live pointer from a historical quotation
and rewriting a quotation makes the record say something never true.

Touches 16 src/ doc comments. Comment-only: cargo check --examples --tests
passes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 13: Archive the closed campaigns

The admission rule: **an artifact lands in the archive when the findings doc or plan that cites it records its campaign as closed, or when nothing in the repository cites it at all.** Name-matching is not the rule, because `examples/leak_probe.rs` matches "leak" and is a live diagnostic for the autograd-tape trap.

**Files:**
- Create: `research/archive/README.md`, `research/archive/scripts/`, `research/archive/examples/`
- Move: 26 scripts, 13 examples
- Modify: `docs/book/usage/running.md` (its example table), `src/data/dataset.rs` and `src/pretrain/mod.rs` (one doc-comment path each)

- [ ] **Step 1: Prove nothing live depends on the candidates**

```bash
grep -rlE "disagg_boundary_verification|disagg_precip_normalization|disagg_transfer_diagnostic|kan_disagg_|kan_sensitivity_sweep|pretrain_disagg_.*_compare|pretrain_disagg_verify|pretrain_reconciliation_check|save_random_kan" src tests .githooks
```

Expected: exactly `src/data/dataset.rs` and `src/pretrain/mod.rs`, both doc comments citing an experiment as evidence for a documented behavior. Those two paths get rewritten in Step 4. Any other hit means the artifact is live: leave it in place.

Confirm `leak_probe` stays:

```bash
grep -rn "leak_probe" .claude/skills/ src/
```

Expected: `traps.md` names it as a discriminating test and `src/experiment/landscape/objective.rs` cites it. It is NOT archived.

- [ ] **Step 2: Move the 26 scripts**

```bash
mkdir -p research/archive/scripts
git mv scripts/{leakance_diagnosis,leakance_subset_analysis,zeta_gradient_analysis,zeta_probe_sites,floor_analysis}.py research/archive/scripts/
git mv scripts/{equif_convergence_analysis,h5_h6_audit_analysis,wave_comparison_plots,parameter_landscape}.py research/archive/scripts/
git mv scripts/run_equif_arms.sh research/archive/scripts/
git mv scripts/{recoverability_analysis,recoverability_sites,synthetic_n_consensus_geometry,synthetic_n_recoverability_analysis,synthetic_n_truth_fields}.py research/archive/scripts/
git mv scripts/tau_sweep.py research/archive/scripts/
git mv scripts/run_tau9_{hourly_eval_chain,remaining,source_trains}.sh research/archive/scripts/
git mv scripts/run_tau_{interp,source}_arms.sh research/archive/scripts/
git mv scripts/{sp8_check_scatter,sp10_check_launches}.sh research/archive/scripts/
git mv scripts/{generate_run_notebooks,hydrograph_comparison,merge_merit_basins}.py research/archive/scripts/
```

- [ ] **Step 3: Move the 13 examples**

```bash
mkdir -p research/archive/examples
git mv examples/disagg_{boundary_verification,precip_normalization_discriminator,transfer_diagnostic}.rs research/archive/examples/
git mv examples/kan_disagg_{mass_balance_real,real_storm_shift,trained_sensitivity}.rs research/archive/examples/
git mv examples/pretrain_disagg_{capacity_storm_compare,storm_compare,verify,window72_storm_compare}.rs research/archive/examples/
git mv examples/{pretrain_reconciliation_check,save_random_kan,kan_sensitivity_sweep}.rs research/archive/examples/
ls examples/*.rs
```

Expected: exactly `benchmark_hydrograph.rs`, `compare_ddr_sandbox.rs`, `dump_init_params.rs`, `leak_probe.rs`.

- [ ] **Step 4: Repoint the two `src/` doc comments and the running.md table**

`src/data/dataset.rs` cites `examples/disagg_precip_normalization_discriminator.rs`; `src/pretrain/mod.rs` cites `examples/pretrain_reconciliation_check.rs`. Prefix both with `research/archive/`. In `docs/book/usage/running.md`, delete the rows for the archived examples from the example table.

- [ ] **Step 5: Write the archive README**

`research/archive/README.md` states the admission rule, records per artifact its closing document and the commit at which it last built, and warns that archived examples are no longer compiled by `cargo check --examples`, so they may not build against current APIs. Use this mapping, which was verified by grep:

| Artifact group | Closing document |
|---|---|
| `leakance_diagnosis.py` | `research/findings/2026-07-02-leakance-diagnosis-findings.md` |
| `leakance_subset_analysis.py` | `research/findings/2026-07-01-leakance-hourly-experiment-handoff.md` |
| `zeta_gradient_analysis.py`, `zeta_probe_sites.py` | `research/findings/2026-07-03-zeta-gradient-probe-findings.md` |
| `floor_analysis.py` | `research/plans/2026-07-04-phase-b2-state-cache.md` |
| `equif_convergence_analysis.py`, `run_equif_arms.sh` | `research/findings/2026-07-07-lstm-equifinality-findings.md` |
| `h5_h6_audit_analysis.py` | `research/findings/2026-07-09-h5-h6-equifinality-v2-findings.md` |
| `wave_comparison_plots.py` | `research/findings/2026-07-16-wave2-cross-wave-findings.md` |
| `recoverability_analysis.py`, `recoverability_sites.py` | `research/findings/2026-07-04-synthetic-recoverability-findings.md` |
| `synthetic_n_recoverability_analysis.py` | `research/findings/2026-07-22-synthetic-n-recoverability-findings.md` |
| `synthetic_n_consensus_geometry.py`, `synthetic_n_truth_fields.py` | `research/plans/2026-07-22-synthetic-n-recoverability.md` |
| `tau_sweep.py`, `run_tau9_*.sh`, `run_tau_*_arms.sh` | `research/findings/2026-08-06-tau-sweep-pilot-findings.md` |
| `sp8_check_scatter.sh` | `research/specs/2026-05-22-sp8-mc-timestep-fusion-design.md` |
| `sp10_check_launches.sh` | `research/specs/2026-05-26-sp10-cuda-graphs-design.md` |
| `kan_disagg_*.rs` | `research/findings/2026-07-16-disagg-head-sensitivity-findings.md` |
| `parameter_landscape.py`, `generate_run_notebooks.py`, `hydrograph_comparison.py`, `merge_merit_basins.py` | none: no inbound reference anywhere. `parameter_landscape.py` is the H6-era equifinality instrument (R1/R2/R3 arms, `output/equif/`), not the current `experiments/landscape/` study |
| `disagg_*.rs`, `pretrain_disagg_*.rs`, `pretrain_reconciliation_check.rs`, `save_random_kan.rs`, `kan_sensitivity_sweep.rs` | none beyond the two `src/` doc comments repointed in Step 4 |

- [ ] **Step 6: Verify**

```bash
cargo check --examples --tests
python3 scripts/verify_doc_paths.py
```

Expected: both pass. `cargo check` now compiles four examples instead of seventeen.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
research/archive: retire 26 scripts and 13 examples from closed campaigns

Admission rule, recorded in research/archive/README.md: an artifact lands
here when the findings doc or plan citing it records its campaign closed,
or when nothing in the repository cites it. Not name-matching:
examples/leak_probe.rs matches "leak" and stays, because it is the
autograd-tape-leak repro that traps.md names as a live discriminating test.

Campaigns: leakance and the zeta probe, equifinality, synthetic-n
recoverability, the tau sweep, the sp8/sp10 spike checks, and the
disaggregation pretraining probes. Four had no inbound reference at all.

Cargo declares no [[example]] entries and no test depends on any of these,
so nothing gated on them. cargo check now builds 4 examples, not 17.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 14: Untrack `output/`

Eight files are tracked inside a directory `.gitignore` excludes, cited by the synthetic-n findings doc from a campaign that stopped 2026-07-29.

**Files:**
- Move: `output/synthetic_n/plots/*` to `research/figures/synthetic-n/`
- Modify: `research/findings/2026-07-22-synthetic-n-recoverability-findings.md` (one path)

- [ ] **Step 1: Confirm the state**

```bash
git ls-files output/
git check-ignore -v output/
```

Expected: eight files listed, and `output/` shown as ignored. That combination is the artifact: they were force-added.

- [ ] **Step 2: Move them into the record**

```bash
mkdir -p research/figures/synthetic-n
git mv output/synthetic_n/plots/* research/figures/synthetic-n/
git ls-files output/   # expect empty
```

- [ ] **Step 3: Repoint the one live citation**

In `research/findings/2026-07-22-synthetic-n-recoverability-findings.md` line 36, change `output/synthetic_n/plots/synthetic_n_recovery_distributed.ipynb` to `research/figures/synthetic-n/synthetic_n_recovery_distributed.ipynb`. Leave line 30's `output/synthetic_n/run_students_sequential.sh` alone: that is a resume command for a run that never happened, a historical quotation, and the file does not exist.

- [ ] **Step 4: Verify and commit**

```bash
git ls-files output/
python3 scripts/verify_doc_paths.py
git commit -m "$(cat <<'EOF'
research/figures: untrack the synthetic-n plots from gitignored output/

Eight files force-added inside a directory .gitignore excludes, from the
synthetic-n campaign that stopped 2026-07-29. They are results figures, so
they belong with the record. The findings doc's live citation is repointed;
its resume-command line is left as the historical quotation it is.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

### Task 15: Final gates, skill update, and open the single PR

**Files:**
- Modify: `.claude/skills/ddrs-dev/SKILL.md` and its references (the new layout), `CLAUDE.md` ("When in doubt" paths)
- Modify: `research/findings/2026-09-11-repo-cleanup-findings.md` (append the Phase 2 section)

- [ ] **Step 1: Update `ddrs-dev` and `CLAUDE.md` for the new layout**

Per the repository's rule that the skills library is the reproducibility source of truth, record: findings live in `research/findings/`, specs and plans in `research/specs/` and `research/plans/`, the book is `docs/book/` with `mdbook build` reading `book.toml`'s `src = "docs/book"`, and closed-campaign tooling is in `research/archive/` under the admission rule stated in its README.

- [ ] **Step 2: Run the full gate set for the whole branch**

```bash
python3 scripts/test_verify_doc_paths.py
python3 scripts/verify_doc_paths.py
cargo check --examples --tests
cargo test --test ddr_sandbox_match --test gridded_bundle
mdbook build && find target/book -type f | sort > /tmp/book-final.txt
diff /tmp/book-before.txt /tmp/book-final.txt
```

Expected: every command passes, the verifier exits 0 with `0 unresolved in agent context`, and the book diff shows only removals of copied research assets. `/tmp/book-before.txt` is the snapshot Task 10 Step 1 captured.

- [ ] **Step 3: Confirm no old path survives anywhere**

```bash
grep -rIn -e "docs/20" -e "docs/superpowers" -e "docs/why-analysis" \
          -e "docs/figures" -e "docs/journal" -e "\.claude/references" \
  --include=*.md --include=*.rs --include=*.yaml --include=*.toml \
  --include=*.sh --include=*.py . \
  | grep -v 2026-09-11-repo-cleanup \
  | grep -v 2026-07-30-docs-and-skills-audit
```

Expected: no output. The two excluded files are this cleanup's own spec, plan and findings doc, and the 2026-07-30 audit, all of which name the old paths as history.

- [ ] **Step 4: Append the Phase 2 section to the findings doc and commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
docs: repository cleanup findings, phase 2

What moved, what the verifier now prevents, and the admission rule the
archive is held to.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

- [ ] **Step 5: Push and open the one PR**

```bash
git push -u origin worktree-repo-cleanup
gh pr create --title "Repository cleanup: true agent context, a book-only docs tree, and an archived campaign record" --body "$(cat <<'EOF'
Two phases in one PR. Phase 1 is content-only and renames nothing; Phase 2
does the restructure. Reviewing them as separate commit ranges is easier
than reviewing the combined diff: Phase 1 ends at the "repository cleanup
findings, phase 1" commit.

## Phase 1: every claim in agent context is now true

Three parallel auditors verified every concrete claim in `CLAUDE.md`, the
four skills, and the 14 book pages against source. Nine claims in
`CLAUDE.md` and seven in the skills were verified false, including two
places where the skill library contradicted itself:

- `nse-batch`, `optimizer`, `use_grad_accum` and `grad_accum_steps` were
  documented as unlanded work on branch `exp_train`; all four are on
  master with passing tests
- `ddrs-run` said `ddrs experiment` is not on master while `ddrs-dev`
  documented the same command as working
- a skill asserted `.claude/references/` had been deleted; its 12 files
  were still there, and now are not
- the architecture diagram showed `sparse.rs` as a file and
  `spike_backward/` as present, and omitted eight subsystems
- three `src/` line citations had drifted by 26, 2 and 136 lines

Two items from the 2026-07-30 audit's P1 list are fixed here, and both had
broken *after* that audit: `merit_training.yaml` ships
`use_cuda_graphs: false` (flipped 2026-08-19), and `tau`'s default is 9,
not 3 (changed 2026-08-08, with 3 on the retired scale).

`CLAUDE.md` sheds the two closed NO-GO campaigns' operational detail,
which keeps their verdict and do-not-reopen reasoning in always-loaded
context while the enable steps and gate commands move into `ddrs-dev`.
Quote the MEASURED before and after line counts here, from Task 5's
report. Do not quote a predicted number.

New gate: `scripts/verify_doc_paths.py` fails on any unresolved citation
in agent-loaded context. It is seeded against the known-bad set, so a
false negative cannot pass silently. An earlier draft reported 288
failures, almost all noise; the tuning that took it to 9 is recorded in
the plan.

## Phase 2: the book tree holds only the book

`book.toml` had `src = "docs"`, so mdBook copied 36 findings docs,
`docs/superpowers/` and `docs/figures/` (9.3 MB total) into the published
site as unlinked assets beside a 14-page book. It now points at
`docs/book/`. Published URLs are unchanged, proven by diffing
`find target/book -type f` before and after.

`research/` holds findings, specs, plans, why-analysis, figures, journal
and archive. `.claude/specs/` folds in: the same spec-and-plan series as
`docs/superpowers/`, split only by the date the convention changed.

26 scripts and 13 examples retired to `research/archive/` under a stated
admission rule: an artifact lands there when the findings doc or plan
citing it records its campaign closed, or when nothing cites it at all.
Not name-matching, and that distinction caught a real mistake:
`examples/leak_probe.rs` matches "leak" but is the autograd-tape-leak
repro that `traps.md` names as a live trap's discriminating test. It stays.

414 references repointed. The roughly 130 outside the moving set were read
individually rather than swept, because a prefix substitution cannot tell
a live pointer from a historical quotation.

## Out of scope, deliberately

`experiments/` is untouched. Its bundles are the active landscape and
adjoint workstream, not residue. No behavior change anywhere in `src/`:
the only source edits are doc-comment path strings, and
`cargo check --examples --tests` is the proof.

Spec: `research/specs/2026-09-11-repo-cleanup-design.md`
Plan: `research/plans/2026-09-11-repo-cleanup.md`
Findings: `research/findings/2026-09-11-repo-cleanup-findings.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01EqnrtmU91yzuVpGnADnXLP
EOF
)"
```

---

## Self-review notes

**Spec coverage.** §4.1 to Tasks 4 and 5; §4.2 to Task 6; §4.3 to Task 7; §4.4 to Task 8; §4.5 to Tasks 2 and 3; §4.6 to Tasks 1 and 9; §5.1 to Task 10; §5.2 to Task 11; §5.3 to Task 12; §5.4 to Task 13; §5.5 to Task 14; §5.6 is a constraint, enforced by no task touching `experiments/`; §5.7 to Task 15; §8 deliverables to Tasks 1, 9, 13 and 15. §3's one-PR sequencing to Task 15 Step 5, which is the only place a PR is opened.

**One spec correction, made here.** §5.4 listed `examples/leak_probe.rs` for archiving. It is the autograd-tape-leak repro, named by `traps.md` as a live trap's discriminating test and cited from `src/experiment/landscape/objective.rs`. Task 13 keeps it and Step 1 asserts so.

**Task 1's code was executed, not just written.** Both files were extracted verbatim from this plan and run: the test suite passes all fourteen assertions, and the verifier reports exactly the nine strict failures Task 1 Step 5 predicts. An earlier draft reported 288, almost all noise, which is why the `ROOTS` predicate and the sibling-directory resolution are in the final version.
