# ddrs documentation system — design spec

**Date:** 2026-05-29
**Status:** Design — awaiting plan
**Audience:** dual-purpose — published documentation site on GitHub Pages
plus reusable agent skills for future coding agents working in this repo.

## Motivation

Today the repo has:
- A nearly-empty `README.md`.
- A single agent-focused `CLAUDE.md`.
- One existing skill at `.claude/skills/burn_custom_backward.md`.
- No `docs/`, no Pages site, no contributor onboarding path.

A new contributor (human or agent) opening this repo has no narrative on
what the project is, how to set it up, how to read its inputs, how to
construct graph objects, how to interpret outputs, or how to compare against
the DDR reference. We need a docs system that serves both readers.

## Goals

1. **Single source of truth.** Each concept lives in one canonical file —
   a `.claude/skills/ddrs-*.md` — that is compressed, instructional, and
   agent-readable.
2. **Published mdBook site.** A polished, narrative mdBook published to
   GitHub Pages. Each chapter is the expanded form of one canonical skill.
3. **AI-expanded build, karpathy-style.** A `/regenerate-docs` meta-skill
   reads the canonical skills + current repo state and rewrites the affected
   mdBook chapters. Run pre-PR. Drift between source and published site is
   bounded by the PR review cycle, not by deterministic templating.
4. **Rust-idiomatic toolchain.** mdBook + cargo-installed preprocessors.
   No Python dependency for contributors who just want to read the docs;
   the build runs in CI.
5. **No new mandatory tooling at first.** CI consistency check (does
   `/regenerate-docs` reproduce committed `docs/`?) deferred to Phase 2.

## Non-goals

- API reference auto-generation (rustdoc already covers this; we link to it
  rather than re-rendering).
- Versioned docs for old releases (single live version).
- Search backend beyond mdBook's built-in.
- Custom domain (`taddyb.github.io/ddrs` is fine).
- Notebook gallery / Jupyter rendering.

## Architecture

### Source-of-truth hierarchy

```
.claude/skills/ddrs-*.md      (canonical, compressed, agent-readable)
            │
            │   /regenerate-docs reads, expands, writes
            ▼
docs/**.md                    (narrative, human-readable, mdBook chapters)
            │
            │   mdbook build  (in CI)
            ▼
target/book/                  (gitignored — built site)
            │
            │   actions/deploy-pages@v4
            ▼
taddyb.github.io/ddrs         (published)
```

Drift prevention:
- **Pre-PR ritual:** contributor invokes `/regenerate-docs` after editing
  any canonical skill. The meta-skill rewrites affected chapters; the PR
  contains both the skill and the regenerated chapter.
- **PR CI build-only:** mdBook builds the proposed `docs/` to catch broken
  references, missing chapters, malformed SUMMARY.md.
- **Future work (deferred):** add a CI step that re-runs `/regenerate-docs`
  in dry-run mode and fails if the result differs significantly from what
  was committed. Deferred because non-determinism complicates the diff.

### Canonical skill set

12 files under `.claude/skills/`. 11 canonical + 1 meta-skill.

| Skill | Chapter target | Sources it reads |
|-------|----------------|------------------|
| `ddrs-setup.md` | `docs/setup.md` | `Cargo.toml`, `CLAUDE.md`, README, data file paths, cubecl fork branch |
| `ddrs-architecture.md` | `docs/architecture.md` | `src/`, `.claude/ARCHITECTURE.md` |
| `ddrs-algorithm.md` | `docs/algorithm.md` | `src/routing/`, `src/geometry.rs`, DDR reference |
| `ddrs-running-the-code.md` | `docs/usage/running.md` | `src/bin/`, `examples/` |
| `ddrs-reading-inputs.md` | `docs/usage/inputs-reading.md` | `src/data/store/` |
| `ddrs-formatting-inputs.md` | `docs/usage/inputs-formatting.md` | `src/config.rs`, `config/merit_training.yaml` |
| `ddrs-graph-objects.md` | `docs/usage/graph-objects.md` | `src/sparse/mod.rs`, `src/data/ids.rs`, `src/routing/mmc.rs` |
| `ddrs-reading-outputs.md` | `docs/usage/outputs.md` | `examples/compare_ddr_sandbox.rs`, training checkpoint format |
| `ddrs-comparing-to-ddr.md` | `docs/reference/ddr-comparison.md` | `examples/compare_ddr_sandbox.rs`, `scripts/export_ddr_sandbox.py` |
| `ddrs-perf-and-cuda-graphs.md` | `docs/reference/perf.md` | `.claude/ARCHITECTURE.md` SP-7..SP-10 sections, `src/cuda_graph/` |
| `ddrs-burn-autograd.md` | `docs/reference/burn-autograd.md` | renamed from `burn_custom_backward.md`; `src/sparse/mod.rs`, `src/routing/mmc_op.rs` |
| `regenerate-docs.md` | (no chapter — meta-skill) | reads all of the above |

Each canonical skill carries YAML frontmatter:

```yaml
---
name: ddrs-running-the-code
description: How to build, train, evaluate, and run the regression examples.
output: usage/running.md
sources:
  - src/bin/train.rs
  - src/bin/eval.rs
  - src/bin/train_and_test.rs
  - examples/compare_ddr_sandbox.rs
  - examples/benchmark_hydrograph.rs
---
```

The `output` field tells the meta-skill where to write the expanded chapter.
The `sources` field defines what files Claude reads when expanding, AND drives
selective regeneration: a skill is only re-expanded if either the skill itself
or any of its `sources` changed since the last regeneration (detected via
`git log` from the meta-skill).

### mdBook source layout

```
ddrs/
├── book.toml                          NEW
├── docs/                              NEW (mdBook source — overrides default `src/`)
│   ├── SUMMARY.md                     written by /regenerate-docs
│   ├── intro.md                       landing page (project pitch, dataflow)
│   ├── setup.md
│   ├── architecture.md
│   ├── algorithm.md
│   ├── usage/
│   │   ├── running.md
│   │   ├── inputs-reading.md
│   │   ├── inputs-formatting.md
│   │   ├── graph-objects.md
│   │   └── outputs.md
│   ├── reference/
│   │   ├── ddr-comparison.md
│   │   ├── perf.md
│   │   └── burn-autograd.md
│   ├── images/                        diagrams, sandbox plots
│   └── theme/                         optional — custom CSS/logo only if needed
└── target/book/                       gitignored (under existing target/ rule)
```

`book.toml`:

```toml
[book]
title = "ddrs"
authors = ["Tadd Bindas"]
description = "BURN-based differentiable Muskingum-Cunge routing in Rust"
src = "docs"
language = "en"

[build]
build-dir = "target/book"
create-missing = false

[preprocessor.katex]
no-css = false

[preprocessor.mermaid]
command = "mdbook-mermaid"

[output.html]
default-theme = "rust"
preferred-dark-theme = "ayu"
git-repository-url = "https://github.com/taddyb/ddrs"
edit-url-template = "https://github.com/taddyb/ddrs/edit/master/docs/{path}"
mathjax-support = false                # use mdbook-katex instead

[output.html.search]
enable = true
limit-results = 30
use-boolean-and = true
```

### `/regenerate-docs` meta-skill behavior

The meta-skill's body instructs Claude to:

1. **Read inventory.** List `.claude/skills/ddrs-*.md`. For each, parse the
   frontmatter to get `output`, `sources`, and `name`.

2. **Detect what changed.** For each canonical skill:
   - Run `git log --format=%H --no-merges -- <skill_path> <source1> <source2> ...`.
   - Compare against the last commit recorded in
     `.claude/skills/.regenerate-state.json` (a small state file the
     meta-skill maintains, mapping skill name → last-regenerated commit SHA).
   - Skill needs regeneration if any new commit appears.

3. **Expand each changed skill.** For each:
   - Read the canonical skill body.
   - Read each file in `sources`.
   - Read the last ~20 lines of `git log --oneline -- <sources>` for context
     on recent changes worth surfacing.
   - Write a narrative mdBook chapter to `docs/<output>` that expands the
     skill's compressed instructions into prose, includes runnable code
     blocks, diagrams (mermaid for dataflow, KaTeX for math), and
     cross-references to other chapters.

4. **Inline directives.** Canonical skills may contain expansion hints:
   - `<!-- expand-with: examples/compare_ddr_sandbox.rs -->` inlines the
     referenced file as a code block.
   - `<!-- diagram: dataflow -->` triggers a mermaid block from a fixed
     library of named diagrams stored in the meta-skill.
   - `<!-- cross-ref: architecture, usage/inputs-reading -->` produces
     anchor links to other chapters.

5. **Regenerate SUMMARY.md** from a fixed chapter order embedded in the
   meta-skill. Every skill with an `output` entry contributes one SUMMARY
   line. Manual edits to SUMMARY.md will be overwritten; warn contributors
   in the meta-skill body.

6. **Sanity-check.**
   - Verify every `[link](path)` in SUMMARY.md resolves to a file.
   - Run `mdbook build` (capture stderr, report any KaTeX / mermaid /
     markdown errors).

7. **Update state.** Write the current HEAD SHA per regenerated skill into
   `.claude/skills/.regenerate-state.json`.

8. **Report.** Print a summary: which chapters were rewritten, which were
   skipped (unchanged), any sanity-check errors. The contributor `git diff`s
   `docs/` before staging.

The meta-skill is NOT autonomous. It is invoked on demand via
`/regenerate-docs` (slash-command). It does not run on save, on commit, or
in CI. The contract is: skills + AI = chapters, refreshed on contributor
intent before each PR.

### CI / delivery

`.github/workflows/docs.yml`:

```yaml
name: docs
on:
  push:
    branches: [master]
    paths: [docs/**, book.toml, .github/workflows/docs.yml]
  pull_request:
    paths: [docs/**, book.toml, .github/workflows/docs.yml]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: docs-mdbook
      - name: Install mdbook + preprocessors
        run: |
          cargo install --locked mdbook
          cargo install --locked mdbook-katex
          cargo install --locked mdbook-mermaid
      - name: Build
        run: mdbook build
      - uses: actions/upload-pages-artifact@v3
        if: github.ref == 'refs/heads/master'
        with:
          path: target/book

  deploy:
    needs: build
    if: github.ref == 'refs/heads/master'
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

One-time setup (Task 1 of implementation):
- Repo Settings → Pages → Source: **GitHub Actions** (not branch).
- No `gh-pages` branch needed; `actions/deploy-pages@v4` is branchless.

Pre-PR contributor flow:

```
1. Edit one or more canonical skills in .claude/skills/ddrs-*.md
2. /regenerate-docs           (Claude expands affected chapters)
3. git diff docs/             (review expansion)
4. git add .claude/skills/ docs/ .claude/skills/.regenerate-state.json
5. git commit + gh pr create
6. PR CI builds the book; reviewer can preview from the artifact
7. Merge → docs.yml deploys to taddyb.github.io/ddrs
```

## Concerns

1. **Non-determinism.** Two runs of `/regenerate-docs` won't produce
   byte-identical chapters. The PR review is the gate, not a diff check.
   Future work may add a softer CI check (e.g., "skills updated but docs
   unchanged?" warning) but the initial implementation ships without it.

2. **Hallucination risk in expansion.** Claude could invent API details
   that don't exist. Mitigations:
   - Each canonical skill cites file paths and line numbers explicitly.
   - The meta-skill instructs Claude to verify any claim against the
     `sources` files via Read tool before asserting in the chapter.
   - `mdbook test` (future addition) runs code blocks tagged
     ` ```rust,ignore ` through `rustc --edition 2021` syntax-check; catches
     hallucinated function signatures that don't parse.

3. **Drift between canonical skill and chapter prose.** Even with the
   pre-PR ritual, a reviewer who edits the chapter directly (without
   updating the skill) introduces drift. Documented convention: chapters
   are write-once-by-AI; edits go to the skill, then regenerate.

4. **Selective regeneration false negatives.** If a canonical skill cites a
   file not in its `sources` list, changes to that file won't trigger
   regeneration. Mitigation: contributors update `sources` when adding
   cross-references; reviewers spot-check.

5. **Initial expansion cost.** First-time `/regenerate-docs` expands 11
   chapters from scratch. Each expansion is one Claude turn reading
   ~500-2000 lines of source + writing ~1000 lines of prose. Total: probably
   10-30 minutes of subagent work for the initial run. Subsequent runs
   regenerate only changed chapters (typically 1-3 per PR).

6. **Preprocessor version drift.** mdbook-katex and mdbook-mermaid release
   independently. Pinning via `cargo install --locked` mitigates surprise
   breakage. Update only when a build fails in CI.

7. **Existing `burn_custom_backward.md` rename.** `CLAUDE.md` references
   it at line ~94 (`.claude/skills/burn_custom_backward.md`). The rename
   to `ddrs-burn-autograd.md` requires a one-line `CLAUDE.md` update in the
   same commit.

## Assumptions

- Repo stays on `master` as the default branch; published site only updates
  from `master` pushes.
- Contributors use `claude` (or compatible CLI with the skill system) to
  invoke `/regenerate-docs`. Without the CLI, they edit `docs/` directly —
  drift becomes possible but the build still ships.
- `taddyb.github.io/ddrs` is the live URL. If a custom domain is added
  later, a `docs/CNAME` file lands at that point.
- DDR documentation at `~/projects/ddr/docs/` is read-reference only for
  visual cues and content depth; ddrs docs are written independently.
- `book.toml`'s `src = "docs"` override is honored by mdbook 0.4.40+; the CI
  cargo install gives us the latest stable. If we ever need a pinned
  version, set `cargo install --locked --version X.Y.Z mdbook`.

## Tests / verification

There is no test suite for a documentation system per se. Verification is
gate-based:

| Gate | Verification |
|------|--------------|
| D1 — mdBook builds | `mdbook build` from a clean checkout succeeds with zero warnings. |
| D2 — All chapters reachable | Every `output:` in canonical skills exists at `docs/<output>` and is linked from `SUMMARY.md`. |
| D3 — No broken links | `mdbook-linkcheck` (added as Phase 2 preprocessor) passes. Phase 1 ships without; visual spot-check. |
| D4 — Site deploys | First push to `master` after the workflow lands deploys to `taddyb.github.io/ddrs`. |
| D5 — Pre-PR ritual works | `/regenerate-docs` on a no-change run reports "no chapters needed regeneration"; on a skill edit, regenerates only the affected chapter. |

## Implementation phases

Single implementation plan; phased internally by task ordering.

**Phase 1 — Infrastructure (no docs content yet).**
1. Land `book.toml` + empty `docs/SUMMARY.md` + `docs/intro.md` placeholder.
2. Land `.github/workflows/docs.yml`.
3. Enable GitHub Pages in repo settings.
4. Verify D4 (first deploy works) with the placeholder page.

**Phase 2 — Canonical skills.**
5. Rename `burn_custom_backward.md` → `ddrs-burn-autograd.md`; update
   `CLAUDE.md` reference. (Smallest skill change first to validate
   convention.)
6. Write each of the 11 canonical skills with frontmatter (`name`,
   `description`, `output`, `sources`) plus compressed instructional body.
7. Order: setup → architecture → algorithm → running → inputs-reading →
   inputs-formatting → graph-objects → outputs → ddr-comparison → perf
   → burn-autograd. (Onboarding order.)

**Phase 3 — Meta-skill.**
8. Write `.claude/skills/regenerate-docs.md` with the full expansion
   contract (Section: "`/regenerate-docs` meta-skill behavior" above).
9. Write the chapter-order definition and embedded mermaid diagram library
   inside the meta-skill.
10. Initialize `.claude/skills/.regenerate-state.json` empty.

**Phase 4 — Initial expansion.**
11. Invoke `/regenerate-docs` for the first time. Expand all 11 chapters
    from scratch.
12. Review `docs/` diff; spot-check each chapter against its source skill.
13. Run `mdbook build` locally; fix any KaTeX / mermaid errors.

**Phase 5 — Ship.**
14. Single PR: skills + docs + workflow + Pages enablement note.
15. Merge → CI builds + deploys.
16. Verify D1-D5.

**Out of scope for Phase 1 (deferred):**
- `mdbook test` syntax-check of code blocks.
- `mdbook-linkcheck` preprocessor.
- CI consistency check (does `/regenerate-docs` reproduce committed docs?).
- Custom theme / logo.
- Versioned docs.
