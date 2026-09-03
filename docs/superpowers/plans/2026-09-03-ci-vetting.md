# CI Vetting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GitHub Actions CI that vets every PR and every master push with the full debug test suite plus the release routing-acceptance gates, backed by branch protection and an opt-in local pre-push hook.

**Architecture:** One workflow (`ci.yml`) with two independent `ubuntu-latest` jobs — `test` (debug `cargo test --features fixtures`) and `acceptance` (release `compare_ddr_sandbox` example + `juniata_acceptance`). Both install a CUDA toolkit because `burn-cuda`/`cudarc` are non-optional compile-time deps. Branch protection requires both checks but leaves admin direct pushes possible (hybrid posture). Spec: `docs/superpowers/specs/2026-09-03-ci-vetting-design.md`.

**Tech Stack:** GitHub Actions, `dtolnay/rust-toolchain`, `Swatinem/rust-cache`, `Jimver/cuda-toolkit`, `gh` CLI, git `core.hooksPath`.

## Global Constraints

- No path filters on the workflow (required checks + path-skips = PRs stuck on "expected").
- Job names must be exactly `test` and `acceptance` — branch protection references them by name.
- Both jobs run on every `pull_request` and every push to `master`.
- No fmt/clippy gates; no `--features cuda` execution; no data-dependent test coverage (they self-skip).
- Pin all third-party actions to a tag.
- Commits end with the Co-Authored-By + Claude-Session trailer used throughout this session.
- The CUDA install step is the acknowledged iteration point: expect fixup commits on the bring-up PR; that is normal and planned.

---

### Task 1: Local pre-push hook

**Files:**
- Create: `.githooks/pre-push`

**Interfaces:**
- Produces: a committed, executable hook script; enabled per-clone via `git config core.hooksPath .githooks`. Task 6 documents it.

- [ ] **Step 0: Create the working branch (both tasks' commits ride the bring-up PR)**

```bash
git checkout -b ci-vetting
```

- [ ] **Step 1: Write the hook**

```bash
mkdir -p .githooks
cat > .githooks/pre-push <<'EOF'
#!/usr/bin/env bash
# Fast local gate before any push: the DDR sandbox parity test (invariant 1).
# Enable per-clone:  git config core.hooksPath .githooks
# Bypass once:       git push --no-verify
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
cargo test --test ddr_sandbox_match
EOF
chmod +x .githooks/pre-push
```

- [ ] **Step 2: Verify it runs and passes**

Run: `.githooks/pre-push`
Expected: `test sandbox_routes_to_absolute_match_against_ddr ... ok` and exit 0 (seconds on a warm build; up to ~1 min if the debug tree recompiles).

- [ ] **Step 3: Verify the failure path**

Run: `bash -c 'set -euo pipefail; false; echo unreachable'; echo "exit: $?"`
Expected: `exit: 1` — confirms `set -e` semantics the hook relies on. (Do not break the real test to check this; the hook is a thin wrapper and `cargo test`'s nonzero exit propagates through `set -e`.)

- [ ] **Step 4: Commit**

```bash
git add .githooks/pre-push
git commit -m "chore(hooks): opt-in pre-push hook running the DDR sandbox parity gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RHusoRV4Th4tio8bh2oPca"
```

---

### Task 2: The CI workflow file

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the repo's existing gates — `tests/ddr_sandbox_match.rs`, `tests/juniata_acceptance.rs` (release-only), `examples/compare_ddr_sandbox.rs` (exits 1 on mismatch), KAN fixture tests behind `--features fixtures`.
- Produces: status checks named `test` and `acceptance` (Task 5 requires them by these exact names).

- [ ] **Step 1: Pin the CUDA action tag**

Run: `gh api repos/Jimver/cuda-toolkit/tags --jq '.[].name' | head -5`
Pick the newest `v0.2.x` tag and use it in Step 2 (the YAML below says `v0.2.19`; replace if a newer v0.2.x exists).

- [ ] **Step 2: Write the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  pull_request:
  push:
    branches: [master]
  workflow_dispatch:

# No path filters: a required check skipped by a path filter leaves PRs
# stuck on "expected — waiting for status". Doc-only runs are cheap warm.
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

# Branch protection (applied once, after the first green master run):
#   gh api -X PUT repos/taddyb/ddrs/branches/master/protection --input - <<'EOF'
#   {
#     "required_status_checks": {
#       "strict": false,
#       "checks": [{"context": "test"}, {"context": "acceptance"}]
#     },
#     "enforce_admins": false,
#     "required_pull_request_reviews": null,
#     "restrictions": null
#   }
#   EOF

jobs:
  # Full default suite, debug. Data-dependent tests self-skip on missing
  # paths; --features cuda tests are compiled out; juniata_acceptance
  # self-skips under debug_assertions. `fixtures` enables the KAN parity
  # sweep (tests/fixtures/ is tracked).
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      # plotters (examples) needs fontconfig headers on Linux; cargo test
      # compiles examples even though it does not run them.
      - name: System deps
        run: sudo apt-get update && sudo apt-get install -y libfontconfig1-dev
      # burn-cuda/cudarc are non-optional COMPILE-time deps (see
      # .claude/skills/ddrs-dev/references/build-and-env.md). No GPU is
      # needed — nothing GPU-gated runs in CI.
      - uses: Jimver/cuda-toolkit@v0.2.19
        with:
          cuda: "12.4.1"
          method: network
          sub-packages: '["nvcc", "cudart-dev", "nvrtc-dev"]'
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: ci-debug
      - run: cargo test --features fixtures --no-fail-fast

  # Release routing-acceptance gates: DDR sandbox parity (example exits 1
  # unless ABSOLUTE MATCH) + Juniata end-to-end metric floors.
  acceptance:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: System deps
        run: sudo apt-get update && sudo apt-get install -y libfontconfig1-dev
      - uses: Jimver/cuda-toolkit@v0.2.19
        with:
          cuda: "12.4.1"
          method: network
          sub-packages: '["nvcc", "cudart-dev", "nvrtc-dev"]'
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: ci-release
      - name: DDR sandbox parity (invariant 1)
        run: mkdir -p output && cargo run --release --example compare_ddr_sandbox
      - name: Juniata end-to-end acceptance
        run: cargo test --release --test juniata_acceptance -- --nocapture
```

- [ ] **Step 3: Validate the YAML parses**

Run: `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok` (any Python with PyYAML; `uv run --with pyyaml python3 -c ...` if the system python lacks it).

- [ ] **Step 4: Commit on a branch and open the bring-up PR**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: two-job vetting workflow — debug suite + release routing acceptance

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RHusoRV4Th4tio8bh2oPca"
git push -u origin ci-vetting
gh pr create --title "CI: vet PRs and master pushes with the routing acceptance gates" \
  --body "$(cat <<'EOF'
Implements docs/superpowers/specs/2026-09-03-ci-vetting-design.md: `test`
(debug suite + KAN fixtures) and `acceptance` (release sandbox parity +
Juniata metric floors) on every PR and master push. This PR is the bring-up
vehicle — CUDA-install iteration commits land here until both jobs are green.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01RHusoRV4Th4tio8bh2oPca
EOF
)"
```

(The branch was created in Task 1 Step 0, so the hook commit rides this PR too.)

---

### Task 3: Bring-up — iterate until both jobs are green

**Files:**
- Modify: `.github/workflows/ci.yml` (fixup commits as needed)

**Interfaces:**
- Consumes: the PR from Task 2.
- Produces: a PR whose `test` and `acceptance` checks both pass.

- [ ] **Step 1: Watch the first run**

Run: `gh pr checks --watch` (or `gh run watch <run-id>`)
First runs are uncached: expect 30–60 min per job. Do not kill them for slowness alone.

- [ ] **Step 2: Triage failures against the known-risk table**

| Symptom | Fix |
|---|---|
| `Jimver/cuda-toolkit` fails on sub-package names | Drop `sub-packages` entirely (full toolkit install, slower but reliable) |
| CUDA still not found by `cudarc` (`cuda-version-from-build-system` build error) | Full toolkit install; if that fails, replace the action with `sudo apt-get install -y nvidia-cuda-toolkit` and set no env (apt puts nvcc on PATH) |
| `error: failed to execute process 'cmake'` | Should not happen (`cmake` preinstalled); if it does, add cmake to the System deps apt line |
| plotters/font-kit build error mentioning fontconfig | Already handled by the System deps step; if a different `-dev` lib is named, add it to the same apt line |
| A data-dependent test panics instead of skipping (no `/mnt/ssd1` etc.) | Add the repo-standard guard to that test only: check the path with `Path::exists`, `eprintln!("skipping: <path> not present")`, `return`. Model: `tests/training_verification.rs:79-100` |
| Runner disk exhaustion in `acceptance` | Add as the job's first step: `- name: Free disk space` / `run: sudo rm -rf /usr/local/lib/android /usr/share/dotnet /opt/ghc` |
| Git-dep fetch failure for taddyb/burn, taddyb/cubecl, or rskan | Verify the branch/tag still exists on the fork; this failure is real, not CI-specific |

Each fix is one commit on the branch (`ci: fix <symptom>` + trailers), then re-watch.

- [ ] **Step 3: Confirm green**

Run: `gh pr checks`
Expected: `test` pass, `acceptance` pass. Record both job wall-times in the PR conversation (one comment) for the docs task.

- [ ] **Step 4: Re-run once to confirm the warm-cache path**

Run: `gh workflow run ci --ref ci-vetting`, then `gh run watch` the new run.
Expected: both jobs green again, materially faster (cache hit visible in the `Swatinem/rust-cache` step log). This validates that repeat PR runs are tolerable.

---

### Task 4: Merge and verify the master-push path

**Files:** none (GitHub operations only)

- [ ] **Step 1: Merge the PR**

Run: `gh pr merge --merge` (merge commit, matching the repo's existing PR style — see `git log`'s "Merge pull request" entries).

- [ ] **Step 2: Verify the push-triggered run on master**

Run: `gh run list --branch master --workflow ci --limit 1`, then `gh run watch <id>`.
Expected: both jobs green on the merge commit. This is the "direct pushes get vetted too" path working.

---

### Task 5: Branch protection

**Files:** none (GitHub API only). Requires the checks to exist (Task 4 done).

- [ ] **Step 1: Apply protection**

```bash
gh api -X PUT repos/taddyb/ddrs/branches/master/protection --input - <<'EOF'
{
  "required_status_checks": {
    "strict": false,
    "checks": [{"context": "test"}, {"context": "acceptance"}]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
EOF
```

- [ ] **Step 2: Verify**

Run: `gh api repos/taddyb/ddrs/branches/master/protection --jq '{checks: .required_status_checks.checks, enforce_admins: .enforce_admins.enabled}'`
Expected: both contexts listed, `enforce_admins: false`.

- [ ] **Step 3: Verify direct push still works (the hybrid escape hatch)**

The docs commit in Task 6 is pushed directly to master; its success IS this verification. No separate probe push.

---

### Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md` (Commands section, after the acceptance-test block added 2026-09-02)
- Modify: `.claude/skills/ddrs-dev/SKILL.md` (after the "Commands you actually need" block)
- Modify: `.claude/skills/ddrs-dev/references/testing.md` (new section after "Tier gates")

**Interfaces:**
- Consumes: measured job wall-times from Task 3 Step 3 (replace the `~N min` placeholders below with the real numbers).

- [ ] **Step 1: CLAUDE.md — append to the Commands section**

```markdown
### CI (GitHub Actions)

`.github/workflows/ci.yml` runs on every PR and every master push: job
`test` (debug `cargo test --features fixtures`) and job `acceptance`
(release `compare_ddr_sandbox` + `juniata_acceptance`). Branch protection
requires both for PR merges; direct pushes stay possible and are vetted by
the push-triggered run. CI has no GPU and no real data — green CI does NOT
cover `--features cuda` tests or the data-dependent suite; run those
locally per the tier gates. Local pre-push hook (opt-in):
`git config core.hooksPath .githooks`.
```

- [ ] **Step 2: SKILL.md — add one row to the change→gate table and one command**

Row (after the `config/**/*.yaml` row):

```markdown
| `.github/workflows/ci.yml`, `.githooks/**` | — | push to a branch and let the PR run vet it; there is no local gate for workflow changes |
```

Command (in "Commands you actually need"):

```bash
gh pr checks --watch                                # CI status for the current PR
```

- [ ] **Step 3: testing.md — new section after "Tier gates"**

```markdown
## CI (added 2026-09-03)

`.github/workflows/ci.yml`, on every PR and master push (no path filters —
required checks must never be skipped): job `test` = debug
`cargo test --features fixtures --no-fail-fast` (~N min warm); job
`acceptance` = release `compare_ddr_sandbox` + `juniata_acceptance`
(~N min warm). Branch protection requires both; `enforce_admins` is off so
admin direct pushes remain possible (vetted post-hoc by the push run).

**What green CI does NOT prove:** `--features cuda` tests (compiled out),
data-dependent tests (self-skip — no `/mnt/ssd1`/cluster data on runners),
and anything in the do-not-use list. A data-touching change still needs the
local tier gates.

Both jobs install a CUDA toolkit only because compilation requires it
(build-and-env.md). Local hook: `git config core.hooksPath .githooks`
enables `.githooks/pre-push` (runs `ddr_sandbox_match`; bypass with
`git push --no-verify`).
```

Replace both `~N min` with Task 3's measured warm times.

- [ ] **Step 4: Commit directly to master (this doubles as Task 5 Step 3's verification)**

```bash
git add CLAUDE.md .claude/skills/ddrs-dev/SKILL.md .claude/skills/ddrs-dev/references/testing.md
git commit -m "docs: CI vetting — what the workflow covers, what it does not, hook setup

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RHusoRV4Th4tio8bh2oPca"
git push origin master
```

Expected: push succeeds (admin direct push through protection), and `gh run list --branch master --workflow ci --limit 1` shows a fresh run vetting it.
