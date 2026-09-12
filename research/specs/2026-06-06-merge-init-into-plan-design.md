# Merge `ddrs init` into `ddrs plan`

**Date:** 2026-06-06
**Status:** Approved (brainstorming session)
**Supersedes:** the `init`/`plan` command split in
`2026-05-30-ddrs-cli-lifecycle-design.md` §Command semantics. Everything else
in that spec (run, show, status, gc, manifest schema, exit codes) stands.

## Motivation

`ddrs init` and `ddrs plan` overlap heavily: both bootstrap `ddrs.yaml` via
`$EDITOR` when missing, both fingerprint every data source, both parse the
config the same way. Worse, they are circularly dependent on first run —
init Phase B (source locking) needs a config that plan bootstraps, while plan
needs the lockfile init writes. The original spec documents the first-time
flow as `init → plan → init → run`. This design collapses the two commands
into a single idempotent `ddrs plan`, making the flow `plan → run`.

## 1. Command surface

```bash
ddrs plan [--workflow <wf>] [--json] [--force] [--min-free-gpu-gb N]
```

- The `init` subcommand is removed from clap. A **hidden stub** remains for
  one release: `ddrs init` prints
  `"ddrs init has been merged into ddrs plan — run \`ddrs plan\`"` and exits
  with code 2, so scripts fail loudly. The stub is deleted in 0.4 alongside
  the legacy binaries.
- `--force`: invalidates the smoke-test cache and re-runs the smoke test.
  **Never touches `.ddrs/runs/`.** (Today's `init --force` removes the whole
  `.ddrs/` workspace; that behavior is dropped. Deleting run history is
  exclusively `ddrs gc`'s job.)
- `--min-free-gpu-gb` migrates from `init` unchanged (warning floor only,
  default 8.0).
- `--workflow` / `--json` keep today's `plan` semantics.

## 2. Merged pipeline

One linear pass, no circular dance:

```
ddrs plan
 ├─ 0. ensure .ddrs/ skeleton + version file          (from init Phase A)
 ├─ 1. GPU probe (in-process cudarc; Ok(None) = CPU)   (from init Phase A)
 ├─ 2. smoke test, cached by driver/cuda/sm/ddrs key   (from init Phase A)
 │     → write .ddrs/system.json
 ├─ 3. bootstrap ddrs.yaml via $EDITOR if missing      (shared, TTY only)
 ├─ 4. parse config, resolve workflow                  (today's plan)
 ├─ 5. fingerprint sources (reuse-if-unchanged)        (shared)
 ├─ 6. diff vs sources.lock → report drift             (today's plan)
 ├─ 7. refresh sources.lock (auto-relock)              (from init Phase B)
 ├─ 8. resolve/validate adjacency, build cache on miss (today's plan)
 ├─ 9. compute/load summed-Q' baseline                 (today's plan)
 └─ 10. print plan summary
```

The GPU probe still runs *before* the interactive `$EDITOR` step, preserving
the original spec's "GPU confidence before editing" rationale.

## 3. Lock semantics

Order matters: **fingerprint → diff → decide → relock**.

- `plan` (CLI): print drift, then relock. The lock means "sources as of my
  last plan."
- `run` (calls `plan()` as a library, as today): the library function carries
  the strict flag `run --strict` already passes. On drift + strict → exit
  `LockDrift` (4) **before relocking**, preserving the evidence. On drift +
  non-strict → warn, relock, proceed.
- First-ever plan (no lock yet): no diff possible; just write the lock.
- `WorkspaceNotInitialized` (6) disappears from `plan`/`run`'s vocabulary —
  plan cannot encounter an uninitialized workspace because it initializes.
  The exit code stays defined for `status`/`show`/`gc`, which still require
  an existing workspace.

Side effect worth naming: `ddrs run` from a fresh clone with a valid
`ddrs.yaml` now just works (its internal `plan()` call initializes
everything). The smoke test adds ~2 s to that very first `run`, cached
thereafter.

## 4. Bootstrap prompt

`pick_source` (`src/cli/plan_bootstrap.rs:58`) currently prefers the last
successful run's `config.yaml` **silently**. It becomes interactive when a
prior successful run exists:

```
No ddrs.yaml found. Start from:
  [1] config of last successful run (<run-id>)
  [2] clean template (config/merit_training.yaml)
```

Then `$EDITOR` opens as today. When `interactive: false` (tests), the
current last-run preference is kept so existing tests don't break. Non-TTY
callers still must pass `--config` — unchanged.

`rm ddrs.yaml && ddrs plan` + picking [2] becomes the documented
"start clean" path. The CLAUDE.md "bootstrap gotcha" section shrinks to a
note about the prompt.

## 5. Help text overhaul

clap derives help from doc comments; today almost none exist
(`src/bin/ddrs.rs` — only `batch_order_from` is documented).

**Top-level `ddrs --help`** gets `long_about` + `after_help`:

```
ddrs — Differentiable Distributed Routing

Lifecycle:
  ddrs plan    Prepare + preview: probes the GPU (first run), bootstraps
               ddrs.yaml if missing, locks data sources, validates config,
               builds adjacency/baseline caches. Idempotent — run anytime.
  ddrs run     Execute a workflow. Re-plans internally, then trains/evals.
  ddrs show    Inspect a past run's manifest.
  ddrs status  Workspace summary + disk usage.
  ddrs gc      Prune old runs from .ddrs/runs/.

Workflows (--workflow or `workflow:` in ddrs.yaml):
  train           Train the KAN head           (requires mode: training)
  eval            Evaluate a checkpoint        (requires mode: testing)
  train-and-test  Train, then eval, then compare vs. summed-Q' baseline

Starting fresh:
  rm ddrs.yaml && ddrs plan — you'll be asked whether to start from your
  last successful run's config or the clean bundled template.
```

**Every subcommand and flag** gets a one-line doc comment (e.g. `--strict` →
"Fail with exit 4 if data sources changed since last plan, instead of
warning"; `--force` → "Re-run the GPU smoke test even if cached"). The
`Workflow` ValueEnum variants get per-variant help including the
`mode:` ↔ workflow agreement rule.

## 6. Template comments rewrite (`config/merit_training.yaml`)

Every field gets a plain-language comment: what it is, units, and what
changing it does. Internal jargon goes:

- `"SP-10: forward CUDA Graph capture+replay (V7a=0.385)"` →
  `"Capture the forward pass as a CUDA graph and replay it each timestep
  (faster; CUDA only)"`.
- `"matches mock_config in tests"` is deleted.

Target register:

```yaml
experiment:
  rho: 90          # Training sequence length in DAYS per mini-batch sample.
  warmup: 5        # Days at the start of each sequence excluded from the loss
                   # (lets routing state spin up from the cold-start estimate).
```

**Values change by zero bytes** — only comments — so DDR config parity
(CLAUDE.md: "mirrors merit_training_config.yaml verbatim") holds for
everything the parser sees. Brief DDR cross-references stay where they
explain *why* a value is fixed (e.g. the KAN `k: 2` override), as a
secondary clause, not the headline.

Note the comments only reach users via the **template** bootstrap path —
which is why §4's prompt matters. Run-dir `config.yaml` snapshots are byte
copies, so comments survive round-trips through the last-run path too.

## 7. Code changes (blast radius)

| File | Change |
|---|---|
| `src/bin/ddrs.rs` | drop `Init` variant; add `--force`, `--min-free-gpu-gb` to `Plan`; hidden `init` stub; doc comments on everything |
| `src/cli/init.rs` | **deleted**; probe/smoke/skeleton logic moves to `system.rs` as `ensure_system_ready(ws, force, min_gb) -> SystemProbe` |
| `src/cli/plan.rs` | prepend pipeline steps 0–2; replace lockfile-read-or-die with diff-then-relock; drop `WorkspaceNotInitialized` path |
| `src/cli/plan_bootstrap.rs` | interactive source prompt (§4) |
| `src/cli/run.rs` | strict-abort-before-relock ordering (mostly already present via `PlanResult.drift`) |
| `config/merit_training.yaml` | comments only (§6) |
| `src/cli/lockfile.rs`, `fingerprint.rs` | unchanged |

No touch to `src/routing/`, `src/sparse.rs`, `src/geometry.rs` → no parity
re-run strictly required, though `compare_ddr_sandbox` is cheap insurance
per invariant 1.

## 8. Error handling

Exit codes unchanged in meaning:

| Code | Meaning | Emitted by |
|---|---|---|
| 0 | success | all |
| 2 | `ConfigInvalid` (YAML errors, non-TTY without config, init stub) | plan, run |
| 3 | `DataSourceMissing` | plan, run |
| 4 | `LockDrift` | run `--strict` only |
| 5 | `Runtime` (incl. smoke-test failure, same fatality as init today) | plan, run |
| 6 | `WorkspaceNotInitialized` | status, show, gc only |

## 9. Testing

1. **Fresh-dir plan**: valid `--config`, empty dir → creates
   `.ddrs/{version,system.json,sources.lock}`, exits 0.
2. **Idempotency**: second `plan` reuses the smoke verdict (`system.json`
   `passed_at` unchanged) and fingerprints (no blake3 re-hash).
3. **Drift + relock**: mutate a source, plan → drift reported, lock
   refreshed; third plan → no drift.
4. **`run --strict` drift**: lock byte-identical after the aborted run
   (evidence preserved).
5. **`--force`**: smoke re-runs; `.ddrs/runs/` contents untouched.
6. **Stub**: `ddrs init` exits 2 with the redirect message.
7. **Bootstrap prompt**: template choice honored when input simulated;
   `interactive: false` keeps last-run preference.

## 10. Docs

- README "Getting started": `init → plan → run` becomes `plan → run`.
- CLAUDE.md CLI section: update flow, shrink the bootstrap-gotcha note.
- This spec supersedes the init/plan split in the 2026-05-30 lifecycle spec.

## Considered and rejected: `ddrs plot`

A `ddrs plot` subcommand (template + execute the three standard plot
families — hydrograph, parameter map, metrics — from the
`ddrs-eval-plots` skill against the latest run) was designed and then
dropped: baking the recipes into the binary loses the flexibility the
skill provides (custom gauges, regions, one-off plot requests), and would
have created a second canonical copy of the notebook templates. Plotting
stays with the `.claude/skills/ddrs-eval-plots` skill.

## Concerns

- **Auto-relock erases drift evidence at plan time.** If you plan, miss the
  drift warning, the lock now blesses the changed data and `run --strict`
  won't catch it. Why acceptable: drift between *plan and run* is what
  strict mode is for, and that still works; long-lived drift tracking was
  never a real workflow here. If it becomes one, `--no-relock` is a trivial
  later addition.
- **`run` gains first-run side effects** (workspace creation, smoke test) in
  non-interactive contexts. Why acceptable: cached after the first run, and
  `run` already does GPU preflight anyway.
- **`init --force`'s workspace-nuke silently changes meaning.** Muscle-memory
  `ddrs init --force` now gets a stub error rather than a wiped workspace —
  strictly safer, but worth knowing.
- **Help text / template comments can drift from reality** if future flags
  skip doc comments. Cheap mitigation: missing help is visually obvious in
  `--help` output.
- **`plan` is no longer "cheap and side-effect-free"** as the 2026-05-30 spec
  promised. That promise was already broken (adjacency builds, baseline
  computation, ~370 MB icechunk reads); this design makes the docs honest
  rather than making plan dirtier.

## Assumptions

- **ddrs has no external users** (pre-0.4, legacy binaries mid-deprecation),
  so a breaking CLI change with a one-release stub is fine. If wrong, a real
  alias would be needed instead.
- **The smoke cache key** (`driver;cuda;ddrs;sm;backend`) is trusted to make
  repeated plans cheap — the merged plan's cost profile rests on it.
- **Benefit justifying the change**: kills the documented
  `init → plan → init → run` circular flow, removes the duplicated
  bootstrap/fingerprint orchestration, makes the docs honest about plan's
  side effects, and gives users a self-explanatory `--help` and config
  template.
