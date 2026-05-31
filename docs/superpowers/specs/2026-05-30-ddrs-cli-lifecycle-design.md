# ddrs CLI Lifecycle (Spec A)

**Date:** 2026-05-30
**Status:** Draft — pending review
**Author:** Tadd Bindas (with Claude)
**Companion spec:** B — Native parameter-map rendering (TBD, separate brainstorm)

## Summary

Replace the three single-purpose binaries (`train`, `eval`, `train_and_test`)
with one `ddrs` CLI organized around a terraform-inspired lifecycle:
`init → plan → run`. The CLI's primary job is **validation** (catch bad
configs and missing data before launching a long job) and **reproducibility**
(every run writes a self-contained manifest binding config, code, data, and
outputs together). Data-source pinning (the terraform-lockfile analog) is a
byproduct of `init`, not its own command.

## Goals

1. One entrypoint binary (`ddrs`) replacing three. Existing binaries get a
   one-release-cycle deprecation shim.
2. `ddrs plan` is a cheap, deterministic dry-run that fails fast on missing
   data sources, unreadable zarrs, out-of-range date windows, or schema errors
   in the YAML — before any GPU work. When no `ddrs.yaml` exists, `plan`
   bootstraps one interactively from the latest successful run (or the
   bundled template) via `$EDITOR`. Agent / non-TTY callers bypass this by
   passing `--config`.
3. Every `ddrs run` produces a self-contained directory under
   `.ddrs/runs/<timestamp>-<workflow>/` containing: a snapshot of the config,
   data-source fingerprints captured at run start, the git SHA, captured
   stdout/stderr, output checkpoints, and a `manifest.json`.
4. `ddrs init` writes `.ddrs/sources.lock` capturing data-source fingerprints
   as part of its setup. Subsequent `plan` / `run` invocations compare the
   live sources against the lock and **warn** on drift; `--strict` upgrades
   to a hard failure. Re-pinning is "re-run `ddrs init`" — no separate lock
   command.
5. All commands emit a `--json` mode suitable for agent consumption.

## Non-goals

- **Native Rust map rendering** (Spec B). v1 ships `--plot` as a thin wrapper
  around `dump_parameters` writing only `kan_parameters.csv` plus a hint
  pointing at DDR's `plot_parameter_map.ipynb`.
- **Multi-config workspaces** (one `ddrs.yaml` per project directory).
- **Resource graph / dependency tracking.** Terraform's planning DAG has no
  analog for routing — a routing run is a single computation, not a graph of
  resources.
- **A `destroy` analog.** No infrastructure to tear down.
- **Distributed-run orchestration.** Single-machine CLI.
- **Scheduled / cron-driven runs.** Out of v1; the run-manifest design
  already supports multiple runs from the same lock, so this is a pure
  addition later.
- **`ddrs taint <run-id>`** (mark a run as bad for downstream tools).
- **Remote backends for manifests** (S3, etc.).

## Concerns

- **`ddrs init` runs in two phases by config availability.** Compile (source
  only), GPU probe, GPU memory check, workspace creation, and smoke test are
  config-independent and always run. Data-source stat + `sources.lock`
  writing require `ddrs.yaml` to exist; they are skipped with a clear
  message ("no `ddrs.yaml` found — run `ddrs plan` to bootstrap one, then
  re-run `ddrs init` to lock data sources") when it doesn't. First-time
  flow is therefore `init → plan → init → run`.
- **CWD walk-up has surprise factor.** A `ddrs.yaml` in `~/` could be picked
  up from anywhere. Mitigation: walk-up stops at the first `.git/` ancestor
  (matches cargo's `Cargo.toml` lookup).
- **Tee'ing CUDA stderr is finicky.** CUDA flushes diagnostics on its own
  schedule; naive buffered tee loses output on crash. Mitigation: spawn the
  workflow as a subprocess with `os_pipe` and write each pipe chunk to the
  log file with `O_APPEND` synchronously while forwarding to the inherited
  fd. Documented choice, not "we'll figure it out."
- **`Cargo.lock` snapshot is best-effort.** Manifest always records
  `ddrs --version`; `Cargo.lock` is captured only when reachable.
- **Scope creep into routing internals.** This spec touches **only** new code
  under `src/cli/` and `src/bin/ddrs.rs`, plus one focused extraction in
  `src/training/` (a `bootstrap_head_and_state` helper that the new CLI and
  the deprecation shims both call to avoid quadruple-implementing backend
  setup). It does NOT modify `src/routing/`, `src/sparse.rs`,
  `src/geometry.rs`, or `src/nn/`.
- **Fixture embedding for installed binaries.** Step 6 of `init` needs the
  sandbox fixture at runtime. `include_bytes!` of the existing
  `fixtures/sandbox/*.csv` adds ~hundreds of KB to the installed binary.
  Acceptable for now; revisit if binary size becomes a problem.
- **Unbounded `.ddrs/runs/` growth.** Each run can persist hundreds of MB
  (checkpoints + log files). A multi-week sweep with hourly runs trivially
  reaches tens of GB. Mitigation: ship `ddrs gc` (see Commands) and have
  `ddrs status` print total `runs/` size so growth is visible.

## Assumptions

- Existing library entrypoints (`training::driver::train`, `training::eval`)
  are stable enough to be called directly. Justification: they were just
  stabilized for the three current binaries and have integration tests.
- One ddrs project = one `ddrs.yaml`. Justification: every existing use case
  is single-config.
- `.ddrs/` is gitignored by default. `.ddrs/sources.lock` is the only
  candidate for committing; project owners decide that per-repo.
- `clap` is the CLI framework. Justification: zero new deps, matches all
  existing binaries.
- Daily-driver GPU is single CUDA device. No multi-GPU dispatch logic in v1.

## Reused infrastructure & new dependencies

### Already in the repo (reuse, do not re-implement)

- `Config::from_yaml_file_with_mode` (`src/config.rs:315`) — every command
  that needs YAML uses this. The CLI maps `--workflow {train,eval,...}` to
  `ConfigMode::{Training, Testing}` (`src/config.rs:20`) and passes it in.
- `training::driver::train`, `training::eval::eval`, `training::load_kan_head`
  — the workflow dispatch in `ddrs run` is a thin shim over these.
- `dump_parameters` body (`src/bin/dump_parameters.rs`) — `ddrs run --plot`
  invokes the same code path; do not duplicate the attribute-tensor build or
  the batched forward.
- `examples/compare_ddr_sandbox` fixture (`fixtures/sandbox/*.csv`) and its
  `read_int_csv` / `read_matrix_csv` / `setup_inputs` boilerplate
  (`examples/compare_ddr_sandbox.rs:74-151`) — the smoke test uses the
  **same** fixture files. As part of this spec, those loader helpers move
  to `src/sandbox.rs` (or `tests/common.rs`) so the smoke test and the
  existing example both call them; no second fixture copy.
- `cudarc` (already a dep — `Cargo.toml`) — used in-process for GPU model,
  compute capability, total/free memory. **Do not shell out to `nvidia-smi`**.
- `clap` derive — CLI parsing pattern from existing binaries.

### New dependencies to add

- `blake3` — source fingerprinting.
- `os_pipe` — subprocess stdout/stderr capture for `ddrs run`.
- `serde_json` — manifest + lockfile + `--json` output. (Add if not present.)
- `humantime` — `ddrs gc --older-than 30d` duration parsing.

### Standard library / no crate needed

- `std::io::IsTerminal` (1.70+) — TTY detection for `plan` bootstrap.
- `std::process::Command` — shelling out to `git rev-parse HEAD` /
  `--abbrev-ref HEAD` / `--porcelain` for the manifest's `git` block. No
  `git2` / `gix` dep; we need three short outputs and `git` is already a
  hard build-time requirement.
- `$EDITOR` invocation — `std::process::Command` with fallback to `vi`. No
  `edit` crate (its tmpfile flow is the opposite of what we want — we're
  editing a real file in CWD, not a scratch buffer).

## CLI surface

```
ddrs init   [--force] [--min-free-gpu-gb N]        # setup + lock data sources
ddrs plan   [--json] [--workflow W]                # dry-run validation
ddrs run    [--workflow W] [--plot] [--strict]     # execute
            [--max-mini-batches N] [--json]
ddrs show   <run-id> [--json]                      # pretty-print past manifest
ddrs status [--json]                               # summarize .ddrs/ state
ddrs gc     [--keep N] [--keep-successful]         # prune .ddrs/runs/
            [--older-than DUR] [--dry-run]
```

Global flags accepted by every subcommand:

- `--config <path>` — overrides CWD walk-up.
- `--workspace <dir>` — overrides the default sibling `.ddrs/`.
- `-q | --quiet`, `-v | --verbose`.

### Types (Rust enums, not bare strings)

```rust
// crate-public, with #[derive(clap::ValueEnum, serde::Serialize, serde::Deserialize)]
enum Workflow { Train, Eval, TrainAndTest }

// in manifest.json
enum RunStatus { Ok, Failed, Interrupted }

// returned from every subcommand
#[repr(i32)]
enum ExitCode {
    Success = 0,
    Generic = 1,
    ConfigInvalid = 2,
    DataSourceMissing = 3,
    LockDrift = 4,
    RuntimeFailure = 5,
    WorkspaceNotInitialized = 6,
}
```

`Workflow` default is sourced from a top-level `workflow:` key in
`ddrs.yaml`; the CLI flag overrides. If neither is set, the command fails
with `ExitCode::ConfigInvalid` and a clear message.

## Workspace layout

```
your-project/
├── ddrs.yaml                                # the config (renamed from merit_training.yaml)
└── .ddrs/                                   # workspace (gitignored)
    ├── version                              # ddrs binary version that ran init
    ├── system.json                          # probe output + cached smoke-test verdict
    ├── sources.lock                         # data-source fingerprints, written by `ddrs init`
    └── runs/
        └── 2026-05-30T14-22-07-train/       # one dir per `ddrs run`
            ├── manifest.json                # the reproducibility record
            ├── config.yaml                  # copy of ddrs.yaml at run start
            ├── Cargo.lock                   # copy if reachable from binary
            ├── stdout.log
            ├── stderr.log
            ├── checkpoints/                 # .mpk files
            │   └── epoch_5_mb_0.mpk
            └── plot/                        # only if --plot was passed
                └── kan_parameters.csv       # v1: CSV only (Spec B adds .png)
```

## Schemas

### `sources.lock` (project-level, written by `ddrs init`)

```jsonc
{
  "ddrs_version": "0.3.0",
  "created_at": "2026-05-30T14:22:07Z",
  "sources": {
    "attributes":      { "path": "...", "mtime": "...", "size": 12345678, "fp": "blake3:..." },
    "conus_adjacency": { "path": "...", "mtime": "...", "size": ..., "fp": "..." },
    "gages_adjacency": { "path": "...", "mtime": "...", "size": ..., "fp": "..." },
    "streamflow":      { "path": "...", "mtime": "...", "size": ..., "fp": "..." },
    "observations":    { "path": "...", "mtime": "...", "size": ..., "fp": "..." },
    "gages":           { "path": "...", "mtime": "...", "size": ..., "fp": "..." }
  }
}
```

`fp` is an **opaque comparison token**; consumers must treat it as such and
only test for equality. The current format is `"blake3:<hex>"`; the rule for
how it is computed may evolve between ddrs versions:

- **Zarr / icechunk paths:** blake3 of the root metadata file (e.g.
  `zarr.json`). Full byte content is multi-GB and too slow for an
  interactive command.
- **CSV files:** full-content blake3. Acceptable while the gauges CSV
  (currently `gages_3000.csv`, ~495 KB) stays under ~10 MB; above that,
  fall back to a stat-only fingerprint.

`mtime` and `size` are recorded alongside `fp` so `plan`/`run` can skip
re-hashing when `(path, mtime, size)` is unchanged — see "Fingerprint reuse"
under `ddrs run`.

### `manifest.json` (per-run)

```jsonc
{
  "run_id": "2026-05-30T14-22-07-train",
  "ddrs_version": "0.3.0",
  "git": { "sha": "ab1d7f4...", "dirty": false, "branch": "rskan-head-swap" },
  "workflow": "train",                  // Workflow enum, serialized kebab-case
  "config_path": ".ddrs/runs/2026-05-30T14-22-07-train/config.yaml",
  "started_at":  "2026-05-30T14:22:07Z",
  "finished_at": "2026-05-30T18:41:33Z",
  "status": "ok",                       // RunStatus: "ok" | "failed" | "interrupted"
  "exit_reason": null,                  // string when status != "ok"
  "system": { /* same shape as .ddrs/system.json, snapshotted at run start */ },
  "sources": { /* same shape as sources.lock.sources, captured at run start */ },
  "source_lock": {
    "lockfile": ".ddrs/sources.lock",
    "matched": true,
    "drift": []                         // field names whose fp differed
  },
  "outputs": {
    "checkpoints": ["checkpoints/epoch_5_mb_0.mpk"],
    "plot": null                        // populated to "plot/kan_parameters.csv" if --plot
  },
  "metrics": { "final_loss": 0.385, "epochs_completed": 5 }
}
```

## Command semantics

### `ddrs init`

`init` is the install/setup command. It runs in two **phases**:

- **Phase A (always runs)**: install-level checks that don't depend on a
  config — compile, GPU probe, GPU memory, workspace creation, smoke test.
- **Phase B (only when `ddrs.yaml` exists)**: data-source reachability and
  `sources.lock` write. Skipped with a clear message when no config is
  found: `"no ddrs.yaml found — run 'ddrs plan' to bootstrap one, then re-run
  'ddrs init' to lock data sources."`

This is the documented first-run flow: `init → plan → init → run`. Phase A
gives the user immediate confidence the install works; phase B happens once
they have a real config.

#### Phase A — install setup

1. **Compile** (source-checkout only). Detect mode by walking up from the
   binary's location for a `Cargo.toml` whose `[package].name = "ddrs"`. If
   found, run `cargo build --release` with CUDA features auto-detected via
   `cudarc::driver::result::init` (CPU-only otherwise). Skip in
   installed-binary mode.
2. **GPU readiness check.** Use `cudarc` in-process: driver version, CUDA
   runtime, device name, compute capability (`sm_*`). No shelling out to
   `nvidia-smi`. Fail loudly with a remediation hint if absent ("no CUDA
   device found; install nvidia driver ≥ 530 or build with `--features cpu`").
3. **GPU memory check.** Query free GPU memory via `cudarc`. Compare against
   `--min-free-gpu-gb` (default 8). Print a warning if below the floor; do
   not fail.
4. **Workspace creation.** Make `.ddrs/runs/`; write `version` (the ddrs
   binary version that ran init) and `system.json`. `system.json` schema:

   ```jsonc
   {
     "ddrs_version": "0.3.0",
     "probed_at": "2026-05-30T14:22:07Z",
     "gpu": "RTX 4090", "cuda_runtime": "12.4", "driver": "550.78",
     "sm": "8.9", "free_gpu_gb_at_probe": 22.1,
     "smoke_test": {
       "key": "driver=550.78;cuda=12.4;ddrs=0.3.0;sm=8.9",
       "passed_at": "2026-05-30T14:22:09Z"
     }
   }
   ```

   `smoke_test.key` is checked on re-runs to skip step 5 when nothing
   changed.
5. **Functional smoke test.** Run a small routing problem on the bundled
   5-reach RAPID sandbox fixture (reusing `examples/compare_ddr_sandbox`'s
   `fixtures/sandbox/*.csv` and the loader helpers lifted to `src/sandbox.rs`
   — no second fixture copy). Verify the output discharge tensor:

   - all values finite (no NaN, no Inf),
   - all values ≥ 0,
   - at least one value > 0 (the solver actually did something).

   No DDR reference comparison. ~2 seconds on a CUDA GPU.

   **Caching.** The smoke test is re-run only when
   `smoke_test.key` in `system.json` differs from the current
   `(driver, cuda_runtime, ddrs_version, sm)` tuple, or when `--force` is
   passed. On idempotent re-init, the cached verdict is reused and the GPU
   smoke is skipped. Steps 1–3 always run (they're cheap).

   **Fixture distribution.** From source the files are read from the repo;
   from an installed binary the same files are embedded via `include_bytes!`.

#### Phase B — data-source lock (requires `ddrs.yaml`)

6. **Data-source reachability.** Stat every path under `data_sources:` in
   parallel (`std::thread::scope` or `rayon::join`); report any missing or
   unreadable path with the actual filesystem path it tried. Exits
   `DataSourceMissing` (3) on failure.
7. **Lockfile write.** Compute fingerprints for all sources (parallel; see
   schema). Write `.ddrs/sources.lock` **atomically**: write to
   `.ddrs/sources.lock.tmp` in the same directory, `fsync` it, then
   `rename(2)` over `.ddrs/sources.lock`. Idempotent re-runs always refresh
   the lock (cheap and correct — captures upstream data changes).

#### `--force`

`--force` removes `.ddrs/` and rebuilds. Otherwise re-runs are idempotent:
phase A always re-runs cheap steps (1–3), reuses the cached smoke verdict
when possible (5), and never re-creates the workspace skeleton (4). Phase B
always refreshes the lock when `ddrs.yaml` is present.

Bootstrapping a `ddrs.yaml` is **not** done by `init` — see `ddrs plan`.

Exits `Success` (0) if all required steps pass; non-zero with a clear
single-line message and the matching `ExitCode` otherwise.

### `ddrs plan`

0. **Bootstrap if no config exists.** If neither `--config` was passed nor a
   `ddrs.yaml` is found by CWD walk-up, enter the bootstrap flow:

   - Scan `.ddrs/runs/` for the latest run whose `manifest.json` has
     `status == "ok"`. If one exists, prompt:

     ```
     no ddrs.yaml found.
     start from the latest successful run? [Y/n/template]
       → 2026-05-28T09-14-01-train (final_loss=0.391)
     ```

     `Y` (default) copies that run's `config.yaml` to `./ddrs.yaml`.
     `template` copies the bundled `config/merit_training.yaml` instead.
     `n` aborts with `ConfigInvalid` (2).
   - If no successful past runs exist, skip the prompt and copy the
     bundled template directly.
   - Open `$EDITOR` (fallback `vi`) on `./ddrs.yaml`. Wait for editor to
     exit.
   - On editor exit, validate (steps 2–6 below). If validation fails,
     print the error and prompt `[r]e-edit / [q]uit`. Loop until valid or
     user quits.

   The bootstrap is **strictly interactive** and skipped entirely when
   `--config` is passed, keeping agent runs non-interactive. Detection uses
   `std::io::IsTerminal` on stdin. If stdin is not a TTY and no config is
   found, `plan` exits `ConfigInvalid` (2) with
   `"no ddrs.yaml found; pass --config or run interactively"`.

1. **Resolve config** (`--config` or CWD walk-up to first `.git/`).
2. **Parse YAML** via `Config::from_yaml_file_with_mode(path, mode)` where
   `mode` derives from the resolved `Workflow`. Schema errors propagated
   verbatim (the loader already reports line/column).
3. **Stat data sources in parallel**; report missing/unreadable.
4. **Read `.ddrs/sources.lock`** (which `init` wrote). For each entry,
   compare current `(path, mtime, size)` against the lock; only re-blake3
   when one of those changed. Report drift. If the lock is missing, exit
   `WorkspaceNotInitialized` (6) with `"workspace not initialized; run 'ddrs init'"`.
5. **Open zarr/icechunk metadata-only:**
   - Confirm `start_time..end_time` is within the streamflow time axis.
   - Confirm gauges in the CSV exist in the gages-adjacency store.
6. **Compute and print** the plan summary:
   - **Workflow** that will run.
   - **Train summary:** `n_gauges`, `batches_per_epoch = ceil(n_gauges / experiment.batch_size)`,
     `epochs`, `est_timesteps = rho * batches_per_epoch * epochs`.
   - **Test summary:** `n_days = (end_time - start_time)`,
     `chunks = ceil(n_days / testing.batch_size)`.
   - **GPU memory upper bound (rough):**
     `mem_gb ≈ (rho * max_subgraph_size * 4 * SAFETY_FACTOR) / 1e9`, where
     `4` is `bytes_per_f32` and `SAFETY_FACTOR = 8` accounts for activations
     + autograd tape + Adam state (m, v) + grads + a margin for the sparse
     workspace. Print as a **rough upper bound**; it intentionally
     overestimates. If the computed value exceeds free GPU memory captured
     in `system.json`, print a warning.
7. **Return a `PlanResult`** (see below) for in-process consumers. Exit
   `Success` (0) if all checks pass, non-zero with the matching `ExitCode`
   otherwise.
8. `--json` emits the entire `PlanResult` as JSON for agent consumption.

#### `PlanResult` (in-process struct, also the `--json` shape)

```rust
pub struct PlanResult {
    pub config: Config,              // parsed once; reused by run
    pub config_path: PathBuf,
    pub workflow: Workflow,
    pub sources: SourceFingerprints, // current fingerprints, ready for manifest
    pub drift: Vec<String>,          // data_sources field names that drifted
    pub summary: PlanSummary,        // counts + memory estimate above
}
```

`ddrs run` invokes plan as a library call and consumes this struct directly
— no re-parsing, no re-stat'ing, no re-fingerprinting on the cold path.

### `ddrs run`

1. **Plan as a library call.** Invoke the `plan` function in-process; abort
   with the propagated `ExitCode` if it fails. Receive a `PlanResult` with
   parsed `Config`, current source fingerprints, drift list, and summary —
   no second pass over YAML or filesystem on the success path.
2. **Drift policy.** Read `PlanResult.drift`; if non-empty, warn by default
   or exit `LockDrift` (4) under `--strict`. (Phase B of `init` writes the
   lock; missing lock already produced `WorkspaceNotInitialized` (6) in
   step 1.)
3. **Create run directory.** `.ddrs/runs/<timestamp>-<workflow>/`; copy
   `ddrs.yaml` (already in memory; just serialize) and `Cargo.lock` if
   reachable from the binary location.
4. **Capture stdout/stderr.** Spawn the workflow body in a subprocess (or
   in-thread with `os_pipe`-redirected `stdout`/`stderr` fds). Each captured
   chunk is written synchronously to `stdout.log` / `stderr.log` opened
   `O_APPEND` and also forwarded to the inherited fds. Documented choice;
   see Concerns.
5. **Dispatch to the workflow** via `Workflow` enum match:
   - `Train` → `training::driver::train(&config, &dataset, &mut state, &mut optimizer, &device, &run_dir.join("checkpoints"), max_mini_batches)`
   - `Eval` → `training::eval::eval(&config, &dataset, &checkpoint_dir, &device)`
   - `TrainAndTest` → train then eval, both targeting paths inside the run dir.

   Backend + head setup (~30 lines of `KanHeadConfig::new(...).with_*` +
   `<I as Backend>::seed` + `build_adam` currently duplicated across the
   four binaries) is extracted to
   `training::bootstrap::bootstrap_head_and_state(&config, &device) -> (KanHead<AB>, TrainState, Optimizer)`.
   Both the new CLI and the deprecated binaries call this; no
   re-implementation.
6. **`--plot` post-step.** If set and the workflow produced a final
   checkpoint, call `dump_parameters::dump(&config, &final_checkpoint, &run_dir.join("plot/kan_parameters.csv"))`
   — same `dump_parameters` body extracted to a library function for the
   spec's purposes. Print the notebook hint. Spec B replaces the print with
   native rendering behind the same flag.
7. **Finalize `manifest.json`** with `status`, `finished_at`, `outputs`,
   `metrics`, and `source_lock.{matched,drift}` populated from
   `PlanResult.drift`. Atomic write: `manifest.json.tmp` in the same
   directory, `fsync`, `rename(2)`.
8. **`--max-mini-batches N`** is passed through and recorded in
   `manifest.json` for clarity.

#### Fingerprint reuse

`run` never re-blake3s a source file whose `(path, mtime, size)` matches
the lockfile. In the common case where nothing has moved on disk, a full
`ddrs run` does **zero** blake3 work beyond what `plan` already did.

### `ddrs show <run-id>`

Pretty-print `.ddrs/runs/<run-id>/manifest.json`. `--json` emits raw.

### `ddrs status`

Summarize: workspace path, init time, last run, lock state (present? when?),
**total `.ddrs/runs/` size on disk** (so growth is visible without `du`),
and a hint to run `ddrs gc` if the total exceeds a configurable threshold
(default 10 GB).

### `ddrs gc`

Prune `.ddrs/runs/`. Filters compose with AND:

- `--keep N` — keep the N most-recent runs regardless of other filters.
- `--keep-successful` — never delete a run whose `manifest.json` has
  `status == "ok"`.
- `--older-than DUR` — only consider runs older than DUR (e.g., `30d`,
  `12h`); `DUR` parsed by `humantime`.
- `--dry-run` — print what would be deleted; do nothing.

Default with no flags: print a summary and exit `Success` without deleting
anything. Deletion is opt-in via at least one of the filter flags.

## Migration plan (deprecating old binaries)

- **One release cycle of overlap.**
- **Shared bootstrap helper.** As part of this spec, the ~30 lines of
  backend + head + optimizer setup currently duplicated across
  `src/bin/{train,eval,train_and_test,dump_parameters}.rs` move to
  `training::bootstrap::bootstrap_head_and_state`. The new `ddrs` binary
  and the deprecation shims both call it. Without this, the new CLI would
  be the fourth copy.
- `src/bin/{train,eval,train_and_test}.rs` keep their existing arg parsers
  but, on entry, print to stderr:

  ```
  warning: `train` is deprecated and will be removed in 0.4.
  use `ddrs run --workflow train` instead.
  ```

  …and then dispatch to the same library entrypoints via the shared
  bootstrap helper. **Behavior unchanged for one release.**
- `src/bin/dump_parameters.rs` stays as-is; its logic also moves into a
  library function (`dump_parameters::dump`) so `ddrs run --plot` and the
  standalone binary share one path. Deprecation deferred until Spec B
  ships.
- After one release: delete the three deprecated binaries.

### Documentation deliverables

- `CLAUDE.md` Commands section updated to lead with `ddrs run --workflow ...`
  examples; old binary commands moved to a "Legacy" subsection that is
  removed when the binaries are deleted.
- CLAUDE.md "When in doubt" section currently points at `.claude/`. This
  spec lives at `docs/superpowers/specs/` (the brainstorming-skill default).
  Add a line: "`docs/superpowers/specs/` holds design docs from `/superpowers`
  brainstorms" so the doc-root split is documented.
- `.claude/ARCHITECTURE.md` gains a short "CLI lifecycle" section linking
  to this spec.

## Error handling & exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Generic unexpected error |
| 2 | Config invalid (YAML parse, schema, missing required keys) |
| 3 | Data source missing or unreadable |
| 4 | Lock drift detected with `--strict` |
| 5 | Runtime failure during workflow execution |
| 6 | Workspace not initialized (no `.ddrs/` and command requires it) |

All non-zero exits print a single-line human-readable error and, in
`--json` mode, a `{ "ok": false, "code": N, "error": "..." }` object.

## Testing

New integration tests under `tests/cli_*.rs` (each its own crate per the
existing convention):

- `tests/cli_init.rs` — `init` on a tmp dir from a source-checkout
  Cargo.toml produces a populated `.ddrs/` with `version`, `system.json`,
  and `sources.lock` (one entry per `data_sources` field), and the
  functional smoke test passes (output discharge is finite, non-negative,
  and not all zeros). Test variants for: GPU-absent host (skip GPU steps,
  print remediation), preexisting `.ddrs/` (idempotent re-check, no
  workspace re-creation but lock refreshed), and `--force` (rebuilds).
- `tests/cli_plan.rs` — `plan` against `config/merit_training.yaml`
  succeeds; `plan` against a mutated YAML with a bogus path exits 3.
- `tests/cli_plan_bootstrap.rs` — in a tmp dir with no `ddrs.yaml` and a
  TTY-emulating stdin, `plan` enters the bootstrap flow: prompts for
  starting point, opens a stub `$EDITOR` (env override pointing at a test
  script that writes a valid YAML), validates, and exits 0 with
  `ddrs.yaml` present. Negative test: non-TTY stdin and no config → exit
  2 with the expected message.
- `tests/cli_run_drift.rs` — `init` then `run --max-mini-batches 1`
  produces a valid `manifest.json` whose `source_lock.matched` is `true`.
  Mutating the CSV gauges file between `init` and `run` produces
  `source_lock.matched == false` and a warning by default; the same
  scenario with `--strict` exits 4.
- `tests/cli_plot.rs` — `run --plot --max-mini-batches 1` produces a non-empty
  `plot/kan_parameters.csv` with the expected COMID-keyed columns.
- `tests/cli_show.rs` — `show <run-id>` round-trips a manifest written by
  `run`.
- `tests/cli_status_gc.rs` — `status` reports `runs/` size and last-run
  timestamp; `gc --keep 1 --older-than 0s` deletes all but the most recent
  run; `gc --keep-successful` preserves an OK run targeted by other
  filters; `gc --dry-run` deletes nothing.
- `tests/cli_workspace_uninit.rs` — `plan` (with `--config`) and `run` in a
  directory with no `.ddrs/` exit `WorkspaceNotInitialized` (6) with the
  expected message.
- `tests/cli_runtime_failure.rs` — a workflow body that panics produces a
  `manifest.json` with `status == "failed"`, `exit_reason` populated, and
  the CLI exits `RuntimeFailure` (5).
- `tests/cli_json_contract.rs` — `--json` output from `plan`, `show`, and
  `status` parses as JSON and contains the documented top-level keys.
- `tests/cli_deprecation_shim.rs` — `train --config ...` and
  `train_and_test --config ...` print `"deprecated"` to stderr and still
  succeed (behavior-equivalent to today).

The existing regression test (`examples/compare_ddr_sandbox`) is **not**
ported to use the CLI. It must keep calling the library API directly so a
CLI bug never masks a routing regression.

