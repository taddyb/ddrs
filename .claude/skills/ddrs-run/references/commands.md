# `ddrs` command reference

Every subcommand and flag, captured from `ddrs --help` on master, 2026-09-09.
Global flags `--config` and `--workspace` are accepted by **every** subcommand.

## Global

| Flag | Default | Notes |
|---|---|---|
| `--config <PATH>` | discover `ddrs.yaml` upward from cwd, stopping at the first `.git` ancestor | |
| `--workspace <DIR>` | `.ddrs/` **beside the config** | Pass it explicitly whenever you pass `--config` |

Config discovery walking up to `.git` means a bare `ddrs plan` from a subdirectory
finds the repo-root `ddrs.yaml`. It does **not** search sibling directories.

## `ddrs plan`

Prepare + preview. Idempotent, but **not side-effect free**: on first run it opens
the forcing store to compute the summed-Q' baseline and builds the managed adjacency
caches. Subsequent plans on the same inputs are cache hits and instant.

| Flag | Notes |
|---|---|
| `--workflow <train\|eval\|train-and-test>` | Override the config's `workflow:` for this invocation |
| `--json` | Plan result as JSON |
| `--force` | Re-run the GPU smoke test even if a cached verdict exists |
| `--min-free-gpu-gb <N>` | Warn below this free GPU memory at probe time (default 8) |

What it does, in order: workspace skeleton + GPU probe + cached smoke test →
locate/bootstrap `ddrs.yaml` → load and validate the config → fingerprint data
sources and diff against `.ddrs/sources.lock` → resolve adjacency (explicit paths,
or a managed build from `geospatial_fabric` / `gridded_network`) → summed-Q'
baseline.

Lock semantics: `plan` reports drift, then **refreshes** the lock ("sources as of my
last plan"). Use `run --strict` to abort on drift *before* relocking, preserving the
evidence.

If `ddrs.yaml` is missing, `plan` bootstraps one interactively, offering the last
successful run's `config.yaml` snapshot or the bundled template. **Non-TTY callers
must pass `--config`**, otherwise it errors with
*"no ddrs.yaml found; pass --config or run interactively"*.

## `ddrs run`

Executes a workflow: re-plans internally, then trains and/or evaluates into
`.ddrs/runs/<id>/`.

| Flag | Notes |
|---|---|
| `--workflow <train\|eval\|train-and-test>` | Overrides the config's `workflow:` |
| `--plot` | After success, dump per-COMID KAN parameters to `plot/kan_parameters.nc` |
| `--strict` | Exit 4 if sources drifted since the last plan, instead of warning + relocking |
| `--max-mini-batches <N>` | Stop each epoch after N mini-batches (smoke testing) |
| `--batch-order-from <PATH>` | Replay a captured mini-batch order from JSON; overrides the per-epoch shuffle. Schema: `[{"epoch": int, "mb": int, "staids": [str, …]}]` |
| `--backend <cuda\|cpu>` | **Default `cuda`.** `cpu` is NdArray, deterministic, and forces `sparse_solver` to cpu |
| `--json` | Run result as JSON |

Diagnostics belong on `--backend cpu`: it is deterministic, and the default is
CUDA, so a diagnostic run that omits the flag is not the run you think you made.
Never mix backends within one comparison set.

Run id format: `<UTC timestamp>-[<group>-]<workflow>`, where the optional group
segment is the active data-source group when the config's `data_sources` block
matches a saved `config/sources/<name>.yaml` (compared structurally, so comments
and key order do not break the match).

Consequence of the group lookup path: a run launched with
`--config config/experiments/foo.yaml` **never** gets a group segment, because
`active_group` looks in `config/experiments/config/sources/`, which does not
exist. Launch from the repo root against `./ddrs.yaml` if you want the dataset
named in the run id.

## `ddrs show <RUN_ID>`

Prints a past run's manifest. `--json` for the raw record.

Human output: run id, status, workflow, started/finished, `git <sha> (dirty)`,
drift, and the resolved adjacency store paths + cache key with hit/built.

Manifest keys: `run_id`, `ddrs_version`, `git` {sha, dirty, branch}, `workflow`,
`config_path`, `started_at`, `finished_at`, `status`, `exit_reason`, `system`,
`sources` (per input: path, mtime, size, `fp: blake3:…`), `resolved_adjacency`,
`source_lock` {lockfile, matched, drift}, `outputs` (checkpoint list),
`metrics`, `max_mini_batches`.

## `ddrs status`

Workspace summary: workspace path, lockfile present/absent, last run id, and disk
usage of `.ddrs/runs/` and `.ddrs/adjacency/`. `--json` supported. Use it to list
run ids for `ddrs show`.

## `ddrs gc`

Deletes run directories under `.ddrs/runs/`.

| Flag | Notes |
|---|---|
| `--keep <N>` | Keep the N most recent runs |
| `--keep-successful` | Never delete successful runs |
| `--older-than <DUR>` | Only delete runs older than e.g. `30d`, `12h` |
| `--dry-run` | List what would be deleted, delete nothing |

Prints `note: .ddrs/adjacency/ caches are kept (not pruned by gc in v1)`.

## `ddrs sources`

Named data-source groups stored at `<config dir>/config/sources/<name>.yaml`. For
the standard repo-root `ddrs.yaml` that is `config/sources/`.

| Subcommand | Notes |
|---|---|
| `list` | Lists groups; `*` marks the one matching the current config |
| `use <name>` | Splice the group into the config's `data_sources:` and refresh `sources.lock` |
| `save <name>` | Snapshot the current `data_sources` block as a group (`--force` to overwrite) |

`save`/`use` are textual: comments inside the block travel with the group,
everything outside it is untouched, and `use` validates that the spliced config
parses before committing.

Groups shipped in-repo: `conus`, `conus-gridded`, `conus-hourly`,
`conus-experimental`, `conus-hydrodl2`, `global`, `daily-lstm`, `hourly-lstm`,
`aorc_dhbv_distributed`.

## `ddrs import <STORE>`

Validates a Q' store against the DDR store contract
(`docs/nh-qprime-store-contract.md`) and registers it as a source group.

| Flag | Notes |
|---|---|
| `--name <NAME>` | Group to register under `config/sources/` |
| `--dry-run` | Validate + report coverage only |
| `--force` | Overwrite an existing group of that name |

The reader sniffs daily vs hourly-native resolution from the CF time units. Hourly
stores are sliced natively. An hourly store combined with
`kan_head.disaggregation` is a config error.

## Workspace layout

| Path | Written by | Holds |
|---|---|---|
| `ddrs.yaml` | `plan` (via `$EDITOR`) | Workflow + experiment config (gitignored) |
| `.ddrs/system.json` | `plan` | GPU/driver/smoke-test record |
| `.ddrs/sources.lock` | `plan` | Fingerprints of `data_sources` paths |
| `.ddrs/adjacency/<key>/` | `plan`, first time | Cached adjacency stores, content-addressed |
| `.ddrs/baselines/<key>/` | `plan`, first time | Cached summed-Q' baseline |
| `.ddrs/runs/<id>/manifest.json` | `run` | Per-run audit record |
| `.ddrs/runs/<id>/config.yaml` | `run` | Snapshot of the config that produced the run |
| `.ddrs/runs/<id>/run.log` | `run` | fd-level timestamped tee of stdout+stderr |
| `.ddrs/runs/<id>/Cargo.lock` | `run` | Dependency snapshot |
| `.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/` | `run`, train phase | `head.mpk`, `optim.mpk`, `state.json` |
| `.ddrs/runs/<id>/baseline/` | `run`, train-and-test | `manifest.json` + raw `predictions.f32` / `observations.f32` |
| `.ddrs/runs/<id>/eval/predictions.zarr` | `run`, eval phase | Routed predictions over the eval window |

## Exit codes

`src/cli/types.rs`: 0 Success · 1 Generic · 2 ConfigInvalid · 3 DataSourceMissing ·
4 LockDrift · 5 RuntimeFailure · 6 WorkspaceNotInitialized.

## Legacy binaries (deprecated, removed in 0.4)

`train`, `eval`, `train_and_test` still build and each prints a deprecation warning
pointing at the `ddrs run` equivalent. The legacy `eval` binary remains the only way
to evaluate an existing checkpoint without retraining, since
`ddrs run --workflow eval` is unimplemented:

```bash
cargo build --release --bin eval
target/release/eval --config <cfg> \
  --checkpoint .ddrs/runs/<id>/checkpoints/epoch_E_mb_M \
  --output /tmp/out.zarr
```

`--checkpoint` means different things per binary: `eval` and `probe_zeta_gradient`
take the **directory**; `dump_parameters` takes the **head base**
(`epoch_E_mb_M/head`, no `.mpk`).
