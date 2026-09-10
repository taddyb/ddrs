---
name: ddrs-run
description: Use when launching, watching, resuming, or auditing a ddrs training or evaluation job: starting a train / train-and-test run, switching datasets between MERIT and gridded DDM30, finding a past run's metrics or git SHA, resuming after a crash, pruning old runs, or when a run used the wrong code or the wrong data. Trigger on "train a model", "run ddrs", "which run was that", "resume from epoch N", "did it beat the baseline", "ddrs status/show/gc/sources", "my config change did nothing".
---

# Running and tracking ddrs models

The `ddrs` binary is a terraform-style lifecycle: **`plan`** prepares and validates,
**`run`** executes a workflow into `.ddrs/runs/<id>/`, **`show`/`status`** audit what
happened, **`gc`** prunes. Every run writes a self-describing manifest (config
snapshot, input fingerprints, git SHA), so a finished run can be reproduced without
guessing what produced it.

This skill is the runbook. For building/testing ddrs itself use `ddrs-dev`; for
plotting a finished run use `ddrs-eval-plots`.

## Preflight: do these three before EVERY run

Skipping any one of them produces a run that looks fine and is wrong.

**1. Refresh the binary if `src/` changed since the last install.**

```bash
cargo install --path .          # canonical; updates ~/.cargo/bin/ddrs
```

`cargo build` does **not** update the `ddrs` on your PATH. A stale binary does not
say it is stale. It fails as if your *config* were wrong. Reproduced 2026-09-09:
a pre-gridded `~/.cargo/bin/ddrs` given a valid `gridded_network:` config answered

```
error: config invalid: data_sources: adjacency sources are missing —
either set both `conus_adjacency` and `gages_adjacency`, or set `geospatial_fabric`
```

The key was right there in the file. Alternatives to `cargo install`: run
`cargo run --release --bin ddrs -- <args>`, or use an **absolute** path to
`target/release/ddrs` (a relative one resolves to the main tree from a worktree).

`cargo install` writes the machine-global `~/.cargo/bin/ddrs`, shared by every
checkout and worktree on the box. Installing from a worktree silently changes what
plain `ddrs` means everywhere until someone installs from a different checkout. When
several branches are in play, prefer the absolute `target/release/ddrs` of the tree
you mean.

**2. Pass `--workspace` whenever you pass `--config`.**

The workspace defaults to `.ddrs/` *beside the config*, so
`--config config/experiments/x.yaml` silently creates
`config/experiments/.ddrs/`, a fresh workspace with no run history and a cold
adjacency/baseline cache. Always:

```bash
ddrs --config config/experiments/x.yaml --workspace /path/to/repo/.ddrs plan
```

The one sanctioned exception is `examples/juniata*/ddrs.yaml`, whose workspace is
meant to land beside it.

**3. `plan` before `run`, and read the baseline it prints.**

`run` re-plans internally, but running `plan` first is where you see the summed-Q'
baseline, source drift, and the resolved adjacency cache **before** burning hours.
If the trained median NSE does not beat that baseline number, the routing is not
earning its keep.

## Start from a ready-made config, not from scratch

`ddrs.yaml` is gitignored and per-workspace; the committed configs are the
starting points. Copy one in rather than hand-building a config:

| Want | Copy |
|---|---|
| MERIT CONUS | `config/merit_training.yaml` |
| Gridded ISIMIP DDM30 CONUS | `config/experiments/gridded_conus.yaml` |
| Single-basin MERIT sample | `examples/juniata/ddrs.yaml` (run in place) |
| Single-basin gridded sample | `examples/juniata_gridded/ddrs.yaml` (run in place) |

```bash
cp config/experiments/gridded_conus.yaml ddrs.yaml   # gitignored; safe to overwrite
```

**`ddrs plan`'s interactive bootstrap does not offer these.** With no `ddrs.yaml`
present it offers exactly two choices: the last successful run's config snapshot,
or the clean template, and the clean template is hardcoded to
`config/merit_training.yaml` (`src/cli/plan_bootstrap.rs`). Bootstrapping and then
expecting a gridded config gets you a MERIT one. Copy the file you want first.

## Launch a training run

```bash
# from the repo root, with ./ddrs.yaml as the config
ddrs plan                                        # validate + baseline (cached)
ddrs run --workflow train-and-test               # train, eval, compare vs baseline
ddrs run --workflow train                        # train only, no eval
ddrs run --workflow train --max-mini-batches 2   # mechanics smoke, minutes not hours
ddrs run --workflow train-and-test --backend cpu # CPU (diagnostics; slower)
```

Always smoke-test a new config with `--max-mini-batches 2` first. It exercises the
whole data → forward → backward → checkpoint path in minutes.

`--workflow` must agree with the config's `mode:` (`training` ↔ `train` /
`train-and-test`, `testing` ↔ `eval`); `plan` rejects a contradiction at load.

## Switch datasets without editing YAML

Data-source groups are named save files under `config/sources/`. Switching is one
command and re-locks the workspace:

```bash
ddrs sources list                 # '*' marks the group matching the current config
ddrs sources use conus-gridded    # gridded ISIMIP DDM30 network
ddrs sources use conus            # MERIT CONUS
ddrs sources save my-group        # snapshot the current data_sources block
```

**`ddrs sources` resolves groups at `<config dir>/config/sources/`.** Run it from
the repo root against `./ddrs.yaml`. Pointing `--config` at
`config/experiments/foo.yaml` makes it look in `config/experiments/config/sources/`
and report *"no groups"*. The groups are fine, the lookup was wrong.

Switching is textual: the whole `data_sources:` block is replaced, so keys that
belong to only one network (`gridded_network`, `geospatial_fabric`, `aorc_precip`)
appear and disappear correctly. Everything outside the block is untouched, so
`geodataset:` and `params.subdivision` do **not** follow the switch.

Fix `geodataset:` after switching, or just delete the key. It is a provenance label
(written into managed-build zarr attrs and the run's `config.yaml` snapshot) and
nothing at runtime reads it, but since 2026-09-09 a value contradicting the
adjacency source is a **load error**, and an absent one is **inferred** from the
source. So omitting it is always right, and a stale one now stops the run instead of
silently mislabelling its audit trail. Only the two managed-build sources are
decidable; with explicit `conus_adjacency`/`gages_adjacency` any label is accepted,
because a pre-built store could be either network.

## Track a run

**While it runs**, every run tees a timestamped log:

`run` prints `run output → <run_dir>` to **stderr as it starts**, before training
begins, so take the path from there rather than guessing. If you already lost it:

```bash
tail -f "$(ls -dt .ddrs/runs/*/ | head -1)run.log"
```

It captures stdout+stderr at fd level (`dup2`), so CUDA driver messages and any
child process land there too, and it is opened `O_APPEND` and flushed per line, so
the log survives a crash. That makes `run.log` the first thing to read after a run
dies. Per-epoch lines carry loss and median `n`; a flat loss across epochs is a real
signal, not noise (see `ddrs-dev` traps).

**Afterwards:**

```bash
ddrs status                    # workspace summary: last run, lock state, disk usage
ddrs show <run-id>             # status, git SHA, adjacency, metrics, baseline delta
ddrs show <run-id> --json      # the full manifest, machine-readable
```

`ddrs show` prints a `metrics` block and, for a `train-and-test` run, the
summed-Q' baseline reduced to a median per metric with the routed value and the
delta beside it, flagging the case where routed does not beat baseline:

```
metrics
  median_nse_finite      0.7511
  median_kge_finite      0.7299
baseline (summed Q', median over gauges)
  NSE                    0.5943   routed 0.7511   Δ +0.1568
  KGE                    0.6715   routed 0.7299   Δ +0.0583
```

(Before 2026-09-09 text mode printed no metrics at all, so `ddrs show <id> | grep
nse` came back empty on a healthy run. If you are on an older binary and see that,
it is the binary, not the run.)

`.ddrs/runs/<id>/` holds `manifest.json`, `config.yaml` (the exact config that ran),
`run.log`, `checkpoints/`, and `Cargo.lock`. The manifest is the audit record:
`git` (sha, **dirty**, branch), `sources` (blake3 fingerprint per input file),
`source_lock.drift`, `resolved_adjacency`, and `metrics`.

**`metrics` differs by workflow.** A `train` run reports only
`epochs_completed`, `final_mini_batch`, `phase1_seconds`. There are no NSE/KGE
numbers, because nothing was evaluated. Skill numbers
(`median_nse_finite`, `median_kge_finite`, `n_gauges_total`, `phase2_seconds`) come
from `train-and-test`. Comparing a `train` run's metrics to a baseline is a category
error; re-run as `train-and-test`.

**If you read the baseline file yourself, reduce it first.** `train-and-test` copies
the baseline to `<run_dir>/baseline/manifest.json`, and its `metrics` are
**per-gauge arrays** (`nse`, `kge`, `rmse`, `bias`, `fhv`, `flv`, one entry per
gauge in `gage_ids` order), not pre-reduced medians. Comparing that raw array to the
run manifest's `median_nse_finite` compares a scalar against a 620-element list.
`ddrs show` does the reduction for you; `--json` does not, and `run` never re-prints
the baseline table that `plan` shows.

`git.dirty: true` means the working tree had uncommitted edits, so the SHA alone
does not reproduce that run.

## Resume after a crash

A checkpoint is a **directory** `.ddrs/runs/<id>/checkpoints/epoch_E_mb_M/` holding
`head.mpk`, `optim.mpk`, `state.json` (epoch, next mini-batch, RNG, sampler
permutation). To resume, set in the config:

```yaml
experiment:
  checkpoint: .ddrs/runs/<id>/checkpoints/epoch_6_mb_0
  epochs: 20        # MUST exceed the checkpoint's epoch or zero batches train
```

Resume restores all three files, so the run draws the same gauge batches and
windows the original would have. Weights are stored f16, so a resumed trajectory
drifts slowly from an uninterrupted one. That is expected, not a bug.

**A resume produces a NEW run id; it does not continue the old directory.** The run
id is built from the invocation's own start timestamp, so every `ddrs run` mints a
fresh `.ddrs/runs/<id>/`. The interrupted run's manifest, log, and checkpoints stay
frozen at the crash, and the resumed run starts an empty log of its own. Expect two
run ids for one training curve, and keep both: the first holds the epochs the second
does not.

Flat `epoch_E_mb_M.mpk` **files** instead of directories mean a stale pre-resume
binary wrote them. Go back to preflight step 1.

## Prune

```bash
ddrs gc --keep 5 --keep-successful --dry-run   # always dry-run first
ddrs gc --keep 5 --keep-successful
ddrs gc --older-than 30d
```

`gc` prunes only `.ddrs/runs/`. Adjacency and baseline caches are kept, because they are
content-addressed and expensive to rebuild.

## What does not work

| You might try | Reality |
|---|---|
| `ddrs run --workflow eval` | Fails: *"standalone --workflow eval needs a --from-run <run-id> flag"*, and `--from-run` is unimplemented. Use `train-and-test`, or the legacy `eval` binary against a checkpoint. |
| `ddrs init` | Removed. Prints *"ddrs init has been merged into ddrs plan"*. Use `ddrs plan`. |
| `ddrs experiment <name>` | **Not on master.** The subcommand and `src/experiment/` live on the `experiment-adjoint` branch. On master it exits with *"unrecognized subcommand"*. |
| `ddrs run --epochs 10` | No such flag. `experiment.epochs` is YAML-only; edit the config. |
| `ddrs run --checkpoint <dir>` | No such flag. Resume goes through `experiment.checkpoint:` in the config. |
| A typo'd config key | Now a load error naming the key. Every config section sets `deny_unknown_fields` as of 2026-09-09; before that a typo silently took its default, which was the most common cause of "my config change did nothing". On an older binary, that silence is still the first thing to suspect. |

## Red flags: stop and re-check preflight

- You edited `src/` and then typed `ddrs …` without `cargo install --path .`
- You passed `--config` without `--workspace`
- A config error names a key you did not set, or denies a key you did set
- `ddrs sources list` reports "no groups"
- Checkpoints are flat `.mpk` files rather than directories
- You are comparing a `train` run's metrics against a baseline

Full flag-by-flag reference: `references/commands.md`.
