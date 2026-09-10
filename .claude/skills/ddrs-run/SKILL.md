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
`geodataset:` and `params.subdivision` do **not** follow the switch. Check them by
hand when moving between MERIT and gridded.

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
ddrs show <run-id>             # status, workflow, git SHA, adjacency, drift
ddrs show <run-id> --json      # the FULL manifest, including metrics
```

**Text-mode `ddrs show` never prints metrics. `--json` is required for them.**
Verified 2026-09-09 on a finished train-and-test run: the human view has zero
NSE/KGE lines, so `ddrs show <id> | grep nse` returns nothing on a perfectly good
run and reads like a failed one.

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

**Comparing against the baseline takes one extra step.** `train-and-test` copies the
baseline to `<run_dir>/baseline/manifest.json`, but its `metrics` are **per-gauge
arrays** (`nse`, `kge`, `rmse`, `bias`, `fhv`, `flv`, one entry per gauge in
`gage_ids` order), not pre-reduced medians. The run manifest's
`median_nse_finite` is already a median, so take the median of the baseline array
yourself or you will compare a median against a 620-element list:

```bash
RUN=$(ls -t .ddrs/runs | head -1)
ddrs show "$RUN" --json | python3 -c 'import json,sys; m=json.load(sys.stdin)["metrics"]; print("routed  NSE", m["median_nse_finite"], "KGE", m["median_kge_finite"])'
python3 -c 'import json,statistics as s; m=json.load(open(".ddrs/runs/'"$RUN"'/baseline/manifest.json"))["metrics"]; print("baseline NSE", s.median(m["nse"]), "KGE", s.median(m["kge"]))'
```

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
| A typo'd config key | Silently takes its default almost everywhere. Only `kan_head.disaggregation` and `params.subdivision` set `deny_unknown_fields` (verified 2026-09-09). This is the most common cause of "my config change did nothing". |

## Red flags: stop and re-check preflight

- You edited `src/` and then typed `ddrs …` without `cargo install --path .`
- You passed `--config` without `--workspace`
- A config error names a key you did not set, or denies a key you did set
- `ddrs sources list` reports "no groups"
- Checkpoints are flat `.mpk` files rather than directories
- You are comparing a `train` run's metrics against a baseline

Full flag-by-flag reference: `references/commands.md`.
