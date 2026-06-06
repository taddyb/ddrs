# ddrs

Differentiable distributed routing. A BURN-based Rust port of the
Muskingum-Cunge routing solver from DDR (Python/PyTorch),
gradient-exact against the reference at single precision.

## Getting started

### Install

```bash
cargo install --path .
```

This puts the `ddrs` binary in `~/.cargo/bin/`. If that directory isn't on
your `PATH`:

```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### First-time setup

From your project root:

```bash
ddrs init      # creates ./.ddrs/, probes GPU, runs smoke test,
               # opens $EDITOR on ddrs.yaml, locks data sources
ddrs plan      # validates ddrs.yaml against locked sources, prints summary
ddrs run       # executes the workflow, writes manifest + outputs
```

`init` runs a 5-reach RAPID sandbox parity check on CUDA when available and
falls back to CPU otherwise — so the install path works on laptops and CI.
The bundled `config/merit_training.yaml` is the editor template; the
`workflow:` key is already set to `train-and-test`.

### What lives where

| Path | Written by | Purpose |
|---|---|---|
| `ddrs.yaml` | `ddrs init` (via `$EDITOR`) | Workflow + experiment config |
| `.ddrs/system.json` | `ddrs init` | GPU/driver/smoke-test record |
| `.ddrs/sources.lock` | `ddrs init` | Fingerprints of `data_sources` paths |
| `.ddrs/runs/<id>/manifest.json` | `ddrs run` | Per-run manifest (config + sources + git SHA + outputs) |
| `output/predictions_latest.zarr` | `ddrs run --workflow eval` / `train-and-test` Phase 2 | Predictions for plotting |
| `output/saved_models_*/epoch_*_mb_*.mpk` | `ddrs run --workflow train` / `train-and-test` Phase 1 | KAN checkpoints |

### Override workflow on the command line

The `workflow:` key in `ddrs.yaml` is what `plan`/`run` use by default. To
override for a single invocation:

```bash
ddrs plan --workflow eval
ddrs run --workflow train
```

`mode:` and `workflow:` must agree (`mode: training` ↔ `workflow ∈ {train, train-and-test}`; `mode: testing` ↔ `workflow: eval`). `ddrs init` will reject contradictions at load time.

The top-level `device:` key in `ddrs.yaml` selects the CUDA device ordinal
(default `0`, mirrors DDR's `device:` key) — on multi-GPU hosts set e.g.
`device: 1` to keep training off the display/shared GPU.

### Advanced

- `ddrs show <run_id>` — inspect a past run's manifest
- `ddrs status` — list runs
- `ddrs gc` — clean up old run directories
- `ddrs <cmd> --help` for full flag list
