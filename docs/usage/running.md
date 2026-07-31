# Running the code

This chapter covers how to build ddrs, plan and execute a run through the
`ddrs` CLI, drive the standalone diagnostic binaries, use the examples, and
flip between the CPU and CUDA paths.

The supported lifecycle is **`plan → run`**. There is no `ddrs init` step —
the `init` subcommand is a dead stub that prints
`ddrs init has been merged into ddrs plan — run \`ddrs plan\`` and exits with
status `2` (`src/bin/ddrs.rs:167-170`; asserted by `tests/cli_init_stub.rs`).

Configuration defaults come from `config/merit_training.yaml`, the verbatim
mirror of DDR's `merit_training_config.yaml`, which ships with
`sparse_solver: cuda` and `use_cuda_graphs: true`.

## What it is

`src/bin/` holds **ten** binaries. They fall into three groups, and only the
middle group is deprecated:

**1. The `ddrs` CLI — the supported entrypoint.** A terraform-style
lifecycle: `plan`, `run`, `show`, `import`, `sources`, `status`, `gc` (plus
the `init` stub above). Everything a run produces lands under
`.ddrs/runs/<id>/`; see [Reading outputs](outputs.md).

**2. The deprecated legacy trio** — `train` (Phase 1 only), `eval` (Phase 2
only), and `train_and_test` (both phases in one process). Each is
`clap`-parsed, shares the `--config <yaml>` shape, and prints a warning to
`stderr` before doing any work:

```
warning: `train` is deprecated and will be removed in 0.4. use `ddrs run --workflow train` instead.
```

(likewise for `eval` → `ddrs run --workflow eval` and `train_and_test` →
`ddrs run --workflow train-and-test`; see `tests/cli_deprecation_shim.rs`).
They were the original interface before the CLI landed. **One of them is
still load-bearing**: `eval --zeta-output` is the documented way to extract
the leakance zeta diagnostic from an existing checkpoint without retraining
(see [Diagnostic and tooling binaries](#diagnostic-and-tooling-binaries)).

**3. Six current, non-deprecated tool binaries** — `dump_parameters`,
`probe_zeta_gradient`, and the four `pretrain_disagg*` drivers. These print
no deprecation warning and have no `ddrs run` equivalent; they are the
research tooling around the trained head.

**Sixteen `examples/`** round out the set. Two are the standing regression
and sanity checks (`compare_ddr_sandbox`, `benchmark_hydrograph`); the other
fourteen are one-off diagnostics from the disaggregation and pretraining
work. Every example carries a `//!` module docstring stating what it does and
what it needs — read it before running one.

## How to use it

### Build

```bash
cargo build --release            # lib + the ten binaries (LTO=thin). NOT examples.
cargo build --release --examples # add the sixteen examples
cargo install --path .           # put `ddrs` on PATH (~/.cargo/bin/ddrs)
cargo test                       # CPU-only suite
```

`cargo build` does **not** build `examples/` — pass `--examples`, or use
`cargo run --example <name>`, which builds just the one you asked for.

The `--release` profile is mandatory for anything that touches the routing
core. Debug builds are roughly 20× slower and not useful for the V1
regression. The release profile also enables `lto = "thin"` (`Cargo.toml`
`[profile.release]`), which gives the fused routing chain a measurable extra
inlining win across the routing/sparse boundary.

> **⚠️ Stale-binary trap.** `cargo build` does not refresh
> `~/.cargo/bin/ddrs`. After editing `src/`, re-run `cargo install --path .`
> or invoke `cargo run --release --bin ddrs -- …` instead. The manifest's
> `git.sha` is stamped from `.git` at runtime, not from the binary, so a
> stale binary can produce a run that *looks* like current code. See
> `CLAUDE.md` for the full write-up and the self-check.

`cargo test` runs CPU-only (`burn-ndarray`) and needs no GPU: the
CUDA-specific and performance tests are `#[ignore]`d (`sp8_*`, `sp10_*`,
`sparse_cusparse_*`) or probe for a device and skip cleanly
(`tests/cusparse_ptr_spike.rs`, `tests/device_selection.rs`). The KAN-head
parity tests are behind a cargo feature and need `--features fixtures`. Don't
reason about coverage from a test count — use the gates in
[Verification matrix](#verification-matrix).

### Plan

```bash
ddrs plan                               # bootstrap + validate + cache
ddrs plan --workflow train-and-test     # override the config's `workflow:` key
ddrs plan --json                        # machine-readable PlanResult
ddrs plan --force                       # re-run the GPU smoke test
ddrs plan --min-free-gpu-gb 12          # warn threshold (default 8.0)
```

`plan` is idempotent and safe to run anytime. It probes the GPU (first run
only), runs a cached smoke test, bootstraps `./ddrs.yaml` if missing (opening
`$EDITOR`), locks the `data_sources:` paths into `.ddrs/sources.lock`,
validates the config, and builds the adjacency and summed-Q′ baseline caches.

It is **not side-effect-free**: the first plan reads ~370 MB of daily Qr for
the baseline and, when `geospatial_fabric` is configured instead of explicit
adjacency zarr paths, builds the managed adjacency stores into
`.ddrs/adjacency/<key>/`. Both are content-addressed; later plans on the same
inputs are cache hits.

Lock semantics: `plan` reports drift against `.ddrs/sources.lock`, then
refreshes the lock ("sources as of my last plan").

### Run

```bash
ddrs run --workflow train             # Phase 1 only
ddrs run --workflow train-and-test    # Phase 1 + Phase 2 + baseline comparison
```

`run` re-plans internally (as a library call, not a subprocess), does a GPU
pre-flight for `train`/`train-and-test` when `--backend cuda`, creates
`.ddrs/runs/<id>/`, snapshots `config.yaml` and `Cargo.lock` into it, and
tees stdout+stderr at the fd level into `run.log`. On failure it still writes
`manifest.json` (with `status: failed` and `exit_reason`) before exiting
non-zero; a panic inside the workflow is caught and recorded the same way.

Run ids are `<UTC timestamp>-[<group>-]<workflow>` — e.g.
`2026-06-12T14-02-10Z-global-train-and-test`. The `<group>` segment appears
when the config's `data_sources` block matches a saved
`config/sources/<name>.yaml`, so run dirs say which dataset they used.

`mode:` and `workflow:` must agree (`mode: training` ↔
`workflow ∈ {train, train-and-test}`; `mode: testing` ↔ `workflow: eval`);
`plan` rejects contradictions at load time.

Flags:

| Flag | Meaning |
|---|---|
| `--workflow <w>` | Override the `workflow:` key for this invocation |
| `--backend cuda\|cpu` | Default `cuda`. `cpu` binds `NdArray<f32>` and forces `sparse_solver=cpu`, `use_cuda_graphs=false`; it also skips the GPU pre-flight |
| `--plot` | After a successful run, dump per-COMID KAN parameters to `plot/kan_parameters.nc` |
| `--strict` | Exit `4` on data-source drift instead of warning and relocking. Aborts *before* the relock, so the evidence is preserved |
| `--max-mini-batches N` | Stop each training epoch after `N` mini-batches (debugging / profiling) |
| `--batch-order-from <PATH>` | Replay a captured mini-batch order from JSON (`[{"epoch":int,"mb":int,"staids":[str,…]}]`), overriding the per-epoch shuffle |
| `--json` | **Accepted but ignored** — `run` destructures it as `json: _` (`src/bin/ddrs.rs:213`). Read `manifest.json`, or use `ddrs show --json` |

`--plot` runs only when the workflow succeeded *and* a checkpoint exists; on
failure it prints `warning: --plot post-step failed: …` and the run is still
reported as OK, with `outputs.plot` left `null` in the manifest.

### Evaluating a checkpoint

> **`ddrs run --workflow eval` does not work.** It returns
> `"standalone --workflow eval needs a --from-run <run-id> flag; use
> --workflow train-and-test for now"` (`src/cli/run.rs:322-328`), and
> `--from-run` does not exist anywhere in `src/`.

There are two working paths:

**1. `--workflow train-and-test`** — the only way to get eval output from the
`ddrs` CLI. Phase 2 loads the latest Phase-1 checkpoint, writes
`eval/predictions.zarr`, prints the metrics summary, and copies the cached
summed-Q′ baseline into `baseline/`.

**2. The legacy `eval` binary** — for an *existing* checkpoint, with no
retraining:

```bash
cargo build --release --bin eval
target/release/eval \
    --config config/merit_training.yaml \
    --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_8 \
    --output /tmp/eval.zarr \
    --batch-size-days 15
```

`--checkpoint` points at the checkpoint **directory** `epoch_E_mb_M/`;
`eval` derives the recorder base with `head_base`, which returns `dir/head`
— it is `CompactRecorder` that appends `.mpk`
(`src/training/checkpoint.rs:103`). `--batch-size-days` defaults to `15`,
matching DDR's test config.

`--frozen` skips KAN-head loading and runs with scalar default parameters
(the V4 dev path). With `--frozen`, `--checkpoint` is optional; without
either, `eval` prints `--checkpoint is required unless --frozen is set` and
exits with status `2`:

```bash
target/release/eval \
    --config config/merit_training.yaml \
    --frozen \
    --output /tmp/v4_probe.zarr
```

`eval` seeds the backend RNG from `cfg.seed` for deterministic head-template
init, writes a DDR-compatible zarr at `--output`, and logs a metrics summary
on stdout: the count of gauges with finite NSE plus the **median** NSE and
KGE. Means are omitted because the NSE distribution is right-skewed — a few
bad gauges drag the mean. See [Reading outputs](outputs.md) for the zarr
layout.

To extract the leakance zeta diagnostic at the same time, add
`--zeta-output` (requires `params.use_leakance: true` and a non-`--frozen`
run):

```bash
target/release/eval \
    --config config/experiments/leakance_hourly_on.yaml \
    --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
    --output /tmp/eval.zarr \
    --zeta-output .ddrs/runs/<id>/kan_parameters.nc
```

It writes the `COMID_eval`-dimensioned NetCDF and prints the reach count,
median `|zeta|`, and the fraction above the `0.01 m³/s` GO/NO-GO bar. If no
zeta was accumulated it warns rather than failing.

### Checkpoints and resume

A checkpoint is a **directory** `epoch_E_mb_M/` holding three fixed-name
files:

| File | Contents | Format |
|---|---|---|
| `head.mpk` | KAN head weights | burn `CompactRecorder` (f16) |
| `optim.mpk` | Adam record (both moment tensors) | burn `CompactRecorder` (f16) |
| `state.json` | epoch, next mini-batch, serialized rng, sampler permutation + cursor | JSON |

Resume is a `ddrs` CLI feature, driven by `experiment.checkpoint:` in
`ddrs.yaml`:

```yaml
# ddrs.yaml
experiment:
  epochs: 50                                                   # must exceed the checkpoint's epoch
  checkpoint: .ddrs/runs/<run-id>/checkpoints/epoch_25_mb_8    # the directory
```

then `ddrs run --workflow train` (or `train-and-test`). On resume,
`bootstrap_head_and_state` (`src/training/bootstrap.rs`) restores all three
files, so the resumed run continues at the **true epoch / mini-batch** (the
learning-rate schedule keys correctly), draws the **same gauge batches** —
including the remainder of an in-flight epoch's shuffle — and the same
rho-windows, and steps Adam with **warm moments** rather than restarting
cold. Remember to raise `experiment.epochs` past the checkpoint's epoch, or
the resumed run trains zero batches.

Resume position is exact, but the stored weights and moments are f16
(`CompactRecorder` uses half-precision settings), so a resumed trajectory
drifts slowly from an uninterrupted one — a known follow-up tracked in
`docs/2026-06-07-checkpoint-resume-handoff.md`. Old checkpoints written
before the directory layout resume weights-only: Adam cold, epoch counter
back at 1, fresh shuffle.

Covered by `tests/checkpoint_resume.rs`.

### Config and workspace resolution

Two flags are **global** — they attach to every subcommand:

```bash
ddrs --config config/experiments/leakance_hourly_on.yaml plan
ddrs --config config/merit_training.yaml run --workflow train-and-test
ddrs --workspace /scratch/ddrs-ws run --workflow train
```

Without `--config`, `discover_config` walks up from the current directory
looking for `ddrs.yaml`, stopping at the first `.git` ancestor (inclusive).

> **⚠️ Workspace-beside-config trap.** The default workspace is `.ddrs/`
> **beside the config**, not beside the repo root
> (`src/bin/ddrs.rs:158-163`). So
> `ddrs --config config/experiments/x.yaml run …` puts the workspace at
> `config/experiments/.ddrs/` — a fresh workspace with no adjacency cache, no
> baseline cache, and none of your previous runs. **Always pass
> `--workspace` when you pass a non-root `--config`:**
>
> ```bash
> ddrs --config config/experiments/x.yaml --workspace .ddrs run --workflow train-and-test
> ```
>
> If no config is found at all, the workspace falls back to `./.ddrs`.

### Inspecting and managing the workspace

```bash
ddrs show <run-id>              # print a past run's manifest
ddrs show <run-id> --json       # …as JSON
ddrs status                     # runs, lockfile state, disk usage
ddrs status --json
ddrs gc --keep 5 --keep-successful
ddrs gc --older-than 30d --dry-run
```

`gc` takes `--keep N`, `--keep-successful`, `--older-than <duration>` (e.g.
`"30d"`, `"12h"`), and `--dry-run`. It **never** prunes `.ddrs/adjacency/` —
those caches are content-addressed and expensive to rebuild, so v1 leaves
them alone and prints a note saying so.

### Data-source groups and importing a Q′ store

Named "save files" for the `data_sources:` block, stored as
`config/sources/<name>.yaml`:

```bash
ddrs sources list                # '*' marks the group matching the config
ddrs sources save <name>         # snapshot the current data_sources (--force to overwrite)
ddrs sources use  <name>         # splice a group into the config + re-lock
```

`save`/`use` are textual — comments inside the block travel with the group,
everything outside it is untouched — and `use` validates that the spliced
config parses before committing, then refreshes `sources.lock` when a
workspace exists. Starting a global train from a CONUS workspace is
therefore `ddrs sources use global && ddrs plan --workflow train && ddrs run
--workflow train`.

Any store meeting the DDR Q′ contract (`docs/nh-qprime-store-contract.md`)
registers as a group in one command:

```bash
ddrs import <store> --dry-run          # validate + coverage report only
ddrs import <store> --name <group>     # validate + write config/sources/<group>.yaml
ddrs import <store> --name <group> --force   # overwrite an existing group
```

### Diagnostic and tooling binaries

These are current tools, not deprecated shims.

**`dump_parameters`** — sweep every CONUS reach through a trained head and
write the denormalised parameters as NetCDF on a `COMID` dimension
(`n`, `q_spatial`, `p_spatial`, `x_storage`, `slope`, plus `K_D`, `d_gw`,
`leakance_factor` when leakance is on):

```bash
cargo run --release --bin dump_parameters -- \
    --config config/merit_training.yaml \
    --checkpoint .ddrs/runs/<id>/checkpoints/epoch_30_mb_0/head \
    --output output/kan_parameters.nc \
    --batch-size 50000 \
    --backend cuda
```

> **`dump_parameters --checkpoint` is the recorder BASE path
> (`…/epoch_E_mb_M/head`), not the directory** — unlike `eval --checkpoint`,
> which takes the directory. This is the same base `ddrs run --plot` passes
> internally via `latest_checkpoint_base`.

`--batch-size` defaults to `50_000` (matches DDR's `geometry_predictor.py`
and fits comfortably on a 24 GB GPU); `--backend` defaults to `cuda`.

**`probe_zeta_gradient`** — the adjoint-reachability / identifiability probe
driver behind the leakance campaign. `--mode` selects one of `grad`
(default), `perturb`, `teacher`, `floor`, `state-cache`, `eval-loss`,
`landscape`; `--backend` defaults to **`cpu`** here (the GPU is usually busy
training). Key flags: `--checkpoint` (a checkpoint **directory**; omit for
the cold fresh-init point), `--windows` (default 32), `--seed` (default 42),
`--output`, `--params` (comma-separated head outputs; defaults to the
leakance trio), `--eval-days` (default 1095), `--chunk-days` (default 365).

```bash
cargo run --release --bin probe_zeta_gradient -- \
    --config config/experiments/leakance_hourly_on.yaml \
    --checkpoint .ddrs/runs/<id>/checkpoints/epoch_5_mb_9 \
    --windows 32 --seed 42 \
    --output output/grad_probe_trained.nc
```

> Teacher mode is memory-hungry: the default `--chunk-days 365` peaks at
> ~65 GB RSS over the 64,892-reach eval network. Drop to `--chunk-days 180`
> on a 93 GB desktop.

The spec is `docs/superpowers/specs/2026-07-02-zeta-gradient-probe-design.md`;
findings are in the `docs/2026-07-0*-leakance-*` series. Read
`docs/2026-07-06-leakance-nogo-scientific-summary.md` §3 before re-opening
that line of work.

**`pretrain_disagg`, `pretrain_disagg_blend`, `pretrain_disagg_capacity`,
`pretrain_disagg_window72`** — the four daily→hourly disaggregation-head
pretraining drivers. Each writes into `output/disagg_pretrain/`, which they
create themselves. Read the module docstring at the top of each before
running one; they are experiment variants, not interchangeable.

### V1 regression (compare_ddr_sandbox)

```bash
cargo run --release --example compare_ddr_sandbox
```

Reads the DDR-exported fixture under `fixtures/sandbox/` (gitignored —
regenerate via `cd ~/projects/ddr && uv run python
~/projects/ddrs/scripts/export_ddr_sandbox.py`), replays it through ddrs's
`MuskingumCunge` solver, reorders the output to RAPID2 order, and prints a
per-reach diff table followed by a verdict:

```
verdict: ABSOLUTE MATCH (max abs < 1e-3 m³/s)
```

This is the V1 invariant from `CLAUDE.md` — it must hold after every change
to `src/routing/`, `src/geometry.rs`, or `src/sparse/`. The example also
writes `output/ddrs_vs_ddr.csv` (per-reach max/mean abs diff, max rel diff,
means, Pearson correlation) and `output/ddrs_vs_ddr.png` (both hydrographs
overlaid) for visual inspection. If the overall max abs diff exceeds `1e-3`
but max rel diff is under 1%, the verdict softens to `close match`; beyond
that it reports `DIVERGENCE — investigate`.

By default the example runs on the CPU inner backend (`NdArray<f32>`) for
deterministic comparison. To dispatch through the CUDA inner backend
instead:

```bash
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox
```

Two things about this variable are easy to get wrong:

- **Any value triggers it.** The check is
  `std::env::var("DDRS_FORCE_GRAPHS").is_ok()`
  (`examples/compare_ddr_sandbox.rs:113`), so `DDRS_FORCE_GRAPHS=0` and
  `DDRS_FORCE_GRAPHS=` switch the backend just as `=1` does. Unset it to go
  back to CPU.
- **It does not enable graph capture, despite the name.** It only selects
  `Cuda<f32, i32>` as the inner backend. Capture is gated on
  `use_cuda_graphs && sparse_solver == Cuda && backend_is_cuda`
  (`src/routing/mmc.rs:289-293, 415-419`), and the sandbox builds its config
  from `Config::default()` — `use_cuda_graphs: false`,
  `sparse_solver: Cpu` — with `fixtures/sandbox/config.csv` setting neither
  (it only carries `range_n`, `range_q_spatial`,
  `log_space_parameters`, `p_spatial_default`, and geometry scalars). So the
  first two conjuncts are false and `try_capture_forward_graph` never fires.
  To exercise capture, use the `sp10_*` tests.

### Hydrograph plot (benchmark_hydrograph)

```bash
cargo run --release --example benchmark_hydrograph
```

Routes a synthetic diurnal lateral-inflow signal (5 m³/s baseline plus a
±2 m³/s sine sweep) through a 10-reach linear chain for 72 hourly steps
and writes:

- `output/hydrograph.csv` — wide CSV, columns `t_hours, reach_0..reach_9`,
  72 data rows.
- `output/hydrograph.png` — one line per reach, 1500×675 px at 150 dpi,
  styled to match DDR's `plot_routing_hydrograph`.

No fixtures required — useful as a sanity check on the routing core when
the V1 fixture is unavailable, or as a visual smoke test that the routing
core hasn't drifted between dev sessions. It also prints setup/forward
timings and per-reach min/mean/max discharge to the terminal.

### CPU vs CUDA toggles

Two YAML keys under `params:` in `config/merit_training.yaml` switch the
sparse path:

```yaml
params:
  sparse_solver: cuda    # cpu | cuda — selects ndarray vs cuSPARSE SpMV
  use_cuda_graphs: true  # CUDA backend only; forward-only graph capture+replay
```

The shipped defaults (the literal above) are CUDA-on; the *code* defaults, if
a key is absent, are `sparse_solver: cpu` and `use_cuda_graphs: false`
(`src/config.rs:407-411, 453-454`). On CPU-only machines you can override the
YAML, or just pass `--backend cpu` to `ddrs run` / `train` / `eval` /
`dump_parameters`, which patches both keys in memory.

`use_cuda_graphs: true` paired with `sparse_solver: cpu` is a silent no-op:
the captured kernel sequence assumes the cuSPARSE path, and the CPU sparse
solver has nothing to capture. `use_cuda_graphs: true` is also rejected
outright alongside `params.use_leakance: true` at config load
(`src/config.rs:746-749`) — the leakance kernel has no capture path.

A third, top-level key selects the GPU on multi-GPU hosts:

```yaml
device: 0              # CUDA device ordinal (mirrors DDR's `device:` key)
```

`device: 1` runs the entire training (tensors, cuSPARSE cache, graph
capture/replay) on the second GPU. It is read as `cfg.device` and passed to
`CudaDevice::new(...)` by both the CLI and the legacy binaries; validated by
`tests/device_selection.rs` (which skips on hosts with fewer than 2 GPUs).

See [Formatting inputs](inputs-formatting.md) for the complete list of
toggles, and [Performance & CUDA Graphs](../reference/perf.md) for the
capture architecture under the hood.

### Choosing the training objective

`experiment.loss.kind` selects the objective (`src/config.rs:212-226`,
`src/training/loss.rs:152-166`). Omit the `loss:` block and nothing changes —
the default is `l1`, the historical behaviour, byte-for-byte.

```yaml
experiment:
  loss:
    kind: nnse-kge     # l1 (default) | nnse-kge | kge
    nnse_weight: 1.0
    kge_weight: 1.0
    eps: 0.1
```

| `kind` | Objective |
|---|---|
| `l1` | `mean(\|p - o\|)`. Rewards peak attenuation |
| `nnse-kge` | Per gauge `nnse_weight·(1 - NNSE) + kge_weight·(1 - KGE)`. KGE's `(α-1)²` restores the hydrograph variance L1/NSE shrink away |
| `kge` | Per gauge `r_weight·(r-1)² + alpha_weight·(α-1)² + beta_weight·(β-1)² + nnse_weight·(1-NNSE)`, each gauge clamped at `kge_clamp`. Components weighted individually — no sqrt term, so no gradient singularity at perfect KGE, and `alpha_weight` can up-weight the anti-attenuation force |

Defaults for every weight are `1.0`; `kge_clamp` is `10.0` and `eps` is `0.1`
(matching DDR's `hydrograph_loss`). All metrics are per-gauge masked, then
averaged. Autograd is untouched — the loss is a drop-in scalar on the routed
predictions.

## Reference

### `ddrs` subcommands

| Subcommand | Flags |
|---|---|
| *(global)* | `--config <PATH>`, `--workspace <PATH>` |
| `plan` | `--workflow <w>`, `--json`, `--force`, `--min-free-gpu-gb <f32>` (default `8.0`) |
| `run` | `--workflow <w>`, `--backend cuda\|cpu` (default `cuda`), `--plot`, `--strict`, `--max-mini-batches <N>`, `--batch-order-from <PATH>`, `--json` (ignored) |
| `show` | `<run-id>`, `--json` |
| `import` | `<store>`, `--name <group>`, `--dry-run`, `--force` |
| `sources` | `save <name> [--force]`, `use <name>`, `list` |
| `status` | `--json` |
| `gc` | `--keep <N>`, `--keep-successful`, `--older-than <dur>`, `--dry-run` |
| `init` | *(dead stub — exits 2)* |

`<w>` is one of `train`, `eval`, `train-and-test`.

### Exit codes

From `src/cli/types.rs:15-23`:

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Generic (I/O or other) |
| 2 | Config invalid (also: the `init` stub) |
| 3 | Data source missing |
| 4 | Lock drift (only reachable via `run --strict`) |
| 5 | Runtime failure (includes a failed or panicking workflow) |
| 6 | Workspace not initialized |

### Legacy binary flags

| Binary | Flag | Type / default | Meaning |
|---|---|---|---|
| `train` | `--config` | path (required) | YAML config |
| `train` | `--checkpoint-dir` | path (required) | Per-mini-batch checkpoint directory |
| `train` | `--max-mini-batches` | usize (optional) | Cap mini-batches per epoch |
| `train` | `--backend` | `cuda` \| `cpu`, default `cuda` | `cpu` forces `sparse_solver=cpu`, `use_cuda_graphs=false` |
| `eval` | `--config` | path (required) | YAML config |
| `eval` | `--checkpoint` | path (optional) | Checkpoint dir `epoch_E_mb_M/`; required unless `--frozen` |
| `eval` | `--output` | path (required) | Output zarr path |
| `eval` | `--batch-size-days` | usize, default `15` | Days per chunk |
| `eval` | `--frozen` | flag | Use frozen scalar params instead of a KAN head |
| `eval` | `--zeta-output` | path (optional) | Per-reach leakance zeta NetCDF; needs `params.use_leakance: true` |
| `eval` | `--backend` | `cuda` \| `cpu`, default `cuda` | As above |
| `train_and_test` | `--config` | path (required) | YAML config |
| `train_and_test` | `--checkpoint-dir` | path (required) | Phase 1 writes / Phase 2 discovers here |
| `train_and_test` | `--output` | path (required) | Phase 2 predictions zarr |
| `train_and_test` | `--batch-size-days` | usize, default `15` | Days per chunk in Phase 2 |
| `train_and_test` | `--max-mini-batches` | usize (optional) | Cap Phase 1 mini-batches |

`train_and_test` has **no `--backend` flag — it is CUDA-only**
(`type I = Cuda<f32, i32>` is hardcoded at `src/bin/train_and_test.rs:80`).
Use `ddrs run --workflow train-and-test --backend cpu` if you need the CPU
path for the full pipeline.

Between phases, `train_and_test` auto-discovers the latest checkpoint
directory in `--checkpoint-dir` (parsing `epoch_E_mb_M` names and picking the
max by `(epoch, mb)` — numerically, so `epoch_25` outranks `epoch_9`),
reloads the config in `Testing` mode, and drops the Phase 1 optimizer and
dataset to free GPU memory first. Mirrors DDR's `scripts/train_and_test.py`.

### Examples

| Example | Purpose |
|---|---|
| `compare_ddr_sandbox` | V1 cross-language regression vs DDR (`< 1e-3 m³/s`); reads `fixtures/sandbox/`, writes `output/ddrs_vs_ddr.{csv,png}` |
| `benchmark_hydrograph` | Fixture-free routing sanity check; writes `output/hydrograph.{csv,png}` |
| `dump_init_params` | Sweep all CONUS reaches through a freshly-initialised head |
| `save_random_kan` | Throwaway smoke helper: build a `KanHead` from the YAML's `kan_head` section |
| `kan_sensitivity_sweep`, `kan_disagg_trained_sensitivity` | KAN-interpretability sensitivity sweeps |
| `disagg_boundary_verification`, `kan_disagg_mass_balance_real` | Disaggregation mass-balance / boundary verification |
| `disagg_precip_normalization_discriminator`, `disagg_transfer_diagnostic`, `kan_disagg_real_storm_shift` | Disaggregation diagnostics |
| `pretrain_disagg_verify`, `pretrain_reconciliation_check` | Pretraining verification |
| `pretrain_disagg_storm_compare`, `pretrain_disagg_capacity_storm_compare`, `pretrain_disagg_window72_storm_compare` | Storm-day visual comparisons per pretraining variant |

Only the first two are standing gates. The rest are experiment-specific —
read each file's `//!` docstring for its inputs and assumptions.

### Verification matrix

| Path | Covered by |
|---|---|
| Config parse + ranges | `cargo test --lib config::` |
| Routing core (dense + sparse, CPU) | `cargo test --test mmc` |
| Sparse autograd (gradcheck) | `cargo test --test sparse_gradcheck` |
| Data readers (zarr/netcdf) | `cargo test --test data_zarr_store` |
| Checkpoint save/resume round-trip | `cargo test --test checkpoint_resume` |
| CLI lifecycle (plan/run/show/status/gc/lock) | `cargo test --test cli_plan --test cli_manifest --test cli_status_gc --test cli_lockfile` |
| Managed adjacency == engine-built | `cargo test --test adjacency_parity` |
| KAN head parity vs DDR | `cargo test --features fixtures --test kan_head_init_repro --test kan_head_init_parity --test kan_head_fixture_forward --test kan_head_fixture_backward` |
| Leakance gradient exactness | `cargo test --test leakance_gradcheck` |
| Leakance off ⇒ byte-identical | `cargo test --test leakance_off_parity` |
| Eval zeta == what's subtracted from `b` | `cargo test --test zeta_accum` |
| End-to-end bit-match vs DDR | `cargo run --release --example compare_ddr_sandbox` |

The V1 example is the only test that locks the cross-language invariant; the
`cargo test` suite covers everything else CPU-side and runs without CUDA. The
four leakance/KAN gate rows are mandatory on any PR touching
`src/routing/leakance.rs`, the leakance backward op, `src/nn/`, or the
`rskan` pin — see `CLAUDE.md`.

### Gotchas

- **Re-install `ddrs` after every `src/` change.** `cargo build` does not
  refresh `~/.cargo/bin/ddrs`. See [Build](#build).
- **`--workspace` when `--config` is not at the repo root.** Otherwise the
  workspace materializes beside the config and you lose every cache and past
  run. See [Config and workspace resolution](#config-and-workspace-resolution).
- **`ddrs run --workflow eval` errors out**, and `ddrs init` exits 2. Use
  `train-and-test`, or the legacy `eval` binary.
- **`ddrs run --json` is silently ignored.** Read `manifest.json` or use
  `ddrs show --json`.
- **Checkpoint directory is auto-created — but only by `train`.** `train.rs`
  and `train_and_test.rs` call `std::fs::create_dir_all(&cli.checkpoint_dir)`
  before training (`train_and_test` also creates the parent of `--output`).
  `eval` calls `create_dir_all` **nowhere**.
- **`eval --checkpoint` is a directory; `dump_parameters --checkpoint` is a
  base path.** `eval` takes `epoch_E_mb_M/` and derives `dir/head` itself;
  `dump_parameters` takes `epoch_E_mb_M/head` directly. Neither wants a path
  ending in `.mpk` — the recorder appends that, so you'd get `head.mpk.mpk`.
- **Data files must exist.** The binaries fail on missing files at the paths
  listed in [Setup](../setup.md) (MERIT fabric/zarr, attributes netcdf,
  icechunk forcing/observations). If a path differs on a new machine, edit
  the config rather than symlinking. A `kan_head.disaggregation:` block with
  no `data_sources.aorc_precip` is a hard error at dataset open
  (`src/data/dataset.rs:439-445`: *"the disaggregation head always requires
  precip"*), not a silent degrade to flat-daily.
- **`output/` must exist for the examples.** Both `compare_ddr_sandbox` and
  `benchmark_hydrograph` write CSV+PNG directly to `output/` and panic on
  `BufWriter::new(File::create(...))` if the directory is missing. Create it
  once: `mkdir -p output`.
- **`fixtures/sandbox/` is gitignored.** Missing fixtures →
  `compare_ddr_sandbox` panics at the first CSV read. Regenerate via the DDR
  `uv` venv (see [Setup](../setup.md)).
- **KAN-head checkpoints are not transferable from DDR.** `eval` accepts only
  ddrs-trained `.mpk` files; DDR's `.pt` weights match the I/O contract but
  not the internal record layout.

## See also

- [Setup](../setup.md) — the prerequisites these commands assume.
- [Formatting inputs](inputs-formatting.md) — what's in
  `config/merit_training.yaml` and how to edit it.
- [Reading inputs](inputs-reading.md) — what the data-source paths point at.
- [Reading outputs](outputs.md) — the `.ddrs/runs/<id>/` tree and every
  artefact in it.
- [Comparing to DDR](../reference/ddr-comparison.md) — the V1 regression in
  detail.
- [The summed Q′ baseline](../reference/baseline.md) — the reference every
  train-and-test run is judged against.
- [Performance & CUDA Graphs](../reference/perf.md) — what the CUDA toggles
  actually do.
