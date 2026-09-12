# Matched-batch replay parity experiment — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether forcing DDR and DDRS to see the same mini-batches in
the same order eliminates the per-reach `n` scrambling (Spearman 0.347)
that the area-pool fix didn't address.

**Architecture:** Capture DDR's per-`(epoch, mb)` gauge order to JSON by
wrapping `torch.utils.data.RandomSampler` with a logging subclass (no
DDR-side source modification — a thin wrapper script monkey-patches the
sampler before calling DDR's hydra entry point). Add a `BatchSource` enum
to DDRS's `src/data/sampler.rs` that can either generate batches via
shuffle (current behavior) or replay from a captured order. Plumb through
the training driver via a new `--batch-order-from <path>` CLI flag.
Retrain DDRS using DDR's captured order; compare per-reach `n` via PR
#12's `parity_trained` notebook.

**Tech Stack:** Rust 2021 + burn 0.21 (existing). Python wrapper script
in `scripts/` using `hydra` (already DDR's invocation pattern) + JSON.
No new dependencies on either side.

**Spec source of truth:**
`docs/superpowers/specs/2026-06-04-matched-batch-replay-design.md`.

**Branch:** Stay on `training-step-parity` (PR #13).

---

## Pre-flight verified

- DDR's batch construction: `~/projects/ddr/scripts/train.py:40-50`.
  Uses `torch.Generator().manual_seed(cfg.seed)` →
  `RandomSampler(dataset, generator=data_generator)` →
  `DataLoader(batch_size=cfg.experiment.batch_size, sampler=sampler,
  collate_fn=dataset.collate_fn, drop_last=True)`. Per-epoch the sampler
  is re-iterated (PyTorch re-shuffles via the generator's continuing state).
- DDR's entry point: `~/projects/ddr/scripts/train_and_test.py` (hydra
  decorator on `main()` — invokable from a wrapper script).
- DDRS's batch construction: `src/training/driver.rs:62-70`.
  `RandomSampler::new(dataset.len(), exp.batch_size, true)` (drop_last=true);
  `sampler.reshuffle(&mut state.rng)` at the top of each epoch;
  `sampler.next_batch()` yields `Option<Vec<usize>>` (indices); converted
  to STAIDs via `dataset.staids()[i]`.
- DDRS's `RandomSampler`: `src/data/sampler.rs:10-58`. Index-based;
  no STAID awareness.
- PR #12's parity_trained notebook recipe:
  `.claude/skills/ddrs-eval-plots/references/parity_trained.md`.

---

## File structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `scripts/dump_ddr_batch_order_and_train.py` | create | Python wrapper that monkey-patches `torch.utils.data.RandomSampler` with a logging subclass, then calls DDR's `train_and_test.py` `main()` via hydra. Writes `/tmp/ddr_batch_order.json` AND produces the DDR checkpoint we'll later compare against. |
| `src/data/sampler.rs` | modify | Add `BatchSource` enum + `ReplaySampler` struct. Keep `RandomSampler` unchanged so the existing path is byte-for-byte preserved. |
| `src/training/driver.rs` | modify | Plumb `BatchSource` through the training loop. When `Replay`, iterate the externally-provided batches; when `Shuffle`, use the existing `RandomSampler`. |
| `src/cli/run.rs` | modify | Add `--batch-order-from <path>` argument to `ddrs run`. Loads the JSON and converts to `BatchSource::Replay`. |
| `docs/superpowers/specs/2026-06-04-matched-batch-replay-design.md` | modify | Append §5.1 verdict. |
| `/tmp/ddr_batch_order.json` | create (transient) | DDR's full `(epoch, mb_idx, [staids])` trace. Not committed. |

---

## Task 1: DDR-side wrapper script + capture run

**Spec ref:** §5.1.

**Files:**
- Create: `/home/tbindas/projects/ddrs/scripts/dump_ddr_batch_order_and_train.py`

- [ ] **Step 1: Inspect DDR's training entry**

```bash
sed -n '1,30p' ~/projects/ddr/scripts/train_and_test.py
grep -nE '@hydra|def main|RandomSampler' ~/projects/ddr/scripts/train_and_test.py | head
```

Confirm:
- `train_and_test.py` has a hydra-decorated `main(cfg: DictConfig)` entrypoint.
- The training loop inside instantiates `torch.utils.data.RandomSampler`
  somewhere (either directly or by delegating to `train.py:train(...)`).

If `train_and_test.py` just calls `train.py::train(...)`, the monkey-patch
applied at module-import time still catches the sampler because Python
resolves `RandomSampler` at instantiation time.

- [ ] **Step 2: Write the wrapper script**

Create `/home/tbindas/projects/ddrs/scripts/dump_ddr_batch_order_and_train.py`:

```python
"""Wrapper around DDR's train_and_test.py that captures per-mini-batch
gauge order to JSON.

Approach: monkey-patch `torch.utils.data.RandomSampler` with a subclass
that records `__iter__` output to a module-global list. DDR's training
script instantiates the sampler normally; iteration logs as a side effect.

Output: a JSON file with the schema
    [
        {"epoch": 1, "mb": 0, "staids": ["12345678", ...]},
        ...
    ]

Run under DDR's uv venv:
    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_ddr_batch_order_and_train.py \
        --config-name=merit_training_config \
        +dump_batch_order_to=/tmp/ddr_batch_order.json
"""

import json
import sys
from pathlib import Path
from typing import Any

# Ensure DDR's src/ is on path before importing.
sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "scripts"))

import torch
import torch.utils.data as _torch_data

_ORIGINAL_RANDOM_SAMPLER = _torch_data.RandomSampler

# Module-globals for the side-effect log.
_BATCH_ORDER_LOG: list[dict[str, Any]] = []
_EPOCH_COUNTER = [0]   # mutable cell so closures can bump it
_MB_COUNTER = [0]

class LoggingRandomSampler(_ORIGINAL_RANDOM_SAMPLER):
    """Drop-in replacement that records the index permutation each epoch.

    Each call to __iter__ corresponds to a new pass over the dataset (epoch).
    We materialize the iterator's output to a list so we can both log it
    AND yield it to the DataLoader.
    """

    def __iter__(self):
        _EPOCH_COUNTER[0] += 1
        _MB_COUNTER[0] = 0
        indices = list(super().__iter__())
        # Stash indices for later flush — STAID conversion needs the dataset
        # reference, which we capture via the sampler's data_source attribute.
        dataset = self.data_source
        # Materialize batches in-order to match DataLoader's drop_last=True
        # batching. Read batch_size from cfg via a module-global fallback —
        # but the simpler approach is: log raw index list, post-process in
        # Step 4 below when we know batch_size.
        _BATCH_ORDER_LOG.append({
            "epoch": _EPOCH_COUNTER[0],
            "indices": indices,
            "dataset_staids": [str(s) for s in dataset.staids],
        })
        return iter(indices)


def _patch():
    _torch_data.RandomSampler = LoggingRandomSampler
    # Some DDR code does `from torch.utils.data import RandomSampler` —
    # the alias is resolved at import time, so patch the already-imported
    # module references as well.
    import ddr.datasets  # noqa: F401 — force-import so the alias resolves
    for mod_name in list(sys.modules.keys()):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        rs = getattr(mod, "RandomSampler", None)
        if rs is _ORIGINAL_RANDOM_SAMPLER:
            setattr(mod, "RandomSampler", LoggingRandomSampler)


def _flush(path: str, batch_size: int, drop_last: bool):
    """Convert the raw per-epoch index log into per-(epoch, mb, staids)
    records and write to JSON."""
    out: list[dict[str, Any]] = []
    for entry in _BATCH_ORDER_LOG:
        indices = entry["indices"]
        staids = entry["dataset_staids"]
        epoch = entry["epoch"]
        # Walk the indices in batches of batch_size, matching DataLoader's
        # behavior with drop_last.
        n = len(indices)
        mb = 0
        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n:
                if drop_last:
                    break
                end = n
            batch_indices = indices[start:end]
            out.append({
                "epoch": epoch,
                "mb": mb,
                "staids": [staids[i] for i in batch_indices],
            })
            mb += 1
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} mini-batch records → {path}")


# Apply patch before importing DDR's train entrypoint.
_patch()

# Now import + run DDR's main as-is. The hydra decorator will resolve from
# the CLI args.
import hydra
from omegaconf import DictConfig

from train_and_test import main as _ddr_main  # noqa: E402 — imports after patch


@hydra.main(config_path=str(Path.home() / "projects" / "ddr" / "config"),
            config_name="merit_training_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Run DDR's training. This will iterate our logging sampler.
    _ddr_main(cfg)
    # After training, flush the captured order.
    dump_path = cfg.get("dump_batch_order_to", "/tmp/ddr_batch_order.json")
    _flush(dump_path,
           batch_size=cfg.experiment.batch_size,
           drop_last=True)


if __name__ == "__main__":
    main()
```

**Important caveats** for this script — keep them in head while writing:

1. The `_patch()` function uses a string-based search through
   `sys.modules` to catch already-imported `from ... import RandomSampler`
   aliases. If a module imports the sampler AFTER `_patch()` runs, the
   alias will resolve correctly via `_torch_data.RandomSampler`. The
   `import ddr.datasets` line forces DDR's modules to load before we
   patch already-imported aliases.
2. `cfg.experiment.batch_size` must be 64 (the parity config). Verify
   before running.
3. `drop_last=True` matches `train.py:51`. Hardcode in `_flush()` rather
   than re-reading from cfg.

- [ ] **Step 3: Smoke-test the patch logic without training**

Before kicking off a 45-min training run, verify the monkey-patch works.
Tiny test:

```bash
cd /home/tbindas/projects/ddrs && cat > /tmp/test_patch.py <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
sys.path.insert(0, str(Path.home() / "projects" / "ddrs" / "scripts"))

# Import the wrapper module to trigger _patch()
import importlib.util
spec = importlib.util.spec_from_file_location(
    "wrapper",
    "/home/tbindas/projects/ddrs/scripts/dump_ddr_batch_order_and_train.py",
)
wrapper = importlib.util.module_from_spec(spec)
# We don't want to run main(), just trigger module-level _patch().
# Read the source manually and exec up through _patch() call.
# (Alternative: refactor _patch() to be importable; either works.)

import torch.utils.data as td
print(f"Before patch: RandomSampler = {td.RandomSampler.__name__}")
spec.loader.exec_module(wrapper)
print(f"After patch:  RandomSampler = {td.RandomSampler.__name__}")
PY
cd ~/projects/ddr && uv run python /tmp/test_patch.py 2>&1 | tail -5
```

Expected: prints `After patch:  RandomSampler = LoggingRandomSampler`.

If `_patch()` reference is broken (e.g., the wrapper's `@hydra.main` decorator
errors on import), fix before continuing. Worst case: refactor `_patch()`
to be guarded behind `if __name__ != "__main__"` so it runs on import only
when imported by the test harness — but that's a structural change. Simpler:
make `_patch()` execute at module top-level (which it already does).

- [ ] **Step 4: Run DDR training with capture**

```bash
cd ~/projects/ddr && uv run python \
    /home/tbindas/projects/ddrs/scripts/dump_ddr_batch_order_and_train.py \
    --config-name=merit_training_config \
    +dump_batch_order_to=/tmp/ddr_batch_order.json \
    2>&1 | tee /tmp/ddr_train.log
```

Expected: ~30-45 min runtime. Output ends with
`wrote <N> mini-batch records → /tmp/ddr_batch_order.json` where N is
roughly 5 epochs × 35 mb = 175 records (depends on actual dataset size
and `drop_last`).

Capture the DDR checkpoint that's produced — the output directory will
be something like
`~/projects/ddr/output/ddr-vXXX-merit-training/<timestamp>/saved_models/`.
Record the path to the final `epoch_5_mb_*.pt` checkpoint for Task 5.

- [ ] **Step 5: Sanity-check the JSON**

```bash
cd ~/projects/ddr && uv run python -c "
import json
records = json.loads(open('/tmp/ddr_batch_order.json').read())
print(f'records: {len(records)}')
print(f'epochs: {sorted({r[\"epoch\"] for r in records})}')
print(f'first 2 records:')
for r in records[:2]:
    print(f'  epoch={r[\"epoch\"]} mb={r[\"mb\"]} batch_size={len(r[\"staids\"])} first_staid={r[\"staids\"][0]}')
"
```

Expected: ~175 records across 5 epochs, each with 64 STAIDs, batch sizes
all equal (since drop_last=True).

If batch_size != 64, the wrapper read the wrong cfg field — diagnose.

If only 1 epoch worth of records appears, the `__iter__` patching isn't
catching DataLoader's per-epoch re-iteration — DDR may be using a
non-standard iteration pattern; revisit the `LoggingRandomSampler`
design.

- [ ] **Step 6: Commit the wrapper script**

```bash
cd /home/tbindas/projects/ddrs
git add scripts/dump_ddr_batch_order_and_train.py
git commit -m "$(cat <<'EOF'
scripts: capture DDR's per-mini-batch gauge order via sampler monkey-patch

Wrapper around DDR's train_and_test.py that replaces torch.utils.data.
RandomSampler with a logging subclass (no DDR-side source modification).
Records every per-epoch index permutation, then post-processes into
(epoch, mb, [staids]) records keyed by STAID and writes to JSON.

The captured trace lets the matched-batch parity experiment (PR #13
spec/plan: 2026-06-04-matched-batch-replay-design.md) feed DDRS's
training loop exactly the same gauge ordering DDR used at the same
seed=42, isolating "different operations" from "different batch
inputs" as causes of per-reach n divergence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

The `/tmp/ddr_batch_order.json` file is transient and gitignored
(under `/tmp`).

---

## Task 2: DDRS `BatchSource` enum + plumbing

**Spec ref:** §5.2.

**Files:**
- Modify: `src/data/sampler.rs` (add `BatchSource` + `ReplaySampler`)
- Modify: `src/training/driver.rs` (use `BatchSource` instead of raw `RandomSampler`)

The current `RandomSampler::new(n, batch_size, drop_last)` + `reshuffle` +
`next_batch` API stays unchanged. We add a new abstraction layer above it
that can switch between shuffle and replay.

- [ ] **Step 1: Write the failing test**

Add to `src/data/sampler.rs`'s existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn replay_sampler_yields_recorded_batches_in_order() {
        // STAIDs as registered with the dataset.
        let all_staids: Vec<crate::data::ids::Staid> = (0..10)
            .map(|i| crate::data::ids::Staid(format!("STAID_{i:02}")))
            .collect();
        // Record: epoch 1 has 2 mini-batches, epoch 2 has 1 (incomplete).
        let recorded: Vec<Vec<crate::data::ids::Staid>> = vec![
            vec![all_staids[3].clone(), all_staids[1].clone()],
            vec![all_staids[7].clone(), all_staids[2].clone()],
            vec![all_staids[5].clone(), all_staids[8].clone()],
        ];

        let mut replay = ReplaySampler::new(recorded, &all_staids);

        // Each next_batch returns the indices into all_staids that the
        // recorded STAIDs map to.
        let b1 = replay.next_batch().expect("batch 1");
        assert_eq!(b1, vec![3, 1]);
        let b2 = replay.next_batch().expect("batch 2");
        assert_eq!(b2, vec![7, 2]);
        let b3 = replay.next_batch().expect("batch 3");
        assert_eq!(b3, vec![5, 8]);
        assert!(replay.next_batch().is_none(),
            "ReplaySampler should exhaust after the recorded batches");
    }

    #[test]
    fn replay_sampler_rejects_unknown_staid() {
        let all_staids = vec![crate::data::ids::Staid("KNOWN".into())];
        let recorded = vec![vec![crate::data::ids::Staid("UNKNOWN".into())]];
        let result = std::panic::catch_unwind(|| {
            let _ = ReplaySampler::new(recorded, &all_staids);
        });
        assert!(result.is_err(),
            "ReplaySampler::new should panic when a recorded STAID is not in the dataset");
    }
```

- [ ] **Step 2: Run the failing tests**

```bash
cd /home/tbindas/projects/ddrs && cargo test --lib data::sampler::tests::replay 2>&1 | tail -10
```

Expected: 2 failures — `ReplaySampler` doesn't exist yet.

- [ ] **Step 3: Implement `ReplaySampler`**

Add to `src/data/sampler.rs` (after the existing `SequentialSampler`,
before the `#[cfg(test)]` block):

```rust
use crate::data::ids::Staid;

/// Yields pre-recorded mini-batches in order. Companion to `RandomSampler`
/// for the matched-batch replay experiment (PR #13 spec
/// `2026-06-04-matched-batch-replay-design.md`).
pub struct ReplaySampler {
    /// Each inner Vec is one mini-batch of dataset-row indices.
    batches: Vec<Vec<usize>>,
    cursor: usize,
}

impl ReplaySampler {
    /// Build from a list of mini-batches of STAIDs. Resolves each STAID
    /// against the dataset's `all_staids` array to recover dataset-row
    /// indices. Panics if any recorded STAID is not present in `all_staids`.
    pub fn new(recorded: Vec<Vec<Staid>>, all_staids: &[Staid]) -> Self {
        // STAID → row index map. Built once.
        let lookup: std::collections::HashMap<&Staid, usize> = all_staids
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        let batches: Vec<Vec<usize>> = recorded
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|s| {
                        *lookup.get(&s).unwrap_or_else(|| {
                            panic!(
                                "ReplaySampler: STAID {s:?} from recorded order not \
                                 present in dataset's all_staids (len={})",
                                all_staids.len()
                            )
                        })
                    })
                    .collect()
            })
            .collect();
        Self { batches, cursor: 0 }
    }

    /// Yield the next pre-recorded batch's row indices, or `None` if exhausted.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.cursor >= self.batches.len() {
            return None;
        }
        let out = self.batches[self.cursor].clone();
        self.cursor += 1;
        Some(out)
    }

    /// Number of mini-batches in the replay queue. Used for progress logging.
    pub fn remaining(&self) -> usize {
        self.batches.len().saturating_sub(self.cursor)
    }
}
```

Also add a `BatchSource` enum at the top of the file (right after the
existing `pub struct RandomSampler` block, before `impl RandomSampler`):

```rust
/// Source of mini-batches for the training loop. Either generates batches
/// on the fly via `RandomSampler` (default) or replays a captured order
/// via `ReplaySampler` (matched-batch parity experiment).
pub enum BatchSource {
    Shuffle(RandomSampler),
    Replay(ReplaySampler),
}

impl BatchSource {
    /// Re-shuffle the underlying sampler for a new epoch. No-op for replay
    /// (replay always traverses the entire pre-recorded sequence
    /// regardless of epoch — the caller should construct a fresh BatchSource
    /// per "logical epoch" boundary or just feed the entire training trace
    /// as one Replay).
    pub fn reshuffle<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) {
        if let BatchSource::Shuffle(s) = self {
            s.reshuffle(rng);
        }
    }

    /// Get the next batch's row indices, or None if the source is exhausted.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        match self {
            BatchSource::Shuffle(s) => s.next_batch(),
            BatchSource::Replay(s) => s.next_batch(),
        }
    }
}
```

- [ ] **Step 4: Verify tests pass**

```bash
cd /home/tbindas/projects/ddrs && cargo test --lib data::sampler 2>&1 | tail -15
```

Expected: all sampler tests pass, including the 2 new replay tests.

- [ ] **Step 5: Plumb `BatchSource` through `src/training/driver.rs`**

Read the current loop (`src/training/driver.rs:62-90` per pre-flight):

```bash
sed -n '55,100p' /home/tbindas/projects/ddrs/src/training/driver.rs
```

The current code:
```rust
let mut sampler = RandomSampler::new(dataset.len(), exp.batch_size, true);

for epoch in state.epoch..=exp.epochs {
    sampler.reshuffle(&mut state.rng);
    let lr = resolve_lr(&exp.learning_rate, epoch);
    eprintln!("epoch {epoch} lr={lr}");
    let mut mb_done = 0usize;
    while let Some(idx) = sampler.next_batch() {
        // ...
    }
}
```

Replace the local `sampler` with a `BatchSource` taken via a new function
parameter. Signature change:

```rust
pub fn train<I: Backend>(
    cfg: &Config,
    dataset: &MeritGagesDataset,
    state: &mut TrainState<I>,
    optimizer: &mut impl Optimizer<KanHead<Autodiff<I>>, Autodiff<I>>,
    device: &I::Device,
    checkpoint_dir: &Path,
    max_mini_batches: Option<usize>,
    batch_source: Option<BatchSource>,   // NEW — None = use default RandomSampler
) -> Result<()> {
    // ...
    let mut sampler = batch_source.unwrap_or_else(|| {
        BatchSource::Shuffle(RandomSampler::new(dataset.len(), exp.batch_size, true))
    });

    for epoch in state.epoch..=exp.epochs {
        sampler.reshuffle(&mut state.rng);
        // ... rest unchanged
    }
}
```

Import `BatchSource`:
```rust
use crate::data::sampler::{BatchSource, RandomSampler};
```

(Remove the now-redundant standalone `RandomSampler` import if it was
the only consumer.)

- [ ] **Step 6: Update callers of `train(...)`**

```bash
grep -rEn 'training::train\(|training::driver::train\(' /home/tbindas/projects/ddrs/src /home/tbindas/projects/ddrs/tests /home/tbindas/projects/ddrs/examples 2>/dev/null | head
```

Each call site needs to pass `None` as the new last argument (default
behavior unchanged). Update each.

- [ ] **Step 7: Build + run the existing test suite**

```bash
cd /home/tbindas/projects/ddrs && cargo build --lib 2>&1 | tail -5
cargo test --lib 2>&1 | tail -10
cargo test --features fixtures --test training_step_layer_b --test training_step_layer_c --test training_step_layer_d 2>&1 | tail -5
```

Expected:
- Build clean.
- All lib tests pass (including 2 new replay tests).
- All 9 training-step parity tests still pass (they're integration tests
  that don't exercise the new `batch_source` argument; they call the
  existing forward/backward paths).

- [ ] **Step 8: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add src/data/sampler.rs src/training/driver.rs
# Plus any caller updates:
git add src/cli/run.rs src/bin/ tests/ examples/  # if any changed
git commit -m "$(cat <<'EOF'
feat(data): BatchSource enum with Shuffle + Replay variants

Refactors src/training/driver.rs::train() to take an optional
BatchSource argument. Default (None) preserves the existing
RandomSampler behavior; passing Some(BatchSource::Replay(...))
plays back a pre-recorded mini-batch sequence keyed by STAID.

ReplaySampler in src/data/sampler.rs handles the STAID → row-index
resolution and yields batches in order. Unit tests cover the basic
yield-in-order contract and the unknown-STAID panic case.

Next commit plumbs --batch-order-from <path> through src/cli/run.rs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `--batch-order-from` CLI flag

**Spec ref:** §5.2.

**Files:**
- Modify: `src/cli/run.rs` (add CLI argument + JSON parsing + BatchSource construction)
- Modify: `src/cli/types.rs` if necessary (if there's a shared input struct)

- [ ] **Step 1: Find the existing `ddrs run` argument struct**

```bash
grep -nE 'struct.*RunInput|pub struct.*Run|#\[derive\(Parser\)\]|clap::Parser' /home/tbindas/projects/ddrs/src/cli/run.rs /home/tbindas/projects/ddrs/src/cli/types.rs /home/tbindas/projects/ddrs/src/bin/ddrs.rs 2>/dev/null | head -15
```

Identify the clap-derived struct that backs `ddrs run` (likely
`RunInput` or `RunCommand` in one of these files). The flag goes onto
that struct.

- [ ] **Step 2: Add the `--batch-order-from` flag**

To the struct found in Step 1, add:

```rust
    /// Replay a captured mini-batch order from a JSON file (matched-batch
    /// parity experiment; see docs/superpowers/specs/
    /// 2026-06-04-matched-batch-replay-design.md). When set, overrides
    /// the default RandomSampler.
    ///
    /// JSON schema: array of {"epoch": int, "mb": int, "staids": [str, ...]}.
    #[arg(long, value_name = "PATH")]
    pub batch_order_from: Option<PathBuf>,
```

(If the struct uses serde + a config file instead of clap, add it to that
struct's shape too.)

- [ ] **Step 3: Add JSON parsing + BatchSource construction in `cli::run::run()`**

Find where `train(...)` is invoked from the CLI run path. Before the
call, add:

```rust
let batch_source: Option<ddrs::data::sampler::BatchSource> =
    input.batch_order_from.as_ref().map(|path| {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("could not read --batch-order-from {path:?}: {e}"));
        #[derive(serde::Deserialize)]
        struct Record { epoch: u32, mb: u32, staids: Vec<String> }
        let records: Vec<Record> = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("malformed --batch-order-from JSON: {e}"));
        // Convert to Vec<Vec<Staid>> in (epoch, mb) order.
        let mut sorted = records;
        sorted.sort_by_key(|r| (r.epoch, r.mb));
        let batches: Vec<Vec<ddrs::data::ids::Staid>> = sorted
            .into_iter()
            .map(|r| r.staids.into_iter()
                .map(ddrs::data::ids::Staid).collect())
            .collect();
        eprintln!(
            "replaying {} mini-batches from {}",
            batches.len(),
            path.display()
        );
        let all_staids = dataset.staids().to_vec();
        ddrs::data::sampler::BatchSource::Replay(
            ddrs::data::sampler::ReplaySampler::new(batches, &all_staids),
        )
    });
```

Then pass `batch_source` as the new last argument to `train(...)`.

The `dataset` variable should already exist in the calling scope (it's
needed for `train` itself). If not, locate where it's constructed and
move the replay-construction below that point.

- [ ] **Step 4: Build + verify the CLI accepts the flag**

```bash
cd /home/tbindas/projects/ddrs
cargo build --release --bin ddrs 2>&1 | tail -5
./target/release/ddrs run --help 2>&1 | grep -A1 batch-order
```

Expected: `--batch-order-from <PATH>` appears in the help with its
description.

- [ ] **Step 5: Smoke-test with a tiny synthetic JSON**

```bash
cat > /tmp/test_batch_order.json <<'JSON'
[
  {"epoch": 1, "mb": 0, "staids": ["fake_staid_that_will_panic"]}
]
JSON

cd /home/tbindas/projects/ddrs && ./target/release/ddrs --config config/merit_training.yaml \
    run --workflow train-and-test \
    --batch-order-from /tmp/test_batch_order.json 2>&1 | tail -10
```

Expected: panic with "STAID `fake_staid_that_will_panic` from recorded
order not present in dataset's all_staids". This confirms:
- The flag is parsed.
- The JSON is loaded.
- `ReplaySampler::new` runs and rejects unknown STAIDs as designed.

If instead the run proceeds (doesn't panic), the BatchSource isn't being
plumbed through — diagnose.

- [ ] **Step 6: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add src/cli/run.rs src/cli/types.rs # whichever was modified
git commit -m "$(cat <<'EOF'
feat(cli): --batch-order-from <path> for matched-batch parity replay

Wires the BatchSource::Replay variant (commit <Task 2 SHA>) through
the ddrs CLI. When --batch-order-from is set, parses the JSON dumped
by scripts/dump_ddr_batch_order_and_train.py and builds a ReplaySampler
that yields DDR's exact gauge sequence to DDRS's training loop.

Smoke-tested: invalid STAID in the JSON correctly panics with a clear
diagnostic message, confirming the plumbing works end-to-end.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: DDRS replay training run

**Spec ref:** §5.3.

**Files:** none committed.

- [ ] **Step 1: Pre-flight workspace yaml**

The `ddrs.yaml` workspace bootstrap may be stale. Sync from the tracked
source:

```bash
cd /home/tbindas/projects/ddrs
cp config/merit_training.yaml ddrs.yaml
grep -E 'grid:|^  k:|log_space' ddrs.yaml
```

Expected: `grid: 50`, `k: 2`, `log_space_parameters: [p_spatial]`.

- [ ] **Step 2: Confirm DDR's captured order is on disk**

```bash
ls -la /tmp/ddr_batch_order.json
jq 'length' /tmp/ddr_batch_order.json
jq '.[0] | {epoch, mb, n_staids: (.staids | length)}' /tmp/ddr_batch_order.json
```

Expected: file exists, ~175 records, first record has epoch=1 mb=0
batch_size=64.

If missing, re-run Task 1 to capture.

- [ ] **Step 3: Launch DDRS retrain with replay**

```bash
cd /home/tbindas/projects/ddrs && cargo run --release --bin ddrs -- \
    run --workflow train-and-test \
    --batch-order-from /tmp/ddr_batch_order.json \
    > /tmp/ddrs_train_replay.log 2>&1 &
echo "PID: $!"
```

Monitor:

```bash
tail -f /tmp/ddrs_train_replay.log | grep --line-buffered -E 'mb=|epoch|run complete|nan|NaN|error|replaying'
```

Watch for:
- An early log line `replaying <N> mini-batches from /tmp/ddr_batch_order.json`
  confirming the flag is active.
- Loss decreasing as usual.
- Final `run complete → ./.ddrs/runs/<timestamp>-train-and-test`.

Capture the new run's timestamp + path.

- [ ] **Step 4: Dump kan_parameters.nc**

```bash
RUN=.ddrs/runs/<new-timestamp>-train-and-test
CKPT=$RUN/checkpoints/$(ls $RUN/checkpoints/ | grep '\.mpk$' | sort -V | tail -1 | sed 's/\.mpk$//')
echo "CKPT=$CKPT"
cargo run --release --bin dump_parameters -- \
    --config $RUN/config.yaml \
    --checkpoint $CKPT \
    --output $RUN/kan_parameters.nc 2>&1 | tail -2
```

Expected: `wrote 346321 reaches → ...`.

- [ ] **Step 5: Quick distribution check**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py && uv run --extra plots python <<PY
import xarray as xr, numpy as np
RUN = "/home/tbindas/projects/ddrs/.ddrs/runs/<new-timestamp>-train-and-test"
v = xr.open_dataset(f"{RUN}/kan_parameters.nc").n.values
print(f"DDR reference:         median 0.0744  mean 0.0735  p5 0.0387  p95 0.1047  frac<.035 0.031")
print(f"DDRS post-areapool:    median 0.0395  mean 0.0444  p5 0.0176  p95 0.0962  frac<.035 0.404")
print(f"DDRS replay (THIS):    median {np.median(v):.4f}  mean {np.mean(v):.4f}  "
      f"p5 {np.percentile(v,5):.4f}  p95 {np.percentile(v,95):.4f}  "
      f"frac<.035 {(v<0.035).mean():.3f}")
PY
```

**Capture the actual numbers.** They feed Task 5's verdict.

No commit (the retrain output is gitignored).

---

## Task 5: Layer-2 notebook + §5.1 verdict

**Spec ref:** §5.3 + §6 outcome thresholds.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-04-matched-batch-replay-design.md`
  (append §5.1)

- [ ] **Step 1: Regenerate DDR reference NetCDF against new run's COMID order**

The Task 1 DDR training produced a fresh checkpoint with the captured
batch order. Use THAT checkpoint as the DDR reference (it's the apples-to-
apples comparison — both ports trained on the same batch order from the
same init).

```bash
DDR_CKPT=$(ls -t ~/projects/ddr/output/ddr-vXXX-merit-training/<timestamp from Task 1>/saved_models/*.pt | head -1)
echo "DDR_CKPT=$DDR_CKPT"
cd ~/projects/ddr && uv run python /home/tbindas/projects/ddrs/scripts/dump_ddr_trained_params.py \
    --checkpoint "$DDR_CKPT" \
    --conus-comids /home/tbindas/projects/ddrs/.ddrs/runs/<Task 4 run-id>/kan_parameters.nc \
    --out /tmp/kan_params_trained_ddr_matched.nc 2>&1 | tail -3
```

Expected: `wrote 346321 CONUS reaches → ...`.

- [ ] **Step 2: Materialize and execute parity_trained.ipynb in the replay run dir**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
export RUN_DIR=/home/tbindas/projects/ddrs/.ddrs/runs/<Task 4 run-id>
mkdir -p "$RUN_DIR/plots"

uv run --extra plots python <<'PY'
import nbformat as nbf
import os, re
skill_md = open("/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/references/parity_trained.md").read()
section = skill_md.split("## Notebook cells", 1)[1].split("## Pass criterion", 1)[0]
code_blocks = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
assert len(code_blocks) == 5
nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell(
    "# DDR ↔ DDRS trained-`n` parity (matched-batch replay)"
))
# Override the DDR NC path to the matched-batch one.
code_blocks[0] = code_blocks[0].replace(
    '/tmp/kan_params_trained_ddr.nc',
    '/tmp/kan_params_trained_ddr_matched.nc'
)
for src in code_blocks:
    nb.cells.append(nbf.v4.new_code_cell(src.strip()))
out = os.path.join(os.environ["RUN_DIR"], "plots", "parity_trained_matched.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
PY

uv run --extra plots jupyter nbconvert --to notebook --execute \
    "$RUN_DIR/plots/parity_trained_matched.ipynb" \
    --output parity_trained_matched.ipynb \
    --output-dir "$RUN_DIR/plots" 2>&1 | tail -5
```

- [ ] **Step 3: Extract Cell 2 + Cell 5 outputs**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
export RUN_DIR=/home/tbindas/projects/ddrs/.ddrs/runs/<Task 4 run-id>
uv run --extra plots python <<'PY'
import json, os
nb = json.load(open(f"{os.environ['RUN_DIR']}/plots/parity_trained_matched.ipynb"))
for label, idx in [("Cell 2", 2), ("Cell 5", 5)]:
    print(f"\n=== {label} ===")
    for out in nb["cells"][idx].get("outputs", []):
        t = out.get("text") or (out.get("data", {}) or {}).get("text/plain")
        if t:
            print("".join(t) if isinstance(t, list) else t)
PY
```

Capture both verbatim.

- [ ] **Step 4: Apply the §6 verdict thresholds**

Per spec §6:
- KS(n) ≤ 0.10 AND Spearman(n) ≥ 0.85 → **investigation closed.**
- KS(n) ≤ 0.20 AND Spearman(n) ∈ [0.70, 0.85] → "improvement but not
  closed; needs deeper analysis."
- Spearman(n) < 0.70 → "matched batches didn't help; bug elsewhere."

Map the actual Cell 5 verdict to one of these three rows.

- [ ] **Step 5: Append §5.1 to the spec**

Open `docs/superpowers/specs/2026-06-04-matched-batch-replay-design.md`.
Append at EOF:

```markdown
---

## §5.1 Empirical verdict (Task 5 of the plan)

**DDR-side capture (Task 1):**
- Wrapper script: `scripts/dump_ddr_batch_order_and_train.py`
- Captured trace: `/tmp/ddr_batch_order.json` (~175 mini-batches × 64
  gauges over 5 epochs)
- DDR checkpoint: `<path from Task 1 Step 4>`
- DDR's median n at this trained checkpoint: <captured value, expected ~0.074>

**DDRS replay run (Task 4):**
- Run id: `<Task 4 run-id>`
- DDRS's median n after replay: <captured from Task 4 Step 5>

**Cell 2 (per-parameter stats):**

```
<paste Cell 2 verbatim from Task 5 Step 3>
```

**Cell 5 verdict:**

```
<paste Cell 5 verbatim from Task 5 Step 3>
```

**Outcome (from §6 thresholds):**

<one of the three rows; quote it verbatim>

**Comparison vs PR #13 area-pool fix verdict:**

| metric | DDRS pre-fix | DDRS post-areapool | DDRS replay (THIS) | DDR ref |
|---|---:|---:|---:|---:|
| median n | 0.0296 | 0.0395 | <value> | 0.0744 |
| KS(n) | 0.6916 | 0.5685 | <value> | — |
| Spearman(n) | +0.1790 | +0.3471 | <value> | 1.0 |

**Next step:**

<inherited from §6's matched row, OR "n-saturation investigation closed">
```

- [ ] **Step 6: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add docs/superpowers/specs/2026-06-04-matched-batch-replay-design.md
git commit -m "$(cat <<'EOF'
docs/spec: record matched-batch replay verdict

After running DDRS with DDR's captured per-(epoch, mb) batch order
(commit <Task 1 SHA> for capture; commit <Task 3 SHA> for DDRS replay
plumbing; commit <Task 4 SHA> for the BatchSource enum), the
per-reach n distribution comparison produced <Cell 5 verdict from
Task 5 Step 3>.

This <closes the n-saturation investigation / localizes the residual
divergence to ...>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Spec coverage map

| Spec section | Plan task(s) |
|--------------|-------------|
| §2 C1 (DDR-side intrusion) | Task 1 — monkey-patch wrapper avoids editing DDR's source. |
| §2 C2 (DDRS plumbing) | Tasks 2 + 3 |
| §2 C3 (gauge-set agreement) | Task 3 Step 5 (smoke test catches mismatch) + Task 2 `replay_sampler_rejects_unknown_staid` |
| §2 C4 (batch size matching) | Task 1 flush logic uses `cfg.experiment.batch_size`; Task 3 reads the JSON as-is. |
| §2 C5 (tau-asymmetry residual) | Acknowledged in Task 5 — verdict thresholds (0.85) leave headroom for residual. |
| §2 C6 (GPU non-determinism) | Out of scope for this plan; Task 5 §6 thresholds account for it. |
| §2 C7 (per-epoch shuffle semantics) | Task 1 captures via `__iter__` which re-fires per epoch in PyTorch — confirmed by JSON record count (175 ≈ 5 epochs × 35). |
| §3 A1-A5 | All baked into the task implementations. |
| §4 (algorithm + verification) | Tasks 1-5 |
| §5.1 (write the implementation) | Tasks 1-3 |
| §5.2 (DDRS-side replay) | Tasks 2 + 3 |
| §5.3 (run + verify) | Tasks 4 + 5 |
| §6 (outcome thresholds) | Task 5 |
| §7 (effort estimate ~5 hours) | Plan sized accordingly: 1 + 2 + 1 + 0.5 + 1 = ~5.5 hours active work. |

---

Plan complete and saved to
`docs/superpowers/plans/2026-06-04-matched-batch-replay.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task with spec + code-quality review between each. Same flow as PRs #11/#12/#13.

**2. Inline Execution** — `superpowers:executing-plans` with batch checkpoints.

Which?
