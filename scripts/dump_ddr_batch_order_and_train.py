"""Wrapper around DDR's train_and_test.py that captures per-mini-batch
gauge order to JSON.

Approach: monkey-patch `torch.utils.data.RandomSampler` with a subclass
that records `__iter__` output to a module-global list BEFORE importing
DDR's training code.  DDR's training script instantiates the sampler
normally; iteration logs as a side-effect.

Output schema
-------------
[
    {"epoch": 1, "mb": 0, "staids": ["12345678", ...]},
    ...
]

Run under DDR's uv venv
-----------------------
    cd ~/projects/ddr && uv run python \\
        ~/projects/ddrs/scripts/dump_ddr_batch_order_and_train.py \\
        --config-name=merit_training_config \\
        +dump_batch_order_to=/tmp/ddr_batch_order.json

Notes
-----
- The monkey-patch replaces `torch.utils.data.RandomSampler` BEFORE the
  DDR training modules are imported so that `from torch.utils.data import
  RandomSampler` inside `scripts/train.py` resolves to the logging
  subclass.  We also walk `sys.modules` after force-importing DDR's
  training modules to patch any already-resolved aliases.
- DDR's dataset exposes `dataset.gage_ids` (np.ndarray of str) — NOT
  `dataset.staids`.  We use `data_source.gage_ids` in `__iter__`.
- We call `_ddr_main.__wrapped__(cfg)` from inside our own `@hydra.main`
  so that `HydraConfig.get()` is populated (needed for save_path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ── Path setup (must precede all DDR imports) ──────────────────────────────
# Add DDR's root (so `scripts.train_and_test` and `scripts.train` are
# importable as packages) and its src/ (for the `ddr` package).
sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
sys.path.insert(0, str(Path.home() / "projects" / "ddr"))

import torch.utils.data as _torch_data

# ── Save the original before patching ─────────────────────────────────────
_ORIGINAL_RANDOM_SAMPLER = _torch_data.RandomSampler

# Module-level log.  Each entry is {"epoch": int, "indices": [int, ...],
# "gage_ids": [str, ...]}.  Post-processed into (epoch, mb, staids) after
# training completes.
_BATCH_ORDER_LOG: list[dict[str, Any]] = []
_EPOCH_COUNTER = [0]  # mutable cell; bumped by LoggingRandomSampler.__iter__


class LoggingRandomSampler(_ORIGINAL_RANDOM_SAMPLER):
    """Drop-in replacement that records the full index permutation each epoch.

    PyTorch's DataLoader calls `sampler.__iter__()` once per epoch.  We
    materialise the iterator into a list, record it together with the
    dataset's gage_ids, then re-yield from the list.
    """

    def __iter__(self):
        _EPOCH_COUNTER[0] += 1
        indices = list(super().__iter__())
        dataset = self.data_source
        # DDR's MeritGagesDataset (and its base) exposes gage_ids: np.ndarray.
        gage_ids = [str(g) for g in dataset.gage_ids]
        _BATCH_ORDER_LOG.append(
            {
                "epoch": _EPOCH_COUNTER[0],
                "indices": indices,
                "gage_ids": gage_ids,
            }
        )
        return iter(indices)


def _patch() -> None:
    """Replace RandomSampler in torch.utils.data AND in any already-imported
    module aliases (e.g. `train.RandomSampler` from `from torch.utils.data
    import RandomSampler`)."""
    _torch_data.RandomSampler = LoggingRandomSampler

    # Walk already-loaded modules and fix aliases.  We import DDR's training
    # modules first so their aliases are present in sys.modules.
    try:
        import scripts.train as _scripts_train_mod  # noqa: F401
    except ImportError:
        pass

    for mod_name in list(sys.modules.keys()):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if getattr(mod, "RandomSampler", None) is _ORIGINAL_RANDOM_SAMPLER:
            setattr(mod, "RandomSampler", LoggingRandomSampler)


def _flush(path: str, batch_size: int, drop_last: bool = True) -> None:
    """Convert the raw per-epoch index log into per-(epoch, mb, staids)
    records and write to JSON.

    Replicates PyTorch DataLoader's batch construction so the (epoch, mb)
    numbering in the JSON matches what DDR's training loop sees.
    """
    out: list[dict[str, Any]] = []
    for entry in _BATCH_ORDER_LOG:
        indices = entry["indices"]
        gage_ids = entry["gage_ids"]
        epoch = entry["epoch"]
        n = len(indices)
        mb = 0
        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n:
                if drop_last:
                    break
                end = n
            batch_indices = indices[start:end]
            out.append(
                {
                    "epoch": epoch,
                    "mb": mb,
                    "staids": [gage_ids[i] for i in batch_indices],
                }
            )
            mb += 1
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} mini-batch records → {path}", flush=True)


# ── Apply patch at module load time (before any DDR imports) ──────────────
_patch()

# ── Now import DDR's main (aliases are resolved to LoggingRandomSampler) ──
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from scripts.train_and_test import main as _ddr_main  # noqa: E402


@hydra.main(
    version_base="1.3",
    config_path=str(Path.home() / "projects" / "ddr" / "config"),
    config_name="merit_training_config",
)
def main(cfg: DictConfig) -> None:
    """Wrapper entry point.

    Sets up the same hydra context that DDR's main() expects, then calls
    _ddr_main.__wrapped__(cfg) so HydraConfig.get() is available for
    save_path resolution, without double-initialising hydra.
    """
    # Extract our custom key before DDR's validate_config sees it.
    # Hydra sets struct=True on the cfg, so use open_dict() to remove the key.
    dump_path: str = cfg.get("dump_batch_order_to", "/tmp/ddr_batch_order.json")
    if "dump_batch_order_to" in cfg:
        with open_dict(cfg):
            del cfg["dump_batch_order_to"]

    # Call DDR's training + test logic inside the active hydra context.
    _ddr_main.__wrapped__(cfg)

    # Post-process and write the captured batch order.
    _flush(
        path=dump_path,
        batch_size=cfg.experiment.batch_size,
        drop_last=True,
    )


if __name__ == "__main__":
    main()
