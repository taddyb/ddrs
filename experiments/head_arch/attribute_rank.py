#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "netCDF4"]
# ///
"""How many independent directions do the KAN head's input attributes carry?

This bounds every head topology from above. The head is a deterministic
function of these 10 columns, so if the attribute cloud is effectively
one-dimensional then *no* architecture can emit two independent parameter
fields, and the §31 collapse is a property of the inputs rather than of the
`Linear(H, P)` read-out.

Two different quantities are reported, and they answer different questions:

  linear spectrum   PCA of the z-scored attributes. An effective rank near 1
                    would mean the attributes are one variable in disguise.
                    This is necessary but not sufficient: a full-rank input
                    cloud can still be squeezed to rank 1 by the network.

  usable directions the same spectrum weighted by how much each attribute
                    actually varies across the reaches a gauge can see. Listed
                    per attribute as its share of total variance, so the
                    ranking is readable rather than just a number.

Usage:
    experiments/head_arch/attribute_rank.py <run-config.yaml>
"""

import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def parse_yaml_list(text: str, key: str) -> list[str]:
    """Minimal extractor for a `key:` block of `- item` lines. Avoids adding a
    YAML dependency for one field; the run config snapshots are machine
    written, so the shape is stable."""
    out, collecting, indent = [], False, None
    for line in text.splitlines():
        if not collecting:
            if line.strip().startswith(f"{key}:"):
                collecting = True
                indent = len(line) - len(line.lstrip())
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cur = len(line) - len(line.lstrip())
        if stripped.startswith("- "):
            out.append(stripped[2:].strip())
        elif cur <= indent:
            break
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cfg_text = Path(sys.argv[1]).read_text()
    names = parse_yaml_list(cfg_text, "input_var_names")
    if not names:
        print("could not read input_var_names from the config")
        return 1

    attr_path = None
    for line in cfg_text.splitlines():
        if "merit_global_attributes" in line and line.strip().startswith("-"):
            attr_path = line.split("-", 1)[1].strip()
            break
    if attr_path is None:
        attr_path = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"
    stats_path = Path(
        "/home/tbindas/projects/ddr/data/statistics/"
        "merit_attribute_statistics_merit_global_attributes_v2.nc.json"
    )
    stats = json.loads(stats_path.read_text())

    ds = Dataset(attr_path)
    cols = []
    for n in names:
        v = np.asarray(ds[n][:], dtype=np.float64)
        s = stats[n]
        # DDR's stats JSON stores [p10, p90, mean, std] per attribute.
        mean, std = (s[2], s[3]) if isinstance(s, list) else (s["mean"], s["std"])
        v = np.where(np.isfinite(v), v, mean)
        cols.append((v - mean) / std)
    x = np.column_stack(cols)
    print(f"attributes: {x.shape[0]:,} reaches x {x.shape[1]} features")
    print(f"source: {attr_path}")
    print()

    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    var = s**2
    frac = var / var.sum()
    cum = np.cumsum(frac)
    pr = var.sum() ** 2 / (var**2).sum()

    print("PCA spectrum of the z-scored attributes")
    print(f"{'PC':>3} {'var frac':>9} {'cumulative':>11}")
    for i, (f, c) in enumerate(zip(frac, cum), 1):
        print(f"{i:>3} {f:>9.4f} {c:>11.4f}")
    print()
    print(f"effective rank (participation ratio) = {pr:.2f} of {x.shape[1]}")
    print(f"directions holding 90% of variance   = {int(np.searchsorted(cum, 0.90) + 1)}")
    print(f"directions holding 99% of variance   = {int(np.searchsorted(cum, 0.99) + 1)}")
    print()

    print("attribute loadings on PC1 (what the dominant direction actually is)")
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    order = np.argsort(-np.abs(vt[0]))
    for i in order:
        print(f"  {names[i]:<22} {vt[0][i]:+.3f}")
    print()
    print("correlation matrix, largest off-diagonal pairs")
    c = np.corrcoef(x, rowvar=False)
    pairs = [
        (abs(c[i, j]), c[i, j], names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    for a, r, ni, nj in sorted(pairs, reverse=True)[:8]:
        print(f"  {ni:<22} {nj:<22} {r:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
