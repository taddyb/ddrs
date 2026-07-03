#!/usr/bin/env python3
"""Stage-2 site selection: GAGES-II Ref basins, strata, round packing.

Run: cd <ddrs>/ddrs-py && uv run python ../scripts/zeta_probe_sites.py
(or .venv/bin/python on hosts where uv can't rebuild the maturin package)

Zarr layout (verified 2026-07-02):
  merit_gages_conus_adjacency.zarr/
    <STAID_zero_padded>/      # one group per gauge (8945 total)
      order      [int64]      # COMIDs in this gauge's subgraph  <-- we read this
      indices_0  [...]        # adjacency row indices
      indices_1  [...]        # adjacency col indices
      values     [...]        # adjacency values
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import zarr

DDRS = Path("/home/tbindas/projects/ddrs")
GAGES2_DBF = Path("/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.shp")
GAGES_CSV = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
ATTRS_NC = Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
GAGES_ADJ = Path("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr")
GRAD_NC = DDRS / "output/zeta_probe/grad_trained.nc"
DELTAS = [0.01, 0.1]
N_PROBES = 250  # per delta (CPU budget); each extra ROUND costs a full eval


def tercile_labels(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(x, 33), np.nanpercentile(x, 67)
    return np.where(x < lo, 0, np.where(x < hi, 1, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DDRS / "output/zeta_probe/probe_plan.csv")
    ap.add_argument("--gages-adj", type=Path, default=GAGES_ADJ)
    args = ap.parse_args()

    # 1. Gauge classification: STAID → CLASS (Ref / Non-ref).
    g2 = gpd.read_file(GAGES2_DBF)[["STAID", "CLASS"]]
    g2["STAID"] = g2["STAID"].astype(str).str.lstrip("0")
    gages = pd.read_csv(GAGES_CSV, dtype={"STAID": str})
    gages["STAID_KEY"] = gages["STAID"].str.lstrip("0")
    gages = gages.merge(g2, left_on="STAID_KEY", right_on="STAID", how="left",
                        suffixes=("", "_g2"))
    gages["CLASS"] = gages["CLASS"].fillna("Unknown")
    print(gages["CLASS"].value_counts())

    # 2. Reach → containing gauges, from the per-gauge subgraph zarr.
    #    Confirmed layout (2026-07-02): one group per zero-padded gauge STAID,
    #    each subgroup has arrays [indices_0, order, indices_1, values];
    #    'order' holds the COMIDs for that gauge's upstream subgraph.
    root = zarr.open_group(str(args.gages_adj), mode="r")
    print("gages_adjacency layout:", list(root.group_keys())[:5], list(root.array_keys())[:5])
    reach_gauges: dict[int, list[str]] = defaultdict(list)
    gauge_comids: dict[str, np.ndarray] = {}
    for gname in root.group_keys():  # one subgroup per gauge STAID
        sub = root[gname]
        arr_name = "order" if "order" in sub else list(sub.array_keys())[0]
        comids = np.asarray(sub[arr_name][:], dtype=np.int64)
        gauge_comids[gname] = comids
        for c in comids:
            reach_gauges[int(c)].append(gname)
    assert gauge_comids, (
        "no per-gauge groups found — inspect the printed layout and adapt "
        "(the engine's GagesAdjacencyStore builds one subgraph per gauge)"
    )

    # 3. Candidate reaches: probe-covered (stage-1) ∩ subgraphs of Ref gauges.
    grad = xr.open_dataset(GRAD_NC)
    probe_comids = grad["COMID_probe"].values.astype(np.int64)
    reach_abs = dict(zip(probe_comids, grad["grad_factor_abs"].values))

    staid_class = dict(zip(gages["STAID_KEY"], gages["CLASS"]))
    staid_drain = dict(zip(gages["STAID_KEY"], gages["DRAIN_SQKM"]))

    def norm(g: str) -> str:
        return g.lstrip("0")

    rows = []
    for c in probe_comids:
        containing = reach_gauges.get(int(c), [])
        if not containing:
            continue
        classes = {staid_class.get(norm(g), "Unknown") for g in containing}
        # Ref-only population: EVERY containing gauge must be Ref (a Non-ref
        # gauge downstream would receive the perturbation through regulation).
        cls = "Ref" if classes == {"Ref"} else ("Non-ref" if "Non-ref" in classes else "Mixed")
        nearest = min(containing, key=lambda g: staid_drain.get(norm(g), np.inf))
        rows.append((int(c), cls, nearest, len(containing)))
    cand = pd.DataFrame(rows, columns=["comid", "class", "staid_nearest", "n_gauges"])
    print("candidates:", cand["class"].value_counts().to_dict())

    # 4. Strata: uparea tercile × aridity tercile × stage-1 reachability tercile.
    attrs = xr.open_dataset(ATTRS_NC)
    acom = attrs["COMID"].values.astype(np.int64)
    order = np.argsort(acom)
    pos = order[np.clip(np.searchsorted(acom, cand["comid"].values, sorter=order), 0, len(acom) - 1)]
    ok = acom[pos] == cand["comid"].values
    for name in ("log10_uparea", "aridity"):
        v = attrs[name].values.astype(float)[pos]
        v[~ok] = np.nan
        cand[name] = v
    cand["reach_abs"] = cand["comid"].map(reach_abs)
    cand["s_up"] = tercile_labels(cand["log10_uparea"].values)
    cand["s_ar"] = tercile_labels(cand["aridity"].values)
    cand["s_re"] = tercile_labels(np.log10(np.maximum(cand["reach_abs"].values, 1e-30)))

    # 5. Sample: Ref primary (equal per stratum), Non-ref contrast at 20%.
    picked = []
    ref = cand[cand["class"] == "Ref"]
    per_stratum = max(1, N_PROBES // 27)
    for (a, b, c), grp in ref.groupby(["s_up", "s_ar", "s_re"]):
        take = grp.sample(min(per_stratum, len(grp)), random_state=42)
        picked.append(take)
    nonref = cand[cand["class"] == "Non-ref"].sample(
        min(N_PROBES // 5, (cand["class"] == "Non-ref").sum()), random_state=42)
    plan = pd.concat(picked + [nonref]).drop_duplicates("comid")
    print(f"picked {len(plan)} probe reaches ({(plan['class']=='Ref').sum()} Ref)")

    # 6. Round packing: no two probes in a round may share ANY containing gauge.
    plan_rows = []
    rounds: list[set[str]] = []
    for delta in DELTAS:
        for _, r in plan.iterrows():
            gset = set(reach_gauges[int(r["comid"])])
            for k, used in enumerate(rounds):
                if not (used & gset):
                    used |= gset
                    break
            else:
                k = len(rounds)
                rounds.append(set(gset))
            plan_rows.append((k, int(r["comid"]), delta, r["staid_nearest"], r["class"],
                              int(r["s_up"]), int(r["s_ar"]), int(r["s_re"])))
    out = pd.DataFrame(plan_rows, columns=["round", "comid", "delta", "staid_nearest",
                                           "class", "stratum_uparea", "stratum_aridity",
                                           "stratum_reach"])
    n_rounds = out["round"].nunique()
    print(f"{len(out)} probes packed into {n_rounds} rounds")
    if n_rounds > 40:
        print(f"ERROR: round count {n_rounds} > 40 — aborting without writing CSV")
        raise SystemExit(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
