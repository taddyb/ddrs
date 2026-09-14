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

Round packing: conflicts are nearest-gauge containment — probes A and B conflict
iff A.comid lies inside B's staid_nearest subgraph or B.comid lies inside A's,
because the downstream analysis measures each probe only at its staid_nearest gauge.
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

    # Cap reaches per measurement gauge: probes sharing a nearest gauge can never
    # share a round (each lies in the other's measurement subgraph), so per-gauge
    # count × n_deltas is a hard floor on rounds. Keep the 2 most stratum-diverse
    # reaches per gauge (greedy: first-seen distinct (s_up, s_ar, s_re) combos).
    MAX_PER_GAUGE = 2
    capped = []
    for staid, grp in plan.groupby("staid_nearest", sort=False):
        if len(grp) <= MAX_PER_GAUGE:
            capped.append(grp)
            continue
        seen_strata: set[tuple] = set()
        keep_rows = []
        for _, row in grp.sort_values(["s_up", "s_ar", "s_re"]).iterrows():
            key = (row["s_up"], row["s_ar"], row["s_re"])
            if key not in seen_strata or len(keep_rows) < 1:
                seen_strata.add(key)
                keep_rows.append(row)
            if len(keep_rows) == MAX_PER_GAUGE:
                break
        capped.append(pd.DataFrame(keep_rows))
    plan = pd.concat(capped)
    print(f"after per-gauge cap ({MAX_PER_GAUGE}): {len(plan)} probe reaches "
          f"({(plan['class'] == 'Ref').sum()} Ref) on {plan['staid_nearest'].nunique()} gauges")

    # 6. Round packing (first-fit-decreasing): build the combined probe list for
    #    both deltas first, sort by gauge-footprint size DESCENDING (big footprints
    #    placed first pack tighter), then greedy first-fit.  Stable secondary key
    #    (comid, delta) provides deterministic tie-breaking.
    #
    #    Conflict relation: A and B conflict iff A.comid ∈ nearest_set[B] or
    #    B.comid ∈ nearest_set[A], where nearest_set[X] is the COMID set of X's
    #    staid_nearest gauge subgraph.  Two probes sharing only a far-downstream
    #    gauge are independent because each is measured only at its staid_nearest.
    probe_list = [
        (int(r["comid"]), delta, r["staid_nearest"], r["class"],
         int(r["s_up"]), int(r["s_ar"]), int(r["s_re"]))
        for delta in DELTAS
        for _, r in plan.iterrows()
    ]
    probe_list.sort(key=lambda x: (-len(reach_gauges[x[0]]), x[0], x[1]))

    # Precompute nearest-gauge COMID set for each probe entry.
    # gauge_comids keys are zero-padded STAIDs; staid_nearest is also zero-padded.
    nearest_set_map: list[frozenset[int]] = [
        frozenset(int(c) for c in gauge_comids.get(staid_nearest, np.array([], dtype=np.int64)))
        for _, _, staid_nearest, *_ in probe_list
    ]

    plan_rows = []
    rounds_nearest_union: list[set[int]] = []   # union of nearest_sets for round members
    rounds_member_comids: list[set[int]] = []   # comids of probes already in the round

    for idx, (comid, delta, staid_nearest, cls, s_up, s_ar, s_re) in enumerate(probe_list):
        ns = nearest_set_map[idx]
        assigned_round = None
        for k in range(len(rounds_nearest_union)):
            # Candidate conflicts with this round iff its comid falls inside any
            # member's nearest-gauge subgraph, OR any member comid falls inside
            # the candidate's nearest-gauge subgraph.
            if comid not in rounds_nearest_union[k] and rounds_member_comids[k].isdisjoint(ns):
                rounds_nearest_union[k].update(ns)
                rounds_member_comids[k].add(comid)
                assigned_round = k
                break
        if assigned_round is None:
            assigned_round = len(rounds_nearest_union)
            rounds_nearest_union.append(set(ns))
            rounds_member_comids.append({comid})
        plan_rows.append((assigned_round, comid, delta, staid_nearest, cls, s_up, s_ar, s_re))

    out = pd.DataFrame(plan_rows, columns=["round", "comid", "delta", "staid_nearest",
                                           "class", "stratum_uparea", "stratum_aridity",
                                           "stratum_reach"])
    n_rounds = out["round"].nunique()
    rpc = out.groupby("round").size()
    print(f"\n=== PROBE PLAN: {len(out)} probes packed into {n_rounds} ROUNDS ===")
    print(f"probes-per-round: min={rpc.min()}  median={int(rpc.median())}  max={rpc.max()}")

    # Verification: assert no conflicts within any round under the new relation.
    nearest_set_by_staid: dict[str, frozenset[int]] = {
        g: frozenset(int(c) for c in comids)
        for g, comids in gauge_comids.items()
    }
    violations = 0
    for _, grp in out.groupby("round"):
        comids_r = grp["comid"].tolist()
        ns_r = [nearest_set_by_staid.get(g, frozenset()) for g in grp["staid_nearest"]]
        n = len(comids_r)
        for i in range(n):
            for j in range(i + 1, n):
                if comids_r[i] in ns_r[j] or comids_r[j] in ns_r[i]:
                    violations += 1
    print(f"verification: {violations} violations")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
