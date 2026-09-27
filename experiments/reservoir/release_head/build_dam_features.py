#!/usr/bin/env python
"""Dam feature table for the learned dam release (params.reservoir_release: learned).

Input: ../nid/nid_dams_in_eval_network.csv (NID snapped to MERIT, dams inside the eval networks).
Kept: storage_mcm >= 10 (normal storage, NID storage where normal is missing; the same definition the NID
intersection used) and snap class A, B, C or D.

One row per MERIT COMID. Several dams on one reach are aggregated:
  summed    storage_mcm, storage_max_mcm, surface_km2, max_discharge_m3s (NaN only when every dam is NaN)
  max       height_m
  largest   da_km2, reach_uparea_km2, year, primary_purpose  (taken from the dam with the largest storage_mcm)

Features (the release head's inputs; names are what `release_head.input_var_names` lists):
  log10_storage           log10 storage_mcm
  log10_storage_max       log10 storage_max_mcm
  log10_surface           log10 surface_km2
  log10_drainage          log10 da_km2, falling back to the reach's upstream area
  log10_storage_per_area  log10(storage_mcm / reach_uparea_km2)   (MCM per km2 = metres of runoff depth)
  log10_max_discharge     log10 max_discharge_m3s
  height                  height_m
  year                    year completed
  purpose_*               one-hot primary purpose: flood, hydro, supply, irrigation, recreation, navigation, other
Every continuous column with a missing value is filled with its median and gets a <name>_missing 0/1 flag. Continuous
columns are then z-scored; flags and one-hots stay 0/1. Stats (median fill value, mean, std) go to
dam_features_stats.json.

No observed release, schedule or cap enters: these are static structural attributes of the dam.

Outputs (next to this script): dam_features.csv, dam_features_stats.json. Also
examples/juniata/data/juniata_dam_features.csv, the rows inside the committed Juniata network.

Run: ~/projects/ddr/.venv/bin/python experiments/reservoir/release_head/build_dam_features.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NID = HERE.parent / "nid" / "nid_dams_in_eval_network.csv"
JUNIATA_ADJ = ROOT / "examples" / "juniata" / "data" / "juniata_conus_adjacency.zarr"

MIN_STORAGE_MCM = 10.0
SNAP_CLASSES = {"A", "B", "C", "D"}
PURPOSES = {
    "purpose_flood": ["Flood Risk Reduction"],
    "purpose_hydro": ["Hydroelectric"],
    "purpose_supply": ["Water Supply"],
    "purpose_irrigation": ["Irrigation"],
    "purpose_recreation": ["Recreation"],
    "purpose_navigation": ["Navigation"],
}
CONTINUOUS = [
    "log10_storage",
    "log10_storage_max",
    "log10_surface",
    "log10_drainage",
    "log10_storage_per_area",
    "log10_max_discharge",
    "height",
    "year",
]


def nansum_or_nan(x: pd.Series) -> float:
    return float(x.sum()) if x.notna().any() else float("nan")


def aggregate(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comid, g in d.groupby("COMID", sort=True):
        big = g.sort_values("storage_mcm", ascending=False).iloc[0]
        rows.append(
            dict(
                COMID=int(comid),
                n_dams=len(g),
                largest_nid_id=big.nid_id,
                largest_name=big["name"],
                storage_mcm=float(g.storage_mcm.sum()),
                storage_max_mcm=nansum_or_nan(g.storage_max_mcm),
                surface_km2=nansum_or_nan(g.surface_km2),
                max_discharge_m3s=nansum_or_nan(g.max_discharge_m3s),
                height_m=float(g.height_m.max()),
                da_km2=big.da_km2,
                reach_uparea_km2=big.reach_uparea_km2,
                year=big.year,
                primary_purpose=big.primary_purpose,
            )
        )
    return pd.DataFrame(rows)


def safe_log10(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    return np.log10(x.where(x > 0))


def main() -> None:
    nid = pd.read_csv(NID)
    keep = nid[(nid.storage_mcm >= MIN_STORAGE_MCM) & nid.snap_class.isin(SNAP_CLASSES)].copy()
    agg = aggregate(keep)

    f = pd.DataFrame({"COMID": agg.COMID})
    f["log10_storage"] = safe_log10(agg.storage_mcm)
    f["log10_storage_max"] = safe_log10(agg.storage_max_mcm)
    f["log10_surface"] = safe_log10(agg.surface_km2)
    f["log10_drainage"] = safe_log10(agg.da_km2.where(agg.da_km2 > 0, agg.reach_uparea_km2))
    f["log10_storage_per_area"] = safe_log10(agg.storage_mcm / agg.reach_uparea_km2)
    f["log10_max_discharge"] = safe_log10(agg.max_discharge_m3s)
    f["height"] = agg.height_m.astype(float)
    f["year"] = agg.year.astype(float)

    stats: dict[str, dict] = {}
    for c in CONTINUOUS:
        med = float(f[c].median())
        missing = f[c].isna()
        if missing.any():
            f[f"{c}_missing"] = missing.astype(float)
        f[c] = f[c].fillna(med)
        mu, sd = float(f[c].mean()), float(f[c].std(ddof=0))
        sd = sd if sd > 0 else 1.0
        f[c] = (f[c] - mu) / sd
        stats[c] = dict(median_fill=med, mean=mu, std=sd, n_missing=int(missing.sum()))

    purpose = agg.primary_purpose.fillna("")
    named = np.zeros(len(agg), dtype=bool)
    for col, labels in PURPOSES.items():
        hit = purpose.isin(labels).to_numpy()
        f[col] = hit.astype(float)
        named |= hit
    f["purpose_other"] = (~named).astype(float)

    feature_cols = [c for c in f.columns if c != "COMID"]
    assert not f[feature_cols].isna().any().any(), "a feature column still has NaN"
    out = f.merge(agg[["COMID", "n_dams", "largest_nid_id", "largest_name", "storage_mcm"]], on="COMID")
    out.to_csv(HERE / "dam_features.csv", index=False, float_format="%.6g")

    meta = dict(
        source=str(NID.relative_to(ROOT)),
        filter=dict(min_storage_mcm=MIN_STORAGE_MCM, snap_classes=sorted(SNAP_CLASSES)),
        n_dams_kept=int(len(keep)),
        n_comids=int(len(out)),
        n_comids_with_several_dams=int((agg.n_dams > 1).sum()),
        feature_columns=feature_cols,
        continuous=stats,
        purposes={k: v for k, v in PURPOSES.items()},
    )
    (HERE / "dam_features_stats.json").write_text(json.dumps(meta, indent=1))

    order = set(int(c) for c in zarr.open(str(JUNIATA_ADJ), mode="r")["order"][:])
    jun = out[out.COMID.isin(order)]
    jun.to_csv(ROOT / "examples" / "juniata" / "data" / "juniata_dam_features.csv", index=False, float_format="%.6g")
    print(json.dumps(dict(n_dams_kept=meta["n_dams_kept"], n_comids=meta["n_comids"],
                          several=meta["n_comids_with_several_dams"], features=len(feature_cols),
                          juniata=jun[["COMID", "largest_name", "storage_mcm"]].to_dict("records")), indent=1))


if __name__ == "__main__":
    main()
