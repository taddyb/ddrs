#!/usr/bin/env python3
"""Plant plan for the synthetic recoverability positive control.

Per Ref probe reach: target = min(2*band5, 0.5*ceiling) with
ceiling = area_z_mean * K_D_MAX * (depth_mean + 2). Overrides are expressed in
NORMALIZED space: K_D at its log-range ceiling -> 1.0, d_gw at floor -> 0.0,
factor = target/ceiling (physical == normalized for [0,1]).
Drop rule (spec section 2): ceiling < 0.25*band5 -> reach excluded, logged.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/home/tbindas/projects/ddrs")
PROBE = ROOT / "output/zeta_probe"
OUT = ROOT / "output/recoverability"
ZETA_NC = ROOT / ".ddrs/runs/2026-07-01T13-43-32Z-train-and-test/kan_parameters.nc"
K_D_MAX = 1e-5
D_GW_MIN = -2.0

plan = pd.read_csv(PROBE / "probe_plan.csv", dtype={"staid_nearest": str})
ref = plan[(plan["class"] == "Ref") & (plan["delta"] == 0.01)]
ref = ref.drop_duplicates("comid")[["comid", "staid_nearest"]]

rows = pd.read_csv(PROBE / "detectability_rows.csv")
band = rows[(rows["cls"] == "Ref") & (rows["delta"] == 0.01)][["comid", "band5"]]
band = band.drop_duplicates("comid")
sites = ref.merge(band, on="comid", how="inner")
print(f"Ref probe reaches: {len(ref)}; with band5: {len(sites)}")

ds = xr.open_dataset(ZETA_NC)
diag = pd.DataFrame({
    "comid": ds["COMID_eval"].values,
    "depth_mean": ds["depth_mean"].values,
    "area_z_mean": ds["area_z_mean"].values,
})
sites = sites.merge(diag, on="comid", how="left")
missing = sites["depth_mean"].isna()
assert not missing.any(), f"reaches missing diagnostics: {sites[missing]['comid'].tolist()}"

sites["ceiling_flux"] = sites["area_z_mean"] * K_D_MAX * (sites["depth_mean"] - D_GW_MIN)
sites["target_flux"] = np.minimum(2.0 * sites["band5"], 0.5 * sites["ceiling_flux"])
dropped = sites[sites["ceiling_flux"] < 0.25 * sites["band5"]].copy()
kept = sites[sites["ceiling_flux"] >= 0.25 * sites["band5"]].copy()

kept["k_d_norm"] = 1.0
kept["d_gw_norm"] = 0.0
kept["factor_norm"] = (kept["target_flux"] / kept["ceiling_flux"]).clip(0.0, 1.0)
assert (kept["factor_norm"] <= 0.5 + 1e-9).all(), "factor should be <=0.5 by the target rule"
assert (kept["target_flux"] > 0).all()

OUT.mkdir(parents=True, exist_ok=True)
cols = ["comid", "k_d_norm", "d_gw_norm", "factor_norm",
        "staid_nearest", "band5", "target_flux", "ceiling_flux"]
kept[cols].to_csv(OUT / "plants.csv", index=False)

def q(s):
    return " ".join(f"p{p}={np.percentile(s, p):.3e}" for p in (10, 50, 90))

report = [
    f"plant sites kept: {len(kept)}  dropped (ceiling < 0.25*band5): {len(dropped)}",
    f"band5      [m3/s]: {q(kept['band5'])}",
    f"ceiling    [m3/s]: {q(kept['ceiling_flux'])}",
    f"target     [m3/s]: {q(kept['target_flux'])}",
    f"factor_norm      : {q(kept['factor_norm'])}",
    f"targets band-limited (2*band5 < 0.5*ceiling): "
    f"{int((2 * kept['band5'] < 0.5 * kept['ceiling_flux']).sum())}/{len(kept)}",
    "dropped comids: " + (", ".join(map(str, dropped["comid"])) or "none"),
]
(OUT / "sites_report.txt").write_text("\n".join(report) + "\n")
print("\n".join(report))
print(f"\nwrote {OUT / 'plants.csv'}")
