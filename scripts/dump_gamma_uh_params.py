"""Dump the trained gamma-UH routing parameters per MERIT divide.

Pulls routa/routb from the CONUS2717_AORC2F ep100 checkpoint's Ann head (the
model that generated daily_dhbv2_distributed_aorc2f_merit_unit_catchments.ic)
and computes the effective gamma kernel mean a_eff * theta_eff (days) per
divide. Mirrors forward_conus_divides.py --mode export exactly: same
attr_list, same normalization (dapengscaler_stat.json), same divide order
(the export store's divide_id axis), identity/singleton groups so each
divide's params are its own.

Scaling chain (waterlossv18_1.py:57,179-183 + rnn.py UH_gamma):
    r0, r1 = sigmoid Ann outputs, cols 52:54 (nfea2*nmul2 = 13*4)
    tempa = 2.9 * r0          # "routa" in the 2026-07-29 findings
    tempb = 6.5 * r1          # "routb"
    a_eff = tempa + 0.1       # relu(a)+0.1 floor inside UH_gamma
    theta_eff = tempb + 0.5   # relu(b)+0.5 floor
    mean travel = a_eff * theta_eff   [days]

Run under the water_loss venv (has torch/hydroDL/icechunk):
    ~/projects/water_loss/.venv/bin/python scripts/dump_gamma_uh_params.py
Output: output/tau_sweep/gamma_uh_params.csv (divide_id, routa, routb,
a_eff, theta_eff, tau_uh_days, uparea_km2) + printed stats.
"""
import json
import sys
from pathlib import Path

import numpy as np

HYDRODL = Path.home() / "projects/water_loss/dPLHBVrelease-master/hydroDL-dev"
sys.path.append(str(HYDRODL))

MODEL_OUT = Path("/mnt/ssd1/data/water_loss/models/CONUS2717_AORC2F_v3_gradaccum/"
                 "exp_EPOCH100_BS100_RHO365_HS164_MUL14_HS24096_MUL24_trainBuff365_test")
EPOCH = 100
ATTRS_NC = "/mnt/ssd1/data/icechunk/merit_global_attributes_v2.nc"
EXPORT_IC = "/mnt/ssd1/data/icechunk/daily_dhbv2_distributed_aorc2f_merit_unit_catchments.ic"
OUT_CSV = Path(__file__).resolve().parent.parent / "output/tau_sweep/gamma_uh_params.csv"

ATTR_LIST = ["meanP", "ETPOT_Hargr", "aridity", "seasonality_P", "snow_fraction",
             "meanelevation", "meanslope", "NDVI", "Porosity",
             "HWSD_sand", "HWSD_silt", "HWSD_clay", "permeability", "uparea"]
NFEA2, NMUL2 = 13, 4          # routpara = Ann output cols 52:54


def main() -> None:
    import icechunk
    import torch
    import xarray as xr
    import zarr
    from hydroDL.data import scale
    from hydroDL.model.rnn import AnnModel

    sd = torch.load(MODEL_OUT / f"model_Ep{EPOCH}.pt", map_location="cpu",
                    weights_only=False)
    assert isinstance(sd, dict), "expected a state_dict checkpoint"
    ann_sd = {k[len("Ann."):]: v for k, v in sd.items() if k.startswith("Ann.")}
    ny = ann_sd["h2o.weight"].shape[0]
    assert ny == NFEA2 * NMUL2 + 2, f"Ann ny={ny}, expected {NFEA2*NMUL2+2}"
    ann = AnnModel(nx=len(ATTR_LIST), ny=ny, hiddenSize=4096, dropout_rate=0.5)
    ann.load_state_dict(ann_sd)
    ann.eval()

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(EXPORT_IC))
    root = zarr.open_group(store=repo.readonly_session("main").store, mode="r")
    divide_ids = root["divide_id"][:]
    print(f"{len(divide_ids)} divides from export store")

    ads = xr.open_dataset(ATTRS_NC)
    comid_to_aidx = {int(c): i for i, c in enumerate(ads["COMID"].values)}
    aidx = np.array([comid_to_aidx[int(c)] for c in divide_ids])
    uparea_all = np.power(10.0, ads["log10_uparea"].values[aidx]).astype(np.float32)
    attrs_all = np.stack(
        [ads[v].values[aidx].astype(np.float32) for v in ATTR_LIST[:-1]], axis=1)
    attrs_all = np.concatenate([attrs_all, uparea_all[:, None]], axis=1)

    with open(MODEL_OUT / "dapengscaler_stat.json") as f:
        stat_dict = json.load(f)
    attr_norm = scale._trans_norm(attrs_all.copy(), ATTR_LIST, stat_dict,
                                  log_norm_cols=[], to_norm=True)
    attr_norm[attr_norm != attr_norm] = 0

    routpara = np.empty((len(divide_ids), 2), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(divide_ids), 8192):
            e = min(s + 8192, len(divide_ids))
            out = ann(torch.from_numpy(attr_norm[s:e]).float())
            routpara[s:e] = out[:, NFEA2 * NMUL2:NFEA2 * NMUL2 + 2].numpy()
            if s % 65536 == 0:
                print(f"  {e}/{len(divide_ids)}")

    routa = 2.9 * routpara[:, 0]
    routb = 6.5 * routpara[:, 1]
    a_eff = routa + 0.1
    theta_eff = routb + 0.5
    tau_uh = a_eff * theta_eff

    import pandas as pd
    df = pd.DataFrame({"divide_id": divide_ids, "routa": routa, "routb": routb,
                       "a_eff": a_eff, "theta_eff": theta_eff,
                       "tau_uh_days": tau_uh, "uparea_km2": uparea_all})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")

    q = np.percentile(tau_uh, [5, 25, 50, 75, 95])
    print(f"tau_uh_days: median {q[2]:.2f}  IQR [{q[1]:.2f}, {q[3]:.2f}]  "
          f"p5 {q[0]:.2f}  p95 {q[4]:.2f}  mean {tau_uh.mean():.2f}")
    print(f"routa: median {np.median(routa):.3f}   routb: median {np.median(routb):.3f}")
    from scipy.stats import spearmanr
    rho = spearmanr(tau_uh, np.log10(uparea_all)).statistic
    print(f"spearman(tau_uh, log10 uparea) = {rho:.3f}")
    print(f"frac tau_uh > 1 day: {(tau_uh > 1).mean():.1%}   "
          f"> 2 days: {(tau_uh > 2).mean():.1%}   > 4 days: {(tau_uh > 4).mean():.1%}")


if __name__ == "__main__":
    main()
