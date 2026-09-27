"""Summaries of the feature-limited (5-fold CV random forest) rule-curve parameters: dam gain, placebo control gain, DiD."""
import json

import numpy as np
import pandas as pd

import rc
import summarise as S

NAMES = ["cvL1", "cvL3", "cvL3b", "cvL4", "cvL4rot", "cvL4ec", "cvL4rotec", "cvL4pen"]

if __name__ == "__main__":
    df, dam, ctl = S.load()
    r2 = json.load(open(f"{rc.HERE}/features_r2.json"))
    res, lines = {}, []
    for setname, sel in [("on_reach", dam.on_reach.values), ("all_dams", np.ones(len(dam), bool))]:
        f = pd.read_csv(f"{rc.HERE}/features_{setname}_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
        d, c = dam[sel].join(f), ctl[sel]
        fc = f.reindex(c.STAID.values); fc.index = c.index; c = c.join(fc)
        res[setname] = {"r2": r2[setname]["r2"]}
        lines.append(f"\n===== feature-limited, {setname}: {len(d)} dams =====  CV R2: {r2[setname]['r2']}")
        for nm in NAMES:
            res[setname][nm] = {}
            lines.append(f"-- {nm}")
            for sub, idx in S.subsets(d, c):
                dd, cc = d.loc[idx], c.loc[idx]
                gd = (dd[f"{nm}_nse"] - dd.L0_nse).values
                gc = (cc[f"{nm}_nse"] - cc.L0_nse).values
                e = dict(dam=S.paired(gd), placebo_control=S.paired(gc), did=S.paired(gd - gc),
                         mean_dnse_clip=round(float((dd[f"{nm}_nse"].clip(lower=-1) - dd.L0_nse.clip(lower=-1)).mean()), 4),
                         dkge=S.paired(dd[f"{nm}_kge"] - dd.L0_kge), dr=S.paired(dd[f"{nm}_r"] - dd.L0_r),
                         dalpha=S.paired(dd[f"{nm}_alpha"] - dd.L0_alpha), dbeta=S.paired(dd[f"{nm}_beta"] - dd.L0_beta),
                         floor_mean=round(float(dd[f"{nm}_floor_test"].mean()), 4))
                res[setname][nm][sub] = e
                lines.append(f"  {sub:8s} n={len(idx):3d} dam {S.fmt(e['dam'])} mean(clip) {e['mean_dnse_clip']:+.4f} | placebo ctl "
                             f"{S.fmt(e['placebo_control'])} | DiD {S.fmt(e['did'])} | dKGE {e['dkge']['median']:+.4f} "
                             f"dr {e['dr']['median']:+.4f} dalpha {e['dalpha']['median']:+.4f} | floor {e['floor_mean']:.3f}")
    json.dump(res, open(f"{rc.HERE}/features.json", "w"), indent=1)
    open(f"{rc.HERE}/features.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
