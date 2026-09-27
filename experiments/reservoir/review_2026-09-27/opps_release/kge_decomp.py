"""KGE decomposition (r, alpha, beta) of the full-population arms, off vs learned, both seeds, WY1996-2010."""
import json
import numpy as np
import pandas as pd
import zarr
from scipy.stats import binomtest

WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs"
ARMS = {42: ("2026-09-27T07-29-47Z-train-and-test", "2026-09-27T07-29-55Z-train-and-test"),
        43: ("2026-09-27T10-31-30Z-train-and-test", "2026-09-27T10-31-50Z-train-and-test")}


def load(run):
    z = zarr.open(f"{RUNS}/{run}/eval/predictions.zarr", mode="r")
    ids = pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]])
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    return ids, t, z["predictions"][:].astype(float), z["observations"][:].astype(float)


def decomp(P, O, m):
    out = np.full((P.shape[0], 5), np.nan)
    for i in range(P.shape[0]):
        p, o = P[i], O[i]
        k = m & np.isfinite(o) & np.isfinite(p)
        if k.sum() < 100:
            continue
        p, o = p[k], o[k]
        r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
        al, be = p.std() / o.std(), p.mean() / o.mean()
        nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
        out[i] = [r, al, be, 1 - np.sqrt((r - 1) ** 2 + (al - 1) ** 2 + (be - 1) ** 2), nse]
    return pd.DataFrame(out, columns=["r", "alpha", "beta", "kge", "nse"])


def paired(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-9]; up = int((nz > 0).sum())
    rng = np.random.default_rng(42)
    bs = np.median(rng.choice(d, (2000, len(d))), axis=1)
    return dict(n=int(len(d)), median=round(float(np.median(d)), 5), ci=[round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
                n_up=up, n_down=int(len(nz) - up), sign_p=round(float(binomtest(up, len(nz)).pvalue), 4))


meta = pd.read_csv(f"{WT}/experiments/reservoir/full_run/paired_full_run.csv", dtype={"STAID": str}).set_index("STAID")
res = {}
for seed, (off, lrn) in ARMS.items():
    ids_o, t, Po, Oo = load(off)
    ids_l, t2, Pl, Ol = load(lrn)
    assert (ids_o == ids_l).all() and (t == t2).all()
    wy = np.asarray(t.year + (t.month >= 10))
    m = (wy >= 1996) & (wy <= 2010)
    do, dl = decomp(Po, Oo, m), decomp(Pl, Ol, m)
    do.index = ids_o; dl.index = ids_l
    mm = meta.reindex(ids_o)
    groups = {
        "dammed_ge10mcm": mm.n_nid_ge10mcm > 0,
        "dam_on_gauge_reach": mm.nid_on_gauge_reach.astype(str) == "True",
        "dor_gt_0.5": mm.nid_dor > 0.5,
        "undammed": mm.n_nid_ge10mcm == 0,
    }
    r = {}
    for gname, sel in groups.items():
        sel = sel.fillna(False).values
        g = {}
        for col in ["r", "alpha", "beta", "kge", "nse"]:
            g[f"d_{col}"] = paired(dl[col][sel] - do[col][sel])
            g[f"median_{col}_off"] = round(float(do[col][sel].median()), 4)
            g[f"median_{col}_learned"] = round(float(dl[col][sel].median()), 4)
        # how much of the KGE change is the alpha term: gauges where alpha fell and KGE fell
        da = (dl.alpha - do.alpha)[sel]; dk = (dl.kge - do.kge)[sel]
        g["frac_alpha_down"] = round(float((da < 0).mean()), 3)
        g["frac_alpha_off_below_1"] = round(float((do.alpha[sel] < 1).mean()), 3)
        g["spearman_dkge_dalpha"] = round(float(pd.Series(dk.values).corr(pd.Series(da.values), method="spearman")), 3)
        r[gname] = g
    res[f"seed{seed}"] = r
    pd.concat([do.add_suffix("_off"), dl.add_suffix("_learned")], axis=1).to_csv(f"/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_release/kge_decomp_seed{seed}.csv")
json.dump(res, open("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_release/kge_decomp.json", "w"), indent=1)
print(json.dumps(res, indent=1))
