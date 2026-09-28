#!/usr/bin/env python
"""Data for the overnight results page (2026-09-28): every smoke arm scored the same way from its predictions.

Writes data.json next to this script: arm summaries (on-reach DOR > 0.5 target set, all 458 dam gauges, matched
controls, clean subset where a corrected clamp account exists), the featured arm's per-gauge table (map, bins,
purposes), weekly hydrographs for the 117 target gauges, the dam-vs-control gap by DOR on the smoke no-dam arm,
and the cap attribution. Run with /home/tbindas/projects/ddr/.venv/bin/python.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
WT = Path("/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4")
R = WT / ".ddrs/runs"
MAIN = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
SM = Path("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke")
OFF = R / "2026-09-27T04-29-33Z-train-and-test"
FEATURED = "S4v3"

# key, label, group, run dir, run dir holding the corrected clamp account (or None)
ARMS = [
    ("joint", "Joint training, today's law", "trained", MAIN / "2026-09-27T07-30-13Z-train-and-test", None),
    ("S1", "Frozen routing, today's law", "trained", R / "2026-09-27T19-55-11Z-train-and-test", None),
    ("S2", "Additive row, head residence time", "trained", R / "2026-09-27T19-54-53Z-train-and-test", None),
    ("S3", "+ rule curve, first try", "trained", R / "2026-09-27T19-54-59Z-train-and-test", None),
    ("T0v3", "Per-dam residence time only", "trained", R / "2026-09-27T23-36-12Z-train-and-test", R / "2026-09-28T01-45-06Z-train-and-test"),
    ("S3v3", "Rule curve, feasibility penalty", "trained", R / "2026-09-27T23-35-58Z-train-and-test", R / "2026-09-28T01-11-42Z-train-and-test"),
    ("S4v3", "Rule curve + per-dam residence time", "trained", R / "2026-09-27T23-36-04Z-train-and-test", R / "2026-09-28T01-44-59Z-train-and-test"),
    ("S5forgive", "Capped row, faster per-dam fit", "trained", R / "2026-09-28T03-21-27Z-train-and-test", R / "2026-09-28T03-21-27Z-train-and-test"),
    ("S5carry", "Same, mass-conserving floor", "trained", R / "2026-09-28T03-21-23Z-train-and-test", R / "2026-09-28T03-21-23Z-train-and-test"),
    ("L2", "Offline residence time, uncapped row", "replay", R / "2026-09-27T23-22-35Z-train-and-test", None),
    ("L2pos", "Offline residence time, capped row", "replay", R / "2026-09-28T02-49-07Z-train-and-test", R / "2026-09-28T02-49-07Z-train-and-test"),
    ("L4", "Offline rule curve, uncapped row", "replay", R / "2026-09-27T23-22-39Z-train-and-test", None),
    ("L4pos", "Offline rule curve, capped row", "replay", R / "2026-09-28T02-49-11Z-train-and-test", R / "2026-09-28T02-49-11Z-train-and-test"),
    ("cap", "Capped row alone, 1 h storage", "control", R / "2026-09-28T03-55-12Z-train-and-test", R / "2026-09-28T03-55-12Z-train-and-test"),
]


def load(run):
    z = zarr.open(str(run / "eval/predictions.zarr"), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    return pd.Index(ids), t, z["predictions"][:].astype(float), z["observations"][:].astype(float)


def metrics(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1]
    kge = 1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)
    return nse, kge


def med_ci(d, seed=42):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return None
    b = np.median(np.random.default_rng(seed).choice(d, (2000, len(d))), axis=1)
    return dict(m=round(float(np.median(d)), 5), lo=round(float(np.percentile(b, 2.5)), 5),
                hi=round(float(np.percentile(b, 97.5)), 5), up=int((d > 1e-9).sum()), down=int((d < -1e-9).sum()), n=int(len(d)))


ids, t, P0, O = load(OFF)
sm = pd.read_csv(SM / "smoke_gauges.csv", dtype={"STAID": str, "control_for": str, "huc2": str}).set_index("STAID")
fit = pd.read_csv(SM / "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID")
sm["on_reach"] = fit.on_reach.astype(str).reindex(sm.index) == "True"
mp = json.load(open(SM / "page/map.json"))
dam = sm.index[sm.role == "dam"]
ctl = sm.index[sm.role == "control"]
target = [s for s in dam if sm.on_reach[s] and sm.nid_dor[s] > 0.5]
pos = {s: i for i, s in enumerate(ids)}
base = pd.DataFrame([metrics(P0[pos[s]], O[pos[s]]) for s in ids], index=ids, columns=["nse", "kge"])

arms, per_gauge = [], {}
for key, label, group, run, clamp_run in ARMS:
    ids2, t2, P, _ = load(run)
    assert (ids2 == ids).all() and (t2 == t).all(), key
    m = pd.DataFrame([metrics(P[pos[s]], O[pos[s]]) for s in ids], index=ids, columns=["nse", "kge"])
    d = m - base
    share = None
    if clamp_run is not None and (clamp_run / "release_clamp.csv").exists():
        c = pd.read_csv(clamp_run / "release_clamp.csv").set_index("COMID")
        if "created_share" in c:
            share = c.created_share
    entry = dict(key=key, label=label, group=group, run=run.name,
                 target=med_ci(d.nse[target]), target_kge=med_ci(d.kge[target]),
                 dam_all=med_ci(d.nse[dam]), ctl_max=round(float(d.nse[ctl].abs().max()), 4),
                 median_nse=round(float(m.nse.median()), 4), median_kge=round(float(m.kge.median()), 4))
    if share is not None:
        dshare = pd.Series({s: share.get(int(sm.dam_COMID[s]), np.nan) for s in target})
        clean = [s for s in target if not (dshare[s] >= 0.005)]
        entry.update(clean=med_ci(d.nse[clean]), dams=int(len(share)), ge05=int((share >= 0.005).sum()),
                     ge5=int((share >= 0.05).sum()), p90=round(float(share.quantile(0.9)), 5))
    arms.append(entry)
    per_gauge[key] = d
    print(f"{key:10s} target {entry['target']['m']:+.4f} dam_all {entry['dam_all']['m']:+.4f} ctl_max {entry['ctl_max']}")

# Featured arm: per-gauge rows, bins, purposes.
F = per_gauge[FEATURED]
_, _, PF, _ = load(dict((a[0], a[3]) for a in ARMS)[FEATURED])
rows = []
for s in list(dam) + list(ctl):
    xy = mp["gauges"].get(s)
    rows.append(dict(id=s, name=str(sm.staname.get(s, ""))[:60], role=sm.role[s], x=xy[0] if xy else None, y=xy[1] if xy else None,
                     dor=round(float(sm.nid_dor[s]), 3) if sm.role[s] == "dam" else None, purpose=sm.dam_purpose.get(s) if sm.role[s] == "dam" else None,
                     on_reach=bool(sm.on_reach[s]), huc2=sm.huc2[s], nse_off=round(float(base.nse[s]), 4),
                     d=round(float(F.nse[s]), 4), dk=round(float(F.kge[s]), 4)))
dd = pd.DataFrame([r for r in rows if r["role"] == "dam"]).set_index("id")
bins = []
for lo, hi, lab in [(-1, 0.1, "0–0.1"), (0.1, 0.5, "0.1–0.5"), (0.5, 1, "0.5–1"), (1, 2, "1–2"), (2, 1e9, "above 2")]:
    x = dd[(dd.dor > lo) & (dd.dor <= hi)] if lo >= 0 else dd[dd.dor <= hi]
    bins.append(dict(lab=lab, all=med_ci(x.d), onr=med_ci(x[x.on_reach].d)))
top = ["Flood Risk Reduction", "Hydroelectric", "Water Supply", "Irrigation", "Recreation"]
purp = []
for p in top + ["Other"]:
    x = dd[dd.purpose.isin(top) == False] if p == "Other" else dd[dd.purpose == p]
    purp.append(dict(lab=p, all=med_ci(x.d), hi=med_ci(x[x.dor > 0.5].d)))

# Weekly hydrographs for the 117 target gauges: observed, no-dam arm, featured arm.
wk = pd.Series(np.arange(len(t)), index=t).resample("W-SUN").apply(lambda v: v.values)
hyd = {}
def wmean(arr, cols):
    out = []
    for c in cols:
        v = arr[c]
        v = v[np.isfinite(v)]
        out.append(float(f"{v.mean():.4g}") if len(v) >= 4 else None)
    return out
for s in target:
    i = pos[s]
    hyd[s] = dict(o=wmean(O[i], wk.values), n=wmean(P0[i], wk.values), f=wmean(PF[i], wk.values))
weeks = [d.strftime("%Y-%m-%d") for d in wk.index]

# Dam-vs-control gap by DOR on this page's own no-dam arm.
gap = []
for lo, hi, lab in [(-1, 0.1, "0–0.1"), (0.1, 0.5, "0.1–0.5"), (0.5, 1, "0.5–1"), (1, 2, "1–2"), (2, 1e9, "above 2")]:
    ss = [s for s in dam if (sm.nid_dor[s] <= hi and (lo < 0 or sm.nid_dor[s] > lo))]
    pairs = [(s, c) for c, s in sm[sm.role == "control"].control_for.items() if s in ss]
    dn = np.array([base.nse[s] for s, _ in pairs])
    cn = np.array([base.nse[c] for _, c in pairs])
    gap.append(dict(lab=lab, n=len(pairs), dam=round(float(np.median(dn)), 3), ctl=round(float(np.median(cn)), 3),
                    gap=round(float(np.median(dn - cn)), 3)))

out = dict(featured=FEATURED, n_target=len(target), n_dam=len(dam), n_ctl=len(ctl), arms=arms, gauges=rows, bins=bins,
           purposes=purp, weeks=weeks, hyd=hyd, gap=gap,
           cap_attr=json.load(open(WT / "experiments/reservoir/smoke_v2/cap_attribution.json")),
           map=dict(w=mp["w"], h=mp["h"], regions=mp["regions"], outline=mp["outline"]),
           offline=dict(bucket=0.025, rule=0.0494, rule_ci=[0.0276, 0.0811]))
(HERE / "data.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote", HERE / "data.json", round((HERE / "data.json").stat().st_size / 1e6, 2), "MB")
