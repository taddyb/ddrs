"""Seasonal-cycle (monthly climatology) error of summed Q' at dam gauges, by dam size, purpose and region,
and whether it predicts where a linear reservoir or the rule curve helps.

Per smoke dam gauge (458) with its matched control: clim = mean over days of (month-mean pred - month-mean obs)^2
/ var(obs), for summed Q' (no routing) and the routed no-dam model (seed 42, test WY1996-2010); clim_excess = dam
minus its control. Gains: offline per-gauge plain bucket (fit WY1983-95, score WY1996-2010), offline rule curve,
and the in-engine frozen additive bucket (smoke arm 2026-09-27T19-54-53Z vs the smoke no-dam arm).
Writes clim_by_dam.csv and clim_by_dam.png next to this script's OUT dir."""
import json
import numpy as np, pandas as pd, zarr
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/tbindas/.claude/jobs/dacd6d8c/tmp/"
RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs/"
WTR = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/.ddrs/runs/"
SM = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/"
OFF = "2026-09-27T07-29-47Z-train-and-test"


def loadz(path):
    z = zarr.open(path + "/eval/predictions.zarr", mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    return {s: i for i, s in enumerate(ids)}, t, z["predictions"][:].astype(float), z["observations"][:].astype(float)


idx, t, P, O = loadz(RUNS + OFF)
mon = t.month.values
bm = json.load(open(RUNS + OFF + "/baseline/manifest.json"))
B = np.fromfile(RUNS + OFF + "/baseline/predictions.f32", dtype=np.float32).reshape(bm["n_gauges"], bm["n_days"])
bidx = {str(g): i for i, g in enumerate(bm["gage_ids"])}
off0 = int((t[0].to_datetime64().astype("datetime64[D]") - np.datetime64(str(bm["time_range_daily"][0])[:10])).astype(int))
sidx, st, SP0, SO = loadz(WTR + "2026-09-27T04-29-33Z-train-and-test")
_, _, SP2, _ = loadz(WTR + "2026-09-27T19-54-53Z-train-and-test")


def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    return 1 - ((p[m] - o[m]) ** 2).sum() / ((o[m] - o[m].mean()) ** 2).sum()


def clim(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    p, o, mm = p[m], o[m], mon[m]
    cp = pd.Series(p).groupby(mm).transform("mean").values
    co = pd.Series(o).groupby(mm).transform("mean").values
    return ((cp - co) ** 2).mean() / o.var()


sm = pd.read_csv(SM + "smoke_gauges.csv", dtype={"STAID": str, "control_for": str, "huc2": str})
onr = pd.read_csv(SM + "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID").on_reach.astype(str) == "True"
ctl_of = sm[sm.role == "control"].set_index("control_for").STAID
fits = pd.read_csv(OUT + "rulecurve/fits_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
g3 = pd.read_csv("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv", dtype={"STAID": str}).set_index("STAID")

rows = []
for _, x in sm[sm.role == "dam"].iterrows():
    s, c = x.STAID, ctl_of.get(x.STAID)
    if s not in idx or c not in idx or s not in bidx or c not in bidx:
        continue
    b = lambda g: B[bidx[g], off0:off0 + len(t)].astype(float)
    r = dict(STAID=s, dor=x.nid_dor, storage=x.dam_storage_mcm, height=x.dam_height_m, purpose=x.dam_purpose,
             huc2=x.huc2, on_reach=bool(onr.get(s, False)), lat=g3.LAT_GAGE.get(s), lon=g3.LNG_GAGE.get(s),
             clim_base=clim(b(s), O[idx[s]]), clim_base_ctl=clim(b(c), O[idx[c]]),
             clim_routed=clim(P[idx[s]], O[idx[s]]), clim_routed_ctl=clim(P[idx[c]], O[idx[c]]),
             nse_routed=nse(P[idx[s]], O[idx[s]]), nse_routed_ctl=nse(P[idx[c]], O[idx[c]]))
    if s in fits.index:
        f = fits.loc[s]
        r.update(g_bucket_off=f.L2_nse - f.L0_nse, g_seasbucket_off=f.L1_nse - f.L0_nse, g_rule_off=f.L4_nse - f.L0_nse)
    if s in sidx:
        r["g_bucket_engine"] = nse(SP2[sidx[s]], SO[sidx[s]]) - nse(SP0[sidx[s]], SO[sidx[s]])
    rows.append(r)
D = pd.DataFrame(rows)
D["clim_excess"] = D.clim_base - D.clim_base_ctl
D["gap"] = D.nse_routed - D.nse_routed_ctl
D["size"] = pd.cut(D.storage, [0, 100, 1000, 1e9], labels=["10-100 MCM", "100-1000 MCM", ">1000 MCM"])
D["dorbin"] = pd.cut(D.dor, [-1e-9, 0.1, 0.5, 1, 1e9], labels=["<=0.1", "0.1-0.5", "0.5-1", ">1"])
top = ["Flood Risk Reduction", "Hydroelectric", "Water Supply", "Irrigation", "Recreation"]
D["purp"] = np.where(D.purpose.isin(top), D.purpose, "Other")
REG = {"01": "Northeast", "02": "Northeast", "03": "Southeast", "05": "Ohio-Tennessee", "06": "Ohio-Tennessee",
       "04": "Great Lakes-Upper Miss", "07": "Great Lakes-Upper Miss", "09": "Great Lakes-Upper Miss",
       "08": "Plains-Texas", "10": "Plains-Texas", "11": "Plains-Texas", "12": "Plains-Texas",
       "13": "Southwest-Great Basin", "14": "Southwest-Great Basin", "15": "Southwest-Great Basin",
       "16": "Southwest-Great Basin", "17": "Pacific Northwest", "18": "California"}
D["region"] = D.huc2.map(REG)
D.to_csv(OUT + "clim_by_dam.csv", index=False)

cols = ["clim_excess", "gap", "g_bucket_off", "g_bucket_engine", "g_rule_off"]
def table(by):
    g = D.groupby(by, observed=True)
    out = g.size().rename("n").to_frame().join(g[cols].median())
    return out.round(3)
pd.set_option("display.width", 200)
for by in ["dorbin", "size", "purp", "region"]:
    print(f"\n=== by {by} (medians; clim_excess = seasonal-cycle error of summed Q' above the matched control)")
    print(table(by).to_string())

print("\nSpearman with clim_excess (all dam gauges, then DOR > 0.5):")
for g in ["g_bucket_off", "g_bucket_engine", "g_rule_off", "gap"]:
    a = D[["clim_excess", g]].dropna()
    h = D[D.dor > 0.5][["clim_excess", g]].dropna()
    print(f"  {g:16s} all rho {spearmanr(a.clim_excess, a[g])[0]:+.2f} (n={len(a)})   DOR>0.5 rho {spearmanr(h.clim_excess, h[g])[0]:+.2f} (n={len(h)})")
D["clim_tercile"] = pd.qcut(D.clim_excess, 3, labels=["low", "mid", "high"])
print("\nGains by clim_excess tercile (medians):")
print(D.groupby("clim_tercile", observed=True)[["clim_excess", "gap", "g_bucket_off", "g_bucket_engine", "g_rule_off"]].median().round(4).to_string())
print("\nShare of gauges where the linear reservoir helps by > 0.01 (engine / offline), by tercile:")
print(D.groupby("clim_tercile", observed=True).agg(engine=("g_bucket_engine", lambda v: (v > 0.01).mean()),
                                                   offline=("g_bucket_off", lambda v: (v > 0.01).mean()),
                                                   rule=("g_rule_off", lambda v: (v > 0.01).mean())).round(2).to_string())

# Figure: map of clim excess (colour) and storage (size); gains vs clim excess.
fig = plt.figure(figsize=(13, 9.5))
ax = fig.add_axes([0.04, 0.42, 0.62, 0.55])
allg = g3[g3.index.isin(idx.keys())]
ax.scatter(allg.LNG_GAGE, allg.LAT_GAGE, s=3, c="#c9d1d6", lw=0, zorder=1)
sz = 12 + 40 * np.clip(np.log10(D.storage.astype(float)) - 1, 0, 3)
sc = ax.scatter(D.lon, D.lat, s=sz, c=np.clip(D.clim_excess, 0, 0.3), cmap="viridis_r", vmin=0, vmax=0.3,
                edgecolor="white", lw=0.4, zorder=2)
cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
cb.set_label("seasonal-cycle error of summed Q' above matched control")
ax.set_xlim(-125, -66); ax.set_ylim(24, 50); ax.set_aspect(1.25)
ax.set_title("458 smoke dam gauges: colour = seasonal-cycle error excess, size = storage of nearest dam", fontsize=11, loc="left")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
titles = [("g_bucket_engine", "Linear reservoir in the engine (frozen, additive)"),
          ("g_bucket_off", "Linear reservoir, offline per-gauge fit"),
          ("g_rule_off", "Rule curve, offline per-gauge fit")]
for k, (g, ttl) in enumerate(titles):
    a = fig.add_axes([0.06 + k * 0.32, 0.07, 0.27, 0.27])
    x = np.clip(D.clim_excess, -0.05, 0.4); y = np.clip(D[g], -0.1, 0.3)
    col = np.where(D.dor > 0.5, "#2a78d6", "#9aa7ae")
    a.scatter(x, y, s=10, c=col, lw=0, alpha=0.8)
    a.axhline(0, color="#8a959b", lw=0.8); a.axhline(0.03, color="#b8651b", lw=1, ls="--")
    q = D[["clim_excess", g]].dropna()
    rho = spearmanr(q.clim_excess, q[g])[0]
    a.set_title(f"{ttl}\nSpearman {rho:+.2f}", fontsize=10, loc="left")
    a.set_xlabel("seasonal-cycle error excess (summed Q')")
    if k == 0: a.set_ylabel("NSE gain at the gauge")
    a.set_ylim(-0.1, 0.3)
fig.text(0.06, 0.005, "blue: DOR > 0.5; grey: DOR <= 0.5; dashed line: +0.03 target; values clipped to the plotted range",
         fontsize=9, color="#555")
fig.savefig(OUT + "clim_by_dam.png", dpi=110)
print("\nwrote", OUT + "clim_by_dam.png")
