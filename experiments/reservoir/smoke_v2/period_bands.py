"""Timescale of the dam error: split each gauge's MSE (normalised by obs variance) into FFT period bands,
dam gauges vs matched controls, off arm seed 42. Gaps filled by linear interpolation of the error series."""
import numpy as np, pandas as pd, zarr
RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs/"
WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/"
z = zarr.open(RUNS + "2026-09-27T07-29-47Z-train-and-test/eval/predictions.zarr", mode="r")
ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
P, O = z["predictions"][:].astype(float), z["observations"][:].astype(float)
idx = {s: i for i, s in enumerate(ids)}
sm = pd.read_csv(WT + "smoke/smoke_gauges.csv", dtype={"STAID": str, "control_for": str})
BANDS = [(0, 7), (7, 30), (30, 120), (120, 400), (400, 1e9)]
LAB = ["<7d", "7-30d", "30-120d", "120-400d", ">400d"]


def bands(i):
    e = P[i] - O[i]
    o = O[i]
    m = np.isfinite(e)
    if m.sum() < 0.8 * len(e):
        return None
    x = np.arange(len(e))
    e = np.interp(x, x[m], e[m])
    e = e - 0  # keep mean: the mean error is the >400 d band's DC term
    var = np.nanvar(o)
    F = np.fft.rfft(e)
    f = np.fft.rfftfreq(len(e))
    per = np.where(f > 0, 1 / np.maximum(f, 1e-12), 1e12)
    pw = np.abs(F) ** 2
    pw[1:] *= 2
    tot = pw.sum()
    mse = (e**2).mean()
    return [mse * pw[(per >= lo) & (per < hi)].sum() / tot / var for lo, hi in BANDS]


rows = []
dam = sm[sm.role == "dam"]
ctl = sm[sm.role == "control"].set_index("control_for")
for _, x in dam.iterrows():
    if x.STAID not in idx or x.STAID not in ctl.index or ctl.loc[x.STAID, "STAID"] not in idx:
        continue
    a = bands(idx[x.STAID])
    b = bands(idx[ctl.loc[x.STAID, "STAID"]])
    if a is None or b is None:
        continue
    rows.append(dict(dor=x.nid_dor, **{"dam " + l: v for l, v in zip(LAB, a)}, **{"ctl " + l: v for l, v in zip(LAB, b)}))
R = pd.DataFrame(rows)
R["bin"] = pd.cut(R.dor, [-1e-9, 0.1, 0.5, 1, 2, 1e9], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
g = R.groupby("bin", observed=True)
print("median normalised MSE per period band (sum over bands = 1 - NSE)")
print(g[[c for c in R.columns if c.startswith("dam") or c.startswith("ctl")]].median().round(3).T.to_string())
hi = R[R.dor > 0.5]
print("\nDOR>0.5 (n=%d), median excess (dam - control) per band:" % len(hi))
print({l: round((hi["dam " + l] - hi["ctl " + l]).median(), 3) for l in LAB})
print("mean excess:", {l: round((hi["dam " + l] - hi["ctl " + l]).mean(), 3) for l in LAB})
