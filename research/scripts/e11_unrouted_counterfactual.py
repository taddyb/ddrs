"""E-11: what explains the low unrouted NSE of the lumped product? Counterfactual edits of the summed-Q' series.
Read-only on runs. Run with ddrs-py/.venv python (numpy, xarray, zarr) or the ddr venv."""
import json, numpy as np, zarr, sys
R = "/home/tbindas/projects/ddrs/.ddrs/runs/"
RUNS = {"dHBV2 lumped": "2026-09-13T13-56-50Z-train-and-test", "dHBV2 distributed": "2026-09-13T17-22-30Z-train-and-test",
        "UH": "2026-09-12T23-39-03Z-train-and-test", "hydroDL LSTM": "2026-09-13T17-21-58Z-train-and-test"}
def eval_ids(run):
    z = zarr.open_group(R+run+"/eval/predictions.zarr", mode="r")
    a = np.asarray(z["gage_ids"][:])
    if a.dtype == np.uint8: return ["".join(map(chr, r)).strip("\x00 ") for r in a]
    return [str(x) for x in a]
def nse(p, o):
    m = np.isfinite(o) & np.isfinite(p)
    num = np.where(m, (p-o)**2, 0).sum(1); den = np.where(m, (o-np.nanmean(np.where(m, o, np.nan), 1, keepdims=True))**2, 0).sum(1)
    return 1-num/den
def delay(x, d):
    return x if d == 0 else np.concatenate([np.repeat(x[:, :1], d, 1), x[:, :-d]], 1)
def trail(x, k):
    c = np.cumsum(np.concatenate([np.repeat(x[:, :1], k-1, 1), x], 1), 1); c = np.concatenate([np.zeros((x.shape[0], 1)), c], 1)
    return (c[:, k:]-c[:, :-k])/k
out = {}
for name, run in RUNS.items():
    m = json.load(open(R+run+"/baseline/manifest.json")); ids = m["gage_ids"]; n = m["n_days"]
    P = np.fromfile(R+run+"/baseline/predictions.f32", "f4").reshape(len(ids), n).astype("f8")
    O = np.fromfile(R+run+"/baseline/observations.f32", "f4").reshape(len(ids), n).astype("f8")
    ev = eval_ids(run); ix = {g: i for i, g in enumerate(ids)}; sel = [ix[g] for g in ev]
    P = P[sel, 1:-1]; O = O[sel, 1:-1]          # 5477 d, as E-10
    W = 10                                       # common scoring window start so shifted variants are comparable
    res = {"gauges": len(sel), "raw_full": float(np.nanmedian(nse(P, O)))}
    P0, O0 = P[:, W:], O[:, W:]
    res["raw"] = float(np.nanmedian(nse(P0, O0)))
    for d in (1, 2, 3):
        res[f"delay{d}"] = float(np.nanmedian(nse(delay(P, d)[:, W:], O0)))
    best = np.max(np.stack([nse(delay(P, d)[:, W:], O0) for d in range(0, 6)], 1), 1)
    res["oracle_delay0to5"] = float(np.nanmedian(best))
    for k in (2, 3, 5):
        res[f"trail{k}"] = float(np.nanmedian(nse(trail(P, k)[:, W:], O0)))
    res["delay1+trail3"] = float(np.nanmedian(nse(trail(delay(P, 1), 3)[:, W:], O0)))
    res["delay2+trail3"] = float(np.nanmedian(nse(trail(delay(P, 2), 3)[:, W:], O0)))
    mo = np.isfinite(O0)
    vr = np.where(mo, P0, 0).sum(1)/np.where(mo, O0, 0).sum(1)
    res["volume_rescale_only"] = float(np.nanmedian(nse(P0/vr[:, None], O0)))
    res["delay1+trail3+volume"] = float(np.nanmedian(nse(trail(delay(P, 1), 3)[:, W:]/vr[:, None], O0)))
    res["share_nse_below0"] = float((nse(P0, O0) < 0).mean())
    out[name] = res
keys = list(next(iter(out.values())).keys())
print("%-24s" % "median NSE" + "".join("%20s" % k for k in out))
for k in keys:
    print("%-24s" % k + "".join("%20.4f" % out[n][k] if k != "gauges" else "%20d" % out[n][k] for n in out))
