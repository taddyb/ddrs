#!/usr/bin/env python
"""Pass bar for the flood pool (fixed 2026-09-29, before results): replay pair and S6 smoke arms.

  (1) replay: engine pool minus no-pool twin at the on-reach flood-control gauges keeps >= half the offline gain
      (laws_v6 FA minus L2 at the same gauges), and no replay dam creates > 0.5 % of its inflow;
  (2) S6pool minus S6base at all smoke flood-control dam gauges: median > 0 and 95 % CI lower bound >= 0; controls
      unchanged (max |dNSE| vs the no-dam arm is 0 for both arms); no S6pool dam creates >= 5 % of its inflow.
Reads /home/tbindas/.claude/jobs/dacd6d8c/tmp/v6.status; writes smoke_v2/flood_gate.json.
"""
import json, os
import numpy as np, pandas as pd, zarr

W = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/"
R = W + ".ddrs/runs/"
SM = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/"
OFF = "2026-09-27T04-29-33Z-train-and-test"
st = {l.split()[0]: l.split()[1] for l in open("/home/tbindas/.claude/jobs/dacd6d8c/tmp/v6.status") if l.strip()}


def load(run):
    z = zarr.open(R + run + "/eval/predictions.zarr", mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    return pd.Index(ids), z["predictions"][:].astype(float), z["observations"][:].astype(float)


def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    return 1 - ((p[m] - o[m]) ** 2).sum() / ((o[m] - o[m].mean()) ** 2).sum()


def med_ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return None
    b = np.median(np.random.default_rng(42).choice(d, (2000, len(d))), axis=1)
    return dict(m=round(float(np.median(d)), 5), lo=round(float(np.percentile(b, 2.5)), 5),
                hi=round(float(np.percentile(b, 97.5)), 5), up=int((d > 1e-9).sum()), down=int((d < -1e-9).sum()), n=int(len(d)))


ids, P0, O = load(OFF)
pos = {s: i for i, s in enumerate(ids)}
N = {"off": pd.Series({s: nse(P0[i], O[i]) for s, i in pos.items()})}
for name, run in st.items():
    if os.path.exists(R + run + "/eval/predictions.zarr"):
        _, P, _ = load(run)
        N[name] = pd.Series({s: nse(P[i], O[i]) for s, i in pos.items()})
    else:
        print(f"{name}: no predictions yet ({run})")
sm = pd.read_csv(SM + "smoke_gauges.csv", dtype={"STAID": str}).set_index("STAID")
onr = pd.read_csv(SM + "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID").on_reach.astype(str) == "True"
dam = sm.index[sm.role == "dam"]
ctl = sm.index[sm.role == "control"]
fc = [s for s in dam if sm.dam_purpose[s] == "Flood Risk Reduction"]
tab = pd.read_csv(W + "experiments/reservoir/smoke/fixed_FA.csv", dtype={"STAID": str})
fc_rep = [s for s in tab.STAID if s in pos]
target = [s for s in dam if onr.get(s, False) and sm.nid_dor[s] > 0.5]
off_fits = pd.read_csv(W + "experiments/reservoir/laws_v6/laws_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")


def share_stats(run):
    p = R + run + "/release_clamp.csv"
    if not os.path.exists(p):
        return None
    c = pd.read_csv(p)
    s = c.created_share
    return dict(dams=int(len(s)), gt05=int((s > 0.005).sum()), ge5=int((s >= 0.05).sum()), max=round(float(s.max()), 5))


out = {}
if "replay_FA_pool" in N and "replay_FA_nopool" in N:
    d = (N["replay_FA_pool"] - N["replay_FA_nopool"])[fc_rep]
    offd = (off_fits.FA_nse - off_fits.L2_nse).reindex(fc_rep)
    sh = share_stats(st["replay_FA_pool"])
    out["replay"] = dict(engine=med_ci(d), offline=med_ci(offd), created=sh,
                         kept=round(float(np.nanmedian(d) / np.nanmedian(offd)), 3) if np.nanmedian(offd) else None)
    out["pass1"] = bool(out["replay"]["kept"] is not None and out["replay"]["kept"] >= 0.5 and sh and sh["gt05"] == 0)
if "S6pool" in N and "S6base" in N:
    d = N["S6pool"] - N["S6base"]
    out["s6"] = {g: med_ci(d[idx]) for g, idx in (("flood_control", fc), ("target_117", target), ("dam_all", list(dam)))}
    out["s6_vs_off"] = {a: {g: med_ci((N[a] - N["off"])[idx]) for g, idx in (("target_117", target), ("dam_all", list(dam)))}
                        for a in ("S6base", "S6pool")}
    out["controls_max"] = {a: round(float((N[a] - N["off"])[ctl].abs().max()), 6) for a in ("S6base", "S6pool")}
    out["created"] = {a: share_stats(st[a]) for a in ("S6base", "S6pool")}
    f = out["s6"]["flood_control"]
    cr = out["created"]["S6pool"]
    out["pass2"] = bool(f["m"] > 0 and f["lo"] >= 0 and max(out["controls_max"].values()) == 0 and cr and cr["ge5"] == 0)
for a in ("S6pool",):
    p = R + st.get(a, "x") + "/release_pool.csv"
    if os.path.exists(p):
        pl = pd.read_csv(p)
        out["pool_diag"] = dict(n=int(len(pl)), median_max_F_days=round(float(pl.max_F_days.median()), 3),
                                captured_m3=float(pl.captured_m3.sum()), evacuated_m3=float(pl.evacuated_m3.sum()))
print(json.dumps(out, indent=1))
json.dump(out, open(W + "experiments/reservoir/smoke_v2/flood_gate.json", "w"), indent=1)
