#!/usr/bin/env python
"""How much of the v5 replay gain is the dam-row positivity cap itself (review v5 finding 1)?

Controls (test-only, zero-step resume at the smoke no-dam arm, binary ddrs-v5-90aae43): the 202 replay dams at
T = 1 h with no rule curve, cap on (cap_on) and cap off (cap_off). Paired per-gauge increments at on-reach DOR > 0.5:
  cap_off - off   : one hour of reservoir storage alone
  cap_on - cap_off: the cap alone
  L2pos - cap_on  : the fitted per-dam T0, given the cap
  L4pos - cap_on  : the fitted T0 + rule curve, given the cap
  L2 - cap_off    : the fitted per-dam T0 without the cap (v3 replay)
Optional --all <run>: the cap-on control over every dam (for the S5 arms), reported against the off arm.
"""
import argparse, json
import numpy as np, pandas as pd, zarr

W = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/"
R = W + ".ddrs/runs/"
SM = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/"
RUNS = dict(off="2026-09-27T04-29-33Z", cap_off="2026-09-28T03-55-07Z", cap_on="2026-09-28T03-55-03Z",
            L2="2026-09-27T23-22-35Z", L4="2026-09-27T23-22-39Z", L2pos="2026-09-28T02-49-07Z", L4pos="2026-09-28T02-49-11Z")
ap = argparse.ArgumentParser()
ap.add_argument("--all", default=None)
ap.add_argument("--s5", nargs="*", default=[])
a = ap.parse_args()
if a.all:
    RUNS["cap_on_all"] = a.all
for s in a.s5:
    k, v = s.split("=")
    RUNS[k] = v


def load(run):
    z = zarr.open(R + run + "-train-and-test/eval/predictions.zarr", mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    return {s: i for i, s in enumerate(ids)}, z["predictions"][:].astype(float), z["observations"][:].astype(float)


def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    return 1 - ((p[m] - o[m]) ** 2).sum() / ((o[m] - o[m].mean()) ** 2).sum()


def med_ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    b = np.median(np.random.default_rng(42).choice(d, (2000, len(d))), axis=1)
    return dict(median=round(float(np.median(d)), 4), ci=[round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
                up=int((d > 1e-9).sum()), down=int((d < -1e-9).sum()), n=int(len(d)))


sm = pd.read_csv(SM + "smoke_gauges.csv", dtype={"STAID": str}).set_index("STAID")
onr = pd.read_csv(SM + "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID").on_reach.astype(str) == "True"
idx, _, O = load(RUNS["off"])
tgt = [s for s in sm.index[(sm.role == "dam") & (sm.nid_dor > 0.5)] if onr.get(s, False) and s in idx]
dam_all = [s for s in sm.index[sm.role == "dam"] if s in idx]
N = {}
for k, run in RUNS.items():
    _, P, _ = load(run)
    N[k] = pd.Series({s: nse(P[idx[s]], O[idx[s]]) for s in dam_all})
D = pd.DataFrame(N)
T = D.loc[tgt]
out = {}
pairs = [("vs off", k, "off") for k in RUNS if k != "off"] + [
    ("hour of storage", "cap_off", "off"), ("cap alone", "cap_on", "cap_off"),
    ("fitted T0 | cap", "L2pos", "cap_on"), ("fitted T0 + rule curve | cap", "L4pos", "cap_on"),
    ("fitted T0 | no cap", "L2", "cap_off"), ("rule curve over T0 | cap", "L4pos", "L2pos")]
pairs += [(f"{k} | cap-only all", k, "cap_on_all") for k in RUNS if k.startswith("S5") and "cap_on_all" in RUNS]
print(f"on-reach DOR > 0.5, n = {len(T)} (paired medians, 95 % bootstrap CI)")
for lab, x, y in pairs:
    r = med_ci(T[x] - T[y])
    out[f"{lab}: {x} - {y}"] = r
    print(f"  {lab:32s} {x:>11s} - {y:<10s} {r['median']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  up/down {r['up']}/{r['down']}")
if "cap_on_all" in RUNS:
    r = med_ci(D["cap_on_all"] - D["off"])
    out["all 458 dam gauges: cap_on_all - off"] = r
    print(f"all 458 dam gauges: cap-only (all dams) - off {r['median']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
json.dump(out, open(W + "experiments/reservoir/smoke_v2/cap_attribution.json", "w"), indent=1)
