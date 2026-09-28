"""Task 4: translate smoke-set law gains to the FULL 2,365-gauge evaluation population (seed-42 no-dam arm), and
compute the runoff-rescaling null on all 2,365 gauges exactly.

Dam law sets (gains vs the no-dam flow, test WY1996-2010, offline per-dam fits):
  L2        per-dam bucket
  L4        per-dam bucket + rule curve
  SF        flood pool FA (capacity evacuation) at flood-control dams, L2 elsewhere
  SF4       FA + rule curve at flood-control dams, L4 elsewhere          (the combined law set)
Transfer: the 458 smoke dam gauges carry their own measured gain; the other 459 dammed population gauges get
  (a) the median gain of smoke dam gauges in their cell (DOR bin x purpose group), or
  (b) random draws from that cell's smoke gains (2,000 draws; mean and 95 % interval of the median change).
  (c) population_ceiling.py style: every dammed gauge gets its DOR bin's smoke median.
Engine discount: offline gains x0.5 (the in-engine replay of the rule curve kept +0.024 of the offline +0.049).
Runoff rescaling: per-HUC2 scalars fitted on training years of the 916 smoke gauges (kpool.json), applied to the
  population's test-year predictions (exact NSE); also the global scalar, the fixed 0.95, and a per-gauge scalar
  fitted on the TEST years (an in-sample oracle, an upper bound only).
Combined: population rescaled per HUC2, plus the dam law set refitted on the rescaled flow (laws_kh_by_gauge.csv).
Writes population.txt and population.json.
"""
import json

import numpy as np
import pandas as pd
import pyogrio
import zarr

import v6

rng = np.random.default_rng(1)
pop = pd.read_csv(v6.POP, dtype={"STAID": str}).set_index("STAID")
R = "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-47Z-train-and-test/" + "ev" + "al/predictions.zarr"
z = zarr.open(R, mode="r")
pid = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
PP, PO = z["predictions"][:].astype(float), z["observations"][:].astype(float)
wy = np.asarray(t.year + (t.month >= 10))
te = (wy >= 1996) & (wy <= 2010)
pidx = {s: i for i, s in enumerate(pid)}


def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    return 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()


base_chk = pd.Series({s: nse(PP[pidx[s], te], PO[pidx[s], te]) for s in pop.index})
lines = [f"population n={len(pop)}; recomputed no-dam NSE vs paired_full_run nse_off: median abs diff "
         f"{(base_chk - pop.nse_off).abs().median():.2e}, max {(base_chk - pop.nse_off).abs().max():.2e}"]
base = pop.nse_off
BASE_MED = base.median()

# HUC2 of every population gauge (GAGES-II HUC02, as select_smoke_gauges.py)
GII = pyogrio.read_dataframe("/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.shp", columns=["STAID", "HUC02"],
                             read_geometry=False)
huc = GII.set_index("STAID").HUC02.astype(str).str[:2].reindex(pop.index)
kp = json.load(open(v6.HERE + "/kpool.json"))
lines.append(f"HUC2 found for {huc.notna().sum()} of {len(pop)} gauges; missing use the global scalar {kp['kc']:.4f}")

# purpose of the largest upstream dam for every dammed population gauge
nbg = pd.read_csv(v6.WT + "/experiments/reservoir/nid/nid_dams_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
nid = pd.read_csv(v6.WT + "/experiments/reservoir/nid/nid_dams_in_eval_network.csv")
purp_by_name = nid.sort_values("storage_mcm", ascending=False).drop_duplicates("name").set_index("name").primary_purpose
sg, dam, ctl = v6.gauges()
dammed = pop.n_nid_ge10mcm > 0
ppurp = nbg.largest_dam.map(purp_by_name).reindex(pop.index)
ppurp[dam.index] = dam.dam_purpose  # smoke dam gauges: the smoke table's dam
GROUPS = {"Flood Risk Reduction": "flood", "Irrigation": "irrigation", "Water Supply": "supply", "Hydroelectric": "hydro"}
pgrp = ppurp.map(GROUPS).fillna("other")
dor = pop.nid_dor.fillna(0)
BINS = [(-1, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 1e9)]
BLAB = ["0-0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]
dbin = dor.map(lambda x: next(k for k, (lo, hi) in enumerate(BINS) if (x > lo or lo < 0) and x <= hi))
lines.append(f"dammed population gauges {int(dammed.sum())}, smoke dam gauges among them {int(dammed[dam.index].sum())}; "
             f"purpose groups of dammed: {pgrp[dammed].value_counts().to_dict()}")

# smoke gains
F = pd.read_csv(v6.HERE + "/laws_by_gauge.csv", dtype={"STAID": str}).set_index("STAID").loc[dam.index]
FK = pd.read_csv(v6.HERE + "/laws_kh_by_gauge.csv", dtype={"STAID": str}).set_index("STAID").loc[dam.index]
isfc = dam.dam_purpose == "Flood Risk Reduction"


def lawset(T, ref):
    return {"L2": T.L2_nse - T[ref], "L4": T.L4_nse - T[ref],
            "SF": np.where(isfc, T.FA_nse, T.L2_nse) - T[ref], "SF4": np.where(isfc, T.FA4_nse, T.L4_nse) - T[ref]}


G = {k: pd.Series(v, index=dam.index) for k, v in lawset(F, "L0_nse").items()}
GK = {k: pd.Series(v, index=dam.index) for k, v in lawset(FK, "Kh_nse").items()}

lines.append("\nsmoke dam-gauge gains (median), by DOR bin | by purpose group:")
sb, sp = dbin[dam.index], pgrp[dam.index]
for k, v in G.items():
    lines.append(f"  {k:4s} " + " ".join(f"{BLAB[b]}:{v[sb == b].median():+.4f}" for b in range(5)) + " | " +
                 " ".join(f"{p}:{v[sp == p].median():+.4f}" for p in ["flood", "hydro", "supply", "irrigation", "other"]))


def assemble(gains, base_s, mode, factor=1.0):
    """Return new population NSE; mode 'median' / 'draw' imputes non-smoke dammed gauges from their cell."""
    new = base_s.copy()
    new[gains.index] += factor * gains
    rest = pop.index[dammed & ~pop.index.isin(gains.index)]
    for s in rest:
        cell = gains[(sb == dbin[s]) & (sp == pgrp[s])]
        if len(cell) < 5:
            cell = gains[sb == dbin[s]]
        v = cell.median() if mode == "median" else cell.iloc[rng.integers(len(cell))]
        new[s] += factor * v
    return new


def binonly(gains, base_s):
    new = base_s.copy()
    med = [gains[sb == b].median() for b in range(5)]
    new[dammed] += np.array([med[b] for b in dbin[dammed]])
    return new


def boot_med(new, base_s, B=2000):
    a, b = new.values, base_s.values
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


res = {}


def report(name, new, base_s, extra=""):
    d = new.median() - BASE_MED
    lo, hi = boot_med(new, base_s)
    cross = int(((base_s < BASE_MED) & (new >= BASE_MED)).sum())
    res[name] = dict(median=float(new.median()), change=float(d), ci=[float(lo), float(hi)])
    lines.append(f"  {name:58s} median {new.median():.4f}  change {d:+.4f}  (paired CI of the shift vs its own base "
                 f"[{lo:+.4f},{hi:+.4f}]){extra}")


lines.append(f"\n== population median NSE, no-dam arm: {BASE_MED:.4f} (target +0.03 = {BASE_MED + 0.03:.4f})")
lines.append("-- dam law sets only (offline gains; smoke dam gauges direct, other dammed gauges imputed by DOR bin x purpose)")
for k in G:
    report(f"{k}: cell-median imputation", assemble(G[k], base, "median"), base)
    draws = [assemble(G[k], base, "draw").median() - BASE_MED for _ in range(200)]
    lines.append(f"      {k}: random-draw imputation, 200 draws: mean change {np.mean(draws):+.4f} "
                 f"[{np.percentile(draws, 2.5):+.4f},{np.percentile(draws, 97.5):+.4f}]")
    report(f"{k}: population_ceiling-style DOR-bin medians", binonly(G[k], base), base)
    report(f"{k}: cell-median, engine discount x0.5", assemble(G[k], base, "median", 0.5), base)
lines.append("-- reference: close the whole dam-minus-control gap (population_ceiling.py): +0.026")

# runoff rescaling on the full population (exact)
kh = huc.map(kp["kh"]).fillna(kp["kc"])
nse_kh = pd.Series({s: nse(kh[s] * PP[pidx[s], te], PO[pidx[s], te]) for s in pop.index})
nse_kc = pd.Series({s: nse(kp["kc"] * PP[pidx[s], te], PO[pidx[s], te]) for s in pop.index})
nse_95 = pd.Series({s: nse(0.95 * PP[pidx[s], te], PO[pidx[s], te]) for s in pop.index})


def kor(s):
    p, o = PP[pidx[s], te], PO[pidx[s], te]
    m = np.isfinite(p) & np.isfinite(o)
    k = (p[m] * o[m]).sum() / (p[m] ** 2).sum()
    return nse(k * p, o)


nse_kor = pd.Series({s: kor(s) for s in pop.index})
lines.append("-- runoff rescaling on all 2,365 gauges (exact test-year NSE)")
report("per-HUC2 scalar (fit: smoke training years)", nse_kh, base,
       f"; gauges up {int((nse_kh > base).sum())}, down {int((nse_kh < base).sum())}")
report(f"global scalar {kp['kc']:.4f} (fit: smoke training years)", nse_kc, base)
report("fixed 0.95 (tier-0 check, expected 0.7438)", nse_95, base)
report("per-gauge scalar fitted ON TEST YEARS (in-sample oracle bound)", nse_kor, base)

lines.append("-- combined: per-HUC2 rescaling on all gauges + dam law set refitted on the rescaled flow")
for k in GK:
    report(f"Kh + {k}: cell-median imputation", assemble(GK[k], nse_kh, "median"), base)
    report(f"Kh + {k}: cell-median, dam gains x0.5", assemble(GK[k], nse_kh, "median", 0.5), base)
lines.append(f"\nsmoke dam gauges, gains on the rescaled flow (median vs Kh): " +
             " ".join(f"{k} {v.median():+.4f}" for k, v in GK.items()))
lines.append("how far below the median are dammed gauges? share of dammed gauges within 0.03 below the population median: "
             f"{float(((base[dammed] < BASE_MED) & (base[dammed] >= BASE_MED - 0.03)).mean()):.3f}; undammed "
             f"{float(((base[~dammed] < BASE_MED) & (base[~dammed] >= BASE_MED - 0.03)).mean()):.3f}")
lines.append(f"gauges within 0.03 below the median: {int(((base < BASE_MED) & (base >= BASE_MED - 0.03)).sum())} "
             f"(dammed {int(((base < BASE_MED) & (base >= BASE_MED - 0.03) & dammed).sum())})")

# per-gauge scalar (training-year fit) at every gauge + the dam law set refitted on the per-gauge-rescaled flow
FG = pd.read_csv(v6.HERE + "/laws_kg_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
FA_ = pd.read_csv(v6.HERE + "/laws_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
ctl_ids = ctl.index
kg_ctl = (FG.loc[ctl_ids, "Kh_nse"] - FA_.loc[ctl_ids, "L0_nse"])
lines.append("-- per-gauge runoff scalar (training-year fit) at EVERY gauge, + dam law set refitted on that flow "
             "(smoke 916 direct; other undammed gauges draw from smoke controls, other dammed from their DOR x purpose cell)")
for k, col in [("Kg only", None), ("Kg + L2", "L2"), ("Kg + L4", "L4"), ("Kg + SF4", "SF4")]:
    if col is None:
        gd = FG.loc[dam.index, "Kh_nse"] - FA_.loc[dam.index, "L0_nse"]
    else:
        T = FG.loc[dam.index]
        v = {"L2": T.L2_nse, "L4": T.L4_nse, "SF4": np.where(isfc, T.FA4_nse, T.L4_nse)}[col]
        gd = pd.Series(np.asarray(v) - FA_.loc[dam.index, "L0_nse"].values, index=dam.index)
    meds = []
    for rep in range(201):
        new = base.copy()
        new[gd.index] += gd
        new[ctl_ids] += kg_ctl
        rest_u = pop.index[~dammed & ~pop.index.isin(ctl_ids)]
        new[rest_u] += (kg_ctl.median() if rep == 0 else kg_ctl.values[rng.integers(0, len(kg_ctl), len(rest_u))])
        for s_ in pop.index[dammed & ~pop.index.isin(gd.index)]:
            cell = gd[(sb == dbin[s_]) & (sp == pgrp[s_])]
            if len(cell) < 5:
                cell = gd[sb == dbin[s_]]
            new[s_] += cell.median() if rep == 0 else cell.iloc[rng.integers(len(cell))]
        if rep == 0:
            report(f"{k}: median imputation", new, base)
        else:
            meds.append(new.median() - BASE_MED)
    lines.append(f"      {k}: random-draw imputation, 200 draws: mean change {np.mean(meds):+.4f} "
                 f"[{np.percentile(meds, 2.5):+.4f},{np.percentile(meds, 97.5):+.4f}]")
open(v6.HERE + "/population.txt", "w").write("\n".join(lines) + "\n")
json.dump(res, open(v6.HERE + "/population.json", "w"), indent=1)
print("\n".join(lines))
