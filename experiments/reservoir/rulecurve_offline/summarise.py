"""Summaries of fits_by_gauge.csv: per law x DOR bin x gauge set, dam vs matched control, DiD, bands, parameters.

Run: ~/projects/ddr/.venv/bin/python summarise.py  -> summary.json, summary.txt
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import rc

WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
OUT = rc.HERE
LAWS = ["L1", "L2", "L3", "L3b", "L4", "L4r", "L5", "L4ec", "L4asec", "L4lin", "L4pen", "L5ec", "L5pen"]
BINS = [-1e-9, 0.1, 0.5, 1, 2, 1e9]
BLAB = ["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]


def paired(d, rnd=4):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) == 0:
        return None
    nz = d[np.abs(d) > 1e-9]; up = int((nz > 0).sum())
    rng = np.random.default_rng(42)
    bs = np.median(rng.choice(d, (2000, len(d))), axis=1)
    return dict(n=int(len(d)), median=round(float(np.median(d)), rnd), ci=[round(float(np.percentile(bs, 2.5)), rnd),
                round(float(np.percentile(bs, 97.5)), rnd)], n_up=up, n_down=int(len(nz) - up),
                sign_p=float(f"{binomtest(up, len(nz)).pvalue:.3g}") if len(nz) else None)


def load():
    df = pd.read_csv(f"{OUT}/fits_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
    v = pd.read_csv(f"{OUT}/fits_variants.csv", dtype={"STAID": str}).set_index("STAID").drop(columns=["secs"])
    df = df.join(v)
    sg = pd.read_csv(f"{WT}/experiments/reservoir/smoke/smoke_gauges.csv", dtype={"STAID": str, "control_for": str, "huc2": str}).set_index("STAID")
    fit = pd.read_csv(f"{WT}/experiments/reservoir/smoke/expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID")
    df = df.join(sg[["role", "control_for", "nid_dor", "huc2", "dam_COMID", "dam_purpose", "area_ratio"]]).join(fit[["on_reach"]])
    df["on_reach"] = df.on_reach.astype(str) == "True"
    dam = df[df.role == "dam"].copy()
    ctl = df[df.role == "control"].copy()
    ctl = ctl.reset_index().set_index("control_for")  # indexed by the dam gauge it controls for
    ctl = ctl.loc[dam.index]
    ctl["nid_dor_dam"] = dam.nid_dor.values
    return df, dam, ctl


def block(dam, ctl, law):
    """Stats for one gauge subset (dam rows aligned with ctl rows)."""
    out = {}
    for role, g in [("dam", dam), ("control", ctl)]:
        e = dict(dnse=paired(g[f"{law}_nse"] - g.L0_nse),
                 mean_dnse_clip=round(float((g[f"{law}_nse"].clip(lower=-1) - g.L0_nse.clip(lower=-1)).mean()), 4),
                 dkge=paired(g[f"{law}_kge"] - g.L0_kge), dr=paired(g[f"{law}_r"] - g.L0_r),
                 dalpha=paired(g[f"{law}_alpha"] - g.L0_alpha), dbeta=paired(g[f"{law}_beta"] - g.L0_beta),
                 train_dnse_median=round(float((g[f"{law}_train_nse"] - g.L0_train_nse).median()), 4),
                 floor_test=dict(median=round(float(g[f"{law}_floor_test"].median()), 4), mean=round(float(g[f"{law}_floor_test"].mean()), 4),
                                 frac_gt_5pct=round(float((g[f"{law}_floor_test"] > 0.05).mean()), 3)),
                 median_nse=round(float(g[f"{law}_nse"].median()), 4))
        if f"{law}_created" in g:
            e["created_share_of_test_inflow"] = [round(float(x), 4) for x in np.nanpercentile(g[f"{law}_created"], [50, 75, 90])]
        out[role] = e
    gd = (dam[f"{law}_nse"] - dam.L0_nse).values
    gc = (ctl[f"{law}_nse"] - ctl.L0_nse).values
    out["did"] = paired(gd - gc)
    out["did_kge"] = paired((dam[f"{law}_kge"] - dam.L0_kge).values - (ctl[f"{law}_kge"] - ctl.L0_kge).values)
    md, mc = np.nanmedian(gd), np.nanmedian(gc)
    out["control_over_dam_median_gain"] = round(float(mc / md), 3) if md > 0 else None
    return out


def subsets(dam, ctl):
    bins = pd.cut(dam.nid_dor, BINS, labels=BLAB)
    yield "all", dam.index
    for b in BLAB:
        yield b, dam.index[bins == b]
    yield "DOR>0.5", dam.index[dam.nid_dor > 0.5]


def band_table(dam, ctl, laws):
    hi = dam.index[dam.nid_dor > 0.5]
    res = {}
    for law in laws:
        e = {}
        for lab in rc.BAND_LAB:
            a, b = dam.loc[hi, f"{law}_band_{lab}"], ctl.loc[hi, f"{law}_band_{lab}"]
            a0 = dam.loc[hi, f"L0_band_{lab}"]
            e[lab] = dict(dam=round(float(a.median()), 4), control=round(float(b.median()), 4),
                          dam_minus_control=round(float((a - b).median()), 4),
                          dam_change_vs_L0=round(float((a - a0).median()), 4))
        res[law] = e
    return res


def param_table(dam, ctl):
    bins = pd.cut(dam.nid_dor, BINS, labels=BLAB)
    q = lambda s: [round(float(x), 3) for x in np.nanpercentile(s, [25, 50, 75])] if s.notna().any() else None
    res = {}
    for name, g in [("dam", dam), ("control", ctl)]:
        e = {}
        for b in BLAB + ["DOR>0.5", "all"]:
            idx = dam.index[bins == b] if b in BLAB else (dam.index[dam.nid_dor > 0.5] if b == "DOR>0.5" else dam.index)
            gg = g.loc[idx]
            e[b] = dict(n=len(gg), **{f"T0_{l}": q(gg[f"{l}_p_T0"]) for l in ["L1", "L2", "L3", "L3b", "L4", "L5"]},
                        f_L3=q(gg.L3_p_f), f_L3b=q(gg.L3b_p_f), phi_L3b=q(gg.L3b_p_phi),
                        flux_amp_L3b=q(gg.L3b_flux_amp), flux_amp_L4=q(gg.L4_flux_amp),
                        frac_T0_L2_floor=round(float((gg.L2_p_T0 <= 0.0501).mean()), 3),
                        frac_T0_L4_floor=round(float((gg.L4_p_T0 <= 0.0501).mean()), 3),
                        frac_f_L3_zero=round(float((gg.L3_p_f <= 1e-6).mean()), 3),
                        T0_ratio_L3_over_L2=q(gg.L3_p_T0 / gg.L2_p_T0), T0_ratio_L4_over_L2=q(gg.L4_p_T0 / gg.L2_p_T0),
                        frac_T0_L4_gt_L2=round(float((gg.L4_p_T0 > gg.L2_p_T0 * 1.05).mean()), 3),
                        frac_T0_L4_lt_L2=round(float((gg.L4_p_T0 < gg.L2_p_T0 / 1.05).mean()), 3),
                        frac_T0_gt5d={l: round(float((gg[f"{l}_p_T0"] > 5).mean()), 3) for l in ["L1", "L2", "L3", "L3b", "L4", "L5"]})
        res[name] = e
    return res


def fmt(p):
    if p is None:
        return "-"
    return f"{p['median']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] {p['n_up']}/{p['n_down']}"


if __name__ == "__main__":
    df, dam, ctl = load()
    res = {}
    lines = []
    for setname, sel in [("on_reach", dam.on_reach), ("all_dams", dam.on_reach | True)]:
        d, c = dam[sel.values], ctl[sel.values]
        res[setname] = dict(n=len(d), laws={})
        lines.append(f"\n===== {setname}: {len(d)} dam gauges + {len(c)} matched controls =====")
        for law in LAWS:
            res[setname]["laws"][law] = {}
            lines.append(f"-- {law}")
            for sub, idx in subsets(d, c):
                b = block(d.loc[idx], c.loc[idx], law)
                res[setname]["laws"][law][sub] = b
                lines.append(f"  {sub:8s} n={len(idx):3d} dam {fmt(b['dam']['dnse'])} mean(clip) {b['dam']['mean_dnse_clip']:+.4f} "
                             f"| ctl {fmt(b['control']['dnse'])} | DiD {fmt(b['did'])} | dKGE dam {b['dam']['dkge']['median']:+.4f} "
                             f"ctl {b['control']['dkge']['median']:+.4f} | dr {b['dam']['dr']['median']:+.4f} dalpha {b['dam']['dalpha']['median']:+.4f} "
                             f"dbeta {b['dam']['dbeta']['median']:+.4f} | floor dam {b['dam']['floor_test']['mean']:.3f} | train {b['dam']['train_dnse_median']:+.4f}"
                             + (f" | created {b['dam']['created_share_of_test_inflow']}" if 'created_share_of_test_inflow' in b['dam'] else ''))
        res[setname]["params"] = param_table(d, c)
        res[setname]["bands_DOR>0.5"] = band_table(d, c, ["L0", "L1", "L2", "L3", "L3b", "L4", "L4r", "L5", "L4ec", "L4pen"])
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
