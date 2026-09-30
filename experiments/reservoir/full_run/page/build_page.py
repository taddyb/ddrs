#!/usr/bin/env python
"""Fill page_src.html from analysis.py's outputs and the eval-plot images; write the publish copy and a preview folder.

    build_page.py [--rep]      (--rep adds the seed-replicate section from build/replicate.json)

Reads  output/reservoir_full_run/web/build/{results_s42.json,gauges_s42.csv,dams_s42.csv}, publish/img/manifest.json,
       the runs' plots/parameter_*_stats.json, picks.json, ../../smoke/page/map.json
Writes output/reservoir_full_run/web/publish/learned_dam_release.html and web/site/ (preview: index.html + links).
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
WEB = HERE.parents[3] / "output/reservoir_full_run/web"
PUB = WEB / "publish"
RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
ap = argparse.ArgumentParser()
ap.add_argument("--rep", action="store_true")
args = ap.parse_args()

R = json.load(open(WEB / "build/results_s42.json"))
G = pd.read_csv(WEB / "build/gauges_s42.csv", dtype={"STAID": str, "huc2": str}).set_index("STAID")
DM = pd.read_csv(WEB / "build/dams_s42.csv")
IMG = json.load(open(PUB / "img/manifest.json"))
PK = json.loads((HERE / "picks.json").read_text())
OFF, DAM = R["runs"]["off"], R["runs"]["learned"]
PST = {a: json.load(open(RUNS / r / "plots/parameter_convergence_stats.json")) for a, r in (("off", OFF), ("dam", DAM))}
PD = json.load(open(RUNS / DAM / "plots/parameter_delta_stats.json"))
REPN = json.load(open(WEB / "build/replicate_numbers.json")) if args.rep else None
R43 = json.load(open(WEB / "build/results_s43.json")) if args.rep else None


def f3(v):
    return f"{v:.3f}"


def sg(v, d=4):
    if abs(v) < 0.5 * 10 ** -d:
        return f"{0:.{d}f}"
    return ("+" if v >= 0 else "−") + f"{abs(v):.{d}f}"


def ci(p, d=4):
    return f"{sg(p['ci'][0], d)} to {sg(p['ci'][1], d)}"


def pv(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def c(n):
    return f"{n:,}"


def r2(v):
    return f"{v:.2f}".replace("-", "−")


PA = R["paired"]
dmo, omb, dmb = PA["dam_minus_off"], PA["off_minus_base"], PA["dam_minus_base"]
M = R["median"]
N = R["group_n"]
G["dnse"] = G.nse_dam - G.nse_off
dam_g, und_g = G[G.dammed], G[~G.dammed]

# ---------- numbers used in the prose ----------
p_dam, p_und, p_all, p_on, p_up = (dmo["nse"][k] for k in ("dammed", "undammed", "all", "on_reach", "further_up"))
k_dam, k_und, k_on = (dmo["kge"][k] for k in ("dammed", "undammed", "on_reach"))
dor = {k: dmo["nse"]["dor_" + k] for k in ("<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2")}
huc = R["huc2"]
reg_rows = [(h, v) for h, v in huc.items() if "dammed" in v and v["dammed"]["n"] >= 5 and "undammed" in v]
reg_ahead = sum(v["dammed"]["median"] > v["undammed"]["median"] for _, v in reg_rows)
reg_dam_pos = sum(v["dammed"]["median"] > 0 for _, v in reg_rows)
reg_und_neg = sum(v["undammed"]["median"] < 0 for h, v in huc.items() if "undammed" in v and v["undammed"]["n"] >= 5)
n_und_regions = sum(1 for h, v in huc.items() if "undammed" in v and v["undammed"]["n"] >= 5)
big_gain = dam_g[dam_g.dnse > 0.02]
big_loss_d = dam_g[dam_g.dnse < -0.02]
big_loss_u = und_g[und_g.dnse < -0.02]
big_gain_u = und_g[und_g.dnse > 0.02]
HUCN = {"01": "New England", "02": "Mid-Atlantic", "03": "South Atlantic-Gulf", "04": "Great Lakes", "05": "Ohio", "06": "Tennessee",
        "07": "Upper Mississippi", "08": "Lower Mississippi", "09": "Souris-Red-Rainy", "10": "Missouri", "11": "Arkansas-White-Red",
        "12": "Texas-Gulf", "13": "Rio Grande", "14": "Upper Colorado", "15": "Lower Colorado", "16": "Great Basin", "17": "Pacific Northwest",
        "18": "California"}
top_gain_regions = big_gain.huc2.value_counts().head(4)
vsb = (G.nse_dam - G.nse_base).groupby(G.huc2).median().to_dict()
top_dam_reg = sorted([(v["dammed"]["median"], h) for h, v in reg_rows], reverse=True)[:3]
neg_dam_reg = [h for h, v in reg_rows if v["dammed"]["median"] < 0]
onr = G[G.on_reach]
fr_on_up10 = (onr.dnse > 0.01).mean()
fr_on_dn10 = (onr.dnse < -0.01).mean()
fr_und_small = (und_g.dnse.abs() < 0.005).mean()
fr_und_dn10 = (und_g.dnse < -0.01).mean()
rel = R["release"]
T0q = rel["T0"]
yr = DM.year
n_post80 = int((yr > 1981).sum())
n_post95 = int((yr > 1995).sum())
ps_off, ps_dam = PST["off"], PST["dam"]
g_off = ps_off["gamma"]["median by epoch"]
g_dam = ps_dam["gamma"]["median by epoch"]
n_off = ps_off["n"]["median by epoch"]
n_dam = ps_dam["n"]["median by epoch"]


def pnc(run, k):
    import xarray as xr
    return xr.open_dataset(RUNS / run / "plot/kan_parameters.nc")[k].values


gam_off, gam_dam, n0_off, n0_dam = pnc(OFF, "gamma"), pnc(DAM, "gamma"), pnc(OFF, "n"), pnc(DAM, "n")

# picks: NSE within their default water year, straight from the zarr stores
z_off = zarr.open(str(RUNS / OFF / "eval/predictions.zarr"), mode="r")
z_dam = zarr.open(str(RUNS / DAM / "eval/predictions.zarr"), mode="r")
ids = [bytes(r).decode().strip("\x00") for r in z_off["gage_ids"][:]]
tt = pd.to_datetime(z_off["time"][:].astype("datetime64[ns]"))


def wy_nse(gid, wy):
    i = ids.index(gid)
    m = (tt >= f"{wy-1}-10-01") & (tt <= f"{wy}-09-30")
    o = z_off["observations"][i][m]
    out = {}
    for nm, z in (("off", z_off), ("dam", z_dam)):
        p = z["predictions"][i][m]
        k = np.isfinite(o) & np.isfinite(p)
        out[nm] = 1 - ((p[k] - o[k]) ** 2).sum() / ((o[k] - o[k].mean()) ** 2).sum()
        out[nm + "_peak"] = float(np.nanmax(p))
    out["obs_peak"] = float(np.nanmax(o))
    bm = json.load(open(RUNS / OFF / "baseline/manifest.json"))
    B = np.memmap(RUNS / OFF / "baseline/predictions.f32", dtype=np.float32, mode="r", shape=(bm["n_gauges"], bm["n_days"]))
    bi = [str(x) for x in bm["gage_ids"]].index(gid)
    bt = pd.date_range(str(bm["time_range_daily"][0])[:10], periods=bm["n_days"], freq="D")
    bmask = (bt >= f"{wy-1}-10-01") & (bt <= f"{wy}-09-30")
    out["base_peak"] = float(np.nanmax(B[bi][bmask]))
    return out


pk_stats = {g["id"]: wy_nse(g["id"], g["wy"]) for g in PK["gauges"]}
neuse, pact, orov, white = (pk_stats[g["id"]] for g in PK["gauges"])
gN, gP, gO, gW = (G.loc[g["id"]] for g in PK["gauges"])

# ---------- text ----------
V = {}
V["SEED"] = "42"
V["N_ALL"] = c(R["n"])
V["N_DAMMED"] = c(N["dammed"])
V["N_UNDAMMED"] = c(N["undammed"])
V["N_ONREACH"] = c(N["on_reach"])
V["N_DAMS"] = c(rel["n"])
V["N_DAYS"] = c(R["n_days"])
V["REACHES"] = "346,321"
V["WINDOW"] = f"{R['window'][0]} to {R['window'][1]}"
V["RUN_OFF"], V["RUN_DAM"] = OFF, DAM
V["RUNS43"] = ""
V["LEDE"] = (f"Every large dam in the network ({rel['n']:,} reaches holding NID dams of 10 MCM or more) is routed as a storage bucket whose response "
             "time is learned from the dam's NID record, jointly with channel roughness, using gauge observations only as the training target. "
             f"Trained on water years 1982-1995 and scored on the 15 water years 1996-2010 at {R['n']:,} USGS gauges, against the same model "
             "without dams and against summed Q′, the unrouted sum of upstream runoff.")
V["T_DAM"] = sg(p_dam["median"])
V["T_DAM_NOTE"] = f"95 % interval {ci(p_dam)}; {p_dam['n_up']} gauges up, {p_dam['n_down']} down ({pv(p_dam['sign_p'])})"
if args.rep:
    e = REPN["effect"]
    V["T_DAM_NOTE"] += f". Seed 43: {sg(e['dammed']['s43']['median'])} ({ci(e['dammed']['s43'])})."
V["T_UND"] = sg(p_und["median"])
V["T_UND_NOTE"] = f"95 % interval {ci(p_und)}; {p_und['n_up']} up, {p_und['n_down']} down ({pv(p_und['sign_p'])})"
if args.rep:
    V["T_UND_NOTE"] += (f". Seed 43: {sg(e['undammed']['s43']['median'])}; two no-dam runs differing only in seed differ by "
                        f"{sg(e['undammed']['noise_off']['median'])} here.")
V["T_ALL"] = f"{f3(M['all']['nse_off'])} → {f3(M['all']['nse_dam'])}"
V["T_ALL_NOTE"] = (f"no dam → dam release; summed Q′ {f3(M['all']['nse_base'])}. KGE {f3(M['all']['kge_off'])} → {f3(M['all']['kge_dam'])} "
                   f"(summed Q′ {f3(M['all']['kge_base'])}).")
if args.rep:
    m43 = REPN["median_nse"]["s43"]
    V["T_ALL_NOTE"] = (f"seed 42, no dam → dam release (seed 43: {f3(m43['nse_off'])} → {f3(m43['nse_dam'])}); summed Q′ {f3(M['all']['nse_base'])}. "
                       f"KGE {f3(M['all']['kge_off'])} → {f3(M['all']['kge_dam'])} (summed Q′ {f3(M['all']['kge_base'])}).")
claims = [
    f"<b>Routing beats summed Q′ in the test years.</b> Median per-gauge gain in NSE {sg(omb['nse']['all']['median'], 3)} without dams and "
    f"{sg(dmb['nse']['all']['median'], 3)} with them; {c(dmb['nse']['all']['n_up'])} of {c(R['n'])} gauges improve. <a href=\"#f1\">Figure 1</a>.",
    f"<b>The dam release adds a small, consistent gain below large dams.</b> {sg(p_dam['median'])} NSE at the {c(N['dammed'])} dammed gauges "
    f"({p_dam['n_up']} up, {p_dam['n_down']} down), {sg(p_on['median'])} where the dam sits on the gauge reach, largest ({sg(dor['1-2']['median'])}) "
    f"where reservoirs hold one to two years of flow. <a href=\"#f2\">Figures 2</a>, <a href=\"#f3\">3</a>, <a href=\"#f6\">6</a>.",
    f"<b>It costs undammed gauges a little, and KGE does not improve.</b> {sg(p_und['median'])} NSE at undammed gauges ({p_und['n_down']} of "
    f"{c(N['undammed'])} down); KGE falls by {sg(k_dam['median'])} at dammed gauges. The population median NSE does not rise "
    f"({f3(M['all']['nse_off'])} → {f3(M['all']['nse_dam'])}). <a href=\"#f2\">Figure 2</a>.",
    f"<b>The release head learned physically ordered response times from NID features alone.</b> T₀ rises with storage (Spearman ρ "
    f"{rel['rho_storage']['rho']:.2f}) and with residence time (ρ {rel['rho_residence']['rho']:.2f}), and flood-control dams hold longer "
    f"(median {rel['T0_flood']:.2f} d against {rel['T0_notflood']:.2f} d). <a href=\"#f7\">Figures 7</a>, <a href=\"#f8\">8</a>.",
    f"<b>Caveat: the two arms also learned different channel roughness.</b> Median γ is {np.median(gam_off):.3f} without dams and "
    f"{np.median(gam_dam):.3f} with them, and γ was still moving at epoch 50 in both, so part of every per-gauge difference is a different channel "
    f"solution rather than the dams. One seed per arm; a second seed is the test. <a href=\"#f9\">Figure 9</a>, <a href=\"#replicate\">replicate</a>.",
]
if args.rep:
    e, pr, rr = REPN["effect"], REPN["params"], REPN["release"]
    claims[1] = (f"<b>The dam release adds a small gain below large dams, and it reproduces.</b> {sg(p_dam['median'])} NSE at the {c(N['dammed'])} "
                 f"dammed gauges in seed 42 and {sg(e['dammed']['s43']['median'])} in seed 43, both intervals clear of zero; largest where the dam sits on "
                 f"the gauge reach and where reservoirs hold 0.5 to 2 years of flow. Which gauges gain repeats across seeds (rank correlation "
                 f"{r2(e['dammed']['rho'])}, {r2(e['on_reach']['rho'])} on the gauge reach). <a href=\"#f2\">Figures 2</a>, <a href=\"#f6\">6</a>, "
                 f"<a href=\"#replicate\">replicate</a>.")
    claims[2] = (f"<b>Undammed gauges lose a little in both seeds, at the noise floor, and KGE does not improve.</b> {sg(p_und['median'])} and "
                 f"{sg(e['undammed']['s43']['median'])} NSE, about the size of the difference between two no-dam runs that differ only in seed "
                 f"({sg(e['undammed']['noise_off']['median'])}); which undammed gauges lose does not repeat (ρ {r2(e['undammed']['rho'])}). KGE falls "
                 f"at dammed gauges in both seeds ({sg(e['dammed']['kge_s42']['median'])}, {sg(e['dammed']['kge_s43']['median'])}). "
                 f"<a href=\"#replicate\">Replicate</a>.")
    claims[3] = (f"<b>The release head learned physically ordered response times from NID features alone, and the order reproduces.</b> T₀ rises "
                 f"with storage (ρ {rel['rho_storage']['rho']:.2f}) and residence time (ρ {rel['rho_residence']['rho']:.2f}); flood-control dams hold "
                 f"longer; per-dam T₀ correlates {rr['rho_T0']:.2f} between seeds. The seasonal phase does not reproduce. "
                 f"<a href=\"#f7\">Figures 7</a>, <a href=\"#f8\">8</a>.")
    claims[4] = (f"<b>Caveat: the channel parameter γ is not identified.</b> Median γ is {pr['off42']['gamma']['median']:.3f} / "
                 f"{pr['dam42']['gamma']['median']:.3f} (no dam / dam release) in seed 42 and {pr['off43']['gamma']['median']:.3f} / "
                 f"{pr['dam43']['gamma']['median']:.3f} in seed 43: the ordering between arms reverses with the seed, so it is not caused by the dams, "
                 f"and the dammed-gauge gain survives both. <a href=\"#f9\">Figure 9</a>, <a href=\"#replicate\">replicate</a>.")
V["CLAIMS"] = "".join(f'<div class="claim"><p>{t}</p></div>' for t in claims)

# score table
rows = []
for k, lab, sub in [("all", "All gauges", False), ("dammed", "Dammed", False), ("on_reach", "dam on the gauge reach", True),
                    ("further_up", "dam further upstream", True), ("undammed", "Undammed", False)]:
    m = M[k]
    for met in ("nse", "kge"):
        cls = ' class="sub"' if sub else ""
        rows.append(f"<tr><td{cls}>{lab if sub else lab}, median {met.upper()}</td><td class=\"num\">{c(N[k])}</td>"
                    f"<td class=\"num\">{f3(m[met + '_base'])}</td><td class=\"num\">{f3(m[met + '_off'])}</td><td class=\"num\">{f3(m[met + '_dam'])}</td></tr>")
V["SCORE_ROWS"] = "".join(rows)


def ptable(defs, cols):
    h = "".join(f'<th class="num">{x}</th>' for x in [c_[0] for c_ in cols])
    body = []
    for key, lab in defs:
        tds = "".join(f'<td class="num">{fn(key)}</td>' for _, fn in cols)
        body.append(f"<tr><td>{lab}</td>{tds}</tr>")
    return f'<table class="stats"><thead><tr><th>Gauges</th>{h}</tr></thead><tbody>{"".join(body)}</tbody></table>'


V["F1_RESULT"] = (f"Against summed Q′, the no-dam model raises test NSE at {c(omb['nse']['all']['n_up'])} of {c(R['n'])} gauges (median "
                  f"{sg(omb['nse']['all']['median'], 3)}), and the dam release at {c(dmb['nse']['all']['n_up'])} (median {sg(dmb['nse']['all']['median'], 3)}). "
                  f"The gain over Q′ is largest at dammed gauges: {sg(omb['nse']['dammed']['median'], 3)} without the release and "
                  f"{sg(dmb['nse']['dammed']['median'], 3)} with it. At undammed gauges it is {sg(omb['nse']['undammed']['median'], 3)} and "
                  f"{sg(dmb['nse']['undammed']['median'], 3)}. Only above a regulation of 2 is neither arm clearly better than Q′. "
                  f"KGE gains are about half as large: {sg(omb['kge']['all']['median'], 3)} and {sg(dmb['kge']['all']['median'], 3)}.")
V["F1_TABLE"] = ptable([("all", "All"), ("dammed", "Dammed"), ("undammed", "Undammed"), ("dor_>2", "Regulation above 2")],
                       [("No dam − Q′", lambda k: sg(omb["nse"][k]["median"], 3)), ("Dam release − Q′", lambda k: sg(dmb["nse"][k]["median"], 3)),
                        ("95 % interval", lambda k: ci(dmb["nse"][k], 3)), ("Up / down", lambda k: f"{dmb['nse'][k]['n_up']} / {dmb['nse'][k]['n_down']}")])
V["F1_CONCLUSION"] = ("Learned routing earns its keep on years it never saw. The gain over Q′ is larger at dammed gauges, which are mostly larger "
                      "rivers where travel time matters, and it vanishes where reservoirs hold more than two years of flow, which no channel "
                      "router can represent. This is a comparison with an unrouted sum, so it shows the routing is useful, not that it is right: "
                      "Q′ comes from the same runoff model in both arms, and both share its errors.")
V["F2_RESULT"] = (f"At the {c(N['dammed'])} dammed gauges the release raises test NSE by a median {sg(p_dam['median'])} ({ci(p_dam)}), "
                  f"{p_dam['n_up']} up and {p_dam['n_down']} down ({pv(p_dam['sign_p'])}). The gain is {sg(p_on['median'])} where the dam is on the "
                  f"gauge reach ({p_on['n_up']} / {p_on['n_down']}) and {sg(p_up['median'])} where it is further up ({ci(p_up)}). By regulation: "
                  + ", ".join(f"{sg(dor[k]['median'])} at {lab}" for k, lab in [("<=0.1", "0.1 or less"), ("0.1-0.5", "0.1 to 0.5"), ("0.5-1", "0.5 to 1"),
                                                                                 ("1-2", "1 to 2"), (">2", "above 2")])
                  + f". Undammed gauges lose {sg(p_und['median'])} ({ci(p_und)}), {p_und['n_down']} of {c(N['undammed'])} down. "
                  f"KGE falls slightly in almost every group: {sg(k_dam['median'])} at dammed gauges, {sg(k_und['median'])} at undammed.")
V["F2_TABLE"] = ptable([("dammed", "Dammed"), ("on_reach", "Dam on the gauge reach"), ("dor_1-2", "Regulation 1 to 2"), ("undammed", "Undammed")],
                       [("NSE change", lambda k: sg(dmo["nse"][k]["median"])), ("95 % interval", lambda k: ci(dmo["nse"][k])),
                        ("Up / down", lambda k: f"{dmo['nse'][k]['n_up']} / {dmo['nse'][k]['n_down']}"), ("KGE change", lambda k: sg(dmo["kge"][k]["median"]))])
V["F2_CONCLUSION"] = ("The ordering is what a dam effect predicts: nothing where regulation is negligible, gains rising with regulation and "
                      "strongest when the dam is on the gauge reach, and a loss at undammed gauges, where the release cannot act directly. The sizes "
                      "are thousandths of NSE, the population median does not move, and KGE does not follow. The undammed loss must come from the "
                      "channel parameters the dam arm learned (Figure 9); whether it is a systematic cost of training with dams or trajectory noise "
                      "is what the seed replicate decides.")
qo, qu = onr.dnse.quantile([.1, .25, .5, .75, .9]).tolist(), und_g.dnse.quantile([.1, .25, .5, .75, .9]).tolist()
V["F3_RESULT"] = (f"The dammed distributions are wider than the undammed one in both directions. Where the dam is on the gauge reach the 10th, "
                  f"25th, 50th, 75th and 90th percentiles are {', '.join(sg(v, 3) for v in qo)}; at undammed gauges {', '.join(sg(v, 3) for v in qu)}. "
                  f"So from the lower quartile up the on-reach curve sits right of the undammed one, and below it, left: {fr_on_up10:.0%} of on-reach "
                  f"gauges gain more than 0.01 and {fr_on_dn10:.0%} lose more than 0.01. At undammed gauges {fr_und_small:.0%} change by less than "
                  f"0.005 either way. In all, {len(big_gain)} dammed gauges gain more than 0.02 and {len(big_loss_d)} lose more than 0.02.")
V["F3_CONCLUSION"] = ("The release does not shift every dammed gauge a little; it moves a minority of hydrographs a lot, mostly for the better "
                      "(about three gains above 0.02 for every loss that large), and leaves the typical gauge nearly unchanged. The undammed "
                      "curve is narrow and centred just below zero, which is the signature of a small network-wide change.")
V["F4_RESULT"] = (f"Large gains (above +0.02) at dammed gauges cluster in "
                  + ", ".join(f"{HUCN[h]} ({n})" for h, n in top_gain_regions.items())
                  + f": {len(big_gain)} gauges in all. Large losses (below −0.02) hit {len(big_loss_d)} dammed and {len(big_loss_u)} undammed gauges and "
                  f"are scattered. Against summed Q′ the map is mostly ochre; California is the one region where the dam release trails Q′ "
                  f"(regional median {sg(vsb['18'], 3)}), and the Great Basin, Rio Grande and Pacific Northwest sit near zero "
                  f"({sg(vsb['16'], 3)}, {sg(vsb['13'], 3)}, {sg(vsb['17'], 3)}).")
V["F4_CONCLUSION"] = ("Gains follow the big flood-control and multipurpose reservoir systems. Losses have no regional home, which is what trajectory "
                      "noise in the routing parameters would look like, but a map cannot separate noise from a small systematic cost.")
V["F5_RESULT"] = (f"In {reg_ahead} of the {len(reg_rows)} regions with at least five dammed gauges, the dammed median is right of the undammed median, "
                  f"and the dammed median is positive in {reg_dam_pos} of them. The undammed median is negative in {reg_und_neg} of the "
                  f"{n_und_regions} regions with at least five undammed gauges. The largest dammed medians are "
                  + ", ".join(f"{HUCN[h]} {sg(v, 3)} ({huc[h]['dammed']['n']})" for v, h in top_dam_reg)
                  + "; dammed gauges lose in " + ", ".join(f"{HUCN[h]} ({huc[h]['dammed']['n']})" for h in neg_dam_reg) + ".")
V["F5_CONCLUSION"] = ("The national result is not carried by one region. The small loss at undammed gauges is just as widespread, which points to "
                      "a network-wide change in the routing head rather than a local artefact.")
V["F6_RESULT"] = ("Band medians rise from " + sg(dor["<=0.1"]["median"]) + " below 0.1 to " + sg(dor["1-2"]["median"]) + " at 1 to 2, then fall back to "
                  + sg(dor[">2"]["median"]) + f" above 2, where the interval ({ci(dor['>2'])}) spans zero. Every band from 0.1 to 2 has its interval "
                  "clear of zero.")
V["F6_TABLE"] = ptable([("dor_<=0.1", "up to 0.1"), ("dor_0.1-0.5", "0.1 to 0.5"), ("dor_0.5-1", "0.5 to 1"), ("dor_1-2", "1 to 2"), ("dor_>2", "above 2")],
                       [("n", lambda k: str(dmo["nse"][k]["n"])), ("NSE change", lambda k: sg(dmo["nse"][k]["median"])),
                        ("95 % interval", lambda k: ci(dmo["nse"][k])), ("Up / down", lambda k: f"{dmo['nse'][k]['n_up']} / {dmo['nse'][k]['n_down']}")])
V["F6_CONCLUSION"] = ("The dose-response shape is the strongest single piece of evidence that the gain is a dam effect, and it matches the offline "
                      "fit done before this model existed (smoke test: +0.005, +0.007, +0.011, +0.024, 0.000 across the same bands), at about a fifth "
                      "of its size. Reservoirs holding more than two years of flow are run on schedules a seasonal bucket cannot mimic.")
V["F7_RESULT"] = (f"Median T₀ is {T0q[2]*24:.1f} hours (middle half {T0q[1]*24:.1f} h to {T0q[3]:.2f} d); {int((DM.T0_days > 3).sum())} dams exceed 3 days, "
                  f"the longest {rel['T0_max']:.1f} d. The longest response times belong to the largest flood-control reservoirs: "
                  + ", ".join(f"{r.largest_name} {r.T0_days:.0f} d" for r in DM.sort_values("T0_days", ascending=False).head(5).itertuples())
                  + f". The seasonal swing is modest: T max ÷ T min has median {rel['ratio'][2]:.2f} and "
                  f"exceeds 2 at {int((DM.T_max_days / DM.T_min_days > 2).sum())} dams.")
V["F7_CONCLUSION"] = ("The head differentiates dams by size and purpose rather than giving every dam the same bucket. Response times of hours "
                      "to days mean the release acts as attenuation of flood peaks, not as seasonal storage.")
pm = rel["peak_month_counts"]
V["F8_RESULT"] = (f"Learned T₀ rises with residence time (Spearman ρ = {rel['rho_residence']['rho']:.2f}, {rel['rho_residence']['n']} dams on a gauge "
                  f"reach) and with the offline fitted T₀ (ρ = {rel['rho_fit']['rho']:.2f}, {rel['rho_fit']['n']} dams), but it is far shorter than "
                  f"either: {rel['T0_median_onreach']:.2f} d against a median residence time of {rel['res_days_median']:.0f} d. Flood-control dams "
                  f"({rel['n_flood']}) have median T₀ {rel['T0_flood']:.2f} d, the others {rel['T0_notflood']:.2f} d. T is longest from August to "
                  f"February at every dam ({pm[8]} peak in September) and never between March and July.")
V["F8_CONCLUSION_S42"] = ("The ordering is physical, the magnitudes are not residence times. A linear bucket fitted to daily gauges learns the "
                      "attenuation a reservoir applies to floods, which is much shorter than the time water spends in it. The seasonal phase, with "
                      "the shortest T in spring and early summer, is learned from the gauges, not imposed, and has not been checked against "
                      "operating rules.")
V["F9_TABLE"] = ptable([("n", "n₀"), ("gamma", "γ")],
                       [("Median, no dam", lambda k: f"{np.median(n0_off if k == 'n' else gam_off):.4f}"),
                        ("Median, dam release", lambda k: f"{np.median(n0_dam if k == 'n' else gam_dam):.4f}"),
                        ("median |Δ| ÷ IQR", lambda k: f"{PD[k]['median_abs_delta_over_iqr']:.2f}"),
                        ("Moved in last 25 epochs, no dam / dam", lambda k: f"{PST['off'][k]['median |e50-e25| (% of range)']:.1f} % / {PST['dam'][k]['median |e50-e25| (% of range)']:.1f} %")])
V["F9_RESULT"] = (f"n₀ is similar in the two arms (median {np.median(n0_off):.4f} and {np.median(n0_dam):.4f}; median |Δ| is {PD['n']['median_abs_delta_over_iqr']:.2f} "
                  f"of its interquartile range), with regional differences of a few hundredths that do not follow the dams. γ is not: median "
                  f"{np.median(gam_off):.3f} without dams and {np.median(gam_dam):.3f} with them, higher in the dam arm at {PD['gamma']['frac_increase']:.0%} "
                  f"of reaches (median |Δ| {PD['gamma']['median_abs_delta_over_iqr']:.2f} of its IQR). γ was still falling at epoch 50 in both arms "
                  f"(no dam: {g_off['1']:.3f}, {g_off['10']:.3f}, {g_off['25']:.3f}, {g_off['50']:.3f} at epochs 1, 10, 25, 50; dam release: "
                  f"{g_dam['1']:.3f}, {g_dam['10']:.3f}, {g_dam['25']:.3f}, {g_dam['50']:.3f}), moving {PST['off']['gamma']['median |e50-e25| (% of range)']:.0f} % and "
                  f"{PST['dam']['gamma']['median |e50-e25| (% of range)']:.0f} % of its range over the last 25 epochs. n₀ had mostly settled "
                  f"({PST['off']['n']['median |e50-e25| (% of range)']:.1f} % and {PST['dam']['n']['median |e50-e25| (% of range)']:.1f} %).")
V["F9_CONCLUSION_S42"] = ("The dams do not only add buckets: they change the path the routing head takes, and γ in particular is neither converged nor "
                      "the same in the two arms. The per-gauge differences in Figures 2 to 6 therefore mix the release with a different stage "
                      "roughness everywhere. The dose-response in Figure 6 argues that the dammed-gauge gain is mostly the release; the undammed "
                      "loss is most plausibly the γ difference. Longer training, or a replicate seed, is needed before either is final.")
if args.rep:
    rr = REPN["release"]
    pm43 = rr["s43"]["peak_month_counts"]
    V["F8_RESULT"] += (f" Seed 43 disagrees on the phase: there T is longest from March to September ({pm43[7]} dams peak in August), "
                       f"the opposite half of the year, while its T₀ ranks the dams almost the same way (ρ {rr['rho_T0']:.2f} with seed 42).")
    V["F8_CONCLUSION"] = ("The ordering of T₀ is physical and reproducible; the magnitudes are not residence times. A linear bucket fitted to daily "
                          "gauges learns the attenuation a reservoir applies to floods, which is much shorter than the time water spends in it. "
                          "The seasonal phase is not identified: two seeds put the longest T in opposite halves of the year, so the seasonal "
                          "terms a and b should not be read as learned operating rules.")
    pr, pdf = REPN["params"], REPN["param_diff"]
    V["F9_CONCLUSION"] = (f"The second seed settles what the γ difference means. In seed 43 the order reverses: γ is {pr['off43']['gamma']['median']:.3f} "
                          f"without dams and {pr['dam43']['gamma']['median']:.3f} with them, and two no-dam runs that differ only in seed differ by a "
                          f"median {pdf['off43-off42']['gamma']['median']:.2f} in γ ({pdf['off43-off42']['gamma']['median_abs_over_iqr']:.1f} times its "
                          "interquartile range). γ is not identified by daily NSE on this network and wanders with the trajectory; the arm "
                          "difference in γ is noise, not a dam effect. The dammed-gauge gain holds in both seeds despite opposite γ shifts, which "
                          "is further evidence that it comes from the release itself.")
else:
    V["F8_CONCLUSION"] = V["F8_CONCLUSION_S42"]
    V["F9_CONCLUSION"] = V["F9_CONCLUSION_S42"]
V["F10_RESULT"] = (f"Neuse River near Falls, WY1996: the no-dam model sends Hurricane Fran's {neuse['off_peak']:,.0f} m³/s peak straight through "
                   f"(unrouted inflow {neuse['base_peak']:,.0f}); Falls Lake held it and the gauge below the dam saw {neuse['obs_peak']:,.0f}. The release cuts the peak to {neuse['dam_peak']:,.0f} and stretches the recession: "
                   f"NSE within the year {neuse['off']:.2f} → {neuse['dam']:.2f} (test period {f3(gN.nse_off)} → {f3(gN.nse_dam)}). Rapid Creek below Pactola, the "
                   f"typical dammed gauge, changes little ({f3(gP.nse_off)} → {f3(gP.nse_dam)}). Feather River at Oroville, WY1997: the gauge below Oroville "
                   f"Dam recorded the January flood at {orov['obs_peak']:,.0f} m³/s, more than the unrouted inflow ({orov['base_peak']:,.0f}), so the reservoir did "
                   f"not attenuate it. The no-dam model comes close ({orov['off_peak']:,.0f}); the release, with a learned T₀ of 16 days at Oroville, cuts it to "
                   f"{orov['dam_peak']:,.0f}, and NSE within the year falls {orov['off']:.2f} → {orov['dam']:.2f}. White Lick Creek, undammed: the "
                   f"two arms overlap ({f3(gW.nse_off)} → {f3(gW.nse_dam)}).")
V["F10_CONCLUSION"] = ("Where the release helps, it behaves like a flood-control reservoir. Its largest failure is the opposite case: a full "
                       "reservoir that passes a flood cannot be represented by a bucket whose response time depends only on the season. "
                       "A storage-dependent release (spill when full) is the obvious next term.")
V["LIMITS"] = "".join(f"<li>{t}</li>" for t in [
    f"<b>Effect sizes are thousandths of NSE.</b> The dammed-gauge gain ({sg(p_dam['median'])}) is statistically clear but small, the population "
    f"median NSE falls by {M['all']['nse_off'] - M['all']['nse_dam']:.4f}, and KGE falls slightly in almost every group. Nothing here shows the release "
    "improves CONUS-wide skill.",
    (f"<b>The undammed loss.</b> {sg(p_und['median'])} (seed 42) and {sg(REPN['effect']['undammed']['s43']['median'])} (seed 43). The buckets cannot act on "
     "gauges without a large dam upstream, so it comes through the routing head. Its size equals the difference between two no-dam seeds, and which "
     "gauges lose does not repeat; it is either chance or a cost too small for two seeds to resolve." if args.rep else
     f"<b>The undammed loss is real in this pair.</b> {sg(p_und['median'])} with an interval clear of zero and {p_und['n_down']} of {c(N['undammed'])} gauges "
     "down. It is most plausibly the different γ the dam arm learned (Figure 9), not the buckets, which cannot act on gauges without a dam upstream."),
    ("<b>Two seeds per arm.</b> The dammed-gauge gain and its per-gauge pattern reproduce; the undammed loss is at the size of seed-to-seed noise, and "
     "two seeds cannot settle a sign. γ had not converged in any run and depends on the seed." if args.rep else
     "<b>One seed per arm.</b> Each arm is one training trajectory; γ had not converged in either. Until the replicate is in, arm differences cannot be "
     "split into the release's effect and trajectory noise."),
    f"<b>The test is a hold-out in time only.</b> The same {c(R['n'])} gauges train and test, so the result says the model generalises to new years at "
    "known gauges, not to ungauged basins. Nothing was tuned on the test years, and the final checkpoint was used.",
    f"<b>The dam table ignores completion dates.</b> Every one of the {rel['n']:,} dam reaches is routed in every year; by the NID year of the largest dam "
    f"(which can be a later modification), {n_post80} were completed after 1981 and {n_post95} after 1995, and {int(yr.isna().sum())} have no year, so some "
    "buckets act on years before their dam existed.",
    "<b>Daily scores of an hourly model.</b> Sub-daily timing is not scored, and a response time of a few hours is at the edge of what daily "
    "observations can resolve.",
    "<b>Degree of regulation uses NID storage and observed mean flow;</b> it is a descriptor of the gauge, not an input to the model. The dammed / "
    "undammed split uses the 10 MCM threshold; undammed gauges can have smaller dams.",
])

if args.rep:
    e, ab, pr, pdf, rr = REPN["effect"], REPN["abs"], REPN["params"], REPN["param_diff"], REPN["release"]
    m42, m43 = REPN["median_nse"]["s42"], REPN["median_nse"]["s43"]
    V["REP_TITLE"] = "A second seed: what holds and what does not"
    rows = []
    for k, lab in [("all", "All gauges"), ("dammed", "Dammed"), ("on_reach", "Dam on the gauge reach"), ("further_up", "Dam further up"),
                   ("dor_<=0.1", "Regulation up to 0.1"), ("dor_0.1-0.5", "Regulation 0.1 to 0.5"), ("dor_0.5-1", "Regulation 0.5 to 1"),
                   ("dor_1-2", "Regulation 1 to 2"), ("dor_>2", "Regulation above 2"), ("undammed", "Undammed")]:
        x = e[k]
        rows.append(f"<tr><td>{lab}</td><td class=\"num\">{x['s42']['n']}</td><td class=\"num\">{sg(x['s42']['median'])}</td>"
                    f"<td class=\"num\">{sg(x['s43']['median'])}</td><td class=\"num\">{ci(x['s43'])}</td><td class=\"num\">{sg(x['pooled']['median'])}</td>"
                    f"<td class=\"num\">{sg(x['noise_off']['median'])}</td><td class=\"num\">{r2(x['rho'])}</td><td class=\"num\">{x['same_sign']:.0%}</td>"
                    f"<td class=\"num\">{ab[k]['eff42']:.4f} / {ab[k]['eff43']:.4f}</td><td class=\"num\">{ab[k]['noise_off']:.4f}</td></tr>")
    tab1 = ('<div class="table-wrap"><table class="stats"><thead><tr><th>Gauges</th><th class="num">n</th><th class="num">Seed 42</th>'
            '<th class="num">Seed 43</th><th class="num">95 % interval, 43</th><th class="num">Both seeds</th><th class="num">No dam 43 − 42</th>'
            '<th class="num"><span class="g">ρ</span> per gauge</th><th class="num">Same sign</th><th class="num">|Change| 42 / 43</th>'
            '<th class="num">|No dam 43 − 42|</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")
    prow = []
    for tag, lab in [("off42", "No dam, seed 42"), ("dam42", "Dam release, seed 42"), ("off43", "No dam, seed 43"), ("dam43", "Dam release, seed 43")]:
        s_ = "s42" if tag.endswith("42") else "s43"
        arm = "nse_off" if tag.startswith("off") else "nse_dam"
        karm = "kge_off" if tag.startswith("off") else "kge_dam"
        prow.append(f"<tr><td>{lab}</td><td class=\"num\">{REPN['median_nse'][s_][arm]:.4f}</td><td class=\"num\">{REPN['median_nse'][s_][karm]:.4f}</td>"
                    f"<td class=\"num\">{pr[tag]['n']['median']:.4f}</td><td class=\"num\">{pr[tag]['gamma']['median']:.3f}</td>"
                    f"<td class=\"num\">{pr[tag]['gamma']['q25']:.3f} to {pr[tag]['gamma']['q75']:.3f}</td>"
                    + (f"<td class=\"num\">{(rr['T0_median_42'] if s_ == 's42' else rr['T0_median_43']) * 24:.1f} h</td>" if tag.startswith("dam") else "<td class=\"num\">–</td>")
                    + "</tr>")
    tab2 = ('<div class="table-wrap"><table class="stats"><thead><tr><th>Run</th><th class="num">Median NSE</th><th class="num">Median KGE</th>'
            '<th class="num">Median <span class="g">n₀</span></th><th class="num">Median <span class="g">γ</span></th><th class="num"><span class="g">γ</span> IQR</th><th class="num">Median <span class="g">T₀</span></th></tr></thead><tbody>'
            + "".join(prow) + "</tbody></table></div>")
    runs43 = REPN["runs"]["s43"]
    V["REPLICATE"] = f"""
  <p class="prose">Both arms retrained with seed 43 and nothing else changed (no dam <span class="m">{runs43['off']}</span>, dam release
  <span class="m">{runs43['learned']}</span>), scored on the same test years. The question: which of the seed-42 differences belong to the dams, and
  which to the particular training trajectory?</p>
  <div class="brief">
    <div><b>Hypothesis</b><p>If the release causes the gain below dams, a second seed should give a gain of the same sign and similar size, at largely the
    same gauges. If the undammed loss is trajectory noise, its size should match the difference between two no-dam runs that differ only in seed, and
    which gauges lose should not repeat.</p></div>
    <div><b>In plain words</b><p>The same paired comparison as Figure 2, done twice with different random seeds, side by side. Below, each gauge's
    change in one seed against its change in the other: points on the diagonal mean the release did the same thing at that gauge both times.</p></div>
    <div><b>What is drawn</b><p>Top: median per-gauge change in test NSE, dam release minus no dam, with 95 % bootstrap intervals; dark seed 42,
    ochre seed 43. Bottom: one point per gauge, seed 42 across and seed 43 up, symmetric log scales; ochre triangles dammed, teal circles
    undammed; quadrant counts on the right. Hover for details; click a point to open its hydrograph.</p></div>
  </div>
  <div class="fig">
    <div class="pickers">
      <label for="fsmet">Metric<select id="fsmet"><option value="nse">NSE</option><option value="kge">KGE</option></select></label>
      <div class="legend"><span><i class="dot" style="background:var(--ink)"></i>Seed 42</span><span><i class="dot" style="background:var(--dam)"></i>Seed 43</span></div>
    </div>
    <div class="figbox"><svg id="fs" class="wide" viewBox="0 0 900 420" role="img" aria-label="Median per-gauge change in test skill in two seeds"></svg><div class="tip" id="tfs" hidden></div></div>
  </div>
  <div class="fig">
    <div class="pickers">
      <label for="fxgrp">Gauges<select id="fxgrp"><option value="all">All</option><option value="d">Dammed only</option><option value="u">Undammed only</option></select></label>
      <div class="legend"><span><svg width="10" height="10" viewBox="-5 -5 10 10" aria-hidden="true"><path d="M0 -4.2L4 2.3L-4 2.3Z" class="pt-dam"/></svg>Dammed</span><span><i class="dot" style="background:var(--nodam)"></i>Undammed</span><span><i class="sw" style="background:var(--muted)"></i>Same change in both seeds</span></div>
    </div>
    <div class="figbox"><svg id="fx" class="wide" viewBox="0 0 760 560" role="img" aria-label="Per-gauge change in NSE, seed 42 against seed 43"></svg><div class="tip" id="tfx" hidden></div></div>
  </div>
  {tab1}
  <p class="note">Both seeds: median of the per-gauge mean of the two seeds' changes. No dam 43 − 42: median per-gauge difference between the two no-dam runs, the drift two trajectories show with no dams involved. ρ: Spearman rank correlation of the per-gauge change between seeds. Same sign: share of gauges whose change has the same sign in both seeds. The last two columns are medians of absolute per-gauge values: how much the release moves a typical gauge, against how much a change of seed alone moves it.</p>
  <div class="brief">
    <div><b>Result</b><p>At dammed gauges the gain reproduces: {sg(e['dammed']['s42']['median'])} in seed 42 and {sg(e['dammed']['s43']['median'])}
    ({ci(e['dammed']['s43'])}) in seed 43, {sg(e['on_reach']['s43']['median'])} with the dam on the gauge reach, and the gain in every regulation band from
    0.1 to 2 is positive with its interval clear of zero in both seeds. It holds for any pairing of seeds (dam 42 − no dam 43 {sg(e['dammed']['cross_d42_o43']['median'])},
    dam 43 − no dam 42 {sg(e['dammed']['cross_d43_o42']['median'])}). The per-gauge change correlates {r2(e['dammed']['rho'])} between seeds at dammed gauges
    ({r2(e['on_reach']['rho'])} on the gauge reach, {r2(e['dor_1-2']['rho'])} at regulation 1 to 2), and the typical per-gauge change
    ({ab['dammed']['eff42']:.4f} and {ab['dammed']['eff43']:.4f}) is about three times the difference between two no-dam seeds ({ab['dammed']['noise_off']:.4f}).
    At undammed gauges the median is negative in both seeds ({sg(e['undammed']['s42']['median'])}, {sg(e['undammed']['s43']['median'])}), but two no-dam runs
    differ by {sg(e['undammed']['noise_off']['median'])} there, the typical per-gauge change is the size of that seed difference
    ({ab['undammed']['eff42']:.4f}, {ab['undammed']['eff43']:.4f} against {ab['undammed']['noise_off']:.4f}), the per-gauge pattern does not repeat
    (ρ {r2(e['undammed']['rho'])}), and the cross-seed pairings disagree ({sg(e['undammed']['cross_d42_o43']['median'])}, {sg(e['undammed']['cross_d43_o42']['median'])}).
    KGE falls at dammed gauges in both seeds ({sg(e['dammed']['kge_s42']['median'])}, {sg(e['dammed']['kge_s43']['median'])}). Population median NSE:
    {m42['nse_off']:.4f} → {m42['nse_dam']:.4f} in seed 42, {m43['nse_off']:.4f} → {m43['nse_dam']:.4f} in seed 43.</p></div>
    <div><b>Conclusion</b><p>The gain below dams is a property of the method, not of one trajectory: same sign, similar size, largely the same gauges,
    and robust to how the seeds are paired. Its size, a few thousandths of median NSE (a few hundredths at the gauges that respond most), is what a
    peak-attenuating bucket buys on daily flow. The undammed loss is at the noise floor of training: consistent in sign in two seeds, which a coin also
    manages half the time, but not in which gauges lose. The population median moves by less than the spread between seeds, so this experiment shows
    no CONUS-wide gain or loss. KGE does not improve because the bucket smooths peaks and lowers simulated variability.</p></div>
  </div>
  <h3>Parameters and scores of the four runs</h3>
  {tab2}
  <div class="brief">
    <div><b>Result</b><p>γ swings from {min(pr[k]['gamma']['median'] for k in pr):.3f} to {max(pr[k]['gamma']['median'] for k in pr):.3f} across the four
    runs, and the order between arms reverses with the seed: the dam arm has the higher γ in seed 42 and the lower in seed 43. The two no-dam runs alone
    differ by {pdf['off43-off42']['gamma']['median_abs_over_iqr']:.1f} interquartile ranges in γ. n₀ moves less (median |Δ| {pdf['off43-off42']['n']['median_abs_over_iqr']:.2f}
    IQR between the no-dam seeds). The release head's T₀ agrees across seeds (per-dam ρ {rr['rho_T0']:.2f}, medians {rr['T0_median_42'] * 24:.1f} h and
    {rr['T0_median_43'] * 24:.1f} h); its seasonal phase does not (ρ {rr['rho_peak']:.2f}).</p></div>
    <div><b>Conclusion</b><p>Daily NSE does not pin down γ on this network, so no statement about stage-dependent roughness should rest on one run,
    and differences in γ between arms are not evidence about dams. What the release head learns reproducibly is the ranking of response times,
    not their seasonal timing. Longer training or a constraint on γ would be needed before either is interpreted physically.</p></div>
  </div>"""
    V["RUNS43"] = (f"; seed-43 replicate: no dam <span class=\"m\">{runs43['off']}</span>, dam release <span class=\"m\">{runs43['learned']}</span>")
    g43 = json.load(open(WEB / "build/gauges_s43_nse.json"))
    V["REP_JSON"] = json.dumps(dict(res43={"paired": R43["paired"]}, g43=g43), separators=(",", ":"))
else:
    V["REP_TITLE"] = "A second seed is running"
    V["REPLICATE"] = ('<div class="pending"><p>Both arms are being retrained with seed 43 (same configuration otherwise). When they finish, this '
                      "section will show the same paired comparison on the new pair, whether the change at undammed gauges keeps its sign across "
                      "seeds, and how far two no-dam runs that differ only in seed land apart at each gauge. That spread is the noise floor the "
                      f"{sg(p_und['median'])} and {sg(p_dam['median'])} have to clear.</p></div>")
    V["REP_JSON"] = "null"

# ---------- images ----------


def fig(key, cap, alt=None):
    m = IMG[key]
    return (f'<figure><a href="{m["file"]}" target="_blank" rel="noopener"><img src="{m["file"]}" width="{m["w"]}" height="{m["h"]}" loading="lazy" '
            f'alt="{alt or cap}"></a><figcaption>{cap}</figcaption></figure>')


V["IMG_PARAMS"] = "".join([
    fig("off/parameter_map_n_conus_stretched", "n₀, no dam (shared scale, 2nd to 98th percentile of both arms)"),
    fig("dam/parameter_map_n_conus_stretched", "n₀, dam release (same scale)"),
    fig("dam/parameter_map_n_delta_dam_minus_off", "Change in n₀, dam release minus no dam; triangles are dam reaches"),
    fig("off/parameter_map_gamma_conus_stretched", "γ, no dam (shared scale)"),
    fig("dam/parameter_map_gamma_conus_stretched", "γ, dam release (same scale)"),
    fig("dam/parameter_map_gamma_delta_dam_minus_off", "Change in γ, dam release minus no dam"),
    fig("off/parameter_convergence_gamma", "γ by epoch, no dam: still moving at epoch 50"),
    fig("dam/parameter_convergence_gamma", "γ by epoch, dam release: still moving at epoch 50"),
])
HP = pd.read_csv(RUNS / DAM / "plots/hydrograph_picks.csv", dtype={"STAID": str})


def hcap(stem):
    gid, wy = stem.split("_")[1], int(stem.split("_wy")[1])
    r = HP[(HP.STAID == gid) & (HP.wy == wy)].iloc[0]
    return f"{r.label}: {G.loc[gid].STANAME.title()}, WY{wy} ({r.kind} test year)"


GAL = [
    ("Metrics: summed Q′, no dam and dam release on the same gauges", [
        ("dam/metrics_boxplot", "Six metrics across the 2,365 gauges (NSE and KGE clipped to −1..1)"),
        ("dam/metrics_nse_cdf", "NSE, cumulative distribution"),
        ("dam/metrics_kge_cdf", "KGE, cumulative distribution"),
        ("dam/metrics_drainage_area", "NSE by drainage-area class, medians annotated"),
        ("dam/metrics_dnse_dam_minus_off_map", "Change in test NSE, dam release minus no dam (static version of Figure 4)"),
        ("off/metrics_gauge_map", "Test NSE, no dam"),
        ("dam/metrics_gauge_map", "Test NSE, dam release"),
        ("off/metrics_dnse_vs_summed_qprime_map", "Change in test NSE, no dam minus summed Q′"),
        ("dam/metrics_dnse_vs_summed_qprime_map", "Change in test NSE, dam release minus summed Q′"),
    ]),
    ("Hydrographs: the typical and the wettest test year at the four gauges of Figure 10", [
        (f"dam/{p.stem}", hcap(p.stem)) for p in
        sorted((RUNS / DAM / "plots").glob("hydrograph_*.png"), key=lambda q: [g["id"] for g in PK["gauges"]].index(q.stem.split("_")[1]))
    ]),
    ("Learned parameters: declared range, distributions, scale and convergence", [
        ("off/parameter_map_n_conus_fullrange", "n₀ on its declared range [0.015, 0.25], no dam"),
        ("dam/parameter_map_n_conus_fullrange", "n₀ on its declared range, dam release"),
        ("off/parameter_map_gamma_conus_fullrange", "γ on its declared range [0, 0.5], no dam"),
        ("dam/parameter_map_gamma_conus_fullrange", "γ on its declared range, dam release"),
        ("dam/parameter_hist_n_conus", "n₀ distributions, both arms"),
        ("dam/parameter_hist_gamma_conus", "γ distributions, both arms"),
        ("off/parameter_scatter_n_vs_log10_uparea_conus", "n₀ against drainage area, no dam"),
        ("dam/parameter_scatter_n_vs_log10_uparea_conus", "n₀ against drainage area, dam release"),
        ("off/parameter_scatter_gamma_vs_log10_uparea_conus", "γ against drainage area, no dam"),
        ("dam/parameter_scatter_gamma_vs_log10_uparea_conus", "γ against drainage area, dam release"),
        ("off/parameter_convergence_n", "n₀ by epoch, no dam"),
        ("dam/parameter_convergence_n", "n₀ by epoch, dam release"),
    ]),
    ("Stage roughness n(d) over water year 2000 (depth from accumulated Q′, not the routed flow)", [
        ("off/n_of_d_wy2000_traces", "n(d) at sample reaches, no dam"),
        ("dam/n_of_d_wy2000_traces", "n(d) at sample reaches, dam release"),
        ("off/n_of_d_wy2000_maps", "n(d) at each reach's low and high flow, and the swing, no dam"),
        ("dam/n_of_d_wy2000_maps", "n(d) at each reach's low and high flow, and the swing, dam release"),
    ]),
]
gal = []
used = set(re.findall(r'src="(img/[^"]+)"', V["IMG_PARAMS"]))
for title, items in GAL:
    gal.append(f"<h3>{title}</h3><div class=\"imgs\">" + "".join(fig(k, cap) for k, cap in items) + "</div>")
V["GALLERY"] = "".join(gal)
V["GALLERY"] += ("<p class=\"note\">The per-arm maps of n₀ and γ on the shared scale, the change maps and the γ convergence panels are in "
                 "Figure 9. Stage roughness: per-reach n(d) swing between low and high flow, median "
                 "1.09× without dams and 1.28× with them, a direct consequence of the larger γ.</p>")
used |= set(re.findall(r'src="(img/[^"]+)"', V["GALLERY"]))

mp = json.loads((EXP / "smoke/page/map.json").read_text())
V["MAP_JSON"] = json.dumps(dict(w=mp["w"], h=mp["h"], regions=mp["regions"], outline=mp["outline"]), separators=(",", ":")).replace("</", "<\\/")
PK["viewer"], PK["viewer_year"] = "14238000", 1997
V["PICKS_JSON"] = json.dumps(PK, separators=(",", ":"))
V["RES_JSON"] = json.dumps({k: R[k] for k in ("paired", "huc2", "release", "group_n")}, separators=(",", ":"))

src = (HERE / "page_src.html").read_text()
out = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda k: V[k.group(1)], src)
assert "{{" not in out, re.findall(r"\{\{[A-Z0-9_]+\}\}", out)
assert "—" not in out, "em dash in page: " + out[max(0, out.index("—") - 80):out.index("—") + 20]
(PUB / "learned_dam_release.html").write_text(out)
(PUB / "used_images.json").write_text(json.dumps(sorted(used)))
site = WEB / "site"
site.mkdir(exist_ok=True)
(site / "index.html").write_text('<meta charset="utf-8">\n' + out)
for nm in ("data_s42.json", "series", "img") + (("data_s43.json",) if (PUB / "data_s43.json").exists() else ()):
    p = site / nm
    if p.is_symlink() or p.exists():
        p.unlink()
    os.symlink(PUB / nm, p)
print("page bytes", len(out.encode()), "images used", len(used), "placeholders", len(V))
