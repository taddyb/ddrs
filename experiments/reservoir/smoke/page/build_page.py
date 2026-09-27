"""Fill page_src.html from the fit outputs + prose.json, write the publish copy and a local preview folder."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parents[3]
WEB = W / "output/reservoir_smoke/web"
idx = json.load(open(WEB / "index.json"))
S = idx["summary"]
m = pd.DataFrame(idx["meta"]).set_index("STAID")
dam, ctl = m[m.role == "dam"], m[m.role == "control"]
summ = json.load(open(W / "experiments/reservoir/smoke/smoke_summary.json"))
secs = int(open(W / "output/reservoir_smoke/timing.txt").read().split()[1])
prose = json.load(open(HERE / "prose.json"))


def f3(v):
    return f"{v:.3f}".replace("-", "−")


def sg(v, d=3):
    if abs(v) < 0.5 * 10 ** -d:
        return f"{0:.{d}f}"
    return ("+" if v >= 0 else "−") + f"{abs(v):.{d}f}"


def ud(p):
    return f"{p['n_up']} / {p['n_down']}"


N = S["n"]
mt, mk, pd_ = S["median_test_nse"], S["median_test_kge"], S["paired_dnse"]
dmc = S["dam_minus_matched_control"]
V = dict(
    N_DAM=str(N["dam"]), N_CTL=str(N["control"]), N_ALL=str(N["dam"] + N["control"]), N_CASCADE=str(N["cascade"]),
    N_ONREACH=str(N["on_reach"]), N_CROSS=str(summ["controls_cross_huc"]), REACHES=f"{summ['network_reaches']:,}",
    EVAL_MIN=str(round(secs / 60)), N_DAMS_NET=str(summ["dams_ge10mcm_in_network"]),
    T_DAM=sg(pd_["seas"]["dam"]["median"]), T_CTL=sg(pd_["seas"]["control"]["median"]), T_DIFF=sg(dmc["median"]),
    T_DAM_NOTE=f"{pd_['seas']['dam']['n_up']} up, {pd_['seas']['dam']['n_down']} down; median NSE {f3(mt['pass']['dam'])} → {f3(mt['seas']['dam'])}",
    T_CTL_NOTE=f"{pd_['seas']['control']['n_up']} up, {pd_['seas']['control']['n_down']} down; median NSE {f3(mt['pass']['control'])} → {f3(mt['seas']['control'])}",
    T_DIFF_NOTE=f"median over {dmc['n']} pairs, 95 % interval {sg(dmc['ci'][0])} to {sg(dmc['ci'][1])}; dam ahead in {dmc['n_up']}, control in {dmc['n_down']}",
)
for lab, key in [("pass", "PASS"), ("lin", "LIN"), ("seas", "SEAS")]:
    V[f"NSE_DAM_{key}"] = f3(mt[lab]["dam"]); V[f"NSE_CTL_{key}"] = f3(mt[lab]["control"])
    V[f"KGE_DAM_{key}"] = f3(mk[lab]["dam"]); V[f"KGE_CTL_{key}"] = f3(mk[lab]["control"])
for lab, key in [("lin", "LIN"), ("seas", "SEAS")]:
    V[f"UD_DAM_{key}"] = ud(pd_[lab]["dam"]); V[f"UD_CTL_{key}"] = ud(pd_[lab]["control"])
rows = []
for nm, g, c in [("Dam gauges", dam, "var(--dam)"), ("Controls", ctl, "var(--nodam)")]:
    qs = g.area_km2.quantile([0, .25, .5, .75, 1]).tolist()
    rows.append(f'<tr><td><i class="dot" style="background:{c}"></i>{nm} ({len(g)})</td>' + "".join(f'<td class="num">{q:,.0f}</td>' for q in qs) + "</tr>")
V["AREA_ROWS"] = "".join(rows)
V.update(prose["text"])
mp = json.loads((HERE / "map.json").read_text())
mp.pop("source", None)
V["MAP_JSON"] = json.dumps(mp, separators=(",", ":")).replace("</", "<\\/")
V["PICKS_JSON"] = json.dumps(prose["picks"], separators=(",", ":"))

src = (HERE / "page_src.html").read_text()
out = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda k: V[k.group(1)], src)
assert "{{" not in out, re.findall(r"\{\{[A-Z0-9_]+\}\}", out)
assert "—" not in out, "em dash in page"
PUB = WEB / "publish"
PUB.mkdir(parents=True, exist_ok=True)
site = PUB / "site"
site.mkdir(exist_ok=True)
(PUB / "dam_release_smoke.html").write_text(out)
(site / "index.html").write_text('<meta charset="utf-8">\n' + out)
for nm in ["index.json", "series"]:
    p = site / nm
    if not p.exists():
        os.symlink(WEB / nm, p)
print("page bytes", len(out.encode()))
print("placeholders filled", len(V))
