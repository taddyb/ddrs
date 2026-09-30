#!/usr/bin/env python
"""Flag gauges whose USGS station name places them below or at a dam, reservoir or lake outlet.

Classes (first match wins):
  below            BELOW / BL / BLW ... DAM / RES / RESERVOIR / LAKE / LK / POND
  at dam / outlet  TAILWATER, SPILLWAY, POWERHOUSE, FOREBAY, OUTLET, or AT / NR ... DAM
  above            AB / ABV / ABOVE ... structure: an inflow gauge, not regulated by the named dam
  other mention    a structure word elsewhere in the name, usually a place name
  none
Cross-checked against the NWM reservoir table (regulation_sizing.py's regulation_by_gauge.csv), NWIS peak
code 6 in WY1996-2010, and the dam benchmark. Writes gauge_name_dam_flags.csv next to this script.
Run under ~/projects/ddr/.venv.
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GAGES = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"
REG = "/home/tbindas/projects/ddrs/output/dam_sandbox/regulation_by_gauge.csv"
PEAK = "/mnt/ssd1/data/usgs_regulation/derived/peak_regulation_summary.csv"

S = r"(?:DAM|DAMS|RESERVOIR|RES|RSVR|RESV|LAKE|LK|POND)"
OPS = r"\b(?:TAILWATER|TAILRACE|SPILLWAY|POWERHOUSE|POWERPLANT|PWRPLNT|FOREBAY|AFTERBAY)\b"


def classify(n):
    n = n.upper()
    if re.search(r"\b(?:BELOW|BLW|BL)\b[^,]*?\b" + S + r"\b", n):
        return "below"
    if re.search(OPS, n) or re.search(r"\b(?:AT|NR|NEAR)\b[^,]*?\b(?:DAM|DAMS)\b", n) or re.search(r"\bOUTLET\b", n):
        return "at dam / outlet"
    if re.search(r"\b(?:ABOVE|ABV|AB)\b[^,]*?\b" + S + r"\b", n):
        return "above"
    if re.search(r"\b" + S + r"\b", n):
        return "other mention"
    return "none"


g = pd.read_csv(GAGES, dtype={"STAID": str})
reg = pd.read_csv(REG, dtype={"STAID": str}).set_index("STAID")
peak = pd.read_csv(PEAK, dtype={"STAID": str}).set_index("STAID")
bench = set(pd.read_csv(HERE / "dam_benchmark.csv", dtype={"gauge": str}).gauge)

g["cat"] = g.STANAME.map(classify)
g["in_eval"] = g.STAID.isin(reg.index)
g["nwm_res_upstream"] = g.STAID.map(reg.n_res).fillna(0) > 0
g["dor_gt_05"] = g.STAID.map(reg.cls) == "DOR>0.5"
g["code6"] = g.STAID.map(peak.n_years_code6_wy1996_2010).fillna(0) > 0
g["benchmark"] = g.STAID.isin(bench)

e = g[g.in_eval]
print(e.groupby("cat").agg(n=("STAID", "size"), nwm_res_upstream=("nwm_res_upstream", "sum"), dor_gt_05=("dor_gt_05", "sum"),
                           code6=("code6", "sum"), benchmark=("benchmark", "sum")).to_string())
hits = e[e.cat.isin(["below", "at dam / outlet"])]
miss = hits[~hits.nwm_res_upstream]
print(f"\nregulated by name: {len(hits)}; no NWM reservoir upstream: {len(miss)} ({int(miss.code6.sum())} with peak code 6)")
print(miss[["STAID", "STANAME", "DRAIN_SQKM", "code6"]].to_string(index=False))
g[["STAID", "STANAME", "COMID", "cat", "in_eval", "nwm_res_upstream", "dor_gt_05", "code6", "benchmark"]].to_csv(
    HERE / "gauge_name_dam_flags.csv", index=False)
