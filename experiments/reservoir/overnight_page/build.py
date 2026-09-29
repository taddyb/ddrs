#!/usr/bin/env python
"""Inline data.json and the laws_v6 figures into page_src.html -> publish/per_dam_release_calibration.html.

Run prepare.py first. Figures are embedded as base64 data URIs (experiments/reservoir/laws_v6/fig1, fig4).
"""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAWS = HERE.parent / "laws_v6"
src = (HERE / "page_src.html").read_text()
data = (HERE / "data.json").read_text().replace("</", "<\\/")
for key, name in (("{{FIG1}}", "fig1_flood_composite.png"), ("{{FIG4}}", "fig4_flood_law.png")):
    src = src.replace(key, "data:image/png;base64," + base64.b64encode((LAWS / name).read_bytes()).decode())
out = HERE / "publish" / "per_dam_release_calibration.html"
out.parent.mkdir(exist_ok=True)
out.write_text(src.replace("/*DATA*/", data))
print(out, round(out.stat().st_size / 1e6, 2), "MB")
