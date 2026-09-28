#!/usr/bin/env python
"""Inline data.json into page_src.html -> publish/per_dam_release_calibration.html (run prepare.py first)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "page_src.html").read_text()
data = (HERE / "data.json").read_text().replace("</", "<\\/")
out = HERE / "publish" / "per_dam_release_calibration.html"
out.parent.mkdir(exist_ok=True)
out.write_text(src.replace("/*DATA*/", data))
print(out, round(out.stat().st_size / 1e6, 2), "MB")
