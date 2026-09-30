#!/usr/bin/env python
"""Copy the ddrs-eval-plots PNGs of both seed-42 arms into the page's img/ folder at display resolution.

Maps become JPEG (quality 82, 1600 px wide); line and box charts stay PNG, palette-quantised, 1400 px wide.
Writes output/reservoir_full_run/web/publish/img/{off,dam}_<name>.{jpg,png} and img/manifest.json.
"""
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[3] / "output/reservoir_full_run/web/publish/img"
RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
ARMS = {"off": "2026-09-27T07-29-47Z-train-and-test", "dam": "2026-09-27T07-29-55Z-train-and-test"}
MAPS = ("parameter_map_", "_map.png", "n_of_d_wy2000_maps")
OUT.mkdir(parents=True, exist_ok=True)
man, tot = {}, 0
for arm, run in ARMS.items():
    for p in sorted((RUNS / run / "plots").glob("*.png")):
        im = Image.open(p).convert("RGB")
        is_map = any(k in p.name for k in MAPS)
        w = 1600 if is_map else 1400
        if im.width > w:
            im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        if is_map:
            f = OUT / f"{arm}_{p.stem}.jpg"
            im.save(f, "JPEG", quality=82, optimize=True, progressive=True)
        else:
            f = OUT / f"{arm}_{p.stem}.png"
            im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(f, "PNG", optimize=True)
        man[f"{arm}/{p.stem}"] = dict(file=f"img/{f.name}", w=im.width, h=im.height, bytes=f.stat().st_size)
        tot += f.stat().st_size
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print(len(man), "images,", round(tot / 1e6, 2), "MB")
