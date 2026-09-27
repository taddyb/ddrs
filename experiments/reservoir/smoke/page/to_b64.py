"""Artifacts do not serve .bin: publish each region's uint16 series as base64 text (series/hucNN.b64.txt)."""
import base64
from pathlib import Path

WEB = Path(__file__).resolve().parents[4] / "output/reservoir_smoke/web/series"
OUT = WEB.parent / "publish" / "series"
OUT.mkdir(parents=True, exist_ok=True)
tot = 0
for f in sorted(WEB.glob("huc*.bin")):
    b = f.read_bytes()
    o = OUT / (f.stem + ".b64.txt")
    o.write_text(base64.b64encode(b).decode("ascii"))
    assert base64.b64decode(o.read_text()) == b
    tot += o.stat().st_size
    print(o.name, o.stat().st_size)
print("total", tot)
