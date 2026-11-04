"""Fetch one USGS NWIS daily-value series and write it as `date,gage_id,q_m3s`.

Source: https://waterservices.usgs.gov/nwis/dv/ (RDB format), one site, one
parameter code. Built for 01372058 (Hudson River below Poughkeepsie), parameter
72137 (tidally filtered discharge, mean, cfs). Values convert from cfs to
m3/s (x 0.0283168466). Days with no value, or a non-numeric value, are left
out; the extractor writes NaN there. Negative values are kept: a tidally
filtered flow can be negative.

The raw RDB response is saved next to `--raw` and parsed only as tab-separated
text. Prints the day count, the negative-day count, the minimum and the count
of provisional/estimated days (qualifier other than plain `A`).

Run under the DDR venv:

    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/fetch_nwis_daily.py \
        --site 01372058 --param 72137 --start 2001-01-01 --end 2019-12-31 \
        --raw RAW.rdb --out OUT.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import requests

URL = "https://waterservices.usgs.gov/nwis/dv/"
CFS_TO_M3S = 0.0283168466


def main() -> None:
    """Fetch, parse and write the series."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--site", required=True)
    p.add_argument("--param", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--raw", type=Path, required=True, help="where to save the RDB response")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    params = {
        "format": "rdb",
        "sites": args.site,
        "parameterCd": args.param,
        "startDT": args.start,
        "endDT": args.end,
        "statCd": "00003",
    }
    r = requests.get(URL, params=params, timeout=180)
    r.raise_for_status()
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.raw.write_text(r.text)

    lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 3:
        raise SystemExit(f"no data rows in the response ({len(lines)} non-comment lines)")
    header = lines[0].split("\t")
    di = header.index("datetime")
    vi = next(i for i, h in enumerate(header) if h.endswith(f"_{args.param}_00003"))
    qi = vi + 1
    rows = []
    quals: dict[str, int] = {}
    for ln in lines[2:]:  # lines[1] is the RDB field-width row
        f = ln.split("\t")
        if len(f) <= qi:
            continue
        try:
            q_cfs = float(f[vi])
        except ValueError:
            continue
        rows.append((f[di], q_cfs * CFS_TO_M3S))
        quals[f[qi]] = quals.get(f[qi], 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["date", "gage_id", "q_m3s"])
        for d, q in rows:
            w.writerow([d, args.site, f"{q:.4f}"])
    neg = [q for _, q in rows if q < 0]
    print(f"{args.site} {args.param}: {len(rows)} days, {rows[0][0]} .. {rows[-1][0]}", file=sys.stderr)
    print(f"negative days {len(neg)}, min {min(q for _, q in rows):.2f} m3/s", file=sys.stderr)
    print(f"qualifier counts {quals}", file=sys.stderr)


if __name__ == "__main__":
    main()
