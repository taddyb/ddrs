#!/usr/bin/env python3
"""Write a synthetic hourly tide for the stage block's prescribed boundary.

    h_bc(t) = h0 + amp * sin(2 pi t / 12.42 h)

t counts hours from --start 00:00:00Z. Output is a CSV with header
`time,stage_m`; `time` is formatted exactly `%Y-%m-%dT%H:%M:%SZ` (UTC), the one
format `data_sources.stage_boundary` accepts.

WET by choice: tests/juniata_stage.rs re-implements this formula in Rust so
the test needs no Python. Keep the two in step.

    python scripts/make_synthetic_stage.py --start 1990-01-01 --hours 2400 \
        --h0 2.0 --amp 0.5 --out tide.csv
"""
import argparse
import math
from datetime import datetime, timedelta, timezone

PERIOD_H = 12.42  # M2 tide


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", required=True, help="YYYY-MM-DD, first row is 00:00:00Z")
    ap.add_argument("--hours", type=int, required=True)
    ap.add_argument("--h0", type=float, default=2.0)
    ap.add_argument("--amp", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    with open(a.out, "w") as f:
        f.write("time,stage_m\n")
        for t in range(a.hours):
            stamp = (t0 + timedelta(hours=t)).strftime("%Y-%m-%dT%H:%M:%SZ")
            h = a.h0 + a.amp * math.sin(2.0 * math.pi * t / PERIOD_H)
            f.write(f"{stamp},{h:.6f}\n")


if __name__ == "__main__":
    main()
