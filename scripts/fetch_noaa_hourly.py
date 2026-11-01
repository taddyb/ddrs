"""Fetch NOAA CO-OPS hourly water level and write a ddrs stage-boundary CSV.

Source: https://api.tidesandcurrents.noaa.gov/api/prod/datagetter,
`product=hourly_height`, one calendar year per request, GMT, metric.
Output columns: `time,stage_m`, with `time` formatted `%Y-%m-%dT%H:%M:%SZ`
(UTC), the only format `StageBoundaryStore` accepts.

Gaps: a missing hour is filled by linear interpolation only when the run of
missing hours is at most `--max-gap-hours` (6) long and has a valid hour on
both sides. Every filled hour is written to `<out>.gaps.txt` as
`<time>,<method>`. A longer run, or a run at the start or the end of the
record, stops the script with exit code 3 and a list of the runs (first
missing hour, last missing hour, length in hours). Nothing is written to
`--out` in that case. Choose a fill, add it here, and log every hour it fills.

Optional fills for runs longer than 6 h, each an explicit flag (never default):

  --fill-6min START:END   Hours in the span that are missing in the hourly
      product are filled from the CO-OPS 6-minute `water_level` product
      (NAVD, metric, GMT). Only samples with all quality flags 0 (O/I, F, R, L)
      count. The hour value is the mean of the flag-0 samples within +-24 min of
      the hour (up to 9 samples at 6-minute spacing, symmetric about the hour),
      and at least 5 are required. An hour with fewer falls back to
      predictions plus residual, as below, using the span's own bounds.
      Method tag: `6min_mean` (fallback hours: `predictions_plus_residual`).
  --fill-pred START:END   Hours in the span are filled with the CO-OPS hourly
      `predictions` (NAVD, `interval=h`) plus a residual. The residual runs
      linearly across the span between two values: the 7-day mean of
      (observed - predicted) just before the span and the same just after it
      (valid observed hours only). Method tag: `predictions_plus_residual`.

START and END are `YYYY-MM-DD` (END inclusive, through 23:00). A span that does
not cover a >6 h run does not remove the stop: any other run longer than 6 h
still stops the script. Runs of at most 6 h are filled linearly (`linear`).

A value with the CO-OPS limit flag `L=1` counts as missing. Responses are
cached in `--cache-dir` (one file per year), so a re-run does not fetch again.
The cache holds untrusted downloads: it is parsed only as CSV text.

Run under the DDR venv:

    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/fetch_noaa_hourly.py \
        --station 8518750 --start 2001-01-01 --end 2019-12-31 \
        --cache-dir DOWNLOAD_DIR --out OUT.csv
"""

import argparse
import csv
import io
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def fetch_year(station: str, year: int, cache: Path) -> str:
    """Return the CSV text for one calendar year, from the cache or the API."""
    path = cache / f"{station}_{year}.csv"
    if path.exists():
        return path.read_text()
    params = {
        "product": "hourly_height",
        "application": "ddrs_hudson_example",
        "begin_date": f"{year}0101",
        "end_date": f"{year}1231",
        "station": station,
        "datum": "NAVD",
        "units": "metric",
        "time_zone": "gmt",
        "format": "csv",
    }
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=120)
            r.raise_for_status()
            path.write_text(r.text)
            return r.text
        except requests.RequestException as e:
            if attempt == 3:
                raise SystemExit(f"{year}: request failed after 4 tries: {e}") from e
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_cached(params: dict, path: Path) -> str:
    """GET one CO-OPS request, cached at `path`."""
    if path.exists():
        return path.read_text()
    full = {"application": "ddrs_hudson_example", "station": params.pop("station"), "units": "metric",
            "time_zone": "gmt", "datum": "NAVD", "format": "csv", **params}
    for attempt in range(4):
        try:
            r = requests.get(URL, params=full, timeout=120)
            r.raise_for_status()
            path.write_text(r.text)
            time.sleep(2)
            return r.text
        except requests.RequestException as e:
            if attempt == 3:
                raise SystemExit(f"request failed after 4 tries: {e}") from e
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def parse_span(text: str) -> tuple[datetime, datetime]:
    """Parse START:END (YYYY-MM-DD, END inclusive) into hour bounds."""
    a, b = text.split(":")
    return (datetime.strptime(a, "%Y-%m-%d"),
            datetime.strptime(b, "%Y-%m-%d") + timedelta(hours=23))


def six_minute(station: str, span: tuple[datetime, datetime], cache: Path) -> dict[datetime, float]:
    """Flag-0 6-minute samples over the span, +-1 h margin, by timestamp."""
    out: dict[datetime, float] = {}
    day = (span[0] - timedelta(hours=1)).replace(hour=0, minute=0)
    stop = span[1] + timedelta(hours=1)
    while day <= stop:
        end = min(day + timedelta(days=30), stop)
        d0, d1 = day.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        text = fetch_cached(
            {"product": "water_level", "begin_date": d0, "end_date": d1, "station": station},
            cache / f"{station}_6min_{d0}_{d1}.csv",
        )
        rows = csv.reader(io.StringIO(text))
        next(rows, None)
        for row in rows:
            if len(row) < 7:
                continue
            if row[1].strip() == "" or any(x.strip() != "0" for x in row[3:7]):
                continue
            out[datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M")] = float(row[1])
        day = end + timedelta(days=1)
    return out


def predictions(station: str, years: set[int], cache: Path) -> dict[datetime, float]:
    """Hourly harmonic predictions, NAVD, for the given years."""
    out: dict[datetime, float] = {}
    for y in sorted(years):
        text = fetch_cached(
            {"product": "predictions", "interval": "h", "begin_date": f"{y}0101",
             "end_date": f"{y}1231", "station": station},
            cache / f"{station}_pred_{y}.csv",
        )
        rows = csv.reader(io.StringIO(text))
        header = next(rows, None)
        if header is None or [h.strip() for h in header][:2] != ["Date Time", "Prediction"]:
            raise SystemExit(f"unexpected predictions header: {header!r}")
        for row in rows:
            if len(row) >= 2 and row[1].strip():
                out[datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M")] = float(row[1])
    return out


def parse_year(text: str) -> tuple[dict[datetime, float], int, int]:
    """Parse one response. Returns values by hour, count of I=1 and count of L=1 rows."""
    values: dict[datetime, float] = {}
    inferred = limit = 0
    rows = csv.reader(io.StringIO(text))
    header = next(rows, None)
    if header is None or [h.strip() for h in header][:2] != ["Date Time", "Water Level"]:
        raise SystemExit(f"unexpected response header: {header!r}")
    for row in rows:
        if len(row) < 5:
            continue  # an `Error: ...` line
        t = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M")
        if t.minute != 0:
            raise SystemExit(f"hourly product returned a non-hourly time {row[0]!r}")
        i_flag, l_flag = row[3].strip(), row[4].strip()
        inferred += i_flag == "1"
        limit += l_flag == "1"
        v = row[1].strip()
        if v == "" or l_flag == "1":
            continue
        values[t] = float(v)
    return values, inferred, limit


def main() -> None:
    """Fetch, find gaps, fill the short ones, write the CSV and the gap log."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--station", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--max-gap-hours", type=int, default=6)
    p.add_argument("--fill-6min", action="append", default=[], metavar="START:END")
    p.add_argument("--fill-pred", action="append", default=[], metavar="START:END")
    p.add_argument("--sleep", type=float, default=2.0, help="seconds between year requests")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(hours=23)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    values: dict[datetime, float] = {}
    n_inferred = n_limit = 0
    for year in range(start.year, end.year + 1):
        cached = (args.cache_dir / f"{args.station}_{year}.csv").exists()
        v, inf, lim = parse_year(fetch_year(args.station, year, args.cache_dir))
        values.update(v)
        n_inferred += inf
        n_limit += lim
        print(f"{year}: {len(v)} hours (I=1 rows {inf}, L=1 rows {lim})", file=sys.stderr)
        if not cached:
            time.sleep(args.sleep)

    n = int((end - start) / timedelta(hours=1)) + 1
    times = [start + timedelta(hours=k) for k in range(n)]
    y = np.array([values.get(t, np.nan) for t in times], dtype=np.float64)
    missing = np.isnan(y)

    # Runs of missing hours: (first index, last index).
    runs: list[tuple[int, int]] = []
    k = 0
    while k < n:
        if missing[k]:
            j = k
            while j + 1 < n and missing[j + 1]:
                j += 1
            runs.append((k, j))
            k = j + 1
        else:
            k += 1

    index = {t: k for k, t in enumerate(times)}
    method: dict[int, str] = {}
    obs = y.copy()  # observed only; fills below never feed a residual

    def idx_span(span: tuple[datetime, datetime]) -> tuple[int, int]:
        if span[0] not in index or span[1] not in index:
            raise SystemExit(f"fill span {span[0]} .. {span[1]} is outside the record")
        return index[span[0]], index[span[1]]

    spans_6 = [parse_span(t) for t in args.fill_6min]
    spans_p = [parse_span(t) for t in args.fill_pred]
    pred: dict[datetime, float] = {}
    if spans_6 or spans_p:
        years = set()
        for a, b in spans_6 + spans_p:
            years |= {(a - timedelta(days=8)).year, (b + timedelta(days=8)).year}
        pred = predictions(args.station, years, args.cache_dir)

    def pred_residual_fill(a: int, b: int, hours: list[int]) -> None:
        """Fill `hours` in the run a..b with predictions + a linear residual."""
        def mean_resid(lo: int, hi: int) -> float:
            d = [obs[k] - pred[times[k]] for k in range(max(lo, 0), min(hi, n))
                 if not np.isnan(obs[k]) and times[k] in pred]
            if len(d) < 24:
                raise SystemExit(f"fewer than 24 valid hours for the residual near {times[a]}")
            return float(np.mean(d))
        before = mean_resid(a - 168, a)
        after = mean_resid(b + 1, b + 1 + 168)
        for k in hours:
            if times[k] not in pred:
                raise SystemExit(f"no prediction for {times[k]}")
            r = before + (after - before) * (k - (a - 1)) / (b + 1 - (a - 1))
            y[k] = pred[times[k]] + r
            method[k] = "predictions_plus_residual"

    for span in spans_6:
        a, b = idx_span(span)
        s6 = six_minute(args.station, span, args.cache_dir)
        fallback = []
        for k in range(a, b + 1):
            if not missing[k]:
                continue
            vals = [s6[times[k] + timedelta(minutes=m)] for m in range(-24, 25, 6)
                    if times[k] + timedelta(minutes=m) in s6]
            if len(vals) >= 5:
                y[k] = float(np.mean(vals))
                method[k] = "6min_mean"
            else:
                fallback.append(k)
        if fallback:
            print(f"{len(fallback)} hours in {span[0]:%Y-%m-%d}..{span[1]:%Y-%m-%d} fall back to "
                  f"predictions_plus_residual", file=sys.stderr)
            pred_residual_fill(a, b, fallback)
    for span in spans_p:
        a, b = idx_span(span)
        pred_residual_fill(a, b, [k for k in range(a, b + 1) if missing[k]])

    still = np.isnan(y)
    runs2: list[tuple[int, int]] = []
    k = 0
    while k < n:
        if still[k]:
            j = k
            while j + 1 < n and still[j + 1]:
                j += 1
            runs2.append((k, j))
            k = j + 1
        else:
            k += 1
    bad = [(a, b) for a, b in runs2
           if (b - a + 1) > args.max_gap_hours or a == 0 or b == n - 1]
    print(f"{int(missing.sum())} missing hours in {len(runs)} runs; "
          f"{len(method)} filled by flags; {len(bad)} runs cannot be filled", file=sys.stderr)
    if bad:
        print("GAPS TO DECIDE (first missing hour, last missing hour, length in hours):", file=sys.stderr)
        for a, b in bad:
            print(f"  {times[a].strftime(TIME_FORMAT)}  {times[b].strftime(TIME_FORMAT)}  {b - a + 1}", file=sys.stderr)
        raise SystemExit(3)
    for a, b in runs2:
        lo, hi = y[a - 1], y[b + 1]
        for idx in range(a, b + 1):
            y[idx] = lo + (hi - lo) * (idx - (a - 1)) / (b + 1 - (a - 1))
            method[idx] = "linear"
    log = [f"{times[k].strftime(TIME_FORMAT)},{method[k]}" for k in sorted(method)]
    counts = {m: sum(1 for v in method.values() if v == m) for m in sorted(set(method.values()))}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["time", "stage_m"])
        for t, v in zip(times, y, strict=True):
            w.writerow([t.strftime(TIME_FORMAT), f"{v:.3f}"])
    with open(f"{args.out}.gaps.txt", "w") as f:
        f.write(f"# station {args.station}, hourly_height, NAVD88 m, GMT; {len(log)} filled hours {counts}\n")
        f.write("# time,method\n")
        f.writelines(line + "\n" for line in log)
    print(f"wrote {n} rows to {args.out}; filled {len(log)} hours; "
          f"I=1 rows {n_inferred}, L=1 rows {n_limit} (L=1 treated as missing)", file=sys.stderr)


if __name__ == "__main__":
    main()
