"""Replay validation over the WY1996 overlap: how the replay (warm, started 1981-10-01) and the source run (cold start
1995-10-01) differ by date, and whether the difference decays like an initial-condition transient.
Usage: replay_check.py <source run> <replay run>. Writes replay_check_<replay[:20]>.txt."""
import sys

import numpy as np
import pandas as pd

import common as C

src, rep = sys.argv[1], sys.argv[2]
ids, tt, PT, OT = C.load_preds(src)
rid, tr, PR, OR = C.load_preds(rep)
PR = C.align(rid, PR, ids)
common_d = [d for d in tt if d in set(tr)]
it = np.array([tt.get_loc(d) for d in common_d])
ir = np.array([tr.get_loc(d) for d in common_d])
a, b = PT[:, it], PR[:, ir]
rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-3)
out = [f"overlap {common_d[0].date()}..{common_d[-1].date()} ({len(common_d)} d)"]
dates = pd.DatetimeIndex(common_d)
for d0 in ["1995-10-02", "1995-10-15", "1995-11-01", "1995-12-01", "1996-01-01", "1996-03-01", "1996-06-01", "1996-09-01"]:
    k = dates.get_indexer([pd.Timestamp(d0)], method="nearest")[0]
    out.append(f"  {d0}: median rel diff {np.nanmedian(rel[:, k]):.2e}, p90 {np.nanpercentile(rel[:, k], 90):.2e}, "
               f"p99 {np.nanpercentile(rel[:, k], 99):.2e}, share > 1e-3 {np.nanmean(rel[:, k] > 1e-3):.3f}")
# volume over the overlap after Jan 1
m = dates >= pd.Timestamp("1996-01-01")
va, vb = np.nansum(a[:, m], 1), np.nansum(b[:, m], 1)
r = vb / va
out.append(f"  Jan-Sep 1996 volume replay/source: median {np.nanmedian(r):.5f}, p1 {np.nanpercentile(r, 1):.4f}, "
           f"p99 {np.nanpercentile(r, 99):.4f}, share |r-1| > 1e-3 {np.nanmean(np.abs(r - 1) > 1e-3):.3f}")
na = C.metrics(a[:, m], OT[:, it][:, m], min_days=200).nse.values
nb = C.metrics(b[:, m], OT[:, it][:, m], min_days=200).nse.values
d = np.abs(na - nb)
out.append(f"  Jan-Sep 1996 NSE |source - replay|: median {np.nanmedian(d):.2e}, p99 {np.nanpercentile(d, 99):.2e}, "
           f"max {np.nanmax(d):.2e}, gauges > 0.01: {int(np.nansum(d > 0.01))}")
worst = np.argsort(-np.nan_to_num(d))[:5]
G = pd.read_csv(C.HERE / "gauges_table.csv", dtype={"STAID": str}).set_index("STAID").reindex(ids)
for w in worst:
    out.append(f"    {ids[w]} area {G.DRAIN_SQKM.iloc[w]:.0f} km2 NSE source {na[w]:.3f} replay {nb[w]:.3f}; "
               f"mean flow source {np.nanmean(a[w, m]):.3f} replay {np.nanmean(b[w, m]):.3f}")
open(C.HERE / f"replay_check_{rep[:20]}.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
