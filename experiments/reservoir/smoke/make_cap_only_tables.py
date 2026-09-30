#!/usr/bin/env python
"""Cap-only control tables for the dam-row positivity cap (review v5 finding 1).

With `reservoir_dam_row_positivity: true` the cap X_eff = min(X_r, 0.5(1-0.01)dt/K_r) changes a dam reach's routing
even when the reservoir adds nothing. These tables arm the same dams with a one-hour residence time (the fixed-table floor) and
no rule curve, so a replay with them measures the cap alone:
  cap_only_202.csv  - the 202 dams of fixed_L2.csv / fixed_L4.csv (control for the v5 replays)
  cap_only_all.csv  - every dam of release_head/dam_features.csv (control for the S5 training arms)
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
T = 1.0 / 24.0  # the fixed-table reader floor (one routing step); the cap-off twin isolates this hour
l2 = pd.read_csv(HERE / "fixed_L2.csv", dtype={"STAID": str})
a = l2.assign(T_days=T, year_completed=l2.year_completed.astype("Int64"))
a.to_csv(HERE / "cap_only_202.csv", index=False)
f = pd.read_csv(HERE.parent / "release_head" / "dam_features.csv")
b = pd.DataFrame({"COMID": f.COMID, "T_days": T, "year_completed": f.year_completed.astype("Int64")})
b.to_csv(HERE / "cap_only_all.csv", index=False)
print(len(a), "and", len(b), "rows written")
