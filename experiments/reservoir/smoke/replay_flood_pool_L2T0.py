#!/usr/bin/env python
"""The fair no-pool replay twin (review v6 MEDIUM 2).

fixed_FA.csv gives both replay arms the flood-pool fit's residence time (FA_p_T0), which is shorter than the plain
bucket's (median FA/L2 ratio 0.62), so "pool minus no-pool" in the engine is not the offline "FA minus L2". This table
routes the same 69 dams with the plain-bucket residence time L2_T0_days and no pool, so engine (pool - this) matches
the offline comparator. Writes fixed_FA_L2T0.csv next to this script.
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
t = pd.read_csv(HERE / "fixed_FA.csv", dtype={"STAID": str})
t["T_days"] = t["L2_T0_days"].clip(lower=1.0 / 24.0)
t["year_completed"] = t["year_completed"].astype("Int64")
t.to_csv(HERE / "fixed_FA_L2T0.csv", index=False)
print(len(t), "dams; T_days median", round(float(t.T_days.median()), 3), "d")
