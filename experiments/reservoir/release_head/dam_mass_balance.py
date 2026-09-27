#!/usr/bin/env python
"""Storage balance of the ddrs dam row at the on-reach check-2 dams.

For an on-reach dam with no other table dam upstream, the gauge series IS the dam row's outflow, and the dam's
inflow in ddrs equals the no-dam run's routed flow at that reach up to the replaced reach's own storage and clamp
effects. Over the WY1983-2010 window:

  balance = (sum outflow + dS) / sum inflow,   dS = T_end·Q_end − T_start·Q_start (daily means, T from the law)

should be 1 wherever the no-dam reach itself conserves mass. Reports the plain ratio (sum out / sum in) too.

Usage: dam_mass_balance.py <no-dam pred zarr> <ddrs check-2 zarr> <fit_dams.csv> <checks_1_2 check2.csv> <out csv>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import zarr


def load(p):
    z = zarr.open(p, mode="r")
    return pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]), \
        pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]")), z["predictions"][:]


def main():
    ref_p, ddrs_p, fit_p, c2_p, out_p = sys.argv[1:6]
    ids, t, ref = load(ref_p)
    _, _, dd = load(ddrs_p)
    fits = pd.read_csv(fit_p, dtype={"STAID": str}).set_index("STAID")
    c2 = pd.read_csv(c2_p, dtype={"staid": str}).set_index("staid")
    wy = np.asarray(t.year + (t.month >= 10))
    win = np.flatnonzero(wy >= 1983)
    w = 2 * np.pi * np.asarray(t.dayofyear) / 365.25
    rows = []
    for s, f in fits.iterrows():
        i = ids.get_loc(s)
        q_in, q_out = ref[i][win].astype(float), dd[i][win].astype(float)
        T = np.maximum(f.T_days * np.exp(f.a * np.sin(w[win]) + f.b * np.cos(w[win])), 1 / 24)
        ds = T[-1] * q_out[-1] - T[0] * q_out[0]  # m3/s·day
        s_in, s_out = q_in.sum(), q_out.sum()  # m3/s·day
        floor_days = int((ref[i] <= 1.001e-4).sum())
        rows.append(dict(staid=s, T0=f.T_days, a=f.a, b=f.b, cascade=bool(c2.loc[s, "cascade"]),
                         ratio=s_out / s_in, balance=(s_out + ds) / s_in, nodam_floor_days=floor_days))
    df = pd.DataFrame(rows)
    df.to_csv(out_p, index=False)
    for name, g in [("all 202", df), ("isolated", df[~df.cascade]),
                    ("isolated, no floor days", df[~df.cascade & (df.nodam_floor_days == 0)])]:
        print(f"{name:>24}: n {len(g):3d}  ratio median {g.ratio.median():.4f} [{g.ratio.min():.4f}, {g.ratio.max():.4f}]"
              f"  balance median {g.balance.median():.4f} [{g.balance.min():.4f}, {g.balance.max():.4f}]"
              f"  |balance-1|<1e-3: {int(((g.balance - 1).abs() < 1e-3).sum())}")
    print(df.reindex(df.balance.sub(1).abs().sort_values(ascending=False).index).head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
