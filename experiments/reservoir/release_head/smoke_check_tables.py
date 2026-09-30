#!/usr/bin/env python
"""Fixed reservoir tables and eval configs for dam-release smoke checks 1 and 2.

Check 1 (pass-through): every smoke dam (smoke_dams.csv, NID >= 10 MCM in the smoke network) at T_days = 1/24,
a = b = 0, through the seasonal fixed path.
Check 2 (engine vs offline fit): each ON-REACH dam gauge's dam (dam COMID == gauge COMID) at its fitted seas_T0,
seas_a, seas_b from expected_release_fit.csv (fixed seasonal mode). Only on-reach dams go in the table: the offline
fit is per gauge on the no-dam routed flow, so a bucket at an off-reach dam upstream would change another gauge's dam
inflow in ddrs but not in the fit. A COMID shared by several gauges keeps the first gauge's fit (file order).

Writes to <out>/: passthrough_dams.csv, fit_dams.csv, and three test-phase configs derived from the no-dam smoke
eval config (the sr_n0_gamma recipe on gages_smoke.csv, testing window 1981-10-01..2010-09-30):
  check1_off.yaml          verbatim (use_reservoirs off)
  check1_passthrough.yaml  + use_reservoirs, passthrough_dams.csv
  check2_seasonal.yaml     + use_reservoirs, fit_dams.csv

Run: ~/projects/ddr/.venv/bin/python experiments/reservoir/release_head/smoke_check_tables.py <eval_smoke.yaml> <out>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SMOKE = HERE.parent / "smoke"


def main(base_cfg: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dams = pd.read_csv(SMOKE / "smoke_dams.csv")
    comids = sorted(set(int(c) for c in dams.COMID))
    pd.DataFrame({"COMID": comids, "T_days": "0.041666668", "a": 0.0, "b": 0.0}).to_csv(
        out / "passthrough_dams.csv", index=False)

    fit = pd.read_csv(SMOKE / "expected_release_fit.csv", dtype={"STAID": str})
    fit = fit[fit.role == "dam"].copy()
    fit["dam_COMID"] = fit.dam_COMID.astype(int)
    fit["on_reach"] = fit.on_reach.astype(str) == "True"
    fit = fit[fit.on_reach].drop_duplicates("dam_COMID").sort_values("dam_COMID")
    table = pd.DataFrame({
        "COMID": fit.dam_COMID,
        "T_days": fit.seas_T0.map(lambda v: f"{max(v, 1.0 / 24.0):.9g}"),
        "a": fit.seas_a,
        "b": fit.seas_b,
        "STAID": fit.STAID,
        "on_reach": fit.on_reach,
    })
    table.to_csv(out / "fit_dams.csv", index=False)

    text = base_cfg.read_text()
    (out / "check1_off.yaml").write_text(text)
    for name, csv in [("check1_passthrough.yaml", "passthrough_dams.csv"), ("check2_seasonal.yaml", "fit_dams.csv")]:
        t = text.replace("data_sources:\n", f"data_sources:\n  reservoirs: {out / csv}\n", 1)
        assert "\nparams:\n" in t
        t = t.replace("\nparams:\n", "\nparams:\n  use_reservoirs: true\n", 1)
        (out / name).write_text(t)
    print(f"passthrough dams: {len(comids)}; fit dams: {len(table)} ({int(table.on_reach.sum())} on-reach)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
