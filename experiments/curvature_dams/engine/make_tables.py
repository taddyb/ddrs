"""Phase 2 tables: the 69 on-reach flood-control dams of fixed_FA.csv with the pool size z set to a common
value (days of the gauge's offline mean inflow Ibar_offline, converted to the table's dam-inflow units exactly as
replay_flood_pool.py converts the fitted z), everything else (T_days = FA_p_T0, kc, phi) at the fitted values.
Also writes one config per table from the FA pool replay's snapshot config (run 2026-09-30T01-52-25Z)."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AG = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
src = pd.read_csv(os.path.join(AG, "experiments/reservoir/smoke/fixed_FA.csv"), dtype={"STAID": str})
cfg = open(os.path.join(AG, ".ddrs/runs/2026-09-30T01-52-25Z-train-and-test/config.yaml")).read()
assert "reservoirs: experiments/reservoir/smoke/fixed_FA.csv" in cfg
for zd in [0.05, 1.0, 5.0, 20.0]:
    t = src.copy()
    t["z"] = zd * t.Ibar_offline_m3s / t.inflow_mean_m3s
    t["z_offline_days"] = zd
    tag = f"z{zd:g}".replace(".", "p")
    tab = f"experiments/curvature_dams/engine/fixed_FA_{tag}.csv"
    t.to_csv(os.path.join(AG, tab), index=False)
    head = (f"# curvature_dams Phase 2 ({tag}): the FA pool replay (dam_release_smoke_replay_FA_pool.yaml) with every\n"
            f"# flood-pool dam's z set to {zd:g} d of the gauge's offline mean inflow; T0, kc, phi at the fitted values.\n"
            f"# Table {tab} (make_tables.py). Test phase only, zero-step resume. Inherited header below.\n")
    c = head + cfg.replace("reservoirs: experiments/reservoir/smoke/fixed_FA.csv", f"reservoirs: {tab}")
    open(os.path.join(AG, f"experiments/curvature_dams/engine/replay_FA_{tag}.yaml"), "w").write(c)
    print(tag, tab, t.z.describe()[["min", "50%", "max"]].round(3).to_dict())
