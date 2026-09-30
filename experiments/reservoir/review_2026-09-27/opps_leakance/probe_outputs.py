import os, glob
import netCDF4
import pandas as pd

RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs"
for rid in ["2026-09-17T16-38-16Z-train-and-test", "2026-09-12T23-39-03Z-train-and-test",
            "2026-09-27T07-29-47Z-train-and-test", "2026-09-27T07-29-55Z-train-and-test"]:
    d = os.path.join(RUNS, rid)
    print("==", rid)
    for root, dirs, files in os.walk(d):
        depth = root[len(d):].count(os.sep)
        if depth > 2 or "checkpoints" in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            print("  ", p[len(d):], os.path.getsize(p))

for p in ["/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test/kan_parameters.nc",
          "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test/plot/kan_parameters.nc"]:
    try:
        ds = netCDF4.Dataset(p)
        print(p)
        print(" dims", {k: len(v) for k, v in ds.dimensions.items()})
        print(" vars", list(ds.variables))
        ds.close()
    except Exception as e:
        print(p, "ERR", e)

A = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/experiments/reservoir"
for f in [A + "/release_head/results/full_pairing.per_gauge.csv", A + "/smoke/smoke_gauges.csv",
          A + "/release_head/results/checks_4_5_fixed.per_gauge.csv"]:
    df = pd.read_csv(f)
    print(f, df.shape)
    print(" cols", list(df.columns)[:40])
