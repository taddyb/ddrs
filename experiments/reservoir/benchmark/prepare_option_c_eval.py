#!/usr/bin/env python
"""Inputs for the option C sandbox on the dam benchmark: a gauge CSV restricted to the benchmark
gauges and three eval configs derived from a trained run's config snapshot.

Host head: run 2026-09-12T23-39-03Z-train-and-test (sr_n0_gamma: n_0 + gamma, no leakance). The
benchmark's own numbers come from 2026-09-17T16-38-16Z (sr_n0_gamma_leakance), but
`validate_reservoirs` rejects use_reservoirs with use_leakance, so the reservoir arms need a head
trained without leakance; sr_n0_gamma is that run's matched control.

Arms, identical except for the reservoir table:
  off    use_reservoirs false
  fit    reservoirs_T_fit.csv    (44 dams with a ResOpsUS fit)
  prior  reservoirs_T_prior.csv  (all 121 dams; median fitted T where no fit)
Writes output/reservoir_benchmark/{gages_dam_benchmark.csv, eval_<arm>.yaml}.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-12T23-39-03Z-train-and-test")
GAGES = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
OUT = ROOT / "output" / "reservoir_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

bench = pd.read_csv(HERE / "dam_benchmark.csv", dtype={"gauge": str})
g = pd.read_csv(GAGES, dtype={"STAID": str})
sub = g[g.STAID.isin(set(bench.gauge))]
assert len(sub) == bench.gauge.nunique() == len(bench), (len(sub), bench.gauge.nunique())
sub.to_csv(OUT / "gages_dam_benchmark.csv", index=False)

cfg = (RUN / "config.yaml").read_text()
cfg, n = re.subn(r"(?m)^(\s*gages:\s*)\S+", lambda m: m.group(1) + str(OUT / "gages_dam_benchmark.csv"), cfg)
assert n == 1
for arm, table in [("off", None), ("fit", "reservoirs_T_fit.csv"), ("prior", "reservoirs_T_prior.csv")]:
    c = cfg
    if table:
        c, n = re.subn(r"(?m)^(\s*gages:\s*\S+\n)", lambda m: m.group(1) + f"  reservoirs: {HERE / table}\n", c)
        assert n == 1
        c, n = re.subn(r"(?m)^params:\n", "params:\n  use_reservoirs: true\n", c)
        assert n == 1
    (OUT / f"eval_{arm}.yaml").write_text(c)
print(f"{len(sub)} gauges -> {OUT}")
