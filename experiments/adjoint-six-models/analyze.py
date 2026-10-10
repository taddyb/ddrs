#!/usr/bin/env python
"""E-09 analysis: celerity, transfer coefficient, inherited share, FD gate across the six arms.

Usage (ddr venv, for xarray/pandas):
    /home/tbindas/projects/ddr/.venv/bin/python experiments/adjoint-six-models/analyze.py \
        <out-dir> [<out-dir> ...]   # several dirs are merged (arms are disjoint)

Reuses experiments/adjoint/plots.py (load, per_gauge_celerity, inherited_decomposition).
Writes results.md and the csv tables next to the first out-dir under analysis/.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adjoint"))
import plots  # noqa: E402

OLD41 = Path("/home/tbindas/projects/ddrs/.ddrs/experiments/adjoint-conus/2026-09-07T18-18-39Z/gauges.csv")
ORDER = ["uh-fixed-channel", "dhbv2-lumped", "dhbv2-distributed", "hydrodl-lstm", "nh-daily-lstm", "nh-hourly-mts-lstm"]


def main():
    outs = [Path(p) for p in sys.argv[1:]]
    data, arms, val = {}, [], []
    for o in outs:
        man, a, gauges, d = plots.load(o)
        arms += [x for x in a if any(k[0] == x for k in d)]
        data.update(d)
        val += man["validation"]
    arms = [a for a in ORDER if a in arms]
    ana = outs[0] / "analysis"
    ana.mkdir(exist_ok=True)
    lines = []
    P = lines.append

    # FD gate
    P("## FD gate (+5 percent inflow at three reaches, tolerance 5 percent)\n")
    P("| arm | gauge | passed | max rel err (gauge / median / farthest reach) |")
    P("|---|---|---|---|")
    seen = {}
    for v in val:
        e = [c["rel_err"] for c in v["checks"]]
        if v["arm"] in seen:
            continue
        seen[v["arm"]] = 1
        P(f"| {v['arm']} | {v['staid']} | {v['passed']} | {max(e)*100:.3f} % ({', '.join(f'{x*100:.3f}' for x in e)}) |")
    P("")

    cel = plots.per_gauge_celerity(arms, gauges, data)
    dec = plots.inherited_decomposition(arms, gauges, data)
    cel.to_csv(ana / "celerity.csv", index=False)
    dec.to_csv(ana / "inherited.csv", index=False)
    old = set(pd.read_csv(OLD41, dtype=str).staid)
    sets = {"all": set(gauges.staid), "old41_overlap": set(gauges.staid) & old}
    for name, s in sets.items():
        P(f"## Population `{name}`: {len(s)} gauges\n")
        c = cel[cel.staid.isin(s)]
        P("Effective celerity per gauge (origin fit of kernel mean lag vs distance), median over gauges, m/s:\n")
        P("| arm | n gauges (low) | low-flow median | high-flow median | transfer-proxy: median wet volume sens |")
        P("|---|---|---|---|---|")
        for a in arms:
            lo = c[(c.arm == a) & (c.kind == "low")]
            hi = c[(c.arm == a) & (c.kind == "high")]
            P(f"| {a} | {len(lo)} | {lo.celerity_m_s.median():.3f} | {hi.celerity_m_s.median():.3f} | {lo.volume_sens_median.median():.3f} |")
        P("")
        for kind in ["low", "high"]:
            w = c[c.kind == kind].pivot(index="staid", columns="arm", values="celerity_m_s")[arms].dropna()
            if w.empty:
                continue
            ratio = w.max(axis=1) / w.min(axis=1)
            slow = w.idxmin(axis=1).value_counts().reindex(arms).fillna(0).astype(int)
            fast = w.idxmax(axis=1).value_counts().reindex(arms).fillna(0).astype(int)
            P(f"{kind}-flow cross-arm max/min celerity ratio over {len(w)} gauges with all {len(arms)} arms: median {ratio.median():.2f}, IQR {ratio.quantile(.25):.2f} to {ratio.quantile(.75):.2f}")
            P(f"  slowest-arm counts: {dict(slow)}")
            P(f"  fastest-arm counts: {dict(fast)}\n")
            # ratio among the five AORC-or-UH vs excluding hourly
            sub = [a for a in arms if a != "nh-hourly-mts-lstm"]
            if len(sub) < len(arms):
                r2 = (w[sub].max(axis=1) / w[sub].min(axis=1))
                P(f"  without the hourly arm ({len(sub)} arms): median ratio {r2.median():.2f}\n")
        d = dec[dec.staid.isin(s)]
        if not d.empty:
            share = (d.assign(a=d.inherited.abs(), t=d.total.abs()).groupby(["arm", "staid"], as_index=False)
                     .agg(a=("a", "sum"), t=("t", "sum"), tr=("transfer_mean", "first")))
            share["share"] = share.a / share.t.replace(0, np.nan)
            P("Downstream gauges: inherited share of bias and volume transfer coefficient (upstream gauge reach seen at downstream gauge):\n")
            P("| arm | n | inherited share median (IQR) | transfer coefficient median (IQR) |")
            P("|---|---|---|---|")
            for a in arms:
                x = share[share.arm == a]
                if x.empty:
                    continue
                P(f"| {a} | {len(x)} | {x.share.median():.2f} ({x.share.quantile(.25):.2f} to {x.share.quantile(.75):.2f}) | {x.tr.median():.3f} ({x.tr.quantile(.25):.3f} to {x.tr.quantile(.75):.3f}) |")
            P("")
    (ana / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
