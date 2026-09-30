"""Parse run.log of the four full-population arms: loss per optimizer step, release T0 median per micro-batch,
by epoch. Prints epoch tables and the loss difference learned minus off per epoch."""
import re
import sys
from collections import defaultdict

import numpy as np

RUNS = {
    "off42": "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-47Z-train-and-test/run.log",
    "learn42": "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-55Z-train-and-test/run.log",
    "off43": "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T10-31-30Z-train-and-test/run.log",
    "learn43": "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T10-31-50Z-train-and-test/run.log",
}
EPOCH = re.compile(r"\] epoch (\d+) lr=([0-9.e-]+)")
MICRO = re.compile(r"micro (\d+)/(\d+) gauges=(\d+) loss=([0-9.e+-]+) n=(\d+) median_n=([0-9.e+-]+) n_at_floor=([0-9.]+)%(?: release_T0_median=([0-9.]+)d \((\d+) dams\))?")
MB = re.compile(r"mb=(\d+) loss=([0-9.e+-]+) \(accumulated")


def parse(path):
    ep = 0
    lr = None
    micro = defaultdict(list)
    t0 = defaultdict(list)
    mb = defaultdict(list)
    lrs = {}
    with open(path) as f:
        for line in f:
            m = EPOCH.search(line)
            if m:
                ep = int(m.group(1))
                lrs[ep] = float(m.group(2))
                continue
            m = MICRO.search(line)
            if m:
                micro[ep].append(float(m.group(4)))
                if m.group(8):
                    t0[ep].append(float(m.group(8)))
                continue
            m = MB.search(line)
            if m:
                mb[ep].append(float(m.group(2)))
    return micro, t0, mb, lrs


data = {k: parse(v) for k, v in RUNS.items()}
print("epoch  lr     " + "  ".join(f"{k:>9}" for k in RUNS) + "   T0med(l42) T0med(l43) T0min-max(l42)")
for ep in range(1, 51):
    row = [f"{ep:5d}  {data['learn42'][3].get(ep, float('nan')):.4f}"]
    for k in RUNS:
        mbs = data[k][2].get(ep, [])
        row.append(f"{np.mean(mbs):9.4f}" if mbs else f"{'':>9}")
    t42 = data["learn42"][1].get(ep, [])
    t43 = data["learn43"][1].get(ep, [])
    row.append(f"   {np.median(t42):.3f}      {np.median(t43):.3f}     {min(t42):.3f}-{max(t42):.3f}" if t42 and t43 else "")
    print("  ".join(row))

print()
print("Loss learned minus off, mean over epochs 41-50: seed42 %.4f  seed43 %.4f" % (
    np.mean([np.mean(data['learn42'][2][e]) - np.mean(data['off42'][2][e]) for e in range(41, 51)]),
    np.mean([np.mean(data['learn43'][2][e]) - np.mean(data['off43'][2][e]) for e in range(41, 51)]),
))
print("Loss learned minus off, mean over epochs 1-10: seed42 %.4f  seed43 %.4f" % (
    np.mean([np.mean(data['learn42'][2][e]) - np.mean(data['off42'][2][e]) for e in range(1, 11)]),
    np.mean([np.mean(data['learn43'][2][e]) - np.mean(data['off43'][2][e]) for e in range(1, 11)]),
))
# T0 trajectory: fraction of final rise achieved by epoch
for k in ("learn42", "learn43"):
    t = data[k][1]
    med = np.array([np.median(t[e]) for e in range(1, 51)])
    print(k, "T0 median by epoch (d):", " ".join(f"{v:.3f}" for v in med))
    print(k, "epochs 1,5,10,20,21,30,40,41,50:", [round(med[i - 1], 3) for i in (1, 5, 10, 20, 21, 30, 40, 41, 50)])
# n and loss trend for the off arms
for k in RUNS:
    mbs = data[k][2]
    print(k, "mean step loss epochs 1-5 / 21-25 / 46-50: %.4f / %.4f / %.4f" % (
        np.mean([np.mean(mbs[e]) for e in range(1, 6)]),
        np.mean([np.mean(mbs[e]) for e in range(21, 26)]),
        np.mean([np.mean(mbs[e]) for e in range(46, 51)])))
