import re
import numpy as np
from collections import defaultdict
RUNS = {
    "off42": "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-47Z-train-and-test/run.log",
    "learn42": "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-55Z-train-and-test/run.log",
    "off43": "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T10-31-30Z-train-and-test/run.log",
    "learn43": "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T10-31-50Z-train-and-test/run.log",
}
EPOCH = re.compile(r"\] epoch (\d+) lr=")
MICRO = re.compile(r"micro (\d+)/(\d+) gauges=(\d+) loss=([0-9.e+-]+) n=(\d+) median_n=([0-9.e+-]+) n_at_floor=([0-9.]+)%")
for k, path in RUNS.items():
    ep = 0
    micro = defaultdict(list); medn = defaultdict(list); floor = defaultdict(list)
    allm = []
    for line in open(path):
        m = EPOCH.search(line)
        if m:
            ep = int(m.group(1)); continue
        m = MICRO.search(line)
        if m:
            micro[ep].append(float(m.group(4))); medn[ep].append(float(m.group(6))); floor[ep].append(float(m.group(7)))
            allm.append((ep, float(m.group(4))))
    a = np.array([v for _, v in allm])
    print(f"{k}: {len(a)} micro-batches; loss median {np.median(a):.3f}, p90 {np.percentile(a,90):.3f}, p99 {np.percentile(a,99):.3f}, >1: {(a>1).sum()}, >10: {(a>10).sum()}, >100: {(a>100).sum()}, max {a.max():.3g}")
    spikes = [(e, v) for e, v in allm if v > 5]
    print("   spikes >5:", [(e, round(v, 1)) for e, v in spikes][:20])
    blocks = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    print("   median micro loss by 10-epoch block:", [round(float(np.median([v for e in range(b0, b1 + 1) for v in micro[e]])), 4) for b0, b1 in blocks])
    print("   median_n by block (mean of micro medians):", [round(float(np.mean([v for e in range(b0, b1 + 1) for v in medn[e]])), 4) for b0, b1 in blocks])
    print("   n_at_floor by block:", [round(float(np.mean([v for e in range(b0, b1 + 1) for v in floor[e]])), 2) for b0, b1 in blocks])
    print("   median_n epochs 1,2,3,5,10,20,50:", [round(float(np.mean(medn[e])), 4) for e in (1, 2, 3, 5, 10, 20, 50)])
