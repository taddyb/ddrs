"""Baseline-level confound of the DiD: match each dam gauge to controls with similar L0 test NSE."""
import numpy as np
import pandas as pd

OUT = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve_audit'
R = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve'
WT = '/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options'
f = pd.read_csv(f'{R}/fits_by_gauge.csv', dtype={'STAID': str}).set_index('STAID')
p = pd.read_csv(f'{OUT}/pairs117.csv', dtype=str); D = p.dam.tolist()
sg = pd.read_csv(f'{WT}/experiments/reservoir/smoke/smoke_gauges.csv', dtype={'STAID': str, 'control_for': str}).set_index('STAID')
ctl_all = sg.index[sg.role == 'control']
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x)
    bs = np.median(x[rng.integers(0, len(x), (10000, len(x)))], axis=1)
    return f'{np.median(x):+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] up {(x > 1e-9).sum()}/{len(x)}'


for law in ['L1', 'L4']:
    cg = f.loc[ctl_all, f'{law}_nse'] - f.loc[ctl_all, 'L0_nse']; c0 = f.loc[ctl_all, 'L0_nse']
    dg = f.loc[D, f'{law}_nse'] - f.loc[D, 'L0_nse']
    for k in [1, 3, 5, 9]:
        did = []
        for s in D:
            nn = (c0 - f.loc[s, 'L0_nse']).abs().nsmallest(k).index
            did.append(dg[s] - cg[nn].median())
        print(f'{law} DiD vs median of {k} nearest-L0-NSE controls: {ci(did)}')
    # same-HUC2 nearest-baseline
    did = []
    for s in D:
        pool = [c for c in ctl_all if sg.huc2[c] == sg.huc2[s]]
        if len(pool) < 3:
            pool = list(ctl_all)
        nn = (c0[pool] - f.loc[s, 'L0_nse']).abs().nsmallest(3).index
        did.append(dg[s] - cg[nn].median())
    print(f'{law} DiD vs median of 3 nearest-L0-NSE controls in same HUC2: {ci(did)}')
    # baseline-matching on KGE / on train-period NSE (pre-treatment, avoids test-period selection)
    c0t = f.loc[ctl_all, 'L0_train_nse']
    did = []
    for s in D:
        nn = (c0t - f.loc[s, 'L0_train_nse']).abs().nsmallest(3).index
        did.append(dg[s] - cg[nn].median())
    print(f'{law} DiD vs median of 3 nearest TRAIN-period-L0-NSE controls: {ci(did)}')
    # dam gauges with L0 NSE > 0.5 vs matched pairs: the high-baseline stratum
print('L0 test NSE of the 117 dams: quantiles', np.round(np.percentile(f.loc[D, 'L0_nse'], [10, 25, 50, 75, 90]), 3))
print('L0 test NSE of all 458 controls: quantiles', np.round(np.percentile(f.loc[ctl_all, 'L0_nse'], [10, 25, 50, 75, 90]), 3))
