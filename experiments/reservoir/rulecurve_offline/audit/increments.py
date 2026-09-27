"""Paired increments of the rule curve over the buckets, train vs test (their fits_by_gauge.csv and my fits)."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

OUT = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve_audit'
R = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve'
f = pd.read_csv(f'{R}/fits_by_gauge.csv', dtype={'STAID': str}).set_index('STAID')
b = pd.read_csv(f'{OUT}/placebo_234.csv', dtype={'STAID': str}).set_index('STAID')
p = pd.read_csv(f'{OUT}/pairs117.csv', dtype=str); D, C = p.dam.tolist(), p.ctl.tolist()
rng = np.random.default_rng(0)


def s(x):
    x = np.asarray(x, float)
    bs = np.median(x[rng.integers(0, len(x), (10000, len(x)))], axis=1)
    up, dn = int((x > 1e-6).sum()), int((x < -1e-6).sum())
    return (f'median {np.median(x):+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] '
            f'mean(clip +-1) {np.clip(x, -1, 1).mean():+.4f} up/down {up}/{dn} p={binomtest(up, up + dn).pvalue:.2g} '
            f'|x|<0.005: {int((np.abs(x) < 0.005).sum())}')


for role, ids in [('dam', D), ('ctl', C)]:
    print(f'--- {role} (n={len(ids)}), from fits_by_gauge.csv')
    for a_, b_ in [('L4', 'L2'), ('L4', 'L1'), ('L1', 'L2'), ('L2', 'L0'), ('L4', 'L0'), ('L1', 'L0')]:
        te = f.loc[ids, f'{a_}_nse'] - f.loc[ids, f'{b_}_nse']
        tr = f.loc[ids, f'{a_}_train_nse'] - f.loc[ids, f'{b_}_train_nse']
        print(f'{a_}-{b_} test : {s(te)}')
        print(f'{a_}-{b_} train: {s(tr)}')
    print(f'placebo-L2 test : {s(b.loc[ids, "p_PL_nse"] - b.loc[ids, "p_L2_nse"])}')
    print(f'placebo-L2 train: {s(b.loc[ids, "p_PL_train"] - b.loc[ids, "p_L2_train"])}')
