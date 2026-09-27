"""Baseline-matched DiD for the rule-curve increment (L4 - L2, L4 - L1), and the floor-heavy share of the gain."""
import numpy as np
import pandas as pd

OUT = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve_audit'
R = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve'
WT = '/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options'
f = pd.read_csv(f'{R}/fits_by_gauge.csv', dtype={'STAID': str}).set_index('STAID')
p = pd.read_csv(f'{OUT}/pairs117.csv', dtype=str); D, C = p.dam.tolist(), p.ctl.tolist()
sg = pd.read_csv(f'{WT}/experiments/reservoir/smoke/smoke_gauges.csv', dtype={'STAID': str, 'control_for': str}).set_index('STAID')
ctl_all = sg.index[sg.role == 'control']
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x)
    bs = np.median(x[rng.integers(0, len(x), (10000, len(x)))], axis=1)
    return f'{np.median(x):+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] up {(x > 1e-9).sum()}/{len(x)}'


for a_, b_ in [('L4', 'L2'), ('L4', 'L1'), ('L4', 'L0'), ('L1', 'L0'), ('L2', 'L0')]:
    cg = f.loc[ctl_all, f'{a_}_nse'] - f.loc[ctl_all, f'{b_}_nse']
    dg = f.loc[D, f'{a_}_nse'] - f.loc[D, f'{b_}_nse']
    pair = dg.values - (f.loc[C, f'{a_}_nse'] - f.loc[C, f'{b_}_nse']).values
    out = [f'{a_}-{b_}: matched-pair DiD {ci(pair)}']
    for key, lab in [('L0_nse', 'test-L0'), ('L0_train_nse', 'train-L0')]:
        c0 = f.loc[ctl_all, key]
        did = [dg[s] - cg[(c0 - f.loc[s, key]).abs().nsmallest(5).index].median() for s in D]
        out.append(f'5-NN on {lab} NSE {ci(did)}')
    print(' | '.join(out))

g4 = f.loc[D, 'L4_nse'] - f.loc[D, 'L0_nse']
heavy = f.loc[D, 'L4_floor_test'] > 0.2
print(f'dams with >20% floored test days: {int(heavy.sum())}; their median gain {g4[heavy].median():+.4f}; others {g4[~heavy].median():+.4f}')
print('purpose of heavy-floor dams:', sg.loc[np.array(D)[heavy.values], 'dam_purpose'].value_counts().to_dict())
print('huc2 of heavy-floor dams:', sg.loc[np.array(D)[heavy.values], 'huc2'].value_counts().to_dict())
