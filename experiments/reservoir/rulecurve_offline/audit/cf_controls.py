"""Median gain of baseline-skill-matched controls (the counterfactual 'control gain' under baseline matching)."""
import numpy as np
import pandas as pd

R = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve'
WT = '/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options'
OUT = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve_audit'
f = pd.read_csv(f'{R}/fits_by_gauge.csv', dtype={'STAID': str}).set_index('STAID')
p = pd.read_csv(f'{OUT}/pairs117.csv', dtype=str); D = p.dam.tolist()
sg = pd.read_csv(f'{WT}/experiments/reservoir/smoke/smoke_gauges.csv', dtype={'STAID': str}).set_index('STAID')
ca = sg.index[sg.role == 'control']
for law in ['L1', 'L2', 'L4']:
    cg = f.loc[ca, law + '_nse'] - f.loc[ca, 'L0_nse']
    for key in ['L0_nse', 'L0_train_nse']:
        c0 = f.loc[ca, key]
        nn = [cg[(c0 - f.loc[s, key]).abs().nsmallest(5).index].median() for s in D]
        used = set()
        for s in D:
            used |= set((c0 - f.loc[s, key]).abs().nsmallest(5).index)
        print(law, key, 'median counterfactual control gain %+.4f (unique controls used %d)' % (np.median(nn), len(used)))
