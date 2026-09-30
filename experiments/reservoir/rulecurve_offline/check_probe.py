import sys

import pandas as pd

d = pd.read_csv(sys.argv[1], dtype={'STAID': str}).set_index('STAID')
f = pd.read_csv('/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/expected_release_fit.csv',
                dtype={'STAID': str}).set_index('STAID')
j = d.join(f[['seas_T0', 'seas_a', 'seas_b', 'nse_pass_test']])
print('L1 param match', ((j.L1_p_T0 / j.seas_T0 - 1).abs() < 1e-9).mean(), ((j.L1_p_a - j.seas_a).abs() < 1e-9).mean(),
      ((j.L1_p_b - j.seas_b).abs() < 1e-9).mean(), 'L0 nse maxdiff', (j.L0_nse - j.nse_pass_test).abs().max())
cols = [c for c in d.columns if c.endswith('_nse') and 'train' not in c]
print('median dNSE test'); print((d[cols].sub(d.L0_nse, axis=0)).median().round(4).to_string())
tc = [c for c in d.columns if c.endswith('_train_nse')]
print('median dNSE train'); print((d[tc].sub(d.L0_train_nse, axis=0)).median().round(4).to_string())
print(d[[c for c in d.columns if 'floor_test' in c]].mean().round(4).to_string())
print(d[[c for c in d.columns if c.endswith('_p_T0')]].median().round(3).to_string())
print(d.secs.describe())
