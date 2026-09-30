"""Independent recomputation of headline stats from fits_by_gauge.csv (no import of summarise.py)."""
import numpy as np, pandas as pd
from scipy.stats import binomtest
R = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve'
WT = '/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options'
f = pd.read_csv(f'{R}/fits_by_gauge.csv', dtype={'STAID': str}).set_index('STAID')
sg = pd.read_csv(f'{WT}/experiments/reservoir/smoke/smoke_gauges.csv', dtype={'STAID': str, 'control_for': str, 'huc2': str}).set_index('STAID')
ef = pd.read_csv(f'{WT}/experiments/reservoir/smoke/expected_release_fit.csv', dtype={'STAID': str}).set_index('STAID')
dam = sg[sg.role == 'dam']; ctl = sg[sg.role == 'control']
c_of = {v: k for k, v in ctl.control_for.items()}
onr = ef.on_reach.astype(str) == 'True'
sel = [s for s in dam.index if onr[s] and dam.nid_dor[s] > 0.5]
cs = [c_of[s] for s in sel]


def boot(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = np.median(d[idx], axis=1)
    return np.median(d), np.percentile(bs, [2.5, 97.5])


for law in ['L1', 'L4']:
    gd = f.loc[sel, f'{law}_nse'].values - f.loc[sel, 'L0_nse'].values
    gc = f.loc[cs, f'{law}_nse'].values - f.loc[cs, 'L0_nse'].values
    did = gd - gc
    for lab, x in [('dam', gd), ('ctl', gc), ('DiD', did)]:
        m, ci = boot(x); up = (x > 1e-9).sum(); dn = (x < -1e-9).sum()
        print(law, lab, f'{m:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] {up}/{dn} p={binomtest(up, up + dn).pvalue:.2g}')
    tr = f.loc[sel, f'{law}_train_nse'] - f.loc[sel, 'L0_train_nse']
    print(law, 'train dNSE median', round(tr.median(), 4))
x = f.loc[sel, 'L4_nse'] - f.loc[sel, 'L1_nse']; m, ci = boot(x); print('L4-L1 paired', f'{m:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]')
gd = f.loc[sel, 'L4_nse'].values - f.loc[sel, 'L0_nse'].values
print('mean dam gain', gd.mean(), 'clip', (f.loc[sel, 'L4_nse'].clip(lower=-1) - f.loc[sel, 'L0_nse'].clip(lower=-1)).mean())
pd.Series(sel).to_csv('/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve_audit/sel117.csv', index=False)
pd.DataFrame({'dam': sel, 'ctl': cs}).to_csv('/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve_audit/pairs117.csv', index=False)
