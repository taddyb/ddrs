"""Item 6/7: compare the independent fits with fits_by_gauge.csv; recompute headline stats on my fits."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import sys

OUT = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve_audit'
R = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve'
fn = sys.argv[1] if len(sys.argv) > 1 else 'audit_fits_234.csv'
a = pd.read_csv(f'{OUT}/{fn}', dtype={'STAID': str}).set_index('STAID')
f = pd.read_csv(f'{R}/fits_by_gauge.csv', dtype={'STAID': str}).set_index('STAID')
v = pd.read_csv(f'{R}/fits_variants.csv', dtype={'STAID': str}).set_index('STAID')
p = pd.read_csv(f'{OUT}/pairs117.csv', dtype=str)
if len(sys.argv) > 2:
    keep = pd.read_csv(sys.argv[2], dtype=str).iloc[:, 0]
    p = p[p.dam.isin(keep)]
D, C = p.dam.tolist(), p.ctl.tolist()


def boot(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    bs = np.median(d[rng.integers(0, len(d), (B, len(d)))], axis=1)
    up, dn = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    return f'{np.median(d):+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] {up}/{dn} p={binomtest(up, up + dn).pvalue:.2g}'


print(f'n pairs {len(D)}')
for lab, mine, theirs in [('L0', 'a_L0_nse', 'L0_nse'), ('L1', 'a_L1_nse', 'L1_nse'), ('L4', 'a_L4_nse', 'L4_nse')]:
    for role, ids in [('dam', D), ('ctl', C)]:
        d = a.loc[ids, mine] - f.loc[ids, theirs]
        print(f'{lab} {role}: my-their test NSE  median {d.median():+.2e}  mean {d.mean():+.2e}  max|.| {d.abs().max():.2e}  '
              f'|d|<1e-6 {int((d.abs() < 1e-6).sum())}/{len(d)}  |d|<0.01 {int((d.abs() < 0.01).sum())}/{len(d)}  corr {np.corrcoef(a.loc[ids, mine], f.loc[ids, theirs])[0, 1]:.4f}')
d = a.loc[D, 'a_L4lin_nse'] - v.loc[D, 'L4lin_nse']
print(f'L4lin dam: my-their median {d.median():+.2e} max|.| {d.abs().max():.2e}')
print('L1 params identical to theirs (dam+ctl):', int(((a.loc[D + C, 'a_L1_T0'] / f.loc[D + C, 'L1_p_T0'] - 1).abs() < 1e-9).sum()), 'of', len(D + C))
print()
for lab in ['L1', 'L4', 'L4lin']:
    gd = (a.loc[D, f'a_{lab}_nse'] - a.loc[D, 'a_L0_nse']).values
    gc = (a.loc[C, f'a_{lab}_nse'] - a.loc[C, 'a_L0_nse']).values
    print(f'MINE {lab:6s} dam {boot(gd)} | ctl {boot(gc)} | DiD {boot(gd - gc)}')
    tr = (a.loc[D, f'a_{lab}_train'] - a.loc[D, 'a_L0_train']).median()
    print(f'       train dNSE median dam {tr:+.4f}; dKGE dam {np.median(a.loc[D, f"a_{lab}_kge"] - a.loc[D, "a_L0_kge"]):+.4f}')
print('MINE L4 - L1 paired', boot(a.loc[D, 'a_L4_nse'] - a.loc[D, 'a_L1_nse']))
for lab in ['L1', 'L4']:
    gd = (f.loc[D, f'{lab}_nse'] - f.loc[D, 'L0_nse']).values
    gc = (f.loc[C, f'{lab}_nse'] - f.loc[C, 'L0_nse']).values
    print(f'THEIRS {lab:6s} dam {boot(gd)} | ctl {boot(gc)} | DiD {boot(gd - gc)}')
print('L1 scored without the T>=0.05 floor (consistent with its fit): dam', boot(a.loc[D, 'a_L1_nse_noTfloor'] - a.loc[D, 'a_L0_nse']),
      '| n changed', int(((a.loc[D + C, 'a_L1_nse_noTfloor'] - a.loc[D + C, 'a_L1_nse']).abs() > 1e-9).sum()))
print('train objective: my L4 train NSE minus theirs (dam) median %+.2e, share where mine is lower %.2f' % (
    (a.loc[D, 'a_L4_train'] - f.loc[D, 'L4_train_nse']).median(), ((a.loc[D, 'a_L4_train'] - f.loc[D, 'L4_train_nse']) < -1e-6).mean()))
print('floored share test (dam) mine mean %.3f median %.4f ; theirs mean %.3f median %.4f' % (
    a.loc[D, 'a_L4_floor_test'].mean(), a.loc[D, 'a_L4_floor_test'].median(), f.loc[D, 'L4_floor_test'].mean(), f.loc[D, 'L4_floor_test'].median()))
