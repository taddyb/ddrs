"""Item 5: bounds, concentration, train vs test, and control/baseline confound. Uses fits_by_gauge.csv."""
import numpy as np
import pandas as pd

OUT = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve_audit'
R = '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve'
WT = '/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options'
f = pd.read_csv(f'{R}/fits_by_gauge.csv', dtype={'STAID': str}).set_index('STAID')
p = pd.read_csv(f'{OUT}/pairs117.csv', dtype=str); D, C = p.dam.tolist(), p.ctl.tolist()
sg = pd.read_csv(f'{WT}/experiments/reservoir/smoke/smoke_gauges.csv', dtype={'STAID': str, 'control_for': str}).set_index('STAID')
CK = ['c1s', 'c1c', 'c2s', 'c2c']

print('== bounds (L4, T0 in [0.05, 365], coeffs in [-3, 3]) ==')
for lab, ids in [('dam117', D), ('ctl117', C)]:
    x = f.loc[ids]
    t_lo = (x.L4_p_T0 <= 0.0501).sum(); t_hi = (x.L4_p_T0 >= 364.9).sum()
    cb = (x[[f'L4_p_{k}' for k in CK]].abs() >= 2.999).sum(axis=1)
    print(f'{lab}: T0 at floor {t_lo}, at ceiling {t_hi}; gauges with >=1 coeff at +-3: {int((cb > 0).sum())} '
          f'(coeffs at bound total {int(cb.sum())} of {4 * len(ids)}); max |coeff| median {x[[f"L4_p_{k}" for k in CK]].abs().max(axis=1).median():.3f}')
    if (cb > 0).any():
        g = (x.L4_nse - x.L0_nse)[cb > 0]
        print('   gains at gauges with a coeff on bound:', g.round(3).tolist(), ' floor share test', x.L4_floor_test[cb > 0].round(2).tolist())

print('\n== concentration (dam117, L4 test dNSE) ==')
g = (f.loc[D, 'L4_nse'] - f.loc[D, 'L0_nse']).sort_values(ascending=False)
tot = g.sum()
for k in [1, 5, 10, 20]:
    print(f'top {k:2d} share of summed gain {g.iloc[:k].sum() / tot:.2f}; mean without top {k}: {g.iloc[k:].mean():+.4f}, median without top {k}: {g.iloc[k:].median():+.4f}')
print('top 10 gauges:', g.iloc[:10].round(3).to_dict())
print('gain quantiles (10,25,50,75,90):', np.round(np.percentile(g, [10, 25, 50, 75, 90]), 4))
print('share of gauges with gain > +0.01:', round((g > 0.01).mean(), 3), ' < -0.01:', round((g < -0.01).mean(), 3))
# leave-one-HUC2-out median
h = sg.loc[D, 'huc2'] if 'huc2' in sg else None
if h is not None:
    gg = f.loc[D, 'L4_nse'] - f.loc[D, 'L0_nse']
    lo = {hh: round(float(gg[h != hh].median()), 4) for hh in sorted(h.unique())}
    print('leave-one-HUC2-out median L4 gain: min %.4f max %.4f' % (min(lo.values()), max(lo.values())), ' counts', h.value_counts().to_dict())

print('\n== train vs test ==')
for lab in ['L1', 'L4']:
    for role, ids in [('dam', D), ('ctl', C)]:
        tr = f.loc[ids, f'{lab}_train_nse'] - f.loc[ids, 'L0_train_nse']
        te = f.loc[ids, f'{lab}_nse'] - f.loc[ids, 'L0_nse']
        print(f'{lab} {role}: median train gain {tr.median():+.4f} test gain {te.median():+.4f} ratio {te.median() / tr.median():.2f}; '
              f'spearman(train,test) {tr.corr(te, method="spearman"):.2f}; test<0 while train>0: {int(((te < 0) & (tr > 0)).sum())}')
tr4 = f.loc[D, 'L4_train_nse'] - f.loc[D, 'L1_train_nse']; te4 = f.loc[D, 'L4_nse'] - f.loc[D, 'L1_nse']
print(f'L4 minus L1 (dam): train {tr4.median():+.4f}, test {te4.median():+.4f}')

print('\n== control baseline confound: all 458 controls, L4 gain by L0 test NSE ==')
ctl_all = sg.index[sg.role == 'control']; dam_all = sg.index[sg.role == 'dam']
cg = f.loc[ctl_all, 'L4_nse'] - f.loc[ctl_all, 'L0_nse']; c0 = f.loc[ctl_all, 'L0_nse']
dg = f.loc[D, 'L4_nse'] - f.loc[D, 'L0_nse']; d0 = f.loc[D, 'L0_nse']
for lo_, hi_ in [(-99, 0), (0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1)]:
    mc = (c0 > lo_) & (c0 <= hi_); md = (d0 > lo_) & (d0 <= hi_)
    print(f'L0 NSE in ({lo_},{hi_}]: controls n={int(mc.sum()):3d} median L4 gain {cg[mc].median() if mc.any() else np.nan:+.4f} '
          f'(L1 {(f.loc[ctl_all, "L1_nse"] - c0)[mc].median() if mc.any() else np.nan:+.4f}) | dams117 n={int(md.sum()):3d} median {dg[md].median() if md.any() else np.nan:+.4f}')
print('matched-pair L0 NSE: dam median %.3f, control median %.3f' % (d0.median(), f.loc[C, 'L0_nse'].median()))

# Baseline-matched control: for each dam, the control (from ALL 458 controls, not used twice needed) with closest L0 test NSE
rng = np.random.default_rng(0)
cands = pd.DataFrame({'l0': c0, 'g': cg})
did_bm = []
for s in D:
    k = (cands.l0 - f.loc[s, 'L0_nse']).abs().nsmallest(5).index  # average of 5 nearest-baseline controls
    did_bm.append(dg[s] - cands.loc[k, 'g'].mean())
did_bm = np.array(did_bm)
bs = np.median(did_bm[rng.integers(0, len(did_bm), (10000, len(did_bm)))], axis=1)
print('DiD vs 5 nearest-L0-NSE controls (any HUC): median %+.4f [%+.4f, %+.4f] up %d/%d' % (
    np.median(did_bm), *np.percentile(bs, [2.5, 97.5]), (did_bm > 0).sum(), len(did_bm)))
# also: controls restricted to L0 NSE < 0.6
lowc = cands[cands.l0 < 0.6]
print('controls with L0 NSE<0.6: n=%d, median L4 gain %+.4f, median L1 gain %+.4f' % (len(lowc), lowc.g.median(),
      (f.loc[lowc.index, 'L1_nse'] - f.loc[lowc.index, 'L0_nse']).median()))
