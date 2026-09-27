"""Item 1, empirical: perturb test-year observations and test-year inflow, refit with rc.py's own code, and check the
fitted parameters are bit-identical. (rc.py is imported here only to test IT; the reproduction does not use it.)"""
import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, '/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve')
import rc  # noqa: E402
import audit_impl as A  # noqa: E402

A.load()
t = pd.DatetimeIndex(pd.to_datetime(np.arange(len(A.G["train"])), unit='D', origin='1981-10-02'))
doy = np.asarray(t.dayofyear)
train, test = A.G["train"], A.G["test"]
nfit = A.G["nfit"]
print('n_fit', nfit, 'first test idx', int(np.flatnonzero(test)[0]), 'last train idx', int(np.flatnonzero(train)[-1]),
      'spin-up days', int(np.flatnonzero(train)[0]), 'train days', int(train.sum()), 'test days', int(test.sum()))
sel = pd.read_csv(f'{A.OUT}/sel117.csv', dtype=str).iloc[:, 0].tolist()
rng = np.random.default_rng(7)
ids = list(rng.choice(sel, 6, replace=False))
laws = ["L1", "L4"]
allsame = True
for s in ids:
    k = A.G["ids"][s]
    I, O = A.G["P"][k].copy(), A.G["O"][k].copy()
    g = rc.prep(I, O, doy, train, test)
    p0 = rc.fit_all(g, laws)
    O2 = O.copy(); O2[test] = rng.lognormal(0, 2, test.sum())  # garbage test obs
    I2 = I.copy(); I2[test] *= 3.0; I2[test] += 50.0          # very different test inflow
    g2 = rc.prep(I2, O2, doy, train, test)
    p2 = rc.fit_all(g2, laws)
    same = all(abs(p0[l][kk] - p2[l][kk]) == 0.0 for l in laws for kk in p0[l])
    allsame &= same
    print(s, 'params identical after perturbing test obs + test inflow:', same, '| Ibar', g['Ibar'], g2['Ibar'],
          '| L4', {kk: round(v, 4) for kk, v in p0['L4'].items()})
print('ALL IDENTICAL:', allsame)
# also: the doy used by rc (pandas dayofyear of the zarr time axis) equals the one reconstructed here
import zarr  # noqa: E402
z = zarr.open(f"{A.WT}/output/reservoir_smoke/pred_1981_2010.zarr", mode="r")
tz = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
print('time axis reconstructed == zarr:', bool((tz == t).all()))
