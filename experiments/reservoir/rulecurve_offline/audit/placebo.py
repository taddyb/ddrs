"""Freedom-only placebo: L4 with the annual harmonics replaced by non-seasonal ones (period 365.25*phi = 591 d
and its 2nd harmonic), same T0 + 4 coefficient search space, same fit (my TRF implementation), same scoring.
If the test-year gain of L4 were fitting freedom, this placebo would match it. Compare to L2 (T0 only)."""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import audit_impl as A

PER = 365.25 * (1 + 5 ** 0.5) / 2


def init():
    A.load()
    tt = np.arange(A.G["P"].shape[1], dtype=float)
    w = 2 * np.pi * tt / PER
    A.G["Hp"] = np.stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])


def job(s):
    k = A.G["ids"][s]
    I, O = A.G["P"][k], A.G["O"][k]
    train, test, nfit = A.G["train"], A.G["test"], A.G["nfit"]
    Ibar = float(I[train].mean())
    r = dict(STAID=s)
    # L2: T0 only (floor irrelevant since x = I >= 0)
    from scipy.optimize import minimize_scalar
    m = (train & np.isfinite(O))[:nfit]; o = O[:nfit][m]
    grid = np.logspace(np.log10(0.05), np.log10(365), 60)
    sse = [((A.lin_resp(I[:nfit], T)[m] - o) ** 2).sum() for T in grid]
    j = int(np.argmin(sse)); lo, hi = np.log(grid[max(j - 1, 0)]), np.log(grid[min(j + 1, 59)])
    rr = minimize_scalar(lambda lt: ((A.lin_resp(I[:nfit], np.exp(lt))[m] - o) ** 2).sum(), bounds=(lo, hi), method='bounded')
    T2 = float(np.exp(rr.x)) if rr.fun <= sse[j] else float(grid[j])
    q2 = A.lin_resp(I, T2)
    r.update(p_L2_nse=A.nse(q2, O, test), p_L2_train=A.nse(q2, O, train), p_L2_T0=T2)
    T0, c = A.fit_L4(I, O, A.G["Hp"], Ibar, train, nfit, floor=True)
    q, fl = A.L4_series(I, A.G["Hp"], Ibar, T0, c, True)
    r.update(p_PL_nse=A.nse(q, O, test), p_PL_train=A.nse(q, O, train), p_PL_T0=T0, p_PL_floor_test=float(fl[test].mean()))
    return r


if __name__ == "__main__":
    ids = pd.read_csv(sys.argv[1], dtype=str).iloc[:, 0].tolist()
    t0 = time.time()
    with Pool(8, initializer=init) as pool:
        rows = pool.map(job, ids, chunksize=1)
    pd.DataFrame(rows).set_index("STAID").to_csv(f"{A.OUT}/placebo_234.csv")
    print(f"{len(rows)} in {time.time() - t0:.0f} s")
