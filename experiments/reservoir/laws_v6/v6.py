"""laws_v6 library: data, laws, fitting, metrics (offline, daily, smoke set routed no-dam flow).

Protocol as experiments/reservoir/rulecurve_offline/rc.py: fit on WY1983-1995 (WY1982 spin-up), score WY1996-2010.
prep / metrics / bands are copied from rc.py (that module loads a .so that is not checked in).

Laws (all followed by the per-dam linear bucket T0; recursion in laws.c):
  L0            pass-through (no dam)
  L2            per-dam bucket T0
  L4            L2 + harmonic rule curve flux (rc.py)
  FA / FB / FC  L2 + flood pool: capture phi*(I - Qc)+ ; evacuation to the release target Qe = Qc (FA, pool
                capacity Fmax), linear drain F/Te gated off during floods (FB), or min(linear, target headroom) (FC)
  W1 / W2 / W3  L2 + prescribed withdrawal W*_t = w * Ibar * s(doy), s mean 1 over the year:
                W1 smooth May-Sep window (sin^2 from doy 121 to 274), W2 the training-year inflow climatology inside
                Apr-Oct (divert a share of what arrives in season), W3 a window with per-dam centre and width
  K*            runoff rescaling nulls on the no-dam flow: per gauge (Kg), per HUC2 (Kh), global (Kc);
                with a bucket: L2Kg / L2Kh / L2Kc (T0 refitted on the scaled flow)
"""
import ctypes
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
AG = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4"
PRED = f"{WT}/output/reservoir_smoke/pred_1981_2010.zarr"
SMOKE = f"{WT}/experiments/reservoir/smoke/"
RCFITS = f"{AG}/experiments/reservoir/rulecurve_offline/fits_by_gauge.csv"
POP = f"{WT}/experiments/reservoir/full_run/paired_full_run.csv"

_L = ctypes.CDLL(os.path.join(HERE, "liblaws.so"))
_L.law.restype = ctypes.c_double
_L.law.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_double, ctypes.c_int] + [ctypes.c_double] * 5
                   + [ctypes.c_int] + [ctypes.c_void_p] * 8)

T_MIN, T_MAX = 0.05, 365.0
TGRID = np.logspace(np.log10(T_MIN), np.log10(T_MAX), 40)
BANDS = [(0, 7), (7, 30), (30, 120), (120, 400), (400, 1e12)]
BAND_LAB = ["lt7", "7_30", "30_120", "120_400", "gt400"]


def _p(a):
    return None if a is None else a.ctypes.data


def sim(g, n=None, T=T_MIN, mode=-1, Qc=0.0, phi=0.0, Fmax=0.0, Qe=0.0, Te=1.0, r=None, Wstar=None, I=None,
        want=False, floor=1):
    """Run laws.c::law on the first n days. Returns SSE over the training mask (n <= n_fit) or, with want, the
    series dict."""
    n = g["n"] if n is None else n
    I = g["I"] if I is None else I
    I = np.ascontiguousarray(I[:n], dtype=np.float64)
    r = None if r is None else np.ascontiguousarray(r[:n], dtype=np.float64)
    W = None if Wstar is None else np.ascontiguousarray(Wstar[:n], dtype=np.float64)
    if want:
        out = {k: np.empty(n) for k in ("Q", "C", "E", "W", "F")}
        fl = np.empty(n, dtype=np.uint8)
        _L.law(_p(I), _p(r), _p(W), n, T, mode, Qc, phi, Fmax, Qe, Te, floor, None, None,
               *[_p(out[k]) for k in ("Q", "C", "E", "W", "F")], _p(fl))
        out["fl"] = fl
        return out
    return _L.law(_p(I), _p(r), _p(W), n, T, mode, Qc, phi, Fmax, Qe, Te, floor, _p(g["obs0"]), _p(g["mtr"]),
                  None, None, None, None, None, None)


# ------------------------------------------------------------------ data
def load():
    import zarr
    z = zarr.open(PRED, mode="r")
    ids = pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]])
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    P, O = z["predictions"][:].astype(float), z["observations"][:].astype(float)
    wy = np.asarray(t.year + (t.month >= 10))
    train, test = (wy >= 1983) & (wy <= 1995), (wy >= 1996) & (wy <= 2010)
    return dict(ids=ids, t=t, P=P, O=O, train=train, test=test, doy=np.asarray(t.dayofyear), wy=wy)


def gauges():
    sg = pd.read_csv(SMOKE + "smoke_gauges.csv", dtype={"STAID": str, "control_for": str, "huc2": str})
    fit = pd.read_csv(SMOKE + "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID")
    sg["on_reach"] = sg.STAID.map(fit.on_reach.astype(str) == "True").fillna(False).astype(bool)
    dam = sg[sg.role == "dam"].set_index("STAID")
    ctl = sg[sg.role == "control"].set_index("STAID")
    # a control inherits its dam's attributes for grouping (dor, purpose, on_reach)
    for c in ["nid_dor", "dam_purpose", "on_reach", "dam_storage_mcm", "dam_max_discharge_m3s"]:
        ctl[c + "_pair"] = ctl.control_for.map(dam[c])
    return sg, dam, ctl


def prep(I, O, doy, train, test):
    I = np.where(np.isfinite(I), I, np.nanmean(I[train])).astype(np.float64)
    O = O.astype(np.float64)
    n = len(I)
    n_fit = int(np.flatnonzero(train)[-1]) + 1
    mtr = (train & np.isfinite(O)).astype(np.uint8)
    obs0 = np.where(np.isfinite(O), O, 0.0)
    w = 2 * np.pi * doy / 365.25
    H = np.stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    di = np.minimum(doy, 365) - 1
    clim = np.bincount(di[train], weights=I[train], minlength=365) / np.maximum(np.bincount(di[train], minlength=365), 1)
    C = np.convolve(np.concatenate([clim[-15:], clim, clim[:15]]), np.ones(31) / 31.0, mode="valid")
    Ibar = float(I[train].mean())
    otr = O[mtr.astype(bool)]
    den = float(((otr - otr.mean()) ** 2).sum())
    return dict(I=I, O=O, obs0=obs0, n=n, n_fit=n_fit, mtr=mtr, H=H, di=di, doy=doy, C=C, Ibar=Ibar, den=den,
                train=train, test=test)


# ------------------------------------------------------------------ withdrawal shapes (annual mean 1)
def _norm(s):
    return s / s.mean()


def shape_W1(g):
    d = g["di"] + 1.0
    s = np.where((d >= 121) & (d <= 274), np.sin(np.pi * (d - 121) / 153.0) ** 2, 0.0)
    return _norm_days(s, g)


def shape_W2(g):
    d = g["di"]
    win = ((d + 1) >= 91) & ((d + 1) <= 304)
    s = np.where(win, g["C"][d], 0.0)
    return _norm_days(s, g)


def shape_W3(g, centre, width):
    d = g["di"] + 1.0
    u = (np.mod(d - centre + 182.5, 365.0) - 182.5) / width  # circular distance in widths
    s = np.where(np.abs(u) < 0.5, np.cos(np.pi * u) ** 2, 0.0)
    return _norm_days(s, g)


def _norm_days(s, g):
    """Normalise so that the mean over a (training) year of days is 1."""
    m = s[g["train"]].mean()
    return s / m if m > 0 else s


# ------------------------------------------------------------------ metrics
def metrics(p, o, m):
    m = m & np.isfinite(o) & np.isfinite(p)
    p, o = p[m], o[m]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    al, be = p.std() / o.std(), p.mean() / o.mean()
    return dict(nse=float(nse), kge=float(1 - np.sqrt((r - 1) ** 2 + (al - 1) ** 2 + (be - 1) ** 2)), r=float(r),
                alpha=float(al), beta=float(be))


def bands(Q, O, test):
    e = Q[test] - O[test]
    o = O[test]
    m = np.isfinite(e)
    if m.sum() < 0.8 * len(e):
        return [np.nan] * len(BANDS)
    x = np.arange(len(e))
    e = np.interp(x, x[m], e[m])
    var = np.nanvar(o)
    F = np.fft.rfft(e)
    f = np.fft.rfftfreq(len(e))
    per = np.where(f > 0, 1 / np.maximum(f, 1e-12), 1e12)
    pw = np.abs(F) ** 2
    pw[1:] *= 2
    mse = (e ** 2).mean()
    return [float(mse * pw[(per >= lo) & (per < hi)].sum() / pw.sum() / var) for lo, hi in BANDS]


def score(g, name, out, extra=None):
    Q = out["Q"]
    r = {f"{name}_{k}": v for k, v in metrics(Q, g["O"], g["test"]).items()}
    r[f"{name}_train_nse"] = metrics(Q, g["O"], g["train"])["nse"]
    for lab, v in zip(BAND_LAB, bands(Q, g["O"], g["test"])):
        r[f"{name}_band_{lab}"] = v
    te = g["test"]
    Isum = g["I"][te].sum()
    r[f"{name}_cap_share"] = float(out["C"][te].sum() / Isum)       # volume routed through the flood pool
    r[f"{name}_w_share"] = float(out["W"][te].sum() / Isum)         # volume withdrawn (leaves the system)
    r[f"{name}_floor_test"] = float(out["fl"][te].mean())
    r[f"{name}_Fmax_days"] = float(out["F"][te].max() / g["Ibar"]) if g["Ibar"] > 0 else np.nan
    if extra:
        for k, v in extra.items():
            r[f"{name}_p_{k}"] = v
    return r


# ------------------------------------------------------------------ fitting helpers
def train_nse_sse(g, sse):
    return 1.0 - sse / g["den"]


def nm(obj, x0s, bounds, maxfev):
    best = None
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    for x0 in x0s:
        x0 = np.clip(np.asarray(x0, float), lo, hi)
        r = minimize(obj, x0, method="Nelder-Mead", bounds=bounds,
                     options=dict(maxfev=maxfev, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(best.fun)


def fit_T0(g, I=None, r=None, Wstar=None):
    """Plain bucket T0 (grid + NM on log T0)."""
    n = g["n_fit"]
    sse = [sim(g, n, T=T0, I=I, r=r, Wstar=Wstar) for T0 in TGRID]
    k = int(np.argmin(sse))
    x, f = nm(lambda x: sim(g, n, T=float(np.exp(x[0])), I=I, r=r, Wstar=Wstar), [[np.log(TGRID[k])]],
              [(np.log(T_MIN), np.log(T_MAX))], 200)
    return float(np.exp(x[0])) if f <= sse[k] else float(TGRID[k])


KC_GRID = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0]
PHI_GRID = [0.5, 1.0]
TE_GRID = [3.0, 10.0, 30.0, 100.0, 300.0]


def flood_kw(g, mode, lT0, lkc, phi, lz):
    Qc = float(np.exp(lkc)) * g["Ibar"]
    z = float(np.exp(lz))
    if mode == 0:
        return dict(T=float(np.exp(lT0)), mode=0, Qc=Qc, phi=phi, Fmax=z * g["Ibar"], Qe=Qc, Te=1.0)
    return dict(T=float(np.exp(lT0)), mode=mode, Qc=Qc, phi=phi, Fmax=0.0, Qe=Qc, Te=z)


def fit_flood(g, mode, T0_start, r=None, Wstar=None, I=None):
    n = g["n_fit"]
    cells = []
    for T0 in sorted({T0_start, T_MIN}):
        for kc in KC_GRID:
            for phi in PHI_GRID:
                for z in TE_GRID:
                    kw = flood_kw(g, mode, np.log(T0), np.log(kc), phi, np.log(z))
                    cells.append((sim(g, n, r=r, Wstar=Wstar, I=I, **kw), [np.log(T0), np.log(kc), phi, np.log(z)]))
    cells.sort(key=lambda c: c[0])
    bounds = [(np.log(T_MIN), np.log(T_MAX)), (np.log(0.3), np.log(40.0)), (0.0, 1.0), (np.log(0.5), np.log(3650.0))]
    x, f = nm(lambda x: sim(g, n, r=r, Wstar=Wstar, I=I, **flood_kw(g, mode, *x)), [c[1] for c in cells[:3]],
              bounds, 1600)
    return x


def withdraw_star(g, shape, w):
    return w * g["Ibar"] * shape


def fit_withdraw(g, shape, T0_start, I=None, r=None, flood=None):
    """Fit (log T0, w) with a fixed shape."""
    n = g["n_fit"]
    fk = flood or {}
    cells = []
    for T0 in sorted({T0_start, T_MIN}):
        for w in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
            kw = dict(fk)
            kw["T"] = T0
            cells.append((sim(g, n, I=I, r=r, Wstar=withdraw_star(g, shape, w), **kw), [np.log(T0), w]))
    cells.sort(key=lambda c: c[0])

    def obj(x):
        kw = dict(fk)
        kw["T"] = float(np.exp(x[0]))
        return sim(g, n, I=I, r=r, Wstar=withdraw_star(g, shape, x[1]), **kw)

    x, f = nm(obj, [c[1] for c in cells[:2]], [(np.log(T_MIN), np.log(T_MAX)), (0.0, 2.0)], 400)
    return x


def fit_withdraw3(g, T0_start):
    """W3: (log T0, w, centre, width)."""
    n = g["n_fit"]
    cells = []
    for c in [120, 165, 210, 255]:
        for wd in [60, 120, 200]:
            s = shape_W3(g, c, wd)
            for w in [0.05, 0.15, 0.35]:
                cells.append((sim(g, n, T=T0_start, Wstar=withdraw_star(g, s, w)), [np.log(T0_start), w, c, wd]))
    cells.sort(key=lambda c: c[0])

    def obj(x):
        return sim(g, n, T=float(np.exp(x[0])), Wstar=withdraw_star(g, shape_W3(g, x[2], x[3]), x[1]))

    x, f = nm(obj, [c[1] for c in cells[:3]],
              [(np.log(T_MIN), np.log(T_MAX)), (0.0, 2.0), (30.0, 335.0), (30.0, 300.0)], 1200)
    return x


def k_closed(I, O, m):
    """NSE-optimal scalar on the training mask."""
    return float((I[m] * O[m]).sum() / (I[m] ** 2).sum())


def l4_flux(g, c):
    return g["Ibar"] * (np.asarray(c) @ g["H"])


CK = ["c1s", "c1c", "c2s", "c2c"]


def fit_L4(g, I=None):
    """rc.py::fit_L4 ported onto laws.c: closed-form LS for the 4 harmonic coefficients on the unfloored bucket over
    the T0 grid, then Nelder-Mead on the floored training SSE from the best 3 cells. Coefficients in [-3, 3]."""
    n = g["n_fit"]
    I = g["I"] if I is None else I
    Ib = float(I[g["train"]].mean())
    m = g["mtr"][:n].astype(bool)
    y0 = g["obs0"][:n]
    cells = []
    for T0 in TGRID:
        lin = lambda x: sim(g, n, T=T0, I=x, want=True, floor=0)["Q"]
        y = lin(I) - y0
        Z = np.stack([Ib * lin(g["H"][j]) for j in range(4)])
        c = np.clip(np.linalg.lstsq(Z[:, m].T, y[m], rcond=None)[0], -3, 3)
        res = y[m] - c @ Z[:, m]
        cells.append((float(res @ res), T0, c))
    cells.sort(key=lambda c: c[0])

    def obj(x):
        return sim(g, n, T=float(np.exp(x[0])), I=I, r=Ib * (x[1:] @ g["H"][:, :n]))

    x, f = nm(obj, [np.concatenate([[np.log(c[1])], c[2]]) for c in cells[:3]],
              [(np.log(T_MIN), np.log(T_MAX))] + [(-3.0, 3.0)] * 4, 2000)
    return float(np.exp(x[0])), x[1:], Ib
