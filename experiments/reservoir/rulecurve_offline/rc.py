"""Rule-curve release laws, offline, on the smoke set's routed no-dam flow (library).

Bucket recursion: the harness's daily implicit Euler on storage with same-day inflow
(experiments/reservoir/smoke/expected_release_fit.py::simulate), in C (libbucket.so), with the outflow floored at
0 and the floor feeding back into storage (bucket.c). Affine storage law S = T Q + S0(doy) == bucket on
I' = I - r(t), r = dS0/dt a zero-annual-mean seasonal flux.

  L0   pass                      Q = I
  L1   seasonal-T bucket          harness grid (T0 40 log pts 0.05..1000 d, a, b in linspace(-2, 2, 9)), fitted with
                                  no T floor as expected_release_fit.py, scored with T >= 0.05 d as ceiling.py
  L2   plain bucket               T0 log grid 0.05..365 d, bounded refinement
  L3   flatten                    r = f (C(doy) - Cbar), C = 31-d circular MA of the TRAINING doy-mean of I; T0, f in [0, 1.5]
  L3b  flatten + phase            r = f (C(doy - phi) - Cbar), phi in [-90, 90] d
  L4   harmonic rule curve        r = Ibar sum_{k=1,2} (alpha_k sin kw + beta_k cos kw); T0 + 4 coefficients in [-3, 3]
  L4r  rule curve, no bucket      L4 with T0 fixed at 0.05 d (the bucket's floor): a pure seasonal flow correction
  L5   L4 + seasonal T            T_t = max(T0 exp(a sin w + b cos w), 0.05), a, b in [-2, 2]
Fit: training NSE on WY1983-1995 (WY1982 spin-up), closed-form least squares for the flux coefficients on the
unfloored (linear) bucket over a T0 [x phi | x a x b] grid, then Nelder-Mead on the floored training NSE from the
best 3 grid cells. Score WY1996-2010.
"""
import ctypes
import os

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = ctypes.CDLL(os.path.join(HERE, "libbucket.so"))
_LIB.bucket.restype = ctypes.c_double
_LIB.bucket.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

T_MIN, T_MAX = 0.05, 365.0
L1_TGRID = np.logspace(np.log10(0.05), np.log10(1000), 40)
L1_AB = np.linspace(-2.0, 2.0, 9)
TGRID = np.logspace(np.log10(T_MIN), np.log10(T_MAX), 40)
PHIGRID = np.arange(-90.0, 90.1, 7.5)
LAWS = ["L1", "L2", "L3", "L3b", "L4", "L4r", "L5"]
BANDS = [(0, 7), (7, 30), (30, 120), (120, 400), (400, 1e12)]
BAND_LAB = ["lt7", "7_30", "30_120", "120_400", "gt400"]


def _p(a):
    return None if a is None else a.ctypes.data


def cbucket(x, T, n, floor=True, obs=None, mask=None, want=False):
    """SSE over mask[:n] (if obs given); with want=True also the release series and the floored-day flags."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    T = np.ascontiguousarray(T, dtype=np.float64)
    Q = np.empty(n) if want else None
    fl = np.empty(n, dtype=np.uint8) if want else None
    ob = None if obs is None else np.ascontiguousarray(obs, dtype=np.float64)
    mk = None if mask is None else np.ascontiguousarray(mask, dtype=np.uint8)
    sse = _LIB.bucket(_p(x), _p(T), int(n), int(floor), _p(ob), _p(mk), _p(Q), _p(fl))
    return (sse, Q, fl) if want else sse


def prep(I, O, t_doy, train, test):
    """Per-gauge arrays. t_doy: pandas dayofyear per day."""
    I = np.where(np.isfinite(I), I, np.nanmean(I)).astype(np.float64)
    O = O.astype(np.float64)
    n = len(I)
    n_fit = int(np.flatnonzero(train)[-1]) + 1
    mtr = (train & np.isfinite(O)).astype(np.uint8)
    obs0 = np.where(np.isfinite(O), O, 0.0)
    w = 2 * np.pi * t_doy / 365.25
    H = np.stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    di = np.minimum(t_doy, 365) - 1
    clim = np.bincount(di[train], weights=I[train], minlength=365) / np.maximum(np.bincount(di[train], minlength=365), 1)
    k = np.ones(31) / 31.0
    C = np.convolve(np.concatenate([clim[-15:], clim, clim[:15]]), k, mode="valid")  # circular 31-d MA
    Cbar = float(C.mean())
    Ibar = float(I[train].mean())
    otr = O[mtr.astype(bool)]
    den = float(((otr - otr.mean()) ** 2).sum())
    return dict(I=I, O=O, obs0=obs0, n=n, n_fit=n_fit, mtr=mtr, sinw=np.sin(w), cosw=np.cos(w), H=H, di=di, C=C, Cbar=Cbar,
                Ibar=Ibar, den=den, train=train, test=test)


def clim_anom(g, phi):
    """C(doy - phi) - Cbar per day, linear interpolation on the circular climatology."""
    pos = np.mod(g["di"] - phi, 365.0)
    i0 = np.floor(pos).astype(int) % 365
    fr = pos - np.floor(pos)
    C = g["C"]
    return (1 - fr) * C[i0] + fr * C[(i0 + 1) % 365] - g["Cbar"]


VARIANTS = {"ec": 2, "lin": 0, "pen": 1}  # engine clamp (no feedback) / no floor / floor-with-feedback + penalty


def split(law):
    for suf in VARIANTS:
        if law.endswith(suf):
            return law[:-len(suf)], suf
    return law, ""


def mode_of(law):
    return VARIANTS.get(split(law)[1], 1)


def flux(g, law, p):
    law = split(law)[0]
    if law in ("L3", "L3b"):
        return p["f"] * clim_anom(g, p.get("phi", 0.0))
    if law in ("L4", "L4r", "L5"):
        c = np.array([p["c1s"], p["c1c"], p["c2s"], p["c2c"]])
        return g["Ibar"] * (c @ g["H"])
    return None


def Tseries(g, law, p, n):
    law = split(law)[0]
    if law in ("L1", "L5"):
        return np.maximum(p["T0"] * np.exp(p["a"] * g["sinw"][:n] + p["b"] * g["cosw"][:n]), 0.05)
    return np.full(n, p["T0"])


def run(g, law, p, n=None):
    """Full floored simulation; returns (Q, floored flags)."""
    n = g["n"] if n is None else n
    if law == "L0":
        return g["I"][:n].copy(), np.zeros(n, np.uint8)
    r = flux(g, law, p)
    x = g["I"][:n] - (r[:n] if r is not None else 0.0)
    _, Q, fl = cbucket(x, Tseries(g, law, p, n), n, mode_of(law), want=True)
    return Q, fl


def train_nse(g, law, p):
    n = g["n_fit"]
    r = flux(g, law, p)
    x = g["I"][:n] - (r[:n] if r is not None else 0.0)
    if split(law)[1] == "pen":  # training floored-day share held <= 1 %
        sse, _, fl = cbucket(x, Tseries(g, law, p, n), n, 1, g["obs0"], g["mtr"], want=True)
        fs = float(fl[g["train"][:n]].mean())
        return 1.0 - sse / g["den"] - 10.0 * max(0.0, fs - 0.01)
    return 1.0 - cbucket(x, Tseries(g, law, p, n), n, mode_of(law), g["obs0"], g["mtr"]) / g["den"]


def metrics(p, o, m):
    m = m & np.isfinite(o) & np.isfinite(p)
    p, o = p[m], o[m]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    al, be = p.std() / o.std(), p.mean() / o.mean()
    return dict(nse=float(nse), kge=float(1 - np.sqrt((r - 1) ** 2 + (al - 1) ** 2 + (be - 1) ** 2)), r=float(r),
                alpha=float(al), beta=float(be))


def bands(Q, O, test):
    """Test-period MSE split into FFT period bands, normalised by test obs variance (as tmp/bands.py)."""
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


# ------------------------------------------------------------------ fitting
def _lin(g, x, T, n):
    """Unfloored bucket response to x (linear operator), first n days."""
    return cbucket(x, T, n, False, want=True)[1]


def _ls(y, Z, m, lo=None, hi=None):
    """min ||y - Z c||^2 over mask m; returns c, sse. Single-column case clipped to [lo, hi]."""
    ym, Zm = y[m], Z[:, m]
    if Z.shape[0] == 1:
        zz = float(Zm[0] @ Zm[0])
        c = np.array([float(Zm[0] @ ym) / zz if zz > 0 else 0.0])
        if lo is not None:
            c = np.clip(c, lo, hi)
    else:
        c = np.linalg.lstsq(Zm.T, ym, rcond=None)[0]
        if lo is not None:
            c = np.clip(c, lo, hi)
    res = ym - c @ Zm
    return c, float(res @ res)


def _nm(g, law, keys, x0s, bounds, fixed):
    """Nelder-Mead on floored training NSE from several starts; x in (log T0 if present, ...)."""
    def unpack(x):
        p = dict(fixed)
        for k, v in zip(keys, x):
            p[k] = float(np.exp(v)) if k == "T0" else float(v)
        return p

    best = None
    for x0 in x0s:
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(lambda x: -train_nse(g, law, unpack(x)), x0, method="Nelder-Mead", bounds=bounds,
                     options=dict(maxfev=400 * len(keys), xatol=1e-4, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    return unpack(best.x), -float(best.fun)


def fit_L1(g):
    n = g["n_fit"]
    m = g["mtr"][:n].astype(bool)
    best = (np.inf, None)
    for T0 in L1_TGRID:
        for a in L1_AB:
            for b in L1_AB:
                T = T0 * np.exp(a * g["sinw"][:n] + b * g["cosw"][:n])  # no T floor while fitting (harness)
                sse = cbucket(g["I"][:n], T, n, False, g["obs0"], g["mtr"])
                if sse < best[0]:
                    best = (sse, dict(T0=float(T0), a=float(a), b=float(b)))
    return best[1]


def fit_L2(g):
    n = g["n_fit"]
    sse = [cbucket(g["I"][:n], np.full(n, T0), n, True, g["obs0"], g["mtr"]) for T0 in TGRID]
    k = int(np.argmin(sse))
    lo, hi = np.log(TGRID[max(k - 1, 0)]), np.log(TGRID[min(k + 1, len(TGRID) - 1)])
    r = minimize_scalar(lambda lt: cbucket(g["I"][:n], np.full(n, np.exp(lt)), n, True, g["obs0"], g["mtr"]),
                        bounds=(lo, hi), method="bounded", options=dict(xatol=1e-4))
    T0 = float(np.exp(r.x)) if r.fun <= sse[k] else float(TGRID[k])
    return dict(T0=T0)


def fit_L3(g, with_phi):
    n = g["n_fit"]
    m = g["mtr"][:n].astype(bool)
    phis = PHIGRID if with_phi else np.array([0.0])
    anoms = {phi: clim_anom(g, phi)[:n] for phi in phis}
    cells = []
    for T0 in TGRID:
        T = np.full(n, T0)
        y = _lin(g, g["I"][:n], T, n) - g["obs0"][:n]
        for phi in phis:
            Z = _lin(g, anoms[phi], T, n)[None]
            c, sse = _ls(y, Z, m, 0.0, 1.5)
            cells.append((sse, T0, c[0], phi))
    cells.sort(key=lambda c: c[0])
    keys = ["T0", "f", "phi"] if with_phi else ["T0", "f"]
    bounds = [(np.log(T_MIN), np.log(T_MAX)), (0.0, 1.5)] + ([(-90.0, 90.0)] if with_phi else [])
    x0s = [np.array([np.log(c[1]), c[2]] + ([c[3]] if with_phi else [])) for c in cells[:3]]
    p, tn = _nm(g, "L3b" if with_phi else "L3", keys, x0s, bounds, {} if with_phi else {"phi": 0.0})
    return p


CK = ["c1s", "c1c", "c2s", "c2c"]


def fit_L4(g, fixed_T0=None, var=""):
    n = g["n_fit"]
    m = g["mtr"][:n].astype(bool)
    cells = []
    for T0 in ([fixed_T0] if fixed_T0 else TGRID):
        T = np.full(n, T0)
        y = _lin(g, g["I"][:n], T, n) - g["obs0"][:n]
        Z = np.stack([g["Ibar"] * _lin(g, g["H"][j, :n], T, n) for j in range(4)])
        c, sse = _ls(y, Z, m, -3.0, 3.0)
        cells.append((sse, T0, c))
    cells.sort(key=lambda c: c[0])
    cb = [(-3.0, 3.0)] * 4
    if fixed_T0:
        p, _ = _nm(g, "L4r", CK, [cells[0][2]], cb, {"T0": fixed_T0})
    else:
        p, _ = _nm(g, "L4" + var, ["T0"] + CK, [np.concatenate([[np.log(c[1])], c[2]]) for c in cells[:3]],
                   [(np.log(T_MIN), np.log(T_MAX))] + cb, {})
    return p


def fit_L5(g, var=""):
    n = g["n_fit"]
    m = g["mtr"][:n].astype(bool)
    cells = []
    for T0 in TGRID:
        for a in L1_AB:
            for b in L1_AB:
                T = np.maximum(T0 * np.exp(a * g["sinw"][:n] + b * g["cosw"][:n]), 0.05)
                y = _lin(g, g["I"][:n], T, n) - g["obs0"][:n]
                Z = np.stack([g["Ibar"] * _lin(g, g["H"][j, :n], T, n) for j in range(4)])
                c, sse = _ls(y, Z, m, -3.0, 3.0)
                cells.append((sse, T0, a, b, c))
    cells.sort(key=lambda c: c[0])
    bounds = [(np.log(T_MIN), np.log(T_MAX)), (-2.0, 2.0), (-2.0, 2.0)] + [(-3.0, 3.0)] * 4
    x0s = [np.concatenate([[np.log(c[1]), c[2], c[3]], c[4]]) for c in cells[:3]]
    p, _ = _nm(g, "L5" + var, ["T0", "a", "b"] + CK, x0s, bounds, {})
    return p


def fit_all(g, laws=LAWS):
    out = {}
    for law in laws:
        if law == "L1":
            out[law] = fit_L1(g)
        elif law == "L2":
            out[law] = fit_L2(g)
        elif law == "L3":
            out[law] = fit_L3(g, False)
        elif law == "L3b":
            out[law] = fit_L3(g, True)
        elif law == "L4":
            out[law] = fit_L4(g)
        elif law == "L4r":
            out[law] = fit_L4(g, fixed_T0=T_MIN)
        elif law == "L5":
            out[law] = fit_L5(g)
        elif split(law)[0] == "L4" and split(law)[1]:
            out[law] = fit_L4(g, var=split(law)[1])
        elif split(law)[0] == "L5" and split(law)[1]:
            out[law] = fit_L5(g, var=split(law)[1])
    return out


def shape_features(g):
    C, Cbar = g["C"], g["Cbar"]
    I = g["I"][g["train"]]
    pk = 2 * np.pi * (np.argmax(C) + 1) / 365.0
    return dict(clim_amp=float((C.max() - C.min()) / (2 * Cbar)) if Cbar > 0 else 0.0,
                clim_std=float(C.std() / Cbar) if Cbar > 0 else 0.0,
                clim_peak_sin=float(np.sin(pk)), clim_peak_cos=float(np.cos(pk)),
                inflow_cv=float(I.std() / I.mean()) if I.mean() > 0 else 0.0, Ibar=g["Ibar"])


def score(g, law, p):
    Q, fl = run(g, law, p)
    r = {f"{law}_{k}": v for k, v in metrics(Q, g["O"], g["test"]).items()}
    r[f"{law}_train_nse"] = metrics(Q, g["O"], g["train"])["nse"]
    r[f"{law}_floor_test"] = float(fl[g["test"]].mean())
    r[f"{law}_floor_all"] = float(fl[g["n_fit"] - int(g["train"].sum()):].mean())
    for lab, v in zip(BAND_LAB, bands(Q, g["O"], g["test"])):
        r[f"{law}_band_{lab}"] = v
    if law != "L0" and flux(g, law, p) is not None:
        rr = flux(g, law, p)
        r[f"{law}_flux_amp"] = float(np.std(rr) / g["Ibar"]) if g["Ibar"] > 0 else np.nan
    return r
