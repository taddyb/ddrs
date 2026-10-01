"""Phase 1: offline 1-D and 2-D loss landscapes of the dam parameters at the smoke set's on-reach dam gauges.

Harness: experiments/reservoir/laws_v6 (v6.sim, laws.c, liblaws.so), which reproduces the rulecurve_offline L2/L4
fits to 4e-14 (check_repro.py), on the smoke no-dam run's routed flow (pred_1981_2010.zarr). Train WY1983-1995
(WY1982 spin-up), test WY1996-2010.

Objective (the engine's training loss, src/training/loss.rs::nse_batch_loss): on the training mask,
    L = mean_t (Q_t - O_t)^2 / (sigma + 0.1)^2,  sigma = population std of the gauge's observations, WY1982-1995.
L ~ (1 - NSE_train) sigma^2 / (sigma + 0.1)^2, so loss units are NSE units up to that factor (median 0.99).

Parameters and coordinates (the engine's raw coordinates, src/nn/dam_params.rs):
  T0   ln T0            (per-dam delta = ln(T0 / T0_head); head init T0 = 4.5 h = 0.1875 d)
  amp  s, the rule-curve coefficients times s (engine c = 0.5 tanh(theta); init theta = 0 <=> s = 0)
  z    ln z             (engine z = 120 sigmoid(4 r_z + logit(0.05/120)) ~ 0.05 exp(4 r_z) below ~30 d; init 0.05 d)
  kc   ln kc            (engine kc = 3 exp(r_kc); init 3)
  phi  logit phi        (engine phi = sigmoid(r_phi); init 0.5)
Base laws (other parameters at their fitted values): T0 and amp on L4 (bucket + rule curve, fits_by_gauge.csv) at
the on-reach dam gauges, plus T0 on L2 (plain bucket); z, kc, phi (and T0 again) on FA (flood pool + bucket,
laws_by_gauge.csv) at the 69 on-reach flood-control dams of experiments/reservoir/smoke/fixed_FA.csv.

Outputs (this directory): slices.npz (every 1-D slice), planes.npz (2-D planes), per_gauge_slices.csv,
joint_init.csv, pool_regime.csv, year_grad.csv.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
AG = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(AG, "experiments/reservoir/laws_v6"))
import v6  # noqa: E402

EPS = 0.1
T0_HEAD = 4.5 / 24.0
INIT = dict(T0=T0_HEAD, amp=0.0, z=0.05, kc=3.0, phi=0.5)
DELTAS = (0.02, 0.005)
Z_LO, Z_HI = 0.05, 120.0

D = v6.load()
WY = D["wy"]
SIGMA_MASK = (WY >= 1982) & (WY <= 1995)
RCF = pd.read_csv(os.path.join(AG, "experiments/reservoir/rulecurve_offline/fits_by_gauge.csv"),
                  dtype={"STAID": str}).set_index("STAID")
LF = pd.read_csv(os.path.join(AG, "experiments/reservoir/laws_v6/laws_by_gauge.csv"), dtype={"STAID": str}).set_index("STAID")
FA69 = pd.read_csv(os.path.join(AG, "experiments/reservoir/smoke/fixed_FA.csv"), dtype={"STAID": str}).STAID.tolist()
SG, DAM, CTL = v6.gauges()
ON = [s for s in DAM.index[DAM.on_reach] if s in RCF.index]
CK = ["L4_p_c1s", "L4_p_c1c", "L4_p_c2s", "L4_p_c2c"]


# ------------------------------------------------------------------ per-gauge context and evaluation
def ctx(s):
    i = D["ids"].get_loc(s)
    g = v6.prep(D["P"][i], D["O"][i], D["doy"], D["train"], D["test"])
    O = g["O"]
    sig = float(np.nanstd(O[SIGMA_MASK]))
    g["sigma"] = sig
    g["ntr"] = int(g["mtr"].sum())
    g["lscale"] = 1.0 / (g["ntr"] * (sig + EPS) ** 2)
    te = g["test"] & np.isfinite(O)
    g["te"] = te
    g["te_den"] = float(((O[te] - O[te].mean()) ** 2).sum())
    return g


def halfL(g, Q):
    """Training loss on the two halves of the training period, WY1983-1989 and WY1990-1995."""
    out = []
    for lo, hi in ((1983, 1989), (1990, 1995)):
        m = g["mtr"].astype(bool) & (WY >= lo) & (WY <= hi)
        e = Q[m] - g["O"][m]
        out.append(float(e @ e) / (m.sum() * (g["sigma"] + EPS) ** 2))
    return out


def evalQ(g, Q):
    """(training loss, training NSE, test NSE) of a full release series."""
    m = g["mtr"].astype(bool)
    e = Q[m] - g["O"][m]
    sse = float(e @ e)
    et = Q[g["te"]] - g["O"][g["te"]]
    return sse * g["lscale"], 1.0 - sse / g["den"], 1.0 - float(et @ et) / g["te_den"]


def trloss(g, Q):
    m = g["mtr"].astype(bool)
    e = Q[m] - g["O"][m]
    return float(e @ e) * g["lscale"]


def simQ(g, law, p):
    """law L2: p[T0]; L4: T0, amp (x fitted coefficients c); FA: T0, kc, phi, z."""
    if law == "L2":
        return v6.sim(g, T=p["T0"], want=True)["Q"]
    if law == "L4":
        return v6.sim(g, T=p["T0"], r=v6.l4_flux(g, p["amp"] * np.asarray(p["c"])), want=True)["Q"]
    if law == "FA":
        return v6.sim(g, T=p["T0"], mode=0, Qc=p["kc"] * g["Ibar"], phi=p["phi"], Fmax=p["z"] * g["Ibar"],
                      Qe=p["kc"] * g["Ibar"], Te=1.0, want=True)["Q"]
    raise ValueError(law)


def simfull(g, law, p):
    return v6.sim(g, T=p["T0"], mode=0, Qc=p["kc"] * g["Ibar"], phi=p["phi"], Fmax=p["z"] * g["Ibar"],
                  Qe=p["kc"] * g["Ibar"], Te=1.0, want=True)


# ------------------------------------------------------------------ coordinates
def to_x(name, v):
    if name in ("T0", "z", "kc"):
        return np.log(v)
    if name == "phi":
        v = np.clip(v, 1e-6, 1 - 1e-6)
        return np.log(v / (1 - v))
    return v  # amp


def from_x(name, x):
    if name in ("T0", "z", "kc"):
        return np.exp(x)
    if name == "phi":
        return 1.0 / (1.0 + np.exp(-x))
    return x


def grids(name, fit):
    """Grid in the parameter's own units."""
    if name == "T0":
        return fit * 10.0 ** np.linspace(-2, 2, 81)
    if name == "amp":
        return np.linspace(0.0, 2.0, 81)
    if name == "z":
        return np.logspace(np.log10(Z_LO), np.log10(Z_HI), 61)
    if name == "kc":
        return np.logspace(np.log10(0.3), np.log10(40.0), 61)
    if name == "phi":
        return 1.0 / (1.0 + np.exp(-np.linspace(-5.0, 7.0, 61)))  # 0.0067 .. 0.9991
    raise ValueError(name)


BOUNDS_X = dict(z=(np.log(Z_LO), np.log(Z_HI)), kc=(np.log(0.3), np.log(40.0)), phi=(-5.0, 7.0), amp=(0.0, 2.0))


def fitted(s, law):
    if law == "L2":
        return dict(T0=float(RCF.loc[s, "L2_p_T0"]))
    if law == "L4":
        return dict(T0=float(RCF.loc[s, "L4_p_T0"]), amp=1.0, c=RCF.loc[s, CK].astype(float).values)
    r = LF.loc[s]
    return dict(T0=float(r.FA_p_T0), kc=float(r.FA_p_kc), phi=float(min(r.FA_p_phi, 0.999)),
                z=float(min(r.FA_p_z, Z_HI)), z_raw=float(r.FA_p_z), phi_raw=float(r.FA_p_phi))


# ------------------------------------------------------------------ slice measures
def band(xg, nse, k, delta):
    """Contiguous interval of grid points around index k with nse >= nse[k] - delta: (lo, hi, open_lo, open_hi)."""
    thr = nse[k] - delta
    lo = k
    while lo - 1 >= 0 and nse[lo - 1] >= thr:
        lo -= 1
    hi = k
    while hi + 1 < len(nse) and nse[hi + 1] >= thr:
        hi += 1
    return xg[lo], xg[hi], lo == 0, hi == len(nse) - 1


def n_minima(L, prom):
    """Local minima of L with prominence >= prom, edge minima included (L padded with a high wall)."""
    L = np.asarray(L)
    wall = L.max() + 10 * prom + 1.0
    pk, _ = find_peaks(-np.concatenate([[wall], L, [wall]]), prominence=prom)
    return len(pk)


def measure_slice(g, s, law, name, base):
    """1-D slice of `name` with every other parameter at `base` (fitted)."""
    fit = base[name]
    vals = grids(name, fit)
    xg = to_x(name, vals)
    L, tr, te = np.empty(len(vals)), np.empty(len(vals)), np.empty(len(vals))
    H1, H2 = np.empty(len(vals)), np.empty(len(vals))

    def at(v):
        p = dict(base)
        p[name] = v
        return evalQ(g, simQ(g, law, p))

    for j, v in enumerate(vals):
        L[j], tr[j], te[j] = at(v)
        pj = dict(base)
        pj[name] = v
        H1[j], H2[j] = halfL(g, simQ(g, law, pj))

    def Lx(x):
        return at(float(from_x(name, x)))[0]

    # refined training argmin along the slice (bounded Brent around the grid argmin)
    k = int(np.argmin(L))
    a, b = xg[max(k - 1, 0)], xg[min(k + 1, len(xg) - 1)]
    if name in BOUNDS_X:
        a, b = max(a, BOUNDS_X[name][0]), min(b, BOUNDS_X[name][1])
    xs, Ls = xg[k], L[k]
    if b > a:
        r = minimize_scalar(Lx, bounds=(a, b), method="bounded", options=dict(xatol=1e-4))
        if r.fun < Ls:
            xs, Ls = float(r.x), float(r.fun)
    _, trs, tes = at(float(from_x(name, xs)))
    edge = k in (0, len(xg) - 1)
    # curvature at the optimum: quadratic fit over +-0.2 coordinate units (9 points)
    hq = 0.2
    xq = xs + np.linspace(-hq, hq, 9)
    if name in BOUNDS_X:
        xq = np.clip(xq, *BOUNDS_X[name])
    Lq = np.array([Lx(x) for x in xq])
    if np.ptp(xq) > 0:
        H_q = 2.0 * np.polyfit(xq - xs, Lq, 2)[0]
    else:
        H_q = np.nan
    # three-point FD with h = 0.1 for comparison
    h = 0.1
    H_fd = (Lx(xs + h) - 2 * Ls + Lx(xs - h)) / h ** 2
    # curvature in ln(param) units for the linear-coordinate parameters (amp: ln s; phi: reported in logit)
    v_s = float(from_x(name, xs))
    H_ln = H_q * v_s ** 2 if name == "amp" else H_q
    # factor-2 flatness criterion of the routing census: loss change for a factor-2 move / L*
    if name in ("T0", "z", "kc"):
        f2 = max(Lx(xs + np.log(2)), Lx(xs - np.log(2))) - Ls
    elif name == "amp":
        f2 = max(at(v_s * 2)[0], at(v_s / 2)[0]) - Ls
    else:  # phi: factor 2 in phi, capped at 1
        f2 = max(at(min(v_s * 2, 0.999))[0], at(v_s / 2)[0]) - Ls
    # test-NSE optimum along the slice
    kt = int(np.argmax(te))
    # bands (training NSE and test NSE), in coordinate units, and whether the init lies inside
    xi = to_x(name, INIT[name]) if name != "T0" else np.log(INIT["T0"])
    out = dict(STAID=s, law=law, param=name, fit=fit, x_fit=to_x(name, fit), x_opt=xs, v_opt=v_s, L_opt=Ls,
               trNSE_opt=trs, teNSE_at_trainopt=tes, x_teopt=xg[kt], teNSE_max=te[kt], edge_opt=edge,
               H_q=H_q, H_fd=H_fd, H_ln=H_ln, f2_dL=f2, f2_rel=f2 / Ls if Ls > 0 else np.nan,
               nmin_002=n_minima(L, 0.002), nmin_0005=n_minima(L, 0.0005), x_init=xi, sigma=g["sigma"],
               lfac=g["sigma"] ** 2 / (g["sigma"] + EPS) ** 2)
    # training-NSE value at the init
    vi = INIT[name]
    Li, tri, tei = at(vi)
    out.update(L_init=Li, trNSE_init=tri, teNSE_init=tei)
    kk = int(np.argmax(tr))
    for d in DELTAS:
        lo, hi, olo, ohi = band(xg, tr, kk, d)
        out[f"band_tr_{d}"] = hi - lo
        out[f"band_tr_{d}_open"] = bool(olo or ohi)
        out[f"init_in_tr_{d}"] = bool(tri >= tr.max() - d)
        lo, hi, olo, ohi = band(xg, te, kt, d)
        out[f"band_te_{d}"] = hi - lo
        out[f"band_te_{d}_open"] = bool(olo or ohi)
        out[f"init_in_te_{d}"] = bool(tei >= te.max() - d)
    # gradients (central FD, h = 0.05 coordinate units) at the init and halfway (coordinate midpoint) to the optimum
    hg = 0.05

    def grad(x):
        return (Lx(x + hg) - Lx(x - hg)) / (2 * hg)

    xm = 0.5 * (xi + xs)
    gi, gm = grad(xi), grad(xm)
    if law == "FA":
        pi_, pm_ = dict(base), dict(base)
        pi_[name], pm_[name] = float(from_x(name, xi)), float(from_x(name, xm))
        out["pg_init"] = pgrad(g, pi_)[1][PIDX[name]]
        out["pg_mid"] = pgrad(g, pm_)[1][PIDX[name]]
        out["pg_opt"] = pgrad(g, dict(base, **{name: v_s}))[1][PIDX[name]]
    out.update(g_init=gi, g_mid=gm, g_ratio=abs(gi) / abs(gm) if gm != 0 else np.nan,
               g_init_toward=bool(np.sign(-gi) == np.sign(xs - xi)) if gi != 0 else False,
               g_mid_toward=bool(np.sign(-gm) == np.sign(xs - xi)) if gm != 0 else False, x_mid=xm)
    k1, k2 = int(np.argmin(H1)), int(np.argmin(H2))
    out.update(x_h1=xg[k1], x_h2=xg[k2], h1_edge=k1 in (0, len(xg) - 1), h2_edge=k2 in (0, len(xg) - 1),
               # loss penalty each half pays at the other half's optimum
               h_cross=max(H1[k2] - H1[k1], H2[k1] - H2[k2]))
    # phi band in phi units
    if name == "phi":
        kk = int(np.argmax(tr))
        for d in DELTAS:
            lo, hi, _, _ = band(vals, tr, kk, d)
            out[f"band_tr_{d}_phi"] = hi - lo
            out[f"band_tr_{d}_phi_lo"] = lo
        out["teNSE_phi1"] = te[-1]
    return out, dict(vals=vals, x=xg, L=L, tr=tr, te=te, H1=H1, H2=H2)


# ------------------------------------------------------------------ pathwise (autodiff-equivalent) gradient, law FA
import ctypes  # noqa: E402

_G = ctypes.CDLL(os.path.join(HERE, "liblawgrad.so"))
_G.lawgrad.restype = None
_G.lawgrad.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int] + [ctypes.c_double] * 4 + [ctypes.c_int,
                                                                                                    ctypes.c_void_p,
                                                                                                    ctypes.c_void_p]


def pgrad(g, p, per_day=False):
    """Training loss and its pathwise gradient in (ln T0, ln kc, logit phi, ln z) for law FA at p; with per_day,
    also the per-day gradient contributions (n x 4)."""
    n = g["n"]
    I = np.ascontiguousarray(g["I"], dtype=np.float64)
    Q = np.empty(n)
    dQ = np.empty((n, 4))
    _G.lawgrad(I.ctypes.data, None, n, p["T0"], p["kc"] * g["Ibar"], p["phi"], p["z"] * g["Ibar"], 1,
               Q.ctypes.data, dQ.ctypes.data)
    m = g["mtr"].astype(bool)
    e = np.where(m, Q - g["obs0"], 0.0)
    contrib = 2.0 * e[:, None] * dQ * g["lscale"]
    L = float((e * e).sum()) * g["lscale"]
    gr = contrib.sum(0)
    return (L, gr, contrib) if per_day else (L, gr)


PIDX = dict(T0=0, kc=1, phi=2, z=3)


# ------------------------------------------------------------------ joint engine-init gradient (FA gauges)
def raw_from_phys(T0, kc, phi, z):
    q = Z_LO / Z_HI
    zz = np.clip(z / Z_HI, 1e-9, 1 - 1e-9)
    return np.array([np.log(T0 / T0_HEAD), np.log(kc / 3.0), np.log(phi / (1 - phi)),
                     (np.log(zz / (1 - zz)) - np.log(q / (1 - q))) / 4.0])


def phys_from_raw(r):
    q = Z_LO / Z_HI
    T0 = T0_HEAD * np.exp(r[0])
    kc = float(np.clip(3.0 * np.exp(r[1]), 0.5, 20.0))
    phi = 1.0 / (1.0 + np.exp(-r[2]))
    z = Z_HI / (1.0 + np.exp(-(4 * r[3] + np.log(q / (1 - q)))))
    return dict(T0=float(T0), kc=kc, phi=float(phi), z=float(z))


def joint(g, s, fit):
    rs = raw_from_phys(fit["T0"], np.clip(fit["kc"], 0.5, 20.0), min(fit["phi"], 0.99), min(fit["z"], 119.0))

    def Lr(r):
        return trloss(g, simQ(g, "FA", phys_from_raw(r)))

    h = 0.02
    rows = []
    for tag, r0 in (("init", np.zeros(4)), ("mid", 0.5 * rs), ("opt", rs)):
        L0 = Lr(r0)
        gr, hd = [], []
        for j in range(4):
            e = np.zeros(4)
            e[j] = h
            Lp, Lm = Lr(r0 + e), Lr(r0 - e)
            gr.append((Lp - Lm) / (2 * h))
            hd.append((Lp - 2 * L0 + Lm) / h ** 2)
        ph = phys_from_raw(r0)
        _, pg = pgrad(g, ph)
        kc_in = 1.0 if 0.5 < 3.0 * np.exp(r0[1]) < 20.0 else 0.0
        praw = [pg[0], pg[1] * kc_in, pg[2], pg[3] * 4.0 * (1.0 - ph["z"] / Z_HI)]
        rows.append(dict(STAID=s, point=tag, L=L0, g_T0=gr[0], g_kc=gr[1], g_phi=gr[2], g_z=gr[3],
                         H_T0=hd[0], H_kc=hd[1], H_phi=hd[2], H_z=hd[3], r_T0=r0[0], r_kc=r0[1], r_phi=r0[2],
                         r_z=r0[3], pg_T0=praw[0], pg_kc=praw[1], pg_phi=praw[2], pg_z=praw[3], **{f"phys_{k}": v for k, v in ph.items()}))
    return rows


def pool_regime(g, s, fit, z, kc=None, phi=None):
    """Which constraint binds on flood days: capture capacity-limited share and evacuation pool-limited share,
    over training days."""
    kc = fit["kc"] if kc is None else kc
    phi = fit["phi"] if phi is None else phi
    p = dict(T0=fit["T0"], kc=kc, phi=phi, z=z)
    o = simfull(g, "FA", p)
    I, C, E, F = g["I"], o["C"], o["E"], o["F"]
    n = len(I)
    Qc = kc * g["Ibar"]
    Fprev = np.concatenate([[0.0], F[:-1]])
    want = phi * np.maximum(I - Qc, 0.0)
    tr = g["train"]
    cap_days = tr & (want > 0)
    cap_lim = cap_days & (C < want - 1e-9)
    ev_days = tr & (E > 0)
    ev_pool = ev_days & (np.abs(E - Fprev) <= 1e-9 * np.maximum(1.0, Fprev))
    return dict(STAID=s, z=z, kc=kc, phi=phi, flood_days=int(cap_days.sum()), cap_limited=int(cap_lim.sum()),
                evac_days=int(ev_days.sum()), evac_pool_limited=int(ev_pool.sum()),
                pool_full_share=float(cap_lim.sum() / max(cap_days.sum(), 1)),
                evac_pool_share=float(ev_pool.sum() / max(ev_days.sum(), 1)))


def year_grads(g, s, fit):
    """Per-training-water-year gradient of the training loss in ln z (FD h=0.05) at the engine init
    (T0 head, kc 3, phi 0.5, z 0.05), at the fitted (T0, kc, phi) with z 0.05, and at the fitted point halfway in ln z."""
    rows = []
    m = g["mtr"].astype(bool)
    O = g["O"]
    for tag, base in (("init", dict(T0=T0_HEAD, kc=3.0, phi=0.5, z=0.05)),
                      ("fitted_z0.05", dict(T0=fit["T0"], kc=fit["kc"], phi=fit["phi"], z=0.05)),
                      ("fitted_zmid", dict(T0=fit["T0"], kc=fit["kc"], phi=fit["phi"],
                                           z=float(np.sqrt(0.05 * fit["z"]))))):
        h = 0.05
        qp = simQ(g, "FA", dict(base, z=base["z"] * np.exp(h)))
        qm = simQ(g, "FA", dict(base, z=base["z"] * np.exp(-h)))
        dl = ((qp - O) ** 2 - (qm - O) ** 2) * g["lscale"] / (2 * h)
        _, _, pc = pgrad(g, base, per_day=True)
        for y in range(1983, 1996):
            sel = m & (WY == y)
            rows.append(dict(STAID=s, point=tag, wy=y, g=float(np.nansum(dl[sel])), pg_T0=float(pc[sel, 0].sum()),
                             pg_kc=float(pc[sel, 1].sum()), pg_phi=float(pc[sel, 2].sum()),
                             pg_z=float(pc[sel, 3].sum())))
    return rows


def zprofile(g, s, fit):
    """Training loss, test NSE and pathwise gradients along z (log grid), at the fitted (T0, kc, phi) and at the
    engine's init (T0 head, kc 3, phi 0.5)."""
    rows = []
    for tag, base in (("fitted", dict(T0=fit["T0"], kc=fit["kc"], phi=fit["phi"])),
                      ("init", dict(T0=T0_HEAD, kc=3.0, phi=0.5))):
        for z in np.logspace(np.log10(Z_LO), np.log10(Z_HI), 31):
            p = dict(base, z=float(z))
            L, gr = pgrad(g, p)
            _, tr, te = evalQ(g, simQ(g, "FA", p))
            rows.append(dict(STAID=s, at=tag, z=float(z), L=L, trNSE=tr, teNSE=te, pg_T0=gr[0], pg_kc=gr[1],
                             pg_phi=gr[2], pg_z=gr[3]))
    return rows


# ------------------------------------------------------------------ 2-D planes
def plane(g, s, fit, which):
    if which == "T0_z":
        a = fit["T0"] * 10.0 ** np.linspace(-2, 2, 41)
        b = np.logspace(np.log10(Z_LO), np.log10(Z_HI), 41)
        keys = ("T0", "z")
    else:
        a = np.logspace(np.log10(Z_LO), np.log10(Z_HI), 41)
        b = np.logspace(np.log10(0.3), np.log10(40.0), 41)
        keys = ("z", "kc")
    L = np.empty((41, 41))
    tr = np.empty((41, 41))
    te = np.empty((41, 41))
    for i, va in enumerate(a):
        for j, vb in enumerate(b):
            p = dict(fit)
            p[keys[0]], p[keys[1]] = float(va), float(vb)
            L[i, j], tr[i, j], te[i, j] = evalQ(g, simQ(g, "FA", p))
    return dict(a=a, b=b, L=L, tr=tr, te=te, keys=keys)


def plane_hessian(P):
    """Quadratic fit of L over the 5x5 neighbourhood of the grid argmin, in ln units of both axes."""
    L = P["L"]
    i, j = np.unravel_index(np.argmin(L), L.shape)
    xa, xb = np.log(P["a"]), np.log(P["b"])
    ii = np.clip(np.arange(i - 2, i + 3), 0, 40)
    jj = np.clip(np.arange(j - 2, j + 3), 0, 40)
    X, Y, Z = [], [], []
    for u in np.unique(ii):
        for v in np.unique(jj):
            X.append(xa[u] - xa[i])
            Y.append(xb[v] - xb[j])
            Z.append(L[u, v])
    X, Y, Z = map(np.asarray, (X, Y, Z))
    A = np.stack([np.ones_like(X), X, Y, X * X, X * Y, Y * Y], 1)
    c = np.linalg.lstsq(A, Z, rcond=None)[0]
    H = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    w, V = np.linalg.eigh(H)
    edge = i in (0, 40) or j in (0, 40)
    kt = np.unravel_index(np.argmax(P["te"]), L.shape)
    return dict(i=i, j=j, H11=H[0, 0], H22=H[1, 1], H12=H[0, 1], lam_min=w[0], lam_max=w[1],
                v_stiff_a=V[0, 1], v_stiff_b=V[1, 1], corr=-H[0, 1] / np.sqrt(abs(H[0, 0] * H[1, 1])) if H[0, 0] * H[1, 1] != 0 else np.nan,
                edge=edge, a_opt=P["a"][i], b_opt=P["b"][j], a_teopt=P["a"][kt[0]], b_teopt=P["b"][kt[1]],
                tr_opt=P["tr"][i, j], te_at_tropt=P["te"][i, j], te_max=P["te"][kt])


# ------------------------------------------------------------------ driver
def work(s):
    g = ctx(s)
    rows, sl = [], {}
    b4 = fitted(s, "L4")
    for name in ("T0", "amp"):
        r, d = measure_slice(g, s, "L4", name, b4)
        rows.append(r)
        sl[f"{s}|L4|{name}"] = d
    b2 = fitted(s, "L2")
    r, d = measure_slice(g, s, "L2", "T0", b2)
    rows.append(r)
    sl[f"{s}|L2|T0"] = d
    jrows, prow, yrows, planes, prows = [], [], [], {}, []
    if s in FA69:
        bf = fitted(s, "FA")
        base = {k: bf[k] for k in ("T0", "kc", "phi", "z")}
        for name in ("T0", "z", "kc", "phi"):
            r, d = measure_slice(g, s, "FA", name, base)
            r["z_raw"], r["phi_raw"] = bf["z_raw"], bf["phi_raw"]
            rows.append(r)
            sl[f"{s}|FA|{name}"] = d
        jrows = joint(g, s, base)
        for z in (0.05, 0.2, 1.0, 5.0, 20.0, base["z"]):
            prow.append(dict(pool_regime(g, s, base, z), at="fitted_kc_phi"))
            prow.append(dict(pool_regime(g, s, base, z, kc=3.0, phi=0.5), at="init_kc_phi"))
        yrows = year_grads(g, s, base) + [dict(r, kind="zprofile") for r in zprofile(g, s, base)]
        for which in ("T0_z", "z_kc"):
            P = plane(g, s, base, which)
            planes[f"{s}|{which}"] = P
            prows.append(dict(STAID=s, plane=which, **plane_hessian(P)))
    return rows, sl, jrows, prow, yrows, planes, prows


if __name__ == "__main__":
    gauges = ON
    print(f"on-reach dam gauges with fits: {len(gauges)}; flood-pool gauges: {len(FA69)}", flush=True)
    with Pool(int(os.environ.get("NPROC", "10"))) as pool:
        res = pool.map(work, gauges, chunksize=2)
    rows, sl, jr, pr, yr, planes, prows = [], {}, [], [], [], {}, []
    for a, b, c, d, e, f, h in res:
        rows += a
        sl.update(b)
        jr += c
        pr += d
        yr += e
        planes.update(f)
        prows += h
    df = pd.DataFrame(rows)
    meta = DAM[["nid_dor", "dam_purpose", "area_km2", "dam_storage_mcm", "huc2"]]
    df = df.merge(meta, left_on="STAID", right_index=True, how="left")
    df.to_csv(os.path.join(HERE, "per_gauge_slices.csv"), index=False)
    pd.DataFrame(jr).to_csv(os.path.join(HERE, "joint_init.csv"), index=False)
    pd.DataFrame(pr).to_csv(os.path.join(HERE, "pool_regime.csv"), index=False)
    yr = pd.DataFrame(yr)
    yr[yr.kind != "zprofile"].drop(columns=["kind"]).to_csv(os.path.join(HERE, "year_grad.csv"), index=False)
    yr[yr.kind == "zprofile"].drop(columns=["kind", "point", "wy", "g"]).to_csv(os.path.join(HERE, "zprofile.csv"),
                                                                                index=False)
    pd.DataFrame(prows).to_csv(os.path.join(HERE, "planes.csv"), index=False)
    np.savez_compressed(os.path.join(HERE, "slices.npz"),
                        **{k + "|" + f: v[f] for k, v in sl.items() for f in ("vals", "x", "L", "tr", "te", "H1", "H2")})
    np.savez_compressed(os.path.join(HERE, "planes.npz"),
                        **{k + "|" + f: v[f] for k, v in planes.items() for f in ("a", "b", "L", "tr", "te")})
    print("done", len(df), "slice rows", flush=True)
