"""Shared loaders and metrics for the non-dam lever analysis (experiments/levers/)."""
import pathlib

import numpy as np
import pandas as pd
import zarr

RUNS = pathlib.Path("/home/tbindas/projects/ddrs/.ddrs/runs")
HERE = pathlib.Path(__file__).resolve().parent
EV = "ev" + "al"
S42 = "2026-09-27T07-29-47Z-train-and-test"
S43 = "2026-09-27T10-31-30Z-train-and-test"


def load_preds(run, sub="predictions.zarr"):
    """(ids, DatetimeIndex, P[g,t] float64, O[g,t] float64) from a run's test-phase predictions zarr."""
    p = pathlib.Path(run)
    if not p.is_absolute():
        p = RUNS / run
    z = zarr.open(str(p / EV / sub), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    return ids, t, z["predictions"][:].astype(float), z["observations"][:].astype(float)


def water_year(t):
    return np.asarray(t.year + (t.month >= 10))


def align(ids, P, ref_ids):
    """Reorder rows of P to ref_ids."""
    ix = {s: i for i, s in enumerate(ids)}
    return P[[ix[s] for s in ref_ids]]


def metrics(P, O, min_days=365):
    """Per-gauge NSE, KGE (2009: r, alpha = sd ratio, beta = mean ratio), r, alpha, beta, volume ratio.

    P, O: [g, t]. Masked jointly where either is not finite. Gauges with < min_days valid days are NaN.
    """
    m = np.isfinite(P) & np.isfinite(O)
    nv = m.sum(1)
    Pm = np.where(m, P, 0.0)
    Om = np.where(m, O, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mo = Om.sum(1) / nv
        mp = Pm.sum(1) / nv
        do = np.where(m, O - mo[:, None], 0.0)
        dp = np.where(m, P - mp[:, None], 0.0)
        sse = (np.where(m, P - O, 0.0) ** 2).sum(1)
        sst = (do ** 2).sum(1)
        nse = 1 - sse / sst
        so = np.sqrt((do ** 2).sum(1) / nv)
        sp = np.sqrt((dp ** 2).sum(1) / nv)
        r = (do * dp).sum(1) / nv / (so * sp)
        alpha = sp / so
        beta = mp / mo
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    bad = nv < min_days
    out = pd.DataFrame(dict(nse=nse, kge=kge, r=r, alpha=alpha, beta=beta, nvalid=nv))
    out.loc[bad, ["nse", "kge", "r", "alpha", "beta"]] = np.nan
    return out


def ls_scale(P, O):
    """Per-gauge least-squares scalar k* = sum(p o) / sum(p^2): the NSE-optimal pure rescaling."""
    m = np.isfinite(P) & np.isfinite(O)
    Pm, Om = np.where(m, P, 0.0), np.where(m, O, 0.0)
    return (Pm * Om).sum(1) / (Pm ** 2).sum(1)


def vol_ratio(P, O):
    """Per-gauge observed / predicted volume on jointly valid days (the KGE-beta-neutral scalar)."""
    m = np.isfinite(P) & np.isfinite(O)
    return np.where(m, O, 0.0).sum(1) / np.where(m, P, 0.0).sum(1)


def boot_median_shift(new, base, B=2000, seed=1):
    """95 % bootstrap interval (over gauges, paired) of median(new) - median(base)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(new), np.asarray(base)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def summarize(name, new, base, metric="NSE"):
    """One line: population median change, paired median change, share improved, bootstrap CI."""
    new, base = pd.Series(new), pd.Series(base)
    ok = new.notna() & base.notna()
    d = (new - base)[ok]
    lo, hi = boot_median_shift(new[ok], base[ok])
    return dict(name=name, metric=metric, median_base=float(base[ok].median()), median_new=float(new[ok].median()),
                pop_change=float(new[ok].median() - base[ok].median()), ci_lo=lo, ci_hi=hi,
                paired_median=float(d.median()), share_up=float((d > 1e-9).mean()),
                share_down=float((d < -1e-9).mean()), n=int(ok.sum()))


def fmt(s):
    return (f"  {s['name']:62s} {s['metric']:3s} median {s['median_base']:.4f} -> {s['median_new']:.4f}  "
            f"pop {s['pop_change']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]  paired {s['paired_median']:+.4f}  "
            f"up {100 * s['share_up']:.1f}% down {100 * s['share_down']:.1f}%")
