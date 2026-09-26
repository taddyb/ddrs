#!/usr/bin/env python
"""Score the option C sandbox on the dam benchmark: per-gauge NSE / KGE for the three eval arms
(run_option_c_eval.sh), paired differences against reservoirs-off, and wall times.

Window WY1997-2010 (the first eval year is warm-up), as in the benchmark. Paired statistics: median
difference with a 2,000-resample bootstrap 95 % interval (seed 42), the count of gauges that
improve, and a two-sided sign test over the gauges whose NSE changed.
Writes experiments/reservoir/results/option_c_benchmark.json and
output/reservoir_benchmark/option_c_by_gauge.csv.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.stats import binomtest, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "output" / "reservoir_benchmark"
RES = HERE.parent / "results" / "option_c_benchmark.json"
STORED = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-12T23-39-03Z-train-and-test/eval/predictions.zarr")
ARMS = ["off", "fit", "prior"]
rng = np.random.default_rng(42)


def load(path):
    z = zarr.open(str(path), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = z["time"][:].astype("datetime64[ns]").astype("datetime64[D]")
    wy = np.array([d.year + (d.month >= 10) for d in t.astype(object)])
    win = (wy >= 1997) & (wy <= 2010)
    return pd.Index(ids), z["predictions"][:][:, win], z["observations"][:][:, win]


def metrics(p, o):
    """NSE, KGE, and KGE's r, alpha, beta."""
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() < 365:
        return (np.nan,) * 5
    p, o = p[m], o[m]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    a, b = p.std() / o.std(), p.mean() / o.mean()
    return nse, 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2), r, a, b


def boot_median(x):
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    if len(x) == 1:
        return dict(n=1, median=float(x[0]))
    b = np.median(rng.choice(x, (2000, len(x))), axis=1)
    return dict(n=int(len(x)), median=float(np.median(x)), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])


def paired(d):
    d = np.asarray(d)[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-6]
    up = int((nz > 0).sum())
    return dict(**boot_median(d), n_changed=int(len(nz)), n_up=up, n_down=int(len(nz) - up),
                mean=float(d.mean()) if len(d) else np.nan,
                sign_test_p=float(binomtest(up, len(nz)).pvalue) if len(nz) else np.nan)


bench = pd.read_csv(HERE / "dam_benchmark.csv", dtype={"huc2": str, "gauge": str})
fits = pd.read_csv(HERE / "reservoir_T_fits.csv", dtype={"gauge": str}).set_index("gauge")
audit = pd.read_csv(HERE / "attribute_audit.csv", dtype={"gauge": str}).set_index("gauge")
df = bench[["huc2", "tier", "gauge", "GRAND_ID", "COMID", "LAKE_NAME", "gauge_dor", "gauge_nse_trained", "gauge_nse_base"]].copy()
df = df.rename(columns={"gauge_nse_trained": "nse_leakance_run", "gauge_nse_base": "nse_summed_qprime"})
df["T_status"] = df.gauge.map(fits.status)
df["T_days"] = df.gauge.map(fits.T_days)
df["nse_fit_window"] = df.gauge.map(fits.nse_fit)  # linear-reservoir NSE on ResOpsUS days outside WY1997-2010
df["nse_obs_inflow_eval"] = df.gauge.map(fits.nse_eval_obs_inflow)
df["main_purpose"] = df.gauge.map(fits.main_purpose)
df["dam_on_gauge_reach"] = df.gauge.map(audit.dam_on_gauge_reach)

for arm in ARMS:
    ids, P, O = load(OUT / f"pred_{arm}.zarr")
    m = np.array([metrics(P[ids.get_loc(g)], O[ids.get_loc(g)]) for g in df.gauge])
    for j, k in enumerate(["nse", "kge", "r", "alpha", "beta"]):
        df[f"{k}_{arm}"] = m[:, j]
ids, P, O = load(STORED)
df["nse_stored_cuda"] = [metrics(P[ids.get_loc(g)], O[ids.get_loc(g)])[0] for g in df.gauge]
for arm in ["fit", "prior"]:
    df[f"dnse_{arm}"] = df[f"nse_{arm}"] - df.nse_off
    df[f"dkge_{arm}"] = df[f"kge_{arm}"] - df.kge_off
df.to_csv(OUT / "option_c_by_gauge.csv", index=False)

fitted = df.T_status == "fitted"
res = dict(
    host_run="2026-09-12T23-39-03Z-train-and-test (sr_n0_gamma), checkpoint epoch_50_mb_9, CPU",
    window="WY1997-2010", n_gauges=len(df),
    sanity_off_vs_stored_cuda_max_abs_nse=float((df.nse_off - df.nse_stored_cuda).abs().max()),
    median_nse={a: boot_median(df[f"nse_{a}"]) for a in ARMS}
    | {"leakance_run": boot_median(df.nse_leakance_run), "summed_qprime": boot_median(df.nse_summed_qprime)},
    median_kge={a: float(df[f"kge_{a}"].median()) for a in ARMS},
    paired_nse=dict(
        fit_all=paired(df.dnse_fit), fit_on_fitted_dams=paired(df.dnse_fit[fitted]),
        fit_on_unfitted_dams=paired(df.dnse_fit[~fitted]),
        prior_all=paired(df.dnse_prior), prior_on_fitted_dams=paired(df.dnse_prior[fitted]),
        prior_on_prior_only_dams=paired(df.dnse_prior[~fitted]),
    ),
    paired_kge=dict(fit_on_fitted_dams=paired(df.dkge_fit[fitted]), prior_on_prior_only_dams=paired(df.dkge_prior[~fitted])),
    fitted_dams=dict(
        median_nse_off=float(df.nse_off[fitted].median()), median_nse_fit=float(df.nse_fit[fitted].median()),
        median_nse_obs_inflow_ceiling=float(df.nse_obs_inflow_eval[fitted].median()),
        spearman_dnse_vs_log_T=float(spearmanr(np.log(df.T_days[fitted]), df.dnse_fit[fitted]).statistic),
        spearman_dnse_vs_fit_window_nse=float(spearmanr(df.nse_fit_window[fitted], df.dnse_fit[fitted]).statistic),
        kge_parts_median={k: dict(off=float(df[f"{k}_off"][fitted].median()), fit=float(df[f"{k}_fit"][fitted].median()))
                          for k in ["r", "alpha", "beta"]},
    ),
    # Post hoc gates, decided after seeing the off-vs-fit table; both use only information available
    # without the eval window (the fit-window NSE is scored on ResOpsUS days outside WY1997-2010).
    gates_post_hoc={
        "fit_window_nse_ge_0.5": paired(df.dnse_fit[fitted & (df.nse_fit_window >= 0.5)]),
        "fit_window_nse_lt_0.5": paired(df.dnse_fit[fitted & (df.nse_fit_window < 0.5)]),
        "T_le_60d": paired(df.dnse_fit[fitted & (df.T_days <= 60)]),
        "T_gt_60d": paired(df.dnse_fit[fitted & (df.T_days > 60)]),
        "purpose_FCON": paired(df.dnse_fit[fitted & (df.main_purpose == "FCON")]),
        "purpose_other": paired(df.dnse_fit[fitted & (df.main_purpose != "FCON")]),
    },
    by_huc2={h: dict(n=len(g), n_fitted=int((g.T_status == "fitted").sum()),
                     nse_off=float(g.nse_off.median()), nse_fit=float(g.nse_fit.median()), nse_prior=float(g.nse_prior.median()),
                     dnse_fit_median=float(g.dnse_fit.median()), dnse_prior_median=float(g.dnse_prior.median()))
             for h, g in df.groupby("huc2")},
    by_tier={int(t): dict(n=len(g), dnse_fit=paired(g.dnse_fit), dnse_prior=paired(g.dnse_prior)) for t, g in df.groupby("tier")},
)

# Wall time per arm from file birth times: run_option_c_eval.sh creates eval_<arm>.log as the arm
# starts, and the next arm's log (or, for the last arm, the zarr's final write) follows its end.
# (timing.txt holds blanks from the first run: the script used `bc`, which is not installed.)
def birth(p):
    return float(subprocess.run(["stat", "-c", "%W", str(p)], capture_output=True, text=True, check=True).stdout)


def mtime_newest(p):
    return max(f.stat().st_mtime for f in Path(p).rglob("*"))


res["wall_seconds"] = {
    a: (birth(OUT / f"eval_{ARMS[i + 1]}.log") if i + 1 < len(ARMS) else mtime_newest(OUT / f"pred_{a}.zarr"))
    - birth(OUT / f"eval_{a}.log")
    for i, a in enumerate(ARMS)
}
for a in ARMS:
    log = (OUT / f"eval_{a}.log").read_text()
    res.setdefault("log", {})[a] = re.findall(r"reservoirs: .*", log)
    n_chunks = len(re.findall(r"chunk \d+/\d+", log))
    res.setdefault("chunks", {})[a] = n_chunks

RES.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(RES, "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ("by_huc2", "by_tier")}, indent=1, default=float))
pd.set_option("display.width", 250)
print(pd.DataFrame(res["by_huc2"]).T.round(3).to_string())
cols = ["huc2", "tier", "LAKE_NAME", "T_days", "dam_on_gauge_reach", "nse_off", "nse_fit", "dnse_fit", "nse_obs_inflow_eval",
        "nse_leakance_run"]
print(df[fitted].sort_values("dnse_fit")[cols].round(3).to_string())
