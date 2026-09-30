#!/usr/bin/env python
"""Upstream-summed lateral inflow (Q') climatology for every dam COMID in the
learned-release feature table.

Run end to end with DDR's venv, from anywhere:

    /home/tbindas/projects/ddr/.venv/bin/python \\
        experiments/reservoir/release_head/build_dam_inflow_clim.py

It writes the climatology to `output/damclim/` (gitignored) and then joins
the training-period mean `qmean_m3s` into `dam_features.csv` as the raw column
`inflow_mean_m3s` (Ibar_d, m3/s), which scales the learned rule curve
(`release_head.rule_curve`, `src/routing/release.rs`). The join is textual:
every existing column stays byte-identical, and re-running replaces the column.
`--join-only` redoes just the join from an existing `output/damclim/`.

Provenance: written 2026-09-27 as `damclim.py` in a review session and
committed here unchanged apart from the paths, this note and the join. The
values were validated against the ddrs summed-Q' baseline to 8e-7. Known data
issue, left as is: Flaming Gorge (COMID 77013090) is snapped to a 32 km2
MERIT reach (NID drainage 41 km2), so its Ibar is ~0 and its rule curve is inert.

Only model inputs are used (the dHBV2 UH Q' store and the MERIT CONUS
network). No gauge observations and no observed dam data enter the
climatology; the ddrs summed-Q' baseline predictions (themselves Q' sums) are
read only as a validation target.

Method
------
1. One pass over the Q' store (divide-major, chunk-aligned row blocks):
   - per-reach day-of-year sums over the training window 1981-10-01..1995-09-30
     (doy 1..366, 1-based, pandas dayofyear == chrono ordinal), and the
     per-reach training-period mean;
   - per-reach mean over the baseline test window (validation 1 only);
   - the dams' direct upstream-set daily sums D @ Q, D the 0/1 dam x divide
     membership matrix built from a reverse BFS on the adjacency.
   Reaches in the network but absent from the Q' store get 0.001 m3/s, the
   ddrs / DDR fill (icechunk.rs::read_slab, readers.py StreamflowReader).
2. Accumulate through the network with one sparse triangular solve
   (I - N) A = X, N[down, up] = 1 (indices_0 = downstream row, indices_1 =
   upstream column, rows >= cols). Upstream accumulation is linear, so
   A's doy columns are the doy climatology of the upstream-summed series.
3. Validation 1: accumulated test-window means vs the baseline's per-gauge
   time-means. Validation 2: triangular-solve doy climatology vs the doy
   climatology of the directly summed daily series (all dams), plus a third,
   fully independent path for 5 dams (pure-python BFS + xarray reader).
4. 3-harmonic fit of C(doy) - qmean, weighted by the number of days per doy.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from collections import deque
from pathlib import Path

import icechunk as ic
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
import zarr
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.linalg import spsolve_triangular

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "output" / "damclim"
DAMS_CSV = REPO / "experiments/reservoir/release_head/dam_features.csv"
INFLOW_COLUMN = "inflow_mean_m3s"
QPRIME = "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic"
CONUS_ADJ = "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr"
GAGES_ADJ = "/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr"
GAGES_CSV = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"
BASELINE = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-47Z-train-and-test/baseline")

TRAIN_START, TRAIN_END = "1981-10-01", "1995-09-30"
FILL = 0.001  # ddrs/DDR discharge fill for divides absent from the Q' store
DAYS_PER_YEAR = 365.25  # src/routing/release.rs::DAYS_PER_YEAR
N_HARM = 3
SEED = 20260927

T0 = time.time()
TIMINGS: dict[str, float] = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def lap(name: str, t: float) -> float:
    now = time.time()
    TIMINGS[name] = round(now - t, 2)
    return now


def open_qprime():
    repo = ic.Repository.open(ic.local_filesystem_storage(QPRIME))
    session = repo.readonly_session("main")
    return session


def time_index(time_vals: np.ndarray, epoch: pd.Timestamp, date: str) -> int:
    idx = (pd.Timestamp(date) - epoch).days
    got = epoch + pd.Timedelta(days=int(time_vals[idx]))
    assert got == pd.Timestamp(date), (date, got)
    return idx


def main() -> None:
    t = time.time()
    report: dict = {}

    # ------------------------------------------------------------------ network
    g = zarr.open_group(CONUS_ADJ, mode="r")
    order = g["order"][:].astype(np.int64)
    i0 = g["indices_0"][:].astype(np.int64)  # downstream row
    i1 = g["indices_1"][:].astype(np.int64)  # upstream column
    n = order.size
    assert (i0 >= i1).all() and not (i0 == i1).any(), "adjacency not strictly lower-triangular"
    assert np.bincount(i1, minlength=n).max() <= 1, "a reach has more than one downstream"
    N = sp.csr_matrix((np.ones(i0.size), (i0, i1)), shape=(n, n))
    pos_of = pd.Index(order)
    assert pos_of.is_unique
    log(f"network: {n} reaches, {i0.size} edges")

    # --------------------------------------------------------------------- dams
    dams = pd.read_csv(DAMS_CSV)
    dam_comids = dams["COMID"].to_numpy(np.int64)
    assert len(np.unique(dam_comids)) == dam_comids.size, "duplicate dam COMIDs"
    dam_pos = pos_of.get_indexer(dam_comids)
    missing_net = dam_comids[dam_pos < 0]
    report["dams_total"] = int(dam_comids.size)
    report["dams_not_in_network"] = missing_net.tolist()
    if missing_net.size:
        log(f"WARNING {missing_net.size} dam COMIDs not in network; dropped")
    keep = dam_pos >= 0
    dam_comids, dam_pos = dam_comids[keep], dam_pos[keep]
    n_dams = dam_comids.size

    # ------------------------------------------------------------------ Q' store
    session = open_qprime()
    zg = zarr.open_group(session.store, mode="r")
    qr = zg["Qr"]
    assert tuple(qr.metadata.dimension_names) == ("divide_id", "time"), qr.metadata.dimension_names
    divide_id = zg["divide_id"][:].astype(np.int64)
    time_vals = zg["time"][:]
    units = zg["time"].attrs["units"]
    assert units.startswith("days since"), units
    epoch = pd.Timestamp(units.split("since")[1].strip())
    n_store, n_time = qr.shape
    assert n_store == divide_id.size and n_time == time_vals.size
    row_chunk = qr.metadata.chunk_grid.chunk_shape[0]
    log(f"Q' store: {n_store} divides x {n_time} days, epoch {epoch.date()}, row chunk {row_chunk}")

    tr0 = time_index(time_vals, epoch, TRAIN_START)
    tr1 = time_index(time_vals, epoch, TRAIN_END) + 1
    train_dates = pd.date_range(TRAIN_START, TRAIN_END, freq="D")
    assert train_dates.size == tr1 - tr0
    n_train = tr1 - tr0

    man = json.loads((BASELINE / "manifest.json").read_text())
    te_dates = pd.to_datetime(man["time_range_daily"])
    assert (np.diff(te_dates.values).astype("timedelta64[D]").astype(int) == 1).all()
    assert te_dates.size == man["n_days"]
    te0 = time_index(time_vals, epoch, str(te_dates[0].date()))
    te1 = te0 + te_dates.size
    log(f"train window idx [{tr0},{tr1}) = {n_train} days; baseline window "
        f"{te_dates[0].date()}..{te_dates[-1].date()} idx [{te0},{te1})")

    # store row -> network position (-1 if the divide is not in the network)
    row_net = pos_of.get_indexer(divide_id)
    # network position -> store row (-1 if missing from Q')
    net_row = np.full(n, -1, np.int64)
    net_row[row_net[row_net >= 0]] = np.nonzero(row_net >= 0)[0]
    in_q = net_row >= 0
    report["network_reaches_in_qprime"] = int(in_q.sum())
    report["network_reaches_missing_qprime"] = int((~in_q).sum())
    report["store_divides_not_in_network"] = int((row_net < 0).sum())
    dams_not_in_q = dam_comids[~in_q[dam_pos]]
    report["dams_own_divide_not_in_qprime"] = dams_not_in_q.tolist()
    log(f"network reaches in Q': {in_q.sum()} (missing -> {FILL} fill: {(~in_q).sum()}); "
        f"store divides not in network: {(row_net < 0).sum()}; dams whose own divide is not in Q': "
        f"{dams_not_in_q.size}")

    # doy one-hot (n_train x 366), 1-based doy
    doy = train_dates.dayofyear.to_numpy()
    M = sp.csr_matrix((np.ones(n_train), (np.arange(n_train), doy - 1)), shape=(n_train, 366))
    doy_count = np.bincount(doy - 1, minlength=366).astype(np.float64)
    assert (doy_count > 0).all()
    t = lap("setup", t)

    # --------------------------------------------- dam upstream sets (csgraph BFS)
    # N[down, up] = 1: following edges row -> col walks upstream.
    ups: list[np.ndarray] = []
    for p in dam_pos:
        ups.append(np.sort(breadth_first_order(N, int(p), directed=True, return_predecessors=False)))
    n_up = np.array([u.size for u in ups], np.int64)
    rows, cols = [], []
    n_up_missing = np.zeros(n_dams, np.int64)
    for d, u in enumerate(ups):
        r = net_row[u]
        n_up_missing[d] = int((r < 0).sum())
        r = r[r >= 0]
        rows.append(np.full(r.size, d))
        cols.append(r)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    D = sp.csc_matrix((np.ones(rows.size), (rows, cols)), shape=(n_dams, n_store))
    log(f"dam upstream sets: nnz(D) = {D.nnz}, n_up median {int(np.median(n_up))}, max {n_up.max()}")
    t = lap("upstream_bfs", t)

    # ------------------------------------------------------------ one data pass
    doy_sum = np.zeros((n_store, 366))
    mean_train = np.zeros(n_store)
    mean_test = np.zeros(n_store)
    dam_daily = np.zeros((n_dams, n_train))
    n_nan_train = n_nan_test = n_neg_train = 0
    D = D.tocsc()
    for r0 in range(0, n_store, row_chunk):
        r1 = min(r0 + row_chunk, n_store)
        blk = np.asarray(qr[r0:r1, tr0:tr1], dtype=np.float64)
        nan = np.isnan(blk)
        n_nan_train += int(nan.sum())
        if nan.any():
            blk[nan] = 0.0  # baseline convention (summed_q_prime.rs filters non-finite)
        n_neg_train += int((blk < 0).sum())
        doy_sum[r0:r1] = np.asarray((M.T @ blk.T).T)
        mean_train[r0:r1] = blk.mean(axis=1)
        dam_daily += D[:, r0:r1] @ blk
        del blk
        tb = np.asarray(qr[r0:r1, te0:te1], dtype=np.float64)
        nan = np.isnan(tb)
        n_nan_test += int(nan.sum())
        if nan.any():
            tb[nan] = 0.0
        mean_test[r0:r1] = tb.mean(axis=1)
        del tb
    dam_daily += FILL * n_up_missing[:, None]
    report["nan_values_train_window"] = n_nan_train
    report["nan_values_test_window"] = n_nan_test
    report["negative_values_train_window"] = n_neg_train
    log(f"data pass done: NaN train {n_nan_train}, NaN test {n_nan_test}, negative {n_neg_train}")
    t = lap("data_pass", t)

    # ------------------------------------------------------ triangular solve
    # columns: 0..365 doy mean, 366 train mean, 367 test mean, 368 ones, 369 in-Q'
    X = np.empty((n, 370))
    X[:, :366] = FILL
    X[:, 366] = FILL
    X[:, 367] = FILL
    sel = row_net >= 0
    X[row_net[sel], :366] = doy_sum[sel] / doy_count
    X[row_net[sel], 366] = mean_train[sel]
    X[row_net[sel], 367] = mean_test[sel]
    X[:, 368] = 1.0
    X[:, 369] = in_q.astype(np.float64)
    del doy_sum
    L = (sp.identity(n, format="csr") - N).tocsr()
    A = spsolve_triangular(L, X, lower=True)
    del X
    log("triangular solve done")
    t = lap("triangular_solve", t)

    # sanity: ones column == BFS upstream-set size
    tri_nup = np.rint(A[dam_pos, 368]).astype(np.int64)
    assert (tri_nup == n_up).all(), "triangular-solve upstream count != BFS set size"
    assert (np.rint(A[dam_pos, 369]).astype(np.int64) == n_up - n_up_missing).all()

    C = A[dam_pos, :366]  # (n_dams, 366) doy climatology, m3/s
    qmean = A[dam_pos, 366]
    # consistency: count-weighted mean of the doy climatology == training mean
    wmean = (C * doy_count).sum(1) / doy_count.sum()
    report["max_rel_diff_weighted_doy_mean_vs_qmean"] = float(np.max(np.abs(wmean / qmean - 1)))

    # --------------------------------------------------------- validation 2
    C_direct = (dam_daily @ M.toarray()) / doy_count  # sum first, then average
    rel2 = np.abs(C_direct / C - 1)
    qmean_direct = dam_daily.mean(1)
    rng = np.random.default_rng(SEED)
    # 5 dams spread over basin size: one random dam from each n_up quintile
    qs = np.quantile(n_up, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    five = []
    for k in range(5):
        cand = np.nonzero((n_up >= qs[k]) & (n_up <= qs[k + 1]))[0]
        cand = np.setdiff1d(cand, five)
        five.append(int(rng.choice(cand)))
    # Third, independent path for the 5: pure-python BFS on an upstream dict and
    # the xarray reader (DDR read_ic), 0.001 fill for missing divides.
    upstream_of: dict[int, list[int]] = {}
    for a, b in zip(i0.tolist(), i1.tolist()):
        upstream_of.setdefault(a, []).append(b)
    ds = xr.open_zarr(open_qprime().store, consolidated=False)
    v2_rows = []
    for d in five:
        seen = {int(dam_pos[d])}
        dq = deque([int(dam_pos[d])])
        while dq:
            x = dq.popleft()
            for y in upstream_of.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    dq.append(y)
        up_comids = order[np.fromiter(seen, np.int64)]
        present = up_comids[np.isin(up_comids, divide_id)]
        n_miss = up_comids.size - present.size
        q = (
            ds["Qr"]
            .sel(divide_id=np.sort(present), time=slice(TRAIN_START, TRAIN_END))
            .sum("divide_id")
            .values.astype(np.float64)
        ) + FILL * n_miss
        clim3 = pd.Series(q, index=train_dates).groupby(train_dates.dayofyear).mean().to_numpy()
        v2_rows.append(
            {
                "COMID": int(dam_comids[d]),
                "n_upstream_reaches": int(n_up[d]),
                "n_up_python_bfs": len(seen),
                "qmean_m3s": float(qmean[d]),
                "max_rel_diff_tri_vs_direct_Dsum": float(rel2[d].max()),
                "max_rel_diff_tri_vs_xarray_bfs": float(np.max(np.abs(clim3 / C[d] - 1))),
                "max_abs_diff_tri_vs_xarray_bfs_m3s": float(np.max(np.abs(clim3 - C[d]))),
            }
        )
    report["validation2_five_dams"] = v2_rows
    report["validation2_all_dams_max_rel_diff_doy"] = float(rel2.max())
    report["validation2_all_dams_max_rel_diff_qmean"] = float(np.max(np.abs(qmean_direct / qmean - 1)))
    log(f"validation 2: all-dam max rel diff doy {rel2.max():.3e}; five: "
        + ", ".join(f"{r['COMID']}:{r['max_rel_diff_tri_vs_xarray_bfs']:.2e}" for r in v2_rows))
    t = lap("validation2", t)

    # --------------------------------------------------------- validation 1
    gcsv = pd.read_csv(GAGES_CSV, dtype={"STAID": str})
    gcsv["STAID"] = gcsv["STAID"].str.zfill(8)
    staid2comid = dict(zip(gcsv["STAID"], gcsv["COMID"].astype(np.int64)))
    gids = [str(s).zfill(8) for s in man["gage_ids"]]
    pred = np.fromfile(BASELINE / "predictions.f32", dtype=np.float32).reshape(man["n_gauges"], man["n_days"])
    base_mean = pred.astype(np.float64).mean(1)
    del pred
    gcom = np.array([staid2comid[s] for s in gids], np.int64)
    gpos = pos_of.get_indexer(gcom)
    ours = np.where(gpos >= 0, A[np.maximum(gpos, 0), 367], np.nan)
    rel1 = np.abs(ours / base_mean - 1)
    # 20 random gauges
    pick = np.sort(rng.choice(len(gids), 20, replace=False))
    # gauge-subgraph upstream set vs CONUS BFS, for the 20 and for any failure
    gz = zarr.open_group(GAGES_ADJ, mode="r")

    def subgraph_set(staid: str) -> set[int]:
        sg = gz[staid]
        return set(sg["indices_0"][:].tolist()) | set(sg["indices_1"][:].tolist())

    def bfs_set(p: int) -> set[int]:
        return set(breadth_first_order(N, int(p), directed=True, return_predecessors=False).tolist())

    v1_rows = []
    for k in pick:
        s = gids[k]
        sub = subgraph_set(s)
        bfs = bfs_set(gpos[k]) if gpos[k] >= 0 else set()
        v1_rows.append(
            {
                "STAID": s,
                "COMID": int(gcom[k]),
                "baseline_mean_m3s": float(base_mean[k]),
                "tri_mean_m3s": float(ours[k]),
                "rel_diff": float(rel1[k]),
                "n_subgraph": len(sub),
                "n_conus_bfs": len(bfs),
                "same_set": sub == bfs,
            }
        )
    report["validation1_twenty"] = v1_rows
    report["validation1_twenty_max_rel_diff"] = float(np.max(rel1[pick]))
    fin = np.isfinite(rel1)
    report["validation1_all_gauges"] = {
        "n": int(len(gids)),
        "n_comid_not_in_network": int((gpos < 0).sum()),
        "max_rel_diff": float(np.nanmax(rel1)),
        "median_rel_diff": float(np.nanmedian(rel1)),
        "n_rel_diff_ge_1e-3": int((rel1[fin] >= 1e-3).sum()),
    }
    bad = np.nonzero(fin & (rel1 >= 1e-3))[0]
    bad_rows = []
    for k in bad[:50]:
        s = gids[k]
        sub = subgraph_set(s)
        bfs = bfs_set(gpos[k])
        bad_rows.append(
            {
                "STAID": s,
                "COMID": int(gcom[k]),
                "rel_diff": float(rel1[k]),
                "n_subgraph": len(sub),
                "n_conus_bfs": len(bfs),
                "only_in_subgraph": len(sub - bfs),
                "only_in_bfs": len(bfs - sub),
            }
        )
    report["validation1_failures"] = bad_rows
    log(f"validation 1: 20-gauge max rel diff {np.max(rel1[pick]):.3e}; all {len(gids)} gauges max "
        f"{np.nanmax(rel1):.3e}, n >= 1e-3: {(rel1[fin] >= 1e-3).sum()}")
    t = lap("validation1", t)

    # ------------------------------------------------------------ harmonic fit
    doys = np.arange(1, 367, dtype=np.float64)
    w = 2 * np.pi * doys / DAYS_PER_YEAR
    H = np.column_stack([f(k * w) for k in range(1, N_HARM + 1) for f in (np.sin, np.cos)])  # a1,b1,..
    Wt = doy_count / doy_count.sum()
    G = H.T @ (H * Wt[:, None])
    Y = (C - qmean[:, None])  # (n_dams, 366)
    beta = np.linalg.solve(G, (H * Wt[:, None]).T @ Y.T).T  # (n_dams, 6)
    fit = qmean[:, None] + beta @ H.T
    ss_res = ((C - fit) ** 2 * Wt).sum(1)
    ss_tot = ((C - qmean[:, None]) ** 2 * Wt).sum(1)
    r2 = 1 - ss_res / ss_tot
    daily_cv = dam_daily.std(1) / dam_daily.mean(1)

    out = pd.DataFrame(
        {
            "COMID": dam_comids,
            "n_upstream_reaches": n_up,
            "n_upstream_missing_qprime": n_up_missing,
            "qmean_m3s": qmean,
            "a1": beta[:, 0],
            "b1": beta[:, 1],
            "a2": beta[:, 2],
            "b2": beta[:, 3],
            "a3": beta[:, 4],
            "b3": beta[:, 5],
            "r2": r2,
            "seasonal_amp_ratio": (C.max(1) - C.min(1)) / qmean,
            "peak_doy": C.argmax(1) + 1,
            "daily_cv": daily_cv,
            "seasonal_amp_ratio_fit": (fit.max(1) - fit.min(1)) / qmean,
            "peak_doy_fit": fit.argmax(1) + 1,
        }
    )
    out.to_csv(OUT / "dam_inflow_clim.csv", index=False, float_format="%.6g")
    np.savez(
        OUT / "dam_inflow_clim_doy.npz",
        comids=dam_comids.astype(np.int64),
        doy_mean=C,
        qmean=qmean,
        doy_count=doy_count,
    )
    np.savez(
        OUT / "dam_inflow_daily_train.npz",
        comids=dam_comids.astype(np.int64),
        dates=train_dates.strftime("%Y-%m-%d").to_numpy().astype("U10"),
        q=dam_daily.astype(np.float32),
    )
    t = lap("fit_and_write", t)

    # ----------------------------------------------------------------- summary
    qq = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    summ = {}
    for col in ["qmean_m3s", "seasonal_amp_ratio", "seasonal_amp_ratio_fit", "r2", "daily_cv",
                "n_upstream_reaches"]:
        summ[col] = {str(q): float(out[col].quantile(q)) for q in qq}
    report["summary_quantiles"] = summ
    report["peak_doy_is_366"] = int((out["peak_doy"] == 366).sum())
    report["min_doy_is_366"] = int((C.argmin(1) == 365).sum())
    report["qmean_nonpositive"] = int((qmean <= 0).sum())
    report["qmean_lt_0p1"] = int((qmean < 0.1).sum())
    report["doy_min_nonpositive"] = int((C.min(1) <= 0).sum())
    report["dams_all_fill_upstream"] = int((n_up_missing == n_up).sum())
    report["dams_fill_frac_gt_0p5"] = int((n_up_missing / n_up > 0.5).sum())
    report["dams_fill_frac_quantiles"] = {str(q): float(np.quantile(n_up_missing / n_up, q)) for q in qq}
    report["timings_s"] = TIMINGS
    report["total_runtime_s"] = round(time.time() - T0, 1)
    report["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    (OUT / "damclim_report.json").write_text(json.dumps(report, indent=2))
    log(f"done in {report['total_runtime_s']} s, peak RSS {report['peak_rss_gb']} GB")


def join_inflow_mean() -> None:
    """Append (or replace) the `inflow_mean_m3s` column of DAMS_CSV from
    OUT/dam_inflow_clim.csv's `qmean_m3s`, textually: every other field of
    every line keeps its bytes. Every dam must have a value."""
    clim = {}
    with open(OUT / "dam_inflow_clim.csv", newline="") as f:
        lines = f.read().splitlines()
    head = lines[0].split(",")
    ci, qi = head.index("COMID"), head.index("qmean_m3s")
    for line in lines[1:]:
        cells = line.split(",")
        clim[cells[ci]] = cells[qi]
    raw = DAMS_CSV.read_bytes().decode()
    assert raw.endswith("\n")
    rows = raw[:-1].split("\n")
    header = rows[0].split(",")
    assert header[0] == "COMID"
    # The column is appended LAST (a re-run strips the old last field first),
    # so no other byte moves; a quoted dam name ("Bartletts Ferry, Main Dam")
    # never needs re-quoting. COMID, the first field, is never quoted.
    present = header[-1] == INFLOW_COLUMN
    assert present or INFLOW_COLUMN not in header, f"{INFLOW_COLUMN} must be the last column"
    out = []
    for i, row in enumerate(rows):
        if present:
            row = row.rsplit(",", 1)[0]
        value = INFLOW_COLUMN if i == 0 else clim[row.split(",", 1)[0]]
        out.append(f"{row},{value}")
    DAMS_CSV.write_bytes(("\n".join(out) + "\n").encode())
    log(f"joined {INFLOW_COLUMN} into {DAMS_CSV} ({len(rows) - 1} dams)")


if __name__ == "__main__":
    if "--join-only" not in sys.argv[1:]:
        OUT.mkdir(parents=True, exist_ok=True)
        main()
    join_inflow_mean()
