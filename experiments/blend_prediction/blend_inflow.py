"""Per-gauge summed inflow of the three-product blend, computed offline (no training).

Exact route. Reads the three icechunk Q' stores named in
config/experiments/sr_n0_gamma_blend.yaml (multi-product-forcing worktree) over the
test window, blends them per divide and day with the rule of
src/data/store/blend.rs (read only):

    q[t, j] = sum_m a[m, j] q_m[t, j],   a[m, j] = w_m c[m, j] / sum_k w_k c[k, j]

where c[m, j] = 1 when store m contains divide j, and a divide no store contains
reads the 0.001 fill. By linearity the gauge sum is accumulated store by store,
so no blended per-divide array is ever held in memory.

For every gauge of the members' summed-Q' baseline (2,698 gauges, the gauge
adjacency `merit_gages_conus_adjacency.zarr`, upstream set = CONUS `order` at the
union of the subgraph's COO indices, as `GageSubgraph::upstream_comids`) it writes:

    S_blend           the blend's summed inflow
    S_<member>        each member's raw summed inflow (0.001 fill where the store
                      lacks a divide), which must reproduce that member run's
                      baseline/predictions.f32 (checked and printed)
    full_cover[g]     every upstream divide is in all three stores

    /home/tbindas/projects/ddr/.venv/bin/python experiments/blend_prediction/blend_inflow.py --out <dir>
"""
import argparse
import json
import time
from pathlib import Path

import icechunk
import numpy as np
import scipy.sparse as sp
import zarr

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
CONUS = "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr"
GAGES_ADJ = "/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr"
MEMBERS = [
    # label, store, blend weight, member run whose baseline the raw sum must reproduce
    ("uh", "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic", 0.50,
     "2026-09-27T07-29-47Z-train-and-test"),
    ("dist", "/mnt/ssd1/data/icechunk/daily_dhbv2_distributed_aorc2f_merit_unit_catchments_rechunked.ic", 0.25,
     "2026-09-13T17-22-30Z-train-and-test"),
    ("lstm", "/mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic", 0.25,
     "2026-09-13T13-55-03Z-train-and-test"),
]
START, N_DAYS = np.datetime64("1995-10-01"), 5479  # WY1996-2010, both ends inclusive
FILL = 0.001
BLOCK = 3080 * 8


def open_store(path):
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    return zarr.open_group(repo.readonly_session("main").store, mode="r")


def time_index(g):
    t = np.asarray(g["time"][:])
    units = g["time"].attrs["units"]
    origin = np.datetime64(units.split("since")[1].strip()[:10])
    assert np.all(np.diff(t) == 1), "time axis not contiguous"
    i0 = int((START - origin).astype(int) - t[0])
    assert 0 <= i0 and i0 + N_DAYS <= t.size, (units, t[0], t.size)
    return i0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    conus = zarr.open_group(CONUS, mode="r")
    order = np.asarray(conus["order"][:], dtype=np.int64)
    n = order.size
    pos_of = {int(c): i for i, c in enumerate(order)}

    bm = json.loads((RUNS / MEMBERS[0][3] / "baseline" / "manifest.json").read_text())
    gauges = list(bm["gage_ids"])
    gz = zarr.open_group(GAGES_ADJ, mode="r")
    rows, cols = [], []
    for gi, s in enumerate(gauges):
        sg = gz[s]
        p = np.union1d(np.asarray(sg["indices_0"][:]), np.asarray(sg["indices_1"][:]))
        rows.append(np.full(p.size, gi)); cols.append(p)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(len(gauges), n)).tocsc()
    n_up = np.asarray(A.sum(1)).ravel()
    print(f"{len(gauges)} gauges, {n:,} CONUS reaches, upstream divides per gauge median {np.median(n_up):.0f}")

    stores = [open_store(p) for _, p, _, _ in MEMBERS]
    cover = np.zeros((len(MEMBERS), n), dtype=bool)
    store_pos = []
    for k, g in enumerate(stores):
        d = np.asarray(g["divide_id"][:], dtype=np.int64)
        assert np.unique(d).size == d.size, "duplicate divide ids"
        sp_ = np.array([pos_of.get(int(c), -1) for c in d])
        cover[k, sp_[sp_ >= 0]] = True
        store_pos.append(sp_)
        print(f"  {MEMBERS[k][0]:>5}: {d.size:,} divides, {int((sp_ >= 0).sum()):,} in the CONUS network")
    w = np.array([m[2] for m in MEMBERS])[:, None]
    denom = (w * cover).sum(0)
    a = np.where(denom > 0, w * cover / np.where(denom > 0, denom, 1.0), 0.0)  # (members, n)
    none = ~cover.any(0)
    allc = cover.all(0)
    print(f"  CONUS reaches covered by all three {allc.sum():,}, by none {none.sum():,}")

    full_cover = (A @ (~allc).astype(float)) == 0
    print(f"  gauges with every upstream divide in all three stores: {full_cover.sum()} of {len(gauges)}")

    S = {m[0]: np.zeros((len(gauges), N_DAYS)) for m in MEMBERS}
    S_blend = np.zeros((len(gauges), N_DAYS))
    nan_counts = {}
    for k, (lab, path, wk, _) in enumerate(MEMBERS):
        g = stores[k]; Qr = g["Qr"]
        i0 = time_index(g)
        nd = Qr.shape[0]
        n_nan = 0
        t = time.time()
        for r0 in range(0, nd, BLOCK):
            r1 = min(nd, r0 + BLOCK)
            p = store_pos[k][r0:r1]
            keep = p >= 0
            if not keep.any():
                continue
            q = np.asarray(Qr[r0:r1, i0:i0 + N_DAYS], dtype=np.float64)[keep]
            p = p[keep]
            Ab = A[:, p]
            used = np.asarray(Ab.sum(0)).ravel() > 0
            n_nan += int(np.isnan(q[used]).sum())
            q = np.nan_to_num(q, nan=0.0)  # baseline nansum semantics
            S[lab] += Ab @ q
            S_blend += Ab @ (a[k, p][:, None] * q)
        S[lab] += FILL * (A @ (~cover[k]).astype(float))[:, None]
        nan_counts[lab] = n_nan
        print(f"  read {lab} in {time.time() - t:.0f} s, NaN values on upstream divides in window: {n_nan}")
    S_blend += FILL * (A @ none.astype(float))[:, None]

    # The raw per-member sum must reproduce each member run's baseline.
    check = {}
    for lab, _, _, rid in MEMBERS:
        b = RUNS / rid / "baseline"
        m = json.loads((b / "manifest.json").read_text())
        assert list(m["gage_ids"]) == gauges, f"{rid}: baseline gauge order differs"
        ref = np.fromfile(b / "predictions.f32", dtype="<f4").reshape(len(gauges), N_DAYS).astype(np.float64)
        rel = np.abs(S[lab] - ref) / np.maximum(np.abs(ref), 1e-3)
        check[lab] = {"run": rid, "max_rel": float(rel.max()), "p99_rel": float(np.quantile(rel, 0.99))}
        print(f"  {lab} raw sum vs {rid} baseline: max rel diff {rel.max():.2e}, p99 {np.quantile(rel, .99):.2e}")

    obs = np.fromfile(RUNS / MEMBERS[0][3] / "baseline" / "observations.f32", dtype="<f4").reshape(len(gauges), N_DAYS)
    np.savez_compressed(out / "blend_inflow.npz", gauges=np.array(gauges), start=str(START), n_days=N_DAYS,
                        S_blend=S_blend.astype(np.float32), obs=obs,
                        full_cover=full_cover, n_up=n_up,
                        **{f"S_{k}": v.astype(np.float32) for k, v in S.items()})
    allc_comids = order[allc]
    np.save(out / "comids_in_all_three_stores.npy", allc_comids)
    (out / "blend_inflow.json").write_text(json.dumps({
        "route": "exact (icechunk stores read directly, blend.rs renormalisation)",
        "window": [str(START), str(START + N_DAYS - 1)], "n_gauges": len(gauges),
        "n_full_cover_gauges": int(full_cover.sum()), "n_conus_reaches_all_three": int(allc.sum()),
        "n_conus_reaches_none": int(none.sum()), "nan_counts": nan_counts, "baseline_check": check,
        "members": [{"label": l, "path": p, "weight": w_, "run": r} for l, p, w_, r in MEMBERS],
    }, indent=2))
    print(f"wrote {out}/blend_inflow.npz")


if __name__ == "__main__":
    main()
