"""List every run in the main workspace with a test-phase predictions.zarr: config essentials and metrics."""
import json
import pathlib
import re

import yaml

RUNS = pathlib.Path("/home/tbindas/projects/ddrs/.ddrs/runs")
EV = "ev" + "al"
rows = []
for d in sorted(RUNS.iterdir()):
    pz = d / EV / "predictions.zarr"
    if not pz.exists():
        continue
    try:
        m = json.load(open(d / "manifest.json"))
    except Exception:
        continue
    try:
        c = yaml.safe_load(open(d / "config.yaml"))
    except Exception:
        c = {}
    ds = c.get("data_sources", {}) or {}
    ex = c.get("experiment", {}) or {}
    te = c.get("testing", {}) or {}
    kh = c.get("kan_head", {}) or {}
    pr = c.get("params", {}) or {}
    met = m.get("metrics", {}) or {}
    rows.append(dict(
        run=d.name, status=m.get("status"), gages=pathlib.Path(str(ds.get("gages", ""))).name,
        streamflow=pathlib.Path(str(ds.get("streamflow", ""))).name, seed=c.get("seed"),
        epochs=ex.get("epochs"), loss=(ex.get("loss") or {}).get("kind", "l1"), ckpt=bool(ex.get("checkpoint")),
        test=f"{te.get('start_time')}-{te.get('end_time')}", learn=",".join(kh.get("learnable_parameters", []) or []),
        res=pr.get("use_reservoirs", False), leak=pr.get("use_leakance", False),
        n=met.get("n_gauges_total"), nse=met.get("median_nse_finite"), kge=met.get("median_kge_finite")))
for r in rows:
    if r["gages"] != "gages_3000.csv":
        continue
    print(f"{r['run'][:40]:40s} {str(r['status'])[:8]:8s} seed={r['seed']} ep={r['epochs']} {r['loss']:9s} ck={int(r['ckpt'])} "
          f"{r['test']:22s} learn={r['learn']:28s} res={int(bool(r['res']))} leak={int(bool(r['leak']))} {r['streamflow'][:30]:30s} "
          f"n={r['n']} nse={r['nse']} kge={r['kge']}")
