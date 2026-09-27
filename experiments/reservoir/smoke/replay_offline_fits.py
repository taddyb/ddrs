"""Fixed dam tables that replay the offline per-gauge release fits in the engine.

The offline rule-curve review fitted, per smoke gauge, storage-release laws on the routed no-dam
flow at the gauge (``pred_1981_2010.zarr`` from ``run_smoke_eval.sh``), training WY1983-1995,
scoring WY1996-2010 (``rc.py`` / ``run_fits.py`` of that review -> ``fits_by_gauge.csv``). This
script writes two ``reservoir_release: fixed`` tables for ddrs so the same laws can be routed in
the engine with the routing head frozen (``config/experiments/dam_release_smoke_replay_L{2,4}.yaml``):

  fixed_L2.csv   plain bucket:          COMID, T_days = L2_p_T0 (days), year_completed
  fixed_L4.csv   harmonic rule curve:   COMID, T_days = L4_p_T0, c1s, c1c, c2s, c2c,
                                        inflow_mean_m3s, year_completed

Only the dams ON their gauge's own reach (``expected_release_fit.csv``: role == dam and
on_reach true; 214 gauges) go in; every other dam is absent, i.e. an ordinary channel. The dam's
COMID is ``smoke_gauges.csv`` ``dam_COMID``. Twelve dam COMIDs sit on the reach of two gauges each
(214 gauges, 202 dams): for those the fit of the gauge whose drainage area is closest to the dam
reach's upstream area (min |log(area_km2 / dam_reach_uparea_km2)|) is kept. year_completed and
inflow_mean_m3s come from ``experiments/reservoir/release_head/dam_features.csv``.

Conventions, offline vs engine (checked against ``rc.py`` and ``src/routing/release.rs``):

- Flux sign. Offline the bucket runs on ``I' = I - r`` with
  ``r = Ibar_off * (c1s sin w + c1c cos w + c2s sin 2w + c2c cos 2w)``: positive r is stored.
  The engine takes ``r = (S0_{t+1} - S0_t)/dt = Ibar_eng * (same harmonics)`` off the dam row's
  lateral inflow, ``q'_eff = q' - r``: positive r is stored. Same sign, same basis order
  (c1s, c1c, c2s, c2c).
- Flux scale. Offline ``Ibar_off`` is the training-period (WY1983-1995) mean of the routed no-dam
  flow at the GAUGE (``fits_by_gauge.csv`` ``Ibar``). The engine's ``Ibar_eng`` is the dam's
  ``inflow_mean_m3s`` (training-period mean of the upstream-summed Q' at the dam COMID). The flux
  in m3/s is kept: ``c_engine = c_offline * Ibar_off / inflow_mean_m3s``.
- Phase. Offline ``w = 2 pi doy / 365.25`` with ``doy`` the pandas ``dayofyear`` of each DAILY
  value (1-based), the flux constant over the day. The engine's rule curve is hourly with a
  CONTINUOUS phase, ``w = Omega * (seconds since 1970-01-01 00:00) + 2 pi / 365.25``
  (``release::rule_curve_phase_start``), which is day-of-year 1 at 1970-01-01 and stays within
  about +/-0.75 d of day of year (the leap cycle). A day's mean engine flux therefore sits at
  ``doy + 0.5 d`` (+/- the leap drift) against the offline ``doy``: at most ~1.25 d of phase,
  0.02 rad for k = 1 and 0.04 rad for k = 2, a small fraction of the flux; not corrected.
- Bucket. Offline: daily implicit Euler on ``S = T*Q (+ S0)`` applied to the routed flow at the
  gauge, outflow floored at 0 with the floor fed back into storage. Engine: the hourly additive dam
  row (``params.reservoir_dam_row: additive``), the reach's own Muskingum channel plus ``T*Q``,
  with the S28 discharge clamp (no feedback). T0 is used as fitted (days, >= 0.05 d, above the
  fixed reader's one-hour floor).

Run from the repo root (any python3 with the standard library):

  python3 experiments/reservoir/smoke/replay_offline_fits.py \
      --fits /home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve/fits_by_gauge.csv
"""
import argparse
import csv
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", required=True, help="fits_by_gauge.csv of the offline rule-curve review")
    ap.add_argument("--fit-table", default=os.path.join(HERE, "expected_release_fit.csv"))
    ap.add_argument("--gauges", default=os.path.join(HERE, "smoke_gauges.csv"))
    ap.add_argument("--dam-features", default=os.path.join(REPO, "experiments/reservoir/release_head/dam_features.csv"))
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    fit = {r["STAID"]: r for r in csv.DictReader(open(args.fit_table))}
    gauges = {r["STAID"]: r for r in csv.DictReader(open(args.gauges))}
    fits = {r["STAID"]: r for r in csv.DictReader(open(args.fits))}
    feats = {r["COMID"]: r for r in csv.DictReader(open(args.dam_features))}

    on_reach = [s for s, r in fit.items() if r["role"] == "dam" and r["on_reach"] == "True"]
    by_dam = defaultdict(list)
    for s in on_reach:
        by_dam[str(int(float(gauges[s]["dam_COMID"])))].append(s)

    def closeness(s):
        g = gauges[s]
        return abs(math.log(float(g["area_km2"]) / float(g["dam_reach_uparea_km2"])))

    rows = []
    shared = []
    for comid, staids in sorted(by_dam.items(), key=lambda kv: int(kv[0])):
        staid = min(staids, key=closeness)
        if len(staids) > 1:
            shared.append((comid, staids, staid))
        f = fits[staid]
        d = feats[comid]
        ibar_off = float(f["Ibar"])
        ibar_eng = float(d["inflow_mean_m3s"])
        c_off = [float(f[f"L4_p_{k}"]) for k in ("c1s", "c1c", "c2s", "c2c")]
        scale = ibar_off / ibar_eng
        rows.append(dict(
            COMID=comid,
            STAID=staid,
            year_completed=d["year_completed"],
            L2_T0=float(f["L2_p_T0"]),
            L4_T0=float(f["L4_p_T0"]),
            c_engine=[c * scale for c in c_off],
            c_offline=c_off,
            ibar_off=ibar_off,
            ibar_eng=ibar_eng,
        ))

    l2 = os.path.join(args.out_dir, "fixed_L2.csv")
    with open(l2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["COMID", "T_days", "year_completed", "STAID"])
        for r in rows:
            w.writerow([r["COMID"], repr(r["L2_T0"]), r["year_completed"], r["STAID"]])
    l4 = os.path.join(args.out_dir, "fixed_L4.csv")
    with open(l4, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["COMID", "T_days", "c1s", "c1c", "c2s", "c2c", "inflow_mean_m3s", "year_completed", "STAID",
                    "Ibar_offline_m3s", "c1s_offline", "c1c_offline", "c2s_offline", "c2c_offline"])
        for r in rows:
            w.writerow([r["COMID"], repr(r["L4_T0"]), *map(repr, r["c_engine"]), repr(r["ibar_eng"]),
                        r["year_completed"], r["STAID"], repr(r["ibar_off"]), *map(repr, r["c_offline"])])

    ratio = sorted(r["ibar_off"] / r["ibar_eng"] for r in rows)
    amp = sorted(math.hypot(*r["c_engine"][:2]) + math.hypot(*r["c_engine"][2:]) for r in rows)
    med = lambda v: v[len(v) // 2]
    print(f"{len(on_reach)} on-reach dam gauges -> {len(rows)} dams ({len(shared)} dams shared by two gauges; "
          f"kept the gauge closest in area):")
    for comid, staids, kept in shared:
        print(f"  COMID {comid}: gauges {staids}, kept {kept}")
    print(f"Ibar_offline / inflow_mean_m3s: median {med(ratio):.3f}, range [{ratio[0]:.3f}, {ratio[-1]:.3f}]")
    print(f"engine harmonic amplitude |c1| + |c2|: median {med(amp):.3f}, max {amp[-1]:.3f}")
    print(f"wrote {l2}\nwrote {l4}")


if __name__ == "__main__":
    main()
