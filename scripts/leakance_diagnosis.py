#!/usr/bin/env python3
"""Leakance low-zeta diagnosis — 7-hypothesis falsification battery.

Spec: docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md
Run:  cd ~/projects/ddr && uv run python <ddrs>/scripts/leakance_diagnosis.py

H1 structural ceiling   H2 driving head      H3 KAN variance collapse
H4 gauge bias           H5 equifinality      H6 fractional loss
H7 model form (connected-only law)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
ARM_IDS = {
    "hourly_on": "2026-07-01T13-43-32Z-train-and-test",
    "daily_on": "2026-07-01T21-20-27Z-train-and-test",
    "hourly_off": "2026-06-23T02-49-12Z-conus-hourly-train-and-test",
    "daily_off": "2026-06-05T01-41-16Z-train-and-test",
}
K_D_CEIL = 1e-6          # current range top (1/s)
K_D_WIDE = 1e-4          # literature sand-bed leakance (litreview §A1)
D_GW_FLOOR = -2.0        # current d_gw range bottom (m)
D_GW_CEIL = 2.0
ZETA_BAR = 0.01          # GO/NO-GO magnitude bar (m3/s)
ATTRS = ["aridity", "permeability", "Porosity", "log10_uparea", "meanP", "meanslope"]


def sec(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def q(x: np.ndarray, ps=(5, 25, 50, 75, 95)) -> str:
    return " ".join(f"p{p}={np.nanpercentile(x, p):.4g}" for p in ps)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(a[m], b[m]).statistic)


class Arm:
    """One run's kan_parameters.nc: full-CONUS params + eval-network diagnostics."""

    def __init__(self, run_dir: Path):
        self.ds = xr.open_dataset(run_dir / "kan_parameters.nc")
        self.comid = self.ds["COMID"].values.astype(np.int64)
        if "COMID_eval" in self.ds:
            self.comid_eval = self.ds["COMID_eval"].values.astype(np.int64)
            order = np.argsort(self.comid)
            pos = np.searchsorted(self.comid, self.comid_eval, sorter=order)
            self.eval_ix = order[pos]
            assert (self.comid[self.eval_ix] == self.comid_eval).all(), "COMID_eval not a subset of COMID"

    def on_eval(self, var: str) -> np.ndarray:
        """A full-CONUS variable subset to the eval network, or an eval-native one."""
        v = self.ds[var]
        return v.values[self.eval_ix] if v.dims == ("COMID",) else v.values


def attach_attributes(attrs_path: Path, comids: np.ndarray) -> dict[str, np.ndarray]:
    ds = xr.open_dataset(attrs_path)
    acom = ds["COMID"].values.astype(np.int64)
    order = np.argsort(acom)
    pos = np.searchsorted(acom, comids, sorter=order)
    ix = order[np.clip(pos, 0, len(acom) - 1)]
    ok = acom[ix] == comids
    out = {}
    for name in ATTRS:
        v = ds[name].values.astype(np.float64)[ix]
        v[~ok] = np.nan
        out[name] = v
    print(f"attributes matched for {ok.mean() * 100:.1f}% of {len(comids)} reaches")
    r = spearman(out["aridity"], out["meanP"])
    print(f"aridity vs meanP spearman = {r:.2f} → aridity is a "
          f"{'DRYNESS' if r < 0 else 'WETNESS'} index")
    out["_aridity_is_dryness"] = np.array([r < 0])
    return out


def verdict(name: str, supported: bool | None, detail: str) -> str:
    tag = "INCONCLUSIVE" if supported is None else ("SUPPORTED" if supported else "REFUTED")
    line = f"[{tag}] {name}: {detail}"
    print(f"\n  → {line}")
    return line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=RUNS)
    ap.add_argument("--attributes", type=Path,
                    default=Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"))
    ap.add_argument("--gages-csv", type=Path,
                    default=Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"))
    args = ap.parse_args()

    arms = {k: Arm(args.runs_dir / rid) for k, rid in ARM_IDS.items()}
    hon = arms["hourly_on"]
    verdicts: list[str] = []

    zeta = np.abs(hon.on_eval("zeta"))
    depth = hon.on_eval("depth_mean")
    area_z = hon.on_eval("area_z_mean")
    q_mean = hon.on_eval("q_mean")
    k_d = hon.on_eval("K_D")
    d_gw = hon.on_eval("d_gw")
    factor = hon.on_eval("leakance_factor")
    attrs = attach_attributes(args.attributes, hon.comid_eval)

    # ---------------- H1: structural ceiling ----------------
    sec("H1 — structural ceiling: can zeta exceed the bar inside the current box?")
    zmax_now = 1.0 * area_z * K_D_CEIL * (depth - D_GW_FLOOR)
    zmax_wide = 1.0 * area_z * K_D_WIDE * (depth - D_GW_FLOOR)
    frac_now = float((zmax_now > ZETA_BAR).mean())
    frac_wide = float((zmax_wide > ZETA_BAR).mean())
    util = zeta / np.maximum(zmax_now, 1e-30)
    print(f"zeta_max within CURRENT box:  {q(zmax_now)} | frac > {ZETA_BAR}: {frac_now * 100:.1f}%")
    print(f"zeta_max with K_D={K_D_WIDE}: {q(zmax_wide)} | frac > {ZETA_BAR}: {frac_wide * 100:.1f}%")
    print(f"utilization zeta/zeta_max:    {q(util)}")
    verdicts.append(verdict(
        "H1 structural ceiling",
        frac_now < 0.5,
        f"only {frac_now * 100:.1f}% of reaches CAN exceed {ZETA_BAR} m³/s inside the current box "
        f"(vs {frac_wide * 100:.1f}% at K_D={K_D_WIDE}); median utilization {np.median(util):.2f}",
    ))

    # ---------------- H2: driving-head starvation ----------------
    sec("H2 — driving head (depth_mean − d_gw)")
    head = depth - d_gw
    print(f"depth_mean: {q(depth)}  |  d_gw: {q(d_gw)}  |  head: {q(head)}")
    f_neg, f_small = float((head <= 0).mean()), float((head < 0.1).mean())
    print(f"head ≤ 0 (gaining/neutral at the mean): {f_neg * 100:.1f}%   head < 0.1 m: {f_small * 100:.1f}%")
    verdicts.append(verdict(
        "H2 driving-head starvation",
        f_small > 0.5,
        f"{f_small * 100:.1f}% of reaches have <0.1 m mean driving head ({f_neg * 100:.1f}% ≤ 0)",
    ))

    # ---------------- H3: KAN variance collapse ----------------
    sec("H3 — KAN variance collapse (leakance vs routing params, full CONUS)")
    for name in ["K_D", "d_gw", "leakance_factor", "n", "q_spatial", "x_storage"]:
        if name not in hon.ds:
            print(f"{name:16s} SKIPPED (not in this run)")
            continue
        v = hon.ds[name].values.astype(np.float64)
        v = np.log10(v) if name == "K_D" else v
        iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
        rng = np.nanpercentile(v, 95) - np.nanpercentile(v, 5)
        print(f"{name:16s} median={np.nanmedian(v):9.4g}  IQR={iqr:9.4g}  IQR/p5-p95={iqr / max(rng, 1e-30):.3f}")
    print("\nspearman corr of leakance params vs attributes (eval network):")
    max_r = 0.0
    for pname, pv in [("K_D", np.log10(np.maximum(k_d, 1e-30))), ("d_gw", d_gw), ("factor", factor),
                      ("zeta", np.log10(np.maximum(zeta, 1e-12)))]:
        rs = {a: spearman(pv, attrs[a]) for a in ATTRS}
        finite = [abs(r) for r in rs.values() if np.isfinite(r)]
        if pname != "zeta" and finite:
            max_r = max(max_r, max(finite))
        print(f"  {pname:8s} " + "  ".join(f"{a}={r:+.2f}" for a, r in rs.items()))
    verdicts.append(verdict(
        "H3 KAN variance collapse",
        max_r < 0.2,
        f"max |spearman| of any leakance param vs any attribute = {max_r:.2f} "
        "(<0.2 ⇒ no learned spatial structure; ≥0.4 ⇒ clearly attribute-driven)",
    ))

    # ---------------- H4: gauge bias / gradient starvation ----------------
    sec("H4 — stratification by gauged-ness, upstream area, aridity")
    import csv

    with open(args.gages_csv) as f:
        gauged_comids = {int(row["COMID"]) for row in csv.DictReader(f)}
    gmask = np.isin(hon.comid_eval, list(gauged_comids))
    print(f"gauged reaches on eval network: {gmask.sum()} / {len(gmask)}")
    for label, m in [("gauged", gmask), ("ungauged", ~gmask)]:
        print(f"  {label:9s} median|zeta|={np.median(zeta[m]):.3e}  frac>{ZETA_BAR}: "
              f"{(zeta[m] > ZETA_BAR).mean() * 100:.1f}%  median q={np.median(q_mean[m]):.3g}")
    arid = attrs["aridity"]
    dry = (arid >= np.nanpercentile(arid, 67)) if attrs["_aridity_is_dryness"][0] else (arid <= np.nanpercentile(arid, 33))
    wet = ~dry & np.isfinite(arid)
    print(f"  dry tercile  median|zeta|={np.median(zeta[dry]):.3e}  frac>{ZETA_BAR}: {(zeta[dry] > ZETA_BAR).mean() * 100:.1f}%")
    print(f"  wet tercile  median|zeta|={np.median(zeta[wet]):.3e}  frac>{ZETA_BAR}: {(zeta[wet] > ZETA_BAR).mean() * 100:.1f}%")
    r_up = spearman(np.log10(np.maximum(zeta, 1e-12)), attrs["log10_uparea"])
    print(f"  spearman log|zeta| vs log10_uparea = {r_up:+.2f}")
    dry_wet_ratio = np.median(zeta[dry]) / max(np.median(zeta[wet]), 1e-30)
    verdicts.append(verdict(
        "H4 gauge bias / gradient starvation",
        dry_wet_ratio < 2.0 and r_up > 0.5,
        f"dry/wet median-zeta ratio = {dry_wet_ratio:.2f} (physics says dry ≫ wet), "
        f"zeta–uparea corr {r_up:+.2f} (zeta tracks river size, not aridity)",
    ))

    # ---------------- H5: equifinality with routing params ----------------
    sec("H5 — did n / x_storage shift between paired ON/OFF runs?")
    any_shift = False
    for pair, on_key, off_key in [("hourly", "hourly_on", "hourly_off"), ("daily", "daily_on", "daily_off")]:
        on, off = arms[on_key], arms[off_key]
        assert (on.comid == off.comid).all(), f"{pair}: COMID order mismatch"
        for pname in ["n", "x_storage"]:
            if pname not in off.ds:
                print(f"  {pair:6s} Δ{pname:9s} SKIPPED (not learned in the OFF run — constant)")
                continue
            a, b = on.ds[pname].values, off.ds[pname].values
            d = a - b
            shift = abs(np.median(d)) / max(np.nanpercentile(np.abs(b - np.median(b)), 75), 1e-30)
            any_shift |= bool(shift > 0.5)
            print(f"  {pair:6s} Δ{pname:9s} median={np.median(d):+.4g}  IQR={np.percentile(d, 75) - np.percentile(d, 25):.4g}  "
                  f"median-shift/param-IQR={shift:.2f}")
    verdicts.append(verdict(
        "H5 equifinality",
        any_shift,
        "routing params shifted materially between ON/OFF (shift > 0.5 IQR) — "
        "n/storage absorb what leakance would explain" if any_shift else
        "routing params essentially unchanged between ON/OFF pairs",
    ))

    # ---------------- H6: fractional loss ----------------
    sec("H6 — |zeta| / q_mean (is the loss non-trivial RELATIVE to local flow?)")
    frac_loss = zeta / np.maximum(q_mean, 1e-4)
    print(f"|zeta|/q: {q(frac_loss)}")
    f1, f5 = float((frac_loss > 0.01).mean()), float((frac_loss > 0.05).mean())
    print(f"frac loss > 1% of local flow: {f1 * 100:.1f}%   > 5% (gauge-detectability band): {f5 * 100:.1f}%")
    verdicts.append(verdict(
        "H6 wrong yardstick",
        f1 > 0.3,
        f"{f1 * 100:.1f}% of reaches lose >1% of local flow ({f5 * 100:.1f}% >5%) — "
        "the absolute 0.01 m³/s bar under/over-states the term's activity",
    ))

    # ---------------- H7: model form (connected-only law) ----------------
    sec("H7 — d_gw boundary-pinning where disconnection is plausible (dry reaches)")
    lo, hi = D_GW_FLOOR, D_GW_CEIL
    pin_hi = d_gw > hi - 0.05 * (hi - lo)
    pin_lo = d_gw < lo + 0.05 * (hi - lo)
    print(f"d_gw within 5% of bounds: floor {pin_lo.mean() * 100:.1f}%  ceiling {pin_hi.mean() * 100:.1f}% (overall)")
    print(f"  dry tercile: floor {pin_lo[dry].mean() * 100:.1f}%  ceiling {pin_hi[dry].mean() * 100:.1f}%")
    print(f"  wet tercile: floor {pin_lo[wet].mean() * 100:.1f}%  ceiling {pin_hi[wet].mean() * 100:.1f}%")
    dry_pin = float((pin_lo | pin_hi)[dry].mean())
    verdicts.append(verdict(
        "H7 model-form error",
        dry_pin > 0.3,
        f"{dry_pin * 100:.1f}% of dry-tercile reaches pin d_gw at a bound — the linear "
        "connected-regime law is straining toward the saturating (disconnected) regime",
    ))

    # ---------------- summary ----------------
    sec("SUMMARY (suggested verdicts — final judgment in the findings doc)")
    for v in verdicts:
        print(f"  {v}")


if __name__ == "__main__":
    main()
