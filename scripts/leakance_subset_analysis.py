"""Leakance × hourly-disaggregation 2×2 subset analysis.

Computes the losing-stream subset GO/NO-GO verdict for the leakance experiment.
Run this AFTER the two leakance-ON cells finish training:

    cd ~/projects/ddr
    uv run python ~/projects/ddrs/scripts/leakance_subset_analysis.py \\
        --hourly-on  2026-XX-XXTXX-XXZ-conus-hourly-leakance-train-and-test \\
        --daily-on   2026-XX-XXTXX-XXZ-conus-daily-leakance-train-and-test \\
        --hourly-off 2026-06-23T02-49-12Z-conus-hourly-train-and-test \\
        --daily-off  2026-06-05T01-41-16Z-train-and-test

The script expects the standard ddrs run-directory layout produced by
`ddrs run --workflow train-and-test`:

    <run_dir>/
        eval/predictions.zarr      # trained-model predictions (zarr v3 group)
            predictions/           # float64 array [n_gauges, n_days]
            observations/          # float64 array [n_gauges, n_days]
            gage_ids/              # uint8 array [n_gauges, 8] (fixed-width ASCII)
        baseline/
            predictions.f32        # raw float32, row-major [n_gauges, n_days]
            observations.f32       # raw float32, row-major [n_gauges, n_days]
            manifest.json          # keys: n_gauges, n_days, gage_ids, metrics, ...
        manifest.json              # run-level manifest (status, metrics, ...)

GO verdict (per spec §2026-06-29-leakance-hourly-feasibility-design.md):
    GO   = NSE or KGE improves on losing-stream subset (hourly ON−OFF > 0)
           AND |zeta| > 0.01 m³/s on a meaningful fraction of subset reaches
           AND effect absent/weaker in daily arm
    NO-GO = otherwise (skill neutral/negative, or water-deletion everywhere with
            no spatial coherence, or floor pile-up at K_D lower bound)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_baseline(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load baseline summed-Q′ predictions, observations, and gage IDs.

    Returns (predictions [n_gauges, n_days], observations [n_gauges, n_days],
    gage_ids list[str]).
    """
    baseline = run_dir / "baseline"
    manifest_path = baseline / "manifest.json"
    pred_path = baseline / "predictions.f32"
    obs_path = baseline / "observations.f32"

    for p in (manifest_path, pred_path, obs_path):
        if not p.exists():
            print(f"ERROR: expected file not found: {p}", file=sys.stderr)
            sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_gauges = manifest["n_gauges"]
    n_days = manifest["n_days"]
    gage_ids: list[str] = manifest["gage_ids"]

    pred = np.fromfile(pred_path, dtype=np.float32).reshape(n_gauges, n_days)
    obs = np.fromfile(obs_path, dtype=np.float32).reshape(n_gauges, n_days)

    return pred, obs, gage_ids


def load_eval(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load trained-model predictions, observations, and gage IDs from
    eval/predictions.zarr (zarr v3 group written by `ddrs run`).

    Returns (predictions [n_gauges, n_days], observations [n_gauges, n_days],
    gage_ids list[str]).
    """
    try:
        import zarr  # type: ignore
    except ImportError:
        print(
            "ERROR: zarr is not installed. Run: uv pip install zarr",
            file=sys.stderr,
        )
        sys.exit(1)

    zarr_path = run_dir / "eval" / "predictions.zarr"
    if not zarr_path.exists():
        print(
            f"ERROR: eval zarr not found at {zarr_path}. "
            "Has the eval phase completed?",
            file=sys.stderr,
        )
        sys.exit(1)

    store = zarr.open_group(str(zarr_path), mode="r")

    pred = np.array(store["predictions"])     # [n_gauges, n_days]
    obs = np.array(store["observations"])     # [n_gauges, n_days]
    raw_ids = np.array(store["gage_ids"])     # [n_gauges, 8] uint8

    # Decode fixed-width uint8 rows to strings (strip null bytes).
    gage_ids: list[str] = [
        bytes(row).rstrip(b"\x00").decode("ascii") for row in raw_ids
    ]

    return pred.astype(np.float64), obs.astype(np.float64), gage_ids


def nse(pred: np.ndarray, obs: np.ndarray, eps: float = 0.0) -> float:
    """Nash-Sutcliffe Efficiency on 1-D arrays (NaN-safe)."""
    mask = np.isfinite(pred) & np.isfinite(obs)
    if mask.sum() < 2:
        return float("nan")
    p, o = pred[mask], obs[mask]
    denom = np.var(o) + eps
    if denom == 0:
        return float("nan")
    return float(1.0 - np.mean((p - o) ** 2) / denom)


def kge_components(
    pred: np.ndarray, obs: np.ndarray, eps: float = 1e-6
) -> tuple[float, float, float, float]:
    """Return (KGE, r, alpha, beta) on 1-D arrays (NaN-safe)."""
    mask = np.isfinite(pred) & np.isfinite(obs)
    if mask.sum() < 3:
        return (float("nan"),) * 4
    p, o = pred[mask], obs[mask]
    mu_o, mu_p = o.mean(), p.mean()
    std_o, std_p = o.std(ddof=0), p.std(ddof=0)
    if mu_o == 0 or std_o == 0:
        return (float("nan"),) * 4
    r = float(np.corrcoef(p, o)[0, 1])
    alpha = float(std_p / (std_o + eps))
    beta = float(mu_p / (mu_o + eps))
    kge_val = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return kge_val, r, alpha, beta


def kge_beta(pred: np.ndarray, obs: np.ndarray, eps: float = 1e-6) -> float:
    """KGE-β (mean ratio) component only."""
    _, _, _, beta = kge_components(pred, obs, eps=eps)
    return beta


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------


def resolve_run_dir(run_id_or_path: str, runs_dir: Path) -> Path:
    """Accept either a bare run ID or an absolute/relative path."""
    p = Path(run_id_or_path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    candidate = runs_dir / run_id_or_path
    if candidate.exists():
        return candidate
    print(
        f"ERROR: run not found. Tried:\n  {p}\n  {candidate}",
        file=sys.stderr,
    )
    sys.exit(1)


class RunData:
    """All data for one arm of the 2×2."""

    def __init__(self, label: str, run_dir: Path) -> None:
        self.label = label
        self.run_dir = run_dir
        print(f"[{label}] loading from {run_dir}")

        self.eval_pred, self.eval_obs, self.eval_gage_ids = load_eval(run_dir)
        self.base_pred, self.base_obs, self.base_gage_ids = load_baseline(run_dir)

        print(
            f"  eval  : {self.eval_pred.shape}  gages={len(self.eval_gage_ids)}"
        )
        print(
            f"  baseline: {self.base_pred.shape}  gages={len(self.base_gage_ids)}"
        )

    # ------------------------------------------------------------------ #
    # Per-gauge metrics from the TRAINED model                            #
    # ------------------------------------------------------------------ #

    def per_gauge_nse(self) -> tuple[np.ndarray, list[str]]:
        """(NSE array [n_gauges], gage_ids) from eval predictions."""
        vals = np.array(
            [nse(self.eval_pred[i], self.eval_obs[i]) for i in range(len(self.eval_gage_ids))]
        )
        return vals, self.eval_gage_ids

    def per_gauge_kge(self) -> tuple[np.ndarray, list[str]]:
        vals = np.array(
            [kge_components(self.eval_pred[i], self.eval_obs[i])[0]
             for i in range(len(self.eval_gage_ids))]
        )
        return vals, self.eval_gage_ids

    def per_gauge_kge_beta(self) -> tuple[np.ndarray, list[str]]:
        vals = np.array(
            [kge_beta(self.eval_pred[i], self.eval_obs[i])
             for i in range(len(self.eval_gage_ids))]
        )
        return vals, self.eval_gage_ids

    # ------------------------------------------------------------------ #
    # Losing-stream subset mask (on the BASELINE arm)                    #
    # ------------------------------------------------------------------ #

    def losing_stream_mask(self) -> tuple[np.ndarray, list[str]]:
        """Boolean mask over eval gage IDs where summed-Q′ baseline
        mean(pred)/mean(obs) > 1 (over-predicting ⇒ losing stream).

        Uses base_gage_ids ∩ eval_gage_ids intersection so both arrays align.
        """
        base_id_to_idx = {g: i for i, g in enumerate(self.base_gage_ids)}
        mask = np.zeros(len(self.eval_gage_ids), dtype=bool)
        for j, gid in enumerate(self.eval_gage_ids):
            bi = base_id_to_idx.get(gid)
            if bi is None:
                continue  # not in baseline → skip (neither losing nor gaining)
            mu_p = np.nanmean(self.base_pred[bi])
            mu_o = np.nanmean(self.base_obs[bi])
            if mu_o > 0 and mu_p / mu_o > 1.0:
                mask[j] = True
        return mask, self.eval_gage_ids


# ---------------------------------------------------------------------------
# Zeta diagnostics (requires kan_parameters.nc)
# ---------------------------------------------------------------------------


def maybe_load_zeta(run_dir: Path) -> Optional[np.ndarray]:
    """Try to load per-reach net |zeta| from kan_parameters.nc.

    Returns absolute mean zeta array [n_reaches] or None if not found.
    """
    nc_path = run_dir / "kan_parameters.nc"
    if not nc_path.exists():
        return None
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError:
        try:
            import xarray as xr  # type: ignore
            ds = xr.open_dataset(nc_path)
            if "zeta" in ds:
                return np.abs(ds["zeta"].values)
            print(
                "  TODO: kan_parameters.nc found but 'zeta' variable absent. "
                "Run the dump-zeta diagnostic step first.",
                file=sys.stderr,
            )
            return None
        except ImportError:
            print(
                "  WARNING: neither netCDF4 nor xarray installed; "
                "skipping zeta diagnostics.",
                file=sys.stderr,
            )
            return None
    with nc.Dataset(nc_path) as ds:
        if "zeta" not in ds.variables:
            print(
                "  TODO: kan_parameters.nc exists but has no 'zeta' variable. "
                "Run `ddrs run --dump-zeta` (or equivalent) to export per-reach "
                "leakance before running this script.",
                file=sys.stderr,
            )
            return None
        return np.abs(np.array(ds.variables["zeta"][:]))


# ---------------------------------------------------------------------------
# Paired delta computation
# ---------------------------------------------------------------------------


def paired_delta(
    on_vals: np.ndarray,
    on_ids: list[str],
    off_vals: np.ndarray,
    off_ids: list[str],
    mask_ids: list[str],
    mask: np.ndarray,
) -> np.ndarray:
    """Return (on − off) deltas for gauges in `mask_ids[mask]`.

    Aligns on_ids and off_ids by gage ID; unmatchable gauges are NaN.
    """
    off_lookup = dict(zip(off_ids, off_vals))
    on_lookup = dict(zip(on_ids, on_vals))
    deltas = []
    for gid, m in zip(mask_ids, mask):
        if not m:
            continue
        on_v = on_lookup.get(gid, float("nan"))
        off_v = off_lookup.get(gid, float("nan"))
        deltas.append(on_v - off_v)
    return np.array(deltas, dtype=float)


# ---------------------------------------------------------------------------
# K_D bounds floor diagnostic
# ---------------------------------------------------------------------------


def kd_floor_diagnostic(run_dir: Path) -> None:
    """Report if learned K_D values are piling up at the lower bound (1e-8)."""
    nc_path = run_dir / "kan_parameters.nc"
    if not nc_path.exists():
        print(
            "  TODO: kan_parameters.nc not found; run dump_parameters step to "
            "enable K_D floor diagnostic."
        )
        return
    try:
        import xarray as xr  # type: ignore
        ds = xr.open_dataset(nc_path)
        kd_key = next((k for k in ("K_D", "k_d", "kd") if k in ds), None)
        if kd_key is None:
            print(
                "  TODO: kan_parameters.nc exists but has no K_D variable. "
                "Re-run dump_parameters with leakance params enabled."
            )
            return
        kd = ds[kd_key].values.ravel()
        kd = kd[np.isfinite(kd)]
        lb = 1e-8
        floor_frac = np.mean(kd < lb * 2)  # within 2× of lower bound
        print(f"\n  K_D floor diagnostic:")
        print(f"    n_reaches       : {len(kd):,}")
        print(f"    min / median / max: {kd.min():.3e} / {np.median(kd):.3e} / {kd.max():.3e}")
        print(
            f"    fraction < 2×lb (={lb:.0e}): {floor_frac:.1%} "
            f"({'WARNING: floor pile-up' if floor_frac > 0.5 else 'ok'})"
        )
    except ImportError:
        print("  WARNING: xarray not installed; skipping K_D floor diagnostic.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hourly-on",
        required=True,
        metavar="RUN",
        help="Run ID or path for leakance-ON + hourly-disagg arm.",
    )
    parser.add_argument(
        "--daily-on",
        required=True,
        metavar="RUN",
        help="Run ID or path for leakance-ON + daily (flat repeat-24) arm.",
    )
    parser.add_argument(
        "--hourly-off",
        required=True,
        metavar="RUN",
        help="Run ID or path for leakance-OFF + hourly-disagg control "
             "(typically 2026-06-23T02-49-12Z-conus-hourly-train-and-test).",
    )
    parser.add_argument(
        "--daily-off",
        required=True,
        metavar="RUN",
        help="Run ID or path for leakance-OFF + daily control.",
    )
    parser.add_argument(
        "--ddrs-runs-dir",
        default=".ddrs/runs",
        metavar="DIR",
        help="Path to the ddrs runs directory (default: .ddrs/runs).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.ddrs_runs_dir).resolve()
    if not runs_dir.exists():
        print(
            f"ERROR: ddrs runs directory not found: {runs_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load all four arms
    # ------------------------------------------------------------------
    hourly_on = RunData("hourly-ON", resolve_run_dir(args.hourly_on, runs_dir))
    daily_on = RunData("daily-ON", resolve_run_dir(args.daily_on, runs_dir))
    hourly_off = RunData("hourly-OFF", resolve_run_dir(args.hourly_off, runs_dir))
    daily_off = RunData("daily-OFF", resolve_run_dir(args.daily_off, runs_dir))

    # ------------------------------------------------------------------
    # Losing-stream subset — defined on the hourly-OFF baseline
    # (the most direct comparison reference for the hourly arm).
    # ------------------------------------------------------------------
    losing_mask, losing_ids = hourly_off.losing_stream_mask()
    n_losing = int(losing_mask.sum())
    n_total = len(losing_ids)
    print(f"\nLosing-stream subset: {n_losing}/{n_total} gauges "
          f"({100 * n_losing / max(n_total, 1):.1f}%)")
    print("  (defined as baseline mean(pred)/mean(obs) > 1 on the hourly-OFF run)")

    if n_losing == 0:
        print("WARNING: no losing-stream gauges found — subset is empty. "
              "Check that baseline predictions are loaded correctly.")

    # ------------------------------------------------------------------
    # Per-gauge metrics for each arm
    # ------------------------------------------------------------------
    h_on_nse,  h_on_ids  = hourly_on.per_gauge_nse()
    h_off_nse, h_off_ids = hourly_off.per_gauge_nse()
    d_on_nse,  d_on_ids  = daily_on.per_gauge_nse()
    d_off_nse, d_off_ids = daily_off.per_gauge_nse()

    h_on_kge,  _  = hourly_on.per_gauge_kge()
    h_off_kge, _  = hourly_off.per_gauge_kge()
    d_on_kge,  _  = daily_on.per_gauge_kge()
    d_off_kge, _  = daily_off.per_gauge_kge()

    h_on_beta,  _ = hourly_on.per_gauge_kge_beta()
    h_off_beta, _ = hourly_off.per_gauge_kge_beta()
    d_on_beta,  _ = daily_on.per_gauge_kge_beta()
    d_off_beta, _ = daily_off.per_gauge_kge_beta()

    # ------------------------------------------------------------------
    # Paired ON−OFF deltas on the losing-stream subset
    # ------------------------------------------------------------------
    def delta(on_v, on_i, off_v, off_i):
        return paired_delta(on_v, on_i, off_v, off_i, losing_ids, losing_mask)

    h_dnse = delta(h_on_nse, h_on_ids, h_off_nse, h_off_ids)
    h_dkge = delta(h_on_kge, h_on_ids, h_off_kge, h_off_ids)
    h_dbeta = delta(h_on_beta, h_on_ids, h_off_beta, h_off_ids)

    d_dnse = delta(d_on_nse, d_on_ids, d_off_nse, d_off_ids)
    d_dkge = delta(d_on_kge, d_on_ids, d_off_kge, d_off_ids)
    d_dbeta = delta(d_on_beta, d_on_ids, d_off_beta, d_off_ids)

    def nanmedian(a: np.ndarray) -> float:
        v = a[np.isfinite(a)]
        return float(np.median(v)) if len(v) else float("nan")

    def frac_positive(a: np.ndarray) -> float:
        v = a[np.isfinite(a)]
        return float((v > 0).mean()) if len(v) else float("nan")

    print("\n" + "=" * 60)
    print("  PAIRED (ON − OFF) SKILL METRICS — losing-stream subset")
    print("=" * 60)
    header = f"  {'arm':<14} {'ΔNSE med':>9} {'ΔKGE med':>9} {'ΔKGEβ med':>10} {'frac(ΔNSE>0)':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, dn, dk, db in [
        ("hourly ON−OFF", h_dnse, h_dkge, h_dbeta),
        ("daily  ON−OFF", d_dnse, d_dkge, d_dbeta),
    ]:
        print(
            f"  {label:<14} "
            f"{nanmedian(dn):>+9.4f} "
            f"{nanmedian(dk):>+9.4f} "
            f"{nanmedian(db):>+10.4f} "
            f"{frac_positive(dn):>12.1%}"
        )

    # ------------------------------------------------------------------
    # Zeta diagnostics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LEAKANCE MAGNITUDE |zeta| (m³/s)")
    print("=" * 60)
    zeta_thresh = 0.01
    for label, run in [("hourly-ON", hourly_on), ("daily-ON", daily_on)]:
        zeta = maybe_load_zeta(run.run_dir)
        if zeta is None:
            print(
                f"  [{label}] TODO: kan_parameters.nc / 'zeta' variable not found. "
                "Run `ddrs dump-parameters` (or equivalent zeta export) before "
                "interpreting GO/NO-GO."
            )
        else:
            frac_above = float(np.mean(zeta > zeta_thresh))
            print(
                f"  [{label}] |zeta|>0.01 on {frac_above:.1%} of reaches "
                f"(n={len(zeta):,}, "
                f"median={np.median(zeta):.4f} m³/s)"
            )

    # K_D floor diagnostic for hourly-ON (primary GO/NO-GO arm).
    print("\n[hourly-ON K_D floor]")
    kd_floor_diagnostic(hourly_on.run_dir)

    # ------------------------------------------------------------------
    # GO / NO-GO verdict
    # ------------------------------------------------------------------
    # Quantitative thresholds (per spec):
    #   - ΔNSE or ΔKGE > 0 (median) on losing subset, hourly arm
    #   - |zeta| > 0.01 on a "meaningful" fraction (≥ 10% of reaches as proxy)
    #   - daily arm ΔNSE and ΔKGE ≤ 0 or weaker than hourly arm

    h_nse_improves = nanmedian(h_dnse) > 0
    h_kge_improves = nanmedian(h_dkge) > 0
    skill_improves = h_nse_improves or h_kge_improves

    h_zeta = maybe_load_zeta(hourly_on.run_dir)
    if h_zeta is not None:
        zeta_meaningful = float(np.mean(h_zeta > zeta_thresh)) >= 0.10
        zeta_str = f"{float(np.mean(h_zeta > zeta_thresh)):.1%} of reaches"
    else:
        zeta_meaningful = None  # unknown — can't rule in or out
        zeta_str = "UNKNOWN (zeta not exported yet)"

    d_nse_weaker = nanmedian(d_dnse) <= nanmedian(h_dnse)
    d_kge_weaker = nanmedian(d_dkge) <= nanmedian(h_dkge)
    daily_weaker = d_nse_weaker and d_kge_weaker

    print("\n" + "=" * 60)
    print("  GO / NO-GO VERDICT")
    print("=" * 60)
    print(f"  Skill improves on losing subset (NSE or KGE > 0): {'YES' if skill_improves else 'NO'}")
    print(f"    hourly ΔNSE med = {nanmedian(h_dnse):+.4f}   {'✓' if h_nse_improves else '✗'}")
    print(f"    hourly ΔKGE med = {nanmedian(h_dkge):+.4f}   {'✓' if h_kge_improves else '✗'}")
    if zeta_meaningful is not None:
        print(f"  |zeta|>0.01 on meaningful fraction: {'YES' if zeta_meaningful else 'NO'}  ({zeta_str})")
    else:
        print(f"  |zeta|>0.01 diagnostic: {zeta_str}")
    print(f"  Effect weaker in daily arm: {'YES' if daily_weaker else 'NO'}")
    print(f"    daily ΔNSE med = {nanmedian(d_dnse):+.4f}   {'weaker' if d_nse_weaker else 'stronger'}")
    print(f"    daily ΔKGE med = {nanmedian(d_dkge):+.4f}   {'weaker' if d_kge_weaker else 'stronger'}")

    if zeta_meaningful is None:
        verdict = "NEEDS_ZETA_EXPORT"
        rationale = ("Cannot determine GO/NO-GO without zeta export. "
                     "Run `ddrs dump-parameters` and re-run this script.")
    elif skill_improves and zeta_meaningful and daily_weaker:
        verdict = "GO"
        rationale = ("Leakance improves skill on the losing-stream subset under "
                     "hourly forcing, the learned |zeta| is non-trivial, and the "
                     "effect is absent or weaker under daily forcing — consistent "
                     "with genuine GW–SW identifiability from sub-daily signal.")
    else:
        reasons = []
        if not skill_improves:
            reasons.append("no skill gain on losing-stream subset")
        if not zeta_meaningful:
            reasons.append(f"|zeta|>0.01 on too few reaches ({zeta_str})")
        if not daily_weaker:
            reasons.append("effect equally strong in daily arm (possible fudge factor)")
        verdict = "NO-GO"
        rationale = "Leakance is NOT identifiable: " + "; ".join(reasons) + "."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {rationale}")
    print()


if __name__ == "__main__":
    main()
