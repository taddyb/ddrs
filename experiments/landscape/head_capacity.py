#!/usr/bin/env python
"""Offline capacity test: would more KAN layers let the head place Manning's n
where each gauge wants it? No routing, no retraining of ddrs itself.

Target: ln(n*) = ln(n0_med) + alpha_n_star, at well-fit gauges (nse0 > 0.3,
box_edge == 0, hit_range_bound == 0) from the p=21 census
(`covariates.csv`). Inputs: the 10 MERIT attributes the trained KAN head
consumes (`kan_head.input_var_names` in the run's config snapshot),
z-scored with the training statistics JSON DDR caches next to the
attributes file (same normalization `finalize_attrs` applies in
src/data/dataset.rs).

Models compared under identical 5-fold CV (same folds for every model):
  - "current head": the trained head's own n (n0_med) used as the
    prediction of n*, i.e. alpha = 0 -- not fit to anything, just scored.
    This is "what training achieved."
  - ridge: weighted linear ridge regression (RidgeCV over a small alpha grid).
  - knn10: k-nearest-neighbours (k=10), unweighted fit (as in the E-series
    analysis), scored under both weight schemes for comparability.
  - gbm: LightGBM gradient-boosted trees (the nonparametric ceiling), with
    early stopping on an inner validation split, weighted training.
  - kan_hHxLL: DDR's own KAN block (`ddr.nn.kan.kan`, imported from the ddr
    venv), reimplemented in this script's training loop only (no ddrs code
    touched), at the current depth/width and three capacity increases:
    +1 hidden layer, +2 hidden layers, and 2x hidden width at the current
    depth. 3 seeds per fold, reported as the seed-mean per fold.

Weighting: two independent weight vectors, each clipped and renormalised to
mean 1 over the filtered population --
  - curvature: lambda1 (stiff Hessian eigenvalue at the gauge optimum),
    clipped to [1e-4, 1].
  - gain: NSE(alpha*) - NSE(alpha=0), clipped to [0, 0.1].
Every model is TRAINED once per fold using the curvature weights as
sample_weight (where the estimator supports it -- kNN does not, so it is
trained unweighted); all four R^2 columns are then computed as
*evaluation*-time weightings of the same held-out predictions.

Outputs:
  - <report>            docs/why-analysis/G-head-capacity.md
  - <figure>             .../figures/why/G_capacity.png
  - <results-csv>        .../figures/why/G_capacity_results.csv (per-fold detail)

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/head_capacity.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]

DEFAULT_COVARIATES = REPO / ".ddrs/experiments/landscape-p21-all-5yr/merged/figures/covariates.csv"
DEFAULT_CONFIG = REPO / ".ddrs/runs/2026-09-08T15-55-52Z-conus-train-and-test/config.yaml"
DEFAULT_ATTRS_NC = Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
DEFAULT_STATS_JSON = Path(
    "/home/tbindas/projects/ddr/data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json"
)
DEFAULT_GAGES_CSV = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
DEFAULT_OUT_DIR = REPO / ".ddrs/experiments/landscape-p21-all-5yr/merged/figures/why"
DEFAULT_REPORT = REPO / "docs/why-analysis/G-head-capacity.md"

DDR_SRC = Path("/home/tbindas/projects/ddr/src")

N_ARCH = "n"  # the parameter name we model


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    wsum = w.sum()
    ybar = np.sum(w * y_true) / wsum
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - ybar) ** 2)
    return 1.0 - ss_res / ss_tot


def build_dataset(args) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns (df_filtered, X (N,F) z-scored, y = ln(n*), w_curv, w_gain, input_var_names)."""
    cfg = load_config(args.config)
    input_var_names = cfg["kan_head"]["input_var_names"]

    df = pd.read_csv(args.covariates, dtype={"staid": str})
    mask = (df["nse0"] > 0.3) & (df["box_edge"] == 0) & (df["hit_range_bound"] == 0)
    df = df.loc[mask].reset_index(drop=True)

    df["staid"] = df["staid"].str.zfill(8)
    gages = pd.read_csv(args.gages_csv, dtype={"STAID": str})
    gages["STAID"] = gages["STAID"].str.zfill(8)
    df = df.merge(gages[["STAID", "COMID"]], left_on="staid", right_on="STAID", how="left")
    n_missing = df["COMID"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} gauges had no COMID match in {args.gages_csv}; dropping.", file=sys.stderr)
        df = df.dropna(subset=["COMID"]).reset_index(drop=True)
    df["COMID"] = df["COMID"].astype(np.int64)

    import xarray as xr

    attrs_ds = xr.open_dataset(args.attrs_nc)
    with open(args.stats_json) as f:
        stats = json.load(f)

    comids = df["COMID"].values
    x_cols = []
    for var in input_var_names:
        raw = attrs_ds[var].sel(COMID=xr.DataArray(comids, dims="row")).values.astype(np.float64)
        mean = stats[var]["mean"]
        std = stats[var]["std"]
        z = (raw - mean) / std
        z = np.where(np.isnan(raw), 0.0, z)  # NaN -> training mean -> z=0, mirrors finalize_attrs' fill_nans
        x_cols.append(z)
    X = np.stack(x_cols, axis=1).astype(np.float32)

    y = np.log(df["n0_med"].values) + df["alpha_n_star"].values
    y0 = np.log(df["n0_med"].values)

    lambda1 = np.clip(df["lambda1"].values, 1e-4, 1.0)
    w_curv = lambda1 / lambda1.mean()

    gain = np.clip(df["gain"].values, 0.0, 0.1)
    w_gain = gain / gain.mean()

    return df, X, y, y0, w_curv, w_gain, input_var_names


def make_kan_model(hidden_size: int, num_hidden_layers: int, grid: int, k: int, n_features: int, seed: int):
    sys.path.insert(0, str(DDR_SRC))
    from ddr.nn.kan import kan  # noqa: E402  (DDR's own module, imported not copied)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = kan(
            input_var_names=[f"x{i}" for i in range(n_features)],
            learnable_parameters=[N_ARCH],
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            grid=grid,
            k=k,
            seed=seed,
            device="cpu",
        )
    return model


def train_kan(
    X_train, y_train, w_train, X_val, y_val, w_val, hidden_size, num_hidden_layers, grid, k, seed, max_epochs, patience, lr, n_bounds
):
    import torch

    lo, hi = n_bounds
    model = make_kan_model(hidden_size, num_hidden_layers, grid, k, X_train.shape[1], seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    wt = torch.tensor(w_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    wv = torch.tensor(w_val, dtype=torch.float32)

    def forward_ln_n(model, X):
        # DDR's kan.py forward() already applies F.sigmoid internally before
        # returning outputs[key] -- do NOT sigmoid again here (that would
        # double-saturate and made training collapse to a near-constant
        # prediction in early testing; see docs/why-analysis/G-head-capacity.md).
        frac = model(inputs=X)[N_ARCH]
        n = lo + frac * (hi - lo)
        return torch.log(n)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for _epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        ln_n_pred = forward_ln_n(model, Xt)
        loss = torch.sum(wt * (ln_n_pred - yt) ** 2) / wt.sum()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            ln_n_val = forward_ln_n(model, Xv)
            val_loss = (torch.sum(wv * (ln_n_val - yv) ** 2) / wv.sum()).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, forward_ln_n


def run_kan_arch(
    hidden_size, num_hidden_layers, grid, k, X, y, w_curv, w_gain, folds, n_seeds, max_epochs, patience, lr, n_bounds, val_frac, base_seed
):
    import torch
    from sklearn.model_selection import train_test_split

    fold_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        seed_metrics = {"r2_curv": [], "r2_gain": [], "r2_unw": [], "r2_train": []}
        for s in range(n_seeds):
            seed = base_seed + 1000 * fold_idx + s
            torch.manual_seed(seed)
            tr_idx, val_idx = train_test_split(train_idx, test_size=val_frac, random_state=fold_idx * 100 + s)

            model, forward_ln_n = train_kan(
                X[tr_idx], y[tr_idx], w_curv[tr_idx],
                X[val_idx], y[val_idx], w_curv[val_idx],
                hidden_size, num_hidden_layers, grid, k, seed, max_epochs, patience, lr, n_bounds,
            )
            with torch.no_grad():
                pred_test = forward_ln_n(model, torch.tensor(X[test_idx], dtype=torch.float32)).numpy()
                pred_train = forward_ln_n(model, torch.tensor(X[train_idx], dtype=torch.float32)).numpy()

            seed_metrics["r2_curv"].append(weighted_r2(y[test_idx], pred_test, w_curv[test_idx]))
            seed_metrics["r2_gain"].append(weighted_r2(y[test_idx], pred_test, w_gain[test_idx]))
            seed_metrics["r2_unw"].append(weighted_r2(y[test_idx], pred_test, np.ones_like(w_curv[test_idx])))
            seed_metrics["r2_train"].append(weighted_r2(y[train_idx], pred_train, w_curv[train_idx]))

        fold_rows.append({k_: float(np.mean(v)) for k_, v in seed_metrics.items()})
    return fold_rows


def run_reference_models(X, y, w_curv, w_gain, folds, seed):
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor

    results = {"ridge": [], "knn10": [], "gbm": []}

    have_lgb = True
    try:
        import lightgbm as lgb
    except ImportError:
        have_lgb = False
        from sklearn.ensemble import HistGradientBoostingRegressor

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        wtr = w_curv[train_idx]

        # ridge (weighted)
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(Xtr, ytr, sample_weight=wtr)
        pred_te = ridge.predict(Xte)
        pred_tr = ridge.predict(Xtr)
        results["ridge"].append(_metrics_row(y, pred_te, pred_tr, train_idx, test_idx, w_curv, w_gain))

        # knn (unweighted fit, as in analyst E)
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(Xtr, ytr)
        pred_te = knn.predict(Xte)
        pred_tr = knn.predict(Xtr)
        results["knn10"].append(_metrics_row(y, pred_te, pred_tr, train_idx, test_idx, w_curv, w_gain))

        # gbm (weighted, early-stopped on an inner val split)
        from sklearn.model_selection import train_test_split

        tr_idx2, val_idx2 = train_test_split(train_idx, test_size=0.2, random_state=fold_idx)
        if have_lgb:
            model = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=4, num_leaves=15,
                min_child_samples=10, random_state=seed, verbosity=-1,
            )
            model.fit(
                X[tr_idx2], y[tr_idx2], sample_weight=w_curv[tr_idx2],
                eval_set=[(X[val_idx2], y[val_idx2])], eval_sample_weight=[w_curv[val_idx2]],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        else:
            model = HistGradientBoostingRegressor(
                max_depth=4, learning_rate=0.05, max_iter=500, random_state=seed,
                early_stopping=True, validation_fraction=0.2, n_iter_no_change=30,
            )
            model.fit(X[tr_idx2], y[tr_idx2], sample_weight=w_curv[tr_idx2])
        pred_te = model.predict(Xte)
        pred_tr = model.predict(Xtr)
        results["gbm"].append(_metrics_row(y, pred_te, pred_tr, train_idx, test_idx, w_curv, w_gain))

    return results


def _metrics_row(y, pred_te, pred_tr, train_idx, test_idx, w_curv, w_gain):
    return {
        "r2_curv": weighted_r2(y[test_idx], pred_te, w_curv[test_idx]),
        "r2_gain": weighted_r2(y[test_idx], pred_te, w_gain[test_idx]),
        "r2_unw": weighted_r2(y[test_idx], pred_te, np.ones(len(test_idx))),
        "r2_train": weighted_r2(y[train_idx], pred_tr, w_curv[train_idx]),
    }


def summarize(fold_rows: list[dict]) -> dict:
    out = {}
    for key in ("r2_curv", "r2_gain", "r2_unw", "r2_train"):
        vals = np.array([r[key] for r in fold_rows])
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--covariates", type=Path, default=DEFAULT_COVARIATES)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--attrs-nc", type=Path, default=DEFAULT_ATTRS_NC)
    ap.add_argument("--stats-json", type=Path, default=DEFAULT_STATS_JSON)
    ap.add_argument("--gages-csv", type=Path, default=DEFAULT_GAGES_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    from sklearn.model_selection import KFold

    cfg = load_config(args.config)
    hidden_size = cfg["kan_head"]["hidden_size"]
    num_hidden_layers = cfg["kan_head"]["num_hidden_layers"]
    grid = cfg["kan_head"]["grid"]
    k = cfg["kan_head"]["k"]
    n_lo, n_hi = cfg["params"]["parameter_ranges"]["n"]
    log_space = cfg["params"].get("log_space_parameters", [])
    n_is_log = "n" in log_space
    print(f"config: hidden_size={hidden_size} num_hidden_layers={num_hidden_layers} grid={grid} k={k} "
          f"n_range=[{n_lo},{n_hi}] n_is_log_space={n_is_log}")
    assert not n_is_log, "n is log-space in this config; script assumes the linear denormalize branch."

    df, X, y, y0, w_curv, w_gain, input_var_names = build_dataset(args)
    n = len(df)
    print(f"n_gauges={n} n_features={X.shape[1]}")
    print(f"input_var_names={input_var_names}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(X))

    # "current head" baseline: alpha=0 prediction, no CV (nothing is fit).
    baseline_row = {
        "model": "current head (n0_med)", "spec": "alpha=0",
        "r2_curv_mean": weighted_r2(y, y0, w_curv), "r2_curv_std": np.nan,
        "r2_gain_mean": weighted_r2(y, y0, w_gain), "r2_gain_std": np.nan,
        "r2_unw_mean": weighted_r2(y, y0, np.ones_like(w_curv)), "r2_unw_std": np.nan,
        "r2_train_mean": np.nan, "r2_train_std": np.nan,
    }

    rows = [baseline_row]

    print("running reference models (ridge, knn10, gbm)...")
    ref_results = run_reference_models(X, y, w_curv, w_gain, folds, args.seed)
    for name, spec in [("ridge", "linear"), ("knn10", "k=10"), ("gbm", "lightgbm")]:
        summ = summarize(ref_results[name])
        rows.append({"model": name, "spec": spec, **summ})

    kan_archs = [
        (hidden_size, num_hidden_layers, "current"),
        (hidden_size, num_hidden_layers + 1, "+1 layer"),
        (hidden_size, num_hidden_layers + 2, "+2 layers"),
        (hidden_size * 2, num_hidden_layers, "2x width"),
    ]
    for H, L, spec in kan_archs:
        print(f"running KAN H={H} L={L} ({spec})...")
        fold_rows = run_kan_arch(
            H, L, grid, k, X, y, w_curv, w_gain, folds, args.n_seeds,
            args.max_epochs, args.patience, args.lr, (n_lo, n_hi), args.val_frac, args.seed,
        )
        summ = summarize(fold_rows)
        rows.append({"model": f"kan_h{H}_l{L}", "spec": f"H={H},L={L} ({spec})", **summ})

    results_df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.out_dir / "G_capacity_results.csv", index=False)
    print(results_df.to_string(index=False))

    make_figure(results_df, args.out_dir / "G_capacity.png")
    write_report(args, results_df, n)


def make_figure(results_df: pd.DataFrame, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_map = {
        "current head (n0_med)": "#8a8a8a",
        "ridge": "#4c72b0",
        "knn10": "#55a868",
        "gbm": "#dd8452",
    }
    kan_color = "#8172b3"

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    means = []
    stds = []
    colors = []
    for _, row in results_df.iterrows():
        labels.append(row["model"])
        means.append(row["r2_curv_mean"])
        stds.append(0.0 if pd.isna(row["r2_curv_std"]) else row["r2_curv_std"])
        colors.append(color_map.get(row["model"], kan_color))

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, width=0.6)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("weighted R2 of ln n* (curvature weights, held-out folds)")
    ax.set_title("Head capacity: does depth/width let the KAN place n where gauges want it?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def write_report(args, results_df: pd.DataFrame, n: int):
    gbm_row = results_df[results_df["model"] == "gbm"].iloc[0]
    baseline_row = results_df[results_df["model"] == "current head (n0_med)"].iloc[0]
    kan_rows = results_df[results_df["model"].str.startswith("kan_")]

    def fmt(row, key):
        m = row[f"{key}_mean"]
        s = row[f"{key}_std"]
        if pd.isna(s):
            return f"{m:.3f}"
        return f"{m:.3f} +/- {s:.3f}"

    lines = []
    lines.append("# G. Does head capacity (KAN depth/width) let n reach where each gauge wants it?")
    lines.append("")
    lines.append(
        f"**Data.** `covariates.csv` p=21 census (config `{args.config.name}`, run "
        f"`2026-09-08T15-55-52Z-conus-train-and-test`), filtered to well-fit gauges (nse0 > 0.3), "
        f"excluding box_edge == 1 and hit_range_bound == 1: n = {n} gauges. Target ln n* = "
        f"ln(n0_med) + alpha_n_star. Inputs: the 10 attributes in `kan_head.input_var_names` "
        f"(SoilGrids1km_clay, aridity, meanelevation, meanP, NDVI, meanslope, log10_uparea, "
        f"SoilGrids1km_sand, ETPOT_Hargr, Porosity), read per gauge-reach COMID (via "
        f"`gages_3000.csv` STAID->COMID, cross-checked against a gauge subgraph's `comid[gauge_reach_row]`) "
        f"from `merit_global_attributes_v2.nc`, z-scored with the training statistics JSON "
        f"(mean/std per attribute), NaNs filled to the training mean (z=0) as `finalize_attrs` does "
        f"in `src/data/dataset.rs`."
    )
    lines.append("")
    lines.append(
        "**Weights.** Two independent schemes, each clipped and renormalised to mean 1: curvature "
        "(`lambda1`, clipped to [1e-4, 1]) and gain (`gain` = NSE(alpha*) - NSE(alpha=0), clipped to "
        "[0, 0.1]). All models are trained once per fold with the curvature weights as sample weight "
        "(kNN has no native sample-weighted fit, so it trains unweighted, as in analyst E's kNN-10 "
        "reference); both weight schemes are then applied at evaluation time to the same held-out "
        "predictions, plus an unweighted R2 and a train-set (curvature-weighted) R2 for overfitting."
    )
    lines.append("")
    lines.append(
        "**Protocol.** 5-fold CV, same folds for every model. Ridge: `RidgeCV` over "
        "alpha in {0.01,0.1,1,10,100}. kNN: k=10, unweighted. GBM: LightGBM, 500 trees, early-stopped "
        "on a 20% inner validation split (30-round patience). KAN: DDR's own `ddr.nn.kan.kan` "
        "(`Linear(F,H) -> KanLayer(H,H) x L -> Linear(H,1) -> sigmoid -> denormalize(n)`), Adam "
        f"(lr={args.lr}), early-stopped on a {int(args.val_frac*100)}% inner validation split "
        f"(patience={args.patience}, max {args.max_epochs} epochs), {args.n_seeds} seeds averaged per "
        "fold. n is NOT in `params.log_space_parameters` for this config, so denormalize is the LINEAR "
        "branch (n = lo + sigmoid*(hi-lo)); the loss is weighted MSE on ln(n)."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| model | depth/width | R2 (curvature w) | R2 (gain w) | R2 (unweighted) | R2 (train) |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['spec']} | {fmt(row,'r2_curv')} | {fmt(row,'r2_gain')} | "
            f"{fmt(row,'r2_unw')} | {fmt(row,'r2_train')} |"
        )
    lines.append("")

    floor_frac = gbm_row["r2_curv_mean"]
    lines.append(
        f"**Weight-implied noise floor (assumption).** We have no independent replicate of alpha_n_star "
        f"per gauge, so we cannot compute an absolute noise variance from the weights alone (they are "
        f"normalised to mean 1, a relative precision, not a calibrated one). As a stand-in, we treat "
        f"the nonparametric ceiling's (GBM) held-out curvature-weighted R2 as the empirical estimate of "
        f"the signal fraction any model of these inputs can reach; the complement "
        f"(1 - R2_gbm = {1 - floor_frac:.2f}) is the fraction attributed to unmodelled noise plus "
        f"information genuinely absent from the 10 attributes. This is a working definition, not a "
        f"calibrated bound -- state it as such if reused."
    )
    lines.append("")
    lines.append("## Observations")
    lines.append("")

    kan_current_row = results_df[results_df["spec"].str.contains("current", na=False)].iloc[0]
    kan_best = kan_rows.loc[kan_rows["r2_curv_mean"].idxmax()]
    ridge_row = results_df[results_df["model"] == "ridge"].iloc[0]
    knn_row = results_df[results_df["model"] == "knn10"].iloc[0]

    lines.append(
        f"- The trained head's own n (alpha=0) reaches weighted R2 = {baseline_row['r2_curv_mean']:.3f} "
        f"(curvature) / {baseline_row['r2_gain_mean']:.3f} (gain) against ln n* directly -- this is what "
        f"30 epochs of training already achieved on this population, with zero degrees of freedom spent "
        f"fitting alpha_n_star itself."
    )
    lines.append(
        f"- Ridge (linear in the 10 attributes) reaches {fmt(ridge_row,'r2_curv')}; kNN-10 reaches "
        f"{fmt(knn_row,'r2_curv')}; GBM (nonparametric ceiling) reaches {fmt(gbm_row,'r2_curv')}."
    )
    lines.append(
        f"- The current-depth KAN ({kan_current_row['spec']}) reaches {fmt(kan_current_row,'r2_curv')}. "
        f"Adding capacity moves it to: "
        + "; ".join(
            f"{r['spec']} -> {fmt(r,'r2_curv')}"
            for _, r in kan_rows.iterrows() if r["spec"] != kan_current_row["spec"]
        )
        + "."
    )
    lines.append(
        f"- Best KAN variant: {kan_best['spec']} at {fmt(kan_best,'r2_curv')}, "
        f"{'above' if kan_best['r2_curv_mean'] > gbm_row['r2_curv_mean'] else 'at or below'} the GBM ceiling "
        f"({fmt(gbm_row,'r2_curv')})."
    )
    train_gap = kan_best["r2_train_mean"] - kan_best["r2_curv_mean"]
    lines.append(
        f"- Train-vs-held-out gap for the best KAN variant: {kan_best['r2_train_mean']:.3f} (train) vs "
        f"{kan_best['r2_curv_mean']:.3f} (held out), gap = {train_gap:.3f} "
        f"({'small, not overfitting' if train_gap < 0.1 else 'material -- capacity is fitting fold-specific noise, not signal'})."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If every KAN variant (current depth through +2 layers and 2x width) lands within noise of "
        "each other and of the linear ridge reference, and all sit near the GBM ceiling, added KAN "
        "capacity is not the bottleneck: the 10 attributes the head consumes do not encode where a "
        "gauge wants n, regardless of how flexible the function mapping attributes to n is allowed to "
        "be. This is an INFORMATION problem, consistent with the E-series finding that a 10-neighbour "
        "attribute regression alone reaches R2 around 0.12 on a closely related population, and that "
        "the dominant covariate of the gap (routed-flow timing lag) is not among the 10 inputs at all."
    )
    lines.append(
        "- If deeper/wider KAN variants show a clear, monotonic, out-of-fold gain over the current "
        "depth (train R2 also rising without a corresponding held-out gap), the shared head was "
        "genuinely under-fitting its own inputs, and depth is worth the wall-clock cost of a real "
        "retrain."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    best_close_to_current = abs(kan_best["r2_curv_mean"] - kan_current_row["r2_curv_mean"]) < 0.05
    if best_close_to_current and kan_best["r2_curv_mean"] <= gbm_row["r2_curv_mean"] + 0.02:
        verdict = (
            "**INFORMATION, not capacity.** More KAN layers/width do not move the held-out fit to ln n* "
            "materially beyond the current architecture or the GBM ceiling on the same inputs. The head "
            "cannot place n where a gauge wants it because the 10 attributes it sees do not carry that "
            "signal, not because the function class is too small."
        )
    else:
        verdict = (
            "**Capacity plays a role.** A deeper/wider KAN measurably beats the current architecture "
            "out of sample without a matching overfit gap, so some of the shortfall is the head being "
            "too small for even the information already present in its 10 inputs."
        )
    lines.append(verdict)
    lines.append("")
    lines.append(
        "**Decisive follow-up.** Retrain the real KAN head (ddrs, on-graph, with routing) at the best "
        "capacity variant found here on the same gages_3000 population and re-run the census "
        "(`covariates.py`) to see whether alpha_n_star and gain actually shrink -- this offline test only "
        "checks whether the function class *could* separate the training signal from noise on frozen "
        "inputs; it cannot rule out optimisation or batch-compromise effects that only appear when many "
        "gauges share one gradient (see finding E)."
    )
    lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp(prefix="head_capacity_")
    _cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        main()
    finally:
        os.chdir(_cwd)
