# DDR ↔ DDRS trained-`n` saturation parity — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether DDR's trained Manning's `n` distribution matches
DDRS's saturation pattern (median ≈ 0.030, 47 % in [0.020, 0.030]). If yes,
the port is faithful and saturation is a model+data property. If no, localize
the divergence in `src/training/`.

**Architecture:** Three artefact families.
1. **Training-config audit** — read DDR's `scripts/train.py` and DDRS's
   `src/training/{driver,optimizer,loss}.rs` line by line, fill in a table
   in the spec, surface any ✗.
2. **DDR-side trained dump** — new Python script
   `scripts/dump_ddr_trained_params.py` (mirroring
   `scripts/dump_ddr_init_params.py` from the previous plan) that loads a
   trained DDR checkpoint and writes a CONUS `n / q_spatial / p_spatial`
   NetCDF. Reuses DDR's `_predict_kan_params` helper from
   `geometry_predictor.py`.
3. **Comparison notebook + skill reference** — `parity_trained.md` skill
   reference + executed `parity_trained.ipynb` producing KS-test +
   Spearman + per-gauge scatter + verdict.

**Tech Stack:** No new Rust code expected. Python script under DDR's `uv`
venv at `~/projects/ddr/.venv/`. Jupyter notebook via the same venv. xarray,
matplotlib, scipy.stats, pandas (already present).

**Spec source of truth:**
`docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md`.

---

## Pre-flight verified

- Branch: `trained-parity` is rebased on `master` (PR #11 already merged).
  Current head: `218d9a3` (spec commit) on top of `8236cc4` (the merge).
- DDR loss: `torch.nn.functional.l1_loss` at
  `~/projects/ddr/scripts/train.py:94`.
- DDRS loss: `(p_post - o_post).abs().mean()` at
  `src/training/driver.rs:123`. **Already matches.**
- DDR optim: `torch.optim.Adam(params=nn.parameters(), lr=lr)` at
  `~/projects/ddr/scripts/train.py:40` — default `betas=(0.9, 0.999)`,
  `eps=1e-8`, no weight decay.
- DDRS optim: `AdamConfig::new().with_beta_1(0.9).with_beta_2(0.999)
  .with_epsilon(1e-8)` at `src/training/optimizer.rs:41-44`. **Already
  matches.**
- DDR predictor: `~/projects/ddr/scripts/geometry_predictor.py`
  `_predict_kan_params()` returns `(all_comids, n_vals, p_vals, q_vals)`
  for the full global attribute set. We will reuse this.
- DDRS reference NetCDF: existing run
  `.ddrs/runs/2026-06-03T09-45-09Z-train-and-test/kan_parameters.nc`
  (already contains the saturated `n`).

---

## File structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md` | modify | Layer 0 audit table filled in; Layer 0.5 finding appended; §5 verdict appended. |
| `scripts/dump_ddr_trained_params.py` | create | Python helper. Run under DDR's venv. Loads DDR checkpoint, calls `_predict_kan_params`, subsets to DDRS's CONUS COMID order, writes `/tmp/kan_params_trained_ddr.nc`. |
| `.claude/skills/ddrs-eval-plots/references/parity_trained.md` | create | Skill reference with the 5-cell notebook recipe (load → stats → histograms → per-gauge scatter → verdict). |
| `.ddrs/runs/2026-06-03T09-45-09Z-train-and-test/plots/parity_trained.ipynb` | create (transient — gitignored) | Materialized notebook from the skill ref. Executed via `cd ddrs-py && uv run jupyter nbconvert --execute`. |
| `.ddrs/runs/2026-06-03T09-45-09Z-train-and-test/plots/parity_trained_*.png` | create (transient — gitignored) | Plots saved by the notebook. |

`/tmp/kan_params_trained_ddr.nc` is a transient artefact. Do NOT commit it.

---

## Task 1: Layer 0 — training-config audit

**Spec ref:** §4 Layer 0.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md` (the §4 Layer 0 table)

- [ ] **Step 1: Read the DDR-side training entry point**

```bash
sed -n '20,110p' ~/projects/ddr/scripts/train.py
```

Walk every line. Note:
- Optimizer construction: `torch.optim.Adam(params=nn.parameters(), lr=lr)` at line 40 (verified during pre-flight).
- LR schedule: `experiment.learning_rate` mapping driving `for param_group in optimizer.param_groups` at line 57.
- Loss: `torch.nn.functional.l1_loss(...)` at line 94 (verified).
- Gradient clip: `torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=...)` (search for it).
- Batch construction: any `RandomSampler` / `DataLoader` / shuffle invocation.
- Warmup application: how DDR drops the first `warmup` days from the loss.

- [ ] **Step 2: Read the DDRS-side training driver**

```bash
sed -n '1,180p' /home/tbindas/projects/ddrs/src/training/driver.rs
sed -n '1,80p'  /home/tbindas/projects/ddrs/src/training/optimizer.rs
sed -n '1,120p' /home/tbindas/projects/ddrs/src/training/loss.rs
```

Note the same fields on the DDRS side.

- [ ] **Step 3: Fill in §4 Layer 0 table inline**

Open the spec at
`docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md`.
For each row in the table at §4 Layer 0, replace `(audit)` with one of:

- `✓` if the value matches (DDR source line + DDRS source line + identical
  semantic).
- `STAT only` if formulas match but RNG bytes differ (per spec C5 of the
  previous parity plan — batch shuffle is the main case).
- `✗ (actual: X, want: Y)` if there is a real mismatch.

For pre-flight-verified rows (Loss, Optimizer, Adam betas/eps/weight_decay),
just confirm ✓ from pre-flight; you don't have to re-walk them.

- [ ] **Step 4: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md
git commit -m "$(cat <<'EOF'
docs/spec: complete Layer 0 training-config audit

Replaces (audit) cells in §4 Layer 0 with ✓ / STAT-only / ✗ verdicts
based on a line-by-line read of ~/projects/ddr/scripts/train.py and
src/training/{driver,optimizer,loss}.rs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: If any ✗ surfaces**, STOP and bring it to the user. Any
  unexpected training-loop divergence is more likely the saturation cause
  than anything else in this plan.

---

## Task 2: Layer 0.5 — inspect existing DDR training run

**Spec ref:** §4 Layer 0.5.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md` (append finding under §4 Layer 0.5)

- [ ] **Step 1: Locate the most recent existing DDR training output**

```bash
ls -t ~/projects/ddr/output/ddr-v0.5.2.dev2+g21a3a96b5-merit-training/ | head -1
```

Expected: `2026-03-14_06-03-23`. If empty, skip to Task 3 (no existing
artefact to inspect).

- [ ] **Step 2: Find the latest checkpoint in that run**

```bash
ls -t ~/projects/ddr/output/ddr-v0.5.2.dev2+g21a3a96b5-merit-training/2026-03-14_06-03-23/saved_models/ | head -5
```

The pre-flight reconnaissance showed only `epoch_1_mb_*` checkpoints — DDR's
training may have stopped early. Pick whichever has the highest
`epoch_X_mb_Y` (highest `epoch` first, then highest `mb`).

- [ ] **Step 3: Dump the existing checkpoint's `n` distribution**

If `scripts/dump_ddr_trained_params.py` doesn't exist yet (it will be
created in Task 3), skip ahead — you can re-run this step after Task 3
lands. If you want a 30-minute hand-rolled check now, drop into Python
inline:

```bash
cd ~/projects/ddr && uv run python <<'PY'
import sys
from pathlib import Path
import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
from ddr.nn.kan import kan as DdrKan
from ddr.routing.utils import denormalize

CKPT = Path("~/projects/ddr/output/ddr-v0.5.2.dev2+g21a3a96b5-merit-training/2026-03-14_06-03-23/saved_models").expanduser()
ckpt_files = sorted(CKPT.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
print(f"latest checkpoint: {ckpt_files[0].name}")

# pull DDR's training hyperparameters from its YAML
import yaml
y = yaml.safe_load(Path("~/projects/ddr/config/merit_training_config.yaml").expanduser().read_text())
input_vars = y["kan"]["input_var_names"]
learnable  = y["kan"]["learnable_parameters"]

model = DdrKan(
    input_var_names=input_vars, learnable_parameters=learnable,
    hidden_size=y["kan"]["hidden_size"], num_hidden_layers=y["kan"]["num_hidden_layers"],
    grid=y["kan"]["grid"], k=y["kan"]["k"], seed=y["seed"], device="cpu",
)
state = torch.load(ckpt_files[0], map_location="cpu")
model.load_state_dict(state)
model.eval()

ds = xr.open_dataset(Path("~/projects/ddr/data/merit_global_attributes_v2.nc").expanduser())
spatial = torch.tensor(
    ds[input_vars].to_array("variable").values,
    dtype=torch.float32,
)
for r in range(spatial.shape[0]):
    rm = torch.nanmean(spatial[r])
    spatial[r, torch.isnan(spatial[r])] = rm

stats_path = Path("~/projects/ddr/data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json").expanduser()
import json
stats = json.loads(stats_path.read_text())
means = torch.tensor([stats[v]["mean"] for v in input_vars], dtype=torch.float32)
stds  = torch.tensor([stats[v]["std"]  for v in input_vars], dtype=torch.float32)
normalized = ((spatial - means.unsqueeze(1)) / stds.unsqueeze(1)).T

with torch.no_grad():
    out = model(inputs=normalized)
log_space = y.get("params", {}).get("log_space_parameters", ["n"])
n_d = denormalize(out["n"], y["params"]["parameter_ranges"]["n"], "n" in log_space).cpu().numpy()

print(f"global reaches: {n_d.shape[0]}")
print(f"  median n: {np.median(n_d):.4f}")
print(f"  mean n:   {np.mean(n_d):.4f}")
print(f"  p5 / p95: {np.percentile(n_d, 5):.4f} / {np.percentile(n_d, 95):.4f}")
print(f"  frac in [0.020, 0.030]: {np.mean((n_d >= 0.020) & (n_d <= 0.030)):.3f}")
PY
```

- [ ] **Step 4: Append the finding to the spec**

Under §4 Layer 0.5, append the result you got:

```markdown
**Empirical finding (2026-06-03):**
- DDR existing checkpoint inspected: `<path>`
- Median `n`: 0.XXX
- p5 / p95: 0.XXX / 0.XXX
- Fraction in [0.020, 0.030]: X.XXX
- **Verdict:** saturated (matches DDRS) / healthy (DDR doesn't saturate) /
  inconclusive (only epoch 1 — too early to compare)
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md
git commit -m "$(cat <<'EOF'
docs/spec: Layer 0.5 finding from existing DDR checkpoint

Records the median / p5 / p95 / band-fraction of n predicted by DDR's
most recent existing checkpoint as a 30-minute sanity check before
committing to a fresh DDR re-train.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6:** Even if Layer 0.5 gives a strong signal either way,
  Layer 1 still runs (we need a fresh DDR run at parity config for the
  formal comparison). Move to Task 3.

---

## Task 3: Create `scripts/dump_ddr_trained_params.py`

**Spec ref:** §4 Layer 1 step 5 ("write a tiny script that does the same").

**Files:**
- Create: `/home/tbindas/projects/ddrs/scripts/dump_ddr_trained_params.py`

- [ ] **Step 1: Write the script**

```python
"""Dump DDR's *trained* KAN parameters (n / q_spatial / p_spatial) over the
CONUS subset of MERIT reaches.

Mirrors scripts/dump_ddr_init_params.py from the previous parity plan but
loads a trained checkpoint instead of building a fresh head.

Run under DDR's uv venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_ddr_trained_params.py \
        --checkpoint <path.pt> \
        --conus-comids /home/tbindas/projects/ddrs/.ddrs/runs/2026-06-03T09-45-09Z-train-and-test/kan_parameters.nc \
        --out /tmp/kan_params_trained_ddr.nc
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

ATTRS_NC = Path("~/projects/ddr/data/merit_global_attributes_v2.nc").expanduser()
STATS_JSON = Path(
    "~/projects/ddr/data/statistics/"
    "merit_attribute_statistics_merit_global_attributes_v2.nc.json"
).expanduser()
DDR_YAML = Path("~/projects/ddr/config/merit_training_config.yaml").expanduser()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help=".pt file produced by ddr/scripts/train.py")
    parser.add_argument(
        "--conus-comids", type=Path, required=True,
        help="NetCDF whose COMID coord defines the CONUS subset (typically "
             "DDRS's kan_parameters.nc).",
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="Output NetCDF path (e.g. /tmp/...nc)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda:0 — keep cpu for reproducibility.")
    args = parser.parse_args()

    sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
    from ddr.nn.kan import kan as DdrKan
    from ddr.routing.utils import denormalize

    # ── 1. Load DDR's hyperparameters from its YAML ─────────────────────
    y = yaml.safe_load(DDR_YAML.read_text())
    input_vars = y["kan"]["input_var_names"]
    learnable  = y["kan"]["learnable_parameters"]
    ranges     = y["params"]["parameter_ranges"]
    log_space  = set(y.get("params", {}).get("log_space_parameters", ["n"]))

    # ── 2. Build the KAN architecture + load trained weights ────────────
    model = DdrKan(
        input_var_names=input_vars,
        learnable_parameters=learnable,
        hidden_size=y["kan"]["hidden_size"],
        num_hidden_layers=y["kan"]["num_hidden_layers"],
        grid=y["kan"]["grid"],
        k=y["kan"]["k"],
        seed=y["seed"],
        device=args.device,
    )
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    # ── 3. Load global attributes + z-score normalize ───────────────────
    ds_attrs = xr.open_dataset(ATTRS_NC)
    global_comids = ds_attrs["COMID"].values.astype("int64")
    spatial = torch.tensor(
        ds_attrs[input_vars].to_array("variable").values,
        dtype=torch.float32,
        device=args.device,
    )
    # NaN-fill with per-feature mean (matches DDR's _predict_kan_params).
    for r in range(spatial.shape[0]):
        row_mean = torch.nanmean(spatial[r])
        spatial[r, torch.isnan(spatial[r])] = row_mean

    stats = json.loads(STATS_JSON.read_text())
    means = torch.tensor(
        [stats[v]["mean"] for v in input_vars],
        dtype=torch.float32, device=args.device,
    )
    stds = torch.tensor(
        [stats[v]["std"] for v in input_vars],
        dtype=torch.float32, device=args.device,
    )
    normalized = ((spatial - means.unsqueeze(1)) / stds.unsqueeze(1)).T

    # ── 4. Batched KAN inference over the global set ────────────────────
    batch_size = 50_000
    raw_parts: dict[str, list[np.ndarray]] = {k: [] for k in learnable}
    for start in range(0, normalized.shape[0], batch_size):
        chunk = normalized[start:start + batch_size].to(args.device)
        with torch.no_grad():
            out = model(inputs=chunk)
        for k in learnable:
            raw_parts[k].append(out[k].cpu().numpy())

    raw_concat = {k: np.concatenate(parts, axis=0) for k, parts in raw_parts.items()}

    denorm = {}
    for k in learnable:
        v_t = torch.from_numpy(raw_concat[k])
        denorm[k] = denormalize(v_t, ranges[k], k in log_space).numpy().astype("float32")

    # ── 5. Intersect to the CONUS COMID order from --conus-comids ───────
    conus_ds = xr.open_dataset(args.conus_comids)
    conus_comids = conus_ds["COMID"].values.astype("int64")

    # Build a global-COMID -> index map, then select in CONUS order.
    global_pos = {c: i for i, c in enumerate(global_comids)}
    missing = [c for c in conus_comids if c not in global_pos]
    if missing:
        raise SystemExit(
            f"{len(missing)} CONUS COMIDs missing from global attributes; "
            f"first 5: {missing[:5]}"
        )
    select = np.array([global_pos[c] for c in conus_comids], dtype=np.int64)

    out_vars = {k: denorm[k][select] for k in learnable}

    # ── 6. Write the NetCDF in CONUS order ──────────────────────────────
    xr.Dataset(
        {k: (("COMID",), out_vars[k]) for k in learnable},
        coords={"COMID": conus_comids},
        attrs={
            "source": f"ddr.nn.kan + checkpoint={args.checkpoint.name}",
            "seed": y["seed"],
            "grid": y["kan"]["grid"],
            "k": y["kan"]["k"],
            "ddrs_companion": str(args.conus_comids),
        },
    ).to_netcdf(args.out)
    print(f"wrote {len(conus_comids)} CONUS reaches → {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test the script against ANY existing DDR checkpoint**

Use whatever checkpoint Task 2 surfaced (or hand-pick any `.pt` under
`~/projects/ddr/output/`):

```bash
CKPT=$(ls -t ~/projects/ddr/output/ddr-v0.5.2.dev2+g21a3a96b5-merit-training/2026-03-14_06-03-23/saved_models/*.pt | head -1)
echo "smoke-testing with: $CKPT"
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_trained_params.py \
    --checkpoint "$CKPT" \
    --conus-comids /home/tbindas/projects/ddrs/.ddrs/runs/2026-06-03T09-45-09Z-train-and-test/kan_parameters.nc \
    --out /tmp/kan_params_trained_ddr_smoke.nc
```

Expected: writes 346,321 reaches, prints `wrote 346321 CONUS reaches → ...`.

Quick sanity-check on the output:

```bash
cd ~/projects/ddr && uv run python -c "
import xarray as xr, numpy as np
d = xr.open_dataset('/tmp/kan_params_trained_ddr_smoke.nc')
print('vars:', list(d.data_vars))
print('reaches:', d.sizes['COMID'])
print('n stats: median=%.4f mean=%.4f p5=%.4f p95=%.4f' % (
    np.median(d.n), np.mean(d.n),
    np.percentile(d.n, 5), np.percentile(d.n, 95)))
"
rm /tmp/kan_params_trained_ddr_smoke.nc
```

If this fails (shape mismatch, missing COMIDs, NaN output), STOP and
diagnose — likely the DDR YAML or stats path drifted.

- [ ] **Step 3: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add scripts/dump_ddr_trained_params.py
git commit -m "$(cat <<'EOF'
scripts: dump DDR's trained KAN parameters over CONUS

Loads a DDR .pt checkpoint, builds the matching ddr.nn.kan model,
applies the same z-score normalization as geometry_predictor.py, runs
batched inference over the full ~2.94M MERIT reaches, denormalizes,
and subsets to the CONUS COMID order from a companion NetCDF (typically
DDRS's kan_parameters.nc). Output is consumed by the Layer 2 parity
notebook.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Train DDR fresh at parity config

**Spec ref:** §4 Layer 1 steps 1-4.

**Files:**
- (Optional) Read/edit `~/projects/ddr/config/merit_training_config.yaml`
  if drift is found.
- Reads `~/projects/ddr/scripts/train_and_test.py`.

- [ ] **Step 1: Verify DDR's YAML hasn't drifted from parity baseline**

```bash
cd ~/projects/ddr && uv run python -c "
import yaml
y = yaml.safe_load(open('config/merit_training_config.yaml'))
assert y['seed'] == 42, f'seed {y[\"seed\"]} != 42'
assert y['np_seed'] == 42, f'np_seed {y[\"np_seed\"]} != 42'
assert y['kan']['grid'] == 50, f'grid {y[\"kan\"][\"grid\"]} != 50'
assert y['kan']['k'] == 2, f'k {y[\"kan\"][\"k\"]} != 2'
assert y['kan']['hidden_size'] == 21, f'hidden_size {y[\"kan\"][\"hidden_size\"]} != 21'
assert y['kan']['num_hidden_layers'] == 2
assert len(y['kan']['input_var_names']) == 10
assert y['kan']['learnable_parameters'] == ['n', 'q_spatial', 'p_spatial']
assert y['experiment']['epochs'] == 5
print('DDR YAML parity-baseline check: PASS')
"
```

If any assertion fails, STOP and report. Editing DDR's YAML is in scope but
requires user confirmation first — those values are paper-authored.

- [ ] **Step 2: Verify DDR's venv resolves**

```bash
cd ~/projects/ddr && uv run python -c "
import torch, ddr
from ddr.nn.kan import kan as DdrKan
print(f'torch={torch.__version__}  cuda available={torch.cuda.is_available()}')
print('DDR import OK')
"
```

Expected: prints torch version, `True` for CUDA, no traceback.

If `import ddr` fails on a missing dep, STOP — environment issue.

- [ ] **Step 3: Launch DDR training**

This takes 30–90 minutes on the workstation's GPU. Run in the background
and capture output.

```bash
cd ~/projects/ddr && nohup uv run python scripts/train_and_test.py \
    > /tmp/ddr_train.log 2>&1 &
echo "DDR training PID: $!"
```

Note the PID and the run directory that DDR creates under
`~/projects/ddr/output/ddr-v<version>-merit-training/<timestamp>/`. DDR
auto-creates one per run.

- [ ] **Step 4: Monitor for catastrophic divergence**

While training runs, check that the loss is decreasing — NOT going to NaN.

```bash
tail -f /tmp/ddr_train.log | grep --line-buffered -E '(loss|nan|NaN|error)'
```

Watch the first 20–30 mini-batches. Loss should fall smoothly. If it goes
NaN or rises monotonically, STOP — DDR has a separate issue.

- [ ] **Step 5: Confirm DDR finished**

```bash
ls -t ~/projects/ddr/output/ | head -3
ls -t ~/projects/ddr/output/$(ls -t ~/projects/ddr/output/ | head -1)/ | head -1
```

Find the freshest training-run dir + freshest timestamp inside. Confirm
its `saved_models/` directory contains `epoch_5_mb_*` checkpoints. Capture
the latest checkpoint path:

```bash
DDR_LATEST_CKPT=$(ls -t ~/projects/ddr/output/$(ls -t ~/projects/ddr/output/ | grep training | head -1)/$(ls -t ~/projects/ddr/output/$(ls -t ~/projects/ddr/output/ | grep training | head -1) | head -1)/saved_models/*.pt | head -1)
echo "DDR_LATEST_CKPT=$DDR_LATEST_CKPT"
```

Record `$DDR_LATEST_CKPT` for Task 5.

- [ ] **Step 6:** No commit — this task produced an external artefact
  (the DDR checkpoint), not a tracked file.

---

## Task 5: Dump DDR's trained `n` distribution

**Spec ref:** §4 Layer 1 step 5–6.

**Files:**
- Output (transient): `/tmp/kan_params_trained_ddr.nc`

- [ ] **Step 1: Dump trained parameters**

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_trained_params.py \
    --checkpoint "$DDR_LATEST_CKPT" \
    --conus-comids /home/tbindas/projects/ddrs/.ddrs/runs/2026-06-03T09-45-09Z-train-and-test/kan_parameters.nc \
    --out /tmp/kan_params_trained_ddr.nc 2>&1 | tail -10
```

Expected: prints `wrote 346321 CONUS reaches → /tmp/kan_params_trained_ddr.nc`.

If `$DDR_LATEST_CKPT` is unset (started a fresh shell), re-derive it:

```bash
DDR_LATEST_CKPT=$(ls -t ~/projects/ddr/output/$(ls -t ~/projects/ddr/output/ | grep training | head -1)/$(ls -t ~/projects/ddr/output/$(ls -t ~/projects/ddr/output/ | grep training | head -1) | head -1)/saved_models/*.pt | head -1)
echo "$DDR_LATEST_CKPT"
```

- [ ] **Step 2: Quick numerical sanity check**

```bash
cd ~/projects/ddr && uv run python -c "
import xarray as xr, numpy as np
ddr = xr.open_dataset('/tmp/kan_params_trained_ddr.nc')
ddrs = xr.open_dataset('/home/tbindas/projects/ddrs/.ddrs/runs/2026-06-03T09-45-09Z-train-and-test/kan_parameters.nc')
for k in ['n', 'q_spatial', 'p_spatial']:
    a, b = ddr[k].values, ddrs[k].values
    print(f'{k:>10}  ddr  median={np.median(a):.4f} p5={np.percentile(a,5):.4f} p95={np.percentile(a,95):.4f}')
    print(f'{k:>10}  ddrs median={np.median(b):.4f} p5={np.percentile(b,5):.4f} p95={np.percentile(b,95):.4f}')
print(f'frac DDR  n in [0.02, 0.03]: {((ddr.n.values >= 0.02) & (ddr.n.values <= 0.03)).mean():.3f}')
print(f'frac DDRS n in [0.02, 0.03]: {((ddrs.n.values >= 0.02) & (ddrs.n.values <= 0.03)).mean():.3f}')
"
```

Take a screenshot or copy the printed output. This is informational —
the formal verdict comes from Task 6's notebook.

- [ ] **Step 3:** No commit. The NetCDF is transient and consumed by Task 6.

---

## Task 6: Layer 2 — `parity_trained.md` skill ref + executed notebook

**Spec ref:** §4 Layer 2.

**Files:**
- Create: `/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/references/parity_trained.md`
- Create (transient, gitignored): `<RUN_DIR>/plots/parity_trained.ipynb` and `parity_trained_*.png`

- [ ] **Step 1: Write the skill reference**

Create `.claude/skills/ddrs-eval-plots/references/parity_trained.md`:

````markdown
# Layer 2 — DDR ↔ DDRS trained-`n` parity comparison

Counterpart to `parity_init.md`. Where `parity_init.md` compares the
INIT distributions, this compares the TRAINED distributions on the
exact same CONUS reaches.

## When to use

After running DDRS's `train-and-test` workflow at `seed=42` AND DDR's
`scripts/train_and_test.py` at the same `seed=42`. Both runs must use
identical hyperparameters (`grid=50, k=2`, the same 10 attribute
names in the same order, the same 3 learnable params, 5 epochs).

## Inputs

| File | Producer |
|------|----------|
| `<RUN_DIR>/kan_parameters.nc` | `cargo run --release --bin dump_parameters -- --config <RUN_DIR>/config.yaml --checkpoint <CKPT> --output <RUN_DIR>/kan_parameters.nc` |
| `/tmp/kan_params_trained_ddr.nc` | `cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_trained_params.py --checkpoint <DDR_CKPT> --conus-comids <RUN_DIR>/kan_parameters.nc --out /tmp/kan_params_trained_ddr.nc` |

## Notebook cells

Place at `<RUN_DIR>/plots/parity_trained.ipynb`.

### Cell 1 — load
```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ks_2samp, spearmanr

DDRS_NC = Path("../kan_parameters.nc")
DDR_NC  = Path("/tmp/kan_params_trained_ddr.nc")
PLOT_DIR = Path(".").resolve()

ddrs = xr.open_dataset(DDRS_NC)
ddr  = xr.open_dataset(DDR_NC)
assert (ddrs["COMID"].values == ddr["COMID"].values).all(), "COMID order must match"
print(f"{ddrs.sizes['COMID']:,} CONUS reaches in both files")
```

### Cell 2 — per-distribution stats
```python
rows = []
for k in ["n", "q_spatial", "p_spatial"]:
    a, b = ddrs[k].values, ddr[k].values
    ks = ks_2samp(a, b).statistic
    sp = spearmanr(a, b).correlation
    rows.append({
        "param": k,
        "ddrs_med":  float(np.median(a)),
        "ddr_med":   float(np.median(b)),
        "ddrs_p5":   float(np.percentile(a, 5)),
        "ddr_p5":    float(np.percentile(b, 5)),
        "ddrs_p95":  float(np.percentile(a, 95)),
        "ddr_p95":   float(np.percentile(b, 95)),
        "KS":        float(ks),
        "Spearman":  float(sp),
    })
pd.DataFrame(rows).set_index("param")
```

### Cell 3 — side-by-side histograms
```python
RANGES = {"n": (0.015, 0.25), "q_spatial": (0, 1), "p_spatial": (1, 200)}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, key in zip(axes, ["n", "q_spatial", "p_spatial"]):
    lo, hi = RANGES[key]
    bins = np.linspace(lo, hi, 80)
    ax.hist(ddr[key].values,  bins=bins, alpha=0.5, label="DDR  trained", density=True)
    ax.hist(ddrs[key].values, bins=bins, alpha=0.5, label="DDRS trained", density=True)
    ks = ks_2samp(ddr[key].values, ddrs[key].values).statistic
    ax.set_title(f"{key}  KS={ks:.3f}")
    ax.set_xlabel(key)
    ax.legend()
fig.suptitle("DDR ↔ DDRS trained-parameter distributions — seed=42, 5 epochs", fontsize=16)
fig.tight_layout()
fig.savefig(PLOT_DIR / "parity_trained_hist.png", dpi=200, bbox_inches="tight")
plt.show()
```

### Cell 4 — per-gauge scatter (for `n`)
```python
fig, ax = plt.subplots(figsize=(8, 8))
ax.hexbin(ddr["n"].values, ddrs["n"].values, gridsize=80, cmap="viridis", mincnt=1)
lo, hi = 0.015, 0.25
ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
sp = spearmanr(ddr["n"].values, ddrs["n"].values).correlation
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("DDR trained n"); ax.set_ylabel("DDRS trained n")
ax.set_title(f"Per-reach DDR vs DDRS trained n   Spearman = {sp:.3f}")
ax.legend()
fig.savefig(PLOT_DIR / "parity_trained_scatter.png", dpi=200, bbox_inches="tight")
plt.show()
```

### Cell 5 — verdict
```python
KS_PASS  = 0.10   # spec A4
SP_PASS  = 0.90   # spec A5
KS_FAIL  = 0.20
SP_FAIL  = 0.70

print("DDR ↔ DDRS trained-distribution parity (Layer 2):\n")
for k in ["n", "q_spatial", "p_spatial"]:
    a, b = ddrs[k].values, ddr[k].values
    ks = float(ks_2samp(a, b).statistic)
    sp = float(spearmanr(a, b).correlation)
    if   ks <= KS_PASS and sp >= SP_PASS: v = "✓ same saturation"
    elif ks >= KS_FAIL  or  sp <= SP_FAIL: v = "✗ real divergence"
    else:                                  v = "~ borderline"
    print(f"  {k:12s}  KS={ks:.4f}  Spearman={sp:+.4f}   {v}")
```

## Pass criterion

The `n` row of Cell 5 must print `✓ same saturation` for the trained
parity verdict to land. `q_spatial` and `p_spatial` are reported for
completeness — they were not the original symptom.

If `n` lands `~ borderline`: bring to user. If `n` lands `✗`: a new
spec localizing the `src/training/` divergence is warranted.
````

- [ ] **Step 2: Materialize the notebook in the run dir**

The skill reference is read-only documentation; the actual `.ipynb`
file lives under `<RUN_DIR>/plots/`. Convert the markdown cells to a
Jupyter notebook via `nbformat`:

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
RUN_DIR=/home/tbindas/projects/ddrs/.ddrs/runs/2026-06-03T09-45-09Z-train-and-test
mkdir -p "$RUN_DIR/plots"

uv run python <<'PY'
import nbformat as nbf
import os

skill_md = open("/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/references/parity_trained.md").read()
# Extract the 5 ```python ...``` blocks under "## Notebook cells".
import re
notebook_section = skill_md.split("## Notebook cells", 1)[1].split("## Pass criterion", 1)[0]
code_blocks = re.findall(r"```python\n(.*?)```", notebook_section, re.DOTALL)
assert len(code_blocks) == 5, f"expected 5 cells, got {len(code_blocks)}"

nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell(
    "# DDR ↔ DDRS trained-`n` parity (Layer 2)\n\n"
    "Notebook generated from "
    "`.claude/skills/ddrs-eval-plots/references/parity_trained.md`."
))
for src in code_blocks:
    nb.cells.append(nbf.v4.new_code_cell(src.strip()))

out = os.path.join(os.environ["RUN_DIR"], "plots", "parity_trained.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
PY
```

- [ ] **Step 3: Execute the notebook**

```bash
cd /home/tbindas/projects/ddrs/ddrs-py
RUN_DIR=/home/tbindas/projects/ddrs/.ddrs/runs/2026-06-03T09-45-09Z-train-and-test
uv run jupyter nbconvert --to notebook --execute \
    "$RUN_DIR/plots/parity_trained.ipynb" \
    --output parity_trained.ipynb \
    --output-dir "$RUN_DIR/plots" 2>&1 | tail -10
```

Expected: exits cleanly. Confirms 346,321 reaches loaded. Two PNGs saved
(`parity_trained_hist.png`, `parity_trained_scatter.png`). Cell 5 prints
the per-parameter verdict.

Capture Cell 5's output (the three lines starting with `n`, `q_spatial`,
`p_spatial`).

- [ ] **Step 4: Commit the skill reference** (the notebook + PNGs stay
  under `.ddrs/runs/` which is gitignored)

```bash
cd /home/tbindas/projects/ddrs
git add .claude/skills/ddrs-eval-plots/references/parity_trained.md
git commit -m "$(cat <<'EOF'
ddrs-eval-plots: add Layer 2 trained-distribution parity reference

Notebook recipe for the DDR ↔ DDRS trained-parameter histogram +
per-gauge scatter + KS / Spearman verdict — the test that
determines whether DDRS's n saturation faithfully mirrors DDR's
behaviour or indicates a src/training/ divergence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Record verdict in spec §5

**Spec ref:** §5 "What success looks like".

**Files:**
- Modify: `docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md`

- [ ] **Step 1: Append the verdict**

At the end of §5 (after the three-row outcome table), add:

```markdown
---

## §5.1 Empirical verdict (filled by Task 7 of the plan)

**Run dates:**
- DDR train: `<YYYY-MM-DD HH:MM>` at `<DDR run dir>`
- DDRS train: `2026-06-03T09-45-09Z-train-and-test` (existing)

**Cell 5 output:**
```
n             KS=<v>  Spearman=<v>   <verdict>
q_spatial     KS=<v>  Spearman=<v>   <verdict>
p_spatial     KS=<v>  Spearman=<v>   <verdict>
```

**Outcome:** <one of the three rows from §5's table; quote it verbatim>

**Next step:** <inherits from the §5 table's "Next step" column for the
matched outcome>
```

Fill in the brackets with the actual values from Task 6 Step 3.

- [ ] **Step 2: Commit**

```bash
cd /home/tbindas/projects/ddrs
git add docs/superpowers/specs/2026-06-03-ddr-ddrs-trained-saturation-parity-design.md
git commit -m "$(cat <<'EOF'
docs/spec: record empirical verdict for trained-n parity

Appends §5.1 with the Cell-5 KS + Spearman + per-parameter verdict
from a fresh DDR train + Layer 2 notebook execution. The outcome
selects one of the three rows in the §5 outcome table and names
the next investigation step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Spec coverage map

| Spec section | Plan task(s) |
|--------------|-------------|
| §2 C1 (DDR train may have drifted) | Task 4 step 1 (YAML drift check), step 2 (venv resolve) |
| §2 C2 (GPU non-det forces loose KS threshold) | Task 6 cell 5 uses A4's 0.10 / 0.20 cutoffs |
| §2 C3 (training-config audit gap) | Task 1 (full Layer 0) |
| §2 C4 ("saturation may be correct") | Task 7's §5.1 records the verdict; out-of-scope follow-ups noted in spec §7 |
| §2 C5 (stale existing DDR outputs) | Task 2 (Layer 0.5 quick check) and Task 4 (fresh train) |
| §2 C6 (normalization mismatch) | Task 3's script uses DDR's own `merit_attribute_statistics_*.json` + DDR's `denormalize` to eliminate this risk |
| §3 A1-A6 | All used as background; Task 4 + Task 5 cite specific assumptions |
| §4 Layer 0 | Task 1 |
| §4 Layer 0.5 | Task 2 |
| §4 Layer 1 | Tasks 3, 4, 5 |
| §4 Layer 2 | Task 6 |
| §5 outcome | Task 7 |
| §6 implementation order | Plan task order |

---

Plan complete and saved to `docs/superpowers/plans/2026-06-03-ddr-ddrs-trained-saturation-parity.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, two-stage review (spec compliance + code quality), fast iteration in this session.

**2. Inline Execution** — Execute tasks in this session via `superpowers:executing-plans`, batch execution with checkpoints for review.

Which approach?
