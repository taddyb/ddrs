# Adjoint influence map — GAGES-II nested-reference population

Same study as `experiments/adjoint` (see its README for definitions and output
schema), run over every GAGES-II **Ref** gauge in the training population that
has at least one other training gauge nested in its subgraph, plus those
upstream gauges. Selection: `src/experiment/adjoint/gauges.rs`
(`nested_reference_selection`). Arms run concurrently, one thread each
(`--jobs`).

```bash
target/release/ddrs --workspace .ddrs experiment adjoint-conus --dry-run   # writes gauges.csv, stops
nohup target/release/ddrs --workspace .ddrs experiment adjoint-conus --backend cpu > output/adjoint_conus.log 2>&1 &
~/projects/ddr/.venv/bin/python experiments/adjoint/plots.py .ddrs/experiments/adjoint-conus/<ts>
```

Per-gauge failures are recorded in `manifest.json` `notes` and do not stop the
sweep. The finite-difference gate runs once per arm (first gauge, first anchor).
