# DDR ↔ DDRS KAN parity fixtures

These are byte-for-byte snapshots of DDR-Python's `ddr.nn.kan.kan` at a
fixed seed, consumed by the parity integration tests under `tests/`.

## Regenerating

All scripts run under DDR's `uv` venv (`~/projects/ddr/.venv/`):

```bash
cd ~/projects/ddr
# per-tensor (mean, std) for Layer 1
uv run python ~/projects/ddrs/scripts/dump_kan_init_stats.py
# full param + forward + backward fixture for Layers 2-3
uv run python ~/projects/ddrs/scripts/dump_kan_fixture.py
```

Regenerate any time DDR's `nn/kan.py` or `pykan` is updated, then re-run
the parity test suite:

```bash
cargo test --features fixtures --test kan_head_init_parity \
                              --test kan_head_fixture_forward \
                              --test kan_head_fixture_backward
```

## Files

| File | Producer | Consumer |
|------|----------|----------|
| `kan_init_stats_ddr.csv` | `dump_kan_init_stats.py` | `kan_head_init_parity.rs` |
| `kan_head_init_seed42.npz` | `dump_kan_fixture.py` | `kan_head_fixture_forward.rs`, `kan_head_fixture_backward.rs` |

## Streamflow fixture stores

Tiny icechunk Qr stores used by the hourly-streamflow and import-command
integration tests.  Regenerate only if the DDR Q' store contract changes
(see `docs/nh-qprime-store-contract.md`):

```bash
cd ~/projects/ddr
uv run python ~/projects/ddrs/scripts/make_streamflow_fixtures.py
```

Deterministic value formulas (so tests can assert exact elements):

- `qr_daily.ic` — 4 divides × 10 days from 1981-01-01; `Qr[j, t] = (j+1)*100 + t`
- `qr_hourly.ic` — 4 divides × 240 hours from 1981-01-01T00; `Qr[j, h] = (j+1)*1000 + h`
- `qr_minutes.ic` — same shape as the first 48 hours of the hourly store but with
  units `"minutes since 1981-01-01"`; exercises the sniff hard-error path (bad units
  → rejection, no data read).

| File | Producer | Consumer |
|------|----------|----------|
| `qr_daily.ic` | `make_streamflow_fixtures.py` | `hourly_streamflow.rs`, `import_cmd.rs` |
| `qr_hourly.ic` | `make_streamflow_fixtures.py` | `hourly_streamflow.rs`, `import_cmd.rs` |
| `qr_minutes.ic` | `make_streamflow_fixtures.py` | `hourly_streamflow.rs` |
