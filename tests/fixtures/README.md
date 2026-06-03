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
