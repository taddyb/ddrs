# NH Q' Store Contract + Hourly-Native Reading + `ddrs import` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route neural-hydrology LSTM outputs in ddrs: sniff daily vs hourly icechunk Q' stores from their CF time axis, read hourly stores natively (no repeat-24/disagg), and add `ddrs import` to validate + register any conforming store as a data-source group.

**Architecture:** `StreamflowStore` (icechunk reader) gains a `resolution: Frequency` field parsed from the time `units` attribute; its three read methods branch on it, sharing one generic slab reader. A new `src/cli/import.rs` opens a store via the existing `StreamflowSource::open` sniff, validates the contract, reports COMID coverage against the resolved adjacency, and writes a `config/sources/<name>.yaml` group with the `streamflow:` path swapped in. The NH-side producer (`~/projects/neuralhydrology/examples/merit_hydro/forward_merit.py`) is unchanged.

**Tech Stack:** Rust (zarrs + icechunk crates, ndarray, clap, serde_yaml); one Python fixture script run under DDR's uv venv.

**Spec:** `docs/superpowers/specs/2026-07-01-nh-qprime-import-design.md`
**Branch:** work on the current branch `unit_catchments`.

**Confirmed store facts (probed 2026-07-01, drive the sniff grammar and tests):**

| Store (`/mnt/ssd1/data/icechunk/`) | time units | range | n_time | divides |
|---|---|---|---|---|
| `daily_lstm_merit_unit_catchments.ic` | `days since 1981-01-01 00:00:00` | 1981-01-01 → 2020-12-30 | 14,609 | 288,421 |
| `hourly_lstm_merit_unit_catchments.ic` | `hours since 1981-01-01 00:00:00` | 1981-01-01T00 → 2020-12-31T23 | 350,640 | 197,088 |
| `daily_dhbv2_merit_unit_catchments.ic` | `days since 1980-01-01 00:00:00` | 1980-01-01 → 2020-12-30 | 14,975 | 288,421 |
| `merit_dhbv2_UH_retrospective.ic` | `days since 1980-01-01` | 1980-01-01 → 2020-12-31 | 14,976 | 197,088 |

All four: `Qr(divide_id, time)` float32 with `units: m^3/s`, `divide_id` int64, time int64 on disk.

**Known accepted cost (do NOT "fix" in this plan):** `MeritGagesDataset::collate` reads `q_prime_daily` unconditionally (`src/data/dataset.rs:496-500`). On an hourly store that re-reads the same chunks a second time (aggregated). The smoke train measures it; changing collate's read pattern is out of scope.

---

### Task 1: Test fixture stores (Python, run under DDR's venv)

Tiny icechunk fixtures with deterministic values so hourly read alignment is assertable to the exact element. Checked into git (a few KB each).

**Files:**
- Create: `scripts/make_streamflow_fixtures.py`
- Create (generated): `tests/fixtures/qr_daily.ic/`, `tests/fixtures/qr_hourly.ic/`, `tests/fixtures/qr_minutes.ic/`

- [ ] **Step 1: Write the fixture generator**

```python
"""Write tiny icechunk Qr fixture stores for ddrs integration tests.

Run under DDR's uv venv (it has icechunk + xarray):

    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/make_streamflow_fixtures.py

Layout matches the DDR Q' store contract (docs/nh-qprime-store-contract.md):
Qr(divide_id, time) f32 m^3/s, divide_id int64, CF int64 time axis.

Deterministic values so tests can assert exact elements:
  qr_daily.ic   : 4 divides x 10 days,   Qr[j, t] = (j+1)*100  + t
  qr_hourly.ic  : 4 divides x 240 hours, Qr[j, h] = (j+1)*1000 + h
  qr_minutes.ic : sniff-rejection fixture (units "minutes since ...")
"""
from pathlib import Path
import shutil

import icechunk
import numpy as np
import xarray as xr

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DIVIDES = np.array([101, 102, 103, 104], dtype=np.int64)


def write_store(path: Path, times: np.ndarray, qr: np.ndarray, time_units: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    ds = xr.Dataset(
        data_vars={
            "Qr": (["divide_id", "time"], qr.astype(np.float32), {"units": "m^3/s"}),
        },
        coords={
            "divide_id": ("divide_id", DIVIDES),
            "time": ("time", times),
        },
        attrs={"units": "m^3/s", "source": "ddrs test fixture"},
    )
    ds.to_zarr(
        session.store,
        mode="w",
        encoding={"time": {"units": time_units, "dtype": "int64"}},
    )
    session.commit("fixture")
    print(f"wrote {path}")


def main() -> None:
    n_days = 10
    daily_times = np.datetime64("1981-01-01") + np.arange(n_days).astype("timedelta64[D]")
    daily = (np.arange(4)[:, None] + 1) * 100 + np.arange(n_days)[None, :]
    write_store(FIXTURES / "qr_daily.ic", daily_times, daily, "days since 1981-01-01")

    n_hours = n_days * 24
    hourly_times = np.datetime64("1981-01-01T00") + np.arange(n_hours).astype("timedelta64[h]")
    hourly = (np.arange(4)[:, None] + 1) * 1000 + np.arange(n_hours)[None, :]
    write_store(
        FIXTURES / "qr_hourly.ic", hourly_times, hourly,
        "hours since 1981-01-01 00:00:00",
    )

    # Same data, unsupported units string — exercises the sniff hard-error.
    write_store(
        FIXTURES / "qr_minutes.ic", hourly_times[:48], hourly[:, :48],
        "minutes since 1981-01-01",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and sanity-check the output**

Run: `cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/make_streamflow_fixtures.py`
Expected: three `wrote .../tests/fixtures/qr_*.ic` lines.

Run: `du -sh ~/projects/ddrs/tests/fixtures/qr_*.ic`
Expected: each well under 1 MB.

Verify the encoding took (this is what the Rust sniff will read):

```bash
cd ~/projects/ddr && uv run python -c "
import icechunk as ic, xarray as xr
for n in ['qr_daily.ic', 'qr_hourly.ic', 'qr_minutes.ic']:
    p = '/home/tbindas/projects/ddrs/tests/fixtures/' + n
    repo = ic.Repository.open(ic.local_filesystem_storage(p))
    ds = xr.open_zarr(repo.readonly_session('main').store, consolidated=False)
    print(n, ds.time.encoding.get('units'), ds.Qr.shape)
"
```
Expected: `days since 1981-01-01` / `hours since 1981-01-01 00:00:00` / `minutes since 1981-01-01`, shapes `(4, 10)`, `(4, 240)`, `(4, 48)`.

- [ ] **Step 3: Commit**

```bash
cd ~/projects/ddrs
git add scripts/make_streamflow_fixtures.py tests/fixtures/qr_daily.ic tests/fixtures/qr_hourly.ic tests/fixtures/qr_minutes.ic
git commit -m "test: icechunk Qr fixture stores (daily/hourly/bad-units)"
```

---

### Task 2: `parse_cf_units` — resolution sniff (TDD)

**Files:**
- Modify: `src/data/store/icechunk.rs:234-262` (replace `parse_cf_epoch` internals; keep a daily-only wrapper for the USGS obs store)

- [ ] **Step 1: Write the failing unit tests**

Append inside the existing `mod tests` in `src/data/store/icechunk.rs`:

```rust
    fn attrs_with_units(u: &str) -> serde_json::Map<String, serde_json::Value> {
        let mut m = serde_json::Map::new();
        m.insert("units".into(), serde_json::Value::String(u.into()));
        m
    }

    #[test]
    fn parse_cf_units_daily() {
        let (epoch, res) =
            parse_cf_units(&attrs_with_units("days since 1980-01-01"), Path::new("/t")).unwrap();
        assert_eq!(epoch, chrono::NaiveDate::from_ymd_opt(1980, 1, 1).unwrap());
        assert_eq!(res, crate::data::dates::Frequency::Daily);
    }

    #[test]
    fn parse_cf_units_daily_with_time_of_day() {
        // daily_lstm store encodes "days since 1981-01-01 00:00:00".
        let (epoch, res) =
            parse_cf_units(&attrs_with_units("days since 1981-01-01 00:00:00"), Path::new("/t"))
                .unwrap();
        assert_eq!(epoch, chrono::NaiveDate::from_ymd_opt(1981, 1, 1).unwrap());
        assert_eq!(res, crate::data::dates::Frequency::Daily);
    }

    #[test]
    fn parse_cf_units_hourly() {
        // hourly_lstm store encodes "hours since 1981-01-01 00:00:00".
        let (epoch, res) =
            parse_cf_units(&attrs_with_units("hours since 1981-01-01 00:00:00"), Path::new("/t"))
                .unwrap();
        assert_eq!(epoch, chrono::NaiveDate::from_ymd_opt(1981, 1, 1).unwrap());
        assert_eq!(res, crate::data::dates::Frequency::Hourly);
    }

    #[test]
    fn parse_cf_units_rejects_other_resolutions() {
        let err = parse_cf_units(&attrs_with_units("minutes since 1981-01-01"), Path::new("/t"))
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("minutes since"), "error must name the units: {msg}");
        assert!(msg.contains("days since"), "error must name what IS supported: {msg}");
    }

    #[test]
    fn parse_cf_epoch_rejects_hourly_axis() {
        // The daily-only wrapper (used by the USGS observations store) must
        // refuse an hourly axis rather than silently mis-scaling.
        let err = parse_cf_epoch(&attrs_with_units("hours since 1980-01-01"), Path::new("/t"))
            .unwrap_err();
        assert!(err.to_string().contains("daily"), "got: {err}");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib parse_cf`
Expected: FAIL — `parse_cf_units` not found (compile error).

- [ ] **Step 3: Implement `parse_cf_units`, keep `parse_cf_epoch` as the daily-only wrapper**

Replace the whole `parse_cf_epoch` function (`src/data/store/icechunk.rs:234-262`) with:

```rust
/// Parse the CF `units` attribute of a time coordinate and return the epoch
/// plus the native axis resolution. Supported forms (see
/// docs/nh-qprime-store-contract.md):
///   "days since YYYY-MM-DD[ HH:MM:SS]"  → Daily
///   "hours since YYYY-MM-DD[ HH:MM:SS]" → Hourly
/// Anything else is a hard error naming the store and the units string — a
/// mis-scaled time axis must never be silently accepted.
pub(crate) fn parse_cf_units(
    attrs: &serde_json::Map<String, serde_json::Value>,
    path: &Path,
) -> Result<(NaiveDate, crate::data::dates::Frequency)> {
    use crate::data::dates::Frequency;

    let units = attrs
        .get("units")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DataError::Malformed {
            path: path.to_path_buf(),
            message: "time array missing 'units' attribute".into(),
        })?;
    let (date_str, resolution) = if let Some(rest) = units.strip_prefix("days since ") {
        (rest, Frequency::Daily)
    } else if let Some(rest) = units.strip_prefix("hours since ") {
        (rest, Frequency::Hourly)
    } else {
        return Err(DataError::Malformed {
            path: path.to_path_buf(),
            message: format!(
                "unsupported time units {units:?}: expected \"days since …\" \
                 or \"hours since …\""
            ),
        });
    };
    // The date portion may be followed by a time-of-day component, e.g.
    // "1981-01-01 00:00:00" — take only the first token.
    let date_part = date_str.split_whitespace().next().unwrap_or("");
    let epoch =
        NaiveDate::parse_from_str(date_part, "%Y-%m-%d").map_err(|e| DataError::Malformed {
            path: path.to_path_buf(),
            message: format!("cannot parse epoch from units {units:?}: {e}"),
        })?;
    Ok((epoch, resolution))
}

/// Daily-only wrapper for stores whose axis MUST be daily (USGS observations).
pub(crate) fn parse_cf_epoch(
    attrs: &serde_json::Map<String, serde_json::Value>,
    path: &Path,
) -> Result<NaiveDate> {
    match parse_cf_units(attrs, path)? {
        (epoch, crate::data::dates::Frequency::Daily) => Ok(epoch),
        (_, crate::data::dates::Frequency::Hourly) => Err(DataError::Malformed {
            path: path.to_path_buf(),
            message: "expected a daily time axis (\"days since …\"), got hourly".into(),
        }),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib parse_cf`
Expected: 5 tests PASS. Also run `cargo test --lib` — no other lib test may break (`StreamflowStore::open` still calls `parse_cf_epoch`, unchanged behavior until Task 3).

- [ ] **Step 5: Commit**

```bash
git add src/data/store/icechunk.rs
git commit -m "feat(data): parse_cf_units sniffs daily vs hourly CF time axes"
```

---

### Task 3: Resolution-aware `StreamflowStore` (TDD, fixture-backed)

The core change. `StreamflowStore` gains `resolution`; the existing `read_window_daily` body becomes a resolution-agnostic `read_slab` over native time steps; the three public methods branch on resolution. Daily-path behavior must stay byte-identical.

**Files:**
- Modify: `src/data/store/icechunk.rs:177-231` (struct + open), `:286-408` (read methods)
- Create: `tests/hourly_streamflow.rs`

- [ ] **Step 1: Write the failing integration tests**

Create `tests/hourly_streamflow.rs`:

```rust
//! Fixture-backed tests for resolution-aware Q' reading.
//!
//! Fixtures are generated by `scripts/make_streamflow_fixtures.py` (run under
//! DDR's uv venv) and checked into tests/fixtures/. Deterministic values:
//!   qr_daily.ic  : 4 divides [101..104] x 10 days,   Qr[j, t] = (j+1)*100  + t
//!   qr_hourly.ic : 4 divides [101..104] x 240 hours, Qr[j, h] = (j+1)*1000 + h
//! Both axes start 1981-01-01.

use chrono::NaiveDate;

use ddrs::data::dates::{Frequency, RhoWindow, TimeAxis};
use ddrs::data::ids::Comid;
use ddrs::data::store::{StreamflowSource, StreamflowStore};
use ddrs::data::TestWindow;

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

const COMIDS: [Comid; 4] = [Comid(101), Comid(102), Comid(103), Comid(104)];

#[test]
fn hourly_store_opens_with_hourly_resolution() {
    let s = StreamflowStore::open(fixture("qr_hourly.ic")).expect("open");
    assert_eq!(s.resolution, Frequency::Hourly);
    assert_eq!(s.time_start, d(1981, 1, 1));
    assert_eq!(s.n_time, 240);
    assert_eq!(s.index.len(), 4);
}

#[test]
fn daily_store_opens_with_daily_resolution() {
    let s = StreamflowStore::open(fixture("qr_daily.ic")).expect("open");
    assert_eq!(s.resolution, Frequency::Daily);
    assert_eq!(s.time_start, d(1981, 1, 1));
    assert_eq!(s.n_time, 10);
}

#[test]
fn minutes_axis_is_rejected_at_open() {
    let err = StreamflowSource::open(fixture("qr_minutes.ic")).unwrap_err();
    assert!(
        err.to_string().contains("unsupported time units"),
        "got: {err}"
    );
}

#[test]
fn hourly_read_window_slices_natively() {
    let s = StreamflowStore::open(fixture("qr_hourly.ic")).expect("open");
    // Window: days [2, 6) of the axis → hours [48, 120); n_hourly = 3*24 = 72.
    let w = RhoWindow {
        start_day_idx: 2,
        rho_days: 4,
        window_start: d(1981, 1, 3),
    };
    let q = s.read_window(&w, &COMIDS).expect("read_window");
    assert_eq!(q.shape(), &[72, 4]);
    for h in 0..72 {
        for j in 0..4 {
            let expect = (j as f32 + 1.0) * 1000.0 + (48 + h) as f32;
            assert_eq!(q[(h, j)], expect, "mismatch at hour {h}, divide {j}");
        }
    }
}

#[test]
fn hourly_read_window_daily_is_24h_mean() {
    let s = StreamflowStore::open(fixture("qr_hourly.ic")).expect("open");
    let q = s
        .read_window_daily(d(1981, 1, 3), 4, &COMIDS)
        .expect("read_window_daily");
    assert_eq!(q.shape(), &[4, 4]);
    // Day d of the window covers hours 48+24d .. 48+24d+24; the mean of a
    // 24-term arithmetic ramp k..k+23 is k + 11.5.
    for day in 0..4 {
        for j in 0..4 {
            let expect = (j as f32 + 1.0) * 1000.0 + (48 + 24 * day) as f32 + 11.5;
            assert_eq!(q[(day, j)], expect, "mismatch at day {day}, divide {j}");
        }
    }
}

#[test]
fn hourly_read_test_window_is_contiguous() {
    let s = StreamflowStore::open(fixture("qr_hourly.ic")).expect("open");
    let axis = TimeAxis::new(d(1981, 1, 1), d(1981, 1, 10));
    let w = TestWindow::new(&axis, 2, 4); // hours [48, 144), no trailing trim
    let q = s.read_test_window(&w, &COMIDS).expect("read_test_window");
    assert_eq!(q.shape(), &[96, 4]);
    assert_eq!(q[(0, 0)], 1000.0 + 48.0);
    assert_eq!(q[(95, 3)], 4000.0 + 143.0);
}

#[test]
fn hourly_missing_comid_gets_fill() {
    let s = StreamflowStore::open(fixture("qr_hourly.ic")).expect("open");
    let w = RhoWindow {
        start_day_idx: 0,
        rho_days: 2,
        window_start: d(1981, 1, 1),
    };
    let q = s
        .read_window(&w, &[Comid(101), Comid(999)])
        .expect("read_window");
    assert_eq!(q.shape(), &[24, 2]);
    assert_eq!(q[(5, 0)], 1000.0 + 5.0);
    assert_eq!(q[(5, 1)], 0.001, "missing COMID must fill with 0.001");
}

#[test]
fn hourly_out_of_range_windows_hard_error() {
    let s = StreamflowStore::open(fixture("qr_hourly.ic")).expect("open");
    // Before store start.
    let before = RhoWindow {
        start_day_idx: 0,
        rho_days: 2,
        window_start: d(1980, 12, 1),
    };
    let err = s.read_window(&before, &COMIDS).unwrap_err();
    assert!(err.to_string().contains("before store start"), "got: {err}");
    // Past store end (store holds 10 days).
    let past = RhoWindow {
        start_day_idx: 8,
        rho_days: 5,
        window_start: d(1981, 1, 9),
    };
    assert!(s.read_window(&past, &COMIDS).is_err());
}

#[test]
fn daily_fixture_read_window_keeps_repeat24_semantics() {
    // Pins the daily path: values repeat 24x per day with the trailing-day trim.
    let s = StreamflowStore::open(fixture("qr_daily.ic")).expect("open");
    let w = RhoWindow {
        start_day_idx: 2,
        rho_days: 4,
        window_start: d(1981, 1, 3),
    };
    let q = s.read_window(&w, &COMIDS).expect("read_window");
    assert_eq!(q.shape(), &[72, 4]);
    for h in 0..72 {
        for j in 0..4 {
            let expect = (j as f32 + 1.0) * 100.0 + (2 + h / 24) as f32;
            assert_eq!(q[(h, j)], expect, "mismatch at hour {h}, divide {j}");
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test hourly_streamflow`
Expected: compile error — `StreamflowStore` has no field `resolution`.

- [ ] **Step 3: Add `resolution` to the struct and open()**

In `src/data/store/icechunk.rs`, change the struct (`:177-187`):

```rust
pub struct StreamflowStore {
    pub path: PathBuf,
    pub index: IdIndex<Comid>,
    /// First calendar day covered by the store (for hourly stores, the day
    /// containing the first hour — open() enforces hour-0 alignment).
    pub time_start: NaiveDate,
    /// Length of the NATIVE time axis: days for daily stores, hours for
    /// hourly stores.
    pub n_time: usize,
    /// Native axis resolution, sniffed from the CF `units` attribute.
    pub resolution: crate::data::dates::Frequency,
    // SP-3 may consolidate to a shared runtime; keep the Arc alive so the
    // icechunk Store is not dropped while `qr` is in use.
    #[allow(dead_code)]
    storage: Arc<IcZarrStorage>,
    qr: ZarrArray<dyn ReadableStorageTraits>,
}
```

In `open()` (`:190-231`), replace the epoch/time_start block:

```rust
        // 1. Read `time` coord: shape (n_time,), dtype int64. CF units are
        //    "days since …" (daily) or "hours since …" (hourly) — the sniff
        //    that decides this store's native resolution.
        let time_arr = ZarrArray::open(readable.clone(), "/time")
            .map_err(|e| ic_err(&path, e))?;
        let (time_epoch, resolution) = parse_cf_units(time_arr.attributes(), &path)?;
        let time_subset = time_arr.subset_all();
        let time_i64: Vec<i64> = time_arr
            .retrieve_array_subset(&time_subset)
            .map_err(|e| ic_err(&path, e))?;
        let n_time = time_i64.len();
        if n_time == 0 {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: "time axis is empty".into(),
            });
        }
        let time_start = match resolution {
            crate::data::dates::Frequency::Daily => {
                time_epoch + chrono::Duration::days(time_i64[0])
            }
            crate::data::dates::Frequency::Hourly => {
                // Contract: hourly axes start at hour 0 of a day and are
                // contiguous (docs/nh-qprime-store-contract.md). The full
                // scan is cheap (~2.8 MB of i64 for 40 years of hours).
                if time_i64[0] % 24 != 0 {
                    return Err(DataError::Malformed {
                        path: path.clone(),
                        message: format!(
                            "hourly store must start at hour 0 of a day; \
                             first time value is {}",
                            time_i64[0]
                        ),
                    });
                }
                if let Some(i) =
                    (1..time_i64.len()).find(|&i| time_i64[i] - time_i64[i - 1] != 1)
                {
                    return Err(DataError::Malformed {
                        path: path.clone(),
                        message: format!("hourly time axis has a gap at index {i}"),
                    });
                }
                time_epoch + chrono::Duration::days(time_i64[0] / 24)
            }
        };
```

and the constructor return:

```rust
        Ok(Self { path, index, time_start, n_time, resolution, storage, qr })
```

- [ ] **Step 4: Refactor reads — `native_start_index` + `read_slab` + branching public methods**

Replace the whole second `impl StreamflowStore` block (`:286-409`) with:

```rust
impl StreamflowStore {
    /// Store-local index of `window_start` on the NATIVE time axis
    /// (day index for daily stores, hour index for hourly stores).
    fn native_start_index(&self, window_start: NaiveDate) -> Result<usize> {
        let days = (window_start - self.time_start).num_days();
        if days < 0 {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window starts {} before store start {}",
                    window_start, self.time_start
                ),
            });
        }
        Ok(match self.resolution {
            crate::data::dates::Frequency::Daily => days as usize,
            crate::data::dates::Frequency::Hourly => days as usize * 24,
        })
    }

    /// Read `(n_steps, N)` from native time-axis positions
    /// `[start_step, start_step + n_steps)` for `comids`. Missing COMIDs are
    /// filled with `0.001` (discharge minimum, mirrors DDR's
    /// `torch.full(..., fill_value=0.001)` in `readers.py:464-468`).
    fn read_slab(
        &self,
        start_step: usize,
        n_steps: usize,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        let end_step = start_step + n_steps;
        if end_step > self.n_time {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window extends to store step {end_step} but n_time={} \
                     ({:?} axis)",
                    self.n_time, self.resolution
                ),
            });
        }

        // Resolve COMIDs → divide-axis positions.
        // `positions_of` returns positions in the order of non-missing inputs,
        // plus a list of indices (into `comids`) that were missing.
        let (positions, missing_indices) = self.index.positions_of(comids);
        let missing_set: std::collections::HashSet<usize> =
            missing_indices.iter().copied().collect();
        let n_out = comids.len();

        // Pre-fill with the discharge minimum; missing COMIDs keep this value.
        let mut out = Array2::<f32>::from_elem((n_steps, n_out), 0.001);

        if positions.is_empty() {
            return Ok(out);
        }

        // Contiguous divide-axis read covering [min_pos, max_pos].
        let min_pos = *positions.iter().min().unwrap();
        let max_pos = *positions.iter().max().unwrap();
        let div_range_end = max_pos + 1;
        let div_count = div_range_end - min_pos;

        // Qr is stored as (divide_id, time). Subset: axis 0 = divide, axis 1 = time.
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[
            (min_pos as u64)..(div_range_end as u64),
            (start_step as u64)..(end_step as u64),
        ]);
        let raw_f32: Vec<f32> = self
            .qr
            .retrieve_array_subset(&subset)
            .map_err(|e| ic_err(&self.path, e))?;
        // raw_f32 is row-major: shape (div_count, n_steps).
        debug_assert_eq!(raw_f32.len(), div_count * n_steps);

        // Scatter into the output. Walk `comids` in order; for each
        // non-missing entry consume the next element of `positions`.
        let mut next_present = 0usize;
        for (out_col, _) in comids.iter().enumerate() {
            if missing_set.contains(&out_col) {
                continue;
            }
            let div_pos = positions[next_present];
            next_present += 1;
            let local_div = div_pos - min_pos;
            for t in 0..n_steps {
                let raw_idx = local_div * n_steps + t;
                out[(t, out_col)] = raw_f32[raw_idx];
            }
        }

        debug_assert_eq!(
            next_present,
            positions.len(),
            "scatter walked past `positions` — IdIndex::positions_of invariant broken"
        );

        Ok(out)
    }

    /// Read `Qr` daily for `[window_start, window_start + n_days)` and
    /// `comids`. Returns `(n_days, N)` f32 matrix. On hourly-native stores
    /// each day is the mean of its 24 hours (Q' is a rate in m³/s, so the
    /// daily value is the day's average flow — keeps the summed-Q' baseline
    /// meaningful on hourly stores).
    pub fn read_window_daily(
        &self,
        window_start: NaiveDate,
        n_days: usize,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        let start = self.native_start_index(window_start)?;
        match self.resolution {
            crate::data::dates::Frequency::Daily => self.read_slab(start, n_days, comids),
            crate::data::dates::Frequency::Hourly => {
                let hourly = self.read_slab(start, n_days * 24, comids)?;
                Ok(hourly_to_daily_mean(&hourly))
            }
        }
    }

    /// Read `Qr` for `window` and `comids`. Returns `(n_hourly, N)` f32.
    /// Daily stores upsample via repeat-24 + trailing-day trim (unchanged);
    /// hourly stores slice the native axis directly — no upsampling.
    pub fn read_window(&self, window: &RhoWindow, comids: &[Comid]) -> Result<Array2<f32>> {
        match self.resolution {
            crate::data::dates::Frequency::Daily => {
                let daily =
                    self.read_window_daily(window.window_start, window.rho_days, comids)?;
                Ok(daily_to_hourly_trim(&daily, window.n_hourly()))
            }
            crate::data::dates::Frequency::Hourly => {
                let start = self.native_start_index(window.window_start)?;
                self.read_slab(start, window.n_hourly(), comids)
            }
        }
    }

    /// Same as `read_window` but for `TestWindow` — `n_days * 24` hours
    /// (no trailing-day trim) so chunks tile cleanly.
    pub fn read_test_window(
        &self,
        window: &crate::data::TestWindow,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        match self.resolution {
            crate::data::dates::Frequency::Daily => {
                let daily =
                    self.read_window_daily(window.window_start, window.n_days, comids)?;
                Ok(daily_to_hourly_trim(&daily, window.n_hourly()))
            }
            crate::data::dates::Frequency::Hourly => {
                let start = self.native_start_index(window.window_start)?;
                self.read_slab(start, window.n_hourly(), comids)
            }
        }
    }

    /// `units` attribute of the `/Qr` variable, if present. Used by
    /// `ddrs import` to check the m³/s contract.
    pub fn qr_units(&self) -> Option<String> {
        self.qr
            .attributes()
            .get("units")
            .and_then(|v| v.as_str())
            .map(str::to_string)
    }
}
```

Then add next to `daily_to_hourly_trim` (after `:284`):

```rust
/// Collapse a `(n_days * 24, N)` hourly slab to `(n_days, N)` by averaging
/// each 24-hour block. Q' is a rate (m³/s): the daily value is the day's
/// mean flow, so total daily volume is preserved.
pub(crate) fn hourly_to_daily_mean(hourly: &Array2<f32>) -> Array2<f32> {
    let (n_hours, n_div) = hourly.dim();
    debug_assert_eq!(n_hours % 24, 0, "hourly slab length {n_hours} not a multiple of 24");
    let n_days = n_hours / 24;
    let mut daily = Array2::<f32>::zeros((n_days, n_div));
    for d in 0..n_days {
        for j in 0..n_div {
            let mut acc = 0.0f32;
            for h in 0..24 {
                acc += hourly[(d * 24 + h, j)];
            }
            daily[(d, j)] = acc / 24.0;
        }
    }
    daily
}
```

Note: the old `read_window_daily` body moves into `read_slab` with only renames (`n_days`→`n_steps`, `store_start_day`→`start_step`, `end_day`→`end_step`, `daily`→`out`, loop var `d`→`t`) — the window-start validation moves to `native_start_index`. Do not otherwise change the read/scatter logic: the daily path must stay behaviorally identical.

- [ ] **Step 5: Run the new tests and the daily regression tests**

Run: `cargo test --test hourly_streamflow`
Expected: 9 tests PASS.

Run: `cargo test --lib` and `cargo test --test data_zarr_store 2>/dev/null; cargo test streamflow`
Expected: all PASS — in particular the existing real-store tests `streamflow_read_window_returns_expected_shape` and `streamflow_store_open_sees_expected_axes` (daily path unchanged against `merit_dhbv2_UH_retrospective.ic`).

- [ ] **Step 6: Check `n_time` consumers survive the semantics note**

`n_time` now means "native steps" (hours for hourly stores).

Run: `grep -rn "\.n_time" src/ examples/ | grep -v "store/icechunk.rs"`
Expected: only uses on observation stores or daily streamflow contexts. If any call site assumes `StreamflowStore::n_time` is days, fix it to use the new doc'd semantics (divide by 24 on hourly) and note it in the commit message. (As of plan-writing, the baseline and dataset go through `read_window*` only.)

- [ ] **Step 7: Commit**

```bash
git add src/data/store/icechunk.rs tests/hourly_streamflow.rs
git commit -m "feat(data): hourly-native Q' reading in StreamflowStore

Resolution sniffed from CF time units at open. Daily path unchanged
(read_slab is the old read_window_daily body, renames only). Hourly
stores slice the native axis in read_window/read_test_window and
24h-mean in read_window_daily."
```

---

### Task 4: `StreamflowSource::resolution()` + disagg guard in the dataset (TDD)

**Files:**
- Modify: `src/data/store/mod.rs:99-141`
- Modify: `src/data/dataset.rs:332` area (guard + log line)

- [ ] **Step 1: Write the failing tests**

Append to `tests/hourly_streamflow.rs`:

```rust
#[test]
fn streamflow_source_reports_resolution() {
    let daily = StreamflowSource::open(fixture("qr_daily.ic")).expect("open daily");
    assert_eq!(daily.resolution(), Frequency::Daily);
    let hourly = StreamflowSource::open(fixture("qr_hourly.ic")).expect("open hourly");
    assert_eq!(hourly.resolution(), Frequency::Hourly);
}
```

And a unit test for the guard — append inside `mod tests` at the bottom of `src/data/dataset.rs` (create the module if the file has none; check first with `grep -n "mod tests" src/data/dataset.rs`):

```rust
    #[test]
    fn disagg_rejected_on_hourly_native_source() {
        use crate::data::dates::Frequency;
        let p = std::path::Path::new("/mnt/fake/qr_hourly.ic");
        // hourly + disagg block → config contradiction
        let err = validate_disagg_vs_resolution(Frequency::Hourly, true, p).unwrap_err();
        assert!(err.to_string().contains("hourly-native"), "got: {err}");
        // every other combination is fine
        assert!(validate_disagg_vs_resolution(Frequency::Hourly, false, p).is_ok());
        assert!(validate_disagg_vs_resolution(Frequency::Daily, true, p).is_ok());
        assert!(validate_disagg_vs_resolution(Frequency::Daily, false, p).is_ok());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test hourly_streamflow streamflow_source_reports_resolution; cargo test --lib disagg_rejected`
Expected: compile errors — `resolution()` and `validate_disagg_vs_resolution` not defined.

- [ ] **Step 3: Implement**

In `src/data/store/mod.rs`, inside `impl StreamflowSource` (after `open`, `:110`):

```rust
    /// Native time-axis resolution of the underlying store. The global
    /// zarr v2 layout is daily by construction.
    pub fn resolution(&self) -> crate::data::dates::Frequency {
        match self {
            Self::Icechunk(s) => s.resolution,
            Self::GlobalZarr(_) => crate::data::dates::Frequency::Daily,
        }
    }
```

In `src/data/dataset.rs`, add a free function near the other helpers at module level (e.g. directly above `impl MeritGagesDataset` — find it with `grep -n "^impl MeritGagesDataset" src/data/dataset.rs`):

```rust
/// Reject the disaggregation head when the streamflow store is hourly-native:
/// disaggregating an already-hourly signal is a config contradiction, and
/// after the 2026-07-01 stale-binary incident nothing in the forcing path is
/// allowed to silently degrade.
fn validate_disagg_vs_resolution(
    resolution: Frequency,
    has_disagg: bool,
    streamflow_path: &std::path::Path,
) -> Result<()> {
    if resolution == Frequency::Hourly && has_disagg {
        return Err(DataError::Malformed {
            path: streamflow_path.to_path_buf(),
            message: "kan_head.disaggregation is set but the streamflow store is \
                      hourly-native; remove the disaggregation block (an hourly \
                      store needs no daily→hourly head)"
                .into(),
        });
    }
    Ok(())
}
```

(`Frequency` may need adding to the existing `use crate::data::dates::…` import at the top of `dataset.rs` — check with `grep -n "use crate::data::dates" src/data/dataset.rs`.)

Wire it in `MeritGagesDataset::open`, immediately after the streamflow open at `src/data/dataset.rs:332`:

```rust
        let streamflow = Arc::new(StreamflowSource::open(&ds.streamflow)?);
        // The smoke-train self-check line: proves which read path executed.
        eprintln!("streamflow resolution: {:?}", streamflow.resolution());
        validate_disagg_vs_resolution(
            streamflow.resolution(),
            head_cfg.disaggregation.is_some(),
            &ds.streamflow,
        )?;
        let observations = Arc::new(ObservationsStore::open(&ds.observations)?);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test hourly_streamflow; cargo test --lib disagg_rejected`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/store/mod.rs src/data/dataset.rs tests/hourly_streamflow.rs
git commit -m "feat(data): expose Q' resolution; reject disagg head on hourly-native source"
```

---

### Task 5: Store contract doc

**Files:**
- Create: `docs/nh-qprime-store-contract.md`

- [ ] **Step 1: Write the doc**

```markdown
# DDR Q' store contract

The interface between runoff producers (neural-hydrology LSTMs, dHBV2, …)
and ddrs routing. Any store meeting this contract can be validated and
registered with `ddrs import <store> --name <group>` and then routed.

The reference producer is
`~/projects/neuralhydrology/examples/merit_hydro/forward_merit.py`
(`--mode daily|hourly`), which runs a trained NH model over the MERIT unit
catchments and writes a conforming store. Producers that RUN neural
hydrology live in the NH repo; everything downstream of the written store
lives here.

## Contract

- An **icechunk repository** (`main` branch, local filesystem), root group.
- One data variable **`Qr(divide_id, time)`**, dtype **float32**, attr
  `units: m^3/s`.
- `Qr` values are the **local lateral inflow per MERIT unit catchment** —
  no upstream accumulation (routing does that).
- `divide_id`: int64 MERIT COMIDs.
- `time`: int64, CF-encoded as either
  - `days since YYYY-MM-DD[ HH:MM:SS]` — a **daily** store, or
  - `hours since YYYY-MM-DD[ HH:MM:SS]` — an **hourly** store.
  The axis must be contiguous (no gaps); an hourly axis must start at
  hour 0 of a calendar day. Any other units string is rejected at open.
- Values strictly positive: producers floor NaN/negative predictions to
  `1e-6` (as `forward_merit.py::mm_day_to_m3s` does).
- COMIDs **absent** from the store are ddrs's concern, not the producer's:
  reads fill them with `0.001` m³/s, never error.

## How ddrs reads each resolution

| ddrs read | daily store | hourly store |
|---|---|---|
| `read_window` (training) | repeat-24 + trailing-day trim (or disagg head) | native hourly slice |
| `read_test_window` (eval) | repeat-24, `n_days*24` | native hourly slice |
| `read_window_daily` (baseline, disagg input) | direct | mean of each 24-h block |

`kan_head.disaggregation` is **rejected** when the streamflow source is
hourly-native — disaggregating an already-hourly signal is a config
contradiction (`src/data/dataset.rs::validate_disagg_vs_resolution`).

## Conforming stores (2026-07-01)

| Store (`/mnt/ssd1/data/icechunk/`) | resolution | range | divides |
|---|---|---|---|
| `daily_lstm_merit_unit_catchments.ic` | daily | 1981-01-01 → 2020-12-30 | 288,421 |
| `hourly_lstm_merit_unit_catchments.ic` | hourly | 1981-01-01 → 2020-12-31T23 | 197,088 |
| `daily_dhbv2_merit_unit_catchments.ic` | daily | 1980-01-01 → 2020-12-30 | 288,421 |
| `merit_dhbv2_UH_retrospective.ic` | daily | 1980-01-01 → 2020-12-31 | 197,088 |

Note the hourly store starts **1981-01-01** (1980 was LSTM warmup): an
experiment window reaching into 1980 hard-errors rather than clamping.

## Onboarding a new NH dataset

1. In `~/projects/neuralhydrology`, write/adapt a forward script that emits
   a conforming store (start from `forward_merit.py`).
2. `ddrs import <store> --dry-run` — validates the contract + prints a
   COMID-coverage report.
3. `ddrs import <store> --name <group>` — registers it under
   `config/sources/<group>.yaml`.
4. `ddrs sources use <group> && ddrs plan && ddrs run --workflow train`.

Design history: `docs/superpowers/specs/2026-07-01-nh-qprime-import-design.md`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/nh-qprime-store-contract.md
git commit -m "docs: DDR Q' store contract (producer/consumer interface)"
```

---

### Task 6: `ddrs import` module (TDD)

**Files:**
- Create: `src/cli/import.rs`
- Modify: `src/cli/mod.rs:3-17` (register module)
- Modify: `src/cli/sources.rs:52,96,108-130` (make `validate_name` + `extract_block` `pub(crate)`; extract `save_block` from `run_save`)
- Modify: `src/cli/plan.rs:288` (`fn resolve_adjacency` → `pub(crate) fn resolve_adjacency`)
- Create: `tests/import_cmd.rs`

- [ ] **Step 1: Write the failing tests**

Create `tests/import_cmd.rs`:

```rust
//! `ddrs import` behavior against the checked-in fixture stores.

use std::fs;
use std::path::{Path, PathBuf};

use ddrs::cli::import::{run_import, ImportInput};
use ddrs::cli::workspace::Workspace;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Minimal parseable ddrs.yaml (mirrors src/cli/sources.rs test CFG).
const CFG: &str = "\
mode: training
geodataset: merit
seed: 1
np_seed: 1
data_sources:
  attributes: /dev/null/attrs.nc
  conus_adjacency: /dev/null/conus.zarr
  gages_adjacency: /dev/null/gages.zarr
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
";

fn setup() -> (tempfile::TempDir, PathBuf, Workspace) {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tmp.path().join("ddrs.yaml");
    fs::write(&cfg, CFG).unwrap();
    let ws = Workspace::with_root(tmp.path().join(".ddrs"));
    (tmp, cfg, ws)
}

#[test]
fn dry_run_validates_without_writing_a_group() {
    let (_tmp, cfg, ws) = setup();
    run_import(
        Some(&cfg),
        &ws,
        ImportInput {
            store_path: fixture("qr_hourly.ic"),
            name: None,
            dry_run: true,
            force: false,
        },
    )
    .expect("dry-run import of hourly fixture");
    assert!(
        !cfg.parent().unwrap().join("config/sources").exists(),
        "dry-run must not create a group"
    );
}

#[test]
fn import_registers_group_with_swapped_streamflow() {
    let (_tmp, cfg, ws) = setup();
    run_import(
        Some(&cfg),
        &ws,
        ImportInput {
            store_path: fixture("qr_daily.ic"),
            name: Some("test-daily".into()),
            dry_run: false,
            force: false,
        },
    )
    .expect("import daily fixture");

    let group = cfg.parent().unwrap().join("config/sources/test-daily.yaml");
    let text = fs::read_to_string(&group).expect("group file written");
    assert!(text.contains("qr_daily.ic"), "streamflow swapped: {text}");
    assert!(
        text.contains("observations: /dev/null/obs.ic"),
        "other keys carried over from ddrs.yaml: {text}"
    );
    // Registering again without --force refuses; with force succeeds.
    let again = ImportInput {
        store_path: fixture("qr_daily.ic"),
        name: Some("test-daily".into()),
        dry_run: false,
        force: false,
    };
    assert!(run_import(Some(&cfg), &ws, again).is_err());
}

#[test]
fn import_rejects_nonconforming_store() {
    let (_tmp, cfg, ws) = setup();
    let err = run_import(
        Some(&cfg),
        &ws,
        ImportInput {
            store_path: fixture("qr_minutes.ic"),
            name: None,
            dry_run: true,
            force: false,
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("unsupported time units"),
        "got: {err}"
    );
}

#[test]
fn register_without_name_or_dry_run_is_an_error() {
    let (_tmp, cfg, ws) = setup();
    let err = run_import(
        Some(&cfg),
        &ws,
        ImportInput {
            store_path: fixture("qr_daily.ic"),
            name: None,
            dry_run: false,
            force: false,
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("--name"), "got: {err}");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test import_cmd`
Expected: compile error — `ddrs::cli::import` does not exist.

- [ ] **Step 3: Open up the shared helpers**

In `src/cli/sources.rs`:
- `:52` — `fn validate_name` → `pub(crate) fn validate_name`
- `:96` — `fn extract_block` → `pub(crate) fn extract_block`
- Split `run_save` (`:108-130`) so the persistence half is reusable:

```rust
/// Save the current config's `data_sources:` block as group `name`.
pub fn run_save(cfg_path: &Path, name: &str, force: bool) -> Result<PathBuf, CliError> {
    let cfg_text = fs::read_to_string(cfg_path)?;
    let block = extract_block(&cfg_text, cfg_path)?;
    save_block(cfg_path, name, &block, force)
}

/// Persist `block` (a full `data_sources:` block) as group `name`, after
/// validating it deserializes to `DataSources`. Shared by `ddrs sources save`
/// (verbatim block) and `ddrs import` (block with `streamflow:` swapped).
pub(crate) fn save_block(
    cfg_path: &Path,
    name: &str,
    block: &str,
    force: bool,
) -> Result<PathBuf, CliError> {
    validate_name(name)?;
    serde_yaml::from_str::<GroupFile>(block).map_err(|e| CliError::ConfigInvalid {
        path: cfg_path.to_path_buf(),
        source: Box::new(e),
    })?;

    let dest = group_path(cfg_path, name);
    if dest.exists() && !force {
        return Err(CliError::Runtime(format!(
            "group {name:?} already exists at {} — pass --force to overwrite",
            dest.display()
        )));
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&dest, block)?;
    Ok(dest)
}
```

In `src/cli/plan.rs:288`: `fn resolve_adjacency(` → `pub(crate) fn resolve_adjacency(`.

In `src/cli/mod.rs`, add `pub mod import;` to the module list (alphabetical, after `pub mod gc;`).

- [ ] **Step 4: Write `src/cli/import.rs`**

```rust
//! `ddrs import` — validate a Q' store against the DDR store contract and
//! register it as a named data-source group.
//!
//! One command turns a conforming store (see docs/nh-qprime-store-contract.md)
//! into a routable dataset:
//!
//! ```text
//! ddrs import /mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic \
//!     --name hourly-lstm
//! ddrs sources use hourly-lstm && ddrs plan && ddrs run --workflow train
//! ```
//!
//! Validation opens the store through the same `StreamflowSource::open` the
//! training loop uses, so "import succeeded" means "training will read it".
//! The coverage report is best-effort: it needs a resolvable adjacency
//! (explicit paths or a warm `.ddrs/adjacency` cache) and degrades to a
//! warning without one.

use std::fs;
use std::path::{Path, PathBuf};

use crate::cli::sources;
use crate::cli::workspace::Workspace;
use crate::config::{Config, ConfigMode};
use crate::data::dates::Frequency;
use crate::data::store::{ConusAdjacencyStore, StreamflowSource};
use crate::error::CliError;

pub struct ImportInput {
    pub store_path: PathBuf,
    /// Group name to register under `config/sources/`. `None` is only valid
    /// with `dry_run`.
    pub name: Option<String>,
    /// Validate + report only; write nothing.
    pub dry_run: bool,
    /// Overwrite an existing group of the same name.
    pub force: bool,
}

pub fn run_import(
    cfg_path: Option<&Path>,
    ws: &Workspace,
    input: ImportInput,
) -> Result<(), CliError> {
    if input.name.is_none() && !input.dry_run {
        return Err(CliError::Runtime(
            "pass --name <group> to register the store, or --dry-run to \
             validate only"
                .into(),
        ));
    }
    if let Some(name) = &input.name {
        // Fail on a bad name BEFORE the (possibly slow) store open.
        sources::validate_name(name)?;
    }
    if !input.store_path.exists() {
        return Err(CliError::DataSourceMissing {
            path: input.store_path.clone(),
        });
    }

    // ---- 1. Open & detect (same code path the training loop uses) ----
    let source = StreamflowSource::open(&input.store_path)
        .map_err(|e| CliError::Runtime(format!("store failed to open: {e}")))?;

    println!("store       {}", input.store_path.display());
    match &source {
        StreamflowSource::Icechunk(s) => {
            let (res_str, n_days) = match s.resolution {
                Frequency::Daily => ("daily", s.n_time),
                Frequency::Hourly => ("hourly", s.n_time / 24),
            };
            let time_end = s.time_start + chrono::Duration::days(n_days as i64 - 1);
            println!("format      icechunk");
            println!("resolution  {res_str}");
            println!(
                "time        {} .. {} ({} native steps)",
                s.time_start, time_end, s.n_time
            );
            println!("divides     {}", s.index.len());

            // ---- 2. Contract checks ----
            match s.qr_units() {
                Some(u) if u == "m^3/s" => println!("Qr units    m^3/s"),
                Some(u) => println!(
                    "Qr units    WARNING: {u:?} (contract expects \"m^3/s\"; \
                     the solver will treat values as m³/s regardless)"
                ),
                None => println!(
                    "Qr units    WARNING: no units attribute (contract expects \
                     \"m^3/s\")"
                ),
            }
            sample_read(s)?;

            // ---- 3. Coverage report (best-effort) ----
            coverage_report(cfg_path, ws, s);
        }
        StreamflowSource::GlobalZarr(_) => {
            println!("format      global zarr v2 (daily)");
            println!(
                "note        detailed contract validation and coverage are \
                 icechunk-only; open succeeded, which exercises the same \
                 reader the training loop uses"
            );
        }
    }

    // ---- 4. Register ----
    if input.dry_run {
        println!("dry-run     no group written");
        return Ok(());
    }
    let name = input.name.expect("checked at entry");
    let cfg = cfg_path.ok_or_else(|| CliError::ConfigInvalid {
        path: ".".into(),
        source: "no ddrs.yaml found — registration copies its data_sources \
                 block. Run inside a ddrs workspace or pass --config."
            .into(),
    })?;
    let cfg_text = fs::read_to_string(cfg)?;
    let block = sources::extract_block(&cfg_text, cfg)?;
    let swapped = swap_streamflow_line(&block, &input.store_path)?;
    let dest = sources::save_block(cfg, &name, &swapped, input.force)?;
    println!("registered  {}", dest.display());
    println!("activate    ddrs sources use {name}");
    Ok(())
}

/// Read a tiny sample (first 5 divides × up to 3 days) and require finite,
/// positive values — catches unit disasters and all-NaN stores.
fn sample_read(s: &crate::data::store::StreamflowStore) -> Result<(), CliError> {
    let comids: Vec<_> = s.index.ids().iter().take(5).copied().collect();
    let n_days_native = match s.resolution {
        Frequency::Daily => s.n_time,
        Frequency::Hourly => s.n_time / 24,
    };
    let n_days = n_days_native.min(3);
    let q = s
        .read_window_daily(s.time_start, n_days, &comids)
        .map_err(|e| CliError::Runtime(format!("sample read failed: {e}")))?;
    for &v in q.iter() {
        if !v.is_finite() || v <= 0.0 {
            return Err(CliError::Runtime(format!(
                "sample read violates the contract: value {v} (must be \
                 finite and > 0; producers floor to 1e-6)"
            )));
        }
    }
    println!(
        "sample      {} COMIDs × {} days: finite, positive ✓",
        comids.len(),
        n_days
    );
    Ok(())
}

/// Intersect the store's divide_ids with the resolved CONUS adjacency and
/// report coverage. Best-effort: any failure (no config, unreadable
/// adjacency) prints a warning instead of failing the import. NOTE: with a
/// fabric-only config and a cold cache this triggers the managed adjacency
/// build (~10 s CONUS), same as `ddrs plan`.
fn coverage_report(
    cfg_path: Option<&Path>,
    ws: &Workspace,
    s: &crate::data::store::StreamflowStore,
) {
    let Some(cfg_path) = cfg_path else {
        println!("coverage    skipped (no ddrs.yaml — run inside a workspace for a report)");
        return;
    };
    let resolved = Config::from_yaml_file_with_mode(cfg_path, ConfigMode::Training)
        .map_err(|e| e.to_string())
        .and_then(|config| {
            crate::cli::plan::resolve_adjacency(&config, cfg_path, ws)
                .map_err(|e| e.to_string())
        })
        .and_then(|resolved| {
            ConusAdjacencyStore::open(&resolved.conus).map_err(|e| e.to_string())
        });
    match resolved {
        Ok(conus) => {
            let total = conus.order.len();
            let covered = conus.order.iter().filter(|c| s.index.contains(c)).count();
            let pct = 100.0 * covered as f64 / total.max(1) as f64;
            println!(
                "coverage    {covered}/{total} fabric COMIDs ({pct:.1}%); \
                 the rest read as 0.001 m³/s fill"
            );
        }
        Err(e) => println!("coverage    skipped ({e})"),
    }
}

/// Replace the value of the `streamflow:` key inside a `data_sources:` block,
/// preserving indentation and every other line (comments included).
fn swap_streamflow_line(block: &str, store_path: &Path) -> Result<String, CliError> {
    let mut out = String::new();
    let mut swapped = false;
    for line in block.lines() {
        let trimmed = line.trim_start();
        if !swapped && trimmed.starts_with("streamflow:") {
            let indent = &line[..line.len() - trimmed.len()];
            out.push_str(&format!("{indent}streamflow: {}\n", store_path.display()));
            swapped = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !swapped {
        return Err(CliError::Runtime(
            "config's data_sources block has no `streamflow:` key".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_streamflow_preserves_everything_else() {
        let block = "\
data_sources:
  attributes: /a.nc
  # comment stays
  streamflow: /old.ic
  observations: /obs
";
        let out = swap_streamflow_line(block, Path::new("/new/store.ic")).unwrap();
        assert!(out.contains("streamflow: /new/store.ic"));
        assert!(!out.contains("/old.ic"));
        assert!(out.contains("# comment stays"));
        assert!(out.contains("attributes: /a.nc"));
        assert!(out.contains("observations: /obs"));
    }

    #[test]
    fn swap_errors_without_streamflow_key() {
        let err = swap_streamflow_line("data_sources:\n  gages: /g.csv\n", Path::new("/x"))
            .unwrap_err();
        assert!(err.to_string().contains("streamflow"));
    }
}
```

Check the export exists: `grep -n "pub use zarr::" src/data/store/mod.rs` — `ConusAdjacencyStore` is already re-exported (`mod.rs:23`). `StreamflowStore` is re-exported at `mod.rs:21`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --test import_cmd; cargo test --lib swap_streamflow; cargo test --lib sources`
Expected: all PASS (the sources.rs tests confirm `run_save` still behaves after the `save_block` split).

- [ ] **Step 6: Commit**

```bash
git add src/cli/import.rs src/cli/mod.rs src/cli/sources.rs src/cli/plan.rs tests/import_cmd.rs
git commit -m "feat(cli): ddrs import — validate Q' store contract + register source group"
```

---

### Task 7: Wire the `import` subcommand into the binary

**Files:**
- Modify: `src/bin/ddrs.rs:48-119` (Cmd enum), `:150-276` (dispatch)

- [ ] **Step 1: Add the subcommand variant**

In the `Cmd` enum (after `Show`, before `Sources`):

```rust
    /// Validate a Q' store against the DDR store contract
    /// (docs/nh-qprime-store-contract.md) and register it as a data-source
    /// group under config/sources/.
    Import {
        /// Path to the Q' store (icechunk repo or global zarr).
        store: PathBuf,
        /// Group name to register (omit together with --dry-run to validate only).
        #[arg(long)] name: Option<String>,
        /// Validate and report only; don't write a source group.
        #[arg(long)] dry_run: bool,
        /// Overwrite an existing group with the same name.
        #[arg(long)] force: bool,
    },
```

- [ ] **Step 2: Add the dispatch arm**

In `dispatch()`'s match (after the `Cmd::Show` arm):

```rust
        Cmd::Import { store, name, dry_run, force } => {
            ddrs::cli::import::run_import(
                cfg_path.as_deref(),
                &ws,
                ddrs::cli::import::ImportInput {
                    store_path: store,
                    name,
                    dry_run,
                    force,
                },
            )
        }
```

- [ ] **Step 3: Verify it builds and self-documents**

Run: `cargo run --release --bin ddrs -- import --help`
Expected: help text showing `store`, `--name`, `--dry-run`, `--force`.

Run: `cargo run --release --bin ddrs -- import tests/fixtures/qr_hourly.ic --dry-run`
Expected: report with `resolution  hourly`, `divides     4`, `sample … ✓`, `coverage    skipped (…)` (the repo ddrs.yaml, if present, points at real adjacency — either outcome of coverage is acceptable here), `dry-run     no group written`. Exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ddrs.rs
git commit -m "feat(cli): wire ddrs import subcommand"
```

---

### Task 8: Validate + register the real stores

Everything below runs the just-built binary from the working tree — **do not use a stale `~/.cargo/bin/ddrs`** (CLAUDE.md stale-binary trap). Refresh it first.

- [ ] **Step 1: Refresh the installed binary**

```bash
cargo install --path .
```

- [ ] **Step 2: Dry-run all four unit-catchment stores**

```bash
ddrs import /mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic --dry-run
ddrs import /mnt/ssd1/data/icechunk/daily_dhbv2_merit_unit_catchments.ic --dry-run
ddrs import /mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic --dry-run
ddrs import /mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic --dry-run
```

Expected, per store:
- UH retrospective (known-good control): `resolution daily`, `divides 197088`, sample ✓.
- daily dHBV2: `resolution daily`, `divides 288421`, time starting 1980-01-01.
- daily LSTM: `resolution daily`, `divides 288421`, time starting 1981-01-01.
- hourly LSTM: `resolution hourly`, `divides 197088`, `350640 native steps`.
- Coverage line on each (the workspace has a warm adjacency cache) — expect the 288,421-divide stores to cover ~83% of the 346,321-reach CONUS fabric and the 197,088-divide stores proportionally less; any number is fine, the point is the report prints.

The hourly open includes a full contiguity scan of 350,640 time values — expect a few seconds, not minutes. If any store FAILS validation, stop: that's a spec-vs-reality divergence to investigate, not to code around.

- [ ] **Step 3: Register the two LSTM groups**

```bash
ddrs import /mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic --name daily-lstm
ddrs import /mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic --name hourly-lstm
ddrs sources list
```

Expected: both groups listed; `config/sources/daily-lstm.yaml` and `config/sources/hourly-lstm.yaml` exist and differ from `conus.yaml` only in the `streamflow:` line.

- [ ] **Step 4: Commit the groups**

```bash
git add config/sources/daily-lstm.yaml config/sources/hourly-lstm.yaml
git commit -m "config: daily-lstm + hourly-lstm data-source groups (via ddrs import)"
```

---

### Task 9: Smoke trains (the success criterion)

Short trains proving each read path end-to-end: finite loss, directory-style checkpoints, and the `streamflow resolution:` log line. Keep windows SHORT — the summed-Q' baseline that `ddrs plan` computes reads the full eval window, and on the hourly store that's 24× the daily I/O.

- [ ] **Step 1: Back up the current ddrs.yaml**

```bash
cp ddrs.yaml /tmp/ddrs.yaml.pre-nh-smoke
```

- [ ] **Step 2: Daily-LSTM smoke train**

```bash
ddrs sources use daily-lstm
```

Edit `ddrs.yaml`: `mode: training`, `workflow: train`, `experiment.epochs: 1`, and a short in-range window, e.g. `start_time: 1995/10/01`, `end_time: 1996/09/30` (the daily-LSTM store starts **1981-01-01** — any window from 1981 on works; do NOT use 1980). Remove any `kan_head.disaggregation:` block and any `experiment.checkpoint:` left from prior experiments.

```bash
ddrs plan --workflow train
ddrs run --workflow train --max-mini-batches 2 2>&1 | tee /tmp/smoke_daily_lstm.log
grep "streamflow resolution" /tmp/smoke_daily_lstm.log
```

Expected:
- `streamflow resolution: Daily` in the log.
- Finite (non-NaN) loss values for both mini-batches.
- A run dir `.ddrs/runs/<id>/` whose checkpoints are **directories** (`checkpoints/epoch_*_mb_*/head.mpk`) — flat `.mpk` files would mean a stale binary executed.

- [ ] **Step 3: Hourly-LSTM smoke train**

```bash
ddrs sources use hourly-lstm
ddrs plan --workflow train
ddrs run --workflow train --max-mini-batches 2 2>&1 | tee /tmp/smoke_hourly_lstm.log
grep "streamflow resolution" /tmp/smoke_hourly_lstm.log
```

Expected:
- `streamflow resolution: Hourly` — the proof the new path actually ran.
- Finite loss, directory-style checkpoints, no disagg-rejection error (the block was removed in Step 2; if it errors, that's the Task 4 guard working — fix the config, not the guard).
- Note wall-clock vs the daily run in the handoff: `collate` reads the hourly store twice per batch (hourly + 24h-mean daily), so meaningfully slower is expected; hours-per-batch is not, and would justify the chunk-aligned-read follow-up flagged in the spec.

- [ ] **Step 4: Disagg-guard negative test (config level)**

Temporarily add to `ddrs.yaml` under `kan_head:`:

```yaml
  disaggregation:
    use_precip: false
```

```bash
ddrs run --workflow train --max-mini-batches 1; echo "exit: $?"
```

Expected: non-zero exit with the `hourly-native; remove the disaggregation block` message. Then remove the block again.

- [ ] **Step 5: Restore the original config and run the full regression suite**

```bash
cp /tmp/ddrs.yaml.pre-nh-smoke ddrs.yaml
cargo test
cargo run --release --example compare_ddr_sandbox
```

Expected: all tests PASS; sandbox reports **ABSOLUTE MATCH** (nothing in `src/routing/`, `src/geometry.rs`, `src/sparse.rs` was touched, but the invariant demands the check).

- [ ] **Step 6: Commit any smoke-run fallout**

If Steps 2-5 required code fixes, they were committed as they happened. Nothing else to commit here (`.ddrs/` and `ddrs.yaml` are gitignored).

---

### Task 10: Document in CLAUDE.md and close out

**Files:**
- Modify: `CLAUDE.md` (the `### ddrs CLI` section, after the data-source-groups paragraph)

- [ ] **Step 1: Add the import section to CLAUDE.md**

Insert after the `ddrs sources list/save/use` code block:

```markdown
**Importing a Q' store** (`src/cli/import.rs`): any store meeting the DDR Q'
contract (`docs/nh-qprime-store-contract.md` — `Qr(divide_id, time)` f32
m³/s, CF `days since`/`hours since` axis) registers as a source group in one
command:

```bash
ddrs import <store> --dry-run          # validate + coverage report only
ddrs import <store> --name <group>     # validate + register config/sources/<group>.yaml
```

The icechunk reader sniffs daily vs **hourly-native** resolution from the CF
time units (`StreamflowStore.resolution`); hourly stores are sliced natively
(no repeat-24, no disagg — `kan_head.disaggregation` + hourly source is a
config error). `daily-lstm` / `hourly-lstm` groups (NH CudaLSTM / MTS-LSTM
forwards) ship in-repo; the hourly store starts **1981-01-01**, so experiment
windows must not reach into 1980. Dataset open logs
`streamflow resolution: Daily|Hourly` — check it when validating runs.
```

- [ ] **Step 2: Verify the plan's spec coverage one last time**

Re-read `docs/superpowers/specs/2026-07-01-nh-qprime-import-design.md` §§1-5 and confirm: contract doc (Task 5), sniff + three-method behavior + out-of-range hard error + hourly-alignment validation (Tasks 2-3), disagg guard + resolution log (Task 4), import command with dry-run/coverage/registration (Tasks 6-7), all-four-store validation + LSTM smoke trains + daily-path regression (Tasks 8-9).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: ddrs import + hourly-native Q' reading in CLAUDE.md"
```
