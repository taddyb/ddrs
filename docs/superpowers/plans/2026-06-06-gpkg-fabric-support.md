# GeoPackage Fabric Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `data_sources.geospatial_fabric` accept a `.gpkg` (GeoPackage) path in addition to `.shp`/`.dbf`, so the managed adjacency build can ingest the merged global MERIT flowlines at `/projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg` (2,939,408 reaches, layer `flowlines`) and unlock global training.

**Architecture:** GeoPackage is SQLite. The fabric pipeline funnels everything through `Vec<FlowpathRecord>` (`src/adjacency/dbf.rs` → `build` → `gauges` → `zarr_write` → `cache`), so the entire change is one new reader that produces the same `Vec<FlowpathRecord>` via `SELECT` over the attribute columns — geometry blobs are never deserialized. A thin extension-dispatch wrapper (`read_fabric_records`) routes `.gpkg` → the new reader, everything else → the existing dbf reader. The builder, toposort, zarr writer, and cache logic are untouched (toposort is already iterative — safe at 2.9M nodes; indices are `i32` — 2.94M ≪ `i32::MAX`).

**Tech Stack:** add `rusqlite` with the `bundled` feature (compiles SQLite in; no system lib, consistent with the repo's zero-system-deps posture). No GDAL, no geozero — attributes only.

**Closes the DDR gap:** DDR's config key is `geospatial_fabric_gpkg` — DDR already reads GeoPackage. This makes ddrs accept the same artifact (`src/config.rs:58-63` documents the divergence today).

---

## File map

### New files

```
src/adjacency/gpkg.rs        # NEW: rusqlite-based FlowpathRecord reader for .gpkg
src/adjacency/fabric.rs      # NEW: read_fabric_records() extension dispatch + FabricKind
tests/gpkg_fabric.rs         # NEW: synthetic-gpkg integration tests + dbf↔gpkg parity
```

### Modified files

```
Cargo.toml                   # + rusqlite = { version = "0.32", features = ["bundled"] }
src/adjacency/mod.rs         # + pub mod gpkg; pub mod fabric; pipeline diagram update
src/adjacency/cache.rs       # fingerprint + read via fabric dispatch; manifest field rename
src/config.rs                # geospatial_fabric docs; optional geospatial_fabric_layer key
src/cli/init.rs              # lockfile note only (path hashing is already format-agnostic)
CLAUDE.md                    # data-sources table + managed-build prose mention .gpkg
README.md                    # getting-started fabric wording
```

---

## Task 1: gpkg reader module

**Files:** `Cargo.toml`, `src/adjacency/gpkg.rs`, `src/adjacency/mod.rs`

- [x] Add `rusqlite = { version = "0.32", features = ["bundled"] }` to `Cargo.toml`.
- [x] `gpkg.rs`: `pub fn read_flowpath_records_gpkg(path: &Path, layer: Option<&str>) -> Result<Vec<FlowpathRecord>>`.
  - Open read-only (`OpenFlags::SQLITE_OPEN_READ_ONLY`).
  - Layer discovery: `SELECT table_name FROM gpkg_contents WHERE data_type = 'features'`.
    - `layer: Some(name)` → verify present, else `DataError::Malformed` listing available layers.
    - `layer: None` + exactly one feature layer → use it.
    - `layer: None` + multiple layers → error listing layers and pointing at the
      `geospatial_fabric_layer` config key (Task 3).
  - Read: `SELECT COMID, lengthkm, slope, NextDownID, up1, up2, up3, up4 FROM "<layer>" ORDER BY ROWID`.
    - **`ORDER BY ROWID` is load-bearing**: record order feeds graph insertion order feeds
      the deterministic DFS toposort (`build.rs::topological_sort`). The content-addressed
      cache requires byte-identical rebuilds, so row order must be deterministic.
  - Null semantics — mirror `dbf.rs` exactly (its module docs are the contract):
    - float columns (`lengthkm`, `slope`): SQL NULL → `f64::NAN`.
    - integer columns (`COMID`, `NextDownID`, `up1..4`): SQL NULL → `DataError::Malformed`
      with row + column context.
  - SQLite dynamic typing: accept `INTEGER` or `REAL` storage for the int columns
    (`value as i64` on REAL, mirroring `dbf.rs`'s `*v as i64`), error on TEXT/BLOB.
  - Every error variant carries the `PathBuf` (repo convention: `DataError` keeps source context).
- [x] Wire `pub mod gpkg;` into `src/adjacency/mod.rs`; update the pipeline diagram comment
  (`.shp/.dbf | .gpkg ──► fabric::read_fabric_records()`).
- [x] Unit tests in-module: build a synthetic gpkg with raw rusqlite (create
  `gpkg_contents` + a feature table; no GDAL needed), covering: happy path, NULL float → NaN,
  NULL COMID → error, multi-layer error message, explicit-layer selection, REAL-typed ints.

## Task 2: extension dispatch + cache integration

**Files:** `src/adjacency/fabric.rs`, `src/adjacency/cache.rs`, `src/adjacency/mod.rs`

- [x] `fabric.rs`:
  ```rust
  pub enum FabricKind { Dbf(PathBuf), Gpkg(PathBuf) }   // resolved, hashable path
  pub fn resolve_fabric(path: &Path) -> FabricKind       // .shp→sibling .dbf (existing rule); .gpkg→as-is; .dbf→as-is
  pub fn read_fabric_records(path: &Path, layer: Option<&str>) -> Result<Vec<FlowpathRecord>>
  ```
  Unknown extension → error naming the three accepted forms.
- [x] `cache.rs`: replace the two `resolve_dbf_path` + `read_flowpath_records` call sites
  (`cache.rs:113`, `cache.rs:146`) with the fabric dispatch. The blake3 content hash runs over
  the resolved file's bytes — for gpkg that is the whole `.gpkg`.
  - Note in a comment: hashing the 6.1 GB global gpkg costs a few seconds on first `plan`;
    cached thereafter (same fingerprint→key flow as today).
- [x] Manifest: rename `dbf_path` → `fabric_resolved_path` with
  `#[serde(alias = "dbf_path")]` so existing cached manifests still parse. The cache *key*
  derives from content bytes, not the manifest, so no cache invalidation and **no
  `BUILDER_VERSION` bump** (algorithm unchanged; a gpkg input hashes to a fresh key anyway).
- [x] Keep `dbf::resolve_dbf_path` (it has direct tests) but have `fabric.rs` own dispatch.

## Task 3: config key for layer selection

**Files:** `src/config.rs`, `src/cli/init.rs`, `src/cli/plan.rs`

- [x] Add optional `data_sources.geospatial_fabric_layer: Option<String>` (serde default
  `None`). Thread it through `plan.rs`'s managed-build call into `cache::resolve_or_build`
  → `read_fabric_records`.
- [x] Update `geospatial_fabric` doc comment (`config.rs:58-63`): now accepts `.shp`
  (sibling `.dbf` read), `.dbf`, or `.gpkg` (attributes via SQL; geometry never read) —
  and note this closes the gap with DDR's `geospatial_fabric_gpkg`.
- [x] Validation: `geospatial_fabric_layer` set while fabric is not `.gpkg` → config error
  at load time (same fail-fast style as the mode/workflow agreement check).
- [x] Lockfile/init: no code change needed — `init.rs:140` fingerprints the configured
  path's bytes regardless of format (verified by inspection; config-level gpkg
  acceptance covered by `gpkg_fabric_with_layer_valid` in `src/config.rs` tests).

## Task 4: tests — parity, scale smoke, real data

**Files:** `tests/gpkg_fabric.rs`

- [x] **dbf↔gpkg parity (the keystone test):** write the same synthetic records to a `.dbf`
  (via `dbase::TableWriterBuilder`, reusing `dbf.rs`'s test helper pattern) and a `.gpkg`
  (via rusqlite); assert `read_fabric_records` returns element-identical
  `Vec<FlowpathRecord>`, then run both through `build_conus_adjacency` and assert
  byte-identical `order`/`indices_0`/`indices_1` — same bar as `tests/adjacency_parity.rs`.
- [x] Determinism: read the synthetic gpkg twice → identical output (guards the
  `ORDER BY ROWID` contract).
- [x] Real-data smoke, `#[ignore]` (runs on wukong only):
  open `/projects/mhpi/data/MERIT/raw/global_merit_riv.gpkg`, assert layer `flowlines`
  found, 2,939,408 records, all COMIDs unique, and `build_conus_adjacency` completes.
  Record wall time in the test output (expect ~1–3 min at 8.5× CONUS's ~10 s).
- [x] Run the standing gates (2026-06-06, wukong): `cargo test` — all green except
  `data_static::attributes_store_opens_against_conus_subset`, which fails on a
  hardcoded desktop path with incomplete zarr metadata on this host (pre-existing
  environmental failure, untouched by this change). `compare_ddr_sandbox` reports
  ~1% rel diff — the documented wrong-reference-fixture signature on wukong
  (CLAUDE.md invariant 1 caveat; identical at every ddrs commit on this host), not
  a regression: this change touches no routing/geometry/sparse code. Re-verify
  ABSOLUTE MATCH on the desktop before merging.

## Task 5: docs

**Files:** `CLAUDE.md`, `README.md`

- [x] CLAUDE.md data-sources table: fabric row mentions `.shp`/`.dbf`/`.gpkg`; managed-build
  prose: "Only the attribute table is read — `.shp` geometry (or gpkg geometry blobs) is
  never opened." Update the ~10 s build estimate with the global figure once measured.
- [x] README getting-started: one line that a merged global gpkg works as
  `geospatial_fabric`, with `geospatial_fabric_layer` for multi-layer files.

---

## Out of scope here, but required for global *training* (verify before launching)

gpkg ingestion makes the global **network** buildable. A global training run additionally
needs the other `data_sources` to cover the new domain — none of which this plan changes:

| Source | Current value | Global status |
|---|---|---|
| `attributes` | `merit_global_attributes_v2.nc` | already global ✅ — but verify all 2.94 M COMIDs join |
| `gages` | `gages_3000.csv` (CONUS USGS) | ❌ needs a global gauge list (e.g. GRDC) |
| `observations` | USGS daily icechunk | ❌ USGS is CONUS-only; global obs source needed |
| `streamflow` (Qr forcing) | `merit_dhbv2_UH_retrospective` | ⚠️ verify COMID coverage outside CONUS |

Scale notes for when those land: the CONUS store name (`ConusAdjacencyStore`, `conus_*`
keys) is cosmetic at global scale — leave as-is; `i32` zarr indices hold to 2.1 B reaches;
training cost scales with the *per-batch gauge-subgraph union*, not total network size, so
global N mainly grows `ddrs plan` build time and the eval all-gauges union. The summed-Q′
baseline cache key already includes the adjacency paths, so a global fabric produces a
fresh baseline automatically.
