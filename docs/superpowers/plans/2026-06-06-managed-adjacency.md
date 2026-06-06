# Managed Adjacency Builds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `conus_adjacency` / `gages_adjacency` optional in `ddrs.yaml`. When absent, `ddrs plan` builds both stores in pure Rust from the MERIT flowlines attribute table (`.dbf`) into a content-keyed cache under `.ddrs/adjacency/<key>/` — the same pattern as the summed-Q' baseline cache. When present, the stores get structural validation with an actionable error instead of zarrs' bare `array metadata is missing`.

**Architecture:** A new `src/adjacency/` module (dbf reader, graph builder, gauge subsetter, zarr-v3 writer, cache) mirroring `ddr_engine`'s MERIT builder but with `length_m`/`slope` baked into the CONUS store (the arrays `ConusAdjacencyStore` requires and stock `ddr_engine` never writes). `ddrs plan` gains an adjacency-resolution step whose output (`resolved paths + cache key`) flows through `PlanResult` into `run` and the per-run manifest.

**Tech Stack:** Rust 2021, existing `zarrs` (gains write usage), `blake3`, `serde_json`; add `dbase` (pure-Rust dBASE/.dbf reader). No Python, no geopandas, no geometry reads.

**Spec:** No standalone spec — root cause + design decisions captured below (investigated 2026-06-06 on wukong).

---

## Root cause (why `ddrs plan` warns today)

```
warning: summed Q' baseline failed: zarr read failed at
/projects/mhpi/tbindas/ddr/data/merit_conus_adjacency.zarr: array metadata is missing
```

1. `ConusAdjacencyStore::open` (`src/data/store/zarr.rs:61-62`) eagerly reads five
   arrays: `order`, `length_m`, `slope`, `indices_0`, `indices_1`.
2. The store on this machine has only `order`, `indices_0`, `indices_1`, `values`.
   `zarrs` reports a missing array node (`length_m/zarr.json` absent) as
   "array metadata is missing".
3. Stock `ddr_engine.merit` (`~/projects/ddr/engine/src/ddr_engine/merit/build.py`)
   never writes `length_m`/`slope`. DDR-Python doesn't need them in the zarr — it
   reads `lengthkm`/`slope` at runtime from the flowlines shapefile
   (`ddr/src/ddr/geodatazoo/merit.py:72-78, 409-417`,
   `cfg.data_sources.geospatial_fabric_gpkg`). The original dev machine's zarr was
   augmented out-of-band; the rebuild on this machine was not.
4. The gauges store is structurally fine — `GagesAdjacencyStore` needs only
   `indices_0/indices_1` + `gage_idx`/`gage_catchment` attrs, all present.

Enabling facts:

- The flowlines `.dbf` attribute table
  (`/projects/mhpi/data/MERIT/raw/continent/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.dbf`,
  108 MB, 346,327 records) carries everything the builders need:
  `COMID, lengthkm, slope, NextDownID, up1–up4`. No geometry required, so the
  545 MB `.shp` is never read.
- The on-disk store's `order` has 346,321 entries vs 346,327 dbf records — the
  6-record delta is `ddr_engine`'s cycle removal. This delta is a parity check
  for the Rust port.
- `.ddrs/sources.lock` fingerprints directories by stat only: `streamflow` and
  `observations` currently have **identical** fingerprints (`b139…`, size 110),
  so the lockfile could never have caught this drift. The adjacency cache key
  must use content fingerprints.

## Decisions (made 2026-06-06 with Tadd)

| Decision | Choice |
|---|---|
| Build mechanism | **Pure-Rust builder** reading the `.dbf` (not shell-out to `ddr_engine`, not augment-only) |
| Config shape | **Optional override**: adjacency keys absent → managed cache; present → validate and use as-is |
| Quick unblock | **None** — first `ddrs plan` with the new code performs the build |

---

## File map

### New files

```
src/adjacency/
├── mod.rs            # pub re-exports; BUILDER_VERSION const (bump → cache invalidation)
├── dbf.rs            # .dbf attribute-table reader via `dbase` → Vec<FlowpathRecord>
├── build.rs          # CONUS COO + topo order + cycle removal + length_m/slope fills
├── gauges.rs         # per-STAID upstream subgraphs (port of subset_upstream)
├── zarr_write.rs     # zarr-v3 writer matching ddr_engine layout + length_m/slope
└── cache.rs          # .ddrs/adjacency/<key>/ lookup/build/manifest (mirrors baseline/cache.rs)

tests/adjacency_build.rs        # unit tests on synthetic networks
tests/adjacency_parity.rs       # ignored-by-default parity vs engine-built stores on disk
```

### Modified files

```
Cargo.toml                      # + dbase
src/config.rs                   # geospatial_fabric; adjacency keys → Option<PathBuf>
src/cli/init.rs                 # lock geospatial_fabric; adjacency only when configured
src/cli/lockfile.rs             # optional-source handling in diff
src/cli/plan.rs                 # adjacency resolution step (replaces step-6 stub)
src/cli/plan_bootstrap.rs       # template fallback keys
src/cli/run.rs                  # consume resolved paths; manifest records them + cache key
src/cli/manifest.rs             # manifest schema: resolved_adjacency { conus, gages, cache_key? }
src/data/store/zarr.rs          # (no behavior change) cite new builder as the writer
config/merit_training.yaml      # drop adjacency paths; add geospatial_fabric
CLAUDE.md                       # workspace-layout + data-sources tables, plan side-effect note
README.md                       # getting-started: adjacency now managed
```

---

## Task 1: Config & lockfile

- [x] `src/config.rs`: add `data_sources.geospatial_fabric: Option<PathBuf>`; change
      `conus_adjacency`/`gages_adjacency` to `Option<PathBuf>`.
- [x] Load-time validation rule: *(both adjacency keys present)* **or**
      *(geospatial_fabric present)*; otherwise a `ConfigInvalid` naming the missing keys.
      Partial adjacency (one of two) is an error.
- [x] `src/cli/init.rs`: fingerprint+lock `geospatial_fabric` when configured; lock
      adjacency keys only when explicitly configured. `init` smoke flow unchanged.
- [x] `src/cli/lockfile.rs`: `diff_against_live` tolerates sources present in only
      one side when the key is optional (report as drift, not error).
- [x] `config/merit_training.yaml` + `src/cli/plan_bootstrap.rs` template: remove the
      two adjacency paths, add
      `geospatial_fabric: /projects/mhpi/data/MERIT/raw/continent/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp`.
      Note: DDR's config key is `geospatial_fabric_gpkg`; we use `geospatial_fabric`
      since we accept `.shp` (and read only the sibling `.dbf`). Document the mapping
      in a comment.
- [x] Unit tests for the config validation matrix (both/neither/partial keys).

## Task 2: dbf reader

- [x] Add `dbase` to `Cargo.toml`.
- [x] `src/adjacency/dbf.rs`: read the `.dbf` sibling of the configured `.shp`
      (or the `.dbf` directly if configured as such). Extract
      `FlowpathRecord { comid: i64, lengthkm: f64, slope: f64, next_down_id: i64, up: [i64; 4] }`.
      Missing/null numeric → NaN (matches geopandas semantics so fills match DDR).
- [x] `DataError`-style error with `PathBuf` context per repo convention.
- [x] Test: header-only fixture or synthetic dbf; assert column extraction + NaN handling.

## Task 3: CONUS builder

- [x] `src/adjacency/build.rs`: port `ddr_engine/merit/build.py::create_adjacency_matrix`.
      Cite engine line numbers in comments (repo convention):
      - upstream dict from `up1–up4` (`build_upstream_dict`),
      - Kahn topological sort,
      - on cycle: collect all COMIDs on simple cycles, drop them, recurse
        (mirrors `build.py`'s `rx.DAGHasCycle` branch; expect 6 drops on pfaf_7),
      - isolated COMIDs appended **sorted** after the connected order,
      - lower-triangular COO (`row = downstream idx`, `col = upstream idx`),
        single-successor assert ("not dendritic" otherwise),
      - final assert `rows[k] >= cols[k]` (routing invariant 3).
- [x] `length_m` / `slope` (the actual bug fix): `lengthkm × 1000.0`; NaN/inf →
      column mean over finite values, mirroring `merit.py:78` (`naninfmean`) and
      `merit.py:409-417` (`fill_nans`). f32 output, aligned to `order`.
- [x] Unit tests on synthetic networks: chain, confluence, cycle removal,
      isolated reaches, NaN fills, triangularity.

## Task 4: gauge subgraph builder

- [x] `src/adjacency/gauges.rs`: port `build_gauge_adjacencies`/`subset_upstream`.
      Read the gages CSV (already a locked source) mirroring DDR's `MERITGauge`
      validation for the STAID → outlet-COMID mapping. Per STAID: BFS upstream
      over the CONUS graph, emit COO in **CONUS position space** (matching
      `GageSubgraph`'s contract in `src/data/store/zarr.rs:102-116`), with
      `gage_idx` (outlet's CONUS position) and `gage_catchment` attrs.
- [x] STAIDs whose catchment COMID is missing from the network: skip with a
      warning (mirrors `GagesAdjacencyStore::open`'s silent-drop semantics).
- [x] Unit test: synthetic 6-reach network, 2 gauges; assert subgraph node sets
      and `gage_idx`.

## Task 5: zarr writer

- [x] `src/adjacency/zarr_write.rs`: write zarr-v3 groups byte-compatible with
      `ddr_engine/core/zarr_io.py`'s layout — root group attrs
      (`format: "COO"`, `shape`, `geodataset: "merit"`, `data_types`), arrays
      `indices_0`/`indices_1` (int32), `values` (uint8 ones), `order` (int32),
      bytes + zstd codecs, comparable chunking — **plus** `length_m`/`slope`
      (float32) on the CONUS store.
- [x] Gauges store: one subgroup per STAID with the same array set + the two attrs.
- [x] Round-trip test: write a tiny store, reopen with `ConusAdjacencyStore::open`
      and `GagesAdjacencyStore::open` — the readers are the compatibility oracle.

## Task 6: adjacency cache

- [x] `src/adjacency/cache.rs`, mirroring `src/baseline/cache.rs`:
      layout `.ddrs/adjacency/<key>/{merit_conus_adjacency.zarr, merit_gages_conus_adjacency.zarr, manifest.json}`.
- [x] `key = blake3(dbf content-fp ∥ gages-csv content-fp ∥ BUILDER_VERSION)[..16]`.
      **Content** fingerprints (blake3 of file bytes), not paths/stat — see the
      lockfile blind spot above. `BUILDER_VERSION` const in `adjacency/mod.rs`;
      bump on any algorithm change to invalidate caches.
- [x] `manifest.json`: input paths + fingerprints, n / nnz / n_gauges /
      cycle-dropped COMIDs, build duration, ddrs git SHA.
- [x] Build into a temp dir, rename into place (crash-safe, same as baseline cache).

## Task 7: plan resolution + run integration

- [x] `src/cli/plan.rs`: replace the step-6 validation stub:
      - explicit adjacency paths → open both stores, verify every required array
        exists up front; on failure name the missing array and suggest either
        removing the keys (managed build) or repairing the store;
      - keys absent → cache lookup by key; hit → reuse; miss → build with a
        `log`-style progress line ("building MERIT adjacency from <dbf> — first
        run takes ~1–2 min"). Same side-effectful-plan precedent as the Q' baseline.
- [x] `PlanResult` gains `resolved_adjacency { conus: PathBuf, gages: PathBuf, cache_key: Option<String>, cache_hit: Option<bool> }`;
      plan's human output prints it.
- [x] Baseline + dataset paths consume the **resolved** paths (the baseline cache
      key in `src/baseline/cache.rs` hashes these path strings — resolved cache
      paths flow in automatically; a rebuild under a new key correctly invalidates
      the baseline).
- [x] `src/cli/run.rs` + `manifest.rs`: run manifest records resolved paths +
      adjacency cache key; `ddrs show` displays them.
- [x] `ddrs status`/`gc`: report `.ddrs/adjacency/` disk usage; `gc` leaves
      adjacency caches alone in v1 (document; key-based GC is a follow-up).

## Task 8: parity tests & docs

- [x] `tests/adjacency_parity.rs` (`#[ignore]` by default, like
      `conus_adjacency_loads_real_merit_zarr`): build from the real pfaf_7 `.dbf`,
      compare `order`, `indices_0`, `indices_1` element-for-element against the
      engine-built store at the locked path; assert exactly the engine's
      cycle-removal delta; spot-check `length_m`/`slope` against
      hand-computed fills for a few COMIDs. Sample ~10 STAIDs against the
      engine-built gauges store.
- [x] `tests/data_zarr_store.rs`: keep, but point at a resolved/managed store once
      available (or leave targeting the explicit-override path; decide in-task).
- [x] CLAUDE.md: data-sources table (adjacency rows → "managed, see
      `.ddrs/adjacency/`"; add geospatial_fabric row), workspace-layout table
      (+ `.ddrs/adjacency/<key>/`), note that `ddrs plan` may also build adjacency
      on first run.
- [x] README getting-started: reflect the smaller required `data_sources` set.

---

## Verification

1. `cargo test` (unit + round-trip).
2. `cargo test --test adjacency_parity -- --ignored` on wukong — the engine-built
   stores on disk are the oracle.
3. `rm ddrs.yaml && ddrs --config config/merit_training.yaml init` then
   `ddrs --config config/merit_training.yaml plan --workflow train-and-test`:
   first run builds + computes baseline; second run is a double cache hit and
   instant. (Use the `--config` form to dodge the bootstrap-from-last-run gotcha,
   `src/cli/plan_bootstrap.rs:58-63`.)
4. `cargo run --release --example compare_ddr_sandbox` → ABSOLUTE MATCH must hold
   (no routing-core changes expected; run anyway per invariant 1).

## Out of scope / follow-ups

- Fixing stat-only directory fingerprints in `sources.lock` generally
  (the streamflow/observations collision) — separate change.
- `ddrs gc` pruning of stale adjacency cache keys.
- icechunk/netcdf metadata validation in plan (the rest of the old step-6 stub).

## Working-tree note

The tree currently carries uncommitted cuSPARSE/SP-10 changes. Implement this
plan on a dedicated branch (e.g. `managed-adjacency`) off `master`; don't mix.
