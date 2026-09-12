# Handoff: global data sources + group-scoped caches (2026-06-11)

**Branch / PR:** `global-observations-reader` → PR #18 (tip `ff2e032` at
handoff time). All work below is committed and pushed unless marked
IN FLIGHT. Lib suite: 145 passed; the only failing integration test is
`data_static::attributes_store_opens_against_conus_subset`, which is
pre-existing and environmental (incomplete zarr copy under
`~/projects/ddr/data/` on this cluster — not caused by this branch).

## What works now (verified end-to-end on wukong)

`ddrs sources use global && ddrs plan --workflow train-and-test` succeeds:

- Global adjacency builds from `global_merit_riv.gpkg` in ~22 s →
  `.ddrs/adjacency/675e6db5bbba787d/global_merit_riv_adjacency.zarr`
  (+ `_gages_`). Deterministic: full rebuild reproduced identical
  baseline metrics.
- Summed-Q′ baseline: **5,912 gauges × 260,565 upstream divides × 5,479
  days → median NSE 0.639, KGE 0.723** (cache `e11608de7d716c4f`).
  This is the bar a global-trained KAN must beat.
- `ddrs run --workflow train-and-test` has **NOT** been attempted yet.
  First run at 2.9M-reach scale; watch batch-assembly time and GPU memory.

## What landed on this branch

1. **Global observations reader** (`src/data/store/zarr_obs.rs`):
   dMC_global_v3.1 zarr-v2 group, 6,051 `Provider__GageId` arrays, daily
   m³/s, NaN=missing, implicit axis 1980-01-01→2020-12-31 (14,976 d) —
   verified against USGS NWIS. Dispatch enum `ObservationsStore`
   (sniffs `.zgroup`).
2. **Global Q′ streamflow reader** (`src/data/store/zarr_qprime.rs`):
   `merit_global_v2.7` multi-zone zarr v2 (60 pfaf-2 zones, 2,897,147
   COMIDs), `streamflow` is time-major `(time, COMID)` f64. Units
   verified m³/s two ways (CONUS reference match + summed-upstream vs
   NWIS obs). Dispatch enum `StreamflowSource`. Missing COMIDs → 0.001
   fill (~42k fabric reaches lack predictions).
3. **Gage CSV directory support + v3.1 quirks** (`gage_csv.rs`,
   `dataset.rs`, `summed_q_prime.rs`): `gages:` may be a directory of
   per-zone CSVs (sorted, concatenated, deduped by STAID — v3.1 lists
   most gauges 4× with float-format drift; 23,491 rows → 5,975 gauges).
   No DA_VALID column → rows pass (explicit False still drops). Gauges
   absent from the obs store (63) are drop-and-warn in both dataset
   (Filter 4) and baseline, never hard errors.
4. **`ddrs sources` save/use/list** (`src/cli/sources.rs`): named
   data-source groups in `config/sources/<name>.yaml` (tracked; `conus`
   + `global` seeded). Textual splice preserves comments; `use`
   validates the spliced config then re-locks `sources.lock` via
   `init::lock_sources_from_config`.
5. **Adjacency cache fixes** (`src/adjacency/cache.rs`):
   - gages fingerprint handles directories (sorted `*.csv`, name+NUL+bytes);
   - store names carry the fabric stem (`<stem>_adjacency.zarr`,
     `<stem>_gages_adjacency.zarr`) with fallback scan for legacy
     `merit_*conus*` names — old caches still hit;
   - manifest-without-stores (hand-pruned cache) = stale → cleared and
     rebuilt instead of a phantom hit.

## IN FLIGHT: group-scoped cache layout (user request, not started in code)

User wants `.ddrs/adjacency/` and `.ddrs/baselines/` namespaced by
data-source group: `<root>/{adjacency,baselines}/<group>/<key>/`.
Design agreed with the user so far:

- **Group detection**: structural match of the config's current
  `data_sources` block against `config/sources/*.yaml` — exactly what
  `sources::run_list` already does. Extract a
  `sources::active_group(cfg_path) -> Option<String>` helper. No match →
  `None` → keep today's flat layout (legacy behavior, no churn).
- **Plumbing**:
  - `adjacency::cache::resolve_or_build(...)` gains a `group:
    Option<&str>`; cache dir becomes
    `<root>/adjacency/<group>/<key>` when set. Called from
    `cli/plan.rs::resolve_adjacency` (line ~247).
  - `baseline::cache_dir(root, key)`, `load_cached`, the store fn, and
    `compute_or_load_cached(test_cfg, root)` gain the same param.
    Callers: `cli/plan.rs::compute_baseline` (~267) and
    `cli/run.rs::copy_baseline_into_run_dir` (~470).
  - Compute the group ONCE at the top of plan/run from `config_path`
    (the on-disk yaml, NOT the resolved-adjacency-mutated config — group
    files contain `geospatial_fabric`, not resolved paths) and pass down.
- **Migration**: on group-aware lookup miss, check the legacy flat path
  (`<root>/adjacency/<key>`, `<root>/baselines/<key>`); if it's a
  complete cache, `fs::rename` it into the group dir (same filesystem,
  cheap) rather than rebuilding. The user's current caches are flat:
  adjacency `675e6db5bbba787d`, baseline `e11608de7d716c4f`.
- **Watch out**: a group name that looks like a 16-hex key could collide
  with a flat key dir; acceptable risk, maybe reject 16-hex group names
  in `sources::validate_name`.
- Update the CLAUDE.md workspace-layout table and add cache.rs tests
  (group layout + flat→group migration) when done. Commit to PR #18.

## Gotchas for the next agent

- `gh` CLI is not on PATH on wukong; `~/.config/gh/hosts.yml` has a
  valid token — use the REST API via python requests (see PR #18
  creation/update pattern in this branch's history).
- Use `uv run --with "numpy,zarr==2.18.3,numcodecs==0.13.1" --python 3.11`
  for python-side zarr v2 inspection (system python has no numpy; latest
  numcodecs is incompatible with zarr 2.18).
- The locked sibling dirs under `/gpfs/hjj5218/data/dmc_forcing/` are now
  readable. `statistics/`, `soil_properity/`, `zarr/` are unused by ddrs.
- Baseline cache keys hash the *resolved adjacency paths* — renaming
  adjacency stores invalidates baselines by design.
- Attributes for global runs: `merit_global_attributes_v2.nc` is already
  global (2,939,404 COMIDs, zones 11–91); stats JSON lives at
  `~/projects/ddr/data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json`.
- The desktop `~/projects/ddr` holds unpushed DDR reference state (see
  CLAUDE.md invariant 1 caveat) — don't regenerate CONUS fixtures from a
  clean clone.
