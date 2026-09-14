# Handoff: SIGFPE debug for `tests/device_selection.rs` — run on wukong

> **RESOLVED 2026-06-06 (wukong session).** Root cause + fix in commit
> `2bba409`; full findings at the bottom of this doc (§Outcome). The parity
> gate and end-to-end items are BLOCKED by wukong data/DDR-repo state, not by
> code — see §Outcome before doing anything else.

**Date:** 2026-06-06
**Supersedes:** the SIGFPE section of `2026-06-06-gpu-device-config-handoff.md`
(everything else in that doc still stands).
**Where to run:** `tbindas@wukong.ua.edu`, repo at `/projects/mhpi/tbindas/ddrs`
(at commit 5fa8fb6, clean tree, build cache warm from the prior session).
**Why this doc exists:** the desktop session (Arch, 1× RTX 4080) could NOT
reproduce the crash — the SIGFPE is specific to the wukong environment.

## New evidence from the desktop session (changes the diagnosis)

1. **The crash does NOT reproduce on a 1-GPU host.** On the desktop,
   `device_selection` built and ran clean: the raw
   `cudarc::driver::result::init()` + `device::get_count()` gate executed
   fine (found 1 device), printed the skip line, exit 0. So the gate is not
   *inherently* broken — hypothesis 1 from the original handoff is weakened
   but not dead (driver/CUDA versions differ on wukong).
2. **`mmc` test binary ran clean on the desktop** (same statically-linked
   hdf5/netcdf via the new `netcdf = { features = ["static"] }`). But this
   discriminating experiment has NOT yet been run on wukong — and the static
   link landed in the SAME commit as the device work, so a startup-ctor crash
   from the vendored hdf5/netcdf-c **on wukong's toolchain** is still live.
3. `/tmp/sigfpe_probe.rs` from the original handoff no longer exists. Ignore
   that cleanup item.
4. Grep of `src/cuda_graph/scratch.rs` found **no integer `/` or `%`** (the
   matches are comments) — original hypothesis 3 has no obvious candidate
   there.
5. Desktop note (not for wukong): cmake was missing locally and is now
   installed; a debug worktree exists at `.claude/worktrees/sigfpe-debug`
   (branch `sigfpe-debug`, at 5fa8fb6, gitignored fixtures symlinked from the
   main checkout). The desktop main checkout is being mutated by a separate
   managed-adjacency session — do not debug there.

## Gotcha: non-interactive shells on wukong lack cargo

```bash
export PATH=$HOME/.cargo/bin:$PATH
cd /projects/mhpi/tbindas/ddrs
```

## Step-by-step (do these IN ORDER, stop at first divergence)

### Step 1 — discriminating experiment: is it process startup or the test fn?

```bash
cargo test --test mmc 2>&1 | tail -5
```

- **`mmc` also SIGFPEs** → crash is at process startup (static-library
  constructor — vendored hdf5/netcdf-c/zlib built on wukong's gcc, or a CUDA
  lib ctor). The device test is innocent. Go to Step 2 to identify the
  module, and consider rebuilding the netcdf-src/hdf5 build dirs
  (`cargo clean -p netcdf-src -p hdf5-metno-src` then rebuild) or testing
  with the `static` feature temporarily removed to confirm.
- **`mmc` passes** → crash is inside `device_selection` itself. Go to Step 2,
  then Step 4.

### Step 2 — name the faulting module (no gdb needed)

`gdb` is broken on wukong (`libssl.so.1.1` missing). The kernel already logs
every SIGFPE with the instruction pointer and module:

```bash
cargo test --test device_selection --no-run 2>&1 | tail -2   # prints binary path
./target/debug/deps/device_selection-<hash> --nocapture; echo "EXIT: $?"
dmesg | tail -20            # or: journalctl -k --since -5min | grep -iE "trap|fpe"
```

Look for a line like
`traps: device_selection[12345] trap divide error ip:7f... sp:... in libfoo.so[...]`
— that names the faulting library/binary instantly. If it's in the test
binary itself, symbolize with
`addr2line -e ./target/debug/deps/device_selection-<hash> <ip-offset>`.

Also note from the run output: **did the libtest header (`running 1 test`)
print?**
- Header printed → crash is inside the test fn (gate or smoke stages).
- No header at all → process-startup ctor crash (static lib), regardless of
  what Step 1 showed.

Check for cores too: `coredumpctl info device_selection` (if systemd-coredump
exists on wukong) or `ulimit -c unlimited` + rerun + `eu-stack --core core`.

### Step 3 — if the crash is in a CUDA library

Record driver/toolkit (`nvidia-smi | head -4`, `nvcc --version`) in this doc.
The desktop ran driver-side `cuInit` fine; a wukong-specific
driver-vs-`cuda-version-from-build-system` mismatch in cudarc 0.19 would show
up here. Try the minimal probe alone:

```bash
cat > /tmp/probe.rs <<'EOF'
fn main() {
    eprintln!("calling init");
    cudarc::driver::result::init().unwrap();
    eprintln!("init ok; calling get_count");
    let n = cudarc::driver::result::device::get_count().unwrap();
    eprintln!("count = {n}");
}
EOF
# easiest: drop it in examples/ as probe.rs and `cargo run --example probe`
```

### Step 4 — if the crash is in the test fn: swap the gate, then bisect

1. Replace the raw-cudarc gate in `tests/device_selection.rs:23-30` with the
   repo-standard pattern (`tests/cusparse_ptr_spike.rs:14-22`):
   `catch_unwind(|| { let _d: Dev = Dev::new(1); })` — probing device 1
   directly both gates on ≥2 devices and avoids raw `cuInit`. Rerun.
2. Still crashing → comment out stages bottom-up (stage 3, then 2, then 1) to
   find which smoke stage faults. A fault in stage 1 (CPU sparse solver on
   dev 1) = real multi-device bug in src/, likely integer sizing math; in
   stage 2/3 = cusparse cache / graph-capture context binding
   (`src/sparse/cusparse.rs`, `src/routing/mmc_op.rs`).

## After the SIGFPE is fixed — remaining validation (from original handoff)

- [ ] `cargo test --test device_selection` green on wukong, actually running
      (not skipping — 8 GPUs there).
- [ ] Full `cargo test` green. **Read the output; don't trust exit codes**
      (a piped `tail` already produced a misleading exit 0 twice in this
      effort).
- [ ] Parity gate (CLAUDE.md invariant 1):
      `cargo run --release --example compare_ddr_sandbox` → ABSOLUTE MATCH.
- [ ] End-to-end: `device: 1` in `ddrs.yaml`, run a short train, watch
      `nvidia-smi` for memory on GPU 1. This is the user's actual ask.
- [ ] README config section: one-liner documenting the `device:` key.

## Desktop results already banked (no need to repeat)

- `cargo test --test mmc`: 13 passed.
- `cargo test --test device_selection`: passes via the skip path (1 GPU) —
  the <2-device guard works.

## Outcome (2026-06-06 wukong session)

### SIGFPE: root-caused and fixed (commit `2bba409`)

- Step 1: `mmc` passed → not process startup. Step 2: libtest header printed,
  kernel trap log named **libcusparse** (`trap divide error ... in
  libcusparse.so.12.1.3.153`), thread `sandbox_smoke_p…` = the test fn.
- Crashed identically with cusparse 12.1.3 (CUDA 12.3 module) AND 12.5.4
  (cuda-12.6) → not a library-version bug.
- gdb works on wukong with `LD_LIBRARY_PATH= gdb …` (the module path breaks
  it; system gdb is fine). Backtrace: `cusparseSpSV_solve` ← `cusparse_forward`
  ← hotstart solve in `setup_inputs`, **stage 2 on cuda:1** right after cuda:0
  passed. At the fault, `cuCtxGetDevice()` returned **0** while solving with
  device-1 buffers/stream.
- Root cause: cuSPARSE dispatches into the thread-current CUDA context;
  cubecl only sets the calling thread's context at client **creation**
  (cubecl-cuda `runtime.rs`), so after a second device's client exists the
  thread can be left on the wrong device. Device 0 only ever worked by
  accident.
- Fix: `bind_primary_context` called from `ensure_cuda_cache` (the choke
  point all raw-cuSPARSE entries pass through). `tests/device_selection.rs`
  now passes all 3 stages on wukong (8 GPUs), and the single-device GPU
  tests (`sparse_cusparse_v8`, `sp10_*`) still pass.

### Remaining checklist status

- [x] `device_selection` green on wukong, actually running (3 stages).
- [x] Full `cargo test`: green EXCEPT `data_static ::
      attributes_store_opens_against_conus_subset` — environmental, see below.
- [BLOCKED] Parity gate: **DIVERGENCE (~1% rel, max abs 0.55 m³/s)** — but
  NOT caused by any ddrs commit: the original port commit `6083226` diverges
  identically against a fixture regenerated on wukong. Root cause: **the
  desktop's `~/projects/ddr` contains unpushed/uncommitted work** that the
  port mirrors:
  - `src/geometry.rs` cites `ddr/src/ddr/geometry/trapezoidal.py` — that file
    does not exist at ANY commit in wukong's DDR clone (checked `--all`).
  - DDR-at-HEAD's sandbox learns `top_width`/`side_slope` (denorm 0.5 →
    √5000 ≈ 70.7 m, √25 = 5.0); ddrs derives them Leopold-Maddock-style
    (`top_width = p·depth^q`). Different geometry → ~1% wave-shape diff.
  - Bisect: wukong-DDR sandbox output is byte-identical from `ebba510`
    (Feb 7) through HEAD `c68a937`; no commit in history matches ddrs.
  - Resolution needs the desktop: push/commit the DDR-side geometry work
    (or copy the desktop's `fixtures/sandbox/` to wukong) and re-run.
- [BLOCKED] End-to-end `device: 1` train: fails before training —
  `merit_conus_adjacency.zarr` on wukong (both `~/projects/ddr/data` and
  `/projects/mhpi/tbindas/ddr/data`, written Feb 7) **lacks the `length_m`
  and `slope` arrays** ddrs's `ConusAdjacencyStore` reads. Same root cause:
  desktop DDR's `build_merit_adjacency` writes those arrays; wukong's zarr
  predates that. Regenerate the zarr with the desktop's DDR code on wukong.
  (`ddrs.yaml` is left at `device: 1`, ready for the retry; binary
  reinstalled with the fix via `cargo install --path .`.)
- [x] README: `device:` key documented under "Override workflow".
- The `data_static` failure and `sparse_cusparse_v5 end_to_end` /
  `sparse_cusparse_v6` ignored-test failures are all the same broken-zarr
  environment issue.
