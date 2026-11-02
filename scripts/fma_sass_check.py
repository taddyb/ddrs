#!/usr/bin/env python3
"""FFMA counts of the dumped stage kernels.

usage: python3 scripts/fma_sass_check.py <cubecl debug log> <out json>
Dump the kernel source first: run one CUDA block test with CUBECL_DEBUG_LOG=<file> set, for example
  CUBECL_DEBUG_LOG=/tmp/kernels.log cargo test --features cuda --test stage_cuda_blocks
The env var SM names the GPU architecture (default sm_89). nvcc and cuobjdump must be on PATH.
For each of our kernels (names below) the dumped C++ is compiled with nvcc -cubin at --fmad=true and --fmad=false
and the FFMA lines of cuobjdump -sass are counted. 'fma(' counts the explicit calls in the source."""
import hashlib, json, os, re, subprocess, sys, tempfile
NAMES = ["prologue_kernel","i_mid_kernel","depth_kernel","section_kernel","storage_rate_kernel","newton_kernel",
         "control_kernel","matrix_kernel","eo_kernel","rhs_kernel","readout_kernel","mass_terms_kernel",
         "mass_combine_kernel","mass_add_kernel","epilogue_kernel","tree_factor_kernel",
         "output_backward_kernel","readout_backward_kernel","down_transpose_kernel","rhs_backward_kernel",
         "matrix_backward_row_kernel","matrix_backward_col_kernel",
         "section_backward_kernel","newton_backward_kernel","control_backward_kernel","depth_backward_kernel",
         "prologue_backward_kernel","store_top_width_kernel","i_delta_acc_kernel"]
log, out = sys.argv[1], sys.argv[2]
arch = os.environ.get("SM", "sm_89")
text = open(log).read()
units = re.findall(r"```cpp\n(.*?)```", text, re.S)
seen, res = {}, {}
tmp = tempfile.mkdtemp()
for u in units:
    m = re.search(r'__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s*)?(\w+)\s*\(', u)
    if not m or m.group(1) not in NAMES:
        continue
    name = m.group(1)
    key = hashlib.md5(u.encode()).hexdigest()[:8]
    if key in seen:
        continue
    seen[key] = 1
    cu = os.path.join(tmp, f"{name}_{key}.cu")
    open(cu, "w").write(u)
    row = {"explicit_fma": len(re.findall(r"\bfma\(|fmaf\(", u))}
    for mode in ("true", "false"):
        cb = cu[:-3] + f"_{mode}.cubin"
        subprocess.run(["nvcc", "-cubin", f"-arch={arch}", f"--fmad={mode}", "-o", cb, cu], check=True, capture_output=True)
        sass = subprocess.run(["cuobjdump", "-sass", cb], check=True, capture_output=True, text=True).stdout
        # An FFMA whose addend is RZ or -RZ is a plain multiply (a compiler artifact), not a fused add.
        n = 0
        for line in sass.splitlines():
            mm = re.search(r"\bFFMA(?:\.\w+)*\s+([^;]*);", line)
            if mm:
                last = mm.group(1).split(",")[-1].strip()
                if last not in ("RZ", "-RZ"):
                    n += 1
        row["ffma_" + mode] = n
    # variants of one kernel (comptime table / prescribed) get their own rows
    res.setdefault(name, []).append(row)
json.dump(res, open(out, "w"), indent=1)
for n in NAMES:
    for i, r in enumerate(res.get(n, [])):
        print(f"{n}[{i}] explicit_fma={r['explicit_fma']} ffma_true={r['ffma_true']} ffma_false={r['ffma_false']}")
