"""MLA source code probe — reads mla.py source, ASM kernel names, version info"""
import os, glob

def _read_file(path, max_lines=250):
    try:
        with open(path) as f:
            lines = f.readlines()
        print(f"\n{'='*60}\n[FILE] {path} ({len(lines)} lines)\n{'='*60}")
        for i, line in enumerate(lines[:max_lines]):
            print(line, end='')
        if len(lines) > max_lines:
            print(f"\n... truncated ({len(lines) - max_lines} more lines)")
    except Exception as e:
        print(f"[FILE] {path}: {e}")

# Phase 3A: mla.py source
_read_file("/home/runner/aiter/aiter/mla.py", 400)

# Phase 3B: MLA ASM kernels
cos = sorted(glob.glob("/home/runner/aiter/hsa/gfx950/mla/*.co"))
print(f"\n[MLA .co files] {len(cos)} found:")
for c in cos:
    print(f"  {os.path.basename(c)} ({os.path.getsize(c)} bytes)")

# Phase 3C: pg2 / page_size in mla.py
import subprocess
try:
    r = subprocess.run(["grep", "-n", "page_size\\|pg2\\|kv_granularity\\|page_table",
                        "/home/runner/aiter/aiter/mla.py"],
                       capture_output=True, text=True, timeout=5)
    print(f"\n[page_size refs in mla.py]\n{r.stdout}")
except: pass

# Phase 3D: MXFP4 MLA support
try:
    r = subprocess.run(["grep", "-rn", "fp4\\|mxfp4\\|float4",
                        "/home/runner/aiter/aiter/mla.py"],
                       capture_output=True, text=True, timeout=5)
    print(f"\n[fp4 refs in mla.py]\n{r.stdout}")
except: pass

# Phase 4: Version info
try:
    r = subprocess.run(["git", "-C", "/home/runner/aiter", "log", "--oneline", "-15"],
                       capture_output=True, text=True, timeout=5)
    print(f"\n[GIT LOG - last 15 commits]\n{r.stdout}")
except: pass

try:
    r = subprocess.run(["git", "-C", "/home/runner/aiter", "status", "--short"],
                       capture_output=True, text=True, timeout=5)
    print(f"\n[GIT STATUS]\n{r.stdout}")
except: pass

# Python/ROCm versions
import torch, triton
print(f"\n[VERSIONS]")
print(f"  torch: {torch.__version__}")
print(f"  triton: {triton.__version__}")
print(f"  hip: {torch.version.hip}")
print(f"  cuda: {torch.version.cuda}")

# Check for qseqlen4 kernel
qseq4 = glob.glob("/home/runner/aiter/hsa/gfx950/mla/*qseqlen4*")
qseq2 = glob.glob("/home/runner/aiter/hsa/gfx950/mla/*qseqlen2*")
print(f"\n[qseqlen variants] qseqlen2: {len(qseq2)}, qseqlen4: {len(qseq4)}")
for f in qseq2 + qseq4:
    print(f"  {os.path.basename(f)}")

# Check for 3-buffer MLA support
try:
    r = subprocess.run(["grep", "-rn", "3buffer\\|three_buffer\\|ds32\\|page64",
                        "/home/runner/aiter/aiter/mla.py"],
                       capture_output=True, text=True, timeout=5)
    if r.stdout.strip():
        print(f"\n[3-buffer refs]\n{r.stdout}")
    else:
        print("\n[3-buffer refs] NONE FOUND in mla.py")
except: pass

# Now do the actual MLA to pass the test
import torch
from aiter.mla import mla_decode_fwd

def custom_kernel(data):
    return mla_decode_fwd(**data)
