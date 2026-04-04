"""GEMM source code probe — reads Triton kernel source, configs, ASM dispatch logic"""
import os, sys, glob

def _read_file(path, max_lines=200):
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

# Phase 1A: gemm_a16wfp4 source
_read_file("/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py", 300)

# Phase 1B: JSON configs for our shapes
cfg_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm/"
if os.path.isdir(cfg_dir):
    files = sorted(glob.glob(cfg_dir + "*.json"))
    print(f"\n[CONFIGS] {len(files)} JSON files in {cfg_dir}")
    for f in files[:10]:
        print(f"  {os.path.basename(f)}")
    # Read configs matching our shapes
    for pattern in ["fp4", "4x2", "a16w"]:
        matches = [f for f in files if pattern in os.path.basename(f).lower()]
        for f in matches[:5]:
            _read_file(f, 50)

# Phase 1C: gemm_afp4wfp4 source (first 100 lines)
_read_file("/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py", 100)

# Phase 1E: ASM dispatch
for p in ["/home/runner/aiter/aiter/ops/asm/gemm_a4w4.py",
          "/home/runner/aiter/hsa/gemm_a4w4.py"]:
    if os.path.exists(p):
        _read_file(p, 100)

# Phase 1F: .co kernel files
cos = sorted(glob.glob("/home/runner/aiter/hsa/gfx950/f4gemm/*.co"))
print(f"\n[CO FILES] {len(cos)} f4gemm .co files:")
for c in cos:
    sz = os.path.getsize(c)
    print(f"  {os.path.basename(c)} ({sz} bytes)")

# Phase 1G: tuned_gemm.py
_read_file("/home/runner/aiter/aiter/tuned_gemm.py", 200)

# Phase 4: Version info
import subprocess
try:
    r = subprocess.run(["git", "-C", "/home/runner/aiter", "log", "--oneline", "-10"],
                       capture_output=True, text=True, timeout=5)
    print(f"\n[GIT LOG]\n{r.stdout}")
except: pass

try:
    r = subprocess.run(["pip", "list"], capture_output=True, text=True, timeout=10)
    for line in r.stdout.split('\n'):
        if any(x in line.lower() for x in ['triton', 'aiter', 'rocm', 'hip']):
            print(f"[PKG] {line}")
except: pass

# Now do the actual GEMM to pass the test
import torch
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def custom_kernel(data):
    A = data['A']
    B_q = data['B_q'].view(torch.uint8)
    B_scale = _unshuffle_e8m0(data['B_scale_sh'])
    M, K = A.shape
    N = B_q.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    gemm_a16wfp4(A, B_q, B_scale, dtype=torch.bfloat16, y=output)
    return output
