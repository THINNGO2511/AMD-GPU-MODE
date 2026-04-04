#!/usr/bin/env python3
"""
GEMM e8m0_shuffle Probe — Understand the shuffle transformation for fused HIP kernel
Submit as: popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark mxfp4-mm/gemm_shuffle_probe.py --no-tui
"""
import torch
import inspect
import subprocess
import time

print("=" * 60)
print("PROBE: e8m0_shuffle transformation details")
print("=" * 60)

# 1. Print e8m0_shuffle source
from aiter.utility.fp4_utils import e8m0_shuffle
print("\n=== e8m0_shuffle source ===")
try:
    src = inspect.getsource(e8m0_shuffle)
    print(src)
except Exception as e:
    print(f"Error: {e}")

# 2. Test with small inputs to understand the transformation
from aiter.ops.triton.quant import dynamic_mxfp4_quant

for M, K in [(4, 512), (16, 7168), (64, 2048), (256, 1536)]:
    print(f"\n=== Shape M={M}, K={K} ===")
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    
    fp4, scale = dynamic_mxfp4_quant(A)
    print(f"  A:     {A.shape} {A.dtype}")
    print(f"  fp4:   {fp4.shape} {fp4.dtype}")
    print(f"  scale: {scale.shape} {scale.dtype}")
    
    scale_sh = e8m0_shuffle(scale)
    print(f"  scale_shuffled: {scale_sh.shape} {scale_sh.dtype}")
    print(f"  ratio: {scale_sh.shape[0] / scale.shape[0]}x rows, {scale_sh.shape[1] / scale.shape[1]}x cols")
    
    # Print first few values to understand the pattern
    scale_cpu = scale.cpu()
    scale_sh_cpu = scale_sh.cpu()
    print(f"  scale[0,:8]:    {scale_cpu[0,:min(8,scale_cpu.shape[1])].tolist()}")
    print(f"  scale_sh[0,:8]: {scale_sh_cpu[0,:min(8,scale_sh_cpu.shape[1])].tolist()}")
    if scale_sh_cpu.shape[0] > 1:
        print(f"  scale_sh[1,:8]: {scale_sh_cpu[1,:min(8,scale_sh_cpu.shape[1])].tolist()}")

# 3. Time each operation separately
print("\n=== Timing breakdown ===")
for M, K in [(16, 7168), (64, 2048), (256, 1536)]:
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    
    # Warmup
    for _ in range(3):
        fp4, scale = dynamic_mxfp4_quant(A)
        scale_sh = e8m0_shuffle(scale)
    
    torch.cuda.synchronize()
    
    # Time quant
    t0 = time.perf_counter()
    for _ in range(100):
        fp4, scale = dynamic_mxfp4_quant(A)
    torch.cuda.synchronize()
    t_quant = (time.perf_counter() - t0) / 100 * 1e6
    
    # Time shuffle
    t0 = time.perf_counter()
    for _ in range(100):
        scale_sh = e8m0_shuffle(scale)
    torch.cuda.synchronize()
    t_shuffle = (time.perf_counter() - t0) / 100 * 1e6
    
    print(f"  M={M} K={K}: quant={t_quant:.1f}μs, shuffle={t_shuffle:.1f}μs, total={t_quant+t_shuffle:.1f}μs")

# 4. Verify gemm_a4w4 works with shuffled scales
print("\n=== gemm_a4w4 accuracy check ===")
from aiter import gemm_a4w4, dtypes as aiter_dtypes
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

# Use the actual eval format
M, K, N = 16, 7168, 2112
A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
# We need B_shuffle and B_scale_sh from eval — use random for shape check
B_shuffle = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda')

# Check B_scale_shuffle shape
print(f"  Expected B_scale_shuffle shape for N={N}, K={K}: [{N}, {K//32}] = [{N}, {K//32}]")

# 5. Time gemm_a4w4 alone (without quant overhead)
print("\n=== gemm_a4w4 timing (GEMM only, no quant) ===")
for M, N, K in [(4, 2880, 512), (16, 2112, 7168), (32, 4096, 512), (64, 7168, 2048), (256, 3072, 1536)]:
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    fp4_u8, scale_u8 = dynamic_mxfp4_quant(A)
    fp4 = fp4_u8.view(aiter_dtypes.fp4x2)
    scale_sh = e8m0_shuffle(scale_u8).view(aiter_dtypes.fp8_e8m0)
    
    B_sh = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device='cuda').view(aiter_dtypes.fp4x2)
    B_scale = torch.randint(100, 150, (N, K // 32), dtype=torch.uint8, device='cuda').view(aiter_dtypes.fp8_e8m0)
    
    # Warmup
    for _ in range(3):
        try:
            out = gemm_a4w4(fp4, B_sh, scale_sh, B_scale, None, torch.bfloat16, 1.0, 0.0, 1)
        except Exception as e:
            print(f"  M={M} N={N} K={K}: gemm_a4w4 ERROR: {e}")
            break
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            out = gemm_a4w4(fp4, B_sh, scale_sh, B_scale, None, torch.bfloat16, 1.0, 0.0, 1)
        torch.cuda.synchronize()
        t_gemm = (time.perf_counter() - t0) / 100 * 1e6
        print(f"  M={M} N={N} K={K}: gemm_a4w4={t_gemm:.1f}μs")

print("\n" + "=" * 60)
print("PROBE COMPLETE")
print("=" * 60)

# Custom kernel for eval harness
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

def custom_kernel(A, B_shuffle, B_scale_shuffle):
    M, K = A.shape
    N = B_shuffle.shape[0]
    if K == 1536:
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4, B_shuffle, A_scale, B_scale_shuffle)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    gemm_a16wfp4(A, B_shuffle, B_scale_shuffle, out)
    return out
