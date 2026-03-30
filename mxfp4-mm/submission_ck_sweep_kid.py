#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""GEMM: Print all available kernelIds and their timing for shape M=4,K=512.
This is a probe submission to find the best CK kernel configs."""
import torch, time, sys
from task import input_t, output_t

from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
import aiter

_first = True

def custom_kernel(data: input_t) -> output_t:
    global _first
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]
    
    A_q, A_s = dynamic_mxfp4_quant(A)
    A_s_sh = e8m0_shuffle(A_s)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    
    if _first:
        _first = False
        # Sweep kernelIds to find best for this shape
        print(f"\n=== Shape M={M} N={N} K={K} ===", file=sys.stderr)
        for kid in range(35):
            for sk in [0, 1, 2]:
                try:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(10):
                        aiter.gemm_a4w4_blockscale_tune(
                            A_q.view(B_shuffle.dtype),
                            B_shuffle,
                            A_s_sh.view(B_scale_sh.dtype),
                            B_scale_sh,
                            out,
                            kid, sk
                        )
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    us = (t1-t0)/10*1e6
                    print(f"  kid={kid} sk={sk}: {us:.1f}μs", file=sys.stderr)
                except Exception as e:
                    pass  # Skip invalid combos
    
    # Use default gemm_a4w4 for actual output
    return aiter.gemm_a4w4(
        A_q.view(B_shuffle.dtype), B_shuffle,
        A_s_sh.view(B_scale_sh.dtype), B_scale_sh,
        dtype=torch.bfloat16, bpreshuffle=True,
    )
