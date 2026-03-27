#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Dump the FULL quant.py source to find the Triton kernel."""
import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    # Dump lines 1-140 of quant.py (the kernel should be before the wrapper at line 140)
    src_file = "/home/runner/aiter/aiter/ops/triton/quant/quant.py"
    with open(src_file) as f:
        lines = f.readlines()

    print(f"Total lines in quant.py: {len(lines)}")
    # Print lines 1-140 (the kernel definition)
    for i in range(min(140, len(lines))):
        print(f"{i+1}: {lines[i].rstrip()[:150]}")

    # Reference
    def _unshuffle_e8m0(s):
        s = s.view(torch.uint8)
        sm, sn = s.shape
        s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
        s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
        return s.view(sm, sn)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    bu = B_q.view(torch.uint8)
    bs_raw = _unshuffle_e8m0(B_scale_sh)
    return gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)
