#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Read the actual Triton kernel from _triton_kernels/quant/quant.py"""
import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    src_file = "/home/runner/aiter/aiter/ops/triton/_triton_kernels/quant/quant.py"
    with open(src_file) as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines)}")
    # Print all lines containing mxfp4/e8m0/scale/quant
    in_mxfp4_kernel = False
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if '_dynamic_mxfp4_quant_kernel' in stripped or '_mxfp4_quant_op' in stripped:
            in_mxfp4_kernel = True
        if in_mxfp4_kernel:
            print(f"{i+1}: {stripped[:150]}")
        if in_mxfp4_kernel and i > 0 and len(stripped) > 0 and not stripped[0].isspace() and not stripped.startswith('@') and not stripped.startswith('#') and 'def ' not in stripped and 'class' not in stripped:
            if i > 10:
                in_mxfp4_kernel = False

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
