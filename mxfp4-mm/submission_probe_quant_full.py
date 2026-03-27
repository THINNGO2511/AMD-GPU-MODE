#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Read the FULL _dynamic_mxfp4_quant_kernel Triton source."""
import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    # Read the full quant.py file
    src_file = "/home/runner/aiter/aiter/ops/triton/quant/quant.py"
    with open(src_file) as f:
        content = f.read()

    # Find the _dynamic_mxfp4_quant_kernel definition
    lines = content.split('\n')
    in_kernel = False
    kernel_lines = []
    for i, line in enumerate(lines):
        if '_dynamic_mxfp4_quant_kernel' in line and ('def ' in line or '@' in line):
            in_kernel = True
        if in_kernel:
            kernel_lines.append(f"{i+1}: {line}")
            # Stop after finding the end of the kernel (next non-indented def or class)
            if len(kernel_lines) > 5 and line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('@'):
                break
        if len(kernel_lines) > 200:
            break

    print("=== _dynamic_mxfp4_quant_kernel ===")
    print('\n'.join(kernel_lines[:200]))

    # Also search for any "even_round" or scaling functions
    print("\n=== Scaling/rounding functions ===")
    for i, line in enumerate(lines):
        if any(kw in line for kw in ['even_round', 'SCALING_MODE', 'blockscale', 'e8m0_scale', 'amax', 'tl.max']):
            start = max(0, i-1)
            end = min(len(lines), i+2)
            for j in range(start, end):
                print(f"  {j+1}: {lines[j][:150]}")

    # Reference output
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
