#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Read the dynamic_mxfp4_quant Triton source to understand A quantization rounding."""
import torch, inspect
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    # Read the source of dynamic_mxfp4_quant
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    print("=== dynamic_mxfp4_quant source ===")
    try:
        src = inspect.getsource(dynamic_mxfp4_quant)
        print(src[:4000])
    except:
        print("Could not get source")

    # Also find the actual Triton kernel it calls
    print("\n=== Looking for _mxfp4_quant kernel ===")
    import aiter.ops.triton.quant as quant_mod
    src_file = inspect.getfile(quant_mod)
    print(f"Source file: {src_file}")
    with open(src_file) as f:
        content = f.read()

    # Find the quantization kernel
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if any(kw in line for kw in ['mxfp4', 'quant', 'e8m0', 'e2m1', 'dot_scaled', 'block_scale', 'amax']):
            start = max(0, i-1)
            end = min(len(lines), i+3)
            for j in range(start, end):
                print(f"  {j+1}: {lines[j][:150]}")
            print()

    # Also check the gemm_a16wfp4 source for how it handles A quantization
    print("\n=== gemm_a16wfp4 source (A quant section) ===")
    from aiter.ops.triton.gemm.basic import gemm_a16wfp4 as g_mod
    src_file2 = inspect.getfile(g_mod)
    print(f"Source file: {src_file2}")
    with open(src_file2) as f:
        content2 = f.read()
    lines2 = content2.split('\n')
    for i, line in enumerate(lines2):
        if any(kw in line for kw in ['dot_scaled', 'scale', 'amax', 'e2m1', 'quant', 'block']):
            start = max(0, i-1)
            end = min(len(lines2), i+3)
            for j in range(start, end):
                print(f"  {j+1}: {lines2[j][:150]}")
            print()

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
