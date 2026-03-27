#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Dump the default config that gemm_a16wfp4 selects for each benchmark size.
Also probe the Triton kernel's grid calculation to understand CTA count.
"""
from task import input_t, output_t
import torch
import inspect

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_dumped = set()

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    key = (m, n, k)
    if key not in _dumped:
        _dumped.add(key)

        # Probe the gemm_a16wfp4 source to find config selection
        from aiter.ops.triton.gemm.basic import gemm_a16wfp4 as mod
        src = inspect.getsource(mod)
        lines = src.split('\n')

        # Print the full wrapper function (not the Triton kernel)
        in_func = False
        func_lines = []
        for i, line in enumerate(lines):
            if 'def gemm_a16wfp4' in line and 'triton' not in line.lower():
                in_func = True
            if in_func:
                func_lines.append(f"  {i}: {line}")
                if len(func_lines) > 80:
                    break
                # End of function: next def at same indent
                if len(func_lines) > 2 and (line.startswith('def ') or line.startswith('class ')):
                    break

        if key == (16, 2112, 7168):  # Only print once
            print(f"=== gemm_a16wfp4 wrapper source ===")
            for l in func_lines[:60]:
                print(l)

            # Also find the kernel function
            print(f"\n=== Triton kernel signature ===")
            for i, line in enumerate(lines):
                if '@triton.jit' in line or '@triton.autotune' in line:
                    for j in range(i, min(i+5, len(lines))):
                        print(f"  {j}: {lines[j]}")
                    print("  ...")

            # Check for reduce/splitk kernel
            print(f"\n=== Reduce kernel ===")
            for i, line in enumerate(lines):
                if 'reduce' in line.lower() and ('def ' in line or 'kernel' in line.lower()):
                    for j in range(i, min(i+20, len(lines))):
                        print(f"  {j}: {lines[j]}")

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None} if k == 7168 else None
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
