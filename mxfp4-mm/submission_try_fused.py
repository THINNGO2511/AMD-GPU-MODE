#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Try fused_gemm_afp4wfp4_a16w16 for K=1536 M=256 N=3072.
This kernel fuses bf16 A quant + fp4 GEMM, potentially eliminating
the separate quant overhead that hurts M=256.
Also probes its signature and required inputs.
"""
from task import input_t, output_t
import torch
import time
import inspect

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_probed = False

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8, _probed

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    if not _probed:
        _probed = True

        # 1. Probe fused_gemm_afp4wfp4_a16w16 signature & source
        print("=== FUSED GEMM PROBE ===")
        try:
            from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_a16w16 import fused_gemm_afp4wfp4_a16w16
            sig = inspect.signature(fused_gemm_afp4wfp4_a16w16)
            print(f"SIG: {sig}")
            src = inspect.getsource(fused_gemm_afp4wfp4_a16w16)
            # Print function def and key lines
            for i, line in enumerate(src.split('\n')[:60]):
                print(f"  {i}: {line}")
        except Exception as e:
            print(f"FUSED IMPORT FAILED: {e}")

        # 2. Check fused dir for all fused kernels
        print("\n=== FUSED DIR ===")
        import os
        fused_dir = "/home/runner/aiter/aiter/ops/triton/gemm/fused/"
        if os.path.exists(fused_dir):
            for f in sorted(os.listdir(fused_dir)):
                print(f"  {f}")

        # 3. Probe gemm_a16wfp4 full source (we need the kernel itself)
        print("\n=== gemm_a16wfp4 kernel source (reduce part) ===")
        try:
            import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as mod
            src = inspect.getsource(mod)
            lines = src.split('\n')
            # Find split-K reduce kernel
            for i, line in enumerate(lines):
                if 'reduce' in line.lower() and ('def ' in line or 'kernel' in line.lower()):
                    for j in range(i, min(i+30, len(lines))):
                        print(f"  {j}: {lines[j]}")
                    break
        except Exception as e:
            print(f"  {e}")

    # Use standard path for correctness
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
