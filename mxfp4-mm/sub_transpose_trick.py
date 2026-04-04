#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM RADICAL — Transpose trick from NVIDIA GEMM winners.
When M < N (always for us), swap dimensions: compute C^T = B^T × A instead of C = A × B^T.
This puts the large N dimension on the M-parallelism axis of the kernel.

For M=4, N=2880: original has 1 tile in M dimension.
After transpose: M'=2880, N'=4 → 45+ tiles in M → better CU utilization.

The trick: we quantize A to fp4, then call gemm_afp4wfp4 with B and A SWAPPED.
B is already fp4. A needs quantization (same as K=1536 path).

gemm_afp4wfp4(A_fp4=B_q, B_fp4=A_q, A_scale=B_scale, B_scale=A_scale) → C^T
Then transpose C^T to get C.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

from task import input_t, output_t
import torch

_bscale_raw = None
_bscale_ref = None
_bq_u8 = None
_y_cache = {}
_warmed = False

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

def custom_kernel(data: input_t) -> output_t:
    global _bscale_raw, _bscale_ref, _bq_u8, _warmed
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    if not _warmed:
        _warmed = True
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        for wm in [4, 16, 32, 64, 256]:
            try:
                dummy = torch.randn(wm, k, dtype=torch.bfloat16, device=A.device)
                dynamic_mxfp4_quant(dummy)
                gemm_a16wfp4(dummy, _bq_u8, _bscale_raw, dtype=torch.bfloat16)
            except:
                pass
        torch.cuda.synchronize()

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    # TRANSPOSE TRICK: for small M, swap A and B
    # Original: C[m,n] = A[m,k] × B[n,k]^T (a16wfp4: A=bf16, B=fp4)
    # Transposed: C^T[n,m] = B[n,k] × A[m,k]^T
    # Using afp4wfp4: both operands fp4, "A"=B_q[n,k/2], "B"=A_q[m,k/2]
    # Result is [n,m], transpose to get [m,n]

    if m <= 64 and k != 1536:
        # Transpose path: quantize A, then call afp4wfp4 with B as "A" and A as "B"
        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_q_u8 = A_q.view(torch.uint8)

        # C^T[n,m] = gemm(A=B_q[n,k/2], B=A_q[m,k/2], A_scale=B_scale, B_scale=A_scale)
        C_T = gemm_afp4wfp4(
            _bq_u8,     # "A" operand = B weights [n, k/2]
            A_q_u8,     # "B" operand = quantized A [m, k/2]
            _bscale_raw, # "A" scale = B scale
            A_scale,     # "B" scale = A scale
            dtype=torch.bfloat16
        )
        # C_T is [n, m], transpose to [m, n]
        return C_T.t().contiguous()

    elif k == 7168:
        key = (m, n)
        if key not in _y_cache:
            _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        out = _y_cache[key]
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=_K7168_CONFIG)
        return out

    elif k == 1536:
        A_q, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_q.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)

    else:
        key = (m, n)
        if key not in _y_cache:
            _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        out = _y_cache[key]
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out)
        return out
