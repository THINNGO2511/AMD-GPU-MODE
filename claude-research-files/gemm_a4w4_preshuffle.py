#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — gemm_a4w4 with B_shuffle (pre-shuffled weights).

The task provides B_shuffle as a direct input. gemm_a4w4's new API has
bpreshuffle parameter. If we pass B_shuffle with bpreshuffle=1,
it might use the CK ASM .co kernels directly — skipping Triton entirely.

The old gemm_a4w4 path was 19-34μs with 3 kernel launches.
But the API changed — the new API might handle A quant internally.

Probe: try calling gemm_a4w4 with the new signature and see what happens.
If it works and is fast, this is the 8μs path.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

from task import input_t, output_t
import torch
import sys

_warmed = False
_y_cache = {}

def custom_kernel(data: input_t) -> output_t:
    global _warmed
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Try gemm_a4w4 with new API
    try:
        from aiter import gemm_a4w4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        # Quantize A
        A_fp4, A_scale = dynamic_mxfp4_quant(A)

        # Try with B_shuffle and bpreshuffle=1
        key = (m, n)
        if key not in _y_cache:
            _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        out = _y_cache[key]

        # New API: gemm_a4w4(A, B, A_scale, B_scale, bias, dtype, alpha, beta, bpreshuffle)
        result = gemm_a4w4(
            A_fp4,          # quantized A
            B_shuffle,      # pre-shuffled B
            A_scale,        # A scales
            B_scale_sh,     # shuffled B scales
            None,           # bias
            torch.bfloat16, # output dtype
            1.0,            # alpha
            0.0,            # beta
            1,              # bpreshuffle=True
        )
        return result

    except Exception as e:
        # Fallback to proven path if gemm_a4w4 fails
        if not _warmed:
            _warmed = True
            print(f"[GEMM] gemm_a4w4 failed: {e}", flush=True)
            print(f"[GEMM] Falling back to gemm_a16wfp4", flush=True)

        # Standard fallback
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

        # Unshuffle scales
        su = B_scale_sh.view(torch.uint8)
        sm, sn = su.shape
        d0, d1 = sm // 32, sn // 8
        total = sm * sn
        idx = torch.arange(total, dtype=torch.int64, device=su.device)
        idx = idx.view(d0, d1, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(-1)
        bscale_raw = torch.take(su.reshape(-1), idx).view(sm, sn)
        bq_u8 = B_q.view(torch.uint8)

        key = (m, n)
        if key not in _y_cache:
            _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        out = _y_cache[key]

        if k == 1536:
            af, asc = dynamic_mxfp4_quant(A)
            return gemm_afp4wfp4(af.view(torch.uint8), bq_u8, asc, bscale_raw, dtype=torch.bfloat16)
        else:
            gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out, config=None)
            return out
