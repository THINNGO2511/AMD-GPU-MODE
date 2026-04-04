#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Direct ASM .co kernel loading via hipModuleLoad.
Bypasses Triton entirely. Uses the 36 pre-compiled FP4 GEMM kernels.
The GEMM probe confirmed hipModuleLoad works on the runner.

The .co files are at /home/runner/aiter/hsa/gfx950/f4gemm/
Format: f4gemm_bf16_per1x32Fp4_BpreShuffle_{M}x{N}.co
Available M tiles: 32, 64, 128, 192, 256
Available N tiles: 128 (most common)

Strategy: Find the best .co kernel for each benchmark shape via aiter's
gemm_a4w4_asm which already handles dispatch. The key is using B_shuffle
(pre-shuffled) + B_scale_sh (shuffled scales) which the ASM path expects.
"""
from task import input_t, output_t
import torch

_warmed = False
_y_cache = {}


def custom_kernel(data: input_t) -> output_t:
    global _warmed

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if not _warmed:
        _warmed = True
        # Pre-warm: import aiter and trigger module JIT once
        import aiter
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_shuffle
        # Warm quant for all M sizes
        for wm in [4, 16, 32, 64, 256]:
            try:
                dummy = torch.randn(wm, k, dtype=torch.bfloat16, device=A.device)
                dynamic_mxfp4_quant(dummy)
            except:
                pass
        torch.cuda.synchronize()

    # Use aiter's gemm_a4w4 which dispatches to ASM .co kernels
    # This takes BOTH A and B as fp4, with shuffled scales
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle
    import aiter
    from aiter import dtypes

    # Quantize A on-the-fly
    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)

    # Call gemm_a4w4 with pre-shuffled B
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    out = aiter.gemm_a4w4(
        A_q.view(dtypes.fp4x2),
        B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0),
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
    return out
