"""
GEMM: Use gemm_a4w4_asm with pre-compiled ASM kernels.
Bypasses Triton entirely — uses hand-optimized .co binaries.
The runner has 36 f4gemm ASM kernels like f4gemm_bf16_per1x32Fp4_BpreShuffle_NxK.

gemm_a4w4_asm(A, B, A_scale, B_scale, out, kernelName,
              bias=None, alpha=1.0, beta=0.0, bpreshuffle=True, log2_k_split=None)
"""
from task import input_t, output_t
import torch
import sys
import os

_warmed = False
_cache = {}


def _prewarm():
    """Discover and warm ASM kernels."""
    global _warmed
    if _warmed:
        return
    _warmed = True

    # List all available .co files
    co_dir = "/home/runner/aiter/hsa/gfx950/f4gemm/"
    if os.path.exists(co_dir):
        cos = [f for f in sorted(os.listdir(co_dir)) if f.endswith('.co')]
        for co in cos:
            print(f"PROBE_CO: {co}", file=sys.stderr)

    # Also check the CSV for kernel configs
    csv_path = os.path.join(co_dir, "f4gemm_bf16_per1x32Fp4.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            content = f.read()
        print(f"PROBE_CSV: {content[:2000]}", file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle
    from aiter.ops.shuffle import shuffle_weight

    _prewarm()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A to MXFP4
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale_e8m0)

    # Try gemm_a4w4_asm with pre-shuffled B
    # B_shuffle is already pre-shuffled, B_scale_sh is shuffled scales
    out = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    try:
        # gemm_a4w4_asm expects:
        # A: fp4x2, B: fp4x2 (pre-shuffled), A_scale: e8m0 (shuffled), B_scale: e8m0 (shuffled)
        # out: bf16, kernelName: auto-selected based on shape, bpreshuffle=True
        result = aiter.gemm_a4w4_asm(
            A_fp4.view(dtypes.fp4x2),
            B_shuffle,
            A_scale_sh.view(dtypes.fp8_e8m0),
            B_scale_sh,
            out,
            "",  # empty kernelName = auto-select
            bpreshuffle=True,
        )
        return result
    except Exception as e:
        print(f"ASM_ERROR: {e}", file=sys.stderr)
        # Fallback to gemm_a4w4
        out_gemm = aiter.gemm_a4w4(
            A_fp4.view(dtypes.fp4x2),
            B_shuffle,
            A_scale_sh.view(dtypes.fp8_e8m0),
            B_scale_sh,
            dtype=dtypes.bf16,
            bpreshuffle=True,
        )
        return out_gemm
