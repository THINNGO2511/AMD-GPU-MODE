#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""GEMM: Monkey-patch get_GEMM_config for M>1 CK ASM configs"""
import torch
from task import input_t, output_t

# Custom configs for each benchmark shape
# format: (kernelId, splitK, kernelName)
_CK_CONFIGS = {
    (4, 2880, 512): (0, 1, "f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128"),
    (16, 2112, 7168): (0, 4, "f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128"),
    (32, 4096, 512): (0, 1, "f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128"),
    (32, 2880, 512): (0, 1, "f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128"),
    (64, 7168, 2048): (0, 2, "f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128"),
    (256, 3072, 1536): (0, 1, "f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128"),
}

from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
import aiter

def custom_kernel(data):
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]
    
    # Quantize A
    A_q, A_s = dynamic_mxfp4_quant(A)
    A_s_sh = e8m0_shuffle(A_s)
    
    key = (M, N, K)
    if key in _CK_CONFIGS:
        kid, sk, kname = _CK_CONFIGS[key]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
        try:
            return aiter.gemm_a4w4_asm(
                A_q.view(B_shuffle.dtype), B_shuffle,
                A_s_sh.view(B_scale_sh.dtype), B_scale_sh,
                out, kname, bpreshuffle=True,
                log2_k_split=sk.bit_length()-1 if sk > 1 else 0,
            )
        except Exception:
            pass
    
    # Fallback to gemm_a4w4 (auto-config)
    return aiter.gemm_a4w4(
        A_q.view(B_shuffle.dtype), B_shuffle,
        A_s_sh.view(B_scale_sh.dtype), B_scale_sh,
        dtype=torch.bfloat16, bpreshuffle=True,
    )
