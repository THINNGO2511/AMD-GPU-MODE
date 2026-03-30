#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""GEMM: Bypass Triton entirely — call CK ASM kernel directly via aiter's gemm_a4w4_asm.
Use pre-quantized A from dynamic_mxfp4_quant, skip Triton path."""
import torch
from task import input_t, output_t

_cache = {}

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]
    
    # Use CK ASM path: requires pre-quantized A
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle
    import aiter
    
    
    key = id(B_scale_sh)
    if key not in _cache:
        _cache.clear()
        _cache[key] = True
    
    # Quantize A
    A_q, A_s = dynamic_mxfp4_quant(A)
    A_s_sh = e8m0_shuffle(A_s)
    
    # Get padded M for kernel selection
    padded_M = aiter.get_padded_m(M, N, K, 1)
    
    # Output
    out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    
    # Call CK ASM directly — the kernel names are auto-selected
    # from the CSV based on (cu_num, M, N, K)
    try:
        result = aiter.gemm_a4w4_asm(
            A_q.view(B_shuffle.dtype),
            B_shuffle,
            A_s_sh.view(B_scale_sh.dtype),
            B_scale_sh,
            out,
            "",  # empty kernelName = auto-select from CSV
            bpreshuffle=True,
        )
        return result
    except Exception as e:
        # Fallback to Triton
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        def _unshuffle(scale_sh):
            s = scale_sh.view(torch.uint8)
            sm, sn = s.shape
            s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
            s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
            return s.view(sm, sn)
        B_scale_raw = _unshuffle(B_scale_sh)
        B_q_u8 = B_q.view(torch.uint8)
        A_s_raw = A_s.view(torch.uint8)
        return gemm_afp4wfp4(A_q.view(torch.uint8), B_q_u8, A_s_raw, B_scale_raw)
