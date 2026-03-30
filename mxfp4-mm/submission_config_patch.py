"""GEMM with monkey-patched configs for K>1024 shapes."""
import os, torch
from task import input_t, output_t
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle

# Monkey-patch _get_config to try custom configs for K>1024
try:
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config as _orig_get_config
    import aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 as _afp4_mod
    
    # Custom configs to try (tuned for MI355X)
    CUSTOM_CONFIGS = {
        # (N, K): config dict
        (2112, 7168): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
                       "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
                       "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        (7168, 2048): {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                       "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
                       "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        (3072, 1536): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                       "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
                       "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
    }
    
    def _patched_get_config(M, N, K, shuffle=False):
        key = (N, K)
        if key in CUSTOM_CONFIGS:
            return CUSTOM_CONFIGS[key], False
        return _orig_get_config(M, N, K, shuffle)
    
    # Clear LRU cache and replace
    if hasattr(_afp4_mod._get_config, 'cache_clear'):
        _afp4_mod._get_config.cache_clear()
    _afp4_mod._get_config = _patched_get_config
    _PATCHED = True
except Exception as e:
    print(f"[WARN] Config patch failed: {e}")
    _PATCHED = False

_warmed = False
_scale_cache = {}

def _unshuffle(B_scale_sh, N, K):
    key = id(B_scale_sh)
    if key in _scale_cache:
        return _scale_cache[key]
    n_sc = K // 32
    s = B_scale_sh.view(torch.uint8)
    sm = ((N + 255) // 256) * 256
    sn = ((n_sc + 7) // 8) * 8
    padded = torch.zeros(sm, sn, dtype=torch.uint8, device=s.device)
    padded[:N, :n_sc] = s[:N, :n_sc]
    r = padded.view(sm//32, sn//8, 4, 16, 2, 2)
    u = r.permute(0, 5, 3, 1, 4, 2).contiguous()
    result = u.view(sm, sn)[:N, :n_sc]
    _scale_cache[key] = result
    return result

def custom_kernel(data: input_t) -> output_t:
    global _warmed
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]
    
    if K <= 1024:
        B_scale = _unshuffle(B_scale_sh, N, K)
        B_q_u8 = B_q.view(torch.uint8)
        cfg = {"BLOCK_SIZE_M": 32 if M > 4 else 16, "BLOCK_SIZE_N": 128,
               "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 4,
               "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": K * 2,
               "num_warps": 4, "num_stages": 2,
               "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
               "cache_modifier": ""}
        return gemm_a16wfp4(A, B_q_u8, B_scale, config=cfg)
    else:
        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_scale_sh = e8m0_shuffle(A_scale)
        A_q_u8 = A_q.view(torch.uint8); B_sh_u8 = B_shuffle.view(torch.uint8); return gemm_afp4wfp4(A_q_u8, B_sh_u8, A_scale_sh.view(torch.uint8), B_scale_sh.view(torch.uint8))
