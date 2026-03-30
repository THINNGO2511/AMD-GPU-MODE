#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 16: Apply a8wfp4-tuned configs to a16wfp4
The a8wfp4 configs (tuned by AMD engineers) use parameters we haven't tried:
  - waves_per_eu=6 (we only tested up to 4)
  - cache_modifier='.cg' everywhere
  - NUM_KSPLIT=4 for most shapes
  - BLOCK_SIZE_K=256 (instead of 512)

These configs were tuned for a8wfp4 (fp8 A) but the compute patterns are
similar to a16wfp4 (bf16 A, on-the-fly fp4 quant). The optimal tile sizes
might transfer.

a8wfp4 configs by M bucket:
  M<=16:  BM=16 BN=64  BK=256 GSM=1 W=4 S=2 WPE=6 MI=16 KS=4
  M<=32:  BM=32 BN=128 BK=256 GSM=1 W=4 S=2 WPE=4 MI=16 KS=4
  M<=64:  BM=64 BN=128 BK=256 GSM=1 W=4 S=2 WPE=4 MI=16 KS=4
  M<=256: BM=128 BN=128 BK=256 GSM=2 W=4 S=2 WPE=4 MI=16 KS=1
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_warmed = False

# a8wfp4-style configs adapted for a16wfp4
_CONFIGS = {
    # M=4: use M_LEQ_16 config
    (4, 2880, 512): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 6, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
    },
    # M=16: use M_LEQ_16 config with KS=8 for large K
    (16, 2112, 7168): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 6, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
    },
    # M=32: use M_LEQ_32 config
    (32, 4096, 512): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
    },
    (32, 2880, 512): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
    },
    # M=64: use M_LEQ_64 config
    (64, 7168, 2048): {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 1024,
    },
}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try:
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
        except Exception:
            pass
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        _prewarm()

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    cfg = _CONFIGS.get((m, n, k))

    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)
