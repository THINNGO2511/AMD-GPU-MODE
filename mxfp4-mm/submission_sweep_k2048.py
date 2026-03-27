#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Focused sweep: 5 configs for K=2048 (M=64, N=7168).
Current best: 14.2us with default config.
M=64 means BM=32 or BM=64 tiles fit. N=7168 is large → big BN may help.
"""
from task import input_t, output_t
import torch
import time

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None

_CFGS = [
    # [0] Default (baseline ~14.2us) — let library choose
    None,
    # [1] BM=32,BN=128,KS=4
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": None},
    # [2] BM=64,BN=128,KS=2
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
    # [3] BM=64,BN=256,KS=2
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None},
    # [4] BM=32,BN=128,KS=4,stages=3
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512, "cache_modifier": None},
]

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


_swept = set()
_best_cfg_2048 = None

def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8, _best_cfg_2048

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    if k == 2048 and k not in _swept:
        _swept.add(k)
        y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        results = []
        print(f"\n=== K=2048 SWEEP M={m} N={n} ===")

        for i, cfg in enumerate(_CFGS):
            try:
                for _ in range(3):
                    gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
                torch.cuda.synchronize()
                iters = 20
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
                torch.cuda.synchronize()
                us = (time.perf_counter() - t0) / iters * 1e6
                if cfg is None:
                    desc = "DEFAULT"
                else:
                    desc = f"BM={cfg['BLOCK_SIZE_M']},BN={cfg['BLOCK_SIZE_N']},KS={cfg['NUM_KSPLIT']},s={cfg['num_stages']},wpe={cfg['waves_per_eu']}"
                results.append((us, i, desc))
                print(f"  [{i}] {us:7.1f}us | {desc}")
            except Exception as e:
                print(f"  [{i}] FAILED: {e}")

        results.sort()
        print(f"\n  BEST: {results[0][2]} @ {results[0][0]:.1f}us")
        _best_cfg_2048 = _CFGS[results[0][1]]

    # Production path
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    _K7168_CFG = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1,
     "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
     "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024, "cache_modifier": None}

    if k == 7168:
        cfg = _K7168_CFG
    elif k == 2048:
        cfg = _best_cfg_2048
    else:
        cfg = None

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
