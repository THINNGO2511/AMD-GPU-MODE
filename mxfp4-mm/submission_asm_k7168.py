#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Test ASM kernel (gemm_a4w4) for K=7168 M=16 N=2112.
The ASM kernel uses pre-compiled CK code that might be faster than Triton.
Need to quantize A first (small M=16, low overhead).
Also test gemm_a4w4 with log2_k_split for K=7168.
"""
from task import input_t, output_t
import torch
import time

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


def _probe(A, B_shuffle, B_scale_sh, m, n, k):
    global _probed
    if _probed:
        return
    _probed = True

    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    bq_u8 = B_shuffle.view(torch.uint8)  # B_shuffle is already shuffled

    print(f"\n=== ASM vs Triton for M={m},N={n},K={k} ===", flush=True)

    # 1. Time quant(A) overhead
    for _ in range(3):
        A_fp4, A_sc = dynamic_mxfp4_quant(A)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        A_fp4, A_sc = dynamic_mxfp4_quant(A)
    torch.cuda.synchronize()
    quant_us = (time.perf_counter() - t0) / 30 * 1e6
    print(f"  quant(A) M={m},K={k}: {quant_us:.1f}us", flush=True)

    # 2. Time e8m0_shuffle(A_sc)
    A_fp4, A_sc = dynamic_mxfp4_quant(A)
    for _ in range(3):
        A_sc_sh = e8m0_shuffle(A_sc)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        A_sc_sh = e8m0_shuffle(A_sc)
    torch.cuda.synchronize()
    shuffle_us = (time.perf_counter() - t0) / 30 * 1e6
    print(f"  e8m0_shuffle(A_sc): {shuffle_us:.1f}us", flush=True)

    # 3. Time ASM gemm_a4w4 (no k_split)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_sc_sh = e8m0_shuffle(A_sc).view(dtypes.fp8_e8m0)
    B_scale_sh_v = B_scale_sh.view(dtypes.fp8_e8m0)

    for split in [None, 0, 1, 2, 3]:
        try:
            for _ in range(3):
                out = aiter.gemm_a4w4(A_q, B_shuffle, A_sc_sh, B_scale_sh_v,
                                       dtype=dtypes.bf16, bpreshuffle=True,
                                       log2_k_split=split)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                out = aiter.gemm_a4w4(A_q, B_shuffle, A_sc_sh, B_scale_sh_v,
                                       dtype=dtypes.bf16, bpreshuffle=True,
                                       log2_k_split=split)
            torch.cuda.synchronize()
            asm_us = (time.perf_counter() - t0) / 30 * 1e6
            total = quant_us + shuffle_us + asm_us
            print(f"  ASM split={split}: {asm_us:.1f}us (total={total:.1f}us)", flush=True)
        except Exception as e:
            print(f"  ASM split={split}: FAIL {e}", flush=True)

    # 4. Time a16wfp4 baseline (BM=8 KS=8)
    bscale_raw2 = _unshuffle_e8m0(B_scale_sh)
    bq_raw = B_shuffle.view(torch.uint8)
    # Wait, B_q is the raw quantized weights, B_shuffle is the shuffled version
    # For a16wfp4 we need B_q (unshuffled) + unshuffled scales

    # 5. Time a16wfp4 with various configs
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    # Need the raw B_q, not B_shuffle


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    if k == 7168:
        _probe(A, B_shuffle, B_scale_sh, m, n, k)

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
           "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024} if k == 7168 else None

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)
