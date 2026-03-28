#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Custom Triton MXFP4 GEMM v2 — with correct scale shuffle for CDNA4.
Key fix: tl.dot_scaled needs scales in shuffled format. The tutorial
pre-shuffles in host, unshuffles in kernel. We do the same.
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl

_bscale_ref = None
_bscale_sh_flat = None  # Shuffled B scales in (N//32, K) layout
_bq_u8 = None
_c_cache = {}
_pp_cache = {}
_warmed = False

_CONFIGS = {
    (4, 2880, 512):    dict(BM=16, BN=64,  BK=256, NW=4, NS=2, KS=1),
    (32, 4096, 512):   dict(BM=32, BN=128, BK=256, NW=4, NS=2, KS=1),
    (32, 2880, 512):   dict(BM=32, BN=64,  BK=256, NW=4, NS=2, KS=1),
    (16, 2112, 7168):  dict(BM=16, BN=64,  BK=256, NW=4, NS=2, KS=8),
    (64, 7168, 2048):  dict(BM=64, BN=128, BK=256, NW=4, NS=2, KS=2),
    (256, 3072, 1536): dict(BM=64, BN=128, BK=256, NW=4, NS=2, KS=1),
}
_DEFAULT = dict(BM=32, BN=128, BK=256, NW=4, NS=2, KS=1)


def _shuffle_scales_cdna4(scales, mfma_nonkdim=16):
    """Pre-shuffle scales for CDNA4 MFMA (from Triton tutorial).
    Input: (rows, K//32) unshuffled E8M0 scales.
    Output: (rows//32, K//32*32) shuffled for coalesced MFMA access.
    """
    s = scales.clone().view(torch.uint8)
    sm, sn = s.shape
    if mfma_nonkdim == 16:
        s = s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        s = s.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    elif mfma_nonkdim == 32:
        s = s.view(sm // 32, 32, sn // 8, 4, 2, 1)
        s = s.permute(0, 2, 4, 1, 3, 5).contiguous()
    return s.view(sm // 32, sn * 32)


def _get_cfg(m, n, k):
    cfg = _CONFIGS.get((m, n, k))
    if cfg:
        return cfg
    bm = 16 if m <= 16 else (32 if m <= 32 else 64)
    bn = 64 if n <= 2048 else 128
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    ks = 1
    if tiles < 64 and k > 512:
        ks = min(8, max(1, 256 // tiles))
    return dict(BM=bm, BN=bn, BK=256, NW=4, NS=2, KS=ks)


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_k_iter = tl.cdiv(K, BLOCK_K)

    # A: (M, K//2) uint8 packed FP4
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Shuffled scale pointers: A_scales (M//32, K), B_scales (N//32, K)
    # Each 32-row group has K scale bytes in shuffled order
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % (N // 32)
    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)) % (M // 32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    a_scale_ptrs = a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    b_scale_ptrs = b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # Unshuffle scales inside kernel (from tutorial, mfma_nonkdim=16)
        if mfma_nonkdim == 16:
            a_scales = tl.load(a_scale_ptrs).reshape(
                BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
            ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)

            b_scales = tl.load(b_scale_ptrs).reshape(
                BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
            ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        a_scale_ptrs += BLOCK_K * stride_ask
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             c, mask=c_mask)


def _launch(A_fp4, A_scale_shuffled, B_q, B_scale_shuffled, m, n, k, cfg):
    BM, BN, BK, KS = cfg['BM'], cfg['BN'], cfg['BK'], cfg['KS']

    if KS > 1:
        # For now, fall back for split-K shapes (will add split-K later)
        return None

    key = (m, n)
    if key not in _c_cache:
        _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[key]

    grid = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
    _gemm_kernel[grid](
        A_fp4, B_q, C, A_scale_shuffled, B_scale_shuffled,
        m, n, k,
        A_fp4.stride(0), A_fp4.stride(1),
        B_q.stride(1), B_q.stride(0),  # B as (K,N)
        C.stride(0), C.stride(1),
        A_scale_shuffled.stride(0), A_scale_shuffled.stride(1),
        B_scale_shuffled.stride(0), B_scale_shuffled.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        mfma_nonkdim=16,
        num_warps=cfg['NW'], num_stages=cfg['NS'],
        matrix_instr_nonkdim=16,
    )
    return C


_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _fallback(A, B_q_u8, B_scale_raw, m, n, k):
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), B_q_u8, A_scale, B_scale_raw,
                             dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16, config=cfg)


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
    shapes = [(4,2880,512),(32,4096,512),(32,2880,512),(256,3072,1536)]  # Non-splitK shapes
    for m, n, k in shapes:
        try:
            cfg = _get_cfg(m, n, k)
            if cfg['KS'] > 1:
                continue
            wA = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
            wA_fp4, wA_sc = dynamic_mxfp4_quant(wA)
            wA_sc_sh = _shuffle_scales_cdna4(wA_sc)
            wBq = torch.zeros((n, k // 2), dtype=torch.uint8, device='cuda')
            wBs_sh = _shuffle_scales_cdna4(torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda'))
            _launch(wA_fp4.view(torch.uint8), wA_sc_sh, wBq, wBs_sh, m, n, k, cfg)
        except Exception:
            pass
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_sh_flat, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bq_u8 = B_q.view(torch.uint8)
        # Unshuffle B scales to (N, K//32), then re-shuffle for tutorial kernel
        bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bscale_sh_flat = _shuffle_scales_cdna4(bscale_raw)
        _prewarm()

    cfg = _get_cfg(m, n, k)

    # Use custom kernel for non-splitK shapes, fallback for splitK
    if cfg['KS'] == 1:
        try:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            A_scale_shuffled = _shuffle_scales_cdna4(A_scale)
            result = _launch(A_fp4.view(torch.uint8), A_scale_shuffled, _bq_u8, _bscale_sh_flat, m, n, k, cfg)
            if result is not None:
                return result
        except Exception:
            pass

    # Fallback for splitK shapes and errors
    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    return _fallback(A, _bq_u8, bscale_raw, m, n, k)
