#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Custom Triton MXFP4 GEMM based on official Triton block-scaled matmul tutorial.
Uses tl.dot_scaled with UNSHUFFLED scales, external A quantization.
Key: B accessed as (K,N) via strides (no copy), per-shape configs, split-K.
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_c_cache = {}
_pp_cache = {}
_warmed = False

# Per-shape configs
_CONFIGS = {
    # Small M, small K: enough tiles, no split-K
    (4, 2880, 512):    dict(BM=16, BN=64,  BK=256, NW=4, NS=2, KS=1),
    (32, 4096, 512):   dict(BM=32, BN=128, BK=256, NW=4, NS=2, KS=1),
    (32, 2880, 512):   dict(BM=32, BN=64,  BK=256, NW=4, NS=2, KS=1),
    # Large K: split-K for occupancy
    (16, 2112, 7168):  dict(BM=16, BN=64,  BK=256, NW=4, NS=2, KS=8),
    (64, 7168, 2048):  dict(BM=64, BN=128, BK=256, NW=4, NS=2, KS=2),
    # Large M: no split-K needed
    (256, 3072, 1536): dict(BM=64, BN=128, BK=256, NW=4, NS=2, KS=1),
}
_DEFAULT = dict(BM=32, BN=128, BK=256, NW=4, NS=2, KS=1)


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


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


# ============================================================
# KERNEL: Based on Triton block_scaled_matmul_kernel_cdna4 tutorial
# ============================================================
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
):
    SCALE_GROUP_SIZE: tl.constexpr = 32

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_k_iter = tl.cdiv(K, BLOCK_K)

    # A pointers: (BLOCK_M, BLOCK_K//2) uint8
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B accessed as (K, N) via strides: stride_bk=B_q.stride(1), stride_bn=B_q.stride(0)
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Scale pointers: A_scale (M, K//32), B_scale (N, K//32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE)
    a_scale_ptrs = a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    b_scale_ptrs = b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_scales = tl.load(a_scale_ptrs)
        b_scales = tl.load(b_scale_ptrs)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        a_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
        b_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             c, mask=c_mask)


# ============================================================
# SPLIT-K KERNEL: partial sums per K-split
# ============================================================
@triton.jit
def _gemm_splitk_kernel(
    a_ptr, b_ptr, pp_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ppk, stride_ppm, stride_ppn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_split = num_pid_m * num_pid_n

    split_id = pid // tiles_per_split
    tile_id = pid % tiles_per_split

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE)

    # Start pointers offset by split
    k_start = split_id * K_PER_SPLIT
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + (k_start // 2 + offs_k[None, :]) * stride_ak
    b_ptrs = b_ptr + (k_start // 2 + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
    a_scale_ptrs = a_scales_ptr + offs_am[:, None] * stride_asm + (k_start // SCALE_GROUP_SIZE + offs_ks[None, :]) * stride_ask
    b_scale_ptrs = b_scales_ptr + offs_bn[:, None] * stride_bsn + (k_start // SCALE_GROUP_SIZE + offs_ks[None, :]) * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    iters_this_split = tl.cdiv(K_PER_SPLIT, BLOCK_K // 2)

    for k in range(0, iters_this_split):
        k_offset = k_start + k * (BLOCK_K // 2)
        if k_offset < K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            a_scales = tl.load(a_scale_ptrs)
            b_scales = tl.load(b_scale_ptrs)

            accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

            a_ptrs += (BLOCK_K // 2) * stride_ak
            b_ptrs += (BLOCK_K // 2) * stride_bk
            a_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(pp_ptr + split_id * stride_ppk + offs_cm[:, None] * stride_ppm + offs_cn[None, :] * stride_ppn,
             accumulator, mask=c_mask)


@triton.jit
def _reduce_kernel(
    pp_ptr, c_ptr, M, N,
    stride_ppk, stride_ppm, stride_ppn,
    stride_cm, stride_cn,
    NUM_KSPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ks in range(NUM_KSPLIT):
        partial = tl.load(pp_ptr + ks * stride_ppk + offs_m[:, None] * stride_ppm + offs_n[None, :] * stride_ppn,
                          mask=mask, other=0.0)
        acc += partial

    tl.store(c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :].to(tl.int64) * stride_cn,
             acc.to(tl.bfloat16), mask=mask)


# ============================================================
# LAUNCHER
# ============================================================
def _launch(A_fp4, A_scale, B_q, B_scale, m, n, k, cfg):
    BM, BN, BK, KS = cfg['BM'], cfg['BN'], cfg['BK'], cfg['KS']

    if KS == 1:
        key = (m, n)
        if key not in _c_cache:
            _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        C = _c_cache[key]

        grid = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
        _gemm_kernel[grid](
            A_fp4, B_q, C, A_scale, B_scale,
            m, n, k,
            A_fp4.stride(0), A_fp4.stride(1),
            B_q.stride(1), B_q.stride(0),  # B accessed as (K,N): stride_bk=dim1, stride_bn=dim0
            C.stride(0), C.stride(1),
            A_scale.stride(0), A_scale.stride(1),
            B_scale.stride(0), B_scale.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=cfg['NW'], num_stages=cfg['NS'],
            matrix_instr_nonkdim=16,
        )
        return C
    else:
        k_per_split = triton.cdiv(k, KS)
        # Round up to BLOCK_K boundary
        k_per_split = triton.cdiv(k_per_split, BK // 2) * (BK // 2)

        pp_key = (KS, m, n)
        if pp_key not in _pp_cache:
            _pp_cache[pp_key] = torch.empty((KS, m, n), dtype=torch.float32, device='cuda')
        pp = _pp_cache[pp_key]

        tiles = triton.cdiv(m, BM) * triton.cdiv(n, BN)
        _gemm_splitk_kernel[(KS * tiles,)](
            A_fp4, B_q, pp, A_scale, B_scale,
            m, n, k,
            A_fp4.stride(0), A_fp4.stride(1),
            B_q.stride(1), B_q.stride(0),
            pp.stride(0), pp.stride(1), pp.stride(2),
            A_scale.stride(0), A_scale.stride(1),
            B_scale.stride(0), B_scale.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            NUM_KSPLIT=KS, K_PER_SPLIT=k_per_split,
            num_warps=cfg['NW'], num_stages=cfg['NS'],
            matrix_instr_nonkdim=16,
        )

        key = (m, n)
        if key not in _c_cache:
            _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        C = _c_cache[key]
        RED_BM, RED_BN = min(BM, 32), min(BN, 128)
        _reduce_kernel[(triton.cdiv(m, RED_BM) * triton.cdiv(n, RED_BN),)](
            pp, C, m, n,
            pp.stride(0), pp.stride(1), pp.stride(2),
            C.stride(0), C.stride(1),
            NUM_KSPLIT=KS, BLOCK_M=RED_BM, BLOCK_N=RED_BN,
            num_warps=4, num_stages=1,
        )
        return C


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    shapes = [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]
    for m, n, k in shapes:
        try:
            cfg = _get_cfg(m, n, k)
            wA = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
            wA_fp4, wA_sc = dynamic_mxfp4_quant(wA)
            wBq = torch.zeros((n, k // 2), dtype=torch.uint8, device='cuda')
            wBs = torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda')
            _launch(wA_fp4.view(torch.uint8), wA_sc, wBq, wBs, m, n, k, cfg)
        except Exception:
            pass
    torch.cuda.synchronize()


# Fallback config for K=7168
_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _fallback(A, B_q_u8, B_scale_raw, m, n, k):
    """Proven aiter fallback."""
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), B_q_u8, A_scale, B_scale_raw,
                             dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16, config=cfg)


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

    # Quantize A
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4, A_scale = dynamic_mxfp4_quant(A)

    cfg = _get_cfg(m, n, k)

    try:
        return _launch(A_fp4.view(torch.uint8), A_scale, _bq_u8, _bscale_raw, m, n, k, cfg)
    except Exception:
        return _fallback(A, _bq_u8, _bscale_raw, m, n, k)
