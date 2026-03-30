#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Custom AFP4WFP4 GEMM with XCD-aware scheduling for MI355X.
Uses dynamic_mxfp4_quant for A (separate), then custom tl.dot_scaled GEMM.
Key advantage: XCD-aware tile distribution + per-shape configs.
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

# Per-shape configs: (BM, BN, BK, GSM, NW, NS, WPE, KS)
# BK must be >= 128 (FP4 blockscale = 32 elements, min 4 scale groups)
_CONFIGS = {
    (4, 2880, 512):    dict(BM=16, BN=64,  BK=128, GSM=1, NW=4, NS=2, WPE=2, KS=1),
    (16, 2112, 7168):  dict(BM=16, BN=64,  BK=256, GSM=1, NW=4, NS=2, WPE=2, KS=8),
    (32, 4096, 512):   dict(BM=32, BN=128, BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=1),
    (32, 2880, 512):   dict(BM=32, BN=128, BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=1),
    (64, 7168, 2048):  dict(BM=64, BN=128, BK=256, GSM=4, NW=4, NS=2, WPE=2, KS=2),
    (256, 3072, 1536): dict(BM=64, BN=128, BK=256, GSM=4, NW=4, NS=2, WPE=2, KS=1),
}


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
    bm = 32 if m <= 32 else 64
    bn = 64 if n <= 2048 else 128
    bk = 128
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    ks = 1
    if tiles < 64 and k > 512:
        ks = min(8, max(1, 256 // tiles))
        while ks > 1 and triton.cdiv(k, ks) < bk:
            ks //= 2
    return dict(BM=bm, BN=bn, BK=bk, GSM=4, NW=4, NS=2, WPE=2, KS=ks)


@triton.jit
def _afp4_gemm_kernel(
    a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,    # A_fp4: (M, K//2) uint8
    stride_bn, stride_bk,    # B_fp4: (N, K//2) uint8
    stride_cm, stride_cn,
    stride_asm, stride_ask,  # A_scale: (M, K//32) uint8
    stride_bsn, stride_bsk,  # B_scale: (N, K//32) uint8
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_pids = num_pid_m * num_pid_n

    # XCD-aware swizzle for MI355X (8 XCDs)
    if NUM_XCDS > 1:
        pids_per_xcd = total_pids // NUM_XCDS
        extra = total_pids % NUM_XCDS
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        pid = xcd * pids_per_xcd + tl.minimum(xcd, extra) + local_pid

    # Grouped ordering for L2 cache reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in tl.static_range(0, 1):  # single iteration to start accumulator
        pass  # acc already initialized

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K

        # Load A_fp4 tile [BLOCK_M, BLOCK_K//2] as uint8
        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start // 2 + offs_k_packed)[None, :] * stride_ak
        a_fp4 = tl.load(a_ptrs)

        # Load B_fp4 tile [BLOCK_N, BLOCK_K//2] as uint8
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + (k_start // 2 + offs_k_packed)[None, :] * stride_bk
        b_fp4 = tl.load(b_ptrs)

        # Load A scales [BLOCK_M, BLOCK_K//32]
        offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
        as_ptrs = a_scales_ptr + offs_m[:, None] * stride_asm + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_ask
        a_scales = tl.load(as_ptrs)

        # Load B scales [BLOCK_N, BLOCK_K//32]
        bs_ptrs = b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_bsk
        b_scales = tl.load(bs_ptrs)

        # FP4 GEMM via tl.dot_scaled (compiles to MFMA FP4 on gfx950)
        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

    # Store
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c, mask=c_mask)


@triton.jit
def _afp4_splitk_kernel(
    a_ptr, b_ptr, pp_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_ppk, stride_ppm, stride_ppn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    ITERS_PER_SPLIT: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_split = num_pid_m * num_pid_n

    split_id = pid // tiles_per_split
    tile_id = pid % tiles_per_split

    if NUM_XCDS > 1:
        pids_per_xcd = tiles_per_split // NUM_XCDS
        extra = tiles_per_split % NUM_XCDS
        xcd = tile_id % NUM_XCDS
        local_pid = tile_id // NUM_XCDS
        tile_id = xcd * pids_per_xcd + tl.minimum(xcd, extra) + local_pid

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_base = split_id * ITERS_PER_SPLIT * BLOCK_K

    for local_iter in range(ITERS_PER_SPLIT):
        k_start = k_base + local_iter * BLOCK_K
        if k_start < K:
            offs_k_packed = tl.arange(0, BLOCK_K // 2)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start // 2 + offs_k_packed)[None, :] * stride_ak
            a_fp4 = tl.load(a_ptrs)

            b_ptrs = b_ptr + offs_n[:, None] * stride_bn + (k_start // 2 + offs_k_packed)[None, :] * stride_bk
            b_fp4 = tl.load(b_ptrs)

            offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
            as_ptrs = a_scales_ptr + offs_m[:, None] * stride_asm + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_ask
            a_scales = tl.load(as_ptrs)

            bs_ptrs = b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_bsk
            b_scales = tl.load(bs_ptrs)

            acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    pp_ptrs = pp_ptr + split_id * stride_ppk + offs_cm[:, None] * stride_ppm + offs_cn[None, :] * stride_ppn
    tl.store(pp_ptrs, acc, mask=c_mask)


@triton.jit
def _reduce_kernel(
    pp_ptr, c_ptr, M, N,
    stride_ppk, stride_ppm, stride_ppn,
    stride_cm, stride_cn,
    NUM_KSPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
        pp_ptrs = pp_ptr + ks * stride_ppk + offs_m[:, None] * stride_ppm + offs_n[None, :] * stride_ppn
        acc += tl.load(pp_ptrs, mask=mask, other=0.0)

    tl.store(c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :].to(tl.int64) * stride_cn,
             acc.to(tl.bfloat16), mask=mask)


def _launch(A_fp4_u8, A_scale, B_q_u8, B_scale_raw, m, n, k, cfg):
    BM, BN, BK, KS = cfg['BM'], cfg['BN'], cfg['BK'], cfg['KS']

    if KS == 1:
        key = (m, n)
        if key not in _c_cache:
            _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        C = _c_cache[key]
        grid = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
        _afp4_gemm_kernel[grid](
            A_fp4_u8, B_q_u8, C, A_scale, B_scale_raw,
            m, n, k,
            A_fp4_u8.stride(0), A_fp4_u8.stride(1),
            B_q_u8.stride(0), B_q_u8.stride(1),
            C.stride(0), C.stride(1),
            A_scale.stride(0), A_scale.stride(1),
            B_scale_raw.stride(0), B_scale_raw.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            GROUP_SIZE_M=cfg['GSM'], NUM_XCDS=8,
            num_warps=cfg['NW'], num_stages=cfg['NS'],
            waves_per_eu=cfg['WPE'], matrix_instr_nonkdim=16,
        )
        return C
    else:
        total_k_iters = triton.cdiv(k, BK)
        iters_per_split = triton.cdiv(total_k_iters, KS)

        pp_key = (KS, m, n)
        if pp_key not in _pp_cache:
            _pp_cache[pp_key] = torch.empty((KS, m, n), dtype=torch.float32, device='cuda')
        pp = _pp_cache[pp_key]

        tiles = triton.cdiv(m, BM) * triton.cdiv(n, BN)
        _afp4_splitk_kernel[(KS * tiles,)](
            A_fp4_u8, B_q_u8, pp, A_scale, B_scale_raw,
            m, n, k,
            A_fp4_u8.stride(0), A_fp4_u8.stride(1),
            B_q_u8.stride(0), B_q_u8.stride(1),
            pp.stride(0), pp.stride(1), pp.stride(2),
            A_scale.stride(0), A_scale.stride(1),
            B_scale_raw.stride(0), B_scale_raw.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            GROUP_SIZE_M=cfg['GSM'], NUM_KSPLIT=KS,
            ITERS_PER_SPLIT=iters_per_split, NUM_XCDS=8,
            num_warps=cfg['NW'], num_stages=cfg['NS'],
            waves_per_eu=cfg['WPE'], matrix_instr_nonkdim=16,
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

    # Quantize A externally
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4, A_scale = dynamic_mxfp4_quant(A)

    cfg = _get_cfg(m, n, k)

    try:
        return _launch(A_fp4.view(torch.uint8), A_scale, _bq_u8, _bscale_raw, m, n, k, cfg)
    except Exception:
        # Fallback to aiter
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)
