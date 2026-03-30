#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Split-K custom Triton kernel for K=7168 and K=2048.

Strategy:
- K=512 (M=4/32/32): non-splitK custom kernel (proven 0 error in v3)
- K=7168 (M=16): split-K with KSPLIT=7 (1024 per split, 4 BK=256 iters)
- K=2048 (M=64): split-K with KSPLIT=2 (1024 per split, 4 BK=256 iters)
- K=1536 (M=256): aiter afp4wfp4 fallback (proven fastest for this K)

Split-K design:
- Each split computes a partial fp32 result for its K slice
- Partials stored to workspace buffer [KSPLIT, M, N]
- Reduce kernel sums partials and converts to bf16
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl

try:
    from aiter.ops.triton.quant import _mxfp4_quant_op
    _HAS_INLINE_QUANT = True
except ImportError:
    _HAS_INLINE_QUANT = False

_bscale_ref = None
_bscale_sh_flat = None
_bscale_raw = None
_bq_u8 = None
_c_cache = {}
_pp_cache = {}  # partial products workspace
_warmed = False

# Per-shape configs: (BM, BN, BK, NW, NS, GSM, KSPLIT)
_CONFIGS = {
    # K=512: no split-K needed
    (4, 2880, 512):    dict(BM=32, BN=64,  BK=256, NW=4, NS=2, GSM=1,  KSPLIT=1),
    (32, 4096, 512):   dict(BM=32, BN=64,  BK=256, NW=4, NS=2, GSM=4,  KSPLIT=1),
    (32, 2880, 512):   dict(BM=32, BN=64,  BK=256, NW=4, NS=2, GSM=4,  KSPLIT=1),
    # K=7168: KSPLIT=7 => 1024 per split, 4 BK=256 iters. 33 tiles * 7 = 231 blocks.
    (16, 2112, 7168):  dict(BM=32, BN=64,  BK=256, NW=4, NS=2, GSM=1,  KSPLIT=7),
    # K=2048: KSPLIT=2 => 1024 per split, 4 BK=256 iters. 56 tiles * 2 = 112 blocks.
    (64, 7168, 2048):  dict(BM=64, BN=128, BK=256, NW=4, NS=2, GSM=4,  KSPLIT=2),
    # K=1536: aiter fallback (not used by custom kernel)
    (256, 3072, 1536): dict(BM=64, BN=128, BK=256, NW=4, NS=2, GSM=4,  KSPLIT=1),
}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _shuffle_scales_cdna4(scales):
    s = scales.clone().view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    s = s.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return s.view(sm // 32, sn * 32)


def _get_cfg(m, n, k):
    cfg = _CONFIGS.get((m, n, k))
    if cfg:
        return cfg
    bm = 32 if m <= 32 else 64
    bn = 64 if n <= 2048 else 128
    return dict(BM=bm, BN=bn, BK=256, NW=4, NS=2, GSM=4, KSPLIT=1)


# ---------------------------------------------------------------------------
# Non-splitK kernel: direct bf16 output (used for K=512)
# ---------------------------------------------------------------------------
@triton.jit
def _afp4_gemm_kernel(
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
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_pids = num_pid_m * num_pid_n

    if NUM_XCDS > 1:
        pids_per_xcd = total_pids // NUM_XCDS
        extra = total_pids % NUM_XCDS
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        pid = xcd * pids_per_xcd + tl.minimum(xcd, extra) + local_pid

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_k_iter = tl.cdiv(K, BLOCK_K)

    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)) % (M // 32)
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % (N // 32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    a_scale_ptrs = a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    b_scale_ptrs = b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

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


# ---------------------------------------------------------------------------
# Split-K kernel: writes fp32 partial to workspace [KSPLIT, M, N]
# ---------------------------------------------------------------------------
@triton.jit
def _splitk_gemm_kernel(
    a_ptr, b_ptr, pp_ptr, a_scales_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_pps, stride_ppm, stride_ppn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    K_PER_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    """Each program handles one (split, tile_m, tile_n) chunk."""
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_split = num_pid_m * num_pid_n

    # Decode split_id and tile_id from linear pid
    split_id = pid // tiles_per_split
    tile_pid = pid % tiles_per_split

    # XCD-aware swizzle WITHIN each split's tile space
    if NUM_XCDS > 1:
        pids_per_xcd = tiles_per_split // NUM_XCDS
        extra = tiles_per_split % NUM_XCDS
        xcd = tile_pid % NUM_XCDS
        local_pid = tile_pid // NUM_XCDS
        tile_pid = xcd * pids_per_xcd + tl.minimum(xcd, extra) + local_pid

    # Grouped tile ordering for L2 reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_pid % num_pid_in_group) % group_size_m)
    pid_n = (tile_pid % num_pid_in_group) // group_size_m

    # K range for this split
    k_start = split_id * K_PER_SPLIT
    num_k_iter = K_PER_SPLIT // BLOCK_K

    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # A fp4 pointers: advance by k_start/2 (packed fp4x2)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + (k_start // 2 + offs_k[None, :]) * stride_ak
    # B fp4 pointers: B is (N, K//2) accessed as (K//2, N) via strides
    b_ptrs = b_ptr + (k_start // 2 + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn

    # A scale pointers: shuffled format (M//32, K*32) -- advance by k_start elements
    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, BLOCK_M // 32)) % (M // 32)
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % (N // 32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    # Scale stride_ask is stride along the flat K*32 dimension
    # For K_start, need to advance by (k_start / SCALE_GROUP_SIZE) * 32 = k_start
    a_scale_ptrs = a_scales_ptr + offs_asm[:, None] * stride_asm + (k_start + offs_ks[None, :]) * stride_ask
    b_scale_ptrs = b_scales_ptr + offs_asn[:, None] * stride_bsn + (k_start + offs_ks[None, :]) * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ki in range(0, num_k_iter):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

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

    # Store fp32 partial to workspace [KSPLIT, M, N]
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    pp_offset = split_id * stride_pps + offs_cm[:, None] * stride_ppm + offs_cn[None, :] * stride_ppn
    tl.store(pp_ptr + pp_offset, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Reduce kernel: sum fp32 partials -> bf16 output
# ---------------------------------------------------------------------------
@triton.jit
def _splitk_reduce_kernel(
    pp_ptr, c_ptr,
    M, N,
    stride_pps, stride_ppm, stride_ppn,
    stride_cm, stride_cn,
    NUM_KSPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum KSPLIT partial fp32 results and write bf16 output."""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(NUM_KSPLIT):
        pp_offset = s * stride_pps + offs_m[:, None] * stride_ppm + offs_n[None, :] * stride_ppn
        partial = tl.load(pp_ptr + pp_offset, mask=mask, other=0.0)
        acc += partial

    c = acc.to(tl.bfloat16)
    c_offset = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr + c_offset, c, mask=mask)


# ---------------------------------------------------------------------------
# Fused A-quant kernel (non-splitK, for K=512)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_gemm_kernel(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    mfma_nonkdim: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_pids = num_pid_m * num_pid_n

    if NUM_XCDS > 1:
        pids_per_xcd = total_pids // NUM_XCDS
        extra = total_pids % NUM_XCDS
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        pid = xcd * pids_per_xcd + tl.minimum(xcd, extra) + local_pid

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_k_iter = tl.cdiv(K, BLOCK_K)

    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % (N // 32)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)
    b_scale_ptrs = b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    a_bf16_ptrs = a_ptr + offs_am[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        a_bf16 = tl.load(a_bf16_ptrs).to(tl.float32)
        a_fp4, a_scales = _mxfp4_quant_op(a_bf16, BLOCK_K, BLOCK_M, SCALE_GROUP_SIZE)

        a_scales_sh = a_scales.reshape(
            BLOCK_M // 32, 2, 16, (BLOCK_K // SCALE_GROUP_SIZE) // 8, 2, 4, 1
        ).permute(0, 3, 5, 2, 4, 1, 6).reshape(BLOCK_M // 32, (BLOCK_K // SCALE_GROUP_SIZE) * 32)

        a_scales_final = a_scales_sh.reshape(
            BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
        ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)

        b = tl.load(b_ptrs)
        b_scales = tl.load(b_scale_ptrs).reshape(
            BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2, 1
        ).permute(0, 5, 3, 1, 4, 2, 6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)

        accumulator += tl.dot_scaled(a_fp4, a_scales_final, "e2m1", b, b_scales, "e2m1")

        a_bf16_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             c, mask=c_mask)


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

def _launch_nonsplitk(A_fp4, A_scale_sh, B_q, B_scale_sh, m, n, k, cfg):
    """Launch non-splitK AFP4 GEMM (for K=512 shapes)."""
    BM, BN, BK = cfg['BM'], cfg['BN'], cfg['BK']
    key = (m, n)
    if key not in _c_cache:
        _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[key]

    grid = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
    _afp4_gemm_kernel[grid](
        A_fp4, B_q, C, A_scale_sh, B_scale_sh,
        m, n, k,
        A_fp4.stride(0), A_fp4.stride(1),
        B_q.stride(1), B_q.stride(0),
        C.stride(0), C.stride(1),
        A_scale_sh.stride(0), A_scale_sh.stride(1),
        B_scale_sh.stride(0), B_scale_sh.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_SIZE_M=cfg['GSM'], NUM_XCDS=8, mfma_nonkdim=16,
        num_warps=cfg['NW'], num_stages=cfg['NS'],
        matrix_instr_nonkdim=16,
    )
    return C


def _launch_fused(A_bf16, B_q, B_scale_shuffled, m, n, k, cfg):
    """Launch fused A-quant + GEMM (single kernel, K=512 only)."""
    BM, BN, BK = cfg['BM'], cfg['BN'], cfg['BK']
    key = (m, n)
    if key not in _c_cache:
        _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[key]

    grid = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
    _fused_gemm_kernel[grid](
        A_bf16, B_q, C, B_scale_shuffled,
        m, n, k,
        A_bf16.stride(0), A_bf16.stride(1),
        B_q.stride(1), B_q.stride(0),
        C.stride(0), C.stride(1),
        B_scale_shuffled.stride(0), B_scale_shuffled.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_SIZE_M=cfg['GSM'], NUM_XCDS=8, mfma_nonkdim=16,
        num_warps=cfg['NW'], num_stages=cfg['NS'],
        matrix_instr_nonkdim=16,
    )
    return C


def _launch_splitk(A_fp4, A_scale_sh, B_q, B_scale_sh, m, n, k, cfg):
    """Launch split-K AFP4 GEMM: splitK kernel + reduce kernel."""
    BM, BN, BK = cfg['BM'], cfg['BN'], cfg['BK']
    KSPLIT = cfg['KSPLIT']
    K_PER_SPLIT = k // KSPLIT

    # Allocate or reuse workspace for fp32 partials [KSPLIT, M, N]
    pp_key = (KSPLIT, m, n)
    if pp_key not in _pp_cache:
        _pp_cache[pp_key] = torch.empty((KSPLIT, m, n), dtype=torch.float32, device='cuda')
    pp = _pp_cache[pp_key]

    # Allocate or reuse bf16 output
    c_key = (m, n)
    if c_key not in _c_cache:
        _c_cache[c_key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[c_key]

    tiles_per_split = triton.cdiv(m, BM) * triton.cdiv(n, BN)
    total_blocks = tiles_per_split * KSPLIT

    # Launch split-K GEMM kernel
    _splitk_gemm_kernel[(total_blocks,)](
        A_fp4, B_q, pp, A_scale_sh, B_scale_sh,
        m, n, k,
        A_fp4.stride(0), A_fp4.stride(1),
        B_q.stride(1), B_q.stride(0),
        pp.stride(0), pp.stride(1), pp.stride(2),
        A_scale_sh.stride(0), A_scale_sh.stride(1),
        B_scale_sh.stride(0), B_scale_sh.stride(1),
        K_PER_SPLIT=K_PER_SPLIT,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_SIZE_M=cfg['GSM'], NUM_XCDS=8,
        NUM_KSPLIT=KSPLIT,
        mfma_nonkdim=16,
        num_warps=cfg['NW'], num_stages=cfg['NS'],
        matrix_instr_nonkdim=16,
    )

    # Launch reduce kernel: sum partials -> bf16
    reduce_tiles = triton.cdiv(m, BM) * triton.cdiv(n, BN)
    _splitk_reduce_kernel[(reduce_tiles,)](
        pp, C,
        m, n,
        pp.stride(0), pp.stride(1), pp.stride(2),
        C.stride(0), C.stride(1),
        NUM_KSPLIT=KSPLIT,
        BLOCK_M=BM, BLOCK_N=BN,
        num_warps=4, num_stages=1,
    )
    return C


# ---------------------------------------------------------------------------
# aiter fallback paths
# ---------------------------------------------------------------------------

_K7168_AITER_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _fallback(A, B_q_u8, B_scale_raw, m, n, k):
    """Fallback to aiter library for K=1536 and error cases."""
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), B_q_u8, A_scale, B_scale_raw,
                             dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = _K7168_AITER_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16, config=cfg)


# ---------------------------------------------------------------------------
# Pre-warm
# ---------------------------------------------------------------------------

def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    # Warm K=512 AFP4 path (non-splitK)
    for m, n, k in [(4, 2880, 512), (32, 4096, 512), (32, 2880, 512)]:
        try:
            cfg = _get_cfg(m, n, k)
            wA = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
            wA_fp4, wA_sc = dynamic_mxfp4_quant(wA)
            wA_sc_sh = _shuffle_scales_cdna4(wA_sc)
            wBq = torch.zeros((n, k // 2), dtype=torch.uint8, device='cuda')
            wBs_sh = _shuffle_scales_cdna4(
                torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda')
            )
            _launch_nonsplitk(
                wA_fp4.view(torch.uint8), wA_sc_sh, wBq, wBs_sh, m, n, k, cfg
            )
        except Exception:
            pass

    # Warm K=512 fused path if available
    if _HAS_INLINE_QUANT:
        for m, n, k in [(32, 4096, 512)]:
            try:
                cfg = _get_cfg(m, n, k)
                wA = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
                wBq = torch.zeros((n, k // 2), dtype=torch.uint8, device='cuda')
                wBs_sh = _shuffle_scales_cdna4(
                    torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda')
                )
                _launch_fused(wA, wBq, wBs_sh, m, n, k, cfg)
            except Exception:
                pass

    # Warm split-K paths for K=7168 and K=2048
    for m, n, k in [(16, 2112, 7168), (64, 7168, 2048)]:
        try:
            cfg = _get_cfg(m, n, k)
            wA = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
            wA_fp4, wA_sc = dynamic_mxfp4_quant(wA)
            wA_sc_sh = _shuffle_scales_cdna4(wA_sc)
            wBq = torch.zeros((n, k // 2), dtype=torch.uint8, device='cuda')
            wBs_sh = _shuffle_scales_cdna4(
                torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda')
            )
            _launch_splitk(
                wA_fp4.view(torch.uint8), wA_sc_sh, wBq, wBs_sh, m, n, k, cfg
            )
        except Exception:
            pass

    # Warm K=1536 aiter fallback (quant kernel only)
    for wm in (256,):
        try:
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
        except Exception:
            pass

    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_sh_flat, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bq_u8 = B_q.view(torch.uint8)
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bscale_sh_flat = _shuffle_scales_cdna4(_bscale_raw)
        _prewarm()

    cfg = _get_cfg(m, n, k)
    ksplit = cfg.get('KSPLIT', 1)

    # K=1536: always use aiter afp4wfp4 (proven fastest)
    if k == 1536:
        return _fallback(A, _bq_u8, _bscale_raw, m, n, k)

    # K=7168 or K=2048: split-K custom kernel
    if ksplit > 1 and k in (7168, 2048):
        try:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            A_scale_sh = _shuffle_scales_cdna4(A_scale)
            return _launch_splitk(
                A_fp4.view(torch.uint8), A_scale_sh,
                _bq_u8, _bscale_sh_flat,
                m, n, k, cfg,
            )
        except Exception:
            return _fallback(A, _bq_u8, _bscale_raw, m, n, k)

    # K=512: non-splitK custom kernel
    if k == 512:
        try:
            if _HAS_INLINE_QUANT:
                return _launch_fused(A, _bq_u8, _bscale_sh_flat, m, n, k, cfg)
            else:
                from aiter.ops.triton.quant import dynamic_mxfp4_quant
                A_fp4, A_scale = dynamic_mxfp4_quant(A)
                A_scale_sh = _shuffle_scales_cdna4(A_scale)
                return _launch_nonsplitk(
                    A_fp4.view(torch.uint8), A_scale_sh,
                    _bq_u8, _bscale_sh_flat,
                    m, n, k, cfg,
                )
        except Exception:
            pass

    # Fallback for any other shape or error
    return _fallback(A, _bq_u8, _bscale_raw, m, n, k)
