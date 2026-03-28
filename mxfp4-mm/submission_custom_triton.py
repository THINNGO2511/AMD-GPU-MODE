#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Custom Triton MXFP4 GEMM kernel from scratch.

Two kernel paths:
1. _fused_gemm: Fused A quant + GEMM, single kernel launch (all shapes)
2. _fused_splitk_gemm + _reduce: Split-K for large K / small M (K=7168, K=2048)

Key techniques:
- tl.dot_scaled with fused accumulator form for MFMA FP4 on gfx950
- XCD-aware tile swizzle for MI355X (8 XCDs, 304 CUs)
- Per-shape hardcoded configs (no autotune JIT penalty)
- Grouped tile ordering for L2 cache reuse
- Output tensor caching + pre-warmed JIT for all shapes
- _mxfp4_quant_op for exact-match A quantization

Benchmark shapes:
  (M=4,   N=2880, K=512)   -- tiny M, fused
  (M=16,  N=2112, K=7168)  -- small M, very large K -> split-K=8
  (M=32,  N=4096, K=512)   -- medium M, small K, fused
  (M=32,  N=2880, K=512)   -- medium M, small K, fused
  (M=64,  N=7168, K=2048)  -- medium M, large K -> split-K=2
  (M=256, N=3072, K=1536)  -- large M, medium K, fused
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
# Try multiple import paths for _mxfp4_quant_op
_HAS_QUANT_OP = False
try:
    from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
    _HAS_QUANT_OP = True
except ImportError:
    try:
        from aiter.ops.triton.quant import _mxfp4_quant_op
        _HAS_QUANT_OP = True
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Global caches
# ---------------------------------------------------------------------------
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_c_cache = {}       # (m, n) -> output tensor
_pp_cache = {}      # (num_splits, m, n) -> partial-products tensor
_warmed = False

# ---------------------------------------------------------------------------
# Per-shape tuned configs
# ---------------------------------------------------------------------------
# MI355X: 304 CUs, 8 XCDs (38 CUs/XCD), 5300 GB/s HBM
# BLOCK_K must be >= 128 and power of 2 (blockscale alignment + tl.arange)
# For small M: maximize N-parallelism with small BN, use split-K for K
# For large M: use larger BM to amortize overheads

_SHAPE_CONFIGS = {
    # K=512 shapes: fused, no split-K (enough tiles from N)
    # M=4: ceil(4/16)=1 M-tiles. N=2880/32=90 N-tiles. 90 blocks. K=512/128=4 iters.
    (4, 2880, 512):   dict(BM=16, BN=32,  BK=128, GSM=1, NW=4, NS=2, WPE=2, KS=1),
    # M=32: 1 M-tile. N=4096/64=64 N-tiles. 64 blocks.
    (32, 4096, 512):  dict(BM=32, BN=64,  BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=1),
    # M=32: 1 M-tile. N=2880/64=45 N-tiles. 45 blocks.
    (32, 2880, 512):  dict(BM=32, BN=64,  BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=1),

    # K=7168: split-K essential. M=16, ceil(16/32)=1, ceil(2112/64)=33. 33 tiles.
    # Split-K=8: 33*8 = 264 blocks. K/8 = 896 per split. BK=128 -> 7 iters/split.
    (16, 2112, 7168): dict(BM=32, BN=64,  BK=128, GSM=1, NW=4, NS=2, WPE=2, KS=8),

    # K=2048: M=64, ceil(64/64)=1, ceil(7168/128)=56. 56 tiles.
    # Split-K=2: 56*2 = 112 blocks. K/2 = 1024 per split. BK=128 -> 8 iters/split.
    (64, 7168, 2048): dict(BM=64, BN=128, BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=2),

    # K=1536: M=256, ceil(256/64)=4, ceil(3072/128)=24. 96 blocks. No split needed.
    (256, 3072, 1536): dict(BM=64, BN=128, BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=1),
}

_DEFAULT_CONFIG = dict(BM=32, BN=128, BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=1)


def _get_config(m, n, k):
    """Look up per-shape config or compute a reasonable default."""
    cfg = _SHAPE_CONFIGS.get((m, n, k))
    if cfg is not None:
        return cfg

    # Heuristic fallback
    bm = 32 if m <= 32 else (64 if m <= 128 else 128)
    bn = 64 if n <= 2048 else 128
    bk = 128

    num_blocks = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    ks = 1
    if num_blocks < 64 and k > 512:
        ks = min(8, max(1, 304 // num_blocks))
        while ks > 1 and triton.cdiv(k, ks) < bk:
            ks //= 2

    return dict(BM=bm, BN=bn, BK=bk, GSM=4, NW=4, NS=2, WPE=2, KS=ks)


# ---------------------------------------------------------------------------
# Scale unshuffle (undo e8m0_shuffle applied during B quantization)
# ---------------------------------------------------------------------------
def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# ===========================================================================
# KERNEL 1: Fused A-quant + GEMM (no split-K)
# ===========================================================================
@triton.jit
def _fused_gemm_kernel(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
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

    # XCD-aware swizzle for MI355X
    if NUM_XCDS > 1:
        pids_per_xcd = total_pids // NUM_XCDS
        extra = total_pids % NUM_XCDS
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        pid = xcd * pids_per_xcd + tl.minimum(xcd, extra) + local_pid

    # Grouped tile ordering for L2 reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop: iterate over blocks of K
    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K

        # Load A [BLOCK_M, BLOCK_K] bf16 -> fp32
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
        a_tile = tl.load(a_ptrs).to(tl.float32)

        # Quantize A -> FP4 + E8M0 scales (exact aiter match)
        a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)

        # Load B [BLOCK_K//2, BLOCK_N] transposed via strides
        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        b_ptrs = b_ptr + (k_start // 2 + offs_k_packed)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_fp4 = tl.load(b_ptrs)

        # Load B scales [BLOCK_N, BLOCK_K//32]
        offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
        bs_ptrs = b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_bsk
        b_scales = tl.load(bs_ptrs)

        # FP4 GEMM (fused accumulator form -> compiles to MFMA FP4)
        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

    # Store [BLOCK_M, BLOCK_N] as bf16
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c, mask=c_mask)


# ===========================================================================
# KERNEL 2: Fused A-quant + Split-K GEMM
# Each program handles ITERS_PER_SPLIT iterations of K, writes fp32 partials
# ===========================================================================
@triton.jit
def _fused_splitk_gemm_kernel(
    a_ptr, b_ptr, pp_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_ppk, stride_ppm, stride_ppn,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    ITERS_PER_SPLIT: tl.constexpr,  # number of BLOCK_K iters per split
    NUM_XCDS: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_split = num_pid_m * num_pid_n

    # Which K-split and which tile
    split_id = pid // tiles_per_split
    tile_id = pid % tiles_per_split

    # XCD swizzle within split
    if NUM_XCDS > 1:
        pids_per_xcd = tiles_per_split // NUM_XCDS
        extra = tiles_per_split % NUM_XCDS
        xcd = tile_id % NUM_XCDS
        local_pid = tile_id // NUM_XCDS
        tile_id = xcd * pids_per_xcd + tl.minimum(xcd, extra) + local_pid

    # Grouped ordering
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K range for this split
    k_base = split_id * ITERS_PER_SPLIT * BLOCK_K

    for local_iter in range(ITERS_PER_SPLIT):
        k_start = k_base + local_iter * BLOCK_K

        # Guard: skip if this iteration goes past K
        # (last split may have fewer valid iterations)
        if k_start < K:
            # Load A
            offs_k = tl.arange(0, BLOCK_K)
            k_mask = (k_start + offs_k)[None, :] < K
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
            a_tile = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)

            # Load B
            offs_k_packed = tl.arange(0, BLOCK_K // 2)
            b_ptrs = b_ptr + (k_start // 2 + offs_k_packed)[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_fp4 = tl.load(b_ptrs, mask=((k_start // 2 + offs_k_packed)[:, None] < (K // 2)), other=0)

            # Load B scales
            offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
            bs_ptrs = b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_bsk
            b_scales = tl.load(bs_ptrs, mask=((k_start // SCALE_GROUP + offs_k_scale)[None, :] < tl.cdiv(K, SCALE_GROUP)), other=0)

            acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

    # Store partial fp32 results
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    pp_ptrs = pp_ptr + split_id * stride_ppk + offs_cm[:, None] * stride_ppm + offs_cn[None, :] * stride_ppn
    tl.store(pp_ptrs, acc, mask=c_mask)


# ===========================================================================
# KERNEL 3: Split-K reduction (sum partials -> bf16 output)
# ===========================================================================
@triton.jit
def _splitk_reduce_kernel(
    pp_ptr, c_ptr,
    M, N,
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
        pp_ptrs = pp_ptr + ks * stride_ppk + offs_m[:, None] * stride_ppm + offs_n[None, :] * stride_ppn
        partial = tl.load(pp_ptrs, mask=mask, other=0.0)
        acc += partial

    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :].to(tl.int64) * stride_cn
    tl.store(c_ptrs, c, mask=mask)


# ===========================================================================
# Launcher functions
# ===========================================================================
def _launch_fused(A, B_q_u8, B_scale_raw, m, n, k, cfg):
    """Launch fused A-quant + GEMM (no split-K)."""
    BM, BN, BK = cfg['BM'], cfg['BN'], cfg['BK']

    c_key = (m, n)
    if c_key not in _c_cache:
        _c_cache[c_key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[c_key]

    grid = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
    _fused_gemm_kernel[grid](
        A, B_q_u8, C, B_scale_raw,
        m, n, k,
        A.stride(0), A.stride(1),
        B_q_u8.stride(0), B_q_u8.stride(1),
        C.stride(0), C.stride(1),
        B_scale_raw.stride(0), B_scale_raw.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_SIZE_M=cfg['GSM'],
        NUM_XCDS=8,
        num_warps=cfg['NW'], num_stages=cfg['NS'],
        waves_per_eu=cfg['WPE'],
        matrix_instr_nonkdim=16,
    )
    return C


def _launch_splitk(A, B_q_u8, B_scale_raw, m, n, k, cfg):
    """Launch fused A-quant + split-K GEMM with reduction."""
    BM, BN, BK = cfg['BM'], cfg['BN'], cfg['BK']
    KS = cfg['KS']

    # Compute iterations per split (compile-time constant per config)
    # Each split handles ceil(K / KS / BK) iterations of BK
    total_k_iters = triton.cdiv(k, BK)
    iters_per_split = triton.cdiv(total_k_iters, KS)

    # Allocate partial products
    pp_key = (KS, m, n)
    if pp_key not in _pp_cache:
        _pp_cache[pp_key] = torch.empty((KS, m, n), dtype=torch.float32, device='cuda')
    pp = _pp_cache[pp_key]

    tiles_per_split = triton.cdiv(m, BM) * triton.cdiv(n, BN)
    grid_gemm = (KS * tiles_per_split,)

    _fused_splitk_gemm_kernel[grid_gemm](
        A, B_q_u8, pp, B_scale_raw,
        m, n, k,
        A.stride(0), A.stride(1),
        B_q_u8.stride(0), B_q_u8.stride(1),
        pp.stride(0), pp.stride(1), pp.stride(2),
        B_scale_raw.stride(0), B_scale_raw.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_SIZE_M=cfg['GSM'],
        NUM_KSPLIT=KS,
        ITERS_PER_SPLIT=iters_per_split,
        NUM_XCDS=8,
        num_warps=cfg['NW'], num_stages=cfg['NS'],
        waves_per_eu=cfg['WPE'],
        matrix_instr_nonkdim=16,
    )

    # Reduction
    c_key = (m, n)
    if c_key not in _c_cache:
        _c_cache[c_key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[c_key]

    RED_BM = min(BM, 32)
    RED_BN = min(BN, 128)
    grid_reduce = (triton.cdiv(m, RED_BM) * triton.cdiv(n, RED_BN),)
    _splitk_reduce_kernel[grid_reduce](
        pp, C, m, n,
        pp.stride(0), pp.stride(1), pp.stride(2),
        C.stride(0), C.stride(1),
        NUM_KSPLIT=KS,
        BLOCK_M=RED_BM, BLOCK_N=RED_BN,
        num_warps=4, num_stages=1,
    )
    return C


# ===========================================================================
# Pre-warm all kernel variants
# ===========================================================================
def _prewarm_all():
    global _warmed
    if _warmed:
        return
    _warmed = True

    # Warm all benchmark shapes
    shapes = [
        (4, 2880, 512),
        (16, 2112, 7168),
        (32, 4096, 512),
        (32, 2880, 512),
        (64, 7168, 2048),
        (256, 3072, 1536),
    ]

    for m, n, k in shapes:
        try:
            cfg = _get_config(m, n, k)
            wA = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
            wBq = torch.zeros((n, k // 2), dtype=torch.uint8, device='cuda')
            wBs = torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda')

            if cfg['KS'] > 1:
                _launch_splitk(wA, wBq, wBs, m, n, k, cfg)
            else:
                _launch_fused(wA, wBq, wBs, m, n, k, cfg)
        except Exception:
            pass

    torch.cuda.synchronize()


# ===========================================================================
# Main entry point
# ===========================================================================
_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _fallback_kernel(A, B_q_u8, B_scale_raw, m, n, k):
    """Fallback to proven aiter wrappers if custom kernel fails."""
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), B_q_u8, A_scale, B_scale_raw,
                             dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = _K7168_CONFIG if k == 7168 else None
    key = (m, n)
    if key not in _c_cache:
        _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16,
                       y=_c_cache[key], config=cfg)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Cache B preprocessing (done once per unique B)
    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        if _HAS_QUANT_OP:
            _prewarm_all()

    if not _HAS_QUANT_OP:
        return _fallback_kernel(A, _bq_u8, _bscale_raw, m, n, k)

    cfg = _get_config(m, n, k)

    try:
        if cfg['KS'] > 1:
            return _launch_splitk(A, _bq_u8, _bscale_raw, m, n, k, cfg)
        else:
            return _launch_fused(A, _bq_u8, _bscale_raw, m, n, k, cfg)
    except Exception:
        return _fallback_kernel(A, _bq_u8, _bscale_raw, m, n, k)
