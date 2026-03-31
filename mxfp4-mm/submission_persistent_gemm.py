#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Persistent MXFP4 GEMM for AMD MI355X (256 CUs, 8 XCDs).

Key insight: Small-M shapes (M=4/16/32/64) produce very few output tiles,
causing wave quantization — e.g. M=4,N=2880,BM=32,BN=64 = 1*45 = 45 tiles
for 256 CUs. 82% of CUs sit idle.

Persistent kernel: Launch exactly NUM_CUS programs. Each program loops over
its assigned tiles using tl.range(start_pid, num_tiles, NUM_CUS). This:
  1. Eliminates wave quantization (all 256 CUs always active)
  2. Eliminates kernel re-launch overhead between tiles
  3. Improves L2 cache reuse (program stays resident, reuses B data)

For large K (7168, 2048): Use split-K within the persistent loop. Each tile
is subdivided into K_SPLIT chunks, giving more virtual tiles to distribute.

Grouped tile ordering: Cluster M-tiles for L2 cache reuse of B weight data.

Benchmark shapes:
  (M=4,   N=2880, K=512)   — 45 tiles @ BN=64  -> persistent helps most
  (M=16,  N=2112, 7168)    — 33 tiles @ BN=64  -> split-K=4 -> 132 vtiles
  (M=32,  N=4096, K=512)   — 64 tiles @ BN=64  -> persistent helps
  (M=32,  N=2880, K=512)   — 45 tiles @ BN=64  -> persistent helps
  (M=64,  N=7168, K=2048)  — 56 tiles @ BN=128 -> split-K=2 -> 112 vtiles
  (M=256, N=3072, K=1536)  — 96 tiles @ BN=32  -> persistent moderate help
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl

# Try importing the fused quant op (Triton JIT function)
_HAS_QUANT_OP = False
try:
    from aiter.ops.triton.quant import _mxfp4_quant_op
    _HAS_QUANT_OP = True
except ImportError:
    try:
        from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
        _HAS_QUANT_OP = True
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CUS = 256       # MI355X: 256 usable CUs (304 physical, 256 visible)
NUM_XCDS = 8        # 8 XCDs, 32 CUs per XCD

# ---------------------------------------------------------------------------
# Global caches
# ---------------------------------------------------------------------------
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_c_cache = {}
_pp_cache = {}
_warmed = False


def _unshuffle_e8m0(scale_sh):
    """Inverse of e8m0_shuffle: recover raw E8M0 scales from shuffled layout."""
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# ---------------------------------------------------------------------------
# Helper: Grouped tile ordering (Triton JIT)
# ---------------------------------------------------------------------------
@triton.jit
def _compute_tile_pid(tile_id, num_pid_n, num_pid_m,
                      GROUP_SIZE_M: tl.constexpr):
    """
    Map linear tile_id to (pid_m, pid_n) with grouped ordering.
    Groups GROUP_SIZE_M rows together for L2 cache reuse of B tiles.
    """
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ===========================================================================
# KERNEL 1: Persistent fused quant + GEMM (no split-K)
#
# Design: Launch NUM_CUS programs. Each program loops over tiles with
# stride NUM_CUS using tl.range. This eliminates wave quantization:
# even if there are only 45 tiles, all 45 CUs that get work are active
# from the start (no second wave launch overhead).
#
# For tile counts < NUM_CUS, excess programs immediately exit the loop.
# For tile counts >= NUM_CUS, each program processes ceil(tiles/NUM_CUS) tiles.
# ===========================================================================
@triton.jit
def _persistent_gemm_kernel(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,       # B is [N, K//2] uint8
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,     # B_scale is [N, K//32] uint8
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_CUS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    # Persistent loop: each CU processes tiles start_pid, start_pid+NUM_CUS, ...
    # tl.range enables compiler pipelining across iterations
    for tile_id in tl.range(start_pid, num_tiles, NUM_CUS):
        # Map linear tile_id to (pid_m, pid_n) with grouped ordering
        # Grouped ordering clusters M-tiles together for B cache reuse
        pid_m, pid_n = _compute_tile_pid(tile_id, num_pid_n, num_pid_m,
                                         GROUP_SIZE_M)

        # Compute tile offsets with clamping for out-of-bounds safety
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_safe = tl.where(offs_m < M, offs_m, 0)
        offs_n_safe = tl.where(offs_n < N, offs_n, 0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K-loop: full reduction across K dimension
        for k_iter in range(k_tiles):
            k_start = k_iter * BLOCK_K

            # Load A [BLOCK_M, BLOCK_K] bf16
            a_ptrs = a_ptr + offs_m_safe[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak
            a_tile = tl.load(a_ptrs).to(tl.float32)

            # Fused A quantization: bf16 -> MXFP4 + E8M0 scales
            a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)

            # Load B_q [BLOCK_K//2, BLOCK_N] (transposed view via strides)
            b_ptrs = b_ptr + (k_start // 2 + tl.arange(0, BLOCK_K // 2))[:, None] * stride_bk + offs_n_safe[None, :] * stride_bn
            b_fp4 = tl.load(b_ptrs)

            # Load B_scale [BLOCK_N, BLOCK_K//32]
            bs_ptrs = b_scales_ptr + offs_n_safe[:, None] * stride_bsn + (k_start // SCALE_GROUP + tl.arange(0, BLOCK_K // SCALE_GROUP))[None, :] * stride_bsk
            b_scales = tl.load(bs_ptrs)

            # MFMA FP4 GEMM — fused accumulator form (no extra VALU ops)
            acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

        # Store result [BLOCK_M, BLOCK_N] as bf16
        c = acc.to(tl.bfloat16)
        offs_cm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
                 c, mask=c_mask)


# ===========================================================================
# KERNEL 2: Persistent fused quant + Split-K GEMM (partial products)
# ===========================================================================
@triton.jit
def _persistent_splitk_gemm_kernel(
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
    ITERS_PER_SPLIT: tl.constexpr,
    NUM_CUS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_split = num_pid_m * num_pid_n
    # Total virtual tiles = spatial tiles * K splits
    total_vtiles = tiles_per_split * NUM_KSPLIT

    # Persistent loop over virtual tiles (spatial * K_splits)
    for vtile_id in tl.range(start_pid, total_vtiles, NUM_CUS):
        # Decompose: which K split and which spatial tile
        split_id = vtile_id // tiles_per_split
        spatial_id = vtile_id % tiles_per_split

        # Map to (pid_m, pid_n) with grouped ordering
        pid_m, pid_n = _compute_tile_pid(spatial_id, num_pid_n, num_pid_m,
                                         GROUP_SIZE_M)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_safe = tl.where(offs_m < M, offs_m, 0)
        offs_n_safe = tl.where(offs_n < N, offs_n, 0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K range for this split
        k_base = split_id * ITERS_PER_SPLIT * BLOCK_K

        for local_k in range(ITERS_PER_SPLIT):
            k_start = k_base + local_k * BLOCK_K
            # Guard: last split may have fewer valid iterations
            if k_start < K:
                # Load A
                a_ptrs = a_ptr + offs_m_safe[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak
                k_mask = (k_start + tl.arange(0, BLOCK_K))[None, :] < K
                a_tile = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)

                a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)

                # Load B
                b_ptrs = b_ptr + (k_start // 2 + tl.arange(0, BLOCK_K // 2))[:, None] * stride_bk + offs_n_safe[None, :] * stride_bn
                b_mask = (k_start // 2 + tl.arange(0, BLOCK_K // 2))[:, None] < (K // 2)
                b_fp4 = tl.load(b_ptrs, mask=b_mask, other=0)

                # Load B scales
                bs_ptrs = b_scales_ptr + offs_n_safe[:, None] * stride_bsn + (k_start // SCALE_GROUP + tl.arange(0, BLOCK_K // SCALE_GROUP))[None, :] * stride_bsk
                bs_mask = (k_start // SCALE_GROUP + tl.arange(0, BLOCK_K // SCALE_GROUP))[None, :] < tl.cdiv(K, SCALE_GROUP)
                b_scales = tl.load(bs_ptrs, mask=bs_mask, other=0)

                acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

        # Store partial fp32 result
        offs_cm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        pp_ptrs = pp_ptr + split_id * stride_ppk + offs_cm[:, None] * stride_ppm + offs_cn[None, :] * stride_ppn
        tl.store(pp_ptrs, acc, mask=c_mask)


# ===========================================================================
# KERNEL 3: Split-K reduction (sum partials -> bf16 output)
# Also persistent — loops over output tiles
# ===========================================================================
@triton.jit
def _persistent_reduce_kernel(
    pp_ptr, c_ptr,
    M, N,
    stride_ppk, stride_ppm, stride_ppn,
    stride_cm, stride_cn,
    NUM_KSPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CUS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_CUS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

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
# Per-shape configs
# ===========================================================================
# MI355X: 256 CUs, 8 XCDs (32 CUs/XCD), HBM 5300 GB/s
# BLOCK_K must be >= 128 (blockscale alignment, MFMA FP4 K=64 minimum)
# For bandwidth-bound small-M: want many small tiles for parallelism

_SHAPE_CONFIGS = {
    # K=512 shapes: fused, no split-K
    # M=4:  BM=16 -> 1 M-tile. BN=64 -> 45 N-tiles. 45 tiles total.
    #        Persistent: all 45 tiles distributed across 256 CUs (no waste).
    (4, 2880, 512):   dict(BM=16, BN=64,  BK=128, GSM=1, NW=4, NS=2, WPE=2, KS=1),
    # M=32, N=4096: BM=32, BN=64 -> 1*64=64 tiles. Persistent distributes evenly.
    (32, 4096, 512):  dict(BM=32, BN=64,  BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=1),
    # M=32, N=2880: BM=32, BN=64 -> 1*45=45 tiles.
    (32, 2880, 512):  dict(BM=32, BN=64,  BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=1),

    # K=7168: M=16, BM=32, BN=64 -> 1*33=33 tiles. WAY too few.
    # Split-K=8: 33*8=264 virtual tiles. K/8=896 per split, BK=128 -> 7 iters.
    (16, 2112, 7168): dict(BM=32, BN=64,  BK=128, GSM=1, NW=4, NS=2, WPE=2, KS=8),

    # K=2048: M=64, BM=64, BN=128 -> 1*56=56 tiles.
    # Split-K=2: 56*2=112 virtual tiles. K/2=1024, BK=128 -> 8 iters.
    (64, 7168, 2048): dict(BM=64, BN=128, BK=128, GSM=4, NW=4, NS=2, WPE=2, KS=2),

    # K=1536: Use aiter fallback (proven faster, see CLAUDE.md dead ends)
    (256, 3072, 1536): None,  # sentinel: use fallback
}


def _get_config(m, n, k):
    """Look up per-shape config. Returns None for fallback shapes."""
    cfg = _SHAPE_CONFIGS.get((m, n, k))
    if cfg is not None or (m, n, k) in _SHAPE_CONFIGS:
        return cfg

    # Heuristic for unseen shapes
    bm = 16 if m <= 4 else (32 if m <= 64 else 64)
    bn = 64 if n <= 4096 else 128
    bk = 128

    num_tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    ks = 1
    if num_tiles < 64 and k > 512:
        # Need more parallelism — use split-K
        ks = min(8, max(1, NUM_CUS // num_tiles))
        # Ensure each split has at least 1 BK iteration
        while ks > 1 and triton.cdiv(k, ks) < bk:
            ks //= 2

    return dict(BM=bm, BN=bn, BK=bk, GSM=4, NW=4, NS=2, WPE=2, KS=ks)


# ===========================================================================
# Launcher functions
# ===========================================================================
def _launch_persistent(A, B_q_u8, B_scale_raw, m, n, k, cfg):
    """Launch persistent fused A-quant + GEMM (no split-K)."""
    BM, BN, BK = cfg['BM'], cfg['BN'], cfg['BK']

    c_key = (m, n)
    if c_key not in _c_cache:
        _c_cache[c_key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[c_key]

    num_tiles = triton.cdiv(m, BM) * triton.cdiv(n, BN)
    # Launch min(NUM_CUS, num_tiles) programs — no point launching idle CUs
    grid_size = min(NUM_CUS, num_tiles)

    _persistent_gemm_kernel[(grid_size,)](
        A, B_q_u8, C, B_scale_raw,
        m, n, k,
        A.stride(0), A.stride(1),
        B_q_u8.stride(0), B_q_u8.stride(1),
        C.stride(0), C.stride(1),
        B_scale_raw.stride(0), B_scale_raw.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_SIZE_M=cfg['GSM'],
        NUM_CUS=grid_size,
        NUM_XCDS=NUM_XCDS,
        num_warps=cfg['NW'], num_stages=cfg['NS'],
        waves_per_eu=cfg['WPE'],
        matrix_instr_nonkdim=16,
    )
    return C


def _launch_persistent_splitk(A, B_q_u8, B_scale_raw, m, n, k, cfg):
    """Launch persistent fused A-quant + split-K GEMM with reduction."""
    BM, BN, BK = cfg['BM'], cfg['BN'], cfg['BK']
    KS = cfg['KS']

    # Compute iterations per split (must be constexpr-compatible)
    total_k_iters = triton.cdiv(k, BK)
    iters_per_split = triton.cdiv(total_k_iters, KS)

    # Allocate partial products [KS, M, N] fp32
    pp_key = (KS, m, n)
    if pp_key not in _pp_cache:
        _pp_cache[pp_key] = torch.empty((KS, m, n), dtype=torch.float32, device='cuda')
    pp = _pp_cache[pp_key]

    tiles_per_split = triton.cdiv(m, BM) * triton.cdiv(n, BN)
    total_vtiles = tiles_per_split * KS
    grid_gemm = min(NUM_CUS, total_vtiles)

    _persistent_splitk_gemm_kernel[(grid_gemm,)](
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
        NUM_CUS=grid_gemm,
        NUM_XCDS=NUM_XCDS,
        num_warps=cfg['NW'], num_stages=cfg['NS'],
        waves_per_eu=cfg['WPE'],
        matrix_instr_nonkdim=16,
    )

    # Persistent reduction
    c_key = (m, n)
    if c_key not in _c_cache:
        _c_cache[c_key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    C = _c_cache[c_key]

    RED_BM = min(BM, 32)
    RED_BN = min(BN, 128)
    num_red_tiles = triton.cdiv(m, RED_BM) * triton.cdiv(n, RED_BN)
    grid_reduce = min(NUM_CUS, num_red_tiles)

    _persistent_reduce_kernel[(grid_reduce,)](
        pp, C, m, n,
        pp.stride(0), pp.stride(1), pp.stride(2),
        C.stride(0), C.stride(1),
        NUM_KSPLIT=KS,
        BLOCK_M=RED_BM, BLOCK_N=RED_BN,
        NUM_CUS=grid_reduce,
        num_warps=4, num_stages=1,
    )
    return C


# ===========================================================================
# Pre-warm all kernel variants
# ===========================================================================
def _prewarm_all():
    """Pre-warm Triton JIT for all benchmark shapes to kill JIT penalty."""
    global _warmed
    if _warmed:
        return
    _warmed = True

    shapes = [
        (4, 2880, 512),
        (16, 2112, 7168),
        (32, 4096, 512),
        (32, 2880, 512),
        (64, 7168, 2048),
    ]

    for m, n, k in shapes:
        cfg = _get_config(m, n, k)
        if cfg is None:
            continue
        try:
            wA = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
            wBq = torch.zeros((n, k // 2), dtype=torch.uint8, device='cuda')
            wBs = torch.full((n, k // 32), 127, dtype=torch.uint8, device='cuda')
            if cfg['KS'] > 1:
                _launch_persistent_splitk(wA, wBq, wBs, m, n, k, cfg)
            else:
                _launch_persistent(wA, wBq, wBs, m, n, k, cfg)
        except Exception:
            pass

    # Also warm aiter for K=1536 fallback
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        for wm in (4, 16, 32, 64, 256):
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
    except Exception:
        pass

    torch.cuda.synchronize()


# ===========================================================================
# Fallback to proven aiter kernels
# ===========================================================================
_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _fallback(A, B_q_u8, B_scale_raw, m, n, k):
    """Fallback to proven aiter wrappers."""
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), B_q_u8, A_scale, B_scale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _c_cache:
        _c_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    cfg = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16,
                       y=_c_cache[key], config=cfg)


# ===========================================================================
# Main entry point
# ===========================================================================
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

    # Get per-shape config
    cfg = _get_config(m, n, k)

    # Fallback for K=1536 or if quant op not available
    if cfg is None or not _HAS_QUANT_OP:
        return _fallback(A, _bq_u8, _bscale_raw, m, n, k)

    # Persistent kernel path
    try:
        if cfg['KS'] > 1:
            return _launch_persistent_splitk(A, _bq_u8, _bscale_raw, m, n, k, cfg)
        else:
            return _launch_persistent(A, _bq_u8, _bscale_raw, m, n, k, cfg)
    except Exception:
        # Fallback on any error
        return _fallback(A, _bq_u8, _bscale_raw, m, n, k)
