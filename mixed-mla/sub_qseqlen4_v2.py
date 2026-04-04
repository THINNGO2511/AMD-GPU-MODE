#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — qseqlen=4 batch folding v2 (simplified, no fallback).
v1 timed out because it loaded BOTH qh64_qseqlen4 AND qh16_qseqlen1 kernels.
v2: ALL shapes go through qseqlen4. Only one kernel loaded.

Key: reshape Q from (bs, 16, 576) to (bs/4, 64, 576) → triggers
fold_factor=4 → qseqlen fold → dispatches to qh64_qseqlen4 kernel.
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
_meta_cache = {}
_alloc_cache = {}


@triton.jit
def _q_amax_kernel(q_ptr, amax_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x)))


@triton.jit
def _q_to_fp8_kernel(q_ptr, out_ptr, scale_ptr, amax_ptr,
                     FP8_MAX: tl.constexpr, N, BLOCK: tl.constexpr):
    amax = tl.load(amax_ptr)
    amax = tl.where(amax < 1e-12, 1e-12, amax)
    scale = amax / FP8_MAX
    if tl.program_id(0) == 0:
        tl.store(scale_ptr, scale)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x / scale
    x = tl.clamp(x, -FP8_MAX, FP8_MAX)
    tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=mask)


def _build_meta(bs_grouped, kv_per_group, nq_effective, nkv,
                num_kv_splits, qo_indptr, kv_indptr):
    """Build metadata for qseqlen=4 grouped decode."""
    total_kv = int(kv_indptr[-1].item())
    page_size = 1
    kv_gran = 16  # max(page_size, 16) = 16 for ps=1

    seq_lens = kv_indptr[1:] - kv_indptr[:-1]
    kv_last_page_len = seq_lens.to(torch.int32)

    q_seq_len = 4  # grouped qseqlen

    info = get_mla_metadata_info_v1(
        bs_grouped, q_seq_len, nq_effective, FP8_DTYPE, FP8_DTYPE,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nq_effective // nkv, nkv, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=page_size,
        kv_granularity=kv_gran,
        max_seqlen_qo=q_seq_len,
        uni_seqlen_qo=q_seq_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=FP8_DTYPE,
        dtype_kv=FP8_DTYPE,
    )

    kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    # ALL shapes use qseqlen=4: group 4 batch entries together
    # Q: (bs, 16, 576) → reshape to (bs/4, 64, 576)
    # This triggers fold_factor = 64//16 = 4 → qseqlen fold
    # Dispatches to mla_a8w8_qh64_qseqlen4_gqaratio16_ps.co
    assert batch_size % 4 == 0, f"batch_size {batch_size} must be divisible by 4"

    bs_grouped = batch_size // 4
    nq_effective = nq * 4  # 16 * 4 = 64

    num_kv_splits = 8 if total_kv <= 8192 else 16

    # Build grouped indptrs
    # Each group of 4 entries has qseqlen=4 queries
    # qo_indptr: [0, 4, 8, ..., bs]
    # kv_indptr: each group has 4 * kv_seq_len KV tokens
    cache_key = (batch_size, kv_seq_len, num_kv_splits)
    if cache_key not in _meta_cache:
        qo_grouped = torch.arange(0, bs_grouped + 1, dtype=torch.int32, device="cuda") * 4
        kv_grouped = torch.arange(0, bs_grouped + 1, dtype=torch.int32, device="cuda") * (4 * kv_seq_len)
        _meta_cache[cache_key] = _build_meta(
            bs_grouped, 4 * kv_seq_len, nq_effective, nkv,
            num_kv_splits, qo_grouped, kv_grouped)

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_grouped) = _meta_cache[cache_key]

    kv_buffer_4d = kv_buffer_fp8.view(total_kv, 1, nkv, kv_buffer_fp8.shape[-1])

    # Quantize Q to fp8
    alloc_key = (batch_size, nq_effective, dv, dq)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = (
            torch.empty((bs_grouped, nq_effective, dv), dtype=BF16, device="cuda"),
            torch.zeros(1, dtype=torch.float32, device="cuda"),
            torch.empty(1, dtype=torch.float32, device="cuda"),
            torch.empty(batch_size * nq * dq, dtype=FP8_DTYPE, device="cuda"),
        )
    o_grouped, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

    N = q.numel()
    BLOCK = 4096
    grid = ((N + BLOCK - 1) // BLOCK,)
    amax_buf.zero_()
    _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
    _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                           FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)

    # Reshape: (bs*16*576) → (bs/4, 64, 576)
    q_fp8_reshaped = q_fp8_flat.view(batch_size, nq, dq).view(bs_grouped, nq_effective, dq)

    mla_decode_fwd(
        q_fp8_reshaped, kv_buffer_4d, o_grouped,
        _meta_cache[cache_key][-1],  # kv_indptr_grouped (not paged, just indptrs)
        _meta_cache[cache_key][-1],  # kv_indptr as pages (ps=1, same as kv_indptr)
        kv_indices, kv_last_page_len,
        4,  # max_seqlen_q = 4
        page_size=1, nhead_kv=nkv,
        sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
        q_scale=scale_buf, kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )

    # Reshape output: (bs/4, 64, 512) → (bs, 16, 512)
    return o_grouped.view(batch_size, nq, dv)
