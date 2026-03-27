#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Decode — Hybrid a16w8/a8w8 based on kv_seq_len.
kv<=1024: a16w8 (skip Q quant, saves 5-10%)
kv>1024: a8w8 (fp8 compute wins on large attention)
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
PAGE_SIZE = 1
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


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    total_kv = batch_size * kv_seq_len

    # Hybrid decision: a16w8 for small kv, a8w8 for large kv
    use_a16w8 = (kv_seq_len <= 1024)

    if total_kv <= 4096:
        num_kv_splits = 8
    elif total_kv <= 65536:
        num_kv_splits = 16
    else:
        num_kv_splits = 16

    q_dtype = torch.bfloat16 if use_a16w8 else FP8_DTYPE

    cache_key = (batch_size, kv_seq_len, num_kv_splits, use_a16w8)
    if cache_key not in _meta_cache:
        total_kv_len = batch_size * kv_seq_len
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        info = get_mla_metadata_info_v1(
            batch_size, q_seq_len, nq, q_dtype, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            nq // nkv, nkv, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
            fast_mode=False, max_split_per_batch=num_kv_splits,
            intra_batch_mode=True, dtype_q=q_dtype, dtype_kv=FP8_DTYPE,
        )

        kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
        _meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len)

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len) = _meta_cache[cache_key]

    alloc_key = (q.shape[0], nq, dv, dq, use_a16w8)
    if alloc_key not in _alloc_cache:
        allocs = {'o': torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")}
        if not use_a16w8:
            allocs['amax'] = torch.zeros(1, dtype=torch.float32, device="cuda")
            allocs['scale'] = torch.empty(1, dtype=torch.float32, device="cuda")
            allocs['q_fp8'] = torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda")
        _alloc_cache[alloc_key] = allocs
    allocs = _alloc_cache[alloc_key]
    o = allocs['o']

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    if use_a16w8:
        q_input = q
        q_scale = None
    else:
        amax_buf = allocs['amax']
        scale_buf = allocs['scale']
        q_fp8_flat = allocs['q_fp8']
        N = q.numel()
        BLOCK = 4096
        grid = ((N + BLOCK - 1) // BLOCK,)
        amax_buf.zero_()
        _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
        _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf, FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
        q_input = q_fp8_flat.view(q.shape[0], nq, dq)
        q_scale = scale_buf

    mla_decode_fwd(
        q_input, kv_buffer_4d, o,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        q_seq_len, page_size=PAGE_SIZE, nhead_kv=nkv,
        sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
        q_scale=q_scale, kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )
    return o
