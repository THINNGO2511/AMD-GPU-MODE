#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — MXFP4 KV cache approach (1.85x bandwidth savings).
Uses mxfp4 KV data instead of fp8. Dequantizes FP4→bf16 before attention.
This is likely how Borui Xu achieves 12.7μs.

Phase 1: Simple dequant to bf16, then use a16w16 kernel.
Phase 2 (if this works): Fuse dequant into custom Triton attention.
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
_meta_cache = {}
_alloc_cache = {}
_kv_cache = {}


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


def _dequant_mxfp4_kv(fp4_data, scale_e8m0, total_kv, dim):
    """Dequantize MXFP4 KV to bf16."""
    # fp4_data: (total_kv, 1, dim//2) fp4x2
    # scale_e8m0: (total_kv, dim//32) fp8_e8m0
    num_rows = total_kv
    block_size = 32
    num_blocks = dim // block_size

    fp4_2d = fp4_data.reshape(num_rows, dim // 2)
    float_vals = mxfp4_to_f32(fp4_2d)  # (num_rows, dim)

    scale_f32 = e8m0_to_f32(scale_e8m0)  # (padded_rows, padded_blocks)
    scale_f32 = scale_f32[:num_rows, :num_blocks]

    float_vals_blocked = float_vals.view(num_rows, num_blocks, block_size)
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)

    return scaled.view(num_rows, 1, dim).to(BF16)


def _build_meta(batch_size, kv_seq_len, q_seq_len, nq, nkv,
                num_kv_splits, page_size, dtype_q, dtype_kv,
                qo_indptr, kv_indptr):
    total_kv = batch_size * kv_seq_len

    if page_size == 1:
        num_pages = total_kv
        kv_indptr_pages = kv_indptr
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = seq_lens.to(torch.int32)
    else:
        num_pages = total_kv // page_size
        kv_indptr_pages = kv_indptr // page_size
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = (seq_lens % page_size).to(torch.int32)
        kv_last_page_len = torch.where(kv_last_page_len == 0, page_size, kv_last_page_len)

    kv_gran = max(1, 16 // page_size)

    info = get_mla_metadata_info_v1(
        batch_size, q_seq_len, nq, dtype_q, dtype_kv,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr_pages, kv_last_page_len,
        nq // nkv, nkv, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=page_size,
        kv_granularity=kv_gran,
        max_seqlen_qo=q_seq_len,
        uni_seqlen_qo=q_seq_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
    )

    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages, page_size)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    # Strategy: Use MXFP4 KV for large shapes (bandwidth savings)
    # and fp8 KV for small shapes (lower overhead)
    if kv_seq_len >= 8192:
        # MXFP4 path: dequant to bf16, then use a16w16 kernel (bf16 Q + bf16 KV)
        # 1.85x bandwidth savings on KV read
        page_size = 8
        dtype_kv = BF16

        # Dequant MXFP4 KV to bf16
        kv_buffer_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
        kv_bf16 = _dequant_mxfp4_kv(kv_buffer_mxfp4, kv_scale_mxfp4, total_kv, dq)

        num_kv_splits = 16
        dtype_q = BF16

        cache_key = (batch_size, kv_seq_len, num_kv_splits, page_size, "mxfp4")
        if cache_key not in _meta_cache:
            _meta_cache[cache_key] = _build_meta(
                batch_size, kv_seq_len, q_seq_len, nq, nkv,
                num_kv_splits, page_size, dtype_q, dtype_kv,
                qo_indptr, kv_indptr)

        (wm, wi, wis, ri, rfm, rpm,
         kv_indices, kv_last_page_len, kv_indptr_pages, ps) = _meta_cache[cache_key]

        kv_buffer_4d = kv_bf16.view(-1, ps, nkv, kv_bf16.shape[-1])

        alloc_key = ("mxfp4", q.shape[0], nq, dv)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = torch.empty(
                (q.shape[0], nq, dv), dtype=BF16, device="cuda")
        o = _alloc_cache[alloc_key]

        mla_decode_fwd(
            q, kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=ps, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            kv_scale=None,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
        return o

    else:
        # Small kv: use pg2 + bf16 Q (proven approach)
        page_size = 2
        dtype_q = BF16
        num_kv_splits = 8 if total_kv <= 8192 else 16

        cache_key = (batch_size, kv_seq_len, num_kv_splits, page_size, "fp8_bf16q")
        if cache_key not in _meta_cache:
            _meta_cache[cache_key] = _build_meta(
                batch_size, kv_seq_len, q_seq_len, nq, nkv,
                num_kv_splits, page_size, dtype_q, FP8_DTYPE,
                qo_indptr, kv_indptr)

        (wm, wi, wis, ri, rfm, rpm,
         kv_indices, kv_last_page_len, kv_indptr_pages, ps) = _meta_cache[cache_key]

        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        kv_buffer_4d = kv_buffer_fp8.view(-1, ps, nkv, kv_buffer_fp8.shape[-1])

        alloc_key = ("bf16", q.shape[0], nq, dv)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = torch.empty(
                (q.shape[0], nq, dv), dtype=BF16, device="cuda")
        o = _alloc_cache[alloc_key]

        mla_decode_fwd(
            q, kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=ps, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
        return o
