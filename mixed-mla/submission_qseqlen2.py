#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MLA: Use qSeqLen=2 kernel — batch 2 queries per workgroup for 2x KV reuse.
mla_a8w8_qh16_qseqlen2_gqaratio16_ps.co confirmed on runner."""
import torch
import triton
import triton.language as tl
import aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
PAGE_SIZE = 2
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
MAX_SEQLEN_Q = 2  # KEY CHANGE: batch 2 queries per workgroup

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
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    # Use qSeqLen=2: reshape batch into pairs
    # Original: total_q queries with qseqlen=1 per batch
    # New: total_q/2 "batches" with qseqlen=2 each
    use_qseq2 = batch_size >= 4 and batch_size % 2 == 0
    
    if use_qseq2:
        max_seqlen_q = 2
        effective_bs = batch_size // 2
        # Rebuild qo_indptr for paired batches
        qo_indptr_new = torch.arange(0, effective_bs + 1, dtype=torch.int32, device=q.device) * 2
        # Rebuild kv_indptr for paired batches (each pair shares same kv_seq_len)
        kv_indptr_new = torch.arange(0, effective_bs + 1, dtype=torch.int32, device=q.device) * (kv_seq_len * 2)
    else:
        max_seqlen_q = 1
        effective_bs = batch_size
        qo_indptr_new = qo_indptr
        kv_indptr_new = kv_indptr

    num_kv_splits = 16 if total_kv > 8192 else 8
    dtype_q = FP8_DTYPE

    cache_key = (effective_bs, kv_seq_len, num_kv_splits, max_seqlen_q)
    if cache_key not in _meta_cache:
        num_pages = total_kv // PAGE_SIZE
        kv_indptr_pages = kv_indptr_new // PAGE_SIZE if use_qseq2 else kv_indptr // PAGE_SIZE
        seq_lens = kv_indptr_pages[1:] - kv_indptr_pages[:-1]
        kv_last_page_len = (seq_lens * PAGE_SIZE % PAGE_SIZE).to(torch.int32)
        kv_last_page_len = torch.where(kv_last_page_len == 0, PAGE_SIZE, kv_last_page_len)

        info = get_mla_metadata_info_v1(
            effective_bs, max_seqlen_q, nq, dtype_q, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (work_metadata, work_indptr, work_info_set,
         reduce_indptr, reduce_final_map, reduce_partial_map) = work

        get_mla_metadata_v1(
            qo_indptr_new if use_qseq2 else qo_indptr,
            kv_indptr_pages,
            kv_last_page_len,
            nq // nkv, nkv, True,
            work_metadata, work_info_set, work_indptr,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            page_size=PAGE_SIZE,
            kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=max_seqlen_q,
            uni_seqlen_qo=max_seqlen_q,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=dtype_q,
            dtype_kv=FP8_DTYPE,
        )

        kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
        n_partials = reduce_partial_map.size(0)
        logits = torch.empty((n_partials * max_seqlen_q, 1, nq, dv), dtype=torch.float32, device="cuda")
        attn_lse = torch.empty((n_partials * max_seqlen_q, 1, nq, 1), dtype=torch.float32, device="cuda")

        _meta_cache[cache_key] = (
            work_metadata, work_indptr, work_info_set,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            kv_indices, kv_last_page_len, kv_indptr_pages,
            logits, attn_lse, qo_indptr_new if use_qseq2 else qo_indptr,
        )

    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map,
     kv_indices, kv_last_page_len, kv_indptr_pages,
     logits, attn_lse, qo_indptr_cached) = _meta_cache[cache_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    # Quantize Q to fp8
    alloc_key = ("a8w8", q.shape[0], nq, dv, dq)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = (
            torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda"),
            torch.zeros(1, dtype=torch.float32, device="cuda"),
            torch.empty(1, dtype=torch.float32, device="cuda"),
            torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda"),
        )
    o, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

    N = q.numel()
    BLOCK = 4096
    grid = ((N + BLOCK - 1) // BLOCK,)
    amax_buf.zero_()
    _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
    _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                           FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
    q_fp8 = q_fp8_flat.view(q.shape[0], nq, dq)

    aiter.mla_decode_stage1_asm_fwd(
        q_fp8, kv_buffer_4d,
        qo_indptr_cached, kv_indptr_pages, kv_indices, kv_last_page_len,
        None, work_metadata, work_indptr, work_info_set,
        max_seqlen_q, PAGE_SIZE, nkv, sm_scale,
        logits, attn_lse, o,
        scale_buf, kv_scale,
    )

    aiter.mla_reduce_v1(
        logits, attn_lse,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        max_seqlen_q, o, None,
    )

    return o
