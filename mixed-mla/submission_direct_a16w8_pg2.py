#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — Direct stage1+reduce, bf16 Q + fp8 KV (a16w8), page_size=2.
Combines two optimizations:
1. Skip Q quantization (bf16 Q + fp8 KV = a16w8 kernel)
2. page_size=2 for reduced index/metadata overhead
"""
import torch
import aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
PAGE_SIZE = 2

_meta_cache = {}
_alloc_cache = {}


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
    if total_kv <= 8192:
        num_kv_splits = 8
    else:
        num_kv_splits = 16

    cache_key = (batch_size, kv_seq_len, num_kv_splits)
    if cache_key not in _meta_cache:
        num_pages = total_kv // PAGE_SIZE
        kv_indptr_pages = kv_indptr // PAGE_SIZE
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = (seq_lens % PAGE_SIZE).to(torch.int32)
        kv_last_page_len = torch.where(kv_last_page_len == 0, PAGE_SIZE, kv_last_page_len)

        info = get_mla_metadata_info_v1(
            batch_size, q_seq_len, nq, BF16, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (work_metadata, work_indptr, work_info_set,
         reduce_indptr, reduce_final_map, reduce_partial_map) = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr_pages, kv_last_page_len,
            nq // nkv, nkv, True,
            work_metadata, work_info_set, work_indptr,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            page_size=PAGE_SIZE,
            kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=q_seq_len,
            uni_seqlen_qo=q_seq_len,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=BF16,
            dtype_kv=FP8_DTYPE,
        )

        kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")

        # Pre-allocate stage1 intermediates
        n_partials = reduce_partial_map.size(0)
        logits = torch.empty(
            (n_partials * q_seq_len, 1, nq, dv),
            dtype=torch.float32, device="cuda",
        )
        attn_lse = torch.empty(
            (n_partials * q_seq_len, 1, nq, 1),
            dtype=torch.float32, device="cuda",
        )

        _meta_cache[cache_key] = (
            work_metadata, work_indptr, work_info_set,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            kv_indices, kv_last_page_len, kv_indptr_pages,
            logits, attn_lse,
        )

    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map,
     kv_indices, kv_last_page_len, kv_indptr_pages,
     logits, attn_lse) = _meta_cache[cache_key]

    alloc_key = (q.shape[0], nq, dv)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = torch.empty(
            (q.shape[0], nq, dv), dtype=BF16, device="cuda"
        )
    o = _alloc_cache[alloc_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    # Direct stage1 — bf16 Q, fp8 KV, page_size=2
    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv_buffer_4d,
        qo_indptr,
        kv_indptr_pages,
        kv_indices,
        kv_last_page_len,
        None,
        work_metadata,
        work_indptr,
        work_info_set,
        q_seq_len,
        PAGE_SIZE,
        nkv,
        sm_scale,
        logits,
        attn_lse,
        o,
        None,  # q_scale not needed for bf16
        kv_scale,
    )

    # Direct reduce
    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        q_seq_len,
        o,
        None,
    )

    return o
