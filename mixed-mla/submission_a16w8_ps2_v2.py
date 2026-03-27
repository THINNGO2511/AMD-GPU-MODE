#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — bf16 Q + fp8 KV + page_size=2, variant 2.
Try kv_granularity=2 (match page_size exactly) and more splits.
Also try different split tuning since pg2 changes the workload.
"""
import torch
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
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
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    total_kv = batch_size * kv_seq_len
    num_pages = total_kv // PAGE_SIZE
    # With pg2, pages are halved -> adjust splits based on pages not tokens
    if num_pages <= 2048:
        num_kv_splits = 8
    elif num_pages <= 16384:
        num_kv_splits = 16
    else:
        num_kv_splits = 32

    cache_key = (batch_size, kv_seq_len, num_kv_splits)
    if cache_key not in _meta_cache:
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
        (wm, wi, wis, ri, rfm, rpm) = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr_pages, kv_last_page_len,
            nq // nkv, nkv, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE,
            kv_granularity=PAGE_SIZE,  # Match page_size exactly
            max_seqlen_qo=q_seq_len,
            uni_seqlen_qo=q_seq_len,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=BF16,
            dtype_kv=FP8_DTYPE,
        )

        kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
        _meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages)

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages) = _meta_cache[cache_key]

    alloc_key = (q.shape[0], nq, dv)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda")
    o = _alloc_cache[alloc_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    mla_decode_fwd(
        q, kv_buffer_4d, o,
        qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
        q_seq_len, page_size=PAGE_SIZE, nhead_kv=nkv,
        sm_scale=sm_scale, logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )
    return o
